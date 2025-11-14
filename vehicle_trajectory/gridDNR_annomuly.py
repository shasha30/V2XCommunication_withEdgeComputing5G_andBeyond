#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_pipeline_with_anom.py
Merged pipeline for:
 - Algorithm 1: Grid encoding
 - Algorithm 2: Dynamic social network GT
 - Algorithm 3: Dynamic node selection (R-history)
 + Representation Learning (Node2Vec-style walks + Word2Vec)
 + Anomaly Detection (IsolationForest or z-score)
Outputs CSVs and visualizations.

Usage examples:
  python3 grid_pipeline_with_anom.py --demo --visualize
  python3 grid_pipeline_with_anom.py --input sample_1000.csv --output-prefix porto_result --grid-size 1000 --visualize
"""
import os, math, argparse, random, csv, ast, time, warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as _np
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "bool"):
    _np.bool = bool
try:
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    from gensim.models import Word2Vec
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    _HAVE_ALL = True
except Exception:
    # we will try to degrade gracefully
    _HAVE_ALL = False

# ---------- utility exporters ----------
def export_GT_edges_to_csv(GT, out_path):
    with open(out_path, "w", newline="") as fh:
        fields = ["time", "u", "v", "grid", "cooccurrence", "weight"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in sorted(GT.keys()):
            Gt = GT[t]
            for u,v,ed in Gt.edges(data=True):
                w.writerow({
                    "time": t, "u": u, "v": v,
                    "grid": ed.get("grid",""), "cooccurrence": ed.get("cooccurrence",""),
                    "weight": ed.get("weight","")
                })

def export_GT_nodes_to_csv(GT, out_path):
    with open(out_path, "w", newline="") as fh:
        fields = ["time","node","grid_id","grid_count"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in sorted(GT.keys()):
            Gt = GT[t]
            for n,dat in Gt.nodes(data=True):
                w.writerow({"time":t,"node":n,"grid_id":dat.get("grid_id",""),"grid_count":dat.get("grid_count","")})

def export_selected_nodes(V_sel, out_csv, out_summary):
    with open(out_csv,"w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time","selected_node"])
        w.writeheader()
        for t in sorted(V_sel.keys()):
            for n in V_sel[t]:
                w.writerow({"time":t,"selected_node":n})
    with open(out_summary,"w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time","selected_nodes"])
        w.writeheader()
        for t in sorted(V_sel.keys()):
            w.writerow({"time":t,"selected_nodes":" ".join(map(str,V_sel[t]))})

def export_R_history(R_hist, out_csv):
    # R_hist: dict time -> {node:score}
    nodes = sorted({n for d in R_hist.values() for n in d.keys()})
    times = sorted(R_hist.keys())
    with open(out_csv,"w",newline="") as fh:
        fields = ["time"] + nodes
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in times:
            row = {"time":t}
            for n in nodes:
                row[n] = R_hist[t].get(n,0.0)
            w.writerow(row)

# ---------- Dataset loader (Porto support) ----------
def load_or_generate_trajs(input_csv=None, demo_n=12, points_per_traj=80, seed=42):
    if input_csv:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(input_csv)
        df = pd.read_csv(input_csv)
        cols_up = set(c.upper() for c in df.columns)
        if "POLYLINE" in cols_up:
            # Porto format
            print("[INFO] Detected POLYLINE format - expanding")
            rows=[]
            for _,r in df.iterrows():
                try:
                    coords = ast.literal_eval(r["POLYLINE"])
                    tid = r.get("TRIP_ID", str(_))
                    for t,(lon,lat) in enumerate(coords):
                        rows.append({"traj_id":tid,"time":t,"lat":lat,"lon":lon})
                except Exception:
                    continue
            return pd.DataFrame(rows)
        # else expect columns traj_id,time,lat,lon
        needed = {'traj_id','time','lat','lon'}
        if not needed.issubset(set(df.columns)):
            # try common alternative names (TRIP_ID, POLYLINE already handled)
            raise ValueError(f"Input CSV must contain columns: {needed}")
        df['time'] = df['time'].astype(int)
        return df
    # synthetic demo
    random.seed(seed)
    rows=[]
    for traj in range(1, demo_n+1):
        start_lat = random.uniform(12.90,13.10)
        start_lon = random.uniform(77.50,77.90)
        angle = random.uniform(0,2*math.pi)
        for t in range(points_per_traj):
            step = 0.0012 + 0.0006*math.sin(t/8.0 + traj)
            lat = start_lat + step*math.cos(angle) + random.uniform(-0.0003,0.0003)
            lon = start_lon + step*math.sin(angle) + random.uniform(-0.0003,0.0003)
            rows.append({'traj_id':f"v{traj}",'time':t,'lat':lat,'lon':lon})
            angle += 0.02*math.sin(t/15.0 + traj/3.0)
    return pd.DataFrame(rows)

# ---------- Algorithm 1: grid encoding ----------
def compute_grid_indices(df, grid_size=100):
    if df.empty:
        return df, {}
    minLng=float(df['lon'].min()); maxLng=float(df['lon'].max())
    minLat=float(df['lat'].min()); maxLat=float(df['lat'].max())
    eps=1e-12
    lng_span = max(maxLng-minLng, eps)
    lat_span = max(maxLat-minLat, eps)
    lngS = lng_span / grid_size
    latS = lat_span / grid_size
    gx=[]; gy=[]; gid=[]
    for _,r in df.iterrows():
        lng=float(r['lon']); lat=float(r['lat'])
        x = math.ceil((lng-minLng)/lngS)
        y = math.ceil((lat-minLat)/latS)
        x = min(max(1,x), grid_size); y = min(max(1,y), grid_size)
        gx.append(int(x)); gy.append(int(y)); gid.append(int(10**5 * x + y))
    out = df.copy()
    out['grid_x']=gx; out['grid_y']=gy; out['grid_id']=gid
    meta = {'minLng':minLng,'maxLng':maxLng,'minLat':minLat,'maxLat':maxLat,'lngS':lngS,'latS':latS,'grid_size':grid_size}
    return out, meta

# ---------- Plotting helper ----------
def visualize_trajectories_grid(df, meta, out_png="traj_grid.png"):
    fig,ax = plt.subplots(figsize=(10,8))
    ax.set_title("Trajectories on Grid")
    xs = [meta['minLng'] + i*meta['lngS'] for i in range(meta['grid_size']+1)]
    ys = [meta['minLat'] + i*meta['latS'] for i in range(meta['grid_size']+1)]
    for x in xs: ax.plot([x,x],[meta['minLat'],meta['maxLat']], color='lightgray', linewidth=0.3)
    for y in ys: ax.plot([meta['minLng'],meta['maxLng']],[y,y], color='lightgray', linewidth=0.3)
    cmap = plt.get_cmap('tab20')
    for i,tid in enumerate(sorted(df['traj_id'].unique())):
        g = df[df['traj_id']==tid].sort_values('time')
        ax.plot(g['lon'].values, g['lat'].values, marker='o', markersize=2, linewidth=1, color=cmap(i%20))
    plt.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)

# ---------- Algorithm 2: pad + build GT ----------
def pad_and_build_GT(df_grids):
    if df_grids.empty:
        return {}
    max_t = int(df_grids['time'].max())
    padded=[]
    for tid,grp in df_grids.groupby('traj_id'):
        g_sorted = grp.sort_values('time')
        times = list(g_sorted['time'].astype(int).values)
        grids = list(g_sorted['grid_id'].astype(int).values)
        last_t = times[-1]; last_grid = grids[-1]
        for t0,g0 in zip(times,grids):
            padded.append({'traj_id':tid,'time':int(t0),'grid_id':int(g0)})
        for t_pad in range(last_t+1, max_t+1):
            padded.append({'traj_id':tid,'time':int(t_pad),'grid_id':int(last_grid)})
    dfp = pd.DataFrame(padded)
    GT = {}
    vehicle_list = sorted(dfp['traj_id'].unique())
    for t in range(0, max_t+1):
        G = nx.Graph()
        G.add_nodes_from(vehicle_list)
        slice_t = dfp[dfp['time']==t]
        # attach node attr
        for n in vehicle_list:
            sub = slice_t[slice_t['traj_id']==n]
            if len(sub):
                G.nodes[n]['grid_id'] = int(sub['grid_id'].iloc[0])
            else:
                G.nodes[n]['grid_id'] = None
            # grid_count set later
            G.nodes[n]['grid_count'] = 0
        # group by grid to add edges
        for grid_id,grp in slice_t.groupby('grid_id'):
            vids = list(grp['traj_id'].unique())
            L = len(vids)
            for i in range(L):
                for j in range(i+1,L):
                    if not G.has_edge(vids[i],vids[j]):
                        G.add_edge(vids[i],vids[j], grid=grid_id, cooccurrence=L, weight=(1.0/(L-1) if L>1 else 0.0))
            # set grid_count for nodes in this cell
            for v in vids:
                G.nodes[v]['grid_count'] = L
        GT[t] = G
    return GT

# ---------- Algorithm 3: dynamic selection ----------
def compute_delta_E(prev_G, cur_G):
    nodes = set(prev_G.nodes()).union(set(cur_G.nodes()))
    delta={}
    for n in nodes:
        nbr_prev = set(prev_G.neighbors(n)) if prev_G.has_node(n) else set()
        nbr_cur  = set(cur_G.neighbors(n)) if cur_G.has_node(n) else set()
        delta[n] = len(nbr_prev.symmetric_difference(nbr_cur))
    return delta

def partition_graph(G,P,seed=42):
    nodes = list(G.nodes())
    n=len(nodes)
    if n==0:
        return [[] for _ in range(P)]
    P_eff = min(P,n)
    try:
        import nxmetis
        _, parts = nxmetis.partition(G, P_eff, recursive=False)
        while len(parts) < P:
            parts.append([])
        return parts[:P]
    except Exception:
        # fallback chunking by degree
        nodes_sorted = sorted(nodes, key=lambda x: G.degree(x), reverse=True)
        parts = [[] for _ in range(P_eff)]
        for i,n_ in enumerate(nodes_sorted):
            parts[i%P_eff].append(n_)
        for _ in range(P-P_eff):
            parts.append([])
        return parts

def softmax_weights(vals):
    if not vals:
        return {}
    maxv = max(vals.values())
    exps = {k: math.exp(v-maxv) for k,v in vals.items()}
    s = sum(exps.values())
    if s==0:
        return {k:1.0/len(vals) for k in vals}
    return {k:exps[k]/s for k in vals}

def select_from_partition(part_nodes, R_scores, rng):
    if not part_nodes:
        return None
    vals = {n:R_scores.get(n,0.0) for n in part_nodes}
    probs = softmax_weights(vals)
    r = rng.random()
    cum=0.0
    for n in part_nodes:
        cum += probs.get(n,0.0)
        if r <= cum:
            return n
    return part_nodes[-1]

def algorithm3_select(GT, alpha=0.1, seed=42, use_all_first=True,
                      decay=0.98, beta=0.0, return_R_history=False):
    """
    Algorithm 3 – Dynamic Node Selection (cleaned: no visualization)
    GT: dict {t → networkx.Graph}
    alpha: selection ratio
    decay: memory decay for R-scores
    beta: optional boost for crowded cells
    return_R_history: if True, returns (V_sel, R_history)
    """
    rng = random.Random(seed)
    times = sorted(GT.keys())
    if not times:
        return ({}, {}) if return_R_history else {}

    V_sel = {}
    R_prev = defaultdict(float)
    prev_G = GT[times[0]]
    R_history = {}

    for idx, t in enumerate(times):
        G_t = GT[t]
        nodes = list(G_t.nodes())

        # first step: select all nodes
        if t == times[0] and use_all_first:
            V_sel[t] = nodes.copy()
            for v in V_sel[t]:
                R_prev[v] = 0.0
            if return_R_history:
                R_history[t] = dict(R_prev)
            prev_G = G_t
            continue

        # compute |ΔE_t|
        delta = compute_delta_E(prev_G, G_t)

        # update R scores with decay
        R_curr = defaultdict(float)
        for n in set(list(R_prev.keys()) + list(delta.keys()) + list(G_t.nodes())):
            R_curr[n] = R_prev.get(n, 0.0) * decay + float(delta.get(n, 0.0))

        # optional β term based on local density
        if beta > 0.0:
            for n in list(G_t.nodes()):
                gcount = int(G_t.nodes[n].get("grid_count", 0) or 0)
                if gcount > 0:
                    R_curr[n] += beta * float(gcount)

        if return_R_history:
            R_history[t] = dict(R_curr)

        # number of partitions P and representative selection
        P = max(1, math.ceil(alpha * max(1, len(nodes))))
        parts = partition_graph(G_t, P, seed=seed)
        sel = []
        for p in parts:
            rep = select_from_partition(p, R_curr, rng)
            if rep is not None:
                sel.append(rep)
                R_curr[rep] = 0.0

        V_sel[t] = sel
        prev_G = G_t
        R_prev = R_curr

    if return_R_history:
        return V_sel, R_history
    return V_sel



# ---------- Representation learning (random walks + Word2Vec) ----------
def generate_walks_from_GT(GT, num_walks=10, walk_length=10, seed=42):
    rng = random.Random(seed)
    walks=[]
    times = sorted(GT.keys())
    for t in times:
        Gt = GT[t]
        nodes = list(Gt.nodes())
        for _ in range(num_walks):
            rng.shuffle(nodes)
            for n in nodes:
                walk=[str(n)]
                cur = n
                for step in range(walk_length-1):
                    nbrs = list(Gt.neighbors(cur))
                    if not nbrs:
                        break
                    cur = rng.choice(nbrs)
                    walk.append(str(cur))
                walks.append(walk)
    return walks

def learn_embeddings_from_walks(walks, dim=64, window=5, epochs=3, seed=42):
    # gensim Word2Vec (skip-gram)
    model = Word2Vec(walks, vector_size=dim, window=window, min_count=1, sg=1, workers=1, seed=seed, epochs=epochs)
    embeddings = {str(n): model.wv[str(n)] for walk in walks for n in walk}  # unique keys later
    # reduce to unique mapping
    uniq = {}
    for k,v in embeddings.items():
        uniq[k]=v
    return uniq

# ---------- anomaly detection ----------
def detect_anomalies(embeddings_dict, method="iforest", contamination=0.05, random_state=42):
    ids = sorted(embeddings_dict.keys())
    X = [embeddings_dict[i] for i in ids]
    result = []
    if method=="iforest":
        try:
            clf = IsolationForest(contamination=contamination, random_state=random_state)
            clf.fit(X)
            scores = clf.decision_function(X)  # higher -> more normal
            preds = clf.predict(X)  # -1 outlier, 1 normal
            for i,s,p in zip(ids,scores,preds):
                result.append({"node":i,"anom_score":float(-s),"is_anom": int(p==-1)})
            return result
        except Exception:
            method = "zscore"
    # fallback: z-score on L2 norm distance from mean
    import numpy as np
    arr = np.array(X)
    mean = arr.mean(axis=0)
    d = np.linalg.norm(arr - mean, axis=1)
    mu = d.mean(); sigma = d.std() if d.std()>1e-12 else 1.0
    for i,di in zip(ids,d):
        z = (di - mu)/sigma
        is_anom = 1 if z > 2.5 else 0
        result.append({"node":i,"anom_score":float(z),"is_anom":is_anom})
    return result

# ---------- CLI / main ----------
def main():
    parser = argparse.ArgumentParser(description="Pipeline w/ embedding & anomaly detection")
    parser.add_argument("--input", help="input CSV (Porto or traj_id,time,lat,lon). If omitted demo used.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-prefix", default="result")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num-walks", type=int, default=8)
    parser.add_argument("--walk-length", type=int, default=12)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--emb-window", type=int, default=5)
    parser.add_argument("--if-contam", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.demo or not args.input:
        df = load_or_generate_trajs(None, demo_n=20, points_per_traj=80, seed=args.seed)
    else:
        df = load_or_generate_trajs(args.input)
    print(f"[INFO] loaded {len(df)} rows from {df['traj_id'].nunique()} trajectories")

    # algorithm1
    df_grids, meta = compute_grid_indices(df, grid_size=args.grid_size)
    out_grids = f"{args.output_prefix}_grids.csv"
    df_grids.to_csv(out_grids, index=False)
    print("[INFO] grid-encoded saved:", out_grids)
    if args.visualize:
        visualize_trajectories_grid(df_grids, meta, out_png=f"{args.output_prefix}_traj_grid.png")
        print("[INFO] trajectory plot saved")

    # algorithm2
    df_for_gt = df_grids[['traj_id','time','grid_id']].copy()
    GT = pad_and_build_GT(df_for_gt)
    print("[INFO] GT built timesteps:", len(GT))
    export_GT_edges_to_csv(GT, f"{args.output_prefix}_GT_edges.csv")
    export_GT_nodes_to_csv(GT, f"{args.output_prefix}_GT_nodes.csv")
    print("[INFO] Exported GT CSVs")

    # algorithm3
    V_sel, R_hist = algorithm3_select(GT, alpha=args.alpha, seed=args.seed, use_all_first=True, visualize=args.visualize, decay=0.98, beta=0.0, return_R_history=True)
    export_selected_nodes(V_sel, f"{args.output_prefix}_selected_nodes.csv", f"{args.output_prefix}_selected_summary.csv")
    export_R_history(R_hist, f"{args.output_prefix}_R_history.csv")
    print("[INFO] exported selection and R history CSVs")

    # representation learning
    print("[INFO] Generating random walks for representation learning...")
    walks = generate_walks_from_GT(GT, num_walks=args.num_walks, walk_length=args.walk_length, seed=args.seed)
    print("[INFO] walks count:", len(walks))
    print("[INFO] training Word2Vec embeddings...")
    emb = learn_embeddings_from_walks(walks, dim=args.emb_dim, window=args.emb_window, epochs=4, seed=args.seed)
    # save embeddings
    emb_rows=[]
    for nid,vec in emb.items():
        emb_rows.append({"node":nid, **{f"v{i}":float(vec[i]) for i in range(len(vec))}})
    df_emb = pd.DataFrame(emb_rows)
    df_emb.to_csv(f"{args.output_prefix}_embeddings.csv", index=False)
    print("[INFO] embeddings saved")

    # anomaly detection
    print("[INFO] running anomaly detection (IsolationForest or zscore fallback)...")
    anom_res = detect_anomalies({k:v for k,v in emb.items()}, method="iforest", contamination=args.if_contam, random_state=args.seed)
    df_anom = pd.DataFrame(anom_res)
    df_anom.to_csv(f"{args.output_prefix}_anomalies.csv", index=False)
    print("[INFO] anomalies saved:", f"{args.output_prefix}_anomalies.csv")

    # 2D visualization of embeddings (PCA then TSNE fallback)
    try:
        X = df_emb[[c for c in df_emb.columns if c.startswith("v")]].values
        if X.shape[1] > 20:
            red = PCA(n_components=20, random_state=args.seed).fit_transform(X)
        else:
            red = X
        emb2 = TSNE(n_components=2, random_state=args.seed, perplexity=30).fit_transform(red)
        plt.figure(figsize=(8,6))
        labs = df_emb['node'].astype(str).values
        plt.scatter(emb2[:,0], emb2[:,1], s=20)
        for i,lab in enumerate(labs):
            plt.text(emb2[i,0], emb2[i,1], lab, fontsize=6)
        plt.title("Embeddings 2D (TSNE)")
        plt.tight_layout()
        plt.savefig(f"{args.output_prefix}_embeddings_tsne.png", dpi=180)
        plt.close()
        print("[INFO] embeddings TSNE saved")
    except Exception as e:
        print("[WARN] embeddings visualization failed:", e)

    print("[DONE] pipeline finished. Outputs prefix:", args.output_prefix)

if __name__ == "__main__":
    main()
