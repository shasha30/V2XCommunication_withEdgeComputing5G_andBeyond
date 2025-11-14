#!/usr/bin/env python3
"""
anomaly_from_results.py

Reads CSV outputs from the grid/GT/selection pipeline (prefix_* files),
reconstructs GT (dynamic graphs), generates random walks, learns embeddings,
runs anomaly detection, and writes embeddings + anomalies CSVs (plus optional plots).

Example:
 python3 anomaly_from_results.py --prefix porto_result --outdir porto_anom --plot
"""

import os, sys, argparse, math, random, csv, json, warnings
from collections import defaultdict
try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required. Install with: pip install pandas")
    raise

# optional libs
_HAVE_NX = True
try:
    import networkx as nx
except Exception:
    _HAVE_NX = False
    warnings.warn("networkx not available; GT graph reconstruction may fail.")

_HAVE_GENSIM = True
try:
    from gensim.models import Word2Vec
except Exception:
    _HAVE_GENSIM = False
    warnings.warn("gensim not available; embeddings cannot be trained.")

_HAVE_SKLEARN = True
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:
    _HAVE_SKLEARN = False
    warnings.warn("scikit-learn or parts missing; will fallback to z-score and PCA-only.")

_HAVE_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    _HAVE_MPL = False
    warnings.warn("matplotlib missing; plotting disabled.")

# ---------------- helpers ----------------
def safe_read_csv(path):
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return None

def build_GT_from_csvs(edges_df, nodes_df):
    """
    edges_df expected columns: time, u, v, grid, cooccurrence, weight (time numeric)
    nodes_df expected columns: time, node, grid_id, grid_count
    Returns dict: time -> networkx.Graph with node attributes set.
    """
    if edges_df is None and nodes_df is None:
        print("[ERROR] No GT CSVs provided.")
        return {}

    GT = {}
    times = set()
    if edges_df is not None:
        times.update(edges_df['time'].astype(int).unique().tolist())
    if nodes_df is not None:
        times.update(nodes_df['time'].astype(int).unique().tolist())
    times = sorted(times)
    for t in times:
        G = nx.Graph()
        GT[t] = G

    # add nodes and their attrs
    if nodes_df is not None:
        for _,r in nodes_df.iterrows():
            try:
                t = int(r['time'])
                n = str(r['node'])
                if t not in GT: GT[t] = nx.Graph()
                GT[t].add_node(n)
                # attach attrs
                GT[t].nodes[n]['grid_id'] = r.get('grid_id', None)
                GT[t].nodes[n]['grid_count'] = r.get('grid_count', None)
            except Exception:
                continue

    # edges
    if edges_df is not None:
        for _,r in edges_df.iterrows():
            try:
                t = int(r['time'])
                u = str(r['u']); v = str(r['v'])
                if t not in GT: GT[t] = nx.Graph()
                if not GT[t].has_node(u): GT[t].add_node(u)
                if not GT[t].has_node(v): GT[t].add_node(v)
                attrs = {}
                for c in ['grid','cooccurrence','weight']:
                    if c in r.index:
                        attrs[c] = r[c]
                GT[t].add_edge(u,v, **attrs)
            except Exception:
                continue
    return GT

def generate_walks(GT, num_walks=10, walk_length=10, seed=42):
    """Generate node sequences (walks) from each time-slice of GT."""
    rng = random.Random(seed)
    walks = []
    times = sorted(GT.keys())
    for t in times:
        G = GT[t]
        nodes = list(G.nodes())
        if not nodes:
            continue
        for _ in range(num_walks):
            rng.shuffle(nodes)
            for n in nodes:
                walk = [str(n)]
                cur = n
                for _ in range(walk_length - 1):
                    nbrs = list(G.neighbors(cur))
                    if not nbrs:
                        break
                    cur = rng.choice(nbrs)
                    walk.append(str(cur))
                walks.append(walk)
    return walks

def train_embeddings(walks, dim=64, window=5, epochs=5, seed=42):
    if not _HAVE_GENSIM:
        raise RuntimeError("gensim not installed")
    model = Word2Vec(walks, vector_size=dim, window=window, min_count=1, sg=1, workers=1, seed=seed, epochs=epochs)
    embeddings = {}
    for node in model.wv.index_to_key:
        embeddings[node] = model.wv[node]
    return embeddings

def detect_anomalies_on_embeddings(embeddings, method="iforest", contamination=0.05, random_state=42):
    """
    embeddings: dict node->vector
    returns list of dicts: node, score, is_anom (1 if anomaly)
    """
    import numpy as np
    nodes = sorted(embeddings.keys())
    X = np.vstack([embeddings[n] for n in nodes])
    if method == "iforest" and _HAVE_SKLEARN:
        clf = IsolationForest(contamination=contamination, random_state=random_state)
        clf.fit(X)
        scores = clf.decision_function(X)   # higher is more normal
        preds = clf.predict(X)             # -1 outlier, 1 normal
        res = []
        for n,s,p in zip(nodes, scores, preds):
            res.append({"node": n, "anom_score": float(-s), "is_anom": int(p==-1)})
        return res
    # fallback: z-score on distance from mean
    mean = X.mean(axis=0)
    d = np.linalg.norm(X - mean, axis=1)
    mu = d.mean(); sigma = d.std() if d.std()>1e-12 else 1.0
    res=[]
    for n,di in zip(nodes,d):
        z = (di - mu)/sigma
        res.append({"node":n, "anom_score": float(z), "is_anom": int(z>2.5)})
    return res

def save_embeddings_csv(embeddings, out_path):
    import numpy as np
    rows=[]
    for n,v in embeddings.items():
        row={"node": n}
        for i,val in enumerate(v):
            row[f"v{i}"] = float(val)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)

def plot_embeddings_2d(embeddings, anomalies, out_png):
    if not _HAVE_MPL:
        print("[WARN] matplotlib not available; skipping embeddings plot")
        return
    try:
        import numpy as np
        X = np.vstack([embeddings[n] for n in sorted(embeddings.keys())])
        labels = sorted(embeddings.keys())
        # reduce via PCA then TSNE if available
        if _HAVE_SKLEARN:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            Xr = PCA(n_components=min(20, X.shape[1])).fit_transform(X)
            X2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Xr)
        else:
            from sklearn.decomposition import PCA as PCA_fallback  # may fail if sklearn missing
            X2 = PCA_fallback(n_components=2).fit_transform(X)
        # anomaly lookup
        anom_set = {r['node'] for r in anomalies if int(r.get("is_anom",0))==1}
        plt.figure(figsize=(8,6))
        xs = X2[:,0]; ys = X2[:,1]
        plt.scatter(xs, ys, s=18, alpha=0.8)
        for i,n in enumerate(labels):
            col = "red" if n in anom_set else "black"
            plt.text(xs[i], ys[i], n, fontsize=6, color=col)
        plt.title("Node embeddings (2D) — red=anomaly")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception as e:
        warnings.warn(f"Embedding plot failed: {e}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Anomaly detection + embeddings from pipeline CSV results")
    ap.add_argument("--prefix", required=True, help="Prefix used when creating pipeline outputs (e.g. porto_result)")
    ap.add_argument("--outdir", default="anomaly_out", help="Output directory")
    ap.add_argument("--num-walks", type=int, default=8)
    ap.add_argument("--walk-length", type=int, default=12)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--emb-window", type=int, default=5)
    ap.add_argument("--emb-epochs", type=int, default=4)
    ap.add_argument("--if-contam", type=float, default=0.05)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prefix = args.prefix
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # expected filenames (tolerant)
    grids_csv = f"{prefix}_grids.csv"
    gt_edges_csv = f"{prefix}_GT_edges.csv"
    gt_nodes_csv = f"{prefix}_GT_nodes.csv"
    r_history_csv = f"{prefix}_R_history.csv"
    sel_nodes_csv = f"{prefix}_selected_nodes.csv"
    sel_summary_csv = f"{prefix}_selected_summary.csv"

    print("[INFO] Reading CSVs (if present):")
    print("  ", grids_csv)
    print("  ", gt_edges_csv)
    print("  ", gt_nodes_csv)
    print("  ", r_history_csv)
    print("  ", sel_nodes_csv)
    print("  ", sel_summary_csv)

    edges_df = safe_read_csv(gt_edges_csv)
    nodes_df = safe_read_csv(gt_nodes_csv)
    r_hist_df = safe_read_csv(r_history_csv)
    sel_nodes_df = safe_read_csv(sel_nodes_csv)
    sel_summary_df = safe_read_csv(sel_summary_csv)

    # Build GT
    if not _HAVE_NX:
        print("[ERROR] networkx is required to reconstruct GT. Install networkx and retry.")
        sys.exit(1)

    GT = build_GT_from_csvs(edges_df, nodes_df)
    if not GT:
        print("[ERROR] GT reconstruction produced empty GT. Check CSV contents.")
        sys.exit(1)
    print(f"[INFO] Built GT with {len(GT)} timesteps (times: {sorted(GT.keys())[:5]} ... )")

    # Generate walks
    walks = generate_walks(GT, num_walks=args.num_walks, walk_length=args.walk_length, seed=args.seed)
    print(f"[INFO] Generated {len(walks)} walks (num_walks={args.num_walks}, walk_length={args.walk_length})")

    if len(walks) == 0:
        print("[ERROR] No walks generated. Aborting.")
        sys.exit(1)

    # Train embeddings
    if not _HAVE_GENSIM:
        print("[ERROR] gensim required to train embeddings; install with: pip install gensim")
        sys.exit(1)
    print("[INFO] Training Word2Vec embeddings...")
    model = Word2Vec(walks, vector_size=args.emb_dim, window=args.emb_window, min_count=1, sg=1, workers=1, seed=args.seed, epochs=args.emb_epochs)
    embeddings = {k: model.wv[k] for k in model.wv.index_to_key}
    emb_csv = os.path.join(outdir, f"{prefix}_embeddings.csv")
    save_embeddings_csv(embeddings, emb_csv)
    print("[INFO] Saved embeddings ->", emb_csv)

    # Anomaly detection
    print("[INFO] Running anomaly detection (IsolationForest if available)...")
    anomalies = detect_anomalies_on_embeddings(embeddings, method="iforest", contamination=args.if_contam, random_state=args.seed)
    anom_csv = os.path.join(outdir, f"{prefix}_anomalies.csv")
    pd.DataFrame(anomalies).to_csv(anom_csv, index=False)
    print("[INFO] Saved anomalies ->", anom_csv)

    # Optional plot
    if args.plot:
        png = os.path.join(outdir, f"{prefix}_embeddings_2d.png")
        plot_embeddings_2d(embeddings, anomalies, png)
        print("[INFO] Saved embeddings 2D plot ->", png)

    # Also save a summary joined with selection or R history if available
    try:
        df_anom = pd.DataFrame(anomalies)
        # join with R history last timestep if exists
        if r_hist_df is not None:
            # r_hist_df is likely one row per time with columns time + nodes
            # pick last row
            last = r_hist_df.tail(1)
            # melt into node,r_value pairs
            melted = last.melt(id_vars=['time'], var_name='node', value_name='R_value')
            # node column may contain numeric or not, ensure string
            melted['node'] = melted['node'].astype(str)
            merged = pd.merge(df_anom, melted[['node','R_value']], left_on='node', right_on='node', how='left')
            merged.to_csv(os.path.join(outdir, f"{prefix}_anomalies_with_R.csv"), index=False)
            print("[INFO] Saved anomalies joined with last R_history ->", os.path.join(outdir, f"{prefix}_anomalies_with_R.csv"))
    except Exception as e:
        warnings.warn(f"Could not join anomalies and R_history: {e}")

    print("[DONE] Outputs in", outdir)

if __name__ == "__main__":
    main()

