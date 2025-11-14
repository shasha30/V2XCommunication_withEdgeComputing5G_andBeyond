#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_dnr.py
Merged pipeline:
 - Algorithm 1: Grid encoding (trajectories -> grid_id)
 - Algorithm 2: Dynamic social network GT (per-timestep graphs)
 - Algorithm 3: Dynamic node selection (representative sampling)
Outputs CSVs for GT and selection results.

Usage:
  python3 grid_dnr.py --input sample_1000.csv --output-prefix porto_result --grid-size 1000 --visualize
"""

import os
import math
import argparse
import random
import csv
import ast
import warnings
from collections import defaultdict
from typing import Dict, Iterable

# third-party
try:
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
except Exception as e:
    print("Required packages missing. Install: pandas, networkx, matplotlib")
    raise

# optional partitioning lib
try:
    import nxmetis
    _HAS_NXMETIS = True
except Exception:
    _HAS_NXMETIS = False

# global meta used for visualization (set in main)
GLOBAL_META = None

# ---------- Exporter helpers ----------
def export_GT_edges_to_csv(GT: Dict[int, 'nx.Graph'], out_path: str):
    """Write GT edges to CSV with attributes."""
    with open(out_path, "w", newline="") as fh:
        fields = ["time", "u", "v", "grid", "cooccurrence", "weight"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in sorted(GT.keys()):
            Gt = GT[t]
            for u, v, ed in Gt.edges(data=True):
                writer.writerow({
                    "time": t,
                    "u": u,
                    "v": v,
                    "grid": ed.get("grid", ""),
                    "cooccurrence": ed.get("cooccurrence", ""),
                    "weight": ed.get("weight", "")
                })

def export_GT_nodes_to_csv(GT: Dict[int, 'nx.Graph'], out_path: str):
    """Write GT node attributes per time to CSV."""
    with open(out_path, "w", newline="") as fh:
        fields = ["time", "node", "grid_id", "grid_count"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in sorted(GT.keys()):
            Gt = GT[t]
            for n, dat in Gt.nodes(data=True):
                writer.writerow({
                    "time": t,
                    "node": n,
                    "grid_id": dat.get("grid_id", ""),
                    "grid_count": dat.get("grid_count", "")
                })

def export_selected_nodes(V_sel: Dict[int, Iterable], out_csv: str, out_summary: str):
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "selected_node"])
        writer.writeheader()
        for t in sorted(V_sel.keys()):
            for n in V_sel[t]:
                writer.writerow({"time": t, "selected_node": n})
    # summary
    with open(out_summary, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "selected_nodes"])
        writer.writeheader()
        for t in sorted(V_sel.keys()):
            writer.writerow({"time": t, "selected_nodes": " ".join(map(str, V_sel[t]))})

def export_R_history(R_history: Dict[int, Dict[str, float]], out_csv: str):
    """R_history: t -> {node: R_value}"""
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "node", "R_value"])
        writer.writeheader()
        for t in sorted(R_history.keys()):
            for n, r in R_history[t].items():
                writer.writerow({"time": t, "node": n, "R_value": r})

# -----------------------------------------------------------------------------
# PREPROCESSOR for Porto Taxi Dataset or generic CSV
# -----------------------------------------------------------------------------
def load_or_generate_trajs(input_csv=None, demo_n=12, points_per_traj=80, seed=42):
    """
    Load trajectory data. Supports:
      - Standard CSV (traj_id,time,lat,lon)
      - Porto Taxi Dataset (POLYLINE column)
    """
    if input_csv:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(input_csv)
        df = pd.read_csv(input_csv)

        cols = set(c.upper() for c in df.columns)

        # Porto Taxi dataset handler (POLYLINE column with lon-lat pairs)
        if "POLYLINE" in cols:
            print("[INFO] Detected Porto Taxi dataset format. Expanding POLYLINE column...")
            rows = []
            for _, row in df.iterrows():
                try:
                    coords = ast.literal_eval(row["POLYLINE"])
                    trip_id = row.get("TRIP_ID", None) or row.get("trip_id", None)
                    if trip_id is None:
                        continue
                    for t, (lon, lat) in enumerate(coords):
                        rows.append({
                            "traj_id": trip_id,
                            "time": t,
                            "lat": lat,
                            "lon": lon
                        })
                except Exception:
                    continue
            df = pd.DataFrame(rows)
            print(f"[INFO] Expanded to {len(df)} trajectory points from {df['traj_id'].nunique()} trips.")
            return df

        # Generic case (traj_id,time,lat,lon)
        req = {'traj_id', 'time', 'lat', 'lon'}
        if not req.issubset(set(df.columns)):
            raise ValueError(f"Input CSV must contain columns: {req}")
        df['time'] = df['time'].astype(int)
        return df

    # Synthetic fallback (for demo)
    random.seed(seed)
    rows = []
    for traj in range(1, demo_n + 1):
        start_lat = random.uniform(12.90, 13.10)
        start_lon = random.uniform(77.50, 77.90)
        angle = random.uniform(0, 2 * math.pi)
        for t in range(points_per_traj):
            step = 0.0012 + 0.0006 * math.sin(t / 8.0 + traj)
            lat = start_lat + step * math.cos(angle) + random.uniform(-0.0003, 0.0003)
            lon = start_lon + step * math.sin(angle) + random.uniform(-0.0003, 0.0003)
            rows.append({'traj_id': f"v{traj}", 'time': t, 'lat': lat, 'lon': lon})
            angle += 0.02 * math.sin(t / 15.0 + traj / 3.0)
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Algorithm 1: Grid Encoding
# -----------------------------------------------------------------------------
def compute_grid_indices(df, grid_size=100):
    if df.empty:
        return df, {}
    minLng = df["lon"].min()
    maxLng = df["lon"].max()
    minLat = df["lat"].min()
    maxLat = df["lat"].max()
    eps = 1e-12
    lngS = max((maxLng - minLng) / grid_size, eps)
    latS = max((maxLat - minLat) / grid_size, eps)
    gx, gy, gid = [], [], []
    for _, r in df.iterrows():
        x = math.ceil((r["lon"] - minLng) / lngS)
        y = math.ceil((r["lat"] - minLat) / latS)
        x = max(1, min(x, grid_size))
        y = max(1, min(y, grid_size))
        gx.append(x)
        gy.append(y)
        gid.append(10 ** 5 * x + y)
    df["grid_x"], df["grid_y"], df["grid_id"] = gx, gy, gid
    meta = {"minLng": minLng, "maxLng": maxLng, "minLat": minLat, "maxLat": maxLat, "lngS": lngS, "latS": latS, "grid_size": grid_size}
    return df, meta

def visualize_trajectories_grid(df, meta, out_png="traj_grid_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Trajectories on Grid")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    # grid lines
    n = meta.get("grid_size", 100)
    xs = [meta["minLng"] + i * meta["lngS"] for i in range(n + 1)]
    ys = [meta["minLat"] + i * meta["latS"] for i in range(n + 1)]
    for x in xs:
        ax.plot([x, x], [meta["minLat"], meta["maxLat"]], color="lightgray", linewidth=0.3)
    for y in ys:
        ax.plot([meta["minLng"], meta["maxLng"]], [y, y], color="lightgray", linewidth=0.3)
    cmap = plt.get_cmap("tab20")
    for i, tid in enumerate(pd.unique(df["traj_id"])):
        subset = df[df["traj_id"] == tid]
        ax.plot(subset["lon"], subset["lat"], linewidth=1, markersize=2, marker="o", color=cmap(i % 20))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def visualize_network_snapshot(G, t, meta=None, out_dir="network_plots"):
    """
    Plot nodes (vehicles) at approximate grid-center positions and draw edges.
    Requires node attribute 'grid_id' for placement. meta for grid->lon/lat center mapping.
    """
    if meta is None:
        # nothing to plot
        return
    os.makedirs(out_dir, exist_ok=True)
    # compute mapping grid_id -> center lon/lat
    grid_size = meta.get("grid_size", 100)
    lngS, latS = meta["lngS"], meta["latS"]
    minLng, minLat = meta["minLng"], meta["minLat"]

    def grid_to_xy(gid):
        try:
            x = gid // (10 ** 5)
            y = gid % (10 ** 5)
            # center
            cx = minLng + (x - 0.5) * lngS
            cy = minLat + (y - 0.5) * latS
            return cx, cy
        except Exception:
            return None, None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Network snapshot t={t}")
    # draw nodes
    xs, ys, labels = [], [], []
    for n, dat in G.nodes(data=True):
        gid = dat.get("grid_id", None)
        if gid:
            cx, cy = grid_to_xy(gid)
            if cx is not None:
                xs.append(cx); ys.append(cy); labels.append(n)
    ax.scatter(xs, ys, s=16, c='blue')
    for i, lab in enumerate(labels):
        ax.text(xs[i] + 0.00001, ys[i] + 0.00001, lab, fontsize=6)
    # draw edges
    for u, v in G.edges():
        gu = G.nodes[u].get("grid_id", None)
        gv = G.nodes[v].get("grid_id", None)
        if gu and gv:
            xu, yu = grid_to_xy(gu); xv, yv = grid_to_xy(gv)
            if xu is not None and xv is not None:
                ax.plot([xu, xv], [yu, yv], color='gray', linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"network_t_{t}.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# -------------------------
# Algorithm 2: Dynamic Network (pad + build GT)
# -------------------------
def pad_and_build_GT(df_grids):
    """
    Pad each trajectory to max time by repeating last grid and build GT dict t->NetworkX Graph.

    Input df_grids must contain columns: traj_id, time, grid_id

    Output: GT: dict mapping time -> nx.Graph where:
      - graph nodes are traj_id strings
      - node attributes:
          'grid_id'   : int or None
          'grid_count': int (number of vehicles in same grid cell at that time)
      - edge attributes (for edges between vehicles in same grid cell):
          'grid'        : grid_id
          'cooccurrence': number of vehicles in that cell (L)
          'weight'      : normalized weight (1/(L-1) if L>1 else 0)
    """
    if df_grids.empty:
        return {}

    # ensure correct dtypes
    df_grids = df_grids.copy()
    if 'time' not in df_grids.columns or 'traj_id' not in df_grids.columns or 'grid_id' not in df_grids.columns:
        raise ValueError("df_grids must contain columns: traj_id, time, grid_id")

    df_grids['time'] = df_grids['time'].astype(int)
    df_grids['grid_id'] = df_grids['grid_id'].astype(int)
    df_grids['traj_id'] = df_grids['traj_id'].astype(str)

    max_t = int(df_grids['time'].max())

    # collect per-traj rows and pad by repeating last grid
    padded_rows = []
    for tid, g in df_grids.groupby('traj_id'):
        g_sorted = g.sort_values('time')
        times = list(g_sorted['time'].astype(int).values)
        grids = list(g_sorted['grid_id'].astype(int).values)
        last_t = times[-1]; last_grid = grids[-1]
        # existing
        for t0, g0 in zip(times, grids):
            padded_rows.append({'traj_id': tid, 'time': int(t0), 'grid_id': int(g0)})
        # pad to max_t by repeating last grid
        for t_pad in range(last_t + 1, max_t + 1):
            padded_rows.append({'traj_id': tid, 'time': int(t_pad), 'grid_id': int(last_grid)})

    dfp = pd.DataFrame(padded_rows)

    # build GT: one Graph per time step
    GT = {}
    vehicle_list = sorted(dfp['traj_id'].unique())

    for t in range(0, max_t + 1):
        G = nx.Graph()
        # add nodes so node set is stable across snapshots
        G.add_nodes_from(vehicle_list)

        # initialize node attrs to None/0
        nx.set_node_attributes(G, {n: None for n in G.nodes()}, 'grid_id')
        nx.set_node_attributes(G, {n: 0 for n in G.nodes()}, 'grid_count')

        slice_t = dfp[dfp['time'] == t]

        # group by grid cell, connect co-located vehicles
        for grid_id, grp in slice_t.groupby('grid_id'):
            vids = list(grp['traj_id'].unique())
            L = len(vids)
            # set node attributes for vehicles in this grid
            for vid in vids:
                G.nodes[vid]['grid_id'] = int(grid_id)
                G.nodes[vid]['grid_count'] = int(L)

            # create clique among vehicles in same grid cell (only if >=2)
            if L >= 2:
                # normalized pair weight (you can change formula if you prefer)
                pair_weight = 1.0 / (L - 1) if (L - 1) > 0 else 0.0
                for i in range(L):
                    for j in range(i + 1, L):
                        u, v = vids[i], vids[j]
                        # if edge exists, update cooccurrence (increment) and keep grid list if needed
                        if G.has_edge(u, v):
                            prev = G[u][v].get('cooccurrence', 1)
                            G[u][v]['cooccurrence'] = int(prev + 1)
                            G[u][v]['weight'] = float(G[u][v].get('weight', 0.0) + pair_weight)
                        else:
                            G.add_edge(u, v, grid=int(grid_id), cooccurrence=int(1), weight=float(pair_weight))
            else:
                # single vehicle in cell: keep node attrs but no edges
                pass

        GT[t] = G
    return GT

# -------------------------
# Algorithm 3: Dynamic Node Selection
# -------------------------
def compute_delta_E(prev_G, cur_G):
    """
    For each node n compute |ΔE_t^n| = |N_prev(n) Δ N_cur(n)| (symmetric difference).
    """
    nodes = set(prev_G.nodes()).union(set(cur_G.nodes()))
    delta = {}
    for n in nodes:
        nbr_prev = set(prev_G.neighbors(n)) if prev_G.has_node(n) else set()
        nbr_cur = set(cur_G.neighbors(n)) if cur_G.has_node(n) else set()
        delta[n] = len(nbr_prev.symmetric_difference(nbr_cur))
    return delta

def partition_graph(G, P, seed=42):
    """Partition graph G into P parts using nxmetis if available; otherwise fallback."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return [[] for _ in range(P)]
    P_eff = min(P, n)
    if _HAS_NXMETIS and P_eff > 0:
        try:
            _, parts = nxmetis.partition(G, P_eff, recursive=False, seed=seed)
            while len(parts) < P:
                parts.append([])
            return parts[:P]
        except Exception:
            warnings.warn("nxmetis partition failed; falling back to degree chunking")
    # fallback deterministic chunking by degree sort
    nodes_sorted = sorted(nodes, key=lambda x: G.degree(x), reverse=True)
    parts = [[] for _ in range(P_eff)]
    for i, n_ in enumerate(nodes_sorted):
        parts[i % P_eff].append(n_)
    for _ in range(P - P_eff):
        parts.append([])
    return parts

def softmax_weights(vals):
    """vals: dict node->score -> return dict node->prob"""
    if not vals:
        return {}
    maxv = max(vals.values())
    exps = {k: math.exp(v - maxv) for k, v in vals.items()}
    s = sum(exps.values())
    if s == 0:
        return {k: 1.0/len(vals) for k in vals}
    return {k: exps[k]/s for k in vals}

def algorithm3_select(GT, alpha=0.1, seed=42, use_all_first=True,
                      visualize=False, viz_steps=None, decay=0.98, beta=0.0,
                      return_R_history=False):
    """
    Dynamic node selection (Algorithm 3) with optional R_history.

    Returns V_sel or (V_sel, R_history) if return_R_history=True
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

        # warm start
        if t == times[0] and use_all_first:
            V_sel[t] = nodes.copy()
            for v in V_sel[t]:
                R_prev[v] = 0.0
            print(f"[t={t}] V1_sel = all nodes ({len(V_sel[t])})")
            prev_G = G_t
            if return_R_history:
                R_history[t] = dict(R_prev)
            if visualize and (viz_steps is None or t in viz_steps):
                visualize_network_snapshot(G_t, t, meta=GLOBAL_META)
            continue

        # 1) delta
        delta = compute_delta_E(prev_G, G_t)

        # 2) update R with decay
        R_curr = defaultdict(float)
        for n in set(list(R_prev.keys()) + list(delta.keys()) + list(G_t.nodes())):
            decayed = R_prev.get(n, 0.0) * float(decay)
            added = float(delta.get(n, 0.0))
            R_curr[n] = decayed + added

        # optional bias using grid_count
        if beta and beta > 0.0:
            for n in list(G_t.nodes()):
                gcount = int(G_t.nodes[n].get('grid_count', 0) or 0)
                if gcount > 0:
                    R_curr[n] += beta * float(gcount)

        if return_R_history:
            R_history[t] = dict(R_curr)

        # 3) P
        P = max(1, math.ceil(alpha * max(1, len(nodes))))

        # 4) partition
        parts = partition_graph(G_t, P, seed=seed)

        # 5) select reps
        sel = []
        for p in parts:
            if not p:
                continue
            vals = {n: R_curr.get(n, 0.0) for n in p}
            all_zero = all(v == 0.0 for v in vals.values())
            if all_zero:
                # pick node with highest grid_count then highest degree
                best = None; best_score = -1.0
                for n in p:
                    gcount = int(G_t.nodes[n].get('grid_count', 0) or 0)
                    deg = G_t.degree(n)
                    score = float(gcount) + 0.01 * float(deg)
                    if score > best_score:
                        best_score = score; best = n
                rep = best if best is not None else p[0]
            else:
                probs = softmax_weights(vals)
                r = rng.random()
                cum = 0.0
                rep = None
                for n in p:
                    cum += probs.get(n, 0.0)
                    if r <= cum:
                        rep = n
                        break
                if rep is None:
                    rep = max(p, key=lambda x: (vals.get(x, 0.0), G_t.degree(x)))
            sel.append(rep)
            R_curr[rep] = 0.0

        V_sel[t] = sel
        print(f"[t={t}] nodes={len(nodes)} parts={len(parts)} selected={len(sel)} (P={P})")

        if visualize and (viz_steps is None or t in viz_steps):
            visualize_network_snapshot(G_t, t, meta=GLOBAL_META)

        prev_G = G_t
        R_prev = R_curr

    if return_R_history:
        return V_sel, R_history
    return V_sel

# -------------------------
# Main pipeline & CLI
# -------------------------
def main():
    global GLOBAL_META
    ap = argparse.ArgumentParser(description="Pipeline for Porto Taxi Trajectories")
    ap.add_argument("--input", help="Input CSV (Porto Taxi or traj_id,time,lat,lon).")
    ap.add_argument("--output-prefix", dest="output_prefix", default="result",
                    help="Prefix for output files. Files produced: <prefix>_grids.csv, <prefix>_GT_edges.csv, <prefix>_selected_nodes.csv")
    ap.add_argument("--grid-size", type=int, default=100, help="Grid size.")
    ap.add_argument("--alpha", type=float, default=0.1, help="Selection parameter α.")
    ap.add_argument("--visualize", action="store_true", help="Plot trajectories and network snapshots.")
    ap.add_argument("--viz-steps", type=str, default=None, help="Comma-separated times to visualize network snapshots (e.g. 0,5,10).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--beta", type=float, default=0.0, help="grid_count bias for selection (optional)")
    ap.add_argument("--decay", type=float, default=0.98, help="decay for R scores between steps")
    args = ap.parse_args()

    # Load / expand trajectories
    df = load_or_generate_trajs(args.input)
    print(f"[INFO] Loaded {len(df)} rows from {df['traj_id'].nunique()} trips")

    # Grid encoding (Algorithm 1)
    df_grids, meta = compute_grid_indices(df, grid_size=args.grid_size)
    GLOBAL_META = meta  # for visualizer

    # Save grid-encoded trajectories
    out_csv = f"{args.output_prefix}_grids.csv"
    df_grids.to_csv(out_csv, index=False)
    print(f"[INFO] Grid-encoded trajectories saved to: {out_csv}")

    # Optional trajectory plot
    if args.visualize:
        out_png = f"{args.output_prefix}_plot.png"
        visualize_trajectories_grid(df_grids, meta, out_png=out_png)
        print(f"[INFO] Trajectory plot saved to: {out_png}")

    # Algorithm 2: build GT
    print("[INFO] Building dynamic network GT (Algorithm 2)...")
    GT = pad_and_build_GT(df_grids)
    if not GT:
        print("[WARN] GT is empty (no time slices). Exiting.")
        return

    # Export GT
    gt_edges_csv = f"{args.output_prefix}_GT_edges.csv"
    gt_nodes_csv = f"{args.output_prefix}_GT_nodes.csv"
    print("[INFO] Exporting GT edges and nodes...")
    export_GT_edges_to_csv(GT, gt_edges_csv)
    export_GT_nodes_to_csv(GT, gt_nodes_csv)
    print(f"[INFO] GT edges -> {gt_edges_csv}")
    print(f"[INFO] GT nodes -> {gt_nodes_csv}")

    # Algorithm 3: selection with R history
    print("[INFO] Running Algorithm 3 (dynamic node selection) with R history...")
    viz_steps = None
    if args.viz_steps:
        try:
            viz_steps = [int(x.strip()) for x in args.viz_steps.split(",") if x.strip() != ""]
        except Exception:
            viz_steps = None

    V_sel, R_hist = algorithm3_select(GT, alpha=args.alpha, seed=args.seed,
                                      use_all_first=True, visualize=args.visualize,
                                      viz_steps=viz_steps, decay=args.decay,
                                      beta=args.beta, return_R_history=True)

    # Export selection & R history
    selected_nodes_csv = f"{args.output_prefix}_selected_nodes.csv"
    selected_summary_csv = f"{args.output_prefix}_selected_summary.csv"
    export_selected_nodes(V_sel, selected_nodes_csv, selected_summary_csv)
    print(f"[INFO] Selected nodes -> {selected_nodes_csv}")
    print(f"[INFO] Selected summary -> {selected_summary_csv}")

    r_hist_csv = f"{args.output_prefix}_R_history.csv"
    export_R_history(R_hist, r_hist_csv)
    print(f"[INFO] R history -> {r_hist_csv}")

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()

