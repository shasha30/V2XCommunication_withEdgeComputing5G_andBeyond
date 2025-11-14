#!/usr/bin/env python3
"""
plot_embeddings_only.py

Read embeddings CSV and anomalies CSV and create a 2D visualization.
Tries PCA -> t-SNE (if sklearn available). If not, uses PCA-only via numpy SVD.

Expected input files (in outdir):
  <prefix>_embeddings.csv   (columns: node,v0,v1,...)
  <prefix>_anomalies.csv    (columns: node,anom_score,is_anom) -- is_anom flag optional

Produces:
  <outdir>/<prefix>_embeddings_2d_plot.png
"""

import os, sys, argparse, warnings
import math
import pandas as pd
import numpy as np

# plotting
_have_mpl = True
try:
    import matplotlib.pyplot as plt
    import matplotlib
except Exception:
    _have_mpl = False
    warnings.warn("matplotlib not available; cannot plot.")

# sklearn (optional)
_have_sklearn = True
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:
    _have_sklearn = False
    warnings.warn("scikit-learn not available; will use PCA-only (numpy SVD).")

def read_embeddings(path):
    df = pd.read_csv(path)
    if 'node' not in df.columns:
        # assume first column is node id
        firstcol = df.columns[0]
        df = df.rename(columns={firstcol: 'node'})
    nodes = df['node'].astype(str).tolist()
    vec_cols = [c for c in df.columns if c != 'node']
    X = df[vec_cols].values.astype(float)
    return nodes, X

def read_anomalies(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'node' not in df.columns:
        # try lowercase variants
        for c in df.columns:
            if c.lower() == 'node':
                df = df.rename(columns={c: 'node'})
                break
    df['node'] = df['node'].astype(str)
    return df

def reduce_dim(X, method='pca_tsne', random_state=42):
    # X: (n, d)
    if method == 'pca_tsne' and _have_sklearn:
        # PCA to 20 dims (or less) then TSNE to 2D
        n_components = min(20, X.shape[1])
        Xp = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
        X2 = TSNE(n_components=2, init='pca', random_state=random_state, perplexity=30).fit_transform(Xp)
        return X2
    else:
        # fallback PCA via numpy SVD to 2D
        # subtract mean
        Xc = X - X.mean(axis=0)
        # SVD
        try:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            # project onto first 2 principal components
            X2 = np.dot(Xc, Vt.T[:, :2])
            return X2
        except Exception as e:
            raise RuntimeError(f"PCA fallback failed: {e}")

def plot_2d(X2, nodes, anomalies_df=None, out_png='embeddings_2d.png', label_top=10):
    if not _have_mpl:
        raise RuntimeError("matplotlib not installed; cannot create plot.")
    fig, ax = plt.subplots(figsize=(10,8))
    # base scatter
    ax.scatter(X2[:,0], X2[:,1], s=18, alpha=0.7, label='nodes', c='lightgray', edgecolors='none')
    # if anomalies present, color them
    if anomalies_df is not None:
        # prepare lookup
        anom_map = {}
        for _, r in anomalies_df.iterrows():
            anom_map[str(r['node'])] = r
        is_anom = [1 if (n in anom_map and int(anom_map[n].get('is_anom', 0)) == 1) else 0 for n in nodes]
        # indices
        idx_anom = [i for i,v in enumerate(is_anom) if v==1]
        if idx_anom:
            ax.scatter(X2[idx_anom,0], X2[idx_anom,1], s=36, color='red', alpha=0.9, label='anomaly')
        # optionally annotate top N by anomaly score
        if 'anom_score' in anomalies_df.columns:
            # build score lookup (higher -> more anomalous in our scripts)
            score_map = {str(r['node']): float(r['anom_score']) for _,r in anomalies_df.iterrows() if r['anom_score']!=None}
            # compute top N among nodes present
            present_scores = [(i, nodes[i], score_map.get(nodes[i], -1.0)) for i in range(len(nodes))]
            # sort by score desc
            present_scores_sorted = sorted(present_scores, key=lambda x: x[2], reverse=True)
            topn = [t for t in present_scores_sorted if t[2] is not None and t[2] != -1.0][:label_top]
            for idx, node_id, sc in topn:
                ax.text(X2[idx,0], X2[idx,1], str(node_id), fontsize=8, color='darkred')
    ax.set_title("Node Embeddings (2D). Red = anomalies")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png

def main():
    ap = argparse.ArgumentParser(description="Plot embeddings 2D from existing CSV outputs.")
    ap.add_argument("--prefix", required=False, default="porto_result", help="file prefix (default porto_result)")
    ap.add_argument("--outdir", required=False, default="results_v2", help="folder where embeddings/anomalies CSVs live")
    ap.add_argument("--png", required=False, default=None, help="output png filename (optional)")
    ap.add_argument("--label-top", type=int, default=10, help="label top N anomalies on the plot")
    ap.add_argument("--method", choices=['pca_tsne','pca_only'], default='pca_tsne', help="reduce method preference")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emb_csv = os.path.join(args.outdir, f"{args.prefix}_embeddings.csv")
    anom_csv = os.path.join(args.outdir, f"{args.prefix}_anomalies.csv")
    if not os.path.exists(emb_csv):
        print(f"[ERROR] Embeddings file not found: {emb_csv}")
        sys.exit(1)
    nodes, X = read_embeddings(emb_csv)
    print(f"[INFO] Read embeddings: {X.shape[0]} nodes, dim={X.shape[1]}")

    anom_df = None
    if os.path.exists(anom_csv):
        anom_df = read_anomalies(anom_csv)
        print(f"[INFO] Read anomalies: {len(anom_df)} rows")
    else:
        print(f"[WARN] Anomalies file not found: {anom_csv} (continuing without anomaly overlays)")

    # reduce dims
    method = args.method
    if method == 'pca_tsne' and not _have_sklearn:
        print("[WARN] sklearn not available; falling back to PCA-only.")
        method = 'pca_only'
    print(f"[INFO] Reducing dims using method: {method}")
    X2 = reduce_dim(X, method=('pca_tsne' if method=='pca_tsne' else 'pca_only'), random_state=args.seed)

    # output filename
    out_png = args.png if args.png else os.path.join(args.outdir, f"{args.prefix}_embeddings_2d_plot.png")
    print(f"[INFO] Saving plot to: {out_png}")
    try:
        plot_2d(X2, nodes, anomalies_df=anom_df, out_png=out_png, label_top=args.label_top)
        print("[DONE] Plot saved.")
    except Exception as e:
        print("[ERROR] Plotting failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    main()
