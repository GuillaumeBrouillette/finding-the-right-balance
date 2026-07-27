r"""
Generate the paper's geometric figures as TikZ snippets (vector, consistent
with the hand-drawn lune/obstruction figures in RNGscore_regimes.tex).

Two outputs, written to ``plots/`` and ``\input`` by the paper inside figure
environments (the document preamble defines the node styles qnode/cand/
selnode/activenode and the figblue/figred/figgray colours used here):

  fig_rng_graph.tex      A real relative neighbourhood graph over a seeded 2D
                         point set, edges computed by the empty-lune test.
                         Replaces the oversimplified 5-point path.

  fig_knn_rng_mmr.tex    Three side-by-side panels over ONE shared point set
                         (a query with a tight near-duplicate cluster plus
                         scattered candidates): the top-k selected by k-NN, by
                         the RNG-Score, and by MMR. Shows k-NN spending the
                         budget on the near-duplicate cluster while the
                         RNG-Score keeps one representative and disperses, and
                         MMR pushing further out (onto less relevant points).

Everything is Euclidean planar geometry (the natural reading of the lune /
RNG figures); this is illustration, not the cosine production code. Run:

    python plot_geometry.py --out_dir ../plots
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np

Pt = Tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _d(a: Pt, b: Pt) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def rng_edges(pts: Sequence[Pt]) -> List[Tuple[int, int]]:
    """Edges of the relative neighbourhood graph: (i,j) is an edge iff no third
    point z lies in the open lune, i.e. iff for all z,
    not (d(z,i) < d(i,j) and d(z,j) < d(i,j))."""
    n = len(pts)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dij = _d(pts[i], pts[j])
            blocked = any(
                z != i and z != j
                and _d(pts[z], pts[i]) < dij and _d(pts[z], pts[j]) < dij
                for z in range(n)
            )
            if not blocked:
                edges.append((i, j))
    return edges


def knn_select(pts: Sequence[Pt], q: Pt, k: int) -> List[int]:
    return list(np.argsort([_d(p, q) for p in pts])[:k])


def rng_score_select(pts: Sequence[Pt], q: Pt, k: int, alpha: float) -> List[int]:
    dq = np.array([_d(p, q) for p in pts])
    scores = dq.copy()
    for w in range(len(pts)):
        for v in range(len(pts)):
            if dq[v] < dq[w]:
                scores[w] += max(dq[w] - _d(pts[v], pts[w]) + alpha, 0.0)
    return list(np.argsort(scores)[:k])


def mmr_select(pts: Sequence[Pt], q: Pt, k: int, lam: float,
               sigma: float = 1.0) -> List[int]:
    """Greedy MMR with RBF relevance/similarity from Euclidean distances."""
    rel = np.exp(-np.array([_d(p, q) for p in pts]) / sigma)
    selected: List[int] = [int(np.argmax(rel))]
    while len(selected) < k:
        best, best_val = None, -np.inf
        for i in range(len(pts)):
            if i in selected:
                continue
            sim = max(np.exp(-_d(pts[i], pts[j]) / sigma) for j in selected)
            val = lam * rel[i] - (1.0 - lam) * sim
            if val > best_val:
                best_val, best = val, i
        selected.append(int(best))
    return selected


# ---------------------------------------------------------------------------
# TikZ emission
# ---------------------------------------------------------------------------

def _node(style: str, p: Pt, label: str = "") -> str:
    lab = f" {label}" if label else ""
    return f"  \\node[{style}] at ({p[0]:.2f},{p[1]:.2f}) {{}}{lab};"


def emit_rng_graph(path: str, seed: int = 7, n: int = 14) -> None:
    rng = np.random.default_rng(seed)
    # Spread points over a rectangle, nudged to avoid collinear degeneracies.
    pts = [(round(float(x), 2), round(float(y), 2))
           for x, y in rng.uniform([0.2, 0.2], [4.8, 3.4], size=(n, 2))]
    edges = rng_edges(pts)
    # Bare TikZ body (no tikzpicture wrapper): \input inside the lune figure's
    # scope. Requires a ``pt`` node style from the enclosing tikzpicture.
    lines = []
    for i, j in edges:
        lines.append(f"  \\draw[thick, figblue] ({pts[i][0]:.2f},{pts[i][1]:.2f}) "
                     f"-- ({pts[j][0]:.2f},{pts[j][1]:.2f});")
    for p in pts:
        lines.append(f"  \\node[pt] at ({p[0]:.2f},{p[1]:.2f}) {{}};")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}  ({n} points, {len(edges)} RNG edges)")


def _panel(pts: Sequence[Pt], q: Pt, sel: Sequence[int], title: str,
           xshift: float) -> List[str]:
    sset = set(sel)
    out = [f"\\begin{{scope}}[xshift={xshift:.1f}cm]",
           f"  \\node[paneltitle] at (2.4,3.5) {{{title}}};"]
    # faint query-to-selected spokes for readability
    for i in sel:
        out.append(f"  \\draw[figgray!50, thin] ({q[0]:.2f},{q[1]:.2f}) -- "
                   f"({pts[i][0]:.2f},{pts[i][1]:.2f});")
    for i, p in enumerate(pts):
        style = "selnode" if i in sset else "cand"
        out.append(f"  \\node[{style}] at ({p[0]:.2f},{p[1]:.2f}) {{}};")
    out.append(f"  \\node[qnode, label={{[tinylabel]below:$q$}}] "
               f"at ({q[0]:.2f},{q[1]:.2f}) {{}};")
    out.append(r"\end{scope}")
    return out


def emit_knn_rng_mmr(path: str, k: int = 5, alpha: float = 0.0,
                     lam: float = 0.5) -> None:
    # Hand-tuned layout: the near-duplicate cluster is the CLOSEST group to q
    # (so k-NN spends its whole budget on it); scattered distinct candidates
    # sit farther out in varied directions. Deterministic.
    q: Pt = (0.0, 0.0)
    # 5 near-duplicates, a tight but visible clump, the closest group to q
    # (k-NN grabs all five).
    cluster = [(1.00, 0.00), (1.16, 0.13), (1.07, -0.15), (0.90, 0.11),
               (1.19, -0.04)]
    # 4 distinct points just beyond the cluster, in directions AWAY from it
    # (so the obstruction penalty does not fire on them): the RNG-Score
    # disperses here after keeping one cluster representative.
    near = [(0.10, 1.40), (0.00, -1.45), (-1.35, 0.20), (-0.90, 1.05)]
    # 5 far points: only a spread-maximising method (MMR) reaches these.
    far = [(2.60, 0.50), (2.20, -1.70), (2.90, 1.70), (1.90, 2.40),
           (2.70, -0.60)]
    pts: List[Pt] = cluster + near + far
    panels = [("$k$NN", knn_select(pts, q, k), 0.0),
              ("\\RNGSCORE", rng_score_select(pts, q, k, alpha), 5.6),
              (f"\\MMR\\ ($\\lambda={lam:g}$)", mmr_select(pts, q, k, lam), 11.2)]
    lines = [r"\begin{tikzpicture}[scale=0.9]"]
    for title, sel, xs in panels:
        lines += _panel(pts, q, sel, title, xs)
    lines.append(r"\end{tikzpicture}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}  (kNN={knn_select(pts,q,k)}, "
          f"RNG={rng_score_select(pts,q,k,alpha)}, MMR={mmr_select(pts,q,k,lam)})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="plots")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.5)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    emit_rng_graph(os.path.join(args.out_dir, "fig_rng_graph.tex"), seed=args.seed)
    emit_knn_rng_mmr(os.path.join(args.out_dir, "fig_knn_rng_mmr.tex"),
                     alpha=args.alpha, lam=args.lam)


if __name__ == "__main__":
    main()
