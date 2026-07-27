"""
Paper figures derived from the analyze_regimes.py outputs.

Generates, from the analysis CSVs of a redundancy-sweep run directory:

  fig_alpha_sweep.pdf     S-Recall@5 as a function of the margin alpha on
                          clean (rho=0) and redundant (rho=0.25) pools, with
                          the kNN baselines as horizontal references.

  fig_threshold.pdf       (a) pooled validation objective of the decision
                          rule as a function of the trigger threshold tau
                          (fallback diversifier fixed at the selected D*),
                          with the always-kNN / always-D* extremes, the
                          tuned tau* and the deployed tau=h annotated.
                          (b) test S-Recall@5 of always-kNN, always-D*, the
                          frozen rule (tau=h, from
                          frozen_rule_transfer_summary.csv) and the
                          all-methods oracle at every injection level.

  fig_generation.pdf      Answer quality (--gen_metrics, default EM and
                          F1) versus injected redundancy for kNN, the
                          fallback diversifier and the frozen rule; one
                          panel per generation run and metric (--gen_runs).

  fig_crossover.pdf       S-Recall versus measured pool redundancy, one
                          panel per run (2-column grid), shared y-scale.

Every rho axis uses the true injection levels on a symlog scale
(linear below RHO_LINTHRESH so rho=0 stays on-axis), so unequal grid
steps are spaced according to their value rather than evenly.

Multi-panel figures are additionally written as one PDF per panel
(suffix _a, _b, ...) for LaTeX subfigure layouts; the combined PDF is
kept for older drafts.

Usage::

    python plot_regimes.py --run_dir ../results/<redundancy run> \
        --gen_run_dir ../results/<generation run> --out_dir ../plots \
        --crossover "TITLE=path/analysis_summary.csv" ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#005AB5"
RED = "#DC3220"
GRAY = "#828282"

# Linear floor of the symlog rho axes: keeps rho=0 on-axis while the
# refined low end of the grid stays legible.
RHO_LINTHRESH = 0.025

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})


def _panel_path(out_path: str, index: int) -> str:
    root, ext = os.path.splitext(out_path)
    return f"{root}_{chr(ord('a') + index)}{ext}"


def _save_panels(draw_fns, sizes, out_path: str) -> None:
    """Write one single-panel PDF per draw function (suffix _a, _b, ...)."""
    for i, (draw, size) in enumerate(zip(draw_fns, sizes)):
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        fig.tight_layout()
        path = _panel_path(out_path, i)
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}")


def _rho_axis(ax, levels) -> None:
    """True-value symlog x-axis for injection levels.

    Ticks sit at the actual levels; labels that would collide on the
    symlog scale (e.g. 0.15 next to 0.1) are dropped, their ticks kept.
    """
    ax.set_xscale("symlog", linthresh=RHO_LINTHRESH)
    ax.set_xticks(levels)

    def pos(v):
        return v / RHO_LINTHRESH if v <= RHO_LINTHRESH \
            else 1.0 + math.log10(v / RHO_LINTHRESH)

    span = pos(max(levels)) - pos(min(levels))
    labels, last = [], None
    for v in levels:
        if last is not None and (pos(v) - last) < 0.09 * span:
            labels.append("")
        else:
            labels.append(f"{v:g}")
            last = pos(v)
    ax.set_xticklabels(labels, fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel(r"Injected redundancy $\rho$")


def alpha_sweep_figure(run_dir: str, out_path: str, objective: str = "S-Recall@k") -> None:
    members = pd.read_csv(os.path.join(run_dir, "analysis_members.csv"))
    pat = re.compile(r"^RNG\((-?[\d.]+)\)$")
    members["alpha"] = members["Method"].map(
        lambda m: float(pat.match(m).group(1)) if pat.match(m) else None
    )
    rng = members.dropna(subset=["alpha"]).sort_values("alpha")
    knn = members[members.Method == "kNN"].set_index("rho")[objective]

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for rho, color, label in [(0.0, BLUE, r"$\rho=0$ (clean)"),
                              (0.25, RED, r"$\rho=0.25$ (redundant)")]:
        cur = rng[rng.rho == rho]
        ax.plot(cur["alpha"], cur[objective], "-o", ms=3.5, color=color,
                label=f"RNG-Score, {label}")
        ax.axhline(knn[rho], color=color, ls=":", lw=1)
        ax.annotate("kNN", xy=(cur["alpha"].iloc[-1], knn[rho]),
                    xytext=(3, 2), textcoords="offset points",
                    color=color, fontsize=7)
    ax.set_xlabel(r"Margin $\alpha$")
    ax.set_ylabel("S-Recall@5")
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def threshold_figure(run_dir: str, out_path: str,
                     objective: str = "S-Recall@k",
                     frozen_csv: str | None = None,
                     deployed_tau: float = 2.0) -> None:
    curve = pd.read_csv(os.path.join(run_dir, "analysis_threshold_curve.csv"))
    rule = pd.read_csv(os.path.join(run_dir, "analysis_threshold.csv"))
    with open(os.path.join(run_dir, "analysis_threshold_info.json"),
              encoding="utf-8") as f:
        info = json.load(f)
    d_star, tau_star = info["D"], info["tau"]
    run_name = os.path.basename(os.path.normpath(run_dir))

    def draw_left(ax) -> None:
        cur = curve[curve.D == d_star].sort_values("tau")
        ax.plot(cur["tau"], cur["val mean"], "-", color=BLUE, lw=1.5)
        ax.axvline(tau_star, color=GRAY, ls="--", lw=1)
        ax.annotate(rf"$\tau^*={tau_star:.2f}$",
                    xy=(tau_star, cur["val mean"].max()),
                    xytext=(-46, 2), textcoords="offset points",
                    color=GRAY, fontsize=8)
        ax.axvline(deployed_tau, color=RED, ls="--", lw=1)
        ax.annotate(r"$\tau=h$",
                    xy=(deployed_tau, cur["val mean"].min()),
                    xytext=(4, 4), textcoords="offset points",
                    color=RED, fontsize=8)
        ax.annotate("always k-NN", xy=(cur["tau"].min(), cur["val mean"].iloc[0]),
                    xytext=(5, 8), textcoords="offset points",
                    color=GRAY, fontsize=7)
        ax.annotate(f"always {d_star}",
                    xy=(cur["tau"].max(), cur["val mean"].iloc[-1]),
                    xytext=(-56, -10), textcoords="offset points",
                    color=GRAY, fontsize=7)
        ax.set_xlabel(r"Trigger threshold $\tau$")
        ax.set_ylabel(f"Validation {objective.replace('@k', '@5')}")
        ax.spines[["top", "right"]].set_visible(False)

    def draw_right(ax) -> None:
        lev = rule[rule[rule.columns[0]] != "pooled"].copy()
        lev["rho"] = lev[lev.columns[0]].astype(float)
        levels = sorted(lev["rho"].unique())

        # Frozen rule at the deployed tau (the rule used throughout the
        # paper); falls back to the tuned-tau rows when the transfer summary
        # is unavailable.
        frozen = None
        if frozen_csv and os.path.exists(frozen_csv):
            fz = pd.read_csv(frozen_csv)
            fz = fz[(fz["run"] == run_name) & (fz["level"] != "pooled")]
            if not fz.empty:
                frozen = fz.assign(rho=fz["level"].astype(float)).sort_values("rho")

        def sel(name):
            return lev[lev.Selector == name].sort_values("rho")

        knn, alw = sel("kNN"), sel(f"always {d_star}")
        orc = sel("all-methods oracle")
        ax.plot(knn["rho"], knn[objective], "-", marker="s", ms=5,
                color=GRAY, lw=1.2, label="Always k-NN")
        ax.plot(alw["rho"], alw[objective], "-", marker="o", ms=5,
                color=BLUE, lw=1.2, label=f"Always {d_star}")
        if frozen is not None:
            ax.plot(frozen["rho"], frozen["rule"], "-", marker="D", ms=5,
                    color=RED, lw=1.2,
                    label=r"Rule ($\tau=h$)")
        else:
            r = lev[lev.Selector.str.startswith("rule (")]
            r = r.groupby("rho", as_index=False)[objective].mean()
            ax.plot(r["rho"], r[objective], "-", marker="D", ms=5,
                    color=RED, lw=1.2, label=r"Rule ($\tau^*$)")
        ax.plot(orc["rho"], orc[objective], ":", marker="_", ms=5,
                color="black", lw=1.2, label="Oracle")
        _rho_axis(ax, levels)
        ax.set_ylabel("Test S-Recall@5")
        ax.legend(loc="lower left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                   gridspec_kw={"width_ratios": [1.1, 1.4]})
    draw_left(ax1)
    draw_right(ax2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    _save_panels([draw_left, draw_right], [(3.3, 2.6), (3.9, 2.6)], out_path)


def generation_figure(runs, out_path: str,
                      deployed_tau: float = 2.0,
                      metrics=("EM", "F1")) -> None:
    """Answer quality versus injected redundancy for kNN, the fallback
    diversifier, and the reconstructed frozen decision rule.

    ``runs`` is a list of (title, run_dir) pairs; each run contributes one
    panel per metric in ``metrics`` (grid: one row per run when several
    metrics are drawn, one row of runs for a single metric; plus one PDF
    per panel for LaTeX subfigure layouts).  Per run, prefers
    ``analysis_generation_frozen.csv`` (rule reconstructed at the deployed
    tau, matching the paper's generation table); falls back to
    ``analysis_generation.csv``, whose rule column uses the tuned tau* and
    is then labeled accordingly.
    """
    draws, titles = [], []
    for title, run_dir in runs:
        frozen_path = os.path.join(run_dir, "analysis_generation_frozen.csv")
        use_frozen = os.path.exists(frozen_path)
        g = pd.read_csv(frozen_path if use_frozen else
                        os.path.join(run_dir, "analysis_generation.csv"))
        rule_label = (r"Rule ($\tau=h$)" if use_frozen
                      else r"Rule ($\tau^*$)")
        level_col = g.columns[0]
        reserved = {level_col, "metric", "n", "kNN", "rule", "trigger rate"}
        dcol = next(c for c in g.columns
                    if c not in reserved and not c.startswith("p("))
        levels = sorted(g[level_col].unique())

        def make_draw(met, g=g, dcol=dcol, levels=levels,
                      level_col=level_col, rule_label=rule_label,
                      legend=not draws):
            def draw(ax) -> None:
                sub = g[g.metric == met]
                for name, lab, c, mk in [
                        ("kNN", "Always k-NN", GRAY, "s"),
                        (dcol, f"Always {dcol}", BLUE, "o"),
                        ("rule", rule_label, RED, "D")]:
                    cur = sub.set_index(level_col).reindex(levels)
                    ax.plot(levels, cur[name], "-", marker=mk, ms=5,
                            color=c, lw=1.2, label=lab)
                _rho_axis(ax, levels)
                ax.set_ylabel(f"Test {met}")
                if legend:
                    ax.legend(loc="lower left", frameon=False, fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
            return draw

        for met in metrics:
            draws.append(make_draw(met, legend=not draws))
            titles.append(f"{title}, {met}" if len(metrics) > 1 else title)

    ncols = len(metrics) if len(metrics) > 1 else len(runs)
    nrows = math.ceil(len(draws) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.6 * ncols, 2.8 * nrows),
                             squeeze=False)
    for ax, draw, title in zip(axes.ravel(), draws, titles):
        draw(ax)
        if len(draws) > 2 or len(metrics) == 1:
            ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    _save_panels(draws, [(3.6, 2.6)] * len(draws), out_path)


def crossover_figure(panels, out_path: str,
                     objective: str = "S-Recall@k", ncols: int = 2) -> None:
    """S-Recall versus *measured* pool redundancy, one panel per run.

    ``panels`` is a list of (title, analysis_summary.csv path) pairs; each
    summary must carry a PoolRedundancy column (analyze_regimes adds it).
    Methods shown: kNN, tuned MMR, tuned RNG, greedy DPP.  The x-axis is the
    near-duplicate pair fraction, so sweeps from different mechanisms and
    datasets are comparable; symlog scaling keeps the dense low end legible.
    All panels share the same y-limits so levels are comparable across
    panels.
    """
    series = [("kNN", "kNN", "k-NN", GRAY, "s"),
              ("MMR*", "MMR\\*", "MMR*", BLUE, "o"),
              ("RNG*", "RNG\\*", "RNG-Score*", RED, "D"),
              ("Greedy-DPP", "Greedy-DPP", "Greedy DPP", "black", "^")]
    frames = [(title, pd.read_csv(path)) for title, path in panels]
    for title, s in frames:
        if "PoolRedundancy" not in s.columns:
            raise SystemExit(f"{title} lacks PoolRedundancy; rerun analyze_regimes.")

    # Shared y-limits across panels.
    vals = pd.concat(
        s[s.Method.str.match(pat)][objective]
        for _, s in frames for _, pat, _, _, _ in series
    )
    pad = 0.04 * (vals.max() - vals.min())
    ylim = (vals.min() - pad, vals.max() + pad)
    lin = 1e-4  # symlog linear floor: keeps rho=0 on-axis
    xmax = max(s["PoolRedundancy"].max() for _, s in frames) * 1.6

    def make_draw(index, title, s, single: bool):
        def draw(ax) -> None:
            for _, pat, lab, color, mk in series:
                cur = s[s.Method.str.match(pat)].sort_values("PoolRedundancy")
                if cur.empty:
                    continue
                ax.plot(cur["PoolRedundancy"].clip(lower=0), cur[objective],
                        marker=mk, ms=4, lw=1.3, color=color, label=lab)
            ax.set_xscale("symlog", linthresh=lin)
            ax.set_xlim(-lin / 2, xmax)
            ax.set_ylim(*ylim)
            ax.set_xlabel("Measured pool redundancy")
            ax.spines[["top", "right"]].set_visible(False)
            if single:
                # titles become LaTeX subcaptions in the paper
                if index % ncols == 0:
                    ax.set_ylabel(objective.replace("@k", ""))
                if index == 0:
                    ax.legend(loc="lower left", frameon=False, fontsize=7)
            else:
                ax.set_title(title, fontsize=9)
        return draw

    nrows = math.ceil(len(frames) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                             sharey=True, squeeze=False)
    for i, (title, s) in enumerate(frames):
        ax = axes[i // ncols][i % ncols]
        make_draw(i, title, s, single=False)(ax)
        if i % ncols == 0:
            ax.set_ylabel(objective.replace("@k", ""))
    for j in range(len(frames), nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])
    axes[0][0].legend(loc="lower left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")
    _save_panels(
        [make_draw(i, t, s, single=True) for i, (t, s) in enumerate(frames)],
        [(3.0, 2.4)] * len(frames), out_path)


def oracle_figure(panels, out_path: str, x_axis: str = "redundancy") -> None:
    """Fraction of test queries on which the all-methods per-query oracle
    beats k-NN, versus measured pool redundancy: one curve per sweep,
    solid for injection sweeps, dashed for chunk-overlap sweeps.

    ``panels``: (label, oracle_csv, summary_csv, color, linestyle, marker)
    tuples; the level column (rho or overlap) is taken from each file's
    first column and mapped to the PoolRedundancy of the summary's kNN rows.
    """
    lin = 1e-4
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for label, oracle_csv, summary_csv, color, ls, mk in panels:
        o = pd.read_csv(oracle_csv)
        s = pd.read_csv(summary_csv)
        lvl_o, lvl_s = o.columns[0], s.columns[0]
        red = s[s.Method == "kNN"].set_index(lvl_s)["PoolRedundancy"]
        cur = o[o.Selector == "All-methods oracle"].copy()
        if x_axis == "level":
            cur = cur.sort_values(lvl_o)
            ax.plot(cur[lvl_o], cur["% beats kNN"], ls=ls,
                    marker=mk, ms=4, lw=1.3, color=color, label=label)
        else:
            cur["red"] = cur[lvl_o].map(red)
            cur = cur.sort_values("red")
            ax.plot(cur["red"].clip(lower=0), cur["% beats kNN"], ls=ls,
                    marker=mk, ms=4, lw=1.3, color=color, label=label)
    if x_axis == "level":
        ax.set_xlabel("Chunk overlap")
        ax.set_xticks([0, 0.25, 0.5, 0.75])
    else:
        ax.set_xscale("symlog", linthresh=lin)
        ax.set_xlabel("Measured pool redundancy")
    ax.set_ylabel("% of queries oracle beats k-NN")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gen_run_dir", default=None,
                   help="Run directory for fig_generation.pdf when the "
                        "generation sweep lives in a different run.")
    p.add_argument("--gen_runs", nargs="*", default=None, metavar="TITLE=DIR",
                   help="Draw fig_generation.pdf from several generation "
                        "runs (one panel per run and metric); overrides "
                        "--gen_run_dir.")
    p.add_argument("--gen_metrics", nargs="*", default=["EM", "F1"],
                   help="Metrics drawn by fig_generation.pdf (default both; "
                        "the paper uses EM only, F1 behaving analogously).")
    p.add_argument("--out_dir", default="plots")
    p.add_argument("--frozen_csv", default=None,
                   help="frozen_rule_transfer_summary.csv (default: sibling "
                        "of --run_dir); provides the deployed-rule rows of "
                        "fig_threshold's right panel.")
    p.add_argument("--crossover", nargs="*", default=None, metavar="TITLE=CSV",
                   help="Additionally draw fig_crossover.pdf from one panel "
                        "per TITLE=path-to-analysis_summary.csv pair.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frozen = args.frozen_csv or os.path.join(
        os.path.dirname(os.path.normpath(args.run_dir)),
        "frozen_rule_transfer_summary.csv")
    # Draw each figure only if its inputs exist in the run directory.
    if os.path.exists(os.path.join(args.run_dir, "analysis_members.csv")):
        alpha_sweep_figure(args.run_dir,
                           os.path.join(args.out_dir, "fig_alpha_sweep.pdf"))
    if os.path.exists(os.path.join(args.run_dir, "analysis_threshold.csv")):
        threshold_figure(args.run_dir,
                         os.path.join(args.out_dir, "fig_threshold.pdf"),
                         frozen_csv=frozen)
    if args.gen_runs:
        gen_runs = [tuple(spec.split("=", 1)) for spec in args.gen_runs]
    else:
        gen_dir = args.gen_run_dir or args.run_dir
        gen_runs = [(os.path.basename(os.path.normpath(gen_dir)), gen_dir)] \
            if any(os.path.exists(os.path.join(gen_dir, f))
                   for f in ("analysis_generation_frozen.csv",
                             "analysis_generation.csv")) else []
    if gen_runs:
        generation_figure(gen_runs,
                          os.path.join(args.out_dir, "fig_generation.pdf"),
                          metrics=tuple(args.gen_metrics))
    if args.crossover:
        panels = [tuple(spec.split("=", 1)) for spec in args.crossover]
        crossover_figure(panels,
                         os.path.join(args.out_dir, "fig_crossover.pdf"))


if __name__ == "__main__":
    main()
