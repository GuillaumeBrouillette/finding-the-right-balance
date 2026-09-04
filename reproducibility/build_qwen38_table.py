#!/usr/bin/env python3
"""Build the paper's Qwen3.8 clean/heavy answer-quality table."""

import argparse
import csv
import os

METHODS = [("kNN", r"$k$-NN"), ("MMR(0.7)", r"\MMR($\lambda{=}0.7$)"),
           ("rule", r"rule ($\tau{=}h$)")]


def read_cells(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for rho in (0.0, 1.0):
        for metric in ("EM", "F1"):
            found = [row for row in rows
                     if float(row["rho"]) == rho and row["metric"] == metric]
            if len(found) != 1:
                raise SystemExit(f"Expected one rho={rho:g}, metric={metric} row in {path}")
            out[(rho, metric)] = {m: float(found[0][m]) for m, _ in METHODS}
    return out


def marks(values):
    # Rank at the same one-decimal percentage precision shown in the table,
    # so visually tied values receive the same formatting.
    shown = {method: round(100 * value, 1) for method, value in values.items()}
    distinct = sorted(set(shown.values()), reverse=True)
    best, second = distinct[0], distinct[1] if len(distinct) > 1 else None
    return {m: "best" if v == best else "second" if v == second else ""
            for m, v in shown.items()}


def render_value(value, mark):
    number = f"{100 * value:.1f}"
    return rf"\textbf{{{number}}}" if mark == "best" else (
        rf"\underline{{{number}}}" if mark == "second" else number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot", required=True)
    parser.add_argument("--twowiki", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    datasets = [read_cells(args.hotpot), read_cells(args.twowiki)]
    rankings = [{key: marks(vals) for key, vals in data.items()} for data in datasets]

    rows = []
    for method, label in METHODS:
        cells = []
        for data, rank in zip(datasets, rankings):
            for rho in (0.0, 1.0):
                em = render_value(data[(rho, "EM")][method], rank[(rho, "EM")][method])
                f1 = render_value(data[(rho, "F1")][method], rank[(rho, "F1")][method])
                cells.append(f"{em}/{f1}")
        rows.append(f"    {label} & " + " & ".join(cells) + r" \\")

    table = "\n".join([
        r"\begin{table}[ht]", r"  \centering", r"  \small",
        r"  \caption{Answer quality (EM/F1) at the clean and heaviest injection levels, generator \texttt{Qwen3.8-27B} (a single deterministic run per dataset). Best in bold, second best underlined.}",
        r"  \label{tab:generation}", r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{lcc|cc}", r"    \hline",
        r"    & \multicolumn{2}{c|}{HotpotQA} & \multicolumn{2}{c}{2WikiMultiHopQA} \\",
        r"    \textbf{Method} & $\rho{=}0$ & $\rho{=}1$ & $\rho{=}0$ & $\rho{=}1$ \\",
        r"    \hline", *rows, r"    \hline", r"  \end{tabular}", r"\end{table}", "",
    ])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(table)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
