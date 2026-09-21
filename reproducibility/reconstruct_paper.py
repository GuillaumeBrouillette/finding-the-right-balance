#!/usr/bin/env python3
"""Reconstruct the paper's numbered tables and figures from retained files.

This is a presentation-only pass: it never imports an encoder or generator.
Numerical tables are normalized to machine-readable CSV.  Figures are redrawn
from the retained analysis CSVs with the same plotting functions used by the
paper.  Every consumed and produced file is SHA-256 indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "retained"
DEFAULT_OUT = ROOT / "manifests" / "paper_reconstruction"
REVISION_STATS = ROOT / "results" / "revision_stats"
ARCHIVE_RESULTS_ROOT = Path("results/retained")
CROSSOVER_BRACKET = (0.0012, 0.0030)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SOURCE_DATE_EPOCH", "1786233600")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(results: Path, relative: str, inputs: set[Path]) -> pd.DataFrame:
    path = results / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    inputs.add(path)
    return pd.read_csv(path)


def read_revision(relative: str, inputs: set[Path]) -> pd.DataFrame:
    """Read a small, Git-tracked revision statistic and record its lineage."""
    path = REVISION_STATS / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    inputs.add(path)
    return pd.read_csv(path)


def refreshed_means(relative: str, inputs: set[Path]) -> pd.DataFrame:
    """Return corrected S-Recall means with measured pool redundancy."""
    source = read_revision(f"table_refresh/{relative}", inputs)
    coverage = source[source.metric.eq("S-Recall@k")][
        ["condition", "series", "mean", "chosen_members"]
    ].rename(columns={"mean": "S-Recall@k"})
    redundancy = source[
        source.metric.eq("PoolRedundancy") & source.series.eq("kNN")
    ][["condition", "mean"]].rename(columns={"mean": "PoolRedundancy"})
    if redundancy.empty:
        raise ValueError(
            f"{relative} lacks PoolRedundancy; regenerate table_refresh with "
            "reproducibility/run_remaining_stats.py"
        )
    return coverage.merge(redundancy, on="condition", validate="many_to_one")


def selected_member(series: str, members: object) -> str:
    """Collapse seed-labelled refresh selections to the summary-table form."""
    if not series.endswith("*"):
        return series
    prefix = series[:-1] + "("
    selected = []
    for entry in str(members).split(";"):
        member = entry.split(":", 1)[-1]
        if member.startswith(prefix) and member not in selected:
            selected.append(member)
    return ";".join(selected)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")


def portable_input_path(path: Path, results: Path) -> str:
    """Return a path relative to the published repository/archive layout."""
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    if path.is_relative_to(results):
        return (ARCHIVE_RESULTS_ROOT / path.relative_to(results)).as_posix()
    raise ValueError(f"input is outside the repository and retained-results root: {path}")


def table01(results: Path, inputs: set[Path]) -> pd.DataFrame:
    # This compact source is frozen from the retained passage export named in
    # the manuscript provenance.  It is an input artifact and is checksummed.
    path = ROOT / "reproducibility" / "table01_source.csv"
    inputs.add(path)
    return pd.read_csv(path)


def table02(results: Path, inputs: set[Path]) -> pd.DataFrame:
    # Reconstruct every fixed operating point from per-query rows on the
    # frozen test partition.  In particular, do not substitute tuned MMR* for
    # the four fixed lambdas displayed in the paper.
    pq = read(results, "2026-06-24_230733_redundancy_hotpotqa_fullwiki/results_redundancy_per_query.csv", inputs)
    split_path = ROOT / "manifests" / "splits" / "hotpotqa_fullwiki.csv"
    inputs.add(split_path)
    split = pd.read_csv(split_path)
    test = split[split.partition.eq("test")][["seed", "query_id"]].rename(columns={"query_id": "qid"})
    pq = pq[pq.rho.eq(0)].merge(test, on=["seed", "qid"], how="inner", validate="many_to_one")
    metrics = ["Recall@k", "NDCG@k", "alpha-NDCG@k", "S-Recall@k", "APD", "Vendi"]
    ret = pq.groupby(["seed", "Method"], as_index=False)[metrics].mean().groupby("Method")[metrics].mean()
    gen0 = read(results, "2026-06-05_200449_hotpotqa_fullwiki/results_generation.csv", inputs).set_index("Method")
    gen9 = read(results, "2026-07-28_152156_hotpotqa_fullwiki/results_generation.csv", inputs).set_index("Method")
    qwen_rows = read(
        results,
        "2026-09-01_191110_redundancy_hotpotqa_fullwiki/"
        "results_redundancy_gen_per_query.csv",
        inputs,
    ).merge(test, on=["seed", "qid"], how="inner", validate="many_to_one")
    expected_qwen_queries = len(test[test.seed.eq(0)])
    qwen_counts = qwen_rows.groupby("Method").size()
    expected_qwen_methods = {
        "kNN", "MMR(0.3)", "MMR(0.5)", "MMR(0.7)", "MMR(0.9)",
        "Maxmin", "Greedy-DPP", "RNG(0.2)",
    }
    if (set(qwen_counts.index) != expected_qwen_methods
            or set(qwen_counts.values) != {expected_qwen_queries}):
        raise ValueError(f"incomplete seed-0 Qwen test partition: {qwen_counts.to_dict()}")
    qwen = qwen_rows.groupby("Method")[["EM", "F1"]].mean()
    specs = [
        ("kNN", "kNN", "kNN"), ("MMR(0.3)", "MMR(0.3)", "MMR(λ=0.3)"),
        ("MMR(0.5)", "MMR(0.5)", "MMR(λ=0.5)"), ("MMR(0.7)", "MMR(0.7)", "MMR(λ=0.7)"),
        ("MMR(0.9)", "MMR(0.9)", "MMR(λ=0.9)"), ("Maxmin", "Maxmin", "Maxmin"),
        ("Greedy-DPP", "Greedy-DPP", "Greedy-DPP"),
        ("RNG-Score(0.2)", "RNG(0.2)", "RNG-Score(α=0.2)"),
    ]
    rows = []
    for label, rkey, gkey in specs:
        rr = ret.loc[rkey]
        gg = (gen9 if gkey == "MMR(λ=0.9)" else gen0).loc[gkey]
        qg = qwen.loc[rkey]
        rows.append({
            "method": label,
            **{c: rr[c] for c in ["Recall@k", "NDCG@k", "alpha-NDCG@k", "S-Recall@k", "APD", "Vendi"]},
            "FLAN_EM": gg.EM, "FLAN_F1": gg.F1,
            "Qwen_EM": qg.EM, "Qwen_F1": qg.F1,
        })
    return pd.DataFrame(rows)


def table03(results: Path, inputs: set[Path]) -> pd.DataFrame:
    new = read(results, "2026-06-22_211522_beir_scifact_fiqa_trec-covid/results_beir_summary.csv", inputs)
    old = read(results, "2026-06-08_053252_beir_scifact_fiqa_trec-covid_arguana_webis-touche2020/results_beir_summary.csv", inputs)
    tasks = ["scifact", "fiqa", "trec-covid", "arguana", "webis-touche2020"]
    methods = [("kNN", "kNN"), ("MMR", "MMR(λ=0.5)"), ("Maxmin", "Maxmin"), ("Greedy-DPP", "Greedy-DPP"), ("RNG-Score*", "RNG-Score")]
    rows = []
    for label, key in methods:
        row = {"method": label}
        for task in tasks:
            d = new if task in {"scifact", "fiqa", "trec-covid"} else old
            cur = d[d.Task.eq(task)]
            if "MethodKey" in cur and key == "RNG-Score": hit = cur[cur.MethodKey.eq(key)]
            elif "MethodKey" in cur: hit = cur[cur.MethodKey.eq(key)]
            else:
                pattern = "RNG-Score" if key == "RNG-Score" else key
                hit = cur[cur.Method.str.startswith(pattern, na=False)]
            if hit.empty: raise ValueError(f"Table 3 missing {task}/{key}")
            hit = hit.iloc[0]
            row[f"{task}:NDCG@k"] = hit["NDCG@k"]
            row[f"{task}:alpha-NDCG@k"] = hit["alpha-NDCG@k"]
        rows.append(row)
    return pd.DataFrame(rows)


def table04(results: Path, inputs: set[Path]) -> pd.DataFrame:
    specs = [
        ("HotpotQA", "2026-06-24_064052_ce_hotpotqa_fullwiki/results_ce_hotpotqa_fullwiki_summary.csv"),
        ("2WikiMultiHopQA", "2026-06-05_102637_ce_2wikimultihopqa/results_ce_2wikimultihopqa_summary.csv"),
        ("MuSiQue", "2026-06-04_124043_ce_musique/results_ce_musique_summary.csv"),
        ("NQ-Open", "2026-06-06_205350_ce_nq/results_ce_nq_summary.csv"),
    ]
    rows = []
    for dataset, path in specs:
        d = read(results, path, inputs)
        keycol = "MethodKey" if "MethodKey" in d else "Method"
        for label, key in [("CE top-k", "CE-topk"), ("CE+MMR", "CE-MMR(λ=0.7)"), ("CE+DPP", "CE-DPP"), ("CE+RNG-Score", "S2-V1")]:
            hit = d[d[keycol].eq(key)]
            if hit.empty and key == "S2-V1":
                hit = d[d.Method.str.match(r"^S2-V1.*\[ndcg\]$", na=False)]
            if hit.empty: raise ValueError(f"Table 4 missing {dataset}/{key}")
            h = hit.iloc[0]
            rows.append({"dataset": dataset, "method": label, **{c: h[c] for c in ["Recall@k", "NDCG@k", "EM", "F1"]}})
    return pd.DataFrame(rows)


def table05(results: Path, inputs: set[Path]) -> pd.DataFrame:
    d = read(results, "2026-06-24_230733_redundancy_hotpotqa_fullwiki/analysis_regret.csv", inputs)
    minimax = read_revision("minimax_regret/minimax_regret.csv", inputs)
    minimax = minimax.set_index("family")["minimax_regret_pp"].div(100)
    rows = []
    for family in ["MMR", "Maxmin", "Greedy-DPP", "RNG"]:
        a, b = d[(d.Family.eq(family)) & d.rho.eq(0)].iloc[0], d[(d.Family.eq(family)) & d.rho.eq(1)].iloc[0]
        minimax_family = family + "*" if family in {"MMR", "RNG"} else family
        rows.append({"method": family, "clean_best": a.best, "clean_worst": a.worst, "heavy_best": b.best, "heavy_worst": b.worst, "max_downside": max(a["regret(worst)"], b["regret(worst)"]), "minimax_regret": minimax.loc[minimax_family]})
    knn = d[d.rho.isin([0, 1])].groupby("rho").kNN.first()
    kreg = max(float(d[d.rho.eq(r)].best.max() - knn.loc[r]) for r in [0, 1])
    rows.insert(0, {"method": "kNN", "clean_best": knn.loc[0], "clean_worst": knn.loc[0], "heavy_best": knn.loc[1], "heavy_worst": knn.loc[1], "max_downside": kreg, "minimax_regret": minimax.loc["kNN"]})
    return pd.DataFrame(rows)


def table06(results: Path, inputs: set[Path]) -> pd.DataFrame:
    d = refreshed_means(
        "2026-08-07_230907_redundancy_hotpotqa_fullwiki__redundancy_level_means.csv",
        inputs,
    ).rename(columns={"condition": "rho", "series": "Method"})
    d["Chosen"] = [selected_member(series, members) for series, members in
                   zip(d.Method, d.chosen_members)]
    methods = ["kNN", "MMR*", "Maxmin", "Greedy-DPP", "RNG*"]
    d["Method"] = pd.Categorical(d.Method, categories=methods, ordered=True)
    return d.sort_values(["rho", "Method"])[
        ["rho", "PoolRedundancy", "Method", "Chosen", "S-Recall@k"]
    ].reset_index(drop=True)


def table07(results: Path, inputs: set[Path]) -> pd.DataFrame:
    specs = [
        ("HotpotQA bge-m3", "2026-06-24_230733_redundancy_hotpotqa_fullwiki/analysis_summary_per_seed.csv"),
        ("HotpotQA Qwen3", "2026-06-25_172848_redundancy_hotpotqa_fullwiki/analysis_summary_per_seed.csv"),
        ("HotpotQA MiniLM", "2026-06-28_094117_redundancy_hotpotqa_fullwiki/analysis_summary_per_seed.csv"),
        ("SciFact", "2026-06-25_082550_redundancy_scifact/analysis_summary_per_seed.csv"),
        ("FiQA-2018", "2026-07-23_104202_redundancy_fiqa/results_redundancy_per_seed_summary.csv"),
    ]
    rows = []
    for sweep, path in specs:
        d = read(results, path, inputs)
        for rho, g in d[d.Method.eq("RNG*")].groupby("rho", sort=True):
            chosen = list(g.sort_values("seed").Chosen.astype(str))
            vals = [float(x[x.find("(")+1:x.rfind(")")]) for x in chosen]
            signs = {v >= 0 for v in vals}
            if len(signs) > 1:
                display = "+/-"
            else:
                display = f"{max(set(vals), key=vals.count):+g}"
            rows.append({"sweep": sweep, "rho": rho, "published_gamma": display, "chosen_per_seed": ";".join(chosen)})
    return pd.DataFrame(rows)


def table08(results: Path, inputs: set[Path]) -> pd.DataFrame:
    specs = [
        ("HotpotQA", "2026-08-07_113533_redundancy_hotpotqa_fullwiki__chunking_level_means.csv"),
        ("2Wiki", "2026-08-07_031430_redundancy_2wikimultihopqa__chunking_level_means.csv"),
        ("MuSiQue", "2026-08-06_164145_redundancy_musique__chunking_level_means.csv"),
        ("SciFact", "2026-08-07_072703_redundancy_scifact__chunking_level_means.csv"),
        ("NQ-Open", "2026-08-08_070150_redundancy_nq__chunking_level_means.csv"),
        ("TREC-COVID", "2026-08-06_170200_redundancy_trec-covid__chunking_level_means.csv"),
    ]
    rows=[]
    for dataset,path in specs:
        d=refreshed_means(path,inputs)
        for method in ["kNN","Maxmin","Greedy-DPP","MMR*","RNG*"]:
            g=d[d.series.eq(method)].set_index("condition")
            rows.append({"dataset":dataset,"method":method,"redundancy_0.75":g.loc[0.75,"PoolRedundancy"],"SRecall_0":g.loc[0,"S-Recall@k"],"SRecall_0.75":g.loc[0.75,"S-Recall@k"]})
    return pd.DataFrame(rows)


def table09(results: Path, inputs: set[Path]) -> pd.DataFrame:
    pq=read(results,"2026-06-24_230733_redundancy_hotpotqa_fullwiki/results_redundancy_per_query.csv",inputs)
    split_path=ROOT/"manifests"/"splits"/"hotpotqa_fullwiki.csv"; inputs.add(split_path)
    split=pd.read_csv(split_path); test=split[split.partition.eq("test")][["seed","query_id"]].rename(columns={"query_id":"qid"})
    pq=pq.merge(test,on=["seed","qid"],how="inner",validate="many_to_one")
    keep=pq[pq.Method.isin(["kNN","MMR(0.7)"])].pivot(index=["seed","rho","qid"],columns="Method",values=["S-Recall@k","Vendi"])
    per_seed=[]
    for (seed,rho),g in keep.groupby(level=["seed","rho"]):
        fire=g[("Vendi","kNN")].lt(2.0)
        rule=g[("S-Recall@k","MMR(0.7)")].where(fire,g[("S-Recall@k","kNN")])
        per_seed.append({"seed":seed,"level":rho,"kNN":g[("S-Recall@k","kNN")].mean(),"D":g[("S-Recall@k","MMR(0.7)")].mean(),"rule":rule.mean(),"trigger_rate":fire.mean()})
    t=pd.DataFrame(per_seed).groupby("level",as_index=False)[["kNN","D","rule","trigger_rate"]].mean()
    pooled={"level":"pooled"}
    for col in ["kNN","D","rule","trigger_rate"]: pooled[col]=t[col].mean()
    t=pd.concat([t,pd.DataFrame([pooled])],ignore_index=True)
    threshold=read(results,"2026-06-24_230733_redundancy_hotpotqa_fullwiki/analysis_threshold.csv",inputs)
    rows=[]
    for _,r in t.iterrows():
        for selector,col in [("always kNN","kNN"),("always D*","D"),("rule","rule")]: rows.append({"level":r.level,"selector":selector,"S-Recall@k":r[col],"trigger_rate":r.trigger_rate if selector=="rule" else None})
    for selector in ["per-level tuned","all-methods oracle"]:
        for _,r in threshold[threshold.Selector.eq(selector)].iterrows(): rows.append({"level":r.rho,"selector":selector,"S-Recall@k":r["S-Recall@k"],"trigger_rate":None})
    return pd.DataFrame(rows)


def table10(results: Path, inputs: set[Path]) -> pd.DataFrame:
    d=read(results,"frozen_rule_transfer_summary.csv",inputs)
    return d[d.level.astype(str).eq("pooled") & ~d.target.eq("HotpotQA bge-m3 (tuning run)")][["target","cleanDelta","heavyDelta","pooledDelta"]].reset_index(drop=True)


def table11(results: Path, inputs: set[Path]) -> pd.DataFrame:
    specs=[
        ("HotpotQA", "qwen38_legacy_unfixed/2026-08-31_163327_redundancy_hotpotqa_fullwiki/analysis_generation.csv"),
        ("2WikiMultiHopQA", "qwen38_legacy_unfixed/2026-08-31_232318_redundancy_2wikimultihopqa/analysis_generation.csv"),
    ]
    rows=[]
    for dataset,path in specs:
        d=read(results,path,inputs)
        d=d[d.rho.isin([0,1])]
        dcol=next(c for c in d.columns if c not in {"rho","metric","n","kNN","rule","trigger rate"} and not c.startswith("p("))
        for method,col in [("kNN","kNN"),("MMR(0.7)",dcol),("rule","rule")]:
            row={"dataset":dataset,"method":method}
            for rho in [0,1]:
                for metric in ["EM","F1"]: row[f"rho={rho}:{metric}"]=d[(d.rho.eq(rho)) & d.metric.eq(metric)].iloc[0][col]
            rows.append(row)
    return pd.DataFrame(rows)


def table12(results: Path, inputs: set[Path]) -> pd.DataFrame:
    specs=[("2WikiMultiHopQA","2026-06-15_144227_redundancy_2wikimultihopqa/analysis_summary.csv"),("MuSiQue","2026-06-15_142100_redundancy_musique/analysis_summary.csv"),("NQ-Open","2026-06-16_094433_redundancy_nq/analysis_summary.csv")]
    rows=[]
    for dataset,path in specs:
        d=read(results,path,inputs); lvl=d.columns[0]; d=d[d[lvl].eq(0)].set_index("Method")
        row={"dataset":dataset,"PoolRedundancy":d.loc["kNN","PoolRedundancy"]}
        for method in ["kNN","Maxmin","Greedy-DPP","MMR*","RNG*"]: row[method]=d.loc[method,"S-Recall@k"]
        rows.append(row)
    return pd.DataFrame(rows)


TABLES=[table01,table02,table03,table04,table05,table06,table07,table08,table09,table10,table11,table12]


def load_plot_module():
    from analysis import plot_regimes

    return plot_regimes


def make_figures(results: Path, out: Path, inputs: set[Path]) -> list[Path]:
    plots=load_plot_module(); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    def p(run,name):
        path=results/run/name; inputs.add(path); return str(path)
    generated=[]
    # Figures 2 and 3 are deterministic geometric illustrations.
    from analysis import plot_geometry
    plot_geometry.emit_rng_graph(str(figdir/"figure02_rng_graph.tex"),seed=7)
    (figdir/"figure02_lune.tex").write_text(
        "\\begin{tikzpicture}[scale=0.85]\n"
        "  \\clip (-1.2,-2.8) rectangle (4.2,2.8);\n"
        "  \\begin{scope} \\clip (0,0) circle (3); \\fill[blue!12] (3,0) circle (3); \\end{scope}\n"
        "  \\draw[gray] (0,0) circle (3); \\draw[gray] (3,0) circle (3);\n"
        "  \\node[circle,fill=black,inner sep=2pt,label=below left:$x$] at (0,0) {};\n"
        "  \\node[circle,fill=black,inner sep=2pt,label=below right:$y$] at (3,0) {};\n"
        "  \\node[circle,fill=blue,inner sep=2pt,label=above right:$z_1$] at (1.5,1.2) {};\n"
        "  \\node[circle,draw,inner sep=2pt,label=above:$z_2$] at (0.5,-2.2) {};\n"
        "\\end{tikzpicture}\n", encoding="utf-8")
    plot_geometry.emit_knn_rng_mmr(str(figdir/"figure03_selections.tex"))
    generated += [figdir/"figure02_lune.tex",figdir/"figure02_rng_graph.tex",figdir/"figure03_selections.tex"]
    plots.crossover_figure([("bge-m3",p("2026-06-24_230733_redundancy_hotpotqa_fullwiki","analysis_summary.csv")),("Qwen3",p("2026-06-25_172848_redundancy_hotpotqa_fullwiki","analysis_summary.csv")),("MiniLM",p("2026-06-28_094117_redundancy_hotpotqa_fullwiki","analysis_summary.csv"))],str(figdir/"figure04.pdf"))
    generated += sorted(figdir.glob("figure04*.pdf"))
    plots.crossover_figure([("SciFact",p("2026-06-25_082550_redundancy_scifact","analysis_summary.csv")),("FiQA",p("2026-07-23_104202_redundancy_fiqa","results_redundancy_summary.csv"))],str(figdir/"figure05.pdf"))
    generated += sorted(figdir.glob("figure05*.pdf"))
    colors=["#005AB5","#DC3220","#5D3A9B","black","#40B0A6","#E66100"]; marks=["o","s","D","v","P","^"]
    inj=[("HotpotQA","2026-06-24_230733_redundancy_hotpotqa_fullwiki","analysis_oracle.csv","analysis_summary.csv"),("SciFact","2026-06-25_082550_redundancy_scifact","analysis_oracle.csv","analysis_summary.csv"),("FiQA","2026-07-23_104202_redundancy_fiqa","analysis_oracle.csv","results_redundancy_summary.csv"),("TREC-COVID","2026-07-28_115804_redundancy_trec-covid","analysis_oracle.csv","results_redundancy_summary.csv"),("2Wiki","2026-07-28_170313_redundancy_2wikimultihopqa","analysis_oracle.csv","results_redundancy_summary.csv"),("MuSiQue","2026-07-28_154431_redundancy_musique","analysis_oracle.csv","results_redundancy_summary.csv")]
    chunks=[("HotpotQA","2026-06-24_083656_redundancy_hotpotqa_fullwiki","analysis_oracle.csv","analysis_summary.csv"),("SciFact","2026-06-25_082550_redundancy_scifact","analysis_chunking_oracle.csv","analysis_chunking_summary.csv"),("FiQA","2026-07-28_095742_redundancy_fiqa","analysis_oracle.csv","results_chunking_summary.csv"),("TREC-COVID","2026-07-16_095435_redundancy_trec-covid","analysis_oracle.csv","results_chunking_summary.csv"),("2Wiki","2026-07-23_085711_redundancy_2wikimultihopqa","analysis_oracle.csv","analysis_summary.csv"),("MuSiQue","2026-07-23_101022_redundancy_musique","analysis_oracle.csv","results_chunking_summary.csv")]
    for suffix,specs,axis in [("a",inj,"level"),("b",chunks,"level")]:
        panels=[(lab,p(run,ora),p(run,summ),colors[i],"-",marks[i]) for i,(lab,run,ora,summ) in enumerate(specs)]
        plots.oracle_figure(panels,str(figdir/f"figure06_{suffix}.pdf"),x_axis=axis); generated.append(figdir/f"figure06_{suffix}.pdf")
    run=results/"2026-06-24_230733_redundancy_hotpotqa_fullwiki"; inputs.update({run/"analysis_threshold.csv",run/"analysis_threshold_curve.csv",run/"analysis_threshold_info.json",results/"frozen_rule_transfer_summary.csv"})
    plots.threshold_figure(str(run),str(figdir/"figure07.pdf"),frozen_csv=str(results/"frozen_rule_transfer_summary.csv")); generated += sorted(figdir.glob("figure07*.pdf"))
    gruns=[("HotpotQA",str(results/"2026-06-18_173800_redundancy_hotpotqa_fullwiki")),("2Wiki",str(results/"2026-06-18_121117_redundancy_2wikimultihopqa"))]
    for _,d in gruns: inputs.add(Path(d)/"analysis_generation_frozen.csv")
    plots.generation_figure(gruns,str(figdir/"figure08.pdf"),metrics=("EM",)); generated += sorted(figdir.glob("figure08*.pdf"))
    # Figure 1: three policies on measured redundancy, with the fixed-pool
    # rerun's sign-change bracket rather than an interpolated point estimate.
    gate_path=results/"2026-06-24_230733_redundancy_hotpotqa_fullwiki"/"analysis_gate.csv"
    summ_path=results/"2026-06-24_230733_redundancy_hotpotqa_fullwiki"/"analysis_summary.csv"
    frozen_path=results/"frozen_rule_transfer_summary.csv"; inputs.update({gate_path,summ_path,frozen_path})
    plots.regimes_summary_figure(
        str(gate_path.parent), str(figdir/"figure01.pdf"),
        frozen_csv=str(frozen_path), crossover=CROSSOVER_BRACKET,
    )
    generated.append(figdir/"figure01.pdf")
    return sorted(set(generated))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--results",type=Path,default=DEFAULT_RESULTS); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); ap.add_argument("--skip-figures",action="store_true"); ap.add_argument("--reuse-existing-figures",action="store_true",help="Rebuild tables and metadata while retaining already validated figure files and their recorded inputs."); args=ap.parse_args()
    results=args.results.resolve(); out=args.out.resolve(); tables=out/"tables"; tables.mkdir(parents=True,exist_ok=True); inputs:set[Path]=set(); outputs=[]
    for i,builder in enumerate(TABLES,1):
        path=tables/f"table_{i:02d}.csv"; write_table(builder(results,inputs),path); outputs.append(path)
    if args.reuse_existing_figures:
        old_meta_path = out / "metadata.json"
        if not old_meta_path.is_file():
            raise FileNotFoundError("--reuse-existing-figures requires existing metadata.json")
        old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
        for item in old_meta["inputs"]:
            logical = Path(item["path"])
            path = (results / logical.relative_to(ARCHIVE_RESULTS_ROOT)
                    if logical.is_relative_to(ARCHIVE_RESULTS_ROOT)
                    else ROOT / logical)
            inputs.add(path)
        outputs += [
            out / item["path"] for item in old_meta["outputs"]
            if item["path"].startswith("figures/")
        ]
    elif not args.skip_figures:
        outputs += make_figures(results,out,inputs)
    metadata={"schema_version":1,"status":"complete","scope":{"tables":12,"figures":8,"model_execution":False},"figure_parameters":{"figure01":{"crossover_bracket":list(CROSSOVER_BRACKET),"metric_scale":100.0}},"rendering_note":"All numerical cells and curves are rebuilt from retained files and committed revision statistics. Figures 2 and 3 are deterministic, format-equivalent TikZ geometry components; final subfigure composition remains a LaTeX presentation step.","results_root":ARCHIVE_RESULTS_ROOT.as_posix(),"inputs":[{"path":portable_input_path(p,results),"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(inputs)],"outputs":[{"path":str(p.relative_to(out)),"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(set(outputs))]}
    meta=out/"metadata.json"; meta.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    sums=out/"SHA256SUMS"; sums.write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in sorted(set(outputs)|{meta})),encoding="utf-8")
    print(f"Paper reconstructed: 12 tables, 8 figures; {len(inputs)} checksummed inputs; no model execution")


if __name__=="__main__": main()
