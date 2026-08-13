#!/usr/bin/env python3
"""Validate Paper reconstruction lineage, published-value anchors, and determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "manifests" / "paper_reconstruction"
DEFAULT_RESULTS = ROOT / "results" / "retained"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, label, tol=5e-7):
    if abs(float(actual) - expected) > tol:
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    results = args.results.resolve()
    meta = json.loads((ARCHIVE / "metadata.json").read_text())
    assert meta["scope"] == {"figures": 8, "model_execution": False, "tables": 12}
    logical_results = Path(meta["results_root"])
    assert not logical_results.is_absolute(), "results_root must be portable"
    assert len(meta["inputs"]) >= 50
    for item in meta["inputs"]:
        logical_path = Path(item["path"])
        assert not logical_path.is_absolute(), logical_path
        if logical_path.is_relative_to(logical_results):
            path = results / logical_path.relative_to(logical_results)
        else:
            path = ROOT / logical_path
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert digest(path) == item["sha256"], path
    for item in meta["outputs"]:
        path = ARCHIVE / item["path"]
        assert path.is_file(), path
        assert digest(path) == item["sha256"], path

    tables = [pd.read_csv(ARCHIVE / "tables" / f"table_{i:02d}.csv") for i in range(1, 13)]
    expected_rows = [5, 8, 5, 16, 5, 40, 40, 30, 44, 15, 6, 3]
    assert [len(t) for t in tables] == expected_rows
    assert tables[0].iloc[4]["RNG-Score(gamma=0)"] == "George V"
    close(tables[1].set_index("method").loc["MMR(0.3)", "S-Recall@k"], .481150, "Table 2 MMR(.3)")
    close(tables[1].set_index("method").loc["RNG-Score(0.2)", "Recall@k"], .750872, "Table 2 RNG")
    close(tables[2].set_index("method").loc["kNN", "scifact:NDCG@k"], .6405, "Table 3 SciFact")
    t4 = tables[3].set_index(["dataset", "method"])
    close(t4.loc[("MuSiQue", "CE+RNG-Score"), "F1"], .3431, "Table 4 MuSiQue RNG")
    close(tables[4].set_index("method").loc["kNN", "max_regret"], .2203, "Table 5 kNN regret")
    close(tables[5].query("rho == 1 and Method == 'MMR*'").iloc[0]["S-Recall@k"], .7460, "Table 6 heavy MMR")
    t7 = tables[6].set_index(["sweep", "rho"])
    assert t7.loc[("HotpotQA bge-m3", 0.0), "published_gamma"] == "+0.2"
    assert t7.loc[("HotpotQA Qwen3", .025), "published_gamma"] == "+/-"
    close(tables[7].query("dataset == 'TREC-COVID' and method == 'RNG*'").iloc[0]["SRecall_0.75"], .2725, "Table 8 TREC")
    close(tables[8].query("level == '1.0' and selector == 'rule'").iloc[0]["S-Recall@k"], .742882, "Table 9 rule")
    assert "HotpotQA bge-m3 (tuning run)" not in set(tables[9].target)
    close(tables[10].query("dataset == 'HotpotQA' and method == 'rule'").iloc[0]["rho=1:EM"], .3098, "Table 11 rule EM")
    close(tables[11].set_index("dataset").loc["MuSiQue", "RNG*"], .7171, "Table 12 MuSiQue")

    # A clean second build must be byte-identical, including PDFs.  Matplotlib
    # receives SOURCE_DATE_EPOCH in the reconstruction entry point.
    with tempfile.TemporaryDirectory(prefix="paper_reconstruction_validate_") as temp:
        env = os.environ.copy()
        subprocess.run([sys.executable, str(ROOT / "reproducibility" / "reconstruct_paper.py"), "--results", str(results), "--out", temp], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        rebuilt = Path(temp)
        current = {x["path"]: x["sha256"] for x in meta["outputs"]}
        rebuilt_meta = json.loads((rebuilt / "metadata.json").read_text())
        second = {x["path"]: x["sha256"] for x in rebuilt_meta["outputs"]}
        assert current == second, "second build differs"
    print("Paper reconstruction validation OK: 12/12 tables, 8/8 figures, all hashes and deterministic rebuild passed")


if __name__ == "__main__":
    main()
