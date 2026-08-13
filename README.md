# Finding the Right Balance: Relevance and Diversity in LLM Retrieval

Code for the paper **"Finding the Right Balance: Relevance and Diversity in
LLM Retrieval"** (Guillaume Brouillette and Faustin Kagabo).
It measures the geometric **RNG-Score** reranker against
diversification baselines (MMR, Maxmin, Greedy-DPP, cross-encoder pipelines)
and, more centrally, characterises *when* diversification helps: redundancy
injection and chunk-overlap sweeps, per-query oracle headroom, and a label-free
decision rule with a relevant-set-size gate.

Smoke tests run on a laptop CPU; full-scale experiments can use a GPU through
`--device cuda`. The complete artifact and validation workflow is documented
in the [reproducibility package](#reproducibility-package).

## Reproducibility package

The article's reproducibility materials are under
[`reproducibility/`](reproducibility/). Start with:

- [`README.md`](reproducibility/README.md) — artifact, rebuild, and validation
  workflow;
- [`COVERAGE.md`](reproducibility/COVERAGE.md) — evidence coverage and known
  archival limitations;
- [`METHODS.md`](reproducibility/METHODS.md) — detailed technical protocols;
  and
- [`manifests/`](manifests/) — frozen identifiers, provenance, statistical
  evidence, reconstructed tables/figures, and integrity hashes.

Statistical analyses, tables, and figures can be reproduced from the archived
results without rerunning the model experiments.

Large evidence and retained-result files are distributed as versioned assets
in the [reproducibility release](https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/tag/reproducibility-v1).
Download, checksum, and extraction commands are in the
[`reproducibility/README.md`](reproducibility/README.md).

---

## Architecture

```
.
├── evaluate.py               ← retrieval + generation table (RQ1/RQ2; bootstrap CIs + Wilcoxon)
├── evaluate_redundancy.py    ← redundancy-injection / chunk-overlap / oracle sweeps (--seeds)
├── analyze_regimes.py        ← derived analyses from a sweep's per-query CSV: tuned summary,
│                               regret, oracle headroom, decision rule, crossover CI, gate
├── plot_regimes.py           ← paper figures (alpha sweep, threshold, generation)
├── evaluate_beir.py          ← BEIR generalisation table (RQ3; --seeds)
├── evaluate_cross_encoder.py ← cross-encoder integration table (RQ4; --seeds)
├── efficiency.py             ← timing benchmarks + O(m²D) scaling fit (RQ5)
├── stats.py                  ← shared CIs (across-seed Student-t + across-query bootstrap),
│                               Wilcoxon, Holm–Bonferroni, --seeds plumbing
├── learn_alpha.py            ← val-set α* / β* selection (all settings)
├── alpha_selection.py        ← shared hyper-parameter selection utilities
├── config.yaml · requirements.txt
├── data/
│   └── loaders.py            ← HotpotQA · HotpotQA-fullwiki · 2WikiMultiHopQA · MuSiQue ·
│                               SQuAD · NQ-Open (DPR) · BEIR (SciFact/FiQA/TREC-COVID)
├── retrieval/
│   ├── retriever.py          ← SentenceTransformers + FAISS dense retriever
│   ├── rerankers.py          ← kNN · MMR · Maxmin · Greedy-DPP ·
│   │                            RNG-Score · Seg-Score ·
│   │                            CE-topk · CE-MMR · CE-DPP · S1-Blend · S2-Semimetric
│   └── cross_encoder.py      ← CrossEncoderReranker wrapper
├── generation/
│   └── generator.py          ← Flan-T5 (seq2seq) + causal-LM (Llama/Qwen) readers
└── evaluation/
    └── metrics.py            ← EM · F1 · Recall · NDCG · MRR ·
                                APD · Vendi ·
                                alpha-NDCG · S-Recall · ERR-IA
```

---

## Installation

```bash
pip install -r requirements.txt
```

Datasets are not shipped with this repository. The loaders download what they
need on first use: HotpotQA / 2WikiMultiHopQA / MuSiQue / SQuAD / NQ-Open via
HuggingFace `datasets`, and the BEIR collections (SciFact, FiQA, TREC-COVID,
ArguAna, Webis-Touché2020) from the official BEIR mirrors (see `data/beir.py`).
MovieLens archives, if ever needed, are available from
<https://grouplens.org/datasets/movielens/>.

Python ≥ 3.9 recommended.  All dependencies are CPU-only by default.

---

## Seeds and confidence intervals

Every comparison table can carry a 95% interval and a significance marker.
The helpers live in `stats.py` and are wired into all four evaluation
scripts, so the reported numbers no longer rest on a single point estimate.

Two complementary notions of uncertainty are reported, picked per script by
where the randomness actually is:

| Source of variance | Reported as | Where it applies |
|---|---|---|
| **Seed** — which near-duplicates are injected; the random val/test split that tunes α*/λ* | Student-t **across-seed CI** (`<metric>_ci95` half-width) + across-seed median Wilcoxon p | `evaluate_redundancy.py`, `evaluate_beir.py`, `evaluate_cross_encoder.py`, `analyze_regimes.py` |
| **Query sample** — a deterministic method on a finite query set | percentile **across-query bootstrap CI** (`ci95_lo`/`ci95_hi`) + paired Wilcoxon p | `evaluate.py` (deterministic encoder + beam reader) |

**Turnkey 3-seed protocol.** Pass `--seeds 0 1 2` to any of the three
seed-aware scripts (they also keep a single-run `--seed` default for
back-compat). The whole experiment is repeated once per seed; the headline
`results_*_summary.csv` becomes an across-seed mean ± 95% CI with median
Wilcoxon p vs the baseline, and a `results_*_per_seed_summary.csv` keeps the
raw per-seed rows. `analyze_regimes.py` auto-detects the `seed` column in a
multi-seed per-query file and aggregates every derived table (tuned summary,
regret, oracle, decision rule, generation) plus an explicit **crossover
location with a 95% CI** (`analysis_crossover.csv`).

Holm–Bonferroni adjustment for a family of comparisons is available
(`stats.holm_bonferroni`). The archived paired analysis in
[`METHODS.md`](reproducibility/METHODS.md) publishes both
raw and Holm-adjusted p-values using the corrected query-clustered design.

---

## RQ1/RQ2 – Retrieval trade-off and downstream generation

```bash
# Quick smoke-test (100 HotpotQA examples, no generation)
python evaluate.py --max_samples 100 --no_generation

# Full run with generation
python evaluate.py --max_samples 200

# Strong encoder + modern reader via aliases (no config edit needed)
python evaluate.py --max_samples all --device cuda \
    --encoder_model bge-m3 --generator_model qwen3-8b --save_per_query
```

Outputs in `results/`:
- `results_retrieval.csv` / `results_generation.csv` — per-method means.
- `results_retrieval_significance.csv` / `results_generation_significance.csv`
  — one row per (method, metric) with the mean, a **95% across-query
  bootstrap CI** (`ci95_lo`/`ci95_hi`), and the **paired Wilcoxon p-value vs
  kNN** (`p_vs_kNN`). Retrieval and beam-search generation are deterministic
  given the encoder/reader, so the honest error bar here is over the query
  sample, not over a re-seed. (For *seed* variance on generation under
  redundancy, use `evaluate_redundancy.py --run_generation --seeds …`, below.)
- `*_per_query.csv` (with `--save_per_query`) for custom paired tests.

### Metrics reported

| Family | Metrics |
|---|---|
| Relevance | Recall@k, NDCG@k, MRR |
| Intent-aware diversity | alpha-NDCG@k (α_r=0.5), S-Recall@k, ERR-IA@k |
| Annotation-free diversity | Avg Pairwise Distance, Vendi Score |
| Generation | EM, F1, Hallucination Rate |

Intent-aware metrics use the two gold supporting documents of each HotpotQA
question as the subtopics.

---

## RQ3 – BEIR generalisation

```bash
# All three tasks (downloads ~300 MB on first run)
python evaluate_beir.py --max_queries 50

# Single task
python evaluate_beir.py --tasks scifact --max_queries 100

# Cache FAISS indices to avoid rebuilding
python evaluate_beir.py --tasks scifact fiqa \
    --index_dir results/indices --max_queries 100

# 3-seed protocol with across-seed CIs and Wilcoxon vs kNN.
# --index_dir caches each task's index so seeds 1,2 skip the re-encode.
python evaluate_beir.py --tasks scifact --max_queries all --seeds 0 1 2 \
    --objective alpha_ndcg --index_dir results/indices
```

Tasks: **SciFact** (~5k docs), **FiQA** (~57k docs), **TREC-COVID** (~171k docs).

Outputs in `results/`:
- `results_beir_summary.csv` — single run: per-method test means. Multi-seed
  (`--seeds`): one row per (Task, method family) with across-seed mean ±
  95% CI on every metric, the per-seed tuned α* (`alpha*(per seed)`), and the
  across-seed median Wilcoxon p vs kNN on α-NDCG@k and S-Recall@k.
- `results_beir_per_seed_summary.csv` — raw per-seed rows (multi-seed only).
- `results_beir_per_query.csv` (with `--save_per_query`; carries a `seed` column).

The validation-optimal margin α* is selected on a random 20% split of the
available queries (seed = each `--seed`/`--seeds` value) by **maximising
`--objective` on that split**. The default is `alpha_ndcg` (intent-aware
coverage, the metric the BEIR table reports); pass `s_recall` for parity with
the QA tuning, or `ndcg` for pure relevance. `apd`/`vendi` tune for raw
geometric spread (intrinsic credit) and are **not** appropriate for this
instrumental-credit table — selecting a diversifier's operating point by its
diversity defeats the purpose of the comparison. The chosen α* per seed is
reported in the summary (`alpha*(per seed)`).

---

## RQ4 – Cross-encoder pipeline (no training)

```bash
# Quick test: 100 HotpotQA examples, lightweight models
python evaluate_cross_encoder.py --max_samples 100

# SQuAD (also has attached passages)
python evaluate_cross_encoder.py --dataset squad --max_samples 100

# With generation
python evaluate_cross_encoder.py --max_samples 50

# 3-seed protocol: across-seed CIs + Wilcoxon vs CE-topk
python evaluate_cross_encoder.py --max_samples all --seeds 0 1 2 \
    --objective alpha_ndcg
```

### Integration strategies

| Strategy | Description |
|---|---|
| **CE-topk** | Cross-encoder top-k (no diversification baseline) |
| **CE-MMR** | CE relevance + embedding-based MMR |
| **CE-DPP** | CE relevance + greedy MAP-DPP |
| **S1-Blend(α, β)** | Blend CE score with squashed RNG-Score |
| **S2-V1/V2/V3(α)** | RNG-Score in CE-induced semimetric, three d_CE(v,w) variants |

The cross-encoder rescores the pool with the device default (`minilm-ce` =
`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~85 MB, on CPU; `bge-reranker-v2-m3` on
CUDA); override with `--ce_model`. No training is performed; every strategy is
inference-only.

The S1/S2 strategies' α* (and β* for S1) are selected on the 20% validation
split by **maximising `--objective`**, same as RQ3: default `alpha_ndcg`,
`s_recall` for QA parity, or `em`/`f1` to tune on answer quality (requires
generation). `apd`/`vendi` are intrinsic-credit (raw spread) and not
appropriate for this table.

Outputs in `results/`:
- `results_ce_<dataset>_summary.csv` — single run: per-strategy test means.
  Multi-seed (`--seeds`): across-seed mean ± 95% CI per strategy with the
  across-seed median Wilcoxon p vs **CE-topk** on α-NDCG@k, S-Recall@k, and
  (with generation) EM/F1.
- `results_ce_<dataset>_per_seed_summary.csv` (multi-seed only).
- `results_ce_<dataset>_retrieval_per_query.csv` (only with `--save_per_query`; `seed` column)
- `results_ce_<dataset>_generation_per_query.csv` (only with `--save_per_query` and generation enabled)

---

## Redundancy injection and oracle headroom (`evaluate_redundancy.py`)

Two controlled experiments that test *when* diversification pays off:

1. **Redundancy injection** — interpolates between deduplicated benchmark
   pools and production-style chunked pools by injecting a fraction ρ of
   near-duplicate passages (sentence-shuffled copies that keep the source
   title, so coverage metrics treat them as redundant).  Every reranker is
   evaluated at each ρ; the summary reports the measured pool redundancy
   and Wilcoxon p-values against kNN.
2. **Per-query oracle headroom** — sweeps α (RNG/Seg) and λ (MMR) per query
   and reports the gap between kNN, the validation-tuned fixed parameter,
   and the per-query oracle (DF-RAG-style upper bound on any query-adaptive
   diversification policy).

```bash
# Quick smoke test (CPU)
python evaluate_redundancy.py --max_samples 100 --experiment both

# Redundancy sweep on the MDR fullwiki pools
python evaluate_redundancy.py --dataset hotpotqa_fullwiki \
    --encoder_model bge-m3 --device cuda --experiment redundancy

# 3-seed protocol: across-seed CIs on the crossover and every metric
python evaluate_redundancy.py --dataset hotpotqa_fullwiki \
    --encoder_model bge-m3 --device cuda --experiment redundancy \
    --max_samples all --seeds 0 1 2 --objective S-Recall@k
```

With `--seeds`, the whole sweep is repeated per seed (for injection the seed
controls *which* near-duplicates land in each pool; for chunking the pools are
deterministic and only the split moves), and the summary is aggregated to an
across-seed mean ± 95% CI.

Outputs in `results/<timestamp>_redundancy_<dataset>/`:
- `results_redundancy_summary.csv` — across-seed mean ± 95% CI per (level,
  method family) with the tuned member (`Chosen`) and across-seed
  median/max Wilcoxon p vs kNN.
- `results_redundancy_per_seed_summary.csv` — raw per-seed test summaries.
- `results_redundancy_per_query.csv` — per-query metrics with a `seed` column
  (consumed by `analyze_regimes.py`).
- `results_oracle_summary.csv` (across-seed CI on headroom) /
  `results_oracle_per_seed_summary.csv` / `results_oracle_per_query.csv`.
- `results_redundancy_gen_per_query.csv` — per-query EM/F1/Halluc (with
  `--run_generation`; `seed` column), the loop-closer for answer quality.

---

## Derived analyses (`analyze_regimes.py`, `plot_regimes.py`)

`analyze_regimes.py` consumes a `results_*_per_query.csv` from
`evaluate_redundancy.py` and produces, without re-running any retrieval:
the validation-tuned summary under any objective, the mis-specification
regret table (paper RQ3), the per-family and all-methods oracle headroom
(paper RQ5), and the redundancy-threshold decision rule (paper RQ6) —
trigger = Vendi score of the kNN top-k, threshold tau and fallback
diversifier D tuned jointly on pooled validation queries.
`plot_regimes.py` renders the paper figures (`fig_alpha_sweep.pdf`,
`fig_threshold.pdf`) from those outputs.

`analyze_regimes.py` also evaluates the **relevant-set-size gate** on the
tuned rule (`analysis_gate.csv` + `analysis_gate_info.json`): the rule
diversifies only on queries with at least `--gate_min_rel` distinct relevant
subtopics (default 2), so it never fires on single-hop queries (one relevant
item), where reranking for coverage can only displace the answer — the
mis-fire the rule shows on single-hop synthetic injection. The table reports,
per level and pooled, kNN vs the ungated rule vs the gated rule with the
trigger/gate firing rates and the recovery (`gated_minus_kNN`,
`gated_minus_ungated`). The gate uses the gold relevant-set size, so it
validates the gate's *design*; a deployment would estimate the hop count from
the query. Needs `RelSetSize` in the per-query file (logged automatically by
`evaluate_redundancy.py`; older runs skip the gate with a notice).

When the per-query file carries a `seed` column (a `--seeds` run),
`analyze_regimes.py` runs every analysis once per seed — each on its own
val/test split — and aggregates to an across-seed mean with a 95% CI
(`<metric>_ci95` columns), writing the raw per-seed tuned summary alongside
(`analysis_summary_per_seed.csv`). It also emits `analysis_crossover.csv`:
the redundancy level at which the best tuned diversifier overtakes kNN, with
its per-seed values, across-seed mean, and 95% CI half-width — the
crossover-location interval the review asks for. The decision-rule
`analysis_threshold_info.json` gains `tau_ci95` and a modal/joined fallback
`D` across seeds.

```bash
python analyze_regimes.py \
    --per_query ../results/<run>/results_redundancy_per_query.csv \
    --objective S-Recall@k --top_k 5
python plot_regimes.py --run_dir ../results/<run> --out_dir ../plots

# Transfer experiment: evaluate a rule frozen on another sweep
python analyze_regimes.py --per_query .../results_chunking_per_query.csv \
    --level_col overlap --objective S-Recall@k \
    --freeze_tau 2.31 --freeze_d "MMR(0.7)"
```

Current frozen rule (tuned on the hotpotqa_fullwiki/bge-m3 injection sweep):
`tau* = 2.31` (Vendi of the kNN top-5, i.e. 0.46·k), `D* = MMR(0.7)`.
Transfer verified 2026-06-12 on the MiniLM fine-grid replications
(`results/2026-06-12_*`): HotpotQA distractor +2.5 S-Recall
pooled (above kNN at every level); SciFact (k=10, tau=4.62) within 1 point
of kNN everywhere, +0.2 pooled.

---

## RQ5 – Efficiency

```bash
python efficiency.py
python efficiency.py --dim 1024 --repeats 500 --pool_sizes 25 50 100 200 500
```

Times each method over synthetic random embeddings on a single CPU core.
Reports mean latency (ms/query) and fits a power law t ~ m^p to confirm
the theoretical O(m²D) scaling of RNG-Score and Seg-Score.

Outputs in `results/`:
- `results_efficiency.csv`
- `results_efficiency_scaling.csv`

---

## Models used

Pass either the **alias** (left column) or a full HuggingFace ID to
`--encoder_model` / `--ce_model` / `--generator_model`. Defaults are
device-based: the CPU default is chosen for laptop smoke tests, the CUDA
default for the full-scale runs.

### Encoders
| Alias | Model | Notes |
|---|---|---|
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | CPU default; fast |
| `bge-m3` | BAAI/bge-m3 | **CUDA default**; strong encoder |
| `qwen3-embed-4b` | Qwen/Qwen3-Embedding-4B | strongest; encoder-robustness check |

### Cross-encoders (RQ4)
| Alias | Model | Notes |
|---|---|---|
| `minilm-ce` | cross-encoder/ms-marco-MiniLM-L-6-v2 | CPU default; ~85 MB |
| `bge-reranker-v2-m3` | BAAI/bge-reranker-v2-m3 | **CUDA default** |
| `qwen3-reranker-4b` | Qwen/Qwen3-Reranker-4B | strongest |

### Generators / readers (RQ2/RQ4, closing-the-loop)
| Alias | Model | Notes |
|---|---|---|
| `flan-t5-small` | google/flan-t5-small | CPU default |
| `flan-t5-base` | google/flan-t5-base | **CUDA default**; seq2seq |
| `flan-t5-large` | google/flan-t5-large | larger seq2seq |
| `llama-3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | modern instruction-tuned reader (gated on HF) |
| `qwen3-4b` | Qwen/Qwen3-4B-Instruct-2507 | **closing-the-loop reader**: Qwen3 family, non-thinking instruct, ~2× lighter than 8B |
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | modern reader (closing-the-loop) |
| `qwen3-8b` | Qwen/Qwen3-8B | modern reader (closing-the-loop, scale check) |

Causal-LM readers (Llama/Qwen) are prompted with their **native chat template**
when the tokenizer ships one; checkpoints without a chat template fall back to a
plain completion prompt. Seq2seq Flan-T5 readers always use the plain prompt.

---

## RNG-Score margin α

| α | Effect |
|---|---|
| Very negative | All hinge terms inactive → equivalent to kNN |
| Near zero | Selective diversity pressure |
| Positive | Stronger penalty for redundant clusters |
| Very positive | All hinge terms active → ranks by n(w)·α (recovers kNN) |

Both extreme limits recover nearest-neighbor ranking (Propositions 4.4 and
4.5 in the paper).  The empirically useful regime is a bounded interval
around zero.


---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{brouillette2026balance,
  title  = {Finding the Right Balance: Relevance and Diversity in LLM Retrieval},
  author = {Brouillette, Guillaume and Kagabo, Faustin},
  year   = {2026},
  note   = {Preprint}
}
```

## License

This project is licensed under the Apache License 2.0 — see the
[LICENSE](LICENSE) file for details.
