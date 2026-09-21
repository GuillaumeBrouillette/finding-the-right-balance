# Reproducibility coverage

This table maps the article's reproducibility materials to their published evidence.

| Area | Published evidence | Coverage |
|---|---|---|
| Query identities and splits | Frozen query/split files in [`manifests/splits/`](../manifests/splits/) and immutable source groups in [`manifests/source_groups/`](../manifests/source_groups/). | Complete |
| Algorithms, sources, and randomness | Frozen historical code, source links, run parameters, RNG call sites, and effective seeds in [`manifests/execution/`](../manifests/execution/). | Complete |
| Candidate-pool invariants | Runtime assertions and validated per-query pool-size audits in [`manifests/pool_sizes/`](../manifests/pool_sizes/). | Complete |
| Models and inference | Immutable model/tokenizer revisions, precision, batching, retrieval, reranking, and generation settings in [`manifests/models/`](../manifests/models/), plus the Qwen protocol in [`QWEN38_GENERATION.md`](QWEN38_GENERATION.md). | Complete for published runs; Qwen extension staged separately. |
| Environment and hardware | Exact 111-package [`environment-lock.txt`](environment-lock.txt) and captured software/hardware details in [`METHODS.md`](METHODS.md). | Complete |
| Per-query execution artifacts | Forty-two historical runs and 460 retained artifacts indexed in [`manifests/execution_artifacts/`](../manifests/execution_artifacts/), including 1,836,695 clean-pool memberships and 872,032 reconstructed trigger decisions. The staged Qwen clean-pool extension adds 59,240 rows with raw predictions. | Historical limitations remain; the Qwen extension preserves predictions and source hashes prospectively. |
| Statistical inference | 4,186,032 raw paired contrasts, 1,814,424 query-averaged contrasts, and 534 inferential rows in [`manifests/statistics/`](../manifests/statistics/). | All recoverable inference published; aggregate-only historical generation, ArguAna/Touché, and non-Hotpot cross-encoder cells remain descriptive. |
| Revision-specific analyses | CPU-only launcher [`run_remaining_stats.py`](run_remaining_stats.py), 18 small CSVs under [`results/revision_stats/`](../results/revision_stats/), and explicit source provenance in [`SOURCE_MODE.txt`](../results/revision_stats/SOURCE_MODE.txt). | Corrected fixed-pool sources for table refreshes; historical displayed-run sources for rule, gate, generation, oracle, and regret quantities. |
| Dataset descriptions | Recomputable pool and relevant-set summaries in [`dataset_stats.py`](dataset_stats.py), with the committed BEIR values in [`results/dataset_statistics.csv`](../results/dataset_statistics.csv). | The committed CSV covers SciFact, FiQA, and TREC-COVID; QA summaries require reacquiring the documented datasets. |
| Paper reconstruction | Deterministic reconstruction of all 12 tables and all 8 figures from 56 checksummed inputs in [`manifests/paper_reconstruction/`](../manifests/paper_reconstruction/). | Complete |

Detailed definitions, statistical procedures, validation rules, and limitations are in [`METHODS.md`](METHODS.md).

The modern-reader extension is indexed separately in
[`manifests/qwen38/`](../manifests/qwen38/) so the new exact-prediction runs do
not get conflated with historically recovered evidence.
