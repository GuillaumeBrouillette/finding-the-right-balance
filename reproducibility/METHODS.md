# Reproducibility methods

This document preserves the detailed protocols, evidence definitions,
validation rules, and archival limitations for the article's reproducibility
package. Query splits and source groups are introduced in the researcher README
because they are the common foundation for the remaining artifacts.

## Algorithms, source identities, and seeds

The execution manifest is represented by
`manifests/execution/metadata.json`. It maps every recovered reported run to:

- its complete historical primary execution script at the exact 40-character
  Git commit;
- the Git blob ID and SHA-256 checksum of that script;
- the complete injection/chunking parameter values from `run_params.json`;
- the exact effective seeds and every RNG call site in the historical script;
- the immutable source identities from `manifests/source_groups/metadata.json`.

## Injection algorithm

The current executable implementation is in
`experiments/evaluate_redundancy.py`; exact historical implementations are
archived with the execution manifest. The `inject_duplicates` and
`_perturb_text` functions append `round(rho*n)` passages to a source pool of
size `n` at level `rho`. The
recorded `dup_target` determines whether source indices are gold-only, uniformly
random with replacement, or half gold and half random. The recorded
`dup_noise` selects exact copying, sentence permutation, or sentence permutation
with one sentence removed. Every copy retains the source title, which is the
source-group identity for datasets without native passage IDs.

## Chunking algorithm

The current `_chunk_words`, `chunk_pool`, and `run_chunking_experiment`
implementations are in `experiments/evaluate_redundancy.py`; historical
versions remain in the execution manifest. Text is
split on whitespace into `chunk_window`-word windows. At overlap `o`, stride is
`max(1, round(chunk_window*(1-o)))`. A final fragment shorter than half a window
is dropped unless the whole passage fits in one window. Every chunk retains the
source title and therefore the original source group.

## Random streams

- Injection constructs `numpy.random.default_rng(seed)` independently at each
  redundancy level. Selection and text perturbation consume the same stream.
- Validation/test assignment constructs a separate generator with the same
  run seed.
- Generation-query subsampling, where present, uses `seed + 1`.
- Chunk creation itself is deterministic; its seed changes only the split.
- FLAN generation uses deterministic beam decoding with `do_sample=False`.
  Qwen uses temperature 0, top-p 1 and an explicitly recorded API seed of 0.

Early cross-encoder and BEIR scripts used a literal seed `0`; the execution
manifest records that value as code-derived evidence. Later runs explicitly
record seed `0` or seeds `[0, 1, 2]` in `run_params.json`.

## Candidate-pool size assertions

The corrected sweep freezes a target for every query before any redundancy is
introduced:

```text
candidate_target(q) = min(top_m, clean_pool_size(q))
```

`top_m` is an upper bound because attached benchmark pools can contain fewer
than 100 documents. After injection or chunking, dense relevance truncation
must return exactly this clean-pool target at every level and seed.

The execution driver now asserts all of the following before saving results:

1. one transformed pool exists for every query;
2. every transformed pool contains at least its frozen clean target;
3. the number of embeddings equals the transformed-pool size;
4. the post-truncation candidate count equals the frozen target.

Every new sweep writes `results_<experiment>_pool_size_audit.csv` and also adds
the original, transformed, target, and final sizes to every per-query result.

## Historical limitation

The historical implementation used
`min(top_m, transformed_pool_size)`. Therefore pool size could grow with the
manipulation when a clean pool contained fewer than `top_m` documents. Existing
result files contain `PoolRedundancy` but no pool-size columns, so they do not
provide the requested proof. The corrected assertions and audit files apply to
new executions; affected sweeps must be rerun before the paper can claim a
fixed-size causal manipulation.

## Corrected rerun result

All 12 requested source configurations have now been rerun. They produced 13
audit artifacts because the shared SciFact configuration covers both injection
and chunking. Every recorded query, redundancy level, and seed passed the fixed
clean-pool assertion. The artifact paths, row counts, levels, seeds, and SHA-256
checksums are frozen in `manifests/pool_sizes/metadata.json`.

## Models, revisions, precision, and inference

The machine-readable record is `manifests/models/metadata.json`. It maps every
archived run (and each available corrected reproducibility run) to the following five model
repositories and immutable 40-character Hugging Face revisions. Two archived
post-processing runs that performed no model inference are explicitly marked
as analysis-only rather than being silently omitted.

| Role | Model | Immutable model and tokenizer revision | Weight precision |
|---|---|---|---|
| Encoder | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | float32 |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | float32 |
| Encoder | `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` | bfloat16 |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | float32 |
| Generator | `google/flan-t5-base` | `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` | float32 |
| Generator | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | server-managed |

Model and tokenizer were loaded from the same repository identifier without a
separate tokenizer revision, so the same repository SHA freezes both. Each
upstream revision predates its associated archived run and remained unchanged
through manifest recovery. The BGE-M3 and MiniLM revisions are additionally
confirmed by local Hugging Face cache refs; BGE-M3 is the snapshot used for the
corrected reruns.

The per-run manifest records CPU/CUDA device, batch size, normalization,
embedding output precision, similarity metric, retrieval depths, cross-encoder
length and score conversion, and generation length/beam/decoding settings.
Embedding and cross-encoder arrays are stored as float32. No autocast or
automatic mixed precision context is used. FLAN generation is deterministic
beam decoding (`do_sample=False`), with a 512-token input limit and the exact
per-run output-token and beam limits. The Qwen robustness check uses the fixed
`short_direct_v1` prompt, greedy decoding, temperature 0, top-p 1, seed 0, 64
output tokens and thinking disabled. Its NVIDIA vLLM environment and server
command are recorded in `reproducibility/QWEN38_GENERATION.md`.

The clean-pool eight-method Qwen run is protocol-compatible with the fixed-pool
implementation because no candidates are injected at `rho=0`. The first
clean/heavy Qwen attempt used the older draft truncation policy, which allowed
candidate-pool size to grow for clean pools smaller than `top_m`; it is retained
for audit but excluded from corrected heavy-regime evidence. A corrected
fixed-pool run is registered separately when complete.

Regenerate and validate the record with:

```bash
python reproducibility/create_model_manifest.py
python reproducibility/validate_model_manifest.py
```

## Environment and hardware

This record was captured on 2026-08-08 (America/Montreal) from the machine executing the corrected reproducibility sweeps.

## Software environment

| Component | Recorded value |
|---|---|
| Operating system | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Architecture | `aarch64` |
| Kernel | Linux 6.17.0-1026-nvidia |
| Python | CPython 3.12.3 |
| PyTorch | 2.12.0 |
| PyTorch CUDA runtime | 13.0 |
| cuDNN | 9.20.0 (`92000`) |
| NVIDIA driver | 580.173.02 |
| Repository commit at capture | `045293775a2cc0a041d33d063279d848227a73d9` |

The repository had uncommitted reproducibility changes when this record was captured. The complete Python package snapshot is in [`environment-lock.txt`](environment-lock.txt). Recreate it with Python 3.12 on a compatible aarch64/CUDA system:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r reproducibility/environment-lock.txt
```

## Hardware

| Component | Recorded value |
|---|---|
| Host | `spark-252a` |
| CPU architecture | ARM64 (`aarch64`) |
| Logical CPUs | 20 |
| CPU cores reported by `lscpu` | 10 × Cortex-X925 (maximum 3.9 GHz) and 10 × Cortex-A725 (maximum 2.808 GHz) |
| System memory | 121 GiB visible to Linux |
| GPU | NVIDIA GB10 |
| GPU UUID | `GPU-f0b49929-999a-9eec-7893-ecb7c7a6c08b` |
| GPU compute capability | 12.1 |
| GPU memory | Unified-memory platform; `nvidia-smi` reported memory total as `N/A` |

## Capture commands

The values above can be independently refreshed or checked with:

```bash
uname -a
lscpu
free -h
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv
python -m pip freeze --all
```

## Per-query execution artifacts

Status: **addressed with archival limitation**.

The execution-artifact archive preserves every recoverable per-query artifact without
rerunning an encoder, reranker, or generator. It separates evidence that was
retained during execution from evidence reconstructed or derived afterward and
from runtime state that the historical code never serialized.

## Coverage

| Requested item | Status | Evidence |
|---|---|---|
| Per-query candidate pools | Reconstructed input | `clean_pool_members.csv.gz` freezes 1,836,695 ordered clean-pool memberships for 30,890 query-pool configuration records (30,840 distinct dataset/query keys). `clean_pool_checksums.csv.gz` supplies ordered and unordered pool digests. Exact transformed pre-encoding pools remain reproducible from the frozen algorithms, source checksums, levels, seeds, and query order. |
| Embeddings or checksums | Unavailable historically | The runners did not serialize embedding tensors or candidate-keyed embedding hashes. Result-file hashes are retained, but are not represented as embedding checksums. |
| Rankings | Unavailable historically | The runners saved aggregate per-query retrieval metrics, not candidate identities and ranks after encoding/reranking. Scalar metrics cannot be inverted into an ordered ranking. |
| Hyperparameters | Retained | All 42 discovered run configurations and their SHA-256 hashes are indexed in `metadata.json`. |
| Metrics | Retained | Retained artifacts are indexed by path, type, size, and SHA-256; CSV evidence additionally records its schema and row count. |
| Trigger decisions | Exactly derived | `trigger_decisions.csv.gz` contains 872,032 validation/test decisions reconstructed from retained per-query kNN Vendi values, frozen splits, and validation-tuned rules. |
| Generated answers | Historical coverage plus Qwen extension | Two historical redundancy files preserve 199,810 qid-keyed EM/F1/hallucination rows without text. The new clean-pool Qwen run preserves 59,240 qid-keyed rows with raw predictions; its retained artifact is checksummed separately. |

This status is intentionally not marked “complete”: exact historical embedding
hashes, post-encoder top-m candidate membership, method-specific ordered
rankings, and raw generated answer strings would require new model execution.
Such a new run would be a new reproducibility trace, not recovery of the
historical runtime state.

## Archive files

The generated archive is under `manifests/execution_artifacts/`:

- `metadata.json`: coverage declaration, a SHA-256 inventory of 42 runs and
  460 retained artifacts, run hyperparameters, reconstruction configuration,
  exact reconstruction-code hashes, trigger rules, and explicit unavailable
  fields.
- `clean_pool_members.csv.gz`: one row per clean, pre-encoding query-to-source
  membership, in loader order.
- `clean_pool_checksums.csv.gz`: one row per query with membership count and
  ordered/unordered pool SHA-256 digests.
- `trigger_decisions.csv.gz`: one row per run, experiment, level, seed, and
  query with the trigger statistic, threshold, gate result, chosen method, and
  selected objective value.
- `SHA256SUMS`: hashes for the complete archive.

The archive indexes 75,018,201 rows of existing CSV evidence (11,273,200,878
bytes) without duplicating those retained result files.

## Candidate-pool digest definition

The byte strings hashed by the digest use UTF-8, LF line endings, no BOM, and
no additional Unicode normalization. The compressed CSV container itself uses
standard CSV record handling and is read independently of platform newlines.

For each query, the ordered digest is SHA-256 over these lines in loader order:

```text
position<TAB>source_variant_id<TAB>loader_content_sha256<LF>
```

The unordered digest is SHA-256 over these lines after lexicographic sorting:

```text
source_variant_id<TAB>loader_content_sha256<LF>
```

`preencoder_position` is the zero-based position supplied by the dataset
loader. It is not a dense-model rank.

## Trigger reconstruction rule

For every corrected sweep and seed:

1. The frozen split manifest selects validation and test queries.
2. Candidate thresholds are the rounded validation kNN Vendi values plus the
   lower and upper extremes used by the analysis implementation.
3. The threshold and fallback method are jointly selected on pooled validation
   levels using the run's objective. Ties preserve the implementation's first
   method and lowest encountered threshold.
4. A query triggers when its raw, unrounded kNN Vendi value is strictly below
   the selected threshold.
5. The `RelSetSize >= 2` gate is applied after ungated threshold tuning. This
   value comes from gold titles/qrels, so the gate is an analysis-only
   oracle/design gate, not a deployable label-free trigger.

These decisions are labelled **derived**, rather than historically serialized.

## Rebuild and validate

Run the generator using the same environment used for the corrected sweeps,
then run the standard-library validator:

```bash
python reproducibility/create_execution_artifacts.py
/usr/bin/python3 reproducibility/validate_execution_artifacts.py
```

The validator checks archive hashes, source/reference hashes, every pool member
position and query digest, unique trigger keys, trigger and gate Boolean logic,
selected methods/objective values, and all declared totals.

## Suggested disclosure for the article

> We release a SHA-256-indexed per-query archive preserving
> query/run/seed/condition/method keys, hyperparameters, pool-size assertions,
> and reported metrics. Each query's ordered pre-encoding clean pool is frozen
> using immutable source-group IDs and content hashes; exact algorithms, source
> checksums, query order, levels, and RNG seeds reconstruct injected and chunked
> pre-encoding pools. Trigger decisions are deterministically reconstructed from
> retained kNN trigger statistics and validation-tuned thresholds and are
> labelled derived. The historical runner did not serialize embedding tensors
> or hashes, post-encoder top-m membership/method rankings, or prediction
> strings; these cannot be recovered from metric-only files and are disclosed
> as unavailable. Two retained generation files preserve qid-keyed
> EM/F1/hallucination scores; other generation outputs contain only summaries
> or unkeyed scores. The qrel-derived `RelSetSize` gate is reported as an
> analysis-only oracle/design gate, not a deployable label-free trigger.

## Paired contrasts, confidence intervals, and p-values

## Status

**Addressed with archival limitation.** The complete recoverable primary
sweep family is published and validated. Historical result families for which
only aggregate values were retained are identified below and are not presented
as if query-level inference could be reconstructed.

## Statistical estimand and protocol

The primary endpoint is **S-Recall@k** for all 13 corrected injection and
chunking sweep artifacts. For method family `m`, query `q`, seed `s`, and
measured condition `l`, the raw contrast is

```text
d[q,s,l,m] = S-Recall[q,s,l,m] - S-Recall[q,s,l,kNN].
```

The analysis follows the declared sampling logic:

1. Dedup, MMR, VendiG, and RNG are re-tuned separately for each seed and
   condition on that seed's frozen **validation** queries using S-Recall.
   Exact ties select the first member in the frozen grid order. Maxmin and
   Greedy-DPP have no tuning parameter in this comparison.
2. The selected member is applied only to the frozen **test** queries.
3. Repeated seed observations are averaged within `query_id`; seeds are not
   treated as independent studies.
4. Whole query trajectories are resampled with a deterministic ordinary
   nonparametric query-cluster bootstrap. A single query-weight vector is
   shared across every method and condition in a run, preserving paired and
   repeated-measures structure.
5. Level effects use two-sided 95% percentile intervals and centered-bootstrap
   p-values. The primary heavy-minus-clean interaction also reports a one-sided
   95% lower bound and a one-sided p-value for improvement with redundancy.
6. Holm step-down adjustment controls familywise error separately for (a) all
   method-by-level effects in one run and (b) the six primary interactions in
   that run. Raw and adjusted p-values are both retained.

Effects and intervals are reported in **percentage points**. P-values use the
finite-resampling correction `(extreme + 1) / (B + 1)`. The frozen build uses
1,999 bootstrap resamples and seed `20260809`, giving a minimum attainable raw
p-value of `0.0005`.

Five corrected artifacts had historical `run_params.json` files that still
named alpha-NDCG as the objective (HotpotQA injection/chunking, SciFact
injection/chunking, and TREC-COVID chunking). The statistical generator does not
reuse their stale `Chosen` values: it recomputes validation choices on
S-Recall, matching the manuscript's declared primary sweep endpoint.

## Released evidence

| Artifact | Contents | Rows |
|---|---|---:|
| [`validation_selections.csv`](../manifests/statistics/validation_selections.csv) | Exact validation-selected family member, validation mean, query count, and tie rule | 1,320 |
| [`raw_seed_contrasts/`](../manifests/statistics/raw_seed_contrasts/) | Method and kNN values plus their raw paired difference for every selected test observation | 4,186,032 |
| [`query_averaged_contrasts/`](../manifests/statistics/query_averaged_contrasts/) | Seed observations averaged within query, with chosen-member provenance | 1,814,424 |
| [`paired_inference.csv`](../manifests/statistics/paired_inference.csv) | Mean effects, exact reported intervals, raw/adjusted p-values, and improved/tied/harmed fractions | 534 |
| [`metadata.json`](../manifests/statistics/metadata.json) | Statistical policy, counts, source hashes, output hashes, and limitations | 1 manifest |
| [`SHA256SUMS`](../manifests/statistics/SHA256SUMS) | Integrity hashes for every published statistical file | 1 index |

The 534 inferential rows comprise 456 method-by-level effects and 78
prespecified heavy-minus-clean interactions across the 13 corrected sweep
artifacts.

## Rebuild and validate

Run these commands from the repository root in the locked project environment:

```bash
python reproducibility/create_statistical_archive.py --workers 3 --bootstrap-samples 1999
python reproducibility/validate_statistical_archive.py
```

`--workers` parallelizes independent runs. Each job derives its random stream
from the global seed and immutable run name, so changing worker count does not
change any statistical value.

The validator independently:

- checks all source and output SHA-256 hashes;
- redoes validation-set S-Recall tuning from the original per-query files;
- proves every raw row belongs to the frozen test partition;
- recomputes raw and query-averaged paired contrasts;
- reruns the trajectory-preserving bootstrap; and
- exactly reproduces means, intervals, raw p-values, Holm-adjusted p-values,
  sample counts, and improved/tied/harmed fractions.

## Archival limitations

The following historical states were never serialized and cannot be recreated
without new model execution:

- a frozen query-family identifier linking questions that may share source
  documents; `query_id` is therefore the finest available sampling unit;
- per-query historical FLAN Table 2 generation outputs needed for raw paired
  EM/F1 inference (the new Qwen extension does retain these rows);
- per-query ArguAna and Touché outputs underlying the aggregate-only Table 3
  columns; and
- per-query non-Hotpot cross-encoder outputs underlying the aggregate-only
  parts of Table 4.

Those historical results must remain descriptive unless they are rerun with
per-query serialization. This limitation does not affect the released paired
inference for the corrected injection and chunking sweeps.

## Complementary revision statistics

The primary statistical archive above covers the validation-selected method
effects and heavy-minus-clean interactions from the 13 corrected sweeps. The
revision also introduced derived quantities with a different source lineage.
They are generated by [`run_remaining_stats.py`](run_remaining_stats.py) and
published under [`results/revision_stats/`](../results/revision_stats/):

| Output | Published contents |
|---|---|
| `table_refresh/` | Thirteen corrected test-split level-mean CSVs used to refresh the displayed injection and chunk-overlap sweeps |
| `rule_transfer/rule_inference.csv` | Frozen-rule effects, query-clustered intervals, multiplicity-adjusted p-values, trigger rates, and improved/harmed fractions |
| `gate_budgeted/gate_budgeted.csv` | BEIR subset of the frozen-rule analysis under the budgeted threshold |
| `generation_inference/generation_inference.csv` | Paired EM risk differences and F1 mean differences for the two retained historical generation runs |
| `oracle_intervals/oracle_intervals.csv` | Oracle headroom, rule/tuned capture, and misspecification-downside intervals on the headline sweep |
| `minimax_regret/minimax_regret.csv` | Common-oracle minimax regret and the selected fixed member for each displayed family |

These calculations reuse the 1,999-resample query-cluster bootstrap, fixed
seed, centered p-values, and Holm correction described above. No model
inference is performed. Their provenance is intentionally mixed: the table
refresh uses `results/reproducibility_reruns/`, while the rule, gate,
generation, oracle, and regret quantities use the historical runs under
`results/retained/`. This matches which execution family the manuscript
displays and is recorded in
[`SOURCE_MODE.txt`](../results/revision_stats/SOURCE_MODE.txt).

Rebuild the two groups explicitly:

```bash
python reproducibility/run_remaining_stats.py \
  --source-mode historical \
  --only rule_transfer,generation_inference,oracle_minimax
python reproducibility/run_remaining_stats.py \
  --source-mode corrected \
  --only table_refresh
```

Dataset-level descriptive claims are handled separately by
[`dataset_stats.py`](dataset_stats.py). It reads QA loaders or BEIR qrels and
writes [`results/dataset_statistics.csv`](../results/dataset_statistics.csv).
The committed file freezes the SciFact, FiQA, and TREC-COVID values cited in
Section 5.2; other datasets can be reacquired through the documented loaders.

## Table and figure reconstruction

Status: **complete**. One deterministic entry point reconstructs all 12 numbered
tables and all 8 numbered figures without loading a model or using a GPU.

## Commands

```bash
python reproducibility/reconstruct_paper.py
python reproducibility/validate_paper_reconstruction.py
```

If the system Python does not expose the archived environment, use the locked
environment documented above. Outputs are written to
`manifests/paper_reconstruction/`; `metadata.json` records every input/output hash and
`SHA256SUMS` freezes the reconstruction.

## Table lineage

| Paper table | Reconstructed content | Retained source |
|---|---|---|
| 1 | Five displayed passage-title rankings | Frozen displayed rows from the retained passage export |
| 2 | HotpotQA fixed rerankers and generation | Per-query injection rows + frozen split; two FLAN summaries; Qwen extension with 59,240 per-query predictions |
| 3 | Five BEIR tasks | Two BEIR result summaries |
| 4 | Four cross-encoder pipelines | Four cross-encoder summaries; NDCG-tuned S2-V1 rows |
| 5 | Misspecification regret | HotpotQA `analysis_regret.csv` |
| 6 | HotpotQA injection sweep | HotpotQA `analysis_summary.csv` |
| 7 | Selected RNG margins | Five per-seed sweep summaries; modal/sign rule is recomputed |
| 8 | Chunk-overlap sweep | Six retained chunking summaries |
| 9 | Frozen decision rule | Per-query rows + frozen split for the three deployable policies; retained threshold bounds |
| 10 | Transfer summary | Fifteen non-tuning targets in `frozen_rule_transfer_summary.csv` |
| 11 | Answer quality under injection | Two frozen generation analyses |
| 12 | Single-stage QA comparisons | Three non-overlap analysis summaries |

The CSVs preserve unrounded values. The validator checks the complete row
shape plus published-value anchors at the paper's declared rounding. In
particular, Table 2's four fixed MMR settings and Table 9's frozen rule are
recomputed from per-query rows rather than copied from tuned aggregate rows.

## Figure lineage

| Paper figure | Reconstruction |
|---|---|
| 1 | Three-policy HotpotQA summary from gate, measured-redundancy and frozen-rule files |
| 2 | Deterministic lune and seeded RNG TikZ components |
| 3 | Deterministic k-NN/RNG/MMR selection geometry in one TikZ asset |
| 4 | Three-encoder HotpotQA crossover panels |
| 5 | SciFact/FiQA crossover panels |
| 6 | Six-dataset injection and chunk-overlap oracle panels |
| 7 | Threshold curve and frozen-rule panels |
| 8 | HotpotQA and 2Wiki generation panels |

Figures 2 and 3 are code-generated, format-equivalent geometry assets. Their
final subfigure framing is performed by LaTeX and contains no experimental
data. Figures 1 and 4–8 are derived exclusively from retained per-run files.

## Validation

The validator verifies:

- all 53 retained inputs and every output against SHA-256;
- 12/12 table schemas, row counts and manuscript-value anchors;
- 8/8 figure coverage;
- absence of model execution in the reconstruction scope; and
- byte-identical output from a second clean reconstruction, including PDFs.
