# Reproducibility

The coverage of the article's published artifacts is summarized in
[`COVERAGE.md`](COVERAGE.md).
Detailed protocols and limitations are consolidated in
[`METHODS.md`](METHODS.md).

## GitHub Release artifacts

Large artifacts are published in the
[`reproducibility-v1` GitHub Release](https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/tag/reproducibility-v1),
not in the Git history:

- [`finding-the-right-balance-evidence-2026-08-10.tar.zst`](https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1/finding-the-right-balance-evidence-2026-08-10.tar.zst)
  (624 MiB) restores the compressed manifests under `manifests/`;
- [`finding-the-right-balance-retained-results-2026-08-10.tar.zst`](https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1/finding-the-right-balance-retained-results-2026-08-10.tar.zst)
  (934 MiB) restores `results/retained/` and
  `results/reproducibility_reruns/`;
- [`finding-the-right-balance-qwen38-results-2026-09-03.tar.zst`](https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst)
  (18 MiB) adds the Qwen3.8-27B generation results.

```text
Evidence SHA-256: 4603be6aa3463986f390eb3acebf19448b161952bcf6feb6c4378ddf12720a64
Retained-results SHA-256: d32892e10ee8cb0516eb89c6fdbb9993701c8550156f0b3d4f24494114770da8
```

From the repository root, download, verify, and extract the assets:

```bash
mkdir -p artifacts
curl -L -o artifacts/finding-the-right-balance-evidence-2026-08-10.tar.zst https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1/finding-the-right-balance-evidence-2026-08-10.tar.zst
curl -L -o artifacts/finding-the-right-balance-retained-results-2026-08-10.tar.zst https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1/finding-the-right-balance-retained-results-2026-08-10.tar.zst
curl -L -o artifacts/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst
sha256sum --check reproducibility/EVIDENCE_ARTIFACT_SHA256
sha256sum --check reproducibility/COMPANION_ARTIFACT_SHA256
sha256sum --check reproducibility/QWEN38_ARTIFACT_SHA256
tar --use-compress-program=unzstd -xf artifacts/finding-the-right-balance-evidence-2026-08-10.tar.zst
tar --use-compress-program=unzstd -xf artifacts/finding-the-right-balance-retained-results-2026-08-10.tar.zst
tar --use-compress-program=unzstd -xf artifacts/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst
```

The evidence archive is sufficient to inspect the published large manifests.
The retained-results archive is additionally needed to rebuild the execution
archive, statistical analyses, tables, and figures from retained source rows.
Dataset snapshots are reacquired through the documented loaders and immutable
source revisions rather than redistributed.

The frozen split manifests in `manifests/splits/` recover the exact query order
and seed-controlled validation/test partition used by the reported experiments.
They are generated from the recovered clean-level, per-query k-NN rows using the
same rule as the experiment code:

```text
numpy.random.default_rng(seed).permutation(number_of_queries)
```

Run:

```bash
python reproducibility/create_split_manifests.py
python reproducibility/validate_split_manifests.py
python reproducibility/create_source_group_manifests.py
python reproducibility/validate_source_group_manifests.py
python reproducibility/create_execution_manifest.py
python reproducibility/validate_execution_manifest.py
python reproducibility/create_pool_size_manifest.py
python reproducibility/validate_pool_size_manifest.py
python reproducibility/create_model_manifest.py
python reproducibility/validate_model_manifest.py
python reproducibility/create_execution_artifacts.py
/usr/bin/python3 reproducibility/validate_execution_artifacts.py
python reproducibility/create_statistical_archive.py --workers 3 --bootstrap-samples 1999
python reproducibility/validate_statistical_archive.py
python reproducibility/reconstruct_paper.py
python reproducibility/validate_paper_reconstruction.py
python reproducibility/create_qwen38_manifest.py
python reproducibility/validate_qwen38_manifest.py
```

The Qwen3.8 extension is reconstructed from existing retained outputs; these
commands perform no model inference. Its exact reader protocol, environment,
resumable runner and standalone table builders are documented in
[`QWEN38_GENERATION.md`](QWEN38_GENERATION.md). The dedicated manifest marks
the clean eight-method run as pool-policy equivalent and keeps the earlier
clean/heavy runs explicitly identified as using the pre-correction draft pool
truncation policy.

The staged release asset is
`artifacts/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst`;
verify it with:

```bash
sha256sum --check reproducibility/QWEN38_ARTIFACT_SHA256
```

`metadata.json` records the source result file, its SHA-256 digest, the source
run configuration, query-order digest, seed counts, and other recovered runs
whose query order was checked against the canonical source. `SHA256SUMS` covers
the committed manifest files.

## Source-document groups

The compressed manifests in `manifests/source_groups/` assign immutable group
IDs to the original documents exposed by each historical loader. Native passage
IDs are used for DPR NQ and BEIR. Datasets without native passage IDs use the
Wikipedia title as the document identity. Every title/text variant is protected
by SHA-256.

The generator also proves that source query order exactly matches every frozen
split manifest. For 2WikiMultiHopQA, the manifest records the immutable dataset
repository revision used to retrieve `dev.parquet`. The raw datasets remain
git-ignored; their exact checksums are recorded in source-group metadata.

Per-query final candidate pools and transformed duplicate/chunk instances are
documented separately and are not represented as if they had been historically
saved.

## Historical algorithms and seeds

`manifests/execution/` archives the complete primary script
from the exact Git commit of each recovered run and maps all 30 run parameter
files to immutable code hashes, source identities, effective seeds, RNG call
sites, and transformation parameters. The protocol is in [`METHODS.md`](METHODS.md).

## Candidate-pool size assertions

`manifests/pool_sizes/` and [`METHODS.md`](METHODS.md) document the pool-size invariant. New injection
and chunking runs freeze each query's clean candidate count across every level,
assert the invariant before and after encoding/truncation, and save a dedicated
pool-size audit CSV. The manifest explicitly records that historical outputs
lack this proof and require rerunning under the corrected policy.

## Environment and hardware

`environment-lock.txt` and [`METHODS.md`](METHODS.md) record
the exact Python package versions used by the reproducibility reruns together
with the operating system, Python/CUDA stack, CPU, memory, GPU, driver, and
repository revision observed on the execution machine.

## Models and inference

`manifests/models/` and [`METHODS.md`](METHODS.md) map each run to
immutable model and tokenizer repository revisions and preserve model precision,
device, batching, embedding, reranking, and deterministic generation settings.

## Per-query execution artifacts

`manifests/execution_artifacts/` and [`METHODS.md`](METHODS.md) provide the
maximum recoverable per-query evidence without model inference. The archive
indexes all retained per-run artifacts, freezes 1,836,695 clean pre-encoding
candidate memberships across 30,890 query-pool configuration records and their
checksums, and derives 872,032 per-query trigger decisions from retained values
and frozen splits. It also explicitly records
the historical runtime fields that were never serialized and therefore cannot
be recovered: embedding hashes, post-encoder rankings, and raw generated answer
strings.

## Paired statistical inference

`manifests/statistics/` and [`METHODS.md`](METHODS.md) publish
4,186,032 raw seed-level paired contrasts, 1,814,424
query-averaged contrasts, and 534 inferential results for all 13 corrected
sweeps. The analysis uses frozen validation/test splits, validation-selected
S-Recall, query-cluster bootstrap confidence intervals, raw p-values, and
Holm-adjusted p-values. Historical aggregate-only result families are clearly
identified rather than assigned unrecoverable query-level statistics.

## Table and figure reconstruction

`manifests/paper_reconstruction/` and [`METHODS.md`](METHODS.md) provide one
deterministic entry point that reconstructs all 12 numbered tables and all 8
numbered figures from 56 checksummed retained inputs. The
validator checks manuscript-value anchors and proves a second clean build is
byte-identical, including the generated PDFs.
