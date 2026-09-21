# Finding the Right Balance: Relevance and Diversity in LLM Retrieval

Official code and reproducibility package for **“Finding the Right Balance:
Relevance and Diversity in LLM Retrieval”** by Guillaume Brouillette,
Faustin Kagabo, Usef Faghihi and Nadia Ghazzali.

The project evaluates RNG-Score and established diversification methods under
controlled redundancy. It includes retrieval and generation experiments,
cross-encoder evaluations, statistical analyses, and the retained evidence
used to reconstruct the paper.

The manuscript is available at
[`paper/Finding_the_right_balance.pdf`](paper/Finding_the_right_balance.pdf).

## Choose a workflow

| Goal | Start here | Model inference required? |
|---|---|---:|
| Check that the software works | [Quick CPU smoke test](#3-quick-cpu-smoke-test) | Yes, small encoder only |
| Verify the published evidence | [Reproduce from retained results](#4-reproduce-from-retained-results) | No |
| Repeat an experiment | [Run experiments](#5-run-experiments) | Yes |
| Repeat the Qwen3.8-27B reader evaluation | [`reproducibility/QWEN38_GENERATION.md`](reproducibility/QWEN38_GENERATION.md) | Yes |

For paper verification, the retained-results workflow is the recommended
starting point. Repeating every model run is substantially more expensive and
is not required to inspect the reported evidence.

## 1. Installation

Requirements:

- Linux or macOS;
- Python 3.9 or newer (Python 3.10–3.12 recommended);
- Git;
- internet access for the first dataset/model download;
- `curl`, `tar`, and `zstd` only when downloading release artifacts.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/GuillaumeBrouillette/finding-the-right-balance.git
cd finding-the-right-balance

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Confirm the installation:

```bash
python -m pytest -q --ignore=tests/test_split_manifests.py
ftrb-evaluate --help
ftrb-redundancy --help
```

The ignored test module validates large files supplied by the release. Run the
complete suite after extracting those files in Step 4.

GPU execution is the default for experiment commands. Install the PyTorch
build recommended by the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
the editable installation, then verify CUDA:

```bash
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

Use `--device cuda:N` to select a GPU. If CUDA is unavailable, experiment
commands stop with a clear error instead of silently starting a slow CPU run.
Use `--device cpu` only when CPU execution is intentional.

## 2. Resources and downloads

Datasets and Hugging Face models are downloaded automatically on first use.
The default Hugging Face cache is used; set `HF_HOME` before running if the
cache must live on another disk.

### Datasets

| Resource | Used for | Source and local behavior |
|---|---|---|
| HotpotQA distractor | Laptop smoke tests and attached-passage evaluation | [Hugging Face](https://huggingface.co/datasets/hotpotqa/hotpot_qa); cached by `datasets` |
| HotpotQA MDR top-100 | Full-wiki controlled reranking | [MDR candidate file](https://dl.fbaipublicfiles.com/mdpr/data/hotpot/dev_retrieval_top100_sp.json); approximately 652 MB, downloaded to `data/mdr/` |
| 2WikiMultiHopQA | Multi-hop transfer and generation | [Hugging Face](https://huggingface.co/datasets/xanhho/2WikiMultihopQA); the loader pins the recorded revision |
| MuSiQue | Multi-hop transfer | [Hugging Face](https://huggingface.co/datasets/dgslibisey/musique); the loader pins the recorded revision |
| NQ-Open | Single-hop control | [Hugging Face questions](https://huggingface.co/datasets/google-research-datasets/nq_open) or [DPR top-100 candidates](https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz) |
| SQuAD | Cross-encoder control | [Hugging Face](https://huggingface.co/datasets/rajpurkar/squad) |
| BEIR | SciFact, FiQA, TREC-COVID, ArguAna, and Webis-Touché2020 | [BEIR project](https://github.com/beir-cellar/beir); task archives are downloaded to `data/beir/` |

Downloaded datasets are excluded from Git. Immutable revisions, checksums,
and source-group identities used by the paper are recorded under
[`manifests/`](manifests/) and described in
[`reproducibility/METHODS.md`](reproducibility/METHODS.md).

### Main models

| CLI alias | Model |
|---|---|
| `minilm` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `bge-m3` | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) |
| `qwen3-embed-4b` | [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| `minilm-ce` | [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| `bge-reranker-v2-m3` | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| `flan-t5-small` | [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) |
| `flan-t5-base` | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) |
| `qwen3.8-27b` | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |

The complete alias registry is in
[`ftrb/run_utils.py`](ftrb/run_utils.py). A full Hugging Face model ID may be
passed instead of an alias. Llama checkpoints are optional and require users
to accept Meta’s access terms on Hugging Face.

## 3. Quick CPU smoke test

This is the smallest end-to-end retrieval check. It downloads HotpotQA and
MiniLM, evaluates 20 questions, and performs no answer generation:

```bash
ftrb-evaluate \
  --dataset hotpotqa \
  --split validation \
  --max_samples 20 \
  --encoder_model minilm \
  --device cpu \
  --no_generation \
  --output_dir results/smoke
```

The command prints a retrieval table and creates a timestamped directory such
as:

```text
results/smoke/2026-09-04_120000_hotpotqa/
├── run_params.json
├── results_retrieval.csv
└── results_retrieval_significance.csv
```

`run_params.json` records the arguments, Git commit, runtime packages, and
hardware information. Keep it with the CSV files when sharing a run.

## 4. Reproduce from retained results

This workflow reconstructs and validates the published evidence without
running an encoder, cross-encoder, or generator.

### 4.1 Download the release artifacts

The [`reproducibility-v1` release](https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/tag/reproducibility-v1)
contains:

| Archive | Size | Contents |
|---|---:|---|
| `finding-the-right-balance-evidence-2026-08-10.tar.zst` | 624 MiB | Compressed manifests and statistical evidence |
| `finding-the-right-balance-retained-results-2026-08-10.tar.zst` | 934 MiB | Retained results and corrected reproducibility runs |
| `finding-the-right-balance-qwen38-results-2026-09-03.tar.zst` | 18 MiB | Qwen3.8-27B retrieval-conditioned answers and tables |

From the repository root:

```bash
mkdir -p artifacts
release_base="https://github.com/GuillaumeBrouillette/finding-the-right-balance/releases/download/reproducibility-v1"

curl -L -o artifacts/finding-the-right-balance-evidence-2026-08-10.tar.zst \
  "$release_base/finding-the-right-balance-evidence-2026-08-10.tar.zst"
curl -L -o artifacts/finding-the-right-balance-retained-results-2026-08-10.tar.zst \
  "$release_base/finding-the-right-balance-retained-results-2026-08-10.tar.zst"
curl -L -o artifacts/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst \
  "$release_base/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst"
```

Verify all downloads before extraction:

```bash
sha256sum --check reproducibility/EVIDENCE_ARTIFACT_SHA256
sha256sum --check reproducibility/COMPANION_ARTIFACT_SHA256
sha256sum --check reproducibility/QWEN38_ARTIFACT_SHA256
```

Extract them into the repository:

```bash
tar --use-compress-program=unzstd -xf artifacts/finding-the-right-balance-evidence-2026-08-10.tar.zst
tar --use-compress-program=unzstd -xf artifacts/finding-the-right-balance-retained-results-2026-08-10.tar.zst
tar --use-compress-program=unzstd -xf artifacts/finding-the-right-balance-qwen38-results-2026-09-03.tar.zst
```

### 4.2 Validate the archive

Run the validators in this order:

```bash
python -m pytest -q
python reproducibility/validate_split_manifests.py
python reproducibility/validate_source_group_manifests.py
python reproducibility/validate_execution_manifest.py
python reproducibility/validate_pool_size_manifest.py
python reproducibility/validate_model_manifest.py
python reproducibility/validate_execution_artifacts.py
python reproducibility/validate_statistical_archive.py
python reproducibility/validate_qwen38_manifest.py
python reproducibility/validate_paper_reconstruction.py --reuse-existing-figures
```

The final option validates all table values and checksums while reusing the
archived PDFs. Exact PDF bytes can vary with Matplotlib and installed fonts;
the recorded Python environment is
[`reproducibility/environment-lock.txt`](reproducibility/environment-lock.txt).

### 4.3 Rebuild the paper tables and figures

Write a fresh reconstruction outside the committed manifest directory:

```bash
python reproducibility/reconstruct_paper.py \
  --results results/retained \
  --out reproduced/paper
```

Expected output:

```text
reproduced/paper/
├── tables/       # 12 numbered CSV tables
├── figures/      # 8 numbered figure groups
├── metadata.json
└── SHA256SUMS
```

See [`reproducibility/README.md`](reproducibility/README.md) for artifact
lineage and [`reproducibility/COVERAGE.md`](reproducibility/COVERAGE.md) for
known archival limitations.

## 5. Run experiments

Every experiment writes to a timestamped directory and records its parameters.
Start with a small sample before changing `--max_samples` to `all`.

### 5.1 Core retrieval and generation

Retrieval only:

```bash
ftrb-evaluate \
  --dataset hotpotqa \
  --max_samples 100 \
  --encoder_model minilm \
  --device cpu \
  --no_generation \
  --save_per_query
```

Add the default local reader by removing `--no_generation`, or specify a
reader with `--generator_model`:

```bash
ftrb-evaluate \
  --dataset hotpotqa \
  --max_samples 100 \
  --encoder_model bge-m3 \
  --generator_model flan-t5-base \
  --device cuda \
  --save_per_query
```

### 5.2 Controlled redundancy and oracle headroom

Small CPU run:

```bash
ftrb-redundancy \
  --experiment both \
  --dataset hotpotqa \
  --max_samples 100 \
  --encoder_model minilm \
  --device cpu \
  --seeds 0 1 2
```

Full HotpotQA MDR redundancy sweep:

```bash
ftrb-redundancy \
  --experiment redundancy \
  --dataset hotpotqa_fullwiki \
  --split validation \
  --max_samples all \
  --encoder_model bge-m3 \
  --device cuda \
  --batch_size 64 \
  --top_m 100 \
  --top_k 5 \
  --objective alpha-NDCG@k \
  --seeds 0 1 2 \
  --output_dir results
```

The command prints the exact run directory. Use it for the derived analysis:

```bash
run_dir="results/<timestamp>_redundancy_hotpotqa_fullwiki"

ftrb-analyze \
  --per_query "$run_dir/results_redundancy_per_query.csv" \
  --objective alpha-NDCG@k \
  --top_k 5

ftrb-plot-regimes \
  --run_dir "$run_dir" \
  --out_dir "$run_dir/plots"
```

For natural redundancy from overlapping chunks, replace `--experiment
redundancy` with `--experiment chunking`. Use `--experiment all` to run
redundancy injection, chunking, and oracle headroom together.

### 5.3 BEIR evaluation

The following command downloads SciFact, builds or reuses its index, and runs
the three-seed evaluation:

```bash
ftrb-beir \
  --tasks scifact \
  --max_queries all \
  --encoder_model bge-m3 \
  --device cuda \
  --seeds 0 1 2 \
  --objective alpha_ndcg \
  --index_dir results/indices \
  --save_per_query
```

Other supported tasks are `fiqa`, `trec-covid`, `arguana`, and
`webis-touche2020`.

### 5.4 Cross-encoder evaluation

```bash
ftrb-cross-encoder \
  --dataset hotpotqa_fullwiki \
  --max_samples 100 \
  --encoder_model bge-m3 \
  --ce_model bge-reranker-v2-m3 \
  --device cuda \
  --no_generation \
  --seeds 0 1 2 \
  --save_per_query
```

This pipeline performs inference only; it does not train either encoder.

### 5.5 Efficiency and synchronized latency

Method-scaling benchmark:

```bash
ftrb-efficiency \
  --dim 1024 \
  --repeats 500 \
  --pool_sizes 25 50 100 200 500
```

Single-query latency distribution:

```bash
ftrb-benchmark-cpu \
  --pool-size 100 \
  --top-k 5 \
  --dim 1024 \
  --warmup 100 \
  --repeats 10000 \
  --inputs 128 \
  --cpu 5
```

Choose a valid isolated CPU number with `taskset -pc $$` or override `--cpu`.
The benchmark fixes BLAS thread counts and records hardware, warm-up, and
latency percentiles in `results/cpu_latency/`.

### 5.6 Qwen3.8-27B reader evaluation

The Qwen run uses a separately launched OpenAI-compatible vLLM server, a
pinned model revision, deterministic decoding, resumable checkpoints, and
fixed prompts. Follow the exact commands in
[`reproducibility/QWEN38_GENERATION.md`](reproducibility/QWEN38_GENERATION.md).

## Outputs and statistical protocol

Depending on the command, a run directory contains:

- `run_params.json`: arguments, source revision, environment, and hardware;
- `results_*_summary.csv`: aggregated metrics;
- `results_*_per_seed_summary.csv`: seed-level results;
- `results_*_per_query.csv`: paired query-level observations;
- `results_*_significance.csv`: confidence intervals and paired tests;
- `analysis_*.csv` and `analysis_*_info.json`: derived regime analyses.

Deterministic retrieval/generation comparisons use query-bootstrap confidence
intervals and paired Wilcoxon tests. Redundancy, BEIR, and cross-encoder
experiments additionally report variation across seeds. The complete protocol,
including Holm correction and query clustering, is documented in
[`reproducibility/METHODS.md`](reproducibility/METHODS.md).

Revision-specific rule, generation, oracle, regret, and refreshed-table
statistics can be rebuilt without model inference using:

```bash
python reproducibility/run_remaining_stats.py --help
python reproducibility/dataset_stats.py --help
```

The committed outputs are under [`results/revision_stats/`](results/revision_stats/)
and [`results/dataset_statistics.csv`](results/dataset_statistics.csv); their
mixed corrected/historical source lineage is documented in
[`results/revision_stats/SOURCE_MODE.txt`](results/revision_stats/SOURCE_MODE.txt).

## Project structure

```text
.
├── experiments/       evaluation and tuning commands
├── analysis/          retained-result analysis and figures
├── ftrb/              shared selection, statistics, and run utilities
├── data/              dataset loaders
├── retrieval/         dense retrieval, rerankers, and cross-encoders
├── generation/        local and OpenAI-compatible readers
├── evaluation/        retrieval and answer-quality metrics
├── benchmarks/        synchronized performance benchmarks
├── reproducibility/   artifact builders, validators, and methods
├── manifests/         frozen provenance and reconstructed evidence
├── paper/             manuscript PDF
└── tests/             unit and reproducibility regression tests
```

## Citation

```bibtex
@article{brouillette2026balance,
  title  = {Finding the Right Balance: Relevance and Diversity in LLM Retrieval},
  author = {Brouillette, Guillaume and Kagabo, Faustin and Faghihi, Usef and Ghazzali, Nadia},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
