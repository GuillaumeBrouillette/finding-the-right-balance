# Qwen3.8-27B generator validation

This protocol reproduces the modern-reader check on HotpotQA and
2WikiMultiHopQA. The checkpoint, prompt, decoding settings, runtime, and raw
answer predictions are frozen in the retained-result artifact.

## Recorded reader server

The completed runs used the NVIDIA vLLM 26.08 environment on an NVIDIA DGX
Spark GB10. The local evaluation image had image ID
`sha256:94aadc4f880edb4d72a44c2fca17280b363fd5965fec7a8442b5fcfa5e4d2828`
and NVIDIA build ID `409447798`. Its recorded packages were vLLM
`0.27.1+93523f72.nv26.8.64249418`, Transformers `5.14.1`, PyTorch
`2.14.0a0+4fdf77b940.nv26.8.63802676`, Python `3.12.3`, and CUDA runtime
`13.4`.

Start the server with the settings used by the experiment:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name Qwen/Qwen3.8-27B \
  --language-model-only \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.60 \
  --max-num-seqs 8 \
  --seed 0
```

The client fixes temperature `0`, top-p `1`, seed `0`, 64 output tokens,
disables Qwen thinking, and uses prompt version `short_direct_v1`:

```text
Answer the question based on the context. Give only a short, direct answer.
Question: {question}
Context: {title_1}: {text_1} | ... | {title_k}: {text_k}
```

## Clean/heavy RQ5 check

```bash
bash reproducibility/run_qwen38_generation.sh
```

This evaluates k-NN and MMR(0.7) at `rho = 0, 1`. Reconstruct the frozen
rule (`tau = 2`, fallback MMR(0.7)) for each run:

```bash
python -m analysis.analyze_regimes \
  --per_query RUN_DIR/results_redundancy_per_query.csv \
  --objective S-Recall@k --freeze_tau 2 --freeze_d 'MMR(0.7)'
```

Then reconstruct the table:

```bash
python reproducibility/build_qwen38_table.py \
  --hotpot RUN_HOTPOT/analysis_generation.csv \
  --twowiki RUN_2WIKI/analysis_generation.csv \
  --output results/qwen38_generation/table_generation_qwen38.tex
```

## Eight-method clean-pool RQ1 check

The RQ1 extension reuses the compatible k-NN and MMR(0.7) answers from the
HotpotQA clean/heavy run and generates the six missing method outputs:

```bash
bash reproducibility/run_qwen38_rq1_hotpot.sh
```

Validate and reconstruct it with:

```bash
python reproducibility/build_qwen38_rq1_table.py \
  --per-query RUN_HOTPOT/results_redundancy_gen_per_query.csv \
  --run-params RUN_HOTPOT/run_params.json \
  --split-manifest manifests/splits/hotpotqa_fullwiki.csv \
  --partition test \
  --split-seed 0 \
  --output-csv RUN_HOTPOT/analysis_generation_rq1_qwen38_test.csv \
  --output-tex RUN_HOTPOT/table_rq1_qwen38_test.tex
```

This reports the 5,924 seed-0 test queries used by the paper's other Table 2
columns without replacing the archived all-7,405-query audit files. Omit the
three split arguments only when reproducing that original all-query audit.

Generation CSVs are atomically checkpointed every 256 answers. Resume an
interrupted run with `--resume_run_dir` and unchanged arguments. The builder
rejects incomplete methods, duplicate query-method rows, unexpected settings,
and source files whose structure does not match the frozen protocol.

## Provenance limitation

The RQ1 run records hashes of the evaluator, generator, and run utility. The
two earlier clean/heavy runs record their complete runtime and model settings
but have `git_commit: unknown` and predate source-script hash capture. Their
raw outputs are checksummed in the retained-result artifact; this limitation
must remain visible in the execution manifest.
