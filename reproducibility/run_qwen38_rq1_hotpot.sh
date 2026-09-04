#!/usr/bin/env bash
set -euo pipefail

api_base="${QWEN_API_BASE:-http://127.0.0.1:8000/v1}"
output_dir="${QWEN_OUTPUT_DIR:-results/qwen38_rq1_hotpot}"
revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
source_csv="${QWEN_REUSE_CSV:-results/qwen38_generation/2026-08-31_163327_redundancy_hotpotqa_fullwiki/results_redundancy_gen_per_query.csv}"

cd "$(dirname "$0")/.."

python -m experiments.evaluate_redundancy \
  --experiment redundancy \
  --dataset hotpotqa_fullwiki \
  --split validation \
  --max_samples all \
  --encoder_model bge-m3 \
  --device cuda \
  --batch_size 64 \
  --top_m 100 \
  --top_k 5 \
  --objective S-Recall@k \
  --rho_grid 0 \
  --dup_noise light \
  --dup_target mixed \
  --seed 0 \
  --run_generation \
  --generator_model qwen3.8-27b \
  --generator_backend openai-compatible \
  --generator_api_base "$api_base" \
  --generator_revision "$revision" \
  --generator_max_new_tokens 64 \
  --generator_num_beams 1 \
  --generator_batch_size 8 \
  --generator_timeout 300 \
  --generation_checkpoint_every 256 \
  --gen_methods kNN 'MMR(0.3)' 'MMR(0.5)' 'MMR(0.7)' 'MMR(0.9)' Maxmin Greedy-DPP 'RNG(0.2)' \
  --gen_max_samples all \
  --reuse_generation_csv "$source_csv" \
  --output_dir "$output_dir"
