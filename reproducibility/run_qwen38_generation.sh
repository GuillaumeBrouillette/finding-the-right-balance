#!/usr/bin/env bash
set -euo pipefail

# The vLLM server must already be running; see QWEN38_GENERATION.md.
api_base="${QWEN_API_BASE:-http://127.0.0.1:8000/v1}"
output_dir="${QWEN_OUTPUT_DIR:-results/qwen38_generation}"
revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

cd "$(dirname "$0")/.."

common=(
  --experiment redundancy
  --split validation
  --max_samples all
  --encoder_model bge-m3
  --device cuda
  --batch_size 64
  --top_m 100
  --top_k 5
  --objective S-Recall@k
  --rho_grid 0 1
  --dup_noise light
  --dup_target mixed
  --seed 0
  --run_generation
  --generator_model qwen3.8-27b
  --generator_backend openai-compatible
  --generator_api_base "$api_base"
  --generator_revision "$revision"
  --generator_max_new_tokens 64
  --generator_num_beams 1
  --generator_batch_size 8
  --generator_timeout 300
  --generation_checkpoint_every 256
  --gen_methods kNN 'MMR(0.7)'
  --gen_max_samples all
  --output_dir "$output_dir"
)

python -m experiments.evaluate_redundancy --dataset hotpotqa_fullwiki "${common[@]}"
python -m experiments.evaluate_redundancy --dataset 2wikimultihopqa "${common[@]}"
