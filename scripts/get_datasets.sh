#!/usr/bin/env bash
# Download the public datasets used as research "tenants" into data/datasets/ (gitignored).
# Licences: GSM8K MIT, Dolly-15k CC BY-SA 3.0, MBPP CC BY 4.0, BFCL Apache-2.0.
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p data/datasets && cd data/datasets
H=https://huggingface.co/datasets
curl -fsSL -o gsm8k_test.parquet $H/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
curl -fsSL -o dolly.jsonl $H/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
curl -fsSL -o mbpp_test.parquet $H/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet
curl -fsSL -o bfcl_simple.json $H/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_simple.json
ls -la
