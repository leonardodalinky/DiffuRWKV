#!/usr/bin/env bash
# End-to-end: HF tulu-3-sft-mixture -> JSONL (RWKV-v7 G1x template) -> binidx (.bin / .idx)
# Forwards extra args to convert_tulu3_to_jsonl.py (e.g. --limit 100 for smoke testing).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="$REPO_ROOT/train/data"
TOOL_DIR="$REPO_ROOT/third-party/json2binidx_tool"
JSONL_PATH="$DATA_DIR/tulu3.jsonl"
BINIDX_PREFIX="$DATA_DIR/tulu3"
VOCAB="$TOOL_DIR/rwkv_vocab_v20230424.txt"

mkdir -p "$DATA_DIR"

echo "[1/2] HF -> JSONL: $JSONL_PATH"
uv run --group data python "$HERE/convert_tulu3_to_jsonl.py" \
    --output "$JSONL_PATH" "$@"

echo "[2/2] JSONL -> binidx: $BINIDX_PREFIX.bin / .idx"
uv run --group data python "$TOOL_DIR/tools/preprocess_data.py" \
    --input "$JSONL_PATH" \
    --output-prefix "$BINIDX_PREFIX" \
    --vocab "$VOCAB" \
    --dataset-impl mmap \
    --tokenizer-type RWKVTokenizer \
    --append-eod

echo "done. Train with --data_file $BINIDX_PREFIX --vocab_size 65536 (diffusion mode auto-extends to 65537)"
