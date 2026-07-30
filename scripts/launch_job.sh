#!/usr/bin/env bash
# Upload the current code snapshot to the Hub and launch stage-1
# training on HF Jobs.
#
#   scripts/launch_job.sh [flavor] [extra train.py args...]
#   scripts/launch_job.sh l4x1 --epochs 60 --batch 48
#
# List hardware+prices: hf jobs hardware
set -euo pipefail
cd "$(dirname "$0")/.."

FLAVOR="${1:-l4x1}"
shift || true
ARGS="${*:---epochs 60 --batch 48}"

CODE_REPO="${SSTVAE_CODE_REPO:-arodland/sstvae-code}"
TIMEOUT="${SSTVAE_TIMEOUT:-24h}"

echo "== uploading code snapshot to $CODE_REPO"
hf repos create "$CODE_REPO" --private --exist-ok >/dev/null
hf upload "$CODE_REPO" . . \
    --include "sstvae/**" --include "scripts/train.py" \
    --include "scripts/train_refiner.py" \
    --include "scripts/train_job.py" --include "scripts/export_onnx.py" \
    --include "pyproject.toml" \
    --exclude "**/__pycache__/**" \
    --commit-message "code snapshot for training job" --quiet

echo "== launching job (flavor=$FLAVOR, args: $ARGS)"
hf jobs uv run scripts/train_job.py \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --detach \
    --secrets HF_TOKEN \
    --env SSTVAE_CODE_REPO="$CODE_REPO" \
    --env SSTVAE_ARGS="$ARGS" \
    ${SSTVAE_SCRIPT:+--env SSTVAE_SCRIPT="$SSTVAE_SCRIPT"} \
    ${SSTVAE_OUT_REPO:+--env SSTVAE_OUT_REPO="$SSTVAE_OUT_REPO"} \
    ${SSTVAE_RESUME:+--env SSTVAE_RESUME=1}

echo "== follow logs with: hf jobs logs --follow <job id>"
