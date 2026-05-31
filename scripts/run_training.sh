#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/classification_research.yaml}
RESUME=${2:-}

CMD=(python -m src.main --config "$CONFIG")
if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
