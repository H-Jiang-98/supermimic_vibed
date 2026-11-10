#!/bin/bash
# Usage: ./run_eval.sh [--config FILE] [RUN_NAME] [evaluate.py args...]
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
CONFIG_ARGS=()
if [[ ${1:-} == --config ]]; then
  CONFIG_ARGS=(--config "${2:?--config requires a file}")
  shift 2
elif [[ ${1:-} == --config=* ]]; then
  CONFIG_ARGS=(--config "${1#--config=}")
  shift
fi
RUN=lafan1_main
if (( $# )) && [[ $1 != --* ]]; then RUN=$1; shift; fi
if [[ ! "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Run name may contain only letters, digits, underscores and hyphens." >&2
  exit 2
fi
LOGS_ROOT=$(python3 "$REPO/supermimic/config.py" logs_dir "${CONFIG_ARGS[@]}")
exec python3 "$REPO/tools/run.py" "${CONFIG_ARGS[@]}" supermimic/evaluate.py \
  --run_dir "$LOGS_ROOT/$RUN" "$@"
