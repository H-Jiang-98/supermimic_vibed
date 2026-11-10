#!/bin/bash
# Usage: [CPUS=0-23] ./run_train.sh [--config FILE] [RUN_NAME] [train.py args...]
# Runs in the foreground; use tmux to keep a run alive after disconnecting.
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
PIN=()
if [[ -n ${CPUS:-} ]]; then PIN=(taskset -c "$CPUS"); fi
LOGS_ROOT=$(python3 "$REPO/supermimic/config.py" logs_dir "${CONFIG_ARGS[@]}")
LOG_DIR="$LOGS_ROOT/$RUN"
EXPECT_LOG_DIR=false
for ARG in "$@"; do
  if $EXPECT_LOG_DIR; then
    LOG_DIR=$ARG
    EXPECT_LOG_DIR=false
  elif [[ $ARG == --log_dir ]]; then
    EXPECT_LOG_DIR=true
  elif [[ $ARG == --log_dir=* ]]; then
    LOG_DIR=${ARG#--log_dir=}
  fi
done
if $EXPECT_LOG_DIR || [[ -z $LOG_DIR ]]; then
  echo "--log_dir requires a directory." >&2
  exit 2
fi
[[ $LOG_DIR == /* ]] || LOG_DIR="$REPO/$LOG_DIR"
mkdir -p "$LOG_DIR"
"${PIN[@]}" python3 "$REPO/tools/run.py" "${CONFIG_ARGS[@]}" supermimic/train.py \
  --log_dir "$LOG_DIR" "$@" 2>&1 | tee -a "$LOG_DIR/stdout.log"
