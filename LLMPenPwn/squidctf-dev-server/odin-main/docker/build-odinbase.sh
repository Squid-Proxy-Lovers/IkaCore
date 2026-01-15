#!/usr/bin/env bash
set -euo pipefail

mode="${1:-base}"

case "$mode" in
  base)
    docker build --target base -t odin-base:latest .
    ;;
  sage)
    docker build --target sage -t odin-base:sage .
    ;;
  all)
    docker build --target base -t odin-base:latest .
    docker build --target sage -t odin-base:sage .
    ;;
  *)
    echo "unknown mode: $mode (expected: base | sage | all)" >&2
    exit 1
    ;;
esac