#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec torchrun --standalone --nproc_per_node "${NUM_PROCESSES:-1}" \
  --module DyneTrion.train "$@"
