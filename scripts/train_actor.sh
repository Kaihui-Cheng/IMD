#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec accelerate launch \
  --num_processes "${NUM_PROCESSES:-1}" --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  --module DyneTrion.supervised_pretrain "$@"
