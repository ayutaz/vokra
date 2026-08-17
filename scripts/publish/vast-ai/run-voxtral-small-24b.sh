#!/usr/bin/env bash
# Reproducible VAST-only conversion/publish for Voxtral-Small-24B-2507.
#
# The upstream repository contains both an 11-shard checkpoint and a duplicate
# 48 GB consolidated.safetensors. Exact allow-patterns below fetch only the
# sharded copy, and the immutable revision keeps a long-running conversion
# reproducible. run-one.sh remains dry-run by default; pass --push explicitly.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$here/run-one.sh" \
  --hf-repo mistralai/Voxtral-Small-24B-2507 \
  --revision da5b42409f279fdd92febee0511a6c32828569c1 \
  --vokra-slug voxtral-small-24b-2507 \
  --model-kind voxtral \
  --license-spdx apache-2.0 \
  --include 'model-*.safetensors' \
  --include model.safetensors.index.json \
  --include config.json \
  --include generation_config.json \
  --include params.json \
  --include preprocessor_config.json \
  --include tekken.json \
  --input-name model.safetensors.index.json \
  --config-name config.json \
  --tokenizer-name tekken.json \
  --adapter-config "$here/configs/voxtral-small-24b-2507.adapter.json" \
  --expect-adapter-kind frame_stack_mlp \
  --expect-model-name voxtral-small-24b \
  --expect-source 'mistralai/Voxtral-Small-24B-2507 (Apache-2.0)' \
  "$@"
