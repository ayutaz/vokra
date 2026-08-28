#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 OUTPUT_DIR" >&2
  echo "OUTPUT_DIR must exist and be empty; writes tiny-encoder.{mlpackage,mlmodelc}." >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

output_dir="$1"
if [[ ! -d "$output_dir" ]]; then
  echo "error: output directory does not exist: $output_dir" >&2
  exit 2
fi
if [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: output directory must be empty: $output_dir" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
uv run --project "$script_dir" \
  "$script_dir/generate_test_model.py" \
  --output "$output_dir/tiny-encoder.mlpackage"
xcrun coremlcompiler compile \
  "$output_dir/tiny-encoder.mlpackage" \
  "$output_dir"

test -d "$output_dir/tiny-encoder.mlmodelc"
echo "$output_dir/tiny-encoder.mlmodelc"
