#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 MODEL.gguf OUTPUT_DIR [--float32] [--keep-package]" >&2
  echo "OUTPUT_DIR must exist and be empty; convention: MODEL.gguf.coreml" >&2
  echo "--float32 is diagnostic-only and production Whisper ASR rejects it" >&2
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 2
fi

gguf_path="$1"
output_dir="$2"
shift 2

if [[ ! -f "$gguf_path" ]]; then
  echo "error: GGUF path is not a regular file: $gguf_path" >&2
  exit 2
fi
if [[ ! -d "$output_dir" ]]; then
  echo "error: output directory does not exist: $output_dir" >&2
  exit 2
fi
if [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: output directory must be empty: $output_dir" >&2
  exit 2
fi

precision="float16"
keep_package=0
for option in "$@"; do
  case "$option" in
    --float32) precision="float32" ;;
    --keep-package) keep_package=1 ;;
    *)
      echo "error: unknown option: $option" >&2
      usage
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$keep_package" -eq 1 ]]; then
  uv run --project "$script_dir" \
    "$script_dir/generate_whisper_encoder.py" \
    --gguf "$gguf_path" \
    --output-dir "$output_dir" \
    --compute-precision "$precision" \
    --keep-package
else
  uv run --project "$script_dir" \
    "$script_dir/generate_whisper_encoder.py" \
    --gguf "$gguf_path" \
    --output-dir "$output_dir" \
    --compute-precision "$precision"
fi
