#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MODEL.mlmodelc [MIN_ANE_RATE]" >&2
  exit 2
fi

model="$1"
minimum="${2:-0.90}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-coreml-placement.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

xcrun swiftc \
  -O \
  -parse-as-library \
  "$script_dir/inspect_placement.swift" \
  -o "$build_dir/inspect-placement"
"$build_dir/inspect-placement" "$model" "$minimum"
