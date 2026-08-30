#!/usr/bin/env bash
# VAST-only official-reference evidence worker for native BiCodec decode.
#
# This worker does not upload weights or select tolerances. The Python dumper
# authenticates the exact Spark-TTS source/checkpoint/config and records the
# official semantic latent, d-vector, prenet output, and waveform for manager
# review before any Rust parity gate is chosen.
set -euo pipefail

die() {
  echo "run-bicodec-native-parity: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  run-bicodec-native-parity.sh --source-dir <checkout> --model-dir <BiCodec> \
    --output <empty-tmpfs-dir>

Requires Linux x86_64, VAST, and the repository Python environment. Inputs are
authenticated by bicodec_dump_reference.py. The worker never publishes or
uploads model artifacts.
EOF
}

source_dir=""
model_dir=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) [[ $# -ge 2 ]] || die "--source-dir requires a path"; source_dir="$2"; shift 2 ;;
    --model-dir) [[ $# -ge 2 ]] || die "--model-dir requires a path"; model_dir="$2"; shift 2 ;;
    --output) [[ $# -ge 2 ]] || die "--output requires a path"; output="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$source_dir" && -n "$model_dir" && -n "$output" ]] || { usage >&2; exit 1; }
[[ "$(uname -s)" == "Linux" ]] || die "official reference runs on Linux/VAST only"
[[ "$(uname -m)" == "x86_64" ]] || die "official reference requires x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -f Cargo.toml && -d tools/parity ]] || die "run from a Vokra checkout"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
command -v uv >/dev/null 2>&1 || die "uv is required"
command -v findmnt >/dev/null 2>&1 || die "findmnt is required"
command -v cargo >/dev/null 2>&1 || die "cargo is required"
[[ "$(findmnt -T "$(dirname "$output")" -no FSTYPE 2>/dev/null || true)" == "tmpfs" ]] \
  || die "output parent must be tmpfs/RAM disk"

mkdir -p "$output"
[[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "output must be empty"
uv run --frozen --project tools/parity --python 3.12 python \
  tools/parity/bicodec_dump_reference.py \
  --source-dir "$source_dir" --model-dir "$model_dir" --output "$output"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
cargo build --locked --release -p vokra-cli
gguf_path="$output/bicodec.gguf"
target/release/vokra-cli convert \
  --model bicodec \
  --input "$model_dir/model.safetensors" \
  --output "$gguf_path" \
  --license cc-by-nc-sa-4.0
[[ -s "$gguf_path" ]] || die "authenticated BiCodec conversion produced no GGUF"
VOKRA_BICODEC_PARITY_GGUF="$gguf_path" \
VOKRA_BICODEC_PARITY_REFERENCE="$output" \
  cargo test --locked -p vokra-models \
    bicodec::tests::official_reference_report_only -- --ignored --nocapture
echo "BiCodec official reference evidence: $output"
