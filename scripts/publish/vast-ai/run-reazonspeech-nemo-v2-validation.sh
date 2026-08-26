#!/usr/bin/env bash
# VAST-only validation worker for ReazonSpeech NeMo v2.
# It never uploads, publishes, pushes Git refs, stops, or destroys an instance.
# Pull the small evidence/reference files, then destroy the instance externally.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run-reazonspeech-nemo-v2-validation.sh --nemo <reazonspeech-nemo-v2.nemo> \
    [--work-dir /workspace/vokra-reazonspeech-validation]

Requires Linux and VOKRA_PUBLISH_ON_VAST=1 from provision.sh, plus the
rustfmt/clippy components and cargo-deny/cargo-audit executables. Produces a
complete local GGUF, an official NeMo encoder/token reference, exact CPU parity
evidence, and Rust verification logs. It performs no Hugging Face upload.
EOF
}

die() {
  echo "run-reazonspeech-nemo-v2-validation: $*" >&2
  exit 1
}

nemo_path=""
work_dir="/workspace/vokra-reazonspeech-validation"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nemo)
      [[ $# -ge 2 ]] || die "--nemo requires a path"
      nemo_path="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a path"
      work_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || die "actual validation is Linux/VAST-only"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
  || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
[[ -n "$nemo_path" ]] || die "--nemo is required"
[[ -f "$nemo_path" ]] || die "checkpoint is not a regular file: $nemo_path"
[[ -f Cargo.toml && -d crates/vokra-models ]] \
  || die "run from the Vokra repository root"

for command in rustfmt cargo-deny cargo-audit; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required VAST verification tool is missing: $command"
done
cargo clippy --version >/dev/null 2>&1 \
  || die "the clippy component is missing; install rustfmt/clippy on the VAST host"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || die "worktree changes or untracked files are present; validate a clean committed git-bundle checkpoint"

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
nemo_path="$(cd "$(dirname "$nemo_path")" && pwd)/$(basename "$nemo_path")"
log_path="$work_dir/validation.log"
evidence_dir="$work_dir/evidence"
prepared_dir="$work_dir/prepared"
reference_dir="$evidence_dir/reference"
mkdir -p "$evidence_dir" "$prepared_dir" "$reference_dir"

run_logged() {
  echo "+ $*" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
export RUST_BACKTRACE=1

run_logged cargo fmt --all -- --check
run_logged bash scripts/check-forbidden-symbols.sh
run_logged bash scripts/check-zero-deps.sh
run_logged bash scripts/check-bound-arch-coverage.sh

run_logged uv run --frozen --project tools/parity --python 3.12 python \
  tools/parity/reazonspeech_nemo_v2_prepare_checkpoint.py \
  --input "$nemo_path" --output-dir "$prepared_dir"

run_logged cargo build --locked --release -p vokra-convert -p vokra-cli
run_logged target/release/vokra-cli convert \
  --model reazonspeech-nemo-v2 \
  --input "$prepared_dir/reazonspeech-nemo-v2.prepared.safetensors" \
  --tokenizer "$prepared_dir/tokenizer.vocab" \
  --output "$work_dir/reazonspeech-nemo-v2.gguf"

run_logged uv run --frozen --project tools/parity --extra titanet --python 3.12 python \
  tools/parity/reazonspeech_nemo_v2_dump_reference.py \
  --nemo "$nemo_path" --output-dir "$reference_dir"

export VOKRA_REAZONSPEECH_NEMO_V2_GGUF="$work_dir/reazonspeech-nemo-v2.gguf"
export VOKRA_REAZONSPEECH_NEMO_V2_REFERENCE_DIR="$reference_dir"
run_logged cargo test --locked -p vokra-models \
  --test parity_reazonspeech_nemo_v2 \
  released_cpu_encoder_and_tokens_match_official_nemo -- --nocapture

run_logged target/release/vokra-cli run \
  --model "$work_dir/reazonspeech-nemo-v2.gguf" \
  --input tests/fixtures/audio/jfk-30s.wav --backend cpu

run_logged cargo test --locked --workspace
run_logged cargo clippy --locked --workspace --all-targets -- -D warnings
run_logged cargo deny check licenses advisories bans
run_logged cargo audit

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "kernel=$(uname -srmo)"
  echo "cpu=$(awk -F ': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
  echo "nemo_sha256=$(sha256sum "$nemo_path" | awk '{print $1}')"
  echo "gguf_sha256=$(sha256sum "$work_dir/reazonspeech-nemo-v2.gguf" | awk '{print $1}')"
  echo "reference_sha256=$(sha256sum "$reference_dir/reference.json" | awk '{print $1}')"
  echo "reference_encoder_sha256=$(sha256sum "$reference_dir/encoder.f32" | awk '{print $1}')"
  echo "reference_tokens_sha256=$(sha256sum "$reference_dir/tokens.u32" | awk '{print $1}')"
  echo "verdict=PASS"
} > "$evidence_dir/validation-summary.txt"

cp "$prepared_dir/prepare-audit.json" "$evidence_dir/prepare-audit.json"
echo "run-reazonspeech-nemo-v2-validation: PASS"
echo "Pull before destroy: $evidence_dir and $log_path"
echo "Do not pull the multi-GB .nemo/.safetensors/.gguf artifacts to the maintainer Mac."
