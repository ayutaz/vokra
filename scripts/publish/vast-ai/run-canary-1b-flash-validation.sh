#!/usr/bin/env bash
# VAST-only validation worker for the complete NVIDIA Canary-1B-Flash release.
# It never uploads, publishes, pushes Git refs, or destroys the instance.
# The caller pulls the small report/reference files, then destroys the instance.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run-canary-1b-flash-validation.sh --nemo <canary-1b-flash.nemo> \
    [--work-dir /workspace/vokra-canary-validation]

Requires Linux and VOKRA_PUBLISH_ON_VAST=1 from provision.sh, plus the
rustfmt/clippy components and cargo-deny/cargo-audit executables. Produces a
complete local GGUF, official NeMo references, CPU parity evidence, and Rust
verification logs. It performs no Hugging Face upload.
EOF
}

die() {
  echo "run-canary-1b-flash-validation: $*" >&2
  exit 1
}

nemo_path=""
work_dir="/workspace/vokra-canary-validation"
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

# Fail before the multi-gigabyte checkpoint is unpacked if the verification
# host is missing a tool. `provision.sh` installs the minimal Rust profile, so
# rustfmt/clippy are an explicit VAST setup step rather than an assumption.
for command in rustfmt cargo-deny cargo-audit; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required VAST verification tool is missing: $command"
done
cargo clippy --version >/dev/null 2>&1 \
  || die "the clippy component is missing; install rustfmt/clippy on the VAST host"
git diff --quiet \
  || die "tracked worktree changes are present; validate a committed git-bundle checkpoint"
git diff --cached --quiet \
  || die "staged worktree changes are present; validate a committed git-bundle checkpoint"

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
nemo_path="$(cd "$(dirname "$nemo_path")" && pwd)/$(basename "$nemo_path")"
log_path="$work_dir/validation.log"
evidence_dir="$work_dir/evidence"
prepared_dir="$work_dir/prepared"
mkdir -p "$evidence_dir" "$prepared_dir"

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
  tools/parity/canary_1b_flash_prepare_checkpoint.py \
  --input "$nemo_path" --output-dir "$prepared_dir"

run_logged cargo build --locked --release -p vokra-cli
run_logged target/release/vokra-cli convert \
  --model canary-1b-flash \
  --input "$prepared_dir/canary-1b-flash.prepared.safetensors" \
  --tokenizer "$prepared_dir/canary-1b-flash.aggregate.vocab" \
  --output "$work_dir/canary-1b-flash.gguf"

run_logged uv run --frozen --project tools/parity --extra titanet --python 3.12 python \
  tools/parity/canary_1b_flash_dump_reference.py \
  --nemo "$nemo_path" \
  --source-language en --target-language en \
  --output "$evidence_dir/reference-en-en.json"
run_logged uv run --frozen --project tools/parity --extra titanet --python 3.12 python \
  tools/parity/canary_1b_flash_dump_reference.py \
  --nemo "$nemo_path" \
  --source-language en --target-language de \
  --output "$evidence_dir/reference-en-de.json"

export VOKRA_CANARY_REAL_GGUF="$work_dir/canary-1b-flash.gguf"
export VOKRA_CANARY_REFERENCE_PCM="$evidence_dir/reference-en-en.pcm.f32"
export VOKRA_CANARY_REFERENCE_TOKENS="$evidence_dir/reference-en-en.tokens.txt"
export VOKRA_CANARY_SOURCE_LANGUAGE=en
export VOKRA_CANARY_TARGET_LANGUAGE=en
run_logged cargo test --locked -p vokra-models \
  released_checkpoint_matches_official_nemo_greedy_tokens -- --ignored

# A different target language changes the Canary2 prompt and exercises AST,
# so its exact token sequence is a separate independent-oracle gate rather
# than being inferred from an English-ASR pass.
export VOKRA_CANARY_REFERENCE_PCM="$evidence_dir/reference-en-de.pcm.f32"
export VOKRA_CANARY_REFERENCE_TOKENS="$evidence_dir/reference-en-de.tokens.txt"
export VOKRA_CANARY_TARGET_LANGUAGE=de
run_logged cargo test --locked -p vokra-models \
  released_checkpoint_matches_official_nemo_greedy_tokens -- --ignored

run_logged target/release/vokra-cli run \
  --model "$work_dir/canary-1b-flash.gguf" \
  --input tests/fixtures/audio/jfk-30s.wav \
  --backend cpu --language en --target-language en
run_logged target/release/vokra-cli run \
  --model "$work_dir/canary-1b-flash.gguf" \
  --input tests/fixtures/audio/jfk-30s.wav \
  --backend cpu --language en --target-language de

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
  echo "gguf_sha256=$(sha256sum "$work_dir/canary-1b-flash.gguf" | awk '{print $1}')"
  echo "reference_en_en_sha256=$(sha256sum "$evidence_dir/reference-en-en.json" | awk '{print $1}')"
  echo "reference_en_de_sha256=$(sha256sum "$evidence_dir/reference-en-de.json" | awk '{print $1}')"
  echo "verdict=PASS"
} > "$evidence_dir/validation-summary.txt"

cp "$prepared_dir/prepare-audit.json" "$evidence_dir/prepare-audit.json"
echo "run-canary-1b-flash-validation: PASS"
echo "Pull before destroy: $evidence_dir and $log_path"
echo "Do not pull the multi-GB .nemo/.safetensors/.gguf artifacts to the maintainer Mac."
