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
  run-canary-1b-flash-validation.sh --self-test

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

canonical_absent_path() {
  local target="$1" lexical current="/" component suffix="" real
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || die "work path contains .."
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || die "work path contains a symlinked ancestor"
  done
  current="$target"
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || die "work path parent is missing or symlinked"
  real="$(cd -P "$current" 2>/dev/null && pwd)" || die "work path parent is inaccessible"
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_work_dir() {
  local work="$1" input="$2" candidate root_real input_parent input_real
  [[ ! -e "$work" && ! -L "$work" ]] || die "work directory must be absent"
  candidate="$(canonical_absent_path "$work")"
  root_real="$(cd -P "$PWD" && pwd)"
  input_parent="$(cd -P "$(dirname "$input")" 2>/dev/null && pwd)" || die "input parent is inaccessible"
  input_real="$input_parent/$(basename "$input")"
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || die "work directory overlaps checkout"
  [[ "$candidate" != "$input_real" && "$candidate/" != "$input_real/"* && "$input_real/" != "$candidate/"* ]] || die "work directory overlaps checkpoint"
}

UPSTREAM_REPO="nvidia/canary-1b-flash"
UPSTREAM_REVISION="2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e"
MODEL_KIND="canary-1b-flash"
PARITY_TEST="released_checkpoint_matches_official_nemo_greedy_tokens"
GGUF_ENV="VOKRA_CANARY_REAL_GGUF"
REFERENCE_PCM_ENV="VOKRA_CANARY_REFERENCE_PCM"
REFERENCE_TOKENS_ENV="VOKRA_CANARY_REFERENCE_TOKENS"
SOURCE_LANGUAGE_ENV="VOKRA_CANARY_SOURCE_LANGUAGE"
TARGET_LANGUAGE_ENV="VOKRA_CANARY_TARGET_LANGUAGE"

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp fail=0 cases=0 required
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$MODEL_KIND" "$PARITY_TEST" \
    "$GGUF_ENV" "$REFERENCE_PCM_ENV" "$REFERENCE_TOKENS_ENV" \
    "$SOURCE_LANGUAGE_ENV" "$TARGET_LANGUAGE_ENV" \
    "tools/parity/canary_1b_flash_prepare_checkpoint.py" \
    "tools/parity/canary_1b_flash_dump_reference.py" \
    "--frozen --project tools/parity --python 3.12 python" \
    "--target-language de"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-canary-1b-flash-validation: self-test FAIL: contract lost token: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-canary-1b-flash-validation: self-test FAIL: direct Python/pip command found" >&2
    fail=1
  fi
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-canary-1b-flash-validation: self-test FAIL: publication command found" >&2
    fail=1
  fi

  cases=$((cases + 1))
  for required in 'uname -s' 'VOKRA_PUBLISH_ON_VAST' 'git status --porcelain --untracked-files=all' \
    'cargo fmt --all -- --check' 'cargo test --locked --workspace' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-canary-1b-flash-validation: self-test FAIL: fail-closed guard lost token: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$tmp/nonempty" >/dev/null 2>&1; then
    echo "run-canary-1b-flash-validation: self-test FAIL: extra self-test argument accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo >/dev/null 2>&1; then
    echo "run-canary-1b-flash-validation: self-test FAIL: missing --nemo value accepted" >&2
    fail=1
  fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    echo "run-canary-1b-flash-validation: self-test FAIL: unknown argument accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo "$tmp/a" --nemo "$tmp/b" >/dev/null 2>&1; then
    echo "run-canary-1b-flash-validation: self-test FAIL: duplicate --nemo accepted" >&2
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-canary-1b-flash-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

nemo_path=""
work_dir="/workspace/vokra-canary-validation"
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test)
      self_test=1
      shift
      ;;
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

if [[ $self_test -eq 1 ]]; then
  [[ -z "$nemo_path" && "$work_dir" == "/workspace/vokra-canary-validation" ]] \
    || die "--self-test accepts no other arguments"
  run_self_test
  exit $?
fi

[[ "$(uname -s)" == "Linux" ]] || die "actual validation is Linux/VAST-only"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
  || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
[[ -n "$nemo_path" ]] || die "--nemo is required"
[[ -f "$nemo_path" && ! -L "$nemo_path" ]] || die "checkpoint is not a regular non-symlink file: $nemo_path"
[[ -f Cargo.toml && -d crates/vokra-models ]] \
  || die "run from the Vokra repository root"
require_absent_work_dir "$work_dir" "$nemo_path"

# Fail before the multi-gigabyte checkpoint is unpacked if the verification
# host is missing a tool. `provision.sh` installs the minimal Rust profile, so
# rustfmt/clippy are an explicit VAST setup step rather than an assumption.
for command in rustfmt cargo-deny cargo-audit; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required VAST verification tool is missing: $command"
done
cargo clippy --version >/dev/null 2>&1 \
  || die "the clippy component is missing; install rustfmt/clippy on the VAST host"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
  || die "worktree changes or untracked files are present; validate a clean committed git-bundle checkpoint"

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
  --model "$MODEL_KIND" \
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

export "$GGUF_ENV=$work_dir/canary-1b-flash.gguf"
export "$REFERENCE_PCM_ENV=$evidence_dir/reference-en-en.pcm.f32"
export "$REFERENCE_TOKENS_ENV=$evidence_dir/reference-en-en.tokens.txt"
export "$SOURCE_LANGUAGE_ENV=en"
export "$TARGET_LANGUAGE_ENV=en"
run_logged cargo test --locked -p vokra-models \
  "$PARITY_TEST" -- --ignored

# A different target language changes the Canary2 prompt and exercises AST,
# so its exact token sequence is a separate independent-oracle gate rather
# than being inferred from an English-ASR pass.
export "$REFERENCE_PCM_ENV=$evidence_dir/reference-en-de.pcm.f32"
export "$REFERENCE_TOKENS_ENV=$evidence_dir/reference-en-de.tokens.txt"
export "$TARGET_LANGUAGE_ENV=de"
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
