#!/usr/bin/env bash
# VAST-only validation worker for the complete NVIDIA Canary-1B-v2 release.
# It never uploads, publishes, pushes Git refs, stops, or destroys an instance.
# Pull the small evidence directory and log, then destroy the instance.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run-canary-1b-v2-validation.sh --nemo <canary-1b-v2.nemo> \
    --approval-evidence <owner-approval.json> \
    [--work-dir /workspace/vokra-canary-v2-validation]
  run-canary-1b-v2-validation.sh --self-test

Requires Linux and VOKRA_PUBLISH_ON_VAST=1 from provision.sh, plus the
rustfmt/clippy components and cargo-deny/cargo-audit executables. Produces a
complete local GGUF, official NeMo references, exact CPU-token parity evidence,
and Rust verification logs. It performs no Hugging Face upload.
EOF
}

die() {
  echo "run-canary-1b-v2-validation: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PREFLIGHT_GATE="$REPO_ROOT/tools/parity/canary_1b/preflight_gate.py"
PREFLIGHT_MANIFEST="$REPO_ROOT/tools/parity/canary_1b/license_gate_manifest.json"

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

UPSTREAM_REPO="nvidia/canary-1b-v2"
UPSTREAM_REVISION="87bc52657add533cd0156b3fc1aef027280754bf"
MODEL_KIND="canary"
PARITY_TEST="canary::tests::canary_v2_released_checkpoint_matches_official_nemo_greedy_tokens"
GGUF_ENV="VOKRA_CANARY_V2_REAL_GGUF"
REFERENCE_PCM_ENV="VOKRA_CANARY_V2_REFERENCE_PCM"
REFERENCE_TOKENS_ENV="VOKRA_CANARY_V2_REFERENCE_TOKENS"
SOURCE_LANGUAGE_ENV="VOKRA_CANARY_V2_SOURCE_LANGUAGE"
TARGET_LANGUAGE_ENV="VOKRA_CANARY_V2_TARGET_LANGUAGE"
VARIANT="canary-1b-v2"
ARCHIVE_BYTES=6358958080
ARCHIVE_SHA256="ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094"
MAIN_CHECKPOINT_MEMBER="./model_weights.ckpt"
MAIN_CHECKPOINT_BYTES=3853798427

license_preflight() {
  local approval="$1"
  [[ -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && \
    -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] \
    || die "Canary-1B approval gate or manifest is missing or symlinked"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$PREFLIGHT_GATE" --manifest "$PREFLIGHT_MANIFEST" \
    --approval "$approval" --variant "$VARIANT" \
    || die "Canary-1B-v2 approval preflight is unresolved"
}

verify_archive() {
  local path="$1" actual_bytes actual_sha
  [[ -f "$path" && ! -L "$path" ]] \
    || die "checkpoint is not a regular non-symlink file: $path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$ARCHIVE_BYTES" ]] \
    || die "Canary-1B-v2 archive byte count $actual_bytes != $ARCHIVE_BYTES"
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual_sha" == "$ARCHIVE_SHA256" ]] \
    || die "Canary-1B-v2 archive SHA-256 $actual_sha != $ARCHIVE_SHA256"
}

production_order_ok() {
  local script_path="$1" gate_pattern="$2" host_pattern="$3" resource_pattern="$4"
  local checkpoint_pattern="$5" scratch_pattern="$6" cargo_pattern="$7"
  local gate_line host_line resource_line checkpoint_line scratch_line cargo_line
  gate_line="$(grep -nE "$gate_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  host_line="$(grep -nE "$host_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  resource_line="$(grep -nE "$resource_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  checkpoint_line="$(grep -nE "$checkpoint_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  scratch_line="$(grep -nE "$scratch_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  cargo_line="$(grep -nE "$cargo_pattern" "$script_path" | tail -1 | cut -d: -f1 || true)"
  [[ -n "$gate_line" && -n "$host_line" && -n "$resource_line" \
    && -n "$checkpoint_line" && -n "$scratch_line" && -n "$cargo_line" \
    && "$gate_line" -lt "$host_line" && "$gate_line" -lt "$resource_line" \
    && "$gate_line" -lt "$checkpoint_line" && "$gate_line" -lt "$scratch_line" \
    && "$gate_line" -lt "$cargo_line" ]]
}

# shellcheck disable=SC2016
run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp fail=0 cases=0 required
  local parity_invocation="\"\$PARITY_TEST\" -- --exact --ignored"
  local parity_harness_count
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$MODEL_KIND" "$PARITY_TEST" \
    "$GGUF_ENV" "$REFERENCE_PCM_ENV" "$REFERENCE_TOKENS_ENV" \
    "$SOURCE_LANGUAGE_ENV" "$TARGET_LANGUAGE_ENV" \
    "--approval-evidence" "tools/parity/canary_1b/preflight_gate.py" \
    "license_gate_manifest.json" "--variant \"\$VARIANT\"" \
    "$MAIN_CHECKPOINT_MEMBER" "$MAIN_CHECKPOINT_BYTES" \
    "tools/parity/canary_1b_v2_prepare_checkpoint.py" \
    "tools/parity/canary_1b_v2_dump_reference.py" \
    "--frozen --project tools/parity --python 3.12 python" \
    "--target-language de"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-canary-1b-v2-validation: self-test FAIL: contract lost token: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]+(released_checkpoint_matches_official_nemo_greedy_tokens|canary_v2_released_checkpoint_matches_official_nemo_greedy_tokens)[[:space:]]+--' \
    "$script_path" >/dev/null; then
    echo "run-canary-1b-v2-validation: self-test FAIL: bare parity test name found" >&2
    fail=1
  fi
  parity_harness_count="$(grep -Fc -- "$parity_invocation" "$script_path" || true)"
  if [[ "$parity_harness_count" -ne 2 ]]; then
    echo "run-canary-1b-v2-validation: self-test FAIL: expected two exact singleton parity harnesses, found $parity_harness_count" >&2
    fail=1
  fi

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-canary-1b-v2-validation: self-test FAIL: direct Python/pip command found" >&2
    fail=1
  fi
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-canary-1b-v2-validation: self-test FAIL: publication command found" >&2
    fail=1
  fi

  cases=$((cases + 1))
  for required in 'uname -s' 'VOKRA_PUBLISH_ON_VAST' 'git status --porcelain --untracked-files=all' \
    'cargo fmt --all -- --check' 'cargo test --locked --workspace' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-canary-1b-v2-validation: self-test FAIL: fail-closed guard lost token: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  local gate_pattern='^[[:space:]]*license_preflight "\$approval_evidence"[[:space:]]*$'
  local host_pattern='^[[:space:]]*\[\[ "\$\(uname -s\)" == "Linux" \]\]'
  local resource_pattern='^[[:space:]]*\[\[ "\$\{VOKRA_PUBLISH_ON_VAST:-0\}" == "1" \]\]'
  local checkpoint_pattern='^[[:space:]]*verify_archive "\$nemo_path"[[:space:]]*$'
  local scratch_pattern='^[[:space:]]*mkdir -p "\$work_dir"[[:space:]]*$'
  local cargo_pattern='^[[:space:]]*cargo clippy --version'
  if ! production_order_ok "$script_path" "$gate_pattern" "$host_pattern" \
    "$resource_pattern" "$checkpoint_pattern" "$scratch_pattern" "$cargo_pattern"; then
    echo 'run-canary-1b-v2-validation: self-test FAIL: preflight is not before production boundaries' >&2
    fail=1
  fi
  if grep -vE "$gate_pattern" "$script_path" > "$tmp/without-preflight.sh" \
    && production_order_ok "$tmp/without-preflight.sh" "$gate_pattern" "$host_pattern" \
      "$resource_pattern" "$checkpoint_pattern" "$scratch_pattern" "$cargo_pattern"; then
    echo 'run-canary-1b-v2-validation: self-test FAIL: deleted production preflight was accepted' >&2
    fail=1
  fi

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$tmp/nonempty" >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: extra self-test argument accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: missing --nemo value accepted" >&2
    fail=1
  fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: unknown argument accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo "$tmp/a" --nemo "$tmp/b" >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: duplicate --nemo accepted" >&2
    fail=1
  fi
  if "$script_path" --self-test --approval-evidence "$tmp/approval.json" >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: extra approval argument accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo "$tmp/a" --approval-evidence >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: missing approval value accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo "$tmp/a" --approval-evidence "$tmp/a" --approval-evidence "$tmp/b" >/dev/null 2>&1; then
    echo "run-canary-1b-v2-validation: self-test FAIL: duplicate approval accepted" >&2
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-canary-1b-v2-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

nemo_path=""
approval_evidence=""
work_dir="/workspace/vokra-canary-v2-validation"
seen_nemo=0
seen_approval=0
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test)
      self_test=1
      shift
      ;;
    --nemo)
      (( seen_nemo == 0 )) || die "duplicate --nemo"
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--nemo requires a path"
      seen_nemo=1
      nemo_path="$2"
      shift 2
      ;;
    --approval-evidence)
      (( seen_approval == 0 )) || die "duplicate --approval-evidence"
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--approval-evidence requires a path"
      seen_approval=1
      approval_evidence="$2"
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
  [[ -z "$nemo_path$approval_evidence" && "$work_dir" == "/workspace/vokra-canary-v2-validation" ]] \
    || die "--self-test accepts no other arguments"
  run_self_test
  exit $?
fi

# Keep the approval gate before any host/resource probe, input inspection,
# scratch/evidence creation, environment sync, model operation, or Cargo.
[[ -n "$approval_evidence" ]] || die "--approval-evidence is required"
license_preflight "$approval_evidence"

[[ "$(uname -s)" == "Linux" ]] || die "actual validation is Linux/VAST-only"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
  || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
[[ -n "$nemo_path" ]] || die "--nemo is required"
verify_archive "$nemo_path"
[[ -f Cargo.toml && -d crates/vokra-models ]] \
  || die "run from the Vokra repository root"
require_absent_work_dir "$work_dir" "$nemo_path"

# Fail before unpacking the multi-gigabyte checkpoint if verification tooling
# or repository state is incomplete.
for command in rustfmt cargo-deny cargo-audit; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required VAST verification tool is missing: $command"
done
for command in sha256sum wc tr; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required archive identity tool is missing: $command"
done
cargo clippy --version >/dev/null 2>&1 \
  || die "the clippy component is missing; install rustfmt/clippy on the VAST host"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || die "worktree changes or untracked files are present; validate a clean committed git-bundle checkpoint"
verify_archive "$nemo_path"

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
  tools/parity/canary_1b_v2_prepare_checkpoint.py \
  --input "$nemo_path" --output-dir "$prepared_dir"

run_logged cargo build --locked --release -p vokra-cli
run_logged target/release/vokra-cli convert \
  --model "$MODEL_KIND" \
  --input "$prepared_dir/canary-1b-v2.prepared.safetensors" \
  --tokenizer "$prepared_dir/tokenizer.vocab" \
  --output "$work_dir/canary-1b-v2.gguf"

run_logged uv run --frozen --project tools/parity --extra titanet --python 3.12 python \
  tools/parity/canary_1b_v2_dump_reference.py \
  --nemo "$nemo_path" \
  --source-language en --target-language en \
  --output "$evidence_dir/reference-en-en.json"
run_logged uv run --frozen --project tools/parity --extra titanet --python 3.12 python \
  tools/parity/canary_1b_v2_dump_reference.py \
  --nemo "$nemo_path" \
  --source-language en --target-language de \
  --output "$evidence_dir/reference-en-de.json"

export "$GGUF_ENV=$work_dir/canary-1b-v2.gguf"
export "$REFERENCE_PCM_ENV=$evidence_dir/reference-en-en.pcm.f32"
export "$REFERENCE_TOKENS_ENV=$evidence_dir/reference-en-en.tokens.txt"
export "$SOURCE_LANGUAGE_ENV=en"
export "$TARGET_LANGUAGE_ENV=en"
run_logged cargo test --locked -p vokra-models \
  "$PARITY_TEST" -- --exact --ignored

# A different target language changes the prompt and independently exercises
# AST; it is never inferred from the English-ASR pass.
export "$REFERENCE_PCM_ENV=$evidence_dir/reference-en-de.pcm.f32"
export "$REFERENCE_TOKENS_ENV=$evidence_dir/reference-en-de.tokens.txt"
export "$TARGET_LANGUAGE_ENV=de"
run_logged cargo test --locked -p vokra-models \
  "$PARITY_TEST" -- --exact --ignored

run_logged target/release/vokra-cli run \
  --model "$work_dir/canary-1b-v2.gguf" \
  --input tests/fixtures/audio/jfk-30s.wav \
  --backend cpu --language en --target-language en
run_logged target/release/vokra-cli run \
  --model "$work_dir/canary-1b-v2.gguf" \
  --input tests/fixtures/audio/jfk-30s.wav \
  --backend cpu --language en --target-language de

run_logged cargo test --locked --workspace
run_logged cargo clippy --locked --workspace --all-targets -- -D warnings
run_logged cargo deny check licenses advisories bans
run_logged cargo audit

{
  echo "variant=$VARIANT"
  echo "upstream_repo=$UPSTREAM_REPO"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "archive_bytes=$ARCHIVE_BYTES"
  echo "archive_sha256=$ARCHIVE_SHA256"
  echo "main_checkpoint_member=$MAIN_CHECKPOINT_MEMBER"
  echo "main_checkpoint_bytes=$MAIN_CHECKPOINT_BYTES"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "kernel=$(uname -srmo)"
  echo "cpu=$(awk -F ': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
  echo "nemo_sha256=$(sha256sum "$nemo_path" | awk '{print $1}')"
  echo "gguf_sha256=$(sha256sum "$work_dir/canary-1b-v2.gguf" | awk '{print $1}')"
  echo "reference_en_en_sha256=$(sha256sum "$evidence_dir/reference-en-en.json" | awk '{print $1}')"
  echo "reference_en_de_sha256=$(sha256sum "$evidence_dir/reference-en-de.json" | awk '{print $1}')"
  echo "verdict=PASS"
} > "$evidence_dir/validation-summary.txt"

cp "$prepared_dir/prepare-audit.json" "$evidence_dir/prepare-audit.json"
echo "run-canary-1b-v2-validation: PASS"
echo "Pull before destroy: $evidence_dir and $log_path"
echo "Do not pull the multi-GB .nemo/.safetensors/.gguf artifacts to the maintainer Mac."
