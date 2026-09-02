#!/usr/bin/env bash
# VAST/Linux-only native SGMSE score parity consumer.
#
# The independent reference is verified before Cargo is started.  The Rust
# consumer then binds an authenticated SGMSE GGUF, runs the public CPU score
# API, and writes only two score planes into a newly-created disjoint output
# directory.  This worker never generates reference values, copies expected
# outputs, publishes, or uploads model artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
VERIFY_TOOL="$PARITY_PROJECT/sgmse_verify_reference.py"
NATIVE_PARITY_TOOL="$PARITY_PROJECT/sgmse_native_score_parity.py"

log() { printf '[sgmse-native-score-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-sgmse-native-score-parity.sh \
  --gguf <absolute-authenticated-gguf> --gguf-sha256 <64-hex-digest> \
  --reference-dir <absolute-verified-reference-dir> \
  --native-output-dir <absolute-absent-output-dir>
       run-sgmse-native-score-parity.sh --self-test

VAST/Linux-only native SGMSE score consumer.  The reference directory must be
created by sgmse_dump_reference.py and verified by sgmse_verify_reference.py.
The GGUF digest is supplied by the authenticated VAST conversion step.  The
native output directory must not exist; the Rust test creates exactly
score_real.f32 and score_imag.f32 there.  No expected score is generated or
copied by this worker.
EOF
}

reject_symlink_ancestry() {
  local path="$1" label="$2" current="$1"
  while :; do
    [[ ! -L "$current" ]] || die "$label has symlink ancestry: $current"
    [[ "$current" == / ]] && break
    current="$(dirname "$current")"
  done
}

require_absolute() {
  local path="$1" label="$2"
  [[ "$path" == /* ]] || die "$label must be absolute: $path"
}

disjoint() {
  local left right
  left="$(realpath -m "$1")"
  right="$(realpath -m "$2")"
  [[ "$left" != "$right" && "$left" != "$right"/* && "$right" != "$left"/* ]] \
    || die "paths overlap: $1 and $2"
}

run_self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'CARGO_BUILD_JOBS=1' \
    'sgmse_verify_reference.py' 'sgmse_native_score_parity.py' \
    'REFERENCE_MANIFEST_VERIFIED' 'SGMSE_NATIVE_SCORE_PARITY_PASS' \
    'VOKRA_SGMSE_GGUF_SHA256' 'score_real.f32' 'score_imag.f32' \
    'create exactly' 'newly-created' 'realpath -m' 'findmnt' \
    'uv run --frozen --no-sync --project' \
    'cargo test --locked --test sgmse_native_score -p vokra-models' \
    '-- --ignored --exact --show-output'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if ! UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" \
    uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
    "$VERIFY_TOOL" --self-test >/dev/null; then
    log 'self-test FAIL: reference verifier self-test failed'
    fail=1
  fi
  if ! UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" \
    uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
    "$NATIVE_PARITY_TOOL" --self-test >/dev/null; then
    log 'self-test FAIL: native comparator self-test failed'
    fail=1
  fi
  if grep -Fq -- 'REFERENCE_MANIFEST_SHA256' "$NATIVE_PARITY_TOOL"; then
    log 'self-test FAIL: unstable whole-manifest digest pin remains'
    fail=1
  fi
  if ! grep -Fq -- 'REVIEWED_ARTIFACT_SHA256' "$NATIVE_PARITY_TOOL"; then
    log 'self-test FAIL: reviewed per-artifact digest gate is missing'
    fail=1
  fi
  # This deliberately reaches only host/argument guards: no path exists, no
  # reference is opened, no model is loaded, and no network command is run.
  if VOKRA_PUBLISH_ON_VAST=1 "$path" \
    --gguf /tmp/sgmse-no-model.gguf \
    --gguf-sha256 0000000000000000000000000000000000000000000000000000000000000000 \
    --reference-dir /tmp/sgmse-no-reference \
    --native-output-dir /tmp/sgmse-no-native >/dev/null 2>&1; then
    log 'self-test FAIL: nonexistent protected inputs were accepted'
    fail=1
  fi
  if "$path" --self-test --gguf /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted with --self-test'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

GGUF=""
GGUF_SHA256=""
REFERENCE_DIR=""
NATIVE_OUTPUT_DIR=""
SELF_TEST=0
while (($#)); do
  case "$1" in
    --self-test) (( SELF_TEST == 0 )) || die 'duplicate --self-test'; SELF_TEST=1; shift ;;
    --gguf) (($# >= 2)) || die '--gguf requires a path'; GGUF="$2"; shift 2 ;;
    --gguf-sha256) (($# >= 2)) || die '--gguf-sha256 requires a digest'; GGUF_SHA256="$2"; shift 2 ;;
    --reference-dir) (($# >= 2)) || die '--reference-dir requires a path'; REFERENCE_DIR="$2"; shift 2 ;;
    --native-output-dir) (($# >= 2)) || die '--native-output-dir requires a path'; NATIVE_OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( SELF_TEST )); then
  [[ -z "$GGUF$GGUF_SHA256$REFERENCE_DIR$NATIVE_OUTPUT_DIR" ]] || die '--self-test accepts no other arguments'
  run_self_test
  exit $?
fi

[[ -n "$GGUF" && -n "$GGUF_SHA256" && -n "$REFERENCE_DIR" && -n "$NATIVE_OUTPUT_DIR" ]] \
  || { usage >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || die 'SGMSE native score parity is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'SGMSE native score parity requires x86_64 VAST'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST Vokra checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$VERIFY_TOOL" && -f "$NATIVE_PARITY_TOOL" ]] || die 'SGMSE parity tools are missing'
for command in cargo uv realpath findmnt sha256sum awk grep tee mktemp; do
  command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"
done

require_absolute "$GGUF" '--gguf'
require_absolute "$REFERENCE_DIR" '--reference-dir'
require_absolute "$NATIVE_OUTPUT_DIR" '--native-output-dir'
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 must be 64 lowercase hexadecimal characters'
reject_symlink_ancestry "$GGUF" '--gguf'
reject_symlink_ancestry "$REFERENCE_DIR" '--reference-dir'
[[ -f "$GGUF" && ! -L "$GGUF" ]] || die 'GGUF is missing or symlinked'
[[ -d "$REFERENCE_DIR" && ! -L "$REFERENCE_DIR" ]] || die 'reference directory is missing or symlinked'
[[ ! -e "$NATIVE_OUTPUT_DIR" && ! -L "$NATIVE_OUTPUT_DIR" ]] || die 'native output directory must be absent'
native_parent="$(dirname "$NATIVE_OUTPUT_DIR")"
[[ -d "$native_parent" && ! -L "$native_parent" ]] || die 'native output parent must be an existing real directory'
reject_symlink_ancestry "$native_parent" 'native output parent'
[[ "$(findmnt -T "$native_parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] \
  || die 'native output parent must be tmpfs/RAM disk'
disjoint "$GGUF" "$REFERENCE_DIR"
disjoint "$GGUF" "$NATIVE_OUTPUT_DIR"
disjoint "$REFERENCE_DIR" "$NATIVE_OUTPUT_DIR"
actual_gguf_sha256="$(sha256sum "$GGUF" | awk '{print $1}')"
[[ "$actual_gguf_sha256" == "$GGUF_SHA256" ]] || die "GGUF SHA-256 mismatch: got $actual_gguf_sha256"

# Verify the independent reference before any native Cargo/model operation.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" \
  uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
  "$VERIFY_TOOL" --manifest "$REFERENCE_DIR/manifest.json" \
  --output-dir "$REFERENCE_DIR" --vokra-root "$VOKRA_ROOT"

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
native_log="$(mktemp "${TMPDIR:-/tmp}/vokra-sgmse-native-score.XXXXXX")"
trap 'rm -f -- "$native_log"' EXIT
VOKRA_SGMSE_GGUF="$GGUF" \
VOKRA_SGMSE_GGUF_SHA256="$GGUF_SHA256" \
VOKRA_SGMSE_REFERENCE_DIR="$REFERENCE_DIR" \
VOKRA_SGMSE_NATIVE_OUTPUT_DIR="$NATIVE_OUTPUT_DIR" \
VOKRA_PUBLISH_ON_VAST=1 \
  cargo test --locked --test sgmse_native_score -p vokra-models \
    -- --ignored --exact --show-output 2>&1 | tee "$native_log"
[[ "$(grep -Fxc 'test sgmse_native_score_matches_independent_reference ... ok' "$native_log" || true)" == 1 ]] \
  || die 'SGMSE native score test did not pass exactly once'
[[ "$(grep -Ec '^test result: ok[.] 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in .+)?$' "$native_log" || true)" == 1 ]] \
  || die 'SGMSE native score test result was not exactly one pass'
[[ "$(find "$NATIVE_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort | tr '\n' ' ')" == 'score_imag.f32 score_real.f32 ' ]] \
  || die 'native output does not contain exactly score_real.f32 and score_imag.f32'

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" \
  uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
  "$NATIVE_PARITY_TOOL" --reference-dir "$REFERENCE_DIR" --native-dir "$NATIVE_OUTPUT_DIR"
log "SGMSE native CPU score parity complete: GGUF sha256=$GGUF_SHA256; output=$NATIVE_OUTPUT_DIR"
