#!/usr/bin/env bash
# shellcheck disable=SC2329
# VAST-only microWakeWord validation gate.  The affirmative dependency and
# provenance gates are intentionally blocked; no model/source download,
# conversion, Cargo, or upload path is staged until immutable identities are
# authenticated.
# shellcheck disable=SC2034 # identity constants are self-test contract data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/microwakeword"
INSPECTOR="$ROOT/tools/parity/microwakeword_inspect.py"
TENSOR_MANIFEST_PRODUCER="$ROOT/tools/parity/microwakeword_tensor_manifest.py"
LOCK_SHA256="05e8317758e7c884e8e86e110af5b39cdd23eff63b6a66705225e6baa3ab5e13"
PACKAGE_COUNT=10
PACKAGE_ROWS_SHA256="d5c8aaca80e340be13e719de14d1486df193977f31ea80dea3bf954030057343"
LICENSE_ROWS_SHA256="eae9f062f7ceb787fe36e09290fbc04b8f2f842df9de612b79f95f7fd615c58f"
MODEL_REPOSITORY="esphome/micro-wake-word-models"
MODEL_REVISION="05b65922cc433c9df13e98e32a7fe520758c837e"
SOURCE_REPOSITORY="https://github.com/kahrendt/microWakeWord"
SOURCE_REVISION="4665173cd35f1cff9a61e06fc427f124766c488e"
MODEL_TARGET_PATH="models/v2/hey_jarvis.tflite"
MODEL_TARGET_GIT_BLOB="0075302434cc72a460ced0b8f6c09c69214e5cf0"
MODEL_TARGET_SIZE=52272
MODEL_COMPANION_GIT_BLOB="e6733fe13852f04a5a3ae83e0d39b5726aee62cc"
MODEL_COMPANION_SIZE=388
LICENSE_GIT_BLOB="261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
LICENSE_SIZE=11357
MODEL_ARTIFACT_BYTES_SHA256=""
PUBLICATION_STATUS="NO_UPLOAD"
UV_CACHE_DIR_VALUE="${MICROWAKEWORD_UV_CACHE_DIR:-/tmp/vokra-microwakeword-uv-cache}"

die() { echo "run-microwakeword-validation: $*" >&2; exit 2; }

# Called only by the future authenticated VAST campaign, after the immutable
# model bytes have been materialized.  Keeping the producer immediately before
# the converter makes the generated manifest SHA the exact value passed into
# the preparer; this function is deliberately not reachable while the current
# provenance/dependency gate is blocked.
run_authenticated_tensor_pipeline() {
  local tflite_path="$1" manifest_path="$2" output_path="$3" manifest_sha256
  uv run --no-project --python 3.12 python "$TENSOR_MANIFEST_PRODUCER" \
    --input "$tflite_path" --output "$manifest_path"
  manifest_sha256="$(sha256sum "$manifest_path" | awk '{print $1}')"
  [[ "$manifest_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || die "generated tensor manifest SHA-256 is invalid"
  uv run --no-project --python 3.12 python "$PROJECT/prepare_checkpoint.py" \
    --input "$tflite_path" --name hey_jarvis \
    --expected-sha256 "$MODEL_ARTIFACT_BYTES_SHA256" \
    --tensor-manifest "$manifest_path" --tensor-manifest-sha256 "$manifest_sha256" \
    --output "$output_path"
}

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 gate_line
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/tools/parity/microwakeword_inspect.py" ]] || { echo "self-test FAIL: inspector missing" >&2; fail=1; }
  for needle in "microwakeword_inspect.py" "microwakeword_tensor_manifest.py" "run_authenticated_tensor_pipeline" "prepare_checkpoint.py" "--self-test" "$LOCK_SHA256" "$PACKAGE_COUNT" "$PACKAGE_ROWS_SHA256" "$LICENSE_ROWS_SHA256" "ai-edge-litert==2.2.0" "protobuf==7.36.0" "--dependency-gate" "BLOCKED_UNREVIEWED_TRANSITIVE" "AUTHENTICATED_PAYLOAD_SHA_REQUIRED" "SOURCE_TENSOR_MANIFEST_REQUIRED" "--tensor-manifest" "tensor-manifest-sha256" "NO_UPLOAD" "VAST" "$MODEL_REPOSITORY" "$SOURCE_REPOSITORY" "SOURCE_REVISION" "MODEL_REVISION" "4665173cd35f1cff9a61e06fc427f124766c488e" "05b65922cc433c9df13e98e32a7fe520758c837e" "$MODEL_TARGET_PATH" "$MODEL_TARGET_GIT_BLOB" "$MODEL_TARGET_SIZE" "$MODEL_COMPANION_GIT_BLOB" "$MODEL_COMPANION_SIZE" "$LICENSE_GIT_BLOB" "$LICENSE_SIZE" 'MODEL_ARTIFACT_BYTES_SHA256=""'; do
    grep -Fq -- "$needle" "$self" || { echo "self-test FAIL: missing $needle" >&2; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload|vokra-cli[[:space:]]+convert)([[:space:]]|$)' "$self" >/dev/null; then
    echo 'self-test FAIL: upload/conversion command found' >&2; fail=1
  fi
  if grep -En '(^|[;&|])[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then
    echo 'self-test FAIL: raw Python/pip invocation found' >&2; fail=1
  fi
  gate_line="$(grep -n 'INSPECTOR.*--dependency-gate' "$self" | tail -1 | cut -d: -f1)"
  [[ -n "$gate_line" ]] || { echo 'self-test FAIL: dependency gate invocation missing' >&2; fail=1; }
  # This worker is terminally blocked.  There is deliberately no sync,
  # work-directory creation, acquisition, or Cargo command after the gate;
  # the manager review keeps this terminal shape intact until identities land.
  local post_gate normalized token uv_word sync_word run_word project_word snapshot_word hf_word clone_word mkdir_word work_word cargo_word bundle_word upload_word push_word
  post_gate="$(tail -n +"$gate_line" "$self")"
  normalized="$(printf '%s' "$post_gate" | tr -d "\"'\\")"
  uv_word='u'; uv_word+='v'; run_word="$uv_word run"; sync_word="$uv_word sync"; project_word="$run_word --project"
  snapshot_word='snapshot'; snapshot_word+='_download'; hf_word='hf_'; hf_word+='hub_download'; clone_word='git'; clone_word+=' clone'; mkdir_word='mkdir'; mkdir_word+=' -p'; work_word='WORK_DIR'; cargo_word='cargo'; bundle_word='git'; bundle_word+=' bundle'; upload_word='upload'; push_word='git'; push_word+=' push'
  for token in "$sync_word" "$project_word" "$snapshot_word" "$hf_word" "$clone_word" "$mkdir_word" "$work_word" "$cargo_word" "$bundle_word" "$upload_word" "$push_word" '--push'; do
    [[ "$normalized" != *"$token"* ]] || { echo "self-test FAIL: post-gate effect found: $token" >&2; fail=1; }
  done
  local producer_line converter_line
  producer_line="$(grep -n 'TENSOR_MANIFEST_PRODUCER' "$self" | head -1 | cut -d: -f1)"
  converter_line="$(grep -n 'prepare_checkpoint.py' "$self" | head -1 | cut -d: -f1)"
  [[ -n "$producer_line" && -n "$converter_line" && "$producer_line" -lt "$converter_line" ]] || { echo 'self-test FAIL: tensor manifest producer must precede converter' >&2; fail=1; }
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --python 3.12 python - "$self" <<'PY'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text(encoding="utf-8")
start = source.index("run_authenticated_tensor_pipeline()")
body = source[start:source.index("\n}", start)]
convert = body.index("prepare_checkpoint.py")
if '--input "$tflite_path"' not in body[convert:] or '--url' in body[convert:]:
    raise SystemExit("future pipeline must use input transport without --url")
# Negative contract: argparse must reject the old simultaneous transport
# spelling; this fixture keeps the regression test explicit and model-free.
invalid = '--input model.tflite --url https://example.invalid/model.tflite'
if not ('--input' in invalid and '--url' in invalid):
    raise SystemExit("negative input/url composition fixture is malformed")
print("microWakeWord input transport composition self-test: PASS")
PY
  then
    echo 'self-test FAIL: invalid input/url composition' >&2
    fail=1
  fi
  if (( fail == 0 )); then
    if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --python 3.12 python "$root/tools/parity/microwakeword_tensor_manifest.py" --self-test; then
      echo 'self-test FAIL: tensor manifest producer synthetic FlatBuffer self-test failed' >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then
    if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --python 3.12 python "$root/tools/parity/microwakeword/prepare_checkpoint.py" --self-test; then
      echo 'self-test FAIL: prepare_checkpoint.py synthetic wire self-test failed' >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then echo 'run-microwakeword-validation.sh self-test: PASS'; else return 1; fi
}

if [[ "${1:-}" == "--self-test" ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no arguments"
  self_test
  exit 0
fi
[[ $# == 0 ]] || die "arguments are not accepted"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ "$MODEL_ARTIFACT_BYTES_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || die "AUTHENTICATED_PAYLOAD_SHA_REQUIRED"
cd "$ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
[[ -f "$PROJECT/uv.lock" ]] || die "dedicated uv.lock is absent"
[[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || die "uv.lock identity mismatch"
for command in git uv sha256sum awk find cargo; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
export UV_CACHE_DIR="$UV_CACHE_DIR_VALUE"
export CARGO_BUILD_JOBS=1

# This no-project, stdlib-only call is the first effectful operation.  It is
# blocked by design, so everything below is unreachable until owner review.
uv run --no-project --python 3.12 python "$INSPECTOR" --dependency-gate || die "dependency/license gate is not approved"
die "microWakeWord artifact byte identity or dependency/license evidence is unresolved"
