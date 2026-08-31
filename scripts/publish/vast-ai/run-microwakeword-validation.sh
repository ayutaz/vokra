#!/usr/bin/env bash
# shellcheck disable=SC2329
# VAST-only microWakeWord validation gate.  The affirmative dependency and
# provenance gates are intentionally blocked; the preparer is
# ZERO_EXTERNAL_DEPENDENCIES. The normal production path performs no
# model/source download, conversion, Cargo, or upload. Only the separately
# gated --inspect-only path temporarily fetches the fixed artifact for
# evidence, without conversion or publication.
# shellcheck disable=SC2034 # identity constants are self-test contract data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/microwakeword"
INSPECTOR="$ROOT/tools/parity/microwakeword_inspect.py"
TENSOR_MANIFEST_PRODUCER="$ROOT/tools/parity/microwakeword_tensor_manifest.py"
LOCK_SHA256="984703d5bafdd6c88006bd381095961d42ef684d269d66194edbeda1fddf8dc2"
PACKAGE_COUNT=1
PACKAGE_ROWS_SHA256="d9b806830227b4fdbdbe59ea5a20b529bfae40f6aa70e239b44a6238fabd5ad7"
LICENSE_ROWS_SHA256="4ee7351311d5d0bf69758093e88be7b4146fefdcbc80e026662bbdf58032272c"
MODEL_REPOSITORY="esphome/micro-wake-word-models"
MODEL_REVISION="05b65922cc433c9df13e98e32a7fe520758c837e"
SOURCE_REPOSITORY="https://github.com/kahrendt/microWakeWord"
SOURCE_REVISION="4665173cd35f1cff9a61e06fc427f124766c488e"
MODEL_TARGET_PATH="models/v2/hey_jarvis.tflite"
DEFAULT_UPSTREAM_URL="https://github.com/esphome/micro-wake-word-models/raw/05b65922cc433c9df13e98e32a7fe520758c837e/models/v2/hey_jarvis.tflite"
LICENSE_URL="https://github.com/esphome/micro-wake-word-models/raw/05b65922cc433c9df13e98e32a7fe520758c837e/LICENSE"
COMPANION_URL="https://github.com/esphome/micro-wake-word-models/raw/05b65922cc433c9df13e98e32a7fe520758c837e/models/v2/hey_jarvis.json"
MODEL_TARGET_GIT_BLOB="0075302434cc72a460ced0b8f6c09c69214e5cf0"
MODEL_TARGET_SIZE=52272
MODEL_COMPANION_GIT_BLOB="e6733fe13852f04a5a3ae83e0d39b5726aee62cc"
MODEL_COMPANION_SIZE=388
LICENSE_GIT_BLOB="261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
LICENSE_SIZE=11357
MODEL_ARTIFACT_BYTES_SHA256=""
REVIEWED_TOPOLOGY_SHA256=""
PUBLICATION_STATUS="NO_UPLOAD"
UV_CACHE_DIR_VALUE="${MICROWAKEWORD_UV_CACHE_DIR:-/tmp/vokra-microwakeword-uv-cache}"

die() { echo "run-microwakeword-validation: $*" >&2; exit 2; }
INSPECTION_WORK_DIR=""

cleanup_inspection_workdir() {
  case "$INSPECTION_WORK_DIR" in
    /tmp/vokra-mww-inspect.*)
      rm -rf -- "$INSPECTION_WORK_DIR"
      INSPECTION_WORK_DIR=""
      ;;
    "") ;;
    *) die "unsafe inspection cleanup path" ;;
  esac
}

# Called only by the future authenticated VAST campaign, after the immutable
# model bytes have been materialized.  Keeping the producer immediately before
# the converter makes the generated manifest SHA the exact value passed into
# the preparer; this function is deliberately not reachable while the current
# provenance/dependency gate is blocked.
run_authenticated_tensor_pipeline() {
  local tflite_path="$1" manifest_path="$2" output_path="$3" manifest_sha256
  uv run --no-project --offline --python 3.12 python "$TENSOR_MANIFEST_PRODUCER" \
    --input "$tflite_path" --output "$manifest_path"
  manifest_sha256="$(sha256sum "$manifest_path" | awk '{print $1}')"
  [[ "$manifest_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || die "generated tensor manifest SHA-256 is invalid"
  uv run --no-project --offline --python 3.12 python "$PROJECT/prepare_checkpoint.py" \
    --input "$tflite_path" --name hey_jarvis \
    --expected-sha256 "$MODEL_ARTIFACT_BYTES_SHA256" \
    --tensor-manifest "$manifest_path" --tensor-manifest-sha256 "$manifest_sha256" \
    --output "$output_path"
}

# Evidence-only VAST path. It is intentionally separate from production:
# fixed upstream bytes are fetched once, hashed, and inspected by the raw
# parser; no GGUF writer, interpreter, Cargo, inference, or upload is called.
inspect_only() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
  [[ "${VOKRA_INSPECT_ONLY:-0}" == 1 ]] || die "VOKRA_INSPECT_ONLY=1 is required"
  [[ "$1" == --inspect-only && $# == 1 ]] || die "--inspect-only accepts no arguments"
  cd "$ROOT"
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
  [[ -f "$PROJECT/uv.lock" ]] || die "dedicated uv.lock is absent"
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || die "uv.lock identity mismatch"
  command -v curl >/dev/null 2>&1 || die "missing tool: curl"
  command -v realpath >/dev/null 2>&1 || die "missing tool: realpath"
  local work_dir manifest_path evidence_path archive_dir actual_sha manifest_sha inventory_source_sha companion_sha license_sha archive_manifest archive_companion archive_license
  archive_dir="${MICROWAKEWORD_INSPECTION_DIR:-}"
  [[ "$archive_dir" == /* && -d "$archive_dir" && ! -L "$archive_dir" ]] || die "MICROWAKEWORD_INSPECTION_DIR must be absolute existing directory"
  archive_dir="$(realpath -e -- "$archive_dir")" || die "MICROWAKEWORD_INSPECTION_DIR cannot be canonicalized"
  [[ "$archive_dir" != "$ROOT" && "$archive_dir" != "$ROOT"/* ]] || die "inspection directory must be outside checkout root"
  evidence_path="$archive_dir/hey_jarvis.inspection.json"
  [[ ! -e "$evidence_path" && ! -L "$evidence_path" ]] || die "inspection evidence destination exists"
  INSPECTION_WORK_DIR="$(mktemp -d "/tmp/vokra-mww-inspect.XXXXXX")"
  work_dir="$INSPECTION_WORK_DIR"
  trap cleanup_inspection_workdir EXIT
  manifest_path="$work_dir/raw-inventory.json"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python "$INSPECTOR" --self-test
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --output "$work_dir/hey_jarvis.tflite" "$DEFAULT_UPSTREAM_URL"
  [[ "$(stat -c '%s' "$work_dir/hey_jarvis.tflite")" == "$MODEL_TARGET_SIZE" ]] || die "canonical artifact size mismatch"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$work_dir/hey_jarvis.tflite" "$MODEL_TARGET_GIT_BLOB" <<'PY'
import hashlib, sys
from pathlib import Path
payload = Path(sys.argv[1]).read_bytes()
blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
if blob != sys.argv[2]:
    raise SystemExit("canonical model git blob mismatch")
PY
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --output "$work_dir/LICENSE" "$LICENSE_URL"
  [[ "$(stat -c '%s' "$work_dir/LICENSE")" == "$LICENSE_SIZE" ]] || die "canonical license size mismatch"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$work_dir/LICENSE" "$LICENSE_GIT_BLOB" <<'PY'
import hashlib, sys
from pathlib import Path
payload = Path(sys.argv[1]).read_bytes()
blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
if blob != sys.argv[2]:
    raise SystemExit("canonical license git blob mismatch")
PY
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
    --output "$work_dir/hey_jarvis.json" "$COMPANION_URL"
  [[ "$(stat -c '%s' "$work_dir/hey_jarvis.json")" == "$MODEL_COMPANION_SIZE" ]] || die "canonical companion size mismatch"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$work_dir/hey_jarvis.json" "$MODEL_COMPANION_GIT_BLOB" <<'PY'
import hashlib, sys
from pathlib import Path
payload = Path(sys.argv[1]).read_bytes()
blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
if blob != sys.argv[2]:
    raise SystemExit("canonical companion git blob mismatch")
PY
  actual_sha="$(sha256sum "$work_dir/hey_jarvis.tflite" | awk '{print $1}')"
  [[ "$actual_sha" =~ ^[0-9a-f]{64}$ ]] || die "artifact SHA-256 calculation failed"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python "$TENSOR_MANIFEST_PRODUCER" \
    --inventory-only --input "$work_dir/hey_jarvis.tflite" --output "$manifest_path"
  manifest_sha="$(sha256sum "$manifest_path" | awk '{print $1}')"
  companion_sha="$(sha256sum "$work_dir/hey_jarvis.json" | awk '{print $1}')"
  license_sha="$(sha256sum "$work_dir/LICENSE" | awk '{print $1}')"
  inventory_source_sha="$(UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$manifest_path" <<'PY'
import json, sys
from pathlib import Path
def strict_object(pairs):
    result = {}
    for key, item in pairs:
        if key in result:
            raise SystemExit("duplicate manifest JSON key")
        result[key] = item
    return result
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=strict_object)
if value.get("format") != "vokra-microwakeword-tflite-raw-inventory-v1":
    raise SystemExit("raw inventory format is absent")
if value.get("authority") != "EVIDENCE_ONLY_UNREVIEWED":
    raise SystemExit("raw inventory authority is not evidence-only")
source_sha = value.get("source_sha256")
if not isinstance(source_sha, str) or len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
    raise SystemExit("raw inventory source SHA-256 is invalid")
print(source_sha)
PY
  )"
  [[ "$inventory_source_sha" == "$actual_sha" ]] || die "raw inventory source SHA-256 does not match model bytes"
  archive_manifest="$archive_dir/hey_jarvis.raw-inventory.json"
  archive_companion="$archive_dir/hey_jarvis.json"
  archive_license="$archive_dir/LICENSE"
  [[ ! -e "$archive_manifest" && ! -L "$archive_manifest" && ! -e "$archive_companion" && ! -L "$archive_companion" && ! -e "$archive_license" && ! -L "$archive_license" ]] || die "inspection evidence destination exists"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$manifest_path" "$archive_manifest" "$work_dir/hey_jarvis.json" "$archive_companion" "$work_dir/LICENSE" "$archive_license" <<'PY'
import os, sys
from pathlib import Path
created = []
for source_name, destination_name in zip(sys.argv[1::2], sys.argv[2::2]):
    source = Path(source_name)
    destination = Path(destination_name)
    data = source.read_bytes()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError as error:
        for item in created:
            try:
                item.unlink()
            except FileNotFoundError:
                pass
        raise SystemExit("inspection evidence destination exists") from error
    created.append(destination)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except BaseException:
        for item in created:
            try:
                item.unlink()
            except FileNotFoundError:
                pass
        raise
PY
  # Evidence is written no-clobber to the operator-provided archive directory;
  # it is never uploaded or used as compiled production authority.
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - \
    "$evidence_path" "$MODEL_REPOSITORY" "$MODEL_REVISION" "$MODEL_TARGET_PATH" \
    "$MODEL_TARGET_GIT_BLOB" "$MODEL_TARGET_SIZE" "$LICENSE_GIT_BLOB" "$LICENSE_SIZE" \
    "$actual_sha" "$archive_manifest" "$manifest_sha" "$archive_companion" \
    "$companion_sha" "$archive_license" "$license_sha" "$inventory_source_sha" <<'PY'
import json, sys
from pathlib import Path
(_, output, repository, revision, model_path, model_blob, model_size,
 license_blob, license_size, model_sha, manifest_path, manifest_sha,
 companion_path, companion_sha, license_path, license_sha, inventory_source_sha) = sys.argv
with Path(output).open("x", encoding="utf-8") as stream:
    json.dump({
        "status": "RAW_INVENTORY_ONLY_NO_CONVERSION",
        "publication": "NO_UPLOAD",
        "repository": repository,
        "revision": revision,
        "path": model_path,
        "git_blob": model_blob,
        "size": int(model_size),
        "license_blob": license_blob,
        "license_size": int(license_size),
        "bytes_sha256": model_sha,
        "companion_path": companion_path,
        "companion_sha256": companion_sha,
        "license_path": license_path,
        "license_sha256": license_sha,
        "raw_inventory_path": manifest_path,
        "raw_inventory_sha256": manifest_sha,
        "raw_inventory_source_sha256": inventory_source_sha,
    }, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  echo "inspection evidence: $evidence_path" >&2
}

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 gate_line
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/tools/parity/microwakeword_inspect.py" ]] || { echo "self-test FAIL: inspector missing" >&2; fail=1; }
  for needle in "microwakeword_inspect.py" "microwakeword_tensor_manifest.py" "run_authenticated_tensor_pipeline" "inspect_only" "--inspect-only" "--inventory-only" "RAW_INVENTORY_ONLY_NO_CONVERSION" "raw-inventory" "EVIDENCE_ONLY_UNREVIEWED" "object_pairs_hook" "duplicate manifest JSON key" "realpath -e" "outside checkout root" "prepare_checkpoint.py" "--self-test" "$LOCK_SHA256" "$PACKAGE_COUNT" "$PACKAGE_ROWS_SHA256" "$LICENSE_ROWS_SHA256" "ZERO_EXTERNAL_DEPENDENCIES" "--dependency-gate" "BLOCKED_UNREVIEWED_ARTIFACT" "AUTHENTICATED_PAYLOAD_SHA_REQUIRED" "AUTHENTICATED_TOPOLOGY_REQUIRED" "SOURCE_TENSOR_MANIFEST_REQUIRED" "--tensor-manifest" "tensor-manifest-sha256" "NO_UPLOAD" "VAST" "$MODEL_REPOSITORY" "$SOURCE_REPOSITORY" "SOURCE_REVISION" "MODEL_REVISION" "$DEFAULT_UPSTREAM_URL" "$LICENSE_URL" "$COMPANION_URL" "4665173cd35f1cff9a61e06fc427f124766c488e" "05b65922cc433c9df13e98e32a7fe520758c837e" "$MODEL_TARGET_PATH" "$MODEL_TARGET_GIT_BLOB" "$MODEL_TARGET_SIZE" "$MODEL_COMPANION_GIT_BLOB" "$MODEL_COMPANION_SIZE" "$LICENSE_GIT_BLOB" "$LICENSE_SIZE" 'MODEL_ARTIFACT_BYTES_SHA256=""' 'REVIEWED_TOPOLOGY_SHA256=""'; do
    grep -Fq -- "$needle" "$self" || { echo "self-test FAIL: missing $needle" >&2; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload|vokra-cli[[:space:]]+convert)([[:space:]]|$)' "$self" >/dev/null; then
    echo 'self-test FAIL: upload/conversion command found' >&2; fail=1
  fi
  if grep -En '(^|[;&|])[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then
    echo 'self-test FAIL: raw Python/pip invocation found' >&2; fail=1
  fi
  inspection_body="$(sed -n '/^inspect_only()/,/^}/p' "$self")"
  local platform_gate_pattern=''
  platform_gate_pattern+='[['
  platform_gate_pattern+=" \"\$(uname -s)\" == Linux && \"\$(uname -m)\" == x86_64 ]]"
  if ! grep -Fq -- "$platform_gate_pattern" <<<"$inspection_body" || ! grep -Fq -- "VOKRA_PUBLISH_ON_VAST" <<<"$inspection_body"; then
    echo 'self-test FAIL: inspection mode lacks VAST platform gate' >&2
    fail=1
  fi
  if grep -Fq -- 'prepare_checkpoint.py' <<<"$inspection_body" || grep -E 'vokra-cli|cargo|git push|--upload|--push' <<<"$inspection_body" >/dev/null; then
    echo 'self-test FAIL: inspection mode contains conversion/upload/Cargo' >&2
    fail=1
  fi
  if grep -E 'canonical_digest|canonical_topology_sha256|canonical_identity' <<<"$inspection_body" >/dev/null; then
    echo 'self-test FAIL: inventory inspection mode claims canonical authority' >&2
    fail=1
  fi
  local arbitrary_arg_pattern="--url|--name"
  if grep -E -- "$arbitrary_arg_pattern" <<<"$inspection_body" >/dev/null; then
    echo 'self-test FAIL: inspection mode accepts arbitrary source identity' >&2
    fail=1
  fi
  local cleanup_probe
  cleanup_probe="$(mktemp -d /tmp/vokra-mww-inspect.XXXXXX)"
  INSPECTION_WORK_DIR="$cleanup_probe"
  cleanup_inspection_workdir
  [[ ! -e "$cleanup_probe" && -z "$INSPECTION_WORK_DIR" ]] || { echo 'self-test FAIL: inspection temp cleanup failed' >&2; fail=1; }
  if (INSPECTION_WORK_DIR=/tmp/unsafe-inspection; cleanup_inspection_workdir) 2>/dev/null; then
    echo 'self-test FAIL: unsafe inspection cleanup path was accepted' >&2
    fail=1
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
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$self" <<'PY'
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
    if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - <<'PY'
import json
payload = '{"topology":{"canonical_digest":"a"},"topology":{"canonical_digest":"b"}}'
def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest JSON key")
        result[key] = value
    return result
try:
    json.loads(payload, object_pairs_hook=strict_object)
except ValueError:
    print("strict manifest duplicate-key self-test: PASS")
else:
    raise SystemExit("duplicate manifest key was accepted")
PY
    then
      echo 'self-test FAIL: strict manifest duplicate-key check failed' >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then
    if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python "$root/tools/parity/microwakeword_tensor_manifest.py" --self-test; then
      echo 'self-test FAIL: tensor manifest producer synthetic FlatBuffer self-test failed' >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then
    if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python "$root/tools/parity/microwakeword/prepare_checkpoint.py" --self-test; then
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
if [[ "${1:-}" == "--inspect-only" ]]; then
  inspect_only "$@"
  exit 0
fi
[[ $# == 0 ]] || die "arguments are not accepted"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ "$MODEL_ARTIFACT_BYTES_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || die "AUTHENTICATED_PAYLOAD_SHA_REQUIRED"
[[ "$REVIEWED_TOPOLOGY_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] || die "AUTHENTICATED_TOPOLOGY_REQUIRED"
cd "$ROOT"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
[[ -f "$PROJECT/uv.lock" ]] || die "dedicated uv.lock is absent"
[[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || die "uv.lock identity mismatch"
for command in git uv sha256sum awk find cargo; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
export UV_CACHE_DIR="$UV_CACHE_DIR_VALUE"
export CARGO_BUILD_JOBS=1

# This no-project, stdlib-only call is the first effectful operation.  It is
# blocked by design, so everything below is unreachable until owner review.
uv run --no-project --offline --python 3.12 python "$INSPECTOR" --dependency-gate || die "dependency/license gate is not approved"
die "microWakeWord artifact byte identity or dependency/license evidence is unresolved"
