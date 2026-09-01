#!/usr/bin/env bash
# shellcheck disable=SC2329
# VAST-only microWakeWord validation gate.  The preparer is
# ZERO_EXTERNAL_DEPENDENCIES, while the independent LiteRT oracle uses the
# separately locked reference project.  The reviewed end-to-end path below is
# the only route that may acquire the fixed artifact, convert it, generate
# independent fixtures, and run Path C; it is Linux x86_64/VAST/clean-checkout
# gated and never uploads or copies model payloads into the result archive.
# shellcheck disable=SC2034 # identity constants are self-test contract data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/microwakeword"
REFERENCE_PROJECT="$ROOT/tools/parity/microwakeword-reference"
INSPECTOR="$ROOT/tools/parity/microwakeword_inspect.py"
TENSOR_MANIFEST_PRODUCER="$ROOT/tools/parity/microwakeword_tensor_manifest.py"
CONVERTER_LOCK_SHA256="984703d5bafdd6c88006bd381095961d42ef684d269d66194edbeda1fddf8dc2"
REFERENCE_LOCK_SHA256="736fca6145c24984531ef11258cd64aebbb188fa8830300b09232cac0fe567f3"
DEPENDENCY_EVIDENCE_SHA256="2b24695d106665b5cbc17357b1a43ff03ab75235d35e7d3ed03e5c7c7a68069d"
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
MODEL_ARTIFACT_BYTES_SHA256="21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77"
MODEL_COMPANION_GIT_BLOB="e6733fe13852f04a5a3ae83e0d39b5726aee62cc"
MODEL_COMPANION_SIZE=388
LICENSE_GIT_BLOB="261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
LICENSE_SIZE=11357
REVIEWED_TOPOLOGY_SHA256="e17fa0cae8d504ce71b49ad2113fc6f7ebba9e74dd4070d26e7f291dcbfaf621"
RAW_INVENTORY_SHA256="ce57a719f60af3a494cbd8fb22ff30fdb405b0a3037b049333f25f5794749989"
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

# Candidate conversion is a separate VAST-only, NO_UPLOAD path. It consumes
# fixed authenticated bytes and fixed raw inventory already materialized by the
# operator; it never fetches arbitrary model URLs/names or unlocks production.
candidate_conversion() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
  [[ "${VOKRA_CANDIDATE_CONVERSION:-0}" == 1 ]] || die "VOKRA_CANDIDATE_CONVERSION=1 is absent"
  [[ "$#" == 4 ]] || die "--candidate requires input raw-inventory candidate-manifest output"
  local input_path="$1" inventory_path="$2" manifest_path="$3" output_path="$4"
  [[ -f "$input_path" && ! -L "$input_path" && -f "$inventory_path" && ! -L "$inventory_path" ]] || die "candidate inputs must be regular non-symlink files"
  cd "$ROOT"
  command -v realpath >/dev/null 2>&1 || die "missing tool: realpath"
  input_path="$(realpath -e -- "$input_path")" || die "candidate input cannot be canonicalized"
  inventory_path="$(realpath -e -- "$inventory_path")" || die "candidate inventory cannot be canonicalized"
  local output_parent manifest_parent
  output_parent="$(realpath -e -- "$(dirname -- "$output_path")")" || die "candidate output parent cannot be canonicalized"
  manifest_parent="$(realpath -e -- "$(dirname -- "$manifest_path")")" || die "candidate manifest parent cannot be canonicalized"
  output_path="$output_parent/$(basename -- "$output_path")"
  manifest_path="$manifest_parent/$(basename -- "$manifest_path")"
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
  [[ -f "$PROJECT/uv.lock" ]] || die "dedicated uv.lock is absent"
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$CONVERTER_LOCK_SHA256" ]] || die "converter uv.lock identity mismatch"
  [[ -f "$input_path" && ! -L "$input_path" && -f "$inventory_path" && ! -L "$inventory_path" ]] || die "candidate inputs must be regular non-symlink files"
  [[ "$input_path" != "$ROOT"/* && "$inventory_path" != "$ROOT"/* ]] || die "candidate inputs must be outside checkout root"
  [[ "$output_path" != "$ROOT"/* && "$manifest_path" != "$ROOT"/* ]] || die "candidate outputs must be outside checkout root"
  [[ ! -e "$output_path" && ! -L "$output_path" && ! -e "$manifest_path" && ! -L "$manifest_path" ]] || die "candidate outputs must be absent"
  [[ "$(sha256sum "$input_path" | awk '{print $1}')" == "$MODEL_ARTIFACT_BYTES_SHA256" ]] || die "AUTHENTICATED_PAYLOAD_SHA_REQUIRED"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python "$PROJECT/prepare_checkpoint.py" \
    --candidate --input "$input_path" --raw-inventory "$inventory_path" \
    --candidate-manifest "$manifest_path" --name hey_jarvis --output "$output_path"
  echo "candidate conversion complete: dense GGML_TYPE_I8 logical tensors; NO_UPLOAD, CANDIDATE_UNREVIEWED" >&2
}

# Reviewed production conversion is intentionally a separate explicit mode.
# It consumes only the fixed raw inventory and TFLite bytes already materialized
# on VAST, stamps the compiled topology authority, and never uploads output.
reviewed_conversion() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
  [[ "${VOKRA_REVIEWED_CONVERSION:-0}" == 1 ]] || die "VOKRA_REVIEWED_CONVERSION=1 is absent"
  [[ "$#" == 3 ]] || die "--reviewed requires input raw-inventory output"
  local input_path="$1" inventory_path="$2" output_path="$3"
  [[ -f "$input_path" && ! -L "$input_path" && -f "$inventory_path" && ! -L "$inventory_path" ]] || die "reviewed inputs must be regular non-symlink files"
  cd "$ROOT"
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
  [[ -f "$PROJECT/uv.lock" ]] || die "dedicated uv.lock is absent"
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$CONVERTER_LOCK_SHA256" ]] || die "converter uv.lock identity mismatch"
  [[ "$(sha256sum "$input_path" | awk '{print $1}')" == "$MODEL_ARTIFACT_BYTES_SHA256" ]] || die "AUTHENTICATED_PAYLOAD_SHA_REQUIRED"
  [[ "$(sha256sum "$inventory_path" | awk '{print $1}')" == "$RAW_INVENTORY_SHA256" ]] || die "REVIEWED_RAW_INVENTORY_REQUIRED"
  [[ "$input_path" != "$ROOT"/* && "$inventory_path" != "$ROOT"/* && "$output_path" != "$ROOT"/* ]] || die "reviewed paths must be outside checkout root"
  [[ ! -e "$output_path" && ! -L "$output_path" ]] || die "reviewed output must be absent"
  export VOKRA_REVIEWED_CONVERSION=1
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python "$PROJECT/prepare_checkpoint.py" \
    --reviewed --input "$input_path" --raw-inventory "$inventory_path" \
    --name hey_jarvis --expected-sha256 "$MODEL_ARTIFACT_BYTES_SHA256" --output "$output_path"
  echo "reviewed production conversion complete: topology=$REVIEWED_TOPOLOGY_SHA256, NO_UPLOAD" >&2
}

# Verify one fixed Git blob without relying on a project dependency.
verify_git_blob() {
  local path="$1" expected_blob="$2" expected_size="$3"
  UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - \
    "$path" "$expected_blob" "$expected_size" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = path.read_bytes()
if len(payload) != int(sys.argv[3]):
    raise SystemExit(f"unexpected byte size for {path}")
actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"Git blob identity mismatch for {path}")
PY
}

VALIDATION_WORK_DIR=""

paths_disjoint() {
  local left="$1" right="$2"
  [[ "$left" != "$right" && "$left" != "$right"/* && "$right" != "$left"/* ]]
}

require_empty_directory() {
  local directory="$1"
  [[ -d "$directory" && ! -L "$directory" ]] || die "result directory must be an existing non-symlink directory"
  [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "result directory must be empty"
}

require_path_c_sentinel() {
  local log_path="$1"
  local sentinel='Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4'
  [[ "$(grep -Fxc -- "$sentinel" "$log_path" || true)" == 1 ]] || die "Path C authenticated sentinel must occur exactly once"
  [[ "$(grep -Ec '^test result: ok\. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$' "$log_path" || true)" == 1 ]] || die "Path C test result is not the exact 4/0/0/0/0 success line"
}

cleanup_validation_workdir() {
  local status=$?
  case "$VALIDATION_WORK_DIR" in
    /tmp/vokra-mww-validation.*)
      rm -rf -- "$VALIDATION_WORK_DIR" || echo "warning: unable to remove worker temporary directory $VALIDATION_WORK_DIR" >&2
      VALIDATION_WORK_DIR=""
      ;;
    "") ;;
    *) die "unsafe validation cleanup path" ;;
  esac
  return "$status"
}

# Reviewed end-to-end VAST path. The model, reviewed GGUF, and fixtures are
# worker-owned descendants of one private temporary root. Only the JSON result
# and test log belong in result_dir; no payload is copied there.
reviewed_validation() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
  [[ "${VOKRA_REVIEWED_VALIDATION:-0}" == 1 ]] || die "VOKRA_REVIEWED_VALIDATION=1 is absent"
  [[ "$#" == 3 ]] || die "--validate-reviewed requires raw-inventory dependency-evidence result-dir"
  local inventory_path="$1" dependency_evidence_path="$2" result_dir="$3"
  [[ "$inventory_path" == /* && "$dependency_evidence_path" == /* && "$result_dir" == /* ]] || die "reviewed paths must be absolute"
  [[ -f "$inventory_path" && ! -L "$inventory_path" && -f "$dependency_evidence_path" && ! -L "$dependency_evidence_path" ]] || die "reviewed evidence inputs must be regular non-symlink files"
  cd "$ROOT"
  for command in curl git uv sha256sum awk stat realpath find cargo rustc hostname nproc; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
  local commit system machine kernel host_name cpu_model cpu_flags cpu_count rustc_version cargo_version
  commit="$(git rev-parse HEAD)" || die "unable to record git commit"
  system="$(uname -s)"
  machine="$(uname -m)"
  kernel="$(uname -r)"
  host_name="$(hostname)" || die "unable to record hostname"
  cpu_count="$(nproc)" || die "unable to record nproc"
  [[ "$cpu_count" =~ ^[1-9][0-9]*$ ]] || die "invalid nproc evidence"
  [[ -r /proc/cpuinfo ]] || die "CPU evidence /proc/cpuinfo is unreadable"
  cpu_model="$(awk -F: '$1 == "model name" {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F: '$1 == "flags" {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU model/ISA flags evidence is missing"
  rustc_version="$(rustc -Vv)" || die "unable to record rustc version"
  cargo_version="$(cargo -V)" || die "unable to record cargo version"
  [[ -f "$PROJECT/uv.lock" && -f "$REFERENCE_PROJECT/uv.lock" ]] || die "dedicated uv.lock is absent"
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$CONVERTER_LOCK_SHA256" ]] || die "converter uv.lock identity mismatch"
  [[ "$(sha256sum "$REFERENCE_PROJECT/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die "reference uv.lock identity mismatch"
  inventory_path="$(realpath -e -- "$inventory_path")" || die "raw inventory cannot be canonicalized"
  dependency_evidence_path="$(realpath -e -- "$dependency_evidence_path")" || die "dependency evidence cannot be canonicalized"
  result_dir="$(realpath -e -- "$result_dir")" || die "result directory cannot be canonicalized"
  [[ "$inventory_path" != "$ROOT"/* && "$dependency_evidence_path" != "$ROOT"/* && "$result_dir" != "$ROOT"/* ]] || die "reviewed paths must be outside checkout root"
  require_empty_directory "$result_dir"
  for left in "$inventory_path" "$dependency_evidence_path" "$result_dir"; do
    for right in "$inventory_path" "$dependency_evidence_path" "$result_dir"; do
      [[ "$left" == "$right" ]] || paths_disjoint "$left" "$right" || die "reviewed paths must be canonically disjoint"
    done
  done
  [[ "$(sha256sum "$inventory_path" | awk '{print $1}')" == "$RAW_INVENTORY_SHA256" ]] || die "REVIEWED_RAW_INVENTORY_REQUIRED"
  [[ "$(sha256sum "$dependency_evidence_path" | awk '{print $1}')" == "$DEPENDENCY_EVIDENCE_SHA256" ]] || die "REVIEWED_DEPENDENCY_EVIDENCE_REQUIRED"

  export UV_CACHE_DIR="$UV_CACHE_DIR_VALUE"
  uv sync --project "$PROJECT" --frozen
  uv sync --project "$REFERENCE_PROJECT" --frozen
  VALIDATION_WORK_DIR="$(mktemp -d "/tmp/vokra-mww-validation.XXXXXX")"
  trap cleanup_validation_workdir EXIT
  paths_disjoint "$VALIDATION_WORK_DIR" "$ROOT" || die "validation work directory overlaps checkout"
  paths_disjoint "$VALIDATION_WORK_DIR" "$inventory_path" || die "validation work directory overlaps raw inventory"
  paths_disjoint "$VALIDATION_WORK_DIR" "$dependency_evidence_path" || die "validation work directory overlaps dependency evidence"
  paths_disjoint "$VALIDATION_WORK_DIR" "$result_dir" || die "validation work directory overlaps result directory"
  local tflite_path="$VALIDATION_WORK_DIR/hey_jarvis.tflite" license_path="$VALIDATION_WORK_DIR/LICENSE" companion_path="$VALIDATION_WORK_DIR/hey_jarvis.json"
  local output_path="$VALIDATION_WORK_DIR/hey_jarvis.reviewed.gguf" fixture_path="$VALIDATION_WORK_DIR/fixtures"
  [[ ! -e "$output_path" && ! -L "$output_path" && ! -e "$fixture_path" && ! -L "$fixture_path" ]] || die "worker output paths must be absent"
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --output "$tflite_path" "$DEFAULT_UPSTREAM_URL"
  [[ "$(stat -c '%s' "$tflite_path")" == "$MODEL_TARGET_SIZE" ]] || die "canonical artifact size mismatch"
  [[ "$(sha256sum "$tflite_path" | awk '{print $1}')" == "$MODEL_ARTIFACT_BYTES_SHA256" ]] || die "AUTHENTICATED_PAYLOAD_SHA_REQUIRED"
  verify_git_blob "$tflite_path" "$MODEL_TARGET_GIT_BLOB" "$MODEL_TARGET_SIZE"
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --output "$license_path" "$LICENSE_URL"
  verify_git_blob "$license_path" "$LICENSE_GIT_BLOB" "$LICENSE_SIZE"
  curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --output "$companion_path" "$COMPANION_URL"
  verify_git_blob "$companion_path" "$MODEL_COMPANION_GIT_BLOB" "$MODEL_COMPANION_SIZE"

  uv run --project "$PROJECT" --offline --no-sync --python 3.12 python "$PROJECT/prepare_checkpoint.py" \
    --reviewed --input "$tflite_path" --raw-inventory "$inventory_path" \
    --name hey_jarvis --expected-sha256 "$MODEL_ARTIFACT_BYTES_SHA256" --output "$output_path"
  uv run --project "$REFERENCE_PROJECT" --offline --no-sync --python 3.12 python "$PROJECT/dump_reference.py" \
    --tflite-path "$tflite_path" --dependency-evidence "$dependency_evidence_path" --output-dir "$fixture_path"
  local path_c_log="$result_dir/path-c.log"
  [[ ! -e "$path_c_log" && ! -L "$path_c_log" ]] || die "Path C log destination exists"
  if ! (set -o noclobber; VOKRA_KWS_REAL_GGUF="$output_path" VOKRA_KWS_REAL_FIXTURES="$fixture_path" CARGO_BUILD_JOBS=1 \
    cargo test --locked -p vokra-kws-micro --test parity_microwakeword -- --nocapture >"$path_c_log" 2>&1
  ); then
    tail -n 80 "$path_c_log" >&2 || true
    die "authenticated Path C parity failed; see $path_c_log"
  fi
  require_path_c_sentinel "$path_c_log"
  local result_manifest="$result_dir/microwakeword-validation.json"
  [[ ! -e "$result_manifest" && ! -L "$result_manifest" ]] || die "validation result destination exists"
  uv run --no-project --offline --python 3.12 python - "$result_manifest" "$output_path" "$fixture_path" "$path_c_log" \
    "$commit" "$CONVERTER_LOCK_SHA256" "$REFERENCE_LOCK_SHA256" "$MODEL_ARTIFACT_BYTES_SHA256" \
    "$DEPENDENCY_EVIDENCE_SHA256" "$RAW_INVENTORY_SHA256" "$system" "$machine" \
    "$kernel" "$host_name" "$cpu_count" "$cpu_model" "$cpu_flags" "$rustc_version" "$cargo_version" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result, gguf, fixture, path_c_log = map(Path, sys.argv[1:5])
commit, converter_lock, reference_lock, tflite_sha, evidence_sha, inventory_sha, system, machine, kernel, host_name, cpu_count, cpu_model, cpu_flags, rustc_version, cargo_version = sys.argv[5:]

def reject_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate fixture manifest key: {key}")
        value[key] = item
    return value

fixture_manifest_path = fixture / "manifest.json"
if fixture_manifest_path.is_symlink() or not fixture_manifest_path.is_file():
    raise SystemExit("fixture manifest is absent or symlinked")
try:
    fixture_manifest = json.loads(
        fixture_manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit("fixture manifest is not strict JSON") from error
if not isinstance(fixture_manifest, dict):
    raise SystemExit("fixture manifest must be an object")
if fixture_manifest.get("schema") != "microwakeword-reference-v2":
    raise SystemExit("fixture manifest schema drift")
if fixture_manifest.get("status") != "REFERENCE_COMPLETE":
    raise SystemExit("fixture manifest is not complete")
if fixture_manifest.get("source_tflite_sha256") != tflite_sha:
    raise SystemExit("fixture source TFLite SHA drift")
persistent = fixture_manifest.get("persistent_sequence")
if not isinstance(persistent, dict) or persistent.get("invocation_count") != 4:
    raise SystemExit("fixture persistent invocation count drift")
replay = persistent.get("fresh_interpreter_reset_replay")
if not isinstance(replay, dict) or replay.get("status") != "PASS" or replay.get("raw_outputs_match") is not True:
    raise SystemExit("fixture persistent replay is not PASS")
stress = fixture_manifest.get("direct_int8_stress")
indices = [47, 50, 51, 54, 55, 58, 59, 62, 63, 67, 68, 69]
if not isinstance(stress, dict) or stress.get("invocation_count") != 512 or stress.get("stage_tensor_indices") != indices:
    raise SystemExit("fixture stress trace contract drift")
artefacts = fixture_manifest.get("artefacts")
expected_names = {"input_pcm.bin": "input_pcm", "features_ref.bin": "features_ref", "output_ref.bin": "output_ref", "stress_inputs.bin": "stress_inputs"}
for invocation in range(4):
    expected_names.update({
        f"features_invocation_{invocation:02}.bin": f"features_invocation_{invocation:02}",
        f"input_invocation_{invocation:02}.bin": f"input_invocation_{invocation:02}",
        f"output_invocation_{invocation:02}.bin": f"output_invocation_{invocation:02}",
        f"output_invocation_{invocation:02}_f32.bin": f"output_invocation_{invocation:02}_f32",
    })
for index in indices:
    expected_names[f"stress_stage_tensor_{index}.bin"] = f"stress_stage_tensor_{index}"
if not isinstance(artefacts, list) or len(artefacts) != len(expected_names):
    raise SystemExit("fixture artefact count drift")
fixture_root = fixture.resolve()
seen_filenames = set()
for artefact in artefacts:
    if not isinstance(artefact, dict):
        raise SystemExit("malformed fixture artefact row")
    filename = artefact.get("path")
    declared_bytes = artefact.get("bytes")
    declared_sha = artefact.get("sha256")
    if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
        raise SystemExit("fixture artefact path escaped fixture directory")
    if filename in seen_filenames:
        raise SystemExit(f"duplicate fixture artefact: {filename}")
    seen_filenames.add(filename)
    if filename not in expected_names or artefact.get("name") != expected_names[filename]:
        raise SystemExit(f"unexpected fixture artefact: {filename}")
    candidate_path = fixture / filename
    if candidate_path.is_symlink():
        raise SystemExit(f"fixture artefact is symlinked: {filename}")
    candidate = candidate_path.resolve()
    try:
        candidate.relative_to(fixture_root)
    except ValueError as error:
        raise SystemExit("fixture artefact path escaped fixture directory") from error
    if not candidate.is_file():
        raise SystemExit(f"fixture artefact is absent: {filename}")
    payload = candidate.read_bytes()
    if declared_bytes != len(payload) or declared_sha != hashlib.sha256(payload).hexdigest():
        raise SystemExit(f"fixture artefact hash/size drift: {filename}")
if seen_filenames != set(expected_names):
    raise SystemExit("fixture artefact set is incomplete")
directory_names = set()
for entry in fixture.iterdir():
    if entry.name == "manifest.json":
        continue
    if entry.is_symlink() or not entry.is_file():
        raise SystemExit(f"unexpected non-regular fixture entry: {entry.name}")
    directory_names.add(entry.name)
if directory_names != set(expected_names):
    raise SystemExit("fixture directory has extra or missing regular files")

def identity(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"filename": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

summary = {
    "status": "PATH_C_PASS",
    "publication": "NO_UPLOAD",
    "model_payload_transfer": "TEMPORARY_VAST_ONLY",
    "git_commit": commit,
    "converter_lock_sha256": converter_lock,
    "reference_lock_sha256": reference_lock,
    "source_tflite_sha256": tflite_sha,
    "dependency_evidence_sha256": evidence_sha,
    "raw_inventory_sha256": inventory_sha,
    "host": {"system": system, "machine": machine, "kernel": kernel, "hostname": host_name, "python": "3.12", "nproc": int(cpu_count), "cpu_model": cpu_model, "cpu_flags": cpu_flags, "rustc_version": rustc_version, "cargo_version": cargo_version},
    "reviewed_topology_sha256": "e17fa0cae8d504ce71b49ad2113fc6f7ebba9e74dd4070d26e7f291dcbfaf621",
    "gguf": identity(gguf),
    "fixture_manifest": {"filename": fixture_manifest_path.name, "sha256": hashlib.sha256(fixture_manifest_path.read_bytes()).hexdigest(), "status": fixture_manifest.get("status"), "direct_int8_stress": fixture_manifest.get("direct_int8_stress")},
    "path_c_log": identity(path_c_log),
    "verification": {"test_identity": "parity_microwakeword::parity_microwakeword_end_to_end_output", "rust_test": "cargo test --locked -p vokra-kws-micro --test parity_microwakeword -- --nocapture", "sentinel": "Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4", "reset_replay_invocations": 4, "stress_invocations": 512, "preserved_intermediate_stage_count": 11, "final_output_tensor": 69},
}
with result.open("x", encoding="utf-8") as output:
    json.dump(summary, output, indent=2, sort_keys=True)
    output.write("\n")
PY
  echo "reviewed Path C validation complete: $result_manifest (NO_UPLOAD; model payload remains VAST-only)" >&2
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
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$CONVERTER_LOCK_SHA256" ]] || die "converter uv.lock identity mismatch"
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
  local self="${BASH_SOURCE[0]}" root fail=0
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/tools/parity/microwakeword_inspect.py" ]] || { echo "self-test FAIL: inspector missing" >&2; fail=1; }
  for needle in "microwakeword_inspect.py" "microwakeword_tensor_manifest.py" "run_authenticated_tensor_pipeline" "candidate_conversion" "reviewed_conversion" "reviewed_validation" "--validate-reviewed" "requires raw-inventory dependency-evidence result-dir" "VOKRA_REVIEWED_VALIDATION" "VOKRA_KWS_REAL_GGUF" "VOKRA_KWS_REAL_FIXTURES" "uv sync" "--frozen" "--no-sync" "--locked" "paths_disjoint" "require_empty_directory" "require_path_c_sentinel" "Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4" "test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out" "open(\"x\"" "git_commit" "converter_lock_sha256" "reference_lock_sha256" "source_tflite_sha256" "dependency_evidence_sha256" "raw_inventory_sha256" "model_payload_transfer" "preserved_intermediate_stage_count" "final_output_tensor" "--reviewed" "--candidate" "CANDIDATE_UNREVIEWED" "inspect_only" "--inspect-only" "--inventory-only" "RAW_INVENTORY_ONLY_NO_CONVERSION" "raw-inventory" "EVIDENCE_ONLY_UNREVIEWED" "object_pairs_hook" "duplicate manifest JSON key" "realpath -e" "outside checkout root" "prepare_checkpoint.py" "--self-test" "$CONVERTER_LOCK_SHA256" "$REFERENCE_LOCK_SHA256" "$DEPENDENCY_EVIDENCE_SHA256" "$PACKAGE_COUNT" "$PACKAGE_ROWS_SHA256" "$LICENSE_ROWS_SHA256" "ZERO_EXTERNAL_DEPENDENCIES" "--dependency-gate" "BLOCKED_UNREVIEWED_ARTIFACT" "AUTHENTICATED_PAYLOAD_SHA_REQUIRED" "AUTHENTICATED_TOPOLOGY_REQUIRED" "SOURCE_TENSOR_MANIFEST_REQUIRED" "--tensor-manifest" "tensor-manifest-sha256" "NO_UPLOAD" "VAST" "$MODEL_REPOSITORY" "$SOURCE_REPOSITORY" "SOURCE_REVISION" "MODEL_REVISION" "$DEFAULT_UPSTREAM_URL" "$LICENSE_URL" "$COMPANION_URL" "4665173cd35f1cff9a61e06fc427f124766c488e" "05b65922cc433c9df13e98e32a7fe520758c837e" "$MODEL_TARGET_PATH" "$MODEL_TARGET_GIT_BLOB" "$MODEL_TARGET_SIZE" "$MODEL_COMPANION_GIT_BLOB" "$MODEL_COMPANION_SIZE" "$LICENSE_GIT_BLOB" "$LICENSE_SIZE" 'MODEL_ARTIFACT_BYTES_SHA256="21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77"' 'REVIEWED_TOPOLOGY_SHA256="e17fa0cae8d504ce71b49ad2113fc6f7ebba9e74dd4070d26e7f291dcbfaf621"'; do
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
  [[ -f "$root/tools/parity/microwakeword-reference/inspect.py" ]] || { echo 'self-test FAIL: reference inspector missing' >&2; fail=1; }
  validation_body="$(sed -n '/^reviewed_validation()/,/^}/p' "$self")"
  if ! grep -Fq -- "uv sync --project \"\$PROJECT\" --frozen" <<<"$validation_body" || ! grep -Fq -- "cargo test --locked -p vokra-kws-micro --test parity_microwakeword" <<<"$validation_body"; then
    echo 'self-test FAIL: reviewed validation lacks frozen sync and Path C' >&2
    fail=1
  fi
  if grep -E 'git push|--upload|--push|publish-one\.sh|upload\.sh' <<<"$validation_body" >/dev/null; then
    echo 'self-test FAIL: reviewed validation contains publication' >&2
    fail=1
  fi
  local worker_probe missing_sentinel_log bad_result_log good_sentinel_log cleanup_root cleanup_work existing_result
  worker_probe="$(mktemp -d /tmp/vokra-mww-worker-selftest.XXXXXX)"
  mkdir "$worker_probe/nonempty"
  touch "$worker_probe/nonempty/stale"
  if (require_empty_directory "$worker_probe/nonempty") 2>/dev/null; then
    echo 'self-test FAIL: nonempty result directory was accepted' >&2
    fail=1
  fi
  if paths_disjoint "$worker_probe" "$worker_probe/child"; then
    echo 'self-test FAIL: overlapping paths were accepted' >&2
    fail=1
  fi
  missing_sentinel_log="$worker_probe/missing.log"
  printf 'test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' >"$missing_sentinel_log"
  if (require_path_c_sentinel "$missing_sentinel_log") 2>/dev/null; then
    echo 'self-test FAIL: missing Path C sentinel was accepted' >&2
    fail=1
  fi
  bad_result_log="$worker_probe/bad-result.log"
  printf '%s\n4 passed; 0 failed\n' 'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' >"$bad_result_log"
  if (require_path_c_sentinel "$bad_result_log") 2>/dev/null; then
    echo 'self-test FAIL: abbreviated Path C result line was accepted' >&2
    fail=1
  fi
  existing_result="$worker_probe/existing-result.json"
  touch "$existing_result"
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --offline --python 3.12 python - "$existing_result" <<'PY'
import sys
from pathlib import Path
try:
    Path(sys.argv[1]).open("x").close()
except FileExistsError:
    print("exclusive result open self-test: PASS")
else:
    raise SystemExit("existing result file was accepted")
PY
  then
    echo 'self-test FAIL: result JSON exclusive open check failed' >&2
    fail=1
  fi
  good_sentinel_log="$worker_probe/good.log"
  printf '%s\ntest result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n' 'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' >"$good_sentinel_log"
  require_path_c_sentinel "$good_sentinel_log" || fail=1
  cleanup_root="$(mktemp -d /tmp/vokra-mww-cleanup-selftest.XXXXXX)"
  cleanup_work="$(mktemp -d /tmp/vokra-mww-validation.XXXXXX)"
  touch "$cleanup_work/worker.gguf" "$cleanup_root/keep.me"
  VALIDATION_WORK_DIR="$cleanup_work"
  cleanup_validation_workdir
  [[ ! -e "$cleanup_work" && -e "$cleanup_root/keep.me" ]] || {
    echo 'self-test FAIL: cleanup was not exact' >&2
    fail=1
  }
  rm -rf -- "$cleanup_root" "$worker_probe"
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
if [[ "${1:-}" == "--candidate" ]]; then
  shift
  candidate_conversion "$@"
  exit 0
fi
if [[ "${1:-}" == "--reviewed" ]]; then
  shift
  reviewed_conversion "$@"
  exit 0
fi
if [[ "${1:-}" == "--validate-reviewed" ]]; then
  shift
  reviewed_validation "$@"
  exit 0
fi
if [[ "${1:-}" == "--inspect-only" ]]; then
  inspect_only "$@"
  exit 0
fi
[[ $# == 0 ]] || die "arguments are not accepted; use --validate-reviewed on VAST"
  die "reviewed validation requires --validate-reviewed and its three external paths"
