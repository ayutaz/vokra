#!/usr/bin/env bash
# VAST/Linux-only, no-upload SGMSE-VoiceBank end-to-end orchestration.
# Existing authenticated runners own all model math and artifact handling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTION_RUNNER="$SCRIPT_DIR/run-sgmse-voicebank-inspection.sh"
PREPARE_RUNNER="$SCRIPT_DIR/run-sgmse-voicebank-prepare.sh"
SCORE_REFERENCE_RUNNER="$SCRIPT_DIR/run-sgmse-voicebank-reference.sh"
SCORE_PARITY_RUNNER="$SCRIPT_DIR/run-sgmse-native-score-parity.sh"
ENHANCEMENT_RUNNER="$SCRIPT_DIR/run-sgmse-native-enhancement-parity.sh"
WORK_DIR=""
VOKRA_COMMIT=""
SELF_TEST=0
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((32 * 1024 * 1024))

log() { printf '[sgmse-validation-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

usage() {
  cat <<'EOF'
usage: run-sgmse-voicebank-validation.sh --vokra-commit <40-hex> [--work-dir <absent-dir>]
       run-sgmse-voicebank-validation.sh --self-test

VAST/Linux x86_64 only, with at least 128 GiB RAM and 32 GiB free disk. The exact clean Vokra commit is inspected, prepared,
converted with explicit Apache-2.0, tested against independent score and
official waveform enhancement references, then sent through package,
workspace, clippy, deny, audit, and static gates. Status is NO_UPLOAD.
Transfer only small evidence and destroy the disposable VAST instance;
do not transfer generated model artifacts.
EOF
}

require_clean_commit() {
  [[ "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" == "$VOKRA_COMMIT" ]] || die 'Vokra commit changed or differs from --vokra-commit'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST Vokra checkout must be clean'
}

run_logged() {
  local label="$1"
  shift
  log "stage=$label"
  "$@" >"$LOG_DIR/$label.log" 2>&1 || {
    tail -n 80 "$LOG_DIR/$label.log" >&2 || true
    die "stage failed: $label"
  }
  tail -n 20 "$LOG_DIR/$label.log" >&2 || true
}

validate_prepared_sidecar() {
  local artifact="$1" sidecar="$2"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$artifact" "$sidecar" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

artifact, sidecar = map(pathlib.Path, sys.argv[1:])

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

manifest = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
expected_keys = {
    "format", "repository", "model_revision", "source_repository", "source_revision",
    "checkpoint_filename", "checkpoint_size", "checkpoint_sha256", "prepared_sha256",
    "tensor_count", "typed_manifest_sha256", "tensor_rows",
}
if set(manifest) != expected_keys:
    raise SystemExit("prepared sidecar schema mismatch")
expected = {
    "format": "vokra-sgmse-voicebank-prepared-v1",
    "repository": "speechbrain/sgmse-voicebank",
    "model_revision": "8f4ff7b65284c49492a43349b8106e094ac0d365",
    "source_repository": "https://github.com/sp-uhh/sgmse.git",
    "source_revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e",
    "checkpoint_filename": "score_model_ema.ckpt",
    "checkpoint_size": 262593305,
    "checkpoint_sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3",
}
for key, value in expected.items():
    if manifest[key] != value:
        raise SystemExit(f"prepared sidecar identity mismatch: {key}")
if not isinstance(manifest["tensor_count"], int) or manifest["tensor_count"] <= 0:
    raise SystemExit("prepared tensor count is invalid")
if not isinstance(manifest["tensor_rows"], list) or len(manifest["tensor_rows"]) != manifest["tensor_count"]:
    raise SystemExit("prepared tensor rows are invalid")
for key in ("prepared_sha256", "typed_manifest_sha256"):
    if not isinstance(manifest[key], str) or re.fullmatch(r"[0-9a-f]{64}", manifest[key]) is None:
        raise SystemExit(f"prepared sidecar digest is malformed: {key}")
if not artifact.is_file() or artifact.is_symlink():
    raise SystemExit("prepared artifact is missing or symlinked")
if manifest["prepared_sha256"] != hashlib.sha256(artifact.read_bytes()).hexdigest():
    raise SystemExit("prepared artifact digest mismatch")
print("PREPARED_SAFETENSORS_READY")
PY
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token previous=0 line label
  for token in \
    'run_logged inspection' 'run_logged preparation' 'run_logged build-converter' \
    'run_logged strict-conversion' 'run_logged score-reference' 'run_logged score-parity' \
    'run_logged enhancement-reference' 'run_logged enhancement-parity' 'run_logged metadata' \
    'run_logged package' 'run_logged workspace' 'run_logged clippy' 'run_logged deny' \
    'run_logged audit' 'run_logged static-fmt' 'run_logged static-forbidden' \
    'run_logged static-zero-deps' 'run_logged static-bound-arch' \
    'run_logged static-fixture-pins' 'run_logged static-dynamic-load' '--vokra-commit' \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'PREPARED_SAFETENSORS_READY' \
    'expected_keys' 'typed_manifest_sha256' 'cd "$VOKRA_ROOT"' '128 GiB' '32 GiB' \
    'STRICT_BIND_PASS' 'apache-2.0' \
    'sgmse_native_score_matches_independent_reference' \
    'sgmse_native_enhancement_matches_official_reference' \
    'SGMSE_NATIVE_SCORE_PARITY_PASS' 'CPU_ENHANCEMENT_PARITY_PASS' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings' \
    'cargo deny check licenses advisories bans' 'cargo audit' \
    'SGMSE_VALIDATION_COMPLETE_NO_UPLOAD' 'NO_UPLOAD' \
    'destroy the disposable VAST instance'; do
    grep -Fq -- "$token" "$path" || { log "self-test FAIL: missing contract token: $token"; fail=1; }
  done
  for label in inspection preparation build-converter strict-conversion score-reference score-parity enhancement-reference enhancement-parity metadata package workspace clippy deny audit static-fmt static-forbidden; do
    line="$(grep -nE "^run_logged $label([[:space:]]|$)" "$path" | head -n 1 | cut -d: -f1)"
    [[ -n "$line" && "$line" -gt "$previous" ]] || { log "self-test FAIL: stage order: $label"; fail=1; }
    previous="$line"
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if grep -Eq '^[[:space:]]*if[[:space:]]+prepared_data\.get\("publication"\)' "$path"; then
    log 'self-test FAIL: impossible prepared sidecar fields are being required'
    fail=1
  fi
  for runner in "$INSPECTION_RUNNER" "$PREPARE_RUNNER" "$SCORE_REFERENCE_RUNNER" "$SCORE_PARITY_RUNNER" "$ENHANCEMENT_RUNNER"; do
    bash "$runner" --self-test >/dev/null || { log "self-test FAIL: delegated runner: $runner"; fail=1; }
  done
  if "$path" --self-test --work-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  local schema_tmp
  schema_tmp="$(mktemp -d "${TMPDIR:-/tmp}/sgmse-validation-selftest.XXXXXX")"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$schema_tmp" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
artifact = root / "prepared.safetensors"
artifact.write_bytes(b"synthetic-prepared-artifact")
payload = {
    "format": "vokra-sgmse-voicebank-prepared-v1",
    "repository": "speechbrain/sgmse-voicebank",
    "model_revision": "8f4ff7b65284c49492a43349b8106e094ac0d365",
    "source_repository": "https://github.com/sp-uhh/sgmse.git",
    "source_revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e",
    "checkpoint_filename": "score_model_ema.ckpt",
    "checkpoint_size": 262593305,
    "checkpoint_sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3",
    "prepared_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "tensor_count": 1,
    "typed_manifest_sha256": "0" * 64,
    "tensor_rows": [{"name": "synthetic", "dtype": "torch.float32", "shape": [1]}],
}
(root / "prepared.manifest.json").write_text(json.dumps(payload), encoding="utf-8")
PY
  validate_prepared_sidecar "$schema_tmp/prepared.safetensors" "$schema_tmp/prepared.manifest.json" >/dev/null || fail=1
  rm -rf -- "$schema_tmp"
  (( fail == 0 )) || return 1
  log 'self-test PASS (model-free; no VAST mutation performed)'
}

while (($#)); do
  case "$1" in
    --self-test)
      (( SELF_TEST == 0 )) || die 'duplicate --self-test'
      SELF_TEST=1
      shift
      ;;
    --vokra-commit)
      [[ $# -ge 2 ]] || die '--vokra-commit requires a 40-hex commit'
      [[ -z "$VOKRA_COMMIT" ]] || die 'duplicate --vokra-commit'
      VOKRA_COMMIT="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die '--work-dir requires an absent directory'
      [[ -z "$WORK_DIR" ]] || die 'duplicate --work-dir'
      WORK_DIR="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if (( SELF_TEST )); then
  [[ -z "$VOKRA_COMMIT$WORK_DIR" ]] || die '--self-test accepts no arguments'
  self_test
  exit $?
fi

[[ "$VOKRA_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die '--vokra-commit must be a lowercase 40-hex commit'
[[ "$VOKRA_ROOT" == /* && -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ "$(uname -s)" == Linux ]] || die 'SGMSE validation is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'

mem_kib="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)"
[[ "${mem_kib:-0}" -ge "$MIN_VAST_MEM_KIB" ]] || die 'VAST host has less than 128 GiB RAM'
free_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR==2 {print $4}')"
[[ "${free_kib:-0}" -ge "$MIN_FREE_DISK_KIB" ]] || die 'VAST checkout filesystem has less than 32 GiB free'
for command in git uv cargo sha256sum awk df findmnt grep sort tee; do
  command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"
done
for runner in "$INSPECTION_RUNNER" "$PREPARE_RUNNER" "$SCORE_REFERENCE_RUNNER" "$SCORE_PARITY_RUNNER" "$ENHANCEMENT_RUNNER"; do
  [[ -x "$runner" ]] || die "runner is missing or not executable: $runner"
done

require_clean_commit
cd "$VOKRA_ROOT"
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="/workspace/vokra-sgmse-validation-$VOKRA_COMMIT"
fi
[[ "$WORK_DIR" == /* ]] || die '--work-dir must be absolute'
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || die '--work-dir must be absent (no-clobber)'
work_parent="$(dirname "$WORK_DIR")"
[[ -d "$work_parent" && ! -L "$work_parent" ]] || die 'work-dir parent must be an existing real directory'

INSPECTION_DIR="$WORK_DIR/inspection"
PREPARED_DIR="$WORK_DIR/prepared"
CONVERSION_DIR="$WORK_DIR/conversion"
SCORE_REFERENCE_DIR="$WORK_DIR/score-reference"
ENHANCEMENT_REFERENCE_DIR="$WORK_DIR/enhancement-reference"
LOG_DIR="$WORK_DIR/logs"
NATIVE_ROOT="/dev/shm/vokra-sgmse-validation-$VOKRA_COMMIT"
SCORE_NATIVE_DIR="$NATIVE_ROOT/score-native"
ENHANCEMENT_NATIVE_DIR="$NATIVE_ROOT/enhancement-native"
SUMMARY="$WORK_DIR/summary.json"
INPUT_WAV="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
CONVERTER="$VOKRA_ROOT/target/release/vokra-convert"
PREPARED="$PREPARED_DIR/sgmse_voicebank.safetensors"
PREPARED_MANIFEST="$PREPARED_DIR/sgmse_voicebank.manifest.json"
GGUF="$CONVERSION_DIR/sgmse-voicebank.gguf"

[[ -f "$INPUT_WAV" && ! -L "$INPUT_WAV" ]] || die 'fixed 16 kHz input fixture is missing or symlinked'
[[ ! -e "$NATIVE_ROOT" && ! -L "$NATIVE_ROOT" ]] || die 'native tmpfs work root must be absent (no-clobber)'
[[ "$(findmnt -T /dev/shm -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die '/dev/shm is not tmpfs'

mkdir "$WORK_DIR"
mkdir "$LOG_DIR"
mkdir "$NATIVE_ROOT"
cleanup() {
  rm -rf -- "$NATIVE_ROOT" 2>/dev/null || true
}
trap cleanup EXIT

run_logged inspection bash "$INSPECTION_RUNNER" --work-dir "$INSPECTION_DIR"
[[ -s "$INSPECTION_DIR/evidence/sgmse_voicebank_manifest.json" ]] || die 'inspection manifest is missing'

run_logged preparation bash "$PREPARE_RUNNER" \
  --input-dir "$INSPECTION_DIR/hf" --output-dir "$PREPARED_DIR"
[[ -s "$PREPARED" && -s "$PREPARED_MANIFEST" ]] || die 'prepared artifact or sidecar is missing'

validate_prepared_sidecar "$PREPARED" "$PREPARED_MANIFEST" >/dev/null

mkdir "$CONVERSION_DIR"
run_logged build-converter cargo build --locked --release -p vokra-convert
[[ -x "$CONVERTER" ]] || die 'vokra-convert release binary is missing'
run_logged strict-conversion "$CONVERTER" --model sgmse-voicebank \
  --input "$PREPARED" --license apache-2.0 --output "$GGUF"
[[ -s "$GGUF" && ! -L "$GGUF" ]] || die 'strict GGUF conversion output is missing'
GGUF_SHA256="$(sha256_file "$GGUF")"
log "STRICT_BIND_PASS gguf_sha256=$GGUF_SHA256 license=apache-2.0"

run_logged score-reference bash "$SCORE_REFERENCE_RUNNER" \
  --inspection-dir "$INSPECTION_DIR" --output-dir "$SCORE_REFERENCE_DIR"
run_logged score-parity bash "$SCORE_PARITY_RUNNER" \
  --gguf "$GGUF" --gguf-sha256 "$GGUF_SHA256" \
  --reference-dir "$SCORE_REFERENCE_DIR" --native-output-dir "$SCORE_NATIVE_DIR"

run_logged enhancement-reference bash "$ENHANCEMENT_RUNNER" --generate-reference \
  --inspection-dir "$INSPECTION_DIR" --reference-dir "$ENHANCEMENT_REFERENCE_DIR"
run_logged enhancement-parity bash "$ENHANCEMENT_RUNNER" \
  --gguf "$GGUF" --gguf-sha256 "$GGUF_SHA256" \
  --reference-dir "$ENHANCEMENT_REFERENCE_DIR" --native-output-dir "$ENHANCEMENT_NATIVE_DIR"

run_logged metadata cargo metadata --locked --no-deps --format-version 1
run_logged package cargo test --locked -p vokra-convert --lib
run_logged workspace cargo test --locked --workspace --all-targets
run_logged clippy cargo clippy --locked --workspace --all-targets -- -D warnings
run_logged deny cargo deny check licenses advisories bans
run_logged audit cargo audit
run_logged static-fmt cargo fmt --all -- --check
run_logged static-forbidden bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
run_logged static-zero-deps bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
run_logged static-bound-arch bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
run_logged static-fixture-pins bash "$VOKRA_ROOT/scripts/check-fixture-eol-pins.sh"
run_logged static-dynamic-load bash "$VOKRA_ROOT/scripts/check-no-dynamic-load.sh"

require_clean_commit
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
  "$SUMMARY" "$VOKRA_COMMIT" "$INPUT_WAV" "$INSPECTION_DIR" "$PREPARED" \
  "$PREPARED_MANIFEST" "$GGUF" "$SCORE_REFERENCE_DIR" "$SCORE_NATIVE_DIR" \
  "$ENHANCEMENT_REFERENCE_DIR" "$ENHANCEMENT_NATIVE_DIR" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

summary, commit, input_wav, inspection, prepared, prepared_manifest, gguf, score_ref, score_native, enhancement_ref, enhancement_native = map(pathlib.Path, sys.argv[1:])
commit = str(commit)

def evidence(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))

inspection_manifest = inspection / "evidence/sgmse_voicebank_manifest.json"
score_manifest = score_ref / "manifest.json"
enhancement_manifest = enhancement_ref / "manifest.json"
for path in (inspection_manifest, prepared, prepared_manifest, gguf, score_manifest, enhancement_manifest):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing authenticated output: {path}")
prepared_data = manifest(prepared_manifest)
expected_sidecar_keys = {
    "format", "repository", "model_revision", "source_repository", "source_revision",
    "checkpoint_filename", "checkpoint_size", "checkpoint_sha256", "prepared_sha256",
    "tensor_count", "typed_manifest_sha256", "tensor_rows",
}
if set(prepared_data) != expected_sidecar_keys:
    raise SystemExit("prepared sidecar schema drifted")
if prepared_data.get("prepared_sha256") != evidence(prepared)["sha256"]:
    raise SystemExit("prepared hash is not bound")
score_data = manifest(score_manifest)
enhancement_data = manifest(enhancement_manifest)
if score_data.get("status") != "REFERENCE_COMPLETE_NO_UPLOAD" or score_data.get("publication") != "NO_UPLOAD":
    raise SystemExit("score reference is not complete/no-upload")
if enhancement_data.get("status") != "REFERENCE_COMPLETE_NO_UPLOAD" or enhancement_data.get("publication") != "NO_UPLOAD":
    raise SystemExit("enhancement reference is not complete/no-upload")
if not pathlib.Path(input_wav).is_file():
    raise SystemExit("input fixture disappeared")
payloads = {
    "inspection_manifest": evidence(inspection_manifest),
    "input_fixture": evidence(input_wav),
    "prepared": evidence(prepared),
    "prepared_manifest": evidence(prepared_manifest),
    "gguf": evidence(gguf),
    "score_reference_manifest": evidence(score_manifest),
    "enhancement_reference_manifest": evidence(enhancement_manifest),
}
for name, directory in (("score_native", score_native), ("enhancement_native", enhancement_native)):
    files = sorted(path for path in directory.iterdir() if path.is_file() and not path.is_symlink())
    if not files:
        raise SystemExit(f"native output is empty: {directory}")
    payloads[name] = {path.name: evidence(path) for path in files}
data = {
    "format": "vokra-sgmse-validation-v1",
    "status": "SGMSE_VALIDATION_COMPLETE_NO_UPLOAD",
    "publication": "NO_UPLOAD",
    "vokra_commit": commit,
    "clean_checkout": True,
    "artifacts": payloads,
    "reference_status": {"score": score_data["status"], "enhancement": enhancement_data["status"]},
    "gates": {name: "PASS" for name in ("metadata", "package", "workspace", "clippy", "deny", "audit", "static")},
    "execution": {"model_execution": "VAST_ONLY", "local_model_execution": "NOT_RUN", "upload": "NO_UPLOAD"},
}
temporary = summary.with_name(f".{summary.name}.{os.getpid()}.tmp")
if summary.exists() or summary.is_symlink():
    raise SystemExit("summary already exists")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, summary)
temporary.unlink()
print(f"summary={summary}")
PY

log "SGMSE validation complete: summary=$SUMMARY status=SGMSE_VALIDATION_COMPLETE_NO_UPLOAD"
log 'Transfer only small manifests/logs and destroy the disposable VAST instance immediately; no upload/publication path exists.'
