#!/usr/bin/env bash
# VAST/Linux-only CosyVoice2 HiFT conversion and CPU parity worker.
#
# License closure staging/audit and the explicit owner gate are separate from
# the reviewed execution phase. This worker never changes the checked-in gate,
# publishes, uploads, or destroys its VAST instance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PROJECT="$ROOT/tools/parity/cosyvoice2_hift_reference"
PREFLIGHT="$PROJECT/preflight_linux_closure.py"
AUDIT="$PROJECT/audit_linux_closure.py"
GATE="$PROJECT/preflight_gate.py"
PREPARER="$ROOT/tools/parity/cosyvoice2_hift_prepare_checkpoint.py"
DUMPER="$ROOT/tools/parity/cosyvoice2_hift_dump_reference.py"
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'
SOURCE_REVISION='8555549e882236e6541748b1042d95693caa82ba'
GENERATOR_SHA256='f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626'
LICENSE_SHA256='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
MODEL_REPOSITORY='FunAudioLLM/CosyVoice2-0.5B'
MODEL_REVISION='eec1ae6c79877dbd9379285cf8789c9e0879293d'
MODEL_PATH='hift.pt'
MODEL_BYTES=83390254
MODEL_SHA256='3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879'
CONFIG_PATH='cosyvoice2.yaml'
CONFIG_BYTES=7330
CONFIG_SHA256='0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959'
CONFIG_BLOB_SHA1='bc19267bbfd373c9a760b7667a74349ddd487db1'
TENSOR_COUNT=328
TENSOR_MANIFEST_SHA256='cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c'
INSPECTION_MANIFEST_SHA256='6a134122b4b0bdc851b38ca1d41d42e185d70d513ebb3c2e8d15a42b279462ea'
MIN_MEM_GIB=8
MIN_TMPFS_GIB=4
MIN_MEM_KIB=$((MIN_MEM_GIB * 1024 * 1024))
MIN_TMPFS_KIB=$((MIN_TMPFS_GIB * 1024 * 1024))

log() { printf '[cosyvoice2-hift-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-cosyvoice2-hift-validation.sh \
  --license-manifest <absolute-approved-license-manifest.json> \
  --inspection-manifest <absolute-authenticated-inspection-manifest.json> \
  [--work-dir <absolute-empty-tmpfs-directory>]
       run-cosyvoice2-hift-validation.sh --closure-only \
  [--work-dir <absolute-empty-tmpfs-directory>]
       run-cosyvoice2-hift-validation.sh --self-test

`--closure-only` stages/audits the Linux wheel closure and stops after the
candidate evidence is verified; it never needs or accepts owner approval.
The normal validation phase requires the separately reviewed
absolute license manifest to be APPROVED/OWNER_SIGNED_OFF. Only after that
gate passes does this worker sync the frozen Python project, acquire the fixed
source/model/config, prepare safetensors, dump the official CPU reference,
build the release converter, convert one GGUF, and run the ignored release
CPU parity test. No upload or publication is performed.
The inspection manifest must be the authenticated full-file manifest emitted
by the HiFT inspection runner; its nested `checkpoint` object is extracted
into a no-replace state-dict manifest for preparation.
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

require_absolute_file() {
  local path="$1" label="$2"
  [[ "$path" == /* ]] || die "$label must be absolute: $path"
  reject_symlink_ancestry "$path" "$label"
  [[ -f "$path" && ! -L "$path" ]] || die "$label must be an existing regular non-symlink file: $path"
}

disjoint() {
  local left right
  left="$(realpath -m "$1")"
  right="$(realpath -m "$2")"
  [[ "$left" != "$right" && "$left" != "$right"/* && "$right" != "$left"/* ]] \
    || die "paths overlap: $1 and $2"
}

validate_and_extract_inspection() {
  local destination="$1"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-hift-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$INSPECTION_MANIFEST" "$destination" "$INSPECTION_MANIFEST_SHA256" \
    "$MODEL_REPOSITORY" "$MODEL_REVISION" "$MODEL_PATH" "$MODEL_BYTES" "$MODEL_SHA256" \
    "$SOURCE_URL" "$SOURCE_REVISION" "$GENERATOR_SHA256" "$LICENSE_SHA256" \
    "$TENSOR_COUNT" "$TENSOR_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

inspection, destination, expected_sha, model_repo, model_revision, model_path, model_bytes, model_sha, source_url, source_revision, generator_sha, license_sha, tensor_count, tensor_manifest_sha = sys.argv[1:]
model_bytes = int(model_bytes)
tensor_count = int(tensor_count)

def reject_duplicate_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def fail(message):
    raise SystemExit(f"inspection manifest: {message}")

path = Path(inspection)
raw = path.read_bytes()
actual_sha = hashlib.sha256(raw).hexdigest()
if actual_sha != expected_sha:
    fail(f"full-file SHA-256 mismatch: {actual_sha}")
try:
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs)
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    fail(f"strict JSON parse failed: {error}")
if not isinstance(document, dict):
    fail("top-level value is not an object")
expected_keys = {
    "format", "status", "inspection_status", "evidence_stage", "runtime_status",
    "cpu_status", "metal_status", "parity_status", "publication", "artifact",
    "official_source", "checkpoint", "blockers",
}
if set(document) != expected_keys:
    fail(f"top-level schema mismatch: {sorted(document)}")
for key, value in {
    "format": "vokra-cosyvoice2-hift-inspection-v1",
    "status": "BLOCKED",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "evidence_stage": "INSPECTION_ONLY",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "cpu_status": "NOT_RUN",
    "metal_status": "NOT_RUN",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
}.items():
    if document.get(key) != value:
        fail(f"{key} mismatch")
artifact = document["artifact"]
expected_artifact = {
    "repository": model_repo,
    "revision": model_revision,
    "path": model_path,
    "url": f"https://huggingface.co/{model_repo}/resolve/{model_revision}/{model_path}?download=true",
    "bytes": model_bytes,
    "sha256": model_sha,
}
if artifact != expected_artifact:
    fail("artifact identity mismatch")
source = document["official_source"]
expected_source = {
    "repository": source_url,
    "revision": source_revision,
    "resolved_revision": source_revision,
    "clean": True,
    "role": "cosyvoice/hifigan/generator.py",
    "role_sha256": generator_sha,
    "role_git_blob_sha1": "326a1a70ae7707662939c20493b3a8e4b0906216",
    "license": "LICENSE",
    "license_sha256": license_sha,
}
if source != expected_source:
    fail("official source identity mismatch")
blockers = document["blockers"]
expected_blockers = [
    "Native HiFT binder is not implemented; this is structural evidence only.",
    "No CPU, Metal, or numerical parity execution was performed.",
]
if blockers != expected_blockers:
    fail("blocker list mismatch")
checkpoint = document["checkpoint"]
if not isinstance(checkpoint, dict):
    fail("checkpoint is not an object")
checkpoint_keys = {
    "format", "source", "data_pickle_sha256", "tensor_count", "manifest_sha256",
    "float_tensor_count", "float_manifest_sha256", "storage_manifest_sha256",
    "tensors", "archive", "payload_reads",
}
if set(checkpoint) != checkpoint_keys:
    fail("checkpoint schema mismatch")
if checkpoint["format"] != "vokra-pytorch-state-dict-manifest-v1" or checkpoint["source"] != f"hf://{model_repo}@{model_revision}:{model_path}/data.pkl":
    fail("checkpoint source/format mismatch")
if checkpoint["data_pickle_sha256"] != "1e71121d0cd47db0eaa93d5d9a6628ac73ab0f828433c8bfc60adc0118d9312d":
    fail("checkpoint data.pkl digest mismatch")
if checkpoint["tensor_count"] != tensor_count or checkpoint["float_tensor_count"] != tensor_count:
    fail("checkpoint tensor count mismatch")
if checkpoint["manifest_sha256"] != tensor_manifest_sha or checkpoint["float_manifest_sha256"] != tensor_manifest_sha:
    fail("checkpoint tensor manifest digest mismatch")
if checkpoint["storage_manifest_sha256"] != "44d9b17e9794cecd65b74537ce74c2c9ff7e1342b3d40a7f79b57599f86e84cb":
    fail("checkpoint storage manifest digest mismatch")
if checkpoint["payload_reads"] != "DATA_PKL_ONLY; TENSOR_STORAGE_MEMBERS_NOT_OPENED":
    fail("checkpoint payload-read contract mismatch")
tensors = checkpoint["tensors"]
if not isinstance(tensors, dict) or len(tensors) != tensor_count:
    fail("checkpoint tensor rows mismatch")
for name, row in tensors.items():
    if not isinstance(name, str) or not name or any(c in name for c in "\x00/\\"):
        fail(f"unsafe tensor name: {name!r}")
    if not isinstance(row, dict) or "shape" not in row:
        fail(f"tensor row is not an object with a shape: {name!r}")
    shape = row["shape"]
    if not isinstance(shape, list) or not all(type(dim) is int and dim >= 0 for dim in shape):
        fail(f"tensor shape is malformed: {name!r}")
archive = checkpoint["archive"]
if not isinstance(archive, dict) or set(archive) != {"member_count", "uncompressed_bytes", "data_pickle_member", "data_pickle_bytes", "storage_member_count", "storage_members"}:
    fail("checkpoint archive schema mismatch")
if (
    type(archive["member_count"]) is not int
    or type(archive["uncompressed_bytes"]) is not int
    or not isinstance(archive["data_pickle_member"], str)
    or archive["data_pickle_bytes"] != 41754
    or archive["storage_member_count"] != tensor_count
    or not isinstance(archive["storage_members"], list)
    or len(archive["storage_members"]) != tensor_count
    or len(set(archive["storage_members"])) != tensor_count
    or archive["member_count"] != 332
    or archive["uncompressed_bytes"] != 83326982
    or archive["data_pickle_member"] != "hift.new/data.pkl"
    or set(archive["storage_members"]) != {str(index) for index in range(tensor_count)}
):
    fail("checkpoint archive identity mismatch")
for name, row in tensors.items():
    if set(row) != {"dtype", "location", "shape", "storage_key", "storage_numel", "storage_offset", "stride"}:
        fail(f"tensor row schema mismatch: {name!r}")
    if row["dtype"] != "F32" or row["location"] != "cpu":
        fail(f"tensor row dtype/location mismatch: {name!r}")
    if not isinstance(row["storage_key"], str) or row["storage_key"] not in set(archive["storage_members"]):
        fail(f"tensor storage key mismatch: {name!r}")
    if type(row["storage_numel"]) is not int or row["storage_numel"] < 0 or type(row["storage_offset"]) is not int or row["storage_offset"] < 0:
        fail(f"tensor storage bounds malformed: {name!r}")
    stride = row["stride"]
    if not isinstance(stride, list) or not all(type(value) is int and value >= 0 for value in stride) or len(stride) != len(row["shape"]):
        fail(f"tensor stride malformed: {name!r}")
    if all(dim > 0 for dim in row["shape"]):
        max_offset = row["storage_offset"] + sum((dim - 1) * step for dim, step in zip(row["shape"], stride))
        if max_offset >= row["storage_numel"]:
            fail(f"tensor storage bounds exceeded: {name!r}")
if destination == inspection:
    fail("checkpoint extraction would overwrite inspection manifest")
output = Path(destination)
if output.exists() or output.is_symlink():
    fail("checkpoint extraction output already exists")
output.parent.mkdir(parents=True, exist_ok=True)
normalized = {
    "format": checkpoint["format"],
    "source": checkpoint["source"],
    "tensor_count": checkpoint["tensor_count"],
    "manifest_sha256": checkpoint["manifest_sha256"],
    "tensors": {name: {"dtype": row["dtype"], "shape": row["shape"]} for name, row in checkpoint["tensors"].items()},
}
payload = (json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
try:
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
except OSError as error:
    fail(f"checkpoint extraction output cannot be created exclusively: {error}")
try:
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view):]
    os.fsync(fd)
finally:
    os.close(fd)
print("INSPECTION_MANIFEST_VERIFIED")
print(f"inspection_manifest_sha256={actual_sha}")
print(f"checkpoint_manifest_sha256={checkpoint['manifest_sha256']}")
PY
}

write_status() {
  local status="$1" phase="$2" rc="${3:-0}" marker="${4:-}" destination="$EVIDENCE_DIR/status.json"
  [[ ! -e "$destination" && ! -L "$destination" ]] || return 0
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-project --offline --python 3.12 python - \
    "$destination" "$status" "$phase" "$rc" "$marker" "$GIT_COMMIT" "$LICENSE_MANIFEST" <<'PY'
import json
import os
import sys

destination, status, phase, rc, marker, git_commit, license_manifest = sys.argv[1:]
if status == "PASS":
    model_execution = reference_execution = native_cpu_parity = "RUN"
elif status == "CANDIDATE_VERIFIED" or phase in {"preflight", "license-gate", "project-sync", "source-model-acquisition"}:
    model_execution = reference_execution = native_cpu_parity = "NOT_RUN"
elif phase == "checkpoint-preparation":
    model_execution, reference_execution, native_cpu_parity = "NOT_RUN", "NOT_RUN", "NOT_RUN"
elif phase == "reference-execution":
    model_execution, reference_execution, native_cpu_parity = "NOT_RUN", "MAY_HAVE_RUN", "NOT_RUN"
elif phase == "conversion":
    model_execution, reference_execution, native_cpu_parity = "NOT_RUN", "RUN", "NOT_RUN"
else:
    model_execution, reference_execution, native_cpu_parity = "MAY_HAVE_RUN", "MAY_HAVE_RUN", "MAY_HAVE_RUN"
payload = {
    "format": "vokra-cosyvoice2-hift-vast-validation-v1",
    "status": status,
    "phase": phase,
    "return_code": int(rc),
    "publication": "NO_UPLOAD",
    "model_execution": model_execution,
    "reference_execution": reference_execution,
    "native_cpu_parity": native_cpu_parity,
    "success_marker": marker or None,
    "git_commit": git_commit,
    "license_manifest": license_manifest,
}
raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
try:
    view = memoryview(raw)
    while view:
        view = view[os.write(fd, view):]
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

SELF_TEST=0
CLOSURE_ONLY=0
LICENSE_MANIFEST=""
INSPECTION_MANIFEST=""
WORK="${COSYVOICE2_HIFT_WORK_DIR:-/dev/shm/vokra-cosyvoice2-hift-validation}"
while (($#)); do
  case "$1" in
    --self-test) (( SELF_TEST == 0 )) || die 'duplicate --self-test'; SELF_TEST=1; shift ;;
    --closure-only) (( CLOSURE_ONLY == 0 )) || die 'duplicate --closure-only'; CLOSURE_ONLY=1; shift ;;
    --license-manifest) (($# >= 2)) || die '--license-manifest requires a path'; LICENSE_MANIFEST="$2"; shift 2 ;;
    --inspection-manifest) (($# >= 2)) || die '--inspection-manifest requires a path'; INSPECTION_MANIFEST="$2"; shift 2 ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires a path'; WORK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token gate_line sync_line source_line model_line
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'CARGO_BUILD_JOBS=1' \
    'preflight_linux_closure.py' 'audit_linux_closure.py' 'preflight_gate.py' \
    'cosyvoice2_hift_prepare_checkpoint.py' 'cosyvoice2_hift_dump_reference.py' \
    'cosyvoice2-hift' '--license apache-2.0' 'hift.pt' 'cosyvoice2.yaml' \
    '3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879' \
    'f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626' \
    'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4' \
    'cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c' \
    '6a134122b4b0bdc851b38ca1d41d42e185d70d513ebb3c2e8d15a42b279462ea' \
    '--inspection-manifest' 'INSPECTION_MANIFEST_VERIFIED' 'checkpoint' \
    'APPROVED' 'OWNER_SIGNED_OFF' 'OWNER_REVIEW_REQUIRED' 'NO_UPLOAD' 'CANDIDATE_VERIFIED' \
    'reference_execution' 'native_cpu_parity' 'model_execution = reference_execution = native_cpu_parity = "NOT_RUN"' 'checkpoint-preparation' 'MAY_HAVE_RUN' \
    'CARGO_BUILD_JOBS' 'VOKRA_COSYVOICE2_HIFT_GGUF' 'parity_cosyvoice2_hift_real_cpu_parity' 'test result: ok' '--ignored --exact --nocapture' \
    '--closure-only' 'source/model acquisition' 'success_marker' 'os.O_EXCL' \
    'uv sync --frozen' 'cargo build --locked --release -p vokra-convert' \
    'cargo test --locked --release -p vokra-models'; do
    grep -Fq -- "$token" "$path" || { log "self-test missing token: $token"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\\.sh|.*publish-one\\.sh|--push|--upload|vokra-cli[[:space:]]+run)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test found publication/upload/model-run command'; fail=1
  fi
  if grep -En '^[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test found bare Python/pip command'; fail=1
  fi
  bash -n "$path" || fail=1
  gate_line="$(grep -n 'license gate PASS' "$path" | head -n1 | cut -d: -f1)"
  sync_line="$(grep -n 'uv sync --frozen' "$path" | tail -n1 | cut -d: -f1)"
  source_line="$(grep -n 'git clone --no-tags' "$path" | tail -n1 | cut -d: -f1)"
  model_line="$(grep -n 'downloading fixed' "$path" | tail -n1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ && "$source_line" =~ ^[0-9]+$ && "$model_line" =~ ^[0-9]+$ ]] || { log 'self-test cannot establish phase order'; fail=1; }
  (( gate_line < sync_line && gate_line < source_line && gate_line < model_line )) || { log 'self-test gate/order contract failed'; fail=1; }
  (( fail == 0 )) || return 1
  log 'self-test PASS (network/model/Cargo execution not run)'
}

if (( SELF_TEST )); then
  [[ -z "$LICENSE_MANIFEST$INSPECTION_MANIFEST" && "$CLOSURE_ONLY" == 0 && "$WORK" == "${COSYVOICE2_HIFT_WORK_DIR:-/dev/shm/vokra-cosyvoice2-hift-validation}" ]] || die '--self-test accepts no paths'
  self_test
  exit $?
fi

if (( CLOSURE_ONLY == 0 )); then
  [[ -n "$LICENSE_MANIFEST" && -n "$INSPECTION_MANIFEST" ]] || { usage >&2; exit 1; }
  require_absolute_file "$LICENSE_MANIFEST" '--license-manifest'
  require_absolute_file "$INSPECTION_MANIFEST" '--inspection-manifest'
else
  [[ -z "$LICENSE_MANIFEST$INSPECTION_MANIFEST" ]] || die '--closure-only accepts neither license nor inspection manifest'
fi
[[ "$WORK" == /* ]] || die '--work-dir must be absolute'
reject_symlink_ancestry "$WORK" '--work-dir'
[[ ! -e "$WORK" && ! -L "$WORK" ]] || die '--work-dir must be absent before the run'
[[ "$(uname -s)" == Linux ]] || die 'Linux VAST worker required'
[[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST worker required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is required'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'Vokra clean checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
for command in cargo curl df find findmnt git grep ln mktemp realpath sha256sum stat tee uv; do
  command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"
done
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_MEM_KIB" ]] || die "RAM below ${MIN_MEM_GIB} GiB"
work_parent="$(dirname "$WORK")"
[[ -d "$work_parent" && ! -L "$work_parent" ]] || die 'work parent must be an existing non-symlink directory'
[[ "$(findmnt -T "$work_parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_TMPFS_KIB" ]] || die "tmpfs free space below ${MIN_TMPFS_GIB} GiB"
mkdir "$WORK"
WORK="$(cd "$WORK" && pwd)"
EVIDENCE_DIR="$WORK/evidence"
mkdir "$EVIDENCE_DIR"
UV_CACHE_DIR="${COSYVOICE2_HIFT_UV_CACHE_DIR:-$WORK/uv-cache}"
export UV_CACHE_DIR VOKRA_PUBLISH_ON_VAST CARGO_BUILD_JOBS=1
GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
PHASE='preflight'
FINALIZED=0
# shellcheck disable=SC2329  # invoked indirectly by the EXIT trap below
on_exit() {
  local rc=$?
  if (( FINALIZED == 0 )); then
    write_status FAILED "$PHASE" "$rc" '' || true
  fi
  exit "$rc"
}
trap on_exit EXIT

if (( CLOSURE_ONLY == 0 )); then
  disjoint "$LICENSE_MANIFEST" "$WORK"
  disjoint "$INSPECTION_MANIFEST" "$WORK"
fi
[[ -f "$PREFLIGHT" && -f "$AUDIT" && -f "$GATE" && -f "$PREPARER" && -f "$DUMPER" ]] || die 'HiFT gate/preparer/dumper files are missing'
[[ ! -e "$WORK/linux-wheels" && ! -e "$WORK/linux-closure-candidate.json" ]] || die 'phase outputs must be absent'

if (( CLOSURE_ONLY == 0 )); then
  INSPECTION_MANIFEST_CHECKPOINT="$WORK/checkpoint-manifest.json"
  validate_and_extract_inspection "$INSPECTION_MANIFEST_CHECKPOINT" 2>&1 | tee "$EVIDENCE_DIR/inspection-manifest.log"
  [[ -f "$INSPECTION_MANIFEST_CHECKPOINT" && ! -L "$INSPECTION_MANIFEST_CHECKPOINT" ]] || die 'checkpoint manifest extraction missing'
fi

log 'phase 1/6: stage and audit Linux closure (no source/model/import/Cargo)'
VOKRA_PUBLISH_ON_VAST=1 uv run --no-project --offline --python 3.12 python "$PREFLIGHT" \
  --lock "$PROJECT/uv.lock" --output "$WORK/linux-wheels" 2>&1 | tee "$EVIDENCE_DIR/linux-preflight.log"
VOKRA_PUBLISH_ON_VAST=1 uv run --no-project --offline --python 3.12 python "$AUDIT" \
  --lock "$PROJECT/uv.lock" --project "$PROJECT/pyproject.toml" \
  --artifacts "$WORK/linux-wheels" --output "$WORK/linux-closure-candidate.json" \
  2>&1 | tee "$EVIDENCE_DIR/linux-audit.log"
uv run --no-project --offline --python 3.12 python - "$WORK/linux-closure-candidate.json" "$PROJECT/pyproject.toml" "$PROJECT/uv.lock" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

class StrictError(ValueError):
    pass
def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise StrictError(f"duplicate JSON key: {key}")
        out[key] = value
    return out
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
candidate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=pairs)
if candidate.get("format") != "vokra-cosyvoice2-hift-linux-closure-candidate-v1" or candidate.get("status") != "OWNER_REVIEW_REQUIRED" or candidate.get("license_status") != "PENDING_PACKAGE_AND_NATIVE_PAYLOAD_REVIEW" or candidate.get("publication") != "NO_UPLOAD":
    raise SystemExit("closure candidate status/schema mismatch")
if sha(Path(sys.argv[1])) != "2f5174af6cff51dc2b71121861e989de793e6121d5ed88c890a45a308daf55f9":
    raise SystemExit("closure candidate digest mismatch")
if candidate.get("project_sha256") != sha(Path(sys.argv[2])) or candidate.get("uv_lock_sha256") != sha(Path(sys.argv[3])):
    raise SystemExit("closure candidate project/lock digest mismatch")
wheels = candidate.get("wheels")
rows = candidate.get("package_rows")
if not isinstance(wheels, list) or len(wheels) != 12 or not isinstance(rows, list) or len(rows) != 13:
    raise SystemExit("closure candidate count mismatch")
licenses = sum(len(row.get("license_notice", [])) for row in wheels)
native = sum(len(row.get("native", [])) for row in wheels)
suspicious = sum(len(row.get("suspicious_markers", [])) for row in wheels)
if (licenses, native, suspicious, candidate.get("archive_aggregate_sha256")) != (44, 285, 1964, "dd7f26947e07f490e858d6311ba14008db6aa3ef2359de8742ec4093a5cabd3c"):
    raise SystemExit("closure candidate evidence counts/digest mismatch")
print("LINUX_CLOSURE_CANDIDATE_VERIFIED")
PY

if (( CLOSURE_ONLY )); then
  write_status CANDIDATE_VERIFIED "$PHASE" 0 LINUX_CLOSURE_CANDIDATE_VERIFIED
  FINALIZED=1
  log "LINUX_CLOSURE_CANDIDATE_VERIFIED evidence=$EVIDENCE_DIR"
  exit 0
fi

PHASE='license-gate'
log 'phase 2/6: run stdlib owner license gate before any sync/source/model action'
uv run --no-project --offline --python 3.12 python "$GATE" \
  --project "$PROJECT/pyproject.toml" --lock "$PROJECT/uv.lock" \
  --license-manifest "$LICENSE_MANIFEST" 2>&1 | tee "$EVIDENCE_DIR/license-gate.log"
log 'license gate PASS; source/model acquisition and project sync are now authorized by the supplied owner manifest'

PHASE='project-sync'
log 'phase 3/6: sync frozen approved reference project'
[[ ! -e "$WORK/venv" && ! -L "$WORK/venv" ]] || die 'isolated VAST environment must be absent'
UV_PROJECT_ENVIRONMENT="$WORK/venv" uv sync --frozen --project "$PROJECT" --python 3.12 2>&1 | tee "$EVIDENCE_DIR/uv-sync.log"

PHASE='source-model-acquisition'
source="$WORK/source/CosyVoice"
checkpoint="$WORK/model/$MODEL_PATH"
config="$WORK/model/$CONFIG_PATH"
prepared="$WORK/prepared/hift.safetensors"
gguf="$WORK/output/cosyvoice2-0.5b-hift.gguf"
reference="$WORK/reference"
mkdir -p "$WORK/source" "$WORK/model" "$WORK/prepared" "$WORK/output"
for path in "$source" "$checkpoint" "$config" "$prepared" "$gguf" "$reference"; do
  [[ ! -e "$path" && ! -L "$path" ]] || die "claimed output must be absent: $path"
done
log 'acquiring fixed source checkout after the license gate'
git clone --no-tags --filter=blob:none "$SOURCE_URL" "$source" 2>&1 | tee "$EVIDENCE_DIR/source-clone.log"
git -C "$source" checkout --detach "$SOURCE_REVISION" 2>&1 | tee "$EVIDENCE_DIR/source-checkout.log"
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'CosyVoice source revision mismatch'
[[ "$(git -C "$source" remote get-url origin)" == "$SOURCE_URL" ]] || die 'CosyVoice source origin mismatch'
[[ -z "$(git -C "$source" status --porcelain --untracked-files=all)" ]] || die 'CosyVoice source checkout is dirty'
log 'downloading fixed hift.pt after the license gate'
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors \
  --output "$checkpoint" "https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$MODEL_PATH?download=true" \
  2>&1 | tee "$EVIDENCE_DIR/model-download.log"
[[ "$(stat -c '%s' "$checkpoint")" == "$MODEL_BYTES" ]] || die 'hift.pt byte count mismatch'
[[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$MODEL_SHA256" ]] || die 'hift.pt SHA-256 mismatch'
log 'downloading fixed cosyvoice2.yaml after the license gate'
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors \
  --output "$config" "https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$CONFIG_PATH?download=true" \
  2>&1 | tee "$EVIDENCE_DIR/config-download.log"
[[ "$(stat -c '%s' "$config")" == "$CONFIG_BYTES" ]] || die 'cosyvoice2.yaml byte count mismatch'
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || die 'cosyvoice2.yaml SHA-256 mismatch'
[[ "$(git hash-object "$config")" == "$CONFIG_BLOB_SHA1" ]] || die 'cosyvoice2.yaml Git blob SHA-1 mismatch'

PHASE='reference-preparation'
log 'phase 4/6: prepare exact 328-tensor F32 safetensors and dump independent official CPU reference'
PHASE='checkpoint-preparation'
UV_PROJECT_ENVIRONMENT="$WORK/venv" VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --no-sync --project "$PROJECT" --python 3.12 python \
  - "$INSPECTION_MANIFEST_CHECKPOINT" "$TENSOR_MANIFEST_SHA256" "$TENSOR_COUNT" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
doc = json.loads(path.read_text(encoding="utf-8"))
if doc.get("format") != "vokra-pytorch-state-dict-manifest-v1" or doc.get("tensor_count") != int(sys.argv[3]) or doc.get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("tensor manifest identity mismatch")
tensors = doc.get("tensors")
if not isinstance(tensors, dict) or len(tensors) != int(sys.argv[3]):
    raise SystemExit("tensor manifest count mismatch")
canonical = bytearray()
for name in sorted(tensors):
    row = tensors[name]
    if not isinstance(name, str) or not name or any(c in name for c in "\x00/\\") or not isinstance(row, dict) or row.get("dtype") != "F32" or not isinstance(row.get("shape"), list) or not all(type(x) is int and x >= 0 for x in row["shape"]):
        raise SystemExit(f"invalid tensor manifest row: {name!r}")
    canonical.extend(name.encode())
    canonical.append(0)
    canonical.extend(struct.pack("<Q", len(row["shape"])))
    for dim in row["shape"]:
        canonical.extend(struct.pack("<q", dim))
if hashlib.sha256(canonical).hexdigest() != sys.argv[2]:
    raise SystemExit("tensor manifest canonical digest mismatch")
print("TENSOR_MANIFEST_VERIFIED")
PY
UV_PROJECT_ENVIRONMENT="$WORK/venv" VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --no-sync --project "$PROJECT" --python 3.12 python \
  "$PREPARER" --checkpoint "$checkpoint" --manifest "$INSPECTION_MANIFEST_CHECKPOINT" --license-manifest "$LICENSE_MANIFEST" --output "$prepared" \
  2>&1 | tee "$EVIDENCE_DIR/prepare.log"
[[ -f "$prepared" && ! -L "$prepared" ]] || die 'prepared safetensors output missing'
PHASE='reference-execution'
UV_PROJECT_ENVIRONMENT="$WORK/venv" VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --no-sync --project "$PROJECT" --python 3.12 python \
  "$DUMPER" --source "$source" --checkpoint "$checkpoint" --config "$config" --license-manifest "$LICENSE_MANIFEST" --output "$reference" \
  2>&1 | tee "$EVIDENCE_DIR/reference.log"
[[ -f "$reference/manifest.json" && ! -L "$reference/manifest.json" ]] || die 'reference completion manifest missing'

PHASE='conversion'
log 'phase 5/6: build release converter and convert the authenticated prepared input'
CARGO_TARGET_DIR="$WORK/cargo-target" cargo build --locked --release -p vokra-convert 2>&1 | tee "$EVIDENCE_DIR/cargo-build.log"
target="$WORK/cargo-target/release/vokra-convert"
[[ -x "$target" ]] || die 'release vokra-convert binary missing'
"$target" --model cosyvoice2-hift --input "$prepared" --config "$config" --license apache-2.0 --output "$gguf" \
  2>&1 | tee "$EVIDENCE_DIR/convert.log"
[[ -f "$gguf" && ! -L "$gguf" ]] || die 'converted GGUF missing'
GGUF_SHA256="$(sha256sum "$gguf" | awk '{print $1}')"
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'converted GGUF digest is not lowercase SHA-256'
log "strict conversion PASS gguf_sha256=$GGUF_SHA256"

PHASE='cpu-parity'
log 'phase 6/6: run ignored release CPU parity test with the measured GGUF digest binding'
VOKRA_PUBLISH_ON_VAST=1 VOKRA_COSYVOICE2_HIFT_LICENSE_MANIFEST="$LICENSE_MANIFEST" \
  VOKRA_COSYVOICE2_HIFT_GGUF="$gguf" VOKRA_COSYVOICE2_HIFT_GGUF_SHA256="$GGUF_SHA256" \
  VOKRA_COSYVOICE2_HIFT_REFERENCE_DIR="$reference" \
  CARGO_TARGET_DIR="$WORK/cargo-target" CARGO_BUILD_JOBS=1 \
  cargo test --locked --release -p vokra-models --test parity_cosyvoice2_hift_real cosyvoice2_hift_real_cpu_parity -- \
  --ignored --exact --nocapture 2>&1 | tee "$EVIDENCE_DIR/parity.log"
[[ "$(grep -Ec '^test cosyvoice2_hift_real_cpu_parity \.\.\. ok$' "$EVIDENCE_DIR/parity.log")" == 1 ]] || die 'named CPU parity test did not pass exactly once'
grep -Fq 'CosyVoice2 HiFT F0 CPU parity:' "$EVIDENCE_DIR/parity.log" || die 'F0 parity measurement marker missing'
grep -Fq 'CosyVoice2 HiFT PCM CPU parity:' "$EVIDENCE_DIR/parity.log" || die 'PCM parity measurement marker missing'
grep -Eq 'test result: ok\. 1 passed; 0 failed' "$EVIDENCE_DIR/parity.log" || die 'Cargo did not report one passing parity test'

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-project --offline --python 3.12 python - \
  "$EVIDENCE_DIR/final-evidence.json.tmp" "$WORK" "$GGUF_SHA256" "$GIT_COMMIT" "$LICENSE_MANIFEST" "$INSPECTION_MANIFEST" "$reference" "$checkpoint" "$config" "$prepared" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

destination, work, gguf_sha, git_commit, license_manifest, inspection_manifest, reference, checkpoint, config, prepared = sys.argv[1:]
root = Path(work)

def digest(path):
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            h.update(block)
    return {"bytes": size, "sha256": h.hexdigest()}

logs = {}
for path in sorted((root / "evidence").glob("*.log")):
    logs[path.name] = digest(path)["sha256"]
payload = {
    "format": "vokra-cosyvoice2-hift-vast-validation-evidence-v1",
    "status": "PASS",
    "success_marker": "COSYVOICE2_HIFT_CPU_PARITY_PASS",
    "publication": "NO_UPLOAD",
    "execution": {"model_execution": "RUN", "reference_execution": "RUN", "native_cpu_parity": "RUN", "metal": "NOT_RUN", "upload": "NO_UPLOAD"},
    "git_commit": git_commit,
    "license_manifest": {"path": license_manifest, **digest(license_manifest)},
    "inspection_manifest": {"path": inspection_manifest, **digest(inspection_manifest)},
    "inspection_manifest_sha256": "6a134122b4b0bdc851b38ca1d41d42e185d70d513ebb3c2e8d15a42b279462ea",
    "model": {"repository": "FunAudioLLM/CosyVoice2-0.5B", "revision": "eec1ae6c79877dbd9379285cf8789c9e0879293d", "path": "hift.pt", **digest(checkpoint)},
    "config": {"path": "cosyvoice2.yaml", **digest(config)},
    "tensor_manifest_sha256": "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c",
    "prepared": {"path": prepared, **digest(prepared)},
    "gguf": {"path": str(root / "output/cosyvoice2-0.5b-hift.gguf"), **digest(root / "output/cosyvoice2-0.5b-hift.gguf")},
    "gguf_sha256_env": gguf_sha,
    "reference_manifest": {"path": str(Path(reference) / "manifest.json"), **digest(Path(reference) / "manifest.json")},
    "logs_sha256": logs,
}
if payload["gguf"]["sha256"] != gguf_sha:
    raise SystemExit("final GGUF digest binding mismatch")
if payload["inspection_manifest"]["sha256"] != "6a134122b4b0bdc851b38ca1d41d42e185d70d513ebb3c2e8d15a42b279462ea":
    raise SystemExit("final inspection manifest digest binding mismatch")
raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
try:
    view = memoryview(raw)
    while view:
        view = view[os.write(fd, view):]
    os.fsync(fd)
finally:
    os.close(fd)
PY
ln "$EVIDENCE_DIR/final-evidence.json.tmp" "$EVIDENCE_DIR/final-evidence.json"
rm "$EVIDENCE_DIR/final-evidence.json.tmp"
write_status PASS "$PHASE" 0 COSYVOICE2_HIFT_CPU_PARITY_PASS
FINALIZED=1
log "COSYVOICE2_HIFT_CPU_PARITY_PASS evidence=$EVIDENCE_DIR/final-evidence.json gguf_sha256=$GGUF_SHA256"
exit 0
