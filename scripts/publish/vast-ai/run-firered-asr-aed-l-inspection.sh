#!/usr/bin/env bash
# shellcheck disable=SC2317
# VAST-only FireRedASR-AED-L inspection and safe checkpoint preparation.
# No runtime execution, parity claim, or publication.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# The pinned source/release contract and VAST checkpoint identity are recorded;
# native conversion/runtime/parity remain fail-closed pending operator work.
INSPECTOR="$ROOT/tools/parity/firered_asr_aed_l_inspect.py"
PREPARER="$ROOT/tools/parity/firered_asr_aed_l_prepare_checkpoint.py"
REFERENCE="$ROOT/tools/parity/firered_asr_aed_l_reference.py"
AUDITOR="$ROOT/tools/parity/firered_asr_aed_l_audit.py"
FIRERED_PROJECT="$ROOT/tools/parity/firered_asr_aed_l"
# The dedicated lock selects the official CPU-only wheel index:
# https://download.pytorch.org/whl/cpu (no CUDA/NVIDIA/Triton closure).
REPOSITORY="FireRedTeam/FireRedASR-AED-L"
REVISION="e57f5960d03cff1071ff7acbb409314d1e70ed3d"
SOURCE_URL="https://github.com/FireRedTeam/FireRedASR.git"
SOURCE_REVISION="834635e4cf277ed8ca92049fc375b17c3dc20748"
KALDI_NATIVE_FBANK_URL="https://github.com/csukuangfj/kaldi-native-fbank.git"
KALDI_NATIVE_FBANK_REVISION="f68c6b43f739697d7ab02ff6debacee130e1d541"
WORK="/dev/shm/vokra-firered-asr-aed-l-inspection"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((32 * 1024 * 1024))
UV_CACHE_DIR="${FIRERED_ASR_UV_CACHE_DIR:-/tmp/vokra-firered-asr-uv-cache}"

log() { printf '[firered-asr-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() { echo 'usage: run-firered-asr-aed-l-inspection.sh [--work-dir DIR] [--owner-approval JSON] | --self-test'; }

canonical_absent_candidate() {
  local candidate="$1" parent suffix resolved
  [[ "$candidate" == /* ]] || die 'path candidate must be absolute'
  [[ "$candidate" != *'/../'* && "$candidate" != */.. ]] || die 'path candidate must not contain ..'
  [[ ! -e "$candidate" && ! -L "$candidate" ]] || die "path candidate must be absent: $candidate"
  parent="$(dirname "$candidate")"
  suffix="$(basename "$candidate")"
  while [[ ! -e "$parent" ]]; do
    [[ "$parent" != / ]] || die "cannot resolve path parent: $candidate"
    suffix="$(basename "$parent")/$suffix"
    parent="$(dirname "$parent")"
  done
  [[ -d "$parent" && ! -L "$parent" ]] || die "path parent must be a non-symlink directory: $parent"
  resolved="$(cd "$parent" && pwd -P)" || die "cannot canonicalize path parent: $parent"
  printf '%s/%s\n' "$resolved" "$suffix"
}

canonical_existing_file() {
  local candidate="$1" parent
  [[ "$candidate" == /* ]] || die 'approval path must be absolute after argument resolution'
  [[ -f "$candidate" && ! -L "$candidate" ]] || die 'approval must be an existing regular JSON file, not a symlink'
  parent="$(cd "$(dirname "$candidate")" && pwd -P)" || die 'cannot canonicalize approval parent'
  printf '%s/%s\n' "$parent" "$(basename "$candidate")"
}

paths_overlap() {
  local left="$1" right="$2"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token path_test candidate
  path_test="$(mktemp -d /tmp/firered-path-selftest.XXXXXX)" || { log 'self-test FAIL: mktemp failed'; return 1; }
  candidate="$(canonical_absent_candidate "$path_test/nested/missing")" || fail=1
  path_test="$(cd "$path_test" && pwd -P)" || fail=1
  [[ "$candidate" == "$path_test/nested/missing" ]] || fail=1
  mkdir "$path_test/existing"
  if (canonical_absent_candidate "$path_test/existing") >/dev/null 2>&1; then fail=1; fi
  ln -s missing "$path_test/dangling"
  if (canonical_absent_candidate "$path_test/dangling") >/dev/null 2>&1; then fail=1; fi
  paths_overlap "$path_test" "$path_test/nested" || fail=1
  paths_overlap "$path_test/nested" "$path_test" || fail=1
  if paths_overlap "$path_test" "/tmp/another-root"; then fail=1; fi
  rm -rf "$path_test"
  (( fail == 0 )) || { log 'self-test FAIL: path candidate/overlap contract'; return 1; }
  for token in \
    'FireRedTeam/FireRedASR-AED-L' 'e57f5960d03cff1071ff7acbb409314d1e70ed3d' \
    'FireRedASR.git' '834635e4cf277ed8ca92049fc375b17c3dc20748' \
    'model.pth.tar' '12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3' \
    'train_bpe1000.model' '473bbc157cb4eade2059b30a3c877a1c29bd50cadbfbed869ae36eeade7fee07' \
    'model_info' 'list_repo_tree' 'path_in_repo' 'git_blob_sha1' 'lfs_sha256' 'weights_only=True' \
    '128' '32' '/dev/shm' 'findmnt' 'CARGO_BUILD_JOBS=1' 'status": "BLOCKED"' 'INSPECTION_ONLY' 'NO_UPLOAD' 'runtime_status_scope' 'full_pcm_transcription_only' \
    'config.yaml' 'BLOCKER_EMPTY_CONFIG' 'git ls-files' 'git status' \
    'source_contract' 'AUTHENTICATED_SOURCE_CONTRACT' 'SOURCE_FACTS_AUTHENTICATED' 'unlock_requirements' 'vast_first_pass' 'expected_artifacts' \
    'pinned-source frontend' 'SentencePiece/TokenDict' 'PREPARED' 'archive_members' \
    'tensor_count' 'publication' '--audit-output' 'BLOCKED_NOT_RUN' 'fp32_atol_status' \
    'firered_asr_aed_l_reference.py' 'tensor_mapping' 'REFERENCE_CAPTURED' 'decoder_logits' 'tgt_word_prj' 'source_records' 'firered-asr-aed-l-reference-trace-v1' 'encoder_each_layer' 'decoder_each_layer' 'frontend_fbank_cmvn' \
    'firered_asr_aed_l_audit.py' 'BLOCKED_UNREVIEWED_TRANSITIVE' 'OWNER_APPROVED' 'OWNER_REVIEW_REQUIRED' 'INVALID' 'distribution_evidence' 'distribution_evidence_sha256' 'lock_artifact' 'source_identity_aggregate' 'native_payloads' 'publisher_urls' 'publisher_url_aggregate' 'license_candidate_aggregate' 'native_payload_aggregate' 'review_ledger' 'exact_digest_gate' 'collection_protocol' 'owner_approval' 'owner-approval-v1' '--owner-approval' 'owner_approval_path' 'yousan' 'approved_at_utc' 'publisher_urls_sha256' 'license_candidates_sha256' 'native_payloads_sha256' 'scope_sha256' 'collection_failures' 'approved_mode' 'is_symlink' 'regular JSON' 'must not overlap' 'reject_duplicate_pairs' 'duplicate JSON key' 'exactly 27 active closure rows' 'native_source_license' 'source_revision_verified' 'source_url_verified' 'license_path' 'license_bytes' 'license_sha256_verified' 'approved_route_expected_artifacts' \
    'NamedTemporaryFile' 'os.link' 'manifest-with-preparation.json' \
    'manifest-with-reference.json' 'final no-clobber manifest' \
    'kaldiio==2.18.1' 'kaldi-native-fbank==1.15' 'name = "setuptools"' 'version = "83.0.0"' 'specifier = "==83.0.0"' 'importlib.metadata' 'setuptools>=83' '397a4cd18977acaae7acabfba6807ee0a6978c620064381a266eac15b3c1a0a0' \
    'f68c6b43f739697d7ab02ff6debacee130e1d541' 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30' \
    'uv lock --check' 'source/kaldi-native-fbank' 'setup.py' 'cmake' 'make' 'cc' 'c++' 'g++' 'native build toolchain' \
    'forbidden CUDA dependency row' 'download.pytorch.org/whl/cpu' 'license hash is not authenticated' \
    '--no-sync' 'FIRERED_PROJECT' 'firered_asr_aed_l/pyproject.toml' 'firered_asr_aed_l/uv.lock' \
    "cargo fmt --manifest-path \"\$ROOT/Cargo.toml\" --all -- --check" \
    "cargo metadata --manifest-path \"\$ROOT/Cargo.toml\" --locked --no-deps --format-version 1"; do
    if ! grep -Fq -- "$token" "$path"; then log "self-test FAIL: missing token $token"; fail=1; fi
  done
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
calls = re.findall(r"list_repo_tree\([^\n]*\)", source)
if not calls:
    raise SystemExit("FireRedASR tree walk call missing")
for call in calls:
    if "path_in_repo=" not in call or re.search(r"(?<![A-Za-z0-9_])path=", call):
        raise SystemExit(f"FireRedASR tree walk has incompatible path keyword: {call}")
PY
  then
    log 'self-test FAIL: frozen HfApi.list_repo_tree path_in_repo contract regression'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

def classify_entry(entry):
    if isinstance(entry, RepoFolder):
        if getattr(entry, "type", None) not in {None, "directory"}:
            raise RuntimeError("unknown RepoFolder type")
        return "directory"
    if isinstance(entry, RepoFile):
        if getattr(entry, "type", None) not in {None, "file"}:
            raise RuntimeError("unknown RepoFile type")
        return "file"
    raise RuntimeError(f"unknown HF tree entry: {entry!r}")

file_entry = RepoFile(path="README.md", size=1, oid="a" * 40)
file_entry.type = None
assert classify_entry(file_entry) == "file"
folder_entry = RepoFolder(path="nested", oid="b" * 40)
folder_entry.type = None
assert classify_entry(folder_entry) == "directory"
try:
    classify_entry(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unknown HF tree entry was accepted")
print("FireRedASR RepoFile/RepoFolder self-test: PASS")
PY
  then
    log 'self-test FAIL: RepoFile/RepoFolder class-identity regression'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text(encoding="utf-8")
audit = source.index('dependency-audit.json')
snapshot = source.index('\nfrom huggingface_hub import snapshot_download')
if audit >= snapshot:
    raise SystemExit("dependency audit must precede model snapshot")
if source.index('BLOCKED_UNREVIEWED_TRANSITIVE; owner review') < audit:
    raise SystemExit("dependency audit status gate is missing")
if source.index('\n  audit_args+=(--owner-approval') >= snapshot:
    raise SystemExit("approved owner artifact is not wired before model snapshot")
if source.index('\nexpected_status = "OWNER_APPROVED" if approved_mode') >= snapshot:
    raise SystemExit("approved packet verifier is not before model snapshot")
if source.index("\n  die 'FireRed dependency closure is BLOCKED_UNREVIEWED_TRANSITIVE") >= snapshot:
    raise SystemExit("no-approval block is not before model snapshot")
print("FireRed dependency gate ordering self-test: PASS")
PY
  then
    log 'self-test FAIL: dependency audit/model snapshot ordering regression'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text(encoding="utf-8")
final = source.index('\nfinal_manifest="$work_dir/evidence/manifest-with-reference.json"')
link = source.index('os.link(temporary, ' + 'final_path)')
if link <= final or "open(manifest_path, " + '"w"' in source:
    raise SystemExit("final reference merge is not a distinct no-clobber publication")
print("FireRed final manifest publication self-test: PASS")
PY
  then
    log 'self-test FAIL: final reference manifest no-clobber regression'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
import os, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix="firered-final-manifest-") as directory:
    root = Path(directory); destination = root / "final.json"; destination.write_text("sentinel", encoding="utf-8")
    temporary = root / ".final.json.tmp"; temporary.write_text("replacement", encoding="utf-8")
    try:
        os.link(temporary, destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError("final manifest race sentinel was overwritten")
    assert destination.read_text(encoding="utf-8") == "sentinel"
    temporary.unlink()
    assert not list(root.glob("*.tmp"))
print("FireRed final manifest race self-test: PASS")
PY
  then
    log 'self-test FAIL: final reference manifest race sentinel regression'
    fail=1
  fi
  if grep -En '^[[:space:]]*git[[:space:]]+push|^[[:space:]]*(curl|wget)[^#]*(upload|push)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'; fail=1
  fi
  # Self-tests only exercise stdlib/synthetic validation.  Avoid syncing the
  # VAST-only native-fbank git dependency here; production commands below use
  # the normal frozen project sync on Linux VAST.
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python "$PREPARER" --self-test >/dev/null || fail=1
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test >/dev/null || fail=1
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python "$AUDITOR" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="$WORK"
owner_approval_path=""
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; work_dir="$2"; shift 2 ;;
    --owner-approval) (($# >= 2)) || die '--owner-approval requires JSON'; owner_approval_path="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then [[ "$work_dir" == "$WORK" && -z "$owner_approval_path" ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
[[ "$(uname -s)" == Linux ]] || die 'Linux VAST required'
[[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$ROOT/tools/parity/pyproject.toml" && -f "$ROOT/tools/parity/uv.lock" ]] || die 'locked parity project missing'
[[ -f "$FIRERED_PROJECT/pyproject.toml" && -f "$FIRERED_PROJECT/uv.lock" ]] || die 'dedicated FireRed uv project missing'
[[ -f "$AUDITOR" ]] || die 'dedicated FireRed dependency auditor missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
(( mem_kib >= MIN_MEM_KIB )) || die '128 GiB memory guard failed'
[[ "$work_dir" == /* ]] || die '--work-dir must be absolute'
work_dir="$(canonical_absent_candidate "$work_dir")"
root_path="$(cd "$ROOT" && pwd -P)" || die 'cannot canonicalize checkout root'
if paths_overlap "$work_dir" "$root_path"; then
  die '--work-dir must not overlap the checkout'
fi
if [[ -n "$owner_approval_path" ]]; then
  [[ "$owner_approval_path" == /* ]] || owner_approval_path="$(cd "$(dirname "$owner_approval_path")" && pwd -P)/$(basename "$owner_approval_path")" || die 'cannot resolve --owner-approval path'
  owner_approval_path="$(canonical_existing_file "$owner_approval_path")"
  if paths_overlap "$owner_approval_path" "$root_path" || paths_overlap "$owner_approval_path" "$work_dir"; then
    die '--owner-approval must not overlap the checkout or work directory'
  fi
fi
mkdir -p "$(dirname "$work_dir")"
mkdir "$work_dir" || die 'work directory candidate was created concurrently or is not absent'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
(( free_kib >= MIN_DISK_KIB )) || die '32 GiB disk guard failed'
# kaldi-native-fbank is a pinned git source and its uv build invokes CMake;
# fail before any dependency download/build (and therefore before model
# snapshot) if the native toolchain is absent.
for tool in cargo git uv awk find df findmnt sha256sum cmake make cc c++ g++; do
  command -v "$tool" >/dev/null 2>&1 || die "missing native/reference tool: $tool (run scripts/publish/vast-ai/provision.sh as root on Debian/VAST)"
done
[[ "$(findmnt -T "$(dirname "$work_dir")" -no FSTYPE)" == tmpfs ]] || die 'parent work filesystem must be tmpfs'
mkdir -p "$work_dir/model" "$work_dir/source" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR

# Gate every toolchain, lock, source, and license invariant before the model
# snapshot is requested.  A failure here must not spend model bandwidth.
{
  echo 'gate=cargo'
  cargo fmt --manifest-path "$ROOT/Cargo.toml" --all -- --check
  cargo metadata --manifest-path "$ROOT/Cargo.toml" --locked --no-deps --format-version 1 >/dev/null
} > "$work_dir/evidence/validation.log" 2>&1 || die 'rooted Cargo gate failed'
UV_CACHE_DIR="$UV_CACHE_DIR" uv lock --check --project "$FIRERED_PROJECT" --python 3.12 >> "$work_dir/evidence/validation.log" 2>&1 || die 'dedicated FireRed uv.lock is stale or unavailable'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - \
  "$FIRERED_PROJECT/uv.lock" "$REFERENCE" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1 || die 'FireRed dependency/license lock gate failed'
import sys
from pathlib import Path

lock = Path(sys.argv[1]).read_text(encoding="utf-8")
reference = Path(sys.argv[2]).read_text(encoding="utf-8")
for forbidden in ('name = "cuda-', 'name = "nvidia-', 'name = "triton"'):
    if forbidden in lock:
        raise SystemExit(f"forbidden CUDA dependency row: {forbidden}")
for required in (
    'name = "torch"',
    'source = { registry = "https://download.pytorch.org/whl/cpu" }',
    'name = "kaldi-native-fbank"',
    'f68c6b43f739697d7ab02ff6debacee130e1d541',
    'name = "kaldiio"',
    'version = "2.18.1"',
    'name = "setuptools"',
    'version = "83.0.0"',
    'specifier = "==83.0.0"',
):
    if required not in lock:
        raise SystemExit(f"missing locked FireRed dependency identity: {required}")
if 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30' not in reference:
    raise SystemExit("kaldi-native-fbank license hash is not authenticated")
print("FireRed dependency/license lock gate: PASS")
PY
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python - <<'PY' || die 'locked FireRed project is missing the audited exact frontend dependencies: require kaldiio==2.18.1, kaldi-native-fbank==1.15 and setuptools==83.0.0'
import kaldi_native_fbank
import kaldiio
import sentencepiece
import torch
from importlib.metadata import version
assert version("kaldi-native-fbank") == "1.15"
assert version("kaldiio") == "2.18.1"
assert version("setuptools") == "83.0.0"
print("FireRedASR upstream dependency preflight: PASS")
PY
# shellcheck disable=SC2129
git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
git clone --filter=blob:none --no-checkout "$KALDI_NATIVE_FBANK_URL" "$work_dir/source/kaldi-native-fbank" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/kaldi-native-fbank" checkout --detach "$KALDI_NATIVE_FBANK_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
[[ -f "$work_dir/source/kaldi-native-fbank/LICENSE" && -f "$work_dir/source/kaldi-native-fbank/setup.py" ]] || die 'pinned kaldi-native-fbank source/build files are incomplete'
[[ "$(sha256sum "$work_dir/source/kaldi-native-fbank/LICENSE" | awk '{print $1}')" == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" ]] || die 'pinned kaldi-native-fbank LICENSE hash mismatch'
# License/native closure is a separate, gate-first audit. An inventory is
# useful evidence, but it is not approval. In approved mode the owner artifact
# is only passed through for exact validation; the worker never creates it.
# shellcheck disable=SC2317
set +e
audit_args=(
  --lock "$FIRERED_PROJECT/uv.lock"
  --project "$work_dir/source/kaldi-native-fbank"
  --output "$work_dir/evidence/dependency-audit.json"
)
if [[ -n "$owner_approval_path" ]]; then
  audit_args+=(--owner-approval "$owner_approval_path")
fi
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$FIRERED_PROJECT" --python 3.12 python "$AUDITOR" "${audit_args[@]}" >> "$work_dir/evidence/validation.log" 2>&1
audit_rc=$?
set -e
if [[ -z "$owner_approval_path" ]]; then
  (( audit_rc == 2 )) || die 'FireRed dependency closure auditor failed unexpectedly'
else
  (( audit_rc == 0 || audit_rc == 2 )) || die 'FireRed dependency closure auditor failed unexpectedly'
fi
# Treat the emitted audit as an immutable, machine-readable review packet. A
# malformed or self-approved packet is rejected before the model snapshot
# boundary; this worker never supplies an owner approval artifact itself.
approved_mode=0
[[ -n "$owner_approval_path" ]] && approved_mode=1
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity" --python 3.12 python - "$work_dir/evidence/dependency-audit.json" "$FIRERED_PROJECT/uv.lock" "$approved_mode" "$owner_approval_path" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1 || die 'FireRed dependency audit review ledger/digest gate failed'
import hashlib
import json
import sys
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
lock_path = Path(sys.argv[2])
approved_mode = sys.argv[3] == "1"
owner_approval = manifest.get("owner_approval", {})
expected_status = "OWNER_APPROVED" if approved_mode else "BLOCKED_UNREVIEWED_TRANSITIVE"
if manifest.get("status") != expected_status or manifest.get("gate", {}).get("status") != expected_status:
    raise SystemExit(f"dependency audit status mismatch for approved_mode={approved_mode}")
if owner_approval.get("status") != ("VALIDATED" if approved_mode else "MISSING"):
    raise SystemExit("owner approval status mismatch")
expected_protocol = "OWNER_APPROVED" if approved_mode else "BLOCKED_NO_OWNER_APPROVAL"
if manifest.get("collection_protocol", {}).get("status") != expected_protocol:
    raise SystemExit("collection protocol status mismatch")
collection_failures = manifest.get("collection_failures")
if not isinstance(collection_failures, list):
    raise SystemExit("collection failure report is malformed")
if approved_mode:
    approval_path = Path(sys.argv[4])
    artifact = owner_approval.get("artifact", {})
    if artifact.get("path") != str(approval_path) or artifact.get("bytes") != approval_path.stat().st_size or artifact.get("sha256") != hashlib.sha256(approval_path.read_bytes()).hexdigest():
        raise SystemExit("owner approval artifact identity mismatch")
    if collection_failures:
        raise SystemExit("approved audit still has incomplete distribution evidence")
else:
    if owner_approval.get("artifact") is not None:
        raise SystemExit("unapproved packet contains an approval artifact")
closure = manifest.get("active_closure", {})
rows = closure.get("rows")
ledger = manifest.get("review_ledger", {})
ledger_rows = ledger.get("rows")
if not isinstance(rows, list) or len(rows) != 27 or len(rows) != closure.get("row_count"):
    raise SystemExit("active closure row count is malformed")
if not isinstance(ledger_rows, list) or len(ledger_rows) != 27 or ledger.get("row_count") != len(rows):
    raise SystemExit("review ledger must cover exactly 27 active closure rows")
if ledger.get("sha256") != digest(ledger_rows):
    raise SystemExit("review ledger digest mismatch")
if closure.get("row_digest") != digest(rows):
    raise SystemExit("active closure digest mismatch")
by_identity = {(row.get("name"), row.get("version"), row.get("row_sha256")) for row in rows}
seen = set()
for item in ledger_rows:
    identity = (item.get("name"), item.get("version"), item.get("row_sha256"))
    if identity not in by_identity or identity in seen:
        raise SystemExit("review ledger row identity mismatch")
    if item.get("review_status") != "OWNER_REVIEW_REQUIRED" or item.get("owner_decision") is not None:
        raise SystemExit("review ledger contains an implicit owner decision")
    seen.add(identity)
if seen != by_identity:
    raise SystemExit("review ledger row set is incomplete")
distribution_evidence = manifest.get("distribution_evidence")
if not isinstance(distribution_evidence, list) or len(distribution_evidence) != len(rows):
    raise SystemExit("distribution evidence must cover every active closure row")
if manifest.get("distribution_evidence_sha256") != digest(distribution_evidence):
    raise SystemExit("distribution evidence digest mismatch")
for item in distribution_evidence:
    if item.get("installed") is not True or item.get("version_match") is not True or item.get("metadata") is None:
        raise SystemExit(f"incomplete installed distribution evidence: {item.get('name')}")
kaldi_rows = [item for item in distribution_evidence if item.get("name") == "kaldi-native-fbank"]
if len(kaldi_rows) != 1:
    raise SystemExit("kaldi-native-fbank source evidence row is missing or duplicated")
source_license = kaldi_rows[0].get("native_source_license")
if not isinstance(source_license, dict):
    raise SystemExit("kaldi-native-fbank pinned source/LICENSE evidence is missing")
if source_license not in kaldi_rows[0].get("license_candidates", []):
    raise SystemExit("kaldi-native-fbank source/LICENSE is not included in license candidates")
if {
    source_license.get("kind"),
    source_license.get("source_url"),
    source_license.get("source_revision"),
    source_license.get("license_path"),
    source_license.get("license_sha256"),
} != {
    "pinned_source_license",
    "https://github.com/csukuangfj/kaldi-native-fbank.git",
    "f68c6b43f739697d7ab02ff6debacee130e1d541",
    "LICENSE",
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}:
    raise SystemExit("kaldi-native-fbank pinned source/LICENSE identity mismatch")
if not isinstance(source_license.get("license_bytes"), int) or source_license["license_bytes"] <= 0:
    raise SystemExit("kaldi-native-fbank LICENSE byte evidence is missing")
if not all(source_license.get(key) is True for key in ("source_revision_verified", "source_url_verified", "license_sha256_verified")):
    raise SystemExit("kaldi-native-fbank pinned source/LICENSE verification failed")
for key in ("publisher_url_aggregate", "license_candidate_aggregate", "native_payload_aggregate"):
    aggregate = manifest.get(key)
    if not isinstance(aggregate, dict) or aggregate.get("sha256") != digest(aggregate.get("rows")):
        raise SystemExit(f"{key} digest mismatch")
gate = manifest.get("exact_digest_gate", {})
scope = gate.get("scope")
if not isinstance(scope, dict) or gate.get("scope_sha256") != digest(scope):
    raise SystemExit("exact digest gate scope is malformed")
lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
lock_artifact = manifest.get("lock_artifact")
if not isinstance(lock_artifact, dict) or lock_artifact.get("format") != "uv.lock" or lock_artifact.get("sha256") != lock_sha256:
    raise SystemExit("immutable uv.lock artifact identity mismatch")
if scope.get("lock_sha256") != lock_sha256:
    raise SystemExit("exact digest gate lock identity mismatch")
if scope.get("active_closure_sha256") != closure.get("row_digest"):
    raise SystemExit("exact digest gate active closure mismatch")
if scope.get("distribution_evidence_sha256") != manifest.get("distribution_evidence_sha256"):
    raise SystemExit("exact digest gate distribution evidence mismatch")
source_aggregate = manifest.get("source_identity_aggregate")
if not isinstance(source_aggregate, dict) or source_aggregate.get("sha256") != digest(source_aggregate.get("rows")):
    raise SystemExit("source identity aggregate digest mismatch")
if scope.get("source_identity_aggregate_sha256") != source_aggregate.get("sha256"):
    raise SystemExit("exact digest gate source identity mismatch")
if scope.get("review_ledger_sha256") != ledger.get("sha256"):
    raise SystemExit("exact digest gate review ledger mismatch")
if scope.get("license_candidate_aggregate_sha256") != manifest["license_candidate_aggregate"]["sha256"]:
    raise SystemExit("exact digest gate license aggregate mismatch")
if scope.get("native_payload_aggregate_sha256") != manifest["native_payload_aggregate"]["sha256"]:
    raise SystemExit("exact digest gate native aggregate mismatch")
if scope.get("publisher_url_aggregate_sha256") != manifest["publisher_url_aggregate"]["sha256"]:
    raise SystemExit("exact digest gate publisher aggregate mismatch")
if not isinstance(scope.get("lock_sha256"), str) or len(scope["lock_sha256"]) != 64:
    raise SystemExit("exact digest gate lock digest is malformed")
print("FireRed dependency review ledger/exact digest gate: PASS")
PY
if [[ -z "$owner_approval_path" ]]; then
  die 'FireRed dependency closure is BLOCKED_UNREVIEWED_TRANSITIVE; owner review is required before model snapshot'
fi
# shellcheck disable=SC2129
{
  if [[ -n "$owner_approval_path" ]]; then
    echo 'status=OWNER_APPROVED'
    echo 'evidence_stage=OWNER_APPROVED_PRE_MODEL'
  else
    echo 'status=BLOCKED'
    echo 'evidence_stage=INSPECTION_ONLY'
  fi
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'publication=NO_UPLOAD'
} >> "$work_dir/evidence/validation.log" 2>&1

# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python - "$work_dir/server_tree.json" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder
repo, rev = "FireRedTeam/FireRedASR-AED-L", "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
api = HfApi()
info = api.model_info(repo, revision=rev)
if info.sha != rev: raise RuntimeError(f"resolved revision {info.sha!r} != {rev!r}")
rows=[]; pending=[""]; visited=set()
while pending:
    path=pending.pop()
    if path in visited: continue
    visited.add(path)
    for item in api.list_repo_tree(repo, revision=rev, path_in_repo=path, recursive=False):
        if isinstance(item, RepoFolder):
            if getattr(item,"type",None) not in {None,"directory"}: raise RuntimeError(f"invalid RepoFolder type {item!r}")
            item_type="directory"
        elif isinstance(item, RepoFile):
            if getattr(item,"type",None) not in {None,"file"}: raise RuntimeError(f"invalid RepoFile type {item!r}")
            item_type="file"
        else:
            raise RuntimeError(f"unknown HF tree entry type: {type(item).__name__}")
        item_path=getattr(item,"path",None)
        if not isinstance(item_path,str): raise RuntimeError(f"invalid HF entry path {item!r}")
        if item_type=="directory": pending.append(item_path); continue
        lfs=getattr(item,"lfs",None)
        lfs_sha=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None)
        blob=getattr(item,"blob_id",None) or getattr(item,"oid",None)
        size=getattr(item,"size",None)
        if not isinstance(size,int) or isinstance(size,bool) or size<0 or not isinstance(blob,str): raise RuntimeError(f"invalid identity {item_path}")
        rows.append({"path":item_path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha})
if len({x["path"] for x in rows}) != len(rows): raise RuntimeError("duplicate server path")
payload = (json.dumps({"repository":repo,"revision":rev,"resolved_revision":info.sha,"files":sorted(rows,key=lambda x:x["path"])},indent=2,sort_keys=True)+"\n").encode()
target = Path(sys.argv[1])
if target.exists() or target.is_symlink(): raise RuntimeError(f"server tree output exists: {target}")
with __import__("tempfile").NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(payload); stream.flush(); __import__("os").fsync(stream.fileno())
try:
    __import__("os").link(temporary, target)
finally:
    temporary.unlink(missing_ok=True)
PY
# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python - "$REPOSITORY" "$REVISION" "$work_dir/model" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))
PY
set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python "$INSPECTOR" --snapshot "$work_dir/model" --server-tree "$work_dir/server_tree.json" --source "$work_dir/source/repo" --evidence "$work_dir/evidence" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must exit 2, got $inspect_rc"
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'manifest missing'
grep -Fq '"status": "BLOCKED"' "$work_dir/evidence/manifest.json" || die 'blocked status missing'
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$work_dir/evidence/manifest.json" || die 'inspection stage missing'
grep -Fq '"publication": "NO_UPLOAD"' "$work_dir/evidence/manifest.json" || die 'publication status missing'
grep -Fq '"inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE"' "$work_dir/evidence/manifest.json" || die 'inspection did not complete authenticated evidence'
prepared_path="$work_dir/evidence/firered-asr-aed-l.prepared.safetensors"
preparation_manifest="$work_dir/evidence/firered-asr-aed-l.prepared.safetensors.manifest.json"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python "$PREPARER" \
  --ckpt "$work_dir/model/model.pth.tar" \
  --output "$prepared_path" \
  --audit-output "$preparation_manifest" >> "$work_dir/evidence/validation.log" 2>&1
[[ -s "$prepared_path" ]] || die 'prepared safetensors missing'
[[ -s "$preparation_manifest" ]] || die 'preparation manifest missing'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python "$PREPARER" \
  --validate-manifest \
  --inspection-manifest "$work_dir/evidence/manifest.json" \
  --preparation-manifest "$preparation_manifest" \
  --prepared "$prepared_path" >> "$work_dir/evidence/validation.log" 2>&1
combined_manifest="$work_dir/evidence/manifest-with-preparation.json"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python - \
  "$work_dir/evidence/manifest.json" "$preparation_manifest" "$combined_manifest" <<'PY'
import json
import sys
from pathlib import Path
import os, tempfile

def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result

manifest_path, preparation_path, combined_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
preparation = json.loads(preparation_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
preparation["runtime_status_scope"] = "full_pcm_transcription_only; feature-to-feature and feature-to-token primitives are parity-pending"
manifest["preparation"] = preparation
payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
if combined_path.exists() or combined_path.is_symlink(): raise RuntimeError("combined manifest output exists")
with tempfile.NamedTemporaryFile(prefix=f".{combined_path.name}.", suffix=".tmp", dir=combined_path.parent, delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
try:
    os.link(temporary, combined_path)
finally:
    temporary.unlink(missing_ok=True)
PY
reference_path="$work_dir/evidence/upstream_reference.json"
final_manifest="$work_dir/evidence/manifest-with-reference.json"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python "$REFERENCE" \
  --source "$work_dir/source/repo" \
  --checkpoint "$work_dir/model/model.pth.tar" \
  --cmvn "$work_dir/model/cmvn.ark" \
  --output "$reference_path" >> "$work_dir/evidence/validation.log" 2>&1 \
  || die 'independent upstream reference capture failed; inspect validation.log for the exact pinned dependency/source/API blocker'
[[ -s "$reference_path" ]] || die 'upstream reference manifest missing'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python - \
  "$combined_manifest" "$reference_path" "$final_manifest" <<'PY'
import json, os, re, sys, tempfile
from pathlib import Path
def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
manifest_path = sys.argv[1]
reference_path = sys.argv[2]
final_path = sys.argv[3]
manifest = json.loads(open(manifest_path, encoding="utf-8").read(), object_pairs_hook=reject_duplicate_pairs)
reference = json.loads(open(reference_path, encoding="utf-8").read(), object_pairs_hook=reject_duplicate_pairs)
assert manifest["status"] == "BLOCKED"
assert manifest["evidence_stage"] == "INSPECTION_ONLY"
assert manifest["runtime_status"] == "NOT_IMPLEMENTED_FAIL_CLOSED"
assert manifest["runtime_status_scope"] == "full_pcm_transcription_only; feature-to-feature and feature-to-token primitives are parity-pending"
assert manifest["publication"] == "NO_UPLOAD"
assert manifest["inspection_status"] == "AUTHENTICATED_EVIDENCE_COMPLETE"
preparation = manifest.get("preparation")
assert isinstance(preparation, dict)
assert preparation["status"] == "PREPARED"
assert preparation["publication"] == "NO_UPLOAD"
assert preparation["runtime_status"] == "NOT_IMPLEMENTED_FAIL_CLOSED"
assert preparation["runtime_status_scope"] == "full_pcm_transcription_only; feature-to-feature and feature-to-token primitives are parity-pending"
assert preparation["parity_status"] == "NOT_RUN"
assert preparation["future_gate"]["status"] == "BLOCKED_NOT_RUN"
assert preparation["future_gate"]["fp32_atol_status"] == "PREREGISTERED_NOT_RUN"
assert "independent upstream capture" in preparation["future_gate"]["blocker"]
state_audit = preparation["audit"]["state_dict"]
assert state_audit["tensor_count"] > 0
assert state_audit["tensor_count"] == len(state_audit["tensors"])
assert preparation["output"]["bytes"] > 0
contract = manifest.get("source_contract")
assert isinstance(contract, dict)
assert contract.get("status") == "AUTHENTICATED_SOURCE_CONTRACT"
expected_paths = [
    "fireredasr/models/fireredasr_aed.py",
    "fireredasr/data/asr_feat.py",
    "fireredasr/tokenizer/aed_tokenizer.py",
    "README.md",
]
records = contract.get("records")
assert isinstance(records, list) and len(records) == len(expected_paths)
assert [record.get("path") for record in records] == expected_paths
for record in records:
    assert set(record) == {"path", "sha256", "markers", "status"}
    assert record["status"] == "SOURCE_FACTS_AUTHENTICATED"
    assert isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    assert isinstance(record["markers"], list) and record["markers"]
assert "INSPECTION_ERROR" not in json.dumps(manifest)
assert reference["format"] == "vokra-firered-asr-aed-l-upstream-reference-v1"
assert reference["status"] == "REFERENCE_CAPTURED"
assert reference["publication"] == "NO_UPLOAD"
assert reference["model"] == {"repository": "FireRedTeam/FireRedASR-AED-L", "revision": "e57f5960d03cff1071ff7acbb409314d1e70ed3d"}
assert reference["checkpoint"] == {"repository": "FireRedTeam/FireRedASR-AED-L", "revision": "e57f5960d03cff1071ff7acbb409314d1e70ed3d", "bytes": 4678597714, "sha256": "12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3"}
assert reference["source"]["revision"] == "834635e4cf277ed8ca92049fc375b17c3dc20748"
assert reference["dependencies"] == {
    "python": "3.12",
    "kaldiio": {"version": "2.18.1", "source": "pypi", "wheel_sha256": "397a4cd18977acaae7acabfba6807ee0a6978c620064381a266eac15b3c1a0a0"},
    "kaldi-native-fbank": {
        "repository": "https://github.com/csukuangfj/kaldi-native-fbank.git",
        "revision": "f68c6b43f739697d7ab02ff6debacee130e1d541",
        "version": "1.15",
        "license": "Apache-2.0",
        "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
}
source_records = reference["source"]["records"]
assert isinstance(source_records, list) and len(source_records) == 7
assert len({record["path"] for record in source_records}) == len(source_records)
assert {record["path"] for record in source_records} == {
    "fireredasr/data/asr_feat.py",
    "fireredasr/models/fireredasr_aed.py",
    "fireredasr/models/module/conformer_encoder.py",
    "fireredasr/models/module/transformer_decoder.py",
    "fireredasr/tokenizer/aed_tokenizer.py",
    "fireredasr/data/token_dict.py",
    "README.md",
}
for record in source_records:
    assert set(record) == {"path", "role", "sha256", "markers"}
    assert isinstance(record["path"], str) and record["path"]
    assert isinstance(record["role"], str) and record["role"]
    assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    assert isinstance(record["markers"], list) and record["markers"]
    assert all(isinstance(marker, str) and marker for marker in record["markers"])
assert len(reference["tensor_mapping"]) == 940
assert reference["reference"]["status"] == "REFERENCE_CAPTURED"
assert reference["reference"]["encoder"] is not None
assert reference["reference"]["decoder_logits"] is not None
trace = reference["reference"]["trace"]
assert trace["schema"] == "firered-asr-aed-l-reference-trace-v1"
required = trace["required"]
assert required == {
    "frontend_fbank_cmvn": True,
    "encoder_input_preprocessor": True,
    "encoder_each_layer": 16,
    "encoder_final": True,
    "decoder_each_layer": 16,
    "decoder_logits_each_step": True,
    "token_ids": True,
}
encoder_stages = {item["name"]: item["invocations"] for item in trace["encoder_stages"]}
decoder_stages = {item["name"]: item["invocations"] for item in trace["decoder_stages"]}
assert all(encoder_stages.get(f"encoder.layer_stack.{index}") for index in range(16))
assert all(decoder_stages.get(f"decoder.layer_stack.{index}") for index in range(16))
assert decoder_stages.get("decoder_logits")
assert reference["reference"]["frontend"].get("values")
assert reference["parity"]["status"] == "NOT_RUN"
manifest["upstream_reference"] = reference
payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
if final_path.exists() or final_path.is_symlink(): raise RuntimeError("final no-clobber manifest already exists")
if Path(final_path).parent.is_symlink(): raise RuntimeError("final manifest parent is a symlink")
with tempfile.NamedTemporaryFile(prefix=f".{Path(final_path).name}.", suffix=".tmp", dir=Path(final_path).parent, delete=False) as stream:
    temporary = Path(stream.name)
    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
linked = False
try:
    os.link(temporary, final_path)
    linked = True
finally:
    try:
        temporary.unlink()
    except OSError:
        if linked and Path(final_path).is_file() and os.stat(final_path).st_ino == os.stat(temporary).st_ino:
            Path(final_path).unlink()
        raise
PY
die 'FireRedASR inspection, preparation and independent upstream reference evidence preserved as final no-clobber manifest; native conversion/runtime/parity remain blocked'
