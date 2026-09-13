#!/usr/bin/env bash
# shellcheck disable=SC2317
# VAST-only FireRedASR-AED-L inspection and safe checkpoint preparation.
# No runtime execution, parity claim, or publication.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# The pinned source/release contract and VAST checkpoint identity are recorded;
# native conversion/runtime seams are source-implemented and authenticated, but
# real CPU parity and the complete transcription route remain fail-closed.
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
MODEL_FREE_MIN_MEM_KIB=$((4 * 1024 * 1024))
MODEL_FREE_MIN_DISK_KIB=$((4 * 1024 * 1024))
UV_CACHE_DIR="${FIRERED_ASR_UV_CACHE_DIR:-/tmp/vokra-firered-asr-uv-cache}"
APPROVAL_SCHEMA="vokra-firered-asr-aed-l-blocked-approval-v1"
MODEL_FREE_FORMAT="vokra-firered-asr-aed-l-model-free-audit-v1"
APPROVAL_SCOPE_JSON='{"cmvn_status":"AUTHENTICATED_CMVN_TXT_BINDING_PARITY_PENDING","config_status":"BLOCKED_EMPTY_CONFIG","dependency_status":"BLOCKED_UNREVIEWED_TRANSITIVE","kaldi_native_fbank_revision":"f68c6b43f739697d7ab02ff6debacee130e1d541","kaldi_native_fbank_url":"https://github.com/csukuangfj/kaldi-native-fbank.git","license_status":"BLOCKED_TRAINING_AND_DEPENDENCY_PROVENANCE","model_repository":"FireRedTeam/FireRedASR-AED-L","model_revision":"e57f5960d03cff1071ff7acbb409314d1e70ed3d","native_status":"SOURCE_IMPLEMENTED_PARITY_PENDING","source_revision":"834635e4cf277ed8ca92049fc375b17c3dc20748","source_status":"AUTHENTICATED_SOURCE_CONTRACT","source_url":"https://github.com/FireRedTeam/FireRedASR.git","tokenizer_status":"AUTHENTICATED_OUTPUT_DICTIONARY_BINDING"}'

log() { printf '[firered-asr-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() {
  cat <<'EOF'
usage: run-firered-asr-aed-l-inspection.sh --model-free --expected-head HEX40 [--work-dir DIR]
       run-firered-asr-aed-l-inspection.sh --dependency-audit-only --expected-head HEX40 [--work-dir DIR]
       run-firered-asr-aed-l-inspection.sh --expected-head HEX40 --approval-sha256 SHA256 --owner-approval JSON [--work-dir DIR]
       run-firered-asr-aed-l-inspection.sh --self-test

--model-free collects only the frozen dependency graph and fixed source,
config, model-card, and training-provenance scope. It never downloads or
executes a model, accesses an upstream checkout, or uploads anything. The
normal owner-approval route retains its existing gate-first behavior.
--dependency-audit-only prepares the frozen Python closure and collects only
the pinned FireRed source, kaldi-native-fbank source/LICENSE, and installed
publisher/license/native-payload evidence. It never contacts the model repo,
acquires a checkpoint, imports or executes a model, runs reference/parity, or
uploads anything; it always exits blocked pending owner review.
EOF
}

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die 'expected HEAD must be exactly 40 lowercase hexadecimal characters'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean before FireRed acquisition'
  actual="$(git -C "$ROOT" rev-parse --verify HEAD)"
  [[ "$actual" == "$expected" ]] || die "HEAD mismatch: expected $expected, observed $actual"
}
require_approval_path() {
  local path="$1" parent
  [[ "$path" == /* && "$path" != *'/../'* && "$path" != */.. && "$path" != *'/./'* && "$path" != *'/.' ]] || return 1
  [[ "$path" != "$ROOT" && "$path" != "$ROOT"/* ]] || return 1
  [[ -f "$path" && ! -L "$path" ]] || return 1
  parent="$path"
  while [[ -n "$parent" ]]; do
    parent="$(dirname "$parent")"
    [[ ! -L "$parent" ]] || return 1
    [[ "$parent" == / ]] && break
  done
}
require_approval_binding() {
  local path="$1" expected_sha="$2"
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || die 'approval SHA-256 must be exactly 64 lowercase hexadecimal characters'
  require_approval_path "$path" || die 'approval must be an external regular non-symlink JSON file with safe ancestry'
  [[ "$(sha256_file "$path")" == "$expected_sha" ]] || die 'approval SHA-256 differs from caller binding'
}
require_blocked_approval() {
  local path="$1" expected_head="$2"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$APPROVAL_SCOPE_JSON" "$APPROVAL_SCHEMA" <<'PY'
import hashlib, json, re, sys
path, expected_head, scope, schema = sys.argv[1:]
def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out
with open(path, encoding="utf-8") as handle:
    value = json.load(handle, object_pairs_hook=pairs)
keys = {"schema", "decision", "status", "evidence_stage", "no_upload", "expected_head", "model_repository", "model_revision", "source_url", "source_revision", "kaldi_native_fbank_url", "kaldi_native_fbank_revision", "cmvn_status", "config_status", "dependency_status", "license_status", "native_status", "source_status", "tokenizer_status", "scope_sha256"}
if set(value) != keys:
    raise ValueError("approval key set mismatch")
scope_value = json.loads(scope, object_pairs_hook=pairs)
canonical = json.dumps(scope_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
expected = {"schema": schema, "decision": "BLOCKED", "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "no_upload": True, "expected_head": expected_head, **scope_value, "scope_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
if value != expected or not re.fullmatch(r"[0-9a-f]{40}", value["expected_head"]):
    raise ValueError("approval identity or blocked disposition mismatch")
print("FIRERED_APPROVAL_VALID_BUT_BLOCKED: BLOCKED/INSPECTION_ONLY/NO_UPLOAD")
PY
  then return 0; else return 1; fi
}

canonical_absent_candidate() {
  local candidate="$1" parent suffix resolved ancestor
  [[ "$candidate" == /* ]] || die 'path candidate must be absolute'
  [[ "$candidate" != *'/../'* && "$candidate" != */.. && "$candidate" != *'/./'* && "$candidate" != */. ]] \
    || die 'path candidate must not contain lexical dot components'
  [[ ! -e "$candidate" && ! -L "$candidate" ]] || die "path candidate must be absent: $candidate"
  parent="$(dirname "$candidate")"
  suffix="$(basename "$candidate")"
  while [[ ! -e "$parent" ]]; do
    [[ "$parent" != / ]] || die "cannot resolve path parent: $candidate"
    suffix="$(basename "$parent")/$suffix"
    parent="$(dirname "$parent")"
  done
  ancestor="$parent"
  while [[ "$ancestor" != / ]]; do
    [[ ! -L "$ancestor" ]] || die "path parent has a symlink ancestor: $ancestor"
    ancestor="$(dirname "$ancestor")"
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

model_free_disk_root() {
  local candidate="$1" parent ancestor
  parent="$(dirname "$candidate")"
  while [[ ! -e "$parent" && ! -L "$parent" ]]; do
    [[ "$parent" != / ]] || die "cannot resolve disk parent: $candidate"
    parent="$(dirname "$parent")"
  done
  ancestor="$parent"
  while [[ "$ancestor" != / ]]; do
    [[ ! -L "$ancestor" ]] || die "disk parent has a symlink ancestor: $ancestor"
    ancestor="$(dirname "$ancestor")"
  done
  [[ -d "$parent" && ! -L "$parent" ]] || die "disk parent must be a non-symlink directory: $parent"
  (cd -P "$parent" && pwd -P)
}

require_model_free_host() {
  local work_dir="$1" mem_kib free_kib disk_root
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
  [[ "$(uname -s)" == Linux ]] || die 'Linux VAST required for model-free audit'
  [[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST required for model-free audit'
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
  (( mem_kib >= MODEL_FREE_MIN_MEM_KIB )) || die '4 GiB model-free memory guard failed'
  disk_root="$(model_free_disk_root "$work_dir")"
  free_kib="$(df -Pk "$disk_root" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
  (( free_kib >= MODEL_FREE_MIN_DISK_KIB )) || die '4 GiB model-free disk guard failed'
}

run_model_free() {
  local expected="$1" requested_work_dir="$2" work_dir audit_output audit_rc root_path
  require_clean_expected_head "$expected"
  [[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
  [[ -f "$FIRERED_PROJECT/pyproject.toml" && -f "$FIRERED_PROJECT/uv.lock" ]] || die 'dedicated FireRed uv project missing'
  [[ -f "$AUDITOR" && ! -L "$AUDITOR" ]] || die 'dedicated FireRed model-free auditor missing'
  command -v uv >/dev/null 2>&1 || die 'uv is required for the model-free audit'
  command -v git >/dev/null 2>&1 || die 'git is required for the model-free audit'
  work_dir="${requested_work_dir:-$WORK/model-free}"
  [[ "$work_dir" == /* ]] || die '--work-dir must be absolute'
  work_dir="$(canonical_absent_candidate "$work_dir")"
  root_path="$(cd "$ROOT" && pwd -P)" || die 'cannot canonicalize checkout root'
  paths_overlap "$work_dir" "$root_path" && die '--work-dir must not overlap the checkout'
  require_model_free_host "$work_dir"
  mkdir -p "$(dirname "$work_dir")"
  mkdir "$work_dir" || die 'model-free work directory candidate was created concurrently'
  mkdir "$work_dir/evidence"
  audit_output="$work_dir/evidence/model-free-audit.json"
  set +e
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --offline --project "$FIRERED_PROJECT" --python 3.12 python "$AUDITOR" \
    --model-free --expected-head "$expected" --lock "$FIRERED_PROJECT/uv.lock" --output "$audit_output" >"$work_dir/evidence/model-free-audit.log" 2>&1
  audit_rc=$?
  set -e
  [[ "$audit_rc" == 2 && -s "$audit_output" ]] || die "model-free audit returned unexpected status: $audit_rc"
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --offline --project "$ROOT/tools/parity" --python 3.12 python - "$audit_output" "$expected" "$MODEL_FREE_FORMAT" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("format") != sys.argv[3]:
    raise SystemExit("model-free audit format mismatch")
if manifest.get("expected_head") != sys.argv[2] or manifest.get("approval_scope", {}).get("scope", {}).get("expected_head") != sys.argv[2]:
    raise SystemExit("model-free audit expected HEAD binding mismatch")
if manifest.get("status") != "BLOCKED_OWNER_REVIEW" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("model-free audit did not remain blocked/no-upload")
if manifest.get("payload_status") != "NOT_ACQUIRED" or manifest.get("execution_status") != "NOT_PERFORMED":
    raise SystemExit("model-free audit crossed the payload/execution boundary")
if manifest.get("model", {}).get("repository") != "FireRedTeam/FireRedASR-AED-L":
    raise SystemExit("model-free model identity mismatch")
if manifest.get("active_closure", {}).get("row_count") != 27:
    raise SystemExit("model-free active dependency closure count mismatch")
scope = manifest.get("approval_scope", {})
if scope.get("status") != "PENDING_OWNER_REVIEW" or not isinstance(scope.get("scope_sha256"), str):
    raise SystemExit("model-free pending approval scope is malformed")
print(f"FireRed model-free audit: BLOCKED_OWNER_REVIEW expected_head={sys.argv[2]}")
PY
  log "model-free evidence: $audit_output"
  return 2
}

require_no_hf_tokens() {
  local token_name
  for token_name in HF HF_TOKEN HF_HUB_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_ACCESS_TOKEN HUGGINGFACE_TOKEN HF_API_TOKEN HUGGINGFACE_API_TOKEN HUGGING_FACE_TOKEN; do
    [[ -z "${!token_name:-}" ]] || die "${token_name} must be unset for dependency-audit-only"
  done
}

require_dependency_audit_host() {
  local work_dir="$1" mem_kib free_kib disk_root
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
  [[ "$(uname -s)" == Linux ]] || die 'Linux VAST required for dependency-audit-only'
  [[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST required for dependency-audit-only'
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
  (( mem_kib >= MIN_MEM_KIB )) || die '128 GiB dependency-audit memory guard failed'
  disk_root="$(model_free_disk_root "$work_dir")"
  free_kib="$(df -Pk "$disk_root" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
  (( free_kib >= MIN_DISK_KIB )) || die '32 GiB dependency-audit disk guard failed'
}

run_dependency_audit_only() {
  local expected="$1" requested_work_dir="$2" work_dir audit_output audit_rc root_path
  require_clean_expected_head "$expected"
  [[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
  [[ -f "$FIRERED_PROJECT/pyproject.toml" && -f "$FIRERED_PROJECT/uv.lock" ]] || die 'dedicated FireRed uv project missing'
  [[ -f "$AUDITOR" && ! -L "$AUDITOR" ]] || die 'dedicated FireRed dependency auditor missing'
  command -v uv >/dev/null 2>&1 || die 'uv is required for the dependency audit'
  command -v git >/dev/null 2>&1 || die 'git is required for the dependency audit'
  require_no_hf_tokens
  work_dir="${requested_work_dir:-$WORK/dependency-audit-only}"
  [[ "$work_dir" == /* ]] || die '--work-dir must be absolute'
  work_dir="$(canonical_absent_candidate "$work_dir")"
  root_path="$(cd "$ROOT" && pwd -P)" || die 'cannot canonicalize checkout root'
  paths_overlap "$work_dir" "$root_path" && die '--work-dir must not overlap the checkout'
  require_dependency_audit_host "$work_dir"
  mkdir -p "$(dirname "$work_dir")"
  mkdir "$work_dir" || die 'dependency-audit-only work directory candidate was created concurrently'
  mkdir "$work_dir/evidence"
  audit_output="$work_dir/evidence/dependency-audit-only.json"
  export UV_PROJECT_ENVIRONMENT="$work_dir/venv"
  export CARGO_BUILD_JOBS=1
  export UV_CACHE_DIR
  {
    echo "mode=DEPENDENCY_AUDIT_ONLY"
    echo "expected_head=$expected"
    echo 'payload_status=NOT_ACQUIRED'
    echo 'execution_status=NOT_PERFORMED'
    echo 'publication=NO_UPLOAD'
    echo 'hf_token_environment=ABSENT_REQUIRED'
    echo 'gate=pre-model-boundary'
    cargo fmt --manifest-path "$ROOT/Cargo.toml" --all -- --check
    cargo metadata --manifest-path "$ROOT/Cargo.toml" --locked --no-deps --format-version 1 >/dev/null
    UV_CACHE_DIR="$UV_CACHE_DIR" uv lock --check --project "$FIRERED_PROJECT" --python 3.12
  } > "$work_dir/evidence/validation.log" 2>&1 || die 'dependency-audit-only rooted/lock gate failed'
  for tool in cargo git uv awk find df findmnt sha256sum cmake make cc c++ g++; do
    command -v "$tool" >/dev/null 2>&1 || die "missing dependency-audit tool: $tool"
  done
  [[ "$(findmnt -T "$(dirname "$work_dir")" -no FSTYPE)" == tmpfs ]] || die 'parent work filesystem must be tmpfs'
  mkdir -p "$work_dir/source"
  {
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION"
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$KALDI_NATIVE_FBANK_URL" "$work_dir/source/kaldi-native-fbank"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$work_dir/source/kaldi-native-fbank" checkout --detach "$KALDI_NATIVE_FBANK_REVISION"
  } >> "$work_dir/evidence/validation.log" 2>&1
  [[ -f "$work_dir/source/kaldi-native-fbank/LICENSE" && ! -L "$work_dir/source/kaldi-native-fbank/LICENSE" ]] || die 'pinned kaldi-native-fbank LICENSE is missing'
  [[ "$(sha256_file "$work_dir/source/kaldi-native-fbank/LICENSE")" == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30" ]] || die 'pinned kaldi-native-fbank LICENSE hash mismatch'
  # Install only the dedicated frozen dependency closure into the absent work
  # directory.  No model package/repository is an input to this operation.
  UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv sync --frozen --project "$FIRERED_PROJECT" --python 3.12 >> "$work_dir/evidence/validation.log" 2>&1 || die 'dedicated frozen dependency preparation failed'
  set +e
  UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run --frozen --no-sync --project "$FIRERED_PROJECT" --python 3.12 python "$AUDITOR" \
    --dependency-audit-only \
    --lock "$FIRERED_PROJECT/uv.lock" \
    --project "$work_dir/source/kaldi-native-fbank" \
    --source "$work_dir/source/repo" \
    --expected-head "$expected" \
    --output "$audit_output" >> "$work_dir/evidence/validation.log" 2>&1
  audit_rc=$?
  set -e
  [[ "$audit_rc" == 2 && -s "$audit_output" ]] || die "dependency-audit-only returned unexpected status: $audit_rc"
  UV_CACHE_DIR="$UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run --frozen --no-sync --project "$FIRERED_PROJECT" --python 3.12 python - "$audit_output" "$expected" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1 || die 'dependency-audit-only manifest contract failed'
import json
import re
import sys
from pathlib import Path

def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
expected = sys.argv[2]
if manifest.get("format") != "vokra-firered-asr-aed-l-dependency-audit-only-v1":
    raise SystemExit("dependency-audit-only format mismatch")
if manifest.get("expected_head") != expected:
    raise SystemExit("dependency-audit-only expected HEAD mismatch")
if manifest.get("status") != "BLOCKED_UNREVIEWED_TRANSITIVE" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("dependency-audit-only did not remain blocked/no-upload")
for key in ("payload_status", "checkpoint_status", "model_repo_status"):
    if manifest.get(key) != "NOT_ACQUIRED":
        raise SystemExit(f"{key} crossed the acquisition boundary")
for key in ("model_import_status", "execution_status", "reference_status", "conversion_status"):
    if manifest.get(key) != "NOT_PERFORMED":
        raise SystemExit(f"{key} crossed the execution boundary")
if manifest.get("owner_approval_artifact_created") is not False:
    raise SystemExit("dependency-audit-only created an owner approval artifact")
acquisition = manifest.get("model_acquisition")
if acquisition != {"hf_api": "NOT_CONTACTED", "snapshot_download": "NOT_CALLED", "checkpoint": "NOT_ACQUIRED"}:
    raise SystemExit("dependency-audit-only touched the model repository boundary")
token_environment = manifest.get("token_environment")
if token_environment.get("status") != "ABSENT" or token_environment.get("present_names") != [] or token_environment.get("values_recorded") is not False:
    raise SystemExit("HF token environment was not absent and unrecorded")
source = manifest.get("fire_red_source")
if source.get("status") != "AUTHENTICATED_PINNED_SOURCE" or source.get("observed_revision") != "834635e4cf277ed8ca92049fc375b17c3dc20748" or not source.get("revision_verified") or not source.get("origin_verified") or not source.get("clean_verified"):
    raise SystemExit("pinned FireRed source identity is incomplete")
aggregate = source.get("tracked_files", {}).get("sha256")
if not isinstance(aggregate, str) or not re.fullmatch(r"[0-9a-f]{64}", aggregate):
    raise SystemExit("pinned FireRed source hash aggregate is malformed")
if not isinstance(manifest.get("active_closure", {}).get("rows"), list) or len(manifest["active_closure"]["rows"]) != 27:
    raise SystemExit("active closure does not cover exactly 27 rows")
if manifest.get("collection_status") != "COMPLETE" or manifest.get("collection_failures"):
    raise SystemExit("dependency evidence collection failed; packet is explicitly blocked")
scope = manifest.get("dependency_audit_digest_gate", {})
if not isinstance(scope.get("scope_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", scope["scope_sha256"]):
    raise SystemExit("dependency audit digest scope is malformed")
print("FireRed dependency-audit-only manifest: BLOCKED_UNREVIEWED_TRANSITIVE/NO_UPLOAD; collection complete")
PY
  log "dependency-audit-only evidence: $audit_output"
  log 'dependency-audit-only remains BLOCKED_UNREVIEWED_TRANSITIVE pending owner review; no model boundary was entered'
  return 2
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token path_test candidate tmp_parent
  tmp_parent="$(cd -P "${TMPDIR:-/tmp}" 2>/dev/null && pwd -P)" || { log 'self-test FAIL: temp parent is not canonical'; return 1; }
  path_test="$(mktemp -d "$tmp_parent/firered-path-selftest.XXXXXX")" || { log 'self-test FAIL: mktemp failed'; return 1; }
  candidate="$(canonical_absent_candidate "$path_test/nested/missing")" || fail=1
  path_test="$(cd "$path_test" && pwd -P)" || fail=1
  [[ "$candidate" == "$path_test/nested/missing" ]] || fail=1
  [[ "$(model_free_disk_root "$path_test/nested/missing")" == "$path_test" ]] || fail=1
  mkdir "$path_test/existing"
  if (canonical_absent_candidate "$path_test/existing") >/dev/null 2>&1; then fail=1; fi
  if (canonical_absent_candidate "$path_test/./dot") >/dev/null 2>&1; then fail=1; fi
  if (canonical_absent_candidate "$path_test/../dot") >/dev/null 2>&1; then fail=1; fi
  ln -s missing "$path_test/dangling"
  if (canonical_absent_candidate "$path_test/dangling") >/dev/null 2>&1; then fail=1; fi
  mkdir -p "$path_test/real/child"
  ln -s "$path_test/real" "$path_test/real-link"
  if (canonical_absent_candidate "$path_test/real-link/child/missing") >/dev/null 2>&1; then fail=1; fi
  paths_overlap "$path_test" "$path_test/nested" || fail=1
  paths_overlap "$path_test/nested" "$path_test" || fail=1
  if paths_overlap "$path_test" "/tmp/another-root"; then fail=1; fi
  local approval approval_head scope_sha approval_sha
  approval="$path_test/approval.json"; approval_head="$(printf '0%.0s' {1..40})"; scope_sha="$(printf '%s\n' "$APPROVAL_SCOPE_JSON" | sha256sum | awk '{print $1}')"
  printf '{"schema":"%s","decision":"BLOCKED","status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","no_upload":true,"expected_head":"%s",%s,"scope_sha256":"%s"}\n' "$APPROVAL_SCHEMA" "$approval_head" "$(printf '%s' "$APPROVAL_SCOPE_JSON" | sed 's/^{//; s/}$//')" "$scope_sha" >"$approval"
  approval_sha="$(sha256_file "$approval")"; require_approval_binding "$approval" "$approval_sha" || fail=1
  require_blocked_approval "$approval" "$approval_head" >/dev/null || fail=1
  cp "$approval" "$path_test/wrong-head.json"; sed -i.bak 's/"expected_head":"[0-9a-f][0-9a-f]*/"expected_head":"1111111111111111111111111111111111111111/' "$path_test/wrong-head.json"
  if require_blocked_approval "$path_test/wrong-head.json" "$approval_head" >/dev/null 2>&1; then fail=1; fi
  cp "$approval" "$path_test/wrong-scope.json"; sed -i.bak 's/"scope_sha256":"[0-9a-f][0-9a-f]*/"scope_sha256":"0000000000000000000000000000000000000000000000000000000000000000/' "$path_test/wrong-scope.json"
  if require_blocked_approval "$path_test/wrong-scope.json" "$approval_head" >/dev/null 2>&1; then fail=1; fi
  printf '%s\n' '{"schema":"x","schema":"y"}' >"$path_test/duplicate.json"
  if require_blocked_approval "$path_test/duplicate.json" "$approval_head" >/dev/null 2>&1; then fail=1; fi
  rm -rf "$path_test"
  (( fail == 0 )) || { log 'self-test FAIL: path candidate/overlap contract'; return 1; }
  for token in \
    'FireRedTeam/FireRedASR-AED-L' 'e57f5960d03cff1071ff7acbb409314d1e70ed3d' \
    'FireRedASR.git' '834635e4cf277ed8ca92049fc375b17c3dc20748' \
    'model.pth.tar' '12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3' \
    'train_bpe1000.model' '473bbc157cb4eade2059b30a3c877a1c29bd50cadbfbed869ae36eeade7fee07' \
    'model_info' 'list_repo_tree' 'path_in_repo' 'git_blob_sha1' 'lfs_sha256' 'weights_only=True' \
    '128' '32' '/dev/shm' 'findmnt' 'CARGO_BUILD_JOBS=1' 'status": "BLOCKED"' 'INSPECTION_ONLY' 'NO_UPLOAD' 'LOUD_PARTIAL_FAIL_CLOSED' 'PARTIAL' 'runtime_status_scope' 'full_pcm_transcription_only' \
    'config.yaml' 'BLOCKER_EMPTY_CONFIG' 'git ls-files' 'git status' \
    'source_contract' 'AUTHENTICATED_SOURCE_CONTRACT' 'SOURCE_FACTS_AUTHENTICATED' 'unlock_requirements' 'vast_first_pass' 'expected_artifacts' \
    '--model-free' 'MODEL_FREE_FORMAT' 'build_model_free_manifest' 'MODEL_FREE_MIN_MEM_KIB' 'MODEL_FREE_MIN_DISK_KIB' 'model_free_disk_root' 'run_model_free' 'NOT_ACQUIRED' 'NOT_PERFORMED' 'PENDING_OWNER_REVIEW' 'model_card_architecture' 'model_card_search' 'training_provenance_status' 'expected_head' \
    '--dependency-audit-only' 'DEPENDENCY_AUDIT_ONLY_FORMAT' 'build_dependency_audit_only_manifest' 'fire_red_source_evidence' 'HF' 'HF_TOKEN' 'HF_HUB_TOKEN' 'HUGGING_FACE_HUB_TOKEN' 'HUGGINGFACE_HUB_TOKEN' 'HF_ACCESS_TOKEN' 'HUGGINGFACE_TOKEN' 'HF_API_TOKEN' 'HUGGINGFACE_API_TOKEN' 'HUGGING_FACE_TOKEN' 'require_no_hf_tokens' 'require_dependency_audit_host' 'run_dependency_audit_only' 'UV_PROJECT_ENVIRONMENT' 'uv sync --frozen' 'GIT_LFS_SKIP_SMUDGE=1' '120000' 'readlink' 'symlink' 'git submodule' 'submodule update' 'dependency-audit-only.json' 'DEPENDENCY_AUDIT_ONLY' 'BLOCKED_COLLECTION_FAILURE' 'AUTHENTICATED_PINNED_SOURCE' 'model_repo_status' 'checkpoint_status' 'reference_status' 'conversion_status' \
    'pinned-source frontend' 'SentencePiece/TokenDict' 'transformer_decoder.py' 'batch_beam_search' 'softmax_smoothing' 'length_penalty' 'eos_penalty' 'PREPARED' 'archive_members' \
    'tensor_count' 'publication' '--audit-output' 'BLOCKED_NOT_RUN' 'fp32_atol_status' \
    'firered_asr_aed_l_reference.py' 'tensor_mapping' 'REFERENCE_CAPTURED' 'decoder_logits' 'tgt_word_prj' 'source_records' 'firered-asr-aed-l-reference-trace-v1' 'encoder_each_layer' 'decoder_each_layer' 'frontend_fbank_cmvn' 'official_hypotheses' 'normalized_log_score' 'upstream_cost' 'firered-asr-aed-l-official-beam-trace-v1' 'token_topk' 'beam_prune_topk' 'torch.topk' 'torch_git_version' 'environment' \
    'firered_asr_aed_l_audit.py' 'BLOCKED_UNREVIEWED_TRANSITIVE' 'OWNER_APPROVED' 'OWNER_REVIEW_REQUIRED' 'INVALID' 'distribution_evidence' 'distribution_evidence_sha256' 'lock_artifact' 'source_identity_aggregate' 'native_payloads' 'publisher_urls' 'publisher_url_aggregate' 'license_candidate_aggregate' 'native_payload_aggregate' 'review_ledger' 'exact_digest_gate' 'collection_protocol' 'owner_approval' 'owner-approval-v1' '--owner-approval' 'owner_approval_path' 'yousan' 'approved_at_utc' 'publisher_urls_sha256' 'license_candidates_sha256' 'native_payloads_sha256' 'scope_sha256' 'collection_failures' 'approved_mode' 'is_symlink' 'regular JSON' 'must not overlap' 'reject_duplicate_pairs' 'duplicate JSON key' 'exactly 27 active closure rows' 'native_source_license' 'source_revision_verified' 'source_url_verified' 'license_path' 'license_bytes' 'license_sha256_verified' 'approved_route_expected_artifacts' \
    'NamedTemporaryFile' 'os.link' 'manifest-with-preparation.json' \
    'manifest-with-reference.json' 'final no-clobber manifest' \
    'kaldiio==2.18.1' 'kaldi-native-fbank==1.15' 'name = "setuptools"' 'version = "83.0.0"' 'specifier = "==83.0.0"' 'importlib.metadata' 'setuptools>=83' '397a4cd18977acaae7acabfba6807ee0a6978c620064381a266eac15b3c1a0a0' \
    'f68c6b43f739697d7ab02ff6debacee130e1d541' 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30' \
    'uv lock --check' 'source/kaldi-native-fbank' 'setup.py' 'cmake' 'make' 'cc' 'c++' 'g++' 'native build toolchain' \
    'forbidden CUDA dependency row' 'download.pytorch.org/whl/cpu' 'license hash is not authenticated' \
    'sidecar_binding' 'AUTHENTICATED_EXTERNAL_SIDECARS_BOUND' 'AUTHENTICATED_EXTERNAL_SIDECAR_BOUND' 'converter_and_executable_runtime' \
    'vokra.firered_asr_aed_l.cmvn_txt' 'vokra.firered_asr_aed_l.dict_txt' 'structural_marker_policy' 'forbidden_decoder_ids' 'unknown_policy' 'render_dictionary_token' 'strip_only_if_terminal' 'ordinary_content_id_range' \
    '--no-sync' 'FIRERED_PROJECT' 'firered_asr_aed_l/pyproject.toml' 'firered_asr_aed_l/uv.lock' \
    '--expected-head' '--approval-sha256' 'FIRERED_APPROVAL_VALID_BUT_BLOCKED' 'AUTHENTICATED_CMVN_TXT_BINDING_PARITY_PENDING' 'SOURCE_IMPLEMENTED_PARITY_PENDING' 'AUTHENTICATED_OUTPUT_DICTIONARY_BINDING' 'BLOCKED_EMPTY_CONFIG' 'BLOCKED_UNREVIEWED_TRANSITIVE' 'BLOCKED_TRAINING_AND_DEPENDENCY_PROVENANCE' 'AUTHENTICATED_SOURCE_CONTRACT' \
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
model_free = source.index('\nrun_model_free()')
model_free_audit = source.index('\n    --model-free --expected-head')
if model_free_audit <= model_free or model_free_audit >= snapshot:
    raise SystemExit("model-free audit is not isolated before model snapshot")
approval = source.index('\nrequire_blocked_approval "$owner_approval_path"')
host = source.index('\n[[ "$(uname -s)" == Linux ]]')
work = source.index('\nmkdir "$work_dir" || die')
if not approval < host < work:
    raise SystemExit("blocked approval must precede host/work gates")
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
audit_only = source.index("run_dependency_audit_only()")
audit_only_end = source.index("\nself_test()", audit_only)
snapshot = source.index("\nfrom huggingface_hub import snapshot_download")
if not audit_only < audit_only_end < snapshot:
    raise SystemExit("dependency-audit-only route range is not before model API import")
audit_only_source = source[audit_only:audit_only_end]
for forbidden in ("\nfrom huggingface_hub import", "\nimport huggingface_hub", "HfApi(", "snapshot_download("):
    if forbidden in audit_only_source:
        raise SystemExit(f"dependency-audit-only route reaches model API: {forbidden}")
for forbidden in ("git submodule", "submodule update", "submodule init", "submodule fetch"):
    if forbidden in audit_only_source:
        raise SystemExit(f"dependency-audit-only route initializes/fetches a submodule: {forbidden}")
if source.index("uv sync --frozen", audit_only, audit_only_end) >= audit_only_end:
    raise SystemExit("dependency-audit-only does not prepare the dedicated frozen environment")
if source.index('GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$SOURCE_URL"', audit_only, audit_only_end) >= audit_only_end:
    raise SystemExit("dependency-audit-only does not collect pinned FireRed source")
if source.index("require_no_hf_tokens", audit_only, audit_only_end) >= audit_only_end:
    raise SystemExit("dependency-audit-only token guard is after model boundary")
print("FireRed dependency-audit-only ordering self-test: PASS")
PY
  then
    log 'self-test FAIL: dependency-audit-only/model snapshot ordering regression'
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
  local test_head
  test_head="$(printf '0%.0s' {1..40})"
  if "$path" --model-free >/dev/null 2>&1; then
    log 'self-test FAIL: model-free route accepted without expected HEAD'; fail=1
  fi
  if "$path" --model-free --expected-head 0 >/dev/null 2>&1; then
    log 'self-test FAIL: model-free route accepted malformed expected HEAD'; fail=1
  fi
  if "$path" --model-free --expected-head "$(printf 'g%.0s' {1..40})" >/dev/null 2>&1; then
    log 'self-test FAIL: model-free route accepted non-hex expected HEAD'; fail=1
  fi
  if "$path" --model-free --model-free --expected-head "$test_head" >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate model-free flag accepted'; fail=1
  fi
  if "$path" --self-test --model-free >/dev/null 2>&1; then
    log 'self-test FAIL: self-test/model-free mix accepted'; fail=1
  fi
  if "$path" --model-free --expected-head "$test_head" --approval-sha256 "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    log 'self-test FAIL: model-free approval argument mix accepted'; fail=1
  fi
  if "$path" --dependency-audit-only >/dev/null 2>&1; then
    log 'self-test FAIL: dependency-audit-only route accepted without expected HEAD'; fail=1
  fi
  if "$path" --dependency-audit-only --expected-head "$test_head" --dependency-audit-only >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate dependency-audit-only flag accepted'; fail=1
  fi
  if "$path" --dependency-audit-only --expected-head "$test_head" --model-free >/dev/null 2>&1; then
    log 'self-test FAIL: dependency-audit-only/model-free mix accepted'; fail=1
  fi
  if "$path" --self-test --dependency-audit-only >/dev/null 2>&1; then
    log 'self-test FAIL: self-test/dependency-audit-only mix accepted'; fail=1
  fi
  if "$path" --dependency-audit-only --expected-head "$test_head" --owner-approval /tmp/approval.json >/dev/null 2>&1; then
    log 'self-test FAIL: dependency-audit-only owner approval accepted'; fail=1
  fi
  if UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --no-sync --project "$ROOT/tools/parity/firered_asr_aed_l" --python 3.12 python "$AUDITOR" \
    --lock "$path" --output "$path" --source "$path" >/dev/null 2>&1; then
    log 'self-test FAIL: normal dependency audit accepted --source'; fail=1
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
approval_sha256=""
expected_head=""
self=0
model_free=0
dependency_audit_only=0
seen_self=0
seen_approval=0
seen_sha=0
seen_head=0
seen_model_free=0
seen_dependency_audit_only=0
seen_work=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --model-free) (( seen_model_free == 0 )) || die 'duplicate --model-free'; seen_model_free=1; model_free=1; shift ;;
    --dependency-audit-only) (( seen_dependency_audit_only == 0 )) || die 'duplicate --dependency-audit-only'; seen_dependency_audit_only=1; dependency_audit_only=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; work_dir="$2"; seen_work=1; shift 2 ;;
    --owner-approval) (( seen_approval == 0 && $# >= 2 )) || die 'duplicate or missing --owner-approval'; owner_approval_path="$2"; seen_approval=1; shift 2 ;;
    --approval-sha256) (( seen_sha == 0 && $# >= 2 )) || die 'duplicate or missing --approval-sha256'; approval_sha256="$2"; seen_sha=1; shift 2 ;;
    --expected-head) (( seen_head == 0 && $# >= 2 )) || die 'duplicate or missing --expected-head'; expected_head="$2"; seen_head=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then [[ "$work_dir" == "$WORK" && "$model_free" == 0 && "$dependency_audit_only" == 0 && -z "$owner_approval_path" && -z "$approval_sha256" && -z "$expected_head" ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
if (( model_free )); then
  [[ "$seen_head" == 1 && "$seen_approval" == 0 && "$seen_sha" == 0 ]] || die '--model-free requires --expected-head and accepts no approval arguments'
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die 'expected HEAD must be exactly 40 lowercase hexadecimal characters'
  if (( seen_work )); then run_model_free "$expected_head" "$work_dir"; else run_model_free "$expected_head" ''; fi
  exit $?
fi
if (( dependency_audit_only )); then
  [[ "$seen_head" == 1 && "$seen_approval" == 0 && "$seen_sha" == 0 && "$model_free" == 0 ]] || die '--dependency-audit-only requires --expected-head and accepts no model-free or approval arguments'
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die 'expected HEAD must be exactly 40 lowercase hexadecimal characters'
  if (( seen_work )); then run_dependency_audit_only "$expected_head" "$work_dir"; else run_dependency_audit_only "$expected_head" ''; fi
  exit $?
fi
[[ $seen_approval == 1 && $seen_sha == 1 && $seen_head == 1 ]] || die '--owner-approval, --approval-sha256, and --expected-head are required'
require_clean_expected_head "$expected_head"
require_approval_binding "$owner_approval_path" "$approval_sha256"
require_blocked_approval "$owner_approval_path" "$expected_head" || die 'approval schema/identity/disposition validation failed'
die 'BLOCKED_APPROVAL/INSPECTION_ONLY: FireRed dependency/license/source/config/native-beam/parity facts remain unresolved; no acquisition or upload'
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
  echo 'runtime_status=LOUD_PARTIAL_FAIL_CLOSED'
  echo 'cpu_status=PARTIAL'
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
# The inspector must publish one explicit artifact-to-GGUF-key binding for
# each inference sidecar.  Keep this check before preparation so a worker can
# never proceed with independently validated, but swapped, cmvn/dict inputs.
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$FIRERED_PROJECT" --python 3.12 python - "$work_dir/evidence/manifest.json" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
binding = manifest.get("sidecar_binding")
if not isinstance(binding, dict) or binding.get("status") != "AUTHENTICATED_EXTERNAL_SIDECARS_BOUND":
    raise SystemExit("external sidecar binding is missing or unauthenticated")
if binding.get("required_for") != "converter_and_executable_runtime":
    raise SystemExit("external sidecar binding scope drifted")
records = binding.get("records")
if not isinstance(records, dict) or set(records) != {"cmvn.txt", "dict.txt"}:
    raise SystemExit("external sidecar binding record set drifted")
expected_marker_policy = {
    "status": "AUTHENTICATED_DICTIONARY_ANCHORS",
    "dictionary_anchor_rows": [["<blank>", 0], ["<unk>", 1], ["<pad>", 2], ["<sos>", 3], ["<eos>", 4]],
    "forbidden_decoder_ids": [["blank", 0], ["pad", 2], ["sos", 3]],
    "unknown_id": 1,
    "unknown_policy": "render_dictionary_token",
    "eos_id": 4,
    "eos_policy": "strip_only_if_terminal",
    "ordinary_content_id_range": [5, 7832],
    "sentencepiece_boundary": "replace U+2581 with ASCII space, then trim",
}
dictionary_structure = records["dict.txt"].get("structure")
if not isinstance(dictionary_structure, dict) or dictionary_structure.get("structural_marker_policy") != expected_marker_policy:
    raise SystemExit("dictionary structural-marker policy drifted from the runtime contract")
source_contract = manifest.get("official_source_contract") or manifest.get("source_contract")
if not isinstance(source_contract, dict) or source_contract.get("structural_marker_policy") != expected_marker_policy:
    raise SystemExit("source structural-marker policy is missing or drifted")
expected = {
    "cmvn.txt": ("inference_cmvn_text", "vokra.firered_asr_aed_l.cmvn_txt", "vokra.firered_asr_aed_l.cmvn_txt_sha256", 2985, "11816db612b43318ab01f9cfd05ee121dd3900b7a39d893f59d0104a06c199d2"),
    "dict.txt": ("inference_output_dictionary_text", "vokra.firered_asr_aed_l.dict_txt", "vokra.firered_asr_aed_l.dict_txt_sha256", 71448, "6907215aeb034f6926b26bf8abfd650f756781622480a2342ec1f29b2072cafe"),
}
for name, (role, metadata_key, digest_key, size, sha256) in expected.items():
    row = records[name]
    if row.get("status") != "AUTHENTICATED_EXTERNAL_SIDECAR_BOUND":
        raise SystemExit(f"{name} sidecar status is not authenticated")
    if (row.get("role"), row.get("metadata_key"), row.get("digest_key"), row.get("bytes"), row.get("sha256")) != (role, metadata_key, digest_key, size, sha256):
        raise SystemExit(f"{name} sidecar-to-GGUF binding drifted")
    artifact = row.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("bytes") != size or artifact.get("sha256") != sha256:
        raise SystemExit(f"{name} sidecar artifact identity is not bound")
print("FireRed external sidecar artifact/worker binding: PASS")
PY
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
assert manifest["runtime_status"] == "LOUD_PARTIAL_FAIL_CLOSED"
assert manifest["runtime_status_scope"] == "full_pcm_transcription_only; feature-to-feature and feature-to-token primitives are parity-pending"
assert manifest["cpu_status"] == "PARTIAL"
assert manifest["publication"] == "NO_UPLOAD"
assert manifest["inspection_status"] == "AUTHENTICATED_EVIDENCE_COMPLETE"
preparation = manifest.get("preparation")
assert isinstance(preparation, dict)
assert preparation["status"] == "PREPARED"
assert preparation["publication"] == "NO_UPLOAD"
assert preparation["runtime_status"] == "LOUD_PARTIAL_FAIL_CLOSED"
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
    "fireredasr/models/module/transformer_decoder.py",
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
assert contract["search"] == {
    "name": "batch_beam_search",
    "beam_size": 3,
    "nbest": 1,
    "decode_max_len": 0,
    "softmax_smoothing": 1.25,
    "length_penalty": 0.6,
    "eos_penalty": 1.0,
}
assert "INSPECTION_ERROR" not in json.dumps(manifest)
assert reference["format"] == "vokra-firered-asr-aed-l-upstream-reference-v1"
assert reference["status"] == "REFERENCE_CAPTURED"
assert reference["publication"] == "NO_UPLOAD"
assert reference["model"] == {"repository": "FireRedTeam/FireRedASR-AED-L", "revision": "e57f5960d03cff1071ff7acbb409314d1e70ed3d"}
assert reference["checkpoint"] == {"repository": "FireRedTeam/FireRedASR-AED-L", "revision": "e57f5960d03cff1071ff7acbb409314d1e70ed3d", "bytes": 4678597714, "sha256": "12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3"}
assert reference["source"]["revision"] == "834635e4cf277ed8ca92049fc375b17c3dc20748"
environment = reference.get("environment")
assert isinstance(environment, dict)
assert isinstance(environment.get("python"), str) and environment["python"]
assert isinstance(environment.get("platform"), str) and environment["platform"]
assert isinstance(environment.get("machine"), str) and environment["machine"]
assert isinstance(environment.get("torch"), str) and environment["torch"]
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
assert [record["path"] for record in source_records] == [
    "fireredasr/data/asr_feat.py",
    "fireredasr/models/fireredasr_aed.py",
    "fireredasr/models/module/conformer_encoder.py",
    "fireredasr/models/module/transformer_decoder.py",
    "fireredasr/tokenizer/aed_tokenizer.py",
    "fireredasr/data/token_dict.py",
    "README.md",
]
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
assert reference["reference"]["official_search"] == contract["search"]
official_hypotheses = reference["reference"].get("official_hypotheses")
assert isinstance(official_hypotheses, list) and len(official_hypotheses) == contract["search"]["nbest"]
for hypothesis in official_hypotheses:
    assert set(hypothesis) == {"token_ids", "normalized_log_score", "upstream_cost"}
    assert isinstance(hypothesis["token_ids"], list)
    assert isinstance(hypothesis["normalized_log_score"], (int, float))
    assert isinstance(hypothesis["upstream_cost"], (int, float))
    assert hypothesis["upstream_cost"] == -hypothesis["normalized_log_score"]
beam_trace = reference["reference"].get("beam_trace")
assert isinstance(beam_trace, dict)
assert beam_trace["schema"] == "firered-asr-aed-l-official-beam-trace-v1"
assert isinstance(beam_trace.get("steps"), list) and beam_trace["steps"]
assert all("token_topk" in step and "beam_prune_topk" in step for step in beam_trace["steps"])
assert beam_trace["final_ranking"]["k"] == contract["search"]["nbest"]
assert "torch.topk" in beam_trace["tie_behavior"]
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
die 'FireRedASR inspection, preparation and independent upstream reference evidence preserved as final no-clobber manifest; native seams are implemented but real CPU parity and full transcription remain blocked'
