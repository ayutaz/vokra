#!/usr/bin/env bash
# VAST-only fixed-HF Sortformer inspection. Mutable weight-build provenance
# remains unresolved, so production is blocked before sync/download/clone.
set -euo pipefail
ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/sortformer_diar_4spk_v1_inspect.py"
REPO="nvidia/diar_sortformer_4spk-v1"
HF_REVISION="9f17b10df44c0a4c8f3c86fbddc9ee2d6ab9ac08"
SOURCE_URL="https://github.com/NVIDIA/NeMo.git"
SOURCE_REVISION="505acacf6444a67ff9a4020fb03a5e6d59953e05"
MODEL_BYTES=494206256
MODEL_SHA256="e8abcc5f3a82ff23134c98a37f70fef3f159611f394bb191a0ad0a6f4b052974"
WORK="/workspace/vokra-sortformer-diar-4spk-v1-inspection"
MIN_MEM_KIB=$((64 * 1024 * 1024))
MIN_DISK_KIB=$((150 * 1024 * 1024))
die() { echo "sortformer inspection: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else
    parent="$(dirname "$path")"; base="$(basename "$path")"
    parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"
  fi
}
canonical_absent_path() {
  local path="$1" target rest component current=/ suffix='' real
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  target="$path"; rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do component="$(basename "$target")"; suffix="/$component$suffix"; target="$(dirname "$target")"; done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  real="$(cd -P "$target" && pwd)" || return 1; printf '%s%s\n' "$real" "$suffix"
}
require_clean_head() {
  local expected="$1" actual
  [[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'not a Vokra checkout'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse HEAD)" || die 'cannot read checkout HEAD'
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual does not match --expected-head $expected"
}
require_absent_work_path() {
  local path="$1" approval="$2" absolute candidate root_real approval_real
  [[ "$path" == /* ]] || return 1
  [[ ! -e "$path" && ! -L "$path" ]] || return 1
  absolute="$path"; candidate="$(canonical_absent_path "$absolute")" || return 1
  root_real="$(canonical_existing_path "$ROOT")" || return 1
  approval_real="$(canonical_existing_path "$approval")" || return 1
  [[ "$candidate" != "$root_real" && "$candidate" != "$root_real"/* ]] || return 1
  [[ "$candidate" != "$approval_real" && "$candidate" != "$approval_real"/* && "$approval_real" != "$candidate"/* ]] || return 1
}
validate_approval() {
  local path="$1" expected="$2" supplied_sha="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected" "$supplied_sha" "$REPO" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" <<'PY'
import hashlib, json, pathlib, sys
path, expected_head, supplied_sha, repo, hf_revision, source_url, source_revision = sys.argv[1:]
raw = pathlib.Path(path).read_bytes()
if hashlib.sha256(raw).hexdigest() != supplied_sha:
    raise SystemExit("approval bytes changed")
def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate key: " + key)
        out[key] = value
    return out
data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
keys = {"schema", "status", "disposition", "model_repo", "hf_revision", "source_url", "source_revision", "license_status", "no_upload", "expected_head", "scope_sha256"}
if set(data) != keys:
    raise ValueError("approval schema is not exact")
identity = ("vokra-sortformer-diar-approval-v1", "BLOCKED", "INSPECTION_ONLY", repo, hf_revision, source_url, source_revision, "UNRESOLVED", True, expected_head)
actual = tuple(data[key] for key in ("schema", "status", "disposition", "model_repo", "hf_revision", "source_url", "source_revision", "license_status", "no_upload", "expected_head"))
if actual != identity:
    raise ValueError("approval identity/status mismatch")
scope = {key: data[key] for key in ("disposition", "expected_head", "hf_revision", "license_status", "model_repo", "no_upload", "source_revision", "source_url")}
if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
    raise ValueError("approval scope mismatch")
PY
}
usage() { echo "usage: run-sortformer-diar-4spk-v1-inspection.sh --expected-head <40-hex> --approval-evidence <file> --approval-sha256 <64-hex> [--work-dir ABSENT_DIR] | --self-test"; }
self_test() {
  local fail=0 token temporary
  for token in "$REPO" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" "$MODEL_BYTES" "$MODEL_SHA256" "493434880" "bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689" x86_64 MIN_MEM_KIB MIN_DISK_KIB /proc/meminfo 'df -Pk' VOKRA_PUBLISH_ON_VAST CARGO_BUILD_JOBS snapshot_download HfApi requested_revision lfs_pointer_git_blob_sha1 config.json processor_config.json model.safetensors diar_sortformer_4spk-v1.nemo sortformer_diar_4spk_v1_inspect.py weights_only=True 'status=BLOCKED' 'evidence_stage=INSPECTION_ONLY' INSPECTION_ONLY REFERENCE_SOURCE_SELECTED WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN NO_UPLOAD BLOCKED_PROVENANCE/NO_UPLOAD sha256sum --server-tree '--expected-head' '--approval-evidence' '--approval-sha256'; do
    grep -Fq -- "$token" "${BASH_SOURCE[0]}" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)' "${BASH_SOURCE[0]}" >/dev/null; then
    echo 'publication command found' >&2
    fail=1
  fi
  if "${BASH_SOURCE[0]}" --self-test --expected-head bad >/dev/null 2>&1 || \
    "${BASH_SOURCE[0]}" --self-test --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1 || \
    "${BASH_SOURCE[0]}" --self-test --approval-sha256 bad >/dev/null 2>&1; then
    echo 'malformed or duplicate approval/head option accepted' >&2; fail=1
  fi
  temporary="$(mktemp -d /private/tmp/sortformer-inspection-self-test.XXXXXX 2>/dev/null || mktemp -d /tmp/sortformer-inspection-self-test.XXXXXX)"
  trap 'rm -rf "$temporary"' RETURN
  printf '%s\n' approval > "$temporary/approval.json"
  require_absent_work_path "$temporary/work" "$temporary/approval.json" || { echo 'safe canonical work path rejected' >&2; fail=1; }
  ln -s "$temporary/approval.json" "$temporary/approval-link"
  if require_absent_work_path "$temporary/safe-work" "$temporary/approval-link" >/dev/null 2>&1; then echo 'symlinked approval accepted' >&2; fail=1; fi
  if require_absent_work_path "$temporary/./dot-work" "$temporary/approval.json" >/dev/null 2>&1; then echo 'dot-component work path accepted' >&2; fail=1; fi
  if require_absent_work_path "$ROOT/overlap" "$temporary/approval.json" >/dev/null 2>&1; then echo 'checkout overlap accepted' >&2; fail=1; fi
  trap - RETURN
  rm -rf "$temporary"
  UV_CACHE_DIR="${SORTFORMER_UV_CACHE_DIR:-/tmp/vokra-sortformer-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
  if ! UV_CACHE_DIR="${SORTFORMER_UV_CACHE_DIR:-/tmp/vokra-sortformer-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

def classify(entry):
    if isinstance(entry, RepoFolder):
        return "directory"
    if isinstance(entry, RepoFile):
        return "file"
    raise RuntimeError(f"unexpected entry: {entry!r}")

file_entry = RepoFile(path=".gitattributes", size=1584, oid="a" * 40)
file_entry.type = None
assert classify(file_entry) == "file"
assert classify(RepoFolder(path="nested", oid="b" * 40)) == "directory"
try:
    classify(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unexpected server-tree entry was accepted")
PY
  then
    echo 'hermetic RepoFile/RepoFolder regression failed' >&2
    fail=1
  fi
  (( fail == 0 )) || return 1
  echo 'run-sortformer-diar-4spk-v1-inspection.sh self-test: OK'
}
self=0
expected_head=''; approval_evidence=''; approval_sha256=''
seen_head=0; seen_approval=0; seen_approval_sha=0
seen_work=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; (( $# >= 2 )) && [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; seen_head=1; expected_head="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty file'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; (( $# >= 2 )) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; seen_approval_sha=1; approval_sha256="$2"; shift 2 ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; (($# >= 2)) || die '--work-dir requires DIR'; WORK="$2"; seen_work=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$WORK" == /workspace/vokra-sortformer-diar-4spk-v1-inspection && "$seen_head$seen_approval$seen_approval_sha$seen_work" == 0000 ]] || die '--self-test accepts no other arguments'
  self_test
  exit 0
fi
[[ "$seen_head$seen_approval$seen_approval_sha" == 111 ]] || die '--expected-head, --approval-evidence, and --approval-sha256 are required'
[[ "$approval_evidence" == /* ]] || die '--approval-evidence must be absolute'
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die 'approval evidence is missing, empty, or symlinked'
[[ "$(sha256_file "$approval_evidence")" == "$approval_sha256" ]] || die 'approval evidence SHA-256 does not match caller-supplied digest'
validate_approval "$approval_evidence" "$expected_head" "$approval_sha256" || die 'approval evidence schema or scope is invalid'
require_clean_head "$expected_head"
require_absent_work_path "$WORK" "$approval_evidence" || die '--work-dir is unsafe, overlaps an input, or has symlinked ancestry'
die 'BLOCKED_PROVENANCE/NO_UPLOAD: mutable NeMo weight-build provenance and incomplete native/reference surface'
# shellcheck disable=SC2317,SC2329
inspection_after_license_resolution() {
[[ "$(uname -s)" == Linux ]] || die 'Linux/VAST required'
[[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST host required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$INSPECTOR" ]] || die 'inspector missing'
parent="$(dirname "$WORK")"
mkdir -p "$parent"
[[ ! -e "$WORK" || -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
if ! [[ "$mem_kib" =~ ^[0-9]+$ ]] || (( mem_kib < MIN_MEM_KIB )); then die 'memory guard failed'; fi
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"
if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < MIN_DISK_KIB )); then die 'disk guard failed'; fi
for tool in cargo git uv sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mkdir -p "$WORK/evidence"
WORK="$(cd "$WORK" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="${SORTFORMER_UV_CACHE_DIR:-/tmp/vokra-sortformer-uv-cache}"
{
  echo "status=BLOCKED"
  echo "evidence_stage=INSPECTION_ONLY"
  echo "publication=NO_UPLOAD"
  echo "hf_revision=$HF_REVISION"
  echo "expected_model_bytes=$MODEL_BYTES"
  echo "expected_model_sha256=$MODEL_SHA256"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$WORK/evidence/validation.log" 2>&1
snapshot="$(uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$REPO" "$HF_REVISION" "$WORK/hf" "$WORK/hf-cache" 2>> "$WORK/evidence/validation.log" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, local_dir, cache_dir = sys.argv[1:]
print(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir, local_dir=local_dir, allow_patterns=["*"]))
PY
)"
printf '%s\n' "snapshot_path=$snapshot" | tee -a "$WORK/evidence/validation.log" >/dev/null
# shellcheck disable=SC2129 # heredoc output is one validation stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$REPO" "$HF_REVISION" "$snapshot" "$WORK/server-tree.json" <<'PY' >> "$WORK/evidence/validation.log" 2>&1
import json, re, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder
repo, revision, snapshot, output = sys.argv[1:]
api = HfApi(); info = api.model_info(repo, revision=revision)
if info.sha != revision: raise RuntimeError("resolved revision mismatch")
rows=[]
for entry in api.list_repo_tree(repo, revision=revision, recursive=True):
    if isinstance(entry, RepoFolder): continue
    if not isinstance(entry, RepoFile): raise RuntimeError(f"unexpected HF tree entry: {entry!r}")
    path = getattr(entry, "path", None); blob = getattr(entry, "blob_id", None) or getattr(entry, "oid", None); size = getattr(entry, "size", None)
    lfs = getattr(entry, "lfs", None); lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    local = Path(snapshot) / path if isinstance(path, str) else None
    if local is None or not local.is_file() or local.is_symlink() or not isinstance(size, int) or local.stat().st_size != size or not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob) or (lfs_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", lfs_sha)):
        raise RuntimeError(f"incomplete or unmaterialized HF row: {path}")
    rows.append({"path": path, "type": "file", "size": size, "git_blob_sha1": blob if lfs_sha is None else None, "lfs_pointer_git_blob_sha1": blob if lfs_sha is not None else None, "lfs_sha256": lfs_sha})
if len({row["path"] for row in rows}) != len(rows): raise RuntimeError("duplicate HF tree path")
Path(output).write_text(json.dumps({"repository": repo, "requested_revision": revision, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(rows, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n")
PY
git clone --filter=blob:none --no-tags "$SOURCE_URL" "$WORK/source/repo" >> "$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" checkout --detach "$SOURCE_REVISION" >> "$WORK/evidence/validation.log" 2>&1
set +e
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$snapshot" --evidence "$WORK/evidence" --server-tree "$WORK/server-tree.json" --source "$WORK/source/repo" >> "$WORK/evidence/validation.log" 2>&1
rc=$?
set -e
[[ "$rc" == 2 ]] || die "expected inspection blocker exit=2, got $rc"
[[ -s "$WORK/evidence/manifest.json" ]] || die 'inspection manifest missing before blocker exit'
grep -Fq '"status": "BLOCKED"' "$WORK/evidence/manifest.json" || die 'manifest did not remain BLOCKED'
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$WORK/evidence/manifest.json" || die 'evidence stage missing'
echo 'source_status=SEE_MANIFEST_AFTER_AUTHENTICATION' | tee -a "$WORK/evidence/validation.log"
echo 'weight_build_provenance=WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN' | tee -a "$WORK/evidence/validation.log"
echo 'verdict=BLOCKED; evidence_stage=INSPECTION_ONLY; publication=NO_UPLOAD' | tee -a "$WORK/evidence/validation.log"
return 2
}
