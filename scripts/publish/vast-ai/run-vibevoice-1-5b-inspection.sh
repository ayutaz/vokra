#!/usr/bin/env bash
# shellcheck disable=SC2317
# VAST-only VibeVoice-1.5B evidence collection. No conversion, upload, or publish.
set -euo pipefail

HF_REPOSITORY="microsoft/VibeVoice-1.5B"
HF_REVISION="142f4a5dda029212cda8b118e9d99c3da27018d8"
QWEN_REPOSITORY="Qwen/Qwen2.5-1.5B"
QWEN_REVISION="8faed761d45a263340a0528343f099c05c9a4323"
SOURCE_REPOSITORY="https://github.com/microsoft/VibeVoice.git"
SOURCE_REVISION="2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION="5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
INSPECTOR="tools/parity/vibevoice_1_5b_inspect.py"
MIN_MEM_KIB=$((64 * 1024 * 1024))
MIN_DISK_KIB=$((32 * 1024 * 1024))
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)

die() { echo "run-vibevoice-1-5b-inspection: $*" >&2; exit 2; }
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

validate_approval() {
  local path="$1" expected_head="$2" supplied_sha="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$supplied_sha" "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_REPOSITORY" "$TRANSFORMERS_REVISION" <<'PY'
import hashlib, json, pathlib, sys
path, expected_head, supplied_sha, hf_repo, hf_rev, qwen_repo, qwen_rev, source_repo, source_rev, transformers_repo, transformers_rev = sys.argv[1:]
raw = pathlib.Path(path).read_bytes()
if hashlib.sha256(raw).hexdigest() != supplied_sha: raise SystemExit("approval bytes changed")
def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError("duplicate approval key: " + key)
        result[key] = value
    return result
data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
keys = {"schema", "status", "disposition", "expected_head", "hf_repository", "hf_revision", "qwen_repository", "qwen_revision", "source_repository", "source_revision", "transformers_repository", "transformers_revision", "no_upload", "scope_sha256"}
if set(data) != keys: raise ValueError("approval schema is not exact")
expected = {"schema": "vokra-vibevoice-1-5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY", "expected_head": expected_head, "hf_repository": hf_repo, "hf_revision": hf_rev, "qwen_repository": qwen_repo, "qwen_revision": qwen_rev, "source_repository": source_repo, "source_revision": source_rev, "transformers_repository": transformers_repo, "transformers_revision": transformers_rev, "no_upload": True}
if any(data[key] != value for key, value in expected.items()): raise ValueError("approval identity/status mismatch")
scope = {key: data[key] for key in ("disposition", "expected_head", "hf_repository", "hf_revision", "no_upload", "qwen_repository", "qwen_revision", "source_repository", "source_revision", "status", "transformers_repository", "transformers_revision")}
if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): raise ValueError("approval scope mismatch")
PY
}

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 token
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  for token in "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_REVISION" "$INSPECTOR" "snapshot_download" "list_repo_tree" "recursive=True" "expand=True" "RepoFile" "RepoFolder" "isinstance(item, RepoFolder)" "git_blob_sha1" "lfs_pointer_git_blob_sha1" "lfs_payload_sha256" "INSPECTION_ONLY" "AUTHENTICATED_EVIDENCE_COMPLETE" "INSPECTION_ERROR" "NO_UPLOAD" "--expected-head" "--approval-evidence" "--approval-sha256" "BLOCKED_PROVENANCE/NO_UPLOAD"; do
    if ! grep -Fq -- "$token" "$self" && ! grep -Fq -- "$token" "$root/$INSPECTOR"; then echo "self-test FAIL: missing $token" >&2; fail=1; fi
  done
  for token in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'findmnt' 'git status --porcelain --untracked-files=all' 'git rev-parse HEAD' 'canonical_existing_path' 'CARGO_BUILD_JOBS'; do
    grep -Fq -- "$token" "$self" || { echo "self-test FAIL: missing VAST gate $token" >&2; fail=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check)' "$self" >/dev/null; then echo "self-test FAIL: mutation/conversion/Cargo found" >&2; fail=1; fi
  if grep -En '(^|[;&|][[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: bare Python found" >&2; fail=1; fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$self" <<'PY'
import re, sys
source = open(sys.argv[1], encoding="utf-8").read()
for index, block in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)):
    compile(block, f"<inline-python-{index}>", "exec")
PY
  then echo 'self-test FAIL: inline Python syntax' >&2; fail=1; fi
  if (( fail == 0 )); then echo "run-vibevoice-1-5b-inspection.sh self-test: OK"; else return 1; fi
}

work_dir="/dev/shm/vokra-vibevoice-1-5b-inspection"; self=0; expected_head=''; approval=''; approval_sha=''; seen_head=0; seen_approval=0; seen_approval_sha=0; seen_work=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; seen_head=1; expected_head="$2"; shift 2;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && "$2" == /* && "$2" != -* ]] || die '--approval-evidence requires absolute path'; seen_approval=1; approval="$2"; shift 2;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; seen_approval_sha=1; approval_sha="$2"; shift 2;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && "$2" == /* && "$2" != -* ]] || die '--work-dir requires absolute path'; seen_work=1; work_dir="$2"; shift 2;;
    -h|--help) echo "usage: $0 --expected-head HEX40 --approval-evidence FILE --approval-sha256 HEX64 [--work-dir ABSENT_DIR] | --self-test"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
if (( self == 1 )); then [[ "$work_dir" == "/dev/shm/vokra-vibevoice-1-5b-inspection" && "$seen_head$seen_approval$seen_approval_sha$seen_work" == 0000 ]] || die "self-test accepts no other arguments"; self_test; exit $?; fi
[[ "$seen_head$seen_approval$seen_approval_sha" == 111 ]] || die 'expected-head, approval-evidence, and approval-sha256 are required'
[[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence is missing, empty, or symlinked'
canonical_existing_path "$approval" >/dev/null || die 'approval path has unsafe ancestry'
[[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 mismatch'
validate_approval "$approval" "$expected_head" "$approval_sha"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -d "$root/.git" && -z "$(git status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD differs from --expected-head'
die 'BLOCKED_PROVENANCE/NO_UPLOAD: VibeVoice provenance/runtime/reference surface is not authenticated'
# shellcheck disable=SC2317
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "clean checkout required"
[[ -f "$root/tools/parity/uv.lock" ]] || die "locked parity project is missing before acquisition"
parent="$(dirname "$work_dir")"; [[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die "work path is not a directory"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work path is not empty"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 64 GiB"
free="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_DISK_KIB" ]] || die "scratch below 32 GiB"
for command in git uv sha256sum findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; export CARGO_BUILD_JOBS=1
cache="$work_dir/cache"; snapshot="$work_dir/model"; packet="$work_dir/model-tree.json"; qwen="$work_dir/qwen"; qwen_packet="$work_dir/qwen-tree.json"; source="$work_dir/source"; transformers="$work_dir/transformers"; evidence="$work_dir/evidence"; mkdir -p "$cache" "$snapshot" "$qwen" "$evidence"

"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$cache" "$snapshot" "$qwen" "$packet" "$qwen_packet" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download

repo, revision, qwen_repo, qwen_revision, cache, destination, qwen_destination, packet, qwen_packet = sys.argv[1:]
api = HfApi()

def write_packet(repository, revision, destination, packet, patterns):
    info = api.model_info(repo_id=repository, revision=revision)
    if info.sha != revision:
        raise SystemExit(f"HF revision drift: {info.sha} != {revision}")
    snapshot = Path(snapshot_download(repo_id=repository, revision=revision, cache_dir=cache, local_dir=destination, allow_patterns=patterns, token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
    if snapshot.resolve() != Path(destination).resolve():
        raise SystemExit("snapshot local_dir mismatch")
    rows = []
    selected = set(patterns)
    seen = set()
    for item in api.list_repo_tree(repo_id=repository, revision=revision, recursive=True, expand=True):
        if isinstance(item, RepoFolder):
            continue
        if not isinstance(item, RepoFile):
            raise SystemExit(f"unsupported server member: {repository}:{item}")
        kind = getattr(item, "type", None)
        if kind not in {None, "file"}:
            raise SystemExit(f"unsupported RepoFile type: {repository}:{item}")
        lfs = getattr(item, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        blob = getattr(item, "blob_id", None)
        path = item.path
        if not isinstance(path, str) or not path or "\\" in path or path.startswith("/") or ".." in path.split("/") or path in seen:
            raise SystemExit(f"unsafe/duplicate server path: {repository}:{path!r}")
        if not isinstance(item.size, int) or isinstance(item.size, bool) or item.size < 0 or not isinstance(blob, str) or not __import__("re").fullmatch(r"[0-9a-f]{40}", blob):
            raise SystemExit(f"incomplete server identity: {repository}:{item}")
        if lfs_sha is not None and (not isinstance(lfs_sha, str) or not __import__("re").fullmatch(r"[0-9a-f]{64}", lfs_sha) or not isinstance(lfs_size, int) or isinstance(lfs_size, bool) or lfs_size < 0):
            raise SystemExit(f"invalid LFS identity: {repository}:{item.path}")
        if lfs_sha is not None and lfs_size != item.size:
            raise SystemExit(f"invalid LFS size: {repository}:{item.path}")
        if path in selected and lfs_sha is None:
            row = {"path": path, "type": "file", "size": item.size, "git_blob_sha1": blob, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None}
        elif path in selected:
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {item.size}\n".encode()
            pointer_git = __import__("hashlib").sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
            if pointer_git != blob:
                raise SystemExit(f"LFS pointer Git blob mismatch: {repository}:{path}")
            row = {"path": path, "type": "file", "size": item.size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": blob, "lfs_payload_sha256": lfs_sha, "lfs_payload_size": item.size}
        else:
            # Validate every unselected file, but only materialize the exact
            # tokenizer allowlist for the Qwen companion.
            if lfs_sha is not None:
                pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {item.size}\n".encode()
                if __import__("hashlib").sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest() != blob:
                    raise SystemExit(f"LFS pointer Git blob mismatch: {repository}:{path}")
            seen.add(path)
            continue
        rows.append(row); seen.add(path)
    if not rows:
        raise SystemExit(f"empty server tree: {repository}")
    expected = set(patterns)
    if set(row["path"] for row in rows) != expected:
        raise SystemExit(f"fixed selected tree drift: {repository}")
    Path(packet).write_text(json.dumps({"repository": repository, "requested_revision": revision, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(rows, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")

write_packet(repo, revision, destination, packet, [".gitattributes", "README.md", "config.json", "preprocessor_config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"])
write_packet(qwen_repo, qwen_revision, qwen_destination, qwen_packet, ["LICENSE", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"])
PY

git init "$source" >/dev/null 2>&1 || die "source init failed"
git -C "$source" remote add origin "$SOURCE_REPOSITORY" || die "source origin setup failed"
git -C "$source" fetch --no-tags --filter=blob:none --depth=1 origin "$SOURCE_REVISION" >/dev/null 2>&1 || die "source commit fetch failed"
git -C "$source" checkout --detach FETCH_HEAD >/dev/null 2>&1 || die "source checkout failed"
git clone --no-tags --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers" >/dev/null 2>&1 || die "Transformers clone failed"
git -C "$transformers" checkout --detach "$TRANSFORMERS_REVISION" >/dev/null 2>&1 || die "Transformers checkout failed"
set +e
"${UV_CMD[@]}" "$INSPECTOR" --snapshot "$snapshot" --source "$source" --transformers "$transformers" --server-tree "$packet" --qwen-snapshot "$qwen" --qwen-tree "$qwen_packet" --output "$evidence"
inspection_rc=$?
set -e
[[ "$inspection_rc" == 2 ]] || die "inspector must terminate with exit 2"
"${UV_CMD[@]}" - "$evidence/manifest.json" <<'PY'
import json, sys
def no_dupes(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate manifest key: {key}")
        result[key] = value
    return result
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read(), object_pairs_hook=no_dupes)
required = {"status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE"}
for key, value in required.items():
    if manifest.get(key) != value:
        raise SystemExit(f"inspection evidence incomplete: {key}={manifest.get(key)!r}")
if manifest.get("inspection_status") == "INSPECTION_ERROR":
    raise SystemExit("inspection error was treated as complete")
if manifest.get("collection_status") != "AUTHENTICATED":
    raise SystemExit("inspection collection was not independently authenticated")
PY
echo "VibeVoice-1.5B evidence complete but runtime blocked; evidence=$evidence" >&2
exit 2
