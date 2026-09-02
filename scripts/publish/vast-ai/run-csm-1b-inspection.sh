#!/usr/bin/env bash
# VAST-only CSM-1B inspection. No conversion, Cargo build/test, upload, or publish.
set -euo pipefail

HF_REPOSITORY="sesame/csm-1b"
HF_REVISION="c92a71e1c419772e25be7dc14d952c2521a740ab"
SOURCE_REPOSITORY="https://github.com/SesameAILabs/csm.git"
SOURCE_REVISION="8f6d947a26f6301deec9696f9bfb28e9e2e0d7d5"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers.git"
TRANSFORMERS_TAG="v4.52.1"
TRANSFORMERS_COMMIT="945727948c1143a10ac6f7d811aa58bb0d126b5b"
PUBLIC_REPOSITORY="vokra/csm-1b"
PUBLIC_REVISION="81613fc840fa995f4c8f1c48749fd731ed6424b8"
INSPECTOR="tools/parity/csm_1b_inspect.py"
REFERENCE_PROJECT="tools/parity/csm_1b_reference"
REFERENCE_LOCK_SHA256="62b70ae227b81a2eda59716c2a613f8322405abbf352dc74a5774ffa541a75bc"
UV_CMD=(uv run --no-sync --frozen --project "$REFERENCE_PROJECT" --python 3.12 python)
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_SHM_KIB=$((40 * 1024 * 1024))

die() { echo "run-csm-1b-inspection: $*" >&2; exit 2; }

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 status needle py gate_line sync_line download_line
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  py=python
  [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  [[ -f "$root/$REFERENCE_PROJECT/pyproject.toml" ]] || die "dedicated reference project missing"
  for needle in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_REPOSITORY" "$TRANSFORMERS_TAG" "$TRANSFORMERS_COMMIT" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$INSPECTOR" "CSM_PUBLIC_CONTRACT" "public/model.gguf" "AUTHENTICATED_EVIDENCE_COMPLETE" "INSPECTION_ERROR" "INSPECTION_ONLY" "NO_UPLOAD" "collection_status" "requested_revision" "snapshot_download" "list_repo_tree" "RepoFolder" "get_hf_file_metadata" "hf_hub_url" "ckpt.pt" "transformers.safetensors.index.json" "FULL_CSM_TRANSFORMERS_COMPOSITE_ROLES_PRESENT" "codec_model." "tokenizer.json" "caller-owned NumPy" "librosa/soxr" "pytorch-cpu" "uv.lock" "$REFERENCE_LOCK_SHA256" "BLOCKED_LICENSE_METADATA_REVIEW" "REVIEWED_LICENSE_AUDIT_COMPLETE" "source_transformers_requirement" "source_huggingface_hub_requirement" "isolated_transformers_pin" "isolated_huggingface_hub_pin" "GHSA-xrqw-3rrv-vx5w" "BLOCKED_UNVERIFIED_API_SMOKE" "--no-sync" "uv sync --project" "NOT_RUN_OFFICIAL_ONLY"; do
    if ! grep -Fq -- "$needle" "$self" && ! grep -Fq -- "$needle" "$root/$INSPECTOR"; then echo "self-test FAIL: missing $needle" >&2; fail=1; fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: mutation/conversion/Cargo test found" >&2; fail=1; fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: raw Python/pip found" >&2; fail=1; fi
  if ! grep -Fq 'torch.load(path, weights_only=True, map_location="cpu")' "$root/$INSPECTOR" || grep -En 'weights_only=False|pickle\.loads[[:space:]]*\(|pickle\.Unpickler' "$root/$INSPECTOR" >/dev/null; then echo "self-test FAIL: restricted checkpoint loader contract missing/unsafe" >&2; fail=1; fi
  if (( MIN_SHM_KIB != 40 * 1024 * 1024 )); then echo "self-test FAIL: tmpfs threshold unit drift" >&2; fail=1; fi
  gate_line="$(grep -n 'csm_1b_dump_reference.py" --dependency-gate || die' "$self" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^uv sync --project' "$self" | tail -1 | cut -d: -f1)"
  download_line="$(grep -n 'snapshot_download(repo_id' "$self" | tail -1 | cut -d: -f1)"
  if [[ -z "$gate_line" || -z "$sync_line" || -z "$download_line" || "$gate_line" -ge "$sync_line" || "$sync_line" -ge "$download_line" ]]; then
    echo "self-test FAIL: dependency sync must follow the affirmative gate and precede download" >&2; fail=1
  fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/csm-uv-cache}" uv run --no-sync --frozen --project "$root/tools/parity" --python 3.12 "$py" "$root/$INSPECTOR" --self-test || fail=1
  if bash "$self" --self-test --work-dir /tmp/csm-self-test >/dev/null 2>&1; then echo "self-test FAIL: extra arguments accepted" >&2; fail=1; else status=$?; [[ "$status" == 2 ]] || { echo "self-test FAIL: expected exit 2, got $status" >&2; fail=1; }; fi
  (( fail == 0 )) && echo "run-csm-1b-inspection.sh self-test: OK" || return 1
}

work_dir="/dev/shm/vokra-csm-1b-inspection"; self=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self=1; shift;;
    --work-dir) [[ $# -ge 2 ]] || die "--work-dir requires path"; work_dir="$2"; shift 2;;
    -h|--help) echo "usage: $0 [--work-dir /dev/shm/path] | --self-test"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
if (( self == 1 )); then [[ "$work_dir" == "/dev/shm/vokra-csm-1b-inspection" ]] || die "--self-test accepts no other arguments"; self_test; exit 0; fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
parent="$(dirname "$work_dir")"; [[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die "work path is not a directory"; [[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work path is not empty"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 128 GiB"
shm="$(df -Pk /dev/shm | awk 'NR == 2 {print $4; exit}')"; [[ "$shm" =~ ^[0-9]+$ && "$shm" -ge "$MIN_SHM_KIB" ]] || die "tmpfs below 40 GiB"
for command in git uv findmnt sha256sum; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
[[ -n "${CSM_PUBLIC_CONTRACT:-}" && -f "$CSM_PUBLIC_CONTRACT" ]] || die "CSM_PUBLIC_CONTRACT is required"
[[ -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$REFERENCE_PROJECT/uv.lock" ]] || die "dedicated CSM reference uv.lock is required before download"
[[ "$(sha256sum "$REFERENCE_PROJECT/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die "dedicated CSM reference uv.lock identity mismatch"
"${UV_CMD[@]}" "$root/tools/parity/csm_1b_dump_reference.py" --dependency-gate || die "CSM dependency/license gate is not explicitly approved"
uv sync --project "$REFERENCE_PROJECT" --frozen --python 3.12
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; cache="$work_dir/cache"; model="$work_dir/model"; tree="$work_dir/server-tree.json"; source="$work_dir/source"; transformers="$work_dir/transformers"; evidence="$work_dir/evidence"; public="$work_dir/public"; mkdir -p "$cache" "$model" "$evidence" "$public"

"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$cache" "$model" "$tree" <<'PY'
import json, os, re, sys, hashlib
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, get_hf_file_metadata, hf_hub_url, snapshot_download
repo, rev, cache, model, tree = sys.argv[1:]
api = HfApi(); info = api.model_info(repo_id=repo, revision=rev); token = os.environ.get("HF_TOKEN") or os.environ.get("HF")
if info.sha != rev: raise SystemExit(f"HF revision drift: {info.sha} != {rev}")
snapshot = Path(snapshot_download(repo_id=repo, revision=rev, cache_dir=cache, local_dir=model, allow_patterns=["*"], token=token))
if snapshot.resolve() != Path(model).resolve(): raise SystemExit("local_dir materialization mismatch")
def pointer_blob(size, sha):
    value = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
files=[]
for item in api.list_repo_tree(repo_id=repo, revision=rev, recursive=True, expand=True):
    if isinstance(item, RepoFolder): continue
    if not isinstance(item, RepoFile): raise SystemExit(f"unknown HF tree entry type: {type(item).__name__}")
    blob=getattr(item,"blob_id",None) or getattr(item,"oid",None)
    if not isinstance(item.path,str) or not isinstance(item.size,int) or not isinstance(blob,str) or len(blob)!=40: raise SystemExit(f"incomplete server identity: {item}")
    metadata = get_hf_file_metadata(hf_hub_url(repo_id=repo, filename=item.path, revision=rev), token=token)
    metadata_size = getattr(metadata, "size", None)
    etag = str(getattr(metadata, "etag", "") or "").strip('"')
    if etag.startswith("sha256:"): etag = etag[7:]
    metadata_revision = getattr(metadata, "commit_hash", None)
    if metadata_revision not in (None, info.sha) or metadata_size != item.size: raise SystemExit(f"HF HEAD metadata mismatch: {item.path}")
    if re.fullmatch(r"[0-9a-f]{40}", etag):
        if etag != blob: raise SystemExit(f"regular Git HEAD identity mismatch: {item.path}")
        lfs_sha, lfs_size = None, None
    elif re.fullmatch(r"[0-9a-f]{64}", etag):
        if pointer_blob(item.size, etag) != blob: raise SystemExit(f"LFS pointer identity mismatch: {item.path}")
        lfs_sha, lfs_size = etag, item.size
    else: raise SystemExit(f"unclassifiable HF HEAD identity: {item.path}")
    files.append({"path":item.path,"type":"file","size":item.size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha,"lfs_size":lfs_size})
expected={x["path"] for x in files}; actual=set()
for p in snapshot.rglob("*"):
    rel=p.relative_to(snapshot)
    if rel.parts[:2] == (".cache", "huggingface"):
        if p.is_symlink(): raise SystemExit(f"invalid transport cache symlink: {rel}")
        continue
    if p.is_symlink():
        raise SystemExit(f"invalid local payload symlink: {rel}")
    elif p.is_file(): actual.add(rel.as_posix())
    elif not p.is_dir(): raise SystemExit(f"non-regular local member: {rel}")
if expected != actual: raise SystemExit(f"server/local tree mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
Path(tree).write_text(json.dumps({"repository":repo,"requested_revision":rev,"resolved_revision":info.sha,"files":sorted(files,key=lambda x:x["path"])},sort_keys=True,indent=2)+"\n",encoding="utf-8")
PY

"${UV_CMD[@]}" - "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$public" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
repo, rev, destination = sys.argv[1:]
api = HfApi(); info = api.model_info(repo_id=repo, revision=rev)
if info.sha != rev: raise SystemExit(f"public HF revision drift: {info.sha} != {rev}")
path = Path(hf_hub_download(repo_id=repo, filename="model.gguf", revision=rev, local_dir=destination, token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
if path.name != "model.gguf" or path.is_symlink() or not path.is_file(): raise SystemExit("invalid public GGUF materialization")
PY

git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source" >/dev/null 2>&1; git -C "$source" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1; [[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "source revision mismatch"; [[ -z "$(git -C "$source" status --porcelain --untracked-files=all)" ]] || die "source checkout dirty"; [[ "$(git -C "$source" remote get-url origin | sed 's/\.git$//')" == "${SOURCE_REPOSITORY%.git}" ]] || die "source origin mismatch"
git clone --no-tags --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers" >/dev/null 2>&1; git -C "$transformers" fetch --tags --quiet origin "$TRANSFORMERS_TAG"; git -C "$transformers" checkout --detach "$TRANSFORMERS_COMMIT" >/dev/null 2>&1; [[ "$(git -C "$transformers" describe --exact-match --tags HEAD)" == "$TRANSFORMERS_TAG" ]] || die "Transformers tag mismatch"; [[ "$(git -C "$transformers" rev-parse HEAD)" == "$TRANSFORMERS_COMMIT" ]] || die "Transformers commit mismatch"; [[ -z "$(git -C "$transformers" status --porcelain --untracked-files=all)" ]] || die "Transformers checkout dirty"
set +e; "${UV_CMD[@]}" "$INSPECTOR" --snapshot "$model" --source "$source" --transformers "$transformers" --server-tree "$tree" --public-contract "$CSM_PUBLIC_CONTRACT" --public-gguf "$public/model.gguf" --output "$evidence"; status=$?; set -e
[[ "$status" == 2 ]] || die "inspector did not return exit 2"; [[ -f "$evidence/manifest.json" ]] || die "manifest missing"
"${UV_CMD[@]}" - "$evidence/manifest.json" <<'PY'
import json, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key, value in {"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","composite_status":"BLOCKED_ROLE_MAPPING_AND_PARITY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","publication":"NO_UPLOAD","collection_status":"AUTHENTICATED"}.items():
    if m.get(key)!=value: raise SystemExit(f"invalid fail-closed manifest: {key}={m.get(key)!r}")
if m.get("inspection_status")=="INSPECTION_ERROR": raise SystemExit("inspection failed; evidence incomplete")
if m.get("inspection_status")!="AUTHENTICATED_EVIDENCE_COMPLETE": raise SystemExit("authenticated completion marker missing")
PY
echo "CSM-1B inspection BLOCKED (evidence only; no upload); evidence=$evidence" >&2
exit 2
