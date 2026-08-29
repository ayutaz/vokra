#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HF_REPOSITORY="ibm-granite/granite-speech-4.1-2b"
HF_REVISION="de575db64086f84fdc79da4932d1076e965bc546"
SOURCE_URL="https://github.com/ibm-granite/granite-speech.git"
SOURCE_REVISION="77b7b12fff71f577105b517645750717a1598caa"
TRANSFORMERS_URL="https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION="753d61104116eefc8ffc977327b441ee0c8d599f"
INSPECTOR="$ROOT/tools/parity/granite_speech_4_1_2b_inspect.py"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((16 * 1024 * 1024))
die() { echo "granite-speech-vast: ERROR: $*" >&2; exit 2; }
self_test() {
  local path="${BASH_SOURCE[0]}" token fail=0
  for token in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" \
    "$TRANSFORMERS_URL" "$TRANSFORMERS_REVISION" \
    'list_repo_tree' 'model_info' 'get_hf_file_metadata' 'hf_hub_url' 'authenticated HEAD metadata' 'commit_hash' 'etag' 'requested_revision' 'resolved_revision' 'recursive=True' 'expand=True' 'RepoFile' 'RepoFolder' 'isinstance(item, RepoFolder)' 'unknown HF tree entry type' 'recursive_file_only' 'lfs_payload_size' 'git_blob_sha1' 'lfs_pointer_git_blob_sha1' 'lfs_sha256' 'model.sig' 'predicate.resources' 'hash_type' 'allow_symlinks' 'verificationMaterial' 'SIGSTORE_CRYPTOGRAPHIC_VERIFICATION_NOT_PERFORMED' \
    'classify_entry' 'GRANITE_HF_TREE_SELF_TEST' 'RepoFile(type=None)' \
    'HEADER_ONLY' '64 * 1024 * 1024' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'AUTHENTICATED' 'INSPECTION_ERROR' 'UNVERIFIED' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'UNSUPPORTED' \
    'BLOCKED_BY_CPU' 'NOT_RUN' 'NO_UPLOAD' 'UNREVIEWED_BLOCKER' 'UNAUTHENTICATED_BLOCKER' \
    'CARGO_BUILD_JOBS=1' 'cargo metadata --locked --no-deps --format-version 1' 'exit 2' \
    '--transformers-source' 'arguments are not accepted; revisions are fixed'; do
    if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$INSPECTOR"; then echo "missing contract $token" >&2; fail=1; fi
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$path" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  if grep -Eq '^(HF|SOURCE|TRANSFORMERS)_REVISION=.*\$\{' "$path"; then echo 'revision override found' >&2; fail=1; fi
  if grep -En 'weights_only=False|pickle\.load|torch\.load' "$INSPECTOR" >/dev/null; then echo 'unsafe loader found' >&2; fail=1; fi
  UV_CACHE_DIR="${GRANITE_UV_CACHE_DIR:-/tmp/vokra-granite-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test || fail=1
  local python_source
  python_source="$(mktemp "${TMPDIR:-/tmp}/granite-hf-tree-self-test.XXXXXX.py")"
  awk '/<<'"'"'PY'"'"'/{capture=1; next} capture && /^PY$/{exit} capture' \
    "$path" > "$python_source"
  if ! GRANITE_HF_TREE_SELF_TEST=1 UV_CACHE_DIR="${GRANITE_UV_CACHE_DIR:-/tmp/vokra-granite-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$python_source"; then
    echo 'HF tree class-identity self-test failed' >&2
    fail=1
  fi
  rm -f -- "$python_source"
  UV_CACHE_DIR="${GRANITE_UV_CACHE_DIR:-/tmp/vokra-granite-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY' || fail=1
import json
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result
def accepted(raw):
    packet = json.loads(raw, object_pairs_hook=unique)
    return packet.get("status") == "BLOCKED" and packet.get("evidence_stage") == "INSPECTION_ONLY" and packet.get("inspection_status") == "AUTHENTICATED_EVIDENCE_COMPLETE" and packet.get("collection_status") == "AUTHENTICATED" and packet.get("publication") == "NO_UPLOAD"
valid = json.dumps({"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","publication":"NO_UPLOAD"})
assert accepted(valid)
assert not accepted(valid.replace("AUTHENTICATED_EVIDENCE_COMPLETE", "INSPECTION_ERROR"))
assert not accepted(valid.replace('"AUTHENTICATED"', '"UNVERIFIED"'))
try: json.loads('{"status":"BLOCKED","status":"BLOCKED"}', object_pairs_hook=unique)
except ValueError: pass
else: raise AssertionError("duplicate manifest key accepted")
PY
  (( fail == 0 )) || return 1
  echo 'run-granite-speech-4-1-2b-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
[[ $# == 0 ]] || die 'arguments are not accepted; revisions are fixed'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '128 GiB memory guard failed'
mkdir -p /dev/shm/vokra-granite-speech-inspection
[[ -z "$(find /dev/shm/vokra-granite-speech-inspection -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'inspection directory must be empty'
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die 'tmpfs disk guard failed'
for command in cargo git uv awk find df; do command -v "$command" >/dev/null || die "missing tool: $command"; done
work=/dev/shm/vokra-granite-speech-inspection; mkdir -p "$work/model" "$work/source" "$work/transformers" "$work/evidence"; export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${GRANITE_UV_CACHE_DIR:-/tmp/vokra-granite-uv-cache}"
{
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1
} > "$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129 # heredoc output is one validation stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/tree.json" <<'PY' >> "$work/evidence/validation.log" 2>&1
import json,os,re,sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, get_hf_file_metadata, hf_hub_url

def classify_entry(entry):
    if isinstance(entry, RepoFolder):
        return False
    if isinstance(entry, RepoFile):
        return True
    raise RuntimeError(f"unknown HF tree entry type: {type(entry).__name__}")


if os.environ.get("GRANITE_HF_TREE_SELF_TEST") == "1":
    file_entry = object.__new__(RepoFile)
    file_entry.type = None
    folder_entry = object.__new__(RepoFolder)
    folder_entry.type = None
    assert classify_entry(file_entry)
    assert not classify_entry(folder_entry)
    try:
        classify_entry(object())
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown HF tree entry was accepted")
    print("Granite HF tree class-identity self-test: PASS")
    raise SystemExit(0)

repo,rev,out=sys.argv[1:]
api=HfApi(); info=api.model_info(repo,revision=rev); assert info.sha==rev and re.fullmatch(r"[0-9a-f]{40}",info.sha or "")
rows=[]
for entry in api.list_repo_tree(repo_id=repo,revision=rev,recursive=True,expand=True):
    if not classify_entry(entry):
        continue
    path=getattr(entry,"path",None); size=getattr(entry,"size",None)
    lfs=getattr(entry,"lfs",None)
    lfs_sha=getattr(lfs,"sha256",None) if lfs is not None else None
    if isinstance(lfs,dict): lfs_sha=lfs.get("sha256")
    blob=getattr(entry,"blob_id",None) or getattr(entry,"oid",None)
    if not isinstance(path,str):
        raise RuntimeError(f"HF entry has no safe path: {path!r}")
    metadata = get_hf_file_metadata(hf_hub_url(repo_id=repo, filename=path, revision=rev), token=os.environ.get("HF_TOKEN"))
    metadata_revision = getattr(metadata,"commit_hash",None)
    metadata_size = getattr(metadata,"size",None)
    metadata_etag = str(getattr(metadata,"etag","") or "").strip('"')
    if metadata_etag.startswith("sha256:"):
        metadata_etag = metadata_etag[7:]
    if metadata_revision != info.sha or metadata_size != size:
        raise RuntimeError(f"authenticated HEAD metadata mismatch: {path}")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", metadata_etag):
        raise RuntimeError(f"missing authenticated HEAD etag: {path}")
    # Xet-backed files can report lfs=None.  The fixed model artifacts are
    # still LFS payloads; bind their payload SHA/size from the reviewed table
    # and treat the API blob_id as the canonical pointer Git object.
    fixed={
      "model-00001-of-00003.safetensors": (2143518808,"3c987fdc29940c49d2498ea5925e8d57f88661af3ef30f73e56e2434ded3e42f"),
      "model-00002-of-00003.safetensors": (2143963456,"8e18d6d3fbe009a95a4cf305e31c2aab4a3484eccbce29aa1aa1454fc8c046ee"),
      "model-00003-of-00003.safetensors": (339045512,"32f823497bc179f6f346efdd46984ab60e44b3d443bf40a18d757ddce626a2d2"),
      "out_llm.safetensors": (205723810,"6cc10d68fe05aec359aceffd597617c875b23f27211ee6dcdb7510d9e90fc64e"),
    }
    fixed_identity=fixed.get(path)
    if fixed_identity is not None:
        if size != fixed_identity[0]: raise RuntimeError(f"fixed artifact size mismatch: {path}")
        if metadata_etag != fixed_identity[1]: raise RuntimeError(f"fixed artifact HEAD etag mismatch: {path}")
        if lfs_sha is not None and lfs_sha != fixed_identity[1]: raise RuntimeError(f"fixed artifact payload SHA mismatch: {path}")
        lfs_sha = fixed_identity[1]
    is_lfs = bool(re.fullmatch(r"[0-9a-f]{64}", metadata_etag))
    if lfs_sha is not None and metadata_etag != lfs_sha:
        raise RuntimeError(f"HF LFS payload etag mismatch: {path}")
    if not path or "\\" in path or "\x00" in path or path.startswith("/") or ".." in Path(path).parts or not isinstance(size,int) or size < 0 or not isinstance(blob,str) or not re.fullmatch(r"[0-9a-f]{40}",blob) or (is_lfs and not re.fullmatch(r"[0-9a-f]{64}",lfs_sha)) or (not is_lfs and metadata_etag != blob):
        raise RuntimeError(f"incomplete canonical HF identity: {path}")
    rows.append({"path":path,"type":"file","size":size,"lfs_payload_size":size if is_lfs else None,"head_commit":metadata_revision,"head_size":metadata_size,"head_etag":metadata_etag,"git_blob_sha1":blob if not is_lfs else None,"lfs_pointer_git_blob_sha1":blob if is_lfs else None,"lfs_sha256":lfs_sha})
if len({row["path"] for row in rows}) != len(rows): raise RuntimeError("duplicate canonical HF tree path")
Path(out).write_text(json.dumps({"repository":repo,"requested_revision":rev,"resolved_revision":info.sha,"head_commit":info.sha,"walk":"recursive_file_only","files":sorted(rows,key=lambda row:row["path"])},sort_keys=True,indent=2)+"\n")
PY
# shellcheck disable=SC2129 # heredoc output is one validation stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/model" <<'PY' >> "$work/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=["*"])
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >> "$work/evidence/validation.log" 2>&1
git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work/evidence/validation.log" 2>&1
[[ "$(git -C "$work/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'
[[ "$(git -C "$work/source/repo" remote get-url origin)" == "$SOURCE_URL" ]] || die 'source origin mismatch'
git clone --filter=blob:none "$TRANSFORMERS_URL" "$work/transformers/repo" >> "$work/evidence/validation.log" 2>&1
git -C "$work/transformers/repo" checkout --detach "$TRANSFORMERS_REVISION" >> "$work/evidence/validation.log" 2>&1
[[ "$(git -C "$work/transformers/repo" rev-parse HEAD)" == "$TRANSFORMERS_REVISION" ]] || die 'Transformers revision mismatch'
[[ "$(git -C "$work/transformers/repo" remote get-url origin)" == "$TRANSFORMERS_URL" ]] || die 'Transformers origin mismatch'
set +e
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --source "$work/source/repo" --transformers-source "$work/transformers/repo" --server-tree "$work/tree.json" --output "$work/evidence" >> "$work/evidence/validation.log" 2>&1
status=$?; set -e
[[ "$status" == 2 ]] || die 'inspector must exit 2'
grep -Fq '"status": "BLOCKED"' "$work/evidence/manifest.json" || die 'blocker manifest missing'
grep -Fq '"publication": "NO_UPLOAD"' "$work/evidence/manifest.json" || die 'publication status missing'
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/manifest.json" <<'PY'
import json, sys
from pathlib import Path
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise SystemExit(f"duplicate manifest key: {key}")
        out[key] = value
    return out
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=unique)
expected = {
    "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "collection_status": "AUTHENTICATED", "publication": "NO_UPLOAD",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"manifest contract mismatch: {key}={manifest.get(key)!r}")
if manifest.get("sigstore_crypto", {}).get("status") != "SIGSTORE_CRYPTOGRAPHIC_VERIFICATION_NOT_PERFORMED":
    raise SystemExit("Sigstore crypto status is not an explicit blocker")
PY
echo "Granite Speech inspection blocked; evidence preserved at $work/evidence" >&2
exit 2
