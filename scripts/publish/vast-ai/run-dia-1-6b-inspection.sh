#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HF_REPOSITORY="nari-labs/Dia-1.6B"
HF_REVISION="257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd"
PUBLIC_REPOSITORY="vokra/dia-1.6b"
PUBLIC_REVISION="dd1df2a129fed7d15c365caeabaae227ccfe8537"
SOURCE_URL="https://github.com/nari-labs/dia.git"
SOURCE_REVISION="2811af1c5f476b1f49f4744fabf56cf352be21e5"
INSPECTOR="$ROOT/tools/parity/dia_1_6b_inspect.py"
REFERENCE_PROJECT="$ROOT/tools/parity/dia_1_6b_reference"
REFERENCE_LOCK_SHA256="ccdfaf4cfedd7780f8c1032a42341f28ac56bec7353f4563f9a1b44b764cf29c"
REFERENCE_PYPROJECT_SHA256="56430b6f50620df9ce3383f535dec1755843a4a9bab9758e34cf69e9913b6fc2"
# dedicated locked-reference project; its uv.lock is a hard pre-download gate
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_SHM_KIB=$((40 * 1024 * 1024))
die(){ echo "dia-vast: ERROR: $*" >&2; exit 2; }
self_test(){
 local fail=0 token
  for token in "$HF_REPOSITORY" "$HF_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" "$REFERENCE_LOCK_SHA256" "$REFERENCE_PYPROJECT_SHA256"   'list_repo_tree' 'recursive_file_only' 'git_blob_sha1' 'lfs_sha256'   'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'PARTIAL_RUNTIME_FAIL_CLOSED'   'CPU_UNSUPPORTED_FULL_TTS' 'BLOCKED_BY_CPU' 'NO_UPLOAD' 'weights_only=True'   'lfs_pointer_sha1' 'PTH↔safetensors mapping evidence unavailable' '40 * 1024 * 1024' 'cargo metadata --locked --no-deps --format-version 1' 'uv.lock' 'dependency_license_audit' 'BLOCKED_UNREVIEWED_TRANSITIVE' 'AUDITED_ALLOW' 'sha256sum' '--no-project' 'dedicated locked-reference project' 'exit 2'; do
  grep -Fq -- "$token" "$INSPECTOR" "$0" || { echo "missing contract $token" >&2; fail=1; }
 done
 for token in 'torch.load' 'snapshot_download' 'model.safetensors' 'dia-v0_1.pth' 'public-gguf'; do
  grep -Fq -- "$token" "$INSPECTOR" || { echo "missing dia contract $token" >&2; fail=1; }
 done
 if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$INSPECTOR" "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
 if grep -Eq '^(HF|SOURCE)_REVISION=.*\$\{' "$0"; then fail=1; fi
 [[ -f "$REFERENCE_PROJECT/uv.lock" ]] || { echo 'dedicated Dia reference uv.lock absent; self-test deliberately blocked' >&2; return 2; }
 [[ "$(sha256sum "$REFERENCE_PROJECT/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || { echo 'Dia uv.lock identity mismatch' >&2; fail=1; }
 [[ "$(sha256sum "$REFERENCE_PROJECT/pyproject.toml" | awk '{print $1}')" == "$REFERENCE_PYPROJECT_SHA256" ]] || { echo 'Dia pyproject identity mismatch' >&2; fail=1; }
 grep -Fq 'dependency_license_audit = "BLOCKED_UNREVIEWED_TRANSITIVE"' "$REFERENCE_PROJECT/pyproject.toml" || { echo 'dependency audit gate missing' >&2; fail=1; }
 if grep -Eq 'librosa|soxr|gradio|triton|nvidia-|descript-audio-codec' "$REFERENCE_PROJECT/uv.lock"; then echo 'forbidden/UI/GPL/CUDA reference dependency in lock' >&2; fail=1; fi
 UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/private/tmp/vokra-dia-uv-cache}" uv run --no-project --python 3.12 python "$INSPECTOR" --self-test || fail=1
 (( fail == 0 )) || return 1
 echo 'run-dia-1-6b-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 0 ]] || die 'arguments are fixed; revisions cannot be overridden'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated Dia reference uv.lock is absent; refuse before any download'
[[ "$(sha256sum "$REFERENCE_PROJECT/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die 'dedicated Dia uv.lock identity mismatch'
[[ "$(sha256sum "$REFERENCE_PROJECT/pyproject.toml" | awk '{print $1}')" == "$REFERENCE_PYPROJECT_SHA256" ]] || die 'dedicated Dia pyproject identity mismatch'
grep -Fq 'dependency_license_audit = "AUDITED_ALLOW"' "$REFERENCE_PROJECT/pyproject.toml" || die 'dependency license/provenance audit is not affirmatively allowed; refuse before download'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '128 GiB memory guard failed'
command -v findmnt >/dev/null || die 'findmnt is required'
findmnt -T /dev/shm -no FSTYPE | grep -Fxq tmpfs || die '/dev/shm must be tmpfs'
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_SHM_KIB ]] || die 'tmpfs space guard failed'
for command in cargo git uv awk find df; do command -v "$command" >/dev/null || die "missing tool: $command"; done
WORK="/dev/shm/vokra-dia-1-6b-inspection"
[[ ! -e "$WORK" ]] || [[ -z "$(find "$WORK" -mindepth 1 -print -quit 2>/dev/null)" ]] || die 'inspection directory must be absent or empty'
mkdir -p "$WORK/model" "$WORK/public" "$WORK/source" "$WORK/evidence"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="${DIA_UV_CACHE_DIR:-/tmp/vokra-dia-uv-cache}"
cd "$ROOT"
{
 cargo fmt --all -- --check
 cargo metadata --locked --no-deps --format-version 1
} >"$WORK/evidence/validation.log" 2>&1
# shellcheck disable=SC2129
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$WORK/tree.json" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
import json,sys
from pathlib import Path
from huggingface_hub import HfApi
repo,rev,out=sys.argv[1:]
api=HfApi()
if api.model_info(repo,revision=rev).sha != rev: raise SystemExit("HF revision mismatch")
rows=[]
for entry in api.list_repo_tree(repo,revision=rev,recursive=True):
 if getattr(entry,"type",None)!="file": continue
 lfs=getattr(entry,"lfs",None); lfs_oid=getattr(lfs,"oid",None); lfs_size=getattr(lfs,"size",None)
 if isinstance(lfs,dict): lfs_oid=lfs.get("oid") or lfs.get("sha256"); lfs_size=lfs.get("size")
 path=getattr(entry,"path",None); size=getattr(entry,"size",None); blob=getattr(entry,"blob_id",None) or getattr(entry,"oid",None)
 if not isinstance(path,str) or not isinstance(size,int) or isinstance(size,bool) or size<0: raise SystemExit(f"bad server row: {entry!r}")
 if not isinstance(blob,str) or len(blob)!=40: raise SystemExit(f"bad Git blob: {path}")
 if lfs_oid is not None and (not isinstance(lfs_oid,str) or len(lfs_oid)!=64 or not isinstance(lfs_size,int)): raise SystemExit(f"bad LFS row: {path}")
 rows.append({"type":"file","path":path,"size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_oid,"lfs_size":lfs_size})
 if {r["path"] for r in rows} != {".gitattributes","README.md","config.json","preprocessor_config.json","dia-v0_1.pth","model.safetensors"}: raise SystemExit("fixed six-file tree mismatch")
Path(out).write_text(json.dumps({"repository":repo,"revision":rev,"resolved_revision":rev,"walk":"recursive_file_only","files":rows},sort_keys=True,indent=2)+"\n")
PY
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$WORK/model" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=[".gitattributes","README.md","config.json","preprocessor_config.json","dia-v0_1.pth","model.safetensors"])
PY
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$WORK/public" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=["dia-1.6b.gguf"])
PY
git clone --filter=blob:none "$SOURCE_URL" "$WORK/source/repo" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" checkout --detach "$SOURCE_REVISION" >>"$WORK/evidence/validation.log" 2>&1
[[ "$(git -C "$WORK/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'
[[ "$(git -C "$WORK/source/repo" remote get-url origin)" == "$SOURCE_URL" ]] || die 'source origin mismatch'
set +e
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python "$INSPECTOR" --snapshot "$WORK/model" --source "$WORK/source/repo" --server-tree "$WORK/tree.json" --public-gguf "$WORK/public/dia-1.6b.gguf" --output "$WORK/evidence" >>"$WORK/evidence/validation.log" 2>&1
status=$?
set -e
[[ "$status" == 2 ]] || die "inspector returned $status, expected 2"
grep -Fq '"status": "BLOCKED"' "$WORK/evidence/manifest.json" || die 'blocked manifest missing'
grep -Fq '"publication": "NO_UPLOAD"' "$WORK/evidence/manifest.json" || die 'NO_UPLOAD marker missing'
grep -Fq 'AUTHENTICATED_EVIDENCE_COMPLETE' "$WORK/evidence/manifest.json" || die 'inspection evidence did not complete'
echo "Dia inspection is blocked for runtime work; evidence preserved at $WORK/evidence" >&2
exit 2
