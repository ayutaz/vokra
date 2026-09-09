#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
INSPECTOR="$ROOT/tools/parity/cosyvoice2_inspect.py"
REFERENCE="$ROOT/tools/parity/cosyvoice2_dump_reference.py"
REFERENCE_PROJECT="$ROOT/tools/parity/cosyvoice2_reference"
MODEL_REVISION="eec1ae6c79877dbd9379285cf8789c9e0879293d"
SOURCE_REVISION="8555549e882236e6541748b1042d95693caa82ba"
MATCHA_REVISION="dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
TRANSFORMERS_SOURCE_REQUIREMENT="transformers==4.40.1"
TRANSFORMERS_SECURITY_ADVISORY="GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM="5.10.0"
ISOLATED_TRANSFORMERS_PIN="5.10.4"
die(){ echo "cosyvoice2-vast: ERROR: $*" >&2; exit 2; }
self_test(){
  local fail=0 token
  for token in "$MODEL_REVISION" "$SOURCE_REVISION" "$MATCHA_REVISION" "$TRANSFORMERS_SOURCE_REQUIREMENT" "$TRANSFORMERS_SECURITY_ADVISORY" "$TRANSFORMERS_SECURITY_PATCHED_MINIMUM" "$ISOLATED_TRANSFORMERS_PIN" 'BLOCKED_UNVERIFIED_API_SMOKE' 'weights_only=True' 'tensor_manifest_sha256' 'INSPECTION_ONLY' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'NO_UPLOAD' 'runtime_status' 'cpu_status' 'metal_status' 'parity_status' 'llm.pt' 'flow.pt' 'hift.pt' 'rand_noise' 'uv.lock'; do
    grep -Fq -- "$token" "$INSPECTOR" "$REFERENCE" "$REFERENCE_PROJECT/pyproject.toml" "$0" || { echo "missing contract $token" >&2; fail=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  UV_CACHE_DIR="${COSYVOICE2_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-uv-cache}" uv run --no-sync --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test || fail=1
  UV_CACHE_DIR="${COSYVOICE2_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-uv-cache}" uv run --no-sync --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test || fail=1
  (( fail == 0 )) || return 1
  echo 'run-cosyvoice2-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 0 ]] || die 'usage: run-cosyvoice2-inspection.sh'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated CosyVoice2 uv.lock is absent; fail closed before any download'
for command in cargo git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $((128*1024*1024)) ]] || die '128 GiB memory guard failed'
[[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $((32*1024*1024)) ]] || die 'tmpfs disk guard failed'
work="/dev/shm/vokra-cosyvoice2"; [[ ! -e "$work" ]] || [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection directory must be absent or empty'; mkdir -p "$work/model" "$work/source" "$work/matcha" "$work/evidence"
export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${COSYVOICE2_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-uv-cache}"
{ cargo fmt --all -- --check; cargo metadata --locked --no-deps --format-version 1; } >"$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$work/tree.json" "$work/patterns.json" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys,re
from pathlib import Path
from huggingface_hub import HfApi
from tools.parity.cosyvoice2_inspect import EXPECTED,REPOSITORY,REVISION
tree,patterns=map(Path,sys.argv[1:]); api=HfApi(); info=api.model_info(REPOSITORY,revision=REVISION); assert info.sha==REVISION
rows=[]
for item in api.list_repo_tree(REPOSITORY,revision=REVISION,recursive=True):
    if getattr(item,"type",None)!="file": continue
    path=getattr(item,"path",None); size=getattr(item,"size",None); blob=getattr(item,"blob_id",None) or getattr(item,"oid",None); lfs=getattr(item,"lfs",None); lfs_sha=(getattr(lfs,"sha256",None) or getattr(lfs,"oid",None)) if lfs is not None else None
    lfs_size=getattr(lfs,"size",None) if lfs is not None else None
    if isinstance(lfs,dict): lfs_sha=lfs.get("sha256") or lfs.get("oid"); lfs_size=lfs.get("size")
    assert isinstance(path,str) and not path.startswith("/") and "\\" not in path and ".." not in Path(path).parts and isinstance(size,int) and not isinstance(size,bool) and size>=0 and re.fullmatch(r"[0-9a-f]{40}",str(blob)) and (lfs_sha is None or re.fullmatch(r"[0-9a-f]{64}",str(lfs_sha)))
    assert lfs is None or isinstance(lfs_size,int) and not isinstance(lfs_size,bool) and lfs_size==size
    rows.append({"path":path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha,"lfs_size":lfs_size})
assert len(rows)==19, len(rows)
assert {r["path"] for r in rows} == set(EXPECTED)
for row in rows:
    expected_size, expected_blob, expected_lfs = EXPECTED[row["path"]]
    assert (row["size"], row["git_blob_sha1"], row["lfs_sha256"]) == (expected_size, expected_blob, expected_lfs)
tree.write_text(json.dumps({"repository":REPOSITORY,"revision":REVISION,"resolved_revision":info.sha,"walk":"recursive_file_only","files":rows},sort_keys=True,indent=2)+"\n")
patterns.write_text(json.dumps([r["path"] for r in rows])+"\n")
PY
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$work/patterns.json" "$work/model" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys
from huggingface_hub import snapshot_download
from tools.parity.cosyvoice2_inspect import REPOSITORY,REVISION
patterns,snapshot=sys.argv[1:]; snapshot_download(repo_id=REPOSITORY,revision=REVISION,local_dir=snapshot,allow_patterns=json.load(open(patterns)))
PY
git clone --filter=blob:none https://github.com/FunAudioLLM/CosyVoice.git "$work/source/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/validation.log" 2>&1
git -C "$work/source/repo" submodule update --init --recursive >>"$work/evidence/validation.log" 2>&1
git clone --filter=blob:none https://github.com/shivammehta25/Matcha-TTS.git "$work/matcha/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/matcha/repo" checkout --detach "$MATCHA_REVISION" >>"$work/evidence/validation.log" 2>&1
set +e; uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --source "$work/source/repo" --matcha-source "$work/matcha/repo" --server-tree "$work/tree.json" --output "$work/evidence" >>"$work/evidence/validation.log" 2>&1; rc=$?; set -e
[[ "$rc" == 2 ]] || die 'inspector must exit 2'
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$work/evidence/manifest.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p["status"]=="BLOCKED" and p["inspection_status"]=="AUTHENTICATED_EVIDENCE_COMPLETE" and p["evidence_stage"]=="INSPECTION_ONLY" and p["runtime_status"]=="NOT_IMPLEMENTED_FAIL_CLOSED" and p["cpu_status"]=="UNSUPPORTED" and p["metal_status"]=="BLOCKED_BY_CPU" and p["parity_status"]=="NOT_RUN" and p["publication"]=="NO_UPLOAD"
PY
exit 2
