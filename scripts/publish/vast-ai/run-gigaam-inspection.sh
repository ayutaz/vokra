#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/gigaam_inspect.py"
SOURCE_URL="https://github.com/salute-developers/GigaAM.git"
SOURCE_REVISION="7447938d791c4f3e643386ee22c33777004293a5"
die(){ echo "gigaam-vast: ERROR: $*" >&2; exit 2; }
self_test(){ local path="${BASH_SOURCE[0]}" token fail=0; for token in "$SOURCE_URL" "$SOURCE_REVISION" 'weights_only=True' 'INSPECTION_ONLY' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'NO_UPLOAD' 'CARGO_BUILD_JOBS=1' 'model_class' 'RNNT' 'CTC'; do grep -Fq -- "$token" "$path" || grep -Fq -- "$token" "$INSPECTOR" || { echo "missing contract $token" >&2; fail=1; }; done; if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$path" | grep -v 'grep -En' >/dev/null; then fail=1; fi; UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" v3 --self-test || fail=1; UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" multilingual --self-test || fail=1; ((fail==0)) || return 1; echo 'run-gigaam-inspection.sh self-test: OK'; }
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no other arguments'; self_test; exit 0; fi
[[ $# == 1 && ( "$1" == v3 || "$1" == multilingual ) ]] || die 'usage: run-gigaam-inspection.sh {v3|multilingual}'
variant="$1"; [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'; [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
for command in cargo git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $((128*1024*1024)) ]] || die '128 GiB memory guard failed'; [[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'; free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $((32*1024*1024)) ]] || die 'tmpfs disk guard failed'
work="/dev/shm/vokra-gigaam-$variant"; [[ ! -e "$work" ]] || [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection directory must be empty'; mkdir -p "$work/model" "$work/source" "$work/evidence"; export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}"
{ cargo fmt --all -- --check; cargo metadata --locked --no-deps --format-version 1; } >"$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129 # validation output is one evidence stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$variant" "$work/tree.json" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys,re
from pathlib import Path
from huggingface_hub import HfApi
from tools.parity.gigaam_inspect import VARIANTS
variant,out=sys.argv[1:]; spec=VARIANTS[variant]; api=HfApi(); info=api.model_info(spec["repository"],revision=spec["revision"]); assert info.sha==spec["revision"]; rows=[]
for item in api.list_repo_tree(spec["repository"],revision=spec["revision"],recursive=True):
 if getattr(item,"type",None)!="file": continue
 lfs=getattr(item,"lfs",None); lfs_sha=getattr(lfs,"sha256",None) if lfs is not None else None
 if isinstance(lfs,dict): lfs_sha=lfs.get("sha256")
 path=getattr(item,"path",None); blob=getattr(item,"blob_id",None) or getattr(item,"oid",None); size=getattr(item,"size",None)
 assert isinstance(path,str) and path and "\\" not in path and ".." not in Path(path).parts and not path.startswith("/") and isinstance(size,int) and not isinstance(size,bool) and size>=0 and re.fullmatch(r"[0-9a-f]{40}",str(blob)) and (lfs_sha is None or re.fullmatch(r"[0-9a-f]{64}",str(lfs_sha)))
 rows.append({"path":path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha})
assert {row["path"] for row in rows}==set(spec["files"])
Path(out).write_text(json.dumps({"repository":spec["repository"],"revision":spec["revision"],"resolved_revision":info.sha,"walk":"recursive_file_only","files":rows},sort_keys=True,indent=2)+"\n")
PY
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$variant" "$work/model" <<'PY' >>"$work/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
from tools.parity.gigaam_inspect import VARIANTS
spec=VARIANTS[sys.argv[1]]; snapshot_download(repo_id=spec["repository"],revision=spec["revision"],local_dir=sys.argv[2],allow_patterns=list(spec["files"]))
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/validation.log" 2>&1; [[ "$(git -C "$work/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'; [[ "$(git -C "$work/source/repo" remote get-url origin)" == "$SOURCE_URL" ]] || die 'source origin mismatch'
set +e; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" "$variant" --snapshot "$work/model" --source "$work/source/repo" --server-tree "$work/tree.json" --output "$work/evidence" >>"$work/evidence/validation.log" 2>&1; rc=$?; set -e; [[ "$rc" == 2 ]] || die 'inspector must exit 2'; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/manifest.json" <<'PY'
import json,sys
p=json.loads(open(sys.argv[1]).read()); assert p["status"]=="BLOCKED" and p["inspection_status"]=="AUTHENTICATED_EVIDENCE_COMPLETE" and p["evidence_stage"]=="INSPECTION_ONLY" and p["runtime_status"]=="NOT_IMPLEMENTED_FAIL_CLOSED" and p["publication"]=="NO_UPLOAD"
PY
exit 2
