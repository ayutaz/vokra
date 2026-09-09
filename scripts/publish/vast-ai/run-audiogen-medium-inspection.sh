#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audiogen_medium_reference"
INSPECTOR="$ROOT/tools/parity/audiogen_medium_inspect.py"
METADATA_AUDIT="$ROOT/tools/parity/audiogen_medium_reference/metadata_audit.py"
T5_METADATA_AUDIT="$ROOT/tools/parity/audiogen_medium_reference/t5_metadata_audit.py"
DEPENDENCY_AUDIT="$ROOT/tools/parity/audiogen_medium_reference/dependency_audit.py"
REFERENCE="$ROOT/tools/parity/audiogen_medium_dump_reference.py"
HF_REPOSITORY="facebook/audiogen-medium"; HF_REVISION="1277dd7dfd8fa57a205a70acc5de0ee90804502f"
SOURCE_URL="https://github.com/facebookresearch/audiocraft.git"; SOURCE_REVISION="a2b96756956846e194c9255d0cdadc2b47c93f1b"
MIN_MEM_KIB=$((64*1024*1024)); MIN_DISK_KIB=$((16*1024*1024)); MODEL_FREE_MIN_DISK_KIB=$((256*1024))
die(){ echo "audiogen-medium-vast: BLOCKED: $*" >&2; exit 2; }
usage(){ cat >&2 <<'EOF'
usage: run-audiogen-medium-inspection.sh --model-free --expected-head <40-hex>
   or: run-audiogen-medium-inspection.sh --expected-head <40-hex> \
       --approval-evidence <file> --approval-sha256 <64-hex>
   or: run-audiogen-medium-inspection.sh --self-test
EOF
}
validate_model_free_manifest(){
  local path="$1" expected_head="$2"
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python - "$path" "$expected_head" "$PROJECT" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
sys.path.insert(0, str(Path(sys.argv[3]).parent))
from companion_contract import contract as canonical_companion_contract
from dependency_audit import audit as dependency_audit
from audiogen_medium_inspect import pending_approval_scope
def unique(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise SystemExit(f"duplicate manifest key: {key}")
  out[key]=value
 return out
m=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"),object_pairs_hook=unique)
expected_head=sys.argv[2]
expected_keys={"format","status","evidence_stage","runtime_status","cpu_status","metal_status","parity_status","publication","companion_contract","t5_server_metadata","inspection_status","collection_status","expected_head","approval_evidence","approval_scope","dependency_closure","upstream","archives","compression_companion","external_text_conditioner","official_source","license_evidence","blockers"}
if set(m) != expected_keys: raise SystemExit("model-free manifest root closure mismatch")
required={"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","runtime_status":"LOUD_PARTIAL_FAIL_CLOSED","cpu_status":"NOT_RUN","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD"}
for key,want in required.items():
 if m.get(key)!=want: raise SystemExit(f"model-free marker mismatch: {key}={m.get(key)!r}")
if m.get("expected_head") != expected_head: raise SystemExit("model-free expected HEAD mismatch")
if m.get("approval_evidence") != {"status":"PENDING_OWNER_APPROVAL"}: raise SystemExit("model-free approval state is not pending")
up=m.get("upstream")
if not isinstance(up,dict) or up.get("repository")!="facebook/audiogen-medium" or up.get("requested_revision")!="1277dd7dfd8fa57a205a70acc5de0ee90804502f" or up.get("resolved_revision")!="1277dd7dfd8fa57a205a70acc5de0ee90804502f": raise SystemExit("model-free upstream identity mismatch")
archives=m.get("archives")
if not isinstance(archives,dict) or set(archives) != {"compression_state_dict.bin","state_dict.bin"}: raise SystemExit("model-free archive closure mismatch")
for name,row in archives.items():
 if not isinstance(row,dict) or row.get("payload_status") != "NOT_DOWNLOADED" or row.get("execution") != "NOT_PERFORMED": raise SystemExit(f"model-free payload execution marker mismatch: {name}")
for key in ("compression_companion","external_text_conditioner"):
 companion=m.get(key)
 if not isinstance(companion,dict) or companion.get("payload") != "NOT_DOWNLOADED": raise SystemExit(f"model-free companion payload marker mismatch: {key}")
contract=m.get("companion_contract")
if contract != canonical_companion_contract(): raise SystemExit("model-free companion contract identity mismatch")
if m.get("t5_server_metadata") != contract["text_conditioner"]["server_metadata"]: raise SystemExit("model-free T5 server metadata binding mismatch")
if m["t5_server_metadata"].get("payload") != "NOT_DOWNLOADED": raise SystemExit("model-free T5 metadata crossed payload boundary")
if contract.get("text_conditioner",{}).get("payload") != "NOT_DOWNLOADED" or contract.get("compression_companion",{}).get("payload") != "NOT_DOWNLOADED": raise SystemExit("model-free companion contract crossed payload boundary")
dependency=dependency_audit(Path(sys.argv[3]))
if m.get("dependency_closure") != dependency: raise SystemExit("model-free dependency closure identity mismatch")
if m.get("approval_scope") != pending_approval_scope(expected_head, dependency): raise SystemExit("model-free pending approval scope mismatch")
PY
}
validate_manifest(){
  local path="$1"
  local expected_head="$2" approval_sha256="$3"
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python - "$path" "$expected_head" "$approval_sha256" "$PROJECT" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from companion_contract import contract as canonical_companion_contract
from dependency_audit import audit as dependency_audit
def unique(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise SystemExit(f"duplicate manifest key: {key}")
  out[key]=value
 return out
m=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"),object_pairs_hook=unique); expected_head=sys.argv[2]; approval_sha256=sys.argv[3]
expected_keys={"format","status","evidence_stage","runtime_status","cpu_status","metal_status","parity_status","publication","companion_contract","t5_server_metadata","inspection_status","collection_status","expected_head","approval_evidence","dependency_closure","upstream","archives","compression_companion","external_text_conditioner","official_source","license_evidence","blockers"}
if set(m) != expected_keys: raise SystemExit(f"manifest root closure mismatch: {sorted(set(m)^expected_keys)}")
required={"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","runtime_status":"LOUD_PARTIAL_FAIL_CLOSED","cpu_status":"NOT_RUN","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD"}
for key,want in required.items():
 if m.get(key)!=want: raise SystemExit(f"manifest marker mismatch: {key}={m.get(key)!r}")
up=m.get("upstream")
if not isinstance(up,dict) or up.get("repository")!="facebook/audiogen-medium" or up.get("requested_revision")!="1277dd7dfd8fa57a205a70acc5de0ee90804502f" or up.get("resolved_revision")!="1277dd7dfd8fa57a205a70acc5de0ee90804502f": raise SystemExit("upstream identity mismatch")
if m.get("inspection_status") in {"INSPECTION_ERROR","FAILED"} or m.get("collection_status")!="AUTHENTICATED": raise SystemExit("incomplete/error evidence rejected")
if m.get("expected_head") != expected_head: raise SystemExit("expected HEAD was not recorded")
contract=m.get("companion_contract")
if contract != canonical_companion_contract(): raise SystemExit("companion contract identity mismatch")
if m.get("t5_server_metadata") != contract["text_conditioner"]["server_metadata"]: raise SystemExit("T5 server metadata binding mismatch")
if m["compression_companion"].get("payload") != "PRESENT" or m["compression_companion"].get("path") != "compression_state_dict.bin": raise SystemExit("normal compression companion state mismatch")
if m["external_text_conditioner"].get("payload") != "NOT_DOWNLOADED": raise SystemExit("normal T5 payload boundary mismatch")
if m["t5_server_metadata"].get("payload") != "NOT_DOWNLOADED": raise SystemExit("normal T5 metadata crossed payload boundary")
if contract.get("text_conditioner",{}).get("payload") != "NOT_DOWNLOADED" or contract.get("compression_companion",{}).get("payload") != "NOT_DOWNLOADED": raise SystemExit("companion contract crossed payload boundary")
if m.get("dependency_closure") != dependency_audit(Path(sys.argv[4])): raise SystemExit("dependency closure identity mismatch")
approval=m.get("approval_evidence")
if not isinstance(approval,dict) or approval.get("evidence_sha256") != approval_sha256 or approval.get("schema") != "vokra-audiogen-medium-inspection-approval-v1": raise SystemExit("approval evidence binding mismatch")
PY
}
self_test(){
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq '${TMPDIR:-/tmp}/vokra-audiogen-uv-cache' "$0"
  linux_cache="$(AUDIOGEN_UV_CACHE_DIR= TMPDIR=/var/tmp bash -c 'printf "%s" "${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}"')"
  [[ "$linux_cache" == /var/tmp/vokra-audiogen-uv-cache ]] || die 'Linux TMPDIR cache fallback drifted'
  grep -Fq 'LOUD_PARTIAL_FAIL_CLOSED' "$INSPECTOR"
  grep -Fq 'AUTHENTICATED_EVIDENCE_COMPLETE' "$INSPECTOR"
  grep -Fq -- '--model-free' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'PENDING_OWNER_APPROVAL' "$INSPECTOR"
  grep -Fq 'NOT_DOWNLOADED' "$ROOT/tools/parity/audiogen_medium_reference/metadata_audit.py"
  grep -Fq 'snapshot_download' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'dedicated AudioGen uv.lock absent; fail before downloads' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'PASS_MODEL_FREE' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'evidence_sha256' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'exit 0' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'payload_status' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 't5_metadata_audit.py' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq -- '--t5-server-metadata' "$INSPECTOR"
  grep -Fq 'BLOCKED_MISSING_LOCK_INPUTS' "$DEPENDENCY_AUDIT"
  grep -Fq 'NO_LOCAL_EXECUTION' "$DEPENDENCY_AUDIT"
  grep -Fq 'companion_contract' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq 'VOKRA_PUBLISH_ON_VAST=1' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq -- '--model-free' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq -- '--expected-head' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq -- '--approval-sha256' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  grep -Fq -- 'uv run --no-project --offline' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  synthetic="$(mktemp -d "${TMPDIR:-/tmp}/audiogen-medium-manifest-self-test.XXXXXX")"
  linux_template="$(TMPDIR=/var/tmp bash -c 'printf "%s" "${TMPDIR:-/tmp}/audiogen-medium-linux-self-test.XXXXXX"')"
  [[ "$linux_template" == /var/tmp/audiogen-medium-linux-self-test.XXXXXX ]] || die 'Linux TMPDIR synthetic path drifted'
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python - "$synthetic" "$PROJECT" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, str(Path(sys.argv[2]).parent))
from companion_contract import contract
from dependency_audit import audit as dependency_audit
from audiogen_medium_inspect import pending_approval_scope
root=Path(sys.argv[1]); canonical=contract(); head="a"*40; approval_sha="b"*64; dependency=dependency_audit(Path(sys.argv[2]))
common={"format":"vokra-audiogen-medium-inspection-v2","status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","runtime_status":"LOUD_PARTIAL_FAIL_CLOSED","cpu_status":"NOT_RUN","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","expected_head":head,"upstream":{"repository":"facebook/audiogen-medium","requested_revision":"1277dd7dfd8fa57a205a70acc5de0ee90804502f","resolved_revision":"1277dd7dfd8fa57a205a70acc5de0ee90804502f"},"archives":{"compression_state_dict.bin":{"payload_status":"NOT_DOWNLOADED","execution":"NOT_PERFORMED"},"state_dict.bin":{"payload_status":"NOT_DOWNLOADED","execution":"NOT_PERFORMED"}},"official_source":{},"license_evidence":{},"blockers":[],"companion_contract":canonical,"t5_server_metadata":canonical["text_conditioner"]["server_metadata"],"external_text_conditioner":{**canonical["text_conditioner"],"payload":"NOT_DOWNLOADED"},"dependency_closure":dependency}
model_free={**common,"approval_evidence":{"status":"PENDING_OWNER_APPROVAL"},"approval_scope":pending_approval_scope(head, dependency),"compression_companion":{**canonical["compression_companion"],"payload":"NOT_DOWNLOADED"}}
normal={**common,"approval_evidence":{"schema":"vokra-audiogen-medium-inspection-approval-v1","evidence_sha256":approval_sha},"compression_companion":{**canonical["compression_companion"],"payload":"PRESENT"}}
for name,value in (("model-free.json",model_free),("normal.json",normal)):
    (root/name).write_text(json.dumps(value),encoding="utf-8")
drift=json.loads(json.dumps(model_free)); drift["t5_server_metadata"]["files"][0]["bytes"]=0
(root/"drift.json").write_text(json.dumps(drift),encoding="utf-8")
PY
  validate_model_free_manifest "$synthetic/model-free.json" aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa || die 'synthetic model-free manifest was rejected'
  validate_manifest "$synthetic/normal.json" aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb || die 'synthetic normal manifest was rejected'
  if validate_model_free_manifest "$synthetic/drift.json" aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa >/dev/null 2>&1; then die 'synthetic T5 metadata drift was accepted'; fi
  if UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$T5_METADATA_AUDIT" --self-test "$synthetic/illegal.json" >/dev/null 2>&1; then die 'T5 metadata CLI accepted mixed self-test/output modes'; fi
  rm -rf "$synthetic"
  if "$0" --expected-head 0000000000000000000000000000000000000000 --expected-head 1111111111111111111111111111111111111111 >/dev/null 2>&1; then die 'duplicate --expected-head was accepted'; fi
  if "$0" --expected-head 0000000000000000000000000000000000000000 --approval-evidence a --approval-evidence b >/dev/null 2>&1; then die 'duplicate --approval-evidence was accepted'; fi
  if "$0" --expected-head 0000000000000000000000000000000000000000 --approval-evidence a --approval-sha256 0000000000000000000000000000000000000000000000000000000000000000 --approval-sha256 1111111111111111111111111111111111111111111111111111111111111111 >/dev/null 2>&1; then die 'duplicate --approval-sha256 was accepted'; fi
  if "$0" --model-free --expected-head 0000000000000000000000000000000000000000 --approval-evidence a >/dev/null 2>&1; then die 'model-free approval mix was accepted'; fi
  if "$0" --model-free --expected-head 0000000000000000000000000000000000000000 --approval-sha256 0000000000000000000000000000000000000000000000000000000000000000 >/dev/null 2>&1; then die 'model-free approval hash mix was accepted'; fi
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$INSPECTOR" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$METADATA_AUDIT" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$T5_METADATA_AUDIT" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$DEPENDENCY_AUDIT" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$REFERENCE" --self-test
  echo 'run-audiogen-medium-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then self_test "$@"; exit 0; fi
expected_head=''; approval_evidence=''; approval_sha256=''; model_free=0; seen_head=0; seen_approval=0; seen_approval_sha=0
while (($#)); do
  case "$1" in
    --model-free) (( model_free == 0 )) || die 'duplicate --model-free'; model_free=1; shift ;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_approval_sha=1; shift 2 ;;
    *) usage; die "unexpected argument: $1" ;;
  esac
done
(( seen_head == 1 )) || die '--expected-head is required'
if (( model_free == 1 )); then
  (( seen_approval == 0 )) || die '--model-free cannot mix with --approval-evidence'
  (( seen_approval_sha == 0 )) || die '--model-free cannot mix with --approval-sha256'
else
  (( seen_approval == 1 )) || die '--approval-evidence is required'
  (( seen_approval_sha == 1 )) || die '--approval-sha256 is required'
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
if (( model_free == 1 )); then
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
  for command in git uv awk df sha256sum realpath; do command -v "$command" >/dev/null || die "missing tool: $command"; done
  work="/dev/shm/vokra-audiogen-medium-model-free-${expected_head}"
  [[ -d /dev/shm && ! -L /dev/shm && "$(realpath -e /dev/shm)" == /dev/shm ]] || die 'model-free work parent is not the fixed /dev/shm path'
  root_real="$(realpath -e "$ROOT")"; [[ "$root_real" != /dev/shm && "$root_real" != /dev/shm/* ]] || die 'model-free work path overlaps checkout'
  [[ ! -e "$work" && ! -L "$work" ]] || die 'model-free work path must be absent (no-clobber)'
  free_kib="$(df -Pk "$(dirname "$work")" | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MODEL_FREE_MIN_DISK_KIB ]] || die 'model-free small disk guard failed'
  mkdir "$work"; mkdir -p "$work/source" "$work/evidence"
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$METADATA_AUDIT" "$work/tree.json" >>"$work/evidence/acquisition.log" 2>&1
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$T5_METADATA_AUDIT" "$work/t5-tree.json" >>"$work/evidence/acquisition.log" 2>&1
  git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >>"$work/evidence/acquisition.log" 2>&1
  git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/acquisition.log" 2>&1
  set +e
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$INSPECTOR" --model-free --source "$work/source/repo" --server-tree "$work/tree.json" --t5-server-metadata "$work/t5-tree.json" --output "$work/evidence" --expected-head "$expected_head" --vokra-root "$ROOT" >>"$work/evidence/acquisition.log" 2>&1
  inspector_rc=$?
  set -e
  [[ "$inspector_rc" == 2 ]] || die "model-free inspector returned unexpected status: $inspector_rc"
  validate_model_free_manifest "$work/evidence/manifest.json" "$expected_head" || die 'model-free inspection did not produce complete authenticated evidence'
  evidence_sha256="$(sha256sum "$work/evidence/manifest.json" | awk '{print $1}')"
  printf '{"status":"PASS_MODEL_FREE","evidence_sha256":"%s","payload":"NOT_DOWNLOADED","load":"NOT_PERFORMED","run":"NOT_PERFORMED","upload":"NO_UPLOAD","approval":"PENDING_OWNER_APPROVAL"}\n' "$evidence_sha256"
  exit 0
fi
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$INSPECTOR" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die 'external approval evidence is invalid'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '64 GiB memory guard failed'
for command in git uv awk find df; do command -v "$command" >/dev/null || die "missing tool: $command"; done
work=/dev/shm/vokra-audiogen-medium-inspection
[[ ! -e "$work" && ! -L "$work" ]] || die 'inspection tmpfs path must be absent (no-clobber)'
mkdir "$work"
mkdir -p "$work/model" "$work/source" "$work/evidence"
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die '16 GiB tmpfs guard failed'
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --no-project --offline --python 3.12 python "$T5_METADATA_AUDIT" "$work/t5-tree.json" >>"$work/evidence/acquisition.log" 2>&1
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/model" "$work/tree.json" <<'PY' >>"$work/evidence/acquisition.log" 2>&1
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder,snapshot_download
repo,rev,destination,out=sys.argv[1:]; api=HfApi(); info=api.model_info(repo_id=repo,revision=rev)
if info.sha != rev: raise SystemExit("resolved HF revision mismatch")
snapshot=Path(snapshot_download(repo_id=repo,revision=rev,local_dir=destination,allow_patterns=["*"]))
if snapshot.resolve()!=Path(destination).resolve(): raise SystemExit("snapshot_download escaped local_dir")
expected={".gitattributes","README.md","compression_state_dict.bin","state_dict.bin"}; rows=[]
fixed={".gitattributes":(1519,"a6344aac8c09253b3b630fb776ae94478aa0275b"),"README.md":(2240,"31a77819df582937de900237706f104a325e223f"),"compression_state_dict.bin":(235740815,"0cc8de6c4cf0c16326ee3c693385370b98bbf0f2","5a520e64ca99226a9956f83b06df0617b713183fcdc384779883a6bb46dc1095"),"state_dict.bin":(3678455287,"ae572ad32705a0a9ba679b0d2813cbae716d869e","f3b20997834de1ca47d6a31d00a5dc37019b279c7c8f250fd482d56def04faaa")}
def pointer_sha1(sha,size):
 pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode(); h=hashlib.sha1(f"blob {len(pointer)}\0".encode()); h.update(pointer); return h.hexdigest()
for item in api.list_repo_tree(repo_id=repo,revision=rev,recursive=True,expand=True):
 if isinstance(item,RepoFolder): continue
 if not isinstance(item,RepoFile): raise SystemExit(f"unknown HF tree entry: {item!r}")
 if item.path not in expected: raise SystemExit(f"unexpected HF tree file: {item.path}")
 if not isinstance(item.size,int) or item.size<=0 or not isinstance(item.blob_id,str) or len(item.blob_id)!=40 or item.size!=fixed[item.path][0]: raise SystemExit(f"incomplete/fixed HF identity: {item.path}")
 lfs=getattr(item,"lfs",None); sha=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None); lfs_size=lfs.get("size") if isinstance(lfs,dict) else getattr(lfs,"size",None)
 if len(fixed[item.path])==2:
  if sha is not None or item.blob_id!=fixed[item.path][1]: raise SystemExit(f"regular Git identity mismatch: {item.path}")
  row={"path":item.path,"type":"file","size":item.size,"git_blob_sha1":item.blob_id,"lfs_pointer_git_blob_sha1":None,"lfs_payload_sha256":None,"lfs_payload_size":None}
 else:
  if sha is not None and (not isinstance(sha,str) or sha!=fixed[item.path][2]) or lfs_size is not None and lfs_size!=item.size: raise SystemExit(f"invalid LFS metadata: {item.path}")
  if item.blob_id!=fixed[item.path][1] or pointer_sha1(fixed[item.path][2],item.size)!=item.blob_id: raise SystemExit(f"invalid fixed LFS identity: {item.path}")
  row={"path":item.path,"type":"file","size":item.size,"git_blob_sha1":None,"lfs_pointer_git_blob_sha1":item.blob_id,"lfs_payload_sha256":fixed[item.path][2],"lfs_payload_size":item.size}
 rows.append(row)
if {row["path"] for row in rows}!=expected or len(rows)!=len(expected): raise SystemExit("HF tree is not exact four-file set")
Path(out).write_text(json.dumps({"repository":repo,"requested_revision":rev,"resolved_revision":info.sha,"walk":"recursive_file_only","files":sorted(rows,key=lambda row:row["path"])},sort_keys=True,indent=2)+"\n")
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >>"$work/evidence/acquisition.log" 2>&1
git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/acquisition.log" 2>&1
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-audiogen-uv-cache}" uv run --frozen --project "$PROJECT" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --source "$work/source/repo" --server-tree "$work/tree.json" --t5-server-metadata "$work/t5-tree.json" --output "$work/evidence" --expected-head "$expected_head" --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --vokra-root "$ROOT" >>"$work/evidence/acquisition.log" 2>&1 || [[ $? == 2 ]]
validate_manifest "$work/evidence/manifest.json" "$expected_head" "$approval_sha256" || die 'inspection did not produce complete authenticated evidence'
exit 2
