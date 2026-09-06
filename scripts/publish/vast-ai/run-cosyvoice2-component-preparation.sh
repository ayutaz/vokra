#!/usr/bin/env bash
# VAST-only, explicit preparation of one pinned CosyVoice2 component.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PREPARER="$ROOT/tools/parity/cosyvoice2_component_prepare.py"
MODEL_REPOSITORY='FunAudioLLM/CosyVoice2-0.5B'
MODEL_REVISION='eec1ae6c79877dbd9379285cf8789c9e0879293d'
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'
SOURCE_REVISION='8555549e882236e6541748b1042d95693caa82ba'
SOURCE_CLOSURE_SHA256='7c6d5da3fa037a2d89d6f9db298d3570cccdbeb6a7dad238cbb36732539a6090'
LICENSE_SHA256='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
CONFIG_PATH='cosyvoice2.yaml'; CONFIG_BYTES=7330
CONFIG_SHA256='0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959'
CONFIG_GIT_BLOB_SHA1='bc19267bbfd373c9a760b7667a74349ddd487db1'
QWEN_CONFIG_PATH='CosyVoice-BlankEN/config.json'; QWEN_CONFIG_BYTES=659
QWEN_CONFIG_SHA256='168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185'
QWEN_CONFIG_GIT_BLOB_SHA1='463b055262b6c66c4629a74a4b300bfe2ed31d3c'
LLM_BYTES=2023316821; LLM_SHA256='b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608'
FLOW_BYTES=450575567; FLOW_SHA256='ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541'
WORK="${COSYVOICE2_COMPONENT_PREPARATION_WORK_DIR:-/dev/shm/vokra-cosyvoice2-component-preparation}"
UV_CACHE_DIR="${COSYVOICE2_COMPONENT_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-component-uv-cache}"
MIN_MEM_GIB=8; MIN_TMPFS_GIB=8
MIN_MEM_KIB=$((MIN_MEM_GIB * 1024 * 1024)); MIN_TMPFS_KIB=$((MIN_TMPFS_GIB * 1024 * 1024))
export UV_CACHE_DIR
log() { printf '[cosyvoice2-preparation-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
self_test() {
  local fail=0 token
  for token in "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" \
    "$LICENSE_SHA256" "$SOURCE_CLOSURE_SHA256" "$CONFIG_PATH" "$CONFIG_BYTES" "$CONFIG_SHA256" "$CONFIG_GIT_BLOB_SHA1" \
    "$QWEN_CONFIG_PATH" "$QWEN_CONFIG_BYTES" "$QWEN_CONFIG_SHA256" "$QWEN_CONFIG_GIT_BLOB_SHA1" \
    "$LLM_BYTES" "$LLM_SHA256" "$FLOW_BYTES" "$FLOW_SHA256" 'MIN_MEM_GIB=8' 'MIN_TMPFS_GIB=8' \
    'Linux x86_64 VAST' '--prepare' 'PREPARED_SAFETENSORS_READY' 'PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED' \
    'DATA_PKL_AND_AUTHENTICATED_F32_STORAGE_ONLY' 'NOT_RUN' 'NO_UPLOAD' 'atomic' 'no-replace' 'safetensors'; do
    grep -Fq -- "$token" "$PREPARER" "$0" || { log "self-test missing contract: $token"; fail=1; }
  done
  grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$0" >/dev/null && { log 'mutation command found'; fail=1; } || true
  grep -En '^[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$0" >/dev/null && { log 'raw Python command found'; fail=1; } || true
  uv run --no-project --python 3.12 python "$PREPARER" --self-test >/dev/null || { log 'preparer self-test failed'; fail=1; }
  (( fail == 0 )) && log 'self-test: OK' || return 1
}
self=0; prepare=0; component=''; component_count=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --prepare) (( prepare == 0 )) || die 'duplicate --prepare'; prepare=1; shift ;;
    --component) (( self == 0 )) || die '--self-test accepts no component'; (( component_count == 0 )) || die 'duplicate --component'; (($# >= 2)) || die 'missing component'; component="$2"; component_count=1; shift 2 ;;
    --component=*) (( self == 0 )) || die '--self-test accepts no component'; (( component_count == 0 )) || die 'duplicate --component'; component="${1#*=}"; component_count=1; shift ;;
    -h|--help) echo "usage: $0 --prepare --component llm|flow | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then (( prepare == 0 && component_count == 0 )) || die '--self-test accepts no preparation options'; self_test; exit $?; fi
(( prepare == 1 )) || die 'explicit --prepare is required'
(( component_count == 1 )) || die 'exactly one --component llm|flow is required'
case "$component" in
  llm) MODEL_PATH='llm.pt'; MODEL_BYTES="$LLM_BYTES"; MODEL_SHA256="$LLM_SHA256" ;;
  flow) MODEL_PATH='flow.pt'; MODEL_BYTES="$FLOW_BYTES"; MODEL_SHA256="$FLOW_SHA256" ;;
  *) die 'component must be llm or flow' ;;
esac
MODEL_URL="https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$MODEL_PATH?download=true"
CONFIG_URL="https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$CONFIG_PATH?download=true"
QWEN_CONFIG_URL="https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$QWEN_CONFIG_PATH?download=true"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in awk curl df find findmnt git sha256sum stat uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_MEM_KIB" ]] || die "RAM below $MIN_MEM_GIB GiB"
parent="$(dirname "$WORK")"
[[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_TMPFS_KIB" ]] || die "tmpfs free space below $MIN_TMPFS_GIB GiB"
[[ ! -e "$WORK" || -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be absent or empty'
mkdir -p "$WORK/model" "$WORK/source" "$WORK/evidence" "$WORK/prepared"; WORK="$(cd "$WORK" && pwd)"
checkpoint="$WORK/model/$MODEL_PATH"; config="$WORK/model/$CONFIG_PATH"; qwen_config=''
source="$WORK/source/CosyVoice"; evidence="$WORK/evidence"; log_file="$WORK/validation.log"
log "downloading only pinned $MODEL_PATH"
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$checkpoint" "$MODEL_URL" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$checkpoint")" == "$MODEL_BYTES" ]] || die 'checkpoint byte count mismatch'
[[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$MODEL_SHA256" ]] || die 'checkpoint SHA-256 mismatch'
log "downloading pinned config"; curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$config" "$CONFIG_URL" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$config")" == "$CONFIG_BYTES" ]] || die 'config byte count mismatch'
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$CONFIG_SHA256" && "$(git hash-object "$config")" == "$CONFIG_GIT_BLOB_SHA1" ]] || die 'config identity mismatch'
if [[ "$component" == llm ]]; then
  qwen_config="$WORK/model/$QWEN_CONFIG_PATH"; mkdir -p "$(dirname "$qwen_config")"
  curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$qwen_config" "$QWEN_CONFIG_URL" >>"$log_file" 2>&1
  [[ "$(stat -c '%s' "$qwen_config")" == "$QWEN_CONFIG_BYTES" ]] || die 'Qwen byte count mismatch'
  [[ "$(sha256sum "$qwen_config" | awk '{print $1}')" == "$QWEN_CONFIG_SHA256" && "$(git hash-object "$qwen_config")" == "$QWEN_CONFIG_GIT_BLOB_SHA1" ]] || die 'Qwen identity mismatch'
fi
git clone --no-tags --filter=blob:none "$SOURCE_URL" "$source" >>"$log_file" 2>&1
git -C "$source" checkout --detach "$SOURCE_REVISION" >>"$log_file" 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" && -z "$(git -C "$source" status --porcelain --untracked-files=all)" ]] || die 'source identity/clean checkout mismatch'
[[ "$(sha256sum "$source/LICENSE" | awk '{print $1}')" == "$LICENSE_SHA256" ]] || die 'Apache LICENSE SHA-256 mismatch'
output="$WORK/prepared/$MODEL_PATH.safetensors"; report="$evidence/manifest.json"
args=(--component "$component" --checkpoint "$checkpoint" --source "$source" --config "$config" --output "$output" --report "$report")
if [[ "$component" == llm ]]; then args+=(--qwen-config "$qwen_config"); fi
log 'preparing deterministic safetensors'; uv run --no-project --python 3.12 python "$PREPARER" "${args[@]}" >>"$log_file" 2>&1 || die 'preparation failed closed'
uv run --no-project --python 3.12 python - "$report" <<'PY' >>"$log_file" 2>&1
import json, sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if m.get("format") != "vokra-cosyvoice2-component-prepared-safetensors-v1" or m.get("status") != "PREPARED_SAFETENSORS_READY": raise SystemExit("prepared manifest mismatch")
if m.get("execution") != {"model_execution":"NOT_RUN","publication":"NO_UPLOAD","torch_import":"NOT_RUN"}: raise SystemExit("execution contract mismatch")
if m.get("component") == "flow":
    closure = m.get("official_source", {}).get("flow_source_closure", {})
    expected = {"path":"tools/parity/cosyvoice2_flow_source_closure.json","sha256":"7c6d5da3fa037a2d89d6f9db298d3570cccdbeb6a7dad238cbb36732539a6090","node_count":15,"edge_count":15,"status":"REPO_LOCAL_SOURCE_CLOSURE_COMPLETE_EXTERNAL_MATCHA_PENDING"}
    if any(closure.get(key) != value for key, value in expected.items()): raise SystemExit("Flow source closure authentication contract missing")
o=m.get("output",{})
if o.get("bytes",0)<=0 or len(o.get("sha256",""))!=64: raise SystemExit("output digest missing")
print(f"component={m.get('component')} output_bytes={o.get('bytes')} output_sha256={o.get('sha256')}")
PY
mv "$log_file" "$evidence/validation.log"
log "CosyVoice2 $component preparation complete; evidence=$evidence; publication=NO_UPLOAD"
