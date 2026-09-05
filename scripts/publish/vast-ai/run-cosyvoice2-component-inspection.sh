#!/usr/bin/env bash
# VAST-only structural inspection of exactly one pinned CosyVoice2 component.
# No model execution, conversion, numerical parity, publication, or upload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/cosyvoice2_component_inspect.py"
MODEL_REPOSITORY='FunAudioLLM/CosyVoice2-0.5B'
MODEL_REVISION='eec1ae6c79877dbd9379285cf8789c9e0879293d'
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'
SOURCE_REVISION='8555549e882236e6541748b1042d95693caa82ba'
LICENSE_SHA256='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
CONFIG_PATH='cosyvoice2.yaml'
CONFIG_BYTES=7330
CONFIG_SHA256='0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959'
CONFIG_GIT_BLOB_SHA1='bc19267bbfd373c9a760b7667a74349ddd487db1'
QWEN_CONFIG_PATH='CosyVoice-BlankEN/config.json'
QWEN_CONFIG_BYTES=659
QWEN_CONFIG_SHA256='168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185'
QWEN_CONFIG_GIT_BLOB_SHA1='463b055262b6c66c4629a74a4b300bfe2ed31d3c'
LLM_BYTES=2023316821
LLM_SHA256='b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608'
FLOW_BYTES=450575567
FLOW_SHA256='ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541'
WORK="${COSYVOICE2_COMPONENT_WORK_DIR:-/dev/shm/vokra-cosyvoice2-component-inspection}"
UV_CACHE_DIR="${COSYVOICE2_COMPONENT_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-component-uv-cache}"
MIN_MEM_GIB=8
MIN_TMPFS_GIB=4
MIN_MEM_KIB=$((MIN_MEM_GIB * 1024 * 1024))
MIN_TMPFS_KIB=$((MIN_TMPFS_GIB * 1024 * 1024))
export UV_CACHE_DIR
UV_CMD=(uv run --no-project --python 3.12 python)

log() { printf '[cosyvoice2-component-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

self_test() {
  local fail=0 token
  for token in \
    "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" \
    "$LICENSE_SHA256" "$CONFIG_PATH" "$CONFIG_BYTES" "$CONFIG_SHA256" "$CONFIG_GIT_BLOB_SHA1" \
    "$QWEN_CONFIG_PATH" "$QWEN_CONFIG_BYTES" "$QWEN_CONFIG_SHA256" "$QWEN_CONFIG_GIT_BLOB_SHA1" \
    "$LLM_BYTES" "$LLM_SHA256" "$FLOW_BYTES" "$FLOW_SHA256" \
    'MIN_MEM_GIB=8' 'MIN_TMPFS_GIB=4' 'Linux x86_64 VAST' \
    '--component' '--config' '--qwen-config' 'llm|flow' 'Qwen' 'ACQUIRED_AND_HASH_VERIFIED' 'INSPECTION_ONLY' 'NOT_RUN' 'NO_UPLOAD' \
    'DATA_PKL_ONLY' 'torch_pickle_manifest.py' 'TENSOR_STORAGE_MEMBERS_NOT_OPENED' \
    '--retry 4' '--retry-delay 5' '--retry-max-time 120' '--retry-all-errors'; do
    grep -Fq -- "$token" "$INSPECTOR" "$0" || { log "self-test missing contract: $token"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$0" >/dev/null; then
    log 'self-test found mutation/conversion/Cargo command'; fail=1
  fi
  if grep -En '^[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$0" >/dev/null; then
    log 'self-test found raw Python/pip command'; fail=1
  fi
  if ! "${UV_CMD[@]}" "$INSPECTOR" --self-test >/dev/null; then
    log 'inspector self-test failed'; fail=1
  fi
  if (( fail == 0 )); then log 'self-test: OK'; else return 1; fi
}

self=0
component=''
component_count=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --component)
      (( self == 0 )) || die '--self-test accepts no component'
      (( component_count == 0 )) || die 'exactly one --component is required'
      (($# >= 2)) || die '--component requires llm or flow'
      component="$2"; component_count=1; shift 2 ;;
    --component=*)
      (( self == 0 )) || die '--self-test accepts no component'
      (( component_count == 0 )) || die 'exactly one --component is required'
      component="${1#*=}"; component_count=1; shift ;;
    -h|--help) echo "usage: $0 --component llm|flow | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  (( component_count == 0 )) || die '--self-test accepts no component'
  self_test; exit $?
fi
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
mkdir -p "$WORK/model" "$WORK/source" "$WORK/evidence"
WORK="$(cd "$WORK" && pwd)"

checkpoint="$WORK/model/$MODEL_PATH"
config="$WORK/model/$CONFIG_PATH"
qwen_config=''
source="$WORK/source/CosyVoice"
evidence="$WORK/evidence"
log_file="$WORK/validation.log"
log "downloading only the pinned $MODEL_PATH payload"
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$checkpoint" "$MODEL_URL" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$checkpoint")" == "$MODEL_BYTES" ]] || die "downloaded $MODEL_PATH byte count mismatch"
[[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$MODEL_SHA256" ]] || die "downloaded $MODEL_PATH SHA-256 mismatch"
log "downloading only the pinned $CONFIG_PATH sidecar"
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$config" "$CONFIG_URL" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$config")" == "$CONFIG_BYTES" ]] || die 'downloaded cosyvoice2.yaml byte count mismatch'
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || die 'downloaded cosyvoice2.yaml SHA-256 mismatch'
if [[ "$component" == llm ]]; then
  qwen_config="$WORK/model/$QWEN_CONFIG_PATH"
  mkdir -p "$(dirname "$qwen_config")"
  log "downloading only the pinned $QWEN_CONFIG_PATH sidecar for llm"
  curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$qwen_config" "$QWEN_CONFIG_URL" >>"$log_file" 2>&1
  [[ "$(stat -c '%s' "$qwen_config")" == "$QWEN_CONFIG_BYTES" ]] || die 'downloaded Qwen config.json byte count mismatch'
  [[ "$(sha256sum "$qwen_config" | awk '{print $1}')" == "$QWEN_CONFIG_SHA256" ]] || die 'downloaded Qwen config.json SHA-256 mismatch'
  [[ "$(git hash-object "$qwen_config")" == "$QWEN_CONFIG_GIT_BLOB_SHA1" ]] || die 'downloaded Qwen config.json Git blob mismatch'
fi

git clone --no-tags --filter=blob:none "$SOURCE_URL" "$source" >>"$log_file" 2>&1
git -C "$source" checkout --detach "$SOURCE_REVISION" >>"$log_file" 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'CosyVoice source revision mismatch'
[[ -z "$(git -C "$source" status --porcelain --untracked-files=all)" ]] || die 'source checkout is dirty'
[[ "$(sha256sum "$source/LICENSE" | awk '{print $1}')" == "$LICENSE_SHA256" ]] || die 'Apache LICENSE SHA-256 mismatch'

inspector_args=(--component "$component" --checkpoint "$checkpoint" --source "$source" --config "$config" --output "$evidence")
if [[ "$component" == llm ]]; then
  inspector_args+=(--qwen-config "$qwen_config")
fi
set +e
"${UV_CMD[@]}" "$INSPECTOR" "${inspector_args[@]}" >>"$log_file" 2>&1
rc=$?
set -e
[[ "$rc" == 2 ]] || die "inspector returned unexpected status $rc"
"${UV_CMD[@]}" - "$evidence/manifest.json" <<'PY' >>"$log_file" 2>&1
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "status": "BLOCKED",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "evidence_stage": "INSPECTION_ONLY",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "cpu_status": "NOT_RUN",
    "metal_status": "NOT_RUN",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"manifest {key} mismatch: {manifest.get(key)!r}")
if manifest.get("component") not in {"llm", "flow"}:
    raise SystemExit("component identity missing")
checkpoint = manifest.get("checkpoint", {})
if checkpoint.get("payload_reads") != "DATA_PKL_ONLY; TENSOR_STORAGE_MEMBERS_NOT_OPENED":
    raise SystemExit("tensor payload-read contract missing")
config = manifest.get("model_config", {})
if config.get("verification") != "ACQUIRED_AND_HASH_VERIFIED":
    raise SystemExit("config acquisition contract missing")
qwen = manifest.get("qwen_config")
if manifest["component"] == "llm":
    if qwen != {
        "path": "CosyVoice-BlankEN/config.json",
        "bytes": 659,
        "sha256": "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185",
        "git_blob_sha1": "463b055262b6c66c4629a74a4b300bfe2ed31d3c",
        "verification": "ACQUIRED_AND_HASH_VERIFIED",
    }:
        raise SystemExit("LLM Qwen config authentication contract missing")
elif qwen is not None:
    raise SystemExit("flow must not claim a Qwen config")
print(f"component={manifest['component']} tensor_count={checkpoint.get('tensor_count')} manifest_sha256={checkpoint.get('manifest_sha256')}")
PY
mv "$log_file" "$evidence/validation.log"
log "CosyVoice2 $component structural inspection BLOCKED; evidence=$evidence"
exit 2
