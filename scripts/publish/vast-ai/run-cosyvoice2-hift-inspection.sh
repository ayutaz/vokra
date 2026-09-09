#!/usr/bin/env bash
# VAST-only structural inspection of the pinned CosyVoice2 HiFT checkpoint.
# No model execution, conversion, numerical parity, publication, or upload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/cosyvoice2_hift_inspect.py"
MODEL_URL='https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/eec1ae6c79877dbd9379285cf8789c9e0879293d/hift.pt?download=true'
MODEL_BYTES=83390254
MODEL_SHA256='3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879'
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'
SOURCE_REVISION='8555549e882236e6541748b1042d95693caa82ba'
GENERATOR_SHA256='f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626'
LICENSE_SHA256='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
WORK="${COSYVOICE2_HIFT_WORK_DIR:-/dev/shm/vokra-cosyvoice2-hift-inspection}"
UV_CACHE_DIR="${COSYVOICE2_HIFT_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-hift-uv-cache}"
MIN_MEM_GIB=8
MIN_TMPFS_GIB=2
MIN_MEM_KIB=$((MIN_MEM_GIB * 1024 * 1024))
MIN_TMPFS_KIB=$((MIN_TMPFS_GIB * 1024 * 1024))
export UV_CACHE_DIR
UV_CMD=(uv run --no-project --python 3.12 python)

log() { printf '[cosyvoice2-hift-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

self_test() {
  local fail=0 token log_file_token command_token move_token
  log_file_token="log_file=\"\$WORK/validation.log\""
  command_token="\$INSPECTOR\" --checkpoint \"\$checkpoint\" --source \"\$source\" --output \"\$evidence\" >>\"\$log_file\""
  move_token="mv \"\$log_file\" \"\$evidence/validation.log\""
  for token in "$MODEL_URL" "$MODEL_SHA256" "$SOURCE_REVISION" "$GENERATOR_SHA256" "$LICENSE_SHA256" 'MIN_MEM_GIB=8' 'MIN_TMPFS_GIB=2' 'RAM below 8 GiB' 'tmpfs free space below 2 GiB' '--retry 4' '--retry-delay 5' '--retry-max-time 120' '--retry-all-errors' "$log_file_token" "$command_token" "$move_token" 'INSPECTION_ONLY' 'NOT_RUN' 'NO_UPLOAD' 'DATA_PKL_ONLY' 'torch_pickle_manifest.py' 'HiFTGenerator'; do
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
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    -h|--help) echo "usage: $0 | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then self_test; exit $?; fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in awk curl df find findmnt git sha256sum stat uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_MEM_KIB" ]] || die 'RAM below 8 GiB'
parent="$(dirname "$WORK")"
[[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_TMPFS_KIB" ]] || die 'tmpfs free space below 2 GiB'
[[ ! -e "$WORK" || -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be absent or empty'
mkdir -p "$WORK/model" "$WORK/source" "$WORK/evidence"
WORK="$(cd "$WORK" && pwd)"

checkpoint="$WORK/model/hift.pt"
source="$WORK/source/CosyVoice"
evidence="$WORK/evidence"
log_file="$WORK/validation.log"
log 'downloading only the pinned hift.pt payload'
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$checkpoint" "$MODEL_URL" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$checkpoint")" == "$MODEL_BYTES" ]] || die 'downloaded hift.pt byte count mismatch'
[[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$MODEL_SHA256" ]] || die 'downloaded hift.pt SHA-256 mismatch'

git clone --no-tags --filter=blob:none "$SOURCE_URL" "$source" >>"$log_file" 2>&1
git -C "$source" checkout --detach "$SOURCE_REVISION" >>"$log_file" 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'CosyVoice source revision mismatch'
[[ -z "$(git -C "$source" status --porcelain --untracked-files=all)" ]] || die 'source checkout is dirty'
[[ "$(sha256sum "$source/cosyvoice/hifigan/generator.py" | awk '{print $1}')" == "$GENERATOR_SHA256" ]] || die 'generator.py SHA-256 mismatch'
[[ "$(sha256sum "$source/LICENSE" | awk '{print $1}')" == "$LICENSE_SHA256" ]] || die 'Apache LICENSE SHA-256 mismatch'

set +e
"${UV_CMD[@]}" "$INSPECTOR" --checkpoint "$checkpoint" --source "$source" --output "$evidence" >>"$log_file" 2>&1
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
checkpoint = manifest.get("checkpoint", {})
if checkpoint.get("payload_reads") != "DATA_PKL_ONLY; TENSOR_STORAGE_MEMBERS_NOT_OPENED":
    raise SystemExit("tensor payload-read contract missing")
print(f"tensor_count={checkpoint.get('tensor_count')} manifest_sha256={checkpoint.get('manifest_sha256')}")
PY
mv "$log_file" "$evidence/validation.log"
log "HiFT structural inspection BLOCKED; evidence=$evidence"
exit 2
