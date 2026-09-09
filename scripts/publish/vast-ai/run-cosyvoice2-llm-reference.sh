#!/usr/bin/env bash
# VAST-only official CosyVoice2 LLM reference. Never upload.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DUMPER="$ROOT/tools/parity/cosyvoice2_llm_dump_reference.py"
PROJECT="$ROOT/tools/parity/cosyvoice2_llm_reference"
MODEL_REPOSITORY='FunAudioLLM/CosyVoice2-0.5B'; MODEL_REVISION='eec1ae6c79877dbd9379285cf8789c9e0879293d'
MODEL_BYTES=2023316821; MODEL_SHA256='b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608'
QWEN_PATH='CosyVoice-BlankEN/config.json'; QWEN_BYTES=659
QWEN_SHA256='168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185'; QWEN_BLOB='463b055262b6c66c4629a74a4b300bfe2ed31d3c'
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'; SOURCE_REVISION='8555549e882236e6541748b1042d95693caa82ba'
LICENSE_SHA256='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
WORK="${COSYVOICE2_LLM_REFERENCE_WORK_DIR:-/dev/shm/vokra-cosyvoice2-llm-reference}"
UV_CACHE_DIR="${COSYVOICE2_LLM_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-llm-uv-cache}"; export UV_CACHE_DIR
log() { printf '[cosyvoice2-llm-reference-vast] %s\n' "$*" >&2; }; die() { log "ERROR: $*"; exit 2; }
self_test() {
  local fail=0 token
  for token in "$MODEL_REPOSITORY" "$MODEL_REVISION" "$MODEL_BYTES" "$MODEL_SHA256" "$QWEN_PATH" "$QWEN_BYTES" "$QWEN_SHA256" "$QWEN_BLOB" "$SOURCE_URL" "$SOURCE_REVISION" "$LICENSE_SHA256" 'REFERENCE_READY' 'NO_UPLOAD' 'Qwen2ForCausalLM' 'eager' 'torch.load' 'weights_only=True' 'VAST Linux x86_64' '32*1024*1024' 'reference output directory'; do
    grep -Fq -- "$token" "$DUMPER" "$PROJECT/preflight_gate.py" "$0" || { log "missing contract: $token"; fail=1; }
  done
  grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$0" >/dev/null && { log 'mutation command found'; fail=1; } || true
  uv run --no-project --offline --python 3.12 python "$PROJECT/preflight_gate.py" --self-test >/dev/null || fail=1
  uv run --no-project --offline --python 3.12 python "$DUMPER" --self-test >/dev/null || fail=1
  (( fail == 0 )) && log 'self-test: OK' || return 1
}
self=0; reference=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --reference) (( reference == 0 )) || die 'duplicate --reference'; reference=1; shift ;;
    -h|--help) echo "usage: $0 --reference | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then (( reference == 0 )) || die '--self-test accepts no mode'; self_test; exit $?; fi
(( reference == 1 )) || die 'explicit --reference is required'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST Linux x86_64 required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in awk curl df find findmnt git sha256sum stat uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
uv run --no-project --offline --python 3.12 python "$PROJECT/preflight_gate.py" --project "$PROJECT/pyproject.toml" --lock "$PROJECT/uv.lock" --license-manifest "$PROJECT/license_gate_manifest.json" >/dev/null || die 'license/lock preflight blocked before model acquisition'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((32*1024*1024)) ]] || die 'RAM below 32 GiB'
parent="$(dirname "$WORK")"; [[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge $((8*1024*1024)) ]] || die 'tmpfs free space below 8 GiB'
[[ ! -e "$WORK" || -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be absent or empty'
mkdir -p "$(dirname "$WORK/model/$QWEN_PATH")" "$WORK/source" "$WORK/logs"; WORK="$(cd "$WORK" && pwd)"
checkpoint="$WORK/model/llm.pt"; qwen="$WORK/model/$QWEN_PATH"; source="$WORK/source/CosyVoice"; output="$WORK/reference"; log_file="$WORK/logs/validation.log"
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$checkpoint" "https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/llm.pt?download=true" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$checkpoint")" == "$MODEL_BYTES" && "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$MODEL_SHA256" ]] || die 'llm.pt identity mismatch'
curl --fail --location --proto '=https' --tlsv1.2 --retry 4 --retry-delay 5 --retry-max-time 120 --retry-all-errors --output "$qwen" "https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$QWEN_PATH?download=true" >>"$log_file" 2>&1
[[ "$(stat -c '%s' "$qwen")" == "$QWEN_BYTES" && "$(sha256sum "$qwen" | awk '{print $1}')" == "$QWEN_SHA256" && "$(git hash-object "$qwen")" == "$QWEN_BLOB" ]] || die 'Qwen config identity mismatch'
git clone --no-tags --filter=blob:none "$SOURCE_URL" "$source" >>"$log_file" 2>&1; git -C "$source" checkout --detach "$SOURCE_REVISION" >>"$log_file" 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" && -z "$(git -C "$source" status --porcelain --untracked-files=all)" && "$(sha256sum "$source/LICENSE" | awk '{print $1}')" == "$LICENSE_SHA256" ]] || die 'source identity mismatch'
uv run --frozen --project "$PROJECT" python "$DUMPER" --checkpoint "$checkpoint" --qwen-config "$qwen" --source "$source" --license-manifest "$PROJECT/license_gate_manifest.json" --output "$output" >>"$log_file" 2>&1 || die 'reference execution failed closed'
[[ -f "$output/manifest.json" ]] || die 'reference completion manifest missing'
mv "$log_file" "$output/validation.log"
log "CosyVoice2 LLM reference complete; output=$output; publication=NO_UPLOAD"
