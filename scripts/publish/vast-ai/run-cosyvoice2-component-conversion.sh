#!/usr/bin/env bash
# VAST-only conversion and strict bind gate for one prepared CosyVoice2 component.
# No model forward and no publication are performed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORK="${COSYVOICE2_COMPONENT_WORK_DIR:-/dev/shm/vokra-cosyvoice2-component-conversion}"
log() { printf '[cosyvoice2-component-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
self_test() {
  local token
  for token in '--convert' '--component llm|flow' 'PREPARED_SAFETENSORS_READY' 'NOT_RUN' 'NO_UPLOAD' 'INSPECTION_ONLY' '295' '1121' 'Linux x86_64' 'clean checkout' 'uv run --no-project --python 3.12 python -c'; do grep -Fq -- "$token" "$0" || die "self-test missing contract: $token"; done
  if grep -En '(^|[[:space:]])(curl|wget|git[[:space:]]+clone|git[[:space:]]+push|upload\.sh|publish-one\.sh)([[:space:]]|$)' "$0" >/dev/null; then die 'network/publish command found in runner'; fi
  bash -n "$0"
  log 'self-test: OK (no cargo, model, or network command executed)'
}
self=0; convert=0; component=''
while (($#)); do
  case "$1" in
    --self-test) ((self == 0)) || die 'duplicate --self-test'; self=1; shift ;;
    --convert) ((convert == 0)) || die 'duplicate --convert'; convert=1; shift ;;
    --component) [[ $# -ge 2 ]] || die 'missing component'; [[ -z "$component" ]] || die 'duplicate component'; component="$2"; shift 2 ;;
    --component=*) [[ -z "$component" ]] || die 'duplicate component'; component="${1#*=}"; shift ;;
    -h|--help) echo "usage: $0 --convert --component llm|flow | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if ((self)); then ((convert == 0)) && [[ -z "$component" ]] || die '--self-test accepts no conversion options'; self_test; exit 0; fi
((convert == 1)) || die 'explicit --convert is required'
[[ "$component" == llm || "$component" == flow ]] || die '--component must be llm or flow'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in awk cargo df findmnt git sha256sum stat uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((8 * 1024 * 1024)) ]] || die 'RAM below 8 GiB'
parent="$(dirname "$WORK")"; [[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge $((8 * 1024 * 1024)) ]] || die 'tmpfs below 8 GiB'
required_env() { local name="$1" value="${!1-}"; [[ -n "$value" ]] || die "$name is required"; printf '%s' "$value"; }
require_file() { local path="$1" label="$2"; [[ "$path" = /* ]] || die "$label must be absolute"; [[ -f "$path" && ! -L "$path" ]] || die "$label must be regular non-symlink: $path"; }
require_absent() { local path="$1" label="$2"; [[ "$path" = /* ]] || die "$label must be absolute"; [[ ! -e "$path" && ! -L "$path" ]] || die "$label must be absent: $path"; }
if [[ "$component" == llm ]]; then prepared="$(required_env VOKRA_COSYVOICE2_LLM_PREPARED)"; manifest="$(required_env VOKRA_COSYVOICE2_LLM_PREPARED_MANIFEST)"; config="$(required_env VOKRA_COSYVOICE2_LLM_CONFIG)"; qwen_config="$(required_env VOKRA_COSYVOICE2_LLM_QWEN_CONFIG)"; output="$(required_env VOKRA_COSYVOICE2_LLM_OUTPUT)"; license="$(required_env VOKRA_COSYVOICE2_LLM_LICENSE)"; test_filter='models::cosyvoice2::tests::vast_real_prepared_llm_conversion'; bind_filter='cosyvoice2::llm_component::tests::vast_real_llm_gguf_binds_exact_component'; count=295; else prepared="$(required_env VOKRA_COSYVOICE2_FLOW_PREPARED)"; manifest="$(required_env VOKRA_COSYVOICE2_FLOW_PREPARED_MANIFEST)"; config="$(required_env VOKRA_COSYVOICE2_FLOW_CONFIG)"; output="$(required_env VOKRA_COSYVOICE2_FLOW_OUTPUT)"; license="$(required_env VOKRA_COSYVOICE2_FLOW_LICENSE)"; qwen_config=''; test_filter='models::cosyvoice2_flow::tests::vast_real_prepared_flow_conversion'; bind_filter='cosyvoice2::flow_weights::tests::vast_real_flow_gguf_binds_exact_component'; count=1121; fi
require_file "$prepared" 'prepared safetensors'; require_file "$manifest" 'prepared manifest'; require_file "$config" 'config sidecar'; [[ "${license,,}" == apache-2.0 ]] || die 'Apache-2.0 license attestation required'; [[ "$component" != llm ]] || require_file "$qwen_config" 'Qwen config sidecar'; require_absent "$output" 'GGUF output'
uv run --no-project --python 3.12 python -c 'import hashlib,json,sys; from pathlib import Path; m,c,p=map(Path,sys.argv[1:]); d=json.loads(m.read_text(encoding="utf-8")); r=d.get("output"); assert d.get("format")=="vokra-cosyvoice2-component-prepared-safetensors-v1" and d.get("status")=="PREPARED_SAFETENSORS_READY" and d.get("component")==c.name, "prepared manifest identity/status mismatch"; assert isinstance(r,dict) and r.get("path")==str(p) and r.get("bytes")==p.stat().st_size and r.get("sha256")==hashlib.sha256(p.read_bytes()).hexdigest(), "prepared output path/bytes/SHA mismatch"; assert d.get("execution")=={"model_execution":"NOT_RUN","torch_import":"NOT_RUN","publication":"NO_UPLOAD"}, "execution/publication contract mismatch"; print("prepared manifest verified")' "$manifest" "$component" "$prepared"
mkdir -p "$WORK"; export VOKRA_COSYVOICE2_LLM_LICENSE="$license" VOKRA_COSYVOICE2_FLOW_LICENSE="$license"
if [[ "$component" == llm ]]; then export VOKRA_COSYVOICE2_LLM_PREPARED="$prepared" VOKRA_COSYVOICE2_LLM_PREPARED_MANIFEST="$manifest" VOKRA_COSYVOICE2_LLM_CONFIG="$config" VOKRA_COSYVOICE2_LLM_QWEN_CONFIG="$qwen_config" VOKRA_COSYVOICE2_LLM_OUTPUT="$output"; else export VOKRA_COSYVOICE2_FLOW_PREPARED="$prepared" VOKRA_COSYVOICE2_FLOW_PREPARED_MANIFEST="$manifest" VOKRA_COSYVOICE2_FLOW_CONFIG="$config" VOKRA_COSYVOICE2_FLOW_OUTPUT="$output"; fi
log "converting prepared $component component"; CARGO_BUILD_JOBS=1 cargo test -p vokra-convert --lib "$test_filter" -- --ignored --exact
[[ -f "$output" && ! -L "$output" ]] || die 'converter did not emit regular GGUF'; if [[ "$component" == llm ]]; then export VOKRA_COSYVOICE2_LLM_GGUF="$output"; else export VOKRA_COSYVOICE2_FLOW_GGUF="$output"; fi
log "binding exact $component component"; CARGO_BUILD_JOBS=1 cargo test -p vokra-models --lib "$bind_filter" -- --ignored --exact
evidence="$output.evidence.json"; require_absent "$evidence" 'evidence record'; uv run --no-project --python 3.12 python -c 'import hashlib,json,sys; from pathlib import Path; o,c,g,n=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),int(sys.argv[4]); b={"format":"vokra-cosyvoice2-component-conversion-v1","component":c,"gguf":{"path":str(g),"bytes":g.stat().st_size,"sha256":hashlib.sha256(g.read_bytes()).hexdigest(),"tensor_count":n},"bind_status":"STRICT_BIND_PASS","model_execution":"NOT_RUN","publication":"NO_UPLOAD"}; f=o.open("x",encoding="utf-8"); json.dump(b,f,indent=2,sort_keys=True); f.write("\\n"); f.close(); print(json.dumps(b,sort_keys=True))' "$evidence" "$component" "$output" "$count"
log "strict bind complete; evidence=$evidence; NO_UPLOAD; MODEL_EXECUTION_NOT_RUN"
