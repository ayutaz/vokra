#!/usr/bin/env bash
# VAST-only CosyVoice2 LLM component parity against official Transformers data.
# This runner never acquires reference/model artifacts or publishes a model;
# the exact Cargo parity test intentionally executes Vokra's LLM forward.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORK="${COSYVOICE2_LLM_PARITY_WORK_DIR:-/dev/shm/vokra-cosyvoice2-llm-component-parity}"
TEST_NAME='parity_cosyvoice2_llm_component_real_transformers'

log() { printf '[cosyvoice2-llm-parity-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
self_test() {
  local token
  for token in '--parity' 'Linux x86_64' '32 GiB' 'REFERENCE_READY' 'NO_UPLOAD' 'eager' 'F32' 'MEASURED_PASS' 'atomic' 'no-replace' 'COSYVOICE2_LLM_PARITY_RESULT' 'vokra_model_execution' "$TEST_NAME"; do
    grep -Fq -- "$token" "$0" || die "self-test missing contract: $token"
  done
  if grep -En '(^|[[:space:]])(curl|wget|git[[:space:]]+clone|git[[:space:]]+push|upload\.sh|publish-one\.sh)([[:space:]]|$)' "$0" >/dev/null; then die 'network/publish command found in self-test runner'; fi
  bash -n "$0"
  log 'self-test: OK (no model, Cargo, or network command executed)'
}
self=0; parity=0
while (($#)); do
  case "$1" in
    --self-test) ((self == 0)) || die 'duplicate --self-test'; self=1; shift ;;
    --parity) ((parity == 0)) || die 'duplicate --parity'; parity=1; shift ;;
    -h|--help) echo "usage: $0 --parity | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if ((self)); then ((parity == 0)) || die '--self-test accepts no mode'; self_test; exit 0; fi
((parity == 1)) || die 'explicit --parity is required'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in cargo df findmnt git sha256sum stat uv awk; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((32 * 1024 * 1024)) ]] || die 'RAM below 32 GiB'
parent="$(dirname "$WORK")"; [[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'parity work parent must be tmpfs'
require_file() { local path="$1" label="$2" ancestor; [[ "$path" = /* ]] || die "$label must be absolute"; ancestor="$path"; while [[ "$ancestor" != / ]]; do ancestor="$(dirname "$ancestor")"; [[ ! -L "$ancestor" ]] || die "$label has a symlinked ancestor: $ancestor"; done; [[ -f "$path" && ! -L "$path" ]] || die "$label must be regular non-symlink: $path"; }
require_dir() { local path="$1" label="$2" ancestor; [[ "$path" = /* ]] || die "$label must be absolute"; ancestor="$path"; while [[ "$ancestor" != / ]]; do ancestor="$(dirname "$ancestor")"; [[ ! -L "$ancestor" ]] || die "$label has a symlinked ancestor: $ancestor"; done; [[ -d "$path" && ! -L "$path" ]] || die "$label must be a real directory: $path"; }
required_env() { local name="$1" value="${!1-}"; [[ -n "$value" ]] || die "$name is required"; printf '%s' "$value"; }
gguf="$(required_env VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF)"; reference="$(required_env VOKRA_COSYVOICE2_LLM_REFERENCE)"; evidence="$(required_env VOKRA_COSYVOICE2_LLM_PARITY_EVIDENCE)"
require_file "$gguf" 'LLM component GGUF'; require_dir "$reference" 'official reference directory'; [[ "$evidence" = /* ]] || die 'parity evidence must be absolute'; require_dir "$(dirname "$evidence")" 'parity evidence parent'; [[ ! -e "$evidence" && ! -L "$evidence" ]] || die 'parity evidence must be absent'
for name in manifest.json token-ids.json true_hf_logits.npy diagnostics.json; do require_file "$reference/$name" "reference $name"; done
uv run --no-project --python 3.12 python -c 'import hashlib,json,sys; from pathlib import Path; m=Path(sys.argv[1]); d=json.loads(m.read_text(encoding="utf-8")); assert d.get("format")=="vokra-cosyvoice2-llm-reference-v1" and d.get("status")=="REFERENCE_READY" and d.get("component")=="llm", "reference status/format mismatch"; assert d["model"]["repository"]=="FunAudioLLM/CosyVoice2-0.5B" and d["model"]["revision"]=="eec1ae6c79877dbd9379285cf8789c9e0879293d" and d["model"]["path"]=="llm.pt" and d["model"]["bytes"]==2023316821 and d["model"]["sha256"]=="b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608", "model identity mismatch"; assert d["execution"]["implementation"]=="transformers.Qwen2ForCausalLM" and d["execution"]["attention"]=="eager" and d["execution"]["dtype"]=="F32" and d["execution"]["model_execution"]=="RUN" and d["execution"]["threads"]==1 and d["execution"]["deterministic_algorithms"] is True and d["execution"]["publication"]=="NO_UPLOAD", "reference execution/publication mismatch"; assert d["state"]["tensor_count"]==295 and d["state"]["manifest_sha256"]=="07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2", "reference state mismatch"; [(lambda p,a: (_ for _ in ()).throw(SystemExit("reference artifact digest mismatch")) if p.stat().st_size!=a["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest()!=a["sha256"] else None)(m.parent/n,d["artifacts"][n]) for n in ("true_hf_logits.npy","token-ids.json","diagnostics.json")]; print("reference manifest verified")' "$reference/manifest.json"
mkdir -p "$WORK"; log_file="$WORK/parity.log"; [[ ! -e "$log_file" ]] || die 'parity log must be absent'
export VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF="$gguf" VOKRA_COSYVOICE2_LLM_REFERENCE="$reference"
log 'running exact official-reference parity test'; CARGO_BUILD_JOBS=1 cargo test -p vokra-models --test parity_cosyvoice2_llm_component_real "$TEST_NAME" -- --ignored --exact --nocapture >"$log_file" 2>&1 || { tail -n 80 "$log_file" >&2; die 'CosyVoice2 LLM parity failed'; }
marker_count="$(grep -c '^COSYVOICE2_LLM_PARITY_RESULT ' "$log_file" || true)"; [[ "$marker_count" == 1 ]] || die "expected exactly one parity result marker, got $marker_count"
result="$(grep '^COSYVOICE2_LLM_PARITY_RESULT ' "$log_file")"
read -r max_abs mean_abs argmax_matches argmax_total atol <<<"$(uv run --no-project --python 3.12 python -c 'import math,re,sys; line=sys.argv[1]; fields=dict(re.findall(r"(max_abs_delta|mean_abs_delta|argmax_matches|argmax_total|atol)=([^ ]+)",line)); required=("max_abs_delta","mean_abs_delta","argmax_matches","argmax_total","atol"); assert line.startswith("COSYVOICE2_LLM_PARITY_RESULT ") and all(k in fields for k in required), "malformed parity marker"; max_abs=float(fields["max_abs_delta"]); mean_abs=float(fields["mean_abs_delta"]); argmax_matches=int(fields["argmax_matches"]); argmax_total=int(fields["argmax_total"]); atol=float(fields["atol"]); assert all(math.isfinite(value) and value >= 0 for value in (max_abs,mean_abs,atol)) and max_abs <= 3e-4 and atol == 3e-4 and argmax_matches == argmax_total == 11, "parity marker failed gate"; print(*(fields[k] for k in required))' "$result")"
export COSYVOICE2_PARITY_MAX_ABS="$max_abs" COSYVOICE2_PARITY_MEAN_ABS="$mean_abs" COSYVOICE2_PARITY_ARGMAX_MATCHES="$argmax_matches" COSYVOICE2_PARITY_ARGMAX_TOTAL="$argmax_total" COSYVOICE2_PARITY_ATOL="$atol"
gguf_bytes="$(stat -c '%s' "$gguf")"; gguf_sha="$(sha256sum "$gguf" | awk '{print $1}')"; log_sha="$(sha256sum "$log_file" | awk '{print $1}')"; manifest_sha="$(sha256sum "$reference/manifest.json" | awk '{print $1}')"
uv run --no-project --python 3.12 python -c 'import json,os,sys; from pathlib import Path; out,gguf,log,manifest,gb,gs,ls,ms=map(Path,sys.argv[1:9]); body={"format":"vokra-cosyvoice2-llm-component-parity-v1","status":"MEASURED_PASS","component":"llm","gguf":{"path":str(gguf),"bytes":int(gb.name),"sha256":gs.name},"reference":{"path":str(manifest.parent),"manifest_sha256":ms.name},"log":{"path":str(log),"sha256":ls.name},"measured":{"max_abs_delta":float(os.environ["COSYVOICE2_PARITY_MAX_ABS"]),"mean_abs_delta":float(os.environ["COSYVOICE2_PARITY_MEAN_ABS"]),"argmax_matches":int(os.environ["COSYVOICE2_PARITY_ARGMAX_MATCHES"]),"argmax_total":int(os.environ["COSYVOICE2_PARITY_ARGMAX_TOTAL"]),"atol":float(os.environ["COSYVOICE2_PARITY_ATOL"])},"reference_acquisition":"NOT_RUN","vokra_model_execution":"RUN","publication":"NO_UPLOAD"}; temp=out.parent/("."+out.name+"."+str(os.getpid())+".tmp"); fd=None
try:
 fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644); data=(json.dumps(body,indent=2,sort_keys=True)+"\\n").encode()
 while data:
  written=os.write(fd,data)
  if written <= 0: raise OSError("short write while writing parity evidence")
  data=data[written:]
 os.fsync(fd); os.close(fd); fd=None; os.link(temp,out)
except BaseException:
 if fd is not None: os.close(fd)
 temp.unlink(missing_ok=True)
 raise
else:
 temp.unlink()
print(json.dumps(body,sort_keys=True))' "$evidence" "$gguf" "$log_file" "$reference/manifest.json" "$gguf_bytes" "$gguf_sha" "$log_sha" "$manifest_sha"
log "MEASURED_PASS recorded atomically at $evidence; NO_UPLOAD; VOKRA_MODEL_EXECUTION_RUN; REFERENCE_ACQUISITION_NOT_RUN"
