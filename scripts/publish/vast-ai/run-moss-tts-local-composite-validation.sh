#!/usr/bin/env bash
# VAST-only staging for the MOSS-TTS Local + Audio-Tokenizer-v2 composite.
# This script deliberately never publishes or uploads an artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
LOCAL_PROJECT="$VOKRA_ROOT/tools/parity/moss_tts_local"
LOCAL_GATE="$LOCAL_PROJECT/preflight_gate.py"
LOCAL_MANIFEST="$LOCAL_PROJECT/license_gate_manifest.json"
V2_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_v2"
V2_GATE="$V2_PROJECT/license_gate.py"
V2_MANIFEST="$V2_PROJECT/license_gate_manifest.json"
PARITY_ROOT="$VOKRA_ROOT/tools/parity"
REFERENCE_DUMPER="$VOKRA_ROOT/tools/parity/moss_tts_local_dump_reference.py"
CODEC_REFERENCE_DUMPER="$PARITY_ROOT/moss_audio_tokenizer_dump_reference.py"
LOCAL_REPO="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
LOCAL_REVISION="be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
CODEC_REPO="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
CODEC_REVISION="f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
CODEC_MANIFEST="a83915cffe78cee7f031e18ac3de1bbd64e93b3e4af843ff28d531ccf81748c6"
MIN_MEM_KIB=120000000
MIN_DISK_KIB=180000000

log() { printf '[moss-tts-local-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
require_prompt_rows() {
  local prompt_rows="$1"
  local expected_sha="${MOSS_TTS_LOCAL_PROMPT_ROWS_SHA256:-}"
  [[ -n "$prompt_rows" && -f "$prompt_rows" && ! -L "$prompt_rows" ]] \
    || { die "MOSS_TTS_LOCAL_PROMPT_ROWS must name a regular non-symlink [rows,13] prompt"; return 2; }
  [[ -n "$expected_sha" && "$expected_sha" =~ ^[0-9a-f]{64}$ ]] \
    || { die "MOSS_TTS_LOCAL_PROMPT_ROWS_SHA256 must be caller-supplied lowercase SHA-256"; return 2; }
  local bytes; bytes="$(wc -c < "$prompt_rows" | tr -d '[:space:]')"
  [[ "$bytes" =~ ^[0-9]+$ && "$bytes" -gt 0 && $((bytes % 52)) -eq 0 ]] \
    || { die "prompt rows must be non-empty [rows,13] u32le"; return 2; }
  [[ "$(sha256sum "$prompt_rows" | awk '{print $1}')" == "$expected_sha" ]] \
    || { die "prompt row SHA-256 mismatch"; return 2; }
}
require_cpu_evidence() {
  local log_path="$1" test_name='moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official' test_count named_count result_count total_result rows_count codes_count pcm_count composite_count rows_mentions codes_mentions pcm_mentions metal_mentions
  test_count="$(grep -Ec '^test .* \.\.\.' "$log_path" || true)"
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result="$(grep -Ec '^test result:' "$log_path" || true)"
  rows_count="$(grep -Ec '^MOSS_TTS_LOCAL_ROWS_MEASURED backend=cpu exact=true differing_values=0$' "$log_path" || true)"
  codes_count="$(grep -Ec '^MOSS_TTS_LOCAL_CODES_MEASURED backend=cpu exact=true differing_values=0$' "$log_path" || true)"
  pcm_count="$(grep -Ec '^MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu samples=[0-9]+ channels=[0-9]+ rms=[0-9eE.+-]+ peak=[0-9eE.+-]+$' "$log_path" || true)"
  rows_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_ROWS_MEASURED' "$log_path" || true)"
  codes_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_CODES_MEASURED' "$log_path" || true)"
  pcm_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_PCM_MEASURED' "$log_path" || true)"
  metal_mentions="$(grep -Ec 'MOSS_TTS_LOCAL_METAL_(ROWS|CODES|PCM)_MEASURED' "$log_path" || true)"
  composite_count="$(grep -Ec '^COMPOSITE_PCM_NOT_RUN reason=official_v2_pcm_sidecar_not_supplied$' "$log_path" || true)"
  [[ "$test_count" == 1 && "$named_count" == 1 && "$result_count" == 1 && "$total_result" == 1 && "$rows_count" == 1 && "$codes_count" == 1 && "$pcm_count" == 1 && "$rows_mentions" == 1 && "$codes_mentions" == 1 && "$pcm_mentions" == 1 && "$metal_mentions" == 0 && "$composite_count" == 1 ]] || { die 'CPU Cargo/sentinel evidence is not an exact singleton set'; return 2; }
  ! grep -Eq '^MOSS_TTS_LOCAL_(ROWS|CODES|PCM)_MEASURED .*FAIL$' "$log_path" || { die 'CPU measurement FAIL marker present'; return 2; }
}
require_v2_reference() {
  local reference="$1"
  [[ -f "$reference" && ! -L "$reference" ]] || { die 'v2 reference is not a regular file'; return 2; }
  [[ "$(wc -l < "$reference" | tr -d '[:space:]')" == 22 ]] || { die 'v2 reference has unexpected row count'; return 2; }
  [[ "$(grep -Fxc "source,v2,$CODEC_REPO,$CODEC_REVISION" "$reference" || true)" == 1 ]] || { die 'v2 source identity is not a singleton'; return 2; }
  [[ "$(grep -Ec '^runtime,torch-[^,]+,transformers-[^,]+$' "$reference" || true)" == 1 ]] || { die 'v2 runtime row is not a singleton'; return 2; }
  [[ "$(grep -Fxc 'environment,device,cuda' "$reference" || true)" == 1 ]] || { die 'v2 reference is not CUDA'; return 2; }
  [[ "$(grep -Ec '^environment,cpu,[^,]+,machine-[^,]+,logical-[0-9]+,torch-capability-.+$' "$reference" || true)" == 1 ]] || { die 'v2 CPU environment row is malformed'; return 2; }
  [[ "$(grep -Fxc 'contract,1,12,1024,48000,2,3840' "$reference" || true)" == 1 ]] || { die 'v2 contract drifted'; return 2; }
  [[ "$(grep -Ec '^codes(,[0-9]+){12}$' "$reference" || true)" == 1 ]] || { die 'v2 code packet is malformed'; return 2; }
  for row in model config; do
    local expected; [[ "$row" == model ]] && expected='7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9' || expected='f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529'
    [[ "$(awk -F, -v e="$expected" -v r="$row" '$1=="source_file"&&$2==r&&NF==4&&$3~/transformers_modules/&&$4==e{n++}END{print n+0}' "$reference")" == 1 ]] || { die "v2 source row malformed: $row"; return 2; }
  done
  for label in decoder_0 decoder_1 decoder_2 decoder_3 decoder_4 decoder_5 decoder_6 decoder_7 decoder_8 decoder_9 decoder_10 decoder_11 quantizer audio; do
    [[ "$(awk -F, -v l="$label" '$1=="tensor"&&$2==l{n++}END{print n+0}' "$reference")" == 1 ]] || { die "v2 tensor row missing/duplicate: $label"; return 2; }
  done
  [[ "$(grep -Ec '^tensor,(decoder_[0-9]+|quantizer|audio),[^,]+,.+$' "$reference" || true)" == 14 ]] || { die 'v2 tensor family has extra/missing rows'; return 2; }
}
require_local_reference() {
  local manifest="$1" prompt="$2" rows="$3" codes="$4" prompt_sha="$5" rows_sha="$6" codes_sha="$7" snapshot_root="${8:-}"
  [[ -f "$manifest" && ! -L "$manifest" ]] || { die 'Local reference manifest is not a regular file'; return 2; }
  local args=("$manifest" "$prompt" "$rows" "$codes" "$prompt_sha" "$rows_sha" "$codes_sha")
  [[ -z "$snapshot_root" ]] || args+=("$snapshot_root")
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/moss_tts_local/reference_validator.py" \
    "${args[@]}"
}
require_resolver_artifacts() {
  local lock_path="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$lock_path" <<'PY'
import sys, tomllib
from pathlib import Path
lock = tomllib.loads(Path(sys.argv[1]).read_bytes().decode())
for package in lock.get("package", []):
    if package.get("source", {}).get("virtual") is not None: continue
    artifacts = ([package["sdist"]] if package.get("sdist") else []) + list(package.get("wheels", []))
    if not artifacts: raise SystemExit(f"resolver artifacts missing: {package.get('name')}")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("url"), str) or not artifact["url"].startswith("https://") or not isinstance(artifact.get("hash"), str) or not artifact["hash"].startswith("sha256:") or not isinstance(artifact.get("size"), int) or artifact["size"] <= 0: raise SystemExit(f"resolver artifact metadata incomplete: {package.get('name')}")
PY
}
verify_downloaded_snapshot() {
  local snapshot="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$snapshot" <<'PY'
import hashlib, sys
from pathlib import Path
root=Path(sys.argv[1])
for member in root.rglob("*"):
 if ".cache" in member.relative_to(root).parts: continue
 if member.is_symlink() or (not member.is_file() and not member.is_dir()): raise SystemExit(f"snapshot contains symlink/non-regular member: {member}")
expected={
 "model.safetensors": (9100859544, "608f1ff64bc6caa9be836060fc7c78a15c4658c4a07b8d73c78d6f70d1b39c23"),
 "config.json": (10045, "826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411"),
 "configuration_moss_tts.py": (7160, "ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be"),
 "modeling_moss_tts.py": (26379, "b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f"),
 "processing_moss_tts.py": (37496, "3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad"),
 "gpt2_decoder.py": (30896, "f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989"),
 "qwen3_decoder.py": (25473, "100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0"),
 "processor_config.json": (210, "db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7"),
}
for name,(size,want) in expected.items():
 p=root/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size != size: raise SystemExit(f"snapshot identity mismatch: {name}")
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
 if h.hexdigest() != want: raise SystemExit(f"snapshot hash mismatch: {name}")
print("authenticated Local snapshot fixed identities")
PY
}

usage() {
  cat >&2 <<'EOF'
usage: run-moss-tts-local-composite-validation.sh --local-approval-evidence FILE --v2-approval-evidence FILE [--work-dir DIR]
       run-moss-tts-local-composite-validation.sh --self-test

VAST-only real-weight staging. It authenticates the fixed Local Transformer
and MOSS Audio Tokenizer v2 inputs, runs the independent official row/code
reference and native CPU measurement, and leaves all results as
MEASURED_NOT_GATED. It never uploads, publishes, or pushes a model.
The disposable Apple worker consumes this bundle for the corresponding Metal
comparison.
EOF
}

require_host() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is required"
  [[ "$(uname -s)" == Linux ]] || die "Local composite work is Linux/VAST-only"
  [[ "$(uname -m)" == x86_64 ]] || die "VAST worker requires Linux x86_64"
  [[ -r /proc/meminfo ]] || die "missing /proc/meminfo"
  local mem free
  mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  free="$(df -Pk "$VOKRA_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}')"
  [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 128 GiB"
  [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_DISK_KIB" ]] || die "free disk below 180 GB"
}

require_tools() {
  local tool
  for tool in uv cargo rustc git awk df sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"
  done
  [[ -f "$LOCAL_PROJECT/uv.lock" && -f "$LOCAL_GATE" && -f "$LOCAL_MANIFEST" ]] || die "dedicated Local gate/closure missing"
  [[ -f "$REFERENCE_DUMPER" && -f "$CODEC_REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
}

assert_fixed_contract() {
  grep -F 'REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"' "$VOKRA_ROOT/tools/audit/moss_audio_tokenizer_v2_manifest.py" >/dev/null \
    || die "v2 revision is not pinned in the auditor"
  grep -F 'TENSOR_COUNT = 2_094' "$VOKRA_ROOT/tools/audit/moss_audio_tokenizer_v2_manifest.py" >/dev/null \
    || die "v2 tensor count is not pinned in the auditor"
  grep -F 'MossTtsLocalSynthesis' "$VOKRA_ROOT/crates/vokra-models/src/moss_tts/local_transformer.rs" >/dev/null \
    || die "Local synthesis API is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "Local official reference dumper is missing"
  grep -F 'MEASURED_NOT_GATED' "$VOKRA_ROOT/crates/vokra-models/src/moss_audio_tokenizer/full_decoder.rs" >/dev/null \
    || die "measurement-only posture is missing"
}

run_remote_validation() {
  local work_dir="$1" snapshot local_input codec_snapshot codec_merged prompt_copy
  snapshot="$work_dir/local-snapshot"
  codec_snapshot="$work_dir/codec-snapshot"
  codec_merged="$work_dir/moss-audio-tokenizer-v2.safetensors"
  mkdir -p "$snapshot" "$codec_snapshot"
  uv sync --project "$LOCAL_PROJECT" --frozen --python 3.12
  uv sync --project "$V2_PROJECT" --frozen --python 3.12
  uv run --project "$LOCAL_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download
  snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' \
    "$LOCAL_REPO" "$LOCAL_REVISION" "$snapshot"
  verify_downloaded_snapshot "$snapshot"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["LICENSE", "config.json", "configuration_moss_audio_tokenizer.py", "modeling_moss_audio_tokenizer.py", "model.safetensors.index.json", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"])' \
    "$CODEC_REPO" "$CODEC_REVISION" "$codec_snapshot"
  local_input="$snapshot/model.safetensors"
  [[ -f "$local_input" ]] || die "fixed Local snapshot has no model.safetensors; refuse to guess a shard"
  local prompt_rows="${MOSS_TTS_LOCAL_PROMPT_ROWS:-}"
  require_prompt_rows "$prompt_rows"
  prompt_copy="$work_dir/prompt-rows.u32le"
  cp -- "$prompt_rows" "$prompt_copy"
  [[ "$(sha256sum "$prompt_copy" | awk '{print $1}')" == "${MOSS_TTS_LOCAL_PROMPT_ROWS_SHA256:-}" ]] \
    || die "copied prompt row SHA-256 mismatch"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python \
    "$VOKRA_ROOT/tools/audit/moss_audio_tokenizer_v2_manifest.py" \
    --config "$codec_snapshot/config.json" \
    --index "$codec_snapshot/model.safetensors.index.json" \
    --shard-dir "$codec_snapshot" > "$work_dir/codec-audit.json"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python \
    "$CODEC_REFERENCE_DUMPER" --variant v2 --frames 1 --num-quantizers 12 --device cuda \
    --output "$work_dir/v2-reference.csv"
  grep -F "source,v2,$CODEC_REPO,$CODEC_REVISION" "$work_dir/v2-reference.csv" >/dev/null \
    || die 'independent v2 codec reference lost its pinned source'
  log 'V2_CODEC_INDEPENDENT_MEASURED backend=cuda source=v2-reference.csv'
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_prepare_checkpoint.py" \
    --hf-repo "$CODEC_REPO" --revision "$CODEC_REVISION" \
    --local-dir "$codec_snapshot" --output "$codec_merged"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model moss-tts-local --input "$local_input" --output "$work_dir/moss-tts-local.gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model moss-audio-tokenizer-v2 --input "$codec_merged" \
    --output "$work_dir/moss-audio-tokenizer-v2.gguf"
  uv run --project "$LOCAL_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_DUMPER" --snapshot "$snapshot" --prompt-rows "$prompt_copy" \
    --max-new-frames "${MOSS_TTS_LOCAL_MAX_NEW_FRAMES:-1}" \
    --output "$work_dir/reference-rows.u32le" \
    --assistant-codes "$work_dir/reference-assistant-codes.u32le" \
    --manifest-output "$work_dir/reference-manifest.json" > "$work_dir/reference.log"
  grep -F 'verdict=MEASURED_NOT_GATED' "$work_dir/reference.log" >/dev/null \
    || die 'official Local reference did not report measurement-only posture'
  require_local_reference "$work_dir/reference-manifest.json" "$prompt_copy" \
    "$work_dir/reference-rows.u32le" "$work_dir/reference-assistant-codes.u32le" \
    "$(sha256sum "$prompt_copy" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/reference-rows.u32le" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/reference-assistant-codes.u32le" | awk '{print $1}')" "$snapshot"
  require_v2_reference "$work_dir/v2-reference.csv"
  local test_selector='moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official'
  VOKRA_MOSS_TTS_LOCAL_GGUF="$work_dir/moss-tts-local.gguf" \
    VOKRA_MOSS_AUDIO_TOKENIZER_V2_GGUF="$work_dir/moss-audio-tokenizer-v2.gguf" \
    VOKRA_MOSS_TTS_LOCAL_PROMPT_ROWS="$prompt_copy" \
    VOKRA_MOSS_TTS_LOCAL_REFERENCE_ROWS="$work_dir/reference-rows.u32le" \
    VOKRA_MOSS_TTS_LOCAL_REFERENCE_CODES="$work_dir/reference-assistant-codes.u32le" \
    VOKRA_MOSS_TTS_LOCAL_MAX_FRAMES="${MOSS_TTS_LOCAL_MAX_NEW_FRAMES:-1}" \
    CARGO_BUILD_JOBS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib "$test_selector" -- --ignored --exact --nocapture \
      > "$work_dir/native-cpu.log" 2>&1
  require_cpu_evidence "$work_dir/native-cpu.log"
  grep -F 'MOSS_TTS_LOCAL_ROWS_MEASURED' "$work_dir/native-cpu.log" > "$work_dir/native-row-metrics.txt"
  grep -F 'MOSS_TTS_LOCAL_PCM_MEASURED' "$work_dir/native-cpu.log" > "$work_dir/native-pcm-metrics.txt"
  sha256sum "$prompt_copy" "$work_dir/reference-rows.u32le" \
    "$work_dir/reference-assistant-codes.u32le" "$work_dir/reference-manifest.json" \
    "$work_dir/v2-reference.csv" \
    > "$work_dir/reference-sha256.txt"
  sha256sum "$work_dir/moss-tts-local.gguf" "$work_dir/moss-audio-tokenizer-v2.gguf" \
    > "$work_dir/artifact-sha256.txt"
  write_apple_command "$work_dir/apple-verifier-command.txt" \
    "$(sha256sum "$work_dir/moss-tts-local.gguf" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/moss-audio-tokenizer-v2.gguf" | awk '{print $1}')" \
    "$(sha256sum "$prompt_copy" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/reference-rows.u32le" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/reference-assistant-codes.u32le" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/reference-manifest.json" | awk '{print $1}')" \
    "$(sha256sum "$work_dir/v2-reference.csv" | awk '{print $1}')"
  log 'Native CPU rows and decoded PCM were executed; official composite PCM comparison remains explicitly not run.'
}

pre_sync_gates() {
  local local_approval_path="$1" v2_approval_path="$2" rc=0
  log 'Validating dedicated Local and accepted v2 closures before host/scratch/cache/network'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LOCAL_GATE" \
    --project "$LOCAL_PROJECT" --manifest "$LOCAL_MANIFEST" --approval-evidence "$local_approval_path" || rc=$?
  require_resolver_artifacts "$V2_PROJECT/uv.lock" || rc=$?
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$V2_GATE" \
    --lock "$V2_PROJECT/uv.lock" --project "$V2_PROJECT/pyproject.toml" \
    --manifest "$V2_MANIFEST" --approval "$v2_approval_path" || rc=$?
  return "$rc"
}

write_apple_command() {
  local output="$1" local_sha="$2" v2_sha="$3" prompt_sha="$4" rows_sha="$5" codes_sha="$6" local_manifest_sha="$7" v2_reference_sha="$8"
  {
    printf '%s\n' "VOKRA_REMOTE_APPLE_SILICON=1 \\"
    printf '%s\n' "scripts/verify/apple-silicon-moss-tts-local.sh \\"
    printf '%s\n' "  --local-gguf '<APPLE_LOCAL_GGUF_PATH>' \\"
    printf '%s\n' "  --local-gguf-sha256 $local_sha \\"
    printf '%s\n' "  --v2-gguf '<APPLE_V2_GGUF_PATH>' \\"
    printf '%s\n' "  --v2-gguf-sha256 $v2_sha \\"
    printf '%s\n' "  --prompt '<APPLE_PROMPT_PATH>' \\"
    printf '%s\n' "  --prompt-sha256 $prompt_sha \\"
    printf '%s\n' "  --reference-rows '<APPLE_REFERENCE_ROWS_PATH>' \\"
    printf '%s\n' "  --reference-rows-sha256 $rows_sha \\"
    printf '%s\n' "  --assistant-codes '<APPLE_ASSISTANT_CODES_PATH>' \\"
    printf '%s\n' "  --assistant-codes-sha256 $codes_sha \\"
    printf '%s\n' "  --local-reference-manifest '<APPLE_LOCAL_MANIFEST_PATH>' \\"
    printf '%s\n' "  --local-reference-manifest-sha256 $local_manifest_sha \\"
    printf '%s\n' "  --v2-reference '<APPLE_V2_REFERENCE_PATH>' \\"
    printf '%s\n' "  --v2-reference-sha256 $v2_reference_sha \\"
    printf '%s\n' "  --local-approval-evidence '<APPLE_LOCAL_APPROVAL_EVIDENCE>' \\"
    printf '%s\n' "  --v2-approval-evidence '<APPLE_V2_APPROVAL_EVIDENCE>' \\"
    printf '%s\n' "  --evidence-dir '<APPLE_EMPTY_EVIDENCE_DIR>'"
  } > "$output"
}

canonical_candidate() {
  local value="$1" suffix='' parent rest component scan
  [[ "$value" = /* ]] || value="$PWD/$value"
  rest="${value#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || { die "path contains symlink component: $scan"; return 2; }
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die 'work-dir has no canonical parent'; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die "work-dir parent is not a real directory: $value"; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

canonical_file() {
  local value="$1" parent base
  [[ -f "$value" && ! -L "$value" ]] || { die "approval evidence must be a regular non-symlink file: $value"; return 2; }
  parent="$(dirname "$value")"; base="$(basename "$value")"
  (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$base")
}

paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

validate_work_dir() {
  local candidate="$1" local_approval="$2" v2_approval="$3" prompt="$4"
  local canonical_work canonical_root canonical_local canonical_v2 canonical_local_approval canonical_v2_approval canonical_prompt protected
  [[ -n "$candidate" ]] || { die 'work-dir is empty'; return 2; }
  [[ ! -L "$candidate" ]] || { die 'work-dir must not be a symlink'; return 2; }
  [[ ! -e "$candidate" && ! -L "$candidate" ]] || { die 'work-dir must be absent/nonexistent (existing empty directories are rejected)'; return 2; }
  canonical_work="$(canonical_candidate "$candidate")" || return 2
  canonical_root="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  canonical_local="$(canonical_candidate "$LOCAL_PROJECT")" || return 2
  canonical_v2="$(canonical_candidate "$V2_PROJECT")" || return 2
  canonical_local_approval="$(canonical_file "$local_approval")" || return 2
  canonical_v2_approval="$(canonical_file "$v2_approval")" || return 2
  for protected in "$canonical_root" "$canonical_local" "$canonical_v2" "$canonical_local_approval" "$canonical_v2_approval"; do
    paths_overlap "$canonical_work" "$protected" && { die "work-dir overlaps protected path: $protected"; return 2; }
  done
  if [[ -f "$prompt" && ! -L "$prompt" ]]; then
    canonical_prompt="$(canonical_file "$prompt")" || return 2
    paths_overlap "$canonical_work" "$canonical_prompt" && { die "work-dir overlaps prompt input: $canonical_prompt"; return 2; }
  fi
  printf '%s\n' "$canonical_work"
}

# shellcheck disable=SC2016
self_test() {
  local gate_line host_line sync_line command_file probe_root
  gate_line="$(grep -n '^  pre_sync_gates ' "$0" | tail -1 | cut -d: -f1)"
  host_line="$(grep -n '^  require_host$' "$0" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^  run_remote_validation' "$0" | tail -1 | cut -d: -f1)"
  (( gate_line > 0 && gate_line < host_line && gate_line < sync_line )) || die 'preflight gates are not first'
  grep -F 'UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12' "$0" >/dev/null || die 'gates must disable UV cache'
  grep -F 'UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$V2_GATE"' "$0" >/dev/null || die 'v2 gate must disable UV cache'
  grep -F -- '--approval-evidence "$local_approval_path"' "$0" >/dev/null || die 'Local external approval evidence option is missing'
  grep -F -- '--approval "$v2_approval_path"' "$0" >/dev/null || die 'v2 external approval evidence option is missing'
  grep -F 'uv sync --project "$LOCAL_PROJECT" --frozen --python 3.12' "$0" >/dev/null || die 'Local closure sync is missing'
  grep -F 'uv sync --project "$V2_PROJECT" --frozen --python 3.12' "$0" >/dev/null || die 'v2 closure sync is missing'
  grep -F '"$CODEC_REFERENCE_DUMPER" --variant v2 --frames 1 --num-quantizers 12 --device cuda' "$0" >/dev/null || die 'v2 reference must run on CUDA'
  ! sed -n '/^run_remote_validation()/,/^pre_sync_gates()/p' "$0" | grep -Fq "\"$CODEC_REFERENCE_DUMPER\" --variant v2 --frames 1 --num-quantizers 12 --device cpu" || die 'v2 reference must not run on CPU'
  ! sed -n '/^run_remote_validation()/,/^pre_sync_gates()/p' "$0" | grep -Fq 'uv run --with' || die 'composite gate must not use dynamic uv dependencies'
  ! sed -n '/^run_remote_validation()/,/^pre_sync_gates()/p' "$0" | grep -Eq 'uv sync --project "[^"]*moss_audio' || die 'composite gate must use only the dedicated Local project'
  assert_fixed_contract
  if grep -En '(^|[[:space:]])(curl|wget|huggingface-cli)[[:space:]].*(upload|push)' "$0" >/dev/null; then
    die 'self-test found a download/upload command'
  fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|--push|--upload)' "$0" >/dev/null; then
    die 'self-test found a publication command'
  fi
  grep -F "$LOCAL_REPO" "$0" >/dev/null
  grep -F "$LOCAL_REVISION" "$0" >/dev/null
  grep -F "$CODEC_REPO" "$0" >/dev/null
  grep -F 'MEASURED_NOT_GATED' "$0" >/dev/null
  grep -F 'moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official' "$0" >/dev/null
  grep -F 'COMPOSITE_PCM_NOT_RUN' "$0" >/dev/null
  grep -F 'V2_CODEC_INDEPENDENT_MEASURED' "$0" >/dev/null
  grep -F 'require_local_reference' "$0" >/dev/null || die 'strict Local reference validator is missing'
  grep -F 'require_v2_reference' "$0" >/dev/null || die 'strict v2 CSV validator is missing'
  grep -F 'require_resolver_artifacts "$V2_PROJECT/uv.lock"' "$0" >/dev/null || die 'v2 resolver artifact validation is missing'
  grep -F 'exact=true' "$0" >/dev/null
  grep -F 'exact_to_cpu=true' "$0" >/dev/null
  if "$0" --self-test unexpected >/dev/null 2>&1; then die '--self-test accepted an extra argument'; fi
  if VOKRA_PUBLISH_ON_VAST=1 "$0" --work-dir >/dev/null 2>&1; then die 'missing work-dir value was accepted'; fi
  if VOKRA_PUBLISH_ON_VAST=1 "$0" --work-dir a --work-dir b >/dev/null 2>&1; then die 'duplicate work-dir was accepted'; fi
  if VOKRA_PUBLISH_ON_VAST=1 "$0" trailing >/dev/null 2>&1; then die 'unknown/trailing argument was accepted'; fi
  local evidence_log
  evidence_log="$(mktemp "${TMPDIR:-/tmp}/moss-tts-local-evidence.XXXXXX")"
  local v2_fixture="$evidence_log.v2"
  {
    printf '%s\n' "source,v2,$CODEC_REPO,$CODEC_REVISION" 'runtime,torch-2.7.1,transformers-5.5.0' 'environment,cpu,arm64,machine-selftest,logical-1,torch-capability-cpu' 'environment,device,cuda' 'source_file,model,transformers_modules/moss/modeling.py,7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9' 'source_file,config,transformers_modules/moss/configuration.py,f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529' 'contract,1,12,1024,48000,2,3840' 'codes,0,1,2,3,4,5,6,7,8,9,10,11'
    for label in decoder_0 decoder_1 decoder_2 decoder_3 decoder_4 decoder_5 decoder_6 decoder_7 decoder_8 decoder_9 decoder_10 decoder_11 quantizer audio; do printf 'tensor,%s,[1],0\n' "$label"; done
  } > "$v2_fixture"
  require_v2_reference "$v2_fixture"
  printf '%s\n' extra,unexpected >> "$v2_fixture"
  if require_v2_reference "$v2_fixture" >/dev/null 2>&1; then rm -f "$evidence_log" "$v2_fixture"; die 'v2 extra row was accepted'; fi
  printf '%s\n' 'test moss_tts::local_transformer::tests::measure_local_real_cpu_and_optional_metal_against_official ... ok' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s' 'MOSS_TTS_LOCAL_ROWS_MEASURED backend=cpu exact=true differing_values=0' 'MOSS_TTS_LOCAL_CODES_MEASURED backend=cpu exact=true differing_values=0' 'MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu samples=3840 channels=2 rms=0.000000000e+00 peak=0.000000000e+00' 'COMPOSITE_PCM_NOT_RUN reason=official_v2_pcm_sidecar_not_supplied' > "$evidence_log"
  require_cpu_evidence "$evidence_log"
  for malformed in extra failed duplicate-result malformed-suffix prefix fail nonzero; do
    cp "$evidence_log" "$evidence_log.$malformed"
    case "$malformed" in
      extra) printf '%s\n' 'test another::test ... ok' >> "$evidence_log.$malformed";;
      failed) sed 's/\.\.\. ok$/... FAILED/' "$evidence_log.$malformed" > "$evidence_log.tmp" && mv "$evidence_log.tmp" "$evidence_log.$malformed";;
      duplicate-result) printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$evidence_log.$malformed";;
      malformed-suffix) sed -E 's/filtered out(; finished in [0-9]+\.[0-9]+s)?$/filtered out; finished in nope/' "$evidence_log.$malformed" > "$evidence_log.tmp" && mv "$evidence_log.tmp" "$evidence_log.$malformed";;
      prefix) sed 's/^MOSS_TTS_LOCAL_ROWS_MEASURED /prefix MOSS_TTS_LOCAL_ROWS_MEASURED /' "$evidence_log.$malformed" > "$evidence_log.tmp" && mv "$evidence_log.tmp" "$evidence_log.$malformed";;
      fail) printf '%s\n' 'MOSS_TTS_LOCAL_PCM_MEASURED backend=cpu FAIL' >> "$evidence_log.$malformed";;
      nonzero) sed 's/differing_values=0/differing_values=1/g' "$evidence_log.$malformed" > "$evidence_log.tmp" && mv "$evidence_log.tmp" "$evidence_log.$malformed";;
    esac
    if require_cpu_evidence "$evidence_log.$malformed" >/dev/null 2>&1; then rm -f "$evidence_log" "$evidence_log."* "$v2_fixture"; die "accepted malformed Cargo evidence: $malformed"; fi
  done
  rm -f "$evidence_log" "$evidence_log."* "$v2_fixture"
  command_file="$(mktemp "${TMPDIR:-/tmp}/moss-tts-local-command.XXXXXX")"
  write_apple_command "$command_file" "$(printf '%064d' 1)" "$(printf '%064d' 2)" "$(printf '%064d' 3)" "$(printf '%064d' 4)" "$(printf '%064d' 5)" "$(printf '%064d' 6)" "$(printf '%064d' 7)"
  grep -Fq "'<APPLE_LOCAL_GGUF_PATH>'" "$command_file" || die 'portable Local GGUF placeholder is not quoted'
  grep -Fq "'<APPLE_REFERENCE_ROWS_PATH>'" "$command_file" || die 'portable reference placeholder is not quoted'
  grep -Fq "'<APPLE_EMPTY_EVIDENCE_DIR>'" "$command_file" || die 'portable absent evidence placeholder is not quoted'
  bash -n "$command_file" || die 'portable Apple command is not shell-valid'
  grep -Fq "'<APPLE_LOCAL_APPROVAL_EVIDENCE>'" "$command_file" || die 'portable Local approval placeholder is missing'
  grep -Fq "'<APPLE_V2_APPROVAL_EVIDENCE>'" "$command_file" || die 'portable v2 approval placeholder is missing'
  grep -Fq "'<APPLE_EMPTY_EVIDENCE_DIR>'" "$command_file" || die 'portable absent evidence placeholder is missing'
  rm -f "$command_file"
  if VOKRA_PUBLISH_ON_VAST=1 "$0" --local-approval-evidence >/dev/null 2>&1 || VOKRA_PUBLISH_ON_VAST=1 "$0" --v2-approval-evidence >/dev/null 2>&1 || VOKRA_PUBLISH_ON_VAST=1 "$0" --local-approval-evidence -bad --v2-approval-evidence good >/dev/null 2>&1 || VOKRA_PUBLISH_ON_VAST=1 "$0" --local-approval-evidence a --local-approval-evidence b --v2-approval-evidence c >/dev/null 2>&1 || VOKRA_PUBLISH_ON_VAST=1 "$0" --local-approval-evidence a --v2-approval-evidence b --v2-approval-evidence c >/dev/null 2>&1 || VOKRA_PUBLISH_ON_VAST=1 "$0" --local-approval-evidence a --v2-approval-evidence b trailing >/dev/null 2>&1; then
    die 'approval option parser accepted malformed input'
  fi
  local work_probe work_approval_a work_approval_b
  work_probe="$(mktemp -d "${TMPDIR:-/tmp}/moss-tts-local-work.XXXXXX")"
  work_approval_a="$work_probe/approval-a.json"; work_approval_b="$work_probe/approval-b.json"
  printf '{}\n' > "$work_approval_a"; printf '{}\n' > "$work_approval_b"
  validate_work_dir "$work_probe/new-work" "$work_approval_a" "$work_approval_b" "$work_probe/missing-prompt" >/dev/null || { rm -rf "$work_probe"; die 'empty work-dir was rejected'; }
  mkdir "$work_probe/empty"
  if validate_work_dir "$work_probe/empty" "$work_approval_a" "$work_approval_b" "$work_probe/missing-prompt" >/dev/null 2>&1; then rm -rf "$work_probe"; die 'existing empty work-dir was accepted'; fi
  mkdir "$work_probe/nonempty"; : > "$work_probe/nonempty/file"
  if validate_work_dir "$work_probe/nonempty" "$work_approval_a" "$work_approval_b" "$work_probe/missing-prompt" >/dev/null 2>&1; then rm -rf "$work_probe"; die 'nonempty work-dir was accepted'; fi
  ln -s "$work_probe/missing" "$work_probe/symlink-work"
  if validate_work_dir "$work_probe/symlink-work" "$work_approval_a" "$work_approval_b" "$work_probe/missing-prompt" >/dev/null 2>&1; then rm -rf "$work_probe"; die 'symlink work-dir was accepted'; fi
  mkdir -p "$work_probe/real-parent/child"; ln -s "$work_probe/real-parent" "$work_probe/link-parent"
  if validate_work_dir "$work_probe/link-parent/child/new" "$work_approval_a" "$work_approval_b" "$work_probe/missing-prompt" >/dev/null 2>&1; then rm -rf "$work_probe"; die 'descendant under symlink work-dir was accepted'; fi
  if validate_work_dir "$VOKRA_ROOT" "$work_approval_a" "$work_approval_b" "$work_probe/missing-prompt" >/dev/null 2>&1; then rm -rf "$work_probe"; die 'checkout-overlapping work-dir was accepted'; fi
  rm -rf "$work_probe"
  local test_prompt
  test_prompt="$(mktemp "${TMPDIR:-/tmp}/moss-tts-local-prompt.XXXXXX")"
  printf '%052d' 0 > "$test_prompt"
  MOSS_TTS_LOCAL_PROMPT_ROWS_SHA256="$(sha256sum "$test_prompt" | awk '{print $1}')" \
    require_prompt_rows "$test_prompt"
  rm -f "$test_prompt"
  if require_prompt_rows "$test_prompt" >/dev/null 2>&1; then
    die 'self-test accepted a missing prompt/reference input'
  fi
  printf '%052d' 0 > "$test_prompt"
  if MOSS_TTS_LOCAL_PROMPT_ROWS_SHA256="$(printf '%064d' 0)" require_prompt_rows "$test_prompt" >/dev/null 2>&1; then
    die 'self-test accepted a prompt hash mismatch'
  fi
  ln -s "$test_prompt" "$test_prompt.symlink"
  if MOSS_TTS_LOCAL_PROMPT_ROWS_SHA256="$(sha256sum "$test_prompt" | awk '{print $1}')" require_prompt_rows "$test_prompt.symlink" >/dev/null 2>&1; then
    die 'self-test accepted a prompt symlink'
  fi
  rm -f "$test_prompt" "$test_prompt.symlink"
  probe_root="$(mktemp -d "${TMPDIR:-/tmp}/moss-tts-local-production.XXXXXX")"
  printf '{}\n' > "$probe_root/local-approval.json"
  printf '{}\n' > "$probe_root/v2-approval.json"
  if VOKRA_PUBLISH_ON_VAST=1 VOKRA_SCRATCH="$probe_root/scratch" UV_CACHE_DIR="$probe_root/cache" \
    "$0" --local-approval-evidence "$probe_root/local-approval.json" --v2-approval-evidence "$probe_root/v2-approval.json" --work-dir "$probe_root/work" > "$probe_root/production.log" 2>&1; then
    rm -rf "$probe_root"
    die 'production-shaped self-test unexpectedly passed with unresolved gates'
  fi
  grep -Fq 'moss Local gate: BLOCKED' "$probe_root/production.log" || { rm -rf "$probe_root"; die 'production self-test did not stop at Local gate'; }
  [[ ! -e "$probe_root/work" && ! -e "$probe_root/scratch" && ! -e "$probe_root/cache" ]] || { rm -rf "$probe_root"; die 'blocked production path created work/scratch/cache'; }
  rm -rf "$probe_root"
  log 'self-test: OK (contract and no-upload guards)'
}

main() {
  if [[ "${1:-}" == --self-test ]]; then (($# == 1)) || { die '--self-test does not accept extra arguments'; return 2; }; self_test; return 0; fi
  local work_dir="${VOKRA_SCRATCH}/moss-tts-local-composite" local_approval='' v2_approval='' seen_work_dir=0 seen_local_approval=0 seen_v2_approval=0
  while (($#)); do
    case "$1" in
      --work-dir)
        (($# >= 2)) && [[ -n "${2:-}" && "${2:-}" != -* ]] || { usage; die '--work-dir requires a non-empty value'; return 2; }
        ((seen_work_dir == 0)) || { die 'duplicate --work-dir'; return 2; }
        seen_work_dir=1; work_dir="$2"; shift 2;;
      --local-approval-evidence)
        (($# >= 2)) && [[ -n "${2:-}" && "${2:-}" != -* ]] || { usage; die '--local-approval-evidence requires a non-empty file'; return 2; }
        ((seen_local_approval == 0)) || { die 'duplicate --local-approval-evidence'; return 2; }
        seen_local_approval=1; local_approval="$2"; shift 2;;
      --v2-approval-evidence)
        (($# >= 2)) && [[ -n "${2:-}" && "${2:-}" != -* ]] || { usage; die '--v2-approval-evidence requires a non-empty file'; return 2; }
        ((seen_v2_approval == 0)) || { die 'duplicate --v2-approval-evidence'; return 2; }
        seen_v2_approval=1; v2_approval="$2"; shift 2;;
      *) usage; die "unknown or trailing argument: $1"; return 2;;
    esac
  done
  [[ -n "$local_approval" && -n "$v2_approval" ]] || die 'both explicit approval evidence files are required'
  [[ -f "$local_approval" && ! -L "$local_approval" && -s "$local_approval" ]] || die 'Local approval evidence must be nonempty regular file'
  [[ -f "$v2_approval" && ! -L "$v2_approval" && -s "$v2_approval" ]] || die 'v2 approval evidence must be nonempty regular file'
  pre_sync_gates "$local_approval" "$v2_approval"
  require_prompt_rows "${MOSS_TTS_LOCAL_PROMPT_ROWS:-}"
  work_dir="$(validate_work_dir "$work_dir" "$local_approval" "$v2_approval" "${MOSS_TTS_LOCAL_PROMPT_ROWS:-}")" || return 2
  require_host
  require_tools
  assert_fixed_contract
  [[ -d "$work_dir" ]] || mkdir -p "$work_dir"
  [[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work directory must be empty"
  run_remote_validation "$work_dir"
  log "fixed local source=${LOCAL_REPO}@${LOCAL_REVISION}"
  log "fixed codec source=${CODEC_REPO}@${CODEC_REVISION} manifest=${CODEC_MANIFEST}"
  log 'Download/convert/reference/native real-weight stages are intentionally VAST-only.'
  log 'Record executed CPU rows/codes/PCM as MEASURED_NOT_GATED; Apple consumes this bundle for Metal, composite PCM remains not run.'
  return 0
}

main "$@"
