#!/usr/bin/env bash
# VAST-only Kyutai STT dep_q=0 decoder-component measurement orchestrator.
# It downloads only the pinned local inputs on VAST, converts the exact
# decoder artifact, and records a first measurement without a PASS claim.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DUMPER="$ROOT/tools/parity/kyutai_stt_decoder_dump_reference.py"
MODEL_REPO="kyutai/stt-2.6b-en"
MODEL_REVISION="a07aec56d22be5589cd0bc8709c75b6cf3e3039d"
MODEL_SHA256="2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2"
DSM_REVISION="4c4f65e147df056adf3346290d64c7b9649b18c9"
MOSHI_REVISION="e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
WORK="/workspace/vokra-kyutai-stt-decoder-parity"

log() { printf '[kyutai-stt-decoder-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() { printf '%s\n' "usage: run-kyutai-stt-decoder-parity.sh --expected-head HEX40 [--work-dir DIR] | --self-test"; }

validate_measurement_log() {
  local path="$1"
  [[ "$(grep -Ec '^test parity_kyutai_stt_decoder_real_cpu \.\.\. ok$' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed;' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec '^KYUTAI_STT_DECODER_MEASUREMENT backend=Cpu .* verdict=MEASUREMENT_ONLY$' "$path")" == 1 ]] || return 1
  [[ "$(grep -Ec 'verdict=MEASUREMENT_ONLY' "$path")" == 1 ]] || return 1
  ! grep -Fq 'verdict=PASS' "$path"
}

self_test() {
  local self="${BASH_SOURCE[0]}" token fail=0
  for token in "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_SHA256" "$DSM_REVISION" "$MOSHI_REVISION" \
    'NO_UPLOAD' 'uv run' 'official Moshi' 'dep_q=0' '323' 'CARGO_BUILD_JOBS=1' \
    'MEASUREMENT_ONLY' 'vokra-convert' 'model.safetensors' '--expected-head' 'test result' '0 failed'; do
    grep -Fq -- "$token" "$self" || { log "self-test missing contract token: $token"; fail=1; }
  done
  grep -Eq '^[[:space:]]*git[[:space:]]+push|^[[:space:]]*(curl|wget)[[:space:]]' "$self" && fail=1 || true
  synthetic="$(mktemp)"
  printf '%s\n' 'test parity_kyutai_stt_decoder_real_cpu ... ok' 'test result: ok. 1 passed; 0 failed;' 'KYUTAI_STT_DECODER_MEASUREMENT backend=Cpu max_abs=0 verdict=MEASUREMENT_ONLY' > "$synthetic"
  validate_measurement_log "$synthetic" || fail=1
  printf '%s\n' 'KYUTAI_STT_DECODER_MEASUREMENT backend=Cpu max_abs=0 verdict=MEASUREMENT_ONLY' >> "$synthetic"
  validate_measurement_log "$synthetic" && fail=1 || true
  rm -f "$synthetic"
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python "$DUMPER" self-test || fail=1
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="$WORK"
expected_head=""
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi
while (($#)); do
  case "$1" in
    --expected-head) (($# >= 2)) || die '--expected-head requires HEX40'; [[ -z "$expected_head" ]] || die 'duplicate --expected-head'; expected_head="$2"; shift 2;;
    --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; work_dir="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage; die "unknown argument: $1";;
  esac
done
[[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase HEX40'

[[ "$(uname -s)" == Linux ]] || die 'decoder parity requires Linux VAST'
[[ "$(uname -m)" == x86_64 ]] || die 'decoder parity requires x86_64 VAST'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
actual_head="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
[[ -f "$DUMPER" ]] || die 'decoder reference dumper is missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((64 * 1024 * 1024)) ]] || die '64 GiB memory guard failed'
mkdir -p "$work_dir"
chmod 700 "$work_dir"
[[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
mkdir -m 700 "$work_dir/evidence" "$work_dir/model"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export CARGO_NET_OFFLINE=true
validation_log="$work_dir/evidence/validation.log"
[[ ! -e "$validation_log" ]] || die 'validation log already exists'
set -o noclobber
download_hf_file() {
  local filename="$1" destination="$2"
  UV_NO_CACHE=1 uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$MODEL_REPO" "$MODEL_REVISION" "$filename" "$(dirname "$destination")" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo, revision, filename, destination = sys.argv[1:]
path = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename, local_dir=destination))
expected = Path(destination) / filename
if path != expected or not expected.is_file() or expected.is_symlink():
    raise SystemExit(f"unexpected downloaded path: {path} != {expected}")
PY
}
{
  echo 'status=BLOCKED'
  echo 'scope=dep_q=0 decoder component only'
  echo "model=$MODEL_REPO@$MODEL_REVISION"
  echo "model_sha256=$MODEL_SHA256"
  echo "dsm_revision=$DSM_REVISION"
  echo "moshi_revision=$MOSHI_REVISION"
  echo "expected_head=$expected_head"
  echo "actual_head=$actual_head"
  echo 'publication=NO_UPLOAD'
  echo 'phase=VAST_MEASUREMENT_ONLY; no PASS claim or fixed tolerance'
  git clone --no-checkout "https://github.com/kyutai-labs/delayed-streams-modeling.git" "$work_dir/dsm"
  git -C "$work_dir/dsm" checkout --detach "$DSM_REVISION"
  git clone --no-checkout "https://github.com/kyutai-labs/moshi.git" "$work_dir/moshi"
  git -C "$work_dir/moshi" checkout --detach "$MOSHI_REVISION"
  for file in model.safetensors config.json mimi-pytorch-e351c8d8@125.safetensors tokenizer_en_audio_4000.model; do
    download_hf_file "$file" "$work_dir/model/$file"
  done
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python - "$work_dir/model" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
cache = root / ".cache"
if cache.exists() or cache.is_symlink():
    if cache.is_symlink():
        raise SystemExit("model .cache symlink is not accepted")
    shutil.rmtree(cache)
expected = {
    "model.safetensors",
    "config.json",
    "mimi-pytorch-e351c8d8@125.safetensors",
    "tokenizer_en_audio_4000.model",
}
if {item.name for item in root.iterdir()} != expected:
    raise SystemExit("model snapshot does not contain exactly the four authenticated files")
if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
    raise SystemExit("model snapshot contains a non-regular or symlink entry")
PY
  cargo build --release -p vokra-convert
  "$ROOT/target/release/vokra-convert" --model kyutai-stt --input "$work_dir/model/model.safetensors" --output "$work_dir/decoder.gguf"
  UV_NO_CACHE=1 uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$DUMPER" real \
    --model "$work_dir/model" --dsm-source "$work_dir/dsm" --moshi-source "$work_dir/moshi" --out "$work_dir/reference"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
  gguf_sha="$(sha256sum "$work_dir/decoder.gguf" | awk '{print $1}')"
  reference_sha="$(sha256sum "$work_dir/reference/manifest.json" | awk '{print $1}')"
  packet_sha="$(sha256sum "$work_dir/reference/input.json" | awk '{print $1}')"
  logits_sha="$(sha256sum "$work_dir/reference/logits.f32" | awk '{print $1}')"
  echo "gguf_sha256=$gguf_sha"
  echo "reference_manifest_sha256=$reference_sha"
  echo "packet_sha256=$packet_sha"
  echo "reference_logits_sha256=$logits_sha"
  VOKRA_KYUTAI_STT_DECODER_GGUF="$work_dir/decoder.gguf" \
  VOKRA_KYUTAI_STT_DECODER_GGUF_SHA256="$gguf_sha" \
  VOKRA_KYUTAI_STT_DECODER_REFERENCE="$work_dir/reference" \
  VOKRA_KYUTAI_STT_DECODER_REFERENCE_MANIFEST_SHA256="$reference_sha" \
  VOKRA_KYUTAI_STT_DECODER_MEASUREMENT_ONLY=1 \
    cargo test --test parity_kyutai_stt_decoder_real -p vokra-models --offline -- \
      --ignored --exact parity_kyutai_stt_decoder_real_cpu --nocapture
} > "$validation_log" 2>&1 || die 'VAST decoder measurement failed; evidence log preserved'
set +o noclobber
validate_measurement_log "$validation_log" || die 'measurement log singleton/result validation failed'
log_sha="$(sha256sum "$validation_log" | awk '{print $1}')"
log "MEASUREMENT_ONLY complete; log_sha256=$log_sha; review a fixed bound before enabling parity"
exit 0
