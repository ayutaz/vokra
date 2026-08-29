#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/zonos_inspect.py"
HF_REPOSITORY="vokra/zonos-v0.1-transformer"
HF_REVISION="b1bf5c56d470eb9097e9b04f9deca364576574ba"
UPSTREAM_HF_REPOSITORY="Zyphra/Zonos-v0.1-transformer"
UPSTREAM_HF_REVISION="9d8331fc49cb5ba8aad2bb56cafd809c66598f4e"
SOURCE_REPOSITORY="https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION="bc40d98e1e1ab54fc65c483be127a90e3c7c0645"

die() { echo "zonos-vast: ERROR: $*" >&2; exit 2; }

self_test() {
  local failed=0 token
  for token in \
    "$HF_REPOSITORY" "$HF_REVISION" "$UPSTREAM_HF_REPOSITORY" "$UPSTREAM_HF_REVISION" \
    "$SOURCE_REPOSITORY" "$SOURCE_REVISION" 'zonos_vast_stage.py' 'zonos_dump_reference.py' \
    'zonos_prepare_conditioning_packet.py' '--phoneme-ids' 'projected_prefix' \
    '12d542bd219f7f31c91b893810d85b0d810285e603029c69fbd19fd3c7da2c5c' \
    '6543af3747d3e85bde862c3337744eea31f0105f9df6d8617c1c9afdae805847' \
    'INSPECTION_ONLY' 'INSPECTION_ERROR' \
    'NOT_IMPLEMENTED_FAIL_CLOSED' 'UNSUPPORTED' 'BLOCKED_BY_CPU' 'NOT_RUN' \
    'NO_UPLOAD' 'recursive_file_only' 'lfs_sha256' '246' 'MEASURED_NOT_GATED' \
    'parity_zonos_real.rs' 'VOKRA_ZONOS_DAC_GGUF' 'cargo test --locked -p vokra-models' \
    'reference-codes.u32le' 'native-cpu.log' '--native-log' 'AUTHENTICATED_ARTIFACT_SOURCE_EVIDENCE' 'exit 2'; do
    grep -Fq -- "$token" "$INSPECTOR" "$0" || { echo "missing Zonos contract: $token" >&2; failed=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$INSPECTOR" "$0" | grep -v 'grep -En' >/dev/null; then
    echo 'upload/publish command found' >&2
    failed=1
  fi
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/zonos_vast_stage.py" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/zonos_prepare_conditioning_packet.py" --self-test || failed=1
  (( failed == 0 )) || return 1
  echo 'run-zonos-inspection.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 0 ]] || die 'arguments are fixed; revisions and artifacts cannot be overridden'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_ZONOS_VAST_VALIDATION:-0}" == 1 ]] || die 'VOKRA_ZONOS_VAST_VALIDATION=1 is absent'
[[ -n "${ZONOS_CONDITIONING_PACKET:-}" ]] || die 'ZONOS_CONDITIONING_PACKET must name a v1 packet from zonos_prepare_conditioning_packet.py (--phoneme-ids, --speaker, --emotion)'
[[ -f "$ZONOS_CONDITIONING_PACKET" ]] || die 'ZONOS_CONDITIONING_PACKET is missing; run the deterministic offline preparer first'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((128 * 1024 * 1024)) ]] || die '128 GiB memory guard failed'
for command in git uv awk cp sha256sum cargo; do command -v "$command" >/dev/null || die "missing tool: $command"; done

WORK="/dev/shm/vokra-zonos-inspection"
[[ ! -e "$WORK" ]] || die 'inspection directory must be absent before worker start'
mkdir -p "$WORK/evidence"
cd "$ROOT"
cp -- "$ZONOS_CONDITIONING_PACKET" "$WORK/evidence/conditioning.packet"
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python \
  "$ROOT/tools/parity/zonos_vast_stage.py" --root "$WORK" \
  --upstream-safetensors "$WORK/upstream/model.safetensors" \
  --manifest-output "$WORK/evidence/upstream-tensor-manifest.json" \
  --public-gguf "$WORK/public/zonos-v0.1-transformer.gguf" \
  --public-manifest-output "$WORK/evidence/public-tensor-manifest.json"
git clone --filter=blob:none --no-checkout "$SOURCE_REPOSITORY" "$WORK/source"
git -C "$WORK/source" checkout --detach "$SOURCE_REVISION"

set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python \
  "$ROOT/tools/parity/zonos_dump_reference.py" \
  --source "$WORK/source" --upstream-snapshot "$WORK/upstream" \
  --conditioning-packet "$WORK/evidence/conditioning.packet" \
  --codes-output "$WORK/evidence/reference-codes.u32le" \
  --pcm-output "$WORK/evidence/reference-pcm.f32le" \
  >"$WORK/evidence/reference.log" 2>&1
reference_status=$?
set -e
[[ "$reference_status" == 0 ]] || die "official Zonos reference generation failed; see $WORK/evidence/reference.log"
grep -Fq '"reference_status": "MEASURED_NOT_GATED"' \
  "$WORK/evidence/reference-codes.json" || die 'reference is not marked MEASURED_NOT_GATED'

[[ -n "${ZONOS_DAC_GGUF:-}" && -f "$ZONOS_DAC_GGUF" ]] || die 'ZONOS_DAC_GGUF must name the authenticated 44.1-kHz DAC GGUF'
packet_digest="$(awk -F'"' '/conditioning_packet_content_digest/{print $4; exit}' \
  "$WORK/evidence/reference-codes.json")"
[[ "$packet_digest" =~ ^[0-9a-f]{64}$ ]] || die 'reference packet content digest is missing'
set +e
VOKRA_ZONOS_GGUF="$WORK/public/zonos-v0.1-transformer.gguf" \
VOKRA_ZONOS_DAC_GGUF="$ZONOS_DAC_GGUF" \
VOKRA_ZONOS_CONDITIONING_PACKET="$WORK/evidence/conditioning.packet" \
VOKRA_ZONOS_PACKET_SHA256="$packet_digest" \
VOKRA_ZONOS_REFERENCE_CODES="$WORK/evidence/reference-codes.u32le" \
VOKRA_ZONOS_REFERENCE_PCM="$WORK/evidence/reference-pcm.f32le" \
VOKRA_ZONOS_MAX_STEPS="${ZONOS_MAX_STEPS:-32}" \
CARGO_BUILD_JOBS=1 cargo test --locked -p vokra-models --test parity_zonos_real \
  zonos_real_cpu_codes_and_pcm_boundary -- --ignored --nocapture \
  >"$WORK/evidence/native-cpu.log" 2>&1
native_status=$?
set -e
[[ "$native_status" == 0 ]] || die "native Zonos CPU validation failed; see $WORK/evidence/native-cpu.log"
grep -Fq 'ZONOS_CPU_REFERENCE codes=EXACT' "$WORK/evidence/native-cpu.log" || die 'native CPU exact-code marker missing'
grep -Fq 'verdict=MEASURED_NOT_GATED' "$WORK/evidence/native-cpu.log" || die 'native CPU must remain MEASURED_NOT_GATED'

set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" \
  --snapshot "$WORK/public" --server-tree "$WORK/public-server-tree.json" \
  --tensor-manifest "$WORK/evidence/public-tensor-manifest.json" \
  --upstream-snapshot "$WORK/upstream" --upstream-server-tree "$WORK/upstream-server-tree.json" \
  --upstream-tensor-manifest "$WORK/evidence/upstream-tensor-manifest.json" \
  --source "$WORK/source" --reference-record "$WORK/evidence/reference-codes.json" \
  --native-log "$WORK/evidence/native-cpu.log" \
  --output "$WORK/evidence"
status=$?
set -e
[[ "$status" == 0 ]] || die "inspector returned $status"
grep -Fq '"inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE"' "$WORK/evidence/manifest.json" || die 'authenticated evidence was not complete'
grep -Fq '"publication": "NO_UPLOAD"' "$WORK/evidence/manifest.json" || die 'NO_UPLOAD marker missing'
grep -Fq '"cpu_status": "MEASURED_NOT_GATED"' "$WORK/evidence/manifest.json" || die 'CPU measurement status missing'
grep -Fq '"metal_status": "PENDING_REAL_APPLE_RUN"' "$WORK/evidence/manifest.json" || die 'Metal pending status missing'
grep -Fq '"reference_status": "MEASURED_NOT_GATED"' "$WORK/evidence/reference-codes.json" || die 'reference gate marker missing'
