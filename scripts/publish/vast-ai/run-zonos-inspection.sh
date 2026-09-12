#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/zonos_inspect.py"
DEPENDENCY_AUDIT_WRAPPER="$ROOT/scripts/publish/vast-ai/audit-zonos-v0-1-dependencies.sh"
DEPENDENCY_APPROVAL_VALIDATOR="$ROOT/tools/parity/zonos_v0_1_reference/dependency_approval.py"
HF_REPOSITORY="vokra/zonos-v0.1-transformer"
HF_REVISION="b1bf5c56d470eb9097e9b04f9deca364576574ba"
UPSTREAM_HF_REPOSITORY="Zyphra/Zonos-v0.1-transformer"
UPSTREAM_HF_REVISION="9d8331fc49cb5ba8aad2bb56cafd809c66598f4e"
UPSTREAM_MODEL_PRIVATE=false
UPSTREAM_MODEL_GATED=false
UPSTREAM_MODEL_DISABLED=false
UPSTREAM_MODEL_CARD_DATA_LICENSE="apache-2.0"
SOURCE_REPOSITORY="https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION="bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
SOURCE_LICENSE_PATH="LICENSE"
SOURCE_LICENSE_SPDX="Apache-2.0"
SOURCE_LICENSE_BYTES=11357
SOURCE_LICENSE_GIT_BLOB_SHA1="7a4a3ea2424c09fbe48d455aed1eaa94d9124835"
SOURCE_LICENSE_SHA256="58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd"
DAC_SOURCE_MODEL_ID="descript/dac_44khz"
DAC_NUM_CODEBOOKS=9
DAC_SAMPLE_RATE=44100
DAC_IDENTITY_STATUS="SOURCE_REQUEST_ONLY"
WORK="/dev/shm/vokra-zonos-inspection"

die() { echo "zonos-vast: ERROR: $*" >&2; exit 2; }

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be exactly 40 lowercase hexadecimal characters'
  [[ -d "$ROOT/.git" ]] || die 'checkout is missing .git'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from expected $expected"
}

require_approval_binding() {
  local approval="$1" expected_sha="$2"
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || die '--approval-evidence-sha256 must be exactly 64 lowercase hexadecimal characters'
  [[ -f "$approval" && ! -L "$approval" ]] || die 'approval evidence must be a regular non-symlink file'
  [[ "$(sha256_file "$approval")" == "$expected_sha" ]] || die 'approval evidence SHA-256 differs from caller binding'
}

require_source_license_identity() {
  local source="$1" license actual_sha actual_blob
  license="$source/$SOURCE_LICENSE_PATH"
  [[ -f "$license" && ! -L "$license" ]] || die 'cloned Zonos source LICENSE is missing or symlinked'
  [[ "$(wc -c < "$license" | tr -d '[:space:]')" == "$SOURCE_LICENSE_BYTES" ]] || die 'cloned Zonos source LICENSE byte-size mismatch'
  actual_sha="$(sha256sum "$license" | awk '{print $1}')"
  [[ "$actual_sha" == "$SOURCE_LICENSE_SHA256" ]] || die 'cloned Zonos source LICENSE SHA-256 mismatch'
  actual_blob="$(git -C "$source" rev-parse "$SOURCE_REVISION:$SOURCE_LICENSE_PATH")" || die 'cloned Zonos source LICENSE Git blob is unavailable'
  [[ "$actual_blob" == "$SOURCE_LICENSE_GIT_BLOB_SHA1" ]] || die 'cloned Zonos source LICENSE Git blob mismatch'
  [[ "$(git -C "$source" remote get-url origin)" == "$SOURCE_REPOSITORY" ]] || die 'cloned Zonos source origin mismatch'
}

require_native_cpu_log() {
  local log="$1" named result tests
  named="$(grep -Ec '^test zonos_real_cpu_codes_and_pcm_boundary \.\.\. ok$' "$log" || true)"
  result="$(grep -Ec '^test result:' "$log" || true)"
  tests="$(grep -Ec '^test ' "$log" || true)"
  (( named == 1 && result == 1 && tests - result == 1 )) || die 'native CPU Cargo test/result lines are not singleton'
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log" || die 'native CPU Cargo result is not an exact singleton pass'
  [[ "$(grep -Ec '^ZONOS_CPU_REFERENCE codes=EXACT .*verdict=MEASURED_NOT_GATED$' "$log" || true)" == 1 ]] || die 'native CPU sentinel is not singleton'
}

write_transfer_manifest() {
  local output="$1" expected_head="$2" approval_sha="$3" public_manifest="$4" gguf="$5" dac="$6" packet="$7" codes="$8" pcm="$9" native_log="${10}" dependency_approval_sha="${11}"
  [[ ! -e "$output" && ! -L "$output" && ! -e "$output.sha256" && ! -L "$output.sha256" ]] || die 'transfer manifest outputs must be absent'
  (set -C; {
    printf 'schema=zonos-apple-transfer-v1\n'
    printf 'expected_head=%s\n' "$expected_head"
    printf 'approval_evidence_sha256=%s\n' "$approval_sha"
    printf 'dependency_approval_sha256=%s\n' "$dependency_approval_sha"
    printf 'source_license_path=%s\n' "$SOURCE_LICENSE_PATH"
    printf 'source_license_spdx=%s\n' "$SOURCE_LICENSE_SPDX"
    printf 'source_license_bytes=%s\n' "$SOURCE_LICENSE_BYTES"
    printf 'source_license_git_blob_sha1=%s\n' "$SOURCE_LICENSE_GIT_BLOB_SHA1"
    printf 'source_license_sha256=%s\n' "$SOURCE_LICENSE_SHA256"
    printf 'upstream_model_id=%s\n' "$UPSTREAM_HF_REPOSITORY"
    printf 'upstream_model_revision=%s\n' "$UPSTREAM_HF_REVISION"
    printf 'upstream_model_private=%s\n' "$UPSTREAM_MODEL_PRIVATE"
    printf 'upstream_model_gated=%s\n' "$UPSTREAM_MODEL_GATED"
    printf 'upstream_model_disabled=%s\n' "$UPSTREAM_MODEL_DISABLED"
    printf 'upstream_model_card_data_license=%s\n' "$UPSTREAM_MODEL_CARD_DATA_LICENSE"
    printf 'manifest_sha256=%s\n' "$(sha256_file "$public_manifest")"
    printf 'gguf_sha256=%s\n' "$(sha256_file "$gguf")"
    printf 'dac_gguf_sha256=%s\n' "$(sha256_file "$dac")"
    printf 'dac_source_model_id=%s\n' "$DAC_SOURCE_MODEL_ID"
    printf 'dac_num_codebooks=%s\n' "$DAC_NUM_CODEBOOKS"
    printf 'dac_sample_rate=%s\n' "$DAC_SAMPLE_RATE"
    printf 'dac_identity_status=%s\n' "$DAC_IDENTITY_STATUS"
    printf 'conditioning_packet_sha256=%s\n' "$(sha256_file "$packet")"
    printf 'reference_codes_sha256=%s\n' "$(sha256_file "$codes")"
    printf 'reference_pcm_sha256=%s\n' "$(sha256_file "$pcm")"
    printf 'native_cpu_log_sha256=%s\n' "$(sha256_file "$native_log")"
    printf 'cpu_result=ONE_PASS\n'
    printf 'cpu_sentinel_summary=ZONOS_CPU_REFERENCE codes=EXACT verdict=MEASURED_NOT_GATED\n'
    printf 'status=MEASURED_NOT_GATED\n'
    printf 'metal_status=NOT_RUN\n'
    printf 'publication=NO_UPLOAD\n'
  } > "$output") || die 'transfer manifest was concurrently created'
  (set -C; sha256_file "$output" > "$output.sha256") || die 'transfer manifest SHA sidecar was concurrently created'
}

self_test() {
  local failed=0 token temporary valid_log audit_line approval_line acquisition_line
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
    'reference-codes.u32le' 'native-cpu.log' '--native-log' 'AUTHENTICATED_ARTIFACT_SOURCE_EVIDENCE' 'exit 2' \
    '--approval-evidence' '--approval-evidence-sha256' '--expected-head' 'preflight-only' 'write_transfer_manifest' \
    'dependency_approval.py' 'ZONOS_DEPENDENCY_APPROVAL' 'ZONOS_DEPENDENCY_APPROVAL_SHA256' 'dependency_approval_sha256=' \
    'APPROVED_FOR_PRE_ACQUISITION' 'DEPENDENCY_SCOPE_ONLY' 'ALLOW_SOURCE_CHECKPOINT_ACQUISITION' \
    'CARGO_BUILD_JOBS=1' '--offline --locked' 'require_native_cpu_log' 'cpu_sentinel_summary=' 'metal_status=NOT_RUN' 'NO_UPLOAD' \
    'LICENSE' '11357' '7a4a3ea2424c09fbe48d455aed1eaa94d9124835' '58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd' 'apache-2.0' 'card_data_license' \
    'source_semantic_marker_status' 'expected_transformer_config' 'eos_token_id: int = 1024' 'masked_token_id: int = 1025' \
    'self.required_keys = {c.name for c in self.conditioners if c.uncond_vector is None}' \
    'logits[..., 1025:].fill_(-torch.inf)' 'unknown_token = -1' \
    'DacModel.from_pretrained("descript/dac_44khz")' 'descript/dac_44khz' 'dac_num_codebooks=9' 'dac_sample_rate=44100' 'dac_identity_status=SOURCE_REQUEST_ONLY' 'roll(k + 1)' 'attn_mlp_d_intermediate'; do
    grep -Fq -- "$token" "$INSPECTOR" "$0" || { echo "missing Zonos contract: $token" >&2; failed=1; }
  done
  audit_line="$(awk '/VOKRA_ZONOS_DEPENDENCY_AUDIT=1 bash/{print NR; exit}' "$0")"
  approval_line="$(awk '/--approval \"\$dependency_approval\"/{print NR; exit}' "$0")"
  acquisition_line="$(awk '/^for command in git uv awk cp sha256sum cargo;/{print NR; exit}' "$0")"
  if [[ ! "$audit_line" =~ ^[0-9]+$ || ! "$approval_line" =~ ^[0-9]+$ || ! "$acquisition_line" =~ ^[0-9]+$ ]] || \
    (( audit_line >= approval_line || approval_line >= acquisition_line )); then
    echo 'dependency approval transition is not ordered after audit and before acquisition' >&2
    failed=1
  fi
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-zonos-cpu-log.XXXXXX")"
  valid_log="$temporary/valid.log"
  printf '%s\n' \
    'test zonos_real_cpu_codes_and_pcm_boundary ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' > "$valid_log"
  (require_native_cpu_log "$valid_log") || { echo 'valid native CPU log was rejected' >&2; failed=1; }
  printf '%s\n' 'test extra ... ok' >> "$valid_log"
  if (require_native_cpu_log "$valid_log") >/dev/null 2>&1; then
    echo 'extra native CPU test was accepted' >&2
    failed=1
  fi
  printf '%s\n' \
    'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' > "$valid_log"
  if (require_native_cpu_log "$valid_log") >/dev/null 2>&1; then
    echo 'zero-pass native CPU log was accepted' >&2
    failed=1
  fi
  printf '%s\n' \
    'test zonos_real_cpu_codes_and_pcm_boundary ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' \
    'ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' > "$valid_log"
  if (require_native_cpu_log "$valid_log") >/dev/null 2>&1; then
    echo 'duplicate native CPU sentinel was accepted' >&2
    failed=1
  fi
  rm -rf "$temporary"
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$INSPECTOR" "$0" | grep -v 'grep -En' >/dev/null; then
    echo 'upload/publish command found' >&2
    failed=1
  fi
  for consumer in "$ROOT/tools/parity/zonos_dump_reference.py" "$ROOT/tools/parity/zonos_inspect.py" "$ROOT/tools/parity/zonos_vast_stage.py" "$0"; do
    grep -Fq 'zonos_v0_1_reference' "$consumer" || { echo "dedicated project binding missing: $consumer" >&2; failed=1; }
    generic_project="--project tools/"'parity --'
    if grep -Fq -- "$generic_project" "$consumer"; then
      echo "generic parity project fallback found: $consumer" >&2
      failed=1
    fi
  done
  bash "$DEPENDENCY_AUDIT_WRAPPER" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$INSPECTOR" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$ROOT/tools/parity/zonos_vast_stage.py" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$ROOT/tools/parity/zonos_prepare_conditioning_packet.py" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$ROOT/tools/parity/zonos_v0_1_reference/dependency_audit.py" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$DEPENDENCY_APPROVAL_VALIDATOR" --self-test || failed=1
  UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$ROOT/tools/parity/zonos_v0_1_reference/import_policy.py" || failed=1
  (( failed == 0 )) || return 1
  echo 'run-zonos-inspection.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 6 ]] || die 'expected --approval-evidence FILE --approval-evidence-sha256 SHA --expected-head HEX40'
[[ "$1" == --approval-evidence && "$3" == --approval-evidence-sha256 && "$5" == --expected-head ]] || die 'arguments must be approval evidence, caller SHA, and expected HEAD'
approval="$2"; approval_sha="$4"; expected_head="$6"
require_clean_expected_head "$expected_head"
require_approval_binding "$approval" "$approval_sha"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_ZONOS_VAST_VALIDATION:-0}" == 1 ]] || die 'VOKRA_ZONOS_VAST_VALIDATION=1 is absent'
[[ -n "${ZONOS_CONDITIONING_PACKET:-}" ]] || die 'ZONOS_CONDITIONING_PACKET must name a v1 packet from zonos_prepare_conditioning_packet.py (--phoneme-ids, --speaker, --emotion)'
[[ -f "$ZONOS_CONDITIONING_PACKET" && ! -L "$ZONOS_CONDITIONING_PACKET" ]] || die 'ZONOS_CONDITIONING_PACKET must be a regular non-symlink file'

# The installed/native/publisher closure is an independent exact-owner fact.
# A project/lock hash approval alone cannot authorize source or checkpoint
# acquisition.  The dedicated wrapper intentionally returns 2 while the
# owner/legal decision remains WITHHOLD, so this worker must stop here.
dependency_output="${ZONOS_DEPENDENCY_AUDIT_OUTPUT:-/dev/shm/vokra-zonos-dependency-audit-${expected_head}.json}"
dependency_archive="${ZONOS_DEPENDENCY_AUDIT_ARCHIVE:-/dev/shm/vokra-zonos-publisher-license-${expected_head}}"
set +e
VOKRA_ZONOS_DEPENDENCY_AUDIT=1 bash "$DEPENDENCY_AUDIT_WRAPPER" \
  --expected-head "$expected_head" --output "$dependency_output" \
  --publisher-archive "$dependency_archive"
dependency_status=$?
set -e
[[ "$dependency_status" == 2 ]] || die 'Zonos dependency audit did not fail closed before acquisition'

# The factual audit stays BLOCKED_UNREVIEWED_TRANSITIVE.  Only a separately
# supplied, external dependency approval bound to that exact report/scope may
# cross this transition; source/model approval above is a different contract.
dependency_approval="${ZONOS_DEPENDENCY_APPROVAL:-}"
dependency_approval_sha="${ZONOS_DEPENDENCY_APPROVAL_SHA256:-}"
[[ -n "$dependency_approval" ]] || die 'ZONOS_DEPENDENCY_APPROVAL must name an external dependency approval record'
[[ -n "$dependency_approval_sha" ]] || die 'ZONOS_DEPENDENCY_APPROVAL_SHA256 must bind the external dependency approval record'
set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --offline --frozen --project "$ROOT/tools/parity/zonos_v0_1_reference" --no-sync --python 3.12 python \
  "$DEPENDENCY_APPROVAL_VALIDATOR" \
  --approval "$dependency_approval" --approval-sha256 "$dependency_approval_sha" \
  --audit "$dependency_output" --expected-head "$expected_head"
dependency_approval_status=$?
set -e
[[ "$dependency_approval_status" == 0 ]] || die 'Zonos dependency approval did not authenticate the exact audit scope before acquisition'
echo 'Zonos dependency approval PASS; factual audit remains BLOCKED_UNREVIEWED_TRANSITIVE; publication=NO_UPLOAD' >&2

for command in git uv awk cp sha256sum cargo; do command -v "$command" >/dev/null || die "missing tool: $command"; done
UV_NO_CACHE=1 uv run --no-cache --frozen --project "$ROOT/tools/parity/zonos_v0_1_reference" --no-sync --offline --python 3.12 python "$ROOT/tools/parity/zonos_vast_stage.py" \
  --preflight-only --approval-evidence "$approval" || die 'Zonos license/preflight gate blocked before acquisition'
require_clean_expected_head "$expected_head"
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((128 * 1024 * 1024)) ]] || die '128 GiB memory guard failed'

cd "$ROOT"
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity/zonos_v0_1_reference" --python 3.12 python \
  "$ROOT/tools/parity/zonos_vast_stage.py" --root "$WORK" \
  --approval-evidence "$approval" \
  --upstream-safetensors "$WORK/upstream/model.safetensors" \
  --manifest-output "$WORK/evidence/upstream-tensor-manifest.json" \
  --public-gguf "$WORK/public/zonos-v0.1-transformer.gguf" \
  --public-manifest-output "$WORK/evidence/public-tensor-manifest.json"
cp -- "$ZONOS_CONDITIONING_PACKET" "$WORK/evidence/conditioning.packet"
git clone --filter=blob:none --no-checkout "$SOURCE_REPOSITORY" "$WORK/source"
git -C "$WORK/source" checkout --detach "$SOURCE_REVISION"
require_source_license_identity "$WORK/source"

set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity/zonos_v0_1_reference" --python 3.12 python \
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

[[ -n "${ZONOS_DAC_GGUF:-}" && -f "$ZONOS_DAC_GGUF" && ! -L "$ZONOS_DAC_GGUF" ]] || die 'ZONOS_DAC_GGUF must name a topology-compatible DAC GGUF; native binding authenticates its own metadata/manifest'
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
CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --offline --locked -p vokra-models --test parity_zonos_real \
  zonos_real_cpu_codes_and_pcm_boundary -- --ignored --exact --nocapture --test-threads=1 \
  >"$WORK/evidence/native-cpu.log" 2>&1
native_status=$?
set -e
[[ "$native_status" == 0 ]] || die "native Zonos CPU validation failed; see $WORK/evidence/native-cpu.log"
require_native_cpu_log "$WORK/evidence/native-cpu.log"

set +e
UV_CACHE_DIR="${ZONOS_UV_CACHE_DIR:-/tmp/vokra-zonos-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity/zonos_v0_1_reference" --python 3.12 python "$INSPECTOR" \
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
write_transfer_manifest "$WORK/evidence/apple-transfer-manifest.txt" "$expected_head" "$approval_sha" \
  "$WORK/public-server-tree.json" "$WORK/public/zonos-v0.1-transformer.gguf" "$ZONOS_DAC_GGUF" \
  "$WORK/evidence/conditioning.packet" "$WORK/evidence/reference-codes.u32le" "$WORK/evidence/reference-pcm.f32le" \
  "$WORK/evidence/native-cpu.log" "$dependency_approval_sha"
