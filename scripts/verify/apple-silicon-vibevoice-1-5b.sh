#!/usr/bin/env bash
# Disposable Darwin arm64 VibeVoice measurement worker.
# Inputs are authenticated by VAST; this script never downloads, converts,
# publishes, uploads, or fabricates a parity result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REFERENCE="$VOKRA_ROOT/tools/parity/vibevoice_1_5b_dump_reference.py"
REFERENCE_AUDIT_UV=(uv run --no-cache --no-project --offline --python 3.12 python)

die() { printf '[vibevoice-apple] ERROR: %s\n' "$*" >&2; exit 2; }

self_test() {
  local fail=0 token
  for token in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON=1 VOKRA_VIBEVOICE_GGUF VOKRA_VIBEVOICE_REFERENCE_DIR \
    parity_vibevoice_1_5b_real VIBEVOICE_CPU_TOKENS_MEASURED VIBEVOICE_METAL_TOKENS_MEASURED \
    VIBEVOICE_CPU_OFFICIAL_DIFFUSION_LATENTS_CAPTURED VIBEVOICE_METAL_OFFICIAL_DIFFUSION_LATENTS_CAPTURED \
    exact=true MEASURED_NOT_GATED official_pcm.f32le packet.json vibevoice-apple-summary.json NO_UPLOAD \
    reference_environment license_audit BLOCKED_UNREVIEWED_TRANSITIVE BLOCKED_UNVERIFIED_API_SMOKE GHSA-xrqw-3rrv-vx5w AUTHENTICATED_CLEAR --license-audit --no-project package-resolution-and-dependency-markers-v2 1ea002fe37f4ddc4df9f7535b5ae3a42661fc1eaa0a28e8ae6dbba0fa7e9649b 987a1f7204c2d7f2baa1c537ebaa06ca4bc872d2aae60f25a78393967da7bf8c ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909 \
    'VOKRA_VIBEVOICE_BACKEND=cpu' 'VOKRA_VIBEVOICE_BACKEND=metal' '--ignored --exact --nocapture'; do
    grep -Fq -- "$token" "$0" || { printf 'self-test missing %s\n' "$token" >&2; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)git[[:space:]]+push|(^|[;&|][[:space:]]*)(curl|wget|snapshot_download)[[:space:]]' "$0" >/dev/null; then
    printf 'self-test found download/publication command\n' >&2; fail=1
  fi
  local gate_line line
  gate_line="$(awk '/^[[:space:]]+license_audit_preflight$/{print NR; exit}' "$0")"
  [[ "$gate_line" =~ ^[0-9]+$ ]] || { printf 'self-test cannot locate license gate\n' >&2; fail=1; }
  for token in 'local bundle=' 'require_bundle ' 'cargo test --manifest-path'; do
    line="$(awk -v gate="$gate_line" -v token="$token" 'NR > gate && index($0, token) {print NR; exit}' "$0")"
    [[ "$line" =~ ^[0-9]+$ && "$line" -gt "$gate_line" ]] || { printf 'self-test operation precedes license gate: %s\n' "$token" >&2; fail=1; }
  done
  grep -Fq -- 'REFERENCE_AUDIT_UV=(uv run --no-cache --no-project --offline --python 3.12 python)' "$0" || { printf 'self-test missing no-cache audit command\n' >&2; fail=1; }
  if grep -En '(^|[[:space:]])uv[[:space:]]+sync([[:space:]]|$)' "$0" >/dev/null; then
    printf 'self-test found an implicit uv sync\n' >&2; fail=1
  fi
  (( fail == 0 )) && printf '[vibevoice-apple] self-test: OK\n' || return 1
}

require_bundle() {
  local bundle="$1"
  [[ -d "$bundle" ]] || die "bundle missing: $bundle"
  for file in manifest.json inspection-manifest.json token_ids.u32le prompt_pcm.f32le \
    prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le \
    speech_input_mask.u8 speech_masks.u8 \
    speech_replacement_positions.u32le generated_tokens.u32le official_pcm.f32le \
    official_diffusion_latents.f32le \
    packet.json vibevoice-1.5b.gguf; do
    [[ -f "$bundle/$file" && ! -L "$bundle/$file" ]] || die "bundle input missing or symlinked: $file"
  done
  uv run --frozen --project "$VOKRA_ROOT/tools/parity" --python 3.12 python - "$bundle/manifest.json" "$bundle/inspection-manifest.json" "$bundle/packet.json" <<'PY'
import hashlib, json, sys
for name in sys.argv[1:]:
    if name.endswith("packet.json"):
        continue
    data = json.loads(open(name, encoding="utf-8").read())
    required = {
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "metal_status": "BLOCKED_BY_CPU",
        "publication": "NO_UPLOAD",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise SystemExit(f"{name}: {key} is not fail-closed")
    expected_cpu = "MEASURED_NOT_GATED" if name == sys.argv[1] else "UNSUPPORTED"
    if data.get("cpu_status") != expected_cpu:
        raise SystemExit(f"{name}: unexpected CPU validation status")
    expected_parity = "MEASURED_NOT_GATED" if name == sys.argv[1] else "NOT_RUN"
    if data.get("parity_status") != expected_parity:
        raise SystemExit(f"{name}: unexpected parity validation status")
    if name == sys.argv[2] and data.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
        raise SystemExit(f"{name}: inspection evidence is not complete")
    if name == sys.argv[1] and data.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
        raise SystemExit(f"{name}: combined manifest lost inspection evidence")
    if name == sys.argv[1] and data.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE":
        raise SystemExit(f"{name}: official reference evidence is not complete")
    if name == sys.argv[1] and data.get("validation_status") != "CPU_NATIVE_REFERENCE_EXECUTED":
        raise SystemExit(f"{name}: native CPU validation is not recorded")
    if data.get("upstream", {}).get("revision") != "142f4a5dda029212cda8b118e9d99c3da27018d8":
        raise SystemExit(f"{name}: fixed HF revision mismatch")
    if name == sys.argv[1] and data.get("source", {}).get("revision") != "2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c":
        raise SystemExit(f"{name}: fixed source revision mismatch")
    if data.get("reference_status") == "REFERENCE_ERROR":
        raise SystemExit(f"{name}: official reference failed")
combined = json.loads(open(sys.argv[1], encoding="utf-8").read())
environment = combined.get("reference_environment")
if not isinstance(environment, dict):
    raise SystemExit("combined manifest is missing reference environment identity")
lock = environment.get("lock")
audit = environment.get("license_audit")
if not isinstance(lock, dict) or lock.get("sha256") != "ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909":
    raise SystemExit("combined manifest has an unreviewed VibeVoice lock")
if lock.get("package_rows_schema") != "package-resolution-and-dependency-markers-v2" or lock.get("package_rows_sha256") != "1ea002fe37f4ddc4df9f7535b5ae3a42661fc1eaa0a28e8ae6dbba0fa7e9649b":
    raise SystemExit("combined manifest has unreviewed VibeVoice dependency qualifiers")
if not isinstance(audit, dict) or audit.get("status") != "AUTHENTICATED_CLEAR":
    raise SystemExit("combined manifest has an unexpected VibeVoice license status")
if audit.get("license_audit_rows_sha256") != "987a1f7204c2d7f2baa1c537ebaa06ca4bc872d2aae60f25a78393967da7bf8c":
    raise SystemExit("combined manifest has unreviewed VibeVoice license rows")
security = environment.get("transformers_security")
if not isinstance(security, dict) or security.get("transformers_security_advisory") != "GHSA-xrqw-3rrv-vx5w" or security.get("transformers_security_patched_minimum") != "5.10.0" or security.get("isolated_transformers_pin") != "transformers==5.10.4" or security.get("transformers_compatibility_status") != "AUTHENTICATED_API_SMOKE":
    raise SystemExit("combined manifest has an unverified Transformers security closure")
packet_hash = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
if combined.get("input_packet_sha256") != packet_hash:
    raise SystemExit("caller-owned packet hash mismatch")
PY
}

license_audit_preflight() {
  local audit_output audit_rc
  [[ -f "$REFERENCE" ]] || die 'VibeVoice reference gate is missing'
  set +e
  audit_output="$("${REFERENCE_AUDIT_UV[@]}" "$REFERENCE" --license-audit 2>&1)"
  audit_rc=$?
  set -e
  if [[ "$audit_rc" == 2 ]]; then
    printf '%s\n' "$audit_output" >&2
    die 'dependency/license gate is unresolved; no Apple bundle or Cargo execution is permitted'
  fi
  [[ "$audit_rc" == 0 ]] || die "license audit command returned $audit_rc"
  [[ "$audit_output" == *"AUTHENTICATED_CLEAR"* ]] || die 'license audit did not return authenticated clearance'
  [[ "$audit_output" == *"ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909"* ]] || die 'license audit lock identity is missing'
}

main() {
  if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; return 0; fi
  [[ $# == 1 ]] || die 'usage: apple-silicon-vibevoice-1-5b.sh VAST_BUNDLE | --self-test'
  license_audit_preflight
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'disposable Darwin arm64 is required'
  local mem
  mem="$(sysctl -n hw.memsize 2>/dev/null || true)"
  [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge 34359738368 ]] || die 'at least 32 GiB RAM is required'
  [[ -d "$VOKRA_ROOT" && -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
  command -v uv >/dev/null 2>&1 || die 'uv is required'
  command -v cargo >/dev/null 2>&1 || die 'cargo is required'
  command -v xcrun >/dev/null 2>&1 || die 'xcrun is required'
  xcrun -sdk macosx metal -v >/dev/null 2>&1 || die 'Metal compiler unavailable'
  local bundle="$1"; bundle="$(cd "$bundle" 2>/dev/null && pwd)" || die 'bundle path invalid'
  case "$bundle/" in "$VOKRA_ROOT/"*) die 'evidence bundle must be outside checkout';; esac
  require_bundle "$bundle"
  local selector='parity_vibevoice_1_5b_real::vibevoice_1_5b_real_cpu_matches_official_reference'
  local cpu_log="$bundle/vibevoice-cpu.log" metal_log="$bundle/vibevoice-metal.log"
  VOKRA_VIBEVOICE_GGUF="$bundle/vibevoice-1.5b.gguf" \
    VOKRA_VIBEVOICE_REFERENCE_DIR="$bundle" \
    VOKRA_VIBEVOICE_BACKEND=cpu CARGO_BUILD_JOBS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release --features metal \
      -p vokra-models --test parity_vibevoice_1_5b_real "$selector" -- --ignored --exact --nocapture \
      2>&1 | tee "$cpu_log"
  grep -F 'VIBEVOICE_CPU_TOKENS_MEASURED exact=true' "$cpu_log" >/dev/null || die 'CPU exact token measurement missing'
  grep -F 'VIBEVOICE_CPU_PCM_MEASURED' "$cpu_log" >/dev/null || die 'CPU PCM measurement missing'
  grep -F 'VIBEVOICE_CPU_OFFICIAL_DIFFUSION_LATENTS_CAPTURED' "$cpu_log" >/dev/null || die 'CPU diffusion-latent evidence missing'
  VOKRA_VIBEVOICE_GGUF="$bundle/vibevoice-1.5b.gguf" \
    VOKRA_VIBEVOICE_REFERENCE_DIR="$bundle" \
    VOKRA_VIBEVOICE_BACKEND=metal CARGO_BUILD_JOBS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release --features metal \
      -p vokra-models --test parity_vibevoice_1_5b_real "$selector" -- --ignored --exact --nocapture \
      2>&1 | tee "$metal_log"
  grep -F 'VIBEVOICE_METAL_TOKENS_MEASURED exact=true' "$metal_log" >/dev/null || die 'Metal exact token measurement missing'
  grep -F 'VIBEVOICE_METAL_PCM_MEASURED' "$metal_log" >/dev/null || die 'Metal PCM measurement missing'
  grep -F 'VIBEVOICE_METAL_OFFICIAL_DIFFUSION_LATENTS_CAPTURED' "$metal_log" >/dev/null || die 'Metal diffusion-latent evidence missing'
  grep -F 'MEASURED_NOT_GATED' "$cpu_log" >/dev/null || die 'CPU PCM gate posture missing'
  grep -F 'MEASURED_NOT_GATED' "$metal_log" >/dev/null || die 'Metal PCM gate posture missing'
  uv run --frozen --project "$VOKRA_ROOT/tools/parity" --python 3.12 python - "$bundle/vibevoice-apple-summary.json" "$cpu_log" "$metal_log" <<'PY'
import hashlib
import json
import sys

summary, cpu, metal = sys.argv[1:]

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

with open(summary, "w", encoding="utf-8") as stream:
    json.dump({
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "cpu_status": "MEASURED_NOT_GATED",
        "metal_status": "MEASURED_NOT_GATED",
        "parity_status": "MEASURED_NOT_GATED",
        "publication": "NO_UPLOAD",
        "cpu_log_sha256": digest(cpu),
        "metal_log_sha256": digest(metal),
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  printf '[vibevoice-apple] CPU and Metal executions completed; PCM remains MEASURED_NOT_GATED; no upload.\n' >&2
  return 2
}

main "$@"
