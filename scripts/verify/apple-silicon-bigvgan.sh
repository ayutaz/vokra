#!/usr/bin/env bash
# shellcheck disable=SC2016 # literal source tokens are intentional self-test contracts
# Real Apple Silicon CPU/Metal BigVGAN model validation.
# The GGUF and reference fixture must have been produced/authenticated by the
# VAST worker. This script does not download, convert, upload, or publish.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bigvgan"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
VARIANT_MANIFEST="$PARITY_PROJECT/variant_manifest.json"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_bigvgan_real.rs"
MODEL_REPOSITORY=''
MODEL_VARIANT=''
MODEL_KIND=''
TEST_NAME=''
GGUF_ENV=''
REFERENCE_ENV=''
EXPECTED_MODEL_REVISION=''
EXPECTED_CHECKPOINT_BYTES=''
EXPECTED_CHECKPOINT_SHA256=''
EXPECTED_CONFIG_BYTES=''
EXPECTED_CONFIG_SHA256=''
EXPECTED_SOURCE_REVISION=''
EXPECTED_STATUS=''
CPU_ATOL="0.000020000"
METAL_ATOL="0.010000000"
MIN_MEMORY_BYTES=16000000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[bigvgan-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-bigvgan.sh --variant <slug> --gguf <file> --gguf-sha256 <64-hex> \
       --reference <file> --reference-sha256 <64-hex> \
       --model-revision <40-hex> [--checkpoint-bytes <decimal>] \
       --checkpoint-sha256 <64-hex> [--config-bytes <decimal>] \
       --config-sha256 <64-hex> --source-revision <40-hex> \
       --approval-evidence <file> \
       --evidence-dir <absent-dir>
       apple-silicon-bigvgan.sh --self-test

Runs the exact real-weight BigVGAN variant parity test once on CPU and once with
the real Metal feature. The model binder enforces its resident route and one
final readback internally. The GGUF hash is mandatory and must come from the
VAST conversion worker; no model conversion or network operation is performed.
The checkpoint/config byte arguments are required for variants with registered
primary-source sizes and are checked against the authoritative variant
manifest. The evidence directory must be absent/nonexistent before validation;
it is created only after all input and approval checks succeed.
EOF
}

configure_variant() {
  case "$1" in
    v2_22khz_80band_256x)
      MODEL_REPOSITORY='nvidia/bigvgan_v2_22khz_80band_256x'; MODEL_VARIANT='v2_22khz_80band_256x'; MODEL_KIND='bigvgan-v2-22khz-80band-256x'; TEST_NAME='parity_bigvgan_v2_22khz_80band_256x_real_weight_mel_to_waveform'; GGUF_ENV='VOKRA_BIGVGAN_V2_22KHZ_80BAND_256X_GGUF'; REFERENCE_ENV='VOKRA_BIGVGAN_V2_22KHZ_80BAND_256X_REFERENCE'; EXPECTED_SOURCE_REVISION='7d2b454564a6c7d014227f635b7423881f14bdac'; EXPECTED_MODEL_REVISION='633ff708ed5b74903e86ff1298cf4a98e921c513'; EXPECTED_CHECKPOINT_BYTES='449228171'; EXPECTED_CHECKPOINT_SHA256='e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced'; EXPECTED_CONFIG_BYTES='1405'; EXPECTED_CONFIG_SHA256='88a1f47acf747db0b21e97a389d838566147f7a5464583ff5c8d819d870f03ee'; EXPECTED_STATUS='IDENTITY_REVIEWED_APPROVAL_PENDING';;
    v2_44khz_128band_512x)
      MODEL_REPOSITORY='nvidia/bigvgan_v2_44khz_128band_512x'; MODEL_VARIANT='v2_44khz_128band_512x'; MODEL_KIND='bigvgan-v2-44khz-128band-512x'; TEST_NAME='parity_bigvgan_v2_44khz_128band_512x_real_weight_mel_to_waveform'; GGUF_ENV='VOKRA_BIGVGAN_V2_44KHZ_128BAND_512X_GGUF'; REFERENCE_ENV='VOKRA_BIGVGAN_V2_44KHZ_128BAND_512X_REFERENCE'; EXPECTED_SOURCE_REVISION='7d2b454564a6c7d014227f635b7423881f14bdac'; EXPECTED_MODEL_REVISION='95a9d1dcb12906c03edd938d77b9333d6ded7dfb'; EXPECTED_CHECKPOINT_BYTES='489041291'; EXPECTED_CHECKPOINT_SHA256='d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357'; EXPECTED_CONFIG_BYTES='1403'; EXPECTED_CONFIG_SHA256='65b7f487bfaf15256056a75f6667ef5f640214c981922a111927873705ea4ddb'; EXPECTED_STATUS='IDENTITY_REVIEWED_APPROVAL_PENDING';;
    v2_24khz_100band_256x)
      MODEL_REPOSITORY='nvidia/bigvgan_v2_24khz_100band_256x'; MODEL_VARIANT='v2_24khz_100band_256x'; MODEL_KIND='bigvgan-v2-24khz-100band-256x'; TEST_NAME='parity_bigvgan_v2_24khz_100band_256x_real_weight_mel_to_waveform'; GGUF_ENV='VOKRA_BIGVGAN_V2_24KHZ_100BAND_256X_GGUF'; REFERENCE_ENV='VOKRA_BIGVGAN_V2_24KHZ_100BAND_256X_REFERENCE'; EXPECTED_SOURCE_REVISION='7d2b454564a6c7d014227f635b7423881f14bdac'; EXPECTED_MODEL_REVISION='c329ede9e9bbc100ddf5c91e2330a61921262370'; EXPECTED_CHECKPOINT_BYTES='450088331'; EXPECTED_CHECKPOINT_SHA256='6f9c5715550c9d0f11159ceb8935638da5aeb19e27d1e63677632df095e376f5'; EXPECTED_CONFIG_BYTES='1402'; EXPECTED_CONFIG_SHA256='d77e2c96583ca2296ac112a56ec7cc6bd5da4bf7681ceff18448bedc4fcf6512'; EXPECTED_STATUS='IDENTITY_REVIEWED_APPROVAL_PENDING';;
    base_v1_24khz_100band)
      MODEL_REPOSITORY='nvidia/bigvgan_base_24khz_100band'; MODEL_VARIANT='base_v1_24khz_100band'; MODEL_KIND='bigvgan-base-24khz-100band'; TEST_NAME='parity_bigvgan_base_v1_24khz_100band_real_weight_mel_to_waveform'; GGUF_ENV='VOKRA_BIGVGAN_BASE_V1_24KHZ_100BAND_GGUF'; REFERENCE_ENV='VOKRA_BIGVGAN_BASE_V1_24KHZ_100BAND_REFERENCE'; EXPECTED_MODEL_REVISION='0f6305d0e010eaafdbf649978f46c3b5af099343'; EXPECTED_CHECKPOINT_BYTES='56297637'; EXPECTED_CHECKPOINT_SHA256='ca8bced4d3ef588e654742f732455c16abb004e49d7d3bf03edade84d3e982f2'; EXPECTED_CONFIG_BYTES='1034'; EXPECTED_CONFIG_SHA256='885553969751bfd87f1980017364e968917cd34347376ed08238db673ea5b46b'; EXPECTED_SOURCE_REVISION='7d2b454564a6c7d014227f635b7423881f14bdac'; EXPECTED_STATUS='IDENTITY_REVIEWED_APPROVAL_PENDING';;
    *) die "unsupported BigVGAN variant '$1'; see $VARIANT_MANIFEST"; return 2;;
  esac
}

manifest_preflight() {
  [[ -f "$VARIANT_MANIFEST" && ! -L "$VARIANT_MANIFEST" ]] || die "variant manifest is missing or symlinked: $VARIANT_MANIFEST"
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python - \
    "$VARIANT_MANIFEST" "$MODEL_VARIANT" "$MODEL_REPOSITORY" "$MODEL_KIND" \
    "$EXPECTED_MODEL_REVISION" "$EXPECTED_CHECKPOINT_BYTES" "$EXPECTED_CHECKPOINT_SHA256" \
    "$EXPECTED_CONFIG_BYTES" "$EXPECTED_CONFIG_SHA256" \
    "$EXPECTED_SOURCE_REVISION" "$EXPECTED_STATUS" <<'PY'
import json
import sys

path, slug, repository, model_kind, revision, checkpoint_bytes, checkpoint, config_bytes, config, source_revision, status = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    manifest = json.load(stream)
row = next((item for item in manifest["variants"] if item.get("slug") == slug), None)
if row is None:
    raise SystemExit(f"variant manifest has no row for {slug!r}")
expected = {
    "hf_repository": repository,
    "model_kind": model_kind,
    "variant_tag": slug,
    "model_revision": revision or None,
    "checkpoint_bytes": int(checkpoint_bytes) if checkpoint_bytes else None,
    "checkpoint_sha256": checkpoint or None,
    "config_bytes": int(config_bytes) if config_bytes else None,
    "config_sha256": config or None,
}
for key, value in expected.items():
    if row.get(key) != value:
        raise SystemExit(f"variant manifest mismatch for {slug}: {key}={row.get(key)!r}, expected {value!r}")
if manifest.get("source_revision") != source_revision:
    raise SystemExit("variant manifest source revision does not match the selected gate")
if row.get("review_status") != status:
    raise SystemExit(f"variant manifest mismatch for {slug}: review_status={row.get('review_status')!r}, expected {status!r}")
if any(row.get(key) is None for key in ("model_revision", "checkpoint_sha256", "config_sha256")):
    if row["review_status"] != "PRIMARY_SOURCE_REQUIRED":
        raise SystemExit(f"incomplete identity must remain PRIMARY_SOURCE_REQUIRED for {slug}")
elif row["review_status"] != "IDENTITY_REVIEWED_APPROVAL_PENDING":
    raise SystemExit(f"complete identity has unexpected status for {slug}: {row['review_status']!r}")
print(f"bigvgan variant manifest preflight: {slug} {row['review_status']}")
PY
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_regular_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or a symlink: $path"
}

require_disjoint_evidence() {
  local evidence="$1" candidate parent other real other_parent
  if [[ -e "$evidence" || -L "$evidence" ]]; then
    die "evidence directory must be absent before validation: $evidence"
    return 2
  fi
  if ! parent="$(cd -P "$(dirname "$evidence")" 2>/dev/null && pwd)"; then
    die 'evidence parent is inaccessible'
    return 2
  fi
  candidate="$parent/$(basename "$evidence")"
  shift
  for other in "$@"; do
    if [[ -L "$other" ]]; then
      die "validation input is a symlink: $other"
      return 2
    fi
    if ! other_parent="$(cd -P "$(dirname "$other")" 2>/dev/null && pwd)"; then
      die "validation input is inaccessible: $other"
      return 2
    fi
    real="$other_parent/$(basename "$other")"
    if [[ "$candidate" == "$real" || "$candidate/" == "$real/"* || "$real/" == "$candidate/"* ]]; then
      die 'evidence directory overlaps validation input'
      return 2
    fi
  done
  mkdir -p "$evidence"
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die 'required tool missing: uv'
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] || die 'BigVGAN license gate or manifest is missing'
  require_regular_file 'approval evidence' "$approval"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST" --license-evidence "$approval" \
    --source-revision "$EXPECTED_SOURCE_REVISION" --model-revision "$EXPECTED_MODEL_REVISION" \
    --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" --config-sha256 "$EXPECTED_CONFIG_SHA256"
}

require_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'real Metal validation requires Darwin arm64'
  memory_bytes="$(sysctl -n hw.memsize)"; [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die 'invalid memory value'
  (( memory_bytes >= MIN_MEMORY_BYTES )) || die 'Apple memory guard failed'
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"; [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) || die 'Apple disk guard failed'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee sysctl xcrun sw_vers wc tar; do command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"; done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" && -f "$VARIANT_MANIFEST" ]] || die 'not a Vokra checkout or BigVGAN variant matrix is missing'
  [[ -f "$TEST_SOURCE" ]] || die 'BigVGAN real parity source is missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
}

run_self_test() {
  # Self-test tokens intentionally remain literal source contracts.
  # shellcheck disable=SC2016
  local path="${BASH_SOURCE[0]}" fail=0 token cpu_block metal_block identity_variant
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'xcrun -f metal' \
    'parity_bigvgan_real.rs' 'real-weight BigVGAN variant parity' 'BigVGAN Metal' 'variant_manifest.json' \
    'one final readback' 'NO_UPLOAD' 'CPU_ATOL' 'METAL_ATOL' 'archive_sha256=' 'tar -czf' '--gguf-sha256' '--reference-sha256' \
    '--model-revision' '--checkpoint-bytes' '--checkpoint-sha256' '--config-bytes' '--config-sha256' '--source-revision' \
    '--approval-evidence' 'license_preflight' 'approval_scope_sha256' 'require_disjoint_evidence' '! -L' \
    'evidence directory must be absent before validation'; do
    grep -Fq -- "$token" "$path" || { log "self-test FAIL: missing argument/host contract: $token"; fail=1; }
  done
  for identity_variant in v2_22khz_80band_256x v2_44khz_128band_512x v2_24khz_100band_256x base_v1_24khz_100band; do
    configure_variant "$identity_variant"
    manifest_preflight || { log "self-test FAIL: authoritative identity preflight: $identity_variant"; fail=1; }
  done
  configure_variant base_v1_24khz_100band
  cpu_block="$(awk '/env \"\$GGUF_ENV=\$gguf\"/{seen=1} seen {print} seen && /BIGVGAN_CPU_PARITY_SENTINEL/{exit}' "$path")"
  metal_block="$(awk '/env \"\$GGUF_ENV=\$gguf\"/{seen++} seen == 2 {print} seen == 2 && /BIGVGAN_METAL_PARITY_SENTINEL/{exit}' "$path")"
  for token in '"$REFERENCE_ENV=$reference"' '"$TEST_NAME" --exact --nocapture' 'tee "$cpu_log"' \
    'BIGVGAN_CPU_PARITY_SENTINEL'; do
    grep -Fq -- "$token" <<<"$cpu_block" || { log "self-test FAIL: CPU command contract: $token"; fail=1; }
  done
  for token in '"$REFERENCE_ENV=$reference"' '"$TEST_NAME" --exact --nocapture' 'tee "$metal_log"' \
    'BIGVGAN_METAL_PARITY_SENTINEL'; do
    grep -Fq -- "$token" <<<"$metal_block" || { log "self-test FAIL: Metal command contract: $token"; fail=1; }
  done
  for token in 'readback_count' 'forward_with_resident_ops' 'Metal resident' \
    'BIGVGAN_CPU_PARITY_SENTINEL' 'BIGVGAN_METAL_PARITY_SENTINEL'; do
    grep -Fq -- "$token" "$VOKRA_ROOT/crates/vokra-models/src/bigvgan/mod.rs" \
      || grep -Fq -- "$token" "$TEST_SOURCE" \
      || { log "self-test FAIL: resident implementation token missing: $token"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'; fail=1
  fi
  if "$path" --self-test --gguf foo >/dev/null 2>&1; then log 'self-test FAIL: extra self-test argument accepted'; fail=1; fi
  if "$path" --self-test --self-test >/dev/null 2>&1; then log 'self-test FAIL: duplicate self-test accepted'; fail=1; fi
  if "$path" --gguf foo --gguf bar >/dev/null 2>&1; then log 'self-test FAIL: duplicate GGUF accepted'; fail=1; fi
  if "$path" --gguf >/dev/null 2>&1; then log 'self-test FAIL: missing option value accepted'; fail=1; fi
  if "$path" --gguf '' >/dev/null 2>&1; then log 'self-test FAIL: empty option value accepted'; fail=1; fi
  if "$path" --gguf --reference foo >/dev/null 2>&1; then log 'self-test FAIL: leading-dash option value accepted'; fail=1; fi
  if "$path" --variant unknown >/dev/null 2>&1; then log 'self-test FAIL: unknown variant accepted'; fail=1; fi
  if "$path" --unknown-flag >/dev/null 2>&1; then log 'self-test FAIL: unknown flag accepted'; fail=1; fi
  local temporary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-bigvgan-apple.XXXXXX")"
  trap '[[ -n "${temporary:-}" ]] && rm -rf -- "$temporary"' EXIT
  printf 'approval\n' > "$temporary/approval"
  configure_variant v2_22khz_80band_256x
  local identity_args=(
    --variant "$MODEL_VARIANT"
    --gguf "$temporary/gguf" --gguf-sha256 0000000000000000000000000000000000000000000000000000000000000000
    --reference "$temporary/reference" --reference-sha256 0000000000000000000000000000000000000000000000000000000000000000
    --model-revision "$EXPECTED_MODEL_REVISION"
    --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
    --config-sha256 "$EXPECTED_CONFIG_SHA256"
    --source-revision "$EXPECTED_SOURCE_REVISION"
    --approval-evidence "$temporary/approval" --evidence-dir "$temporary/evidence"
  )
  if "$path" "${identity_args[@]}" >/dev/null 2>&1; then
    log 'self-test FAIL: reviewed variant accepted missing checkpoint/config bytes'; fail=1
  fi
  if "$path" "${identity_args[@]}" --checkpoint-bytes "$EXPECTED_CHECKPOINT_BYTES" --config-bytes 1 >/dev/null 2>&1; then
    log 'self-test FAIL: reviewed variant accepted mismatched config bytes'; fail=1
  fi
  if "$path" "${identity_args[@]}" --checkpoint-bytes 489041291 --config-bytes 1403 >/dev/null 2>&1; then
    log 'self-test FAIL: reviewed variant accepted another variant byte identity'; fail=1
  fi
  configure_variant base_v1_24khz_100band
  ln -s approval "$temporary/approval-link"
  if require_regular_file 'self-test symlink approval' "$temporary/approval-link" >/dev/null 2>&1; then
    log 'self-test FAIL: symlink approval accepted'; fail=1
  fi
  if require_disjoint_evidence "$temporary/approval" "$temporary/approval" >/dev/null 2>&1; then
    log 'self-test FAIL: overlapping evidence accepted'; fail=1
  fi
  mkdir "$temporary/preexisting-empty-evidence"
  if require_disjoint_evidence "$temporary/preexisting-empty-evidence" "$temporary/approval" >/dev/null 2>&1; then
    log 'self-test FAIL: pre-existing empty evidence accepted'; fail=1
  fi
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_CPU_PARITY_METRICS variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' > "$temporary/valid.log"
  require_test_pass "$temporary/valid.log" 'BIGVGAN_CPU_PARITY_SENTINEL' || { log 'self-test FAIL: valid test evidence rejected'; fail=1; }
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_CPU_PARITY_METRICS variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' > "$temporary/duplicate.log"
  if require_test_pass "$temporary/duplicate.log" 'BIGVGAN_CPU_PARITY_SENTINEL' >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate sentinel accepted'; fail=1
  fi
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; unexpected' \
    'BIGVGAN_CPU_PARITY_METRICS variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL variant=base_v1_24khz_100band samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' > "$temporary/suffix.log"
  if require_test_pass "$temporary/suffix.log" 'BIGVGAN_CPU_PARITY_SENTINEL' >/dev/null 2>&1; then
    log 'self-test FAIL: malformed result accepted'; fail=1
  fi
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_CPU_PARITY_METRICS variant=base_v1_24khz_100band samples=256 max_abs=0.000020001 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL variant=base_v1_24khz_100band samples=256 max_abs=0.000020001 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' > "$temporary/cpu-over-bound.log"
  if require_test_pass "$temporary/cpu-over-bound.log" BIGVGAN_CPU_PARITY_SENTINEL >/dev/null 2>&1; then
    log 'self-test FAIL: CPU over-bound metric accepted'; fail=1
  fi
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_METAL_PARITY_METRICS variant=base_v1_24khz_100band samples=256 max_abs=0.010000001 atol=0.010000000 route=resident_one_final_readback reference=CPU' \
    'BIGVGAN_METAL_PARITY_SENTINEL variant=base_v1_24khz_100band samples=256 max_abs=0.010000001 atol=0.010000000 route=resident_one_final_readback reference=CPU' > "$temporary/metal-over-bound.log"
  if require_test_pass "$temporary/metal-over-bound.log" BIGVGAN_METAL_PARITY_SENTINEL >/dev/null 2>&1; then
    log 'self-test FAIL: Metal over-bound metric accepted'; fail=1
  fi
  (( fail == 0 )) || return 1
  echo 'apple-silicon-bigvgan.sh self-test: OK'
}

require_test_pass() {
  local output="$1" sentinel="$2" test_count named_count result_count result_lines metric_count sentinel_count bound
  test_count="$(grep -Ev '^test result:' "$output" | grep -Ec '^test ' || true)"
  named_count="$(grep -Ec "^test ${TEST_NAME//./\\.} \.\.\. ok$" "$output" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$output" || true)"
  result_lines="$(grep -Ec '^test result:' "$output" || true)"
  case "$sentinel" in
    BIGVGAN_CPU_PARITY_SENTINEL)
  metric_count="$(grep -Ec "^BIGVGAN_CPU_PARITY_METRICS variant=${MODEL_VARIANT} samples=[0-9]+ max_abs=[0-9]+\\.[0-9]{9} atol=0\\.000020000 reference=NVIDIA\\.BigVGAN fixture=vast_generated_official$" "$output" || true)"
  sentinel_count="$(grep -Ec "^BIGVGAN_CPU_PARITY_SENTINEL variant=${MODEL_VARIANT} samples=[0-9]+ max_abs=[0-9]+\\.[0-9]{9} atol=0\\.000020000 reference=NVIDIA\\.BigVGAN fixture=vast_generated_official$" "$output" || true)"
      bound=0.00002 ;;
    BIGVGAN_METAL_PARITY_SENTINEL)
      metric_count="$(grep -Ec "^BIGVGAN_METAL_PARITY_METRICS variant=${MODEL_VARIANT} samples=[0-9]+ max_abs=[0-9]+\\.[0-9]{9} atol=0\\.010000000 route=resident_one_final_readback reference=CPU$" "$output" || true)"
      sentinel_count="$(grep -Ec "^BIGVGAN_METAL_PARITY_SENTINEL variant=${MODEL_VARIANT} samples=[0-9]+ max_abs=[0-9]+\\.[0-9]{9} atol=0\\.010000000 route=resident_one_final_readback reference=CPU$" "$output" || true)"
      bound=0.01 ;;
    *) die "unknown BigVGAN sentinel: $sentinel"; return 2 ;;
  esac
  [[ "$test_count" == 1 && "$named_count" == 1 && "$result_count" == 1 && "$result_lines" == 1 && "$metric_count" == 1 && "$sentinel_count" == 1 ]] \
    || { die "BigVGAN evidence must contain one exact test/result/metric/sentinel"; return 2; }
  awk -v bound="$bound" '/_METRICS / { for (i = 1; i <= NF; i++) { split($i, pair, "="); if (pair[1] == "max_abs" && (pair[2] + 0) > bound) exit 1 } }' "$output" \
    || { die "BigVGAN metric exceeds registered bound"; return 2; }
}

main() {
  local self_test=0 variant='' gguf='' digest='' reference='' reference_digest='' approval='' evidence='' model_revision='' checkpoint_bytes='' checkpoint_digest='' config_bytes='' config_digest='' source_revision='' cpu_log metal_log archive archive_sha
  local seen_variant=0 seen_gguf=0 seen_digest=0 seen_reference=0 seen_reference_digest=0 seen_approval=0 seen_model=0 seen_checkpoint_bytes=0 seen_checkpoint=0 seen_config_bytes=0 seen_config=0 seen_source=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
      --variant) (( seen_variant == 0 )) || die 'duplicate --variant'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--variant requires a nonempty value'; seen_variant=1; variant="$2"; shift 2 ;;
      --gguf) (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty value'; seen_gguf=1; gguf="$2"; shift 2 ;;
      --gguf-sha256) (( seen_digest == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-sha256 requires a nonempty value'; seen_digest=1; digest="$2"; shift 2 ;;
      --reference) (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty value'; seen_reference=1; reference="$2"; shift 2 ;;
      --reference-sha256) (( seen_reference_digest == 0 )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-sha256 requires a nonempty value'; seen_reference_digest=1; reference_digest="$2"; shift 2 ;;
      --model-revision) (( seen_model == 0 )) || die 'duplicate --model-revision'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--model-revision requires a nonempty value'; seen_model=1; model_revision="$2"; shift 2 ;;
      --checkpoint-bytes) (( seen_checkpoint_bytes == 0 )) || die 'duplicate --checkpoint-bytes'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--checkpoint-bytes requires a nonempty value'; seen_checkpoint_bytes=1; checkpoint_bytes="$2"; shift 2 ;;
      --checkpoint-sha256) (( seen_checkpoint == 0 )) || die 'duplicate --checkpoint-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--checkpoint-sha256 requires a nonempty value'; seen_checkpoint=1; checkpoint_digest="$2"; shift 2 ;;
      --config-bytes) (( seen_config_bytes == 0 )) || die 'duplicate --config-bytes'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--config-bytes requires a nonempty value'; seen_config_bytes=1; config_bytes="$2"; shift 2 ;;
      --config-sha256) (( seen_config == 0 )) || die 'duplicate --config-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--config-sha256 requires a nonempty value'; seen_config=1; config_digest="$2"; shift 2 ;;
      --source-revision) (( seen_source == 0 )) || die 'duplicate --source-revision'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--source-revision requires a nonempty value'; seen_source=1; source_revision="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1; evidence="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test )); then
    [[ -z "$variant$gguf$evidence$approval$digest$reference$reference_digest$model_revision$checkpoint_bytes$checkpoint_digest$config_bytes$config_digest$source_revision" ]] || die '--self-test accepts no other arguments'
    configure_variant base_v1_24khz_100band
    manifest_preflight
    run_self_test; return $?
  fi
  (( seen_variant == 1 )) || { usage; die '--variant is required'; }
  configure_variant "$variant"
  manifest_preflight
  [[ -n "$gguf" && -n "$digest" && -n "$reference" && -n "$reference_digest" && -n "$model_revision" && -n "$checkpoint_digest" && -n "$config_digest" && -n "$source_revision" && -n "$approval" && -n "$evidence" ]] || { usage; die 'all required arguments must be supplied'; }
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 must be 64 lowercase hex characters'
  [[ "$reference_digest" =~ ^[0-9a-f]{64}$ ]] || die '--reference-sha256 must be 64 lowercase hex characters'
  [[ "$model_revision" =~ ^[0-9a-f]{40}$ ]] || die '--model-revision must be a 40-character lowercase revision'
  if [[ -n "$EXPECTED_CHECKPOINT_BYTES" ]]; then
    [[ "$checkpoint_bytes" =~ ^[0-9]+$ ]] || die '--checkpoint-bytes must be an exact decimal byte count'
    [[ "$checkpoint_bytes" == "$EXPECTED_CHECKPOINT_BYTES" ]] || die '--checkpoint-bytes does not match the authoritative manifest'
  fi
  [[ "$checkpoint_digest" =~ ^[0-9a-f]{64}$ ]] || die '--checkpoint-sha256 must be 64 lowercase hex characters'
  if [[ -n "$EXPECTED_CONFIG_BYTES" ]]; then
    [[ "$config_bytes" =~ ^[0-9]+$ ]] || die '--config-bytes must be an exact decimal byte count'
    [[ "$config_bytes" == "$EXPECTED_CONFIG_BYTES" ]] || die '--config-bytes does not match the authoritative manifest'
  fi
  [[ "$config_digest" =~ ^[0-9a-f]{64}$ ]] || die '--config-sha256 must be 64 lowercase hex characters'
  [[ "$source_revision" =~ ^[0-9a-f]{40}$ ]] || die '--source-revision must be a 40-character lowercase revision'
  [[ "$model_revision" == "$EXPECTED_MODEL_REVISION" ]] || die '--model-revision does not match the reviewed HF revision'
  [[ "$checkpoint_digest" == "$EXPECTED_CHECKPOINT_SHA256" ]] || die '--checkpoint-sha256 does not match the reviewed LFS payload'
  [[ "$config_digest" == "$EXPECTED_CONFIG_SHA256" ]] || die '--config-sha256 does not match the reviewed config payload'
  [[ "$source_revision" == "$EXPECTED_SOURCE_REVISION" ]] || die '--source-revision does not match the reviewed source revision'
  license_preflight "$approval"
  require_host; require_tooling
  require_regular_file 'GGUF' "$gguf"
  [[ "$(sha256_file "$gguf")" == "$digest" ]] || die 'GGUF digest mismatch'
  require_regular_file 'reference' "$reference"
  [[ "$(sha256_file "$reference")" == "$reference_digest" ]] || die 'reference digest mismatch'
  require_disjoint_evidence "$evidence" "$VOKRA_ROOT" "$gguf" "$reference" "$approval"
  archive="$evidence.tar.gz"
  [[ ! -e "$archive" && ! -L "$archive" ]] || die 'evidence archive must be absent before validation'
  log_file="$evidence/validation.log"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "model_repository=$MODEL_REPOSITORY"
    echo "model_revision=$model_revision"
    echo "checkpoint_bytes=${checkpoint_bytes:-$EXPECTED_CHECKPOINT_BYTES}"
    echo "checkpoint_sha256=$checkpoint_digest"
    echo "config_bytes=${config_bytes:-$EXPECTED_CONFIG_BYTES}"
    echo "config_sha256=$config_digest"
    echo "source_repository=https://github.com/NVIDIA/BigVGAN"
    echo "source_revision=$source_revision"
    echo "gguf_sha256=$digest"
    echo "reference_sha256=$reference_digest"
    sw_vers
    sysctl -n hw.memsize
    xcrun -f metal
  } > "$log_file"
  cpu_log="$evidence/cpu.log"
  metal_log="$evidence/metal.log"
  env "$GGUF_ENV=$gguf" "$REFERENCE_ENV=$reference" CARGO_BUILD_JOBS=1 \
    cargo test --locked --release -p vokra-models --test parity_bigvgan_real -- \
    "$TEST_NAME" --exact --nocapture 2>&1 | tee "$cpu_log" | tee -a "$log_file"
  require_test_pass "$cpu_log" BIGVGAN_CPU_PARITY_SENTINEL
  env "$GGUF_ENV=$gguf" "$REFERENCE_ENV=$reference" CARGO_BUILD_JOBS=1 \
    cargo test --locked --release -p vokra-models --features metal --test parity_bigvgan_real -- \
    "$TEST_NAME" --exact --nocapture 2>&1 | tee "$metal_log" | tee -a "$log_file"
  require_test_pass "$metal_log" BIGVGAN_METAL_PARITY_SENTINEL
  printf 'execution_status=PASS\nmodel_repository=%s\nmodel_variant=%s\nmodel_revision=%s\ncheckpoint_bytes=%s\ncheckpoint_sha256=%s\nconfig_bytes=%s\nconfig_sha256=%s\nsource_repository=https://github.com/NVIDIA/BigVGAN\nsource_revision=%s\ngguf_sha256=%s\nreference_sha256=%s\nregistered_cpu_atol=%s\nregistered_metal_atol=%s\nmetal_route=RESIDENT_ONE_FINAL_READBACK\npublication=NO_UPLOAD\n' \
    "$MODEL_REPOSITORY" "$MODEL_VARIANT" "$model_revision" "${checkpoint_bytes:-$EXPECTED_CHECKPOINT_BYTES}" "$checkpoint_digest" "${config_bytes:-$EXPECTED_CONFIG_BYTES}" "$config_digest" "$source_revision" "$digest" "$reference_digest" "$CPU_ATOL" "$METAL_ATOL" > "$evidence/summary.txt"
  tar -czf "$archive" -C "$(dirname "$evidence")" "$(basename "$evidence")"
  archive_sha="$(sha256_file "$archive")"
  printf 'archive=%s\narchive_sha256=%s\n' "$archive" "$archive_sha" >> "$evidence/summary.txt"
  log "PASS: evidence written to $evidence"
}

main "$@"
