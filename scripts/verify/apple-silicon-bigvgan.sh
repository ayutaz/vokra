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
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_bigvgan_real.rs"
MODEL_REPOSITORY="nvidia/bigvgan_base_24khz_100band"
TEST_NAME="parity_bigvgan_base_real_weight_mel_to_waveform"
EXPECTED_MODEL_REVISION="0f6305d0e010eaafdbf649978f46c3b5af099343"
EXPECTED_CHECKPOINT_SHA256="ca8bced4d3ef588e654742f732455c16abb004e49d7d3bf03edade84d3e982f2"
EXPECTED_CONFIG_SHA256="885553969751bfd87f1980017364e968917cd34347376ed08238db673ea5b46b"
EXPECTED_SOURCE_REVISION="7d2b454564a6c7d014227f635b7423881f14bdac"
MIN_MEMORY_BYTES=16000000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[bigvgan-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-bigvgan.sh --gguf <file> --gguf-sha256 <64-hex> \
       --reference <file> --reference-sha256 <64-hex> \
       --model-revision <40-hex> --checkpoint-sha256 <64-hex> \
       --config-sha256 <64-hex> --source-revision <40-hex> \
       --approval-evidence <file> \
       --evidence-dir <absent-dir>
       apple-silicon-bigvgan.sh --self-test

Runs the exact real-weight BigVGAN base parity test once on CPU and once with
the real Metal feature. The model binder enforces its resident route and one
final readback internally. The GGUF hash is mandatory and must come from the
VAST conversion worker; no model conversion or network operation is performed.
The evidence directory must be absent/nonexistent before validation; it is
created only after all input and approval checks succeed.
EOF
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
  for tool in cargo rustc git shasum awk find tee sysctl xcrun sw_vers wc; do command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"; done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die 'not a Vokra checkout'
  [[ -f "$TEST_SOURCE" ]] || die 'BigVGAN real parity source is missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
}

run_self_test() {
  # Self-test tokens intentionally remain literal source contracts.
  # shellcheck disable=SC2016
  local path="${BASH_SOURCE[0]}" fail=0 token cpu_block metal_block
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'xcrun -f metal' \
    'parity_bigvgan_real.rs' 'real-weight BigVGAN base parity' 'BigVGAN Metal' \
    'one final readback' 'NO_UPLOAD' '--gguf-sha256' '--reference-sha256' \
    '--model-revision' '--checkpoint-sha256' '--config-sha256' '--source-revision' \
    '--approval-evidence' 'license_preflight' 'require_disjoint_evidence' '! -L' \
    'evidence directory must be absent before validation'; do
    grep -Fq -- "$token" "$path" || { log "self-test FAIL: missing argument/host contract: $token"; fail=1; }
  done
  cpu_block="$(awk '/^  VOKRA_BIGVGAN_BASE_GGUF=.*VOKRA_BIGVGAN_REFERENCE=/{seen=1} seen {print} seen && /BIGVGAN_CPU_PARITY_SENTINEL/{exit}' "$path")"
  metal_block="$(awk '/^  VOKRA_BIGVGAN_BASE_GGUF=.*VOKRA_BIGVGAN_REFERENCE=/{seen++} seen == 2 {print} seen == 2 && /BIGVGAN_METAL_PARITY_SENTINEL/{exit}' "$path")"
  for token in 'VOKRA_BIGVGAN_REFERENCE=' '"$TEST_NAME" --exact --nocapture' 'tee "$cpu_log"' \
    'BIGVGAN_CPU_PARITY_SENTINEL'; do
    grep -Fq -- "$token" <<<"$cpu_block" || { log "self-test FAIL: CPU command contract: $token"; fail=1; }
  done
  for token in 'VOKRA_BIGVGAN_REFERENCE=' '"$TEST_NAME" --exact --nocapture' 'tee "$metal_log"' \
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
  if "$path" --unknown-flag >/dev/null 2>&1; then log 'self-test FAIL: unknown flag accepted'; fail=1; fi
  local temporary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-bigvgan-apple.XXXXXX")"
  trap '[[ -n "${temporary:-}" ]] && rm -rf -- "$temporary"' EXIT
  printf 'approval\n' > "$temporary/approval"
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
    'BIGVGAN_CPU_PARITY_SENTINEL max_abs=1.0e-5' > "$temporary/valid.log"
  require_test_pass "$temporary/valid.log" 'BIGVGAN_CPU_PARITY_SENTINEL' || { log 'self-test FAIL: valid test evidence rejected'; fail=1; }
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_CPU_PARITY_SENTINEL max_abs=1.0e-5' \
    'BIGVGAN_CPU_PARITY_SENTINEL max_abs=1.0e-5' > "$temporary/duplicate.log"
  if require_test_pass "$temporary/duplicate.log" 'BIGVGAN_CPU_PARITY_SENTINEL' >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate sentinel accepted'; fail=1
  fi
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; unexpected' \
    'BIGVGAN_CPU_PARITY_SENTINEL max_abs=1.0e-5' > "$temporary/suffix.log"
  if require_test_pass "$temporary/suffix.log" 'BIGVGAN_CPU_PARITY_SENTINEL' >/dev/null 2>&1; then
    log 'self-test FAIL: malformed result accepted'; fail=1
  fi
  (( fail == 0 )) || return 1
  echo 'apple-silicon-bigvgan.sh self-test: OK'
}

require_test_pass() {
  local output="$1" sentinel="$2" test_count named_count result_count result_lines sentinel_count
  test_count="$(grep -Ev '^test result:' "$output" | grep -Ec '^test ' || true)"
  named_count="$(grep -Ec "^test ${TEST_NAME//./\\.} \.\.\. ok$" "$output" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$output" || true)"
  result_lines="$(grep -Ec '^test result:' "$output" || true)"
  case "$sentinel" in
    BIGVGAN_CPU_PARITY_SENTINEL) sentinel_count="$(grep -Ec '^BIGVGAN_CPU_PARITY_SENTINEL max_abs=[0-9.eE+-]+$' "$output" || true)" ;;
    BIGVGAN_METAL_PARITY_SENTINEL) sentinel_count="$(grep -Ec '^BIGVGAN_METAL_PARITY_SENTINEL max_abs=[0-9.eE+-]+ route=resident_one_final_readback$' "$output" || true)" ;;
    *) die "unknown BigVGAN sentinel: $sentinel"; return 2 ;;
  esac
  [[ "$test_count" == 1 && "$named_count" == 1 && "$result_count" == 1 && "$result_lines" == 1 && "$sentinel_count" == 1 ]] \
    || { die "BigVGAN evidence must contain one exact test/result/sentinel"; return 2; }
}

main() {
  local self_test=0 gguf='' digest='' reference='' reference_digest='' approval='' evidence='' model_revision='' checkpoint_digest='' config_digest='' source_revision='' cpu_log metal_log
  local seen_gguf=0 seen_digest=0 seen_reference=0 seen_reference_digest=0 seen_approval=0 seen_model=0 seen_checkpoint=0 seen_config=0 seen_source=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
      --gguf) (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty value'; seen_gguf=1; gguf="$2"; shift 2 ;;
      --gguf-sha256) (( seen_digest == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-sha256 requires a nonempty value'; seen_digest=1; digest="$2"; shift 2 ;;
      --reference) (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty value'; seen_reference=1; reference="$2"; shift 2 ;;
      --reference-sha256) (( seen_reference_digest == 0 )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-sha256 requires a nonempty value'; seen_reference_digest=1; reference_digest="$2"; shift 2 ;;
      --model-revision) (( seen_model == 0 )) || die 'duplicate --model-revision'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--model-revision requires a nonempty value'; seen_model=1; model_revision="$2"; shift 2 ;;
      --checkpoint-sha256) (( seen_checkpoint == 0 )) || die 'duplicate --checkpoint-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--checkpoint-sha256 requires a nonempty value'; seen_checkpoint=1; checkpoint_digest="$2"; shift 2 ;;
      --config-sha256) (( seen_config == 0 )) || die 'duplicate --config-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--config-sha256 requires a nonempty value'; seen_config=1; config_digest="$2"; shift 2 ;;
      --source-revision) (( seen_source == 0 )) || die 'duplicate --source-revision'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--source-revision requires a nonempty value'; seen_source=1; source_revision="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1; evidence="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test )); then
    [[ -z "$gguf$evidence$approval$digest$reference$reference_digest$model_revision$checkpoint_digest$config_digest$source_revision" ]] || die '--self-test accepts no other arguments'
    run_self_test; return $?
  fi
  [[ -n "$gguf" && -n "$digest" && -n "$reference" && -n "$reference_digest" && -n "$model_revision" && -n "$checkpoint_digest" && -n "$config_digest" && -n "$source_revision" && -n "$approval" && -n "$evidence" ]] || { usage; die 'all required arguments must be supplied'; }
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 must be 64 lowercase hex characters'
  [[ "$reference_digest" =~ ^[0-9a-f]{64}$ ]] || die '--reference-sha256 must be 64 lowercase hex characters'
  [[ "$model_revision" =~ ^[0-9a-f]{40}$ ]] || die '--model-revision must be a 40-character lowercase revision'
  [[ "$checkpoint_digest" =~ ^[0-9a-f]{64}$ ]] || die '--checkpoint-sha256 must be 64 lowercase hex characters'
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
  log_file="$evidence/validation.log"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "model_repository=$MODEL_REPOSITORY"
    echo "model_revision=$model_revision"
    echo "checkpoint_sha256=$checkpoint_digest"
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
  VOKRA_BIGVGAN_BASE_GGUF="$gguf" VOKRA_BIGVGAN_REFERENCE="$reference" CARGO_BUILD_JOBS=1 \
    cargo test --locked --release -p vokra-models --test parity_bigvgan_real -- \
    "$TEST_NAME" --exact --nocapture 2>&1 | tee "$cpu_log" | tee -a "$log_file"
  require_test_pass "$cpu_log" BIGVGAN_CPU_PARITY_SENTINEL
  VOKRA_BIGVGAN_BASE_GGUF="$gguf" VOKRA_BIGVGAN_REFERENCE="$reference" CARGO_BUILD_JOBS=1 \
    cargo test --locked --release -p vokra-models --features metal --test parity_bigvgan_real -- \
    "$TEST_NAME" --exact --nocapture 2>&1 | tee "$metal_log" | tee -a "$log_file"
  require_test_pass "$metal_log" BIGVGAN_METAL_PARITY_SENTINEL
  printf 'execution_status=PASS\nmodel_repository=%s\nmodel_revision=%s\ncheckpoint_sha256=%s\nconfig_sha256=%s\nsource_repository=https://github.com/NVIDIA/BigVGAN\nsource_revision=%s\ngguf_sha256=%s\nreference_sha256=%s\nmetal_route=RESIDENT_ONE_FINAL_READBACK\npublication=NO_UPLOAD\n' \
    "$MODEL_REPOSITORY" "$model_revision" "$checkpoint_digest" "$config_digest" "$source_revision" "$digest" "$reference_digest" > "$evidence/summary.txt"
  log "PASS: evidence written to $evidence"
}

main "$@"
