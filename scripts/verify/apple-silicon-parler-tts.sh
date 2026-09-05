#!/usr/bin/env bash
# Real-weight Parler-TTS Mini English/Multilingual CPU/reference/Metal parity
# on a disposable Apple Silicon host. Inputs are staged by VAST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/parler_tts"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
DUMPER="$PARITY_PROJECT/dump_reference.py"
TRANSFORMERS_COMPATIBILITY_STATUS="BLOCKED_UNVERIFIED_API_SMOKE"
TRANSFORMERS_SECURITY_ADVISORY="GHSA-xrqw-3rrv-vx5w"

ENGLISH_PUBLIC_BYTES=3511459168
ENGLISH_PUBLIC_SHA256="7f69b811edae6cbe82fdfa8e72e6181945d4466748349aa74d994fb566785ddc"
MULTILINGUAL_PUBLIC_BYTES=3751292736
MULTILINGUAL_PUBLIC_SHA256="d1edf792305a486192be73dfb279891febb6e81735abf06b2ae90b29da94134d"
PARLER_SOURCE_REVISION="d108732cd57788ec86bc857d99a6cabd66663d68"
DAC_REPO="parler-tts/dac_44khZ_8kbps"
DAC_REVISION="5cf6b8ad50fbb17e52c341410a1d00083201b6a9"
ENGLISH_UPSTREAM_REPO="parler-tts/parler-tts-mini-v1"
MULTILINGUAL_UPSTREAM_REPO="parler-tts/parler-tts-mini-multilingual-v1.1"
ENGLISH_UPSTREAM_REVISION="0392b9451a601e528fd863bbb0598431fee810d9"
MULTILINGUAL_UPSTREAM_REVISION="11b27d57855dec1ce0914ba1f12363bf2ea75ba3"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_parler_tts_real.rs"
TEST_TARGET="parity_parler_tts_real"
ENGLISH_TEST="real_parler_english_matches_official"
MULTILINGUAL_TEST="real_parler_multilingual_matches_official"

log() { printf '[parler-tts-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

require_transformers_api_smoke() {
  case "$TRANSFORMERS_COMPATIBILITY_STATUS" in
    BLOCKED_UNVERIFIED_API_SMOKE)
      die "Parler Transformers route is BLOCKED_UNVERIFIED_API_SMOKE; authorized API smoke is required before gate, host, download, or model work"
      ;;
    AUTHENTICATED_API_SMOKE) ;;
    *) die "Parler Transformers compatibility status is not reviewed: $TRANSFORMERS_COMPATIBILITY_STATUS" ;;
  esac
}

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-parler-tts.sh \
  --english-gguf <parler-tts-mini-v1.gguf> \
  --english-reference <dir> \
  --english-reference-sha256 <64-hex> \
  --multilingual-gguf <parler-tts-mini-multilingual.gguf> \
  --multilingual-reference <dir> \
  --multilingual-reference-sha256 <64-hex> \
  --approval-evidence <file> \
  --evidence-dir <absent-dir>
       apple-silicon-parler-tts.sh --self-test

Runs the exact real-weight Parler-TTS Mini English and Mini Multilingual tests
once on CPU and once on real Metal. Each test must compare FLAN-T5 hidden
states, generated DAC codes, and decoded PCM to the independent official
reference. The verifier also requires a real test-emitted Metal-vs-CPU PASS
marker; it never creates that marker from two unrelated successful runs.

The host must be a disposable Darwin/arm64 checkout with
VOKRA_REMOTE_APPLE_SILICON=1, at least 32 GB physical memory, 20 GB free disk,
and Xcode's Metal compiler. Inputs are VAST-produced or VAST-authenticated.
The evidence directory must be absent/nonexistent before validation; it is
created only after all input and approval checks succeed.
This script does not download, convert, upload, publish, or delete models.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
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
  [[ -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die 'Parler preflight gate or manifest is missing'
  require_file 'approval evidence' "$approval"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 "$PREFLIGHT_GATE" \
    --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_reference_manifest_digest() {
  local label="$1" manifest="$2" expected="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { die "$label expected manifest SHA-256 is missing or malformed"; return 2; }
  require_file "$label manifest" "$manifest"
  actual="$(sha256_file "$manifest")"
  [[ "$actual" == "$expected" ]] || { die "$label manifest SHA-256 $actual != VAST evidence $expected"; return 2; }
}

manifest_file_hash() {
  local manifest="$1" name="$2" matches value
  matches="$(grep -Eo '"'"$name"'"[[:space:]]*:[[:space:]]*"[0-9a-f]{64}"' "$manifest" || true)"
  [[ "$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l | tr -d '[:space:]')" == 1 ]] \
    || { die "manifest has no unique hash for $name"; return 2; }
  value="${matches##*: \"}"
  printf '%s\n' "${value%\"}"
}

verify_reference_payload_hashes() {
  local label="$1" directory="$2" name expected actual
  local manifest="$directory/manifest.json"
  for name in description_token_ids.u32le prompt_token_ids.u32le text_hidden.f32 codes.u32le decoded_pcm.f32; do
    expected="$(manifest_file_hash "$manifest" "$name")" || return 2
    actual="$(sha256_file "$directory/$name")"
    [[ "$actual" == "$expected" ]] || { die "$label $name SHA-256 $actual != manifest $expected"; return 2; }
  done
}

require_reference() {
  local label="$1" directory="$2" variant="$3" revision upstream_repo
  [[ -d "$directory" && ! -L "$directory" ]] || die "$label is not a regular directory: $directory"
  for name in manifest.json description_token_ids.u32le prompt_token_ids.u32le \
    text_hidden.f32 codes.u32le decoded_pcm.f32; do
    require_file "$label $name" "$directory/$name"
  done
  case "$variant" in
    english) revision="$ENGLISH_UPSTREAM_REVISION"; upstream_repo="$ENGLISH_UPSTREAM_REPO" ;;
    multilingual) revision="$MULTILINGUAL_UPSTREAM_REVISION"; upstream_repo="$MULTILINGUAL_UPSTREAM_REPO" ;;
    *) die "unknown Parler variant: $variant" ;;
  esac
  grep -Fq '"format": "vokra-parler-tts-official-reference-v1"' \
    "$directory/manifest.json" \
    || die "$label manifest is not the pinned official Parler reference format"
  grep -Fq '"variant": "'"$variant"'"' "$directory/manifest.json" \
    || die "$label manifest has the wrong variant"
  grep -Fq '"upstream_hf": "'"$upstream_repo"'"' "$directory/manifest.json" \
    || die "$label manifest has the wrong upstream model identity"
  grep -Fq '"upstream_revision": "'"$revision"'"' \
    "$directory/manifest.json" \
    || die "$label manifest lost upstream revision $revision"
  grep -Fq '"parler_source_revision": "'"$PARLER_SOURCE_REVISION"'"' \
    "$directory/manifest.json" \
    || die "$label manifest lost official parler-tts source revision"
  grep -Fq '"dac_repo": "'"$DAC_REPO"'"' "$directory/manifest.json" \
    || die "$label manifest lost DAC repository identity"
  grep -Fq '"dac_revision": "'"$DAC_REVISION"'"' "$directory/manifest.json" \
    || die "$label manifest lost DAC revision identity"
  grep -Fq '"transformers_version": "5.10.4"' "$directory/manifest.json" \
    || die "$label manifest lost the locked Transformers 5.10.4 oracle"
}

verify_public_gguf() {
  local label="$1" path="$2" expected_bytes="$3" expected_sha="$4"
  local actual_bytes actual_sha
  require_file "$label" "$path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label byte size $actual_bytes != exact public artifact $expected_bytes"
  actual_sha="$(sha256_file "$path")"
  [[ "$actual_sha" == "$expected_sha" ]] \
    || die "$label SHA-256 $actual_sha != exact public artifact $expected_sha"
}

hash_reference_tree() {
  local directory="$1" output="$2" path relative
  : > "$output"
  while IFS= read -r path; do
    relative="${path#"$directory"/}"
    printf '%s  %s\n' "$(sha256_file "$path")" "$relative" >> "$output"
  done < <(find "$directory" -type f -print | sort)
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "real Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "real Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the 20-GB run guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun wc tr sort; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra checkout"
  [[ -f "$TEST_SOURCE" ]] || die "Parler parity source is missing: $TEST_SOURCE"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "machine=$(uname -m)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPHardwareDataType
    system_profiler SPDisplaysDataType
  } > "$output"
}

run_self_test() (
  local temporary script_path required manifest_digest api_line license_line
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-parler-tts-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  [[ -f "$DUMPER" && ! -L "$DUMPER" ]] || die 'Parler dumper is missing'
  grep -Fq -- 'BLOCKED_UNVERIFIED_API_SMOKE' "$DUMPER" || die 'Parler dumper lost API smoke blocker'
  grep -Fq -- 'TRANSFORMERS_SECURITY_ADVISORY' "$DUMPER" || die 'Parler dumper lost security advisory contract'
  if require_transformers_api_smoke >/dev/null 2>&1; then
    die 'blocked Transformers API smoke route was accepted'
  fi
  local original_transformers_status="$TRANSFORMERS_COMPATIBILITY_STATUS"
  TRANSFORMERS_COMPATIBILITY_STATUS="AUTHENTICATED_API_SMOKE"
  require_transformers_api_smoke || die 'authenticated Transformers API smoke route was rejected'
  TRANSFORMERS_COMPATIBILITY_STATUS="UNREVIEWED_STATUS"
  if require_transformers_api_smoke >/dev/null 2>&1; then
    die 'unknown Transformers API smoke status was accepted'
  fi
  TRANSFORMERS_COMPATIBILITY_STATUS="$original_transformers_status"
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  ln -s value "$temporary/symlink"
  if require_file 'self-test symlink approval' "$temporary/symlink" >/dev/null 2>&1; then
    die 'symlink approval input was accepted'
  fi
  if require_disjoint_evidence "$temporary/value" "$temporary/value" >/dev/null 2>&1; then
    die 'pre-existing/overlapping evidence path was accepted'
  fi
  mkdir "$temporary/preexisting-empty-evidence"
  if require_disjoint_evidence "$temporary/preexisting-empty-evidence" "$temporary/value" >/dev/null 2>&1; then
    die 'pre-existing empty evidence directory was accepted'
  fi
  ln -s value "$temporary/evidence-link"
  if require_disjoint_evidence "$temporary/evidence-link" "$temporary/value" >/dev/null 2>&1; then
    die 'symlink evidence directory was accepted'
  fi
  manifest_digest="$(sha256_file "$temporary/value")"
  require_reference_manifest_digest 'self-test valid' "$temporary/value" "$manifest_digest"
  if require_reference_manifest_digest 'self-test mismatch' "$temporary/value" "$(printf '%064d' 0)" >/dev/null 2>&1; then
    die "manifest digest mismatch was accepted"
  fi
  if require_reference_manifest_digest 'self-test missing' "$temporary/missing" "$manifest_digest" >/dev/null 2>&1; then
    die "missing manifest was accepted"
  fi
  mkdir "$temporary/reference"
  for name in description_token_ids.u32le prompt_token_ids.u32le text_hidden.f32 codes.u32le decoded_pcm.f32; do
    printf 'self-test-%s\n' "$name" > "$temporary/reference/$name"
  done
  {
    printf '{"files": {'
    first=1
    for name in description_token_ids.u32le prompt_token_ids.u32le text_hidden.f32 codes.u32le decoded_pcm.f32; do
      (( first )) || printf ', '
      first=0
      printf '\"%s\": \"%s\"' "$name" "$(sha256_file "$temporary/reference/$name")"
    done
    printf '}}\n'
  } > "$temporary/reference/manifest.json"
  verify_reference_payload_hashes 'self-test' "$temporary/reference"
  printf 'tampered\n' >> "$temporary/reference/codes.u32le"
  if verify_reference_payload_hashes 'self-test tamper' "$temporary/reference" >/dev/null 2>&1; then
    die "manifest payload tamper was accepted"
  fi
  require_disjoint_evidence "$temporary/evidence" "$temporary/value"
  script_path="${BASH_SOURCE[0]}"
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' \
    'xcrun -f metal' 'parity_parler_tts_real.rs' \
    'real_parler_english_matches_official' \
    'real_parler_multilingual_matches_official' \
    '--features metal --test' '-- --exact --nocapture' \
    'VOKRA_PARLER_BACKEND=cpu' 'VOKRA_PARLER_BACKEND=metal' \
    'PARLER_APPLE_PARITY variant=english metal_vs_cpu=PASS' \
    'PARLER_APPLE_PARITY variant=multilingual metal_vs_cpu=PASS' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' \
    '--english-reference-sha256' '--multilingual-reference-sha256' '--approval-evidence' \
    'license_preflight' 'require_transformers_api_smoke' 'BLOCKED_UNVERIFIED_API_SMOKE' \
    'GHSA-xrqw-3rrv-vx5w' 'dump_reference.py' 'require_disjoint_evidence' '! -L' \
    'evidence directory must be absent before validation' 'upload=NOT_PERFORMED'; do
    grep -Fq -- "$required" "$script_path" \
      || die "self-test contract token is missing: $required"
  done
  api_line="$(grep -n '^  require_transformers_api_smoke$' "$script_path" | tail -1 | cut -d: -f1)"
  license_line="$(grep -n '^  license_preflight ' "$script_path" | tail -1 | cut -d: -f1)"
  [[ "$api_line" =~ ^[0-9]+$ && "$license_line" =~ ^[0-9]+$ && "$api_line" -lt "$license_line" ]] \
    || die 'self-test API smoke gate is not before license preflight'
  [[ "$(grep -Ec '^[[:space:]]+--english-gguf\)' "$script_path" || true)" == 1 ]] \
    || die "English GGUF parser arm is duplicated or missing"
  printf '%s\n' \
    'test real_parler_english_matches_official ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'Parler ENGLISH Cpu: frames=4, T5_max_abs=1.0e-3, T5_rmse=1.0e-4, codes=exact, decode_max_abs=1.0e-3, decode_rmse=1.0e-4, end_to_end_max_abs=1.0e-3, end_to_end_rmse=1.0e-4' \
    'PARLER_APPLE_PARITY variant=english metal_vs_cpu=PASS' > "$temporary/log"
  require_one_named_test_passed "$temporary/log" real_parler_english_matches_official
  require_exact_sentinel "$temporary/log" ENGLISH Cpu
  for malformed in duplicate prefix suffix FAIL; do
    cp "$temporary/log" "$temporary/$malformed.log"
    case "$malformed" in
      duplicate) printf '%s\n' 'Parler ENGLISH Cpu: frames=4, T5_max_abs=1.0e-3, T5_rmse=1.0e-4, codes=exact, decode_max_abs=1.0e-3, decode_rmse=1.0e-4, end_to_end_max_abs=1.0e-3, end_to_end_rmse=1.0e-4' >> "$temporary/$malformed.log" ;;
      prefix) sed 's/^Parler /prefix Parler /' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      suffix) sed 's/$/ trailing/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
      FAIL) sed 's/codes=exact/codes=FAIL/' "$temporary/$malformed.log" > "$temporary/$malformed.tmp" && mv "$temporary/$malformed.tmp" "$temporary/$malformed.log" ;;
    esac
    if require_exact_sentinel "$temporary/$malformed.log" ENGLISH Cpu >/dev/null 2>&1; then
      die "malformed $malformed sentinel was accepted"
    fi
  done
  printf '%s\n' 'test real_parler_english_matches_official ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; nope' > "$temporary/malformed.log"
  if require_one_named_test_passed "$temporary/malformed.log" real_parler_english_matches_official >/dev/null 2>&1; then
    die "malformed Cargo result suffix was accepted"
  fi
  if grep -En -- '(^|[[:space:]])(curl|wget|python3?|pip|git[[:space:]]+(clone|fetch|pull|push)|.*(upload|publish|convert))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    die "download, conversion, or publication command found"
  fi
  if "$script_path" --self-test --english-gguf "$temporary/model.gguf" \
    >/dev/null 2>&1; then
    die "--self-test accepted an extra argument"
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then
    die "duplicate --self-test was accepted"
  fi
  if "$script_path" --english-gguf >/dev/null 2>&1; then die 'missing option value was accepted'; fi
  if "$script_path" --english-gguf '' >/dev/null 2>&1; then die 'empty option value was accepted'; fi
  if "$script_path" --english-gguf --english-reference value >/dev/null 2>&1; then die 'leading-dash option value was accepted'; fi
  if "$script_path" --unknown-option >/dev/null 2>&1; then die 'unknown option was accepted'; fi
  log "self-test PASS"
)

run_variant() {
  local variant="$1" backend="$2" gguf="$3" reference="$4" output="$5"
  local test_name="$6" variant_label backend_label
  case "$variant" in
    english) variant_label='ENGLISH' ;;
    multilingual) variant_label='MULTILINGUAL' ;;
    *) die "unknown Parler variant: $variant" ;;
  esac
  case "$backend" in
    cpu) backend_label='Cpu' ;;
    metal) backend_label='Metal' ;;
    *) die "unknown Parler backend: $backend" ;;
  esac
  env VOKRA_PARLER_ENGLISH_GGUF="$gguf" \
    VOKRA_PARLER_ENGLISH_PARITY_DIR="$reference" \
    VOKRA_PARLER_MULTILINGUAL_GGUF="$gguf" \
    VOKRA_PARLER_MULTILINGUAL_PARITY_DIR="$reference" \
    VOKRA_PARLER_BACKEND="$backend" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test "$TEST_TARGET" "$test_name" \
      -- --exact --nocapture --test-threads=1 2>&1 | tee "$output"
  require_one_named_test_passed "$output" "$test_name"
  require_exact_sentinel "$output" "$variant_label" "$backend_label"
}

require_one_named_test_passed() {
  local output="$1" test_name="$2" count
  count="$(grep -Ev '^test result:' "$output" | grep -Ec '^test ' || true)"
  [[ "$count" == 1 ]] || { die "$test_name output contains extra test lines"; return 2; }
  count="$(grep -Ec "^test ${test_name//./\\.} \.\.\. ok$" "$output" || true)"
  [[ "$count" == 1 ]] || { die "$test_name did not have exactly one passing test line"; return 2; }
  count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$output" || true)"
  [[ "$count" == 1 ]] || { die "$test_name did not have exactly one exact Cargo result"; return 2; }
  count="$(grep -Ec '^test result:' "$output" || true)"
  [[ "$count" == 1 ]] || { die "$test_name has more than one Cargo result line"; return 2; }
}

require_exact_sentinel() {
  local output="$1" variant="$2" backend="$3" count
  count="$(grep -Ec "^Parler ${variant} ${backend}: frames=[0-9]+, T5_max_abs=[0-9]+([.][0-9]+)?e[+-][0-9]+, T5_rmse=[0-9]+([.][0-9]+)?e[+-][0-9]+, codes=exact, decode_max_abs=[0-9]+([.][0-9]+)?e[+-][0-9]+, decode_rmse=[0-9]+([.][0-9]+)?e[+-][0-9]+, end_to_end_max_abs=[0-9]+([.][0-9]+)?e[+-][0-9]+, end_to_end_rmse=[0-9]+([.][0-9]+)?e[+-][0-9]+$" "$output" || true)"
  [[ "$count" == 1 ]] || { die "Parler ${variant} ${backend} dynamic PASS sentinel is not exactly one anchored line"; return 2; }
}

main() {
  local english_gguf='' english_reference='' english_reference_sha256=''
  local multilingual_gguf='' multilingual_reference='' multilingual_reference_sha256=''
  local evidence_dir='' approval_evidence='' self_test=0
  local seen_english_gguf=0 seen_english_reference=0 seen_english_sha=0 seen_multilingual_gguf=0 seen_multilingual_reference=0 seen_multilingual_sha=0 seen_approval=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --english-gguf) (( seen_english_gguf == 0 )) || die 'duplicate --english-gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--english-gguf requires a nonempty value'; seen_english_gguf=1; english_gguf="$2"; shift 2 ;;
      --english-reference) (( seen_english_reference == 0 )) || die 'duplicate --english-reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--english-reference requires a nonempty value'; seen_english_reference=1; english_reference="$2"; shift 2 ;;
      --english-reference-sha256) (( seen_english_sha == 0 )) || die 'duplicate --english-reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--english-reference-sha256 requires a nonempty value'; seen_english_sha=1; english_reference_sha256="$2"; shift 2 ;;
      --multilingual-gguf) (( seen_multilingual_gguf == 0 )) || die 'duplicate --multilingual-gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--multilingual-gguf requires a nonempty value'; seen_multilingual_gguf=1; multilingual_gguf="$2"; shift 2 ;;
      --multilingual-reference) (( seen_multilingual_reference == 0 )) || die 'duplicate --multilingual-reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--multilingual-reference requires a nonempty value'; seen_multilingual_reference=1; multilingual_reference="$2"; shift 2 ;;
      --multilingual-reference-sha256) (( seen_multilingual_sha == 0 )) || die 'duplicate --multilingual-reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--multilingual-reference-sha256 requires a nonempty value'; seen_multilingual_sha=1; multilingual_reference_sha256="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$english_gguf$english_reference$english_reference_sha256$multilingual_gguf$multilingual_reference$multilingual_reference_sha256$approval_evidence$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$english_gguf" && -n "$english_reference" && \
    -n "$english_reference_sha256" && -n "$multilingual_gguf" && -n "$multilingual_reference" && \
    -n "$multilingual_reference_sha256" && -n "$approval_evidence" && -n "$evidence_dir" ]] \
    || { usage; die "all GGUF/reference pairs and --evidence-dir are required"; }

  require_transformers_api_smoke
  license_preflight "$approval_evidence"
  require_remote_apple_host
  require_tooling
  verify_public_gguf 'Parler English GGUF' "$english_gguf" "$ENGLISH_PUBLIC_BYTES" "$ENGLISH_PUBLIC_SHA256"
  verify_public_gguf 'Parler Multilingual GGUF' "$multilingual_gguf" "$MULTILINGUAL_PUBLIC_BYTES" "$MULTILINGUAL_PUBLIC_SHA256"
  require_reference_manifest_digest 'Parler English' "$english_reference/manifest.json" "$english_reference_sha256"
  require_reference_manifest_digest 'Parler Multilingual' "$multilingual_reference/manifest.json" "$multilingual_reference_sha256"
  require_reference 'Parler English reference' "$english_reference" english
  require_reference 'Parler Multilingual reference' "$multilingual_reference" multilingual
  verify_reference_payload_hashes 'Parler English reference' "$english_reference"
  verify_reference_payload_hashes 'Parler Multilingual reference' "$multilingual_reference"
  require_disjoint_evidence "$evidence_dir" "$VOKRA_ROOT" "$english_gguf" "$english_reference" "$multilingual_gguf" "$multilingual_reference" "$approval_evidence"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "english_gguf_sha256=$ENGLISH_PUBLIC_SHA256"
    echo "multilingual_gguf_sha256=$MULTILINGUAL_PUBLIC_SHA256"
    echo "english_reference_manifest_sha256=$english_reference_sha256"
    echo "multilingual_reference_manifest_sha256=$multilingual_reference_sha256"
    echo "apple_english_reference_sha256=$english_reference_sha256"
    echo "apple_multilingual_reference_sha256=$multilingual_reference_sha256"
  } > "$evidence_dir/input-hashes.txt"
  hash_reference_tree "$english_reference" "$evidence_dir/english-reference-sha256.txt"
  hash_reference_tree "$multilingual_reference" "$evidence_dir/multilingual-reference-sha256.txt"

  run_variant english cpu "$english_gguf" "$english_reference" \
    "$evidence_dir/english-cpu.log" "$ENGLISH_TEST"
  run_variant multilingual cpu "$multilingual_gguf" "$multilingual_reference" \
    "$evidence_dir/multilingual-cpu.log" "$MULTILINGUAL_TEST"
  run_variant english metal "$english_gguf" "$english_reference" \
    "$evidence_dir/english-metal.log" "$ENGLISH_TEST"
  run_variant multilingual metal "$multilingual_gguf" "$multilingual_reference" \
    "$evidence_dir/multilingual-metal.log" "$MULTILINGUAL_TEST"

  # These markers must come from a real test that compares the same decoded
  # packet on CPU and Metal. Two successful official-reference runs alone do
  # not prove a CPU-vs-Metal bound, so never synthesize these lines here.
  [[ "$(grep -Ec '^PARLER_APPLE_PARITY variant=english metal_vs_cpu=PASS$' "$evidence_dir/english-metal.log" || true)" == 1 ]] \
    || die 'Parler English test emitted no real Metal-vs-CPU PASS marker; refusing to fabricate PASS'
  [[ "$(grep -Ec '^PARLER_APPLE_PARITY variant=multilingual metal_vs_cpu=PASS$' "$evidence_dir/multilingual-metal.log" || true)" == 1 ]] \
    || die 'Parler Multilingual test emitted no real Metal-vs-CPU PASS marker; refusing to fabricate PASS'
  {
    echo 'verdict=PASS'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "english_gguf_sha256=$ENGLISH_PUBLIC_SHA256"
    echo "multilingual_gguf_sha256=$MULTILINGUAL_PUBLIC_SHA256"
    echo 'english_cpu_reference=PASS'
    echo 'multilingual_cpu_reference=PASS'
    echo 'english_metal_vs_cpu=PASS'
    echo 'multilingual_metal_vs_cpu=PASS'
    echo 'upload=NOT_PERFORMED'
    echo 'conversion=NOT_PERFORMED'
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, remove staged model/reference data, then destroy the remote worker"
}

main "$@"
