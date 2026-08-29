#!/usr/bin/env bash
# Real-weight Bark Small/Full CPU/reference/Metal parity on a disposable
# Apple Silicon host. All model and reference inputs are staged by VAST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bark"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

SMALL_PUBLIC_BYTES=1674074848
SMALL_PUBLIC_SHA256="43b781a0dcd66f1e7451005e461ec20e2141bc9c4f529feb4a9a8c0e352ea137"
FULL_PUBLIC_BYTES=4466390272
FULL_PUBLIC_SHA256="fd628312ce7d8e1cbc41718741614116d5c7f08d0763f81622edbac320b208ec"
TRANSFORMERS_SOURCE_REVISION="c1c34249fa27deefbd4a377dfbf883a39baf5c6d"
GENERATION_CONFIG_SHA256="ab2969fcd40e085bc924ad99ad419c27f62f5acb61afac5de7490ab0c796b5b9"
SMALL_UPSTREAM_REVISION="1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd"
FULL_UPSTREAM_REVISION="70a8a7d34168586dc5d028fa9666aceade177992"
SMALL_CHECKPOINT_SHA256="f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9"
FULL_CHECKPOINT_SHA256="4e3d407b9b3b619da184c85786c88e5e35f90f9089303e16db696ed0be477989"
SMALL_CONFIG_SHA256="9d95e9c3027cd79cf5f762cc03a69b6393cea87c51e9dd6b998fde3a7f01510e"
FULL_CONFIG_SHA256="48be144c0232acd8c55786d1eea9161ae6c973f21ec4a2f02627c844065ea695"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_bark_real.rs"
TEST_TARGET="parity_bark_real"
SMALL_TEST="real_bark_small_matches_official_transformers"
FULL_TEST="real_bark_full_matches_official_transformers"

log() { printf '[bark-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-bark.sh \
  --small-gguf <bark-small.gguf> --small-reference <dir> \
  --full-gguf <bark-full.gguf> --full-reference <dir> \
  --small-reference-manifest-sha256 <sha256> \
  --full-reference-manifest-sha256 <sha256> \
  --approval-evidence <file> \
  --evidence-dir <absent-dir>
       apple-silicon-bark.sh --self-test

Runs the exact real-weight Bark Small and Bark Full tests once on CPU and
once on real Metal. Each test must compare its generated codes and decoded
PCM to the independently generated official Transformers reference. The
verifier also requires a real test-emitted Metal-vs-CPU PASS marker; it never
creates that marker from two unrelated successful invocations.

The host must be a disposable Darwin/arm64 checkout with
VOKRA_REMOTE_APPLE_SILICON=1, at least 32 GB physical memory, 20 GB free disk,
and Xcode's Metal compiler. Inputs are VAST-produced or VAST-authenticated.
The evidence directory must be absent/nonexistent before validation; it is
created only after all input and approval checks succeed.
This script does not download, convert, upload, publish, or delete models.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_one_line() {
  local log_path="$1" pattern="$2" count
  count="$(grep -Ec "$pattern" "$log_path" || true)"
  [[ "$count" == 1 ]] || { die "evidence must contain exactly one anchored line matching $pattern"; return 2; }
}

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
    die "evidence parent is not accessible: $(dirname "$evidence")"
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
      die "validation input is not accessible: $other"
      return 2
    fi
    real="$other_parent/$(basename "$other")"
    if [[ "$candidate" == "$real" || "$candidate/" == "$real/"* || "$real/" == "$candidate/"* ]]; then
      die "evidence directory overlaps validation input: $evidence / $other"
      return 2
    fi
  done
  mkdir -p "$evidence"
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die "required tool missing: uv"
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] \
    || die "Bark license gate or tracked manifest is missing"
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] \
    || die "approval evidence must be a nonempty regular file: $approval"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
    --manifest "$LICENSE_MANIFEST" --approval "$approval" \
    --small-public-repo vokra/bark-small --small-upstream-repo suno/bark-small \
    --full-public-repo vokra/bark --full-upstream-repo suno/bark \
    --transformers-version 5.5.0 --small-public-bytes "$SMALL_PUBLIC_BYTES" \
    --small-checkpoint-bytes 1676663913 --small-config-bytes 8803 \
    --full-public-bytes "$FULL_PUBLIC_BYTES" --full-checkpoint-bytes 4486643861 \
    --full-config-bytes 8806 --generation-config-bytes 4908 \
    --small-public-revision 09802c56a2b2e8ad87835115b94b38031fde29b6 \
    --small-upstream-revision "$SMALL_UPSTREAM_REVISION" \
    --full-public-revision f304ddcdfd9218994731ec3b09e89b9961b8b751 \
    --full-upstream-revision "$FULL_UPSTREAM_REVISION" \
    --transformers-source-revision "$TRANSFORMERS_SOURCE_REVISION" \
    --small-public-sha256 "$SMALL_PUBLIC_SHA256" --full-public-sha256 "$FULL_PUBLIC_SHA256" \
    --small-checkpoint-sha256 "$SMALL_CHECKPOINT_SHA256" --full-checkpoint-sha256 "$FULL_CHECKPOINT_SHA256" \
    --small-config-sha256 "$SMALL_CONFIG_SHA256" --full-config-sha256 "$FULL_CONFIG_SHA256" \
    --generation-config-sha256 "$GENERATION_CONFIG_SHA256" \
    --transformers-sdist-sha256 c8db656cf51c600cd8c75f06b20ef85c72e8b8ff9abc880c5d3e8bc70e0ddcbd \
    --transformers-wheel-sha256 821a9ff0961abbb29eb1eb686d78df1c85929fdf213a3fe49dc6bd94f9efa944
}

require_reference() {
  local label="$1" directory="$2" variant="$3" expected_manifest_sha="$4" revision checkpoint_sha config_sha
  [[ -d "$directory" && ! -L "$directory" ]] || die "$label is not a regular directory: $directory"
  for name in manifest.json text_token_ids.u32le semantic_tokens.u32le \
    codes.u32le decoded_pcm.f32; do
    require_file "$label $name" "$directory/$name"
  done
  case "$variant" in
    small) revision="$SMALL_UPSTREAM_REVISION"; checkpoint_sha="$SMALL_CHECKPOINT_SHA256"; config_sha="$SMALL_CONFIG_SHA256" ;;
    full) revision="$FULL_UPSTREAM_REVISION"; checkpoint_sha="$FULL_CHECKPOINT_SHA256"; config_sha="$FULL_CONFIG_SHA256" ;;
    *) die "unknown Bark variant: $variant" ;;
  esac
  [[ "$(sha256_file "$directory/manifest.json")" == "$expected_manifest_sha" ]] \
    || die "$label manifest SHA-256 does not match VAST-authenticated evidence"
  grep -Fq '"format": "vokra-bark-transformers-5.5-reference-v1"' \
    "$directory/manifest.json" \
    || die "$label manifest is not the pinned official Bark reference format"
  grep -Fq '"upstream_revision": "'"$revision"'"' \
    "$directory/manifest.json" \
    || die "$label manifest lost upstream revision $revision"
  grep -Fq '"transformers_version": "5.5.0"' "$directory/manifest.json" \
    || die "$label manifest lost the locked Transformers 5.5.0 oracle"
  grep -Fq '"transformers_source_revision": "'"$TRANSFORMERS_SOURCE_REVISION"'"' "$directory/manifest.json" \
    || die "$label manifest lost the pinned Transformers source revision"
  grep -Fq '"checkpoint_sha256": "'"$checkpoint_sha"'"' "$directory/manifest.json" \
    || die "$label manifest lost the pinned checkpoint hash"
  grep -Fq '"config_sha256": "'"$config_sha"'"' "$directory/manifest.json" \
    || die "$label manifest lost the pinned config hash"
  grep -Fq '"generation_config_sha256": "'"$GENERATION_CONFIG_SHA256"'"' "$directory/manifest.json" \
    || die "$label manifest lost the pinned generation-config hash"
  local name expected actual count
  for name in text_token_ids.u32le semantic_tokens.u32le codes.u32le decoded_pcm.f32; do
    count="$(awk -F'"' -v key="$name" '$2 == key {count++} END {print count + 0}' "$directory/manifest.json")"
    [[ "$count" == 1 ]] || die "$label manifest must contain exactly one hash for $name"
    expected="$(awk -F'"' -v key="$name" '$2 == key {print $4}' "$directory/manifest.json")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label manifest hash is invalid for $name"
    actual="$(sha256_file "$directory/$name")"
    [[ "$actual" == "$expected" ]] || die "$label artifact hash mismatch for $name"
  done
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
  [[ -f "$TEST_SOURCE" ]] || die "Bark parity source is missing: $TEST_SOURCE"
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
  local temporary script_path required temporary_log
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-bark-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
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
  require_disjoint_evidence "$temporary/evidence" "$temporary/value"
  script_path="${BASH_SOURCE[0]}"
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' \
    'xcrun -f metal' 'parity_bark_real.rs' \
    'real_bark_small_matches_official_transformers' \
    'real_bark_full_matches_official_transformers' \
    '--features metal --test' '-- --exact --nocapture' \
    'VOKRA_BARK_BACKEND=cpu' 'VOKRA_BARK_BACKEND=metal' \
    '--approval-evidence' 'license_preflight' 'require_disjoint_evidence' '! -L' \
    'evidence directory must be absent before validation' \
    'BARK_APPLE_PARITY variant=small metal_vs_cpu=PASS' \
    'BARK_APPLE_PARITY variant=full metal_vs_cpu=PASS' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'require_one_test_pass' 'upload=NOT_PERFORMED'; do
    grep -Fq -- "$required" "$script_path" \
      || die "self-test contract token is missing: $required"
  done
  if grep -En -- '(^|[[:space:]])(curl|wget|pip|git[[:space:]]+(clone|fetch|pull|push)|.*(upload|publish|convert))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    die "download, conversion, or publication command found"
  fi
  if "$script_path" --self-test --small-gguf "$temporary/model.gguf" \
    >/dev/null 2>&1; then
    die "--self-test accepted an extra argument"
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then
    die "duplicate --self-test was accepted"
  fi
  if "$script_path" --small-gguf >/dev/null 2>&1; then die 'missing option value was accepted'; fi
  if "$script_path" --small-gguf '' >/dev/null 2>&1; then die 'empty option value was accepted'; fi
  if "$script_path" --small-gguf --small-reference value >/dev/null 2>&1; then die 'leading-dash option value was accepted'; fi
  if "$script_path" --unknown-option >/dev/null 2>&1; then die 'unknown option was accepted'; fi
  temporary_log="$temporary/result.log"
  printf 'test %s ... ok\ntest %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' \
    "$SMALL_TEST" "$SMALL_TEST" > "$temporary_log"
  if require_one_test_pass "$temporary_log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    die "duplicate named test was accepted"
  fi
  printf 'BARK_APPLE_PARITY variant=small metal_vs_cpu=PASS\nBARK_APPLE_PARITY variant=small metal_vs_cpu=PASS\n' > "$temporary/marker.log"
  if require_one_line "$temporary/marker.log" '^BARK_APPLE_PARITY variant=small metal_vs_cpu=PASS$'; then
    die "duplicate Metal-vs-CPU marker was accepted"
  fi
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' "$SMALL_TEST" > "$temporary/metric.log"
  if require_one_test_pass "$temporary/metric.log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    die "duplicate metric sentinel was accepted"
  fi
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' "$SMALL_TEST" > "$temporary/result-duplicate.log"
  if require_one_test_pass "$temporary/result-duplicate.log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    die "duplicate test result was accepted"
  fi
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: PREFIX frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' "$SMALL_TEST" > "$temporary/prefix.log"
  if require_one_test_pass "$temporary/prefix.log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    die "prefixed metric sentinel was accepted"
  fi
  printf 'BARK_APPLE_PARITY variant=small metal_vs_cpu=FAIL\n' > "$temporary/fail-marker.log"
  if require_one_line "$temporary/fail-marker.log" '^BARK_APPLE_PARITY variant=small metal_vs_cpu=PASS$'; then
    die "FAIL Metal-vs-CPU marker was accepted"
  fi
  log "self-test PASS"
)

require_one_test_pass() {
  local log_path="$1" test_name="$2" marker="$3" test_count named_count result_count result_lines marker_count
  test_count="$(grep -Ev '^test result:' "$log_path" | grep -Ec '^test ' || true)"
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  result_lines="$(grep -Ec '^test result:' "$log_path" || true)"
  marker_count="$(grep -Ec "^${marker} frames=[0-9]+, codes=exact, decode_max_abs=[0-9.eE+-]+, decode_rmse=[0-9.eE+-]+, end_to_end_max_abs=[0-9.eE+-]+, end_to_end_rmse=[0-9.eE+-]+$" "$log_path" || true)"
  if [[ "$test_count" != 1 || "$named_count" != 1 || "$result_count" != 1 || "$result_lines" != 1 || "$marker_count" != 1 ]]; then
    die "${test_name} evidence must contain exactly one named pass, full result, and metric sentinel"; return 2
  fi
}

run_variant() {
  local variant="$1" backend="$2" gguf="$3" reference="$4" output="$5"
  local test_name="$6" variant_label backend_label
  case "$variant" in
    small) variant_label='SMALL' ;;
    full) variant_label='FULL' ;;
    *) die "unknown Bark variant: $variant" ;;
  esac
  case "$backend" in
    cpu) backend_label='Cpu' ;;
    metal) backend_label='Metal' ;;
    *) die "unknown Bark backend: $backend" ;;
  esac
  env VOKRA_BARK_SMALL_GGUF="$gguf" \
    VOKRA_BARK_SMALL_PARITY_DIR="$reference" \
    VOKRA_BARK_FULL_GGUF="$gguf" \
    VOKRA_BARK_FULL_PARITY_DIR="$reference" \
    VOKRA_BARK_BACKEND="$backend" RUST_TEST_THREADS=1 \
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test "$TEST_TARGET" "$test_name" \
      -- --exact --nocapture --test-threads=1 2>&1 | tee "$output"
  require_one_test_pass "$output" "$test_name" "Bark $variant_label $backend_label:"
}

main() {
  local small_gguf='' small_reference='' full_gguf='' full_reference='' approval_evidence=''
  local small_reference_manifest_sha='' full_reference_manifest_sha=''
  local evidence_dir='' self_test=0 self_test_count=0
  local seen_small_gguf=0 seen_small_reference=0 seen_full_gguf=0 seen_full_reference=0
  local seen_small_manifest=0 seen_full_manifest=0 seen_approval=0 seen_evidence=0
  while (( $# > 0 )); do
    case "$1" in
      --small-gguf) (( seen_small_gguf == 0 )) || die "duplicate --small-gguf"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--small-gguf requires a nonempty value"; seen_small_gguf=1; small_gguf="$2"; shift 2 ;;
      --small-reference) (( seen_small_reference == 0 )) || die "duplicate --small-reference"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--small-reference requires a nonempty value"; seen_small_reference=1; small_reference="$2"; shift 2 ;;
      --full-gguf) (( seen_full_gguf == 0 )) || die "duplicate --full-gguf"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--full-gguf requires a nonempty value"; seen_full_gguf=1; full_gguf="$2"; shift 2 ;;
      --full-reference) (( seen_full_reference == 0 )) || die "duplicate --full-reference"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--full-reference requires a nonempty value"; seen_full_reference=1; full_reference="$2"; shift 2 ;;
      --small-reference-manifest-sha256) (( seen_small_manifest == 0 )) || die "duplicate --small-reference-manifest-sha256"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--small-reference-manifest-sha256 requires a nonempty value"; seen_small_manifest=1; small_reference_manifest_sha="$2"; shift 2 ;;
      --full-reference-manifest-sha256) (( seen_full_manifest == 0 )) || die "duplicate --full-reference-manifest-sha256"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--full-reference-manifest-sha256 requires a nonempty value"; seen_full_manifest=1; full_reference_manifest_sha="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die "duplicate --approval-evidence"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--approval-evidence requires a nonempty value"; seen_approval=1; approval_evidence="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die "duplicate --evidence-dir"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--evidence-dir requires a nonempty value"; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
      --self-test) (( self_test_count == 0 )) || die "duplicate --self-test"; self_test_count=1; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$small_gguf$small_reference$full_gguf$full_reference$small_reference_manifest_sha$full_reference_manifest_sha$approval_evidence$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$small_gguf" && -n "$small_reference" && -n "$full_gguf" && \
    -n "$full_reference" && -n "$small_reference_manifest_sha" && \
    -n "$full_reference_manifest_sha" && -n "$approval_evidence" && \
    -n "$evidence_dir" ]] || { usage; die "all GGUF/reference pairs, --approval-evidence, and --evidence-dir are required"; }

  license_preflight "$approval_evidence"
  require_remote_apple_host
  require_tooling
  verify_public_gguf 'Bark Small GGUF' "$small_gguf" "$SMALL_PUBLIC_BYTES" "$SMALL_PUBLIC_SHA256"
  verify_public_gguf 'Bark Full GGUF' "$full_gguf" "$FULL_PUBLIC_BYTES" "$FULL_PUBLIC_SHA256"
  [[ "$small_reference_manifest_sha" =~ ^[0-9a-f]{64}$ && "$full_reference_manifest_sha" =~ ^[0-9a-f]{64}$ ]] \
    || { die "reference manifest SHA-256 arguments must be exact 64-hex values"; return 2; }
  require_reference 'Bark Small reference' "$small_reference" small "$small_reference_manifest_sha"
  require_reference 'Bark Full reference' "$full_reference" full "$full_reference_manifest_sha"
  require_disjoint_evidence "$evidence_dir" "$VOKRA_ROOT" "$small_gguf" "$small_reference" "$full_gguf" "$full_reference" "$approval_evidence"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "small_gguf_sha256=$SMALL_PUBLIC_SHA256"
    echo "full_gguf_sha256=$FULL_PUBLIC_SHA256"
    echo "small_reference_manifest_sha256=$small_reference_manifest_sha"
    echo "full_reference_manifest_sha256=$full_reference_manifest_sha"
  } > "$evidence_dir/input-hashes.txt"
  hash_reference_tree "$small_reference" "$evidence_dir/small-reference-sha256.txt"
  hash_reference_tree "$full_reference" "$evidence_dir/full-reference-sha256.txt"

  run_variant small cpu "$small_gguf" "$small_reference" \
    "$evidence_dir/small-cpu.log" "$SMALL_TEST"
  run_variant full cpu "$full_gguf" "$full_reference" \
    "$evidence_dir/full-cpu.log" "$FULL_TEST"
  run_variant small metal "$small_gguf" "$small_reference" \
    "$evidence_dir/small-metal.log" "$SMALL_TEST"
  run_variant full metal "$full_gguf" "$full_reference" \
    "$evidence_dir/full-metal.log" "$FULL_TEST"

  # These markers must come from a real test that compares the same decoded
  # packet on CPU and Metal. Two successful official-reference runs alone do
  # not prove a CPU-vs-Metal bound, so never synthesize these lines here.
  require_one_test_pass "$evidence_dir/small-cpu.log" "$SMALL_TEST" 'Bark SMALL Cpu:'
  require_one_test_pass "$evidence_dir/full-cpu.log" "$FULL_TEST" 'Bark FULL Cpu:'
  require_one_test_pass "$evidence_dir/small-metal.log" "$SMALL_TEST" 'Bark SMALL Metal:'
  require_one_test_pass "$evidence_dir/full-metal.log" "$FULL_TEST" 'Bark FULL Metal:'
  require_one_line "$evidence_dir/small-metal.log" '^BARK_APPLE_PARITY variant=small metal_vs_cpu=PASS$'
  require_one_line "$evidence_dir/full-metal.log" '^BARK_APPLE_PARITY variant=full metal_vs_cpu=PASS$'
  {
    echo 'verdict=PASS'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "small_gguf_sha256=$SMALL_PUBLIC_SHA256"
    echo "full_gguf_sha256=$FULL_PUBLIC_SHA256"
    echo "small_reference_manifest_sha256=$small_reference_manifest_sha"
    echo "full_reference_manifest_sha256=$full_reference_manifest_sha"
    echo 'small_cpu_reference=PASS'
    echo 'full_cpu_reference=PASS'
    echo 'small_metal_vs_cpu=PASS'
    echo 'full_metal_vs_cpu=PASS'
    echo 'upload=NOT_PERFORMED'
    echo 'conversion=NOT_PERFORMED'
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, remove staged model/reference data, then destroy the remote worker"
}

main "$@"
