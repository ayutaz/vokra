#!/usr/bin/env bash
# Real-weight MOSS Audio Tokenizer v2 CPU/reference/Metal measurement on a
# disposable Apple Silicon host. The VAST worker stages the 8.49 GB GGUF and
# independent official reference; this verifier never acquires model data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
V2_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_v2"
LICENSE_GATE="$V2_PROJECT/license_gate.py"
LICENSE_MANIFEST="$V2_PROJECT/license_gate_manifest.json"

OFFICIAL_REVISION="f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
MODEL_SOURCE_SHA256="7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9"
CONFIG_SOURCE_SHA256="f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/src/moss_audio_tokenizer/full_decoder.rs"
TEST_NAME="measure_v2_real_cpu_and_optional_metal_against_official"
TEST_SELECTOR="moss_audio_tokenizer::full_decoder::tests::$TEST_NAME"

log() { printf '[moss-tokenizer-v2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-moss-audio-tokenizer-v2.sh \
  --gguf <vast-moss-audio-tokenizer-v2.gguf> \
  --reference <vast-independent-reference.csv> \
  --gguf-sha256 <lowercase-sha256> --reference-sha256 <lowercase-sha256> \
  --approval-evidence <external-evidence.json> \
  --evidence-dir <absent-dir>
       apple-silicon-moss-audio-tokenizer-v2.sh --self-test

Runs the exact ignored MOSS Audio Tokenizer v2 test with real Metal enabled.
The test itself decodes the same official code packet on CPU and Metal and
prints the independent official-reference and Metal-vs-CPU measurements. The
numeric bound is intentionally unset, so the verdict remains
MEASURED_NOT_GATED; this verifier never turns an observation into PASS.

The host must be a disposable Darwin/arm64 checkout with
VOKRA_REMOTE_APPLE_SILICON=1, at least 32 GB physical memory, 20 GB free disk,
and Xcode's Metal compiler. Inputs are produced/authenticated by VAST. This
script does not download, convert, upload, publish, or delete model files.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_absent_evidence_directory() {
  local directory="$1"
  [[ ! -e "$directory" && ! -L "$directory" ]] || { die "evidence directory must be absent: $directory"; return 2; }
  mkdir -p "$directory"
}

require_disjoint_evidence() {
  local evidence="$1" gguf="$2" reference="$3" root="$4" approval="$5" evidence_parent evidence_real gguf_real reference_real root_real approval_real
  [[ ! -L "$evidence" && ! -L "$reference" && ! -L "$gguf" && ! -L "$root" && ! -L "$approval" ]] || { die 'evidence/reference/GGUF/checkout/approval paths must not be symlinks'; return 2; }
  evidence_parent="$(cd "$(dirname "$evidence")" 2>/dev/null && pwd -P)" || { die 'evidence parent is unavailable'; return 2; }
  gguf_real="$(cd "$(dirname "$gguf")" 2>/dev/null && pwd -P)/$(basename "$gguf")" || { die 'GGUF parent is unavailable'; return 2; }
  reference_real="$(cd "$(dirname "$reference")" 2>/dev/null && pwd -P)/$(basename "$reference")" || { die 'reference parent is unavailable'; return 2; }
  approval_real="$(cd "$(dirname "$approval")" 2>/dev/null && pwd -P)/$(basename "$approval")" || { die 'approval parent is unavailable'; return 2; }
  root_real="$(cd "$root" 2>/dev/null && pwd -P)" || { die 'checkout path is unavailable'; return 2; }
  evidence_real="$evidence_parent/$(basename "$evidence")"
  [[ "$evidence_real" != "$root_real" && "$evidence_real" != "$gguf_real" && "$evidence_real" != "$reference_real" && "$evidence_real" != "$approval_real" ]] || { die 'evidence path aliases checkout or input'; return 2; }
  case "$evidence_real/" in "$root_real/"*|"$gguf_real/"*|"$reference_real/"*|"$approval_real/"*) die 'evidence path overlaps checkout or input'; return 2 ;; esac
  case "$root_real/" in "$evidence_real/"*) die 'checkout overlaps evidence'; return 2 ;; esac
  case "$gguf_real/" in "$evidence_real/"*) die 'GGUF overlaps evidence'; return 2 ;; esac
  case "$reference_real/" in "$evidence_real/"*) die 'reference overlaps evidence'; return 2 ;; esac
  case "$approval_real/" in "$evidence_real/"*) die 'approval overlaps evidence'; return 2 ;; esac
}

license_preflight() {
  local approval="$1" gate_args=(--lock "$V2_PROJECT/uv.lock" --project "$V2_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST")
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] || { die 'v2 approval gate/manifest is missing or symlinked'; return 2; }
  [[ -n "$approval" && -f "$approval" && ! -L "$approval" ]] || { die 'approval evidence must be a required regular non-symlink file'; return 2; }
  gate_args+=(--approval "$approval")
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${gate_args[@]}"
}

hash_reference_tree() {
  local directory="$1" output="$2" path relative
  : > "$output"
  while IFS= read -r path; do
    relative="${path#"$directory"/}"
    printf '%s  %s\n' "$(sha256_file "$path")" "$relative" >> "$output"
  done < <(find "$directory" -type f -print | sort)
}

require_exact_csv_line() {
  local path="$1" expected="$2" count
  count="$(awk -F, -v wanted="$expected" '$0 == wanted {count++} END {print count + 0}' "$path")"
  [[ "$count" == 1 ]] || die "reference must contain exactly one line: $expected"
}

require_test_evidence() {
  local path="$1" named result result_lines test_lines cpu metal
  named="$(grep -Ec "^test ${TEST_SELECTOR} \.\.\. ok$" "$path" || true)"
  result="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)"
  result_lines="$(grep -Ec '^test result:' "$path" || true)"
  test_lines="$(awk '/^test / && $0 !~ /^test result:/ {count++} END {print count + 0}' "$path")"
  cpu="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ actual=[0-9.eE+-]+ reference=[0-9.eE+-]+$' "$path" || true)"
  metal="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ metal=[0-9.eE+-]+ cpu=[0-9.eE+-]+$' "$path" || true)"
  if [[ "$named" != 1 || "$result" != 1 || "$result_lines" != 1 || "$test_lines" != 1 || "$cpu" != 1 || "$metal" != 1 ]]; then
    die 'MOSS v2 test evidence requires exactly one named pass/result and CPU/Metal measurement sentinel'; return 2
  fi
}

require_reference() {
  local path="$1" index count
  require_file 'MOSS v2 independent reference' "$path"
  awk -F, -v source_row="source,v2,OpenMOSS-Team/MOSS-Audio-Tokenizer-v2,$OFFICIAL_REVISION" '
    $0 == source_row ||
    $0 ~ /^runtime,torch-[^,]+,transformers-[^,]+$/ ||
    $0 ~ /^environment,cpu,[^,]+,machine-[^,]+,logical-[0-9]+,torch-capability-[^,]+$/ ||
    $0 == "environment,device,cuda" ||
    $0 ~ /^source_file,(model|config),transformers_modules\/[^,]+,[0-9a-f]{64}$/ ||
    $0 == "contract,2,12,1024,48000,2,3840" ||
    $0 ~ /^codes,[0-9]+(,[0-9]+){23}$/ ||
    $0 ~ /^tensor,(quantizer|decoder_([0-9]|1[01])),[0-9]+(x[0-9]+)+,[-+0-9.eE]+(,[-+0-9.eE]+)*$/ ||
    $0 ~ /^tensor,audio,1x2x7680,[-+0-9.eE]+(,[-+0-9.eE]+)*$/ { next }
    { exit 1 }
  ' "$path" || { die 'reference contains an unknown or malformed row'; return 2; }
  require_exact_csv_line "$path" "source,v2,OpenMOSS-Team/MOSS-Audio-Tokenizer-v2,$OFFICIAL_REVISION"
  count="$(awk -F, '$1 == "source_file" && $2 == "model" {count++} END {print count + 0}' "$path")"
  [[ "$count" == 1 ]] || die 'reference must contain exactly one model source-file row'
  awk -F, -v expected="$MODEL_SOURCE_SHA256" '$1 == "source_file" && $2 == "model" {if (NF != 4 || $3 !~ /transformers_modules/ || $3 == "" || $4 != expected) exit 1; found=1} END {exit(found ? 0 : 1)}' "$path" \
    || die 'reference model source row is not an authenticated transformers_modules path/hash'
  count="$(awk -F, '$1 == "source_file" && $2 == "config" {count++} END {print count + 0}' "$path")"
  [[ "$count" == 1 ]] || die 'reference must contain exactly one config source-file row'
  awk -F, -v expected="$CONFIG_SOURCE_SHA256" '$1 == "source_file" && $2 == "config" {if (NF != 4 || $3 !~ /transformers_modules/ || $3 == "" || $4 != expected) exit 1; found=1} END {exit(found ? 0 : 1)}' "$path" \
    || die 'reference config source row is not an authenticated transformers_modules path/hash'
  count="$(awk -F, '$1 == "runtime" && $0 ~ /^runtime,torch-[^,]+,transformers-[^,]+$/ {count++} END {print count + 0}' "$path")"
  [[ "$count" == 1 ]] || die 'reference must contain exactly one Torch/Transformers runtime row'
  require_exact_csv_line "$path" 'environment,device,cuda'
  count="$(awk -F, '$1 == "environment" && $2 == "cpu" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || die 'reference must contain exactly one CPU/ISA row'
  require_exact_csv_line "$path" 'contract,2,12,1024,48000,2,3840'
  count="$(awk -F, '$1 == "codes" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || die 'reference must contain exactly one code packet'
  for index in $(seq 0 11); do
    count="$(awk -F, -v key="decoder_${index}" '$1 == "tensor" && $2 == key {count++} END {print count + 0}' "$path")"
    [[ "$count" == 1 ]] || die "reference decoder tap is missing or duplicated: decoder_${index}"
  done
  count="$(awk -F, '$1 == "tensor" && $2 == "quantizer" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || die 'reference quantizer tap is missing or duplicated'
  count="$(awk -F, '$1 == "tensor" && $2 == "audio" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || die 'reference audio tap is missing or duplicated'
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "real Metal measurement requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "real Metal measurement requires Apple arm64"
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
    system_profiler xcrun wc tr sort seq; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra checkout"
  [[ -f "$TEST_SOURCE" ]] || die "MOSS v2 test source is missing: $TEST_SOURCE"
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
  local temporary script_path required reference index
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-moss-tokenizer-v2-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_absent_evidence_directory "$temporary/evidence"
  mkdir -p "$temporary/root/nested"
  cp "$temporary/value" "$temporary/root/gguf"
  cp "$temporary/value" "$temporary/root/reference.csv"
  if require_disjoint_evidence "$temporary/root/nested/evidence" "$temporary/root/gguf" "$temporary/root/reference.csv" "$temporary/root" "$temporary/value"; then die 'checkout-contained evidence was accepted'; fi
  if require_disjoint_evidence "$temporary/root/gguf" "$temporary/root/gguf" "$temporary/root/reference.csv" "$temporary/root" "$temporary/value"; then die 'evidence/input equality was accepted'; fi
  ln -s "$temporary/value" "$temporary/input-link"
  if require_file input "$temporary/input-link"; then die 'symlink input was accepted'; fi
  ln -s "$temporary/value" "$temporary/approval-link"
  if require_disjoint_evidence "$temporary/disjoint-evidence" "$temporary/value" "$temporary/value" "$temporary" "$temporary/approval-link"; then die 'symlink approval was accepted'; fi
  mkdir -p "$temporary/existing"
  if require_absent_evidence_directory "$temporary/existing"; then die 'pre-existing evidence directory was accepted'; fi
  ln -s "$temporary/value" "$temporary/evidence-link"
  if require_absent_evidence_directory "$temporary/evidence-link"; then die 'symlink evidence path was accepted'; fi
  script_path="${BASH_SOURCE[0]}"
  fake_root="$temporary/fake-root"
  mkdir -p "$fake_root/tools/parity/moss_audio_tokenizer_v2"
  cp "$V2_PROJECT/license_gate.py" "$V2_PROJECT/license_gate_manifest.json" "$V2_PROJECT/uv.lock" "$V2_PROJECT/pyproject.toml" "$fake_root/tools/parity/moss_audio_tokenizer_v2/"
  if VOKRA_ROOT="$fake_root" VOKRA_REMOTE_APPLE_SILICON=0 "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --approval-evidence "$fake_root/missing-approval" --evidence-dir "$temporary/missing-evidence" >/dev/null 2>&1; then
    die 'Apple verifier accepted missing approval evidence'
  fi
  [[ ! -e "$temporary/missing-evidence" ]] || die 'Apple verifier created evidence before approval gate'
  if "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --evidence-dir "$temporary/missing-option-evidence" >/dev/null 2>&1; then die 'missing approval option was accepted'; fi
  [[ ! -e "$temporary/missing-option-evidence" ]] || die 'missing approval option created evidence'
  if "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --approval-evidence "$temporary/value" --approval-evidence "$temporary/value" --evidence-dir "$temporary/duplicate-approval-evidence" >/dev/null 2>&1; then die 'duplicate approval option was accepted'; fi
  if "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --approval-evidence "" --evidence-dir "$temporary/empty-approval-evidence" >/dev/null 2>&1; then die 'empty approval value was accepted'; fi
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' \
    'xcrun -f metal' 'full_decoder.rs' "$TEST_NAME" \
    '--features metal --lib' '-- --ignored --exact --nocapture' \
    'VOKRA_MOSS_AUDIO_TOKENIZER_V2_GGUF' \
    'VOKRA_MOSS_AUDIO_TOKENIZER_V2_REFERENCE' \
    'VOKRA_MOSS_AUDIO_TOKENIZER_V2_METAL_MEASUREMENT=1' \
    'MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET' \
    'MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET' \
    'verdict=MEASURED_NOT_GATED' 'numeric_bounds=UNSET' 'upload=NOT_PERFORMED'; do
    grep -Fq -- "$required" "$script_path" \
      || die "self-test contract token is missing: $required"
  done
  if grep -En -- '(^|[[:space:]])(curl|wget|pip|git[[:space:]]+(clone|fetch|pull|push))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    die "download, conversion, or publication command found"
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    die "--self-test accepted an extra argument"
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then
    die "duplicate --self-test was accepted"
  fi
  if "$script_path" --gguf --reference x --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --evidence-dir x >/dev/null 2>&1; then
    die "bare option value was accepted"
  fi
  for option in gguf reference gguf-sha256 reference-sha256 approval-evidence evidence-dir; do
    if "$script_path" --self-test "--$option" -x >/dev/null 2>&1; then
      die "negative --$option value was accepted"
    fi
  done
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 metal=1.0e-9 cpu=1.0e-9\n' "$TEST_SELECTOR" > "$temporary/test.log"
  require_test_evidence "$temporary/test.log"
  printf 'test %s ... ok\ntest %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 metal=1.0e-9 cpu=1.0e-9\n' "$TEST_SELECTOR" "$TEST_SELECTOR" > "$temporary/duplicate.log"
  if require_test_evidence "$temporary/duplicate.log"; then die 'duplicate named test was accepted'; fi
  printf 'test %s ... ok\ntest other_test ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 metal=1.0e-9 cpu=1.0e-9\n' "$TEST_SELECTOR" > "$temporary/extra-test.log"
  if require_test_evidence "$temporary/extra-test.log"; then die 'extra test was accepted'; fi
  cp "$temporary/test.log" "$temporary/duplicate-result.log"
  printf 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n' >> "$temporary/duplicate-result.log"
  if require_test_evidence "$temporary/duplicate-result.log"; then die 'duplicate test result was accepted'; fi
  awk 'NR == 2 { print "test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"; next } { print }' \
    "$temporary/test.log" > "$temporary/malformed-result.log"
  if require_test_evidence "$temporary/malformed-result.log"; then die 'malformed test result was accepted'; fi
  sed 's/filtered out$/filtered out; finished in nope/' "$temporary/test.log" > "$temporary/bad-timing.log"
  if require_test_evidence "$temporary/bad-timing.log"; then die 'malformed timing was accepted'; fi
  reference="$temporary/reference.csv"
  printf 'source,v2,OpenMOSS-Team/MOSS-Audio-Tokenizer-v2,%s\nruntime,torch-2.7.1+cu126,transformers-5.5.0\nenvironment,cpu,test,machine-test,logical-1,torch-capability-test\nenvironment,device,cuda\nsource_file,model,transformers_modules/test/model.py,%s\nsource_file,config,transformers_modules/test/config.py,%s\ncontract,2,12,1024,48000,2,3840\ncodes,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\ntensor,quantizer,1x1,0\n' "$OFFICIAL_REVISION" "$MODEL_SOURCE_SHA256" "$CONFIG_SOURCE_SHA256" > "$reference"
  for index in $(seq 0 11); do printf 'tensor,decoder_%s,1x1,0\n' "$index" >> "$reference"; done
  printf 'tensor,audio,1x2x7680,0\n' >> "$reference"
  require_reference "$reference"
  cp "$reference" "$temporary/extra-reference.csv"
  printf 'unexpected,row\n' >> "$temporary/extra-reference.csv"
  if require_reference "$temporary/extra-reference.csv"; then die 'extra reference row was accepted'; fi
  log "self-test PASS"
)

main() {
  local gguf='' reference='' gguf_sha='' reference_sha='' approval='' evidence_dir='' self_test=0
  local seen_gguf=0 seen_reference=0 seen_gguf_sha=0 seen_reference_sha=0 seen_approval=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf) (( ! seen_gguf++ )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf="$2"; shift 2 ;;
      --reference) (( ! seen_reference++ )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference="$2"; shift 2 ;;
      --gguf-sha256) (( ! seen_gguf_sha++ )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf_sha="$2"; shift 2 ;;
      --reference-sha256) (( ! seen_reference_sha++ )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference_sha="$2"; shift 2 ;;
      --approval-evidence) (( ! seen_approval++ )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; approval="$2"; shift 2 ;;
      --evidence-dir) (( ! seen_evidence++ )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; evidence_dir="$2"; shift 2 ;;
      --self-test) (( ! seen_self_test++ )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$gguf$reference$gguf_sha$reference_sha$approval$evidence_dir" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference" && -n "$gguf_sha" && -n "$reference_sha" && -n "$approval" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --reference, both SHA-256 values, --approval-evidence, and --evidence-dir are required"; }
  [[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$reference_sha" =~ ^[0-9a-f]{64}$ ]] \
    || { die "expected hashes must be lowercase 64-hex SHA-256 values"; return 2; }

  license_preflight "$approval"
  require_remote_apple_host
  require_tooling
  require_file 'VAST-produced MOSS v2 GGUF' "$gguf"
  [[ "$(sha256_file "$gguf")" == "$gguf_sha" ]] || { die 'GGUF SHA-256 differs from VAST evidence'; return 2; }
  require_file 'VAST-produced MOSS v2 reference' "$reference"
  [[ "$(sha256_file "$reference")" == "$reference_sha" ]] || { die 'reference SHA-256 differs from VAST evidence'; return 2; }
  require_reference "$reference"
  require_disjoint_evidence "$evidence_dir" "$gguf" "$reference" "$VOKRA_ROOT" "$approval"
  require_absent_evidence_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf_sha256=$gguf_sha"
    echo "reference_sha256=$reference_sha"
    echo "official_revision=$OFFICIAL_REVISION"
    echo "model_source_sha256=$MODEL_SOURCE_SHA256"
    echo "config_source_sha256=$CONFIG_SOURCE_SHA256"
  } > "$evidence_dir/input-hashes.txt"
  printf '%s  %s\n' "$(sha256_file "$reference")" "$(basename "$reference")" \
    > "$evidence_dir/reference-tree-sha256.txt"

  log 'running exact ignored real-weight CPU/reference/Metal measurement'
  env VOKRA_MOSS_AUDIO_TOKENIZER_V2_GGUF="$gguf" \
    VOKRA_MOSS_AUDIO_TOKENIZER_V2_REFERENCE="$reference" \
    VOKRA_MOSS_AUDIO_TOKENIZER_V2_METAL_MEASUREMENT=1 \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --lib "$TEST_SELECTOR" \
      -- --ignored --exact --nocapture --test-threads=1 \
      2>&1 | tee "$evidence_dir/parity.log"
  grep -Fq "test $TEST_SELECTOR ... ok" "$evidence_dir/parity.log" \
    || die 'MOSS v2 exact test did not report success'
  require_test_evidence "$evidence_dir/parity.log"

  {
    echo 'verdict=MEASURED_NOT_GATED'
    echo 'parity_status=MEASURED_NOT_GATED'
    echo 'numeric_bounds=UNSET'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_sha256=$(sha256_file "$reference")"
    echo "test=$TEST_SELECTOR"
    echo 'cpu_reference=MEASURED_NOT_GATED'
    echo 'metal_vs_cpu=MEASURED_NOT_GATED'
    echo 'upload=NOT_PERFORMED'
    echo 'conversion=NOT_PERFORMED'
  } > "$evidence_dir/summary.txt"
  log 'MEASURED_NOT_GATED: pull only evidence, remove staged inputs, then destroy the remote worker'
}

main "$@"
