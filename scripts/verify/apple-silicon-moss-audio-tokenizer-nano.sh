#!/usr/bin/env bash
# Disposable Apple Silicon Metal measurement for a VAST-produced corrected
# MOSS Audio Tokenizer Nano GGUF and independent reference CSV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
APPLE_CONTRACT_SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
NANO_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_nano"
LICENSE_GATE="$NANO_PROJECT/license_gate.py"
LICENSE_MANIFEST="$NANO_PROJECT/license_gate_manifest.json"
VAST_CONTRACT_SCRIPT="$VOKRA_ROOT/scripts/publish/vast-ai/run-moss-audio-tokenizer-nano-validation.sh"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_NET_OFFLINE=true

OFFICIAL_REPO="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
OFFICIAL_REVISION="6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_moss_audio_tokenizer_nano_real.rs"
TEST_NAME="official_nano_decode_measurement"
TEST_SELECTOR="parity_moss_audio_tokenizer_nano_real::$TEST_NAME"
# These identities are synchronized with the authenticated VAST worker and
# Nano license/source contract.  They establish readiness only: owner
# approval, real-weight execution, numeric bounds, and publication remain
# separate gates below.
EXPECTED_MODEL_SOURCE_PATH="transformers_modules/hf/cfb29bb1bac555fe/modeling_moss_audio_tokenizer.py"
EXPECTED_CONFIG_SOURCE_PATH="transformers_modules/hf/1f68fe91b6890e3e/configuration_moss_audio_tokenizer.py"
EXPECTED_MODEL_SOURCE_SHA256="b14af7c188944da5101adbd4aaa9c3617d66b83507f0efbd6eb416381a105930"
EXPECTED_CONFIG_SOURCE_SHA256="b2d67dc4581e70f4b69b2d7eccefe32581d0c5192fe4d97fe1830e94a255b8aa"
EXPECTED_TORCH_VERSION="2.7.1+cpu"
EXPECTED_TRANSFORMERS_VERSION="5.10.4"
EXPECTED_QUANTIZER_SHAPE="1x768x2"
EXPECTED_DECODER_TAP_COUNT="9"
EXPECTED_DECODER_TAP_SHAPES="1x192x8,1x768x8,1x384x16,1x768x16,1x384x32,1x768x32,1x384x64,1x240x64,1x1x15360"
EXPECTED_CPU_TORCH_VERSION="2.7.1+cpu"
EXPECTED_CPU_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
EXPECTED_CODES="17,520,1023,502,1005,484,987,466,969,448,951,430,933,412,915,394,274,777,256,759,238,741,220,723,202,705,184,687,166,669,148,651"

CONTRACT_NAMES=(
  EXPECTED_MODEL_SOURCE_PATH EXPECTED_CONFIG_SOURCE_PATH
  EXPECTED_MODEL_SOURCE_SHA256 EXPECTED_CONFIG_SOURCE_SHA256
  EXPECTED_TORCH_VERSION EXPECTED_TRANSFORMERS_VERSION
  EXPECTED_QUANTIZER_SHAPE EXPECTED_DECODER_TAP_COUNT
  EXPECTED_DECODER_TAP_SHAPES EXPECTED_CPU_TORCH_VERSION
  EXPECTED_CPU_TORCH_INDEX EXPECTED_CODES
)

log() { printf '[moss-tokenizer-nano-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-moss-audio-tokenizer-nano.sh \
  --gguf <vast-corrected-nano.gguf> \
  --reference <vast-independent-nano-reference.csv> \
  --gguf-sha256 <lowercase-sha256> --reference-sha256 <lowercase-sha256> \
  --approval-evidence <external-evidence.json> --expected-head <40-hex-commit> \
  --evidence-dir <absent-dir>
       apple-silicon-moss-audio-tokenizer-nano.sh --self-test

Runs the exact ignored Nano test with the real Metal feature/backend. The
test binds the corrected Nano metadata/374-tensor manifest, decodes the same
official code packet on CPU and Metal, and records both comparisons. No
reviewed Nano numeric bound exists, so the verdict is always
MEASURED_NOT_GATED; this script never reports numeric PASS.

The disposable host must be Darwin/arm64 with VOKRA_REMOTE_APPLE_SILICON=1,
at least 32 GB physical memory, 20 GB free disk, and Xcode's Metal compiler.
Inputs must already be staged and hashed by the VAST worker. This script does
not download, convert, publish, upload, or delete model files.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

contract_value() {
  local script="$1" name="$2"
  awk -F= -v expected="$name" '
    $1 == expected && $0 ~ /^[^=]+="[^"]*"$/ {
      count++
      value=$0
      sub(/^[^=]+="/, "", value)
      sub(/"$/, "", value)
    }
    END {
      if (count != 1) exit 2
      print value
    }
  ' "$script"
}

validate_contract_definitions() {
  local script="$1" line name count
  [[ -f "$script" && ! -L "$script" ]] || { die "contract script is missing or symlinked: $script"; return 2; }
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ "$line" == EXPECTED_* ]] || continue
    [[ "$line" == *=* ]] || continue
    [[ "$line" =~ ^EXPECTED_[A-Z0-9_]+=\"[^\"]*\"$ ]] \
      || { die "malformed contract definition in $script: $line"; return 2; }
    name="${line%%=*}"
    case " ${CONTRACT_NAMES[*]} " in
      *" $name "*) ;;
      *) die "unknown contract definition in $script: $name"; return 2 ;;
    esac
  done < "$script"
  for name in "${CONTRACT_NAMES[@]}"; do
    count="$(awk -F= -v expected="$name" '$1 == expected && $0 ~ /^[^=]+="[^"]*"$/ {count++} END {print count + 0}' "$script")"
    [[ "$count" == 1 ]] || { die "contract definition $name occurs $count times in $script"; return 2; }
  done
}

require_cross_contract() {
  local counterpart="$1" current="$2" name current_value counterpart_value
  validate_contract_definitions "$current" || return 2
  validate_contract_definitions "$counterpart" || return 2
  for name in "${CONTRACT_NAMES[@]}"; do
    current_value="$(contract_value "$current" "$name")" || { die "cannot parse $name from $current"; return 2; }
    counterpart_value="$(contract_value "$counterpart" "$name")" || { die "cannot parse $name from $counterpart"; return 2; }
    [[ "$current_value" == "$counterpart_value" ]] \
      || { die "cross-contract drift for $name: $current_value != $counterpart_value"; return 2; }
  done
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_absent_evidence_directory() {
  local directory="$1"
  [[ "$directory" = /* ]] || { die 'evidence directory must be an absolute path'; return 2; }
  [[ ! -e "$directory" && ! -L "$directory" ]] || { die "evidence directory must be absent: $directory"; return 2; }
  mkdir -p "$directory"
}

canonical_existing_path() {
  local path="$1" parent
  [[ "$path" == /* ]] || path="$PWD/$path"
  [[ -e "$path" && ! -L "$path" ]] || return 1
  local scan rest component
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  if [[ -d "$path" ]]; then
    (cd -P "$path" && printf '%s\n' "$PWD")
  else
    parent="$(dirname "$path")"
    (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$path")")
  fi
}

canonical_absent_path() {
  local path="$1" suffix='' name parent
  [[ "$path" == /* ]] || path="$PWD/$path"
  [[ ! -e "$path" && ! -L "$path" ]] || return 1
  local scan rest component
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -e "$path" && ! -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="$(dirname "$path")"
    [[ "$parent" != "$path" ]] || return 1
    path="$parent"
  done
  [[ -d "$path" && ! -L "$path" ]] || return 1
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

require_disjoint_evidence() {
  local evidence="$1" gguf="$2" reference="$3" root="$4" approval="$5" evidence_real gguf_real reference_real root_real approval_real
  [[ "$evidence" = /* ]] || { die 'evidence directory must be an absolute path'; return 2; }
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || { die 'evidence directory must be absent and non-symlink'; return 2; }
  [[ -f "$reference" && ! -L "$reference" && -f "$gguf" && ! -L "$gguf" && -f "$approval" && ! -L "$approval" && -d "$root" && ! -L "$root" ]] || { die 'existing inputs and checkout must have exact non-symlink types'; return 2; }
  [[ -d "$(dirname "$evidence")" ]] || { die 'evidence parent is unavailable'; return 2; }
  evidence_real="$(canonical_absent_path "$evidence")" || { die 'evidence path cannot be canonicalized'; return 2; }
  gguf_real="$(canonical_existing_path "$gguf")" || { die 'GGUF path cannot be canonicalized'; return 2; }
  reference_real="$(canonical_existing_path "$reference")" || { die 'reference path cannot be canonicalized'; return 2; }
  approval_real="$(canonical_existing_path "$approval")" || { die 'approval path cannot be canonicalized'; return 2; }
  root_real="$(canonical_existing_path "$root")" || { die 'checkout path cannot be canonicalized'; return 2; }
  [[ "$evidence_real" != "$root_real" && "$evidence_real" != "$gguf_real" && "$evidence_real" != "$reference_real" && "$evidence_real" != "$approval_real" ]] || { die 'evidence path aliases checkout or input'; return 2; }
  case "$evidence_real/" in "$root_real/"*|"$gguf_real/"*|"$reference_real/"*|"$approval_real/"*) die 'evidence path overlaps checkout or input'; return 2 ;; esac
  case "$root_real/" in "$evidence_real/"*) die 'checkout overlaps evidence'; return 2 ;; esac
  case "$gguf_real/" in "$evidence_real/"*) die 'GGUF overlaps evidence'; return 2 ;; esac
  case "$reference_real/" in "$evidence_real/"*) die 'reference overlaps evidence'; return 2 ;; esac
  case "$approval_real/" in "$evidence_real/"*) die 'approval overlaps evidence'; return 2 ;; esac
}

license_preflight() {
  local approval="$1" gate_args=(--lock "$NANO_PROJECT/uv.lock" --project "$NANO_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST")
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] || { die 'Nano approval gate/manifest is missing or symlinked'; return 2; }
  [[ -f "$NANO_PROJECT/pyproject.toml" && ! -L "$NANO_PROJECT/pyproject.toml" ]] || { die 'Nano parity pyproject is missing or symlinked'; return 2; }
  grep -Fqx "url = \"$EXPECTED_CPU_TORCH_INDEX\"" "$NANO_PROJECT/pyproject.toml" \
    || { die 'Nano parity project CPU Torch index drifted'; return 2; }
  [[ -n "$approval" && -f "$approval" && ! -L "$approval" ]] || { die 'approval evidence must be a required regular non-symlink file'; return 2; }
  gate_args+=(--approval-evidence "$approval")
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${gate_args[@]}"
}

require_test_evidence() {
  local path="$1" named result result_lines test_lines cpu cpu_lines metal_reference metal_reference_lines metal_cpu metal_cpu_lines
  named="$(grep -Ec '^test official_nano_decode_measurement \.\.\. ok$' "$path" || true)"
  result="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)"
  result_lines="$(grep -Ec '^test result:' "$path" || true)"
  test_lines="$(awk '/^test / && $0 !~ /^test result:/ {count++} END {print count + 0}' "$path")"
  cpu="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ actual=[0-9.eE+-]+ reference=[0-9.eE+-]+$' "$path" || true)"
  metal_reference="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_reference numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ actual=[0-9.eE+-]+ reference=[0-9.eE+-]+$' "$path" || true)"
  metal_cpu="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ metal=[0-9.eE+-]+ cpu=[0-9.eE+-]+$' "$path" || true)"
  cpu_lines="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu ' "$path" || true)"
  metal_reference_lines="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_reference ' "$path" || true)"
  metal_cpu_lines="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_cpu ' "$path" || true)"
  if [[ "$named" != 1 || "$result" != 1 || "$result_lines" != 1 || "$test_lines" != 1 || "$cpu" != 1 || "$cpu_lines" != 1 || "$metal_reference" != 1 || "$metal_reference_lines" != 1 || "$metal_cpu" != 1 || "$metal_cpu_lines" != 1 ]]; then
    die 'Nano evidence requires exactly one named pass/result and CPU/reference, Metal/reference, Metal/CPU sentinels'; return 2
  fi
}

require_reference() {
  local path="$1" count role runtime model_source config_source quantizer_shape decoder_count decoder_shapes
  require_file 'VAST independent Nano reference' "$path"
  awk -F, -v source_row="source,nano,$OFFICIAL_REPO,$OFFICIAL_REVISION" -v codes_row="codes,$EXPECTED_CODES" '
    $0 == source_row ||
    $0 ~ /^runtime,torch-[^,]+,transformers-[^,]+$/ ||
    $0 ~ /^environment,cpu,[^,]+,machine-[^,]+,logical-[0-9]+,torch-capability-[^,]+$/ ||
    $0 == "environment,device,cpu" ||
    ($1 == "source_file" && ($2 == "model" || $2 == "config") && NF == 4 &&
      $3 ~ /^transformers_modules\/[^,]+$/ && length($4) == 64 && $4 !~ /[^0-9a-f]/) ||
    $0 == "contract,2,16,1024,48000,2,3840" ||
    $0 == codes_row ||
    $0 ~ /^tensor,(quantizer|decoder_[0-9]+),[0-9]+(x[0-9]+)+,[-+0-9.eE]+(,[-+0-9.eE]+)*$/ ||
    $0 ~ /^tensor,audio,1x2x7680,[-+0-9.eE]+(,[-+0-9.eE]+)*$/ { next }
    { exit 1 }
  ' "$path" || { die 'reference contains an unknown or malformed row'; return 2; }
  count="$(awk -F, -v wanted="source,nano,$OFFICIAL_REPO,$OFFICIAL_REVISION" '$0 == wanted {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference must contain exactly one pinned official source row'; return 2; }
  for role in model config; do
    count="$(awk -F, -v role="$role" '$1 == "source_file" && $2 == role {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die "reference must contain exactly one $role source row"; return 2; }
    awk -F, -v role="$role" '$1 == "source_file" && $2 == role {if (NF != 4 || $3 !~ /^transformers_modules\/[^,]+$/ || length($4) != 64 || $4 ~ /[^0-9a-f]/) exit 1; found=1} END {exit(found ? 0 : 1)}' "$path" || { die "reference $role source row is not authenticated"; return 2; }
  done
  runtime="$(awk -F, '$1 == "runtime" {print; count++} END {if (count != 1) exit 1}' "$path")" || { die 'reference must contain exactly one runtime row'; return 2; }
  [[ "$runtime" == "runtime,torch-${EXPECTED_TORCH_VERSION},transformers-${EXPECTED_TRANSFORMERS_VERSION}" ]] || { die 'reference Transformers route is not the reviewed exact route'; return 2; }
  [[ "$EXPECTED_TORCH_VERSION" == "$EXPECTED_CPU_TORCH_VERSION" ]] || { die 'reviewed Torch route is not the CPU Torch closure'; return 2; }
  [[ "$EXPECTED_TORCH_VERSION" != UNRESOLVED && "$EXPECTED_TRANSFORMERS_VERSION" != UNRESOLVED ]] || { die 'reference Transformers route is unresolved; owner review is required'; return 2; }
  model_source="$(awk -F, '$1 == "source_file" && $2 == "model" {print $3 "," $4}' "$path")"
  config_source="$(awk -F, '$1 == "source_file" && $2 == "config" {print $3 "," $4}' "$path")"
  [[ "$model_source" == "$EXPECTED_MODEL_SOURCE_PATH,$EXPECTED_MODEL_SOURCE_SHA256" && "$config_source" == "$EXPECTED_CONFIG_SOURCE_PATH,$EXPECTED_CONFIG_SOURCE_SHA256" ]] || { die 'reference source identities differ from reviewed fixed identities'; return 2; }
  [[ "$EXPECTED_MODEL_SOURCE_PATH" != UNRESOLVED && "$EXPECTED_CONFIG_SOURCE_PATH" != UNRESOLVED && "$EXPECTED_MODEL_SOURCE_SHA256" != UNRESOLVED && "$EXPECTED_CONFIG_SOURCE_SHA256" != UNRESOLVED ]] || { die 'reference source identities are unresolved; owner review is required'; return 2; }
  count="$(awk -F, '$1 == "environment" && $2 == "cpu" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference must contain exactly one CPU row'; return 2; }
  count="$(awk -F, '$0 == "environment,device,cpu" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference must contain exactly one CPU device row'; return 2; }
  count="$(awk -F, '$0 == "contract,2,16,1024,48000,2,3840" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference contract is not exact'; return 2; }
  count="$(awk -F, -v wanted="codes,$EXPECTED_CODES" '$0 == wanted {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference codes are not the exact deterministic 32-code packet'; return 2; }
  count="$(awk -F, '$0 ~ /^tensor,quantizer,/ {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference quantizer tap missing or duplicated'; return 2; }
  quantizer_shape="$(awk -F, '$1 == "tensor" && $2 == "quantizer" {print $3}' "$path")"
  [[ "$quantizer_shape" == "$EXPECTED_QUANTIZER_SHAPE" ]] || { die 'reference quantizer shape differs from reviewed contract'; return 2; }
  [[ "$EXPECTED_QUANTIZER_SHAPE" != UNRESOLVED ]] || { die 'reference quantizer shape is unresolved; owner review is required'; return 2; }
  count="$(awk -F, '$0 ~ /^tensor,audio,1x2x7680,/ {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference audio tap missing or duplicated'; return 2; }
  awk -F, '$1 == "tensor" && $2 ~ /^decoder_[0-9]+$/ {idx=$2; sub(/^decoder_/, "", idx); if (seen[idx]++) exit 1; count++} END {for (idx=0; idx<count; idx++) if (!(idx in seen)) exit 1}' "$path" || { die 'reference decoder tensor rows are not a unique contiguous sequence'; return 2; }
  decoder_count="$(awk -F, '$1 == "tensor" && $2 ~ /^decoder_[0-9]+$/ {count++} END {print count + 0}' "$path")"
  (( decoder_count > 0 )) || { die 'reference must contain a nonzero decoder tap sequence'; return 2; }
  [[ "$decoder_count" == "$EXPECTED_DECODER_TAP_COUNT" ]] || { die 'reference decoder tap count differs from reviewed contract'; return 2; }
  decoder_shapes="$(awk -F, '$1 == "tensor" && $2 ~ /^decoder_[0-9]+$/ {idx=$2; sub(/^decoder_/, "", idx); shape_by_idx[idx]=$3; count++} END {for (idx=0; idx<count; idx++) if (!(idx in shape_by_idx)) exit 1; for (idx=0; idx<count; idx++) {if (idx) printf ","; printf "%s", shape_by_idx[idx]}}' "$path")" || { die 'reference decoder shape sequence is malformed'; return 2; }
  [[ "$decoder_shapes" == "$EXPECTED_DECODER_TAP_SHAPES" ]] || { die 'reference decoder shape sequence differs from reviewed contract'; return 2; }
  [[ "$EXPECTED_DECODER_TAP_COUNT" != UNRESOLVED && "$EXPECTED_DECODER_TAP_SHAPES" != UNRESOLVED ]] || { die 'reference decoder tap contract is unresolved; owner review is required'; return 2; }
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution'
  [[ "$(uname -s)" == 'Darwin' ]] || die 'real Nano Metal measurement requires Darwin'
  [[ "$(uname -m)" == 'arm64' ]] || die 'real Nano Metal measurement requires Apple arm64'
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die 'could not read hw.memsize'
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die 'could not read free disk'
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
  [[ -f "$TEST_SOURCE" ]] || die "Nano test source is missing: $TEST_SOURCE"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die 'remote Apple checkout must be clean so evidence names one exact commit'
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

require_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head must be a lowercase 40-hex commit'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" ]] || { die 'expected-head check requires a git checkout'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'Apple checkout must be clean before approval/model work'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || { die 'cannot resolve Apple checkout HEAD'; return 2; }
  [[ "$actual" == "$expected" ]] || { die "Apple checkout HEAD $actual differs from --expected-head $expected"; return 2; }
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
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPHardwareDataType
    system_profiler SPDisplaysDataType
  } > "$output"
}

run_self_test() (
  local temporary temporary_parent script_path required
  temporary_parent="$(cd -P "${TMPDIR:-/tmp}" && pwd -P)" || die 'temporary test parent is unavailable'
  temporary="$(mktemp -d "$temporary_parent/vokra-moss-tokenizer-nano-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  expect_exit_2_no_path() {
    local label="$1" path="$2" status
    shift 2
    if "$@" >/dev/null 2>&1; then
      status=0
    else
      status=$?
    fi
    if [[ $status -ne 2 || -e "$path" || -L "$path" ]]; then
      die "self-test $label was not a controlled reject without evidence"
      return 1
    fi
  }
  expect_exit_2() {
    local label="$1" status
    shift
    if "$@" >/dev/null 2>&1; then
      status=0
    else
      status=$?
    fi
    [[ $status -eq 2 ]] || die "self-test $label did not return controlled exit 2"
  }
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' ]] \
    || die 'SHA-256 helper self-test failed'
  if require_absent_evidence_directory 'relative-evidence'; then die 'relative evidence path was accepted'; fi
  require_absent_evidence_directory "$temporary/evidence"
  mkdir -p "$temporary/root/nested"
  cp "$temporary/value" "$temporary/root/gguf"
  cp "$temporary/value" "$temporary/root/reference.csv"
  expect_exit_2_no_path 'checkout-contained evidence' "$temporary/root/nested/evidence" \
    require_disjoint_evidence "$temporary/root/nested/evidence" "$temporary/root/gguf" "$temporary/root/reference.csv" "$temporary/root" "$temporary/value"
  expect_exit_2_no_path 'lexical checkout overlap' "$temporary/root/nested/../lexical-alias-evidence" \
    require_disjoint_evidence "$temporary/root/nested/../lexical-alias-evidence" "$temporary/root/gguf" "$temporary/root/reference.csv" "$temporary/root" "$temporary/value"
  expect_exit_2 'evidence/input equality' \
    require_disjoint_evidence "$temporary/root/gguf" "$temporary/root/gguf" "$temporary/root/reference.csv" "$temporary/root" "$temporary/value"
  ln -s "$temporary/value" "$temporary/input-link"
  if require_file input "$temporary/input-link"; then die 'symlink input was accepted'; fi
  ln -s "$temporary/value" "$temporary/approval-link"
  expect_exit_2_no_path 'symlink approval' "$temporary/disjoint-evidence" \
    require_disjoint_evidence "$temporary/disjoint-evidence" "$temporary/value" "$temporary/value" "$temporary" "$temporary/approval-link"
  mkdir -p "$temporary/real-parent/child"
  ln -s "$temporary/real-parent" "$temporary/link-parent"
  expect_exit_2_no_path 'symlink evidence ancestor' "$temporary/link-parent/child/new-evidence" \
    require_disjoint_evidence "$temporary/link-parent/child/new-evidence" "$temporary/value" "$temporary/value" "$temporary" "$temporary/value"
  mkdir -p "$temporary/existing"
  if require_absent_evidence_directory "$temporary/existing"; then die 'pre-existing evidence directory was accepted'; fi
  ln -s "$temporary/value" "$temporary/evidence-link"
  if require_absent_evidence_directory "$temporary/evidence-link"; then die 'symlink evidence path was accepted'; fi
  script_path="${BASH_SOURCE[0]}"
  fake_root="$temporary/fake-root"
  mkdir -p "$fake_root/tools/parity/moss_audio_tokenizer_nano"
  cp "$NANO_PROJECT/license_gate.py" "$NANO_PROJECT/license_gate_manifest.json" "$NANO_PROJECT/uv.lock" "$NANO_PROJECT/pyproject.toml" "$fake_root/tools/parity/moss_audio_tokenizer_nano/"
  expect_exit_2_no_path 'missing approval evidence' "$temporary/missing-evidence" \
    env VOKRA_ROOT="$fake_root" VOKRA_REMOTE_APPLE_SILICON=0 "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --approval-evidence "$fake_root/missing-approval" --evidence-dir "$temporary/missing-evidence"
  expect_exit_2_no_path 'missing approval option' "$temporary/missing-option-evidence" \
    "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --evidence-dir "$temporary/missing-option-evidence"
  expect_exit_2_no_path 'duplicate approval option' "$temporary/duplicate-approval-evidence" \
    "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --approval-evidence "$temporary/value" --approval-evidence "$temporary/value" --evidence-dir "$temporary/duplicate-approval-evidence"
  expect_exit_2_no_path 'empty approval value' "$temporary/empty-approval-evidence" \
    "$script_path" --gguf "$temporary/value" --reference "$temporary/value" --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --approval-evidence "" --evidence-dir "$temporary/empty-approval-evidence"
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' \
    'xcrun -f metal' 'parity_moss_audio_tokenizer_nano_real.rs' "$TEST_NAME" \
    '--features metal --test parity_moss_audio_tokenizer_nano_real' \
    '-- --ignored --exact --nocapture' \
    '--gguf-sha256' '--reference-sha256' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'VOKRA_MOSS_AUDIO_TOKENIZER_NANO_GGUF' \
    'VOKRA_MOSS_AUDIO_TOKENIZER_NANO_REFERENCE' \
    'MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET' \
    'MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_reference numeric_bounds=UNSET' \
    'MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_cpu numeric_bounds=UNSET' \
    '--expected-head' \
    'verdict=MEASURED_NOT_GATED' 'numeric_bounds=UNSET' 'upload=NOT_PERFORMED'; do
    grep -Fq -- "$required" "$script_path" \
      || die "self-test contract token is missing: $required"
  done
  grep -Fq 'object_pairs_hook=reject' "$LICENSE_GATE" \
    || die 'approval gate duplicate-key rejection is missing'
  grep -Fq "REVISION = \"$OFFICIAL_REVISION\"" \
    "$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_nano/license_gate.py" \
    || die 'Apple/gate upstream revision contract diverged'
  grep -Fq "\"revision\": \"$OFFICIAL_REVISION\"" \
    "$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_dump_reference.py" \
    || die 'Apple/dumper upstream revision contract diverged'
  for required in \
    'EXPECTED_MODEL_SOURCE_PATH="transformers_modules/hf/cfb29bb1bac555fe/modeling_moss_audio_tokenizer.py"' \
    'EXPECTED_CONFIG_SOURCE_PATH="transformers_modules/hf/1f68fe91b6890e3e/configuration_moss_audio_tokenizer.py"' \
    'EXPECTED_MODEL_SOURCE_SHA256="b14af7c188944da5101adbd4aaa9c3617d66b83507f0efbd6eb416381a105930"' \
    'EXPECTED_CONFIG_SOURCE_SHA256="b2d67dc4581e70f4b69b2d7eccefe32581d0c5192fe4d97fe1830e94a255b8aa"' \
    'EXPECTED_TORCH_VERSION="2.7.1+cpu"' 'EXPECTED_TRANSFORMERS_VERSION="5.10.4"' \
    'EXPECTED_QUANTIZER_SHAPE="1x768x2"' 'EXPECTED_DECODER_TAP_COUNT="9"' \
    'EXPECTED_DECODER_TAP_SHAPES="1x192x8,1x768x8,1x384x16,1x768x16,1x384x32,1x768x32,1x384x64,1x240x64,1x1x15360"' \
    'EXPECTED_CPU_TORCH_VERSION="2.7.1+cpu"' \
    'EXPECTED_CPU_TORCH_INDEX="https://download.pytorch.org/whl/cpu"'; do
    grep -Fqx -- "$required" "$script_path" \
      || die "Apple reviewed Nano identity drifted: $required"
  done
  require_cross_contract "$VAST_CONTRACT_SCRIPT" "$script_path" \
    || die 'Apple/VAST Nano contract drifted or contains duplicate/unknown definitions'
  cp "$VAST_CONTRACT_SCRIPT" "$temporary/vast-contract-drift.sh"
  sed 's/^EXPECTED_DECODER_TAP_COUNT="9"$/EXPECTED_DECODER_TAP_COUNT="8"/' \
    "$temporary/vast-contract-drift.sh" > "$temporary/vast-contract-drift.tmp"
  mv "$temporary/vast-contract-drift.tmp" "$temporary/vast-contract-drift.sh"
  if require_cross_contract "$temporary/vast-contract-drift.sh" "$script_path"; then
    die 'Apple self-test accepted VAST contract drift'
  fi
  cp "$VAST_CONTRACT_SCRIPT" "$temporary/vast-contract-duplicate.sh"
  printf '%s\n' 'EXPECTED_DECODER_TAP_COUNT="9"' >> "$temporary/vast-contract-duplicate.sh"
  if require_cross_contract "$temporary/vast-contract-duplicate.sh" "$script_path"; then
    die 'Apple self-test accepted a duplicate VAST contract definition'
  fi
  cp "$VAST_CONTRACT_SCRIPT" "$temporary/vast-contract-unknown.sh"
  printf '%s\n' 'EXPECTED_UNTRUSTED="malicious"' >> "$temporary/vast-contract-unknown.sh"
  if require_cross_contract "$temporary/vast-contract-unknown.sh" "$script_path"; then
    die 'Apple self-test accepted an unknown VAST contract definition'
  fi
  grep -Fq '(frame * 257 + quantizer * 503 + 17) % CODEBOOK_SIZE' \
    "$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_dump_reference.py" \
    || die 'Apple/dumper deterministic code contract diverged'
  for token in 'require_transformers_api_smoke' 'BLOCKED_UNVERIFIED_API_SMOKE' 'transformers==5.10.4'; do
    grep -Fq "$token" "$VOKRA_ROOT/tools/parity/moss_audio_tokenizer_dump_reference.py" \
      || die "Apple/dumper blocker is missing: $token"
  done
  printf -v EXPECTED_MODEL_SOURCE_PATH '%s' 'transformers_modules/OpenMOSS-Team/Nano/model.py'
  printf -v EXPECTED_CONFIG_SOURCE_PATH '%s' 'transformers_modules/OpenMOSS-Team/Nano/config.py'
  printf -v EXPECTED_MODEL_SOURCE_SHA256 '%s' 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
  printf -v EXPECTED_CONFIG_SOURCE_SHA256 '%s' 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
  printf -v EXPECTED_TORCH_VERSION '%s' '2.13.0'
  printf -v EXPECTED_CPU_TORCH_VERSION '%s' '2.13.0'
  printf -v EXPECTED_TRANSFORMERS_VERSION '%s' '5.15.0'
  printf -v EXPECTED_QUANTIZER_SHAPE '%s' '1x1'
  printf -v EXPECTED_DECODER_TAP_COUNT '%s' '2'
  printf -v EXPECTED_DECODER_TAP_SHAPES '%s' '1x1,1x2'
  printf '%s\n' \
    "source,nano,$OFFICIAL_REPO,$OFFICIAL_REVISION" \
    'runtime,torch-2.13.0,transformers-5.15.0' \
    'environment,cpu,test,machine-x,logical-1,torch-capability-unknown' \
    'environment,device,cpu' \
    'source_file,model,transformers_modules/OpenMOSS-Team/Nano/model.py,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
    'source_file,config,transformers_modules/OpenMOSS-Team/Nano/config.py,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' \
    'contract,2,16,1024,48000,2,3840' \
    "codes,$EXPECTED_CODES" \
    'tensor,quantizer,1x1,0' 'tensor,decoder_0,1x1,0' \
    'tensor,decoder_1,1x2,0' 'tensor,audio,1x2x7680,0' > "$temporary/reference.csv"
  require_reference "$temporary/reference.csv" || die 'valid decoder shape contract rejected'
  for tamper in shape order extra missing hash; do
    cp "$temporary/reference.csv" "$temporary/reference-$tamper.csv"
    case "$tamper" in
      shape) sed 's/tensor,decoder_1,1x2/tensor,decoder_1,9x9/' "$temporary/reference-$tamper.csv" > "$temporary/reference-$tamper.tmp" ;;
      order) sed -e 's/tensor,decoder_0,1x1/tensor,decoder_0,1x2/' -e 's/tensor,decoder_1,1x2/tensor,decoder_1,1x1/' "$temporary/reference-$tamper.csv" > "$temporary/reference-$tamper.tmp" ;;
      extra) printf 'tensor,decoder_2,1x3,0\n' >> "$temporary/reference-$tamper.csv"; cp "$temporary/reference-$tamper.csv" "$temporary/reference-$tamper.tmp" ;;
      missing) sed '/tensor,decoder_1,/d' "$temporary/reference-$tamper.csv" > "$temporary/reference-$tamper.tmp" ;;
      hash) sed 's/^\(source_file,model,[^,]*,\)./\1A/' "$temporary/reference-$tamper.csv" > "$temporary/reference-$tamper.tmp" ;;
    esac
    mv "$temporary/reference-$tamper.tmp" "$temporary/reference-$tamper.csv"
    if require_reference "$temporary/reference-$tamper.csv"; then die "decoder $tamper tamper accepted"; fi
  done
  if grep -En -- '(^|[[:space:]])(curl|wget|pip|git[[:space:]]+(clone|fetch|pull|push))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    die 'download or external checkout command found'
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    die '--self-test accepted an extra argument'
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then
    die 'duplicate --self-test was accepted'
  fi
  if "$script_path" --gguf --reference x --gguf-sha256 "$(printf '%064d' 0)" --reference-sha256 "$(printf '%064d' 0)" --evidence-dir x >/dev/null 2>&1; then
    die 'bare option value was accepted'
  fi
  for option in gguf reference gguf-sha256 reference-sha256 approval-evidence expected-head evidence-dir; do
    if "$script_path" --self-test "--$option" -x >/dev/null 2>&1; then
      die "negative --$option value was accepted"
    fi
  done
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\nMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_reference numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\nMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal_cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 metal=1.0e-9 cpu=1.0e-9\n' official_nano_decode_measurement > "$temporary/test.log"
  require_test_evidence "$temporary/test.log"
  cp "$temporary/test.log" "$temporary/duplicate-result.log"
  printf 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n' >> "$temporary/duplicate-result.log"
  if require_test_evidence "$temporary/duplicate-result.log"; then die 'duplicate test result was accepted'; fi
  awk 'NR == 2 { print "test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"; next } { print }' "$temporary/test.log" > "$temporary/malformed-result.log"
  if require_test_evidence "$temporary/malformed-result.log"; then die 'malformed test result was accepted'; fi
  for bad in duplicate prefix suffix fail; do
    cp "$temporary/test.log" "$temporary/$bad-sentinel.log"
    case "$bad" in
      duplicate) cat "$temporary/test.log" >> "$temporary/$bad-sentinel.log" ;;
      prefix) sed 's/^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY /xMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY /' "$temporary/$bad-sentinel.log" > "$temporary/$bad-sentinel.tmp"; mv "$temporary/$bad-sentinel.tmp" "$temporary/$bad-sentinel.log" ;;
      suffix) sed 's/$/ trailing/' "$temporary/$bad-sentinel.log" > "$temporary/$bad-sentinel.tmp"; mv "$temporary/$bad-sentinel.tmp" "$temporary/$bad-sentinel.log" ;;
      fail) sed 's/verdict=MEASURED_NOT_GATED/verdict=FAIL/g' "$temporary/$bad-sentinel.log" > "$temporary/$bad-sentinel.tmp"; mv "$temporary/$bad-sentinel.tmp" "$temporary/$bad-sentinel.log" ;;
    esac
    if require_test_evidence "$temporary/$bad-sentinel.log"; then die "$bad sentinel was accepted"; fi
  done
  cp "$temporary/test.log" "$temporary/extra-test.log"
  { printf 'test extra_case ... ok\n'; cat "$temporary/test.log"; } > "$temporary/extra-test.tmp"
  mv "$temporary/extra-test.tmp" "$temporary/extra-test.log"
  if require_test_evidence "$temporary/extra-test.log"; then die 'extra named test was accepted'; fi
  cp "$temporary/test.log" "$temporary/failed-test.log"
  sed 's/\.\.\. ok$/... FAILED/' "$temporary/failed-test.log" > "$temporary/failed-test.tmp"
  mv "$temporary/failed-test.tmp" "$temporary/failed-test.log"
  if require_test_evidence "$temporary/failed-test.log"; then die 'failed named test was accepted'; fi
  sed 's/filtered out$/filtered out; finished in nope/' "$temporary/test.log" > "$temporary/bad-timing.log"
  if require_test_evidence "$temporary/bad-timing.log"; then die 'malformed timing was accepted'; fi
  log 'self-test PASS'
)

main() {
  local gguf='' reference='' gguf_sha='' reference_sha='' approval='' evidence_dir='' expected_head='' self_test=0
  local seen_gguf=0 seen_reference=0 seen_gguf_sha=0 seen_reference_sha=0 seen_approval=0 seen_evidence=0 seen_head=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf) (( ! seen_gguf++ )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf="$2"; shift 2 ;;
      --reference) (( ! seen_reference++ )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference="$2"; shift 2 ;;
      --gguf-sha256) (( ! seen_gguf_sha++ )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf_sha="$2"; shift 2 ;;
      --reference-sha256) (( ! seen_reference_sha++ )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference_sha="$2"; shift 2 ;;
      --approval-evidence) (( ! seen_approval++ )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; approval="$2"; shift 2 ;;
      --evidence-dir) (( ! seen_evidence++ )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; evidence_dir="$2"; shift 2 ;;
      --expected-head) (( ! seen_head++ )) || die 'duplicate --expected-head'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; expected_head="$2"; shift 2 ;;
      --self-test) (( ! seen_self_test++ )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$gguf$reference$gguf_sha$reference_sha$approval$evidence_dir$expected_head" ]] || die '--self-test accepts no other arguments'
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference" && -n "$gguf_sha" && -n "$reference_sha" && -n "$approval" && -n "$evidence_dir" && -n "$expected_head" ]] \
    || { usage; die '--gguf, --reference, both SHA-256 values, --approval-evidence, --expected-head, and --evidence-dir are required'; }
  [[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$reference_sha" =~ ^[0-9a-f]{64}$ ]] \
    || { die 'expected hashes must be lowercase 64-hex SHA-256 values'; return 2; }

  require_cross_contract "$VAST_CONTRACT_SCRIPT" "$APPLE_CONTRACT_SCRIPT" \
    || die 'Apple/VAST Nano contract drifted or contains duplicate/unknown definitions'
  require_expected_head "$expected_head"
  license_preflight "$approval"
  require_remote_apple_host
  require_tooling
  require_file 'VAST-produced corrected Nano GGUF' "$gguf"
  [[ "$(sha256_file "$gguf")" == "$gguf_sha" ]] || die 'GGUF SHA-256 differs from VAST evidence'
  require_file 'VAST independent Nano reference' "$reference"
  [[ "$(sha256_file "$reference")" == "$reference_sha" ]] || die 'reference SHA-256 differs from VAST evidence'
  require_reference "$reference"
  require_disjoint_evidence "$evidence_dir" "$gguf" "$reference" "$VOKRA_ROOT" "$approval"
  require_absent_evidence_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf_sha256=$gguf_sha"
    echo "reference_sha256=$reference_sha"
    echo "official_repo=$OFFICIAL_REPO"
    echo "official_revision=$OFFICIAL_REVISION"
    echo 'corrected_variant=nano'
  } > "$evidence_dir/input-hashes.txt"
  printf '%s  %s\n' "$(sha256_file "$reference")" "$(basename "$reference")" \
    > "$evidence_dir/reference-tree-sha256.txt"

  log 'running exact ignored real-weight CPU/reference/Metal measurement'
  env VOKRA_MOSS_AUDIO_TOKENIZER_NANO_GGUF="$gguf" \
    VOKRA_MOSS_AUDIO_TOKENIZER_NANO_REFERENCE="$reference" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --offline --release \
      -p vokra-models --features metal --test parity_moss_audio_tokenizer_nano_real \
      "$TEST_NAME" -- --ignored --exact --nocapture --test-threads=1 \
      2>&1 | tee "$evidence_dir/parity.log"
  require_test_evidence "$evidence_dir/parity.log"

  {
    echo 'verdict=MEASURED_NOT_GATED'
    echo 'parity_status=MEASURED_NOT_GATED'
    echo 'numeric_bounds=UNSET'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "expected_head=$expected_head"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_sha256=$reference_sha"
    echo "test=$TEST_SELECTOR"
    echo 'cpu_vs_upstream=MEASURED_NOT_GATED'
    echo 'metal_vs_upstream=MEASURED_NOT_GATED'
    echo 'metal_vs_cpu=MEASURED_NOT_GATED'
    echo 'upload=NOT_PERFORMED'
    echo 'apple_conversion=NOT_PERFORMED'
    echo 'input_conversion=VAST'
  } > "$evidence_dir/summary.txt"
  log 'MEASURED_NOT_GATED: pull only evidence, remove staged inputs, then destroy the remote worker'
}

main "$@"
