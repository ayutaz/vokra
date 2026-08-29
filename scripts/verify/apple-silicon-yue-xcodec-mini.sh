#!/usr/bin/env bash
# Disposable Apple Silicon CPU/Metal measurement for the exact VAST-staged
# YuE xcodec-mini public GGUF and independent upstream reference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/yue_xcodec_mini"
PRE_FLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PRE_FLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
REFERENCE_VALIDATOR="$VOKRA_ROOT/tools/parity/yue_xcodec_mini/reference_validator.py"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

PUBLIC_REPO="vokra/yue-xcodec-mini"
PUBLIC_REVISION="83c14a67ed792a0d5b3b61fff8ae35a04c6da8fa"
PUBLIC_BYTES=1810001760
PUBLIC_SHA256="60e21aa5335646080102196454d7ffad5e012467d6f5eb9b776bf07d666b02bc"
PUBLIC_TENSOR_COUNT=2145
REFERENCE_FILES=(manifest.json codes.u32le features.f32le backbone.f32le waveform.f32le)
REFERENCE_KEYS=(backbone_dtype bytes_backbone_f32le bytes_codes_u32le bytes_features_f32le bytes_waveform_f32le
  codes_dtype codebook_size codebooks codec_checkpoint_bytes codec_checkpoint_file codec_checkpoint_sha256
  contiguous decoder_checkpoint_bytes decoder_checkpoint_file decoder_checkpoint_sha256 device feature_dim features_dtype
  format frames output_hop_length output_sample_rate pickle_load_policy runtime samples sha256_backbone_f32le
  sha256_codes_u32le sha256_features_f32le sha256_waveform_f32le source_package source_package_wheel_sha256
  source_files_sha256 token_frame_rate token_hop_length token_sample_rate torch upstream_hf upstream_revision
  vocos_decoder_tensor_count waveform_dtype)
UPSTREAM_REPO="m-a-p/xcodec_mini_infer"
UPSTREAM_REVISION="fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/src/yue_xcodec_mini.rs"
CPU_TEST="measure_real_cpu_against_official_xcodec_and_vocos"
METAL_TEST="measure_real_metal_against_cpu_and_official_xcodec_and_vocos"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[yue-xcodec-mini-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-yue-xcodec-mini.sh \
  --gguf <vast-public-yue-xcodec-mini.gguf> --gguf-sha256 <sha256> \
  --reference <vast-reference-dir> --reference-manifest-sha256 <sha256> \
  --approval-evidence <external-evidence.json> \
  --evidence-dir <empty-dir>
       apple-silicon-yue-xcodec-mini.sh --self-test

Runs the exact ignored YuE xcodec-mini CPU and Metal real-weight tests using
the same staged GGUF/reference. The production binder strictly authenticates
the historical 2,145-tensor public artifact; corrected replacement binding is
a separate production task. Numeric results remain MEASURED_NOT_GATED.

The disposable host must be Darwin/arm64 with VOKRA_REMOTE_APPLE_SILICON=1,
at least 32 GB physical memory, 20 GB free disk, and Xcode's Metal compiler.
This verifier does not download, convert, publish, upload, or delete inputs.
EOF
}

pre_sync_gate() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die 'uv is required before the YuE Apple gate'
  [[ -f "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && -f "$PRE_FLIGHT_GATE" && -f "$PRE_FLIGHT_MANIFEST" ]] || die 'YuE gate inputs are missing'
  [[ -f "$approval" && ! -L "$approval" ]] || die 'approval evidence must be a regular file'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PRE_FLIGHT_GATE" \
    --project "$PARITY_PROJECT" --manifest "$PRE_FLIGHT_MANIFEST" --approval-evidence "$approval"
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] \
    || { die "$label is missing, empty, or symlinked: $path"; return 2; }
}

verify_file() {
  local label="$1" path="$2" expected_bytes="$3" expected_hash="$4" actual_bytes actual_hash
  require_file "$label" "$path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || { die "$label byte size $actual_bytes != $expected_bytes"; return 2; }
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || { die "$label SHA-256 $actual_hash != $expected_hash"; return 2; }
}

require_reference() {
  local directory="$1" manifest="$1/manifest.json" name key expected
  [[ -d "$directory" && ! -L "$directory" ]] \
    || { die 'reference root is missing or symlinked'; return 2; }
  require_file 'VAST YuE xcodec-mini reference manifest' "$manifest"
  local expected_keys actual_keys
  expected_keys="$(printf '%s\n' "${REFERENCE_KEYS[@]}" | sort)"
  actual_keys="$(sed -n 's/^  "\([^"]*\)":.*/\1/p' "$manifest" | sort)"
  [[ "$actual_keys" == "$expected_keys" && -z "$(printf '%s\n' "$actual_keys" | uniq -d)" ]] \
    || { die 'reference manifest key set is not exact'; return 2; }
  for name in "${REFERENCE_FILES[@]}"; do require_file "reference $name" "$directory/$name" || return 2; done
  local actual_files
  actual_files="$(find "$directory" -mindepth 1 -maxdepth 1 -exec basename {} \; | sort)"
  [[ "$actual_files" == "$(printf '%s\n' "${REFERENCE_FILES[@]}" | sort)" ]] \
    || { die 'reference file set is not exact'; return 2; }
  for key in format pickle_load_policy upstream_hf upstream_revision codec_checkpoint_sha256 decoder_checkpoint_sha256 frames codebooks codebook_size feature_dim token_sample_rate token_hop_length token_frame_rate output_sample_rate output_hop_length samples runtime device codes_dtype features_dtype backbone_dtype waveform_dtype contiguous; do
    grep -Eq "^  \"$key\":" "$manifest" || die "reference manifest field missing: $key"
  done
  grep -Eq '^  "format": "vokra-yue-xcodec-mini-reference-v2",$' "$manifest" || die 'reference format mismatch'
  grep -Eq '^  "pickle_load_policy": "weights_only=True_required",$' "$manifest" || die 'unsafe pickle policy'
  grep -Eq '^  "upstream_revision": "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5",$' "$manifest" || die 'source revision mismatch'
  grep -Eq '^  "frames": 5,$' "$manifest" || die 'reference frame count mismatch'
  grep -Eq '^  "codebooks": 12,$' "$manifest" || die 'reference codebook count mismatch'
  grep -Eq '^  "feature_dim": 1024,$' "$manifest" || die 'reference feature dimension mismatch'
  grep -Eq '^  "runtime": "torch-cpu",$' "$manifest" || die 'reference runtime mismatch'
  grep -Eq '^  "device": "cpu",$' "$manifest" || die 'reference device mismatch'
  grep -Eq '^  "contiguous": true,$' "$manifest" || die 'reference tensors are not contiguous'
  for name in "${REFERENCE_FILES[@]:1}"; do
    key="sha256_${name//./_}"
    expected="$(sed -n "s/^  \"$key\": \"\([0-9a-f]*\)\",$/\1/p" "$manifest")"
    verify_file "reference $name" "$directory/$name" "$(sed -n "s/^  \"bytes_${name//./_}\": \([0-9]*\),$/\1/p" "$manifest")" "$expected"
  done
}

require_one_cargo_result() {
  local log_path="$1" test_name="$2"
  [[ "$(grep -Ec "^test $test_name \.\.\. ok$" "$log_path" || true)" == 1 ]] \
    || { die "named Cargo test did not pass exactly once: $test_name"; return 2; }
  [[ "$(grep -Ec '^test [^ ]+ \.\.\.' "$log_path" || true)" == 1 ]] \
    || { die 'Cargo emitted extra or missing test lines'; return 2; }
  [[ "$(grep -Ec '^test result:' "$log_path" || true)" == 1 ]] \
    || { die 'Cargo emitted extra or missing result lines'; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" \
    || { die 'Cargo result is not the exact one-pass result'; return 2; }
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == '1' ]] \
    || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution'
  [[ "$(uname -s)" == 'Darwin' ]] || die 'Metal measurement requires Darwin'
  [[ "$(uname -m)" == 'arm64' ]] || die 'Metal measurement requires Apple arm64'
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die 'could not read hw.memsize'
  (( memory_bytes >= MIN_MEMORY_BYTES )) || die 'physical memory is below 32-GB guard'
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die 'could not read free disk'
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) || die 'free disk is below 20-GB guard'
}

require_tooling() {
  local tool
  for tool in cargo rustc git uv shasum awk find tee grep sysctl sw_vers system_profiler xcrun wc tr sort; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die "$VOKRA_ROOT is not a Vokra checkout"
  [[ -f "$TEST_SOURCE" ]] || die "YuE xcodec-mini test source is missing: $TEST_SOURCE"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'remote Apple checkout must be clean'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
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

require_marker() {
  local log_path="$1" backend="$2" family=''
  family="$(grep -Ec "^YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=$backend " "$log_path" || true)"
  [[ "$family" == 1 ]] || { die "$backend measurement sentinel family is not singleton"; return 2; }
  [[ "$(grep -Ec "^YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=$backend numeric_bounds=UNSET verdict=MEASURED_NOT_GATED$" "$log_path" || true)" == 1 ]] \
    || { die "$backend measurement marker is not exactly one full line"; return 2; }
}

run_self_test() (
  local temporary script_path required reference_root manifest hash name
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-yue-xcodec-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' ]] || die 'SHA-256 helper self-test failed'
  script_path="${BASH_SOURCE[0]}"
  for required in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' 'xcrun -f metal' \
    'yue_xcodec_mini.rs' "$CPU_TEST" "$METAL_TEST" \
    '--features metal --lib' '-- --ignored --exact --nocapture' \
    'VOKRA_YUE_XCODEC_MINI_GGUF' 'VOKRA_YUE_XCODEC_MINI_REFERENCE_DIR' \
    '--approval-evidence' 'pre_sync_gate' 'canonical_path' 'paths_overlap' \
    'YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET' \
    'YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET' \
    'verdict=MEASURED_NOT_GATED' 'numeric_bounds=UNSET' 'upload=NOT_PERFORMED'; do
    grep -Fq -- "$required" "$script_path" || die "self-test contract token is missing: $required"
  done
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nYUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED\n' "$CPU_TEST" > "$temporary/cpu.log"
  require_one_cargo_result "$temporary/cpu.log" "$CPU_TEST"
  require_marker "$temporary/cpu.log" cpu
  printf '\nYUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED\n' >> "$temporary/cpu.log"
  if require_marker "$temporary/cpu.log" cpu; then die 'duplicate sentinel accepted'; fi
  printf 'YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED trailing\n' > "$temporary/suffix.log"
  if require_marker "$temporary/suffix.log" cpu; then die 'sentinel suffix accepted'; fi
  : > "$temporary/fail.log"
  printf 'YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=FAIL\n' >> "$temporary/fail.log"
  if require_marker "$temporary/fail.log" cpu; then die 'sentinel failure line accepted'; fi
  printf 'test extra ... ok\n' >> "$temporary/cpu.log"
  if require_one_cargo_result "$temporary/cpu.log" "$CPU_TEST"; then die 'extra Cargo test line accepted'; fi
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in nope\n' "$CPU_TEST" > "$temporary/malformed-result.log"
  if require_one_cargo_result "$temporary/malformed-result.log" "$CPU_TEST"; then die 'malformed Cargo timing accepted'; fi
  reference_root="$temporary/reference"
  mkdir "$reference_root"
  printf 'abc' > "$reference_root/value"
  hash='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
  for name in codes.u32le features.f32le backbone.f32le waveform.f32le; do
    printf 'abc' > "$reference_root/$name"
  done
  manifest="$reference_root/manifest.json"
  {
    echo '{'
    printf '  "backbone_dtype": "float32-le",\n'
    for name in backbone_f32le codes_u32le features_f32le waveform_f32le; do
      printf '  "bytes_%s": 3,\n' "$name"
    done
    printf '  "codes_dtype": "uint32-le",\n  "codebook_size": 1024,\n  "codebooks": 12,\n'
    printf '  "codec_checkpoint_bytes": 1,\n  "codec_checkpoint_file": "codec",\n  "codec_checkpoint_sha256": "%s",\n' "$hash"
    printf '  "contiguous": true,\n  "decoder_checkpoint_bytes": 1,\n  "decoder_checkpoint_file": "decoder",\n  "decoder_checkpoint_sha256": "%s",\n' "$hash"
    printf '  "device": "cpu",\n  "feature_dim": 1024,\n  "features_dtype": "float32-le",\n  "format": "vokra-yue-xcodec-mini-reference-v2",\n  "frames": 5,\n'
    printf '  "output_hop_length": 882,\n  "output_sample_rate": 44100,\n  "pickle_load_policy": "weights_only=True_required",\n  "runtime": "torch-cpu",\n  "samples": 4410,\n'
    for name in backbone_f32le codes_u32le features_f32le waveform_f32le; do
      printf '  "sha256_%s": "%s",\n' "$name" "$hash"
    done
    printf '  "source_package": "vocos==0.1.0",\n  "source_package_wheel_sha256": "%s",\n  "source_files_sha256": {},\n' "$hash"
    printf '  "token_frame_rate": 50,\n  "token_hop_length": 320,\n  "token_sample_rate": 16000,\n  "torch": "2.7.1",\n'
    printf '  "upstream_hf": "m-a-p/xcodec_mini_infer",\n  "upstream_revision": "%s",\n  "vocos_decoder_tensor_count": 81,\n  "waveform_dtype": "float32-le"\n' "$UPSTREAM_REVISION"
    echo '}'
  } > "$manifest"
  rm "$reference_root/value"
  require_reference "$reference_root"
  printf 'extra\n' > "$reference_root/extra"
  if require_reference "$reference_root"; then die 'extra reference file accepted'; fi
  rm "$reference_root/extra"
  mv "$reference_root/codes.u32le" "$reference_root/codes.real"
  ln -s codes.real "$reference_root/codes.u32le"
  if require_reference "$reference_root"; then die 'expected-name symlink accepted'; fi
  rm "$reference_root/codes.u32le"
  mv "$reference_root/codes.real" "$reference_root/codes.u32le"
  cp "$reference_root"/manifest.json "$reference_root/manifest.real"
  printf '  "extra_key": true,\n' >> "$manifest"
  if require_reference "$reference_root"; then die 'extra manifest key accepted'; fi
  mv "$reference_root/manifest.real" "$manifest"
  mv "$reference_root" "$reference_root.real"
  ln -s "$(basename "$reference_root.real")" "$reference_root"
  if require_reference "$reference_root"; then die 'reference-root symlink accepted'; fi
  if grep -En -- '^[[:space:]]*(curl|wget|python3?|pip)([[:space:]]|$)|git[[:space:]]+(clone|fetch|pull|push)' "$script_path" >/dev/null; then
    die 'download or external checkout command found'
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    die '--self-test accepted an extra argument'
  fi
  for option in --gguf --gguf-sha256 --reference --reference-manifest-sha256 --approval-evidence --evidence-dir; do
    if "$script_path" "$option" -bad >/dev/null 2>&1; then die "leading-dash value accepted for $option"; fi
  done
  for option in --gguf --gguf-sha256 --reference --reference-manifest-sha256 --approval-evidence --evidence-dir; do
    if "$script_path" "$option" one "$option" two >/dev/null 2>&1; then die "duplicate option accepted for $option"; fi
  done
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then die 'duplicate --self-test accepted'; fi
  paths_overlap /tmp/yue-a /tmp/yue-a || die 'equal paths not detected as overlap'
  paths_overlap /tmp/yue-a /tmp/yue-a/child || die 'containment paths not detected as overlap'
  if paths_overlap /tmp/yue-a /tmp/yue-b; then die 'disjoint paths reported as overlap'; fi
  if VOKRA_ROOT="$VOKRA_ROOT" "$script_path" --gguf "$temporary/model.gguf" --gguf-sha256 "$(printf '%064d' 0)" \
    --reference "$temporary/reference" --reference-manifest-sha256 "$(printf '%064d' 0)" \
    --approval-evidence "$temporary/missing-approval.json" --evidence-dir "$temporary/evidence" >/dev/null 2>&1; then
    die 'missing approval unexpectedly passed'
  fi
  [[ ! -e "$temporary/evidence" ]] || die 'missing approval created evidence'
  log 'self-test PASS'
)

canonical_path() {
  local value="$1" parent base
  parent="$(dirname "$value")"; base="$(basename "$value")"
  [[ -d "$parent" ]] || { die "path parent is missing: $value"; return 2; }
  (cd -P "$parent" && printf '%s/%s' "$PWD" "$base")
}

paths_overlap() {
  local left="$1" right="$2"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

main() {
  local gguf='' gguf_sha='' reference='' reference_sha='' approval='' evidence_dir='' self_test=0 seen=''
  while (( $# > 0 )); do
    case "$1" in
      --gguf) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$seen" != *'|gguf|'* ]] || { usage; return 2; }; seen+="|gguf|"; gguf="$2"; shift 2 ;;
      --gguf-sha256) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$seen" != *'|gguf_sha|'* ]] || { usage; return 2; }; seen+="|gguf_sha|"; gguf_sha="$2"; shift 2 ;;
      --reference) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$seen" != *'|reference|'* ]] || { usage; return 2; }; seen+="|reference|"; reference="$2"; shift 2 ;;
      --reference-manifest-sha256) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$seen" != *'|reference_sha|'* ]] || { usage; return 2; }; seen+="|reference_sha|"; reference_sha="$2"; shift 2 ;;
      --approval-evidence) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$seen" != *'|approval|'* ]] || { usage; return 2; }; seen+="|approval|"; approval="$2"; shift 2 ;;
      --evidence-dir) [[ $# -ge 2 && -n "$2" && "$2" != -* && "$seen" != *'|evidence|'* ]] || { usage; return 2; }; seen+="|evidence|"; evidence_dir="$2"; shift 2 ;;
      --self-test) [[ "$seen" != *'|self_test|'* ]] || { usage; return 2; }; seen+="|self_test|"; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$gguf$gguf_sha$reference$reference_sha$approval$evidence_dir" ]] || die '--self-test accepts no other arguments'
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$gguf_sha" && -n "$reference" && -n "$reference_sha" && -n "$approval" && -n "$evidence_dir" ]] || { usage; die 'all explicit GGUF/reference hashes, approval evidence, and --evidence-dir are required'; }

  pre_sync_gate "$approval"
  [[ ! -e "$evidence_dir" ]] || die 'evidence directory must be absent before validation'
  local root_real gguf_real reference_real approval_real evidence_real
  root_real="$(canonical_path "$VOKRA_ROOT")" || return 2
  gguf_real="$(canonical_path "$gguf")" || return 2
  reference_real="$(canonical_path "$reference")" || return 2
  approval_real="$(canonical_path "$approval")" || return 2
  evidence_real="$(canonical_path "$evidence_dir")" || return 2
  for existing in "$root_real" "$gguf_real" "$reference_real" "$approval_real"; do
    paths_overlap "$evidence_real" "$existing" && die 'evidence path overlaps an input or checkout'
  done

  require_remote_apple_host
  require_tooling
  [[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$reference_sha" =~ ^[0-9a-f]{64}$ ]] || die 'expected hashes must be lowercase 64-hex'
  [[ "$gguf_sha" == "$PUBLIC_SHA256" ]] || die 'GGUF expected hash is not the fixed public identity'
  verify_file 'exact public YuE xcodec-mini GGUF' "$gguf" "$PUBLIC_BYTES" "$gguf_sha"
  verify_file 'reference manifest' "$reference/manifest.json" "$(wc -c < "$reference/manifest.json" | tr -d '[:space:]')" "$reference_sha"
  require_reference "$reference"
  [[ -f "$REFERENCE_VALIDATOR" ]] || die 'shared YuE reference validator is missing'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE_VALIDATOR" --reference "$reference"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_gguf_bytes=$PUBLIC_BYTES"
    echo "public_gguf_sha256=$PUBLIC_SHA256"
    echo "public_tensor_count=$PUBLIC_TENSOR_COUNT"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    echo 'runtime_artifact=exact_public_gguf'
    echo 'corrected_replacement_binding=SEPARATE_PRODUCTION_TASK'
  } > "$evidence_dir/input-hashes.txt"

  log 'running exact ignored CPU real-weight measurement'
  VOKRA_YUE_XCODEC_MINI_GGUF="$gguf" VOKRA_YUE_XCODEC_MINI_REFERENCE_DIR="$reference" \
    RUST_TEST_THREADS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --lib "$CPU_TEST" -- --ignored --exact --nocapture --test-threads=1 \
      2>&1 | tee "$evidence_dir/cpu.log"
  require_one_cargo_result "$evidence_dir/cpu.log" "$CPU_TEST"
  require_marker "$evidence_dir/cpu.log" cpu

  log 'running exact ignored Metal-vs-CPU real-weight measurement'
  VOKRA_YUE_XCODEC_MINI_GGUF="$gguf" VOKRA_YUE_XCODEC_MINI_REFERENCE_DIR="$reference" \
    RUST_TEST_THREADS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --lib "$METAL_TEST" -- --ignored --exact --nocapture --test-threads=1 \
      2>&1 | tee "$evidence_dir/metal.log"
  require_one_cargo_result "$evidence_dir/metal.log" "$METAL_TEST"
  require_marker "$evidence_dir/metal.log" metal

  {
    echo 'verdict=MEASURED_NOT_GATED'
    echo 'numeric_bounds=UNSET'
    echo 'cpu_reference=MEASURED_NOT_GATED'
    echo 'metal_vs_cpu=MEASURED_NOT_GATED'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    echo 'runtime_artifact=exact_public_gguf'
    echo 'corrected_replacement_binding=SEPARATE_PRODUCTION_TASK'
    echo 'upload=NOT_PERFORMED'
  } > "$evidence_dir/summary.txt"
  log 'MEASURED_NOT_GATED: pull evidence, remove staged inputs, then destroy the remote worker'
}

main "$@"
