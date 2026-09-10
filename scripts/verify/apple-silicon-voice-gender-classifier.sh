#!/usr/bin/env bash
# Real-weight voice-gender CPU/reference, Metal/reference, and Metal-vs-CPU parity on a
# disposable Apple Silicon host. GGUF and fixtures must already be staged by
# the VAST worker; this verifier never downloads, converts, publishes, or
# manufactures a test result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_voice_gender_classifier.rs"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/voice_gender_classifier_dump_reference.py"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000
UPSTREAM_REPOSITORY="https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_REVISION="49bcbecfd929ba5a043bde645fdff1a375eb79c7"
UPSTREAM_HF_REVISION="db1222153bd60337e900be22add7af180452adc0"
UPSTREAM_HF_FILE="model.safetensors"
CHECKPOINT_BYTES=61907512
CHECKPOINT_SHA256="2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5"
CORRECTED_GGUF_SHA256="afb03696d8a640d5d701ea0c136bb065cac648cbfe905a5dcc4eae04e0769b1a"
UPSTREAM_LICENSE_FILE="LICENSE"
UPSTREAM_LICENSE_SPDX="MIT"
UPSTREAM_LICENSE_COPYRIGHT="Copyright (c) 2024 jaesunghuh"
UPSTREAM_HF_LICENSE="mit"
FP32_PARITY_BOUND="0.010000000"
FIXTURE_KIND="official_canned_synthetic_tone"

log() { printf '[voice-gender-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

preflight_gate() {
  [[ "$UPSTREAM_REPOSITORY" == "https://github.com/JaesungHuh/voice-gender-classifier.git" ]] \
    || die "upstream repository contract drifted"
  [[ "$UPSTREAM_REVISION" == "49bcbecfd929ba5a043bde645fdff1a375eb79c7" ]] \
    || die "upstream source revision contract drifted"
  [[ "$UPSTREAM_HF_REVISION" == "db1222153bd60337e900be22add7af180452adc0" ]] \
    || die "upstream Hub revision contract drifted"
  [[ "$UPSTREAM_HF_FILE" == "model.safetensors" && "$CHECKPOINT_BYTES" == 61907512 ]] \
    || die "upstream checkpoint file contract drifted"
  [[ "$CHECKPOINT_SHA256" == "2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5" ]] \
    || die "fixed checkpoint digest contract drifted"
  [[ "$UPSTREAM_LICENSE_FILE" == "LICENSE" && "$UPSTREAM_LICENSE_SPDX" == "MIT" ]] \
    || die "upstream license contract drifted"
  [[ "$UPSTREAM_LICENSE_COPYRIGHT" == "Copyright (c) 2024 jaesunghuh" ]] \
    || die "upstream license copyright contract drifted"
  [[ "$UPSTREAM_HF_LICENSE" == "mit" ]] || die "HF cardData license contract drifted"
}

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-voice-gender-classifier.sh \
  --gguf <vast-corrected-voice-gender-classifier.gguf> --gguf-sha256 <HEX64> \
  --reference-dir <vast-voice-gender-fixtures> --reference-manifest-sha256 <HEX64> \
  --evidence-dir <absent-dir>
       apple-silicon-voice-gender-classifier.sh --self-test

Runs the exact official CPU/reference, Metal/reference, and Metal-vs-CPU tests using only VAST
outputs. It requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least
32 GB physical memory, free disk, a clean checkout, and the Metal compiler.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

valid_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "$2 must be lowercase SHA-256"; }

reject_path() {
  local path="$1" label="$2" current="$1"
  [[ "$path" == /* ]] || die "$label must be absolute"
  case "$path" in */./*|*/../*|./*|../*|*/.|*/..) die "$label contains a lexical dot component";; esac
  while :; do
    [[ ! -L "$current" ]] || die "$label has symlink ancestry: $current"
    [[ "$current" == / ]] && break
    current="$(dirname "$current")"
  done
}

scope() {
  local path="$1" suffix='' name parent
  reject_path "$path" path
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"
    [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"
    [[ "$parent" == "$path" ]] && parent=/
    path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix") || die "cannot canonicalize scope: $1"
}

disjoint() {
  local left right
  left="$(scope "$1")"; right="$(scope "$2")"
  [[ "$left" != "$right" && "$left" != "$right"/* && "$right" != "$left"/* ]] \
    || die "paths overlap: $1 and $2"
}

require_file() {
  local label="$1" path="$2"
  reject_path "$path" "$label"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

directory_mode() {
  case "$(uname -s)" in
    Darwin) stat -f '%Lp' "$1" ;;
    Linux) stat -c '%a' "$1" ;;
    *) die "unsupported platform for directory mode check" ;;
  esac
}

require_empty_directory() {
  local directory="$1" parent
  reject_path "$directory" evidence
  [[ ! -e "$directory" && ! -L "$directory" ]] \
    || die "evidence directory must be absent and not a symlink: $directory"
  parent="$(dirname "$directory")"
  [[ -d "$parent" && ! -L "$parent" ]] || die "evidence parent must exist and be non-symlink: $parent"
  mkdir -m 700 "$directory"
  [[ "$(directory_mode "$directory")" == 700 ]] || die "evidence directory mode is not 700"
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution"
  [[ "$(uname -s)" == Darwin ]] || die "voice-gender Metal parity requires Darwin"
  [[ "$(uname -m)" == arm64 ]] || die "voice-gender Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the exact 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the exact 10-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun stat uv; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] || die "parity source is missing: $PARITY_SOURCE"
  grep -Fq 'fn real_voice_gender_classifier_matches_official_reference' \
    "$PARITY_SOURCE" || die "official CPU parity test is missing"
  grep -Fq 'VOICE_GENDER_OFFICIAL_PARITY PASS' "$PARITY_SOURCE" \
    || die "CPU parity test does not own its measurement marker"
  grep -Fq 'VOICE_GENDER_METAL_VS_CPU PASS' "$PARITY_SOURCE" \
    || die "Metal parity test does not own its measurement marker"
  grep -Fq 'VOICE_GENDER_METAL_VS_REFERENCE PASS' "$PARITY_SOURCE" \
    || die "Metal/reference parity test does not own its measurement marker"
  grep -Fq 'VOICE_GENDER_ARGMAX_LABEL_AGREEMENT PASS' "$PARITY_SOURCE" \
    || die "argmax/label agreement marker is missing"
  grep -Fq 'BackendKind::Metal' "$PARITY_SOURCE" \
    || die "parity source lacks explicit Metal backend selection"
  grep -Fq 'max_abs(&metal_logits, &actual_logits)' "$PARITY_SOURCE" \
    || die "Metal/CPU finite metric assertion is missing"
  grep -Fq 'VOKRA_VOICE_GENDER_FIXTURE_KIND' "$PARITY_SOURCE" \
    || die "fixed synthetic fixture identity gate is missing"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

require_reference() {
  local directory="$1" expected_sha="$2"
  reject_path "$directory" reference
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference directory is missing: $directory"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$directory" "$expected_sha" <<'PY'
import hashlib, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
expected_manifest_sha = sys.argv[2]
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
expected_files = {"pcm.f32", "features.f32", "embedding.f32", "logits.f32", "probabilities.f32", "argmax.u32", "meta.json"}
entries = list(root.iterdir())
if {entry.name for entry in entries} != expected_files or any(entry.is_symlink() or not entry.is_file() for entry in entries):
    raise SystemExit("reference packet file set or regular-file contract mismatch")
manifest_path = root / "meta.json"
manifest_bytes = manifest_path.read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
    raise SystemExit("reference manifest digest mismatch")
manifest = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=pairs)
expected = {"dumper_version", "upstream_repository", "upstream_revision", "upstream_hf_revision", "checkpoint_file", "checkpoint_bytes", "upstream_class", "checkpoint_sha256", "upstream_license", "upstream_license_file", "upstream_license_copyright", "upstream_hf_license", "checkpoint_identity_status", "sample_rate", "n_mels", "n_fft", "win_length", "hop_length", "feature_frames", "feature_dim", "embedding_dim", "class_labels", "outputs"}
if set(manifest) != expected:
    raise SystemExit("reference metadata schema mismatch")
fixed = {
    "dumper_version": 3,
    "upstream_repository": "https://github.com/JaesungHuh/voice-gender-classifier.git",
    "upstream_revision": "49bcbecfd929ba5a043bde645fdff1a375eb79c7",
    "upstream_hf_revision": "db1222153bd60337e900be22add7af180452adc0",
    "checkpoint_file": "model.safetensors",
    "checkpoint_bytes": 61907512,
    "upstream_class": "model.ECAPA_gender",
    "checkpoint_sha256": "2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5",
    "upstream_license": "MIT",
    "upstream_license_file": "LICENSE",
    "upstream_license_copyright": "Copyright (c) 2024 jaesunghuh",
    "upstream_hf_license": "mit",
    "checkpoint_identity_status": "AUTHENTICATED_FIXED",
    "sample_rate": 16000,
    "n_mels": 80,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "feature_dim": 80,
    "embedding_dim": 192,
    "class_labels": ["male", "female"],
}
for key, value in fixed.items():
    if manifest[key] != value:
        raise SystemExit(f"reference identity mismatch: {key}")
if type(manifest["feature_frames"]) is not int or manifest["feature_frames"] <= 0:
    raise SystemExit("reference feature_frames must be a positive integer")
outputs = manifest["outputs"]
if not isinstance(outputs, dict) or set(outputs) != expected_files - {"meta.json"}:
    raise SystemExit("reference output set mismatch")
for name, record in outputs.items():
    if not isinstance(record, dict) or set(record) != {"sha256", "bytes"} or type(record["bytes"]) is not int or record["bytes"] <= 0:
        raise SystemExit(f"reference output record mismatch: {name}")
    data = (root / name).read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise SystemExit(f"reference output digest mismatch: {name}")
expected_bytes = {
    "pcm.f32": 32000 * 4,
    "features.f32": manifest["feature_frames"] * 80 * 4,
    "embedding.f32": 192 * 4,
    "logits.f32": 2 * 4,
    "probabilities.f32": 2 * 4,
    "argmax.u32": 4,
}
for name, size in expected_bytes.items():
    if outputs[name]["bytes"] != size:
        raise SystemExit(f"reference shape/byte contract mismatch: {name}")
PY
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "upstream_repository=$UPSTREAM_REPOSITORY"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "uname=$(uname -a)"
    echo "machine=$(uname -m)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPDisplaysDataType
  } > "$output"
}

hash_reference_directory() {
  local directory="$1" output="$2" path
  find "$directory" -mindepth 1 -maxdepth 1 -type f -print \
    | LC_ALL=C sort \
    | while IFS= read -r path; do
        printf '%s  %s\n' "$(sha256_file "$path")" "${path#"$directory"/}"
      done > "$output"
}

require_cargo_singleton() {
  local log_path="$1" summary_pattern test_line_count named_line_count inline_pass_count standalone_pass_count named_line standalone_line summary_line
  summary_pattern='^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$'
  test_line_count="$(grep -Ec '^test [^[:space:]].* \.\.\.( ok)?[[:space:]]*$' "$log_path" || true)"
  named_line_count="$(grep -Ec '^test real_voice_gender_classifier_matches_official_reference \.\.\.( ok)?[[:space:]]*$' "$log_path" || true)"
  inline_pass_count="$(grep -Ec '^test real_voice_gender_classifier_matches_official_reference \.\.\. ok[[:space:]]*$' "$log_path" || true)"
  standalone_pass_count="$(grep -Ec '^ok[[:space:]]*$' "$log_path" || true)"
  if [[ "$test_line_count" != 1 || "$named_line_count" != 1 ]]; then
    die "Cargo did not report exactly one test case result"
    return 2
  fi
  if [[ $((inline_pass_count + standalone_pass_count)) != 1 ]]; then
    die "official CPU parity test did not pass exactly once"
    return 2
  fi
  if [[ "$(grep -Ec "$summary_pattern" "$log_path" || true)" != 1 ]]; then
    die "parity log does not prove one exact passing result"
    return 2
  fi
  if [[ "$inline_pass_count" == 0 ]]; then
    named_line="$(grep -n '^test real_voice_gender_classifier_matches_official_reference \.\.\.' "$log_path" | cut -d: -f1)"
    standalone_line="$(grep -n '^ok[[:space:]]*$' "$log_path" | cut -d: -f1)"
    summary_line="$(grep -nE "$summary_pattern" "$log_path" | cut -d: -f1)"
    if [[ -z "$named_line" || -z "$standalone_line" || -z "$summary_line" || "$standalone_line" -le "$named_line" || "$standalone_line" -ge "$summary_line" ]]; then
      die "standalone Cargo status is not between the exact named test and result lines"
      return 2
    fi
  fi
}

verify_parity_log() {
  local log_path="$1" cpu_metrics metal_metrics marker_count pass_count
  marker_count="$(grep -Ec '^VOICE_GENDER_OFFICIAL_PARITY(_METRICS| ).*$' "$log_path" || true)"
  [[ "$marker_count" == 2 ]] || { die "CPU parity marker count is not exactly 2: $marker_count"; return 2; }
  cpu_metrics="$(grep -E '^VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=[0-9]+\.[0-9]{9} embedding_max_abs=[0-9]+\.[0-9]{9} logits_max_abs=[0-9]+\.[0-9]{9} probability_max_abs=[0-9]+\.[0-9]{9} bound=0\.010000000 fixture=official_canned_synthetic_tone$' "$log_path" || true)"
  [[ "$(printf '%s\n' "$cpu_metrics" | wc -l | tr -d '[:space:]')" == 1 && -n "$cpu_metrics" ]] || { die "CPU parity metrics marker is missing or malformed"; return 2; }
  pass_count="$(grep -Ec '^VOICE_GENDER_OFFICIAL_PARITY PASS bound=0\.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream$' "$log_path" || true)"
  [[ "$pass_count" == 1 ]] || { die "CPU parity PASS marker is missing, duplicated, or malformed"; return 2; }
  printf '%s\n' "$cpu_metrics" | awk '{ for (i = 2; i <= NF; i++) { split($i, pair, "="); if (pair[1] ~ /_max_abs$/ && (pair[2] + 0) > 0.01) exit 1 } }' \
    || { die "CPU parity metric exceeds the fixed FP32 bound"; return 2; }

  marker_count="$(grep -Ec '^VOICE_GENDER_METAL_VS_CPU(_METRICS| ).*$' "$log_path" || true)"
  [[ "$marker_count" == 2 ]] || { die "Metal/CPU parity marker count is not exactly 2: $marker_count"; return 2; }
  metal_metrics="$(grep -E '^VOICE_GENDER_METAL_VS_CPU_METRICS logits_cpu_max_abs=[0-9]+\.[0-9]{9} probabilities_cpu_max_abs=[0-9]+\.[0-9]{9} bound=0\.010000000 fixture=official_canned_synthetic_tone$' "$log_path" || true)"
  [[ "$(printf '%s\n' "$metal_metrics" | wc -l | tr -d '[:space:]')" == 1 && -n "$metal_metrics" ]] || { die "Metal/CPU metrics marker is missing or malformed"; return 2; }
  pass_count="$(grep -Ec '^VOICE_GENDER_METAL_VS_CPU PASS bound=0\.010000000 fixture=official_canned_synthetic_tone label=(male|female)$' "$log_path" || true)"
  [[ "$pass_count" == 1 ]] || { die "Metal parity PASS marker is missing, duplicated, or malformed"; return 2; }
  printf '%s\n' "$metal_metrics" | awk '{ for (i = 2; i <= NF; i++) { split($i, pair, "="); if (pair[1] ~ /_max_abs$/ && (pair[2] + 0) > 0.01) exit 1 } }' \
    || { die "Metal/CPU metric exceeds the fixed FP32 bound"; return 2; }

  marker_count="$(grep -Ec '^VOICE_GENDER_METAL_VS_REFERENCE(_METRICS| ).*$' "$log_path" || true)"
  [[ "$marker_count" == 2 ]] || { die "Metal/reference parity marker count is not exactly 2: $marker_count"; return 2; }
  metal_metrics="$(grep -E '^VOICE_GENDER_METAL_VS_REFERENCE_METRICS logits_reference_max_abs=[0-9]+\.[0-9]{9} probabilities_reference_max_abs=[0-9]+\.[0-9]{9} bound=0\.010000000 fixture=official_canned_synthetic_tone$' "$log_path" || true)"
  [[ "$(printf '%s\n' "$metal_metrics" | wc -l | tr -d '[:space:]')" == 1 && -n "$metal_metrics" ]] || { die "Metal/reference metrics marker is missing or malformed"; return 2; }
  pass_count="$(grep -Ec '^VOICE_GENDER_METAL_VS_REFERENCE PASS bound=0\.010000000 fixture=official_canned_synthetic_tone label=(male|female)$' "$log_path" || true)"
  [[ "$pass_count" == 1 ]] || { die "Metal/reference parity PASS marker is missing, duplicated, or malformed"; return 2; }
  printf '%s\n' "$metal_metrics" | awk '{ for (i = 2; i <= NF; i++) { split($i, pair, "="); if (pair[1] ~ /_max_abs$/ && (pair[2] + 0) > 0.01) exit 1 } }' \
    || { die "Metal/reference metric exceeds the fixed FP32 bound"; return 2; }

  pass_count="$(grep -Ec '^VOICE_GENDER_ARGMAX_LABEL_AGREEMENT PASS (argmax=0 label=male official=male cpu=male metal=male|argmax=1 label=female official=female cpu=female metal=female)$' "$log_path" || true)"
  [[ "$pass_count" == 1 ]] || { die "argmax/label agreement marker is missing, duplicated, or malformed"; return 2; }
  printf 'CPU and Metal parity gates authenticated: bound=%s fixture=%s\n' "$FP32_PARITY_BOUND" "$FIXTURE_KIND"
}

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required temp_root
  local cpu_metrics_marker cpu_pass_marker metal_metrics_marker metal_pass_marker
  local metal_reference_metrics_marker metal_reference_pass_marker argmax_pass_marker
  local malformed_cpu_metrics_marker over_bound_cpu_metrics_marker over_bound_metal_metrics_marker
  local over_bound_metal_reference_metrics_marker nonpass_metal_pass_marker
  local cargo_result_line
  temp_root=/private/tmp
  [[ -d "$temp_root" ]] || temp_root=/tmp
  temporary="$(mktemp -d "$temp_root/vokra-voice-gender-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_empty_directory "$temporary/evidence"
  # shellcheck disable=SC2016 # literal contract token intentionally keeps quoting
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=10000000' \
    'hw.memsize' 'df -Pk' 'xcrun -f metal' \
    "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$UPSTREAM_HF_REVISION" \
    "$UPSTREAM_HF_FILE" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256" \
    "$CORRECTED_GGUF_SHA256" \
    "$UPSTREAM_LICENSE_FILE" "$UPSTREAM_LICENSE_SPDX" "$UPSTREAM_LICENSE_COPYRIGHT" "$UPSTREAM_HF_LICENSE" \
    'real_voice_gender_classifier_matches_official_reference' \
    'VOKRA_VOICE_GENDER_GGUF' 'VOKRA_VOICE_GENDER_PCM' \
    'VOKRA_VOICE_GENDER_FEATURES' 'VOKRA_VOICE_GENDER_LOGITS' \
    'VOKRA_VOICE_GENDER_EMBEDDING' \
    'VOKRA_VOICE_GENDER_PROBABILITIES' \
    'VOKRA_VOICE_GENDER_FIXTURE_KIND' 'official_canned_synthetic_tone' \
    'VOICE_GENDER_OFFICIAL_PARITY_METRICS' 'VOICE_GENDER_OFFICIAL_PARITY PASS' \
    'VOICE_GENDER_METAL_VS_CPU_METRICS' 'VOICE_GENDER_METAL_VS_CPU PASS' \
    'VOICE_GENDER_METAL_VS_REFERENCE_METRICS' 'VOICE_GENDER_METAL_VS_REFERENCE PASS' \
    'VOICE_GENDER_ARGMAX_LABEL_AGREEMENT PASS' 'VOKRA_VOICE_GENDER_ARGMAX' \
    'FP32_PARITY_BOUND' 'FIXTURE_KIND' 'verify_parity_log' \
    'require_cargo_singleton' 'finished in [0-9]+(\\.[0-9]+)?s' \
    'test result: ok. 1 passed' '--features metal' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml"'; do
    grep -Fq -- "$required" "$script_path" \
      || { log "self-test missing contract token: $required"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(curl|wget|python3?|pip|.*convert|git[[:space:]]+(clone|fetch|pull)|.*(upload|publish)|git[[:space:]]+push)([[:space:]]|$)' \
    "$script_path" | grep -v 'UV_NO_CACHE=1 uv run' >/dev/null; then
    log "self-test found acquisition/conversion/publication command"
    fail=1
  fi
  local printf_token='printf'
  if grep -F 'VOICE_GENDER_METAL_VS_CPU PASS' "$script_path" | grep -Fq "$printf_token "; then
    log "self-test found a manufactured Metal PASS marker"
    fail=1
  fi
  cpu_metrics_marker='VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.001000000 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.010000000 fixture=official_canned_synthetic_tone'
  cpu_pass_marker='VOICE_GENDER_OFFICIAL_PARITY PASS bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream'
  metal_metrics_marker='VOICE_GENDER_METAL_VS_CPU_METRICS logits_cpu_max_abs=0.005000000 probabilities_cpu_max_abs=0.006000000 bound=0.010000000 fixture=official_canned_synthetic_tone'
  metal_pass_marker='VOICE_GENDER_METAL_VS_CPU PASS bound=0.010000000 fixture=official_canned_synthetic_tone label=male'
  metal_reference_metrics_marker='VOICE_GENDER_METAL_VS_REFERENCE_METRICS logits_reference_max_abs=0.007000000 probabilities_reference_max_abs=0.008000000 bound=0.010000000 fixture=official_canned_synthetic_tone'
  metal_reference_pass_marker='VOICE_GENDER_METAL_VS_REFERENCE PASS bound=0.010000000 fixture=official_canned_synthetic_tone label=male'
  argmax_pass_marker='VOICE_GENDER_ARGMAX_LABEL_AGREEMENT PASS argmax=0 label=male official=male cpu=male metal=male'
  malformed_cpu_metrics_marker='VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.001000000 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.020000000 fixture=official_canned_synthetic_tone'
  over_bound_cpu_metrics_marker='VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.010000001 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.010000000 fixture=official_canned_synthetic_tone'
  over_bound_metal_metrics_marker='VOICE_GENDER_METAL_VS_CPU_METRICS logits_cpu_max_abs=0.010000001 probabilities_cpu_max_abs=0.006000000 bound=0.010000000 fixture=official_canned_synthetic_tone'
  over_bound_metal_reference_metrics_marker='VOICE_GENDER_METAL_VS_REFERENCE_METRICS logits_reference_max_abs=0.010000001 probabilities_reference_max_abs=0.008000000 bound=0.010000000 fixture=official_canned_synthetic_tone'
  nonpass_metal_pass_marker='VOICE_GENDER_METAL_VS_CPU FAIL bound=0.010000000 fixture=official_canned_synthetic_tone label=male'
  printf '%s\n' "$cpu_metrics_marker" "$cpu_pass_marker" "$metal_metrics_marker" "$metal_pass_marker" "$metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/valid-parity.log"
  verify_parity_log "$temporary/valid-parity.log" >/dev/null || { log "self-test rejected a valid parity log"; fail=1; }
  cp "$temporary/valid-parity.log" "$temporary/duplicate-parity.log"
  printf '%s\n' "$metal_pass_marker" >> "$temporary/duplicate-parity.log"
  if verify_parity_log "$temporary/duplicate-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a duplicate parity marker"; fail=1
  fi
  printf '%s\n' "$malformed_cpu_metrics_marker" "$cpu_pass_marker" "$metal_metrics_marker" "$metal_pass_marker" "$metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/malformed-parity.log"
  if verify_parity_log "$temporary/malformed-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a malformed parity metrics marker"; fail=1
  fi
  printf '%s\n' "$over_bound_cpu_metrics_marker" "$cpu_pass_marker" "$metal_metrics_marker" "$metal_pass_marker" "$metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/over-bound-parity.log"
  if verify_parity_log "$temporary/over-bound-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a parity metric above the fixed bound"; fail=1
  fi
  printf '%s\n' "$cpu_metrics_marker" "$cpu_pass_marker" "$over_bound_metal_metrics_marker" "$metal_pass_marker" "$metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/over-bound-metal-parity.log"
  if verify_parity_log "$temporary/over-bound-metal-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a Metal parity metric above the fixed bound"; fail=1
  fi
  printf '%s\n' "$cpu_metrics_marker" "$cpu_pass_marker" "$metal_metrics_marker" "$metal_pass_marker" "$over_bound_metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/over-bound-metal-reference-parity.log"
  if verify_parity_log "$temporary/over-bound-metal-reference-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a Metal/reference parity metric above the fixed bound"; fail=1
  fi
  cargo_result_line='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.25s'
  printf '%s\n' 'test real_voice_gender_classifier_matches_official_reference ... ok' "$cargo_result_line" \
    > "$temporary/valid-cargo.log"
  require_cargo_singleton "$temporary/valid-cargo.log" || { log "self-test rejected a valid Cargo result"; fail=1; }
  printf '%s\n%s\n%s\n' \
    'test real_voice_gender_classifier_matches_official_reference ... ' \
    'ok' "$cargo_result_line" \
    > "$temporary/valid-nocapture-cargo.log"
  require_cargo_singleton "$temporary/valid-nocapture-cargo.log" || { log "self-test rejected a valid nocapture Cargo result"; fail=1; }
  printf '%s\n%s\n%s\n' \
    'ok' 'test real_voice_gender_classifier_matches_official_reference ... ' "$cargo_result_line" \
    > "$temporary/reordered-nocapture-cargo.log"
  if require_cargo_singleton "$temporary/reordered-nocapture-cargo.log" >/dev/null 2>&1; then
    log "self-test accepted an out-of-order standalone Cargo status"; fail=1
  fi
  printf '%s\n' 'test real_voice_gender_classifier_matches_official_reference ... ok' "$cargo_result_line" "$cargo_result_line" \
    > "$temporary/duplicate-cargo.log"
  if require_cargo_singleton "$temporary/duplicate-cargo.log" >/dev/null 2>&1; then
    log "self-test accepted a duplicate Cargo summary"; fail=1
  fi
  printf '%s\n%s\n%s\n' \
    'test real_voice_gender_classifier_matches_official_reference ... ' \
    'ok' 'ok' \
    > "$temporary/duplicate-nocapture-cargo.log"
  if require_cargo_singleton "$temporary/duplicate-nocapture-cargo.log" >/dev/null 2>&1; then
    log "self-test accepted a duplicate standalone Cargo status"; fail=1
  fi
  printf '%s\n' 'test real_voice_gender_classifier_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in .25s' \
    > "$temporary/malformed-cargo.log"
  if require_cargo_singleton "$temporary/malformed-cargo.log" >/dev/null 2>&1; then
    log "self-test accepted a malformed Cargo duration"; fail=1
  fi
  printf '%s\n%s\n' \
    'test real_voice_gender_classifier_matches_official_reference ... ' "$cargo_result_line" \
    > "$temporary/missing-nocapture-status.log"
  if require_cargo_singleton "$temporary/missing-nocapture-status.log" >/dev/null 2>&1; then
    log "self-test accepted a missing standalone Cargo status"; fail=1
  fi
  printf '%s\n' "$cpu_metrics_marker" "$cpu_pass_marker" "$metal_metrics_marker" "$nonpass_metal_pass_marker" "$metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/nonpass-parity.log"
  if verify_parity_log "$temporary/nonpass-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a non-PASS parity marker"; fail=1
  fi
  printf '%s\n' "$cpu_pass_marker" "$metal_metrics_marker" "$metal_pass_marker" "$metal_reference_metrics_marker" "$metal_reference_pass_marker" "$argmax_pass_marker" \
    > "$temporary/missing-parity.log"
  if verify_parity_log "$temporary/missing-parity.log" >/dev/null 2>&1; then
    log "self-test accepted a missing parity metrics marker"; fail=1
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    log "self-test accepted an extra argument"
    fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PARITY_DUMPER" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' reference_dir='' evidence_dir='' self_test=0 gguf_seen=0 reference_seen=0 evidence_seen=0
  local gguf_sha='' reference_sha=''
  while (( $# > 0 )); do
    case "$1" in
      --gguf) (( gguf_seen == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf="$2"; gguf_seen=1; shift 2 ;;
      --gguf-sha256) [[ -z "$gguf_sha" ]] || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf_sha="$2"; shift 2 ;;
      --reference-dir) (( reference_seen == 0 )) || die 'duplicate --reference-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference_dir="$2"; reference_seen=1; shift 2 ;;
      --reference-manifest-sha256) [[ -z "$reference_sha" ]] || die 'duplicate --reference-manifest-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference_sha="$2"; shift 2 ;;
      --evidence-dir) (( evidence_seen == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; evidence_dir="$2"; evidence_seen=1; shift 2 ;;
      --self-test) (( self_test == 0 )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1"; return 2 ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$gguf_sha$reference_dir$reference_sha$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$gguf_sha" && -n "$reference_dir" && -n "$reference_sha" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --gguf-sha256, --reference-dir, --reference-manifest-sha256 and --evidence-dir are required"; }

  preflight_gate
  require_remote_apple_host
  require_tooling
  require_file "VAST-produced corrected voice-gender GGUF" "$gguf"
  valid_sha256 "$gguf_sha" gguf-sha256
  valid_sha256 "$reference_sha" reference-manifest-sha256
  [[ "$gguf_sha" == "$CORRECTED_GGUF_SHA256" ]] || die "GGUF digest is not the fixed corrected artifact identity"
  [[ "$reference_dir" != "$VOKRA_ROOT"/* ]] || die "reference packet must be outside checkout"
  [[ "$gguf" != "$VOKRA_ROOT"/* ]] || die "GGUF must be outside checkout"
  require_reference "$reference_dir" "$reference_sha"
  for protected in "$gguf" "$reference_dir" "$VOKRA_ROOT"; do
    disjoint "$evidence_dir" "$protected"
  done
  disjoint "$gguf" "$reference_dir"
  disjoint "$gguf" "$VOKRA_ROOT"
  disjoint "$reference_dir" "$VOKRA_ROOT"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  hash_reference_directory "$reference_dir" "$evidence_dir/reference-hashes.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_dir=$reference_dir"
    echo "reference_manifest_sha256=$reference_sha"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact official CPU and Metal-vs-CPU parity"
  env \
    VOKRA_VOICE_GENDER_GGUF="$gguf" \
    VOKRA_VOICE_GENDER_GGUF_SHA256="$gguf_sha" \
    VOKRA_VOICE_GENDER_REFERENCE_DIR="$reference_dir" \
    VOKRA_VOICE_GENDER_REFERENCE_MANIFEST_SHA256="$reference_sha" \
    VOKRA_VOICE_GENDER_EVIDENCE_DIR="$evidence_dir" \
    VOKRA_VOICE_GENDER_PCM="$reference_dir/pcm.f32" \
    VOKRA_VOICE_GENDER_FEATURES="$reference_dir/features.f32" \
    VOKRA_VOICE_GENDER_EMBEDDING="$reference_dir/embedding.f32" \
    VOKRA_VOICE_GENDER_LOGITS="$reference_dir/logits.f32" \
    VOKRA_VOICE_GENDER_PROBABILITIES="$reference_dir/probabilities.f32" \
    VOKRA_VOICE_GENDER_ARGMAX="$reference_dir/argmax.u32" \
    VOKRA_VOICE_GENDER_FIXTURE_KIND="$FIXTURE_KIND" \
    RUST_TEST_THREADS=1 \
    CARGO_NET_OFFLINE=true \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_voice_gender_classifier \
      real_voice_gender_classifier_matches_official_reference -- --ignored --exact --nocapture 2>&1 | tee "$evidence_dir/parity.log"
  require_cargo_singleton "$evidence_dir/parity.log"
  verify_parity_log "$evidence_dir/parity.log" | tee "$evidence_dir/parity-gate.log"

  {
    echo 'verdict=PASS'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repository=$UPSTREAM_REPOSITORY"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_manifest_sha256=$reference_sha"
    echo 'official_cpu_parity=PASS'
    echo 'metal_vs_reference=PASS'
    echo 'metal_vs_cpu=PASS'
    echo 'argmax_label_agreement=PASS'
    echo "numeric_bound=$FP32_PARITY_BOUND"
    echo "fixture_kind=$FIXTURE_KIND"
    echo "metal_compiler=$(xcrun -f metal)"
    echo 'download=NOT_PERFORMED'
    echo 'conversion=NOT_PERFORMED'
    echo 'publication=NOT_PERFORMED'
  } > "$evidence_dir/summary.txt"
  log "PASS: CPU and Metal parity gates completed; remove staged model data after evidence capture"
}

main "$@"
