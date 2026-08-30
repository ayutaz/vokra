#!/usr/bin/env bash
# Real-weight ReazonSpeech NeMo v2 CPU/reference/Metal parity on a
# disposable remote Apple Silicon host.
#
# The GGUF and official NeMo reference directory are produced on VAST.  This
# verifier never downloads, converts, publishes, uploads, or mutates model
# artifacts.  It only consumes staged inputs and writes small evidence files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
GGUF_ENV="VOKRA_REAZONSPEECH_NEMO_V2_GGUF"
REFERENCE_DIR_ENV="VOKRA_REAZONSPEECH_NEMO_V2_REFERENCE_DIR"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_reazonspeech_nemo_v2.rs"
PARITY_TARGET="parity_reazonspeech_nemo_v2"
CPU_TEST="released_cpu_encoder_and_alsd_tokens_text_match_official_nemo"
METAL_TEST="released_metal_matches_cpu_encoder_and_alsd_tokens_text"

log() { printf '[reazonspeech-nemo-v2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-reazonspeech-nemo-v2.sh \
  --gguf <vast-generated-reazonspeech-nemo-v2.gguf> \
  --reference <vast-official-reference-dir> --approval-evidence <owner-approval.json> \
  --evidence-dir <absent-dir>
       apple-silicon-reazonspeech-nemo-v2.sh --self-test

Runs the exact existing ReazonSpeech NeMo v2 real-weight CPU/reference and
Metal-vs-CPU tests on a disposable Darwin/arm64 host.  It requires
VOKRA_REMOTE_APPLE_SILICON=1, a clean checkout, at least 32 GB physical
memory, free disk, and the Xcode Metal compiler.  The GGUF and all reference
files must already have been produced by the VAST worker.

This script performs no download, conversion, upload, publication, or model
mutation.  Pull only the evidence directory after the run, then remove staged
inputs or destroy the disposable Apple worker.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, symlinked, or non-regular: $path"
}

license_preflight() {
  local approval="$1" project="$VOKRA_ROOT/tools/parity/pyproject.toml" lock="$VOKRA_ROOT/tools/parity/uv.lock" project_sha lock_sha
  [[ -f "$project" && ! -L "$project" && -f "$lock" && ! -L "$lock" ]] || die "locked parity project is missing or symlinked"
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die "approval evidence must be a nonempty regular non-symlink file"
  project_sha="$(shasum -a 256 "$project" | awk '{print $1}')"
  lock_sha="$(shasum -a 256 "$lock" | awk '{print $1}')"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
def hook(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=hook)
    expected = {"schema", "model", "upstream_repo", "upstream_revision", "license_spdx", "project_sha256", "lock_sha256", "no_upload", "decision", "signer", "scope_sha256"}
    if set(data) != expected: raise ValueError("approval schema is not exact")
    if data["schema"] != "vokra-validation-approval-v1" or data["model"] != "reazonspeech-nemo-v2" or data["upstream_repo"] != "reazon-research/reazonspeech-nemo-v2" or data["upstream_revision"] != "33693408be76b7cba9fd4a7546a0a8772430211b": raise ValueError("approval identity mismatch")
    if data["license_spdx"] != "apache-2.0" or data["project_sha256"] != sys.argv[2] or data["lock_sha256"] != sys.argv[3] or data["no_upload"] is not True or data["decision"] != "APPROVED": raise ValueError("approval facts mismatch")
    if not isinstance(data["signer"], str) or not data["signer"].strip() or data["signer"].strip().upper() in {"TODO", "UNRESOLVED", "OWNER_SIGNOFF_REQUIRED"}: raise ValueError("approval signer unresolved")
    scope = {"license_spdx": data["license_spdx"], "lock_sha256": sys.argv[3], "model": data["model"], "no_upload": True, "project_sha256": sys.argv[2], "upstream_repo": data["upstream_repo"], "upstream_revision": data["upstream_revision"]}
    if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): raise ValueError("approval scope digest mismatch")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit("approval gate BLOCKED: " + str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$VOKRA_ROOT/scripts/publish/signoff_match.py" --check-repo reazonspeech-nemo-v2 --audit "$VOKRA_ROOT/docs/license-audit.md"
  then :; else die 'repository signoff is unresolved'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence_dir() {
  local target="$1" candidate protected other; shift
  [[ ! -e "$target" && ! -L "$target" ]] || { die "evidence directory must be absent and non-symlink"; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die "evidence directory has a symlinked ancestor"; return 2; }
  for protected in "$VOKRA_ROOT" "$@"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected input is symlinked"; return 2; }
    other="$(canonical_absent_path "$protected")" || { die "protected path cannot be canonicalized"; return 2; }
    paths_overlap "$candidate" "$other" && { die "evidence directory overlaps a protected input"; return 2; }
  done
  return 0
}

require_cargo_result() {
  local file="$1" test_name="$2" named tests results
  named="$(grep -Ec "^test $test_name \.\.\. ok$" "$file" || true)"
  tests="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$tests" == 1 && "$results" == 1 ]] || die 'Cargo evidence has duplicate/missing test or result lines'
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || die 'Cargo result is not the exact one-pass result'
}

require_cpu_sentinel() {
  local file="$1"
  [[ "$(grep -Ec '^ReazonSpeech-NeMo-v2 CPU encoder: .+$' "$file" || true)" == 1 ]] || die 'official CPU reference sentinel is missing, malformed, or duplicated'
}

require_reference() {
  local directory="$1" name
  [[ -d "$directory" ]] || die "reference is not a directory: $directory"
  for name in pcm.f32 encoder.f32 tokens.u32 text.txt encoder.frames.txt \
    reference.json; do
    require_file "ReazonSpeech reference $name" "$directory/$name"
  done
  require_reference_metadata "$directory/reference.json"
}

require_reference_metadata() {
  local path="$1"
  local directory="${path%/reference.json}"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$directory" <<'PY'
import hashlib
import json
import math
import pathlib
import struct
import sys

REFERENCE_FORMAT = "vokra-reazonspeech-nemo-v2-reference-v1"
REFERENCE_IMPLEMENTATION = "nemo.collections.asr.models.EncDecRNNTBPEModel.restore_from"
REFERENCE_PACKAGE = "nemo-toolkit[asr]==3.0.0"
UPSTREAM_HF = "reazon-research/reazonspeech-nemo-v2"
UPSTREAM_REVISION = "33693408be76b7cba9fd4a7546a0a8772430211b"
ARCHIVE_SHA256 = "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae"
JFK_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
EXPECTED_KEYS = {
    "format", "reference_implementation", "reference_package", "nemo_version",
    "torch_version", "environment", "upstream_hf", "upstream_revision",
    "checkpoint_sha256", "audio", "audio_sha256", "sample_rate", "sample_count",
    "pcm_sha256", "decoding_strategy", "decoding_beam_size",
    "decoding_alsd_max_target_len", "decoding_score_norm", "decoding_search_type",
    "decoding_softmax_temperature", "decoding_return_best_hypothesis",
    "decoding_preserve_alignments", "encoder_frames", "encoder_width",
    "encoder_sha256", "tokens", "tokens_sha256", "text", "text_file_sha256",
}
EXPECTED_ENVIRONMENT_KEYS = {
    "platform", "machine", "cpu_model", "logical_cpu_count",
    "torch_cpu_capability", "device", "cuda_device",
}

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def require_string(data, key):
    value = data[key]
    if type(value) is not str:
        raise ValueError(f"{key} must be a string")
    return value

def require_hash(data, key, expected=None):
    value = require_string(data, key)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{key} is not a lowercase SHA-256 digest")
    if expected is not None and value != expected:
        raise ValueError(f"{key} identity mismatch")
    return value

def require_int(data, key, positive=False):
    value = data[key]
    if type(value) is not int or (positive and value <= 0):
        raise ValueError(f"{key} must be a positive integer")
    return value

def require_float(data, key, expected):
    value = data[key]
    if type(value) is not float or not math.isfinite(value) or value != expected:
        raise ValueError(f"{key} must be the exact finite float {expected}")
    return value

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fail(message):
    raise ValueError(message)

try:
    report_path = pathlib.Path(sys.argv[1])
    directory = pathlib.Path(sys.argv[2])
    report = json.loads(
        report_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if type(report) is not dict:
        fail("reference.json top level must be an object")
    if set(report) != EXPECTED_KEYS:
        fail("reference.json schema is not exact")
    exact_strings = {
        "format": REFERENCE_FORMAT,
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "reference_package": REFERENCE_PACKAGE,
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "audio": "tests/fixtures/audio/jfk-30s.wav",
        "decoding_strategy": "alsd",
        "decoding_search_type": "default",
    }
    for key, expected in exact_strings.items():
        if require_string(report, key) != expected:
            fail(f"{key} identity mismatch")
    for key in ("nemo_version", "torch_version"):
        require_string(report, key)
    environment = report["environment"]
    if type(environment) is not dict or set(environment) != EXPECTED_ENVIRONMENT_KEYS:
        fail("reference environment schema is not exact")
    for key in ("platform", "machine", "cpu_model", "torch_cpu_capability", "device"):
        if type(environment[key]) is not str:
            fail(f"environment.{key} must be a string")
    if type(environment["logical_cpu_count"]) is not int or environment["logical_cpu_count"] <= 0:
        fail("environment.logical_cpu_count must be a positive integer")
    if environment["cuda_device"] is not None and type(environment["cuda_device"]) is not str:
        fail("environment.cuda_device must be a string or null")
    require_hash(report, "checkpoint_sha256", ARCHIVE_SHA256)
    require_hash(report, "audio_sha256", JFK_SHA256)
    if require_int(report, "sample_rate") != 16000:
        fail("sample_rate identity mismatch")
    sample_count = require_int(report, "sample_count", positive=True)
    require_hash(report, "pcm_sha256")
    if require_string(report, "decoding_strategy") != "alsd":
        fail("decoding_strategy identity mismatch")
    if require_int(report, "decoding_beam_size") != 4:
        fail("decoding_beam_size identity mismatch")
    require_float(report, "decoding_alsd_max_target_len", 1.0)
    if type(report["decoding_score_norm"]) is not bool or report["decoding_score_norm"] is not True:
        fail("decoding_score_norm identity mismatch")
    require_string(report, "decoding_search_type")
    require_float(report, "decoding_softmax_temperature", 1.0)
    if type(report["decoding_return_best_hypothesis"]) is not bool or report["decoding_return_best_hypothesis"] is not True:
        fail("decoding_return_best_hypothesis identity mismatch")
    if type(report["decoding_preserve_alignments"]) is not bool or report["decoding_preserve_alignments"] is not False:
        fail("decoding_preserve_alignments identity mismatch")
    encoder_frames = require_int(report, "encoder_frames", positive=True)
    if require_int(report, "encoder_width", positive=True) != 1024:
        fail("encoder_width identity mismatch")
    require_hash(report, "encoder_sha256")
    tokens = report["tokens"]
    if type(tokens) is not list or not tokens or any(type(token) is not int or not 0 <= token < 3000 for token in tokens):
        fail("tokens must be a nonempty list of nonblank token integers")
    require_hash(report, "tokens_sha256")
    text = report["text"]
    if type(text) is not str:
        fail("text must be a string")
    require_hash(report, "text_file_sha256")

    pcm = directory / "pcm.f32"
    encoder = directory / "encoder.f32"
    token_file = directory / "tokens.u32"
    text_file = directory / "text.txt"
    frames_file = directory / "encoder.frames.txt"
    if len(pcm.read_bytes()) != sample_count * 4 or digest(pcm) != report["pcm_sha256"]:
        fail("pcm.f32 shape or hash does not match reference.json")
    if len(encoder.read_bytes()) != encoder_frames * report["encoder_width"] * 4 or digest(encoder) != report["encoder_sha256"]:
        fail("encoder.f32 shape or hash does not match reference.json")
    token_bytes = token_file.read_bytes()
    if len(token_bytes) != len(tokens) * 4 or digest(token_file) != report["tokens_sha256"]:
        fail("tokens.u32 shape or hash does not match reference.json")
    if list(struct.unpack(f"<{len(tokens)}I", token_bytes)) != tokens:
        fail("tokens.u32 values do not match reference.json")
    text_bytes = text_file.read_bytes()
    if text_bytes != (text + "\n").encode("utf-8") or digest(text_file) != report["text_file_sha256"]:
        fail("text.txt content or hash does not match reference.json")
    if frames_file.read_text(encoding="utf-8").strip() != str(encoder_frames):
        fail("encoder.frames.txt does not match reference.json")
except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
    raise SystemExit("reference gate BLOCKED: " + str(exc))
PY
  then :; else die 'reference.json is invalid, incomplete, or does not bind staged artifacts'; fi
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] \
    || die "real Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] \
    || die "real Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the exact 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the exact 20-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep uv sysctl sw_vers \
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] \
    || die "ReazonSpeech parity source is missing: $PARITY_SOURCE"
  # These are the currently checked-in test contracts.  In particular, do
  # not replace the real Metal test with a device-less build or a CPU rerun.
  grep -Fq "fn $CPU_TEST" "$PARITY_SOURCE" \
    || die "ReazonSpeech CPU parity test is missing: $CPU_TEST"
  grep -Fq "fn $METAL_TEST" "$PARITY_SOURCE" \
    || die "ReazonSpeech Metal parity test is missing: $METAL_TEST"
  grep -Fq 'BackendKind::Metal' "$PARITY_SOURCE" \
    || die "ReazonSpeech parity source lacks an explicit Metal backend"
  grep -Fq 'Metal encoder max_abs' "$PARITY_SOURCE" \
    || die "ReazonSpeech parity source lacks the Metal-vs-CPU metric assertion"
  grep -Fq 'Metal ALSD RNN-T token sequence must match CPU exactly' "$PARITY_SOURCE" \
    || die "ReazonSpeech parity source lacks the exact token assertion"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 \
    || die "Xcode Metal compiler is unavailable"
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

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-reazonspeech-apple.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  require_absent_evidence_dir "$temporary/evidence" "$temporary/value"
  mkdir "$temporary/empty-evidence"
  if require_absent_evidence_dir "$temporary/empty-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: existing empty evidence accepted'; fail=1; fi
  ln -s "$temporary/missing-evidence" "$temporary/dangling-evidence"
  if require_absent_evidence_dir "$temporary/dangling-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: dangling evidence symlink accepted'; fail=1; fi
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=20000000' \
    'reazon-research/reazonspeech-nemo-v2' \
    '33693408be76b7cba9fd4a7546a0a8772430211b' \
    'vokra-reazonspeech-nemo-v2-reference-v1' \
    'nemo.collections.asr.models.EncDecRNNTBPEModel.restore_from' \
    'nemo-toolkit[asr]==3.0.0' \
    'uv run --no-cache --no-project --offline --python 3.12' \
    'object_pairs_hook=reject_duplicates' 'reference.json schema is not exact' \
    'pcm_sha256' 'text_file_sha256' \
    'xcrun -f metal' "$GGUF_ENV" "$REFERENCE_DIR_ENV" \
    "$PARITY_TARGET" "$CPU_TEST" "$METAL_TEST" \
    '--features metal' '-- --exact --nocapture' \
    'test released_cpu_encoder_and_alsd_tokens_text_match_official_nemo ... ok' \
    'test released_metal_matches_cpu_encoder_and_alsd_tokens_text ... ok' \
    'test result: ok. 1 passed' 'ReazonSpeech-NeMo-v2 CPU encoder:' \
    'REAZONSPEECH_NEMO_V2_CPU_VS_OFFICIAL PASS' \
    'REAZONSPEECH_NEMO_V2_METAL_VS_CPU PASS' \
    'network=NOT_PERFORMED' 'conversion=NOT_PERFORMED'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: contract token missing: $required"
      fail=1
    fi
  done
  mkdir "$temporary/duplicate" "$temporary/typed"
  printf '%s\n' '{"format":"x","format":"y"}' > "$temporary/duplicate/reference.json"
  printf '%s\n' '{"format":7,"extra":true}' > "$temporary/typed/reference.json"
  if require_reference_metadata "$temporary/duplicate/reference.json" >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate JSON key accepted'; fail=1
  fi
  if require_reference_metadata "$temporary/typed/reference.json" >/dev/null 2>&1; then
    log 'self-test FAIL: wrong typed/extra JSON field accepted'; fail=1
  fi
  if grep -En -- '^[[:space:]]*(curl|wget|python3?|pip|git[[:space:]]+(clone|fetch|pull))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: download, direct Python, or publication command found"
    fail=1
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    log "self-test FAIL: extra --self-test argument accepted"
    fail=1
  fi
  if "$script_path" --gguf >/dev/null 2>&1; then
    log "self-test FAIL: missing --gguf value accepted"
    fail=1
  fi
  if "$script_path" --unknown-flag >/dev/null 2>&1; then
    log "self-test FAIL: unknown argument accepted"
    fail=1
  fi
  if "$script_path" --gguf -bad >/dev/null 2>&1 || "$script_path" --gguf a --gguf b >/dev/null 2>&1 || "$script_path" --approval-evidence >/dev/null 2>&1 || "$script_path" --self-test --approval-evidence x >/dev/null 2>&1; then
    log "self-test FAIL: malformed or duplicate options accepted"
    fail=1
  fi
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' reference='' evidence_dir='' approval='' self_test=0
  local seen_gguf=0 seen_reference=0 seen_evidence=0 seen_approval=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty path'; seen_gguf=1
        gguf="$2"; shift 2 ;;
      --reference)
        (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty path'; seen_reference=1
        reference="$2"; shift 2 ;;
      --evidence-dir)
        (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty path'; seen_evidence=1
        evidence_dir="$2"; shift 2 ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; seen_approval=1
        approval="$2"; shift 2 ;;
      --self-test)
        (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
      -h|--help)
        usage; return 0 ;;
      *)
        usage; die "unknown argument $1" ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$reference$evidence_dir$approval" && $# -eq 0 ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference" && -n "$evidence_dir" && -n "$approval" ]] \
    || { usage; die "--gguf, --reference, --approval-evidence and --evidence-dir are required"; }

  license_preflight "$approval"
  require_absent_evidence_dir "$evidence_dir" "$gguf" "$reference" "$approval"
  require_remote_apple_host
  require_tooling
  require_file "VAST-generated ReazonSpeech NeMo v2 GGUF" "$gguf"
  require_reference "$reference"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    hash_reference_directory "$reference" "$evidence_dir/reference-hashes.txt"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact real-weight CPU vs official NeMo parity"
  env "$GGUF_ENV=$gguf" "$REFERENCE_DIR_ENV=$reference" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test "$PARITY_TARGET" "$CPU_TEST" \
      -- --exact --nocapture --test-threads=1 \
      2>&1 | tee "$evidence_dir/cpu-parity.log"
  require_cargo_result "$evidence_dir/cpu-parity.log" "$CPU_TEST"
  require_cpu_sentinel "$evidence_dir/cpu-parity.log"
  printf 'REAZONSPEECH_NEMO_V2_CPU_VS_OFFICIAL PASS test=%s\n' "$CPU_TEST" \
    | tee -a "$evidence_dir/cpu-parity.log" >/dev/null

  log "running exact real-weight Metal vs CPU parity"
  env "$GGUF_ENV=$gguf" "$REFERENCE_DIR_ENV=$reference" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test "$PARITY_TARGET" "$METAL_TEST" \
      -- --exact --nocapture --test-threads=1 \
      2>&1 | tee "$evidence_dir/metal-parity.log"
  require_cargo_result "$evidence_dir/metal-parity.log" "$METAL_TEST"
  printf 'REAZONSPEECH_NEMO_V2_METAL_VS_CPU PASS test=%s\n' "$METAL_TEST" \
    | tee -a "$evidence_dir/metal-parity.log" >/dev/null
  grep -F 'REAZONSPEECH_NEMO_V2_METAL_VS_CPU PASS' \
    "$evidence_dir/metal-parity.log" >/dev/null \
    || die "Metal-vs-CPU PASS marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "cpu_vs_official=PASS"
    echo "metal_vs_cpu=PASS"
    echo "cpu_test=$CPU_TEST"
    echo "metal_test=$METAL_TEST"
    echo "network=NOT_PERFORMED"
    echo "conversion=NOT_PERFORMED"
    echo "upload=NOT_PERFORMED"
    echo "publication=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged inputs or destroy the remote worker"
}

main "$@"
