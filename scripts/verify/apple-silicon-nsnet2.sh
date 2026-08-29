#!/usr/bin/env bash
# Exact NSNet2 CPU/reference/Metal parity on a disposable Apple Silicon host.
# The GGUF and both WAVs must already have been produced/staged elsewhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000
TEST_NAME="parity_nsnet2_gguf_smoke"
TEST_TARGET="parity_nsnet2"
GGUF_ENV="VOKRA_NSNET2_REAL_GGUF"
WAV_ENV="VOKRA_NSNET2_REAL_WAV"
REFERENCE_WAV_ENV="VOKRA_NSNET2_REFERENCE_WAV"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_nsnet2.rs"

log() { printf '[nsnet2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-nsnet2.sh --gguf <corrected.gguf> \
  --input <input.wav> --reference <reference.wav> --approval-evidence <owner-approval.json> \
  --evidence-dir <absent-dir>
       apple-silicon-nsnet2.sh --self-test

Runs the exact existing parity_nsnet2_gguf_smoke test with the VAST-produced
corrected NSNet2 GGUF, input WAV, and independent reference WAV. It requires
VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least 32 GB physical memory,
10 GB free disk, a clean checkout, and the Xcode Metal compiler.

This verifier performs no download, conversion, upload, publish, or model
mutation. --self-test is pure offline and performs no Cargo invocation.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, symlinked, or non-regular: $path"
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''; [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue; scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1; done
  while [[ ! -d "$path" || -L "$path" ]]; do name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"; done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence_dir() {
  local directory="$1" candidate protected other
  shift
  [[ ! -e "$directory" && ! -L "$directory" ]] || { die "evidence directory must be absent and non-symlink: $directory"; return 2; }
  candidate="$(canonical_absent_path "$directory")" || { die 'evidence directory has a symlinked ancestor'; return 2; }
  for protected in "$VOKRA_ROOT" "$@"; do
    [[ -e "$protected" && ! -L "$protected" ]] || { die "protected path is missing or symlinked"; return 2; }
    other="$(canonical_absent_path "$protected")" || { die 'protected path cannot be canonicalized'; return 2; }
    paths_overlap "$candidate" "$other" && { die 'evidence overlaps protected path'; return 2; }
  done
  return 0
}

require_cargo_result() {
  local file="$1" named tests results
  named="$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$file" || true)"
  tests="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$tests" == 1 && "$results" == 1 ]] || die 'Cargo evidence has duplicate/missing test or result lines'
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || die 'Cargo result is not the exact one-pass result'
}

require_metric_sentinel() {
  local file="$1"
  [[ "$(grep -Ec '^NSNet2 real CPU/Metal PCM max_abs=[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$' "$file" || true)" == 1 ]] || die 'NSNet2 metric sentinel is missing, malformed, or duplicated'
}

license_preflight() {
  local approval="$1" project="$VOKRA_ROOT/tools/parity/pyproject.toml" lock="$VOKRA_ROOT/tools/parity/uv.lock" project_sha lock_sha
  [[ -f "$project" && ! -L "$project" && -f "$lock" && ! -L "$lock" ]] || die 'locked parity project is missing or symlinked'
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence must be a nonempty regular non-symlink file'
  project_sha="$(shasum -a 256 "$project" | awk '{print $1}')"; lock_sha="$(shasum -a 256 "$lock" | awk '{print $1}')"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
def hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result
try:
    d=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'), object_pairs_hook=hook)
    keys={'schema','model','upstream_repo','upstream_revision','license_spdx','project_sha256','lock_sha256','no_upload','decision','signer','scope_sha256'}
    if set(d)!=keys: raise ValueError('approval schema is not exact')
    if (d['schema'],d['model'],d['upstream_repo'],d['upstream_revision'],d['license_spdx']) != ('vokra-validation-approval-v1','nsnet2','microsoft/DNS-Challenge','8b87a33b2892f147b5c7ad39ea978453730db269','cc-by-4.0'): raise ValueError('approval identity mismatch')
    if d['project_sha256']!=sys.argv[2] or d['lock_sha256']!=sys.argv[3] or d['no_upload'] is not True or d['decision']!='APPROVED': raise ValueError('approval facts mismatch')
    if not isinstance(d['signer'],str) or not d['signer'].strip() or d['signer'].strip().upper() in {'TODO','UNRESOLVED','OWNER_SIGNOFF_REQUIRED'}: raise ValueError('approval signer unresolved')
    scope={'license_spdx':d['license_spdx'],'lock_sha256':sys.argv[3],'model':d['model'],'no_upload':True,'project_sha256':sys.argv[2],'upstream_repo':d['upstream_repo'],'upstream_revision':d['upstream_revision']}
    if d['scope_sha256'] != hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise ValueError('approval scope digest mismatch')
except (OSError,TypeError,ValueError,json.JSONDecodeError) as exc: raise SystemExit('approval gate BLOCKED: '+str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$VOKRA_ROOT/scripts/publish/signoff_match.py" --check-repo nsnet2 --audit "$VOKRA_ROOT/docs/license-audit.md"
  then :; else die 'repository signoff is unresolved'; fi
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
    || die "free disk $free_disk_kib KiB is below the 10-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] || die "NSNet2 parity source is missing: $PARITY_SOURCE"
  grep -Fq 'if let Ok(ref_path) = env::var(REFERENCE_WAV_ENV)' "$PARITY_SOURCE" \
    || die "NSNet2 parity source lacks the reference-WAV leg"
  grep -Fq 'NSNet2 real CPU/Metal PCM max_abs=' "$PARITY_SOURCE" \
    || die "NSNet2 parity source lacks the CPU/Metal sentinel"
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

# shellcheck disable=SC2016
run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-nsnet2-apple.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON' 'Darwin' 'arm64' 'MIN_MEMORY_BYTES=32000000000' \
    'MIN_FREE_DISK_KIB=10000000' 'xcrun -f metal' \
    'VOKRA_NSNET2_REAL_GGUF' 'VOKRA_NSNET2_REAL_WAV' \
    'VOKRA_NSNET2_REFERENCE_WAV' 'parity_nsnet2_gguf_smoke' \
    '--features metal --test parity_nsnet2' '-- --exact --nocapture' \
    'NSNet2 real CPU/Metal PCM max_abs=' 'NSNet2_APPLE_PARITY cpu_reference=PASS' \
    'NSNet2_APPLE_PARITY metal_vs_cpu=PASS' \
    'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml"'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: contract token missing: $required"
      fail=1
    fi
  done
  if grep -En '^[[:space:]]*(curl|wget|python3?|pip|.*convert|git[[:space:]]+(clone|fetch|pull)|.*(upload|publish))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: forbidden acquisition/conversion/publication command found"
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
  require_absent_evidence_dir "$temporary/new/nested/evidence" "$temporary/value" || { log 'self-test FAIL: nested absent evidence rejected'; fail=1; }
  mkdir "$temporary/empty-evidence"
  if require_absent_evidence_dir "$temporary/empty-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: existing empty evidence accepted'; fail=1; fi
  ln -s "$temporary/missing-evidence" "$temporary/dangling-evidence"
  if require_absent_evidence_dir "$temporary/dangling-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: dangling evidence symlink accepted'; fail=1; fi
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' input='' reference='' approval='' evidence_dir='' self_test=0 gguf_sha
  local seen_gguf=0 seen_input=0 seen_reference=0 seen_approval=0 seen_evidence=0 seen_self=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty path'; seen_gguf=1
        gguf="$2"; shift 2 ;;
      --input)
        (( seen_input == 0 )) || die 'duplicate --input'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--input requires a nonempty path'; seen_input=1
        input="$2"; shift 2 ;;
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
        (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self_test=1; shift ;;
      -h|--help)
        usage; return 0 ;;
      *)
        usage; die "unknown argument $1"; return 2 ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$input$reference$approval$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$input" && -n "$reference" && -n "$approval" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --input, --reference, --approval-evidence and --evidence-dir are required"; }

  license_preflight "$approval"
  require_absent_evidence_dir "$evidence_dir" "$gguf" "$input" "$reference" "$approval"
  require_remote_apple_host
  require_tooling
  require_file "corrected NSNet2 GGUF" "$gguf"
  require_file "NSNet2 input WAV" "$input"
  require_file "independent NSNet2 reference WAV" "$reference"
  mkdir -p "$evidence_dir"
  gguf_sha="$(sha256_file "$gguf")"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$gguf_sha"
    echo "input_wav=$input"
    echo "input_wav_sha256=$(sha256_file "$input")"
    echo "reference_wav=$reference"
    echo "reference_wav_sha256=$(sha256_file "$reference")"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact NSNet2 CPU/reference/Metal parity"
  env \
    "$GGUF_ENV=$gguf" \
    "$WAV_ENV=$input" \
    "$REFERENCE_WAV_ENV=$reference" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test "$TEST_TARGET" "$TEST_NAME" \
      -- --exact --nocapture 2>&1 | tee "$evidence_dir/parity.log"

  require_cargo_result "$evidence_dir/parity.log"
  require_metric_sentinel "$evidence_dir/parity.log"
  # The existing harness does not print a success line for its optional
  # reference leg. Its exact passing test, the required reference env, and
  # the source contract above jointly prove that this leg was enabled; only
  # now may this verifier emit its explicit evidence marker.
  printf 'NSNet2_APPLE_PARITY cpu_reference=PASS test=%s\n' "$TEST_NAME" \
    | tee -a "$evidence_dir/parity.log"
  printf 'NSNet2_APPLE_PARITY metal_vs_cpu=PASS test=%s\n' "$TEST_NAME" \
    | tee -a "$evidence_dir/parity.log"

  grep -F 'NSNet2_APPLE_PARITY cpu_reference=PASS' "$evidence_dir/parity.log" >/dev/null \
    || die "CPU/reference PASS marker is absent"
  grep -F 'NSNet2_APPLE_PARITY metal_vs_cpu=PASS' "$evidence_dir/parity.log" >/dev/null \
    || die "Metal/CPU PASS marker is absent"
  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "gguf_sha256=$gguf_sha"
    echo "nsnet2_cpu_reference=PASS"
    echo "nsnet2_metal_vs_cpu=PASS"
    echo "test=$TEST_TARGET::$TEST_NAME"
    echo "network=NOT_PERFORMED"
    echo "conversion=NOT_PERFORMED"
    echo "publication=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged inputs or destroy the remote worker"
}

main "$@"
