#!/usr/bin/env bash
# Real RMVPE CPU/upstream and Metal-vs-CPU parity on a disposable Apple host.
# The GGUF and upstream fixtures must already have been produced by VAST.
# This verifier never downloads, converts, publishes, uploads, or pushes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000
UPSTREAM_REPO="yxlllc/RMVPE"
UPSTREAM_REVISION="0aabafba18289ca938a73af0b0297686abf4922d"
GGUF_ENV="VOKRA_RMVPE_REAL_GGUF"
PCM_ENV="VOKRA_RMVPE_REAL_PCM"
HIDDEN_ENV="VOKRA_RMVPE_REAL_HIDDEN"
HIDDEN_FEATURE_DIM_ENV="VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM"
ARGMAX_ENV="VOKRA_RMVPE_REAL_ARGMAX"
F0_ENV="VOKRA_RMVPE_REAL_F0"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_rmvpe.rs"

log() { printf '[rmvpe-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-rmvpe.sh --gguf <vast-rmvpe.gguf> \
         --reference-dir <vast-rmvpe-fixtures> --expected-gguf-sha256 <64-hex> \
         --expected-reference-sha256 <64-hex> --checkpoint-sha256 <64-hex> \
         --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-rmvpe.sh --self-test

Runs the exact checked-in RMVPE real-weight CPU/upstream and
Metal-vs-CPU tests with VAST-produced GGUF and raw reference fixtures.  It
requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least 32 GB physical
memory, free disk, a clean checkout, and the Xcode Metal compiler.

This verifier performs no download, conversion, upload, publication, or model
mutation.  --self-test is hermetic and performs no Cargo invocation.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, symlinked, or non-regular: $path"
}

license_preflight() {
  local approval="$1" checkpoint_sha="$2" project="$VOKRA_ROOT/tools/parity/rmvpe/pyproject.toml" lock="$VOKRA_ROOT/tools/parity/rmvpe/uv.lock" project_sha lock_sha
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence must be a nonempty regular non-symlink file'
  project_sha="$(shasum -a 256 "$project" | awk '{print $1}')"; lock_sha="$(shasum -a 256 "$lock" | awk '{print $1}')"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" "$checkpoint_sha" <<'PY'
import hashlib, json, pathlib, sys
def reject(pairs):
    d = {}
    for k, v in pairs:
        if k in d: raise ValueError("duplicate JSON key: " + k)
        d[k] = v
    return d
try:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
    keys = {"schema", "model", "upstream_repo", "upstream_revision", "license_spdx", "project_sha256", "lock_sha256", "checkpoint_sha256", "no_upload", "decision", "signer", "scope_sha256"}
    if set(d) != keys: raise ValueError("approval schema is not exact")
    if d["schema"] != "vokra-validation-approval-v1" or d["model"] != "rmvpe" or d["upstream_repo"] != "yxlllc/RMVPE" or d["upstream_revision"] != "0aabafba18289ca938a73af0b0297686abf4922d": raise ValueError("RMVPE identity mismatch")
    if d["license_spdx"] != "unknown" or d["project_sha256"] != sys.argv[2] or d["lock_sha256"] != sys.argv[3] or d["checkpoint_sha256"] != sys.argv[4] or d["no_upload"] is not True or d["decision"] != "APPROVED": raise ValueError("approval facts mismatch")
    if not isinstance(d["signer"], str) or not d["signer"].strip() or d["signer"].strip().upper() in {"TODO", "UNRESOLVED", "OWNER_SIGNOFF_REQUIRED"}: raise ValueError("approval signer unresolved")
    scope = {"checkpoint_sha256": sys.argv[4], "license_spdx": d["license_spdx"], "lock_sha256": sys.argv[3], "model": d["model"], "no_upload": True, "project_sha256": sys.argv[2], "upstream_repo": d["upstream_repo"], "upstream_revision": d["upstream_revision"]}
    if d["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): raise ValueError("approval scope digest mismatch")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit("approval gate BLOCKED: " + str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$VOKRA_ROOT/tools/parity/rmvpe_inspect.py" --dependency-gate; then :; else die 'RMVPE source/license/checkpoint gate is unresolved'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == /var ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence() {
  local target="$1" candidate other
  shift
  [[ ! -e "$target" && ! -L "$target" ]] || { die 'evidence directory must be absent and non-symlink'; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die 'evidence directory has a symlinked ancestor'; return 2; }
  for other in "$VOKRA_ROOT" "$@"; do
    [[ ! -L "$other" ]] || { die 'protected input is symlinked'; return 2; }
    local resolved; resolved="$(canonical_absent_path "$other")" || return 2
    paths_overlap "$candidate" "$resolved" && { die 'evidence directory overlaps protected input'; return 2; }
  done
}

require_reference() {
  local directory="$1" expected_meta_sha="$2" path
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference directory is missing or symlinked: $directory"
  for path in pcm.f32 hidden.f32 probabilities.f32 argmax.u32 f0.f32 meta.json; do
    require_file "RMVPE reference $path" "$directory/$path"
  done
  for field in \
    '"upstream_repository": "https://github.com/yxlllc/RMVPE"' \
    '"upstream_revision": "0aabafba18289ca938a73af0b0297686abf4922d"' \
    '"upstream_class": "src.inference.RMVPE / src.model.E2E0"' \
    '"sample_rate": 16000' '"hop_length": 160' '"feature_dim": 384' \
    '"n_class": 360'; do
    grep -Fq -- "$field" "$directory/meta.json" \
      || die "RMVPE reference metadata is missing exact field: $field"
  done
  [[ "$(sha256_file "$directory/meta.json")" == "$expected_meta_sha" ]] || die 'RMVPE reference manifest hash does not match the VAST-supplied expected hash'
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "real RMVPE Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "real RMVPE Metal parity requires Apple arm64"
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
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] || die "RMVPE parity source is missing: $PARITY_SOURCE"
  grep -Fq "fn parity_rmvpe_gguf_smoke" "$PARITY_SOURCE" \
    || die "RMVPE real GGUF smoke test is missing"
  grep -Fq "fn parity_rmvpe_full_upstream_f0" "$PARITY_SOURCE" \
    || die "RMVPE CPU/upstream parity test is missing"
  grep -Fq "fn parity_rmvpe_from_hidden_argmax_match_rate" "$PARITY_SOURCE" \
    || die "RMVPE post-CNN parity test is missing"
  grep -Fq "fn parity_rmvpe_metal_matches_cpu_when_real_gguf_is_supplied" "$PARITY_SOURCE" \
    || die "RMVPE Metal-vs-CPU parity test is missing"
  grep -Fq 'BackendKind::Metal' "$PARITY_SOURCE" \
    || die "RMVPE parity source lacks explicit Metal backend selection"
  grep -Fq 'const ARGMAX_MATCH_RATE_MIN: f32 = 0.99;' "$PARITY_SOURCE" \
    || die "RMVPE argmax bound changed or is absent"
  grep -Fq 'confidence - expected.confidence).abs() <= 0.01' "$PARITY_SOURCE" \
    || die "RMVPE Metal confidence bound changed or is absent"
  grep -Fq 'RMVPE_METAL_VS_CPU PASS' "$PARITY_SOURCE" \
    || die "RMVPE Metal test does not own the required PASS marker"
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
    echo "upstream_repo=$UPSTREAM_REPO"
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

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required marker_prefix
  marker_prefix='RMVPE_METAL_VS_CPU'
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-rmvpe-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  mkdir "$temporary/evidence"
  if require_absent_evidence "$temporary/evidence"; then
    log "self-test FAIL: existing evidence path was accepted"
    fail=1
  fi
  # shellcheck disable=SC2016 # literal strings are contract tokens
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' \
    'MIN_MEMORY_BYTES=32000000000' 'MIN_FREE_DISK_KIB=10000000' \
    'hw.memsize' 'df -Pk' 'xcrun -f metal' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" \
    'parity_rmvpe_gguf_smoke' 'parity_rmvpe_full_upstream_f0' \
    'parity_rmvpe_from_hidden_argmax_match_rate' \
    'parity_rmvpe_metal_matches_cpu_when_real_gguf_is_supplied' \
    'VOKRA_RMVPE_REAL_GGUF' 'VOKRA_RMVPE_REAL_PCM' \
    'VOKRA_RMVPE_REAL_HIDDEN' 'VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM' \
    'VOKRA_RMVPE_REAL_ARGMAX' 'VOKRA_RMVPE_REAL_F0' \
    'full RMVPE parity: .*\\([1-9][0-9]*/[1-9][0-9]*\\)' \
    'path-B: [1-9][0-9]* / [1-9][0-9]* voiced frames' \
    'RMVPE_METAL_VS_CPU PASS' 'test result: ok. [1-9] passed' \
    '--expected-gguf-sha256' '--expected-reference-sha256' '--checkpoint-sha256' \
    'checkpoint_sha256' 'RMVPE GGUF hash does not match' 'reference manifest hash' \
    'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml"'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: verifier contract lost token: $required"
      fail=1
    fi
  done
  if grep -En '^[[:space:]]*(curl|wget|python3?|pip|git[[:space:]]+(clone|fetch|pull|push))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: acquisition/conversion/publication command found"
    fail=1
  fi
  if grep -Fq "printf '${marker_prefix} PASS" "$script_path"; then
    log "self-test FAIL: Apple verifier manufactures the Metal PASS marker"
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
  for bad in '--gguf' '--gguf -bad' '--gguf a --gguf b' '--reference-dir' '--reference-dir -bad' '--reference-dir a --reference-dir b' '--checkpoint-sha256' '--checkpoint-sha256 -bad' '--checkpoint-sha256 a --checkpoint-sha256 b' '--expected-gguf-sha256' '--expected-gguf-sha256 -bad' '--expected-gguf-sha256 a --expected-gguf-sha256 b' '--expected-reference-sha256' '--expected-reference-sha256 -bad' '--expected-reference-sha256 a --expected-reference-sha256 b' '--approval-evidence' '--approval-evidence -bad' '--approval-evidence a --approval-evidence b' '--evidence-dir' '--evidence-dir -bad' '--evidence-dir a --evidence-dir b'; do
    if eval "\"$script_path\" $bad" >/dev/null 2>&1; then
      log "self-test FAIL: malformed or duplicate option accepted: $bad"
      fail=1
    fi
  done
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' reference_dir='' evidence_dir='' approval_evidence='' checkpoint_sha256='' expected_gguf_sha256='' expected_reference_sha256='' self_test=0
  local seen_gguf=0 seen_reference=0 seen_evidence=0 seen_approval=0 seen_checkpoint=0 seen_expected_gguf=0 seen_expected_reference=0 seen_self=0
  local gguf_sha
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        (( seen_gguf == 0 )) || { usage; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_gguf=1
        gguf="$2"; shift 2 ;;
      --reference-dir)
        (( seen_reference == 0 )) || { usage; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_reference=1
        reference_dir="$2"; shift 2 ;;
      --approval-evidence)
        (( seen_approval == 0 )) || { usage; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_approval=1
        approval_evidence="$2"; shift 2 ;;
      --checkpoint-sha256)
        (( seen_checkpoint == 0 )) || { usage; return 2; }; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { usage; return 2; }; seen_checkpoint=1; checkpoint_sha256="$2"; shift 2 ;;
      --expected-gguf-sha256)
        (( seen_expected_gguf == 0 )) || { usage; return 2; }; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { usage; return 2; }; seen_expected_gguf=1; expected_gguf_sha256="$2"; shift 2 ;;
      --expected-reference-sha256)
        (( seen_expected_reference == 0 )) || { usage; return 2; }; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { usage; return 2; }; seen_expected_reference=1; expected_reference_sha256="$2"; shift 2 ;;
      --evidence-dir)
        (( seen_evidence == 0 )) || { usage; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_evidence=1
        evidence_dir="$2"; shift 2 ;;
      --self-test)
        (( seen_self == 0 )) || { usage; return 2; }; seen_self=1
        self_test=1; shift ;;
      -h|--help)
        usage; return 0 ;;
      *)
        usage; die "unknown argument: $1"; return 2 ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$reference_dir$evidence_dir$approval_evidence$checkpoint_sha256$expected_gguf_sha256$expected_reference_sha256" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference_dir" && -n "$evidence_dir" && -n "$approval_evidence" && -n "$checkpoint_sha256" && -n "$expected_gguf_sha256" && -n "$expected_reference_sha256" ]] \
    || { usage; die "all input, hash, checkpoint, approval, and evidence options are required"; }

  license_preflight "$approval_evidence" "$checkpoint_sha256"
  require_file "VAST-produced RMVPE GGUF" "$gguf"
  [[ "$(sha256_file "$gguf")" == "$expected_gguf_sha256" ]] || die 'RMVPE GGUF hash does not match the VAST-supplied expected hash'
  require_reference "$reference_dir" "$expected_reference_sha256"
  require_absent_evidence "$evidence_dir" "$gguf" "$reference_dir" "$approval_evidence"
  require_remote_apple_host
  require_tooling
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  gguf_sha="$(sha256_file "$gguf")"
  hash_reference_directory "$reference_dir" "$evidence_dir/reference-hashes.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_dir=$reference_dir"
    echo "reference_meta_sha256=$(sha256_file "$reference_dir/meta.json")"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact RMVPE CPU/upstream and Metal-vs-CPU parity"
  env \
    "$GGUF_ENV=$gguf" \
    "$PCM_ENV=$reference_dir/pcm.f32" \
    "$HIDDEN_ENV=$reference_dir/hidden.f32" \
    "$HIDDEN_FEATURE_DIM_ENV=384" \
    "$ARGMAX_ENV=$reference_dir/argmax.u32" \
    "$F0_ENV=$reference_dir/f0.f32" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_rmvpe \
      -- --nocapture 2>&1 | tee "$evidence_dir/parity.log"

  for test_name in parity_rmvpe_gguf_smoke parity_rmvpe_full_upstream_f0 \
    parity_rmvpe_from_hidden_argmax_match_rate \
    parity_rmvpe_metal_matches_cpu_when_real_gguf_is_supplied; do
    grep -Fq "test $test_name ... ok" "$evidence_dir/parity.log" \
      || die "exact RMVPE test did not pass: $test_name"
  done
  grep -Eq 'full RMVPE parity: .*\([1-9][0-9]*/[1-9][0-9]*\)' "$evidence_dir/parity.log" \
    || die "CPU/upstream RMVPE parity lacks nonzero voiced proof"
  grep -Eq 'path-B: [1-9][0-9]* / [1-9][0-9]* voiced frames' "$evidence_dir/parity.log" \
    || die "post-CNN RMVPE parity lacks nonzero voiced proof"
  grep -Eq 'test result: ok\. [1-9][0-9]* passed' "$evidence_dir/parity.log" \
    || die "RMVPE parity log does not prove a nonzero passing test count"
  grep -Fq 'RMVPE_METAL_VS_CPU PASS' "$evidence_dir/parity.log" \
    || die "Metal-vs-CPU RMVPE PASS marker is absent"

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_meta_sha256=$(sha256_file "$reference_dir/meta.json")"
    echo "rmvpe_cpu_upstream=PASS"
    echo "rmvpe_post_cnn=PASS"
    echo "rmvpe_metal_vs_cpu=PASS"
    echo "metal_compiler=$(xcrun -f metal)"
    echo "download=NOT_PERFORMED"
    echo "conversion=NOT_PERFORMED"
    echo "publication=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir; remove staged model data after evidence capture"
}

main "$@"
