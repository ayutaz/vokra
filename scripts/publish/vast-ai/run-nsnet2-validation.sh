#!/usr/bin/env bash
# VAST-only validation worker for the pinned Microsoft NSNet2 baseline.
# It downloads no private material and never uploads, publishes, pushes, or
# destroys an instance. Pull the small evidence directory, then destroy the
# VAST instance externally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
PROJECT_FILE="$PARITY_PROJECT/pyproject.toml"
LOCK_FILE="$PARITY_PROJECT/uv.lock"
INPUT_FILE="nsnet2-20ms-baseline.onnx"
UPSTREAM_REPO="microsoft/DNS-Challenge"
UPSTREAM_REVISION="8b87a33b2892f147b5c7ad39ea978453730db269"
UPSTREAM_SUBDIR="NSNet2-baseline"
UPSTREAM_ARTIFACT_PATH="NSNet2-baseline/nsnet2-20ms-baseline.onnx"
ONNX_BYTES=10752263
ONNX_SHA256="88429b6253600be840ab816f46f466811d20078142fb12bff8cafe2b27bd4ca9"
MODEL_KIND="nsnet2"
LICENSE_SPDX="cc-by-4.0"
PARITY_TEST="parity_nsnet2_gguf_smoke"
GGUF_ENV="VOKRA_NSNET2_REAL_GGUF"
WAV_ENV="VOKRA_NSNET2_REAL_WAV"
REFERENCE_WAV_ENV="VOKRA_NSNET2_REFERENCE_WAV"
REFERENCE_INPUT="$VOKRA_ROOT/tests/parity/silero_vad/test_16k.wav"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

license_preflight() {
  local approval="$1" expected_head="$2" approval_sha="$3" project_sha lock_sha
  [[ "$approval_sha" =~ ^[0-9a-f]{64}$ ]] || die 'approval SHA-256 must be lowercase 64-hex'
  [[ -f "$PROJECT_FILE" && ! -L "$PROJECT_FILE" && -f "$LOCK_FILE" && ! -L "$LOCK_FILE" ]] || die 'locked parity project is missing or symlinked'
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die '--approval-evidence must be a nonempty regular non-symlink file'
  project_sha="$(sha256_file "$PROJECT_FILE")"; lock_sha="$(sha256_file "$LOCK_FILE")"
  [[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 changed before preflight'
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" "$expected_head" "$approval_sha" <<'PY'
import hashlib, json, pathlib, sys
def hook(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out
try:
    raw = pathlib.Path(sys.argv[1]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sys.argv[5]: raise ValueError('approval SHA-256 mismatch')
    d = json.loads(raw.decode('utf-8'), object_pairs_hook=hook)
    keys = {"schema", "model", "upstream_repo", "upstream_revision", "license_spdx", "project_sha256", "lock_sha256", "expected_head", "no_upload", "decision", "signer", "scope_sha256"}
    if set(d) != keys: raise ValueError('approval schema is not exact')
    if (d['schema'], d['model'], d['upstream_repo'], d['upstream_revision'], d['license_spdx']) != ('vokra-validation-approval-v1', 'nsnet2', 'microsoft/DNS-Challenge', '8b87a33b2892f147b5c7ad39ea978453730db269', 'cc-by-4.0'): raise ValueError('approval identity mismatch')
    if d['project_sha256'] != sys.argv[2] or d['lock_sha256'] != sys.argv[3] or d['expected_head'] != sys.argv[4] or d['no_upload'] is not True or d['decision'] != 'APPROVED': raise ValueError('approval facts mismatch')
    if not isinstance(d['signer'], str) or not d['signer'].strip() or d['signer'].strip().upper() in {'TBD','TODO','UNKNOWN','PENDING','UNRESOLVED','OWNER_SIGNOFF_REQUIRED'}: raise ValueError('approval signer unresolved')
    scope = {'expected_head': sys.argv[4], 'license_spdx': d['license_spdx'], 'lock_sha256': sys.argv[3], 'model': d['model'], 'no_upload': True, 'project_sha256': sys.argv[2], 'upstream_repo': d['upstream_repo'], 'upstream_revision': d['upstream_revision']}
    if d['scope_sha256'] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(',', ':')).encode()).hexdigest(): raise ValueError('approval scope digest mismatch')
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit('approval gate BLOCKED: ' + str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python scripts/publish/signoff_match.py --check-repo nsnet2 --audit docs/license-audit.md
  then :; else die 'repository signoff is unresolved'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''; [[ -n "$component" ]] || continue; [[ "$component" != . && "$component" != .. ]] || return 1; scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1; done
  while [[ ! -d "$path" || -L "$path" ]]; do name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"; done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_work_dir() {
  local target="$1" approval="$2" candidate protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die '--work-dir must be absent and non-symlink'; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die '--work-dir has a symlinked ancestor'; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$PROJECT_FILE" "$LOCK_FILE" "$approval" "$REFERENCE_INPUT"; do
    [[ -e "$protected" || -L "$protected" ]] || continue; [[ ! -L "$protected" ]] || { die 'protected input is symlinked'; return 2; }; other="$(canonical_absent_path "$protected")" || { die 'protected path cannot be canonicalized'; return 2; }; paths_overlap "$candidate" "$other" && { die '--work-dir overlaps protected input'; return 2; }
  done
  return 0
}

log() { printf '[nsnet2-vast] %s\n' "$*" >&2; }
step() { printf '\n[nsnet2-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-nsnet2-validation.sh --expected-head <exact-40-hex-git-commit> \
       --approval-evidence <owner-approval.json> --approval-sha256 <sha256> \
       [--work-dir <absent-dir>]
       run-nsnet2-validation.sh --self-test

VAST-only, non-publishing NSNet2 validation worker. It downloads the exact
Microsoft DNS-Challenge ONNX release at the immutable source revision, checks
its byte size and SHA-256, bridges it through the existing offline
ONNX-to-safetensors preparation tool, converts the strict `nsnet2` model with
the audited CC-BY-4.0 content license, generates an independent ONNX
ReferenceEvaluator waveform, runs the real-weight CPU parity harness and CLI
smoke, then runs the workspace gates.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, at least
64 GiB RAM and 150 GB free disk. `--self-test` is hermetic: it performs no
network, model download, Python, Cargo, or credential access.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is available"
  fi
}

verify_file() {
  local path="$1" expected_bytes="$2" expected_hash="$3" actual_bytes actual_hash
  [[ -f "$path" && ! -L "$path" ]] || { die "missing, symlinked, or non-regular pinned input: $path"; return 2; }
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || { die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"; return 2; }
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || { die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"; return 2; }
  log "identity OK: $path bytes=$actual_bytes sha256=$actual_hash"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "NSNet2 checkpoint work is VAST/Linux-only; refusing $(uname -s)"
  [[ "$(uname -m)" == "x86_64" ]] \
    || die "the locked ONNX/reference environment targets Linux x86_64, got $(uname -m)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 64-GiB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 150-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git curl awk grep find tee wc tr rustfmt cargo-deny cargo-audit; do
    command -v "$tool" >/dev/null 2>&1 || die "required VAST tool missing: $tool"
  done
  cargo clippy --version >/dev/null 2>&1 \
    || die "the clippy component is missing on the VAST host"
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "VOKRA_ROOT is not the repository checkout: $VOKRA_ROOT"
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] \
    || die "tools/parity locked Python project is missing"
  for path in \
    "$VOKRA_ROOT/tools/parity/nsnet2_prepare_checkpoint.py" \
    "$VOKRA_ROOT/tools/parity/nsnet2_dump_reference.py" \
    "$REFERENCE_INPUT"; do
    [[ -f "$path" ]] || die "required NSNet2 validation input is missing: $path"
  done
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "VAST checkout must be clean so evidence names an exact commit"
}

download_upstream_onnx() {
  local output="$1"
  local url="https://raw.githubusercontent.com/$UPSTREAM_REPO/$UPSTREAM_REVISION/$UPSTREAM_ARTIFACT_PATH"
  mkdir -p "$(dirname "$output")"
  curl --fail --location --retry 5 --retry-delay 2 --output "$output" "$url"
  verify_file "$output" "$ONNX_BYTES" "$ONNX_SHA256"
}

require_cargo_result() {
  local file="$1" test_name="$2" named tests results
  named="$(grep -Ec "^test $test_name \.\.\. ok$" "$file" || true)"
  tests="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$tests" == 1 && "$results" == 1 ]] || { die 'Cargo evidence has duplicate/missing test or result lines'; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || { die 'Cargo result is not the exact one-pass result'; return 2; }
}

require_cpu_reference_evidence() {
  local file="$1"
  [[ "$(grep -Ec '^NSNet2 real CPU/reference PCM max_abs=[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$' "$file" || true)" == 1 ]] \
    || { die 'CPU/reference metric is missing, malformed, or duplicated'; return 2; }
  [[ "$(grep -Ec '^NSNet2_PARITY cpu_reference=PASS$' "$file" || true)" == 1 ]] \
    || { die 'CPU/reference PASS marker is missing, malformed, or duplicated'; return 2; }
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "upstream_path=$UPSTREAM_SUBDIR/$INPUT_FILE"
    echo "upstream_sha256=$ONNX_SHA256"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
  } | tee "$output"
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp fail=0 cases=0 required
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_ARTIFACT_PATH" \
    "$ONNX_BYTES" "$ONNX_SHA256" "$MODEL_KIND" "$LICENSE_SPDX" \
    "PINNED_ONNX_FILENAME" "PINNED_ONNX_BYTES" "PINNED_ONNX_SHA256" \
    "$PARITY_TEST" "$GGUF_ENV" "$WAV_ENV" "$REFERENCE_WAV_ENV" \
    "tools/parity/nsnet2_prepare_checkpoint.py" \
    "tools/parity/nsnet2_dump_reference.py" \
    "uv run --project \"\$PARITY_PROJECT\" --frozen --python 3.12 python" \
    "target/release/vokra-cli convert" "  --model \"\$MODEL_KIND\"" \
    "  --license \"\$LICENSE_SPDX\"" "--expected-head" "expected_head" \
    "actual_head" "expected_head" "CARGO_NET_OFFLINE=true" "--approval-sha256" \
    'vokra-nsnet2-transfer-v1' 'manifest.sha256' 'NSNet2_PARITY cpu_reference=PASS' \
    '--packet-manifest' '--packet-manifest-sha256' 'NO_UPLOAD'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  for required in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' 'cargo fmt --all -- --check' \
    'cargo test --locked --offline --workspace' \
    'cargo clippy --locked --offline --workspace --all-targets -- -D warnings' \
    'NSNet2 real CPU/reference PCM max_abs=' 'NSNet2_PARITY cpu_reference=PASS' \
    '--ignored --exact --nocapture --test-threads=1' \
    'apple-transfer-args.txt' 'CPU_PASS_METAL_NOT_RUN' 'metal_reference=NOT_RUN' \
    '64-GiB guard' '150-GB run guard'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: fail-closed guard lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: publication command found"
    fail=1
  fi

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$tmp/nonempty" >/dev/null 2>&1; then
    log "self-test FAIL: extra self-test argument accepted"
    fail=1
  fi
  if "$script_path" --work-dir >/dev/null 2>&1; then
    log "self-test FAIL: missing --work-dir value accepted"
    fail=1
  fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    log "self-test FAIL: unknown argument accepted"
    fail=1
  fi
  if "$script_path" --work-dir -bad >/dev/null 2>&1 || "$script_path" --work-dir a --work-dir b >/dev/null 2>&1 || "$script_path" --approval-evidence >/dev/null 2>&1 || "$script_path" --approval-sha256 >/dev/null 2>&1 || "$script_path" --approval-sha256 bad >/dev/null 2>&1 || "$script_path" --approval-sha256 "$(printf '0%.0s' {1..64})" --approval-sha256 "$(printf '1%.0s' {1..64})" >/dev/null 2>&1 || "$script_path" --expected-head >/dev/null 2>&1 || "$script_path" --expected-head bad >/dev/null 2>&1 || "$script_path" --expected-head 0000000000000000000000000000000000000000 --expected-head 1111111111111111111111111111111111111111 >/dev/null 2>&1 || "$script_path" --self-test --approval-evidence x >/dev/null 2>&1; then
    log "self-test FAIL: malformed or duplicate options accepted"
    fail=1
  fi
  printf '{}\n' > "$tmp/approval.json"
  require_absent_work_dir "$tmp/new/nested/work" "$tmp/approval.json" || { log 'self-test FAIL: nested absent work path rejected'; fail=1; }
  if require_absent_work_dir "$tmp/new/../dotdot-work" "$tmp/approval.json" >/dev/null 2>&1; then log 'self-test FAIL: dot-dot work path accepted'; fail=1; fi
  mkdir "$tmp/empty-work"
  if require_absent_work_dir "$tmp/empty-work" "$tmp/approval.json" >/dev/null 2>&1; then log 'self-test FAIL: existing empty work accepted'; fail=1; fi
  ln -s "$tmp/missing" "$tmp/dangling-work"
  if require_absent_work_dir "$tmp/dangling-work" "$tmp/approval.json" >/dev/null 2>&1; then log 'self-test FAIL: dangling work symlink accepted'; fail=1; fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-nsnet2-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" approval_sha="" expected_head="" run_stamp work_dir input_dir evidence_dir
  local seen_self=0 seen_work=0 seen_approval=0 seen_approval_sha=0 seen_head=0 actual_head
  local onnx_path prepared_path gguf_path reference_wav output_wav apple_evidence_dir
  local run_log env_log parity_log cli_log workspace_log clippy_log summary_file transfer_args_file packet_dir packet_manifest packet_sha

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        (( seen_work == 0 )) || { die 'duplicate --work-dir'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a nonempty directory"; return 2; }
        seen_work=1
        requested_work_dir="$2"
        shift 2
        ;;
      --expected-head)
        (( seen_head == 0 )) || { die 'duplicate --expected-head'; return 2; }
        [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head requires a lowercase 40-hex commit'; return 2; }
        seen_head=1
        expected_head="$2"
        shift 2
        ;;
      --approval-sha256)
        (( seen_approval_sha == 0 )) || { die 'duplicate --approval-sha256'; return 2; }
        [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { die '--approval-sha256 requires a lowercase 64-hex digest'; return 2; }
        seen_approval_sha=1
        approval_sha="$2"
        shift 2
        ;;
      --self-test)
        (( seen_self == 0 )) || { die 'duplicate --self-test'; return 2; }
        seen_self=1
        self_test=1
        shift
        ;;
      --approval-evidence)
        (( seen_approval == 0 )) || { die 'duplicate --approval-evidence'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--approval-evidence requires a nonempty file'; return 2; }
        seen_approval=1
        approval_evidence="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        die "unknown argument: $1"
        usage
        return 2
        ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    [[ -z "$requested_work_dir$approval_evidence$approval_sha$expected_head" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi

  [[ $seen_approval -eq 1 ]] || { die '--approval-evidence is required'; return 2; }
  [[ $seen_approval_sha -eq 1 ]] || { die '--approval-sha256 is required'; return 2; }
  [[ $seen_head -eq 1 ]] || { die '--expected-head is required'; return 2; }
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/nsnet2-validation/$run_stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  require_tooling
  cd "$VOKRA_ROOT"
  actual_head="$(git rev-parse HEAD)" || { die 'could not read checkout HEAD'; return 2; }
  [[ "$actual_head" == "$expected_head" ]] || { die "checkout HEAD $actual_head does not match --expected-head $expected_head"; return 2; }
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || { die 'checkout must be clean before approval and model processing'; return 2; }
  license_preflight "$approval_evidence" "$expected_head" "$approval_sha"

  input_dir="$work_dir/input"
  evidence_dir="$work_dir/evidence"
  onnx_path="$input_dir/$INPUT_FILE"
  prepared_path="$work_dir/nsnet2.safetensors"
  gguf_path="$work_dir/nsnet2.gguf"
  reference_wav="$evidence_dir/reference.wav"
  output_wav="$work_dir/nsnet2-cli.wav"
  run_log="$evidence_dir/run.log"
  env_log="$evidence_dir/environment.txt"
  parity_log="$evidence_dir/parity.log"
  cli_log="$evidence_dir/cli.log"
  workspace_log="$evidence_dir/workspace-test.log"
  clippy_log="$evidence_dir/workspace-clippy.log"
  summary_file="$evidence_dir/summary.txt"
  transfer_args_file="$evidence_dir/apple-transfer-args.txt"
  apple_evidence_dir="$work_dir/apple-evidence"
  # Claim the previously absent work directory with mkdir (not mkdir -p): a
  # concurrent creator must make this run fail rather than sharing its tree.
  mkdir -p "$(dirname "$work_dir")"
  mkdir "$work_dir"
  mkdir "$input_dir" "$evidence_dir"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Download and authenticate official Microsoft NSNet2 ONNX"
  download_upstream_onnx "$onnx_path"

  step "Bridge ONNX to safetensors with the existing offline preparation tool"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/nsnet2_prepare_checkpoint.py" \
    --onnx "$onnx_path" --output "$prepared_path"
  [[ -s "$prepared_path" ]] || die "NSNet2 preparation emitted no safetensors: $prepared_path"

  step "Convert strict NSNet2 GGUF"
  export CARGO_NET_OFFLINE=true
  CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo build --locked --offline --release -p vokra-cli
  target/release/vokra-cli convert \
    --model "$MODEL_KIND" --input "$prepared_path" --output "$gguf_path" \
    --license "$LICENSE_SPDX"
  [[ -s "$gguf_path" ]] || die "converter emitted no GGUF: $gguf_path"

  step "Generate independent official ONNX reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/nsnet2_dump_reference.py" \
    --onnx "$onnx_path" --input-wav "$REFERENCE_INPUT" \
    --output-wav "$reference_wav" --dump-npz "$evidence_dir/reference.npz"

  export "$GGUF_ENV=$gguf_path"
  export "$WAV_ENV=$REFERENCE_INPUT"
  export "$REFERENCE_WAV_ENV=$reference_wav"
  step "Run real-weight CPU parity harness"
  CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --locked --offline -p vokra-models \
    --test parity_nsnet2 "$PARITY_TEST" \
    -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$parity_log"
  require_cargo_result "$parity_log" "$PARITY_TEST"
  require_cpu_reference_evidence "$parity_log"

  step "Run CLI CPU smoke"
  target/release/vokra-cli run --model "$gguf_path" --input "$REFERENCE_INPUT" \
    --backend cpu --output "$output_wav" 2>&1 | tee "$cli_log"
  [[ -s "$output_wav" ]] || die "NSNet2 CLI emitted no output WAV: $output_wav"

  step "Run repository gates"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
  cargo fmt --all -- --check
  CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --locked --offline --workspace -- --test-threads=1 2>&1 | tee "$workspace_log"
  CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo clippy --locked --offline --workspace --all-targets -- -D warnings 2>&1 | tee "$clippy_log"
  CARGO_NET_OFFLINE=true cargo deny --locked --offline check licenses advisories bans
  CARGO_NET_OFFLINE=true cargo audit --no-fetch

  step "Build portable authenticated Apple transfer packet"
  packet_dir="$evidence_dir/apple-packet"
  packet_manifest="$packet_dir/manifest.json"
  mkdir "$packet_dir"
  cp -p "$gguf_path" "$packet_dir/nsnet2.gguf"
  cp -p "$REFERENCE_INPUT" "$packet_dir/input.wav"
  cp -p "$reference_wav" "$packet_dir/reference.wav"
  cp -p "$approval_evidence" "$packet_dir/approval.json"
  cp -p "$parity_log" "$packet_dir/parity.log"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$packet_dir" "$expected_head" "$approval_sha" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_head, approval_sha = sys.argv[2:4]
names = {"gguf": "nsnet2.gguf", "input_wav": "input.wav", "reference_wav": "reference.wav", "approval": "approval.json", "cpu_parity": "parity.log"}
def digest(name):
    path = root / name
    if not path.is_file() or path.is_symlink(): raise SystemExit(f"invalid packet artifact: {name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
rows = {key: {"filename": name, "sha256": digest(name)} for key, name in names.items()}
if rows["approval"]["sha256"] != approval_sha: raise SystemExit("approval packet hash mismatch")
if (root / "parity.log").read_text(encoding="utf-8").splitlines().count("NSNet2_PARITY cpu_reference=PASS") != 1: raise SystemExit("CPU parity marker missing or duplicated")
rows["cpu_parity"]["marker"] = "NSNet2_PARITY cpu_reference=PASS"
manifest = {"schema": "vokra-nsnet2-transfer-v1", "expected_head": expected_head, "git_commit": expected_head, "license_spdx": "cc-by-4.0", "no_upload": True, "status": "CPU_PASS_METAL_NOT_RUN", "artifacts": rows}
(root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="")
PY
  packet_sha="$(sha256_file "$packet_manifest")"
  printf '%s  %s\n' "$packet_sha" manifest.json > "$packet_dir/manifest.sha256"
  [[ "$(sha256_file "$packet_manifest")" == "$packet_sha" ]] || die 'packet manifest changed during publication'
  {
    printf '%q' 'scripts/verify/apple-silicon-nsnet2.sh'
    printf ' %q %q' --gguf '<NSNET2_PACKET>/nsnet2.gguf'
    printf ' %q %q' --gguf-sha256 "$(sha256_file "$packet_dir/nsnet2.gguf")"
    printf ' %q %q' --input '<NSNET2_PACKET>/input.wav'
    printf ' %q %q' --input-sha256 "$(sha256_file "$packet_dir/input.wav")"
    printf ' %q %q' --reference '<NSNET2_PACKET>/reference.wav'
    printf ' %q %q' --reference-sha256 "$(sha256_file "$packet_dir/reference.wav")"
    printf ' %q %q' --expected-head "$expected_head"
    printf ' %q %q' --approval-evidence '<NSNET2_PACKET>/approval.json'
    printf ' %q %q' --approval-sha256 "$approval_sha"
    printf ' %q %q' --packet-manifest '<NSNET2_PACKET>/manifest.json'
    printf ' %q %q' --packet-manifest-sha256 "$packet_sha"
    printf ' %q %q\n' --evidence-dir '<NSNET2_EVIDENCE_DIR>'
  } > "$transfer_args_file"

  actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || die 'could not re-read checkout HEAD before summary'
  [[ "$actual_head" == "$expected_head" ]] || die 'checkout HEAD changed before summary publication'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before summary publication'

  {
    echo "git_commit=$actual_head"
    echo "expected_head=$expected_head"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "onnx_sha256=$(sha256_file "$onnx_path")"
    echo "safetensors_sha256=$(sha256_file "$prepared_path")"
    echo "gguf_sha256=$(sha256_file "$gguf_path")"
    echo "input_wav_sha256=$(sha256_file "$REFERENCE_INPUT")"
    echo "reference_wav_sha256=$(sha256_file "$reference_wav")"
    echo "vast_local_gguf_path=$gguf_path"
    echo "vast_local_gguf_sha256=$(sha256_file "$gguf_path")"
    echo "vast_local_input_path=$REFERENCE_INPUT"
    echo "vast_local_input_sha256=$(sha256_file "$REFERENCE_INPUT")"
    echo "vast_local_reference_path=$reference_wav"
    echo "vast_local_reference_sha256=$(sha256_file "$reference_wav")"
    echo "vast_local_approval_evidence=$approval_evidence"
    echo "approval_sha256=$approval_sha"
    echo "packet_manifest_sha256=$packet_sha"
    echo "packet_dir=$packet_dir"
    echo "apple_expected_head=$expected_head"
    echo "apple_evidence_dir=$apple_evidence_dir"
    echo "transfer_packet_schema=vokra-nsnet2-transfer-v1"
    echo "transfer_packet_no_upload=NO_UPLOAD"
    echo "verdict=CPU_PASS_METAL_NOT_RUN"
    echo "nsnet2_cpu_reference=PASS"
    echo "nsnet2_metal_reference=NOT_RUN"
    echo "nsnet2_metal_vs_cpu=NOT_RUN"
    echo "apple_transfer_args=$transfer_args_file"
  } > "$summary_file"
  echo "run-nsnet2-validation: PASS"
  echo "Pull before destroy: $evidence_dir and $run_log"
  echo "Do not pull the generated model artifacts to the maintainer Mac."
}

main "$@"
