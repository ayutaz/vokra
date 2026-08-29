#!/usr/bin/env bash
# VAST-only real-weight validation for YuE xcodec-mini.
# The exact public GGUF is used for runtime parity because the production
# binder is intentionally pinned to its historical 2,145-tensor manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/yue_xcodec_mini"
REFERENCE_DUMPER="$VOKRA_ROOT/tools/parity/yue_xcodec_mini_dump_reference.py"
REFERENCE_VALIDATOR="$PARITY_PROJECT/reference_validator.py"
PRE_FLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PRE_FLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

PUBLIC_REPO="vokra/yue-xcodec-mini"
PUBLIC_REVISION="83c14a67ed792a0d5b3b61fff8ae35a04c6da8fa"
PUBLIC_FILE="yue-xcodec-mini.gguf"
PUBLIC_BYTES=1810001760
PUBLIC_SHA256="60e21aa5335646080102196454d7ffad5e012467d6f5eb9b776bf07d666b02bc"
PUBLIC_TENSOR_COUNT=2145
UPSTREAM_REPO="m-a-p/xcodec_mini_infer"
UPSTREAM_REVISION="fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"
SOURCE_README_BYTES=31
SOURCE_README_SHA256="4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7"
SOURCE_FILES=("quantization/__init__.py" "quantization/vq.py" "quantization/core_vq_lsx_version.py" "quantization/distrib.py" "utils/utils.py" "utils/ddp_utils.py")
CODEC_FILE="final_ckpt/ckpt_00360000.pth"
CODEC_BYTES=1360444883
CODEC_SHA256="c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c"
SEMANTIC_FILE="semantic_ckpts/hf_1_325000/pytorch_model.bin"
SEMANTIC_BYTES=377555286
SEMANTIC_SHA256="c5ddbd7fa2468483cb9b2aa53117813471543dd278e65870333a56c54305f527"
DECODER_FILE="decoders/decoder_151000.pth"
DECODER_BYTES=72610550
DECODER_SHA256="8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=120000000

log() { printf '[yue-xcodec-mini-vast] %s\n' "$*" >&2; }
step() { printf '\n[yue-xcodec-mini-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

pre_sync_gate() {
  local approval_evidence="$1"
  command -v uv >/dev/null 2>&1 || die 'uv is required before the YuE gate'
  [[ -f "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" &&
     -f "$PRE_FLIGHT_GATE" && -f "$PRE_FLIGHT_MANIFEST" ]] || die 'YuE gate inputs are missing'
  [[ -f "$approval_evidence" && ! -L "$approval_evidence" ]] || die 'approval evidence must be a regular file'
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PRE_FLIGHT_GATE" \
    --project "$PARITY_PROJECT" --manifest "$PRE_FLIGHT_MANIFEST" --approval-evidence "$approval_evidence"; then
    die 'YuE preflight gate rejected the manifest or approval evidence'
    return 2
  fi
}

usage() {
  cat <<'EOF' >&2
usage: run-yue-xcodec-mini-validation.sh --approval-evidence <external-evidence.json> [--work-dir <empty-dir>]
       run-yue-xcodec-mini-validation.sh --self-test

VAST-only validation of the immutable public YuE xcodec-mini GGUF against an
independent upstream RVQ+Vocos oracle. The public artifact is authenticated
by revision, byte length, and SHA-256, then bound by the strict production
2,145-tensor binder. Corrected conversion is intentionally not attempted:
the historical production SPEC is pinned to this public manifest, so a
replacement binding remains a separate production task. Numeric results are
MEASURED_NOT_GATED; this worker never uploads or publishes.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die 'neither sha256sum nor shasum is available'
  fi
}

verify_file() {
  local path="$1" expected_bytes="$2" expected_hash="$3" actual_bytes actual_hash
  [[ -f "$path" && ! -L "$path" ]] || die "missing pinned input or symlink: $path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"
  log "identity OK: $path bytes=$actual_bytes sha256=$actual_hash"
}

require_one_cargo_result() {
  local log_path="$1" test_name="$2"
  [[ "$(grep -Ec "^test $test_name \.\.\. ok$" "$log_path" || true)" == 1 ]] \
    || { die 'named Cargo test did not pass exactly once'; return 2; }
  [[ "$(grep -Ec '^test [^ ]+ \.\.\.' "$log_path" || true)" == 1 ]] \
    || { die 'Cargo emitted extra/missing test lines'; return 2; }
  [[ "$(grep -Ec '^test result:' "$log_path" || true)" == 1 ]] \
    || { die 'Cargo emitted extra/missing result lines'; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" \
    || { die 'Cargo result is not exact'; return 2; }
}

require_cpu_measurement() {
  local log_path="$1"
  [[ "$(grep -Ec '^YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu ' "$log_path" || true)" == 1 ]] \
    || { die 'CPU measurement sentinel family is not singleton'; return 2; }
  [[ "$(grep -Ec '^YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED$' "$log_path" || true)" == 1 ]] \
    || { die 'CPU measurement sentinel is not exactly one full line'; return 2; }
}

write_apple_args() {
  local output="$1" gguf_sha="$2" reference_sha="$3" approval_path="$4"
  {
    printf '# Portable Apple YuE xcodec-mini validation command; no VAST paths.\n'
    printf 'scripts/verify/apple-silicon-yue-xcodec-mini.sh \\\n'
    printf "  --gguf '%s' \\\n" '<APPLE_YUE_XCODEC_MINI_GGUF_PATH>'
    printf "  --gguf-sha256 '%s' \\\n" "$gguf_sha"
    printf "  --reference '%s' \\\n" '<APPLE_YUE_XCODEC_MINI_REFERENCE_DIR>'
    printf "  --reference-manifest-sha256 '%s' \\\n" "$reference_sha"
    printf "  --approval-evidence '%s' \\\n" "$approval_path"
    printf "  --evidence-dir '%s'\n" '<APPLE_EMPTY_EVIDENCE_DIR>'
  } > "$output"
}

canonical_candidate() {
  local value="$1" suffix='' parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"; [[ -n "$value" ]] || { die 'path is empty'; return 2; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || { die "path contains a symlink ancestor: $parent"; return 2; }
    parent="$(dirname "$parent")"
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die 'work-dir path has no canonical parent'; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die "work-dir parent is not a real directory: $value"; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

canonical_file() {
  local value="$1" parent base
  [[ -f "$value" && ! -L "$value" ]] || { die "approval evidence must be a regular file: $value"; return 2; }
  parent="$(dirname "$value")"; base="$(basename "$value")"
  parent="$(canonical_candidate "$parent")" || return 2
  printf '%s/%s\n' "$parent" "$base"
}

paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

validate_work_dir() {
  local candidate="$1" approval="$2" canonical_work canonical_root canonical_project canonical_approval
  [[ -n "$candidate" ]] || { die 'work-dir is empty'; return 2; }
  if [[ -L "$candidate" ]]; then
    die "work-dir must not be a symlink: $candidate"; return 2
  fi
  if [[ -e "$candidate" ]]; then
    [[ -d "$candidate" ]] || { die "work-dir must be a directory: $candidate"; return 2; }
    [[ -z "$(find "$candidate" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || { die "work-dir must be absent or empty: $candidate"; return 2; }
  fi
  canonical_work="$(canonical_candidate "$candidate")" || return 2
  canonical_root="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  canonical_project="$(canonical_candidate "$PARITY_PROJECT")" || return 2
  canonical_approval="$(canonical_file "$approval")" || return 2
  for protected in "$canonical_root" "$canonical_project" "$canonical_approval"; do
    if paths_overlap "$canonical_work" "$protected"; then
      die "work-dir overlaps protected path: $protected"; return 2
    fi
  done
  printf '%s\n' "$canonical_work"
}

require_vast_host() {
  local mem_kib free_kib disk_root
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == '1' ]] \
    || die 'VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first'
  [[ "$(uname -s)" == 'Linux' ]] \
    || die "YuE xcodec-mini source work is VAST-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die '/proc/meminfo is unavailable'
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'could not read MemTotal'
  (( mem_kib >= MIN_VAST_MEM_KIB )) \
    || die "MemTotal=${mem_kib} KiB is below the 60-GB VAST guard"
  disk_root="$VOKRA_SCRATCH"
  while [[ ! -e "$disk_root" && ! -L "$disk_root" ]]; do
    [[ "$disk_root" != / ]] || die 'scratch path has no existing disk ancestor'
    disk_root="$(dirname "$disk_root")"
  done
  disk_root="$(canonical_candidate "$disk_root")" || return 2
  free_kib="$(df -Pk "$disk_root" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'could not read free disk'
  (( free_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk=${free_kib} KiB is below the 120-GB VAST guard"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr sort nproc; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die 'parity uv.lock is missing'
  [[ -f "$REFERENCE_DUMPER" ]] || die 'xcodec-mini reference dumper is missing'
  [[ -n "$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel)" ]] \
    || die 'Vokra git root is unavailable'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die 'VAST checkout must be clean so evidence names one exact commit'
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4])' \
    "$repository" "$revision" "$filename" "$output_dir"
  [[ -f "$output_dir/$filename" && ! -L "$output_dir/$filename" ]] || die "HF download did not produce regular $output_dir/$filename"
  # huggingface_hub may leave a transport-only child; it is not part of the
  # authenticated source snapshot and must never reach the oracle.
  [[ ! -e "$output_dir/.cache" ]] || rm -rf -- "$output_dir/.cache"
}

stage_source() {
  local output="$1"
  mkdir -p "$output"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" README.md "$output"
  for relative in "${SOURCE_FILES[@]}"; do
    download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$relative" "$output"
  done
  verify_file "$output/README.md" "$SOURCE_README_BYTES" "$SOURCE_README_SHA256"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "uname=$(uname -a)"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform,torch,vocos; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print("vocos=0.1.0")'
  } | tee "$output"
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" fail=0 cases=0 required temporary apple_args approval
  for required in "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" \
    "$PUBLIC_SHA256" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" \
    "$CODEC_SHA256" "$SEMANTIC_SHA256" "$DECODER_SHA256" \
    "yue_xcodec_mini_dump_reference.py" "weights_only=True_required" \
    "yue_xcodec_mini::tests::measure_real_cpu_against_official_xcodec_and_vocos" \
    "--frozen --python 3.12" "--ignored --exact --nocapture" \
    "YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET" \
    "MEASURED_NOT_GATED" "replacement binding remains a separate production task" \
    "--approval-evidence" "reference_validator.py"; do
    cases=$((cases + 1))
    grep -Fq -- "$required" "$script_path" || { log "self-test FAIL: missing token: $required"; fail=1; }
  done
  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log 'self-test FAIL: direct Python/pip command found'; fail=1
  fi
  cases=$((cases + 1))
  if grep -En '(publish-one\.sh|upload\.sh|--push([[:space:]]|$)|huggingface-cli[[:space:]])' "$script_path" >/dev/null; then
    log 'self-test FAIL: upload/publication operation found'; fail=1
  fi
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-yue-xcodec-vast.XXXXXX")"
  apple_args="$temporary/apple.args.sh"
  write_apple_args "$apple_args" "$PUBLIC_SHA256" "$(printf '%064d' 0)" '<APPLE_YUE_XCODEC_APPROVAL_EVIDENCE>'
  bash -n "$apple_args" || { log 'self-test FAIL: generated Apple args are not shell syntax'; fail=1; }
  grep -Fq -- "--gguf '<APPLE_YUE_XCODEC_MINI_GGUF_PATH>'" "$apple_args" \
    || { log 'self-test FAIL: GGUF placeholder is not quoted'; fail=1; }
  grep -Fq -- "--reference '<APPLE_YUE_XCODEC_MINI_REFERENCE_DIR>'" "$apple_args" \
    || { log 'self-test FAIL: reference placeholder is not quoted'; fail=1; }
  if grep -Fq "$VOKRA_SCRATCH" "$apple_args"; then
    log 'self-test FAIL: generated Apple args contain a VAST path'; fail=1
  fi
  grep -Fq -- "--approval-evidence '<APPLE_YUE_XCODEC_APPROVAL_EVIDENCE>'" "$apple_args" || { log 'self-test FAIL: approval placeholder is not portable/quoted'; fail=1; }
  for option in --work-dir --approval-evidence; do
    if "$script_path" "$option" -bad >/dev/null 2>&1; then log "self-test FAIL: leading-dash value accepted for $option"; fail=1; fi
  done
  if "$script_path" --work-dir one --work-dir two >/dev/null 2>&1; then log 'self-test FAIL: duplicate --work-dir accepted'; fail=1; fi
  if "$script_path" --approval-evidence one --approval-evidence two >/dev/null 2>&1; then log 'self-test FAIL: duplicate --approval-evidence accepted'; fail=1; fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then log 'self-test FAIL: duplicate --self-test accepted'; fail=1; fi
  printf 'test measure_real_cpu_against_official_xcodec_and_vocos ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s\n' > "$temporary/cargo.log"
  require_one_cargo_result "$temporary/cargo.log" measure_real_cpu_against_official_xcodec_and_vocos || fail=1
  printf 'test extra ... ok\n' >> "$temporary/cargo.log"
  if require_one_cargo_result "$temporary/cargo.log" measure_real_cpu_against_official_xcodec_and_vocos; then log 'self-test FAIL: duplicate Cargo test line accepted'; fail=1; fi
  printf 'test measure_real_cpu_against_official_xcodec_and_vocos ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in nope\n' > "$temporary/malformed.log"
  if require_one_cargo_result "$temporary/malformed.log" measure_real_cpu_against_official_xcodec_and_vocos; then log 'self-test FAIL: malformed Cargo timing accepted'; fail=1; fi
  rm -rf "$temporary"
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-yue-xcodec-workdir.XXXXXX")"
  # macOS commonly exposes TMPDIR through /var -> /private/var.  Exercise the
  # positive path contract with its physical spelling so the symlink-ancestor
  # rejection remains meaningful rather than rejecting every self-test path.
  temporary="$(cd -P "$temporary" && pwd)"
  approval="$temporary/approval.json"
  printf '{}\n' > "$approval"
  if validate_work_dir "$temporary/work" "$approval" >/dev/null 2>&1; then :; else
    log 'self-test FAIL: absent work-dir was rejected'; fail=1
  fi
  mkdir "$temporary/nonempty"
  : > "$temporary/nonempty/file"
  if validate_work_dir "$temporary/nonempty" "$approval" >/dev/null 2>&1; then log 'self-test FAIL: nonempty work-dir accepted'; fail=1; fi
  ln -s "$temporary/missing" "$temporary/symlink-work"
  if validate_work_dir "$temporary/symlink-work" "$approval" >/dev/null 2>&1; then log 'self-test FAIL: symlink work-dir accepted'; fail=1; fi
  if validate_work_dir "$VOKRA_ROOT" "$approval" >/dev/null 2>&1; then log 'self-test FAIL: repository-overlapping work-dir accepted'; fail=1; fi
  if validate_work_dir "$temporary" "$approval" >/dev/null 2>&1; then log 'self-test FAIL: approval-overlapping work-dir accepted'; fail=1; fi
  if validate_work_dir "$temporary/approval-child" "$approval" >/dev/null 2>&1; then :; else
    log 'self-test FAIL: unrelated work-dir was rejected'; fail=1
  fi
  rm -rf "$temporary"
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-yue-xcodec-gate-proof.XXXXXX")"
  temporary="$(cd -P "$temporary" && pwd)"
  set +e
  approval="$temporary/approval.json"
  printf '{}\n' > "$approval"
  VOKRA_SCRATCH="$temporary/scratch" VOKRA_PUBLISH_ON_VAST=1 "$script_path" \
    --approval-evidence "$approval" --work-dir "$temporary/work" >"$temporary/production.log" 2>&1
  local production_rc=$?
  set -e
  if [[ $production_rc -ne 2 || -e "$temporary/scratch" || -e "$temporary/work" || -e "$temporary/uv-cache-yue-xcodec-mini" ]]; then
    log 'self-test FAIL: production gate did not stop before work/cache creation'; fail=1
  fi
  rm -rf "$temporary"
  if [[ $fail -eq 0 ]]; then
    echo "run-yue-xcodec-mini-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

on_exit() {
  local rc=$?
  if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then
    printf 'execution_status=FAIL\nexit_code=%s\n' "$rc" > "$summary_file"
  fi
  exit "$rc"
}

main() {
  local self_test=0 requested_work_dir='' approval_evidence='' seen_self_test=0 run_stamp work_dir inputs logs reference
  local public_dir upstream_dir gguf codec semantic decoder source_root env_log cpu_log summary_file run_log
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$requested_work_dir" ]] || { die '--work-dir requires one directory'; return 2; }; requested_work_dir="$2"; shift 2 ;;
      --approval-evidence) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$approval_evidence" ]] || { die '--approval-evidence requires one file'; return 2; }; approval_evidence="$2"; shift 2 ;;
      --self-test) [[ $seen_self_test -eq 0 ]] || { die '--self-test may appear only once'; return 2; }; seen_self_test=1; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ -z "$approval_evidence$requested_work_dir" ]] || { die '--self-test accepts no other arguments'; return 2; }
    run_self_test
    return $?
  fi
  [[ -n "$approval_evidence" ]] || { usage; die '--approval-evidence is required'; }
  pre_sync_gate "$approval_evidence"
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/yue-xcodec-mini-validation/$run_stamp}"
  work_dir="$(validate_work_dir "$work_dir" "$approval_evidence")" || return 2
  require_vast_host
  require_tooling
  inputs="$work_dir/inputs"
  logs="$work_dir/logs"
  reference="$work_dir/reference"
  public_dir="$inputs/public"
  upstream_dir="$inputs/upstream"
  source_root="$inputs/source"
  gguf="$public_dir/$PUBLIC_FILE"
  codec="$upstream_dir/$CODEC_FILE"
  semantic="$upstream_dir/$SEMANTIC_FILE"
  decoder="$upstream_dir/$DECODER_FILE"
  mkdir -p "$public_dir" "$upstream_dir" "$logs" "$reference"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-yue-xcodec-mini"
  export HF_HOME="$VOKRA_SCRATCH/hf-home-yue-xcodec-mini"
  run_log="$logs/run.log"
  env_log="$logs/environment.txt"
  cpu_log="$logs/cpu.log"
  summary_file="$logs/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap on_exit EXIT

  step 'Sync locked Python 3.12 parity environment'
  UV_NO_CACHE=1 uv sync --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12

  step 'Download and authenticate exact public GGUF and upstream checkpoints'
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CODEC_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$SEMANTIC_FILE" "$upstream_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$DECODER_FILE" "$upstream_dir"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  verify_file "$codec" "$CODEC_BYTES" "$CODEC_SHA256"
  verify_file "$semantic" "$SEMANTIC_BYTES" "$SEMANTIC_SHA256"
  verify_file "$decoder" "$DECODER_BYTES" "$DECODER_SHA256"

  step 'Download and pin the independent upstream HF source snapshot'
  stage_source "$source_root"

  step 'Record environment before official oracle output'
  record_environment "$env_log"

  step 'Generate independent official RVQ plus Vocos reference'
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_DUMPER" --source-root "$source_root" --codec-checkpoint "$codec" \
    --decoder-checkpoint "$decoder" --frames 5 --output-dir "$reference"
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$REFERENCE_VALIDATOR" --reference "$reference"; then
    die 'YuE reference validator rejected the generated manifest or payloads'
    return 2
  fi
  grep -Fq '"format": "vokra-yue-xcodec-mini-reference-v2"' "$reference/manifest.json" \
    || die 'reference format v2 marker is missing'
  grep -Fq '"pickle_load_policy": "weights_only=True_required"' "$reference/manifest.json" \
    || die 'reference safe pickle policy marker is missing'
  write_apple_args "$logs/apple-silicon-yue-xcodec-mini.args.sh" \
    "$PUBLIC_SHA256" "$(sha256_file "$reference/manifest.json")" '<APPLE_YUE_XCODEC_APPROVAL_EVIDENCE>'

  step 'Run named nonzero native CPU measurement against official output'
  VOKRA_YUE_XCODEC_MINI_GGUF="$gguf" \
  VOKRA_YUE_XCODEC_MINI_REFERENCE_DIR="$reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      yue_xcodec_mini::tests::measure_real_cpu_against_official_xcodec_and_vocos \
      -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$cpu_log"
  require_cpu_measurement "$cpu_log"
  require_one_cargo_result "$cpu_log" measure_real_cpu_against_official_xcodec_and_vocos

  step 'Write evidence summary and checksums'
  {
    echo 'execution_status=PASS'
    echo 'numeric_verdict=MEASURED_NOT_GATED'
    echo 'numeric_bounds=UNSET'
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_gguf_bytes=$PUBLIC_BYTES"
    echo "public_gguf_sha256=$PUBLIC_SHA256"
    echo "public_tensor_count=$PUBLIC_TENSOR_COUNT"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "codec_sha256=$CODEC_SHA256"
    echo "semantic_sha256=$SEMANTIC_SHA256"
    echo "decoder_sha256=$DECODER_SHA256"
    echo 'runtime_artifact=exact_public_gguf'
    echo 'corrected_replacement_binding=SEPARATE_PRODUCTION_TASK'
    echo 'upload=NOT_PERFORMED'
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    grep -F 'YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu' "$cpu_log"
  } | tee "$summary_file"
  (
    cd "$work_dir"
    find inputs reference logs -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum > logs/SHA256SUMS
  )
  trap - EXIT
  log 'MEASURED_NOT_GATED: pull evidence, remove staged inputs, then destroy the VAST instance'
}

main "$@"
