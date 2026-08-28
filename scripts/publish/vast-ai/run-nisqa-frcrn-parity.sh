#!/usr/bin/env bash
# Reproduce the first real-weight CPU measurements for NISQA v2 and FRCRN.
#
# This is an in-instance worker. It consumes the exact public vokra GGUFs that
# users download, plus independent pinned upstream implementations. It does not
# convert, publish, or upload anything. The numeric result stays measurement-
# only until VAST CPU and Apple-silicon Metal observations establish bounds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

NISQA_PUBLIC_REPO="vokra/nisqa-v2-weight"
NISQA_PUBLIC_REVISION="89718b026e17d3d048aa394ef8c8ddd14fee9cd8"
NISQA_PUBLIC_FILE="nisqa-v2-weight.gguf"
NISQA_PUBLIC_SHA256="a2cacbe6f81ea2e8255eb0e2137d70d245823758e1cc4bb180c6b7cccc131e07"
NISQA_SOURCE_REPO="https://github.com/gabrielmittag/NISQA"
NISQA_SOURCE_REVISION="fe84f0f252abec382b24367d5b22498a7ce34dbb"
NISQA_CHECKPOINT_SHA256="7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c"

FRCRN_PUBLIC_REPO="vokra/frcrn"
FRCRN_PUBLIC_REVISION="e4badbcb1dda0a91a59318f29417dde6c65e9f8b"
FRCRN_PUBLIC_FILE="frcrn.gguf"
FRCRN_PUBLIC_SHA256="04b8810e3f9e6391d9b95158fc34a2050bcac8618a3b25deb534a1b9cd42d7b6"
FRCRN_UPSTREAM_REPO="alibabasglab/FRCRN_SE_16K"
FRCRN_UPSTREAM_REVISION="3766e6a64b0d8cb58f08d913d617bf129f11ed53"
FRCRN_CHECKPOINT_FILE="last_best_checkpoint.pt"
FRCRN_CHECKPOINT_SHA256="b22256adbb91b68cf5a3db8f6657a4fb17066eecd5f069803e59c186c1cf3ebb"
FRCRN_SOURCE_REPO="https://github.com/modelscope/ClearerVoice-Studio"
FRCRN_SOURCE_REVISION="6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=20000000

log()  { printf '[nisqa-frcrn-vast] %s\n' "$*" >&2; }
step() { printf '\n[nisqa-frcrn-vast] ==== %s ====\n' "$*" >&2; }
die()  { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-nisqa-frcrn-parity.sh [--work-dir <empty-dir>]
       run-nisqa-frcrn-parity.sh --self-test

VAST-only real-weight CPU measurement worker. It:
  1. downloads the exact pinned public NISQA and FRCRN GGUFs;
  2. checks out the exact official source revisions and checkpoints;
  3. verifies every pinned SHA-256 before model execution;
  4. generates independent official reference outputs;
  5. records the CPU/ISA/torch/Rust/git environment;
  6. compiles vokra-models and runs both ignored measurement-only tests.

Options:
  --work-dir DIR  fresh run directory for artifacts and logs
                  (default: $VOKRA_SCRATCH/nisqa-frcrn-parity/<UTC timestamp>)
  --self-test     hermetic contract tests only; no network, models, or Cargo
  -h, --help      show this help

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh,
approximately 64 GB RAM, and at least 20 GB free disk. All inputs are public;
HF_TOKEN is unnecessary. This worker contains no publish or upload operation.
Pull only logs/manifests, then destroy the instance from the owner-side VAST
lifecycle.
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

verify_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || die "missing pinned input: $path"
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    die "SHA-256 mismatch for $path: got $actual, expected $expected"
    return 2
  fi
  log "SHA-256 OK: $(basename "$path") = $actual"
}

require_vast_marker() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh on VAST first"
}

require_vast_host() {
  local mem_kib free_kib
  require_vast_marker
  [[ "$(uname -s)" == "Linux" ]] \
    || die "actual model work is Linux/VAST-only; refusing host $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" { print $2; exit }' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal from /proc/meminfo"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 64 GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 { print $4 }')"
  [[ -n "$free_kib" ]] || die "could not read free disk for $VOKRA_SCRATCH"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 20 GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "$PARITY_PROJECT/uv.lock is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so the evidence names an exact commit"
  fi
}

checkout_exact_source() {
  local repository="$1" revision="$2" output="$3"
  [[ ! -e "$output" ]] || die "source target already exists: $output"
  mkdir -p "$output"
  git -C "$output" init -q
  git -C "$output" remote add origin "$repository"
  git -C "$output" fetch -q --depth=1 origin "$revision"
  git -C "$output" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$output" rev-parse HEAD)" == "$revision" ]] \
    || die "source checkout did not land on $revision"
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] \
    || die "source checkout is not clean: $output"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$repository" "$revision" "$filename" "$output"
  [[ -f "$output/$filename" ]] || die "Hugging Face download did not produce $output/$filename"
}

cpuinfo_value() {
  local wanted="$1" source="${2:-/proc/cpuinfo}"
  awk -F ':' -v wanted="$wanted" '
    {
      name = $1
      gsub(/[[:space:]]/, "", name)
      if (name == wanted) {
        value = $2
        sub(/^[[:space:]]+/, "", value)
        print value
        exit
      }
    }
  ' "$source"
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(cpuinfo_value modelname)"
  cpu_flags="$(cpuinfo_value flags)"
  [[ -n "$cpu_model" ]] || die "CPU model provenance could not be resolved"
  [[ -n "$cpu_flags" ]] || die "CPU ISA provenance could not be resolved"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" { print "mem_total_kib=" $2; exit }' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual cpuinfo script_path cases fail
  cases=0
  fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-nisqa-frcrn-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_hash "$payload" "$actual" >/dev/null 2>&1 \
    || { log "self-test FAIL: valid hash rejected"; fail=1; }

  cases=$((cases + 1))
  if verify_hash "$payload" "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid hash accepted"
    fail=1
  fi

  cases=$((cases + 1))
  if (unset VOKRA_PUBLISH_ON_VAST; require_vast_marker) >/dev/null 2>&1; then
    log "self-test FAIL: missing VAST marker accepted"
    fail=1
  fi

  cases=$((cases + 1))
  VOKRA_PUBLISH_ON_VAST=1 require_vast_marker >/dev/null 2>&1 \
    || { log "self-test FAIL: explicit VAST marker rejected"; fail=1; }

  cases=$((cases + 1))
  cpuinfo="$tmp/cpuinfo"
  printf 'model name\t: VAST Test CPU\nflags\t\t: avx avx2 fma\n' > "$cpuinfo"
  if [[ "$(cpuinfo_value modelname "$cpuinfo")" != "VAST Test CPU" ]] \
    || [[ "$(cpuinfo_value flags "$cpuinfo")" != "avx avx2 fma" ]]; then
    log "self-test FAIL: /proc/cpuinfo parser drifted"
    fail=1
  fi

  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in \
    "$NISQA_PUBLIC_REVISION" "$FRCRN_PUBLIC_REVISION" \
    "$NISQA_SOURCE_REVISION" "$FRCRN_SOURCE_REVISION" \
    "measure_real_cpu_against_official_nisqa" \
    "measure_real_cpu_against_official_clearervoice" \
    "--frozen --python 3.12" "--ignored --exact --nocapture"
  do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-nisqa-frcrn-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir logs_dir inputs_dir sources_dir
  local nisqa_dir frcrn_dir nisqa_source frcrn_source nisqa_gguf frcrn_gguf
  local nisqa_checkpoint frcrn_checkpoint nisqa_reference frcrn_reference
  local run_log env_log compile_log nisqa_log frcrn_log summary_file hash

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" ]] || { die "--work-dir requires a directory"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    run_self_test
    return $?
  fi

  require_vast_host
  require_tooling

  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/nisqa-frcrn-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  logs_dir="$work_dir/logs"
  inputs_dir="$work_dir/inputs"
  sources_dir="$work_dir/sources"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache"
  mkdir -p "$logs_dir" "$inputs_dir" "$sources_dir"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/compile.log"
  nisqa_log="$logs_dir/nisqa-cpu.log"
  frcrn_log="$logs_dir/frcrn-cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap 'rc=$?; summary_path="${summary_file:-}"; if [[ -n "$summary_path" && ! -f "$summary_path" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_path"; fi; exit "$rc"' EXIT

  nisqa_dir="$inputs_dir/nisqa"
  frcrn_dir="$inputs_dir/frcrn"
  nisqa_source="$sources_dir/nisqa"
  frcrn_source="$sources_dir/clearervoice"
  nisqa_gguf="$nisqa_dir/$NISQA_PUBLIC_FILE"
  frcrn_gguf="$frcrn_dir/$FRCRN_PUBLIC_FILE"
  nisqa_checkpoint="$nisqa_source/weights/nisqa.tar"
  frcrn_checkpoint="$frcrn_dir/$FRCRN_CHECKPOINT_FILE"
  nisqa_reference="$work_dir/reference/nisqa"
  frcrn_reference="$work_dir/reference/frcrn"

  step "Sync the locked Python 3.12 environment through uv"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download exact public GGUFs and official FRCRN checkpoint"
  download_hf_file "$NISQA_PUBLIC_REPO" "$NISQA_PUBLIC_REVISION" \
    "$NISQA_PUBLIC_FILE" "$nisqa_dir"
  download_hf_file "$FRCRN_PUBLIC_REPO" "$FRCRN_PUBLIC_REVISION" \
    "$FRCRN_PUBLIC_FILE" "$frcrn_dir"
  download_hf_file "$FRCRN_UPSTREAM_REPO" "$FRCRN_UPSTREAM_REVISION" \
    "$FRCRN_CHECKPOINT_FILE" "$frcrn_dir"

  step "Check out exact independent upstream sources"
  checkout_exact_source "$NISQA_SOURCE_REPO" "$NISQA_SOURCE_REVISION" "$nisqa_source"
  checkout_exact_source "$FRCRN_SOURCE_REPO" "$FRCRN_SOURCE_REVISION" "$frcrn_source"

  step "Verify every model/checkpoint hash before execution"
  verify_hash "$nisqa_gguf" "$NISQA_PUBLIC_SHA256"
  verify_hash "$frcrn_gguf" "$FRCRN_PUBLIC_SHA256"
  verify_hash "$nisqa_checkpoint" "$NISQA_CHECKPOINT_SHA256"
  verify_hash "$frcrn_checkpoint" "$FRCRN_CHECKPOINT_SHA256"

  step "Record environment before any numerical result"
  record_environment "$env_log"

  step "Generate independent official NISQA and ClearerVoice references"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/nisqa_dump_reference.py" \
    --source "$nisqa_source" --checkpoint "$nisqa_checkpoint" \
    --output "$nisqa_reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/frcrn_dump_reference.py" \
    --source "$frcrn_source" --checkpoint "$frcrn_checkpoint" \
    --output "$frcrn_reference"
  cp "$nisqa_reference/manifest.json" "$logs_dir/nisqa-reference-manifest.json"
  cp "$frcrn_reference/manifest.json" "$logs_dir/frcrn-reference-manifest.json"

  step "Compile the complete vokra-models library test target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib --no-run 2>&1 | tee "$compile_log"

  step "Measure NISQA CPU against the official NISQA_DIM forward"
  VOKRA_NISQA_GGUF="$nisqa_gguf" \
  VOKRA_NISQA_REFERENCE_DIR="$nisqa_reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      nisqa::tests::measure_real_cpu_against_official_nisqa \
      -- --ignored --exact --nocapture 2>&1 | tee "$nisqa_log"

  step "Measure FRCRN CPU against the official ClearerVoice DCCRN forward"
  VOKRA_FRCRN_GGUF="$frcrn_gguf" \
  VOKRA_FRCRN_REFERENCE_DIR="$frcrn_reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      frcrn::tests::measure_real_cpu_against_official_clearervoice \
      -- --ignored --exact --nocapture 2>&1 | tee "$frcrn_log"

  {
    echo "execution_status=PASS"
    echo "numeric_status=MEASURED_NOT_GATED"
    echo "numeric_bounds=UNSET"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "work_dir=$work_dir"
    for artifact in "$nisqa_gguf" "$frcrn_gguf" "$nisqa_checkpoint" "$frcrn_checkpoint"; do
      hash="$(sha256_file "$artifact")"
      printf 'sha256 %s %s\n' "$hash" "$(basename "$artifact")"
    done
    grep -E 'NISQA_MEASUREMENT|FRCRN_MEASUREMENT' "$nisqa_log" "$frcrn_log"
  } | tee "$summary_file"
  trap - EXIT

  step "Execution complete; numeric gates remain unset"
  log "pull this small evidence directory before destroy: $logs_dir"
  log "model artifacts and raw reference tensors remain on the disposable VAST instance"
}

main "$@"
