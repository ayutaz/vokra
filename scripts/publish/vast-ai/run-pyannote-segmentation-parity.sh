#!/usr/bin/env bash
# Reproduce the official pyannote.audio PyanNet parity on VAST.
# Downloads public inputs only; never publishes or uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/pyannote_segmentation"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/pyannote_segmentation_dump_reference.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/pyannote-segmentation-3.0"
PUBLIC_REVISION="50bf4e510e0c689668384aec0f866f02e0fcaea8"
PUBLIC_FILE="pyannote-seg.gguf"
PUBLIC_BYTES=5898272
PUBLIC_SHA256="22ff05fddf19e69c8d9aac8daa6d99014e6718bcd8d8c527d26da677d00c63f1"

PYANNOTE_SOURCE_REPO="https://github.com/pyannote/pyannote-audio.git"
PYANNOTE_SOURCE_REVISION="795b92ab265888c58d160f90ae4d91b7bcc6aa2c"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=20000000

log() { printf '[pyannote-vast] %s\n' "$*" >&2; }
step() { printf '\n[pyannote-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-pyannote-segmentation-parity.sh [--work-dir <empty-dir>]
       run-pyannote-segmentation-parity.sh --self-test

VAST-only official parity worker. It downloads the exact public PyanNet GGUF,
checks out immutable pyannote.audio 3.0.0 source, restores the GGUF into the
official PyanNet modules, generates reference probabilities, and compares the
strict native CPU forward at the standard FP32 absolute bound of 0.01.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, a 64-GB
RAM class and at least 20 GB free disk. This script has no publish or upload
operation. Pull the small logs/manifests, then destroy the instance.
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
  [[ -f "$path" ]] || die "missing pinned input: $path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"
    return 2
  fi
  actual_hash="$(sha256_file "$path")"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"
    return 2
  fi
  log "identity OK: $path bytes=$actual_bytes sha256=$actual_hash"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4], local_dir_use_symlinks=False))' \
    "$repository" "$revision" "$filename" "$output"
  [[ -f "$output/$filename" ]] \
    || die "Hugging Face download did not produce $output/$filename"
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

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "model work is Linux/VAST-only; refusing host $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 64-GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 20-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "dedicated parity uv.lock is missing"
  [[ -f "$PARITY_DUMPER" ]] || die "official parity dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import importlib.metadata, platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}"); print("pyannote_audio=" + importlib.metadata.version("pyannote.audio"))'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-pyannote-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" \
    >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }

  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid byte size accepted"
    fail=1
  fi

  cases=$((cases + 1))
  if verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" \
    "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid SHA-256 accepted"
    fail=1
  fi

  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in "$PUBLIC_REVISION" "$PUBLIC_SHA256" "$PYANNOTE_SOURCE_REVISION" \
    "pyannote_segmentation_dump_reference.py" \
    "parity_pyannote_official_probabilities" \
    "PYANNOTE_OFFICIAL_PARITY backend=cpu" \
    "--frozen --python 3.12" "--ignored --exact --nocapture"; do
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

  cases=$((cases + 1))
  UV_CACHE_DIR="${UV_CACHE_DIR:-$tmp/uv-cache}" \
    uv run --no-project --python 3.12 python "$PARITY_DUMPER" --self-test \
      >/dev/null 2>&1 || { log "self-test FAIL: dumper self-test failed"; fail=1; }

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-pyannote-segmentation-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir sources_dir logs_dir
  local public_dir pyannote_source gguf reference
  local run_log env_log compile_log cli_log smoke_log cpu_log summary_file
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/pyannote-segmentation-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  sources_dir="$work_dir/sources"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/public"
  pyannote_source="$sources_dir/pyannote-audio"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$public_dir" "$sources_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-pyannote-segmentation"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/models-compile.log"
  cli_log="$logs_dir/cli-route.log"
  smoke_log="$logs_dir/public-smoke.log"
  cpu_log="$logs_dir/official-cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  # `rc` is assigned inside the single-quoted EXIT trap body.
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync dedicated locked Python 3.12 environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download and verify exact public GGUF"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  gguf="$public_dir/$PUBLIC_FILE"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"

  step "Check out exact official pyannote.audio source"
  checkout_exact_source "$PYANNOTE_SOURCE_REPO" "$PYANNOTE_SOURCE_REVISION" "$pyannote_source"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official PyanNet reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_DUMPER" --pyannote-source "$pyannote_source" \
    --gguf "$gguf" --output-dir "$reference"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Compile the focused vokra-models parity target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test parity_pyannote_segmentation --no-run \
    2>&1 | tee "$compile_log"

  step "Compile and verify the pyannote CLI dispatch route on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_detects_pyannote_as_segment_task \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Run strict public-GGUF CPU smoke"
  PARITY_PYANNOTE_REAL_GGUF="$gguf" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_pyannote_segmentation \
      parity_pyannote_public_gguf_smoke -- --exact --nocapture \
      2>&1 | tee "$smoke_log"

  step "Compare native CPU probabilities with official PyanNet.forward"
  PARITY_PYANNOTE_REAL_GGUF="$gguf" \
  PARITY_PYANNOTE_REFERENCE_DIR="$reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_pyannote_segmentation \
      parity_pyannote_official_probabilities \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"

  grep -F "PYANNOTE_OFFICIAL_PARITY backend=cpu" "$cpu_log" | grep -F "verdict=PASS" >/dev/null \
    || die "official CPU parity PASS sentinel missing"
  {
    echo "execution_status=PASS"
    echo "numeric_verdict=PASS"
    echo "numeric_bound=0.01"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "pyannote_source_revision=$PYANNOTE_SOURCE_REVISION"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    grep -F "PYANNOTE_OFFICIAL_PARITY" "$cpu_log"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir, then destroy the VAST instance"
}

main "$@"
