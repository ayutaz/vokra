#!/usr/bin/env bash
# Reproduce canonical T5-base encoder parity on VAST. Downloads public inputs
# only; never publishes or uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/t5_encoder"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/t5_encoder_dump_reference.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/musicgen-small"
PUBLIC_REVISION="30e7e356c9d8326c42965a337e810162d7cdbc70"
PUBLIC_FILE="model.gguf"
PUBLIC_BYTES=2364405568
PUBLIC_SHA256="be0a1d823cd4b4570e39cb87ce05a707959ffdffdc0aef23eb90fffa5c084a98"

T5_REPO="google-t5/t5-base"
T5_REVISION="a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
T5_WEIGHT_FILE="model.safetensors"
T5_WEIGHT_BYTES=891646390
T5_WEIGHT_SHA256="a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=100000000

log() { printf '[t5-encoder-vast] %s\n' "$*" >&2; }
step() { printf '\n[t5-encoder-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-t5-encoder-parity.sh [--work-dir <empty-dir>]
       run-t5-encoder-parity.sh --self-test

VAST-only official parity worker. It downloads the immutable public
vokra/musicgen-small GGUF and canonical google-t5/t5-base snapshot, verifies
their exact identities, calls the official Transformers T5EncoderModel
forward, and compares Vokra's native CPU result. It also runs focused unit
tests and an aarch64-apple-darwin Metal-feature cross-check.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, a
64-GB-class host (at least 60,000,000 KiB RAM) and 100,000,000 KiB free disk.
The script contains no publish or upload operation. Pull logs/reference
evidence, then destroy the instance.
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

verify_reference_manifest() {
  local reference="$1" checkpoint="$2"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); checkpoint=pathlib.Path(sys.argv[2]); repo=sys.argv[3]; revision=sys.argv[4]
manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
def file_identity(path):
    digest=hashlib.sha256(); size=0
    with path.open("rb") as handle:
        while block := handle.read(8*1024*1024):
            size += len(block); digest.update(block)
    return size, digest.hexdigest()
assert manifest["format"] == "vokra-t5-encoder-reference-v1"
assert manifest["oracle"] == "transformers.T5EncoderModel.forward"
assert manifest["source_repo"] == repo
assert manifest["source_revision"] == revision
for name, expected in manifest["fixtures"].items():
    size, sha256=file_identity(root/name)
    assert size == expected["bytes"], (name, "bytes")
    assert sha256 == expected["sha256"], (name, "sha256")
for name, expected in manifest["checkpoint_files"].items():
    size, sha256=file_identity(checkpoint/name)
    assert size == expected["bytes"], (name, "bytes")
    assert sha256 == expected["sha256"], (name, "sha256")
print(f"reference manifest OK: {repo}@{revision}")' \
    "$reference" "$checkpoint" "$T5_REPO" "$T5_REVISION"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$repository" "$revision" "$filename" "$output"
  [[ -f "$output/$filename" ]] \
    || die "Hugging Face download did not produce $output/$filename"
}

download_t5_snapshot() {
  local output="$1"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], allow_patterns=["config.json", "model.safetensors"], local_dir=sys.argv[3]))' \
    "$T5_REPO" "$T5_REVISION" "$output"
  [[ -f "$output/config.json" && -f "$output/$T5_WEIGHT_FILE" ]] \
    || die "T5 snapshot did not produce config.json + $T5_WEIGHT_FILE"
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
    die "MemTotal=${mem_kib} KiB is below the 64-GB class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 100-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup awk grep find tee wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "dedicated T5 parity uv.lock is missing"
  [[ -f "$PARITY_DUMPER" ]] || die "official T5 parity dumper is missing"
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
      'import platform, torch, transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-t5-encoder-self-test\n' > "$payload"
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
  script_path="${BASH_SOURCE[0]}"
  for required in "$PUBLIC_REVISION" "$PUBLIC_SHA256" "$T5_REVISION" \
    "$T5_WEIGHT_SHA256" "t5_encoder_dump_reference.py" \
    "parity_t5_base_official_hidden_states_cpu_and_metal" \
    "T5_BASE_OFFICIAL_PARITY backend=cpu" \
    "verify_reference_manifest" \
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
    uv run --no-project --python 3.12 python -c \
      'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' \
      "$PARITY_DUMPER" >/dev/null 2>&1 \
    || { log "self-test FAIL: dumper syntax parse failed"; fail=1; }

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-t5-encoder-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir
  local public_dir t5_dir reference gguf
  local run_log env_log ops_log models_log cross_log cpu_log summary_file
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/t5-encoder-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/musicgen-small"
  t5_dir="$inputs_dir/t5-base"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$public_dir" "$t5_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-t5-encoder"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  ops_log="$logs_dir/ops-tests.log"
  models_log="$logs_dir/models-tests.log"
  cross_log="$logs_dir/apple-metal-cross-check.log"
  cpu_log="$logs_dir/official-cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync dedicated locked Python 3.12 CPU environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download and verify exact public MusicGen-Small GGUF"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  gguf="$public_dir/$PUBLIC_FILE"
  verify_file "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"

  step "Download and verify canonical T5-base snapshot"
  download_t5_snapshot "$t5_dir"
  verify_file "$t5_dir/$T5_WEIGHT_FILE" "$T5_WEIGHT_BYTES" "$T5_WEIGHT_SHA256"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official T5EncoderModel reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_DUMPER" \
    --checkpoint "$t5_dir" \
    --source-repo "$T5_REPO" \
    --source-revision "$T5_REVISION" \
    --output-dir "$reference"
  verify_reference_manifest "$reference" "$t5_dir"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Run focused T5 relative-position and native encoder tests on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-ops t5_relative_position 2>&1 | tee "$ops_log"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib t5_encoder::tests 2>&1 | tee "$models_log"

  step "Cross-check the Apple Metal feature route"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$cross_log"

  step "Run opted-in official CPU parity against exact public GGUF"
  VOKRA_T5_BASE_GGUF="$gguf" \
  VOKRA_T5_BASE_REFERENCE_DIR="$reference" \
  VOKRA_T5_BASE_PREFIX="text_encoder" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_t5_encoder \
      parity_t5_base_official_hidden_states_cpu_and_metal \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"

  step "Write evidence summary and checksums"
  grep -F "T5_BASE_OFFICIAL_PARITY backend=cpu" "$cpu_log" \
    || die "official CPU parity verdict line is missing"
  {
    echo "execution_status=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_file=$PUBLIC_FILE"
    echo "public_bytes=$PUBLIC_BYTES"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "t5_repo=$T5_REPO"
    echo "t5_revision=$T5_REVISION"
    echo "t5_weight_bytes=$T5_WEIGHT_BYTES"
    echo "t5_weight_sha256=$T5_WEIGHT_SHA256"
    echo "metal_runtime=NOT_RUN_LINUX_VAST"
    echo "metal_cross_compile=PASS"
  } | tee "$summary_file"
  (
    cd "$work_dir"
    find logs reference -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > logs/SHA256SUMS
  )
  log "PASS evidence: $logs_dir and $reference"
}

main "$@"
