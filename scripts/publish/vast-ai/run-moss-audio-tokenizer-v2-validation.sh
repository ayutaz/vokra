#!/usr/bin/env bash
# Authenticate, convert, and generate the first official reference for the
# 8.49 GB MOSS Audio Tokenizer v2 release. VAST-only; never publishes/uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
AUDITOR="$VOKRA_ROOT/tools/audit/moss_audio_tokenizer_v2_manifest.py"
PREPARER="$PARITY_PROJECT/moss_audio_tokenizer_prepare_checkpoint.py"
REFERENCE_DUMPER="$PARITY_PROJECT/moss_audio_tokenizer_dump_reference.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

UPSTREAM_REPO="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
UPSTREAM_REVISION="f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
CANDIDATE_MANIFEST_SHA256="a83915cffe78cee7f031e18ac3de1bbd64e93b3e4af843ff28d531ccf81748c6"
SHARD1_SHA256="2d9f9182f17b143a23937feb87c63c08221bd28e685e4bc2fa55dcdce17fcde7"
SHARD2_SHA256="d4e48106d0254fe3b00ea0707e88fc6aee076993825e108dd9cef847f9db236e"
SHARD3_SHA256="d0449fe1b0ef1f6045946867148d8166b9a91a58d0feca4a18b641494d0b22da"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=100000000
MIN_GPU_MEM_MIB=20000

log() { printf '[moss-tokenizer-v2-vast] %s\n' "$*" >&2; }
step() { printf '\n[moss-tokenizer-v2-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-moss-audio-tokenizer-v2-validation.sh [--work-dir <empty-dir>]
       run-moss-audio-tokenizer-v2-validation.sh --self-test

VAST-only artifact/reference gate for MOSS Audio Tokenizer v2. It downloads
the immutable official three-shard release, verifies every exact file and real
safetensors header, merges and converts it, authenticates the resulting GGUF
header, and invokes the pinned official model's decode path on CUDA.

This worker deliberately does not invent a numerical bound: after producing
the authenticated manifest and independent reference it runs the mapped native
CPU decoder as a measurement-only comparison and cross-checks the Apple Metal
feature build. It contains no publish, upload, or Hugging Face push operation.
Pull only logs/reference evidence, then destroy the instance.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, at least
60,000,000 KiB RAM, 100,000,000 KiB free disk, and a CUDA GPU with at least
20,000 MiB memory. The public checkpoint requires no token.
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
  if [[ -n "$expected_bytes" && "$actual_bytes" != "$expected_bytes" ]]; then
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

require_vast_marker() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
}

require_vast_host() {
  local mem_kib free_kib gpu_mem_mib
  require_vast_marker
  [[ "$(uname -s)" == "Linux" ]] \
    || die "large-model work is Linux/VAST-only; refusing host $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
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
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
  gpu_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  [[ "$gpu_mem_mib" =~ ^[0-9]+$ ]] || die "could not read CUDA GPU memory"
  if (( gpu_mem_mib < MIN_GPU_MEM_MIB )); then
    die "GPU memory=${gpu_mem_mib} MiB is below the 20,000-MiB reference guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk grep find tee wc tr nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "parity uv.lock is missing"
  [[ -f "$AUDITOR" ]] || die "v2 manifest auditor is missing"
  [[ -f "$PREPARER" ]] || die "v2 checkpoint preparer is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
}

download_snapshot() {
  local output="$1"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["LICENSE", "config.json", "configuration_moss_audio_tokenizer.py", "modeling_moss_audio_tokenizer.py", "model.safetensors.index.json", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"])' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output"
}

verify_audit_json() {
  local path="$1" require_gguf="$2"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import json,pathlib,sys
data=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected=sys.argv[2]; require_gguf=sys.argv[3] == "1"
expected_shards={
    "model-00001-of-00003.safetensors": sys.argv[4],
    "model-00002-of-00003.safetensors": sys.argv[5],
    "model-00003-of-00003.safetensors": sys.argv[6],
}
assert data["revision"] == "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
assert data["tensor_count"] == 2094
assert data["parameter_count"] == 2123701248
assert data["tensor_bytes_f32"] == 8494804992
assert data["manifest_sha256_candidate"] == expected
assert data["manifest_sha256"] == expected
assert data["requires_vast_header_confirmation"] is False
assert len(data["shards"]) == 3
for name, expected_hash in expected_shards.items():
    assert data["shards"][name]["sha256"] == expected_hash
if require_gguf:
    assert data["gguf"]["tensor_count"] == 2094
    assert data["gguf"]["manifest_sha256"] == expected
print(f"authenticated manifest: {expected} gguf={require_gguf}")' \
    "$path" "$CANDIDATE_MANIFEST_SHA256" "$require_gguf" \
    "$SHARD1_SHA256" "$SHARD2_SHA256" "$SHARD3_SHA256"
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
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform,torch,transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}")'
  } | tee "$output"
}

run_self_test() {
  local tmp payload actual cases=0 fail=0 script_path
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-moss-tokenizer-v2-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" \
    >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid size accepted"
    fail=1
  fi
  cases=$((cases + 1))
  if verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" \
    "$(printf '%064d' 0)" >/dev/null 2>&1; then
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
  script_path="${BASH_SOURCE[0]}"
  for required in "$UPSTREAM_REVISION" "$CANDIDATE_MANIFEST_SHA256" \
    "$SHARD1_SHA256" "$SHARD2_SHA256" "$SHARD3_SHA256" \
    "moss_audio_tokenizer_v2_manifest.py" \
    "moss_audio_tokenizer_prepare_checkpoint.py" \
    "moss_audio_tokenizer_dump_reference.py" \
    "--shard-dir" "--gguf" "--model moss-audio-tokenizer-v2" \
    "--variant v2" "--num-quantizers 12" "--frozen --python 3.12" \
    "measure_v2_real_cpu_and_optional_metal_against_official" \
    "numeric_bounds=UNSET" "aarch64-apple-darwin"; do
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
  if grep -En '(publish-one\.sh|upload\.sh|--push([[:space:]]|$))' "$script_path" >/dev/null; then
    log "self-test FAIL: external publication operation found"
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-moss-audio-tokenizer-v2-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir snapshot stage logs reference
  local merged gguf audit_before audit_after reference_csv run_log env_log summary_file
  local compile_log cpu_log cross_log
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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/moss-tokenizer-v2-validation/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  snapshot="$work_dir/upstream"
  stage="$work_dir/stage"
  logs="$work_dir/logs"
  reference="$work_dir/reference"
  merged="$stage/moss-audio-tokenizer-v2.safetensors"
  gguf="$stage/moss-audio-tokenizer-v2.gguf"
  audit_before="$logs/safetensors-audit.json"
  audit_after="$logs/gguf-audit.json"
  reference_csv="$reference/moss-audio-tokenizer-v2-reference.csv"
  mkdir -p "$snapshot" "$stage" "$logs" "$reference"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-moss-tokenizer-v2"
  export HF_HOME="$VOKRA_SCRATCH/hf-home-moss-tokenizer-v2"
  run_log="$logs/run.log"
  env_log="$logs/environment.txt"
  summary_file="$logs/summary.txt"
  compile_log="$logs/compile.log"
  cpu_log="$logs/cpu-measurement.log"
  cross_log="$logs/apple-metal-cross-check.log"
  exec > >(tee -a "$run_log") 2>&1
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync locked Python 3.12 parity environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download immutable official three-shard snapshot"
  download_snapshot "$snapshot"

  step "Authenticate exact files and real safetensors headers"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$AUDITOR" \
    --config "$snapshot/config.json" \
    --index "$snapshot/model.safetensors.index.json" \
    --shard-dir "$snapshot" | tee "$audit_before"
  verify_audit_json "$audit_before" 0

  step "Merge exact shards into one converter input"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PREPARER" \
    --hf-repo "$UPSTREAM_REPO" \
    --revision "$UPSTREAM_REVISION" \
    --local-dir "$snapshot" \
    --output "$merged"

  step "Build vokra-cli on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli

  step "Convert the authenticated v2 checkpoint"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model moss-audio-tokenizer-v2 \
    --input "$merged" \
    --output "$gguf"

  step "Authenticate converted GGUF metadata and complete header"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$AUDITOR" \
    --config "$snapshot/config.json" \
    --index "$snapshot/model.safetensors.index.json" \
    --shard-dir "$snapshot" \
    --gguf "$gguf" | tee "$audit_after"
  verify_audit_json "$audit_after" 1

  step "Record environment before official numerical output"
  record_environment "$env_log"

  step "Generate independent official CUDA reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_DUMPER" \
    --variant v2 \
    --frames 2 \
    --num-quantizers 12 \
    --device cuda \
    --output "$reference_csv"
  grep -F "source,v2,$UPSTREAM_REPO,$UPSTREAM_REVISION" "$reference_csv" >/dev/null \
    || die "reference lost its pinned official source"
  grep -F "contract,2,12,1024,48000,2,3840" "$reference_csv" >/dev/null \
    || die "reference lost its v2 decode contract"

  step "Compile the native mapped decoder test target on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --lib --no-run 2>&1 | tee "$compile_log"

  step "Measure native CPU decode against the independent official reference"
  VOKRA_MOSS_AUDIO_TOKENIZER_V2_GGUF="$gguf" \
  VOKRA_MOSS_AUDIO_TOKENIZER_V2_REFERENCE="$reference_csv" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --lib \
      moss_audio_tokenizer::full_decoder::tests::measure_v2_real_cpu_and_optional_metal_against_official \
      -- --ignored --exact --nocapture 2>&1 | tee "$cpu_log"
  grep -F "MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET" \
    "$cpu_log" >/dev/null || die "CPU measurement sentinel is missing"

  step "Cross-check the Apple Metal feature route"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$cross_log"

  step "Write evidence summary and checksums"
  {
    echo "execution_status=PASS"
    echo "scope=AUTHENTICATED_ARTIFACT_REFERENCE_AND_CPU_MEASUREMENT"
    echo "numeric_verdict=MEASURED_NOT_GATED"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "manifest_sha256=$CANDIDATE_MANIFEST_SHA256"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_sha256=$(sha256_file "$reference_csv")"
    echo "metal_runtime=NOT_RUN_LINUX_VAST"
    echo "metal_cross_compile=PASS"
    grep -F "MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu" "$cpu_log"
  } | tee "$summary_file"
  (
    cd "$work_dir"
    find logs reference -type f ! -name SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > logs/SHA256SUMS
  )
  trap - EXIT
  log "PASS: pull $logs and $reference, then destroy the VAST instance"
}

main "$@"
