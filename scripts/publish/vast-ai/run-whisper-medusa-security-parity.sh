#!/usr/bin/env bash
# Reproduce the real-weight Whisper-Medusa module-0 parity oracle on VAST.
# This worker downloads public inputs and converts locally on VAST only.  It
# never uploads, publishes, or pushes any artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/whisper_medusa"
REFERENCE_SCRIPT="$VOKRA_ROOT/tools/parity/whisper_medusa/dump_reference.py"
PREPARE_SCRIPT="$VOKRA_ROOT/tools/parity/whisper_medusa_prepare_checkpoint.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

HF_REPO="aiola/whisper-medusa-v1"
HF_REVISION="6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462"
SOURCE_REPO="https://github.com/aiola-lab/whisper-medusa"
SOURCE_REVISION="19819c37ab15db6e68826e406614a2c86fbb946e"
CONFIG_SHA256="16346762b14c116eeda12b48f20e2281b327a11b516f8b004ce065fcb1450186"
INDEX_SHA256="0b80666c06d5054aa425a07d9f2f4ecabf9e6d7b8333f0dc5d85d4f79c9ff449"
SHARD_1_SHA256="b09e03326f4a9e3cd9bac17a55e17c60a3463e720a1cf0a51b8ba246a2b70b67"
SHARD_2_SHA256="6c496a29e2d131f999bbec815e4bd7a38b2ca436ce0d902237fdbd2971b35b74"
SHARD_1_BYTES=4992720328
SHARD_2_BYTES=1252815184
EXPECTED_SHARD_TOTAL_BYTES=6245535512
EXPECTED_TENSORS=1281
EXPECTED_MEDUSA_TENSORS=22
EXPECTED_TRANSFORMERS_VERSION="5.10.4"
EXPECTED_TORCH_VERSION="2.13.0"
LOGITS_ATOL="5e-4"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=120000000

# EXIT traps run after main's local scope has unwound.  Keep tee cleanup state
# global so a failed conversion cannot leave a FIFO reader or a paid worker.
run_log_fifo=""
tee_pid=""
finalization_complete=0
summary_file=""

log() { printf '[whisper-medusa-vast] %s\n' "$*" >&2; }
step() { printf '\n[whisper-medusa-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-whisper-medusa-security-parity.sh [--work-dir <empty-dir>]
       run-whisper-medusa-security-parity.sh --self-test

VAST-only real-weight security/parity worker.  It downloads the immutable
aiola/whisper-medusa-v1 snapshot and pinned official source, verifies the two
6.25-GB shards, merges and converts on VAST, generates an independent official
Transformers 5.10.4 oracle, runs native CPU parity, and cross-checks the
aarch64-apple-darwin Metal feature route.  Upload and publish are disabled.
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
  if [[ ! -f "$path" ]]; then
    die "missing pinned input: $path"
    return 2
  fi
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

verify_hash_only() {
  local path="$1" expected_hash="$2" actual_hash
  if [[ ! -f "$path" ]]; then
    die "missing pinned input: $path"
    return 2
  fi
  actual_hash="$(sha256_file "$path")"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"
    return 2
  fi
  log "identity OK: $path sha256=$actual_hash"
}

stop_log_tee() {
  local tee_status=0
  if [[ -n "${tee_pid:-}" ]]; then
    exec 1>&3 2>&4
    wait "$tee_pid" || tee_status=$?
    tee_pid=""
  fi
  if [[ -n "${run_log_fifo:-}" ]]; then
    rm -f "$run_log_fifo"
    run_log_fifo=""
  fi
  exec 3>&- 4>&-
  return "$tee_status"
}

cleanup_log_tee() {
  if [[ -n "${tee_pid:-}" ]]; then
    # Close the writer, let tee observe EOF, and drain the failure tail.
    exec 1>&3 2>&4
    wait "$tee_pid" 2>/dev/null || true
    tee_pid=""
  fi
  if [[ -n "${run_log_fifo:-}" ]]; then
    rm -f "$run_log_fifo"
    run_log_fifo=""
  fi
  exec 3>&- 4>&-
}

download_snapshot() {
  local output="$1"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))' \
    "$HF_REPO" "$HF_REVISION" "$output"
}

verify_snapshot() {
  local source_dir="$1" shard_1="$source_dir/model-00001-of-00002.safetensors" \
    shard_2="$source_dir/model-00002-of-00002.safetensors" total
  verify_hash_only "$source_dir/config.json" "$CONFIG_SHA256"
  verify_hash_only "$source_dir/model.safetensors.index.json" "$INDEX_SHA256"
  verify_file "$shard_1" "$SHARD_1_BYTES" "$SHARD_1_SHA256"
  verify_file "$shard_2" "$SHARD_2_BYTES" "$SHARD_2_SHA256"
  total=$(( $(wc -c < "$shard_1") + $(wc -c < "$shard_2") ))
  [[ "$total" == "$EXPECTED_SHARD_TOTAL_BYTES" ]] \
    || die "shard total bytes=$total, expected $EXPECTED_SHARD_TOTAL_BYTES"
  log "pinned shard total OK: bytes=$total tensors=$EXPECTED_TENSORS medusa_tensors=$EXPECTED_MEDUSA_TENSORS"
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
      'import platform,sys,torch,transformers; expected_transformers=sys.argv[1]; expected_torch=sys.argv[2]; actual_transformers=transformers.__version__; actual_torch=torch.__version__; assert actual_transformers == expected_transformers, (actual_transformers, expected_transformers); assert actual_torch == expected_torch, (actual_torch, expected_torch); print(f"python={platform.python_version()}"); print(f"transformers={actual_transformers}"); print(f"torch={actual_torch}")' \
      "$EXPECTED_TRANSFORMERS_VERSION" "$EXPECTED_TORCH_VERSION"
  } | tee "$output"
}

verify_reference_manifest() {
  local reference="$1" source_parent="$2"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); source_parent=pathlib.Path(sys.argv[2]); expected_transformers=sys.argv[3]
m=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
assert m["hf_repo"] == "aiola/whisper-medusa-v1"
assert m["hf_revision"] == "6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462"
assert m["source_revision"] == "19819c37ab15db6e68826e406614a2c86fbb946e"
assert m["reference_import"] == "whisper_medusa.models.WhisperMedusaModel"
assert m["transformers"] == expected_transformers
assert m["transformers_expected"] == expected_transformers
assert abs(float(m["max_abs_bound"]) - 5e-4) < 1e-12
assert m["device"] == "cpu"
assert (source_parent/m["model_source"]).is_file()
assert (source_parent/m["config_source"]).is_file()
for name in ("manifest.json","pcm.f32","prefix_logits.f32","greedy_tokens.u32"):
 assert (root/name).is_file(), name
print(f"reference manifest OK: transformers={expected_transformers}")' \
    "$reference" "$source_parent" "$EXPECTED_TRANSFORMERS_VERSION"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "${VOKRA_NO_UPLOAD:-1}" == "1" ]] \
    || die "VOKRA_NO_UPLOAD must remain 1; refusing an upload-capable run"
  [[ "$(uname -s)" == "Linux" ]] || die "model work is VAST/Linux-only"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ -n "$mem_kib" && "$mem_kib" -ge "$MIN_VAST_MEM_KIB" ]] \
    || die "MemTotal=${mem_kib:-unknown} KiB is below the 64-GB guard"
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" && "$free_kib" -ge "$MIN_FREE_DISK_KIB" ]] \
    || die "free disk=${free_kib:-unknown} KiB is below the 120-GB guard"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk grep find mkfifo tee wc tr xargs; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Whisper-Medusa uv.lock is missing"
  [[ -f "$REFERENCE_SCRIPT" && -f "$PREPARE_SCRIPT" ]] \
    || die "official reference/prepare scripts are missing"
  [[ -z "${HF_PUSH:-}" && -z "${HF_UPLOAD:-}" ]] \
    || die "upload environment variables are forbidden"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-whisper-medusa-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"
  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" >/dev/null 2>&1 \
    || { log "self-test FAIL: valid identity rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid byte size accepted"; fail=1
  fi
  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in "$HF_REVISION" "$SOURCE_REVISION" "$SHARD_1_SHA256" "$SHARD_2_SHA256" \
    "EXPECTED_TRANSFORMERS_VERSION=\"5.10.4\"" "EXPECTED_TORCH_VERSION=\"2.13.0\"" "$EXPECTED_SHARD_TOTAL_BYTES" \
    "whisper_medusa_prepare_checkpoint.py" "parity_whisper_medusa_real" \
    "aarch64-apple-darwin" "VOKRA_NO_UPLOAD" "SHA256SUMS" \
    "--frozen --python 3.12" "--config" "$SHARD_1_BYTES" "$SHARD_2_BYTES" \
    "no_upload=ENFORCED" "upload=NOT_PERFORMED" "publication=NO_UPLOAD"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"; fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"; fail=1
  fi
  cases=$((cases + 1))
  for forbidden in "publish-one."sh "upload."sh "--"push "hf_"hub_upload "upload_"file; do
    if grep -Fq -- "$forbidden" "$script_path"; then
      log "self-test FAIL: forbidden upload token found: $forbidden"; fail=1
    fi
  done
  cases=$((cases + 1))
  if ! (
    finalization_dir="$tmp/finalization"
    finalization_log="$finalization_dir/run.log"
    mkdir -p "$finalization_dir"
    run_log_fifo="$finalization_dir/.run.log.pipe"
    exec 3>&1 4>&2
    mkfifo "$run_log_fifo"
    tee -a "$finalization_log" < "$run_log_fifo" >/dev/null &
    tee_pid=$!
    exec > "$run_log_fifo" 2>&1
    printf 'whisper-medusa-finalization-marker\n'
    stop_log_tee
    (
      cd "$finalization_dir"
      printf '%s  run.log\n' "$(sha256_file run.log)" > SHA256SUMS
      if command -v sha256sum >/dev/null 2>&1; then sha256sum -c SHA256SUMS >/dev/null; else shasum -a 256 -c SHA256SUMS >/dev/null; fi
    )
    grep -Fxq 'whisper-medusa-finalization-marker' "$finalization_log"
  ); then
    log "self-test FAIL: evidence finalization did not preserve/hash run.log"; fail=1
  fi
  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-whisper-medusa-security-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir
  local upstream_dir source_parent source_checkout merged gguf reference
  local run_log env_log source_log merge_log convert_log cpu_log metal_log summary_file
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir) [[ $# -ge 2 && -n "$2" ]] || { die "--work-dir requires a directory"; return 2; }; requested_work_dir="$2"; shift 2 ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then run_self_test; return $?; fi

  require_vast_host
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/whisper-medusa-security-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  upstream_dir="$inputs_dir/hf-snapshot"
  source_parent="$inputs_dir/source-parent"
  source_checkout="$source_parent/whisper_medusa"
  merged="$work_dir/whisper-medusa-v1-merged.safetensors"
  gguf="$work_dir/whisper-medusa-v1.gguf"
  reference="$work_dir/reference"
  mkdir -p "$logs_dir" "$upstream_dir" "$source_parent"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-whisper-medusa"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  source_log="$logs_dir/source.txt"
  merge_log="$logs_dir/merge.log"
  convert_log="$logs_dir/convert.log"
  cpu_log="$logs_dir/cpu-parity.log"
  metal_log="$logs_dir/apple-metal-cross-check.log"
  summary_file="$logs_dir/summary.txt"
  exec 3>&1 4>&2
  run_log_fifo="$logs_dir/.run.log.pipe"
  mkfifo "$run_log_fifo"
  tee -a "$run_log" < "$run_log_fifo" &
  tee_pid=$!
  exec > "$run_log_fifo" 2>&1
  trap 'rc=$?; if [[ "${finalization_complete:-0}" != "1" && -n "${summary_file:-}" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; cleanup_log_tee; exit "$rc"' EXIT

  step "Sync locked Python 3.12 Transformers 5.10.4 oracle"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download and verify exact 6.25-GB HF snapshot"
  download_snapshot "$upstream_dir"
  verify_snapshot "$upstream_dir"

  step "Clone and verify pinned official source"
  git clone --quiet "$SOURCE_REPO" "$source_checkout"
  git -C "$source_checkout" checkout --quiet --detach "$SOURCE_REVISION"
  [[ "$(git -C "$source_checkout" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
    || die "official source revision verification failed"
  printf 'source_repo=%s\nsource_revision=%s\n' "$SOURCE_REPO" "$SOURCE_REVISION" | tee "$source_log"

  step "Record VAST environment"
  record_environment "$env_log"

  step "Merge pinned shards without publishing"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PREPARE_SCRIPT" --source-dir "$upstream_dir" --output "$merged" 2>&1 | tee "$merge_log"
  [[ -s "$merged" && -s "$merged.sha256" ]] || die "merged checkpoint evidence is missing"
  merged_sha256="$(sha256_file "$merged")"
  [[ "$(awk '{print $1}' "$merged.sha256")" == "$merged_sha256" ]] \
    || die "merged checkpoint sidecar hash mismatch"

  step "Build and convert the exact merged checkpoint"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli 2>&1 | tee "$convert_log"
  "$VOKRA_ROOT/target/release/vokra-cli" convert --model whisper-medusa-v1 \
    --input "$merged" --config "$upstream_dir/config.json" --output "$gguf" 2>&1 | tee -a "$convert_log"
  [[ -s "$gguf" ]] || die "conversion did not produce GGUF"
  gguf_bytes="$(wc -c < "$gguf" | tr -d '[:space:]')"
  gguf_sha256="$(sha256_file "$gguf")"
  log "converted GGUF identity: bytes=$gguf_bytes sha256=$gguf_sha256"

  step "Generate independent official Transformers reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_SCRIPT" --model-dir "$upstream_dir" --source-parent "$source_parent" \
    --output-dir "$reference" --max-new-tokens 8 --device cpu 2>&1 | tee "$cpu_log"
  verify_reference_manifest "$reference" "$source_parent"
  cp "$reference/manifest.json" "$logs_dir/reference-manifest.json"

  step "Run real-weight native CPU parity"
  VOKRA_WHISPER_MEDUSA_GGUF="$gguf" \
  VOKRA_WHISPER_MEDUSA_REFERENCE="$reference" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_whisper_medusa_real \
      official_module_zero_logits_and_greedy_tokens -- --nocapture 2>&1 | tee -a "$cpu_log"
  grep -F "Whisper-Medusa module-0 logits max_abs=" "$cpu_log" >/dev/null \
    || die "Whisper-Medusa CPU parity evidence line is missing"

  step "Cross-check Apple Metal feature route"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin 2>&1 | tee "$metal_log"

  step "Finalize NO_UPLOAD evidence and checksums"
  {
    echo "execution_status=PASS"
    echo "no_upload=ENFORCED"
    echo "upload=NOT_PERFORMED"
    echo "publication=NO_UPLOAD"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "hf_repo=$HF_REPO"
    echo "hf_revision=$HF_REVISION"
    echo "source_repo=$SOURCE_REPO"
    echo "source_revision=$SOURCE_REVISION"
    echo "transformers_version=$EXPECTED_TRANSFORMERS_VERSION"
    echo "shard_total_bytes=$EXPECTED_SHARD_TOTAL_BYTES"
    echo "merged_safetensors_sha256=$merged_sha256"
    echo "gguf_bytes=$gguf_bytes"
    echo "gguf_sha256=$gguf_sha256"
    echo "logits_atol=$LOGITS_ATOL"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    echo "metal_feature_cross_compile=PASS"
    grep -F "Whisper-Medusa module-0 logits max_abs=" "$cpu_log"
  } | tee "$summary_file"
  log "PASS evidence: $logs_dir and $reference"
  stop_log_tee
  (
    cd "$work_dir"
    find logs reference -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > logs/SHA256SUMS
  )
  finalization_complete=1
}

main "$@"
