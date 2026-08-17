#!/usr/bin/env bash
# Reproduce the real-weight SBV2 SDP-body parity measurement on VAST.
#
# This is an in-instance worker. Rent and provision the VAST instance first,
# then run this script from the checkout. It deliberately refuses macOS and
# any Linux host not marked by scripts/publish/vast-ai/provision.sh.
#
# The three upstream revisions are resolved once at the start and pinned for
# every download in the run. Real weights, converted GGUFs, and raw reference
# tensors remain gitignored and on the disposable VAST instance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
FIXTURE_DIR="$VOKRA_ROOT/tests/fixtures/sbv2"

SBV2_REPO="litagin/Style-Bert-VITS2-2.0-base-JP-Extra"
BERT_JA_REPO="ku-nlp/deberta-v2-large-japanese-char-wwm"
BERT_EN_REPO="microsoft/deberta-v3-large"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=50000000

log()  { printf '[sbv2-sdp-vast] %s\n' "$*" >&2; }
step() { printf '\n[sbv2-sdp-vast] ==== %s ====\n' "$*" >&2; }
die()  { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-sbv2-sdp-parity.sh [--work-dir <empty-dir>]
       run-sbv2-sdp-parity.sh --self-test

VAST-only real-weight gate for SBV2's deterministic SDP body. It:
  1. resolves and pins the three public Hugging Face source revisions;
  2. downloads/prepares them with the tools/parity uv lock;
  3. builds vokra-cli and converts all three GGUFs on VAST;
  4. verifies the GGUFs against the committed SHA-256 sidecars;
  5. generates the independent MIT VITS SDP-body reference fixture;
  6. records CPU/ISA/torch/Rust/git provenance and runs the ignored Rust test.

Options:
  --work-dir DIR  fresh run directory for checkpoints and logs
                  (default: $VOKRA_SCRATCH/sbv2-sdp-parity/<UTC timestamp>)
  --self-test     hermetic contract tests only; no network, models, or Cargo
  -h, --help      show this help

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh,
approximately 64 GB RAM, and at least 50 GB free disk. Public checkpoints do
not require HF_TOKEN. Never pass a token on argv.
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

expected_sidecar_hash() {
  local sidecar="$1"
  local expected
  expected="$(awk 'NF && $1 !~ /^#/ { print $1; exit }' "$sidecar")"
  if ! printf '%s\n' "$expected" | grep -Eq '^[0-9a-f]{64}$'; then
    die "$sidecar has no valid lowercase SHA-256 record"
  fi
  printf '%s\n' "$expected"
}

verify_sidecar() {
  local artifact="$1" sidecar="$2"
  local expected actual
  [[ -f "$artifact" ]] || die "missing generated artifact $artifact"
  [[ -f "$sidecar" ]] || die "missing committed sidecar $sidecar"
  expected="$(expected_sidecar_hash "$sidecar")"
  actual="$(sha256_file "$artifact")"
  if [[ "$actual" != "$expected" ]]; then
    die "SHA-256 mismatch for $(basename "$artifact"): got $actual, expected $expected"
    return 2
  fi
  log "SHA-256 OK: $(basename "$artifact") = $actual"
}

require_vast_marker() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh on VAST first"
}

require_vast_host() {
  local mem_kib free_kib
  require_vast_marker
  [[ "$(uname -s)" == "Linux" ]] \
    || die "actual parity is Linux/VAST-only; refusing host $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"

  mem_kib="$(awk '$1 == "MemTotal:" { print $2; exit }' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal from /proc/meminfo"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 64 GB class guard (${MIN_VAST_MEM_KIB} KiB)"
  fi

  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 { print $4 }')"
  [[ -n "$free_kib" ]] || die "could not read free disk for $VOKRA_SCRATCH"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 50 GB run guard (${MIN_FREE_DISK_KIB} KiB)"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "$PARITY_PROJECT/uv.lock is missing"
  [[ -f "$FIXTURE_DIR/sbv2-v2-multilingual-base.gguf.sha256" ]] \
    || die "SBV2 fixture sidecars are missing from $FIXTURE_DIR"
}

run_self_test() {
  local tmp payload sidecar actual cases fail script_path
  cases=0
  fail=0
  tmp="$(mktemp -d)"
  # Expand the validated mktemp path now; `tmp` is local and would be out of
  # scope by the time an EXIT trap runs under `set -u`.
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload.gguf"
  sidecar="$tmp/payload.gguf.sha256"
  printf 'vokra-sbv2-sdp-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"
  printf '%s  payload.gguf\n' "$actual" > "$sidecar"

  cases=$((cases + 1))
  verify_sidecar "$payload" "$sidecar" >/dev/null 2>&1 \
    || { log "self-test FAIL: valid sidecar rejected"; fail=1; }

  cases=$((cases + 1))
  printf '%064d  payload.gguf\n' 0 > "$sidecar"
  if verify_sidecar "$payload" "$sidecar" >/dev/null 2>&1; then
    log "self-test FAIL: bad sidecar accepted"
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
  for required in \
    "$SBV2_REPO" "$BERT_JA_REPO" "$BERT_EN_REPO" \
    "uv sync --locked" "--test sbv2_sdp_torch_parity" "--ignored --nocapture"
  do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found; all Python must run through uv"
    fail=1
  fi

  if [[ $fail -eq 0 ]]; then
    echo "run-sbv2-sdp-parity.sh self-test: OK ($cases cases)"
    rm -rf "$tmp"
    trap - EXIT
    return 0
  fi
  rm -rf "$tmp"
  trap - EXIT
  return 1
}

resolve_hf_revision() {
  local repo="$1"
  uv run --project "$PARITY_PROJECT" python -c \
    'import sys; from huggingface_hub import HfApi; info=HfApi().model_info(sys.argv[1]); assert info.sha; print(info.sha)' \
    "$repo"
}

download_ja_checkpoint() {
  local revision="$1" output_dir="$2"
  uv run --project "$PARITY_PROJECT" python -c \
    'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["*.safetensors", "*.json", "vocab.txt"])' \
    "$BERT_JA_REPO" "$revision" "$output_dir"
}

record_environment() {
  local env_log="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$(awk -F ': ' '$1 == "model name" { print $2; exit }' /proc/cpuinfo)"
    echo "cpu_flags=$(awk -F ': ' '$1 == "flags" { print $2; exit }' /proc/cpuinfo)"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" { print "mem_total_kib=" $2; exit }' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    fi
    uv run --project "$PARITY_PROJECT" python -c \
      'import platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$env_log"
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir logs_dir
  local checkpoints_dir sbv2_dir bert_ja_dir bert_en_dir
  local sbv2_gguf bert_ja_gguf bert_en_gguf
  local sbv2_revision bert_ja_revision bert_en_revision
  local sbv2_input bert_ja_input bert_en_input bert_en_tokenizer
  local run_log env_log parity_log summary_file fixture_metadata
  local target fixture hash status_rc

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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/sbv2-sdp-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  logs_dir="$work_dir/logs"
  checkpoints_dir="$work_dir/checkpoints"
  mkdir -p "$logs_dir" "$checkpoints_dir" "$FIXTURE_DIR"

  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  parity_log="$logs_dir/parity.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1

  status_rc=1
  # shellcheck disable=SC2154
  trap 'rc=$?; summary_path="${summary_file:-}"; if [[ -n "$summary_path" && ! -f "$summary_path" ]]; then printf "status=FAIL\nexit_code=%s\n" "$rc" > "$summary_path"; fi; exit "$rc"' EXIT

  sbv2_dir="$checkpoints_dir/sbv2-v2-base"
  bert_ja_dir="$checkpoints_dir/deberta-v2-ja"
  bert_en_dir="$checkpoints_dir/deberta-v3-en"
  sbv2_gguf="$FIXTURE_DIR/sbv2-v2-multilingual-base.gguf"
  bert_ja_gguf="$FIXTURE_DIR/deberta-v2-large-japanese-char-wwm.gguf"
  bert_en_gguf="$FIXTURE_DIR/deberta-v3-large.gguf"

  for target in \
    "$sbv2_gguf" "$bert_ja_gguf" "$bert_en_gguf" \
    "$FIXTURE_DIR/sdp_body_hidden_seed0_T50.f32.bin" \
    "$FIXTURE_DIR/sdp_body_g_seed0.f32.bin" \
    "$FIXTURE_DIR/sdp_body_seed0_T50.f32.bin" \
    "$FIXTURE_DIR/sdp_body_seed0_T50.json"
  do
    [[ ! -e "$target" ]] || die "generated target already exists: $target (use a fresh VAST checkout)"
  done

  step "Sync locked Python environment through uv"
  UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache" uv sync --locked --project "$PARITY_PROJECT"

  step "Resolve immutable upstream revisions"
  sbv2_revision="$(resolve_hf_revision "$SBV2_REPO")"
  bert_ja_revision="$(resolve_hf_revision "$BERT_JA_REPO")"
  bert_en_revision="$(resolve_hf_revision "$BERT_EN_REPO")"
  {
    printf '%s %s\n' "$SBV2_REPO" "$sbv2_revision"
    printf '%s %s\n' "$BERT_JA_REPO" "$bert_ja_revision"
    printf '%s %s\n' "$BERT_EN_REPO" "$bert_en_revision"
  } | tee "$logs_dir/source-revisions.txt"

  step "Download and prepare pinned checkpoints"
  uv run --project "$PARITY_PROJECT" python \
    "$PARITY_PROJECT/sbv2_prepare_checkpoint.py" \
    --hf-repo "$SBV2_REPO" --revision "$sbv2_revision" \
    --output-dir "$sbv2_dir" --clean-room-defaults
  download_ja_checkpoint "$bert_ja_revision" "$bert_ja_dir"
  uv run --project "$PARITY_PROJECT" python \
    "$PARITY_PROJECT/bin_to_safetensors.py" \
    --hf-repo "$BERT_EN_REPO" --revision "$bert_en_revision" \
    --output-dir "$bert_en_dir"

  sbv2_input="$(find "$sbv2_dir" -name 'G_*.safetensors' -type f | sort | head -n 1)"
  bert_ja_input="$(find "$bert_ja_dir" -name 'model.safetensors' -type f | sort | head -n 1)"
  bert_en_input="$(find "$bert_en_dir" -name 'model.safetensors' -type f | sort | head -n 1)"
  [[ -n "$sbv2_input" ]] || die "no SBV2 G_*.safetensors found"
  [[ -n "$bert_ja_input" ]] || die "no JA model.safetensors found"
  [[ -n "$bert_en_input" ]] || die "no EN model.safetensors found"
  [[ -f "$bert_ja_dir/vocab.txt" ]] || die "JA vocab.txt missing"
  [[ -f "$bert_en_dir/spm.model" ]] || die "EN spm.model missing"

  step "Build vokra-cli on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --release -p vokra-cli

  step "Convert all three GGUFs on VAST"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model sbv2 --input "$sbv2_input" \
    --config "$sbv2_dir/vokra-sbv2-config.json" --output "$sbv2_gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model deberta-v2 --input "$bert_ja_input" \
    --tokenizer "$bert_ja_dir/vocab.txt" --output "$bert_ja_gguf"
  bert_en_tokenizer="$bert_en_dir/tokenizer_spm.json"
  uv run --project "$PARITY_PROJECT" python \
    "$PARITY_PROJECT/extract_spm_metadata.py" \
    --input "$bert_en_dir/spm.model" --output "$bert_en_tokenizer"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model deberta-v3 --input "$bert_en_input" \
    --tokenizer "$bert_en_tokenizer" --output "$bert_en_gguf"

  step "Verify committed GGUF sidecar hashes"
  verify_sidecar "$sbv2_gguf" "$FIXTURE_DIR/sbv2-v2-multilingual-base.gguf.sha256"
  verify_sidecar "$bert_ja_gguf" "$FIXTURE_DIR/deberta-v2-large-japanese-char-wwm.gguf.sha256"
  verify_sidecar "$bert_en_gguf" "$FIXTURE_DIR/deberta-v3-large.gguf.sha256"

  step "Generate independent upstream SDP-body fixture"
  uv run --project "$PARITY_PROJECT" python \
    "$PARITY_PROJECT/sbv2_sdp_body_dump.py" \
    --checkpoint "$sbv2_dir" --output-dir "$FIXTURE_DIR" --seed 0 --T 50

  step "Record execution environment before numerical result"
  record_environment "$env_log"

  step "Run the explicit VAST-only parity test"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" \
    -p vokra-models --test sbv2_sdp_torch_parity -- --ignored --nocapture \
    2>&1 | tee "$parity_log"

  fixture_metadata="$FIXTURE_DIR/sdp_body_seed0_T50.json"
  {
    echo "status=PASS"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "work_dir=$work_dir"
    echo "sbv2_revision=$sbv2_revision"
    echo "bert_ja_revision=$bert_ja_revision"
    echo "bert_en_revision=$bert_en_revision"
    for fixture in \
      "$sbv2_gguf" "$bert_ja_gguf" "$bert_en_gguf" \
      "$FIXTURE_DIR/sdp_body_hidden_seed0_T50.f32.bin" \
      "$FIXTURE_DIR/sdp_body_g_seed0.f32.bin" \
      "$FIXTURE_DIR/sdp_body_seed0_T50.f32.bin" "$fixture_metadata"
    do
      hash="$(sha256_file "$fixture")"
      printf 'sha256 %s %s\n' "$hash" "$(basename "$fixture")"
    done
    grep -E 'SDP body max \|Δ\|' "$parity_log" || true
  } | tee "$summary_file"
  status_rc=0
  trap - EXIT

  step "PASS"
  log "summary: $summary_file"
  log "logs:    $logs_dir"
  log "real model artifacts remain only on this disposable VAST instance"
  return "$status_rc"
}

main "$@"
