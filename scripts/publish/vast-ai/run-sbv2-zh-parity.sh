#!/usr/bin/env bash
# Reproduce the four-file SBV2 ZH real-checkpoint parity leg on VAST.
#
# This worker downloads public checkpoints, performs every conversion and
# vokra-models Cargo invocation on the disposable VAST host, and never uploads
# an artefact. It intentionally refuses unmarked hosts and undersized boxes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/sbv2"
FIXTURE_DIR="$VOKRA_ROOT/tests/fixtures/sbv2"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

SBV2_REPO="litagin/Style-Bert-VITS2-2.0-base-JP-Extra"
SBV2_REVISION="a731761009f3c96d104487be6ad332bf1bb5a3a5"
BERT_JA_REPO="ku-nlp/deberta-v2-large-japanese-char-wwm"
BERT_JA_REVISION="547b0e8b044fba3f9b84d0ab9f990440bd130c8b"
BERT_EN_REPO="microsoft/deberta-v3-large"
BERT_EN_REVISION="64a8c8eab3e352a784c658aef62be1662607476f"
BERT_ZH_REPO="hfl/chinese-roberta-wwm-ext-large"
BERT_ZH_REVISION="a25cc9e05974bd9687e528edd516f2cfdb3f5db9"

MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=50000000

log()  { printf '[sbv2-zh-vast] %s\n' "$*" >&2; }
step() { printf '\n[sbv2-zh-vast] ==== %s ====\n' "$*" >&2; }
die()  { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-sbv2-zh-parity.sh [--work-dir <empty-dir>]
       run-sbv2-zh-parity.sh --self-test

VAST-only real-weight gate for SBV2's four-file Chinese path. It:
  1. downloads four public checkpoints at immutable revisions;
  2. converts the SBV2 main, JA/EN DeBERTa, and ZH plain-BERT files to GGUF;
  3. checks every GGUF against its committed SHA-256 sidecar;
  4. generates a real transformers ZH BERT + clean-room VITS reference dump;
  5. runs the named non-ignored vokra-models parity consumer.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1, approximately 64 GB RAM,
and at least 50 GB free disk. The checkpoints are public; no HF token is
needed. This script does not publish or upload anything.
EOF
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

expected_sidecar_hash() {
  local expected
  expected="$(awk 'NF && $1 !~ /^#/ { print $1; exit }' "$1")"
  if ! [[ "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    die "$1 has no valid lowercase SHA-256 record"
    return 2
  fi
  printf '%s\n' "$expected"
}

verify_sidecar() {
  local artifact="$1" sidecar="$2" expected actual
  [[ -f "$artifact" ]] || { die "missing generated artifact $artifact"; return 2; }
  [[ -f "$sidecar" ]] || { die "missing committed sidecar $sidecar"; return 2; }
  expected="$(expected_sidecar_hash "$sidecar")"
  actual="$(sha256_file "$artifact")"
  if [[ "$actual" != "$expected" ]]; then
    die "SHA-256 mismatch for $(basename "$artifact"): got $actual, expected $expected"
    return 2
  fi
  log "SHA-256 OK: $(basename "$artifact") = $actual"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || { die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh on VAST first"; return 2; }
  [[ "$(uname -s)" == "Linux" ]] \
    || { die "actual parity is Linux/VAST-only; refusing host $(uname -s)"; return 2; }
  mem_kib="$(awk '$1 == "MemTotal:" { print $2; exit }' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || { die "could not read MemTotal"; return 2; }
  (( mem_kib >= MIN_VAST_MEM_KIB )) \
    || { die "MemTotal=${mem_kib} KiB is below the VAST 64 GB class guard"; return 2; }
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 { print $4 }')"
  [[ -n "$free_kib" ]] || { die "could not read free disk"; return 2; }
  (( free_kib >= MIN_FREE_DISK_KIB )) \
    || { die "free disk=${free_kib} KiB is below the 50 GB run guard"; return 2; }
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk find grep sha256sum tee; do
    command -v "$tool" >/dev/null 2>&1 \
      || { die "required tool missing: $tool"; return 2; }
  done
  [[ -d "$VOKRA_ROOT/.git" ]] \
    || { die "$VOKRA_ROOT is not a git checkout"; return 2; }
  [[ -f "$PARITY_PROJECT/uv.lock" ]] \
    || { die "$PARITY_PROJECT/uv.lock is missing"; return 2; }
}

run_self_test() {
  local tmp payload sidecar hash token cases=0 fail=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload.gguf"
  sidecar="$tmp/payload.gguf.sha256"
  printf 'vokra-sbv2-zh-self-test\n' > "$payload"
  hash="$(sha256_file "$payload")"
  printf '%s  payload.gguf\n' "$hash" > "$sidecar"

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
  for token in \
    "$SBV2_REVISION" "$BERT_JA_REVISION" "$BERT_EN_REVISION" "$BERT_ZH_REVISION" \
    "uv sync --project" "--model bert-base" "--language zh" \
    "--test parity_sbv2_real" "--exact --nocapture"
  do
    if ! grep -Fq -- "$token" "${BASH_SOURCE[0]}"; then
      log "self-test FAIL: worker contract lost token: $token"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "${BASH_SOURCE[0]}" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-sbv2-zh-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    awk -F ':' '$1 ~ /^model name/ { sub(/^[[:space:]]+/, "", $2); print "cpu_model=" $2; exit }' /proc/cpuinfo
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" { print "mem_total_kib=" $2; exit }' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    fi
    uv run --project "$PARITY_PROJECT" --frozen python -c \
      'import platform, torch, transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
  } | tee "$output"
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir logs_dir checkpoints_dir
  local sbv2_dir bert_ja_dir bert_en_dir bert_zh_dir
  local sbv2_input bert_ja_input bert_en_input bert_en_tokenizer
  local sbv2_gguf bert_ja_gguf bert_en_gguf bert_zh_gguf
  local run_log dump_log parity_log env_log summary_file target hash

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
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/sbv2-zh-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  logs_dir="$work_dir/logs"
  checkpoints_dir="$work_dir/checkpoints"
  mkdir -p "$logs_dir" "$checkpoints_dir" "$FIXTURE_DIR"
  run_log="$logs_dir/run.log"
  dump_log="$logs_dir/dump.log"
  parity_log="$logs_dir/parity.log"
  env_log="$logs_dir/environment.txt"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154 # rc is assigned inside the trap body.
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "${summary_file:-}" ]]; then printf "status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  sbv2_dir="$checkpoints_dir/sbv2-main"
  bert_ja_dir="$checkpoints_dir/bert-ja"
  bert_en_dir="$checkpoints_dir/bert-en"
  bert_zh_dir="$checkpoints_dir/bert-zh"
  sbv2_gguf="$FIXTURE_DIR/sbv2-v2-multilingual-base.gguf"
  bert_ja_gguf="$FIXTURE_DIR/deberta-v2-large-japanese-char-wwm.gguf"
  bert_en_gguf="$FIXTURE_DIR/deberta-v3-large.gguf"
  bert_zh_gguf="$FIXTURE_DIR/chinese-roberta-wwm-ext-large.gguf"

  for target in "$sbv2_gguf" "$bert_ja_gguf" "$bert_en_gguf" "$bert_zh_gguf" \
    "$FIXTURE_DIR/reference_dump.manifest.json" "$FIXTURE_DIR/reference_dump"; do
    [[ ! -e "$target" ]] \
      || { die "generated target already exists: $target"; return 2; }
  done

  step "Sync the locked Python oracle through uv"
  UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache" \
    uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download four pinned public checkpoints"
  uv run --project "$PARITY_PROJECT" --frozen python \
    "$VOKRA_ROOT/tools/parity/sbv2_prepare_checkpoint.py" \
    --hf-repo "$SBV2_REPO" --revision "$SBV2_REVISION" \
    --output-dir "$sbv2_dir" --clean-room-defaults
  uv run --project "$PARITY_PROJECT" --frozen python -c \
    'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["*.safetensors", "*.json", "vocab.txt"])' \
    "$BERT_JA_REPO" "$BERT_JA_REVISION" "$bert_ja_dir"
  uv run --project "$PARITY_PROJECT" --frozen python \
    "$VOKRA_ROOT/tools/parity/bin_to_safetensors.py" \
    --hf-repo "$BERT_EN_REPO" --revision "$BERT_EN_REVISION" --output-dir "$bert_en_dir"
  uv run --project "$PARITY_PROJECT" --frozen python \
    "$VOKRA_ROOT/tools/parity/bin_to_safetensors.py" \
    --hf-repo "$BERT_ZH_REPO" --revision "$BERT_ZH_REVISION" --output-dir "$bert_zh_dir"

  sbv2_input="$(find "$sbv2_dir" -name 'G_*.safetensors' -type f | sort | head -n 1)"
  bert_ja_input="$(find "$bert_ja_dir" -name 'model.safetensors' -type f | sort | head -n 1)"
  bert_en_input="$(find "$bert_en_dir" -name 'model.safetensors' -type f | sort | head -n 1)"
  [[ -n "$sbv2_input" ]] || { die "no SBV2 G_*.safetensors found"; return 2; }
  [[ -n "$bert_ja_input" ]] || { die "no JA model.safetensors found"; return 2; }
  [[ -n "$bert_en_input" ]] || { die "no EN model.safetensors found"; return 2; }
  [[ -f "$bert_zh_dir/model.safetensors" ]] \
    || { die "no ZH model.safetensors found"; return 2; }
  [[ -f "$bert_ja_dir/vocab.txt" ]] || { die "JA vocab.txt missing"; return 2; }
  [[ -f "$bert_en_dir/spm.model" ]] || { die "EN spm.model missing"; return 2; }
  [[ -f "$bert_zh_dir/vocab.txt" ]] || { die "ZH vocab.txt missing"; return 2; }

  step "Build vokra-cli on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --release -p vokra-cli

  step "Convert the four-file bundle on VAST"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model sbv2 --input "$sbv2_input" \
    --config "$sbv2_dir/vokra-sbv2-config.json" --output "$sbv2_gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model deberta-v2 --input "$bert_ja_input" \
    --tokenizer "$bert_ja_dir/vocab.txt" --output "$bert_ja_gguf"
  bert_en_tokenizer="$bert_en_dir/tokenizer_spm.json"
  uv run --project "$PARITY_PROJECT" --frozen python \
    "$VOKRA_ROOT/tools/parity/extract_spm_metadata.py" \
    --input "$bert_en_dir/spm.model" --output "$bert_en_tokenizer"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model deberta-v3 --input "$bert_en_input" \
    --tokenizer "$bert_en_tokenizer" --output "$bert_en_gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model bert-base --input "$bert_zh_dir/model.safetensors" \
    --tokenizer "$bert_zh_dir/vocab.txt" --output "$bert_zh_gguf"

  step "Verify all four committed sidecar hashes"
  verify_sidecar "$sbv2_gguf" "$FIXTURE_DIR/sbv2-v2-multilingual-base.gguf.sha256"
  verify_sidecar "$bert_ja_gguf" "$FIXTURE_DIR/deberta-v2-large-japanese-char-wwm.gguf.sha256"
  verify_sidecar "$bert_en_gguf" "$FIXTURE_DIR/deberta-v3-large.gguf.sha256"
  verify_sidecar "$bert_zh_gguf" "$FIXTURE_DIR/chinese-roberta-wwm-ext-large.gguf.sha256"

  step "Generate independent ZH reference dump"
  uv run --project "$PARITY_PROJECT" --frozen python \
    "$VOKRA_ROOT/tools/parity/sbv2_dump_reference.py" \
    --checkpoint "$sbv2_dir" --output-dir "$FIXTURE_DIR" \
    --bert-ja-repo "$bert_ja_dir" --bert-en-repo "$bert_en_dir" \
    --bert-zh-repo "$bert_zh_dir" --language zh --do-dump \
    2>&1 | tee "$dump_log"

  step "Record execution environment"
  record_environment "$env_log"

  step "Run the named four-file Rust parity consumer"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" \
    -p vokra-models --test parity_sbv2_real \
    parity_sbv2_real_waveform_matches_reference_dump -- --exact --nocapture \
    2>&1 | tee "$parity_log"

  {
    echo "status=PASS"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "sbv2_revision=$SBV2_REVISION"
    echo "bert_ja_revision=$BERT_JA_REVISION"
    echo "bert_en_revision=$BERT_EN_REVISION"
    echo "bert_zh_revision=$BERT_ZH_REVISION"
    for target in "$sbv2_gguf" "$bert_ja_gguf" "$bert_en_gguf" "$bert_zh_gguf" \
      "$FIXTURE_DIR/reference_dump.manifest.json"; do
      hash="$(sha256_file "$target")"
      printf 'sha256 %s %s\n' "$hash" "$(basename "$target")"
    done
    grep -E 'PASS|FAIL|max \|Δ\||mel-loss|waveform' "$parity_log" | tail -n 80 || true
  } | tee "$summary_file"

  trap - EXIT
  step "PASS"
  log "summary: $summary_file"
}

main "$@"
