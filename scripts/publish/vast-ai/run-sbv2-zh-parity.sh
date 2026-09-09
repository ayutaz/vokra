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
FIXTURE_TEMPLATE_DIR="$VOKRA_ROOT/tests/fixtures/sbv2"
FIXTURE_DIR=""

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
usage: run-sbv2-zh-parity.sh --expected-head <40-lower-hex> --approval-evidence <json> [--language ja] [--work-dir <absent-dir>]
       run-sbv2-zh-parity.sh --self-test

VAST-only real-weight gate for SBV2's four-file JP-Extra Japanese bundle. A
fixture-G2P result is never production Japanese-G2P evidence.
  1. downloads four public checkpoints at immutable revisions;
  2. converts the SBV2 main, JA/EN DeBERTa, and ZH plain-BERT files to GGUF;
  3. checks every GGUF against its committed SHA-256 sidecar;
  4. generates a real transformers JA/ZH BERT + clean-room VITS reference dump;
  5. runs the named ignored vokra-models parity consumer exactly once.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1, approximately 64 GB RAM,
and at least 50 GB free disk. The checkpoints are public; no HF token is
needed. This script does not publish or upload anything.
EOF
}

require_expected_head() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || { die "expected HEAD must be lowercase 40-hex"; return 2; }
  local actual
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || { die "could not read checkout HEAD"; return 2; }
  [[ "$actual" == "$1" ]] || { die "checkout HEAD mismatch: got $actual, expected $1"; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || { die "VAST checkout must be clean before model work"; return 2; }
  log "verified clean expected HEAD: $actual"
}

require_license_approval() {
  [[ -f "$VOKRA_ROOT/docs/license-audit.md" && ! -L "$VOKRA_ROOT/docs/license-audit.md" ]] \
    || { die "license audit is missing or symlinked"; return 2; }
  uv run --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/scripts/publish/signoff_match.py" \
    --check-repo sbv2-v2-jp-extra-base --audit "$VOKRA_ROOT/docs/license-audit.md" \
    >/dev/null \
    || { die "SBV2 JP-Extra owner/license sign-off is not approved"; return 2; }
  log "verified SBV2 JP-Extra owner/license sign-off before acquisition"
}

reject_symlink_ancestors() {
  local path="$1" rest component current=/
  [[ "$path" == /* ]] || { die "path must be absolute: $path"; return 2; }
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "path contains symlink ancestry: $path"; return 2; }
    current="$current/"
  done
}

require_external_approval() {
  local approval="$1" expected_head="$2"
  [[ "$approval" == /* && -f "$approval" && ! -L "$approval" ]] \
    || { die "approval evidence must be an absolute regular non-symlink file"; return 2; }
  reject_symlink_ancestors "$approval"
  approval_sha="$(uv run --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/sbv2_approval_preflight.py" \
    --approval "$approval" --expected-head "$expected_head")" \
    || { die "external SBV2 owner approval is invalid"; return 2; }
  [[ "$approval_sha" =~ ^[0-9a-f]{64}$ ]] || { die "approval preflight returned malformed digest"; return 2; }
  log "verified external SBV2 owner approval: $approval_sha"
}

require_external_work_dir() {
  local path="$1" root parent current rest component
  [[ "$path" == /* ]] || { die "--work-dir must be absolute"; return 2; }
  [[ "$path" != */./* && "$path" != */../* && "$path" != */. && "$path" != */.. ]] \
    || { die "--work-dir may not contain dot components"; return 2; }
  rest="${path#/}"; current=/
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "--work-dir has a symlink ancestor"; return 2; }
    current="$current/"
  done
  parent="$path"
  while [[ ! -e "$parent" ]]; do
    parent="${parent%/*}"; [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || { die "--work-dir parent must be a real directory"; return 2; }
  root="$(cd -P "$VOKRA_ROOT" && pwd)" || { die "could not resolve checkout"; return 2; }
  case "$path" in "$root"|"$root"/*) die "--work-dir overlaps checkout"; return 2 ;; esac
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

hash_directory() {
  local directory="$1" output="$2" path
  find "$directory" -type f -print | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s  %s\n' "$(sha256_file "$path")" "${path#"$directory"/}"
  done > "$output"
}

require_cargo_singleton() {
  local log_file="$1" test_name="$2" named_count result_count all_test_lines
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_file" || true)"
  result_count="$(grep -Ec '^test result:' "$log_file" || true)"
  all_test_lines="$(grep -Ec '^test ' "$log_file" || true)"
  (( named_count == 1 && result_count == 1 && all_test_lines - result_count == 1 )) \
    || { die "Cargo log is not one exact passing SBV2 test"; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" \
    || { die "Cargo result is not an exact singleton pass"; return 2; }
}

require_sbv2_cpu_sentinel() {
  local log_file="$1"
  [[ "$(grep -Ec '^\[parity_sbv2_real\] waveform parity OK: rust=[0-9]+ samples ref=[0-9]+ samples \(ratio [0-9]+\.[0-9]{4}, band ±[0-9]+(\.[0-9]+)?%, overlap [0-9]+ samples: max \|Δ\| = [0-9]+\.[0-9]+e[+-][0-9]+, RMS \|Δ\| = [0-9]+\.[0-9]+e[+-][0-9]+ <= atol [0-9]+(\.[0-9]+)?\)$' "$log_file" || true)" == 1 ]] \
    || { die "SBV2 CPU/reference sentinel is not one complete line"; return 2; }
}

require_packet_closure() {
  local directory="$1" entry base top_count
  [[ -d "$directory" && ! -L "$directory" ]] || { die "fixture packet is missing or symlinked"; return 2; }
  uv run --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/tools/parity/sbv2_approval_preflight.py" --packet "$directory" \
    || { die "strict SBV2 reference packet validation failed"; return 2; }
  [[ -z "$(find "$directory" -type l -print -quit)" ]] || { die "fixture packet contains a symlink"; return 2; }
  top_count="$(find "$directory" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')"
  [[ "$top_count" == 10 ]] || { die "fixture packet top-level entry count is not exact: $top_count"; return 2; }
  while IFS= read -r entry; do
    base="${entry##*/}"
    case "$base" in
      reference_dump.manifest.json|reference_dump|sbv2-v2-jp-extra-base.gguf|sbv2-v2-jp-extra-base.gguf.sha256|deberta-v2-large-japanese-char-wwm.gguf|deberta-v2-large-japanese-char-wwm.gguf.sha256|deberta-v3-large.gguf|deberta-v3-large.gguf.sha256|chinese-roberta-wwm-ext-large.gguf|chinese-roberta-wwm-ext-large.gguf.sha256) ;;
      *) die "unexpected fixture packet entry: $entry"; return 2 ;;
    esac
  done < <(find "$directory" -mindepth 1 -maxdepth 1 -print)
  [[ -z "$(find "$directory/reference_dump" -mindepth 1 -type d -print -quit)" ]] || { die "reference_dump contains a directory"; return 2; }
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
  cargo deny --version >/dev/null 2>&1 || { die "cargo-deny is required before model work"; return 2; }
  cargo audit --version >/dev/null 2>&1 || { die "cargo-audit is required before model work"; return 2; }
}

run_self_test() {
  local tmp payload sidecar hash token cases=0 fail=0
  [[ "$BERT_ZH_REVISION" == "a25cc9e05974bd9687e528edd516f2cfdb3f5db9" ]] \
    || { log "self-test FAIL: authenticated ZH revision drifted"; return 1; }
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
  local synthetic_log="$tmp/parity.log"
  printf 'test parity_sbv2_real_waveform_matches_reference_dump ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n[parity_sbv2_real] waveform parity OK: rust=10 samples ref=10 samples (ratio 1.0000, band ±10.0000%%, overlap 10 samples: max |Δ| = 1.000000e-04, RMS |Δ| = 1.000000e-05 <= atol 0.01)\n' > "$synthetic_log"
  require_cargo_singleton "$synthetic_log" parity_sbv2_real_waveform_matches_reference_dump
  require_sbv2_cpu_sentinel "$synthetic_log"

  cases=$((cases + 1))
  for token in \
    "$SBV2_REVISION" "$BERT_JA_REVISION" "$BERT_EN_REVISION" "$BERT_ZH_REVISION" \
    "uv sync --project" "--model bert-base" "--language" "language_set" \
    "--test parity_sbv2_real" "--ignored --exact --test-threads=1 --nocapture" \
    "--expected-head" "require_license_approval" "cargo deny --locked --offline check" \
    "cargo audit --no-fetch" "check-forbidden-symbols.sh" "check-zero-deps.sh" "check-bound-arch-coverage.sh" \
    "cargo clippy --workspace --all-targets --all-features --locked --offline" \
    "cargo test --workspace --locked --offline --all-targets" "CPU_PASS_METAL_NOT_RUN" "require_packet_closure" \
    "sbv2_approval_preflight.py" "--approval-evidence" "scripts/verify/apple-silicon-sbv2.sh" \
    "--packet-sha256" "apple-transfer-args.txt" "--packet"
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

  if "${BASH_SOURCE[0]}" --self-test --self-test >/dev/null 2>&1; then
    log "self-test FAIL: duplicate --self-test accepted"
    fail=1
  fi
  if "${BASH_SOURCE[0]}" --expected-head 0000000000000000000000000000000000000000 \
    --expected-head 0000000000000000000000000000000000000000 >/dev/null 2>&1; then
    log "self-test FAIL: duplicate --expected-head accepted"
    fail=1
  fi
  if "${BASH_SOURCE[0]}" --approval-evidence /tmp/a --approval-evidence /tmp/b \
    --expected-head 0000000000000000000000000000000000000000 >/dev/null 2>&1; then
    log "self-test FAIL: duplicate --approval-evidence accepted"
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
  local self_test=0 requested_work_dir="" expected_head="" approval_evidence="" language=ja run_stamp work_dir logs_dir checkpoints_dir approval_sha=""
  local sbv2_dir bert_ja_dir bert_en_dir bert_zh_dir
  local sbv2_input bert_ja_input bert_en_input bert_en_tokenizer
  local sbv2_gguf bert_ja_gguf bert_en_gguf bert_zh_gguf
  local run_log dump_log parity_log env_log summary_file packet_hash_file transfer_args target hash gguf_sha reference_manifest_sha packet_sha
  local workspace_log

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" ]] || { die "--work-dir requires a directory"; return 2; }
        [[ -z "$requested_work_dir" ]] || { die "duplicate --work-dir"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --expected-head)
        [[ -z "$expected_head" ]] || { die "duplicate --expected-head"; return 2; }
        [[ $# -ge 2 ]] || { die "--expected-head requires a value"; return 2; }
        [[ "$2" =~ ^[0-9a-f]{40}$ ]] || { die "expected HEAD must be lowercase 40-hex"; return 2; }
        expected_head="$2"
        shift 2
        ;;
      --language)
        [[ -z "${language_set:-}" ]] || { die "duplicate --language"; return 2; }
        [[ $# -ge 2 && "$2" == ja ]] || { die "--language must be ja"; return 2; }
        language_set=1
        language="$2"
        shift 2
        ;;
      --approval-evidence)
        [[ -z "$approval_evidence" ]] || { die "duplicate --approval-evidence"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence requires a path"; return 2; }
        approval_evidence="$2"
        shift 2
        ;;
      --self-test) (( self_test == 0 )) || { die "duplicate --self-test"; return 2; }; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    [[ -z "$expected_head$requested_work_dir$approval_evidence${language_set:-}" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi

  [[ -n "$expected_head$approval_evidence" ]] || { usage; die "--expected-head and --approval-evidence are required"; return 2; }
  [[ "$language" == ja ]] || { die "production SBV2 JP-Extra worker only accepts --language ja"; return 2; }
  require_expected_head "$expected_head"
  require_license_approval
  require_external_approval "$approval_evidence" "$expected_head"
  require_vast_host
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/sbv2-zh-parity/$run_stamp}"
  if [[ -e "$work_dir" || -L "$work_dir" ]]; then
    die "--work-dir must be absent and non-symlinked: $work_dir"
  fi
  require_external_work_dir "$work_dir"
  logs_dir="$work_dir/logs"
  checkpoints_dir="$work_dir/checkpoints"
  FIXTURE_DIR="$work_dir/fixture-packet"
  mkdir -m 700 "$work_dir"
  mkdir -m 700 "$logs_dir" "$checkpoints_dir" "$FIXTURE_DIR"
  # The checkout's sidecars are read-only authenticated inputs. Copy only
  # those four expected records into the disposable packet; no generated
  # output is ever written under the checkout.
  for target in \
    sbv2-v2-jp-extra-base.gguf.sha256 \
    deberta-v2-large-japanese-char-wwm.gguf.sha256 \
    deberta-v3-large.gguf.sha256 \
    chinese-roberta-wwm-ext-large.gguf.sha256; do
    cp -p "$FIXTURE_TEMPLATE_DIR/$target" "$FIXTURE_DIR/$target"
  done
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
  sbv2_gguf="$FIXTURE_DIR/sbv2-v2-jp-extra-base.gguf"
  bert_ja_gguf="$FIXTURE_DIR/deberta-v2-large-japanese-char-wwm.gguf"
  bert_en_gguf="$FIXTURE_DIR/deberta-v3-large.gguf"
  bert_zh_gguf="$FIXTURE_DIR/chinese-roberta-wwm-ext-large.gguf"

  # reference_dump.manifest.json is a tracked schema template and is
  # intentionally replaced by the real dumper. Binary fixtures remain
  # gitignored and must be absent so a run cannot reuse stale evidence.
  for target in "$sbv2_gguf" "$bert_ja_gguf" "$bert_en_gguf" "$bert_zh_gguf" \
    "$FIXTURE_DIR/reference_dump"; do
    [[ ! -e "$target" ]] \
      || { die "generated target already exists: $target"; return 2; }
  done

  step "Run locked offline source gates before model acquisition"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
  cargo deny --locked --offline check
  cargo audit --no-fetch
  CARGO_NET_OFFLINE=true cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
  CARGO_NET_OFFLINE=true cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --offline --release -p vokra-cli

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
  verify_sidecar "$sbv2_gguf" "$FIXTURE_TEMPLATE_DIR/sbv2-v2-jp-extra-base.gguf.sha256"
  verify_sidecar "$bert_ja_gguf" "$FIXTURE_TEMPLATE_DIR/deberta-v2-large-japanese-char-wwm.gguf.sha256"
  verify_sidecar "$bert_en_gguf" "$FIXTURE_TEMPLATE_DIR/deberta-v3-large.gguf.sha256"
  verify_sidecar "$bert_zh_gguf" "$FIXTURE_TEMPLATE_DIR/chinese-roberta-wwm-ext-large.gguf.sha256"

  step "Generate independent Japanese reference dump"
  uv run --project "$PARITY_PROJECT" --frozen python \
    "$VOKRA_ROOT/tools/parity/sbv2_dump_reference.py" \
    --checkpoint "$sbv2_dir" --output-dir "$FIXTURE_DIR" \
    --bert-ja-repo "$bert_ja_dir" --bert-en-repo "$bert_en_dir" \
    --bert-zh-repo "$bert_zh_dir" --language "$language" --do-dump \
    2>&1 | tee "$dump_log"
  require_packet_closure "$FIXTURE_DIR"
  packet_hash_file="$logs_dir/packet-hashes.txt"
  hash_directory "$FIXTURE_DIR" "$packet_hash_file"
  packet_sha="$(sha256_file "$packet_hash_file")"
  gguf_sha="$(sha256_file "$sbv2_gguf")"
  reference_manifest_sha="$(sha256_file "$FIXTURE_DIR/reference_dump.manifest.json")"

  step "Record execution environment"
  record_environment "$env_log"

  step "Run the named four-file Rust parity consumer"
  VOKRA_SBV2_FIXTURE_DIR="$FIXTURE_DIR" VOKRA_SBV2_G2P_MODE=fixture-replay \
    CARGO_NET_OFFLINE=true cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" \
    --locked --offline \
    -p vokra-models --test parity_sbv2_real \
    parity_sbv2_real_waveform_matches_reference_dump -- --ignored --exact --test-threads=1 --nocapture \
    2>&1 | tee "$parity_log"
  require_cargo_singleton "$parity_log" parity_sbv2_real_waveform_matches_reference_dump
  require_sbv2_cpu_sentinel "$parity_log"

  step "Run the locked offline workspace regression after parity"
  workspace_log="$logs_dir/workspace.log"
  CARGO_NET_OFFLINE=true cargo test --workspace --locked --offline --all-targets -- --test-threads=1 \
    2>&1 | tee "$workspace_log"

  {
    echo "execution_status=PASS"
    echo "verdict=CPU_PASS_METAL_NOT_RUN"
    echo "cpu_vs_upstream=PASS"
    echo "metal_vs_upstream=NOT_RUN"
    echo "metal_vs_cpu=NOT_RUN"
    echo "workspace=PASS"
    echo "g2p=FIXTURE_REPLAY_ONLY"
    echo "production_japanese_g2p=UNRESOLVED"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "expected_head=$expected_head"
    echo "approval_evidence_sha256=$approval_sha"
    echo "packet_sha256=$packet_sha"
    echo "reference_manifest_sha256=$reference_manifest_sha"
    echo "sbv2_revision=$SBV2_REVISION"
    echo "bert_ja_revision=$BERT_JA_REVISION"
    echo "bert_en_revision=$BERT_EN_REVISION"
    echo "bert_zh_revision=$BERT_ZH_REVISION"
    echo "language=$language"
    for target in "$sbv2_gguf" "$bert_ja_gguf" "$bert_en_gguf" "$bert_zh_gguf" \
      "$FIXTURE_DIR/reference_dump.manifest.json"; do
      hash="$(sha256_file "$target")"
      printf 'sha256 %s %s\n' "$hash" "$(basename "$target")"
    done
    grep -E 'PASS|FAIL|max \|Δ\||mel-loss|waveform' "$parity_log" | tail -n 80 || true
  } | tee "$summary_file"

  transfer_args="$logs_dir/apple-transfer-args.txt"
  [[ ! -e "$transfer_args" ]] || { die "transfer args already exists"; return 2; }
  {
    printf '%q ' scripts/verify/apple-silicon-sbv2.sh \
      --expected-head "$expected_head" \
      --gguf '<APPLE_PACKET>/sbv2-v2-jp-extra-base.gguf' \
      --gguf-sha256 "$gguf_sha" \
      --reference-dir '<APPLE_PACKET>' \
      --reference-manifest-sha256 "$reference_manifest_sha" \
      --packet-sha256 "$packet_sha" \
      --approval-evidence '<APPLE_APPROVAL_EVIDENCE>' \
      --approval-sha256 "$approval_sha" \
      --evidence-dir '<APPLE_EVIDENCE>'
    printf '\n'
  } > "$transfer_args"

  trap - EXIT
  step "PASS"
  log "summary: $summary_file"
}

main "$@"
