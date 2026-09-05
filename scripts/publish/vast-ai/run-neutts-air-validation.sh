#!/usr/bin/env bash
# Validate exact public NeuTTS Air against the official model. VAST-only; no upload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/neutts_air"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"
DEPENDENCY_AUDIT="$PARITY_PROJECT/dependency_audit.py"
DEPENDENCY_AUDIT_WRAPPER="$VOKRA_ROOT/scripts/publish/vast-ai/audit-neutts-air-dependencies.sh"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

PUBLIC_REPO="vokra/neutts-air"
PUBLIC_REVISION="df2b47ec81862f0e3a19eb2638a6a2bcd2f13b8c"
PUBLIC_FILE="neutts-air.gguf"
PUBLIC_BYTES="1495883328"
PUBLIC_SHA256="f6caf559e919b16d77ac28177e59ee5427a5de92bdeedd719ecab00b4afbb754"

COMPANION_REPO="vokra/distill-neucodec"
COMPANION_REVISION="1471e4d9b82bfb98ae201f02e746fca346c3eb56"
COMPANION_FILE="model.gguf"
COMPANION_BYTES="1025417504"
COMPANION_SHA256="15e60e7e5f7242255b18e1386b26c2a8f872c77a56ca241ee82c8aa5d8b6327f"

UPSTREAM_REPO="neuphonic/neutts-air"
UPSTREAM_REVISION="3b58b776406b62fdc137e31ea53d728f5c22a4ed"
SOURCE_REPO="https://github.com/neuphonic/neutts.git"
SOURCE_REVISION="3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e"
SOURCE_RELATIVE="neuttsair/neutts.py"
SOURCE_BYTES="9035"
SOURCE_SHA256="e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1"

MIN_VAST_MEM_KIB=48000000
MIN_FREE_DISK_KIB=25000000
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[neutts-air-vast] %s\n' "$*" >&2; }
step() { printf '\n[neutts-air-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-neutts-air-validation.sh --approval-evidence <file> [--work-dir <absent-dir>]
       run-neutts-air-validation.sh --self-test

VAST-only, non-publishing gate for the exact public NeuTTS Air release. It
downloads the pinned public LM GGUF and Distill NeuCodec companion, downloads
the exact gated upstream snapshot, executes the fixed Neuphonic prompt method
and official Transformers Qwen2 model in CPU FP32, then compares Vokra logits
at atol=0.01 and greedy ids exactly. Workspace and Apple-target feature builds
run only after the real CPU gate.

There is no --push flag and no upload path. Pull only the small evidence and
reference directories, never model payloads, then destroy the VAST instance.
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

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "NeuTTS Air model work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 48-GB-class guard"
  fi
  local disk_path="$VOKRA_SCRATCH"
  [[ -e "$disk_path" ]] || disk_path="$(dirname "$disk_path")"
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 25-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk find tee wc df grep readelf; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" ]] \
    || die "NeuTTS Air parity lock/project is missing"
  [[ -f "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" ]] \
    || die "NeuTTS Air preflight gate inputs are missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  [[ -f "$DEPENDENCY_AUDIT" && ! -L "$DEPENDENCY_AUDIT" ]] || die "dependency audit is missing or symlinked"
  [[ -f "$DEPENDENCY_AUDIT_WRAPPER" && ! -L "$DEPENDENCY_AUDIT_WRAPPER" ]] || die "dependency audit wrapper is missing or symlinked"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names one exact commit"
  fi
}

pre_sync_gate() {
  local approval="$1"
  step "Validate exact NeuTTS Air closure before synchronization"
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die "approval evidence is missing, symlinked, or empty"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
      --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" \
      --evidence "$approval"
}

write_apple_verifier_command() {
  local output="$1" reference_manifest_sha256="$2"
  {
    printf '%s\n' "VOKRA_REMOTE_APPLE_SILICON=1 \\"
    printf '%s\n' "scripts/verify/apple-silicon-neutts-air.sh \\"
    printf '%s\n' "  --gguf '<APPLE_GGUF_PATH>' \\"
    printf '%s\n' "  --gguf-sha256 $PUBLIC_SHA256 \\"
    printf '%s\n' "  --companion '<APPLE_COMPANION_PATH>' \\"
    printf '%s\n' "  --companion-sha256 $COMPANION_SHA256 \\"
    printf '%s\n' "  --reference '<APPLE_REFERENCE_DIR>' \\"
    printf '%s\n' "  --reference-sha256 $reference_manifest_sha256 \\"
    printf '%s\n' "  --approval-evidence '<APPLE_APPROVAL_EVIDENCE>' \\"
    printf '%s\n' "  --evidence-dir '<APPLE_EVIDENCE_DIR>'"
  } > "$output"
}

require_disjoint_work_dir() {
  local work="$1" approval="$2" candidate root_real approval_parent approval_real
  candidate="$(canonical_absent_path "$work")" || return 2
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" || die "Vokra checkout is inaccessible"
  approval_parent="$(cd -P "$(dirname "$approval")" 2>/dev/null && pwd)" || die "approval parent is inaccessible"
  approval_real="$approval_parent/$(basename "$approval")"
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || die "work-dir overlaps the checkout"
  [[ "$candidate" != "$approval_real" && "$candidate/" != "$approval_real/"* && "$approval_real/" != "$candidate/"* ]] || die "work-dir overlaps approval evidence"
}

canonical_absent_path() {
  local target="$1" current suffix component real lexical
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"; current="/"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die "work-dir path contains .."; return 2; }
    current="${current%/}/$component"
    if [[ -L "$current" ]]; then
      real="$(cd -P "$current" 2>/dev/null && pwd)" || { die "work-dir path contains an inaccessible component"; return 2; }
      case "$current:$real" in
        /var:/private/var|/tmp:/private/tmp) current="$real" ;;
        *) die "work-dir path contains a symlinked component"; return 2 ;;
      esac
    fi
  done
  current="$target"; suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die "work-dir has an inaccessible or symlinked existing parent"; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die "work-dir parent is inaccessible"; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_work_dir() {
  local work="$1" approval="$2"
  require_disjoint_work_dir "$work" "$approval" || return 2
  [[ ! -e "$work" && ! -L "$work" ]] || { die "--work-dir must be absent before validation: $work"; return 2; }
}

require_identity() {
  local label="$1" path="$2" expected_bytes="$3" expected_sha="$4"
  [[ -f "$path" && ! -L "$path" ]] || { die "$label is missing, symlinked, or non-regular: $path"; return 2; }
  local actual_bytes actual_sha
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  actual_sha="$(sha256_file "$path")"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label bytes=$actual_bytes, expected $expected_bytes"
  [[ "$actual_sha" == "$expected_sha" ]] \
    || die "$label SHA-256=$actual_sha, expected $expected_sha"
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
  } > "$output"
}

download_hf_file() {
  local repo="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import os,sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3],
    local_dir=sys.argv[4], token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$repo" "$revision" "$filename" "$output_dir"
}

download_upstream_snapshot() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
    allow_patterns=["config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output_dir"
  printf '%s\n' "$UPSTREAM_REVISION" > "$output_dir/.vokra-source-revision"
}

checkout_source() {
  local output_dir="$1"
  git init -q "$output_dir"
  git -C "$output_dir" remote add origin "$SOURCE_REPO"
  git -C "$output_dir" fetch --depth 1 origin "$SOURCE_REVISION"
  git -C "$output_dir" checkout --detach -q FETCH_HEAD
  [[ "$(git -C "$output_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
    || die "Neuphonic source checkout revision drift"
  require_identity "Neuphonic release source" "$output_dir/$SOURCE_RELATIVE" \
    "$SOURCE_BYTES" "$SOURCE_SHA256"
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" backend="$3"
  local test_count named_line_count result_count total_result_count parity_count composition_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_line_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  parity_count="$(grep -Fxc "NEUTTS_AIR_PARITY ${backend}_vs_official logits_atol=0.01 greedy_ids=exact PASS" "$log_path" || true)"
  composition_count="$(grep -Exc "NEUTTS_AIR_COMPOSITION ${backend} codes=[0-9]+ samples=[0-9]+ PASS" "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing $test_name, got $test_count"; return 2; }
  [[ "$named_line_count" == 1 ]] || { die "expected exactly one total $test_name result line, got $named_line_count"; return 2; }
  [[ "$result_count" == 1 && "$total_result_count" == 1 ]] \
    || { die "expected exactly one exact Cargo result with 1 passed/0 failed/0 ignored"; return 2; }
  [[ "$parity_count" == 1 ]] || { die "expected exactly one full-line ${backend} parity marker"; return 2; }
  [[ "$composition_count" == 1 ]] || { die "expected exactly one full-line ${backend} composition marker"; return 2; }
  ! grep -Eq '^NEUTTS_AIR_(PARITY|COMPOSITION).*FAIL$' "$log_path" \
    || { die "a NEUTTS_AIR FAIL marker is present"; return 2; }
}

run_self_test() {
  local failed=0 probe_root probe_output gate_line host_line tooling_line sync_line audit_line download_line identity_size identity_sha
  [[ "$PUBLIC_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$UPSTREAM_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$PUBLIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$COMPANION_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  gate_line="$(grep -n '^  pre_sync_gate' "$0" | tail -1 | cut -d: -f1)"
  host_line="$(grep -n '^  require_vast_host$' "$0" | tail -1 | cut -d: -f1)"
  tooling_line="$(grep -n '^  require_tooling$' "$0" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^  uv sync --project' "$0" | tail -1 | cut -d: -f1)"
  audit_line="$(grep -n 'DEPENDENCY_AUDIT_WRAPPER.*--output' "$0" | tail -1 | cut -d: -f1)"
  download_line="$(grep -n '^  download_hf_file' "$0" | tail -1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$host_line" =~ ^[0-9]+$ && "$tooling_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ && "$audit_line" =~ ^[0-9]+$ && "$download_line" =~ ^[0-9]+$ ]] || failed=1
  (( gate_line < host_line && gate_line < tooling_line && tooling_line < sync_line && sync_line < audit_line && audit_line < download_line )) || failed=1
  probe_root="$(mktemp -d "${TMPDIR:-/tmp}/vokra-neutts-air-sentinel.XXXXXX")"
  printf '{}\n' > "$probe_root/path-approval.json"
  mkdir -p "$probe_root/nested-parent"
  require_absent_work_dir "$probe_root/nested-parent/model/work" "$probe_root/path-approval.json" || failed=1
  mkdir -p "$probe_root/intermediate"
  ln -s "$VOKRA_ROOT" "$probe_root/intermediate/checkout-link"
  if require_absent_work_dir "$probe_root/intermediate/checkout-link/work" "$probe_root/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  mkdir -p "$probe_root/real/existing"
  ln -s "$probe_root/real" "$probe_root/ancestor-link"
  if require_absent_work_dir "$probe_root/ancestor-link/existing/nested/new" "$probe_root/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  ln -s "$probe_root/missing-target" "$probe_root/dangling-work"
  if require_absent_work_dir "$probe_root/dangling-work" "$probe_root/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$VOKRA_ROOT/tools" "$probe_root/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$probe_root/path-approval.json/child" "$probe_root/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  mkdir "$probe_root/existing-empty"
  if require_absent_work_dir "$probe_root/existing-empty" "$probe_root/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  printf 'identity-self-test\n' > "$probe_root/payload"
  identity_size="$(wc -c < "$probe_root/payload" | tr -d '[:space:]')"
  identity_sha="$(sha256_file "$probe_root/payload")"
  require_identity "self-test payload" "$probe_root/payload" "$identity_size" "$identity_sha" || failed=1
  ln -s "$probe_root/payload" "$probe_root/payload-link"
  if require_identity "self-test symlink payload" "$probe_root/payload-link" "$identity_size" "$identity_sha" >/dev/null 2>&1; then failed=1; fi
  printf '%s\n' \
    'test neutts_air_public_cpu_or_metal_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'NEUTTS_AIR_PARITY Cpu_vs_official logits_atol=0.01 greedy_ids=exact PASS' \
    'NEUTTS_AIR_COMPOSITION Cpu codes=8 samples=3840 PASS' > "$probe_root/valid.log"
  require_one_named_test_passed "$probe_root/valid.log" neutts_air_public_cpu_or_metal_matches_official_reference Cpu || failed=1
  for malformed in duplicate_named duplicate_result duplicate_marker prefix suffix result_suffix FAIL; do
    cp "$probe_root/valid.log" "$probe_root/$malformed.log"
    case "$malformed" in
      duplicate_named) printf '%s\n' 'test neutts_air_public_cpu_or_metal_matches_official_reference ... FAILED' >> "$probe_root/$malformed.log" ;;
      duplicate_result) printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$probe_root/$malformed.log" ;;
      duplicate_marker) printf '%s\n' 'NEUTTS_AIR_PARITY Cpu_vs_official logits_atol=0.01 greedy_ids=exact PASS' >> "$probe_root/$malformed.log" ;;
      prefix) sed 's/^NEUTTS_AIR_/prefix NEUTTS_AIR_/' "$probe_root/$malformed.log" > "$probe_root/$malformed.tmp" && mv "$probe_root/$malformed.tmp" "$probe_root/$malformed.log" ;;
      suffix) sed 's/ PASS$/ PASS trailing/' "$probe_root/$malformed.log" > "$probe_root/$malformed.tmp" && mv "$probe_root/$malformed.tmp" "$probe_root/$malformed.log" ;;
      result_suffix) sed 's/filtered out$/filtered out; finished in nonsense/' "$probe_root/$malformed.log" > "$probe_root/$malformed.tmp" && mv "$probe_root/$malformed.tmp" "$probe_root/$malformed.log" ;;
      FAIL) sed 's/ PASS$/ FAIL/' "$probe_root/$malformed.log" > "$probe_root/$malformed.tmp" && mv "$probe_root/$malformed.tmp" "$probe_root/$malformed.log" ;;
    esac
    if require_one_named_test_passed "$probe_root/$malformed.log" neutts_air_public_cpu_or_metal_matches_official_reference Cpu >/dev/null 2>&1; then failed=1; fi
  done
  rm -rf "$probe_root"
  local command_file
  command_file="$(mktemp "${TMPDIR:-/tmp}/vokra-neutts-air-apple-command.XXXXXX")"
  write_apple_verifier_command "$command_file" "$(printf '%064d' 7)"
  [[ "$(grep -Fxc -- "  --approval-evidence '<APPLE_APPROVAL_EVIDENCE>' \\" "$command_file" || true)" == 1 ]] || failed=1
  [[ "$(grep -Fxc -- "  --evidence-dir '<APPLE_EVIDENCE_DIR>'" "$command_file" || true)" == 1 ]] || failed=1
  bash -n "$command_file" || failed=1
  grep -Fq -- "$VOKRA_ROOT" "$command_file" && failed=1
  # shellcheck disable=SC2016 # verify the literal production writer call
  [[ "$(grep -Fxc -- '  write_apple_verifier_command "$evidence_dir/apple-verifier-command.txt" "$reference_manifest_sha256"' "$0" || true)" == 1 ]] || failed=1
  rm -f "$command_file"
  if command -v uv >/dev/null 2>&1; then
    UV_CACHE_DIR="${NEUTTS_AIR_UV_CACHE_DIR:-/private/tmp/vokra-neutts-air-uv-cache}" \
      uv run --no-project --offline --python 3.12 python "$REFERENCE_DUMPER" --self-test || failed=1
    UV_CACHE_DIR="${NEUTTS_AIR_UV_CACHE_DIR:-/private/tmp/vokra-neutts-air-uv-cache}" \
      uv run --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --self-test || failed=1
  fi
  probe_root="$(mktemp -d "${TMPDIR:-/tmp}/vokra-neutts-air-gate.XXXXXX")"
  probe_output="$probe_root/worker.log"
  printf '%s\n' 'invalid approval evidence' > "$probe_root/approval.json"
  if VOKRA_PUBLISH_ON_VAST=0 VOKRA_SCRATCH="$probe_root/scratch" \
    bash "$0" --approval-evidence "$probe_root/approval.json" --work-dir "$probe_root/work" >"$probe_output" 2>&1; then failed=1; fi
  grep -Fq 'preflight gate' "$probe_output" || failed=1
  grep -Eq 'uv sync|download_hf_file|download_upstream_snapshot|git -C .* fetch|cargo (build|test|check|clippy)' "$probe_output" && failed=1
  [[ ! -e "$probe_root/scratch" && ! -e "$probe_root/work" ]] || failed=1
  rm -rf "$probe_root"
  # shellcheck disable=SC2086 # Each case intentionally models argv tokenization.
  for bad_args in "--self-test --approval-evidence x" "--self-test --self-test" "--work-dir x --work-dir y" "--approval-evidence" "--approval-evidence --work-dir x" "--unknown x"; do
    if bash "$0" $bad_args >/dev/null 2>&1; then failed=1; fi
  done
  if (( failed != 0 )); then
    die "self-test FAIL"
  fi
  log "self-test PASS"
}

main() {
  local work_dir='' approval_evidence='' self_test=0
  local seen_work=0 seen_approval=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --work-dir)
        (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_work=1
        work_dir="$2"
        shift 2
        ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; seen_approval=1
        approval_evidence="$2"; shift 2
        ;;
      --self-test)
        (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1
        self_test=1
        shift
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        usage
        die "unknown argument $1"
        ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$work_dir$approval_evidence" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi

  [[ -n "$approval_evidence" ]] || { usage; die "--approval-evidence is required"; }
  pre_sync_gate "$approval_evidence"
  require_vast_host
  require_tooling
  if [[ -z "$work_dir" ]]; then
    work_dir="$VOKRA_SCRATCH/neutts-air-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  require_absent_work_dir "$work_dir" "$approval_evidence"
  mkdir -p "$work_dir"

  local evidence_dir="$work_dir/evidence"
  local public_dir="$work_dir/public-neutts"
  local companion_dir="$work_dir/public-neucodec"
  local upstream_dir="$work_dir/upstream"
  local source_dir="$work_dir/source"
  local reference_dir="$evidence_dir/reference"
  local gguf="$public_dir/$PUBLIC_FILE"
  local companion="$companion_dir/$COMPANION_FILE"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Install locked official reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Audit exact synchronized dependency closure before model acquisition"
  VOKRA_PUBLISH_ON_VAST=1 VOKRA_ROOT="$VOKRA_ROOT" \
    "$DEPENDENCY_AUDIT_WRAPPER" --output "$evidence_dir/dependency-audit.json"

  step "Download and authenticate exact public GGUFs"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$COMPANION_REPO" "$COMPANION_REVISION" "$COMPANION_FILE" "$companion_dir"
  require_identity "NeuTTS Air public GGUF" "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"
  require_identity "Distill NeuCodec public GGUF" "$companion" \
    "$COMPANION_BYTES" "$COMPANION_SHA256"

  step "Download exact gated upstream snapshot and official source"
  download_upstream_snapshot "$upstream_dir"
  checkout_source "$source_dir"

  step "Generate independent official FP32 reference"
  VOKRA_REFERENCE_TORCH_THREADS="${VOKRA_REFERENCE_TORCH_THREADS:-8}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
      "$REFERENCE_DUMPER" \
      --model-dir "$upstream_dir" \
      --source-file "$source_dir/$SOURCE_RELATIVE" \
      --output "$reference_dir" \
      --max-new-tokens 4 \
      2>&1 | tee "$evidence_dir/reference.log"

  step "Build Vokra and compare CPU logits/greedy generation/composition"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"
  env \
    VOKRA_NEUTTS_AIR_GGUF="$gguf" \
    VOKRA_NEUTTS_AIR_REFERENCE_DIR="$reference_dir" \
    VOKRA_NEUTTS_AIR_COMPANION_GGUF="$companion" \
    VOKRA_NEUTTS_AIR_BACKEND=cpu \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test neutts_air_real \
      neutts_air_public_cpu_or_metal_matches_official_reference -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity-cpu.log"
  require_one_named_test_passed "$evidence_dir/parity-cpu.log" \
    neutts_air_public_cpu_or_metal_matches_official_reference Cpu
  local prompt_ids
  prompt_ids="$(awk -F= '$1 == "prompt_ids_csv" {print substr($0, index($0, "=") + 1); exit}' "$reference_dir/manifest.txt")"
  [[ -n "$prompt_ids" ]] || die "reference manifest has no prompt_ids_csv"
  "$VOKRA_ROOT/target/release/vokra-cli" run \
    --model "$gguf" \
    --neutts-companion "$companion" \
    --token-ids "$prompt_ids" \
    --neutts-greedy \
    --neutts-max-new-tokens 4 \
    --output "$evidence_dir/neutts-air-cpu.wav" \
    2>&1 | tee "$evidence_dir/cli-cpu.log"
  [[ -s "$evidence_dir/neutts-air-cpu.wav" ]] || die "CLI emitted no WAV"

  step "Run repository gates and full VAST verification"
  cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
  bash "$VOKRA_ROOT/scripts/check-arch-handshake.sh"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    2>&1 | tee "$evidence_dir/workspace-test.log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    --all-targets -- -D warnings 2>&1 | tee "$evidence_dir/workspace-clippy.log"

  step "Cross-check Apple Metal feature compilation"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$evidence_dir/apple-metal-cross-check.log"

  local reference_manifest_sha256
  reference_manifest_sha256="$(sha256_file "$reference_dir/manifest.txt")"
  write_apple_verifier_command "$evidence_dir/apple-verifier-command.txt" "$reference_manifest_sha256"
  {
    echo "apple_verifier_command=$(tr '\n' ' ' < "$evidence_dir/apple-verifier-command.txt")"
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "companion_repo=$COMPANION_REPO"
    echo "companion_revision=$COMPANION_REVISION"
    echo "companion_sha256=$COMPANION_SHA256"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "source_revision=$SOURCE_REVISION"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "next_logits_atol=$FP32_ATOL"
    echo "greedy_ids=exact"
    echo "composition=PASS"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then destroy the VAST instance"
}

FP32_ATOL="0.01"
main "$@"
