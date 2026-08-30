#!/usr/bin/env bash
# Authenticate, convert, and generate the first official reference for the
# 8.49 GB MOSS Audio Tokenizer v2 release. VAST-only; never publishes/uploads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
V2_PROJECT="$PARITY_PROJECT/moss_audio_tokenizer_v2"
LICENSE_GATE="$V2_PROJECT/license_gate.py"
LICENSE_MANIFEST="$V2_PROJECT/license_gate_manifest.json"
DEPENDENCY_AUDIT_WRAPPER="$VOKRA_ROOT/scripts/publish/vast-ai/audit-moss-audio-tokenizer-v2-dependencies.sh"
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
usage: run-moss-audio-tokenizer-v2-validation.sh --approval-evidence <file> [--work-dir <absent-dir>]
       run-moss-audio-tokenizer-v2-validation.sh --self-test

VAST-only artifact/reference gate for MOSS Audio Tokenizer v2. It downloads
the immutable official three-shard release, verifies every exact file and real
safetensors header, merges and converts it, authenticates the resulting GGUF
header, and invokes the pinned official model's decode path on CUDA.

This worker deliberately does not invent a numerical bound: after producing
the authenticated manifest and independent CUDA reference it runs the mapped
native CPU decoder as a measurement-only comparison. Metal execution is an
Apple-only concern handled by the separate verifier. It contains no publish,
upload, or Hugging Face push operation.
After owner preflight and a separately authorized frozen sync, the
model-free dependency audit runs before model acquisition or Cargo.
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
  local mem_kib free_kib gpu_mem_mib disk_path parent
  require_vast_marker
  [[ "$(uname -s)" == "Linux" ]] \
    || die "large-model work is Linux/VAST-only; refusing host $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 64-GB class guard"
  fi
  disk_path="$VOKRA_SCRATCH"
  while [[ ! -e "$disk_path" ]]; do
    parent="$(dirname "$disk_path")"
    [[ "$parent" != "$disk_path" ]] || die "scratch parent cannot be resolved"
    disk_path="$parent"
  done
  [[ -d "$disk_path" && ! -L "$disk_path" ]] || die "scratch filesystem path is not a real directory"
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
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
  for tool in uv cargo rustc rustup git awk grep find tee wc tr readelf nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$V2_PROJECT/uv.lock" && -f "$V2_PROJECT/pyproject.toml" ]] || die "dedicated v2 uv project is missing"
  [[ -f "$AUDITOR" ]] || die "v2 manifest auditor is missing"
  [[ -f "$DEPENDENCY_AUDIT_WRAPPER" ]] || die "v2 dependency audit wrapper is missing"
  [[ -f "$PREPARER" ]] || die "v2 checkpoint preparer is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
}

require_cpu_test_evidence() {
  local path="$1" named result result_lines test_lines cpu
  named="$(grep -Ec '^test moss_audio_tokenizer::full_decoder::tests::measure_v2_real_cpu_and_optional_metal_against_official \.\.\. ok$' "$path" || true)"
  result="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)"
  result_lines="$(grep -Ec '^test result:' "$path" || true)"
  test_lines="$(awk '/^test / && $0 !~ /^test result:/ {count++} END {print count + 0}' "$path")"
  cpu="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ actual=[0-9.eE+-]+ reference=[0-9.eE+-]+$' "$path" || true)"
  if [[ "$named" != 1 || "$result" != 1 || "$result_lines" != 1 || "$test_lines" != 1 || "$cpu" != 1 ]]; then
    die 'MOSS v2 CPU evidence requires exactly one named pass/result/sentinel'; return 2
  fi
}

write_apple_args() {
  local output="$1" gguf_sha="$2" reference_sha="$3"
  {
    printf '#!/usr/bin/env bash\nset -eu\n'
    printf '%s ' 'scripts/verify/apple-silicon-moss-audio-tokenizer-v2.sh'
    printf '%s ' --gguf "'<APPLE_MOSS_AUDIO_TOKENIZER_V2_GGUF_PATH>'" --reference "'<APPLE_MOSS_AUDIO_TOKENIZER_V2_REFERENCE_PATH>'"
    printf '%q ' --gguf-sha256 "$gguf_sha" --reference-sha256 "$reference_sha"
    printf '%s ' --approval-evidence "'<APPLE_MOSS_AUDIO_TOKENIZER_V2_APPROVAL_EVIDENCE>'"
    printf '%s\n' --evidence-dir "'<APPLE_MOSS_AUDIO_TOKENIZER_V2_EVIDENCE_DIR>'"
  } > "$output"
  chmod +x "$output"
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || { die "required tool missing: uv"; return 2; }
  [[ -f "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" ]] || { die "v2 approval gate/manifest missing"; return 2; }
  [[ -f "$approval" && ! -L "$approval" ]] || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --lock "$V2_PROJECT/uv.lock" --project "$V2_PROJECT/pyproject.toml" \
    --manifest "$LICENSE_MANIFEST" --approval-evidence "$approval"
}

canonical_candidate() {
  local value="$1" suffix='' parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"
  [[ -n "$value" ]] || { die "path is empty"; return 2; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || { die "path contains a symlink ancestor: $parent"; return 2; }
    parent="$(dirname "$parent")"
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die "path has no canonical parent"; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die "path parent is not a real directory"; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

canonical_existing_path() {
  local value="$1" parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"
  [[ -e "$value" && ! -L "$value" ]] || return 1
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || return 1
    parent="$(dirname "$parent")"
  done
  if [[ -d "$value" ]]; then
    (cd -P "$value" && printf '%s\n' "$PWD")
  else
    parent="$(dirname "$value")"
    (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$value")")
  fi
}

paths_overlap() { local left="${1%/}" right="${2%/}"; [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]; }

validate_work_dir() {
  local work="$1" approval="$2" canonical_work canonical_root canonical_project approval_real
  [[ "$work" = /* ]] || { die "--work-dir must be an absolute path"; return 2; }
  [[ ! -e "$work" && ! -L "$work" ]] || { die "--work-dir must be absent/nonexistent"; return 2; }
  canonical_work="$(canonical_candidate "$work")" || return 2
  canonical_root="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  canonical_project="$(canonical_candidate "$V2_PROJECT")" || return 2
  [[ -f "$approval" && ! -L "$approval" ]] || { die "approval evidence must be a regular non-symlink file"; return 2; }
  approval_real="$(canonical_existing_path "$approval")" || { die "approval evidence path contains a symlink ancestor"; return 2; }
  paths_overlap "$canonical_work" "$canonical_root" && { die "--work-dir overlaps checkout"; return 2; }
  paths_overlap "$canonical_work" "$canonical_project" && { die "--work-dir overlaps project"; return 2; }
  paths_overlap "$canonical_work" "$approval_real" && { die "--work-dir overlaps approval"; return 2; }
}

download_snapshot() {
  local output="$1"
  mkdir -p "$output"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["LICENSE", "config.json", "configuration_moss_audio_tokenizer.py", "modeling_moss_audio_tokenizer.py", "model.safetensors.index.json", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"])' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output"
}

verify_audit_json() {
  local path="$1" require_gguf="$2"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python -c \
    'import json,pathlib,sys
def reject(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise ValueError(f"duplicate JSON key: {k}")
  d[k]=v
 return d
data=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
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
    "$SHARD1_SHA256" "$SHARD2_SHA256" "$SHARD3_SHA256" \
    || { die 'v2 audit JSON validation failed'; return 2; }
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
    uv run --project "$V2_PROJECT" --frozen --python 3.12 python -c \
      'import platform,torch,transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}")'
  } | tee "$output"
}

run_self_test_work_paths() {
  local probe status=0
  probe="$(cd -P "$(mktemp -d)" && pwd -P)"
  mkdir -p "$probe/real/existing"
  ln -s "$probe/real" "$probe/link"
  if validate_work_dir "$probe/link/existing/nested/new" "$probe/approval.json" >/dev/null 2>&1; then
    die 'existing descendant under symlink ancestor accepted'
    status=1
  fi
  rm -rf "$probe"
  return "$status"
}

run_self_test() {
  local tmp payload actual cases=0 fail=0 script_path fake_root fake_home fake_log rc audit_marker
  run_self_test_work_paths
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  expect_exit_2_no_path() {
    local label="$1" path="$2" status
    shift 2
    if "$@" >/dev/null 2>&1; then
      status=0
    else
      status=$?
    fi
    if [[ $status -ne 2 || -e "$path" || -L "$path" ]]; then
      log "self-test FAIL: $label was not a controlled reject without output"
      return 1
    fi
  }
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
    "license_preflight" "--no-project --offline" "license_gate.py" \
    "audit-moss-audio-tokenizer-v2-dependencies.sh" "--no-sync" "dependency_audit.py" "readelf" \
    "object_pairs_hook=reject" \
    "measure_v2_real_cpu_and_optional_metal_against_official" \
    "numeric_bounds=UNSET"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done
  if ! grep -Fq 'object_pairs_hook=reject_duplicate_json_keys' "$PREPARER"; then
    log 'self-test FAIL: checkpoint index duplicate-key rejection is missing'
    fail=1
  fi
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
  cases=$((cases + 1))
  audit_marker="  VOKRA_PUBLISH_ON_VAST=1 \"\$DEPENDENCY_AUDIT_WRAPPER\" --output \"\$dependency_audit_json\""
  check_dependency_audit_order() {
    local candidate="$1" sync_line audit_line snapshot_line sync_marker snapshot_marker
    sync_marker="  uv sync --project \"\$V2_PROJECT\" --frozen --python 3.12"
    snapshot_marker='  step "Download immutable official three-shard snapshot"'
    sync_line="$(grep -nF "$sync_marker" "$candidate" | tail -n 1 | cut -d: -f1)"
    audit_line="$(grep -nF "$audit_marker" "$candidate" | tail -n 1 | cut -d: -f1)"
    snapshot_line="$(grep -nF "$snapshot_marker" "$candidate" | tail -n 1 | cut -d: -f1)"
    [[ "$sync_line" =~ ^[0-9]+$ && "$audit_line" =~ ^[0-9]+$ && "$snapshot_line" =~ ^[0-9]+$ ]] || return 1
    (( sync_line < audit_line && audit_line < snapshot_line ))
  }
  if ! check_dependency_audit_order "$script_path"; then
    log 'self-test FAIL: dependency audit is not after sync and before model acquisition'
    fail=1
  fi
  cases=$((cases + 1))
  local without_audit="$tmp/worker-without-dependency-audit.sh"
  grep -vF "$audit_marker" "$script_path" > "$without_audit"
  if check_dependency_audit_order "$without_audit"; then
    log 'self-test FAIL: production audit-call deletion was accepted'
    fail=1
  fi
  printf approval > "$tmp/approval-target"
  ln -s "$tmp/approval-target" "$tmp/approval-link"
  if license_preflight "$tmp/approval-link" >/dev/null 2>&1; then
    log "self-test FAIL: symlink approval was accepted"; fail=1
  fi
  cases=$((cases + 1))
  expect_exit_2_no_path 'duplicate --self-test' "$tmp/duplicate-self-test-output" \
    "$script_path" --self-test --self-test || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'bare --work-dir' "$tmp/bare-work-output" \
    "$script_path" --work-dir --self-test || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'negative --work-dir' "$tmp/negative-work-output" \
    "$script_path" --work-dir -x || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'empty --work-dir' "$tmp/empty-work-output" \
    "$script_path" --work-dir "" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'trailing argument' "$tmp/trailing-output" \
    "$script_path" --self-test trailing || fail=1
  mkdir -p "$tmp/real/existing"
  ln -s "$tmp/real" "$tmp/link"
  cases=$((cases + 1))
  expect_exit_2_no_path 'work path under symlink ancestor' "$tmp/link/existing/nested/new" \
    validate_work_dir "$tmp/link/existing/nested/new" "$tmp/approval.json" || fail=1
  mkdir -p "$tmp/approval-real"
  printf '{}\n' > "$tmp/approval-real/evidence.json"
  ln -s "$tmp/approval-real" "$tmp/approval-parent-link"
  cases=$((cases + 1))
  expect_exit_2_no_path 'approval path under symlink ancestor' "$tmp/approval-work" \
    validate_work_dir "$tmp/approval-work" "$tmp/approval-parent-link/evidence.json" || fail=1
  printf '{}\n' > "$tmp/approval.json"
  cases=$((cases + 1))
  expect_exit_2_no_path 'relative work path' 'relative-v2-work' \
    validate_work_dir 'relative-v2-work' "$tmp/approval.json" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'lexical checkout overlap' "$V2_PROJECT/../v2-lexical-work" \
    validate_work_dir "$V2_PROJECT/../v2-lexical-work" "$tmp/approval.json" || fail=1
  mkdir "$tmp/empty-work"
  cases=$((cases + 1))
  if validate_work_dir "$tmp/empty-work" "$tmp/approval.json" >/dev/null 2>&1; then
    log "self-test FAIL: pre-existing empty work directory accepted"
    fail=1
  fi
  cases=$((cases + 1))
  expect_exit_2_no_path 'checkout-overlapping work path' "$VOKRA_ROOT/v2-self-test-work" \
    validate_work_dir "$VOKRA_ROOT/v2-self-test-work" "$tmp/approval.json" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'approval-overlapping work path' "$tmp/approval.json/child" \
    validate_work_dir "$tmp/approval.json/child" "$tmp/approval.json" || fail=1

  cases=$((cases + 1))
  printf 'test moss_audio_tokenizer::full_decoder::tests::measure_v2_real_cpu_and_optional_metal_against_official ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\n' > "$tmp/cpu.log"
  require_cpu_test_evidence "$tmp/cpu.log" || { log "self-test FAIL: valid CPU evidence rejected"; fail=1; }
  cases=$((cases + 1))
  cp "$tmp/cpu.log" "$tmp/duplicate-result.log"
  printf 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n' >> "$tmp/duplicate-result.log"
  if require_cpu_test_evidence "$tmp/duplicate-result.log"; then
    log "self-test FAIL: duplicate test result accepted"
    fail=1
  fi
  cases=$((cases + 1))
  awk 'NR == 2 { print "test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"; next } { print }' \
    "$tmp/cpu.log" > "$tmp/malformed-result.log"
  if require_cpu_test_evidence "$tmp/malformed-result.log"; then
    log "self-test FAIL: malformed test result accepted"
    fail=1
  fi
  cases=$((cases + 1))
  printf 'test moss_audio_tokenizer::full_decoder::tests::measure_v2_real_cpu_and_optional_metal_against_official ... ok\ntest another ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_V2_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\n' > "$tmp/extra-test.log"
  if require_cpu_test_evidence "$tmp/extra-test.log"; then
    log "self-test FAIL: extra test accepted"
    fail=1
  fi
  cases=$((cases + 1))
  sed 's/filtered out$/filtered out; finished in nope/' "$tmp/cpu.log" > "$tmp/bad-timing.log"
  if require_cpu_test_evidence "$tmp/bad-timing.log"; then
    log "self-test FAIL: malformed timing accepted"
    fail=1
  fi
  cases=$((cases + 1))
  write_apple_args "$tmp/apple-args.sh" "$(printf '%064d' 0)" "$(printf '%064d' 0)"
  bash -n "$tmp/apple-args.sh"
  grep -Fq "scripts/verify/apple-silicon-moss-audio-tokenizer-v2.sh --gguf '<APPLE_MOSS_AUDIO_TOKENIZER_V2_GGUF_PATH>' --reference '<APPLE_MOSS_AUDIO_TOKENIZER_V2_REFERENCE_PATH>'" "$tmp/apple-args.sh" || { log 'self-test FAIL: Apple args are not portable placeholders'; fail=1; }
  grep -Fq -- "--approval-evidence '<APPLE_MOSS_AUDIO_TOKENIZER_V2_APPROVAL_EVIDENCE>'" "$tmp/apple-args.sh" || { log 'self-test FAIL: Apple approval placeholder missing'; fail=1; }
  if grep -Eq '(/stage/|/reference/|VOKRA_ROOT=|moss-tokenizer-v2-validation/)' "$tmp/apple-args.sh"; then log 'self-test FAIL: Apple args embed VAST paths'; fail=1; fi

  fake_root="$tmp/root"; fake_home="$tmp/home"; fake_log="$tmp/fake-uv.log"
  mkdir -p "$fake_root/tools/parity/moss_audio_tokenizer_v2" "$fake_home/.local/bin"
  cp "$V2_PROJECT/license_gate.py" "$V2_PROJECT/license_gate_manifest.json" \
    "$V2_PROJECT/uv.lock" "$V2_PROJECT/pyproject.toml" "$fake_root/tools/parity/moss_audio_tokenizer_v2/"
  cat > "$fake_home/.local/bin/uv" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${MOSS_V2_SELF_TEST_UV_LOG:?}"
exit 2
EOF
  chmod +x "$fake_home/.local/bin/uv"
  printf '{}' > "$tmp/approval.json"
  set +e
  HOME="$fake_home" PATH="$fake_home/.local/bin:$PATH" \
    MOSS_V2_SELF_TEST_UV_LOG="$fake_log" VOKRA_ROOT="$fake_root" \
    VOKRA_SCRATCH="$tmp/scratch" "$script_path" --approval-evidence "$tmp/approval.json" \
      --work-dir "$tmp/blocked-work" >"$tmp/worker.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 2 || ! -s "$fake_log" || -e "$tmp/scratch" ]]; then
    log "self-test FAIL: approval gate did not block before VAST host/scratch"; fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-moss-audio-tokenizer-v2-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

write_failure_summary_on_exit() {
  local rc=$?
  if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then
    printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"
  fi
  exit "$rc"
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir snapshot stage logs reference
  local seen_work_dir=0 seen_self_test=0 seen_approval=0
  local merged gguf audit_before audit_after dependency_audit_json dependency_audit_log reference_csv reference_sha256 gguf_sha256 run_log env_log summary_file
  local compile_log cpu_log
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --approval-evidence)
        (( ! seen_approval++ )) || { die "duplicate --approval-evidence"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence requires a file"; return 2; }
        approval_evidence="$2"; shift 2 ;;
      --work-dir)
        (( ! seen_work_dir++ )) || { die "duplicate --work-dir"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a directory"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --self-test)
        (( ! seen_self_test++ )) || { die "duplicate --self-test"; return 2; }
        self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ -z "$approval_evidence$requested_work_dir" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi

  # Approval is checked before host probing, scratch/HF cache creation, sync,
  # downloads, merge, Cargo, or CUDA work.
  [[ -n "$approval_evidence" ]] || { die "--approval-evidence is required"; usage; return 2; }
  [[ -f "$approval_evidence" && ! -L "$approval_evidence" ]] || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  license_preflight "$approval_evidence"
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/moss-tokenizer-v2-validation/$run_stamp}"
  validate_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  snapshot="$work_dir/upstream"
  stage="$work_dir/stage"
  logs="$work_dir/logs"
  reference="$work_dir/reference"
  merged="$stage/moss-audio-tokenizer-v2.safetensors"
  gguf="$stage/moss-audio-tokenizer-v2.gguf"
  audit_before="$logs/safetensors-audit.json"
  audit_after="$logs/gguf-audit.json"
  dependency_audit_json="$logs/dependency-audit.json"
  dependency_audit_log="$logs/dependency-audit.log"
  reference_csv="$reference/moss-audio-tokenizer-v2-reference.csv"
  mkdir -p "$snapshot" "$stage" "$logs" "$reference"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-moss-tokenizer-v2"
  export HF_HOME="$VOKRA_SCRATCH/hf-home-moss-tokenizer-v2"
  run_log="$logs/run.log"
  env_log="$logs/environment.txt"
  summary_file="$logs/summary.txt"
  compile_log="$logs/compile.log"
  cpu_log="$logs/cpu-measurement.log"
  exec > >(tee -a "$run_log") 2>&1
  trap write_failure_summary_on_exit EXIT

  step "Sync locked Python 3.12 parity environment"
  uv sync --project "$V2_PROJECT" --frozen --python 3.12

  step "Audit synchronized dependency closure before model acquisition"
  VOKRA_PUBLISH_ON_VAST=1 "$DEPENDENCY_AUDIT_WRAPPER" --output "$dependency_audit_json" 2>&1 | tee "$dependency_audit_log"

  step "Download immutable official three-shard snapshot"
  download_snapshot "$snapshot"

  step "Authenticate exact files and real safetensors headers"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python "$AUDITOR" \
    --config "$snapshot/config.json" \
    --index "$snapshot/model.safetensors.index.json" \
    --shard-dir "$snapshot" | tee "$audit_before"
  verify_audit_json "$audit_before" 0

  step "Merge exact shards into one converter input"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python "$PREPARER" \
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
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python "$AUDITOR" \
    --config "$snapshot/config.json" \
    --index "$snapshot/model.safetensors.index.json" \
    --shard-dir "$snapshot" \
    --gguf "$gguf" | tee "$audit_after"
  verify_audit_json "$audit_after" 1

  step "Record environment before official numerical output"
  record_environment "$env_log"

  step "Generate independent official CUDA reference"
  uv run --project "$V2_PROJECT" --frozen --python 3.12 python \
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
  reference_sha256="$(sha256_file "$reference_csv")"

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
  require_cpu_test_evidence "$cpu_log"

  step "Write evidence summary and checksums"
  gguf_sha256="$(sha256_file "$gguf")"
  write_apple_args "$logs/apple-silicon-moss-audio-tokenizer-v2-args.sh" "$gguf_sha256" "$reference_sha256"
  {
    echo "execution_status=PASS"
    echo "scope=AUTHENTICATED_ARTIFACT_REFERENCE_AND_CPU_MEASUREMENT"
    echo "numeric_verdict=MEASURED_NOT_GATED"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "manifest_sha256=$CANDIDATE_MANIFEST_SHA256"
    echo "gguf_sha256=$gguf_sha256"
    echo "reference_sha256=$reference_sha256"
    echo "metal_runtime=NOT_RUN_LINUX_VAST"
    echo "metal_cross_compile=NOT_ATTEMPTED_LINUX_VAST"
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
