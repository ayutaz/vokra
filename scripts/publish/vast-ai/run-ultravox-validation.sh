#!/usr/bin/env bash
# Validate exact public Ultravox against the official model. VAST-only; no upload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/ultravox"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

PUBLIC_REPO="vokra/ultravox-v0-5-llama-3-2-1b"
PUBLIC_REVISION="ddbbeec5bfcb09c71a1f88971b794e3e5da811f9"
PUBLIC_FILE="ultravox-v0-5-llama-3-2-1b.gguf"
PUBLIC_BYTES="1366275264"
PUBLIC_SHA256="376c79a7219bb38fc6a857b0bd9ccf57daff878e7bb4723c4801000c0d7b8c9c"

UPSTREAM_REPO="fixie-ai/ultravox-v0_5-llama-3_2-1b"
UPSTREAM_REVISION="b95bec8ab291eeb04b5cd600dd473377f6b79026"
COMPANION_REPO="meta-llama/Llama-3.2-1B-Instruct"
COMPANION_REVISION="9213176726f574b556790deb65791e0c5aa438b6"

MODEL_SOURCE_BYTES="41578"
MODEL_SOURCE_SHA256="df618218561375da01bb53bd2764ea123e0cbf782f3326753f669f63ff6c6d3f"
PROCESSOR_SOURCE_BYTES="17087"
PROCESSOR_SOURCE_SHA256="2ae6682f3deecb22539fae6a6631688fc1675282f1a5b31145d9f95d2347ff7b"
CONFIG_SOURCE_BYTES="7057"
CONFIG_SOURCE_SHA256="99cf5ad911189f2351c2232234025db56b23763283583c0a848ebf2a1ecc40fc"
UPSTREAM_MODEL_BYTES="1366293736"
UPSTREAM_MODEL_SHA256="f3a3bf7e9137f3219a0d27ba71668deeee8c60aaf0ea587b48d8f71178763f31"
ULTRAVOX_SNAPSHOT_FILES=(
  config.json generation_config.json model.safetensors preprocessor_config.json
  processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json
  ultravox_config.py ultravox_model.py ultravox_processing.py
)
COMPANION_SNAPSHOT_FILES=(config.json model.safetensors)

MIN_VAST_MEM_KIB=64000000
MIN_FREE_DISK_KIB=30000000
FP32_ATOL="0.01"
TEST_NAME="ultravox_public_cpu_or_metal_matches_official_reference"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[ultravox-vast] %s\n' "$*" >&2; }
step() { printf '\n[ultravox-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  local scan rest component
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_work_dir() {
  local target="$1" approval="$2" canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "--work-dir must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize --work-dir: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$LICENSE_GATE" "$LICENSE_MANIFEST" \
    "$PARITY_PROJECT/uv.lock" "$PARITY_PROJECT/pyproject.toml" "$approval"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected path is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected path: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "--work-dir overlaps protected path: $protected"; return 2; }
  done
  return 0
}

usage() {
  cat <<'EOF' >&2
usage: run-ultravox-validation.sh --approval-evidence <json> [--work-dir <absent-dir>]
       run-ultravox-validation.sh --self-test

VAST-only, non-publishing gate for the exact public Ultravox v0.5 audio GGUF.
It downloads and authenticates the fixed public artifact, fixed Fixie snapshot,
and exact gated Meta Llama companion. It converts the companion locally,
executes the official custom model and processor in CPU FP32, compares Vokra
frontend/audio embeddings/logits at atol=0.01 and greedy IDs exactly, then runs
the repository verification gates.

There is no --push flag and no upload path. Pull only the small evidence and
reference directories. Do not pull model payloads to the maintainer Mac; send
them directly to a disposable Apple worker if Metal parity is required, then
destroy the VAST instance.
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

license_preflight() {
  local approval_file="$1"
  local -a gate_args=(
    --lock "$PARITY_PROJECT/uv.lock"
    --project "$PARITY_PROJECT/pyproject.toml"
    --manifest "$LICENSE_MANIFEST"
    --public-repo "$PUBLIC_REPO"
    --public-revision "$PUBLIC_REVISION"
    --public-file "$PUBLIC_FILE"
    --public-sha256 "$PUBLIC_SHA256"
    --upstream-repo "$UPSTREAM_REPO"
    --upstream-revision "$UPSTREAM_REVISION"
    --companion-repo "$COMPANION_REPO"
    --companion-revision "$COMPANION_REVISION"
    --upstream-model-sha256 "$UPSTREAM_MODEL_SHA256"
  )
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" && -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] \
    || die "tracked Ultravox license gate and manifest are missing"
  [[ -s "$approval_file" && ! -L "$approval_file" ]] || die "approval evidence must be a non-empty regular non-symlink file"
  gate_args+=(--approval "$approval_file")
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${gate_args[@]}"
}

file_bytes() {
  wc -c < "$1" | tr -d '[:space:]'
}

require_identity() {
  local label="$1" path="$2" expected_bytes="$3" expected_sha="$4"
  [[ -f "$path" ]] || die "$label is missing: $path"
  local actual_bytes actual_sha
  actual_bytes="$(file_bytes "$path")"
  actual_sha="$(sha256_file "$path")"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label bytes=$actual_bytes, expected $expected_bytes"
  [[ "$actual_sha" == "$expected_sha" ]] \
    || die "$label SHA-256=$actual_sha, expected $expected_sha"
}

require_revision_stamp() {
  local directory="$1" expected="$2" label="$3"
  local stamp="$directory/.vokra-source-revision"
  [[ -f "$stamp" ]] || die "$label exact-revision stamp is missing: $stamp"
  [[ "$(tr -d '[:space:]' < "$stamp")" == "$expected" ]] \
    || die "$label revision does not match fixed revision $expected"
}

require_pass_marker() {
  local log_path="$1" marker="$2"
  grep -F "$marker" "$log_path" >/dev/null \
    || die "required PASS sentinel is absent: $marker"
}

require_test_pass() {
  local log_path="$1" backend="$2" marker="$3" test_count result_count marker_count
  test_count="$(grep -Ec "^test ${TEST_NAME} \.\.\. ok$" "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out($|;)' "$log_path" || true)"
  marker_count="$(grep -Fc "$marker" "$log_path" || true)"
  if [[ "$test_count" != 1 ]]; then
    die "expected exactly one passing $TEST_NAME for $backend, got $test_count"
    return 2
  fi
  if [[ "$result_count" != 1 ]]; then
    die "expected exactly one Cargo result with 1 passed/0 failed/0 ignored for $backend"
    return 2
  fi
  if [[ "$marker_count" != 1 ]]; then
    die "expected exactly one parity marker for $backend, got $marker_count"
    return 2
  fi
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "Ultravox model work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 64-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 30-GB run guard"
  fi
  [[ -n "${HF_TOKEN:-${HF:-}}" ]] \
    || die "HF_TOKEN/HF is required for the gated Meta snapshot"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk find tee wc df grep tr cargo-deny; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Ultravox parity uv.lock is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names one exact commit"
  fi
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
    cargo deny --version
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

download_verified_snapshot() {
  local repo="$1" revision="$2" output_dir="$3"
  shift 3
  mkdir -p "$output_dir"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - "$repo" "$revision" "$output_dir" "$@" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo, revision, output_name, *wanted = sys.argv[1:]
output = Path(output_name)
token = os.environ.get("HF_TOKEN") or os.environ.get("HF")
entries = {}
for entry in HfApi(token=token).list_repo_tree(
    repo_id=repo, revision=revision, recursive=True
):
    name = getattr(entry, "rfilename", getattr(entry, "path", ""))
    if name in wanted:
        lfs = getattr(entry, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        blob_id = getattr(entry, "blob_id", None) or getattr(entry, "oid", None)
        size = getattr(entry, "size", None)
        if not isinstance(size, int) or size <= 0:
            raise SystemExit(f"{repo}: {name} has no positive remote size")
        if name == "model.safetensors" and (
            not isinstance(lfs_sha, str) or len(lfs_sha) != 64
        ):
            raise SystemExit(f"{repo}: {name} has no authenticated LFS SHA-256")
        if name != "model.safetensors" and (
            not isinstance(blob_id, str) or len(blob_id) != 40
        ):
            raise SystemExit(f"{repo}: {name} has no authenticated Git blob identity")
        entries[name] = {
            "size": size,
            "blob_id": blob_id,
            "lfs_sha256": lfs_sha,
        }
if set(entries) != set(wanted):
    raise SystemExit(f"{repo}: exact input closure mismatch: {sorted(entries)}")
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=output,
    allow_patterns=wanted,
    token=token,
)
actual_files = {
    path.relative_to(output).as_posix()
    for path in output.rglob("*")
    if path.is_file() and ".cache" not in path.relative_to(output).parts
}
if actual_files != set(wanted):
    raise SystemExit(f"{repo}: downloaded input closure mismatch: {sorted(actual_files)}")
for name in wanted:
    path = output / name
    digest_state = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    local = {"bytes": path.stat().st_size, "sha256": digest}
    if local["bytes"] != entries[name]["size"]:
        raise SystemExit(f"{repo}: {name} local size differs from remote metadata")
    if name == "model.safetensors" and digest != entries[name]["lfs_sha256"]:
        raise SystemExit(f"{repo}: {name} differs from remote LFS SHA-256")
    entries[name]["local"] = local
(output / ".vokra-source-inventory.json").write_text(
    json.dumps({"repo": repo, "revision": revision, "files": entries}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

download_ultravox_snapshot() {
  local output_dir="$1"
  download_verified_snapshot "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output_dir" \
    "${ULTRAVOX_SNAPSHOT_FILES[@]}"
  printf '%s\n' "$UPSTREAM_REVISION" > "$output_dir/.vokra-source-revision"
  require_revision_stamp "$output_dir" "$UPSTREAM_REVISION" "official Ultravox snapshot"
  require_identity "official Ultravox model source" "$output_dir/ultravox_model.py" \
    "$MODEL_SOURCE_BYTES" "$MODEL_SOURCE_SHA256"
  require_identity "official Ultravox model weights" "$output_dir/model.safetensors" \
    "$UPSTREAM_MODEL_BYTES" "$UPSTREAM_MODEL_SHA256"
  require_identity "official Ultravox processor source" "$output_dir/ultravox_processing.py" \
    "$PROCESSOR_SOURCE_BYTES" "$PROCESSOR_SOURCE_SHA256"
  require_identity "official Ultravox config source" "$output_dir/ultravox_config.py" \
    "$CONFIG_SOURCE_BYTES" "$CONFIG_SOURCE_SHA256"
}

download_companion_snapshot() {
  local output_dir="$1"
  download_verified_snapshot "$COMPANION_REPO" "$COMPANION_REVISION" "$output_dir" \
    "${COMPANION_SNAPSHOT_FILES[@]}"
  printf '%s\n' "$COMPANION_REVISION" > "$output_dir/.vokra-source-revision"
  require_revision_stamp "$output_dir" "$COMPANION_REVISION" "gated companion snapshot"
}

run_self_test() {
  local failed=0 temporary fake_root fakebin trace worker_status
  [[ "$PUBLIC_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$UPSTREAM_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$COMPANION_REVISION" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$PUBLIC_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$MODEL_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$PROCESSOR_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$CONFIG_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$UPSTREAM_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || failed=1
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-ultravox-worker.XXXXXX")"
  trap 'rm -rf "${temporary:-}"' RETURN
  printf 'abc' > "$temporary/value"
  printf '{}\n' > "$temporary/approval.json"
  require_absent_work_dir "$temporary/new-work" "$temporary/approval.json" || failed=1
  mkdir "$temporary/empty-work"
  if require_absent_work_dir "$temporary/empty-work" "$temporary/approval.json" >/dev/null 2>&1; then failed=1; fi
  rmdir "$temporary/empty-work"
  mkdir -p "$temporary/real-parent/child"
  ln -s "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_work_dir "$temporary/link-parent/child/new-work" "$temporary/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$temporary/real-parent" "$temporary/link-parent"
  ln -s "$temporary/missing-work" "$temporary/link-work"
  if require_absent_work_dir "$temporary/link-work" "$temporary/approval.json" >/dev/null 2>&1; then failed=1; fi
  rm "$temporary/link-work"
  if require_absent_work_dir "$VOKRA_ROOT/ultravox-self-test-work" "$temporary/approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$temporary/approval.json/child" "$temporary/approval.json" >/dev/null 2>&1; then failed=1; fi
  require_identity "self-test value" "$temporary/value" "3" \
    "ba7816bf8f01cfea414140de5dae41?" >/dev/null 2>&1 && failed=1 || true
  local download_block snapshot_block
  download_block="$(awk '/^download_hf_file\(\)/,/^\}/ {print}' "$0")"
  snapshot_block="$(awk '/^download_verified_snapshot\(\)/,/^\}/ {print}' "$0")"
  [[ "$download_block" != *"--with"* && "$download_block" != *"--no-project"* && \
    "$snapshot_block" != *"--with"* && "$snapshot_block" != *"--no-project"* ]] || failed=1
  UV_NO_CACHE=1 UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/vokra-ultravox-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" --self-test
  if (license_preflight "$temporary/missing-approval.json") >/dev/null 2>&1; then
    log "self-test FAIL: missing license approval accepted"
    failed=1
  fi
  printf 'ULTRAVOX_TEST_SENTINEL\n' > "$temporary/sentinel"
  require_pass_marker "$temporary/sentinel" 'ULTRAVOX_TEST_SENTINEL'
  if require_pass_marker "$temporary/sentinel" 'ULTRAVOX_MISSING_SENTINEL' >/dev/null 2>&1; then
    log "self-test FAIL: missing PASS sentinel accepted"
    failed=1
  fi
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nULTRAVOX_TEST_SENTINEL\n' \
    "$TEST_NAME" > "$temporary/test.log"
  require_test_pass "$temporary/test.log" self-test 'ULTRAVOX_TEST_SENTINEL'
  printf 'test %s ... ok\nULTRAVOX_TEST_SENTINEL\n' "$TEST_NAME" > "$temporary/test-missing-result.log"
  if require_test_pass "$temporary/test-missing-result.log" self-test 'ULTRAVOX_TEST_SENTINEL' >/dev/null 2>&1; then
    log "self-test FAIL: missing exact test result accepted"
    failed=1
  fi
  cp "$temporary/test.log" "$temporary/test-duplicate.log"
  printf 'test %s ... ok\n' "$TEST_NAME" >> "$temporary/test-duplicate.log"
  if require_test_pass "$temporary/test-duplicate.log" self-test 'ULTRAVOX_TEST_SENTINEL' >/dev/null 2>&1; then
    log "self-test FAIL: duplicate named test accepted"
    failed=1
  fi
  cp "$temporary/test.log" "$temporary/test-duplicate-result.log"
  grep -F 'test result: ok.' "$temporary/test.log" >> "$temporary/test-duplicate-result.log"
  if require_test_pass "$temporary/test-duplicate-result.log" self-test 'ULTRAVOX_TEST_SENTINEL' >/dev/null 2>&1; then
    log "self-test FAIL: duplicate test result accepted"
    failed=1
  fi
  cp "$temporary/test.log" "$temporary/test-duplicate-marker.log"
  printf 'ULTRAVOX_TEST_SENTINEL\n' >> "$temporary/test-duplicate-marker.log"
  if require_test_pass "$temporary/test-duplicate-marker.log" self-test 'ULTRAVOX_TEST_SENTINEL' >/dev/null 2>&1; then
    log "self-test FAIL: duplicate parity sentinel accepted"
    failed=1
  fi
  fake_root="$temporary/fake-root"
  fakebin="$temporary/fake-bin"
  trace="$temporary/fake-trace"
  mkdir -p "$fake_root/tools/parity/ultravox" "$fakebin"
  cp "$PARITY_PROJECT/uv.lock" "$PARITY_PROJECT/pyproject.toml" "$fake_root/tools/parity/ultravox/"
  cp "$LICENSE_GATE" "$LICENSE_MANIFEST" "$fake_root/tools/parity/ultravox/"
  for tool in mkdir df cargo; do
    # shellcheck disable=SC2016
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" "%s" >> "$VOKRA_FAKE_TRACE"\nexit 97\n' "$tool" > "$fakebin/$tool"
    chmod +x "$fakebin/$tool"
  done
  # shellcheck disable=SC2016
  printf '#!/usr/bin/env bash\nprintf "uv %%s\\n" "$*" >> "$VOKRA_FAKE_TRACE"\nexit 2\n' > "$fakebin/uv"
  chmod +x "$fakebin/uv"
  set +e
  HOME="$temporary/home" VOKRA_ROOT="$fake_root" VOKRA_SCRATCH="$fake_root/scratch" \
    VOKRA_PUBLISH_ON_VAST=1 HF_TOKEN=synthetic VOKRA_FAKE_TRACE="$trace" PATH="$fakebin:$PATH" \
    "$SCRIPT_DIR/run-ultravox-validation.sh" --approval-evidence "$temporary/approval.json" >/dev/null 2>&1
  worker_status=$?
  set -e
  [[ "$worker_status" -eq 2 ]] || { log "self-test FAIL: fake worker exited $worker_status, expected 2"; failed=1; }
  [[ -s "$trace" ]] || { log "self-test FAIL: fake worker did not invoke offline gate"; failed=1; }
  [[ ! -e "$fake_root/scratch" ]] || { log "self-test FAIL: fake worker created host scratch before gate"; failed=1; }
  if grep -Eq '^(mkdir|df|cargo)($| )' "$trace"; then
    log "self-test FAIL: fake worker reached host/tool/model path before gate"
    failed=1
  fi
  if grep -Eiq 'sync|download|huggingface|convert' "$trace"; then
    log "self-test FAIL: fake worker reached dependency/model acquisition before gate"
    failed=1
  fi
  if (( failed != 0 )); then
    die "self-test FAIL"
  fi
  log "self-test PASS"
}

main() {
  local work_dir='' approval='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$work_dir" ]] || { usage; return 2; }
        work_dir="$2"
        shift 2
        ;;
      --approval-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$approval" ]] || { usage; return 2; }
        approval="$2"
        shift 2
        ;;
      --self-test)
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
    [[ -z "$work_dir$approval" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi

  [[ -n "$approval" ]] || { usage; die "--approval-evidence is required"; }
  license_preflight "$approval"
  require_vast_host
  require_tooling
  if [[ -z "$work_dir" ]]; then
    work_dir="$VOKRA_SCRATCH/ultravox-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  require_absent_work_dir "$work_dir" "$approval"
  mkdir -p "$work_dir"

  local evidence_dir="$work_dir/evidence"
  local public_dir="$work_dir/public-vokra"
  local upstream_dir="$work_dir/upstream-ultravox"
  local companion_source_dir="$work_dir/upstream-llama"
  local converted_dir="$work_dir/converted"
  local reference_dir="$evidence_dir/reference"
  local gguf="$public_dir/$PUBLIC_FILE"
  local companion_gguf="$converted_dir/ultravox-llama-companion.gguf"
  mkdir -p "$evidence_dir" "$converted_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Download and authenticate exact public GGUF"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  require_identity "Ultravox public GGUF" "$gguf" "$PUBLIC_BYTES" "$PUBLIC_SHA256"

  step "Download exact official Ultravox and gated Llama snapshots"
  download_ultravox_snapshot "$upstream_dir"
  download_companion_snapshot "$companion_source_dir"

  step "Build Vokra and stream-convert the separately licensed companion"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model ultravox-llama-companion \
    --input "$companion_source_dir/model.safetensors" \
    --config "$companion_source_dir/config.json" \
    --revision "$COMPANION_REVISION" \
    --output "$companion_gguf" \
    2>&1 | tee "$evidence_dir/convert-companion.log"
  [[ -s "$companion_gguf" ]] || die "companion conversion emitted no GGUF"

  step "Install locked official reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Generate independent official FP32 reference"
  VOKRA_REFERENCE_TORCH_THREADS="${VOKRA_REFERENCE_TORCH_THREADS:-8}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
      "$REFERENCE_DUMPER" \
      --ultravox-dir "$upstream_dir" \
      --companion-dir "$companion_source_dir" \
      --output "$reference_dir" \
      --max-new-tokens 4 \
      2>&1 | tee "$evidence_dir/reference.log"

  step "Compare real Vokra CPU frontend, embeddings, logits and greedy IDs"
  env \
    VOKRA_ULTRAVOX_GGUF="$gguf" \
    VOKRA_ULTRAVOX_COMPANION_GGUF="$companion_gguf" \
    VOKRA_ULTRAVOX_COMPANION_GGUF_SHA256="$(sha256_file "$companion_gguf")" \
    VOKRA_ULTRAVOX_REFERENCE_DIR="$reference_dir" \
    VOKRA_ULTRAVOX_BACKEND=cpu \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test ultravox_real \
      "$TEST_NAME" -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity-cpu.log"
  require_test_pass "$evidence_dir/parity-cpu.log" Cpu \
    'ULTRAVOX_PARITY Cpu_vs_official frontend_atol=0.01 audio_embeddings_atol=0.01 logits_atol=0.01 greedy_ids=exact PASS'

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
  cargo deny --manifest-path "$VOKRA_ROOT/Cargo.toml" check licenses advisories bans \
    2>&1 | tee "$evidence_dir/cargo-deny.log"

  step "Cross-check Apple Metal feature compilation"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$evidence_dir/apple-metal-cross-check.log"

  {
    echo "public_gguf_bytes=$(file_bytes "$gguf")"
    echo "public_gguf_sha256=$(sha256_file "$gguf")"
    echo "companion_source_bytes=$(file_bytes "$companion_source_dir/model.safetensors")"
    echo "companion_source_sha256=$(sha256_file "$companion_source_dir/model.safetensors")"
    echo "companion_gguf_bytes=$(file_bytes "$companion_gguf")"
    echo "companion_gguf_sha256=$(sha256_file "$companion_gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
  } > "$evidence_dir/input-hashes.txt"
  {
    printf "scripts/verify/apple-silicon-ultravox.sh \\\n"
    printf "  --gguf '<APPLE_ULTRAVOX_GGUF>' \\\n"
    printf "  --companion '<APPLE_ULTRAVOX_COMPANION_GGUF>' \\\n"
    printf "  --companion-sha256 '%s' \\\n" "$(sha256_file "$companion_gguf")"
    printf "  --reference '<APPLE_ULTRAVOX_REFERENCE>' \\\n"
    printf "  --reference-manifest-sha256 '%s' \\\n" "$(sha256_file "$reference_dir/manifest.txt")"
    printf "  --approval-evidence '<APPLE_ULTRAVOX_APPROVAL_EVIDENCE>' \\\n"
    printf "  --evidence-dir '<APPLE_ULTRAVOX_EVIDENCE_DIR>'\n"
  } > "$evidence_dir/apple-verifier-command.txt"
  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "companion_repo=$COMPANION_REPO"
    echo "companion_revision=$COMPANION_REVISION"
    echo "companion_gguf_sha256=$(sha256_file "$companion_gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "frontend_atol=$FP32_ATOL"
    echo "audio_embeddings_atol=$FP32_ATOL"
    echo "next_logits_atol=$FP32_ATOL"
    echo "greedy_ids=exact"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir; send model data directly to the Apple worker if needed, then destroy VAST"
}

main "$@"
