#!/usr/bin/env bash
# VAST/Linux-only authenticated Transformers API smoke for Qwen3-TTS 0.6B.
# This worker deliberately does not convert, publish, upload, or push anything.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_tts"
API_SMOKE="$PARITY_PROJECT/api_smoke.py"
MODEL_FREE_SMOKE="$PARITY_PROJECT/model_free_api_smoke.py"
SOURCE_COMPAT="$PARITY_PROJECT/qwen_source_compat.py"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
REFERENCE_AUDIO="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

SOURCE_REPOSITORY="QwenLM/Qwen3-TTS"
SOURCE_URL="https://github.com/QwenLM/Qwen3-TTS.git"
SOURCE_REVISION="022e286b98fbec7e1e916cb940cdf532cd9f488e"
MODEL_REPOSITORY="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_REVISION="5d83992436eae1d760afd27aff78a71d676296fc"
DECODER_REPOSITORY="Qwen/Qwen3-TTS-Tokenizer-12Hz"
DECODER_REVISION="a87c50897bb00837eb857d0538b29d117541d7f6"
DECODER_CHECKPOINT_SHA256="836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
TRANSFORMERS_VERSION="5.10.4"
LOCK_SHA256="549809c62df6e2ad37b7494b6b9d9cc18dade54e7b1f19804771787281781ca8"
SECURITY_ADVISORY_STATUS="BLOCKED_SECURITY_ADVISORY"
SECURITY_ADVISORY_REASON="accelerate==1.12.0 is affected by GHSA-4j2p-28q2-5m79; no verified safe real-weight load alternative is available"
MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=100000000

# These are assigned only by the hermetic shell self-test. Production always
# uses the host probes below and cannot spoof the platform or resource gates.
SELFTEST_OS=""
SELFTEST_ARCH=""
SELFTEST_MEM_KIB=""
SELFTEST_FREE_DISK_KIB=""

log() { printf '[qwen3-tts-api-smoke] %s\n' "$*" >&2; }
step() { printf '\n[qwen3-tts-api-smoke] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
host_os() { if [[ -n "$SELFTEST_OS" ]]; then printf '%s\n' "$SELFTEST_OS"; else uname -s; fi; }
host_arch() { if [[ -n "$SELFTEST_ARCH" ]]; then printf '%s\n' "$SELFTEST_ARCH"; else uname -m; fi; }
host_mem_kib() { if [[ -n "$SELFTEST_MEM_KIB" ]]; then printf '%s\n' "$SELFTEST_MEM_KIB"; else awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo; fi; }

nearest_existing_dir() {
  local path="$1"
  [[ "$path" == /* ]] || path="$PWD/$path"
  while [[ ! -d "$path" ]]; do
    [[ "$path" != / ]] || { printf '/\n'; return 0; }
    path="${path%/*}"; [[ -n "$path" ]] || path=/
  done
  (cd -P "$path" && printf '%s\n' "$PWD")
}

free_disk_kib() {
  if [[ -n "$SELFTEST_FREE_DISK_KIB" ]]; then
    printf '%s\n' "$SELFTEST_FREE_DISK_KIB"
  else
    local root
    root="$(nearest_existing_dir "$1")" || return 1
    df -Pk "$root" | awk 'NR == 2 {print $4}'
  fi
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent rest component scan
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent=/
    path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_absent_work_dir() {
  local target="$1" approval="$2" canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "work directory must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize work directory: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$API_SMOKE" "$MODEL_FREE_SMOKE" "$SOURCE_COMPAT" "$LICENSE_GATE" "$LICENSE_MANIFEST" \
    "$PARITY_PROJECT/uv.lock" "$PARITY_PROJECT/pyproject.toml" "$REFERENCE_AUDIO" "$approval"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected path is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected path: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "work directory overlaps protected path: $protected"; return 2; }
  done
  return 0
}

usage() {
  cat >&2 <<'EOF'
usage: run-qwen3-tts-api-smoke.sh --approval-evidence FILE [--work-dir ABSENT_DIR]
       run-qwen3-tts-api-smoke.sh --model-free --variant <0.6b-base|0.6b-customvoice|1.7b-base|1.7b-customvoice|all> \
         --expected-head <40-hex> [--work-dir ABSENT_DIR]
       run-qwen3-tts-api-smoke.sh --self-test

On VAST/Linux this stages the fixed Qwen3-TTS 0.6B-Base and official 12-Hz
decoder snapshots, then calls the official Transformers wrapper with local-only
loading and a two-token greedy request. It emits strict JSON evidence and
never uploads, publishes, or pushes artifacts.

The --model-free phase is independent of owner approval: it stages only exact
source and metadata files, imports the official classes, constructs config and
processor objects, and never calls the wrapper checkpoint loader. Its evidence
is validated immediately by the offline evidence contract, records its exact
variant scope and SHA-256 hand-off identity, keeps source/model/operator
approvals pending, and is not a parity result.
EOF
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; return 2; }
  [[ "$(host_os)" == Linux && "$(host_arch)" == x86_64 ]] || { die 'API smoke requires VAST Linux x86_64'; return 2; }
  mem_kib="$(host_mem_kib)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || { die 'cannot read VAST MemTotal'; return 2; }
  (( mem_kib >= MIN_VAST_MEM_KIB )) || { die 'VAST host has less than 60-GB memory'; return 2; }
  free_kib="$(free_disk_kib "$VOKRA_SCRATCH")"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || { die 'cannot read VAST free disk'; return 2; }
  (( free_kib >= MIN_FREE_DISK_KIB )) || { die 'VAST scratch has less than 100-GB free disk'; return 2; }
}

require_tooling() {
  local tool
  for tool in uv git awk find df grep sed sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { die "required tool missing: $tool"; return 2; }
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || { die 'not a Vokra checkout'; return 2; }
  [[ -f "$API_SMOKE" && ! -L "$API_SMOKE" && -f "$SOURCE_COMPAT" && ! -L "$SOURCE_COMPAT" && -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] || { die 'Qwen3-TTS API smoke inputs are incomplete or symlinked'; return 2; }
  [[ -f "$REFERENCE_AUDIO" && ! -L "$REFERENCE_AUDIO" ]] || { die 'fixed reference audio is missing or symlinked'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
}

require_model_free_tooling() {
  local tool
  for tool in uv git awk find df grep sed sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { die "required tool missing: $tool"; return 2; }
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || { die 'not a Vokra checkout'; return 2; }
  [[ -f "$MODEL_FREE_SMOKE" && ! -L "$MODEL_FREE_SMOKE" && -f "$SOURCE_COMPAT" && ! -L "$SOURCE_COMPAT" ]] || { die 'Qwen3-TTS model-free API smoke inputs are incomplete or symlinked'; return 2; }
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || { die 'Qwen3-TTS model-free project inputs are incomplete or symlinked'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
}

license_gate() {
  local approval="$1"
  local args=(
    --project "$PARITY_PROJECT/pyproject.toml" --lock "$PARITY_PROJECT/uv.lock" --manifest "$LICENSE_MANIFEST"
    --source-revision "$SOURCE_REVISION" --decoder-revision "$DECODER_REVISION"
    --decoder-checkpoint-sha256 "$DECODER_CHECKPOINT_SHA256" --license-evidence "$approval"
  )
  local variant
  for variant in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do
    case "$variant" in
      0.6b-base) args+=(--variant-revision "$variant=5d83992436eae1d760afd27aff78a71d676296fc") ;;
      0.6b-customvoice) args+=(--variant-revision "$variant=85e237c12c027371202489a0ec509ded67b5e4b5") ;;
      1.7b-base) args+=(--variant-revision "$variant=fd4b254389122332181a7c3db7f27e918eec64e3") ;;
      1.7b-customvoice) args+=(--variant-revision "$variant=0c0e3051f131929182e2c023b9537f8b1c68adfe") ;;
    esac
  done
  UV_NO_CACHE=1 UV_CACHE_DIR="${QWEN3_TTS_UV_CACHE_DIR:-/tmp/vokra-qwen3-tts-api-smoke-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${args[@]}"
}

preflight() {
  local approval="$1"
  die "$SECURITY_ADVISORY_STATUS: $SECURITY_ADVISORY_REASON"
  [[ -s "$approval" && ! -L "$approval" ]] || { die 'approval evidence must be a non-empty regular non-symlink file'; return 2; }
  license_gate "$approval"
}

download_snapshot() {
  local repo="$1" revision="$2" output="$3"
  [[ ! -e "$output" ]] || { die "snapshot target must be absent: $output"; return 2; }
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import os,sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], token=os.environ.get("HF_TOKEN") or os.environ.get("HF"), allow_patterns=["LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "model-*.safetensors", "model.safetensors.index.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "speech_tokenizer/**"])' \
    "$repo" "$revision" "$output"
}

download_metadata_snapshot() {
  local repo="$1" revision="$2" output="$3"
  [[ ! -e "$output" ]] || { die "metadata snapshot target must be absent: $output"; return 2; }
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import os,sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], token=os.environ.get("HF_TOKEN") or os.environ.get("HF"), allow_patterns=["config.json", "generation_config.json", "merges.txt", "tokenizer_config.json", "vocab.json"])' \
    "$repo" "$revision" "$output"
}

download_source() {
  local output="$1"
  [[ ! -e "$output" ]] || { die "official source target must be absent: $output"; return 2; }
  mkdir -p "$output"
  git -C "$output" init -q
  git -C "$output" remote add origin "$SOURCE_URL"
  git -C "$output" fetch --depth 1 origin "$SOURCE_REVISION"
  git -C "$output" checkout --detach FETCH_HEAD
  [[ "$(git -C "$output" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'official source revision drifted'
  [[ -f "$output/pyproject.toml" && -f "$output/qwen_tts/__init__.py" ]] || die 'official source tree is incomplete'
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] || die 'official source checkout is dirty'
}

run_self_test() {
  local path_probe worker_probe approval worker_log rc gate_line sync_line download_line validate_line evidence_line failed=0 metadata_block
  local script_path="${BASH_SOURCE[0]}"
  for required in "$SOURCE_REPOSITORY" "$SOURCE_URL" "$SOURCE_REVISION" "$MODEL_REPOSITORY" "$MODEL_REVISION" "$DECODER_REPOSITORY" "$DECODER_REVISION" "$DECODER_CHECKPOINT_SHA256" "$TRANSFORMERS_VERSION" "$LOCK_SHA256" \
    'VOKRA_PUBLISH_ON_VAST=1' 'platform.system()' 'platform.machine()' 'local_files_only=True' 'dtype=float32' 'device_map=cpu' 'Qwen3TTSModel.from_pretrained' \
    'generate_voice_clone' 'max_new_tokens' 'min_new_tokens' 'NO_UPLOAD' 'strict JSON' 'uv sync' 'download_snapshot' 'download_source' 'require_absent_work_dir' '--project' '--manifest' '--license-gate' '--vokra-root' '--approval-evidence' '--validate-evidence' 'variant_scope=' 'evidence_sha256=' 'clean' 'x86_64'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing contract token: $required"; failed=1; }
  done
  grep -Fq -- 'MODEL_FREE_SMOKE=' "$script_path" || { log 'self-test missing model-free worker'; failed=1; }
  grep -Fq -- 'SOURCE_COMPAT=' "$script_path" || { log 'self-test missing shared compatibility adapter'; failed=1; }
  for helper in step require_model_free_tooling require_vast_host require_absent_work_dir download_source download_metadata_snapshot; do
    grep -Eq "^${helper}[[:space:]]*\(\)" "$script_path" || { log "self-test missing helper definition: $helper"; failed=1; }
  done
  metadata_block="$(sed -n '/^download_metadata_snapshot()/,/^}/p' "$script_path")"
  [[ -n "$metadata_block" ]] || { log 'self-test missing metadata-only downloader'; failed=1; }
  if grep -En 'safetensors|model-' <<<"$metadata_block" >/dev/null; then
    log 'self-test found checkpoint patterns in metadata-only downloader'; failed=1
  fi
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log 'self-test found a direct Python or pip invocation'; failed=1
  fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|.*--push)([[:space:]]|$)' "$script_path" | grep -v 'never uploads' >/dev/null; then
    log 'self-test found a publication command'; failed=1
  fi
  # shellcheck disable=SC2016
  validate_line="$(grep -nF -- '--validate-evidence --evidence "$evidence"' "$script_path" | grep -v 'validate_line=' | cut -d: -f1)"
  evidence_line="$(grep -nE '^  evidence_sha=' "$script_path" | cut -d: -f1)"
  [[ "$validate_line" =~ ^[0-9]+$ && "$evidence_line" =~ ^[0-9]+$ && "$validate_line" -lt "$evidence_line" ]] || { log 'self-test evidence validator does not precede hand-off hash'; failed=1; }
  UV_NO_CACHE=1 UV_CACHE_DIR="${QWEN3_TTS_UV_CACHE_DIR:-/tmp/vokra-qwen3-tts-api-smoke-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" --self-test || failed=1
  UV_NO_CACHE=1 UV_CACHE_DIR="${QWEN3_TTS_UV_CACHE_DIR:-/tmp/vokra-qwen3-tts-api-smoke-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$MODEL_FREE_SMOKE" --self-test || failed=1

  set +e
  VOKRA_PUBLISH_ON_VAST=0 uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" \
    --source-dir /private/tmp/qwen3-api-no-source --model-dir /private/tmp/qwen3-api-no-model \
    --decoder-dir /private/tmp/qwen3-api-no-decoder --reference-audio /private/tmp/qwen3-api-no-audio \
    --lock /private/tmp/qwen3-api-no-lock --project /private/tmp/qwen3-api-no-project --manifest /private/tmp/qwen3-api-no-manifest \
    --license-gate /private/tmp/qwen3-api-no-license-gate --vokra-root /private/tmp/qwen3-api-no-vokra \
    --approval-evidence /private/tmp/qwen3-api-no-approval --output /private/tmp/qwen3-api-no-output.json >/dev/null 2>&1
  rc=$?
  set -e
  [[ "$rc" == 2 ]] || { log 'self-test direct Python worker accepted a non-VAST invocation'; failed=1; }

  SELFTEST_OS=Darwin SELFTEST_ARCH=arm64 SELFTEST_MEM_KIB=999999999 SELFTEST_FREE_DISK_KIB=999999999 VOKRA_PUBLISH_ON_VAST=1
  if require_vast_host >/dev/null 2>&1; then log 'self-test accepted non-VAST Darwin/arm64'; failed=1; fi
  SELFTEST_OS=Linux SELFTEST_ARCH=aarch64
  if require_vast_host >/dev/null 2>&1; then log 'self-test accepted non-x86_64 Linux'; failed=1; fi
  SELFTEST_OS=Linux SELFTEST_ARCH=x86_64 SELFTEST_MEM_KIB=60000000 SELFTEST_FREE_DISK_KIB=100000000
  require_vast_host >/dev/null 2>&1 || { log 'self-test rejected a minimum VAST host'; failed=1; }
  SELFTEST_OS='' SELFTEST_ARCH='' SELFTEST_MEM_KIB='' SELFTEST_FREE_DISK_KIB=''

  path_probe="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-api-path-selftest.XXXXXX")"
  approval="$path_probe/approval.json"; printf '{}\n' > "$approval"
  require_absent_work_dir "$path_probe/new-work" "$approval" || failed=1
  mkdir "$path_probe/empty-work"
  if require_absent_work_dir "$path_probe/empty-work" "$approval" >/dev/null 2>&1; then failed=1; fi
  rmdir "$path_probe/empty-work"
  ln -s "$path_probe/missing-work" "$path_probe/link-work"
  if require_absent_work_dir "$path_probe/link-work" "$approval" >/dev/null 2>&1; then failed=1; fi
  rm "$path_probe/link-work"
  mkdir -p "$path_probe/real-parent/child"; ln -s "$path_probe/real-parent" "$path_probe/link-parent"
  if require_absent_work_dir "$path_probe/link-parent/child/new-work" "$approval" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$path_probe/real-parent" "$path_probe/link-parent"
  if require_absent_work_dir "$VOKRA_ROOT/tools" "$approval" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$path_probe/approval.json/child" "$approval" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$path_probe"

  # shellcheck disable=SC2016
  gate_line="$(grep -nF 'preflight "$approval"' "$script_path" | tail -n1 | cut -d: -f1)"
  sync_line="$(grep -n 'uv sync --project' "$script_path" | tail -n1 | cut -d: -f1)"
  # shellcheck disable=SC2016
  download_line="$(grep -nF 'download_snapshot "$MODEL_REPOSITORY"' "$script_path" | tail -n1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ && "$download_line" =~ ^[0-9]+$ && "$gate_line" -lt "$sync_line" && "$gate_line" -lt "$download_line" ]] || { log 'self-test gate ordering is invalid'; failed=1; }

  worker_probe="$(mktemp -d "${TMPDIR:-/tmp}/qwen3-tts-api-worker-selftest.XXXXXX")"
  worker_log="$(mktemp "${TMPDIR:-/tmp}/qwen3-tts-api-worker-log.XXXXXX")"
  printf '{}\n' > "$worker_probe/approval.json"
  mkdir "$worker_probe/work"
  set +e
  VOKRA_ROOT="$worker_probe" VOKRA_SCRATCH="$worker_probe/scratch" VOKRA_PUBLISH_ON_VAST=0 \
    bash "$script_path" --approval-evidence "$worker_probe/approval.json" --work-dir "$worker_probe/work" >"$worker_log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 2 ]] || { log "self-test blocked worker returned $rc"; failed=1; }
  [[ ! -d "$worker_probe/scratch" ]] || { log 'blocked worker created scratch before gates'; failed=1; }
  rm -rf "$worker_probe" "$worker_log"
  (( failed == 0 )) || return 1
  echo 'run-qwen3-tts-api-smoke.sh self-test: PASS'
}

run_model_free() {
  local variant="$1" expected_head="$2" work_dir="$3"
  local selected source_dir snapshot_root evidence smoke_rc evidence_sha
  [[ "$variant" == 0.6b-base || "$variant" == 0.6b-customvoice || "$variant" == 1.7b-base || "$variant" == 1.7b-customvoice || "$variant" == all ]] || { die '--variant is invalid for model-free smoke'; return 2; }
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head must be lowercase 40-hex'; return 2; }
  require_model_free_tooling
  require_vast_host
  [[ -n "$work_dir" ]] || work_dir="$VOKRA_SCRATCH/qwen3-tts-model-free-api-smoke-${expected_head:0:12}"
  require_absent_work_dir "$work_dir" ''
  mkdir -p "$work_dir/evidence"
  selected="$variant"
  export HF_HOME="$work_dir/hf-home" HF_HUB_CACHE="$work_dir/hf-home/hub" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TOKENIZERS_PARALLELISM=false
  source_dir="$work_dir/source-qwen3-tts"
  snapshot_root="$work_dir/metadata"
  evidence="$work_dir/evidence/model-free-api-smoke.json"
  step 'Install the reviewed frozen model-free API smoke environment'
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12
  step "Stage official source $SOURCE_REPOSITORY@$SOURCE_REVISION"
  download_source "$source_dir"
  mkdir -p "$snapshot_root"
  if [[ "$selected" == all ]]; then
    for variant_name in 0.6b-base 0.6b-customvoice 1.7b-base 1.7b-customvoice; do
      local repo revision
      case "$variant_name" in
        0.6b-base) repo='Qwen/Qwen3-TTS-12Hz-0.6B-Base'; revision='5d83992436eae1d760afd27aff78a71d676296fc' ;;
        0.6b-customvoice) repo='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'; revision='85e237c12c027371202489a0ec509ded67b5e4b5' ;;
        1.7b-base) repo='Qwen/Qwen3-TTS-12Hz-1.7B-Base'; revision='fd4b254389122332181a7c3db7f27e918eec64e3' ;;
        1.7b-customvoice) repo='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'; revision='0c0e3051f131929182e2c023b9537f8b1c68adfe' ;;
      esac
      step "Acquire $variant_name metadata only"
      download_metadata_snapshot "$repo" "$revision" "$snapshot_root/$variant_name"
    done
  else
    local repo revision
    case "$selected" in
      0.6b-base) repo="$MODEL_REPOSITORY"; revision="$MODEL_REVISION" ;;
      0.6b-customvoice) repo='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'; revision='85e237c12c027371202489a0ec509ded67b5e4b5' ;;
      1.7b-base) repo='Qwen/Qwen3-TTS-12Hz-1.7B-Base'; revision='fd4b254389122332181a7c3db7f27e918eec64e3' ;;
      1.7b-customvoice) repo='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'; revision='0c0e3051f131929182e2c023b9537f8b1c68adfe' ;;
    esac
    step "Acquire $selected metadata only"
    download_metadata_snapshot "$repo" "$revision" "$snapshot_root/$selected"
  fi
  step 'Run official model-free Transformers API probe'
  set +e
  PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$MODEL_FREE_SMOKE" \
    --vokra-root "$VOKRA_ROOT" --project "$PARITY_PROJECT" --source-dir "$source_dir" \
    --snapshot-root "$snapshot_root" --expected-head "$expected_head" --variant "$selected" --output "$evidence"
  smoke_rc=$?
  set -e
  if [[ "$smoke_rc" == 0 ]]; then
    step 'Validate model-free API smoke evidence before hand-off'
    UV_NO_CACHE=1 UV_CACHE_DIR="${QWEN3_TTS_UV_CACHE_DIR:-/tmp/vokra-qwen3-tts-api-smoke-uv-cache}" \
      uv run --no-cache --no-project --offline --python 3.12 python "$MODEL_FREE_SMOKE" \
      --validate-evidence --evidence "$evidence" --expected-head "$expected_head" --variant "$selected" \
      || { die "model-free API evidence validation failed: $evidence"; return 2; }
  fi
  evidence_sha='UNAVAILABLE'
  if [[ -f "$evidence" && ! -L "$evidence" ]]; then evidence_sha="$(sha256sum "$evidence" | awk '{print $1}')"; fi
  cat > "$work_dir/evidence/summary.txt" <<EOF
schema=vokra-qwen3-tts-model-free-api-summary-v1
status=$([[ "$smoke_rc" == 0 ]] && echo PASS_MODEL_FREE || echo BLOCKED_OR_FAILED)
exit_status=$smoke_rc
expected_head=$expected_head
variant_scope=$selected
evidence_sha256=$evidence_sha
checkpoint_load=NOT_PERFORMED
publication=NO_UPLOAD
approvals=PENDING_OWNER_APPROVAL
EOF
  [[ "$smoke_rc" == 0 ]] || { die "model-free API probe failed (exit=$smoke_rc); evidence=$evidence"; return 2; }
  log "PASS_MODEL_FREE: no checkpoint load and no upload; evidence=$evidence; destroy the VAST instance"
}

main() {
  local approval='' work_dir='' variant='' expected_head='' self_test=0 model_free=0
  local seen_approval=0 seen_work=0 seen_self=0 seen_model_free=0 seen_variant=0 seen_head=0
  while (( $# > 0 )); do
    case "$1" in
      --approval-evidence) (( seen_approval == 0 )) || { die 'duplicate --approval-evidence'; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--approval-evidence requires a path'; return 2; }; approval="$2"; seen_approval=1; shift 2 ;;
      --work-dir) (( seen_work == 0 )) || { die 'duplicate --work-dir'; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--work-dir requires a path'; return 2; }; work_dir="$2"; seen_work=1; shift 2 ;;
      --self-test) (( seen_self == 0 )) || { die 'duplicate --self-test'; return 2; }; self_test=1; seen_self=1; shift ;;
      --model-free) (( seen_model_free == 0 )) || { die 'duplicate --model-free'; return 2; }; model_free=1; seen_model_free=1; shift ;;
      --variant) (( seen_variant == 0 )) || { die 'duplicate --variant'; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--variant requires a value'; return 2; }; variant="$2"; seen_variant=1; shift 2 ;;
      --expected-head) (( seen_head == 0 )) || { die 'duplicate --expected-head'; return 2; }; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--expected-head requires a value'; return 2; }; expected_head="$2"; seen_head=1; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ "$seen_approval" == 0 && "$seen_work" == 0 && "$seen_model_free" == 0 && "$seen_variant" == 0 && "$seen_head" == 0 ]] || { die '--self-test accepts no other arguments'; return 2; }
    run_self_test; return
  fi
  if (( model_free == 1 )); then
    [[ "$seen_approval" == 0 ]] || { die '--model-free does not accept approval evidence'; return 2; }
    [[ "$seen_variant" == 1 && "$seen_head" == 1 ]] || { die '--model-free requires --variant and --expected-head'; return 2; }
    run_model_free "$variant" "$expected_head" "$work_dir"
    return
  fi
  [[ "$seen_variant" == 0 && "$seen_head" == 0 ]] || { die '--variant/--expected-head require --model-free'; return 2; }
  [[ "$seen_approval" == 1 ]] || { usage; die '--approval-evidence is required'; return 2; }
  preflight "$approval"
  require_tooling
  require_vast_host
  [[ -n "$work_dir" ]] || work_dir="$VOKRA_SCRATCH/qwen3-tts-api-smoke-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  require_absent_work_dir "$work_dir" "$approval"
  mkdir -p "$work_dir/evidence"
  export HF_HOME="$work_dir/hf-home" HF_HUB_CACHE="$work_dir/hf-home/hub" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 TOKENIZERS_PARALLELISM=false
  local source_dir="$work_dir/source-qwen3-tts" model_dir="$work_dir/model-0.6b-base" decoder_dir="$work_dir/decoder-12hz" evidence="$work_dir/evidence/api-smoke.json" api_rc
  step() { printf '\n[qwen3-tts-api-smoke] ==== %s ====\n' "$*" >&2; }
  exec > >(tee -a "$work_dir/evidence/run.log") 2>&1
  step 'Install the reviewed frozen API smoke environment'
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12
  step "Stage official source $SOURCE_REPOSITORY@$SOURCE_REVISION"
  download_source "$source_dir"
  step "Stage fixed model $MODEL_REPOSITORY@$MODEL_REVISION"
  download_snapshot "$MODEL_REPOSITORY" "$MODEL_REVISION" "$model_dir"
  step "Stage official decoder $DECODER_REPOSITORY@$DECODER_REVISION"
  download_snapshot "$DECODER_REPOSITORY" "$DECODER_REVISION" "$decoder_dir"
  step 'Run official local-only Transformers API smoke'
  set +e
  PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}" uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$API_SMOKE" \
    --source-dir "$source_dir" --model-dir "$model_dir" --decoder-dir "$decoder_dir" --reference-audio "$REFERENCE_AUDIO" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST" --license-gate "$LICENSE_GATE" \
    --vokra-root "$VOKRA_ROOT" --approval-evidence "$approval" --output "$evidence"
  api_rc=$?
  set -e
  uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" --validate-evidence "$evidence"
  [[ "$api_rc" == 0 ]] || die "official API smoke failed; evidence retained at $evidence"
  (cd "$work_dir" && find evidence -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > evidence/SHA256SUMS)
  log "PASS: strict JSON evidence written to $evidence; publication=NO_UPLOAD"
}

main "$@"
