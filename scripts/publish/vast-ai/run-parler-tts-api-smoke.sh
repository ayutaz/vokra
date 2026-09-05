#!/usr/bin/env bash
# Authenticate the fixed Parler-TTS Transformers API route on disposable VAST.
# This worker never runs Vokra, converts, uploads, or publishes a model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/parler_tts"
API_SMOKE="$PARITY_PROJECT/api_smoke.py"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
SOURCE_REPOSITORY="https://github.com/huggingface/parler-tts.git"
SOURCE_REVISION="d108732cd57788ec86bc857d99a6cabd66663d68"
TRANSFORMERS_VERSION="5.10.4"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

MIN_VAST_MEM_KIB=30000000
MIN_FREE_DISK_KIB=60000000

log() { printf '[parler-api-smoke-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-parler-tts-api-smoke.sh \
  --approval-evidence <nonempty-regular-file> \
  --work-dir <absent-absolute-directory> \
  --evidence-dir <absent-absolute-directory>
       run-parler-tts-api-smoke.sh --self-test

Runs the exact English and Multilingual Parler-TTS checkpoints through the
official ParlerTTSForConditionalGeneration API on Linux x86_64 VAST. The
locked Transformers 5.10.4 project and fixed source/model revisions are
authenticated before each local-only model load. Evidence is strict JSON and
records revision, lock, package, input, output, and API-call checkpoint hashes.
Publication is always NO_UPLOAD. The work and evidence directories are
created only after all gate checks pass; destroy the disposable VAST instance
after pulling the evidence.
EOF
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'
  fi
}

require_regular_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or symlinked: $path"
}

canonical_candidate() {
  local value="$1" suffix='' parent
  [[ "$value" = /* ]] || { die 'path must be absolute'; return 2; }
  value="${value%/}"
  [[ -n "$value" ]] || { die 'path is empty'; return 2; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    if [[ -L "$parent" ]]; then
      case "$parent:$(cd -P "$parent" 2>/dev/null && pwd)" in
        /var:/private/var|/tmp:/private/tmp) ;;
        *) die "path contains symlink ancestor: $parent"; return 2 ;;
      esac
    fi
    parent="$(dirname "$parent")"
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die 'path has no canonical parent'; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die 'path parent is not a real directory'; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

canonical_existing_path() {
  local value="$1" parent
  [[ "$value" = /* ]] || { die 'path must be absolute'; return 2; }
  [[ -e "$value" && ! -L "$value" ]] || { die 'path must be an existing non-symlink'; return 2; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    if [[ -L "$parent" ]]; then
      case "$parent:$(cd -P "$parent" 2>/dev/null && pwd)" in
        /var:/private/var|/tmp:/private/tmp) ;;
        *) die "path contains symlink ancestor: $parent"; return 2 ;;
      esac
    fi
    parent="$(dirname "$parent")"
  done
  if [[ -d "$value" ]]; then
    (cd -P "$value" && printf '%s\n' "$PWD")
  else
    parent="$(dirname "$value")"
    (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$value")")
  fi
}

paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

validate_paths() {
  local work="$1" evidence="$2" approval="$3" root_real work_real evidence_real approval_real
  [[ "$work" = /* && "$evidence" = /* ]] || { die '--work-dir and --evidence-dir must be absolute'; return 2; }
  [[ ! -e "$work" && ! -L "$work" ]] || { die '--work-dir must be absent'; return 2; }
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || { die '--evidence-dir must be absent'; return 2; }
  work_real="$(canonical_candidate "$work")" || return 2
  evidence_real="$(canonical_candidate "$evidence")" || return 2
  root_real="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  require_regular_file 'approval evidence' "$approval"
  approval_real="$(canonical_existing_path "$approval")" || return 2
  paths_overlap "$work_real" "$evidence_real" && { die 'work and evidence directories overlap'; return 2; }
  paths_overlap "$work_real" "$root_real" && { die 'work directory overlaps checkout'; return 2; }
  paths_overlap "$evidence_real" "$root_real" && { die 'evidence directory overlaps checkout'; return 2; }
  paths_overlap "$work_real" "$approval_real" && { die 'work directory overlaps approval evidence'; return 2; }
  paths_overlap "$evidence_real" "$approval_real" && { die 'evidence directory overlaps approval evidence'; return 2; }
  return 0
}

require_vast_host() {
  local mem_kib free_kib disk_path
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'Linux x86_64 VAST is required'; return 2; }
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_VAST_MEM_KIB" ]] || { die 'RAM is below the 30-GB VAST guard'; return 2; }
  disk_path="$VOKRA_ROOT"
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_FREE_DISK_KIB" ]] || { die 'free disk is below the 60-GB VAST guard'; return 2; }
}

preflight_gate() {
  local approval="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
    "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$MANIFEST" \
    --evidence "$approval"
}

require_tooling_and_clean_checkout() {
  local tool
  for tool in uv curl git awk df find; do command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"; done
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || die 'sha256sum or shasum is required'
  [[ -d "$VOKRA_ROOT/.git" && -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" && -f "$MANIFEST" && -f "$PREFLIGHT_GATE" && -f "$API_SMOKE" ]] || die 'Parler API smoke project is incomplete'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
}

download_hf_file() {
  local repo="$1" revision="$2" name="$3" output="$4"
  curl --fail --location --retry 5 --retry-delay 2 \
    --output "$output" "https://huggingface.co/$repo/resolve/$revision/$name?download=true"
}

run_self_test() (
  local self="${BASH_SOURCE[0]}" tmp fail=0 fake_root fake_uv
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  [[ -f "$API_SMOKE" && ! -L "$API_SMOKE" ]] || { log 'self-test FAIL: API smoke Python worker missing'; fail=1; }
  for required in "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_VERSION" \
    'ParlerTTSForConditionalGeneration' 'local_files_only=True' 'model.generate' 'audio_encoder.decode' 'NO_UPLOAD' \
    'load_checkpoint_sha256' 'package_rows_sha256' 'pre_call_sha256' 'post_call_sha256' \
    'preflight_gate.py' 'PREFLIGHT_GATE' '--preflight-gate' '--vokra-root' '--validate-evidence' 'schema":"v1' 'scope_sha256' 'manifest_sha256' 'pyproject_sha256' 'signer' \
    'VOKRA_PUBLISH_ON_VAST=1' 'uv sync --project' '--frozen --python 3.12' \
    'validate_paths' 'canonical_candidate' 'canonical_existing_path' 'work and evidence directories overlap' '--contract-only'; do
    grep -Fq -- "$required" "$self" "$API_SMOKE" || { log "self-test FAIL: missing contract $required"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*publish-one\.sh|.*upload\.sh|--push([[:space:]]|$))' "$self" >/dev/null; then
    log 'self-test FAIL: publication operation found'; fail=1
  fi
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$self" >/dev/null; then
    log 'self-test FAIL: direct Python/pip invocation found'; fail=1
  fi
  if ! UV_CACHE_DIR="$tmp/uv-cache" uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" --self-test >/dev/null 2>&1; then
    log 'self-test FAIL: offline API smoke Python self-test failed'; fail=1
  fi
  printf '%s\n' '{"schema":"v1","decision":"APPROVED","scope_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","manifest_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","lock_sha256":"0b37648f20d26197ba4a5dbeac5e6336b57454b5f7d2306dd1ddcbf321952bac","pyproject_sha256":"bea3b5f3c5e83b7af88e37a156a3ac8df2eccc5a1883a5daa229eecd080f3a1e","signer":"self-test","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}' > "$tmp/approval"
  validate_paths "$tmp/work" "$tmp/evidence" "$tmp/approval" || { log 'self-test FAIL: valid disjoint paths rejected'; fail=1; }
  mkdir "$tmp/existing-work"
  if validate_paths "$tmp/existing-work" "$tmp/evidence-2" "$tmp/approval" >/dev/null 2>&1; then log 'self-test FAIL: existing work directory accepted'; fail=1; fi
  if validate_paths "$tmp/work-2" "$tmp/work-2/evidence" "$tmp/approval" >/dev/null 2>&1; then log 'self-test FAIL: overlapping work/evidence accepted'; fail=1; fi
  ln -s "$tmp" "$tmp/symlink-parent"
  if validate_paths "$tmp/symlink-parent/work" "$tmp/evidence-3" "$tmp/approval" >/dev/null 2>&1; then log 'self-test FAIL: symlinked work ancestor accepted'; fail=1; fi
  if [[ -e "$tmp/missing-model" ]]; then log 'self-test FAIL: unexpected model fixture exists'; fail=1; fi
  fake_root="$tmp/root"; fake_uv="$tmp/uv"
  mkdir -p "$fake_root/tools/parity/parler_tts"
  cp "$API_SMOKE" "$fake_root/tools/parity/parler_tts/"
  printf '#!/usr/bin/env bash\nexit 99\n' > "$fake_uv"; chmod +x "$fake_uv"
  if VOKRA_ROOT="$fake_root" PATH="$tmp:$PATH" VOKRA_PUBLISH_ON_VAST=0 "$self" --approval-evidence "$tmp/missing" --work-dir "$tmp/work" --evidence-dir "$tmp/evidence" >/dev/null 2>&1; then
    log 'self-test FAIL: non-VAST production path was accepted'; fail=1
  fi
  if (( fail == 0 )); then log 'self-test PASS'; return 0; fi
  return 1
)

main() {
  local self_test=0 approval='' work='' evidence='' seen_approval=0 seen_work=0 seen_evidence=0
  while (( $# > 0 )); do
    case "$1" in
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a value'; seen_approval=1; approval="$2"; shift 2 ;;
      --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--work-dir requires a value'; seen_work=1; work="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a value'; seen_evidence=1; evidence="$2"; shift 2 ;;
      --self-test) (( self_test == 0 )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ $seen_approval == 0 && $seen_work == 0 && $seen_evidence == 0 ]] || die '--self-test accepts no other arguments'
    run_self_test; return $?
  fi
  [[ $seen_approval == 1 && $seen_work == 1 && $seen_evidence == 1 ]] || { usage; die 'all production arguments are required'; }
  # Every check below is before scratch/evidence creation, dependency sync, or model acquisition.
  require_tooling_and_clean_checkout
  preflight_gate "$approval"
  validate_paths "$work" "$evidence" "$approval"
  require_vast_host
  uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" \
    --contract-only --project "$PARITY_PROJECT" --manifest "$MANIFEST" \
    --vokra-root "$VOKRA_ROOT" --preflight-gate "$PREFLIGHT_GATE" --operator-evidence "$approval"
  mkdir -p "$work/source" "$work/models/english" "$work/models/multilingual" "$evidence"
  work="$(cd -P "$work" && pwd)"; evidence="$(cd -P "$evidence" && pwd)"
  export UV_CACHE_DIR="$work/uv-cache"
  git -C "$work/source" init -q
  git -C "$work/source" remote add origin "$SOURCE_REPOSITORY"
  git -C "$work/source" fetch --depth 1 origin "$SOURCE_REVISION"
  git -C "$work/source" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$work/source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'official Parler source revision mismatch'
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12
  download_hf_file "parler-tts/parler-tts-mini-v1" "0392b9451a601e528fd863bbb0598431fee810d9" model.safetensors "$work/models/english/model.safetensors"
  download_hf_file "parler-tts/parler-tts-mini-v1" "0392b9451a601e528fd863bbb0598431fee810d9" config.json "$work/models/english/config.json"
  download_hf_file "parler-tts/parler-tts-mini-v1" "0392b9451a601e528fd863bbb0598431fee810d9" generation_config.json "$work/models/english/generation_config.json"
  download_hf_file "parler-tts/parler-tts-mini-multilingual-v1.1" "11b27d57855dec1ce0914ba1f12363bf2ea75ba3" model.safetensors "$work/models/multilingual/model.safetensors"
  download_hf_file "parler-tts/parler-tts-mini-multilingual-v1.1" "11b27d57855dec1ce0914ba1f12363bf2ea75ba3" config.json "$work/models/multilingual/config.json"
  download_hf_file "parler-tts/parler-tts-mini-multilingual-v1.1" "11b27d57855dec1ce0914ba1f12363bf2ea75ba3" generation_config.json "$work/models/multilingual/generation_config.json"
  for variant in english multilingual; do
    PARLER_TTS_SOURCE_DIR="$work/source" PYTHONPATH="$work/source${PYTHONPATH:+:$PYTHONPATH}" \
      uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$API_SMOKE" \
      --variant "$variant" --project "$PARITY_PROJECT" --manifest "$MANIFEST" \
      --source-dir "$work/source" --model-dir "$work/models/$variant" \
      --output "$evidence/$variant.json" --vokra-root "$VOKRA_ROOT" --preflight-gate "$PREFLIGHT_GATE" --operator-evidence "$approval"
    uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" \
      --validate-evidence "$evidence/$variant.json" --vokra-root "$VOKRA_ROOT" --preflight-gate "$PREFLIGHT_GATE" \
      --operator-evidence "$approval"
  done
  log "PASS: API smoke evidence is in $evidence; publication=NO_UPLOAD; destroy the VAST instance"
}

main "$@"
