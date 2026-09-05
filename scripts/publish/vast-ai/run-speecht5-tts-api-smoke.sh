#!/usr/bin/env bash
# VAST-only authenticated SpeechT5 Transformers API smoke.  No upload path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/speecht5_tts"
API_SMOKE="$PARITY_PROJECT/api_smoke.py"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
POST_SYNC_AUDIT="$PARITY_PROJECT/post_sync_audit.py"
COMPACT_AUDIT="$PARITY_PROJECT/dependency_audit_evidence.json"
TTS_PREP="$VOKRA_ROOT/tools/parity/speecht5_tts_prepare_checkpoint.py"
TTS_REVISION="30fcde30f19b87502b8435427b5f5068e401d5f6"
TTS_SOURCE_SHA256="d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"
VOCODER_REVISION="bb6f429406e86a9992357a972c0698b22043307d"
VOCODER_SOURCE_SHA256="b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b"
LOCK_SHA256="418fb6b6516e0284b503ed20872e2dc6dd375aff918e253f3e7f9d27b62f904c"
PYPROJECT_SHA256="1e61ad26749c1ad5ba05fe139ef8bfcf4698e3b030cad6182e18309789779346"
MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=30000000
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[speecht5-api-smoke] %s\n' "$*" >&2; }
step() { printf '\n[speecht5-api-smoke] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

paths_overlap() {
  [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]
}

canonical_absent_path() {
  local target="$1" current suffix component rest real
  [[ "$target" == /* ]] || { die 'path must be absolute'; return 2; }
  rest="${target#/}"; current="/"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"
    if [[ "$rest" == "$component" ]]; then rest=""; else rest="${rest#*/}"; fi
    [[ -z "$component" || "$component" == "." ]] && continue
    [[ "$component" != ".." ]] || { die 'path contains ..'; return 2; }
    current="${current%/}/$component"
    if [[ -L "$current" ]]; then
      real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'path has inaccessible symlink component'; return 2; }
      case "$current:$real" in
        /var:/private/var|/tmp:/private/tmp) current="$real" ;;
        *) die 'path has a symlinked component'; return 2 ;;
      esac
    fi
  done
  current="$target"; suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die 'existing parent is missing or symlinked'; return 2; }
  real="$(cd -P "$current" && pwd)" || { die 'existing parent is inaccessible'; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_dir() {
  local path="$1" label="$2" canonical
  [[ "$path" == /* ]] || { die "$label must be absolute: $path"; return 2; }
  [[ ! -e "$path" && ! -L "$path" ]] || { die "$label must be absent and non-symlink: $path"; return 2; }
  canonical="$(canonical_absent_path "$path")" || return 2
  [[ -n "$canonical" ]] || { die "$label path cannot be canonicalized"; return 2; }
}

require_disjoint_paths() {
  local work="$1" evidence="$2" approval="$3" root_real work_real evidence_real approval_real approval_parent
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" || { die 'checkout is inaccessible'; return 2; }
  work_real="$(canonical_absent_path "$work")" || return 2
  evidence_real="$(canonical_absent_path "$evidence")" || return 2
  approval_parent="$(cd -P "$(dirname "$approval")" 2>/dev/null && pwd)" || { die 'approval parent is inaccessible'; return 2; }
  approval_real="$approval_parent/$(basename "$approval")"
  paths_overlap "$work_real" "$evidence_real" && { die 'work/evidence paths overlap'; return 2; }
  paths_overlap "$work_real" "$approval_real" && { die 'work/approval paths overlap'; return 2; }
  paths_overlap "$evidence_real" "$approval_real" && { die 'evidence/approval paths overlap'; return 2; }
  paths_overlap "$work_real" "$root_real" && { die 'work-dir overlaps checkout'; return 2; }
  paths_overlap "$evidence_real" "$root_real" && { die 'evidence-dir overlaps checkout'; return 2; }
  return 0
}

require_vokra_root_path() {
  local root_real
  [[ "$VOKRA_ROOT" == /* ]] || { die 'VOKRA_ROOT must be absolute'; return 2; }
  root_real="$(canonical_absent_path "$VOKRA_ROOT")" || return 2
  [[ "$root_real" == "$VOKRA_ROOT" ]] || { die 'VOKRA_ROOT must not use a symlinked path'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || { die 'VOKRA_ROOT is not a checkout'; return 2; }
}

require_vast_host() {
  local mem_kib free_kib disk_path="$1"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'API smoke requires Linux x86_64 VAST'; return 2; }
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_VAST_MEM_KIB" ]] || { die "RAM is below ${MIN_VAST_MEM_KIB} KiB"; return 2; }
  while [[ ! -e "$disk_path" && "$disk_path" != / ]]; do disk_path="$(dirname "$disk_path")"; done
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_FREE_DISK_KIB" ]] || { die "free disk is below ${MIN_FREE_DISK_KIB} KiB"; return 2; }
}

require_clean_checkout() {
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || { die 'VOKRA_ROOT is not a checkout'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
}

require_tooling() {
  local tool
  for tool in uv git curl awk df sha256sum find sort tee; do
    command -v "$tool" >/dev/null 2>&1 || { die "required tool missing: $tool"; return 2; }
  done
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] || { die 'SpeechT5 pyproject is missing or symlinked'; return 2; }
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || { die 'SpeechT5 uv.lock is missing or symlinked'; return 2; }
  [[ -f "$API_SMOKE" && ! -L "$API_SMOKE" && -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" && -f "$POST_SYNC_AUDIT" && -f "$COMPACT_AUDIT" && -f "$TTS_PREP" ]] || { die 'SpeechT5 API smoke inputs are incomplete'; return 2; }
}

preflight_gate() {
  local approval="$1"
  [[ "$approval" == /* ]] || { die 'approval evidence must be an absolute path'; return 2; }
  [[ -s "$approval" && -f "$approval" && ! -L "$approval" ]] || { die 'approval evidence must be a non-empty regular non-symlink file'; return 2; }
  step 'Run the existing SpeechT5 license preflight before host, scratch, sync, or download'
  UV_NO_CACHE=1 UV_CACHE_DIR="${SPEECHT5_API_SMOKE_UV_CACHE_DIR:-/private/tmp/vokra-speecht5-api-smoke-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
      --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_sentinel() {
  local evidence="$1" count
  count="$(grep -Ec '^SPEECHT5_API_SMOKE status=PASS publication=NO_UPLOAD revision=[0-9a-f]{40} input_sha256=[0-9a-f]{64} output_sha256=[0-9a-f]{64} call_checkpoint_sha256=[0-9a-f]{64}$' "$evidence" || true)"
  [[ "$count" == 1 ]] || { die 'API smoke sentinel is missing or duplicated'; return 2; }
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp fail=0
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/speecht5-api-smoke-self-test.XXXXXX")"
  trap 'rm -rf "$tmp"' EXIT
  UV_NO_CACHE=1 UV_CACHE_DIR="$tmp/cache" uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" --self-test >/dev/null || fail=1
  for required in "$TTS_REVISION" "$TTS_SOURCE_SHA256" "$VOCODER_REVISION" "$VOCODER_SOURCE_SHA256" \
    "$LOCK_SHA256" "$PYPROJECT_SHA256" 'VOKRA_PUBLISH_ON_VAST=1' 'clean' 'Linux x86_64' \
    'preflight_gate.py' 'run_preflight_gate' '--manifest' '--evidence' '--approval-evidence' '--vokra-root' 'NO_UPLOAD' 'uv sync --project' '--frozen --python 3.12' \
    'post_sync_audit.py' 'SPEECHT5_API_SMOKE status=PASS' 'SPEECHT5_API_SMOKE status=FAIL' 'require_absent_dir' 'require_disjoint_paths'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing contract token: $required"; fail=1; }
  done
  local preflight_call mkdir_line sync_line
  preflight_call="$(grep -nF "  preflight_gate \"\$approval\"" "$script_path" | tail -1 | cut -d: -f1 || true)"
  mkdir_line="$(grep -nF "  mkdir -p \"\$checkpoint\" \"\$controller\"" "$script_path" | tail -1 | cut -d: -f1 || true)"
  sync_line="$(grep -nF '  uv sync --project' "$script_path" | tail -1 | cut -d: -f1 || true)"
  [[ "$preflight_call" =~ ^[0-9]+$ && "$mkdir_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ ]] || fail=1
  (( preflight_call < mkdir_line && mkdir_line < sync_line )) || fail=1
  for bad_args in '--self-test --approval-evidence x' '--self-test --work-dir x' '--unknown x' '--approval-evidence x' '--work-dir x --work-dir y'; do
    # shellcheck disable=SC2086
    if "$script_path" $bad_args >/dev/null 2>&1; then fail=1; fi
  done
  printf '{}\n' > "$tmp/approval.json"
  require_absent_dir "$tmp/work" work-dir || fail=1
  require_absent_dir "$tmp/evidence" evidence-dir || fail=1
  require_disjoint_paths "$tmp/work" "$tmp/evidence" "$tmp/approval.json" || fail=1
  mkdir "$tmp/existing"
  require_absent_dir "$tmp/existing" existing-dir >/dev/null 2>&1 && fail=1 || :
  require_disjoint_paths "$tmp/work" "$tmp/work/evidence" "$tmp/approval.json" >/dev/null 2>&1 && fail=1 || :
  ln -s "$tmp/existing" "$tmp/link-parent"
  require_absent_dir "$tmp/link-parent/new" symlinked-dir >/dev/null 2>&1 && fail=1 || :
  rm "$tmp/link-parent"
  grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null && fail=1 || :
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push' "$script_path" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  printf '%s\n' 'SPEECHT5_API_SMOKE status=PASS publication=NO_UPLOAD revision=30fcde30f19b87502b8435427b5f5068e401d5f6 input_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa output_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb call_checkpoint_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' > "$tmp/sentinel"
  require_sentinel "$tmp/sentinel" || fail=1
  printf '%s\n%s\n' "$(cat "$tmp/sentinel")" "$(cat "$tmp/sentinel")" > "$tmp/duplicate"
  require_sentinel "$tmp/duplicate" >/dev/null 2>&1 && fail=1 || :
  rm -rf "$tmp"; trap - EXIT
  (( fail == 0 )) || return 1
  echo 'run-speecht5-tts-api-smoke.sh self-test: PASS (NO_UPLOAD)'
}

main() {
  local self_test=0 approval='' work='' evidence='' seen_approval=0 seen_work=0 seen_evidence=0
  while (( $# > 0 )); do
    case "$1" in
      --self-test) (( self_test == 0 )) || { die 'duplicate --self-test'; return 2; }; self_test=1; shift ;;
      --approval-evidence) (( seen_approval == 0 )) && [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die 'invalid or duplicate --approval-evidence'; return 2; }; seen_approval=1; approval="$2"; shift 2 ;;
      --work-dir) (( seen_work == 0 )) && [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die 'invalid or duplicate --work-dir'; return 2; }; seen_work=1; work="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) && [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die 'invalid or duplicate --evidence-dir'; return 2; }; seen_evidence=1; evidence="$2"; shift 2 ;;
      -h|--help) cat >&2 <<'EOF'
usage: run-speecht5-tts-api-smoke.sh --approval-evidence FILE --work-dir ABSENT_DIR --evidence-dir ABSENT_DIR
       run-speecht5-tts-api-smoke.sh --self-test

VAST/Linux-only authenticated Transformers 5.10.4 SpeechT5ForTextToSpeech
generate_speech smoke. The worker emits hashed evidence and NO_UPLOAD only.
EOF
        return 0 ;;
      *) die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self_test )); then
    (( seen_approval == 0 && seen_work == 0 && seen_evidence == 0 )) || { die '--self-test accepts no production arguments'; return 2; }
    run_self_test; return $?
  fi
  (( seen_approval && seen_work && seen_evidence )) || { die '--approval-evidence, --work-dir, and --evidence-dir are required'; return 2; }
  require_vokra_root_path
  preflight_gate "$approval"
  require_absent_dir "$work" work-dir
  require_absent_dir "$evidence" evidence-dir
  require_disjoint_paths "$work" "$evidence" "$approval"
  require_vast_host "$(dirname "$work")"
  require_clean_checkout
  require_tooling

  local checkpoint="$work/checkpoint" controller="$work/controller" audit="$work/controller/post-sync-audit.json" api_log="$work/controller-api-smoke.log" env_log="$work/controller-environment.txt"
  mkdir -p "$checkpoint" "$controller"
  export HF_HOME="$work/hf-home" HF_HUB_CACHE="$work/hf-home/hub" UV_CACHE_DIR="$work/uv-cache"
  step 'Synchronize the frozen Python 3.12 project'
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12
  step 'Audit the synchronized dependency closure before model acquisition'
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$POST_SYNC_AUDIT" \
    --compact-evidence "$COMPACT_AUDIT" --output "$audit"
  step "Prepare exact SpeechT5 checkpoint $TTS_REVISION"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$TTS_PREP" --output-dir "$checkpoint"
  step 'Run official generate_speech API smoke'
  set +e
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$API_SMOKE" \
    --checkpoint "$checkpoint" --project-dir "$PARITY_PROJECT" --output-dir "$evidence" \
    --approval-evidence "$approval" --vokra-root "$VOKRA_ROOT" 2>&1 | tee "$api_log"
  local api_rc="${PIPESTATUS[0]}"
  set -e
  if (( api_rc == 0 )); then
    uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" \
      --validate-evidence --output-dir "$evidence" --status PASS
    require_sentinel "$api_log"
  else
    [[ -f "$evidence/evidence.json" && ! -L "$evidence/evidence.json" ]] || { die "API smoke failed without failure evidence (rc=$api_rc)"; return 2; }
    uv run --no-cache --no-project --offline --python 3.12 python "$API_SMOKE" \
      --validate-evidence --output-dir "$evidence" --status FAIL
    printf 'execution_status=FAIL\napi_smoke_exit_code=%s\nevidence_sha256=%s\n' \
      "$api_rc" "$(sha256_file "$evidence/evidence.json")" > "$controller/failure-summary.txt"
    (cd "$evidence" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    log "FAIL: preserved failure evidence at $evidence; destroy the disposable VAST instance"
    return "$api_rc"
  fi
  {
    echo 'execution_status=PASS'
    echo 'publication=NO_UPLOAD'
    echo "upstream_repo=microsoft/speecht5_tts"
    echo "upstream_revision=$TTS_REVISION"
    echo "upstream_checkpoint_sha256=$TTS_SOURCE_SHA256"
    echo "vocoder_revision=$VOCODER_REVISION"
    echo "vocoder_source_sha256=$VOCODER_SOURCE_SHA256"
    echo "lock_sha256=$LOCK_SHA256"
    echo "pyproject_sha256=$PYPROJECT_SHA256"
    echo "evidence_sha256=$(sha256_file "$evidence/evidence.json")"
    cat "$api_log"
  } > "$evidence/summary.txt"
  {
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_status=$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)"
    uname -a
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
  } > "$env_log"
  (cd "$evidence" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  log "PASS: pull only $evidence; destroy the disposable VAST instance; do not pull checkpoint artifacts"
}

main "$@"
