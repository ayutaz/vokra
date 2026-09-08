#!/usr/bin/env bash
# Generate the authenticated SBV2 JP-Extra Japanese G2P contract.
#
# This worker is intentionally separate from the four-checkpoint SBV2 parity
# worker.  It downloads and executes only the pinned upstream ``text`` source;
# no model/checkpoint bytes, Cargo command, or publication/upload is allowed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
GENERATOR="$VOKRA_ROOT/tools/parity/sbv2_jp_extra/generate_contract.py"
PYOPENJTALK_AUDIT="$VOKRA_ROOT/tools/parity/sbv2_jp_extra/audit_pyopenjtalk.py"
PARITY_PROJECT="${SBV2_G2P_PROJECT:-$VOKRA_ROOT/tools/parity/sbv2}"
SOURCE_URL="https://github.com/litagin02/Style-Bert-VITS2.git"
SOURCE_COMMIT="ef93f388fc1ddf0dc0f598126c1964923f1df94f"
PYOPENJTALK_SOURCE_URL="https://github.com/r9y9/pyopenjtalk.git"
PYOPENJTALK_COMMIT="0f0fc44e782a8134cd9a51d80b57b48a7c95bb80"
PYOPENJTALK_TAG="v0.4.1"
OPEN_JTALK_DICT_URL="https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz"
OPEN_JTALK_DICT_ARCHIVE_NAME="open_jtalk_dic_utf_8-1.11.tar.gz"
LOGURU_SOURCE_URL="https://github.com/Delgan/loguru.git"
LOGURU_COMMIT="ae3bfd1b85b6b4a3db535f69b975687c79498be4"
LOGURU_TAG="0.7.3"
LOGURU_TAG_OBJECT="eb27ef8546577adbb88ad36b62b4eca9e9dae217"

log() { printf '[sbv2-jp-extra-g2p] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: run-sbv2-jp-extra-g2p-contract.sh --expected-head <40-lower-hex> \
  --output <absent-absolute-json> \
  [--work-dir <absent-absolute-dir>]
       run-sbv2-jp-extra-g2p-contract.sh --self-test

VAST-only, model-free official Style-Bert-VITS2 Japanese G2P contract worker.
Vocabulary/tone dimensions are derived from authenticated source symbols;
model compatibility is checked separately by the strict binder/parity worker.
Publication is always NO_UPLOAD.
EOF
}

require_vast() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux ]] || { die 'source acquisition/execution is Linux/VAST-only'; return 2; }
}

require_tooling() {
  local tool
  for tool in git uv sha256sum curl tar findmnt; do
    command -v "$tool" >/dev/null 2>&1 || { die "required tool missing: $tool"; return 2; }
  done
  [[ -f "$GENERATOR" && ! -L "$GENERATOR" ]] || { die 'contract generator is missing or symlinked'; return 2; }
  [[ -f "$PYOPENJTALK_AUDIT" && ! -L "$PYOPENJTALK_AUDIT" ]] || { die 'pyopenjtalk license audit is missing or symlinked'; return 2; }
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || { die 'pinned SBV2 reference uv.lock is missing'; return 2; }
}

reject_dot_components() {
  local path="$1"
  [[ "$path" == /* ]] || { die "path must be absolute: $path"; return 2; }
  case "$path" in
    */./*|*/../*|*/.|*/..) die "path contains dot components: $path"; return 2 ;;
  esac
}

reject_symlink_ancestors() {
  local path="$1" current=/ rest component
  reject_dot_components "$path"
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "path has symlink ancestor: $path"; return 2; }
    current="$current/"
  done
}

require_absent_path() {
  local path="$1" label="$2"
  reject_symlink_ancestors "$path"
  [[ ! -e "$path" && ! -L "$path" ]] || { die "$label must be absent: $path"; return 2; }
  [[ -d "${path%/*}" && ! -L "${path%/*}" ]] || { die "$label parent must be a real directory: ${path%/*}"; return 2; }
}

require_exec_mount_parent() {
  local path="$1" parent options
  parent="${path%/*}"
  [[ -d "$parent" && ! -L "$parent" ]] || { die "worker directory parent must be a real directory: $parent"; return 2; }
  options="$(findmnt -T "$parent" -n -o OPTIONS 2>/dev/null)" || {
    die "cannot inspect worker directory parent mount options: $parent"
    return 2
  }
  [[ -n "$options" ]] || { die "worker directory parent mount options are missing: $parent"; return 2; }
  case "$options" in
    *$'\n'*|*$'\r'*|*[[:space:]]*) die "worker directory parent mount options are ambiguous: $parent"; return 2 ;;
  esac
  [[ "$options" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]] || {
    die "worker directory parent mount options are malformed: $parent"
    return 2
  }
  case ",$options," in
    *,noexec,*) die "worker directory parent mount is noexec; choose an exec-capable work directory: $parent"; return 2 ;;
  esac
}

require_expected_head() {
  local expected="$1" actual status
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { die 'expected HEAD must be lowercase 40-hex'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || { die 'cannot inspect Vokra HEAD'; return 2; }
  [[ "$actual" == "$expected" ]] || { die "Vokra HEAD mismatch: $actual != $expected"; return 2; }
  status="$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)"
  [[ -z "$status" ]] || { die 'Vokra checkout must be clean'; return 2; }
}

run_self_test() {
  local failed=0 token fixture fixture_root guard_body
  UV_CACHE_DIR="${UV_CACHE_DIR:-$VOKRA_ROOT/.cache/uv-sbv2-jp-extra}" \
    uv run --no-project --offline --python 3.12 python "$GENERATOR" --self-test \
    || failed=1
  for token in \
    "$SOURCE_URL" "$SOURCE_COMMIT" 'VOKRA_PUBLISH_ON_VAST=1' 'Linux/VAST-only' \
    '--expected-head' '--output' '--work-dir' \
    '--project' 'SBV2_G2P_PROJECT' \
    'PYOPENJTALK_SOURCE_URL' 'PYOPENJTALK_COMMIT' 'PYOPENJTALK_TAG' 'audit_pyopenjtalk.py' \
    'pyopenjtalk license audit' 'build-constraint-dependencies' \
    'LICENSE_mei_normal.htsvoice' 'submodule update' \
    'OPEN_JTALK_DICT_URL' 'OPEN_JTALK_DICT_ARCHIVE_NAME' 'open_jtalk_dic_utf_8-1.11.tar.gz' \
    'fe6ba0e43542cef98339abdffd903e062008ea170b04e7e2a35da805902f382a' '23646843' \
    'DICTIONARY_ARCHIVE_PASS' 'DICTIONARY_PAYLOAD_PASS' 'OPEN_JTALK_DICT_DIR' 'curl' 'tar -xzf' \
    'UV_PROJECT_ENVIRONMENT' 'sbv2-venv' \
    'LOGURU_SOURCE_URL' 'LOGURU_COMMIT' 'LOGURU_TAG' 'LOGURU_TAG_OBJECT' 'loguru-source' '--loguru-source-dir' 'verify_loguru_source' '5285f420ff222526f9afa7acf507362367132f9c' '4ea6eb8e860bee2582875b19ceac328ac17dc7af' \
    'HF_HUB_OFFLINE=1' 'TRANSFORMERS_OFFLINE=1' \
    'NO_UPLOAD' 'git clone' 'git checkout' 'verify_source_tree' \
    'model/checkpoint bytes' 'model_weight_acquisition' 'cargo' \
    'g2p_en' 'distance' 'num2words' '__vokra_num2words_sentinel__' 'numeric-text G2P' \
    'forbidden GPL frontend dependency' findmnt; do
    grep -Fq -- "$token" "${BASH_SOURCE[0]}" || { log "self-test missing contract token: $token"; failed=1; }
  done
  for token in symbols_blob japanese_blob mora_blob common_log_blob stdout_wrapper_blob init_blob license_blob; do
    grep -Fq -- "$token" "$GENERATOR" || { log "self-test missing authenticated blob token: $token"; failed=1; }
  done
  for token in deberta_config_blob deberta_special_tokens_blob deberta_tokenizer_config_blob deberta_vocab_blob \
    'bert/deberta-v2-large-japanese-char-wwm/config.json' \
    'bert/deberta-v2-large-japanese-char-wwm/special_tokens_map.json' \
    'bert/deberta-v2-large-japanese-char-wwm/tokenizer_config.json' \
    'bert/deberta-v2-large-japanese-char-wwm/vocab.txt' \
    '9fb6b0ac2ec49b6556e58b5ed9492eb33166714d' \
    'a8b3208c2884c4efb86e49300fdd3dc877220cdf' \
    '8ab2175580e45760875557201e5543019ca3039b' \
    'ef3652a1877f4c898e6fcb3e605c432c7bcc56b1'; do
    grep -Fq -- "$token" "$GENERATOR" || { log "self-test missing authenticated tokenizer token: $token"; failed=1; }
  done
  for token in PYOPENJTALK_COMMIT SOURCE_BLOBS 'pyopenjtalk/__init__.py' '656c5089f529150828b5b6fe512b0ca942d9a3a8' \
    OPEN_JTALK_DICT_BASE_URL \
    OPEN_JTALK_DICT_FILES OPEN_JTALK_DICT_ROOT verify_dictionary_archive verify_dictionary_directory \
    SUBMODULES BUILD_CONSTRAINTS UPSTREAM_BUILD_REQUIREMENTS \
    STATIC_SOURCE_LOCK_LICENSE_PASS POST_INSTALL_PAYLOAD_PASS 'residual=build-only archive hashes'; do
    grep -Fq -- "$token" "$PYOPENJTALK_AUDIT" || { log "self-test missing pyopenjtalk evidence token: $token"; failed=1; }
  done
  grep -Fq -- 'build-constraint-dependencies' "$PARITY_PROJECT/pyproject.toml" || {
    log 'self-test missing uv build constraint configuration'
    failed=1
  }
  UV_CACHE_DIR="${UV_CACHE_DIR:-$VOKRA_ROOT/.cache/uv-sbv2-jp-extra}" \
    uv run --no-project --offline --python 3.12 python "$PYOPENJTALK_AUDIT" --self-test \
    || failed=1
  static_line="$(grep -n -- '--phase static' "${BASH_SOURCE[0]}" | head -n 1 | cut -d: -f1)"
  post_line="$(grep -n -- '--phase post' "${BASH_SOURCE[0]}" | head -n 1 | cut -d: -f1)"
  generator_line="$(grep -n -- 'generate_contract.py' "${BASH_SOURCE[0]}" | tail -n 1 | cut -d: -f1)"
  [[ -n "$static_line" && -n "$post_line" && -n "$generator_line" && "$static_line" -lt "$post_line" && "$post_line" -lt "$generator_line" ]] || {
    log 'self-test phase ordering is not static-audit, project-build/post-audit, generator'
    failed=1
  }
  if grep -En '(^|[[:space:]])(python3?|pip)([[:space:]]|$)' "${BASH_SOURCE[0]}" | grep -v 'uv run' >/dev/null; then
    log 'self-test found forbidden bare Python/pip invocation'
    failed=1
  fi
  if grep -En '(^|[[:space:]])(cargo|hf|huggingface-cli)([[:space:]]|$)|--push|upload\.sh|publish-one\.sh' "${BASH_SOURCE[0]}" | grep -v 'grep -En' >/dev/null; then
    log 'self-test found forbidden Cargo/model/upload command'
    failed=1
  fi
  if "${BASH_SOURCE[0]}" --self-test --self-test >/dev/null 2>&1; then
    log 'self-test accepted duplicate --self-test'
    failed=1
  fi
  fixture_root="$(cd -P "${TMPDIR:-/tmp}" && pwd -P)" || {
    log 'self-test temporary directory root is unavailable'
    return 1
  }
  fixture="$(mktemp -d "$fixture_root/vokra-sbv2-exec-guard.XXXXXX")"
  guard_body="$(awk '/^require_exec_mount_parent\(\) \{/{capture=1} capture {print} capture && /^\}/{exit}' "${BASH_SOURCE[0]}")"
  if ! bash -c '
    set -euo pipefail
    findmnt() { printf "%s\n" "rw,exec"; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture"; then
    log 'self-test rejected an exec-capable mount'
    failed=1
  fi
  if bash -c '
    set -euo pipefail
    findmnt() { printf "%s\n" "rw,noexec"; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture" >/dev/null 2>&1; then
    log 'self-test accepted a noexec mount'
    failed=1
  fi
  if bash -c '
    set -euo pipefail
    findmnt() { return 0; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture" >/dev/null 2>&1; then
    log 'self-test accepted missing mount options'
    failed=1
  fi
  if bash -c '
    set -euo pipefail
    findmnt() { printf "%s\n%s\n" "rw,exec" "rw,exec"; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture" >/dev/null 2>&1; then
    log 'self-test accepted multiple mount option lines'
    failed=1
  fi
  if ! bash -c '
    set -euo pipefail
    findmnt() { printf "%s\n" "rw,notnoexec"; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture" >/dev/null 2>&1; then
    log 'self-test rejected a non-exact noexec token'
    failed=1
  fi
  if bash -c '
    set -euo pipefail
    findmnt() { printf "%s\n" "rw,,exec"; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture" >/dev/null 2>&1; then
    log 'self-test accepted an ambiguous mount option list'
    failed=1
  fi
  if ! bash -c '
    set -euo pipefail
    findmnt() { printf "%s\n" "rw,relatime,nouserxattr"; }
    '"$guard_body"'
    require_exec_mount_parent "$1/child"
  ' bash "$fixture" >/dev/null 2>&1; then
    log 'self-test rejected an overlay mount without noexec'
    failed=1
  fi
  if [[ "$fixture" == "$fixture_root"/vokra-sbv2-exec-guard.* && -d "$fixture" && ! -L "$fixture" ]]; then
    rmdir "$fixture"
  else
    log 'self-test fixture cleanup guard failed; preserving fixture'
    failed=1
  fi
  local unsafe_mktemp_option='-u'
  if grep -Fq -- "mktemp $unsafe_mktemp_option" "$0"; then
    log 'self-test found unsafe unclaimed temporary workdir allocation'
    failed=1
  fi
  grep -Fq -- 'mktemp -d -p /tmp sbv2-jp-extra-g2p.XXXXXX' "$0" || {
    log 'self-test missing atomic default workdir claim'
    failed=1
  }
  grep -Fq -- 'work_claimed=1' "$0" || {
    log 'self-test missing default workdir claim state'
    failed=1
  }
  grep -Fq -- 'trap - EXIT' "$0" || {
    log 'self-test missing claim-preserving trap release'
    failed=1
  }
  [[ "$failed" == 0 ]] || return 1
  echo 'run-sbv2-jp-extra-g2p-contract.sh self-test: PASS'
}

main() {
  local self_test=0 expected_head='' output='' work='' arg source_dir pyopenjtalk_source_dir loguru_source_dir dictionary_archive dictionary_dir project_env work_claimed=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test) [[ "$self_test" == 0 ]] || { die 'duplicate --self-test'; return 2; }; self_test=1; shift ;;
      --expected-head) [[ -z "$expected_head" && $# -ge 2 ]] || { die 'duplicate or missing --expected-head'; return 2; }; expected_head="$2"; shift 2 ;;
      --output) [[ -z "$output" && $# -ge 2 ]] || { die 'duplicate or missing --output'; return 2; }; output="$2"; shift 2 ;;
      --work-dir) [[ -z "$work" && $# -ge 2 ]] || { die 'duplicate or missing --work-dir'; return 2; }; work="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ "$self_test" == 1 ]]; then
    [[ -z "$expected_head$output$work" ]] || { die '--self-test accepts no execution arguments'; return 2; }
    run_self_test
    return $?
  fi
  [[ -n "$expected_head" && -n "$output" ]] || { usage; die 'expected HEAD and output are required'; return 2; }
  require_vast
  require_tooling
  require_expected_head "$expected_head"
  if [[ -z "$work" ]]; then
    require_exec_mount_parent /tmp/sbv2-jp-extra-g2p-parent-probe
    work="$(mktemp -d -p /tmp sbv2-jp-extra-g2p.XXXXXX)" || { die 'cannot atomically claim default worker directory'; return 2; }
    chmod 700 "$work" || { rmdir "$work" 2>/dev/null || true; die 'cannot secure default worker directory'; return 2; }
    work_claimed=1
    cleanup_claimed_work() { [[ "$work_claimed" == 1 && -d "$work" && ! -L "$work" ]] && rmdir "$work" 2>/dev/null || true; }
    trap cleanup_claimed_work EXIT
  else
    require_absent_path "$work" 'worker directory'
    require_exec_mount_parent "$work"
  fi
  require_absent_path "$output" 'contract output'
  require_absent_path "$output.sha256" 'contract SHA-256 sidecar'
  [[ "$output" != "$work"/* && "$work" != "$output"/* ]] || { die 'output and worker directory must be disjoint'; return 2; }
  if [[ "$work_claimed" == 0 ]]; then
    mkdir -m 700 "$work"
  else
    trap - EXIT
  fi
  source_dir="$work/official-source"
  pyopenjtalk_source_dir="$work/pyopenjtalk-source"
  loguru_source_dir="$work/loguru-source"
  dictionary_archive="$work/$OPEN_JTALK_DICT_ARCHIVE_NAME"
  dictionary_dir="$work/open_jtalk_dic_utf_8-1.11"
  project_env="$work/sbv2-venv"
  git clone --quiet --no-checkout "$SOURCE_URL" "$source_dir"
  git -C "$source_dir" fetch --quiet --depth=1 origin "$SOURCE_COMMIT"
  git -C "$source_dir" checkout --quiet --detach "$SOURCE_COMMIT"
  log "verified source checkout at $SOURCE_COMMIT before official import"
  git clone --quiet --no-checkout "$PYOPENJTALK_SOURCE_URL" "$pyopenjtalk_source_dir"
  git -C "$pyopenjtalk_source_dir" fetch --quiet --depth=1 origin "$PYOPENJTALK_COMMIT"
  git -C "$pyopenjtalk_source_dir" fetch --quiet --depth=1 origin "refs/tags/$PYOPENJTALK_TAG:refs/tags/$PYOPENJTALK_TAG"
  git -C "$pyopenjtalk_source_dir" checkout --quiet --detach "$PYOPENJTALK_COMMIT"
  git -C "$pyopenjtalk_source_dir" submodule update --quiet --init --recursive
  log "verified pyopenjtalk source checkout at $PYOPENJTALK_COMMIT before dependency audit"
  git clone --quiet --no-checkout "$LOGURU_SOURCE_URL" "$loguru_source_dir"
  git -C "$loguru_source_dir" fetch --quiet --depth=1 origin "$LOGURU_COMMIT"
  git -C "$loguru_source_dir" fetch --quiet --depth=1 origin "refs/tags/$LOGURU_TAG:refs/tags/$LOGURU_TAG"
  git -C "$loguru_source_dir" checkout --quiet --detach "$LOGURU_COMMIT"
  log "verified loguru source checkout at $LOGURU_COMMIT before dependency audit"
  curl --fail --location --silent --show-error --output "$dictionary_archive" "$OPEN_JTALK_DICT_URL"
  log "downloaded fixed Open JTalk dictionary archive before audit"
  VOKRA_PUBLISH_ON_VAST=1 UV_CACHE_DIR="${UV_CACHE_DIR:-$work/uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$PYOPENJTALK_AUDIT" --phase static \
    --project-dir "$PARITY_PROJECT" --source-dir "$pyopenjtalk_source_dir" --loguru-source-dir "$loguru_source_dir" \
    --dictionary-archive "$dictionary_archive"
  tar -xzf "$dictionary_archive" -C "$work"
  VOKRA_PUBLISH_ON_VAST=1 UV_PROJECT_ENVIRONMENT="$project_env" UV_CACHE_DIR="${UV_CACHE_DIR:-$work/uv-cache}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PYOPENJTALK_AUDIT" --phase post \
    --project-dir "$PARITY_PROJECT" --source-dir "$pyopenjtalk_source_dir" --loguru-source-dir "$loguru_source_dir" \
    --dictionary-archive "$dictionary_archive" --dictionary-dir "$dictionary_dir"
  VOKRA_PUBLISH_ON_VAST=1 UV_PROJECT_ENVIRONMENT="$project_env" OPEN_JTALK_DICT_DIR="$dictionary_dir" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_CACHE_DIR="${UV_CACHE_DIR:-$work/uv-cache}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$GENERATOR" \
    --vokra-root "$VOKRA_ROOT" --expected-head "$expected_head" \
    --source-dir "$source_dir" \
    --work-dir "$work" --output "$output"
  VOKRA_PUBLISH_ON_VAST=1 UV_PROJECT_ENVIRONMENT="$project_env" UV_CACHE_DIR="${UV_CACHE_DIR:-$work/uv-cache}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PYOPENJTALK_AUDIT" --phase post \
    --project-dir "$PARITY_PROJECT" --source-dir "$pyopenjtalk_source_dir" --loguru-source-dir "$loguru_source_dir" \
    --dictionary-archive "$dictionary_archive" --dictionary-dir "$dictionary_dir"
  log 'contract generation complete; publication=NO_UPLOAD; model weights=NOT_ACQUIRED'
  return 0
}

main "$@"
