#!/usr/bin/env bash
set -euo pipefail

# VAST-only official reference gate.  This script intentionally has no
# conversion/publication path.  A dedicated transitive lock is mandatory
# before any model or source download is attempted.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REFERENCE="$ROOT/tools/parity/cosyvoice3_dump_reference.py"
PROJECT="$ROOT/tools/parity/cosyvoice3_reference"
SOURCE_REVISION="0d990d60740bf174904a5185cce910b847bd3684"
MODEL_REVISION="29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
MATCHA_REVISION="dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
die(){ echo "cosyvoice3-vast: ERROR: $*" >&2; exit 2; }
COSYVOICE3_SELF_TEST_TMP=""
# shellcheck disable=SC2329 # Invoked by the EXIT trap below.
cleanup_self_test() {
  [[ -n "$COSYVOICE3_SELF_TEST_TMP" ]] && rm -rf -- "$COSYVOICE3_SELF_TEST_TMP"
}

require_reference_project() {
  local project="$1"
  [[ -d "$project" && ! -L "$project" ]] || { echo 'reference project directory is missing or symlinked' >&2; return 2; }
  [[ -f "$project/pyproject.toml" && ! -L "$project/pyproject.toml" ]] || { echo 'reference pyproject.toml is missing or symlinked' >&2; return 2; }
  [[ -f "$project/uv.lock" && ! -L "$project/uv.lock" ]] || { echo 'dedicated CosyVoice3 uv.lock is absent' >&2; return 2; }
}

validate_work_path() {
  local work="$1" component rest current parent candidate item
  local -a suffix=()
  [[ "$work" == /* && "$work" != *$'\n'* && "$work" != *$'\r'* ]] || return 2
  [[ "$work" != */../* && "$work" != */.. && "$work" != *'/./'* && "$work" != *'/.' ]] || return 2
  rest="${work#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || return 2
    current="$current/"
  done
  [[ ! -e "$work" && ! -L "$work" ]] || return 2
  parent="$work"
  while [[ ! -e "$parent" ]]; do
    [[ ! -L "$parent" ]] || return 2
    item="${parent##*/}"
    [[ -n "$item" ]] || return 2
    suffix+=("$item")
    [[ "$parent" != / ]] || return 2
    parent="${parent%/*}"
    [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || return 2
  candidate="$(cd -P "$parent" && pwd)"
  for (( item = ${#suffix[@]} - 1; item >= 0; item-- )); do candidate="$candidate/${suffix[item]}"; done
  local root_real project_real
  root_real="$(cd -P "$ROOT" && pwd)" || return 2
  project_real="$(cd -P "$PROJECT" && pwd)" || return 2
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || return 2
  [[ "$candidate" != "$project_real" && "$candidate/" != "$project_real/"* && "$project_real/" != "$candidate/"* ]] || return 2
}

self_test(){
  local fail=0 token tmp fake_project effects fake_bin fake_uname rc
  for token in "$SOURCE_REVISION" "$MODEL_REVISION" "$MATCHA_REVISION" 'AUTHENTICATED_REFERENCE_EVIDENCE' 'REFERENCE_ERROR' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'NO_UPLOAD' 'CausalMaskedDiffWithDiT' 'CausalHiFTGenerator' 'flow_rand_noise_full' 'official_output_pcm' 'prompt_sha256' 'cosyvoice3_validate_reference.py'; do
    grep -Fq -- "$token" "$REFERENCE" "$0" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload|convert' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  tmp="$(mktemp -d)"
  tmp="$(cd -P "$tmp" && pwd)"
  COSYVOICE3_SELF_TEST_TMP="$tmp"
  trap cleanup_self_test EXIT
  fake_project="$tmp/project"
  mkdir -p "$fake_project"
  printf '[project]\nname = "synthetic-missing-lock"\nversion = "0.0.0"\n' >"$fake_project/pyproject.toml"
  if require_reference_project "$fake_project" >/dev/null 2>&1; then
    echo 'self-test accepted a project without uv.lock' >&2; fail=1
  fi
  effects="$tmp/effects"
  validate_work_path "$effects" || { echo 'self-test rejected a safe absent work path' >&2; fail=1; }
  if validate_work_path "$ROOT/child" >/dev/null 2>&1; then
    echo 'self-test accepted work path inside checkout' >&2; fail=1
  fi
  if validate_work_path "$PROJECT/child" >/dev/null 2>&1; then
    echo 'self-test accepted work path inside reference project' >&2; fail=1
  fi
  [[ ! -e "$effects" && ! -e "$fake_project/uv.lock" ]] || { echo 'lock/work side effect observed during blocked preflight' >&2; fail=1; }
  fake_bin="$tmp/bin"
  fake_uname="$fake_bin/uname"
  mkdir -p "$fake_bin"
  cat >"$fake_uname" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  -s) printf 'Linux\n' ;;
  -m) printf 'x86_64\n' ;;
  *) exit 2 ;;
esac
EOF
  chmod +x "$fake_uname"
  set +e
  PATH="$fake_bin:$PATH" VOKRA_PUBLISH_ON_VAST=1 \
    COSYVOICE3_WORK_DIR="$tmp/production-work" COSYVOICE3_UV_CACHE_DIR="$tmp/production-cache" \
    "$0" >"$tmp/production.log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 2 && ! -e "$tmp/production-work" && ! -e "$tmp/production-cache" ]] || {
    echo 'production-shaped lock probe did not stop before work/cache' >&2; fail=1;
  }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --self-test || fail=1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/cosyvoice3_validate_reference.py" --self-test || fail=1
  (( fail == 0 )) || return 1
  echo 'run-cosyvoice3-validation.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 0 ]] || die 'usage: run-cosyvoice3-validation.sh [--self-test]'
require_reference_project "$PROJECT" || die 'dedicated CosyVoice3 pyproject.toml/uv.lock is absent; refuse before host probing or downloads'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
for command in git uv awk find df findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $((128*1024*1024)) ]] || die '128 GiB memory guard failed'
[[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $((32*1024*1024)) ]] || die 'tmpfs disk guard failed'
WORK="${COSYVOICE3_WORK_DIR:-/dev/shm/vokra-cosyvoice3-validation}"
validate_work_path "$WORK" || die 'validation directory must be absent and have no symlinked ancestor'
mkdir -p "$WORK/source" "$WORK/matcha" "$WORK/model" "$WORK/evidence/reference"
export UV_CACHE_DIR="${COSYVOICE3_UV_CACHE_DIR:-/tmp/vokra-cosyvoice3-uv-cache}"
# shellcheck disable=SC2129
git clone --filter=blob:none https://github.com/FunAudioLLM/CosyVoice.git "$WORK/source/repo" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" checkout --detach "$SOURCE_REVISION" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" submodule update --init --recursive >>"$WORK/evidence/validation.log" 2>&1
git clone --filter=blob:none https://github.com/shivammehta25/Matcha-TTS.git "$WORK/matcha/repo" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/matcha/repo" checkout --detach "$MATCHA_REVISION" >>"$WORK/evidence/validation.log" 2>&1
uv run --frozen --project "$PROJECT" --python 3.12 python - "$WORK/model" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
from huggingface_hub import snapshot_download
import sys
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", revision="29e01c4e8d000f4bcd70751be16fa94bf3d85a18", local_dir=sys.argv[1])
PY
uv run --frozen --project "$PROJECT" --python 3.12 python - "$WORK/source/repo/asset/zero_shot_prompt.wav" "$WORK/input.json" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
import hashlib, json, sys
from pathlib import Path
wav=Path(sys.argv[1])
if not wav.is_file(): raise RuntimeError("fixed source prompt missing")
sha=hashlib.sha256(wav.read_bytes()).hexdigest()
json.dump({"target_text":"八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。", "prompt_text":"You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。", "prompt_wav":"asset/zero_shot_prompt.wav", "prompt_sha256":sha, "seed":0}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
PY
set +e
uv run --frozen --project "$PROJECT" --python 3.12 python "$REFERENCE" --source "$WORK/source/repo" --matcha-source "$WORK/matcha/repo" --model-dir "$WORK/model" --input "$WORK/input.json" --output "$WORK/evidence/reference" >>"$WORK/evidence/validation.log" 2>&1
rc=$?
set -e
[[ "$rc" == 0 ]] || die 'official CosyVoice3 reference did not produce evidence'
uv run --frozen --project "$PROJECT" --python 3.12 python "$ROOT/tools/parity/cosyvoice3_validate_reference.py" "$WORK/evidence/reference/manifest.json" >>"$WORK/evidence/validation.log" 2>&1
echo 'run-cosyvoice3-validation.sh: authenticated official reference evidence; native route remains blocked' >&2
