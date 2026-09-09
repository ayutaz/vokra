#!/usr/bin/env bash
# VAST/Linux-only model-free primary-source audit for Fun-CosyVoice3.
# This never acquires model weights and never creates a Python lock.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
AUDIT="$ROOT/tools/parity/cosyvoice3_source_audit.py"
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'
SOURCE_REV='0d990d60740bf174904a5185cce910b847bd3684'
MATCHA_URL='https://github.com/shivammehta25/Matcha-TTS.git'
MATCHA_REV='dd9105b34bf2be2230f4aa1e4769fb586a3c824e'
WORK="${COSYVOICE3_SOURCE_AUDIT_WORK_DIR:-/dev/shm/vokra-cosyvoice3-source-audit}"
die(){ printf '[cosyvoice3-source-audit] ERROR: %s\n' "$*" >&2; exit 2; }

self_test(){
  local token fail=0
  for token in "$SOURCE_REV" "$MATCHA_REV" 'librosa==0.10.2' 'soxr>=0.3.2' 'BLOCKED_FORBIDDEN_SOXR_CLOSURE' 'NOT_ACQUIRED' 'NO_UPLOAD' 'cosyvoice3_source_audit.py'; do
    grep -Fq -- "$token" "$AUDIT" "$0" || { printf 'missing contract: %s\n' "$token" >&2; fail=1; }
  done
  if grep -En 'snapshot_download|model_dir|checkpoint|publish-one|upload|git[[:space:]]+push' "$0" | grep -v 'grep -En' >/dev/null; then
    echo 'source audit worker contains model/publication operation' >&2; fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDIT" --self-test || fail=1
  ((fail == 0)) || return 1
  echo 'audit-cosyvoice3-source.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 0 ]] || die 'usage: audit-cosyvoice3-source.sh [--self-test]'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required as explicit remote-audit guard'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for command in git uv; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
[[ ! -e "$WORK" && ! -L "$WORK" ]] || die 'work directory must be absent'
parent="$(dirname "$WORK")"
[[ "$(findmnt -T "$parent" -n -o FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
mkdir -p "$WORK/source" "$WORK/matcha" "$WORK/evidence"
git clone --filter=blob:none "$SOURCE_URL" "$WORK/source/CosyVoice" >>"$WORK/evidence/audit.log" 2>&1
git -C "$WORK/source/CosyVoice" checkout --detach "$SOURCE_REV" >>"$WORK/evidence/audit.log" 2>&1
git clone --filter=blob:none "$MATCHA_URL" "$WORK/matcha/Matcha-TTS" >>"$WORK/evidence/audit.log" 2>&1
git -C "$WORK/matcha/Matcha-TTS" checkout --detach "$MATCHA_REV" >>"$WORK/evidence/audit.log" 2>&1
UV_NO_CACHE=1 uv run --no-cache --no-project --python 3.12 python "$AUDIT" \
  --source "$WORK/source/CosyVoice" \
  --matcha-source "$WORK/matcha/Matcha-TTS" \
  --output "$WORK/evidence/source-dependency-audit.json" >>"$WORK/evidence/audit.log" 2>&1
echo 'audit-cosyvoice3-source.sh: source/dependency facts authenticated; complete composite remains blocked by forbidden soxr' >&2
exit 2
