#!/usr/bin/env bash
# VAST/Linux-only Kyutai TTS composite inspection. This worker never
# converts, publishes, uploads, or claims runtime/parity support.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/kyutai_tts_1_6b_en_fr_inspect.py"
REPOSITORY="kyutai/tts-1.6b-en_fr"
HF_REVISION="f65439609986c392cb12df63938abcc550c3fb15"
HF_ARTIFACT_BYTES=4068484990
HF_TOTAL_BYTES=4068492850
MOSHI_URL="https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION="e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
DSM_URL="https://github.com/kyutai-labs/delayed-streams-modeling.git"
DSM_REVISION="4c4f65e147df056adf3346290d64c7b9649b18c9"
VOICE_REPOSITORY="kyutai/tts-voices"
VOICE_REVISION="323332d33f997de8394f24a193e1a76df720e01a"
VOICE_FILE="voice-donations/robert.wav.1e68beda@240.safetensors"
TTS_BYTES=3683719712
TTS_SHA256="726ddadd90a080c89cbc6b217745296ef32d8e25666d30f81a09e8ae5c9e0f0c"
MIMI_BYTES=384644900
MIMI_SHA256="09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
SPM_BYTES=120378
SPM_SHA256="cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69"
VOICE_BYTES=256136
VOICE_SHA256="bc79b0162c94862aadd6c5d351b5b4984274af0616e3a56b0df9973ff7c793c7"
WORK="/workspace/vokra-kyutai-tts-1-6b-en-fr-inspection"
MIN_MEM_KIB=$((64 * 1024 * 1024))
MIN_DISK_KIB=$((40 * 1024 * 1024))
UV_CACHE_DIR="${KYUTAI_TTS_UV_CACHE_DIR:-/tmp/vokra-kyutai-tts-uv-cache}"

log() { printf '[kyutai-tts-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() { echo 'usage: run-kyutai-tts-1-6b-en-fr-inspection.sh [--work-dir DIR] | --self-test'; }

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'kyutai/tts-1.6b-en_fr' 'f65439609986c392cb12df63938abcc550c3fb15' \
    'moshi.git' 'e6a55d2722a65870ef52a6c9f6ecfc0e90f38362' \
    'delayed-streams-modeling.git' '4c4f65e147df056adf3346290d64c7b9649b18c9' \
    'kyutai/tts-voices' '323332d33f997de8394f24a193e1a76df720e01a' \
    'dsm_tts_1e68beda@240.safetensors' 'tokenizer-e351c8d8-checkpoint125.safetensors' \
    'tokenizer_spm_8k_en_fr_audio.model' '.gitattributes' 'README.md' 'config.json' 'voice-donations/robert.wav.1e68beda@240.safetensors' \
    '4068484990' '4068492850' '3683719712' '384644900' '120378' '256136' '726ddadd90a080c89cbc6b217745296ef32d8e25666d30f81a09e8ae5c9e0f0c' \
    '09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50' \
    'cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69' \
    'bc79b0162c94862aadd6c5d351b5b4984274af0616e3a56b0df9973ff7c793c7' \
    'model_info' 'list_repo_tree' 'path_in_repo' 'lfs_sha256' 'git_blob_sha1' 'weights_only=True' \
    '64' '40' 'CARGO_BUILD_JOBS=1' 'status": "BLOCKED"' 'evidence_stage' 'NO_UPLOAD' 'exit 2'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
    python - "$path" <<'PY'
import inspect
import sys
from pathlib import Path

from huggingface_hub import HfApi

parameters = inspect.signature(HfApi.list_repo_tree).parameters
if "path_in_repo" not in parameters or "path" in parameters:
    raise SystemExit(f"unexpected frozen HfApi.list_repo_tree signature: {parameters}")
source = Path(sys.argv[1]).read_text(encoding="utf-8")
tree_calls = [line for line in source.splitlines() if "for item in api.list_repo_tree" in line]
if not any("path_in_repo=path" in line for line in tree_calls):
    raise SystemExit("Kyutai tree walk does not use path_in_repo")
if any(("path=" + "path") in line for line in tree_calls):
    raise SystemExit("Kyutai tree walk still uses removed path keyword")
PY
  then
    log 'self-test FAIL: frozen HfApi.list_repo_tree contract regression'
    fail=1
  fi
  if grep -En '^[[:space:]]*git[[:space:]]+push|^[[:space:]]*(curl|wget)[^#]*(upload|push)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if grep -Eq "^(HF_REVISION|MOSHI_REVISION|DSM_REVISION|VOICE_REVISION)=.*\\\${" "$path"; then
    log 'self-test FAIL: fixed identity is operator-overridable'
    fail=1
  fi
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="$WORK"
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$work_dir" == "$WORK" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'inspection requires Linux VAST'
[[ "$(uname -m)" == x86_64 ]] || die 'inspection requires x86_64 VAST'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$ROOT/tools/parity/pyproject.toml" && -f "$ROOT/tools/parity/uv.lock" ]] || die 'locked parity project missing'
[[ -f "$INSPECTOR" ]] || die 'inspector missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
(( mem_kib >= MIN_MEM_KIB )) || die '64 GiB memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
(( free_kib >= MIN_DISK_KIB )) || die '40 GiB disk guard failed'
for tool in cargo git uv awk find df sha256sum; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/model" "$work_dir/voices" "$work_dir/source" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR
{
  echo 'status=BLOCKED'
  echo 'evidence_stage=INSPECTION_ONLY'
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'publication=NO_UPLOAD'
  echo "hf_artifact_bytes=$HF_ARTIFACT_BYTES hf_total_bytes=$HF_TOTAL_BYTES"
  echo "tts_bytes=$TTS_BYTES tts_sha256=$TTS_SHA256"
  echo "mimi_bytes=$MIMI_BYTES mimi_sha256=$MIMI_SHA256"
  echo "spm_bytes=$SPM_BYTES spm_sha256=$SPM_SHA256"
  echo "voice_bytes=$VOICE_BYTES voice_sha256=$VOICE_SHA256"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$work_dir/evidence/validation.log" 2>&1

emit_tree() {
  local output="$1" repository="$2" revision="$3" patterns="$4"
  # shellcheck disable=SC2129
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
    python - "$output" "$repository" "$revision" "$patterns" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json, sys
from pathlib import Path
from huggingface_hub import HfApi
output, repository, revision, pattern_text = sys.argv[1:]
patterns = set(filter(None, pattern_text.split("\n")))
api = HfApi()
info = api.model_info(repository, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"resolved revision {info.sha!r} != {revision!r}")
rows = []
pending, visited = [""], set()
while pending:
    path = pending.pop()
    if path in visited:
        continue
    visited.add(path)
    for item in api.list_repo_tree(repository, revision=revision, path_in_repo=path, recursive=False):
        item_path = getattr(item, "path", None)
        item_type = getattr(item, "type", None)
        if not isinstance(item_path, str) or item_type not in {"file", "directory"}:
            raise RuntimeError(f"invalid HF tree entry: {item!r}")
        if item_type == "directory":
            pending.append(item_path)
            continue
        if patterns and item_path not in patterns:
            continue
        lfs = getattr(item, "lfs", None)
        if isinstance(lfs, dict):
            lfs_sha256 = lfs.get("sha256")
        else:
            lfs_sha256 = getattr(lfs, "sha256", None)
        git_blob_sha1 = getattr(item, "blob_id", None) or getattr(item, "oid", None)
        size = getattr(item, "size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(git_blob_sha1, str) or not isinstance(lfs_sha256, (str, type(None))):
            raise RuntimeError(f"invalid HF file identity: {item_path}")
        rows.append({"path": item_path, "type": "file", "size": size, "git_blob_sha1": git_blob_sha1, "lfs_sha256": lfs_sha256})
paths = [row["path"] for row in rows]
if len(paths) != len(set(paths)):
    raise RuntimeError("duplicate HF tree path")
Path(output).write_text(json.dumps({"repository": repository, "revision": revision, "resolved_revision": info.sha, "files": sorted(rows, key=lambda row: row["path"])}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

emit_tree "$work_dir/server_tree.json" "$REPOSITORY" "$HF_REVISION" ''
emit_tree "$work_dir/voice_server_tree.json" "$VOICE_REPOSITORY" "$VOICE_REVISION" $'README.md\n'$VOICE_FILE
# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python - "$REPOSITORY" "$HF_REVISION" "$work_dir/model" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))
PY
# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python - "$VOICE_REPOSITORY" "$VOICE_REVISION" "$work_dir/voices" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["README.md", "voice-donations/robert.wav.1e68beda@240.safetensors"]))
PY
git clone --filter=blob:none --no-checkout "$MOSHI_URL" "$work_dir/source/moshi" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/moshi" checkout --detach "$MOSHI_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
git clone --filter=blob:none --no-checkout "$DSM_URL" "$work_dir/source/delayed-streams-modeling" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/delayed-streams-modeling" checkout --detach "$DSM_REVISION" >> "$work_dir/evidence/validation.log" 2>&1

set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" \
  --snapshot "$work_dir/model" --server-tree "$work_dir/server_tree.json" \
  --voice-snapshot "$work_dir/voices" --voice-server-tree "$work_dir/voice_server_tree.json" \
  --source "$work_dir/source/moshi" --dsm-source "$work_dir/source/delayed-streams-modeling" \
  --evidence "$work_dir/evidence" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must exit 2, got $inspect_rc"
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'manifest missing'
grep -Fq '"status": "BLOCKED"' "$work_dir/evidence/manifest.json" || die 'blocked status missing'
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$work_dir/evidence/manifest.json" || die 'inspection stage missing'
grep -Fq '"publication": "NO_UPLOAD"' "$work_dir/evidence/manifest.json" || die 'publication status missing'
grep -Fq '"runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED"' "$work_dir/evidence/manifest.json" || die 'runtime blocker missing'
grep -Fq '"cpu_status": "UNSUPPORTED"' "$work_dir/evidence/manifest.json" || die 'CPU status missing'
grep -Fq '"metal_status": "BLOCKED_BY_CPU"' "$work_dir/evidence/manifest.json" || die 'Metal status missing'
grep -Fq '"parity_status": "NOT_RUN"' "$work_dir/evidence/manifest.json" || die 'parity status missing'
echo 'verdict=BLOCKED; evidence_stage=INSPECTION_ONLY; publication=NO_UPLOAD' | tee -a "$work_dir/evidence/validation.log"
die 'Kyutai TTS inspection evidence preserved; native runtime/parity remain blocked'
