#!/usr/bin/env bash
# VAST/Linux-only Kimi-Audio composite inspection. No conversion, runtime
# validation, upload, or publication is performed by this worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/kimi_audio_7b_instruct_inspect.py"
REPOSITORY="moonshotai/Kimi-Audio-7B-Instruct"
HF_REVISION="9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
SOURCE_URL="https://github.com/MoonshotAI/Kimi-Audio.git"
SOURCE_REVISION="349251e1d8f4f98d58fda59246381faecd7392e0"
GLM4_REVISION="eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a"
GLM4_URL="https://github.com/THUDM/GLM-4-Voice.git"
SOURCE_SUBMODULE="kimia_infer/models/tokenizer/glm4"
INDEX_TOTAL_SIZE=19532673280
INDEX_WEIGHT_MAP_COUNT=453
AUDIO_DETOK_BYTES=19008505142
AUDIO_DETOK_SHA256="cdeeec41e629565439cd8ef807c8a014ad6ce052cce0c259c7bfe3fe6ada3f51"
VOCODER_BYTES=964918850
VOCODER_SHA256="a043a75ae865a9f3264500966a2622399e6b29cf362f4e2134adaefd4ba1252c"
VOCODER_CONFIG_BYTES=1402
WHISPER_BYTES=3087131376
WHISPER_SHA256="d677ab655d1916439c5868c819a0e48cdac574defab83c69b0bbc2b7b31a9f06"
WORK="/workspace/vokra-kimi-audio-7b-instruct-inspection"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((250 * 1024 * 1024))
UV_CACHE_DIR="${KIMI_AUDIO_UV_CACHE_DIR:-/tmp/vokra-kimi-audio-uv-cache}"

log() { printf '[kimi-audio-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() { echo "usage: run-kimi-audio-7b-instruct-inspection.sh [--work-dir DIR] | --self-test"; }

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'moonshotai/Kimi-Audio-7B-Instruct' '9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b' \
    'https://github.com/MoonshotAI/Kimi-Audio.git' '349251e1d8f4f98d58fda59246381faecd7392e0' \
    'https://github.com/THUDM/GLM-4-Voice.git' 'eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a' 'kimia_infer/models/tokenizer/glm4' '128' 'x86_64' 'CARGO_BUILD_JOBS=1' \
    '19532673280' '453' '19008505142' 'cdeeec41e629565439cd8ef807c8a014ad6ce052cce0c259c7bfe3fe6ada3f51' \
    '964918850' 'a043a75ae865a9f3264500966a2622399e6b29cf362f4e2134adaefd4ba1252c' '1402' \
    '3087131376' 'd677ab655d1916439c5868c819a0e48cdac574defab83c69b0bbc2b7b31a9f06' \
    'model.safetensors.index.json' 'model-36-of-36.safetensors' 'audio_detokenizer/model.pt' \
    'vocoder/model.pt' 'whisper-large-v3/model.safetensors' 'weights_only=True' \
    'get_unsafe_globals_in_checkpoint' 'server_tree' 'item.lfs' 'lfs_sha256' 'blob_id' 'path_in_repo' 'resolved_origin' 'fixed_components' 'MATCHED' 'inspection_error' 'status": "BLOCKED"' \
    'evidence_stage' 'INSPECTION_ONLY' 'NO_UPLOAD' 'cargo fmt --all -- --check'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
calls = re.findall(r"list_repo_tree\([^\n]*\)", source)
if not calls:
    raise SystemExit("Kimi-Audio tree walk call missing")
for call in calls:
    if "path_in_repo=" not in call or re.search(r"(?<![A-Za-z0-9_])path=", call):
        raise SystemExit(f"Kimi-Audio tree walk has incompatible path keyword: {call}")
PY
  then
    log 'self-test FAIL: frozen HfApi.list_repo_tree path_in_repo contract regression'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

def classify_entry(entry):
    if isinstance(entry, RepoFolder):
        if getattr(entry, "type", None) not in {None, "directory"}:
            raise RuntimeError("unknown RepoFolder type")
        return "directory"
    if isinstance(entry, RepoFile):
        if getattr(entry, "type", None) not in {None, "file"}:
            raise RuntimeError("unknown RepoFile type")
        return "file"
    raise RuntimeError(f"unknown HF tree entry: {entry!r}")

file_entry = RepoFile(path="README.md", size=1, oid="a" * 40)
file_entry.type = None
assert classify_entry(file_entry) == "file"
folder_entry = RepoFolder(path="nested", oid="b" * 40)
folder_entry.type = None
assert classify_entry(folder_entry) == "directory"
try:
    classify_entry(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unknown HF tree entry was accepted")
print("Kimi-Audio RepoFile/RepoFolder self-test: PASS")
PY
  then
    log 'self-test FAIL: RepoFile/RepoFolder class-identity regression'
    fail=1
  fi
  if grep -En '^[[:space:]]*git[[:space:]]+push|^[[:space:]]*(curl|wget)[^#]*(upload|push)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if grep -Eq '^HF_REVISION=.*\$\{|^SOURCE_REVISION=.*\$\{|^SOURCE_URL=.*\$\{' "$path"; then
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

[[ "$(uname -s)" == Linux ]] || die 'Kimi-Audio inspection requires Linux VAST'
[[ "$(uname -m)" == x86_64 ]] || die 'Kimi-Audio inspection requires x86_64 VAST'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$ROOT/tools/parity/pyproject.toml" && -f "$ROOT/tools/parity/uv.lock" ]] || die 'locked parity project missing'
[[ -f "$INSPECTOR" ]] || die 'Kimi-Audio inspector missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
(( mem_kib >= MIN_MEM_KIB )) || die '128 GiB memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
(( free_kib >= MIN_DISK_KIB )) || die '250 GiB disk guard failed'
for tool in cargo git uv awk find df sha256sum; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/model" "$work_dir/source" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR
{
  echo "status=BLOCKED"
  echo "evidence_stage=INSPECTION_ONLY"
  echo "publication=NO_UPLOAD"
  echo "repository=$REPOSITORY"
  echo "hf_revision=$HF_REVISION"
  echo "source_url=$SOURCE_URL"
  echo "source_revision=$SOURCE_REVISION"
  echo "glm4_revision=$GLM4_REVISION"
  echo "glm4_url=$GLM4_URL"
  echo "index_total_size=$INDEX_TOTAL_SIZE"
  echo "index_weight_map_count=$INDEX_WEIGHT_MAP_COUNT"
  echo "vocoder_config_bytes=$VOCODER_CONFIG_BYTES"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$work_dir/evidence/validation.log" 2>&1

# Build a complete recursive server-tree envelope. Some HF API versions do
# not expand children when `recursive=True`, so directories are walked.
# shellcheck disable=SC2129 # this heredoc intentionally appends to validation evidence
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python - "$work_dir/server_tree.json" "$REPOSITORY" "$HF_REVISION" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder

output, repository, revision = sys.argv[1:]
api = HfApi()
info = api.model_info(repository, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"HF revision resolved {info.sha!r}, expected {revision!r}")
rows = []
def walk(path=""):
    for item in api.list_repo_tree(repository, revision=revision, path_in_repo=path, recursive=False):
        if isinstance(item, RepoFolder):
            if getattr(item, "type", None) not in {None, "directory"}:
                raise RuntimeError(f"invalid RepoFolder type: {item!r}")
            item_type = "directory"
        elif isinstance(item, RepoFile):
            if getattr(item, "type", None) not in {None, "file"}:
                raise RuntimeError(f"invalid RepoFile type: {item!r}")
            item_type = "file"
        else:
            raise RuntimeError(f"unknown HF tree entry type: {type(item).__name__}")
        item_path = getattr(item, "path", None)
        if not item_path:
            raise RuntimeError("HF tree entry has no path")
        if item_type == "directory":
            walk(item_path)
        elif item_type == "file":
            size = getattr(item, "size", None)
            lfs = getattr(item, "lfs", None)
            if isinstance(lfs, dict):
                lfs_sha256 = lfs.get("sha256")
            else:
                lfs_sha256 = getattr(lfs, "sha256", None)
            oid = lfs_sha256 or getattr(item, "blob_id", None) or getattr(item, "oid", None) or getattr(item, "xet_hash", None)
            if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(oid, str) or lfs_sha256 is not None and not isinstance(lfs_sha256, str):
                raise RuntimeError(f"HF tree file has invalid identity fields: {item_path}")
            rows.append({"path": item_path, "type": item_type, "size": size, "oid": oid, "lfs_sha256": lfs_sha256})
        else:
            raise RuntimeError(f"unexpected HF tree entry type: {item_type!r}")
walk()
Path(output).write_text(json.dumps({"repository": repository, "revision": revision, "resolved_revision": info.sha, "files": sorted(rows, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python - "$REPOSITORY" "$HF_REVISION" "$work_dir/model" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))
PY

git clone --recurse-submodules --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" submodule update --init --recursive >> "$work_dir/evidence/validation.log" 2>&1
source_origin="$(git -C "$work_dir/source/repo" remote get-url origin)"
[[ "$source_origin" == "$SOURCE_URL" ]] || die 'source origin mismatch'
source_head="$(git -C "$work_dir/source/repo" rev-parse HEAD)"
[[ "$source_head" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'
glm4_head="$(git -C "$work_dir/source/repo/$SOURCE_SUBMODULE" rev-parse HEAD)"
[[ "$glm4_head" == "$GLM4_REVISION" ]] || die 'GLM-4-Voice submodule revision mismatch'
glm4_origin="$(git -C "$work_dir/source/repo/$SOURCE_SUBMODULE" remote get-url origin)"
[[ "$glm4_origin" == "$GLM4_URL" ]] || die 'GLM-4-Voice submodule origin mismatch'
{
  echo "source_resolved_origin=$source_origin"
  echo "source_resolved_revision=$source_head"
  echo "glm4_resolved_revision=$glm4_head"
  echo "glm4_resolved_origin=$glm4_origin"
} >> "$work_dir/evidence/validation.log"

set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" \
  --snapshot "$work_dir/model" --source "$work_dir/source/repo" --evidence "$work_dir/evidence" \
  --server-tree "$work_dir/server_tree.json" --revision "$HF_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must exit 2, got $inspect_rc"
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'manifest missing'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python - "$work_dir/evidence/manifest.json" "$AUDIO_DETOK_BYTES" "$AUDIO_DETOK_SHA256" \
    "$VOCODER_BYTES" "$VOCODER_SHA256" "$VOCODER_CONFIG_BYTES" "$WHISPER_BYTES" "$WHISPER_SHA256" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json
import re
import sys

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result

manifest_path, audio_bytes, audio_sha, vocoder_bytes, vocoder_sha, config_bytes, whisper_bytes, whisper_sha = sys.argv[1:]
manifest = json.loads(open(manifest_path, encoding="utf-8").read(), object_pairs_hook=reject_duplicate_keys)
if not isinstance(manifest, dict):
    raise SystemExit("manifest root is not an object")
for key, expected in (("status", "BLOCKED"), ("evidence_stage", "INSPECTION_ONLY"), ("publication", "NO_UPLOAD")):
    if manifest.get(key) != expected:
        raise SystemExit(f"manifest {key} mismatch: {manifest.get(key)!r}")
if "inspection_error" in json.dumps(manifest, sort_keys=True).lower():
    raise SystemExit("inspection_error was accepted")

model = manifest.get("model")
tree = model.get("server_tree") if isinstance(model, dict) else None
if not isinstance(tree, dict) or tree.get("status") != "MATCHED":
    raise SystemExit(f"server tree is not MATCHED: {tree!r}")
transport = tree.get("transport_cache")
if not isinstance(transport, dict):
    raise SystemExit("transport cache evidence is missing")
if transport.get("path") != ".cache/huggingface" or transport.get("scope") != "snapshot_root_exact_transport_subtree":
    raise SystemExit(f"transport cache scope mismatch: {transport!r}")
if transport.get("identity_role") != "NON_IDENTITY_TRANSPORT_METADATA":
    raise SystemExit(f"transport cache identity role mismatch: {transport!r}")
if transport.get("status") not in {"ABSENT", "EXCLUDED"} or not isinstance(transport.get("present"), bool):
    raise SystemExit(f"invalid transport cache evidence: {transport!r}")

expected = {
    "audio_detokenizer/model.pt": (int(audio_bytes), audio_sha),
    "vocoder/model.pt": (int(vocoder_bytes), vocoder_sha),
    "vocoder/config.json": (int(config_bytes), None),
    "whisper-large-v3/model.safetensors": (int(whisper_bytes), whisper_sha),
}
fixed = manifest.get("fixed_components")
if not isinstance(fixed, dict) or set(fixed) != set(expected):
    raise SystemExit(f"fixed component paths mismatch: {fixed!r}")
for path, (expected_bytes, expected_sha) in expected.items():
    identity = fixed.get(path)
    if not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}:
        raise SystemExit(f"fixed component identity schema mismatch: {path}: {identity!r}")
    if identity["bytes"] != expected_bytes or not isinstance(identity["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]):
        raise SystemExit(f"fixed component identity mismatch: {path}: {identity!r}")
    if expected_sha is not None and identity["sha256"] != expected_sha:
        raise SystemExit(f"fixed component SHA-256 mismatch: {path}")
print("Kimi-Audio manifest structural assertion: PASS")
PY
echo 'verdict=BLOCKED; evidence_stage=INSPECTION_ONLY; publication=NO_UPLOAD' | tee -a "$work_dir/evidence/validation.log"
die 'Kimi-Audio inspection evidence preserved; conversion/runtime/parity remain blocked'
