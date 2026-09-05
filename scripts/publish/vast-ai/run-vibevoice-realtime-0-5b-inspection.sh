#!/usr/bin/env bash
# VAST-only VibeVoice Realtime 0.5B inspection. Never converts, uploads, or publishes.
set -euo pipefail

HF_REPOSITORY="microsoft/VibeVoice-Realtime-0.5B"
HF_REVISION="6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
SOURCE_REPOSITORY="https://github.com/microsoft/VibeVoice.git"
SOURCE_REVISION="94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers.git"
TRANSFORMERS_TAG="v4.51.3"
TRANSFORMERS_REVISION="5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
TOKENIZER_REPOSITORY="Qwen/Qwen2.5-0.5B"
TOKENIZER_REVISION="060db6499f32faf8b98477b0a26969ef7d8b9987"
INSPECTOR="tools/parity/vibevoice_realtime_0_5b_inspect.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_MEM_KIB=$((64 * 1024 * 1024))
MIN_DISK_KIB=$((16 * 1024 * 1024))

die() { echo "run-vibevoice-realtime-0-5b-inspection: $*" >&2; exit 2; }
self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 required status
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  for required in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_TAG" "$TRANSFORMERS_REVISION" "$TOKENIZER_REPOSITORY" "$TOKENIZER_REVISION" "$INSPECTOR" "MODEL_FILES" "PARAMETERS" "BF16" "MAX_HEADER_BYTES" "INSPECTION_ONLY" "BLOCKED" "NO_UPLOAD" "local_dir" ".cache/huggingface" "requested_revision" "recursive_file_only" "RepoFolder" "isinstance(item, RepoFolder)" "expand=True" "re.fullmatch" "git_blob_sha1" "lfs_pointer_git_blob_sha1" "lfs_sha256" "companion-server-tree" "SOURCE_ROLE_BLOBS" "TRANSFORMERS_ROLE_BLOBS" "vibevoice/modular/modular_vibevoice_text_tokenizer.py" "inspection_status" "AUTHENTICATED_EVIDENCE_COMPLETE" "INSPECTION_ERROR" "model_weights" "NOT_DOWNLOADED"; do
    if ! grep -Fq -- "$required" "$self" && ! grep -Fq -- "$required" "$root/$INSPECTOR"; then echo "self-test FAIL: missing $required" >&2; fail=1; fi
  done
  for required in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'findmnt' 'git status --porcelain --untracked-files=all' 'snapshot_download' 'model_info' 'CARGO_BUILD_JOBS'; do
    if ! grep -Fq -- "$required" "$self"; then echo "self-test FAIL: missing VAST gate $required" >&2; fail=1; fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: mutation/conversion/Cargo found" >&2; fail=1; fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: raw Python/pip found" >&2; fail=1; fi
  if bash "$self" --self-test --work-dir /tmp/vibevoice-realtime-self-test >/dev/null 2>&1; then echo "self-test FAIL: extra argument accepted" >&2; fail=1; else status=$?; [[ "$status" == 2 ]] || { echo "self-test FAIL: expected exit 2, got $status" >&2; fail=1; }; fi
  (( fail == 0 )) && echo "run-vibevoice-realtime-0-5b-inspection.sh self-test: OK" || return 1
}

work_dir="/dev/shm/vokra-vibevoice-realtime-0-5b-inspection"; self=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self=1; shift;;
    --work-dir) [[ $# -ge 2 ]] || die "--work-dir requires path"; work_dir="$2"; shift 2;;
    -h|--help) echo "usage: $0 [--work-dir TMPFS] | --self-test"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
if (( self == 1 )); then [[ "$work_dir" == "/dev/shm/vokra-vibevoice-realtime-0-5b-inspection" ]] || die "--self-test accepts no other arguments"; self_test; exit $?; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"; [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
parent="$(dirname "$work_dir")"; [[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die "work path is not a directory"; [[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work path is not empty"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 64 GiB"
free="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_DISK_KIB" ]] || die "tmpfs below 16 GiB"
for command in git uv sha256sum findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; export CARGO_BUILD_JOBS=1
cache="$work_dir/cache"; model="$work_dir/model"; companion="$work_dir/companion"; model_tree="$work_dir/model-tree.json"; companion_tree="$work_dir/companion-tree.json"; snapshot_file="$work_dir/snapshot"; companion_file="$work_dir/companion-path"; source="$work_dir/source"; transformers="$work_dir/transformers"; evidence="$work_dir/evidence"; mkdir -p "$cache" "$model" "$companion" "$evidence"
"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$TOKENIZER_REPOSITORY" "$TOKENIZER_REVISION" "$cache" "$model" "$companion" "$snapshot_file" "$companion_file" "$model_tree" "$companion_tree" <<'PY'
import hashlib, json, os, re, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download
from tools.parity.vibevoice_realtime_0_5b_inspect import select_server_rows

model_repo, model_rev, tok_repo, tok_rev, cache, model_dir, tok_dir, model_out, tok_out, model_packet, tok_packet = sys.argv[1:]
api = HfApi()
model_selected = {".gitattributes", "README.md", "config.json", "figures/Fig1.png", "model.safetensors", "preprocessor_config.json"}
tokenizer_selected = {"LICENSE", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"}
def fetch(repo, rev, destination, patterns, selected):
    info = api.model_info(repo_id=repo, revision=rev)
    if info.sha != rev: raise SystemExit(f"HF revision drift: {repo} {info.sha} != {rev}")
    snapshot = Path(snapshot_download(repo_id=repo, revision=rev, cache_dir=cache, local_dir=destination, allow_patterns=patterns, token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
    if snapshot.resolve() != Path(destination).resolve(): raise SystemExit(f"local_dir mismatch: {snapshot} != {destination}")
    all_rows = []; seen = set()
    for item in api.list_repo_tree(repo_id=repo, revision=rev, recursive=True, expand=True):
        if isinstance(item, RepoFolder): continue
        if not isinstance(item, RepoFile): raise SystemExit(f"unknown HF tree entry type: {type(item).__name__}")
        if not isinstance(item.path, str) or not item.path or "\\" in item.path or "\x00" in item.path or item.path.startswith("/") or ".." in item.path.split("/"): raise SystemExit(f"unsafe HF file path: {item.path!r}")
        if item.path in seen: raise SystemExit(f"duplicate HF file path: {item.path}")
        seen.add(item.path)
        lfs = getattr(item, "lfs", None)
        lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        lfs_oid = lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "oid", None)
        lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        git_id = getattr(item, "blob_id", None)
        if not isinstance(item.size, int) or isinstance(item.size, bool) or item.size < 0 or not isinstance(git_id, str) or not re.fullmatch(r"[0-9a-f]{40}", git_id): raise SystemExit(f"incomplete Git identity: {item}")
        if lfs_sha is not None:
            if not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha) or (lfs_oid is not None and (not isinstance(lfs_oid, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_oid) or lfs_oid != lfs_sha)) or not isinstance(lfs_size, int) or isinstance(lfs_size, bool) or lfs_size < 0 or lfs_size != item.size: raise SystemExit(f"incomplete LFS identity/size: {item}")
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {lfs_size}\n".encode()
            pointer_id = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
            if pointer_id != git_id: raise SystemExit(f"canonical LFS pointer Git blob mismatch: {item.path}")
        elif lfs_size is not None:
            raise SystemExit(f"regular entry unexpectedly has LFS size: {item.path}")
        all_rows.append({"path": item.path, "type": "file", "size": item.size, "git_blob_sha1": git_id if lfs_sha is None else None, "lfs_pointer_git_blob_sha1": git_id if lfs_sha is not None else None, "lfs_sha256": lfs_sha})
    files = select_server_rows(all_rows, selected)
    if not files: raise SystemExit(f"empty server packet: {repo}")
    if {row["path"] for row in files} != selected: raise SystemExit(f"HF tree selected file set mismatch: {repo}")
    return snapshot, {"repository": repo, "requested_revision": rev, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(files, key=lambda row: row["path"])}
model_snapshot, model_packet_data = fetch(model_repo, model_rev, model_dir, sorted(model_selected), model_selected)
tok_snapshot, tok_packet_data = fetch(tok_repo, tok_rev, tok_dir, sorted(tokenizer_selected), tokenizer_selected)
Path(model_out).write_text(str(model_snapshot) + "\n", encoding="utf-8"); Path(tok_out).write_text(str(tok_snapshot) + "\n", encoding="utf-8")
Path(model_packet).write_text(json.dumps(model_packet_data, sort_keys=True, indent=2) + "\n", encoding="utf-8"); Path(tok_packet).write_text(json.dumps(tok_packet_data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
snapshot="$(< "$snapshot_file")"; companion_path="$(< "$companion_file")"
git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source" >/dev/null 2>&1; git -C "$source" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1; [[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "source revision mismatch"; source_origin_expected="${SOURCE_REPOSITORY%.git}"; [[ "$(git -C "$source" remote get-url origin | sed 's#/$##; s#\.git$##')" == "$source_origin_expected" ]] || die "source origin mismatch"
git clone --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers" >/dev/null 2>&1; git -C "$transformers" checkout --detach "$TRANSFORMERS_REVISION" >/dev/null 2>&1; [[ "$(git -C "$transformers" rev-parse HEAD)" == "$TRANSFORMERS_REVISION" ]] || die "Transformers revision mismatch"; [[ "$(git -C "$transformers" describe --exact-match --tags HEAD)" == "$TRANSFORMERS_TAG" ]] || die "Transformers tag mismatch"; transformers_origin_expected="${TRANSFORMERS_REPOSITORY%.git}"; [[ "$(git -C "$transformers" remote get-url origin | sed 's#/$##; s#\.git$##')" == "$transformers_origin_expected" ]] || die "Transformers origin mismatch"
set +e; "${UV_CMD[@]}" "$INSPECTOR" --snapshot "$snapshot" --companion "$companion_path" --source "$source" --transformers "$transformers" --server-tree "$model_tree" --companion-server-tree "$companion_tree" --output "$evidence"; status=$?; set -e
[[ "$status" == 2 ]] || die "inspection did not return exit 2"
set +e; "${UV_CMD[@]}" - "$evidence/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("status") != "BLOCKED" or manifest.get("evidence_stage") != "INSPECTION_ONLY": raise SystemExit("invalid fail-closed manifest")
if manifest.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED" or manifest.get("publication") != "NO_UPLOAD": raise SystemExit("unsafe manifest verdict")
if manifest.get("inspection_status") == "INSPECTION_ERROR": raise SystemExit("inspection failed; evidence is incomplete")
if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE": raise SystemExit("missing complete inspection marker")
PY
manifest_status=$?; set -e
[[ "$manifest_status" == 0 ]] || { echo "VibeVoice Realtime inspection failed; incomplete evidence=$evidence" >&2; exit 2; }
echo "VibeVoice Realtime inspection BLOCKED (evidence complete; runtime remains unavailable); evidence=$evidence" >&2; exit 2
