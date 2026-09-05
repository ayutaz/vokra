#!/usr/bin/env bash
# VAST-only structural evidence for NVIDIA Canary-Qwen-2.5B.
# This worker never converts, builds, uploads, or publishes an artifact.
set -euo pipefail

HF_REPOSITORY="nvidia/canary-qwen-2.5b"
HF_REVISION="b1469e1bba1cfe140205529c79c434ca47180960"
SOURCE_REPOSITORY="https://github.com/NVIDIA/NeMo.git"
SOURCE_TAG="v2.5.0"
SOURCE_REVISION="ddcb2d6935045a556329f1afa653b8d918c36479"
TOKENIZER_REPOSITORY="Qwen/Qwen3-1.7B"
TOKENIZER_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
INSPECTOR="tools/parity/canary_qwen_2_5b_inspect.py"
TOKENIZER_COMPLETE_FILES=".gitattributes LICENSE README.md config.json generation_config.json merges.txt model-00001-of-00002.safetensors model-00002-of-00002.safetensors model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json"
TOKENIZER_SELECTED_FILES="LICENSE README.md config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_TMPFS_KIB=$((32 * 1024 * 1024))

die() { echo "run-canary-qwen-2-5b-inspection: $*" >&2; exit 2; }
normalize_origin() {
  local value="${1%/}"
  value="${value%.git}"
  printf '%s' "$value"
}

canonical_absent_path() {
  local target="$1" lexical current="/" component suffix="" real
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { echo "work path contains .." >&2; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { echo "work path contains a symlinked ancestor" >&2; return 2; }
  done
  current="$target"
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { echo "work path parent is missing or symlinked" >&2; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { echo "work path parent is inaccessible" >&2; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_work_dir() {
  local work="$1" protected="$2" candidate protected_real
  [[ ! -e "$work" && ! -L "$work" ]] || { echo "work directory must be absent" >&2; return 2; }
  candidate="$(canonical_absent_path "$work")" || return 2
  protected_real="$(cd -P "$protected" 2>/dev/null && pwd)" || { echo "protected root is inaccessible" >&2; return 2; }
  [[ "$candidate" != "$protected_real" && "$candidate/" != "$protected_real/"* && "$protected_real/" != "$candidate/"* ]] || { echo "work directory overlaps checkout" >&2; return 2; }
}

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 required status tmp
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  for required in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_TAG" "$SOURCE_REVISION" "$TOKENIZER_REPOSITORY" "$TOKENIZER_REVISION" "$TOKENIZER_COMPLETE_FILES" "$TOKENIZER_SELECTED_FILES" "$INSPECTOR" "MODEL_FILES" "MODEL_REQUIRED_FILES" "MAX_HEADER_BYTES" "INSPECTION_ONLY" "BLOCKED" "NO_UPLOAD" "local_dir" ".cache" "git_blob_sha1" "lfs_sha256" "inspection_status" "AUTHENTICATED_EVIDENCE_COMPLETE" "INSPECTION_ERROR"; do
    if ! grep -Fq -- "$required" "$self" && ! grep -Fq -- "$required" "$root/$INSPECTOR"; then
      echo "self-test FAIL: missing $required" >&2; fail=1
    fi
  done
  for required in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'findmnt' 'git status --porcelain --untracked-files=all' 'snapshot_download' 'model_info' 'CARGO_BUILD_JOBS' 'cargo fmt --all -- --check' 'cargo metadata --locked --no-deps --format-version 1'; do
    if ! grep -Fq -- "$required" "$self"; then echo "self-test FAIL: missing VAST gate $required" >&2; fail=1; fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|hf_hub_upload|upload_file|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check|build))([[:space:]]|$)' "$self" >/dev/null; then
    echo "self-test FAIL: mutation/conversion/Cargo found" >&2; fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: raw Python/pip found" >&2; fail=1; fi
  local selector_prefix='patterns = [row["path"] for row in '
  local selector_bad="${selector_prefix}rows]"
  if grep -Fq "$selector_bad" "$self" || ! grep -Fq "${selector_prefix}selected]" "$self"; then echo "self-test FAIL: download selector is not selected-row bound" >&2; fail=1; fi
  if [[ "$(normalize_origin 'https://github.com/NVIDIA/NeMo.git/')" != "https://github.com/NVIDIA/NeMo" || "$(normalize_origin "$SOURCE_REPOSITORY")" != "https://github.com/NVIDIA/NeMo" ]]; then echo "self-test FAIL: origin normalization contract" >&2; fail=1; fi
  if bash "$self" --self-test --work-dir /tmp/canary-qwen-self-test >/dev/null 2>&1; then
    echo "self-test FAIL: extra argument accepted" >&2; fail=1
  else
    status=$?; [[ "$status" == 2 ]] || { echo "self-test FAIL: expected exit 2, got $status" >&2; fail=1; }
  fi
  tmp="$(cd -P "$(mktemp -d)" && pwd)"; trap 'rm -rf -- "$tmp"' RETURN
  mkdir -p "$tmp/checkout" "$tmp/real/existing"
  require_absent_work_dir "$tmp/new/nested" "$tmp/checkout"
  ln -s "$tmp/real" "$tmp/link"
  if require_absent_work_dir "$tmp/link/existing/nested" "$tmp/checkout" >/dev/null 2>&1; then echo "self-test FAIL: symlink ancestor accepted" >&2; fail=1; fi
  mkdir "$tmp/empty"
  if require_absent_work_dir "$tmp/empty" "$tmp/checkout" >/dev/null 2>&1; then echo "self-test FAIL: existing empty work accepted" >&2; fail=1; fi
  trap - RETURN; rm -rf -- "$tmp"
  (( fail == 0 )) && echo "run-canary-qwen-2-5b-inspection.sh self-test: OK" || return 1
}

work_dir="/dev/shm/vokra-canary-qwen-2-5b-inspection"; self=0; seen_self=0; seen_work=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die "duplicate --self-test"; seen_self=1; self=1; shift;;
    --work-dir) (( seen_work == 0 )) || die "duplicate --work-dir"; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die "--work-dir requires a nonempty path"; seen_work=1; work_dir="$2"; shift 2;;
    -h|--help) echo "usage: $0 [--work-dir TMPFS] | --self-test"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
if (( self == 1 )); then
  [[ "$seen_work" == 0 ]] || die "--self-test accepts no other arguments"
  self_test; exit $?
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
require_absent_work_dir "$work_dir" "$root"
parent="$(dirname "$work_dir")"
[[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 128 GiB"
free="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_TMPFS_KIB" ]] || die "tmpfs below 32 GiB"
for command in git uv sha256sum findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; export CARGO_BUILD_JOBS=1
cache="$work_dir/cache"; model="$work_dir/model"; tokenizer="$work_dir/tokenizer"; model_tree="$work_dir/model-tree.json"; tokenizer_tree="$work_dir/tokenizer-tree.json"; tokenizer_selected_tree="$work_dir/tokenizer-selected-tree.json"; source="$work_dir/nemo"; evidence="$work_dir/evidence"
mkdir -p "$cache" "$model" "$tokenizer" "$evidence"

"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$TOKENIZER_REPOSITORY" "$TOKENIZER_REVISION" "$cache" "$model" "$tokenizer" "$model_tree" "$tokenizer_tree" "$tokenizer_selected_tree" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, snapshot_download

model_repo, model_rev, tok_repo, tok_rev, cache, model_dir, tok_dir, model_packet, tok_packet, tok_selected_packet = sys.argv[1:]
api = HfApi()

def identity(item):
    lfs = getattr(item, "lfs", None)
    lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    git_id = getattr(item, "blob_id", None)
    if not isinstance(item.path, str) or not isinstance(item.size, int) or isinstance(item.size, bool) or item.size < 0:
        raise SystemExit(f"invalid server file: {item}")
    if not isinstance(git_id, str) or len(git_id) != 40 or any(c not in "0123456789abcdefABCDEF" for c in git_id):
        raise SystemExit(f"missing Git blob identity: {item.path}")
    if lfs_sha is not None and (not isinstance(lfs_sha, str) or len(lfs_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in lfs_sha)):
        raise SystemExit(f"invalid LFS identity: {item.path}")
    return {"path": item.path, "type": "file", "size": item.size, "git_blob_sha1": git_id, "lfs_sha256": lfs_sha}

def fetch(repo, rev, destination, selected_paths=None):
    info = api.model_info(repo_id=repo, revision=rev)
    if info.sha != rev: raise SystemExit(f"HF revision drift: {repo} {info.sha}")
    all_items = [item for item in api.list_repo_tree(repo_id=repo, revision=rev, recursive=True) if isinstance(item, RepoFile)]
    if not all_items: raise SystemExit(f"empty server tree: {repo}")
    all_rows = [identity(item) for item in all_items]
    selected = all_rows if selected_paths is None else [row for row in all_rows if row["path"] in selected_paths]
    if selected_paths is not None and {row["path"] for row in selected} != set(selected_paths):
        raise SystemExit("Qwen tokenizer selected semantic set is incomplete")
    if selected_paths is not None and any(Path(row["path"]).suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt") for row in selected):
        raise SystemExit("Qwen tokenizer semantic set contains model weights")
    patterns = [row["path"] for row in selected]
    snapshot_download(repo_id=repo, revision=rev, cache_dir=cache, local_dir=destination, allow_patterns=patterns, token=os.environ.get("HF_TOKEN") or os.environ.get("HF"))
    if not Path(destination).is_dir(): raise SystemExit(f"local_dir materialization missing: {destination}")
    return {"repository": repo, "revision": rev, "resolved_revision": info.sha, "files": sorted(selected, key=lambda row: row["path"])}, {"repository": repo, "revision": rev, "resolved_revision": info.sha, "files": sorted(all_rows, key=lambda row: row["path"])}

model_selected, _ = fetch(model_repo, model_rev, model_dir)
Path(model_packet).write_text(json.dumps(model_selected, sort_keys=True, indent=2) + "\n", encoding="utf-8")
tok_selected, tok_complete = fetch(tok_repo, tok_rev, tok_dir, {"LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"})
if {row["path"] for row in tok_complete["files"]} != {".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}:
    raise SystemExit("Qwen tokenizer complete server tree mismatch")
Path(tok_packet).write_text(json.dumps(tok_complete, sort_keys=True, indent=2) + "\n", encoding="utf-8")
Path(tok_selected_packet).write_text(json.dumps(tok_selected, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

git clone --filter=blob:none "$SOURCE_REPOSITORY" "$source" >/dev/null 2>&1
git -C "$source" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "NeMo revision mismatch"
[[ "$(git -C "$source" describe --exact-match --tags HEAD)" == "$SOURCE_TAG" ]] || die "NeMo tag mismatch"
[[ "$(normalize_origin "$(git -C "$source" remote get-url origin)")" == "$(normalize_origin "$SOURCE_REPOSITORY")" ]] || die "NeMo origin mismatch"
export CARGO_BUILD_JOBS=1
cargo fmt --all -- --check
cargo metadata --locked --no-deps --format-version 1 >/dev/null

set +e
"${UV_CMD[@]}" "$INSPECTOR" --snapshot "$model" --tokenizer "$tokenizer" --source "$source" --server-tree "$model_tree" --tokenizer-complete-tree "$tokenizer_tree" --tokenizer-server-tree "$tokenizer_selected_tree" --output "$evidence"
status=$?
set -e
[[ "$status" == 2 ]] || die "inspection did not return exit 2"
set +e
"${UV_CMD[@]}" - "$evidence/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("status") != "BLOCKED" or manifest.get("evidence_stage") != "INSPECTION_ONLY": raise SystemExit("invalid fail-closed manifest")
if manifest.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED" or manifest.get("publication") != "NO_UPLOAD": raise SystemExit("unsafe verdict fields")
if manifest.get("inspection_status") == "INSPECTION_ERROR": raise SystemExit("inspection failed; evidence incomplete")
if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE": raise SystemExit("missing complete inspection marker")
PY
manifest_status=$?
set -e
[[ "$manifest_status" == 0 ]] || { echo "Canary-Qwen inspection failed; incomplete evidence=$evidence" >&2; exit 2; }
echo "Canary-Qwen inspection BLOCKED (evidence complete; runtime/conversion remain unavailable); evidence=$evidence" >&2
exit 2
