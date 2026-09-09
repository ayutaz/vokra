#!/usr/bin/env bash
# VAST-only ChatTTS composite inspection. Never converts, uploads, or publishes.
set -euo pipefail

HF_REPOSITORY="2Noise/ChatTTS"
HF_REVISION="1a3c04a8b0651689bd9242fbb55b1f4b5a9aef84"
SOURCE_REPOSITORY="https://github.com/2noise/ChatTTS.git"
SOURCE_REVISION="77b89ee281cd479f5b1a787ada330dc975ca1f2a"
SOURCE_TAG="v0.2.5"
INSPECTOR="tools/parity/chattts_inspect.py"
REFERENCE="tools/parity/chattts_dump_reference.py"
REFERENCE_LOCK_SHA256="6099870e3685fec99e8ae68745d37ce4e71138d353cf056540d092b3d55ac4c5"
REFERENCE_PYPROJECT_SHA256="d16bc02489442ee0134fb82903f3d08dbdc3f9d7b23a67d730cf9202fee23b9e"
REFERENCE_PACKAGE_INVENTORY_SHA256="74c0c3ef9afd095594e24afc48e0d2148717308aa317ea9762eb9b10d2f0ec7f"
REFERENCE_LOCK_PACKAGE_ROWS_SHA256="19395b8e7796dc26af01df77e3b786299391c38f3f861d2e9b59e29175b1cb4c"
REFERENCE_LICENSE_AUDIT_SHA256="3e5b662aa2134be84ee6645a7c483345d46550b690c582f414896c990a7f1dff"
UV_CMD=(uv run --no-sync --frozen --project tools/parity/chattts --python 3.12 python)
UV_GATE_CMD=(uv run --no-cache --no-project --offline --python 3.12 python)
MIN_MEM_KIB=$((64 * 1024 * 1024))

die() { echo "run-chattts-inspection: $*" >&2; exit 2; }

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 status needle gate_pattern
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  [[ -f "$root/tools/parity/chattts/pyproject.toml" ]] || die "ChatTTS dedicated project missing"
  for needle in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$SOURCE_TAG" "$INSPECTOR" "$REFERENCE" "$REFERENCE_LOCK_SHA256" "$REFERENCE_PYPROJECT_SHA256" "$REFERENCE_PACKAGE_INVENTORY_SHA256" "$REFERENCE_LOCK_PACKAGE_ROWS_SHA256" "$REFERENCE_LICENSE_AUDIT_SHA256" "transformers==5.10.4" "huggingface-hub==1.5.0" "GHSA-xrqw-3rrv-vx5w" "dependency-gate" "--validate-approval" "AUDITED_ALLOW" "AUTHENTICATED_EVIDENCE_COMPLETE" "INSPECTION_ERROR" "INSPECTION_ONLY" "NO_UPLOAD" "snapshot_download" "list_repo_tree" "README.md" "asset/DVAE.safetensors" "asset/gpt/model.safetensors" "res/sha256_map.json" "legacy"; do
    if ! grep -Fq -- "$needle" "$self" && ! grep -Fq -- "$needle" "$root/$INSPECTOR"; then echo "self-test FAIL: missing $needle" >&2; fail=1; fi
  done
  if ! grep -Fq 'allow_patterns=selected' "$self"; then echo "self-test FAIL: selected-only materialization contract missing" >&2; fail=1; fi
  if ! grep -Fq 'canonical_absent_path' "$self" || ! grep -Fq 'INSPECTION_ONLY' "$self"; then echo "self-test FAIL: canonical/blocked-disposition contract missing" >&2; fail=1; fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: mutation/conversion/Cargo test found" >&2; fail=1; fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: raw Python/pip found" >&2; fail=1; fi
  gate_pattern="\"\$root/\$REFERENCE\" --dependency-gate"
  gate_line="$(grep -nF "$gate_pattern" "$self" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^uv sync --project' "$self" | tail -1 | cut -d: -f1)"
  download_line="$(grep -n 'snapshot_download(repo_id' "$self" | tail -1 | cut -d: -f1)"
  gate_command="$(sed -n "${gate_line}p" "$self")"
  if [[ -z "$gate_line" || -z "$sync_line" || -z "$download_line" || "$gate_line" -ge "$sync_line" || "$sync_line" -ge "$download_line" || "$gate_command" != *"UV_GATE_CMD"* || "$gate_command" == *"--project tools/parity/chattts"* ]] || ! grep -Fq 'UV_GATE_CMD=(uv run --no-cache --no-project --offline --python 3.12 python)' "$self"; then
    echo "self-test FAIL: dependency sync must follow the affirmative gate and precede download" >&2; fail=1
  fi
  if bash "$self" --self-test --work-dir /tmp/chattts-self-test >/dev/null 2>&1; then echo "self-test FAIL: extra args accepted" >&2; fail=1; else status=$?; [[ "$status" == 2 ]] || { echo "self-test FAIL: expected exit 2, got $status" >&2; fail=1; }; fi
  (( fail == 0 )) && echo "run-chattts-inspection.sh self-test: OK" || return 1
}

work_dir="/dev/shm/vokra-chattts-inspection"
self=0; expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_sha=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self=1; shift;;
    --work-dir) [[ $# -ge 2 ]] || die "--work-dir requires path"; work_dir="$2"; shift 2;;
    --expected-head) (( seen_head == 0 )) || die "duplicate --expected-head"; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die "--expected-head requires lowercase 40-hex"; expected_head="$2"; seen_head=1; shift 2;;
    --approval-evidence) (( seen_approval == 0 )) || die "duplicate --approval-evidence"; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--approval-evidence requires a path"; approval_evidence="$2"; seen_approval=1; shift 2;;
    --approval-sha256) (( seen_sha == 0 )) || die "duplicate --approval-sha256"; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die "--approval-sha256 requires lowercase 64-hex"; approval_sha256="$2"; seen_sha=1; shift 2;;
    -h|--help) echo "usage: $0 [--work-dir /dev/shm/path] | --self-test"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
if (( self == 1 )); then
  [[ "$work_dir" == "/dev/shm/vokra-chattts-inspection" && $seen_head == 0 && $seen_approval == 0 && $seen_sha == 0 ]] || die "--self-test accepts no other arguments"
  self_test
  exit 0
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
[[ -f "$root/tools/parity/chattts/uv.lock" ]] || die "ChatTTS dedicated locked environment missing"
[[ "$(sha256sum "$root/tools/parity/chattts/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die "ChatTTS dedicated uv.lock identity mismatch"
[[ "$(sha256sum "$root/tools/parity/chattts/pyproject.toml" | awk '{print $1}')" == "$REFERENCE_PYPROJECT_SHA256" ]] || die "ChatTTS dedicated pyproject identity mismatch"
(( seen_head == 1 && seen_approval == 1 && seen_sha == 1 )) || die "expected HEAD and external approval are required"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die "checkout HEAD does not match --expected-head"
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die "approval evidence is missing or symlinked"
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" "$root/$REFERENCE" --validate-approval --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --expected-head "$expected_head" >/dev/null || die "external approval evidence is invalid"
UV_NO_CACHE=1 "${UV_GATE_CMD[@]}" "$root/$REFERENCE" --dependency-gate || die "ChatTTS dependency/license gate is not explicitly approved"
die 'ChatTTS approval scope is INSPECTION_ONLY; acquisition and official reference execution remain unconditionally blocked'
canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && "$path" != */ && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else parent="$(dirname "$path")"; base="$(basename "$path")"; parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"; fi
}
canonical_absent_path() {
  local path="$1" target="$1" rest component current=/ suffix='' parent
  [[ "$path" == /* && "$path" != */ && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do
    suffix="/$(basename "$target")$suffix"; parent="$(dirname "$target")"; [[ "$parent" != "$target" ]] || return 1; target="$parent"
  done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  printf '%s%s\n' "$(cd -P "$target" && pwd)" "$suffix"
}
root_real="$(canonical_existing_path "$root")" || die "checkout path is not canonical"
approval_real="$(canonical_existing_path "$approval_evidence")" || die "approval path has invalid canonical/symlink ancestry"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
parent="$(dirname "$work_dir")"
[[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
work_real="$(canonical_absent_path "$work_dir")" || die "work path must be absent and free of dot/symlink ancestry"
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
paths_overlap "$work_real" "$root_real" && die "work path overlaps checkout"
paths_overlap "$work_real" "$approval_real" && die "work path overlaps approval"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 64 GiB"
for command in git cargo rustfmt uv findmnt sha256sum; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
uv sync --project "$root/tools/parity/chattts" --frozen --python 3.12
mkdir "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; export CARGO_BUILD_JOBS=1
cargo fmt --all -- --check || die "cargo fmt check failed"
cargo metadata --locked --no-deps --format-version 1 >/dev/null || die "cargo metadata failed"
cache="$work_dir/cache"; model="$work_dir/model"; tree="$work_dir/server-tree.json"; source="$work_dir/source"; evidence="$work_dir/evidence"
mkdir "$cache" "$model" "$evidence"

"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$cache" "$model" "$tree" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, snapshot_download

repo, rev, cache, model, tree = sys.argv[1:]
api = HfApi()
info = api.model_info(repo_id=repo, revision=rev)
if info.sha != rev:
    raise SystemExit(f"HF revision drift: {info.sha} != {rev}")
# local_dir materializes cache symlinks; the inspector excludes internal .cache.
# README is a required semantic/license input; the nine selected release assets
# are downloaded alongside it, while legacy .pt files remain server-authenticated
# but deliberately not downloaded.
selected = ["README.md", "asset/DVAE.safetensors", "asset/Decoder.safetensors", "asset/Embed.safetensors", "asset/Vocos.safetensors", "asset/gpt/config.json", "asset/gpt/model.safetensors", "asset/tokenizer/special_tokens_map.json", "asset/tokenizer/tokenizer.json", "asset/tokenizer/tokenizer_config.json"]
snapshot = Path(snapshot_download(repo_id=repo, revision=rev, cache_dir=cache, local_dir=model, allow_patterns=selected, token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
if snapshot.resolve() != Path(model).resolve():
    raise SystemExit("materialized local_dir mismatch")
files = []
for item in api.list_repo_tree(repo_id=repo, revision=rev, recursive=True):
    if not isinstance(item, RepoFile):
        continue
    lfs = getattr(item, "lfs", None)
    lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    blob = getattr(item, "oid", None) or getattr(item, "blob_id", None)
    if not isinstance(item.path, str) or not isinstance(item.size, int) or not isinstance(blob, str) or len(blob) != 40:
        raise SystemExit(f"incomplete server identity: {item}")
    if lfs_sha is not None and (not isinstance(lfs_sha, str) or len(lfs_sha) != 64):
        raise SystemExit(f"invalid LFS identity: {item.path}")
    files.append({"path": item.path, "type": "file", "size": item.size, "git_blob_sha1": blob, "lfs_sha256": lfs_sha})
expected = {row["path"] for row in files}
actual = set()
for path in snapshot.rglob("*"):
    rel = path.relative_to(snapshot)
    if ".cache" in rel.parts:
        continue
    if path.is_symlink():
        if not path.exists() or not path.is_file() or snapshot not in path.resolve().parents:
            raise SystemExit(f"invalid materialized symlink: {path}")
        actual.add(rel.as_posix())
    elif path.is_file():
        actual.add(rel.as_posix())
    elif not path.is_dir():
        raise SystemExit(f"non-regular snapshot member: {path}")
if actual != set(selected):
    raise SystemExit(f"selected/local mismatch: missing={sorted(set(selected)-actual)} extra={sorted(actual-set(selected))}")
Path(tree).write_text(json.dumps({"repository": repo, "revision": rev, "resolved_revision": info.sha, "files": sorted(files, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source" >/dev/null 2>&1
git -C "$source" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "source revision mismatch"
[[ "$(git -C "$source" describe --exact-match --tags HEAD)" == "$SOURCE_TAG" ]] || die "source tag mismatch"
set +e
"${UV_CMD[@]}" "$INSPECTOR" --snapshot "$model" --source "$source" --server-tree "$tree" --output "$evidence" --expected-head "$expected_head" --approval-sha256 "$approval_sha256"
status=$?
set -e
[[ "$status" == 2 ]] || die "inspector did not return exit 2"
[[ -f "$evidence/manifest.json" ]] || die "inspection manifest missing"
"${UV_CMD[@]}" - "$evidence/manifest.json" "$expected_head" "$approval_sha256" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {"status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "native_status": "BLOCKED_NATIVE_BINDING", "publication": "NO_UPLOAD", "expected_head": sys.argv[2], "approval_sha256": sys.argv[3]}
if any(manifest.get(key) != value for key, value in required.items()):
    raise SystemExit(f"invalid fail-closed manifest: {manifest}")
if manifest.get("inspection_status") == "INSPECTION_ERROR":
    raise SystemExit("ChatTTS inspection failed; evidence is not complete")
if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
    raise SystemExit("ChatTTS evidence completion marker missing")
PY
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree became dirty during inspection"
[[ "$(git rev-parse HEAD)" == "$expected_head" ]] || die "checkout HEAD changed during inspection"
echo "ChatTTS inspection BLOCKED (authenticated evidence only, no upload); evidence=$evidence" >&2
exit 2
