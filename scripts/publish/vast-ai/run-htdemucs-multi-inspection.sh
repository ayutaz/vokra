#!/usr/bin/env bash
# VAST/Linux-only HT-Demucs ensemble inspection.  This worker downloads only
# the five official registry members and pinned source configs, attempts safe
# loading, and never converts, uploads, or publishes a product artifact.
# Every member is bound to its exact model id, URL, response identity, and
# authenticated full SHA-256 digest.  A filename prefix is retained as a
# diagnostic only and never substitutes for the full digest contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
AUDIT="$VOKRA_ROOT/tools/parity/htdemucs_multi/audit.py"
INSPECTOR="$VOKRA_ROOT/tools/parity/htdemucs_multi_inspect.py"
UPSTREAM_URL="https://github.com/facebookresearch/demucs.git"
UPSTREAM_REVISION="e976d93ecc3865e5757426930257e200846a520a"
WEIGHT_ROOT="https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))
HTDEMUCS_UV_CACHE_DIR="${HTDEMUCS_UV_CACHE_DIR:-/tmp/vokra-htdemucs-uv-cache}"
MEMBERS=(
  f7e0c4bc-ba3fe64a.th
  d12395a8-e57c48e6.th
  92cfc3b6-ef3bcb9c.th
  04573f0d-f3cf25b2.th
  5c90dfd2-34c22ccb.th
)

log() { printf '[htdemucs-multi-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

download_member() {
  local member="$1" target="$work_dir/weights/$1" headers="$work_dir/response/$1.headers" meta="$work_dir/response/$1.meta"
  local temporary_target temporary_headers temporary_meta temporary_target_identity temporary_headers_identity temporary_meta_identity
  local target_identity headers_identity meta_identity
  same_inode() {
    local candidate="$1" expected="$2" actual
    [[ -f "$candidate" && ! -L "$candidate" ]] || return 1
    actual="$(stat -c '%d:%i' "$candidate")" || return 1
    [[ "$actual" == "$expected" ]]
  }
  unlink_owned() {
    local candidate="$1" expected="$2"
    if same_inode "$candidate" "$expected"; then
      rm -f "$candidate" || true
    fi
  }
  temporary_target="$(mktemp "${target}.tmp.XXXXXX")" || { die "checkpoint temp reservation failed: $member"; return 2; }
  temporary_target_identity="$(stat -c '%d:%i' "$temporary_target")" || { unlink_owned "$temporary_target" ""; die "checkpoint temp identity failed: $member"; return 2; }
  temporary_headers="$(mktemp "${headers}.tmp.XXXXXX")" || { unlink_owned "$temporary_target" "$temporary_target_identity"; die "checkpoint temp reservation failed: $member"; return 2; }
  temporary_headers_identity="$(stat -c '%d:%i' "$temporary_headers")" || { unlink_owned "$temporary_target" "$temporary_target_identity"; unlink_owned "$temporary_headers" ""; die "checkpoint temp identity failed: $member"; return 2; }
  temporary_meta="$(mktemp "${meta}.tmp.XXXXXX")" || { unlink_owned "$temporary_target" "$temporary_target_identity"; unlink_owned "$temporary_headers" "$temporary_headers_identity"; die "checkpoint temp reservation failed: $member"; return 2; }
  temporary_meta_identity="$(stat -c '%d:%i' "$temporary_meta")" || { unlink_owned "$temporary_target" "$temporary_target_identity"; unlink_owned "$temporary_headers" "$temporary_headers_identity"; unlink_owned "$temporary_meta" ""; die "checkpoint temp identity failed: $member"; return 2; }
  cleanup_temp() {
    unlink_owned "$temporary_target" "$temporary_target_identity"
    unlink_owned "$temporary_headers" "$temporary_headers_identity"
    unlink_owned "$temporary_meta" "$temporary_meta_identity"
  }
  if ! curl --fail --location --retry 3 --silent --show-error \
    --dump-header "$temporary_headers" \
    --write-out '%{http_code}\t%{url_effective}\t%{size_download}\n' \
    "$WEIGHT_ROOT/$member" --output "$temporary_target" > "$temporary_meta"; then
    cleanup_temp
    die "checkpoint download failed: $member"
    return 2
  fi
  same_inode "$temporary_target" "$temporary_target_identity" || { cleanup_temp; die "checkpoint temp identity changed: $member"; return 2; }
  same_inode "$temporary_headers" "$temporary_headers_identity" || { cleanup_temp; die "checkpoint temp identity changed: $member"; return 2; }
  same_inode "$temporary_meta" "$temporary_meta_identity" || { cleanup_temp; die "checkpoint temp identity changed: $member"; return 2; }
  target_identity="$temporary_target_identity"
  headers_identity="$temporary_headers_identity"
  meta_identity="$temporary_meta_identity"
  if ln "$temporary_target" "$target" && same_inode "$target" "$target_identity"; then :; else
    cleanup_temp
    die "checkpoint output already exists: $member"
    return 2
  fi
  if ln "$temporary_headers" "$headers" && same_inode "$headers" "$headers_identity"; then :; else
    unlink_owned "$target" "$target_identity"
    cleanup_temp
    die "checkpoint response headers already exists: $member"
    return 2
  fi
  if ln "$temporary_meta" "$meta" && same_inode "$meta" "$meta_identity"; then :; else
    unlink_owned "$headers" "$headers_identity"
    unlink_owned "$target" "$target_identity"
    cleanup_temp
    die "checkpoint response metadata already exists: $member"
    return 2
  fi
  cleanup_temp
}

reject_symlink_ancestors() {
  local path="$1" rest component current
  if [[ "$path" != /* || "$path" == */ ]]; then
    die 'work-dir must be an absolute path without a trailing slash'
    return 2
  fi
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then
      component="${rest%%/*}"
      rest="${rest#*/}"
    else
      component="$rest"
      rest=""
    fi
    if [[ -z "$component" || "$component" == . || "$component" == .. ]]; then
      die 'work-dir contains an empty or dot path component'
      return 2
    fi
    current="$current$component"
    if [[ -L "$current" ]]; then
      die "work-dir has a symlinked ancestor: $path"
      return 2
    fi
    current="$current/"
  done
}

validate_work_dir() {
  local candidate parent root_real approval_abs
  reject_symlink_ancestors "$work_dir" || return 2
  [[ ! -e "$work_dir" && ! -L "$work_dir" ]] || die 'work-dir must be absent and non-symlink'
  parent="${work_dir%/*}"
  [[ -n "$parent" && "$parent" != "$work_dir" ]] || die 'work-dir parent is invalid'
  [[ -d "$parent" && ! -L "$parent" ]] || die 'work-dir parent must be an existing non-symlink directory'
  candidate="$(cd -P "$parent" && pwd)/${work_dir##*/}" || die 'could not resolve work-dir parent'
  root_real="$(cd -P "$VOKRA_ROOT" && pwd)" || die 'could not resolve Vokra checkout'
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && \
    "$root_real/" != "$candidate/"* ]] || die 'work-dir overlaps the Vokra checkout'
  approval_abs="$approval_evidence"
  [[ "$approval_abs" == /* ]] || approval_abs="$PWD/$approval_abs"
  [[ "$approval_abs" != "$candidate" && "$approval_abs" != "$candidate/"* && \
    "$candidate/" != "$approval_abs/"* ]] || die 'work-dir overlaps approval evidence'
}

usage() {
  cat <<'EOF'
usage: run-htdemucs-multi-inspection.sh --expected-head <HEX40> --approval-evidence <file> [--work-dir <empty-dir>]
       run-htdemucs-multi-inspection.sh --self-test

VAST/Linux-only inspection of the pinned official Demucs source and five
registry checkpoints. Safe loading uses torch.load(weights_only=True) only;
unsafe pickle fallback is forbidden. The result is INSPECTION_ONLY and no
GGUF conversion, upload, publication, or parity verdict is produced.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token temporary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-htdemucs-inspect.XXXXXX")"
  temporary="$(cd -P "$temporary" && pwd)"
  HTDEMUCS_SELF_TEST_TMP="$temporary"
  trap 'rm -rf "$HTDEMUCS_SELF_TEST_TMP"' EXIT
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'MIN_VAST_MEM_KIB' '/proc/meminfo' \
    'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --locked --no-deps --format-version 1' 'facebookresearch/demucs.git' \
    "$UPSTREAM_REVISION" "$WEIGHT_ROOT" 'f7e0c4bc-ba3fe64a.th' \
    'd12395a8-e57c48e6.th' '92cfc3b6-ef3bcb9c.th' '04573f0d-f3cf25b2.th' \
    '5c90dfd2-34c22ccb.th' 'weights_only=True' 'no pickle fallback' \
    'KNOWN_HEAD_BYTES' '84141271' '54996327' \
    'htdemucs_multi_inspect.py' 'collect_dependency_evidence.py' 'INSPECTION_ONLY' 'NOT_IMPLEMENTED' 'UNSUPPORTED' 'BLOCKED_BY_CPU' 'NOT_RUN' 'NO_UPLOAD' \
    'git status --porcelain' 'htdemucs_multi_inspect.py --self-test' \
    'response-packet' 'x-amz-version-id' 'x-amz-meta-s3cmd-attrs' \
    'sha256_filename_prefix_match' 'sha256_exact_match' \
    'FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER' 'FULL_WEIGHT_DIGESTS_AUTHENTICATED' \
    'expected_sha256' 'response member id mismatch' \
    'inspection_status' 'COMPLETE' 'ERROR' 'variant_contracts' 'flattened 2,132-tensor' \
    '--expected-head' '--approval-evidence' 'BLOCKED_PENDING_PRIMARY_BYTES' 'REFERENCE_ROUTE_EXCLUDES_UNUSED_AUDIO_PACKAGES' \
    'dora-search' 'openunmix' 'torchaudio' 'lameenc' 'excluded_upstream_packages' 'BLOCKED_OWNER_REVIEW' \
    'safe_global_allowlist' 'BLOCKED_SOURCE_ALLOWLIST' 'verdict=BLOCKED' 'blocker_exit=2' \
    'reject_symlink_ancestors' 'work-dir overlaps' 'work-dir must be absent' \
    'CARGO_NET_OFFLINE=true' 'BLOCKED_PENDING_AUTHENTICATED_MANIFEST' \
    'transfer-packet' 'inspection_manifest' 'manifest.sha256' 'transfer_manifest_sha256' \
    'checkout HEAD changed before inspection evidence completion' 'download_member' 'mktemp' 'same_inode' 'cleanup_temp' \
    'temporary_target_identity' 'cli_path' 'O_NOFOLLOW' 'write_no_clobber' 'regular_identity' 'unlink_owned' 'os.link' 'os.fsync'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if "$path" --self-test --work-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  mkdir "$temporary/real"
  work_dir="$temporary/new-work-dir"
  approval_evidence="$temporary/approval.json"
  if ! validate_work_dir; then
    log 'self-test FAIL: absent non-symlink work-dir was rejected'
    fail=1
  fi
  work_dir="$temporary/../dot-work-dir"
  if validate_work_dir >/dev/null 2>&1; then
    log 'self-test FAIL: dot-component work-dir was accepted'
    fail=1
  fi
  ln -s "$temporary/real" "$temporary/work-link"
  work_dir="$temporary/work-link/new-work-dir"
  if validate_work_dir >/dev/null 2>&1; then
    log 'self-test FAIL: symlink-ancestor work-dir was accepted'
    fail=1
  fi
  if ! UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" \
    --python 3.12 python "$INSPECTOR" --self-test >/dev/null; then
    log 'self-test FAIL: inspector self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-htdemucs-multi-inspection"
expected_head=""
approval_evidence=""
self=0
seen_head=0; seen_approval=0; seen_self=0; seen_work=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires exactly 40 lowercase hexadecimal characters'; seen_head=1; expected_head="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; (($# >= 2)) || die '--work-dir requires a path'; seen_work=1; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$work_dir" == "/workspace/vokra-htdemucs-multi-inspection" && -z "$expected_head$approval_evidence" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$seen_head" == 1 && "$seen_approval" == 1 ]] || die '--expected-head and --approval-evidence are required'
[[ "$(uname -s)" == Linux ]] || die 'HT-Demucs checkpoint work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ "$(git -C "$VOKRA_ROOT" rev-parse --verify HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
validate_work_dir
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$INSPECTOR" ]] || die 'HT-Demucs inspector is missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv curl sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

# Approval/dependency/source identity must pass before work directories,
# downloads, or upstream source acquisition are created.
uv run --no-project --offline --python 3.12 python "$AUDIT" \
  --dependency-gate --expected-head "$expected_head" --approval-evidence "$approval_evidence" \
  >/dev/null || die 'dependency/license/approval gate is blocked'

mkdir "$work_dir"
mkdir "$work_dir/source" "$work_dir/weights" "$work_dir/response" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export CARGO_NET_OFFLINE=true
export UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR"
{
  echo "expected_head=$expected_head"
  echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse --verify HEAD)"
  echo "upstream_url=$UPSTREAM_URL"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "weight_root=$WEIGHT_ROOT"
  echo 'phase=INSPECTION'
  echo 'publication_policy=NO_UPLOAD'
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1
} > "$work_dir/evidence/validation.log" 2>&1

git clone --filter=blob:none --no-checkout "$UPSTREAM_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$UPSTREAM_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
[[ "$(git -C "$work_dir/source/repo" rev-parse HEAD)" == "$UPSTREAM_REVISION" ]] || die 'pinned Demucs source revision mismatch'
for config in htdemucs_ft.yaml htdemucs_6s.yaml; do
  [[ -f "$work_dir/source/repo/demucs/remote/$config" ]] || die "official config missing: $config"
done

for member in "${MEMBERS[@]}"; do
  download_member "$member"
done

[[ "$(git -C "$VOKRA_ROOT" rev-parse --verify HEAD)" == "$expected_head" ]] \
  || die 'checkout HEAD changed during inspection acquisition'

UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$work_dir/response" "$work_dir/weights" "$work_dir/evidence/response-packet.json" <<'PY'
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

response_dir, weights_dir, output = map(Path, sys.argv[1:])
members = [
    "f7e0c4bc-ba3fe64a.th",
    "d12395a8-e57c48e6.th",
    "92cfc3b6-ef3bcb9c.th",
    "04573f0d-f3cf25b2.th",
    "5c90dfd2-34c22ccb.th",
]
member_ids = {
    "f7e0c4bc-ba3fe64a.th": "f7e0c4bc",
    "d12395a8-e57c48e6.th": "d12395a8",
    "92cfc3b6-ef3bcb9c.th": "92cfc3b6",
    "04573f0d-f3cf25b2.th": "04573f0d",
    "5c90dfd2-34c22ccb.th": "5c90dfd2",
}
rows = {}
for member in members:
    meta = (response_dir / f"{member}.meta").read_text(encoding="utf-8").strip().split("\t")
    status, effective, observed_bytes = int(meta[0]), meta[1], int(float(meta[2]))
    headers = {}
    for block in (response_dir / f"{member}.headers").read_text(encoding="latin-1").split("\r\n\r\n"):
        current = {}
        for line in block.splitlines()[1:]:
            key, separator, value = line.partition(":")
            if separator:
                current[key.lower()] = value.strip()
        if current:
            headers = current
    path = weights_dir / member
    digest = hashlib.sha256()
    counted = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            counted += len(chunk)
            digest.update(chunk)
    rows[member] = {
        "filename": member,
        "model_id": member_ids[member],
        "requested_url": f"https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/{member}",
        "effective_url": effective,
        "status": status,
        "content_length": int(headers["content-length"]) if "content-length" in headers else None,
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "x_amz_version_id": headers.get("x-amz-version-id"),
        "x_amz_meta_s3cmd_attrs": headers.get("x-amz-meta-s3cmd-attrs"),
        "bytes": counted,
        "sha256": digest.hexdigest(),
    }
    if observed_bytes != counted:
        raise SystemExit(f"curl observed size differs from local size: {member}")
output_path = Path(output)
fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
temporary_path = Path(temporary_name)
temporary_stat = os.fstat(fd)
if not stat.S_ISREG(temporary_stat.st_mode):
    raise SystemExit("packet reservation is not a regular file")
temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
def regular_identity(path, expected=None):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise SystemExit("packet output is not a regular file")
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if expected is not None and identity != expected:
            raise SystemExit("packet output inode changed during publish")
        return identity
    finally:
        os.close(descriptor)
def unlink_owned(path, expected):
    try:
        if regular_identity(path, expected) == expected:
            path.unlink()
    except (FileNotFoundError, OSError, SystemExit):
        pass
try:
    with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
        json.dump({"members": rows}, stream, sort_keys=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    identity = regular_identity(temporary_path, temporary_identity)
    os.link(temporary_path, output_path, follow_symlinks=False)
    try:
        regular_identity(output_path, identity)
    except (OSError, SystemExit):
        unlink_owned(output_path, identity)
        raise
finally:
    try:
        os.close(fd)
    except OSError:
        pass
    unlink_owned(temporary_path, temporary_identity)
PY

set +e
UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$INSPECTOR" --source-dir "$work_dir/source/repo" --weights-dir "$work_dir/weights" \
  --response-packet "$work_dir/evidence/response-packet.json" \
  --evidence-dir "$work_dir/evidence" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must exit 2 with BLOCKED evidence: $inspect_rc"
[[ -s "$work_dir/evidence/htdemucs_multi_manifest.json" ]] || die 'inspection manifest is missing'
actual_head="$(git -C "$VOKRA_ROOT" rev-parse --verify HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die 'checkout HEAD changed before inspection publication'
approval_sha="$(sha256sum "$approval_evidence" | awk '{print $1}')"
gate_sha="$(sha256sum "$PROJECT/license_gate_manifest.json" | awk '{print $1}')"
uv run --no-project --offline --python 3.12 python - \
  "$work_dir/evidence/htdemucs_multi_manifest.json" "$work_dir/transfer-packet" \
  "$expected_head" "$actual_head" "$approval_sha" "$gate_sha" <<'PY'
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
manifest_path, transfer_dir = map(Path, sys.argv[1:3])
expected_head, actual_head, approval_sha, gate_sha = sys.argv[3:]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "BLOCKED" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("inspection manifest is not blocked/no-upload")
manifest.update({
    "expected_head": expected_head,
    "git_commit": actual_head,
    "approval_sha256": approval_sha,
    "license_gate_sha256": gate_sha,
})
members = manifest.get("members")
if not isinstance(members, dict):
    raise SystemExit("inspection manifest has no member digest rows")
transfer_dir.mkdir(mode=0o700)
inspection_copy = transfer_dir / "htdemucs_multi_manifest.json"
def write_no_clobber(path, data):
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    temporary_stat = os.fstat(fd)
    if not stat.S_ISREG(temporary_stat.st_mode):
        raise RuntimeError("transfer reservation is not a regular file")
    temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    def regular_identity(candidate, expected=None):
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode):
                raise RuntimeError("transfer output is not a regular file")
            identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            if expected is not None and identity != expected:
                raise RuntimeError("transfer output inode changed during publish")
            return identity
        finally:
            os.close(descriptor)
    def unlink_owned(candidate, expected):
        try:
            if regular_identity(candidate, expected) == expected:
                candidate.unlink()
        except (FileNotFoundError, OSError, RuntimeError):
            pass
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        identity = regular_identity(temporary, temporary_identity)
        os.link(temporary, path, follow_symlinks=False)
        try:
            regular_identity(path, identity)
        except (OSError, RuntimeError):
            unlink_owned(path, identity)
            raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        unlink_owned(temporary, temporary_identity)

enriched_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
write_no_clobber(inspection_copy, enriched_manifest)
inspection_digest = hashlib.sha256(inspection_copy.read_bytes()).hexdigest()
transfer = {
    "schema": "vokra-htdemucs-multi-transfer-v1",
    "status": "BLOCKED_PENDING_AUTHENTICATED_MANIFEST",
    "publication": "NO_UPLOAD",
    "expected_head": expected_head,
    "git_commit": actual_head,
    "approval_sha256": approval_sha,
    "license_gate_sha256": gate_sha,
    "inspection_manifest": {
        "filename": "htdemucs_multi_manifest.json",
        "sha256": inspection_digest,
    },
    "upstream_url": "https://github.com/facebookresearch/demucs",
    "upstream_revision": "e976d93ecc3865e5757426930257e200846a520a",
    "members": [
        {"model_id": model_id, "filename": row.get("filename"), "sha256": row.get("sha256")}
        for model_id, row in members.items()
    ],
}
transfer_path = transfer_dir / "manifest.json"
write_no_clobber(transfer_path, (json.dumps(transfer, indent=2) + "\n").encode("utf-8"))
digest = hashlib.sha256(transfer_path.read_bytes()).hexdigest()
write_no_clobber(transfer_dir / "manifest.sha256", f"{digest}  manifest.json\n".encode("ascii"))
PY
UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$work_dir/transfer-packet/htdemucs_multi_manifest.json" "$expected_head" "$actual_head" "$approval_sha" "$gate_sha" <<'PY'
import json
import re
import sys
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected_head, actual_head, approval_sha, gate_sha = sys.argv[2:]
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "inspection_status": "COMPLETE",
    "runtime_status": "NOT_IMPLEMENTED",
    "cpu_status": "UNSUPPORTED",
    "metal_status": "BLOCKED_BY_CPU",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"manifest contract mismatch: {key}={manifest.get(key)!r}")
if manifest.get("expected_head") != expected_head or manifest.get("git_commit") != actual_head:
    raise SystemExit("manifest checkout identity mismatch")
for key, expected in (("approval_sha256", approval_sha), ("license_gate_sha256", gate_sha)):
    if manifest.get(key) != expected or not re.fullmatch(r"[0-9a-f]{64}", manifest.get(key, "")):
        raise SystemExit(f"manifest digest binding mismatch: {key}")
if manifest.get("inspection_status") == "ERROR" or manifest.get("collection_status") == "FAILED":
    raise SystemExit("inspection error was treated as complete")
blockers = manifest.get("blockers", [])
if manifest.get("weight_digest_status") != "FULL_WEIGHT_DIGESTS_AUTHENTICATED":
    raise SystemExit(f"full weight digest authentication missing: {manifest.get('weight_digest_status')!r}")
if "FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER" in blockers:
    raise SystemExit("full weight digest blocker remained after exact authentication")
expected_digests = {
    "f7e0c4bc": ("f7e0c4bc-ba3fe64a.th", "ba3fe64ae8ef66ac9a4857222ce48efbdc5eb3ad375cb79dd13debee5aaa4066"),
    "d12395a8": ("d12395a8-e57c48e6.th", "e57c48e6b0e38af4f7118d7bd08c49f0a0c0edf7d09143bdd902ea0d237303e6"),
    "92cfc3b6": ("92cfc3b6-ef3bcb9c.th", "ef3bcb9c8b40d14ae5d51b6db2587339cc12c6b77c0be151ce6d69002e087bf2"),
    "04573f0d": ("04573f0d-f3cf25b2.th", "f3cf25b222c4eed7cd49dd8b2c9597d50c18bd154090f7b919cfa5f93cf22c49"),
    "5c90dfd2": ("5c90dfd2-34c22ccb.th", "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"),
}
members = manifest.get("members")
if not isinstance(members, dict) or list(members) != list(expected_digests):
    raise SystemExit("member digest manifest order/set mismatch")
for model_id, (filename, digest) in expected_digests.items():
    row = members.get(model_id)
    if not isinstance(row, dict) or row.get("filename") != filename or row.get("model_id") != model_id:
        raise SystemExit(f"member identity mismatch: {model_id}")
    if row.get("sha256") != digest or row.get("expected_sha256") != digest or row.get("sha256_exact_match") is not True:
        raise SystemExit(f"member exact digest mismatch: {model_id}")
contracts = manifest.get("variant_contracts")
expected_contracts = {
    "htdemucs_ft": {
        "member_ids": ["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"],
        "source_count": 4,
        "weights": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "weight_semantics": "DECLARED_IDENTITY_MATRIX",
    },
    "htdemucs_6s": {
        "member_ids": ["5c90dfd2"],
        "source_count": 6,
        "weights": [[1.0]],
        "weight_semantics": "DERIVED_SINGLE_MEMBER_IDENTITY",
    },
}
if contracts != expected_contracts:
    raise SystemExit("variant member/matrix contract drifted")
PY
actual_head="$(git -C "$VOKRA_ROOT" rev-parse --verify HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die 'checkout HEAD changed before inspection evidence completion'
transfer_sha="$(sha256sum "$work_dir/transfer-packet/manifest.json" | awk '{print $1}')"
grep -Fqx "$transfer_sha  manifest.json" "$work_dir/transfer-packet/manifest.sha256" \
  || die 'transfer manifest sidecar digest mismatch'
uv run --no-project --offline --python 3.12 python - \
  "$work_dir/transfer-packet" "$expected_head" "$actual_head" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
packet, expected_head, actual_head = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
paths = list(packet.iterdir())
if any(not path.is_file() or path.is_symlink() for path in paths):
    raise SystemExit("transfer packet contains a nonregular or symlink entry")
if sorted(path.name for path in paths) != [
    "htdemucs_multi_manifest.json", "manifest.json", "manifest.sha256"
]:
    raise SystemExit("transfer packet file set is not exact")
manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
identity = manifest.get("inspection_manifest")
if identity != {
    "filename": "htdemucs_multi_manifest.json",
    "sha256": hashlib.sha256((packet / "htdemucs_multi_manifest.json").read_bytes()).hexdigest(),
}:
    raise SystemExit("transfer packet inspection manifest binding drifted")
if manifest.get("expected_head") != expected_head or manifest.get("git_commit") != actual_head:
    raise SystemExit("transfer packet HEAD binding drifted")
PY
{
  echo "expected_head=$expected_head"
  echo "git_commit=$actual_head"
  echo 'runtime_status=NOT_IMPLEMENTED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'weight_digest_status=FULL_WEIGHT_DIGESTS_AUTHENTICATED'
  echo 'verdict=BLOCKED'
  echo 'blocker_exit=2'
  echo 'publication=NO_UPLOAD'
  echo "approval_sha256=$approval_sha"
  echo "license_gate_sha256=$gate_sha"
  echo "transfer_manifest_sha256=$transfer_sha"
  echo 'native_blocker=see htdemucs_multi_manifest.json blockers and per-member safe-load status'
} | tee -a "$work_dir/evidence/validation.log"
log "inspection blocked by contract: evidence=$work_dir; no conversion or upload performed"
exit 2
