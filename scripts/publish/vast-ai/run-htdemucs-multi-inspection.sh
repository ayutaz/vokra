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

usage() {
  cat <<'EOF'
usage: run-htdemucs-multi-inspection.sh [--work-dir <empty-dir>]
       run-htdemucs-multi-inspection.sh --self-test

VAST/Linux-only inspection of the pinned official Demucs source and five
registry checkpoints. Safe loading uses torch.load(weights_only=True) only;
unsafe pickle fallback is forbidden. The result is INSPECTION_ONLY and no
GGUF conversion, upload, publication, or parity verdict is produced.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'MIN_VAST_MEM_KIB' '/proc/meminfo' \
    'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'facebookresearch/demucs.git' \
    "$UPSTREAM_REVISION" "$WEIGHT_ROOT" 'f7e0c4bc-ba3fe64a.th' \
    'd12395a8-e57c48e6.th' '92cfc3b6-ef3bcb9c.th' '04573f0d-f3cf25b2.th' \
    '5c90dfd2-34c22ccb.th' 'weights_only=True' 'no pickle fallback' \
    'KNOWN_HEAD_BYTES' '84141271' '54996327' \
    'htdemucs_multi_inspect.py' 'INSPECTION_ONLY' 'NOT_IMPLEMENTED' 'UNSUPPORTED' 'BLOCKED_BY_CPU' 'NOT_RUN' 'NO_UPLOAD' \
    'git status --porcelain' 'htdemucs_multi_inspect.py --self-test' \
    'response-packet' 'x-amz-version-id' 'x-amz-meta-s3cmd-attrs' \
    'sha256_filename_prefix_match' 'sha256_exact_match' \
    'FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER' 'FULL_WEIGHT_DIGESTS_AUTHENTICATED' \
    'expected_sha256' 'response member id mismatch' \
    'inspection_status' 'COMPLETE' 'ERROR' 'variant_contracts' 'flattened 2,132-tensor' \
    'safe_global_allowlist' 'BLOCKED_SOURCE_ALLOWLIST' 'verdict=BLOCKED' 'blocker_exit=2'; do
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
  if ! UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" \
    --python 3.12 python "$INSPECTOR" --self-test >/dev/null; then
    log 'self-test FAIL: inspector self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-htdemucs-multi-inspection"
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires a path'; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$work_dir" == "/workspace/vokra-htdemucs-multi-inspection" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'HT-Demucs checkpoint work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$INSPECTOR" ]] || die 'HT-Demucs inspector is missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work-dir must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv curl sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/source" "$work_dir/weights" "$work_dir/response" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR"
{
  echo "upstream_url=$UPSTREAM_URL"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "weight_root=$WEIGHT_ROOT"
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'publication=NO_UPLOAD'
  cargo fmt --all -- --check
  cargo metadata --no-deps --format-version 1
} > "$work_dir/evidence/validation.log" 2>&1

git clone --filter=blob:none --no-checkout "$UPSTREAM_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$UPSTREAM_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
[[ "$(git -C "$work_dir/source/repo" rev-parse HEAD)" == "$UPSTREAM_REVISION" ]] || die 'pinned Demucs source revision mismatch'
for config in htdemucs_ft.yaml htdemucs_6s.yaml; do
  [[ -f "$work_dir/source/repo/demucs/remote/$config" ]] || die "official config missing: $config"
done

for member in "${MEMBERS[@]}"; do
  curl --fail --location --retry 3 --silent --show-error \
    --dump-header "$work_dir/response/$member.headers" \
    --write-out '%{http_code}\t%{url_effective}\t%{size_download}\n' \
    "$WEIGHT_ROOT/$member" --output "$work_dir/weights/$member" \
    > "$work_dir/response/$member.meta"
done

UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$work_dir/response" "$work_dir/weights" "$work_dir/evidence/response-packet.json" <<'PY'
import hashlib
import json
import sys
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
Path(output).write_text(json.dumps({"members": rows}, sort_keys=False, indent=2) + "\n", encoding="utf-8")
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
UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$work_dir/evidence/htdemucs_multi_manifest.json" <<'PY'
import json
import sys
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
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
{
  echo 'runtime_status=NOT_IMPLEMENTED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'weight_digest_status=FULL_WEIGHT_DIGESTS_AUTHENTICATED'
  echo 'verdict=BLOCKED'
  echo 'blocker_exit=2'
  echo 'native_blocker=see htdemucs_multi_manifest.json blockers and per-member safe-load status'
} | tee -a "$work_dir/evidence/validation.log"
log "inspection blocked by contract: evidence=$work_dir; no conversion or upload performed"
exit 2
