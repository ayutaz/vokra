#!/usr/bin/env bash
# VAST/Linux-only source-contract inspection for MOSS Audio Tokenizer Nano.
# This phase authenticates fixed-revision bytes and a model-free API/shape
# route. It never converts, executes model weights, publishes, or uploads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/moss_audio_tokenizer_nano"
INSPECTOR="$PROJECT/source_contract_inspector.py"
HF_REPOSITORY="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
HF_REVISION="6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
MIN_MEM_KIB=30000000
MIN_DISK_KIB=5000000
DEFAULT_WORK_DIR="/dev/shm/vokra-moss-audio-tokenizer-nano-inspection"

log() { printf '[moss-tokenizer-nano-inspection] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 2; }

canonical_candidate() {
  local value="$1" suffix='' parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"
  [[ -n "$value" ]] || { die 'path is empty'; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || { die "path contains a symlink ancestor: $parent"; }
    parent="$(dirname "$parent")"
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die 'path has no canonical parent'; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die 'path parent is not a real directory'; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

self_test() {
  local script="${BASH_SOURCE[0]}" fail=0 token
  [[ -f "$INSPECTOR" ]] || { log 'self-test FAIL: inspector missing'; fail=1; }
  for token in "$HF_REPOSITORY" "$HF_REVISION" 'source_contract_inspector.py' \
    'download.pytorch.org/whl/cpu' '2.7.1+cpu' 'torch.version.cuda is None' 'torch.cuda.is_available' \
    '.gitattributes' '__init__.py' 'model-00001-of-00001.safetensors' \
    'cardData_license' 'private' 'gated' 'disabled' \
    'snapshot_download' 'list_repo_tree' 'lfs_payload_sha256' \
    'AutoConfig.from_pretrained' 'AutoModel.from_config' 'init_empty_weights' \
    'api_path' 'api_methods' 'MossAudioTokenizerModel' 'AUTHENTICATED_META_SHAPE_PROBE' 'AUTHENTICATED_EVIDENCE_COMPLETE' \
    '1x768x2' '1x1x15360' '1x2x7680' \
    'INSPECTION_ERROR' 'weights_loaded' 'weights_executed' 'NO_UPLOAD' \
    'REVIEWED' 'BLOCKED_UNRESOLVED_PYTHON_CLOSURE_API_RUNTIME_PARITY' \
    'vokra_checkout' 'source_and_weight_review' 'docs/license-audit.md:671' \
    'refusing pre-existing output' \
    'exist_ok=False' 'reserve_output' '--expected-head' \
    '--work-dir' 'VOKRA_PUBLISH_ON_VAST' 'findmnt' 'CARGO_BUILD_JOBS'; do
    if ! grep -Fq -- "$token" "$script" && ! grep -Fq -- "$token" "$INSPECTOR"; then
      log "self-test FAIL: missing contract token: $token"; fail=1
    fi
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check)' "$INSPECTOR" >/dev/null || \
    sed '/grep -En/d' "$script" | grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check)' >/dev/null; then
    log 'self-test FAIL: conversion/publication/Cargo command found'; fail=1
  fi
  if grep -En '(^|[;&|][[:space:]])(python|python3|pip)([[:space:]]|$)' "$script" >/dev/null; then
    log 'self-test FAIL: bare Python/pip command found'; fail=1
  fi
  if grep -Fq 'AutoModel.from_pretrained' "$INSPECTOR"; then
    log 'self-test FAIL: model from_pretrained route found'; fail=1
  fi
  if awk '!/grep -Eq.*allow_patterns/ && /allow_patterns=[^#]*model-00001-of-00001\.safetensors/' "$script" | grep -q .; then
    log 'self-test FAIL: weight shard appears in snapshot allow-patterns'; fail=1
  fi
  if sed -n '/^selected = {/,/^}/p' "$script" | grep -Fq '"LICENSE"'; then
    log 'self-test FAIL: nonexistent LICENSE appears in fixed server tree'; fail=1
  fi
  if ! grep -Fq 'weight payload was materialized' "$INSPECTOR" || ! grep -Fq 'content_not_downloaded' "$INSPECTOR"; then
    log 'self-test FAIL: snapshot validator does not enforce server-only weight identity'; fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$INSPECTOR" --self-test >/dev/null; then
    log 'self-test FAIL: Python inspector self-test failed'; fail=1
  fi
  local sync_line download_line probe_line
  sync_line="$(grep -n '^uv sync --project' "$script" | tail -n 1 | cut -d: -f1)"
  download_line="$(grep -n 'snapshot_download(' "$script" | tail -n 1 | cut -d: -f1)"
  # The self-test searches for the literal shell variable reference.
  # shellcheck disable=SC2016
  probe_line="$(grep -n -- 'python "\$INSPECTOR"' "$script" | tail -n 1 | cut -d: -f1)"
  if [[ -z "$sync_line$download_line$probe_line" || "$sync_line" -ge "$download_line" || "$download_line" -ge "$probe_line" ]]; then
    log 'self-test FAIL: sync/acquisition/probe order is not fail-closed'; fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="$DEFAULT_WORK_DIR"
expected_head=""
self=0
seen_self=0
seen_head=0
seen_work=0
while (($#)); do
  case "$1" in
    --self-test)
      (( seen_self == 0 )) || die 'duplicate --self-test'
      seen_self=1; self=1; shift ;;
    --expected-head)
      (( seen_head == 0 )) || die 'duplicate --expected-head'
      [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'
      seen_head=1; expected_head="$2"; shift 2 ;;
    --work-dir)
      (( seen_work == 0 )) || die 'duplicate --work-dir'
      [[ $# -ge 2 && "$2" = /* && "$2" != -* ]] || die '--work-dir requires an absolute path'
      seen_work=1; work_dir="$2"; shift 2 ;;
    -h|--help)
      cat >&2 <<'EOF'
usage: run-moss-audio-tokenizer-nano-inspection.sh --expected-head HEX40 [--work-dir ABSENT_TMPFS_DIR]
       run-moss-audio-tokenizer-nano-inspection.sh --self-test

VAST-only fixed-revision source-contract inspection. The output is blocked
evidence: exact non-weight hashes, server-only weight identity, an official
Transformers meta-device route, and decoder tap shapes are collected without
loading model weights.
EOF
      exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if (( self )); then
  [[ "$seen_self" == 1 && "$seen_head" == 0 && "$seen_work" == 0 ]] \
    || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$seen_head" == 1 ]] || die '--expected-head is required'
[[ "$seen_self" == 0 ]] || die '--self-test cannot be combined with normal arguments'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST is required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD differs from --expected-head'
[[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" ]] || die 'Nano uv project is missing'
[[ -f "$INSPECTOR" ]] || die 'Nano source-contract inspector is missing'

canonical_work="$(canonical_candidate "$work_dir")" || exit 2
canonical_root="$(canonical_candidate "$ROOT")" || exit 2
paths_overlap "$canonical_work" "$canonical_root" && die 'work directory overlaps checkout'
[[ ! -e "$work_dir" && ! -L "$work_dir" ]] || die 'work directory must be absent'
parent="$(dirname "$canonical_work")"
[[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work directory parent must be tmpfs'

mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_MEM_KIB" ]] || die 'RAM guard failed'
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_DISK_KIB" ]] || die 'tmpfs free-disk guard failed'
for command in git uv awk df find findmnt sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing required tool: $command"
done

export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="${MOSS_NANO_UV_CACHE_DIR:-/dev/shm/vokra-moss-nano-uv-cache}"
mkdir "$work_dir"
mkdir "$work_dir/hf"

# This sync installs only the locked reference tooling. It does not acquire a
# model and is intentionally after all host/worktree guards.
uv sync --project "$PROJECT" --frozen --python 3.12

# Bind the synced environment to the exact Linux CPU wheel before acquiring
# any upstream source bytes. This imports Torch only; no model code/weights.
uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python -c \
  'import platform,torch; assert platform.system() == "Linux" and platform.machine() == "x86_64"; assert torch.__version__ == "2.7.1+cpu"; assert torch.version.cuda is None; assert not torch.cuda.is_available(); print("Nano CPU closure: torch=2.7.1+cpu cuda=None")'

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python - \
  "$HF_REPOSITORY" "$HF_REVISION" "$work_dir/hf" "$work_dir/server-tree.json" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo, revision, destination, tree_path = sys.argv[1:]
selected = {
    ".gitattributes", "README.md", "__init__.py", "config.json",
    "configuration_moss_audio_tokenizer.py",
    "modeling_moss_audio_tokenizer.py",
    "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
}
materialized = selected - {"model-00001-of-00001.safetensors"}
api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HF"))
info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision or not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
    raise SystemExit(f"HF revision mismatch: {info.sha!r} != {revision!r}")
card_data = getattr(info, "cardData", None)
if card_data is None:
    card_data = getattr(info, "card_data", None)
card_license = card_data.get("license") if isinstance(card_data, dict) else getattr(card_data, "license", None)
model_info = {
    "id": getattr(info, "id", None),
    "sha": info.sha,
    "private": getattr(info, "private", None),
    "gated": getattr(info, "gated", None),
    "disabled": getattr(info, "disabled", None),
    "cardData_license": card_license,
}
if model_info != {
    "id": repo, "sha": revision, "private": False, "gated": False,
    "disabled": False, "cardData_license": "apache-2.0",
}:
    raise SystemExit(f"HF model_info contract mismatch: {model_info!r}")
rows = {}
for item in api.list_repo_tree(repo_id=repo, revision=revision, recursive=True, expand=True):
    kind = getattr(item, "type", None)
    if kind in {"directory", "folder", "dir"} or item.__class__.__name__ == "RepoFolder":
        continue
    path = getattr(item, "path", None)
    if path not in selected:
        continue
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path or path.startswith("/") or ".." in Path(path).parts:
        raise SystemExit(f"unsafe selected path: {path!r}")
    size = getattr(item, "size", None)
    blob = getattr(item, "blob_id", None) or getattr(item, "oid", None)
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise SystemExit(f"invalid selected size: {path}")
    if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
        raise SystemExit(f"missing selected Git blob identity: {path}")
    lfs = getattr(item, "lfs", None)
    lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if lfs_sha is not None:
        if not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha) or lfs_size != size:
            raise SystemExit(f"invalid selected LFS identity: {path}")
        pointer = ("version https://git-lfs.github.com/spec/v1\n" f"oid sha256:{lfs_sha}\nsize {size}\n").encode()
        pointer_blob = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
        if pointer_blob != blob:
            raise SystemExit(f"selected LFS pointer Git blob mismatch: {path}")
        row = {"path": path, "size": size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": blob, "lfs_payload_sha256": lfs_sha}
    else:
        row = {"path": path, "size": size, "git_blob_sha1": blob, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None}
    if path in rows:
        raise SystemExit(f"duplicate selected path: {path}")
    rows[path] = row
if set(rows) != selected:
    raise SystemExit(f"selected tree incomplete: {sorted(selected - set(rows))}")
snapshot = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=destination,
    allow_patterns=sorted(materialized),
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
))
if snapshot.resolve() != Path(destination).resolve():
    raise SystemExit("snapshot local_dir mismatch")
Path(tree_path).write_text(json.dumps({
    "repository": repo,
    "revision": revision,
    "resolved_revision": info.sha,
    "model_info": model_info,
    "files": [rows[name] for name in sorted(rows)],
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python "$INSPECTOR" \
  --repository "$HF_REPOSITORY" --revision "$HF_REVISION" \
  --vokra-root "$ROOT" --expected-head "$expected_head" \
  --snapshot "$work_dir/hf" --server-tree "$work_dir/server-tree.json" \
  --output "$work_dir/evidence"
inspection_rc=$?
set -e
[[ "$inspection_rc" == 2 ]] || die "inspector returned unexpected status: $inspection_rc"
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'inspection manifest is missing'

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python - \
  "$work_dir/evidence/manifest.json" "$expected_head" <<'PY'
import json
import sys

def reject(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate manifest key: {key}")
        result[key] = value
    return result

manifest = json.loads(open(sys.argv[1], encoding="utf-8").read(), object_pairs_hook=reject)
expected_head = sys.argv[2]
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "cpu_status": "BLOCKED_UNRESOLVED_PYTHON_CLOSURE_API_RUNTIME_PARITY",
    "metal_status": "BLOCKED_BY_CPU",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "collection_status": "AUTHENTICATED",
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"inspection did not complete the source contract: {key}={manifest.get(key)!r}")
files = manifest.get("files")
if not isinstance(files, list) or len(files) != 8:
    raise SystemExit("inspection file identities are incomplete")
materialized = [row for row in files if row.get("materialized") is True]
server_only = [row for row in files if row.get("materialized") is False]
if len(materialized) != 7 or any(row.get("status") != "AUTHENTICATED" for row in materialized):
    raise SystemExit("materialized source identities are incomplete")
if len(server_only) != 1 or server_only[0].get("path") != "model-00001-of-00001.safetensors" or server_only[0].get("status") != "AUTHENTICATED_SERVER_IDENTITY_ONLY" or server_only[0].get("content_not_downloaded") is not True or not server_only[0].get("lfs_payload_sha256"):
    raise SystemExit("server-only weight identity is incomplete")
if manifest.get("license") != {
    "status": "REVIEWED",
    "source_and_weight_review": "REVIEWED",
    "source_and_weight_license": "Apache-2.0",
    "source_and_weight_conclusion": "Commercial",
    "owner_signoff": "2026-08-01 yousan",
    "owner_signoff_citation": "docs/license-audit.md:671",
    "license_file_present": False,
    "hf_cardData_license": "apache-2.0",
}:
    raise SystemExit("license-file absence/cardData metadata/owner signoff facts are not bound")
checkout = manifest.get("vokra_checkout")
if checkout != {"expected_head": expected_head, "head": expected_head, "clean": True}:
    raise SystemExit(f"Vokra checkout binding is not exact: {checkout!r}")
if manifest.get("unresolved_gates") != {
    "python_dependency_closure": "UNRESOLVED",
    "transformers_api_compatibility": "UNRESOLVED",
    "real_weight_runtime": "UNRESOLVED",
    "numerical_parity": "NOT_RUN",
    "overall_execution_approval": "NOT_APPROVED",
}:
    raise SystemExit("unresolved approval gates were weakened")
route = manifest.get("transformers_route")
if not isinstance(route, dict) or route.get("status") != "AUTHENTICATED_META_SHAPE_PROBE" or route.get("weights_loaded") is not False or route.get("weights_executed") is not False:
    raise SystemExit("inspection did not authenticate the model-free Transformers route")
if route.get("api_path") != {
    "config": "transformers.AutoConfig.from_pretrained",
    "model": "transformers.AutoModel.from_config",
    "trust_remote_code": True,
    "local_files_only": True,
} or route.get("api_methods") != ["encode", "decode", "forward", "create_decode_session"]:
    raise SystemExit("inspection did not authenticate the official Nano API path")
if route.get("taps") != [
    {"name": "quantizer", "shape": "1x768x2"},
    {"name": "decoder_0", "shape": "1x192x8"},
    {"name": "decoder_1", "shape": "1x768x8"},
    {"name": "decoder_2", "shape": "1x384x16"},
    {"name": "decoder_3", "shape": "1x768x16"},
    {"name": "decoder_4", "shape": "1x384x32"},
    {"name": "decoder_5", "shape": "1x768x32"},
    {"name": "decoder_6", "shape": "1x384x64"},
    {"name": "decoder_7", "shape": "1x240x64"},
    {"name": "decoder_8", "shape": "1x1x15360"},
]:
    raise SystemExit("inspection decoder tap shapes are not the authenticated Nano contract")
if route.get("audio_shape") != "1x2x7680":
    raise SystemExit("inspection decoded audio shape is not the authenticated Nano contract")
PY

(
  cd "$work_dir"
  find hf evidence -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
log "source contract evidence complete but blocked; recover $work_dir/evidence and destroy the VAST instance"
exit 2
