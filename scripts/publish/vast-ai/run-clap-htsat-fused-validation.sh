#!/usr/bin/env bash
# VAST/Linux-only CLAP checkpoint and independent-reference inspection.
# This worker is deliberately non-publishing. The native CLAP binder remains
# disabled until its exact tensor manifest and topology audit are reviewed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/clap"
REFERENCE_DUMPER="tools/parity/clap_dump_reference.py"
MODEL_FREE_AUDIT="$PARITY_PROJECT/clap_model_free_audit.py"
DEPENDENCY_LICENSE_AUDIT="$PARITY_PROJECT/clap_dependency_license_audit.py"
UPSTREAM_REPO="laion/clap-htsat-fused"
UPSTREAM_REVISION="365dea6ef167def6676140ed93bbc43f84dabb28"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
# Model-free only stages a few metadata files plus the locked Python env. The
# real-weight inspection keeps the separate 150 GiB guard below.
MIN_MODEL_FREE_DISK_KIB=$((16 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))
CLAP_UV_CACHE_DIR="${CLAP_UV_CACHE_DIR:-/tmp/vokra-clap-uv-cache}"

log() { printf '[clap-htsat-fused-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
CLAP_SELF_TEST_TMP=""
# shellcheck disable=SC2329 # Invoked by the EXIT trap below.
cleanup_self_test() {
  [[ -n "$CLAP_SELF_TEST_TMP" ]] && rm -rf -- "$CLAP_SELF_TEST_TMP"
}

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { log 'expected HEAD must be exactly 40 lowercase hexadecimal characters'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" ]] || { log 'checkout is missing .git'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { log 'checkout must be clean'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || return 2
  [[ "$actual" == "$expected" ]] || { log "checkout HEAD $actual differs from expected $expected"; return 2; }
}

require_regular_approval_path() {
  local input="$1" path="$1" rest component current base
  [[ -n "$path" && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || return 2
  if [[ "$path" != /* ]]; then
    base="$(pwd -P)" || return 2
    path="$base/$path"
  fi
  [[ "$path" != */../* && "$path" != */.. && "$path" != *'/./'* && "$path" != *'/.' ]] || return 2
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 2
    current="$current$component"
    [[ ! -L "$current" ]] || return 2
    current="$current/"
  done
  [[ -f "$input" && ! -L "$input" ]] || return 2
}

require_approval_binding() {
  local approval="$1" expected_sha="$2"
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || { log 'approval SHA must be exactly 64 lowercase hexadecimal characters'; return 2; }
  require_regular_approval_path "$approval" || { log 'approval evidence must be a regular file with safe non-symlink ancestry'; return 2; }
  [[ "$(sha256_file "$approval")" == "$expected_sha" ]] || { log 'approval evidence SHA-256 differs from caller binding'; return 2; }
}

claim_absent_directory() {
  local path="$1"
  [[ ! -e "$path" && ! -L "$path" ]] || return 2
  mkdir "$path" || return 2
  [[ -d "$path" && ! -L "$path" ]] || return 2
}

require_preflight() {
  local project="$1"
  [[ "$project" == "$PARITY_PROJECT" ]] || { log 'CLAP reference project path is not the current dedicated project'; return 2; }
  [[ -f "$project/pyproject.toml" && ! -L "$project/pyproject.toml" ]] || { log 'CLAP pyproject.toml is missing or symlinked'; return 2; }
  [[ -f "$project/uv.lock" && ! -L "$project/uv.lock" ]] || { log 'CLAP uv.lock is missing or symlinked'; return 2; }
  [[ -f "$project/license_gate.py" && ! -L "$project/license_gate.py" ]] || { log 'CLAP license gate is missing or symlinked'; return 2; }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$project/license_gate.py" --self-test >/dev/null || { log 'CLAP license/reference gate self-test failed'; return 2; }
  # The real-weight path remains blocked until owner approval is supplied.
  log 'BLOCKED_OWNER_APPROVAL_REQUIRED/NO_UPLOAD'
  return 2
}

validate_absent_work() {
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
  local root_real project_parent project_real
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" || return 2
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || return 2
  project_real="$(cd -P "$PARITY_PROJECT" 2>/dev/null && pwd)" || return 2
  [[ "$candidate" != "$project_real" && "$candidate/" != "$project_real/"* && "$project_real/" != "$candidate/"* ]] || return 2
}

usage() {
  cat <<'EOF'
usage: run-clap-htsat-fused-validation.sh --approval-evidence <file> --approval-sha256 <64-hex> --expected-head <40-hex> [--work-dir <absent-dir>]
       run-clap-htsat-fused-validation.sh --model-free --expected-head <40-hex> [--work-dir <absent-dir>]
       run-clap-htsat-fused-validation.sh --self-test

The --model-free path is Linux/VAST-only. It resolves only the exact
Hugging Face config/preprocessor metadata with the frozen CLAP project,
authenticates the official Transformers config and preprocessing API, and
records dependency/license facts. It never acquires or loads checkpoint
weights. The approval path remains a separate real-weight inspection gate.
EOF
}

require_model_free_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'CLAP model-free audit requires VAST Linux x86_64'; return 2; }
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || { die 'cannot read VAST MemTotal'; return 2; }
  (( mem_kib >= MIN_VAST_MEM_KIB )) || { die 'VAST memory guard failed'; return 2; }
  free_kib="$(df -Pk /tmp | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || { die 'cannot read VAST free disk'; return 2; }
  (( free_kib >= MIN_MODEL_FREE_DISK_KIB )) || { die 'VAST model-free disk guard failed'; return 2; }
}

require_model_free_tooling() {
  local tool
  for tool in uv git awk find df grep sed sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { die "required tool missing: $tool"; return 2; }
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || { die 'not a Vokra checkout'; return 2; }
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] || { die 'CLAP pyproject.toml is missing or symlinked'; return 2; }
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || { die 'CLAP uv.lock is missing or symlinked'; return 2; }
  [[ -f "$PARITY_PROJECT/license_gate.py" && ! -L "$PARITY_PROJECT/license_gate.py" ]] || { die 'CLAP license gate is missing or symlinked'; return 2; }
  [[ -f "$MODEL_FREE_AUDIT" && ! -L "$MODEL_FREE_AUDIT" ]] || { die 'CLAP model-free audit is missing or symlinked'; return 2; }
  [[ -f "$DEPENDENCY_LICENSE_AUDIT" && ! -L "$DEPENDENCY_LICENSE_AUDIT" ]] || { die 'CLAP dependency/license audit is missing or symlinked'; return 2; }
  [[ -f "$VOKRA_ROOT/$REFERENCE_DUMPER" && ! -L "$VOKRA_ROOT/$REFERENCE_DUMPER" ]] || { die 'CLAP reference dumper is missing or symlinked'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
}

run_model_free() {
  local expected="$1" work="$2" metadata_dir metadata_snapshot evidence dependency_inventory dependency_inventory_rc
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head must be exactly 40 lowercase hex'; return 2; }
  require_model_free_tooling
  require_model_free_host
  require_clean_expected_head "$expected" || die 'checkout is not the caller-bound clean expected HEAD'
  [[ -n "$work" ]] || work="/tmp/vokra-clap-htsat-fused-model-free-${expected:0:12}"
  validate_absent_work "$work" || die 'model-free work-dir must be absent, disjoint, and free of symlink ancestors'
  claim_absent_directory "$work" || die 'model-free work-dir could not be claimed'
  work="$(cd "$work" && pwd)"
  metadata_dir="$work/metadata"
  evidence="$work/model-free-audit.json"
  dependency_inventory="$work/dependency-license-inventory.json"

  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$PARITY_PROJECT/license_gate.py" --self-test >/dev/null || die 'CLAP license/reference gate self-test failed'

  # Metadata-only HF snapshot. The allow-list intentionally contains no
  # checkpoint suffix; model-free closure must not materialize weights.
  UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$metadata_dir" <<'PY'
import json
import hashlib
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo, revision, output = sys.argv[1:]
output_path = Path(output)
output_path.mkdir(parents=True, exist_ok=False)
cache_dir = output_path / "hf-cache"
api = HfApi()
info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"pinned CLAP resolved revision drifted: {info.sha!r} != {revision!r}")
repo_files = sorted(api.list_repo_files(repo_id=repo, revision=revision))
card_data = getattr(info, "card_data", None)
card_data_source = "info.card_data(None)"
if card_data is None:
    card_data = {}
elif isinstance(card_data, dict):
    card_data = dict(card_data)
    card_data_source = "info.card_data"
else:
    to_dict = getattr(card_data, "to_dict", None)
    converted = to_dict() if callable(to_dict) else None
    if not isinstance(converted, dict):
        raise SystemExit("ModelInfo.card_data is neither a dict nor to_dict mapping")
    card_data = dict(converted)
    card_data_source = "info.card_data.to_dict"
repo_license_files = [
    name for name in repo_files
    if Path(name).name.upper() in {"LICENSE", "LICENSE.TXT", "LICENSE.MD"}
]
path = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "README.md",
        "LICENSE",
        "merges.txt",
        "tokenizer_config.json",
        "vocab.json",
    ],
))
if path.name != revision:
    raise SystemExit(f"pinned CLAP metadata snapshot drifted: {path.name!r} != {revision!r}")
if not (path / "config.json").is_file() or not (path / "preprocessor_config.json").is_file():
    raise SystemExit("pinned CLAP metadata snapshot is missing config or preprocessor")
for item in path.rglob("*"):
    if item.is_file() and item.suffix.lower() in {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}:
        raise SystemExit(f"model-free metadata snapshot contains weights: {item.name}")
(output_path / "materialized").mkdir()
def git_blob_sha1(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()

def local_identity(destination):
    data = destination.read_bytes()
    return {
        "local_size": len(data),
        "local_sha256": hashlib.sha256(data).hexdigest(),
        "local_git_blob_sha1": git_blob_sha1(data),
    }

tree_entries = list(api.list_repo_tree(
    repo_id=repo,
    revision=revision,
    recursive=True,
    expand=True,
))
def entry_mapping(entry):
    to_dict = getattr(entry, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    return {
        key: getattr(entry, key, None)
        for key in ("path", "size", "blob_id", "oid", "lfs")
    }

def mapping(value):
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    converted = to_dict() if callable(to_dict) else None
    return converted if isinstance(converted, dict) else {}

tree_by_path = {}
for entry in tree_entries:
    value = entry_mapping(entry)
    path_name = value.get("path")
    if path_name in {"config.json", "preprocessor_config.json"}:
        tree_by_path[path_name] = value
remote_files = {}
for filename in ("config.json", "preprocessor_config.json"):
    source = path / filename
    destination = output_path / "materialized" / filename
    if not source.is_file() or source.is_symlink() and not source.resolve().is_file():
        raise SystemExit(f"pinned CLAP metadata source is missing: {filename}")
    with destination.open("xb") as stream:
        stream.write(source.read_bytes())
    if destination.is_symlink() or not destination.is_file():
        raise SystemExit(f"materialized CLAP metadata is not a regular file: {filename}")
    tree = tree_by_path.get(filename)
    if tree is None:
        raise SystemExit(f"pinned CLAP remote tree is missing: {filename}")
    lfs = mapping(tree.get("lfs"))
    remote_size = tree.get("size")
    if remote_size is None:
        remote_size = lfs.get("size")
    if not isinstance(remote_size, int):
        raise SystemExit(f"pinned CLAP remote tree has no size: {filename}")
    lfs_sha = lfs.get("sha256") or lfs.get("oid")
    lfs_size = lfs.get("size")
    blob_id = tree.get("blob_id") or tree.get("oid")
    if lfs_sha is not None:
        if lfs_size != remote_size:
            raise SystemExit(f"pinned CLAP remote LFS size differs: {filename}")
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {remote_size}\n".encode("utf-8")
        if blob_id != git_blob_sha1(pointer):
            raise SystemExit(f"pinned CLAP remote LFS pointer differs: {filename}")
    local = local_identity(destination)
    remote_files[filename] = {
        "remote_size": remote_size,
        "remote_blob_id": blob_id,
        "remote_lfs_sha256": lfs_sha,
        "remote_lfs_size": lfs_size,
        **local,
    }
(output_path / "remote-identity.json").write_text(
    json.dumps(
        {
            "repository": repo,
            "requested_revision": revision,
            "resolved_revision": info.sha,
            "metadata_api": "HfApi.model_info/list_repo_files",
            "tree_api": "HfApi.list_repo_tree(expand=True)",
            "card_data_source": card_data_source,
            "card_data_license_status": "PRESENT" if card_data.get("license") else "MISSING",
            "card_data_license": card_data.get("license"),
            "card_data_keys": sorted(str(key) for key in card_data),
            "repo_license_file_status": "PRESENT" if repo_license_files else "MISSING",
            "repo_license_files": repo_license_files,
            "remote_files": remote_files,
            "weights": "NOT_ACQUIRED",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(output_path / "snapshot-path.txt").write_text(str(path) + "\n", encoding="utf-8")
PY

  metadata_snapshot="$metadata_dir/materialized"
  [[ -d "$metadata_snapshot" && ! -L "$metadata_snapshot" ]] || die 'materialized CLAP metadata directory is missing'
  [[ -f "$metadata_dir/remote-identity.json" && ! -L "$metadata_dir/remote-identity.json" ]] || die 'remote identity evidence is missing'

  set +e
  UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
    "$DEPENDENCY_LICENSE_AUDIT" \
    --project "$PARITY_PROJECT/pyproject.toml" \
    --lock "$PARITY_PROJECT/uv.lock" \
    --output "$dependency_inventory"
  dependency_inventory_rc=$?
  set -e
  [[ -f "$dependency_inventory" && ! -L "$dependency_inventory" ]] || die 'dependency/license inventory was not emitted'
  if [[ "$dependency_inventory_rc" != 0 ]]; then
    log "dependency/license inventory remains fail-closed (exit=$dependency_inventory_rc); retaining packet for owner review"
  fi

  UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
    "$MODEL_FREE_AUDIT" \
    --project "$PARITY_PROJECT/pyproject.toml" \
    --lock "$PARITY_PROJECT/uv.lock" \
    --config "$metadata_snapshot/config.json" \
    --preprocessor "$metadata_snapshot/preprocessor_config.json" \
    --remote-identity "$metadata_dir/remote-identity.json" \
    --dependency-inventory "$dependency_inventory" \
    --output "$evidence" || die 'CLAP model-free audit failed'
  printf '%s\n' \
    'schema=vokra-clap-htsat-fused-model-free-summary-v1' \
    'status=PASS_MODEL_FREE' \
    "expected_head=$expected" \
    "evidence_sha256=$(sha256_file "$evidence")" \
    "dependency_inventory_sha256=$(sha256_file "$dependency_inventory")" \
    'dependency_audit_status=PENDING_VAST_AUDIT' \
    'weights=NOT_ACQUIRED' \
    'model_load=NOT_PERFORMED' \
    'publication=NO_UPLOAD' \
    'owner_approval=PENDING_OWNER_APPROVAL' > "$work/summary.txt"
  log "PASS_MODEL_FREE: metadata/API/dependency/license facts audited; no checkpoint load or upload; evidence=$evidence"
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token tmp fake_project approval work rc approval_sha
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'uname -s' 'uname -m' 'MIN_VAST_MEM_KIB' \
    'MIN_MODEL_FREE_DISK_KIB' 'require_model_free_host' \
    '/proc/meminfo' 'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'snapshot_download' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$REFERENCE_DUMPER" \
    'tools/parity/clap' 'MODEL_FREE_AUDIT=' 'DEPENDENCY_LICENSE_AUDIT=' '--model-free' 'config.json' \
    'preprocessor_config.json' 'remote-identity.json' 'materialized' \
    'HfApi' 'list_repo_tree' 'resolved_revision' 'card_data_source' \
    'card_data_license_status' 'repo_license_file_status' '--remote-identity' '--dependency-inventory' \
    'remote_files' 'local_git_blob_sha1' 'remote_lfs_sha256' 'dependency_audit_status=PENDING_VAST_AUDIT' 'weights=NOT_ACQUIRED' \
    'transformers_clap_model_source_sha256' 'tensor_manifest' \
    'owner_review_candidate.py' 'owner_review_candidate.json' 'PENDING_OWNER_REVIEW' 'SIGNED_COMMERCIAL' \
    'row_sha256' 'docs/license-audit.md' 'payload_sha256' \
    'validate_feature_extractor_serializer_contract' 'processor_class' \
    'INSPECTION_ONLY' 'no upload' 'VOKRA_CLAP_REAL_GGUF' 'GGUFReader' \
    'clap_dump_reference.py" --self-test' \
    'git status --porcelain'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*publish-one\.sh|.*upload\.sh)([[:space:]]|$)' "$path" >/dev/null; then
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
  if "$path" --approval-evidence /tmp/a --approval-sha256 "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    log 'self-test FAIL: missing --expected-head accepted'
    fail=1
  fi
  if "$path" --approval-evidence /tmp/a --approval-sha256 "$(printf '0%.0s' {1..64})" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --expected-head accepted'
    fail=1
  fi
  if "$path" --approval-evidence /tmp/a --approval-sha256 "$(printf '0%.0s' {1..64})" --approval-sha256 "$(printf '1%.0s' {1..64})" --expected-head "$(printf '0%.0s' {1..40})" >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --approval-sha256 accepted'
    fail=1
  fi
  if "$path" --model-free >/dev/null 2>&1; then
    log 'self-test FAIL: model-free invocation without --expected-head accepted'
    fail=1
  fi
  tmp="$(mktemp -d)"
  tmp="$(cd -P "$tmp" && pwd)"
  CLAP_SELF_TEST_TMP="$tmp"
  trap cleanup_self_test EXIT
  fake_project="$tmp/project"
  mkdir -p "$fake_project"
  printf '[project]\nname = "synthetic-clap"\nversion = "0.0.0"\n' >"$fake_project/pyproject.toml"
  if require_preflight "$fake_project" "$tmp/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: missing dedicated lock/gate was accepted'
    fail=1
  fi
  : >"$fake_project/uv.lock"
  : >"$fake_project/license_gate.py"
  : >"$fake_project/license_gate_manifest.json"
  if require_preflight "$fake_project" "$tmp/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: placeholder dedicated lock/gate overrode blocked disposition'
    fail=1
  fi
  approval="$tmp/approval.json"
  work="$tmp/work"
  printf '{}\n' >"$approval"
  approval_sha="$(sha256_file "$approval")"
  validate_absent_work "$work" || { log 'self-test FAIL: safe absent work path rejected'; fail=1; }
  if validate_absent_work "$tmp/../escape" >/dev/null 2>&1; then
    log 'self-test FAIL: dot-dot work path accepted'; fail=1
  fi
  mkdir "$tmp/real"
  ln -s "$tmp/real" "$tmp/link"
  if validate_absent_work "$tmp/link/work" >/dev/null 2>&1; then
    log 'self-test FAIL: symlink-ancestor work path accepted'; fail=1
  fi
  if require_approval_binding "$approval" "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    log 'self-test FAIL: wrong approval SHA accepted'; fail=1
  fi
  require_approval_binding "$approval" "$approval_sha" || { log 'self-test FAIL: correct approval SHA rejected'; fail=1; }
  if "$path" --self-test --self-test >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --self-test accepted'; fail=1
  fi
  set +e
  VOKRA_PUBLISH_ON_VAST=1 CLAP_UV_CACHE_DIR="$tmp/cache" \
    "$path" --approval-evidence "$approval" --approval-sha256 "$approval_sha" \
    --expected-head "$(printf '0%.0s' {1..40})" --work-dir "$work" >/dev/null 2>&1
  rc=$?
  set -e
  if [[ "$rc" != 2 || -e "$work" || -e "$tmp/cache" ]]; then
    log 'self-test FAIL: production-shaped missing-lock probe had effects or wrong status'
    fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/$REFERENCE_DUMPER" --self-test >/dev/null; then
    log 'self-test FAIL: independent dumper self-test failed'
    fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$MODEL_FREE_AUDIT" --self-test >/dev/null; then
    log 'self-test FAIL: model-free audit self-test failed'
    fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$DEPENDENCY_LICENSE_AUDIT" --self-test >/dev/null; then
    log 'self-test FAIL: dependency/license audit self-test failed'
    fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$PARITY_PROJECT/owner_review_candidate.py" --self-test >/dev/null; then
    log 'self-test FAIL: owner-review candidate self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-clap-htsat-fused-validation"
approval_evidence=''
approval_sha256=''
expected_head=''
self=0
model_free=0
seen_self=0
seen_model_free=0
seen_work=0
seen_approval=0
seen_approval_sha=0
seen_head=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --model-free) (( seen_model_free == 0 )) || die 'duplicate --model-free'; seen_model_free=1; model_free=1; shift ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; (( $# >= 2 )) || die '--work-dir requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--work-dir must be a nonempty path'; seen_work=1; work_dir="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) || die '--approval-evidence requires a file'; [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence must be a nonempty file path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; (( $# >= 2 )) || die '--approval-sha256 requires a SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; seen_approval_sha=1; approval_sha256="$2"; shift 2 ;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; (( $# >= 2 )) || die '--expected-head requires a commit'; [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; seen_head=1; expected_head="$2"; shift 2 ;;
    -h|--help) [[ $self == 0 && $# == 1 ]] || die '--help cannot be combined with other arguments'; usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_work" == 0 && "$seen_model_free" == 0 && "$seen_approval" == 0 && "$seen_approval_sha" == 0 && "$seen_head" == 0 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

if (( model_free )); then
  [[ "$seen_approval" == 0 && "$seen_approval_sha" == 0 ]] || die '--model-free does not accept approval evidence'
  [[ "$seen_head" == 1 ]] || die '--model-free requires --expected-head'
  run_model_free "$expected_head" "$work_dir"
  exit $?
fi

[[ -n "$approval_evidence" ]] || die '--approval-evidence is required'
[[ "$seen_approval_sha" == 1 ]] || die '--approval-sha256 is required'
[[ "$seen_head" == 1 ]] || die '--expected-head is required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required for caller binding'
require_clean_expected_head "$expected_head" || die 'checkout is not the caller-bound clean expected HEAD'
require_approval_binding "$approval_evidence" "$approval_sha256" || die 'approval evidence caller binding is invalid'
require_preflight "$PARITY_PROJECT" || die 'CLAP owner approval gate is unresolved; refuse before host/work/network'

[[ "$(uname -s)" == Linux ]] || die 'inspection is Linux/VAST-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$VOKRA_ROOT/$REFERENCE_DUMPER" ]] || die 'reference dumper is missing'

mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
validate_absent_work "$work_dir" || die 'work-dir must be absent, disjoint, and free of symlink ancestors'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

claim_absent_directory "$work_dir" || die 'work-dir could not be atomically claimed as an absent directory'
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
printf 'repository=%s\nrevision=%s\n' "$UPSTREAM_REPO" "$UPSTREAM_REVISION" > "$work_dir/validation.log"
cargo fmt --all -- --check >> "$work_dir/validation.log" 2>&1
cargo metadata --no-deps --format-version 1 >> "$work_dir/validation.log" 2>&1

snapshot_path_file="$work_dir/snapshot-path.txt"
UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$work_dir/hf-cache" "$snapshot_path_file" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, cache_dir, output = sys.argv[1:]
path = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.safetensors.index.json"],
))
if path.name != revision:
    raise SystemExit(f"snapshot revision drift: {path.name!r} != {revision!r}")
for required in ("config.json", "preprocessor_config.json"):
    if not (path / required).is_file():
        raise SystemExit(f"missing pinned upstream file: {required}")
Path(output).write_text(str(path) + "\n", encoding="utf-8")
PY

snapshot_dir="$(< "$snapshot_path_file")"
UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$VOKRA_ROOT/$REFERENCE_DUMPER" --model-dir "$snapshot_dir" \
  --output-dir "$work_dir/reference"
find "$snapshot_dir" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work_dir/upstream-file-sha256.txt"
sha256sum "$work_dir/reference"/* > "$work_dir/reference-sha256.txt"
if [[ -n "${VOKRA_CLAP_REAL_GGUF:-}" ]]; then
  [[ -s "$VOKRA_CLAP_REAL_GGUF" ]] || die "VOKRA_CLAP_REAL_GGUF is missing or empty"
  UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
    "$VOKRA_CLAP_REAL_GGUF" "$work_dir/supplied-gguf-manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from gguf import GGUFReader

path, output = sys.argv[1:]
reader = GGUFReader(path)
manifest = {
    "file": str(Path(path).resolve()),
    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    "identity_status": "UNAUTHENTICATED_SUPPLIED_FILE",
    "public_artifact_status": "NOT_ASSERTED",
    "tensor_manifest": {
        tensor.name: {
            "shape": [int(axis) for axis in tensor.shape],
            "dtype": str(tensor.tensor_type),
        }
        for tensor in sorted(reader.tensors, key=lambda item: item.name)
    },
    "parity_status": "INSPECTION_ONLY",
}
Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "supplied_gguf_sha256=$(sha256sum "$VOKRA_CLAP_REAL_GGUF" | awk '{print $1}')" | tee -a "$work_dir/validation.log"
  echo 'supplied_gguf_identity=UNAUTHENTICATED_SUPPLIED_FILE' | tee -a "$work_dir/validation.log"
else
  echo 'supplied_gguf=NOT_SUPPLIED' | tee -a "$work_dir/validation.log"
fi
require_clean_expected_head "$expected_head" || die 'checkout HEAD/clean state changed before final inspection summary'
{
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=NO_UPLOAD'
  echo "upstream_repo=$UPSTREAM_REPO"
  echo "upstream_revision=$UPSTREAM_REVISION"
} | tee -a "$work_dir/validation.log"
log "inspection complete: evidence remains at $work_dir; no upload or publication was performed"
