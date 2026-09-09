#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/voxcpm_0_5b_inspect.py"
PREPARER="$ROOT/tools/parity/voxcpm_0_5b_prepare_checkpoint.py"
REFERENCE="$ROOT/tools/parity/voxcpm_0_5b_reference.py"
MODEL_REPOSITORY="openbmb/VoxCPM-0.5B"
MODEL_REVISION="e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_REPOSITORY="https://github.com/OpenBMB/VoxCPM.git"
SOURCE_REVISION="38a76704ee67935ccbafbe5b6725e83dbb1e9305"
PUBLIC_REPOSITORY="vokra/voxcpm-0.5b"
PUBLIC_REVISION="ee0ca6d5b9fab27bbb626b5cb3f01236e582d004"
AUDIOVAE_SOURCE="src/voxcpm/modules/audiovae/audio_vae.py"
TOKENIZER_FILES='["config.json","special_tokens_map.json","tokenizer.json","tokenizer_config.json"]'
die() { echo "voxcpm-vast: ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else
    parent="$(dirname "$path")"; base="$(basename "$path")"
    parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"
  fi
}

canonical_absent_path() {
  local path="$1" target="$1" rest component current=/ suffix='' real
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do
    component="$(basename "$target")"; suffix="/$component$suffix"; target="$(dirname "$target")"
  done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  real="$(cd -P "$target" && pwd)" || return 1; printf '%s%s\n' "$real" "$suffix"
}

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
  [[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout is missing'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse HEAD)" || die 'cannot read checkout HEAD'
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from --expected-head $expected"
}

validate_approval() {
  local path="$1" expected_head="$2" supplied_sha="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$supplied_sha" "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$AUDIOVAE_SOURCE" "$TOKENIZER_FILES" <<'PY'
import hashlib, json, pathlib, sys
path, expected_head, supplied_sha, model_repo, model_rev, source_repo, source_rev, public_repo, public_rev, audio_vae_source, tokenizer_files = sys.argv[1:]
raw = pathlib.Path(path).read_bytes()
if hashlib.sha256(raw).hexdigest() != supplied_sha:
    raise SystemExit("approval bytes changed")
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate approval key: " + key)
        result[key] = value
    return result
data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
keys = {"schema", "status", "disposition", "expected_head", "model_repository", "model_revision", "source_repository", "source_revision", "public_repository", "public_revision", "audio_vae_source", "tokenizer_files", "license_status", "dependency_status", "audio_vae_status", "tokenizer_status", "native_status", "no_upload", "scope_sha256"}
if set(data) != keys:
    raise ValueError("approval schema is not exact")
expected = {
    "schema": "vokra-voxcpm-0.5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY",
    "expected_head": expected_head, "model_repository": model_repo, "model_revision": model_rev,
    "source_repository": source_repo, "source_revision": source_rev, "public_repository": public_repo,
    "public_revision": public_rev, "audio_vae_source": audio_vae_source, "tokenizer_files": json.loads(tokenizer_files),
    "license_status": "DOCS_SIGNED_APACHE_2_0", "dependency_status": "BLOCKED_UNREVIEWED",
    "audio_vae_status": "UNRESOLVED", "tokenizer_status": "UNRESOLVED",
    "native_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "no_upload": True,
}
if any(data[key] != value for key, value in expected.items()):
    raise ValueError("approval identity or unresolved gate mismatch")
scope = {key: data[key] for key in keys if key != "scope_sha256"}
if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
    raise ValueError("approval scope mismatch")
PY
}

blocked_composite_gate() {
  echo 'voxcpm-vast: ERROR: BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE: no cache, download, model, or Cargo work is permitted' >&2
  return 2
}

write_approval_fixture() {
  local path="$1" expected_head="$2"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$AUDIOVAE_SOURCE" "$TOKENIZER_FILES" <<'PY'
import hashlib, json, pathlib, sys
path, head, model_repo, model_rev, source_repo, source_rev, public_repo, public_rev, audio_vae_source, tokenizer_files = sys.argv[1:]
data = {
    "schema": "vokra-voxcpm-0.5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY",
    "expected_head": head, "model_repository": model_repo, "model_revision": model_rev,
    "source_repository": source_repo, "source_revision": source_rev, "public_repository": public_repo,
    "public_revision": public_rev, "audio_vae_source": audio_vae_source, "tokenizer_files": json.loads(tokenizer_files),
    "license_status": "DOCS_SIGNED_APACHE_2_0", "dependency_status": "BLOCKED_UNREVIEWED",
    "audio_vae_status": "UNRESOLVED", "tokenizer_status": "UNRESOLVED",
    "native_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "no_upload": True,
}
data["scope_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
pathlib.Path(path).write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
PY
}

self_test() {
  local failed=0 token
  for token in "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$AUDIOVAE_SOURCE" \
    'ee0ca6d5b9fab27bbb626b5cb3f01236e582d004' \
    'weights_only=True' 'INSPECTION_ONLY' 'AUTHENTICATED_EVIDENCE_COMPLETE' \
    'NO_UPLOAD' 'audiovae.pth' 'pytorch_model.bin' 'local_dir' \
    'voxcpm_0_5b_prepare_checkpoint.py' 'voxcpm_0_5b_reference.py' \
    'VOXCPM_REFERENCE_PACKET' 'target_text' 'torch.randn' \
    'PREPARATION_EVIDENCE_COMPLETE' 'REFERENCE_EVIDENCE_COMPLETE' \
    '--expected-head' '--approval-evidence' '--approval-sha256' '--work-dir' \
    'BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE' 'duplicate --expected-head' \
    'duplicate approval key' 'approval schema is not exact' 'work path overlaps checkout' \
    'DOCS_SIGNED_APACHE_2_0'; do
    grep -Fq -- "$token" "$INSPECTOR" "$PREPARER" "$REFERENCE" "$0" || { echo "missing contract: $token" >&2; failed=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then
    echo 'upload/publish command found' >&2; failed=1
  fi
  if ! (
    fixture="$(mktemp "${TMPDIR:-/tmp}/voxcpm-approval.XXXXXX")"
    cleanup_fixture() { rm -f -- "$fixture" "$fixture.blocked"; }
    trap cleanup_fixture EXIT
    fixture_head="$(printf '0%.0s' {1..40})"
    write_approval_fixture "$fixture" "$fixture_head"
    fixture_sha="$(sha256_file "$fixture")"
    validate_approval "$fixture" "$fixture_head" "$fixture_sha"
    if "$0" --expected-head "$fixture_head" --expected-head "$fixture_head" --approval-evidence "$fixture" --approval-sha256 "$fixture_sha" >/dev/null 2>&1 || \
      "$0" --expected-head "$fixture_head" --approval-evidence "$fixture" --approval-evidence "$fixture" --approval-sha256 "$fixture_sha" >/dev/null 2>&1; then
      exit 1
    fi
    if blocked_composite_gate 2>"$fixture.blocked"; then exit 1; fi
    grep -Fq 'BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE' "$fixture.blocked"
  ); then
    echo 'valid blocked approval or gate self-test failed' >&2; failed=1
  fi
  if "$0" --self-test --self-test >/dev/null 2>&1 || \
    "$0" --expected-head bad >/dev/null 2>&1 || \
    "$0" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1 || \
    "$0" --approval-sha256 bad >/dev/null 2>&1 || \
    "$0" --work-dir /tmp/./voxcpm-self-test >/dev/null 2>&1; then
    echo 'self-test accepted malformed or duplicate options/path' >&2; failed=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$INSPECTOR" --self-test || failed=1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREPARER" --self-test || failed=1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --self-test || failed=1
  (( failed == 0 )) || return 1
  echo 'run-voxcpm-0-5b-inspection.sh self-test: OK'
}

expected_head=''; approval=''; approval_sha=''; work='/dev/shm/vokra-voxcpm-0-5b'
seen_head=0; seen_approval=0; seen_approval_sha=0; seen_work=0
while (( $# )); do
  case "$1" in
    --self-test) [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 ]] || die '--expected-head requires a value'; expected_head="$2"; seen_head=1; shift 2;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && "$2" == /* ]] || die '--approval-evidence requires an absolute path'; approval="$2"; seen_approval=1; shift 2;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha="$2"; seen_approval_sha=1; shift 2;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && "$2" == /* ]] || die '--work-dir requires an absolute path'; work="$2"; seen_work=1; shift 2;;
    -h|--help) echo 'usage: run-voxcpm-0-5b-inspection.sh --expected-head HEX40 --approval-evidence FILE --approval-sha256 HEX64 [--work-dir ABSENT_DIR]'; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
[[ $seen_head == 1 && $seen_approval == 1 && $seen_approval_sha == 1 ]] || die 'expected-head and external approval are required'
[[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
[[ -f "$approval" && ! -L "$approval" ]] || die 'approval evidence is missing or symlinked'
canonical_existing_path "$approval" >/dev/null || die 'approval evidence has unsafe ancestry'
canonical_absent_path "$work" >/dev/null || die 'work path must be absent with non-symlink ancestry'
work_real="$(canonical_absent_path "$work")" || die 'work path cannot be canonicalized'
approval_real="$(canonical_existing_path "$approval")" || die 'approval path cannot be canonicalized'
root_real="$(canonical_existing_path "$ROOT")" || die 'checkout path cannot be canonicalized'
[[ "$work_real" != "$approval_real" && "$work_real" != "$approval_real"/* && "$approval_real" != "$work_real"/* ]] || die 'work path overlaps approval evidence'
[[ "$work_real" != "$root_real" && "$work_real" != "$root_real"/* && "$root_real" != "$work_real"/* ]] || die 'work path overlaps checkout'
[[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 mismatch'
require_clean_expected_head "$expected_head"
validate_approval "$approval" "$expected_head" "$approval_sha"
# This external record is deliberately a blocker, not an execution permit.
blocked_composite_gate
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ -n "${HF_TOKEN:-}" ]] || die 'HF_TOKEN is required for the gated snapshot'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
for command in cargo git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $((128*1024*1024)) ]] || die '128 GiB memory guard failed'
[[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $((40*1024*1024)) ]] || die '40 GiB tmpfs guard failed'
[[ ! -e "$work" ]] || [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection directory is not empty'
mkdir -p "$work/model" "$work/source" "$work/public" "$work/evidence"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="${VOXCPM_UV_CACHE_DIR:-/tmp/vokra-voxcpm-uv-cache}"
{ cargo fmt --all -- --check; cargo metadata --locked --no-deps --format-version 1; } >"$work/evidence/tooling.log" 2>&1 || die 'repository tooling checks failed'

uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/tree.json" "$work/model" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from tools.parity.voxcpm_0_5b_inspect import HF_REPOSITORY, HF_REVISION

tree_path, local_dir = map(Path, sys.argv[1:])
api = HfApi()
info = api.model_info(HF_REPOSITORY, revision=HF_REVISION)
if info.sha != HF_REVISION:
    raise RuntimeError(f"resolved revision mismatch: {info.sha}")
rows = []
for item in api.list_repo_tree(HF_REPOSITORY, revision=HF_REVISION, recursive=True, expand=True):
    if getattr(item, "type", None) != "file":
        continue
    path = getattr(item, "path", None)
    size = getattr(item, "size", None)
    blob = getattr(item, "blob_id", None)
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        lfs_sha, lfs_size = lfs.get("sha256") or lfs.get("oid"), lfs.get("size")
    else:
        lfs_sha = (getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)) if lfs else None
        lfs_size = getattr(lfs, "size", None) if lfs else None
    if not isinstance(path, str) or not isinstance(size, int) or isinstance(size, bool) or not isinstance(blob, str):
        raise RuntimeError("malformed HF tree member")
    rows.append({"path": path, "type": "file", "size": size, "git_blob_sha1": blob, "lfs_sha256": lfs_sha, "lfs_size": lfs_size})
required = {".gitattributes", "README.md", "config.json", "pytorch_model.bin", "audiovae.pth", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}
if not required.issubset({row["path"] for row in rows}):
    raise RuntimeError("required VoxCPM files absent from server tree")
tree_path.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": info.sha, "files": rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
snapshot_download(repo_id=HF_REPOSITORY, revision=HF_REVISION, local_dir=str(local_dir), allow_patterns=[row["path"] for row in rows])
PY

git clone --filter=blob:none "https://github.com/OpenBMB/VoxCPM.git" "$work/source/repo" >"$work/evidence/source-clone.log" 2>&1 || die 'source clone failed'
git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/source-clone.log" 2>&1 || die 'source checkout failed'
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/public" <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id="vokra/voxcpm-0.5b", revision="ee0ca6d5b9fab27bbb626b5cb3f01236e582d004", local_dir=sys.argv[1], allow_patterns=["model.gguf"])
PY
set +e
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --server-tree "$work/tree.json" --source "$work/source/repo" --public-gguf "$work/public/model.gguf" --output "$work/evidence"
rc=$?
set -e
[[ "$rc" == 2 ]] || die "inspector returned $rc instead of deliberate exit 2"
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/manifest.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"status":"BLOCKED", "evidence_stage":"INSPECTION_ONLY", "inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE", "runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status":"UNSUPPORTED", "metal_status":"BLOCKED_BY_CPU", "parity_status":"NOT_RUN", "publication":"NO_UPLOAD"}
if any(p.get(k) != v for k, v in expected.items()):
    raise SystemExit(f"unexpected inspection manifest: {p}")
PY
[[ -n "${VOXCPM_REFERENCE_PACKET:-}" && -f "$VOXCPM_REFERENCE_PACKET" ]] || die 'VOXCPM_REFERENCE_PACKET must name caller-owned token/PCM/draw packet'
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$PREPARER" --main "$work/model/pytorch_model.bin" --audiovae "$work/model/audiovae.pth" --output "$work/evidence/preparation.json" || die 'composite preparation evidence failed'
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/preparation.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "preparation_status": "PREPARATION_EVIDENCE_COMPLETE",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "publication": "NO_UPLOAD",
}
if any(manifest.get(key) != value for key, value in required.items()):
    raise SystemExit(f"unexpected preparation manifest: {manifest}")
for component in ("main", "audiovae"):
    row = manifest.get(component)
    if not isinstance(row, dict) or not row.get("tensor_count") or not row.get("manifest_sha256"):
        raise SystemExit(f"incomplete {component} manifest")
if not manifest.get("composite", {}).get("rows_use_original_and_staged_names"):
    raise SystemExit("composite manifest lost original/staged tensor identity")
PY
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --source "$work/source/repo" --snapshot "$work/model" --packet "$VOXCPM_REFERENCE_PACKET" --output "$work/evidence/reference" || die 'official reference evidence failed'
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/reference/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "reference_status": "REFERENCE_EVIDENCE_COMPLETE",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "parity_status": "MEASURED_NOT_GATED",
    "publication": "NO_UPLOAD",
}
if any(manifest.get(key) != value for key, value in required.items()):
    raise SystemExit(f"unexpected reference manifest: {manifest}")
if manifest.get("draw_calls") != 1 or not manifest.get("tokenizer_calls"):
    raise SystemExit("reference packet was not consumed exactly")
taps = manifest.get("taps", [])
names = [tap.get("name") for tap in taps if isinstance(tap, dict)]
if len(names) != len(set(names)):
    raise SystemExit("reference tap names are not unique")
if "final_pcm" not in names or not any(name.startswith("generated_features_") for name in names):
    raise SystemExit("reference manifest lacks generated feature/final PCM taps")
if names.count("decoded_pcm_0000") != 1:
    raise SystemExit("reference manifest lacks the direct AudioVAE decode tap")
packet = json.loads(Path(sys.argv[1]).with_name("packet.json").read_text(encoding="utf-8"))
if packet.get("prompt_pcm") and not any(name.startswith("prompt_latent") for name in names):
    raise SystemExit("reference manifest lacks the direct prompt AudioVAE encode tap")
draw = manifest.get("random_draw", {})
if draw.get("shape") != [1, 64, 2] or not draw.get("dtype") or draw.get("device") != "cpu":
    raise SystemExit("reference manifest lacks exact random draw type/device evidence")
PY
echo 'VoxCPM evidence/preparation/reference complete; native conversion/CPU/Metal parity remain blocked and no upload occurred.' >&2
exit 2
