#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/voxcpm_0_5b_inspect.py"
PREPARER="$ROOT/tools/parity/voxcpm_0_5b_prepare_checkpoint.py"
REFERENCE="$ROOT/tools/parity/voxcpm_0_5b_reference.py"
MODEL_REPOSITORY="openbmb/VoxCPM-0.5B"
MODEL_REVISION="e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_REVISION="38a76704ee67935ccbafbe5b6725e83dbb1e9305"
die() { echo "voxcpm-vast: ERROR: $*" >&2; exit 2; }

self_test() {
  local failed=0 token
  for token in "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REVISION" \
    'ee0ca6d5b9fab27bbb626b5cb3f01236e582d004' \
    'weights_only=True' 'INSPECTION_ONLY' 'AUTHENTICATED_EVIDENCE_COMPLETE' \
    'NO_UPLOAD' 'audiovae.pth' 'pytorch_model.bin' 'local_dir' \
    'voxcpm_0_5b_prepare_checkpoint.py' 'voxcpm_0_5b_reference.py' \
    'VOXCPM_REFERENCE_PACKET' 'target_text' 'torch.randn' \
    'PREPARATION_EVIDENCE_COMPLETE' 'REFERENCE_EVIDENCE_COMPLETE'; do
    grep -Fq -- "$token" "$INSPECTOR" "$PREPARER" "$REFERENCE" "$0" || { echo "missing contract: $token" >&2; failed=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then
    echo 'upload/publish command found' >&2; failed=1
  fi
  local uv_cache="${VOXCPM_UV_CACHE_DIR:-/tmp/vokra-voxcpm-uv-cache}"
  UV_CACHE_DIR="$uv_cache" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test || failed=1
  UV_CACHE_DIR="$uv_cache" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$PREPARER" --self-test || failed=1
  UV_CACHE_DIR="$uv_cache" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test || failed=1
  (( failed == 0 )) || return 1
  echo 'run-voxcpm-0-5b-inspection.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 0 ]] || die 'usage: run-voxcpm-0-5b-inspection.sh [--self-test]'
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
work="/dev/shm/vokra-voxcpm-0-5b"
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
