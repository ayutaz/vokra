#!/usr/bin/env bash
set -euo pipefail

# VAST-only validation contract for the authenticated GigaAM Multilingual
# checkpoint. This worker never uploads a model and never changes publication
# status: dataset provenance is still unauthenticated, so NO_UPLOAD remains.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
die() { echo "gigaam-multilingual-vast: BLOCKED: $*" >&2; exit 2; }

require_tool() { command -v "$1" >/dev/null 2>&1 || die "required tool is missing: $1"; }
reject_symlink_ancestry() {
  local candidate="$1"
  [[ "$candidate" == /* ]] || die "path must be absolute: $candidate"
  while [[ "$candidate" != "/" ]]; do
    [[ ! -L "$candidate" ]] || die "symlink ancestry is forbidden: $candidate"
    candidate="$(dirname "$candidate")"
  done
}
reject_path_overlap() {
  local left="$1"
  local right="$2"
  case "$left" in
    "$right"|"$right"/*) die "path overlap: $left and $right";;
  esac
  case "$right" in
    "$left"|"$left"/*) die "path overlap: $left and $right";;
  esac
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no other arguments"
  for required in \
    tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py \
    tools/parity/sber_gigaam_multilingual_dump_reference.py \
    crates/vokra-convert/src/models/sber_gigaam_multilingual.rs \
    crates/vokra-models/src/gigaam/multilingual.rs \
    crates/vokra-models/tests/parity_gigaam_multilingual_real.rs \
    tools/parity/gigaam_multilingual_validation.py; do
    [[ -f "$ROOT/$required" ]] || die "missing required contract: $required"
  done
  if rg -n -- "git push|publish-one.sh|upload.sh|--push|--upload" "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" | grep -v 'rg -n' >/dev/null; then
    die "upload command found"
  fi
  rg -n -- "safe_open|read_safetensors_header|actual_names|phase (measure|parity)|snapshot_download|parity_gigaam_multilingual_real" "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "phase/header/parity validation missing"
  [[ "$(rg -c -- '^  cat > \"\$EVIDENCE_DIR/source\.json\" <<EOF$' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh")" == 1 ]] || die "source evidence heredoc must occur exactly once"
  rg -n -- 'real_gigaam_multilingual_cpu_trace_matches_official' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "parity test filter is missing"
  rg -n -- --exact "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "parity command must require exact test matching"
  rg -n -- --ignored "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "parity command must run the ignored real-weight test"
  rg -n -- --nocapture "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "parity command must preserve the test log"
  rg -n -- --test-threads=1 "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "parity command must be serial"
  obsolete_ctc='remove_blank_then_'
  obsolete_ctc+='adjacent_repeat'
  if rg -n -- "$obsolete_ctc" "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" "$ROOT/tools/parity/gigaam_multilingual_validation.py" >/dev/null; then
    die "obsolete CTC contract remains"
  fi
  rg -n -- 'collapse_adjacent_repeat_then_remove_blank' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" "$ROOT/tools/parity/gigaam_multilingual_validation.py" >/dev/null || die "CTC collapse contract is missing"
  rg -n -- 'REFERENCE_MANIFEST_SHA256=""' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "reference manifest digest is not initialized"
  rg -n -- 'GIGAAM_PCM_NPY="\$GIGAAM_STAGE_DIR/fixed_pcm\.npy"' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "fixed PCM path is not code-bound"
  rg -n -- '17179869184|20971520' "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" >/dev/null || die "resource gate is too weak"
  rg -n -- "encoded\.f32le|logits\.f32le|raw_argmax\.u32le|token_ids\.u32le" "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" >/dev/null || die "raw reference artifacts missing"
  rg -n -- 'transpose\(0, 1\)|logaddexp\.reduce' "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" >/dev/null || die "official encoder/head axis contract missing"
  if rg -n -- '\.npz|np\.savez' "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" >/dev/null; then die "reference must not be npz-only"; fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/gigaam_multilingual_validation.py" --self-test 2>/dev/null || die "reference validator self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" --self-test 2>/dev/null || die "reference dumper self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$ROOT/scripts/publish/vast-ai/run-gigaam-multilingual-validation.sh" <<'PY'
import ast
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
if len(blocks) < 5:
    raise SystemExit(f"expected embedded Python blocks, found {len(blocks)}")
if not re.search(
    r"--test parity_gigaam_multilingual_real[\\\s]+"
    r"real_gigaam_multilingual_cpu_trace_matches_official[\\\s]+"
    r"-- --exact --ignored --nocapture --test-threads=1",
    source,
):
    raise SystemExit("parity command filter/flags contract is missing")
for index, block in enumerate(blocks):
    compile(block, f"embedded-python-{index}", "exec")
    tree = ast.parse(block, filename=f"embedded-python-{index}")
    json_imports = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "json" for alias in node.names)
    )
    pathlib_imports = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "pathlib"
        and any(alias.name == "Path" for alias in node.names)
    )
    if json_imports > 1 or pathlib_imports > 1:
        raise SystemExit(f"duplicate embedded imports in block {index}")
print(f"embedded Python compile: OK ({len(blocks)} blocks)")
PY
  echo "run-gigaam-multilingual-validation.sh self-test: OK (NO_UPLOAD)"
  exit 0
fi

[[ $# == 2 && "${1:-}" == --phase ]] || die "usage: $0 --phase {measure|parity}"
PHASE="$2"
[[ "$PHASE" == measure || "$PHASE" == parity ]] || die "usage: $0 --phase {measure|parity}"

for tool in git realpath sha256sum stat awk uv free df; do require_tool "$tool"; done
[[ "$PHASE" != parity ]] || require_tool cargo
[[ "$(free -b | awk '/^Mem:/ {print $2}')" =~ ^[0-9]+$ ]] || die "cannot read host memory"
[[ "$(free -b | awk '/^Mem:/ {print $2}')" -ge 17179869184 ]] || die "at least 16 GiB RAM is required"
[[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" =~ ^[0-9]+$ ]] || die "cannot read host disk"
[[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -ge 20971520 ]] || die "at least 20 GiB free disk is required"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "VAST requires Linux x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
if [[ "$PHASE" == measure ]]; then
  [[ -n "${GIGAAM_STAGE_DIR:-}" ]] || die "set an absent GIGAAM_STAGE_DIR outside checkout"
  [[ -n "${GIGAAM_EVIDENCE_DIR:-}" ]] || die "set an absent GIGAAM_EVIDENCE_DIR outside checkout"
  reject_symlink_ancestry "$GIGAAM_STAGE_DIR"
  [[ ! -e "$GIGAAM_STAGE_DIR" && ! -L "$GIGAAM_STAGE_DIR" ]] || die "GIGAAM_STAGE_DIR must be absent"
  STAGE_PARENT="$(dirname "$GIGAAM_STAGE_DIR")"
  [[ -d "$STAGE_PARENT" ]] || die "GIGAAM_STAGE_DIR parent must already exist"
  STAGE_REAL="$(realpath -m "$GIGAAM_STAGE_DIR")"
  ROOT_REAL="$(realpath "$ROOT")"
  case "$STAGE_REAL" in "$ROOT_REAL"/*|"$ROOT_REAL") die "GIGAAM_STAGE_DIR must be outside checkout";; esac
  EVIDENCE_CANDIDATE_REAL="$(realpath -m "${GIGAAM_EVIDENCE_DIR:-}")"
  case "$EVIDENCE_CANDIDATE_REAL" in "$STAGE_REAL"/*|"$STAGE_REAL") die "evidence path overlaps GIGAAM_STAGE_DIR";; esac
  case "$STAGE_REAL" in "$EVIDENCE_CANDIDATE_REAL"/*|"$EVIDENCE_CANDIDATE_REAL") die "GIGAAM_STAGE_DIR overlaps evidence path";; esac
  mkdir "$GIGAAM_STAGE_DIR"
  MODEL_DIR="$GIGAAM_STAGE_DIR/model"
  GIGAAM_PCM_NPY="$GIGAAM_STAGE_DIR/fixed_pcm.npy"
  GIGAAM_PREPARED_SAFETENSORS="$GIGAAM_STAGE_DIR/prepared.safetensors"
  REFERENCE_DIR="$GIGAAM_STAGE_DIR/reference"
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$GIGAAM_PCM_NPY" <<'PY'
import sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
samples = np.arange(16000, dtype=np.int32)
pcm = ((samples % 97) - 48).astype(np.float32) / np.float32(48.0)
if pcm.shape != (16000,) or not np.isfinite(pcm).all() or not np.any(pcm != 0):
    raise SystemExit("fixed PCM generation contract failed")
np.save(path, pcm, allow_pickle=False)
PY
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
from pathlib import Path

destination = Path(sys.argv[1])
if destination.exists() or destination.is_symlink():
    raise SystemExit("model snapshot destination must be absent")
snapshot_download(
    repo_id="ai-sage/GigaAM-Multilingual",
    revision="2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
    local_dir=str(destination),
    allow_patterns=["config.json", "modeling_gigaam.py", "pytorch_model.bin"],
)
PY
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py" --input "$MODEL_DIR/pytorch_model.bin" --output "$GIGAAM_PREPARED_SAFETENSORS"
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_multilingual_dump_reference.py" --model-dir "$MODEL_DIR" --pcm-npy "$GIGAAM_PCM_NPY" --output "$REFERENCE_DIR"
fi
[[ -n "${GIGAAM_PREPARED_SAFETENSORS:-}" ]] || die "set GIGAAM_PREPARED_SAFETENSORS to the VAST-prepared file"
[[ -f "$GIGAAM_PREPARED_SAFETENSORS" ]] || die "prepared safetensors is missing"
[[ ! -L "$GIGAAM_PREPARED_SAFETENSORS" ]] || die "prepared safetensors must not be a symlink"
SIDECAR="${GIGAAM_PREPARED_SAFETENSORS%.*}.manifest.json"
[[ -f "$SIDECAR" ]] || die "prepared manifest sidecar is missing"
[[ ! -L "$SIDECAR" ]] || die "prepared manifest must not be a symlink"
if [[ "$PHASE" == parity ]]; then
  [[ -n "${GIGAAM_REFERENCE_DIR:-}" ]] || die "set GIGAAM_REFERENCE_DIR from the measure phase"
  [[ -n "${GIGAAM_GGUF:-}" ]] || die "set an absent GIGAAM_GGUF output path"
  REFERENCE_DIR="$GIGAAM_REFERENCE_DIR"
  [[ -d "$REFERENCE_DIR" && ! -L "$REFERENCE_DIR" ]] || die "reference directory is missing or symlinked"
  [[ -f "$REFERENCE_DIR/manifest.json" && ! -L "$REFERENCE_DIR/manifest.json" ]] || die "reference manifest is missing or symlinked"
  reject_symlink_ancestry "$REFERENCE_DIR"
  reject_symlink_ancestry "$GIGAAM_GGUF"
  [[ ! -e "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "GIGAAM_GGUF must be absent"
  REFERENCE_REAL="$(realpath "$REFERENCE_DIR")"
  ROOT_REAL="$(realpath "$ROOT")"
  case "$REFERENCE_REAL" in "$ROOT_REAL"/*|"$ROOT_REAL") die "GIGAAM_REFERENCE_DIR must be outside checkout";; esac
fi
REFERENCE_MANIFEST_SHA256=""
[[ -n "${GIGAAM_EVIDENCE_DIR:-}" ]] || die "set an absent GIGAAM_EVIDENCE_DIR"
EVIDENCE_DIR="$GIGAAM_EVIDENCE_DIR"
[[ ! -e "$EVIDENCE_DIR" && ! -L "$EVIDENCE_DIR" ]] || die "evidence directory must be absent and non-symlink"
EVIDENCE_PARENT="$(dirname "$EVIDENCE_DIR")"
[[ -d "$EVIDENCE_PARENT" ]] || die "evidence parent must already exist"
reject_symlink_ancestry "$GIGAAM_PREPARED_SAFETENSORS"
reject_symlink_ancestry "$SIDECAR"
reject_symlink_ancestry "$EVIDENCE_DIR"
reject_symlink_ancestry "$EVIDENCE_PARENT"

PREPARED_REAL="$(realpath "$GIGAAM_PREPARED_SAFETENSORS")"
SIDECAR_REAL="$(realpath "$SIDECAR")"
EVIDENCE_PARENT_REAL="$(realpath -m "$EVIDENCE_PARENT")"
EVIDENCE_REAL="$(realpath -m "$EVIDENCE_DIR")"
ROOT_REAL="$(realpath "$ROOT")"
[[ "$EVIDENCE_PARENT_REAL" != "$PREPARED_REAL" && "$EVIDENCE_PARENT_REAL" != "$SIDECAR_REAL" ]] || die "evidence path overlaps an input"
case "$PREPARED_REAL" in "$EVIDENCE_REAL"/*|"$EVIDENCE_REAL") die "evidence path overlaps prepared input";; esac
case "$SIDECAR_REAL" in "$EVIDENCE_REAL"/*|"$EVIDENCE_REAL") die "evidence path overlaps sidecar input";; esac
if [[ "$PHASE" == parity ]]; then
  case "$REFERENCE_REAL" in "$PREPARED_REAL"/*|"$PREPARED_REAL") die "reference path overlaps prepared input";; esac
  case "$PREPARED_REAL" in "$REFERENCE_REAL"/*|"$REFERENCE_REAL") die "prepared input overlaps reference path";; esac
  case "$REFERENCE_REAL" in "$SIDECAR_REAL"/*|"$SIDECAR_REAL") die "reference path overlaps sidecar input";; esac
  case "$SIDECAR_REAL" in "$REFERENCE_REAL"/*|"$REFERENCE_REAL") die "sidecar input overlaps reference path";; esac
  case "$REFERENCE_REAL" in "$EVIDENCE_REAL"/*|"$EVIDENCE_REAL") die "evidence path overlaps reference input";; esac
  case "$EVIDENCE_REAL" in "$REFERENCE_REAL"/*|"$REFERENCE_REAL") die "evidence path is inside reference input";; esac
fi
case "$EVIDENCE_REAL" in "$ROOT_REAL"/*|"$ROOT_REAL") die "evidence path must be outside checkout";; esac

export GIGAAM_PREPARED_SAFETENSORS SIDECAR EVIDENCE_DIR GIGAAM_REPO_ROOT="$ROOT"
PREPARED_SHA256="$(sha256sum "$GIGAAM_PREPARED_SAFETENSORS" | awk '{print $1}')"
SIDECAR_SHA256="$(sha256sum "$SIDECAR" | awk '{print $1}')"
PREPARED_BYTES="$(stat -c '%s' "$GIGAAM_PREPARED_SAFETENSORS")"
SIDECAR_BYTES="$(stat -c '%s' "$SIDECAR")"
[[ "$PREPARED_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "prepared SHA-256 is not lowercase hex"
[[ "$SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "sidecar SHA-256 is not lowercase hex"
[[ "$PREPARED_BYTES" =~ ^[0-9]+$ && "$SIDECAR_BYTES" =~ ^[0-9]+$ ]] || die "artifact size is not numeric"
if [[ "$PHASE" == parity ]]; then
  CODE_SHA256="$(uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$ROOT/crates/vokra-models/src/gigaam/multilingual.rs" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(
    r'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some\("([0-9a-f]{64})"\);',
    source,
)
if len(matches) != 1:
    raise SystemExit("AUTHENTICATED_PREPARED_SHA256 must be exactly one Some(lowercase SHA-256)")
print(matches[0])
PY
  )"
  [[ "$CODE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "prepared SHA constant is not lowercase hex"
  CONVERTER_CODE_SHA256="$(uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$ROOT/crates/vokra-convert/src/models/sber_gigaam_multilingual.rs" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(
    r'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some\("([0-9a-f]{64})"\);',
    source,
)
if len(matches) != 1:
    raise SystemExit("converter AUTHENTICATED_PREPARED_SHA256 must be exactly one Some(lowercase SHA-256)")
print(matches[0])
PY
  )"
  [[ "$CONVERTER_CODE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "converter prepared SHA constant is not lowercase hex"
  [[ "$PREPARED_SHA256" == "$CODE_SHA256" && "$CODE_SHA256" == "$CONVERTER_CODE_SHA256" ]] || die "prepared bytes and reviewed code SHA constants disagree"
  [[ "$SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "sidecar digest is not lowercase hex"
fi
GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "checkout commit is not lowercase hex"

# Validate the exact sidecar schema and authenticated identity with the
# repository's Python 3.12 environment. This records evidence only; it does
# not import a model or perform inference.
MANIFEST_SHA256="$(UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$GIGAAM_PREPARED_SAFETENSORS" "$SIDECAR" "$PREPARED_SHA256" "$REFERENCE_DIR" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("os").environ["GIGAAM_REPO_ROOT"]) / "tools/parity"))
from gigaam_multilingual_validation import validate_reference_bundle  # noqa: E402
from sber_gigaam_multilingual_prepare_checkpoint import (  # noqa: E402
    CHECKPOINT_SHA256,
    CONFIG_SHA256,
    HF_REPOSITORY,
    HF_REVISION,
    PREPARED_FORMAT,
    SOURCE_REVISION,
    expected_manifest,
)

prepared = Path(sys.argv[1])
sidecar = Path(sys.argv[2])
actual_sha = sys.argv[3]
reference_dir = Path(sys.argv[4])

def reject(message):
    raise SystemExit(f"gigaam-multilingual-vast: BLOCKED: {message}")

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            reject(f"duplicate JSON key: {key}")
        result[key] = value
    return result

try:
    validate_reference_bundle(reference_dir)
except (OSError, ValueError) as exc:
    reject(f"reference artifact validation failed: {exc}")

def read_safetensors_header(path):
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            reject("prepared safetensors header length is truncated")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length == 0 or header_length > 16 * 1024 * 1024:
            reject("prepared safetensors header length is outside bounds")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            reject("prepared safetensors header is truncated")
    try:
        return json.loads(raw_header.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        reject(f"invalid prepared safetensors header: {exc}")

try:
    document = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
except (OSError, json.JSONDecodeError) as exc:
    reject(f"invalid sidecar JSON: {exc}")
if set(document) != {
    "format", "repository", "revision", "source_revision", "config_sha256",
    "checkpoint_sha256", "prepared_sha256", "tensor_count", "tensors",
}:
    reject("sidecar root schema mismatch")
expected = {
    "format": PREPARED_FORMAT,
    "repository": HF_REPOSITORY,
    "revision": HF_REVISION,
    "source_revision": SOURCE_REVISION,
    "config_sha256": CONFIG_SHA256,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "prepared_sha256": actual_sha,
    "tensor_count": 552,
}
for key, value in expected.items():
    if document.get(key) != value:
        reject(f"sidecar identity mismatch for {key}")
rows = document["tensors"]
manifest = expected_manifest()
if not isinstance(rows, list) or len(rows) != len(manifest):
    reject("sidecar tensor count mismatch")
for row, (name, shape) in zip(rows, manifest):
    if not isinstance(row, dict) or set(row) != {"name", "shape", "dtype"}:
        reject("sidecar tensor row schema mismatch")
    if row != {"name": name, "shape": shape, "dtype": "F32"}:
        reject(f"sidecar tensor mismatch: {name}")
header = read_safetensors_header(prepared)
if not isinstance(header, dict) or "__metadata__" in header:
    reject("prepared safetensors header has unexpected metadata")
expected_names = {name for name, _ in manifest}
if set(header) != expected_names:
    reject("prepared safetensors header tensor name set mismatch")
try:
    from safetensors import safe_open
    with safe_open(str(prepared), framework="pt", device="cpu") as handle:
        actual_names = set(handle.keys())
        actual_shapes = {
            name: list(handle.get_slice(name).get_shape()) for name in actual_names
        }
except Exception as exc:
    reject(f"prepared safetensors header cannot be opened: {exc}")
if actual_names != expected_names:
    reject("prepared safetensors safe_open key set mismatch")
for name, shape in manifest:
    tensor_header = header[name]
    if not isinstance(tensor_header, dict) or set(tensor_header) != {"dtype", "shape", "data_offsets"}:
        reject(f"prepared safetensors header schema mismatch: {name}")
    if tensor_header["dtype"] != "F32" or tensor_header["shape"] != shape:
        reject(f"prepared safetensors header dtype/shape mismatch: {name}")
    if actual_shapes[name] != shape:
        reject(f"prepared safetensors safe_open shape mismatch: {name}")
digest = hashlib.sha256()
with prepared.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
if digest.hexdigest() != actual_sha:
    reject("prepared SHA-256 changed during validation")
print(hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
 )"
[[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "manifest SHA-256 is not lowercase hex"

if [[ "$PHASE" == measure || "$PHASE" == parity ]]; then
  REFERENCE_MANIFEST_SHA256="$(uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$REFERENCE_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
def reject(message):
    raise SystemExit(f"gigaam-multilingual-vast: BLOCKED: {message}")
def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            reject(f"duplicate reference manifest key: {key}")
        result[key] = value
    return result
try:
    document = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
except (OSError, ValueError, json.JSONDecodeError) as exc:
    reject(f"invalid reference manifest: {exc}")
if set(document) != {"format", "status", "repository", "revision", "source_revision", "config_sha256", "modeling_gigaam_sha256", "source_files", "pcm_input", "artifacts", "encoded_length", "ctc", "runtime", "parity"}:
    reject("reference manifest root schema mismatch")
if document["format"] != "vokra-gigaam-multilingual-reference-v1" or document["status"] != "REFERENCE_DUMP_OPEN_NOT_PARITY":
    reject("reference manifest status/format mismatch")
if document["repository"] != "ai-sage/GigaAM-Multilingual" or document["revision"] != "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8" or document["source_revision"] != "7447938d791c4f3e643386ee22c33777004293a5":
    reject("reference identity mismatch")
if document["config_sha256"] != "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653" or document["modeling_gigaam_sha256"] != "6d02e640fbb5738ab11c030520a68654ef32f4ff363723db10534cf8b5d5c0e7":
    reject("reference source digest mismatch")
source_files = document["source_files"]
if not isinstance(source_files, dict) or set(source_files) != {"config", "modeling_gigaam", "checkpoint"}:
    reject("reference source file set mismatch")
for name, row in source_files.items():
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or not isinstance(row["path"], str) or not row["path"].startswith("/") or "\\" in row["path"] or ".." in Path(row["path"]).parts or not isinstance(row["bytes"], int) or row["bytes"] <= 0 or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in row["sha256"]):
        reject(f"reference source file row mismatch: {name}")
if source_files["config"]["sha256"] != document["config_sha256"] or source_files["modeling_gigaam"]["sha256"] != document["modeling_gigaam_sha256"] or source_files["checkpoint"]["sha256"] != "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728":
    reject("reference source file digest mismatch")
if source_files["checkpoint"]["bytes"] != 883170115:
    reject("reference checkpoint size mismatch")
if document["parity"] != "OPEN_MEASURED_NOT_GATED":
    reject("reference parity status must remain open")
if document["ctc"] != {"vocab_size": 71, "blank_id": 70, "collapse": "collapse_adjacent_repeat_then_remove_blank"}:
    reject("reference CTC contract mismatch")
artifacts = document["artifacts"]
if set(artifacts) != {"pcm", "encoded", "logits", "raw_argmax", "token_ids"}:
    reject("reference artifact set mismatch")
expected_artifacts = {
    "pcm": ("pcm.f32le", "float32"),
    "encoded": ("encoded.f32le", "float32"),
    "logits": ("logits.f32le", "float32"),
    "raw_argmax": ("raw_argmax.u32le", "uint32"),
    "token_ids": ("token_ids.u32le", "uint32"),
}
for name, row in artifacts.items():
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256", "shape", "dtype"}:
        reject(f"reference artifact schema mismatch: {name}")
    if Path(row["path"]).name != row["path"] or not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] < 0 or (row["bytes"] == 0 and name != "token_ids") or not isinstance(row["shape"], list) or not isinstance(row["dtype"], str):
        reject(f"reference artifact metadata mismatch: {name}")
    if (row["path"], row["dtype"]) != expected_artifacts[name] or row["bytes"] % 4 != 0:
        reject(f"reference artifact path/dtype mismatch: {name}")
    artifact = root / row["path"]
    if artifact.is_symlink() or not artifact.is_file():
        reject(f"reference artifact is missing or symlinked: {name}")
    digest = hashlib.sha256()
    size = 0
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk); size += len(chunk)
    if digest.hexdigest() != row["sha256"] or size != row["bytes"]:
        reject(f"reference artifact digest/size mismatch: {name}")
if not isinstance(document["encoded_length"], int) or isinstance(document["encoded_length"], bool) or document["encoded_length"] <= 0 or artifacts["encoded"]["shape"] != [document["encoded_length"], 768] or artifacts["logits"]["shape"] != [document["encoded_length"], 71] or artifacts["raw_argmax"]["shape"] != [document["encoded_length"]]:
    reject("reference encoded/logits shape contract mismatch")
if len(artifacts["pcm"]["shape"]) != 1 or not isinstance(artifacts["pcm"]["shape"][0], int) or isinstance(artifacts["pcm"]["shape"][0], bool) or artifacts["pcm"]["shape"][0] <= 0 or len(artifacts["token_ids"]["shape"]) != 1 or artifacts["token_ids"]["shape"] != [artifacts["token_ids"]["bytes"] // 4]:
    reject("reference PCM/token shape contract mismatch")
print(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
PY
  )"
  [[ "$REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest SHA-256 is not lowercase hex"
fi

mkdir "$EVIDENCE_DIR"
if [[ "$PHASE" == parity ]]; then
  DIGEST_STATUS="PREPARED_ARTIFACT_DIGEST_MEASURED_NOT_GATED"
  CPU_PARITY_STATUS="PENDING"
else
  DIGEST_STATUS="PREPARED_ARTIFACT_DIGEST_MEASURED_NOT_GATED"
  CPU_PARITY_STATUS="OPEN_MEASURED_NOT_GATED"
fi
cat > "$EVIDENCE_DIR/digest.json" <<EOF
{
  "phase": "$PHASE",
  "status": "$DIGEST_STATUS",
  "cpu_parity_status": "$CPU_PARITY_STATUS",
  "metal_apple_status": "OPEN_UNSUPPORTED",
  "git_commit": "$GIT_COMMIT",
  "repository": "ai-sage/GigaAM-Multilingual",
  "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
  "source_revision": "7447938d791c4f3e643386ee22c33777004293a5",
  "config_sha256": "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653",
  "checkpoint_sha256": "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728",
  "prepared_sha256": "$PREPARED_SHA256",
  "prepared_bytes": $PREPARED_BYTES,
  "sidecar_sha256": "$SIDECAR_SHA256",
  "sidecar_bytes": $SIDECAR_BYTES,
  "tensor_count": 552,
  "manifest_sha256": "$MANIFEST_SHA256",
  "reference_manifest_sha256": "$REFERENCE_MANIFEST_SHA256",
  "publication": "NO_UPLOAD"
}
EOF

if [[ "$PHASE" == measure ]]; then
  MODEL_CONFIG_SHA256="$(sha256sum "$MODEL_DIR/config.json" | awk '{print $1}')"
  MODELING_SHA256="$(sha256sum "$MODEL_DIR/modeling_gigaam.py" | awk '{print $1}')"
  MODEL_CHECKPOINT_SHA256="$(sha256sum "$MODEL_DIR/pytorch_model.bin" | awk '{print $1}')"
  for value in "$MODEL_CONFIG_SHA256" "$MODELING_SHA256" "$MODEL_CHECKPOINT_SHA256"; do
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "downloaded source digest is not lowercase hex"
  done
  cat > "$EVIDENCE_DIR/source.json" <<EOF
{
  "repository": "ai-sage/GigaAM-Multilingual",
  "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8",
  "config_sha256": "$MODEL_CONFIG_SHA256",
  "config_bytes": $(stat -c '%s' "$MODEL_DIR/config.json"),
  "modeling_gigaam_sha256": "$MODELING_SHA256",
  "modeling_gigaam_bytes": $(stat -c '%s' "$MODEL_DIR/modeling_gigaam.py"),
  "checkpoint_sha256": "$MODEL_CHECKPOINT_SHA256",
  "checkpoint_bytes": $(stat -c '%s' "$MODEL_DIR/pytorch_model.bin"),
  "download": "huggingface_hub.snapshot_download pinned revision",
  "publication": "NO_UPLOAD"
}
EOF
  echo "GigaAM Multilingual measure phase completed; evidence: $EVIDENCE_DIR."
  echo "Reference is OPEN/MEASURED_NOT_GATED; no parity claim was made."
  exit 0
fi

[[ -d "$(dirname "$GIGAAM_GGUF")" ]] || die "GIGAAM_GGUF parent must already exist"
[[ ! -e "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "GIGAAM_GGUF must remain absent before conversion"
ROOT_REAL="$(realpath "$ROOT")"
GGUF_REAL="$(realpath -m "$GIGAAM_GGUF")"
case "$GGUF_REAL" in "$ROOT_REAL"/*|"$ROOT_REAL") die "GIGAAM_GGUF must be outside checkout";; esac
reject_path_overlap "$GGUF_REAL" "$PREPARED_REAL"
reject_path_overlap "$GGUF_REAL" "$SIDECAR_REAL"
reject_path_overlap "$GGUF_REAL" "$REFERENCE_REAL"
reject_path_overlap "$GGUF_REAL" "$EVIDENCE_REAL"
export GIGAAM_MULTILINGUAL_GGUF="$GIGAAM_GGUF"
export GIGAAM_MULTILINGUAL_REFERENCE_DIR="$REFERENCE_DIR"
export GIGAAM_MULTILINGUAL_PARITY_REPORT="$EVIDENCE_DIR/parity.json"
cargo run --locked -p vokra-cli -- convert \
  --model sber-gigaam-multilingual \
  --input "$GIGAAM_PREPARED_SAFETENSORS" \
  --output "$GIGAAM_GGUF" \
  --license mit
[[ -f "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "converter did not create a regular GGUF"
CARGO_BUILD_JOBS=1 cargo test --locked -p vokra-models \
  --test parity_gigaam_multilingual_real \
  real_gigaam_multilingual_cpu_trace_matches_official \
  -- --exact --ignored --nocapture --test-threads=1 \
  > "$EVIDENCE_DIR/parity.log" 2>&1
[[ -f "$EVIDENCE_DIR/parity.json" && ! -L "$EVIDENCE_DIR/parity.json" ]] || die "native parity report is missing"
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$EVIDENCE_DIR/parity.json" <<'PY'
import json
import math
import sys
from pathlib import Path

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate parity report key: {key}")
        result[key] = value
    return result
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
if set(data) != {"format", "status", "encoded_max_abs", "encoded_mean_abs", "logits_max_abs", "logits_mean_abs", "raw_argmax", "token_ids"}:
    raise SystemExit("parity report schema mismatch")
if data["format"] != "vokra-gigaam-multilingual-parity-v1" or data["status"] != "PASS" or data["raw_argmax"] != "EXACT" or data["token_ids"] != "EXACT":
    raise SystemExit("parity report is not an exact PASS")
for key in ("encoded_max_abs", "encoded_mean_abs", "logits_max_abs", "logits_mean_abs"):
    if not isinstance(data[key], (int, float)) or isinstance(data[key], bool) or not math.isfinite(data[key]) or data[key] < 0:
        raise SystemExit(f"invalid parity metric: {key}")
    bound = 1.0e-2 if key.endswith("max_abs") else 1.0e-3
    if data[key] > bound:
        raise SystemExit(f"parity metric exceeds registered bound: {key}")
PY
[[ "$(grep -Ec '^test ' "$EVIDENCE_DIR/parity.log")" == 1 ]] || die "parity.log must contain exactly one test line"
[[ "$(grep -Ec '^test [^ ]*real_gigaam_multilingual_cpu_trace_matches_official \.\.\. ok$' "$EVIDENCE_DIR/parity.log")" == 1 ]] || die "parity.log must record exactly one named GigaAM test pass"
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$' "$EVIDENCE_DIR/parity.log")" == 1 ]] || die "parity.log must report the exact one-test result"
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$EVIDENCE_DIR/digest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
document = json.loads(path.read_text(encoding="utf-8"))
if document.get("status") != "PREPARED_ARTIFACT_DIGEST_MEASURED_NOT_GATED" or document.get("cpu_parity_status") != "PENDING":
    raise SystemExit("digest evidence was not in the expected pre-parity state")
document["status"] = "CPU_PARITY_PASS"
document["cpu_parity_status"] = "PASS"
path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate digest key: {key}")
        result[key] = value
    return result
document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
if document.get("status") != "CPU_PARITY_PASS" or document.get("cpu_parity_status") != "PASS" or document.get("metal_apple_status") != "OPEN_UNSUPPORTED" or document.get("publication") != "NO_UPLOAD":
    raise SystemExit("digest parity status mismatch")
PY
echo "GigaAM Multilingual real CPU parity PASS; evidence: $EVIDENCE_DIR."
echo "CPU parity: PASS; Metal/Apple: OPEN_UNSUPPORTED; publication: NO_UPLOAD."
echo "NO_UPLOAD: dataset provenance is not authenticated."
echo "GigaAM Multilingual prepared artifact digest recorded at $EVIDENCE_DIR/digest.json."
