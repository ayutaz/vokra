#!/usr/bin/env bash
set -euo pipefail

# Two-phase, no-upload GigaAM v3 worker.  Measure authenticates the fixed
# remote source and records prepared/reference digests.  Parity is unavailable
# until the Sol-reviewed prepared SHA is written into the Rust constant.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
die() { echo "gigaam-v3-vast: BLOCKED: $*" >&2; exit 2; }
require_tool() { command -v "$1" >/dev/null 2>&1 || die "required tool missing: $1"; }
reject_symlink_ancestry() {
  local candidate="$1"
  [[ "$candidate" == /* ]] || die "path must be absolute: $candidate"
  while [[ "$candidate" != "/" ]]; do
    [[ ! -L "$candidate" ]] || die "symlink ancestry forbidden: $candidate"
    candidate="$(dirname "$candidate")"
  done
}
disjoint() {
  local left="$1" right="$2"
  case "$left" in "$right"|"$right"/*) die "path overlap: $left / $right";; esac
  case "$right" in "$left"|"$left"/*) die "path overlap: $left / $right";; esac
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no arguments"
  for path in \
    tools/parity/sber_gigaam_v3_dump_reference.py \
    tools/parity/sber_gigaam_v3_prepare_checkpoint.py \
    tools/parity/gigaam_v3_validation.py \
    crates/vokra-convert/src/models/sber_gigaam_v3.rs \
    crates/vokra-models/src/gigaam/v3.rs; do
    [[ -f "$ROOT/$path" ]] || die "missing contract: $path"
  done
  rg -n -- 'AUTHENTICATED_PREPARED_SHA256|--exact --ignored --nocapture --test-threads=1|OPEN_UNSUPPORTED|GigaAM-v3' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-validation.sh" >/dev/null || die "phase contract missing"
  rg -n -- 'cargo run --locked -p vokra-cli -- convert --model sber-gigaam-v3|export GIGAAM_V3_GGUF|export GIGAAM_V3_REFERENCE_DIR|CONVERTER_APPROVED_SHA|RUNTIME_APPROVED_SHA' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-validation.sh" >/dev/null || die "converter/env approval chain missing"
  rg -n -- 'decision_argmax\.u32le|decision_frames.*decision_symbols|joint_output.*log_softmax' "$ROOT/tools/parity/sber_gigaam_v3_dump_reference.py" "$ROOT/tools/parity/gigaam_v3_validation.py" >/dev/null || die "decision trace contract missing"
  if rg -n -- 'git push|upload\.sh|publish-one\.sh' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-validation.sh" | grep -v 'if rg -n' >/dev/null; then die "upload command found"; fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_dump_reference.py" --self-test >/dev/null || die "reference dumper self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" --self-test >/dev/null || die "validator self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_prepare_checkpoint.py" --self-test >/dev/null || die "preparer self-test failed"
  echo "run-gigaam-v3-validation.sh self-test: OK (NO_UPLOAD; parity OPEN)"
  exit 0
fi

[[ $# == 2 && "${1:-}" == --phase ]] || die "usage: $0 --phase {measure|parity}"
PHASE="$2"
[[ "$PHASE" == measure || "$PHASE" == parity ]] || die "usage: $0 --phase {measure|parity}"
for tool in git realpath sha256sum stat free df awk uv; do require_tool "$tool"; done
[[ "$(free -b | awk '/^Mem:/ {print $2}')" =~ ^[0-9]+$ ]] || die "cannot read RAM"
[[ "$(free -b | awk '/^Mem:/ {print $2}')" -ge 17179869184 ]] || die "at least 16 GiB RAM is required"
[[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" =~ ^[0-9]+$ ]] || die "cannot read free disk"
[[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -ge 20971520 ]] || die "at least 20 GiB free disk is required"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "VAST requires Linux x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
# The broad parity environment is not an approved production dependency
# closure. Keep both phases fail-closed until this dedicated project and lock
# are reviewed and committed.
[[ -f "$ROOT/tools/parity/gigaam_v3/pyproject.toml" && -f "$ROOT/tools/parity/gigaam_v3/uv.lock" ]] || die "dedicated GigaAM v3 pyproject.toml+uv.lock are not reviewed"
V3_PROJECT="$ROOT/tools/parity/gigaam_v3"

[[ -n "${GIGAAM_EVIDENCE_DIR:-}" ]] || die "set evidence path"
reject_symlink_ancestry "$GIGAAM_EVIDENCE_DIR"
[[ ! -e "$GIGAAM_EVIDENCE_DIR" && ! -L "$GIGAAM_EVIDENCE_DIR" ]] || die "evidence directory must be absent"
EVIDENCE_REAL="$(realpath -m "$GIGAAM_EVIDENCE_DIR")"
ROOT_REAL="$(realpath "$ROOT")"
case "$EVIDENCE_REAL" in "$ROOT_REAL"|"$ROOT_REAL"/*) die "evidence must be outside checkout";; esac

if [[ "$PHASE" == measure ]]; then
  [[ -n "${GIGAAM_MODEL_DIR:-}" && -n "${GIGAAM_REFERENCE_DIR:-}" && -n "${GIGAAM_WORK_DIR:-}" ]] || die "set model, reference, and absent work paths"
  reject_symlink_ancestry "$GIGAAM_MODEL_DIR"
  reject_symlink_ancestry "$GIGAAM_REFERENCE_DIR"
  reject_symlink_ancestry "$GIGAAM_WORK_DIR"
  [[ -d "$GIGAAM_MODEL_DIR" && ! -L "$GIGAAM_MODEL_DIR" ]] || die "model snapshot is missing"
  [[ ! -e "$GIGAAM_REFERENCE_DIR" && ! -L "$GIGAAM_REFERENCE_DIR" ]] || die "reference output must be absent"
  [[ ! -e "$GIGAAM_WORK_DIR" && ! -L "$GIGAAM_WORK_DIR" ]] || die "work directory must be absent"
  WORK_REAL="$(realpath -m "$GIGAAM_WORK_DIR")"
  disjoint "$EVIDENCE_REAL" "$WORK_REAL"
  disjoint "$EVIDENCE_REAL" "$(realpath -m "$GIGAAM_REFERENCE_DIR")"
  case "$WORK_REAL" in "$ROOT_REAL"|"$ROOT_REAL"/*) die "work must be outside checkout";; esac
  mkdir "$GIGAAM_WORK_DIR"
  GIGAAM_PREPARED_SAFETENSORS="$GIGAAM_WORK_DIR/prepared.safetensors"
  uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_prepare_checkpoint.py" --input "$GIGAAM_MODEL_DIR/pytorch_model.bin" --output "$GIGAAM_PREPARED_SAFETENSORS"
else
  [[ -n "${GIGAAM_PREPARED_SAFETENSORS:-}" ]] || die "set prepared path"
  reject_symlink_ancestry "$GIGAAM_PREPARED_SAFETENSORS"
  [[ -f "$GIGAAM_PREPARED_SAFETENSORS" && ! -L "$GIGAAM_PREPARED_SAFETENSORS" ]] || die "prepared safetensors must be regular"
fi
PREPARED_REAL="$(realpath "$GIGAAM_PREPARED_SAFETENSORS")"
case "$PREPARED_REAL" in "$ROOT_REAL"|"$ROOT_REAL"/*) die "prepared file must be outside checkout";; esac
disjoint "$EVIDENCE_REAL" "$PREPARED_REAL"
SIDECAR="${GIGAAM_PREPARED_SAFETENSORS%.*}.manifest.json"
reject_symlink_ancestry "$SIDECAR"
[[ -f "$SIDECAR" && ! -L "$SIDECAR" ]] || die "prepared sidecar must be regular"
disjoint "$EVIDENCE_REAL" "$(realpath "$SIDECAR")"
PREPARED_SHA256="$(sha256sum "$GIGAAM_PREPARED_SAFETENSORS" | awk '{print $1}')"
[[ "$PREPARED_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "prepared digest is not lowercase SHA-256"
SIDECAR_SHA256="$(sha256sum "$SIDECAR" | awk '{print $1}')"
[[ "$SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "sidecar digest is not lowercase SHA-256"
SIDECAR_PREPARED_SHA256="$(uv run --frozen --project "$V3_PROJECT" --python 3.12 python - "$SIDECAR" <<'PY'
import json
import sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = doc.get("prepared_sha256")
if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
    raise SystemExit("sidecar prepared_sha256 is not lowercase SHA-256")
print(value)
PY
)"
[[ "$SIDECAR_PREPARED_SHA256" == "$PREPARED_SHA256" ]] || die "sidecar prepared SHA does not match input bytes"

if [[ "$PHASE" == measure ]]; then
  [[ -n "${GIGAAM_MODEL_DIR:-}" && -n "${GIGAAM_REFERENCE_DIR:-}" ]] || die "set model snapshot and reference paths"
  reject_symlink_ancestry "$GIGAAM_MODEL_DIR"
  reject_symlink_ancestry "$GIGAAM_REFERENCE_DIR"
  [[ -d "$GIGAAM_MODEL_DIR" && ! -L "$GIGAAM_MODEL_DIR" ]] || die "model snapshot is missing"
  REFERENCE_DIR="$GIGAAM_REFERENCE_DIR"
  [[ ! -e "$REFERENCE_DIR" && ! -L "$REFERENCE_DIR" ]] || die "reference output must be absent"
  REFERENCE_REAL="$(realpath -m "$REFERENCE_DIR")"
  disjoint "$EVIDENCE_REAL" "$REFERENCE_REAL"
  disjoint "$PREPARED_REAL" "$REFERENCE_REAL"
  uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_dump_reference.py" --model-dir "$GIGAAM_MODEL_DIR" --output "$REFERENCE_DIR"
  uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" "$REFERENCE_DIR"
  REFERENCE_MANIFEST_SHA256="$(sha256sum "$REFERENCE_DIR/manifest.json" | awk '{print $1}')"
  [[ "$REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest digest is invalid"
  python_commit="$(git -C "$ROOT" rev-parse HEAD)"
  [[ "$python_commit" =~ ^[0-9a-f]{40}$ ]] || die "invalid commit identity"
  mkdir "$GIGAAM_EVIDENCE_DIR"
  cat > "$GIGAAM_EVIDENCE_DIR/prepared-digest.json" <<EOF
{"format":"vokra-gigaam-v3-prepared-digest-v1","prepared_path":"$PREPARED_REAL","prepared_sha256":"$PREPARED_SHA256","prepared_bytes":$(stat -c '%s' "$GIGAAM_PREPARED_SAFETENSORS"),"sidecar_sha256":"$SIDECAR_SHA256","reference_manifest_sha256":"$REFERENCE_MANIFEST_SHA256","git_commit":"$python_commit","status":"MEASURED_OPEN_NO_UPLOAD"}
EOF
  echo "GigaAM v3 measure: prepared_sha256=$PREPARED_SHA256 reference=$REFERENCE_DIR status=MEASURED_OPEN_NO_UPLOAD"
  exit 0
fi

[[ -n "${GIGAAM_REFERENCE_DIR:-}" && -n "${GIGAAM_GGUF:-}" ]] || die "set reference and absent GGUF paths"
reject_symlink_ancestry "$GIGAAM_REFERENCE_DIR"
reject_symlink_ancestry "$GIGAAM_GGUF"
[[ -d "$GIGAAM_REFERENCE_DIR" && ! -L "$GIGAAM_REFERENCE_DIR" ]] || die "reference directory is missing"
[[ ! -e "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "GGUF output must be absent"
REFERENCE_REAL="$(realpath "$GIGAAM_REFERENCE_DIR")"
GGUF_REAL="$(realpath -m "$GIGAAM_GGUF")"
disjoint "$EVIDENCE_REAL" "$REFERENCE_REAL"
disjoint "$EVIDENCE_REAL" "$GGUF_REAL"
disjoint "$PREPARED_REAL" "$REFERENCE_REAL"
disjoint "$PREPARED_REAL" "$GGUF_REAL"
CONVERTER_APPROVED_SHA="$(sed -n 's/.*AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some("\([0-9a-f]\{64\}\)").*/\1/p' "$ROOT/crates/vokra-convert/src/models/sber_gigaam_v3.rs")"
RUNTIME_APPROVED_SHA="$(sed -n 's/.*AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some("\([0-9a-f]\{64\}\)").*/\1/p' "$ROOT/crates/vokra-models/src/gigaam/v3.rs")"
[[ "$CONVERTER_APPROVED_SHA" =~ ^[0-9a-f]{64}$ && "$RUNTIME_APPROVED_SHA" =~ ^[0-9a-f]{64}$ ]] || die "approved prepared SHA is not stamped in converter/runtime"
[[ "$CONVERTER_APPROVED_SHA" == "$RUNTIME_APPROVED_SHA" && "$CONVERTER_APPROVED_SHA" == "$PREPARED_SHA256" ]] || die "approved SHA does not equal prepared bytes"
uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" "$GIGAAM_REFERENCE_DIR"
[[ -d "$(dirname "$GIGAAM_GGUF")" ]] || die "GGUF parent must already exist"
export GIGAAM_V3_GGUF="$GIGAAM_GGUF"
export GIGAAM_V3_REFERENCE_DIR="$GIGAAM_REFERENCE_DIR"
GIGAAM_V3_REFERENCE_MANIFEST_SHA256="$(sha256sum "$GIGAAM_REFERENCE_DIR/manifest.json" | awk '{print $1}')"
export GIGAAM_V3_REFERENCE_MANIFEST_SHA256
[[ "$GIGAAM_V3_REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest digest is invalid"
cargo run --locked -p vokra-cli -- convert --model sber-gigaam-v3 --input "$GIGAAM_PREPARED_SAFETENSORS" --output "$GIGAAM_GGUF" --license mit
[[ -f "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "converter did not create GGUF"
mkdir "$GIGAAM_EVIDENCE_DIR"
cargo test --locked -p vokra-models --test parity_gigaam_v3_real real_gigaam_v3_cpu_trace_matches_official -- --exact --ignored --nocapture --test-threads=1 > "$GIGAAM_EVIDENCE_DIR/parity.log" 2>&1
[[ "$(grep -Ec '^test [^ ]*real_gigaam_v3_cpu_trace_matches_official \.\.\. ok$' "$GIGAAM_EVIDENCE_DIR/parity.log")" == 1 ]] || die "parity log must contain one named test pass"
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$' "$GIGAAM_EVIDENCE_DIR/parity.log")" == 1 ]] || die "parity log result is not exact"
GGUF_SHA256="$(sha256sum "$GIGAAM_GGUF" | awk '{print $1}')"
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "GGUF digest is invalid"
cat > "$GIGAAM_EVIDENCE_DIR/validation-summary.json" <<EOF
{"format":"vokra-gigaam-v3-validation-v1","phase":"parity","status":"CPU_PARITY_PASS","publication":"NO_UPLOAD","prepared_sha256":"$PREPARED_SHA256","sidecar_sha256":"$SIDECAR_SHA256","gguf_sha256":"$GGUF_SHA256","reference_manifest_sha256":"$GIGAAM_V3_REFERENCE_MANIFEST_SHA256","metal_apple_status":"OPEN_UNSUPPORTED"}
EOF
echo "GigaAM v3 parity: CPU PASS; Metal/Apple OPEN_UNSUPPORTED; publication NO_UPLOAD"
