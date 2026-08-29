#!/usr/bin/env bash
set -euo pipefail

# Apple/Scaleway consumes the VAST packet through a remote mount.  It never
# pulls a large GGUF locally; only this small JSON summary may be returned.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
die() { echo "gigaam-v3-apple: BLOCKED: $*" >&2; exit 2; }
reject_symlink_ancestry() {
  local candidate="$1"
  while [[ "$candidate" != "/" ]]; do
    [[ ! -L "$candidate" ]] || die "symlink ancestry forbidden: $candidate"
    candidate="$(dirname "$candidate")"
  done
}
canonical_path() {
  local candidate="$1" parent base
  if [[ -d "$candidate" ]]; then
    (cd "$candidate" && pwd -P)
  else
    parent="$(dirname "$candidate")"
    base="$(basename "$candidate")"
    (cd "$parent" && printf '%s/%s\n' "$(pwd -P)" "$base")
  fi
}
hash_file() {
  if [[ "$(uname -s)" == Darwin ]]; then
    command -v shasum >/dev/null 2>&1 || die "shasum is required on Darwin"
    shasum -a 256 "$1" | awk '{print $1}'
  else
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required off Darwin"
    sha256sum "$1" | awk '{print $1}'
  fi
}
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no arguments"
  rg -n -- 'REMOTE_BUNDLE_NO_LOCAL_PULL|OPEN_UNSUPPORTED|parity_gigaam_v3_real|validation-summary' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple contract missing"
  rg -n -- 'uname -s.*Darwin|shasum -a 256|sha256sum' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "portable hash branch missing"
  rg -n -- 'grep -Ec.*real_gigaam_v3_cpu_trace_matches_official|test result: ok.*1 passed' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple parity log gate missing"
  rg -n -- '== 1 \]\]|test result: ok\\\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple exact parity result gate missing"
  rg -n -- 'REMOTE_PACKET_REAL=|TARGET_REAL=|REMOTE_PACKET_REAL/\*' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "remote packet containment gate missing"
  rg -n -- 'tools/parity/gigaam_v3/uv.lock|V3_PROJECT' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "dedicated dependency gate missing"
  echo "run-gigaam-v3-apple-validation.sh self-test: OK"
  exit 0
fi
[[ $# == 2 && "${1:-}" == --backend ]] || die "usage: $0 --backend {cpu|metal}"
BACKEND="$2"
[[ "$BACKEND" == cpu || "$BACKEND" == metal ]] || die "backend must be cpu or metal"
require_tool() { command -v "$1" >/dev/null 2>&1 || die "required tool missing: $1"; }
require_tool uv
require_tool cargo
require_tool rg
require_tool awk
require_tool grep
if [[ "$(uname -s)" == Darwin ]]; then
  require_tool shasum
else
  require_tool sha256sum
fi
[[ -f "$ROOT/tools/parity/gigaam_v3/pyproject.toml" && -f "$ROOT/tools/parity/gigaam_v3/uv.lock" ]] || die "dedicated GigaAM v3 pyproject.toml+uv.lock are not reviewed"
V3_PROJECT="$ROOT/tools/parity/gigaam_v3"
[[ "${REMOTE_BUNDLE_NO_LOCAL_PULL:-0}" == 1 ]] || die "REMOTE_BUNDLE_NO_LOCAL_PULL=1 is required"
[[ -n "${GIGAAM_REMOTE_PACKET:-}" && -n "${GIGAAM_V3_GGUF:-}" && -n "${GIGAAM_V3_REFERENCE_DIR:-}" && -n "${GIGAAM_V3_APPROVAL_JSON:-}" ]] || die "set remote packet, GGUF, reference, and approval paths"
[[ -d "$GIGAAM_REMOTE_PACKET" && ! -L "$GIGAAM_REMOTE_PACKET" ]] || die "remote packet directory is missing"
reject_symlink_ancestry "$GIGAAM_REMOTE_PACKET"
[[ -f "$GIGAAM_V3_GGUF" && ! -L "$GIGAAM_V3_GGUF" ]] || die "remote GGUF is missing"
[[ -d "$GIGAAM_V3_REFERENCE_DIR" && ! -L "$GIGAAM_V3_REFERENCE_DIR" ]] || die "remote reference directory is missing"
[[ -f "$GIGAAM_V3_REFERENCE_DIR/manifest.json" && ! -L "$GIGAAM_V3_REFERENCE_DIR/manifest.json" ]] || die "reference manifest is missing"
[[ -f "$GIGAAM_V3_APPROVAL_JSON" && ! -L "$GIGAAM_V3_APPROVAL_JSON" ]] || die "approval JSON is missing"
reject_symlink_ancestry "$GIGAAM_V3_GGUF"
reject_symlink_ancestry "$GIGAAM_V3_REFERENCE_DIR"
reject_symlink_ancestry "$GIGAAM_V3_APPROVAL_JSON"
REMOTE_PACKET_REAL="$(canonical_path "$GIGAAM_REMOTE_PACKET")"
for target in "$GIGAAM_V3_GGUF" "$GIGAAM_V3_REFERENCE_DIR" "$GIGAAM_V3_APPROVAL_JSON"; do
  TARGET_REAL="$(canonical_path "$target")"
  case "$TARGET_REAL" in
    "$REMOTE_PACKET_REAL"/*) ;;
    *) die "remote artifact is not a canonical descendant of GIGAAM_REMOTE_PACKET: $target" ;;
  esac
done
GGUF_SHA256="$(hash_file "$GIGAAM_V3_GGUF")"
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "GGUF SHA-256 is invalid"
REFERENCE_MANIFEST_SHA256="$(hash_file "$GIGAAM_V3_REFERENCE_DIR/manifest.json")"
[[ "$REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest SHA-256 is invalid"
uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" --portable "$GIGAAM_V3_REFERENCE_DIR"
uv run --frozen --project "$V3_PROJECT" --python 3.12 python - "$GIGAAM_V3_APPROVAL_JSON" "$GGUF_SHA256" "$REFERENCE_MANIFEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path
def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit("duplicate approval key")
        result[key] = value
    return result
doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
if set(doc) != {"format", "phase", "status", "publication", "prepared_sha256", "sidecar_sha256", "gguf_sha256", "reference_manifest_sha256", "metal_apple_status"}:
    raise SystemExit("approval schema mismatch")
if doc["format"] != "vokra-gigaam-v3-validation-v1" or doc["phase"] != "parity" or doc["status"] != "CPU_PARITY_PASS" or doc["publication"] != "NO_UPLOAD" or doc["metal_apple_status"] != "OPEN_UNSUPPORTED":
    raise SystemExit("approval status mismatch")
if doc["gguf_sha256"] != sys.argv[2] or doc["reference_manifest_sha256"] != sys.argv[3]:
    raise SystemExit("approval digest mismatch")
for key in ("prepared_sha256", "sidecar_sha256", "gguf_sha256", "reference_manifest_sha256"):
    if not isinstance(doc[key], str) or len(doc[key]) != 64 or any(char not in "0123456789abcdef" for char in doc[key]):
        raise SystemExit(f"invalid digest: {key}")
PY
if [[ "$BACKEND" == metal ]]; then
  echo '{"format":"vokra-gigaam-v3-apple-validation-v1","status":"OPEN_UNSUPPORTED","backend":"metal","remote_bundle":"REMOTE_BUNDLE_NO_LOCAL_PULL","publication":"NO_UPLOAD"}'
  exit 0
fi
export GIGAAM_V3_REFERENCE_MANIFEST_SHA256="$REFERENCE_MANIFEST_SHA256"
PARITY_LOG="$GIGAAM_REMOTE_PACKET/parity-apple.log"
[[ ! -e "$PARITY_LOG" && ! -L "$PARITY_LOG" ]] || die "Apple parity log must be absent"
cargo test --locked -p vokra-models --test parity_gigaam_v3_real real_gigaam_v3_cpu_trace_matches_official -- --exact --ignored --nocapture --test-threads=1 > "$PARITY_LOG" 2>&1
[[ "$(grep -Ec '^test [^ ]*real_gigaam_v3_cpu_trace_matches_official \.\.\. ok$' "$PARITY_LOG")" == 1 ]] || die "Apple parity log named test mismatch"
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$' "$PARITY_LOG")" == 1 ]] || die "Apple parity log result mismatch"
echo '{"format":"vokra-gigaam-v3-apple-validation-v1","status":"CPU_PARITY_PASS","backend":"cpu","metal":"OPEN_UNSUPPORTED","remote_bundle":"REMOTE_BUNDLE_NO_LOCAL_PULL","publication":"NO_UPLOAD"}'
