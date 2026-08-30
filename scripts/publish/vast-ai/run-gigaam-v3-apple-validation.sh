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
  rg -n -- 'VOKRA_REMOTE_APPLE_SILICON=1|VOKRA_EXPECTED_COMMIT|git rev-parse HEAD|status --porcelain|REMOTE_BUNDLE_NO_LOCAL_PULL|parity_gigaam_v3_real|validation-summary.json|GIGAAM_V3_APPLE_EVIDENCE_DIR' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple contract missing"
  rg -n -- 'uname -s.*Darwin|uname -m.*arm64|xcrun -f metal|shasum -a 256' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple platform/tool gate missing"
  rg -n -- '--features metal' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Metal feature build gate missing"
  rg -n -- 'grep -Ec.*real_gigaam_v3_trace_matches_official|test result: ok.*1 passed' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple parity log gate missing"
  rg -n -- '== 1 \]\]|test result: ok\\\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "Apple exact parity result gate missing"
  rg -n -- 'REMOTE_PACKET_REAL=|TARGET_REAL=|EVIDENCE_REAL=|REMOTE_PACKET_REAL/\*|validation-summary.json' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "remote packet/evidence containment gate missing"
  rg -n -- 'git_commit|metal_apple_status.*PENDING_APPLE|input_metal_apple_status|fingerprint.txt' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "commit/PENDING_APPLE/fingerprint gate missing"
  rg -n -- 'tools/parity/gigaam_v3/uv.lock|V3_PROJECT' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-apple-validation.sh" >/dev/null || die "dedicated dependency gate missing"
  echo "run-gigaam-v3-apple-validation.sh self-test: OK"
  exit 0
fi
[[ $# == 2 && "${1:-}" == --backend ]] || die "usage: $0 --backend {cpu|metal}"
BACKEND="$2"
[[ "$BACKEND" == cpu || "$BACKEND" == metal ]] || die "backend must be cpu or metal"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die "VOKRA_REMOTE_APPLE_SILICON=1 is required"
[[ "$(uname -s)" == Darwin ]] || die "requires macOS"
[[ "$(uname -m)" == arm64 ]] || die "requires Apple Silicon"
require_tool() { command -v "$1" >/dev/null 2>&1 || die "required tool missing: $1"; }
require_tool uv
require_tool cargo
require_tool rg
require_tool awk
require_tool grep
require_tool git
require_tool rustc
require_tool sw_vers
require_tool sysctl
require_tool shasum
if [[ "$BACKEND" == metal ]]; then
  require_tool xcrun
  [[ -n "$(xcrun -f metal 2>/dev/null)" ]] || die "Metal SDK is unavailable"
fi
[[ -n "${VOKRA_EXPECTED_COMMIT:-}" && "$VOKRA_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "VOKRA_EXPECTED_COMMIT must be a lowercase 40-hex commit"
ACTUAL_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$VOKRA_EXPECTED_COMMIT" ]] || die "checkout commit does not match VOKRA_EXPECTED_COMMIT"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
[[ -f "$ROOT/tools/parity/gigaam_v3/pyproject.toml" && -f "$ROOT/tools/parity/gigaam_v3/uv.lock" ]] || die "dedicated GigaAM v3 pyproject.toml+uv.lock are not reviewed"
V3_PROJECT="$ROOT/tools/parity/gigaam_v3"
[[ "${REMOTE_BUNDLE_NO_LOCAL_PULL:-0}" == 1 ]] || die "REMOTE_BUNDLE_NO_LOCAL_PULL=1 is required"
[[ -n "${GIGAAM_REMOTE_PACKET:-}" && -n "${GIGAAM_V3_GGUF:-}" && -n "${GIGAAM_V3_REFERENCE_DIR:-}" && -n "${GIGAAM_V3_APPROVAL_JSON:-}" && -n "${GIGAAM_V3_APPLE_EVIDENCE_DIR:-}" ]] || die "set remote packet, GGUF, reference, approval, and external evidence paths"
for path in "$GIGAAM_REMOTE_PACKET" "$GIGAAM_V3_GGUF" "$GIGAAM_V3_REFERENCE_DIR" "$GIGAAM_V3_APPROVAL_JSON" "$GIGAAM_V3_APPLE_EVIDENCE_DIR"; do
  [[ "$path" == /* ]] || die "all packet/artifact/evidence paths must be absolute: $path"
done
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
uv run --frozen --project "$V3_PROJECT" --python 3.12 python - "$GIGAAM_V3_APPROVAL_JSON" "$GGUF_SHA256" "$REFERENCE_MANIFEST_SHA256" "$ACTUAL_COMMIT" <<'PY'
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
if set(doc) != {"format", "phase", "status", "publication", "git_commit", "prepared_sha256", "sidecar_sha256", "gguf_sha256", "reference_manifest_sha256", "metal_apple_status"}:
    raise SystemExit("approval schema mismatch")
if doc["format"] != "vokra-gigaam-v3-validation-v1" or doc["phase"] != "parity" or doc["status"] != "CPU_PARITY_PASS" or doc["publication"] != "NO_UPLOAD" or doc["metal_apple_status"] != "PENDING_APPLE":
    raise SystemExit("approval status mismatch")
if doc["gguf_sha256"] != sys.argv[2] or doc["reference_manifest_sha256"] != sys.argv[3]:
    raise SystemExit("approval digest mismatch")
if doc["git_commit"] != sys.argv[4] or not isinstance(doc["git_commit"], str) or len(doc["git_commit"]) != 40 or any(char not in "0123456789abcdef" for char in doc["git_commit"]):
    raise SystemExit("approval commit mismatch")
for key in ("prepared_sha256", "sidecar_sha256", "gguf_sha256", "reference_manifest_sha256"):
    if not isinstance(doc[key], str) or len(doc[key]) != 64 or any(char not in "0123456789abcdef" for char in doc[key]):
        raise SystemExit(f"invalid digest: {key}")
PY
EVIDENCE_DIR="$GIGAAM_V3_APPLE_EVIDENCE_DIR"
[[ ! -e "$EVIDENCE_DIR" && ! -L "$EVIDENCE_DIR" ]] || die "Apple evidence directory must be absent and non-symlink"
[[ -d "$(dirname "$EVIDENCE_DIR")" ]] || die "Apple evidence parent directory is missing"
reject_symlink_ancestry "$EVIDENCE_DIR"
EVIDENCE_REAL="$(canonical_path "$EVIDENCE_DIR")"
case "$EVIDENCE_REAL" in
  "$REMOTE_PACKET_REAL"|"$REMOTE_PACKET_REAL"/*) die "Apple evidence must be outside GIGAAM_REMOTE_PACKET" ;;
esac
case "$REMOTE_PACKET_REAL" in
  "$EVIDENCE_REAL"|"$EVIDENCE_REAL"/*) die "GIGAAM_REMOTE_PACKET must be outside Apple evidence" ;;
esac
mkdir "$EVIDENCE_DIR"
{
  echo "git_commit=$ACTUAL_COMMIT"
  echo "expected_commit=$VOKRA_EXPECTED_COMMIT"
  rustc --version
  cargo --version
  sw_vers
  echo "uname_s=$(uname -s)"
  echo "uname_m=$(uname -m)"
  sw_vers
  echo "hw.model=$(sysctl -n hw.model)"
  echo "hw.machine=$(sysctl -n hw.machine)"
  echo "hw.memsize=$(sysctl -n hw.memsize)"
  echo "hw.ncpu=$(sysctl -n hw.ncpu)"
  if [[ "$BACKEND" == metal ]]; then
    xcrun --version
    echo "metal_sdk=$(xcrun -f metal)"
  fi
} > "$EVIDENCE_DIR/fingerprint.txt"
[[ -f "$EVIDENCE_DIR/fingerprint.txt" && ! -L "$EVIDENCE_DIR/fingerprint.txt" ]] || die "fingerprint evidence was not written safely"
export GIGAAM_V3_REFERENCE_MANIFEST_SHA256="$REFERENCE_MANIFEST_SHA256"
PARITY_LOG="$EVIDENCE_DIR/parity.log"
[[ ! -e "$PARITY_LOG" && ! -L "$PARITY_LOG" ]] || die "Apple parity log must be absent"
export GIGAAM_BACKEND="$BACKEND"
CARGO_BUILD_JOBS=1 cargo test --locked --features metal -p vokra-models --test parity_gigaam_v3_real real_gigaam_v3_trace_matches_official -- --exact --ignored --nocapture --test-threads=1 > "$PARITY_LOG" 2>&1
[[ "$(grep -Ec '^test [^ ]*real_gigaam_v3_trace_matches_official \.\.\. ok$' "$PARITY_LOG")" == 1 ]] || die "Apple parity log named test mismatch"
EXPECTED_BACKEND_SENTINEL="Cpu"
[[ "$BACKEND" == metal ]] && EXPECTED_BACKEND_SENTINEL="Metal"
[[ "$(grep -Ec "^GIGAAM_V3_PARITY backend=$EXPECTED_BACKEND_SENTINEL PASS; publication NO_UPLOAD$" "$PARITY_LOG")" == 1 ]] || die "Apple backend sentinel mismatch"
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$' "$PARITY_LOG")" == 1 ]] || die "Apple parity log result mismatch"
if [[ "$BACKEND" == cpu ]]; then
  STATUS=CPU_PARITY_PASS
else
  STATUS=METAL_PARITY_PASS
fi
cat > "$EVIDENCE_DIR/validation-summary.json" <<EOF
{"format":"vokra-gigaam-v3-apple-validation-v1","status":"$STATUS","backend":"$BACKEND","expected_commit":"$VOKRA_EXPECTED_COMMIT","actual_commit":"$ACTUAL_COMMIT","git_commit":"$ACTUAL_COMMIT","gguf_sha256":"$GGUF_SHA256","reference_manifest_sha256":"$REFERENCE_MANIFEST_SHA256","parity_log":"parity.log","fingerprint":"fingerprint.txt","input_metal_apple_status":"PENDING_APPLE","remote_bundle":"REMOTE_BUNDLE_NO_LOCAL_PULL","publication":"NO_UPLOAD"}
EOF
[[ -f "$EVIDENCE_DIR/validation-summary.json" && ! -L "$EVIDENCE_DIR/validation-summary.json" ]] || die "validation summary was not written safely"
echo "$(<"$EVIDENCE_DIR/validation-summary.json")"
