#!/usr/bin/env bash
set -euo pipefail

# Authenticated Apple worker for the fixed GigaAM Multilingual packet. The
# packet and reference stay on the remote worker; only a structured result is
# emitted and publication remains disabled.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "apple-gigaam-multilingual: BLOCKED: $*" >&2; exit 2; }
reject_symlink_ancestry() {
  local candidate="$1"
  while [[ "$candidate" != "/" ]]; do
    [[ ! -L "$candidate" ]] || die "symlink ancestry forbidden: $candidate"
    candidate="$(dirname "$candidate")"
  done
}
hash_file() {
  command -v shasum >/dev/null 2>&1 || die "shasum is required"
  shasum -a 256 "$1" | awk '{print $1}'
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

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no other arguments"
  rg -n -- 'VOKRA_REMOTE_APPLE_SILICON=1|VOKRA_EXPECTED_COMMIT|git rev-parse HEAD|status --porcelain|GIGAAM_BACKEND|METAL_PARITY_PASS|REMOTE_BUNDLE_NO_LOCAL_PULL|parity_gigaam_multilingual_real|validation-summary.json|GIGAAM_MULTILINGUAL_APPLE_EVIDENCE_DIR|GIGAAM_MULTILINGUAL_APPROVAL_JSON' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" >/dev/null || die "authenticated Apple contract missing"
  rg -n -- 'uname -s.*Darwin|uname -m.*arm64|xcrun -f metal' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" >/dev/null || die "Apple platform/tool gate missing"
  rg -n -- '--features metal' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" >/dev/null || die "Metal feature build gate missing"
  rg -n -- 'EVIDENCE_REAL=|REMOTE_PACKET_REAL/\*|PENDING_APPLE|fingerprint.txt|input_metal_apple_status' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" >/dev/null || die "evidence/approval contract missing"
  rg -n -- 'backend=\$EXPECTED_BACKEND_SENTINEL|backend=\{backend:\?\}' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" "$ROOT/crates/vokra-models/tests/parity_gigaam_multilingual_real.rs" >/dev/null || die "backend sentinel gate missing"
  rg -n -- 'uv run --no-project --python 3.12 python' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" >/dev/null || die "stdlib-only approval parser environment missing"
  rg -n -- 'cargo test .*--exact --ignored --nocapture --test-threads=1' "$ROOT/scripts/verify/apple-silicon-gigaam-multilingual.sh" >/dev/null || die "serial exact parity command missing"
  echo "apple-silicon-gigaam-multilingual contract self-test: OK (authenticated CPU/Metal; no upload)"
  exit 0
fi

[[ $# == 2 && "${1:-}" == --backend ]] || die "usage: $0 --backend {cpu|metal}"
BACKEND="$2"
[[ "$BACKEND" == cpu || "$BACKEND" == metal ]] || die "backend must be cpu or metal"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die "VOKRA_REMOTE_APPLE_SILICON=1 is required"
[[ "$(uname -s)" == Darwin ]] || die "requires macOS"
[[ "$(uname -m)" == arm64 ]] || die "requires Apple Silicon"
for tool in cargo rg grep git rustc uv sw_vers sysctl; do command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"; done
if [[ "$BACKEND" == metal ]]; then
  command -v xcrun >/dev/null 2>&1 || die "required tool missing: xcrun"
  [[ -n "$(xcrun -f metal 2>/dev/null)" ]] || die "Metal SDK is unavailable"
fi

[[ -n "${VOKRA_EXPECTED_COMMIT:-}" && "$VOKRA_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "VOKRA_EXPECTED_COMMIT must be a lowercase 40-hex commit"
ACTUAL_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$VOKRA_EXPECTED_COMMIT" ]] || die "checkout commit does not match VOKRA_EXPECTED_COMMIT"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"

[[ "${REMOTE_BUNDLE_NO_LOCAL_PULL:-0}" == 1 ]] || die "REMOTE_BUNDLE_NO_LOCAL_PULL=1 is required"
[[ -n "${GIGAAM_REMOTE_PACKET:-}" && -n "${GIGAAM_MULTILINGUAL_GGUF:-}" && -n "${GIGAAM_MULTILINGUAL_REFERENCE_DIR:-}" && -n "${GIGAAM_MULTILINGUAL_APPROVAL_JSON:-}" && -n "${GIGAAM_MULTILINGUAL_APPLE_EVIDENCE_DIR:-}" ]] || die "set remote packet, GGUF, reference, approval, and external evidence paths"
for path in "$GIGAAM_REMOTE_PACKET" "$GIGAAM_MULTILINGUAL_GGUF" "$GIGAAM_MULTILINGUAL_REFERENCE_DIR" "$GIGAAM_MULTILINGUAL_APPROVAL_JSON" "$GIGAAM_MULTILINGUAL_APPLE_EVIDENCE_DIR"; do
  [[ "$path" == /* ]] || die "all packet/artifact/evidence paths must be absolute: $path"
done
[[ -n "${GIGAAM_MULTILINGUAL_GGUF_SHA256:-}" ]] || die "approved GGUF SHA-256 is required"
[[ "$GIGAAM_MULTILINGUAL_GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "GGUF digest must be lowercase SHA-256"
[[ -d "$GIGAAM_REMOTE_PACKET" && ! -L "$GIGAAM_REMOTE_PACKET" ]] || die "remote packet directory is missing"
[[ -f "$GIGAAM_MULTILINGUAL_GGUF" && ! -L "$GIGAAM_MULTILINGUAL_GGUF" ]] || die "remote GGUF is missing"
[[ -d "$GIGAAM_MULTILINGUAL_REFERENCE_DIR" && ! -L "$GIGAAM_MULTILINGUAL_REFERENCE_DIR" ]] || die "reference directory is missing"
[[ -f "$GIGAAM_MULTILINGUAL_APPROVAL_JSON" && ! -L "$GIGAAM_MULTILINGUAL_APPROVAL_JSON" ]] || die "approval JSON is missing"
REMOTE_PACKET_REAL="$(canonical_path "$GIGAAM_REMOTE_PACKET")"
for path in "$GIGAAM_MULTILINGUAL_GGUF" "$GIGAAM_MULTILINGUAL_REFERENCE_DIR" "$GIGAAM_MULTILINGUAL_APPROVAL_JSON"; do
  reject_symlink_ancestry "$path"
  TARGET_REAL="$(canonical_path "$path")"
  case "$TARGET_REAL" in
    "$REMOTE_PACKET_REAL"/*) ;;
    *) die "artifact is outside remote packet: $path" ;;
  esac
done
EVIDENCE_DIR="$GIGAAM_MULTILINGUAL_APPLE_EVIDENCE_DIR"
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
[[ -n "${GIGAAM_MULTILINGUAL_REFERENCE_MANIFEST_SHA256:-}" ]] || die "reference manifest approval digest is required"
[[ "$GIGAAM_MULTILINGUAL_REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest digest must be lowercase SHA-256"
[[ -f "$GIGAAM_MULTILINGUAL_REFERENCE_DIR/manifest.json" && ! -L "$GIGAAM_MULTILINGUAL_REFERENCE_DIR/manifest.json" ]] || die "reference manifest is missing"
GGUF_SHA256="$(hash_file "$GIGAAM_MULTILINGUAL_GGUF")"
[[ "$GGUF_SHA256" == "$GIGAAM_MULTILINGUAL_GGUF_SHA256" ]] || die "GGUF digest mismatch"
MANIFEST_SHA256="$(hash_file "$GIGAAM_MULTILINGUAL_REFERENCE_DIR/manifest.json")"
[[ "$MANIFEST_SHA256" == "$GIGAAM_MULTILINGUAL_REFERENCE_MANIFEST_SHA256" ]] || die "reference manifest digest mismatch"

uv run --no-project --python 3.12 python - "$GIGAAM_MULTILINGUAL_APPROVAL_JSON" "$GGUF_SHA256" "$MANIFEST_SHA256" "$ACTUAL_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate approval key: {key}")
        result[key] = value
    return result

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
expected = {"phase", "status", "cpu_parity_status", "metal_apple_status", "git_commit", "repository", "revision", "source_revision", "config_sha256", "checkpoint_sha256", "prepared_sha256", "prepared_bytes", "sidecar_sha256", "sidecar_bytes", "tensor_count", "manifest_sha256", "reference_manifest_sha256", "publication", "gguf_sha256"}
if set(document) != expected:
    raise SystemExit("approval schema mismatch")
if document["phase"] != "parity" or document["status"] != "CPU_PARITY_PASS" or document["cpu_parity_status"] != "PASS" or document["metal_apple_status"] != "PENDING_APPLE" or document["publication"] != "NO_UPLOAD":
    raise SystemExit("approval status mismatch")
if document["gguf_sha256"] != sys.argv[2] or document["reference_manifest_sha256"] != sys.argv[3]:
    raise SystemExit("approval digest mismatch")
if document["git_commit"] != sys.argv[4] or not isinstance(document["git_commit"], str) or len(document["git_commit"]) != 40 or any(char not in "0123456789abcdef" for char in document["git_commit"]):
    raise SystemExit("approval commit is invalid")
PY

[[ ! -e "$EVIDENCE_DIR" && ! -L "$EVIDENCE_DIR" ]] || die "Apple evidence directory changed during preflight"
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

PARITY_LOG="$EVIDENCE_DIR/parity.log"
[[ ! -e "$PARITY_LOG" && ! -L "$PARITY_LOG" ]] || die "parity log must be absent"
export GIGAAM_BACKEND="$BACKEND"
export GIGAAM_MULTILINGUAL_GGUF GIGAAM_MULTILINGUAL_REFERENCE_DIR
CARGO_BUILD_JOBS=1 cargo test --locked --features metal -p vokra-models --test parity_gigaam_multilingual_real real_gigaam_multilingual_trace_matches_official -- --exact --ignored --nocapture --test-threads=1 > "$PARITY_LOG" 2>&1

metric='[+-]?[0-9]+(\.[0-9]+)?e[+-][0-9]+'
[[ "$(grep -Ec '^test [^ ]*real_gigaam_multilingual_trace_matches_official \.\.\. ' "$PARITY_LOG")" == 1 ]] || die "named parity test mismatch"
EXPECTED_BACKEND_SENTINEL="Cpu"
[[ "$BACKEND" == metal ]] && EXPECTED_BACKEND_SENTINEL="Metal"
[[ "$(grep -Ec "^GIGAAM_MULTILINGUAL_PARITY backend=$EXPECTED_BACKEND_SENTINEL PASS$" "$PARITY_LOG")" == 1 ]] || die "Apple backend sentinel mismatch"
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(\.[0-9]+)?s$' "$PARITY_LOG")" == 1 ]] || die "parity result mismatch"
[[ "$(grep -Ec "^GIGAAM_MULTILINGUAL_PARITY encoded max_abs=${metric} index=[0-9]+ mean_abs=${metric}$" "$PARITY_LOG")" == 1 ]] || die "encoded metric sentinel mismatch"
[[ "$(grep -Ec "^GIGAAM_MULTILINGUAL_PARITY logits max_abs=${metric} index=[0-9]+ mean_abs=${metric}$" "$PARITY_LOG")" == 1 ]] || die "logits metric sentinel mismatch"
[[ "$(grep -Ec '^GIGAAM_MULTILINGUAL_PARITY token_ids=exact PASS$' "$PARITY_LOG")" == 1 ]] || die "token sentinel mismatch"

if [[ "$BACKEND" == cpu ]]; then STATUS=CPU_PARITY_PASS; else STATUS=METAL_PARITY_PASS; fi
cat > "$EVIDENCE_DIR/validation-summary.json" <<EOF
{"format":"vokra-gigaam-multilingual-apple-validation-v1","status":"$STATUS","backend":"$BACKEND","expected_commit":"$VOKRA_EXPECTED_COMMIT","actual_commit":"$ACTUAL_COMMIT","gguf_sha256":"$GGUF_SHA256","reference_manifest_sha256":"$MANIFEST_SHA256","parity_log":"parity.log","fingerprint":"fingerprint.txt","input_metal_apple_status":"PENDING_APPLE","remote_bundle":"REMOTE_BUNDLE_NO_LOCAL_PULL","publication":"NO_UPLOAD"}
EOF
[[ -f "$EVIDENCE_DIR/validation-summary.json" && ! -L "$EVIDENCE_DIR/validation-summary.json" ]] || die "validation summary was not written safely"
