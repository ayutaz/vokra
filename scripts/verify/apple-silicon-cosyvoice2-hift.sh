#!/usr/bin/env bash
# Offline real-weight CosyVoice2 HiFT CPU/Metal verifier for Apple Silicon.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/tools/parity/cosyvoice2_hift_reference/preflight_gate.py"
TEST_NAME=cosyvoice2_hift_apple_cpu_metal_parity
die(){ echo "cosyvoice2-hift-apple: ERROR: $*" >&2; exit 2; }
usage(){ echo "usage: $0 --gguf FILE --gguf-sha256 SHA --reference-dir DIR --reference-manifest-sha256 SHA --license-manifest FILE --evidence-dir ABSENT_DIR" >&2; }
reject_path(){ local p="$1" label="$2" c="$1"; [[ "$p" == /* ]] || die "$label must be absolute"; case "$p" in */./*|*/../*|./*|../*|*/.|*/..) die "$label has dot path components";; esac; while :; do [[ ! -L "$c" ]] || die "$label has symlink ancestry: $c"; [[ "$c" == / ]] && break; c="$(dirname "$c")"; done; }
require_file(){ reject_path "$1" "$2"; [[ -f "$1" && ! -L "$1" ]] || die "$2 must be a regular non-symlink file"; }
require_dir(){ reject_path "$1" "$2"; [[ -d "$1" && ! -L "$1" ]] || die "$2 must be a regular non-symlink directory"; }
scope(){ [[ -e "$1" ]] && realpath "$1" || realpath "$(dirname "$1")/$(basename "$1")"; }
disjoint(){ local a="$(scope "$1")" b="$(scope "$2")"; [[ "$a" != "$b" && "$a" != "$b"/* && "$b" != "$a"/* ]] || die "paths overlap: $1 and $2"; }
sha(){ shasum -a 256 "$1" | awk '{print $1}'; }
valid_sha(){ [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "$2 must be lowercase SHA-256"; }
self_test(){
  local fail=0 token
  for token in "$TEST_NAME" "VOKRA_REMOTE_APPLE_SILICON=1" "CARGO_NET_OFFLINE=true" "NO_UPLOAD" "preflight_gate.py" "create_new"; do grep -Fq -- "$token" "$0" "$ROOT/crates/vokra-models/tests/parity_cosyvoice2_hift_apple.rs" || { echo "missing contract: $token" >&2; fail=1; }; done
  if grep -En 'curl|wget|git[[:space:]]+(clone|pull|fetch|push)|upload\.sh|publish-one\.sh|--upload|--push|cargo[[:space:]]+add' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  (( fail == 0 )) || return 1
  echo 'apple-silicon-cosyvoice2-hift.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 12 ]] || { usage; exit 2; }
GGUF= GGUF_SHA= REFERENCE= REFERENCE_SHA= LICENSE= EVIDENCE=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf) GGUF="$2";; --gguf-sha256) GGUF_SHA="$2";; --reference-dir) REFERENCE="$2";;
    --reference-manifest-sha256) REFERENCE_SHA="$2";; --license-manifest) LICENSE="$2";; --evidence-dir) EVIDENCE="$2";;
    *) usage; die "unknown argument: $1";;
  esac
  shift 2
done
[[ "$(uname -s)" == Darwin ]] || die 'non-Apple host refused'
[[ "$(uname -m)" == arm64 ]] || die 'non-arm64 host refused'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" != 1 ]] || die 'VAST worker is not an Apple verifier'
require_file "$GGUF" GGUF; require_dir "$REFERENCE" reference; require_file "$LICENSE" license-manifest; reject_path "$EVIDENCE" evidence
[[ ! -e "$EVIDENCE" && ! -L "$EVIDENCE" ]] || die 'evidence directory must be absent'
valid_sha "$GGUF_SHA" gguf-sha256; valid_sha "$REFERENCE_SHA" reference-manifest-sha256
[[ "$(sha "$GGUF")" == "$GGUF_SHA" ]] || die 'GGUF digest mismatch'
[[ -f "$REFERENCE/manifest.json" && ! -L "$REFERENCE/manifest.json" ]] || die 'reference manifest missing'
[[ "$(sha "$REFERENCE/manifest.json")" == "$REFERENCE_SHA" ]] || die 'reference manifest digest mismatch'
CHECKED_IN="$ROOT/tools/parity/cosyvoice2_hift_reference/license_gate_manifest.json"
[[ "$(realpath "$LICENSE")" != "$(realpath "$CHECKED_IN")" ]] || die 'checked-in pending manifest cannot authorize execution'
CHECKOUT="$ROOT"
for pair in "$GGUF|$REFERENCE" "$GGUF|$LICENSE" "$GGUF|$EVIDENCE" "$REFERENCE|$LICENSE" "$REFERENCE|$EVIDENCE" "$LICENSE|$EVIDENCE" "$CHECKOUT|$GGUF" "$CHECKOUT|$REFERENCE" "$CHECKOUT|$LICENSE" "$CHECKOUT|$EVIDENCE"; do disjoint "${pair%%|*}" "${pair#*|}"; done
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
TMP="$(mktemp -d "${TMPDIR:-/tmp}/vokra-hift-apple.XXXXXX")"; trap 'rm -rf -- "$TMP"' EXIT
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$GATE" --license-manifest "$LICENSE" >"$TMP/license.log" 2>&1 || { cat "$TMP/license.log" >&2; die 'external approved license preflight failed'; }
grep -Fq '"status": "PASS"' "$TMP/license.log" || die 'license preflight did not report PASS'
VOKRA_REMOTE_APPLE_SILICON=1 VOKRA_COSYVOICE2_HIFT_GGUF="$GGUF" VOKRA_COSYVOICE2_HIFT_GGUF_SHA256="$GGUF_SHA" VOKRA_COSYVOICE2_HIFT_REFERENCE_DIR="$REFERENCE" VOKRA_COSYVOICE2_HIFT_REFERENCE_MANIFEST_SHA256="$REFERENCE_SHA" VOKRA_COSYVOICE2_HIFT_LICENSE_MANIFEST="$LICENSE" VOKRA_COSYVOICE2_HIFT_APPLE_EVIDENCE_DIR="$EVIDENCE" CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1 cargo test --offline --locked --release -p vokra-models --test parity_cosyvoice2_hift_apple "$TEST_NAME" -- --ignored --exact --nocapture >"$TMP/cargo.log" 2>&1 || { cat "$TMP/cargo.log" >&2; die 'Apple HiFT parity failed'; }
grep -Eq "^test $TEST_NAME \.\.\. ok$" "$TMP/cargo.log" || die 'named test did not pass exactly once'
grep -Eq 'test result: ok\. 1 passed; 0 failed' "$TMP/cargo.log" || die 'Cargo did not report one passing test'
[[ -f "$EVIDENCE/evidence.json" && ! -L "$EVIDENCE/evidence.json" ]] || die 'evidence marker missing'
echo "COSYVOICE2_HIFT_APPLE_PARITY_PASS evidence=$EVIDENCE/evidence.json gguf_sha256=$GGUF_SHA reference_manifest_sha256=$REFERENCE_SHA"
