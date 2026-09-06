#!/usr/bin/env bash
# Offline real-weight CosyVoice2 HiFT CPU/Metal verifier for Apple Silicon.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/tools/parity/cosyvoice2_hift_reference/preflight_gate.py"
TEST_NAME=cosyvoice2_hift_apple_cpu_metal_parity
die(){ echo "cosyvoice2-hift-apple: ERROR: $*" >&2; exit 2; }
usage(){ echo "usage: $0 --gguf FILE --gguf-sha256 SHA --reference-dir DIR --reference-manifest-sha256 SHA --license-manifest FILE --evidence-dir ABSENT_DIR" >&2; }
reject_path(){
  local p label current
  p="$1"
  label="$2"
  current="$1"
  [[ "$p" == /* ]] || die "$label must be absolute"
  case "$p" in */./*|*/../*|./*|../*|*/.|*/..) die "$label has dot path components";; esac
  while :; do
    [[ ! -L "$current" ]] || die "$label has symlink ancestry: $current"
    [[ "$current" == / ]] && break
    current="$(dirname "$current")"
  done
}
require_file(){
  local path label
  path="$1"
  label="$2"
  reject_path "$path" "$label"
  [[ -f "$path" && ! -L "$path" ]] || die "$label must be a regular non-symlink file"
}
require_dir(){
  local path label
  path="$1"
  label="$2"
  reject_path "$path" "$label"
  [[ -d "$path" && ! -L "$path" ]] || die "$label must be a regular non-symlink directory"
}
scope(){
  local path parent
  path="$1"
  if [[ -e "$path" ]]; then
    realpath "$path"
  else
    parent="$(dirname "$path")"
    [[ -d "$parent" && ! -L "$parent" ]] || die "scope parent is missing or symlinked: $parent"
    (
      cd -P "$parent"
      printf '%s/%s\n' "$PWD" "$(basename "$path")"
    ) || die "cannot canonicalize scope parent: $parent"
  fi
}
disjoint(){
  local a b
  a="$(scope "$1")"; b="$(scope "$2")"
  if [[ "$a" == "$b" || "$a" == "$b"/* || "$b" == "$a"/* ]]; then die "paths overlap: $1 and $2"; fi
}
sha(){ shasum -a 256 "$1" | awk '{print $1}'; }
valid_sha(){ [[ "$1" =~ ^[0-9a-f]{64}$ ]] || die "$2 must be lowercase SHA-256"; }
reject_checked_in(){
  local candidate checked_in
  candidate="$1"
  checked_in="$ROOT/tools/parity/cosyvoice2_hift_reference/license_gate_manifest.json"
  [[ "$(scope "$candidate")" != "$(scope "$checked_in")" ]] || die 'checked-in pending manifest cannot authorize execution'
}
validate_preflight_json(){
  local output expected_project expected_lock expected_license_sha
  output="$1"
  expected_project="$2"
  expected_lock="$3"
  expected_license_sha="$4"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$output" "$expected_project" "$expected_lock" "$expected_license_sha" <<'PY'
import json, pathlib, sys
def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result
path, expected_project, expected_lock, expected_license_sha = sys.argv[1:]
document = json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
expected = {"status","project","lock","package_count","platform","license_status","owner_signoff","publication","project_sha256","lock_sha256","license_manifest_sha256","approval_scope_sha256"}
if set(document) != expected or document["status"] != "PASS" or document["platform"] != "linux-x86_64-cpu" or document["license_status"] != "APPROVED" or document["owner_signoff"] != "OWNER_SIGNED_OFF" or document["publication"] != "NO_UPLOAD" or document["package_count"] != 13:
    raise SystemExit("preflight JSON schema/status mismatch")
if document["project"] != expected_project or document["lock"] != expected_lock or document["license_manifest_sha256"] != expected_license_sha:
    raise SystemExit("preflight JSON input binding mismatch")
for key in ("project_sha256","lock_sha256","license_manifest_sha256","approval_scope_sha256"):
    if not isinstance(document[key], str) or len(document[key]) != 64 or any(c not in "0123456789abcdef" for c in document[key]):
        raise SystemExit(f"invalid preflight digest: {key}")
PY
}
validate_evidence_json(){
  local output gguf_sha reference_sha license_sha
  output="$1"
  gguf_sha="$2"
  reference_sha="$3"
  license_sha="$4"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$output" "$gguf_sha" "$reference_sha" "$license_sha" <<'PY'
import json, math, pathlib, sys
def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result
path, gguf, reference, license_sha = sys.argv[1:]
document = json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
expected = {"format","status","publication","backend","device_evidence","gguf_sha256","reference_manifest_sha256","license_manifest_sha256","atol","f0_cpu_reference_max_abs","pcm_cpu_reference_max_abs","pcm_metal_reference_max_abs","pcm_metal_cpu_max_abs","scope"}
if set(document) != expected: raise SystemExit("evidence schema mismatch")
if document["format"] != "vokra-cosyvoice2-hift-apple-evidence-v1" or document["status"] != "PASS" or document["publication"] != "NO_UPLOAD" or document["backend"] != "CPU+Metal" or document["device_evidence"] != "system_profiler:success": raise SystemExit("evidence identity mismatch")
if document["gguf_sha256"] != gguf or document["reference_manifest_sha256"] != reference or document["license_manifest_sha256"] != license_sha or document["atol"] != 0.01: raise SystemExit("evidence digest/tolerance mismatch")
if document["scope"] != {"component":"standalone_cosyvoice2_hift","f0":"CPU/reference only; Metal F0 not separately exposed","pcm":"CPU/reference, Metal/reference, Metal/CPU","full_cosyvoice2_e2e":"NOT_CLAIMED"}: raise SystemExit("evidence scope mismatch")
for key in ("f0_cpu_reference_max_abs","pcm_cpu_reference_max_abs","pcm_metal_reference_max_abs","pcm_metal_cpu_max_abs"):
    value = document[key]
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 0.01: raise SystemExit(f"invalid metric: {key}")
PY
}
self_test(){
  local fail token duplicate_output temp
  fail=0
  temp="$(mktemp -d "${TMPDIR:-/tmp}/vokra-hift-self-test.XXXXXX")"
  mkdir "$temp/nested"
  trap '[[ -n "${temp:-}" ]] && rm -rf -- "$temp"' EXIT
  for token in "$TEST_NAME" "VOKRA_REMOTE_APPLE_SILICON=1" "CARGO_NET_OFFLINE=true" "NO_UPLOAD" "preflight_gate.py" "create_new"; do
    grep -Fq -- "$token" "$0" "$ROOT/crates/vokra-models/tests/parity_cosyvoice2_hift_apple.rs" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En 'curl|wget|git[[:space:]]+(clone|pull|fetch|push)|upload\.sh|publish-one\.sh|--upload|--push|cargo[[:space:]]+add' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  if duplicate_output="$("$0" --gguf /tmp/a --gguf /tmp/b --gguf-sha256 00 --reference-dir /tmp/r --reference-manifest-sha256 00 --license-manifest /tmp/l --evidence-dir /tmp/e 2>&1)"; then fail=1; else grep -Fq 'duplicate option' <<<"$duplicate_output" || fail=1; fi
  if (disjoint "$temp" "$temp/nested") 2>/dev/null; then fail=1; fi
  if (reject_checked_in "$ROOT/tools/parity/cosyvoice2_hift_reference/license_gate_manifest.json") 2>/dev/null; then fail=1; fi
  if (reject_checked_in "//$ROOT/tools/parity/cosyvoice2_hift_reference/license_gate_manifest.json") 2>/dev/null; then fail=1; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$temp/preflight.json" "$temp/evidence.json" <<'PY'
import json, pathlib, sys
preflight = {"status":"PASS","project":"p","lock":"l","package_count":13,"platform":"linux-x86_64-cpu","license_status":"APPROVED","owner_signoff":"OWNER_SIGNED_OFF","publication":"NO_UPLOAD","project_sha256":"0"*64,"lock_sha256":"1"*64,"license_manifest_sha256":"2"*64,"approval_scope_sha256":"3"*64}
scope = {"component":"standalone_cosyvoice2_hift","f0":"CPU/reference only; Metal F0 not separately exposed","pcm":"CPU/reference, Metal/reference, Metal/CPU","full_cosyvoice2_e2e":"NOT_CLAIMED"}
evidence = {"format":"vokra-cosyvoice2-hift-apple-evidence-v1","status":"PASS","publication":"NO_UPLOAD","backend":"CPU+Metal","device_evidence":"system_profiler:success","gguf_sha256":"0"*64,"reference_manifest_sha256":"1"*64,"license_manifest_sha256":"2"*64,"atol":0.01,"f0_cpu_reference_max_abs":0.0,"pcm_cpu_reference_max_abs":0.0,"pcm_metal_reference_max_abs":0.0,"pcm_metal_cpu_max_abs":0.0,"scope":scope}
pathlib.Path(sys.argv[1]).write_text(json.dumps(preflight), encoding="utf-8")
pathlib.Path(sys.argv[2]).write_text(json.dumps(evidence), encoding="utf-8")
PY
  validate_preflight_json "$temp/preflight.json" "p" "l" "2222222222222222222222222222222222222222222222222222222222222222" || fail=1
  validate_evidence_json "$temp/evidence.json" "0000000000000000000000000000000000000000000000000000000000000000" "1111111111111111111111111111111111111111111111111111111111111111" "2222222222222222222222222222222222222222222222222222222222222222" || fail=1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$temp/evidence.json" "$temp/bool.json" <<'PY'
import json, pathlib, sys
document = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
document["f0_cpu_reference_max_abs"] = True
pathlib.Path(sys.argv[2]).write_text(json.dumps(document), encoding="utf-8")
PY
  if validate_evidence_json "$temp/bool.json" "0000000000000000000000000000000000000000000000000000000000000000" "1111111111111111111111111111111111111111111111111111111111111111" "2222222222222222222222222222222222222222222222222222222222222222" >/dev/null 2>&1; then fail=1; fi
  printf '%s\n' '{"status":"PASS","status":"PASS"}' >"$temp/duplicate.json"
  if validate_preflight_json "$temp/duplicate.json" "p" "l" "2222222222222222222222222222222222222222222222222222222222222222" >/dev/null 2>&1; then fail=1; fi
  printf '%s\n' '{"format":"vokra-cosyvoice2-hift-apple-evidence-v1","format":"duplicate"}' >"$temp/evidence-duplicate.json"
  if validate_evidence_json "$temp/evidence-duplicate.json" "0000000000000000000000000000000000000000000000000000000000000000" "1111111111111111111111111111111111111111111111111111111111111111" "2222222222222222222222222222222222222222222222222222222222222222" >/dev/null 2>&1; then fail=1; fi
  printf '%s\n' '{' >"$temp/malformed.json"
  if validate_preflight_json "$temp/malformed.json" "p" "l" "2222222222222222222222222222222222222222222222222222222222222222" >/dev/null 2>&1; then fail=1; fi
  if validate_evidence_json "$temp/malformed.json" "0000000000000000000000000000000000000000000000000000000000000000" "1111111111111111111111111111111111111111111111111111111111111111" "2222222222222222222222222222222222222222222222222222222222222222" >/dev/null 2>&1; then fail=1; fi
  printf '%s\n' '{"format":"vokra-cosyvoice2-hift-apple-evidence-v1","status":"PASS","publication":"NO_UPLOAD","backend":"CPU+Metal","device_evidence":"system_profiler:success","gguf_sha256":"0000000000000000000000000000000000000000000000000000000000000000","reference_manifest_sha256":"1111111111111111111111111111111111111111111111111111111111111111","license_manifest_sha256":"2222222222222222222222222222222222222222222222222222222222222222","atol":0.01,"f0_cpu_reference_max_abs":0.02,"pcm_cpu_reference_max_abs":0.0,"pcm_metal_reference_max_abs":0.0,"pcm_metal_cpu_max_abs":0.0,"scope":{"component":"standalone_cosyvoice2_hift","f0":"CPU/reference only; Metal F0 not separately exposed","pcm":"CPU/reference, Metal/reference, Metal/CPU","full_cosyvoice2_e2e":"NOT_CLAIMED"}}' >"$temp/bound.json"
  if validate_evidence_json "$temp/bound.json" "0000000000000000000000000000000000000000000000000000000000000000" "1111111111111111111111111111111111111111111111111111111111111111" "2222222222222222222222222222222222222222222222222222222222222222" >/dev/null 2>&1; then fail=1; fi
  (( fail == 0 )) || return 1
  echo 'apple-silicon-cosyvoice2-hift.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
parse_args(){
  local seen key value
  GGUF=""
  GGUF_SHA=""
  REFERENCE=""
  REFERENCE_SHA=""
  LICENSE=""
  EVIDENCE=""
  seen=""
  while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || die "missing value for option: $1"
    key="$1"
    value="$2"
    [[ "$value" != --* ]] || die "missing value for option: $key"
    case "$key" in
      --gguf) [[ "$seen" != *'|gguf|'* ]] || die 'duplicate option: --gguf'; GGUF="$value"; seen="$seen|gguf|";;
      --gguf-sha256) [[ "$seen" != *'|gguf-sha256|'* ]] || die 'duplicate option: --gguf-sha256'; GGUF_SHA="$value"; seen="$seen|gguf-sha256|";;
      --reference-dir) [[ "$seen" != *'|reference-dir|'* ]] || die 'duplicate option: --reference-dir'; REFERENCE="$value"; seen="$seen|reference-dir|";;
      --reference-manifest-sha256) [[ "$seen" != *'|reference-manifest-sha256|'* ]] || die 'duplicate option: --reference-manifest-sha256'; REFERENCE_SHA="$value"; seen="$seen|reference-manifest-sha256|";;
      --license-manifest) [[ "$seen" != *'|license-manifest|'* ]] || die 'duplicate option: --license-manifest'; LICENSE="$value"; seen="$seen|license-manifest|";;
      --evidence-dir) [[ "$seen" != *'|evidence-dir|'* ]] || die 'duplicate option: --evidence-dir'; EVIDENCE="$value"; seen="$seen|evidence-dir|";;
      *) usage; die "unknown argument: $key";;
    esac
    shift 2
  done
  [[ -n "$GGUF" && -n "$GGUF_SHA" && -n "$REFERENCE" && -n "$REFERENCE_SHA" && -n "$LICENSE" && -n "$EVIDENCE" ]] || die 'all six options are required'
}
parse_args "$@"
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
REFERENCE_FILES="$(find "$REFERENCE" -mindepth 1 -maxdepth 1 -print | sort)"
EXPECTED_REFERENCE_FILES="$REFERENCE/f0.f32"$'\n'"$REFERENCE/manifest.json"$'\n'"$REFERENCE/mel.f32"$'\n'"$REFERENCE/pcm.f32"
[[ "$REFERENCE_FILES" == "$EXPECTED_REFERENCE_FILES" ]] || die 'reference packet must contain exactly f0.f32, manifest.json, mel.f32, pcm.f32'
for reference_file in "$REFERENCE/f0.f32" "$REFERENCE/manifest.json" "$REFERENCE/mel.f32" "$REFERENCE/pcm.f32"; do require_file "$reference_file" "reference artifact"; done
reject_checked_in "$LICENSE"
CHECKOUT="$ROOT"
for pair in "$GGUF|$REFERENCE" "$GGUF|$LICENSE" "$GGUF|$EVIDENCE" "$REFERENCE|$LICENSE" "$REFERENCE|$EVIDENCE" "$LICENSE|$EVIDENCE" "$CHECKOUT|$GGUF" "$CHECKOUT|$REFERENCE" "$CHECKOUT|$LICENSE" "$CHECKOUT|$EVIDENCE"; do disjoint "${pair%%|*}" "${pair#*|}"; done
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
TMP="$(mktemp -d "${TMPDIR:-/tmp}/vokra-hift-apple.XXXXXX")"
trap 'rm -rf -- "$TMP"' EXIT
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$GATE" --license-manifest "$LICENSE" >"$TMP/license.log" 2>&1 || { cat "$TMP/license.log" >&2; die 'external approved license preflight failed'; }
validate_preflight_json "$TMP/license.log" "$ROOT/tools/parity/cosyvoice2_hift_reference/pyproject.toml" "$ROOT/tools/parity/cosyvoice2_hift_reference/uv.lock" "$(sha "$LICENSE")" || die 'license preflight JSON was not an exact PASS object'
VOKRA_REMOTE_APPLE_SILICON=1 VOKRA_COSYVOICE2_HIFT_GGUF="$GGUF" VOKRA_COSYVOICE2_HIFT_GGUF_SHA256="$GGUF_SHA" VOKRA_COSYVOICE2_HIFT_REFERENCE_DIR="$REFERENCE" VOKRA_COSYVOICE2_HIFT_REFERENCE_MANIFEST_SHA256="$REFERENCE_SHA" VOKRA_COSYVOICE2_HIFT_LICENSE_MANIFEST="$LICENSE" VOKRA_COSYVOICE2_HIFT_APPLE_EVIDENCE_DIR="$EVIDENCE" CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=1 cargo test --offline --locked --release -p vokra-models --test parity_cosyvoice2_hift_apple "$TEST_NAME" -- --ignored --exact --nocapture >"$TMP/cargo.log" 2>&1 || { cat "$TMP/cargo.log" >&2; die 'Apple HiFT parity failed'; }
[[ "$(grep -Ec '^test [^[:space:]].* \.\.\. (ok|ignored|FAILED)$' "$TMP/cargo.log")" == 1 ]] || die 'Cargo did not report exactly one test case result'
[[ "$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$TMP/cargo.log")" == 1 ]] || die 'named test did not pass exactly once'
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9.]+s$' "$TMP/cargo.log")" == 1 ]] || die 'Cargo did not report one exact passing result summary'
[[ -f "$EVIDENCE/evidence.json" && ! -L "$EVIDENCE/evidence.json" ]] || die 'evidence marker missing'
validate_evidence_json "$EVIDENCE/evidence.json" "$GGUF_SHA" "$REFERENCE_SHA" "$(sha "$LICENSE")" || die 'evidence JSON failed strict validation'
echo "COSYVOICE2_HIFT_APPLE_PARITY_PASS evidence=$EVIDENCE/evidence.json gguf_sha256=$GGUF_SHA reference_manifest_sha256=$REFERENCE_SHA"
