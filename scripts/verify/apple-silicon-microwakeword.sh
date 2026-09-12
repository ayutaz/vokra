#!/usr/bin/env bash
# Apple Silicon consumer for the authenticated microWakeWord VAST packet.
# This worker never downloads, converts, uploads, or publishes model bytes.
# The embedded KWS crate has no Metal seam; that absence is tested and is
# reported as an explicit UnsupportedOp (CPU fallback is forbidden).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CRATE="$ROOT/crates/vokra-kws-micro"
TEST_FILE="$CRATE/tests/parity_microwakeword.rs"
CPU_TEST="parity_microwakeword_end_to_end_output"
METAL_TEST="microwakeword_metal_backend_is_explicitly_unsupported"
SOURCE_TFLITE_SHA256="21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77"
CONVERTER_LOCK_SHA256="984703d5bafdd6c88006bd381095961d42ef684d269d66194edbeda1fddf8dc2"
REFERENCE_LOCK_SHA256="736fca6145c24984531ef11258cd64aebbb188fa8830300b09232cac0fe567f3"
TOPOLOGY_SHA256="e17fa0cae8d504ce71b49ad2113fc6f7ebba9e74dd4070d26e7f291dcbfaf621"
RAW_INVENTORY_SHA256="ce57a719f60af3a494cbd8fb22ff30fdb405b0a3037b049333f25f5794749989"

log() { printf '[microwakeword-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

usage() {
  cat >&2 <<'EOF'
usage: apple-silicon-microwakeword.sh \
  --gguf ABS_FILE --fixtures ABS_DIR --validation-json ABS_FILE \
  --path-c-log ABS_FILE --dependency-evidence ABS_FILE \
  --expected-head LOWERCASE_40_HEX --evidence-dir ABSENT_DIR
       apple-silicon-microwakeword.sh --self-test

Consumes the reviewed, NO_UPLOAD VAST packet on a disposable Darwin/arm64
host. It reruns the exact non-ignored Path-C CPU/reference test. The second
exact non-ignored test proves that this embedded crate has no Metal seam and
records Metal/reference and Metal/CPU as explicit UnsupportedOp; it never
falls back to CPU and never claims Apple completion.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label must be a non-empty regular file: $path"
}

reject_symlink_ancestors() {
  local path="$1" label="$2" current=/ component
  [[ "$path" == /* ]] || die "$label must be absolute: $path"
  local rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=''; fi
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || die "$label contains dot components"
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || die "$label has a symlink ancestor: $current"
  done
}

canonical_existing() {
  local path="$1" label="$2"
  reject_symlink_ancestors "$path" "$label"
  [[ -e "$path" && ! -L "$path" ]] || die "$label is missing or symlinked: $path"
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd -P); else
    local parent; parent="$(dirname "$path")"
    (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$path")")
  fi
}

canonical_absent() {
  local path="$1" label="$2" cursor parent suffix='' component
  reject_symlink_ancestors "$path" "$label"
  [[ ! -e "$path" && ! -L "$path" ]] || die "$label must be absent before validation: $path"
  cursor="$path"
  while [[ ! -e "$cursor" && ! -L "$cursor" ]]; do
    component="$(basename "$cursor")"; suffix="/$component$suffix"; parent="$(dirname "$cursor")"
    [[ "$parent" != "$cursor" ]] || die "$label cannot be canonicalized"
    cursor="$parent"
  done
  [[ -d "$cursor" && ! -L "$cursor" ]] || die "$label parent is unavailable"
  (cd -P "$cursor" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_disjoint() {
  local left="$1" right="$2"
  [[ "$left" != "$right" && "$left" != "$right"/* && "$right" != "$left"/* ]]
}

require_clean_head() {
  local expected="$1" actual status
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
  status="$(git -C "$ROOT" status --porcelain --untracked-files=all)"
  [[ -z "$status" ]] || die 'checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null)" || die 'cannot resolve checkout HEAD'
  [[ "$actual" == "$expected" ]] || die "HEAD $actual does not match --expected-head $expected"
}

require_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'Darwin arm64 is required'
  command -v cargo >/dev/null 2>&1 || die 'cargo is required'
  command -v xcrun >/dev/null 2>&1 || die 'xcrun is required'
  xcrun -sdk macosx metal -v >/dev/null 2>&1 || die 'Metal compiler is unavailable'
}

require_fixture_tree() {
  local directory="$1" actual entry
  [[ -d "$directory" && ! -L "$directory" ]] || die 'fixtures must be a real directory'
  [[ -z "$(find -P "$directory" -type l -print -quit)" ]] || die 'fixtures contain a symlink'
  actual="$(find -P "$directory" -mindepth 1 -maxdepth 1 -type f -print | sed 's#.*/##' | LC_ALL=C sort | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  local expected='features_invocation_00.bin features_invocation_01.bin features_invocation_02.bin features_invocation_03.bin features_ref.bin input_invocation_00.bin input_invocation_01.bin input_invocation_02.bin input_invocation_03.bin input_pcm.bin manifest.json output_invocation_00.bin output_invocation_00_f32.bin output_invocation_01.bin output_invocation_01_f32.bin output_invocation_02.bin output_invocation_02_f32.bin output_invocation_03.bin output_invocation_03_f32.bin output_ref.bin stress_inputs.bin stress_stage_tensor_47.bin stress_stage_tensor_50.bin stress_stage_tensor_51.bin stress_stage_tensor_54.bin stress_stage_tensor_55.bin stress_stage_tensor_58.bin stress_stage_tensor_59.bin stress_stage_tensor_62.bin stress_stage_tensor_63.bin stress_stage_tensor_67.bin stress_stage_tensor_68.bin stress_stage_tensor_69.bin'
  [[ "$actual" == "$expected" ]] || die 'fixture entry set drifted'
  for entry in $expected; do [[ -s "$directory/$entry" && ! -L "$directory/$entry" ]] || die "fixture is missing or empty: $entry"; done
}

validate_packet() {
  local gguf="$1" fixtures="$2" validation="$3" path_log="$4" dependency="$5" expected_head="$6"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$gguf" "$fixtures" "$validation" "$path_log" "$dependency" "$expected_head" \
    "$SOURCE_TFLITE_SHA256" "$CONVERTER_LOCK_SHA256" "$REFERENCE_LOCK_SHA256" "$TOPOLOGY_SHA256" "$RAW_INVENTORY_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

args = sys.argv[1:]
gguf, fixtures, validation, path_log, dependency = map(Path, args[:5])
expected_head, source_sha, converter_lock, reference_lock, topology, inventory = args[5:]

def reject(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out

def read_json(path, label):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(path):
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}

v = read_json(validation, "validation JSON")
if v.get("status") != "PATH_C_PASS" or v.get("publication") != "NO_UPLOAD":
    raise ValueError("VAST packet is not PATH_C_PASS/NO_UPLOAD")
if v.get("model_payload_transfer") != "STAGED_FOR_AUTHENTICATED_APPLE_TRANSFER":
    raise ValueError("VAST packet was not staged for authenticated Apple transfer")
if v.get("git_commit") != expected_head:
    raise ValueError("VAST git commit is not the requested exact HEAD")
if v.get("source_tflite_sha256") != source_sha:
    raise ValueError("source TFLite SHA drift")
if v.get("converter_lock_sha256") != converter_lock or v.get("reference_lock_sha256") != reference_lock:
    raise ValueError("uv lock SHA drift")
if v.get("reviewed_topology_sha256") != topology or v.get("raw_inventory_sha256") != inventory:
    raise ValueError("reviewed topology/raw inventory drift")
for key, path in (("gguf", gguf), ("path_c_log", path_log)):
    if v.get(key) != identity(path):
        raise ValueError(f"{key} identity does not match VAST packet")
manifest_path = fixtures / "manifest.json"
fixture_row = v.get("fixture_manifest")
if not isinstance(fixture_row, dict) or fixture_row.get("filename") != "manifest.json" or fixture_row.get("sha256") != digest(manifest_path) or fixture_row.get("status") != "REFERENCE_COMPLETE":
    raise ValueError("fixture manifest identity/status drift")
verification = v.get("verification", {})
if verification.get("test_identity") != "parity_microwakeword::parity_microwakeword_end_to_end_output":
    raise ValueError("VAST test identity drift")
if verification.get("reset_replay_invocations") != 4 or verification.get("stress_invocations") != 512 or verification.get("preserved_intermediate_stage_count") != 11 or verification.get("final_output_tensor") != 69:
    raise ValueError("VAST Path-C contract drift")
log = path_log.read_text(encoding="utf-8")
sentinel = "Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4"
if log.count(sentinel) != 1 or log.count("test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;") != 1:
    raise ValueError("VAST Path-C log is not the exact five-test PASS")
d = read_json(dependency, "dependency evidence")
if digest(dependency) != v.get("dependency_evidence_sha256"):
    raise ValueError("dependency evidence SHA drift")
if d.get("schema") != "microwakeword-reference-dependency-evidence-v1":
    raise ValueError("dependency evidence schema drift")
m = read_json(manifest_path, "reference manifest")
if m.get("schema") != "microwakeword-reference-v2" or m.get("status") != "REFERENCE_COMPLETE" or m.get("source_tflite_sha256") != source_sha:
    raise ValueError("reference manifest source/schema drift")
p = m.get("persistent_sequence", {})
replay = p.get("fresh_interpreter_reset_replay", {})
if p.get("invocation_count") != 4 or p.get("single_persistent_interpreter") is not True or replay.get("status") != "PASS" or replay.get("raw_outputs_match") is not True:
    raise ValueError("persistent/reset replay contract drift")
s = m.get("direct_int8_stress", {})
if s.get("invocation_count") != 512 or s.get("stage_tensor_indices") != [47, 50, 51, 54, 55, 58, 59, 62, 63, 67, 68, 69]:
    raise ValueError("stress trace contract drift")
dep = m.get("dependency_evidence", {})
if dep.get("schema") != "microwakeword-reference-dependency-evidence-v1" or dep.get("audit_status") != "PASS" or dep.get("review_status") != "VALIDATED_EXACT_OWNER_REVIEWED" or dep.get("sha256") != digest(dependency):
    raise ValueError("reference dependency review is not exact owner-reviewed")
artefacts = m.get("artefacts")
if not isinstance(artefacts, list) or len(artefacts) != 32:
    raise ValueError("fixture artefact count drift")
seen = set()
for row in artefacts:
    if not isinstance(row, dict):
        raise ValueError("malformed fixture artefact")
    name = row.get("path")
    if not isinstance(name, str) or name in seen or "/" in name or "\\" in name:
        raise ValueError("fixture artefact path/key drift")
    seen.add(name)
    path = fixtures / name
    if path.is_symlink() or not path.is_file() or row.get("bytes") != path.stat().st_size or row.get("sha256") != digest(path):
        raise ValueError(f"fixture artefact identity drift: {name}")
actual_files = {p.name for p in fixtures.iterdir() if p.name != "manifest.json"}
if seen != actual_files or "manifest.json" in seen:
    raise ValueError("fixture artefact set drift")
print("microWakeWord Apple packet validation: PASS")
PY
}

digest_inputs() {
  local gguf="$1" fixtures="$2" validation="$3" path_log="$4" dependency="$5"
  {
    sha256_file "$gguf"
    find -P "$fixtures" -type f -exec shasum -a 256 {} + | LC_ALL=C sort
    sha256_file "$validation"
    sha256_file "$path_log"
    sha256_file "$dependency"
  }
}

parse_cpu_log() {
  local file="$1" marker same split interleaved standalone marker_count test_lines
  marker='Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4'
  same="$(grep -Fxc -- "test ${CPU_TEST} ... ok" "$file" || true)"
  split="$(grep -Fxc -- "test ${CPU_TEST} ..." "$file" || true)"
  interleaved="$(grep -Fxc -- "test ${CPU_TEST} ... $marker" "$file" || true)"
  standalone="$(grep -Fxc -- 'ok' "$file" || true)"
  marker_count="$(awk -v marker="$marker" '{ count += gsub(marker, "&") } END { print count + 0 }' "$file")"
  test_lines="$(grep -Ec '^test ' "$file" || true)"
  [[ "$test_lines" == 2 ]] || die 'CPU log must contain exactly one test line and one result line'
  [[ "$same" == 1 && "$split" == 0 && "$interleaved" == 0 && "$standalone" == 0 || "$same" == 0 && "$split" == 1 && "$interleaved" == 0 && "$standalone" == 1 || "$same" == 0 && "$split" == 0 && "$interleaved" == 1 && "$standalone" == 1 ]] || die 'CPU named Path-C test did not pass in a supported libtest shape'
  [[ "$(grep -Ec '^test .* \.\.\. (FAILED|ignored|skipped)$' "$file" || true)" == 0 ]] || die 'CPU named test was failed, ignored, or skipped'
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || true)" == 1 ]] || die 'CPU test result was not exactly one non-ignored pass'
  [[ "$marker_count" == 1 ]] || die 'CPU Path-C parity sentinel missing/duplicated'
}

parse_metal_log() {
  local file="$1" marker same split interleaved standalone marker_count test_lines
  marker='MICROWAKEWORD_METAL_UNSUPPORTED_OP backend=metal reason=no-vokra-kws-micro-metal-seam CPU_FALLBACK=FORBIDDEN'
  same="$(grep -Fxc -- "test ${METAL_TEST} ... ok" "$file" || true)"
  split="$(grep -Fxc -- "test ${METAL_TEST} ..." "$file" || true)"
  interleaved="$(grep -Fxc -- "test ${METAL_TEST} ... $marker" "$file" || true)"
  standalone="$(grep -Fxc -- 'ok' "$file" || true)"
  marker_count="$(awk -v marker="$marker" '{ count += gsub(marker, "&") } END { print count + 0 }' "$file")"
  test_lines="$(grep -Ec '^test ' "$file" || true)"
  [[ "$test_lines" == 2 ]] || die 'Metal log must contain exactly one test line and one result line'
  [[ "$same" == 1 && "$split" == 0 && "$interleaved" == 0 && "$standalone" == 0 || "$same" == 0 && "$split" == 1 && "$interleaved" == 0 && "$standalone" == 1 || "$same" == 0 && "$split" == 0 && "$interleaved" == 1 && "$standalone" == 1 ]] || die 'Metal contract test did not pass in a supported libtest shape'
  [[ "$(grep -Ec '^test .* \.\.\. (FAILED|ignored|skipped)$' "$file" || true)" == 0 ]] || die 'Metal contract test was failed, ignored, or skipped'
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || true)" == 1 ]] || die 'Metal contract result was not exactly one non-ignored pass'
  [[ "$marker_count" == 1 ]] || die 'explicit Metal UnsupportedOp marker missing/duplicated'
}

self_test() {
  local self="${BASH_SOURCE[0]}" fail=0
  for token in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON "$CPU_TEST" "$METAL_TEST" \
    'MICROWAKEWORD_METAL_UNSUPPORTED_OP' 'CPU_FALLBACK=FORBIDDEN' \
    'parity_microwakeword::parity_microwakeword_end_to_end_output' \
    'PATH_C_PASS' 'NO_UPLOAD' 'VALIDATED_EXACT_OWNER_REVIEWED' \
    'CARGO_BUILD_JOBS=1' 'CARGO_NET_OFFLINE=true' '--offline' '--locked' \
    'xcrun -sdk macosx metal -v' 'source_tflite_sha256' 'reviewed_topology_sha256' \
    'fresh_interpreter_reset_replay' 'stage_tensor_indices' 'CPU_REFERENCE_PASS_METAL_UNSUPPORTED' \
    'model_payload_transfer' 'STAGED_FOR_AUTHENTICATED_APPLE_TRANSFER' \
    'test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'; do
    grep -Fq -- "$token" "$self" || { log "self-test missing contract token: $token"; fail=1; }
  done
  if grep -En '(^|[;&|[:space:]])(curl|wget|git[[:space:]]+(clone|fetch|pull|push)|publish-one\.sh|upload\.sh|--push|--upload)([[:space:]]|$)' "$self" >/dev/null; then
    log 'self-test found download/publication command'; fail=1
  fi
  if grep -En '(^|[;&|])[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then
    log 'self-test found raw Python/pip invocation'; fail=1
  fi
  local temporary same_cpu split_cpu interleaved_cpu same_metal split_metal interleaved_metal extra ignored incomplete arbitrary
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/microwakeword-apple-selftest.XXXXXX")"
  same_cpu="$temporary/cpu-same.log"
  split_cpu="$temporary/cpu-split.log"
  interleaved_cpu="$temporary/cpu-interleaved.log"
  same_metal="$temporary/metal-same.log"
  split_metal="$temporary/metal-split.log"
  interleaved_metal="$temporary/metal-interleaved.log"
  extra="$temporary/extra.log"
  ignored="$temporary/ignored.log"
  incomplete="$temporary/incomplete.log"
  arbitrary="$temporary/arbitrary.log"
  printf '%s\n' \
    "test ${CPU_TEST} ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' \
    'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' > "$same_cpu"
  printf '%s\n' \
    "test ${CPU_TEST} ..." \
    'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' \
    'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' > "$split_cpu"
  printf '%s\n' \
    "test ${CPU_TEST} ... Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4" \
    'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' > "$interleaved_cpu"
  printf '%s\n' \
    "test ${METAL_TEST} ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' \
    'MICROWAKEWORD_METAL_UNSUPPORTED_OP backend=metal reason=no-vokra-kws-micro-metal-seam CPU_FALLBACK=FORBIDDEN' > "$same_metal"
  printf '%s\n' \
    "test ${METAL_TEST} ..." \
    'MICROWAKEWORD_METAL_UNSUPPORTED_OP backend=metal reason=no-vokra-kws-micro-metal-seam CPU_FALLBACK=FORBIDDEN' \
    'ok' \
    'MICROWAKEWORD_METAL_UNSUPPORTED_OP backend=metal reason=no-vokra-kws-micro-metal-seam CPU_FALLBACK=FORBIDDEN' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' > "$split_metal"
  printf '%s\n' \
    "test ${METAL_TEST} ... MICROWAKEWORD_METAL_UNSUPPORTED_OP backend=metal reason=no-vokra-kws-micro-metal-seam CPU_FALLBACK=FORBIDDEN" \
    'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' > "$interleaved_metal"
  printf '%s\n' \
    "test ${CPU_TEST} ... ok" "test ${METAL_TEST} ... ok" \
    'test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.01s' \
    'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' > "$extra"
  printf '%s\n' \
    "test ${CPU_TEST} ... ignored" \
    'test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 3 filtered out; finished in 0.01s' \
    'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' > "$ignored"
  printf '%s\n' \
    "test ${CPU_TEST} ..." \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' \
    'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' > "$incomplete"
  printf '%s\n' \
    "test ${CPU_TEST} ... unexpected suffix" 'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s' \
    'Path-C authenticated streaming parity PASS: 512 invocations, 11 preserved intermediates, final output, reset replay=4' > "$arbitrary"
  parse_cpu_log "$same_cpu" || fail=1
  parse_cpu_log "$split_cpu" || fail=1
  parse_cpu_log "$interleaved_cpu" || fail=1
  parse_metal_log "$same_metal" || fail=1
  parse_metal_log "$interleaved_metal" || fail=1
  if (parse_metal_log "$split_metal") >/dev/null 2>&1; then log 'self-test accepted duplicated Metal marker'; fail=1; fi
  if (parse_cpu_log "$extra") >/dev/null 2>&1; then log 'self-test accepted an extra test'; fail=1; fi
  if (parse_cpu_log "$ignored") >/dev/null 2>&1; then log 'self-test accepted ignored test'; fail=1; fi
  if (parse_cpu_log "$incomplete") >/dev/null 2>&1; then log 'self-test accepted incomplete split test'; fail=1; fi
  if (parse_cpu_log "$arbitrary") >/dev/null 2>&1; then log 'self-test accepted arbitrary interleaved suffix'; fail=1; fi
  rm -rf -- "$temporary"
  grep -Fq 'include_str!("../Cargo.toml")' "$TEST_FILE" || { log 'Metal contract test is missing'; fail=1; }
  if "$self" --unknown >/dev/null 2>&1; then log 'self-test accepted unknown option'; fail=1; fi
  if "$self" --self-test extra >/dev/null 2>&1; then log 'self-test accepted an extra argument'; fail=1; fi
  (( fail == 0 )) && log 'self-test: PASS' || return 1
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi

GGUF=''; FIXTURES=''; VALIDATION=''; PATH_LOG=''; DEPENDENCY=''; EXPECTED_HEAD=''; EVIDENCE=''
while (($#)); do
  case "$1" in
    --gguf) [[ $# -ge 2 && -z "$GGUF" ]] || die '--gguf is missing or duplicated'; GGUF="$2"; shift 2;;
    --fixtures) [[ $# -ge 2 && -z "$FIXTURES" ]] || die '--fixtures is missing or duplicated'; FIXTURES="$2"; shift 2;;
    --validation-json) [[ $# -ge 2 && -z "$VALIDATION" ]] || die '--validation-json is missing or duplicated'; VALIDATION="$2"; shift 2;;
    --path-c-log) [[ $# -ge 2 && -z "$PATH_LOG" ]] || die '--path-c-log is missing or duplicated'; PATH_LOG="$2"; shift 2;;
    --dependency-evidence) [[ $# -ge 2 && -z "$DEPENDENCY" ]] || die '--dependency-evidence is missing or duplicated'; DEPENDENCY="$2"; shift 2;;
    --expected-head) [[ $# -ge 2 && -z "$EXPECTED_HEAD" ]] || die '--expected-head is missing or duplicated'; EXPECTED_HEAD="$2"; shift 2;;
    --evidence-dir) [[ $# -ge 2 && -z "$EVIDENCE" ]] || die '--evidence-dir is missing or duplicated'; EVIDENCE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage; die "unknown option: $1";;
  esac
done
[[ -n "$GGUF$FIXTURES$VALIDATION$PATH_LOG$DEPENDENCY$EXPECTED_HEAD$EVIDENCE" ]] || { usage; die 'all options are required'; }
[[ "$EVIDENCE" == /* ]] || die '--evidence-dir must be absolute'
require_host
require_clean_head "$EXPECTED_HEAD"
[[ -f "$TEST_FILE" && -f "$CRATE/Cargo.toml" ]] || die 'microWakeWord crate/test is missing'
require_file 'reviewed GGUF' "$GGUF"
require_file 'VAST validation JSON' "$VALIDATION"
require_file 'VAST Path-C log' "$PATH_LOG"
require_file 'dependency evidence' "$DEPENDENCY"
require_fixture_tree "$FIXTURES"
gguf_real="$(canonical_existing "$GGUF" gguf)"
fixtures_real="$(canonical_existing "$FIXTURES" fixtures)"
validation_real="$(canonical_existing "$VALIDATION" validation-json)"
path_log_real="$(canonical_existing "$PATH_LOG" path-c-log)"
dependency_real="$(canonical_existing "$DEPENDENCY" dependency-evidence)"
root_real="$(canonical_existing "$ROOT" checkout)"
evidence_real="$(canonical_absent "$EVIDENCE" evidence-dir)"
for input in "$gguf_real" "$fixtures_real" "$validation_real" "$path_log_real" "$dependency_real"; do
  [[ "$input" != "$root_real" && "$input" != "$root_real"/* ]] || die 'packet input must be outside checkout'
  paths_disjoint "$input" "$evidence_real" || die 'evidence overlaps packet input'
done
paths_disjoint "$root_real" "$evidence_real" || die 'evidence overlaps checkout'
validate_packet "$gguf_real" "$fixtures_real" "$validation_real" "$path_log_real" "$dependency_real" "$EXPECTED_HEAD"
input_digest_before="$(digest_inputs "$gguf_real" "$fixtures_real" "$validation_real" "$path_log_real" "$dependency_real")"
run_dir="$(mktemp -d "${TMPDIR:-/tmp}/microwakeword-apple.XXXXXX")"
trap 'rm -rf -- "$run_dir"' EXIT
cpu_log="$run_dir/cpu.log"
set +e
VOKRA_KWS_REAL_GGUF="$gguf_real" VOKRA_KWS_REAL_FIXTURES="$fixtures_real" CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true \
  cargo test --manifest-path "$ROOT/Cargo.toml" --locked --offline -p vokra-kws-micro \
  --test parity_microwakeword "$CPU_TEST" -- --exact --nocapture >"$cpu_log" 2>&1
cpu_status=$?
set -e
[[ "$cpu_status" == 0 ]] || { tail -n 80 "$cpu_log" >&2 || true; die 'Apple CPU/reference Path-C parity failed'; }
parse_cpu_log "$cpu_log"
[[ "$(digest_inputs "$gguf_real" "$fixtures_real" "$validation_real" "$path_log_real" "$dependency_real")" == "$input_digest_before" ]] || die 'packet changed during CPU parity'
metal_log="$run_dir/metal-contract.log"
set +e
CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --manifest-path "$ROOT/Cargo.toml" --locked --offline -p vokra-kws-micro \
  --test parity_microwakeword "$METAL_TEST" -- --exact --nocapture >"$metal_log" 2>&1
metal_status=$?
set -e
[[ "$metal_status" == 0 ]] || { tail -n 80 "$metal_log" >&2 || true; die 'Metal no-seam contract test failed'; }
parse_metal_log "$metal_log"
[[ "$(digest_inputs "$gguf_real" "$fixtures_real" "$validation_real" "$path_log_real" "$dependency_real")" == "$input_digest_before" ]] || die 'packet changed during Metal contract test'
mkdir "$EVIDENCE"
cp -p "$cpu_log" "$EVIDENCE/cpu.log"
cp -p "$metal_log" "$EVIDENCE/metal-contract.log"
{
  echo 'schema=microwakeword-apple-validation-v1'
  echo 'status=CPU_REFERENCE_PASS_METAL_UNSUPPORTED'
  echo 'verdict=BLOCKED_UNSUPPORTED'
  echo 'publication=NO_UPLOAD'
  echo "expected_head=$EXPECTED_HEAD"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "gguf_sha256=$(sha256_file "$gguf_real")"
  echo "fixture_manifest_sha256=$(sha256_file "$fixtures_real/manifest.json")"
  echo "validation_json_sha256=$(sha256_file "$validation_real")"
  echo "path_c_log_sha256=$(sha256_file "$path_log_real")"
  echo "dependency_evidence_sha256=$(sha256_file "$dependency_real")"
  echo "cpu_test=$CPU_TEST"
  echo 'cpu_reference=PASS'
  echo 'persistent_state=PASS'
  echo 'reset_replay=PASS'
  echo 'metal_reference=UNSUPPORTED_OP'
  echo 'metal_cpu=UNSUPPORTED_OP'
  echo 'cpu_fallback=FORBIDDEN'
  echo 'metal_test=EXPLICIT_UNSUPPORTED'
} > "$EVIDENCE/summary.txt"
{
  echo 'backend=metal'
  echo 'status=UNSUPPORTED_OP'
  echo 'metal_reference=UNSUPPORTED_OP'
  echo 'metal_cpu=UNSUPPORTED_OP'
  echo 'cpu_fallback=FORBIDDEN'
  echo 'reason=no-vokra-kws-micro-metal-seam'
  echo 'publication=NO_UPLOAD'
  grep -Fxc -- 'MICROWAKEWORD_METAL_UNSUPPORTED_OP backend=metal reason=no-vokra-kws-micro-metal-seam CPU_FALLBACK=FORBIDDEN' "$metal_log"
} > "$EVIDENCE/metal.txt"
{
  echo "uname=$(uname -a)"
  echo "machine=$(uname -m)"
  echo "expected_head=$EXPECTED_HEAD"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "sysctl_hw.model=$(sysctl -n hw.model 2>/dev/null || echo unavailable)"
  echo "sysctl_hw.machine=$(sysctl -n hw.machine 2>/dev/null || echo unavailable)"
  echo "sysctl_hw.memsize=$(sysctl -n hw.memsize 2>/dev/null || echo unavailable)"
  xcrun -sdk macosx metal -v 2>&1 | sed 's/[[:space:]]\+/ /g'
} > "$EVIDENCE/environment.txt"
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || die 'HEAD changed before evidence handoff'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before evidence handoff'
log 'CPU/reference PASS; Metal/reference and Metal/CPU explicit UnsupportedOp; CPU fallback forbidden; NO_UPLOAD'
exit 3
