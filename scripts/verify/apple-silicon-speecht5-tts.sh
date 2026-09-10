#!/usr/bin/env bash
# Exact public SpeechT5-TTS CPU/Metal parity on a disposable remote Apple host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/speecht5_tts"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

PUBLIC_GGUF_SHA256="f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0"
TTS_REVISION="30fcde30f19b87502b8435427b5f5068e401d5f6"
PREVIOUS_ISOLATED_TRANSFORMERS_PIN="transformers==5.5.0"
ISOLATED_TRANSFORMERS_PIN="transformers==5.10.4"
TRANSFORMERS_SECURITY_ADVISORY="GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM="5.10.0"
TRANSFORMERS_COMPATIBILITY_STATUS="AUTHENTICATED_API_SMOKE"
REFERENCE_IMPLEMENTATION="transformers.models.speecht5.modeling_speecht5.SpeechT5ForTextToSpeech.generate_speech"
SOURCE_WEIGHT_SHA256="d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"
SAFE_TENSOR_WEIGHT_SHA256="87d96b215548dfba6251e15ad0b861e9d01d640d4715767759d6b12a12c62582"
PACKAGE_ROWS_SHA256="d49d3091fffcc8df5a0d75f0cf2a94e14ddd90a164698f37bc64bc788bc545bf"
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000

log() { printf '[speecht5-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-speecht5-tts.sh \
  --gguf <public-speecht5.gguf> --reference <official-reference-dir> \
  --reference-sha256 <64-hex-from-vast-evidence> \
  --approval-evidence <external-approval.json> \
  --api-smoke-evidence <vast-api-smoke.json> \
  --api-smoke-sha256 <64-hex-from-vast-evidence> \
  --evidence-dir <empty-dir>
       apple-silicon-speecht5-tts.sh --self-test

Runs the official-reference CPU/Metal text-to-mel parity test against the
exact public vokra/speecht5-tts GGUF. It refuses the maintainer class of
machine and requires VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least
32 GB physical memory, a clean checkout, and pre-staged real inputs.

This script does not download, upload, convert, publish, or delete a model.
Transfer the fixed public GGUF and VAST-produced reference directly to a
disposable remote Apple host. The expected reference.json hash is mandatory
and must come from VAST evidence. The API-smoke evidence is a separate,
hash-bound VAST file and is not the signed owner/license approval. Pull only
the evidence directory afterward, then remove staged model data or destroy the
remote worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

json_scalar() {
  local reference_json="$1" key="$2"
  if [[ "${VOKRA_SPEECHT5_SELF_TEST:-0}" == 1 && "$(uname -s)" == Linux ]]; then
    command -v uv >/dev/null 2>&1 || die "uv is required for the Linux self-test JSON fallback"
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$reference_json" "$key" <<'PY'
import json
import pathlib
import sys

path, key = sys.argv[1:]

def reject(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result

with pathlib.Path(path).open(encoding="utf-8") as handle:
    data = json.load(handle, object_pairs_hook=reject)
if key not in data:
    raise ValueError(f"missing JSON key: {key}")
value = data[key]
if key == "text":
    if not isinstance(value, str) or not value:
        raise ValueError("text must be a non-empty string")
elif type(value) is not int or value < 0:
    raise ValueError(f"{key} must be a non-negative integer")
print(value)
PY
    return
  fi
  plutil -extract "$key" raw -o - "$reference_json"
}

license_preflight() {
  local approval="$1"
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "SpeechT5 preflight inputs are missing or symlinked"
  [[ -f "$approval" && -s "$approval" && ! -L "$approval" ]] || die "approval evidence must be a non-empty regular non-symlink file"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && -s "$path" && ! -L "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_empty_directory() {
  local directory="$1"
  if [[ -e "$directory" ]]; then
    [[ -d "$directory" ]] || die "evidence path is not a directory: $directory"
    [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "evidence directory must be empty: $directory"
  else
    mkdir -p "$directory"
  fi
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent scan rest component
  [[ ! -L "$path" ]] || return 1
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence_dir() {
  local target="$1"; shift; local canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "evidence directory must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize evidence directory: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$PREFLIGHT_GATE" "$PREFLIGHT_MANIFEST" "$@"; do
    [[ -n "$protected" && ( -e "$protected" || -L "$protected" ) ]] || continue
    [[ ! -L "$protected" ]] || { die "protected input is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected input: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "evidence directory overlaps protected input: $protected"; return 2; }
  done
  return 0
}

require_reference() {
  local directory="$1" filename reference_json
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference path is not a directory or is symlinked: $directory"
  reference_json="$directory/reference.json"
  for filename in text.txt tokens.u32 speaker.f32 before_postnet.f32 \
    after_postnet.f32 frames.txt decoder_steps.txt reference.json; do
    require_file "SpeechT5 reference $filename" "$directory/$filename"
  done
  for filename in tokens.u32 speaker.f32 before_postnet.f32 after_postnet.f32; do
    verify_reference_hash "$reference_json" "$directory/$filename"
  done
  grep -F '"format": "vokra-speecht5-tts-reference-v1"' "$reference_json" >/dev/null \
    || die "reference format is not the pinned SpeechT5 schema"
  grep -F '"reference_package": "transformers==5.10.4"' "$reference_json" >/dev/null \
    || die "reference Transformers route is not 5.10.4"
  require_reference_identity "$reference_json"
  verify_reference_scalars "$directory"
}

require_reference_identity() {
  local reference_json="$1" field expected key_count exact_count
  for field in upstream_hf upstream_revision previous_isolated_transformers_pin \
    reference_package reference_implementation transformers_compatibility_status \
    transformers_security_advisory transformers_security_patched_minimum transformers_version; do
    case "$field" in
      upstream_hf) expected="microsoft/speecht5_tts" ;;
      upstream_revision) expected="$TTS_REVISION" ;;
      previous_isolated_transformers_pin) expected="$PREVIOUS_ISOLATED_TRANSFORMERS_PIN" ;;
      reference_package) expected="$ISOLATED_TRANSFORMERS_PIN" ;;
      reference_implementation) expected="$REFERENCE_IMPLEMENTATION" ;;
      transformers_compatibility_status) expected="$TRANSFORMERS_COMPATIBILITY_STATUS" ;;
      transformers_security_advisory) expected="$TRANSFORMERS_SECURITY_ADVISORY" ;;
      transformers_security_patched_minimum) expected="$TRANSFORMERS_SECURITY_PATCHED_MINIMUM" ;;
      transformers_version) expected="5.10.4" ;;
    esac
    key_count="$(grep -Ec "^[[:space:]]+\"$field\": " "$reference_json" || true)"
    exact_count="$(grep -Fxc "  \"$field\": \"$expected\"," "$reference_json" || true)"
    if [[ "$exact_count" != 1 ]]; then
      exact_count="$(grep -Fxc "  \"$field\": \"$expected\"" "$reference_json" || true)"
    fi
    if [[ "$key_count" != 1 || "$exact_count" != 1 ]]; then
      die "reference API-smoke identity is missing, duplicated, or mismatched: $field"
      return 2
    fi
  done
}

validate_api_smoke_evidence() {
  local api_smoke="$1" reference_json="$2" approval="$3" expected_head="$4" supplied_sha="$5"
  local project_sha lock_sha gate_sha manifest_sha owner_sha api_sha_before api_sha_after
  require_file "SpeechT5 API-smoke evidence" "$api_smoke"
  require_file "SpeechT5 owner/license approval" "$approval"
  [[ "$supplied_sha" =~ ^[0-9a-f]{64}$ ]] || { die "API-smoke SHA-256 must be lowercase 64-hex"; return 2; }
  api_sha_before="$(sha256_file "$api_smoke")"
  [[ "$api_sha_before" == "$supplied_sha" ]] || { die "API-smoke evidence SHA-256 differs before validation"; return 2; }
  project_sha="$(sha256_file "$PARITY_PROJECT/pyproject.toml")"
  lock_sha="$(sha256_file "$PARITY_PROJECT/uv.lock")"
  gate_sha="$(sha256_file "$PREFLIGHT_GATE")"
  manifest_sha="$(sha256_file "$PREFLIGHT_MANIFEST")"
  owner_sha="$(sha256_file "$approval")"
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$api_smoke" "$reference_json" "$approval" "$expected_head" \
    "$project_sha" "$lock_sha" "$PACKAGE_ROWS_SHA256" "$gate_sha" "$manifest_sha" "$owner_sha" \
    "$SOURCE_WEIGHT_SHA256" "$SAFE_TENSOR_WEIGHT_SHA256" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

api_path, reference_path, owner_path, expected_head, project_sha, lock_sha, rows_sha, gate_sha, manifest_sha, owner_sha, source_sha, safe_sha = sys.argv[1:]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

def load(path, label):
    raw = pathlib.Path(path).read_bytes()
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {label} key: {key}")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value

def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()

def digest_canonical(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

api = load(api_path, "API-smoke")
owner = load(owner_path, "owner approval")
reference = load(reference_path, "SpeechT5 reference")
if digest_bytes(pathlib.Path(owner_path).read_bytes()) != owner_sha:
    raise ValueError("owner approval bytes changed during API-smoke validation")
api_keys = {
    "call", "call_checkpoint_sha256", "checkpoint_files", "environment", "format",
    "frames", "input_sha256", "lock_sha256", "mel_bins", "output_count",
    "output_sha256", "package_rows_sha256", "package_sha256", "previous_isolated_transformers_pin",
    "project_sha256", "publication", "reference_implementation", "reference_package",
    "revision_sha256", "status", "transformers_security_advisory", "transformers_security_patched_minimum",
    "upstream_hf", "upstream_revision", "upload", "vocoder", "vokra_head", "vokra_root",
    "vokra_clean", "approval_evidence_sha256", "approval_scope_sha256", "approval_signer", "project_dir",
    "preflight_gate", "preflight_gate_sha256", "preflight_manifest_sha256", "float8_import_compat",
    "conversion_source_sha256", "weight_loading",
}
if set(api) != api_keys:
    raise ValueError("API-smoke PASS schema is not exact")
if api["format"] != "vokra-speecht5-api-smoke-v1" or api["status"] != "PASS" or api["publication"] != "NO_UPLOAD" or api["upload"] != "NOT_PERFORMED":
    raise ValueError("API-smoke status/publication is not the authenticated no-upload contract")
if api["vokra_clean"] is not True or api["vokra_head"] != expected_head or not isinstance(api["vokra_root"], str) or not api["vokra_root"] or not pathlib.Path(api["vokra_root"]).is_absolute() or not isinstance(api["project_dir"], str) or not api["project_dir"] or not pathlib.Path(api["project_dir"]).is_absolute() or api["preflight_gate"] != "PASS":
    raise ValueError("API-smoke evidence is not bound to the current clean Vokra HEAD")
if owner.keys() != {"decision", "digest", "manifest_sha256", "schema", "scope_sha256", "signer"} or owner["decision"] != "APPROVED" or owner["schema"] != "vokra-speecht5-owner-approval-v1":
    raise ValueError("owner/license approval schema is not exact")
if api["approval_evidence_sha256"] != owner_sha or api["approval_scope_sha256"] != owner["scope_sha256"] or owner["digest"] != owner["scope_sha256"] or owner["manifest_sha256"] != manifest_sha or api["approval_signer"] != owner["signer"]:
    raise ValueError("API-smoke evidence is not bound to signed owner/license approval")
for key in ("input_sha256", "output_sha256", "call_checkpoint_sha256", "project_sha256", "lock_sha256", "package_rows_sha256", "package_sha256", "preflight_gate_sha256", "preflight_manifest_sha256", "approval_evidence_sha256", "approval_scope_sha256"):
    if not HEX64.fullmatch(str(api[key])):
        raise ValueError(f"API-smoke {key} is not lowercase SHA-256")
for key, expected in (("project_sha256", project_sha), ("lock_sha256", lock_sha), ("package_rows_sha256", rows_sha), ("package_sha256", "d98947914bfc0a204a1baf1d71890da6c00f88a816cca621df4dfffc6fd999e4"), ("preflight_gate_sha256", gate_sha), ("preflight_manifest_sha256", manifest_sha), ("conversion_source_sha256", source_sha)):
    if api[key] != expected:
        raise ValueError(f"API-smoke {key} differs from this checkout")
if api["upstream_hf"] != "microsoft/speecht5_tts" or api["upstream_revision"] != "30fcde30f19b87502b8435427b5f5068e401d5f6" or api["revision_sha256"] != digest_bytes(api["upstream_revision"].encode()):
    raise ValueError("API-smoke upstream identity drifted")
if api["previous_isolated_transformers_pin"] != "transformers==5.5.0" or api["reference_package"] != "transformers==5.10.4" or api["reference_implementation"] != "transformers.models.speecht5.modeling_speecht5.SpeechT5ForTextToSpeech.generate_speech":
    raise ValueError("API-smoke Transformers route identity drifted")
if api["transformers_security_advisory"] != "GHSA-xrqw-3rrv-vx5w" or api["transformers_security_patched_minimum"] != "5.10.0":
    raise ValueError("API-smoke Transformers security identity drifted")
if api["environment"] != {"platform": api["environment"].get("platform"), "python": "3.12.14", "torch": "2.4.1+cpu", "transformers": "5.10.4"} or not isinstance(api["environment"]["platform"], str) or not api["environment"]["platform"].startswith("Linux-"):
    raise ValueError("API-smoke runtime identity drifted")
safe_contract = {"file": "model.safetensors", "format": "safetensors", "pickle_fallback": "DISABLED", "sha256": safe_sha, "use_safetensors": True}
if api["weight_loading"] != safe_contract or api["float8_import_compat"] not in {"native", "shimmed"}:
    raise ValueError("API-smoke safe-tensor loading contract drifted")
if api["call_checkpoint_sha256"] != digest_canonical(api["call"]):
    raise ValueError("API-smoke call identity hash is inconsistent")
call = api["call"]
if call["input_sha256"] != api["input_sha256"] or api["frames"] != 16 or api["mel_bins"] != 80 or api["output_count"] != api["frames"] * api["mel_bins"]:
    raise ValueError("API-smoke input/output shape identity drifted")
if set(call) != {"api", "checkpoint_sha256", "conversion_source_sha256", "input_sha256", "kwargs", "weight_loading"} or call["api"] != api["reference_implementation"] or call["checkpoint_sha256"] != safe_sha or call["conversion_source_sha256"] != source_sha or call["weight_loading"] != safe_contract:
    raise ValueError("API-smoke call/load identity drifted")
if not isinstance(call["kwargs"], dict) or set(call["kwargs"]) != {"attention_mask", "finegrained_fp8", "float8_import_compat", "maxlenratio", "minlenratio", "output_cross_attentions", "quantization", "return_output_lengths", "speaker_embeddings", "threshold", "vocoder"}:
    raise ValueError("API-smoke generate_speech kwargs schema drifted")
if call["kwargs"] != {"attention_mask": "input_attention_mask", "finegrained_fp8": "not_used", "float8_import_compat": api["float8_import_compat"], "maxlenratio": 20.0, "minlenratio": 0.0, "output_cross_attentions": False, "quantization": "disabled", "return_output_lengths": True, "speaker_embeddings": "input_speaker_embeddings", "threshold": 0.5, "vocoder": None}:
    raise ValueError("API-smoke generate_speech kwargs drifted")
expected_files = {
    "added_tokens.json": {"bytes": 40, "sha256": "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad"},
    "config.json": {"bytes": 2062, "sha256": "2caf62dde93699a90cfc35ff2a8de27b02b479a0c98881cbc55f9682cc43e258"},
    "model.safetensors": {"sha256": safe_sha},
    "pytorch_model.bin": {"bytes": 585476837, "sha256": source_sha},
    "special_tokens_map.json": {"bytes": 234, "sha256": "2a098b61fe8ec4cfd7674832ca00b4268c07569743a4ad15c8164e8f60ebf981"},
    "spm_char.model": {"bytes": 238473, "sha256": "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560"},
    "tokenizer_config.json": {"bytes": 232, "sha256": "d589430c619db2d95ff0fa757a187b55ef5ea44eff7fb08a6fbf0e78e32a6247"},
}
if api["checkpoint_files"] != expected_files:
    raise ValueError("API-smoke checkpoint file identity drifted")
if api["vocoder"] != {"pytorch_model_sha256": "b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b", "repo": "microsoft/speecht5_hifigan", "revision": "bb6f429406e86a9992357a972c0698b22043307d"}:
    raise ValueError("API-smoke vocoder identity drifted")
reference_fields = {
    "upstream_hf": "microsoft/speecht5_tts", "upstream_revision": api["upstream_revision"],
    "previous_isolated_transformers_pin": "transformers==5.5.0", "reference_package": "transformers==5.10.4",
    "reference_implementation": api["reference_implementation"], "transformers_compatibility_status": "AUTHENTICATED_API_SMOKE",
    "transformers_security_advisory": "GHSA-xrqw-3rrv-vx5w", "transformers_security_patched_minimum": "5.10.0",
    "transformers_version": "5.10.4",
}
if any(reference.get(key) != value for key, value in reference_fields.items()) or reference.get("float8_import_compat") != api["float8_import_compat"] or reference.get("weight_loading") != safe_contract:
    raise ValueError("SpeechT5 reference identity is not bound to authenticated API smoke")
provenance = reference.get("provenance")
if not isinstance(provenance, dict) or provenance.get("conversion_source", {}).get("sha256") != source_sha or provenance.get("loaded_weight", {}).get("sha256") != safe_sha:
    raise ValueError("SpeechT5 reference weight provenance is not bound to API smoke")
PY
  then
    die "SpeechT5 API-smoke evidence failed strict authentication"
    return 2
  fi
  api_sha_after="$(sha256_file "$api_smoke")"
  [[ "$api_sha_after" == "$supplied_sha" ]] || { die "API-smoke evidence changed during validation"; return 2; }
}

verify_reference_hash() {
  local reference_json="$1" artifact="$2" filename field expected actual field_lines matches
  filename="$(basename "$artifact")"
  case "$filename" in
    tokens.u32) field=tokens_sha256 ;;
    speaker.f32) field=speaker_sha256 ;;
    before_postnet.f32) field=before_postnet_sha256 ;;
    after_postnet.f32) field=after_postnet_sha256 ;;
    *) die "no authenticated reference hash field for $filename" ;;
  esac
  field_lines="$(grep -Ec "^[[:space:]]+\"$field\": " "$reference_json" || true)"
  [[ "$field_lines" == 1 ]] || { die "reference hash field missing or duplicated: $field"; return 2; }
  matches="$(grep -Ec "^[[:space:]]+\"$field\": \"[0-9a-f]{64}\",?$" "$reference_json" || true)"
  [[ "$matches" == 1 ]] || { die "reference hash field is malformed: $field"; return 2; }
  expected="$(grep -E "^[[:space:]]+\"$field\": " "$reference_json" | tr -cd '0-9a-f' | tail -c 64)"
  actual="$(sha256_file "$artifact")"
  [[ "$actual" == "$expected" ]] || { die "reference artifact hash mismatch: $artifact"; return 2; }
}

verify_reference_manifest_digest() {
  local reference_json="$1" expected="$2" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "reference SHA-256 must be 64 lowercase hex characters"
  require_file "SpeechT5 reference manifest" "$reference_json"
  actual="$(sha256_file "$reference_json")"
  [[ "$actual" == "$expected" ]] || die "reference.json SHA-256 mismatch"
}

verify_reference_scalars() {
  local directory="$1" reference_json="$1/reference.json"
  local text_lines json_text text_file frames_lines json_frames file_frames
  local steps_lines json_steps file_steps

  text_lines="$(grep -Ec '^[[:space:]]+"text": "[^"\\]*(\\\\.[^"\\\\]*)*",?$' "$reference_json" || true)"
  [[ "$text_lines" == 1 ]] || { die "reference text field missing or duplicated"; return 2; }
  if ! json_text="$(json_scalar "$reference_json" text 2>/dev/null)"; then
    die "reference text field is not valid JSON"
    return 2
  fi
  [[ -n "$json_text" ]] || { die "reference text field is empty or malformed"; return 2; }
  text_file="$(<"$directory/text.txt")"
  [[ "$(wc -l < "$directory/text.txt" | tr -d '[:space:]')" == 1 ]] \
    || { die "reference text.txt must contain exactly one newline-terminated line"; return 2; }
  [[ "$text_file" == "$json_text" ]] \
    || { die "reference text.txt does not match reference.json text"; return 2; }

  frames_lines="$(grep -Ec '^[[:space:]]+"frames": [0-9]+,?$' "$reference_json" || true)"
  [[ "$frames_lines" == 1 ]] || { die "reference frames field missing or duplicated"; return 2; }
  if ! json_frames="$(json_scalar "$reference_json" frames 2>/dev/null)"; then
    die "reference frames field is not valid JSON"
    return 2
  fi
  file_frames="$(<"$directory/frames.txt")"
  [[ "$file_frames" =~ ^[0-9]+$ && "$file_frames" == "$json_frames" ]] \
    || { die "reference frames.txt does not match reference.json frames"; return 2; }

  steps_lines="$(grep -Ec '^[[:space:]]+"decoder_steps": [0-9]+,?$' "$reference_json" || true)"
  [[ "$steps_lines" == 1 ]] || { die "reference decoder_steps field missing or duplicated"; return 2; }
  if ! json_steps="$(json_scalar "$reference_json" decoder_steps 2>/dev/null)"; then
    die "reference decoder_steps field is not valid JSON"
    return 2
  fi
  file_steps="$(<"$directory/decoder_steps.txt")"
  [[ "$file_steps" =~ ^[0-9]+$ && "$file_steps" == "$json_steps" ]] \
    || { die "reference decoder_steps.txt does not match reference.json decoder_steps"; return 2; }
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" test_count named_line_count interleaved_line_count completion_line_count standalone_ok_count
  local result_count total_result_count failed_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_line_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  interleaved_line_count="$(grep -Ec "^test ${test_name} \.\.\. SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu " "$log_path" || true)"
  completion_line_count=$((test_count + interleaved_line_count))
  standalone_ok_count="$(grep -Ec '^ok$' "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  failed_count="$(grep -Ec 'FAILED' "$log_path" || true)"
  [[ "$failed_count" == 0 ]] || { die "named test log contains FAILED or a failed Cargo result"; return 2; }
  [[ "$test_count" == 0 || "$test_count" == 1 ]] || { die "expected at most one same-line named test success, got $test_count"; return 2; }
  [[ "$named_line_count" == 1 ]] || { die "expected exactly one total named test line, got $named_line_count"; return 2; }
  [[ "$completion_line_count" == 1 ]] || { die "named test completion line is malformed"; return 2; }
  if [[ "$test_count" == 1 ]]; then
    [[ "$standalone_ok_count" == 0 ]] || { die "same-line named success was followed by standalone ok"; return 2; }
  else
    [[ "$standalone_ok_count" == 1 ]] || { die "expected exactly one standalone ok for interleaved named test output"; return 2; }
  fi
  [[ "$result_count" == 1 ]] || { die "expected exactly one Cargo result with 1 passed/0 failed/0 ignored"; return 2; }
  [[ "$total_result_count" == 1 ]] || { die "expected exactly one total Cargo result line, got $total_result_count"; return 2; }
}

require_exact_parity_sentinels() {
  local log_path="$1" test_name="$2" cpu_count prefixed_cpu_count metal_count
  local cpu_pattern='^SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames=[0-9]+ decoder_steps=[0-9]+ before_max_abs=[-+0-9.eE]+ before_index=[0-9]+ after_max_abs=[-+0-9.eE]+ after_index=[0-9]+ bound=[-+0-9.eE]+ verdict=PASS$'
  local metal_pattern='^SPEECHT5_TTS_OFFICIAL_PARITY backend=metal frames=[0-9]+ decoder_steps=[0-9]+ before_max_abs=[-+0-9.eE]+ before_index=[0-9]+ after_max_abs=[-+0-9.eE]+ after_index=[0-9]+ cpu_max_abs=[-+0-9.eE]+ bound=[-+0-9.eE]+ verdict=PASS$'
  local prefixed_cpu_pattern="^test ${test_name} \.\.\. ${cpu_pattern#^}"
  cpu_count="$(grep -Ec "$cpu_pattern" "$log_path" || true)"
  prefixed_cpu_count="$(grep -Ec "$prefixed_cpu_pattern" "$log_path" || true)"
  metal_count="$(grep -Ec "$metal_pattern" "$log_path" || true)"
  [[ $((cpu_count + prefixed_cpu_count)) == 1 && "$metal_count" == 1 ]] \
    || die "expected exactly one CPU sentinel (standalone or named-line) and one standalone Metal sentinel"
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "remote Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "remote Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  if (( memory_bytes < MIN_MEMORY_BYTES )); then
    die "physical memory $memory_bytes bytes is below the 32-GB remote-worker guard"
  fi
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_disk_kib < MIN_FREE_DISK_KIB )); then
    die "free disk $free_disk_kib KiB is below the 10-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep plutil sysctl sw_vers \
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPDisplaysDataType
  } > "$output"
}

run_self_test() (
  # shellcheck disable=SC2016
  grep -Fq 'require_absent_evidence_dir "$evidence_dir" "$gguf" "$reference" "$approval"' "$0" || return 1
  VOKRA_SPEECHT5_SELF_TEST=1
  local temporary script_path expected_head project_sha lock_sha gate_sha manifest_sha api_sha tampered_sha duplicate_sha
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-speecht5-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/speaker.f32"
  printf '{}\n' > "$temporary/approval.json"
  require_absent_evidence_dir "$temporary/new-evidence" "$temporary/speaker.f32" "$temporary/approval.json" || die "absent evidence path self-test failed"
  mkdir "$temporary/empty-evidence"
  if require_absent_evidence_dir "$temporary/empty-evidence" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "existing empty evidence was accepted"; fi
  rmdir "$temporary/empty-evidence"
  ln -s "$temporary/missing-evidence" "$temporary/link-evidence"
  if require_absent_evidence_dir "$temporary/link-evidence" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "evidence symlink was accepted"; fi
  rm "$temporary/link-evidence"
  mkdir -p "$temporary/real-parent/child"
  ln -s "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_evidence_dir "$temporary/link-parent/child/new-evidence" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "intermediate evidence symlink was accepted"; fi
  rm -rf "$temporary/real-parent" "$temporary/link-parent"
  if require_absent_evidence_dir "$VOKRA_ROOT/speecht5-apple-self-test" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "checkout overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/approval.json/child" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "approval overlap was accepted"; fi
  if require_absent_evidence_dir "$temporary/speaker.f32/child" "$temporary/speaker.f32" "$temporary/approval.json" >/dev/null 2>&1; then die "input overlap was accepted"; fi
  [[ "$(sha256_file "$temporary/speaker.f32")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  printf '  "speaker_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",\n' > "$temporary/reference.json"
  local reference_digest
  reference_digest="$(sha256_file "$temporary/reference.json")"
  verify_reference_manifest_digest "$temporary/reference.json" "$reference_digest"
  if verify_reference_manifest_digest "$temporary/reference.json" "0000000000000000000000000000000000000000000000000000000000000000" >/dev/null 2>&1; then
    die "reference manifest mismatch self-test failed"
  fi
  if verify_reference_manifest_digest "$temporary/missing.json" "$reference_digest" >/dev/null 2>&1; then
    die "missing reference manifest self-test failed"
  fi
  verify_reference_hash "$temporary/reference.json" "$temporary/speaker.f32"
  {
    printf '  "speaker_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",\n'
    printf '  "speaker_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "tampered": true,\n'
  } > "$temporary/reference.json"
  if verify_reference_hash "$temporary/reference.json" "$temporary/speaker.f32" >/dev/null 2>&1; then
    die "duplicate reference hash self-test failed"
  fi
  printf '  "speaker_sha256": "0000000000000000000000000000000000000000000000000000000000000000",\n' > "$temporary/reference.json"
  if verify_reference_hash "$temporary/reference.json" "$temporary/speaker.f32" >/dev/null 2>&1; then
    die "reference artifact tamper self-test failed"
  fi
  cat > "$temporary/api-smoke-identity.json" <<EOF
  "upstream_hf": "microsoft/speecht5_tts",
  "upstream_revision": "$TTS_REVISION",
  "previous_isolated_transformers_pin": "$PREVIOUS_ISOLATED_TRANSFORMERS_PIN",
  "reference_package": "$ISOLATED_TRANSFORMERS_PIN",
  "reference_implementation": "$REFERENCE_IMPLEMENTATION",
  "transformers_compatibility_status": "$TRANSFORMERS_COMPATIBILITY_STATUS",
  "transformers_security_advisory": "$TRANSFORMERS_SECURITY_ADVISORY",
  "transformers_security_patched_minimum": "$TRANSFORMERS_SECURITY_PATCHED_MINIMUM",
  "transformers_version": "5.10.4"
EOF
  require_reference_identity "$temporary/api-smoke-identity.json"
  sed "s/$TRANSFORMERS_COMPATIBILITY_STATUS/BLOCKED_UNVERIFIED_API_SMOKE/" \
    "$temporary/api-smoke-identity.json" > "$temporary/blocked-api-smoke-identity.json"
  if require_reference_identity "$temporary/blocked-api-smoke-identity.json" >/dev/null 2>&1; then
    die "blocked API-smoke status was accepted"
  fi
  cat >> "$temporary/api-smoke-identity.json" <<EOF
  "transformers_compatibility_status": "$TRANSFORMERS_COMPATIBILITY_STATUS"
EOF
  if require_reference_identity "$temporary/api-smoke-identity.json" >/dev/null 2>&1; then
    die "duplicate API-smoke status was accepted"
  fi
  expected_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  project_sha="$(sha256_file "$PARITY_PROJECT/pyproject.toml")"
  lock_sha="$(sha256_file "$PARITY_PROJECT/uv.lock")"
  gate_sha="$(sha256_file "$PREFLIGHT_GATE")"
  manifest_sha="$(sha256_file "$PREFLIGHT_MANIFEST")"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$temporary" "$expected_head" "$project_sha" "$lock_sha" "$PACKAGE_ROWS_SHA256" "$gate_sha" "$manifest_sha" \
    "$PARITY_PROJECT" "$SOURCE_WEIGHT_SHA256" "$SAFE_TENSOR_WEIGHT_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys

root, head, project_sha, lock_sha, rows_sha, gate_sha, manifest_sha, project_dir, source_sha, safe_sha = sys.argv[1:]
root = pathlib.Path(root)

def digest(value):
    return hashlib.sha256(value).hexdigest()

def write(name, value):
    path = root / name
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path

owner = {
    "decision": "APPROVED", "digest": "b" * 64,
    "manifest_sha256": manifest_sha, "schema": "vokra-speecht5-owner-approval-v1",
    "scope_sha256": "b" * 64, "signer": "yousan",
}
owner_path = write("owner.json", owner)
owner_sha = digest(owner_path.read_bytes())
safe_contract = {"file": "model.safetensors", "format": "safetensors", "pickle_fallback": "DISABLED", "sha256": safe_sha, "use_safetensors": True}
kwargs = {"attention_mask": "input_attention_mask", "finegrained_fp8": "not_used", "float8_import_compat": "shimmed", "maxlenratio": 20.0, "minlenratio": 0.0, "output_cross_attentions": False, "quantization": "disabled", "return_output_lengths": True, "speaker_embeddings": "input_speaker_embeddings", "threshold": 0.5, "vocoder": None}
call = {"api": "transformers.models.speecht5.modeling_speecht5.SpeechT5ForTextToSpeech.generate_speech", "checkpoint_sha256": safe_sha, "conversion_source_sha256": source_sha, "input_sha256": "d" * 64, "kwargs": kwargs, "weight_loading": safe_contract}
api = {
    "call": call, "call_checkpoint_sha256": digest(json.dumps(call, sort_keys=True, separators=(",", ":")).encode()),
    "checkpoint_files": {
        "added_tokens.json": {"bytes": 40, "sha256": "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad"},
        "config.json": {"bytes": 2062, "sha256": "2caf62dde93699a90cfc35ff2a8de27b02b479a0c98881cbc55f9682cc43e258"},
        "model.safetensors": {"sha256": safe_sha}, "pytorch_model.bin": {"bytes": 585476837, "sha256": source_sha},
        "special_tokens_map.json": {"bytes": 234, "sha256": "2a098b61fe8ec4cfd7674832ca00b4268c07569743a4ad15c8164e8f60ebf981"},
        "spm_char.model": {"bytes": 238473, "sha256": "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560"},
        "tokenizer_config.json": {"bytes": 232, "sha256": "d589430c619db2d95ff0fa757a187b55ef5ea44eff7fb08a6fbf0e78e32a6247"},
    },
    "environment": {"platform": "Linux-test", "python": "3.12.14", "torch": "2.4.1+cpu", "transformers": "5.10.4"},
    "format": "vokra-speecht5-api-smoke-v1", "frames": 16, "input_sha256": "d" * 64, "lock_sha256": lock_sha,
    "mel_bins": 80, "output_count": 1280, "output_sha256": "e" * 64, "package_rows_sha256": rows_sha,
    "package_sha256": "d98947914bfc0a204a1baf1d71890da6c00f88a816cca621df4dfffc6fd999e4", "previous_isolated_transformers_pin": "transformers==5.5.0", "project_sha256": project_sha,
    "publication": "NO_UPLOAD", "reference_implementation": call["api"], "reference_package": "transformers==5.10.4",
    "revision_sha256": digest("30fcde30f19b87502b8435427b5f5068e401d5f6".encode()), "status": "PASS",
    "transformers_security_advisory": "GHSA-xrqw-3rrv-vx5w", "transformers_security_patched_minimum": "5.10.0",
    "upstream_hf": "microsoft/speecht5_tts", "upstream_revision": "30fcde30f19b87502b8435427b5f5068e401d5f6",
    "upload": "NOT_PERFORMED", "vocoder": {"pytorch_model_sha256": "b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b", "repo": "microsoft/speecht5_hifigan", "revision": "bb6f429406e86a9992357a972c0698b22043307d"},
    "vokra_head": head, "vokra_root": str(root), "vokra_clean": True, "approval_evidence_sha256": owner_sha,
    "approval_scope_sha256": owner["scope_sha256"], "approval_signer": owner["signer"], "project_dir": project_dir,
    "preflight_gate": "PASS", "preflight_gate_sha256": gate_sha, "preflight_manifest_sha256": manifest_sha,
    "float8_import_compat": "shimmed", "conversion_source_sha256": source_sha, "weight_loading": safe_contract,
}
write("api.json", api)
reference = {
    "float8_import_compat": "shimmed", "format": "vokra-speecht5-tts-reference-v1",
    "previous_isolated_transformers_pin": "transformers==5.5.0", "reference_implementation": call["api"],
    "reference_package": "transformers==5.10.4", "transformers_compatibility_status": "AUTHENTICATED_API_SMOKE",
    "transformers_security_advisory": "GHSA-xrqw-3rrv-vx5w", "transformers_security_patched_minimum": "5.10.0",
    "transformers_version": "5.10.4", "upstream_hf": "microsoft/speecht5_tts", "upstream_revision": api["upstream_revision"],
    "weight_loading": safe_contract, "provenance": {"conversion_source": {"sha256": source_sha}, "loaded_weight": {"sha256": safe_sha}},
}
write("reference.json", reference)
PY
  api_sha="$(sha256_file "$temporary/api.json")"
  validate_api_smoke_evidence "$temporary/api.json" "$temporary/reference.json" "$temporary/owner.json" "$expected_head" "$api_sha"
  sed 's/"status": "PASS"/"status": "FAIL"/' "$temporary/api.json" > "$temporary/tampered-api.json"
  tampered_sha="$(sha256_file "$temporary/tampered-api.json")"
  if validate_api_smoke_evidence "$temporary/tampered-api.json" "$temporary/reference.json" "$temporary/owner.json" "$expected_head" "$tampered_sha" >/dev/null 2>&1; then
    die "tampered API-smoke status was accepted"
  fi
  awk '{ print; if (index($0, "\"status\": \"PASS\",")) print "  \"status\": \"PASS\","; }' "$temporary/api.json" > "$temporary/duplicate-api.json"
  duplicate_sha="$(sha256_file "$temporary/duplicate-api.json")"
  if validate_api_smoke_evidence "$temporary/duplicate-api.json" "$temporary/reference.json" "$temporary/owner.json" "$expected_head" "$duplicate_sha" >/dev/null 2>&1; then
    die "duplicate API-smoke key was accepted"
  fi
  sed 's/AUTHENTICATED_API_SMOKE/BLOCKED_UNVERIFIED_API_SMOKE/' "$temporary/reference.json" > "$temporary/blocked-reference.json"
  if validate_api_smoke_evidence "$temporary/api.json" "$temporary/blocked-reference.json" "$temporary/owner.json" "$expected_head" "$api_sha" >/dev/null 2>&1; then
    die "reference/API-smoke status mismatch was accepted"
  fi
  mkdir "$temporary/scalars"
  printf 'Hello, SpeechT5!\n' > "$temporary/scalars/text.txt"
  printf '2\n' > "$temporary/scalars/frames.txt"
  printf '1\n' > "$temporary/scalars/decoder_steps.txt"
  cat > "$temporary/scalars/reference.json" <<'EOF'
{
  "text": "Hello, SpeechT5!",
  "decoder_steps": 1,
  "frames": 2
}
EOF
  verify_reference_scalars "$temporary/scalars"
  printf '3\n' > "$temporary/scalars/frames.txt"
  if verify_reference_scalars "$temporary/scalars" >/dev/null 2>&1; then
    die "reference scalar tamper self-test failed"
  fi
  cat > "$temporary/scalars/reference.json" <<'EOF'
{
  "text": "Hello, SpeechT5!",
  "decoder_steps": 1,
  "frames": 2,
}
EOF
  if verify_reference_scalars "$temporary/scalars" >/dev/null 2>&1; then
    die "invalid JSON scalar self-test failed"
  fi
  local cargo_log
  cargo_log="$temporary/cargo.log"
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; unexpected' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "malformed Cargo result suffix self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "duplicate Cargo result self-test failed"
  fi
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... FAILED' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "failed named test self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "duplicate named test self-test failed"
  fi
  local sentinel_log sentinel_cpu sentinel_metal
  sentinel_log="$temporary/sentinels.log"
  sentinel_cpu='SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames=2 decoder_steps=1 before_max_abs=1.000000000e-3 before_index=0 after_max_abs=2.000000000e-3 after_index=1 bound=1.000000000e-2 verdict=PASS'
  sentinel_metal='SPEECHT5_TTS_OFFICIAL_PARITY backend=metal frames=2 decoder_steps=1 before_max_abs=1.000000000e-3 before_index=0 after_max_abs=2.000000000e-3 after_index=1 cpu_max_abs=3.000000000e-3 bound=1.000000000e-2 verdict=PASS'
  printf '%s\n%s\n%s\n%s\n' \
    "test released_cpu_mel_matches_official_transformers ... $sentinel_cpu" \
    "$sentinel_metal" 'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  require_exact_parity_sentinels "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n%s\n' \
    "$sentinel_cpu" "test released_cpu_mel_matches_official_transformers ... $sentinel_cpu" \
    "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "mixed standalone and named-line CPU sentinel self-test failed"
  fi
  printf '%s\n%s\n' \
    "test another_test ... $sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "wrong test name sentinel self-test failed"
  fi
  printf '%s\n%s\n' \
    "test released_cpu_mel_matches_official_transformers ... prefix$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "generic prefix sentinel self-test failed"
  fi
  printf '%s\n%s\n' \
    "test released_cpu_mel_matches_official_transformers ... $sentinel_cpu suffix" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "generic suffix sentinel self-test failed"
  fi
  printf '%s\n%s\n' \
    "$sentinel_cpu" "test released_cpu_mel_matches_official_transformers ... $sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "named-line Metal sentinel self-test failed"
  fi
  printf '%s\n%s\n%s\n%s\n%s\n' \
    "test released_cpu_mel_matches_official_transformers ... $sentinel_cpu" \
    "$sentinel_metal" 'ok' 'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "duplicate standalone ok self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' 'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "same-line and standalone completion self-test failed"
  fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ...' 'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then
    die "malformed interleaved named test self-test failed"
  fi
  printf '%s\n%s\n' "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n%s\n' "$sentinel_cpu" "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then die "duplicate sentinel self-test failed"; fi
  printf 'prefix%s\n%s\n' "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then die "prefix sentinel self-test failed"; fi
  printf '%s suffix\n%s\n' "$sentinel_cpu" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then die "suffix sentinel self-test failed"; fi
  printf '%s\n%s\n' "${sentinel_cpu/verdict=PASS/verdict=FAIL}" "$sentinel_metal" > "$sentinel_log"
  if require_exact_parity_sentinels "$sentinel_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then die "FAIL sentinel self-test failed"; fi
  require_empty_directory "$temporary/evidence"
  script_path="${BASH_SOURCE[0]}"
  grep -F "$PUBLIC_GGUF_SHA256" "$script_path" >/dev/null \
    || die "public GGUF SHA contract is missing"
  grep -F "SPEECHT5_TTS_OFFICIAL_PARITY backend=metal" "$script_path" >/dev/null \
    || die "Metal PASS marker contract is missing"
  for token in "$PREVIOUS_ISOLATED_TRANSFORMERS_PIN" "$ISOLATED_TRANSFORMERS_PIN" "$TRANSFORMERS_SECURITY_ADVISORY" "$TRANSFORMERS_SECURITY_PATCHED_MINIMUM" 'BLOCKED_UNVERIFIED_API_SMOKE'; do
    grep -F "$token" "$PARITY_PROJECT/pyproject.toml" >/dev/null || die "Transformers provenance is missing: $token"
  done
  for token in "$TRANSFORMERS_COMPATIBILITY_STATUS" "$REFERENCE_IMPLEMENTATION" 'require_reference_identity'; do
    grep -F "$token" "$script_path" >/dev/null || die "authenticated API-smoke contract is missing: $token"
  done
  for token in '--api-smoke-evidence' '--api-smoke-sha256' 'validate_api_smoke_evidence' 'vokra-speecht5-api-smoke-v1' 'NO_UPLOAD' 'NOT_PERFORMED' 'model.safetensors'; do
    grep -F -- "$token" "$script_path" >/dev/null || die "API-smoke evidence contract is missing: $token"
  done
  if "$script_path" --self-test --api-smoke-evidence "$temporary/api.json" >/dev/null 2>&1 || \
    "$script_path" --api-smoke-evidence "$temporary/api.json" --api-smoke-evidence "$temporary/api.json" >/dev/null 2>&1 || \
    "$script_path" --api-smoke-evidence "$temporary/api.json" --api-smoke-sha256 "$api_sha" --api-smoke-sha256 "$api_sha" >/dev/null 2>&1; then
    die "self-test accepted malformed or duplicate API-smoke CLI options"
  fi
  log "self-test PASS"
)

main() {
  local gguf='' reference='' reference_digest='' approval='' api_smoke='' api_smoke_sha='' evidence_dir='' self_test=0 gguf_sha expected_head
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$gguf" ]] || { usage; return 2; }
        gguf="$2"
        shift 2
        ;;
      --reference)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$reference" ]] || { usage; return 2; }
        reference="$2"
        shift 2
        ;;
      --reference-sha256)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$reference_digest" ]] || { usage; return 2; }
        reference_digest="$2"
        shift 2
        ;;
      --approval-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ -z "$approval" ]] || { usage; return 2; }
        approval="$2"
        shift 2
        ;;
      --api-smoke-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ -z "$api_smoke" ]] || { usage; return 2; }
        api_smoke="$2"
        shift 2
        ;;
      --api-smoke-sha256)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        [[ -z "$api_smoke_sha" ]] || { usage; return 2; }
        api_smoke_sha="$2"
        shift 2
        ;;
      --evidence-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$evidence_dir" ]] || { usage; return 2; }
        evidence_dir="$2"
        shift 2
        ;;
      --self-test)
        self_test=1
        shift
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        usage
        die "unknown argument $1"
        ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$reference$reference_digest$approval$api_smoke$api_smoke_sha$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$reference" && -n "$reference_digest" && -n "$approval" && -n "$api_smoke" && -n "$api_smoke_sha" && -n "$evidence_dir" ]] \
    || { usage; die "--gguf, --reference, --reference-sha256, --approval-evidence, --api-smoke-evidence, --api-smoke-sha256 and --evidence-dir are required"; }
  license_preflight "$approval"
  [[ "$reference_digest" =~ ^[0-9a-f]{64}$ ]] \
    || die "--reference-sha256 must be 64 lowercase hex characters"

  require_remote_apple_host
  require_tooling
  require_file "public SpeechT5 GGUF" "$gguf"
  verify_reference_manifest_digest "$reference/reference.json" "$reference_digest"
  require_reference "$reference"
  expected_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  validate_api_smoke_evidence "$api_smoke" "$reference/reference.json" "$approval" "$expected_head" "$api_smoke_sha"
  gguf_sha="$(sha256_file "$gguf")"
  [[ "$gguf_sha" == "$PUBLIC_GGUF_SHA256" ]] \
    || die "GGUF SHA-256 $gguf_sha != exact public artifact $PUBLIC_GGUF_SHA256"
  require_absent_evidence_dir "$evidence_dir" "$gguf" "$reference" "$approval" "$api_smoke"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  {
    echo "public_gguf_sha256=$gguf_sha"
    echo "reference_manifest_sha256=$reference_digest"
    echo "api_smoke_evidence_sha256=$(sha256_file "$api_smoke")"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact public SpeechT5 CPU/Metal parity on remote Apple Silicon"
  env \
    VOKRA_SPEECHT5_TTS_GGUF="$gguf" \
    VOKRA_SPEECHT5_TTS_REFERENCE_DIR="$reference" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --features metal --test parity_speecht5_tts_real \
      released_cpu_mel_matches_official_transformers -- --exact --nocapture \
      2>&1 | tee "$evidence_dir/parity.log"

  require_one_named_test_passed "$evidence_dir/parity.log" \
    released_cpu_mel_matches_official_transformers
  require_exact_parity_sentinels "$evidence_dir/parity.log" released_cpu_mel_matches_official_transformers

  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_gguf_sha256=$gguf_sha"
    echo "speecht5_cpu_vs_official=PASS"
    echo "speecht5_metal_vs_official=PASS"
    echo "speecht5_metal_vs_cpu=PASS"
    echo "bound=0.01"
    echo "api_smoke_evidence_sha256=$(sha256_file "$api_smoke")"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged data or destroy the remote worker"
}

main "$@"
