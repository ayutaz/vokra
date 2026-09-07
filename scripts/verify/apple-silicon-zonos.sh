#!/usr/bin/env bash
# Apple-only Zonos validation. Inputs and approval are staged by VAST.
# This worker performs no acquisition, conversion, upload, or CPU fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_NAME='zonos_real_cpu_codes_and_pcm_boundary'
SOURCE_REVISION='bc40d98e1e1ab54fc65c483be127a90e3c7c0645'
UPSTREAM_REVISION='9d8331fc49cb5ba8aad2bb56cafd809c66598f4e'
PUBLIC_REVISION='b1bf5c56d470eb9097e9b04f9deca364576574ba'
SOURCE_LICENSE_PATH='LICENSE'
SOURCE_LICENSE_SPDX='Apache-2.0'
SOURCE_LICENSE_BYTES=11357
SOURCE_LICENSE_GIT_BLOB_SHA1='7a4a3ea2424c09fbe48d455aed1eaa94d9124835'
SOURCE_LICENSE_SHA256='58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd'
UPSTREAM_MODEL_ID='Zyphra/Zonos-v0.1-transformer'
UPSTREAM_MODEL_CARD_DATA_LICENSE='apache-2.0'

log() { printf '[zonos-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
lower_sha() { printf '%s' "$1" | tr 'A-F' 'a-f'; }
require_sha() { [[ "$1" =~ ^[0-9a-fA-F]{64}$ ]] || die 'SHA-256 must be exactly 64 hexadecimal digits'; }
require_size() { [[ "$1" =~ ^[0-9]+$ && "$1" != 0 ]] || die 'size must be a positive decimal integer'; }

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be exactly 40 lowercase hexadecimal characters'
  [[ -d "$ROOT/.git" ]] || die 'checkout is missing .git'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from expected $expected"
}

claim_evidence_dir() {
  local evidence="$1"
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || die 'Apple evidence directory must be absent'
  mkdir "$evidence" || die 'Apple evidence directory was concurrently claimed or already exists'
}

usage() {
  cat <<'EOF'
usage: apple-silicon-zonos.sh --expected-head HEX40 --approval-evidence FILE \
  --approval-evidence-sha256 SHA --transfer-manifest FILE --transfer-manifest-sha256 SHA \
  --manifest FILE \
  --manifest-sha256 SHA --gguf FILE --gguf-sha256 SHA --gguf-size BYTES \
  --dac-gguf FILE --dac-gguf-sha256 SHA --dac-gguf-size BYTES \
  --conditioning-packet FILE --conditioning-packet-sha256 SHA --conditioning-packet-size BYTES \
  --reference-codes FILE --reference-codes-sha256 SHA --reference-codes-size BYTES \
  --reference-pcm FILE --reference-pcm-sha256 SHA --reference-pcm-size BYTES \
  --native-cpu-log FILE --native-cpu-log-sha256 SHA \
  --evidence-dir ABSENT_DIR
       apple-silicon-zonos.sh --self-test
EOF
}

reject_symlink_ancestors() {
  local path="$1" rest component current
  [[ "$path" == /* ]] || { die "path must be absolute: $path"; return 2; }
  rest="${path#/}"; current=/
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "path contains symlink ancestor: $path"; return 2; }
    current="$current/"
  done
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label must be a nonempty regular non-symlink file"
  reject_symlink_ancestors "$path" || return 2
}

require_input() {
  local label="$1" path="$2" expected_sha="$3" expected_size="$4" actual_size
  require_sha "$expected_sha"; require_size "$expected_size"; require_file "$label" "$path"
  actual_size="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_size" == "$expected_size" ]] || die "$label byte-size mismatch"
  [[ "$(sha256_file "$path")" == "$(lower_sha "$expected_sha")" ]] || die "$label SHA-256 mismatch"
}

validate_json_manifest() {
  local path="$1"
  command -v uv >/dev/null 2>&1 || die 'uv is required for duplicate-key JSON validation'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" <<'PY'
import json, pathlib, sys
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate manifest key: " + key)
        result[key] = value
    return result
try:
    value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=unique)
    required = {"repository", "revision", "resolved_revision", "walk", "complete_recursive", "files"}
    if not isinstance(value, dict) or set(value) != required: raise ValueError("manifest is not the exact VAST server-tree schema")
    if value["repository"] != "vokra/zonos-v0.1-transformer" or value["revision"] != "b1bf5c56d470eb9097e9b04f9deca364576574ba" or value["resolved_revision"] != value["revision"]: raise ValueError("manifest public revision mismatch")
    if value["walk"] != "recursive_file_only" or value["complete_recursive"] is not True or not isinstance(value["files"], list) or not value["files"]: raise ValueError("manifest file tree is incomplete")
except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit("manifest BLOCKED: " + str(error))
PY
}

validate_transfer_manifest() {
  local path="$1" expected_sha="$2" expected_head="$3" approval_sha="$4" manifest_sha="$5" gguf_sha="$6" dac_sha="$7" packet_sha="$8" codes_sha="$9" pcm_sha="${10}" native_sha="${11}"
  require_sha "$expected_sha" || return 2
  require_file 'Zonos transfer manifest' "$path" || return 2
  [[ "$(sha256_file "$path")" == "$(lower_sha "$expected_sha")" ]] || { die 'transfer manifest SHA-256 mismatch'; return 2; }
  awk '
    BEGIN { allowed["schema"]; allowed["expected_head"]; allowed["approval_evidence_sha256"]; allowed["source_license_path"]; allowed["source_license_spdx"]; allowed["source_license_bytes"]; allowed["source_license_git_blob_sha1"]; allowed["source_license_sha256"]; allowed["upstream_model_id"]; allowed["upstream_model_revision"]; allowed["upstream_model_private"]; allowed["upstream_model_gated"]; allowed["upstream_model_disabled"]; allowed["upstream_model_card_data_license"]; allowed["manifest_sha256"]; allowed["gguf_sha256"]; allowed["dac_gguf_sha256"]; allowed["conditioning_packet_sha256"]; allowed["reference_codes_sha256"]; allowed["reference_pcm_sha256"]; allowed["native_cpu_log_sha256"]; allowed["cpu_result"]; allowed["cpu_sentinel_summary"]; allowed["status"]; allowed["metal_status"]; allowed["publication"] }
    index($0, "=") == 0 { bad=1; next }
    { key=substr($0, 1, index($0, "=") - 1); if (!(key in allowed)) bad=1; count[key]++ }
    END { if (bad) exit 2; for (key in allowed) if (count[key] != 1) exit 3 }
  ' "$path" || { die 'transfer manifest schema is not exact'; return 2; }
  grep -Fxq 'schema=zonos-apple-transfer-v1' "$path" || { die 'transfer manifest schema mismatch'; return 2; }
  grep -Fxq "expected_head=$expected_head" "$path" || { die 'transfer manifest expected HEAD mismatch'; return 2; }
  grep -Fxq "approval_evidence_sha256=$approval_sha" "$path" || { die 'transfer manifest approval binding mismatch'; return 2; }
  grep -Fxq "source_license_path=$SOURCE_LICENSE_PATH" "$path" || { die 'transfer manifest source license path mismatch'; return 2; }
  grep -Fxq "source_license_spdx=$SOURCE_LICENSE_SPDX" "$path" || { die 'transfer manifest source license SPDX mismatch'; return 2; }
  grep -Fxq "source_license_bytes=$SOURCE_LICENSE_BYTES" "$path" || { die 'transfer manifest source license byte-size mismatch'; return 2; }
  grep -Fxq "source_license_git_blob_sha1=$SOURCE_LICENSE_GIT_BLOB_SHA1" "$path" || { die 'transfer manifest source license Git blob mismatch'; return 2; }
  grep -Fxq "source_license_sha256=$SOURCE_LICENSE_SHA256" "$path" || { die 'transfer manifest source license SHA-256 mismatch'; return 2; }
  grep -Fxq "upstream_model_id=$UPSTREAM_MODEL_ID" "$path" || { die 'transfer manifest upstream model ID mismatch'; return 2; }
  grep -Fxq "upstream_model_revision=$UPSTREAM_REVISION" "$path" || { die 'transfer manifest upstream model revision mismatch'; return 2; }
  grep -Fxq 'upstream_model_private=false' "$path" || { die 'transfer manifest upstream private flag mismatch'; return 2; }
  grep -Fxq 'upstream_model_gated=false' "$path" || { die 'transfer manifest upstream gated flag mismatch'; return 2; }
  grep -Fxq 'upstream_model_disabled=false' "$path" || { die 'transfer manifest upstream disabled flag mismatch'; return 2; }
  grep -Fxq "upstream_model_card_data_license=$UPSTREAM_MODEL_CARD_DATA_LICENSE" "$path" || { die 'transfer manifest HF cardData license mismatch'; return 2; }
  grep -Fxq "manifest_sha256=$manifest_sha" "$path" || { die 'transfer manifest public manifest mismatch'; return 2; }
  grep -Fxq "gguf_sha256=$gguf_sha" "$path" || { die 'transfer manifest GGUF mismatch'; return 2; }
  grep -Fxq "dac_gguf_sha256=$dac_sha" "$path" || { die 'transfer manifest DAC mismatch'; return 2; }
  grep -Fxq "conditioning_packet_sha256=$packet_sha" "$path" || { die 'transfer manifest conditioning packet mismatch'; return 2; }
  grep -Fxq "reference_codes_sha256=$codes_sha" "$path" || { die 'transfer manifest reference codes mismatch'; return 2; }
  grep -Fxq "reference_pcm_sha256=$pcm_sha" "$path" || { die 'transfer manifest reference PCM mismatch'; return 2; }
  grep -Fxq "native_cpu_log_sha256=$native_sha" "$path" || { die 'transfer manifest CPU log mismatch'; return 2; }
  grep -Fxq 'cpu_result=ONE_PASS' "$path" || { die 'transfer manifest CPU result mismatch'; return 2; }
  grep -Fxq 'cpu_sentinel_summary=ZONOS_CPU_REFERENCE codes=EXACT verdict=MEASURED_NOT_GATED' "$path" || { die 'transfer manifest CPU sentinel summary mismatch'; return 2; }
  grep -Fxq 'status=MEASURED_NOT_GATED' "$path" || { die 'transfer manifest status mismatch'; return 2; }
  grep -Fxq 'metal_status=NOT_RUN' "$path" || { die 'transfer manifest Metal status mismatch'; return 2; }
  grep -Fxq 'publication=NO_UPLOAD' "$path" || { die 'transfer manifest publication mismatch'; return 2; }
}

validate_approval() {
  local approval="$1" project="$ROOT/tools/parity/pyproject.toml" lock="$ROOT/tools/parity/uv.lock"
  require_file 'approval evidence' "$approval"
  [[ -f "$project" && ! -L "$project" && -f "$lock" && ! -L "$lock" ]] || die 'parity project lock is missing or symlinked'
  local project_sha lock_sha
  project_sha="$(sha256_file "$project")"
  lock_sha="$(sha256_file "$lock")"
  command -v uv >/dev/null 2>&1 || die 'uv is required for duplicate-key approval validation'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate approval key: " + key)
        result[key] = value
    return result
try:
    value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=unique)
    keys = {"schema", "decision", "signer", "project_sha256", "lock_sha256", "scope_sha256", "no_upload", "source_repository", "source_revision", "upstream_repository", "upstream_revision", "public_repository", "public_revision", "license_review"}
    if not isinstance(value, dict) or set(value) != keys: raise ValueError("approval schema is not exact")
    if value["schema"] != "zonos-vast-approval-v1" or value["decision"] != "APPROVED" or value["no_upload"] is not True: raise ValueError("approval decision/publication mismatch")
    if not isinstance(value["signer"], str) or not value["signer"].strip() or value["signer"].strip().upper() in {"TODO", "UNRESOLVED", "PENDING", "OWNER_REVIEW_REQUIRED"}: raise ValueError("approval signer unresolved")
    if value["project_sha256"] != sys.argv[2] or value["lock_sha256"] != sys.argv[3]: raise ValueError("approval project/lock mismatch")
    scope = {
        "schema": "zonos-vast-approval-scope-v1",
        "project_sha256": sys.argv[2],
        "lock_sha256": sys.argv[3],
        "source_repository": "https://github.com/Zyphra/Zonos.git",
        "source_revision": "bc40d98e1e1ab54fc65c483be127a90e3c7c0645",
        "upstream_repository": "Zyphra/Zonos-v0.1-transformer",
        "upstream_revision": "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
        "public_repository": "vokra/zonos-v0.1-transformer",
        "public_revision": "b1bf5c56d470eb9097e9b04f9deca364576574ba",
        "no_upload": True,
        "license_review": "AUTHENTICATED_LICENSE_IDENTITY_REQUIRED",
        "license_identity": {
            "source": {
                "repository": "https://github.com/Zyphra/Zonos.git",
                "revision": "bc40d98e1e1ab54fc65c483be127a90e3c7c0645",
                "path": "LICENSE",
                "spdx": "Apache-2.0",
                "bytes": 11357,
                "git_blob_sha1": "7a4a3ea2424c09fbe48d455aed1eaa94d9124835",
                "sha256": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
            },
            "upstream_model": {
                "id": "Zyphra/Zonos-v0.1-transformer",
                "revision": "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
                "private": False,
                "gated": False,
                "disabled": False,
                "card_data_license": "apache-2.0",
            },
        },
    }
    expected = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if value["scope_sha256"] != expected: raise ValueError("approval scope mismatch")
    for key, expected in (("source_repository", scope["source_repository"]), ("source_revision", scope["source_revision"]), ("upstream_repository", scope["upstream_repository"]), ("upstream_revision", scope["upstream_revision"]), ("public_repository", scope["public_repository"]), ("public_revision", scope["public_revision"]), ("license_review", scope["license_review"])):
        if value[key] != expected: raise ValueError("approval fixed identity mismatch: " + key)
except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit("approval BLOCKED: " + str(error))
PY
}

validate_absent_evidence() {
  local evidence="$1" protected parent item candidate root_real
  shift
  local -a suffix=()
  reject_symlink_ancestors "$evidence" || return 2
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || die 'Apple evidence directory must be absent'
  parent="$evidence"
  while [[ ! -e "$parent" ]]; do
    item="${parent##*/}"; [[ -n "$item" ]] || { die 'evidence path has an invalid parent'; return 2; }
    suffix+=("$item"); [[ "$parent" != / ]] || { die 'evidence path parent does not exist'; return 2; }
    parent="${parent%/*}"; [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || { die 'evidence nearest parent is unsafe'; return 2; }
  candidate="$(cd -P "$parent" && pwd)" || { die 'could not resolve evidence parent'; return 2; }
  for (( item = ${#suffix[@]} - 1; item >= 0; item-- )); do candidate="$candidate/${suffix[item]}"; done
  root_real="$(cd -P "$ROOT" && pwd)" || { die 'could not resolve checkout'; return 2; }
  for protected in "$root_real" "$@"; do
    protected="$(cd -P "$(dirname "$protected")" && pwd)/$(basename "$protected")" || die 'could not resolve protected input'
    [[ "$candidate" != "$protected" && "$candidate/" != "$protected/"* && "$protected/" != "$candidate/"* ]] || die 'evidence overlaps a protected input'
  done
}

require_cargo_singleton() {
  local log_file="$1" named result tests
  named="$(grep -Ec "^test ${TEST_NAME} \.\.\. ok$" "$log_file" || true)"; result="$(grep -Ec '^test result:' "$log_file" || true)"; tests="$(grep -Ec '^test ' "$log_file" || true)"
  (( named == 1 && result == 1 && tests - result == 1 )) || { die 'Cargo output is not one exact named test/result'; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" || { die 'Cargo result is not an exact singleton pass'; return 2; }
}

require_sentinel() {
  local log_file="$1" prefix="$2" pattern="$3" label="$4" family exact
  family="$(grep -Ec "^${prefix}" "$log_file" || true)"; exact="$(grep -Ec "^${pattern}$" "$log_file" || true)"
  [[ "$family" == 1 && "$exact" == 1 ]] || { die "$label is not one complete line"; return 2; }
}

self_test() (
  local path="${BASH_SOURCE[0]}" temporary fail=0 sha old_license_gate
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-zonos-apple.XXXXXX")"; temporary="$(cd -P "$temporary" && pwd)"; trap 'rm -rf "$temporary"' EXIT
  printf abc > "$temporary/value"; sha='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'; require_input 'self-test input' "$temporary/value" "$sha" 3
  printf '%s\n' '{"repository":"vokra/zonos-v0.1-transformer","revision":"b1bf5c56d470eb9097e9b04f9deca364576574ba","resolved_revision":"b1bf5c56d470eb9097e9b04f9deca364576574ba","walk":"recursive_file_only","complete_recursive":true,"files":[{"type":"file","path":"x","size":1,"git_blob_sha1":"0000000000000000000000000000000000000000"}]}' > "$temporary/manifest.json"
  validate_json_manifest "$temporary/manifest.json"
  printf '%s\n' "test $TEST_NAME ... ok" 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' 'ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' 'ZONOS_METAL_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' > "$temporary/log"
  require_cargo_singleton "$temporary/log"; require_sentinel "$temporary/log" ZONOS_CPU_REFERENCE 'ZONOS_CPU_REFERENCE codes=EXACT.*verdict=MEASURED_NOT_GATED' 'CPU sentinel'; require_sentinel "$temporary/log" ZONOS_METAL_REFERENCE 'ZONOS_METAL_REFERENCE codes=EXACT.*verdict=MEASURED_NOT_GATED' 'Metal sentinel'
  printf '%s\n' 'ZONOS_METAL_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED extra' >> "$temporary/log"
  if require_sentinel "$temporary/log" ZONOS_METAL_REFERENCE 'ZONOS_METAL_REFERENCE codes=EXACT.*verdict=MEASURED_NOT_GATED' 'Metal sentinel' >/dev/null 2>&1; then fail=1; fi
  local expected_head approval approval_sha project_sha lock_sha transfer transfer_sha digest native native_sha
  expected_head='0123456789012345678901234567890123456789'; digest="$sha"; approval="$temporary/approval.json"; transfer="$temporary/transfer.txt"; native="$temporary/native.log"
  project_sha="$(sha256_file "$ROOT/tools/parity/pyproject.toml")"; lock_sha="$(sha256_file "$ROOT/tools/parity/uv.lock")"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
path, project_sha, lock_sha = sys.argv[1:]
scope = {
    "schema": "zonos-vast-approval-scope-v1",
    "project_sha256": project_sha,
    "lock_sha256": lock_sha,
    "source_repository": "https://github.com/Zyphra/Zonos.git",
    "source_revision": "bc40d98e1e1ab54fc65c483be127a90e3c7c0645",
    "upstream_repository": "Zyphra/Zonos-v0.1-transformer",
    "upstream_revision": "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
    "public_repository": "vokra/zonos-v0.1-transformer",
    "public_revision": "b1bf5c56d470eb9097e9b04f9deca364576574ba",
    "no_upload": True,
    "license_review": "AUTHENTICATED_LICENSE_IDENTITY_REQUIRED",
    "license_identity": {
        "source": {"repository": "https://github.com/Zyphra/Zonos.git", "revision": "bc40d98e1e1ab54fc65c483be127a90e3c7c0645", "path": "LICENSE", "spdx": "Apache-2.0", "bytes": 11357, "git_blob_sha1": "7a4a3ea2424c09fbe48d455aed1eaa94d9124835", "sha256": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd"},
        "upstream_model": {"id": "Zyphra/Zonos-v0.1-transformer", "revision": "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e", "private": False, "gated": False, "disabled": False, "card_data_license": "apache-2.0"},
    },
}
value = {"schema": "zonos-vast-approval-v1", "decision": "APPROVED", "signer": "self-test", "project_sha256": project_sha, "lock_sha256": lock_sha, "scope_sha256": hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "no_upload": True, "source_repository": scope["source_repository"], "source_revision": scope["source_revision"], "upstream_repository": scope["upstream_repository"], "upstream_revision": scope["upstream_revision"], "public_repository": scope["public_repository"], "public_revision": scope["public_revision"], "license_review": scope["license_review"]}
pathlib.Path(path).write_text(json.dumps(value), encoding="utf-8")
PY
  approval_sha="$(sha256_file "$approval")"; validate_approval "$approval"
  cp "$approval" "$temporary/approval-drift.json"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$temporary/approval-drift.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["scope_sha256"] = "0" * 64
path.write_text(json.dumps(value), encoding="utf-8")
PY
  if validate_approval "$temporary/approval-drift.json" >/dev/null 2>&1; then
    echo 'approval scope drift was accepted' >&2
    fail=1
  fi
  printf '%s\n' "test $TEST_NAME ... ok" 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' 'ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0.000000e+00 pcm_mean_abs=0.000000e+00 verdict=MEASURED_NOT_GATED' > "$native"
  native_sha="$(sha256_file "$native")"
  printf 'schema=zonos-apple-transfer-v1\nexpected_head=%s\napproval_evidence_sha256=%s\nmanifest_sha256=%s\ngguf_sha256=%s\ndac_gguf_sha256=%s\nconditioning_packet_sha256=%s\nreference_codes_sha256=%s\nreference_pcm_sha256=%s\nnative_cpu_log_sha256=%s\ncpu_result=ONE_PASS\ncpu_sentinel_summary=ZONOS_CPU_REFERENCE codes=EXACT verdict=MEASURED_NOT_GATED\nstatus=MEASURED_NOT_GATED\nmetal_status=NOT_RUN\npublication=NO_UPLOAD\n' \
    "$expected_head" "$approval_sha" "$digest" "$digest" "$digest" "$digest" "$digest" "$digest" "$native_sha" > "$transfer"
  printf '%s\n' "source_license_path=$SOURCE_LICENSE_PATH" "source_license_spdx=$SOURCE_LICENSE_SPDX" "source_license_bytes=$SOURCE_LICENSE_BYTES" "source_license_git_blob_sha1=$SOURCE_LICENSE_GIT_BLOB_SHA1" "source_license_sha256=$SOURCE_LICENSE_SHA256" "upstream_model_id=$UPSTREAM_MODEL_ID" "upstream_model_revision=$UPSTREAM_REVISION" 'upstream_model_private=false' 'upstream_model_gated=false' 'upstream_model_disabled=false' "upstream_model_card_data_license=$UPSTREAM_MODEL_CARD_DATA_LICENSE" >> "$transfer"
  validate_transfer_manifest "$transfer" "$(sha256_file "$transfer")" "$expected_head" "$approval_sha" "$digest" "$digest" "$digest" "$digest" "$digest" "$digest" "$native_sha"
  printf 'extra=value\n' >> "$transfer"
  if validate_transfer_manifest "$transfer" "$(sha256_file "$transfer")" "$expected_head" "$approval_sha" "$digest" "$digest" "$digest" "$digest" "$digest" "$digest" "$native_sha" >/dev/null 2>&1; then fail=1; fi
  local claimed existing
  claimed="$temporary/claimed-evidence"; claim_evidence_dir "$claimed"
  if claim_evidence_dir "$claimed" >/dev/null 2>&1; then fail=1; fi
  existing="$temporary/existing-evidence"; mkdir "$existing"
  if claim_evidence_dir "$existing" >/dev/null 2>&1; then fail=1; fi
  validate_absent_evidence "$temporary/nested/evidence" "$temporary/value" "$temporary/manifest.json"; mkdir -p "$temporary/real"; ln -s "$temporary/real" "$temporary/link"
  if validate_absent_evidence "$temporary/link/new/evidence" "$temporary/value" >/dev/null 2>&1; then fail=1; fi
  if "$path" --self-test --manifest x >/dev/null 2>&1 || "$path" --unknown >/dev/null 2>&1; then fail=1; fi
  [[ ! -e "$temporary/evidence" ]] || fail=1
  for token in 'VOKRA_ZONOS_BACKEND' 'ZONOS_CPU_REFERENCE' 'ZONOS_METAL_REFERENCE' 'verdict=MEASURED_NOT_GATED' 'NO_UPLOAD' 'CARGO_BUILD_JOBS=1' '--offline' '--test-threads=1' '--expected-head' '--approval-evidence-sha256' '--transfer-manifest-sha256' '--native-cpu-log-sha256' 'claim_evidence_dir' 'reject_symlink_ancestors' 'Apple evidence directory must be absent' 'source LICENSE' 'Apache-2.0' '11357' '7a4a3ea2424c09fbe48d455aed1eaa94d9124835' '58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd' 'cardData.license' 'upstream_model_card_data_license'; do grep -Fq -- "$token" "$path" || fail=1; done
  old_license_gate='LICENSE_IDENTITY_AUTHENTICATED'; old_license_gate+=' = False'
  grep -Fq "$old_license_gate" "$path" && fail=1 || true
  grep -Eq '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push|upload\.sh|publish-one\.sh)([[:space:]]|$)' "$path" && fail=1 || true
  (( fail == 0 )) || return 1; log 'self-test PASS'
)

main() {
  local self=0 expected_head='' approval='' approval_sha='' transfer='' transfer_sha='' manifest='' manifest_sha='' gguf='' gguf_sha='' gguf_size='' dac='' dac_sha='' dac_size='' packet='' packet_sha='' packet_size='' codes='' codes_sha='' codes_size='' pcm='' pcm_sha='' pcm_size='' native='' native_sha='' evidence='' key value_name seen_options='' self_seen=0 option_count=0
  while (( $# )); do
    key="$1"
    if [[ "$key" == --self-test ]]; then (( self_seen == 0 )) || die 'duplicate --self-test'; self_seen=1; self=1; shift; continue; fi
    case "$key" in
      --expected-head) value_name=expected_head;; --approval-evidence) value_name=approval;; --approval-evidence-sha256) value_name=approval_sha;; --transfer-manifest) value_name=transfer;; --transfer-manifest-sha256) value_name=transfer_sha;; --manifest) value_name=manifest;; --manifest-sha256) value_name=manifest_sha;;
      --gguf) value_name=gguf;; --gguf-sha256) value_name=gguf_sha;; --gguf-size) value_name=gguf_size;;
      --dac-gguf) value_name=dac;; --dac-gguf-sha256) value_name=dac_sha;; --dac-gguf-size) value_name=dac_size;;
      --conditioning-packet) value_name=packet;; --conditioning-packet-sha256) value_name=packet_sha;; --conditioning-packet-size) value_name=packet_size;;
      --reference-codes) value_name=codes;; --reference-codes-sha256) value_name=codes_sha;; --reference-codes-size) value_name=codes_size;;
      --reference-pcm) value_name=pcm;; --reference-pcm-sha256) value_name=pcm_sha;; --reference-pcm-size) value_name=pcm_size;; --native-cpu-log) value_name=native;; --native-cpu-log-sha256) value_name=native_sha;; --evidence-dir) value_name=evidence;;
      -h|--help) [[ $# == 1 && $self == 0 ]] || die '--help cannot be combined'; usage; return 0;; *) die "unknown argument: $key";;
    esac
    [[ " $seen_options " != *" $value_name "* ]] || die "duplicate $key"; seen_options="$seen_options $value_name"; (( option_count += 1 ))
    [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "$key requires a nonempty value"
    printf -v "$value_name" '%s' "$2"; shift 2
  done
  if (( self )); then [[ -z "$seen_options" ]] || die '--self-test accepts no other arguments'; self_test; return; fi
  (( option_count == 25 )) || die 'all explicit artifact, manifest, approval, transfer, native-log, and evidence arguments are required'
  : "$SOURCE_REVISION" "$UPSTREAM_REVISION" "$PUBLIC_REVISION"
  require_clean_expected_head "$expected_head"
  require_sha "$approval_sha"; require_file 'approval evidence' "$approval"; [[ "$(sha256_file "$approval")" == "$(lower_sha "$approval_sha")" ]] || die 'approval evidence SHA-256 mismatch'
  validate_approval "$approval"
  require_sha "$manifest_sha"; require_file 'VAST manifest' "$manifest"; [[ "$(sha256_file "$manifest")" == "$(lower_sha "$manifest_sha")" ]] || die 'manifest SHA-256 mismatch'; validate_json_manifest "$manifest"
  require_sha "$transfer_sha"; validate_transfer_manifest "$transfer" "$transfer_sha" "$expected_head" "$approval_sha" "$manifest_sha" "$gguf_sha" "$dac_sha" "$packet_sha" "$codes_sha" "$pcm_sha" "$native_sha"
  require_input 'Zonos GGUF' "$gguf" "$gguf_sha" "$gguf_size"; require_input 'Zonos DAC GGUF' "$dac" "$dac_sha" "$dac_size"; require_input 'conditioning packet' "$packet" "$packet_sha" "$packet_size"; require_input 'reference codes' "$codes" "$codes_sha" "$codes_size"; require_input 'reference PCM' "$pcm" "$pcm_sha" "$pcm_size"; require_file 'VAST native CPU log' "$native"; require_sha "$native_sha"; [[ "$(sha256_file "$native")" == "$(lower_sha "$native_sha")" ]] || die 'native CPU log SHA-256 mismatch'; require_cargo_singleton "$native"; require_sentinel "$native" ZONOS_CPU_REFERENCE 'ZONOS_CPU_REFERENCE codes=EXACT.*verdict=MEASURED_NOT_GATED' 'VAST CPU sentinel'
  validate_absent_evidence "$evidence" "$approval" "$manifest" "$transfer" "$native" "$gguf" "$dac" "$packet" "$codes" "$pcm"
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 && "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'real remote Darwin arm64 is required'
  command -v cargo >/dev/null 2>&1 || die 'cargo is required'; command -v xcrun >/dev/null 2>&1 || die 'xcrun is required'; xcrun --find metal >/dev/null 2>&1 || die 'Metal toolchain unavailable'
  claim_evidence_dir "$evidence"; export VOKRA_ZONOS_GGUF="$gguf" VOKRA_ZONOS_DAC_GGUF="$dac" VOKRA_ZONOS_CONDITIONING_PACKET="$packet" VOKRA_ZONOS_PACKET_SHA256="$packet_sha" VOKRA_ZONOS_REFERENCE_CODES="$codes" VOKRA_ZONOS_REFERENCE_PCM="$pcm"
  for backend in cpu metal; do
    backend_upper="$(printf '%s' "$backend" | tr '[:lower:]' '[:upper:]')"
    log "running Zonos $backend exact test"; VOKRA_ZONOS_BACKEND="$backend" CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --offline --locked -p vokra-models --features metal --test parity_zonos_real "$TEST_NAME" -- --ignored --exact --nocapture --test-threads=1 > "$evidence/$backend.log" 2>&1 || die "$backend Zonos test failed"
    require_cargo_singleton "$evidence/$backend.log"; require_sentinel "$evidence/$backend.log" "ZONOS_${backend_upper}_REFERENCE" "ZONOS_${backend_upper}_REFERENCE codes=EXACT.*verdict=MEASURED_NOT_GATED" "$backend sentinel"
  done
  require_clean_expected_head "$expected_head"
  printf '%s\n' '{"publication":"NO_UPLOAD","cpu_status":"MEASURED_NOT_GATED","metal_status":"MEASURED_NOT_GATED","codes_status":"EXACT","pcm_status":"MEASURED_NOT_GATED"}' > "$evidence/summary.json"; return 2
}

main "$@"
