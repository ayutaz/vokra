#!/usr/bin/env bash
# VAST/Linux-only, no-upload SGMSE-VoiceBank end-to-end orchestration.
# Existing authenticated runners own all model math and artifact handling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTION_RUNNER="$SCRIPT_DIR/run-sgmse-voicebank-inspection.sh"
PREPARE_RUNNER="$SCRIPT_DIR/run-sgmse-voicebank-prepare.sh"
SCORE_REFERENCE_RUNNER="$SCRIPT_DIR/run-sgmse-voicebank-reference.sh"
SCORE_PARITY_RUNNER="$SCRIPT_DIR/run-sgmse-native-score-parity.sh"
ENHANCEMENT_RUNNER="$SCRIPT_DIR/run-sgmse-native-enhancement-parity.sh"
WORK_DIR=""
VOKRA_COMMIT=""
SELF_TEST=0
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((32 * 1024 * 1024))

log() { printf '[sgmse-validation-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

usage() {
  cat <<'EOF'
usage: run-sgmse-voicebank-validation.sh --vokra-commit <40-hex> [--work-dir <absent-dir>]
       run-sgmse-voicebank-validation.sh --self-test

VAST/Linux x86_64 only, with at least 128 GiB RAM and 32 GiB free disk. The exact clean Vokra commit is inspected, prepared,
converted with explicit Apache-2.0, tested against independent score and
official waveform enhancement references, then sent through package,
workspace, clippy, deny, audit, and static gates. Status is NO_UPLOAD.
Transfer only small evidence and destroy the disposable VAST instance;
do not transfer generated model artifacts.
EOF
}

require_clean_commit() {
  [[ "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" == "$VOKRA_COMMIT" ]] || die 'Vokra commit changed or differs from --vokra-commit'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST Vokra checkout must be clean'
}

run_logged() {
  local label="$1"
  shift
  log "stage=$label"
  if ! "$@" >"$LOG_DIR/$label.log" 2>&1; then
    tail -n 80 "$LOG_DIR/$label.log" >&2 || true
    die "stage failed: $label"
  fi
  tail -n 20 "$LOG_DIR/$label.log" >&2 || true
}

inspection_exit_contract() {
  local runner="$1" logfile="$2" status=0
  shift 2
  set +e
  "$runner" "$@" >"$logfile" 2>&1
  status=$?
  set -e
  [[ "$status" == 2 ]]
}

run_inspection_stage() {
  local runner="$1" logfile="$2" manifest="$3"
  shift 3
  log 'stage=inspection'
  if ! inspection_exit_contract "$runner" "$logfile" "$@"; then
    tail -n 80 "$logfile" >&2 || true
    die 'inspection must exit 2 with the four expected blockers'
  fi
  [[ -f "$manifest" && ! -L "$manifest" && -s "$manifest" ]] || die 'inspection manifest is missing, symlinked, or empty'
  validate_inspection_manifest "$manifest" >/dev/null || die 'inspection unresolved evidence failed authentication'
  tail -n 20 "$logfile" >&2 || true
}

validate_inspection_manifest() {
  local manifest="$1"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$manifest" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
expected_blockers = [
    "BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION: reviewed SpeechBrain 1.0.3 distribution lacks source-backed SGMSE integration",
    "BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE: independent upstream reference was not executed",
    "BLOCKED_EMA_SELECTION_UNVERIFIED: safe-loaded tensor map was not selected by a reviewed EMA loader",
    "BLOCKED_EXACT_NCSNPP_TENSOR_MAPPING_UNPROVEN: source construction has not yielded a reviewed one-to-one role map",
]
if data.get("format") != "vokra-sgmse-voicebank-inspection-v1":
    raise SystemExit("inspection format mismatch")
if data.get("model_repository") != "speechbrain/sgmse-voicebank" or data.get("model_revision") != "8f4ff7b65284c49492a43349b8106e094ac0d365":
    raise SystemExit("inspection model identity mismatch")
if data.get("runtime_status") != "INSPECTION_ONLY" or data.get("parity_status") != "INSPECTION_ONLY" or data.get("publication") != "NO_UPLOAD":
    raise SystemExit("inspection status/publication mismatch")
if data.get("blockers") != expected_blockers:
    raise SystemExit("inspection blockers are not exactly the four reviewed blockers")
if data.get("checkpoint") != {
    "expected_filename": "score_model_ema.ckpt",
    "expected_size": 262593305,
    "expected_sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3",
    "filename": "score_model_ema.ckpt",
    "size": 262593305,
    "sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3",
}:
    raise SystemExit("inspection checkpoint identity mismatch")
algorithm = data.get("algorithm_source")
if not isinstance(algorithm, dict) or algorithm.get("repository") != "https://github.com/sp-uhh/sgmse.git" or any(algorithm.get(key) != "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e" for key in ("expected_revision", "resolved_revision", "revision")):
    raise SystemExit("inspection algorithm source identity mismatch")
if algorithm.get("license_spdx") != "mit" or algorithm.get("license_sha256") != "8748956d2e5afe9dfc8311188b4119dacc7c5293b0561e7cca7a21cf80e54caa":
    raise SystemExit("inspection algorithm license mismatch")
expected_roles = {
    "ncsnpp": [("sgmse/backbones/__init__.py", "345d30172586d9bc866e167b44642fb3d8af793ec53fe85112972c75badedca0"), ("sgmse/backbones/ncsnpp_v2.py", "53e83bccd8fb00ea6afc4f5433c79c94137763bf8fd103c8d26d680ad23631c3")],
    "sampler_corrector": [("sgmse/sampling/correctors.py", "5253b099827b913e5200fe6326c9759a991d673754cbc2436906824aa6f7a289")],
    "sampler_predictor": [("sgmse/sampling/predictors.py", "85819b1cfb4e853b92858803beae63a6c287a1efa84cd71c95a070ff8be342e1")],
    "score_model": [("sgmse/model.py", "ed7b959bfa2c589f9eef2c9b5076712323db5ee28f5b8c9e21b8c6f8ebbb0ce5")],
    "sde": [("sgmse/model.py", "ed7b959bfa2c589f9eef2c9b5076712323db5ee28f5b8c9e21b8c6f8ebbb0ce5"), ("sgmse/sdes.py", "40d7bbd9cfec79a66eedbd55283be2fc67ec6b3eb5668611994cf870690bf3e4")],
}

observed_roles = algorithm.get("files_by_role")
if not isinstance(observed_roles, dict) or {
    role: [(row.get("path"), row.get("sha256")) for row in rows]
    for role, rows in observed_roles.items()
} != expected_roles:
    raise SystemExit("inspection algorithm role hashes mismatch")
speechbrain = data.get("speechbrain_source")
if not isinstance(speechbrain, dict) or speechbrain.get("repository") != "https://github.com/speechbrain/speechbrain.git" or any(speechbrain.get(key) != "2b3f4f44351fd08a627c4ab307de5c420351bc19" for key in ("expected_revision", "resolved_revision")):
    raise SystemExit("inspection SpeechBrain source identity mismatch")
if speechbrain.get("license_spdx") != "apache-2.0" or speechbrain.get("license_sha256") != "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4":
    raise SystemExit("inspection SpeechBrain license mismatch")
if speechbrain.get("locked_distribution") != {
    "version": "1.0.3",
    "sdist_sha256": "fcab3c6e90012cecb1eed40ea235733b550137e73da6bfa2340ba191ec714052",
    "wheel_sha256": "9859d4c1b1fb3af3b85523c0c89f52e45a04f305622ed55f31aa32dd2fba19e9",
}:
    raise SystemExit("inspection SpeechBrain distribution identity mismatch")
distribution_audit = speechbrain.get("locked_distribution_audit")
expected_distribution_roles = {
    "speechbrain/inference/enhancement.py": {"present": True, "required_markers": {"class SGMSEEnhancement": False}},
    "speechbrain/integrations/models/sgmse_plus.py": {"present": False, "required_markers": {}},
}
if (
    not isinstance(distribution_audit, dict)
    or distribution_audit.get("status") != "BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION"
    or distribution_audit.get("expected_version") != "1.0.3"
    or distribution_audit.get("version") != "1.0.3"
    or distribution_audit.get("expected_source_roles") != expected_distribution_roles
    or distribution_audit.get("observed_source_roles") != expected_distribution_roles
):
    raise SystemExit("inspection distribution blocker evidence mismatch")
if data.get("wheel_integration") is True or speechbrain.get("wheel_integration") is True:
    raise SystemExit("inspection must not claim wheel SGMSE integration")
expected_speech_files = {
    "speechbrain/inference/enhancement.py": ("019e79bb489ba4c7f1ddd681e0cc007d7034386636ffd156128ec85058476995", 11693),
    "speechbrain/integrations/models/sgmse_plus.py": ("b70ecde1d7326282b339348c739e91413c6dbac07ef98d34b540be07d8e70935", 21777),
}
speech_files = speechbrain.get("executable_files")
if not isinstance(speech_files, dict) or set(speech_files) != set(expected_speech_files):
    raise SystemExit("inspection SpeechBrain executable source set mismatch")
for name, (digest, size) in expected_speech_files.items():
    row = speech_files[name]
    if row.get("sha256") != digest or row.get("size") != size:
        raise SystemExit(f"inspection SpeechBrain source mismatch: {name}")
safe_load = data.get("safe_load")
if not isinstance(safe_load, dict) or safe_load.get("safe_load_status") != "SAFE_LOADED" or safe_load.get("tensor_count") != 647 or safe_load.get("all_finite") is not True or safe_load.get("parameter_count") != 65590822:
    raise SystemExit("inspection safe-load identity mismatch")
tensor_manifest = safe_load.get("tensor_manifest")
if not isinstance(tensor_manifest, dict) or len(tensor_manifest) != 647 or any(
    not isinstance(row, dict) or row.get("dtype") != "torch.float32" or row.get("finite") is not True or not isinstance(row.get("count"), int) or row["count"] <= 0
    for row in tensor_manifest.values()
):
    raise SystemExit("inspection tensor manifest is not exactly 647 finite F32 rows")
hyperparams = data.get("hyperparams")
if not isinstance(hyperparams, dict) or hyperparams.get("filename") != "hyperparams.yaml" or hyperparams.get("sha256") != "5ebd87c6257537c3997c134b279d85cd7bebccce0e6d3fc68f7a36f15096aa51":
    raise SystemExit("inspection hyperparams identity mismatch")
tensor_contract = data.get("tensor_contract")
if not isinstance(tensor_contract, dict) or tensor_contract.get("format") != "vokra-sgmse-typed-role-manifest-v2" or tensor_contract.get("reviewed_manifest_sha256") != "409690f70b534771055dc4f740cc66bdb4d1b25dba5e22fd066109adce77278c" or tensor_contract.get("checkpoint_tensor_count") != 647:
    raise SystemExit("inspection typed contract identity mismatch")
if data.get("weight_license_spdx") != "apache-2.0":
    raise SystemExit("inspection weight license mismatch")
print("INSPECTION_UNRESOLVED_FOUR_BLOCKERS_AUTHENTICATED")
PY
}

write_resolution_ledger() {
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$LEDGER" "$INSPECTION_DIR/evidence/sgmse_voicebank_manifest.json" "$PREPARED" \
    "$PREPARED_MANIFEST" "$SCORE_REFERENCE_DIR/manifest.json" "$ENHANCEMENT_REFERENCE_DIR/manifest.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

ledger, inspection_path, prepared, prepared_manifest, score_path, enhancement_path = map(pathlib.Path, sys.argv[1:])

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def load(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)

inspection = load(inspection_path)
expected_blockers = [
    "BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION: reviewed SpeechBrain 1.0.3 distribution lacks source-backed SGMSE integration",
    "BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE: independent upstream reference was not executed",
    "BLOCKED_EMA_SELECTION_UNVERIFIED: safe-loaded tensor map was not selected by a reviewed EMA loader",
    "BLOCKED_EXACT_NCSNPP_TENSOR_MAPPING_UNPROVEN: source construction has not yielded a reviewed one-to-one role map",
]
if inspection.get("blockers") != expected_blockers or inspection.get("runtime_status") != "INSPECTION_ONLY":
    raise SystemExit("inspection was not retained as unresolved four-blocker evidence")
prepared_data = load(prepared_manifest)
expected_keys = {
    "format", "repository", "model_revision", "source_repository", "source_revision",
    "checkpoint_filename", "checkpoint_size", "checkpoint_sha256", "prepared_sha256",
    "tensor_count", "typed_manifest_sha256", "tensor_rows",
}
if set(prepared_data) != expected_keys or prepared_data["tensor_count"] != 647 or prepared_data["typed_manifest_sha256"] != "409690f70b534771055dc4f740cc66bdb4d1b25dba5e22fd066109adce77278c":
    raise SystemExit("prepared mapping evidence is not the reviewed 647-row contract")
if prepared_data["prepared_sha256"] != hashlib.sha256(prepared.read_bytes()).hexdigest():
    raise SystemExit("prepared artifact digest is not bound")
score = load(score_path)
enhancement = load(enhancement_path)
ema_transfer_hashes = []
for name, data in (("score", score), ("enhancement", enhancement)):
    if data.get("status") != "REFERENCE_COMPLETE_NO_UPLOAD" or data.get("publication") != "NO_UPLOAD":
        raise SystemExit(f"{name} reference is incomplete or publishable")
    ema = data.get("ema_route")
    source_files = ema.get("source_files") if isinstance(ema, dict) else None
    score_source = source_files.get("score_model") if isinstance(source_files, dict) else None
    transfer_source = source_files.get("parameter_transfer") if isinstance(source_files, dict) else None
    if not isinstance(ema, dict) or ema.get("status") != "SOURCE_ROUTE_VERIFIED_STRICT_LOAD" or ema.get("unsafe_pickle_fallback") is not False or ema.get("parameter_load") != "strict_state_dict" or ema.get("loadable") != "score_model_ema" or not isinstance(score_source, dict) or score_source.get("path") != "speechbrain/integrations/models/sgmse_plus.py" or score_source.get("sha256") != "b70ecde1d7326282b339348c739e91413c6dbac07ef98d34b540be07d8e70935" or score_source.get("size") != 21777 or not isinstance(transfer_source, dict) or transfer_source.get("path") != "speechbrain/utils/parameter_transfer.py" or not isinstance(transfer_source.get("sha256"), str) or len(transfer_source["sha256"]) != 64 or not isinstance(transfer_source.get("size"), int) or transfer_source["size"] <= 0:
        raise SystemExit(f"{name} EMA route is not strict source-authenticated")
    ema_transfer_hashes.append(transfer_source["sha256"])
    model = data.get("model")
    if not isinstance(model, dict) or model.get("load") != "torch.load(weights_only=True)+load_state_dict(strict=True)" or model.get("tensor_count") != 647 or model.get("parameter_count") != 65590822:
        raise SystemExit(f"{name} model load evidence is not strict or exact")
    source = data.get("source")
    speechbrain = data.get("speechbrain_source")
    if not isinstance(source, dict) or source.get("repository") != "https://github.com/sp-uhh/sgmse.git" or source.get("revision") != "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e" or not isinstance(speechbrain, dict) or speechbrain.get("repository") != "https://github.com/speechbrain/speechbrain.git" or speechbrain.get("revision") != "2b3f4f44351fd08a627c4ab307de5c420351bc19":
        raise SystemExit(f"{name} source checkout identity is not pinned")
if len(set(ema_transfer_hashes)) != 1:
    raise SystemExit("EMA parameter-transfer source hash differs between references")
data = {
    "format": "vokra-sgmse-resolution-ledger-v1",
    "status": "RESOLUTION_LEDGER_COMPLETE_NO_UPLOAD",
    "publication": "NO_UPLOAD",
    "inspection": {"status": "PRE_REFERENCE_UNRESOLVED", "blockers": inspection["blockers"], "runtime_status": inspection["runtime_status"]},
    "distribution_resolution": {"status": "RESOLVED_BY_AUTHENTICATED_SOURCE_CHECKOUT", "wheel_integration_claim": False},
    "reference_resolution": {"score": "REFERENCE_COMPLETE_NO_UPLOAD", "official_waveform": "REFERENCE_COMPLETE_NO_UPLOAD"},
    "ema_resolution": {"score": "SOURCE_ROUTE_VERIFIED_STRICT_LOAD", "official_waveform": "SOURCE_ROUTE_VERIFIED_STRICT_LOAD"},
    "mapping_resolution": {"status": "REVIEWED_647_ROW_CONTRACT_BOUND", "typed_manifest_sha256": prepared_data["typed_manifest_sha256"], "tensor_count": prepared_data["tensor_count"]},
}
if ledger.exists() or ledger.is_symlink():
    raise SystemExit("resolution ledger already exists")
temporary = ledger.with_name(f".{ledger.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, ledger)
temporary.unlink()
print("RESOLUTION_LEDGER_COMPLETE_NO_UPLOAD")
PY
}

validate_prepared_sidecar() {
  local artifact="$1" sidecar="$2"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$artifact" "$sidecar" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

artifact, sidecar = map(pathlib.Path, sys.argv[1:])

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

manifest = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
expected_keys = {
    "format", "repository", "model_revision", "source_repository", "source_revision",
    "checkpoint_filename", "checkpoint_size", "checkpoint_sha256", "prepared_sha256",
    "tensor_count", "typed_manifest_sha256", "tensor_rows",
}
if set(manifest) != expected_keys:
    raise SystemExit("prepared sidecar schema mismatch")
expected = {
    "format": "vokra-sgmse-voicebank-prepared-v1",
    "repository": "speechbrain/sgmse-voicebank",
    "model_revision": "8f4ff7b65284c49492a43349b8106e094ac0d365",
    "source_repository": "https://github.com/sp-uhh/sgmse.git",
    "source_revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e",
    "checkpoint_filename": "score_model_ema.ckpt",
    "checkpoint_size": 262593305,
    "checkpoint_sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3",
}
for key, value in expected.items():
    if manifest[key] != value:
        raise SystemExit(f"prepared sidecar identity mismatch: {key}")
if not isinstance(manifest["tensor_count"], int) or manifest["tensor_count"] <= 0:
    raise SystemExit("prepared tensor count is invalid")
if not isinstance(manifest["tensor_rows"], list) or len(manifest["tensor_rows"]) != manifest["tensor_count"]:
    raise SystemExit("prepared tensor rows are invalid")
for key in ("prepared_sha256", "typed_manifest_sha256"):
    if not isinstance(manifest[key], str) or re.fullmatch(r"[0-9a-f]{64}", manifest[key]) is None:
        raise SystemExit(f"prepared sidecar digest is malformed: {key}")
if not artifact.is_file() or artifact.is_symlink():
    raise SystemExit("prepared artifact is missing or symlinked")
if manifest["prepared_sha256"] != hashlib.sha256(artifact.read_bytes()).hexdigest():
    raise SystemExit("prepared artifact digest mismatch")
print("PREPARED_SAFETENSORS_READY")
PY
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token previous=0 line label
  for token in \
    'run_inspection_stage' 'inspection_exit_contract' 'validate_inspection_manifest' 'write_resolution_ledger' \
    'run_logged preparation' 'run_logged build-converter' \
    'run_logged strict-conversion' 'run_logged score-reference' 'run_logged score-parity' \
    'run_logged enhancement-reference' 'run_logged enhancement-parity' 'run_logged metadata' \
    'run_logged package' 'run_logged workspace' 'run_logged clippy' 'run_logged deny' \
    'run_logged audit' 'run_logged static-fmt' 'run_logged static-forbidden' \
    'run_logged static-zero-deps' 'run_logged static-bound-arch' \
    'run_logged static-fixture-pins' 'run_logged static-dynamic-load' '--vokra-commit' \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'PREPARED_SAFETENSORS_READY' \
    'expected_keys' 'typed_manifest_sha256' 'INSPECTION_UNRESOLVED_FOUR_BLOCKERS_AUTHENTICATED' \
    'resolution-ledger.json' 'RESOLUTION_LEDGER_COMPLETE_NO_UPLOAD' 'PRE_REFERENCE_UNRESOLVED' \
    'BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION' 'BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE' \
    'BLOCKED_EMA_SELECTION_UNVERIFIED' 'BLOCKED_EXACT_NCSNPP_TENSOR_MAPPING_UNPROVEN' \
    "cd \"\$VOKRA_ROOT\"" '128 GiB' '32 GiB' \
    'STRICT_BIND_PASS' 'apache-2.0' \
    'sgmse_native_score_matches_independent_reference' \
    'sgmse_native_enhancement_matches_official_reference' \
    'SGMSE_NATIVE_SCORE_PARITY_PASS' 'CPU_ENHANCEMENT_PARITY_PASS' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings' \
    'cargo deny check licenses advisories bans' 'cargo audit' \
    'SGMSE_VALIDATION_COMPLETE_NO_UPLOAD' 'NO_UPLOAD' \
    'destroy the disposable VAST instance'; do
    grep -Fq -- "$token" "$path" || { log "self-test FAIL: missing contract token: $token"; fail=1; }
  done
  line="$(awk '/^run_inspection_stage / && $0 !~ /stage_runner/ {print NR; exit}' "$path")"
  [[ -n "$line" ]] || { log 'self-test FAIL: dedicated inspection stage missing'; fail=1; }
  previous="$line"
  for label in preparation build-converter strict-conversion score-reference score-parity enhancement-reference enhancement-parity metadata package workspace clippy deny audit static-fmt static-forbidden; do
    line="$(grep -nE "^run_logged $label([[:space:]]|$)" "$path" | head -n 1 | cut -d: -f1)"
    [[ -n "$line" && "$line" -gt "$previous" ]] || { log "self-test FAIL: stage order: $label"; fail=1; }
    previous="$line"
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if grep -Eq '^[[:space:]]*if[[:space:]]+prepared_data\.get\("publication"\)' "$path"; then
    log 'self-test FAIL: impossible prepared sidecar fields are being required'
    fail=1
  fi
  for runner in "$INSPECTION_RUNNER" "$PREPARE_RUNNER" "$SCORE_REFERENCE_RUNNER" "$SCORE_PARITY_RUNNER" "$ENHANCEMENT_RUNNER"; do
    bash "$runner" --self-test >/dev/null || { log "self-test FAIL: delegated runner: $runner"; fail=1; }
  done
  if "$path" --self-test --work-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  local exit_tmp exit_runner
  exit_tmp="$(mktemp -d "${TMPDIR:-/tmp}/sgmse-validation-exit-selftest.XXXXXX")"
  exit_runner="$exit_tmp/exit-2.sh"
  printf '#!/bin/sh\nexit 2\n' >"$exit_runner"
  chmod +x "$exit_runner"
  inspection_exit_contract "$exit_runner" "$exit_tmp/exit-2.log" || fail=1
  for unexpected in 0 1 3; do
    exit_runner="$exit_tmp/exit-$unexpected.sh"
    printf '#!/bin/sh\nexit %s\n' "$unexpected" >"$exit_runner"
    chmod +x "$exit_runner"
    if inspection_exit_contract "$exit_runner" "$exit_tmp/exit-$unexpected.log"; then
      log "self-test FAIL: inspection exit $unexpected accepted"
      fail=1
    fi
  done
  rm -rf -- "$exit_tmp"
  local inspection_tmp case_file
  inspection_tmp="$(mktemp -d "${TMPDIR:-/tmp}/sgmse-validation-inspection-selftest.XXXXXX")"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$inspection_tmp" <<'PY'
import copy
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
blockers = [
    "BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION: reviewed SpeechBrain 1.0.3 distribution lacks source-backed SGMSE integration",
    "BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE: independent upstream reference was not executed",
    "BLOCKED_EMA_SELECTION_UNVERIFIED: safe-loaded tensor map was not selected by a reviewed EMA loader",
    "BLOCKED_EXACT_NCSNPP_TENSOR_MAPPING_UNPROVEN: source construction has not yielded a reviewed one-to-one role map",
]
algorithm_roles = {
    "ncsnpp": [{"path": "sgmse/backbones/__init__.py", "sha256": "345d30172586d9bc866e167b44642fb3d8af793ec53fe85112972c75badedca0"}, {"path": "sgmse/backbones/ncsnpp_v2.py", "sha256": "53e83bccd8fb00ea6afc4f5433c79c94137763bf8fd103c8d26d680ad23631c3"}],
    "sampler_corrector": [{"path": "sgmse/sampling/correctors.py", "sha256": "5253b099827b913e5200fe6326c9759a991d673754cbc2436906824aa6f7a289"}],
    "sampler_predictor": [{"path": "sgmse/sampling/predictors.py", "sha256": "85819b1cfb4e853b92858803beae63a6c287a1efa84cd71c95a070ff8be342e1"}],
    "score_model": [{"path": "sgmse/model.py", "sha256": "ed7b959bfa2c589f9eef2c9b5076712323db5ee28f5b8c9e21b8c6f8ebbb0ce5"}],
    "sde": [{"path": "sgmse/model.py", "sha256": "ed7b959bfa2c589f9eef2c9b5076712323db5ee28f5b8c9e21b8c6f8ebbb0ce5"}, {"path": "sgmse/sdes.py", "sha256": "40d7bbd9cfec79a66eedbd55283be2fc67ec6b3eb5668611994cf870690bf3e4"}],
}
distribution_roles = {
    "speechbrain/inference/enhancement.py": {"present": True, "required_markers": {"class SGMSEEnhancement": False}},
    "speechbrain/integrations/models/sgmse_plus.py": {"present": False, "required_markers": {}},
}
base = {
    "format": "vokra-sgmse-voicebank-inspection-v1",
    "model_repository": "speechbrain/sgmse-voicebank",
    "model_revision": "8f4ff7b65284c49492a43349b8106e094ac0d365",
    "runtime_status": "INSPECTION_ONLY",
    "parity_status": "INSPECTION_ONLY",
    "publication": "NO_UPLOAD",
    "blockers": blockers,
    "checkpoint": {"expected_filename": "score_model_ema.ckpt", "expected_size": 262593305, "expected_sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3", "filename": "score_model_ema.ckpt", "size": 262593305, "sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3"},
    "algorithm_source": {"repository": "https://github.com/sp-uhh/sgmse.git", "expected_revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e", "resolved_revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e", "revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e", "license_spdx": "mit", "license_sha256": "8748956d2e5afe9dfc8311188b4119dacc7c5293b0561e7cca7a21cf80e54caa", "files_by_role": algorithm_roles},
    "speechbrain_source": {"repository": "https://github.com/speechbrain/speechbrain.git", "expected_revision": "2b3f4f44351fd08a627c4ab307de5c420351bc19", "resolved_revision": "2b3f4f44351fd08a627c4ab307de5c420351bc19", "license_spdx": "apache-2.0", "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "locked_distribution": {"version": "1.0.3", "sdist_sha256": "fcab3c6e90012cecb1eed40ea235733b550137e73da6bfa2340ba191ec714052", "wheel_sha256": "9859d4c1b1fb3af3b85523c0c89f52e45a04f305622ed55f31aa32dd2fba19e9"}, "locked_distribution_audit": {"status": "BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION", "expected_version": "1.0.3", "version": "1.0.3", "expected_source_roles": distribution_roles, "observed_source_roles": distribution_roles}, "executable_files": {"speechbrain/inference/enhancement.py": {"sha256": "019e79bb489ba4c7f1ddd681e0cc007d7034386636ffd156128ec85058476995", "size": 11693}, "speechbrain/integrations/models/sgmse_plus.py": {"sha256": "b70ecde1d7326282b339348c739e91413c6dbac07ef98d34b540be07d8e70935", "size": 21777}}},
    "safe_load": {"safe_load_status": "SAFE_LOADED", "tensor_count": 647, "all_finite": True, "parameter_count": 65590822, "tensor_manifest": {f"tensor.{i}": {"dtype": "torch.float32", "finite": True, "count": 1} for i in range(647)}},
    "hyperparams": {"filename": "hyperparams.yaml", "sha256": "5ebd87c6257537c3997c134b279d85cd7bebccce0e6d3fc68f7a36f15096aa51"},
    "tensor_contract": {"format": "vokra-sgmse-typed-role-manifest-v2", "reviewed_manifest_sha256": "409690f70b534771055dc4f740cc66bdb4d1b25dba5e22fd066109adce77278c", "checkpoint_tensor_count": 647},
    "weight_license_spdx": "apache-2.0",
}
(root / "base.json").write_text(json.dumps(base), encoding="utf-8")
for name, mutate in (("missing", lambda value: value["blockers"].pop()), ("extra", lambda value: value["blockers"].append("EXTRA")), ("wrong-safe-load", lambda value: value["safe_load"].update(safe_load_status="BLOCKED")), ("wrong-count", lambda value: value["safe_load"].update(tensor_count=646)), ("wrong-digest", lambda value: value["tensor_contract"].update(reviewed_manifest_sha256="0" * 64)), ("wheel-integration", lambda value: value.update(wheel_integration=True))):
    candidate = copy.deepcopy(base)
    mutate(candidate)
    (root / f"{name}.json").write_text(json.dumps(candidate), encoding="utf-8")
(root / "duplicate.json").write_text('{"format":"a","format":"b"}', encoding="utf-8")
PY
  local stage_runner stage_manifest
  stage_runner="$inspection_tmp/inspection-stage.sh"
  stage_manifest="$inspection_tmp/stage-manifest.json"
  printf '#!/bin/sh\ncp %q %q\nexit 2\n' "$inspection_tmp/base.json" "$stage_manifest" >"$stage_runner"
  chmod +x "$stage_runner"
  run_inspection_stage "$stage_runner" "$inspection_tmp/stage.log" "$stage_manifest"
  for unexpected in 0 3; do
    printf '#!/bin/sh\nexit %s\n' "$unexpected" >"$stage_runner"
    if (run_inspection_stage "$stage_runner" "$inspection_tmp/stage-$unexpected.log" "$stage_manifest") >/dev/null 2>&1; then
      log "self-test FAIL: dedicated inspection stage accepted exit $unexpected"
      fail=1
    fi
  done
  validate_inspection_manifest "$inspection_tmp/base.json" >/dev/null || fail=1
  for case_file in missing extra wrong-safe-load wrong-count wrong-digest wheel-integration duplicate; do
    if validate_inspection_manifest "$inspection_tmp/$case_file.json" >/dev/null 2>&1; then
      log "self-test FAIL: inspection mutation accepted: $case_file"
      fail=1
    fi
  done
  rm -rf -- "$inspection_tmp"
  local schema_tmp
  schema_tmp="$(mktemp -d "${TMPDIR:-/tmp}/sgmse-validation-selftest.XXXXXX")"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
    "$schema_tmp" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
artifact = root / "prepared.safetensors"
artifact.write_bytes(b"synthetic-prepared-artifact")
payload = {
    "format": "vokra-sgmse-voicebank-prepared-v1",
    "repository": "speechbrain/sgmse-voicebank",
    "model_revision": "8f4ff7b65284c49492a43349b8106e094ac0d365",
    "source_repository": "https://github.com/sp-uhh/sgmse.git",
    "source_revision": "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e",
    "checkpoint_filename": "score_model_ema.ckpt",
    "checkpoint_size": 262593305,
    "checkpoint_sha256": "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3",
    "prepared_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "tensor_count": 1,
    "typed_manifest_sha256": "0" * 64,
    "tensor_rows": [{"name": "synthetic", "dtype": "torch.float32", "shape": [1]}],
}
(root / "prepared.manifest.json").write_text(json.dumps(payload), encoding="utf-8")
PY
  validate_prepared_sidecar "$schema_tmp/prepared.safetensors" "$schema_tmp/prepared.manifest.json" >/dev/null || fail=1
  rm -rf -- "$schema_tmp"
  (( fail == 0 )) || return 1
  log 'self-test PASS (model-free; no VAST mutation performed)'
}

while (($#)); do
  case "$1" in
    --self-test)
      (( SELF_TEST == 0 )) || die 'duplicate --self-test'
      SELF_TEST=1
      shift
      ;;
    --vokra-commit)
      [[ $# -ge 2 ]] || die '--vokra-commit requires a 40-hex commit'
      [[ -z "$VOKRA_COMMIT" ]] || die 'duplicate --vokra-commit'
      VOKRA_COMMIT="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die '--work-dir requires an absent directory'
      [[ -z "$WORK_DIR" ]] || die 'duplicate --work-dir'
      WORK_DIR="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if (( SELF_TEST )); then
  [[ -z "$VOKRA_COMMIT$WORK_DIR" ]] || die '--self-test accepts no arguments'
  self_test
  exit $?
fi

[[ "$VOKRA_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die '--vokra-commit must be a lowercase 40-hex commit'
[[ "$VOKRA_ROOT" == /* && -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ "$(uname -s)" == Linux ]] || die 'SGMSE validation is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'

mem_kib="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)"
[[ "${mem_kib:-0}" -ge "$MIN_VAST_MEM_KIB" ]] || die 'VAST host has less than 128 GiB RAM'
free_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR==2 {print $4}')"
[[ "${free_kib:-0}" -ge "$MIN_FREE_DISK_KIB" ]] || die 'VAST checkout filesystem has less than 32 GiB free'
for command in git uv cargo sha256sum awk df findmnt grep sort tee; do
  command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"
done
for runner in "$INSPECTION_RUNNER" "$PREPARE_RUNNER" "$SCORE_REFERENCE_RUNNER" "$SCORE_PARITY_RUNNER" "$ENHANCEMENT_RUNNER"; do
  [[ -x "$runner" ]] || die "runner is missing or not executable: $runner"
done

require_clean_commit
cd "$VOKRA_ROOT"
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="/workspace/vokra-sgmse-validation-$VOKRA_COMMIT"
fi
[[ "$WORK_DIR" == /* ]] || die '--work-dir must be absolute'
[[ ! -e "$WORK_DIR" && ! -L "$WORK_DIR" ]] || die '--work-dir must be absent (no-clobber)'
work_parent="$(dirname "$WORK_DIR")"
[[ -d "$work_parent" && ! -L "$work_parent" ]] || die 'work-dir parent must be an existing real directory'

INSPECTION_DIR="$WORK_DIR/inspection"
PREPARED_DIR="$WORK_DIR/prepared"
CONVERSION_DIR="$WORK_DIR/conversion"
SCORE_REFERENCE_DIR="$WORK_DIR/score-reference"
ENHANCEMENT_REFERENCE_DIR="$WORK_DIR/enhancement-reference"
LOG_DIR="$WORK_DIR/logs"
NATIVE_ROOT="/dev/shm/vokra-sgmse-validation-$VOKRA_COMMIT"
SCORE_NATIVE_DIR="$NATIVE_ROOT/score-native"
ENHANCEMENT_NATIVE_DIR="$NATIVE_ROOT/enhancement-native"
SUMMARY="$WORK_DIR/summary.json"
LEDGER="$WORK_DIR/resolution-ledger.json"
INPUT_WAV="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
CONVERTER="$VOKRA_ROOT/target/release/vokra-convert"
PREPARED="$PREPARED_DIR/sgmse_voicebank.safetensors"
PREPARED_MANIFEST="$PREPARED_DIR/sgmse_voicebank.manifest.json"
GGUF="$CONVERSION_DIR/sgmse-voicebank.gguf"

[[ -f "$INPUT_WAV" && ! -L "$INPUT_WAV" ]] || die 'fixed 16 kHz input fixture is missing or symlinked'
[[ ! -e "$NATIVE_ROOT" && ! -L "$NATIVE_ROOT" ]] || die 'native tmpfs work root must be absent (no-clobber)'
[[ "$(findmnt -T /dev/shm -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die '/dev/shm is not tmpfs'

mkdir "$WORK_DIR"
mkdir "$LOG_DIR"
mkdir "$NATIVE_ROOT"
cleanup() {
  rm -rf -- "$NATIVE_ROOT" 2>/dev/null || true
}
trap cleanup EXIT

run_inspection_stage "$INSPECTION_RUNNER" "$LOG_DIR/inspection.log" \
  "$INSPECTION_DIR/evidence/sgmse_voicebank_manifest.json" --work-dir "$INSPECTION_DIR"

run_logged preparation bash "$PREPARE_RUNNER" \
  --input-dir "$INSPECTION_DIR/hf" --output-dir "$PREPARED_DIR"
[[ -s "$PREPARED" && -s "$PREPARED_MANIFEST" ]] || die 'prepared artifact or sidecar is missing'

validate_prepared_sidecar "$PREPARED" "$PREPARED_MANIFEST" >/dev/null

mkdir "$CONVERSION_DIR"
run_logged build-converter cargo build --locked --release -p vokra-convert
[[ -x "$CONVERTER" ]] || die 'vokra-convert release binary is missing'
run_logged strict-conversion "$CONVERTER" --model sgmse-voicebank \
  --input "$PREPARED" --license apache-2.0 --output "$GGUF"
[[ -s "$GGUF" && ! -L "$GGUF" ]] || die 'strict GGUF conversion output is missing'
GGUF_SHA256="$(sha256_file "$GGUF")"
log "STRICT_BIND_PASS gguf_sha256=$GGUF_SHA256 license=apache-2.0"

run_logged score-reference bash "$SCORE_REFERENCE_RUNNER" \
  --inspection-dir "$INSPECTION_DIR" --output-dir "$SCORE_REFERENCE_DIR"
run_logged score-parity bash "$SCORE_PARITY_RUNNER" \
  --gguf "$GGUF" --gguf-sha256 "$GGUF_SHA256" \
  --reference-dir "$SCORE_REFERENCE_DIR" --native-output-dir "$SCORE_NATIVE_DIR"

run_logged enhancement-reference bash "$ENHANCEMENT_RUNNER" --generate-reference \
  --inspection-dir "$INSPECTION_DIR" --reference-dir "$ENHANCEMENT_REFERENCE_DIR"
run_logged enhancement-parity bash "$ENHANCEMENT_RUNNER" \
  --gguf "$GGUF" --gguf-sha256 "$GGUF_SHA256" \
  --reference-dir "$ENHANCEMENT_REFERENCE_DIR" --native-output-dir "$ENHANCEMENT_NATIVE_DIR"

run_logged metadata cargo metadata --locked --no-deps --format-version 1
run_logged package cargo test --locked -p vokra-convert --lib
run_logged workspace cargo test --locked --workspace --all-targets
run_logged clippy cargo clippy --locked --workspace --all-targets -- -D warnings
run_logged deny cargo deny check licenses advisories bans
run_logged audit cargo audit
run_logged static-fmt cargo fmt --all -- --check
run_logged static-forbidden bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
run_logged static-zero-deps bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
run_logged static-bound-arch bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
run_logged static-fixture-pins bash "$VOKRA_ROOT/scripts/check-fixture-eol-pins.sh"
run_logged static-dynamic-load bash "$VOKRA_ROOT/scripts/check-no-dynamic-load.sh"

require_clean_commit
write_resolution_ledger
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}" uv run --no-project --offline --python 3.12 python - \
  "$SUMMARY" "$VOKRA_COMMIT" "$INPUT_WAV" "$INSPECTION_DIR" "$PREPARED" \
  "$PREPARED_MANIFEST" "$GGUF" "$SCORE_REFERENCE_DIR" "$SCORE_NATIVE_DIR" \
  "$ENHANCEMENT_REFERENCE_DIR" "$ENHANCEMENT_NATIVE_DIR" "$LEDGER" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

summary, commit, input_wav, inspection, prepared, prepared_manifest, gguf, score_ref, score_native, enhancement_ref, enhancement_native, ledger = map(pathlib.Path, sys.argv[1:])
commit = str(commit)

def evidence(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))

inspection_manifest = inspection / "evidence/sgmse_voicebank_manifest.json"
score_manifest = score_ref / "manifest.json"
enhancement_manifest = enhancement_ref / "manifest.json"
for path in (inspection_manifest, prepared, prepared_manifest, gguf, score_manifest, enhancement_manifest, ledger):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing authenticated output: {path}")
prepared_data = manifest(prepared_manifest)
expected_sidecar_keys = {
    "format", "repository", "model_revision", "source_repository", "source_revision",
    "checkpoint_filename", "checkpoint_size", "checkpoint_sha256", "prepared_sha256",
    "tensor_count", "typed_manifest_sha256", "tensor_rows",
}
if set(prepared_data) != expected_sidecar_keys:
    raise SystemExit("prepared sidecar schema drifted")
if prepared_data.get("prepared_sha256") != evidence(prepared)["sha256"]:
    raise SystemExit("prepared hash is not bound")
score_data = manifest(score_manifest)
enhancement_data = manifest(enhancement_manifest)
if score_data.get("status") != "REFERENCE_COMPLETE_NO_UPLOAD" or score_data.get("publication") != "NO_UPLOAD":
    raise SystemExit("score reference is not complete/no-upload")
if enhancement_data.get("status") != "REFERENCE_COMPLETE_NO_UPLOAD" or enhancement_data.get("publication") != "NO_UPLOAD":
    raise SystemExit("enhancement reference is not complete/no-upload")
if not pathlib.Path(input_wav).is_file():
    raise SystemExit("input fixture disappeared")
payloads = {
    "inspection_manifest": evidence(inspection_manifest),
    "input_fixture": evidence(input_wav),
    "prepared": evidence(prepared),
    "prepared_manifest": evidence(prepared_manifest),
    "gguf": evidence(gguf),
    "score_reference_manifest": evidence(score_manifest),
    "enhancement_reference_manifest": evidence(enhancement_manifest),
    "resolution_ledger": evidence(ledger),
}
for name, directory in (("score_native", score_native), ("enhancement_native", enhancement_native)):
    files = sorted(path for path in directory.iterdir() if path.is_file() and not path.is_symlink())
    if not files:
        raise SystemExit(f"native output is empty: {directory}")
    payloads[name] = {path.name: evidence(path) for path in files}
data = {
    "format": "vokra-sgmse-validation-v1",
    "status": "SGMSE_VALIDATION_COMPLETE_NO_UPLOAD",
    "publication": "NO_UPLOAD",
    "vokra_commit": commit,
    "clean_checkout": True,
    "artifacts": payloads,
    "reference_status": {"score": score_data["status"], "enhancement": enhancement_data["status"]},
    "gates": {name: "PASS" for name in ("metadata", "package", "workspace", "clippy", "deny", "audit", "static")},
    "execution": {"model_execution": "VAST_ONLY", "local_model_execution": "NOT_RUN", "upload": "NO_UPLOAD"},
}
temporary = summary.with_name(f".{summary.name}.{os.getpid()}.tmp")
if summary.exists() or summary.is_symlink():
    raise SystemExit("summary already exists")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, summary)
temporary.unlink()
print(f"summary={summary}")
PY

log "SGMSE validation complete: summary=$SUMMARY status=SGMSE_VALIDATION_COMPLETE_NO_UPLOAD"
log 'Transfer only small manifests/logs and destroy the disposable VAST instance immediately; no upload/publication path exists.'
