#!/usr/bin/env bash
# Real-weight SpeechBrain VoxLingua107 Metal measurement on a disposable
# remote Apple Silicon host. Inputs are staged by the VAST workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="$(printenv VOKRA_ROOT 2>/dev/null || true)"
if [[ -z "$VOKRA_ROOT" ]]; then VOKRA_ROOT="$DEFAULT_ROOT"; fi
LANG_ID_PROJECT="$VOKRA_ROOT/tools/parity/speechbrain_lang_id"
LICENSE_GATE="$LANG_ID_PROJECT/preflight_gate.py"
LICENSE_MANIFEST="$LANG_ID_PROJECT/license_gate_manifest.json"

MODEL_KIND="lang-id-voxlingua107"
UPSTREAM_REPO="speechbrain/lang-id-voxlingua107-ecapa"
UPSTREAM_REVISION="0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
GGUF_ENV="VOKRA_LANG_ID_GGUF"
REFERENCE_DIR_ENV="VOKRA_LANG_ID_REFERENCE_DIR"
PARITY_TEST="measure_metal_against_cpu_and_independent_speechbrain"
PARITY_TEST_FILE="crates/vokra-models/tests/parity_speechbrain_lang_id_real.rs"
REFERENCE_FILES=(manifest.json pcm.f32.bin features.f32.bin embedding.f32.bin scores.f32.bin labels.json)
MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=20000000
FIXTURE_BYTES=352078
FIXTURE_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
CHECKPOINT_EMBEDDING_SHA256="ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a"
CHECKPOINT_CLASSIFIER_SHA256="a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc"
CHECKPOINT_LABELS_SHA256="9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f"

log() { printf '[speechbrain-lang-id-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-speechbrain-lang-id.sh \
  --expected-head <lowercase-40-hex> \
  --gguf <vast-generated-lang-id-voxlingua107.gguf> \
  --reference <vast-independent-reference-dir> \
  --gguf-sha256 <lowercase-sha256> \
  --reference-manifest-sha256 <lowercase-sha256> \
  --approval-evidence <regular-json-file> \
  --approval-evidence-sha256 <lowercase-sha256> \
  --transfer-manifest <regular-transfer-manifest> \
  --transfer-manifest-sha256 <lowercase-sha256> \
  --evidence-dir <empty-dir>
       apple-silicon-speechbrain-lang-id.sh --self-test

Runs the exact ignored SpeechBrain VoxLingua107 CPU/Metal measurement on a
disposable Darwin/arm64 host using only VAST-generated inputs. It requires
VOKRA_REMOTE_APPLE_SILICON=1, a clean checkout, at least 32 GB RAM, free disk,
and the Xcode Metal compiler. Bounds remain unset: this worker reports
MEASURED_NOT_GATED. It does not download, convert, upload, publish, push, or
delete model data.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_transfer_manifest() {
  local manifest="$1" manifest_sha="$2" expected_head="$3" gguf_sha="$4" reference_manifest_sha="$5" approval_sha="$6"
  require_file "transfer manifest" "$manifest"
  [[ "$manifest_sha" =~ ^[0-9a-f]{64}$ ]] || { die 'transfer manifest SHA-256 must be lowercase 64-hex'; return 2; }
  [[ "$(sha256_file "$manifest")" == "$manifest_sha" ]] || { die 'transfer manifest SHA-256 differs from VAST evidence'; return 2; }
  awk -F= '
    BEGIN { allowed["schema"]; allowed["expected_head"]; allowed["gguf_sha256"]; allowed["reference_manifest_sha256"]; allowed["approval_evidence_sha256"]; allowed["publication"]; }
    NF != 2 || !($1 in allowed) { exit 2 }
    { count[$1]++ }
    END { for (key in allowed) if (count[key] != 1) exit 3 }
  ' "$manifest" || { die 'transfer manifest schema is not exact'; return 2; }
  grep -Fxq 'schema=speechbrain-lang-id-apple-transfer-v1' "$manifest" || { die 'transfer manifest schema version differs'; return 2; }
  grep -Fxq "expected_head=$expected_head" "$manifest" || { die 'transfer manifest expected HEAD differs'; return 2; }
  grep -Fxq "gguf_sha256=$gguf_sha" "$manifest" || { die 'transfer manifest GGUF SHA-256 differs'; return 2; }
  grep -Fxq "reference_manifest_sha256=$reference_manifest_sha" "$manifest" || { die 'transfer manifest reference SHA-256 differs'; return 2; }
  grep -Fxq "approval_evidence_sha256=$approval_sha" "$manifest" || { die 'transfer manifest approval SHA-256 differs'; return 2; }
  grep -Fxq 'publication=NO_UPLOAD' "$manifest" || { die 'transfer manifest publication policy differs'; return 2; }
}

require_clean_expected_head() {
  local expected_head="$1" actual_head
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head must be exactly 40 lowercase hexadecimal characters'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" ]] || { die 'Apple checkout is missing .git'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'Apple checkout must be clean'; return 2; }
  actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual_head" == "$expected_head" ]] || { die "checkout HEAD $actual_head does not match expected $expected_head"; return 2; }
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, symlinked, or empty: $path"
}

require_absent_evidence_directory() {
  local directory="$1"
  [[ ! -e "$directory" && ! -L "$directory" ]] || die "evidence directory must be absent: $directory"
  mkdir -p "$directory"
}

canonical_candidate() {
  local value="$1" suffix='' parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"; [[ -n "$value" ]] || { die 'path is empty'; return 2; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || { die "path contains a symlink ancestor: $parent"; return 2; }
    parent="$(dirname "$parent")"
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"; suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die 'path has no canonical parent'; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die 'path parent is not a real directory'; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

require_disjoint_evidence() {
  local evidence="$1" gguf="$2" reference="$3" root="$4" approval="$5" transfer_manifest="$6" evidence_real gguf_real reference_real root_real approval_real transfer_manifest_real
  evidence_real="$(canonical_candidate "$evidence")" || return 2
  gguf_real="$(canonical_candidate "$gguf")" || return 2
  reference_real="$(canonical_candidate "$reference")" || return 2
  root_real="$(canonical_candidate "$root")" || return 2
  approval_real="$(canonical_candidate "$approval")" || return 2
  transfer_manifest_real="$(canonical_candidate "$transfer_manifest")" || return 2
  [[ "$evidence_real" != "$root_real" && "$evidence_real" != "$gguf_real" && "$evidence_real" != "$reference_real" && "$evidence_real" != "$approval_real" && "$evidence_real" != "$transfer_manifest_real" ]] || { die "evidence path aliases checkout or input"; return 2; }
  case "$evidence_real/" in
    "$reference_real/"*|"$root_real/"*|"$approval_real/"*|"$transfer_manifest_real/"*) die "evidence path overlaps an input, approval, or transfer manifest"; return 2 ;;
  esac
  case "$reference_real/" in "$evidence_real/"*) die "reference path overlaps evidence"; return 2 ;; esac
  case "$root_real/" in "$evidence_real/"*) die "checkout path overlaps evidence"; return 2 ;; esac
  case "$approval_real/" in "$evidence_real/"*) die "approval path overlaps evidence"; return 2 ;; esac
  case "$transfer_manifest_real/" in "$evidence_real/"*) die "transfer manifest path overlaps evidence"; return 2 ;; esac
}

license_preflight() {
  local approval="$1" gate_args=(--lock "$LANG_ID_PROJECT/uv.lock" --project "$LANG_ID_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST")
  [[ -f "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" ]] || { die "SpeechBrain Lang-ID gate/manifest is missing"; return 2; }
  [[ -f "$approval" && ! -L "$approval" ]] || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  gate_args+=(--approval "$approval")
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${gate_args[@]}"; then
    die 'SpeechBrain Lang-ID preflight gate rejected the manifest or approval evidence'
    return 2
  fi
}

require_reference() {
  local directory="$1" entry name
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference is not a regular directory: $directory"
  for entry in "$directory"/* "$directory"/.[!.]*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="${entry##*/}"
    case " ${REFERENCE_FILES[*]} " in *" $name "*) ;; *) die "reference contains unexpected entry: $name";; esac
    [[ -f "$entry" && ! -L "$entry" ]] || die "reference entry is not a regular file: $name"
  done
  for name in "${REFERENCE_FILES[@]}"; do require_file "reference $name" "$directory/$name"; done
  command -v uv >/dev/null 2>&1 || die 'uv is required for strict reference JSON validation'
  local lock_path="$VOKRA_ROOT/tools/parity/speechbrain_lang_id/uv.lock"
  require_file "SpeechBrain Lang-ID lock" "$lock_path"
  uv run --no-cache --no-project --offline --python 3.12 python - "$directory" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$FIXTURE_BYTES" "$FIXTURE_SHA256" "$lock_path" "$CHECKPOINT_EMBEDDING_SHA256" "$CHECKPOINT_CLASSIFIER_SHA256" "$CHECKPOINT_LABELS_SHA256" <<'PY' || { die 'reference JSON validation failed'; return 2; }
import hashlib, json, os, sys, tomllib
from pathlib import Path
root, source, revision, fixture_bytes, fixture_sha256, lock_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], Path(sys.argv[6])
expected_checkpoint_hashes = {
    "embedding_model.ckpt": sys.argv[7],
    "classifier.ckpt": sys.argv[8],
    "label_encoder.txt": sys.argv[9],
}
def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
data = json.loads((root / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
if data.get("format") != "vokra-speechbrain-lang-id-reference-v1" or data.get("source") != source or data.get("revision") != revision:
    raise SystemExit("reference source/revision/format mismatch")
expected_manifest_keys = {
    "artifact_bytes", "artifact_sha256", "best_index", "best_label", "best_score",
    "checkpoint_sha256", "device", "embedding_shape", "feature_shape", "format",
    "numpy", "pcm_samples", "raw_feature_shape", "revision", "sample_rate",
    "score_shape", "source", "speechbrain", "torch", "torchaudio", "wav_bytes", "wav_sha256",
}
if set(data) != expected_manifest_keys:
    raise SystemExit("reference manifest has missing or extra top-level keys")
if data.get("sample_rate") != 16000 or data.get("device") != "cpu" or not isinstance(data.get("feature_shape"), list) or len(data["feature_shape"]) != 3 or data["feature_shape"][0] != 1 or not isinstance(data["feature_shape"][1], int) or data["feature_shape"][1] <= 0 or data["feature_shape"][2] != 60 or data.get("embedding_shape") != [1,1,256] or data.get("score_shape") != [1,107] or data.get("raw_feature_shape") != data.get("feature_shape"):
    raise SystemExit("reference shape/sample-rate contract mismatch")
if data.get("wav_bytes") != fixture_bytes or data.get("wav_sha256") != fixture_sha256:
    raise SystemExit("reference fixture identity mismatch")
lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
versions = {item.get("name"): item.get("version") for item in lock.get("package", []) if isinstance(item, dict)}
for key in ("speechbrain", "torch", "torchaudio", "numpy"):
    expected = versions.get(key)
    if not isinstance(expected, str) or data.get(key) != expected: raise SystemExit(f"reference {key} version mismatch")
labels = json.loads((root / "labels.json").read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
if not isinstance(labels, list) or len(labels) != 107 or any(not isinstance(x, str) or not x for x in labels): raise SystemExit("reference labels are not exactly 107 nonempty strings")
checkpoint_hashes = data.get("checkpoint_sha256")
if checkpoint_hashes != expected_checkpoint_hashes: raise SystemExit("reference checkpoint identity map differs from the audited upstream payloads")
if any(not isinstance(x, str) or not __import__("re").fullmatch(r"[0-9a-f]{64}", x) for x in checkpoint_hashes.values()): raise SystemExit("reference checkpoint identity map is not exact")
hashes, sizes = data.get("artifact_sha256"), data.get("artifact_bytes")
expected_files = ("pcm.f32.bin", "features.f32.bin", "embedding.f32.bin", "scores.f32.bin", "labels.json")
if set(hashes or ()) != set(expected_files) or set(sizes or ()) != set(expected_files): raise SystemExit("reference artifact identity map is not exact")
for name in expected_files:
    payload = root / name
    if sizes[name] != payload.stat().st_size or hashes[name] != hashlib.sha256(payload.read_bytes()).hexdigest(): raise SystemExit(f"reference artifact identity mismatch: {name}")
expected_sizes = {
    "pcm.f32.bin": int(data["pcm_samples"]) * 4,
    "features.f32.bin": int(data["feature_shape"][1]) * int(data["feature_shape"][2]) * 4,
    "embedding.f32.bin": 256 * 4,
    "scores.f32.bin": 107 * 4,
}
if not isinstance(data["pcm_samples"], int) or data["pcm_samples"] <= 0 or any(sizes[name] != size for name, size in expected_sizes.items()): raise SystemExit("reference artifact byte counts do not follow exact shapes")
if not isinstance(data.get("pcm_samples"), int) or data["pcm_samples"] <= 0: raise SystemExit("reference PCM length is invalid")
PY
}

require_test_evidence() {
  local path="$1" tests named result result_lines metal metal_lines
  tests="$(grep -Ec '^test [^ ]+ \.\.\. ' "$path" || true)"
  named="$(grep -Ec '^test measure_metal_against_cpu_and_independent_speechbrain \.\.\. ok$' "$path" || true)"
  result="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)"
  result_lines="$(grep -Ec '^test result:' "$path" || true)"
  metal="$(grep -Ec '^LANG_ID_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED$' "$path" || true)"
  metal_lines="$(grep -Ec '^LANG_ID_MEASUREMENT_ONLY backend=metal ' "$path" || true)"
  [[ "$tests" == 1 && "$named" == 1 && "$result" == 1 && "$result_lines" == 1 && "$metal" == 1 && "$metal_lines" == 1 ]] \
    || die 'Lang-ID evidence requires exactly one named test/result/Metal sentinel'
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "$(printenv VOKRA_REMOTE_APPLE_SILICON 2>/dev/null || true)" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "remote Metal measurement requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "remote Metal measurement requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the exact 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the 20-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers system_profiler xcrun uv; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra checkout"
  [[ -f "$VOKRA_ROOT/$PARITY_TEST_FILE" ]] \
    || die "Lang-ID parity test is missing: $VOKRA_ROOT/$PARITY_TEST_FILE"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "machine=$(uname -m)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPHardwareDataType
    system_profiler SPDisplaysDataType
  } > "$output"
}

run_self_test() (
  local script_path="$0" required fail=0 temporary
  script_path="$(cd "$(dirname "$script_path")" && pwd)/$(basename "$script_path")"
  for required in "$MODEL_KIND" "$GGUF_ENV" "$REFERENCE_DIR_ENV" "$PARITY_TEST" \
    "$PARITY_TEST_FILE" "${REFERENCE_FILES[@]}" "MIN_MEMORY_BYTES=32000000000" \
    "MIN_FREE_DISK_KIB=20000000" "VOKRA_REMOTE_APPLE_SILICON=1" "Darwin" \
    "arm64" "hw.memsize" "xcrun -f metal" \
    "git status --porcelain --untracked-files=all" \
    "cargo test --manifest-path" "-p vokra-models --features metal" \
    "--test parity_speechbrain_lang_id_real" "test result: ok. 1 passed" \
    "--expected-head" "--gguf-sha256" "--reference-manifest-sha256" "--approval-evidence" "--approval-evidence-sha256" "--transfer-manifest" "--transfer-manifest-sha256" "APPLE_LANG_ID_APPROVAL_EVIDENCE" "speechbrain-lang-id-apple-transfer-v1" "publication=NO_UPLOAD" "LANG_ID_MEASUREMENT_ONLY backend=metal" "--offline --locked" "--test-threads=1" \
    "LANG_ID_MEASUREMENT_ONLY backend=metal" "MEASURED_NOT_GATED" "preflight_gate.py" "--manifest" \
    "$CHECKPOINT_EMBEDDING_SHA256" "$CHECKPOINT_CLASSIFIER_SHA256" "$CHECKPOINT_LABELS_SHA256"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: missing contract token: $required"
      fail=1
    fi
  done
  if grep -En -- '(^|[[:space:]])(curl|wget|vokra-cli[[:space:]]+convert|git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: download/conversion/publication command found"
    fail=1
  fi
  if grep -En '^[[:space:]]*(echo[[:space:]]+)?(verdict|parity_status)=PASS' "$script_path" >/dev/null; then
    log "self-test FAIL: false PASS verdict found"
    fail=1
  fi
  if grep -En -- '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python command found"
    fail=1
  fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then log 'self-test FAIL: duplicate --self-test accepted'; fail=1; fi
  if "$script_path" --self-test trailing >/dev/null 2>&1; then log 'self-test FAIL: trailing self-test argument accepted'; fail=1; fi
  if "$script_path" --gguf >/dev/null 2>&1; then log 'self-test FAIL: bare --gguf accepted'; fail=1; fi
  if "$script_path" --gguf "" >/dev/null 2>&1; then log 'self-test FAIL: empty --gguf accepted'; fail=1; fi
  if "$script_path" --gguf --reference ref >/dev/null 2>&1; then log 'self-test FAIL: option used as --gguf value accepted'; fail=1; fi
  if "$script_path" --gguf one --gguf two >/dev/null 2>&1; then log 'self-test FAIL: duplicate --gguf accepted'; fail=1; fi
  if "$script_path" --unknown >/dev/null 2>&1; then log 'self-test FAIL: unknown option accepted'; fail=1; fi
  if "$script_path" --approval-evidence >/dev/null 2>&1; then log 'self-test FAIL: missing approval value accepted'; fail=1; fi
  if "$script_path" --approval-evidence "" >/dev/null 2>&1; then log 'self-test FAIL: empty approval value accepted'; fail=1; fi
  if "$script_path" --approval-evidence --gguf x >/dev/null 2>&1; then log 'self-test FAIL: option used as approval value accepted'; fail=1; fi
  if "$script_path" --approval-evidence one --approval-evidence two >/dev/null 2>&1; then log 'self-test FAIL: duplicate approval accepted'; fail=1; fi
  if "$script_path" --gguf one --reference ref --gguf-sha256 "$(printf 'a%.0s' {1..64})" --reference-manifest-sha256 "$(printf 'b%.0s' {1..64})" --approval-evidence approval --evidence-dir evidence >/dev/null 2>&1; then log 'self-test FAIL: missing expected-head accepted'; fail=1; fi
  if "$script_path" --expected-head >/dev/null 2>&1; then log 'self-test FAIL: missing expected-head value accepted'; fail=1; fi
  if "$script_path" --expected-head "$(printf '%040d' 1)" --expected-head "$(printf '%040d' 2)" >/dev/null 2>&1; then log 'self-test FAIL: duplicate expected-head accepted'; fail=1; fi
  if "$script_path" --approval-evidence-sha256 >/dev/null 2>&1; then log 'self-test FAIL: missing approval SHA value accepted'; fail=1; fi
  if "$script_path" --transfer-manifest-sha256 >/dev/null 2>&1; then log 'self-test FAIL: missing transfer SHA value accepted'; fail=1; fi
  if "$script_path" --approval-evidence-sha256 "$(printf '%064d' 1)" --approval-evidence-sha256 "$(printf '%064d' 2)" >/dev/null 2>&1; then log 'self-test FAIL: duplicate approval SHA accepted'; fail=1; fi
  if "$script_path" --transfer-manifest-sha256 "$(printf '%064d' 1)" --transfer-manifest-sha256 "$(printf '%064d' 2)" >/dev/null 2>&1; then log 'self-test FAIL: duplicate transfer SHA accepted'; fail=1; fi
  temporary="$(cd -P "$(mktemp -d)" && pwd -P)"
  trap 'rm -rf "$temporary"' EXIT
  mkdir -p "$temporary/root/reference/nested" "$temporary/root/nested" "$temporary/parent"; : > "$temporary/root/gguf"
  printf approval > "$temporary/approval"
  local test_head test_gguf_sha test_reference_sha test_approval_sha test_transfer
  test_head="$(printf '%040d' 1)"; test_gguf_sha="$(printf '%064d' 0)"; test_reference_sha="$(printf '%064d' 0)"; test_approval_sha="$(printf '%064d' 1)"; test_transfer="$temporary/root/transfer-manifest"
  printf 'schema=speechbrain-lang-id-apple-transfer-v1\nexpected_head=%s\ngguf_sha256=%s\nreference_manifest_sha256=%s\napproval_evidence_sha256=%s\npublication=NO_UPLOAD\n' "$test_head" "$test_gguf_sha" "$test_reference_sha" "$test_approval_sha" > "$test_transfer"
  local test_transfer_sha
  test_transfer_sha="$(sha256_file "$test_transfer")"
  require_transfer_manifest "$test_transfer" "$test_transfer_sha" "$test_head" "$test_gguf_sha" "$test_reference_sha" "$test_approval_sha" || { log 'self-test FAIL: valid transfer manifest rejected'; fail=1; }
  sed -i.bak 's/publication=NO_UPLOAD/publication=UPLOAD/' "$test_transfer"; rm -f "$test_transfer.bak"
  if require_transfer_manifest "$test_transfer" "$(sha256_file "$test_transfer")" "$test_head" "$test_gguf_sha" "$test_reference_sha" "$test_approval_sha"; then log 'self-test FAIL: transfer publication policy accepted'; fail=1; fi
  printf 'schema=speechbrain-lang-id-apple-transfer-v1\nexpected_head=%s\ngguf_sha256=%s\nreference_manifest_sha256=%s\napproval_evidence_sha256=%s\npublication=NO_UPLOAD\nextra=value\n' "$test_head" "$test_gguf_sha" "$test_reference_sha" "$test_approval_sha" > "$test_transfer"; if require_transfer_manifest "$test_transfer" "$(sha256_file "$test_transfer")" "$test_head" "$test_gguf_sha" "$test_reference_sha" "$test_approval_sha"; then log 'self-test FAIL: extra transfer key accepted'; fail=1; fi
  if require_disjoint_evidence "$temporary/root/reference/nested/evidence" "$temporary/root/gguf" "$temporary/root/reference" "$temporary/root" "$temporary/approval" "$temporary/root/transfer-manifest"; then log 'self-test FAIL: evidence under reference accepted'; fail=1; fi
  if require_disjoint_evidence "$temporary/root/nested/evidence" "$temporary/root/gguf" "$temporary/root/reference" "$temporary/root" "$temporary/approval" "$temporary/root/transfer-manifest"; then log 'self-test FAIL: evidence under checkout accepted'; fail=1; fi
  ln -s "$temporary/root/reference" "$temporary/reference-link"
  if require_disjoint_evidence "$temporary/parent/evidence" "$temporary/root/gguf" "$temporary/reference-link" "$temporary/root" "$temporary/approval" "$temporary/root/transfer-manifest"; then log 'self-test FAIL: symlink reference accepted'; fail=1; fi
  ln -s "$temporary/root/gguf" "$temporary/gguf-link"
  if require_disjoint_evidence "$temporary/parent/evidence2" "$temporary/gguf-link" "$temporary/root/reference" "$temporary/root" "$temporary/approval" "$temporary/root/transfer-manifest"; then log 'self-test FAIL: symlink GGUF accepted'; fail=1; fi
  ln -s "$temporary/root/reference" "$temporary/evidence-link"
  if require_disjoint_evidence "$temporary/evidence-link" "$temporary/root/gguf" "$temporary/root/reference" "$temporary/root" "$temporary/approval" "$temporary/root/transfer-manifest"; then log 'self-test FAIL: symlink evidence accepted'; fail=1; fi
  mkdir -p "$temporary/root/real/existing"
  ln -s "$temporary/root/real" "$temporary/root/link"
  if require_disjoint_evidence "$temporary/root/link/existing/new-evidence" "$temporary/root/gguf" "$temporary/root/reference" "$temporary/root" "$temporary/approval" "$temporary/root/transfer-manifest"; then log 'self-test FAIL: evidence under symlink ancestor accepted'; fail=1; fi
  local fake_root fake_evidence
  fake_root="$(mktemp -d)"; fake_evidence="$fake_root/evidence"
  mkdir -p "$fake_root/tools/parity"
  cp -R "$VOKRA_ROOT/tools/parity/speechbrain_lang_id" "$fake_root/tools/parity/speechbrain_lang_id"
  if VOKRA_ROOT="$fake_root" "$script_path" --expected-head "$(printf '%040d' 1)" --approval-evidence "$fake_root/missing-approval" --gguf "$fake_root/gguf" --reference "$fake_root/reference" --gguf-sha256 "$(printf 'a%.0s' {1..64})" --reference-manifest-sha256 "$(printf 'b%.0s' {1..64})" --evidence-dir "$fake_evidence" >/dev/null 2>&1; then log 'self-test FAIL: invalid approval passed before host/input checks'; fail=1; fi
  [[ ! -e "$fake_evidence" ]] || { log 'self-test FAIL: invalid approval created evidence'; fail=1; }
  printf '{"status":"PENDING_REVIEW","status":"APPROVED"}' > "$fake_root/duplicate-approval.json"
  if VOKRA_ROOT="$fake_root" "$script_path" --expected-head "$(printf '%040d' 1)" --approval-evidence "$fake_root/duplicate-approval.json" --gguf "$fake_root/gguf" --reference "$fake_root/reference" --gguf-sha256 "$(printf 'a%.0s' {1..64})" --reference-manifest-sha256 "$(printf 'b%.0s' {1..64})" --evidence-dir "$fake_evidence" >/dev/null 2>&1; then log 'self-test FAIL: duplicate approval JSON passed'; fail=1; fi
  [[ ! -e "$fake_evidence" ]] || { log 'self-test FAIL: duplicate approval created evidence'; fail=1; }
  rm -rf "$fake_root"
  printf 'test measure_metal_against_cpu_and_independent_speechbrain ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nLANG_ID_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED\n' > "$temporary/valid.log"
  require_test_evidence "$temporary/valid.log" || { log 'self-test FAIL: valid evidence rejected'; fail=1; }
  cp "$temporary/valid.log" "$temporary/duplicate-result.log"
  printf 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n' >> "$temporary/duplicate-result.log"
  if require_test_evidence "$temporary/duplicate-result.log"; then log 'self-test FAIL: duplicate result accepted'; fail=1; fi
  cp "$temporary/valid.log" "$temporary/extra-test.log"
  printf 'test unrelated_smoke ... ok\n' >> "$temporary/extra-test.log"
  if require_test_evidence "$temporary/extra-test.log"; then log 'self-test FAIL: extra test accepted'; fail=1; fi
  cp "$temporary/valid.log" "$temporary/failed-test.log"
  sed -i.bak 's/measure_metal_against_cpu_and_independent_speechbrain \.\.\. ok/measure_metal_against_cpu_and_independent_speechbrain ... FAILED/' "$temporary/failed-test.log"
  rm -f "$temporary/failed-test.log.bak"
  if require_test_evidence "$temporary/failed-test.log"; then log 'self-test FAIL: failed named test accepted'; fail=1; fi
  cp "$temporary/valid.log" "$temporary/malformed-result.log"
  sed -i.bak 's/filtered out$/filtered out; unexpected/' "$temporary/malformed-result.log"
  rm -f "$temporary/malformed-result.log.bak"
  if require_test_evidence "$temporary/malformed-result.log"; then log 'self-test FAIL: malformed result accepted'; fail=1; fi
  cp "$temporary/valid.log" "$temporary/duplicate-sentinel.log"
  printf 'LANG_ID_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED\n' >> "$temporary/duplicate-sentinel.log"
  if require_test_evidence "$temporary/duplicate-sentinel.log"; then log 'self-test FAIL: duplicate sentinel accepted'; fail=1; fi
  [[ "$fail" -eq 0 ]] || return 1
  log "self-test OK (offline contract checks only)"
)

main() {
  local expected_head="" gguf="" reference="" gguf_sha="" reference_manifest_sha="" approval="" approval_sha="" transfer_manifest="" transfer_manifest_sha="" evidence_dir="" self_test=0
  local seen_expected_head=0 seen_gguf=0 seen_reference=0 seen_gguf_sha=0 seen_reference_manifest_sha=0 seen_approval=0 seen_approval_sha=0 seen_transfer_manifest=0 seen_transfer_manifest_sha=0 seen_evidence_dir=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --expected-head) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_expected_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; expected_head="$2"; seen_expected_head=1; shift 2 ;;
      --gguf) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf="$2"; seen_gguf=1; shift 2 ;;
      --reference) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference="$2"; seen_reference=1; shift 2 ;;
      --gguf-sha256) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_gguf_sha == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; gguf_sha="$2"; seen_gguf_sha=1; shift 2 ;;
      --reference-manifest-sha256) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_reference_manifest_sha == 0 )) || die 'duplicate --reference-manifest-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; reference_manifest_sha="$2"; seen_reference_manifest_sha=1; shift 2 ;;
      --approval-evidence) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; approval="$2"; seen_approval=1; shift 2 ;;
      --approval-evidence-sha256) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_approval_sha == 0 )) || die 'duplicate --approval-evidence-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; approval_sha="$2"; seen_approval_sha=1; shift 2 ;;
      --transfer-manifest) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_transfer_manifest == 0 )) || die 'duplicate --transfer-manifest'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; transfer_manifest="$2"; seen_transfer_manifest=1; shift 2 ;;
      --transfer-manifest-sha256) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_transfer_manifest_sha == 0 )) || die 'duplicate --transfer-manifest-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; transfer_manifest_sha="$2"; seen_transfer_manifest_sha=1; shift 2 ;;
      --evidence-dir) (( self_test == 0 )) || die '--self-test must be exclusive'; (( seen_evidence_dir == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }; evidence_dir="$2"; seen_evidence_dir=1; shift 2 ;;
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; self_test=1; seen_self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$expected_head$gguf$reference$gguf_sha$reference_manifest_sha$approval$approval_sha$transfer_manifest$transfer_manifest_sha$evidence_dir" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$expected_head" && -n "$gguf" && -n "$reference" && -n "$evidence_dir" && -n "$gguf_sha" && -n "$reference_manifest_sha" && -n "$approval" && -n "$approval_sha" && -n "$transfer_manifest" && -n "$transfer_manifest_sha" ]] \
    || { usage; die "all exact-head, input, approval, transfer-manifest, SHA-256 and evidence arguments are required"; }
  [[ "$gguf_sha" =~ ^[0-9a-f]{64}$ && "$reference_manifest_sha" =~ ^[0-9a-f]{64}$ && "$approval_sha" =~ ^[0-9a-f]{64}$ && "$transfer_manifest_sha" =~ ^[0-9a-f]{64}$ ]] \
    || die 'expected hashes must be lowercase 64-hex SHA-256 values'

  require_clean_expected_head "$expected_head"
  require_file "approval evidence" "$approval"
  [[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 differs from the external approval binding'
  require_transfer_manifest "$transfer_manifest" "$transfer_manifest_sha" "$expected_head" "$gguf_sha" "$reference_manifest_sha" "$approval_sha"
  license_preflight "$approval"
  require_disjoint_evidence "$evidence_dir" "$gguf" "$reference" "$VOKRA_ROOT" "$approval" "$transfer_manifest"
  require_remote_apple_host
  require_tooling
  require_file "VAST-generated Lang-ID GGUF" "$gguf"
  [[ "$(sha256_file "$gguf")" == "$gguf_sha" ]] || die 'GGUF SHA-256 differs from VAST evidence'
  [[ "$(sha256_file "$reference/manifest.json")" == "$reference_manifest_sha" ]] || die 'reference manifest SHA-256 differs from VAST evidence'
  require_reference "$reference"
  require_absent_evidence_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf_sha256=$(sha256_file "$gguf")"
    for name in "${REFERENCE_FILES[@]}"; do
      echo "reference_${name}_sha256=$(sha256_file "$reference/$name")"
    done
  } > "$evidence_dir/input-hashes.txt"

  log "running exact real-weight CPU/Metal Lang-ID measurement"
  local jobs
  jobs="$(printenv CARGO_BUILD_JOBS 2>/dev/null || true)"
  if [[ -z "$jobs" ]]; then jobs=2; fi
  env "$GGUF_ENV=$gguf" "$REFERENCE_DIR_ENV=$reference" \
    CARGO_BUILD_JOBS="$jobs" RUST_TEST_THREADS=1 \
    CARGO_NET_OFFLINE=true cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --offline --locked --release \
      -p vokra-models --features metal --test parity_speechbrain_lang_id_real \
      "$PARITY_TEST" -- --exact --nocapture --test-threads=1 2>&1 | tee "$evidence_dir/parity.log"

  require_test_evidence "$evidence_dir/parity.log"
  require_clean_expected_head "$expected_head"

  {
    echo "expected_head=$expected_head"
    echo "verdict=MEASURED_NOT_GATED"
    echo "parity_status=MEASURED_NOT_GATED"
    echo "numeric_bounds=UNSET"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference/manifest.json")"
    echo "test=$PARITY_TEST"
    echo "test_sentinel=LANG_ID_MEASUREMENT_ONLY backend=metal"
    echo "upload=NOT_PERFORMED"
    echo "publish=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "MEASURED_NOT_GATED: pull only $evidence_dir; remove staged inputs or destroy the remote worker"
}

main "$@"
