#!/usr/bin/env bash
# VAST/Linux-only real-weight SpeechBrain VoxLingua107 validation.
# This worker does not publish, upload, push, or claim numerical readiness.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="$(printenv VOKRA_ROOT 2>/dev/null || true)"
if [[ -z "$VOKRA_ROOT" ]]; then VOKRA_ROOT="$DEFAULT_ROOT"; fi
VOKRA_SCRATCH="$(printenv VOKRA_SCRATCH 2>/dev/null || true)"
if [[ -z "$VOKRA_SCRATCH" ]]; then VOKRA_SCRATCH="$HOME/scratchpad"; fi
LANG_ID_PROJECT="$VOKRA_ROOT/tools/parity/speechbrain_lang_id"
LICENSE_GATE="$LANG_ID_PROJECT/preflight_gate.py"
LICENSE_MANIFEST="$LANG_ID_PROJECT/license_gate_manifest.json"
MODEL_FREE_AUDIT="$LANG_ID_PROJECT/model_free_audit.py"

MODEL_KIND="lang-id-voxlingua107"
UPSTREAM_REPO="speechbrain/lang-id-voxlingua107-ecapa"
UPSTREAM_REVISION="0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
PREPARER="tools/parity/speechbrain_lang_id_prepare_checkpoint.py"
REFERENCE_DUMPER="tools/parity/speechbrain_lang_id_dump_reference.py"
REFERENCE_INPUT="tests/fixtures/audio/jfk-30s.wav"
PARITY_TEST="measure_cpu_against_independent_speechbrain"
PARITY_TEST_FILE="crates/vokra-models/tests/parity_speechbrain_lang_id_real.rs"
GGUF_ENV="VOKRA_LANG_ID_GGUF"
REFERENCE_DIR_ENV="VOKRA_LANG_ID_REFERENCE_DIR"
EXPECTED_N_MELS=60
EXPECTED_EMBEDDING_DIM=256
EXPECTED_CLASS_COUNT=107
FIXTURE_BYTES=352078
FIXTURE_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
# These names are recorded by the SpeechBrain runbook and reference worker.
CHECKPOINT_FILES=(embedding_model.ckpt classifier.ckpt label_encoder.txt)
SNAPSHOT_FILES=(embedding_model.ckpt classifier.ckpt label_encoder.txt hyperparams.yaml config.json)

MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[speechbrain-lang-id-vast] %s\n' "$*" >&2; }
step() { printf '\n[speechbrain-lang-id-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-speechbrain-lang-id-validation.sh --approval-evidence <regular-json-file>
       --approval-evidence-sha256 <lowercase-sha256> --expected-head <lowercase-40-hex>
       [--work-dir <absent-dir>]
       run-speechbrain-lang-id-validation.sh --model-free --expected-head <lowercase-40-hex>
       [--work-dir <absent-dir>]
       run-speechbrain-lang-id-validation.sh --self-test

VAST/Linux-only non-publishing VoxLingua107 worker. It uses the official
pinned SpeechBrain loader, prepares a strict Vokra checkpoint, dumps an
independent reference, converts GGUF, runs CPU measurement and CLI
classification smoke, and records hashes/manifests. Parity stays NOT_GATED
until numeric bounds are reviewed and Metal is measured separately.
Actual runs require Linux x86_64, VOKRA_PUBLISH_ON_VAST=1, exact 64 GiB RAM,
150 GB free disk, and a clean checkout. --self-test is offline and hermetic.
--model-free performs only the checkout/lock/source-identity/fixture audit and
emits a blocked, no-upload report; it never downloads or executes a model.
EOF
}

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

require_clean_expected_head() {
  local expected_head="$1" actual_head
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head must be exactly 40 lowercase hexadecimal characters'; return 2; }
  [[ -d "$VOKRA_ROOT/.git" ]] || { die 'VAST checkout is missing .git'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
  actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual_head" == "$expected_head" ]] || { die "checkout HEAD $actual_head does not match expected $expected_head"; return 2; }
}

expected_snapshot_bytes() {
  case "$1" in
    embedding_model.ckpt) printf '84474355\n' ;;
    classifier.ckpt) printf '762555\n' ;;
    label_encoder.txt) printf '2204\n' ;;
    hyperparams.yaml) printf '1519\n' ;;
    config.json) printf '51\n' ;;
    *) return 1 ;;
  esac
}

expected_snapshot_sha256() {
  case "$1" in
    embedding_model.ckpt) printf 'ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a\n' ;;
    classifier.ckpt) printf 'a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc\n' ;;
    label_encoder.txt) printf '9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f\n' ;;
    hyperparams.yaml) printf '88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f\n' ;;
    config.json) printf 'a861f8fbc2e23c0fc0823b3c0fd2b3d1e839563c2d4e3f9663a1237cce62bc89\n' ;;
    *) return 1 ;;
  esac
}

download_upstream_snapshot() {
  local output="$1"
  uv run --frozen --project "$LANG_ID_PROJECT" --python 3.12 python - \
    "$output" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "${SNAPSHOT_FILES[@]}" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

output, repo, revision, *allow_patterns = sys.argv[1:]
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=Path(output),
    allow_patterns=allow_patterns,
)
PY
}

verify_upstream_snapshot() {
  local directory="$1" entry name expected_bytes expected_sha actual_bytes actual_sha
  [[ -d "$directory" && ! -L "$directory" ]] || die "upstream snapshot is not a regular directory"
  for entry in "$directory"/* "$directory"/.[!.]*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="${entry##*/}"
    if [[ "$name" == ".cache" ]]; then
      [[ -d "$entry" && ! -L "$entry" ]] || die "snapshot transport cache is not a directory"
      continue
    fi
    case " ${SNAPSHOT_FILES[*]} " in *" $name "*) ;; *) die "snapshot contains unexpected entry: $name" ;; esac
    [[ -f "$entry" && ! -L "$entry" ]] || die "snapshot entry is not a regular file: $name"
  done
  for name in "${SNAPSHOT_FILES[@]}"; do
    entry="$directory/$name"
    [[ -f "$entry" && ! -L "$entry" && -s "$entry" ]] || die "snapshot payload is missing or empty: $name"
    expected_bytes="$(expected_snapshot_bytes "$name")"
    expected_sha="$(expected_snapshot_sha256 "$name")"
    [[ "$expected_bytes" =~ ^[1-9][0-9]*$ && "$expected_sha" =~ ^[0-9a-f]{64}$ ]] \
      || die "code-bound upstream identity is unresolved: $name"
    actual_bytes="$(wc -c < "$entry" | tr -d '[:space:]')"
    actual_sha="$(sha256_file "$entry")"
    [[ "$actual_bytes" == "$expected_bytes" && "$actual_sha" == "$expected_sha" ]] \
      || die "authenticated upstream identity mismatch: $name"
  done
}

require_cpu_test_evidence() {
  local path="$1" tests named result result_lines cpu cpu_lines
  tests="$(grep -Ec '^test [^ ]+ \.\.\. ' "$path" || true)"
  named="$(grep -Ec '^test measure_cpu_against_independent_speechbrain \.\.\. ok$' "$path" || true)"
  result="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)"
  result_lines="$(grep -Ec '^test result:' "$path" || true)"
  cpu="$(grep -Ec '^LANG_ID_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED$' "$path" || true)"
  cpu_lines="$(grep -Ec '^LANG_ID_MEASUREMENT_ONLY backend=cpu ' "$path" || true)"
  [[ "$tests" == 1 && "$named" == 1 && "$result" == 1 && "$result_lines" == 1 && "$cpu" == 1 && "$cpu_lines" == 1 ]] \
    || die 'Lang-ID CPU evidence requires exactly one named test/result/sentinel'
}

write_apple_args() {
  local output="$1" expected_head="$2" gguf_sha="$3" reference_manifest_sha="$4" approval_sha="$5" transfer_manifest_sha="$6"
  {
    printf '#!/usr/bin/env bash\nset -eu\n'
    printf '%s ' 'scripts/verify/apple-silicon-speechbrain-lang-id.sh'
    printf '%s ' --expected-head "$expected_head"
    printf '%s ' --gguf "'<VAST_LANG_ID_GGUF_PATH>'" --reference "'<VAST_LANG_ID_REFERENCE_DIR>'"
    printf '%s ' --gguf-sha256 "$gguf_sha" --reference-manifest-sha256 "$reference_manifest_sha"
    printf '%s ' --approval-evidence "'<APPLE_LANG_ID_APPROVAL_EVIDENCE>'" --approval-evidence-sha256 "$approval_sha"
    printf '%s ' --transfer-manifest "'<VAST_LANG_ID_TRANSFER_MANIFEST>'" --transfer-manifest-sha256 "$transfer_manifest_sha"
    printf '%s\n' --evidence-dir "'<APPLE_LANG_ID_EMPTY_EVIDENCE_DIR>'"
  } > "$output"
  chmod +x "$output"
}

write_transfer_manifest() {
  local output="$1" expected_head="$2" gguf_sha="$3" reference_manifest_sha="$4" approval_sha="$5"
  [[ ! -e "$output" && ! -L "$output" && ! -e "${output}.sha256" && ! -L "${output}.sha256" ]] || { die "transfer manifest output must be absent: $output"; return 2; }
  {
    printf 'schema=speechbrain-lang-id-apple-transfer-v1\n'
    printf 'expected_head=%s\n' "$expected_head"
    printf 'gguf_sha256=%s\n' "$gguf_sha"
    printf 'reference_manifest_sha256=%s\n' "$reference_manifest_sha"
    printf 'approval_evidence_sha256=%s\n' "$approval_sha"
    printf 'publication=NO_UPLOAD\n'
  } > "$output"
  sha256_file "$output" > "${output}.sha256"
}

license_preflight() {
  local approval="$1" gate_args=(--lock "$LANG_ID_PROJECT/uv.lock"
    --project "$LANG_ID_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST")
  [[ -f "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" ]] || { die "SpeechBrain Lang-ID gate/manifest is missing"; return 2; }
  [[ -f "$approval" && ! -L "$approval" ]] \
    || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  gate_args+=(--approval "$approval")
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" "${gate_args[@]}"; then
    die 'SpeechBrain Lang-ID preflight gate rejected the manifest or approval evidence'
    return 2
  fi
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

paths_overlap() {
  local left="${1%/}" right="${2%/}"
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

validate_work_dir() {
  local work="$1" approval="$2" canonical_work canonical_root canonical_project approval_real
  [[ ! -e "$work" && ! -L "$work" ]] || { die '--work-dir must be absent/nonexistent'; return 2; }
  canonical_work="$(canonical_candidate "$work")" || return 2
  canonical_root="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  canonical_project="$(canonical_candidate "$LANG_ID_PROJECT")" || return 2
  approval_real="$(canonical_candidate "$approval")" || return 2
  paths_overlap "$canonical_work" "$canonical_root" && { die '--work-dir overlaps checkout'; return 2; }
  paths_overlap "$canonical_work" "$canonical_project" && { die '--work-dir overlaps project'; return 2; }
  paths_overlap "$canonical_work" "$approval_real" && { die '--work-dir overlaps approval'; return 2; }
}

require_vast_host() {
  local mem_kib free_kib disk_root
  [[ "$(printenv VOKRA_PUBLISH_ON_VAST 2>/dev/null || true)" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] || die "this worker is VAST/Linux-only"
  [[ "$(uname -m)" == "x86_64" ]] || die "VAST host must be x86_64"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  (( mem_kib >= MIN_VAST_MEM_KIB )) || die "MemTotal=$mem_kib KiB is below the exact 64-GiB guard"
  disk_root="$VOKRA_SCRATCH"
  while [[ ! -e "$disk_root" && ! -L "$disk_root" ]]; do
    [[ "$disk_root" != / ]] || die "scratch path has no existing disk ancestor"
    disk_root="$(dirname "$disk_root")"
  done
  disk_root="$(canonical_candidate "$disk_root")" || return 2
  free_kib="$(df -Pk "$disk_root" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_kib >= MIN_FREE_DISK_KIB )) || die "free disk=$free_kib KiB is below the exact 150-GB guard"
}

require_tooling() {
  local tool path
  for tool in uv cargo rustc rustfmt git sha256sum awk grep find tee wc tr sort df; do
    command -v "$tool" >/dev/null 2>&1 || die "required VAST tool missing: $tool"
  done
  cargo clippy --version >/dev/null 2>&1 || die "clippy component is missing"
  cargo deny --version >/dev/null 2>&1 || die "cargo-deny is missing"
  cargo audit --version >/dev/null 2>&1 || die "cargo-audit is missing"
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die "not a Vokra checkout"
  [[ -f "$LANG_ID_PROJECT/pyproject.toml" && -f "$LANG_ID_PROJECT/uv.lock" ]] || die "dedicated Lang-ID uv project is missing"
  for path in "$VOKRA_ROOT/$PREPARER" "$VOKRA_ROOT/$REFERENCE_DUMPER" \
    "$VOKRA_ROOT/$PARITY_TEST_FILE" "$VOKRA_ROOT/$REFERENCE_INPUT"; do
    [[ -f "$path" ]] || die "required Lang-ID input is missing: $path"
  done
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "VAST checkout must be clean"
}

require_model_free_tooling() {
  local tool path
  for tool in uv git sha256sum awk grep find wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required model-free VAST tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$LANG_ID_PROJECT/pyproject.toml" && -f "$LANG_ID_PROJECT/uv.lock" ]] \
    || die "model-free VAST checkout or dedicated Lang-ID project is missing"
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] \
    || die "model-free Lang-ID preflight inputs are missing or symlinked"
  [[ -f "$MODEL_FREE_AUDIT" && ! -L "$MODEL_FREE_AUDIT" ]] \
    || die "model-free Lang-ID audit script is missing or symlinked"
  path="$VOKRA_ROOT/$REFERENCE_INPUT"
  [[ -f "$path" && ! -L "$path" ]] || die "fixed Lang-ID fixture is missing or symlinked"
}

run_model_free_audit() {
  local expected_head="$1" requested_work_dir="$2" run_stamp work_dir evidence_dir audit_log audit_report audit_rc summary_file
  require_clean_expected_head "$expected_head"
  require_vast_host
  require_model_free_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -n "$requested_work_dir" ]]; then work_dir="$requested_work_dir"
  else work_dir="$VOKRA_SCRATCH/speechbrain-lang-id-model-free/$run_stamp"; fi
  validate_work_dir "$work_dir" "$LICENSE_MANIFEST"
  evidence_dir="$work_dir/evidence"
  audit_log="$evidence_dir/model-free-audit.log"
  audit_report="$evidence_dir/model-free-audit.json"
  summary_file="$evidence_dir/summary.txt"
  mkdir -p "$evidence_dir"
  set +e
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$MODEL_FREE_AUDIT" \
    --lock "$LANG_ID_PROJECT/uv.lock" --project "$LANG_ID_PROJECT/pyproject.toml" \
    --manifest "$LICENSE_MANIFEST" --fixture "$VOKRA_ROOT/$REFERENCE_INPUT" \
    --output "$audit_report" >"$audit_log" 2>&1
  audit_rc=$?
  set -e
  cat "$audit_log" >&2
  [[ "$audit_rc" == 2 && -s "$audit_report" ]] || die "model-free audit did not produce its intentional blocked report"
  {
    echo "execution_status=BLOCKED_OWNER_REVIEW"
    echo "evidence_stage=MODEL_FREE_SOURCE_LICENSE_FIXTURE_AUDIT"
    echo "expected_head=$expected_head"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "audit_report=$audit_report"
    echo "publication_status=NO_UPLOAD"
    echo "model_status=NOT_ACQUIRED"
    echo "parity_status=NOT_RUN"
    echo "verdict=BLOCKED"
  } > "$summary_file"
  echo "run-speechbrain-lang-id-validation: BLOCKED_OWNER_REVIEW (model-free, no-upload)" >&2
  echo "Model-free audit report: $audit_report" >&2
  return 2
}

run_logged() {
  local label="$1" output="$2" status
  shift 2
  step "$label"
  set +e
  "$@" >"$output" 2>&1
  status=$?
  set -e
  cat "$output" | tee -a "$run_log"
  (( status == 0 )) || die "$label failed with exit code $status"
}

validate_prepared_manifest() {
  local manifest="$1" savedir="$2" count_file="$3" output_path="$4"
  uv run --frozen --project "$LANG_ID_PROJECT" --python 3.12 python - \
    "$manifest" "$savedir" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" \
    "$EXPECTED_N_MELS" "$EXPECTED_EMBEDDING_DIM" "$EXPECTED_CLASS_COUNT" \
    "$count_file" "$output_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
manifest, savedir, source, revision, n_mels, embedding_dim, class_count, count_path, output_path = sys.argv[1:]
def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
data = json.loads(Path(manifest).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
contract = data.get("contract")
if not isinstance(contract, dict):
    raise SystemExit("prepared manifest has no contract")
expected = {"source": source, "revision": revision, "n_mels": int(n_mels),
            "embedding_dim": int(embedding_dim), "class_count": int(class_count),
            "classifier_kind": "xvector-mlp-log-softmax-v1"}
for key, value in expected.items():
    if contract.get(key) != value:
        raise SystemExit(f"prepared contract {key}={contract.get(key)!r}, expected {value!r}")
labels = contract.get("labels")
if not isinstance(labels, list) or len(labels) != int(class_count) or any(not isinstance(x, str) or not x for x in labels):
    raise SystemExit("prepared contract does not contain 107 non-empty labels")
tensors = data.get("tensor_manifest")
if not isinstance(tensors, dict) or not tensors:
    raise SystemExit("prepared manifest has no tensor manifest")
digest = data.get("output_sha256")
if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    raise SystemExit("prepared manifest has no valid output SHA-256")
if hashlib.sha256(Path(output_path).read_bytes()).hexdigest() != digest:
    raise SystemExit("prepared output SHA-256 does not match its manifest")
Path(count_path).write_text(str(len(tensors)) + "\n", encoding="utf-8")
print(f"prepared_contract_ok tensors={len(tensors)}")
for filename in ("embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"):
    path = Path(savedir) / filename
    if not path.is_file():
        raise SystemExit(f"missing official checkpoint file: {path}")
    print(f"checkpoint_sha256 {filename}={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY
}

validate_reference_manifest() {
  local manifest="$1" savedir="$2" labels_path="$3" wav_path="$4" reference_dir="$5"
  [[ "$(wc -c < "$wav_path" | tr -d '[:space:]')" == "$FIXTURE_BYTES" ]] || die "fixed WAV fixture byte count differs"
  [[ "$(sha256_file "$wav_path")" == "$FIXTURE_SHA256" ]] || die "fixed WAV fixture SHA-256 differs"
  uv run --frozen --project "$LANG_ID_PROJECT" --python 3.12 python - \
    "$manifest" "$savedir" "$labels_path" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" \
    "$EXPECTED_N_MELS" "$EXPECTED_EMBEDDING_DIM" "$EXPECTED_CLASS_COUNT" \
    "$wav_path" "$reference_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
manifest, savedir, labels_path, source, revision, n_mels, embedding_dim, class_count, wav_path, reference_dir = sys.argv[1:]
def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
data = json.loads(Path(manifest).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
expected_manifest_keys = {
    "artifact_bytes", "artifact_sha256", "best_index", "best_label", "best_score",
    "checkpoint_sha256", "device", "embedding_shape", "feature_shape", "format",
    "numpy", "pcm_samples", "raw_feature_shape", "revision", "sample_rate",
    "score_shape", "source", "speechbrain", "torch", "torchaudio", "wav_bytes", "wav_sha256",
}

verify_reference_inventory() {
  local directory="$1" entry name
  local expected=(manifest.json pcm.f32.bin features.f32.bin embedding.f32.bin scores.f32.bin labels.json)
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference is not a regular directory"
  for entry in "$directory"/* "$directory"/.[!.]*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="${entry##*/}"
    case " ${expected[*]} " in *" $name "*) ;; *) die "reference contains unexpected entry: $name" ;; esac
    [[ -f "$entry" && ! -L "$entry" ]] || die "reference entry is not a regular file: $name"
  done
  for name in "${expected[@]}"; do
    [[ -f "$directory/$name" && ! -L "$directory/$name" && -s "$directory/$name" ]] || die "reference file is missing, symlinked, or empty: $name"
  done
}
if set(data) != expected_manifest_keys:
    raise SystemExit("reference manifest has missing or extra top-level keys")
if data.get("source") != source or data.get("revision") != revision:
    raise SystemExit("reference source/revision is not the pinned VoxLingua107 contract")
if data.get("sample_rate") != 16000 or data.get("device") != "cpu":
    raise SystemExit("reference sample rate is not 16000")
if not isinstance(data.get("feature_shape"), list) or len(data["feature_shape"]) != 3 or data["feature_shape"][0] != 1 or not isinstance(data["feature_shape"][1], int) or data["feature_shape"][1] <= 0 or data["feature_shape"][2] != int(n_mels):
    raise SystemExit("reference feature shape is not 60 mel bins")
if data.get("embedding_shape") != [1, 1, int(embedding_dim)] or data.get("score_shape") != [1, int(class_count)] or data.get("raw_feature_shape") != data.get("feature_shape"):
    raise SystemExit("reference shapes are not 256 embedding / 107 classes")
wav_hash = data.get("wav_sha256")
if data.get("wav_bytes") != 352078 or wav_hash != "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f":
    raise SystemExit("reference manifest has no WAV SHA-256")
if hashlib.sha256(Path(wav_path).read_bytes()).hexdigest() != wav_hash:
    raise SystemExit("reference WAV SHA-256 does not match the fixed fixture")
checkpoint_hashes = data.get("checkpoint_sha256")
if not isinstance(checkpoint_hashes, dict):
    raise SystemExit("reference manifest has no checkpoint SHA-256 map")
if set(checkpoint_hashes) != {"embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"}:
    raise SystemExit("reference checkpoint identity map is not exact")
if any(not isinstance(value, str) or not __import__("re").fullmatch(r"[0-9a-f]{64}", value) for value in checkpoint_hashes.values()):
    raise SystemExit("reference checkpoint identity map has malformed SHA-256")
for filename in ("embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"):
    expected = checkpoint_hashes.get(filename)
    path = Path(savedir) / filename
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise SystemExit(f"reference manifest has no authoritative digest for {filename}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"checkpoint digest mismatch for {filename}")
labels = json.loads(Path(labels_path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
if not isinstance(labels, list) or len(labels) != int(class_count) or any(not isinstance(x, str) or not x for x in labels):
    raise SystemExit("reference labels are not an ordered 107-label inventory")
artifact_hashes, artifact_bytes = data.get("artifact_sha256"), data.get("artifact_bytes")
artifact_names = ("pcm.f32.bin", "features.f32.bin", "embedding.f32.bin", "scores.f32.bin", "labels.json")
if set(artifact_hashes or {}) != set(artifact_names) or set(artifact_bytes or {}) != set(artifact_names):
    raise SystemExit("reference artifact identity maps are not exact")
for filename in artifact_names:
    path = Path(reference_dir) / filename
    if filename == "labels.json": path = Path(labels_path)
    if artifact_bytes[filename] != path.stat().st_size or artifact_hashes[filename] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise SystemExit(f"reference artifact identity mismatch: {filename}")
expected_sizes = {
    "pcm.f32.bin": int(data["pcm_samples"]) * 4,
    "features.f32.bin": int(data["feature_shape"][1]) * int(data["feature_shape"][2]) * 4,
    "embedding.f32.bin": int(embedding_dim) * 4,
    "scores.f32.bin": int(class_count) * 4,
}
if not isinstance(data["pcm_samples"], int) or data["pcm_samples"] <= 0 or any(artifact_bytes[name] != size for name, size in expected_sizes.items()):
    raise SystemExit("reference artifact byte counts do not follow the exact shapes")
best = data.get("best_index")
if not isinstance(best, int) or not 0 <= best < int(class_count) or data.get("best_label") != labels[best]:
    raise SystemExit("reference winner is inconsistent with ordered labels")
print("reference_contract_ok checkpoint_sha256=3 wav_sha256=1")
PY
}

run_self_test() {
  local script_path="$0" fail=0 cases=0 required fake_root fake_scratch rc probe approval
  probe="$(cd -P "$(mktemp -d)" && pwd -P)"
  trap 'rm -rf "$probe"' RETURN
  mkdir -p "$probe/real/existing"
  ln -s "$probe/real" "$probe/link"
  printf '{}' > "$probe/approval.json"
  if validate_work_dir "$probe/link/existing/nested/new" "$probe/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: existing descendant under symlink ancestor accepted'; fail=1
  fi
  if validate_work_dir "$probe/real/existing/nested/new" "$probe/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: work path overlapping approval parent accepted'; fail=1
  fi
  rm -rf "$probe"
  trap - RETURN
  cases=$((cases + 1))
  for required in "$MODEL_KIND" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$PREPARER" \
    "$REFERENCE_DUMPER" "$REFERENCE_INPUT" "$PARITY_TEST" "$PARITY_TEST_FILE" \
    "$GGUF_ENV" "$REFERENCE_DIR_ENV" "EXPECTED_N_MELS=60" \
    "EXPECTED_EMBEDDING_DIM=256" "EXPECTED_CLASS_COUNT=107" \
    "embedding_model.ckpt" "classifier.ckpt" "label_encoder.txt" "hyperparams.yaml" "config.json" \
    "snapshot_download" "code-bound upstream identity is unresolved" "model_free_audit.py" "MODEL_FREE_SOURCE_LICENSE_FIXTURE_AUDIT" \
    "--approval-evidence" "--approval-evidence-sha256" "write_transfer_manifest" "publication=NO_UPLOAD" "APPLE_LANG_ID_APPROVAL_EVIDENCE"; do
    if ! grep -Fq -- "$required" "$script_path"; then log "self-test FAIL: missing $required"; fail=1; fi
  done
  cases=$((cases + 1))
  # shellcheck disable=SC2016
  for required in 'uv run --frozen --project "$LANG_ID_PROJECT" --python 3.12 python' \
    'cargo build --offline --locked --release -p vokra-cli' \
    'cargo test --offline --locked --release -p vokra-models' \
    'cargo test --offline --locked --workspace -- --test-threads=1' \
    'cargo clippy --offline --locked --workspace --all-targets -- -D warnings' \
    'cargo deny --locked --offline check' 'cargo audit --no-fetch' 'cargo deny --version' 'cargo audit --version' \
    'test measure_cpu_against_independent_speechbrain' \
    'LANG_ID_MEASUREMENT_ONLY backend=cpu' 'test result: ok. 1 passed' \
    'lang-id[' 'lang-id: 107 scores in official label order' '--backend cpu' \
    'MIN_VAST_MEM_KIB=67108864' 'MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))' '--expected-head' 'require_clean_expected_head' '--test-threads=1' \
    '/proc/meminfo' 'df -Pk' 'VOKRA_PUBLISH_ON_VAST=1' \
    'git status --porcelain --untracked-files=all' 'mindepth 1 -maxdepth 1'; do
    if ! grep -Fq -- "$required" "$script_path"; then log "self-test FAIL: missing gate/sentinel $required"; fail=1; fi
  done
  cases=$((cases + 1))
  local unsafe_weights unsafe_pickle unsafe_torch
  unsafe_weights="weights_only=$(printf False)"
  unsafe_pickle="pickle.$(printf load)"
  unsafe_torch="torch.$(printf 'load(')"
  if grep -Fq "$unsafe_weights" "$VOKRA_ROOT/$PREPARER" "$VOKRA_ROOT/$REFERENCE_DUMPER" \
    || grep -Fq "$unsafe_pickle" "$VOKRA_ROOT/$PREPARER" "$VOKRA_ROOT/$REFERENCE_DUMPER" \
    || grep -Fq "$unsafe_torch" "$VOKRA_ROOT/$PREPARER" "$VOKRA_ROOT/$REFERENCE_DUMPER"; then
    log "self-test FAIL: unsafe pickle/checkpoint loading text found"; fail=1
  fi
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: publication command found"; fail=1
  fi
  if grep -En -- '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"; fail=1
  fi
  if grep -En '^[[:space:]]*(echo[[:space:]]+)?(PARITY|parity_status|verdict)=PASS' "$script_path" >/dev/null; then
    log "self-test FAIL: false parity/publication claim found"; fail=1
  fi
  cases=$((cases + 1))
  local evidence_dir
  evidence_dir="$(mktemp -d)"
  trap 'rm -rf "$evidence_dir"' EXIT
  printf 'test measure_cpu_against_independent_speechbrain ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nLANG_ID_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED\n' > "$evidence_dir/valid.log"
  require_cpu_test_evidence "$evidence_dir/valid.log" || { log 'self-test FAIL: valid evidence rejected'; fail=1; }
  cp "$evidence_dir/valid.log" "$evidence_dir/extra-test.log"
  printf 'test unrelated_smoke ... ok\n' >> "$evidence_dir/extra-test.log"
  if require_cpu_test_evidence "$evidence_dir/extra-test.log"; then log 'self-test FAIL: extra test accepted'; fail=1; fi
  cp "$evidence_dir/valid.log" "$evidence_dir/malformed-result.log"
  sed -i.bak 's/filtered out$/filtered out; unexpected/' "$evidence_dir/malformed-result.log"
  rm -f "$evidence_dir/malformed-result.log.bak"
  if require_cpu_test_evidence "$evidence_dir/malformed-result.log"; then log 'self-test FAIL: malformed result accepted'; fail=1; fi
  write_apple_args "$evidence_dir/apple-args.sh" "$(printf '%040d' 1)" "$(printf '%064d' 0)" "$(printf '%064d' 0)" "$(printf '%064d' 1)" "$(printf '%064d' 2)"
  bash -n "$evidence_dir/apple-args.sh"
  grep -Fq -- "--approval-evidence '<APPLE_LANG_ID_APPROVAL_EVIDENCE>'" "$evidence_dir/apple-args.sh" \
    || { log 'self-test FAIL: Apple approval placeholder missing'; fail=1; }
  grep -Fq -- '--expected-head 0000000000000000000000000000000000000001' "$evidence_dir/apple-args.sh" \
    || { log 'self-test FAIL: Apple exact-head binding missing'; fail=1; }
  grep -Fq -- '--approval-evidence-sha256 0000000000000000000000000000000000000000000000000000000000000001' "$evidence_dir/apple-args.sh" \
    || { log 'self-test FAIL: Apple approval SHA binding missing'; fail=1; }
  grep -Fq -- '--transfer-manifest-sha256 0000000000000000000000000000000000000000000000000000000000000002' "$evidence_dir/apple-args.sh" \
    || { log 'self-test FAIL: Apple transfer-manifest SHA binding missing'; fail=1; }
  write_transfer_manifest "$evidence_dir/transfer-manifest" "$(printf '%040d' 1)" "$(printf '%064d' 0)" "$(printf '%064d' 0)" "$(printf '%064d' 1)"
  grep -Fxq 'publication=NO_UPLOAD' "$evidence_dir/transfer-manifest" \
    || { log 'self-test FAIL: transfer publication policy missing'; fail=1; }
  [[ "$(cat "$evidence_dir/transfer-manifest.sha256")" == "$(sha256_file "$evidence_dir/transfer-manifest")" ]] \
    || { log 'self-test FAIL: transfer manifest SHA sidecar mismatch'; fail=1; }
  if grep -Eq '/(scratchpad|speechbrain-lang-id-validation)/|VOKRA_ROOT=' "$evidence_dir/apple-args.sh"; then
    log 'self-test FAIL: Apple args embed a VAST path'; fail=1
  fi
  rm -rf "$evidence_dir"
  trap - EXIT
  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$VOKRA_SCRATCH" >/dev/null 2>&1; then log "self-test FAIL: extra argument accepted"; fail=1; fi
  if "$script_path" --self-test --self-test >/dev/null 2>&1; then log "self-test FAIL: duplicate --self-test accepted"; fail=1; fi
  if "$script_path" --work-dir >/dev/null 2>&1; then log "self-test FAIL: missing work-dir value accepted"; fail=1; fi
  if "$script_path" --work-dir "" >/dev/null 2>&1; then log "self-test FAIL: empty work-dir value accepted"; fail=1; fi
  if "$script_path" --work-dir --not-a-directory >/dev/null 2>&1; then log "self-test FAIL: option used as work-dir value accepted"; fail=1; fi
  if "$script_path" --work-dir "$VOKRA_SCRATCH" --work-dir "$VOKRA_SCRATCH" >/dev/null 2>&1; then log "self-test FAIL: duplicate work-dir accepted"; fail=1; fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then log "self-test FAIL: unknown argument accepted"; fail=1; fi
  if "$script_path" --approval-evidence >/dev/null 2>&1; then log "self-test FAIL: missing approval value accepted"; fail=1; fi
  if "$script_path" --approval-evidence "" >/dev/null 2>&1; then log "self-test FAIL: empty approval value accepted"; fail=1; fi
  if "$script_path" --approval-evidence --work-dir x >/dev/null 2>&1; then log "self-test FAIL: option used as approval value accepted"; fail=1; fi
  if "$script_path" --approval-evidence one --approval-evidence two >/dev/null 2>&1; then log "self-test FAIL: duplicate approval accepted"; fail=1; fi
  if "$script_path" --approval-evidence one --approval-evidence-sha256 one >/dev/null 2>&1; then log "self-test FAIL: malformed approval SHA accepted"; fail=1; fi
  if "$script_path" --model-free >/dev/null 2>&1; then log "self-test FAIL: model-free route accepted without expected-head"; fail=1; fi
  if "$script_path" --model-free --model-free >/dev/null 2>&1; then log "self-test FAIL: duplicate model-free accepted"; fail=1; fi
  if "$script_path" --model-free --expected-head "$(printf '%040d' 1)" --expected-head "$(printf '%040d' 2)" >/dev/null 2>&1; then log "self-test FAIL: duplicate model-free expected-head accepted"; fail=1; fi
  if "$script_path" --self-test --model-free >/dev/null 2>&1; then log "self-test FAIL: mixed self-test/model-free accepted"; fail=1; fi
  if "$script_path" --model-free --expected-head "$(printf '%040d' 1)" --approval-evidence one --approval-evidence-sha256 "$(printf '%064d' 1)" >/dev/null 2>&1; then log "self-test FAIL: model-free route accepted approval arguments"; fail=1; fi
  if "$script_path" --approval-evidence one --approval-evidence-sha256 "$(printf '%064d' 1)" --approval-evidence-sha256 "$(printf '%064d' 2)" >/dev/null 2>&1; then log "self-test FAIL: duplicate approval SHA accepted"; fail=1; fi
  if "$script_path" --approval-evidence one >/dev/null 2>&1; then log "self-test FAIL: missing expected-head accepted"; fail=1; fi
  if "$script_path" --expected-head >/dev/null 2>&1; then log "self-test FAIL: missing expected-head value accepted"; fail=1; fi
  if "$script_path" --expected-head "$(printf '%040d' 1)" --expected-head "$(printf '%040d' 2)" >/dev/null 2>&1; then log "self-test FAIL: duplicate expected-head accepted"; fail=1; fi
  cases=$((cases + 1))
  fake_root="$(mktemp -d)"; fake_scratch="$fake_root/must-not-exist"
  mkdir -p "$fake_root/tools/parity"
  cp -R "$LANG_ID_PROJECT" "$fake_root/tools/parity/speechbrain_lang_id"
  approval="$fake_root/approval.json"
  printf '{}' > "$approval"
  set +e
  VOKRA_ROOT="$fake_root" VOKRA_SCRATCH="$fake_scratch" "$script_path" \
    --approval-evidence "$approval" --approval-evidence-sha256 "$(sha256_file "$approval")" --expected-head "$(printf '%040d' 1)" >"$fake_root/worker.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 2 || -e "$fake_scratch" ]] || ! grep -Eq 'SpeechBrain Lang-ID gate: BLOCKED:|checkout HEAD|checkout must be clean|missing \.git|not a git repository' "$fake_root/worker.log"; then
    log "self-test FAIL: license gate did not block before host/scratch"
    fail=1
  fi
  set +e
  VOKRA_ROOT="$fake_root" VOKRA_SCRATCH="$fake_scratch" "$script_path" \
    --model-free --expected-head "$(printf '%040d' 1)" >"$fake_root/model-free.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 2 || -e "$fake_scratch" ]] || ! grep -Eq 'missing \.git|not a git repository' "$fake_root/model-free.log"; then
    log "self-test FAIL: model-free audit did not stop before VAST scratch"
    fail=1
  fi
  rm -rf "$fake_root"
  if [[ $fail -eq 0 ]]; then
    echo "run-speechbrain-lang-id-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 model_free=0 requested_work_dir="" approval_evidence="" approval_evidence_sha="" expected_head="" run_stamp work_dir upstream_dir evidence_dir
  local seen_work_dir=0 seen_self_test=0 seen_model_free=0 seen_approval=0 seen_approval_sha=0 seen_expected_head=0
  local prepared_path prepared_manifest gguf_path reference_dir score_path
  local run_log env_log prep_log prep_contract_log reference_log reference_contract_log convert_log parity_log cli_log gate_log summary_file
  local tensor_count source_hashes tensor_count_file reference_manifest_sha approval_sha gguf_sha transfer_manifest transfer_manifest_sha
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        (( self_test == 0 )) || { die "--self-test must be exclusive"; return 2; }
        (( seen_work_dir == 0 )) || { die "duplicate --work-dir"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a non-option directory"; return 2; }
        requested_work_dir="$2"; seen_work_dir=1; shift 2 ;;
      --approval-evidence)
        (( self_test == 0 )) || { die "--self-test must be exclusive"; return 2; }
        (( seen_approval == 0 )) || { die "duplicate --approval-evidence"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence requires a non-option file"; return 2; }
        approval_evidence="$2"; seen_approval=1; shift 2 ;;
      --approval-evidence-sha256)
        (( self_test == 0 )) || { die "--self-test must be exclusive"; return 2; }
        (( seen_approval_sha == 0 )) || { die "duplicate --approval-evidence-sha256"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence-sha256 requires a non-option value"; return 2; }
        approval_evidence_sha="$2"; seen_approval_sha=1; shift 2 ;;
      --expected-head)
        (( self_test == 0 )) || { die "--self-test must be exclusive"; return 2; }
        (( seen_expected_head == 0 )) || { die "duplicate --expected-head"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--expected-head requires a lowercase 40-hex value"; return 2; }
        expected_head="$2"; seen_expected_head=1; shift 2 ;;
      --model-free)
        (( self_test == 0 )) || { die "--self-test must be exclusive"; return 2; }
        (( seen_model_free == 0 )) || { die "duplicate --model-free"; return 2; }
        seen_model_free=1; model_free=1; shift ;;
      --self-test)
        (( seen_self_test == 0 )) || { die "duplicate --self-test"; return 2; }
        seen_self_test=1; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ -z "$requested_work_dir$approval_evidence$approval_evidence_sha$expected_head" && "$model_free" == 0 ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test; return $?
  fi
  if [[ "$model_free" == 1 ]]; then
    [[ "$seen_expected_head" == 1 && "$seen_approval" == 0 && "$seen_approval_sha" == 0 ]] \
      || { usage; die "--model-free requires --expected-head and accepts no approval evidence"; return 2; }
    [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || { die '--expected-head must be exactly 40 lowercase hexadecimal characters'; return 2; }
    run_model_free_audit "$expected_head" "$requested_work_dir"
    return $?
  fi
  [[ -n "$approval_evidence" && "$seen_approval_sha" == 1 && "$seen_expected_head" == 1 ]] || { usage; die "--approval-evidence, --approval-evidence-sha256 and --expected-head are required"; return 2; }
  [[ "$approval_evidence_sha" =~ ^[0-9a-f]{64}$ ]] || { die '--approval-evidence-sha256 must be exactly 64 lowercase hexadecimal characters'; return 2; }
  require_clean_expected_head "$expected_head"
  [[ -f "$approval_evidence" && ! -L "$approval_evidence" ]] || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  [[ "$(sha256_file "$approval_evidence")" == "$approval_evidence_sha" ]] || { die '--approval-evidence SHA-256 differs from the external approval binding'; return 2; }
  license_preflight "$approval_evidence"
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -n "$requested_work_dir" ]]; then work_dir="$requested_work_dir"
  else work_dir="$VOKRA_SCRATCH/speechbrain-lang-id-validation/$run_stamp"; fi
  validate_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  require_tooling
  cd "$VOKRA_ROOT"
  upstream_dir="$work_dir/upstream"
  evidence_dir="$work_dir/evidence"
  prepared_path="$work_dir/lang-id-voxlingua107.prepared.safetensors"
  prepared_manifest="$prepared_path.manifest.json"
  gguf_path="$work_dir/lang-id-voxlingua107.gguf"
  reference_dir="$work_dir/reference"
  score_path="$work_dir/lang-id-voxlingua107.scores.f32"
  run_log="$evidence_dir/run.log"
  env_log="$evidence_dir/environment.txt"
  prep_log="$evidence_dir/prepare.log"
  prep_contract_log="$evidence_dir/prepare-contract.log"
  reference_log="$evidence_dir/reference.log"
  reference_contract_log="$evidence_dir/reference-contract.log"
  convert_log="$evidence_dir/convert.log"
  parity_log="$evidence_dir/parity.log"
  cli_log="$evidence_dir/cli.log"
  gate_log="$evidence_dir/gates.log"
  summary_file="$evidence_dir/summary.txt"
  transfer_manifest="$evidence_dir/apple-transfer-manifest.txt"
  mkdir -p "$evidence_dir"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Record environment and exact pinned source"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "model_kind=$MODEL_KIND"
    echo "uname=$(uname -a)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose; cargo --version; uv --version
  } | tee "$env_log"

  run_logged "Synchronize locked parity environment" "$evidence_dir/uv-sync.log" \
    uv sync --frozen --project "$LANG_ID_PROJECT" --python 3.12
  run_logged "Download exact pinned SpeechBrain snapshot" "$evidence_dir/upstream-download.log" \
    download_upstream_snapshot "$upstream_dir"
  run_logged "Verify exact pinned SpeechBrain snapshot" "$evidence_dir/upstream-contract.log" \
    verify_upstream_snapshot "$upstream_dir"
  run_logged "Prepare official pinned SpeechBrain checkpoint" "$prep_log" \
    uv run --frozen --project "$LANG_ID_PROJECT" --python 3.12 python \
    "$VOKRA_ROOT/$PREPARER" --source "$UPSTREAM_REPO" --revision "$UPSTREAM_REVISION" \
    --savedir "$upstream_dir" --output "$prepared_path"
  [[ -s "$prepared_path" && -s "$prepared_manifest" ]] || die "preparer emitted no checkpoint or manifest"
  tensor_count_file="$evidence_dir/prepared-tensor-count.txt"
  run_logged "Validate prepared checkpoint contract" "$prep_contract_log" \
    validate_prepared_manifest "$prepared_manifest" "$upstream_dir" "$tensor_count_file" "$prepared_path"
  [[ -s "$tensor_count_file" ]] || die "prepared tensor count was not recorded"
  tensor_count="$(<"$tensor_count_file")"
  [[ "$tensor_count" =~ ^[1-9][0-9]*$ ]] || die "invalid prepared tensor count: $tensor_count"

  step "Record repository-recorded SpeechBrain checkpoint digests"
  source_hashes="$evidence_dir/source-sha256.txt"; : > "$source_hashes"
  for filename in "${CHECKPOINT_FILES[@]}"; do
    [[ -f "$upstream_dir/$filename" ]] || die "required checkpoint absent: $filename"
    sha256sum "$upstream_dir/$filename" | tee -a "$source_hashes"
  done
  sha256sum "$prepared_path" | tee "$evidence_dir/prepared-sha256.txt"

  run_logged "Dump independent official SpeechBrain reference" "$reference_log" \
    uv run --frozen --project "$LANG_ID_PROJECT" --python 3.12 python \
    "$VOKRA_ROOT/$REFERENCE_DUMPER" --source "$UPSTREAM_REPO" --revision "$UPSTREAM_REVISION" \
    --savedir "$upstream_dir" --wav "$VOKRA_ROOT/$REFERENCE_INPUT" --output-dir "$reference_dir"
  verify_reference_inventory "$reference_dir"
  run_logged "Validate independent reference contract" "$reference_contract_log" \
    validate_reference_manifest "$reference_dir/manifest.json" "$upstream_dir" \
    "$reference_dir/labels.json" "$VOKRA_ROOT/$REFERENCE_INPUT" "$reference_dir"
  cp "$prepared_manifest" "$evidence_dir/prepared.manifest.json"
  cp "$reference_dir/manifest.json" "$evidence_dir/reference.manifest.json"
  cp "$reference_dir/labels.json" "$evidence_dir/reference.labels.json"
  sha256sum "$reference_dir"/* | tee "$evidence_dir/reference-sha256.txt"

  run_logged "Build strict Vokra CLI" "$evidence_dir/build.log" env CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo build --offline --locked --release -p vokra-cli
  run_logged "Convert strict Lang-ID GGUF" "$convert_log" target/release/vokra-cli convert \
    --model "$MODEL_KIND" --input "$prepared_path" --output "$gguf_path"
  grep -Eq "^converted $MODEL_KIND: $tensor_count tensors," "$convert_log" || die "converter count assertion failed"
  [[ -s "$gguf_path" ]] || die "converter emitted no GGUF"
  sha256sum "$gguf_path" | tee "$evidence_dir/gguf-sha256.txt"
  reference_manifest_sha="$(sha256_file "$reference_dir/manifest.json")"
  gguf_sha="$(sha256_file "$gguf_path")"
  approval_sha="$approval_evidence_sha"

  export "$GGUF_ENV=$gguf_path" "$REFERENCE_DIR_ENV=$reference_dir"
  run_logged "Run real-weight CPU parity measurement" "$parity_log" \
    CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --offline --locked --release -p vokra-models --test parity_speechbrain_lang_id_real \
    "$PARITY_TEST" -- --ignored --nocapture --test-threads=1
  require_cpu_test_evidence "$parity_log"

  run_logged "Run CLI classification smoke" "$cli_log" target/release/vokra-cli run \
    --model "$gguf_path" --input "$VOKRA_ROOT/$REFERENCE_INPUT" --backend cpu --output "$score_path"
  grep -Fq "lang-id[" "$cli_log" || die "CLI emitted no ranked classification"
  grep -Fq "lang-id: 107 scores in official label order" "$cli_log" || die "CLI did not emit 107 scores"
  [[ -s "$score_path" ]] || die "CLI emitted no score vector"

  run_logged "Run exact locked offline workspace tests" "$evidence_dir/workspace-test.log" \
    env CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --offline --locked --workspace -- --test-threads=1
  run_logged "Run exact locked offline workspace Clippy" "$evidence_dir/clippy.log" \
    env CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo clippy --offline --locked --workspace --all-targets -- -D warnings
  run_logged "Run locked offline cargo-deny" "$evidence_dir/cargo-deny.log" \
    env CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo deny --locked --offline check
  run_logged "Run no-fetch cargo-audit" "$evidence_dir/cargo-audit.log" \
    env CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo audit --no-fetch

  step "Run focused repository gates"
  set +e
  {
    bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
    bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
    bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
    bash "$VOKRA_ROOT/scripts/check-parity-sidecar-citations.sh"
    cargo fmt --all -- --check
  } >"$gate_log" 2>&1
  local gate_status=$?
  set -e
  cat "$gate_log" | tee -a "$run_log"
  (( gate_status == 0 )) || die "focused repository gate failed"

  require_clean_expected_head "$expected_head"
  write_transfer_manifest "$transfer_manifest" "$expected_head" "$gguf_sha" "$reference_manifest_sha" "$approval_sha"
  transfer_manifest_sha="$(sha256_file "$transfer_manifest")"
  require_clean_expected_head "$expected_head"
  write_apple_args "$evidence_dir/apple-silicon-speechbrain-lang-id-args.sh" \
    "$expected_head" "$gguf_sha" "$reference_manifest_sha" "$approval_sha" "$transfer_manifest_sha"
  require_clean_expected_head "$expected_head"

  {
    echo "execution_status=MEASURED_NOT_GATED"
    echo "expected_head=$expected_head"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "prepared_tensor_count=$tensor_count"
    echo "source_checkpoint_sha256_evidence=$source_hashes"
    echo "prepared_sha256=$(sha256_file "$prepared_path")"
    echo "gguf_sha256=$gguf_sha"
    echo "reference_manifest_sha256=$reference_manifest_sha"
    echo "approval_evidence_sha256=$approval_sha"
    echo "transfer_manifest=$transfer_manifest"
    echo "transfer_manifest_sha256=$transfer_manifest_sha"
    echo "apple_args=$evidence_dir/apple-silicon-speechbrain-lang-id-args.sh"
    echo "reference_dir=$reference_dir"
    echo "gguf_path=$gguf_path"
    echo "cpu_test=$PARITY_TEST"
    echo "cpu_test_sentinel=LANG_ID_MEASUREMENT_ONLY backend=cpu"
    echo "parity_status=MEASURED_NOT_GATED"
    echo "metal_status=NOT_RUN"
    echo "runtime_status=CPU_SMOKE_ONLY"
    echo "publication_status=NOT_REQUESTED"
    echo "verdict=MEASUREMENT_ONLY"
  } > "$summary_file"
  echo "run-speechbrain-lang-id-validation: MEASUREMENT_ONLY"
  echo "Direct VAST-to-Apple transfer packet: GGUF=$gguf_path reference=$reference_dir transfer_manifest=$transfer_manifest transfer_manifest_sha256=${transfer_manifest}.sha256 args=$evidence_dir/apple-silicon-speechbrain-lang-id-args.sh"
  echo "Pull only this packet and small evidence before destroy; do not pull the prepared checkpoint or upstream snapshot."
}
main "$@"
