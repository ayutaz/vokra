#!/usr/bin/env bash
# VAST-only validation worker for the dedicated voice-gender classifier.
# No upload, publication, or Git push is performed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/voice_gender_classifier_dump_reference.py"
PREPARE_CHECKPOINT="$VOKRA_ROOT/tools/parity/voice_gender_classifier_prepare_checkpoint.py"
MODEL_KIND="voice-gender-classifier"
LICENSE_SPDX="mit"
UPSTREAM_REPO="JaesungHuh/voice-gender-classifier"
UPSTREAM_REVISION="49bcbecfd929ba5a043bde645fdff1a375eb79c7"
UPSTREAM_GITHUB_URL="https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_HF_REVISION="db1222153bd60337e900be22add7af180452adc0"
UPSTREAM_FILE="model.safetensors"
CHECKPOINT_BYTES=61907512
UPSTREAM_LICENSE_FILE="LICENSE"
UPSTREAM_LICENSE_SPDX="MIT"
UPSTREAM_LICENSE_COPYRIGHT="Copyright (c) 2024 jaesunghuh"
UPSTREAM_HF_LICENSE="mit"
PUBLIC_REPO="vokra/voice-gender-classifier"
PUBLIC_REVISION="94c8d0ba41cfe2f7b8a773eb4a7982cf4facbc84"
PUBLIC_FILE="voice-gender-classifier.restamped.gguf"
PUBLIC_SHA256="e1e61f1493601087f5db5867c4f750ec99d6b11223b5323bd120e3c21e8f957f"
CHECKPOINT_SHA256="2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5"
FP32_PARITY_BOUND="0.010000000"
FIXTURE_KIND="official_canned_synthetic_tone"
MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

log() { printf '[voice-gender-vast] %s\n' "$*" >&2; }
step() { printf '\n[voice-gender-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

preflight_gate() {
  local requested_sha256="${1:-}"
  [[ "$UPSTREAM_REPO" == "JaesungHuh/voice-gender-classifier" ]] || die "upstream repository contract drifted"
  [[ "$UPSTREAM_GITHUB_URL" == "https://github.com/JaesungHuh/voice-gender-classifier.git" ]] || die "upstream source URL contract drifted"
  [[ "$UPSTREAM_REVISION" == "49bcbecfd929ba5a043bde645fdff1a375eb79c7" ]] || die "upstream source revision contract drifted"
  [[ "$UPSTREAM_HF_REVISION" == "db1222153bd60337e900be22add7af180452adc0" ]] || die "upstream Hub revision contract drifted"
  [[ "$UPSTREAM_FILE" == "model.safetensors" && "$CHECKPOINT_BYTES" == 61907512 ]] || die "upstream checkpoint file contract drifted"
  [[ "$CHECKPOINT_SHA256" == "2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5" ]] || die "fixed checkpoint digest contract drifted"
  [[ "$UPSTREAM_LICENSE_FILE" == "LICENSE" && "$UPSTREAM_LICENSE_SPDX" == "MIT" ]] || die "upstream license contract drifted"
  [[ "$UPSTREAM_LICENSE_COPYRIGHT" == "Copyright (c) 2024 jaesunghuh" ]] || die "upstream license copyright contract drifted"
  [[ "$UPSTREAM_HF_LICENSE" == "mit" ]] || die "HF cardData license contract drifted"
  [[ "$requested_sha256" == "$CHECKPOINT_SHA256" ]] || die "checkpoint digest is not the fixed authenticated identity"
}

usage() {
  cat <<'EOF' >&2
usage: run-voice-gender-classifier-validation.sh --checkpoint-sha256 <64-hex> \
         [--work-dir <empty-dir>]
       run-voice-gender-classifier-validation.sh --self-test

VAST-only staging validation. The checkpoint digest is required because
the upstream Hub file can change independently of the fixed Git source.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is available"
  fi
}

verify_hf_identity() {
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - \
    "$PARITY_PROJECT" "$UPSTREAM_REPO" "$UPSTREAM_HF_REVISION" "$UPSTREAM_FILE" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256" <<'PY'
import sys
from huggingface_hub import HfApi, RepoFile, RepoFolder
sys.path.insert(0, sys.argv[1])
from voice_gender_classifier_hf_identity import verify_info

_, repository, revision, filename, expected_bytes, expected_sha256 = sys.argv[1:]
expected_bytes = int(expected_bytes)
api = HfApi()
info = api.model_info(repo_id=repository, revision=revision)
tree = []
for item in api.list_repo_tree(repo_id=repository, revision=revision, recursive=True, expand=True):
    if isinstance(item, RepoFolder):
        continue
    if not isinstance(item, RepoFile):
        raise SystemExit(f"unsupported HF tree entry: {item!r}")
    tree.append(item)
verify_info(
    info,
    tree,
    repository=repository,
    revision=revision,
    filename=filename,
    expected_bytes=expected_bytes,
    expected_sha256=expected_sha256,
    expected_license="mit",
)
print(f"HF identity authenticated: repository={repository} revision={revision} file={filename} bytes={expected_bytes} sha256={expected_sha256} license=MIT")
PY
}

verify_source_identity() {
  local source_dir="$1"
  [[ "$(git -C "$source_dir" rev-parse HEAD)" == "$UPSTREAM_REVISION" ]] || die "upstream source checkout is not the fixed revision"
  [[ -z "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]] || die "upstream source checkout is dirty"
  local license_path="$source_dir/$UPSTREAM_LICENSE_FILE"
  [[ -f "$license_path" && ! -L "$license_path" ]] || die "upstream source primary license file is missing or symlinked"
  grep -Fqi -- 'MIT License' "$license_path" || die "upstream source does not contain MIT license evidence"
  grep -Fqi -- "$UPSTREAM_LICENSE_COPYRIGHT" "$license_path" || die "upstream source copyright evidence is missing"
  grep -Fqi -- 'Permission is hereby granted, free of charge' "$license_path" || die "upstream source MIT grant evidence is missing"
  echo "source identity authenticated: revision=$UPSTREAM_REVISION license=$UPSTREAM_LICENSE_SPDX license_file=$UPSTREAM_LICENSE_FILE"
}

verify_file() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || { die "missing or symlinked file: $path"; return 2; }
  [[ "$(sha256_file "$path")" == "$expected" ]] || {
    die "SHA-256 mismatch for $path"
    return 2
  }
}

verify_checkpoint() {
  local path="$1"
  verify_file "$path" "$CHECKPOINT_SHA256"
  [[ "$(wc -c < "$path" | tr -d '[:space:]')" == "$CHECKPOINT_BYTES" ]] \
    || die "checkpoint byte size mismatch for $path"
}

verify_prepared_audit() {
  local audit_path="$1" prepared_path="$2"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - \
    "$audit_path" "$prepared_path" "$UPSTREAM_REPO" "$UPSTREAM_HF_REVISION" \
    "$UPSTREAM_GITHUB_URL" "$UPSTREAM_REVISION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit_path, prepared_path, repository, revision, source_repository, source_revision = sys.argv[1:]
audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
expected = {
    "schema": "vokra-voice-gender-checkpoint-normalization-v1",
    "status": "AUTHENTICATED_NORMALIZED",
    "source_repository": repository,
    "source_revision": revision,
    "upstream_source_repository": source_repository,
    "upstream_source_revision": source_revision,
    "transform": "remove_authenticated_batchnorm_num_batches_tracked_v1",
    "input_tensor_count": 233,
    "input_floating_tensor_count": 202,
    "input_counter_count": 31,
    "output_tensor_count": 202,
}
for key, value in expected.items():
    if audit.get(key) != value:
        raise SystemExit(f"prepared checkpoint audit mismatch: {key}={audit.get(key)!r}, expected {value!r}")

counter_names = {"bn1.num_batches_tracked", "bn5.num_batches_tracked", "bn6.num_batches_tracked", "attention.2.num_batches_tracked"}
for layer in range(1, 4):
    counter_names.add(f"layer{layer}.bn1.num_batches_tracked")
    counter_names.update(f"layer{layer}.bns.{inner}.num_batches_tracked" for inner in range(7))
    counter_names.add(f"layer{layer}.bn3.num_batches_tracked")
removed = audit.get("removed_counter_names")
if not isinstance(removed, list) or len(removed) != 31 or len(set(removed)) != 31 or set(removed) != counter_names:
    raise SystemExit("prepared checkpoint audit counter manifest is not the exact 31-counter set")

manifest = audit.get("floating_tensor_manifest")
if not isinstance(manifest, list) or len(manifest) != 202:
    raise SystemExit("prepared checkpoint audit floating manifest count is not 202")
names = [row.get("name") if isinstance(row, dict) else None for row in manifest]
if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 202:
    raise SystemExit("prepared checkpoint audit floating manifest names are not unique")
for row in manifest:
    if not isinstance(row, dict) or set(row) != {"name", "shape", "dtype", "sha256"}:
        raise SystemExit("prepared checkpoint audit floating manifest row is malformed")
manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
if audit.get("floating_tensor_manifest_sha256") != manifest_sha256:
    raise SystemExit("prepared checkpoint audit floating manifest digest mismatch")

prepared = Path(prepared_path).read_bytes()
if audit.get("output_bytes") != len(prepared):
    raise SystemExit("prepared checkpoint audit output byte count mismatch")
actual_sha256 = hashlib.sha256(prepared).hexdigest()
if audit.get("output_sha256") != actual_sha256:
    raise SystemExit(f"prepared checkpoint audit output SHA-256 mismatch: {audit.get('output_sha256')!r} != {actual_sha256}")
print("prepared checkpoint audit authenticated: status=AUTHENTICATED_NORMALIZED input=233 floating=202 counters=31 output=202")
PY
}

require_vast_host() {
  local memory free_disk
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is required"
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "VAST worker requires Linux x86_64"
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ ]] || die "could not read physical memory"
  (( memory >= MIN_VAST_MEM_KIB )) || die "at least 64 GiB RAM is required"
  mkdir -p "$VOKRA_SCRATCH"
  free_disk="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk >= MIN_FREE_DISK_KIB )) || die "at least 150 GB free disk is required"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr df nproc rustfmt cargo-deny cargo-audit; do
    command -v "$tool" >/dev/null 2>&1 || die "required VAST tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die "repository checkout is missing"
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die "locked parity project is missing"
  [[ -f "$PARITY_DUMPER" ]] || die "dedicated reference dumper is missing"
  [[ -f "$PREPARE_CHECKPOINT" ]] || die "dedicated checkpoint preparation sidecar is missing"
  grep -Fq 'EXPECTED_INPUT_TENSOR_COUNT = 233' "$PREPARE_CHECKPOINT" || die "checkpoint preparation tensor contract is missing"
  grep -Fq 'EXPECTED_COUNTER_COUNT = 31' "$PREPARE_CHECKPOINT" || die "checkpoint preparation counter contract is missing"
  grep -Fq 'model.ECAPA_gender' "$PARITY_DUMPER" || die "dumper is not importing official model"
  grep -Fq "$UPSTREAM_REVISION" "$PARITY_DUMPER" || die "dumper revision is not pinned"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "VAST checkout must be clean"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$repository" "$revision" "$filename" "$output_dir"
}

verify_artifact_contract() {
  local artifact="$1"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - "$VOKRA_ROOT" "$artifact" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools" / "audit"))
from gguf_manifest import read_manifest
metadata, tensors = read_manifest(Path(sys.argv[2]))
if metadata.get("vokra.model.arch") != "ecapa_tdnn" or len(tensors) != 202:
    raise SystemExit("historical artifact identity is not the known 202-tensor ecapa_tdnn stamp")
print("historical artifact contract confirmed: arch=ecapa_tdnn tensors=202")
PY
}

verify_corrected_provenance() {
  local artifact="$1"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - "$VOKRA_ROOT" "$artifact" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools" / "audit"))
from gguf_manifest import read_manifest
metadata, tensors = read_manifest(Path(sys.argv[2]))
if metadata.get("vokra.model.arch") != "voice_gender_classifier":
    raise SystemExit("corrected artifact did not receive the dedicated architecture")
if len(tensors) != 202:
    raise SystemExit(f"corrected artifact tensor count is {len(tensors)}, expected 202")
if metadata.get("vokra.voice_gender.upstream_revision") != "49bcbecfd929ba5a043bde645fdff1a375eb79c7":
    raise SystemExit("corrected artifact source revision is not the pinned official Git revision")
if metadata.get("vokra.voice_gender.upstream_hf_revision") != "db1222153bd60337e900be22add7af180452adc0":
    raise SystemExit("corrected artifact checkpoint revision is not the pinned official Hub revision")
if metadata.get("vokra.provenance.license") != "mit" or metadata.get("vokra.provenance.weight_license") != "permissive":
    raise SystemExit("corrected artifact provenance is not MIT/permissive")
print("corrected provenance confirmed: arch=voice_gender_classifier license=mit weight_license=permissive")
PY
}

verify_cpu_parity_log() {
  local log_path="$1" metrics marker_count pass_count
  marker_count="$(grep -Ec '^VOICE_GENDER_OFFICIAL_PARITY(_METRICS| ).*$' "$log_path" || true)"
  [[ "$marker_count" == 2 ]] || { die "CPU parity marker count is not exactly 2: $marker_count"; return 2; }
  metrics="$(grep -E '^VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=[0-9]+\.[0-9]{9} embedding_max_abs=[0-9]+\.[0-9]{9} logits_max_abs=[0-9]+\.[0-9]{9} probability_max_abs=[0-9]+\.[0-9]{9} bound=0\.010000000 fixture=official_canned_synthetic_tone$' "$log_path" || true)"
  [[ -n "$metrics" ]] || { die "CPU parity metrics marker is missing or malformed"; return 2; }
  [[ "$(printf '%s\n' "$metrics" | wc -l | tr -d '[:space:]')" == 1 ]] || { die "CPU parity metrics marker is duplicated"; return 2; }
  pass_count="$(grep -Ec '^VOICE_GENDER_OFFICIAL_PARITY PASS bound=0\.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream$' "$log_path" || true)"
  [[ "$pass_count" == 1 ]] || { die "CPU parity PASS marker is missing, duplicated, or malformed"; return 2; }
  printf '%s\n' "$metrics" | awk '{ for (i = 2; i <= NF; i++) { split($i, pair, "="); if (pair[1] ~ /_max_abs$/ && (pair[2] + 0) > 0.01) exit 1 } }' \
    || { die "CPU parity metric exceeds the fixed FP32 bound"; return 2; }
  printf 'CPU parity gate authenticated: bound=%s fixture=%s\n%s\n' "$FP32_PARITY_BOUND" "$FIXTURE_KIND" "$metrics"
}

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" fail=0 required marker_dir
  # shellcheck disable=SC2016 # literal contract tokens intentionally keep quoting
  for required in "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_GITHUB_URL" \
    "$UPSTREAM_HF_REVISION" "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" \
    "$PUBLIC_SHA256" "$MODEL_KIND" "$LICENSE_SPDX" "voice_gender_classifier_dump_reference.py" \
    "voice_gender_classifier_prepare_checkpoint.py" 'checkpoint-prepare.log' \
    "$CHECKPOINT_SHA256" "$CHECKPOINT_BYTES" "$UPSTREAM_LICENSE_FILE" "$UPSTREAM_LICENSE_SPDX" \
    "$UPSTREAM_LICENSE_COPYRIGHT" "$UPSTREAM_HF_LICENSE" 'verify_hf_identity' 'verify_source_identity' \
    "CARGO_BUILD_JOBS=\"\${CARGO_BUILD_JOBS:-1}\"" 'VOKRA_PUBLISH_ON_VAST' \
    'MIN_VAST_MEM_KIB=67108864' 'MIN_FREE_DISK_KIB=150000000' \
    '"$VOKRA_ROOT/target/release/vokra-cli" convert' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked -p vokra-models' \
    'cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace --all-targets -- -D warnings | tee -a "$evidence_dir/gates.log"' \
    '(cd "$VOKRA_ROOT" && cargo deny check licenses advisories bans) | tee -a "$evidence_dir/gates.log"' \
    '(cd "$VOKRA_ROOT" && cargo audit) | tee -a "$evidence_dir/gates.log"' \
    '--checkpoint "$checkpoint" --upstream-src "$source_dir" --canned --out-dir "$fixture_dir"' \
    '--input "$checkpoint" --output "$prepared_checkpoint" --audit-json "$prepare_audit"' \
    'verify_prepared_audit "$prepare_audit" "$prepared_checkpoint" | tee "$evidence_dir/checkpoint-prepare-verified.log"' \
    'prepared checkpoint audit authenticated: status=AUTHENTICATED_NORMALIZED input=233 floating=202 counters=31 output=202' \
    '--input "$prepared_checkpoint" --output "$corrected"' \
    'parity_voice_gender_classifier' 'VOKRA_VOICE_GENDER_FIXTURE_KIND' \
    'VOICE_GENDER_OFFICIAL_PARITY_METRICS' 'VOICE_GENDER_OFFICIAL_PARITY PASS' \
    'FP32_PARITY_BOUND' 'verify_cpu_parity_log' 'verify_corrected_provenance'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing: $required"; fail=1; }
  done
  # shellcheck disable=SC2016 # literal summary fields intentionally keep quoting
  for required in 'prepared_checkpoint_sha256=$(sha256_file "$prepared_checkpoint")' 'prepare_audit_sha256=$(sha256_file "$prepare_audit")' 'publication=NOT_PERFORMED'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing summary evidence: $required"; fail=1; }
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test found direct Python invocation"; fail=1
  fi
  if grep -En -- '(^|[[:space:]])(git[[:space:]]+push|.*upload|.*publish|--push|--upload|--publish)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test found publication operation"; fail=1
  fi
  # shellcheck disable=SC2016 # literal log-write contracts intentionally keep quoting
  if grep -En 'cargo (clippy|deny|audit).*gates\.log' "$script_path" | grep -vFq 'tee -a "$evidence_dir/gates.log"'; then
    log "self-test found a non-append Cargo repository gate log write"; fail=1
  fi
  # shellcheck disable=SC2016 # literal log-write contracts intentionally keep quoting
  if ! grep -Fq 'bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh" | tee "$evidence_dir/gates.log"' "$script_path" \
    || ! grep -Fq 'bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" | tee -a "$evidence_dir/gates.log"' "$script_path" \
    || ! grep -Fq 'cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check | tee -a "$evidence_dir/gates.log"' "$script_path" \
    || ! grep -Fq 'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace | tee -a "$evidence_dir/gates.log"' "$script_path"; then
    log "self-test found a repository gate log write contract gap"; fail=1
  fi
  marker_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-voice-gender-marker.XXXXXX")"
  trap 'rm -rf "$marker_dir"' EXIT
  printf '%s\n' \
    'VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.001000000 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.010000000 fixture=official_canned_synthetic_tone' \
    'VOICE_GENDER_OFFICIAL_PARITY PASS bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream' \
    > "$marker_dir/valid.log"
  verify_cpu_parity_log "$marker_dir/valid.log" >/dev/null || { log "self-test rejected a valid CPU parity log"; fail=1; }
  cp "$marker_dir/valid.log" "$marker_dir/duplicate.log"
  printf '%s\n' 'VOICE_GENDER_OFFICIAL_PARITY PASS bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream' >> "$marker_dir/duplicate.log"
  if verify_cpu_parity_log "$marker_dir/duplicate.log" >/dev/null 2>&1; then
    log "self-test accepted a duplicate CPU parity marker"; fail=1
  fi
  printf '%s\n' \
    'VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.001000000 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.020000000 fixture=official_canned_synthetic_tone' \
    'VOICE_GENDER_OFFICIAL_PARITY PASS bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream' \
    > "$marker_dir/malformed.log"
  if verify_cpu_parity_log "$marker_dir/malformed.log" >/dev/null 2>&1; then
    log "self-test accepted a malformed CPU parity metrics marker"; fail=1
  fi
  printf '%s\n' \
    'VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.010000001 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.010000000 fixture=official_canned_synthetic_tone' \
    'VOICE_GENDER_OFFICIAL_PARITY PASS bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream' \
    > "$marker_dir/over-bound.log"
  if verify_cpu_parity_log "$marker_dir/over-bound.log" >/dev/null 2>&1; then
    log "self-test accepted a CPU parity metric above the fixed bound"; fail=1
  fi
  printf '%s\n' \
    'VOICE_GENDER_OFFICIAL_PARITY_METRICS feature_max_abs=0.001000000 embedding_max_abs=0.002000000 logits_max_abs=0.003000000 probability_max_abs=0.004000000 bound=0.010000000 fixture=official_canned_synthetic_tone' \
    'VOICE_GENDER_OFFICIAL_PARITY FAIL bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream' \
    > "$marker_dir/nonpass.log"
  if verify_cpu_parity_log "$marker_dir/nonpass.log" >/dev/null 2>&1; then
    log "self-test accepted a non-PASS CPU parity marker"; fail=1
  fi
  printf '%s\n' 'VOICE_GENDER_OFFICIAL_PARITY PASS bound=0.010000000 fixture=official_canned_synthetic_tone oracle=official_upstream' > "$marker_dir/missing.log"
  if verify_cpu_parity_log "$marker_dir/missing.log" >/dev/null 2>&1; then
    log "self-test accepted a missing CPU parity metrics marker"; fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PARITY_DUMPER" --self-test >/dev/null || fail=1
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --offline --python 3.12 python "$PREPARE_CHECKPOINT" --self-test >/dev/null || fail=1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$PARITY_PROJECT" "$UPSTREAM_REPO" "$UPSTREAM_HF_REVISION" "$UPSTREAM_FILE" "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256" <<'PY' || fail=1
import sys
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
from voice_gender_classifier_hf_identity import IdentityError, verify_info

repository, revision, filename, expected_bytes, expected_sha256 = sys.argv[2:]
expected_bytes = int(expected_bytes)
good_lfs = SimpleNamespace(size=expected_bytes, sha256=expected_sha256, pointer_size=128)
good_item = SimpleNamespace(path=filename, size=expected_bytes, lfs=good_lfs)
good_info = SimpleNamespace(
    id=repository,
    sha=revision,
    card_data=SimpleNamespace(license="mit"),
)

def clone(value, **changes):
    fields = vars(value).copy()
    fields.update(changes)
    return SimpleNamespace(**fields)

def check(info=good_info, tree=(good_item,)):
    verify_info(
        info,
        tree,
        repository=repository,
        revision=revision,
        filename=filename,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )

check()
failures = {
    "repository": clone(good_info, id="other/repository"),
    "revision": clone(good_info, sha="0" * 40),
    "license": clone(good_info, card_data=SimpleNamespace(license="apache-2.0")),
    "legacy-cardData": SimpleNamespace(id=repository, sha=revision, cardData={"license": "mit"}),
}
for name, info in failures.items():
    try:
        check(info=info)
    except IdentityError:
        pass
    else:
        raise AssertionError(f"accepted {name} drift")

bad_items = {
    "missing-file": (),
    "duplicate-file": (good_item, good_item),
    "size": (clone(good_item, size=expected_bytes + 1),),
    "missing-lfs": (clone(good_item, lfs=None),),
    "lfs-size": (clone(good_item, lfs=clone(good_lfs, size=expected_bytes + 1)),),
    "lfs-sha256": (clone(good_item, lfs=clone(good_lfs, sha256="1" * 64)),),
    "malformed-lfs-sha256": (clone(good_item, lfs=clone(good_lfs, sha256="not-a-sha")),),
}
for name, tree in bad_items.items():
    try:
        check(tree=tree)
    except IdentityError:
        pass
    else:
        raise AssertionError(f"accepted {name} drift")
print("voice_gender_classifier_hf_identity self-test: PASS")
PY
  "$script_path" --self-test --work-dir /tmp/invalid >/dev/null 2>&1 && { log "self-test accepted extra argument"; fail=1; } || true
  (( fail == 0 )) || return 1
  echo "run-voice-gender-classifier-validation.sh self-test: PASS"
)

main() {
  local self_test=0 work_arg='' checkpoint_sha256='' checkpoint_seen=0 work_seen=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint-sha256) (( checkpoint_seen == 0 )) || die 'duplicate --checkpoint-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || return 2; checkpoint_sha256="$2"; checkpoint_seen=1; shift 2 ;;
      --work-dir) (( work_seen == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || return 2; work_arg="$2"; work_seen=1; shift 2 ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self_test )); then
    [[ -z "$work_arg$checkpoint_sha256" ]] || die "--self-test accepts no other arguments"
    run_self_test; return $?
  fi
  (( checkpoint_seen )) || { usage; die "--checkpoint-sha256 is required"; return 2; }
  [[ "$checkpoint_sha256" =~ ^[0-9A-Fa-f]{64}$ ]] || die "checkpoint digest must be 64 hexadecimal characters"
  checkpoint_sha256="$(printf '%s' "$checkpoint_sha256" | tr '[:upper:]' '[:lower:]')"
  preflight_gate "$checkpoint_sha256"
  require_vast_host
  require_tooling
  local run_stamp work_dir input_dir source_dir fixture_dir evidence_dir checkpoint prepared_checkpoint prepare_audit public_artifact corrected
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${work_arg:-$VOKRA_SCRATCH/voice-gender-validation/$run_stamp}"
  [[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work directory must be absent or empty"
  input_dir="$work_dir/input"; source_dir="$input_dir/source"; fixture_dir="$work_dir/fixtures"; evidence_dir="$work_dir/evidence"
  checkpoint="$input_dir/checkpoint/$UPSTREAM_FILE"; prepared_checkpoint="$input_dir/checkpoint/voice-gender-classifier.prepared.safetensors"; prepare_audit="$evidence_dir/checkpoint-prepare.json"; public_artifact="$input_dir/public/$PUBLIC_FILE"; corrected="$work_dir/voice-gender-classifier-corrected.gguf"
  mkdir -p "$fixture_dir" "$evidence_dir" "$(dirname "$checkpoint")" "$(dirname "$public_artifact")"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-voice-gender"
  step "Record environment"
  { date -u; uname -a; git -C "$VOKRA_ROOT" rev-parse HEAD; echo "upstream_revision=$UPSTREAM_REVISION"; echo "upstream_hf_revision=$UPSTREAM_HF_REVISION"; echo "checkpoint_sha256=$CHECKPOINT_SHA256"; } > "$evidence_dir/environment.txt"
  step "Authenticate historical public artifact"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$(dirname "$public_artifact")"
  verify_file "$public_artifact" "$PUBLIC_SHA256"
  verify_artifact_contract "$public_artifact" | tee "$evidence_dir/public-contract.log"
  step "Fetch checkpoint and exact upstream source"
  verify_hf_identity | tee "$evidence_dir/hf-identity.log"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_HF_REVISION" "$UPSTREAM_FILE" "$(dirname "$checkpoint")"
  verify_checkpoint "$checkpoint"
  git clone --no-checkout "$UPSTREAM_GITHUB_URL" "$source_dir"
  git -C "$source_dir" checkout --detach "$UPSTREAM_REVISION"
  verify_source_identity "$source_dir" | tee "$evidence_dir/source-identity.log"
  step "Generate independent official fixtures"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PARITY_DUMPER" --checkpoint "$checkpoint" --upstream-src "$source_dir" --canned --out-dir "$fixture_dir" 2>&1 | tee "$evidence_dir/dumper.log"
  step "Normalize inference checkpoint for converter"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PREPARE_CHECKPOINT" --input "$checkpoint" --output "$prepared_checkpoint" --audit-json "$prepare_audit" 2>&1 | tee "$evidence_dir/checkpoint-prepare.log"
  [[ -s "$prepared_checkpoint" && -s "$prepare_audit" ]] || die "checkpoint preparation did not produce auditable outputs"
  verify_prepared_audit "$prepare_audit" "$prepared_checkpoint" | tee "$evidence_dir/checkpoint-prepare-verified.log"
  step "Convert with dedicated architecture"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli 2>&1 | tee "$evidence_dir/build.log"
  "$VOKRA_ROOT/target/release/vokra-cli" convert --model "$MODEL_KIND" --input "$prepared_checkpoint" --output "$corrected" --license "$LICENSE_SPDX" 2>&1 | tee "$evidence_dir/convert.log"
  verify_corrected_provenance "$corrected" | tee "$evidence_dir/corrected-contract.log"
  export VOKRA_VOICE_GENDER_GGUF="$corrected" VOKRA_VOICE_GENDER_PCM="$fixture_dir/pcm.f32" VOKRA_VOICE_GENDER_FEATURES="$fixture_dir/features.f32" VOKRA_VOICE_GENDER_EMBEDDING="$fixture_dir/embedding.f32" VOKRA_VOICE_GENDER_LOGITS="$fixture_dir/logits.f32" VOKRA_VOICE_GENDER_PROBABILITIES="$fixture_dir/probabilities.f32" VOKRA_VOICE_GENDER_FIXTURE_KIND="$FIXTURE_KIND"
  step "Run CPU parity"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked -p vokra-models --test parity_voice_gender_classifier -- --nocapture 2>&1 | tee "$evidence_dir/parity.log"
  verify_cpu_parity_log "$evidence_dir/parity.log" | tee "$evidence_dir/cpu-parity-gate.log"
  step "Run repository gates on VAST"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh" | tee "$evidence_dir/gates.log"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" | tee -a "$evidence_dir/gates.log"
  cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check | tee -a "$evidence_dir/gates.log"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace | tee -a "$evidence_dir/gates.log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace --all-targets -- -D warnings | tee -a "$evidence_dir/gates.log"
  (cd "$VOKRA_ROOT" && cargo deny check licenses advisories bans) | tee -a "$evidence_dir/gates.log"
  (cd "$VOKRA_ROOT" && cargo audit) | tee -a "$evidence_dir/gates.log"
  { echo 'execution_status=CPU_PASS_METAL_NOT_RUN'; echo "corrected_sha256=$(sha256_file "$corrected")"; echo "public_sha256=$(sha256_file "$public_artifact")"; echo "prepared_checkpoint_sha256=$(sha256_file "$prepared_checkpoint")"; echo "prepare_audit_sha256=$(sha256_file "$prepare_audit")"; echo "upstream_revision=$UPSTREAM_REVISION"; echo "cpu_parity=PASS"; echo "cpu_parity_bound=$FP32_PARITY_BOUND"; echo 'publication=NOT_PERFORMED'; } | tee "$evidence_dir/summary.txt"
}

main "$@"
