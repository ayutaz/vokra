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
MODEL_KIND="voice-gender-classifier"
LICENSE_SPDX="mit"
UPSTREAM_REPO="JaesungHuh/voice-gender-classifier"
UPSTREAM_REVISION="49bcbecfd929ba5a043bde645fdff1a375eb79c7"
UPSTREAM_GITHUB_URL="https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_HF_REVISION="db1222153bd60337e900be22add7af180452adc0"
UPSTREAM_FILE="model.safetensors"
PUBLIC_REPO="vokra/voice-gender-classifier"
PUBLIC_REVISION="94c8d0ba41cfe2f7b8a773eb4a7982cf4facbc84"
PUBLIC_FILE="voice-gender-classifier.restamped.gguf"
PUBLIC_SHA256="e1e61f1493601087f5db5867c4f750ec99d6b11223b5323bd120e3c21e8f957f"
CHECKPOINT_SHA256=""
MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

log() { printf '[voice-gender-vast] %s\n' "$*" >&2; }
step() { printf '\n[voice-gender-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

preflight_gate() {
  die "fixed upstream checkpoint byte identity/license gate is unresolved; arbitrary --checkpoint-sha256 is not accepted"
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

verify_file() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || { die "missing or symlinked file: $path"; return 2; }
  [[ "$(sha256_file "$path")" == "$expected" ]] || {
    die "SHA-256 mismatch for $path"
    return 2
  }
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

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" fail=0 required
  # shellcheck disable=SC2016 # literal contract tokens intentionally keep quoting
  for required in "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_GITHUB_URL" \
    "$UPSTREAM_HF_REVISION" "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" \
    "$PUBLIC_SHA256" "$MODEL_KIND" "$LICENSE_SPDX" "voice_gender_classifier_dump_reference.py" \
    "CARGO_BUILD_JOBS=\"\${CARGO_BUILD_JOBS:-1}\"" 'VOKRA_PUBLISH_ON_VAST' \
    'MIN_VAST_MEM_KIB=67108864' 'MIN_FREE_DISK_KIB=150000000' \
    '"$VOKRA_ROOT/target/release/vokra-cli" convert' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked -p vokra-models' \
    'parity_voice_gender_classifier' 'VOICE_GENDER_OFFICIAL_PARITY MEASURED_NOT_GATED' \
    'VOICE_GENDER_METAL_VS_CPU MEASURED_NOT_GATED' 'verify_corrected_provenance'; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing: $required"; fail=1; }
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test found direct Python invocation"; fail=1
  fi
  if grep -En -- '(^|[[:space:]])(git[[:space:]]+push|.*upload|.*publish|--push|--upload|--publish)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test found publication operation"; fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PARITY_DUMPER" --self-test >/dev/null || fail=1
  "$script_path" --self-test --work-dir /tmp/invalid >/dev/null 2>&1 && { log "self-test accepted extra argument"; fail=1; } || true
  (( fail == 0 )) || return 1
  echo "run-voice-gender-classifier-validation.sh self-test: PASS"
}

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
  CHECKPOINT_SHA256="$(printf '%s' "$checkpoint_sha256" | tr '[:upper:]' '[:lower:]')"
  preflight_gate
  require_vast_host
  require_tooling
  local run_stamp work_dir input_dir source_dir fixture_dir evidence_dir checkpoint public_artifact corrected
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${work_arg:-$VOKRA_SCRATCH/voice-gender-validation/$run_stamp}"
  [[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work directory must be absent or empty"
  input_dir="$work_dir/input"; source_dir="$input_dir/source"; fixture_dir="$work_dir/fixtures"; evidence_dir="$work_dir/evidence"
  checkpoint="$input_dir/checkpoint/$UPSTREAM_FILE"; public_artifact="$input_dir/public/$PUBLIC_FILE"; corrected="$work_dir/voice-gender-classifier-corrected.gguf"
  mkdir -p "$fixture_dir" "$evidence_dir" "$(dirname "$checkpoint")" "$(dirname "$public_artifact")"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-voice-gender"
  step "Record environment"
  { date -u; uname -a; git -C "$VOKRA_ROOT" rev-parse HEAD; echo "upstream_revision=$UPSTREAM_REVISION"; echo "upstream_hf_revision=$UPSTREAM_HF_REVISION"; echo "checkpoint_sha256=$CHECKPOINT_SHA256"; } > "$evidence_dir/environment.txt"
  step "Authenticate historical public artifact"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$(dirname "$public_artifact")"
  verify_file "$public_artifact" "$PUBLIC_SHA256"
  verify_artifact_contract "$public_artifact" | tee "$evidence_dir/public-contract.log"
  step "Fetch checkpoint and exact upstream source"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_HF_REVISION" "$UPSTREAM_FILE" "$(dirname "$checkpoint")"
  verify_file "$checkpoint" "$CHECKPOINT_SHA256"
  git clone --no-checkout "$UPSTREAM_GITHUB_URL" "$source_dir"
  git -C "$source_dir" checkout --detach "$UPSTREAM_REVISION"
  [[ -z "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]] || die "upstream source checkout is dirty"
  step "Generate independent official fixtures"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PARITY_DUMPER" --checkpoint "$checkpoint" --upstream-src "$source_dir" --canned --out-dir "$fixture_dir" 2>&1 | tee "$evidence_dir/dumper.log"
  step "Convert with dedicated architecture"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli 2>&1 | tee "$evidence_dir/build.log"
  "$VOKRA_ROOT/target/release/vokra-cli" convert --model "$MODEL_KIND" --input "$checkpoint" --output "$corrected" --license "$LICENSE_SPDX" 2>&1 | tee "$evidence_dir/convert.log"
  verify_corrected_provenance "$corrected" | tee "$evidence_dir/corrected-contract.log"
  export VOKRA_VOICE_GENDER_GGUF="$corrected" VOKRA_VOICE_GENDER_PCM="$fixture_dir/pcm.f32" VOKRA_VOICE_GENDER_FEATURES="$fixture_dir/features.f32" VOKRA_VOICE_GENDER_EMBEDDING="$fixture_dir/embedding.f32" VOKRA_VOICE_GENDER_LOGITS="$fixture_dir/logits.f32" VOKRA_VOICE_GENDER_PROBABILITIES="$fixture_dir/probabilities.f32"
  step "Run CPU parity"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked -p vokra-models --test parity_voice_gender_classifier -- --nocapture 2>&1 | tee "$evidence_dir/parity.log"
  grep -Fq 'VOICE_GENDER_OFFICIAL_PARITY MEASURED_NOT_GATED' "$evidence_dir/parity.log" || die "CPU parity measurement marker absent"
  step "Run repository gates on VAST"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh" | tee "$evidence_dir/gates.log"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" | tee -a "$evidence_dir/gates.log"
  cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check | tee -a "$evidence_dir/gates.log"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace | tee -a "$evidence_dir/gates.log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace --all-targets -- -D warnings | tee "$evidence_dir/gates.log"
  (cd "$VOKRA_ROOT" && cargo deny check licenses advisories bans) | tee -a "$evidence_dir/gates.log"
  (cd "$VOKRA_ROOT" && cargo audit) | tee -a "$evidence_dir/gates.log"
  { echo 'execution_status=MEASURED_NOT_GATED'; echo "corrected_sha256=$(sha256_file "$corrected")"; echo "public_sha256=$(sha256_file "$public_artifact")"; echo "upstream_revision=$UPSTREAM_REVISION"; echo 'cpu_parity=MEASURED_NOT_GATED'; echo 'publication=NOT_PERFORMED'; } | tee "$evidence_dir/summary.txt"
}

main "$@"
