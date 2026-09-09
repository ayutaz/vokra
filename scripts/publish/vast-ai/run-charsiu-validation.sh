#!/usr/bin/env bash
# VAST-only real-checkpoint Charsiu conversion and CPU parity worker.
# The worker never uploads, publishes, or leaves a model artifact behind.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
REFERENCE_DUMPER="$PARITY_PROJECT/charsiu_dump_reference.py"
BRIDGE="$PARITY_PROJECT/nemo_pt_to_safetensors.py"
FIXTURE_DIR="$VOKRA_ROOT/crates/vokra-models/tests/fixtures/charsiu"
MODEL_KIND="charsiu"
UPSTREAM_REPO="charsiu/en_w2v2_fc_10ms"
UPSTREAM_REVISION="e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f"
CHECKPOINT_FILE="pytorch_model.bin"
CHECKPOINT_BYTES=377706220
CHECKPOINT_SHA256="6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1"
CONFIG_FILE="config.json"
CONFIG_SHA256="7406aa4f917267640865688aa62f2337664a3abb9a49a2f204d932b53aeb6cb7"
FIXTURE_PCM_SHA256="77658830c60a39ff6269db6d3c5bd6b3a3d596e8ba4c61d3b30c7c9b27343e5c"
FIXTURE_LOGITS_SHA256="6785ffc5426a71193ebe37614f434e2220853164acdf53250838762abfcef8b3"
FP32_ATOL="0.000200000"
MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

log() { printf '[charsiu-vast] %s\n' "$*" >&2; }
step() { printf '\n[charsiu-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-charsiu-validation.sh [--work-dir <empty-dir>]
       run-charsiu-validation.sh --self-test

VAST/Linux-only real Charsiu checkpoint conversion and Transformers parity.
The normal path requires VOKRA_PUBLISH_ON_VAST=1 and performs no upload.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

verify_file() {
  local path="$1" expected_sha256="$2" expected_bytes="${3:-}" actual_bytes actual_sha256
  [[ -f "$path" && ! -L "$path" ]] || { die "missing or symlinked file: $path"; return 2; }
  if [[ -n "$expected_bytes" ]]; then
    actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
    [[ "$actual_bytes" == "$expected_bytes" ]] || { die "byte-size mismatch for $path"; return 2; }
  fi
  actual_sha256="$(sha256_file "$path")"
  [[ "$actual_sha256" == "$expected_sha256" ]] || { die "SHA-256 mismatch for $path"; return 2; }
  log "identity OK: $(basename "$path") sha256=$actual_sha256${actual_bytes:+ bytes=$actual_bytes}"
}

require_vast_host() {
  local memory free_disk scratch
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is required'
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Charsiu worker requires Linux x86_64'
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ ]] || die 'could not read physical memory'
  (( memory >= MIN_VAST_MEM_KIB )) || die 'at least 64 GiB RAM is required'
  scratch="${VOKRA_SCRATCH:-$HOME/scratchpad}"
  mkdir -p "$scratch"
  free_disk="$(df -Pk "$scratch" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk" =~ ^[0-9]+$ ]] || die 'could not read free disk'
  (( free_disk >= MIN_FREE_DISK_KIB )) || die 'at least 150 GB free disk is required'
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr df; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project missing'
  [[ -f "$REFERENCE_DUMPER" && -f "$BRIDGE" ]] || die 'Charsiu parity tools missing'
  [[ -f "$FIXTURE_DIR/manifest.json" && -f "$FIXTURE_DIR/pcm_400.f32.bin" && -f "$FIXTURE_DIR/logits_1x42.f32.bin" ]] || die 'committed Charsiu fixture missing'
  grep -Fq "Charsiu (\`lingjzhu/charsiu\`; runtime checkpoint \`charsiu/en_w2v2_fc_10ms\`)" "$VOKRA_ROOT/docs/license-audit.md" \
    || die 'existing MIT Charsiu sign-off row is missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
}

download_hf_file() {
  local filename="$1" output_dir="$2"
  mkdir -p "$output_dir"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$filename" "$output_dir"
}

verify_generated_reference() {
  local generated="$1"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python - \
    "$generated" "$FIXTURE_DIR" "$UPSTREAM_REVISION" "$CHECKPOINT_SHA256" "$CONFIG_SHA256" "$FIXTURE_PCM_SHA256" "$FIXTURE_LOGITS_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

generated = Path(sys.argv[1])
fixture = Path(sys.argv[2])
revision, checkpoint_sha, config_sha, pcm_sha, logits_sha = sys.argv[3:]
manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("revision") != revision or manifest.get("checkpoint_sha256") != checkpoint_sha or manifest.get("config_sha256") != config_sha:
    raise SystemExit("generated reference identity mismatch")
for path, expected in ((generated / "pcm_400.f32.bin", pcm_sha), (generated / "logits_1x42.f32.bin", logits_sha)):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"generated reference {path.name} hash {actual} != committed {expected}")
for name in ("pcm_400.f32.bin", "logits_1x42.f32.bin"):
    if (generated / name).read_bytes() != (fixture / name).read_bytes():
        raise SystemExit(f"generated reference differs from committed {name}")
print("Charsiu reference authenticated: official Transformers implementation and committed fixture are byte-identical")
PY
}

verify_prepared_manifest() {
  local manifest="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (d.get("kept_count"), d.get("dropped_count"), d.get("unknown_stripped")) != (213, 0, []):
    raise SystemExit(f"unexpected Charsiu bridge manifest: {d}")
print("Charsiu safetensors bridge authenticated: kept=213 dropped_int=0 stripped_unknown=0")
PY
}

verify_gguf() {
  local artifact="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$VOKRA_ROOT" "$artifact" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools" / "audit"))
from gguf_manifest import read_manifest
metadata, tensors = read_manifest(Path(sys.argv[2]))
if metadata.get("general.architecture") != "charsiu" or len(tensors) != 211:
    raise SystemExit("Charsiu GGUF manifest is not the canonical 211-tensor artifact")
if metadata.get("vokra.charsiu.revision") != "e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f" or metadata.get("vokra.charsiu.checkpoint_sha256") != "6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1":
    raise SystemExit("Charsiu GGUF provenance mismatch")
print("Charsiu GGUF contract authenticated: arch=charsiu tensors=211 pinned provenance=OK")
PY
}

verify_parity_log() {
  local log_path="$1" metrics marker_count pass_count
  marker_count="$(grep -Ec '^CHARSIU_OFFICIAL_PARITY(_METRICS| ).*$' "$log_path" || true)"
  [[ "$marker_count" == 2 ]] || { die "Charsiu parity marker count is not exactly 2: $marker_count"; return 2; }
  metrics="$(grep -E '^CHARSIU_OFFICIAL_PARITY_METRICS frames=[0-9]+ logits=[0-9]+ max_abs=[0-9]+\.[0-9]{9} index=[0-9]+ rust=-?[0-9]+\.[0-9]{9} transformers=-?[0-9]+\.[0-9]{9} atol=0\.000200000$' "$log_path" || true)"
  [[ -n "$metrics" ]] || { die 'Charsiu parity metrics marker missing or malformed'; return 2; }
  [[ "$(printf '%s\n' "$metrics" | wc -l | tr -d '[:space:]')" == 1 ]] || { die 'Charsiu parity metrics marker duplicated'; return 2; }
  pass_count="$(grep -Ec '^CHARSIU_OFFICIAL_PARITY PASS max_abs=[0-9]+\.[0-9]{9} atol=0\.000200000 frames=[0-9]+ reference=transformers\.Wav2Vec2ForCTC fixture=official_canned_pcm$' "$log_path" || true)"
  [[ "$pass_count" == 1 ]] || { die 'Charsiu parity PASS marker missing, duplicated, or malformed'; return 2; }
  printf '%s\n' "$metrics" | awk '{ for (i = 2; i <= NF; i++) { split($i, pair, "="); if (pair[1] == "max_abs" && (pair[2] + 0) > 0.0002) exit 1 } }' \
    || { die 'Charsiu parity metric exceeds FP32 bound'; return 2; }
  printf 'Charsiu parity authenticated: atol=%s\n%s\n' "$FP32_ATOL" "$metrics"
}

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-charsiu-worker.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  for required in "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_SHA256" "$CONFIG_SHA256" \
    'charsiu_dump_reference.py' 'nemo_pt_to_safetensors.py' 'parity_charsiu' 'FP32_ATOL' \
    'CHARSIU_OFFICIAL_PARITY_METRICS' 'CHARSIU_OFFICIAL_PARITY PASS' 'verify_parity_log' \
    'publication=NO_UPLOAD' 'archive_sha256='; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing contract: $required"; fail=1; }
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log 'self-test found direct Python invocation'; fail=1
  fi
  if grep -En 'git[[:space:]]+(push|clone|fetch|pull)|--(push|upload|publish)' "$script_path" >/dev/null; then
    log 'self-test found forbidden publication/source command'; fail=1
  fi
  printf '%s\n' \
    'CHARSIU_OFFICIAL_PARITY_METRICS frames=1 logits=42 max_abs=0.000199999 index=0 rust=1.000000000 transformers=1.000000000 atol=0.000200000' \
    'CHARSIU_OFFICIAL_PARITY PASS max_abs=0.000199999 atol=0.000200000 frames=1 reference=transformers.Wav2Vec2ForCTC fixture=official_canned_pcm' \
    > "$temporary/valid.log"
  verify_parity_log "$temporary/valid.log" >/dev/null || { log 'self-test rejected valid parity log'; fail=1; }
  printf '%s\n' \
    'CHARSIU_OFFICIAL_PARITY_METRICS frames=1 logits=42 max_abs=0.000200001 index=0 rust=1.000000000 transformers=1.000000000 atol=0.000200000' \
    'CHARSIU_OFFICIAL_PARITY PASS max_abs=0.000200001 atol=0.000200000 frames=1 reference=transformers.Wav2Vec2ForCTC fixture=official_canned_pcm' \
    > "$temporary/over-bound.log"
  if verify_parity_log "$temporary/over-bound.log" >/dev/null 2>&1; then
    log 'self-test accepted over-bound parity metric'; fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --offline --python 3.12 python "$REFERENCE_DUMPER" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  echo 'run-charsiu-validation.sh self-test: PASS'
)

main() {
  local work_dir='' self_test=0 work_seen=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir) (( work_seen == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || return 2; work_dir="$2"; work_seen=1; shift 2 ;;
      --self-test) (( self_test == 0 )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self_test )); then
    [[ -z "$work_dir" ]] || die '--self-test accepts no other arguments'
    run_self_test
    return $?
  fi
  require_vast_host
  require_tooling
  local stamp root input checkpoint config generated prepared manifest artifact evidence parity_log archive
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  root="${work_dir:-${VOKRA_SCRATCH:-$HOME/scratchpad}/charsiu-validation/$stamp}"
  [[ ! -e "$root" && ! -L "$root" ]] || die 'work directory must be absent and non-symlink'
  mkdir -p "$root/input" "$root/reference" "$root/evidence"
  input="$root/input"; checkpoint="$input/$CHECKPOINT_FILE"; config="$input/$CONFIG_FILE"
  generated="$root/reference"; prepared="$root/input/charsiu.safetensors"; manifest="$prepared.stripped-manifest.json"
  artifact="$root/charsiu.gguf"; evidence="$root/evidence"; parity_log="$evidence/parity.log"

  step 'Download and authenticate exact Charsiu checkpoint/config'
  download_hf_file "$CHECKPOINT_FILE" "$input"
  download_hf_file "$CONFIG_FILE" "$input"
  verify_file "$checkpoint" "$CHECKPOINT_SHA256" "$CHECKPOINT_BYTES"
  verify_file "$config" "$CONFIG_SHA256"
  step 'Generate independent official Transformers reference'
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python "$REFERENCE_DUMPER" \
    --checkpoint-bin "$checkpoint" --config "$config" --outdir "$generated" | tee "$evidence/reference.json"
  verify_generated_reference "$generated" | tee "$evidence/reference-verified.log"
  step 'Flatten checkpoint to safetensors for the converter'
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python "$BRIDGE" \
    --input "$checkpoint" --output "$prepared" | tee "$evidence/bridge.log"
  verify_prepared_manifest "$manifest" | tee "$evidence/bridge-verified.log"
  step 'Build converter and produce strict Charsiu GGUF'
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli 2>&1 | tee "$evidence/build.log"
  "$VOKRA_ROOT/target/release/vokra-cli" convert --model "$MODEL_KIND" --input "$prepared" --output "$artifact" 2>&1 | tee "$evidence/convert.log"
  verify_gguf "$artifact" | tee "$evidence/gguf-verified.log"
  step 'Run real CPU parity against independent fixture'
  VOKRA_CHARSIU_GGUF="$artifact" cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test parity_charsiu -- --nocapture 2>&1 | tee "$parity_log"
  grep -Eq 'test result: ok\. [1-9][0-9]* passed' "$parity_log" || die 'Charsiu parity test did not pass'
  verify_parity_log "$parity_log" | tee "$evidence/parity-verified.log"
  step 'Run lightweight repository gates'
  cargo fmt --all -- --check 2>&1 | tee "$evidence/fmt.log"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" 2>&1 | tee "$evidence/zero-deps.log"
  archive="$root/charsiu-evidence.tar.gz"
  tar -czf "$archive" -C "$root" evidence reference
  {
    echo 'execution_status=PASS'
    echo "upstream_repository=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "checkpoint_sha256=$CHECKPOINT_SHA256"
    echo "config_sha256=$CONFIG_SHA256"
    echo "gguf_sha256=$(sha256_file "$artifact")"
    echo "reference_manifest_sha256=$(sha256_file "$generated/manifest.json")"
    echo "evidence_archive_sha256=$(sha256_file "$archive")"
    echo "fp32_atol=$FP32_ATOL"
    echo 'cpu_parity=PASS'
    echo 'publication=NO_UPLOAD'
    echo 'vast_destroy=REQUIRED_AFTER_EVIDENCE_CAPTURE'
  } | tee "$evidence/summary.txt"
  log "PASS: evidence archive=$archive; pull only logs/reference manifest, then destroy this VAST instance"
}

main "$@"
