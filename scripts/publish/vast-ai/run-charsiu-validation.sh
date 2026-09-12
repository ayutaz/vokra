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
# The committed fixture was generated under the same pinned source/runtime as
# the VAST worker, but its logits are not required to be byte-identical across
# the two separately observed library builds.  The measured regeneration
# delta was max_abs=5.7220458984375e-06 and RMSE=2.048987880698405e-06;
# these independent reference bounds are exactly 2x those observations. Keep
# them distinct from the native Rust parity tolerance above: this gate checks
# reference regeneration stability, not Rust-vs-reference parity.
REFERENCE_REGEN_MEASURED_MAX_ABS="0.0000057220458984375"
REFERENCE_REGEN_MEASURED_RMSE="0.000002048987880698405"
REFERENCE_REGEN_BOUND_FACTOR="2.000000000"
REFERENCE_REGEN_MAX_ABS_BOUND="0.0000114440917968750"
REFERENCE_REGEN_RMSE_BOUND="0.00000409797576139681"
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

verify_charsiu_license_signoff() {
  local audit_file="$1"
  [[ -f "$audit_file" ]] || { die "license audit file missing: $audit_file"; return 2; }
  # Match one complete Markdown table row only.  Removing lightweight
  # decoration makes **Charsiu**, `repo`, and plain-text variants equivalent,
  # while checking the model, license, and approval columns independently.
  # Identity fragments from notes or separate rows can never combine into
  # approval, and malformed/duplicate identity rows fail closed.
  if ! awk -F'|' '
    function normalized_cell(cell) {
      # Keep underscores: they are part of the exact checkpoint identity.
      gsub(/[\`*]/, "", cell)
      sub(/^[[:space:]]+/, "", cell)
      sub(/[[:space:]]+$/, "", cell)
      return cell
    }
    /^\|/ {
      # With leading/trailing separators, the audit schema is $2=model,
      # $3=license, and $6=Commercial/Research-only/Rejected decision.
      model = normalized_cell($2)
      has_name = model ~ /(^|[^[:alnum:]])Charsiu([^[:alnum:]]|$)/
      has_repo = model ~ /(^|[^[:alnum:]_.-])lingjzhu\/charsiu([^[:alnum:]_.-]|$)/
      has_checkpoint = model ~ /(^|[^[:alnum:]_.-])charsiu\/en_w2v2_fc_10ms([^[:alnum:]_.-]|$)/
      # Any row carrying a Charsiu identity fragment is a candidate.  This
      # catches split/malformed rows instead of allowing a valid row plus a
      # stray fragment to pass unnoticed.
      if (has_name || has_repo || has_checkpoint) {
        identity_rows++
        if (NF != 8) {
          malformed = 1
          next
        }
        license = normalized_cell($3)
        approval = normalized_cell($6)
        if (has_name && has_repo && has_checkpoint \
            && license == "MIT" \
            && approval ~ /☑[[:space:]]+Commercial/ \
            && approval !~ /☐[[:space:]]+Commercial/) {
          accepted++
        }
      }
    }
    END { exit(!malformed && identity_rows == 1 && accepted == 1 ? 0 : 1) }
  ' "$audit_file"; then
    die 'existing MIT Charsiu sign-off row is missing'
    return 2
  fi
  log "Charsiu license gate authenticated: exact repo/checkpoint, MIT, and Commercial sign-off share one audit row"
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
  verify_charsiu_license_signoff "$VOKRA_ROOT/docs/license-audit.md"
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
    "$generated" "$FIXTURE_DIR" "$UPSTREAM_REVISION" "$CHECKPOINT_SHA256" "$CONFIG_SHA256" \
    "$FIXTURE_PCM_SHA256" "$FIXTURE_LOGITS_SHA256" "$REFERENCE_REGEN_MEASURED_MAX_ABS" \
    "$REFERENCE_REGEN_MEASURED_RMSE" "$REFERENCE_REGEN_BOUND_FACTOR" \
    "$REFERENCE_REGEN_MAX_ABS_BOUND" "$REFERENCE_REGEN_RMSE_BOUND" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

generated = Path(sys.argv[1])
fixture = Path(sys.argv[2])
revision, checkpoint_sha, config_sha, pcm_sha, logits_sha = sys.argv[3:8]
measured_max, measured_rmse, factor, max_bound, rmse_bound = map(float, sys.argv[8:])


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"Charsiu reference authentication failed: {message}")


if (
    not all(math.isfinite(value) for value in (measured_max, measured_rmse, factor, max_bound, rmse_bound))
    or measured_max <= 0.0
    or measured_rmse <= 0.0
    or factor <= 1.0
    or max_bound <= measured_max
    or rmse_bound <= measured_rmse
    or not math.isclose(max_bound, measured_max * factor, rel_tol=0.0, abs_tol=1e-15)
    or not math.isclose(rmse_bound, measured_rmse * factor, rel_tol=0.0, abs_tol=1e-15)
):
    fail("reference regeneration bound is not finite, derived, and strictly above its measured baseline")


def load_manifest(root: Path, label: str) -> dict:
    path = root / "manifest.json"
    if not path.is_file() or path.is_symlink():
        fail(f"{label} manifest is missing or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"{label} manifest is malformed: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} manifest root is not an object")
    return value


def bytes_hash(path: Path, label: str) -> tuple[bytes, str]:
    if not path.is_file() or path.is_symlink():
        fail(f"{label} is missing or symlinked")
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def f32_values(path: Path, expected_shape: tuple[int, ...], label: str):
    import numpy as np

    data, digest = bytes_hash(path, label)
    if len(data) % 4:
        fail(f"{label} is not f32-aligned")
    values = np.frombuffer(data, dtype="<f4")
    if int(np.prod(expected_shape)) != values.size:
        fail(f"{label} shape {values.shape} != {expected_shape}")
    if not np.isfinite(values).all():
        fail(f"{label} contains non-finite values")
    return values.reshape(expected_shape), digest


fixture_manifest = load_manifest(fixture, "committed fixture")
generated_manifest = load_manifest(generated, "generated reference")
for label, manifest in (("committed fixture", fixture_manifest), ("generated reference", generated_manifest)):
    if manifest.get("revision") != revision:
        fail(f"{label} revision is not the pinned revision")
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        fail(f"{label} checkpoint identity is not pinned")
    if manifest.get("config_sha256") != config_sha:
        fail(f"{label} config identity is not pinned")
    if manifest.get("reference_implementation") != "transformers.Wav2Vec2ForCTC":
        fail(f"{label} is not generated by the official Transformers implementation")

fixture_pcm, fixture_pcm_hash = f32_values(fixture / "pcm_400.f32.bin", (400,), "committed PCM")
fixture_logits, fixture_logits_hash = f32_values(
    fixture / "logits_1x42.f32.bin", (1, 42), "committed logits"
)
if fixture_pcm_hash != pcm_sha:
    fail(f"committed PCM hash {fixture_pcm_hash} != pinned {pcm_sha}")
if fixture_logits_hash != logits_sha:
    fail(f"committed logits hash {fixture_logits_hash} != pinned {logits_sha}")

generated_pcm, generated_pcm_hash = f32_values(
    generated / "pcm_400.f32.bin", (400,), "generated PCM"
)
generated_logits, generated_logits_hash = f32_values(
    generated / "logits_1x42.f32.bin", (1, 42), "generated logits"
)
if generated_manifest.get("pcm_shape") != [400] or generated_manifest.get("logits_shape") != [1, 42]:
    fail("generated reference manifest shape is malformed")
if generated_manifest.get("pcm_sha256") != generated_pcm_hash:
    fail("generated PCM manifest hash does not match its bytes")
if generated_manifest.get("logits_sha256") != generated_logits_hash:
    fail("generated logits manifest hash does not match its bytes")
if generated_pcm_hash != pcm_sha or generated_pcm.tobytes() != fixture_pcm.tobytes():
    fail("generated PCM is not byte-identical to the pinned fixture")

import numpy as np

delta = np.abs(generated_logits.astype(np.float64) - fixture_logits.astype(np.float64))
max_abs = float(delta.max())
rmse = float(math.sqrt(np.mean(np.square(delta))))
differing = int(np.count_nonzero(generated_logits != fixture_logits))
if max_abs > max_bound or rmse > rmse_bound:
    fail(
        f"regeneration delta exceeds independent bound: max_abs={max_abs:.17g} "
        f"(bound={max_bound:.17g}), rmse={rmse:.17g} (bound={rmse_bound:.17g})"
    )
print(
    "CHARSIU_REFERENCE_REGEN_METRICS "
    f"schema=vokra-charsiu-reference-regeneration-v1 "
    f"committed_logits_sha256={fixture_logits_hash} generated_logits_sha256={generated_logits_hash} "
    f"frames=1 logits=42 max_abs={max_abs:.17g} rmse={rmse:.17g} differing_values={differing} "
    f"measured_max_abs={measured_max:.17g} measured_rmse={measured_rmse:.17g} "
    f"bound_factor={factor:.9f} max_abs_bound={max_bound:.17g} rmse_bound={rmse_bound:.17g}"
)
print(
    "CHARSIU_REFERENCE_REGEN_VERIFICATION "
    f"status=PASS pcm=byte-identical revision={revision} checkpoint_sha256={checkpoint_sha} "
    f"config_sha256={config_sha} reference=transformers.Wav2Vec2ForCTC "
    f"torch={generated_manifest.get('torch_version')} transformers={generated_manifest.get('transformers_version')}"
)
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
import math
import struct
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools" / "audit"))
from gguf_manifest import read_manifest
metadata, tensors = read_manifest(Path(sys.argv[2]))
expected_revision = "e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f"
expected_checkpoint = "6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1"
expected_vocab = [
    "[SIL]", "NG", "F", "M", "AE", "R", "UW", "N", "IY", "AW", "V", "UH", "OW",
    "AA", "ER", "HH", "Z", "K", "CH", "W", "EY", "ZH", "T", "EH", "Y", "AH", "B",
    "P", "TH", "DH", "AO", "G", "L", "JH", "OY", "SH", "D", "AY", "S", "IH", "[UNK]",
    "[PAD]",
]
expected_metadata = {
    "general.gguf_version", "vokra.model.arch", "vokra.model.name", "vokra.model.category",
    "vokra.charsiu.revision", "vokra.charsiu.checkpoint_sha256", "vokra.charsiu.hidden_size",
    "vokra.charsiu.ffn_dim", "vokra.charsiu.n_layer", "vokra.charsiu.n_head",
    "vokra.charsiu.vocab_size", "vokra.charsiu.silence_id", "vokra.charsiu.pad_id",
    "vokra.charsiu.sample_rate", "vokra.charsiu.frame_shift_sec", "vokra.charsiu.layer_norm_eps",
    "vokra.charsiu.pos_conv_kernel", "vokra.charsiu.pos_conv_groups",
    "vokra.charsiu.silence_threshold", "vokra.charsiu.vocab", "vokra.provenance.weight_license",
    "vokra.provenance.license", "vokra.provenance.model_id", "vokra.provenance.source",
    "vokra.schema.version", "vokra.schema.producer",
}
if set(metadata) != expected_metadata:
    raise SystemExit(
        "Charsiu GGUF metadata key set mismatch: "
        f"missing={sorted(expected_metadata - set(metadata))} extra={sorted(set(metadata) - expected_metadata)}"
    )
expected_values = {
    "general.gguf_version": 3,
    "vokra.model.arch": "charsiu",
    "vokra.model.name": "charsiu/en_w2v2_fc_10ms",
    "vokra.model.category": "alignment",
    "vokra.charsiu.revision": expected_revision,
    "vokra.charsiu.checkpoint_sha256": expected_checkpoint,
    "vokra.charsiu.hidden_size": 768,
    "vokra.charsiu.ffn_dim": 3072,
    "vokra.charsiu.n_layer": 12,
    "vokra.charsiu.n_head": 12,
    "vokra.charsiu.vocab_size": 42,
    "vokra.charsiu.silence_id": 0,
    "vokra.charsiu.pad_id": 41,
    "vokra.charsiu.sample_rate": 16000,
    "vokra.charsiu.pos_conv_kernel": 128,
    "vokra.charsiu.pos_conv_groups": 16,
    "vokra.charsiu.silence_threshold": 4,
    "vokra.charsiu.vocab": expected_vocab,
    "vokra.provenance.weight_license": "permissive",
    "vokra.provenance.license": "MIT",
    "vokra.provenance.model_id": "charsiu",
    "vokra.provenance.source": "charsiu/en_w2v2_fc_10ms",
    "vokra.schema.version": 1,
    "vokra.schema.producer": "vokra-core 0.3.0",
}
for key, expected in expected_values.items():
    actual = metadata.get(key)
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-7):
            raise SystemExit(f"Charsiu GGUF metadata {key}={actual!r} != {expected!r}")
    elif actual != expected:
        raise SystemExit(f"Charsiu GGUF metadata {key}={actual!r} != {expected!r}")
for key, expected in {
    "vokra.charsiu.frame_shift_sec": 0.01,
    "vokra.charsiu.layer_norm_eps": 1e-5,
}.items():
    actual = metadata.get(key)
    if struct.pack("<f", float(actual)) != struct.pack("<f", expected):
        raise SystemExit(f"Charsiu GGUF metadata {key} has non-canonical F32 bits: {actual!r}")

expected_names = []
stem_kernels = [10, 3, 3, 3, 3, 2, 2]
for index, _kernel in enumerate(stem_kernels):
    expected_names.append(f"wav2vec2.feature_extractor.conv_layers.{index}.conv.weight")
    if index == 0:
        expected_names.extend([
            "wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight",
            "wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias",
        ])
expected_names.extend([
    "wav2vec2.feature_projection.layer_norm.weight",
    "wav2vec2.feature_projection.layer_norm.bias",
    "wav2vec2.feature_projection.projection.weight",
    "wav2vec2.feature_projection.projection.bias",
    "charsiu.pos_conv.weight",
    "charsiu.pos_conv.bias",
    "wav2vec2.encoder.layer_norm.weight",
    "wav2vec2.encoder.layer_norm.bias",
])
for index in range(12):
    prefix = f"wav2vec2.encoder.layers.{index}"
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
        for suffix in ("weight", "bias"):
            expected_names.append(f"{prefix}.attention.{projection}.{suffix}")
    for norm in ("layer_norm", "final_layer_norm"):
        for suffix in ("weight", "bias"):
            expected_names.append(f"{prefix}.{norm}.{suffix}")
    for dense in ("intermediate_dense", "output_dense"):
        for suffix in ("weight", "bias"):
            expected_names.append(f"{prefix}.feed_forward.{dense}.{suffix}")
expected_names.extend(["lm_head.weight", "lm_head.bias"])
actual_names = [tensor["name"] for tensor in tensors]
if len(tensors) != 211 or actual_names != expected_names:
    raise SystemExit(
        f"Charsiu GGUF tensor manifest mismatch: count={len(tensors)} expected=211 "
        f"names_match={actual_names == expected_names}"
    )
for tensor in tensors:
    if tensor["ggml_type"] != 0:
        raise SystemExit(f"Charsiu GGUF tensor {tensor['name']} is not F32")
print(
    "CHARSIU_GGUF_VERIFICATION "
    "status=PASS arch_key=vokra.model.arch arch=charsiu metadata_keys=26 "
    "metadata_on_disk=25 tensors=211 tensor_dtype=F32 provenance=PINNED"
)
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
    'REFERENCE_REGEN_MEASURED_MAX_ABS' 'REFERENCE_REGEN_MEASURED_RMSE' \
    'REFERENCE_REGEN_BOUND_FACTOR' 'REFERENCE_REGEN_MAX_ABS_BOUND' 'REFERENCE_REGEN_RMSE_BOUND' \
    'CHARSIU_REFERENCE_REGEN_METRICS' 'CHARSIU_REFERENCE_REGEN_VERIFICATION' \
    'vokra.model.arch' 'CHARSIU_GGUF_VERIFICATION' \
    'CHARSIU_OFFICIAL_PARITY_METRICS' 'CHARSIU_OFFICIAL_PARITY PASS' 'verify_parity_log' \
    'verify_charsiu_license_signoff' 'lingjzhu/charsiu' 'charsiu/en_w2v2_fc_10ms' \
    'publication=NO_UPLOAD' 'archive_sha256='; do
    grep -Fq -- "$required" "$script_path" || { log "self-test missing contract: $required"; fail=1; }
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log 'self-test found direct Python invocation'; fail=1
  fi
  if grep -En 'git[[:space:]]+(push|clone|fetch|pull)|--(push|upload|publish)' "$script_path" >/dev/null; then
    log 'self-test found forbidden publication/source command'; fail=1
  fi
  local obsolete_arch_key="general."'architecture'
  if grep -Fq "metadata.get(\"$obsolete_arch_key\")" "$script_path"; then
    log 'self-test found obsolete GGUF architecture key'; fail=1
  fi
  grep -Fq '"vokra.model.arch": "charsiu"' "$script_path" || {
    log 'self-test missing canonical Vokra architecture key'; fail=1;
  }
  local obsolete_producer='vokra-convert '"0.3.0"
  if grep -Fq "\"vokra.schema.producer\": \"$obsolete_producer\"" "$script_path"; then
    log 'self-test found obsolete converter producer stamp'; fail=1
  fi
  grep -Fq '"vokra.schema.producer": "vokra-core 0.3.0"' "$script_path" || {
    log 'self-test missing canonical core producer stamp'; fail=1;
  }
  verify_charsiu_license_signoff "$VOKRA_ROOT/docs/license-audit.md" >/dev/null || {
    log 'self-test rejected the current Charsiu license row'; fail=1;
  }
  printf '%s\n' \
    '| **Charsiu** (`lingjzhu/charsiu`; runtime checkpoint `charsiu/en_w2v2_fc_10ms`) | MIT | source | signoff | **☑ Commercial** | notes |' \
    > "$temporary/license-current.md"
  verify_charsiu_license_signoff "$temporary/license-current.md" >/dev/null || {
    log 'self-test rejected a valid decorated Charsiu license row'; fail=1;
  }
  printf '%s\n' \
    '| Charsiu (`lingjzhu/charsiu`; runtime checkpoint `charsiu/en_w2v2_fc_10ms`) | Apache-2.0 | notes include an MIT grant | signoff | ☑ Commercial | notes |' \
    > "$temporary/license-apache-notes-mit.md"
  if verify_charsiu_license_signoff "$temporary/license-apache-notes-mit.md" >/dev/null 2>&1; then
    log 'self-test accepted Charsiu row with Apache-2.0 license despite notes mentioning MIT'; fail=1
  fi
  printf '%s\n' \
    '| Charsiu (`lingjzhu/charsiu`; runtime checkpoint `charsiu/en_w2v2_fc_10ms`) | Apache-2.0 | notes | signoff | ☑ Commercial | notes |' \
    > "$temporary/license-no-mit.md"
  if verify_charsiu_license_signoff "$temporary/license-no-mit.md" >/dev/null 2>&1; then
    log 'self-test accepted Charsiu row without MIT'; fail=1
  fi
  printf '%s\n' \
    '| Charsiu (`lingjzhu/charsiu`; runtime checkpoint `charsiu/en_w2v2_fc_10ms`) | MIT | notes | signoff | ☐ Commercial | notes |' \
    > "$temporary/license-no-commercial.md"
  if verify_charsiu_license_signoff "$temporary/license-no-commercial.md" >/dev/null 2>&1; then
    log 'self-test accepted Charsiu row without Commercial sign-off'; fail=1
  fi
  printf '%s\n' \
    '| Charsiu (`lingjzhu/charsiu`) | MIT | notes | signoff | ☐ Commercial | notes |' \
    '| runtime checkpoint `charsiu/en_w2v2_fc_10ms` | Apache-2.0 | notes | signoff | ☑ Commercial | notes |' \
    > "$temporary/license-split.md"
  if verify_charsiu_license_signoff "$temporary/license-split.md" >/dev/null 2>&1; then
    log 'self-test combined Charsiu license fragments from separate rows'; fail=1
  fi
  local under_bound over_bound nonfinite malformed_shape malformed_manifest
  under_bound="$temporary/reference-under-bound"
  over_bound="$temporary/reference-over-bound"
  nonfinite="$temporary/reference-nonfinite"
  malformed_shape="$temporary/reference-malformed-shape"
  malformed_manifest="$temporary/reference-malformed-manifest"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --offline --python 3.12 python - \
    "$FIXTURE_DIR" "$under_bound" "$over_bound" "$nonfinite" "$malformed_shape" "$malformed_manifest" \
    "$REFERENCE_REGEN_MAX_ABS_BOUND" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

source, under, over, nonfinite, malformed_shape, malformed_manifest = map(Path, sys.argv[1:7])
bound = float(sys.argv[7])
pcm_name = "pcm_400.f32.bin"
logits_name = "logits_1x42.f32.bin"
source_logits = np.fromfile(source / logits_name, dtype="<f4")
source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))


def write_case(destination: Path, values: np.ndarray, shape: list[int] | None = None) -> None:
    destination.mkdir(parents=True)
    shutil.copyfile(source / pcm_name, destination / pcm_name)
    values.astype("<f4", copy=False).tofile(destination / logits_name)
    manifest = dict(source_manifest)
    manifest["logits_shape"] = shape if shape is not None else [1, 42]
    manifest["logits_sha256"] = hashlib.sha256((destination / logits_name).read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


under_values = source_logits.copy()
candidate = under_values[0]
while True:
    next_candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
    if float(next_candidate) - float(source_logits[0]) >= bound:
        break
    candidate = next_candidate
under_values[0] = candidate
write_case(under, under_values)

over_values = under_values.copy()
candidate = over_values[0]
while float(candidate) - float(source_logits[0]) <= bound:
    candidate = np.nextafter(candidate, np.float32(np.inf), dtype=np.float32)
over_values[0] = candidate
write_case(over, over_values)

nonfinite_values = source_logits.copy()
nonfinite_values[0] = np.nan
write_case(nonfinite, nonfinite_values)
write_case(malformed_shape, source_logits[:-1], [1, 42])
malformed_manifest.mkdir(parents=True)
shutil.copyfile(source / pcm_name, malformed_manifest / pcm_name)
shutil.copyfile(source / logits_name, malformed_manifest / logits_name)
(malformed_manifest / "manifest.json").write_text('{"revision":', encoding="utf-8")
PY
  verify_generated_reference "$under_bound" >/dev/null || {
    log 'self-test rejected a just-under-bound regeneration fixture'; fail=1;
  }
  if verify_generated_reference "$over_bound" >/dev/null 2>&1; then
    log 'self-test accepted an over-bound regeneration fixture'; fail=1
  fi
  if verify_generated_reference "$nonfinite" >/dev/null 2>&1; then
    log 'self-test accepted a non-finite regeneration fixture'; fail=1
  fi
  if verify_generated_reference "$malformed_shape" >/dev/null 2>&1; then
    log 'self-test accepted a malformed-shape regeneration fixture'; fail=1
  fi
  if verify_generated_reference "$malformed_manifest" >/dev/null 2>&1; then
    log 'self-test accepted a malformed regeneration manifest'; fail=1
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
