#!/usr/bin/env bash
# Darwin arm64 CSM-1B worker.  Evidence from VAST is a prerequisite; this
# script never downloads, converts, substitutes CPU for Metal, or reports
# readiness. It authenticates the complete official reference packet, but
# native binding is still BLOCKED; PCM bounds are MEASURED_NOT_GATED until
# registered separately.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
REFERENCE_PROJECT="$ROOT/tools/parity/csm_1b_reference"
die() { printf '[csm-1b-apple] ERROR: %s\n' "$*" >&2; exit 2; }

self_test() {
  local fail=0 token
  for token in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON CSM_VAST_BUNDLE \
    ACCEPTED_VAST_CPU_BASELINE COMPLETE_COMPOSITE_GGUF complete-artifact-manifest.json \
    VOKRA_CSM_PARITY_DIR VOKRA_CSM_BACKEND=cpu VOKRA_CSM_BACKEND=metal \
    exact=true MEASURED_NOT_GATED BLOCKED_BY_CPU BLOCKED_NATIVE_BINDING NO_UPLOAD \
    NOT_RUN_OFFICIAL_ONLY apply_chat_template audio_kwargs 2051; do
    grep -Fq -- "$token" "$0" || { echo "self-test missing $token" >&2; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push)([[:space:]]|$)' "$0" >/dev/null; then
    echo 'self-test found download/publication command' >&2; fail=1
  fi
  (( fail == 0 )) && echo 'apple-silicon-csm-1b.sh self-test: OK' || return 1
}

if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; exit 0; fi
[[ $# == 1 ]] || die 'usage: apple-silicon-csm-1b.sh VAST_BUNDLE | --self-test'
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'Darwin arm64 is required'
memory="$(sysctl -n hw.memsize 2>/dev/null || true)"
[[ "$memory" =~ ^[0-9]+$ && "$memory" -ge 34359738368 ]] || die 'at least 32 GiB RAM is required'
command -v uv >/dev/null 2>&1 || die 'uv is required'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is required'
xcrun -sdk macosx metal -v >/dev/null 2>&1 || die 'Metal compiler unavailable'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
bundle="$(cd "$1" 2>/dev/null && pwd)" || die 'bundle path invalid'
inspection_manifest="${CSM_INSPECTION_MANIFEST:-$bundle/inspection-manifest.json}"
[[ -f "$inspection_manifest" ]] || die 'authenticated inspection manifest is required'
for file in validation-manifest.json complete-artifact-manifest.json csm-1b-complete.gguf packet.json; do
  [[ -f "$bundle/$file" ]] || die "bundle input missing: $file"
done
if [[ -f "$bundle/reference/manifest.json" ]]; then
  reference_dir="$bundle/reference"
  reference_manifest="$reference_dir/manifest.json"
else
  reference_dir="$bundle"
  reference_manifest="$bundle/reference-manifest.json"
fi
[[ -f "$reference_manifest" ]] || die 'bundle input missing: reference/manifest.json or reference-manifest.json'
[[ -f "${CSM_ACCEPTED_VAST_CPU_BASELINE:-}" ]] || die 'ACCEPTED_VAST_CPU_BASELINE is required'
[[ -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated CSM reference pyproject.toml + uv.lock are required'
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - \
  "$bundle/validation-manifest.json" "$reference_manifest" "$bundle/complete-artifact-manifest.json" "$bundle/csm-1b-complete.gguf" "$reference_dir" "${CSM_ACCEPTED_VAST_CPU_BASELINE}" "$inspection_manifest" "$bundle/packet.json" <<'PY'
import hashlib, json, math, re, struct, sys
from pathlib import Path
def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SystemExit(f"duplicate manifest key: {key}")
            result[key] = value
        return result
    return json.load(open(path, encoding="utf-8"), object_pairs_hook=pairs)
validation = load(sys.argv[1])
inspection = load(sys.argv[7])
packet = load(sys.argv[8])
if validation.get("status") != "BLOCKED" or validation.get("publication") != "NO_UPLOAD":
    raise SystemExit("VAST validation manifest is not fail-closed")
if validation.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE" or inspection.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
    raise SystemExit("authenticated inspection/reference evidence is incomplete")
complete = load(sys.argv[3])
if complete.get("status") != "ACCEPTED" or complete.get("artifact_role") != "COMPLETE_CSM_MIMI_COMPOSITE":
    raise SystemExit("complete artifact is not accepted composite evidence")
if complete.get("model", {}).get("revision") != "c92a71e1c419772e25be7dc14d952c2521a740ab":
    raise SystemExit("complete artifact snapshot revision mismatch")
artifact_hash = complete.get("sha256") or complete.get("artifact_sha256") or complete.get("artifact", {}).get("sha256")
if not isinstance(artifact_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
    raise SystemExit("complete artifact manifest has no strong SHA-256 binding")
hasher = hashlib.sha256()
with open(sys.argv[4], "rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        hasher.update(block)
if hasher.hexdigest() != artifact_hash:
    raise SystemExit("complete artifact SHA-256 does not match its manifest")
baseline = load(sys.argv[6])
if baseline.get("status") != "ACCEPTED" or baseline.get("source") != "VAST_CPU":
    raise SystemExit("CPU baseline is not accepted VAST evidence")

ref_manifest = load(sys.argv[2])
if ref_manifest.get("status") != "BLOCKED" or ref_manifest.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE" or ref_manifest.get("comparison_status") != "NOT_RUN_OFFICIAL_ONLY" or ref_manifest.get("native_status") != "BLOCKED_NATIVE_BINDING" or ref_manifest.get("cpu_status") != "UNSUPPORTED" or ref_manifest.get("metal_status") != "BLOCKED_BY_CPU" or ref_manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("official reference evidence is incomplete")
if ref_manifest.get("model", {}).get("repository") != "sesame/csm-1b" or ref_manifest.get("model", {}).get("revision") != "c92a71e1c419772e25be7dc14d952c2521a740ab":
    raise SystemExit("official reference model identity is not pinned")
identity = ref_manifest.get("inspection_identity", {})
if identity.get("source_repository") != "https://github.com/SesameAILabs/csm.git" or identity.get("source_revision") != "8f6d947a26f6301deec9696f9bfb28e9e2e0d7d5" or identity.get("transformers_commit") != "945727948c1143a10ac6f7d811aa58bb0d126b5b":
    raise SystemExit("official reference source/Transformers identity is incomplete")
if identity.get("transformers_version") != "4.52.1" or ref_manifest.get("generation", {}).get("input_boundary") != "official processor.apply_chat_template(conversation, tokenize=False) then official CSM processor(text=rendered_prompt, audio=caller_numpy, sampling_rate=24000, return_tensors=pt)":
    raise SystemExit("official reference Transformers/input boundary is not pinned")
environment = ref_manifest.get("reference_environment", {})
if not isinstance(environment, dict) or not re.fullmatch(r"[0-9a-f]{64}", environment.get("lock_sha256", "")) or environment.get("python") != "3.12":
    raise SystemExit("dedicated reference lock identity is missing")
if environment.get("source_transformers_requirement") != "transformers==4.52.1" or environment.get("source_huggingface_hub_requirement") != "huggingface-hub>=0.30,<1.0" or environment.get("transformers_security_advisory") != "GHSA-xrqw-3rrv-vx5w" or environment.get("transformers_security_patched_minimum") != "5.10.0" or environment.get("isolated_transformers_pin") != "5.10.4" or environment.get("isolated_huggingface_hub_pin") != "1.5.0" or environment.get("transformers_compatibility_status") != "BLOCKED_UNVERIFIED_API_SMOKE":
    raise SystemExit("reference Transformers security/provenance metadata is incomplete")
if environment.get("packages", {}).get("numpy") != "2.2.6" or environment.get("packages", {}).get("torch_distribution") != "2.7.1" or environment.get("packages", {}).get("transformers") != "5.10.4":
    raise SystemExit("dedicated reference package versions are not pinned")
torch_runtime = environment.get("packages", {}).get("torch_runtime")
if not isinstance(torch_runtime, str) or not re.fullmatch(r"2\.7\.1(?:\+[^+ ]+)?", torch_runtime):
    raise SystemExit("runtime torch version is missing or incompatible with the locked distribution")
if ref_manifest.get("generation", {}).get("processor_call") != {"method": "apply_chat_template_then_official_processor", "chat_template_kwargs": {"tokenize": False}, "processor_kwargs": {"sampling_rate": 24000, "return_tensors": "pt"}, "audio_input": "authenticated caller-owned NumPy arrays", "audio_argument": "None when audio_placeholder_count=0; otherwise the ordered non-empty authenticated array list", "adapter": "pinned_upstream_ProcessorMixin_boundary; no_reference_mirror"}:
    raise SystemExit("official processor call arguments are not the fixed boundary")
tokenizer = ref_manifest.get("tokenizer", {})
if tokenizer.get("tokenizer_json_git_blob_sha1") != "8de5df033b78de76dbe15fdd8b934678b5017aaf" or not re.fullmatch(r"[0-9a-f]{64}", tokenizer.get("tokenizer_json_sha256", "")):
    raise SystemExit("official reference tokenizer identity is incomplete")
required = {"processor_input_ids.u32le", "generated_frame_codes.u32le", "backbone_logits.f32le", "backbone_scores.f32le", "backbone_hidden_last.f32le", "depth_decoder_logits.f32le", "official_pcm_pre_watermark.f32le"}
rows = ref_manifest.get("artifacts")
if not isinstance(rows, list) or len(rows) != len(required) or {row.get("path") for row in rows} != required:
    raise SystemExit("official reference artifact set is incomplete")
ref_dir = Path(sys.argv[5])
messages = packet.get("messages")
if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "0":
    raise SystemExit("reference conversation boundary is invalid")
packet_paths = []
for message in messages:
    if not isinstance(message, dict) or set(message) != {"role", "content"} or message.get("role") not in {"0", "1"} or not isinstance(message.get("content"), list):
        raise SystemExit("reference conversation message schema is invalid")
    for item in message["content"]:
        if not isinstance(item, dict) or item.get("type") not in {"text", "audio"}:
            raise SystemExit("reference conversation content item is invalid")
        if item["type"] == "audio":
            if set(item) != {"type", "path"} or not isinstance(item["path"], str) or Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
                raise SystemExit("reference audio content path is unsafe")
            packet_paths.append(item["path"])
        elif set(item) != {"type", "text"} or not isinstance(item["text"], str) or not item["text"] or "\x00" in item["text"]:
            raise SystemExit("reference text content item is invalid")
    if message is messages[-1] and (any(item.get("type") == "audio" for item in message["content"]) or not any(item.get("type") == "text" for item in message["content"])):
        raise SystemExit("last conversation message must be target text without audio")
audio_rows = ref_manifest.get("audio_inputs")
if not isinstance(audio_rows, list) or len(audio_rows) != len(packet_paths):
    raise SystemExit("reference audio placeholder count is not authenticated")
for raw_path, row in zip(packet_paths, audio_rows):
    if not isinstance(row, dict) or row.get("path") != raw_path:
        raise SystemExit("reference audio placeholder order differs from packet evidence")
if ref_manifest.get("generation", {}).get("audio_placeholder_count") != len(packet_paths) or ref_manifest.get("generation", {}).get("decoded_audio_count") != 1:
    raise SystemExit("reference audio placeholder/decoded output cardinality is invalid")
expected_conversation_paths = [str((Path(sys.argv[8]).resolve().parent / raw_path).resolve()) for raw_path in packet_paths]
if ref_manifest.get("generation", {}).get("conversation_audio_paths") != expected_conversation_paths:
    raise SystemExit("official conversation audio paths are not bound to packet paths")
if {p.name for p in ref_dir.iterdir()} != required | {"manifest.json"}:
    raise SystemExit("reference directory contains stale or unlisted files")
for row in rows:
    name = row.get("path")
    path = ref_dir / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise SystemExit(f"reference artifact missing or empty: {name}")
    if row.get("bytes") != path.stat().st_size or hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
        raise SystemExit(f"reference artifact hash/size mismatch: {name}")
    shape = row.get("shape")
    if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise SystemExit(f"reference artifact shape invalid: {name}")
    if row.get("dtype") == "float32":
        values = struct.unpack("<" + "f" * (row["bytes"] // 4), path.read_bytes())
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"reference artifact contains non-finite values: {name}")
shapes = {row["path"]: row["shape"] for row in rows}
codes_shape = shapes["generated_frame_codes.u32le"]
if len(codes_shape) != 3 or codes_shape[0] <= 0 or codes_shape[1] <= 0 or codes_shape[2] != 32:
    raise SystemExit("generated codes must have shape [batch,frames,32]")
batch, frames = codes_shape[:2]
if batch != 1:
    raise SystemExit("reference artifact format is mono and requires batch=1")
generation = ref_manifest.get("generation", {})
if generation.get("logit_selection_contract") != "backbone_processed_scores_argmax_selected_codebook0; backbone_raw_logits_recorded; depth_raw_hook_argmax_exact_trace" or generation.get("depth_selection_contract") != "depth_raw_hook_logits_argmax_matches_codebooks1_to31; processed_depth_scores_unavailable":
    raise SystemExit("reference logit selection contract is not source-shaped")
generation_config_contract = generation.get("generation_config_contract")
if not isinstance(generation_config_contract, dict) or generation_config_contract.get("reference_overrides") != {"do_sample": False, "depth_decoder_do_sample": False} or generation_config_contract.get("final_generation_semantics") != "reference_overrides_are_applied_at_model_generate_boundary":
    raise SystemExit("generation config source/override contract is missing")
if not re.fullmatch(r"[0-9a-f]{64}", generation.get("generation_config_sha256", "")):
    raise SystemExit("authenticated generation_config identity is missing")
if shapes["backbone_logits.f32le"] != [batch, frames, 2051] or shapes["backbone_scores.f32le"] != [batch, frames, 2051] or shapes["backbone_hidden_last.f32le"] != [batch, frames, 2048] or shapes["depth_decoder_logits.f32le"] != [frames * 31, batch, 1, 2051]:
    raise SystemExit("reference frame/depth/PCM shape relation failed")
codes_row = next(row for row in rows if row["path"] == "generated_frame_codes.u32le")
codes = struct.unpack("<" + "I" * (codes_row["bytes"] // 4), (ref_dir / codes_row["path"]).read_bytes())
if any(code >= 2048 for code in codes):
    raise SystemExit("generated code exceeds the 2048-entry codebook")
official_eos = [index for index in range(frames) if all(code == 0 for code in codes[index * 32:index * 32 + 31])]
codec_eos = [index for index in range(frames) if all(code == 0 for code in codes[index * 32:(index + 1) * 32])]
if official_eos and official_eos != [frames - 1]:
    raise SystemExit("official 31-codebook EOS is not a final frame")
if codec_eos and codec_eos != [frames - 1]:
    raise SystemExit("codec 32-codebook EOS is not a final frame")
if generation.get("official_eos_frame_indices") != official_eos or generation.get("codec_eos_frame_indices") != codec_eos:
    raise SystemExit("official/codec EOS evidence differs from generated codes")
if generation.get("official_eos_codebook_count") != 31 or generation.get("codec_eos_codebook_count") != 32:
    raise SystemExit("official/codec EOS codebook cardinalities are not authenticated")
decoded_frames = generation.get("codec_decoded_frame_count_by_batch", [None])[0]
if not isinstance(decoded_frames, int) or decoded_frames != (codec_eos[0] if codec_eos else frames):
    raise SystemExit("codec EOS decoded-frame cutoff is not source-shaped")
if decoded_frames <= 0 or shapes["official_pcm_pre_watermark.f32le"] != [decoded_frames * 1920]:
    raise SystemExit("PCM is not aligned to the pre-EOS decoded frame count")
depth_row = next(row for row in rows if row["path"] == "depth_decoder_logits.f32le")
depth_values = struct.unpack("<" + "f" * (depth_row["bytes"] // 4), (ref_dir / depth_row["path"]).read_bytes())
for call in range(frames * 31):
    logits = depth_values[call * 2051:(call + 1) * 2051]
    if max(range(2051), key=lambda index: logits[index]) != codes[(call // 31) * 32 + 1 + call % 31]:
        raise SystemExit("depth decoder argmax does not match generated codebook 1..31")
backbone_row = next(row for row in rows if row["path"] == "backbone_scores.f32le")
backbone_values = struct.unpack("<" + "f" * (backbone_row["bytes"] // 4), (ref_dir / backbone_row["path"]).read_bytes())
backbone_vocab = backbone_row["shape"][2]
for frame in range(frames):
    logits = backbone_values[frame * backbone_vocab:(frame + 1) * backbone_vocab]
    if max(range(backbone_vocab), key=lambda index: logits[index]) != codes[frame * 32]:
        raise SystemExit("backbone argmax does not match generated codebook 0")
if ref_manifest.get("generation", {}).get("depth_decoder_call_count") != frames * 31 or ref_manifest["generation"].get("backbone_hidden_generation_steps") != frames or ref_manifest["generation"].get("decoded_frame_count_by_batch") != [decoded_frames]:
    raise SystemExit("depth decoder/hidden cardinality is not frame-aligned")
input_shapes = ref_manifest["generation"].get("depth_decoder_input_shapes")
if not isinstance(input_shapes, list) or len(input_shapes) != frames * 31 or any(shape[-1] != (2 if index % 31 == 0 else 1) for index, shape in enumerate(input_shapes)):
    raise SystemExit("depth decoder prefill/step input cardinality is missing or incorrect")
count = ref_manifest["generation"]["depth_decoder_call_count"]
if ref_manifest["generation"].get("depth_decoder_call_order") != list(range(count)):
    raise SystemExit("depth decoder call order evidence is missing")
PY
# The current Rust test still consumes the retired manifest.txt fixture and
# cannot authenticate generated_frame_codes.u32le.  Stop after authenticating
# the VAST packet instead of running a stale test or implying CPU/Metal parity.
echo '[csm-1b-apple] BLOCKED_NATIVE_BINDING: authenticated VAST evidence is present, but no native CSM composite test consumes the new official reference packet; CPU/Metal not run; PCM=MEASURED_NOT_GATED; NO_UPLOAD.' >&2
exit 2
