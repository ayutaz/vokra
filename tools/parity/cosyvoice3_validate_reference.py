#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice3_reference --python 3.12 python
"""Independent strict validator for CosyVoice3 reference evidence."""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from typing import Any

EXPECTED_FORMAT = "vokra-cosyvoice3-official-reference-v1"
TARGET_TEXT = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
PROMPT_SHA256 = "c7b31d6dbe7cc6a716dded00550db5b50940bf209e424e4ad207b12e657c8ff6"
SOURCE_TRANSFORMERS_REQUIREMENT = "transformers==4.51.3"
SOURCE_HUGGINGFACE_HUB_REQUIREMENT = "huggingface-hub==0.24.7"
TRANSFORMERS_SECURITY_ADVISORY = "GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM = "5.10.0"
ISOLATED_TRANSFORMERS_PIN = "5.10.4"
ISOLATED_HUGGINGFACE_HUB_PIN = "1.5.0"
TRANSFORMERS_COMPATIBILITY_STATUS = "BLOCKED_UNVERIFIED_API_SMOKE"
PROJECT_VERSIONS = {
    "conformer": "0.3.2", "diffusers": "0.29.0", "hyperpyyaml": "1.2.2",
    "einops": "0.8.0", "inflect": "7.3.1", "huggingface-hub": "1.5.0",
    "librosa": "0.10.2", "modelscope": "1.20.0", "numpy": "1.26.4",
    "omegaconf": "2.3.0", "onnx": "1.16.0", "onnxruntime": "1.18.0",
    "openai-whisper": "20231117", "pyworld": "0.3.4", "pyyaml": "6.0.2",
    "soundfile": "0.12.1", "torch": "2.3.1", "torchaudio": "2.3.1",
    "transformers": ISOLATED_TRANSFORMERS_PIN, "tqdm": "4.66.5", "wetext": "0.0.4",
}
# Must match the reference dumper's transport sanity bound.  It is not a
# model-value tolerance and does not relax numerical parity thresholds.
MAX_TENSOR_ABS = 1_000_000.0
REQUIRED = {
    "tokenizer_ids", "prompt_speech_tokens", "campplus_embedding", "qwen_prompt_embeddings",
    "ras_calls", "ras_logits", "ras_multinomial_probability", "generated_speech_tokens",
    "flow_rand_noise_full", "flow_rand_noise_slice", "flow_encoder_output", "cfm_terminal_mel",
    "prompt_mel", "generated_mel", "hift_input_mel", "hift_output_pcm", "official_output_pcm",
    "decoder_mu", "decoder_noise", "cfm_estimator_x", "cfm_estimator_mu", "cfm_estimator_t",
    "cfm_estimator_output", "cfm_estimator_mask", "cfm_estimator_spks",
    "cfm_estimator_cond", "decoder_mask", "decoder_spks", "decoder_cond",
    "cfm_solver_trace",
}
JSON_ROLES = {"ras_calls", "generated_speech_tokens", "cfm_solver_trace"}
CFM_CFG_RATE = 0.7
SOURCE_ROLES = {
    "cosyvoice/llm/llm.py", "cosyvoice/flow/flow.py", "cosyvoice/flow/flow_matching.py",
    "cosyvoice/hifigan/generator.py", "cosyvoice/cli/model.py",
    "examples/libritts/cosyvoice3/conf/cosyvoice3.yaml", "example.py",
    "asset/zero_shot_prompt.wav",
}
SOURCE_ROLE_BLOBS = {
    "cosyvoice/llm/llm.py": "b17bd3af7abc3135c32f0b1d5d4dba7b59f15b1f",
    "cosyvoice/flow/flow.py": "c25518621bf98e95d5ed75b83c5a2a610d0822be",
    "cosyvoice/flow/flow_matching.py": "d3beb9ec2ce8c26972433080c458f90c3f2c0467",
    "cosyvoice/hifigan/generator.py": "bbc2a2112bfd260963765af33760c95c3161fe14",
    "cosyvoice/cli/model.py": "25610a4235c6a971821e64e275d2b3d1d9f0c669",
    "examples/libritts/cosyvoice3/conf/cosyvoice3.yaml": "36dfee4889b6f1c5a1e85e59a0e716005aa0063f",
    "example.py": "7e9dd98e4bf72790664c86223ae991fa2a48bb57",
    "asset/zero_shot_prompt.wav": "a7b9d954289ddf5c90a4dc4f0a912edaea10945a",
}
MATCHA_ROLE_BLOBS = {
    "matcha/models/components/flow_matching.py": "5cad7431ef66a8d11da32a77c1af7f6e31d6b774",
    "matcha/models/components/decoder.py": "1137cd7008e9d07b4f306926a82e44c2b2cddbdf",
}
MODEL_IDENTITIES = {
    ".gitattributes": (1574, "cb90d04a72c0b51e3a115db0a9342dd88af903a4", None),
    "CosyVoice-BlankEN/config.json": (659, "463b055262b6c66c4629a74a4b300bfe2ed31d3c", None),
    "CosyVoice-BlankEN/generation_config.json": (242, "dfc11073787daf1b0f9c0f1499487ab5f4c93738", None),
    "CosyVoice-BlankEN/merges.txt": (1402109, "90d3d82d027eadcc6a5e77c38eb82d43fc51b53b", None),
    "CosyVoice-BlankEN/model.safetensors": (988097824, "3dff8ababe3dbf3bd7a556f5f143503ab2ef3c98", "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96"),
    "CosyVoice-BlankEN/tokenizer_config.json": (1287, "ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e", None),
    "CosyVoice-BlankEN/vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569", None),
    "README.md": (11982, "d816a921470cff1b6926d31c89e4ec7dea185f32", None),
    "asset/dingding.png": (122824, "e407a9d3c0fc5a7fcac46aef09181a0bef330d37", "7f04815e2e676d31b089af6fa270135f3214f2193d5e0ad98b491d007d48f1c6"),
    "campplus.onnx": (28303423, "7b08523b2e28e437cfb1a0312723a5ab0bac287e", "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"),
    "config.json": (2, "9e26dfeeb6e641a33dae4961196235bdb965b21b", None),
    "configuration.json": (47, "5e812fae901c12933ac69ebf3eb79d0eb49bbab4", None),
    "cosyvoice3.yaml": (6934, "2eda7e5007d99f6b17fbe7bd751cf54e3cde29ea", None),
    "flow.decoder.estimator.fp32.onnx": (1326216933, "3f880eeae966a725cd7c875b8e4c929bf2035489", "9b51b9533a55937762b262bf2cf9c6220ce40760f76d6532cb16a6a6d84059a8"),
    "flow.pt": (1329116148, "074b96e9cfbf3e511067528bde8e76a308f94904", "a6fab32a7825e5b0bc855ddd948f8db9370b0a786fbc249caa4595e95b608e4b"),
    "hift.pt": (83202622, "c5088ac4f7db1314a4efe06ca60e9f47ea2a1900", "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65"),
    "llm.pt": (2024669519, "d9813d1f616910e9117d612e6e725b0350f98115", "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970"),
    "llm.rl.pt": (2024682701, "bc852bfe713ca73dcb2d900145731e6ce2c4a3c2", "74d34b01a80c7154670ae75ac372d1b1712c78bceae9f467eb9f1f6f61ec764f"),
    "speech_tokenizer_v3.batch.onnx": (969451579, "3a4fb34ea654cdfc4e4228b2d485c196af2985fa", "b156b8a7bbff436585e153f4637b9a368009005ac66efa108a6c8bfb34e5ee43"),
    "speech_tokenizer_v3.onnx": (969451503, "91daac1b6f0bbcb54b8885dc7a1cbf054de22f94", "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d"),
}
MATCHA_ROLES = {"matcha/models/components/flow_matching.py", "matcha/models/components/decoder.py"}
MODEL_TREE = {
    ".gitattributes", "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt", "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json", "CosyVoice-BlankEN/vocab.json", "README.md",
    "asset/dingding.png", "campplus.onnx", "config.json", "configuration.json", "cosyvoice3.yaml",
    "flow.decoder.estimator.fp32.onnx", "flow.pt", "hift.pt", "llm.pt", "llm.rl.pt",
    "speech_tokenizer_v3.batch.onnx", "speech_tokenizer_v3.onnx",
}


def pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError("unsafe artifact path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path escapes evidence directory")
    return value


def check_tensor(record: dict[str, Any], root: Path) -> None:
    required = {"path", "bytes", "sha256", "shape", "dtype", "storage_dtype", "source"}
    if set(record) - required - {"metadata"} or not required <= set(record):
        raise ValueError("tensor record schema mismatch")
    path = root / safe_relative(record["path"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError("tensor artifact missing/non-regular/empty")
    if not isinstance(record["bytes"], int) or isinstance(record["bytes"], bool) or record["bytes"] != path.stat().st_size:
        raise ValueError("tensor byte count mismatch")
    if not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in record["sha256"]):
        raise ValueError("tensor digest schema mismatch")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError("tensor digest mismatch")
    shape = record["shape"]
    if not isinstance(shape, list) or not shape or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError("tensor shape mismatch")
    if not isinstance(record["dtype"], str) or not record["dtype"] or not isinstance(record["source"], str) or not record["source"]:
        raise ValueError("tensor dtype/source metadata mismatch")
    elements = math.prod(shape)
    if record["storage_dtype"] != "float32" or record["bytes"] != elements * 4:
        raise ValueError("tensor storage dtype/shape/size mismatch")
    raw = path.read_bytes()
    for (value,) in struct.iter_unpack("<f", raw):
        if not math.isfinite(value) or abs(value) > MAX_TENSOR_ABS:
            raise ValueError("non-finite or unbounded tensor artifact")
    metadata = record.get("metadata")
    if record["source"].endswith("PCM") or "pcm" in record["source"].lower() or "pcm" in record["path"].lower():
        if not isinstance(metadata, dict) or metadata.get("sample_rate") != 24_000:
            raise ValueError("PCM sample-rate metadata mismatch")
        if metadata.get("samples") != shape[-1]:
            raise ValueError("PCM sample-count metadata mismatch")
        if any(abs(value) > 1.1 for (value,) in struct.iter_unpack("<f", raw)):
            raise ValueError("PCM exceeds normalized bounds")


def tensor_values(record: dict[str, Any], root: Path) -> list[float]:
    path = root / safe_relative(record["path"])
    return [value for (value,) in struct.iter_unpack("<f", path.read_bytes())]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def f32_ulp(value: float) -> float:
    """Return one adjacent float32 step (not a float64 nextafter step)."""
    value = f32(value)
    if not math.isfinite(value):
        raise ValueError("ULP requires a finite float32")
    bits = f32_bits(value)
    if value == 0.0:
        return f32(struct.unpack("<f", struct.pack("<I", 1))[0])
    # Ordered by numerical value, with negative numbers descending in the
    # sign-magnitude encoding.  This is sufficient for the adjacent step.
    next_bits = bits - 1 if bits & 0x80000000 else bits + 1
    adjacent = struct.unpack("<f", struct.pack("<I", next_bits))[0]
    return abs(f32(adjacent) - value)


def f32_close(actual: float, expected: float) -> bool:
    """Allow four float32 ulps for separate multiply/add rounding."""
    actual, expected = f32(actual), f32(expected)
    return abs(actual - expected) <= 4.0 * max(f32_ulp(actual), f32_ulp(expected))


def validate_cfg_rows(values: list[float], base_values: list[float], role: str, duplicate: bool) -> None:
    """Check the exact two-row construction in flow_matching.py."""
    if len(values) != 2 * len(base_values) or not base_values:
        raise ValueError(f"CFM {role} row cardinality is invalid")
    row_size = len(base_values)
    if values[:row_size] != base_values:
        raise ValueError(f"CFM {role} conditional row is not bound to the solver input")
    if duplicate:
        if values[row_size:] != base_values:
            raise ValueError(f"CFM {role} rows are not duplicated by the source CFG setup")
    elif any(value != 0.0 for value in values[row_size:]):
        raise ValueError(f"CFM {role} unconditional row is not zero")


def validate_project() -> dict[str, Any]:
    project = Path(__file__).parent / "cosyvoice3_reference"
    pyproject, lock = project / "pyproject.toml", project / "uv.lock"
    if not pyproject.is_file() or not lock.is_file():
        raise ValueError("dedicated CosyVoice3 pyproject/uv.lock missing")
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    route = config.get("tool", {}).get("vokra", {}).get("cosyvoice3_reference", {})
    expected_route = {
        "source_transformers_requirement": SOURCE_TRANSFORMERS_REQUIREMENT,
        "source_huggingface_hub_requirement": SOURCE_HUGGINGFACE_HUB_REQUIREMENT,
        "transformers_security_advisory": TRANSFORMERS_SECURITY_ADVISORY,
        "transformers_security_patched_minimum": TRANSFORMERS_SECURITY_PATCHED_MINIMUM,
        "isolated_transformers_pin": ISOLATED_TRANSFORMERS_PIN,
        "isolated_huggingface_hub_pin": ISOLATED_HUGGINGFACE_HUB_PIN,
        "transformers_compatibility_status": TRANSFORMERS_COMPATIBILITY_STATUS,
    }
    if any(route.get(key) != value for key, value in expected_route.items()):
        raise ValueError("CosyVoice3 Transformers security metadata drifted")
    dependencies = {}
    for raw in config.get("project", {}).get("dependencies", []):
        name, pinned = raw.split("==")
        dependencies[name.lower()] = pinned
    if dependencies != PROJECT_VERSIONS:
        raise ValueError("CosyVoice3 direct dependency inventory drift")
    lock_text = lock.read_text(encoding="utf-8")
    for name, expected in PROJECT_VERSIONS.items():
        match = re.search(rf'(?m)^name = "{re.escape(name)}"\nversion = "([^"]+)"', lock_text)
        if not match or match.group(1) != expected:
            raise ValueError(f"CosyVoice3 lock version mismatch: {name}")
    actual = {}
    for name, expected in PROJECT_VERSIONS.items():
        try:
            value = installed_version(name)
        except PackageNotFoundError as error:
            raise ValueError(f"CosyVoice3 installed dependency missing: {name}") from error
        if value != expected:
            raise ValueError(f"CosyVoice3 installed dependency drift: {name}")
        actual[name] = value
    return {
        "python": ">=3.12,<3.13",
        "pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "dependencies": PROJECT_VERSIONS,
        "actual_versions": actual,
        "source_transformers_requirement": SOURCE_TRANSFORMERS_REQUIREMENT,
        "source_huggingface_hub_requirement": SOURCE_HUGGINGFACE_HUB_REQUIREMENT,
        "transformers_security_advisory": TRANSFORMERS_SECURITY_ADVISORY,
        "transformers_security_patched_minimum": TRANSFORMERS_SECURITY_PATCHED_MINIMUM,
        "isolated_transformers_pin": ISOLATED_TRANSFORMERS_PIN,
        "isolated_huggingface_hub_pin": ISOLATED_HUGGINGFACE_HUB_PIN,
        "transformers_compatibility_status": TRANSFORMERS_COMPATIBILITY_STATUS,
    }


def validate(manifest_path: Path) -> None:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    project = validate_project()
    if not isinstance(manifest, dict) or manifest.get("format") != EXPECTED_FORMAT or manifest.get("status") != "AUTHENTICATED_REFERENCE_EVIDENCE":
        raise ValueError("manifest status/format is not authenticated")
    if manifest.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED" or manifest.get("native_status") != "BLOCKED" or manifest.get("publication") != "NO_UPLOAD" or manifest.get("comparison_status") != "NOT_RUN_OFFICIAL_ONLY":
        raise ValueError("manifest fail-closed/publication status mismatch")
    source = manifest.get("source")
    matcha = manifest.get("matcha")
    model = manifest.get("model")
    if not isinstance(source, dict) or source.get("repository") != "FunAudioLLM/CosyVoice" or source.get("origin") != "https://github.com/FunAudioLLM/CosyVoice" or source.get("clean") is not True or source.get("revision") != "0d990d60740bf174904a5185cce910b847bd3684":
        raise ValueError("source/model revision mismatch")
    if not isinstance(matcha, dict) or matcha.get("repository") != "shivammehta25/Matcha-TTS" or matcha.get("origin") != "https://github.com/shivammehta25/Matcha-TTS" or matcha.get("clean") is not True or matcha.get("revision") != "dd9105b34bf2be2230f4aa1e4769fb586a3c824e":
        raise ValueError("Matcha revision mismatch")
    if manifest.get("project") != project or not isinstance(manifest.get("input"), dict) or manifest["input"].get("target_text") != TARGET_TEXT or manifest["input"].get("prompt_text") != PROMPT_TEXT or manifest["input"].get("prompt_wav") != "asset/zero_shot_prompt.wav" or manifest["input"].get("prompt_sha256") != PROMPT_SHA256 or manifest["input"].get("source_role", {}).get("git_blob_sha1") != SOURCE_ROLE_BLOBS["asset/zero_shot_prompt.wav"] or manifest["input"].get("source_role", {}).get("sha256") != PROMPT_SHA256 or isinstance(manifest["input"].get("seed"), bool) or not isinstance(manifest["input"].get("seed"), int):
        raise ValueError("fixed prompt/input identity mismatch")
    if not isinstance(model, dict) or model.get("repository") != "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" or model.get("revision") != "29e01c4e8d000f4bcd70751be16fa94bf3d85a18" or model.get("resolved_revision") != model.get("revision"):
        raise ValueError("HF model revision mismatch")
    for identity, roles in ((source, SOURCE_ROLES), (matcha, MATCHA_ROLES), (model, MODEL_TREE)):
        files = identity.get("files")
        if not isinstance(files, dict) or set(files) != roles:
            raise ValueError("authenticated source/Matcha/HF tree mismatch")
        for relative, record in files.items():
            if not isinstance(record, dict) or not isinstance(record.get("bytes"), int) or record["bytes"] <= 0 or not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                raise ValueError("source/Matcha/HF file identity record mismatch")
            if identity is source and (relative not in SOURCE_ROLE_BLOBS or record.get("git_blob_sha1") != SOURCE_ROLE_BLOBS[relative]):
                raise ValueError(f"source role Git blob mismatch: {relative}")
            if identity is matcha and (relative not in MATCHA_ROLE_BLOBS or record.get("git_blob_sha1") != MATCHA_ROLE_BLOBS[relative]):
                raise ValueError(f"Matcha role Git blob mismatch: {relative}")
            if identity is model:
                expected = MODEL_IDENTITIES.get(relative)
                if expected is None or record.get("bytes") != expected[0] or record.get("git_blob_sha1") != expected[1] or record.get("lfs_sha256") != expected[2]:
                    raise ValueError(f"HF model role identity mismatch: {relative}")
    if manifest.get("flow_noise_shape") != [1, 80, 15000] or manifest.get("sample_rate") != 24_000:
        raise ValueError("flow/sample-rate contract mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED:
        raise ValueError(f"artifact role set mismatch: {set(artifacts or {})}")
    for role in REQUIRED:
        rows = artifacts[role]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"empty artifact role: {role}")
        if role in JSON_ROLES:
            if len(rows) != 1 or not isinstance(rows[0], dict) or set(rows[0]) != {"value", "source"}:
                raise ValueError(f"JSON artifact schema/cardinality mismatch: {role}")
        else:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"tensor row is not an object: {role}")
                check_tensor(row, root)
    if len(artifacts["tokenizer_ids"]) != 2 or len(artifacts["prompt_speech_tokens"]) != 1 or len(artifacts["campplus_embedding"]) != 1 or any(len(artifacts[role]) != 1 for role in ("decoder_mu", "decoder_noise", "decoder_mask", "decoder_spks", "decoder_cond")):
        raise ValueError("frontend artifact cardinality mismatch")
    generated = artifacts["generated_speech_tokens"][0]["value"]
    if not isinstance(generated, list) or not generated or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 6761 for x in generated):
        raise ValueError("generated speech-token list mismatch")
    mel_shape = artifacts["generated_mel"][0]["shape"]
    if mel_shape != [1, 80, len(generated) * 2]:
        raise ValueError("generated mel/token ratio mismatch")
    noise_shape = artifacts["flow_rand_noise_full"][0]["shape"]
    if noise_shape != [1, 80, 15000] or artifacts["flow_rand_noise_slice"][0]["shape"] != artifacts["decoder_noise"][0]["shape"] or artifacts["cfm_terminal_mel"][0]["shape"][0:2] != [1, 80]:
        raise ValueError("flow noise tensor axes mismatch")
    observations = manifest.get("observations")
    if not isinstance(observations, dict) or observations.get("flow_noise_seed") != 0:
        raise ValueError("CFM seed evidence missing")
    ras_contract = observations.get("ras_sampling_contract")
    if not isinstance(ras_contract, dict) or ras_contract.get("ignore_eos_max_ras_calls") != 101 or ras_contract.get("sampling") != 25 or ras_contract.get("top_p") != 0.8 or ras_contract.get("top_k") != 25 or ras_contract.get("win_size") != 10 or ras_contract.get("tau_r") != 0.1 or ras_contract.get("vocab") != 6761:
        raise ValueError("RAS ignore_eos retry bound is not authenticated")
    if observations.get("nonstream_finalize") != {"streaming": False, "finalize": True, "path": "CosyVoice3Model.tts -> token2wav -> flow.inference -> CausalConditionalCFM.forward"}:
        raise ValueError("official non-stream finalize=true path evidence missing")
    sampled = observations.get("sampled_tokens")
    yielded = observations.get("yielded_tokens")
    stop = observations.get("terminal_stop_token")
    if not isinstance(sampled, list) or not sampled or not isinstance(yielded, list) or yielded != generated or not isinstance(stop, int) or not 6561 <= stop < 6761 or sampled != yielded + [stop]:
        raise ValueError("sampled/yielded/terminal-stop trace mismatch")
    sampling_calls = observations.get("sampling_calls")
    if not isinstance(sampling_calls, list) or len(sampling_calls) != len(sampled) or sampling_calls[-1].get("ignore_eos") is not False or any(not isinstance(row, dict) or row.get("index") != index or row.get("selected") != sampled[index] or not isinstance(row.get("ras_calls"), int) or not 1 <= row["ras_calls"] <= 101 or not isinstance(row.get("ras_call_indices"), list) or len(row["ras_call_indices"]) != row["ras_calls"] or row["ras_call_indices"] != list(range(row["ras_call_indices"][0], row["ras_call_indices"][0] + row["ras_calls"])) or not isinstance(row.get("attempts"), list) or len(row["attempts"]) != row["ras_calls"] or any(not isinstance(attempt, dict) or attempt.get("ras_index") != row["ras_call_indices"][attempt_index] or not isinstance(attempt.get("internal_attempts"), list) or not attempt["internal_attempts"] or attempt.get("selected") != attempt["internal_attempts"][-1].get("selected") or attempt.get("accepted") is not (attempt_index == len(row["attempts"]) - 1) for attempt_index, attempt in enumerate(row["attempts"])) or (row.get("ignore_eos") is True and (any(attempt["accepted"] for attempt in row["attempts"][:-1]) or not row["attempts"][-1]["accepted"] or row["selected"] >= 6561)) or (row.get("ignore_eos") is False and (row["ras_calls"] != 1 or not 6561 <= row["selected"] < 6761)) for index, row in enumerate(sampling_calls)):
        raise ValueError("sampling call order/ignore_eos cardinality mismatch")
    ras = artifacts["ras_calls"][0].get("value")
    if not isinstance(ras, list) or not ras:
        raise ValueError("RAS nucleus/fallback cardinality mismatch")
    sampling_by_index = {row["index"]: row for row in sampling_calls}
    if len(artifacts["ras_logits"]) != len(ras):
        raise ValueError("RAS logits cardinality does not equal RAS call count")
    expected_multinomial = 0
    expected_probability_metadata: list[tuple[int, str]] = []
    for ras_index, row in enumerate(ras):
        if not isinstance(row, dict) or row.get("index") != ras_index or row.get("sampling_call_index") not in sampling_by_index or ras_index not in sampling_by_index[row["sampling_call_index"]].get("ras_call_indices", []):
            raise ValueError("RAS row to outer sampling call mapping is missing")
        if not isinstance(row, dict) or row.get("nucleus_multinomial_count") != 1 or row.get("repetition_fallback_multinomial_count") not in (0, 1):
            raise ValueError("RAS multinomial cardinality mismatch")
        if row.get("vocab") != 6761 or row.get("sampling") != 25 or row.get("top_p") != 0.8 or row.get("top_k") != 25 or row.get("win_size") != 10 or row.get("tau_r") != 0.1:
            raise ValueError("RAS controls/vocabulary differ from the authenticated source contract")
        expected_multinomial += row["nucleus_multinomial_count"] + row["repetition_fallback_multinomial_count"]
        attempts = row.get("attempts")
        fallback = row.get("fallback_triggered")
        expected_fallback = row.get("repetition_count") >= row.get("fallback_threshold")
        if not isinstance(fallback, bool) or fallback != expected_fallback or not isinstance(attempts, list) or len(attempts) != (2 if fallback else 1):
            raise ValueError("RAS retry/fallback boundary is not source-exact")
        if attempts[0].get("phase") != "nucleus" or attempts[0].get("accepted") != (not fallback) or attempts[-1].get("accepted") is not True or row.get("selected") != attempts[-1].get("selected"):
            raise ValueError("RAS attempt acceptance/order mismatch")
        if fallback and attempts[1].get("phase") != "repetition_fallback":
            raise ValueError("RAS fallback attempt is missing")
        expected_probability_metadata.append((ras_index, "nucleus"))
        if fallback:
            expected_probability_metadata.append((ras_index, "repetition_fallback"))
    if len(artifacts["ras_multinomial_probability"]) != expected_multinomial:
        raise ValueError("RAS multinomial probability cardinality does not match nucleus/fallback attempts")
    for call_index, record in enumerate(artifacts["ras_logits"]):
        if record.get("metadata", {}).get("ras_call_index") != call_index:
            raise ValueError("RAS logits artifact call index is not ordered")
        if math.prod(record.get("shape", [])) != 6761:
            raise ValueError("RAS logits vocabulary shape is not 6761")
    actual_probability_metadata: list[tuple[Any, Any]] = []
    for record in artifacts["ras_multinomial_probability"]:
        metadata = record.get("metadata", {})
        if metadata.get("ras_call_index") not in range(len(ras)) or metadata.get("phase") not in {"nucleus", "repetition_fallback"}:
            raise ValueError("RAS multinomial artifact call mapping is missing")
        actual_probability_metadata.append((metadata.get("ras_call_index"), metadata.get("phase")))
    if actual_probability_metadata != expected_probability_metadata:
        raise ValueError("RAS multinomial phase/call order does not match each source RAS call")
    cfm = artifacts["cfm_solver_trace"][0].get("value")
    if not isinstance(cfm, dict) or cfm.get("n_timesteps") != 10 or cfm.get("schedule") != "cosine" or cfm.get("cfg_rows") != 2 or cfm.get("cfg_rate") != CFM_CFG_RATE or cfm.get("cfg_row_semantics") != "row0=conditional; row1=zero/unconditional for mu/spks/cond; mask rows both equal input mask" or cfm.get("euler_relation") != "x_next=x+dt*((1+cfg_rate)*row0_conditional-cfg_rate*row1_unconditional)" or not isinstance(cfm.get("estimator_calls"), list) or len(cfm["estimator_calls"]) != 10 or not all(isinstance(cfm.get(key), str) and cfm[key] for key in ("mask_path", "spks_path", "cond_path")):
        raise ValueError("CFM solver trace cardinality/schedule mismatch")
    span = cfm.get("t_span")
    dt = cfm.get("dt")
    expected_span = [1.0 - math.cos(index * math.pi / 20.0) for index in range(11)]
    if not isinstance(span, list) or len(span) != 11 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in span) or any(abs(span[index] - expected_span[index]) > 1e-6 for index in range(11)) or not isinstance(dt, list) or len(dt) != 10 or any(abs(dt[i] - (span[i + 1] - span[i])) > 1e-6 for i in range(10)):
        raise ValueError("CFM t-grid is not strict and finite")
    estimator_calls = cfm["estimator_calls"]
    if any(row.get("index") != index or row.get("x_shape") != row.get("output_shape") or row.get("x_shape", [0])[0] != 2 or row.get("x_shape", [0, 0])[1] != 80 or row.get("mu_shape") != row.get("x_shape") or row.get("cfg_rows") != 2 or row.get("t_shape") != [2] or not isinstance(row.get("t"), list) or len(row["t"]) != 2 or any(abs(row["t"][axis] - span[index]) > 1e-6 for axis in (0, 1)) or not re.fullmatch(r"[0-9a-f]{64}", row.get("x_sha256", "")) or not all(isinstance(row.get(key), str) and row[key] for key in ("x_path", "mu_path", "t_path", "output_path", "mask_path", "spks_path", "cond_path")) for index, row in enumerate(estimator_calls)):
        raise ValueError("CFM estimator x/output shape trace mismatch")
    if any(len(artifacts[role]) != 10 for role in ("cfm_estimator_x", "cfm_estimator_mu", "cfm_estimator_t", "cfm_estimator_output", "cfm_estimator_mask", "cfm_estimator_spks", "cfm_estimator_cond")):
        raise ValueError("CFM estimator evidence cardinality mismatch")
    path_records = {row["path"]: row for rows in artifacts.values() if isinstance(rows, list) for row in rows if isinstance(row, dict) and "path" in row}
    decoder_mu_values = tensor_values(artifacts["decoder_mu"][0], root)
    decoder_mask_values = tensor_values(artifacts["decoder_mask"][0], root)
    decoder_spks_values = tensor_values(artifacts["decoder_spks"][0], root)
    decoder_cond_values = tensor_values(artifacts["decoder_cond"][0], root)
    base_paths = {key: artifacts[f"decoder_{key}"][0]["path"] for key in ("mask", "spks", "cond")}
    if any(cfm.get(f"{key}_path") != path for key, path in base_paths.items()):
        raise ValueError("CFM solver trace does not bind base mask/spks/cond artifacts")
    if any(artifacts[f"decoder_{key}"][0]["shape"][0] != 1 for key in ("mask", "spks", "cond")):
        raise ValueError("CFM base mask/spks/cond must be single conditional rows")
    call_values = []
    for index, row in enumerate(estimator_calls):
        records = {key: path_records.get(row[f"{key}_path"]) for key in ("x", "mu", "t", "output", "mask", "spks", "cond")}
        if any(record is None or record.get("metadata", {}).get("call_index") != index for record in records.values()):
            raise ValueError("CFM estimator artifact call mapping is incomplete")
        values = {key: tensor_values(record, root) for key, record in records.items()}
        if row.get("x_sha256") != records["x"]["sha256"]:
            raise ValueError("CFM estimator x_sha256 does not bind the x artifact")
        if records["x"]["shape"] != row["x_shape"] or records["mu"]["shape"] != row["mu_shape"] or records["t"]["shape"] != row["t_shape"] or records["output"]["shape"] != row["output_shape"]:
            raise ValueError("CFM estimator artifact shape metadata differs from call trace")
        for role in ("x", "mu", "mask", "spks", "cond"):
            base_shape = artifacts["decoder_noise"][0]["shape"] if role == "x" else artifacts[f"decoder_{role}"][0]["shape"]
            if records[role]["shape"] != [2, *base_shape[1:]]:
                raise ValueError(f"CFM {role} CFG shape is not [2] + base shape tail")
        per_row = len(values["x"]) // 2
        if per_row <= 0 or len(values["x"]) != len(values["mu"]) or len(values["x"]) != len(values["output"]) or len(values["t"]) != 2:
            raise ValueError("CFM estimator CFG row cardinality is invalid")
        if values["x"][:per_row] != values["x"][per_row:]:
            raise ValueError("CFM CFG x rows are not identical")
        validate_cfg_rows(values["mu"], decoder_mu_values, "mu", duplicate=False)
        validate_cfg_rows(values["mask"], decoder_mask_values, "mask", duplicate=True)
        validate_cfg_rows(values["spks"], decoder_spks_values, "spks", duplicate=False)
        validate_cfg_rows(values["cond"], decoder_cond_values, "cond", duplicate=False)
        call_values.append(values)
    for index, values in enumerate(call_values):
        dt_value = f32(span[index + 1] - span[index])
        per_row = len(values["x"]) // 2
        expected_next = []
        for position in range(per_row):
            guided = f32(f32((1.0 + CFM_CFG_RATE) * values["output"][position]) - f32(CFM_CFG_RATE * values["output"][per_row + position]))
            expected_next.append(f32(values["x"][position] + f32(dt_value * guided)))
        if index + 1 < len(call_values):
            next_x = call_values[index + 1]["x"]
            if not all(f32_close(actual, expected) for actual, expected in zip(next_x[:per_row], expected_next)) or next_x[:per_row] != next_x[per_row:]:
                raise ValueError("CFM Euler/CFG state progression differs from the source trace")
        else:
            terminal = tensor_values(artifacts["cfm_terminal_mel"][0], root)
            if len(terminal) != per_row or not all(f32_close(actual, expected) for actual, expected in zip(terminal, expected_next)):
                raise ValueError("CFM terminal state differs from the source Euler/CFG trace")
    if artifacts["decoder_mu"][0]["shape"] != artifacts["decoder_noise"][0]["shape"]:
        raise ValueError("decoder mu/noise shape must come from the same official call")
    seen = set()
    for rows in artifacts.values():
        for row in rows:
            if "path" in row:
                path = row["path"]
                if path in seen:
                    raise ValueError("duplicate artifact path")
                seen.add(path)
    allowed = seen | {manifest_path.name}
    extras = {p.name for p in root.iterdir() if p.is_file()} - allowed
    if extras:
        raise ValueError(f"stale/orphan evidence files: {sorted(extras)}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        try:
            pairs([("x", 1), ("x", 2)])
            raise AssertionError("duplicate key accepted")
        except ValueError:
            pass
        for unsafe in ("../x", "/x", "a\\b", "a\0b"):
            try:
                safe_relative(unsafe)
                raise AssertionError(f"unsafe path accepted: {unsafe!r}")
            except ValueError:
                pass
        assert SOURCE_ROLE_BLOBS["cosyvoice/flow/flow.py"] == "c25518621bf98e95d5ed75b83c5a2a610d0822be"
        assert MATCHA_ROLE_BLOBS["matcha/models/components/decoder.py"] == "1137cd7008e9d07b4f306926a82e44c2b2cddbdf"
        assert SOURCE_TRANSFORMERS_REQUIREMENT == "transformers==4.51.3"
        assert SOURCE_HUGGINGFACE_HUB_REQUIREMENT == "huggingface-hub==0.24.7"
        assert TRANSFORMERS_SECURITY_ADVISORY == "GHSA-xrqw-3rrv-vx5w"
        assert TRANSFORMERS_SECURITY_PATCHED_MINIMUM == "5.10.0"
        assert ISOLATED_TRANSFORMERS_PIN == "5.10.4"
        assert ISOLATED_HUGGINGFACE_HUB_PIN == "1.5.0"
        assert PROJECT_VERSIONS["transformers"] == ISOLATED_TRANSFORMERS_PIN
        assert PROJECT_VERSIONS["huggingface-hub"] == ISOLATED_HUGGINGFACE_HUB_PIN
        grid = [1.0 - math.cos(index * math.pi / 20.0) for index in range(11)]
        assert len(grid) == 11 and grid[0] == 0.0 and abs(grid[-1] - 1.0) < 1e-12
        bad_grid = list(grid); bad_grid[4] += 0.01
        assert any(abs(actual - wanted) > 1e-6 for actual, wanted in zip(bad_grid, grid))
        ras_row = {"repetition_count": 0, "fallback_threshold": 1.0, "fallback_triggered": False, "selected": 12, "attempts": [{"phase": "nucleus", "selected": 12, "accepted": True}]}
        assert ras_row["attempts"][-1]["accepted"] is True and ras_row["selected"] == ras_row["attempts"][-1]["selected"]
        bad_ras = dict(ras_row, fallback_triggered=True)
        assert bad_ras["fallback_triggered"] != (bad_ras["repetition_count"] >= bad_ras["fallback_threshold"])
        fractional_threshold = dict(ras_row, repetition_count=1, fallback_threshold=0.5, fallback_triggered=True)
        assert fractional_threshold["fallback_triggered"] == (fractional_threshold["repetition_count"] >= fractional_threshold["fallback_threshold"])
        retry = {"ras_calls": 101, "ras_call_indices": list(range(101))}
        assert len(retry["ras_call_indices"]) <= 101
        assert len(retry["ras_call_indices"]) + 1 > 101
        uncond, cond, x_value, dt_value = 0.25, 0.75, 1.0, 0.1
        guided = f32(f32((1.0 + CFM_CFG_RATE) * cond) - f32(CFM_CFG_RATE * uncond))
        expected = f32(x_value + f32(dt_value * guided))
        assert f32_close(expected, 1.11)
        base_bits = f32_bits(1.0)
        four_ulp = struct.unpack("<f", struct.pack("<I", base_bits + 4))[0]
        five_ulp = struct.unpack("<f", struct.pack("<I", base_bits + 5))[0]
        assert f32_close(1.0, four_ulp)
        assert not f32_close(1.0, five_ulp)
        conditional, unconditional = 1.0, 0.0
        normal = f32(f32((1.0 + CFM_CFG_RATE) * conditional) - f32(CFM_CFG_RATE * unconditional))
        swapped = f32(f32((1.0 + CFM_CFG_RATE) * unconditional) - f32(CFM_CFG_RATE * conditional))
        assert normal != swapped and normal > 0.0 and swapped < 0.0
        validate_cfg_rows([1.0, 2.0, 0.0, 0.0], [1.0, 2.0], "spks", duplicate=False)
        validate_cfg_rows([1.0, 2.0, 1.0, 2.0], [1.0, 2.0], "mask", duplicate=True)
        for bad_rows, base, role, duplicate in (
            ([0.0, 0.0, 1.0, 2.0], [1.0, 2.0], "spks", False),
            ([1.0, 2.0, 1.0, 2.0], [1.0, 2.0], "spks", False),
            ([0.0, 0.0, 1.0, 2.0], [1.0, 2.0], "mask", True),
            ([1.0, 2.0, 0.0, 0.0], [1.0, 2.0], "mask", True),
            ([1.0, 2.0, 1.0, 2.0], [1.0, 2.0], "cond", False),
        ):
            try:
                validate_cfg_rows(bad_rows, base, role, duplicate)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted invalid {role} CFG rows")
        print("cosyvoice3_validate_reference self-test: OK")
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit("usage: cosyvoice3_validate_reference.py MANIFEST")
    try:
        validate(Path(sys.argv[1]))
    except Exception as exc:
        print(f"cosyvoice3 reference validation blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print("cosyvoice3 reference validation: PASS")
