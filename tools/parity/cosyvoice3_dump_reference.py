#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice3_reference --python 3.12 python
"""Execute the pinned upstream CosyVoice3 adapter and emit parity evidence.

This is an adapter around upstream classes, never a Python mirror.  It is
intentionally VAST-only: the 2 GB+ composite is not loaded on the maintainer
machine.  The runtime remains blocked even when this script succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version as installed_version
import json
import math
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

SOURCE_REVISION = "0d990d60740bf174904a5185cce910b847bd3684"
MODEL_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
SOURCE_ORIGIN = "https://github.com/FunAudioLLM/CosyVoice"
MATCHA_REVISION = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
MATCHA_ORIGIN = "https://github.com/shivammehta25/Matcha-TTS"
SOURCE_TRANSFORMERS_REQUIREMENT = "transformers==4.51.3"
SOURCE_HUGGINGFACE_HUB_REQUIREMENT = "huggingface-hub==0.24.7"
TRANSFORMERS_SECURITY_ADVISORY = "GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM = "5.10.0"
ISOLATED_TRANSFORMERS_PIN = "5.10.4"
ISOLATED_HUGGINGFACE_HUB_PIN = "1.5.0"
TRANSFORMERS_COMPATIBILITY_STATUS = "BLOCKED_UNVERIFIED_API_SMOKE"
TRANSFORMERS_LICENSE_METADATA = "METADATA_DECLARED_APACHE-2.0_PRIMARY_BYTES_UNREVIEWED"
TRANSFORMERS_METADATA_EVIDENCE = "https://pypi.org/pypi/transformers/5.10.4/json"
REFERENCE_PROJECT = Path(__file__).parent / "cosyvoice3_reference"
PROJECT_VERSIONS = {
    "conformer": "0.3.2", "diffusers": "0.29.0", "hyperpyyaml": "1.2.2",
    "einops": "0.8.0", "inflect": "7.3.1", "huggingface-hub": "1.5.0",
    "librosa": "0.10.2", "modelscope": "1.20.0", "numpy": "1.26.4",
    "omegaconf": "2.3.0", "onnx": "1.16.0", "onnxruntime": "1.18.0",
    "openai-whisper": "20231117", "pyworld": "0.3.4", "pyyaml": "6.0.2",
    "soundfile": "0.12.1", "torch": "2.3.1", "torchaudio": "2.3.1",
    "transformers": ISOLATED_TRANSFORMERS_PIN, "tqdm": "4.66.5", "wetext": "0.0.4",
}
MODEL_REPOSITORY = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TARGET_TEXT = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
FORMAT = "vokra-cosyvoice3-official-reference-v1"
# Transport sanity bound only; this is not a model-value tolerance.
MAX_TENSOR_ABS = 1_000_000.0
REQUIRED_ARTIFACTS = {
    "tokenizer_ids", "prompt_speech_tokens", "campplus_embedding",
    "qwen_prompt_embeddings", "ras_calls", "ras_logits", "ras_multinomial_probability", "generated_speech_tokens",
    "flow_rand_noise_full", "flow_rand_noise_slice", "flow_encoder_output",
    "cfm_terminal_mel", "prompt_mel", "generated_mel", "hift_input_mel",
    "hift_output_pcm", "official_output_pcm", "decoder_mu", "decoder_noise",
    "cfm_estimator_x", "cfm_estimator_mu", "cfm_estimator_t",
    "cfm_estimator_output", "cfm_estimator_mask", "cfm_estimator_spks",
    "cfm_estimator_cond", "decoder_mask", "decoder_spks", "decoder_cond",
    "cfm_solver_trace",
}
JSON_ARTIFACTS = {"ras_calls", "generated_speech_tokens", "cfm_solver_trace"}
# Exact Git blobs from the pinned, clean upstream checkouts.  A revision alone
# is insufficient: these rows bind the call sites that the adapter instruments.
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_project_identity() -> dict[str, Any]:
    """Require and authenticate the frozen reference environment before loading weights."""
    pyproject = REFERENCE_PROJECT / "pyproject.toml"
    lock = REFERENCE_PROJECT / "uv.lock"
    if not pyproject.is_file() or not lock.is_file():
        raise RuntimeError("dedicated CosyVoice3 pyproject.toml and uv.lock are required before acquisition")
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
        "transformers_license": TRANSFORMERS_LICENSE_METADATA,
        "transformers_metadata_evidence": TRANSFORMERS_METADATA_EVIDENCE,
    }
    if any(route.get(key) != value for key, value in expected_route.items()):
        raise RuntimeError("CosyVoice3 Transformers security metadata drifted")
    raw_dependencies = config.get("project", {}).get("dependencies")
    if not isinstance(raw_dependencies, list):
        raise RuntimeError("CosyVoice3 dependency inventory is missing")
    dependencies = {}
    for raw in raw_dependencies:
        if not isinstance(raw, str) or raw.count("==") != 1:
            raise RuntimeError("CosyVoice3 dependencies must be exact pins")
        name, pinned = raw.split("==")
        dependencies[name.lower()] = pinned
    if dependencies != PROJECT_VERSIONS:
        raise RuntimeError("CosyVoice3 dependency inventory drifted")
    lock_text = lock.read_text(encoding="utf-8")
    for name, expected in PROJECT_VERSIONS.items():
        match = re.search(rf'(?m)^name = "{re.escape(name)}"\nversion = "([^"]+)"', lock_text)
        if not match or match.group(1) != expected:
            raise RuntimeError(f"CosyVoice3 lock package/version mismatch: {name}")
    actual_versions = {}
    for name, expected in PROJECT_VERSIONS.items():
        try:
            actual = installed_version(name)
        except PackageNotFoundError as error:
            raise RuntimeError(f"CosyVoice3 installed dependency missing: {name}") from error
        if actual != expected:
            raise RuntimeError(f"CosyVoice3 installed dependency drift: {name}={actual}")
        actual_versions[name] = actual
    return {
        "python": ">=3.12,<3.13",
        "pyproject_sha256": sha256_file(pyproject),
        "uv_lock_sha256": sha256_file(lock),
        "dependencies": PROJECT_VERSIONS,
        "actual_versions": actual_versions,
        "source_transformers_requirement": SOURCE_TRANSFORMERS_REQUIREMENT,
        "source_huggingface_hub_requirement": SOURCE_HUGGINGFACE_HUB_REQUIREMENT,
        "transformers_security_advisory": TRANSFORMERS_SECURITY_ADVISORY,
        "transformers_security_patched_minimum": TRANSFORMERS_SECURITY_PATCHED_MINIMUM,
        "isolated_transformers_pin": ISOLATED_TRANSFORMERS_PIN,
        "isolated_huggingface_hub_pin": ISOLATED_HUGGINGFACE_HUB_PIN,
        "transformers_compatibility_status": TRANSFORMERS_COMPATIBILITY_STATUS,
    }


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def git_blob(root: Path, relative: str) -> str:
    """Return the complete blob identity from the checked-out revision."""
    return git(root, "rev-parse", f"HEAD:{relative}")


def authenticate_source(root: Path) -> dict[str, Any]:
    if git(root, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("source revision mismatch")
    origin = git(root, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    expected = SOURCE_ORIGIN.removesuffix("/").removesuffix(".git")
    if origin != expected:
        raise RuntimeError(f"source origin mismatch: {origin!r}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source checkout is dirty")
    gitlink = git(root, "ls-files", "-s", "third_party/Matcha-TTS").split()
    if len(gitlink) < 3 or gitlink[0] != "160000" or gitlink[1] != MATCHA_REVISION:
        raise RuntimeError("source Matcha submodule revision mismatch")
    configured_matcha = git(root, "config", "-f", ".gitmodules", "--get", "submodule.third_party/Matcha-TTS.url")
    if configured_matcha.removesuffix("/").removesuffix(".git") != MATCHA_ORIGIN:
        raise RuntimeError("source Matcha submodule origin mismatch")
    required = {
        "cosyvoice/llm/llm.py": ("class CosyVoice3LM", "self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200"),
        "cosyvoice/flow/flow.py": ("class CausalMaskedDiffWithDiT", "feat = feat[:, :, mel_len1:]"),
        "cosyvoice/flow/flow_matching.py": ("class CausalConditionalCFM", "self.rand_noise = torch.randn([1, 80, 50 * 300])"),
        "cosyvoice/hifigan/generator.py": ("class CausalHiFTGenerator", "conv_pre_look_right"),
        "cosyvoice/cli/model.py": ("class CosyVoice3Model", "token_offset * self.flow.token_mel_ratio"),
        "examples/libritts/cosyvoice3/conf/cosyvoice3.yaml": ("sample_rate: 24000", "token_mel_ratio: 2"),
        "example.py": ("八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。", "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"),
        "asset/zero_shot_prompt.wav": (),
    }
    files = {}
    for relative, needles in required.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing source role: {relative}")
        text = path.read_text(encoding="utf-8") if needles else ""
        if any(needle not in text for needle in needles):
            raise RuntimeError(f"source role contract mismatch: {relative}")
        blob = git_blob(root, relative)
        if blob != SOURCE_ROLE_BLOBS[relative]:
            raise RuntimeError(f"source role Git blob mismatch: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "git_blob_sha1": blob}
    return {"repository": "FunAudioLLM/CosyVoice", "revision": SOURCE_REVISION, "origin": origin, "clean": True, "files": files}


def authenticate_matcha(root: Path) -> dict[str, Any]:
    if git(root, "rev-parse", "HEAD") != MATCHA_REVISION:
        raise RuntimeError("Matcha revision mismatch")
    origin = git(root, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    if origin != MATCHA_ORIGIN:
        raise RuntimeError(f"Matcha origin mismatch: {origin!r}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Matcha checkout is dirty")
    files = {}
    for relative in MATCHA_ROLE_BLOBS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing Matcha source role: {relative}")
        blob = git_blob(root, relative)
        if blob != MATCHA_ROLE_BLOBS[relative]:
            raise RuntimeError(f"Matcha role Git blob mismatch: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "git_blob_sha1": blob}
    return {"repository": "shivammehta25/Matcha-TTS", "revision": MATCHA_REVISION, "origin": origin, "clean": True, "files": files}


def authenticate_model(model_dir: Path) -> dict[str, Any]:
    inspector_path = Path(__file__).with_name("cosyvoice3_inspect.py")
    spec = importlib.util.spec_from_file_location("cosyvoice3_inspector", inspector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load model identity table")
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    local = inspector.local_files(model_dir)
    if set(local) != set(inspector.TREE):
        raise RuntimeError(f"model tree file set mismatch: missing={sorted(set(inspector.TREE) - set(local))} extra={sorted(set(local) - set(inspector.TREE))}")
    files: dict[str, Any] = {}
    for relative, (size, blob, lfs) in inspector.TREE.items():
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"model file identity mismatch: {relative}")
        actual = sha256_file(path) if lfs else inspector.git_blob(path)
        if actual != (lfs or blob):
            raise RuntimeError(f"model digest mismatch: {relative}")
        files[relative] = {"bytes": size, "sha256": sha256_file(path), "git_blob_sha1": blob, "lfs_sha256": lfs}
    return {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "resolved_revision": MODEL_REVISION, "files": files}


def load_input(path: Path, source: Path) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    required = {"target_text", "prompt_text", "prompt_wav", "prompt_sha256", "seed"}
    if not isinstance(packet, dict) or set(packet) != required:
        raise ValueError(f"input keys must be exactly {sorted(required)}")
    for key in ("target_text", "prompt_text"):
        value = packet[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise ValueError(f"{key} must be non-empty text <=4096 chars")
    if packet["target_text"] != TARGET_TEXT or packet["prompt_text"] != PROMPT_TEXT:
        raise ValueError("input text packet does not match the fixed official parity prompt")
    if not isinstance(packet["prompt_wav"], str) or "\\" in packet["prompt_wav"]:
        raise ValueError("prompt_wav must be a portable relative path")
    wav = Path(packet["prompt_wav"])
    if wav.is_absolute() or ".." in wav.parts or wav.as_posix() != "asset/zero_shot_prompt.wav":
        raise ValueError("prompt_wav must be the fixed source asset/zero_shot_prompt.wav")
    wav = (source / wav).resolve()
    source_root = source.resolve()
    if source_root not in wav.parents or not wav.is_file() or wav.is_symlink() or wav.stat().st_size == 0:
        raise ValueError("prompt_wav must be a regular file contained by the fixed source checkout")
    if not isinstance(packet["prompt_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", packet["prompt_sha256"]):
        raise ValueError("prompt_sha256 must be lowercase SHA-256")
    if sha256_file(wav) != packet["prompt_sha256"]:
        raise ValueError("prompt_wav SHA-256 mismatch")
    seed = packet["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    packet["prompt_wav"] = str(wav)
    return packet


class Evidence:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, list[dict[str, Any]]] = {}
        self.observations: dict[str, Any] = {}

    def tensor(self, role: str, value: Any, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if role not in REQUIRED_ARTIFACTS or role in JSON_ARTIFACTS or not re.fullmatch(r"[a-z][a-z0-9_]*", role):
            raise RuntimeError(f"unsupported evidence role: {role}")
        if not hasattr(value, "detach"):
            raise RuntimeError(f"official tap {role} is not a tensor")
        import torch
        if value.numel() == 0:
            raise RuntimeError(f"official tap {role} is empty or non-finite")
        finite = torch.isfinite(value)
        max_abs = float(value.detach().abs().max())
        if not bool(finite.all()) or not max_abs <= MAX_TENSOR_ABS:
            raise RuntimeError(f"official tap {role} is non-finite or exceeds transport bound")
        if role.endswith("pcm") and max_abs > 1.1:
            raise RuntimeError(f"official tap {role} exceeds normalized PCM bounds")
        if value.numel() > 100_000_000:
            raise RuntimeError(f"official tap {role} exceeds bounded evidence size")
        raw = value.detach().cpu().contiguous().float().numpy().tobytes(order="C")
        index = len(self.artifacts.get(role, []))
        file = self.output / f"{role}.{index}.f32.bin"
        file.write_bytes(raw)
        shape = list(value.shape)
        if not shape or any(isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0 for axis in shape):
            raise RuntimeError(f"official tap {role} has invalid shape")
        record = {"path": file.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "shape": shape, "dtype": str(value.dtype), "source": source, "storage_dtype": "float32"}
        if metadata:
            record["metadata"] = metadata
        self.artifacts.setdefault(role, []).append(record)
        return record

    def json_value(self, role: str, value: Any, source: str) -> None:
        if role not in REQUIRED_ARTIFACTS or role not in JSON_ARTIFACTS or not re.fullmatch(r"[a-z][a-z0-9_]*", role):
            raise RuntimeError(f"unsupported evidence role: {role}")
        self.artifacts.setdefault(role, []).append({"value": value, "source": source})

    def validate(self) -> None:
        missing = sorted(REQUIRED_ARTIFACTS - set(self.artifacts))
        empty = sorted(role for role in REQUIRED_ARTIFACTS if not self.artifacts.get(role))
        if missing or empty:
            raise RuntimeError(f"required evidence missing={missing} empty={empty}")


def run_reference(source: Path, matcha_source: Path, model_dir: Path, packet: dict[str, Any], evidence: Evidence) -> None:
    # Import and execute only upstream classes.  Seeding occurs before model
    import torch
    sys.path.insert(0, str(matcha_source))
    sys.path.insert(0, str(source))
    from cosyvoice.utils import common  # type: ignore
    from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore
    from cosyvoice.utils.common import set_all_random_seed  # type: ignore

    original_ras = common.ras_sampling
    original_nucleus = common.nucleus_sampling
    original_random = common.random_sampling
    original_multinomial = torch.Tensor.multinomial
    ras_active = False
    ras_phase: str | None = None
    ras_calls: list[dict[str, Any]] = []
    current_ras: dict[str, Any] | None = None
    sampling_calls: list[dict[str, Any]] = []
    generated_tokens: list[int] = []

    def multinomial(self, *args, **kwargs):
        if ras_active:
            source = f"cosyvoice.utils.common.{ras_phase or 'unknown'}_sampling -> Tensor.multinomial"
            if current_ras is None:
                raise RuntimeError("RAS multinomial occurred outside an active RAS call")
            evidence.tensor(
                "ras_multinomial_probability",
                self,
                source,
                {"ras_call_index": current_ras["index"], "phase": ras_phase},
            )
            if ras_calls:
                key = "nucleus_multinomial_count" if ras_phase == "nucleus" else "repetition_fallback_multinomial_count"
                ras_calls[-1][key] = ras_calls[-1].get(key, 0) + 1
        return original_multinomial(self, *args, **kwargs)

    def nucleus(*args, **kwargs):
        nonlocal ras_phase
        ras_phase = "nucleus"
        try:
            selected = original_nucleus(*args, **kwargs)
            if current_ras is not None:
                current_ras["nucleus_selected"] = int(selected)
            return selected
        finally:
            ras_phase = None

    def repetition(*args, **kwargs):
        nonlocal ras_phase
        ras_phase = "repetition_fallback"
        try:
            selected = original_random(*args, **kwargs)
            if current_ras is not None:
                current_ras["fallback_selected"] = int(selected)
            return selected
        finally:
            ras_phase = None

    def ras(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
        nonlocal ras_active, current_ras
        ras_index = len(ras_calls)
        evidence.tensor(
            "ras_logits",
            weighted_scores,
            "cosyvoice.utils.common.ras_sampling.weighted_scores",
            {"ras_call_index": ras_index},
        )
        if int(sampling) != 25 or float(top_p) != 0.8 or int(top_k) != 25 or int(win_size) != 10 or float(tau_r) != 0.1:
            raise RuntimeError("official RAS controls differ from the authenticated CosyVoice3 config")
        if int(weighted_scores.numel()) != 6761:
            raise RuntimeError("official RAS vocabulary must be speech_token_size + 200 (6761)")
        current_ras = {"index": ras_index, "vocab": int(weighted_scores.numel()), "sampling": int(sampling), "top_p": float(top_p), "top_k": int(top_k), "win_size": int(win_size), "tau_r": float(tau_r), "decoded_count": len(decoded_tokens), "decoded_tail": list(decoded_tokens[-win_size:]), "nucleus_multinomial_count": 0, "repetition_fallback_multinomial_count": 0, "accepted": False}
        ras_calls.append(current_ras)
        ras_active = True
        try:
            selected = original_ras(weighted_scores, decoded_tokens, sampling, top_p=top_p, top_k=top_k, win_size=win_size, tau_r=tau_r)
            nucleus_selected = current_ras.get("nucleus_selected")
            if not isinstance(nucleus_selected, int):
                raise RuntimeError("official RAS nucleus selection was not captured")
            repeats = sum(token == nucleus_selected for token in decoded_tokens[-int(win_size):])
            fallback_threshold = float(win_size) * float(tau_r)
            fallback = repeats >= fallback_threshold
            current_ras.update({"repetition_count": repeats, "fallback_threshold": fallback_threshold, "fallback_triggered": fallback, "selected": int(selected), "attempts": [{"phase": "nucleus", "selected": nucleus_selected, "accepted": not fallback}]})
            if fallback:
                fallback_selected = current_ras.get("fallback_selected")
                if not isinstance(fallback_selected, int) or int(selected) != fallback_selected:
                    raise RuntimeError("official RAS fallback selection was not captured")
                current_ras["attempts"].append({"phase": "repetition_fallback", "selected": fallback_selected, "accepted": True})
            return selected
        finally:
            ras_active = False
            current_ras = None

    torch.Tensor.multinomial = multinomial
    common.ras_sampling = ras
    common.nucleus_sampling = nucleus
    common.random_sampling = repetition
    model = None
    original_sampling = original_flow = original_hift = original_frontend = None
    original_llm_inference = original_wrapper = original_pre = original_solve = original_estimator = None
    cfm_estimator_calls: list[dict[str, Any]] = []
    try:
        model = CosyVoice3(model_dir=str(model_dir), load_trt=False, load_vllm=False, fp16=False)
        # The official CFM constructor calls set_all_random_seed(0) itself
        # immediately before creating rand_noise.  Seed AR separately after
        # construction using the same upstream helper.
        set_all_random_seed(packet["seed"])
        frontend = model.frontend

        original_frontend = frontend.frontend_zero_shot
        def frontend_zero(*args, **kwargs):
            inputs = original_frontend(*args, **kwargs)
            evidence.tensor("tokenizer_ids", inputs["text"], "CosyVoiceFrontEnd.frontend_zero_shot.text")
            evidence.tensor("tokenizer_ids", inputs["prompt_text"], "CosyVoiceFrontEnd.frontend_zero_shot.prompt_text")
            evidence.tensor("prompt_mel", inputs["prompt_speech_feat"], "CosyVoiceFrontEnd.frontend_zero_shot.prompt_speech_feat")
            evidence.tensor("prompt_speech_tokens", inputs["llm_prompt_speech_token"], "CosyVoiceFrontEnd.frontend_zero_shot.speech_token")
            evidence.tensor("campplus_embedding", inputs["flow_embedding"], "CosyVoiceFrontEnd.frontend_zero_shot.embedding")
            return inputs
        frontend.frontend_zero_shot = frontend_zero

        original_llm_inference = model.model.llm.inference
        def llm_inference(*args, **kwargs):
            prompt = kwargs.get("prompt_speech_token")
            if prompt is not None:
                evidence.tensor("qwen_prompt_embeddings", model.model.llm.speech_embedding(prompt), "CosyVoice3LM.inference.prompt_speech_embedding")
            for token in original_llm_inference(*args, **kwargs):
                generated_tokens.append(int(token))
                yield token
            if generated_tokens:
                evidence.json_value("generated_speech_tokens", list(generated_tokens), "CosyVoice3LM.inference.yield")
        model.model.llm.inference = llm_inference

        original_wrapper = model.model.llm.inference_wrapper
        def inference_wrapper(*args, **kwargs):
            lm_input = kwargs.get("lm_input", args[0] if args else None)
            evidence.tensor("qwen_prompt_embeddings", lm_input, "Qwen2LM.inference_wrapper.lm_input")
            return (yield from original_wrapper(*args, **kwargs))
        model.model.llm.inference_wrapper = inference_wrapper

        original_sampling = model.model.llm.sampling_ids
        def sampling_ids(*args, **kwargs):
            call_index = len(sampling_calls)
            before = len(ras_calls)
            selected = original_sampling(*args, **kwargs)
            ignore_eos = bool(kwargs.get("ignore_eos", args[3] if len(args) > 3 else True))
            rows = ras_calls[before:]
            if not rows or len(rows) > 101:
                raise RuntimeError("official sampling RAS retry count is outside the source bound 1..101")
            outer_attempts = []
            for row in rows:
                candidate = int(row["selected"])
                accepted = candidate < 6561 if ignore_eos else True
                row["accepted"] = accepted
                outer_attempts.append({"ras_index": row["index"], "selected": candidate, "accepted": accepted, "internal_attempts": row.get("attempts", [])})
            if ignore_eos:
                if any(attempt["accepted"] for attempt in outer_attempts[:-1]) or not outer_attempts[-1]["accepted"]:
                    raise RuntimeError("ignore_eos RAS retries must reject control tokens until the final speech token")
            elif len(rows) != 1 or not outer_attempts[-1]["accepted"]:
                raise RuntimeError("ignore_eos=false must accept exactly one control/speech RAS call")
            if outer_attempts[-1]["selected"] != int(selected):
                raise RuntimeError("official sampling call did not expose its final RAS attempt")
            sampling_calls.append({"index": call_index, "selected": int(selected), "ignore_eos": ignore_eos, "ras_calls": len(rows), "ras_call_indices": [row["index"] for row in rows], "attempts": outer_attempts})
            for row in rows:
                row["sampling_call_index"] = call_index
            return selected
        model.model.llm.sampling_ids = sampling_ids

        pre = model.model.flow.pre_lookahead_layer
        original_pre = pre.forward
        def pre_forward(*args, **kwargs):
            result = original_pre(*args, **kwargs)
            evidence.tensor("flow_encoder_output", result, "PreLookaheadLayer.forward.return")
            return result
        pre.forward = pre_forward

        decoder = model.model.flow.decoder
        original_solve = decoder.solve_euler
        def solve_euler(*args, **kwargs):
            def argument(name: str, index: int):
                return kwargs.get(name, args[index] if len(args) > index else None)
            x, t_span, mu = argument("x", 0), argument("t_span", 1), argument("mu", 2)
            mask, spks, cond = argument("mask", 3), argument("spks", 4), argument("cond", 5)
            if any(value is None for value in (x, t_span, mu, mask, spks, cond)):
                raise RuntimeError("CFM solve_euler call missing official x/t_span/mu/mask/spks/cond")
            evidence.tensor("decoder_noise", x, "CausalConditionalCFM.solve_euler.x")
            evidence.tensor("decoder_mu", mu, "CausalConditionalCFM.solve_euler.mu")
            mask_record = evidence.tensor("decoder_mask", mask, "CausalConditionalCFM.solve_euler.mask")
            spks_record = evidence.tensor("decoder_spks", spks, "CausalConditionalCFM.solve_euler.spks")
            cond_record = evidence.tensor("decoder_cond", cond, "CausalConditionalCFM.solve_euler.cond")
            grid = [float(v) for v in t_span.detach().cpu().flatten()]
            expected = [1.0 - math.cos(index * math.pi / 20.0) for index in range(11)]
            if len(grid) != 11 or any(abs(actual - wanted) > 1e-6 for actual, wanted in zip(grid, expected)) or list(x.shape)[0] != 1 or list(mu.shape)[0] != 1:
                raise RuntimeError("official CFM solver did not use the exact 10-step cosine grid")
            evidence.json_value("cfm_solver_trace", {"t_span": grid, "dt": [grid[i + 1] - grid[i] for i in range(len(grid) - 1)], "n_timesteps": 10, "input_shape": list(x.shape), "mu_shape": list(mu.shape), "mask_shape": list(mask.shape), "spks_shape": list(spks.shape), "cond_shape": list(cond.shape), "mask_path": mask_record["path"], "spks_path": spks_record["path"], "cond_path": cond_record["path"], "schedule": "cosine", "cfg_rows": 2, "cfg_rate": 0.7, "cfg_row_semantics": "row0=conditional; row1=zero/unconditional for mu/spks/cond; mask rows both equal input mask", "euler_relation": "x_next=x+dt*((1+cfg_rate)*row0_conditional-cfg_rate*row1_unconditional)", "estimator_calls": cfm_estimator_calls}, "CausalConditionalCFM.solve_euler.arguments")
            result = original_solve(*args, **kwargs)
            evidence.artifacts["cfm_solver_trace"][-1]["value"]["estimator_calls"] = cfm_estimator_calls
            evidence.tensor("cfm_terminal_mel", result, "CausalConditionalCFM.solve_euler.return")
            return result
        decoder.solve_euler = solve_euler
        original_estimator = decoder.forward_estimator
        def forward_estimator(*args, **kwargs):
            def argument(name: str, index: int):
                return kwargs.get(name, args[index] if len(args) > index else None)
            x, mask, mu, t = argument("x", 0), argument("mask", 1), argument("mu", 2), argument("t", 3)
            spks, cond = argument("spks", 4), argument("cond", 5)
            if any(value is None for value in (x, mask, mu, t, spks, cond)):
                raise RuntimeError("CFM estimator call missing official x/mask/mu/t/spks/cond")
            call_index = len(cfm_estimator_calls)
            x_record = evidence.tensor("cfm_estimator_x", x, "CausalConditionalCFM.forward_estimator.x", {"call_index": call_index})
            mu_record = evidence.tensor("cfm_estimator_mu", mu, "CausalConditionalCFM.forward_estimator.mu", {"call_index": call_index})
            t_record = evidence.tensor("cfm_estimator_t", t, "CausalConditionalCFM.forward_estimator.t", {"call_index": call_index})
            mask_record = evidence.tensor("cfm_estimator_mask", mask, "CausalConditionalCFM.forward_estimator.mask", {"call_index": call_index})
            spks_record = evidence.tensor("cfm_estimator_spks", spks, "CausalConditionalCFM.forward_estimator.spks", {"call_index": call_index})
            cond_record = evidence.tensor("cfm_estimator_cond", cond, "CausalConditionalCFM.forward_estimator.cond", {"call_index": call_index})
            out = original_estimator(*args, **kwargs)
            out_record = evidence.tensor("cfm_estimator_output", out, "CausalConditionalCFM.forward_estimator.return", {"call_index": call_index})
            x_raw = x.detach().cpu().contiguous().float().numpy().tobytes(order="C")
            cfm_estimator_calls.append({"index": call_index, "x_shape": list(x.shape), "mu_shape": list(mu.shape), "t_shape": list(t.shape), "output_shape": list(out.shape), "cfg_rows": int(x.shape[0]), "t": [float(v) for v in t.detach().cpu().flatten()], "x_sha256": hashlib.sha256(x_raw).hexdigest(), "x_path": x_record["path"], "mu_path": mu_record["path"], "t_path": t_record["path"], "output_path": out_record["path"], "mask_path": mask_record["path"], "spks_path": spks_record["path"], "cond_path": cond_record["path"]})
            return out
        decoder.forward_estimator = forward_estimator

        original_flow = model.model.flow.inference
        def flow(*args, **kwargs):
            if kwargs.get("finalize") is not True or kwargs.get("streaming") is not False:
                raise RuntimeError("reference must exercise official non-stream finalize=true flow path")
            if not hasattr(decoder, "rand_noise"):
                raise RuntimeError("CausalConditionalCFM.rand_noise is absent")
            evidence.tensor("flow_rand_noise_full", decoder.rand_noise, "CausalConditionalCFM.rand_noise")
            result = original_flow(*args, **kwargs)
            prompt_feat = kwargs.get("prompt_feat")
            if prompt_feat is None or prompt_feat.ndim != 3 or result[0].ndim != 3 or result[0].shape[1] != 80:
                raise RuntimeError("flow tensors have unexpected axes")
            if generated_tokens and int(result[0].shape[2]) != len(generated_tokens) * 2:
                raise RuntimeError("generated mel frame count does not equal yielded speech tokens * token_mel_ratio")
            if not evidence.artifacts.get("decoder_mu"):
                raise RuntimeError("official decoder mu tap is missing")
            decoder_frames = int(evidence.artifacts["decoder_mu"][0]["shape"][-1])
            evidence.tensor("flow_rand_noise_slice", decoder.rand_noise[:, :, :decoder_frames], "CausalConditionalCFM.rand_noise[:, :, :official_mu.size(2)]")
            evidence.tensor("generated_mel", result[0], "CausalMaskedDiffWithDiT.inference.return.generated_only")
            return result
        model.model.flow.inference = flow

        original_hift = model.model.hift.inference
        def hift(*args, **kwargs):
            mel = kwargs.get("speech_feat", args[0] if args else None)
            if mel is None or mel.ndim != 3 or mel.shape[1] != 80:
                raise RuntimeError("HiFT input mel axes mismatch")
            evidence.tensor("hift_input_mel", mel, "CausalHiFTGenerator.inference.speech_feat")
            result = original_hift(*args, **kwargs)
            pcm = result[0]
            mel_frames = int(mel.shape[-1])
            samples = int(pcm.shape[-1])
            if samples != mel_frames * 480:
                raise RuntimeError(f"HiFT sample length {samples} != mel_frames*480 ({mel_frames * 480})")
            if kwargs.get("finalize") is not True:
                raise RuntimeError("reference must exercise official non-stream finalize=true HiFT path")
            evidence.tensor("hift_output_pcm", pcm, "CausalHiFTGenerator.inference.return", {"sample_rate": 24000, "mel_frames": mel_frames, "samples": samples, "finalize": True})
            return result
        model.model.hift.inference = hift
        # Consume the single non-streaming official result to completion.  A
        # bare ``next`` would leave the outer generator paused before its
        # cleanup and would not prove the source termination path.
        outputs = list(model.inference_zero_shot(packet["target_text"], packet["prompt_text"], packet["prompt_wav"], stream=False))
        if len(outputs) != 1:
            raise RuntimeError(f"official non-stream inference yielded {len(outputs)} results")
        result = outputs[0]
        output = result["tts_speech"]
        generated_frames = int(evidence.artifacts["generated_mel"][0]["shape"][-1])
        if output.ndim != 2 or int(output.shape[-1]) != generated_frames * 480:
            raise RuntimeError("official PCM length does not equal generated mel frames * 480")
        evidence.tensor("official_output_pcm", output, "CosyVoice3.inference_zero_shot.return", {"sample_rate": 24000, "samples": int(output.shape[-1])})
        if not ras_calls or not all("selected" in row for row in ras_calls) or not generated_tokens:
            raise RuntimeError("official RAS sampling evidence is incomplete")
        if any("sampling_call_index" not in row for row in ras_calls) or any(row["ras_calls"] == 0 for row in sampling_calls):
            raise RuntimeError("official RAS-to-sampling call order evidence is incomplete")
        sampled = [row["selected"] for row in sampling_calls]
        yielded = [token for token in sampled if token < 6561]
        stops = [token for token in sampled if 6561 <= token < 6761]
        if yielded != generated_tokens or len(stops) != 1:
            raise RuntimeError("official yielded-token/stop-token trace is inconsistent")
        evidence.json_value("ras_calls", ras_calls, "cosyvoice.utils.common.ras_sampling")
        if any(token >= 6761 for token in generated_tokens):
            raise RuntimeError("official generation produced invalid speech/control tokens")
        evidence.observations.update({
            "sampling_order": "Qwen2LM.forward_one_step -> llm_decoder -> log_softmax -> ras_sampling -> multinomial -> speech_embedding",
            "sampling_call_count": len(sampling_calls),
            "sampling_calls": sampling_calls,
            "sampled_tokens": sampled,
            "yielded_tokens": generated_tokens,
            "terminal_stop_token": stops[0],
            "ras_domain": {"top_p": "(0,1]", "tau_r": "source accepts finite non-negative values; fixed=0.1"},
            "termination": "official CosyVoice3Model.tts generator completion",
            "flow_return_contract": "CausalMaskedDiffWithDiT returns generated-only frames after prompt mel slice",
            "flow_noise_seed": 0,
            "flow_noise_initializer": "CausalConditionalCFM.__init__: set_all_random_seed(0); torch.randn([1,80,50*300])",
            "ras_sampling_contract": {"nucleus_multinomial_per_call": 1, "repetition_fallback_multinomial_per_call": "0 or 1 only when official repetition threshold is reached", "ignore_eos_max_ras_calls": 101, "sampling": 25, "top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1, "vocab": 6761, "source": "cosyvoice.utils.common.ras_sampling"},
            "cfm_solver": {"steps": 10, "schedule": "cosine", "estimator_calls": cfm_estimator_calls, "cfg_rows": 2, "row_semantics": "row0=conditional; row1=zero/unconditional for mu/spks/cond; mask rows both equal input mask", "state_progression": "official solve_euler x at every forward_estimator call"},
            "nonstream_finalize": {"streaming": False, "finalize": True, "path": "CosyVoice3Model.tts -> token2wav -> flow.inference -> CausalConditionalCFM.forward"},
        })
    finally:
        if model is not None:
            if original_sampling is not None: model.model.llm.sampling_ids = original_sampling
            if original_llm_inference is not None: model.model.llm.inference = original_llm_inference
            if original_wrapper is not None: model.model.llm.inference_wrapper = original_wrapper
            if original_flow is not None: model.model.flow.inference = original_flow
            if original_hift is not None: model.model.hift.inference = original_hift
            if original_frontend is not None: model.frontend.frontend_zero_shot = original_frontend
            if original_pre is not None: model.model.flow.pre_lookahead_layer.forward = original_pre
            if original_solve is not None: model.model.flow.decoder.solve_euler = original_solve
            if original_estimator is not None: model.model.flow.decoder.forward_estimator = original_estimator
        common.ras_sampling = original_ras
        common.nucleus_sampling = original_nucleus
        common.random_sampling = original_random
        torch.Tensor.multinomial = original_multinomial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path); parser.add_argument("--matcha-source", type=Path); parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--input", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        assert len(SOURCE_REVISION) == 40 and len(MODEL_REVISION) == 40
        assert SOURCE_TRANSFORMERS_REQUIREMENT == "transformers==4.51.3"
        assert SOURCE_HUGGINGFACE_HUB_REQUIREMENT == "huggingface-hub==0.24.7"
        assert TRANSFORMERS_SECURITY_ADVISORY == "GHSA-xrqw-3rrv-vx5w"
        assert TRANSFORMERS_SECURITY_PATCHED_MINIMUM == "5.10.0"
        assert ISOLATED_TRANSFORMERS_PIN == "5.10.4"
        assert ISOLATED_HUGGINGFACE_HUB_PIN == "1.5.0"
        assert PROJECT_VERSIONS["transformers"] == ISOLATED_TRANSFORMERS_PIN
        assert PROJECT_VERSIONS["huggingface-hub"] == ISOLATED_HUGGINGFACE_HUB_PIN
        assert len(REQUIRED_ARTIFACTS) == 30 and "official_output_pcm" in REQUIRED_ARTIFACTS and "cfm_solver_trace" in REQUIRED_ARTIFACTS
        try:
            json.loads('{"duplicate": 1, "duplicate": 2}', object_pairs_hook=strict_pairs)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate JSON key accepted")
        exact_grid = [1.0 - math.cos(index * math.pi / 20.0) for index in range(11)]
        assert exact_grid[0] == 0.0 and abs(exact_grid[-1] - 1.0) < 1e-12
        assert all(exact_grid[index] < exact_grid[index + 1] for index in range(10))
        bad_grid = list(exact_grid); bad_grid[5] += 0.01
        assert any(abs(actual - wanted) > 1e-6 for actual, wanted in zip(bad_grid, exact_grid))
        assert 1 >= 10 * 0.1
        assert not (0 >= 10 * 0.1)
        print("cosyvoice3_dump_reference self-test: OK")
        return 0
    if any(x is None for x in (args.source, args.matcha_source, args.model_dir, args.input, args.output)):
        parser.error("normal run requires --source --matcha-source --model-dir --input --output")
    assert args.source and args.matcha_source and args.model_dir and args.input and args.output
    try:
        if args.output.exists():
            if args.output.is_symlink() or not args.output.is_dir() or any(args.output.iterdir()):
                raise RuntimeError("output directory must be absent or empty (stale evidence refused)")
        project_identity = reference_project_identity()
        source_identity = authenticate_source(args.source)
        matcha_identity = authenticate_matcha(args.matcha_source)
        model_identity = authenticate_model(args.model_dir)
        packet = load_input(args.input, args.source)
        evidence = Evidence(args.output)
        run_reference(args.source, args.matcha_source, args.model_dir, packet, evidence)
        evidence.validate()
        input_identity = {"target_text": packet["target_text"], "prompt_text": packet["prompt_text"], "prompt_wav": "asset/zero_shot_prompt.wav", "prompt_sha256": packet["prompt_sha256"], "seed": packet["seed"], "source_role": {"git_blob_sha1": source_identity["files"]["asset/zero_shot_prompt.wav"]["git_blob_sha1"], "sha256": source_identity["files"]["asset/zero_shot_prompt.wav"]["sha256"]}}
        manifest = {"format": FORMAT, "status": "AUTHENTICATED_REFERENCE_EVIDENCE", "reference_status": "AUTHENTICATED_REFERENCE_EVIDENCE", "comparison_status": "NOT_RUN_OFFICIAL_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "native_status": "BLOCKED", "cpu_status": "UNSUPPORTED_FULL_TTS_PENDING", "metal_status": "BLOCKED_BY_CPU", "publication": "NO_UPLOAD", "sample_rate": 24_000, "llm_input_size": 896, "llm_output_size": 896, "speech_token_size": 6561, "head_size": 6761, "flow_noise_shape": [1, 80, 15000], "flow_steps": 10, "flow_cfg_rate": 0.7, "project": project_identity, "input": input_identity, "matcha": matcha_identity, "source": source_identity, "model": model_identity, "artifacts": evidence.artifacts, "observations": evidence.observations}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        if args.output.exists() and (args.output.is_symlink() or not args.output.is_dir() or any(args.output.iterdir())):
            print(f"cosyvoice3 reference blocked without touching stale output: {exc}", file=sys.stderr)
            return 2
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "manifest.json").write_text(json.dumps({"format": FORMAT, "status": "REFERENCE_ERROR", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "publication": "NO_UPLOAD", "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(f"cosyvoice3 reference blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
