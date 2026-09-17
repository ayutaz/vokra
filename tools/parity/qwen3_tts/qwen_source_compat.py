"""Audited source-only compatibility patches for the Qwen3-TTS API probes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

SOURCE_REPOSITORY = "QwenLM/Qwen3-TTS"
SOURCE_BASE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
SOURCE_HEAD_REVISION = "00969daa8064e23adc9e5f52cdf20cf247f94159"
SOURCE_PR_URL = "https://github.com/QwenLM/Qwen3-TTS/pull/360"
SOURCE_PR_STATUS = "OPEN_UNMERGED"
TRANSFORMERS_VERSION = "5.10.4"
PATCH_STATUS = "COMPATIBILITY_PATCH_APPLIED"
TRANSFORMERS_API = "Transformers 5.10.4 runtime compatibility (RoPE/configuration/mask/generation/Mimi)"

PATCH_INIT_TARGET = "qwen_tts/__init__.py"
PATCH_HELPER_TARGET = "qwen_tts/_transformers_compat.py"
PATCH_CORE_25HZ_TARGET = "qwen_tts/core/__init__.py"
PATCH_CONFIG_TARGET = "qwen_tts/core/models/configuration_qwen3_tts.py"
PATCH_MODEL_TARGET = "qwen_tts/core/models/modeling_qwen3_tts.py"
PATCH_TARGET = "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
PATCH_25HZ_TARGET = "qwen_tts/inference/qwen3_tts_tokenizer.py"
COMPATIBILITY_PATCH_TARGETS = (
    PATCH_INIT_TARGET, PATCH_HELPER_TARGET, PATCH_CORE_25HZ_TARGET,
    PATCH_CONFIG_TARGET, PATCH_MODEL_TARGET, PATCH_TARGET, PATCH_25HZ_TARGET,
)

PATCH_INIT_ORIGINAL_BYTES = 839
PATCH_INIT_ORIGINAL_SHA256 = "ea52de59d070fde366467a6902d0edcfc1b0575b8c570a0c71020c41d6a593ed"
PATCH_INIT_PATCHED_BYTES = 944
PATCH_INIT_PATCHED_SHA256 = "eb4312049f767f591b24d2d7be06cf2ec19a6759c7382c54fac3bd5d671d32c5"
PATCH_HELPER_PATCHED_BYTES = 4763
PATCH_HELPER_PATCHED_SHA256 = "a24b2124843f5c76abc8c7023133b8be7503c80d883d0e9a987cdea5ef2319a0"
PATCH_CORE_25HZ_ORIGINAL_BYTES = 990
PATCH_CORE_25HZ_ORIGINAL_SHA256 = "1b380d9de843b6d585d938c339d066136567ca7125412674234204af4386679e"
PATCH_CORE_25HZ_PATCHED_BYTES = 814
PATCH_CORE_25HZ_PATCHED_SHA256 = "c3d2f2f28cae7a0ec4d8dd8251470c8871acd2bf143d2fc239fcfbe8f2938497"
PATCH_CONFIG_ORIGINAL_BYTES = 26428
PATCH_CONFIG_ORIGINAL_SHA256 = "f52867f14fde06a416dd14864d503ce6d13d0a08d5f5da30191e1c80c13f5d18"
PATCH_CONFIG_PATCHED_BYTES = 26499
PATCH_CONFIG_PATCHED_SHA256 = "4f50b37285f413c05e5d6e257c9969abf9f31a24cd000473f15e31468dbe8461"
PATCH_MODEL_ORIGINAL_BYTES = 100211
PATCH_MODEL_ORIGINAL_SHA256 = "25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0"
PATCH_MODEL_PATCHED_BYTES = 100994
PATCH_MODEL_PATCHED_SHA256 = "78b23efd51dfb92f7deb7ff91b9dd0b7f960376d45d0e6714f365b4ffc691451"
PATCH_ORIGINAL_BYTES = 40519
PATCH_ORIGINAL_SHA256 = "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628"
PATCHED_BYTES = 40366
PATCHED_SHA256 = "55a7e3428a7ca3cbc9a1a8f2275d47dd765e5ffb0258724d471c6d6586c4f573"
PATCH_25HZ_ORIGINAL_BYTES = 15699
PATCH_25HZ_ORIGINAL_SHA256 = "ac2d855022a1bd21d33ab7b267ec952eef71f81d3f8d969a306139ff1a929515"
PATCH_25HZ_PATCHED_BYTES = 15659
PATCH_25HZ_PATCHED_SHA256 = "c19a5e795e90f79b0943b7baf0903e0467cabdbc81b5f55d00c974a067d4f421"

PATCH_OPERATION = "apply_qwen3_tts_pr_360_runtime_hunks_and_25hz_v1_removals"
PATCH_25HZ_OPERATION = "apply_pr_360_runtime_hunks_and_remove_25hz_v1_registration"
PATCH_CORE_25HZ_OPERATION = "remove_exactly_two_core_25hz_imports"
PATCH_CONFIG_OPERATION = "apply_pr_360_config_runtime_hunks"
PATCH_MODEL_OPERATION = "apply_pr_360_model_and_strict_reload_runtime_hunks"
PATCH_TOKENIZER_OPERATION = "apply_pr_360_tokenizer_runtime_hunks"
PATCH_INIT_OPERATION = "apply_pr_360_init_runtime_hunks"

_HELPER_SOURCE = b'''# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Qwen3-TTS on Transformers 5.x."""

from __future__ import annotations

import os
import torch


_PATCHED = False


def _default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    """Initialize the unscaled RoPE variant removed from the 5.x registry."""
    del seq_len, layer_type
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


def patch_transformers_rope_registry() -> None:
    """Register the default RoPE initializer expected by Qwen3-TTS configs."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_parameters)
    _PATCHED = True


@torch.no_grad()
def restore_rope_buffers(model) -> None:
    """Rebuild non-persistent RoPE buffers after low-memory model loading."""
    for module in model.modules():
        if not all(hasattr(module, name) for name in ("rope_init_fn", "config", "inv_freq")):
            continue
        inv_freq, attention_scaling = module.rope_init_fn(module.config, module.inv_freq.device)
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = module.inv_freq
        module.attention_scaling = attention_scaling


def restore_mimi_full_attention(model) -> None:
    """Preserve the full causal Mimi attention used by Transformers 4.57.3."""
    for module in model.modules():
        if not module.__class__.__module__.startswith("transformers.models.mimi"):
            continue
        config = getattr(module, "config", None)
        if config is None or not hasattr(config, "sliding_window"):
            continue
        full_window = config.max_position_embeddings
        config.sliding_window = full_window
        if hasattr(module, "sliding_window"):
            module.sliding_window = full_window


def _strict_key_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return sorted(str(key) for key in value)
    except TypeError:
        return [repr(value)]


def _strict_regular_file(path):
    if path is None or os.path.islink(path) or not os.path.isfile(path):
        raise RuntimeError(f"strict local safetensors checkpoint is missing or symlinked: {path}")
    return path


@torch.no_grad()
def strict_reload_local_safetensors(
    model,
    model_name_or_path,
    *,
    cache_dir=None,
    revision=None,
    token=None,
    local_files_only=False,
):
    """Reload the complete official checkpoint and reject partial state."""
    if os.path.isdir(model_name_or_path):
        checkpoint = os.path.join(model_name_or_path, "model.safetensors")
    else:
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise RuntimeError("strict safetensors reload requires an immutable repo revision")
        from transformers.utils.hub import cached_file

        checkpoint = cached_file(
            model_name_or_path,
            "model.safetensors",
            cache_dir=cache_dir,
            force_download=False,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
    checkpoint = _strict_regular_file(checkpoint)
    try:
        from safetensors.torch import load_model

        result = load_model(model, checkpoint, strict=True, device="cpu")
    except Exception as error:
        raise RuntimeError(f"strict safetensors reload failed: {error}") from error
    missing = []
    unexpected = []
    if isinstance(result, (tuple, list)) and len(result) == 2:
        missing, unexpected = result
    missing = _strict_key_list(missing)
    unexpected = _strict_key_list(unexpected)
    if missing or unexpected:
        raise RuntimeError(
            f"strict safetensors reload returned missing={missing!r} unexpected={unexpected!r}"
        )
    return {
        "status": "STRICT_RELOAD_PASS",
        "checkpoint": os.path.abspath(checkpoint),
        "return_type": type(result).__name__,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
'''

FORBIDDEN_IMPORT_MODULES = ("onnxruntime", "sox")
FORBIDDEN_IMPORT_PREFIXES = ("qwen_tts.core.tokenizer_25hz",)


class CompatibilityPatchError(RuntimeError):
    """The staged source did not satisfy the fixed patch contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def loaded_forbidden_imports() -> list[str]:
    import sys

    return sorted(
        name for name in sys.modules
        if name in FORBIDDEN_IMPORT_MODULES
        or any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    )


def _reject_generated_bytecode(source: Path) -> None:
    for path in source.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.name == "__pycache__" or path.suffix == ".pyc":
            raise CompatibilityPatchError(f"generated Python bytecode is not allowed in source checkout: {path}")


def _replace_once(original: bytes, old: bytes, new: bytes, label: str) -> bytes:
    count = original.count(old)
    if count != 1:
        raise CompatibilityPatchError(f"{label} expected one exact hunk, found {count}")
    return original.replace(old, new)


def _identity(content: bytes, size: int, digest: str, label: str) -> bytes:
    if len(content) != size or sha256_bytes(content) != digest:
        raise CompatibilityPatchError(f"{label} identity drifted")
    return content


def patch_init_source_bytes(original: bytes) -> bytes:
    _identity(original, PATCH_INIT_ORIGINAL_BYTES, PATCH_INIT_ORIGINAL_SHA256, "Qwen3-TTS init original")
    patched = _replace_once(
        original,
        b'"""\n\nfrom .inference.qwen3_tts_model import',
        b'"""\n\nfrom ._transformers_compat import patch_transformers_rope_registry\n\npatch_transformers_rope_registry()\n\nfrom .inference.qwen3_tts_model import',
        "Qwen3-TTS init PR #360 hunk",
    )
    if not patched.endswith(b"\n"):
        patched += b"\n"
    return _identity(patched, PATCH_INIT_PATCHED_BYTES, PATCH_INIT_PATCHED_SHA256, "Qwen3-TTS init patched")


def patch_core_25hz_source_bytes(original: bytes) -> bytes:
    _identity(original, PATCH_CORE_25HZ_ORIGINAL_BYTES, PATCH_CORE_25HZ_ORIGINAL_SHA256, "core 25Hz original")
    patched = _replace_once(
        original,
        b"from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config\n"
        b"from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model\n",
        b"",
        "core 25Hz V1 import removal",
    )
    return _identity(patched, PATCH_CORE_25HZ_PATCHED_BYTES, PATCH_CORE_25HZ_PATCHED_SHA256, "core 25Hz patched")


def patch_config_source_bytes(original: bytes) -> bytes:
    _identity(original, PATCH_CONFIG_ORIGINAL_BYTES, PATCH_CONFIG_ORIGINAL_SHA256, "Qwen3-TTS config original")
    patched = _replace_once(
        original,
        b"from transformers.configuration_utils import PretrainedConfig, layer_type_validation\n"
        b"from transformers.modeling_rope_utils import rope_config_validation\n",
        b"from transformers.configuration_utils import PretrainedConfig\n",
        "Qwen3-TTS config imports PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b'    model_type = "qwen3_tts_talker_code_predictor"\n    keys_to_ignore_at_inference',
        b'    model_type = "qwen3_tts_talker_code_predictor"\n    pad_token_id = None\n    bos_token_id = None\n    eos_token_id = None\n    keys_to_ignore_at_inference',
        "Qwen3-TTS predictor token defaults PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b"        rope_config_validation(self)\n",
        b"        self.standardize_rope_params()\n        self.validate_rope()\n",
        "Qwen3-TTS rope validation PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b"                else \"full_attention\"\n                for i in range(self.num_hidden_layers)\n            ]\n        layer_type_validation(self.layer_types)\n",
        b"                else \"full_attention\"\n                for i in range(self.num_hidden_layers)\n            ]\n        self.validate_layer_type()\n",
        "Qwen3-TTS layer validation PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b'    model_type = "qwen3_tts_talker"\n    keys_to_ignore_at_inference',
        b'    model_type = "qwen3_tts_talker"\n    pad_token_id = None\n    bos_token_id = None\n    eos_token_id = None\n    keys_to_ignore_at_inference',
        "Qwen3-TTS talker token defaults PR #360 hunk",
    )
    return _identity(patched, PATCH_CONFIG_PATCHED_BYTES, PATCH_CONFIG_PATCHED_SHA256, "Qwen3-TTS config patched")


def patch_model_source_bytes(original: bytes) -> bytes:
    _identity(original, PATCH_MODEL_ORIGINAL_BYTES, PATCH_MODEL_ORIGINAL_SHA256, "Qwen3-TTS model original")
    patched = _replace_once(
        original,
        b"            # Prepare mask arguments\n"
        b"            mask_kwargs = {\n"
        b"                \"config\": self.config,\n"
        b"                \"input_embeds\": inputs_embeds,\n"
        b"                \"attention_mask\": attention_mask,\n"
        b"                \"cache_position\": cache_position,\n"
        b"                \"past_key_values\": past_key_values,\n"
        b"            }\n"
        b"            # Create the masks\n"
        b"            causal_mask_mapping = {\n"
        b'                "full_attention": create_causal_mask(**mask_kwargs),\n'
        b"            }",
        b"            mask_kwargs = {\n"
        b"                \"config\": self.config,\n"
        b"                \"inputs_embeds\": inputs_embeds,\n"
        b"                \"attention_mask\": attention_mask,\n"
        b"                \"past_key_values\": past_key_values,\n"
        b"                \"position_ids\": position_ids,\n"
        b"            }\n"
        b"            causal_mask_mapping = {\n"
        b'                "full_attention": create_causal_mask(**mask_kwargs),\n'
        b"            }",
        "Qwen3-TTS model mask kwargs PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b"        causal_mask = mask_function(\n"
        b"            config=self.config,\n"
        b"            input_embeds=inputs_embeds,\n"
        b"            attention_mask=attention_mask,\n"
        b"            cache_position=cache_position,\n"
        b"            past_key_values=past_key_values,\n"
        b"            position_ids=text_position_ids,\n"
        b"        )",
        b"        causal_mask = mask_function(\n"
        b"            config=self.config,\n"
        b"            inputs_embeds=inputs_embeds,\n"
        b"            attention_mask=attention_mask,\n"
        b"            past_key_values=past_key_values,\n"
        b"            position_ids=text_position_ids,\n"
        b"        )",
        "Qwen3-TTS model causal mask PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b'        ```"""\n        # Prefill\n',
        b'        ```"""\n        if cache_position is None:\n'
        b"            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0\n"
        b"            sequence_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]\n"
        b"            cache_position = torch.arange(\n"
        b"                past_seen_tokens,\n"
        b"                past_seen_tokens + sequence_length,\n"
        b"                device=(inputs_embeds if inputs_embeds is not None else input_ids).device,\n"
        b"            )\n\n        # Prefill\n",
        "Qwen3-TTS model cache position PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b"            attn_implementation=requested_attn_implementation,\n            **kwargs,\n        )\n        if not local_files_only",
        b"            attn_implementation=requested_attn_implementation,\n            **kwargs,\n        )\n        from ..._transformers_compat import restore_rope_buffers\n\n        restore_rope_buffers(model)\n        if not local_files_only",
        "Qwen3-TTS model RoPE restoration PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b"        from ..._transformers_compat import restore_rope_buffers\n\n"
        b"        restore_rope_buffers(model)\n        if not local_files_only",
        b"        from ..._transformers_compat import restore_rope_buffers, strict_reload_local_safetensors\n\n"
        b"        model._qwen3_tts_strict_reload = strict_reload_local_safetensors(\n"
        b"            model,\n"
        b"            pretrained_model_name_or_path,\n"
        b"            cache_dir=cache_dir,\n"
        b"            revision=revision,\n"
        b"            token=token,\n"
        b"            local_files_only=local_files_only,\n"
        b"        )\n\n"
        b"        restore_rope_buffers(model)\n        if not local_files_only",
        "Qwen3-TTS strict local safetensors reload hunk",
    )
    return _identity(patched, PATCH_MODEL_PATCHED_BYTES, PATCH_MODEL_PATCHED_SHA256, "Qwen3-TTS model patched")


def patch_source_bytes(original: bytes) -> bytes:
    _identity(original, PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, "Qwen3-TTS tokenizer original")
    patched = _replace_once(original, b"from transformers.utils.generic import check_model_inputs\n", b"", "Qwen3-TTS tokenizer decorator import PR #360 hunk")
    patched = _replace_once(
        patched,
        b"    @check_model_inputs()\n    @auto_docstring\n    def forward(",
        b"    def forward(",
        "Qwen3-TTS tokenizer decorator PR #360 hunk",
    )
    patched = _replace_once(
        patched,
        b'                "input_embeds": inputs_embeds,\n'
        b"                \"attention_mask\": attention_mask,\n"
        b"                \"cache_position\": cache_position,\n",
        b'                "inputs_embeds": inputs_embeds,\n'
        b"                \"attention_mask\": attention_mask,\n",
        "Qwen3-TTS tokenizer mask kwargs PR #360 hunk",
    )
    return _identity(patched, PATCHED_BYTES, PATCHED_SHA256, "Qwen3-TTS tokenizer patched")


def patch_25hz_source_bytes(original: bytes) -> bytes:
    _identity(original, PATCH_25HZ_ORIGINAL_BYTES, PATCH_25HZ_ORIGINAL_SHA256, "Qwen3-TTS inference original")
    patched = _replace_once(
        original,
        b"        inst.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)\n        inst.config = inst.model.config",
        b"        inst.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)\n"
        b"        from .._transformers_compat import restore_mimi_full_attention, restore_rope_buffers\n\n"
        b"        restore_rope_buffers(inst.model)\n"
        b"        restore_mimi_full_attention(inst.model)\n"
        b"        inst.config = inst.model.config",
        "Qwen3-TTS inference restore PR #360 hunk",
    )
    if not patched.endswith(b"\n"):
        patched += b"\n"
    patched = _replace_once(
        patched,
        b"    Qwen3TTSTokenizerV1Config,\n    Qwen3TTSTokenizerV1Model,\n",
        b"",
        "Qwen3-TTS inference 25Hz V1 import removal",
    )
    patched = _replace_once(
        patched,
        b'        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)\n'
        b"        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)\n\n",
        b"\n",
        "Qwen3-TTS inference 25Hz V1 registration removal",
    )
    return _identity(patched, PATCH_25HZ_PATCHED_BYTES, PATCH_25HZ_PATCHED_SHA256, "Qwen3-TTS inference patched")


def _patch_records(originals: dict[str, bytes | None], patched: dict[str, bytes]) -> list[dict[str, Any]]:
    specs = (
        (PATCH_INIT_TARGET, PATCH_INIT_OPERATION, 1),
        (PATCH_HELPER_TARGET, "create_exact_helper", 1),
        (PATCH_CORE_25HZ_TARGET, PATCH_CORE_25HZ_OPERATION, 2),
        (PATCH_CONFIG_TARGET, PATCH_CONFIG_OPERATION, 5),
        (PATCH_MODEL_TARGET, PATCH_MODEL_OPERATION, 5),
        (PATCH_TARGET, PATCH_TOKENIZER_OPERATION, 3),
        (PATCH_25HZ_TARGET, PATCH_25HZ_OPERATION, 3),
    )
    rows = []
    for target, operation, count in specs:
        original = originals[target]
        output = patched[target]
        rows.append({
            "status": PATCH_STATUS,
            "target": target,
            "operation": operation,
            "original_state": "ABSENT" if original is None else "PRESENT",
            "original_bytes": 0 if original is None else len(original),
            "original_sha256": None if original is None else sha256_bytes(original),
            "patched_bytes": len(output),
            "patched_sha256": sha256_bytes(output),
            "replacement_count": count,
        })
    return rows


def _expected_patch_records() -> list[dict[str, Any]]:
    return [
        {"status": PATCH_STATUS, "target": PATCH_INIT_TARGET, "operation": PATCH_INIT_OPERATION, "original_state": "PRESENT", "original_bytes": PATCH_INIT_ORIGINAL_BYTES, "original_sha256": PATCH_INIT_ORIGINAL_SHA256, "patched_bytes": PATCH_INIT_PATCHED_BYTES, "patched_sha256": PATCH_INIT_PATCHED_SHA256, "replacement_count": 1},
        {"status": PATCH_STATUS, "target": PATCH_HELPER_TARGET, "operation": "create_exact_helper", "original_state": "ABSENT", "original_bytes": 0, "original_sha256": None, "patched_bytes": PATCH_HELPER_PATCHED_BYTES, "patched_sha256": PATCH_HELPER_PATCHED_SHA256, "replacement_count": 1},
        {"status": PATCH_STATUS, "target": PATCH_CORE_25HZ_TARGET, "operation": PATCH_CORE_25HZ_OPERATION, "original_state": "PRESENT", "original_bytes": PATCH_CORE_25HZ_ORIGINAL_BYTES, "original_sha256": PATCH_CORE_25HZ_ORIGINAL_SHA256, "patched_bytes": PATCH_CORE_25HZ_PATCHED_BYTES, "patched_sha256": PATCH_CORE_25HZ_PATCHED_SHA256, "replacement_count": 2},
        {"status": PATCH_STATUS, "target": PATCH_CONFIG_TARGET, "operation": PATCH_CONFIG_OPERATION, "original_state": "PRESENT", "original_bytes": PATCH_CONFIG_ORIGINAL_BYTES, "original_sha256": PATCH_CONFIG_ORIGINAL_SHA256, "patched_bytes": PATCH_CONFIG_PATCHED_BYTES, "patched_sha256": PATCH_CONFIG_PATCHED_SHA256, "replacement_count": 5},
        {"status": PATCH_STATUS, "target": PATCH_MODEL_TARGET, "operation": PATCH_MODEL_OPERATION, "original_state": "PRESENT", "original_bytes": PATCH_MODEL_ORIGINAL_BYTES, "original_sha256": PATCH_MODEL_ORIGINAL_SHA256, "patched_bytes": PATCH_MODEL_PATCHED_BYTES, "patched_sha256": PATCH_MODEL_PATCHED_SHA256, "replacement_count": 5},
        {"status": PATCH_STATUS, "target": PATCH_TARGET, "operation": PATCH_TOKENIZER_OPERATION, "original_state": "PRESENT", "original_bytes": PATCH_ORIGINAL_BYTES, "original_sha256": PATCH_ORIGINAL_SHA256, "patched_bytes": PATCHED_BYTES, "patched_sha256": PATCHED_SHA256, "replacement_count": 3},
        {"status": PATCH_STATUS, "target": PATCH_25HZ_TARGET, "operation": PATCH_25HZ_OPERATION, "original_state": "PRESENT", "original_bytes": PATCH_25HZ_ORIGINAL_BYTES, "original_sha256": PATCH_25HZ_ORIGINAL_SHA256, "patched_bytes": PATCH_25HZ_PATCHED_BYTES, "patched_sha256": PATCH_25HZ_PATCHED_SHA256, "replacement_count": 3},
    ]


def compatibility_patch_record() -> dict[str, Any]:
    """Return provenance and the canonical seven-target source contract."""
    return {
        "status": PATCH_STATUS,
        "operation": "apply_exactly_seven_source_transforms",
        "patch_count": 7,
        "source_repository": SOURCE_REPOSITORY,
        "source_base_revision": SOURCE_BASE_REVISION,
        "source_head_revision": SOURCE_HEAD_REVISION,
        "source_pr_url": SOURCE_PR_URL,
        "source_pr_status": SOURCE_PR_STATUS,
        "transformers_version": TRANSFORMERS_VERSION,
        "patches": _expected_patch_records(),
    }


def validate_patch_record(record: Any) -> None:
    if record != compatibility_patch_record():
        raise CompatibilityPatchError("compatibility patch evidence is not the fixed seven-target contract")


def _source_targets(source: Path) -> dict[str, Path]:
    targets = {relative: source / relative for relative in COMPATIBILITY_PATCH_TARGETS}
    for relative, target in targets.items():
        try:
            target.resolve(strict=False).relative_to(source)
        except ValueError as error:
            raise CompatibilityPatchError(f"compatibility patch target escapes source checkout: {relative}") from error
    return targets


def _git_status(source: Path) -> list[str]:
    return subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout.rstrip("\r\n").splitlines()


def verify_patched_source_checkout(source: Path, record: Any = None) -> dict[str, Any]:
    """Verify a reused seven-target patched checkout without changing it."""
    if source.is_symlink() or not source.is_dir():
        raise CompatibilityPatchError("patched source checkout is missing or symlinked")
    source = source.resolve(strict=False)
    _reject_generated_bytecode(source)
    targets = _source_targets(source)
    for relative, target in targets.items():
        if target.is_symlink() or not target.is_file():
            raise CompatibilityPatchError(f"patched compatibility target is missing or symlinked: {relative}")
    expected_status = [f" M {relative}" for relative in sorted(COMPATIBILITY_PATCH_TARGETS) if relative != PATCH_HELPER_TARGET]
    expected_status.append(f"?? {PATCH_HELPER_TARGET}")
    actual_status = _git_status(source)
    status_key = lambda line: line[3:] if len(line) >= 3 else line
    if sorted(actual_status, key=status_key) != sorted(expected_status, key=status_key):
        raise CompatibilityPatchError(f"patched source has unexpected status: {actual_status!r}")
    expected = _expected_patch_records()
    for row in expected:
        target = targets[row["target"]]
        content = target.read_bytes()
        if len(content) != row["patched_bytes"] or sha256_bytes(content) != row["patched_sha256"]:
            raise CompatibilityPatchError(f"patched source identity drifted: {row['target']}")
    canonical = compatibility_patch_record()
    if record is not None:
        validate_patch_record(record)
    return canonical


def _rollback(source: Path, targets: dict[str, Path], originals: dict[str, bytes | None], replaced: list[str]) -> None:
    for relative in reversed(replaced):
        target = targets[relative]
        if target.is_symlink():
            raise CompatibilityPatchError(f"cannot rollback symlinked target: {relative}")
        original = originals[relative]
        if original is None:
            target.unlink(missing_ok=True)
            continue
        rollback = target.with_name(f".{target.name}.{os.getpid()}.{len(replaced)}.rollback.tmp")
        if rollback.exists() or rollback.is_symlink():
            raise CompatibilityPatchError(f"rollback temporary path already exists: {rollback.name}")
        fd = os.open(rollback, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(original)
            os.replace(rollback, target)
        except BaseException:
            rollback.unlink(missing_ok=True)
            raise
    if _git_status(source):
        raise CompatibilityPatchError("rollback left source dirty")


def patch_source_checkout(source: Path, *, fail_after_helper_creation: bool = False) -> dict[str, Any]:
    """Apply the seven transforms atomically, or leave the clean checkout untouched."""
    if source.is_symlink() or not source.is_dir():
        raise CompatibilityPatchError("official source checkout path must not be a symlink")
    source = source.resolve(strict=False)
    _reject_generated_bytecode(source)
    targets = _source_targets(source)
    for relative, target in targets.items():
        if target.is_symlink() or (relative == PATCH_HELPER_TARGET and target.exists()) or (relative != PATCH_HELPER_TARGET and not target.is_file()):
            raise CompatibilityPatchError(f"compatibility target is missing or symlinked: {relative}")
    if _git_status(source):
        raise CompatibilityPatchError("official source must be clean before compatibility patch")
    originals: dict[str, bytes | None] = {}
    for relative, target in targets.items():
        originals[relative] = None if relative == PATCH_HELPER_TARGET and not target.exists() else target.read_bytes()
        if relative == PATCH_HELPER_TARGET and originals[relative] is not None:
            raise CompatibilityPatchError("initially absent helper is already present")
    patched = {
        PATCH_INIT_TARGET: patch_init_source_bytes(originals[PATCH_INIT_TARGET]),
        PATCH_HELPER_TARGET: _identity(_HELPER_SOURCE, PATCH_HELPER_PATCHED_BYTES, PATCH_HELPER_PATCHED_SHA256, "Transformers compatibility helper"),
        PATCH_CORE_25HZ_TARGET: patch_core_25hz_source_bytes(originals[PATCH_CORE_25HZ_TARGET]),
        PATCH_CONFIG_TARGET: patch_config_source_bytes(originals[PATCH_CONFIG_TARGET]),
        PATCH_MODEL_TARGET: patch_model_source_bytes(originals[PATCH_MODEL_TARGET]),
        PATCH_TARGET: patch_source_bytes(originals[PATCH_TARGET]),
        PATCH_25HZ_TARGET: patch_25hz_source_bytes(originals[PATCH_25HZ_TARGET]),
    }
    temporary: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for index, relative in enumerate(COMPATIBILITY_PATCH_TARGETS):
            target = targets[relative]
            temporary_path = target.with_name(f".{target.name}.{os.getpid()}.{index}.compat.tmp")
            if temporary_path.exists() or temporary_path.is_symlink():
                raise CompatibilityPatchError(f"compatibility patch temporary path already exists: {temporary_path.name}")
            temporary[relative] = temporary_path
            fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(patched[relative])
            if relative != PATCH_HELPER_TARGET:
                os.chmod(temporary_path, target.stat().st_mode & 0o777)
        for relative in COMPATIBILITY_PATCH_TARGETS:
            os.replace(temporary[relative], targets[relative])
            replaced.append(relative)
            if relative == PATCH_HELPER_TARGET and fail_after_helper_creation:
                raise CompatibilityPatchError("injected failure after helper creation")
    except BaseException as error:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        try:
            _rollback(source, targets, originals, replaced)
        except BaseException as rollback_error:
            raise CompatibilityPatchError(f"compatibility patch failed and rollback failed: {rollback_error}") from error
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    try:
        verify = verify_patched_source_checkout(source)
        record = compatibility_patch_record()
        record["patches"] = _patch_records(originals, patched)
        validate_patch_record(record)
        if verify != record:
            raise CompatibilityPatchError("patched source verification record drifted")
        return record
    except BaseException as error:
        try:
            _rollback(source, targets, originals, list(COMPATIBILITY_PATCH_TARGETS))
        except BaseException as rollback_error:
            raise CompatibilityPatchError(f"compatibility patch validation failed and rollback failed: {rollback_error}") from error
        raise


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _clean_repo(root: Path, files: dict[str, bytes]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git("init", "--quiet", cwd=root)
    _git("config", "user.email", "self-test@example.invalid", cwd=root)
    _git("config", "user.name", "Qwen self-test", cwd=root)
    _git("add", *files.keys(), cwd=root)
    _git("commit", "--quiet", "-m", "fixture", cwd=root)


def self_test_filesystem() -> None:
    """Exercise all seven identities, reused verification, and rollback safety."""
    global _identity
    names = (
        "PATCH_INIT_ORIGINAL_BYTES", "PATCH_INIT_ORIGINAL_SHA256", "PATCH_INIT_PATCHED_BYTES", "PATCH_INIT_PATCHED_SHA256",
        "PATCH_CORE_25HZ_ORIGINAL_BYTES", "PATCH_CORE_25HZ_ORIGINAL_SHA256", "PATCH_CORE_25HZ_PATCHED_BYTES", "PATCH_CORE_25HZ_PATCHED_SHA256",
        "PATCH_CONFIG_ORIGINAL_BYTES", "PATCH_CONFIG_ORIGINAL_SHA256", "PATCH_CONFIG_PATCHED_BYTES", "PATCH_CONFIG_PATCHED_SHA256",
        "PATCH_MODEL_ORIGINAL_BYTES", "PATCH_MODEL_ORIGINAL_SHA256", "PATCH_MODEL_PATCHED_BYTES", "PATCH_MODEL_PATCHED_SHA256",
        "PATCH_ORIGINAL_BYTES", "PATCH_ORIGINAL_SHA256", "PATCHED_BYTES", "PATCHED_SHA256",
        "PATCH_25HZ_ORIGINAL_BYTES", "PATCH_25HZ_ORIGINAL_SHA256", "PATCH_25HZ_PATCHED_BYTES", "PATCH_25HZ_PATCHED_SHA256",
    )
    saved = {name: globals()[name] for name in names}
    init = b'"""\n\nfrom .inference.qwen3_tts_model import Model\n'
    core = b"from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config\nfrom .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model\nfrom .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config\n"
    config = (b"from transformers.configuration_utils import PretrainedConfig, layer_type_validation\n"
              b"from transformers.modeling_rope_utils import rope_config_validation\n"
              b'class Predictor:\n    model_type = "qwen3_tts_talker_code_predictor"\n    keys_to_ignore_at_inference = []\n'
              b"        rope_config_validation(self)\n"
              b'                else "full_attention"\n                for i in range(self.num_hidden_layers)\n            ]\n        layer_type_validation(self.layer_types)\n'
              b'class Talker:\n    model_type = "qwen3_tts_talker"\n    keys_to_ignore_at_inference = []\n')
    model = (b"            # Prepare mask arguments\n            mask_kwargs = {\n                \"config\": self.config,\n                \"input_embeds\": inputs_embeds,\n                \"attention_mask\": attention_mask,\n                \"cache_position\": cache_position,\n                \"past_key_values\": past_key_values,\n            }\n            # Create the masks\n            causal_mask_mapping = {\n                \"full_attention\": create_causal_mask(**mask_kwargs),\n            }\n"
             b"        causal_mask = mask_function(\n            config=self.config,\n            input_embeds=inputs_embeds,\n            attention_mask=attention_mask,\n            cache_position=cache_position,\n            past_key_values=past_key_values,\n            position_ids=text_position_ids,\n        )\n"
             b'        ```"""\n        # Prefill\n'
             b"            attn_implementation=requested_attn_implementation,\n            **kwargs,\n        )\n        if not local_files_only\n")
    tokenizer = (b"from transformers.utils.generic import check_model_inputs\n"
                 b"    @check_model_inputs()\n    @auto_docstring\n    def forward(\n"
                 b'                "input_embeds": inputs_embeds,\n                "attention_mask": attention_mask,\n                "cache_position": cache_position,\n')
    inference = (b"from ..core import (\n    Qwen3TTSTokenizerV1Config,\n    Qwen3TTSTokenizerV1Model,\n    Qwen3TTSTokenizerV2Config,\n)\n"
                 b'        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)\n'
                 b"        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)\n\n"
                 b"        inst.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)\n        inst.config = inst.model.config")
    files = {PATCH_INIT_TARGET: init, PATCH_CORE_25HZ_TARGET: core, PATCH_CONFIG_TARGET: config, PATCH_MODEL_TARGET: model, PATCH_TARGET: tokenizer, PATCH_25HZ_TARGET: inference}
    old_identity = _identity
    def expect_error(action: Callable[[], Any], label: str) -> None:
        try:
            action()
        except CompatibilityPatchError:
            return
        raise AssertionError(f"{label} was accepted")

    try:
        _identity = lambda content, size, digest, label: content
        patched = {PATCH_INIT_TARGET: patch_init_source_bytes(init), PATCH_CORE_25HZ_TARGET: patch_core_25hz_source_bytes(core), PATCH_CONFIG_TARGET: patch_config_source_bytes(config), PATCH_MODEL_TARGET: patch_model_source_bytes(model), PATCH_TARGET: patch_source_bytes(tokenizer), PATCH_25HZ_TARGET: patch_25hz_source_bytes(inference), PATCH_HELPER_TARGET: _HELPER_SOURCE}
        assert b"import os\n" in _HELPER_SOURCE
        assert b"def strict_reload_local_safetensors(" in _HELPER_SOURCE
        assert b"strict_reload_local_safetensors" in patched[PATCH_MODEL_TARGET]
        contracts = (("PATCH_INIT", init, patched[PATCH_INIT_TARGET]), ("PATCH_CORE_25HZ", core, patched[PATCH_CORE_25HZ_TARGET]), ("PATCH_CONFIG", config, patched[PATCH_CONFIG_TARGET]), ("PATCH_MODEL", model, patched[PATCH_MODEL_TARGET]), ("PATCH_TOKENIZER", tokenizer, patched[PATCH_TARGET]), ("PATCH_25HZ", inference, patched[PATCH_25HZ_TARGET]))
        for prefix, original, output in contracts:
            if prefix == "PATCH_TOKENIZER":
                globals()["PATCH_ORIGINAL_BYTES"] = len(original); globals()["PATCH_ORIGINAL_SHA256"] = sha256_bytes(original)
                globals()["PATCHED_BYTES"] = len(output); globals()["PATCHED_SHA256"] = sha256_bytes(output)
            else:
                globals()[prefix + "_ORIGINAL_BYTES"] = len(original); globals()[prefix + "_ORIGINAL_SHA256"] = sha256_bytes(original)
                globals()[prefix + "_PATCHED_BYTES"] = len(output); globals()[prefix + "_PATCHED_SHA256"] = sha256_bytes(output)
        _identity = old_identity
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-seven-") as directory:
            root = Path(directory); _clean_repo(root, files); record = patch_source_checkout(root)
            assert record["patch_count"] == 7 and verify_patched_source_checkout(root, record) == record
            assert (root / PATCH_HELPER_TARGET).read_bytes() == _HELPER_SOURCE
            expect_error(lambda: patch_source_checkout(root), "already-patched source")
            reordered = {**record, "patches": list(reversed(record["patches"]))}
            expect_error(lambda: verify_patched_source_checkout(root, reordered), "reordered patch evidence")
            tampered = {**record, "patches": [dict(row) for row in record["patches"]]}
            tampered["patches"][0]["patched_bytes"] += 1
            expect_error(lambda: verify_patched_source_checkout(root, tampered), "tampered patch evidence")
            (root / "extra.txt").write_text("unexpected", encoding="utf-8")
            expect_error(lambda: verify_patched_source_checkout(root, record), "extra untracked source")
            (root / "extra.txt").unlink()
            (root / PATCH_CORE_25HZ_TARGET).write_bytes(files[PATCH_CORE_25HZ_TARGET])
            expect_error(lambda: verify_patched_source_checkout(root), "partial patched source")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-unpatched-") as directory:
            root = Path(directory); _clean_repo(root, files)
            expect_error(lambda: verify_patched_source_checkout(root), "clean unpatched source")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-rollback-") as directory:
            root = Path(directory); _clean_repo(root, files); before = {key: (root / key).read_bytes() for key in files}
            try: patch_source_checkout(root, fail_after_helper_creation=True)
            except CompatibilityPatchError: pass
            else: raise AssertionError("injected helper failure was accepted")
            assert not (root / PATCH_HELPER_TARGET).exists() and _git_status(root) == []
            assert before == {key: (root / key).read_bytes() for key in files}
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-helper-present-") as directory:
            root = Path(directory); _clean_repo(root, files); (root / PATCH_HELPER_TARGET).write_bytes(_HELPER_SOURCE)
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("pre-existing helper was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-helper-directory-") as directory:
            root = Path(directory); _clean_repo(root, files); (root / PATCH_HELPER_TARGET).mkdir()
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("helper directory was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-dirty-") as directory:
            root = Path(directory); _clean_repo(root, files); (root / "unrelated.txt").write_text("dirty", encoding="utf-8")
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("dirty source was accepted")
            assert not (root / PATCH_HELPER_TARGET).exists()
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-bytecode-") as directory:
            root = Path(directory); _clean_repo(root, files); cache = root / "qwen_tts" / "__pycache__"; cache.mkdir(); (cache / "x.pyc").write_bytes(b"bytecode")
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("generated bytecode source was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-symlink-") as directory:
            root = Path(directory); _clean_repo(root, files); outside = root / "outside.py"; outside.write_bytes(files[PATCH_MODEL_TARGET])
            target = root / PATCH_MODEL_TARGET; target.unlink(); target.symlink_to(outside)
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("symlink target was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-escape-") as directory:
            root = Path(directory); _clean_repo(root, files); old_targets = COMPATIBILITY_PATCH_TARGETS
            try:
                globals()["COMPATIBILITY_PATCH_TARGETS"] = ("../outside.py",) + old_targets[1:]
                try: patch_source_checkout(root)
                except CompatibilityPatchError: pass
                else: raise AssertionError("path escape was accepted")
            finally:
                globals()["COMPATIBILITY_PATCH_TARGETS"] = old_targets
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-tampered-") as directory:
            root = Path(directory); _clean_repo(root, files); record = patch_source_checkout(root)
            (root / PATCH_MODEL_TARGET).write_bytes(b"tampered")
            expect_error(lambda: verify_patched_source_checkout(root, record), "tampered identity")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-drift-") as directory:
            root = Path(directory); _clean_repo(root, files)
            (root / PATCH_MODEL_TARGET).write_bytes(b"drift")
            before = {key: (root / key).read_bytes() for key in files}
            expect_error(lambda: patch_source_checkout(root), "pre-patch single-target hash drift")
            assert before == {key: (root / key).read_bytes() for key in files}
            assert not (root / PATCH_HELPER_TARGET).exists()
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-temp-") as directory:
            root = Path(directory); _clean_repo(root, files)
            temporary = (root / PATCH_INIT_TARGET).with_name(f".{Path(PATCH_INIT_TARGET).name}.{os.getpid()}.0.compat.tmp")
            temporary.write_bytes(b"must-not-clobber")
            _git("add", str(temporary.relative_to(root)), cwd=root)
            _git("commit", "--quiet", "-m", "temp guard", cwd=root)
            before = (root / PATCH_INIT_TARGET).read_bytes()
            expect_error(lambda: patch_source_checkout(root), "existing compatibility temp path")
            assert temporary.read_bytes() == b"must-not-clobber"
            assert (root / PATCH_INIT_TARGET).read_bytes() == before
    finally:
        _identity = old_identity
        for name, value in saved.items(): globals()[name] = value
