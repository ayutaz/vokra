#!/usr/bin/env -S uv run --frozen --project tools/parity/vibevoice_realtime_0_5b_reference python
"""Probe the pinned Transformers DynamicCache API without loading a model.

``--self-test`` is stdlib-only and is safe on the maintainer Mac.  ``--probe``
is intentionally VAST/Linux-x86_64-only: it imports the pinned Transformers
wheel and creates tiny synthetic tensors solely to exercise the public
``DynamicCache(ddp_cache_data=...)`` constructor and layer API.  It never
downloads or loads VibeVoice weights, a preset, or a checkpoint, and it makes
no model-shape or parity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


TRANSFORMERS_VERSION = "5.10.4"
TRANSFORMERS_WHEEL_SHA256 = "8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e"
TRANSFORMERS_WHEEL_URL = "https://files.pythonhosted.org/packages/d7/f1/d66881f28d3e64002a21d043c7c8db306c0ad5a711c85337ff551bfbc040/transformers-5.10.4-py3-none-any.whl"
CACHE_UTILS_SHA256 = "7827cec593e6e6fa2ea123abce94eb422424d30a3391f15b38414684d0bbcd33"
SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
FORMAT = "vokra-vibevoice-dynamic-cache-compat-probe-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_contract(lock_text: str) -> dict[str, str]:
    block_match = re.search(
        r'\[\[package\]\]\s+name = "transformers"\s+version = "([^"]+)"(?P<body>.*?)(?=\n\[\[package\]\]|\Z)',
        lock_text,
        re.DOTALL,
    )
    if block_match is None:
        raise RuntimeError("uv.lock has no transformers package block")
    version = block_match.group(1)
    body = block_match.group("body")
    wheel_match = re.search(
        r'url = "([^"]+/transformers-5\.10\.4-py3-none-any\.whl)", hash = "sha256:([0-9a-f]{64})"',
        body,
    )
    if version != TRANSFORMERS_VERSION or wheel_match is None:
        raise RuntimeError("uv.lock does not bind the required Transformers wheel")
    wheel_url, wheel_sha256 = wheel_match.groups()
    if wheel_url != TRANSFORMERS_WHEEL_URL or wheel_sha256 != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("uv.lock Transformers wheel URL/SHA-256 drifted")
    return {"version": version, "wheel_url": wheel_url, "wheel_sha256": wheel_sha256}


def _platform_guard() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("DynamicCache probe is VAST/Linux x86_64 only")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise RuntimeError("DynamicCache probe requires VOKRA_PUBLISH_ON_VAST=1")


def _source_facts(source_root: Path | None) -> dict[str, Any]:
    if source_root is None:
        return {"checked": False}
    source_file = source_root / "vibevoice/modular/modeling_vibevoice_streaming_inference.py"
    if not source_file.is_file():
        raise RuntimeError("pinned Microsoft source file is missing")
    try:
        head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"], text=True)
        origin = subprocess.check_output(["git", "-C", str(source_root), "remote", "get-url", "origin"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("official source must be a git checkout") from error
    if head != SOURCE_REVISION or status:
        raise RuntimeError("official source is not clean at the pinned revision")
    if origin not in {"https://github.com/microsoft/VibeVoice.git", "git@github.com:microsoft/VibeVoice.git", "ssh://git@github.com/microsoft/VibeVoice.git"}:
        raise RuntimeError(f"unexpected official source origin: {origin!r}")
    source = source_file.read_text(encoding="utf-8")
    required = ("class MockCacheLayer", "def _ensure_cache_has_layers", "def _init_cache_for_generation")
    if any(marker not in source for marker in required):
        raise RuntimeError("official source cache compatibility markers are missing")
    return {
        "checked": True,
        "revision": head,
        "origin": "https://github.com/microsoft/VibeVoice.git",
        "helper": "_ensure_cache_has_layers",
        "official_helper_has_get_seq_length": "get_seq_length" in source[source.index("class MockCacheLayer") : source.index("def _update_model_kwargs_for_generation")],
    }


def _probe(lock_path: Path, source_root: Path | None) -> None:
    _platform_guard()
    lock_path = lock_path.resolve()
    lock_contract = _lock_contract(lock_path.read_text(encoding="utf-8"))
    source_facts = _source_facts(source_root)

    # Imports and synthetic tensors are deliberately below the VAST guard.
    import torch
    import transformers
    from transformers.cache_utils import DynamicCache
    import transformers.cache_utils as cache_utils
    from transformers.modeling_outputs import BaseModelOutputWithPast

    cache_utils_path = Path(cache_utils.__file__).resolve()
    if _sha256_file(cache_utils_path) != CACHE_UTILS_SHA256:
        raise RuntimeError("installed transformers.cache_utils.py does not match the authenticated wheel source")

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"loaded Transformers {transformers.__version__}, expected {TRANSFORMERS_VERSION}")
    signature = inspect.signature(DynamicCache)
    required_parameters = {"ddp_cache_data", "config", "offloading", "offload_only_non_sliding"}
    if not required_parameters.issubset(signature.parameters):
        raise RuntimeError("DynamicCache constructor lacks the pinned public parameters")

    # Four official DynamicCache instances carry synthetic legacy attributes;
    # no fake cache class or numerical mirror is introduced. This exercises
    # the production migration and F32 cast helpers without a model/preset.
    reference_path = Path(__file__).with_name("run_streaming_reference.py")
    spec = importlib.util.spec_from_file_location("vokra_streaming_reference", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the production reference helper")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    key_a = torch.zeros((1, 1, 2, 1), dtype=torch.bfloat16)
    value_a = torch.ones((1, 1, 2, 1), dtype=torch.bfloat16)
    key_b = torch.full((1, 1, 2, 1), 2.0, dtype=torch.bfloat16)
    value_b = torch.full((1, 1, 2, 1), 3.0, dtype=torch.bfloat16)
    source_pairs = ((key_a, value_a), (key_b, value_b))
    outputs = {}
    for branch in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
        legacy = DynamicCache()
        legacy.key_cache = [key.clone() for key, _value in source_pairs]
        legacy.value_cache = [value.clone() for _key, value in source_pairs]
        migrated = reference._migrate_legacy_cache(legacy, branch, torch, DynamicCache)
        outputs[branch] = BaseModelOutputWithPast(
            last_hidden_state=torch.zeros((1, 1, 4), dtype=torch.bfloat16),
            past_key_values=migrated,
        )
    outputs = reference._cast_preset_to_f32(outputs, torch)
    for branch, output in outputs.items():
        cache = output.past_key_values
        if int(cache.get_seq_length(0)) != 2 or tuple(cache.get_mask_sizes(1, 0)) != (3, 0):
            raise RuntimeError(f"{branch} cache failed post-cast query-length API")
        for index, (expected_key, expected_value) in enumerate(source_pairs):
            layer = cache.layers[index]
            if layer.keys.dtype != torch.float32 or layer.values.dtype != torch.float32 or layer.dtype != torch.float32 or layer.device != layer.keys.device:
                raise RuntimeError(f"{branch} cache layer {index} has stale dtype/device metadata after cast")
            if not torch.equal(layer.keys.to(dtype=torch.bfloat16), expected_key) or not torch.equal(layer.values.to(dtype=torch.bfloat16), expected_value):
                raise RuntimeError(f"{branch} cache layer {index} changed values during migration/cast")
    appended_key = torch.full((1, 1, 1, 1), 4.0, dtype=torch.float32)
    appended_value = torch.full((1, 1, 1, 1), 5.0, dtype=torch.float32)
    cache = outputs["tts_lm"].past_key_values
    returned_key, returned_value = cache.update(appended_key, appended_value, layer_idx=0)
    if returned_key.shape[2] != 3 or returned_value.shape[2] != 3 or tuple(cache.get_mask_sizes(1, 0)) != (4, 0) or not torch.equal(returned_key[..., -1:, :], appended_key) or not torch.equal(returned_value[..., -1:, :], appended_value):
        raise RuntimeError("official DynamicCache update did not preserve post-cast values/length")
    result = {
        "format": FORMAT,
        "status": "API_ONLY_NO_MODEL_EXECUTION",
        "publication": "NO_UPLOAD",
        "lock": {"path": str(lock_path), "sha256": _sha256_file(lock_path), **lock_contract},
        "source": source_facts,
        "transformers": {"version": transformers.__version__, "wheel_sha256": TRANSFORMERS_WHEEL_SHA256, "cache_utils_sha256": CACHE_UTILS_SHA256, "cache_utils_path": str(cache_utils_path), "constructor_parameters": sorted(signature.parameters)},
        "synthetic_probe": {"branches": ["lm", "tts_lm", "neg_lm", "neg_tts_lm"], "layer_count": 2, "initial_seq_length": 2, "updated_layer_0_seq_length": int(cache.get_seq_length(0)), "shape_claim": "GENERIC_API_ONLY_NOT_VIBEVOICE_SHAPES", "production_helpers": ["_migrate_legacy_cache", "_cast_preset_to_f32"]},
        "model_download": "NO",
        "model_load": "NO",
        "model_forward": "NO",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def _self_test() -> None:
    sample = """[[package]]\nname = \"transformers\"\nversion = \"5.10.4\"\nwheels = [\n    { url = \"https://files.pythonhosted.org/packages/d7/f1/d66881f28d3e64002a21d043c7c8db306c0ad5a711c85337ff551bfbc040/transformers-5.10.4-py3-none-any.whl\", hash = \"sha256:8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e\" },\n]\n"""
    assert _lock_contract(sample)["version"] == TRANSFORMERS_VERSION
    try:
        _lock_contract(sample.replace(TRANSFORMERS_WHEEL_SHA256, "0" * 64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("tampered Transformers wheel hash was accepted")
    source = Path(__file__).read_text(encoding="utf-8")
    assert "model_forward" in source and "NO" in source
    print("DynamicCache compatibility probe self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.probe:
        parser.error("real API probing requires --probe; --self-test is the local model-free path")
    _probe(args.lock, args.source_root)


if __name__ == "__main__":
    main()
