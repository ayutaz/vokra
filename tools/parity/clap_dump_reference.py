#!/usr/bin/env python3
"""Dump an independent CLAP audio reference from official Transformers.

The model and processor are resolved from one immutable Hugging Face
revision. Tensor names and shapes are read from the loaded state dict (never
reconstructed here), which makes the resulting manifest the evidence needed
before a native GGUF binder can be enabled. This sidecar is VAST-only and is
not part of the Vokra runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


REPOSITORY = "laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
SAMPLE_RATE = 48_000
PCM_SECONDS = 10
PCM_SAMPLES = SAMPLE_RATE * PCM_SECONDS
DUMPER_VERSION = 2

# These are the released preprocessor axes at ``REVISION``.  They are the
# contract we authenticate from the official processor on VAST; they are not
# a clean-room reconstruction of its mel implementation.
PREPROCESSOR_CONTRACT = {
    "chunk_length_s": 10,
    "feature_extractor_type": "ClapFeatureExtractor",
    "feature_size": 64,
    "fft_window_size": 1024,
    "frequency_max": 14000,
    "frequency_min": 50,
    "hop_length": 480,
    "max_length_s": 10,
    "n_fft": 1024,
    "nb_frequency_bins": 513,
    "nb_max_frames": 1000,
    "nb_max_samples": 480000,
    "padding": "repeatpad",
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "ClapProcessor",
    "return_attention_mask": False,
    "sampling_rate": 48000,
    "top_db": None,
    "truncation": "fusion",
}
SERIALIZER_EXCLUDED_KEYS = {"processor_class"}

# Transformers' released ClapModel has explicit audio/text towers and two
# projection modules.  The inspector records every observed shape/dtype but
# never supplies expected dimensions from this table.  Unknown names fail
# closed so a future upstream rename cannot silently enter a native binder.
ROLE_PREFIXES = {
    "audio_tower": ("audio_model.",),
    "text_tower": ("text_model.",),
    "audio_projection": ("audio_projection.",),
    "text_projection": ("text_projection.",),
}
# These names are the two independent contrastive temperatures in the pinned
# Transformers 5.10.4 CLAP implementation. Treating an old/sibling scalar
# name as equivalent would hide a source or checkpoint topology drift.
SCALAR_ROLES = {"logit_scale_a", "logit_scale_t"}

# These are architecture/configuration facts from the pinned release
# ``config.json``.  They describe the released HTSAT-fused and RoBERTa
# topology, not a guessed tensor manifest: individual state-dict shapes are
# still read from the official loaded model and recorded below.  Keeping this
# contract separate makes a config drift fail before a future native binder
# can mistake a sibling CLAP variant for this release.
MODEL_CONFIG_CONTRACT = {
    ("model_type",): "clap",
    ("hidden_size",): 768,
    ("projection_dim",): 512,
    ("projection_hidden_act",): "relu",
    ("num_hidden_layers",): 16,
    ("audio_config", "model_type"): "clap_audio_model",
    ("audio_config", "hidden_size"): 768,
    ("audio_config", "projection_dim"): 512,
    ("audio_config", "projection_hidden_size"): 768,
    ("audio_config", "patch_embeds_hidden_size"): 96,
    ("audio_config", "spec_size"): 256,
    ("audio_config", "patch_size"): 4,
    ("audio_config", "patch_stride"): [4, 4],
    ("audio_config", "num_mel_bins"): 64,
    ("audio_config", "num_hidden_layers"): 4,
    ("audio_config", "depths"): [2, 2, 6, 2],
    ("audio_config", "num_attention_heads"): [4, 8, 16, 32],
    ("audio_config", "window_size"): 8,
    ("audio_config", "enable_fusion"): True,
    ("audio_config", "enable_patch_fusion"): True,
    ("audio_config", "fusion_num_hidden_layers"): 2,
    ("audio_config", "fusion_type"): None,
    ("audio_config", "enable_patch_layer_norm"): True,
    ("text_config", "model_type"): "clap_text_model",
    ("text_config", "hidden_size"): 768,
    ("text_config", "projection_dim"): 512,
    ("text_config", "intermediate_size"): 3072,
    ("text_config", "num_hidden_layers"): 12,
    ("text_config", "num_attention_heads"): 12,
    ("text_config", "max_position_embeddings"): 514,
    ("text_config", "vocab_size"): 50265,
    ("text_config", "type_vocab_size"): 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> tuple[str, str]:
    from transformers import ClapModel

    source = inspect.getsourcefile(ClapModel)
    if source is None:
        raise RuntimeError("cannot locate official Transformers ClapModel source")
    path = Path(source).resolve()
    return str(path), sha256_file(path)


def validate_preprocessor_contract(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("official CLAP feature extractor config is not a dict")
    missing = sorted(set(PREPROCESSOR_CONTRACT) - set(value))
    mismatched = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in PREPROCESSOR_CONTRACT.items()
        if value.get(key) != expected
    }
    if missing or mismatched:
        raise RuntimeError(
            f"official CLAP preprocessing contract drifted: missing={missing}, mismatched={mismatched}"
        )
    return {key: value[key] for key in sorted(PREPROCESSOR_CONTRACT)}


def validate_feature_extractor_serializer_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact contract retained by ClapFeatureExtractor.to_dict()."""

    expected = {
        key: PREPROCESSOR_CONTRACT[key]
        for key in sorted(PREPROCESSOR_CONTRACT)
        if key not in SERIALIZER_EXCLUDED_KEYS
    }
    if not isinstance(value, dict):
        raise RuntimeError("official CLAP feature extractor serialization is not a dict")
    missing = sorted(set(expected) - set(value))
    mismatched = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if missing or mismatched:
        raise RuntimeError(
            "official CLAP feature extractor serializer contract drifted: "
            f"missing={missing}, mismatched={mismatched}"
        )
    return {key: value[key] for key in sorted(expected)}


def state_dict_role(name: str) -> str | None:
    if name in SCALAR_ROLES:
        return "contrastive_scalar"
    for role, prefixes in ROLE_PREFIXES.items():
        if name.startswith(prefixes):
            return role
    return None


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the released model topology without loading weights."""

    if not isinstance(config, dict):
        raise RuntimeError("official CLAP model config is not a dict")
    mismatched: dict[str, dict[str, Any]] = {}
    for path, expected in MODEL_CONFIG_CONTRACT.items():
        value: Any = config
        for component in path:
            if not isinstance(value, dict) or component not in value:
                mismatched[".".join(path)] = {
                    "expected": expected,
                    "actual": "<missing>",
                }
                break
            value = value[component]
        else:
            if value != expected:
                mismatched[".".join(path)] = {
                    "expected": expected,
                    "actual": value,
                }
    if mismatched:
        raise RuntimeError(f"official CLAP model config drifted: {mismatched}")
    return {".".join(path): config_value(config, path) for path in MODEL_CONFIG_CONTRACT}


def config_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for component in path:
        value = value[component]
    return value


def build_state_dict_manifest(state_dict: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    tensor_manifest: dict[str, dict[str, Any]] = {}
    roles: dict[str, list[str]] = {
        **{role: [] for role in ROLE_PREFIXES},
        "contrastive_scalar": [],
    }
    unknown: list[str] = []
    for name, tensor in sorted(state_dict.items()):
        role = state_dict_role(name)
        if role is None:
            unknown.append(name)
            continue
        roles[role].append(name)
        tensor_manifest[name] = {
            "role": role,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    required = [role for role in ROLE_PREFIXES if not roles[role]]
    missing_scalars = sorted(SCALAR_ROLES - set(roles["contrastive_scalar"]))
    if unknown or required or missing_scalars:
        raise RuntimeError(
            "official CLAP state-dict role contract is incomplete: "
            f"unknown={unknown[:8]}, missing_roles={required}, "
            f"missing_scalars={missing_scalars}"
        )
    return tensor_manifest, roles


def validate_tensor_manifest(
    tensor_manifest: dict[str, dict[str, Any]],
    roles: dict[str, list[str]],
) -> None:
    """Validate the emitted role/shape schema without touching tensor data."""

    if not tensor_manifest:
        raise RuntimeError("official CLAP state-dict tensor manifest is empty")
    expected_roles = set(ROLE_PREFIXES) | {"contrastive_scalar"}
    if set(roles) != expected_roles:
        raise RuntimeError(f"CLAP state-dict roles drifted: {sorted(roles)}")
    missing_roles = [role for role in ROLE_PREFIXES if not roles[role]]
    if missing_roles:
        raise RuntimeError(
            "CLAP state-dict role manifest is missing observed tensors: "
            f"{missing_roles}"
        )
    for name, entry in tensor_manifest.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("CLAP tensor manifest contains an empty tensor name")
        if not isinstance(entry, dict) or set(entry) != {"role", "shape", "dtype"}:
            raise RuntimeError(f"CLAP tensor manifest entry is malformed: {name}")
        role = entry["role"]
        shape = entry["shape"]
        dtype = entry["dtype"]
        if role not in expected_roles:
            raise RuntimeError(f"CLAP tensor manifest contains unknown role: {name}={role}")
        if not isinstance(shape, list) or any(
            not isinstance(axis, int) or axis < 0 for axis in shape
        ):
            raise RuntimeError(f"CLAP tensor manifest shape is malformed: {name}={shape}")
        if not isinstance(dtype, str) or not dtype.startswith("torch."):
            raise RuntimeError(f"CLAP tensor manifest dtype is malformed: {name}={dtype}")
    for scalar in sorted(SCALAR_ROLES):
        entry = tensor_manifest.get(scalar)
        if entry is None or entry["role"] != "contrastive_scalar" or entry["shape"] != []:
            raise RuntimeError(
                f"CLAP contrastive scalar {scalar} must be an observed scalar tensor"
            )


def deterministic_pcm() -> Any:
    import numpy as np

    samples = PCM_SAMPLES
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    signal = (
        0.31 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.17 * np.sin(2.0 * np.pi * 440.0 * time + 0.2)
        + 0.07 * np.sin(2.0 * np.pi * 880.0 * time + 0.7)
    )
    return np.ascontiguousarray(signal.astype(np.float32))


def self_test() -> None:
    """Check the deterministic, provenance-critical pieces without a model."""

    assert REPOSITORY == "laion/clap-htsat-fused"
    assert len(REVISION) == 40 and all(c in "0123456789abcdef" for c in REVISION)
    assert validate_preprocessor_contract(dict(PREPROCESSOR_CONTRACT)) == {
        key: PREPROCESSOR_CONTRACT[key] for key in sorted(PREPROCESSOR_CONTRACT)
    }
    serializer_contract = {
        key: value
        for key, value in PREPROCESSOR_CONTRACT.items()
        if key not in SERIALIZER_EXCLUDED_KEYS
    }
    assert validate_feature_extractor_serializer_contract(serializer_contract) == {
        key: serializer_contract[key] for key in sorted(serializer_contract)
    }
    synthetic_config = {
        "model_type": "clap",
        "hidden_size": 768,
        "projection_dim": 512,
        "projection_hidden_act": "relu",
        "num_hidden_layers": 16,
        "audio_config": {
            "model_type": "clap_audio_model",
            "hidden_size": 768,
            "projection_dim": 512,
            "projection_hidden_size": 768,
            "patch_embeds_hidden_size": 96,
            "spec_size": 256,
            "patch_size": 4,
            "patch_stride": [4, 4],
            "num_mel_bins": 64,
            "num_hidden_layers": 4,
            "depths": [2, 2, 6, 2],
            "num_attention_heads": [4, 8, 16, 32],
            "window_size": 8,
            "enable_fusion": True,
            "enable_patch_fusion": True,
            "fusion_num_hidden_layers": 2,
            "fusion_type": None,
            "enable_patch_layer_norm": True,
        },
        "text_config": {
            "model_type": "clap_text_model",
            "hidden_size": 768,
            "projection_dim": 512,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 514,
            "vocab_size": 50265,
            "type_vocab_size": 1,
        },
    }
    assert validate_model_config(synthetic_config)["audio_config.model_type"] == "clap_audio_model"
    tampered_config = dict(synthetic_config)
    tampered_config["audio_config"] = dict(synthetic_config["audio_config"])
    tampered_config["audio_config"]["enable_fusion"] = False
    try:
        validate_model_config(tampered_config)
    except RuntimeError:
        pass
    else:
        raise AssertionError("model config drift accepted")
    for key, value in PREPROCESSOR_CONTRACT.items():
        tampered = dict(PREPROCESSOR_CONTRACT)
        tampered[key] = "tampered" if isinstance(value, str) else -1
        try:
            validate_preprocessor_contract(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"preprocessor tamper accepted: {key}")
    class SyntheticTensor:
        shape = (1,)
        dtype = "torch.float32"

    class SyntheticScalar:
        shape = ()
        dtype = "torch.float32"

    synthetic_state = {
        "audio_model.audio_encoder.weight": SyntheticTensor(),
        "text_model.embeddings.weight": SyntheticTensor(),
        "audio_projection.linear1.weight": SyntheticTensor(),
        "text_projection.linear1.weight": SyntheticTensor(),
        "logit_scale_a": SyntheticScalar(),
        "logit_scale_t": SyntheticScalar(),
    }
    manifest, roles = build_state_dict_manifest(synthetic_state)
    validate_tensor_manifest(manifest, roles)
    assert set(roles) == set(ROLE_PREFIXES) | {"contrastive_scalar"}
    assert manifest["audio_model.audio_encoder.weight"]["role"] == "audio_tower"
    missing_audio_manifest = dict(manifest)
    missing_audio_roles = {role: list(names) for role, names in roles.items()}
    del missing_audio_manifest["audio_model.audio_encoder.weight"]
    missing_audio_roles["audio_tower"] = []
    try:
        validate_tensor_manifest(missing_audio_manifest, missing_audio_roles)
    except RuntimeError as error:
        assert "audio_tower" in str(error)
    else:
        raise AssertionError("missing audio role accepted by tensor manifest validator")
    try:
        build_state_dict_manifest(
            {**synthetic_state, "unexpected.weight": SyntheticTensor()}
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown state-dict role accepted")
    try:
        build_state_dict_manifest(
            {**synthetic_state, "logit_scale": SyntheticTensor()}
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected contrastive scalar accepted")
    incomplete_scalar_state = dict(synthetic_state)
    del incomplete_scalar_state["logit_scale_t"]
    try:
        build_state_dict_manifest(incomplete_scalar_state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing contrastive scalar accepted")
    try:
        build_state_dict_manifest(
            {
                "audio_model.audio_encoder.weight": SyntheticTensor(),
                "logit_scale_a": SyntheticTensor(),
                "logit_scale_t": SyntheticTensor(),
            }
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete state-dict role manifest accepted")
    # Keep this contract check dependency-free: the real NumPy/Torch path is
    # intentionally imported only by ``dump`` on the VAST reference host.
    pcm = [
        0.31 * math.sin(2.0 * math.pi * 220.0 * index / SAMPLE_RATE)
        + 0.17 * math.sin(2.0 * math.pi * 440.0 * index / SAMPLE_RATE + 0.2)
        + 0.07 * math.sin(2.0 * math.pi * 880.0 * index / SAMPLE_RATE + 0.7)
        for index in range(SAMPLE_RATE)
    ]
    assert len(pcm) == SAMPLE_RATE and all(math.isfinite(value) for value in pcm)


def dump(model_dir: str | None, output_dir: Path) -> None:
    import numpy as np
    import torch
    import transformers
    from transformers import ClapModel, ClapProcessor

    output_dir.mkdir(parents=True, exist_ok=False)
    source_path, source_sha256 = source_hash()
    processor_source_path = inspect.getsourcefile(ClapProcessor)
    if processor_source_path is None:
        raise RuntimeError("cannot locate official Transformers ClapProcessor source")
    processor_source_path = str(Path(processor_source_path).resolve())
    processor_source_sha256 = sha256_file(Path(processor_source_path))
    kwargs = {"revision": REVISION, "local_files_only": model_dir is not None}
    model_source = model_dir or REPOSITORY
    processor = ClapProcessor.from_pretrained(model_source, **kwargs)
    model = ClapModel.from_pretrained(model_source, torch_dtype=torch.float32, **kwargs)
    model.eval()
    resolved_revision = (
        Path(model_dir).resolve().name
        if model_dir is not None
        else getattr(model.config, "_commit_hash", None)
    )
    if resolved_revision != REVISION:
        raise RuntimeError(
            f"resolved snapshot revision is {resolved_revision!r}; expected {REVISION!r}"
        )

    state_dict = model.state_dict()
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is None or not hasattr(feature_extractor, "to_dict"):
        raise RuntimeError("official CLAP processor has no inspectable feature extractor")
    preprocessing = validate_feature_extractor_serializer_contract(feature_extractor.to_dict())
    if PREPROCESSOR_CONTRACT["processor_class"] != ClapProcessor.__name__:
        raise RuntimeError("official CLAP processor class source drifted")
    tensor_manifest, state_dict_roles = build_state_dict_manifest(state_dict)
    model_config_contract = validate_model_config(model.config.to_dict())
    validate_tensor_manifest(tensor_manifest, state_dict_roles)
    pcm = deterministic_pcm()
    inputs = processor(audios=[pcm], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    embedding = outputs.audio_embeds[0].detach().cpu().to(torch.float32).numpy()
    if embedding.ndim != 1 or embedding.size != 512:
        raise RuntimeError(f"official audio embedding shape is {embedding.shape}, expected model output")
    np.asarray(pcm, dtype="<f4").tofile(output_dir / "pcm.f32")
    np.asarray(embedding, dtype="<f4").tofile(output_dir / "audio_embedding.f32")

    metadata = {
        "contract": "vokra-clap-htsat-fused-reference-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "resolved_revision": resolved_revision,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "preprocessing": preprocessing,
        "preprocessing_processor_class": {
            "raw_release_value": PREPROCESSOR_CONTRACT["processor_class"],
            "api_class": ClapProcessor.__name__,
            "source": processor_source_path,
            "source_sha256": processor_source_sha256,
        },
        "state_dict_roles": state_dict_roles,
        "model_config_contract": model_config_contract,
        "dumper_version": DUMPER_VERSION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "runtime_platform": platform.platform(),
        "transformers_clap_model_source": source_path,
        "transformers_clap_model_source_sha256": source_sha256,
        "tensor_manifest": tensor_manifest,
        "model_config": model.config.to_dict(),
        "files_sha256": {},
        "parity_status": "INSPECTION_ONLY",
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "meta.json":
            metadata["files_sha256"][path.name] = sha256_file(path)
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir", help="local snapshot at the pinned revision")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.model_dir is not None or args.output_dir is not None:
            parser.error("--self-test accepts no model or output arguments")
        self_test()
        print("clap_dump_reference self-test: OK")
        return 0
    if args.output_dir is None:
        parser.error("normal runs require --output-dir")
    dump(args.model_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
