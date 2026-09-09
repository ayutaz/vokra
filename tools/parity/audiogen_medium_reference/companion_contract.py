"""Model-free AudioGen-Medium companion identity contract.

The AudioCraft release is a composite even though the public Vokra artifact
currently contains only the autoregressive LM.  This module records the
identity that a future, weight-bearing run must satisfy without downloading
or opening either companion.  Values are deliberately limited to facts
exposed by the upstream AudioCraft source/config and the public Hugging Face
file metadata; a payload-derived tensor manifest is not invented here.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA = "vokra-audiogen-medium-companion-contract-v1"

AUDIOGEN_REPOSITORY = "facebook/audiogen-medium"
AUDIOGEN_REVISION = "1277dd7dfd8fa57a205a70acc5de0ee90804502f"
AUDIOCRAFT_REPOSITORY = "https://github.com/facebookresearch/audiocraft.git"
AUDIOCRAFT_REVISION = "a2b96756956846e194c9255d0cdadc2b47c93f1b"
AUDIOCRAFT_TAG = "v1.0.0"

# AudioCraft's v1.0.0 config passes the mutable name ``t5-large`` to
# T5Tokenizer/T5EncoderModel.from_pretrained.  The canonical HF repository
# and immutable revision below are the reviewed companion candidate.  The
# source checkpoint does not record this revision, so the contract does not
# call historical linkage proven until a payload-bearing run authenticates it.
T5_REPOSITORY = "google-t5/t5-large"
T5_REVISION = "150ebc2c4b72291e770f58e6057481c8d2ed331a"
T5_SOURCE_NAME = "t5-large"
T5_WEIGHT_PATH = "model.safetensors"
T5_WEIGHT_BYTES = 2_950_736_730
T5_WEIGHT_SHA256 = "bb566a699a6939ffa2ac2fa7eb26fd02b13d4d14956186d7af1e46ce19c67d32"
T5_METADATA_SCHEMA = "vokra-audiogen-medium-t5-server-metadata-v1"
T5_METADATA_FILES: dict[str, dict[str, Any]] = {
    "config.json": {
        "bytes": 1_209,
        "git_blob_sha1": "e4be09f2bfa7ed03596de9bdad4ddc0c7bdff4c1",
        "lfs_pointer_git_blob_sha1": None,
        "lfs_payload_sha256": None,
        "lfs_payload_size": None,
    },
    "spiece.model": {
        "bytes": 791_656,
        "git_blob_sha1": "4e28ff6ebdf584f5372d9de68867399142435d9a",
        "lfs_pointer_git_blob_sha1": None,
        "lfs_payload_sha256": None,
        "lfs_payload_size": None,
    },
    T5_WEIGHT_PATH: {
        "bytes": T5_WEIGHT_BYTES,
        "git_blob_sha1": None,
        "lfs_pointer_git_blob_sha1": "af02eb3a7cce91694f5e24d82daec0c6aed59337",
        "lfs_payload_sha256": T5_WEIGHT_SHA256,
        "lfs_payload_size": T5_WEIGHT_BYTES,
    },
}

# The values below are the canonical T5-large config axes.  AudioGen uses the
# encoder only; decoder tensors in the upstream full T5 checkpoint are not a
# license to bind an arbitrary alternate T5 variant.
T5_CONFIG: dict[str, Any] = {
    "model_type": "t5",
    "architectures": ["T5ForConditionalGeneration"],
    "encoder_consumer": "T5EncoderModel",
    "vocab_size": 32_128,
    "d_model": 1_024,
    "d_kv": 64,
    "d_ff": 4_096,
    "num_layers": 24,
    "num_heads": 16,
    "n_positions": 512,
    "relative_attention_num_buckets": 32,
    "layer_norm_epsilon": 1.0e-6,
}

T5_REQUIRED_FILES: tuple[dict[str, Any], ...] = (
    {"path": "config.json", "role": "config", "payload": "REQUIRED"},
    {"path": "spiece.model", "role": "sentencepiece_tokenizer", "payload": "REQUIRED"},
    {"path": T5_WEIGHT_PATH, "role": "full_t5_weights_for_encoder_extract", "payload": "REQUIRED"},
)

# AudioGen's official 16-kHz EnCodec is not the public 24-kHz Meta EnCodec
# model.  The model card contains its release-specific compression payload;
# the AudioCraft solver config points at an internal training checkpoint that
# is not a separate public repository.  This distinction prevents a silent
# substitution of ``facebook/encodec_24khz`` or MusicGen's 32-kHz companion.
ENCODEC_COMPANION: dict[str, Any] = {
    "repository": AUDIOGEN_REPOSITORY,
    "revision": AUDIOGEN_REVISION,
    "path": "compression_state_dict.bin",
    "bytes": 235_740_815,
    "sha256": "5a520e64ca99226a9956f83b06df0617b713183fcdc384779883a6bb46dc1095",
    "source_repository": AUDIOCRAFT_REPOSITORY,
    "source_revision": AUDIOCRAFT_REVISION,
    "source_config": "config/solver/audiogen/audiogen_base_16khz.yaml",
    "internal_checkpoint_reference": "//reference/bd44a852/checkpoint.th",
    "internal_checkpoint_public": False,
    "config": {
        "sample_rate_hz": 16_000,
        "channels": 1,
        "total_stride": 320,
        "frame_rate_hz": 50,
        "num_codebooks": 4,
        "codebook_size": 2_048,
        "latent_dimension": 128,
        "n_filters": 64,
        "ratios": [8, 5, 4, 2],
        "quantizer_dropout": False,
    },
    "weight_topology": {
        "format": "AudioCraft compression state_dict",
        "components": ["SEANet encoder", "RVQ quantizer", "SEANet decoder"],
        "tensor_manifest": "PENDING_PAYLOAD_AUTHENTICATION",
    },
    "payload": "NOT_DOWNLOADED",
}


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def t5_server_metadata_packet() -> dict[str, Any]:
    """Canonical metadata-only packet expected from the HF API audit."""

    return {
        "schema": T5_METADATA_SCHEMA,
        "repository": T5_REPOSITORY,
        "requested_revision": T5_REVISION,
        "resolved_revision": T5_REVISION,
        "license": {"card_data": "apache-2.0", "status": "AUTHENTICATED"},
        "files": [{"path": name, "type": "file", **_copy_json(T5_METADATA_FILES[name])} for name in sorted(T5_METADATA_FILES)],
        "payload": "NOT_DOWNLOADED",
    }


def contract() -> dict[str, Any]:
    """Return a JSON-safe immutable-by-convention contract snapshot."""

    return {
        "schema": SCHEMA,
        "audio_generation": {
            "repository": AUDIOGEN_REPOSITORY,
            "revision": AUDIOGEN_REVISION,
            "source_repository": AUDIOCRAFT_REPOSITORY,
            "source_revision": AUDIOCRAFT_REVISION,
            "source_tag": AUDIOCRAFT_TAG,
        },
        "text_conditioner": {
            "source_name": T5_SOURCE_NAME,
            "repository": T5_REPOSITORY,
            "revision": T5_REVISION,
            "license": "Apache-2.0",
            "config": _copy_json(T5_CONFIG),
            "required_files": _copy_json(T5_REQUIRED_FILES),
            "server_metadata": t5_server_metadata_packet(),
            "weight": {
                "path": T5_WEIGHT_PATH,
                "bytes": T5_WEIGHT_BYTES,
                "sha256": T5_WEIGHT_SHA256,
            },
            "historical_linkage": "UNVERIFIED_SOURCE_NAME_HAS_NO_REVISION",
            "payload": "NOT_DOWNLOADED",
        },
        "compression_companion": _copy_json(ENCODEC_COMPANION),
    }


def canonical_sha256() -> str:
    """SHA-256 of the canonical JSON contract (for evidence binding)."""

    encoded = json.dumps(contract(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(value: dict[str, Any]) -> None:
    """Reject identity, topology, and payload-state drift fail-closed."""

    if value != contract():
        raise ValueError("AudioGen companion contract drift")
    text = value["text_conditioner"]
    if not re.fullmatch(r"[0-9a-f]{40}", text["revision"]):
        raise ValueError("T5 companion revision is not immutable")
    if text["payload"] != "NOT_DOWNLOADED":
        raise ValueError("T5 model-free contract crossed the payload boundary")
    weight = text["weight"]
    if weight["path"] != T5_WEIGHT_PATH or weight["bytes"] != T5_WEIGHT_BYTES or not re.fullmatch(r"[0-9a-f]{64}", weight["sha256"]):
        raise ValueError("T5 weight identity is incomplete")
    codec = value["compression_companion"]
    if codec["payload"] != "NOT_DOWNLOADED" or codec["internal_checkpoint_public"] is not False:
        raise ValueError("EnCodec model-free contract crossed the payload boundary")
    if codec["repository"] != AUDIOGEN_REPOSITORY or codec["revision"] != AUDIOGEN_REVISION:
        raise ValueError("EnCodec companion is not bound to the release repository/revision")
    if codec["config"] != ENCODEC_COMPANION["config"]:
        raise ValueError("EnCodec config drift")
    if codec["weight_topology"]["tensor_manifest"] != "PENDING_PAYLOAD_AUTHENTICATION":
        raise ValueError("unverified EnCodec tensor topology was accepted")


def self_test() -> None:
    value = contract()
    validate(value)
    assert len(canonical_sha256()) == 64
    tampered = _copy_json(value)
    tampered["text_conditioner"]["revision"] = "0" * 40
    try:
        validate(tampered)
    except ValueError as error:
        assert str(error) == "AudioGen companion contract drift"
    else:
        raise AssertionError("T5 revision drift was accepted")
    tampered = _copy_json(value)
    tampered["compression_companion"]["config"]["sample_rate_hz"] = 32_000
    try:
        validate(tampered)
    except ValueError as error:
        assert str(error) == "AudioGen companion contract drift"
    else:
        raise AssertionError("32-kHz codec substitution was accepted")
    tampered = _copy_json(value)
    tampered["compression_companion"]["payload"] = "DOWNLOADED"
    try:
        validate(tampered)
    except ValueError as error:
        assert str(error) == "AudioGen companion contract drift"
    else:
        raise AssertionError("payload boundary drift was accepted")


if __name__ == "__main__":
    self_test()
    print("audiogen_medium_companion_contract --self-test: OK")
