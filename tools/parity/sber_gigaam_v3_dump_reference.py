#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Dump the fixed official GigaAM v3 RNNT reference packet.

This tool is intended for the remote VAST leg only.  It loads the pinned
Hugging Face remote-code model and records raw frontend/encoder/joint output;
it does not convert weights and does not claim parity.  The runtime wrapper's
``model.model.head`` boundary is checked explicitly so a local mirror cannot
silently substitute the model implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

HF_REPOSITORY = "ai-sage/GigaAM-v3"
HF_REVISION = "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e"
SOURCE_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
CONFIG_SHA256 = "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52"
MODELING_SHA256 = "269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86"
CHECKPOINT_BYTES = 448_928_167
CHECKPOINT_SHA256 = "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a"
TOKENIZER_SHA256 = "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a"
SAMPLE_RATE_HZ = 16_000
PCM_SAMPLES = 16_000
PCM_F32LE_SHA256 = "f92e4a0422c513ab107975f5c9bd7a8e7a92532b37508a769c92d2496625229b"
NUM_CLASSES = 1_025
BLANK_ID = 1_024
MAX_SYMBOLS_PER_STEP = 10


def stream_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_fixed_config(config: object) -> None:
    """Validate the exact nested config emitted by the official checkpoint."""

    def mapping(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"fixed config {label} must be an object")
        return value

    root = mapping(config, "root")
    outer = mapping(root.get("cfg"), "cfg")
    model = mapping(outer.get("model"), "cfg.model")
    inner = mapping(model.get("cfg"), "cfg.model.cfg")
    model_class = inner.get("model_class")
    sample_rate = inner.get("sample_rate")
    if not isinstance(model_class, str) or model_class != "rnnt":
        raise ValueError("fixed config RNNT model_class mismatch")
    if type(sample_rate) is not int or sample_rate != SAMPLE_RATE_HZ:
        raise ValueError("fixed config sample_rate mismatch")

    preprocessor = mapping(inner.get("preprocessor"), "cfg.model.cfg.preprocessor")
    expected = {
        "sample_rate": SAMPLE_RATE_HZ,
        "center": False,
        "mel_scale": "htk",
        "mel_norm": None,
        "n_fft": 320,
        "win_length": 320,
        "hop_length": 160,
        "features": 64,
    }
    for name, expected_value in expected.items():
        actual = preprocessor.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(f"fixed preprocessor {name} mismatch")


def reject_symlink_ancestry(path: Path, label: str) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise SystemExit(f"{label} has symlink ancestry: {ancestor}")


def write_raw(path: Path, values, dtype: str) -> dict[str, object]:
    import numpy as np

    array = np.asarray(values, dtype=dtype, order="C")
    raw = array.tobytes(order="C")
    path.write_bytes(raw)
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(array.shape),
        "dtype": "float32" if dtype == "<f4" else "uint32",
    }


def fixed_pcm():
    import numpy as np

    samples = np.arange(PCM_SAMPLES, dtype=np.int32)
    return ((samples % 97) - 48).astype(np.float32) / np.float32(48.0)


def decode_greedy(frames: list[list[list[float]]]) -> tuple[list[int], list[int]]:
    tokens: list[int] = []
    argmaxes: list[int] = []
    for frame in frames:
        emitted = 0
        for row in frame[:MAX_SYMBOLS_PER_STEP]:
            if not all(float(value) == float(value) for value in row):
                raise SystemExit("official joint produced NaN")
            value = max(row)
            index = row.index(value)
            argmaxes.append(index)
            if index == BLANK_ID:
                break
            tokens.append(index)
            emitted += 1
    return tokens, argmaxes


def validate_decision_lengths(logits, frames, symbols, argmax) -> None:
    if len(logits) != len(frames) or len(frames) != len(symbols) or len(symbols) != len(argmax):
        raise ValueError("RNNT decision trace lengths disagree")


def load_official_model(auto_model, model_dir: Path):
    """Load the pinned remote-code model with CPU-safe Transformers 5 init.

    Transformers 5.16 unconditionally enters a ``meta`` device context while
    constructing custom models.  The pinned official FeatureExtractor creates
    torchaudio filter-bank buffers in its constructor, which must be allocated
    on CPU.  Appending an explicit CPU context preserves the official
    ``AutoModel.from_pretrained`` loader and still lets it load every weight.
    The temporary legacy tied-weight shim is needed because this official
    remote class predates the renamed Transformers 5.16 attribute; v3 has no
    tied parameters, so its empty mapping is semantically inert.
    """
    import torch
    from transformers.modeling_utils import PreTrainedModel

    original_context = PreTrainedModel.__dict__["get_init_context"]
    original_tied = PreTrainedModel.__dict__.get("all_tied_weights_keys")
    had_tied = "all_tied_weights_keys" in PreTrainedModel.__dict__
    add_tied = not hasattr(PreTrainedModel, "all_tied_weights_keys")

    def cpu_safe_init_context(cls, dtype, is_quantized, is_ds_init_called, allow_all_kernels):
        contexts = original_context.__func__(
            cls, dtype, is_quantized, is_ds_init_called, allow_all_kernels
        )
        contexts.append(torch.device("cpu"))
        return contexts

    PreTrainedModel.get_init_context = classmethod(cpu_safe_init_context)
    if add_tied:
        PreTrainedModel.all_tied_weights_keys = property(
            lambda self: getattr(self, "_tied_weights_keys", {}) or {}
        )
    try:
        return auto_model.from_pretrained(
            model_dir,
            revision=HF_REVISION,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=False,
        )
    finally:
        PreTrainedModel.get_init_context = original_context
        if had_tied:
            PreTrainedModel.all_tied_weights_keys = original_tied
        elif add_tied:
            del PreTrainedModel.all_tied_weights_keys


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert source.count("            decision_frames.append(frame)\n") == 1
    assert "decision_argmax.u32le" in source
    assert "contexts.append(torch.device(\"cpu\"))" in source
    assert "low_cpu_mem_usage=False" in source
    assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40
    assert len(CONFIG_SHA256) == len(MODELING_SHA256) == len(CHECKPOINT_SHA256) == 64
    assert BLANK_ID == NUM_CLASSES - 1 and MAX_SYMBOLS_PER_STEP == 10
    assert len(fixed_pcm()) == PCM_SAMPLES
    assert hashlib.sha256(fixed_pcm().astype("<f4").tobytes()).hexdigest() == PCM_F32LE_SHA256
    nested_config = {
        "cfg": {
            "model": {
                "cfg": {
                    "model_class": "rnnt",
                    "sample_rate": SAMPLE_RATE_HZ,
                    "preprocessor": {
                        "sample_rate": SAMPLE_RATE_HZ,
                        "center": False,
                        "mel_scale": "htk",
                        "mel_norm": None,
                        "n_fft": 320,
                        "win_length": 320,
                        "hop_length": 160,
                        "features": 64,
                    },
                }
            }
        }
    }
    validate_fixed_config(nested_config)
    for malformed in (
        {
            "model_class": "rnnt",
            "sample_rate": SAMPLE_RATE_HZ,
            "preprocessor": nested_config["cfg"]["model"]["cfg"]["preprocessor"],
        },
        {"cfg": {"model": {"model_class": "rnnt"}}},
    ):
        try:
            validate_fixed_config(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong or missing config nesting was accepted")
    row = [0.0] * NUM_CLASSES
    row[3] = 2.0
    row[BLANK_ID] = 3.0
    assert decode_greedy([[row]]) == ([], [BLANK_ID])
    validate_decision_lengths([[row]], [0], [0], [BLANK_ID])
    try:
        validate_decision_lengths([[row]], [0, 1], [0], [BLANK_ID])
    except ValueError:
        pass
    else:
        raise AssertionError("decision length tamper was accepted")
    print("sber_gigaam_v3_dump_reference self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.model_dir or args.output:
            parser.error("--self-test accepts no paths")
        self_test()
        return 0
    if args.model_dir is None or args.output is None:
        parser.error("--model-dir and --output are required")
    reject_symlink_ancestry(args.model_dir, "model directory")
    reject_symlink_ancestry(args.output, "reference output")
    if not args.model_dir.is_dir() or args.model_dir.is_symlink():
        raise SystemExit("official model directory is missing or symlinked")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("reference output must be absent and non-symlink")
    model_real = args.model_dir.resolve()
    output_real = args.output.resolve()
    if output_real == model_real or model_real in output_real.parents or output_real in model_real.parents:
        raise SystemExit("reference output must be disjoint from the model snapshot")
    paths = {
        "config": args.model_dir / "config.json",
        "modeling_gigaam": args.model_dir / "modeling_gigaam.py",
        "checkpoint": args.model_dir / "pytorch_model.bin",
        "tokenizer": args.model_dir / "tokenizer.model",
    }
    for name, path in paths.items():
        reject_symlink_ancestry(path, name)
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"official {name} is missing or symlinked")
    expected = {
        "config": CONFIG_SHA256,
        "modeling_gigaam": MODELING_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
        "tokenizer": TOKENIZER_SHA256,
    }
    source_files: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        digest, size = stream_sha256(path)
        if digest != expected[name]:
            raise SystemExit(f"{name} SHA-256 mismatch")
        if name == "checkpoint" and size != CHECKPOINT_BYTES:
            raise SystemExit("checkpoint byte-size mismatch")
        source_files[name] = {"path": str(path), "bytes": size, "sha256": digest}
    try:
        config = json.loads(paths["config"].read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid fixed config: {exc}") from exc
    try:
        validate_fixed_config(config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel

    pcm = fixed_pcm()
    if hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest() != PCM_F32LE_SHA256:
        raise SystemExit("fixed PCM digest mismatch")
    model = load_official_model(AutoModel, args.model_dir).eval()
    inner = getattr(model, "model", None)
    head = getattr(inner, "head", None)
    if type(model).__name__ != "GigaAMModel" or inner is None or head is None:
        raise SystemExit("official GigaAMModel wrapper/inner head contract is missing")
    if not type(inner).__module__.endswith("modeling_gigaam"):
        raise SystemExit("inner model is not from pinned modeling_gigaam.py")
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    lengths = torch.tensor([PCM_SAMPLES], dtype=torch.long)
    with torch.inference_mode():
        features, feature_lengths = inner.preprocessor(waveform, lengths)
        encoded, encoded_lengths = model.forward(waveform, lengths)
    mel = np.asarray(features[0].transpose(0, 1).cpu().numpy(), dtype=np.float32)
    encoded_np = np.asarray(encoded[0].transpose(0, 1).cpu().numpy(), dtype=np.float32)
    frame_count = int(encoded_lengths[0])
    if feature_lengths.shape != (1,) or int(feature_lengths[0]) <= 0:
        raise SystemExit("official frontend frame count is invalid")
    if encoded_np.shape != (frame_count, 768) or mel.ndim != 2 or not np.isfinite(mel).all():
        raise SystemExit("official frontend/encoder shape or finite contract mismatch")
    decoder = getattr(head, "decoder", None)
    joint = getattr(head, "joint", None)
    if decoder is None or joint is None:
        raise SystemExit("official RNNT decoder/joint modules are missing")
    decision_rows: list[list[float]] = []
    decision_argmax: list[int] = []
    decision_frames: list[int] = []
    decision_symbols: list[int] = []
    token_ids: list[int] = []
    state = None
    label = None
    for frame in range(frame_count):
        symbols = 0
        while symbols < MAX_SYMBOLS_PER_STEP:
            prediction, next_state = decoder.predict(label, state)
            enc_frame = encoded[0, :, frame].reshape(1, 1, -1)
            with torch.inference_mode():
                output = joint(enc_frame, prediction)
            row = np.asarray(output.reshape(-1).cpu().numpy(), dtype=np.float32)
            if row.shape != (NUM_CLASSES,) or not np.isfinite(row).all():
                raise SystemExit("official RNNT joint row shape/finite contract mismatch")
            index = int(np.argmax(row))
            decision_rows.append(row.tolist())
            decision_argmax.append(index)
            decision_frames.append(frame)
            decision_symbols.append(symbols)
            if index == BLANK_ID:
                break
            token_ids.append(index)
            state = next_state
            label = torch.tensor([[index]], dtype=torch.long)
            symbols += 1
    validate_decision_lengths(decision_rows, decision_frames, decision_symbols, decision_argmax)
    args.output.mkdir(parents=True)
    artifacts = {
        "pcm": write_raw(args.output / "pcm.f32le", pcm, "<f4"),
        "log_mel": write_raw(args.output / "log_mel.f32le", mel, "<f4"),
        "encoded": write_raw(args.output / "encoded.f32le", encoded_np, "<f4"),
        "rnnt_logits": write_raw(args.output / "rnnt_logits.f32le", np.asarray(decision_rows, dtype="<f4"), "<f4"),
        "decision_frames": write_raw(args.output / "decision_frames.u32le", np.asarray(decision_frames, dtype="<u4"), "<u4"),
        "decision_symbols": write_raw(args.output / "decision_symbols.u32le", np.asarray(decision_symbols, dtype="<u4"), "<u4"),
        "decision_argmax": write_raw(args.output / "decision_argmax.u32le", np.asarray(decision_argmax, dtype="<u4"), "<u4"),
        "token_ids": write_raw(args.output / "token_ids.u32le", np.asarray(token_ids, dtype="<u4"), "<u4"),
    }
    manifest = {
        "format": "vokra-gigaam-v3-reference-v1",
        "status": "REFERENCE_DUMP_OPEN_NOT_PARITY",
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "source_revision": SOURCE_REVISION,
        "config_sha256": CONFIG_SHA256,
        "modeling_gigaam_sha256": MODELING_SHA256,
        "source_files": source_files,
        "pcm_input": {"path": "pcm.f32le", "sample_rate_hz": SAMPLE_RATE_HZ, "shape": [PCM_SAMPLES], "dtype": "float32", "f32le_sha256": PCM_F32LE_SHA256},
        "artifacts": artifacts,
        "mel_frames": int(feature_lengths[0]),
        "encoded_frames": frame_count,
        # RNNTJoint.forward in the pinned modeling_gigaam.py applies
        # joint_net(...).log_softmax(-1); this artifact therefore preserves
        # the official log-probability output, not an assumed raw affine row.
        "rnnt": {"num_classes": NUM_CLASSES, "blank_id": BLANK_ID, "max_symbols_per_step": MAX_SYMBOLS_PER_STEP, "decode": "greedy", "joint_output": "log_softmax"},
        "frontend": {"center": False, "mel_scale": "htk", "mel_norm": None, "power": 2, "n_fft": 320, "win_length": 320, "hop_length": 160, "n_mels": 64},
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "official_import": "AutoModel.from_pretrained(trust_remote_code=True)"},
        "parity": "OPEN_MEASURED_NOT_GATED",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote official GigaAM v3 raw reference artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
