#!/usr/bin/env -S uv run --frozen --project tools/parity/firered_asr_aed_l --python 3.12 python
"""VAST-only independent FireRedASR-AED-L reference capture.

This script imports the pinned upstream FireRed implementation directly.  It
does not mirror the model in Vokra and never runs on the maintainer machine.
It records the checkpoint's exact state-dict-to-module mapping, then captures
frontend/encoder/decoder observations for deterministic synthetic int16 PCM.
Any missing upstream dependency or source/API drift is a hard failure; no
PASS/parity claim is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "FireRedTeam/FireRedASR-AED-L"
MODEL_REVISION = "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
SOURCE_REVISION = "834635e4cf277ed8ca92049fc375b17c3dc20748"
CHECKPOINT_BYTES = 4_678_597_714
CHECKPOINT_SHA256 = "12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3"
EXPECTED_TENSORS = 940
KALDI_NATIVE_FBANK_SOURCE = {
    "repository": "https://github.com/csukuangfj/kaldi-native-fbank.git",
    "revision": "f68c6b43f739697d7ab02ff6debacee130e1d541",
    "version": "1.15",
    "license": "Apache-2.0",
    "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
EXPECTED_ARGS = {
    "idim": 80,
    "odim": 7832,
    "n_layers_enc": 16,
    "n_layers_dec": 16,
    "d_model": 1280,
    "n_head": 20,
    "kernel_size": 33,
    "sos_id": 3,
    "eos_id": 4,
    "pad_id": 2,
}


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guard_output_path(output: Path, *inputs: Path) -> None:
    """Reject aliases and pre-existing targets before importing upstream code."""
    if output.is_symlink() or output.exists():
        raise RuntimeError(f"refusing to overwrite reference output: {output}")
    if output.parent.is_symlink():
        raise RuntimeError(f"reference output path ancestor is a symlink: {output.parent}")
    normalized = output.resolve(strict=False)
    for input_path in inputs:
        if normalized == input_path.resolve(strict=False):
            raise RuntimeError(f"reference output aliases input: {output}")


def publish_json_no_clobber(path: Path, value: Any) -> None:
    """Durably publish JSON with a same-directory, no-clobber link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise RuntimeError(f"refusing to overwrite reference output: {path}")
    if path.parent.is_symlink():
        raise RuntimeError(f"reference output path ancestor is a symlink: {path.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


SOURCE_ROLES = {
    "fireredasr/data/asr_feat.py": "frontend_and_cmvn",
    "fireredasr/models/fireredasr_aed.py": "aed_model_wiring",
    "fireredasr/models/module/conformer_encoder.py": "conformer_encoder",
    "fireredasr/models/module/transformer_decoder.py": "aed_decoder_and_beam_search",
    "fireredasr/tokenizer/aed_tokenizer.py": "sentencepiece_token_dict_mapping",
    "fireredasr/data/token_dict.py": "token_dictionary",
    "README.md": "release_input_contract",
}


# These are deliberately small, literal anchors from the pinned upstream
# files.  A source hash alone authenticates bytes but does not make the role
# assignment auditable to a later consumer; checking the anchors here keeps
# the reference archive tied to the semantics it claims to tap.  Keep this
# list restricted to facts already established by the pinned source contract.
SOURCE_MARKERS = {
    "fireredasr/data/asr_feat.py": (
        "class CMVN:",
        "stats = kaldiio.load_mat(kaldi_cmvn_file)",
        "KaldifeatFbank(num_mel_bins=80, frame_length=25,",
        "frame_shift=10, dither=0.0",
        "sample_rate, wav_np = kaldiio.load_mat(",
        "fbank.accept_waveform(sample_rate, wav_np.tolist())",
    ),
    "fireredasr/models/fireredasr_aed.py": (
        "self.sos_id = args.sos_id",
        "self.eos_id = args.eos_id",
        "self.decoder = TransformerDecoder(",
    ),
    "fireredasr/models/module/conformer_encoder.py": (
        "self.input_preprocessor = Conv2dSubsampling(idim, d_model)",
        "self.positional_encoding = RelPositionalEncoding(d_model)",
        "self.mhsa = RelPosMultiHeadAttention(",
        "nn.Conv2d(1, out_channels, 3, 2)",
        "nn.Conv2d(out_channels, out_channels, 3, 2)",
        "subsample_idim = ((idim - 1) // 2 - 1) // 2",
        "def _rel_shift(self, x):",
    ),
    "fireredasr/models/module/transformer_decoder.py": (
        "self.tgt_word_prj.weight = self.tgt_word_emb.weight",
        "t_logit = self.tgt_word_prj(dec_output[:, -1])",
        "t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)",
        "cache=caches[i]",
    ),
    "fireredasr/tokenizer/aed_tokenizer.py": (
        "spm.SentencePieceProcessor()",
        "self.sp.EncodeAsPieces(part.strip())",
        "self.dict.get(token, self.dict.unk)",
        "tokens = [self.dict[id] for id in inputs]",
        "s.replace(self.SPM_SPACE, ' ').strip()",
    ),
    "fireredasr/data/token_dict.py": (
        "class TokenDict:",
    ),
    "README.md": (
        "ffmpeg -i input_audio -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav",
    ),
}


def source_records(source_root: Path) -> list[dict[str, Any]]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"pinned upstream source is not a git checkout: {error}") from error
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"pinned upstream source revision mismatch: {revision!r} != {SOURCE_REVISION!r}")
    records = []
    for relative, role in SOURCE_ROLES.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required pinned source role is missing or non-regular: {relative}")
        text = path.read_text(encoding="utf-8")
        markers = SOURCE_MARKERS[relative]
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(f"pinned source markers missing in {relative}: {missing!r}")
        records.append(
            {
                "path": relative,
                "role": role,
                "sha256": sha256_bytes(path.read_bytes()),
                "markers": list(markers),
            }
        )
    return records


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "sha256": sha256_bytes(raw),
    }


def tensor_values(tensor: Any) -> list[Any]:
    return tensor.detach().to(device="cpu").contiguous().tolist()


def load_upstream(source_root: Path, checkpoint: Path) -> tuple[Any, Any, dict[str, Any]]:
    import argparse as argparse_module
    import torch

    sys.path.insert(0, str(source_root))
    try:
        from fireredasr.models.fireredasr_aed import FireRedAsrAed
    except Exception as error:  # pragma: no cover - exercised only on VAST
        raise RuntimeError(
            "pinned upstream FireRed import failed; requirements must provide "
            "kaldiio==2.18.0 and kaldi-native-fbank==1.15: " + str(error)
        ) from error

    unsafe = list(torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint)))
    if unsafe != ["argparse.Namespace"]:
        raise RuntimeError(f"unexpected checkpoint unsafe globals: {unsafe!r}")
    with torch.serialization.safe_globals([argparse_module.Namespace]):
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or set(package) != {"args", "model_state_dict"}:
        raise RuntimeError("upstream checkpoint envelope is not exactly args/model_state_dict")
    args = package["args"]
    state_dict = package["model_state_dict"]
    if not isinstance(args, argparse_module.Namespace) or not isinstance(state_dict, dict):
        raise RuntimeError("upstream checkpoint args/state_dict types are not authenticated")
    observed_args = {name: getattr(args, name, None) for name in EXPECTED_ARGS}
    for name, expected in EXPECTED_ARGS.items():
        if observed_args[name] != expected:
            raise RuntimeError(f"checkpoint args.{name} mismatch: {observed_args[name]!r} != {expected!r}")
    model = FireRedAsrAed.from_args(args).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict upstream state load mismatch: missing={missing}, unexpected={unexpected}")
    return model, args, state_dict


def field_mapping(model: Any, state_dict: dict[str, Any]) -> list[dict[str, Any]]:
    # The pinned decoder intentionally ties ``tgt_word_prj.weight`` to the
    # embedding value.  The checkpoint stores the two entries as distinct
    # payloads, while PyTorch's default ``named_parameters()`` suppresses
    # duplicate objects.  Keep duplicate names visible so strict state-dict
    # accounting covers both authenticated fields.
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    rows: list[dict[str, Any]] = []
    for name, tensor in state_dict.items():
        if name in parameters:
            role = "parameter"
        elif name in buffers:
            role = "buffer"
        else:
            raise RuntimeError(f"state-dict tensor is not exposed by upstream named modules: {name}")
        module_value = parameters[name] if role == "parameter" else buffers[name]
        summary = tensor_summary(tensor)
        module_summary = tensor_summary(module_value)
        if summary != module_summary:
            raise RuntimeError(f"upstream module binding changed tensor {name}")
        rows.append({"name": name, "role": role, **summary})
    if len(rows) != EXPECTED_TENSORS or len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError(f"upstream tensor mapping count/uniqueness mismatch: {len(rows)}")
    return rows


def capture_reference(model: Any, args: Any, source_root: Path, cmvn_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(0, str(source_root))
    try:
        from fireredasr.data.asr_feat import ASRFeatExtractor
    except Exception as error:  # pragma: no cover - exercised only on VAST
        raise RuntimeError(
            "pinned upstream frontend import failed; requirements must provide "
            "kaldiio==2.18.0 and kaldi-native-fbank==1.15: " + str(error)
        ) from error

    # Fixed nonzero int16-range PCM; no external audio file is acquired.
    samples = np.asarray(
        [int(12000 * np.sin(index * 0.017) + 3000 * np.cos(index * 0.031)) for index in range(16000)],
        dtype=np.int16,
    )
    extractor = ASRFeatExtractor(str(cmvn_path))
    fbank = extractor.fbank((16000, samples))
    if extractor.cmvn is not None:
        fbank = extractor.cmvn(fbank)
    features = torch.from_numpy(np.asarray(fbank)).float().unsqueeze(0)
    lengths = torch.tensor([features.shape[1]], dtype=torch.long)

    taps: dict[str, Any] = {}

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            values = output[0] if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor) else output
            if isinstance(values, torch.Tensor):
                taps[name] = {"summary": tensor_summary(values), "values": tensor_values(values)}

        return hook

    handles = [
        model.encoder.register_forward_hook(capture("encoder")),
        # ``batch_beam_search`` is an ordinary method and is called directly
        # below; it does not pass through ``model.decoder.__call__``.  A
        # module-level decoder hook would therefore never observe logits.
        # Hook the projection that the pinned upstream method invokes at
        # transformer_decoder.py's ``t_logit = self.tgt_word_prj(...)``.
        model.decoder.tgt_word_prj.register_forward_hook(capture("decoder_logits")),
    ]
    try:
        with torch.no_grad():
            enc_outputs, _, enc_mask = model.encoder(features, lengths)
            hypotheses = model.decoder.batch_beam_search(
                enc_outputs, enc_mask, 1, 1, 32, 1.0, 0.0, 1.0
            )
    finally:
        for handle in handles:
            handle.remove()
    if not hypotheses:
        raise RuntimeError("upstream decoder returned no hypothesis")
    first = hypotheses[0][0]
    token_ids = first.get("yseq") if isinstance(first, dict) else getattr(first, "yseq", None)
    if token_ids is None:
        raise RuntimeError(f"upstream hypothesis has no yseq: {type(first).__name__}")
    if isinstance(token_ids, torch.Tensor):
        token_ids = [int(value) for value in token_ids.detach().cpu().tolist()]
    else:
        token_ids = [int(value) for value in token_ids]
    return {
        "pcm": {"sample_rate": 16000, "samples": len(samples), "dtype": "int16"},
        "frontend": {"shape": list(np.asarray(fbank).shape), "dtype": str(np.asarray(fbank).dtype), "sha256": sha256_bytes(np.asarray(fbank, dtype=np.float32).tobytes())},
        "encoder": taps.get("encoder"),
        "decoder_logits": taps.get("decoder_logits"),
        "greedy": {"beam_size": 1, "nbest": 1, "decode_max_len": 32, "softmax_smoothing": 1.0, "length_penalty": 0.0, "eos_penalty": 1.0, "token_ids": token_ids},
        "status": "REFERENCE_CAPTURED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cmvn", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.source or not args.checkpoint or not args.cmvn or not args.output:
        parser.error("--source, --checkpoint, --cmvn and --output are required")
    guard_output_path(args.output, args.checkpoint, args.cmvn)
    model, checkpoint_args, state_dict = load_upstream(args.source, args.checkpoint)
    result = {
        "format": "vokra-firered-asr-aed-l-upstream-reference-v1",
        "status": "REFERENCE_CAPTURED",
        "publication": "NO_UPLOAD",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "checkpoint": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "bytes": CHECKPOINT_BYTES,
            "sha256": CHECKPOINT_SHA256,
        },
        "source": {
            "revision": SOURCE_REVISION,
            "path": str(args.source),
            "records": source_records(args.source),
        },
        "dependencies": {
            "python": "3.12",
            "kaldiio": {"version": "2.18.0", "source": "pypi"},
            "kaldi-native-fbank": KALDI_NATIVE_FBANK_SOURCE,
        },
        "args": {name: getattr(checkpoint_args, name) for name in EXPECTED_ARGS},
        "tensor_mapping": field_mapping(model, state_dict),
        "reference": capture_reference(model, checkpoint_args, args.source, args.cmvn),
        "parity": {"status": "NOT_RUN", "fp32_atol": 0.01},
    }
    publish_json_no_clobber(args.output, result)
    return 0


def self_test() -> None:
    """Exercise only the manifest constants and duplicate-key guard."""
    assert MODEL_REPOSITORY == "FireRedTeam/FireRedASR-AED-L"
    assert len(SOURCE_ROLES) == 7
    assert set(SOURCE_MARKERS) == set(SOURCE_ROLES)
    assert all(SOURCE_MARKERS[path] for path in SOURCE_ROLES)
    assert EXPECTED_TENSORS == 940
    assert KALDI_NATIVE_FBANK_SOURCE["version"] == "1.15"
    assert len(KALDI_NATIVE_FBANK_SOURCE["license_sha256"]) == 64
    assert len(CHECKPOINT_SHA256) == 64
    with tempfile.TemporaryDirectory(prefix="firered-reference-") as directory:
        root = Path(directory)
        output = root / "reference.json"
        publish_json_no_clobber(output, {"status": "synthetic"})
        assert output.is_file() and not list(root.glob("*.tmp"))
        try:
            publish_json_no_clobber(output, {"status": "sentinel"})
        except RuntimeError:
            pass
        else:
            raise AssertionError("reference output clobber accepted")
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "synthetic"
        try:
            guard_output_path(output, root / "input.bin")
        except RuntimeError:
            pass
        else:
            raise AssertionError("existing reference output accepted")
    try:
        reject_duplicate_pairs([("x", 1), ("x", 2)])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    assert reject_duplicate_pairs([("x", 1)]) == {"x": 1}
    print("firered upstream reference self-test PASS")


if __name__ == "__main__":
    raise SystemExit(main())
