#!/usr/bin/env python3
"""Generate FSMN-VAD parity data by executing pinned FunASR source.

This is an independent oracle: it loads `encoder.py` and `wav_frontend.py`
directly from the pinned FunASR checkout, instantiates the official classes,
and runs the official checkpoint.  It does not mirror the Rust equations.

Run with uv-managed Python 3.12, for example:

    uv run --no-project --python 3.12 --with torch --with torchaudio \
      --with numpy python tools/parity/fsmn_vad_reference.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

FUNASR_REVISION = "3c58cb56a56598232c3efffa15d313d7e82a4307"
MODEL_REVISION = "df20e6b30c653645fa4ff125cacfcabd1020a669"
MODEL_SHA256 = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5"
CMVN_SHA256 = "df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise SystemExit(f"{label}: got {actual}, expected {expected}")


class _Tables:
    @staticmethod
    def register(*_args, **_kwargs):
        return lambda value: value


def install_import_stubs() -> None:
    """Stub registration-only imports without replacing the SUT classes."""

    funasr = types.ModuleType("funasr")
    funasr.__path__ = []
    register = types.ModuleType("funasr.register")
    register.tables = _Tables()
    frontends = types.ModuleType("funasr.frontends")
    frontends.__path__ = []
    eend = types.ModuleType("funasr.frontends.eend_ola_feature")
    sys.modules.update(
        {
            "funasr": funasr,
            "funasr.register": register,
            "funasr.frontends": frontends,
            "funasr.frontends.eend_ola_feature": eend,
        }
    )


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load official source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_state_dict(checkpoint):
    for key in ("state_dict", "model_state_dict", "model", "module"):
        candidate = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(candidate, dict) and candidate:
            return candidate
    return checkpoint


def deterministic_pcm_i16(length: int) -> list[int]:
    # Integer-only stimulus: Rust consumes the exact same samples with no
    # cross-language libm drift before frontend comparison.
    return [
        int((index * 7919 + (index % 160) * 113 + ((index // 400) % 7) * 997) % 24001)
        - 12000
        for index in range(length)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funasr-source", type=Path, required=True)
    parser.add_argument("--model-pt", type=Path, required=True)
    parser.add_argument("--cmvn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_revision = subprocess.run(
        ["git", "-C", str(args.funasr_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require_equal(source_revision, FUNASR_REVISION, "FunASR revision")
    require_equal(sha256(args.model_pt), MODEL_SHA256, "model.pt SHA-256")
    require_equal(sha256(args.cmvn), CMVN_SHA256, "am.mvn SHA-256")

    install_import_stubs()
    encoder_source = args.funasr_source / "funasr/models/fsmn_vad_streaming/encoder.py"
    frontend_source = args.funasr_source / "funasr/frontends/wav_frontend.py"
    encoder_module = load_module("vokra_official_fsmn_encoder", encoder_source)
    frontend_module = load_module("vokra_official_wav_frontend", frontend_source)

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    frontend = frontend_module.WavFrontendOnline(
        cmvn_file=str(args.cmvn),
        fs=16000,
        window="hamming",
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        lfr_m=5,
        lfr_n=1,
    ).eval()
    encoder = encoder_module.FSMN(
        input_dim=400,
        input_affine_dim=140,
        fsmn_layers=4,
        linear_dim=250,
        proj_dim=128,
        lorder=20,
        rorder=0,
        lstride=1,
        rstride=0,
        output_affine_dim=140,
        output_dim=248,
        use_softmax=True,
    ).eval()
    checkpoint = torch.load(args.model_pt, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(checkpoint)
    if not state or any(not name.startswith("encoder.") for name in state):
        raise SystemExit("checkpoint keys must all carry the official `encoder.` prefix")
    encoder_state = {name.removeprefix("encoder."): value for name, value in state.items()}
    result = encoder.load_state_dict(encoder_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"checkpoint load mismatch: {result}")

    pcm_i16 = deterministic_pcm_i16(16000)
    pcm = torch.tensor(pcm_i16, dtype=torch.float32).div_(32768.0).unsqueeze(0)
    features, lengths = frontend(
        pcm,
        torch.tensor([pcm.shape[1]], dtype=torch.int64),
        is_final=False,
        cache={},
    )
    frame_count = int(lengths[0])
    features = features[0, :frame_count].contiguous()
    probabilities = encoder(features.unsqueeze(0), cache={})[0].contiguous()
    if tuple(features.shape) != (96, 400):
        raise SystemExit(f"unexpected online frontend shape {tuple(features.shape)}")
    if tuple(probabilities.shape) != (96, 248):
        raise SystemExit(f"unexpected encoder shape {tuple(probabilities.shape)}")
    scores = 1.0 - probabilities[:, 0]

    payload = {
        "provenance": {
            "funasr_revision": FUNASR_REVISION,
            "model_revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA256,
            "cmvn_sha256": CMVN_SHA256,
            "encoder_source": "funasr/models/fsmn_vad_streaming/encoder.py",
            "frontend_source": "funasr/frontends/wav_frontend.py",
            "stream_final": False,
        },
        "sample_rate": 16000,
        "pcm_i16": pcm_i16,
        "n_frames": frame_count,
        "feature_width": 400,
        "output_width": 248,
        "features": features.tolist(),
        "probabilities": probabilities.tolist(),
        "speech_scores": scores.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(
        f"fsmn_vad_reference: wrote {frame_count} frames from official FunASR "
        f"{FUNASR_REVISION} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
