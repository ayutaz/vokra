#!/usr/bin/env python3
"""Independent official-source reference for Whisper-Medusa v1 module 0.

The caller supplies a checkout of github.com/aiola-lab/whisper-medusa at the
pinned revision and the exact HF snapshot.  This dumper imports that checkout's
`WhisperMedusaModel`; it does not mirror the Rust equations.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import inspect
import json
import os
import platform
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperProcessor

HF_REVISION = "6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462"
SOURCE_REVISION = "19819c37ab15db6e68826e406614a2c86fbb946e"
PREFIX = [50258, 50259, 50359, 50363]
EOT = 50257


def isolate_training_only_utils(package: Path) -> None:
    """Expose exact upstream utils modules without executing their package init.

    The pinned upstream ``utils/__init__.py`` eagerly imports trainer, metrics,
    and wandb code even when only ``config_and_args`` is needed by the model.
    Its requirements.txt does not declare wandb.  Registering a package shell
    with the exact upstream ``utils`` path avoids those unrelated side effects;
    Python still imports ``config_and_args.py`` and every model file from the
    pinned checkout.
    """

    name = "whisper_medusa.utils"
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(package / "utils")]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module


def install_transformers_compat() -> None:
    """Restore a removed, import-only Transformers 4.49 re-export.

    The pinned upstream ``models/model.py`` imports
    ``NEED_SETUP_CACHE_CLASSES_MAPPING`` from ``generation.utils`` but never
    reads it. Transformers 5 removed that internal re-export. Supplying the
    original module attribute as an empty mapping keeps the exact upstream
    source importable without changing its model equations or forward path.
    """

    from transformers.generation import utils as generation_utils  # noqa: PLC0415

    if not hasattr(generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
        generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {}


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().float().cpu().contiguous().numpy().astype("<f4").tofile(path)


def deterministic_pcm() -> np.ndarray:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    # Fixed low-amplitude multi-tone fixture: no dataset/license dependency.
    return (
        0.12 * np.sin(2 * np.pi * 220.0 * time)
        + 0.07 * np.sin(2 * np.pi * 440.0 * time + 0.2)
        + 0.03 * np.sin(2 * np.pi * 880.0 * time + 0.7)
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    package = args.source_parent / "whisper_medusa"
    if not package.is_dir():
        raise SystemExit(
            f"expected pinned upstream checkout at {package}; clone/rename it exactly"
        )
    sys.path.insert(0, str(args.source_parent))
    isolate_training_only_utils(package)
    install_transformers_compat()
    from whisper_medusa.models import WhisperMedusaModel  # noqa: PLC0415
    from whisper_medusa.models.model import (  # noqa: PLC0415
        Whisper2MedusaHeadsConditionalGeneration,
    )
    from whisper_medusa.utils import config_and_args  # noqa: PLC0415

    # Transformers 4.49 instantiated the outer model eagerly on CPU unless
    # low_cpu_mem_usage was requested. Transformers 5 always instantiates the
    # outer model under a meta-device context, but the pinned upstream
    # constructor performs a nested Whisper `from_pretrained`, which is
    # intentionally rejected under that context. Keep the upstream class and
    # checkpoint loader, but restore its former eager CPU boundary for that
    # nested call only.
    nested_from_pretrained = (
        Whisper2MedusaHeadsConditionalGeneration.from_pretrained.__func__
    )

    @classmethod
    def eager_nested_from_pretrained(cls, *model_args, **model_kwargs):
        with torch.device("cpu"):
            return nested_from_pretrained(cls, *model_args, **model_kwargs)

    Whisper2MedusaHeadsConditionalGeneration.from_pretrained = (
        eager_nested_from_pretrained
    )
    # The pinned wrapper predates Transformers 5's per-instance tied-weight
    # map. Its checkpoint intentionally omits `proj_out.weight`: canonical
    # Whisper ties that projection to the decoder token embedding. Prefix the
    # official Whisper mapping through the wrapper so the v5 loader neither
    # initializes random logits nor rejects the old class shape.
    WhisperMedusaModel.all_tied_weights_keys = {
        "whisper_model.proj_out.weight": (
            "whisper_model.model.decoder.embed_tokens.weight"
        )
    }

    model_source = Path(inspect.getsourcefile(WhisperMedusaModel) or "").resolve()
    config_source = Path(inspect.getsourcefile(config_and_args) or "").resolve()
    if package.resolve() not in model_source.parents:
        raise SystemExit(f"model import escaped pinned source checkout: {model_source}")
    if package.resolve() not in config_source.parents:
        raise SystemExit(f"config import escaped pinned source checkout: {config_source}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is false")
    selected_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if selected_device == "auto":
        selected_device = "cpu"
    device = torch.device(selected_device)
    torch.manual_seed(0)
    model = WhisperMedusaModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
    ).eval().to(device)
    processor = WhisperProcessor.from_pretrained(args.model_dir, local_files_only=True)
    pcm = deterministic_pcm()
    features = processor(
        pcm, sampling_rate=16_000, return_tensors="pt"
    ).input_features.to(device)

    ids = torch.tensor([PREFIX], dtype=torch.long, device=device)
    generated: list[int] = []
    prefix_logits = None
    with torch.inference_mode():
        encoder = model.whisper_model.model.encoder(features, return_dict=True)
        for _ in range(args.max_new_tokens):
            output = model(
                encoder_outputs=encoder,
                decoder_input_ids=ids,
                disable_medusa=True,
                use_cache=False,
                return_dict=True,
            )
            # disable_medusa=True leaves exactly module 0:
            # [head=1, batch=1, decoder_time, vocab].
            logits = output.logits[0, 0, -1]
            if prefix_logits is None:
                prefix_logits = logits
            token = int(torch.argmax(logits).item())
            generated.append(token)
            if token == EOT:
                break
            ids = torch.cat(
                [ids, torch.tensor([[token]], dtype=torch.long, device=device)], dim=1
            )

    assert prefix_logits is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pcm.astype("<f4").tofile(args.output_dir / "pcm.f32")
    write_f32(args.output_dir / "prefix_logits.f32", prefix_logits)
    np.asarray(generated, dtype="<u4").tofile(args.output_dir / "greedy_tokens.u32")
    text = processor.batch_decode([generated], skip_special_tokens=True)[0]
    manifest = {
        "hf_repo": "aiola/whisper-medusa-v1",
        "hf_revision": HF_REVISION,
        "source_repo": "https://github.com/aiola-lab/whisper-medusa",
        "source_revision": SOURCE_REVISION,
        "sample_rate": 16000,
        "pcm_samples": int(pcm.size),
        "decoder_prefix": PREFIX,
        "logits": int(prefix_logits.numel()),
        "greedy_tokens": generated,
        "max_new_tokens": args.max_new_tokens,
        "device": str(device),
        "reference_import": "whisper_medusa.models.WhisperMedusaModel",
        "model_source": str(model_source.relative_to(args.source_parent.resolve())),
        "config_source": str(config_source.relative_to(args.source_parent.resolve())),
        "training_utils_init": "bypassed; model/config modules are exact upstream files",
        "transformers_compat": [
            "restore unused 4.49 NEED_SETUP_CACHE_CLASSES_MAPPING re-export",
            "nested Whisper from_pretrained uses the former eager CPU boundary",
            "outer wrapper declares canonical tied proj_out/token embedding mapping",
        ],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "text": text,
        "max_abs_bound": 5e-4,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
