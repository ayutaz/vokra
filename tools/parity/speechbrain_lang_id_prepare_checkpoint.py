#!/usr/bin/env python3
"""Prepare an official SpeechBrain language-ID release for Vokra.

The input oracle is ``speechbrain.inference.classifiers.EncoderClassifier``
loaded from one immutable Hugging Face revision.  This script does not mirror
ECAPA or either classifier in Python.  It asks the official package to build
and load the release, then serializes its two real inference modules under the
unambiguous ``embedding_model.`` and ``classifier.`` prefixes.  BatchNorm
training counters are the only state entries removed.

The label encoder and the runtime contract are stored in safetensors
``__metadata__``.  The Rust converter requires that metadata and refuses the
historical embedding-only input, so a future public GGUF cannot silently lose
the classifier or its index-to-language mapping again.

Run through the parity tree's uv environment on VAST::

    uv run --project tools/parity --python 3.12 python \
      tools/parity/speechbrain_lang_id_prepare_checkpoint.py \
      --source speechbrain/lang-id-voxlingua107-ecapa \
      --revision 0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9 \
      --savedir /root/lang-id-upstream \
      --output /root/lang-id-voxlingua107.prepared.safetensors

If the official model, classifier, or label encoder cannot be loaded, the
script aborts.  There is no local architecture or label fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import huggingface_hub
import torch
import torchaudio
from huggingface_hub.errors import RemoteEntryNotFoundError
from requests.exceptions import HTTPError
from safetensors.torch import save_file

# SpeechBrain 1.0.3 still probes APIs removed from recent torchaudio and uses
# the retired huggingface_hub ``use_auth_token`` spelling.  These compatibility
# shims are transport-only; inference and checkpoint values remain upstream.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    try:
        return _hf_hub_download(*args, **kwargs)
    except RemoteEntryNotFoundError as error:
        raise HTTPError(f"404 Client Error: {error}") from error


huggingface_hub.hf_hub_download = _hf_hub_download_compat

try:
    import speechbrain
    from speechbrain.inference.classifiers import EncoderClassifier
except Exception as error:  # noqa: BLE001 - loud official-loader failure
    raise SystemExit(
        "speechbrain_lang_id_prepare_checkpoint: could not import the real "
        f"SpeechBrain implementation ({type(error).__name__}: {error}); a "
        "mirror fallback is forbidden"
    ) from error


FORMAT = "vokra-speechbrain-lang-id-prepared-v1"
DEFAULT_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
DEFAULT_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
SUPPORTED_SOURCES = {
    "speechbrain/lang-id-voxlingua107-ecapa": "lang-id-voxlingua107-ecapa",
    "speechbrain/lang-id-commonlanguage_ecapa": "lang-id-commonlanguage-ecapa",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_labels(encoder: object) -> list[str]:
    lab2ind = getattr(encoder, "lab2ind", None)
    if not isinstance(lab2ind, dict) or not lab2ind:
        raise RuntimeError("official label encoder has no non-empty lab2ind mapping")
    labels: list[str | None] = [None] * len(lab2ind)
    for label, index in lab2ind.items():
        if not isinstance(label, str) or not isinstance(index, int):
            raise RuntimeError("official label encoder has non string->int entry")
        if index < 0 or index >= len(labels) or labels[index] is not None:
            raise RuntimeError(f"official label encoder has invalid index {index}")
        labels[index] = label
    if any(label is None or not label for label in labels):
        raise RuntimeError("official label encoder indices are not contiguous and non-empty")
    return [label for label in labels if label is not None]


def float_state(prefix: str, module: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[str]]:
    state = module.state_dict()
    counters = sorted(name for name in state if name.endswith(".num_batches_tracked"))
    output: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if name in counters:
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise RuntimeError(f"{prefix}{name}: non-floating inference state is unsupported")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{prefix}{name}: official checkpoint contains non-finite values")
        output[f"{prefix}{name}"] = value.detach().cpu().contiguous()
    if not output:
        raise RuntimeError(f"official module {prefix.rstrip('.')} has zero inference tensors")
    return output, counters


def classifier_kind(module: torch.nn.Module) -> str:
    identity = f"{type(module).__module__}.{type(module).__name__}"
    if identity == "speechbrain.lobes.models.Xvector.Classifier":
        return "xvector-mlp-log-softmax-v1"
    if identity == "speechbrain.lobes.models.ECAPA_TDNN.Classifier":
        return "ecapa-cosine-v1"
    raise RuntimeError(f"unsupported official classifier class {identity}")


def classifier_dims(
    module: torch.nn.Module, kind: str, label_count: int
) -> tuple[int, int, int | None]:
    rank2 = [
        (name, tuple(int(dim) for dim in value.shape))
        for name, value in module.state_dict().items()
        if value.ndim == 2
    ]
    if kind == "ecapa-cosine-v1":
        if len(rank2) != 1 or rank2[0][1][0] != label_count:
            raise RuntimeError(
                f"official cosine classifier rank-2 state {rank2} does not match {label_count} labels"
            )
        return rank2[0][1][1], label_count, None

    if kind != "xvector-mlp-log-softmax-v1" or len(rank2) != 2:
        raise RuntimeError(f"unsupported official classifier rank-2 state {rank2}")
    output = [entry for entry in rank2 if label_count in entry[1]]
    if len(output) != 1:
        raise RuntimeError(
            f"could not identify one {label_count}-class output projection in {rank2}"
        )
    output_name, output_shape = output[0]
    hidden_entry = next(entry for entry in rank2 if entry[0] != output_name)
    output_non_class = [dim for dim in output_shape if dim != label_count]
    if len(output_non_class) != 1:
        raise RuntimeError(f"ambiguous output projection shape {output_shape}")
    hidden_dim = output_non_class[0]
    hidden_shape = hidden_entry[1]
    if hidden_shape.count(hidden_dim) != 1:
        raise RuntimeError(
            f"hidden projection {hidden_entry} does not connect to output width {hidden_dim}"
        )
    input_dim = next(dim for dim in hidden_shape if dim != hidden_dim)
    return input_dim, label_count, hidden_dim


def ecapa_contract_dims(state: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    def shape(name: str) -> tuple[int, ...]:
        value = state.get(name)
        if value is None:
            raise RuntimeError(f"official ECAPA state is missing {name}")
        return tuple(int(dim) for dim in value.shape)

    stem = shape("embedding_model.blocks.0.conv.conv.weight")
    mfa = shape("embedding_model.mfa.conv.conv.weight")
    attention = shape("embedding_model.asp.tdnn.conv.conv.weight")
    if len(stem) != 3 or stem[2] != 5:
        raise RuntimeError(f"unexpected official ECAPA stem shape {stem}")
    if mfa != (mfa[0], mfa[0], 1):
        raise RuntimeError(f"unexpected official ECAPA MFA shape {mfa}")
    if len(attention) != 3 or attention[1] != mfa[0] * 3 or attention[2] != 1:
        raise RuntimeError(f"unexpected official ECAPA attention shape {attention}")
    return stem[1], stem[0], mfa[0], attention[0]


def batch_norm_eps(module: torch.nn.Module) -> float:
    values = {
        float(child.eps)
        for child in module.modules()
        if isinstance(child, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
    }
    if len(values) != 1:
        raise RuntimeError(f"official modules expose non-uniform BatchNorm eps values {values}")
    return values.pop()


def leaky_relu_slope(module: torch.nn.Module) -> float:
    values = {
        float(child.negative_slope)
        for child in module.modules()
        if isinstance(child, torch.nn.LeakyReLU)
    }
    if len(values) != 1:
        raise RuntimeError(f"official classifier exposes unexpected LeakyReLU slopes {values}")
    return values.pop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(SUPPORTED_SOURCES))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--savedir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        inference = EncoderClassifier.from_hparams(
            source=args.source,
            revision=args.revision,
            savedir=args.savedir,
            run_opts={"device": "cpu"},
        )
    except Exception as error:  # noqa: BLE001 - retain official failure detail
        raise SystemExit(
            "speechbrain_lang_id_prepare_checkpoint: the real pinned model "
            f"could not be loaded ({type(error).__name__}: {error})"
        ) from error

    embedding = inference.mods.embedding_model
    classifier = inference.mods.classifier
    embedding.eval()
    classifier.eval()
    labels = contiguous_labels(inference.hparams.label_encoder)
    embedding_state, embedding_counters = float_state("embedding_model.", embedding)
    classifier_state, classifier_counters = float_state("classifier.", classifier)
    tensors = {**embedding_state, **classifier_state}
    if len(tensors) != len(embedding_state) + len(classifier_state):
        raise RuntimeError("embedding/classifier tensor names collide after prefixing")

    kind = classifier_kind(classifier)
    input_dim, class_count, hidden_dim = classifier_dims(classifier, kind, len(labels))
    if class_count != len(labels):
        raise RuntimeError(
            f"classifier outputs {class_count} classes but label encoder has {len(labels)}"
        )
    embedding_dim = input_dim
    n_mels, tdnn_channels, mfa_channels, attention_channels = ecapa_contract_dims(tensors)
    res2net_indices = {
        int(name.split("res2net_block.blocks.", 1)[1].split(".", 1)[0])
        for name in embedding_state
        if "blocks.1.res2net_block.blocks." in name
    }
    if res2net_indices != set(range(len(res2net_indices))):
        raise RuntimeError(f"official ECAPA Res2Net block indices are not contiguous: {res2net_indices}")
    res2net_scale = len(res2net_indices) + 1

    contract = {
        "format": FORMAT,
        "model_name": SUPPORTED_SOURCES[args.source],
        "source": args.source,
        "revision": args.revision,
        "sample_rate": 16_000,
        "n_mels": n_mels,
        "tdnn_channels": tdnn_channels,
        "mfa_channels": mfa_channels,
        "attention_channels": attention_channels,
        "res2net_scale": res2net_scale,
        "embedding_dim": embedding_dim,
        "classifier_kind": kind,
        "classifier_hidden_dim": hidden_dim,
        "class_count": class_count,
        "labels": labels,
        "bn_eps": batch_norm_eps(inference.mods),
        "stats_eps": 1.0e-12,
        "leaky_relu_slope": (
            leaky_relu_slope(classifier)
            if kind == "xvector-mlp-log-softmax-v1"
            else None
        ),
    }
    metadata = {
        "vokra.lang_id.contract": json.dumps(contract, sort_keys=True, separators=(",", ":")),
        "vokra.lang_id.transform": "official-modules-prefix-and-remove-bn-counters-only",
        "vokra.lang_id.speechbrain_version": speechbrain.__version__,
        "vokra.lang_id.torch_version": torch.__version__,
        "vokra.lang_id.python_version": platform.python_version(),
        "vokra.lang_id.machine": platform.machine(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    manifest = {
        "contract": contract,
        "embedding_tensors": len(embedding_state),
        "classifier_tensors": len(classifier_state),
        "embedding_counters_removed": embedding_counters,
        "classifier_counters_removed": classifier_counters,
        "tensor_manifest": {
            name: list(value.shape) for name, value in sorted(tensors.items())
        },
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
