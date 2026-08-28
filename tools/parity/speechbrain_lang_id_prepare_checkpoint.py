#!/usr/bin/env python3
"""Prepare an official SpeechBrain language-ID release for Vokra.

The input oracle is ``speechbrain.inference.classifiers.EncoderClassifier``
loaded from one immutable Hugging Face revision.  This script does not mirror
ECAPA or either classifier in Python.  It asks the official package to build
and load the release, then serializes the real embedding module under the
unambiguous ``embedding_model.`` prefix.  The classifier is reduced to a
small, version-independent canonical tensor vocabulary after its official
class and complete state layout have been checked.  BatchNorm training
counters are the only inference-irrelevant state entries removed.

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


FORMAT = "vokra-speechbrain-lang-id-prepared-v2"
DEFAULT_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
SUPPORTED_SOURCES = {
    "speechbrain/lang-id-voxlingua107-ecapa": "lang-id-voxlingua107-ecapa",
    "speechbrain/lang-id-commonlanguage_ecapa": "lang-id-commonlanguage-ecapa",
}
PINNED_REVISIONS = {
    "speechbrain/lang-id-voxlingua107-ecapa": (
        "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
    ),
    "speechbrain/lang-id-commonlanguage_ecapa": (
        "70a742bbc513f693efcf73d6d64a5ed14b3a34a4"
    ),
}


def resolve_revision(source: str, revision: str | None) -> str:
    resolved = revision or PINNED_REVISIONS[source]
    if len(resolved) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in resolved
    ):
        raise SystemExit("--revision must be a full 40-hex commit")
    return resolved.lower()


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


def finite_contiguous(name: str, value: torch.Tensor) -> torch.Tensor:
    if not value.is_floating_point():
        raise RuntimeError(f"{name}: non-floating inference state is unsupported")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name}: official checkpoint contains non-finite values")
    return value.detach().cpu().contiguous()


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


def canonical_matrix(
    name: str, value: torch.Tensor, output_dim: int, input_dim: int
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in value.shape)
    if shape == (output_dim, input_dim):
        return finite_contiguous(name, value)
    if shape == (input_dim, output_dim):
        return finite_contiguous(name, value.transpose(0, 1))
    raise RuntimeError(
        f"{name}: projection shape {shape}, expected {(output_dim, input_dim)} "
        f"or its transpose"
    )


def canonical_classifier_state(
    module: torch.nn.Module,
    kind: str,
    input_dim: int,
    class_count: int,
    hidden_dim: int | None,
) -> tuple[dict[str, torch.Tensor], list[str], dict[str, str]]:
    state = module.state_dict()
    counters = sorted(name for name in state if name.endswith(".num_batches_tracked"))
    float_values = {
        name: finite_contiguous(name, value)
        for name, value in state.items()
        if name not in counters
    }
    provenance: dict[str, str] = {}
    if kind == "ecapa-cosine-v1":
        rank2 = [(name, value) for name, value in float_values.items() if value.ndim == 2]
        if len(rank2) != 1:
            raise RuntimeError(
                "official cosine classifier does not expose exactly one rank-2 tensor: "
                f"{[(name, tuple(value.shape)) for name, value in rank2]}"
            )
        source_name, source = rank2[0]
        unclaimed = sorted(name for name in float_values if name != source_name)
        if unclaimed:
            raise RuntimeError(
                f"official cosine classifier has unsupported extra state: {unclaimed}"
            )
        canonical = canonical_matrix(
            source_name, source, output_dim=class_count, input_dim=input_dim
        )
        provenance["classifier.cosine.weight"] = source_name
        return {"classifier.cosine.weight": canonical}, counters, provenance

    if kind != "xvector-mlp-log-softmax-v1" or hidden_dim is None:
        raise RuntimeError(f"cannot canonicalize classifier kind={kind} hidden={hidden_dim}")

    rank2 = [(name, value) for name, value in float_values.items() if value.ndim == 2]
    hidden_candidates = [
        (name, value)
        for name, value in rank2
        if sorted(value.shape) == sorted((input_dim, hidden_dim))
    ]
    output_candidates = [
        (name, value)
        for name, value in rank2
        if sorted(value.shape) == sorted((hidden_dim, class_count))
    ]
    if len(hidden_candidates) != 1 or len(output_candidates) != 1:
        raise RuntimeError(
            "official XVector classifier does not expose exactly one hidden and output "
            f"projection: rank2={[(name, tuple(value.shape)) for name, value in rank2]}"
        )

    used: set[str] = set()
    output: dict[str, torch.Tensor] = {}
    for canonical_name, (source_name, source), out_dim, in_dim in [
        (
            "classifier.hidden.weight",
            hidden_candidates[0],
            hidden_dim,
            input_dim,
        ),
        (
            "classifier.output.weight",
            output_candidates[0],
            class_count,
            hidden_dim,
        ),
    ]:
        output[canonical_name] = canonical_matrix(source_name, source, out_dim, in_dim)
        provenance[canonical_name] = source_name
        used.add(source_name)

    bn_groups: list[tuple[str, int]] = []
    for name, value in float_values.items():
        if not name.endswith(".running_mean"):
            continue
        prefix = name.removesuffix(".running_mean")
        width = int(value.numel())
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            key = f"{prefix}.{suffix}"
            candidate = float_values.get(key)
            if candidate is None or tuple(candidate.shape) != (width,):
                raise RuntimeError(f"official BatchNorm group {prefix} is incomplete at {key}")
        bn_groups.append((prefix, width))
    for canonical_prefix, width in [
        ("classifier.input_norm", input_dim),
        ("classifier.hidden_norm", hidden_dim),
    ]:
        candidates = [prefix for prefix, found_width in bn_groups if found_width == width]
        if len(candidates) != 1:
            raise RuntimeError(
                f"official classifier has {len(candidates)} BatchNorm groups at width {width}: {bn_groups}"
            )
        source_prefix = candidates[0]
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            source_name = f"{source_prefix}.{suffix}"
            canonical_name = f"{canonical_prefix}.{suffix}"
            output[canonical_name] = float_values[source_name]
            provenance[canonical_name] = source_name
            used.add(source_name)

    remaining_rank1 = [
        (name, value)
        for name, value in float_values.items()
        if value.ndim == 1 and name not in used
    ]
    for canonical_name, width in [
        ("classifier.hidden.bias", hidden_dim),
        ("classifier.output.bias", class_count),
    ]:
        candidates = [
            (name, value) for name, value in remaining_rank1 if value.numel() == width
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"official classifier has {len(candidates)} unclaimed biases at width {width}: "
                f"{[(name, tuple(value.shape)) for name, value in remaining_rank1]}"
            )
        source_name, source = candidates[0]
        output[canonical_name] = source
        provenance[canonical_name] = source_name
        used.add(source_name)
        remaining_rank1 = [entry for entry in remaining_rank1 if entry[0] != source_name]

    unclaimed = sorted(name for name in float_values if name not in used)
    if unclaimed:
        raise RuntimeError(f"official XVector classifier has unclaimed state: {unclaimed}")
    if len(output) != 12:
        raise RuntimeError(f"canonical XVector classifier has {len(output)} tensors, expected 12")
    return output, counters, provenance


def ecapa_contract_dims(
    state: dict[str, torch.Tensor]
) -> tuple[int, int, int, int, list[int]]:
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
    block_kernels: list[int] = []
    for block in range(1, 4):
        block_shape = shape(
            f"embedding_model.blocks.{block}.res2net_block.blocks.0.conv.conv.weight"
        )
        if len(block_shape) != 3 or block_shape[2] <= 0:
            raise RuntimeError(
                f"unexpected official ECAPA block {block} shape {block_shape}"
            )
        block_kernels.append(block_shape[2])
    return stem[1], stem[0], mfa[0], attention[0], block_kernels


def ecapa_block_dilations(module: torch.nn.Module) -> list[int]:
    output: list[int] = []
    for block in range(1, 4):
        conv = module.blocks[block].res2net_block.blocks[0].conv
        dilation = getattr(conv, "dilation", None)
        if dilation is None:
            dilation = getattr(getattr(conv, "conv", None), "dilation", None)
        if isinstance(dilation, tuple):
            if len(dilation) != 1:
                raise RuntimeError(f"official ECAPA block {block} dilation={dilation}")
            dilation = dilation[0]
        if not isinstance(dilation, int) or dilation <= 0:
            raise RuntimeError(f"official ECAPA block {block} has invalid dilation={dilation}")
        output.append(dilation)
    return output


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
    parser.add_argument(
        "--revision",
        help="full upstream commit (defaults to the source-specific audited pin)",
    )
    parser.add_argument("--savedir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = resolve_revision(args.source, args.revision)

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        inference = EncoderClassifier.from_hparams(
            source=args.source,
            revision=revision,
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
    kind = classifier_kind(classifier)
    input_dim, class_count, hidden_dim = classifier_dims(classifier, kind, len(labels))
    if class_count != len(labels):
        raise RuntimeError(
            f"classifier outputs {class_count} classes but label encoder has {len(labels)}"
        )
    embedding_dim = input_dim
    classifier_state, classifier_counters, classifier_sources = canonical_classifier_state(
        classifier, kind, input_dim, class_count, hidden_dim
    )
    tensors = {**embedding_state, **classifier_state}
    if len(tensors) != len(embedding_state) + len(classifier_state):
        raise RuntimeError("embedding/classifier tensor names collide after prefixing")

    n_mels, tdnn_channels, mfa_channels, attention_channels, block_kernels = (
        ecapa_contract_dims(tensors)
    )
    block_dilations = ecapa_block_dilations(embedding)
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
        "revision": revision,
        "sample_rate": 16_000,
        "n_mels": n_mels,
        "tdnn_channels": tdnn_channels,
        "mfa_channels": mfa_channels,
        "attention_channels": attention_channels,
        "res2net_scale": res2net_scale,
        "block_kernels": block_kernels,
        "block_dilations": block_dilations,
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
        "vokra.lang_id.transform": "official-embedding-prefix-canonical-classifier-v2",
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
        "classifier_canonical_sources": classifier_sources,
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
