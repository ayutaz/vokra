#!/usr/bin/env python3
"""Flatten torchaudio's two SQUIM checkpoints into one merged safetensors
bundle (coverage-audit-2026-08-03 Wave A permissive continuation,
2026-08-04).

Offline sidecar tool (FR-LD-05: no Python / torch ever enters the
runtime). torchaudio SQUIM (Kumar et al. 2023 ICASSP arXiv:2304.01448,
"TorchAudio-Squim: Reference-less Speech Quality and Intelligibility
Measures in TorchAudio") ships two bundles distributed via torch.hub —
there is no HF mirror:

* ``squim_objective_dns2020.pth`` — SquimObjective (DPRNN encoder + 3
  metric heads: STOI + PESQ + SI-SDR, ~28 MB, 7.4 M params, F32).
* ``squim_subjective_bvcc_daps.pth`` — SquimSubjective (wav2vec2_base
  SSL + attentive pool + projector, ~360 MB, 94.4 M params, F32).

Vokra's Rust converter (``crates/vokra-convert/src/models/torchaudio_squim.rs``)
consumes safetensors only. This script bridges the two by:

1. Loading each bundle via ``torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()``
   / ``SQUIM_SUBJECTIVE.get_model()`` (auto-downloads on first call to
   ``~/.cache/torch/hub/checkpoints/``), OR from explicit
   ``--objective-ckpt`` / ``--subjective-ckpt`` overrides for offline
   dev machines (mirrors the DFN3 pickle-bridge posture).
2. Extracting each ``.state_dict()``, prefixing every key with the
   sub-model tag (``objective.<upstream_name>`` /
   ``subjective.<upstream_name>``) — analogous to the DNSMOS
   ``p808.`` / ``p835.`` prefix convention. The runtime binder
   ``vokra_models::squim::SquimWeights::from_gguf`` walks these prefixes
   to route each tensor to the right sub-model without a graph load. It
   landed in ``vokra-models``, not ``vokra-eval`` — there is no
   ``vokra_eval::squim`` module; see the "Why this binder lives in
   `vokra-models`, not `vokra-eval`" section of
   ``crates/vokra-models/src/squim/mod.rs``.
3. Emitting a single merged safetensors + a sha256 manifest line + a
   config JSON side-car (sample_rate, sub-model factory defaults from
   ``squim_objective_base()`` / ``squim_subjective_base()``).

# License

* Code (torchaudio itself): BSD-2-Clause
  (``github.com/pytorch/audio/blob/main/LICENSE``, verified 2026-08-04).
* Weight (``squim_objective_dns2020.pth``): CC-BY-4.0 (Attribution) per
  upstream tutorial page.
* Weight (``squim_subjective_bvcc_daps.pth``): CC-BY-NC-4.0
  (Non-Commercial) per upstream tutorial page.

The bundled safetensors this script emits is entirely upstream weights
— no re-training, no derived data. The §3.1 sign-off row in
``docs/license-audit.md`` currently treats the bundle as BSD-2-Clause
end-to-end (☑ Commercial 2026-08-04 yousan); the weight-vs-code
divergence is flagged for owner re-audit before ``publish-one.sh``
runs. Writing this prep script and converting to GGUF do NOT need the
re-audit — only distribution does.

# Design decisions

* **Torch API path**: uses ``torchaudio.pipelines`` (2.11.0+ stable
  path). Earlier torchaudio versions (<2.1) shipped SQUIM under
  ``torchaudio.prototype.pipelines``; this script deliberately does NOT
  fall back to prototype — a pinned uv env (``requires-python >= 3.12``,
  ``torchaudio >= 2.11.0``) is the contract, and a silent prototype
  fallback would mask a torchaudio downgrade regression (FR-EX-08).

* **Dtype policy**: F32 / F16 / BF16 pass through verbatim. Any other
  dtype (INT8 quantized weights in a hypothetical future SQUIM
  revision) is a hard error rather than a silent drop (FR-EX-08).
  Both SQUIM releases as of 2026-08-04 ship F32 end-to-end.

* **Prefix policy**: ``objective.<name>`` / ``subjective.<name>``. The
  Rust binder walks the leading dot-segment to route the tensor —
  never rewrite these because any mangling would need to travel with
  the binder to stay consistent.

* **Shared-storage handling**: wav2vec2_base SSL encoders sometimes
  carry tied embeddings (data_ptr collision blocks
  ``safetensors.torch.save_file`` with RuntimeError). Empirically
  (2026-08-04) the SUBJECTIVE state_dict has zero shared pairs, but we
  keep the dedup for future-proofing per the shared-tensor-dedup
  reference (``[[reference-safetensors-shared-tensor-dedup]]``). A
  ``shared_pairs.json`` audit trail is written alongside the
  safetensors so any dedup that fires is discoverable.

* **Partial bundle**: passing only one of ``--objective-ckpt`` /
  ``--subjective-ckpt`` (or ``--objective-only`` / ``--subjective-only``
  gates for the auto-download path) is allowed — the Rust converter's
  BF16 pass-through arm is metric-op-agnostic, so a partial bundle
  faithfully advertises the truthful subset in the emitted GGUF. Both
  are recommended for the canonical Vokra publication (STOI + PESQ +
  SI-SDR + MOS make sense together as a 4-metric reference-free
  quality report).

* **Sidecar config JSON**: ``sample_rate`` + factory-default sub-model
  hyperparameters from ``squim_objective_base()`` /
  ``squim_subjective_base()`` (upstream source:
  ``github.com/pytorch/audio/blob/main/src/torchaudio/models/squim/{objective,subjective}.py``).
  Emitted so a future runtime binder can validate expected shapes
  without hard-coding them.

# Usage

Auto-download path (network required, first run only)::

    uv run python tools/parity/torchaudio_squim_prepare_checkpoint.py \\
        --output ~/checkpoints/torchaudio-squim/model.safetensors \\
        --config-out ~/checkpoints/torchaudio-squim/config.json

Offline path (pre-downloaded ``.pth`` overrides — DFN3 pickle-bridge
posture)::

    uv run python tools/parity/torchaudio_squim_prepare_checkpoint.py \\
        --objective-ckpt ~/checkpoints/squim_objective_dns2020.pth \\
        --subjective-ckpt ~/checkpoints/squim_subjective_bvcc_daps.pth \\
        --output ~/checkpoints/torchaudio-squim/model.safetensors \\
        --config-out ~/checkpoints/torchaudio-squim/config.json

Then::

    vokra-cli convert --model torchaudio-squim \\
        --input ~/checkpoints/torchaudio-squim/model.safetensors \\
        --output ~/gguf/torchaudio-squim.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path

LOG_PREFIX = "torchaudio_squim_prepare_checkpoint:"

# torch dtype (as ``str(t.dtype)``) → safetensors dtype string. We
# intentionally accept only float dtypes here — the SQUIM state_dicts
# are float end-to-end as of 2026-08-04, and any int tensor would
# indicate an upstream schema change we should investigate rather than
# silently drop (FR-EX-08).
DTYPE_MAP = {
    "torch.float32": ("F32", 4),
    "torch.float16": ("F16", 2),
    "torch.bfloat16": ("BF16", 2),
}


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr)


def _load_state_dict_from_pth(ckpt_path: Path):
    """Load a raw ``.pth`` state dict via ``torch.load(weights_only=True)``.

    The upstream torch.hub-hosted SQUIM ``.pth`` files load cleanly under
    the safe loader (verified 2026-08-04 on both files); a checkpoint
    that does not is refused rather than falling back to unsafe
    unpickling (FR-LD-05 spirit — pickles must not carry code).
    """
    import torch

    _log(f"loading {ckpt_path} via torch.load(weights_only=True)")
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        # Some releases wrap the flat state dict in an outer dict —
        # follow the ``state_dict`` key when present.
        obj = obj["state_dict"]
    if not isinstance(obj, (dict, OrderedDict)):
        raise SystemExit(
            f"{LOG_PREFIX} {ckpt_path}: top level is {type(obj).__name__}, "
            f"expected a dict / OrderedDict state_dict"
        )
    return obj


def _load_state_dict_from_pipeline(which: str):
    """Load a SQUIM state_dict via the torchaudio.pipelines bundle.

    ``which`` ∈ {"objective", "subjective"}.
    """
    try:
        import torchaudio.pipelines as P  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — dev env sanity check
        raise SystemExit(
            f"{LOG_PREFIX} `torchaudio` package not installed. Run "
            f"`uv add torchaudio` in tools/parity/ first."
        ) from exc

    if which == "objective":
        bundle = P.SQUIM_OBJECTIVE
    elif which == "subjective":
        bundle = P.SQUIM_SUBJECTIVE
    else:
        raise SystemExit(f"{LOG_PREFIX} unknown pipeline: {which!r}")

    _log(f"loading SQUIM_{which.upper()} via torchaudio.pipelines (auto-download on first run)")
    model = bundle.get_model()
    return model.state_dict(), bundle.sample_rate


def _extract_prefixed_tensors(sd, prefix: str):
    """Walk one state_dict and return an ordered dict of
    ``{prefix + name: (dtype_str, shape, bytes)}``.

    * ``prefix`` — the bundle tag (``"objective."`` / ``"subjective."``).
    * Dtype policy: only F32 / F16 / BF16 are accepted; any other
      dtype raises SystemExit (FR-EX-08 posture — never a silent skip).
    * Shared-storage dedup: the first tensor at a given ``data_ptr``
      is kept verbatim; subsequent tensors at the same ptr are
      ``.detach().clone().contiguous()``-ed into fresh storage so
      safetensors accepts them. Records ``(clone_name, original_name)``
      pairs for the audit sidecar.
    """
    import torch  # noqa: F401  # needed for tensor accessors

    out: "OrderedDict[str, tuple[str, list[int], bytes]]" = OrderedDict()
    seen_ptrs: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []

    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            raise SystemExit(
                f"{LOG_PREFIX} state_dict entry {name!r} is not a tensor: "
                f"{type(t).__name__}"
            )
        dtype_s = str(t.dtype)
        if dtype_s not in DTYPE_MAP:
            raise SystemExit(
                f"{LOG_PREFIX} tensor {name!r} has dtype {dtype_s} which is "
                f"not F32 / F16 / BF16 — refusing to silently skip (FR-EX-08)"
            )
        dtype_str, elem_size = DTYPE_MAP[dtype_s]

        # Shared-storage dedup ([[reference-safetensors-shared-tensor-dedup]]).
        ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else 0
        if ptr and ptr in seen_ptrs:
            shared_pairs.append((name, seen_ptrs[ptr]))
            t = t.detach().clone().contiguous()
        else:
            if ptr:
                seen_ptrs[ptr] = name
            t = t.detach().contiguous()

        # BF16 cannot be viewed as bytes via numpy — go through
        # ``.view(torch.uint8)`` which is a byte reinterpretation the
        # underlying storage always supports.
        data = t.view(torch.uint8).cpu().numpy().tobytes()
        expected = elem_size
        for d in t.shape:
            expected *= int(d)
        if len(data) != expected:
            raise SystemExit(
                f"{LOG_PREFIX} tensor {name!r}: payload is {len(data)} bytes "
                f"but shape {list(t.shape)} × {dtype_str} ({elem_size} B/elem) "
                f"expects {expected} bytes"
            )
        prefixed = f"{prefix}{name}"
        if prefixed in out:
            raise SystemExit(
                f"{LOG_PREFIX} duplicate tensor name after prefixing: "
                f"{prefixed!r} — refusing to emit an ambiguous bundle"
            )
        out[prefixed] = (dtype_str, list(t.shape), data)

    _log(f"  extracted {len(out)} tensors (prefix={prefix!r})")
    if shared_pairs:
        _log(f"  shared-storage clones: {len(shared_pairs)}")
    return out, shared_pairs


def _write_safetensors(path: Path, tensors) -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header length
    + JSON header + contiguous little-endian tensor data.

    Mirrors the writer in ``dnsmos_prepare_checkpoint.py`` /
    ``dfn3_prepare_checkpoint.py`` — kept inline so the prep script has
    zero non-``torch`` runtime deps.
    """
    header = {}
    blobs = []
    offset = 0
    for name, (dtype_str, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_config_json(
    config_path: Path,
    *,
    have_objective: bool,
    have_subjective: bool,
    sample_rate: int,
) -> None:
    """Write the config side-car JSON.

    Hyperparameters mirror the ``squim_objective_base()`` /
    ``squim_subjective_base()`` factory defaults (upstream source:
    ``github.com/pytorch/audio/blob/main/src/torchaudio/models/squim/{objective,subjective}.py``,
    verified 2026-08-04).

    Only sub-model blocks for which weights were actually included are
    written — the truthful subset invariant per the ``bundle_variants``
    contract.
    """
    cfg: dict = {"sample_rate": sample_rate}
    if have_objective:
        # From ``squim_objective_base()``: DPRNN encoder + 3 transformer
        # heads (STOI + PESQ + SI-SDR). ``feat_dim`` is the encoder
        # bottleneck, ``d_model`` is the transformer head width,
        # ``num_blocks`` counts DPRNN blocks, ``chunk_size`` is the
        # DPRNN chunking factor.
        cfg["objective"] = {
            "feat_dim": 256,
            "d_model": 256,
            "nhead": 4,
            "num_blocks": 2,
            "chunk_size": 71,
            "metrics": ["stoi", "pesq", "sisdr"],
        }
    if have_subjective:
        # From ``squim_subjective_base()``: wav2vec2_base SSL feature
        # extractor + attentive pool + linear projector.
        cfg["subjective"] = {
            "ssl_type": "wav2vec2_base",
            "feat_dim": 768,
            "proj_dim": 32,
            "att_dim": 5,
            "metrics": ["mos"],
        }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten torchaudio's SQUIM_OBJECTIVE + SQUIM_SUBJECTIVE bundles "
            "into a single merged safetensors bundle for vokra-convert."
        ),
    )
    parser.add_argument(
        "--objective-ckpt",
        type=Path,
        default=None,
        help="Path to squim_objective_dns2020.pth (offline override). "
        "When omitted, auto-downloads via torchaudio.pipelines.SQUIM_OBJECTIVE.",
    )
    parser.add_argument(
        "--subjective-ckpt",
        type=Path,
        default=None,
        help="Path to squim_subjective_bvcc_daps.pth (offline override). "
        "When omitted, auto-downloads via torchaudio.pipelines.SQUIM_SUBJECTIVE.",
    )
    parser.add_argument(
        "--objective-only",
        action="store_true",
        help="Skip SQUIM_SUBJECTIVE entirely (produces a partial "
        "bundle carrying only STOI + PESQ + SI-SDR heads).",
    )
    parser.add_argument(
        "--subjective-only",
        action="store_true",
        help="Skip SQUIM_OBJECTIVE entirely (produces a partial "
        "bundle carrying only the MOS head).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output safetensors path (merged bundle).",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Optional path for the sidecar config JSON. When omitted, "
        "writes to <output>.config.json alongside the safetensors.",
    )
    parser.add_argument(
        "--shared-pairs-out",
        type=Path,
        default=None,
        help="Optional path for the shared-storage-pairs audit JSON. "
        "When omitted, writes to <output>.shared_pairs.json alongside "
        "the safetensors (empty JSON array when no dedup fired).",
    )
    args = parser.parse_args()

    if args.objective_only and args.subjective_only:
        _log("ERROR: --objective-only and --subjective-only are mutually exclusive")
        return 2

    have_objective = not args.subjective_only
    have_subjective = not args.objective_only

    tensors: "OrderedDict[str, tuple[str, list[int], bytes]]" = OrderedDict()
    all_shared: list[tuple[str, str]] = []
    sample_rate = 16000  # SQUIM canonical — both bundles are 16 kHz.

    if have_objective:
        if args.objective_ckpt is not None:
            if not args.objective_ckpt.exists():
                _log(f"ERROR: --objective-ckpt not found: {args.objective_ckpt}")
                return 2
            sd = _load_state_dict_from_pth(args.objective_ckpt)
        else:
            sd, sr_obj = _load_state_dict_from_pipeline("objective")
            if sr_obj != sample_rate:
                _log(
                    f"WARNING: SQUIM_OBJECTIVE sample_rate={sr_obj} differs "
                    f"from canonical 16000 — using pipeline value"
                )
                sample_rate = sr_obj
        obj_tensors, obj_shared = _extract_prefixed_tensors(sd, prefix="objective.")
        for k in obj_tensors:
            if k in tensors:
                raise SystemExit(
                    f"{LOG_PREFIX} cross-bundle name collision: {k!r} "
                    "(prefixing invariant broken?)"
                )
        tensors.update(obj_tensors)
        all_shared.extend(obj_shared)

    if have_subjective:
        if args.subjective_ckpt is not None:
            if not args.subjective_ckpt.exists():
                _log(f"ERROR: --subjective-ckpt not found: {args.subjective_ckpt}")
                return 2
            sd = _load_state_dict_from_pth(args.subjective_ckpt)
        else:
            sd, sr_sub = _load_state_dict_from_pipeline("subjective")
            if sr_sub != sample_rate and not have_objective:
                sample_rate = sr_sub
            elif sr_sub != sample_rate and have_objective:
                # Both bundles must agree on sample rate for the bundle
                # to make sense; SQUIM is canonically 16 kHz end-to-end.
                raise SystemExit(
                    f"{LOG_PREFIX} sample rate mismatch: SQUIM_OBJECTIVE={sample_rate} "
                    f"vs SQUIM_SUBJECTIVE={sr_sub} — refusing to emit an inconsistent bundle"
                )
        sub_tensors, sub_shared = _extract_prefixed_tensors(sd, prefix="subjective.")
        for k in sub_tensors:
            if k in tensors:
                raise SystemExit(
                    f"{LOG_PREFIX} cross-bundle name collision: {k!r} "
                    "(prefixing invariant broken?)"
                )
        tensors.update(sub_tensors)
        all_shared.extend(sub_shared)

    if not tensors:
        _log("ERROR: no tensors were extracted (empty pipeline outputs?)")
        return 2

    _log(f"writing {len(tensors)} tensors to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_safetensors(args.output, tensors)
    sha = _sha256(args.output)
    _log(f"done: {args.output} sha256={sha}")

    # Sidecar config JSON — enables future runtime binder shape checks
    # without hard-coding hyperparameters.
    config_path = args.config_out or Path(str(args.output) + ".config.json")
    _write_config_json(
        config_path,
        have_objective=have_objective,
        have_subjective=have_subjective,
        sample_rate=sample_rate,
    )
    _log(f"config: {config_path}")

    # Shared-storage audit sidecar — empty array when no dedup fired.
    shared_pairs_path = args.shared_pairs_out or Path(
        str(args.output) + ".shared_pairs.json"
    )
    with open(shared_pairs_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"clone": a, "original": b} for a, b in all_shared],
            f,
            indent=2,
        )
        f.write("\n")
    _log(f"shared-pairs audit: {shared_pairs_path} ({len(all_shared)} pair(s))")

    # Manifest line for CI logs / fixture pipelines (matches the
    # dnsmos_prepare_checkpoint.py format).
    print(f"{args.output.name} {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
