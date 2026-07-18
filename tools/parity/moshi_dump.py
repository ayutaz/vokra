#!/usr/bin/env python3
"""Dump Moshi staged reference fixtures (M4-06 T15/T07).

This is an **offline** tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). It drives the pinned upstream implementation and writes the
staged intermediate tensors that ``crates/vokra-models/tests/
parity_moshi.rs`` compares against (env-gated on
``VOKRA_MOSHI_PARITY_DIR``). CI never runs Python — fixtures are
file-based (csm_dump.py / mimi_dump.py precedent).

Pins (ADR M4-06 §D2; re-pin the exact commit SHA at the T29 hand-off)
---------------------------------------------------------------------

- GitHub ``kyutai-labs/moshi`` (``moshi/models/loaders.py`` `_lm_kwargs` /
  ``lm.py`` `LMGen`); HF checkpoint ``kyutai/moshiko-pytorch-bf16``
  ``model.safetensors`` (CC-BY 4.0 — the T02 manifest fixture
  ``tests/parity/moshi/moshiko_tensor_manifest.json`` pins its 355-tensor
  shape table); text tokenizer ``tokenizer_spm_32k_3.model`` (same repo);
  Mimi weights ``tokenizer-e351c8d8-checkpoint125.safetensors`` (shared
  with CSM / M4-04 — its parity fixtures are the mimi_dump.py /
  csm_dump.py assets, not duplicated here).

Determinism contract (fabricated pass 禁止)
-------------------------------------------

Stages are dumped under ``use_sampling=False`` (argmax on both the text
and audio channels) so the token streams are exactly reproducible — the
Vokra comparison drives its greedy mode against them. A stochastic dump
must never be used as a parity reference (RNG streams are not
bit-portable across frameworks).

Fixture layout (little-endian, shapes in context.json)
------------------------------------------------------

- ``context.json``       — config snapshot + stage shapes
- ``user_codes.u32``     — ``[n_steps, n_user]`` user (mic) codes fed
- ``input_tokens.u32``   — ``[n_steps, n_channels]`` the gathered rows the
  temporal transformer actually consumed (post delay-ring / initial-token
  substitution) — lets the Rust side pin the backbone independently of
  the ring plumbing
- ``backbone_hidden.f32``— ``[n_steps, d]`` out_norm-applied hidden states
- ``text_logits.f32``    — ``[n_steps, text_card]``
- ``text_tokens.u32``    — ``[n_steps]`` argmax text stream
- ``frame_codes.u32``    — ``[n_emitted, dep_q]`` undelayed own audio
- ``manifest.txt``       — ``sha256 <name> <hex>`` per file

self-test mode
--------------

``python3 tools/parity/moshi_dump.py self-test --out
tests/parity/moshi/self-test`` writes a **synthetic, stdlib-only**
fixture that pins the file/manifest *format* (committed; carries no
reference semantics — the shapes mirror `MoshiConfig::tiny_for_tests`).

real mode (owner, post-T29 weight sourcing)
-------------------------------------------

``python3 tools/parity/moshi_dump.py real --out <dir> --steps 12``
requires the parity venv (torch + the moshi package) and the downloaded
checkpoint; it hooks ``LMGen`` to record the staged tensors above and is
deliberately import-guarded so running it without the venv fails with an
actionable message instead of a half-written fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def _write(path: Path, blob: bytes, manifest: list[str]) -> None:
    path.write_bytes(blob)
    manifest.append(f"sha256 {path.name} {hashlib.sha256(blob).hexdigest()}")


def _pack_f32(values) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _pack_u32(values) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def self_test(out_dir: Path) -> None:
    """Synthetic format pin — mirrors MoshiConfig::tiny_for_tests shapes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps, n_user, n_channels = 3, 2, 5
    d, text_card, dep_q = 16, 13, 2
    max_delay = 1
    n_emitted = n_steps - max_delay
    manifest: list[str] = []

    # Deterministic synthetic values (a fixed LCG — NOT a reference).
    state = 0x2545F491
    def nxt() -> int:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state

    user = [nxt() % 9 for _ in range(n_steps * n_user)]
    inputs = [nxt() % 14 for _ in range(n_steps * n_channels)]
    hidden = [((nxt() % 2000) - 1000) / 997.0 for _ in range(n_steps * d)]
    tlogits = [((nxt() % 2000) - 1000) / 991.0 for _ in range(n_steps * text_card)]
    ttoks = [nxt() % text_card for _ in range(n_steps)]
    codes = [1 + nxt() % 8 for _ in range(n_emitted * dep_q)]

    _write(out_dir / "user_codes.u32", _pack_u32(user), manifest)
    _write(out_dir / "input_tokens.u32", _pack_u32(inputs), manifest)
    _write(out_dir / "backbone_hidden.f32", _pack_f32(hidden), manifest)
    _write(out_dir / "text_logits.f32", _pack_f32(tlogits), manifest)
    _write(out_dir / "text_tokens.u32", _pack_u32(ttoks), manifest)
    _write(out_dir / "frame_codes.u32", _pack_u32(codes), manifest)
    context = {
        "kind": "self-test (synthetic format pin — no reference semantics)",
        "n_steps": n_steps,
        "n_user": n_user,
        "n_channels": n_channels,
        "d_model": d,
        "text_card": text_card,
        "dep_q": dep_q,
        "max_delay": max_delay,
        "n_emitted": n_emitted,
    }
    blob = (json.dumps(context, indent=1, sort_keys=True) + "\n").encode()
    _write(out_dir / "context.json", blob, manifest)
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"self-test fixture written to {out_dir}", file=sys.stderr)


def real(
    out_dir: Path,
    steps: int,
    hf_repo: str,
    checkpoint: Path | None = None,
    mimi_weight: Path | None = None,
    mimi_codebooks: int = 8,
    lm_layers: int | None = None,
    depformer_layers: int | None = None,
) -> None:
    """Owner-side staged dump against the pinned upstream (module docs).

    With ``--checkpoint`` (plus ``--mimi-weight``) the dump runs over
    LOCAL safetensors instead of the HF hub — the truncated-checkpoint
    leg the parity-moshi-real workflow can afford: the FULL-7B fp32 cast
    needs ~43 GiB (14.32 GiB BF16 dict + fp32 copies), so a 16 GB
    machine / ubuntu-latest dumps the ``moshi_truncate.py`` artifact with
    ``--lm-layers/--depformer-layers`` overriding the layer counts to
    match (all other ``_lm_kwargs`` stay the upstream 7B constants;
    dtype=float32 is the same exact BF16→f32 widening the converter's
    runtime performs). Campaign-2 (2026-07-17) validated this recipe:
    tm=2/dt=2 → 11/11 emitted frames bit-exact vs the Vokra runtime.
    """
    try:
        import torch  # noqa: F401
        from moshi.models import loaders  # type: ignore
    except ImportError as e:  # pragma: no cover - owner-side path
        raise SystemExit(
            "moshi_dump.py real: the parity venv (torch + the `moshi` package) "
            f"is required — {e}. This is the T29 owner step; CI and the "
            "runtime never import it."
        ) from e

    import torch
    from moshi.models import LMGen

    if checkpoint is not None:
        if mimi_weight is None:
            raise SystemExit(
                "moshi_dump.py real: --checkpoint needs --mimi-weight (the "
                "tokenizer-e351c8d8-checkpoint125.safetensors Mimi codec file) "
                "— refusing to silently fall back to the hub"
            )
        overrides = {}
        if lm_layers is not None:
            overrides["num_layers"] = lm_layers
        if depformer_layers is not None:
            overrides["depformer_num_layers"] = depformer_layers
        mimi = loaders.get_mimi(
            str(mimi_weight), device="cpu", num_codebooks=mimi_codebooks
        )
        lm = loaders.get_moshi_lm(
            str(checkpoint),
            device="cpu",
            dtype=torch.float32,
            lm_kwargs_overrides=overrides or None,
        )
    else:
        info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
        mimi = info.get_mimi(device="cpu")
        lm = info.get_moshi(device="cpu", dtype=torch.float32)
    lm_gen = LMGen(lm, use_sampling=False)  # argmax both channels

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    n_user = lm.n_q - lm.dep_q
    n_channels = lm.num_codebooks

    # A deterministic synthetic "user" stream: Mimi-encode a fixed chirp.
    sr = mimi.sample_rate
    frame = int(sr / mimi.frame_rate)
    t = torch.arange(steps * frame, dtype=torch.float32) / sr
    chirp = (0.25 * torch.sin(2 * torch.pi * (220.0 + 40.0 * t) * t)).view(1, 1, -1)
    with torch.no_grad(), mimi.streaming(1):
        user_codes = []
        for s in range(steps):
            chunk = chirp[:, :, s * frame : (s + 1) * frame]
            user_codes.append(mimi.encode(chunk)[:, :n_user, :])

    inputs_rows: list[int] = []
    hidden_rows: list[float] = []
    tlogit_rows: list[float] = []
    ttoks: list[int] = []
    frame_rows: list[int] = []

    orig_forward_text = lm.forward_text

    def hooked_forward_text(seq, *args, **kwargs):
        inputs_rows.extend(int(x) for x in seq[0, :, 0].tolist())
        out, tl = orig_forward_text(seq, *args, **kwargs)
        hidden_rows.extend(float(x) for x in out[0, 0].tolist())
        tlogit_rows.extend(float(x) for x in tl[0, 0, 0].tolist())
        return out, tl

    lm.forward_text = hooked_forward_text  # type: ignore
    with torch.no_grad(), lm_gen.streaming(1):
        for s in range(steps):
            out = lm_gen.step(user_codes[s])
            # Text token written this step is out-of-band; recover it from
            # the emitted (undelayed) frame instead.
            if out is not None:
                ttoks.append(int(out[0, 0, 0]))
                frame_rows.extend(int(x) for x in out[0, 1:, 0].tolist())

    user_flat = [int(x) for uc in user_codes for x in uc[0, :, 0].tolist()]
    _write(out_dir / "user_codes.u32", _pack_u32(user_flat), manifest)
    _write(out_dir / "input_tokens.u32", _pack_u32(inputs_rows), manifest)
    _write(out_dir / "backbone_hidden.f32", _pack_f32(hidden_rows), manifest)
    _write(out_dir / "text_logits.f32", _pack_f32(tlogit_rows), manifest)
    _write(out_dir / "text_tokens.u32", _pack_u32(ttoks), manifest)
    _write(out_dir / "frame_codes.u32", _pack_u32(frame_rows), manifest)
    if checkpoint is not None:
        kind = (
            f"real (local {Path(checkpoint).name}"
            + (
                f", layer overrides tm={lm_layers}/dt={depformer_layers}"
                if (lm_layers is not None or depformer_layers is not None)
                else ""
            )
            + ")"
        )
    else:
        kind = f"real ({hf_repo})"
    context = {
        "kind": kind,
        "n_steps": steps,
        "n_user": n_user,
        "n_channels": n_channels,
        "d_model": lm.dim,
        "text_card": lm.text_card,
        "dep_q": lm.dep_q,
        "max_delay": max(lm.delays),
        "n_emitted": len(ttoks),
        "note": "text_tokens/frame_codes are the *undelayed emitted* stream "
        "(LMGen.step output); backbone stages are per *step*.",
    }
    blob = (json.dumps(context, indent=1, sort_keys=True) + "\n").encode()
    _write(out_dir / "context.json", blob, manifest)
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"real fixture written to {out_dir}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    st = sub.add_parser("self-test")
    st.add_argument("--out", type=Path, required=True)
    re = sub.add_parser("real")
    re.add_argument("--out", type=Path, required=True)
    re.add_argument("--steps", type=int, default=12)
    re.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    re.add_argument(
        "--checkpoint", type=Path, default=None,
        help="LOCAL LM safetensors (e.g. the moshi_truncate.py output) "
             "instead of the hub download; requires --mimi-weight",
    )
    re.add_argument(
        "--mimi-weight", type=Path, default=None,
        help="LOCAL Mimi codec safetensors "
             "(tokenizer-e351c8d8-checkpoint125.safetensors)",
    )
    re.add_argument(
        "--mimi-codebooks", type=int, default=8,
        help="Mimi codebooks for the local path (n_user; moshiko = 8)",
    )
    re.add_argument(
        "--lm-layers", type=int, default=None,
        help="lm_kwargs num_layers override (truncated checkpoints)",
    )
    re.add_argument(
        "--depformer-layers", type=int, default=None,
        help="lm_kwargs depformer_num_layers override (truncated checkpoints)",
    )
    args = p.parse_args()
    if args.mode == "self-test":
        self_test(args.out)
    else:
        real(
            args.out,
            args.steps,
            args.hf_repo,
            checkpoint=args.checkpoint,
            mimi_weight=args.mimi_weight,
            mimi_codebooks=args.mimi_codebooks,
            lm_layers=args.lm_layers,
            depformer_layers=args.depformer_layers,
        )


if __name__ == "__main__":
    main()
