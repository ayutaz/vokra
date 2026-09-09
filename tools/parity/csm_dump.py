#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Write the legacy synthetic CSM fixture (M4-05 T23).

This is an **offline** plumbing tool (FR-LD-05: no Python / PyTorch ever
enters the runtime). Its ``self-test`` subcommand writes only the committed
synthetic fixture; that fixture carries no reference semantics. The official
Transformers reference is produced by ``csm_1b_dump_reference.py`` on VAST.

Pins (ADR M4-05 §D2; re-pin the exact commit SHA at T29 checkpoint hand-off)
---------------------------------------------------------------------------

- GitHub ``SesameAILabs/csm`` (models.py / generator.py — the reference
  ``Generator``); HF checkpoint ``sesame/csm-1b`` (gated acceptance = T29
  owner step); text tokenizer ``meta-llama/Llama-3.2-1B`` (gated);
  Mimi weights ``kyutai/moshiko-pytorch-bf16``
  ``tokenizer-e351c8d8-checkpoint125.safetensors`` (CC-BY 4.0).

Determinism contract (fabricated pass 禁止)
-------------------------------------------

Stages are dumped under **temperature-0 (argmax) sampling** so the code
sequence is exactly reproducible; the upstream ``generate_frame`` samples
with gumbel-max top-k whose RNG stream is *not* bit-portable, so a
stochastic dump must never be used as a parity reference. The Vokra parity
test replays the same context and compares:

- ``backbone_hidden.f32``  — ``[n_frames, backbone_dim]`` final-norm hidden
  at each generated frame position;
- ``c0_logits.f32``        — ``[n_frames, audio_vocab]``;
- ``depth_logits.f32``     — ``[n_frames, n_codebooks-1, audio_vocab]``;
- ``frame_codes.u32``      — ``[n_frames, n_codebooks]`` (discrete —
  bit-exact primary judgement, ADR §D7);
- ``decode_pcm.f32``       — ``[n_frames * frame_hop]`` Mimi-decoded audio;
- ``context.json``         — the exact prompt (speaker/text/audio refs),
  hyperparameters and stage shapes;
- ``manifest.txt``         — sha256 of every file + package/pin versions.

Usage
-----

The old ``dump`` command is retained as an explicit deprecated blocker and
never writes a reference. Use ``csm_1b_dump_reference.py`` through the VAST
validation worker for official evidence.

Self-test (no ML deps — validates the writer/manifest/reader plumbing that
the committed synthetic fixture uses)::

    uv run --frozen --project tools/parity --python 3.12 python \
        tools/parity/csm_dump.py self-test --out tests/parity/csm/self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

TOOL_VERSION = "m4-05-t23-r1"


# ---------------------------------------------------------------------------
# Shared writers (stdlib only — the self-test path must run without numpy)
# ---------------------------------------------------------------------------


def write_f32(path: Path, values, shape) -> None:
    n = 1
    for s in shape:
        n *= s
    assert len(values) == n, f"{path.name}: {len(values)} values != shape {shape}"
    with path.open("wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def write_u32(path: Path, values, shape) -> None:
    n = 1
    for s in shape:
        n *= s
    assert len(values) == n, f"{path.name}: {len(values)} values != shape {shape}"
    with path.open("wb") as f:
        f.write(struct.pack(f"<{len(values)}I", *values))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_manifest(out: Path, meta: dict) -> None:
    lines = [f"tool_version: {TOOL_VERSION}"]
    for k, v in sorted(meta.items()):
        lines.append(f"{k}: {v}")
    for p in sorted(out.iterdir()):
        if p.name == "manifest.txt":
            continue
        lines.append(f"sha256 {p.name} {sha256_of(p)}")
    (out / "manifest.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# self-test — deterministic synthetic staged files (plumbing validation)
# ---------------------------------------------------------------------------


def splitmix64(seed: int):
    """The same SplitMix64 stream vokra_core::rng uses (u64 wraparound)."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        yield z ^ (z >> 31)


def unit_f32(rng) -> float:
    # Match SplitMix64::next_unit_f32: top 24 bits / 2^24.
    return (next(rng) >> 40) / float(1 << 24)


def self_test(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n_frames, backbone_dim, audio_vocab, n_codebooks, hop = 3, 8, 11, 4, 8
    rng = splitmix64(0xC5A0)
    f = lambda n: [unit_f32(rng) * 2.0 - 1.0 for _ in range(n)]
    write_f32(out / "backbone_hidden.f32", f(n_frames * backbone_dim), (n_frames, backbone_dim))
    write_f32(out / "c0_logits.f32", f(n_frames * audio_vocab), (n_frames, audio_vocab))
    write_f32(
        out / "depth_logits.f32",
        f(n_frames * (n_codebooks - 1) * audio_vocab),
        (n_frames, n_codebooks - 1, audio_vocab),
    )
    codes = [next(rng) % audio_vocab for _ in range(n_frames * n_codebooks)]
    # Never an all-zero (EOS) frame in the synthetic fixture.
    for t in range(n_frames):
        if all(c == 0 for c in codes[t * n_codebooks : (t + 1) * n_codebooks]):
            codes[t * n_codebooks] = 1
    write_u32(out / "frame_codes.u32", codes, (n_frames, n_codebooks))
    write_f32(out / "decode_pcm.f32", f(n_frames * hop), (n_frames * hop,))
    (out / "context.json").write_text(
        json.dumps(
            {
                "kind": "self-test (synthetic — plumbing only, no reference semantics)",
                "n_frames": n_frames,
                "backbone_dim": backbone_dim,
                "audio_vocab": audio_vocab,
                "n_codebooks": n_codebooks,
                "frame_hop": hop,
                "temperature": 0.0,
                "seed": "0xC5A0 (SplitMix64)",
            },
            indent=2,
        )
        + "\n"
    )
    write_manifest(out, {"kind": "self-test"})
    # Re-read + verify every sha256 (the reader-side check parity_csm.rs
    # repeats in Rust).
    for line in (out / "manifest.txt").read_text().splitlines():
        if line.startswith("sha256 "):
            _, name, want = line.split()
            got = sha256_of(out / name)
            assert got == want, f"{name}: sha mismatch"
    print(f"self-test OK -> {out} ({len(list(out.iterdir()))} files)")
    return 0


# ---------------------------------------------------------------------------
# dump — the real upstream staged dump (owner runs after T29)
# ---------------------------------------------------------------------------


def dump(args) -> int:
    print(
        "dump: deprecated legacy route; no reference is written. Use "
        "scripts/publish/vast-ai/run-csm-1b-validation.sh, which invokes the "
        "official Transformers dumper and requires authenticated evidence.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="staged reference dump (owner, post-T29)")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--text", required=True)
    d.add_argument("--speaker", type=int, default=0)
    d.add_argument("--max-frames", type=int, default=25)
    d.add_argument("--out", required=True)
    st = sub.add_parser("self-test", help="writer/manifest plumbing self-test (no ML deps)")
    st.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "self-test":
        return self_test(Path(args.out))
    return dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
