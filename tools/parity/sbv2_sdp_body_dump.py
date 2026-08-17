#!/usr/bin/env python3
"""Emit a real-checkpoint SBV2 SDP-body parity fixture (VAST only).

This is the reference producer for
``crates/vokra-models/tests/sbv2_sdp_torch_parity.rs``'s
``sdp_body_matches_torch_ref`` test. It must run only on a VAST instance:
it loads an upstream SBV2 checkpoint and its torch dependencies. The script
never downloads, converts, or loads model weights on a development Mac.

Reference independence
----------------------
The output is captured from the vendored, byte-attributed MIT
``jaywalnut310/vits`` ``StochasticDurationPredictor`` implementation in
``tools/parity/vendor/vits/sdp.py``. The hook is attached to its real
``proj`` module while the upstream ``forward(reverse=True)`` body executes;
this script does not reproduce Vokra's Rust ``SbV2SDP::body`` in Python.
The hook aborts immediately after the body so the fixture isolates
``pre -> +cond(g) -> DDSConv -> proj`` from RNG and flow-inverse noise.

The generated files are deliberately gitignored real-fixture artifacts:

* ``sdp_body_hidden_seed<seed>_T<T>.f32.bin`` — row-major ``[T, d_hidden]``;
* ``sdp_body_g_seed<seed>.f32.bin`` — speaker conditioning ``[gin]``;
* ``sdp_body_seed<seed>_T<T>.f32.bin`` — reference output ``[d_hidden, T]``;
* ``sdp_body_seed<seed>_T<T>.json`` — dimensions and reference provenance.

Run on VAST after ``cd tools/parity && uv sync``:

    uv run python sbv2_sdp_body_dump.py \\
      --checkpoint /root/sbv2-checkpoint \\
      --output-dir /root/vokra/tests/fixtures/sbv2 --seed 0 --T 50

``--self-test`` only checks the filename contract and does not import torch
or inspect a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


GENERATOR_ID = "tools/parity/sbv2_sdp_body_dump.py"
GENERATOR_VERSION = 1
REFERENCE_ID = (
    "tools/parity/vendor/vits/sdp.py::StochasticDurationPredictor "
    "(jaywalnut310/vits@2e561ba58618d021b5b8323d3765880f7e0ecfdb, MIT)"
)


def artifact_paths(output_dir: Path, seed: int, text_len: int) -> dict[str, Path]:
    """Return the fixed, Rust-test-consumed artifact names."""
    stem = f"sdp_body_seed{seed}_T{text_len}"
    return {
        "hidden": output_dir / f"sdp_body_hidden_seed{seed}_T{text_len}.f32.bin",
        "speaker": output_dir / f"sdp_body_g_seed{seed}.f32.bin",
        "body": output_dir / f"{stem}.f32.bin",
        "metadata": output_dir / f"{stem}.json",
    }


def _die(message: str) -> "None":
    raise SystemExit(f"[sbv2-sdp-body-dump] {message}")


def _sha256(path: Path) -> str:
    """Return a streaming hash without mapping an entire checkpoint at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sdp_state(checkpoint: Path, torch) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Load only the upstream ``sdp.*`` tensors and record their source files."""
    from safetensors import safe_open

    if not checkpoint.is_dir():
        _die(
            f"--checkpoint {checkpoint} is not a directory; point it at the "
            "VAST-staged output of sbv2_prepare_checkpoint.py"
        )

    files = sorted(checkpoint.rglob("*.safetensors"))
    if not files:
        _die(f"no .safetensors files under {checkpoint}")
    if any("of-" in path.stem for path in files):
        _die(
            "multi-shard safetensors are unsupported for this fixture; merge "
            "them on VAST before running the dumper"
        )

    state: dict[str, object] = {}
    source_files: list[dict[str, str]] = []
    for path in files:
        # ``safe_open`` reads only the requested tensor. This matters for the
        # usual VAST staging layout, which may put multi-gigabyte BERT weights
        # beside the one SBV2 generator safetensors file this fixture needs.
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            sdp_names = [name for name in tensors.keys() if name.startswith("sdp.")]
            if not sdp_names:
                continue
            source_files.append(
                {
                    "path_relative_to_checkpoint": str(path.relative_to(checkpoint)),
                    "sha256": _sha256(path),
                }
            )
            for name in sdp_names:
                key = name.removeprefix("sdp.")
                if key in state:
                    _die(
                        f"duplicate SDP tensor {name!r} across checkpoint files; "
                        "refusing to choose one silently"
                    )
                state[key] = tensors.get_tensor(name).detach().to(dtype=torch.float32)

    if not state:
        _die("no sdp.* tensors found; this is not a supported SBV2 checkpoint")
    return state, source_files


def _require_rank3_weight(state: dict[str, object], key: str) -> tuple[int, int, int]:
    tensor = state.get(key)
    if tensor is None:
        _die(f"missing required upstream tensor sdp.{key}")
    shape = tuple(int(v) for v in tensor.shape)
    if len(shape) != 3:
        _die(f"sdp.{key} must be rank 3, got {shape}")
    return shape


def _build_upstream_sdp(state: dict[str, object], torch):
    """Instantiate the upstream MIT module and load every SDP tensor strictly."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from vendor.vits.sdp import StochasticDurationPredictor
    except ImportError as exc:
        _die(
            "vendored MIT VITS SDP import failed "
            f"({exc}); repair tools/parity/vendor/vits before generating a reference"
        )

    d_hidden, input_width, kernel = _require_rank3_weight(state, "pre.weight")
    if input_width != d_hidden or kernel != 1:
        _die(
            "sdp.pre.weight must be [d_hidden, d_hidden, 1] for the SBV2 "
            f"body fixture, got {(d_hidden, input_width, kernel)}"
        )
    cond_out, gin, cond_kernel = _require_rank3_weight(state, "cond.weight")
    if cond_out != d_hidden or cond_kernel != 1:
        _die(
            "sdp.cond.weight must be [d_hidden, gin, 1], got "
            f"{(cond_out, gin, cond_kernel)}"
        )

    conv_kernel_shape = _require_rank3_weight(state, "convs.convs1.0.weight")
    conv_kernel = conv_kernel_shape[2]
    if conv_kernel <= 0:
        _die(f"invalid DDS kernel width {conv_kernel}")

    flow_indices: set[int] = set()
    for key in state:
        parts = key.split(".", 2)
        if len(parts) >= 2 and parts[0] == "flows" and parts[1].isdigit():
            flow_indices.add(int(parts[1]))
    if not flow_indices:
        _die("sdp.flows.* tensors are absent; expected ElementwiseAffine + ConvFlows")
    ordered_indices = sorted(flow_indices)
    if ordered_indices != list(range(ordered_indices[-1] + 1)):
        _die(f"sdp.flows indices must be contiguous from 0, got {ordered_indices}")
    if len(ordered_indices) < 3 or (len(ordered_indices) - 1) % 2 != 0:
        _die(
            "sdp.flows must have upstream [EA, ConvFlow, Flip, ...] layout; "
            f"got {len(ordered_indices)} indexed modules"
        )
    n_flows = (len(ordered_indices) - 1) // 2

    model = StochasticDurationPredictor(
        d_hidden,
        d_hidden,
        conv_kernel,
        0.0,
        n_flows=n_flows,
        gin_channels=gin,
    ).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        _die(
            "upstream StochasticDurationPredictor state mismatch; "
            f"missing={list(missing)[:12]}, unexpected={list(unexpected)[:12]}. "
            "Do not drop or rename tensors in this reference path."
        )
    return model, d_hidden, gin, n_flows


def _write_tensor_f32(path: Path, tensor, torch) -> None:
    if sys.byteorder != "little":
        _die("only little-endian hosts are supported for raw f32 fixture output")
    normalized = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    path.write_bytes(normalized.view(torch.uint8).numpy().tobytes())


class _BodyCaptured(RuntimeError):
    """Internal non-error signal raised from a torch forward hook."""


def run_dump(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        _die(
            f"missing torch ({exc}); on VAST run `cd tools/parity && uv sync` "
            "then rerun through `uv run python`"
        )

    state, source_files = _load_sdp_state(args.checkpoint, torch)
    reference, d_hidden, gin, n_flows = _build_upstream_sdp(state, torch)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    # Persist inputs as raw bytes. Rust consumes these exact values rather
    # than reproducing a second RNG algorithm, so the only numerical oracle is
    # the independent upstream SDP body.
    hidden_row_major = torch.randn(
        (1, args.text_len, d_hidden), generator=generator, dtype=torch.float32
    )
    speaker = torch.randn((1, gin, 1), generator=generator, dtype=torch.float32)
    hidden_channel_major = hidden_row_major.transpose(1, 2).contiguous()
    x_mask = torch.ones((1, 1, args.text_len), dtype=torch.float32)

    captured: dict[str, object] = {}

    def capture_proj(_module, _inputs, output):
        captured["body"] = output.detach().to(dtype=torch.float32).cpu().clone()
        # Stop directly after the upstream body's final `proj`. This ensures
        # the reference does not enter random sampling or flow inversion.
        raise _BodyCaptured()

    hook = reference.proj.register_forward_hook(capture_proj)
    try:
        with torch.no_grad():
            reference(
                hidden_channel_major,
                x_mask,
                g=speaker,
                reverse=True,
                noise_scale=0.0,
            )
    except _BodyCaptured:
        pass
    finally:
        hook.remove()

    body = captured.get("body")
    if body is None:
        _die("upstream proj hook did not fire; refusing to emit an empty fixture")
    expected_shape = (1, d_hidden, args.text_len)
    if tuple(body.shape) != expected_shape:
        _die(f"captured body has shape {tuple(body.shape)}, expected {expected_shape}")

    paths = artifact_paths(args.output_dir, args.seed, args.text_len)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_tensor_f32(paths["hidden"], hidden_row_major.squeeze(0), torch)
    _write_tensor_f32(paths["speaker"], speaker.squeeze(0).squeeze(-1), torch)
    _write_tensor_f32(paths["body"], body.squeeze(0), torch)
    paths["metadata"].write_text(
        json.dumps(
            {
                "generator": GENERATOR_ID,
                "generator_version": GENERATOR_VERSION,
                "reference": REFERENCE_ID,
                "checkpoint": {
                    "path": str(args.checkpoint),
                    "sdp_tensor_source_files": source_files,
                },
                "seed": args.seed,
                "text_seq_len": args.text_len,
                "d_hidden": d_hidden,
                "gin": gin,
                "n_flows": n_flows,
                "hidden_layout": "row-major [T, d_hidden]",
                "speaker_layout": "[gin]",
                "body_layout": "channel-major [d_hidden, T]",
                "files": {name: path.name for name, path in paths.items() if name != "metadata"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[sbv2-sdp-body-dump] OK: captured independent upstream SDP body "
        f"for T={args.text_len}, d_hidden={d_hidden}, gin={gin}; wrote "
        f"{paths['body']}"
    )
    return 0


def run_self_test() -> int:
    paths = artifact_paths(Path("fixtures"), 0, 50)
    expected = {
        "hidden": "sdp_body_hidden_seed0_T50.f32.bin",
        "speaker": "sdp_body_g_seed0.f32.bin",
        "body": "sdp_body_seed0_T50.f32.bin",
        "metadata": "sdp_body_seed0_T50.json",
    }
    got = {name: path.name for name, path in paths.items()}
    if got != expected:
        _die(f"artifact filename contract drifted: got {got}, expected {expected}")
    print("sbv2_sdp_body_dump.py self-test: OK (artifact contract)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="VAST-staged SBV2 safetensors directory from sbv2_prepare_checkpoint.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Gitignored SBV2 fixture directory, normally tests/fixtures/sbv2",
    )
    parser.add_argument("--seed", type=int, default=0, help="torch CPU generator seed (default: 0)")
    parser.add_argument("--T", dest="text_len", type=int, default=50, help="text length (default: 50)")
    parser.add_argument("--self-test", action="store_true", help="validate the no-dependency artifact contract")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.text_len <= 0:
        parser.error("--T must be positive")
    if not args.self_test:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --self-test is used")
        if args.output_dir is None:
            parser.error("--output-dir is required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    return run_dump(args)


if __name__ == "__main__":
    sys.exit(main())
