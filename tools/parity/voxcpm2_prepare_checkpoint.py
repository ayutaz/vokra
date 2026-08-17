#!/usr/bin/env python3
"""Prepare the complete pinned VoxCPM2-2B checkpoint for Vokra conversion.

The upstream ``openbmb/VoxCPM2`` release is a two-file weight bundle:

* ``model.safetensors`` contains the LM, local encoder, and LocDiT weights.
* ``audiovae.pth`` contains the separately loaded AudioVAE V2 state dict.

The upstream loader imports the second state dict under ``audio_vae.`` before
calling ``load_state_dict``.  Converting only ``model.safetensors`` therefore
produces a success-shaped but non-decodable artifact.  This VAST-only sidecar
reproduces that namespace operation with ``torch.load(..., weights_only=True)``,
drops only the pinned integer ``decoder.sr_bin_boundaries`` buffer (the same
values are emitted as GGUF metadata), and writes one safetensors file.

The release contract is deliberately strict: revision, input hashes, tensor
counts, dtypes, dimensionality, config axes, and tokenizer hash are pinned.
Unknown non-float state, a 5-D tensor, a namespace collision, or a missing VAE
is a hard failure.  Python is run through the repository's uv project; the
actual 4.96-GB preparation belongs on VAST, never on the M1 iMac.

Usage::

    uv run --project tools/parity python tools/parity/voxcpm2_prepare_checkpoint.py \
      --snapshot-dir /path/to/openbmb--VoxCPM2/snapshot \
      --output /root/scratchpad/staging/voxcpm2-2b/complete.safetensors \
      --manifest /root/scratchpad/staging/voxcpm2-2b/prepare-manifest.json

Use ``--self-test`` for bounded synthetic coverage.  It never downloads or
loads the real checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UPSTREAM_REPO = "openbmb/VoxCPM2"
UPSTREAM_REVISION = "bffb3df5a29440629464e5e839f4d214c8714c3d"

MODEL_SHA256 = "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"
AUDIOVAE_SHA256 = "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"
CONFIG_SHA256 = "405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c"
TOKENIZER_SHA256 = "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20"

MODEL_TENSOR_COUNT = 577
AUDIOVAE_STATE_COUNT = 312
AUDIOVAE_FLOAT_COUNT = 311
COMPLETE_TENSOR_COUNT = MODEL_TENSOR_COUNT + AUDIOVAE_FLOAT_COUNT

BOUNDARY_KEY = "decoder.sr_bin_boundaries"
BOUNDARIES = [20_000, 30_000, 40_000]

MODEL_SENTINELS = {
    "base_lm.embed_tokens.weight": (73_448, 2_048),
    "base_lm.layers.27.self_attn.q_proj.weight": (2_048, 2_048),
    "feat_encoder.encoder.layers.11.self_attn.q_proj.weight": (2_048, 1_024),
    "stop_head.weight": (2, 2_048),
}

AUDIOVAE_SENTINELS = {
    "encoder.fc_mu.weight_g": (64, 1, 1),
    "encoder.fc_logvar.weight_g": (64, 1, 1),
    "decoder.model.0.bias": (64,),
    "decoder.sr_cond_model.7.bias_embed.weight": (4, 64),
}

CONFIG_EXPECTATIONS = {
    "architecture": "voxcpm2",
    "dtype": "bfloat16",
    "lm_config.hidden_size": 2_048,
    "lm_config.intermediate_size": 6_144,
    "lm_config.num_hidden_layers": 28,
    "lm_config.num_attention_heads": 16,
    "lm_config.num_key_value_heads": 2,
    "lm_config.kv_channels": 128,
    "lm_config.vocab_size": 73_448,
    "patch_size": 4,
    "feat_dim": 64,
    "residual_lm_num_layers": 8,
    "residual_lm_no_rope": True,
    "encoder_config.num_layers": 12,
    "encoder_config.kv_channels": 128,
    "dit_config.num_layers": 12,
    "dit_config.kv_channels": 128,
    "dit_config.mean_mode": False,
    "audio_vae_config.latent_dim": 64,
    "audio_vae_config.sr_bin_boundaries": BOUNDARIES,
    "audio_vae_config.sample_rate": 16_000,
    "audio_vae_config.out_sample_rate": 48_000,
    "max_length": 8_192,
}


class PrepError(RuntimeError):
    """A fail-closed checkpoint contract violation."""


@dataclass(frozen=True)
class Contract:
    model_sha256: str
    audiovae_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    model_tensor_count: int
    audiovae_state_count: int
    audiovae_float_count: int
    model_sentinels: dict[str, tuple[int, ...]]
    audiovae_sentinels: dict[str, tuple[int, ...]]

    @property
    def complete_tensor_count(self) -> int:
        return self.model_tensor_count + self.audiovae_float_count


RELEASE_CONTRACT = Contract(
    model_sha256=MODEL_SHA256,
    audiovae_sha256=AUDIOVAE_SHA256,
    config_sha256=CONFIG_SHA256,
    tokenizer_sha256=TOKENIZER_SHA256,
    model_tensor_count=MODEL_TENSOR_COUNT,
    audiovae_state_count=AUDIOVAE_STATE_COUNT,
    audiovae_float_count=AUDIOVAE_FLOAT_COUNT,
    model_sentinels=MODEL_SENTINELS,
    audiovae_sentinels=AUDIOVAE_SENTINELS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PrepError(f"{label} missing: {path}")


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PrepError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )
    return actual


def nested_get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PrepError(f"config.json missing pinned field {dotted!r}")
        current = current[part]
    return current


def validate_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise PrepError(f"{path}: expected a JSON object")
    for dotted, expected in CONFIG_EXPECTATIONS.items():
        actual = nested_get(config, dotted)
        if actual != expected:
            raise PrepError(
                f"config field {dotted!r}: expected {expected!r}, got {actual!r}"
            )
    return config


def validate_tokenizer(path: Path) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepError(f"cannot parse tokenizer {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise PrepError(f"{path}: tokenizer JSON must be a non-empty object")


def inspect_main(path: Path, contract: Contract) -> dict[str, tuple[tuple[int, ...], str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - operator environment guard
        raise PrepError("safetensors missing; run through `uv run --project tools/parity`") from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        meta = {
            name: (tuple(handle.get_slice(name).get_shape()), str(handle.get_slice(name).get_dtype()))
            for name in names
        }
    if len(meta) != contract.model_tensor_count:
        raise PrepError(
            f"model.safetensors tensor count: expected {contract.model_tensor_count}, got {len(meta)}"
        )
    bad_dtype = [(name, dtype) for name, (_, dtype) in meta.items() if dtype != "BF16"]
    if bad_dtype:
        raise PrepError(f"main checkpoint contains non-BF16 tensors: {bad_dtype[:8]}")
    too_wide = [(name, shape) for name, (shape, _) in meta.items() if len(shape) > 4]
    if too_wide:
        raise PrepError(f"main checkpoint contains >4D tensors unsupported by GGUF: {too_wide[:8]}")
    for name, expected_shape in contract.model_sentinels.items():
        actual = meta.get(name)
        if actual is None:
            raise PrepError(f"main checkpoint missing sentinel {name!r}")
        if actual[0] != expected_shape:
            raise PrepError(
                f"main sentinel {name!r}: expected shape {expected_shape}, got {actual[0]}"
            )
    return meta


def load_audiovae(path: Path, contract: Contract) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - operator environment guard
        raise PrepError("torch missing; run through `uv run --project tools/parity`") from exc

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - translate external checkpoint failures
        raise PrepError(f"torch.load({path}, weights_only=True) failed: {exc}") from exc
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise PrepError("audiovae.pth must be a dict containing `state_dict`")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict):
        raise PrepError("audiovae.pth `state_dict` must be a dict")
    if len(state) != contract.audiovae_state_count:
        raise PrepError(
            f"AudioVAE state count: expected {contract.audiovae_state_count}, got {len(state)}"
        )

    floating: dict[str, Any] = {}
    dropped: list[dict[str, Any]] = []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise PrepError(f"AudioVAE state {name!r} is {type(tensor).__name__}, not Tensor")
        if tensor.dtype.is_floating_point:
            if tensor.dtype != torch.float32:
                raise PrepError(f"AudioVAE tensor {name!r} must be float32, got {tensor.dtype}")
            if tensor.ndim > 4:
                raise PrepError(
                    f"AudioVAE tensor {name!r} has {tensor.ndim} dimensions; GGUF supports <=4"
                )
            floating[name] = tensor.detach().contiguous().to("cpu")
            continue
        if (
            name == BOUNDARY_KEY
            and tensor.dtype == torch.int32
            and tuple(tensor.shape) == (3,)
            and tensor.tolist() == BOUNDARIES
        ):
            dropped.append(
                {
                    "name": name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "values": tensor.tolist(),
                    "reason": "duplicated exactly by vokra.vae_continuous.sr_bin_boundaries metadata",
                }
            )
            continue
        raise PrepError(
            f"unexpected non-float AudioVAE tensor {name!r}: dtype={tensor.dtype}, "
            f"shape={tuple(tensor.shape)}"
        )

    if len(floating) != contract.audiovae_float_count:
        raise PrepError(
            f"AudioVAE float count: expected {contract.audiovae_float_count}, got {len(floating)}"
        )
    if len(dropped) != 1:
        raise PrepError(f"expected exactly one pinned AudioVAE metadata buffer, got {len(dropped)}")
    for name, expected_shape in contract.audiovae_sentinels.items():
        tensor = floating.get(name)
        if tensor is None:
            raise PrepError(f"AudioVAE missing sentinel {name!r}")
        if tuple(tensor.shape) != expected_shape:
            raise PrepError(
                f"AudioVAE sentinel {name!r}: expected shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )

    ptr_counts = Counter(t.untyped_storage().data_ptr() for t in floating.values())
    shared = sum(count - 1 for count in ptr_counts.values() if count > 1)
    if shared:
        raise PrepError(f"AudioVAE contains {shared} shared-storage aliases; refuse ambiguous merge")
    return floating, dropped


def validate_output(path: Path, contract: Contract) -> dict[str, int]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        dtypes = Counter(str(handle.get_slice(name).get_dtype()) for name in names)
        max_ndim = max((len(handle.get_slice(name).get_shape()) for name in names), default=0)
    if len(names) != contract.complete_tensor_count:
        raise PrepError(
            f"merged output count: expected {contract.complete_tensor_count}, got {len(names)}"
        )
    if sum(name.startswith("audio_vae.") for name in names) != contract.audiovae_float_count:
        raise PrepError("merged output lost or duplicated the audio_vae namespace")
    if max_ndim > 4:
        raise PrepError(f"merged output contains a {max_ndim}D tensor")
    return dict(sorted(dtypes.items()))


def prepare(
    snapshot_dir: Path,
    output: Path,
    manifest: Path,
    contract: Contract,
    *,
    check_release_config: bool,
) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - operator environment guard
        raise PrepError("safetensors missing; run through `uv run --project tools/parity`") from exc

    model_path = snapshot_dir / "model.safetensors"
    audiovae_path = snapshot_dir / "audiovae.pth"
    config_path = snapshot_dir / "config.json"
    tokenizer_path = snapshot_dir / "tokenizer.json"
    for path, label in [
        (model_path, "main weight"),
        (audiovae_path, "AudioVAE weight"),
        (config_path, "config"),
        (tokenizer_path, "tokenizer"),
    ]:
        require_file(path, label)
    if output.resolve() in {model_path.resolve(), audiovae_path.resolve()}:
        raise PrepError("--output must not overwrite an upstream input")
    if output.exists():
        raise PrepError(f"--output already exists: {output}; remove or choose a new staging path")

    hashes = {
        "model.safetensors": require_hash(model_path, contract.model_sha256, "model.safetensors"),
        "audiovae.pth": require_hash(audiovae_path, contract.audiovae_sha256, "audiovae.pth"),
        "config.json": require_hash(config_path, contract.config_sha256, "config.json"),
        "tokenizer.json": require_hash(tokenizer_path, contract.tokenizer_sha256, "tokenizer.json"),
    }
    if check_release_config:
        validate_config(config_path)
        validate_tokenizer(tokenizer_path)

    main_meta = inspect_main(model_path, contract)
    vae, dropped = load_audiovae(audiovae_path, contract)

    main = load_file(str(model_path), device="cpu")
    combined: dict[str, Any] = {name: main[name] for name in sorted(main)}
    collisions: list[str] = []
    for name in sorted(vae):
        prefixed = f"audio_vae.{name}"
        if prefixed in combined:
            collisions.append(prefixed)
        else:
            combined[prefixed] = vae[name]
    if collisions:
        raise PrepError(f"audio_vae namespace collisions: {collisions[:8]}")
    if len(combined) != contract.complete_tensor_count:
        raise PrepError(
            f"combined tensor count: expected {contract.complete_tensor_count}, got {len(combined)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temp_output.exists():
        temp_output.unlink()
    try:
        save_file(
            combined,
            str(temp_output),
            metadata={
                # safetensors accepts a string map but serializes that map
                # through a Rust HashMap. Multiple keys can therefore change
                # header order between identical runs. Keep one canonical
                # JSON value so the complete release artifact is byte-stable.
                "vokra_preparation": json.dumps(
                    {
                        "prep_tool": "tools/parity/voxcpm2_prepare_checkpoint.py",
                        "upstream_repo": UPSTREAM_REPO,
                        "upstream_revision": UPSTREAM_REVISION,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )
        dtype_counts = validate_output(temp_output, contract)
        output_sha256 = sha256_file(temp_output)
        os.replace(temp_output, output)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    report: dict[str, Any] = {
        "schema": 1,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_revision": UPSTREAM_REVISION,
        "inputs": {
            name: {
                "sha256": digest,
                "bytes": (snapshot_dir / name).stat().st_size,
            }
            for name, digest in hashes.items()
        },
        "main": {
            "tensors": len(main_meta),
            "dtype_counts": {"BF16": len(main_meta)},
        },
        "audiovae": {
            "state_entries": contract.audiovae_state_count,
            "float_tensors": len(vae),
            "dtype_counts": {"F32": len(vae)},
            "namespace": "audio_vae.",
            "dropped_metadata_buffers": dropped,
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": output_sha256,
            "tensors": contract.complete_tensor_count,
            "dtype_counts": dtype_counts,
        },
    }
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def self_test() -> None:
    import torch
    from safetensors.torch import save_file

    cases = 0
    with tempfile.TemporaryDirectory(prefix="voxcpm2-prep-self-test-") as raw:
        root = Path(raw)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        model = {
            "base_lm.embed_tokens.weight": torch.zeros((3, 2), dtype=torch.bfloat16),
            "base_lm.norm.weight": torch.ones((2,), dtype=torch.bfloat16),
        }
        save_file(model, str(snapshot / "model.safetensors"))
        vae_state = {
            "encoder.fc_mu.weight_g": torch.ones((2, 1, 1), dtype=torch.float32),
            "decoder.sr_cond_model.7.bias_embed.weight": torch.ones((4, 2), dtype=torch.float32),
            BOUNDARY_KEY: torch.tensor(BOUNDARIES, dtype=torch.int32),
        }
        torch.save({"metadata": {}, "state_dict": vae_state}, snapshot / "audiovae.pth")
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text('{"model": {}}\n', encoding="utf-8")

        tiny = Contract(
            model_sha256=sha256_file(snapshot / "model.safetensors"),
            audiovae_sha256=sha256_file(snapshot / "audiovae.pth"),
            config_sha256=sha256_file(snapshot / "config.json"),
            tokenizer_sha256=sha256_file(snapshot / "tokenizer.json"),
            model_tensor_count=2,
            audiovae_state_count=3,
            audiovae_float_count=2,
            model_sentinels={
                "base_lm.embed_tokens.weight": (3, 2),
                "base_lm.norm.weight": (2,),
            },
            audiovae_sentinels={
                "encoder.fc_mu.weight_g": (2, 1, 1),
                "decoder.sr_cond_model.7.bias_embed.weight": (4, 2),
            },
        )
        first = prepare(
            snapshot,
            root / "complete-1.safetensors",
            root / "manifest-1.json",
            tiny,
            check_release_config=False,
        )
        cases += 1
        if first["output"]["tensors"] != 4:
            raise AssertionError("complete synthetic merge did not retain four float tensors")

        second = prepare(
            snapshot,
            root / "complete-2.safetensors",
            root / "manifest-2.json",
            tiny,
            check_release_config=False,
        )
        cases += 1
        if first["output"]["sha256"] != second["output"]["sha256"]:
            raise AssertionError("identical inputs did not produce byte-identical safetensors")

        bad_state = dict(vae_state)
        bad_state["unexpected.counter"] = torch.tensor([1], dtype=torch.int64)
        bad_path = root / "bad-audiovae.pth"
        torch.save({"metadata": {}, "state_dict": bad_state}, bad_path)
        bad_contract = Contract(
            **{
                **tiny.__dict__,
                "audiovae_sha256": sha256_file(bad_path),
                "audiovae_state_count": 4,
            }
        )
        try:
            load_audiovae(bad_path, bad_contract)
        except PrepError as exc:
            if "unexpected non-float" not in str(exc):
                raise AssertionError(f"wrong unknown-int diagnostic: {exc}") from exc
        else:
            raise AssertionError("unknown integer state was accepted")
        cases += 1

        bad_main = root / "bad-5d.safetensors"
        save_file({"five.d": torch.zeros((1, 1, 1, 1, 1), dtype=torch.bfloat16)}, str(bad_main))
        bad_main_contract = Contract(
            **{
                **tiny.__dict__,
                "model_sha256": sha256_file(bad_main),
                "model_tensor_count": 1,
                "model_sentinels": {"five.d": (1, 1, 1, 1, 1)},
            }
        )
        try:
            inspect_main(bad_main, bad_main_contract)
        except PrepError as exc:
            if ">4D" not in str(exc):
                raise AssertionError(f"wrong 5D diagnostic: {exc}") from exc
        else:
            raise AssertionError("5D tensor was accepted")
        cases += 1

    print(f"voxcpm2_prepare_checkpoint --self-test: OK ({cases} cases)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    missing = [
        flag
        for flag, value in [
            ("--snapshot-dir", args.snapshot_dir),
            ("--output", args.output),
            ("--manifest", args.manifest),
        ]
        if value is None
    ]
    if missing:
        raise PrepError(f"required arguments missing: {', '.join(missing)}")
    report = prepare(
        args.snapshot_dir,
        args.output,
        args.manifest,
        RELEASE_CONTRACT,
        check_release_config=True,
    )
    print(
        "voxcpm2_prepare_checkpoint: "
        f"{report['output']['tensors']} tensors, "
        f"{report['output']['bytes']} bytes, "
        f"sha256={report['output']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrepError as exc:
        print(f"voxcpm2_prepare_checkpoint: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
