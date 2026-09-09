#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice2_hift_reference python
"""Dump the official CosyVoice2 HiFT CPU oracle for downstream parity work.

The source checkout, config, and checkpoint are all authenticated before the
upstream classes are imported.  This utility is VAST-only: it performs no
downloads and refuses to overwrite an existing reference directory.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
MODEL_REPOSITORY = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
MODEL_PATH = "hift.pt"
MODEL_BYTES = 83_390_254
MODEL_SHA256 = "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879"
CONFIG_PATH = "cosyvoice2.yaml"
CONFIG_BYTES = 7_330
CONFIG_SHA256 = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959"
CONFIG_BLOB_SHA1 = "bc19267bbfd373c9a760b7667a74349ddd487db1"
SOURCE_ROLES = {
    "cosyvoice/hifigan/generator.py": "326a1a70ae7707662939c20493b3a8e4b0906216",
    "cosyvoice/hifigan/f0_predictor.py": "5797c31aada757ac7ef65a70ff8ee21867a25df8",
    "cosyvoice/transformer/activation.py": "8cea54816385d3b6585ccc2417bc71630d578177",
    "cosyvoice/utils/common.py": "6f5a3dd8b7ae99601783c3a4ed91b3b64270fab3",
}
EXPECTED_TENSOR_COUNT = 328
EXPECTED_MANIFEST_SHA256 = "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c"
FORMAT = "vokra-cosyvoice2-hift-reference-v1"
SOURCE_ORIGIN_FORMS = frozenset(
    {
        "https://github.com/FunAudioLLM/CosyVoice",
        "https://github.com/FunAudioLLM/CosyVoice/",
        SOURCE_REPOSITORY,
        f"{SOURCE_REPOSITORY}/",
    }
)


class DumpError(ValueError):
    """An input or upstream execution result failed closed."""


def authorize_license(path: Path) -> dict[str, Any]:
    gate_path = Path(__file__).with_name("cosyvoice2_hift_reference") / "preflight_gate.py"
    spec = importlib.util.spec_from_file_location("cosyvoice2_hift_preflight", gate_path)
    if spec is None or spec.loader is None:
        raise DumpError("preflight gate module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.gate(gate_path.with_name("pyproject.toml"), gate_path.with_name("uv.lock"), path)
    except Exception as error:
        raise DumpError(f"license preflight blocked: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DumpError(f"{label} must be a regular non-symlink file: {path}")


def run_git(source: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DumpError(f"git authentication command failed: {error}") from error
    return result.stdout.strip()


def normalize_source_origin(origin: str) -> str:
    if origin not in SOURCE_ORIGIN_FORMS:
        raise DumpError(f"source origin is not canonical HTTPS: {origin!r}")
    return origin.rstrip("/").removesuffix(".git")


def authenticate_source(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir():
        raise DumpError(f"source checkout must be a directory, not a symlink: {source}")
    if run_git(source, "status", "--porcelain", "--untracked-files=all"):
        raise DumpError("source checkout is not clean")
    head = run_git(source, "rev-parse", "HEAD")
    if head != SOURCE_REVISION:
        raise DumpError(f"source HEAD mismatch: {head}")
    remote = run_git(source, "config", "--get", "remote.origin.url")
    if normalize_source_origin(remote) != normalize_source_origin(SOURCE_REPOSITORY):
        raise DumpError(f"source origin mismatch: {remote!r}")
    records: dict[str, Any] = {}
    for relative, expected_blob in SOURCE_ROLES.items():
        path = source / relative
        regular_file(path, f"source role {relative}")
        actual_blob = git_blob_sha1(path)
        if actual_blob != expected_blob:
            raise DumpError(f"source role blob mismatch: {relative}")
        records[relative] = {"git_blob_sha1": actual_blob, "sha256": sha256_file(path)}
    return {"repository": SOURCE_REPOSITORY, "revision": head, "clean": True, "roles": records}


def authenticate_file(path: Path, label: str, bytes_expected: int, sha_expected: str) -> dict[str, Any]:
    regular_file(path, label)
    size = path.stat().st_size
    if size != bytes_expected:
        raise DumpError(f"{label} byte size mismatch: {size}")
    digest = sha256_file(path)
    if digest != sha_expected:
        raise DumpError(f"{label} SHA256 mismatch")
    return {"path": path.name, "bytes": size, "sha256": digest}


def authenticate_config(path: Path) -> dict[str, Any]:
    regular_file(path, "config")
    if path.name != CONFIG_PATH:
        raise DumpError(f"config filename must be {CONFIG_PATH}")
    size = path.stat().st_size
    digest = sha256_file(path)
    blob = git_blob_sha1(path)
    if (size, digest, blob) != (CONFIG_BYTES, CONFIG_SHA256, CONFIG_BLOB_SHA1):
        raise DumpError("cosyvoice2.yaml does not match the registered config snapshot")
    return {"path": CONFIG_PATH, "bytes": size, "sha256": digest, "git_blob_sha1": blob}


def tensor_manifest_digest(state: dict[str, torch.Tensor]) -> str:
    canonical = bytearray()
    for name, tensor in sorted(state.items()):
        canonical.extend(name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            canonical.extend(struct.pack("<Q", dimension))
    return hashlib.sha256(canonical).hexdigest()


def is_plain_state_dict(value: object) -> bool:
    return type(value) in (dict, collections.OrderedDict)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise DumpError(f"torch.load(weights_only=True) failed: {error}") from error
    if not is_plain_state_dict(state) or len(state) != EXPECTED_TENSOR_COUNT:
        raise DumpError("checkpoint is not the exact plain 328-tensor state dict")
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise DumpError("checkpoint contains a non-tensor state-dict entry")
    if any(value.dtype != torch.float32 or value.device.type != "cpu" for value in state.values()):
        raise DumpError("checkpoint contains a non-CPU-F32 tensor")
    if tensor_manifest_digest(state) != EXPECTED_MANIFEST_SHA256:
        raise DumpError("checkpoint name/shape manifest digest mismatch")
    return state


def build_generator(source: Path, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(source))
    try:
        from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
        from cosyvoice.hifigan.generator import HiFTGenerator
    except Exception as error:
        raise DumpError(f"official HiFT imports failed: {error}") from error
    predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
    model = HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=predictor,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise DumpError(f"official HiFT state-dict binding failed: {error}") from error
    if not isinstance(model.f0_predictor, ConvRNNF0Predictor):
        raise DumpError("official ConvRNNF0Predictor was not attached")
    model.eval()
    return model


def write_tensor(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise DumpError(f"oracle output is not F32: {path.name}")
    if tensor.device.type != "cpu" or not torch.isfinite(tensor).all().item():
        raise DumpError(f"oracle output is not finite CPU data: {path.name}")
    array = tensor.detach().contiguous().numpy()
    raw = array.tobytes(order="C")
    path.write_bytes(raw)
    return {"file": path.name, "dtype": "F32", "shape": list(array.shape), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require_vast_environment() -> None:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise DumpError("VAST guard requires VOKRA_PUBLISH_ON_VAST=1")
    if sys.platform != "linux" or platform.machine().lower() != "x86_64":
        raise DumpError("dumper is restricted to Linux x86_64 VAST workers")


def dump(source: Path, checkpoint: Path, config: Path, output: Path, license_manifest: Path) -> None:
    if output.exists() or output.is_symlink():
        raise DumpError(f"refusing to overwrite existing output: {output}")
    require_vast_environment()
    authorization = authorize_license(license_manifest)
    global np, scipy, torch
    import numpy as np
    import scipy
    import torch

    if checkpoint.name != MODEL_PATH:
        raise DumpError(f"checkpoint filename must be {MODEL_PATH}")
    source_record = authenticate_source(source)
    model_record = authenticate_file(checkpoint, MODEL_PATH, MODEL_BYTES, MODEL_SHA256)
    config_record = authenticate_config(config)
    state = load_checkpoint(checkpoint)
    model = build_generator(source, state)
    torch.manual_seed(20260906)
    random.seed(20260906)
    np.random.seed(20260906)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    mel = torch.linspace(-0.25, 0.25, steps=80 * 8, dtype=torch.float32).reshape(1, 80, 8)
    original_rand = torch.rand
    original_randn_like = torch.randn_like

    def zero_rand(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(*args, **kwargs)

    def zero_randn_like(input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.zeros_like(input_tensor, **kwargs)

    try:
        torch.rand = zero_rand  # type: ignore[assignment]
        torch.randn_like = zero_randn_like  # type: ignore[assignment]
        with torch.inference_mode():
            result = model({"speech_feat": mel.transpose(1, 2)}, torch.device("cpu"))
    finally:
        torch.rand = original_rand  # type: ignore[assignment]
        torch.randn_like = original_randn_like  # type: ignore[assignment]
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise DumpError("official HiFT forward did not return (PCM, F0)")
    pcm, f0 = result
    if tuple(pcm.shape) != (1, 3_840) or tuple(f0.shape) != (1, 8):
        raise DumpError(f"unexpected official output shapes: PCM={tuple(pcm.shape)}, F0={tuple(f0.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    claimed = False
    try:
        try:
            output.mkdir()
            claimed = True
        except FileExistsError as error:
            raise DumpError(f"refusing to replace output claimed concurrently: {output}") from error
        input_record = write_tensor(output / "mel.f32", mel)
        pcm_record = write_tensor(output / "pcm.f32", pcm)
        f0_record = write_tensor(output / "f0.f32", f0)
        manifest = {
            "format": FORMAT,
            "status": "AUTHENTICATED_REFERENCE",
            "publication": "NO_UPLOAD",
            "source": source_record,
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, **model_record},
            "config": config_record,
            "license_manifest_sha256": authorization["license_manifest_sha256"],
            "approval_scope_sha256": authorization["approval_scope_sha256"],
            "project_sha256": authorization["project_sha256"],
            "uv_lock_sha256": authorization["lock_sha256"],
            "checkpoint_tensor_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "input": {
                **input_record,
                "seed": 20260906,
                "formula": "torch.linspace(-0.25,0.25,640).reshape(1,80,8)",
            },
            "outputs": {"pcm": pcm_record, "f0": f0_record},
            "execution": {
                "device": "cpu",
                "torch_deterministic_algorithms": True,
                "threads": 1,
                "entropy_override": "torch.rand and torch.randn_like -> zeros only during official forward",
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "uv_lock_sha256": sha256_file(Path(__file__).with_name("cosyvoice2_hift_reference") / "uv.lock"),
            },
        }
        # The manifest is deliberately written last: its presence marks a
        # complete directory, while pcm/f0 writes may be cleaned on failure.
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        if claimed:
            shutil.rmtree(output, ignore_errors=True)
        raise


def self_test() -> None:
    assert SOURCE_ROLES["cosyvoice/transformer/activation.py"] == "8cea54816385d3b6585ccc2417bc71630d578177"
    assert MODEL_BYTES == 83_390_254 and CONFIG_BYTES == 7_330
    for origin in SOURCE_ORIGIN_FORMS:
        assert normalize_source_origin(origin) == "https://github.com/FunAudioLLM/CosyVoice"
    for origin in ("git@github.com:FunAudioLLM/CosyVoice.git", "https://github.com/FunAudioLLM/CosyVoice?x=1"):
        try:
            normalize_source_origin(origin)
        except DumpError:
            pass
        else:
            raise AssertionError("non-canonical source origin was accepted")
    assert is_plain_state_dict({})
    assert is_plain_state_dict(collections.OrderedDict())
    assert not is_plain_state_dict(collections.defaultdict(dict))
    print("cosyvoice2_hift dumper self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--license-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.source or not args.checkpoint or not args.config or not args.output or not args.license_manifest:
            parser.error("--source, --checkpoint, --config, --output, and --license-manifest are required")
        dump(args.source, args.checkpoint, args.config, args.output, args.license_manifest)
        print(json.dumps({"status": "PASS", "output": args.output.as_posix()}))
        return 0
    except (DumpError, OSError, RuntimeError) as error:
        print(f"cosyvoice2_hift dumper: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
