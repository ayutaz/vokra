#!/usr/bin/env -S uv run --frozen --project tools/parity/owsm_v4_medium_1b_reference --python 3.12 python
"""Independent ESPnet OWSM frontend contract and reference dumper.

The contract mode only reads the pinned ESPnet source files.  The numerical
mode imports the official ``espnet2.asr.frontend.default.DefaultFrontend`` and
executes that implementation on a deterministic waveform; it never loads an
OWSM checkpoint.  The worker invokes this file through the dedicated
``tools/parity/owsm_v4_medium_1b_reference`` frozen project. Numerical output
is intentionally written outside the repository by the VAST worker and is not
a committed fixture until the owner approves the dependency and parity gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SOURCE_REVISION = "cccc29023d43a3f504e28df7d1324bb4eb6daedd"
SOURCE_URL = "https://github.com/espnet/espnet.git"
FORMAT = "vokra-owsm-v4-medium-1b-frontend-reference-v1"

SOURCE_FILES = {
    "default": "espnet2/asr/frontend/default.py",
    "stft": "espnet2/layers/stft.py",
    "log_mel": "espnet2/layers/log_mel.py",
    "global_mvn": "espnet2/layers/global_mvn.py",
    "enh_frontend": "espnet/nets/pytorch_backend/frontends/frontend.py",
    "default_kwargs": "espnet2/utils/get_default_kwargs.py",
}
SOURCE_BLOBS = {
    "default": "1cceef269d5d3baba80cc87942163fbb58690eae",
    "stft": "de466714c385392aba5eb63a464cc785e1da4572",
    "log_mel": "631c83d46c9d7459709dce1e394ca60b4cdbb5ef",
    "global_mvn": "27fe77f36ee5fbaf7a0319bcb001cd8bfc5aea93",
    "enh_frontend": "3e458a0c6e4ab1545ba0ce10d52d543a6f60a5b4",
    "default_kwargs": "0f11e8af43ef38cad69c530824be702dbfed5981",
}
STATS_RELATIVE = "exp/s2t_stats_raw_bpe50000/train/feats_stats.npz"
STATS_SHA256 = "00c22dba27594df8f1d8f74a491b20c6e6e8c17e92159f81dfd634f98c098654"
STATS_GIT_BLOB_SHA1 = "81dc0b816d8ccfe65c4606442e80552f2bec95ed"
STATS_BYTES = 1_786
CONFIG_RELATIVE = "exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml"
CONFIG_GIT_BLOB_SHA1 = "fbf425c85d183f9103cb5e2c84ebffb0f425a930"
CONFIG_BYTES = 494_398

# These are source expressions, not a second implementation.  Contract mode
# must fail if the upstream code changes its operation order or edge policy.
CONTRACT = {
    "default": (
        ("stft_path", r"input_stft\s*,\s*feats_lens\s*=\s*self\._compute_stft\(input\s*,\s*input_lengths\)"),
        ("power_from_complex", r"input_power\s*=\s*input_stft\.real\s*\*\*\s*2\s*\+\s*input_stft\.imag\s*\*\*\s*2"),
        ("logmel_call", r"input_feats\s*,\s*_\s*=\s*self\.logmel\(input_power\s*,\s*feats_lens\)"),
        ("return_lengths", r"return\s+input_feats\s*,\s*feats_lens"),
    ),
    "stft": (
        ("torch_stft", r"output\s*=\s*torch\.stft\(input\.float\(\),\s*\*\*stft_kwargs\)"),
        ("torch_stft_return_complex", r"stft_kwargs\[\s*[\"']return_complex[\"']\s*\]\s*=\s*True"),
        ("hann_default", r"window\s*:\s*Optional\[str\]\s*=\s*[\"']hann[\"']"),
        ("center_default", r"center\s*:\s*bool\s*=\s*True"),
        ("normalized_default", r"normalized\s*:\s*bool\s*=\s*False"),
        ("onesided_default", r"onesided\s*:\s*bool\s*=\s*True"),
        ("torch_stft_kwargs_without_pad_mode", r"stft_kwargs\s*=\s*dict\(\s*n_fft=self\.n_fft,\s*win_length=self\.win_length,\s*hop_length=self\.hop_length,\s*center=self\.center,\s*window=window,\s*normalized=self\.normalized,\s*onesided=self\.onesided,\s*\)"),
        ("center_length_pad", r"if\s+self\.center\s*:\s*pad\s*=\s*self\.n_fft\s*//\s*2"),
        ("frame_length_formula", r"torch\.div\(\s*ilens\s*-\s*self\.n_fft\s*,\s*self\.hop_length,\s*rounding_mode\s*=\s*[\"']trunc[\"']\s*\)"),
    ),
    "log_mel": (
        ("mel_matrix", r"self\.melmat|register_buffer\s*\(\s*[\"']melmat[\"']"),
        ("matrix_application", r"torch\.matmul\s*\(\s*feat\s*,\s*self\.melmat\s*\)"),
        ("clamp_floor", r"torch\.clamp\s*\(\s*mel_feat\s*,\s*min\s*=\s*1e-10\s*\)"),
        ("natural_log", r"mel_feat\.log\s*\(\s*\)"),
        ("zero_padding", r"masked_fill\s*\(.*make_pad_mask"),
    ),
    "global_mvn": (
        ("stats_count", r"stats\s*\[\s*[\"']count[\"']\s*\]"),
        ("stats_sum", r"stats\s*\[\s*[\"']sum[\"']\s*\]"),
        ("stats_sum_square", r"stats\s*\[\s*[\"']sum_square[\"']\s*\]"),
        ("mean_formula", r"mean\s*=\s*sum_v\s*/\s*count"),
        ("variance_formula", r"var\s*=\s*sum_square_v\s*/\s*count\s*-\s*mean\s*\*\s*mean"),
        ("eps_floor", r"np\.maximum\s*\(\s*var\s*,\s*eps\s*\)"),
        ("mean_application", r"x\s*-\=\s*self\.mean"),
        ("std_application", r"x\s*/=\s*self\.std"),
    ),
    "enh_frontend": (
        ("frontend_class", r"class\s+Frontend\b"),
        ("wpe_default_false", r"use_wpe\s*:\s*bool\s*=\s*False"),
        ("beamformer_default_false", r"use_beamformer\s*:\s*bool\s*=\s*False"),
        ("mono_noop_return", r"return\s+h\s*,\s*ilens\s*,\s*mask"),
    ),
    "default_kwargs": (
        ("default_kwargs_function", r"def\s+get_default_kwargs\s*\("),
        ("inspect_signature", r"inspect\.signature\(func\)\.parameters"),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_bytes(source_root: Path, relative: str) -> tuple[bytes, bool]:
    path = source_root / relative
    if path.is_file() and not path.is_symlink():
        return path.read_bytes(), True
    result = subprocess.run(
        ["git", "-C", str(source_root), "show", f"HEAD:{relative}"],
        check=True,
        capture_output=True,
    )
    return result.stdout, False


def source_git_identity(source_root: Path) -> dict[str, Any]:
    """Authenticate the checkout identity, not only a caller-provided SHA."""

    errors: list[str] = []
    identity: dict[str, Any] = {
        "origin": None,
        "head": None,
        "clean": False,
        "blobs": {},
        "expected_blobs": SOURCE_BLOBS,
    }
    try:
        top = Path(_git(source_root, "rev-parse", "--show-toplevel")).resolve()
        if top != source_root:
            errors.append(f"source checkout root mismatch: {top} != {source_root}")
        identity["origin"] = _git(source_root, "remote", "get-url", "origin")
        if identity["origin"] != SOURCE_URL:
            errors.append(f"source origin mismatch: {identity['origin']!r}")
        identity["head"] = _git(source_root, "rev-parse", "HEAD")
        if identity["head"] != SOURCE_REVISION:
            errors.append(f"source HEAD mismatch: {identity['head']!r}")
        dirty = _git(source_root, "status", "--porcelain", "--untracked-files=all")
        identity["clean"] = not dirty
        if dirty:
            errors.append("source checkout is dirty")
        for role, relative in SOURCE_FILES.items():
            observed = _git(source_root, "rev-parse", f"HEAD:{relative}")
            identity["blobs"][role] = observed
            if observed != SOURCE_BLOBS[role]:
                errors.append(f"source blob mismatch for {relative}: {observed!r}")
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"source git identity unavailable: {error}")
    identity["status"] = "AUTHENTICATED" if not errors else "BLOCKED"
    identity["errors"] = errors
    return identity


def stats_identity(stats_path: Path) -> dict[str, Any]:
    stats_path = stats_path.resolve()
    result = {
        "relative": STATS_RELATIVE,
        "path": str(stats_path),
        "expected_sha256": STATS_SHA256,
        "expected_git_blob_sha1": STATS_GIT_BLOB_SHA1,
        "expected_bytes": STATS_BYTES,
        "sha256": None,
        "git_blob_sha1": None,
        "bytes": None,
        "status": "BLOCKED",
    }
    if not stats_path.is_file() or stats_path.is_symlink():
        result["error"] = "stats path must be a regular non-symlink file"
        return result
    result["sha256"] = sha256(stats_path)
    result["git_blob_sha1"] = git_blob_sha1(stats_path)
    result["bytes"] = stats_path.stat().st_size
    if (
        result["sha256"] != STATS_SHA256
        or result["git_blob_sha1"] != STATS_GIT_BLOB_SHA1
        or result["bytes"] != STATS_BYTES
    ):
        result["error"] = "stats identity mismatch"
        return result
    result["status"] = "AUTHENTICATED"
    return result


def config_identity(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    result: dict[str, Any] = {
        "relative": CONFIG_RELATIVE,
        "path": str(config_path),
        "expected_git_blob_sha1": CONFIG_GIT_BLOB_SHA1,
        "expected_bytes": CONFIG_BYTES,
        "git_blob_sha1": None,
        "bytes": None,
        "status": "BLOCKED",
    }
    if not config_path.is_file() or config_path.is_symlink():
        result["error"] = "config path must be a regular non-symlink file"
        return result
    raw = config_path.read_bytes()
    result["git_blob_sha1"] = git_blob_sha1(config_path)
    result["bytes"] = len(raw)
    if result["git_blob_sha1"] != CONFIG_GIT_BLOB_SHA1 or result["bytes"] != CONFIG_BYTES:
        result["error"] = "config identity mismatch"
        return result
    try:
        import yaml

        value = yaml.safe_load(raw.decode("utf-8"))
    except Exception as error:
        result["error"] = f"config parse blocked: {error}"
        return result
    if not isinstance(value, dict):
        result["error"] = "config top level is not a mapping"
        return result
    frontend_conf = value.get("frontend_conf")
    normalize_conf = value.get("normalize_conf")
    expected_frontend = {"fs": 16000, "n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 128}
    if value.get("frontend") != "default" or frontend_conf != expected_frontend:
        result["error"] = "config frontend/default values are not exact"
        return result
    if value.get("normalize") != "global_mvn" or not isinstance(normalize_conf, dict):
        result["error"] = "config GlobalMVN binding is not exact"
        return result
    if normalize_conf.get("stats_file") != STATS_RELATIVE:
        result["error"] = "config stats_file is not the fixed HF relative path"
        return result
    result.update(
        {
            "status": "AUTHENTICATED",
            "frontend": {"name": value["frontend"], "conf": frontend_conf},
            "normalize": {"name": value["normalize"], "stats_file": normalize_conf["stats_file"]},
            "inner_frontend_defaults": {
                "source": "get_default_kwargs(Frontend)",
                "use_wpe": False,
                "use_beamformer": False,
                "mono_path": "Frontend.forward returns h, ilens, mask unchanged",
            },
        }
    )
    return result


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def imported_module_evidence(
    module: Any, source_root: Path, relative: str, role: str, label: str
) -> dict[str, Any]:
    module_file = getattr(module, "__file__", None)
    expected_file = (source_root / relative).resolve()
    if not isinstance(module_file, str) or Path(module_file).resolve() != expected_file:
        raise RuntimeError(f"{label} imported outside fixed source: {module_file!r}")
    observed_blob = git_blob_sha1(Path(module_file).resolve())
    if observed_blob != SOURCE_BLOBS[role]:
        raise RuntimeError(f"{label} imported source blob mismatch: {observed_blob}")
    return {
        "module": module.__name__,
        "file": str(Path(module_file).resolve()),
        "expected_file": str(expected_file),
        "git_blob_sha1": observed_blob,
        "expected_git_blob_sha1": SOURCE_BLOBS[role],
    }


def import_official_modules(source_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Import only the pinned frontend classes and restore process import state."""

    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "espnet" or name.startswith("espnet.") or name == "espnet2" or name.startswith("espnet2.")
    }
    if original_modules:
        raise RuntimeError(
            "pre-existing espnet module contamination: " + ", ".join(sorted(original_modules)[:8])
        )
    source_root = source_root.resolve()
    try:
        sys.path[:] = [str(source_root)] + [entry for entry in original_path if entry != str(source_root)]
        default_module = importlib.import_module("espnet2.asr.frontend.default")
        global_mvn_module = importlib.import_module("espnet2.layers.global_mvn")
        DefaultFrontend = default_module.DefaultFrontend
        GlobalMVN = global_mvn_module.GlobalMVN
        stft_module = importlib.import_module("espnet2.layers.stft")
        log_mel_module = importlib.import_module("espnet2.layers.log_mel")
        frontend_module = importlib.import_module(
            "espnet.nets.pytorch_backend.frontends.frontend"
        )
        default_kwargs_module = importlib.import_module("espnet2.utils.get_default_kwargs")
        expected = {
            "DefaultFrontend": (default_module, "espnet2/asr/frontend/default.py", "default"),
            "Stft": (stft_module, "espnet2/layers/stft.py", "stft"),
            "LogMel": (log_mel_module, "espnet2/layers/log_mel.py", "log_mel"),
            "GlobalMVN": (global_mvn_module, "espnet2/layers/global_mvn.py", "global_mvn"),
            "Frontend": (
                frontend_module,
                "espnet/nets/pytorch_backend/frontends/frontend.py",
                "enh_frontend",
            ),
            "get_default_kwargs": (
                default_kwargs_module,
                "espnet2/utils/get_default_kwargs.py",
                "default_kwargs",
            ),
        }
        evidence: dict[str, Any] = {}
        for label, (module, relative, role) in expected.items():
            evidence[label] = imported_module_evidence(module, source_root, relative, role, label)
        return DefaultFrontend, GlobalMVN, evidence
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "espnet" or name.startswith("espnet.") or name == "espnet2" or name.startswith("espnet2."):
                del sys.modules[name]
        sys.modules.update(original_modules)


def contract_for_source(source_root: Path, *, require_git_identity: bool = True) -> dict[str, Any]:
    import re

    source_root = source_root.resolve()
    files: dict[str, Any] = {}
    errors: list[str] = []
    source_git = source_git_identity(source_root) if require_git_identity else {
        "status": "NOT_CHECKED_IN_SELF_TEST",
        "expected_blobs": SOURCE_BLOBS,
    }
    if source_git.get("status") == "BLOCKED":
        errors.extend(source_git.get("errors", []))
    for role, relative in SOURCE_FILES.items():
        path = source_root / relative
        try:
            raw, materialized = source_bytes(source_root, relative)
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"missing or non-regular official source file: {relative}: {error}")
            continue
        text = raw.decode("utf-8")
        checks = CONTRACT[role]
        matched = [label for label, pattern in checks if re.search(pattern, text, re.MULTILINE | re.DOTALL)]
        missing = [label for label, _ in checks if label not in matched]
        if missing:
            errors.append(f"official source contract mismatch: {role}: {missing}")
        files[role] = {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "materialized": materialized,
            "git_blob_sha1": SOURCE_BLOBS[role],
            "checks": [label for label, _ in checks],
            "matched": matched,
            "missing": missing,
        }
    return {
        "format": FORMAT,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "reference_kind": "OFFICIAL_ESPnet_IMPORT_ONLY",
        "execution": "SOURCE_CONTRACT_ONLY",
        "source_git": source_git,
        "stats_binding": {
            "relative": STATS_RELATIVE,
            "sha256": STATS_SHA256,
            "git_blob_sha1": STATS_GIT_BLOB_SHA1,
            "bytes": STATS_BYTES,
            "runtime_tensors": ["normalize.mean", "normalize.std"],
            "derivation": "official GlobalMVN(stats_file, norm_means=True, norm_vars=True, eps=1e-20)",
        },
        "operation_order": ["stft", "complex_power", "melmat_matmul", "clamp_1e-10", "natural_log", "global_mvn"],
        "config": {
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 128,
            "center": True,
            "pad_mode": "torch.stft_default_reflect",
            "window": "hann",
            "normalized": False,
            "onesided": True,
            "log_base": "natural",
            "log_floor": 1.0e-10,
            "global_mvn_eps": 1.0e-20,
        },
        "source_files": files,
        "status": "SOURCE_CONTRACT_AUTHENTICATED" if not errors else "BLOCKED_SOURCE_CONTRACT",
        "errors": errors,
        "fixture_status": "NOT_GENERATED",
        "publication": "NO_UPLOAD",
    }


def write_exclusive(path: Path, payload: bytes) -> None:
    """Atomically create a file and never clobber an existing evidence file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to clobber existing evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def run_official_reference(source_root: Path, config_path: Path, stats_path: Path, output: Path) -> None:
    if os.environ.get("OWSM_ALLOW_OFFICIAL_REFERENCE") != "1":
        raise RuntimeError(
            "official numerical reference is VAST-only; set OWSM_ALLOW_OFFICIAL_REFERENCE=1 on the disposable worker"
        )
    contract = contract_for_source(source_root)
    if contract["status"] != "SOURCE_CONTRACT_AUTHENTICATED":
        raise RuntimeError(json.dumps(contract, sort_keys=True))
    config = config_identity(config_path)
    if config["status"] != "AUTHENTICATED":
        raise RuntimeError(json.dumps(config, sort_keys=True))
    stats = stats_identity(stats_path)
    if stats["status"] != "AUTHENTICATED":
        raise RuntimeError(json.dumps(stats, sort_keys=True))
    try:
        import torch
        import numpy
    except Exception as error:  # pragma: no cover - depends on VAST's pinned ESPnet env
        raise RuntimeError(f"official ESPnet import failed: {error}") from error
    process_path_before = list(sys.path)
    process_modules_before = {
        name: module
        for name, module in sys.modules.items()
        if name == "espnet" or name.startswith("espnet.") or name == "espnet2" or name.startswith("espnet2.")
    }
    DefaultFrontend, GlobalMVN, import_evidence = import_official_modules(source_root)
    process_path_after = list(sys.path)
    process_modules_after = {
        name: module
        for name, module in sys.modules.items()
        if name == "espnet" or name.startswith("espnet.") or name == "espnet2" or name.startswith("espnet2.")
    }
    if process_path_after != process_path_before or process_modules_after != process_modules_before:
        raise RuntimeError("official import did not restore sys.path/sys.modules")
    import_evidence["process_import_state_restored"] = True
    import_evidence["preexisting_module_names"] = sorted(process_modules_before)

    torch.manual_seed(0)
    # This is a synthetic PCM contract probe, not model execution or a fixture.
    pcm = torch.linspace(-0.25, 0.25, 16000, dtype=torch.float32).reshape(1, -1)
    lengths = torch.tensor([pcm.shape[1]], dtype=torch.long)
    frontend = DefaultFrontend(**config["frontend"]["conf"]).eval()
    normalize = GlobalMVN(
        stats_path,
        norm_means=True,
        norm_vars=True,
        eps=1.0e-20,
    ).eval()
    with torch.no_grad():
        features, feature_lengths = frontend(pcm, lengths)
        features, feature_lengths = normalize(features, feature_lengths)
    features = features.detach().cpu().contiguous().to(dtype=torch.float32)
    output_array = numpy.array(features.numpy(), dtype=numpy.dtype("<f4"), order="C", copy=True)
    payload = output_array.tobytes(order="C")
    payload_path = output.with_name(f"{output.name}.f32")
    input_payload_path = output.with_name(f"{output.name}.input.f32")
    input_array = numpy.array(
        pcm.detach().cpu().contiguous().to(dtype=torch.float32).numpy(),
        dtype=numpy.dtype("<f4"),
        order="C",
        copy=True,
    )
    input_payload = input_array.tobytes(order="C")
    mean = normalize.mean.detach().cpu().contiguous().to(dtype=torch.float32).numpy()
    std = normalize.std.detach().cpu().contiguous().to(dtype=torch.float32).numpy()
    metadata = {
        **contract,
        "execution": "OFFICIAL_ESPnet_DEFAULT_FRONTEND_THEN_GLOBAL_MVN",
        "status": "REFERENCE_DUMP_COMPLETE",
        "import_evidence": import_evidence,
        "config": config,
        "stats": stats,
        "input": {
            "samples": int(pcm.shape[1]),
            "shape": list(pcm.shape),
            "lengths": lengths.detach().cpu().tolist(),
            "dtype": "torch.float32",
            "seed": 0,
            "synthetic": True,
            "payload_path": str(input_payload_path),
            "payload_bytes": len(input_payload),
            "payload_format": "f32-le-row-major",
            "payload_sha256": hashlib.sha256(input_payload).hexdigest(),
        },
        "output": {
            "shape": list(features.shape),
            "lengths": feature_lengths.detach().cpu().tolist(),
            "dtype": "torch.float32",
            "payload_path": str(payload_path),
            "payload_bytes": len(payload),
            "payload_format": "f32-le-row-major",
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "global_mvn": {
            "stats_file": STATS_RELATIVE,
            "norm_means": True,
            "norm_vars": True,
            "eps": 1.0e-20,
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "mean_dtype": "torch.float32",
            "std_dtype": "torch.float32",
            "mean_finite": bool(torch.isfinite(torch.from_numpy(mean)).all()),
            "std_finite": bool(torch.isfinite(torch.from_numpy(std)).all()),
            "mean_sha256": hashlib.sha256(mean.tobytes(order="C")).hexdigest(),
            "std_sha256": hashlib.sha256(std.tobytes(order="C")).hexdigest(),
            "runtime_tensor_bindings": {
                "mean": "normalize.mean",
                "std": "normalize.std",
            },
        },
    }
    if output.exists() or payload_path.exists() or input_payload_path.exists():
        raise RuntimeError(
            f"refusing to clobber existing evidence set: {output} / {payload_path} / {input_payload_path}"
        )
    write_exclusive(input_payload_path, input_payload)
    write_exclusive(payload_path, payload)
    write_exclusive(output, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def self_test() -> None:
    import types

    with tempfile.TemporaryDirectory(prefix="owsm-frontend-reference-") as temporary:
        root = Path(temporary)
        for relative in SOURCE_FILES.values():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / SOURCE_FILES["default"]).write_text(
            "input_stft, feats_lens = self._compute_stft(input, input_lengths)\ninput_power = input_stft.real ** 2 + input_stft.imag ** 2\ninput_feats, _ = self.logmel(input_power, feats_lens)\nreturn input_feats, feats_lens\n",
            encoding="utf-8",
        )
        (root / SOURCE_FILES["stft"]).write_text(
            "window: Optional[str] = 'hann'\ncenter: bool = True\nnormalized: bool = False\nonesided: bool = True\nstft_kwargs = dict(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, center=self.center, window=window, normalized=self.normalized, onesided=self.onesided,)\nstft_kwargs['return_complex'] = True\noutput = torch.stft(input.float(), **stft_kwargs)\nif self.center:\n pad = self.n_fft // 2\nolens = torch.div(ilens - self.n_fft, self.hop_length, rounding_mode='trunc') + 1\n",
            encoding="utf-8",
        )
        (root / SOURCE_FILES["log_mel"]).write_text(
            "self.melmat\ntorch.matmul(feat, self.melmat)\ntorch.clamp(mel_feat, min=1e-10)\nmel_feat.log()\nmasked_fill(make_pad_mask(ilens, x, 1))\n",
            encoding="utf-8",
        )
        (root / SOURCE_FILES["global_mvn"]).write_text(
            "stats['count']\nstats['sum']\nstats['sum_square']\nmean = sum_v / count\nvar = sum_square_v / count - mean * mean\nnp.maximum(var, eps)\nx -= self.mean\nx /= self.std\n",
            encoding="utf-8",
        )
        (root / SOURCE_FILES["enh_frontend"]).write_text(
            "class Frontend:\n def __init__(self, use_wpe: bool = False, use_beamformer: bool = False): pass\n def forward(self, x, ilens):\n  mask = None\n  h = x\n  return h, ilens, mask\n",
            encoding="utf-8",
        )
        (root / SOURCE_FILES["default_kwargs"]).write_text(
            "import inspect\ndef get_default_kwargs(func):\n params = inspect.signature(func).parameters\n return {p.name: p.default for p in params.values()}\n",
            encoding="utf-8",
        )
        contract = contract_for_source(root, require_git_identity=False)
        assert contract["status"] == "SOURCE_CONTRACT_AUTHENTICATED", contract
        (root / SOURCE_FILES["log_mel"]).write_text("class LogMel: pass\n", encoding="utf-8")
        assert contract_for_source(root, require_git_identity=False)["status"] == "BLOCKED_SOURCE_CONTRACT"
        evidence = root / "evidence.json"
        write_exclusive(evidence, b"{}\n")
        try:
            write_exclusive(evidence, b"clobber\n")
        except RuntimeError:
            pass
        else:
            raise AssertionError("evidence writer clobbered an existing path")
        wrong_stats = root / "wrong-stats.npz"
        wrong_stats.write_bytes(b"x" * STATS_BYTES)
        assert stats_identity(wrong_stats)["status"] == "BLOCKED"
        wrong_stats_size = root / "wrong-stats-size.npz"
        wrong_stats_size.write_bytes(b"x")
        assert stats_identity(wrong_stats_size)["status"] == "BLOCKED"
        assert stats_identity(root / "missing-stats.npz")["status"] == "BLOCKED"
        fake_module = types.SimpleNamespace(
            __name__="espnet2.asr.frontend.default",
            __file__=str(root / "outside-default.py"),
        )
        try:
            imported_module_evidence(
                fake_module,
                root,
                SOURCE_FILES["default"],
                "default",
                "DefaultFrontend",
            )
        except RuntimeError as error:
            assert "outside fixed source" in str(error)
        else:
            raise AssertionError("import escape was accepted")
        contaminated = types.ModuleType("espnet")
        sys.modules["espnet"] = contaminated
        try:
            try:
                import_official_modules(root)
            except RuntimeError as error:
                assert "pre-existing espnet module contamination" in str(error)
            else:
                raise AssertionError("pre-existing module contamination was accepted")
        finally:
            sys.modules.pop("espnet", None)
    print("owsm_v4_medium_1b_frontend_reference self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--dump-official", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source, args.config, args.stats, args.contract, args.dump_official)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.source is None or (args.contract is None) == (args.dump_official is None):
        parser.error("normal mode requires --source and exactly one of --contract/--dump-official")
    if args.contract is not None:
        if args.config is not None or args.stats is not None:
            parser.error("--config/--stats are only valid with --dump-official")
        result = contract_for_source(args.source)
        write_exclusive(args.contract, (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        print(result["status"])
        return 0 if result["status"] == "SOURCE_CONTRACT_AUTHENTICATED" else 2
    if args.config is None or args.stats is None:
        parser.error("--config and --stats are required with --dump-official")
    run_official_reference(args.source, args.config, args.stats, args.dump_official)
    print("REFERENCE_DUMP_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
