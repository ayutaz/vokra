#!/usr/bin/env -S uv run --frozen --project tools/parity/htdemucs_multi --python 3.12 python
"""VAST-only official HT-Demucs reference report generator.

The script intentionally emits a JSON tap manifest plus selected raw
little-endian f32 tap files, not model or audio artifacts.  It
loads the authenticated package with ``weights_only=True`` and then
instantiates only the pinned upstream ``HTDemucs`` class.  There is no pickle
fallback and no import before all identities have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SOURCE_REVISION = "e976d93ecc3865e5757426930257e200846a520a"
WEIGHT_ORDER = ("f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d", "5c90dfd2")
FT_IDS = WEIGHT_ORDER[:4]
SIX_IDS = WEIGHT_ORDER[4:]
MAX_INTERMEDIATE_TAP_ELEMENTS = 1 << 20
CONFIGS = {
    "htdemucs_ft": {"ids": FT_IDS, "sources": 4, "config": "htdemucs_ft.yaml"},
    "htdemucs_6s": {"ids": SIX_IDS, "sources": 6, "config": "htdemucs_6s.yaml"},
}


def load_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_path(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.exists():
        raise ValueError(f"{label} must be an existing absolute non-symlink path")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} must not contain symlink or relative path components")
    return resolved


def absent_path(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be an absent absolute non-symlink path")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError(f"{label} parent must be an existing non-symlink directory")
    if path.resolve(strict=False) != path:
        raise ValueError(f"{label} must not contain symlink or relative path components")
    return path


def reject_overlap(candidate: Path, protected: list[tuple[str, Path]]) -> None:
    candidate_text = str(candidate)
    for label, path in protected:
        path_text = str(path)
        if candidate_text == path_text or candidate_text.startswith(path_text + os.sep) or path_text.startswith(candidate_text + os.sep):
            raise ValueError(f"output path overlaps {label}")


def verify_inputs(source: Path, weights: Path, variant: str, gate: dict[str, Any]) -> dict[str, Any]:
    if gate.get("status") != "APPROVED_FOR_VAST_REFERENCE" or gate.get("publication") != "NO_UPLOAD" or gate.get("blockers"):
        raise ValueError("license/provenance gate is not approved for reference execution")
    if variant not in CONFIGS:
        raise ValueError(f"unsupported variant: {variant}")
    upstream = gate.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("revision") != SOURCE_REVISION:
        raise ValueError("upstream revision gate mismatch")
    source = existing_path(source, "source-dir")
    weights = existing_path(weights, "weights-dir")
    if not (source / ".git").is_dir():
        raise ValueError("source checkout must be a real git directory")
    import subprocess

    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_REVISION:
        raise ValueError("source HEAD does not match the gate")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True):
        raise ValueError("source checkout is dirty")
    rows = gate.get("weights")
    if not isinstance(rows, list):
        raise ValueError("weight gate rows are missing")
    if len(rows) != len(WEIGHT_ORDER) or [row.get("model_id") for row in rows] != list(WEIGHT_ORDER):
        raise ValueError("weight gate member order drifted")
    by_id = {row.get("model_id"): row for row in rows if isinstance(row, dict)}
    for model_id in CONFIGS[variant]["ids"]:
        row = by_id.get(model_id)
        if not isinstance(row, dict):
            raise ValueError(f"weight gate row missing: {model_id}")
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"weight filename is not a plain basename: {model_id}")
        path = weights / filename
        if path.resolve(strict=False) != path:
            raise ValueError(f"weight path contains a symlink or relative component: {filename}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"weight is missing or symlinked: {path}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"weight SHA-256 mismatch: {path.name}")
    config_path = source / "demucs" / "remote" / CONFIGS[variant]["config"]
    expected_config = upstream["config_sha256"][CONFIGS[variant]["config"]]
    if not config_path.is_file() or sha256(config_path) != expected_config:
        raise ValueError("variant config identity mismatch")
    import subprocess
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip().removesuffix("/").removesuffix(".git")
    if origin != "https://github.com/facebookresearch/demucs":
        raise ValueError("source origin does not match the gate")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True):
        raise ValueError("source checkout is dirty")
    role_blobs: dict[str, str] = {}
    for role, expected_blob in upstream.get("roles", {}).items():
        role_path = source / role
        if not role_path.is_file() or role_path.is_symlink() or role_path.resolve(strict=True) != role_path:
            raise ValueError(f"source role is missing or symlinked: {role}")
        actual_blob = subprocess.check_output(["git", "-C", str(source), "rev-parse", f"HEAD:{role}"], text=True).strip()
        if actual_blob != expected_blob:
            raise ValueError(f"source role drifted: {role}")
        role_blobs[role] = actual_blob
    return {"repository": origin, "revision": SOURCE_REVISION, "origin": origin, "dirty": False, "role_blobs": role_blobs}


def f32_tap(value: Any, name: str, raw_dir: Path, selected: set[str]) -> dict[str, Any] | None:
    import torch

    if not isinstance(value, torch.Tensor):
        return None
    data = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if sys.byteorder != "little":
        raise ValueError("reference raw-f32 contract requires a little-endian VAST host")
    full_count = int(data.numel())
    truncated = not name.endswith(".stems") and full_count > MAX_INTERMEDIATE_TAP_ELEMENTS
    raw_data = data.reshape(-1)[:MAX_INTERMEDIATE_TAP_ELEMENTS] if truncated else data.reshape(-1)
    raw = raw_data.numpy().astype("<f4", copy=False).tobytes(order="C")
    digest = hashlib.sha256(raw).hexdigest()
    filename = f"{len(selected):04d}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.f32"
    raw_path = raw_dir / filename
    if raw_path.exists() or raw_path.is_symlink():
        raise ValueError(f"raw tap filename collision: {filename}")
    with raw_path.open("xb") as stream:
        stream.write(raw)
    selected.add(name)
    return {"name": name, "shape": [int(axis) for axis in data.shape], "count": full_count, "raw_count": int(raw_data.numel()), "bytes": len(raw), "sha256": digest, "raw_file": filename, "raw_offset": 0, "truncated": truncated}


def tensor_taps(value: Any, name: str, raw_dir: Path, selected: set[str]) -> list[dict[str, Any]]:
    if name in selected:
        return []
    if (tap := f32_tap(value, name, raw_dir, selected)) is not None:
        return [tap]
    if isinstance(value, (tuple, list)):
        taps: list[dict[str, Any]] = []
        for index, child in enumerate(value):
            taps.extend(tensor_taps(child, f"{name}[{index}]", raw_dir, selected))
        return taps
    if isinstance(value, dict):
        taps = []
        for key in sorted(value):
            taps.extend(tensor_taps(value[key], f"{name}.{key}", raw_dir, selected))
        return taps
    return []


def load_official_model(source: Path, checkpoint: Path, model_class: Any, model_id: str) -> Any:
    import numpy as np
    import torch
    from fractions import Fraction

    expected = {"demucs.htdemucs.HTDemucs", "fractions.Fraction"}
    if model_id != "5c90dfd2":
        expected |= {"numpy.core.multiarray.scalar", "numpy.dtype"}
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None or set(scanner(str(checkpoint))) != expected:
        raise ValueError(f"static-global set mismatch for {checkpoint.name}")
    safe: list[Any] = [(model_class, "demucs.htdemucs.HTDemucs"), Fraction]
    if "numpy.core.multiarray.scalar" in expected:
        numpy_core = getattr(np, "_core", None)
        if numpy_core is None:
            raise ValueError("NumPy _core implementation is unavailable")
        safe.extend([(numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype(np.float64))])
    with torch.serialization.safe_globals(safe):
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if type(package) is not dict or tuple(package) != ("klass", "args", "kwargs", "state", "training_args", "metrics"):
        raise ValueError(f"package schema mismatch: {checkpoint.name}")
    if package["klass"] is not model_class:
        raise ValueError(f"package class mismatch: {checkpoint.name}")
    if type(package["args"]) is not tuple or type(package["kwargs"]) is not dict or type(package["state"]) is not dict:
        raise ValueError(f"package field types mismatch: {checkpoint.name}")
    model = model_class(*package["args"], **package["kwargs"])
    model.load_state_dict(package["state"], strict=True)
    model.eval()
    return model


def run(source: Path, weights: Path, fixture: Path, fixture_sha: str, variant: str, output: Path, raw_dir: Path, gate_path: Path) -> None:
    gate_path = existing_path(gate_path, "gate")
    source = existing_path(source, "source-dir")
    weights = existing_path(weights, "weights-dir")
    fixture = existing_path(fixture, "audio-fixture")
    output = absent_path(output, "output")
    raw_dir = absent_path(raw_dir, "raw-dir")
    reject_overlap(output, [("source-dir", source), ("weights-dir", weights), ("fixture", fixture), ("gate", gate_path)])
    reject_overlap(raw_dir, [("source-dir", source), ("weights-dir", weights), ("fixture", fixture), ("gate", gate_path), ("output", output)])
    gate = load_json(gate_path)
    source_evidence = verify_inputs(source, weights, variant, gate)
    if sha256(fixture) != fixture_sha:
        raise ValueError("audio fixture SHA-256 mismatched")
    if len(fixture_sha) != 64 or any(char not in "0123456789abcdef" for char in fixture_sha):
        raise ValueError("audio fixture SHA-256 must be lowercase 64-hex")

    dependency_path = existing_path(Path(__file__).with_name("dependency_audit.json"), "dependency-audit")
    pyproject_path = existing_path(Path(__file__).with_name("pyproject.toml"), "pyproject")
    lock_path = existing_path(Path(__file__).with_name("uv.lock"), "uv.lock")
    dependency = load_json(dependency_path)
    dependency_gate = gate.get("dependency_audit")
    if (dependency.get("status") != "APPROVED" or dependency.get("blockers")
            or not isinstance(dependency_gate, dict)
            or dependency_gate.get("status") != "APPROVED"
            or dependency_gate.get("lock_sha256") != sha256(lock_path)
            or dependency_gate.get("package_rows_sha256") != dependency.get("package_rows_sha256")
            or dependency_gate.get("license_rows_sha256") != dependency.get("license_rows_sha256")):
        raise ValueError("dependency audit is not approved")
    if dependency_gate.get("pyproject_sha256") != sha256(pyproject_path):
        raise ValueError("pyproject digest is not bound to gate")

    sys.path.insert(0, str(source))
    htdemucs = importlib.import_module("demucs.htdemucs")
    apply = importlib.import_module("demucs.apply")
    audio = importlib.import_module("demucs.audio")
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(fixture))
    waveform = audio.convert_audio(waveform, sample_rate, 44100, 2)
    waveform = waveform.unsqueeze(0)
    config_path = source / "demucs" / "remote" / CONFIGS[variant]["config"]
    yaml = importlib.import_module("yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("models") != list(CONFIGS[variant]["ids"]):
        raise ValueError("official model config member order drifted")
    if variant == "htdemucs_ft":
        config_weights = config.get("weights")
        expected_weights = [[1.0 if i == j else 0.0 for j in range(len(FT_IDS))] for i in range(len(FT_IDS))]
        if config_weights != expected_weights:
            raise ValueError("official FT BagOfModels weights drifted")
    elif "weights" in config:
        raise ValueError("6s config unexpectedly defines bag weights")
    bag_weights = config.get("weights") if variant == "htdemucs_ft" else None
    model_rows = {row["model_id"]: row for row in gate["weights"]}
    raw_dir.mkdir()
    taps: list[dict[str, Any]] = []
    models: list[Any] = []
    hook_contracts: list[dict[str, Any]] = []
    for model_id in CONFIGS[variant]["ids"]:
        model = load_official_model(source, weights / model_rows[model_id]["filename"], htdemucs.HTDemucs, model_id)
        models.append(model)
        hooks = []
        selected: set[str] = set()
        hook_seen: set[str] = set()
        hook_labels: set[str] = set()
        hook_calls: dict[str, int] = {}

        def capture_hook(label: str):
            hook_labels.add(label)
            def capture(_module: Any, _inputs: Any, value: Any) -> None:
                hook_calls[label] = hook_calls.get(label, 0) + 1
                if label in hook_seen:
                    return
                hook_seen.add(label)
                taps.extend(tensor_taps(value, f"{model_id}.{label}", raw_dir, selected))
            return capture

        for prefix, module_list in (("encoder", model.encoder), ("tencoder", model.tencoder), ("decoder", model.decoder), ("tdecoder", model.tdecoder)):
            selected_indices = {0, len(module_list) - 1} if module_list else set()
            for index in sorted(selected_indices):
                module = module_list[index]
                hooks.append(module.register_forward_hook(capture_hook(f"{prefix}.{index}")))
        if model.crosstransformer is not None:
            hooks.append(model.crosstransformer.register_forward_hook(capture_hook("crosstransformer")))
        with torch.no_grad():
            spec = model._spec(waveform)
            taps.extend(tensor_taps(spec.real, f"{model_id}.stft.re", raw_dir, selected))
            taps.extend(tensor_taps(spec.imag, f"{model_id}.stft.im", raw_dir, selected))
            stems = apply.apply_model(
                model, waveform, shifts=0, split=True, overlap=0.25,
                transition_power=1.0, progress=False, device="cpu", num_workers=0,
                segment=None,
            )
        if hook_seen != hook_labels or any(hook_calls.get(label, 0) < 1 for label in hook_labels):
            raise ValueError(f"selected hook did not run: {model_id}")
        taps.extend(tensor_taps(stems, f"{model_id}.stems", raw_dir, selected))
        for hook in hooks:
            hook.remove()
        hook_contracts.append({"model_id": model_id, "first_invocation_only": True, "calls": hook_calls, "selected": sorted(hook_seen)})
        del model, stems
    bag = apply.BagOfModels(models, bag_weights)
    effective_bag_weights = [list(map(float, row)) for row in bag.weights]
    with torch.no_grad():
        bag_stems = apply.apply_model(
            bag, waveform, shifts=0, split=True, overlap=0.25,
            transition_power=1.0, progress=False, device="cpu", num_workers=0,
            segment=None,
        )
    bag_taps: set[str] = set()
    taps.extend(tensor_taps(bag_stems, "bag.stems", raw_dir, bag_taps))
    expected_terminals = {f"{model_id}.stems" for model_id in CONFIGS[variant]["ids"]} | {"bag.stems"}
    if not expected_terminals.issubset({tap["name"] for tap in taps}):
        raise ValueError("terminal stem taps are incomplete")
    del bag, bag_stems, models
    report = {
        "format": "vokra-htdemucs-multi-reference-report-v1",
        "status": "REPORT_ONLY",
        "publication": "NO_UPLOAD",
        "source_revision": SOURCE_REVISION,
        "variant": variant,
        "source_count": CONFIGS[variant]["sources"],
        "audio_fixture": {"path": str(fixture), "sha256": fixture_sha, "resampled_rate": 44100, "channels": 2},
        "raw_f32": {"directory": str(raw_dir), "dtype": "f32", "endianness": "little"},
        "gate": {"path": str(gate_path), "sha256": sha256(gate_path), "status": gate["status"], "publication": gate["publication"]},
        "dependency_audit": {"path": str(dependency_path), "sha256": sha256(dependency_path), "status": dependency["status"], "package_rows_sha256": dependency["package_rows_sha256"], "license_rows_sha256": dependency["license_rows_sha256"]},
        "pyproject": {"path": str(pyproject_path), "sha256": sha256(pyproject_path), "status": "PINNED"},
        "uv_lock": {"path": str(lock_path), "sha256": sha256(lock_path), "status": "LOCK_IDENTITY_OK", "gate_sha256": gate["dependency_audit"]["lock_sha256"]},
        "provenance": {
            "source": source_evidence,
            "config": {
                "path": str(config_path.relative_to(source)),
                "sha256": gate["upstream"]["config_sha256"][CONFIGS[variant]["config"]],
                "models": list(CONFIGS[variant]["ids"]),
            },
            "checkpoints": [
                {
                    "model_id": model_id,
                    "filename": model_rows[model_id]["filename"],
                    "sha256": model_rows[model_id]["sha256"],
                    "tensor_count": model_rows[model_id]["tensor_count"],
                    "parameter_count": model_rows[model_id]["parameter_count"],
                }
                for model_id in CONFIGS[variant]["ids"]
            ],
        },
        "contracts": {
            "members": [{"model_id": model_id, "terminal_tap": f"{model_id}.stems"} for model_id in CONFIGS[variant]["ids"]],
            "intermediate_tap_selection": {"hook_first_invocation_only": True, "max_elements": MAX_INTERMEDIATE_TAP_ELEMENTS, "members": hook_contracts},
            "bag": {
                "terminal_tap": "bag.stems",
                "models": list(CONFIGS[variant]["ids"]),
                "weights": effective_bag_weights,
                "apply_model": {"shifts": 0, "split": True, "overlap": 0.25, "transition_power": 1.0, "segment": None},
            },
        },
        "taps": taps,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--audio-fixture", type=Path)
    parser.add_argument("--audio-sha256")
    parser.add_argument("--variant", choices=sorted(CONFIGS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--gate", type=Path, default=Path(__file__).with_name("license_gate_manifest.json"))
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source_dir, args.weights_dir, args.audio_fixture, args.audio_sha256, args.variant, args.output, args.raw_dir)):
            parser.error("--self-test accepts no execution options")
        assert tuple(CONFIGS["htdemucs_ft"]["ids"]) == FT_IDS
        assert tuple(CONFIGS["htdemucs_6s"]["ids"]) == SIX_IDS
        assert CONFIGS["htdemucs_ft"]["sources"] == 4 and CONFIGS["htdemucs_6s"]["sources"] == 6
        assert MAX_INTERMEDIATE_TAP_ELEMENTS == 1 << 20
        assert "truncated" in f32_tap.__code__.co_varnames
        source = Path(__file__).read_text(encoding="utf-8")
        assert source.index("dependency_path =") < source.index("sys.path.insert")
        assert "effective_bag_weights" in source
        assert '"gate_sha256"' in source
        print("htdemucs multi reference dumper self-test: PASS")
        return 0
    required = {
        "--source-dir": args.source_dir,
        "--weights-dir": args.weights_dir,
        "--audio-fixture": args.audio_fixture,
        "--audio-sha256": args.audio_sha256,
        "--variant": args.variant,
        "--output": args.output,
        "--raw-dir": args.raw_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required options: " + ", ".join(missing))
    try:
        run(args.source_dir, args.weights_dir, args.audio_fixture, args.audio_sha256, args.variant, args.output, args.raw_dir, args.gate)
    except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError, ImportError) as error:
        print(f"htdemucs reference report BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"wrote report-only evidence: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
