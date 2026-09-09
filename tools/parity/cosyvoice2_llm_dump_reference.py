#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice2_llm_reference python
"""VAST-only official Transformers reference for CosyVoice2 LLM logits.

The execution path is deliberately unreachable until the dedicated lock and
license gate are approved.  Once enabled, it uses only the official eager
``Qwen2ForCausalLM`` implementation and fixed token IDs; no CosyVoice frontend,
tokenizer download, mirror implementation, or model upload is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cosyvoice2_llm_reference"))
import preflight_gate as pinned  # noqa: E402


FORMAT = "vokra-cosyvoice2-llm-reference-v1"
TOKEN_IDS = [151643, 785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]
TOKEN_TEXT = "The quick brown fox jumps over the lazy dog."
OUTPUT_NAMES = ("true_hf_logits.npy", "token-ids.json", "diagnostics.json", "manifest.json")
REQUIRED_WRAPPER_KEYS = {
    "llm_embedding.weight",
    "speech_embedding.weight",
    "llm_decoder.weight",
    "llm_decoder.bias",
}


class ReferenceError(ValueError):
    """Reference inputs or output destination are not authenticated."""


def fail(message: str) -> None:
    raise ReferenceError(message)


def require_regular(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        fail(f"{label} must be an absolute regular non-symlink file")


def require_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        fail(f"{label} must be an absolute existing real directory")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink() or not current.is_dir():
            fail(f"{label} contains a symlink or non-directory component")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_qwen_config(path: Path) -> dict[str, Any]:
    require_regular(path, pinned.QWEN_CONFIG_PATH)
    if path.stat().st_size != pinned.QWEN_CONFIG_BYTES or sha256_file(path) != pinned.QWEN_CONFIG_SHA256 or git_blob_sha1(path) != pinned.QWEN_CONFIG_BLOB_SHA1:
        fail("Qwen config identity mismatch")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"Qwen config is not valid JSON: {error}")
    expected = {
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected.items()):
        fail("Qwen config runtime fields mismatch")
    return {"path": pinned.QWEN_CONFIG_PATH, "bytes": pinned.QWEN_CONFIG_BYTES, "sha256": pinned.QWEN_CONFIG_SHA256, "git_blob_sha1": pinned.QWEN_CONFIG_BLOB_SHA1, "verification": "ACQUIRED_AND_HASH_VERIFIED", "fields": expected}


def authenticate_source(path: Path) -> dict[str, Any]:
    require_directory(path, "CosyVoice source")

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            fail(f"source git command failed: {error}")

    if git("rev-parse", "HEAD") != pinned.SOURCE_REVISION or git("remote", "get-url", "origin") != pinned.SOURCE_REPOSITORY or git("status", "--porcelain", "--untracked-files=all"):
        fail("source revision/origin/clean checkout mismatch")
    roles: dict[str, Any] = {}
    for role, identity in pinned.SOURCE_ROLES.items():
        role_path = path / role
        require_regular(role_path, f"source role {role}")
        text = role_path.read_text(encoding="utf-8", errors="replace")
        if sha256_file(role_path) != identity["sha256"] or git_blob_sha1(role_path) != identity["blob_sha1"] or identity["marker"] not in text:
            fail(f"source role identity mismatch: {role}")
        roles[role] = {**identity, "bytes": role_path.stat().st_size}
    license_path = path / "LICENSE"
    require_regular(license_path, "CosyVoice LICENSE")
    if license_path.stat().st_size != pinned.LICENSE_BYTES or sha256_file(license_path) != pinned.LICENSE_SHA256 or git_blob_sha1(license_path) != pinned.LICENSE_BLOB_SHA1:
        fail("CosyVoice LICENSE identity mismatch")
    return {"repository": pinned.SOURCE_REPOSITORY, "revision": pinned.SOURCE_REVISION, "roles": roles, "license": {"path": "LICENSE", "bytes": pinned.LICENSE_BYTES, "sha256": pinned.LICENSE_SHA256, "git_blob_sha1": pinned.LICENSE_BLOB_SHA1, "declared": "Apache-2.0"}}


def canonical_manifest(state: dict[str, Any]) -> str:
    import struct

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        encoded = name.encode("utf-8")
        digest.update(encoded + b"\0" + struct.pack("<Q", len(tuple(tensor.shape))))
        for dimension in tuple(tensor.shape):
            digest.update(struct.pack("<Q", int(dimension)))
    return digest.hexdigest()


def validate_state_dict(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or len(state) != pinned.TENSOR_COUNT:
        fail("checkpoint state_dict tensor count mismatch")
    keys = set(state)
    if not all(isinstance(key, str) and key for key in keys):
        fail("checkpoint state_dict contains a non-string key")
    if not REQUIRED_WRAPPER_KEYS.issubset(keys):
        fail("checkpoint wrapper tensor keys are incomplete")
    if keys - REQUIRED_WRAPPER_KEYS and not all(key.startswith("llm.model.") for key in keys - REQUIRED_WRAPPER_KEYS):
        fail("checkpoint contains an unauthenticated top-level tensor")
    if canonical_manifest(state) != pinned.TENSOR_MANIFEST_SHA256:
        fail("checkpoint complete name/shape manifest mismatch")
    for name, tensor in state.items():
        if str(tensor.dtype) != "torch.float32" or tensor.device.type != "cpu" or not tensor.is_contiguous():
            fail(f"{name}: checkpoint must contain contiguous CPU F32 tensors")
    return {"tensor_count": len(state), "manifest_sha256": pinned.TENSOR_MANIFEST_SHA256}


def remove_owned_output(path: Path) -> None:
    """Remove only files created by this transaction, never recursive data."""
    for name in OUTPUT_NAMES:
        child = path / name
        try:
            if child.is_file() and not child.is_symlink():
                child.unlink()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def execute_reference(checkpoint: Path, qwen_config: Path, source: Path, output: Path, license_manifest: Path) -> dict[str, Any]:
    del license_manifest
    import numpy as np
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config_record = authenticate_qwen_config(qwen_config)
    source_record = authenticate_source(source)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    state_record = validate_state_dict(state)
    if not torch.equal(state["llm.model.lm_head.weight"], state["llm.model.model.embed_tokens.weight"]):
        fail("checkpoint tied embedding/lm_head identity is not exact")
    config = Qwen2Config.from_json_file(str(qwen_config))
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config)
    remap = {key[len("llm.model.") :]: value for key, value in state.items() if key.startswith("llm.model.")}
    missing, unexpected = model.load_state_dict(remap, strict=False)
    missing = [name for name in missing if "rotary" not in name]
    if missing or unexpected:
        fail(f"Qwen state_dict binding mismatch: missing={missing}, unexpected={unexpected}")
    model = model.float().eval()
    with torch.no_grad():
        logits = model(torch.tensor([TOKEN_IDS], dtype=torch.long)).logits[0].float().cpu().numpy()
    if logits.ndim != 2 or logits.shape != (len(TOKEN_IDS), config.vocab_size) or not np.isfinite(logits).all():
        fail("reference logits shape or finiteness mismatch")
    diagnostics = {
        "lm_head_vs_embed_max_abs_delta": float((state["llm.model.lm_head.weight"] - state["llm.model.model.embed_tokens.weight"]).abs().max().item()),
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "threads": 1, "deterministic_algorithms": True, "attention_implementation": "eager",
        "state": state_record, "source": source_record,
    }
    parent = output.parent
    require_directory(parent, "reference output parent")
    if output.exists() or output.is_symlink():
        fail("reference output directory must be absent")
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    output_created = False
    try:
        npy = temp / "true_hf_logits.npy"
        with npy.open("wb") as stream:
            np.save(stream, logits, allow_pickle=False)
        token_body = (json.dumps({"ids": TOKEN_IDS, "text": TOKEN_TEXT, "provenance": "fixed documented ids; no tokenizer download"}, sort_keys=True, indent=2) + "\n").encode()
        diag_body = (json.dumps(diagnostics, sort_keys=True, indent=2) + "\n").encode()
        artifacts = {name: {"bytes": (temp / name).stat().st_size, "sha256": sha256_file(temp / name)} for name in ("true_hf_logits.npy",)}
        (temp / "token-ids.json").write_bytes(token_body); artifacts["token-ids.json"] = {"bytes": len(token_body), "sha256": hashlib.sha256(token_body).hexdigest()}
        (temp / "diagnostics.json").write_bytes(diag_body); artifacts["diagnostics.json"] = {"bytes": len(diag_body), "sha256": hashlib.sha256(diag_body).hexdigest()}
        manifest = {"format": FORMAT, "status": "REFERENCE_READY", "component": "llm", "model": {"repository": pinned.MODEL_REPOSITORY, "revision": pinned.MODEL_REVISION, "path": pinned.MODEL_PATH, "bytes": pinned.MODEL_BYTES, "sha256": pinned.MODEL_SHA256}, "qwen_config": config_record, "source": source_record, "state": state_record, "execution": {"implementation": "transformers.Qwen2ForCausalLM", "attention": "eager", "dtype": "F32", "threads": 1, "deterministic_algorithms": True, "model_execution": "RUN", "publication": "NO_UPLOAD"}, "artifacts": artifacts}
        manifest_body = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        (temp / "manifest.json").write_bytes(manifest_body)
        if output.exists() or output.is_symlink():
            fail("reference output directory appeared during preparation")
        os.replace(temp, output)
        output_created = True
        return manifest
    except Exception:
        if output_created:
            remove_owned_output(output)
        raise
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def self_test() -> None:
    assert len(TOKEN_IDS) == 11 and all(isinstance(value, int) and value >= 0 for value in TOKEN_IDS)
    assert OUTPUT_NAMES[-1] == "manifest.json"
    assert pinned.MODEL_SHA256 == "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"
    for flag in ("--self-test", "--checkpoint", "--qwen-config", "--source", "--license-manifest", "--output"):
        try:
            cli_flag_counts([flag, flag])
        except ReferenceError:
            pass
        else:
            raise AssertionError(f"duplicate CLI option accepted: {flag}")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-llm-dump-selftest-", dir=Path(tempfile.gettempdir()).resolve()) as temp:
        root = Path(temp)
        existing = root / "existing"
        existing.mkdir()
        try:
            if existing.exists():
                raise ReferenceError("reference output directory must be absent")
        except ReferenceError:
            pass
        malformed = root / "manifest.json"
        malformed.write_text("{}", encoding="utf-8")
        data = json.loads(malformed.read_text(encoding="utf-8"))
        assert data.get("format") != FORMAT
    print("cosyvoice2_llm dump self-test: OK")


def cli_flag_counts(argv: list[str]) -> dict[str, int]:
    flags = ("--self-test", "--checkpoint", "--qwen-config", "--source", "--license-manifest", "--output")
    counts = {flag: sum(arg == flag or arg.startswith(flag + "=") for arg in argv) for flag in flags}
    if any(count > 1 for count in counts.values()):
        raise ReferenceError("duplicate CLI option is not allowed")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--qwen-config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--license-manifest", type=Path, default=HERE / "cosyvoice2_llm_reference" / "license_gate_manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        cli_flag_counts(sys.argv[1:])
    except ReferenceError as error:
        parser.error(str(error))
    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.qwen_config, args.source, args.output)):
            parser.error("--self-test accepts no execution paths")
    elif any(value is None for value in (args.checkpoint, args.qwen_config, args.source, args.output)):
        parser.error("execution requires checkpoint, qwen-config, source, and output")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        try:
            self_test()
        except Exception as error:
            print(f"cosyvoice2_llm dump self-test FAILED: {error}", file=sys.stderr)
            return 1
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64" or os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        print("VAST Linux x86_64 execution is required", file=sys.stderr)
        return 2
    root = HERE.parents[2]
    try:
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True):
            fail("clean Vokra checkout is required")
        pinned.gate(HERE / "cosyvoice2_llm_reference" / "pyproject.toml", HERE / "cosyvoice2_llm_reference" / "uv.lock", args.license_manifest)
        require_regular(args.checkpoint, pinned.MODEL_PATH)
        if args.checkpoint.stat().st_size != pinned.MODEL_BYTES or sha256_file(args.checkpoint) != pinned.MODEL_SHA256:
            fail("llm.pt identity mismatch")
        manifest = execute_reference(args.checkpoint, args.qwen_config, args.source, args.output, args.license_manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    except Exception as error:
        print(f"cosyvoice2_llm dump: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
