#!/usr/bin/env -S uv run --frozen --project tools/parity/dia_1_6b_reference python
"""Authenticate the pinned Dia source contract without loading model weights.

The executable oracle in this file is the official source itself.  The worker
loads only ``dia.audio`` and extracts the small pure methods
``Dia._encode_text`` and ``_sample_next_token`` from the authenticated
``dia/model.py`` AST.  No checkpoint, DAC payload, or public GGUF is accepted
or opened here.  The resulting packet is structural/source evidence only; it
does not claim numerical model parity or PCM readiness.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SOURCE_REPOSITORY = "https://github.com/nari-labs/dia.git"
SOURCE_REVISION = "2811af1c5f476b1f49f4744fabf56cf352be21e5"
FORMAT = "vokra-dia-1-6b-source-contract-v1"
SOURCE_ROLE_BLOBS = {
    "LICENSE": "483d716cc886695f19971a99658c59851a8a2866",
    "dia/audio.py": "5c1947103bc0d95255d97618c699fa0a18993beb",
    "dia/config.py": "09c6d136a41e0296483d2617061d4261cbf4c42c",
    "dia/layers.py": "f9aed506b25e99d053dd71d6def7a0bd33075ace",
    "dia/model.py": "a3b0f9730a810fa170019511a2696e7f813090de",
    "dia/state.py": "172ec52c7c344781aad0552a6cddd6e5f1933894",
    "pyproject.toml": "dd844dd2fb0ab0c016520c4b070beaa7c159e3e1",
}
RUST_FILES = (
    "crates/vokra-models/src/dia/tokenizer.rs",
    "crates/vokra-models/src/dia/forward.rs",
    "crates/vokra-models/src/dia/mod.rs",
)
DELAY_PATTERN = [0, 8, 9, 10, 11, 12, 13, 14, 15]
SPEAKER_MARKERS = {"[S1]": 1, "[S2]": 2}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def run_git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def authenticate_source(source: Path) -> dict[str, Any]:
    if not source.is_dir() or run_git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("official Dia source revision is not authenticated")
    origin = run_git(source, "remote", "get-url", "origin")
    if origin.removesuffix(".git").rstrip("/") != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("official Dia source origin is not authenticated")
    if run_git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("official Dia source checkout is dirty")
    files: dict[str, dict[str, Any]] = {}
    for relative, expected_blob in SOURCE_ROLE_BLOBS.items():
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing official source role: {relative}")
        actual_blob = git_blob_sha1(path)
        if actual_blob != expected_blob:
            raise RuntimeError(f"official source role changed: {relative}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "git_blob_sha1": actual_blob,
        }
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "resolved_revision": SOURCE_REVISION,
        "clean": True,
        "files": files,
    }


def find_function(tree: ast.AST, name: str, class_name: str | None = None) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if class_name is not None and not isinstance(node, ast.ClassDef):
            continue
        if class_name is not None and node.name != class_name:  # type: ignore[union-attr]
            continue
        body = node.body if isinstance(node, ast.ClassDef) else [node]
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
                if isinstance(child, ast.AsyncFunctionDef):
                    raise RuntimeError(f"official function is unexpectedly async: {name}")
                return child
    raise RuntimeError(f"official source function is missing: {class_name or ''}.{name}")


def compile_official_function(path: Path, name: str, class_name: str | None = None) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = copy.deepcopy(find_function(tree, name, class_name))
    # Decorators may import model-only state.  The body remains byte-for-byte
    # official source; removing decoration only makes this pure function
    # executable without constructing Dia or loading a checkpoint.
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    torch = importlib.import_module("torch")
    namespace: dict[str, Any] = {
        "torch": torch,
        "np": importlib.import_module("numpy"),
        "__name__": "dia_source_contract",
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - official AST only
    return namespace[name]


def tensor_record(value: Any, *, sample_limit: int = 12) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.numpy().tobytes()
    flat = tensor.reshape(-1).tolist()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": sha256_bytes(raw),
        "sample": flat[:sample_limit],
    }


def write_json_create_new_atomic(path: Path, packet: dict[str, Any]) -> None:
    """Publish JSON atomically without replacing an existing user path."""
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"source-contract output already exists or is a symlink: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError(f"source-contract output parent must be an existing regular directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_identity = os.fstat(descriptor)
    if not stat.S_ISREG(temporary_identity.st_mode):
        os.close(descriptor)
        raise RuntimeError("source-contract temporary output is not regular")
    owned = (temporary_identity.st_dev, temporary_identity.st_ino)

    def unlink_owned(candidate: str) -> None:
        try:
            current = os.stat(candidate, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == owned and stat.S_ISREG(current.st_mode):
                os.unlink(candidate)
        except OSError:
            pass

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(json.dumps(packet, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        verify_fd = os.open(temporary_name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            current = os.fstat(verify_fd)
        finally:
            os.close(verify_fd)
        if (current.st_dev, current.st_ino) != owned or not stat.S_ISREG(current.st_mode):
            raise RuntimeError("source-contract temporary output identity changed")
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise RuntimeError(f"source-contract output appeared during publication: {path}") from error
        try:
            final_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                final = os.fstat(final_fd)
            finally:
                os.close(final_fd)
            if (final.st_dev, final.st_ino) != owned or not stat.S_ISREG(final.st_mode):
                raise RuntimeError("source-contract output claim identity changed")
            path_stat = os.stat(path, follow_symlinks=False)
            if (path_stat.st_dev, path_stat.st_ino) != owned or not stat.S_ISREG(path_stat.st_mode):
                raise RuntimeError("source-contract output path changed after claim")
        except Exception:
            unlink_owned(path)
            raise
    finally:
        if descriptor != -1:
            os.close(descriptor)
        unlink_owned(temporary_name)


def validate_output_file(path: Path) -> None:
    if not path.is_absolute() or path == Path(path.anchor) or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError("output path must be absolute and free of dot components")
    cursor = Path(path.anchor)
    for part in path.parts[1:-1]:
        cursor /= part
        if cursor.is_symlink() or not cursor.is_dir():
            raise RuntimeError("output path has unsafe or missing parent ancestry")
    if path.exists() or path.is_symlink():
        raise RuntimeError("output path must be absent and must not be a symlink")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError("output parent must be an existing regular directory")


def cli_path(raw: str, label: str) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise RuntimeError(f"{label} must be an absolute path")
    components = raw.split("/")
    if any(component in {"", ".", ".."} for component in components[1:]):
        raise RuntimeError(f"{label} must be free of dot, empty, and trailing components")
    path = Path(raw)
    if path == Path(path.anchor) or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError(f"{label} must be free of dot components and root")
    return path


def load_official_audio_module(source: Path) -> Any:
    """Load official ``dia.audio`` while restoring every ``dia.*`` entry."""
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "dia" or name.startswith("dia.")
    }
    for name in list(previous):
        sys.modules.pop(name, None)
    try:
        package = types.ModuleType("dia")
        package.__path__ = [str(source / "dia")]  # type: ignore[attr-defined]
        sys.modules["dia"] = package
        config_spec = importlib.util.spec_from_file_location("dia.config", source / "dia/config.py")
        if config_spec is None or config_spec.loader is None:
            raise RuntimeError("official Dia config module cannot be loaded")
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules["dia.config"] = config_module
        config_spec.loader.exec_module(config_module)
        return importlib.import_module("dia.audio")
    finally:
        for name in list(sys.modules):
            if name == "dia" or name.startswith("dia."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


def execute_official_audio(source: Path) -> dict[str, Any]:
    """Run only official delay/revert helpers with a tiny synthetic tensor."""
    torch = importlib.import_module("torch")
    audio = load_official_audio_module(source)

    batch, frames, channels = 1, 32, len(DELAY_PATTERN)
    raw = torch.arange(batch * frames * channels, dtype=torch.long).reshape(batch, frames, channels)
    delay_precomp = audio.build_delay_indices(batch, frames, channels, DELAY_PATTERN)
    delayed = audio.apply_audio_delay(
        audio_BxTxC=raw,
        pad_value=1025,
        bos_value=1026,
        precomp=delay_precomp,
    )
    revert_precomp = audio.build_revert_indices(batch, frames, channels, DELAY_PATTERN)
    reverted = audio.revert_audio_delay(
        audio_BxTxC=delayed,
        pad_value=1025,
        precomp=revert_precomp,
        T=frames,
    )
    max_delay = max(DELAY_PATTERN)
    valid = frames - max_delay
    if not torch.equal(reverted[:, :valid, :], raw[:, :valid, :]):
        raise RuntimeError("official delay/revert source route failed its valid-prefix round trip")
    bos_count = int((delayed == 1026).sum().item())
    # build_revert_indices clamps every source index to T-1.  The official
    # finite-window revert therefore never emits PAD; its terminal duplicate
    # is a clamp, not padding.
    pad_count = int((reverted == 1025).sum().item())
    if bos_count != sum(DELAY_PATTERN):
        raise RuntimeError("official delay source route changed the BOS cardinality")
    if pad_count != 0:
        raise RuntimeError("official revert source route changed the PAD cardinality")
    return {
        "implementation": "official dia.audio.build_delay_indices/apply_audio_delay/build_revert_indices/revert_audio_delay",
        "shape": [batch, frames, channels],
        "delay_pattern": DELAY_PATTERN,
        "bos_value": 1026,
        "pad_value": 1025,
        "valid_prefix_frames": valid,
        "bos_count": bos_count,
        "reverted_pad_count": pad_count,
        "revert_index_clamp": True,
        "revert_out_of_bounds_count": 0,
        "delay_indices": tensor_record(delay_precomp[0]),
        "delay_gather_indices": tensor_record(delay_precomp[1]),
        "delayed": tensor_record(delayed),
        "revert_indices": tensor_record(revert_precomp[0]),
        "revert_gather_indices": tensor_record(revert_precomp[1]),
        "reverted": tensor_record(reverted),
    }


def execute_official_tokenizer(source: Path) -> dict[str, Any]:
    model_path = source / "dia/model.py"
    try:
        method_name = "_encode_text"
        encode_text = compile_official_function(model_path, method_name, "Dia")
    except RuntimeError:
        # Older authenticated Dia source revisions expose the same official
        # byte boundary as _prepare_text_input, which additionally pads and
        # builds masks.  We execute that method and retain only its source
        # token result; no model object is constructed.
        method_name = "_prepare_text_input"
        encode_text = compile_official_function(model_path, method_name, "Dia")
    import torch

    fake_self = SimpleNamespace(
        config=SimpleNamespace(data=SimpleNamespace(text_length=64, text_pad_value=0)),
        device=torch.device("cpu"),
        _create_attn_mask=lambda *args, **kwargs: torch.empty(0),
    )
    text = "A[S1]é[S2]"
    result = encode_text(fake_self, text)
    if method_name == "_prepare_text_input":
        result = result[0]
    ids = result.detach().cpu().reshape(-1).tolist()
    expected = list(text.encode("utf-8").replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02"))[:64]
    if ids != expected:
        raise RuntimeError("official Dia _encode_text differs from the fixed byte/speaker contract")
    short_self = SimpleNamespace(
        config=SimpleNamespace(data=SimpleNamespace(text_length=2, text_pad_value=0)),
        device=torch.device("cpu"),
        _create_attn_mask=lambda *args, **kwargs: torch.empty(0),
    )
    short = encode_text(short_self, "[S1]x").detach().cpu().reshape(-1).tolist()
    if short != [1, 120]:
        raise RuntimeError("official Dia _encode_text does not replace speaker markers before truncation")
    return {
        "implementation": f"official Dia.{method_name}",
        "encoding": "UTF-8 bytes, [S1]=1, [S2]=2, then text_length truncation",
        "markers": SPEAKER_MARKERS,
        "input": text,
        "ids": ids,
        "truncation_probe": short,
    }


def execute_official_sampler(source: Path) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    sampler = compile_official_function(source / "dia/model.py", "_sample_next_token")
    # The authenticated source reserves token 1024 for audio EOS.  A narrow
    # fixture would be out of range and could not exercise either EOS branch.
    logits_not_highest = torch.zeros((9, 1025), dtype=torch.float32)
    logits_not_highest[:, 0] = 3.0
    logits_not_highest[:, 1024] = 2.0
    logits_highest = logits_not_highest.clone()
    logits_highest[0, 1024] = 4.0
    calls: list[dict[str, Any]] = []
    original = torch.multinomial
    active_branch = ""

    def deterministic_multinomial(probability: Any, num_samples: int, *args: Any, **kwargs: Any) -> Any:
        calls.append({
            "branch": active_branch,
            "probability": tensor_record(probability),
            "shape": list(probability.shape),
            "eos_probability": probability[:, 1024].detach().cpu().tolist(),
            "num_samples": num_samples,
        })
        return torch.argmax(probability, dim=-1, keepdim=True)

    torch.multinomial = deterministic_multinomial
    try:
        signature = __import__("inspect").signature(sampler)
        kwargs: dict[str, Any] = {"temperature": 1.0, "top_p": 1.0}
        for name in signature.parameters:
            if name == "top_k":
                kwargs[name] = None
            elif name == "use_cfg_filter":
                kwargs[name] = False
            elif name == "cfg_filter_top_k":
                kwargs[name] = None
            elif name == "audio_eos_value":
                kwargs[name] = 1024
        active_branch = "eos_not_highest"
        result_not_highest = sampler(logits_not_highest, **kwargs)
        active_branch = "eos_highest"
        result_highest = sampler(logits_highest, **kwargs)
    finally:
        torch.multinomial = original
    if len(calls) != 2 or any(call["num_samples"] != 1 for call in calls):
        raise RuntimeError("official sampler did not call torch.multinomial once per branch")
    selected_not_highest = result_not_highest.detach().cpu().reshape(-1).tolist()
    selected_highest = result_highest.detach().cpu().reshape(-1).tolist()
    if len(selected_not_highest) != 9 or len(selected_highest) != 9:
        raise RuntimeError("official sampler did not return one token per Dia channel")
    if len(calls) != 2 or any(call["shape"] != [9, 1025] for call in calls):
        raise RuntimeError("official sampler branch probes did not preserve the [9,1025] seam")
    if any(value != 0.0 for value in calls[0]["eos_probability"]):
        raise RuntimeError("official sampler did not mask EOS when EOS was not highest")
    if selected_not_highest != [0] * 9:
        raise RuntimeError("official sampler deterministic probe did not select the non-EOS maximum")
    if calls[1]["eos_probability"][0] <= 0.0 or selected_highest[0] != 1024 or selected_highest[1:] != [0] * 8:
        raise RuntimeError("official sampler did not preserve the EOS-highest branch")
    return {
        "implementation": "official _sample_next_token",
        "input_shape": [9, 1025],
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "audio_eos_value": 1024,
        "eos_not_highest_probe": True,
        "eos_highest_probe": True,
        "call_order": [
            "temperature scaling",
            "EOS-not-highest mask",
            "top-k (disabled)",
            "top-p (disabled)",
            "softmax",
            "torch.multinomial",
            "selected channel ids",
        ],
        "multinomial_calls": calls,
        "selected": selected_not_highest,
        "branch_results": {
            "eos_not_highest": {
                "selected": selected_not_highest,
                "eos_probability": calls[0]["eos_probability"],
            },
            "eos_highest": {
                "selected": selected_highest,
                "eos_probability": calls[1]["eos_probability"],
            },
        },
        "rng_equivalence": "NOT_CLAIMED; multinomial was replaced by a deterministic probe",
    }


def has_official_compare_assignment(function: ast.FunctionDef, target: str, left: str, operator: type[ast.cmpop], right: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not any(isinstance(item, ast.Name) and item.id == target for item in node.targets):
            continue
        value = node.value
        if not isinstance(value, ast.Compare) or len(value.ops) != 1 or not isinstance(value.ops[0], operator) or len(value.comparators) != 1:
            continue
        if isinstance(value.left, ast.Name) and value.left.id == left and isinstance(value.comparators[0], ast.Name) and value.comparators[0].id == right:
            return True
    return False


def has_countdown_decrement(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.AugAssign) or not isinstance(node.op, ast.Sub) or not isinstance(node.value, ast.Constant) or node.value.value != 1:
            continue
        target = node.target
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != "eos_countdown_Bx":
            continue
        if isinstance(target.slice, ast.Name) and target.slice.id == "padding_mask_Bx":
            return True
    return False


def static_generation_contract(source: Path) -> dict[str, Any]:
    model_text = (source / "dia/model.py").read_text(encoding="utf-8")
    config_text = (source / "dia/config.py").read_text(encoding="utf-8")
    tree = ast.parse(model_text, filename="dia/model.py")
    generate = find_function(tree, "generate", "Dia")
    decoder_step = find_function(tree, "_decoder_step", "Dia")
    segment = ast.get_source_segment(model_text, generate) or ""
    required = (
        "eos_detected_Bx",
        "eos_countdown_Bx",
        "max_delay_pattern",
        "start_countdown_mask_Bx",
        "padding_mask_Bx",
        "audio_eos_value",
        "audio_pad_value",
        "_decoder_step",
    )
    missing = [token for token in required if token not in segment]
    if missing:
        raise RuntimeError(f"official Dia.generate is missing stop/sampling source markers: {missing}")
    if any(marker in segment for marker in ("extra_steps_after_eos", "eos_detected_channel_0")):
        raise RuntimeError("official Dia.generate contains a stop contract from another revision")
    if not re.search(r"eos_countdown_Bx\s*=\s*torch\.full\([^\n]*-1", segment):
        raise RuntimeError("official Dia.generate did not initialize the EOS countdown to -1")
    if not re.search(r"eos_countdown_Bx\[[^\n]+\]\s*=\s*max_delay_pattern", segment):
        raise RuntimeError("official Dia.generate did not start the max-delay EOS countdown")
    if not re.search(r"max_delay_pattern\s*=\s*max\([^\n]+\)", segment):
        raise RuntimeError("official Dia.generate did not derive max_delay_pattern")
    if not re.search(r"delay_pattern[\s\S]{0,300}\[\s*0\s*,\s*8\s*,\s*9\s*,\s*10\s*,\s*11\s*,\s*12\s*,\s*13\s*,\s*14\s*,\s*15\s*\]", config_text):
        raise RuntimeError("official Dia config does not expose the authenticated delay pattern")
    if not has_official_compare_assignment(generate, "eos_mask_NxC", "step_after_eos_Bx_", ast.Eq, "delay_pattern_Cx_"):
        raise RuntimeError("official decoder step lacks the source EOS stagger mask expression")
    if not has_official_compare_assignment(generate, "pad_mask_NxC", "step_after_eos_Bx_", ast.Gt, "delay_pattern_Cx_"):
        raise RuntimeError("official decoder step lacks the source PAD stagger mask expression")
    if not has_countdown_decrement(generate):
        raise RuntimeError("official generation loop lacks the source countdown decrement")
    def is_sampler_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        return (isinstance(node.func, ast.Name) and node.func.id == "_sample_next_token") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "_sample_next_token"
        )

    calls_generate = [node for node in ast.walk(generate) if is_sampler_call(node)]
    calls_decoder = [node for node in ast.walk(decoder_step) if is_sampler_call(node)]
    if len(calls_generate) != 0 or len(calls_decoder) != 1:
        raise RuntimeError(f"official sampler call sites drifted: generate={len(calls_generate)}, decoder_step={len(calls_decoder)}")
    for marker in ("eos_not_highest_mask", "torch.multinomial", "top_k", "top_p"):
        if marker not in model_text:
            raise RuntimeError(f"official sampler implementation marker is missing: {marker}")
    return {
        "method": "official Dia.generate",
        "decoder_step_method": "official Dia._decoder_step",
        "delay_pattern": DELAY_PATTERN,
        "sampler_call_sites": 0,
        "sampler_call_sites_generate": 0,
        "sampler_call_sites_decoder_step": 1,
        "sampling_position": "after CFG logits and before generated token write",
        "stop": {
            "eos_channel": 0,
            "eos_detected_variable": "eos_detected_Bx",
            "countdown_variable": "eos_countdown_Bx",
            "initial_countdown": -1,
            "countdown_start": "eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern",
            "max_delay_pattern": 15,
            "staggered_eos_pad": True,
            "drain_steps": 15,
            "eos_value_source": "audio_eos_value",
            "pad_value_source": "audio_pad_value",
            "source_checks": {
                "eos_mask": "step_after_eos_Bx_ == delay_pattern_Cx_",
                "pad_mask": "step_after_eos_Bx_ > delay_pattern_Cx_",
                "countdown_decrement": "eos_countdown_Bx[padding_mask_Bx] -= 1",
            },
        },
    }


def authenticate_rust_contract(root: Path) -> dict[str, Any]:
    texts: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}
    for relative in RUST_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing Rust Dia contract file: {relative}")
        text = path.read_text(encoding="utf-8")
        texts[relative] = text
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    required = {
        RUST_FILES[0]: (
            'DIA_SPEAKER_ONE_MARKER: &[u8] = b"[S1]"',
            'DIA_SPEAKER_TWO_MARKER: &[u8] = b"[S2]"',
            "take(self.text_length)",
        ),
        RUST_FILES[1]: (
            "pub(crate) fn generate_codes(",
            "fn apply_generation_drain(",
            "eos_countdown",
            "current_step_idx",
            "let elapsed =",
            "remaining - 1",
            "sample_tokens_inner",
            "audio_eos_value",
            "audio_pad_value",
        ),
        RUST_FILES[2]: (
            "delay_pattern: vec![0, 8, 9, 10, 11, 12, 13, 14, 15]",
            "audio_bos_value: 1026",
            "audio_eos_value: 1024",
            "audio_pad_value: 1025",
        ),
    }
    for relative, markers in required.items():
        missing = [marker for marker in markers if marker not in texts[relative]]
        if missing:
            raise RuntimeError(f"Rust Dia contract drift in {relative}: {missing}")
    return {
        "files": files,
        "status": "RUST_SOURCE_CONTRACT_MATCHED",
        "delay_pattern": DELAY_PATTERN,
        "speaker_markers": SPEAKER_MARKERS,
        "staggered_eos_drain": {"max_delay": 15, "apply_generation_drain": True, "generate_codes": True},
    }


def run(source: Path, rust_root: Path, output: Path) -> dict[str, Any]:
    source_evidence = authenticate_source(source)
    rust_evidence = authenticate_rust_contract(rust_root)
    tokenizer = execute_official_tokenizer(source)
    audio = execute_official_audio(source)
    sampler = execute_official_sampler(source)
    generation = static_generation_contract(source)
    packet = {
        "format": FORMAT,
        "status": "SOURCE_CONTRACT_COMPLETE_MODEL_FREE",
        "source": source_evidence,
        "rust": rust_evidence,
        "tokenizer": tokenizer,
        "audio_delay_revert": audio,
        "sampler": sampler,
        "generation": generation,
        "model_payload_access": "NONE",
        "checkpoint_loaded": False,
        "dac_loaded": False,
        "pcm_generated": False,
        "parity_status": "NOT_RUN_SOURCE_CONTRACT_ONLY",
        "publication": "NO_UPLOAD",
    }
    write_json_create_new_atomic(output, packet)
    return packet


def self_test() -> None:
    assert FORMAT == "vokra-dia-1-6b-source-contract-v1"
    assert DELAY_PATTERN == [0, 8, 9, 10, 11, 12, 13, 14, 15]
    assert SPEAKER_MARKERS == {"[S1]": 1, "[S2]": 2}
    assert len(SOURCE_ROLE_BLOBS) == 7
    with tempfile.TemporaryDirectory(prefix="dia-source-contract-self-test-") as directory:
        path = Path(directory) / "packet.json"
        for unsafe in ("relative", "/private/tmp/../tmp/dia-output", "/private/tmp/./dia-output", "//", "/"):
            try:
                cli_path(unsafe, "output")
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe CLI path accepted")
        link = Path(directory) / "link"
        link.mkdir()
        try:
            validate_output_file(link / "../packet.json")
        except RuntimeError:
            pass
        else:
            raise AssertionError("dot component output path accepted")
        write_json_create_new_atomic(path, {"status": "ok"})
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "ok"
        try:
            write_json_create_new_atomic(path, {"status": "overwrite"})
        except RuntimeError:
            pass
        else:
            raise AssertionError("existing source-contract output was overwritten")
        completed = Path(directory) / "cleanup-failure.json"
        original_unlink = os.unlink
        os.unlink = lambda _path: (_ for _ in ()).throw(PermissionError("synthetic cleanup failure"))
        try:
            write_json_create_new_atomic(completed, {"status": "complete"})
        finally:
            os.unlink = original_unlink
        assert json.loads(completed.read_text(encoding="utf-8"))["status"] == "complete"
    tree = ast.parse("def generate(self):\n  self._decoder_step()\n\ndef _decoder_step(self):\n  self._sample_next_token()")
    # The synthetic AST check is intentionally limited to parser behavior; it
    # never stands in for official source execution.
    assert find_function(tree, "generate").name == "generate"
    try:
        find_function(tree, "missing")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing official function was accepted")
    print("dia source-contract self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--rust-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source, args.rust_root, args.output)):
            parser.error("--self-test accepts no paths")
        self_test()
        return 0
    if args.source is None or args.rust_root is None or args.output is None:
        parser.error("--source, --rust-root and --output are required")
    try:
        args.source = cli_path(args.source, "source")
        args.rust_root = cli_path(args.rust_root, "rust-root")
        args.output = cli_path(args.output, "output")
    except RuntimeError as error:
        parser.error(str(error))
    try:
        validate_output_file(args.output)
    except RuntimeError as error:
        parser.error(str(error))
    run(args.source, args.rust_root, args.output)
    print(f"Dia source contract written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
