#!/usr/bin/env -S uv run --no-sync --frozen --project tools/parity/csm_1b_reference --python 3.12 python
"""Capture an official Transformers CSM-1B greedy reference on VAST.

This is deliberately an adapter around the pinned Transformers checkout.  It
does not implement a tokenizer, codec, or generation loop. The caller owns a
small JSON packet containing an authenticated conversation plus contained
audio inputs; the official ``apply_chat_template`` and CSM processor produce
the actual input IDs, generated codes, exposed taps, and PCM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import tomllib
import wave
from pathlib import Path
from typing import Any

HF_REPOSITORY = "sesame/csm-1b"
HF_REVISION = "c92a71e1c419772e25be7dc14d952c2521a740ab"
TOKENIZER_REPOSITORY = "sesame/csm-1b"
TOKENIZER_REVISION = HF_REVISION
# This 40-character value is the Git blob identity reported by the fixed HF
# tree.  It is deliberately named as SHA-1: it must never be presented as a
# SHA-256 payload digest.  The payload SHA-256 is computed from the local,
# authenticated snapshot at reference time.
TOKENIZER_JSON_GIT_BLOB_SHA1 = "8de5df033b78de76dbe15fdd8b934678b5017aaf"
TOKENIZER_JSON_BYTES = 17_209_980
TRANSFORMERS_COMMIT = "945727948c1143a10ac6f7d811aa58bb0d126b5b"
GENERATION_CSM_GIT_BLOB_SHA1 = "2fec3ea8919fa0c0e0782b54dcafe79e317ec9f3"
FORMAT = "vokra-csm-1b-official-reference-v1"
# This is the reviewed lock identity for the adapted Python 3.12 reference
# selection. It does not claim that Sesame or Transformers upstream require
# these torch/numpy/audio versions.
REFERENCE_LOCK_SHA256 = "62b70ae227b81a2eda59716c2a613f8322405abbf352dc74a5774ffa541a75bc"
SOURCE_TRANSFORMERS_REQUIREMENT = "transformers==4.52.1"
SOURCE_HUGGINGFACE_HUB_REQUIREMENT = "huggingface-hub>=0.30,<1.0"
TRANSFORMERS_SECURITY_ADVISORY = "GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM = "5.10.0"
ISOLATED_TRANSFORMERS_PIN = "5.10.4"
ISOLATED_HUGGINGFACE_HUB_PIN = "1.5.0"
TRANSFORMERS_COMPATIBILITY_STATUS = "BLOCKED_UNVERIFIED_API_SMOKE"
REFERENCE_PACKAGE_SELECTION = {
    "huggingface-hub": ISOLATED_HUGGINGFACE_HUB_PIN,
    "numpy": "2.2.6",
    "torch": "2.7.1",
    "transformers": ISOLATED_TRANSFORMERS_PIN,
}
REFERENCE_TORCH_DISTRIBUTIONS = {"2.7.1", "2.7.1+cpu"}
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
DEPENDENCY_LICENSE_AUDIT = {
    "annotated-doc": ("MIT", "https://pypi.org/pypi/annotated-doc/0.0.5/json"),
    "anyio": ("MIT", "https://pypi.org/pypi/anyio/4.14.2/json"),
    "certifi": ("MPL-2.0", "https://pypi.org/pypi/certifi/2026.7.22/json"),
    "colorama": ("BSD-3-Clause", "https://pypi.org/pypi/colorama/0.4.6/json"),
    "filelock": ("MIT", "https://pypi.org/pypi/filelock/3.32.4/json"),
    "fsspec": ("BSD-3-Clause", "https://pypi.org/pypi/fsspec/2026.7.0/json"),
    "hf-xet": ("Apache-2.0", "https://pypi.org/pypi/hf-xet/1.6.0/json"),
    "h11": ("MIT", "https://pypi.org/pypi/h11/0.16.0/json"),
    "httpcore": ("BSD-3-Clause", "https://pypi.org/pypi/httpcore/1.0.9/json"),
    "httpx": ("BSD-3-Clause", "https://pypi.org/pypi/httpx/0.28.1/json"),
    "huggingface-hub": ("Apache-2.0", "https://pypi.org/pypi/huggingface-hub/1.5.0/json"),
    "idna": ("BSD-3-Clause", "https://pypi.org/pypi/idna/3.19/json"),
    "jinja2": ("BSD-3-Clause", "https://pypi.org/pypi/jinja2/3.1.6/json"),
    "markupsafe": ("BSD-3-Clause", "https://pypi.org/pypi/markupsafe/3.0.3/json"),
    "markdown-it-py": ("MIT", "https://pypi.org/pypi/markdown-it-py/4.2.0/json"),
    "mdurl": ("MIT", "https://pypi.org/pypi/mdurl/0.1.2/json"),
    "mpmath": ("BSD", "https://pypi.org/pypi/mpmath/1.3.0/json"),
    "networkx": ("BSD-3-Clause", "https://pypi.org/pypi/networkx/3.6.1/json"),
    "numpy": ("BSD-3-Clause + bundled runtime notices (GPL/LGPL)", "https://pypi.org/pypi/numpy/2.2.6/json"),
    "packaging": ("Apache-2.0 AND BSD-2-Clause", "https://pypi.org/pypi/packaging/26.3/json"),
    "pyyaml": ("MIT", "https://pypi.org/pypi/pyyaml/6.0.3/json"),
    "pygments": ("BSD-2-Clause", "https://pypi.org/pypi/pygments/2.21.0/json"),
    "regex": ("Apache-2.0", "https://pypi.org/pypi/regex/2026.7.19/json"),
    "rich": ("MIT", "https://pypi.org/pypi/rich/15.0.0/json"),
    "safetensors": ("Apache-2.0", "https://pypi.org/pypi/safetensors/0.8.0/json"),
    "setuptools": ("MIT", "https://pypi.org/pypi/setuptools/84.0.0/json"),
    "shellingham": ("ISC", "https://pypi.org/pypi/shellingham/1.5.4/json"),
    "sympy": ("BSD-3-Clause", "https://pypi.org/pypi/sympy/1.14.0/json"),
    "tokenizers": ("Apache-2.0", "https://pypi.org/pypi/tokenizers/0.22.2/json"),
    "torch": ("BSD-3-Clause; official CPU index", PYTORCH_CPU_INDEX),
    "tqdm": ("MPL-2.0 AND MIT", "https://pypi.org/pypi/tqdm/4.70.0/json"),
    "transformers": ("Apache-2.0", "https://pypi.org/pypi/transformers/5.10.4/json"),
    "typer": ("MIT", "https://pypi.org/pypi/typer/0.27.2/json"),
    "typing-extensions": ("PSF-2.0; blocked by owner policy", "https://pypi.org/pypi/typing-extensions/4.16.0/json"),
    "vokra-csm-1b-reference": ("PROJECT_METADATA_ONLY", "tools/parity/csm_1b_reference/pyproject.toml"),
}
MAX_TEXT = 16_384
MAX_AUDIO_SAMPLES = 24_000 * 60 * 10
WAV_AUDIO_FORMAT = "wav-pcm16-le"
MAX_TENSOR_BYTES = 2 * 1024 * 1024 * 1024
MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_HOP = 1_920
CODEBOOK_SIZE = 2_048
LOGIT_VOCAB_SIZE = 2_051
OFFICIAL_CHAT_TEMPLATE_KWARGS = {"tokenize": False}
OFFICIAL_DIRECT_PROCESSOR_KWARGS = {"sampling_rate": MIMI_SAMPLE_RATE, "return_tensors": "pt"}
PROCESSING_UTILS_GIT_BLOB_SHA1 = "8dbc210fbcd0b4e9b741427f6f4d74d9ecbf7913"
PROCESSING_UTILS_MARKERS = (
    "if not is_batched:\n            prompt = prompt[0]",
    "out = self(\n                text=prompt,",
    "audio=batch_audios if batch_audios else None",
)
LICENSE_APPROVAL_STATUS = "REVIEWED_LICENSE_AUDIT_COMPLETE"
DEPENDENCY_APPROVAL_STATUS = "LOCKED_ADAPTED_SELECTION_LICENSE_AUDIT_COMPLETE"
CSM_REFERENCE_PROJECT_FIELDS = {
    "dependency_status",
    "transformers_compatibility",
    "source_transformers_requirement",
    "source_huggingface_hub_requirement",
    "transformers_security_advisory",
    "transformers_security_patched_minimum",
    "isolated_transformers_pin",
    "isolated_huggingface_hub_pin",
    "transformers_compatibility_status",
    "lock_sha256",
    "selection_note",
    "license_status",
    "license_evidence_lock_sha256",
    "license_blocker",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_dependency_gate(project_path: Path, lock_path: Path) -> None:
    """Require explicit lock-bound license approval before execution."""
    try:
        document = tomllib.loads(project_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("CSM reference pyproject TOML is unreadable") from error
    reference = document.get("tool", {}).get("csm_reference")
    if not isinstance(reference, dict) or set(reference) != CSM_REFERENCE_PROJECT_FIELDS:
        raise RuntimeError("CSM reference dependency gate schema is incomplete or unknown")
    actual_lock_sha256 = digest(lock_path) if lock_path.is_file() else None
    if actual_lock_sha256 != REFERENCE_LOCK_SHA256:
        raise RuntimeError("dedicated CSM reference uv.lock identity mismatch")
    if reference.get("lock_sha256") != actual_lock_sha256:
        raise RuntimeError("CSM reference pyproject lock_sha256 does not match uv.lock")
    if reference.get("dependency_status") != DEPENDENCY_APPROVAL_STATUS:
        raise RuntimeError("CSM reference dependency approval is not explicit")
    if reference.get("license_status") != LICENSE_APPROVAL_STATUS:
        raise RuntimeError("CSM reference dependency license approval is not explicit")
    if reference.get("license_evidence_lock_sha256") != REFERENCE_LOCK_SHA256:
        raise RuntimeError("CSM reference license evidence is not bound to the reviewed lock")
    expected_security = {
        "source_transformers_requirement": SOURCE_TRANSFORMERS_REQUIREMENT,
        "source_huggingface_hub_requirement": SOURCE_HUGGINGFACE_HUB_REQUIREMENT,
        "transformers_security_advisory": TRANSFORMERS_SECURITY_ADVISORY,
        "transformers_security_patched_minimum": TRANSFORMERS_SECURITY_PATCHED_MINIMUM,
        "isolated_transformers_pin": ISOLATED_TRANSFORMERS_PIN,
        "isolated_huggingface_hub_pin": ISOLATED_HUGGINGFACE_HUB_PIN,
        "transformers_compatibility_status": TRANSFORMERS_COMPATIBILITY_STATUS,
    }
    if any(reference.get(key) != value for key, value in expected_security.items()):
        raise RuntimeError("CSM Transformers security metadata is not authenticated")
    if reference.get("transformers_compatibility_status") == TRANSFORMERS_COMPATIBILITY_STATUS:
        raise RuntimeError("CSM secure Transformers closure lacks authenticated API smoke")


def locked_package_rows(lock_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Return deterministic lock package identity rows, including markers."""
    try:
        document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("CSM uv.lock is unreadable") from error
    packages = document.get("package")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("CSM uv.lock has no package rows")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or set(package) - {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}:
            raise RuntimeError("CSM uv.lock package row has unknown fields")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        markers = package.get("resolution-markers", [])
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version or not isinstance(source, dict) or not isinstance(markers, list) or not all(isinstance(item, str) for item in markers):
            raise RuntimeError("CSM uv.lock package row has invalid identity")
        row = {"name": name, "version": version, "source": source, "resolution_markers": markers}
        identity = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if identity in identities:
            raise RuntimeError(f"duplicate CSM uv.lock package identity: {name}")
        identities.add(identity)
        rows.append(row)
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(encoded).hexdigest()


def dependency_license_rows(package_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach primary metadata/license conclusions to every unique lock row."""
    rows: list[dict[str, Any]] = []
    for row in package_rows:
        audit = DEPENDENCY_LICENSE_AUDIT.get(row["name"])
        if audit is None:
            raise RuntimeError(f"locked package lacks a reviewed license row: {row['name']}")
        rows.append({**row, "license": audit[0], "license_source": audit[1]})
    return rows


def read_pcm16_wav(path: Path) -> tuple[Any, int]:
    """Decode bounded caller-owned PCM16 WAV; this is input framing, not model math."""
    if path.suffix.lower() != ".wav" or not path.is_file() or path.is_symlink():
        raise RuntimeError("caller audio must be a regular .wav file")
    max_bytes = 44 + MAX_AUDIO_SAMPLES * 2
    if path.stat().st_size > max_bytes:
        raise RuntimeError("caller WAV exceeds the bounded PCM16 input contract")
    try:
        with wave.open(str(path), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getframerate() != MIMI_SAMPLE_RATE or reader.getcomptype() != "NONE":
                raise RuntimeError("caller WAV must be mono, PCM16, and 24kHz")
            frames = reader.getnframes()
            if frames <= 0 or frames > MAX_AUDIO_SAMPLES:
                raise RuntimeError("caller WAV frame count is empty or unbounded")
            raw = reader.readframes(frames)
            if len(raw) != frames * 2:
                raise RuntimeError("caller WAV data is truncated")
    except (EOFError, OSError, wave.Error) as error:
        raise RuntimeError("caller WAV is malformed") from error
    import numpy as np

    samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if samples.ndim != 1 or samples.size != frames or not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0):
        raise RuntimeError("caller WAV samples are invalid")
    return samples, MIMI_SAMPLE_RATE


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate packet key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def validate_packet(packet: Any) -> dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != {"messages"}:
        raise ValueError("packet fields must be exactly messages")
    messages = packet["messages"]
    if not isinstance(messages, list) or not messages or len(messages) > 32:
        raise ValueError("messages must be a bounded non-empty conversation")
    audio_paths: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("conversation message schema must be exactly role/content")
        content = message["content"]
        if message["role"] not in {"0", "1"} or not isinstance(content, list) or not content:
            raise ValueError("conversation message is invalid or unbounded")
        for item in content:
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise ValueError("conversation content items must be typed objects")
            if item["type"] == "text":
                if set(item) != {"type", "text"} or not isinstance(item["text"], str) or not item["text"] or len(item["text"]) > MAX_TEXT or "\x00" in item["text"]:
                    raise ValueError("text content item is invalid or unbounded")
            elif item["type"] == "audio":
                if set(item) != {"type", "path"} or not isinstance(item["path"], str):
                    raise ValueError("audio content item must contain exactly a relative path")
                candidate = Path(item["path"])
                if not item["path"] or candidate.suffix.lower() != ".wav" or "\x00" in item["path"] or "\\" in item["path"] or candidate.is_absolute() or ".." in candidate.parts:
                    raise ValueError(f"audio path must be a safe relative path: {item['path']!r}")
                audio_paths.append(item["path"])
            else:
                raise ValueError(f"unsupported conversation content type: {item['type']!r}")
        if index == len(messages) - 1 and (message["role"] != "0" or any(item["type"] == "audio" for item in content)):
            raise ValueError("last message must be the target text message without audio context")
    if not any(item["type"] == "text" for item in messages[-1]["content"]):
        raise ValueError("last message must contain target text")
    return {"messages": messages, "audio_paths": audio_paths}


def validate_processor_calls(chat_kwargs: Any, processor_kwargs: Any) -> None:
    """Keep the official chat-template/processor adapter source-shaped."""
    if chat_kwargs != OFFICIAL_CHAT_TEMPLATE_KWARGS:
        raise ValueError("CSM chat-template call must use tokenize=False")
    if processor_kwargs != OFFICIAL_DIRECT_PROCESSOR_KWARGS:
        raise ValueError("CSM direct processor call has an unexpected audio boundary")


def processor_audio_argument(audio: list[Any]) -> list[Any] | None:
    """Match CsmProcessor's empty-audio contract instead of passing ``[]``."""
    if not isinstance(audio, list):
        raise ValueError("CSM processor audio argument must originate from a list")
    return audio if audio else None


def validate_inspection_evidence(inspection: Any) -> dict[str, Any]:
    """Reject stale/error manifests before importing official model code."""
    if not isinstance(inspection, dict) or inspection.get("status") != "BLOCKED" or inspection.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or inspection.get("collection_status") != "AUTHENTICATED":
        raise RuntimeError("authenticated CSM inspection evidence is incomplete")
    return inspection


def write_tensor(
    path: Path,
    tensor: Any,
    torch: Any,
    dtype: str = "float32",
    max_value: int | None = None,
) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        raise RuntimeError(f"official output tensor is missing: {path.name}")
    value = tensor.detach().cpu().contiguous()
    if value.ndim == 0 or any(int(dimension) <= 0 for dimension in value.shape):
        raise RuntimeError(f"official output tensor has an empty/invalid shape: {path.name}")
    if value.numel() > MAX_TENSOR_BYTES // 4:
        raise RuntimeError(f"official output tensor exceeds element bound: {path.name}")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise RuntimeError(f"official output tensor is non-finite: {path.name}")
    if dtype == "u32":
        if value.dtype not in (torch.int32, torch.int64, torch.long):
            raise RuntimeError(f"official code tensor has unexpected dtype: {value.dtype}")
        if bool((value < 0).any().item()) or bool((value > 0xFFFF_FFFF).any().item()):
            raise RuntimeError(f"official code tensor contains an out-of-range ID: {path.name}")
        if max_value is not None and bool((value >= max_value).any().item()):
            raise RuntimeError(f"official code tensor exceeds its vocabulary: {path.name}")
        import numpy as np

        raw = value.numpy().astype("<u4", copy=False).tobytes()
    else:
        raw = value.float().numpy().tobytes()
    if len(raw) == 0 or len(raw) > MAX_TENSOR_BYTES:
        raise RuntimeError(f"official output tensor byte bound violation: {path.name}")
    path.write_bytes(raw)
    return {"path": path.name, "bytes": len(raw), "sha256": digest(path), "shape": list(value.shape), "dtype": dtype}


def stack_generation_logits(values: Any, torch: Any) -> Any:
    """Preserve one ``[batch, vocab]`` logits tensor per generation step.

    Transformers' generation output stores logits as a tuple of step tensors.
    Concatenating on dimension 1 silently produces ``[batch, steps*vocab]``;
    stacking on a new dimension is the only lossless representation.
    """
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("generation logits must be a non-empty sequence")
    if not all(isinstance(item, torch.Tensor) for item in values):
        raise ValueError("generation logits must contain tensors")
    first = values[0]
    if first.ndim != 2 or first.shape[0] <= 0 or first.shape[1] <= 0:
        raise ValueError(f"generation logits must have shape [batch,vocab], got {tuple(first.shape)}")
    if any(item.ndim != 2 or tuple(item.shape) != tuple(first.shape) for item in values):
        raise ValueError("generation logits have inconsistent [batch,vocab] shapes")
    return torch.stack(list(values), dim=1)


def depth_input_ids_from_hook(args: Any, kwargs: Any) -> Any:
    """Read source ``input_ids`` from a kwargs-enabled Torch forward hook."""
    if isinstance(kwargs, dict):
        value = kwargs.get("input_ids")
        if value is not None:
            return value
    if isinstance(args, (tuple, list)) and args:
        return args[0]
    return None


def validate_depth_input_shapes(shapes: Any, batch: int, frames: int, codebooks: int = 31) -> None:
    expected_calls = frames * codebooks
    if not isinstance(shapes, list) or len(shapes) != expected_calls:
        raise RuntimeError("official depth decoder call count is not frame/codebook aligned")
    if any(shape != [batch, 2 if index % codebooks == 0 else 1] for index, shape in enumerate(shapes)):
        raise RuntimeError("official depth decoder prefill/step input cardinality is not source-shaped")


def generation_role_matches(path: Path, expected_blob: str) -> bool:
    return path.name == "generation_csm.py" and not path.is_symlink() and path.is_file() and git_blob(path) == expected_blob


def self_test() -> int:
    import torch

    try:
        lock_path = Path(__file__).resolve().parent / "csm_1b_reference" / "uv.lock"
        assert lock_path.is_file() and digest(lock_path) == REFERENCE_LOCK_SHA256
        project_path = lock_path.parent / "pyproject.toml"
        project_text = project_path.read_text(encoding="utf-8")
        for altered in (
            project_text.replace('dependency_status = "LOCKED_ADAPTED_SELECTION_LICENSE_AUDIT_PENDING"\n', ""),
            project_text.replace('dependency_status = "LOCKED_ADAPTED_SELECTION_LICENSE_AUDIT_PENDING"', 'dependency_status = "UNKNOWN"'),
            project_text.replace('dependency_status = "LOCKED_ADAPTED_SELECTION_LICENSE_AUDIT_PENDING"', 'dependency_status = "LOCKED_ADAPTED_SELECTION_LICENSE_AUDIT_PENDING_EXTRA"'),
            project_text.replace('license_status = "BLOCKED_LICENSE_METADATA_REVIEW"\n', ""),
            project_text.replace('license_status = "BLOCKED_LICENSE_METADATA_REVIEW"', 'license_status = "UNKNOWN"'),
            project_text.replace('license_status = "BLOCKED_LICENSE_METADATA_REVIEW"', 'license_status = "REVIEWED_LICENSE_AUDIT_PENDING"'),
            project_text.replace(
                f'lock_sha256 = "{REFERENCE_LOCK_SHA256}"',
                'lock_sha256 = "' + "0" * 64 + '"',
            ),
        ):
            with tempfile.TemporaryDirectory() as gate_directory:
                altered_path = Path(gate_directory) / "pyproject.toml"
                altered_path.write_text(altered, encoding="utf-8")
                try:
                    validate_dependency_gate(altered_path, lock_path)
                except RuntimeError:
                    pass
                else:
                    return 1
        approved = project_text.replace(
            'dependency_status = "LOCKED_ADAPTED_SELECTION_LICENSE_AUDIT_PENDING"',
            f'dependency_status = "{DEPENDENCY_APPROVAL_STATUS}"',
        ).replace(
            'license_status = "BLOCKED_LICENSE_METADATA_REVIEW"',
            f'license_status = "{LICENSE_APPROVAL_STATUS}"',
        )
        with tempfile.TemporaryDirectory() as gate_directory:
            approved_path = Path(gate_directory) / "pyproject.toml"
            approved_path.write_text(approved, encoding="utf-8")
            try:
                validate_dependency_gate(approved_path, lock_path)
            except RuntimeError:
                pass
            else:
                return 1
        assert CODEBOOK_SIZE == 2048
        assert LOGIT_VOCAB_SIZE == 2051
        assert MIMI_SAMPLE_RATE == 24_000 and MIMI_FRAME_HOP == 1_920
        assert "librosa" not in REFERENCE_PACKAGE_SELECTION
        assert "soundfile" not in REFERENCE_PACKAGE_SELECTION
        assert REFERENCE_PACKAGE_SELECTION["transformers"] == ISOLATED_TRANSFORMERS_PIN
        assert REFERENCE_PACKAGE_SELECTION["huggingface-hub"] == ISOLATED_HUGGINGFACE_HUB_PIN
        assert SOURCE_TRANSFORMERS_REQUIREMENT == "transformers==4.52.1"
        assert SOURCE_HUGGINGFACE_HUB_REQUIREMENT == "huggingface-hub>=0.30,<1.0"
        assert TRANSFORMERS_SECURITY_ADVISORY == "GHSA-xrqw-3rrv-vx5w"
        assert TRANSFORMERS_SECURITY_PATCHED_MINIMUM == "5.10.0"
        assert TRANSFORMERS_COMPATIBILITY_STATUS == "BLOCKED_UNVERIFIED_API_SMOKE"
        assert PYTORCH_CPU_INDEX == "https://download.pytorch.org/whl/cpu"
        assert processor_audio_argument([]) is None
        sentinel_audio = [object()]
        assert processor_audio_argument(sentinel_audio) is sentinel_audio
        rows, rows_hash = locked_package_rows(lock_path)
        assert rows and rows_hash == hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        license_rows = dependency_license_rows(rows)
        assert {row["name"] for row in license_rows} == {row["name"] for row in rows}
        locked_third_party = {row["name"] for row in rows if row["name"] != "vokra-csm-1b-reference"}
        audited_third_party = set(DEPENDENCY_LICENSE_AUDIT) - {"vokra-csm-1b-reference"}
        assert audited_third_party == locked_third_party
        assert "vokra-csm-1b-reference" in DEPENDENCY_LICENSE_AUDIT
        with tempfile.NamedTemporaryFile("w", suffix=".lock", encoding="utf-8") as lock_file:
            lock_file.write('version = 1\nrevision = 3\n[[package]]\nname = "x"\nversion = "1"\nsource = { registry = "https://pypi.org/simple" }\n[[package]]\nname = "x"\nversion = "1"\nsource = { registry = "https://pypi.org/simple" }\n')
            lock_file.flush()
            try:
                locked_package_rows(Path(lock_file.name))
            except RuntimeError:
                pass
            else:
                return 1
        validate_depth_input_shapes([[1, 2], [1, 1], [1, 1]], 1, 1, 3)
        try:
            validate_depth_input_shapes([[1, 1], [1, 1], [1, 1]], 1, 1, 3)
        except RuntimeError:
            pass
        else:
            return 1
        frame_rows = [[4] * 32, [0] * 32]
        codec_eos = [index for index, frame in enumerate(frame_rows) if all(code == 0 for code in frame)]
        assert codec_eos == [1] and len(frame_rows) - 1 == codec_eos[0]
        class FakeDepth(torch.nn.Module):
            def forward(self, **kwargs: Any) -> Any:
                return torch.zeros(1, 1, LOGIT_VOCAB_SIZE)

        captured: list[Any] = []

        def fake_hook(_module: Any, args: Any, kwargs: Any, _output: Any) -> None:
            captured.append(depth_input_ids_from_hook(args, kwargs))

        fake_depth = FakeDepth()
        handle = fake_depth.register_forward_hook(fake_hook, with_kwargs=True)
        try:
            fake_depth(input_ids=torch.zeros(1, 2, dtype=torch.long))
            fake_depth(attention_mask=torch.ones(1, 2, dtype=torch.long))
        finally:
            handle.remove()
        assert isinstance(captured[0], torch.Tensor) and tuple(captured[0].shape) == (1, 2)
        assert captured[1] is None
        stacked = stack_generation_logits([torch.zeros(1, 7), torch.ones(1, 7)], torch)
        assert tuple(stacked.shape) == (1, 2, 7)
        try:
            stack_generation_logits([torch.zeros(1, 7), torch.zeros(1, 8)], torch)
        except ValueError:
            pass
        else:
            return 1
        for invalid in (
            {"messages": [{"role": "0", "content": "hello"}]},
            {"messages": [{"role": "0", "content": [{"type": "audio", "path": "x.wav"}]}]},
            {"messages": [{"role": "1", "content": [{"type": "audio", "path": "x.flac"}]}, {"role": "0", "content": [{"type": "text", "text": "hello"}]}]},
        ):
            try:
                validate_packet(invalid)
            except ValueError:
                pass
            else:
                return 1
        validate_processor_calls(OFFICIAL_CHAT_TEMPLATE_KWARGS, OFFICIAL_DIRECT_PROCESSOR_KWARGS)
        try:
            validate_processor_calls({"tokenize": True}, OFFICIAL_DIRECT_PROCESSOR_KWARGS)
        except ValueError:
            pass
        else:
            return 1
        try:
            validate_inspection_evidence({"status": "BLOCKED", "inspection_status": "INSPECTION_ERROR", "collection_status": "FAILED"})
        except RuntimeError:
            pass
        else:
            return 1
    except (AssertionError, ValueError):
        return 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "packet.json"
        generation_path = Path(directory) / "generation_csm.py"
        generation_path.write_text("# pinned role\n", encoding="utf-8")
        expected = git_blob(generation_path)
        if not generation_role_matches(generation_path, expected) or generation_role_matches(generation_path, "0" * 40):
            return 1
        path.write_text(json.dumps({"messages": [{"role": "0", "content": [{"type": "text", "text": "hello"}]}]}), encoding="utf-8")
        if validate_packet(strict_json(path))["audio_paths"] != []:
            return 1
        path.write_text(json.dumps({"messages": [{"role": "1", "content": [{"type": "audio", "path": "prompt.wav"}]}, {"role": "0", "content": [{"type": "text", "text": "hello"}]}]}), encoding="utf-8")
        if validate_packet(strict_json(path))["audio_paths"] != ["prompt.wav"]:
            return 1
        import wave as wav
        wav_path = Path(directory) / "prompt.wav"
        with wav.open(str(wav_path), "wb") as writer:
            writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(24_000); writer.writeframes(b"\x00\x00" * 8)
        samples, rate = read_pcm16_wav(wav_path)
        assert rate == 24_000 and samples.shape == (8,)
        for suffix, channels, sample_width, rate_value in ((".flac", 1, 2, 24_000), (".wav", 2, 2, 24_000), (".wav", 1, 1, 24_000), (".wav", 1, 2, 16_000)):
            invalid = Path(directory) / ("invalid" + suffix)
            with wav.open(str(invalid), "wb") as writer:
                writer.setnchannels(channels); writer.setsampwidth(sample_width); writer.setframerate(rate_value); writer.writeframes(b"\x00\x00" * 2)
            try:
                read_pcm16_wav(invalid)
            except RuntimeError:
                pass
            else:
                return 1
        path.write_text('{"messages":[],"messages":[]}', encoding="utf-8")
        try:
            strict_json(path)
        except ValueError:
            pass
        else:
            return 1
    print("csm_1b_dump_reference.py self-test: OK")
    return 0


def run(args: argparse.Namespace) -> int:
    lock_path = Path(__file__).resolve().parent / "csm_1b_reference" / "uv.lock"
    if not lock_path.is_file():
        raise RuntimeError("dedicated CSM reference uv.lock is required before execution")
    project_path = lock_path.parent / "pyproject.toml"
    validate_dependency_gate(project_path, lock_path)
    packet = validate_packet(strict_json(args.packet))
    snapshot = args.snapshot.resolve()
    transformers = args.transformers.resolve()
    output = args.output.resolve()
    if not (snapshot.is_dir() and transformers.is_dir()):
        raise RuntimeError("snapshot and pinned Transformers checkout are required")
    if git_output(transformers, "rev-parse", "HEAD") != TRANSFORMERS_COMMIT:
        raise RuntimeError("Transformers checkout is not at the authenticated commit")
    if git_output(transformers, "describe", "--exact-match", "--tags", "HEAD") != "v4.52.1":
        raise RuntimeError("Transformers checkout tag identity mismatch")
    if git_output(transformers, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Transformers checkout is dirty")
    origin = git_output(transformers, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    if origin != "https://github.com/huggingface/transformers":
        raise RuntimeError("Transformers checkout origin mismatch")
    if args.max_new_tokens <= 0 or args.max_new_tokens > 1125:
        raise RuntimeError("max_new_tokens must be in 1..1125")
    inspection: dict[str, Any] | None = None
    generation_contract: dict[str, Any] | None = None
    tokenizer_path = snapshot / "tokenizer.json"
    if (
        not tokenizer_path.is_file()
        or tokenizer_path.is_symlink()
        or tokenizer_path.stat().st_size != TOKENIZER_JSON_BYTES
        or git_blob(tokenizer_path) != TOKENIZER_JSON_GIT_BLOB_SHA1
    ):
        raise RuntimeError("fixed tokenizer.json Git identity/size mismatch")
    if args.inspection_manifest is not None:
        inspection = strict_json(args.inspection_manifest)
        validate_inspection_evidence(inspection)
        if inspection.get("model", {}).get("repository") != HF_REPOSITORY or inspection.get("model", {}).get("revision") != HF_REVISION:
            raise RuntimeError("inspection evidence is bound to a different model revision")
        tree = inspection.get("evidence", {}).get("tree", {})
        tree_rows = tree.get("files", [])
        if not isinstance(tree_rows, list):
            raise RuntimeError("inspection evidence has no authenticated model tree")
        model_rows = [row for row in tree_rows if row.get("path") == "model.safetensors"]
        tokenizer_rows = [row for row in tree_rows if row.get("path") == "tokenizer.json"]
        index_rows = [row for row in tree_rows if row.get("path") == "transformers.safetensors.index.json"]
        shard_rows = {row.get("path"): row for row in tree_rows if row.get("path") in {"transformers-00001-of-00002.safetensors", "transformers-00002-of-00002.safetensors"}}
        if len(model_rows) != 1 or model_rows[0].get("git_blob_sha1") != "67a4748fc437cb9a2fdeb90e6bec9dedb0ad9f86":
            raise RuntimeError("inspection evidence model checkpoint identity mismatch")
        if len(tokenizer_rows) != 1 or tokenizer_rows[0].get("git_blob_sha1") != TOKENIZER_JSON_GIT_BLOB_SHA1:
            raise RuntimeError("inspection evidence tokenizer identity mismatch")
        if len(index_rows) != 1 or index_rows[0].get("git_blob_sha1") != "6bd497e812938dc53a500a7fc941f4f04c3adecd" or index_rows[0].get("size") != 59_730:
            raise RuntimeError("inspection evidence selected Transformers index identity mismatch")
        expected_shards = {
            "transformers-00001-of-00002.safetensors": (4_944_026_784, "f6379cd719f180cfe3a0c3bd954903b632195979"),
            "transformers-00002-of-00002.safetensors": (2_189_474_180, "ca6ac15ccb23215d3813ba049010d5079aa08155"),
        }
        if set(shard_rows) != set(expected_shards) or any((shard_rows[name].get("size"), shard_rows[name].get("git_blob_sha1")) != value for name, value in expected_shards.items()):
            raise RuntimeError("inspection evidence selected Transformers shard identity mismatch")
        chat_template_hash = inspection.get("evidence", {}).get("json_roles", {}).get("chat_template_sha256")
        if chat_template_hash != digest(snapshot / "chat_template.jinja"):
            raise RuntimeError("inspection evidence chat-template identity mismatch")
        generation_contract = inspection.get("evidence", {}).get("json_roles", {}).get("generation_contract")
        if (
            not isinstance(generation_contract, dict)
            or generation_contract.get("reference_overrides")
            != {"do_sample": False, "depth_decoder_do_sample": False}
            or generation_contract.get("final_generation_semantics")
            != "reference_overrides_are_applied_at_model_generate_boundary"
        ):
            raise RuntimeError("inspection evidence does not bind the reference generation overrides")
        transformer_roles = inspection.get("evidence", {}).get("source", {}).get("transformers", {}).get("roles", {})
        generation_roles = [row for name, row in transformer_roles.items() if Path(name).name == "generation_csm.py"]
        if len(generation_roles) != 1 or generation_roles[0].get("git_blob_sha1") != GENERATION_CSM_GIT_BLOB_SHA1:
            raise RuntimeError("inspection evidence lacks authenticated Transformers tensor manifest")
        source_identity = inspection.get("source_identity", {})
        transformers_identity = inspection.get("transformers_identity", {})
        if source_identity.get("repository") != "https://github.com/SesameAILabs/csm.git" or source_identity.get("revision") != "8f6d947a26f6301deec9696f9bfb28e9e2e0d7d5" or transformers_identity.get("commit") != TRANSFORMERS_COMMIT:
            raise RuntimeError("inspection source/Transformers identity mismatch")

    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            raise RuntimeError("reference output must be a new or empty directory")
    else:
        args.output.mkdir(parents=True)

    transformers_src = (transformers / "src").resolve()
    sys.path.insert(0, str(transformers_src))
    import numpy as np
    import torch
    from importlib.metadata import version as package_version
    from transformers import AutoProcessor, CsmForConditionalGeneration
    import transformers as transformers_module
    if transformers_module.__version__ != "4.52.1":
        raise RuntimeError(f"unexpected Transformers version: {transformers_module.__version__}")
    for distribution, expected in REFERENCE_PACKAGE_SELECTION.items():
        actual = package_version(distribution)
        if distribution == "torch" and actual in REFERENCE_TORCH_DISTRIBUTIONS:
            continue
        if actual != expected:
            raise RuntimeError(
                f"unexpected adapted reference package {distribution}: {actual} != {expected}"
            )
    package_rows, package_rows_sha256 = locked_package_rows(lock_path)
    license_rows = dependency_license_rows(package_rows)
    license_rows_sha256 = hashlib.sha256(json.dumps(license_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if module_path and str(module_path).endswith((".py", ".so")):
            try:
                Path(module_path).resolve().relative_to(transformers_src)
            except ValueError as error:
                if getattr(module, "__name__", "").startswith("transformers"):
                    raise RuntimeError(f"Transformers module escaped authenticated checkout: {module_path}") from error
    # Import the exact implementation role explicitly, then verify that the
    # module used by the official model is the pinned checkout file. A merely
    # matching file somewhere on disk is not execution evidence.
    generation_module = importlib.import_module("transformers.models.csm.generation_csm")
    generation_path = Path(generation_module.__file__).resolve()
    expected_generation_path = (transformers_src / "transformers/models/csm/generation_csm.py").resolve()
    if generation_path != expected_generation_path or not generation_role_matches(generation_path, GENERATION_CSM_GIT_BLOB_SHA1):
        raise RuntimeError("authenticated generation_csm.py was not the executed module")
    processing_utils_path = (transformers_src / "transformers/processing_utils.py").resolve()
    if not processing_utils_path.is_file() or git_blob(processing_utils_path) != PROCESSING_UTILS_GIT_BLOB_SHA1:
        raise RuntimeError("authenticated ProcessorMixin processing_utils.py was not selected")
    processing_utils_text = processing_utils_path.read_text(encoding="utf-8")
    if any(marker not in processing_utils_text for marker in PROCESSING_UTILS_MARKERS):
        raise RuntimeError("pinned ProcessorMixin non-batched processor boundary markers are missing")

    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    # Do not coerce dtype: the loaded state must retain the authenticated
    # safetensors dtype manifest exactly.
    model = CsmForConditionalGeneration.from_pretrained(str(snapshot), local_files_only=True, use_safetensors=True)
    model.eval()
    inspection_transformers = inspection.get("evidence", {}).get("transformers", {}) if inspection else {}
    expected_rows = []
    for shard in inspection_transformers.get("shards", {}).values():
        expected_rows.extend(shard.get("tensors", []))
    if not expected_rows:
        raise RuntimeError("inspection has no authenticated Transformers tensor manifest")
    expected_by_name = {row.get("name"): row for row in expected_rows}
    if len(expected_by_name) != len(expected_rows):
        raise RuntimeError("inspection Transformers tensor manifest has duplicate names")
    def dtype_label(value: Any) -> str:
        labels = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
        try:
            return labels[value.dtype]
        except KeyError as error:
            raise RuntimeError(f"unsupported loaded tensor dtype: {value.dtype}") from error

    loaded_rows = [{"name": name, "shape": list(value.shape), "dtype": dtype_label(value), "elements": int(value.numel())} for name, value in sorted(model.state_dict().items())]
    if set(row["name"] for row in loaded_rows) != set(expected_by_name):
        raise RuntimeError("loaded Transformers tensor names differ from authenticated snapshot")
    if any(row["shape"] != expected_by_name[row["name"]].get("shape") or row["dtype"] != expected_by_name[row["name"]].get("dtype") or row["elements"] != expected_by_name[row["name"]].get("elements") for row in loaded_rows):
        raise RuntimeError("loaded Transformers tensor name/shape/dtype differs from authenticated snapshot")
    canonical_rows = sorted(({key: row[key] for key in ("name", "shape", "dtype", "elements")} for row in expected_rows), key=lambda row: row["name"])
    inspection_manifest_hash = hashlib.sha256(json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    loaded_manifest_hash = hashlib.sha256(json.dumps(loaded_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if loaded_manifest_hash != inspection_manifest_hash:
        raise RuntimeError("loaded Transformers tensor manifest differs from authenticated safetensors manifest")
    packet_root = args.packet.resolve().parent
    audio: list[np.ndarray] = []
    audio_inputs: list[dict[str, Any]] = []
    for raw_path in packet["audio_paths"]:
        path = (packet_root / raw_path).resolve()
        try:
            path.relative_to(packet_root)
        except ValueError as error:
            raise RuntimeError(f"caller audio path escapes packet directory: {raw_path}") from error
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"caller audio path is not a regular file: {path}")
        samples, sample_rate = read_pcm16_wav(path)
        audio.append(samples)
        audio_inputs.append({"path": raw_path, "format": WAV_AUDIO_FORMAT, "bytes": path.stat().st_size, "sha256": digest(path), "samples": int(samples.size), "sample_rate_hz": sample_rate})
    conversation = []
    audio_index = 0
    for message in packet["messages"]:
        content = []
        for item in message["content"]:
            if item["type"] == "audio":
                # Keep the caller-owned ndarray attached to the official
                # conversation schema. With tokenize=False the pinned
                # ProcessorMixin renders without invoking load_audio; the
                # array is supplied to the official CSM processor below.
                content.append({
                    "type": "audio",
                    "audio": audio[audio_index],
                })
                audio_index += 1
            else:
                content.append(item)
        conversation.append({"role": message["role"], "content": content})
    if audio_index != len(audio):
        raise RuntimeError("conversation audio placeholder order/count mismatch")
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError("authenticated CSM processor has no chat-template boundary")
    chat_template_call = dict(OFFICIAL_CHAT_TEMPLATE_KWARGS)
    direct_processor_call = dict(OFFICIAL_DIRECT_PROCESSOR_KWARGS)
    validate_processor_calls(chat_template_call, direct_processor_call)
    # The pinned ProcessorMixin calls requires_backends(load_audio, ["librosa"])
    # even for ndarray inputs. Therefore tokenize=False is used for the
    # official template render, followed by the official CSM processor call
    # with the already authenticated caller-owned arrays. This is an adapter
    # around upstream code, not a reimplementation of either boundary.
    rendered_prompt = processor.apply_chat_template(conversation, **chat_template_call)
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise RuntimeError("official chat-template render returned no prompt")
    processor_audio = processor_audio_argument(audio)
    inputs = processor(text=rendered_prompt, audio=processor_audio, **direct_processor_call)
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.numel() == 0:
        raise RuntimeError("official processor returned no input_ids")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(input_ids.shape) or attention_mask.numel() == 0:
        raise RuntimeError("official processor returned no shape-matched attention_mask")
    if attention_mask.dtype not in (torch.int32, torch.int64, torch.long) or bool(((attention_mask != 0) & (attention_mask != 1)).any().item()):
        raise RuntimeError("official processor attention_mask must be a binary integer tensor")
    text_vocab = int(getattr(model.config, "text_vocab_size", 128_256))
    artifacts = [
        write_tensor(
            output / "processor_input_ids.u32le",
            input_ids,
            torch,
            "u32",
            max_value=text_vocab,
        ),
        write_tensor(
            output / "processor_attention_mask.u32le",
            attention_mask,
            torch,
            "u32",
            max_value=2,
        ),
    ]
    depth_logits: list[torch.Tensor] = []
    depth_input_shapes: list[list[int]] = []
    depth_input_ids: list[list[list[int]]] = []

    def capture_depth_logits(_module: Any, _args: Any, _kwargs: Any, output_value: Any) -> None:
        input_ids = depth_input_ids_from_hook(_args, _kwargs)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise RuntimeError("official depth decoder input shape is unavailable")
        depth_input_shapes.append(list(input_ids.shape))
        if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0 or input_ids.shape[1] > 2:
            raise RuntimeError("official depth decoder input cardinality exceeds source contract")
        depth_input_ids.append(input_ids.detach().cpu().tolist())
        logits_value = getattr(output_value, "logits", None)
        if logits_value is None and isinstance(output_value, (tuple, list)) and output_value:
            logits_value = output_value[0]
        if not isinstance(logits_value, torch.Tensor) or logits_value.numel() == 0:
            raise RuntimeError("official depth decoder call exposed no logits")
        if logits_value.ndim != 3 or logits_value.shape[0] <= 0 or logits_value.shape[1] <= 0 or logits_value.shape[2] <= 0:
            raise RuntimeError(f"official depth decoder logits have invalid shape: {tuple(logits_value.shape)}")
        depth_logits.append(logits_value[:, -1:, :].detach())

    depth_hook = model.depth_decoder.register_forward_hook(capture_depth_logits, with_kwargs=True)
    with torch.inference_mode():
        try:
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                depth_decoder_do_sample=False,
                output_hidden_states=True,
                output_logits=True,
                output_scores=True,
                return_dict_in_generate=True,
                output_audio=True,
            )
        finally:
            depth_hook.remove()
    sequences = getattr(generated, "sequences", None)
    if sequences is None and isinstance(generated, torch.Tensor):
        sequences = generated
    if sequences is None or not isinstance(sequences, torch.Tensor) or sequences.ndim != 3 or sequences.shape[0] <= 0 or sequences.shape[1] <= 0 or sequences.shape[2] != 32:
        raise RuntimeError("official CSM generated sequences must have shape [batch, frames, 32]")
    if sequences.shape[0] != 1:
        raise RuntimeError("reference packet format is mono; batch size must be exactly 1")
    frames = int(sequences.shape[1])
    if bool((sequences < 0).any().item()) or bool((sequences >= CODEBOOK_SIZE).any().item()):
        raise RuntimeError("official CSM generated code is outside the 32x2048 codec range")
    frame_rows = sequences[0].tolist()
    official_eos_frames = [index for index, frame in enumerate(frame_rows) if all(code == 0 for code in frame[:-1])]
    codec_eos_frames = [index for index, frame in enumerate(frame_rows) if all(code == 0 for code in frame)]
    if official_eos_frames and official_eos_frames != [frames - 1]:
        raise RuntimeError("CSM official 31-codebook EOS must terminate at the final generated frame")
    if codec_eos_frames and codec_eos_frames != [frames - 1]:
        raise RuntimeError("CSM codec 32-codebook EOS must terminate at the final generated frame")
    decoded_frames = codec_eos_frames[0] if codec_eos_frames else frames
    if decoded_frames <= 0:
        raise RuntimeError("CSM EOS leaves no decodable audio frames")
    artifacts.append(
        write_tensor(
            output / "generated_frame_codes.u32le",
            sequences,
            torch,
            "u32",
            max_value=CODEBOOK_SIZE,
        )
    )
    # Transformers strips the all-codebook EOS frame before Mimi decoding.
    # Keep an explicit decoded-only code artifact so frame cardinality cannot
    # accidentally be inferred from the EOS-inclusive generation tensor.
    artifacts.append(
        write_tensor(
            output / "decoded_frame_codes.u32le",
            sequences[:, :decoded_frames, :],
            torch,
            "u32",
            max_value=CODEBOOK_SIZE,
        )
    )
    logits = getattr(generated, "logits", None)
    if isinstance(logits, (list, tuple)) and logits:
        try:
            stacked_logits = stack_generation_logits(logits, torch)
        except ValueError as error:
            raise RuntimeError(f"official CSM logits have an unexpected structure: {error}") from error
        if stacked_logits.shape[0] != sequences.shape[0] or stacked_logits.shape[1] != frames:
            raise RuntimeError("backbone logits are not frame-aligned with generated codes")
        if tuple(stacked_logits.shape) != (int(sequences.shape[0]), frames, LOGIT_VOCAB_SIZE):
            raise RuntimeError(f"official backbone logits must be [batch,frames,{LOGIT_VOCAB_SIZE}]")
        artifacts.append(write_tensor(output / "backbone_logits.f32le", stacked_logits, torch))
        scores = getattr(generated, "scores", None)
        if isinstance(scores, (list, tuple)) and scores:
            try:
                stacked_scores = stack_generation_logits(scores, torch)
            except ValueError as error:
                raise RuntimeError(f"official CSM processed scores have an unexpected structure: {error}") from error
            if tuple(stacked_scores.shape) != (int(sequences.shape[0]), frames, LOGIT_VOCAB_SIZE):
                raise RuntimeError(f"official processed scores must be [batch,frames,{LOGIT_VOCAB_SIZE}]")
            artifacts.append(write_tensor(output / "backbone_scores.f32le", stacked_scores, torch))
            processed_backbone_codes = stacked_scores.argmax(dim=-1)
            if not torch.equal(processed_backbone_codes.to(dtype=sequences.dtype), sequences[:, :, 0]):
                raise RuntimeError("processed backbone scores differ from generated codebook 0")
        else:
            raise RuntimeError("official CSM generation exposed no processed backbone scores")
    else:
        raise RuntimeError("official CSM generation exposed no backbone logits")
    hidden = getattr(generated, "hidden_states", None)
    if not isinstance(hidden, (list, tuple)) or not hidden:
        raise RuntimeError("official CSM generation exposed no backbone hidden states")
    last_layers = []
    for step in hidden:
        if not isinstance(step, (list, tuple)) or not step or not isinstance(step[-1], torch.Tensor):
            raise RuntimeError("official CSM hidden states have an unexpected structure")
        last = step[-1]
        if last.ndim != 3 or last.shape[0] <= 0 or last.shape[1] <= 0 or last.shape[2] <= 0:
            raise RuntimeError(f"official CSM hidden state has invalid shape: {tuple(last.shape)}")
        # Generation's first hidden-state step is a full prompt prefill;
        # every later step is one generated frame.  Keep only the final
        # position of every step so the packet is frame-aligned.
        last_layers.append(last[:, -1:, :])
    if len(last_layers) != frames:
        raise RuntimeError("backbone hidden states are not frame-aligned")
    hidden_tensor = torch.cat(last_layers, dim=1)
    if tuple(hidden_tensor.shape) != (int(sequences.shape[0]), frames, 2_048):
        raise RuntimeError("backbone hidden state shape does not match generated frames")
    artifacts.append(write_tensor(output / "backbone_hidden_last.f32le", hidden_tensor, torch))
    if not depth_logits:
        raise RuntimeError("official CSM generation called no depth decoder")
    if len(depth_logits) != frames * 31:
        raise RuntimeError(f"official depth decoder call count must be frames*31: {len(depth_logits)} != {frames * 31}")
    depth_shape = tuple(depth_logits[0].shape)
    if depth_shape != (int(sequences.shape[0]), 1, LOGIT_VOCAB_SIZE):
        raise RuntimeError(f"official depth decoder logits must be [batch,1,{LOGIT_VOCAB_SIZE}], got {depth_shape}")
    if any(tuple(value.shape) != depth_shape for value in depth_logits):
        raise RuntimeError("official depth decoder logits have inconsistent call shapes")
    validate_depth_input_shapes(depth_input_shapes, int(sequences.shape[0]), frames)
    expected_depth_inputs: list[list[list[int]]] = []
    for frame in range(frames):
        expected_depth_inputs.append([[0, int(sequences[0, frame, 0].item())]])
        expected_depth_inputs.extend([[[int(sequences[0, frame, codebook].item())]] for codebook in range(1, 31)])
    if depth_input_ids != expected_depth_inputs:
        raise RuntimeError("depth decoder inputs do not follow source placeholder/previous-codebook order")
    selected_depth_codes = [int(value.argmax(dim=-1).item()) for value in depth_logits]
    expected_depth_codes = [int(sequences[0, index // 31, 1 + index % 31].item()) for index in range(frames * 31)]
    if selected_depth_codes != expected_depth_codes:
        raise RuntimeError("depth decoder argmax codes differ from generated codebooks 1..31")
    # ``generated.logits`` is intentionally retained as raw pre-processor
    # evidence. Selection is checked against ``generated.scores`` above;
    # unlike the depth hook, it is not inferred from raw logits.
    artifacts.append(
        write_tensor(output / "depth_decoder_logits.f32le", torch.stack(depth_logits, dim=0), torch)
    )
    audio_output = getattr(generated, "audio", None)
    if not isinstance(audio_output, (list, tuple)) or len(audio_output) != 1:
        raise RuntimeError("official CSM generation exposed no audio; PCM evidence is required")
    pcm = audio_output[0]
    if not isinstance(pcm, torch.Tensor) or pcm.ndim != 1 or pcm.numel() <= 0 or not bool(torch.isfinite(pcm).all().item()):
        raise RuntimeError("official CSM PCM must be a finite non-empty mono tensor")
    codec_config = getattr(model.config, "codec_config", {})
    if not isinstance(codec_config, dict) or codec_config.get("sampling_rate") != MIMI_SAMPLE_RATE:
        raise RuntimeError("authenticated Mimi sample-rate contract is not 24 kHz")
    if pcm.numel() != decoded_frames * MIMI_FRAME_HOP:
        raise RuntimeError("official CSM PCM sample count is not frame-aligned at 12.5 Hz")
    artifacts.append(write_tensor(output / "official_pcm_pre_watermark.f32le", pcm, torch))
    required = {
        "processor_input_ids.u32le",
        "processor_attention_mask.u32le",
        "generated_frame_codes.u32le",
        "decoded_frame_codes.u32le",
        "backbone_logits.f32le",
        "backbone_scores.f32le",
        "backbone_hidden_last.f32le",
        "depth_decoder_logits.f32le",
        "official_pcm_pre_watermark.f32le",
    }
    if {item["path"] for item in artifacts} != required:
        raise RuntimeError("reference artifact set is incomplete or duplicated")
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "reference_status": "REFERENCE_EVIDENCE_COMPLETE",
        "collection_status": "AUTHENTICATED",
        "comparison_status": "NOT_RUN_OFFICIAL_ONLY",
        "native_status": "BLOCKED_NATIVE_BINDING",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "publication": "NO_UPLOAD",
        "model": {"repository": HF_REPOSITORY, "revision": HF_REVISION},
        "tokenizer": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
            "tokenizer_json_git_blob_sha1": TOKENIZER_JSON_GIT_BLOB_SHA1,
            "tokenizer_json_sha256": digest(tokenizer_path),
            "status": "SNAPSHOT_AUTHENTICATED_BUT_NATIVE_ID_PARITY_UNACCEPTED",
            "chat_template_sha256": digest(snapshot / "chat_template.jinja"),
        },
        "transformers": {"commit": TRANSFORMERS_COMMIT},
        "reference_environment": {
            "python": "3.12",
            "lock_sha256": REFERENCE_LOCK_SHA256,
            "selection_status": "REVIEWED_ADAPTED_REFERENCE_ENVIRONMENT_NOT_UPSTREAM_REQUIREMENTS",
            "source_transformers_requirement": SOURCE_TRANSFORMERS_REQUIREMENT,
            "source_huggingface_hub_requirement": SOURCE_HUGGINGFACE_HUB_REQUIREMENT,
            "transformers_security_advisory": TRANSFORMERS_SECURITY_ADVISORY,
            "transformers_security_patched_minimum": TRANSFORMERS_SECURITY_PATCHED_MINIMUM,
            "isolated_transformers_pin": ISOLATED_TRANSFORMERS_PIN,
            "isolated_huggingface_hub_pin": ISOLATED_HUGGINGFACE_HUB_PIN,
            "transformers_compatibility_status": TRANSFORMERS_COMPATIBILITY_STATUS,
            "torch_index": PYTORCH_CPU_INDEX,
            "packages": {
                "numpy": np.__version__,
                # Lock identity is the installed distribution version. CUDA
                # wheels may append a local suffix to the runtime string, so
                # preserve that independently instead of conflating them.
                "torch_distribution": package_version("torch"),
                "torch_runtime": torch.__version__,
                "transformers": package_version("transformers"),
            },
            "locked_package_rows": package_rows,
            "locked_package_rows_sha256": package_rows_sha256,
            "license_audit_status": "BLOCKED_OWNER_POLICY_AND_NATIVE_NOTICE_REVIEW",
            "license_audit_rows": license_rows,
            "license_audit_rows_sha256": license_rows_sha256,
        },
        "loaded_tensor_manifest": {
            "count": len(loaded_rows),
            "sha256": loaded_manifest_hash,
            "inspection_sha256": inspection_manifest_hash,
            "status": "LOADED_STATE_NAMES_AND_SHAPES_BOUND_TO_AUTHENTICATED_SNAPSHOT",
        },
        "inspection_identity": {
            "source_repository": inspection.get("source_identity", {}).get("repository") if inspection else None,
            "source_revision": inspection.get("source_identity", {}).get("revision") if inspection else None,
            "transformers_repository": inspection.get("transformers_identity", {}).get("repository") if inspection else None,
            "transformers_tag": inspection.get("transformers_identity", {}).get("tag") if inspection else None,
            "transformers_commit": inspection.get("transformers_identity", {}).get("commit") if inspection else TRANSFORMERS_COMMIT,
            "transformers_version": "4.52.1",
            "generation_csm_git_blob_sha1": GENERATION_CSM_GIT_BLOB_SHA1,
        },
        "generation": {
            "input_boundary": "official processor.apply_chat_template(conversation, tokenize=False) then official CSM processor(text=rendered_prompt, audio=caller_numpy, sampling_rate=24000, return_tensors=pt)",
            "processor_call": {
                "method": "apply_chat_template_then_official_processor",
                "chat_template_kwargs": chat_template_call,
                "processor_kwargs": direct_processor_call,
                "audio_input": "authenticated caller-owned NumPy arrays",
                "audio_argument": "None when audio_placeholder_count=0; otherwise the ordered non-empty authenticated array list",
                "adapter": "pinned_upstream_ProcessorMixin_boundary; no_reference_mirror",
            },
            "processor_source": {
                "path": "src/transformers/processing_utils.py",
                "git_blob_sha1": PROCESSING_UTILS_GIT_BLOB_SHA1,
                "markers": list(PROCESSING_UTILS_MARKERS),
                "non_batched_text_argument": "text=prompt (a string after prompt=prompt[0])",
            },
            "processor_input_ids_shape": list(input_ids.shape),
            "processor_attention_mask_shape": list(attention_mask.shape),
            "processor_output_keys": sorted(str(key) for key in inputs.keys()),
            "conversation_audio_paths": [
                str((packet_root / raw_path).resolve()) for raw_path in packet["audio_paths"]
            ],
            "conversation_message_count": len(conversation),
            "audio_placeholder_count": len(audio),
            "decoded_audio_count": len(audio_output),
            "do_sample": False,
            "depth_decoder_do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "depth_decoder_call_count": len(depth_logits),
            "depth_decoder_call_order": list(range(len(depth_logits))),
            "backbone_hidden_generation_steps": len(last_layers),
            "depth_decoder_logits_shape_per_call": list(depth_shape),
            "generation_config_contract": generation_contract,
            "generation_config_sha256": inspection.get("evidence", {}).get("json_roles", {}).get("generation_config_sha256") if inspection else None,
            "logit_selection_contract": "backbone_processed_scores_argmax_selected_codebook0; backbone_raw_logits_recorded; depth_raw_hook_argmax_exact_trace",
            "depth_selection_contract": "depth_raw_hook_logits_argmax_matches_codebooks1_to31; processed_depth_scores_unavailable",
            "depth_decoder_input_shapes": depth_input_shapes,
            "depth_decoder_input_ids": depth_input_ids,
            "generated_sequence_shape": list(sequences.shape),
            "official_eos_frame_indices": official_eos_frames,
            "codec_eos_frame_indices": codec_eos_frames,
            "official_eos_codebook_count": 31,
            "codec_eos_codebook_count": 32,
            "decoded_frame_count_by_batch": [decoded_frames],
            "codec_decoded_frame_count_by_batch": [decoded_frames],
            "generated_frame_count": frames,
            "decoded_sequence_shape": [int(sequences.shape[0]), decoded_frames, int(sequences.shape[2])],
            "depth_decoder_frame_codebook_order": [{"call_index": index, "frame": index // 31, "codebook": 1 + index % 31} for index in range(len(depth_logits))],
            "pcm_frame_hop_samples": MIMI_FRAME_HOP,
            "pcm_samples": int(pcm.numel()),
        },
        "artifacts": artifacts,
        "packet_sha256": digest(args.packet),
        "audio_inputs": audio_inputs,
        "pcm_sample_rate_hz": MIMI_SAMPLE_RATE,
        "pcm_semantics": "Transformers codec-decoded PCM before the source CSM watermark/resample stage; not final watermarked PCM",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--transformers", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspection-manifest", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1125)
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.dependency_gate:
        try:
            lock_path = Path(__file__).resolve().parent / "csm_1b_reference" / "uv.lock"
            validate_dependency_gate(lock_path.parent / "pyproject.toml", lock_path)
            return 0
        except Exception as error:
            print(f"CSM dependency gate blocked: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
    if not all((args.snapshot, args.transformers, args.packet, args.output, args.inspection_manifest)):
        parser.error("reference requires --snapshot --transformers --packet --output --inspection-manifest")
    try:
        return run(args)
    except Exception as error:
        print(f"CSM reference blocked: {type(error).__name__}: {error}", file=sys.stderr)
        # Do not overwrite a non-empty directory after a failed run: stale
        # artifacts would otherwise look like fresh evidence to a worker.
        if args.output and (not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir()))):
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "manifest.json").write_text(json.dumps({
                "format": FORMAT,
                "status": "BLOCKED",
                "reference_status": "REFERENCE_ERROR",
                "collection_status": "FAILED",
                "comparison_status": "NOT_RUN_OFFICIAL_ONLY",
                "native_status": "BLOCKED_NATIVE_BINDING",
                "cpu_status": "UNSUPPORTED",
                "metal_status": "BLOCKED_BY_CPU",
                "publication": "NO_UPLOAD",
                "error": f"{type(error).__name__}: {error}",
                "tokenizer": {
                    "repository": TOKENIZER_REPOSITORY,
                    "revision": TOKENIZER_REVISION,
                    "tokenizer_json_git_blob_sha1": TOKENIZER_JSON_GIT_BLOB_SHA1,
                    "status": "SNAPSHOT_AUTHENTICATED_BUT_NATIVE_ID_PARITY_UNACCEPTED",
                },
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
