#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice2_reference python
"""Run pinned upstream CosyVoice2 and dump authenticated tensor taps.

This is an adapter, not a Python model mirror.  It wraps official methods and
modules only, restores every wrapper in ``finally``, and supports the bounded
non-streaming zero-shot/cross-lingual paths.  Native/public status stays
fail-closed after evidence is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import random
import re
import subprocess
import sys
import types
import tomllib
import wave
from pathlib import Path
from typing import Any

SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
MATCHA_REVISION = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
MODEL_REPOSITORY = "FunAudioLLM/CosyVoice2-0.5B"
FORMAT = "vokra-cosyvoice2-official-reference-v2"
LOCK_PATH = Path(__file__).with_name("cosyvoice2_reference") / "uv.lock"
EOS = 6561
OFFICIAL_CONFIG_SEED = 1986
OFFICIAL_TARGET_TEXT = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
OFFICIAL_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
REFERENCE_ENV = {
    "python": ">=3.12,<3.13",
    "torch": "2.3.1",
    "torchaudio": "2.3.1",
    "transformers": "4.40.1",
    "HyperPyYAML": "1.2.2",
    "conformer": "0.3.2",
    "diffusers": "0.29.0",
    "onnxruntime": "1.18.0",
    "lock": "tools/parity/cosyvoice2_reference/uv.lock (required on VAST)",
}
REQUIRED_ARTIFACTS = {
    "tokenizer_ids", "prompt_speech_tokens", "campplus_embedding",
    "qwen_prompt_embeddings", "ras_logits", "ras_pre_ras_probability",
    "ras_nucleus_probability",
    "qwen_prompt_speech_embeddings",
    "ras_calls",
    "ras_multinomial_probability",
    "generated_speech_tokens", "flow_rand_noise_full", "flow_rand_noise_slice",
    "flow_encoder_output", "cfm_terminal_mel", "prompt_mel", "generated_mel",
    "hift_input_mel", "hift_output_pcm",
    "official_output_pcm", "prompt_pcm16k",
}


def dependency_gate() -> int:
    """Refuse the unresolved reference closure before touching caller paths."""
    project_path = Path(__file__).with_name("cosyvoice2_reference") / "pyproject.toml"
    try:
        if not project_path.is_file() or project_path.is_symlink():
            raise RuntimeError("dedicated CosyVoice2 project is absent or symlinked")
        document = tomllib.loads(project_path.read_text(encoding="utf-8"))
        project = document["project"]
        if set(project) != {"name", "version", "description", "requires-python", "dependencies"}:
            raise RuntimeError("CosyVoice2 project schema drifted")
        if project["name"] != "vokra-cosyvoice2-reference" or project["version"] != "0.1.0":
            raise RuntimeError("CosyVoice2 project identity drifted")
        if project["requires-python"] != ">=3.12,<3.13" or not isinstance(project["dependencies"], list):
            raise RuntimeError("CosyVoice2 Python/dependency contract drifted")
        route = document["tool"]["vokra"]["cosyvoice2_reference"]
        audit = document["tool"]["vokra"]["cosyvoice2_reference"]["license_audit"]
        if route["source_revision"] != SOURCE_REVISION or route["matcha_revision"] != MATCHA_REVISION:
            raise RuntimeError("CosyVoice2 fixed source identity drifted")
        if route["lock_status"] != "BLOCKED_FORBIDDEN_SOXR_IN_AUTHENTICATED_OFFICIAL_CLOSURE":
            raise RuntimeError("CosyVoice2 closure status is not authenticated")
        if audit["status"] != "BLOCKED_UNRESOLVED":
            raise RuntimeError("CosyVoice2 license status is not fail-closed")
        lock_path = project_path.with_name("uv.lock")
        if lock_path.exists() or lock_path.is_symlink():
            raise RuntimeError("unexpected executable CosyVoice2 lock appeared")
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError, RuntimeError) as exc:
        print(f"cosyvoice2 dependency gate: BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("cosyvoice2 dependency gate: BLOCKED_UNRESOLVED", file=sys.stderr)
    return 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authenticate_lock() -> str:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink() or LOCK_PATH.stat().st_size == 0:
        raise RuntimeError("dedicated CosyVoice2 uv.lock is absent; refuse before reference execution")
    text = LOCK_PATH.read_text(encoding="utf-8")
    if 'requires-python = ">=3.12,<3.13"' not in text or "[[package]]" not in text:
        raise RuntimeError("dedicated CosyVoice2 uv.lock is not a complete Python 3.12 lock")
    for package, expected in REFERENCE_ENV.items():
        if package in {"python", "lock"}:
            continue
        lock_name = package.lower().replace("_", "-")
        pattern = rf'\[\[package\]\]\s+name = "{re.escape(lock_name)}"\s+version = "{re.escape(expected)}"'
        if re.search(pattern, text) is None:
            raise RuntimeError(f"dedicated CosyVoice2 uv.lock does not pin {package}=={expected}")
    return sha256_file(LOCK_PATH)


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def authenticate_checkout(root: Path, expected_origin: str) -> None:
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True):
        raise RuntimeError(f"source checkout is dirty: {root}")
    origin = subprocess.check_output(["git", "-C", str(root), "remote", "get-url", "origin"], text=True).strip()
    if origin.removesuffix("/").removesuffix(".git") != expected_origin.removesuffix("/").removesuffix(".git"):
        raise RuntimeError(f"source origin mismatch: got {origin!r}")


def authenticate_source(root: Path) -> dict[str, Any]:
    authenticate_checkout(root, "https://github.com/FunAudioLLM/CosyVoice")
    if git_revision(root) != SOURCE_REVISION:
        raise RuntimeError(f"source revision mismatch: expected {SOURCE_REVISION}")
    required = {
        "cosyvoice/cli/cosyvoice.py": ("CosyVoice2", "load_jit=False", "load_trt=False", "load_vllm=False", "fp16=False"),
        "cosyvoice/llm/llm.py": ("class Qwen2LM", "torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb]", "self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)", "if top_ids == self.speech_token_size:", "if top_ids > self.speech_token_size:"),
        "cosyvoice/flow/flow.py": ("feat = feat[:, :, mel_len1:]", "streaming", "finalize"),
        "cosyvoice/flow/flow_matching.py": ("from matcha.models.components.flow_matching import BASECFM", "class CausalConditionalCFM", "self.rand_noise = torch.randn([1, 80, 50 * 300])", "self.rand_noise[:, :, :mu.size(2)]"),
        "examples/libritts/cosyvoice2/conf/cosyvoice2.yaml": ("top_p: 0.8", "top_k: 25", "win_size: 10", "tau_r: 0.1"),
        "asset/zero_shot_prompt.wav": (),
        "vllm_example.py": (OFFICIAL_TARGET_TEXT, OFFICIAL_PROMPT_TEXT),
    }
    files: dict[str, Any] = {}
    for relative, needles in required.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing fixed source file: {relative}")
        if needles:
            text = path.read_text(encoding="utf-8")
            missing = [needle for needle in needles if needle not in text]
            if missing:
                raise RuntimeError(f"source contract missing in {relative}: {missing}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    gitlink = subprocess.check_output(["git", "-C", str(root), "ls-files", "-s", "third_party/Matcha-TTS"], text=True).strip().split()
    configured = subprocess.check_output(["git", "-C", str(root), "config", "-f", ".gitmodules", "--get", "submodule.third_party/Matcha-TTS.url"], text=True).strip()
    if len(gitlink) < 2 or gitlink[0] != "160000" or gitlink[1] != MATCHA_REVISION or configured.removesuffix("/").removesuffix(".git") != "https://github.com/shivammehta25/Matcha-TTS":
        raise RuntimeError("source Matcha gitlink/origin mismatch")
    with wave.open(str(root / "asset/zero_shot_prompt.wav"), "rb") as prompt:
        prompt_meta = {"channels": prompt.getnchannels(), "sample_width": prompt.getsampwidth(), "sample_rate": prompt.getframerate(), "samples": prompt.getnframes(), "compression": prompt.getcomptype()}
    if prompt_meta != {"channels": 1, "sample_width": 2, "sample_rate": 16000, "samples": prompt_meta["samples"], "compression": "NONE"}:
        raise RuntimeError(f"fixed prompt WAV metadata mismatch: {prompt_meta}")
    files["asset/zero_shot_prompt.wav"].update(prompt_meta, git_blob_sha1=git_blob(root / "asset/zero_shot_prompt.wav"))
    return {"repository": "FunAudioLLM/CosyVoice", "revision": SOURCE_REVISION, "resolved_revision": SOURCE_REVISION, "origin": "https://github.com/FunAudioLLM/CosyVoice", "clean": True, "matcha_gitlink": {"mode": gitlink[0], "revision": gitlink[1], "origin": configured}, "tree": sorted(files), "files": files}


def authenticate_matcha(root: Path) -> dict[str, Any]:
    authenticate_checkout(root, "https://github.com/shivammehta25/Matcha-TTS")
    if git_revision(root) != MATCHA_REVISION:
        raise RuntimeError(f"Matcha source revision mismatch: expected {MATCHA_REVISION}")
    files: dict[str, Any] = {}
    markers = {
        "matcha/models/components/flow_matching.py": ("class BASECFM", "def solve_euler"),
        "matcha/models/components/decoder.py": ("class Decoder", "def forward"),
    }
    for relative, needles in markers.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing fixed Matcha source file: {relative}")
        text = path.read_text(encoding="utf-8")
        if any(needle not in text for needle in needles):
            raise RuntimeError(f"Matcha constructor/provision contract missing in {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"repository": "shivammehta25/Matcha-TTS", "revision": MATCHA_REVISION, "resolved_revision": MATCHA_REVISION, "tree": sorted(files), "files": files}


def authenticate_model(model_dir: Path) -> dict[str, Any]:
    inspector_path = Path(__file__).with_name("cosyvoice2_inspect.py")
    spec = importlib.util.spec_from_file_location("cosyvoice2_inspector", inspector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load fixed snapshot identity table")
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    expected = inspector.EXPECTED
    local = {path.relative_to(model_dir).as_posix() for path in inspector.snapshot_files(model_dir)}
    if local != set(expected):
        raise RuntimeError(f"snapshot file set mismatch: missing={sorted(set(expected) - local)} extra={sorted(local - set(expected))}")
    files: dict[str, Any] = {}
    for relative, (size, blob, lfs) in expected.items():
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"snapshot identity mismatch: {relative}")
        actual = sha256_file(path) if lfs else git_blob(path)
        if actual != (lfs or blob):
            raise RuntimeError(f"snapshot digest mismatch: {relative}")
        files[relative] = {
            "bytes": size,
            "sha256": sha256_file(path),
            "git_blob_sha1": blob,
            "lfs_sha256": lfs,
        }
    return {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "resolved_revision": MODEL_REVISION, "tree": sorted(files), "files": files}


def load_packet(path: Path, source: Path) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate input packet key: {key}")
            result[key] = value
        return result

    packet = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    required = {"mode", "target_text", "prompt_text", "prompt_wav", "prompt_wav_sha256", "prompt_wav_rate", "prompt_wav_samples", "seed"}
    if not isinstance(packet, dict) or set(packet) != required:
        raise ValueError(f"input packet keys must be exactly {sorted(required)}")
    if packet["mode"] not in {"zero_shot", "cross_lingual"}:
        raise ValueError("mode must be zero_shot or cross_lingual")
    if not isinstance(packet["target_text"], str) or not packet["target_text"].strip() or len(packet["target_text"]) > 4096:
        raise ValueError("target_text must be non-empty UTF-8 text <=4096 chars")
    if not isinstance(packet["prompt_text"], str) or len(packet["prompt_text"]) > 4096:
        raise ValueError("prompt_text must be UTF-8 text <=4096 chars")
    if packet["mode"] == "zero_shot" and not packet["prompt_text"].strip():
        raise ValueError("zero_shot prompt_text must be non-empty")
    if packet["mode"] == "cross_lingual" and packet["prompt_text"].strip():
        raise ValueError("cross_lingual prompt_text must be empty; source removes prompt text")
    if packet["mode"] == "zero_shot" and (packet["target_text"] != OFFICIAL_TARGET_TEXT or packet["prompt_text"] != OFFICIAL_PROMPT_TEXT):
        raise ValueError("zero_shot packet text must match the fixed upstream vllm_example.py")
    if packet["prompt_wav"] != "asset/zero_shot_prompt.wav" or "\\" in packet["prompt_wav"]:
        raise ValueError("prompt_wav must be the fixed source asset/zero_shot_prompt.wav")
    prompt = (source / packet["prompt_wav"]).resolve()
    if source.resolve() not in prompt.parents or not prompt.is_file() or prompt.is_symlink():
        raise ValueError("prompt_wav must be a contained regular source file")
    if not isinstance(packet["prompt_wav_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", packet["prompt_wav_sha256"]):
        raise ValueError("prompt_wav_sha256 must be lowercase SHA-256")
    if sha256_file(prompt) != packet["prompt_wav_sha256"]:
        raise ValueError("prompt_wav SHA-256 mismatch")
    with wave.open(str(prompt), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000) or wav.getnframes() != packet["prompt_wav_samples"]:
            raise ValueError("prompt_wav metadata mismatch")
    if not isinstance(packet["prompt_wav_rate"], int) or packet["prompt_wav_rate"] != 16000 or not isinstance(packet["prompt_wav_samples"], int) or not 1600 <= packet["prompt_wav_samples"] <= 16_000 * 30:
        raise ValueError("prompt_wav must be mono 16 kHz PCM with 0.1..30 seconds")
    if packet["seed"] != OFFICIAL_CONFIG_SEED:
        raise ValueError("seed must match the fixed upstream cosyvoice2.yaml seed 1986")
    return packet


class Evidence:
    ALLOWED_ROLES = REQUIRED_ARTIFACTS | {"ras_fallback_probability", "prompt_pcm16k", "hift_cache_source"}

    def __init__(self, output: Path) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.meta: dict[str, Any] = {}
        self.execution_id: str | None = None

    def set_execution_id(self, execution_id: str) -> None:
        if not isinstance(execution_id, str) or not re.fullmatch(r"[0-9a-f]{64}", execution_id):
            raise ValueError("execution_id must be a complete SHA-256")
        self.execution_id = execution_id

    def _check_role(self, role: str) -> None:
        base = role.split("_aux", 1)[0]
        if role not in self.ALLOWED_ROLES and base not in self.ALLOWED_ROLES:
            raise RuntimeError(f"unexpected evidence role: {role}")

    def tensor(self, role: str, value: Any, source: str, metadata: dict[str, Any] | None = None) -> None:
        self._check_role(role)
        if not hasattr(value, "detach"):
            raise RuntimeError(f"official tap {role} is not a tensor")
        import torch

        if value.numel() == 0:
            raise RuntimeError(f"official tap {role} is an empty tensor")
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"official tap {role} contains a non-finite value")
        if role.endswith("pcm") or role == "official_output_pcm":
            if value.numel() and float(value.detach().abs().max()) > 1.1:
                raise RuntimeError(f"official tap {role} is outside the expected PCM bound")
        source_dtype = str(value.dtype)
        storage_dtype = source_dtype
        if source_dtype == "torch.bfloat16":
            # NumPy has no portable bfloat16 container. Preserve the source
            # dtype in the manifest while storing an explicit float32 copy.
            value = value.float()
            storage_dtype = "torch.float32"
        array = value.detach().cpu().contiguous().numpy()
        raw = array.tobytes(order="C")
        if not raw or len(raw) > 1 << 30:
            raise RuntimeError(f"official tap {role} has an invalid byte size")
        index = len(self.records.get(role, []))
        suffix = array.dtype.name
        path = self.output / f"{role}.{index}.{suffix}.bin"
        path.write_bytes(raw)
        record = {
            "path": path.name, "sha256": sha256_bytes(raw), "bytes": len(raw),
            "shape": list(array.shape), "dtype": source_dtype, "storage_dtype": storage_dtype,
            "source": source,
        }
        if metadata is not None:
            record["metadata"] = metadata
        if self.execution_id is not None:
            record["execution_id"] = self.execution_id
        self.records.setdefault(role, []).append(record)

    def json_value(self, role: str, value: Any, source: str) -> None:
        self._check_role(role)
        if self.execution_id is None:
            raise RuntimeError("execution_id must be set before evidence capture")
        self.records.setdefault(role, []).append({"value": value, "source": source, "execution_id": self.execution_id})

    def validate(self, required: set[str]) -> None:
        if self.execution_id is None:
            raise RuntimeError("evidence execution identity is missing")
        unexpected = {
            role for role in self.records
            if role not in self.ALLOWED_ROLES and role.split("_aux", 1)[0] not in self.ALLOWED_ROLES
        }
        if unexpected:
            raise RuntimeError(f"unexpected evidence roles: {sorted(unexpected)}")
        if not required <= set(self.records):
            raise RuntimeError(f"required official taps missing: {sorted(required - set(self.records))}")
        for role in required:
            if not self.records[role]:
                raise RuntimeError(f"required official tap is empty: {role}")
        dynamic_roles = {
            "ras_logits", "ras_pre_ras_probability",
            "ras_multinomial_probability", "ras_nucleus_probability", "ras_fallback_probability", "ras_calls",
            "qwen_prompt_speech_embeddings",
        }
        for role in required - dynamic_roles:
            if len(self.records[role]) != 1:
                raise RuntimeError(f"official tap has wrong cardinality: {role}")
        multinomial_rows = self.records["ras_multinomial_probability"]
        nucleus_rows = self.records["ras_nucleus_probability"]
        fallback_rows = self.records.get("ras_fallback_probability", [])
        if len(multinomial_rows) != len(nucleus_rows) + len(fallback_rows):
            raise RuntimeError("RAS multinomial metadata/tensor cardinality differs")
        referenced_paths: set[str] = set()
        widths = {
            "torch.float64": 8, "torch.float32": 4, "torch.float16": 2,
            "torch.bfloat16": 4, "torch.int64": 8, "torch.int32": 4,
            "torch.int16": 2, "torch.int8": 1, "torch.uint8": 1,
            "torch.bool": 1,
        }
        for role, rows in self.records.items():
            for row in rows:
                if row.get("execution_id") != self.execution_id:
                    raise RuntimeError(f"evidence execution identity mismatch: {role}")
                if "path" not in row:
                    continue
                path = self.output / row["path"]
                if row["path"] in referenced_paths:
                    raise RuntimeError(f"duplicate artifact path: {row['path']}")
                referenced_paths.add(row["path"])
                if not path.is_file() or path.stat().st_size != row["bytes"]:
                    raise RuntimeError(f"artifact path/size mismatch: {role}")
                if sha256_file(path) != row["sha256"]:
                    raise RuntimeError(f"artifact digest mismatch: {role}")
                if not isinstance(row.get("shape"), list) or not all(isinstance(dim, int) and dim >= 0 for dim in row["shape"]):
                    raise RuntimeError(f"artifact shape schema mismatch: {role}")
                if not isinstance(row.get("dtype"), str) or not isinstance(row.get("source"), str):
                    raise RuntimeError(f"artifact dtype/source schema mismatch: {role}")
                if row["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                    raise RuntimeError(f"artifact hash/size schema mismatch: {role}")
                width = widths.get(row.get("storage_dtype"))
                if width is None:
                    raise RuntimeError(f"unsupported artifact dtype: {role}")
                elements = 1
                for dimension in row["shape"]:
                    elements *= dimension
                if elements == 0 or elements * width != row["bytes"]:
                    raise RuntimeError(f"artifact shape/bytes mismatch: {role}")
        actual_paths = {
            path.name for path in self.output.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        if actual_paths != referenced_paths:
            raise RuntimeError(f"orphan or missing evidence files: {sorted(actual_paths ^ referenced_paths)}")

    def result(self, role: str, value: Any, source: str) -> None:
        """Capture the primary tensor and lengths from an official tuple."""
        values = value if isinstance(value, (tuple, list)) else (value,)
        tensors = [item for item in values if hasattr(item, "detach") and item.numel() != 0]
        if not tensors:
            raise RuntimeError(f"official result {role} has no non-empty tensor")
        self.tensor(role, tensors[0], source)
        for index, item in enumerate(tensors[1:], start=1):
            self.tensor(f"{role}_aux{index}", item, source)

    def manifest(self, status: str, **extra: Any) -> None:
        payload = {
            "format": FORMAT, "status": status, "reference_status": status,
            "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED",
            "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD",
            "artifacts": self.records, "observations": self.meta,
            "execution_id": self.execution_id,
            "artifact_manifest_sha256": sha256_bytes(json.dumps(self.records, sort_keys=True, separators=(",", ":")).encode()),
            **extra,
        }
        (self.output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_official(source: Path, matcha_source: Path, model_dir: Path, packet: dict[str, Any], evidence: Evidence) -> None:
    import numpy as np
    import torch

    sys.path.insert(0, str(source))
    sys.path.insert(0, str(matcha_source))
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from importlib.metadata import version

    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError(f"reference requires Python 3.12, got {platform.python_version()}")
    seed = packet["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = CosyVoice2(str(model_dir), load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    expected_versions = {name: REFERENCE_ENV[name] for name in ("torch", "torchaudio", "transformers", "HyperPyYAML", "conformer", "diffusers", "onnxruntime")}
    actual_versions = {name: version(name) for name in expected_versions}
    if actual_versions != expected_versions:
        raise RuntimeError(f"reference package versions differ from the dedicated lock: {actual_versions}")
    evidence.meta["python_version"] = platform.python_version()
    evidence.meta["actual_versions"] = actual_versions
    frontend, native_model, llm = model.frontend, model.model, model.model.llm
    patches: list[tuple[Any, str, Any]] = []
    hooks: list[Any] = []

    for name, role in (("_extract_text_token", "tokenizer_ids"), ("_extract_speech_token", "prompt_speech_tokens"), ("_extract_spk_embedding", "campplus_embedding"), ("_extract_speech_feat", "prompt_mel")):
        original = getattr(frontend, name)

        def wrapped(self: Any, *args: Any, _original=original, _role=role, _name=name, **kwargs: Any):
            result = _original(*args, **kwargs)
            evidence.result(_role, result, f"CosyVoiceFrontEnd.{_name}")
            return result

        patches.append((frontend, name, original))
        setattr(frontend, name, types.MethodType(wrapped, frontend))

    original_wrapper = llm.inference_wrapper

    def inference_wrapper_tap(self: Any, *args: Any, **kwargs: Any):
        yielded: list[int] = []
        min_len = int(args[2] if len(args) > 2 else kwargs["min_len"])
        max_len = int(args[3] if len(args) > 3 else kwargs["max_len"])
        evidence.meta["generation_bounds"] = {"min_outer_steps": min_len, "max_outer_steps": max_len}
        for token in original_wrapper(*args, **kwargs):
            value = int(token)
            if value >= EOS:
                raise RuntimeError("official wrapper yielded a control/EOS token")
            yielded.append(value)
            yield token
        calls = evidence.records.get("ras_calls", [])
        sampled = [int(row["value"]["selected_token"]) for row in calls]
        last = sampled[-1] if sampled else None
        evidence.meta["termination"] = {
            "reason": "eos" if last == EOS else "max_tokens",
            "eos": last == EOS,
            "yielded_count": len(yielded),
            "sampled_count": len(sampled),
            "sampled_tokens": sampled,
            "yielded_tokens": yielded,
            "yielded_special_ids": [],
        }
        evidence.tensor("generated_speech_tokens", torch.tensor(yielded, dtype=torch.int32), "Qwen2LM.inference_wrapper yielded IDs")

    patches.append((llm, "inference_wrapper", original_wrapper))
    setattr(llm, "inference_wrapper", types.MethodType(inference_wrapper_tap, llm))
    original_sampling_ids = llm.sampling_ids
    from cosyvoice.utils import common as common_utils
    original_nucleus = common_utils.nucleus_sampling
    original_random = common_utils.random_sampling
    ras_context: dict[str, Any] | None = None

    def random_tap(*args: Any, **kwargs: Any):
        nonlocal ras_context
        if ras_context is not None:
            ras_context["stage"] = "fallback"
        result = original_random(*args, **kwargs)
        if ras_context is not None:
            ras_context["fallback_selected"] = int(result)
        return result

    def sampling_ids_tap(self: Any, weighted_scores: Any, decoded_tokens: Any, sampling: Any, ignore_eos: bool = True):
        evidence.tensor("ras_logits", weighted_scores, "Qwen2LM.sampling_ids weighted_scores")
        step = len(evidence.meta.get("llm_calls", [])) - 1
        attempts = 0
        original_sampler = self.sampling
        sampler_keywords = getattr(original_sampler, "keywords", None)
        if not isinstance(sampler_keywords, dict) or sampler_keywords.get("top_p") != 0.8 or sampler_keywords.get("top_k") != 25 or sampler_keywords.get("win_size") != 10 or sampler_keywords.get("tau_r") != 0.1:
            raise RuntimeError(f"official RAS configuration is not the fixed top_p=.8/top_k=25/window=10/tau=.1: {sampler_keywords}")
        evidence.meta["ras_config"] = {"top_p": 0.8, "top_k": 25, "repetition_window": 10, "repetition_tau": 0.1, "domains": {"top_p": "(0,1]", "repetition_tau": "(0,inf)"}}
        ras_function = getattr(original_sampler, "func", original_sampler)
        ras_lines, ras_start = inspect.getsourcelines(ras_function)
        threshold_line = next(ras_start + index for index, line in enumerate(ras_lines) if "if rep_num >= win_size * tau_r" in line)

        def sampler_tap(scores: Any, prior_tokens: Any, sample_size: Any):
            nonlocal attempts, ras_context
            record: dict[str, Any] = {"generation_step": step, "attempt_index": attempts}
            ras_context = record
            previous_trace = sys.gettrace()

            def trace(frame: Any, event: str, _arg: Any):
                if frame.f_code is ras_function.__code__ and event == "line" and frame.f_lineno == threshold_line:
                    record["repetition_count"] = int(frame.f_locals.get("rep_num", -1))
                    record["repetition_window"] = int(frame.f_locals.get("win_size", -1))
                    record["repetition_threshold"] = float(frame.f_locals.get("win_size", 0) * frame.f_locals.get("tau_r", 0.0))
                    record["repetition_triggered"] = bool(record["repetition_count"] >= record["repetition_threshold"])
                return trace

            sys.settrace(trace)
            try:
                result = original_sampler(scores, prior_tokens, sample_size)
            except BaseException:
                ras_context = None
                raise
            finally:
                sys.settrace(previous_trace)
            selected = int(result.item())
            ras_context = None
            evidence.json_value("ras_calls", {
                "call_index": len(evidence.records.get("ras_calls", [])),
                "generation_step": step,
                "attempt_index": attempts,
                "selected_token": selected,
                "ignore_eos": bool(ignore_eos),
                "decoded_count": len(prior_tokens),
                "yielded": selected < EOS,
                "skipped": selected > EOS,
                "ignored_eos": selected == EOS and bool(ignore_eos),
                "stop": selected == EOS and not bool(ignore_eos),
                "nucleus_selected": record.get("nucleus_selected"),
                "fallback_selected": record.get("fallback_selected"),
                "repetition_count": record.get("repetition_count"),
                "repetition_window": record.get("repetition_window"),
                "repetition_threshold": record.get("repetition_threshold"),
                "repetition_triggered": record.get("repetition_triggered"),
            }, "Qwen2LM.sampling_ids -> official sampler")
            attempts += 1
            return result

        self.sampling = sampler_tap
        try:
            return original_sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos=ignore_eos)
        finally:
            self.sampling = original_sampler

    patches.append((llm, "sampling_ids", original_sampling_ids))
    setattr(llm, "sampling_ids", types.MethodType(sampling_ids_tap, llm))

    original_forward_one_step = llm.llm.forward_one_step

    def forward_one_step_tap(self: Any, *args: Any, **kwargs: Any):
        result = original_forward_one_step(*args, **kwargs)
        input_tensor = kwargs.get("lm_input", args[0] if args else None)
        output_tensor = result[0] if isinstance(result, tuple) else result
        calls = evidence.meta.setdefault("llm_calls", [])
        calls.append({
            "call_index": len(calls),
            "input_shape": list(input_tensor.shape) if hasattr(input_tensor, "shape") else None,
            "output_shape": list(output_tensor.shape) if hasattr(output_tensor, "shape") else None,
            "source": "Qwen2LM.llm.forward_one_step",
        })
        return result

    patches.append((llm.llm, "forward_one_step", original_forward_one_step))
    setattr(llm.llm, "forward_one_step", types.MethodType(forward_one_step_tap, llm.llm))

    # Capture the receiver of every actual Tensor.multinomial call.  The
    # surrounding official nucleus/random wrapper identifies which probability
    # tensor was actually passed; no softmax/top-k mirror is permitted.
    original_multinomial = torch.Tensor.multinomial

    def multinomial_tap(self: Any, *args: Any, **kwargs: Any):
        call_index = len(evidence.records.get("ras_multinomial_probability", []))
        if ras_context is None or ras_context.get("stage") not in {"nucleus", "fallback"}:
            raise RuntimeError("multinomial call escaped official RAS stage")
        stage = ras_context["stage"]
        role = "ras_nucleus_probability" if stage == "nucleus" else "ras_fallback_probability"
        evidence.tensor(role, self, f"torch.Tensor.multinomial receiver (official {stage} sampler)")
        evidence.json_value("ras_multinomial_probability", {
            "call_index": call_index,
            "probability_role": role,
            "stage": stage,
            "generation_step": ras_context["generation_step"],
            "attempt_index": ras_context["attempt_index"],
            "shape": list(self.shape),
            "dtype": str(self.dtype),
        }, "torch.Tensor.multinomial receiver (official sampler)")
        return original_multinomial(self, *args, **kwargs)

    patches.append((torch.Tensor, "multinomial", original_multinomial))
    setattr(torch.Tensor, "multinomial", multinomial_tap)

    # The source's `sorted_value` is the pre-RAS softmax result.  Capture that
    # local after its assignment and before top-p/top-k pruning; do not
    # recompute it from `weighted_scores` in the adapter.
    source_lines, source_start = inspect.getsourcelines(original_nucleus)
    pre_ras_line = next(
        source_start + index
        for index, line in enumerate(source_lines)
        if "for i in range(len(sorted_idx))" in line
    )

    def nucleus_tap(*args: Any, **kwargs: Any):
        nonlocal ras_context
        previous_trace = sys.gettrace()
        if ras_context is not None:
            ras_context["stage"] = "nucleus"

        def trace(frame: Any, event: str, _arg: Any):
            if frame.f_code is original_nucleus.__code__ and event == "line" and frame.f_lineno == pre_ras_line:
                probability = frame.f_locals.get("sorted_value")
                if probability is not None:
                    evidence.tensor("ras_pre_ras_probability", probability, "official nucleus_sampling sorted_value")
            return trace

        sys.settrace(trace)
        try:
            result = original_nucleus(*args, **kwargs)
            if ras_context is not None:
                ras_context["nucleus_selected"] = int(result)
            return result
        finally:
            sys.settrace(previous_trace)

    patches.append((common_utils, "nucleus_sampling", original_nucleus))
    patches.append((common_utils, "random_sampling", original_random))
    common_utils.random_sampling = random_tap
    setattr(common_utils, "nucleus_sampling", nucleus_tap)

    def hook(role: str, source_name: str):
        def callback(_module: Any, _inputs: Any, output: Any) -> None:
            values = output if isinstance(output, (tuple, list)) else (output,)
            for value in values:
                if hasattr(value, "detach"):
                    evidence.tensor(role, value, source_name)
                    if role == "flow_encoder_output":
                        evidence.meta["flow_encoder_shape"] = list(value.shape)
                    break

        return callback

    hooks.extend([
        llm.llm.model.model.embed_tokens.register_forward_hook(hook("qwen_prompt_embeddings", "Qwen2LM.llm.model.model.embed_tokens")),
        llm.speech_embedding.register_forward_hook(hook("qwen_prompt_speech_embeddings", "Qwen2LM.speech_embedding")),
        native_model.flow.encoder.register_forward_hook(hook("flow_encoder_output", "CausalMaskedDiffWithXvec.encoder")),
        native_model.flow.decoder.register_forward_hook(hook("cfm_terminal_mel", "CausalConditionalCFM.forward")),
    ])

    estimator_calls: list[dict[str, Any]] = []
    def estimator_hook(_module: Any, inputs: Any, output: Any) -> None:
        x = inputs[0] if inputs else None
        values = output if isinstance(output, (tuple, list)) else (output,)
        estimate = next((value for value in values if hasattr(value, "shape")), None)
        if x is None or estimate is None:
            raise RuntimeError("CFM estimator tap lacks x/output tensors")
        estimator_calls.append({"input_shape": list(x.shape), "output_shape": list(estimate.shape)})
    hooks.append(native_model.flow.decoder.estimator.register_forward_hook(estimator_hook))

    # The source creates this once in CausalConditionalCFM.__init__ after
    # set_all_random_seed(0), then uses exactly rand_noise[:, :, :T]. Capture
    # both tensors at the official decoder boundary; never substitute a new
    # caller-generated packet.
    original_decoder_forward = native_model.flow.decoder.forward

    def decoder_forward_tap(self: Any, *args: Any, **kwargs: Any):
        mu = kwargs.get("mu", args[0] if args else None)
        noise = self.rand_noise
        evidence.tensor("flow_rand_noise_full", noise, "CausalConditionalCFM.rand_noise initialized by upstream")
        if mu is not None:
            evidence.tensor("flow_rand_noise_slice", noise[:, :, : mu.size(2)], "CausalConditionalCFM.forward rand_noise[:, :, :mu.size(2)]")
        return original_decoder_forward(*args, **kwargs)

    patches.append((native_model.flow.decoder, "forward", original_decoder_forward))
    setattr(native_model.flow.decoder, "forward", types.MethodType(decoder_forward_tap, native_model.flow.decoder))

    original_solve_euler = native_model.flow.decoder.solve_euler
    def solve_euler_tap(self: Any, *args: Any, **kwargs: Any):
        t_span = kwargs.get("t_span", args[1] if len(args) > 1 else None)
        if t_span is None or not hasattr(t_span, "shape"):
            raise RuntimeError("CFM Euler tap lacks official t_span")
        evidence.meta["cfm_time_grid"] = [float(value) for value in t_span.detach().cpu().tolist()]
        evidence.meta["cfm_steps"] = int(t_span.numel() - 1)
        return original_solve_euler(*args, **kwargs)
    patches.append((native_model.flow.decoder, "solve_euler", original_solve_euler))
    setattr(native_model.flow.decoder, "solve_euler", types.MethodType(solve_euler_tap, native_model.flow.decoder))

    original_flow = native_model.flow.inference

    def flow_tap(self: Any, *args: Any, **kwargs: Any):
        if kwargs.get("streaming") is not False or kwargs.get("finalize") is not True:
            raise RuntimeError("CosyVoice2 evidence is restricted to streaming=false, finalize=true")
        prompt_feat = kwargs.get("prompt_feat")
        result = original_flow(*args, **kwargs)
        if prompt_feat is None or prompt_feat.ndim != 3 or result[0].ndim != 3 or result[0].shape[1] != 80:
            raise RuntimeError("official flow axes are not [batch,80,frames]")
        encoder_shape = evidence.meta.get("flow_encoder_shape")
        if not isinstance(encoder_shape, list) or len(encoder_shape) != 3 or encoder_shape[0] != 1:
            raise RuntimeError("official flow encoder shape evidence is missing")
        full_frames = int(encoder_shape[1])
        prompt_frames = int(prompt_feat.shape[1])
        generated_frames = int(result[0].shape[2])
        if full_frames != prompt_frames + generated_frames:
            raise RuntimeError("official finalize=true flow frame subtraction is inconsistent")
        evidence.tensor("generated_mel", result[0], "CausalMaskedDiffWithXvec.inference returned feat[:,:,mel_len1:]")
        evidence.meta["flow_return_contract"] = {"prompt_slice": "official", "double_slice": False, "streaming": False, "finalize": True, "encoder_full_frames": full_frames, "prompt_frames": prompt_frames, "generated_frames": generated_frames, "relation": "generated_frames = encoder_full_frames - prompt_frames"}
        return result

    patches.append((native_model.flow, "inference", original_flow))
    setattr(native_model.flow, "inference", types.MethodType(flow_tap, native_model.flow))
    original_hift = native_model.hift.inference

    def hift_tap(self: Any, *args: Any, **kwargs: Any):
        mel = kwargs.get("speech_feat", args[0] if args else None)
        cache_source = kwargs.get("cache_source")
        if mel is not None:
            evidence.tensor("hift_input_mel", mel, "HiFTGenerator.inference speech_feat")
        if cache_source is not None and cache_source.numel() != 0:
            evidence.tensor("hift_cache_source", cache_source, "HiFTGenerator.inference cache_source")
        result = original_hift(*args, **kwargs)
        if isinstance(result, tuple):
            pcm = result[0]
            if mel is None or pcm.ndim != 2 or pcm.shape[0] != 1 or int(pcm.shape[-1]) != int(mel.shape[-1]) * 480:
                raise RuntimeError("official HiFT PCM is not mono or mel_frames*480")
            evidence.tensor("hift_output_pcm", pcm, "HiFTGenerator.inference generated_speech", {"sample_rate": 24000, "channels": 1, "samples": int(pcm.shape[-1]), "mel_frames": int(mel.shape[-1])})
        return result

    patches.append((native_model.hift, "inference", original_hift))
    setattr(native_model.hift, "inference", types.MethodType(hift_tap, native_model.hift))
    original_token2wav = native_model.token2wav

    def token2wav_tap(self: Any, *args: Any, **kwargs: Any):
        evidence.json_value(
            "token2wav_calls",
            {"token_offset": kwargs.get("token_offset"), "stream": kwargs.get("stream"), "finalize": kwargs.get("finalize")},
            "CosyVoice2Model.token2wav",
        )
        return original_token2wav(*args, **kwargs)

    patches.append((native_model, "token2wav", original_token2wav))
    setattr(native_model, "token2wav", types.MethodType(token2wav_tap, native_model))

    with wave.open(str(source / packet["prompt_wav"]), "rb") as prompt_file:
        raw = prompt_file.readframes(prompt_file.getnframes())
    pcm = torch.frombuffer(raw, dtype=torch.int16).clone().to(torch.float32).div_(32768.0).reshape(1, -1)
    evidence.tensor("prompt_pcm16k", pcm, "strict input packet", {"sample_rate": 16000, "channels": 1, "samples": int(pcm.shape[-1])})
    try:
        if packet["mode"] == "zero_shot":
            outputs = model.inference_zero_shot(packet["target_text"], packet["prompt_text"], pcm, stream=False)
        else:
            outputs = model.inference_cross_lingual(packet["target_text"], pcm, stream=False)
        count = 0
        for output in outputs:
            count += 1
            if not isinstance(output, dict) or "tts_speech" not in output:
                raise RuntimeError("official inference output lacks tts_speech")
            if output["tts_speech"].ndim != 2 or output["tts_speech"].shape[0] != 1:
                raise RuntimeError("official output PCM is not mono [1,samples]")
            evidence.tensor(
                "official_output_pcm",
                output["tts_speech"],
                "CosyVoice2.inference_* tts_speech (codec-decoded pre-watermark PCM; not source final watermarked PCM)",
                {"sample_rate": 24000, "channels": 1, "samples": int(output["tts_speech"].shape[-1])},
            )
        if count != 1:
            raise RuntimeError(f"non-streaming official inference must yield exactly one output, got {count}")
    finally:
        for obj, name, original in reversed(patches):
            setattr(obj, name, original)
        for handle in hooks:
            handle.remove()


def run(source: Path, matcha_source: Path, model_dir: Path, input_path: Path, output: Path) -> int:
    if dependency_gate() != 0:
        return 2
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f"cosyvoice2 reference BLOCKED: evidence output must be absent or empty: {output}", file=sys.stderr)
        return 2
    evidence = Evidence(output)
    try:
        lock_sha = authenticate_lock()
        packet = load_packet(input_path, source)
        source_identity = authenticate_source(source)
        matcha_identity = authenticate_matcha(matcha_source)
        model_identity = authenticate_model(model_dir)
        evidence.set_execution_id(sha256_bytes(json.dumps({
            "source": source_identity,
            "matcha": matcha_identity,
            "model": model_identity,
            "input": packet,
        }, sort_keys=True, separators=(",", ":")).encode()))
        run_official(source, matcha_source, model_dir, packet, evidence)
        evidence.validate(REQUIRED_ARTIFACTS)
        for observation in ("termination", "flow_return_contract", "token2wav_calls"):
            if observation not in evidence.meta:
                raise RuntimeError(f"required official observation missing: {observation}")
        sampling_calls = evidence.records["ras_calls"]
        llm_calls = evidence.meta.get("llm_calls")
        if not isinstance(llm_calls, list) or not llm_calls:
            raise RuntimeError("exact official LLM call trace is missing")
        if any(call.get("call_index") != index for index, call in enumerate(llm_calls)):
            raise RuntimeError("official LLM call trace is not ordered")
        bounds = evidence.meta.get("generation_bounds")
        if not isinstance(bounds, dict) or not isinstance(bounds.get("min_outer_steps"), int) or not isinstance(bounds.get("max_outer_steps"), int) or not 0 <= bounds["min_outer_steps"] <= bounds["max_outer_steps"] or not 1 <= len(llm_calls) <= bounds["max_outer_steps"]:
            raise RuntimeError("official outer generation bounds are missing or malformed")
        for index, call in enumerate(llm_calls):
            input_shape, output_shape = call.get("input_shape"), call.get("output_shape")
            if not isinstance(input_shape, list) or len(input_shape) != 3 or not isinstance(output_shape, list) or len(output_shape) != 3 or input_shape[0] != 1 or output_shape[0] != 1 or input_shape[1] != output_shape[1] or input_shape[2] != output_shape[2] or input_shape[1] <= 0:
                raise RuntimeError("forward_one_step output rows do not match input rows")
            if index > 0:
                previous = llm_calls[index - 1]
                previous_attempts = [
                    row["value"]
                    for row in evidence.records["ras_calls"]
                    if row["value"]["generation_step"] == index - 1
                ]
                if not previous_attempts:
                    raise RuntimeError("previous outer step has no RAS attempts")
                # The source does not emit an ``accepted`` marker.  The last
                # attempt is the only accepted result after any ignored-EOS
                # retry, so derive the next input shape from that explicit
                # position rather than a self-asserted field.
                previous_step = previous_attempts[-1]
                expected_rows = 1 if previous_step["selected_token"] < EOS else previous["input_shape"][1]
                if input_shape[1] != expected_rows:
                    raise RuntimeError("lm_input rows do not follow yielded/control state")
        if len(evidence.records["ras_logits"]) != len(llm_calls):
            raise RuntimeError("LLM forward and pre-RAS logits cardinalities differ")
        if any(row.get("value", {}).get("call_index") != index for index, row in enumerate(sampling_calls)):
            raise RuntimeError("official sampler call trace is not ordered")
        evidence.meta["sampling_call_count"] = len(sampling_calls)
        evidence.meta["llm_call_count"] = len(llm_calls)
        if len(sampling_calls) == 0:
            raise RuntimeError("official sampler produced no attempts")
        by_step: dict[int, list[dict[str, Any]]] = {}
        for index, row in enumerate(sampling_calls):
            value = row.get("value")
            if not isinstance(value, dict) or value.get("call_index") != index:
                raise RuntimeError("official sampler call trace is not ordered")
            by_step.setdefault(int(value["generation_step"]), []).append(value)
        if set(by_step) != set(range(len(llm_calls))):
            raise RuntimeError("RAS attempts do not cover exactly one group per outer step")
        for step, attempts_for_step in by_step.items():
            for attempt_index, value in enumerate(attempts_for_step):
                is_last_attempt = attempt_index == len(attempts_for_step) - 1
                selected = value.get("selected_token")
                if not isinstance(selected, int) or not 0 <= selected <= EOS + 2 or value.get("attempt_index") != attempt_index or value.get("ignore_eos") != (step < bounds["min_outer_steps"]):
                    raise RuntimeError("RAS attempt state does not match source outer step")
                if value.get("decoded_count") != len([token for prior in sampling_calls[:value["call_index"]] for token in ([prior["value"]["selected_token"]] if prior["value"]["selected_token"] < EOS else [])]):
                    raise RuntimeError("RAS decoded-token count is not the yielded history")
                if value.get("yielded") != (selected < EOS) or value.get("skipped") != (selected > EOS) or value.get("ignored_eos") != (selected == EOS and value["ignore_eos"]) or value.get("stop") != (selected == EOS and not value["ignore_eos"]):
                    raise RuntimeError("RAS EOS/control flags do not match source state machine")
                if attempt_index > 0 and attempts_for_step[attempt_index - 1]["selected_token"] != EOS:
                    raise RuntimeError("RAS retried a non-EOS result in one outer step")
                if not is_last_attempt and not (
                    selected == EOS
                    and value.get("ignored_eos") is True
                    and value.get("stop") is False
                ):
                    raise RuntimeError("non-final RAS attempt must be an ignored EOS retry")
                if is_last_attempt and value.get("ignored_eos") is True:
                    raise RuntimeError("final RAS attempt cannot remain an ignored EOS retry")
                if value.get("selected_token") == EOS and value.get("ignore_eos") and value.get("stop"):
                    raise RuntimeError("ignored EOS was marked as terminal")
                if value.get("nucleus_selected") is None or value.get("repetition_count") is None or value.get("repetition_window") is None or value.get("repetition_threshold") is None or value.get("repetition_triggered") is None:
                    raise RuntimeError("RAS repetition/nucleus evidence is incomplete")
                triggered = bool(value["repetition_triggered"])
                if (value.get("fallback_selected") is None) != (not triggered):
                    raise RuntimeError("RAS fallback selection does not match repetition trigger")
                expected_selected = value["fallback_selected"] if triggered else value["nucleus_selected"]
                if value["selected_token"] != expected_selected:
                    raise RuntimeError("RAS final ID does not match official fallback/nucleus result")
        terminal = [row["value"] for row in sampling_calls if row["value"].get("stop")]
        termination = evidence.meta["termination"]
        if termination["reason"] == "eos":
            if len(terminal) != 1 or terminal[0]["generation_step"] < bounds["min_outer_steps"] or len(llm_calls) != terminal[0]["generation_step"] + 1:
                raise RuntimeError("official EOS termination does not match outer-step state")
        elif termination["reason"] == "max_tokens":
            if terminal or len(llm_calls) != bounds["max_outer_steps"]:
                raise RuntimeError("official max termination does not consume max outer steps")
        else:
            raise RuntimeError("unknown official termination reason")
        evidence.meta["sampled_vs_yielded"] = {
            "sampled": [row["value"]["selected_token"] for row in sampling_calls],
            "yielded": evidence.meta["termination"]["yielded_tokens"],
            "filtered_controls": [row["value"]["selected_token"] for row in sampling_calls if row["value"]["selected_token"] >= EOS],
        }
        if not evidence.records["ras_multinomial_probability"]:
            raise RuntimeError("official multinomial call trace is missing")
        noise = evidence.records["flow_rand_noise_full"]
        if not noise or noise[0]["shape"] != [1, 80, 15000]:
            raise RuntimeError("fixed rand_noise artifact is not complete [1,80,15000]")
        grid = evidence.meta.get("cfm_time_grid")
        if evidence.meta.get("cfm_steps") != 10 or not isinstance(grid, list) or len(grid) != 11 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in grid) or not math.isclose(grid[0], 0.0, abs_tol=1e-6) or not math.isclose(grid[-1], 1.0, abs_tol=1e-6) or any(grid[index] >= grid[index + 1] for index in range(len(grid) - 1)) or any(not math.isclose(value, 1.0 - math.cos(index / 10.0 * 0.5 * math.pi), rel_tol=1e-6, abs_tol=1e-6) for index, value in enumerate(grid)):
            raise RuntimeError("official CFM cosine Euler time grid is missing or malformed")
        if len(estimator_calls) != 10 or any(row["input_shape"][0] != 2 or row["output_shape"][0] != 2 for row in estimator_calls):
            raise RuntimeError("official CFM estimator did not execute exactly ten two-row CFG steps")
        evidence.meta["cfm_estimator_calls"] = estimator_calls
        evidence.manifest(
            "AUTHENTICATED_REFERENCE_EVIDENCE",
            source=source_identity,
            matcha=matcha_identity,
            model=model_identity,
            reference_environment={**REFERENCE_ENV, "python_version": evidence.meta.get("python_version"), "lock_sha256": lock_sha, "actual_versions": evidence.meta.get("actual_versions")},
            input={"mode": packet["mode"], "target_text": packet["target_text"], "prompt_text": packet["prompt_text"], "seed": packet["seed"], "prompt_wav": packet["prompt_wav"], "prompt_wav_sha256": packet["prompt_wav_sha256"], "prompt_wav_rate": packet["prompt_wav_rate"], "prompt_wav_samples": packet["prompt_wav_samples"]},
            flow_noise={"shape": [1, 80, 15000], "slice": "[:, :, :mu.size(2)]", "seed": 0, "source": "CausalConditionalCFM.rand_noise"},
            pcm_semantics={
                "official_output_pcm": "codec-decoded pre-watermark PCM from upstream tts_speech; this evidence does not claim source final watermarked PCM",
                "hift_output_pcm": "HiFT decoder PCM before any external watermark stage",
            },
            route={"mode": packet["mode"], "streaming": False, "finalize": True, "native_status": "BLOCKED"},
        )
        return 0
    except Exception as exc:
        evidence.manifest("BLOCKED", blockers=[str(exc)], source={"revision": SOURCE_REVISION}, model={"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}, route={"native_status": "BLOCKED", "parity_status": "NOT_RUN"})
        print(f"cosyvoice2 reference BLOCKED: {exc}", file=sys.stderr)
        return 2


def self_test() -> None:
    assert dependency_gate() == 2
    assert EOS == 6561 and len(SOURCE_REVISION) == len(MODEL_REVISION) == len(MATCHA_REVISION) == 40
    assert {"flow_rand_noise_full", "qwen_prompt_embeddings", "ras_pre_ras_probability", "ras_nucleus_probability"} <= REQUIRED_ARTIFACTS
    assert "torch.Tensor.multinomial" in Path(__file__).read_text(encoding="utf-8")
    assert REFERENCE_ENV["torch"] == REFERENCE_ENV["torchaudio"] == "2.3.1"
    assert REFERENCE_ENV["transformers"] == "4.40.1"
    print("cosyvoice2_dump_reference.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--matcha-source", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reference"))
    args = parser.parse_args()
    if args.dependency_gate:
        if any(value is not None for value in (args.source, args.model_dir, args.matcha_source, args.input)) or args.self_test or args.output != Path("reference"):
            parser.error("--dependency-gate accepts no other arguments")
        return dependency_gate()
    if args.self_test:
        if any(value is not None for value in (args.source, args.model_dir, args.matcha_source, args.input)) or args.output != Path("reference"):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if not args.source or not args.matcha_source or not args.model_dir or not args.input:
        parser.error("--source, --matcha-source, --model-dir, and --input are required")
    return run(args.source, args.matcha_source, args.model_dir, args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
