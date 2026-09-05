#!/usr/bin/env -S uv run --frozen --project tools/parity/dia_1_6b_reference python
"""Execute the pinned official Dia source and emit same-execution evidence.

This file is an adapter around ``dia.model.Dia``.  It does not mirror model
math.  The worker must provide an authenticated source checkout, the exact HF
snapshot, and an independently authenticated DAC checkout/package.  Missing
or incomplete evidence is an error; no public/native readiness is asserted.
The dedicated ``tools/parity/dia_1_6b_reference/uv.lock`` is mandatory before
the worker may download or execute anything.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import tomllib
import types
from pathlib import Path
from typing import Any

HF_REPOSITORY = "nari-labs/Dia-1.6B"
HF_REVISION = "257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd"
SOURCE_REPOSITORY = "https://github.com/nari-labs/dia.git"
SOURCE_REVISION = "2811af1c5f476b1f49f4744fabf56cf352be21e5"
PUBLIC_REPOSITORY = "vokra/dia-1.6b"
PUBLIC_REVISION = "dd1df2a129fed7d15c365caeabaae227ccfe8537"
DEFAULT_TEXT = "[S1] Hello from speaker one. [S2] Hello from speaker two."
SOURCE_ROLE_BLOBS = {
    "LICENSE": "483d716cc886695f19971a99658c59851a8a2866",
    "dia/audio.py": "5c1947103bc0d95255d97618c699fa0a18993beb",
    "dia/config.py": "09c6d136a41e0296483d2617061d4261cbf4c42c",
    "dia/layers.py": "f9aed506b25e99d053dd71d6def7a0bd33075ace",
    "dia/model.py": "a3b0f9730a810fa170019511a2696e7f813090de",
    "dia/state.py": "172ec52c7c344781aad0552a6cddd6e5f1933894",
    "pyproject.toml": "dd844dd2fb0ab0c016520c4b070beaa7c159e3e1",
}
FORMAT = "vokra-dia-1-6b-official-reference-v1"
COMPARISON_STATUS = "NOT_RUN_OFFICIAL_ONLY"
REFERENCE_PROJECT_LOCK_SHA256 = "ccdfaf4cfedd7780f8c1032a42341f28ac56bec7353f4563f9a1b44b764cf29c"
REFERENCE_PROJECT_PYPROJECT_SHA256 = "56430b6f50620df9ce3383f535dec1755843a4a9bab9758e34cf69e9913b6fc2"
DEPENDENCY_LICENSE_AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
DIRECT_DEPENDENCY_VERSIONS = {
    "einops": "0.8.2", "gguf": "0.19.0", "huggingface-hub": "0.30.2",
    "numpy": "2.2.5", "pydantic": "2.11.3", "soundfile": "0.13.1",
    "torch": "2.6.0+cpu", "torchaudio": "2.6.0+cpu",
}
# Conclusions are deliberately conservative.  A package may have an
# otherwise permissive top-level license while its native/bundled components
# still require a separate primary-source review.  No row below grants the
# execution gate; the project remains BLOCKED_UNREVIEWED_TRANSITIVE.
DEPENDENCY_LICENSE_CONCLUSIONS = {
    "annotated-types": "MIT_REVIEWED",
    "certifi": "MPL-2.0_BLOCKED_BY_POLICY",
    "cffi": "MIT_NATIVE_LIBFFI_REVIEW_REQUIRED",
    "charset-normalizer": "MIT_REVIEWED",
    "colorama": "BSD-3-Clause_REVIEWED",
    "einops": "MIT_REVIEWED",
    "filelock": "UNLICENSE_POLICY_REVIEW_REQUIRED",
    "fsspec": "BSD-3-Clause_REVIEWED",
    "gguf": "MIT_REVIEWED",
    "huggingface-hub": "Apache-2.0_REVIEWED",
    "idna": "BSD-3-Clause_REVIEWED",
    "jinja2": "BSD-3-Clause_REVIEWED",
    "markupsafe": "BSD-3-Clause_REVIEWED",
    "mpmath": "BSD_STYLE_PRIMARY_REVIEW_REQUIRED",
    "networkx": "BSD-3-Clause_REVIEWED",
    "numpy": "BSD-3-Clause_NATIVE_BUNDLE_REVIEW_REQUIRED",
    "packaging": "Apache-2.0_REVIEWED",
    "pycparser": "BSD-3-Clause_REVIEWED",
    "pydantic": "MIT_REVIEWED",
    "pydantic-core": "MIT_NATIVE_EXTENSION_REVIEW_REQUIRED",
    "pyyaml": "MIT_NATIVE_EXTENSION_REVIEW_REQUIRED",
    "requests": "Apache-2.0_REVIEWED",
    "setuptools": "MIT_REVIEWED",
    "soundfile": "BSD-3-Clause_NATIVE_LIBSNDFILE_REVIEW_REQUIRED",
    "sympy": "BSD-3-Clause_REVIEWED",
    "torch": "BSD-3-Clause_BUNDLED_COMPONENT_REVIEW_REQUIRED",
    "torchaudio": "BSD-2-Clause_BUNDLED_COMPONENT_REVIEW_REQUIRED",
    "tqdm": "MPL-2.0_OR_MIT_POLICY_REVIEW_REQUIRED",
    "typing-extensions": "PSF-2.0_BLOCKED_BY_POLICY",
    "typing-inspection": "MIT_REVIEWED",
    "urllib3": "MIT_REVIEWED",
    "vokra-dia-1-6b-reference": "FIRST_PARTY_NOT_INDEPENDENT_DEPENDENCY_SCOPE",
}
REQUIRED_ARTIFACTS = {
    "text_ids", "text_padding_mask",
    "conditional_encoder", "unconditional_encoder",
    "decoder_logits", "decoder_sampling_probability", "selected_ids",
    "delayed_codes", "reverted_codes", "dac_latent", "pcm",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lock_dependency_audit(lock: Path) -> dict[str, Any]:
    """Emit canonical, duplicate-safe rows for every resolved lock package."""
    document = tomllib.loads(lock.read_text(encoding="utf-8"))
    rows = []
    for package in document.get("package", []):
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise RuntimeError("uv.lock contains an invalid package row")
        name = package["name"]
        if name not in DEPENDENCY_LICENSE_CONCLUSIONS:
            raise RuntimeError(f"lock package has no license conclusion: {name}")
        source = package.get("source", {})
        if not isinstance(source, dict):
            raise RuntimeError(f"lock package source is invalid: {name}")
        markers = sorted({dependency.get("marker") for dependency in package.get("dependencies", []) if isinstance(dependency, dict) and isinstance(dependency.get("marker"), str)})
        identity = {"name": name, "version": package["version"], "source": source, "markers": markers}
        row = {**identity, "row_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "license_conclusion": DEPENDENCY_LICENSE_CONCLUSIONS[name]}
        rows.append(row)
    rows.sort(key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["markers"]))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "vokra-dia-uv-lock-license-audit-v1", "status": DEPENDENCY_LICENSE_AUDIT_STATUS, "package_count": len(rows), "rows_sha256": digest, "rows": rows}


def reference_project_identity() -> dict[str, Any]:
    """Bind the frozen project and the versions actually imported at runtime."""
    project = Path(__file__).parent / "dia_1_6b_reference"
    lock = project / "uv.lock"
    pyproject = project / "pyproject.toml"
    if not lock.is_file() or not pyproject.is_file():
        raise RuntimeError("dedicated Dia reference project files are missing")
    lock_sha = sha256(lock)
    pyproject_sha = sha256(pyproject)
    if lock_sha != REFERENCE_PROJECT_LOCK_SHA256 or pyproject_sha != REFERENCE_PROJECT_PYPROJECT_SHA256:
        raise RuntimeError("dedicated Dia reference project identity changed")
    from importlib import metadata
    actual = {}
    for package, expected in DIRECT_DEPENDENCY_VERSIONS.items():
        try:
            actual[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required locked distribution is not installed: {package}") from error
        if actual[package] != expected:
            raise RuntimeError(f"locked distribution version mismatch: {package}={actual[package]!r}")
    return {
        "project": project.name,
        "python": "3.12",
        "pyproject_sha256": pyproject_sha,
        "uv_lock_sha256": lock_sha,
        "lock_schema": "uv-lock-v1-python312",
        "expected_versions": dict(DIRECT_DEPENDENCY_VERSIONS),
        "actual_versions": actual,
        "use_torch_compile": False,
        "dependency_audit": lock_dependency_audit(lock),
        "dependency_license_audit": DEPENDENCY_LICENSE_AUDIT_STATUS,
    }


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def git_revision(path: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def authenticate_source(path: Path) -> dict[str, Any]:
    import subprocess
    if not path.is_dir() or git_revision(path) != SOURCE_REVISION:
        raise RuntimeError("official Dia source revision is not authenticated")
    origin = subprocess.check_output(["git", "-C", str(path), "remote", "get-url", "origin"], text=True).strip()
    if origin.removesuffix(".git").rstrip("/") != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("official Dia source origin is not authenticated")
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], text=True):
        raise RuntimeError("official Dia source checkout is dirty")
    required = tuple(SOURCE_ROLE_BLOBS)
    files = {}
    for relative in required:
        file = path / relative
        if not file.is_file():
            raise RuntimeError(f"missing official source role: {relative}")
        blob = git_blob_sha1(file)
        if blob != SOURCE_ROLE_BLOBS[relative]:
            raise RuntimeError(f"official source role changed: {relative}")
        files[relative] = {"bytes": file.stat().st_size, "sha256": sha256(file), "git_blob_sha1": blob}
    actual_roles = {relative: files[relative]["git_blob_sha1"] for relative in files}
    if actual_roles != SOURCE_ROLE_BLOBS:
        raise RuntimeError("official source role set is not the fixed authenticated set")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "resolved_revision": SOURCE_REVISION, "clean": True, "files": files}


def authenticate_model(path: Path) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    inspector = importlib.import_module("dia_1_6b_inspect")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and not p.is_symlink()}
    if actual != set(inspector.EXPECTED_FILES):
        raise RuntimeError(f"HF snapshot has extra/missing files: {sorted(actual ^ set(inspector.EXPECTED_FILES))}")
    files = {}
    for relative, (size, blob, lfs) in inspector.EXPECTED_FILES.items():
        file = path / relative
        if not file.is_file() or file.stat().st_size != size:
            raise RuntimeError(f"HF snapshot identity mismatch: {relative}")
        actual = sha256(file) if lfs else git_blob_sha1(file)
        if actual != (lfs or blob):
            raise RuntimeError(f"HF snapshot digest mismatch: {relative}")
        files[relative] = {"bytes": size, "git_blob_sha1": blob, "lfs_sha256": lfs, "sha256": sha256(file)}
    return {"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "files": files}


def authenticate_public(path: Path) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    inspector = importlib.import_module("dia_1_6b_inspect")
    expected = inspector.EXPECTED_GGUF
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and not p.is_symlink()}
    if actual != {expected["path"]}:
        raise RuntimeError(f"public Dia tree has extra/missing files: {sorted(actual ^ {expected['path']})}")
    file = path / expected["path"]
    if not file.is_file() or file.stat().st_size != expected["bytes"]:
        raise RuntimeError("public Dia GGUF size/path mismatch")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    inspector = importlib.import_module("dia_1_6b_inspect")
    pointer = inspector.lfs_pointer_sha1(expected["lfs_sha256"], expected["bytes"])
    if sha256(file) != expected["lfs_sha256"] or pointer != expected["git_blob_sha1"]:
        raise RuntimeError("public Dia GGUF digest mismatch")
    return {"repository": PUBLIC_REPOSITORY, "revision": PUBLIC_REVISION, "resolved_revision": PUBLIC_REVISION, "file": {**expected, "sha256": sha256(file)}}


def finite_tensor(value: Any, role: str) -> dict[str, Any]:
    import torch
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise RuntimeError(f"{role}: expected a non-empty tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{role}: non-finite tensor")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": True}


def save_tensor(value: Any, role: str, output: Path, records: dict[str, Any]) -> None:
    import numpy as np
    import torch
    meta = finite_tensor(value, role)
    array = value.detach().to("cpu").contiguous().numpy()
    if role.endswith("_ids") or role in {"delayed_codes", "reverted_codes"}:
        if not getattr(array.dtype, "kind", "") in {"i", "u"}:
            raise RuntimeError(f"{role}: ids/codes must use an integer dtype")
        limit = 256 if role == "text_ids" else 1028
        if int(array.min()) < 0 or int(array.max()) >= limit:
            raise RuntimeError(f"{role}: id outside the Dia target vocabulary")
    previous = records.get(role, [])
    index = len(previous)
    file = output / f"{role}-{index:04d}.npy"
    np.save(file, array, allow_pickle=False)
    previous.append({**meta, "path": file.name, "bytes": file.stat().st_size, "sha256": sha256(file)})
    records[role] = previous


def authenticate_dac(evidence: Path, checkpoint: Path, source_path: Path) -> dict[str, Any]:
    """Accept only a separately produced, source-authenticated DAC packet."""
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate DAC evidence key: {key}")
            result[key] = value
        return result
    packet = json.loads(evidence.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    required = {"status", "package", "source", "checkpoint", "mapping"}
    if set(packet) != required or packet["status"] != "DAC_PROOF_REQUIRED":
        raise RuntimeError("DAC evidence is missing exact checkpoint/tree and Vokra manifest proof")
    package = packet["package"]
    # The full wheel is intentionally not in this reference closure: its
    # optional training/UI dependency graph pulls in librosa/soxr.  VAST must
    # instead provide a package shell made from the authenticated upstream DAC
    # checkout, containing only the modules used by ``DAC.load``.  This keeps
    # the executed implementation upstream while making the adapted closure
    # auditable and independent of an installed distribution's metadata.
    if set(package) != {"name", "version", "source_root", "files"} or package.get("name") != "descript-audio-codec-source-shell" or package.get("version") != "1.0.0" or package.get("source_root") != "dac":
        raise RuntimeError("DAC package shell identity is not the pinned upstream 1.0.0 source shell")
    if not isinstance(package.get("files"), dict) or not package["files"]:
        raise RuntimeError("authenticated DAC source-shell file set is missing")
    for relative, record in package["files"].items():
        if not isinstance(relative, str) or not relative.startswith("dac/") or Path(relative).is_absolute() or ".." in Path(relative).parts or not isinstance(record, dict) or set(record) != {"bytes", "sha256"} or not isinstance(record["bytes"], int) or record["bytes"] <= 0 or len(record["sha256"]) != 64:
            raise RuntimeError("DAC package tree entry is unsafe or incomplete")
    installed_files = {}
    for relative, record in package["files"].items():
        file = source_path / relative
        if file.is_symlink() or not file.is_file():
            raise RuntimeError(f"DAC source-shell file is missing or non-regular: {relative}")
        actual = {"bytes": file.stat().st_size, "sha256": sha256(file)}
        if actual != record:
            raise RuntimeError(f"DAC source-shell file differs from authenticated evidence: {relative}")
        installed_files[relative] = actual
    source = packet["source"]
    if source.get("repository") != "https://github.com/descriptinc/descript-audio-codec" or not isinstance(source.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", source["revision"]) or source.get("resolved_revision") != source.get("revision"):
        raise RuntimeError("DAC source origin/revision is not authenticated")
    if not isinstance(source.get("files"), dict) or not source["files"]:
        raise RuntimeError("DAC source role hashes are missing")
    import subprocess
    if not source_path.is_dir() or git_revision(source_path) != source["revision"]:
        raise RuntimeError("DAC source checkout revision differs from evidence")
    origin = subprocess.check_output(["git", "-C", str(source_path), "remote", "get-url", "origin"], text=True).strip()
    if origin.removesuffix(".git").rstrip("/") != source["repository"].removesuffix(".git").rstrip("/"):
        raise RuntimeError("DAC source checkout origin differs from evidence")
    if subprocess.check_output(["git", "-C", str(source_path), "status", "--porcelain"], text=True):
        raise RuntimeError("DAC source checkout is dirty")
    actual_roles = {}
    for relative, record in source["files"].items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or not isinstance(record, dict) or set(record) != {"bytes", "git_blob_sha1", "sha256"} or not isinstance(record["bytes"], int) or len(record["git_blob_sha1"]) != 40 or len(record["sha256"]) != 64:
            raise RuntimeError("DAC source role hash is incomplete")
        file = source_path / relative
        if not file.is_file() or git_blob_sha1(file) != record["git_blob_sha1"] or sha256(file) != record["sha256"]:
            raise RuntimeError(f"DAC source role changed: {relative}")
        actual_roles[relative] = record["git_blob_sha1"]
    if set(actual_roles) != set(source["files"]):
        raise RuntimeError("DAC source role set is not stable")
    expected = packet["checkpoint"]
    if not isinstance(expected, dict) or not isinstance(expected.get("bytes"), int) or expected["bytes"] <= 0 or not isinstance(expected.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) or not isinstance(expected.get("roles"), dict) or not expected["roles"]:
        raise RuntimeError("DAC checkpoint evidence lacks complete body identity")
    for role, record in expected["roles"].items():
        if not isinstance(role, str) or not isinstance(record, dict) or set(record) != {"name", "shape", "dtype", "sha256"} or not isinstance(record["name"], str) or not isinstance(record["shape"], list) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise RuntimeError("DAC checkpoint tensor role mapping is incomplete")
    if checkpoint.stat().st_size != expected["bytes"] or sha256(checkpoint) != expected["sha256"]:
        raise RuntimeError("DAC checkpoint identity mismatch")
    mapping = packet["mapping"]
    # A prose status is not evidence of equivalence.  Until an exact DAC
    # checkpoint URL/bytes/tree and the crate::dac::Dac Khz44 manifest are
    # pinned here, refuse the reference run before loading Dia.
    if mapping.get("status") != "PROVEN_EXACT" or mapping.get("sample_rate") != 44100 or mapping.get("n_codebooks") != 9 or mapping.get("hop_length") != 512 or not isinstance(mapping.get("vokra_dac_manifest_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", mapping["vokra_dac_manifest_sha256"]):
        raise RuntimeError("DAC-to-Vokra Khz44 mapping proof is unavailable; execution is blocked")
    raise RuntimeError("DAC exact checkpoint/tree and Vokra Khz44 manifest are not pinned in this adapter; execution is blocked")
    return packet


def run(source: Path, model: Path, public: Path, dac_evidence: Path, dac_checkpoint: Path, dac_source: Path, output: Path, text: str, seed: int) -> None:
    if not source.is_dir() or not model.is_dir() or not public.is_dir() or not output.is_dir():
        raise RuntimeError("source, model, public, and output directories are required")
    project_evidence = reference_project_identity()
    if project_evidence["dependency_license_audit"] != "AUDITED_ALLOW":
        raise RuntimeError("Dia reference dependency license/provenance audit is blocked")
    source_evidence = authenticate_source(source)
    model_evidence = authenticate_model(model)
    public_evidence = authenticate_public(public)
    dac_evidence_packet = authenticate_dac(dac_evidence, dac_checkpoint, dac_source)
    if seed < 0 or seed >= 2**32:
        raise RuntimeError("seed must be an unsigned 32-bit value")
    import sys
    # The DAC shell is an authenticated local source tree, never a package
    # resolver/download.  Put it on the import path before importing ``dac``.
    sys.path.insert(0, str(dac_source))
    sys.path.insert(0, str(source))
    torch = importlib.import_module("torch")
    dia_model = importlib.import_module("dia.model")
    torch.manual_seed(seed)
    capture: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    original_sample = dia_model._sample_next_token
    original_multinomial = torch.multinomial

    def sample(logits, temperature, top_p, top_k, audio_eos_value):
        before = len(artifacts.get("decoder_sampling_probability", []))
        save_tensor(logits, "decoder_logits", output, artifacts)
        capture["official_sampler_active"] = True
        try:
            selected = original_sample(logits, temperature, top_p, top_k, audio_eos_value)
        finally:
            capture["official_sampler_active"] = False
        if len(artifacts.get("decoder_sampling_probability", [])) != before + 1:
            raise RuntimeError("official Dia sampler did not make exactly one multinomial call")
        capture.setdefault("selected", []).append(selected.detach().cpu())
        return selected

    def multinomial(probability, num_samples, *args, **kwargs):
        if not capture.get("official_sampler_active"):
            return original_multinomial(probability, num_samples, *args, **kwargs)
        save_tensor(probability, "decoder_sampling_probability", output, artifacts)
        capture.setdefault("multinomial_calls", []).append(len(capture.get("multinomial_calls", [])))
        return original_multinomial(probability, num_samples, *args, **kwargs)

    dia_model._sample_next_token = sample
    torch.multinomial = multinomial
    try:
        # Loading is deliberately local-only: the VAST worker authenticated the
        # exact files before invoking this adapter.
        config = dia_model.DiaConfig.load(str(model / "config.json"))
        if config is None:
            raise RuntimeError("official Dia config could not be loaded")
        dac = importlib.import_module("dac")
        if not hasattr(dac, "DAC") or not hasattr(dac.DAC, "load"):
            raise RuntimeError("pinned DAC package has no official load entry point")
        # Do not call upstream ``dac.utils.download``: the worker has already
        # authenticated this exact checkpoint and all network resolution is
        # forbidden during reference execution.
        engine = dia_model.Dia.from_local(str(model / "config.json"), str(model / "dia-v0_1.pth"), load_dac=False)
        engine.dac_model = dac.DAC.load(str(dac_checkpoint)).to(engine.device)
        engine.dac_model.eval()
        engine.load_dac = True
        original_encode = engine._encode_text
        original_prompt = engine._prepare_audio_prompt
        original_output = engine._generate_output

        def encode(self, value):
            result = original_encode(value)
            expected = list(value.encode("utf-8").replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02"))[: self.config.data.text_length]
            if result.detach().cpu().tolist() != expected:
                raise RuntimeError("official Dia byte encoding differs from the fixed [S1]/[S2] contract")
            # Official Dia encodes the complete multi-speaker string once.
            # Splitting this tensor into per-speaker roles fabricates evidence.
            save_tensor(result, "text_ids", output, artifacts)
            return result

        def prompt(self, prompts):
            delayed, steps = original_prompt(prompts)
            return delayed, steps

        def generated_output(self, codes, lengths):
            valid_frames = int(lengths.max().item())
            max_delay = max(self.config.data.delay_pattern)
            save_tensor(codes[:, : valid_frames + max_delay, :], "delayed_codes", output, artifacts)
            from dia.audio import build_revert_indices, revert_audio_delay
            reverted = revert_audio_delay(
                codes,
                self.config.data.audio_pad_value,
                build_revert_indices(codes.shape[0], codes.shape[1], codes.shape[2], self.config.data.delay_pattern),
                codes.shape[1],
            )[:, :-max(self.config.data.delay_pattern), :]
            save_tensor(reverted[:, :valid_frames, :], "reverted_codes", output, artifacts)
            return original_output(codes, lengths)

        engine._encode_text = types.MethodType(encode, engine)
        engine._prepare_audio_prompt = types.MethodType(prompt, engine)
        engine._generate_output = types.MethodType(generated_output, engine)
        if engine.dac_model is None:
            raise RuntimeError("official DAC is not loaded")
        original_from_codes = engine.dac_model.quantizer.from_codes

        def from_codes(self, codes, *args, **kwargs):
            result = original_from_codes(codes, *args, **kwargs)
            save_tensor(result[0], "dac_latent", output, artifacts)
            return result

        engine.dac_model.quantizer.from_codes = types.MethodType(from_codes, engine.dac_model.quantizer)
        original_prepare = engine._prepare_generation

        def prepare(self, padded_text, audio_prompts, max_tokens=None):
            state, output_state = original_prepare(padded_text, audio_prompts, max_tokens)
            save_tensor(state.cross_attn_mask, "text_padding_mask", output, artifacts)
            encoder = state.enc_out
            if encoder.shape[0] != 2:
                raise RuntimeError("batch-one encoder output must contain unconditional and conditional rows")
            save_tensor(encoder[0:1], "unconditional_encoder", output, artifacts)
            save_tensor(encoder[1:2], "conditional_encoder", output, artifacts)
            return state, output_state

        engine._prepare_generation = types.MethodType(prepare, engine)
        if text.count("[S1]") != 1 or text.count("[S2]") != 1 or text.index("[S1]") >= text.index("[S2]"):
            raise RuntimeError("the evidence input must contain exactly one ordered [S1] then [S2] marker")
        generated = engine.generate(text, max_tokens=min(config.data.audio_length, 32), verbose=False)
        if not isinstance(generated, (list, tuple)):
            generated = [generated]
        if not generated or generated[0] is None:
            raise RuntimeError("official Dia produced no PCM")
        save_tensor(torch.as_tensor(generated[0]), "pcm", output, artifacts)
        if capture.get("selected"):
            save_tensor(torch.stack(capture["selected"]), "selected_ids", output, artifacts)
        if len(artifacts.get("decoder_logits", [])) != len(artifacts.get("decoder_sampling_probability", [])):
            raise RuntimeError("decoder logits/probability call cardinality mismatch")
        if len(artifacts.get("selected_ids", [])) != 1 or artifacts["selected_ids"][0]["shape"][0] != len(artifacts["decoder_logits"]):
            raise RuntimeError("selected IDs are not aligned one-for-one with official sampler calls")
        if artifacts["decoder_logits"][0]["shape"] != [9, 1028] or artifacts["decoder_sampling_probability"][0]["shape"] != [9, 1028]:
            raise RuntimeError("Dia decoder seam must be [channels=9,vocab=1028]")
        if artifacts["delayed_codes"][0]["shape"][-1] != 9 or artifacts["reverted_codes"][0]["shape"][-1] != 9:
            raise RuntimeError("Dia delayed/reverted codes must be frame-major [frames,9]")
        if artifacts["dac_latent"][0]["shape"][1] != 1024:
            raise RuntimeError("DAC latent must have 1024 channels")
        if len(artifacts["pcm"]) != 1 or artifacts["pcm"][0]["shape"][-1] <= 0:
            raise RuntimeError("PCM must be a non-empty one-dimensional waveform")
        if set(artifacts) != REQUIRED_ARTIFACTS:
            raise RuntimeError(f"same-execution artifact set incomplete: {sorted(set(artifacts) ^ REQUIRED_ARTIFACTS)}")
        manifest = {
            "format": FORMAT,
            "status": "REFERENCE_COMPLETE",
            "reference_project": project_evidence,
            "source": source_evidence,
            "hf": model_evidence,
            "public": public_evidence,
            "dac": dac_evidence_packet,
            "seed": seed,
            "sampling": {"implementation": "official Dia _sample_next_token", "multinomial_calls": len(capture.get("multinomial_calls", [])), "logits_calls": len(artifacts["decoder_logits"]), "probability_calls": len(artifacts["decoder_sampling_probability"]), "selected_rows": artifacts["selected_ids"][0]["shape"][0], "global_torch_multinomial_scope": "official_sampler_only", "selection_evidence": "exact official selected IDs", "rng_equivalence": "NOT_CLAIMED"},
            "schema": {"text_ids": {"role": "text_ids", "source_vocab": 256, "encoding": "one complete official _encode_text result; [S1]=1,[S2]=2"}, "encoder": {"roles": ["unconditional_encoder", "conditional_encoder"], "rows": 1, "hidden": 1024}, "decoder": {"logits_role": "decoder_logits", "probability_role": "decoder_sampling_probability", "selected_role": "selected_ids", "channels": 9, "vocab": 1028, "call_order": "logits -> official multinomial probability -> selected IDs"}, "audio_codes": {"delayed_role": "delayed_codes", "reverted_role": "reverted_codes", "axis_order": "[batch,frames,channels]", "channels": 9}, "dac": {"latent_role": "dac_latent", "pcm_role": "pcm", "sample_rate": 44100, "hop_length": 512}},
            "artifacts": artifacts,
            "native_status": "BLOCKED_UNTIL_VAST_AND_APPLE_EVIDENCE",
            "publication": "NO_UPLOAD",
            "comparison_status": COMPARISON_STATUS,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    finally:
        dia_model._sample_next_token = original_sample
        torch.multinomial = original_multinomial


def self_test() -> None:
    assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40 and len(PUBLIC_REVISION) == 40
    assert len(REFERENCE_PROJECT_LOCK_SHA256) == 64 and len(REFERENCE_PROJECT_PYPROJECT_SHA256) == 64
    assert DEPENDENCY_LICENSE_AUDIT_STATUS == "BLOCKED_UNREVIEWED_TRANSITIVE"
    assert DIRECT_DEPENDENCY_VERSIONS["torch"] == "2.6.0+cpu"
    assert set(DIRECT_DEPENDENCY_VERSIONS) == {
        "einops", "gguf", "huggingface-hub", "numpy", "pydantic",
        "soundfile", "torch", "torchaudio",
    }
    assert not set(DIRECT_DEPENDENCY_VERSIONS) & {
        "descript-audio-codec", "gradio", "librosa", "soxr", "triton",
    }
    assert DEFAULT_TEXT.count("[S1]") == DEFAULT_TEXT.count("[S2]") == 1
    assert DEFAULT_TEXT.index("[S1]") < DEFAULT_TEXT.index("[S2]")
    assert list(DEFAULT_TEXT.encode().replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02"))[0] == 1
    assert set(SOURCE_ROLE_BLOBS) == {"LICENSE", "dia/audio.py", "dia/config.py", "dia/layers.py", "dia/model.py", "dia/state.py", "pyproject.toml"}
    assert COMPARISON_STATUS == "NOT_RUN_OFFICIAL_ONLY"
    assert REQUIRED_ARTIFACTS == {
        "text_ids", "text_padding_mask", "conditional_encoder",
        "unconditional_encoder", "decoder_logits", "decoder_sampling_probability",
        "selected_ids", "delayed_codes", "reverted_codes", "dac_latent", "pcm",
    }
    assert all(math.isfinite(float(x)) for x in (0.0, 1.0))
    print("dia reference self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dac-evidence", type=Path)
    parser.add_argument("--dac-checkpoint", type=Path)
    parser.add_argument("--dac-source", type=Path)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if None in (args.source, args.model, args.public, args.dac_evidence, args.dac_checkpoint, args.dac_source, args.output):
        parser.error("--source, --model, --public, --dac-evidence, --dac-checkpoint, --dac-source, and --output are required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be absent or empty; stale evidence is rejected")
    try:
        if args.text != DEFAULT_TEXT:
            parser.error(f"--text must be the fixed two-speaker evidence input: {DEFAULT_TEXT!r}")
        run(args.source, args.model, args.public, args.dac_evidence, args.dac_checkpoint, args.dac_source, args.output, args.text, args.seed)
    except Exception as error:
        (args.output / "INSPECTION_ERROR").write_text(str(error) + "\n", encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
