#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Official-upstream Chatterbox T3 trace dumper.

This is a reference harness, not a mirror of T3.  On a VAST worker it imports
the exact pinned checkout, constructs the upstream ``T3`` class, and calls its
real ``inference`` method.  The packet owns the single stochastic draw; no
fixture numbers are generated here and no native Rust implementation is
called.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import types
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/resemble-ai/chatterbox.git"
SOURCE_REVISION = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
SOURCE_PROJECT_VERSION = "0.1.7"
BASE_REPOSITORY = "ResembleAI/chatterbox"
BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
BASE_CHECKPOINT = "t3_mtl23ls_v3.safetensors"
NANO_REVISION = "71ccd1d0081b430592cea481f4307e764e07bc64"
TURBO_REVISION = "749d1c1a46eb10492095d68fbcf55691ccf137cd"
PACKET_KEYS = {
    "variant", "text", "language", "speaker_embedding", "prompt_speech_tokens",
    "emotion_adv", "cfg_weight", "max_new_tokens", "draw",
}
EXPECTED_VARIANTS = {"base", "nano", "turbo"}
SELECTED_FILES = {
    "base": ("grapheme_mtl_merged_expanded_v1.json", BASE_CHECKPOINT),
    "nano": ("t3_nano_v1.safetensors",),
    "turbo": ("t3_turbo_v1.safetensors",),
}
# Git blob identities from the fixed source checkout.  The source marker
# checks below are only a readability guard; this map is the provenance gate.
SOURCE_ROLE_BLOBS = {
    "src/chatterbox/models/t3/t3.py": "d83de261e249648f6654e2bac7cb10390af983c9",
    "src/chatterbox/models/t3/modules/cond_enc.py": "b5f15c685783fbb048f6c0e86fc2ea8fbf1ec3de",
    "src/chatterbox/models/tokenizers/tokenizer.py": "84d45d35d2db9c6c576a4af98a7ab91a704af9f2",
    "src/chatterbox/mtl_tts.py": "ec5ebff418f6abf283127c4de6bbe99580f29e69",
    "pyproject.toml": "381ed774eae577cb244d699f47b64953980ce72f",
    "src/chatterbox/tts_turbo.py": "e708f0c88abd6c615b64da725f344a1312098433",
}

REFERENCE_PROJECT = Path(__file__).with_name("chatterbox_t3")
REFERENCE_LOCK = REFERENCE_PROJECT / "uv.lock"
REFERENCE_LOCK_SHA256 = "83879e5e0a3d16c550df9a13134c9f3cbe44e5869afe54674c28be72b5cdec37"
REFERENCE_PACKAGE_ROWS_SHA256 = "f5cfab32caf3cc2340b434c1e9e0d3f8dbbab73a519925fbb6f08457c03e7e98"
LOCK_CORE_VERSIONS = {
    "numpy": "1.26.4",
    "huggingface-hub": "1.27.0",
    "einops": "0.8.2",
    "safetensors": "0.5.3",
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "tqdm": "4.67.1",
    "transformers": "5.2.0",
}
FORBIDDEN_REFERENCE_PACKAGES = {"diffusers", "resemble-perth", "s3tokenizer", "gradio"}
LICENSE_AUDIT_STATUS = "BLOCKED_UNRESOLVED"
LICENSE_AUDIT_REVIEWED_PACKAGES = {
    "annotated-doc", "anyio", "certifi", "click", "colorama", "einops",
    "filelock", "fsspec", "h11", "hf-xet", "httpcore", "httpx",
    "huggingface-hub", "idna", "jinja2", "markdown-it-py", "markupsafe",
    "mdurl", "mpmath", "networkx", "numpy", "packaging", "pygments", "pyyaml",
    "regex", "rich", "safetensors", "setuptools", "shellingham", "sympy",
    "tokenizers", "torch", "torchaudio", "tqdm", "transformers", "typer", "typer-slim",
    "typing-extensions",
}
LICENSE_AUDIT_BLOCKERS = (
    "typing-extensions==4.16.0 upstream pyproject declares PSF-2.0, outside the repository Apache/MIT/BSD allowlist; owner policy clearance is required",
)
LICENSE_AUDIT_ACCEPTED_REFERENCE_ONLY = (
    "certifi==2026.7.22: upstream bundle carries MPL-2.0 file-level notices; unmodified VAST-only environment, no Vokra redistribution",
    "tqdm==4.67.1: upstream LICENCE identifies MIT for most files and MPL-2.0 for the listed files; unmodified VAST-only environment, no Vokra redistribution",
    "numpy==1.26.4: upstream BSD-3-Clause notices identify bundled OpenBLAS/LAPACK plus GCC libgfortran GPL-3.0-with-GCC-exception and libquadmath LGPL-2.1; these remain external reference wheels and are not shipped by Vokra",
)
LICENSE_AUDIT_PRIMARY_EVIDENCE = (
    "https://pypi.org/pypi/certifi/2026.7.22/json",
    "https://github.com/certifi/python-certifi/blob/master/LICENSE",
    "https://pypi.org/pypi/tqdm/4.67.1/json",
    "https://github.com/tqdm/tqdm/blob/v4.67.1/LICENCE",
    "https://pypi.org/pypi/numpy/1.26.4/json",
    "https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt",
    "https://pypi.org/pypi/typing-extensions/4.16.0/json",
        "https://github.com/python/typing_extensions/blob/4.16.0/LICENSE",
)
LICENSE_CONCLUSION_BY_NAME = {
    "annotated-doc": ("MIT", "https://pypi.org/pypi/annotated-doc/0.0.5/json"),
    "anyio": ("MIT", "https://pypi.org/pypi/anyio/4.14.2/json"),
    "certifi": ("MPL-2.0; reference-only accepted", "https://pypi.org/pypi/certifi/2026.7.22/json"),
    "click": ("BSD-3-Clause", "https://pypi.org/pypi/click/8.5.0/json"),
    "colorama": ("BSD-3-Clause", "https://pypi.org/pypi/colorama/0.4.6/json"),
    "einops": ("MIT", "https://pypi.org/pypi/einops/0.8.2/json"),
    "filelock": ("MIT", "https://pypi.org/pypi/filelock/3.32.4/json"),
    "fsspec": ("BSD-3-Clause", "https://pypi.org/pypi/fsspec/2026.7.0/json"),
    "h11": ("MIT", "https://pypi.org/pypi/h11/0.16.0/json"),
    "hf-xet": ("Apache-2.0", "https://pypi.org/pypi/hf-xet/1.6.0/json"),
    "httpcore": ("BSD-3-Clause", "https://pypi.org/pypi/httpcore/1.0.9/json"),
    "httpx": ("BSD-3-Clause", "https://pypi.org/pypi/httpx/0.28.1/json"),
    "huggingface-hub": ("Apache-2.0", "https://pypi.org/pypi/huggingface-hub/1.27.0/json"),
    "idna": ("BSD-3-Clause", "https://pypi.org/pypi/idna/3.19/json"),
    "jinja2": ("BSD-3-Clause", "https://pypi.org/pypi/jinja2/3.1.6/json"),
    "markdown-it-py": ("MIT", "https://pypi.org/pypi/markdown-it-py/4.2.0/json"),
    "markupsafe": ("BSD-3-Clause", "https://pypi.org/pypi/markupsafe/3.0.3/json"),
    "mdurl": ("MIT", "https://pypi.org/pypi/mdurl/0.1.2/json"),
    "mpmath": ("BSD", "https://pypi.org/pypi/mpmath/1.3.0/json"),
    "networkx": ("BSD-3-Clause", "https://pypi.org/pypi/networkx/3.6.1/json"),
    "numpy": ("BSD-3-Clause + bundled runtime notices; reference-only accepted", "https://pypi.org/pypi/numpy/1.26.4/json"),
    "packaging": ("Apache-2.0", "https://pypi.org/pypi/packaging/26.3/json"),
    "pygments": ("BSD-2-Clause", "https://pypi.org/pypi/pygments/2.21.0/json"),
    "pyyaml": ("MIT", "https://pypi.org/pypi/pyyaml/6.0.3/json"),
    "regex": ("Apache-2.0", "https://pypi.org/pypi/regex/2026.7.19/json"),
    "rich": ("MIT", "https://pypi.org/pypi/rich/15.0.0/json"),
    "safetensors": ("Apache-2.0", "https://pypi.org/pypi/safetensors/0.5.3/json"),
    "setuptools": ("MIT", "https://pypi.org/pypi/setuptools/84.0.0/json"),
    "shellingham": ("ISC", "https://pypi.org/pypi/shellingham/1.5.4/json"),
    "sympy": ("BSD-3-Clause", "https://pypi.org/pypi/sympy/1.13.1/json"),
    "tokenizers": ("Apache-2.0", "https://pypi.org/pypi/tokenizers/0.22.2/json"),
    "torch": ("BSD-3-Clause; official CPU index", "https://download.pytorch.org/whl/cpu"),
    "torchaudio": ("BSD-3-Clause; official CPU index", "https://download.pytorch.org/whl/cpu"),
    "tqdm": ("MPL-2.0 AND MIT; reference-only accepted", "https://pypi.org/pypi/tqdm/4.67.1/json"),
    "transformers": ("Apache-2.0", "https://pypi.org/pypi/transformers/5.2.0/json"),
    "typer": ("MIT", "https://pypi.org/pypi/typer/0.27.2/json"),
    "typer-slim": ("MIT", "https://pypi.org/pypi/typer-slim/0.24.0/json"),
    "typing-extensions": ("PSF-2.0; unresolved under repository allowlist", "https://github.com/python/typing_extensions/blob/4.16.0/LICENSE"),
    "vokra-chatterbox-t3-reference": ("PROJECT_METADATA_ONLY", "tools/parity/chatterbox_t3/pyproject.toml"),
}


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate packet key: {key}")
        result[key] = value
    return result


def load_packet(path: Path) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(packet, dict) or set(packet) != PACKET_KEYS:
        raise ValueError(f"packet keys must be exactly {sorted(PACKET_KEYS)}")
    if packet["variant"] not in EXPECTED_VARIANTS:
        raise ValueError("packet variant must be base, nano, or turbo")
    if not isinstance(packet["text"], str) or not packet["text"]:
        raise ValueError("packet text must be non-empty")
    if len(packet["text"]) > 16_384 or "\x00" in packet["text"]:
        raise ValueError("packet text exceeds the bounded upstream request contract")
    if packet["variant"] == "base" and packet["language"] not in {None, "en"}:
        # Language-specific transforms require the official data files; the
        # worker accepts them only when the official tokenizer receives them.
        if not isinstance(packet["language"], str) or not packet["language"]:
            raise ValueError("base packet language must be a non-empty string or null")
    speaker = packet["speaker_embedding"]
    if not isinstance(speaker, list) or len(speaker) != 256 or any(
        not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x))
        for x in speaker
    ):
        raise ValueError("speaker_embedding must contain 256 finite values")
    prompt = packet["prompt_speech_tokens"]
    if not isinstance(prompt, list) or any(
        not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in prompt
    ):
        raise ValueError("prompt_speech_tokens must contain non-negative integer ids")
    expected_prompt = 150 if packet["variant"] == "base" else 375
    if len(prompt) != expected_prompt:
        raise ValueError(f"prompt_speech_tokens must contain exactly {expected_prompt} ids")
    speech_vocab = 8194 if packet["variant"] == "base" else 6563
    if any(token >= speech_vocab for token in prompt):
        raise ValueError("prompt_speech_tokens contains an id outside the authenticated speech vocabulary")
    for key in ("emotion_adv", "cfg_weight", "draw"):
        if not isinstance(packet[key], (int, float)) or isinstance(packet[key], bool) or not math.isfinite(float(packet[key])):
            raise ValueError(f"packet {key} must be finite")
    if packet["draw"] < 0.0 or packet["draw"] >= 1.0:
        raise ValueError("packet draw must be in [0,1)")
    if packet["variant"] != "base":
        if packet["language"] is not None or packet["cfg_weight"] != 0.0 or packet["emotion_adv"] != 0.0:
            raise ValueError("Nano/Turbo packet must use canonical ignored conditioning fields")
    # Turbo's upstream implementation samples its initial full-prefix token
    # before entering range(max_gen_len); zero loop iterations therefore
    # still produce exactly one multinomial call/token.
    expected_steps = 1 if packet["variant"] == "base" else 0
    if not isinstance(packet["max_new_tokens"], int) or isinstance(packet["max_new_tokens"], bool) or packet["max_new_tokens"] != expected_steps:
        raise ValueError(f"this trace packet requires max_new_tokens={expected_steps} for one upstream generation step")
    return packet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_lock_rows(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise RuntimeError("dedicated lock contains a malformed package row")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"registry"}, {"virtual"}) or not isinstance(next(iter(source.values())), str):
            raise RuntimeError("dedicated lock package source is malformed")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers):
            raise RuntimeError("dedicated lock package markers are malformed")
        rows.append({
            "name": package["name"],
            "version": package["version"],
            "source": {key: source[key] for key in sorted(source)},
            "markers": sorted(markers),
        })
    return sorted(
        rows,
        key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["markers"]),
    )


def lock_rows_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_license_conclusions(lock_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    conclusions: dict[str, dict[str, str]] = {}
    for row in lock_rows:
        name = row["name"]
        conclusion = LICENSE_CONCLUSION_BY_NAME.get(name)
        if conclusion is None:
            raise RuntimeError(f"license conclusion missing for locked package: {name}")
        license_name, evidence = conclusion
        key = f"{name}=={row['version']}"
        if key in conclusions:
            raise RuntimeError(f"duplicate versioned license conclusion: {key}")
        conclusions[key] = {
            "license": license_name,
            "evidence": evidence,
            "source": next(iter(row["source"].values())),
        }
    return {key: conclusions[key] for key in sorted(conclusions)}


def git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def reference_lock_identity(lock_path: Path = REFERENCE_LOCK) -> dict[str, Any]:
    """Authenticate the reviewed, source-compatible dedicated environment."""
    if not lock_path.is_file():
        raise RuntimeError("dedicated Chatterbox T3 uv.lock is absent")
    lock_sha256 = sha256_file(lock_path)
    if lock_sha256 != REFERENCE_LOCK_SHA256:
        raise RuntimeError("dedicated Chatterbox T3 uv.lock SHA-256 is not the reviewed identity")
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("requires-python") != "==3.12.*":
        raise RuntimeError("dedicated lock is not restricted to Python 3.12")
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("dedicated lock package table is malformed")
    rows = [package for package in packages if isinstance(package, dict)]
    lock_rows = canonical_lock_rows(rows)
    if lock_rows_sha256(lock_rows) != REFERENCE_PACKAGE_ROWS_SHA256:
        raise RuntimeError("dedicated Chatterbox T3 lock package rows are not the reviewed identity")
    versions = {
        name: {package.get("version") for package in rows if package.get("name") == name}
        for name in {package.get("name") for package in rows if isinstance(package.get("name"), str)}
    }
    if any(name in versions for name in FORBIDDEN_REFERENCE_PACKAGES | {"triton"} | {name for name in versions if name.startswith("nvidia-")}):
        raise RuntimeError("dedicated lock includes excluded full-pipeline or mutable package")
    for name, version in LOCK_CORE_VERSIONS.items():
        expected_versions = {version}
        if name in {"torch", "torchaudio"}:
            expected_versions = {version, f"{version}+cpu"}
        if versions.get(name) != expected_versions:
            raise RuntimeError(f"dedicated lock core package identity drift: {name}")
    for name in ("torch", "torchaudio"):
        for package in rows:
            if package.get("name") == name and package.get("source", {}).get("registry") != "https://download.pytorch.org/whl/cpu":
                raise RuntimeError(f"{name} is not exclusively routed to the official CPU index")
    cpu_versions = {
        name: f"{LOCK_CORE_VERSIONS[name]}+cpu" for name in ("torch", "torchaudio")
    }
    if not all(cpu_versions[name] in versions.get(name, set()) for name in cpu_versions):
        raise RuntimeError("dedicated lock core package identity drift")
    return {
        "path": str(lock_path),
        "sha256": lock_sha256,
        "python": lock["requires-python"],
        "core_versions": {name: LOCK_CORE_VERSIONS[name] for name in sorted(LOCK_CORE_VERSIONS)},
        "cpu_index": "https://download.pytorch.org/whl/cpu",
        "cpu_distribution_versions": cpu_versions,
        "package_rows_sha256": REFERENCE_PACKAGE_ROWS_SHA256,
        "package_rows": lock_rows,
        "excluded_packages": sorted(FORBIDDEN_REFERENCE_PACKAGES),
        "package_names": sorted(versions),
    }


def license_audit_identity(lock_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind lock membership to the reviewed primary-metadata audit.

    The audit is intentionally unresolved: the CPU-only lock removes the
    CUDA/NVIDIA/Triton closure, but several remaining Linux-wheel/runtime
    licenses still need review.
    Keeping this gate here means neither ``uv sync`` in the dedicated project
    nor a model/reference run can accidentally present an unaudited closure as
    authenticated evidence.
    """
    if lock_record is None:
        lock_record = reference_lock_identity()
    with (REFERENCE_PROJECT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    metadata = (
        project.get("tool", {})
        .get("vokra", {})
        .get("chatterbox_t3_reference", {})
        .get("license_audit", {})
    )
    if not isinstance(metadata, dict):
        raise RuntimeError("dedicated license audit metadata is missing")
    reviewed = metadata.get("reviewed_packages")
    if set(reviewed or ()) != LICENSE_AUDIT_REVIEWED_PACKAGES:
        raise RuntimeError("dedicated license audit package inventory drifted")
    blockers = tuple(metadata.get("blockers", ()))
    if blockers != LICENSE_AUDIT_BLOCKERS:
        raise RuntimeError("dedicated license audit blocker evidence drifted")
    accepted = tuple(metadata.get("accepted_reference_only", ()))
    if accepted != LICENSE_AUDIT_ACCEPTED_REFERENCE_ONLY:
        raise RuntimeError("dedicated accepted reference-only license evidence drifted")
    evidence = tuple(metadata.get("primary_evidence", ()))
    if evidence != LICENSE_AUDIT_PRIMARY_EVIDENCE:
        raise RuntimeError("dedicated license audit primary evidence drifted")
    if metadata.get("status") != LICENSE_AUDIT_STATUS:
        raise RuntimeError("dedicated license audit status is not fail-closed")
    if not isinstance(metadata.get("primary_metadata"), str) or not metadata["primary_metadata"]:
        raise RuntimeError("dedicated license audit primary source is missing")
    if metadata.get("triton_status") != "NOT_IN_LOCK; CPU index closure excludes Triton":
        raise RuntimeError("triton license evidence drifted")
    if metadata.get("cuda_status") != "NOT_IN_LOCK; CPU index closure excludes NVIDIA CUDA distributions":
        raise RuntimeError("CUDA closure evidence drifted")
    if metadata.get("soxr_status") != "NOT_IN_LOCK":
        raise RuntimeError("soxr lock exclusion evidence drifted")
    lock_packages = set(lock_record.get("package_names", ()))
    lock_packages.discard("vokra-chatterbox-t3-reference")
    if lock_packages != LICENSE_AUDIT_REVIEWED_PACKAGES:
        raise RuntimeError("lock package closure is outside the reviewed license inventory")
    package_rows = lock_record.get("package_rows")
    if not isinstance(package_rows, list) or lock_rows_sha256(package_rows) != REFERENCE_PACKAGE_ROWS_SHA256:
        raise RuntimeError("versioned lock package evidence is missing or drifted")
    if metadata.get("license_conclusion_map") != "version-keyed name==version mapping emitted by chatterbox_t3_reference.py from the authenticated package_rows":
        raise RuntimeError("dedicated versioned license mapping declaration drifted")
    if metadata.get("license_conclusion_count") != len(package_rows):
        raise RuntimeError("dedicated versioned license mapping cardinality drifted")
    license_conclusions = package_license_conclusions(package_rows)
    if set(key.split("==", 1)[0] for key in license_conclusions) != set(lock_record["package_names"]):
        raise RuntimeError("versioned license conclusion inventory is incomplete")
    return {
        "status": metadata.get("status"),
        "primary_metadata": metadata.get("primary_metadata"),
        "reviewed_packages": sorted(LICENSE_AUDIT_REVIEWED_PACKAGES),
        "blockers": list(LICENSE_AUDIT_BLOCKERS),
        "accepted_reference_only": list(LICENSE_AUDIT_ACCEPTED_REFERENCE_ONLY),
        "primary_evidence": list(LICENSE_AUDIT_PRIMARY_EVIDENCE),
        "license_conclusions": license_conclusions,
        "triton": metadata.get("triton_status"),
        "cuda": metadata.get("cuda_status"),
        "soxr": metadata.get("soxr_status"),
        "lock_sha256": lock_record["sha256"],
    }


def require_license_clearance(lock_record: dict[str, Any]) -> dict[str, Any]:
    audit = license_audit_identity(lock_record)
    if audit["status"] != "AUTHENTICATED_CLEAR":
        raise RuntimeError(
            "Chatterbox T3 reference dependency license audit is unresolved; "
            "uv sync/model acquisition/reference execution are blocked"
        )
    return audit


def validate_inspection(snapshot: Path, inspection: Path, variant: str) -> dict[str, Any]:
    """Require the inspector's authenticated server tree and local bind."""
    manifest = json.loads(inspection.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    if (
        manifest.get("status") != "BLOCKED"
        or manifest.get("evidence_stage") != "INSPECTION_ONLY"
        or manifest.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED"
        or manifest.get("publication") != "NO_UPLOAD"
        or manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE"
    ):
        raise RuntimeError("Chatterbox inspection evidence is not complete")
    model = manifest.get("model", {})
    expected_repo, expected_revision = {
        "base": (BASE_REPOSITORY, BASE_REVISION),
        "nano": ("ResembleAI/chatterbox-nano", NANO_REVISION),
        "turbo": ("ResembleAI/chatterbox-turbo", TURBO_REVISION),
    }[variant]
    if (model.get("repository"), model.get("revision")) != (expected_repo, expected_revision):
        raise RuntimeError("inspection model identity does not match selected variant")
    server = manifest.get("server_tree", {})
    files = server.get("files") if isinstance(server, dict) else None
    if not isinstance(files, list):
        raise RuntimeError("inspection server tree is missing")
    rows = {row.get("path"): row for row in files if isinstance(row, dict)}
    if len(rows) != len(files) or any(
        not isinstance(row.get("path"), str)
        or not isinstance(row.get("size"), int)
        or row.get("size") < 0
        or (row.get("lfs_sha256") is None and not isinstance(row.get("git_blob_sha1"), str))
        or (row.get("lfs_sha256") is not None and (not isinstance(row.get("lfs_sha256"), str) or len(row["lfs_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["lfs_sha256"])))
        or (isinstance(row.get("git_blob_sha1"), str) and (len(row["git_blob_sha1"]) != 40 or any(c not in "0123456789abcdef" for c in row["git_blob_sha1"])))
        for row in files
        if isinstance(row, dict)
    ):
        raise RuntimeError("inspection server tree contains duplicate or malformed file rows")
    selected = manifest.get("selected_materialization", {}).get("selected", [])
    if not isinstance(selected, list) or any(
        not isinstance(row, dict)
        or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}
        or row.get("type") != "file"
        or not isinstance(row.get("path"), str)
        or not row["path"]
        for row in selected
    ):
        raise RuntimeError("inspection selected materialization is malformed")
    selected_paths = {row.get("path") for row in selected if isinstance(row, dict)}
    if len(selected_paths) != len(selected):
        raise RuntimeError("inspection selected materialization contains duplicate paths")
    if not set(SELECTED_FILES[variant]) <= selected_paths:
        raise RuntimeError("inspection selected materialization omits trace inputs")
    for name in selected_paths:
        row = rows.get(name)
        path = snapshot / name
        if not isinstance(row, dict) or not path.is_file() or path.is_symlink() or path.stat().st_size != row.get("size"):
            raise RuntimeError(f"selected checkpoint asset is not authenticated: {name}")
        actual = sha256_file(path) if row.get("lfs_sha256") else git_blob_sha1(path)
        expected = row.get("lfs_sha256") or row.get("git_blob_sha1")
        if actual != expected:
            raise RuntimeError(f"selected checkpoint asset digest mismatch: {name}")
    if server.get("repository") != expected_repo or server.get("revision") != expected_revision or server.get("resolved_revision") != expected_revision:
        raise RuntimeError("inspection server tree identity does not match selected variant")
    # Reuse the inspector's fixed complete-tree table.  Accepting only the
    # selected local files above is insufficient: a server packet with an
    # altered or omitted unmaterialized role must also be rejected.
    try:
        from chatterbox_family_inspect import validate_packet
        validate_packet({
            "repository": server["repository"],
            "revision": server["revision"],
            "resolved_revision": server["resolved_revision"],
            "files": files,
        }, variant)
    except Exception as error:
        raise RuntimeError(f"inspection server tree is not the fixed authenticated tree: {error}") from error
    source = manifest.get("source", {})
    if source.get("revision") != SOURCE_REVISION:
        raise RuntimeError("inspection source revision mismatch")
    return manifest


def source_identity(source: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_REVISION:
        raise RuntimeError(f"official source HEAD mismatch: {head}")
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip().removesuffix(".git")
    if origin != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError(f"official source origin mismatch: {origin}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True):
        raise RuntimeError("official source checkout is dirty")
    roles = {
        "src/chatterbox/models/t3/t3.py": ("class T3", "def inference", "torch.multinomial", "past_key_values"),
        "src/chatterbox/models/t3/modules/cond_enc.py": ("class T3CondEnc", "cond_spkr", "cond_prompt_speech_emb", "cond_emotion_adv"),
        "src/chatterbox/models/tokenizers/tokenizer.py": ("class MTLTokenizer", "normalize(\"NFKD\"", "[SPACE]", "language_id"),
        "src/chatterbox/mtl_tts.py": ("def punc_norm", "grapheme_mtl_merged_expanded_v1.json", "SUPPORTED_LANGUAGES"),
        "pyproject.toml": ("version = \"0.1.7\"", "torch==2.6.0", "transformers==5.2.0", "safetensors==0.5.3"),
        "src/chatterbox/tts_turbo.py": ("def punc_norm", "inference_turbo", "AutoTokenizer"),
    }
    records = {}
    for role, markers in roles.items():
        path = source / role
        if not path.is_file():
            raise RuntimeError(f"official source role missing: {role}")
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise RuntimeError(f"source marker missing: {role}: {marker}")
        if role.endswith(".py"):
            ast.parse(text, filename=role)
        expected_blob = SOURCE_ROLE_BLOBS.get(role)
        if expected_blob is not None and git_blob_sha1(path) != expected_blob:
            raise RuntimeError(f"official source role Git identity mismatch: {role}")
        records[role] = {
            "sha256": sha256_file(path),
            "git_blob_sha1": git_blob_sha1(path),
            "expected_git_blob_sha1": expected_blob,
            "markers": list(markers),
        }
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "project_version": SOURCE_PROJECT_VERSION,
        "optional_dependency_policy": "mutable Perth dependency is not installed; wrapper imports are bypassed",
        "roles": records,
    }


def load_source_function(source: Path, relative: str, name: str) -> Any:
    """Load one pure function from fixed source without importing its wrapper.

    The wrapper modules import optional audio/runtime dependencies (including
    mutable Perth). AST extraction executes only the pinned upstream function,
    preserving its code path while keeping this T3 packet dependency-isolated.
    """
    path = source / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"fixed source function missing or ambiguous: {relative}:{name}")
    namespace: dict[str, Any] = {}
    module = ast.Module(body=[nodes[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def load_pinned_t3_modules(source: Path) -> tuple[Any, Any, Any, Any]:
    """Load T3 modules without running ``chatterbox/__init__.py``.

    The package wrapper imports optional VC/TTS dependencies and mutable Perth.
    Synthetic package entries are used only for Python's relative-import
    resolution; every executed module is loaded from the fixed checkout.
    """
    root = source / "src/chatterbox"

    def package(name: str, path: Path) -> None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module

    def module(name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot create fixed-source import spec: {path}")
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value

    package("chatterbox", root)
    package("chatterbox.models", root / "models")
    package("chatterbox.models.t3", root / "models/t3")
    package("chatterbox.models.t3.modules", root / "models/t3/modules")
    package("chatterbox.models.t3.inference", root / "models/t3/inference")
    package("chatterbox.models.tokenizers", root / "models/tokenizers")
    module("chatterbox.models.utils", root / "models/utils.py")
    module("chatterbox.models.t3.llama_configs", root / "models/t3/llama_configs.py")
    module("chatterbox.models.t3.modules.t3_config", root / "models/t3/modules/t3_config.py")
    module("chatterbox.models.t3.modules.learned_pos_emb", root / "models/t3/modules/learned_pos_emb.py")
    module("chatterbox.models.t3.modules.perceiver", root / "models/t3/modules/perceiver.py")
    cond = module("chatterbox.models.t3.modules.cond_enc", root / "models/t3/modules/cond_enc.py")
    module("chatterbox.models.t3.inference.t3_hf_backend", root / "models/t3/inference/t3_hf_backend.py")
    t3 = module("chatterbox.models.t3.t3", root / "models/t3/t3.py")
    tokenizer = module("chatterbox.models.tokenizers.tokenizer", root / "models/tokenizers/tokenizer.py")
    return t3.T3, cond.T3Cond, sys.modules["chatterbox.models.t3.modules.t3_config"].T3Config, tokenizer.MTLTokenizer


def import_smoke(source: Path) -> dict[str, str]:
    """Prove the pinned T3 modules load without running the root wrapper."""
    classes = load_pinned_t3_modules(source)
    names = ("T3", "T3Cond", "T3Config", "MTLTokenizer")
    modules = {name: value.__module__ for name, value in zip(names, classes)}
    if any(module == "chatterbox" for module in modules.values()):
        raise RuntimeError("pinned import unexpectedly executed chatterbox root wrapper")
    return modules


def tensor_record(output: Path, name: str, value: Any) -> dict[str, Any]:
    import torch
    if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor) or value.numel() <= 0:
        raise RuntimeError(f"{name} is not a tensor: {type(value).__name__}")
    if Path(name).name != name or ".." in Path(name).parts or "\x00" in name:
        raise RuntimeError(f"{name} is not a safe relative artifact name")
    value = value.detach().to("cpu").contiguous()
    shape = [int(x) for x in value.shape]
    if len(shape) > 4 or any(x <= 0 for x in shape) or value.numel() > 16_777_216:
        raise RuntimeError(f"{name} has an unbounded or empty shape")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains non-finite values")
    widths = {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1,
        torch.uint8: 1, torch.bool: 1,
    }
    width = widths.get(value.dtype)
    if width is None:
        raise RuntimeError(f"{name} has unsupported dtype {value.dtype}")
    raw = value.numpy().tobytes() if value.dtype != torch.bfloat16 else value.view(torch.uint16).numpy().tobytes()
    if len(raw) != value.numel() * width:
        raise RuntimeError(f"{name} byte count does not match shape")
    artifact = output / f"{name}.bin"
    artifact.write_bytes(raw)
    return {"name": name, "shape": shape, "dtype": str(value.dtype), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def deterministic_inverse_cdf(probabilities: Any, draw: float) -> int:
    """Select from one official probability row using a caller-owned draw.

    This is deliberately an injection at the upstream ``torch.multinomial``
    seam, not a replacement claim for PyTorch's native RNG.  The target is
    scaled by the actual row mass because the intercepted row is not assumed
    to be normalized.
    """
    import torch

    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 1:
        raise RuntimeError("inverse-CDF input must be one probability row")
    if not math.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise RuntimeError("inverse-CDF draw must be in [0,1)")
    probabilities = probabilities.detach().to("cpu").float()
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise RuntimeError("official multinomial received invalid probabilities")
    mass = float(probabilities.sum().item())
    if not math.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("official multinomial received zero probability mass")
    target = draw * mass
    cumulative = 0.0
    for index, probability in enumerate(probabilities.tolist()):
        cumulative += float(probability)
        if target < cumulative:
            return index
    return int(probabilities.numel() - 1)


def run_official(source: Path, snapshot: Path, inspection: Path, packet: dict[str, Any], output: Path) -> dict[str, Any]:
    source_record = source_identity(source)
    inspection_record = validate_inspection(snapshot, inspection, packet["variant"])
    lock_record = reference_lock_identity()
    license_record = require_license_clearance(lock_record)
    sys.path.insert(0, str(source / "src"))
    import torch
    try:
        T3, T3Cond, T3Config, MTLTokenizer = load_pinned_t3_modules(source)
        from transformers import AutoTokenizer
    except Exception as error:
        raise RuntimeError(f"cannot import pinned upstream T3/tokenizer classes: {error}") from error
    variant = packet["variant"]
    if variant == "base":
        hp = T3Config.multilingual()
        # mtl_tts.py selects this exact grapheme tokenizer for the multilingual
        # route; do not substitute another tokenizer sidecar.
        tokenizer = MTLTokenizer(str(snapshot / "grapheme_mtl_merged_expanded_v1.json"))
        multilingual_punc_norm = load_source_function(source, "src/chatterbox/mtl_tts.py", "punc_norm")
    else:
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_small" if variant == "nano" else "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = T3(hp).eval()
    checkpoint = snapshot / (BASE_CHECKPOINT if variant == "base" else f"t3_{variant}_v1.safetensors")
    if not checkpoint.is_file():
        raise RuntimeError(f"pinned checkpoint component missing: {checkpoint.name}")
    from safetensors.torch import load_file
    state = load_file(str(checkpoint), device="cpu")
    # This is the exact upstream `from_local` unwrap for the historical
    # safetensors container; do not reinterpret or rename tensor keys here.
    if "model" in state:
        state = state["model"][0]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"upstream T3 state dict mismatch: missing={missing!r}, unexpected={unexpected!r}")
    if tokenizer is None:
        raise RuntimeError("GPT-2 tokenizer sidecar is missing from the pinned snapshot")
    # The official Nano/Turbo wrapper deletes GPT-2's text embedding after
    # loading because T3 owns the text embedding.  Reproduce that exact
    # post-load boundary and bind the resulting key set to the authenticated
    # checkpoint instead of silently retaining a different model.
    removed_prefix = None
    if variant != "base":
        if not hasattr(model.tfmr, "wte"):
            raise RuntimeError("official GPT-2 wrapper expected tfmr.wte before deletion")
        del model.tfmr.wte
        removed_prefix = "tfmr.wte."
    loaded_state = model.state_dict()
    expected_keys = set(state)
    if removed_prefix is not None:
        expected_keys = {key for key in expected_keys if not key.startswith(removed_prefix)}
    if set(loaded_state) != expected_keys:
        raise RuntimeError("loaded T3 manifest does not match checkpoint after wrapper deletion")
    loaded_rows = [
        {"name": name, "shape": [int(dim) for dim in tensor.shape], "dtype": str(tensor.dtype)}
        for name, tensor in sorted(loaded_state.items())
    ]
    loaded_canonical = json.dumps(loaded_rows, separators=(",", ":"), sort_keys=True).encode()
    loaded_manifest = {
        "parameter_count": len(loaded_rows),
        "parameters_sha256": hashlib.sha256(loaded_canonical).hexdigest(),
        "removed_wrapper_parameter_prefix": removed_prefix,
    }
    text = packet["text"]
    token_calls = 0
    if variant == "base":
        text = multilingual_punc_norm(text)
        original_text_to_tokens = tokenizer.text_to_tokens
        def capture_text_to_tokens(*args: Any, **kwargs: Any) -> Any:
            nonlocal token_calls
            token_calls += 1
            return original_text_to_tokens(*args, **kwargs)
        tokenizer.text_to_tokens = capture_text_to_tokens
        text_ids = tokenizer.text_to_tokens(text, language_id=packet["language"])
        text_ids = torch.cat([text_ids, text_ids], dim=0)
        text_ids = torch.nn.functional.pad(text_ids, (1, 0), value=hp.start_text_token)
        text_ids = torch.nn.functional.pad(text_ids, (0, 1), value=hp.stop_text_token)
    else:
        gpt2_punc_norm = load_source_function(source, "src/chatterbox/tts_turbo.py", "punc_norm")
        text = gpt2_punc_norm(text)
        original_tokenizer = tokenizer
        def capture_tokenizer(*args: Any, **kwargs: Any) -> Any:
            nonlocal token_calls
            token_calls += 1
            return original_tokenizer(*args, **kwargs)
        encoded = capture_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_ids = encoded.input_ids
    taps: list[dict[str, Any]] = []
    taps.append(tensor_record(output, "text_tokens", text_ids))
    speaker = torch.tensor(packet["speaker_embedding"], dtype=torch.float32).view(1, 256)
    prompt = torch.tensor(packet["prompt_speech_tokens"], dtype=torch.long).view(1, -1)
    emotion = torch.tensor(packet["emotion_adv"], dtype=torch.float32).view(1, 1, 1)
    cond = T3Cond(
        speaker_emb=speaker,
        cond_prompt_speech_tokens=prompt,
        emotion_adv=emotion if variant == "base" else None,
    )
    # Capture the exact upstream conditioning output, not a reimplementation.
    original_prepare = model.prepare_conditioning
    def prepare_capture(t3_cond: Any) -> Any:
        value = original_prepare(t3_cond)
        taps.append(tensor_record(output, "conditioning", value))
        return value
    model.prepare_conditioning = prepare_capture
    original_multinomial = torch.multinomial
    calls = 0
    draw = float(packet["draw"])
    def guarded_multinomial(probs: Any, num_samples: int, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls != 1 or num_samples != 1:
            raise RuntimeError(f"unexpected upstream multinomial call: {calls}, {num_samples}")
        if probs.ndim != 2 or probs.shape[0] != 1:
            raise RuntimeError(f"unexpected upstream probability shape: {tuple(probs.shape)}")
        taps.append(tensor_record(output, f"multinomial_probs_{calls:04d}", probs))
        probs_cpu = probs.detach().to("cpu").float()
        selected = deterministic_inverse_cdf(probs_cpu[0], draw)
        return torch.tensor([[selected]], dtype=torch.long, device=probs.device)
    torch.multinomial = guarded_multinomial
    try:
        if variant == "base":
            result = model.inference(
                t3_cond=cond,
                text_tokens=text_ids,
                max_new_tokens=packet["max_new_tokens"],
                stop_on_eos=False,
                do_sample=True,
                cfg_weight=float(packet["cfg_weight"]),
            )
        else:
            result = model.inference_turbo(
                t3_cond=cond,
                text_tokens=text_ids,
                # The fixed wrapper samples the initial full-prefix token
                # before `range(max_gen_len)`: zero is the one-step trace.
                max_gen_len=0,
            )
    finally:
        torch.multinomial = original_multinomial
    if calls != 1 or token_calls != 1:
        raise RuntimeError(f"upstream T3 did not consume exactly one caller-owned draw: {calls}")
    generated = result.detach().to("cpu").contiguous()
    taps.append(tensor_record(output, "generated_tokens", generated))
    expected_probability_width = 8194 if variant == "base" else 6563
    expected_names = {"text_tokens", "conditioning", "multinomial_probs_0001", "generated_tokens"}
    if {tap["name"] for tap in taps} != expected_names or len(taps) != len(expected_names):
        raise RuntimeError("T3 trace tap set is incomplete or contains stale/extra taps")
    probability = next(tap for tap in taps if tap["name"] == "multinomial_probs_0001")
    if probability["shape"] != [1, expected_probability_width]:
        raise RuntimeError(f"unexpected probability shape: {probability['shape']}")
    if generated.shape != (1, 1):
        raise RuntimeError(f"unexpected generated token shape: {tuple(generated.shape)}")
    if any(path.name not in {f"{name}.bin" for name in expected_names} for path in output.iterdir()):
        raise RuntimeError("T3 trace output contains stale or unrecognized artifacts")
    return {
        "format": "vokra-chatterbox-t3-reference-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "reference_status": "REFERENCE_EVIDENCE_COMPLETE",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "publication": "NO_UPLOAD",
        "source": source_record,
        "reference_environment": {**lock_record, "license_audit": license_record},
        "inspection": {"status": inspection_record["inspection_status"], "model": inspection_record["model"]},
        "model_repository": BASE_REPOSITORY if variant == "base" else f"ResembleAI/chatterbox-{variant}",
        "model_revision": {"base": BASE_REVISION, "nano": NANO_REVISION, "turbo": TURBO_REVISION}[variant],
        "checkpoint": BASE_CHECKPOINT if variant == "base" else checkpoint.name,
        "loaded_t3_manifest": loaded_manifest,
        "generation_route": "inference(max_new_tokens=1)" if variant == "base" else "inference_turbo(max_gen_len=0)",
        "variant": variant,
        "tokenizer_calls": token_calls,
        "multinomial_calls": calls,
        "multinomial_probability_capture": "exact probability tensor passed to official torch.multinomial is saved in taps",
        "caller_owned_draw": {
            "value": draw,
            "contract": "one deterministic inverse-CDF draw consumed at the official torch.multinomial call",
            "native_torch_rng": False,
        },
        "taps": taps,
        "pcm_status": "BLOCKED_S3GEN_HIFT_WATERMARK_NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--license-audit", action="store_true")
    parser.add_argument("--import-smoke", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        assert SOURCE_REVISION == "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
        assert BASE_CHECKPOINT == "t3_mtl23ls_v3.safetensors"
        source_text = Path(__file__).read_text(encoding="utf-8")
        assert "grapheme_mtl_merged_expanded_v1.json" in source_text
        assert "multinomial_probs_" in source_text
        assert "validate_inspection" in source_text
        assert "native_torch_rng" in source_text
        assert "def import_smoke" in source_text
        lock = reference_lock_identity()
        assert lock["python"] == "==3.12.*"
        assert lock["core_versions"]["torch"] == "2.6.0"
        assert lock["core_versions"]["transformers"] == "5.2.0"
        assert lock["cpu_index"] == "https://download.pytorch.org/whl/cpu"
        assert lock["cpu_distribution_versions"] == {"torch": "2.6.0+cpu", "torchaudio": "2.6.0+cpu"}
        assert lock["package_rows_sha256"] == REFERENCE_PACKAGE_ROWS_SHA256
        assert lock_rows_sha256(lock["package_rows"]) == REFERENCE_PACKAGE_ROWS_SHA256
        assert len(lock["package_rows"]) == 41
        assert set(lock["excluded_packages"]) == FORBIDDEN_REFERENCE_PACKAGES
        assert "triton" not in lock["package_names"]
        assert not any(name.startswith("nvidia-") for name in lock["package_names"])
        audit = license_audit_identity(lock)
        assert audit["status"] == LICENSE_AUDIT_STATUS
        assert audit["triton"].startswith("NOT_IN_LOCK")
        assert audit["cuda"].startswith("NOT_IN_LOCK")
        assert audit["soxr"] == "NOT_IN_LOCK"
        assert len(audit["license_conclusions"]) == 41
        assert audit["license_conclusions"]["typing-extensions==4.16.0"]["license"].startswith("PSF-2.0")
        # Keep this self-test dependency-free on the maintainer machine.  The
        # actual tensor seam is exercised only by the VAST reference project;
        # absence of its heavy torch wheel is not a reason to resolve it here.
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None:
            assert deterministic_inverse_cdf(torch.tensor([2.0, 1.0]), 0.5) == 0
            assert deterministic_inverse_cdf(torch.tensor([2.0, 1.0]), 0.8) == 1
        import tempfile
        with tempfile.TemporaryDirectory(prefix="vokra-chatterbox-t3-packet-") as directory:
            packet_path = Path(directory) / "packet.json"
            artifact_root = Path(directory) / "artifacts"
            artifact_root.mkdir()
            tampered_lock = Path(directory) / "uv.lock"
            tampered_lock.write_bytes(REFERENCE_LOCK.read_bytes() + b"\n# tampered\n")
            try:
                reference_lock_identity(tampered_lock)
            except RuntimeError:
                pass
            else:
                raise AssertionError("accepted a tampered dedicated lock")
            tampered_rows = [dict(row) for row in lock["package_rows"]]
            tampered_rows[0] = {**tampered_rows[0], "source": {"registry": "https://example.invalid/simple"}}
            if lock_rows_sha256(tampered_rows) == REFERENCE_PACKAGE_ROWS_SHA256:
                raise AssertionError("accepted a tampered lock source row")
            if torch is not None:
                try:
                    tensor_record(artifact_root, "../escape", torch.ones(1))
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("unsafe synthetic tap path accepted")
                try:
                    tensor_record(artifact_root, "nonfinite", torch.tensor([float("inf")]))
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("non-finite synthetic tap accepted")
            packet = {
                "variant": "nano",
                "text": "hello",
                "language": None,
                "speaker_embedding": [0.0] * 256,
                "prompt_speech_tokens": [0] * 375,
                "emotion_adv": 0.0,
                "cfg_weight": 0.0,
                "max_new_tokens": 0,
                "draw": 0.5,
            }
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            assert load_packet(packet_path)["variant"] == "nano"
            packet["cfg_weight"] = 0.5
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            try:
                load_packet(packet_path)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted non-canonical Nano CFG field")
        assert "from chatterbox." + "mtl_tts" not in source_text
        assert "from chatterbox." + "tts_turbo" not in source_text
        assert PACKET_KEYS == {
            "variant", "text", "language", "speaker_embedding", "prompt_speech_tokens",
            "emotion_adv", "cfg_weight", "max_new_tokens", "draw",
        }
        print("chatterbox_t3_reference.py self-test: OK")
        return 0
    if args.license_audit:
        lock = reference_lock_identity()
        print(json.dumps({"reference_environment": lock, "license_audit": license_audit_identity(lock)}, sort_keys=True))
        return 2 if LICENSE_AUDIT_STATUS != "AUTHENTICATED_CLEAR" else 0
    if args.import_smoke:
        if args.source is None:
            parser.error("--import-smoke requires --source")
        print(json.dumps(import_smoke(args.source), sort_keys=True))
        return 0
    if not all((args.source, args.snapshot, args.inspection, args.packet, args.output)):
        parser.error("--source, --snapshot, --inspection, --packet and --output are required")
    # Authenticate the dependency/license gate before touching the caller's
    # output path.  The VAST worker performs the same check before workdir
    # creation; keeping it here prevents an unsafe direct invocation from
    # leaving partial evidence behind.
    try:
        require_license_clearance(reference_lock_identity())
    except RuntimeError as error:
        print(f"Chatterbox T3 reference BLOCKED: {error}", file=sys.stderr)
        return 2
    packet = load_packet(args.packet)
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            parser.error("--output must be absent or empty")
    else:
        args.output.mkdir(parents=True)
    manifest = run_official(args.source, args.snapshot, args.inspection, packet, args.output)
    (args.output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
