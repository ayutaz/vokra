#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Model-free source/dependency audit for the Fun-CosyVoice3 route.

The official composite is intentionally kept fail-closed here.  The pinned
CosyVoice requirements declare ``librosa==0.10.2`` and the official Matcha
mel implementation imports ``librosa.filters``.  The authenticated PyPI
metadata for librosa 0.10.2 declares ``soxr>=0.3.2``.  soxr is forbidden by
Vokra's license policy, so creating a frozen environment or claiming an
official complete reference would be incorrect.

This script does not import model code, synchronize a Python environment,
read a checkpoint, or execute an audio/model operation.  It only authenticates
the pinned source checkout shape and records the dependency decision.  The
LLM-only and flow/HiFT source roles are reported separately; neither is a
complete CosyVoice3 composite or an Apple runtime verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "vokra-cosyvoice3-source-dependency-audit-v1"
MODEL_REPOSITORY = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice"
SOURCE_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "0d990d60740bf174904a5185cce910b847bd3684"
MATCHA_REPOSITORY = "https://github.com/shivammehta25/Matcha-TTS"
MATCHA_URL = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_REVISION = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
OFFICIAL_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/"
    f"{SOURCE_REVISION}/requirements.txt"
)
LIBROSA_METADATA_URL = "https://pypi.org/pypi/librosa/0.10.2/json"
LIBROSA_VERSION = "0.10.2"
LIBROSA_SOXR_REQUIREMENT = "soxr>=0.3.2"
FORBIDDEN_PACKAGE = "soxr"
REFERENCE_PROJECT = Path(__file__).with_name("cosyvoice3_reference")

# These are dependency declarations in the pinned official requirements file,
# not an invented replacement environment.  The complete project keeps these
# values in pyproject.toml as an inventory while its uv.lock is intentionally
# absent until the forbidden closure is resolved.
OFFICIAL_DECLARATIONS = {
    "librosa": "librosa==0.10.2",
    "torch": "torch==2.3.1",
    "torchaudio": "torchaudio==2.3.1",
    "transformers": "transformers==4.51.3",
}

SOURCE_CONTRACTS = {
    "requirements.txt": ("librosa==0.10.2",),
    "cosyvoice/cli/cosyvoice.py": (
        "from cosyvoice.cli.frontend import CosyVoiceFrontEnd",
        "from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice3Model",
    ),
    "cosyvoice/flow/flow.py": (
        "class CausalMaskedDiffWithDiT",
        "from cosyvoice.utils.onnx import SpeechTokenExtractor",
    ),
    "cosyvoice/flow/flow_matching.py": (
        "class CausalConditionalCFM",
        "from matcha.models.components.flow_matching import BASECFM",
    ),
    "cosyvoice/hifigan/generator.py": ("class CausalHiFTGenerator",),
    "cosyvoice/cli/model.py": ("class CosyVoice3Model", "token2wav"),
    "examples/libritts/cosyvoice3/conf/cosyvoice3.yaml": (
        "sample_rate: 24000",
        "token_mel_ratio: 2",
    ),
}
MATCHA_CONTRACTS = {
    "matcha/utils/audio.py": (
        "from librosa.filters import mel as librosa_mel_fn",
        "def mel_spectrogram(",
    ),
}


class AuditError(ValueError):
    """A source, dependency, or output-safety contract failed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().casefold())


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(f"git query failed for {root}: {error}") from error


def require_clean_pinned_checkout(root: Path, repository: str, revision: str, url: str) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise AuditError(f"source checkout is missing or symlinked: {root}")
    if git(root, "rev-parse", "HEAD") != revision:
        raise AuditError(f"source revision mismatch for {root}")
    origin = git(root, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    expected = url.removesuffix("/").removesuffix(".git")
    if origin != expected:
        raise AuditError(f"source origin mismatch for {root}: {origin!r}")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise AuditError(f"source checkout is dirty: {root}")
    return {"repository": repository, "url": url, "revision": revision, "origin": origin, "clean": True}


def require_source_files(root: Path, contracts: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative, needles in contracts.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise AuditError(f"required source role is missing/symlinked: {relative}")
        text = path.read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in text]
        if missing:
            raise AuditError(f"source role contract mismatch: {relative}: {missing!r}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_sha1": git(root, "rev-parse", f"HEAD:{relative}"),
            "needles": list(needles),
        }
    return files


def parse_exact_dependencies(project: Path) -> dict[str, str]:
    pyproject = project / "pyproject.toml"
    if not project.is_dir() or project.is_symlink() or not pyproject.is_file() or pyproject.is_symlink():
        raise AuditError("CosyVoice3 reference project/pyproject.toml is missing or symlinked")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = data.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise AuditError("CosyVoice3 direct dependency inventory is missing")
    parsed: dict[str, str] = {}
    for raw in dependencies:
        if not isinstance(raw, str) or raw.count("==") != 1:
            raise AuditError(f"dependency is not an exact pin: {raw!r}")
        name, version = raw.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name.strip().casefold())
        if not normalized or not version:
            raise AuditError(f"malformed dependency pin: {raw!r}")
        if normalized in parsed:
            raise AuditError(f"duplicate dependency pin: {normalized}")
        parsed[normalized] = version
    if parsed.get("librosa") != LIBROSA_VERSION:
        raise AuditError("reference project must retain the official librosa==0.10.2 declaration")
    return parsed


def audit_project(project: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    dependencies = parse_exact_dependencies(project)
    lock = project / "uv.lock"
    if lock.exists() or lock.is_symlink():
        raise AuditError("uv.lock must remain absent while the authenticated closure contains forbidden soxr")
    config = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    route = config.get("tool", {}).get("vokra", {}).get("cosyvoice3_reference", {})
    if route.get("lock_status") != "BLOCKED_FORBIDDEN_SOXR_IN_AUTHENTICATED_OFFICIAL_CLOSURE":
        raise AuditError("reference project does not preserve the forbidden-soxr lock status")
    if route.get("dedicated_sync") != "FORBIDDEN_UNTIL_LICENSE_SIGNOFF_AND_LOCK":
        raise AuditError("reference project sync gate is not fail-closed")
    metadata = fetch_librosa_metadata() if metadata is None else metadata
    return {
        "project": str(project),
        "pyproject_sha256": sha256_file(project / "pyproject.toml"),
        "dependencies": dependencies,
        "uv_lock": {"status": "ABSENT_BY_DESIGN", "reason": "forbidden soxr closure"},
        "librosa": {
            "version": LIBROSA_VERSION,
            "pypi_metadata_url": LIBROSA_METADATA_URL,
            "declared_requirement": LIBROSA_SOXR_REQUIREMENT,
            "forbidden_package": FORBIDDEN_PACKAGE,
            "metadata_sha256": metadata["sha256"],
            "requires_dist": metadata["requires_dist"],
            "metadata_status": "PRIMARY_SOURCE_DECLARATION_AUTHENTICATED",
        },
    }


def fetch_librosa_metadata() -> dict[str, Any]:
    """Fetch and authenticate the immutable PyPI metadata without importing librosa."""
    request = urllib.request.Request(LIBROSA_METADATA_URL, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise AuditError(f"PyPI metadata HTTP status: {response.status}")
            raw = response.read(4 * 1024 * 1024 + 1)
    except (OSError, urllib.error.URLError) as error:
        raise AuditError(f"PyPI librosa metadata unavailable: {error}") from error
    if len(raw) > 4 * 1024 * 1024:
        raise AuditError("PyPI librosa metadata exceeds bounded response size")
    return parse_librosa_metadata(raw)


def parse_librosa_metadata(raw: bytes) -> dict[str, Any]:
    """Validate immutable PyPI JSON, including duplicate-key rejection."""
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_json_pairs)
    except AuditError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuditError("PyPI librosa metadata is not valid JSON") from error
    info = payload.get("info") if isinstance(payload, dict) else None
    requires_dist = info.get("requires_dist") if isinstance(info, dict) else None
    if (
        not isinstance(info, dict)
        or not isinstance(info.get("name"), str)
        or normalized_distribution_name(info["name"]) != "librosa"
        or info.get("version") != LIBROSA_VERSION
        or not isinstance(requires_dist, list)
    ):
        raise AuditError("PyPI librosa metadata identity is malformed")
    normalized = [item for item in requires_dist if isinstance(item, str)]
    if LIBROSA_SOXR_REQUIREMENT not in normalized:
        raise AuditError("PyPI librosa 0.10.2 metadata no longer declares soxr>=0.3.2")
    return {"sha256": sha256_bytes(raw), "name": info["name"], "requires_dist": normalized}


def build_report(project: Path, source: Path, matcha: Path) -> dict[str, Any]:
    project_report = audit_project(project)
    source_identity = require_clean_pinned_checkout(source, SOURCE_REPOSITORY, SOURCE_REVISION, SOURCE_URL)
    source_identity["files"] = require_source_files(source, SOURCE_CONTRACTS)
    matcha_identity = require_clean_pinned_checkout(matcha, MATCHA_REPOSITORY, MATCHA_REVISION, MATCHA_URL)
    matcha_identity["files"] = require_source_files(matcha, MATCHA_CONTRACTS)
    return {
        "format": SCHEMA,
        "status": "BLOCKED_UNRESOLVED_COSYVOICE3_COMPOSITE",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "weights": "NOT_ACQUIRED"},
        "source": source_identity,
        "matcha": matcha_identity,
        "project": project_report,
        "complete_composite": {
            "status": "BLOCKED_FORBIDDEN_SOXR_CLOSURE",
            "entrypoint": "cosyvoice.cli.cosyvoice.AutoModel -> CosyVoice3Model.tts -> token2wav",
            "dependency_path": [
                "official requirements.txt: librosa==0.10.2",
                "Matcha matcha/utils/audio.py: librosa.filters.mel",
                "PyPI librosa 0.10.2 metadata: soxr>=0.3.2",
            ],
            "primary_sources": {
                "official_requirements": OFFICIAL_REQUIREMENTS_URL,
                "librosa_metadata": LIBROSA_METADATA_URL,
                "official_model_card": f"https://huggingface.co/{MODEL_REPOSITORY}",
            },
            "execution": "NOT_RUN_MODEL_FREE_ONLY",
        },
        "components": {
            "llm": {
                "status": "SOURCE_CONTRACT_ONLY",
                "boundary": "CosyVoice3LM/token generation; no PCM or complete TTS claim",
            },
            "flow_hiftnet": {
                "status": "BLOCKED_FORBIDDEN_SOXR_CLOSURE",
                "scope": "official mel/frontend and flow-to-HiFT composition",
            },
        },
        "runtime": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "publication": "NO_UPLOAD",
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise AuditError("output must be absent with an existing parent")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    assert LIBROSA_SOXR_REQUIREMENT == "soxr>=0.3.2"
    assert FORBIDDEN_PACKAGE == "soxr"
    assert MODEL_REVISION and len(MODEL_REVISION) == 40
    assert SOURCE_REVISION and len(SOURCE_REVISION) == 40
    assert MATCHA_REVISION and len(MATCHA_REVISION) == 40
    assert OFFICIAL_DECLARATIONS["librosa"] == "librosa==0.10.2"
    with tempfile.TemporaryDirectory(prefix="cosyvoice3-metadata-") as tmp:
        # The parser contract is exercised without network access; production
        # uses fetch_librosa_metadata() against the primary PyPI JSON endpoint.
        payload = {"info": {"name": "librosa", "version": LIBROSA_VERSION, "requires_dist": [LIBROSA_SOXR_REQUIREMENT]}}
        raw = json.dumps(payload).encode("utf-8")
        metadata = parse_librosa_metadata(raw)
        assert LIBROSA_SOXR_REQUIREMENT in metadata["requires_dist"]
        duplicate = b'{"info":{"name":"librosa","name":"librosa"}}'
        try:
            parse_librosa_metadata(duplicate)
        except AuditError as error:
            assert "duplicate JSON key" in str(error)
        else:
            raise AssertionError("duplicate PyPI JSON key was accepted")
        wrong_name = json.dumps({"info": {"name": "librosa-fork", "version": LIBROSA_VERSION, "requires_dist": [LIBROSA_SOXR_REQUIREMENT]}}).encode("utf-8")
        try:
            parse_librosa_metadata(wrong_name)
        except AuditError as error:
            assert "identity is malformed" in str(error)
        else:
            raise AssertionError("wrong PyPI distribution name was accepted")
    with tempfile.TemporaryDirectory(prefix="cosyvoice3-source-audit-") as tmp:
        root = Path(tmp)
        project = root / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            """[project]\nname = \"audit-test\"\ndependencies = [\"librosa==0.10.2\"]\n\n[tool.vokra.cosyvoice3_reference]\nlock_status = \"BLOCKED_FORBIDDEN_SOXR_IN_AUTHENTICATED_OFFICIAL_CLOSURE\"\ndedicated_sync = \"FORBIDDEN_UNTIL_LICENSE_SIGNOFF_AND_LOCK\"\n""",
            encoding="utf-8",
        )
        metadata = {"sha256": sha256_bytes(raw), "name": "librosa", "requires_dist": [LIBROSA_SOXR_REQUIREMENT]}
        report = audit_project(project, metadata)
        assert report["uv_lock"]["status"] == "ABSENT_BY_DESIGN"
        (project / "uv.lock").write_text("# forbidden test lock\n", encoding="utf-8")
        try:
            audit_project(project, metadata)
        except AuditError as error:
            assert "uv.lock must remain absent" in str(error)
        else:
            raise AssertionError("forbidden lock was accepted")
        (project / "uv.lock").unlink()
        (project / "pyproject.toml").write_text(
            """[project]\nname = \"audit-test\"\ndependencies = [\"librosa==0.10.3\"]\n\n[tool.vokra.cosyvoice3_reference]\nlock_status = \"BLOCKED_FORBIDDEN_SOXR_IN_AUTHENTICATED_OFFICIAL_CLOSURE\"\ndedicated_sync = \"FORBIDDEN_UNTIL_LICENSE_SIGNOFF_AND_LOCK\"\n""",
            encoding="utf-8",
        )
        try:
            audit_project(project, metadata)
        except AuditError as error:
            assert "librosa==0.10.2" in str(error)
        else:
            raise AssertionError("drifted librosa pin was accepted")
    print("cosyvoice3_source_audit self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path, default=REFERENCE_PROJECT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--matcha-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source, args.matcha_source, args.output)):
            parser.error("--self-test cannot be combined with source/matcha/output")
        self_test()
        return 0
    if args.source is None or args.matcha_source is None or args.output is None:
        parser.error("normal run requires --source --matcha-source --output")
    try:
        report = build_report(args.project, args.source, args.matcha_source)
        write_report(report, args.output)
    except (AuditError, OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        print(f"cosyvoice3_source_audit: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"cosyvoice3_source_audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
