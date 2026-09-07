#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Generate the SBV2 JP-Extra G2P contract on an isolated VAST worker.

This program is intentionally an orchestrator around the *authenticated*
Style-Bert-VITS2 ``text`` package.  It contains no Japanese pronunciation
logic and has no fallback implementation.  The real command therefore needs
only an already-cloned official source tree; fixed vocabulary/tone dimensions
are derived from its authenticated symbols module. This file never downloads
a checkpoint or runs Cargo.

``--self-test`` is hermetic.  It exercises the path, identity, import, and
atomic-output guards with tiny fake upstream modules, but never imports an
upstream dependency and never claims those fake rows are production evidence.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import chdir
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "vokra-sbv2-jp-extra-g2p-v1"
GENERATOR = "tools/parity/sbv2_jp_extra/generate_contract.py"
MODEL_REPO = "litagin/Style-Bert-VITS2-2.0-base-JP-Extra"
EXPECTED_SOURCE: dict[str, str] = {
    "hf_revision": "a731761009f3c96d104487be6ad332bf1bb5a3a5",
    "source_commit": "ef93f388fc1ddf0dc0f598126c1964923f1df94f",
    "symbols_blob": "846de64584e9ba4b8d96aab36d4efbcefb1a11e7",
    "japanese_blob": "5c055875626c16bd7d3489d02b4952ec90a3bbf6",
    "mora_blob": "b43e54d8d8297cf1eac0e3e3f0eef6b4f1c24fa3",
    "common_log_blob": "51dca5f3f39047ed9fb58f59d765ca0a332bc49f",
    "stdout_wrapper_blob": "23c6e76462753190d77a1b58dfe022012c90a028",
    "init_blob": "495e57b50d87a4ca3e8fe8dbaf003b4888581927",
    "license_blob": "0ad25db4bd1d86c452db3f9602ccdbe172438f52",
    "deberta_config_blob": "9fb6b0ac2ec49b6556e58b5ed9492eb33166714d",
    "deberta_special_tokens_blob": "a8b3208c2884c4efb86e49300fdd3dc877220cdf",
    "deberta_tokenizer_config_blob": "8ab2175580e45760875557201e5543019ca3039b",
    "deberta_vocab_blob": "ef3652a1877f4c898e6fcb3e605c432c7bcc56b1",
}
SOURCE_PATHS: tuple[tuple[str, str], ...] = (
    ("symbols_blob", "text/symbols.py"),
    ("japanese_blob", "text/japanese.py"),
    ("mora_blob", "text/japanese_mora_list.py"),
    ("common_log_blob", "common/log.py"),
    ("stdout_wrapper_blob", "common/stdout_wrapper.py"),
    ("init_blob", "text/__init__.py"),
    ("license_blob", "LICENSE"),
    (
        "deberta_config_blob",
        "bert/deberta-v2-large-japanese-char-wwm/config.json",
    ),
    (
        "deberta_special_tokens_blob",
        "bert/deberta-v2-large-japanese-char-wwm/special_tokens_map.json",
    ),
    (
        "deberta_tokenizer_config_blob",
        "bert/deberta-v2-large-japanese-char-wwm/tokenizer_config.json",
    ),
    ("deberta_vocab_blob", "bert/deberta-v2-large-japanese-char-wwm/vocab.txt"),
)
LANGUAGE_IDS = {"ZH": 0, "JP": 1, "EN": 2}
TONE_COUNTS = {"ZH": 6, "JP": 2, "EN": 4}
TONE_OFFSETS = {"ZH": 0, "JP": 6, "EN": 8}
CORPUS: tuple[str, ...] = (
    "こんにちは。",
    "おはようございます。",
    "日本語の音声合成を検証します。",
    "東京で会いましょう！",
)
MODEL_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pth",
    ".pt",
    ".ckpt",
    ".gguf",
    ".onnx",
}
MODEL_COMPONENTS = {
    "checkpoint",
    "checkpoints",
    "model",
    "models",
    "weight",
    "weights",
    "hf_cache",
    "huggingface",
}


class ContractError(RuntimeError):
    """A fail-closed contract or evidence error."""


@dataclass(frozen=True)
class SourceEvidence:
    commit: str
    files: tuple[dict[str, Any], ...]


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ContractError(f"git command failed: {args!r}: {error}") from error
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _git_blob(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ContractError(f"cannot read source blob {path}: {error}") from error
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _absolute_real(path: Path, label: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ContractError(f"{label} must be an absolute path")
    if "." in path.parts or ".." in path.parts:
        raise ContractError(f"{label} may not contain dot path components")
    candidate = path if must_exist else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.is_symlink():
        raise ContractError(f"{label} has a symlink ancestor: {path}")
    current = Path(path.anchor or "/")
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"{label} has a symlink component: {current}")
    if must_exist:
        if not path.exists() or path.is_symlink():
            raise ContractError(f"{label} must be a real existing path: {path}")
        return path
    if path.exists() or path.is_symlink():
        raise ContractError(f"{label} must be absent: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ContractError(f"{label} parent must be a real directory: {path.parent}")
    return path


def _reject_model_path(path: Path, label: str) -> None:
    lowered = {part.lower() for part in path.parts}
    if path.suffix.lower() in MODEL_SUFFIXES or lowered & MODEL_COMPONENTS:
        raise ContractError(
            f"{label} looks like a model/checkpoint/weight path; model bytes are forbidden: {path}"
        )


def _require_linux_vast() -> None:
    if platform.system() != "Linux":
        raise ContractError("source acquisition/execution is VAST Linux-only")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise ContractError("VOKRA_PUBLISH_ON_VAST=1 is required")


def validate_checkout(root: Path, expected_head: str) -> dict[str, Any]:
    root = _absolute_real(root, "Vokra checkout", must_exist=True)
    if not root.is_dir() or not (root / ".git").exists():
        raise ContractError("Vokra checkout is not a git directory")
    if len(expected_head) != 40 or any(c not in "0123456789abcdef" for c in expected_head):
        raise ContractError("expected Vokra HEAD must be lowercase 40-hex")
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_head:
        raise ContractError(f"Vokra checkout HEAD mismatch: {head} != {expected_head}")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ContractError("Vokra checkout must be clean")
    return {"expected_head": expected_head, "head": head, "clean": True}


def verify_source_tree(source_dir: Path, expected: Mapping[str, str] = EXPECTED_SOURCE) -> SourceEvidence:
    """Verify HEAD and canonical Git blob identities before any import."""

    source_dir = _absolute_real(source_dir, "official source", must_exist=True)
    if not source_dir.is_dir() or not (source_dir / ".git").exists():
        raise ContractError("official source must be a git checkout")
    commit = _git(source_dir, "rev-parse", "HEAD")
    if commit != expected["source_commit"]:
        raise ContractError(f"official source commit mismatch: {commit}")
    tree_lines = _git(source_dir, "ls-tree", "-r", commit).splitlines()
    tree: dict[str, tuple[str, str]] = {}
    for line in tree_lines:
        fields = line.split(None, 3)
        if len(fields) == 4:
            mode, kind, blob, path = fields
            tree[path] = (kind, blob)
    evidence: list[dict[str, Any]] = []
    for key, relative in SOURCE_PATHS:
        expected_blob = expected[key]
        tree_row = tree.get(relative)
        if tree_row is None or tree_row[0] != "blob":
            raise ContractError(f"official source blob missing: {relative}")
        if tree_row[1] != expected_blob:
            raise ContractError(
                f"official source blob mismatch for {relative}: {tree_row[1]} != {expected_blob}"
            )
        path = source_dir / relative
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"official source path is not a regular file: {relative}")
        actual_blob = _git_blob(path)
        if actual_blob != expected_blob:
            raise ContractError(f"worktree blob mismatch for {relative}: {actual_blob}")
        evidence.append(
            {
                "path": relative,
                "git_blob_sha1": expected_blob,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return SourceEvidence(commit=commit, files=tuple(evidence))


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _purge_text_modules() -> None:
    for name in tuple(sys.modules):
        if name == "text" or name.startswith("text."):
            del sys.modules[name]


def _reject_gpl_closure() -> None:
    """Reject the English GPL closure even when it is installed but unused."""
    for module_name in ("g2p_en", "distance"):
        if module_name in sys.modules or any(name.startswith(f"{module_name}.") for name in sys.modules):
            raise ContractError(f"forbidden GPL frontend dependency already imported: {module_name}")
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError) as error:
            raise ContractError(f"cannot inspect forbidden dependency {module_name}: {error}") from error
        if spec is not None:
            raise ContractError(f"forbidden GPL frontend dependency is installed: {module_name}")


def _install_num2words_sentinel() -> None:
    """Allow the official import while making numeric normalization fail closed."""
    existing = sys.modules.get("num2words")
    if existing is not None:
        if getattr(existing, "__vokra_num2words_sentinel__", False):
            return
        raise ContractError("num2words is already imported; LGPL code is forbidden")
    try:
        spec = importlib.util.find_spec("num2words")
    except (ImportError, ValueError) as error:
        raise ContractError(f"cannot inspect forbidden LGPL dependency num2words: {error}") from error
    if spec is not None:
        raise ContractError("num2words distribution is installed; LGPL code is forbidden")
    sentinel = types.ModuleType("num2words")
    sentinel.__vokra_num2words_sentinel__ = True

    def forbidden_num2words(*_args: Any, **_kwargs: Any) -> str:
        raise ContractError("numeric-text G2P is blocked: num2words is LGPL and unavailable")

    sentinel.num2words = forbidden_num2words
    sys.modules["num2words"] = sentinel


def _load_source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot construct an official module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, ValueError, SyntaxError) as error:
        sys.modules.pop(name, None)
        raise ContractError(f"official upstream module import failed for {path}: {error}") from error
    return module


def _load_official_sequence(init_path: Path, symbols: Any) -> Any:
    """Compile only the upstream sequence mapper, never execute package imports.

    The pinned ``text/__init__.py`` also exposes the shared phone/tone/language
    mapping, but importing the package eagerly imports all language frontends.
    Selecting the authenticated function and its constant/helper definitions
    keeps the official mapping while preventing an English frontend import.
    """
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise ContractError(f"cannot parse authenticated sequence source: {error}") from error
    assignment_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_symbol_to_id" for target in node.targets)
    ]
    if len(assignment_nodes) != 1:
        raise ContractError("authenticated _symbol_to_id assignment is missing or duplicated")
    function_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cleaned_text_to_sequence"
    ]
    if len(function_nodes) != 1:
        raise ContractError("authenticated cleaned_text_to_sequence function is missing or duplicated")
    assignment = assignment_nodes[0]
    target = function_nodes[0]
    allowed_globals = {"_symbol_to_id", "language_tone_start_map", "language_id_map", "range", "len"}
    local_names = {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    local_names.update(argument.arg for argument in (*target.args.posonlyargs, *target.args.args, *target.args.kwonlyargs))
    unexpected = {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    } - local_names - allowed_globals
    if unexpected:
        raise ContractError(f"authenticated sequence mapper has unexpected globals: {sorted(unexpected)}")
    required_globals = {
        "symbols": getattr(symbols, "symbols", None),
        "language_tone_start_map": getattr(symbols, "language_tone_start_map", None),
        "language_id_map": getattr(symbols, "language_id_map", None),
    }
    if not isinstance(required_globals["symbols"], (tuple, list)):
        raise ContractError("authenticated symbols module does not export symbols")
    if required_globals["language_tone_start_map"] != TONE_OFFSETS:
        raise ContractError("authenticated symbols module tone map is unexpected")
    if required_globals["language_id_map"] != LANGUAGE_IDS:
        raise ContractError("authenticated symbols module language map is unexpected")
    module = ast.Module(
        body=sorted((assignment, target), key=lambda node: tree.body.index(node)),
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "__name__": "sbv2_authenticated_sequence",
        "__file__": str(init_path),
        **required_globals,
    }
    try:
        exec(compile(module, str(init_path), "exec"), namespace, namespace)
    except (NameError, TypeError, ValueError, SyntaxError) as error:
        raise ContractError(f"authenticated sequence mapper could not be isolated: {error}") from error
    sequence = namespace.get("cleaned_text_to_sequence")
    if not callable(sequence):
        raise ContractError("isolated authenticated sequence mapper is not callable")
    symbol_to_id = namespace.get("_symbol_to_id")
    if not isinstance(symbol_to_id, dict) or symbol_to_id != {value: index for index, value in enumerate(required_globals["symbols"])}:
        raise ContractError("authenticated _symbol_to_id mapping is not the exact source mapping")
    return sequence


def import_official_text(source_dir: Path, evidence: SourceEvidence) -> tuple[Any, Any, Any]:
    """Load only authenticated Japanese, symbols, and shared sequence code."""

    _purge_text_modules()
    _reject_gpl_closure()
    _install_num2words_sentinel()
    source_dir = source_dir.resolve(strict=True)
    if _git(source_dir, "rev-parse", "HEAD") != evidence.commit:
        raise ContractError("official import source commit differs from pre-import evidence")
    evidence_by_path = {row["path"]: row for row in evidence.files}
    required_modules = (
        "text/symbols.py",
        "text/japanese_mora_list.py",
        "common/stdout_wrapper.py",
        "common/log.py",
        "text/japanese.py",
        "text/__init__.py",
    )
    for relative in required_modules:
        row = evidence_by_path.get(relative)
        path = source_dir / relative
        if row is None or not path.is_file() or _sha256(path) != row["sha256"] or _git_blob(path) != row["git_blob_sha1"]:
            raise ContractError(f"non-official import source bytes for {relative}")
    package = types.ModuleType("text")
    package.__path__ = [str(source_dir / "text")]
    package.__file__ = str(source_dir / "text/__init__.py")
    package.__package__ = "text"
    sys.modules["text"] = package
    symbols = _load_source_module("text.symbols", source_dir / "text/symbols.py")
    punctuation = getattr(symbols, "punctuation", None)
    if not isinstance(punctuation, (tuple, list, str)):
        raise ContractError("authenticated symbols module does not export punctuation")
    package.punctuation = punctuation
    _load_source_module("text.japanese_mora_list", source_dir / "text/japanese_mora_list.py")
    common = types.ModuleType("common")
    common.__path__ = [str(source_dir / "common")]
    common.__package__ = "common"
    sys.modules["common"] = common
    _load_source_module("common.stdout_wrapper", source_dir / "common/stdout_wrapper.py")
    _load_source_module("common.log", source_dir / "common/log.py")
    # The authenticated upstream module resolves its DeBERTa tokenizer from a
    # source-root-relative path.  Keep this narrowly scoped to that module and
    # let contextlib restore the caller's CWD even when import raises.
    with chdir(source_dir):
        japanese = _load_source_module("text.japanese", source_dir / "text/japanese.py")
    sequence = _load_official_sequence(source_dir / "text/__init__.py", symbols)
    expected_files = {
        "text.japanese": source_dir / "text/japanese.py",
        "text.symbols": source_dir / "text/symbols.py",
        "text.japanese_mora_list": source_dir / "text/japanese_mora_list.py",
        "common.stdout_wrapper": source_dir / "common/stdout_wrapper.py",
        "common.log": source_dir / "common/log.py",
    }
    for name, expected in expected_files.items():
        module = sys.modules.get(name)
        actual = getattr(module, "__file__", None)
        if actual is None or Path(actual).resolve() != expected.resolve():
            raise ContractError(f"non-official import for {name}: {actual}")
    if not callable(getattr(japanese, "g2p", None)) or not callable(getattr(japanese, "text_normalize", None)):
        raise ContractError("authenticated upstream Japanese G2P entry point is absent")
    return sequence, symbols, japanese


def _call_japanese(japanese: Any, text: str) -> tuple[str, list[str], list[int], list[int]]:
    normalize = getattr(japanese, "text_normalize", None)
    g2p = getattr(japanese, "g2p", None)
    if not callable(normalize) or not callable(g2p):
        raise ContractError("official Japanese module lacks text_normalize/g2p")
    try:
        signature = inspect.signature(g2p)
    except (TypeError, ValueError) as error:
        raise ContractError("official Japanese g2p has no inspectable signature") from error
    names = signature.parameters
    kwargs: dict[str, Any] = {}
    if "use_jp_extra" in names:
        kwargs["use_jp_extra"] = True
    elif len(names) < 2:
        raise ContractError("official Japanese g2p signature lacks use_jp_extra")
    if "ignore_unknown" in names:
        kwargs["ignore_unknown"] = False
    try:
        normalized = normalize(text)
        result = g2p(normalized, **kwargs)
    except Exception as error:  # upstream errors are evidence failures, never fallback triggers
        raise ContractError(f"official Japanese G2P failed for {text!r}: {error}") from error
    if not isinstance(normalized, str) or not normalized or not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ContractError("official Japanese g2p did not return (phones, tones, word2ph)")
    phones, tones, word2ph = result
    if not isinstance(normalized, str) or not normalized:
        raise ContractError("official normalized Japanese text is empty")
    if not isinstance(phones, (tuple, list)) or not phones or any(not isinstance(v, str) or not v for v in phones):
        raise ContractError("official Japanese G2P phones are invalid")
    if not isinstance(tones, (tuple, list)) or len(tones) != len(phones):
        raise ContractError("official Japanese phone/tone lengths disagree")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in tones):
        raise ContractError("official Japanese tones are not integers")
    if not isinstance(word2ph, (tuple, list)) or not word2ph or any(
        not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in word2ph
    ):
        raise ContractError("official Japanese word2ph is invalid")
    if sum(word2ph) != len(phones):
        raise ContractError("official Japanese word2ph does not cover phones")
    return normalized, list(phones), [int(v) for v in tones], [int(v) for v in word2ph]


def _sequence(sequence: Any, phones: list[str], tones: list[int]) -> tuple[list[int], list[int]]:
    try:
        result = sequence(phones, tones, "JP")
    except Exception as error:
        raise ContractError(f"official cleaned_text_to_sequence failed: {error}") from error
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ContractError("official cleaned_text_to_sequence must return phone, tone, language lists")
    ids, global_tones, languages = result
    if not isinstance(ids, (tuple, list)) or not isinstance(global_tones, (tuple, list)) or not isinstance(languages, (tuple, list)):
        raise ContractError("official sequence output is not list-like")
    if len(ids) != len(phones) or len(global_tones) != len(phones) or len(languages) != len(phones):
        raise ContractError("official sequence output lengths disagree")
    combined = list(ids) + list(global_tones) + list(languages)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in combined):
        raise ContractError("official sequence output contains non-integers")
    if any(value != LANGUAGE_IDS["JP"] for value in languages):
        raise ContractError("official sequence output language IDs disagree with JP")
    return [int(v) for v in ids], [int(v) for v in global_tones]


def _boundaries(word2ph: list[int]) -> list[bool]:
    result = [False] * sum(word2ph)
    position = 0
    for width in word2ph:
        result[position] = True
        position += width
    return result


def _symbol_table(symbols_module: Any) -> tuple[list[str], int]:
    values = getattr(symbols_module, "symbols", None)
    if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or not v for v in values):
        raise ContractError("official symbols.py does not expose a non-empty symbols sequence")
    values = list(values)
    n_tones = getattr(symbols_module, "num_tones", None)
    if len(values) != 178 or len(set(values)) != len(values):
        raise ContractError("authenticated source symbol table drifted from the JP-Extra boundary")
    if not isinstance(n_tones, int) or isinstance(n_tones, bool) or n_tones != 12:
        raise ContractError("authenticated source num_tones drifted from the JP-Extra boundary")
    return values, n_tones


def build_contract(
    source: SourceEvidence,
    source_dir: Path,
    checkout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence, symbols_module, japanese = import_official_text(source_dir, source)
    symbols, n_tones = _symbol_table(symbols_module)
    n_vocab = len(symbols)
    fixtures: list[dict[str, Any]] = []
    for text in CORPUS:
        normalized, phones, raw_tones, word2ph = _call_japanese(japanese, text)
        ids, tones = _sequence(sequence, phones, raw_tones)
        if any(not 6 <= tone < 8 for tone in tones):
            raise ContractError("official JP sequence emitted a tone outside global JP band [6, 8)")
        if any(not 0 <= value < n_vocab for value in ids):
            raise ContractError("official JP sequence emitted an out-of-range phoneme id")
        boundaries = _boundaries(word2ph)
        fixtures.append(
            {
                "language": "JP",
                "text": text,
                "normalized_text": normalized,
                "phones": phones,
                "word2ph": word2ph,
                "phoneme_ids": ids,
                "tones": tones,
                "word_boundaries": boundaries,
            }
        )
    return {
        "schema": SCHEMA,
        "generator": GENERATOR,
        "source": dict(EXPECTED_SOURCE),
        "source_evidence": {
            "commit": source.commit,
            "files": list(source.files),
        },
        "model": {
            "hf_repo": MODEL_REPO,
            "hf_revision": EXPECTED_SOURCE["hf_revision"],
            "n_vocab": n_vocab,
            "n_tones": n_tones,
            "dimensions_proof": "authenticated_source_symbol_and_tone_table",
        },
        "phoneme_symbols": symbols,
        "sorted_phoneme_symbols": sorted(symbols),
        "language_ids": dict(LANGUAGE_IDS),
        "tone_counts": dict(TONE_COUNTS),
        "tone_offsets": dict(TONE_OFFSETS),
        "fixtures": fixtures,
        "corpus": {"ordering": "fixed-source-order", "texts": list(CORPUS)},
        "environment": {
            "platform": platform.platform(),
            "system": platform.system(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "source_execution": "official Style-Bert-VITS2 Japanese module and shared sequence mapper only",
            "vokra_checkout": checkout if checkout is not None else validate_checkout_from_parent(source_dir),
        },
        "activity": {
            "source_acquisition": True,
            "source_execution": True,
            "model_weight_acquisition": False,
            "model_weight_execution": False,
            "cargo_execution": False,
            "local_execution": False,
            "publication": "NO_UPLOAD",
            "upload": False,
        },
    }


def validate_checkout_from_parent(source_dir: Path) -> dict[str, Any]:
    """Record the Vokra checkout when the source tree is nested beside it."""

    current = source_dir.parent
    for _ in range(5):
        if (current / ".git").exists():
            head = _git(current, "rev-parse", "HEAD")
            return {"root": str(current), "head": head}
        if current == current.parent:
            break
        current = current.parent
    return {"root": None, "head": None}


def ensure_disjoint(first: Path, second: Path) -> None:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    if (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    ):
        raise ContractError(f"paths overlap: {first} and {second}")


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _exclusive_atomic_write(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ContractError(f"output already exists; refusing to clobber: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    except FileExistsError as error:
        raise ContractError(f"output appeared concurrently; refusing to clobber: {path}") from error
    except OSError as error:
        raise ContractError(f"atomic output write failed for {path}: {error}") from error
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_contract(payload: dict[str, Any], output: Path) -> tuple[str, Path]:
    output = _absolute_real(output, "contract output", must_exist=False)
    _reject_model_path(output, "contract output")
    sidecar = output.with_name(output.name + ".sha256")
    _absolute_real(sidecar, "contract SHA-256 sidecar", must_exist=False)
    raw = _json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_raw = f"{digest}  {output.name}\n".encode("ascii")
    _exclusive_atomic_write(output, raw)
    try:
        _exclusive_atomic_write(sidecar, sidecar_raw)
    except ContractError:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return digest, sidecar


def run(args: argparse.Namespace) -> int:
    _require_linux_vast()
    _reject_gpl_closure()
    root = Path(args.vokra_root)
    source_dir = Path(args.source_dir)
    output = Path(args.output)
    work_dir = Path(args.work_dir)
    checkout = validate_checkout(root, args.expected_head)
    _absolute_real(work_dir, "worker directory", must_exist=True)
    if not work_dir.is_dir():
        raise ContractError("worker directory must be a real directory")
    _reject_model_path(work_dir, "worker directory")
    source_dir = _absolute_real(source_dir, "official source", must_exist=True)
    _reject_model_path(source_dir, "official source")
    # Output and worker paths must be disjoint even when the output is absent.
    work_resolved = work_dir.resolve()
    output_resolved = output.resolve(strict=False)
    ensure_disjoint(work_resolved, output_resolved)
    root_resolved = root.resolve(strict=True)
    if root_resolved == output_resolved or root_resolved in output_resolved.parents:
        raise ContractError("contract output overlaps the Vokra checkout")
    source = verify_source_tree(source_dir)
    payload = build_contract(source, source_dir, checkout=checkout)
    digest, sidecar = write_contract(payload, output)
    print(f"sbv2 JP-Extra G2P contract: PASS ({output}) sha256={digest} sidecar={sidecar}")
    return 0


def _fake_git_source(directory: Path) -> tuple[dict[str, str], Path]:
    """Create a tiny fake source checkout for self-test only."""

    source = directory / "fake-source"
    (source / "text").mkdir(parents=True)
    (source / "common").mkdir(parents=True)
    (source / "bert/deberta-v2-large-japanese-char-wwm").mkdir(parents=True)
    files = {
        "text/symbols.py": (
            "symbols = [" + ",".join(repr(f"symbol-{i}") for i in range(178)) + "]\n"
            "language_tone_start_map = {'ZH': 0, 'JP': 6, 'EN': 8}\n"
            "language_id_map = {'ZH': 0, 'JP': 1, 'EN': 2}\n"
            "num_tones = 12\n"
            "punctuation = ['。', '！']\n"
        ),
        "text/japanese.py": (
            "from common.log import logger\n"
            "from num2words import num2words\n"
            "from text import punctuation\n"
            "from text.japanese_mora_list import mora_list\n"
            "def text_normalize(text):\n"
            "    if text == '1':\n"
            "        return num2words(text, lang='ja')\n"
            "    return text\n"
            "def g2p(text, use_jp_extra=True, ignore_unknown=False):\n"
            "    assert use_jp_extra and not ignore_unknown\n"
            "    assert punctuation and mora_list and logger\n"
            "    return ['symbol-1'], [0], [1]\n"
        ),
        "text/japanese_mora_list.py": "mora_list = ['a']\n",
        "common/stdout_wrapper.py": "SAFE_STDOUT = None\n",
        "common/log.py": "from common.stdout_wrapper import SAFE_STDOUT\nlogger = object()\n",
        "text/__init__.py": (
            "from .symbols import *\n"
            "_symbol_to_id = {s: i for i, s in enumerate(symbols)}\n"
            "def cleaned_text_to_sequence(phones, tones, language):\n"
            "    phones = [_symbol_to_id[symbol] for symbol in phones]\n"
            "    tones = [tone + language_tone_start_map[language] for tone in tones]\n"
            "    language = [language_id_map[language] for _ in range(len(phones))]\n"
            "    return phones, tones, language\n"
        ),
        "LICENSE": "fake license\n",
        "bert/deberta-v2-large-japanese-char-wwm/config.json": "{}\n",
        "bert/deberta-v2-large-japanese-char-wwm/special_tokens_map.json": "{}\n",
        "bert/deberta-v2-large-japanese-char-wwm/tokenizer_config.json": "{}\n",
        "bert/deberta-v2-large-japanese-char-wwm/vocab.txt": "token\n",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "self-test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "self-test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fake"], check=True)
    commit = _git(source, "rev-parse", "HEAD")
    expected: dict[str, str] = {"source_commit": commit}
    for key, relative in SOURCE_PATHS:
        expected[key] = _git_blob(source / relative)
    return expected, source


def _expect_failure(label: str, operation: Any) -> None:
    try:
        operation()
    except (ContractError, ValueError, OSError):
        return
    raise AssertionError(f"self-test accepted forbidden case: {label}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="sbv2-jp-extra-self-test-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        expected, source = _fake_git_source(root)
        evidence = verify_source_tree(source, expected)
        assert evidence.commit == expected["source_commit"]
        _expect_failure("wrong commit", lambda: verify_source_tree(source, {**expected, "source_commit": "0" * 40}))
        _expect_failure("wrong blob", lambda: verify_source_tree(source, {**expected, "symbols_blob": "0" * 40}))
        _expect_failure(
            "tokenizer blob drift",
            lambda: verify_source_tree(source, {**expected, "deberta_vocab_blob": "0" * 40}),
        )

        cwd_before = Path.cwd()
        sequence, symbols_module, japanese = import_official_text(source, evidence)
        assert Path.cwd() == cwd_before
        assert symbols_module.symbols[1] == "symbol-1"
        _expect_failure(
            "source symbol table drift",
            lambda: _symbol_table(types.SimpleNamespace(symbols=symbols_module.symbols[:-1], num_tones=12)),
        )
        _expect_failure(
            "source tone table drift",
            lambda: _symbol_table(types.SimpleNamespace(symbols=symbols_module.symbols, num_tones=11)),
        )
        assert japanese.text_normalize("こんにちは。") == "こんにちは。"
        assert japanese.g2p("こんにちは。", use_jp_extra=True, ignore_unknown=False) == (["symbol-1"], [0], [1])
        assert sequence(["symbol-1"], [0], "JP") == ([1], [6], [1])
        assert getattr(sys.modules["num2words"], "__vokra_num2words_sentinel__", False)
        _expect_failure("numeric text LGPL sentinel", lambda: _call_japanese(japanese, "1"))
        payload = build_contract(evidence, source)
        output = root / "contract.json"
        digest, sidecar = write_contract(payload, output)
        assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
        assert sidecar.read_text() == f"{digest}  contract.json\n"
        _expect_failure("duplicate JSON", lambda: json.loads('{"x":1,"x":2}', object_pairs_hook=_unique))
        _expect_failure("pre-existing output", lambda: write_contract(payload, output))
        _expect_failure("model-weight path", lambda: write_contract(payload, root / "weights.gguf"))
        overlap = root / "nested"
        overlap.mkdir()
        _expect_failure("path overlap", lambda: ensure_disjoint(overlap, overlap / "output.json"))
        os.symlink(overlap, root / "link")
        _expect_failure("symlink ancestor", lambda: _absolute_real(root / "link" / "child.json", "output", must_exist=False))
        wrong_source = root / "wrong-source"
        shutil.copytree(source, wrong_source)
        (wrong_source / "text/__init__.py").write_text("# tampered\n", encoding="utf-8")
        _expect_failure("non-official import", lambda: build_contract(evidence, wrong_source))
        sys.modules["g2p_en"] = types.ModuleType("g2p_en")
        _expect_failure("g2p-en GPL closure", _reject_gpl_closure)
        sys.modules.pop("g2p_en", None)
        sys.modules["distance"] = types.ModuleType("distance")
        _expect_failure("distance GPL closure", _reject_gpl_closure)
        sys.modules.pop("distance", None)
        sys.modules.pop("num2words", None)
        sys.modules["num2words"] = types.ModuleType("num2words")
        _expect_failure("num2words LGPL distribution", _install_num2words_sentinel)
        sys.modules.pop("num2words", None)
        _install_num2words_sentinel()
    print("sbv2_jp_extra generate_contract self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vokra-root")
    parser.add_argument("--expected-head")
    parser.add_argument("--source-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if any(
            getattr(args, name) is not None
            for name in ("vokra_root", "expected_head", "source_dir", "work_dir", "output")
        ):
            parser.error("--self-test accepts no execution paths")
        self_test()
        return 0
    required = ("vokra_root", "expected_head", "source_dir", "work_dir", "output")
    if any(getattr(args, name) is None for name in required):
        parser.error("execution requires --vokra-root --expected-head --source-dir --work-dir --output")
    try:
        return run(args)
    except (ContractError, OSError, ValueError) as error:
        parser.exit(2, f"sbv2 JP-Extra contract BLOCKED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
