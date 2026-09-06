#!/usr/bin/env -S uv run --frozen --project tools/parity/mms_1b_all --python 3.12 python
"""Safetensors-only MMS backbone + one language-adapter inspector.

This tool does not download checkpoints or select a language implicitly. The
 caller must provide a snapshot resolved at the pinned revision and an official
 adapter code. It does not merge or emit tensors: Transformers' adapter
 mapping remains the source of truth until a composed state-dict comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY = "facebook/mms-1b-all"
REVISION = "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
LANGUAGE_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    def reject(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def require_safe_path(path: str | Path, label: str, *, absent: bool = False) -> None:
    """Reject lexical dot components and symlink ancestry before resolution."""
    raw = str(path)
    if any(part in {".", ".."} for part in raw.split("/")):
        raise ValueError(f"{label} contains a dot path component")
    path = Path(path)
    absolute = Path.cwd() / path if not path.is_absolute() else path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} has symlinked ancestry: {cursor}")
    if absent and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"{label} must be absent and non-symlink: {path}")


def validate_snapshot_layout(snapshot: Path, language: str) -> None:
    expected = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "vocab.json",
        f"adapter.{language}.safetensors",
        f"vocabs/{language}.txt",
    }
    actual: set[str] = set()
    for member in snapshot.rglob("*"):
        relative = member.relative_to(snapshot).as_posix()
        if member.is_symlink():
            raise ValueError(f"snapshot member is symlinked: {relative}")
        if member.is_dir():
            if relative != "vocabs" and not (relative == ".cache" or relative.startswith(".cache/")):
                raise ValueError(f"unexpected snapshot directory: {relative}")
        elif member.is_file():
            if not (relative == ".cache" or relative.startswith(".cache/")):
                actual.add(relative)
        else:
            raise ValueError(f"snapshot member is not regular: {relative}")
    if actual != expected:
        raise ValueError(f"snapshot file-set drift: got {sorted(actual)!r}, expected {sorted(expected)!r}")


def publish_exclusive_directory(temporary: Path, output: Path) -> None:
    """Publish prepared files without ever replacing an existing directory."""
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"output appeared during preparation: {output}") from None
    try:
        for child in sorted(temporary.iterdir(), key=lambda path: path.name):
            child.rename(output / child.name)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def self_test() -> None:
    assert REVISION.isascii() and len(REVISION) == 40
    assert LANGUAGE_RE.fullmatch("eng")
    assert LANGUAGE_RE.fullmatch("azj-script_cyrillic")
    assert LANGUAGE_RE.fullmatch("cac-dialect_sanmateoixtatan")
    for invalid in ("English", "eng/../x", "eng.txt", "../eng", "eng."):
        assert not LANGUAGE_RE.fullmatch(invalid)
    for invalid in ("./snapshot", "nested/../snapshot"):
        try:
            require_safe_path(invalid, "self-test")
        except ValueError:
            pass
        else:
            raise SystemExit(f"self-test accepted unsafe path: {invalid}")
    assert load_json.__name__ == "load_json"
    with tempfile.TemporaryDirectory() as directory:
        duplicate = Path(directory) / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
        try:
            load_json(duplicate)
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted duplicate JSON key")
        snapshot = Path(directory) / REVISION
        (snapshot / "vocabs").mkdir(parents=True)
        for name in (
            "config.json", "preprocessor_config.json", "tokenizer_config.json",
            "special_tokens_map.json", "model.safetensors", "vocab.json",
            "adapter.eng.safetensors",
        ):
            (snapshot / name).write_bytes(b"x")
        (snapshot / "vocabs/eng.txt").write_bytes(b"x")
        validate_snapshot_layout(snapshot, "eng")
        (snapshot / "unexpected.bin").write_bytes(b"x")
        try:
            validate_snapshot_layout(snapshot, "eng")
        except ValueError:
            pass
        else:
            raise SystemExit("self-test accepted an extra snapshot file")
        existing = Path(directory) / "existing-output"
        existing.mkdir()
        staging = Path(tempfile.mkdtemp(dir=directory))
        (staging / "prepared_manifest.json").write_bytes(b"x")
        try:
            publish_exclusive_directory(staging, existing)
        except FileExistsError:
            pass
        else:
            raise SystemExit("self-test replaced an existing output directory")
        assert existing.is_dir() and not (existing / "prepared_manifest.json").exists()
        shutil.rmtree(staging, ignore_errors=True)


def manifest(tensors: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(tensors.items())
    }


def prepare(snapshot: str | Path, language: str, evidence: str | Path) -> None:
    require_safe_path(snapshot, "snapshot")
    require_safe_path(evidence, "evidence output", absent=True)
    snapshot = Path(snapshot)
    evidence = Path(evidence)
    if not language or not LANGUAGE_RE.fullmatch(language):
        raise ValueError(f"language must be the official lowercase adapter code: {language!r}")
    if snapshot.name != REVISION:
        raise ValueError(f"snapshot basename {snapshot.name!r} is not pinned revision {REVISION}")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError("snapshot must be a regular non-symlink directory")
    validate_snapshot_layout(snapshot, language)
    root_path = snapshot / "model.safetensors"
    adapter_path = snapshot / f"adapter.{language}.safetensors"
    vocab_path = snapshot / "vocabs" / f"{language}.txt"
    vocab_json = snapshot / "vocab.json"
    for path in (
        snapshot / "config.json",
        snapshot / "preprocessor_config.json",
        snapshot / "tokenizer_config.json",
        snapshot / "special_tokens_map.json",
        root_path,
        adapter_path,
        vocab_path,
        vocab_json,
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required pinned checkpoint asset is missing: {path}")
    try:
        vocab_payload = load_json(vocab_json)
        selected_vocab = vocab_payload[language]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError(f"vocab.json has no valid selected vocabulary for {language!r}") from None
    if not isinstance(selected_vocab, dict) or not selected_vocab:
        raise ValueError(f"selected vocabulary is empty or malformed for {language!r}")

    # Heavy checkpoint imports stay out of parser/path self-tests.
    import torch
    from safetensors.torch import load_file

    root = load_file(str(root_path), device="cpu")
    adapter = load_file(str(adapter_path), device="cpu")
    tensors = list(root.values()) + list(adapter.values())
    if not tensors or any(tensor.dtype not in (torch.float16, torch.float32, torch.bfloat16) for tensor in tensors):
        raise ValueError("MMS checkpoint contains an unsupported or empty tensor set")
    if not vocab_path.read_bytes().strip():
        raise ValueError(f"selected vocabulary is empty: {vocab_path}")
    payload = {
        "contract": "vokra-mms-1b-all-backbone-adapter-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "language": language,
        "source_files": {
            root_path.name: {"sha256": sha256(root_path), "bytes": root_path.stat().st_size, "tensor_manifest": manifest(root)},
            adapter_path.name: {"sha256": sha256(adapter_path), "bytes": adapter_path.stat().st_size, "tensor_manifest": manifest(adapter)},
            "vocabs/" + vocab_path.name: {
                "sha256": sha256(vocab_path),
                "bytes": vocab_path.stat().st_size,
            },
            vocab_json.name: {
                "sha256": sha256(vocab_json),
                "selected_labels": len(selected_vocab),
            },
        },
        "composition": "UNAUTHENTICATED; compare official Transformers composed state_dict before conversion",
        "license": "cc-by-nc-4.0",
        "runtime_status": "BLOCKED_PENDING_AUTHENTICATED_MANIFEST",
        "parity_status": "BLOCKED_PENDING_AUTHENTICATED_MANIFEST",
    }
    parent = evidence.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"evidence output parent must be an existing regular directory: {parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{evidence.name}.", dir=parent))
    try:
        manifest_path = temporary / "prepared_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        publish_exclusive_directory(temporary, evidence)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--language")
    parser.add_argument("--evidence-dir")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot_dir, args.language, args.evidence_dir)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("mms_1b_all_prepare_checkpoint self-test: OK")
        return 0
    if None in (args.snapshot_dir, args.language, args.evidence_dir):
        parser.error("normal runs require --snapshot-dir, --language, and --evidence-dir")
    try:
        prepare(args.snapshot_dir, args.language, args.evidence_dir)
    except (OSError, ValueError) as error:
        parser.error(f"MMS checkpoint validation blocked: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
