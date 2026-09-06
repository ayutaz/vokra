#!/usr/bin/env -S uv run --frozen --project tools/parity/mms_1b_all --python 3.12 python
"""Dump an official Transformers MMS reference without declaring parity.

The caller supplies a local Hugging Face snapshot already resolved to the
single pinned revision.  This tool deliberately requires one explicit
language adapter and records the resulting state-dict/output manifest.  It
does not download checkpoints, infer a language, or emit tolerances/PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
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


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
    """Publish reference files without replacing an existing directory."""
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"output appeared during reference generation: {output}") from None
    try:
        for child in sorted(temporary.iterdir(), key=lambda path: path.name):
            child.rename(output / child.name)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def tensor_manifest(state_dict: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(state_dict.items())
    }


def self_test() -> None:
    assert REVISION.isascii() and len(REVISION) == 40
    assert REPOSITORY == "facebook/mms-1b-all"
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
    assert "Wav2Vec2ForCTC" in Path(__file__).read_text(encoding="utf-8")
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
        (staging / "reference_manifest.json").write_bytes(b"x")
        try:
            publish_exclusive_directory(staging, existing)
        except FileExistsError:
            pass
        else:
            raise SystemExit("self-test replaced an existing output directory")
        assert existing.is_dir() and not (existing / "reference_manifest.json").exists()
        shutil.rmtree(staging, ignore_errors=True)


def dump_reference(snapshot: str | Path, language: str, output: str | Path) -> None:
    require_safe_path(snapshot, "snapshot")
    require_safe_path(output, "reference output", absent=True)
    snapshot = Path(snapshot)
    output = Path(output)
    if not language or not LANGUAGE_RE.fullmatch(language):
        raise ValueError("--language must be the explicit official lowercase adapter code")
    if snapshot.name != REVISION:
        raise ValueError(f"snapshot basename {snapshot.name!r} is not pinned revision {REVISION}")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError("snapshot must be a regular non-symlink directory")
    validate_snapshot_layout(snapshot, language)
    vocab_json = snapshot / "vocab.json"
    vocab_txt = snapshot / "vocabs" / f"{language}.txt"
    for required in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "vocab.json",
    ):
        if not (snapshot / required).is_file() or (snapshot / required).is_symlink():
            raise FileNotFoundError(f"pinned snapshot is missing {required}")
    adapter = snapshot / f"adapter.{language}.safetensors"
    if not adapter.is_file() or adapter.is_symlink():
        raise FileNotFoundError(f"pinned snapshot is missing selected adapter {adapter.name}")
    if not vocab_txt.is_file() or vocab_txt.is_symlink():
        raise FileNotFoundError(f"pinned snapshot is missing selected vocabulary sidecar {vocab_txt}")
    try:
        vocab_payload = load_json(vocab_json)
        selected_vocab = vocab_payload[language]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError(f"vocab.json has no valid selected vocabulary for {language!r}") from None
    if not isinstance(selected_vocab, dict) or not selected_vocab:
        raise ValueError(f"selected vocabulary is empty or malformed for {language!r}")

    # Heavy imports stay below parser/path self-tests.
    import inspect
    import numpy as np
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC, __version__ as transformers_version

    # `target_lang` is the official Transformers MMS composition surface.  It
    # invokes Wav2Vec2ForCTC.load_adapter for this one language; do not merge
    # or reinterpret adapter tensors in the independent oracle.
    processor = AutoProcessor.from_pretrained(
        str(snapshot), local_files_only=True, revision=REVISION, target_lang=language
    )
    tokenizer = processor.tokenizer
    if getattr(tokenizer, "target_lang", None) != language:
        raise ValueError("processor tokenizer target language does not match selected adapter")
    if dict(getattr(tokenizer, "encoder", {})) != selected_vocab:
        raise ValueError("processor tokenizer vocabulary does not match pinned vocab.json")
    model = Wav2Vec2ForCTC.from_pretrained(
        str(snapshot),
        local_files_only=True,
        revision=REVISION,
        target_lang=language,
    )
    model.eval()

    sample_count = 16_000
    waveform = torch.linspace(-0.25, 0.25, sample_count, dtype=torch.float32).unsqueeze(0)
    inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    if logits.numel() == 0 or not bool(torch.isfinite(logits).all()):
        raise ValueError("official MMS logits are empty or non-finite")
    if not bool(torch.any(logits != 0)):
        raise ValueError("official MMS logits are all zero")
    token_ids = logits.argmax(dim=-1)[0].to(dtype=torch.int64)
    if token_ids.numel() == 0 or bool(torch.any(token_ids < 0)):
        raise ValueError("official MMS greedy token ids are empty or malformed")
    if int(token_ids.max()) >= logits.shape[-1]:
        raise ValueError("official MMS greedy token id exceeds logits vocabulary")
    decoded = processor.batch_decode(token_ids.unsqueeze(0).cpu().numpy())[0]

    source_paths = [
        snapshot / name
        for name in (
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
            "vocab.json",
        )
    ] + [adapter, vocab_txt]
    source_files = {
        path.relative_to(snapshot).as_posix():
        {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in source_paths
    }
    source_file = Path(inspect.getsourcefile(Wav2Vec2ForCTC) or "")
    if not source_file.is_absolute() or not source_file.is_file() or source_file.is_symlink() or any(parent.is_symlink() for parent in source_file.parents):
        raise ValueError("Transformers source file is not an absolute regular symlink-free file")
    evidence = {
        "contract": "vokra-mms-1b-all-backbone-adapter-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "resolved_snapshot": str(snapshot.resolve()),
        "language": language,
        "composition": "AutoProcessor.from_pretrained(target_lang=language) + Wav2Vec2ForCTC.from_pretrained(target_lang=language)",
        "selected_vocabulary": {
            "path": "vocab.json[" + language + "]",
            "sha256": sha256(vocab_json),
            "sidecar_path": f"vocabs/{language}.txt",
            "sidecar_sha256": sha256(vocab_txt),
            "labels": len(selected_vocab),
        },
        "source_files": source_files,
        "transformers_source": {
            "path": str(source_file),
            "sha256": sha256(source_file) if source_file.is_file() else None,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
        "state_dict_tensor_manifest": tensor_manifest(model.state_dict()),
        "state_dict_tensor_manifest_sha256": canonical_sha256(
            tensor_manifest(model.state_dict())
        ),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_nonzero": bool(torch.any(logits != 0)),
        "greedy_token_ids_sha256": hashlib.sha256(
            token_ids.cpu().numpy().tobytes()
        ).hexdigest(),
        "decoded_text": decoded,
        "license": "cc-by-nc-4.0",
        "runtime_status": "BLOCKED_PENDING_AUTHENTICATED_MANIFEST",
        "parity_status": "BLOCKED_PENDING_AUTHENTICATED_MANIFEST",
        "tolerance": None,
    }
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"reference output parent must be an existing regular directory: {parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        np.save(temporary / "logits.npy", logits[0].cpu().numpy())
        np.save(temporary / "greedy_token_ids.npy", token_ids.cpu().numpy())
        evidence["artifacts"] = {
            name: {"sha256": sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
            for name in ("logits.npy", "greedy_token_ids.npy")
        }
        (temporary / "reference_manifest.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        publish_exclusive_directory(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--language")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot_dir, args.language, args.output_dir)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("mms_1b_all_dump_reference self-test: OK")
        return 0
    if None in (args.snapshot_dir, args.language, args.output_dir):
        parser.error("normal runs require --snapshot-dir, --language, and --output-dir")
    try:
        dump_reference(args.snapshot_dir, args.language, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(f"MMS reference validation blocked: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
