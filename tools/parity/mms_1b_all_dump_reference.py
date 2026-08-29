#!/usr/bin/env python3
"""Dump an official Transformers MMS reference without declaring parity.

The caller supplies a local Hugging Face snapshot already resolved to the
single pinned revision.  This tool deliberately requires one explicit
language adapter and records the resulting state-dict/output manifest.  It
does not download checkpoints, infer a language, or emit tolerances/PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC, __version__ as transformers_version


REPOSITORY = "facebook/mms-1b-all"
REVISION = "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
LANGUAGE_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_manifest(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
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
    assert "Wav2Vec2ForCTC" in inspect.getsource(Wav2Vec2ForCTC)


def dump_reference(snapshot: Path, language: str, output: Path) -> None:
    if not language or not LANGUAGE_RE.fullmatch(language):
        raise ValueError("--language must be the explicit official lowercase adapter code")
    if snapshot.name != REVISION:
        raise ValueError(f"snapshot basename {snapshot.name!r} is not pinned revision {REVISION}")
    vocab_json = snapshot / "vocab.json"
    vocab_txt = snapshot / "vocabs" / f"{language}.txt"
    for required in ("config.json", "preprocessor_config.json", "model.safetensors", "vocab.json"):
        if not (snapshot / required).is_file() or (snapshot / required).is_symlink():
            raise FileNotFoundError(f"pinned snapshot is missing {required}")
    adapter = snapshot / f"adapter.{language}.safetensors"
    if not adapter.is_file() or adapter.is_symlink():
        raise FileNotFoundError(f"pinned snapshot is missing selected adapter {adapter.name}")
    if not vocab_txt.is_file() or vocab_txt.is_symlink():
        raise FileNotFoundError(f"pinned snapshot is missing selected vocabulary sidecar {vocab_txt}")
    try:
        vocab_payload = json.loads(vocab_json.read_text(encoding="utf-8"))
        selected_vocab = vocab_payload[language]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        raise ValueError(f"vocab.json has no valid selected vocabulary for {language!r}") from None
    if not isinstance(selected_vocab, dict) or not selected_vocab:
        raise ValueError(f"selected vocabulary is empty or malformed for {language!r}")

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

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"reference output must be absent and non-symlink: {output}")
    output.mkdir(parents=True)
    np.save(output / "logits.npy", logits[0].cpu().numpy())
    np.save(output / "greedy_token_ids.npy", token_ids.cpu().numpy())
    source_paths = [
        snapshot / name
        for name in ("config.json", "preprocessor_config.json", "model.safetensors", "vocab.json")
    ] + [adapter, vocab_txt]
    source_files = {
        path.relative_to(snapshot).as_posix():
        {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in source_paths
    }
    source_file = Path(inspect.getsourcefile(Wav2Vec2ForCTC) or "")
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
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_nonzero": bool(torch.any(logits != 0)),
        "greedy_token_ids_sha256": hashlib.sha256(
            token_ids.cpu().numpy().tobytes()
        ).hexdigest(),
        "decoded_text": decoded,
        "license": "cc-by-nc-4.0",
        "runtime_status": "INSPECTION_ONLY",
        "parity_status": "INSPECTION_ONLY",
        "tolerance": None,
    }
    (output / "reference_manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--language")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.snapshot_dir, args.language, args.output_dir)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("mms_1b_all_dump_reference self-test: OK")
        return 0
    if None in (args.snapshot_dir, args.language, args.output_dir):
        parser.error("normal runs require --snapshot-dir, --language, and --output-dir")
    dump_reference(args.snapshot_dir, args.language, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
