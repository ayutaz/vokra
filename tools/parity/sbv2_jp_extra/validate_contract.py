#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Validate the data-only SBV2 JP-Extra Japanese G2P contract.

The contract is produced by the isolated VAST worker after it executes the
authenticated upstream reference.  This validator intentionally contains no
frontend implementation and never downloads a model or source repository.
It is safe to run in schema/self-test mode on the maintainer Mac.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_SOURCE = {
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
EXPECTED_KEYS = {
    "schema",
    "generator",
    "source",
    "source_evidence",
    "model",
    "phoneme_symbols",
    "sorted_phoneme_symbols",
    "language_ids",
    "tone_counts",
    "tone_offsets",
    "fixtures",
    "corpus",
    "environment",
    "activity",
}
EXPECTED_SOURCE_FILES = (
    ("text/symbols.py", "symbols_blob"),
    ("text/japanese.py", "japanese_blob"),
    ("text/japanese_mora_list.py", "mora_blob"),
    ("common/log.py", "common_log_blob"),
    ("common/stdout_wrapper.py", "stdout_wrapper_blob"),
    ("text/__init__.py", "init_blob"),
    ("LICENSE", "license_blob"),
    (
        "bert/deberta-v2-large-japanese-char-wwm/config.json",
        "deberta_config_blob",
    ),
    (
        "bert/deberta-v2-large-japanese-char-wwm/special_tokens_map.json",
        "deberta_special_tokens_blob",
    ),
    (
        "bert/deberta-v2-large-japanese-char-wwm/tokenizer_config.json",
        "deberta_tokenizer_config_blob",
    ),
    (
        "bert/deberta-v2-large-japanese-char-wwm/vocab.txt",
        "deberta_vocab_blob",
    ),
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} schema is not exact")
    return value


def validate(payload: object) -> None:
    root = _object(payload, "contract", EXPECTED_KEYS)
    if root["schema"] != "vokra-sbv2-jp-extra-g2p-v1":
        raise ValueError("unsupported contract schema")
    if not isinstance(root["generator"], str) or not root["generator"].strip():
        raise ValueError("generator identity is empty")

    source = _object(root["source"], "source", set(EXPECTED_SOURCE))
    if source != EXPECTED_SOURCE:
        raise ValueError("source identity is not the authenticated JP-Extra contract")

    source_evidence = _object(root["source_evidence"], "source_evidence", {"commit", "files"})
    if source_evidence["commit"] != EXPECTED_SOURCE["source_commit"]:
        raise ValueError("source evidence commit disagrees with authenticated source")
    files = source_evidence["files"]
    if not isinstance(files, list) or len(files) != len(EXPECTED_SOURCE_FILES):
        raise ValueError("source evidence file count is not exact")
    for row, (expected_path, expected_key) in zip(files, EXPECTED_SOURCE_FILES):
        item = _object(row, "source_evidence.files[]", {"path", "git_blob_sha1", "sha256", "bytes"})
        if item["path"] != expected_path or item["git_blob_sha1"] != EXPECTED_SOURCE[expected_key]:
            raise ValueError("source evidence file identity drifted")
        if not isinstance(item["sha256"], str) or not HEX64.fullmatch(item["sha256"]):
            raise ValueError("source evidence file SHA-256 is malformed")
        if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or item["bytes"] <= 0:
            raise ValueError("source evidence file byte count is invalid")

    model = _object(
        root["model"],
        "model",
        {"hf_repo", "hf_revision", "n_vocab", "n_tones", "dimensions_proof"},
    )
    if model["hf_repo"] != "litagin/Style-Bert-VITS2-2.0-base-JP-Extra":
        raise ValueError("model repository is not JP-Extra")
    if model["hf_revision"] != EXPECTED_SOURCE["hf_revision"]:
        raise ValueError("model revision disagrees with source identity")
    n_vocab = model["n_vocab"]
    n_tones = model["n_tones"]
    if not isinstance(n_vocab, int) or isinstance(n_vocab, bool) or n_vocab <= 0:
        raise ValueError("model.n_vocab must be a positive integer")
    if not isinstance(n_tones, int) or isinstance(n_tones, bool) or n_tones <= 0:
        raise ValueError("model.n_tones must be a positive integer")
    if model["dimensions_proof"] != "authenticated_source_symbol_and_tone_table":
        raise ValueError("model dimensions proof is not the authenticated source table")

    symbols = root["phoneme_symbols"]
    if not isinstance(symbols, list) or len(symbols) != n_vocab:
        raise ValueError("phoneme_symbols length does not equal model.n_vocab")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError("phoneme_symbols contains an empty or non-string symbol")
    if len(set(symbols)) != len(symbols):
        raise ValueError("phoneme_symbols contains duplicates")
    sorted_symbols = root["sorted_phoneme_symbols"]
    if sorted_symbols != sorted(symbols):
        raise ValueError("sorted_phoneme_symbols is not the sorted vocabulary")

    language_ids = _object(root["language_ids"], "language_ids", {"ZH", "JP", "EN"})
    if language_ids != {"ZH": 0, "JP": 1, "EN": 2}:
        raise ValueError("language_ids drifted from the authenticated ZH/JP/EN source map")
    tone_counts = _object(root["tone_counts"], "tone_counts", {"ZH", "JP", "EN"})
    if tone_counts != {"ZH": 6, "JP": 2, "EN": 4}:
        raise ValueError("tone_counts drifted from the authenticated source values")
    tone_offsets = root["tone_offsets"]
    if tone_offsets != {"ZH": 0, "JP": 6, "EN": 8}:
        raise ValueError("tone_offsets drifted from the authenticated source values")
    if n_vocab != 112 or n_tones != 12:
        raise ValueError("source vocabulary/tone dimensions do not match the authenticated JP table")

    fixtures = root["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixtures must be a non-empty list")
    for index, fixture in enumerate(fixtures):
        row = _object(
            fixture,
            f"fixtures[{index}]",
            {"language", "text", "normalized_text", "phones", "word2ph", "phoneme_ids", "tones", "word_boundaries"},
        )
        if not isinstance(row["language"], str) or row["language"] != "JP":
            raise ValueError(f"fixtures[{index}] names an unknown language")
        if not isinstance(row["text"], str) or not row["text"]:
            raise ValueError(f"fixtures[{index}].text is empty")
        if not isinstance(row["normalized_text"], str) or not row["normalized_text"]:
            raise ValueError(f"fixtures[{index}].normalized_text is empty")
        phones = row["phones"]
        word2ph = row["word2ph"]
        if not isinstance(phones, list) or len(phones) == 0 or any(not isinstance(value, str) or not value for value in phones):
            raise ValueError(f"fixtures[{index}].phones is invalid")
        if not isinstance(word2ph, list) or not word2ph or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in word2ph):
            raise ValueError(f"fixtures[{index}].word2ph is invalid")
        if sum(word2ph) != len(phones):
            raise ValueError(f"fixtures[{index}].word2ph does not cover phones")
        ids = row["phoneme_ids"]
        tones = row["tones"]
        boundaries = row["word_boundaries"]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"fixtures[{index}].phoneme_ids is empty")
        if not isinstance(tones, list) or len(tones) != len(ids):
            raise ValueError(f"fixtures[{index}] phone/tone lengths disagree")
        if not isinstance(boundaries, list) or len(boundaries) != len(ids):
            raise ValueError(f"fixtures[{index}] phone/boundary lengths disagree")
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < n_vocab for value in ids):
            raise ValueError(f"fixtures[{index}] contains an out-of-range phone id")
        if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < n_tones for value in tones):
            raise ValueError(f"fixtures[{index}] contains an out-of-range tone")
        if row["language"] == "JP" and any(not 6 <= value < 8 for value in tones):
            raise ValueError(
                f"fixtures[{index}] contains a non-global JP tone; expected the [6, 8) JP band"
            )
        if any(not isinstance(value, bool) for value in boundaries):
            raise ValueError(f"fixtures[{index}] contains a non-boolean word boundary")
        expected_boundaries = [False] * len(phones)
        position = 0
        for width in word2ph:
            expected_boundaries[position] = True
            position += width
        if boundaries != expected_boundaries:
            raise ValueError(f"fixtures[{index}] word boundaries do not expand word2ph")

    corpus = _object(root["corpus"], "corpus", {"ordering", "texts"})
    if corpus["ordering"] != "fixed-source-order" or corpus["texts"] != [
        "こんにちは。",
        "おはようございます。",
        "日本語の音声合成を検証します。",
        "東京で会いましょう！",
    ]:
        raise ValueError("deterministic corpus ordering drifted")
    if [fixture["text"] for fixture in fixtures] != corpus["texts"]:
        raise ValueError("fixture ordering does not match fixed corpus")

    environment = _object(
        root["environment"],
        "environment",
        {"platform", "system", "python", "executable", "source_execution", "vokra_checkout"},
    )
    if environment["system"] != "Linux" or environment["source_execution"] != "official Style-Bert-VITS2 Japanese module and shared sequence mapper only":
        raise ValueError("contract was not generated by the official Linux source route")
    checkout = _object(environment["vokra_checkout"], "environment.vokra_checkout", {"root", "head"})
    if checkout["root"] is not None and (not isinstance(checkout["root"], str) or not checkout["root"].startswith("/")):
        raise ValueError("recorded checkout root is not absolute")
    if checkout["head"] is not None and (not isinstance(checkout["head"], str) or not HEX40.fullmatch(checkout["head"])):
        raise ValueError("recorded checkout HEAD is malformed")
    activity = _object(
        root["activity"],
        "activity",
        {"source_acquisition", "source_execution", "model_weight_acquisition", "model_weight_execution", "cargo_execution", "local_execution", "publication", "upload"},
    )
    if activity != {
        "source_acquisition": True,
        "source_execution": True,
        "model_weight_acquisition": False,
        "model_weight_execution": False,
        "cargo_execution": False,
        "local_execution": False,
        "publication": "NO_UPLOAD",
        "upload": False,
    }:
        raise ValueError("activity flags are not the model-free NO_UPLOAD contract")


def self_test() -> None:
    symbols = ["_", "a", "i"]
    valid = {
        "schema": "vokra-sbv2-jp-extra-g2p-v1",
        "generator": "vast-worker-test",
        "source": dict(EXPECTED_SOURCE),
        "source_evidence": {
            "commit": EXPECTED_SOURCE["source_commit"],
            "files": [
                {"path": path, "git_blob_sha1": EXPECTED_SOURCE[key], "sha256": "a" * 64, "bytes": 1}
                for path, key in EXPECTED_SOURCE_FILES
            ],
        },
        "model": {
            "hf_repo": "litagin/Style-Bert-VITS2-2.0-base-JP-Extra",
            "hf_revision": EXPECTED_SOURCE["hf_revision"],
            "n_vocab": 112,
            "n_tones": 12,
            "dimensions_proof": "authenticated_source_symbol_and_tone_table",
        },
        "phoneme_symbols": symbols + [f"symbol-{index}" for index in range(109)],
        "sorted_phoneme_symbols": sorted(symbols + [f"symbol-{index}" for index in range(109)]),
        "language_ids": {"ZH": 0, "JP": 1, "EN": 2},
        "tone_counts": {"ZH": 6, "JP": 2, "EN": 4},
        "tone_offsets": {"ZH": 0, "JP": 6, "EN": 8},
        "fixtures": [
            {
                "language": "JP",
                "text": "test",
                "normalized_text": "test",
                "phones": ["a", "i"],
                "word2ph": [2],
                "phoneme_ids": [1, 2],
                "tones": [6, 7],
                "word_boundaries": [True, False],
            }
        ],
        "corpus": {"ordering": "fixed-source-order", "texts": [
            "こんにちは。", "おはようございます。", "日本語の音声合成を検証します。", "東京で会いましょう！"
        ]},
        "environment": {
            "platform": "Linux-self-test",
            "system": "Linux",
            "python": "3.12",
            "executable": "/usr/bin/python",
            "source_execution": "official Style-Bert-VITS2 Japanese module and shared sequence mapper only",
            "vokra_checkout": {"root": None, "head": None},
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
    valid["fixtures"] = [
        dict(valid["fixtures"][0], text=text, normalized_text=text)
        for text in valid["corpus"]["texts"]
    ]
    validate(valid)
    try:
        json.loads('{"schema":"x","schema":"y"}', object_pairs_hook=_unique)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate key was accepted")
    invalid = dict(valid)
    invalid["source"] = dict(EXPECTED_SOURCE, hf_revision="drift")
    try:
        validate(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("source drift was accepted")
    drifted = dict(valid)
    drifted["tone_offsets"] = {"ZH": 0, "JP": 5, "EN": 8}
    try:
        validate(drifted)
    except ValueError:
        pass
    else:
        raise AssertionError("tone-offset drift was accepted")
    wrong_band = dict(valid)
    wrong_band["fixtures"] = [dict(valid["fixtures"][0], tones=[0, 1])]
    try:
        validate(wrong_band)
    except ValueError:
        pass
    else:
        raise AssertionError("raw JP tone band was accepted as global tone ids")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("sbv2_jp_extra contract self-test: PASS")
        return 0
    if args.contract is None:
        parser.error("--contract or --self-test is required")
    payload = json.loads(args.contract.read_bytes(), object_pairs_hook=_unique)
    validate(payload)
    print(f"sbv2_jp_extra contract: PASS ({args.contract})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
