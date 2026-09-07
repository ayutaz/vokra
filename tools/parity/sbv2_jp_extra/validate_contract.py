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
from pathlib import Path


EXPECTED_SOURCE = {
    "hf_revision": "a731761009f3c96d104487be6ad332bf1bb5a3a5",
    "source_commit": "ef93f388fc1ddf0dc0f598126c1964923f1df94f",
    "symbols_blob": "846de64584e9ba4b8d96aab36d4efbcefb1a11e7",
    "japanese_blob": "5c055875626c16bd7d3489d02b4952ec90a3bbf6",
    "english_blob": "4a2af9523f2f96b7b34a0fff7589a82e1122ecae",
    "init_blob": "495e57b50d87a4ca3e8fe8dbaf003b4888581927",
    "license_blob": "0ad25db4bd1d86c452db3f9602ccdbe172438f52",
}
EXPECTED_KEYS = {
    "schema",
    "generator",
    "source",
    "model",
    "phoneme_symbols",
    "sorted_phoneme_symbols",
    "language_ids",
    "tone_counts",
    "tone_offsets",
    "fixtures",
}


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

    model = _object(root["model"], "model", {"hf_repo", "hf_revision", "n_vocab", "n_tones"})
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
    if n_vocab != 178 or n_tones != 12:
        raise ValueError("model vocabulary/tone dimensions do not match the authenticated checkpoint")

    fixtures = root["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixtures must be a non-empty list")
    for index, fixture in enumerate(fixtures):
        row = _object(
            fixture,
            f"fixtures[{index}]",
            {"language", "text", "normalized_text", "phoneme_ids", "tones", "word_boundaries"},
        )
        if not isinstance(row["language"], str) or row["language"] != "JP":
            raise ValueError(f"fixtures[{index}] names an unknown language")
        if not isinstance(row["text"], str) or not row["text"]:
            raise ValueError(f"fixtures[{index}].text is empty")
        if not isinstance(row["normalized_text"], str) or not row["normalized_text"]:
            raise ValueError(f"fixtures[{index}].normalized_text is empty")
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


def self_test() -> None:
    symbols = ["_", "a", "i"]
    valid = {
        "schema": "vokra-sbv2-jp-extra-g2p-v1",
        "generator": "vast-worker-test",
        "source": dict(EXPECTED_SOURCE),
        "model": {
            "hf_repo": "litagin/Style-Bert-VITS2-2.0-base-JP-Extra",
            "hf_revision": EXPECTED_SOURCE["hf_revision"],
            "n_vocab": 178,
            "n_tones": 12,
        },
        "phoneme_symbols": symbols + [f"symbol-{index}" for index in range(175)],
        "sorted_phoneme_symbols": sorted(symbols + [f"symbol-{index}" for index in range(175)]),
        "language_ids": {"ZH": 0, "JP": 1, "EN": 2},
        "tone_counts": {"ZH": 6, "JP": 2, "EN": 4},
        "tone_offsets": {"ZH": 0, "JP": 6, "EN": 8},
        "fixtures": [
            {
                "language": "JP",
                "text": "test",
                "normalized_text": "test",
                "phoneme_ids": [1, 2],
                "tones": [6, 7],
                "word_boundaries": [True, False],
            }
        ],
    }
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
