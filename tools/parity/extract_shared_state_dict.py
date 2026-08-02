"""extract_shared_state_dict — enumerate shard filenames from a sharded HF checkpoint.

Reads ``model.safetensors.index.json`` (HF sharded-safetensors convention:
``{"metadata": ..., "weight_map": {tensor_name: shard_filename, ...}}``) and
prints the unique sorted set of shard filenames referenced by ``weight_map``.

Rationale: HF sharded checkpoints reference their shards indirectly via the
index file. A downloader that fetches only ``model.safetensors`` will silently
miss ``model-00002-of-000NN.safetensors``. A downloader that fetches by glob
``*.safetensors`` will pick them up but may over-fetch (e.g. ``adapter_*.safetensors``
in ``facebook/mms-1b-all``). This helper lets a downloader do the correct thing:
grab the index, parse ``weight_map``, union the shard names into its
allow-set.

Zero third-party deps (Python stdlib only). Not part of the vokra-* runtime
(NFR-DS-02 zero-dep is unaffected). See FR-LD-05.

Also used by resilient_batch.sh (scripts/publish/vast-ai/) to feed the
resilient_download.py driver its per-model shard union.

Usage:
    python extract_shared_state_dict.py <path-to-model.safetensors.index.json>
        -> prints one shard filename per line, sorted, deduped

    python extract_shared_state_dict.py --self-test
        -> exit 0 if internal parse fixtures pass, exit 1 otherwise
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Iterable


def extract_shards(index_json_path: str) -> list[str]:
    """Parse an HF safetensors index.json and return the sorted-unique shard filenames.

    Raises:
        FileNotFoundError: if the index file does not exist.
        ValueError: if the JSON has no ``weight_map`` object, or ``weight_map`` is
                    not a dict of str->str.
    """
    if not os.path.isfile(index_json_path):
        raise FileNotFoundError(f"index file not found: {index_json_path}")
    with open(index_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "weight_map" not in data:
        raise ValueError(f"index.json missing 'weight_map' object: {index_json_path}")
    wm = data["weight_map"]
    if not isinstance(wm, dict):
        raise ValueError(f"'weight_map' must be an object, got {type(wm).__name__}")
    shards: set[str] = set()
    for tensor_name, shard in wm.items():
        if not isinstance(tensor_name, str) or not isinstance(shard, str):
            raise ValueError(
                f"weight_map entry not str->str: {tensor_name!r} -> {shard!r}"
            )
        shards.add(shard)
    return sorted(shards)


def _self_test() -> int:
    cases = 0
    fails = 0

    def _run(label: str, payload: dict | str, expected_shards: list[str] | None,
             should_raise: type | None = None) -> None:
        nonlocal cases, fails
        cases += 1
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.safetensors.index.json")
            if isinstance(payload, dict):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(payload)
            try:
                out = extract_shards(path)
            except Exception as exc:  # noqa: BLE001 — test harness
                if should_raise is not None and isinstance(exc, should_raise):
                    return
                fails += 1
                print(f"self-test FAIL [{label}]: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
                return
            if should_raise is not None:
                fails += 1
                print(f"self-test FAIL [{label}]: expected {should_raise.__name__}, got {out!r}", file=sys.stderr)
                return
            if out != expected_shards:
                fails += 1
                print(f"self-test FAIL [{label}]: got {out!r}, want {expected_shards!r}", file=sys.stderr)

    # Case: two-shard checkpoint (Llama-style).
    _run(
        "two-shard",
        {
            "metadata": {"total_size": 42},
            "weight_map": {
                "embed_tokens.weight": "model-00001-of-00002.safetensors",
                "layers.0.attn.q.weight": "model-00001-of-00002.safetensors",
                "layers.31.attn.q.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        },
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
    )

    # Case: single-shard (weird but valid).
    _run(
        "single-shard",
        {"weight_map": {"w": "model.safetensors"}},
        ["model.safetensors"],
    )

    # Case: many shards — dedup + sort.
    _run(
        "many-shards-dedup-sort",
        {"weight_map": {f"t{i}": f"model-000{(i % 3) + 1:02d}-of-00003.safetensors"
                         for i in range(30)}},
        [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ],
    )

    # Case: missing weight_map — must raise ValueError.
    _run("missing-weight-map", {"metadata": {}}, None, should_raise=ValueError)

    # Case: weight_map not dict — must raise ValueError.
    _run("weight-map-not-dict", {"weight_map": ["a.safetensors"]}, None,
         should_raise=ValueError)

    # Case: entry has non-str shard — must raise ValueError.
    _run("entry-not-str", {"weight_map": {"t": 42}}, None, should_raise=ValueError)

    # Case: malformed JSON.
    _run("malformed-json", "{not-json", None, should_raise=json.JSONDecodeError)

    if fails == 0:
        print(f"extract_shared_state_dict self-test: OK ({cases} cases)")
        return 0
    print(f"extract_shared_state_dict self-test: {fails}/{cases} FAILED", file=sys.stderr)
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("index_json", nargs="?",
                        help="path to model.safetensors.index.json")
    parser.add_argument("--self-test", action="store_true",
                        help="run internal self-test (no HF network I/O)")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.index_json:
        parser.error("index_json path required (or pass --self-test)")

    shards = extract_shards(args.index_json)
    for s in shards:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
