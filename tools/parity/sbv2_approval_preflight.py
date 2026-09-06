#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Fail-closed SBV2 JP-Extra execution-approval validator.

This is deliberately stdlib-only.  It authenticates the external owner
approval before a worker acquires or executes any model.  The approval is
bound to the exact source revisions, expected checkout HEAD, Japanese packet
scope, and no-upload policy; the checked-in audit row remains a separate
license gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

EXPECTED = {
    "schema": "vokra-sbv2-approval-v1",
    "sbv2_repo": "litagin/Style-Bert-VITS2-2.0-base-JP-Extra",
    "sbv2_revision": "a731761009f3c96d104487be6ad332bf1bb5a3a5",
    "bert_ja_repo": "ku-nlp/deberta-v2-large-japanese-char-wwm",
    "bert_ja_revision": "547b0e8b044fba3f9b84d0ab9f990440bd130c8b",
    "bert_en_repo": "microsoft/deberta-v3-large",
    "bert_en_revision": "64a8c8eab3e352a784c658aef62be1662607476f",
    "bert_zh_repo": "hfl/chinese-roberta-wwm-ext-large",
    "bert_zh_revision": "a25cc9e05974bd9687e528edd516f2cfdb3f5db9",
    "language": "ja",
    "no_upload": True,
}
KEYS = set(EXPECTED) | {"expected_head", "signer", "scope_sha256"}
MANIFEST_KEYS = {
    "generator_version",
    "generator",
    "checkpoint",
    "request",
    "phonemize_fixture",
    "tensors",
    "flow_layers",
}
PACKET_ROOT_FILES = {
    "reference_dump.manifest.json",
    "sbv2-v2-multilingual-base.gguf",
    "sbv2-v2-multilingual-base.gguf.sha256",
    "deberta-v2-large-japanese-char-wwm.gguf",
    "deberta-v2-large-japanese-char-wwm.gguf.sha256",
    "deberta-v3-large.gguf",
    "deberta-v3-large.gguf.sha256",
    "chinese-roberta-wwm-ext-large.gguf",
    "chinese-roberta-wwm-ext-large.gguf.sha256",
}


def parse_bytes(raw: bytes) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate approval key: {key}")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError("approval top-level schema is not exact")
    return value


def validate(path: Path, expected_head: str) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("approval must be an absolute regular non-symlink file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    approval = parse_bytes(raw)
    if not isinstance(expected_head, str) or len(expected_head) != 40 or any(
        char not in "0123456789abcdef" for char in expected_head
    ):
        raise ValueError("expected HEAD must be lowercase 40-hex")
    for key, expected in EXPECTED.items():
        if approval[key] != expected:
            raise ValueError(f"approval identity drift: {key}")
    if approval["expected_head"] != expected_head:
        raise ValueError("approval expected_head does not match checkout")
    signer = approval["signer"]
    if not isinstance(signer, str) or not signer.strip() or signer.strip().upper() in {
        "TODO",
        "TBD",
        "UNKNOWN",
        "UNRESOLVED",
        "PENDING",
    }:
        raise ValueError("approval signer is unresolved")
    payload = {key: approval[key] for key in KEYS if key != "scope_sha256"}
    scope = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if approval["scope_sha256"] != scope:
        raise ValueError("approval scope_sha256 mismatch")
    return digest


def validate_packet(directory: Path) -> None:
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("packet must be an absolute regular non-symlink directory")
    manifest_path = directory / "reference_dump.manifest.json"
    manifest = json.loads(manifest_path.read_bytes(), object_pairs_hook=lambda pairs: _unique(pairs))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("reference manifest top-level schema is not exact")
    checkpoint = manifest.get("checkpoint")
    expected_checkpoint = {
        "sbv2_main": "sbv2-v2-multilingual-base.gguf",
        "bert_ja": "deberta-v2-large-japanese-char-wwm.gguf",
        "bert_en": "deberta-v3-large.gguf",
    }
    if checkpoint != expected_checkpoint:
        raise ValueError("reference checkpoint identity is not the authenticated JA bundle")
    request = manifest.get("request")
    if not isinstance(request, dict) or request.get("language") != "JA":
        raise ValueError("reference packet is not a Japanese packet")
    nested: set[str] = set()
    for block_name in ("phonemize_fixture",):
        block = manifest[block_name]
        if not isinstance(block, dict):
            raise ValueError(f"manifest {block_name} is not an object")
        for row in block.values():
            path = _manifest_path(row)
            if path in nested:
                raise ValueError(f"duplicate manifest artifact path: {path}")
            nested.add(path)
    tensors = manifest["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("manifest tensors must be a non-empty array")
    for row in tensors:
        path = _manifest_path(row)
        if path in nested:
            raise ValueError(f"duplicate manifest artifact path: {path}")
        nested.add(path)
    flow_layers = manifest["flow_layers"]
    if not isinstance(flow_layers, dict) or not flow_layers:
        raise ValueError("manifest flow_layers must be a non-empty object")
    for row in flow_layers.values():
        path = _manifest_path(row)
        if path in nested:
            raise ValueError(f"duplicate manifest artifact path: {path}")
        nested.add(path)
    expected = set(PACKET_ROOT_FILES) | nested
    actual: set[str] = set()
    for path in directory.rglob("*"):
        rel = path.relative_to(directory).as_posix()
        if path.is_dir() and rel == "reference_dump":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"packet contains non-regular or symlink entry: {rel}")
        actual.add(rel)
    if actual != expected:
        raise ValueError(f"packet closure mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _manifest_path(row: object) -> str:
    if not isinstance(row, dict) or set(row) - {"name", "path", "count", "dtype", "shape", "atol"} != set():
        raise ValueError("manifest artifact row schema is not exact")
    path = row.get("path")
    if not isinstance(path, str) or not path.startswith("reference_dump/") or "/../" in path:
        raise ValueError("manifest artifact path is unsafe")
    return path


def self_test() -> int:
    assert EXPECTED["bert_zh_revision"] == "a25cc9e05974bd9687e528edd516f2cfdb3f5db9"
    head = "a" * 40
    payload = dict(EXPECTED, expected_head=head, signer="owner", scope_sha256="")
    scope_payload = {key: payload[key] for key in KEYS if key != "scope_sha256"}
    payload["scope_sha256"] = hashlib.sha256(
        json.dumps(scope_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="sbv2-approval-") as directory:
        path = Path(directory) / "approval.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        validate(path, head)
        duplicate = path.with_name("duplicate.json")
        duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        try:
            parse_bytes(duplicate.read_bytes())
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate approval key was accepted")
        for field, value in (("expected_head", "b" * 40), ("bert_zh_revision", "bad"), ("scope_sha256", "0" * 64), ("signer", "TODO")):
            tampered = dict(payload, **{field: value})
            if field != "scope_sha256":
                tampered["scope_sha256"] = hashlib.sha256(json.dumps({key: tampered[key] for key in KEYS if key != "scope_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            tampered_path = path.with_name(f"tampered-{field}.json")
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            try:
                validate(tampered_path, head)
            except (OSError, ValueError):
                pass
            else:
                raise AssertionError(f"tampered approval accepted: {field}")
        relative = Path("relative-approval.json")
        try:
            validate(relative, head)
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError("relative approval path accepted")
        symlink = path.with_name("approval-link.json")
        try:
            symlink.symlink_to(path)
        except OSError:
            pass
        else:
            try:
                validate(symlink, head)
            except ValueError:
                pass
            else:
                raise AssertionError("symlink approval accepted")
        packet = Path(directory) / "packet"
        (packet / "reference_dump").mkdir(parents=True)
        manifest = {
            "generator_version": "1.1",
            "generator": "tools/parity/sbv2_dump_reference.py",
            "checkpoint": {"sbv2_main": "sbv2-v2-multilingual-base.gguf", "bert_ja": "deberta-v2-large-japanese-char-wwm.gguf", "bert_en": "deberta-v3-large.gguf"},
            "request": {"language": "JA"},
            "phonemize_fixture": {"phoneme_ids": {"path": "reference_dump/phoneme_ids.bin"}},
            "tensors": [{"name": "waveform", "path": "reference_dump/waveform.bin"}],
            "flow_layers": {"flow_layer_0_output": {"path": "reference_dump/flow_layer_0_output.bin"}},
        }
        (packet / "reference_dump.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name in PACKET_ROOT_FILES - {"reference_dump.manifest.json"}:
            (packet / name).write_bytes(b"fixture")
        for name in ("phoneme_ids.bin", "waveform.bin", "flow_layer_0_output.bin"):
            (packet / "reference_dump" / name).write_bytes(b"fixture")
        validate_packet(packet)
        (packet / "reference_dump" / "waveform.bin").unlink()
        try:
            validate_packet(packet)
        except ValueError:
            pass
        else:
            raise AssertionError("missing packet artifact accepted")
        (packet / "reference_dump" / "waveform.bin").write_bytes(b"fixture")
        (packet / "reference_dump" / "extra.bin").write_bytes(b"fixture")
        try:
            validate_packet(packet)
        except ValueError:
            pass
        else:
            raise AssertionError("extra packet artifact accepted")
        (packet / "reference_dump" / "extra.bin").unlink()
        link = packet / "reference_dump" / "waveform-link.bin"
        try:
            link.symlink_to(packet / "reference_dump" / "waveform.bin")
        except OSError:
            pass
        else:
            try:
                validate_packet(packet)
            except ValueError:
                pass
            else:
                raise AssertionError("symlink packet artifact accepted")
    print("sbv2_approval_preflight self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval")
    parser.add_argument("--expected-head")
    parser.add_argument("--packet")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.approval or args.expected_head or args.packet:
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if args.packet:
        if args.approval or args.expected_head:
            parser.error("--packet cannot be combined with approval arguments")
        try:
            validate_packet(Path(args.packet))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.exit(2, f"packet gate BLOCKED: {error}\n")
        return 0
    if not args.approval or not args.expected_head:
        parser.error("--approval and --expected-head are required")
    try:
        print(validate(Path(args.approval), args.expected_head))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"approval gate BLOCKED: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
