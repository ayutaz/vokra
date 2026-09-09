#!/usr/bin/env -S uv run --script
"""Official AudioGen reference entry point; disabled until lock/evidence exists."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

LOCK = Path(__file__).with_name("audiogen_medium_reference") / "uv.lock"
REQUIRED_ARTIFACTS = {
    "t5_token_ids", "t5_hidden", "conditional_projection", "null_projection",
    "raw_lm", "delay_codes", "codec_latent", "pcm_16khz",
}


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def validate_packet(packet: dict[str, object]) -> None:
    if not isinstance(packet, dict) or set(packet) != {"schema", "execution", "status", "artifacts"}:
        raise ValueError("reference packet root schema is not exact")
    if packet.get("schema") != "vokra.audiogen_medium.reference.v2":
        raise ValueError("wrong reference schema")
    artifacts = packet.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(REQUIRED_ARTIFACTS) or any(not isinstance(item, dict) or set(item) != {"role", "path", "shape", "dtype", "sha256", "finite"} for item in artifacts) or {item["role"] for item in artifacts} != REQUIRED_ARTIFACTS:
        raise ValueError("reference artifact roles are incomplete")
    seen_paths: set[str] = set()
    for item in artifacts:
        role, path, shape, dtype, sha256, finite = (item["role"], item["path"], item["shape"], item["dtype"], item["sha256"], item["finite"])
        if not isinstance(role, str) or role not in REQUIRED_ARTIFACTS or not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or ".." in Path(path).parts or path in seen_paths:
            raise ValueError("reference artifact path/role is unsafe or duplicated")
        seen_paths.add(path)
        if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
            raise ValueError(f"invalid reference shape: {role}")
        if not isinstance(dtype, str) or not dtype or not re.fullmatch(r"[A-Za-z0-9_]+", dtype):
            raise ValueError(f"invalid reference dtype: {role}")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"invalid reference SHA-256: {role}")
        if finite not in {"TRUE", "NOT_CHECKED"}:
            raise ValueError(f"invalid finite marker: {role}")
    if packet.get("execution") != "official_audiocraft_only" or packet.get("status") != "BLOCKED":
        raise ValueError("reference execution/status contract is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.self_test:
        try:
            validate_packet({"schema": "wrong", "artifacts": []})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reference packet was accepted")
        valid = {"schema": "vokra.audiogen_medium.reference.v2", "execution": "official_audiocraft_only", "status": "BLOCKED", "artifacts": [{"role": role, "path": role, "shape": [1], "dtype": "F32", "sha256": "0" * 64, "finite": "NOT_CHECKED"} for role in sorted(REQUIRED_ARTIFACTS)]}
        validate_packet(valid)
        try:
            validate_packet({**valid, "artifacts": valid["artifacts"] + [valid["artifacts"][0]]})
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate reference role was accepted")
        try:
            json.loads('{"schema":1,"schema":2}', object_pairs_hook=strict_pairs)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate JSON key was accepted")
        print("audiogen_medium_dump_reference --self-test: OK")
        return 0
    if not LOCK.is_file():
        print("audiogen reference BLOCKED: dedicated uv.lock absent; no download/inference", file=sys.stderr)
        return 2
    if args.evidence is None:
        print("audiogen reference BLOCKED: --evidence is required", file=sys.stderr)
        return 2
    try:
        validate_packet(json.loads(args.evidence.read_text(), object_pairs_hook=strict_pairs))
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        print(f"audiogen reference BLOCKED: invalid evidence packet: {exc}", file=sys.stderr)
        return 2
    print("audiogen reference BLOCKED: official execution is not authorized until source/companion evidence is signed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
