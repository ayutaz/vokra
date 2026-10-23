#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stdlib-only metadata and receipt gates for the hosted RC1 pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PINS = {
    "whisper-base": ("bf440f6064ffc882d53d3694729dc4616dc2cfa7", "5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014", 290859488),
    "silero-vad-v5": ("3670dfed054980307eeb1711c6329c29b80bdf14", "23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6", 2183744),
}
MODEL_IDS = {
    "whisper-base": ("vokra/whisper-base", "whisper-base.gguf"),
    "silero-vad-v5": ("vokra/silero-vad-v5", "silero-vad-v5.gguf"),
}
CLI_SHA = "2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4"
CLI_BYTES = 13344272
SIDECAR_SHA = "093cdf4bad82a2ebcd2d3ac7ed8da6cbba0651e9991e9c969c86cfd1caf72e98"
SIDECAR_BYTES = 100
MANIFEST_SHA = "32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92"
MANIFEST_BYTES = 2453
FIXTURE_SHA = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
RELEASE_ID = 416018534
RELEASE_TAG = "v0.4.0-rc.1"
RELEASE_SOURCE = "d84d311e98d0abb2987b5c4e96157ae8f2068e1a"
DRIVER_BLOB_OID = "4107f83b64fe59d6a121d051271c9b41e763649b"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pins(path: Path) -> dict[str, Any]:
    pins = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pins, dict) or set(pins) != set(PINS):
        raise SystemExit("pin model set mismatch")
    for name, (revision, sha, size) in PINS.items():
        row = pins.get(name)
        repo, filename = MODEL_IDS[name]
        if not isinstance(row, dict) or row != {"repo": repo, "revision": revision, "filename": filename, "sha256": sha, "size_bytes": size}:
            raise SystemExit(f"pin identity mismatch: {name}")
    return pins


def validate_inputs(args: argparse.Namespace) -> None:
    release = json.loads(args.release.read_text(encoding="utf-8"))
    tag = json.loads(args.tag.read_text(encoding="utf-8"))
    if release.get("id") != RELEASE_ID or release.get("tag_name") != RELEASE_TAG or release.get("draft") is not False or release.get("prerelease") is not True:
        raise SystemExit("release identity mismatch")
    if tag.get("object", {}).get("type") != "commit" or tag.get("object", {}).get("sha") != RELEASE_SOURCE:
        raise SystemExit("tag source mismatch")
    assets = {row.get("name"): row for row in release.get("assets", []) if isinstance(row, dict)}
    if assets.get(args.cli_name, {}).get("size") != CLI_BYTES or assets.get(f"{args.cli_name}.sha256", {}).get("size") != SIDECAR_BYTES or assets.get("vokra-desktop-manifest.json", {}).get("size") != MANIFEST_BYTES:
        raise SystemExit("release asset size mismatch")
    if args.cli.stat().st_size != CLI_BYTES or digest(args.cli) != CLI_SHA:
        raise SystemExit("CLI identity mismatch")
    sidecar_text = args.sidecar.read_text(encoding="utf-8")
    if args.sidecar.stat().st_size != SIDECAR_BYTES or digest(args.sidecar) != SIDECAR_SHA or not (f"{CLI_SHA}  {args.cli_name}" in sidecar_text or f"{CLI_SHA} *{args.cli_name}" in sidecar_text):
        raise SystemExit("CLI sidecar mismatch")
    if args.manifest.stat().st_size != MANIFEST_BYTES or digest(args.manifest) != MANIFEST_SHA:
        raise SystemExit("payload manifest mismatch")
    if digest(args.audio) != FIXTURE_SHA:
        raise SystemExit("fixture mismatch")
    load_pins(args.pins)
    print(json.dumps({"status": "INPUTS_PASS", "head": args.head}, separators=(",", ":")))


def receipt_facts(args: argparse.Namespace) -> dict[str, Any]:
    return {"workflow_sha": args.workflow_sha, "run_id": args.run_id, "attempt": args.attempt, "driver_blob_oid": args.driver_blob_oid or DRIVER_BLOB_OID}


def input_facts() -> dict[str, Any]:
    return {"release_id": RELEASE_ID, "release_tag": RELEASE_TAG, "release_source": RELEASE_SOURCE, "cli_sha256": CLI_SHA, "cli_bytes": CLI_BYTES, "sidecar_sha256": SIDECAR_SHA, "sidecar_bytes": SIDECAR_BYTES, "manifest_sha256": MANIFEST_SHA, "manifest_bytes": MANIFEST_BYTES, "fixture_sha256": FIXTURE_SHA}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise SystemExit("evidence exceeds 64KiB bound")
    path.write_text(encoded, encoding="utf-8")


def write_blocked(args: argparse.Namespace) -> None:
    write_json(args.output, {"status": "BLOCKED", "reason": args.blocked_reason, "input_verification": input_facts(), **receipt_facts(args)})


def write_failure(args: argparse.Namespace) -> None:
    receipt_raw = args.receipt.read_bytes() if args.receipt.is_file() else b""
    try:
        driver_receipt: Any = json.loads(receipt_raw.decode("utf-8")) if receipt_raw else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        driver_receipt = {"raw_tail": receipt_raw[-65536:].decode("utf-8", "replace")} if receipt_raw else None
    log_tail = args.driver_log.read_bytes()[-8192:].decode("utf-8", "replace") if args.driver_log.is_file() else ""
    payload = {"status": "DRIVER_FAILED", "driver_exit_code": args.driver_exit_code, "driver_log_tail": log_tail, "driver_receipt": driver_receipt, "driver_receipt_bytes": len(receipt_raw), "driver_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest() if receipt_raw else None, "input_verification": input_facts(), **receipt_facts(args)}
    write_json(args.output, payload)


def validate_receipt(args: argparse.Namespace) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise SystemExit("driver receipt status failed")
    if receipt.get("schema") != 1:
        raise SystemExit("driver receipt schema failed")
    commands = receipt.get("commands")
    labels = [row.get("label") for row in commands] if isinstance(commands, list) and all(isinstance(row, dict) for row in commands) else []
    if not isinstance(commands, list) or len(commands) != 3 or len(set(labels)) != 3 or set(labels) != {"asr-jfk", "vad-jfk", "vad-silence"} or any(not isinstance(row, dict) or row.get("returncode") != 0 or not isinstance(row.get("argv"), list) or "--backend" not in row["argv"] or row["argv"][row["argv"].index("--backend") + 1:row["argv"].index("--backend") + 2] != ["cpu"] for row in commands):
        raise SystemExit("driver command gates failed")
    semantic = receipt.get("semantic_checks", {})
    asr_check = semantic.get("asr_jfk_keyphrase") if isinstance(semantic, dict) else None
    vad_check = semantic.get("vad") if isinstance(semantic, dict) else None
    if not isinstance(asr_check, dict) or not isinstance(vad_check, dict) or asr_check.get("passed") is not True or vad_check.get("passed") is not True:
        raise SystemExit("driver semantic gates failed")
    runtime = receipt.get("runtime", {})
    if not isinstance(runtime, dict) or runtime.get("head") != args.head or runtime.get("expected_head") != args.head or runtime.get("platform") != "linux" or runtime.get("github_actions") != "true" or runtime.get("marker") != "1":
        raise SystemExit("driver runtime identity failed")
    pins = load_pins(args.pins)
    reviewed = {name: {**row, "url": f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['filename']}"} for name, row in pins.items()}
    if receipt.get("reviewed_pins") != reviewed:
        raise SystemExit("driver reviewed pins mismatch")
    models = receipt.get("models")
    if not isinstance(models, list) or len(models) != 2 or {row.get("name") for row in models if isinstance(row, dict)} != set(pins):
        raise SystemExit("driver model records incomplete")
    for model in models:
        expected = pins.get(model.get("name")) if isinstance(model, dict) else None
        expected_url = f"https://huggingface.co/{expected['repo']}/resolve/{expected['revision']}/{expected['filename']}" if isinstance(expected, dict) else ""
        if not isinstance(expected, dict) or any(model.get(key) != expected.get(key) for key in ("repo", "revision", "filename")) or model.get("actual_sha256") != expected["sha256"] or model.get("expected_sha256") != expected["sha256"] or model.get("actual_bytes") != expected["size_bytes"] or model.get("expected_bytes") != expected["size_bytes"] or model.get("url") != expected_url:
            raise SystemExit("driver model record mismatch")
    raw = args.receipt.read_bytes()
    write_json(args.output, {"status": "PASS", "driver_exit_code": 0, "driver_receipt_sha256": hashlib.sha256(raw).hexdigest(), "driver_receipt_bytes": len(raw), "source_head": args.head, "driver_blob_oid": args.driver_blob_oid or DRIVER_BLOB_OID, "input_verification": input_facts(), "driver_receipt": receipt, **receipt_facts(args)})
    print(args.output.read_text(encoding="utf-8"), end="")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--tag", type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--cli-name", default="vokra-cli-0.4.0-rc.1-x86_64-linux")
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--head", default="")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--driver-log", type=Path)
    parser.add_argument("--driver-exit-code", type=int)
    parser.add_argument("--driver-blob-oid", default="")
    parser.add_argument("--workflow-sha", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--attempt", default="")
    parser.add_argument("--blocked-reason", default="")
    args = parser.parse_args()
    if args.metadata_only:
        load_pins(args.pins)
        print(json.dumps({"status": "METADATA_PASS", "models": sorted(PINS)}, separators=(",", ":")))
        return 0
    if args.blocked_reason:
        write_blocked(args)
        return 2
    if args.release and args.tag and args.cli and args.sidecar and args.manifest and args.audio:
        validate_inputs(args)
        return 0
    if args.receipt and args.output and args.driver_exit_code is not None and args.driver_log:
        write_failure(args)
        return args.driver_exit_code or 1
    if args.receipt and args.output:
        validate_receipt(args)
        return 0
    parser.error("input or receipt validation arguments are required")


if __name__ == "__main__":
    raise SystemExit(main())
