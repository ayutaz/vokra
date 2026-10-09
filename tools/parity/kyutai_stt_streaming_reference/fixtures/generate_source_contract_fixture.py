#!/usr/bin/env -S uv run --no-project --no-sync --python 3.12 python -S
"""Emit the actual source-contract helper return from retained source bytes.

This generator is intentionally source-only.  It imports the existing decoder
helper, parses the retained Python files with ``ast`` through that helper, and
prints JSON; it never imports Torch/Moshi or executes source code.  The
retained packet has no config TOML bytes, so the Git identity callback injects
only the root-reviewed config metadata (URL, revision, size, SHA-256 and blob
SHA-1).  That boundary is not a clean-checkout or numerical-evidence claim.
"""

from __future__ import annotations

import importlib.util
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
PACKET = Path("/private/tmp/vokra-recovered-primary-source-op22vh18/packet")
HELPER = REPO / "tools/parity/kyutai_stt_decoder_dump_reference.py"
CONTRACT = REPO / "tools/parity/kyutai_stt_streaming_reference/contract.py"
DSM_REPOSITORY = "https://github.com/kyutai-labs/delayed-streams-modeling.git"
DSM_REVISION = "4c4f65e147df056adf3346290d64c7b9649b18c9"
MOSHI_REPOSITORY = "https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
DSM_ROLES = ("configs/config-stt-en-hf.toml", "scripts/stt_from_file_pytorch.py")
MOSHI_ROLES = (
    "moshi/moshi/models/lm.py",
    "moshi/moshi/models/lm_utils.py",
    "moshi/moshi/models/loaders.py",
    "moshi/moshi/utils/sampling.py",
    "moshi/moshi/modules/transformer.py",
)
ROLE_PACKET_FILES = {
    DSM_ROLES[1]: ("dsm-python-0002.py", 8452, "2ac2d9bff71d3d6a874bed070eb9d4e60736209e3cbaa9697fbe735dc79d2955"),
    MOSHI_ROLES[0]: ("moshi-python-0010.py", 37312, "38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f"),
    MOSHI_ROLES[1]: ("moshi-python-0011.py", 5089, "a7d5f1347769d0ffaf2964c97fe25310e35bedb93bd087905c4c1934a3a30dab"),
    MOSHI_ROLES[2]: ("moshi-python-0012.py", 18571, "3043f3be59112d2cb897a9261c2abf4628e802adea98c9d3ad16a3e899732e81"),
    MOSHI_ROLES[3]: ("moshi-python-0036.py", 4699, "113da9542ecdcc521945a337a63bec28abba7fd61c2eacaec5855fc7d7562126"),
    MOSHI_ROLES[4]: ("moshi-python-0024.py", 38159, "f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622"),
}


def load_helper():
    spec = importlib.util.spec_from_file_location("kyutai_decoder_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract():
    spec = importlib.util.spec_from_file_location("kyutai_streaming_contract", CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load contract: {CONTRACT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(repository: str, revision: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {"repository": repository, "revision": revision, "roles": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    packet = args.packet
    helper = load_helper()
    contract = load_contract()
    pcm_receipt = contract.require_pcm_source_packet(packet)
    if pcm_receipt["manifest_sha256"] != contract.PCM_SOURCE_PACKET_MANIFEST_SHA256:
        raise RuntimeError("authenticated PCM manifest SHA does not match pinned contract")
    dsm_rows = [
        {
            "path": DSM_ROLES[0],
            "bytes": 1016,
            "sha256": "81f77d642689e1acb276089f62064dab2e71a5532fcac2d3c12563cf4946552c",
            "git_blob_sha1": "75382f8dabeb31832f28c8754afaaf63aaa7b158",
        },
        {
            "path": DSM_ROLES[1],
            "bytes": 8452,
            "sha256": "2ac2d9bff71d3d6a874bed070eb9d4e60736209e3cbaa9697fbe735dc79d2955",
            "git_blob_sha1": "cf3fb05b0e0c1f265a667276d2886ce2664d79ff",
        },
    ]
    moshi_rows = [
        {
            "path": role,
            "bytes": bytes_,
            "sha256": sha256,
            "git_blob_sha1": blob,
        }
        for role, bytes_, sha256, blob in [
            (MOSHI_ROLES[0], 37312, "38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f", "209b7a59c9c086810a81a7d8b99c5233a0ad87ff"),
            (MOSHI_ROLES[1], 5089, "a7d5f1347769d0ffaf2964c97fe25310e35bedb93bd087905c4c1934a3a30dab", "7397067c4d5eb6cb6eb06bdf0f94bdf5a6985444"),
            (MOSHI_ROLES[2], 18571, "3043f3be59112d2cb897a9261c2abf4628e802adea98c9d3ad16a3e899732e81", "fd0e56a571e1a7f0de9d53827503a0669c1787f2"),
            (MOSHI_ROLES[3], 4699, "113da9542ecdcc521945a337a63bec28abba7fd61c2eacaec5855fc7d7562126", "2a35f6cdff1981bb6914b27410c6b152be2a77b1"),
            (MOSHI_ROLES[4], 38159, "f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622", "244e73a3bc012256325e5b759cd13588a96d2673"),
        ]
    ]
    identities = {
        DSM_REPOSITORY: identity(DSM_REPOSITORY, DSM_REVISION, dsm_rows),
        MOSHI_REPOSITORY: identity(MOSHI_REPOSITORY, MOSHI_REVISION, moshi_rows),
    }

    for role, (packet_name, expected_bytes, expected_sha) in ROLE_PACKET_FILES.items():
        body = (packet / packet_name).read_bytes()
        if len(body) != expected_bytes or hashlib.sha256(body).hexdigest() != expected_sha:
            raise RuntimeError(f"consumed role bytes/digest mismatch: {role}")

    with tempfile.TemporaryDirectory(prefix="vokra-source-contract-") as work:
        dsm_root = Path(work) / "dsm"
        moshi_root = Path(work) / "moshi"
        for root, roles in ((dsm_root, DSM_ROLES), (moshi_root, MOSHI_ROLES)):
            for role in roles:
                (root / role).parent.mkdir(parents=True, exist_ok=True)
        # The DSM config is identity-only: the authenticated helper never AST
        # parses it. Its reviewed identity is injected above; no fake bytes are
        # created for a file absent from the retained packet.
        shutil.copyfile(packet / "dsm-python-0002.py", dsm_root / DSM_ROLES[1])
        for role, packet_name in {
            MOSHI_ROLES[0]: "moshi-python-0010.py",
            MOSHI_ROLES[1]: "moshi-python-0011.py",
            MOSHI_ROLES[2]: "moshi-python-0012.py",
            MOSHI_ROLES[3]: "moshi-python-0036.py",
            MOSHI_ROLES[4]: "moshi-python-0024.py",
        }.items():
            shutil.copyfile(packet / packet_name, moshi_root / role)

        original = helper.git_identity
        helper.git_identity = lambda root, repository, revision, roles: identities[repository]
        try:
            result = helper.authenticate_streaming_source_contract(dsm_root, moshi_root)
        finally:
            helper.git_identity = original
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
