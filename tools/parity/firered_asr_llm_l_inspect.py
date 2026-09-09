#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed identity gate for the FireRed ASR LLM-L sidecar.

The current HF release is a ``model.pth.tar`` bundle, while the existing
preparer accepts an authenticated sharded-safetensors index.  The source
checkout revision needed to interpret that tarball is not pinned in the
repository, so this inspector intentionally emits a blocker and never
imports torch or touches a model artifact.
"""

from __future__ import annotations

import argparse
import json

MODEL_REPOSITORY = "FireRedTeam/FireRedASR-LLM-L"
MODEL_REVISION = "9837461f78d15ee66565d00aaec0bc5497d7fbc1"
SOURCE_REPOSITORY = "https://github.com/FireRedTeam/FireRedASR"
SOURCE_REVISION: str | None = None
STATUS = "BLOCKED_SOURCE_FORMAT_AND_AUTHENTICATION"
BLOCKERS = [
    "the pinned HF release is model.pth.tar, not the sharded safetensors index required by prepare_checkpoint.py",
    "FireRedTeam/FireRedASR source revision for the tar extraction/config contract is not pinned",
    "the composite Qwen2-7B-Instruct license inheritance remains owner-authenticated evidence, not a resolved dependency gate",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    assert MODEL_REPOSITORY and MODEL_REVISION and SOURCE_REPOSITORY
    assert SOURCE_REVISION is None
    assert STATUS.startswith("BLOCKED_")
    if args.self_test:
        print("firered_asr_llm_l inspector self-test: PASS")
        return 0
    if not args.dependency_gate:
        parser.error("use --dependency-gate or --self-test")
    print(json.dumps({"status": STATUS, "publication": "NO_UPLOAD", "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION, "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "blockers": BLOCKERS}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
