#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed identity/license gate for Higgs Audio v3 TTS 4B.

No dedicated lock is staged: the upstream card identifies a custom Boson
Research-and-Non-Commercial license and the exact source/dependency closure
has not been authenticated.  This module is stdlib-only so the blocker is
checked before any dedicated environment can be synchronized.
"""

from __future__ import annotations

import argparse
import json

MODEL_REPOSITORY = "bosonai/higgs-audio-v3-tts-4b"
MODEL_REVISION: str | None = None
SOURCE_REPOSITORY = "https://github.com/boson-ai/higgs-audio"
SOURCE_REVISION: str | None = None
LICENSE_REFERENCE = "LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial"
STATUS = "BLOCKED_CUSTOM_LICENSE_AND_UNAUTHENTICATED_CLOSURE"
BLOCKERS = [
    "the upstream weight/code license is Boson Higgs TTS 3 Research-and-Non-Commercial and forbids redistribution by default",
    "the exact immutable HF and source revisions are not recorded, so the source import closure cannot be authenticated",
    "the SGLang/custom codec execution closure is not represented by an exact Python 3.12 lock",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-gate", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    assert MODEL_REPOSITORY and SOURCE_REPOSITORY and LICENSE_REFERENCE
    assert MODEL_REVISION is None and SOURCE_REVISION is None
    assert STATUS.startswith("BLOCKED_")
    if args.self_test:
        print("higgs_audio_v3_tts_4b inspector self-test: PASS")
        return 0
    if not args.dependency_gate:
        parser.error("use --dependency-gate or --self-test")
    print(json.dumps({"status": STATUS, "publication": "NO_UPLOAD", "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION, "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "license": LICENSE_REFERENCE, "blockers": BLOCKERS}, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
