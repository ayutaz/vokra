#!/usr/bin/env python3
"""Dependency-free CLAP reference contract gate.

The gate authenticates only the local reference contract.  It deliberately
does not fetch a model, import torch/Transformers, or turn inspection into a
parity result.  The actual source/state-dict evidence remains a disposable
VAST artifact until reviewed.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
REPOSITORY = "https://huggingface.co/laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
ENTRYPOINT = "tools/parity/clap_dump_reference.py"
DEPENDENCIES = ["numpy==2.3.5", "torch==2.7.1", "transformers==5.10.4"]
TRANSFORMERS_WHEEL_SHA256 = (
    "8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e"
)


def self_test() -> None:
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    values = project["tool"]["vokra"]["clap_reference"]
    assert values["source_repository"] == REPOSITORY
    assert values["source_revision"] == REVISION
    assert values["reference_entrypoint"] == ENTRYPOINT
    assert values["isolated_transformers_pin"] == "5.10.4"
    assert values["transformers_wheel_sha256"] == TRANSFORMERS_WHEEL_SHA256
    assert values["license_status"] == "OWNER_REVIEW_PENDING"
    assert values["dependency_audit_status"] == "PENDING_VAST_AUDIT"
    assert values["source_contract_status"] == "PENDING_VAST_WHEEL_BINDING"
    assert values["publication"] == "NO_UPLOAD"
    assert project["project"]["dependencies"] == DEPENDENCIES
    assert project["tool"]["uv"]["environments"] == [
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    ]
    assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    assert project["tool"]["uv"]["index"][0] == {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
        "explicit": True,
    }
    assert not (PROJECT / "models").exists(), "model material must not be checked in"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is available; model/license acquisition is gated")
    self_test()
    print("clap license/reference gate self-test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
