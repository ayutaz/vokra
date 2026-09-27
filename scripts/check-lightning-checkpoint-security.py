#!/usr/bin/env python3
"""Regression check for Lightning's checkpoint ``_instantiator`` guard.

This reproduces the security fix from upstream commit
``d710d689510d50e800f53b3cd773cbca20b1f86f``.  The checkpoint contains no
weights: it only carries an empty state dict and a deliberately untrusted
instantiator path.  The test is run once through each public Lightning import
surface used by the parity environments.
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_VERSION = "2.6.6"
IMPORT_SURFACES = ("lightning.pytorch", "pytorch_lightning")
ALLOWED_INSTANTIATORS = frozenset(
    {
        "lightning.pytorch.cli.instantiate_module",
        "pytorch_lightning.cli.instantiate_module",
    }
)
BLOCKED_MESSAGE = "not in the allowlist of trusted instantiators"
_SENTINEL_CALLED = False


def _sentinel(*_args: Any, **_kwargs: Any) -> Any:
    """Harmless tripwire: vulnerable Lightning would call this function."""

    global _SENTINEL_CALLED
    _SENTINEL_CALLED = True
    return None


def _checkpoint(instantiator: str, version: str) -> dict[str, Any]:
    return {
        "state_dict": {},
        "hyper_parameters": {"_instantiator": instantiator},
        "pytorch-lightning_version": version,
    }


def _run_surface(surface: str) -> None:
    global _SENTINEL_CALLED

    package = importlib.import_module(surface)
    root_package = importlib.import_module(surface.split(".", 1)[0])
    actual_version = getattr(root_package, "__version__", None)
    if actual_version != EXPECTED_VERSION:
        raise AssertionError(
            f"{surface} resolved Lightning {actual_version!r}, expected {EXPECTED_VERSION!r}"
        )

    lightning_module = package.LightningModule

    class ProbeModule(lightning_module):
        pass

    import torch

    _SENTINEL_CALLED = False
    with tempfile.TemporaryDirectory(prefix="vokra-lightning-checkpoint-") as directory:
        checkpoint_path = Path(directory) / "untrusted-instantiator.ckpt"
        torch.save(
            _checkpoint(f"{__name__}._sentinel", actual_version), checkpoint_path
        )
        try:
            ProbeModule.load_from_checkpoint(checkpoint_path, strict=False)
        except ValueError as error:
            if BLOCKED_MESSAGE not in str(error):
                raise AssertionError(
                    f"{surface} rejected the checkpoint for an unexpected reason: {error}"
                ) from error
        except Exception as error:
            raise AssertionError(
                f"{surface} raised {type(error).__name__} instead of the checkpoint allowlist ValueError"
            ) from error
        else:
            raise AssertionError(f"{surface} accepted an untrusted checkpoint instantiator")

    if _SENTINEL_CALLED:
        raise AssertionError(f"{surface} executed the untrusted sentinel instantiator")

    print(f"{surface}: Lightning {actual_version} blocked untrusted _instantiator")


def _self_test() -> None:
    if EXPECTED_VERSION != "2.6.6":
        raise AssertionError("the security checkpoint must stay pinned to Lightning 2.6.6")
    if IMPORT_SURFACES != ("lightning.pytorch", "pytorch_lightning"):
        raise AssertionError("both supported Lightning import surfaces must remain covered")
    if ALLOWED_INSTANTIATORS != frozenset(
        {
            "lightning.pytorch.cli.instantiate_module",
            "pytorch_lightning.cli.instantiate_module",
        }
    ):
        raise AssertionError("the upstream trusted-instantiator contract changed")
    checkpoint = _checkpoint("temporary.sentinel", EXPECTED_VERSION)
    if checkpoint["hyper_parameters"]["_instantiator"] != "temporary.sentinel":
        raise AssertionError("self-test failed to construct the untrusted checkpoint")
    print("check-lightning-checkpoint-security self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    for surface in IMPORT_SURFACES:
        _run_surface(surface)


if __name__ == "__main__":
    main()
