"""Pure checks for the authenticated voice-gender Hub identity.

The VAST worker supplies SDK objects to :func:`verify_info`.  Keeping the
identity checks independent from the SDK makes the fail-closed contract
testable without network access or a checkpoint.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


class IdentityError(ValueError):
    """Raised when an upstream identity field does not match its pin."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _lfs_field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    return _field(value, name, default)


def verify_info(
    info: Any,
    tree: Iterable[Any],
    *,
    repository: str,
    revision: str,
    filename: str,
    expected_bytes: int,
    expected_sha256: str,
    expected_license: str = "mit",
) -> None:
    """Verify exact Hub revision, card data, and one LFS file entry.

    ``ModelInfo.card_data`` and ``BlobLfsInfo.sha256`` are the pinned
    huggingface-hub 1.27.0 SDK attributes.  No legacy JSON spelling is
    accepted, so an SDK shape drift fails closed instead of silently skipping
    the license check.
    """

    resolved_repository = _field(info, "id")
    if resolved_repository != repository:
        raise IdentityError(f"HF repository mismatch: {resolved_repository!r} != {repository!r}")
    resolved_revision = _field(info, "sha")
    if resolved_revision != revision:
        raise IdentityError(f"HF resolved revision mismatch: {resolved_revision!r} != {revision!r}")
    card_data = _field(info, "card_data")
    if card_data is None:
        raise IdentityError("HF ModelInfo.card_data is missing")
    if _field(card_data, "license") != expected_license:
        raise IdentityError(
            f"HF card_data license mismatch: {_field(card_data, 'license')!r} != {expected_license!r}"
        )

    matches = []
    for item in tree:
        if _field(item, "path") == filename:
            matches.append(item)
    if len(matches) != 1:
        raise IdentityError(f"expected one exact HF file {filename!r}, found {len(matches)}")
    item = matches[0]
    if _field(item, "size") != expected_bytes:
        raise IdentityError(f"HF file size mismatch: {_field(item, 'size')!r} != {expected_bytes}")
    lfs = _field(item, "lfs")
    if lfs is None:
        raise IdentityError("HF file has no LFS metadata")
    lfs_size = _lfs_field(lfs, "size")
    if lfs_size != expected_bytes:
        raise IdentityError(f"HF LFS size mismatch: {lfs_size!r} != {expected_bytes}")
    lfs_sha256 = _lfs_field(lfs, "sha256")
    if lfs_sha256 != expected_sha256:
        raise IdentityError(f"HF LFS SHA-256 mismatch: {lfs_sha256!r} != {expected_sha256!r}")
    if not isinstance(lfs_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha256):
        raise IdentityError("HF file did not provide a valid LFS SHA-256")
