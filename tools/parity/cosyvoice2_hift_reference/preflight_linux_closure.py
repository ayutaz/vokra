#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stage the exact active Linux x86_64 CPython 3.12 wheel closure.

This model-free helper uses only stdlib and the checked-in lock.  It downloads
hash/size-bound wheels into an absent claimed directory and never installs or
imports them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from audit_linux_closure import active_rows, load_lock, locked_artifact, safe_filename

USER_AGENT = "vokra-cosyvoice2-hift-linux-closure/1.0"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"cosyvoice2 HiFT Linux closure preflight: BLOCKED: {message}")


def require_vast_linux() -> None:
    if (
        os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1"
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        fail("closure staging requires VOKRA_PUBLISH_ON_VAST=1 on Linux x86_64")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_locked(url: str, destination: Path, expected_size: int) -> None:
    initial = safe_filename(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opened = False
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if getattr(response, "status", 200) != 200:
                fail(f"HTTP {response.status} for {initial}")
            redirected = response.geturl()
            if safe_filename(redirected) != initial or urlparse(redirected).hostname not in {"files.pythonhosted.org", "download-r2.pytorch.org"}:
                fail(f"redirect changed or left allowlisted host for {initial}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and (not content_length.isdecimal() or int(content_length) != expected_size):
                fail(f"Content-Length mismatch for {initial}")
            with destination.open("wb") as stream:
                opened = True
                written = 0
                while True:
                    chunk = response.read(min(1 << 20, expected_size - written + 1))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > expected_size:
                        fail(f"oversized response for {initial}")
                    stream.write(chunk)
                if written != expected_size:
                    fail(f"response size mismatch for {initial}")
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        if opened: destination.unlink(missing_ok=True)
        fail(f"download failed for {initial}: {error}")


def stage(lock_path: Path, output: Path, fetch: Callable[[str, Path, int], None] = download_locked) -> list[dict[str, Any]]:
    lock = load_lock(lock_path)
    rows = [row for row in active_rows(lock) if "registry" in row["source"]]
    if output.exists() or output.is_symlink():
        fail(f"staging directory must be absent: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        fail("staging parent must be an existing directory")
    output.mkdir()
    claimed = True
    staged = []
    try:
        for row in rows:
            artifact, filename, size = locked_artifact(row)
            fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=output)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                fetch(artifact["url"], temporary, size)
                if temporary.stat().st_size != size or sha256_file(temporary) != artifact["hash"].removeprefix("sha256:"):
                    fail(f"staged wheel does not match lock: {filename}")
                destination = output / filename
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    fail(f"wheel appeared concurrently: {filename}")
                staged.append({"name": row["name"], "version": row["version"], "filename": filename})
            finally: temporary.unlink(missing_ok=True)
        return staged
    except BaseException:
        if claimed and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


def self_test() -> None:
    assert Path(__file__).with_name("uv.lock").is_file()
    assert Path(__file__).with_name("pyproject.toml").is_file()
    source = Path(__file__).read_text(encoding="utf-8")
    for token in ("import " + "torch", "import " + "numpy", "uv " + "sync", "model " + "download"):
        assert token not in source
    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def geturl(self):
            return "https://evil.example/x.whl"

        def read(self, _):
            return b""
    original = urllib.request.urlopen
    urllib.request.urlopen = lambda *_a, **_k: Response()
    root = Path(tempfile.mkdtemp(prefix="cosyvoice2-hift-stage-test-"))
    destination = root / "x.whl"
    try:
        try:
            download_locked("https://files.pythonhosted.org/x.whl", destination, 0)
        except SystemExit as error:
            assert "redirect" in str(error) or "allowlisted" in str(error)
        else:
            raise AssertionError("unapproved redirect accepted")
        existing = root / "claimed"
        existing.mkdir()
        try:
            stage(Path(__file__).with_name("uv.lock"), existing, lambda *_: None)
        except SystemExit as error:
            assert "absent" in str(error)
        else:
            raise AssertionError("existing staging directory accepted")
        tampered = root / "tampered"

        def fake_tamper(_url: str, path: Path, size: int) -> None:
            path.write_bytes(b"x" * size)

        try:
            stage(Path(__file__).with_name("uv.lock"), tampered, fake_tamper)
        except SystemExit as error:
            assert "match lock" in str(error)
        else:
            raise AssertionError("tampered wheel accepted")
        assert not tampered.exists()
    finally:
        urllib.request.urlopen = original
        shutil.rmtree(root, ignore_errors=True)
    print("cosyvoice2_hift Linux closure preflight self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.output:
        parser.error("--output is required")
    require_vast_linux()
    stage(args.lock, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
