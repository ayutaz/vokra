#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stage the active BigVGAN Linux wheel closure without installing anything.

This is a model-free VAST preflight. It reads the pinned ``uv.lock``, selects
the CPython 3.12 x86_64 glibc wheel for every active registry row, downloads
only those hash/size-bound URLs, and leaves inspection to
``audit_linux_closure.py``. It never imports a third-party package, changes an
environment, or touches a checkpoint/model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

import tomllib

from audit_linux_closure import active_rows, audit, canonical_wheel_name, locked_artifact, sha256_file

HTTP_USER_AGENT = "vokra-bigvgan-preflight/1.0"


def fail(message: str) -> None:
    raise SystemExit(f"bigvgan Linux preflight: BLOCKED: {message}")


def load_lock(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail("uv.lock is missing or symlinked")
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        fail(f"uv.lock is unreadable: {exc}")


def locked_wheel(row: dict[str, Any]) -> tuple[str, str, int]:
    candidate, _basis, _filename, digest, size = locked_artifact(row)
    url = candidate["url"]
    canonical_wheel_name(url)
    return url, digest, size


def download_locked(url: str, destination: Path, expected_size: int) -> None:
    """Stream one locked URL; no package manager or archive import is used."""
    initial_name = canonical_wheel_name(url)
    opened = False
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if getattr(response, "status", 200) != 200:
                fail(f"HTTP {response.status} fetching locked wheel {initial_name}")
            redirect_url = response.geturl()
            try:
                redirect_name = canonical_wheel_name(redirect_url)
            except SystemExit as exc:
                fail(f"redirect for locked wheel {initial_name} rejected: {exc}")
            if redirect_name != initial_name:
                fail(f"redirect changed locked wheel filename {initial_name}: {redirect_url}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                if not content_length.isdecimal() or int(content_length) != expected_size:
                    fail(
                        f"locked wheel {initial_name} Content-Length {content_length!r} != {expected_size}"
                    )
            with destination.open("wb") as stream:
                opened = True
                written = 0
                while True:
                    chunk = response.read(min(1 << 20, expected_size - written + 1))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > expected_size:
                        fail(
                            f"locked wheel {initial_name} response exceeded expected size {expected_size}"
                        )
                    stream.write(chunk)
                if written != expected_size:
                    fail(f"locked wheel {initial_name} response size {written} != {expected_size}")
    except urllib.error.HTTPError as exc:
        if opened:
            destination.unlink(missing_ok=True)
        reason = f" ({exc.reason})" if exc.reason else ""
        fail(f"HTTP {exc.code}{reason} fetching locked wheel {initial_name}")
    except (OSError, urllib.error.URLError) as exc:
        if opened:
            destination.unlink(missing_ok=True)
        fail(f"download failed for locked wheel {initial_name}: {exc}")
    except BaseException:
        if opened:
            destination.unlink(missing_ok=True)
        raise


def require_absent_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        fail(f"artifact staging directory must be absent for a clean run: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        fail(f"artifact staging parent must be an existing non-symlink directory: {path.parent}")
    path.mkdir()


def stage(
    lock_path: Path,
    artifacts_dir: Path,
    fetch: Callable[[str, Path, int], None] = download_locked,
) -> list[dict[str, Any]]:
    """Materialize exact active Linux wheels and return their lock identities."""
    lock = load_lock(lock_path)
    rows = active_rows(lock)
    require_absent_directory(artifacts_dir)
    created = True
    staged: list[dict[str, Any]] = []
    try:
        for row in rows:
            url, expected_hash, expected_size = locked_wheel(row)
            filename = canonical_wheel_name(url)
            destination = artifacts_dir / filename
            temporary: Path | None = None
            try:
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{filename}.", suffix=".tmp", dir=artifacts_dir
                )
                os.close(fd)
                temporary = Path(temporary_name)
                fetch(url, temporary, expected_size)
                if temporary.stat().st_size != expected_size:
                    fail(
                        f"{row['name']} wheel {filename} size {temporary.stat().st_size} != locked {expected_size}"
                    )
                actual_hash = sha256_file(temporary)
                if actual_hash != expected_hash:
                    fail(
                        f"{row['name']} wheel {filename} SHA-256 {actual_hash} != locked {expected_hash}"
                    )
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    fail(f"staged wheel appeared concurrently; refusing overwrite: {destination}")
                staged.append({"name": row["name"], "version": row["version"], "filename": filename})
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        return staged
    except BaseException:
        if created and artifacts_dir.is_dir() and not artifacts_dir.is_symlink():
            shutil.rmtree(artifacts_dir)
        raise


def self_test() -> None:
    import io

    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_tokens = [
        "import " + "torch",
        "import " + "numpy",
        "import " + "transformers",
        "pip " + "install",
        "u" + "v sync",
        "model" + " download",
        "car" + "go",
        "--" + "push",
    ]
    for forbidden in forbidden_tokens:
        assert forbidden not in source
    for unsafe_url in (
        "https://files.pythonhosted.org/packages/name%2Fescape.whl",
        "https://files.pythonhosted.org/packages/name%5Cescape.whl",
        "https://files.pythonhosted.org/packages/name%00escape.whl",
        "https://files.pythonhosted.org/packages/..",
    ):
        try:
            canonical_wheel_name(unsafe_url)
        except SystemExit as exc:
            assert "safe filename" in str(exc)
        else:
            raise AssertionError(f"unsafe encoded wheel basename was accepted: {unsafe_url}")

    class RedirectResponse:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> "RedirectResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def geturl(self) -> str:
            return "https://evil.example/torch-2.7.1+cpu-py3-none-any.whl"

        def read(self, _size: int) -> bytes:
            return b""

    original_urlopen = urllib.request.urlopen
    redirect_requests: list[urllib.request.Request] = []

    def redirect_fetch(request: Any, **_kwargs: Any) -> RedirectResponse:
        assert isinstance(request, urllib.request.Request)
        redirect_requests.append(request)
        return RedirectResponse()

    urllib.request.urlopen = redirect_fetch
    redirect_destination = Path(tempfile.mkdtemp(prefix="bigvgan-redirect-")).joinpath("wheel.whl")
    try:
        try:
            download_locked("https://files.pythonhosted.org/packages/wheel.whl", redirect_destination, 0)
        except SystemExit as exc:
            assert "allowlisted" in str(exc)
        else:
            raise AssertionError("redirect to an untrusted host was accepted")
        assert len(redirect_requests) == 1
        assert redirect_requests[0].get_header("User-agent") == HTTP_USER_AGENT
    finally:
        urllib.request.urlopen = original_urlopen
        shutil.rmtree(redirect_destination.parent)

    class PayloadResponse:
        status = 200

        def __init__(self, payload: bytes, content_length: str | None = None) -> None:
            self.payload = payload
            self.headers = {} if content_length is None else {"Content-Length": content_length}

        def __enter__(self) -> "PayloadResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def geturl(self) -> str:
            return "https://files.pythonhosted.org/packages/wheel.whl"

        def read(self, size: int) -> bytes:
            chunk, self.payload = self.payload[:size], self.payload[size:]
            return chunk

    payload_destination = Path(tempfile.mkdtemp(prefix="bigvgan-size-")).joinpath("wheel.whl")
    original_urlopen = urllib.request.urlopen
    try:
        urllib.request.urlopen = lambda *_args, **_kwargs: PayloadResponse(b"abcd", "4")
        try:
            download_locked("https://files.pythonhosted.org/packages/wheel.whl", payload_destination, 3)
        except SystemExit as exc:
            assert "Content-Length" in str(exc)
        else:
            raise AssertionError("lying Content-Length was accepted")
        assert not payload_destination.exists()
        urllib.request.urlopen = lambda *_args, **_kwargs: PayloadResponse(b"abcd", None)
        try:
            download_locked("https://files.pythonhosted.org/packages/wheel.whl", payload_destination, 3)
        except SystemExit as exc:
            assert "exceeded expected size" in str(exc)
        else:
            raise AssertionError("oversized response was accepted")
        assert not payload_destination.exists()
    finally:
        urllib.request.urlopen = original_urlopen
        shutil.rmtree(payload_destination.parent)

    http_error_destination = Path(tempfile.mkdtemp(prefix="bigvgan-http-error-")).joinpath("wheel.whl")
    original_urlopen = urllib.request.urlopen

    def raise_http_error(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.HTTPError(
            "https://files.pythonhosted.org/packages/wheel.whl",
            403,
            "Cloudflare test response",
            hdrs=None,
            fp=None,
        )

    try:
        urllib.request.urlopen = raise_http_error
        try:
            download_locked(
                "https://files.pythonhosted.org/packages/wheel.whl",
                http_error_destination,
                3,
            )
        except SystemExit as exc:
            message = str(exc)
            assert "HTTP 403" in message
            assert "wheel.whl" in message
        else:
            raise AssertionError("HTTP error was accepted")
        assert not http_error_destination.exists()
    finally:
        urllib.request.urlopen = original_urlopen
        shutil.rmtree(http_error_destination.parent)

    with tempfile.TemporaryDirectory(prefix="bigvgan-preflight-") as directory:
        root = Path(directory)
        metadata = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\nLicense: MIT\n"
        wheel_buffer = io.BytesIO()
        with zipfile.ZipFile(wheel_buffer, "w") as archive:
            archive.writestr("demo-1.0.dist-info/METADATA", metadata)
        wheel = wheel_buffer.getvalue()
        wheel_buffer.close()
        wheel_name = "demo-1.0+cpu-py3-none-any.whl"
        encoded_wheel_name = wheel_name.replace("+", "%2B")
        url = f"https://files.pythonhosted.org/packages/{encoded_wheel_name}"
        lock = root / "uv.lock"
        lock.write_text(
            f"""version = 1
revision = 3
requires-python = '==3.12.*'
resolution-markers = []
supported-markers = []

[[package]]
name = 'demo'
version = '1.0'
source = {{ registry = 'https://pypi.org/simple' }}
wheels = [{{ url = '{url}', hash = 'sha256:{hashlib.sha256(wheel).hexdigest()}', size = {len(wheel)} }}]

[[package]]
name = 'vokra-bigvgan-parity'
version = '0.1.0'
source = {{ virtual = '.' }}
dependencies = [{{ name = 'demo', version = '1.0', source = {{ registry = 'https://pypi.org/simple' }} }}]

[[package]]
name = 'unreachable-orphan'
version = '1.0'
source = {{ registry = 'https://pypi.org/simple' }}
wheels = [{{ url = '{url}', hash = 'sha256:{hashlib.sha256(wheel).hexdigest()}', size = {len(wheel)} }}]
""",
            encoding="utf-8",
        )
        artifacts = root / "artifacts"
        fetch_calls: list[tuple[str, int]] = []

        def fake_fetch(fetch_url: str, destination: Path, expected_size: int) -> None:
            fetch_calls.append((fetch_url, expected_size))
            destination.write_bytes(wheel)

        staged = stage(lock, artifacts, fake_fetch)
        assert staged == [{"name": "demo", "version": "1.0", "filename": wheel_name}]
        assert (artifacts / wheel_name).read_bytes() == wheel
        assert not (artifacts / encoded_wheel_name).exists()
        assert not list(artifacts.glob(".*.tmp"))
        candidate = root / "candidate.json"
        audit(lock, artifacts, candidate)
        assert candidate.is_file() and candidate.read_text(encoding="utf-8").endswith("\n")
        assert fetch_calls == [(url, len(wheel))]
        try:
            stage(lock, artifacts, fake_fetch)
        except SystemExit as exc:
            assert "must be absent" in str(exc)
        else:
            raise AssertionError("preflight rerun reused an existing staging directory")

        bad_hash_lock = root / "bad-hash.lock"
        digest = hashlib.sha256(wheel).hexdigest()
        bad_hash_lock.write_text(
            lock.read_text(encoding="utf-8").replace(digest, "0" * 64), encoding="utf-8"
        )
        bad_hash_artifacts = root / "bad-hash-artifacts"
        try:
            stage(bad_hash_lock, bad_hash_artifacts, fake_fetch)
        except SystemExit as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("hash mismatch was accepted")
        assert not bad_hash_artifacts.exists()

        missing_size_lock = root / "missing-size.lock"
        missing_size_lock.write_text(
            lock.read_text(encoding="utf-8").replace(f", size = {len(wheel)}", ""),
            encoding="utf-8",
        )
        missing_size_artifacts = root / "missing-size-artifacts"
        try:
            stage(missing_size_lock, missing_size_artifacts, fake_fetch)
        except SystemExit as exc:
            assert "malformed" in str(exc)
        else:
            raise AssertionError("missing wheel size was accepted")
        assert not missing_size_artifacts.exists()

        bad_lock = root / "bad.lock"
        bad_lock.write_text(lock.read_text(encoding="utf-8").replace(f"size = {len(wheel)}", "size = 1"), encoding="utf-8")
        bad_artifacts = root / "bad-artifacts"
        try:
            stage(bad_lock, bad_artifacts, fake_fetch)
        except SystemExit as exc:
            assert "size" in str(exc)
        else:
            raise AssertionError("size mismatch was accepted")
        assert not bad_artifacts.exists()
        assert not list(root.glob("bad-artifacts/.*.tmp"))
    print("preflight_linux_closure.py self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.lock is not None or args.artifacts_dir is not None:
            parser.error("--self-test accepts no other arguments")
        self_test()
        return
    if args.lock is None or args.artifacts_dir is None:
        parser.error("--lock and --artifacts-dir are required")
    stage(args.lock, args.artifacts_dir)
    print(f"bigvgan Linux preflight: staged active locked wheels in {args.artifacts_dir}")


if __name__ == "__main__":
    main()
