#!/usr/bin/env python3
"""Model-free factual audit of the pinned MOSS Audio Tokenizer v2 environment.

The audit is intentionally boring: it reads the checked-in lock and gate
manifest, observes ``importlib.metadata`` only, and records primary-source
license bytes.  It never imports torch/transformers/model code and never
executes package code.  Network access is limited to exact LICENSE URLs and
exact locked PyPI sdists when a wheel has no publisher file.
"""
from __future__ import annotations

import argparse, base64, hashlib, importlib.metadata as metadata, io, json
import platform, re, stat, subprocess, sys, tarfile, zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import tomllib

try:
    import license_gate
except ModuleNotFoundError:  # pragma: no cover
    from tools.parity.moss_audio_tokenizer_v2 import license_gate

SCHEMA = "vokra-moss-audio-tokenizer-v2-dependency-audit-v1"
PYPI_HOST = "files.pythonhosted.org"
LICENSE_HOSTS = {"cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co"}
LICENSE_NAMES = {"license", "copying", "notice", "copyright"}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 10_000
MAX_REDIRECTS = 3
SOURCE = {"id": "moss-audio-tokenizer-v2-source", "repo": license_gate.REPO,
          "revision": license_gate.REVISION}


class AuditError(ValueError):
    pass


class SdistError(AuditError):
    def __init__(self, message: str, *, acquired: bool | None = None,
                 verified: bool = False, size: int | None = None,
                 digest: str | None = None) -> None:
        super().__init__(message)
        self.acquired = acquired
        self.verified = verified
        self.observed_size = size
        self.observed_sha256 = digest


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def norm_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def norm_version(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).casefold()


def ident(name: str, version: str) -> str:
    return f"{norm_name(name)}=={norm_version(version)}"


def read_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise AuditError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    pp, lp, mp = (project / x for x in ("pyproject.toml", "uv.lock", "license_gate_manifest.json"))
    if not all(_regular(x) for x in (pp, lp, mp)):
        raise AuditError("MOSS project/lock/manifest is missing or symlinked")
    try:
        pb, lb = pp.read_bytes(), lp.read_bytes()
        pd, ld = tomllib.loads(pb.decode()), tomllib.loads(lb.decode())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"closure is unreadable: {exc}") from exc
    m = read_json(mp)
    if not isinstance(m, dict) or m.get("gate_version") != 1:
        raise AuditError("gate manifest schema/version is unsupported")
    if sha(pb) != license_gate.PROJECT_SHA256 or sha(lb) != license_gate.LOCK_SHA256:
        raise AuditError("project/lock bytes differ from the code-bound contract")
    try:
        rows = license_gate.lock_rows(ld)
        if license_gate.artifact_error(ld):
            raise AuditError(license_gate.artifact_error(ld) or "malformed artifact")
        license_gate.project_identity(pb)
    except (SystemExit, ValueError) as exc:
        raise AuditError(f"closure schema is invalid: {exc}") from exc
    if m.get("lock_sha256") != license_gate.LOCK_SHA256 or m.get("project_sha256") != license_gate.PROJECT_SHA256:
        raise AuditError("manifest does not bind exact closure bytes")
    if m.get("package_rows") != rows or m.get("package_rows_sha256") != sha(canonical(rows).encode()):
        raise AuditError("manifest package rows do not bind exact lock rows")
    reviews = m.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        raise AuditError("manifest package review rows do not cover every locked row")
    expected_reviews = {(row["name"], row["version"]): row["source"] for row in rows}
    seen_reviews: set[tuple[str, str]] = set()
    for review in reviews:
        if not isinstance(review, dict) or not isinstance(review.get("name"), str) or not isinstance(review.get("version"), str):
            raise AuditError("manifest package review row is malformed")
        key = (review["name"], review["version"])
        if key in seen_reviews or key not in expected_reviews or review.get("source") != expected_reviews[key]:
            raise AuditError("manifest package review rows are not one-to-one with lock rows")
        seen_reviews.add(key)
    if seen_reviews != set(expected_reviews):
        raise AuditError("manifest package review rows omit a locked row")
    ids = m.get("identities")
    if not isinstance(ids, dict) or ids.get("repo") != license_gate.REPO or ids.get("revision") != license_gate.REVISION:
        raise AuditError("fixed MOSS upstream identity drifted")
    return pd, ld, m, rows, pb, lb


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    e, a = Counter(expected), Counter(actual)
    return {"expected": sorted(expected), "installed": sorted(actual),
            "missing": sorted((e-a).elements()), "unexpected": sorted((a-e).elements()),
            "duplicate_identities": sorted(k for k, v in a.items() if v > 1),
            "exact": not (e-a or a-e)}


LINUX_X86_64_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"


def row_active(row: dict[str, Any]) -> bool:
    markers = row.get("resolution-markers", [])
    if not markers:
        return True
    return isinstance(markers, list) and LINUX_X86_64_MARKER in markers


def installed_distributions() -> list[dict[str, Any]]:
    result = []
    for dist in metadata.distributions():
        name, version = dist.metadata.get("Name"), dist.version
        if name and version:
            result.append({"distribution": dist, "name": name, "version": version,
                           "identity": ident(name, version),
                           "location": str(Path(dist.locate_file("")))})
    return sorted(result, key=lambda x: (x["identity"], x["location"]))


def metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(x.removeprefix("License :: ") for x in (dist.metadata.get_all("Classifier") or [])
                          if x.startswith("License :: "))
    def clean(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None
    return {"license": clean(dist.metadata.get("License")),
            "license_expression": clean(dist.metadata.get("License-Expression")),
            "license_classifiers": classifiers}


def _entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    lexical_root = Path(dist.locate_file(""))
    lexical_path = Path(dist.locate_file(entry))
    try:
        lexical_relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        return None
    if lexical_root.is_symlink() or any((lexical_root / part).is_symlink() for part in lexical_relative.parts):
        return None
    root = lexical_root.resolve(strict=False)
    path = lexical_path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def is_license_path(path: str) -> bool:
    base = Path(path).name.casefold()
    return any(base == name or (base.startswith(name) and base[len(name)] in ".-_") for name in LICENSE_NAMES)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    good, bad = [], []
    for entry in sorted(dist.files or [], key=str):
        rel = str(entry)
        if not is_license_path(rel):
            continue
        path = _entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            bad.append(rel); continue
        try:
            good.append({"path": rel, "size": path.stat().st_size, "sha256": sha_file(path)})
        except OSError as exc:
            bad.append(f"{rel}: {exc}")
    return good, bad


def _elf(path: Path, magic: bytes) -> dict[str, Any]:
    if magic != ELF_MAGIC:
        return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    try:
        p = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True,
                           check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "inspection": "error", "error": str(exc)}
    needed = sorted(m.group(1) for line in p.stdout.splitlines()
                    if (m := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
    status = "ok" if p.returncode == 0 else ("no-dynamic-section" if "no dynamic section" in (p.stdout+p.stderr).casefold() else "error")
    result = {"format": "elf", "needed": needed, "inspection": status, "readelf_returncode": p.returncode}
    if status == "error": result["error"] = (p.stdout+p.stderr).strip()[-2000:]
    return result


def native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result, errors = [], []
    for entry in sorted(dist.files or [], key=str):
        rel = str(entry); name = Path(rel).name.casefold()
        candidate = Path(rel).suffix.casefold() in NATIVE_SUFFIXES or ".so." in name
        path = _entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            if candidate: errors.append({"path": rel, "stage": "path", "error": "native candidate is not a regular file"})
            continue
        try:
            with path.open("rb") as handle: magic = handle.read(4)
            if not candidate and magic != ELF_MAGIC: continue
            digest, size = sha_file(path), path.stat().st_size
            needed = _elf(path, magic)
            result.append({"path": rel, "size": size, "sha256": digest, "bundled": True,
                           "candidate": "native-suffix" if candidate else "elf-magic", "needed": needed})
        except OSError as exc:
            errors.append({"path": rel, "stage": "read", "error": str(exc)})
    return result, errors


def _safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise AuditError(f"unsafe archive path: {name!r}")
    clean = name.rstrip("/")
    if not clean or any(part in {"", ".", ".."} for part in clean.split("/")):
        raise AuditError(f"archive path traversal: {name!r}")
    return clean


def _archive(url: str, body: bytes) -> tuple[str, list[dict[str, Any]]]:
    lower = urlsplit(url).path.casefold()
    modes = next((("tar.gz", "r:gz") for x in (".tar.gz", ".tgz") if lower.endswith(x)), None)
    if modes is None:
        modes = next((("tar.bz2", "r:bz2") for x in (".tar.bz2", ".tbz2") if lower.endswith(x)), None)
    if modes is None:
        modes = next((("tar.xz", "r:xz") for x in (".tar.xz", ".txz") if lower.endswith(x)), None)
    if modes is None and lower.endswith(".tar"): modes = ("tar", "r:")
    if modes is None and lower.endswith(".zip"): modes = ("zip", "zip")
    if modes is None: raise AuditError("unsupported sdist archive type")
    if len(body) > MAX_SDIST_BYTES: raise AuditError("sdist archive is too large")
    names: set[str] = set(); total = 0; license_total = 0; candidates = []
    def member(name: str, size: int, regular: bool, read: Callable[[], bytes]) -> None:
        nonlocal total, license_total
        clean = _safe_member(name)
        if clean in names: raise AuditError(f"duplicate archive member: {clean}")
        names.add(clean)
        if len(names) > MAX_MEMBERS or size < 0 or size > MAX_MEMBER_BYTES:
            raise AuditError("archive member/count bound exceeded")
        if not regular: return
        total += size
        if total > MAX_TOTAL_MEMBER_BYTES: raise AuditError("archive aggregate bound exceeded")
        if not is_license_path(clean): return
        if size > MAX_LICENSE_BYTES: raise AuditError("license member bound exceeded")
        data = read()
        if len(data) != size: raise AuditError("archive member size changed")
        license_total += size
        if license_total > MAX_LICENSE_TOTAL_BYTES: raise AuditError("license aggregate bound exceeded")
        candidates.append({"path": clean, "size": size, "sha256": sha(data),
                           "content_base64": base64.b64encode(data).decode("ascii")})
    try:
        if modes[0] == "zip":
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                for info in z.infolist():
                    mode = (info.external_attr >> 16) & 0o170000
                    marker = info.is_dir() or info.filename.endswith("/")
                    if mode and mode not in {stat.S_IFREG, stat.S_IFDIR}:
                        raise AuditError(f"zip link/special member: {info.filename!r}")
                    if (mode == stat.S_IFREG and marker) or (mode == stat.S_IFDIR and not marker):
                        raise AuditError(f"zip type/name contradiction: {info.filename!r}")
                    directory = marker or mode == stat.S_IFDIR
                    member(info.filename, 0 if directory else info.file_size, not directory,
                           lambda info=info: z.read(info))
        else:
            with tarfile.open(fileobj=io.BytesIO(body), mode=modes[1]) as t:
                for info in t:
                    if info.issym() or info.islnk() or info.isdev() or info.isfifo():
                        raise AuditError(f"tar link/special member: {info.name!r}")
                    if not info.isdir() and not info.isfile(): raise AuditError("tar member type unsupported")
                    # Do not call extractfile for non-license regular members.
                    reader = (lambda info=info: _read_tar(t, info)) if info.isfile() and is_license_path(info.name) else (lambda: b"")
                    member(info.name, info.size if info.isfile() else 0, info.isfile(), reader)
    except (OSError, EOFError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise AuditError(f"archive inspection failed: {exc}") from exc
    if not candidates: raise AuditError("archive contains no license candidate")
    return modes[0], sorted(candidates, key=lambda x: x["path"])


def _read_tar(t: tarfile.TarFile, info: tarfile.TarInfo) -> bytes:
    handle = t.extractfile(info)
    if handle is None: raise AuditError(f"cannot read license member: {info.name}")
    with handle: return handle.read(MAX_LICENSE_BYTES + 1)


def _validate_pypi(url: str, expected: str, initial: bool) -> None:
    try: parsed = urlsplit(url); port = parsed.port
    except ValueError as exc: raise AuditError("invalid sdist URL") from exc
    if (parsed.scheme != "https" or parsed.hostname != PYPI_HOST or port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise AuditError(f"sdist URL is not exact files.pythonhosted.org HTTPS: {url}")
    if initial and url != expected: raise AuditError("initial URL differs from lock")
    if not initial and parsed.path != urlsplit(expected).path: raise AuditError("redirect changed exact path")


class _SdistRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], expected: str) -> None:
        self.trace, self.expected = trace, expected
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_REDIRECTS: raise AuditError("sdist redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        _validate_pypi(resolved, self.expected, False); self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def fetch_locked_sdist(row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    if row.get("source") != {"registry": "https://pypi.org/simple"}:
        raise SdistError("sdist fallback is restricted to PyPI", acquired=None)
    sdist = row.get("sdist")
    if not isinstance(sdist, dict) or set(sdist) != ARTIFACT_KEYS:
        raise SdistError("exact sdist artifact schema is required", acquired=None)
    url, expected_hash, expected_size = sdist.get("url"), sdist.get("hash"), sdist.get("size")
    if (not isinstance(url, str) or not isinstance(expected_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash)
            or isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0 or expected_size > MAX_SDIST_BYTES
            or not isinstance(sdist.get("upload-time"), str) or not sdist["upload-time"].strip()):
        raise SdistError("malformed or unbounded exact sdist identity", acquired=None)
    _validate_pypi(url, url, True); trace = [url]
    try:
        if fetcher is None:
            opener = build_opener(_SdistRedirects(trace, url)); request = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-moss-audit/1"})
            with opener.open(request, timeout=30) as response:
                final = urljoin(url, response.geturl()); _validate_pypi(final, url, False)
                if final != trace[-1]: trace.append(final)
                length = response.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > MAX_SDIST_BYTES: raise SdistError("Content-Length exceeds bound", acquired=None)
                body = response.read(MAX_SDIST_BYTES + 1)
        else:
            final, body = fetcher(url); final = urljoin(url, final); _validate_pypi(final, url, False)
            if final != trace[-1]: trace.append(final)
    except SdistError: raise
    except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
        raise SdistError(str(exc), acquired=None) from exc
    if not isinstance(body, bytes): raise SdistError("fetcher did not return bytes", acquired=None)
    observed = sha(body)
    if len(body) != expected_size: raise SdistError("sdist size mismatch", acquired=True, size=len(body), digest="sha256:"+observed)
    if "sha256:" + observed != expected_hash: raise SdistError("sdist SHA-256 mismatch", acquired=True, size=len(body), digest="sha256:"+observed)
    try: fmt, files = _archive(final, body)
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        raise SdistError(str(exc), acquired=True, verified=True, size=len(body), digest="sha256:"+observed) from exc
    return {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "package": row["name"], "version": row["version"],
            "archive": {"url": url, "final_url": final, "url_trace": trace, "size": expected_size, "hash": expected_hash, "format": fmt},
            "license_files": files}


def blocked_sdist(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    sdist = row.get("sdist") if isinstance(row.get("sdist"), dict) else {}
    return {"status": "BLOCKED_FACTUAL_SDIST_LICENSE_PATH", "package": row.get("name"), "version": row.get("version"),
            "requested_url": sdist.get("url"), "archive": {"url": sdist.get("url"), "size": sdist.get("size"), "hash": sdist.get("hash"),
            "observed_size": getattr(exc, "observed_size", None), "observed_sha256": getattr(exc, "observed_sha256", None)},
            "acquired_archive_bytes": getattr(exc, "acquired", None), "verified_archive": bool(getattr(exc, "verified", False)),
            "license_files": [], "error": controlled_error(exc)}


def controlled_error(exc: Exception) -> dict[str, Any]:
    if isinstance(getattr(exc, "__cause__", None), Exception):
        return controlled_error(exc.__cause__)
    if isinstance(exc, HTTPError): return {"kind": "HTTP_ERROR", "status": exc.code}
    if isinstance(exc, URLError): return {"kind": "URL_ERROR"}
    if isinstance(exc, (AuditError, ValueError)): return {"kind": "VALIDATION_ERROR"}
    if isinstance(exc, OSError): return {"kind": "OS_ERROR"}
    return {"kind": "UNEXPECTED_ERROR"}


def approval_state(manifest: dict[str, Any]) -> dict[str, Any]:
    reviews = manifest.get("package_review_rows")
    package_blockers = []
    if isinstance(reviews, list):
        for row in reviews:
            if isinstance(row, dict) and (row.get("status") != "REVIEWED" or row.get("license") in {None, "UNRESOLVED"} or row.get("native_bundled_review") in {None, "OWNER_REVIEW_REQUIRED"}):
                package_blockers.append({"name": row.get("name"), "version": row.get("version"), "status": row.get("status"), "license": row.get("license"), "native_bundled_review": row.get("native_bundled_review")})
    license_rows = manifest.get("license_rows")
    license_blockers = []
    if isinstance(license_rows, list):
        for row in license_rows:
            if isinstance(row, dict) and (row.get("status") != "REVIEWED" or row.get("license") in {None, "UNRESOLVED"} or row.get("conclusion") in {None, "UNRESOLVED"}):
                license_blockers.append({"id": row.get("id"), "status": row.get("status"), "license": row.get("license"), "conclusion": row.get("conclusion")})
    approval = manifest.get("approval")
    approval_blockers = []
    if not isinstance(approval, dict) or approval.get("status") != "OWNER_SIGNOFF_APPROVED":
        approval_blockers.append({"status": approval.get("status") if isinstance(approval, dict) else None, "required": "OWNER_SIGNOFF_APPROVED"})
    publication = manifest.get("publication_decision")
    if publication != "APPROVED":
        approval_blockers.append({"publication_decision": publication, "required": "APPROVED"})
    return {"license_rows": license_rows, "approval": approval, "publication_decision": publication,
            "approval_blockers": approval_blockers, "package_review_blockers": package_blockers,
            "license_review_blockers": license_blockers, "publication_permitted": False}


def blocked_approval_state(project: Path) -> dict[str, Any]:
    try:
        manifest = read_json(project / "license_gate_manifest.json")
        if isinstance(manifest, dict):
            return approval_state(manifest)
    except (AuditError, OSError, UnicodeError, ValueError):
        pass
    return {"license_rows": None, "approval": None, "publication_decision": None,
            "approval_blockers": [{"reason": "manifest unavailable"}], "package_review_blockers": [],
            "license_review_blockers": [], "publication_permitted": False}


def license_url() -> str:
    return f"https://huggingface.co/{SOURCE['repo']}/raw/{SOURCE['revision']}/LICENSE"


def validate_license_url(url: str, initial: str | None = None) -> None:
    parsed = urlsplit(url)
    try: port = parsed.port
    except ValueError as exc: raise AuditError("invalid LICENSE port") from exc
    raw = license_url(); cdn = f"/{SOURCE['repo']}/resolve/{SOURCE['revision']}/LICENSE"
    exact = (url == raw if initial is None or initial == raw else False)
    allowed = (parsed.hostname == "huggingface.co" and parsed.path == urlsplit(raw).path and url == raw) or (parsed.hostname in LICENSE_HOSTS and parsed.path == cdn)
    if parsed.scheme != "https" or port not in (None, 443) or parsed.username or parsed.password or parsed.query or parsed.fragment or not allowed:
        raise AuditError(f"non-allow-listed LICENSE URL: {url}")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str]): self.trace = trace
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_REDIRECTS: raise AuditError("LICENSE redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        validate_license_url(resolved, self.trace[0]); self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def fetch_license(fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    requested = license_url(); validate_license_url(requested, requested); trace = [requested]
    if fetcher is None:
        opener = build_opener(_LicenseRedirects(trace)); request = Request(requested, headers={"Accept": "text/plain", "User-Agent": "vokra-moss-audit/1"})
        with opener.open(request, timeout=30) as response:
            final = urljoin(requested, response.geturl()); validate_license_url(final, requested)
            if final != trace[-1]: trace.append(final)
            body = response.read(MAX_LICENSE_BYTES + 1)
    else:
        final, body = fetcher(requested); final = urljoin(requested, final); validate_license_url(final, requested)
        if final != trace[-1]: trace.append(final)
    if not isinstance(body, bytes): raise AuditError("LICENSE fetcher did not return bytes")
    if len(body) > MAX_LICENSE_BYTES: raise AuditError("LICENSE response exceeds bound")
    return {"id": SOURCE["id"], "repo": SOURCE["repo"], "revision": SOURCE["revision"], "requested_url": requested,
            "final_url": final, "url_trace": trace, "size": len(body), "sha256": sha(body), "content_base64": base64.b64encode(body).decode(),
            "claimed_license": None, "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"}


def blocked_license(exc: Exception) -> dict[str, Any]:
    requested = license_url()
    return {"status": "BLOCKED_FACTUAL_LICENSE_PATH", "id": SOURCE["id"], "repo": SOURCE["repo"], "revision": SOURCE["revision"],
            "requested_url": requested, "final_url": None, "url_trace": [requested], "acquired_bytes": False,
            "size": None, "sha256": None, "content_base64": None, "claimed_license": None,
            "license_classification": "UNAVAILABLE_FACTUAL_LICENSE_PATH", "error": controlled_error(exc)}


def inspect_package(row: dict[str, Any], record: dict[str, Any] | None, review: dict[str, Any] | None,
                    fetcher: Callable[[str], tuple[str, bytes]] | None) -> tuple[dict[str, Any], list[str]]:
    base = {"lock": {k: row[k] for k in ("name", "version", "source")}, "review": review}
    if record is None: return {**base, "installed": None}, [f"installed closure missing: {ident(row['name'], row['version'])}"]
    dist = record["distribution"]; publisher, unsafe = publisher_files(dist); native, native_errors = native_files(dist); fields = metadata_fields(dist)
    sdist = None
    if not publisher:
        try: sdist = fetch_locked_sdist(row, fetcher)
        except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc: sdist = blocked_sdist(row, exc)
    installed = {"name": dist.metadata.get("Name"), "version": dist.version, "normalized_identity": record["identity"], **fields,
                 "publisher_files": publisher, "sdist_license_evidence": sdist, "native_files": native, "native_errors": native_errors}
    failures = []
    has_license = bool(fields["license"] or fields["license_expression"] or fields["license_classifiers"])
    verified = bool(sdist and sdist.get("status") == "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES" and sdist.get("license_files"))
    if not has_license and not verified: failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher and not verified:
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
        if sdist: failures.append(f"{sdist['status']}: {record['identity']}: {canonical(sdist.get('error'))}")
    failures.extend(f"unsafe publisher path: {record['identity']}:{x}" for x in unsafe)
    failures.extend(f"native candidate inspection failed: {record['identity']}:{x['path']}:{x['stage']}" for x in native_errors)
    failures.extend([f"ELF NEEDED inspection failed: {record['identity']}" for x in native if x["needed"]["inspection"] == "error"])
    return {**base, "lock": {**base["lock"], "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])}}, "installed": installed}, failures


def git_identity(project: Path) -> dict[str, str]:
    root = project.resolve().parents[2]
    try: out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc: raise AuditError(f"cannot bind git commit: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", out): raise AuditError("invalid git commit identity")
    return {"commit": out, "audit_script_sha256": sha(Path(__file__).read_bytes())}


def audit_environment(project: Path, fetch_model_license: bool = True,
                      sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None,
                      license_fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    pd, lock, manifest, rows, pb, lb = contract(project)
    virtual = [r for r in lock["package"] if r.get("source") == {"virtual": "."}]
    real = [r for r in rows if r.get("source") != {"virtual": "."}]
    active_real = [r for r in real if row_active(r)]
    inactive_real = [r for r in real if not row_active(r)]
    records = installed_distributions(); by_id: dict[str, list[dict[str, Any]]] = {}
    for r in records: by_id.setdefault(r["identity"], []).append(r)
    closure = compare_multiset([ident(r["name"], r["version"]) for r in active_real], [r["identity"] for r in records])
    reviews = {(r.get("name"), r.get("version")): r for r in manifest.get("package_review_rows", []) if isinstance(r, dict)}
    packages, failures = [], []
    for review in manifest.get("package_review_rows", []):
        if isinstance(review, dict) and (review.get("status") != "REVIEWED" or review.get("license") in {None, "UNRESOLVED"} or review.get("native_bundled_review") in {None, "OWNER_REVIEW_REQUIRED"}):
            failures.append(f"manifest package review remains unresolved: {review.get('name')}=={review.get('version')}")
    for license_row in manifest.get("license_rows", []):
        if isinstance(license_row, dict) and (license_row.get("status") != "REVIEWED" or license_row.get("license") in {None, "UNRESOLVED"} or license_row.get("conclusion") in {None, "UNRESOLVED"}):
            failures.append(f"manifest license review remains unresolved: {license_row.get('id')}")
    for row in active_real:
        key = ident(row["name"], row["version"]); candidates = by_id.get(key, [])
        item, errs = inspect_package(row, candidates[0] if len(candidates) == 1 else None, reviews.get((row["name"], row["version"])), sdist_fetcher)
        if len(candidates) > 1: errs.append(f"duplicate installed distribution: {key}")
        if (row["name"], row["version"]) not in reviews: errs.append(f"package review evidence missing: {key}")
        packages.append(item); failures.extend(errs)
    for row in inactive_real:
        packages.append({"lock": row, "installed": None, "review": reviews.get((row["name"], row["version"])),
                         "activity": {"status": "INACTIVE_LOCK_ROW", "reason": "resolution marker is not Linux x86_64", "evidence": row.get("resolution-markers", [])}})
        if (row["name"], row["version"]) not in reviews:
            failures.append(f"package review evidence missing: {ident(row['name'], row['version'])}")
    if virtual:
        packages.append({"lock": virtual[0], "installed": None, "review": reviews.get((virtual[0]["name"], virtual[0]["version"])),
                         "activity": {"status": "INACTIVE_VIRTUAL_PROJECT", "reason": "source={virtual='.'}; not an installed distribution", "evidence": "excluded from installed multiset"}})
        if (virtual[0]["name"], virtual[0]["version"]) not in reviews:
            failures.append(f"package review evidence missing: {ident(virtual[0]['name'], virtual[0]['version'])}")
    if not closure["exact"]: failures.append("installed distribution multiset does not exactly match uv.lock")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}: failures.append("audit host is not Linux x86_64")
    if sys.version_info[:2] != (3, 12): failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    model, model_failures = [], []
    if fetch_model_license:
        try: model.append(fetch_license(license_fetcher))
        except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
            model.append(blocked_license(exc)); model_failures.append(f"BLOCKED_FACTUAL_LICENSE_PATH: {SOURCE['repo']}@{SOURCE['revision']}: {canonical(controlled_error(exc))}")
    failures.extend(model_failures)
    dependency_requests = []
    for item in packages:
        evidence = item.get("installed", {}).get("sdist_license_evidence") if isinstance(item.get("installed"), dict) else None
        if evidence is not None: dependency_requests.append({"package": item["lock"]["name"], "version": item["lock"]["version"], "url": evidence.get("requested_url", evidence.get("archive", {}).get("url")), "status": evidence.get("status")})
    return {"schema": SCHEMA, "status": "BLOCKED" if failures else "PASS", "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True, "model_code_imported": False, "cargo_invoked": False},
            "project": {"name": pd["project"]["name"], "version": pd["project"]["version"], "pyproject_bytes": len(pb), "pyproject_sha256": sha(pb), "uv_lock_bytes": len(lb), "uv_lock_sha256": sha(lb)},
            "git": git_identity(project), "locked_rows": sorted(lock["package"], key=lambda r: (norm_name(r["name"]), norm_version(r["version"]))),
            "approval_state": approval_state(manifest),
            "active_lock_rows": active_real, "inactive_rows": [{"row": r, "status": "INACTIVE_LOCK_ROW", "reason": "resolution marker is not Linux x86_64", "evidence": r.get("resolution-markers", [])} for r in inactive_real],
            "virtual_project_row": {"row": virtual[0] if virtual else None, "status": "INACTIVE_VIRTUAL_PROJECT", "reason": "virtual project row is not installed"},
            "active_installed_rows": [{"name": r["name"], "version": r["version"], "normalized_identity": r["identity"], "location": r["location"]} for r in records], "closure": closure, "packages": packages,
            "source_license_contract": {"status": "EXACT_PINNED_SOURCE_LICENSE_PATH", "repo": SOURCE["repo"], "revision": SOURCE["revision"], "file": "LICENSE"}, "model_license_files": model,
            "model_acquisition": {"policy": "exact source LICENSE only", "requested_files": [license_url()] if fetch_model_license else [], "non_license_files": [], "non_license_requests": [], "proof": "no model-weight request/import path exists"},
            "dependency_acquisition": {"scope": "exact locked PyPI sdist license evidence only", "requests": dependency_requests, "out_of_scope_requests": [], "model_files": [], "non_license_files": []}, "failures": sorted(set(failures))}


def run(project: Path, output: Path, fetch_model_license: bool) -> int:
    try: report = audit_environment(project, fetch_model_license)
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        report = {"schema": SCHEMA, "status": "BLOCKED", "environment": {"model_code_imported": False, "cargo_invoked": False}, "approval_state": blocked_approval_state(project), "locked_rows": [], "packages": [], "failures": [str(exc)]}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(canonical(report) + "\n", encoding="utf-8")
    if report["failures"]: print("moss dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr); return 2
    print(f"moss dependency audit: PASS ({output})"); return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    _, _, manifest, rows, _, _ = contract(project)
    assert len(rows) == 52 and len(manifest["package_review_rows"]) == 52
    state = approval_state(manifest)
    assert len(state["license_rows"]) == 3 and state["approval"]["status"] == "OWNER_SIGNOFF_REQUIRED"
    assert state["publication_decision"] == "NO_UPLOAD" and state["publication_permitted"] is False
    assert state["approval_blockers"] and state["package_review_blockers"] and state["license_review_blockers"]
    assert compare_multiset(["a==1"], ["a==1"])["exact"]
    mismatch = compare_multiset(["a==1"], ["a==2"]); assert mismatch["missing"] == ["a==1"] and mismatch["unexpected"] == ["a==2"]
    duplicate = compare_multiset(["a==1"], ["a==1", "a==1"]); assert duplicate["duplicate_identities"] == ["a==1"]
    assert not is_license_path("unlicensed-file") and not is_license_path("project-license")
    assert is_license_path("LICENSE") and is_license_path("LICENSE.txt") and is_license_path("COPYING-extra")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:gz") as t:
        info = tarfile.TarInfo("demo/LICENSE"); data = b"Apache\n"; info.size = len(data); t.addfile(info, io.BytesIO(data))
    raw = body.getvalue(); row = {"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": {"url": f"https://{PYPI_HOST}/packages/demo-1.tar.gz", "hash": "sha256:"+sha(raw), "size": len(raw), "upload-time": "2026-01-01T00:00:00Z"}}
    got = fetch_locked_sdist(row, lambda u: ("demo-1.tar.gz", raw)); assert got["license_files"][0]["sha256"] == sha(b"Apache\n")
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("demo-1/COPYING", b"copying\n")
        z.writestr("demo-1/README", b"not license\n")
    zip_row = dict(row); zip_row["sdist"] = {**row["sdist"], "url": f"https://{PYPI_HOST}/packages/demo-1.zip", "hash": "sha256:"+sha(zipped.getvalue()), "size": len(zipped.getvalue())}
    assert fetch_locked_sdist(zip_row, lambda u: (u, zipped.getvalue()))["license_files"][0]["path"].endswith("COPYING")
    traversal = io.BytesIO()
    with tarfile.open(fileobj=traversal, mode="w:gz") as t:
        info = tarfile.TarInfo("../LICENSE"); info.size = 1; t.addfile(info, io.BytesIO(b"x"))
    traversal_row = dict(row); traversal_row["sdist"] = {**row["sdist"], "hash": "sha256:"+sha(traversal.getvalue()), "size": len(traversal.getvalue())}
    try: fetch_locked_sdist(traversal_row, lambda u: (u, traversal.getvalue()))
    except SdistError as exc: assert exc.acquired is True and exc.verified is True
    else: raise AssertionError("accepted archive traversal")
    for key in ("url", "hash", "size", "upload-time"):
        bad = dict(row); bad["sdist"] = dict(row["sdist"]); bad["sdist"].pop(key)
        try: fetch_locked_sdist(bad, lambda u: (u, raw))
        except SdistError: pass
        else: raise AssertionError("accepted incomplete artifact")
    for final in ("https://evil.invalid/demo.tar.gz", row["sdist"]["url"]+"?x=1"):
        try: fetch_locked_sdist(row, lambda u, final=final: (final, raw))
        except AuditError: pass
        else: raise AssertionError("accepted unsafe redirect")
    good = license_url(); cdn = f"https://cdn-lfs.huggingface.co/{SOURCE['repo']}/resolve/{SOURCE['revision']}/LICENSE"
    validate_license_url(good, good); validate_license_url(cdn, good)
    assert fetch_license(lambda _url: ("LICENSE", b"source\n"))["final_url"] == good
    for bad in (good.replace("/LICENSE", "/model.safetensors"), f"{good}?x=1", f"https://{SOURCE['repo']}/LICENSE", cdn+".txt"):
        try: validate_license_url(bad, good)
        except AuditError: pass
        else: raise AssertionError("accepted unsafe LICENSE URL")
    print("moss dependency_audit self-test: OK"); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--fetch-model-license", action="store_true"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project or args.output or args.fetch_model_license: parser.error("--self-test accepts no project/output/fetch options")
        return self_test()
    if args.project is None or args.output is None: parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_license)


if __name__ == "__main__": raise SystemExit(main())
