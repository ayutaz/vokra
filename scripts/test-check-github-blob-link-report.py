#!/usr/bin/env -S uv run --script
"""Offline stdlib tests for check-github-blob-link-report.py."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("check-github-blob-link-report.py")
SPEC = importlib.util.spec_from_file_location("github_blob_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


URL = "https://github.com/ayutaz/vokra/blob/main/docs/guide.md#L2-L3"
CONTENT = b"zero\nfirst\nsecond\n"


def _body(url: str = URL, code: int | None = 503) -> dict:
    status = {"text": "503 Service Unavailable"} if code is None else {"text": f"{code} status", "code": code}
    return {
        "url": url,
        "status": status,
        "span": {"line": 1, "column": 1},
        "duration": {"secs": 1, "nanos": 0},
    }


def _report(bodies: list[dict] | None = None, *, errors: int | None = None, total: int = 1) -> dict:
    bodies = [] if bodies is None else bodies
    count = len(bodies) if errors is None else errors
    return {
        "total": total,
        "unique": total,
        "successful": total - count,
        "unknown": 0,
        "unsupported": 0,
        "timeouts": 0,
        "redirects": 0,
        "remaps": 0,
        "excludes": 0,
        "errors": count,
        "cached": 0,
        "success_map": {},
        "error_map": {"README.md": bodies} if bodies else {},
        "timeout_map": {},
        "suggestion_map": {},
        "redirect_map": {},
        "excluded_map": {},
        "duration": {"secs": 0, "nanos": 0},
        "detailed_stats": True,
    }


class Response:
    status = 200

    def __init__(
        self,
        payload: bytes,
        url: str,
        *,
        content_length: str | None = None,
        status: int = 200,
    ):
        self._payload = payload
        self._url = url
        self.status = status
        self.closed = False
        self.headers = {"Content-Length": content_length or str(len(payload))}

    def read(self, limit: int = -1) -> bytes:
        return self._payload if limit < 0 else self._payload[:limit]

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        self.closed = True


class Opener:
    def __init__(
        self,
        payload: dict,
        *,
        redirect_url: str | None = None,
        status: int = 200,
        ref_payload: list[dict] | None = None,
        tag_ref_payload: list[dict] | None = None,
        ref_link: str | None = None,
    ):
        self.payload = payload
        self.redirect_url = redirect_url
        self.status = status
        self.ref_payload = ref_payload
        self.tag_ref_payload = tag_ref_payload
        self.ref_link = ref_link
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append((request, timeout))
        if "/git/matching-refs/" in request.full_url:
            kind = request.full_url.split("/git/matching-refs/", 1)[1].split("/", 1)[0]
            if self.ref_payload is not None:
                response_payload = self.tag_ref_payload if kind == "tags" and self.tag_ref_payload is not None else self.ref_payload
            else:
                response_payload = [{"ref": "refs/heads/main"}] if kind == "heads" else []
        else:
            response_payload = self.payload
        raw = json.dumps(response_payload, separators=(",", ":")).encode()
        response = Response(raw, self.redirect_url or request.full_url, status=self.status)
        if self.ref_link is not None and "/git/matching-refs/" in request.full_url:
            response.headers["Link"] = self.ref_link
        return response


def _payload(url: str = URL, content: bytes = CONTENT) -> dict:
    target = gate.parse_blob_target(url)
    api_url = f"https://api.github.com/repos/{target.owner}/{target.repo}/contents/{target.path}?ref={target.ref}"
    sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    return {
        "type": "file",
        "encoding": "base64",
        "size": len(content),
        "name": target.path.rsplit("/", 1)[-1],
        "path": target.path,
        "sha": sha,
        "content": base64.b64encode(content).decode(),
        "url": api_url,
        "html_url": url.split("#", 1)[0],
        "git_url": f"https://api.github.com/repos/{target.owner}/{target.repo}/git/blobs/{sha}",
        "download_url": f"https://raw.githubusercontent.com/{target.owner}/{target.repo}/{target.ref}/{target.path}",
        "_links": {
            "self": api_url,
            "git": f"https://api.github.com/repos/{target.owner}/{target.repo}/git/blobs/{sha}",
            "html": url.split("#", 1)[0],
        },
    }


class GateTests(unittest.TestCase):
    def test_503_blob_recovery_checks_exact_api_request_and_token_header(self):
        opener = Opener(_payload())
        self.assertTrue(gate.check_report(_report([_body()]), 2, token="secret", opener=opener))
        self.assertEqual(len(opener.requests), 3)
        request, timeout = opener.requests[0]
        self.assertEqual(request.full_url, "https://api.github.com/repos/ayutaz/vokra/contents/docs/guide.md?ref=main")
        self.assertTrue(all(item[0].headers["Authorization"] == "Bearer secret" for item in opener.requests))
        self.assertLessEqual(timeout, 10)

    def test_v0242_optional_span_and_duration_and_unknown_status_shape(self):
        body = _body()
        body.pop("span")
        body.pop("duration")
        non_detailed = _report([body])
        non_detailed["detailed_stats"] = False
        self.assertEqual(len(gate.validate_report(non_detailed, 2)[0]), 1)
        unknown = _body(code=999)
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([unknown]), 2, opener=Opener(_payload()), token="secret")

    def test_recovery_requires_token_and_closes_response(self):
        opener = Opener(_payload())
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([_body()]), 2, opener=opener)
        self.assertEqual(len(opener.requests), 0)

        class Capturing(Opener):
            def open(self, request, timeout):
                response = super().open(request, timeout)
                self.response = response
                return response

        capturing = Capturing(_payload())
        gate.check_report(_report([_body()]), 2, opener=capturing, token="secret")
        self.assertTrue(capturing.response.closed)
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([_body()]), 2, opener=Opener(_payload()), token="bad\nheader")

    def test_success_exit_does_not_call_api_without_token(self):
        class Never:
            def open(self, *_args, **_kwargs):
                raise AssertionError("unexpected API request")

        self.assertTrue(gate.check_report(_report([], errors=0), 0, opener=Never()))

    def test_non_503_and_mixed_fail_closed(self):
        for bodies in ([_body(code=404)], [_body(), _body(url="https://example.invalid/bad", code=503)]):
            report = _report(bodies, total=len(bodies))
            with self.assertRaises(gate.GateError):
                gate.check_report(report, 2, opener=Opener(_payload()))

    def test_wrong_path_ref_hash_line_and_unsupported_fragment_fail(self):
        cases = [
            ({**_payload(), "path": "docs/other.md"}, "API path mismatch"),
            ({**_payload(), "html_url": "https://github.com/ayutaz/vokra/blob/other/docs/guide.md"}, "API ref identity mismatch"),
            ({**_payload(), "sha": "0" * 40}, "API blob identity mismatch"),
            (_payload(content=b"one\n"), "GitHub line anchor is absent"),
        ]
        for payload, message in cases:
            opener = Opener(payload)
            with self.assertRaisesRegex(gate.GateError, message):
                gate.check_report(_report([_body()]), 2, opener=opener, token="test-token")
            self.assertEqual(len(opener.requests), 1)
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([_body(URL.replace("#L2-L3", "#readme"))]), 2, opener=Opener(_payload()))

        unsafe_urls = (
            "https://github.com/ayutaz/vokra/blob/main/docs/./guide.md",
            "https://github.com/ayutaz/vokra/blob/main/docs/../guide.md",
            "https://github.com/ayutaz/vokra/blob/main/docs/bad\\name.md",
            "https://github.com/ayutaz/vokra/blob/main/docs/bad\nname.md",
        )
        for unsafe_url in unsafe_urls:
            with self.assertRaises(gate.GateError):
                gate.parse_blob_target(unsafe_url)

        vertical_break = _payload(content=b"one\vtwo\n")
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([_body()]), 2, opener=Opener(vertical_break), token="secret")

    def test_ambiguous_ref_redirect_and_multiple_ports_are_rejected(self):
        with self.assertRaises(gate.GateError):
            gate.parse_blob_target("https://github.com/ayutaz/vokra/blob/refs/heads/main/docs/guide.md")
        opener = Opener(_payload(), redirect_url="https://api.github.com/redirect")
        with self.assertRaisesRegex(gate.GateError, "API redirect rejected"):
            gate.check_report(_report([_body()]), 2, opener=opener, token="test-token")
        self.assertEqual(len(opener.requests), 1)
        with self.assertRaises(gate.GateError):
            gate.parse_blob_target("https://github.com/ayutaz/vokra/blob/main/docs/guide.md?x=1")
        with self.assertRaises(gate.GateError):
            gate.parse_blob_target("https://user:pass@github.com/ayutaz/vokra/blob/main/docs/guide.md")
        with self.assertRaises(gate.GateError):
            gate.parse_blob_target("https://github.com/ayutaz/vokra/blob/main/docs%2Fguide.md")
        ambiguous = Opener(
            _payload(),
            ref_payload=[{"ref": "refs/heads/main"}, {"ref": "refs/heads/main/feature"}],
        )
        with self.assertRaisesRegex(gate.GateError, "ambiguous GitHub ref"):
            gate.check_report(_report([_body()]), 2, opener=ambiguous, token="test-token")
        self.assertEqual(len(ambiguous.requests), 2)
        missing_ref = Opener(_payload(), ref_payload=[])
        with self.assertRaisesRegex(gate.GateError, "GitHub ref evidence missing exact ref"):
            gate.check_report(_report([_body()]), 2, opener=missing_ref, token="test-token")
        self.assertEqual(len(missing_ref.requests), 3)
        collision = Opener(
            _payload(),
            ref_payload=[{"ref": "refs/heads/main"}],
            tag_ref_payload=[{"ref": "refs/tags/main"}],
        )
        with self.assertRaisesRegex(gate.GateError, "ambiguous GitHub ref"):
            gate.check_report(_report([_body()]), 2, opener=collision, token="test-token")
        self.assertEqual(len(collision.requests), 3)
        paginated = Opener(_payload(), ref_link='<https://api.github.com/...>; rel="next"')
        with self.assertRaisesRegex(gate.GateError, "GitHub ref evidence is paginated"):
            gate.check_report(_report([_body()]), 2, opener=paginated, token="test-token")
        self.assertEqual(len(paginated.requests), 2)

    def test_api_oversize_and_noncanonical_content_are_rejected(self):
        class Oversize:
            def __init__(self):
                self.requests = []

            def open(self, request, timeout):
                self.requests.append(request)
                return Response(b"{}", request.full_url, content_length=str(gate.MAX_API_BYTES + 1))

        oversize = Oversize()
        with self.assertRaisesRegex(gate.GateError, "API response exceeds size limit"):
            gate.check_report(_report([_body()]), 2, opener=oversize, token="test-token")
        self.assertEqual(len(oversize.requests), 1)
        payload = _payload()
        payload["content"] = base64.b64encode(CONTENT).decode() + " "
        with self.assertRaisesRegex(gate.GateError, "invalid API content encoding"):
            gate.check_report(_report([_body()]), 2, opener=Opener(payload), token="test-token")

        noncanonical = _payload()
        noncanonical["content"] = "Zh=="  # decodes to b"f" with non-zero pad bits
        noncanonical["size"] = 1
        noncanonical["sha"] = hashlib.sha1(b"blob 1\0f").hexdigest()
        with self.assertRaisesRegex(gate.GateError, "noncanonical API content encoding"):
            gate.check_report(_report([_body()]), 2, opener=Opener(noncanonical), token="test-token")

        for field in ("target", "submodule_git_url"):
            non_file = _payload()
            non_file[field] = "https://example.invalid/target"
            opener = Opener(non_file)
            with self.assertRaisesRegex(gate.GateError, "API response contains non-file identity"):
                gate.check_report(_report([_body()]), 2, opener=opener, token="test-token")
            self.assertEqual(len(opener.requests), 1)

        symlink = _payload()
        symlink["type"] = "symlink"
        with self.assertRaisesRegex(gate.GateError, "API response is not a file"):
            gate.check_report(_report([_body()]), 2, opener=Opener(symlink), token="test-token")

        expired = Opener(_payload())
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([_body()]), 2, opener=expired, token="secret", deadline_seconds=0.000001)

    def test_http_failure_statuses_are_not_recovered(self):
        for status in (403, 404, 429):
            opener = Opener({}, status=status)
            with self.assertRaisesRegex(gate.GateError, "API response was not successful"):
                gate.check_report(_report([_body()]), 2, opener=opener, token="test-token")
            self.assertEqual(len(opener.requests), 1)

    def test_fallback_limit_and_after_read_deadline(self):
        bodies = [
            _body(f"https://github.com/ayutaz/vokra/blob/main/docs/f{index}.md")
            for index in range(gate.MAX_FALLBACK_URLS + 1)
        ]
        opener = Opener(_payload())
        with self.assertRaisesRegex(gate.GateError, "fallback URL limit exceeded"):
            gate.check_report(_report(bodies, total=len(bodies)), 2, opener=opener, token="test-token")
        self.assertEqual(len(opener.requests), 0)

        response_opener = Opener(_payload())
        with mock.patch.object(gate.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.0, 2.0]):
            with self.assertRaisesRegex(gate.GateError, "fallback deadline exceeded"):
                gate.check_report(
                    _report([_body()]),
                    2,
                    opener=response_opener,
                    token="test-token",
                    deadline_seconds=1.0,
                )
        self.assertEqual(len(response_opener.requests), 1)

    def test_report_schema_rejects_missing_duplicate_nonfinite_and_too_large(self):
        missing = _report([_body()])
        del missing["errors"]
        with self.assertRaises(gate.GateError):
            gate.validate_report(missing, 2)
        dedup = _report([_body()], errors=2, total=2)
        self.assertEqual(len(gate.validate_report(dedup, 2)[0]), 1)
        overfull = _report([_body(), _body(URL.replace("guide", "other"))], errors=1, total=2)
        with self.assertRaisesRegex(gate.GateError, "response map exceeds status counter"):
            gate.validate_report(overfull, 2)
        with self.assertRaises(gate.GateError):
            gate.load_json_bytes(b'{"total":1,"total":1}')
        with self.assertRaises(gate.GateError):
            gate.load_json_bytes(b'{"total":NaN}')
        with self.assertRaises(gate.GateError):
            gate.load_json_bytes(b'{"total":1e999}')
        with self.assertRaises(gate.GateError):
            gate.load_json_bytes(b"x" * (gate.MAX_REPORT_BYTES + 1))

    def test_timeout_and_runtime_exit_are_not_recovered(self):
        report = _report([], errors=0, total=1)
        report["successful"] = 0
        report["timeouts"] = 1
        report["total"] = report["unique"] = 1
        report["timeout_map"] = {"README.md": [_body(code=None)]}
        report["errors"] = 0
        for exit_code in (1, 3):
            with self.assertRaises(gate.GateError):
                gate.check_report(_report([_body()]), exit_code, opener=Opener(_payload()))
        with self.assertRaises(gate.GateError):
            gate.check_report(report, 2, opener=Opener(_payload()))

    def test_api_duplicate_json_and_credential_never_appear_in_cli_output(self):
        class DuplicateOpener:
            def __init__(self):
                self.requests = []

            def open(self, request, timeout):
                self.requests.append(request)
                raw = b'{"type":"file","type":"file"}'
                return Response(raw, request.full_url)

        duplicate = DuplicateOpener()
        with self.assertRaises(gate.GateError):
            gate.check_report(_report([_body()]), 2, opener=duplicate, token="top-secret")
        self.assertEqual(len(duplicate.requests), 1)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                self.assertEqual(gate.main(["--report", str(report_path), "--lychee-exit-code", "2"]), 1)
            self.assertNotIn("top-secret", output.getvalue())

    def test_main_request_error_is_generic_and_never_prints_token_or_headers(self):
        class ErrorOpener:
            def __init__(self):
                self.requests = []

            def open(self, request, timeout):
                self.requests.append(request)
                raise RuntimeError("sentinel-token Authorization: Bearer secret")

        error_opener = ErrorOpener()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text(json.dumps(_report([_body()])), encoding="utf-8")
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "sentinel-token"}, clear=False):
                with contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
                    self.assertEqual(
                        gate.main(
                            ["--report", str(report_path), "--lychee-exit-code", "2"],
                            opener=error_opener,
                        ),
                        1,
                    )
            self.assertEqual(len(error_opener.requests), 1)
            self.assertEqual(output.getvalue(), "GitHub blob link gate failed\n")

    def test_main_rejects_missing_empty_malformed_and_inconsistent_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "missing": None,
                "empty": b"",
                "malformed": b"{not-json",
                "exit255": json.dumps(_report([_body()])).encode(),
                "exit0-errors": json.dumps(_report([_body()])).encode(),
            }
            for name, raw in cases.items():
                report_path = root / f"{name}.json"
                if raw is not None:
                    report_path.write_bytes(raw)
                exit_code = 255 if name == "exit255" else 0 if name == "exit0-errors" else 2
                if name == "exit0-errors":
                    exit_code = 0
                self.assertEqual(
                    gate.main(["--report", str(report_path), "--lychee-exit-code", str(exit_code)]),
                    1,
                    name,
                )


if __name__ == "__main__":
    unittest.main()
