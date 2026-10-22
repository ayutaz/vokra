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
VIBEVOICE_TREE_URL = (
    "https://api.github.com/repos/microsoft/VibeVoice/git/trees/"
    "94da20d98b2fa7688e9cbfaf7692ddb4954f7600?recursive=1"
)
UV_RELEASE_URL = "https://api.github.com/repos/astral-sh/uv/releases/tags/0.12.5"
PYTORCH_COMPLEX_TREE_URL = (
    "https://api.github.com/repos/kamo-naoyuki/pytorch_complex/git/trees/"
    "8a2ad1e47f3df25a30eb426f6ad781b89103fab3?recursive=1"
)
QWEN_TREE_URL = (
    "https://api.github.com/repos/QwenLM/Qwen2-Audio/git/trees/"
    "0d66378d9e4b1a4d6bdba3969aa87ef777f487cd?recursive=1"
)
TREE_FIXTURES = {
    VIBEVOICE_TREE_URL: ("94da20d98b2fa7688e9cbfaf7692ddb4954f7600", 121),
    PYTORCH_COMPLEX_TREE_URL: ("8a2ad1e47f3df25a30eb426f6ad781b89103fab3", 17),
    QWEN_TREE_URL: ("0d66378d9e4b1a4d6bdba3969aa87ef777f487cd", 23),
}


def _body(url: str = URL, code: int | None = 503) -> dict:
    status = {"text": "503 Service Unavailable"} if code is None else {"text": f"{code} status", "code": code}
    return {
        "url": url,
        "status": status,
        "span": {"line": 1, "column": 1},
        "duration": {"secs": 1, "nanos": 0},
    }


def _api_body(url: str, code: int = 403) -> dict:
    return _body(url, code)


def _report(
    bodies: list[dict] | None = None,
    *,
    errors: int | None = None,
    total: int = 1,
    unique: int | None = None,
    cached: int = 0,
    successful: int | None = None,
    unsupported: int = 0,
    redirects: int = 0,
    excludes: int = 0,
) -> dict:
    bodies = [] if bodies is None else bodies
    count = len(bodies) if errors is None else errors
    return {
        "total": total,
        "unique": total if unique is None else unique,
        "successful": total - count if successful is None else successful,
        "unknown": 0,
        "unsupported": unsupported,
        "timeouts": 0,
        "redirects": redirects,
        "remaps": 0,
        "excludes": excludes,
        "errors": count,
        "cached": cached,
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


class ApiMappingOpener:
    def __init__(self, payloads: dict[str, object], *, status: int = 200, raw: bytes | None = None, response_url: str | None = None, link: str | None = None):
        self.payloads = payloads
        self.status = status
        self.raw = raw
        self.response_url = response_url
        self.link = link
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append((request, timeout))
        raw = self.raw
        if raw is None:
            raw = json.dumps(self.payloads[request.full_url], separators=(",", ":")).encode()
        response = Response(raw, self.response_url or request.full_url, status=self.status)
        if self.link is not None:
            response.headers["Link"] = self.link
        return response


def _tree_payload(url: str, *, count: int | None = None, sha: str | None = None, truncated: bool = False) -> dict:
    expected_sha, expected_count = TREE_FIXTURES[url]
    return {
        "sha": expected_sha if sha is None else sha,
        "truncated": truncated,
        "tree": [
            {
                "path": f"fixture-entry-{index}",
                "mode": "100644",
                "type": "blob",
                "sha": f"{index + 1:040x}",
                "size": 0,
            }
            for index in range(expected_count if count is None else count)
        ],
    }


def _release_payload(*, release_id: int = 370794065, tag_name: str = "0.12.5", html_url: str = "https://github.com/astral-sh/uv/releases/tag/0.12.5") -> dict:
    return {"id": release_id, "tag_name": tag_name, "html_url": html_url}


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
    def test_exact_403_api_rechecks_are_authenticated_and_deduplicated(self):
        bodies = [
            _api_body(VIBEVOICE_TREE_URL),
            _api_body(UV_RELEASE_URL),
            _api_body(PYTORCH_COMPLEX_TREE_URL),
            _api_body(PYTORCH_COMPLEX_TREE_URL),
        ]
        bodies[3]["status"]["text"] = "Error (cached)"
        bodies[3].pop("span")
        bodies[3].pop("duration")
        report = _report(
            bodies,
            total=1669,
            unique=769,
            cached=60,
            successful=1652,
            unsupported=2,
            redirects=4,
            excludes=11,
        )
        report["error_map"] = {
            "docs/handoff/public-catalog-security-completion-2026-09-29.md": bodies[:3],
            "docs/handoff/owsm-primary-license-readback-2026-10-18.md": [bodies[3]],
        }
        opener = ApiMappingOpener(
            {
                VIBEVOICE_TREE_URL: _tree_payload(VIBEVOICE_TREE_URL),
                UV_RELEASE_URL: _release_payload(),
                PYTORCH_COMPLEX_TREE_URL: _tree_payload(PYTORCH_COMPLEX_TREE_URL),
            }
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(gate.check_report(report, 2, token="fake-token", opener=opener))
        self.assertEqual(len(opener.requests), 3)
        self.assertTrue(all(request.headers["Authorization"] == "Bearer fake-token" for request, _ in opener.requests))
        self.assertNotIn("fake-token", output.getvalue())
        self.assertIn("entries=121", output.getvalue())
        self.assertIn("entries=17", output.getvalue())
        self.assertIn("release id=370794065 tag=0.12.5", output.getvalue())

    def test_actual_five_error_report_rechecks_four_literal_targets(self):
        """Mirror PR283's five 403 bodies, including one cached duplicate."""
        bodies = [
            _api_body(VIBEVOICE_TREE_URL),
            _api_body(UV_RELEASE_URL),
            _api_body(PYTORCH_COMPLEX_TREE_URL),
            _api_body(PYTORCH_COMPLEX_TREE_URL),
            _api_body(QWEN_TREE_URL),
        ]
        bodies[0]["status"]["text"] = "Rejected status code: 403 Forbidden"
        bodies[0]["span"] = {"line": 452, "column": 13}
        bodies[0]["duration"] = {"secs": 0, "nanos": 253194767}
        bodies[1]["status"]["text"] = "Rejected status code: 403 Forbidden"
        bodies[1]["span"] = {"line": 5513, "column": 18}
        bodies[1]["duration"] = {"secs": 0, "nanos": 251212163}
        bodies[2]["status"]["text"] = "Rejected status code: 403 Forbidden"
        bodies[2]["span"] = {"line": 6210, "column": 2}
        bodies[2]["duration"] = {"secs": 0, "nanos": 328401735}
        bodies[3]["status"]["text"] = "Error (cached)"
        bodies[3]["span"] = {"line": 19, "column": 5}
        bodies[3].pop("duration")
        bodies[4]["status"]["text"] = "Rejected status code: 403 Forbidden"
        bodies[4]["span"] = {"line": 25, "column": 3}
        bodies[4]["duration"] = {"secs": 0, "nanos": 156249638}
        report = _report(
            bodies,
            total=1669,
            unique=769,
            cached=59,
            successful=1651,
            unsupported=2,
            redirects=4,
            excludes=11,
        )
        report["detailed_stats"] = False
        report["duration"] = {"secs": 207, "nanos": 494571936}
        report["error_map"] = {
            "docs/handoff/owsm-primary-license-readback-2026-10-18.md": [bodies[3]],
            "docs/handoff/public-catalog-security-completion-2026-09-29.md": [bodies[1], bodies[2], bodies[0]],
            "docs/handoff/qwen2-audio-fixed-source-license-readback-2026-10-15.md": [bodies[4]],
        }
        opener = ApiMappingOpener(
            {
                VIBEVOICE_TREE_URL: _tree_payload(VIBEVOICE_TREE_URL),
                UV_RELEASE_URL: _release_payload(),
                PYTORCH_COMPLEX_TREE_URL: _tree_payload(PYTORCH_COMPLEX_TREE_URL),
                QWEN_TREE_URL: _tree_payload(QWEN_TREE_URL),
            }
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(gate.check_report(report, 2, token="fake-token", opener=opener))
        self.assertEqual(len(opener.requests), 4)
        self.assertTrue(all(request.headers["Authorization"] == "Bearer fake-token" for request, _ in opener.requests))
        self.assertNotIn("fake-token", output.getvalue())
        self.assertIn("entries=23", output.getvalue())

    def test_api_recheck_accepts_only_literal_https_targets(self):
        for url in (VIBEVOICE_TREE_URL, UV_RELEASE_URL, PYTORCH_COMPLEX_TREE_URL, QWEN_TREE_URL):
            variants = (
                url.replace("https://", "http://", 1),
                url.replace("api.github.com", "www.github.com", 1),
                url.replace("api.github.com", "api.github.com:443", 1),
                url.replace("https://api.github.com", "https://user:pass@api.github.com", 1),
                f"{url}#fragment",
                f"{url}&page=2" if "?" in url else f"{url}?page=2",
            )
            for variant in variants:
                with self.assertRaises(gate.GateError):
                    gate.parse_api_recheck_target(variant)
        with self.assertRaises(gate.GateError):
            gate.parse_api_recheck_target(QWEN_TREE_URL.replace("0d66378d9e4b1a4d6bdba3969aa87ef777f487cd", "0" * 40))

    def test_qwen_tree_identity_failures_are_fail_closed(self):
        malformed = _tree_payload(QWEN_TREE_URL)
        malformed["tree"][0] = {"path": "", "mode": "100644", "type": "blob", "sha": "0" * 40}
        cases = (
            (_tree_payload(QWEN_TREE_URL, sha="0" * 40), "API tree SHA mismatch"),
            (_tree_payload(QWEN_TREE_URL, count=22), "API tree entry count mismatch"),
            (_tree_payload(QWEN_TREE_URL, truncated=True), "API tree response is truncated"),
            (malformed, "invalid API tree entry"),
        )
        for payload, message in cases:
            opener = ApiMappingOpener({QWEN_TREE_URL: payload})
            with self.assertRaisesRegex(gate.GateError, message):
                gate.check_report(_report([_api_body(QWEN_TREE_URL)]), 2, token="fake-token", opener=opener)
            self.assertEqual(len(opener.requests), 1)

    def test_original_non_403_qwen_status_is_rejected_without_request(self):
        for code in (200, 404, 429, 503):
            opener = ApiMappingOpener({QWEN_TREE_URL: _tree_payload(QWEN_TREE_URL)})
            expected = "unsupported GitHub blob URL" if code == 503 else "unhandled non-recoverable link failure"
            with self.assertRaisesRegex(gate.GateError, expected):
                gate.check_report(_report([_api_body(QWEN_TREE_URL, code)], total=1), 2, token="fake-token", opener=opener)
            self.assertEqual(opener.requests, [])

    def test_api_recheck_verifies_tree_and_release_identity(self):
        malformed_entry = _tree_payload(VIBEVOICE_TREE_URL)
        malformed_entry["tree"][0] = {"path": "", "mode": "100644", "type": "blob", "sha": "0" * 40}
        duplicate_entries = _tree_payload(VIBEVOICE_TREE_URL)
        duplicate_entries["tree"][1]["path"] = duplicate_entries["tree"][0]["path"]
        invalid_size = _tree_payload(VIBEVOICE_TREE_URL)
        invalid_size["tree"][0]["size"] = -1
        null_size = _tree_payload(VIBEVOICE_TREE_URL)
        null_size["tree"][0]["size"] = None
        bool_size = _tree_payload(VIBEVOICE_TREE_URL)
        bool_size["tree"][0]["size"] = True
        cases = (
            (_tree_payload(VIBEVOICE_TREE_URL, sha="0" * 40), "API tree SHA mismatch"),
            (_tree_payload(VIBEVOICE_TREE_URL, count=120), "API tree entry count mismatch"),
            (_tree_payload(VIBEVOICE_TREE_URL, truncated=True), "API tree response is truncated"),
            (malformed_entry, "invalid API tree entry"),
            (duplicate_entries, "invalid API tree entry"),
            (invalid_size, "invalid API tree entry"),
            (null_size, "invalid API tree entry"),
            (bool_size, "invalid API tree entry"),
            (_release_payload(release_id=1), "API release identity mismatch"),
            (_release_payload(release_id=370794065.0), "API release identity mismatch"),
            (_release_payload(tag_name="0.12.4"), "API release identity mismatch"),
            (_release_payload(html_url="https://github.com/astral-sh/uv/releases/tag/other"), "API release identity mismatch"),
        )
        urls = (
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            VIBEVOICE_TREE_URL,
            UV_RELEASE_URL,
            UV_RELEASE_URL,
            UV_RELEASE_URL,
            UV_RELEASE_URL,
        )
        for (payload, message), url in zip(cases, urls):
            with self.assertRaisesRegex(gate.GateError, message):
                gate.check_report(_report([_api_body(url)]), 2, token="fake-token", opener=ApiMappingOpener({url: payload}))

    def test_api_recheck_failure_modes_are_fail_closed(self):
        class RaisingOpener:
            def __init__(self, exc):
                self.exc = exc
                self.requests = []

            def open(self, request, timeout):
                self.requests.append((request, timeout))
                raise self.exc

        failures = (
            (403, None, "API response was not successful"),
            (404, None, "API response was not successful"),
            (200, "redirect", "API redirect rejected"),
        )
        for status, response_url, message in failures:
            opener = ApiMappingOpener(
                {VIBEVOICE_TREE_URL: _tree_payload(VIBEVOICE_TREE_URL)},
                status=status,
                response_url=response_url,
            )
            with self.assertRaisesRegex(gate.GateError, message):
                gate.check_report(_report([_api_body(VIBEVOICE_TREE_URL)]), 2, token="fake-token", opener=opener)
            self.assertEqual(len(opener.requests), 1)
        paginated = ApiMappingOpener(
            {VIBEVOICE_TREE_URL: _tree_payload(VIBEVOICE_TREE_URL)},
            link='<https://api.github.com/...>; rel="next"',
        )
        with self.assertRaisesRegex(gate.GateError, "GitHub ref evidence is paginated"):
            gate.check_report(_report([_api_body(VIBEVOICE_TREE_URL)]), 2, token="fake-token", opener=paginated)
        self.assertEqual(len(paginated.requests), 1)

        for exc, message in ((TimeoutError("fake-token"), "API request failed"), (RuntimeError("Bearer fake-token"), "API request failed")):
            opener = RaisingOpener(exc)
            with self.assertRaisesRegex(gate.GateError, message):
                gate.check_report(_report([_api_body(VIBEVOICE_TREE_URL)]), 2, token="fake-token", opener=opener)
            self.assertEqual(len(opener.requests), 1)

        for raw, message in ((b"{bad", "malformed JSON report"), (b'{"sha":1,"sha":2}', "duplicate JSON object key")):
            opener = ApiMappingOpener({}, raw=raw)
            with self.assertRaisesRegex(gate.GateError, message):
                gate.check_report(_report([_api_body(VIBEVOICE_TREE_URL)]), 2, token="fake-token", opener=opener)

        for token in (None, "fake\ntoken"):
            opener = ApiMappingOpener({VIBEVOICE_TREE_URL: _tree_payload(VIBEVOICE_TREE_URL)})
            with self.assertRaises(gate.GateError):
                gate.check_report(_report([_api_body(VIBEVOICE_TREE_URL)]), 2, token=token, opener=opener)
            self.assertEqual(len(opener.requests), 0)

    def test_ineligible_403_and_mixed_errors_never_trigger_recheck(self):
        class Never:
            def __init__(self):
                self.requests = []

            def open(self, *_args, **_kwargs):
                self.requests.append(True)
                raise AssertionError("unexpected API request")

        for url in (
            VIBEVOICE_TREE_URL.replace("recursive=1", "recursive=2"),
            "https://api.github.com/repos/example/project/git/trees/main?recursive=1",
            URL,
        ):
            opener = Never()
            with self.assertRaises(gate.GateError):
                gate.check_report(_report([_api_body(url)]), 2, token="fake-token", opener=opener)
            self.assertEqual(opener.requests, [])

        opener = Never()
        with self.assertRaises(gate.GateError):
            gate.check_report(
                _report([_api_body(VIBEVOICE_TREE_URL), _body(code=404)], total=2),
                2,
                token="fake-token",
                opener=opener,
            )
        self.assertEqual(opener.requests, [])

    def test_shared_budget_and_deadline_cover_both_recovery_branches(self):
        class MixedOpener(Opener):
            def open(self, request, timeout: float):
                if request.full_url == VIBEVOICE_TREE_URL:
                    self.requests.append((request, timeout))
                    raw = json.dumps(_tree_payload(VIBEVOICE_TREE_URL, count=120), separators=(",", ":")).encode()
                    return Response(raw, request.full_url)
                return super().open(request, timeout)

        mixed_report = _report([_body(), _api_body(VIBEVOICE_TREE_URL)], total=2)
        mixed_output = io.StringIO()
        with contextlib.redirect_stdout(mixed_output):
            with self.assertRaisesRegex(gate.GateError, "API tree entry count mismatch"):
                gate.check_report(mixed_report, 2, token="fake-token", opener=MixedOpener(_payload()))
        self.assertNotIn("Recovered original HTTP 503", mixed_output.getvalue())

        too_many = [
            _body(f"https://github.com/ayutaz/vokra/blob/main/docs/f{index}.md")
            for index in range(8)
        ] + [
            _api_body(VIBEVOICE_TREE_URL),
            _api_body(UV_RELEASE_URL),
            _api_body(PYTORCH_COMPLEX_TREE_URL),
        ]
        budget_opener = ApiMappingOpener({})
        with self.assertRaisesRegex(gate.GateError, "fallback URL limit exceeded"):
            gate.check_report(_report(too_many, total=len(too_many)), 2, token="fake-token", opener=budget_opener)
        self.assertEqual(budget_opener.requests, [])

        deadline_opener = ApiMappingOpener(
            {
                VIBEVOICE_TREE_URL: _tree_payload(VIBEVOICE_TREE_URL),
                UV_RELEASE_URL: _release_payload(),
            }
        )
        deadline_output = io.StringIO()
        with mock.patch.object(gate.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.5, 2.0]):
            with contextlib.redirect_stdout(deadline_output):
                with self.assertRaisesRegex(gate.GateError, "fallback deadline exceeded"):
                    gate.check_report(
                        _report([_api_body(VIBEVOICE_TREE_URL), _api_body(UV_RELEASE_URL)], total=2),
                        2,
                        token="fake-token",
                        opener=deadline_opener,
                        deadline_seconds=1.0,
                    )
        self.assertEqual(len(deadline_opener.requests), 1)
        self.assertEqual(deadline_output.getvalue(), "")

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
        # An original 403 on a blob URL is not the bounded API-URL branch.
        opener = Opener({})
        with self.assertRaisesRegex(gate.GateError, "unhandled non-recoverable link failure"):
            gate.check_report(_report([_body(code=403)]), 2, opener=opener, token="test-token")
        self.assertEqual(len(opener.requests), 0)

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

    def test_cli_api_recheck_never_prints_fake_token_and_ineligible_url_is_offline(self):
        class ErrorOpener:
            def __init__(self):
                self.calls = 0

            def open(self, *_args, **_kwargs):
                self.calls += 1
                raise RuntimeError("fake-token Authorization: Bearer fake-token")

        class Never:
            def __init__(self):
                self.calls = 0

            def open(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("unexpected network call")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eligible = root / "eligible.json"
            eligible.write_text(json.dumps(_report([_api_body(VIBEVOICE_TREE_URL)])), encoding="utf-8")
            output = io.StringIO()
            eligible_opener = ErrorOpener()
            with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"}, clear=False):
                with contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
                    self.assertEqual(gate.main(["--report", str(eligible), "--lychee-exit-code", "2"], opener=eligible_opener), 1)
            self.assertEqual(output.getvalue(), "GitHub blob link gate failed\n")
            self.assertNotIn("fake-token", output.getvalue())
            self.assertEqual(eligible_opener.calls, 1)

            ineligible = root / "ineligible.json"
            ineligible.write_text(
                json.dumps(_report([_api_body(VIBEVOICE_TREE_URL.replace("recursive=1", "recursive=2"))])),
                encoding="utf-8",
            )
            ineligible_opener = Never()
            with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"}, clear=False):
                self.assertEqual(gate.main(["--report", str(ineligible), "--lychee-exit-code", "2"], opener=ineligible_opener), 1)
            self.assertEqual(ineligible_opener.calls, 0)

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
