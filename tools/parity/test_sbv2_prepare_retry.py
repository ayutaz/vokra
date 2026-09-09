#!/usr/bin/env python3
"""Network-free tests for SBV2 Hugging Face 429 retry handling."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbv2_prepare_checkpoint as prep  # noqa: E402


class FakeHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        retry_after: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.response = SimpleNamespace(
            status_code=status_code,
            headers=(
                headers
                if headers is not None
                else {} if retry_after is None else {"Retry-After": retry_after}
            ),
        )
        super().__init__(f"HTTP {status_code}")


class RetryAfterTests(unittest.TestCase):
    def test_delta_seconds_header_is_used(self):
        response = SimpleNamespace(headers={"Retry-After": "192"})
        self.assertEqual(prep.retry_after_seconds(response), 192.0)

    def test_rate_limit_reset_header_matches_huggingface_format(self):
        response = SimpleNamespace(headers={"RaTeLiMiT": '"api";r=498;t=192'})
        self.assertEqual(prep.rate_limit_reset_seconds(response), 193.0)
        self.assertEqual(
            prep.retry_delay_seconds(response),
            (193.0, "RateLimit reset (+1s safety)"),
        )

    def test_rate_limit_reset_is_preferred_when_both_headers_exist(self):
        response = SimpleNamespace(
            headers={"Retry-After": "15", "ratelimit": '"api";r=498;t=192'}
        )
        self.assertEqual(
            prep.retry_delay_seconds(response),
            (193.0, "RateLimit reset (+1s safety)"),
        )

    def test_http_date_header_is_rounded_up(self):
        now = datetime(2026, 9, 7, 0, 0, 0, 500000, tzinfo=timezone.utc)
        retry_at = datetime(2026, 9, 7, 0, 0, 2, tzinfo=timezone.utc)
        response = SimpleNamespace(
            headers={"Retry-After": format_datetime(retry_at, usegmt=True)}
        )
        self.assertEqual(prep.retry_after_seconds(response, now=now), 2.0)

    def test_invalid_or_negative_header_falls_back(self):
        for value in (None, "", "-1", "not-a-date"):
            response = SimpleNamespace(
                headers={} if value is None else {"Retry-After": value}
            )
            with self.subTest(value=value):
                self.assertIsNone(prep.retry_after_seconds(response))

    def test_429_uses_ratelimit_reset_before_success(self):
        calls = 0
        sleeps: list[float] = []

        def download(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FakeHTTPError(
                    429,
                    headers={"RateLimit": '"api";r=498;t=192'},
                )
            return "/tmp/sbv2-checkpoint"

        result = prep.snapshot_download_with_429_retry(
            download,
            error_type=FakeHTTPError,
            sleep_fn=sleeps.append,
            clock_fn=lambda: 0.0,
            repo_id="example/repo",
        )
        self.assertEqual(result, "/tmp/sbv2-checkpoint")
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [193.0])

    def test_non_429_is_reraised_without_retry(self):
        sleeps: list[float] = []
        error = FakeHTTPError(503)

        def download(**_kwargs):
            raise error

        with self.assertRaisesRegex(FakeHTTPError, "HTTP 503") as raised:
            prep.snapshot_download_with_429_retry(
                download,
                error_type=FakeHTTPError,
                sleep_fn=sleeps.append,
                clock_fn=lambda: 0.0,
            )
        self.assertIs(raised.exception, error)
        self.assertEqual(sleeps, [])

    def test_exhausted_429_is_reraised(self):
        calls = 0

        def download(**_kwargs):
            nonlocal calls
            calls += 1
            raise FakeHTTPError(429)

        with self.assertRaises(FakeHTTPError):
            prep.snapshot_download_with_429_retry(
                download,
                error_type=FakeHTTPError,
                sleep_fn=lambda _delay: None,
                clock_fn=lambda: 0.0,
            )
        self.assertEqual(calls, 4)

    def test_delay_over_budget_fails_closed(self):
        def download(**_kwargs):
            raise FakeHTTPError(429, "2")

        with self.assertRaisesRegex(RuntimeError, "exceeds remaining retry budget"):
            prep.snapshot_download_with_429_retry(
                download,
                error_type=FakeHTTPError,
                sleep_fn=lambda _delay: None,
                clock_fn=lambda: 0.0,
                max_retry_wait=1.0,
            )


if __name__ == "__main__":
    unittest.main()
