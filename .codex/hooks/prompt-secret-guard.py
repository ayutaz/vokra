#!/usr/bin/env -S uv run --script
"""Block high-confidence credentials pasted into a Codex prompt.

The hook only examines the in-memory prompt payload.  It never prints the
prompt or the matching value and does not make network calls or write files.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys


TOKEN_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-(?:svcacct|admin)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{30,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
)


def looks_like_placeholder(value: str) -> bool:
    lower = value.lower()
    body = lower.split("_", 1)[-1]
    if any(marker in lower for marker in ("example", "placeholder", "yourtoken", "your_token", "changeme")):
        return True
    if len(set(body)) <= 3:
        return True
    if body and all(ord(body[index + 1]) - ord(body[index]) == 1 for index in range(len(body) - 1)):
        return True
    return False


def contains_live_credential(prompt: str) -> bool:
    for pattern in TOKEN_PATTERNS:
        for match in pattern.finditer(prompt):
            value = match.group(0)
            if value.startswith("-----BEGIN") or not looks_like_placeholder(value):
                return True
    return False


def join_fragments(*fragments: str) -> str:
    return "".join(fragments)


def self_test() -> int:
    body = ("A7x9", "Q2mK", "8vR4", "tY6p", "L3nD", "5sF8", "gH2j", "K9mQ")
    positive = (
        join_fragments("please use ", "hf", "_", *body),
        join_fragments("gh", "p", "_", "9Xk3", "Pq7L", "m2Vb", "8Nc4", "Rt6Y", "w1Hs", "5Df9", "Jk3L", "p7Qm", "2"),
        join_fragments("sk-proj", "-", *body),
        join_fragments("sk-svcacct", "-", *body),
        join_fragments("AK", "IA", "IOSF", "ODNN", "7A7B", "8C9D"),
        join_fragments("-----BEGIN ", "OPENSSH ", "PRIVATE KEY-----"),
    )
    negative = (
        "HF_TOKEN=<your-token>",
        "hf_abcdefghijklmnopqrst",
        "sk-proj-<project-key>",
        "the docs mention GitHub and OpenAI keys",
    )
    assert all(contains_live_credential(value) for value in positive)
    assert not any(contains_live_credential(value) for value in negative)
    # A blocked result is boolean-only: no credential value is returned.
    redaction_input = join_fragments(
        "gh", "p", "_", "9Xk3", "Pq7L", "m2Vb", "8Nc4", "Rt6Y", "w1Hs", "5Df9", "Jk3L", "p7Qm", "2"
    )
    assert contains_live_credential(redaction_input) is True
    output = io.StringIO()
    error = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
        result = process_payload({"prompt": redaction_input})
    assert result == 2
    assert redaction_input not in output.getvalue() + error.getvalue()
    assert "live credential" in error.getvalue()
    print("prompt-secret-guard --self-test: OK")
    return 0


def process_payload(payload: object) -> int:
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if isinstance(prompt, str) and contains_live_credential(prompt):
        print(
            "Blocked: the prompt appears to contain a live credential. "
            "Remove it, rotate it if exposed, and pass secrets through the approved environment mechanism.",
            file=sys.stderr,
        )
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0
    return process_payload(payload)


if __name__ == "__main__":
    raise SystemExit(main())
