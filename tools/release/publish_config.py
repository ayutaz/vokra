#!/usr/bin/env python3
"""Validate explicit release-channel decisions without exposing credentials."""

from __future__ import annotations

import argparse
import os
import sys


CHANNELS = {
    "crates_io": "CRATES_IO_PUBLISH_ENABLED",
    "npm": "NPM_PUBLISH_ENABLED",
    "openupm": "OPENUPM_PUBLISH_ENABLED",
    "pypi": "PYPI_PUBLISH_ENABLED",
    "desktop_aar": "DESKTOP_AAR_ENABLED",
}


def enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() == "true"


def validate(strict: bool) -> tuple[dict[str, bool], list[str]]:
    decisions: dict[str, bool] = {}
    errors: list[str] = []
    for channel, variable in CHANNELS.items():
        raw = os.environ.get(variable, "").lower()
        if raw not in {"true", "false"}:
            if strict:
                errors.append(f"repository variable {variable} must be explicitly true or false")
            decisions[channel] = False
        else:
            decisions[channel] = raw == "true"

    credential_contracts = (
        ("crates_io", "CRATES_IO_TOKEN"),
        ("npm", "NPM_TOKEN"),
        ("openupm", "OPENUPM_TOKEN"),
    )
    for channel, secret in credential_contracts:
        if decisions[channel] and not os.environ.get(secret):
            errors.append(f"{CHANNELS[channel]}=true requires secret {secret}")

    if decisions["pypi"]:
        oidc_attested = enabled("PYPI_TRUSTED_PUBLISHER_CONFIGURED")
        token_present = bool(os.environ.get("PYPI_API_TOKEN"))
        if not oidc_attested and not token_present:
            errors.append(
                "PYPI_PUBLISH_ENABLED=true requires either "
                "PYPI_TRUSTED_PUBLISHER_CONFIGURED=true or secret PYPI_API_TOKEN"
            )
    return decisions, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", choices=("true", "false"), required=True)
    parser.add_argument("--github-output")
    args = parser.parse_args()

    strict = args.strict == "true"
    decisions, errors = validate(strict)
    for channel, value in decisions.items():
        print(f"publish-config: {channel}={'enabled' if value else 'disabled'}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            for channel, value in decisions.items():
                handle.write(f"{channel}={'true' if value else 'false'}\n")
    if errors:
        for error in errors:
            print(f"publish-config: FAIL {error}", file=sys.stderr)
        raise SystemExit(1)
    if not strict:
        print(
            "publish-config: advisory dry-run; pass enforce_publish_config=true "
            "to require explicit repository settings"
        )


if __name__ == "__main__":
    main()
