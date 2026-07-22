#!/usr/bin/env python3
"""test_crates_io.py — oracle for the crates.io release legs (X-07-T13).

Asserts the crates.io wiring added by X-07-T09..T12:

  (1) `crates-io-dry-run` job exists and hits the publish set in TOPOLOGICAL
      order (drives tools/release/crates_publish_order.py + `cargo package
      --list`).
  (2) `crates-io-publish` job exists and is TOKEN-GATED (secrets.CRATES_IO_TOKEN
      + tag + dry_run != true), with an honest ::notice:: skip when the token is
      absent (fabricated-pass forbidden), and IMMUTABLE-INDEX fail-loud (no
      --allow-dirty, no skip-existing).
  (3) the publish job re-checks the license/zero-dep closure right before
      publishing (NFR-LC-02 mechanized — crates.io cannot be un-published).
  (4) the publish order tool self-verifies (delegates to `--verify`).
  (5) X-09b docs.rs unblock is recorded (release.yml comment) — publishing is
      the precondition for docs.rs auto-doc.
  (6) the Rust public-API snapshot exists (docs/abi/vokra-rust-public-api.
      v1.0-rc.list) — publishing first exposes the Rust surface as a public
      contract; per IF-01 the freeze is at v1.0 GA (M5-13), so rc is Pre-1.0.

RED before X-07: no crates-io job exists, publish=false blocks vokra-core, and
crates_publish_order.py is absent.

Zero-dep (NFR-DS-02): python3 stdlib only + the publish-order tool.
Usage: python3 tools/release/test_crates_io.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")
ORDER_TOOL = os.path.join(ROOT, "tools", "release", "crates_publish_order.py")
API_SNAPSHOT = os.path.join(ROOT, "docs", "abi", "vokra-rust-public-api.v1.0-rc.list")

_pass = 0
_fail = 0


def ok(msg: str) -> None:
    global _pass
    _pass += 1
    print(f"  ok:   {msg}")


def bad(msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  FAIL: {msg}", file=sys.stderr)


def job_block(text: str, job_id: str) -> str | None:
    """Return the text of the `  <job_id>:` block (2-space indent) or None."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^  {re.escape(job_id)}:\s*$", ln):
            start = i
            break
    if start is None:
        return None
    out = [lines[start]]
    for ln in lines[start + 1:]:
        # A new 2-space job key (or a top-level key) ends the block.
        if re.match(r"^  [A-Za-z_][\w-]*:\s*$", ln) or (ln and not ln[0].isspace()):
            break
        out.append(ln)
    return "\n".join(out)


def main() -> None:
    if not os.path.isfile(RELEASE_YML):
        print(f"error: {RELEASE_YML} not found", file=sys.stderr)
        sys.exit(1)
    text = open(RELEASE_YML, encoding="utf-8").read()

    print("[X-07-T13] crates.io release legs")

    # (1) dry-run job.
    dry = job_block(text, "crates-io-dry-run")
    if dry is None:
        bad("(1) crates-io-dry-run job missing")
    else:
        if "crates_publish_order.py" in dry and "cargo package --list" in dry:
            ok("(1) crates-io-dry-run drives the topo-order tool + cargo package --list")
        else:
            bad("(1) crates-io-dry-run must run crates_publish_order.py + cargo package --list")

    # (2) publish job + gating.
    pub = job_block(text, "crates-io-publish")
    if pub is None:
        bad("(2) crates-io-publish job missing")
    else:
        gated = ("CRATES_IO_TOKEN" in pub
                 and "startsWith(github.ref, 'refs/tags/v')" in pub
                 and "inputs.dry_run != true" in pub)
        if gated:
            ok("(2) crates-io-publish is tag + dry_run + CRATES_IO_TOKEN gated")
        else:
            bad("(2) crates-io-publish gating incomplete (need tag + dry_run + token)")

        if "::notice::" in pub and "CRATES_IO_TOKEN not set" in pub:
            ok("(2) token-absent path is an honest ::notice:: skip (no fabricated pass)")
        else:
            bad("(2) token-absent path must be an announced ::notice:: skip")

        # Immutable index: must NOT weaken publish with --allow-dirty / skip.
        # Only inspect NON-comment lines — the job keeps a comment documenting
        # the deliberate absence of these flags.
        pub_code = "\n".join(
            ln for ln in pub.splitlines() if not ln.lstrip().startswith("#")
        )
        if "--allow-dirty" in pub_code:
            bad("(2) crates-io-publish uses --allow-dirty (immutable index must be clean)")
        elif "skip-existing" in pub_code or "--skip-existing" in pub_code:
            bad("(2) crates-io-publish skips existing (duplicate version must fail loud)")
        else:
            ok("(2) immutable-index fail-loud (no --allow-dirty / skip-existing)")

        # (3) pre-publish license re-check.
        if "check-zero-deps.sh" in pub:
            ok("(3) publish job re-checks the zero-dep/license closure (NFR-LC-02)")
        else:
            bad("(3) publish job must re-run check-zero-deps.sh before publishing")

    # (4) order tool self-verify.
    proc = subprocess.run(
        ["python3", ORDER_TOOL, "--verify"], cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode == 0:
        ok(f"(4) crates_publish_order.py --verify OK ({proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else 'ok'})")
    else:
        bad(f"(4) crates_publish_order.py --verify failed: {proc.stderr.strip()[:200]}")

    # (5) X-09b docs.rs unblock recorded.
    if "docs.rs" in text and "X-09b" in text:
        ok("(5) X-09b docs.rs unblock is recorded in release.yml")
    else:
        bad("(5) release.yml must record the X-09b docs.rs unblock (publish precedes docs.rs)")

    # (6) Rust public-API snapshot present.
    if os.path.isfile(API_SNAPSHOT):
        ok("(6) Rust public-API snapshot present (publish-time contract record)")
    else:
        bad(f"(6) Rust public-API snapshot missing: {API_SNAPSHOT}")

    print()
    print(f"crates-io oracle: {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
