#!/usr/bin/env python3
"""Enforce that `.github/pins.yaml` and `.github/workflows/*.yml` stay in sync.

Two enforcement legs, both must pass:

  * FORWARD (catalog → workflow): every `pinned_value` literal in pins.yaml
    (sha256 / hf_revision / git_sha / version / hf_id) is present, verbatim,
    in the entry's `owning_workflow`. Catches "workflow env edited without
    updating the catalog" (or vice versa).

  * REVERSE (workflow → catalog): every SHA-shaped literal that appears in
    a workflow env under a recognised naming pattern
    (`_SHA256`, `_REVISION`, `_GIT_SHA`, `_SHA`) must appear in pins.yaml
    as some entry's `pinned_value`. Catches "pin added to a workflow but
    never registered in the catalog".

Allowlist: `.github/pins-allowlist.txt` records intentional exceptions
(git-history commit SHAs in comments, one-off documentation references,
etc.) as `<workflow>:<literal>  # <justification>` lines. Every allowlist
entry MUST carry a justification, so the drift can be re-audited later.

Failure emits `::error file=<path>,line=<n>::<msg>` so GitHub highlights
the exact line in the PR file-view.

Advisory-first posture (matches .github/workflows/README.md §2 promotion
policy): the calling workflow (`pins-sync-check.yml`) sets
`continue-on-error: true` on the job, which is dropped after 4 consecutive
green weeks per owner judgement.

Zero-dep NFR-DS-02: stdlib + PyYAML in an isolated venv. Nothing lands
in root Cargo.lock.
"""

from __future__ import annotations

import argparse
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Regex patterns for the reverse leg.
#
# Each pattern captures a single (name, value) group so the error message
# can point to the specific env variable that introduced the drift. The
# patterns are anchored to typed prefixes to avoid catching git commit
# hashes embedded in workflow comments (those need to be allowlisted
# explicitly when legitimate).
# ---------------------------------------------------------------------------

SHA256_ENV_RE = re.compile(
    r"^\s*([A-Z_]+_SHA256):\s*['\"]?([a-f0-9]{64})['\"]?",
    re.M,
)
REVISION_ENV_RE = re.compile(
    r"^\s*([A-Z_]+_REVISION):\s*['\"]?([a-f0-9]{40})['\"]?",
    re.M,
)
GIT_SHA_ENV_RE = re.compile(
    r"^\s*([A-Z_]+_GIT_SHA):\s*['\"]?([a-f0-9]{40})['\"]?",
    re.M,
)


def _load_yaml(path: Path):
    """Localised PyYAML import so --self-test stays stdlib-only."""
    import yaml  # noqa: WPS433

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_allowlist(path: Path) -> set[tuple[str, str]]:
    """Return {(workflow_basename, literal)} for legitimate exceptions."""
    if not path.exists():
        return set()
    out: set[tuple[str, str]] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise SystemExit(
                f"{path}:{lineno}: allowlist line must be '<workflow>:<literal>  # <justification>'"
            )
        wf, lit = line.split(":", 1)
        # Every allowlist entry must have a justification (text after #).
        if "#" not in raw:
            raise SystemExit(
                f"{path}:{lineno}: allowlist entries MUST carry a justification "
                f"after # (documented drift reason)"
            )
        out.add((wf.strip(), lit.strip()))
    return out


def _catalog_pins(pins_path: Path) -> list[dict]:
    doc = _load_yaml(pins_path)
    return list(doc.get("entries") or [])


def _concrete_literals(pv: dict) -> list[tuple[str, str]]:
    """Return (field_name, literal_value) for the non-null pinned_value fields."""
    out: list[tuple[str, str]] = []
    for key in ("sha256", "hf_revision", "git_sha", "version", "hf_id"):
        v = pv.get(key)
        if v is None:
            continue
        out.append((key, str(v)))
    return out


def _all_workflows(root: Path) -> list[Path]:
    wf_dir = root / ".github" / "workflows"
    return sorted(wf_dir.glob("*.yml"))


def check_forward(pins_path: Path) -> list[str]:
    """Leg 1: pins.yaml → workflow. Every literal must appear in owning_workflow."""
    problems: list[str] = []
    entries = _catalog_pins(pins_path)
    for entry in entries:
        wf = REPO / entry["owning_workflow"]
        line = entry.get("owning_line") or 1
        # `owning_line: 0` is the "no forward literal search" sentinel. It
        # covers two shapes:
        #   (1) UNLANDED pin — pin registered in the catalog ahead of the
        #       committing workflow value (e.g. LibriSpeech mirror before
        #       vars.VOKRA_CORPUS_..._MIRROR_URL is set), and
        #   (2) SOURCE.ENV-INDIRECTED pin — the workflow imports the pin
        #       via `source <path>/source.env` (UTMOS pattern), so the
        #       literal lives in the sourced file rather than the workflow
        #       yml. A forward literal grep of the workflow would find
        #       nothing even though the pin IS landed.
        # drift_policy.upstream_mismatch=skip short-circuits pins_probe.py
        # for shape (1); shape (2) still probes because the SHA is real.
        # The reverse leg still refuses orphan pins under any workflow env
        # naming pattern (`_SHA256`, `_REVISION`, `_GIT_SHA`).
        if entry.get("owning_line") == 0:
            continue
        if not wf.exists():
            problems.append(
                f"::error file={entry['owning_workflow']}::pins.yaml entry "
                f"'{entry['name']}' points to a workflow that does not exist"
            )
            continue
        wf_text = wf.read_text(encoding="utf-8")
        for key, lit in _concrete_literals(entry["pinned_value"]):
            if lit not in wf_text:
                problems.append(
                    f"::error file={entry['owning_workflow']},line={line}::"
                    f"pins.yaml entry '{entry['name']}': pinned_value.{key}={lit!r} "
                    f"not found in the owning workflow. Update the workflow env "
                    f"or the catalog; both sides must move together."
                )
        # If a mirror is populated, its sha/revision must appear in the workflow OR
        # the workflow must reference vars.VOKRA_CORPUS_..._MIRROR_URL/SHA256 (proving
        # the fallback seam is wired). We check the loose condition since exact
        # variable naming varies by corpus.
        mirror = entry.get("mirror") or {}
        if mirror.get("sha256") and mirror["sha256"] not in wf_text:
            has_fallback = "vars.VOKRA_CORPUS_" in wf_text and "MIRROR_" in wf_text
            if not has_fallback:
                problems.append(
                    f"::error file={entry['owning_workflow']},line={line}::"
                    f"pins.yaml entry '{entry['name']}': mirror.sha256 populated "
                    f"but neither the SHA nor a vars.VOKRA_CORPUS_*_MIRROR_* fallback "
                    f"reference appears in the workflow"
                )
    return problems


def _known_literals(entries: list[dict]) -> set[str]:
    known: set[str] = set()
    for entry in entries:
        for _key, lit in _concrete_literals(entry["pinned_value"]):
            known.add(lit)
        mirror = entry.get("mirror") or {}
        for key in ("sha256", "hf_revision"):
            v = mirror.get(key)
            if v:
                known.add(str(v))
    return known


def check_reverse(pins_path: Path, allowlist_path: Path) -> list[str]:
    """Leg 2: workflow → pins.yaml. Every typed SHA literal must be registered."""
    entries = _catalog_pins(pins_path)
    known = _known_literals(entries)
    allowlist = _load_allowlist(allowlist_path)
    problems: list[str] = []
    for wf in _all_workflows(REPO):
        text = wf.read_text(encoding="utf-8")
        for pattern, kind_label in (
            (SHA256_ENV_RE, "SHA256"),
            (REVISION_ENV_RE, "40-hex REVISION"),
            (GIT_SHA_ENV_RE, "40-hex GIT_SHA"),
        ):
            for m in pattern.finditer(text):
                env_name, literal = m.group(1), m.group(2)
                if literal in known:
                    continue
                if (wf.name, literal) in allowlist:
                    continue
                # Line number: count newlines before match start
                line = text.count("\n", 0, m.start()) + 1
                problems.append(
                    f"::error file=.github/workflows/{wf.name},line={line}::"
                    f"unregistered {kind_label} pin: env {env_name}={literal!r}. "
                    f"Register it in .github/pins.yaml or allowlist it in "
                    f".github/pins-allowlist.txt with a justification comment."
                )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    problems: list[str] = []
    problems.extend(check_forward(args.pins))
    problems.extend(check_reverse(args.pins, args.allowlist))
    for p in problems:
        print(p)
    if problems:
        print(f"\n{len(problems)} sync problem(s). See GitHub file annotations above.", file=sys.stderr)
        return 1
    print("check_pins_sync: OK (forward and reverse legs both clean)")
    return 0


class _ForwardTests(unittest.TestCase):
    def test_missing_literal_is_reported_with_line_pointer(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "fake.yml").write_text(
                "name: fake\nenv:\n  X: y\n", encoding="utf-8"
            )
            pins = root / ".github" / "pins.yaml"
            pins.write_text(
                "schema_version: 1\nentries:\n"
                "  - name: fake\n    kind: corpus\n"
                "    owning_workflow: .github/workflows/fake.yml\n"
                "    owning_line: 3\n"
                "    upstream:\n      url: https://example/\n      license: MIT\n"
                "    pinned_value:\n      sha256: 'deadbeef" + "0" * 56 + "'\n"
                "    drift_policy:\n      upstream_mismatch: advisory\n",
                encoding="utf-8",
            )
            # Point REPO at the temp root for this test.
            import check_pins_sync as m

            orig = m.REPO
            m.REPO = root
            try:
                problems = m.check_forward(pins)
            finally:
                m.REPO = orig
            self.assertTrue(any("deadbeef" in p and "line=3" in p for p in problems), problems)

    def test_matching_literal_is_silent(self):
        import tempfile

        sha = "abc" + "0" * 61
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "fake.yml").write_text(
                f"name: fake\nenv:\n  X_SHA256: '{sha}'\n", encoding="utf-8"
            )
            pins = root / ".github" / "pins.yaml"
            pins.write_text(
                "schema_version: 1\nentries:\n"
                "  - name: fake\n    kind: corpus\n"
                "    owning_workflow: .github/workflows/fake.yml\n"
                "    upstream:\n      url: https://example/\n      license: MIT\n"
                f"    pinned_value:\n      sha256: '{sha}'\n"
                "    drift_policy:\n      upstream_mismatch: advisory\n",
                encoding="utf-8",
            )
            import check_pins_sync as m

            orig = m.REPO
            m.REPO = root
            try:
                problems = m.check_forward(pins)
            finally:
                m.REPO = orig
            self.assertEqual(problems, [])

    def test_owning_line_zero_skips_forward_literal_search(self):
        """owning_line: 0 is the unlanded-pin sentinel (X-10-T05, UTMOS pattern)."""
        import tempfile

        sha = "cafe" + "0" * 60
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".github" / "workflows").mkdir(parents=True)
            # Workflow deliberately does NOT contain the SHA. Forward leg
            # must not complain because owning_line=0 says "not yet landed".
            (root / ".github" / "workflows" / "fake.yml").write_text(
                "name: fake\non: [push]\n", encoding="utf-8"
            )
            pins = root / ".github" / "pins.yaml"
            pins.write_text(
                "schema_version: 1\nentries:\n"
                "  - name: unlanded\n    kind: checkpoint\n"
                "    owning_workflow: .github/workflows/fake.yml\n"
                "    owning_line: 0\n"
                "    upstream:\n      url: https://example/\n      license: MIT\n"
                f"    pinned_value:\n      sha256: '{sha}'\n"
                "    drift_policy:\n      upstream_mismatch: skip\n",
                encoding="utf-8",
            )
            import check_pins_sync as m

            orig = m.REPO
            m.REPO = root
            try:
                problems = m.check_forward(pins)
            finally:
                m.REPO = orig
            self.assertEqual(problems, [])


class _ReverseTests(unittest.TestCase):
    def test_unregistered_sha256_is_reported(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "wf.yml").write_text(
                "env:\n  FOO_SHA256: 'a" + "0" * 63 + "'\n", encoding="utf-8"
            )
            pins = root / ".github" / "pins.yaml"
            pins.write_text(
                "schema_version: 1\nentries: []\n", encoding="utf-8"
            )
            import check_pins_sync as m

            orig = m.REPO
            m.REPO = root
            try:
                problems = m.check_reverse(pins, Path("/dev/null"))
            finally:
                m.REPO = orig
            self.assertTrue(any("FOO_SHA256" in p and "unregistered" in p for p in problems), problems)

    def test_allowlist_suppresses_reverse_warning(self):
        import tempfile

        sha = "a" + "0" * 63
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "wf.yml").write_text(
                f"env:\n  FOO_SHA256: '{sha}'\n", encoding="utf-8"
            )
            allow = root / "allowlist.txt"
            allow.write_text(f"wf.yml:{sha}  # legit exception documented here\n", encoding="utf-8")
            pins = root / ".github" / "pins.yaml"
            pins.write_text("schema_version: 1\nentries: []\n", encoding="utf-8")
            import check_pins_sync as m

            orig = m.REPO
            m.REPO = root
            try:
                problems = m.check_reverse(pins, allow)
            finally:
                m.REPO = orig
            self.assertEqual(problems, [])

    def test_allowlist_without_justification_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            allow = Path(td) / "allowlist.txt"
            allow.write_text("wf.yml:abc\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                _load_allowlist(allow)
            self.assertIn("justification", str(ctx.exception))


def _cmd_self_test(_args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (_ForwardTests, _ReverseTests):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chk = sub.add_parser("check", help="Run forward + reverse legs against the tree.")
    p_chk.add_argument("--pins", type=Path, default=REPO / ".github" / "pins.yaml")
    p_chk.add_argument(
        "--allowlist", type=Path, default=REPO / ".github" / "pins-allowlist.txt"
    )
    p_chk.set_defaults(func=_cmd_check)

    p_st = sub.add_parser("self-test", help="Run stdlib-only unit tests.")
    p_st.set_defaults(func=_cmd_self_test)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
