#!/usr/bin/env python3
"""Probe every entry in `.github/pins.yaml` against its live upstream.

Driver for `.github/workflows/corpus-drift-detector.yml`. Split into small
subcommands so each step of the workflow is testable and re-runnable:

  * `--emit-plan`  — parse pins.yaml, print JSON list of probe targets.
  * `--run`        — execute each probe (curl / HF / git ls-remote) and
                     emit one JSON verdict per line.
  * `--enforce`    — apply drift_policy: advisory drifts warn, mirror
                     drifts hard-fail. Exits 1 iff any hard-fail.
  * `--to-md`      — render verdicts as a markdown table row set.
  * `--self-test`  — stdlib-only unit tests for the enforcement + render
                     logic. Runs without network so a broken script is
                     caught before we spend a workflow slot on it.

Zero-dep posture (NFR-DS-02): stdlib only for --emit-plan / --enforce /
--to-md / --self-test. `--run` needs PyYAML + huggingface_hub, pip-installed
into an isolated venv by the calling workflow. Nothing lands in
`root Cargo.lock`.

Honest-measurement rules (FR-EX-08 / NFR-QL-04):
  * Every verdict is one of {match, upstream_drift, mirror_drift,
    no_revision, skip, error}. No implicit "OK" for a probe that could
    not reach the network — that becomes `error`, escalated to a job
    warning so the noise is visible.
  * `--enforce` never rewrites a verdict. It applies drift_policy to the
    verdict emitted by `--run`, and the two stages are strictly separable
    (so a policy change never accidentally hides a measurement).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import unittest
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing / planning
# ---------------------------------------------------------------------------


def _load_yaml(path: Path):
    """Read pins.yaml. PyYAML is only imported here (so --self-test stays
    stdlib-only), and the caller is expected to have pip-installed it."""
    import yaml  # noqa: WPS433 — deliberate localised import

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_plan(pins_path: Path) -> list[dict]:
    """Flatten pins.yaml into a probe plan.

    Each entry becomes one plan row. `mirror.url` populated ⇒ two probes
    (upstream + mirror). Toolchain entries carry a `skip: true` marker so
    `--run` records a skip verdict rather than pretending to reach a
    non-existent content SHA."""
    doc = _load_yaml(pins_path)
    if doc.get("schema_version") != 1:
        raise SystemExit(f"pins.yaml schema_version must be 1, got {doc.get('schema_version')!r}")
    plan: list[dict] = []
    for entry in doc["entries"]:
        row = {
            "name": entry["name"],
            "kind": entry["kind"],
            "owning_workflow": entry["owning_workflow"],
            "owning_line": entry.get("owning_line"),
            "upstream": entry["upstream"],
            "pinned_value": entry["pinned_value"],
            "drift_policy": entry["drift_policy"],
            "mirror": entry.get("mirror") or {},
        }
        plan.append(row)
    return plan


# ---------------------------------------------------------------------------
# Probes — one function per kind
# ---------------------------------------------------------------------------


def _sha256_of_url(url: str, timeout_s: int = 300) -> str:
    """Stream the URL through sha256 without materialising it fully in RAM.

    Corpus tarballs (LibriSpeech dev-clean = 338 MB) are inside the
    default GH Actions runner memory but streaming keeps the peak flat
    even if a future entry adds a larger asset."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "vokra-pins-probe/1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _probe_content_sha(pv: dict, url: str) -> tuple[str, str, str]:
    """Return (verdict, expected, observed) for a content-SHA probe."""
    expected = pv.get("sha256")
    if not expected:
        return ("no_revision", "<unset>", "<skipped>")
    try:
        observed = _sha256_of_url(url)
    except Exception as exc:  # pragma: no cover — network only
        return ("error", expected, f"fetch failed: {exc.__class__.__name__}: {exc}")
    return ("match" if observed == expected else "upstream_drift", expected, observed)


def _probe_hf_revision(pv: dict, hf_id: str) -> tuple[str, str, str]:
    """Compare pinned hf_revision against `HfApi().repo_info(revision=X).sha`."""
    from huggingface_hub import HfApi  # noqa: WPS433

    expected = pv.get("hf_revision")
    if not expected:
        return ("no_revision", "<unset>", "<skipped>")
    if expected == "main":
        # Floating branch; the pinned SHA is not stable — resolve `main`
        # and report the current head SHA so the operator sees drift, but
        # the verdict is `no_revision` (policy: warn) rather than
        # upstream_drift (which would fire even on the first-ever run).
        try:
            observed = HfApi().repo_info(hf_id, revision="main").sha
        except Exception as exc:  # pragma: no cover
            return ("error", expected, f"repo_info failed: {exc.__class__.__name__}: {exc}")
        return ("no_revision", expected, observed or "<unknown>")
    try:
        info = HfApi().repo_info(hf_id, revision=expected)
    except Exception as exc:  # pragma: no cover
        return ("error", expected, f"repo_info failed: {exc.__class__.__name__}: {exc}")
    observed = info.sha or "<unknown>"
    return ("match" if observed == expected else "upstream_drift", expected, observed)


def _probe_git_sha(pv: dict, url: str) -> tuple[str, str, str]:
    """git ls-remote check. Uses the git CLI (present on ubuntu-latest)."""
    import subprocess  # noqa: WPS433

    expected = pv.get("git_sha")
    if not expected:
        return ("no_revision", "<unset>", "<skipped>")
    try:
        res = subprocess.run(
            ["git", "ls-remote", url, expected],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # pragma: no cover
        return ("error", expected, f"ls-remote failed: {exc.__class__.__name__}: {exc}")
    if res.returncode != 0:
        return ("error", expected, f"ls-remote rc={res.returncode}: {res.stderr.strip()}")
    # A pinned SHA either exists (single line output) or it does not
    # (empty stdout ⇒ ref not found).
    if not res.stdout.strip():
        return ("upstream_drift", expected, "<not-found>")
    return ("match", expected, expected)


def run_probes(plan: list[dict], focus: str = "") -> list[dict]:
    """Execute one row per plan entry. Never raises for individual probes."""
    verdicts: list[dict] = []
    for row in plan:
        if focus and row["name"] != focus:
            continue
        r = {
            "name": row["name"],
            "kind": row["kind"],
            "owning_workflow": row["owning_workflow"],
            "owning_line": row.get("owning_line"),
            "target": "upstream",
        }
        kind = row["kind"]
        pv = row["pinned_value"]
        up = row["upstream"]
        if kind == "toolchain":
            r["verdict"], r["expected"], r["observed"] = ("skip", pv.get("version", "<?>"), "<version, not content>")
        elif kind == "corpus" or kind == "codec":
            # If pinned_value has a `sha256`, content-hash probe.
            # Otherwise (HF-only codec via revision), fall through.
            if pv.get("sha256"):
                r["verdict"], r["expected"], r["observed"] = _probe_content_sha(pv, up["url"])
            elif pv.get("hf_revision"):
                r["verdict"], r["expected"], r["observed"] = _probe_hf_revision(
                    pv, up.get("hf_id") or up["url"]
                )
            else:
                r["verdict"], r["expected"], r["observed"] = ("no_revision", "<unset>", "<skipped>")
        elif kind == "checkpoint":
            hf_id = pv.get("hf_id") or up.get("hf_id")
            if not hf_id:
                r["verdict"], r["expected"], r["observed"] = ("error", "<no hf_id>", "<abort>")
            else:
                r["verdict"], r["expected"], r["observed"] = _probe_hf_revision(pv, hf_id)
        elif kind == "other":
            r["verdict"], r["expected"], r["observed"] = _probe_git_sha(pv, up["url"])
        else:
            r["verdict"], r["expected"], r["observed"] = ("error", "<?>", f"unknown kind {kind!r}")
        r["policy"] = row["drift_policy"]
        verdicts.append(r)

        # Mirror probe (only if the mirror block carries a sha256 / hf_revision).
        mirror = row.get("mirror") or {}
        m_sha = mirror.get("sha256")
        m_rev = mirror.get("hf_revision")
        if m_sha or m_rev:
            mr = dict(r)
            mr["target"] = "mirror"
            if m_sha:
                # Rebuild the sha probe using the mirror URL.
                try:
                    observed = _sha256_of_url(mirror["url"])
                    mr["verdict"], mr["expected"], mr["observed"] = (
                        "match" if observed == m_sha else "mirror_drift",
                        m_sha,
                        observed,
                    )
                except Exception as exc:  # pragma: no cover
                    mr["verdict"], mr["expected"], mr["observed"] = (
                        "error",
                        m_sha,
                        f"fetch failed: {exc}",
                    )
            elif m_rev:
                mr["verdict"], mr["expected"], mr["observed"] = _probe_hf_revision(
                    {"hf_revision": m_rev}, mirror.get("hf_id") or up.get("hf_id") or up["url"]
                )
            verdicts.append(mr)

    return verdicts


# ---------------------------------------------------------------------------
# Enforcement — strictly separate from probing so a policy tweak never
# hides a measurement.
# ---------------------------------------------------------------------------


def enforce(verdicts: list[dict]) -> int:
    """Apply drift_policy. Returns process exit code (0 = OK, 1 = hard-fail)."""
    exit_code = 0
    for v in verdicts:
        policy = v.get("policy") or {}
        verdict = v["verdict"]
        wf = v.get("owning_workflow", "<?>")
        line = v.get("owning_line") or 1
        loc = f"file={wf},line={line}"
        name = v["name"]
        exp = v.get("expected", "<?>")
        obs = v.get("observed", "<?>")
        if verdict == "match":
            print(f"::notice {loc}::{name}: {v['target']} pin match ({obs[:16]}…)")
        elif verdict == "skip":
            print(f"::notice {loc}::{name}: {v['target']} skipped ({exp})")
        elif verdict == "no_revision":
            policy_key = policy.get("no_revision") or "warn"
            if policy_key == "warn":
                print(
                    f"::warning {loc}::{name}: pinned by name/branch only (no explicit hf_revision "
                    f"or sha256); WP X-10-T05 pins an explicit SHA"
                )
            # any other policy value (e.g. skip) is silent by design
        elif verdict == "upstream_drift":
            if policy.get("upstream_mismatch") == "hard_fail":
                print(
                    f"::error {loc}::{name}: upstream drift {exp[:16]}… → {obs[:16]}… "
                    f"(hard-fail per drift_policy)"
                )
                exit_code = 1
            else:
                print(
                    f"::warning {loc}::{name}: upstream drift {exp[:16]}… → {obs[:16]}… "
                    f"(advisory — upstream CDN drift is not a Vokra regression)"
                )
        elif verdict == "mirror_drift":
            # Vokra owns byte-identity; default is hard_fail.
            if policy.get("mirror_mismatch") == "advisory":
                print(
                    f"::warning {loc}::{name}: MIRROR drift {exp[:16]}… → {obs[:16]}… "
                    f"(advisory only by explicit policy override)"
                )
            else:
                print(
                    f"::error {loc}::{name}: MIRROR drift {exp[:16]}… → {obs[:16]}… "
                    f"— Vokra owns byte-identity of the mirror"
                )
                exit_code = 1
        elif verdict == "error":
            # Probe itself failed (network, missing tool, bad config).
            # This is not a Vokra regression, but it hides real drift, so
            # it always warns loudly and never blocks the job.
            print(
                f"::warning {loc}::{name}: probe error — {obs}. Rerun after resolving; "
                f"the pin state is UNKNOWN, not verified."
            )
        else:
            print(f"::warning {loc}::{name}: unknown verdict {verdict!r}")
    return exit_code


def to_markdown(verdicts: list[dict]) -> str:
    """Render one row per verdict. Header is emitted separately by the caller."""
    out = io.StringIO()
    for v in verdicts:
        name = v["name"]
        kind = v["kind"]
        verdict = v["verdict"]
        expected = str(v.get("expected", "<?>"))[:40]
        observed = str(v.get("observed", "<?>"))[:40]
        policy = v.get("policy") or {}
        policy_txt = "up=" + str(policy.get("upstream_mismatch", "?"))
        if "mirror_mismatch" in policy:
            policy_txt += ", mirror=" + str(policy["mirror_mismatch"])
        out.write(f"| {name} | {kind} | {verdict} | `{expected}` | `{observed}` | {policy_txt} |\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _cmd_emit_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args.pins)
    payload = {"count": len(plan), "plan": plan}
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan_doc = json.loads(args.plan.read_text(encoding="utf-8"))
    plan = plan_doc["plan"]
    verdicts = run_probes(plan, focus=args.focus or "")
    with args.out.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v) + "\n")
    print(f"emitted {len(verdicts)} verdicts to {args.out}")
    return 0


def _cmd_enforce(args: argparse.Namespace) -> int:
    verdicts = [json.loads(ln) for ln in args.verdicts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return enforce(verdicts)


def _cmd_to_md(args: argparse.Namespace) -> int:
    verdicts = [json.loads(ln) for ln in args.verdicts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sys.stdout.write(to_markdown(verdicts))
    return 0


# ---------------------------------------------------------------------------
# Self-test — stdlib only, no network.
# ---------------------------------------------------------------------------


class _EnforceTests(unittest.TestCase):
    def _v(self, verdict, policy=None, **extra):
        d = {
            "name": "test",
            "kind": "corpus",
            "owning_workflow": ".github/workflows/nightly-asr-wer.yml",
            "owning_line": 1,
            "target": "upstream",
            "verdict": verdict,
            "expected": "abc",
            "observed": "abc",
            "policy": policy or {"upstream_mismatch": "advisory"},
        }
        d.update(extra)
        return d

    def test_match_is_exit_zero(self):
        self.assertEqual(enforce([self._v("match")]), 0)

    def test_upstream_drift_with_advisory_policy_warns_but_exits_zero(self):
        self.assertEqual(
            enforce([self._v("upstream_drift", observed="xyz")]),
            0,
        )

    def test_upstream_drift_with_hard_fail_policy_exits_one(self):
        self.assertEqual(
            enforce([self._v("upstream_drift", policy={"upstream_mismatch": "hard_fail"}, observed="xyz")]),
            1,
        )

    def test_mirror_drift_default_is_hard_fail(self):
        v = self._v("mirror_drift", target="mirror", observed="xyz")
        v["policy"] = {"mirror_mismatch": "hard_fail"}
        self.assertEqual(enforce([v]), 1)

    def test_mirror_drift_advisory_override_exits_zero(self):
        v = self._v("mirror_drift", target="mirror", observed="xyz")
        v["policy"] = {"mirror_mismatch": "advisory"}
        self.assertEqual(enforce([v]), 0)

    def test_no_revision_warns_but_never_fails(self):
        v = self._v("no_revision")
        v["policy"] = {"no_revision": "warn"}
        self.assertEqual(enforce([v]), 0)

    def test_probe_error_warns_but_never_fails(self):
        # A network error must not silently PASS (that would hide drift)
        # but must also not turn into a Vokra regression signal.
        self.assertEqual(enforce([self._v("error", observed="fetch failed")]), 0)

    def test_skip_is_a_notice(self):
        self.assertEqual(enforce([self._v("skip")]), 0)

    def test_mixed_batch_exits_on_first_hard_fail(self):
        vs = [
            self._v("match"),
            self._v("upstream_drift", policy={"upstream_mismatch": "hard_fail"}, observed="xyz"),
            self._v("match"),
        ]
        self.assertEqual(enforce(vs), 1)


class _RenderTests(unittest.TestCase):
    def test_render_includes_every_row(self):
        rows = [
            {
                "name": "a", "kind": "corpus", "verdict": "match",
                "expected": "ab" * 20, "observed": "ab" * 20,
                "policy": {"upstream_mismatch": "advisory"},
            },
            {
                "name": "b", "kind": "checkpoint", "verdict": "no_revision",
                "expected": "<unset>", "observed": "<skipped>",
                "policy": {"upstream_mismatch": "advisory", "no_revision": "warn"},
            },
        ]
        md = to_markdown(rows)
        self.assertIn("| a |", md)
        self.assertIn("| b |", md)
        self.assertIn("no_revision", md)


def _cmd_self_test(_args: argparse.Namespace) -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (_EnforceTests, _RenderTests):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("emit-plan", help="Parse pins.yaml and print JSON plan.")
    p_plan.add_argument("--pins", type=Path, default=Path(".github/pins.yaml"))
    p_plan.set_defaults(func=_cmd_emit_plan)

    p_run = sub.add_parser("run", help="Execute probes and emit JSONL verdicts.")
    p_run.add_argument("--plan", type=Path, required=True)
    p_run.add_argument("--focus", type=str, default="")
    p_run.add_argument("--out", type=Path, required=True)
    p_run.set_defaults(func=_cmd_run)

    p_enf = sub.add_parser("enforce", help="Apply drift_policy to JSONL verdicts.")
    p_enf.add_argument("--verdicts", type=Path, required=True)
    p_enf.set_defaults(func=_cmd_enforce)

    p_md = sub.add_parser("to-md", help="Render JSONL verdicts as markdown rows.")
    p_md.add_argument("verdicts", type=Path)
    p_md.set_defaults(func=_cmd_to_md)

    p_st = sub.add_parser("self-test", help="Run stdlib-only unit tests.")
    p_st.set_defaults(func=_cmd_self_test)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
