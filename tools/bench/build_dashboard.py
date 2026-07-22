#!/usr/bin/env python3
"""Vokra benchmark dashboard generator (X-06-T13/T14/T15).

deliverables.md:132 lists a "benchmark dashboard" as a public artifact. There
is no aggregated surface today — perf data is scattered across
docs/bench-baselines/**, docs/perf/*.json, and docs/benchmarks/*.md. This
generator reads that committed corpus (plus nightly CI artifacts, if staged
alongside) and emits ONE self-contained static HTML page.

Design red lines:
  * stdlib-only (zero-dep NFR-DS-02): json / html / pathlib / argparse only.
  * self-contained: inline CSS only, NO external asset/CDN/script/font refs
    (`--self-check` re-scans the output and fails on any external reference).
  * no fabrication (FR-EX-08): a missing input renders a "no data" cell, never
    an invented number. Every value carries a provenance label (rig / date /
    advisory-vs-gate) so a reader cannot mistake an M1-rig-scoped reference
    for a CI gate — the exact confusion docs/bench-baselines/README.md warns
    about.

Usage:
  python3 tools/bench/build_dashboard.py --out docs/dashboard/index.html
  python3 tools/bench/build_dashboard.py --self-check           # build + verify self-contained
  python3 tools/bench/build_dashboard.py --self-check <file>    # verify an existing file
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NO_DATA = '<span class="nodata">no data</span>'


# --------------------------------------------------------------------------- #
# Loaders — each returns structured rows or an empty list (never fabricates).
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    title: str
    subtitle: str
    columns: list
    rows: list = field(default_factory=list)
    note: str = ""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def classify_baseline(path: Path, data: dict) -> str:
    """Provenance class — the one thing a reader must not get wrong."""
    if data.get("$placeholder") is True:
        return "placeholder (unseeded — no verdict)"
    prov = str(data.get("provenance", ""))
    if "M1-RIG-SCOPED" in prov or "M1 iMac" in prov or ".m1." in path.name:
        return "M1-rig reference (NOT a CI gate)"
    if path.name == "mel_frontend_baseline.json":
        return "CI gate (ubuntu-latest, PR-blocking)"
    return "reference"


def load_committed_baselines(repo: Path) -> Section:
    sec = Section(
        title="Committed regression baselines",
        subtitle="report-shaped JSON under docs/bench-baselines/ (the vokra-cli bench --baseline corpus)",
        columns=["file", "task", "rtf", "provenance"],
    )
    root = repo / "docs" / "bench-baselines"
    files = sorted(p for p in root.rglob("*.json") if not p.name.endswith(".spdx.json"))
    for p in files:
        data = _read_json(p)
        if not isinstance(data, dict):
            continue
        is_report = ("rtf" in data) or (data.get("$placeholder") is True)
        if not is_report:
            continue
        rel = p.relative_to(repo).as_posix()
        rtf = data.get("rtf")
        rtf_cell = f"{rtf:.6f}" if isinstance(rtf, (int, float)) and not isinstance(rtf, bool) else NO_DATA
        sec.rows.append([rel, str(data.get("task", "?")), rtf_cell, classify_baseline(p, data)])
    return sec


def load_gpu_perf(repo: Path) -> Section:
    sec = Section(
        title="GPU performance (CUDA / H100)",
        subtitle="docs/perf/*.json — reference measurements, never a required PR check",
        columns=["model", "backend", "hardware", "median RTF", "measured", "gate status"],
    )
    root = repo / "docs" / "perf"
    for p in sorted(root.glob("*.json")):
        data = _read_json(p)
        if not isinstance(data, dict):
            continue
        median = data.get("median_rtf")
        median_cell = f"{median:.4f}" if isinstance(median, (int, float)) else NO_DATA
        sec.rows.append([
            str(data.get("model", "?")),
            str(data.get("backend", "?")),
            str(data.get("hardware", "?")),
            median_cell,
            str(data.get("measured_at", "?")),
            str(data.get("gate_status", "?")),
        ])
    return sec


def load_rtf_variance(repo: Path) -> Section:
    """CUDA RTF variance from the vast.ai rtf-*.jsonl runs: N, mean, CV."""
    sec = Section(
        title="CUDA RTF variance (vast.ai reference)",
        subtitle="docs/bench-baselines/**/rtf-*.jsonl — N-iteration probes (mean, coefficient of variation)",
        columns=["source", "label", "N", "mean RTF", "CV"],
    )
    for p in sorted(repo.glob("docs/bench-baselines/**/rtf-*.jsonl")):
        by_label = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtf = rec.get("rtf")
            if not isinstance(rtf, (int, float)):
                continue
            by_label.setdefault(str(rec.get("label", "?")), []).append(float(rtf))
        rel = p.relative_to(repo).as_posix()
        for label, vals in sorted(by_label.items()):
            n = len(vals)
            mean = sum(vals) / n if n else None
            if mean and n > 1:
                var = sum((v - mean) ** 2 for v in vals) / n
                cv = (var ** 0.5) / mean if mean else None
            else:
                cv = None
            sec.rows.append([
                rel,
                label,
                str(n),
                f"{mean:.4f}" if mean is not None else NO_DATA,
                f"{cv:.3f}" if cv is not None else NO_DATA,
            ])
    return sec


def parse_m5_14_table(repo: Path) -> Section:
    """Tolerant markdown-table parse of the M5-14 RTF-vs-ORT headline table.

    Degrades to an empty section (rendered as 'no data') if the report or its
    table is not found — never fabricates rows."""
    sec = Section(
        title="CPU RTF vs ONNX Runtime (M5-14)",
        subtitle="docs/bench-baselines/m5-14-final-2026-07-18/report.md — CPU hot-path optimization outcome",
        columns=["model (leg)", "now (RTF)", "speedup", "vs-ORT", "target", "met?"],
    )
    report = repo / "docs" / "bench-baselines" / "m5-14-final-2026-07-18" / "report.md"
    if not report.is_file():
        return sec
    text = report.read_text(encoding="utf-8")
    # Find the first markdown table whose header mentions vs-ORT/speedup.
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
        if len(lines) < 3:
            continue
        header = lines[0].lower()
        if "vs-ort" not in header and "speedup" not in header:
            continue
        # lines[1] is the |---|---| separator; data rows follow.
        for ln in lines[2:]:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 7:
                continue
            # Strip markdown bold; the table is source-of-truth text.
            clean = [re.sub(r"\*\*(.*?)\*\*", r"\1", c) for c in cells]
            # columns: model | wave0 | now | speedup | vs-ort | target | met
            sec.rows.append([clean[0], clean[2], clean[3], clean[4], clean[5], clean[6]])
        break
    return sec


def load_device_docs(repo: Path) -> Section:
    sec = Section(
        title="Device benchmarks (Tier 1/2)",
        subtitle="docs/benchmarks/*.md — owner-measured on real hardware (scaffold until the device lab lands)",
        columns=["document", "status"],
    )
    root = repo / "docs" / "benchmarks"
    for p in sorted(root.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        scaffold = "scaffold" in text.lower() or "実測値は依頼者" in text or "実測は依頼者" in text
        status = "scaffold — awaiting owner device measurements" if scaffold else "populated"
        sec.rows.append([p.relative_to(repo).as_posix(), status])
    return sec


# --------------------------------------------------------------------------- #
# Rendering — self-contained HTML, inline CSS only.
# --------------------------------------------------------------------------- #
CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       background: #fbfbfd; color: #1c1c1e; padding: 2rem 1rem; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .25rem; border-bottom: 2px solid #d9d9de; padding-bottom: .25rem; }
.sub { color: #6b6b70; font-size: .85rem; margin: 0 0 .75rem; }
.legend { font-size: .82rem; color: #6b6b70; margin: .5rem 0 1.5rem; }
.tablewrap { overflow-x: auto; border: 1px solid #e2e2e7; border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #ececf1; white-space: nowrap; }
th { background: #f1f1f5; font-weight: 600; position: sticky; top: 0; }
tr:last-child td { border-bottom: 0; }
.nodata { color: #b0b0b8; font-style: italic; }
code, td:first-child { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; }
footer { margin-top: 2.5rem; font-size: .78rem; color: #8a8a90; }
@media (prefers-color-scheme: dark) {
  body { background: #0f0f11; color: #e6e6ea; }
  h2 { border-color: #303036; }
  .sub, .legend, footer { color: #9a9aa2; }
  .tablewrap { border-color: #2a2a30; }
  th, td { border-color: #26262b; }
  th { background: #1a1a1f; }
  .nodata { color: #55555c; }
}
"""


def render_section(sec: Section) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in sec.columns)
    if sec.rows:
        body_rows = []
        for row in sec.rows:
            cells = "".join(f"<td>{_cell(v)}</td>" for v in row)
            body_rows.append(f"<tr>{cells}</tr>")
        body = "\n".join(body_rows)
    else:
        body = f'<tr><td colspan="{len(sec.columns)}">{NO_DATA}</td></tr>'
    note = f'<p class="sub">{html.escape(sec.note)}</p>' if sec.note else ""
    return (
        f"<h2>{html.escape(sec.title)}</h2>\n"
        f'<p class="sub">{html.escape(sec.subtitle)}</p>\n'
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>\n{body}\n</tbody></table></div>{note}"
    )


def _cell(v) -> str:
    """A cell is either a pre-rendered no-data span or plain text to escape."""
    if v is NO_DATA or v == NO_DATA:
        return NO_DATA
    return html.escape(str(v))


def build_html(repo: Path, stamp: str) -> str:
    sections = [
        parse_m5_14_table(repo),
        load_gpu_perf(repo),
        load_rtf_variance(repo),
        load_committed_baselines(repo),
        load_device_docs(repo),
    ]
    body = "\n".join(render_section(s) for s in sections)
    legend = (
        "Provenance matters: a <b>CI gate</b> baseline is measured on ubuntu-latest and "
        "blocks PRs; an <b>M1-rig reference</b> is a single-machine number and is NOT a gate "
        "(an M1/NEON baseline false-fails the 5% CI gate — see docs/bench-baselines/README.md); "
        "a <b>placeholder</b> is unseeded and claims no verdict. Empty cells are labelled "
        "<span class='nodata'>no data</span> — never a fabricated number (FR-EX-08)."
    )
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Vokra benchmark dashboard</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
        "<h1>Vokra benchmark dashboard</h1>\n"
        f'<p class="sub">Generated {html.escape(stamp)} from the committed perf corpus '
        "(docs/bench-baselines/, docs/perf/, docs/benchmarks/). Self-contained; advisory.</p>\n"
        f'<p class="legend">{legend}</p>\n'
        f"{body}\n"
        "<footer>Vokra — X-06 benchmark dashboard. All values carry provenance; "
        "no external assets are loaded (CSP-safe, zero-dep tooling).</footer>\n"
        "</div>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# Self-check — reject any external resource reference.
# --------------------------------------------------------------------------- #
def find_external_refs(html_text: str) -> list:
    """Return descriptions of external-resource references (empty = clean).

    Flags resource-LOADING vectors, not plain-text URLs (all text is escaped,
    so a URL in prose never lands inside an attribute)."""
    findings = []
    lowered = html_text.lower()
    for tag in ("<script", "<link", "<img", "<iframe", "<object", "<embed", "<use ", "<audio", "<video", "<source"):
        if tag in lowered:
            findings.append(f"forbidden tag {tag!r}")
    for token in ("@import", "srcset", "url("):
        if token in lowered:
            findings.append(f"forbidden css/attr token {token!r}")
    # src=/href= pointing off-box (http(s):// or protocol-relative //).
    for m in re.finditer(r"(?:src|href)\s*=\s*[\"']?(https?:)?//", lowered):
        findings.append(f"external ref: ...{lowered[max(0, m.start()-10):m.end()+20]}...")
    return findings


def self_check(html_text: str) -> list:
    return find_external_refs(html_text)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="Build the Vokra benchmark dashboard.")
    ap.add_argument("--out", help="write the HTML here (default: stdout)")
    ap.add_argument("--repo", default=str(REPO), help="repo root (default: inferred)")
    ap.add_argument("--stamp", default="at build time", help="generation stamp text")
    ap.add_argument(
        "--self-check",
        nargs="?",
        const="__generate__",
        help="verify self-containment; with no arg, build then verify; with a path, verify that file",
    )
    args = ap.parse_args(argv)
    repo = Path(args.repo)

    if args.self_check and args.self_check != "__generate__":
        text = Path(args.self_check).read_text(encoding="utf-8")
        findings = self_check(text)
        if findings:
            print("SELF-CHECK FAILED — external references:", file=sys.stderr)
            for f in findings:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("self-check OK: no external references")
        return 0

    html_text = build_html(repo, args.stamp)

    if args.self_check == "__generate__":
        findings = self_check(html_text)
        if findings:
            print("SELF-CHECK FAILED — external references:", file=sys.stderr)
            for f in findings:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("self-check OK: no external references")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_text, encoding="utf-8")
        print(f"wrote {out} ({len(html_text)} bytes)")
    elif args.self_check != "__generate__":
        sys.stdout.write(html_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
