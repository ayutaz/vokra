#!/usr/bin/env python3
"""cuda_rtf_analyze.py — reduce ``cuda_rtf_variance.sh`` JSONL into a report.

Reads the JSON-lines emitted by ``cuda_rtf_variance.sh`` (one object per
iteration, plus one optional ``"type":"summary"`` trailer line) and emits a
markdown report with mean / median / stddev / CV / p50 / p95 / p99 / min /
max and an ASCII histogram over the ``rtf`` samples.

**Position in the plan** — this is the *variance analysis* companion to the
sanity numbers reported by
``crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs``. Per
``docs/adr/M2-03-followup-rtf.md`` §D6, the formal < 0.10 always-on gate
lives at **M2-14** (owner self-hosted CUDA runner) + **M3-01** (5%
regression gate) — this analyzer never asserts an RTF ceiling and never
promotes any threshold. It only surfaces CV > 0.20 as a **warning** so the
owner can see whether their single-shot 0.081 – 0.115 range on RTX 4090
was hardware variability (high CV) or a single-shot outlier (low CV).

**Zero-dep constraint** (NFR-DS-02): stdlib only — no numpy, no pandas, no
matplotlib. ``statistics`` gives us mean / median / pstdev / quantiles and
that is exactly what this report needs. The histogram is a simple ``█``
bar chart printed to stdout.

Usage::

    ./cuda_rtf_variance.sh --gguf lv3.gguf --audio jfk.wav --iters 10 \\
        --output rtf.jsonl

    ./cuda_rtf_analyze.py rtf.jsonl
    ./cuda_rtf_analyze.py --input rtf.jsonl --format markdown
    ./cuda_rtf_analyze.py < rtf.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Iterable, Optional


# CV > this threshold surfaces as a warning in the report. This is a *report*
# threshold, not an always-on gate — see the module docstring.
CV_WARN_THRESHOLD = 0.20


@dataclass
class Sample:
    """One successful iteration."""

    iter: int
    rtf: float
    latency_ms: Optional[float]
    fa_mode: str
    fa_v2_mode: str
    backend: str
    timestamp: str


@dataclass
class Failure:
    """One failed iteration."""

    iter: int
    error: str
    timestamp: str


@dataclass
class Summary:
    """Trailer summary line from cuda_rtf_variance.sh (if present)."""

    iters_requested: int
    iters_failed: int
    started_at: str
    ended_at: str
    fa_mode: str
    fa_v2_mode: str
    backend: str
    label: str
    host: str
    gpu: str
    driver: str
    gguf: str
    audio: str


def _fa_mode_of(obj: dict) -> str:
    """The M4-07 ``fa_mode`` field, with the pre-M4-07 legacy mapping.

    Old JSONL (e.g. ``docs/bench-baselines/vast-2026-07-10/``) carries only
    ``fa_v2_mode``: ``on`` was the gated FA v2 leg and ``off`` the decomposed
    leg, and FA v3 did not exist — so the two map losslessly onto the new
    3-value label space.
    """
    mode = str(obj.get("fa_mode", ""))
    if mode:
        return mode
    return {"on": "v2", "off": "decomposed"}.get(str(obj.get("fa_v2_mode", "")), "")


def parse_jsonl(lines: Iterable[str]) -> tuple[list[Sample], list[Failure], Optional[Summary]]:
    """Parse the harness JSONL into samples / failures / trailer.

    Silently skips blank lines. Non-JSON lines (stderr accidentally
    concatenated) are collected as failures with a synthetic iter=-1 so the
    reader can see them.
    """
    samples: list[Sample] = []
    failures: list[Failure] = []
    summary: Optional[Summary] = None

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            failures.append(
                Failure(iter=-1, error=f"non-JSON line: {raw!r} ({e})", timestamp="")
            )
            continue

        if not isinstance(obj, dict):
            failures.append(
                Failure(iter=-1, error=f"non-object JSON line: {raw!r}", timestamp="")
            )
            continue

        if obj.get("type") == "summary":
            summary = Summary(
                iters_requested=int(obj.get("iters_requested", 0)),
                iters_failed=int(obj.get("iters_failed", 0)),
                started_at=str(obj.get("started_at", "")),
                ended_at=str(obj.get("ended_at", "")),
                fa_mode=_fa_mode_of(obj),
                fa_v2_mode=str(obj.get("fa_v2_mode", "")),
                backend=str(obj.get("backend", "")),
                label=str(obj.get("label", "")),
                host=str(obj.get("host", "")),
                gpu=str(obj.get("gpu", "")),
                driver=str(obj.get("driver", "")),
                gguf=str(obj.get("gguf", "")),
                audio=str(obj.get("audio", "")),
            )
            continue

        status = obj.get("status", "ok")
        if status == "ok":
            rtf = obj.get("rtf")
            if not isinstance(rtf, (int, float)):
                failures.append(
                    Failure(
                        iter=int(obj.get("iter", -1)),
                        error=f"missing / non-numeric rtf: {obj!r}",
                        timestamp=str(obj.get("timestamp", "")),
                    )
                )
                continue
            samples.append(
                Sample(
                    iter=int(obj.get("iter", -1)),
                    rtf=float(rtf),
                    latency_ms=(
                        float(obj["latency_ms"])
                        if isinstance(obj.get("latency_ms"), (int, float))
                        else None
                    ),
                    fa_mode=_fa_mode_of(obj),
                    fa_v2_mode=str(obj.get("fa_v2_mode", "")),
                    backend=str(obj.get("backend", "")),
                    timestamp=str(obj.get("timestamp", "")),
                )
            )
        else:
            failures.append(
                Failure(
                    iter=int(obj.get("iter", -1)),
                    error=str(obj.get("error", "unknown"))[:400],
                    timestamp=str(obj.get("timestamp", "")),
                )
            )

    return samples, failures, summary


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    """Reduced statistics over an RTF sample vector."""

    count: int
    mean: float
    median: float
    stddev: float           # population stddev (statistics.pstdev)
    cv: float               # coefficient of variation = stddev / mean
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float


def _nearest_rank_percentile(sorted_samples: list[float], q: float) -> float:
    """Nearest-rank percentile matching ``vokra-cli`` ``report.rs::percentile``.

    Duplicated in Python so this analyzer produces the *same* p50/p95/p99
    values a follow-up Rust-side aggregator would produce over the exact
    same samples (report.rs uses ``rank = ceil(q * n - 1e-9)``). We match
    that formula exactly so the two paths never disagree by a float epsilon.
    """
    n = len(sorted_samples)
    if n == 0:
        return float("nan")
    rank = math.ceil(q * n - 1e-9)
    if rank < 1:
        rank = 1
    if rank > n:
        rank = n
    return sorted_samples[rank - 1]


def summarize(samples: list[float]) -> Optional[Stats]:
    """Reduce a sample list to :class:`Stats`. ``None`` if empty."""
    if not samples:
        return None

    sorted_samples = sorted(samples)
    mean = statistics.fmean(sorted_samples)
    median = statistics.median(sorted_samples)

    # Population stddev — matches what ``vokra-cli`` ``report.rs::summarize``
    # calls "jitter" (var = mean of squared deviations, not the sample
    # stddev). This keeps CV values comparable to what a follow-up Rust
    # aggregator would emit.
    if len(sorted_samples) >= 2:
        stddev = statistics.pstdev(sorted_samples)
    else:
        stddev = 0.0

    cv = stddev / mean if mean != 0.0 else float("inf")

    return Stats(
        count=len(sorted_samples),
        mean=mean,
        median=median,
        stddev=stddev,
        cv=cv,
        p50=_nearest_rank_percentile(sorted_samples, 0.50),
        p95=_nearest_rank_percentile(sorted_samples, 0.95),
        p99=_nearest_rank_percentile(sorted_samples, 0.99),
        minimum=sorted_samples[0],
        maximum=sorted_samples[-1],
    )


# ---------------------------------------------------------------------------
# Histogram rendering
# ---------------------------------------------------------------------------

def render_histogram(samples: list[float], bins: int = 10, width: int = 40) -> list[str]:
    """Render an ASCII histogram of ``samples`` as a list of markdown lines.

    Fixed-width bar chart using ``█`` blocks; the ``width`` argument is the
    max bar width in characters (the widest bin gets ``width`` blocks). Bin
    edges are equi-spaced between ``min`` and ``max``; a single-valued
    sample list collapses to a single bin.
    """
    if not samples:
        return ["*(no samples)*"]

    if bins < 1:
        bins = 1

    lo = min(samples)
    hi = max(samples)

    if hi == lo:
        # All samples identical — one bin.
        return [
            "| bin | range | count | bar |",
            "|---|---|---|---|",
            f"| 0 | `{lo:.6f}` | {len(samples)} | {'█' * width} |",
        ]

    step = (hi - lo) / bins
    edges = [lo + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for s in samples:
        # Assign to bin ``k`` where ``edges[k] <= s < edges[k+1]``; the
        # rightmost edge is inclusive so ``max(samples)`` lands in the last
        # bin, not out of range.
        k = int((s - lo) / step)
        if k >= bins:
            k = bins - 1
        if k < 0:
            k = 0
        counts[k] += 1

    max_count = max(counts) if counts else 0
    scale = width / max_count if max_count > 0 else 0.0

    lines = [
        "| bin | range | count | bar |",
        "|---|---|---|---|",
    ]
    for i, c in enumerate(counts):
        bar_len = int(round(c * scale)) if scale > 0 else 0
        bar = "█" * bar_len
        lines.append(
            f"| {i} | `[{edges[i]:.6f}, {edges[i + 1]:.6f}{']' if i == bins - 1 else ')'}` "
            f"| {c} | {bar} |"
        )
    return lines


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def format_markdown(
    samples: list[Sample],
    failures: list[Failure],
    summary: Optional[Summary],
) -> str:
    """Render the full markdown report string."""
    parts: list[str] = []
    parts.append("# CUDA large-v3 RTF variance report\n")
    parts.append(
        "_Generated by `tools/parity/cuda_rtf_analyze.py`. This is a **reference**\n"
        "measurement — the formal `RTF < 0.10` always-on gate lives at **M2-14**\n"
        "(owner self-hosted CUDA runner) + **M3-01** (5% regression gate); see\n"
        "`docs/adr/M2-03-followup-rtf.md` §D6._\n"
    )

    # ---- run metadata ----
    parts.append("## Run metadata\n")
    if summary is not None:
        parts.append("| field | value |")
        parts.append("|---|---|")
        parts.append(f"| iters requested | {summary.iters_requested} |")
        parts.append(f"| iters failed | {summary.iters_failed} |")
        parts.append(f"| started_at (UTC) | `{summary.started_at}` |")
        parts.append(f"| ended_at   (UTC) | `{summary.ended_at}` |")
        parts.append(f"| backend | `{summary.backend}` |")
        parts.append(f"| fa_mode | `{summary.fa_mode}` |")
        parts.append(f"| fa_v2_mode (legacy) | `{summary.fa_v2_mode}` |")
        parts.append(f"| label | `{summary.label}` |")
        parts.append(f"| host | `{summary.host}` |")
        parts.append(f"| gpu | `{summary.gpu}` |")
        parts.append(f"| driver | `{summary.driver}` |")
        parts.append(f"| gguf | `{summary.gguf}` |")
        parts.append(f"| audio | `{summary.audio}` |")
    else:
        parts.append("_(no `type=summary` trailer line found — running against a partial JSONL?)_")
    parts.append("")

    # ---- stats ----
    parts.append("## RTF statistics\n")
    stats = summarize([s.rtf for s in samples])
    if stats is None:
        parts.append("_(no successful samples — every iteration failed)_")
    else:
        parts.append("| metric | value |")
        parts.append("|---|---|")
        parts.append(f"| n (successful samples) | {stats.count} |")
        parts.append(f"| mean   | `{stats.mean:.6f}` |")
        parts.append(f"| median | `{stats.median:.6f}` |")
        parts.append(f"| stddev (population) | `{stats.stddev:.6f}` |")
        parts.append(f"| CV (stddev / mean) | `{stats.cv:.6f}` |")
        parts.append(f"| p50 | `{stats.p50:.6f}` |")
        parts.append(f"| p95 | `{stats.p95:.6f}` |")
        parts.append(f"| p99 | `{stats.p99:.6f}` |")
        parts.append(f"| min | `{stats.minimum:.6f}` |")
        parts.append(f"| max | `{stats.maximum:.6f}` |")
    parts.append("")

    # ---- CV warning ----
    parts.append("## Coefficient-of-variation warning\n")
    if stats is None:
        parts.append(
            "_no samples — CV cannot be computed. Every iteration failed; "
            "the harness must be re-run before any judgment._"
        )
    elif stats.cv > CV_WARN_THRESHOLD:
        parts.append(
            f"**WARNING**: CV = `{stats.cv:.4f}` > `{CV_WARN_THRESHOLD:.2f}` — "
            "the measurement is unstable. Likely causes: thermal throttling, "
            "GPU boost-clock jitter, PCIe contention with other workloads, "
            "or a mixed instance state. Recommendation: extend `--iters`, "
            "run on a dedicated (non-shared) host, or add cooldown pauses "
            "between iterations. Do **NOT** promote the formal `<0.10` gate "
            "off this run — hand the raw JSONL to M2-14."
        )
    else:
        parts.append(
            f"OK: CV = `{stats.cv:.4f}` <= `{CV_WARN_THRESHOLD:.2f}`. The "
            "measurement is stable enough that mean / median are meaningful. "
            "(This is *not* a formal gate — the `<0.10` always-on gate lives "
            "at M2-14 / M3-01 per `docs/adr/M2-03-followup-rtf.md` §D6.)"
        )
    parts.append("")

    # ---- per-iter samples ----
    parts.append("## Per-iteration RTF samples\n")
    if samples:
        parts.append("| iter | timestamp (UTC) | fa_mode | backend | rtf | latency_ms |")
        parts.append("|---|---|---|---|---|---|")
        for s in sorted(samples, key=lambda x: x.iter):
            lat = f"{s.latency_ms:.4f}" if s.latency_ms is not None else "n/a"
            parts.append(
                f"| {s.iter} | `{s.timestamp}` | `{s.fa_mode}` "
                f"| `{s.backend}` | `{s.rtf:.6f}` | `{lat}` |"
            )
    else:
        parts.append("_(no successful samples)_")
    parts.append("")

    # ---- failures ----
    if failures:
        parts.append("## Failures\n")
        parts.append("| iter | timestamp (UTC) | error (first 400 chars) |")
        parts.append("|---|---|---|")
        for f in sorted(failures, key=lambda x: x.iter):
            # Newlines in the captured error break markdown tables — collapse.
            err = f.error.replace("|", "\\|").replace("\n", " ⏎ ")
            parts.append(f"| {f.iter} | `{f.timestamp}` | {err} |")
        parts.append("")

    # ---- histogram ----
    parts.append("## RTF histogram (10 bins)\n")
    parts.extend(render_histogram([s.rtf for s in samples], bins=10, width=40))
    parts.append("")

    # ---- footer ----
    parts.append("## Interpretation guide (owner)\n")
    parts.append(
        "- **CV** — coefficient of variation `stddev / mean`. On a well-cooled, "
        "dedicated RTX 4090 with a single Whisper session, CV values in the "
        "`0.001 – 0.01` range are typical for the decomposed path baseline "
        "(see `docs/bench-baselines/whisper_large_v3_cuda_rtf.json`)."
    )
    parts.append(
        "- **p99 vs mean** — if p99 exceeds mean by `>2x` in a run of `N >= 10`, "
        "the tail is dominated by a single outlier (session-build variability, "
        "or a transient scheduler blip). Re-run with a larger `N` before "
        "attaching significance."
    )
    parts.append(
        "- **Formal `<0.10` gate** — do NOT promote off this report. Hand the "
        "raw JSONL and this markdown to M2-14 (owner self-hosted CUDA runner) "
        "and M3-01 (5% regression gate); the always-on decision lives there "
        "per `docs/adr/M2-03-followup-rtf.md` §D6, not in this analyzer."
    )
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="cuda_rtf_analyze.py",
        description=(
            "Reduce cuda_rtf_variance.sh JSONL into a markdown variance report. "
            "This is a reference analyzer — never asserts an RTF ceiling. "
            "See tools/parity/README-cuda-rtf-variance.md for the full workflow."
        ),
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Path to the JSONL file (or '-' for stdin). Default: '-'.",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="-",
        help="Where to write the markdown report ('-' for stdout).",
    )
    ap.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help=(
            "Output format. 'markdown' (default) is the human-readable report; "
            "'json' emits the summary + samples as a machine-readable object "
            "so this analyzer can feed a downstream harness."
        ),
    )
    args = ap.parse_args(argv)

    # Read input.
    if args.input == "-":
        raw_lines = sys.stdin.readlines()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

    samples, failures, summary = parse_jsonl(raw_lines)

    if args.format == "markdown":
        report = format_markdown(samples, failures, summary)
    else:
        stats = summarize([s.rtf for s in samples])
        obj = {
            "summary": (
                None
                if summary is None
                else {
                    "iters_requested": summary.iters_requested,
                    "iters_failed": summary.iters_failed,
                    "started_at": summary.started_at,
                    "ended_at": summary.ended_at,
                    "fa_mode": summary.fa_mode,
                    "fa_v2_mode": summary.fa_v2_mode,
                    "backend": summary.backend,
                    "label": summary.label,
                    "host": summary.host,
                    "gpu": summary.gpu,
                    "driver": summary.driver,
                    "gguf": summary.gguf,
                    "audio": summary.audio,
                }
            ),
            "stats": (
                None
                if stats is None
                else {
                    "count": stats.count,
                    "mean": stats.mean,
                    "median": stats.median,
                    "stddev": stats.stddev,
                    "cv": stats.cv,
                    "p50": stats.p50,
                    "p95": stats.p95,
                    "p99": stats.p99,
                    "min": stats.minimum,
                    "max": stats.maximum,
                    "cv_warn_threshold": CV_WARN_THRESHOLD,
                    "cv_warn": (stats.cv > CV_WARN_THRESHOLD),
                }
            ),
            "samples": [
                {
                    "iter": s.iter,
                    "rtf": s.rtf,
                    "latency_ms": s.latency_ms,
                    "fa_mode": s.fa_mode,
                    "fa_v2_mode": s.fa_v2_mode,
                    "backend": s.backend,
                    "timestamp": s.timestamp,
                }
                for s in samples
            ],
            "failures": [
                {"iter": f.iter, "error": f.error, "timestamp": f.timestamp}
                for f in failures
            ],
        }
        report = json.dumps(obj, indent=2) + "\n"

    if args.output == "-":
        sys.stdout.write(report)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)

    # Exit code contract:
    #   0 — analysis completed (regardless of CV warning)
    #   1 — no successful samples (every iteration failed)
    # We intentionally do NOT surface CV > 0.20 as a non-zero exit — it is a
    # report warning, not a gate. See module docstring.
    if not samples:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
