#!/usr/bin/env python3
"""npu_rtf_analyze.py — reduce ``npu_rtf_variance.sh`` JSONL into a report.

Reads the JSON-lines emitted by ``npu_rtf_variance.sh`` (one object per
iteration, plus one optional ``"type":"summary"`` trailer line) and emits
a markdown report with mean / median / stddev / CV / p50 / p95 / p99 /
min / max and an ASCII histogram over the ``rtf`` samples, PLUS a
placement summary (share of iterations where the NPU carried ≥ 90 % of
the hot ops).

**Position in the plan** — this is the *variance analysis* companion to
the M5-01 / M5-02 NPU bakeoff. The NFR-PF-12 acceptance criterion (≥ 2×
over the CPU baseline) is an owner judgment based on the report this
tool produces. Per the sibling ``cuda_rtf_analyze.py`` red-line, this
analyzer NEVER asserts an RTF ceiling and NEVER promotes any threshold.
It only surfaces two WARNs:

1. ``CV > 0.20`` — measurement is noisy; do not attach significance
   until the run is stable.
2. ``placement < 90 %`` — the NPU delegate is silently falling back to
   CPU; per FR-EX-08 this disqualifies the run from feeding a 2×
   verdict. The owner must fix the placement (kernel not supported,
   shape not supported, precision mismatch, driver too old) or count
   the delegate as "did not run" for that bakeoff.

**Zero-dep constraint** (NFR-DS-02): stdlib only — no numpy, no pandas,
no matplotlib. ``statistics`` gives us mean / median / pstdev /
quantiles and that is exactly what this report needs. The histogram is
a simple ``█`` bar chart printed to stdout.

Usage::

    ./npu_rtf_variance.sh --gguf lv3.gguf --audio jfk.wav --iters 10 \\
        --backend coreml --output rtf.jsonl

    ./npu_rtf_analyze.py rtf.jsonl
    ./npu_rtf_analyze.py --input rtf.jsonl --format markdown
    ./npu_rtf_analyze.py < rtf.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Iterable, Optional


# CV > this threshold surfaces as a warning in the report. This is a
# *report* threshold, not an always-on gate — see the module docstring.
CV_WARN_THRESHOLD = 0.20

# Placement fraction below this triggers the FR-EX-08 silent-CPU-fallback
# WARN. The 90 % floor matches the M5-01 / M5-02 bakeoff templates.
PLACEMENT_WARN_THRESHOLD = 0.90


@dataclass
class Sample:
    """One successful iteration."""

    iter: int
    rtf: float
    latency_ms: Optional[float]
    backend: str
    timestamp: str
    # Placement fraction on the NPU-of-interest (ANE for coreml, HTP for
    # qnn). ``None`` when the probe was absent / failed to parse.
    npu_frac: Optional[float]
    # Retain the raw dict so the report can echo per-axis breakdown.
    placement_raw: Optional[dict]


@dataclass
class Failure:
    """One failed iteration."""

    iter: int
    error: str
    timestamp: str


@dataclass
class Summary:
    """Trailer summary line from npu_rtf_variance.sh (if present)."""

    iters_requested: int
    iters_failed: int
    started_at: str
    ended_at: str
    backend: str
    label: str
    host: str
    device_name: str
    device_os: str
    device_soc: str
    gguf: str
    audio: str
    placement_probe: str


def _npu_key_for_backend(backend: str) -> str:
    """The placement dict key that represents the NPU-of-interest.

    - CoreML runs on Apple silicon: the ANE is the target; ``ane_frac``
      is what the probe should report. If the probe reports ``gpu_frac``
      + ``cpu_frac`` only (e.g. an older Instruments trace), the caller
      is expected to reconcile — this analyzer treats the missing
      ``ane_frac`` as "unknown", not "0".
    - QNN runs on Snapdragon: the Hexagon HTP is the target
      (``htp_frac``). Some probes also report ``dsp_frac`` (older
      terminology) — we accept either.
    - Anything else (cuda / cpu) does not have an NPU seat; the
      placement check simply does not fire.
    """
    return {
        "coreml": "ane_frac",
        "qnn": "htp_frac",
    }.get(backend, "")


def _extract_npu_frac(backend: str, placement: Optional[dict]) -> Optional[float]:
    """Read the NPU fraction from a placement dict.

    Returns None if the placement dict is absent, if the expected key is
    missing, or if the value is not in [0, 1]. Accepts both ``htp_frac``
    and ``dsp_frac`` for the QNN backend for compatibility with older
    QNN profiler dumps.
    """
    if placement is None:
        return None
    if not isinstance(placement, dict):
        return None

    key = _npu_key_for_backend(backend)
    if not key:
        return None

    value = placement.get(key)
    if value is None and backend == "qnn":
        # QNN legacy compatibility — accept dsp_frac when htp_frac absent.
        value = placement.get("dsp_frac")

    if not isinstance(value, (int, float)):
        return None
    v = float(value)
    if not (0.0 <= v <= 1.0):
        return None
    return v


def parse_jsonl(lines: Iterable[str]) -> tuple[list[Sample], list[Failure], Optional[Summary]]:
    """Parse the harness JSONL into samples / failures / trailer.

    Silently skips blank lines. Non-JSON lines (stderr accidentally
    concatenated) are collected as failures with a synthetic iter=-1 so
    the reader can see them.
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
                backend=str(obj.get("backend", "")),
                label=str(obj.get("label", "")),
                host=str(obj.get("host", "")),
                device_name=str(obj.get("device_name", "")),
                device_os=str(obj.get("device_os", "")),
                device_soc=str(obj.get("device_soc", "")),
                gguf=str(obj.get("gguf", "")),
                audio=str(obj.get("audio", "")),
                placement_probe=str(obj.get("placement_probe", "")),
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
            backend = str(obj.get("backend", ""))
            placement = obj.get("placement")
            samples.append(
                Sample(
                    iter=int(obj.get("iter", -1)),
                    rtf=float(rtf),
                    latency_ms=(
                        float(obj["latency_ms"])
                        if isinstance(obj.get("latency_ms"), (int, float))
                        else None
                    ),
                    backend=backend,
                    timestamp=str(obj.get("timestamp", "")),
                    npu_frac=_extract_npu_frac(backend, placement),
                    placement_raw=placement if isinstance(placement, dict) else None,
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
    that formula exactly so the two paths never disagree by a float
    epsilon.
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

    # Population stddev — matches ``vokra-cli`` ``report.rs::summarize``
    # ("jitter") so CV values stay comparable to a future Rust aggregator.
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
# Placement summary
# ---------------------------------------------------------------------------

@dataclass
class PlacementReport:
    """Reduced placement over an iteration list."""

    total: int
    with_probe: int              # iterations that had a placement dict
    with_npu_frac: int           # iterations that had a valid NPU fraction
    below_threshold: int         # iterations where npu_frac < threshold
    mean_npu_frac: Optional[float]
    min_npu_frac: Optional[float]


def placement_report(samples: list[Sample]) -> PlacementReport:
    total = len(samples)
    with_probe = sum(1 for s in samples if s.placement_raw is not None)
    fracs = [s.npu_frac for s in samples if isinstance(s.npu_frac, float)]
    with_npu_frac = len(fracs)
    below = sum(1 for f in fracs if f < PLACEMENT_WARN_THRESHOLD)
    mean = statistics.fmean(fracs) if fracs else None
    minimum = min(fracs) if fracs else None
    return PlacementReport(
        total=total,
        with_probe=with_probe,
        with_npu_frac=with_npu_frac,
        below_threshold=below,
        mean_npu_frac=mean,
        min_npu_frac=minimum,
    )


# ---------------------------------------------------------------------------
# Histogram rendering
# ---------------------------------------------------------------------------

def render_histogram(samples: list[float], bins: int = 10, width: int = 40) -> list[str]:
    """Render an ASCII histogram of ``samples`` as a list of markdown lines.

    Fixed-width bar chart using ``█`` blocks; the ``width`` argument is
    the max bar width in characters (the widest bin gets ``width``
    blocks). Bin edges are equi-spaced between ``min`` and ``max``; a
    single-valued sample list collapses to a single bin.
    """
    if not samples:
        return ["*(no samples)*"]

    if bins < 1:
        bins = 1

    lo = min(samples)
    hi = max(samples)

    if hi == lo:
        return [
            "| bin | range | count | bar |",
            "|---|---|---|---|",
            f"| 0 | `{lo:.6f}` | {len(samples)} | {'█' * width} |",
        ]

    step = (hi - lo) / bins
    edges = [lo + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for s in samples:
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
    parts.append("# NPU delegate RTF variance report\n")
    parts.append(
        "_Generated by `tools/parity/npu_rtf_analyze.py`. This is a **reference**\n"
        "measurement — the formal NFR-PF-12 `≥ 2× vs CPU baseline` verdict is an\n"
        "**owner judgment** based on this report, per `docs/system-requirements.md`\n"
        "NFR-PF-12 hazard clause (silent-CPU-fallback disqualifies the run) and the\n"
        "sibling `cuda_rtf_analyze.py` red-line (this tool never asserts an RTF\n"
        "ceiling and never promotes any threshold)._\n"
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
        parts.append(f"| label | `{summary.label}` |")
        parts.append(f"| host | `{summary.host}` |")
        parts.append(f"| device_name | `{summary.device_name}` |")
        parts.append(f"| device_os | `{summary.device_os}` |")
        parts.append(f"| device_soc | `{summary.device_soc}` |")
        parts.append(f"| gguf | `{summary.gguf}` |")
        parts.append(f"| audio | `{summary.audio}` |")
        parts.append(f"| placement_probe | `{summary.placement_probe or '(none)'}` |")
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
            "background contention, first-launch delegate JIT variability, "
            "or a mixed session state (Neural Engine sleep transitions). "
            "Recommendation: extend `--iters`, run on a dedicated (non-shared) "
            "host, or add cooldown pauses between iterations. Do **NOT** "
            "attach a 2× verdict to this run — feed the raw JSONL back to "
            "the bakeoff template as insufficient-data."
        )
    else:
        parts.append(
            f"OK: CV = `{stats.cv:.4f}` <= `{CV_WARN_THRESHOLD:.2f}`. The "
            "measurement is stable enough that mean / median are meaningful. "
            "(The formal 2× verdict remains an owner judgment; this WARN is "
            "an information floor, not a gate.)"
        )
    parts.append("")

    # ---- placement warning ----
    parts.append("## Placement (silent-CPU-fallback guard)\n")
    prep = placement_report(samples)
    parts.append("| metric | value |")
    parts.append("|---|---|")
    parts.append(f"| iterations | {prep.total} |")
    parts.append(f"| iterations with placement probe | {prep.with_probe} |")
    parts.append(f"| iterations with valid NPU fraction | {prep.with_npu_frac} |")
    parts.append(f"| iterations below {int(PLACEMENT_WARN_THRESHOLD * 100)}% NPU | {prep.below_threshold} |")
    mean_str = f"`{prep.mean_npu_frac:.4f}`" if prep.mean_npu_frac is not None else "`unknown`"
    min_str = f"`{prep.min_npu_frac:.4f}`" if prep.min_npu_frac is not None else "`unknown`"
    parts.append(f"| mean NPU fraction | {mean_str} |")
    parts.append(f"| min NPU fraction  | {min_str} |")
    parts.append("")

    if prep.total == 0:
        parts.append("_no successful iterations — placement check cannot fire._")
    elif prep.with_npu_frac == 0:
        parts.append(
            f"**WARNING**: no iteration produced a valid NPU fraction. Either "
            f"``--placement-probe`` was not passed or the probe failed to emit "
            f"the expected key (``{_npu_key_for_backend(samples[0].backend)}`` "
            f"for `{samples[0].backend}`). Per FR-EX-08 a missing placement "
            f"probe **disqualifies** the run from feeding a 2× verdict — the "
            f"NPU may have been silently falling back to CPU. Wire up the "
            f"per-delegate probe (Xcode Instruments MLModel trace for CoreML; "
            f"``qnn-net-run --profiling_option op`` for QNN) and re-run."
        )
    elif prep.below_threshold > 0:
        parts.append(
            f"**WARNING**: {prep.below_threshold} / {prep.with_npu_frac} "
            f"iterations placed less than "
            f"{int(PLACEMENT_WARN_THRESHOLD * 100)}% of hot ops on the NPU. "
            "This is the silent-CPU-fallback pattern FR-EX-08 forbids: an "
            "unsupported op (kernel not implemented, unsupported shape, "
            "unsupported dtype, driver too old) is being run on the CPU "
            "arm without an error surface. Do NOT read the RTF number as an "
            "NPU-vs-CPU comparison — you are measuring "
            "``NPU || CPU-fallback`` vs pure CPU. Fix the placement first "
            "(promote the op to the delegate, or accept the delegate as "
            "unsuitable for this workload) and re-run."
        )
    else:
        parts.append(
            f"OK: all {prep.with_npu_frac} iterations kept ≥ "
            f"{int(PLACEMENT_WARN_THRESHOLD * 100)}% of hot ops on the NPU. "
            "The RTF numbers are a fair NPU-vs-CPU comparison; the 2× "
            "verdict is available for the owner to record."
        )
    parts.append("")

    # ---- per-iter samples ----
    parts.append("## Per-iteration RTF samples\n")
    if samples:
        parts.append("| iter | timestamp (UTC) | backend | rtf | latency_ms | NPU frac |")
        parts.append("|---|---|---|---|---|---|")
        for s in sorted(samples, key=lambda x: x.iter):
            lat = f"{s.latency_ms:.4f}" if s.latency_ms is not None else "n/a"
            npu = f"{s.npu_frac:.4f}" if s.npu_frac is not None else "unknown"
            parts.append(
                f"| {s.iter} | `{s.timestamp}` | `{s.backend}` "
                f"| `{s.rtf:.6f}` | `{lat}` | `{npu}` |"
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
        "- **CV** — coefficient of variation `stddev / mean`. On a "
        "quiescent Apple M-series with the ANE dedicated to this workload, "
        "CV values in the `0.01 – 0.05` range are typical for a "
        "well-behaved delegate. Snapdragon HTP has higher inherent jitter "
        "on shared silicon; `0.05 – 0.15` is normal on a hot devboard."
    )
    parts.append(
        "- **NPU fraction** — the share of hot ops the delegate actually "
        "kept on the NPU. `1.00` = pure NPU, `0.00` = pure CPU-fallback. "
        "The FR-EX-08 hazard clause treats anything below `0.90` as a "
        f"silent fallback that disqualifies the 2× verdict."
    )
    parts.append(
        "- **NFR-PF-12 baseline** — the CPU baseline for the 2× ratio is "
        "**M5-14-post CPU** (SIMD hot-path optimised, libm-route). Pair "
        "every NPU JSONL with a same-host CPU JSONL captured in the same "
        "session — an NPU RTF without a matched CPU baseline cannot feed "
        "the 2× verdict."
    )
    parts.append(
        "- **Formal 2× verdict** — do NOT promote off this report. Fold it "
        "into `docs/handoff/m5-01-coreml-bakeoff-template.md` (or the QNN "
        "sibling) and let the owner record the PASS / FAIL against the "
        "matched CPU baseline. The analyzer never returns non-zero on a "
        "'too slow' verdict."
    )
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="npu_rtf_analyze.py",
        description=(
            "Reduce npu_rtf_variance.sh JSONL into a markdown variance report. "
            "This is a reference analyzer — never asserts an RTF ceiling, "
            "but WARNs on CV > 0.20 and on placement < 90% (silent CPU "
            "fallback, FR-EX-08). See docs/m5-owner-verification-checklist.md "
            "§1.5 for the full workflow."
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
            "Output format. 'markdown' (default) is the human-readable "
            "report; 'json' emits the summary + samples as a machine-"
            "readable object so this analyzer can feed a downstream harness."
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
        prep = placement_report(samples)
        obj = {
            "summary": (
                None
                if summary is None
                else {
                    "iters_requested": summary.iters_requested,
                    "iters_failed": summary.iters_failed,
                    "started_at": summary.started_at,
                    "ended_at": summary.ended_at,
                    "backend": summary.backend,
                    "label": summary.label,
                    "host": summary.host,
                    "device_name": summary.device_name,
                    "device_os": summary.device_os,
                    "device_soc": summary.device_soc,
                    "gguf": summary.gguf,
                    "audio": summary.audio,
                    "placement_probe": summary.placement_probe,
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
            "placement": {
                "total": prep.total,
                "with_probe": prep.with_probe,
                "with_npu_frac": prep.with_npu_frac,
                "below_threshold": prep.below_threshold,
                "mean_npu_frac": prep.mean_npu_frac,
                "min_npu_frac": prep.min_npu_frac,
                "warn_threshold": PLACEMENT_WARN_THRESHOLD,
                "placement_warn": (
                    prep.with_npu_frac == 0
                    or prep.below_threshold > 0
                ),
            },
            "samples": [
                {
                    "iter": s.iter,
                    "rtf": s.rtf,
                    "latency_ms": s.latency_ms,
                    "backend": s.backend,
                    "timestamp": s.timestamp,
                    "npu_frac": s.npu_frac,
                    "placement": s.placement_raw,
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
    #   0 — analysis completed (regardless of CV or placement WARN)
    #   1 — no successful samples (every iteration failed)
    # We intentionally do NOT surface CV > 0.20 or placement < 90 % as a
    # non-zero exit — they are report warnings, not gates. See module
    # docstring.
    if not samples:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
