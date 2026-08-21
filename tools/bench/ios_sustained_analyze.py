"""Validate and summarize a physical-iPhone sustained codec JSONL log.

Run through the repository's uv environment:

    uv run --project tools/parity python tools/bench/ios_sustained_analyze.py LOG.jsonl

The analyzer is deliberately fail-closed: a 30-minute target in metadata is
not enough. It also requires the corresponding frame count, contiguous frame
indices, and at least 1800 seconds of observed wall time before reporting
percentiles. No pass/fail performance threshold is invented; the report gives
the exact last-decile/first-decile median ratio and whether the last median was
numerically slower.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA = "vokra.ios-codec-sustained.v1"
MIN_SUSTAINED_SECONDS = 1_800.0
THERMAL_STATES = {"nominal", "fair", "serious", "critical"}


class AnalysisError(Exception):
    """The log cannot substantiate the claimed physical-device run."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects, retaining line numbers in failures."""
    source = Path(path)
    if not source.is_file():
        raise AnalysisError(f"log file not found: {source}")
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise AnalysisError(f"line {line_no} is not a JSON object")
        records.append(value)
    if not records:
        raise AnalysisError("log contains no JSON records")
    return records


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        suffix = " and > 0" if positive else ""
        raise AnalysisError(f"{label} must be finite{suffix}")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AnalysisError(f"{label} must be an integer > 0")
    return value


def _required_string(mapping: dict[str, Any], key: str, prefix: str = "metadata") -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{prefix}.{key} must be a non-empty string")
    return value


def _nearest_rank(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise AnalysisError("cannot compute a percentile over no frames")
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": _nearest_rank(ordered, 0.50),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
    }


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a v1 JSONL record list and return a reproducible summary."""
    if not records or records[0].get("kind") != "metadata":
        raise AnalysisError("line 1 must be the single metadata record")
    if any(record.get("kind") == "metadata" for record in records[1:]):
        raise AnalysisError("log must contain exactly one metadata record")
    meta = records[0]
    if meta.get("schema") != SCHEMA:
        raise AnalysisError(f"metadata.schema must equal {SCHEMA!r}")

    build_sha = _required_string(meta, "build_sha")
    model_sha256 = _required_string(meta, "model_sha256")
    if len(build_sha) != 40 or any(c not in "0123456789abcdef" for c in build_sha.lower()):
        raise AnalysisError("metadata.build_sha must be 40 hexadecimal characters")
    if len(model_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in model_sha256.lower()
    ):
        raise AnalysisError("metadata.model_sha256 must be 64 hexadecimal characters")
    device_model = _required_string(meta, "device_model")
    ios_version = _required_string(meta, "ios_version")
    backend = _required_string(meta, "backend")

    conditions = meta.get("conditions")
    if not isinstance(conditions, dict):
        raise AnalysisError("metadata.conditions must be an object")
    ambient = _finite_number(
        conditions.get("ambient_temperature_c"),
        "metadata.conditions.ambient_temperature_c",
    )
    starting_thermal = _required_string(conditions, "starting_thermal_state", "conditions")
    if starting_thermal not in THERMAL_STATES:
        raise AnalysisError(
            "metadata.conditions.starting_thermal_state must be nominal/fair/serious/critical"
        )
    screen = _required_string(conditions, "screen", "conditions")
    if screen not in {"on", "off"}:
        raise AnalysisError("metadata.conditions.screen must be 'on' or 'off'")
    charging = conditions.get("charging")
    if not isinstance(charging, bool):
        raise AnalysisError("metadata.conditions.charging must be boolean")
    case = _required_string(conditions, "case", "conditions")
    if case not in {"installed", "removed"}:
        raise AnalysisError("metadata.conditions.case must be 'installed' or 'removed'")

    sample_rate = _positive_int(meta.get("sample_rate"), "metadata.sample_rate")
    frame_hop = _positive_int(meta.get("frame_hop"), "metadata.frame_hop")
    n_codebooks = _positive_int(meta.get("n_codebooks"), "metadata.n_codebooks")
    target_duration = _finite_number(
        meta.get("target_duration_s"), "metadata.target_duration_s", positive=True
    )

    frames = records[1:]
    if not frames:
        raise AnalysisError("log contains no frame records")
    decode_ms: list[float] = []
    peak_rss = 0
    end_thermal = starting_thermal
    wall_elapsed = 0.0
    for expected_index, frame in enumerate(frames):
        if frame.get("kind") != "frame":
            raise AnalysisError(f"record {expected_index + 2} kind must be 'frame'")
        if frame.get("index") != expected_index:
            raise AnalysisError(
                f"frame indices must be contiguous from 0: expected {expected_index}, "
                f"got {frame.get('index')!r}"
            )
        latency = _finite_number(frame.get("decode_ms"), f"frame[{expected_index}].decode_ms")
        if latency < 0.0:
            raise AnalysisError(f"frame[{expected_index}].decode_ms must be >= 0")
        elapsed = _finite_number(
            frame.get("wall_elapsed_s"),
            f"frame[{expected_index}].wall_elapsed_s",
            positive=True,
        )
        if elapsed <= wall_elapsed:
            raise AnalysisError("frame wall_elapsed_s values must be strictly increasing")
        rss = _positive_int(frame.get("peak_rss_bytes"), f"frame[{expected_index}].peak_rss_bytes")
        thermal = frame.get("thermal_state")
        if thermal not in THERMAL_STATES:
            raise AnalysisError(
                f"frame[{expected_index}].thermal_state must be nominal/fair/serious/critical"
            )
        decode_ms.append(latency)
        wall_elapsed = elapsed
        peak_rss = max(peak_rss, rss)
        end_thermal = thermal

    # A log cannot claim a reduced target and still satisfy issue #52.
    if target_duration < MIN_SUSTAINED_SECONDS:
        raise AnalysisError(
            f"target_duration_s {target_duration:g} is below the required "
            f"{MIN_SUSTAINED_SECONDS:g} seconds"
        )
    frame_period_s = frame_hop / sample_rate
    expected_frames = math.ceil(target_duration / frame_period_s)
    if len(frames) < expected_frames:
        raise AnalysisError(
            f"frame count {len(frames)} is below {expected_frames} required for "
            f"{target_duration:g}s at hop={frame_hop}, rate={sample_rate}"
        )
    if wall_elapsed < target_duration:
        raise AnalysisError(
            f"observed duration {wall_elapsed:g}s is below target {target_duration:g}s"
        )

    decile_count = max(1, len(decode_ms) // 10)
    first_p50 = _percentiles(decode_ms[:decile_count])["p50"]
    last_p50 = _percentiles(decode_ms[-decile_count:])["p50"]
    ratio = last_p50 / first_p50 if first_p50 > 0.0 else math.inf
    deadline_ms = frame_period_s * 1_000.0
    deadline_misses = sum(latency > deadline_ms for latency in decode_ms)

    return {
        "schema": "vokra.ios-codec-sustained-report.v1",
        "source_schema": SCHEMA,
        "build_sha": build_sha,
        "model_sha256": model_sha256,
        "device_model": device_model,
        "ios_version": ios_version,
        "backend": backend,
        "sample_rate": sample_rate,
        "frame_hop": frame_hop,
        "n_codebooks": n_codebooks,
        "target_duration_s": target_duration,
        "actual_duration_s": wall_elapsed,
        "frame_count": len(frames),
        "decode_ms": _percentiles(decode_ms),
        "frame_deadline_ms": deadline_ms,
        "deadline_miss_frames": deadline_misses,
        "peak_rss_bytes": peak_rss,
        "start_thermal_state": starting_thermal,
        "end_thermal_state": end_thermal,
        "conditions": {
            "ambient_temperature_c": ambient,
            "screen": screen,
            "charging": charging,
            "case": case,
        },
        "degradation": {
            "comparison": "last 10% p50 / first 10% p50",
            "first_decile_p50_ms": first_p50,
            "last_decile_p50_ms": last_p50,
            "last_to_first_ratio": ratio,
            "last_decile_slower": last_p50 > first_p50,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the summary in a form ready for the device benchmark doc."""
    latency = report["decode_ms"]
    trend = report["degradation"]
    conditions = report["conditions"]
    return "\n".join(
        [
            "# iOS sustained codec run",
            "",
            f"- Device: `{report['device_model']}` / iOS `{report['ios_version']}`",
            f"- Backend: `{report['backend']}`",
            f"- Build SHA: `{report['build_sha']}`",
            f"- Model SHA-256: `{report['model_sha256']}`",
            f"- Duration: `{report['actual_duration_s']:.3f} s` "
            f"({report['frame_count']} frames)",
            f"- Decode latency: p50 `{latency['p50']:.6f} ms`, "
            f"p95 `{latency['p95']:.6f} ms`, p99 `{latency['p99']:.6f} ms`",
            f"- Frame deadline: `{report['frame_deadline_ms']:.6f} ms`; "
            f"misses: `{report['deadline_miss_frames']}`",
            f"- Peak RSS: `{report['peak_rss_bytes']} bytes`",
            f"- Thermal state: `{report['start_thermal_state']}` → "
            f"`{report['end_thermal_state']}`",
            f"- Last/first decile p50 ratio: `{trend['last_to_first_ratio']:.6f}`; "
            f"last decile slower: `{str(trend['last_decile_slower']).lower()}`",
            f"- Conditions: ambient `{conditions['ambient_temperature_c']:.1f} °C`, "
            f"screen `{conditions['screen']}`, charging "
            f"`{str(conditions['charging']).lower()}`, case `{conditions['case']}`",
            "",
        ]
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        report = analyze_records(load_jsonl(args.log))
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.markdown_output:
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
