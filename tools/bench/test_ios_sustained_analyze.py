import json
import math
import tempfile
import unittest
from pathlib import Path

from ios_sustained_analyze import AnalysisError, analyze_records, load_jsonl


def metadata(target_duration_s=1_800.0):
    return {
        "kind": "metadata",
        "schema": "vokra.ios-codec-sustained.v1",
        "build_sha": "a" * 40,
        "model_sha256": "b" * 64,
        "device_model": "iPhone Test",
        "ios_version": "26.0",
        "backend": "cpu",
        "sample_rate": 24_000,
        "frame_hop": 1_920,
        "n_codebooks": 8,
        "target_duration_s": target_duration_s,
        "conditions": {
            "ambient_temperature_c": 23.5,
            "starting_thermal_state": "nominal",
            "screen": "on",
            "charging": False,
            "case": "removed",
        },
    }


def complete_records(target_duration_s=1_800.0):
    meta = metadata(target_duration_s)
    period = meta["frame_hop"] / meta["sample_rate"]
    count = math.ceil(target_duration_s / period)
    records = [meta]
    for index in range(count):
        # Deliberately make the final decile slower so the trend field is
        # independently testable from the aggregate percentiles.
        decode_ms = 2.0 if index < count * 0.9 else 3.0
        records.append(
            {
                "kind": "frame",
                "index": index,
                "wall_elapsed_s": (index + 1) * period,
                "decode_ms": decode_ms,
                "peak_rss_bytes": 100_000_000 + index,
                "thermal_state": "nominal",
            }
        )
    return records


class IosSustainedAnalyzeTests(unittest.TestCase):
    def test_complete_30_minute_run_reports_percentiles_rss_and_trend(self):
        records = complete_records()
        report = analyze_records(records)
        self.assertEqual(report["frame_count"], 22_500)
        self.assertAlmostEqual(report["actual_duration_s"], 1_800.0)
        self.assertEqual(report["decode_ms"]["p50"], 2.0)
        self.assertEqual(report["decode_ms"]["p95"], 3.0)
        self.assertEqual(report["decode_ms"]["p99"], 3.0)
        self.assertEqual(report["peak_rss_bytes"], 100_000_000 + 22_499)
        self.assertEqual(report["degradation"]["first_decile_p50_ms"], 2.0)
        self.assertEqual(report["degradation"]["last_decile_p50_ms"], 3.0)
        self.assertTrue(report["degradation"]["last_decile_slower"])
        self.assertEqual(report["deadline_miss_frames"], 0)

    def test_short_or_sparse_run_is_rejected_instead_of_looking_complete(self):
        records = complete_records(target_duration_s=1.0)
        records[0]["target_duration_s"] = 1_800.0
        with self.assertRaisesRegex(AnalysisError, "frame count"):
            analyze_records(records)

    def test_full_frame_count_cannot_hide_short_observed_wall_time(self):
        records = complete_records()
        period = records[0]["frame_hop"] / records[0]["sample_rate"]
        for frame in records[1:]:
            frame["wall_elapsed_s"] -= period / 2
        with self.assertRaisesRegex(AnalysisError, "observed duration"):
            analyze_records(records)

    def test_missing_measurement_condition_is_rejected(self):
        records = complete_records(target_duration_s=1.0)
        del records[0]["conditions"]["case"]
        with self.assertRaisesRegex(AnalysisError, "conditions.case"):
            analyze_records(records)

    def test_non_contiguous_frame_index_is_rejected(self):
        records = complete_records(target_duration_s=1.0)
        records[3]["index"] = 99
        with self.assertRaisesRegex(AnalysisError, "contiguous"):
            analyze_records(records)

    def test_non_finite_latency_is_rejected(self):
        records = complete_records(target_duration_s=1.0)
        records[1]["decode_ms"] = float("nan")
        with self.assertRaisesRegex(AnalysisError, "decode_ms"):
            analyze_records(records)

    def test_jsonl_loader_rejects_invalid_json_with_line_number(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            path.write_text(json.dumps(metadata(1.0)) + "\n{bad}\n", encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "line 2"):
                load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
