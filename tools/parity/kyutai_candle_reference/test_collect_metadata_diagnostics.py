from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("collect_metadata.py")
SPEC = importlib.util.spec_from_file_location("collect_metadata_diagnostics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _BinaryStderr:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def flush(self) -> None:
        pass


def _row(name: str, *, source: str | None = "registry+https://registry.invalid/index", dependencies: list[str] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "version": "1.0.0",
        "source": source,
        "checksum": "a" * 64,
        "dependencies": dependencies or [],
    }
    return row


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_key(value: str) -> str:
    if all(char.isalnum() or char in "_-" for char in value):
        return value
    return json.dumps(value)


def _write_lock(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["version = 4", ""]
    for row in rows:
        lines.append("[[package]]")
        for key, value in row.items():
            if isinstance(value, list):
                lines.append(f"{_toml_key(key)} = {json.dumps(value)}")
            elif value is None:
                continue
            else:
                lines.append(f"{_toml_key(key)} = {_toml_string(str(value))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate(original_rows: list[dict[str, object]], projected_rows: list[dict[str, object]]) -> tuple[object, bytes]:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        original = root / "original.lock"
        projected = root / "projected.lock"
        _write_lock(original, original_rows)
        _write_lock(projected, projected_rows)
        stderr = _BinaryStderr()
        old_stderr = MODULE.sys.stderr
        MODULE.sys.stderr = stderr
        try:
            result = MODULE.validate_base_lock(original, projected)
        except Exception:
            raise
        finally:
            MODULE.sys.stderr = old_stderr
        return result, stderr.buffer.getvalue()


def _reject(original_rows: list[dict[str, object]], projected_rows: list[dict[str, object]], expected_error: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        original = root / "original.lock"
        projected = root / "projected.lock"
        _write_lock(original, original_rows)
        _write_lock(projected, projected_rows)
        stderr = _BinaryStderr()
        old_stderr = MODULE.sys.stderr
        MODULE.sys.stderr = stderr
        try:
            with unittest.TestCase().assertRaisesRegex(MODULE.CaptureError, expected_error):
                MODULE.validate_base_lock(original, projected)
        finally:
            MODULE.sys.stderr = old_stderr
        output = stderr.buffer.getvalue()
    assert len(output) <= MODULE.MAX_LOCK_DIAGNOSTIC_BYTES
    line = output.decode("utf-8").strip()
    assert line.startswith("LOCK_DIAGNOSTIC: ")
    payload = json.loads(line.removeprefix("LOCK_DIAGNOSTIC: "))
    assert "https://" not in line
    return payload


class LockDiagnosticTests(unittest.TestCase):
    def test_dependency_membership_and_order_are_rejected_with_plain_labels(self) -> None:
        base = _row("backtrace", dependencies=["foo 1.0 (registry+https://secret.invalid/a)", "bar 2.0 (registry+https://secret.invalid/b)"])
        membership = dict(base)
        membership["dependencies"] = ["foo 1.0 (registry+https://secret.invalid/a)", "baz 3.0 (registry+https://secret.invalid/c)"]
        payload = _reject([base], [membership], "projected base lock mutated retained package: backtrace")
        diff = payload["retained_mismatches"]["items"][0]["field_diffs"]["dependencies"]
        self.assertTrue(diff["added"][0].startswith("baz@3.0[sha256:"))
        self.assertTrue(diff["removed"][0].startswith("bar@2.0[sha256:"))
        self.assertFalse(diff["order_changed"])
        self.assertNotIn("secret.invalid", json.dumps(payload))

        url_only = dict(base)
        url_only["dependencies"] = ["foo 1.0 (registry+https://secret.invalid/changed)", "bar 2.0 (registry+https://secret.invalid/b)"]
        payload = _reject([base], [url_only], "projected base lock mutated retained package: backtrace")
        diff = payload["retained_mismatches"]["items"][0]["field_diffs"]["dependencies"]
        self.assertEqual(diff["added_total"], 1)
        self.assertEqual(diff["removed_total"], 1)
        self.assertNotEqual(diff["added"][0], diff["removed"][0])
        self.assertNotIn("secret.invalid", json.dumps(payload))

        reordered = dict(base)
        reordered["dependencies"] = ["bar 2.0 (registry+https://secret.invalid/b)", "foo 1.0 (registry+https://secret.invalid/a)"]
        payload = _reject([base], [reordered], "projected base lock mutated retained package: backtrace")
        diff = payload["retained_mismatches"]["items"][0]["field_diffs"]["dependencies"]
        self.assertTrue(diff["order_changed"])
        self.assertEqual(diff["added_total"], 0)
        self.assertEqual(diff["removed_total"], 0)

    def test_scalar_checksum_and_unknown_field_are_rejected(self) -> None:
        base = _row("backtrace")
        changed_checksum = dict(base)
        changed_checksum["checksum"] = "b" * 64
        payload = _reject([base], [changed_checksum], "projected base lock mutated retained package: backtrace")
        self.assertEqual(payload["retained_mismatches"]["items"][0]["fields"], ["checksum"])
        self.assertEqual(payload["retained_mismatches"]["items"][0]["field_diffs"]["checksum"]["kind"], "value")

        unsafe_field = "https://secret.invalid/field"
        old_unknown = dict(base)
        old_unknown[unsafe_field] = "old"
        changed_unknown = dict(base)
        changed_unknown[unsafe_field] = "https://secret.invalid/not-output"
        payload = _reject([old_unknown], [changed_unknown], "projected base lock mutated retained package: backtrace")
        field_name = payload["retained_mismatches"]["items"][0]["fields"][0]
        self.assertTrue(field_name.startswith("<sha256:"))
        self.assertNotIn("secret.invalid", json.dumps(payload))

    def test_source_and_unknown_package_changes_are_rejected(self) -> None:
        base = _row("backtrace")
        changed_source = dict(base)
        changed_source["source"] = "registry+https://secret.invalid/other-index"
        payload = _reject([base], [changed_source], "projected base lock introduced an unknown package")
        self.assertEqual(payload["package_keys"]["added_total"], 1)
        self.assertEqual(payload["package_keys"]["removed_total"], 1)
        self.assertNotEqual(payload["package_keys"]["added"][0], payload["package_keys"]["removed"][0])
        self.assertNotIn("secret.invalid", json.dumps(payload))

        unknown = _row("new-package")
        payload = _reject([base], [base, unknown], "projected base lock introduced an unknown package")
        self.assertEqual(payload["package_keys"]["added_total"], 1)

    def test_unchanged_and_subset_locks_remain_positive(self) -> None:
        base = [_row("backtrace"), _row("keep")]
        result, diagnostic = _validate(base, base)
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(diagnostic, b"")
        result, diagnostic = _validate(base, [base[0]])
        self.assertEqual(result["status"], "SUBSET_UNCHANGED")
        self.assertEqual(diagnostic, b"")

    def test_large_diagnostic_is_bounded_and_marked_truncated(self) -> None:
        original = []
        projected = []
        for index in range(40):
            old = _row(f"pkg-{index}")
            new = dict(old)
            for field in range(20):
                new[f"unknown_{field}"] = "new-value"
                old[f"unknown_{field}"] = "old-value"
            original.append(old)
            projected.append(new)
        payload = _reject(original, projected, "projected base lock mutated retained package: pkg-0")
        self.assertTrue(payload["truncated"] or payload["retained_mismatches"].get("truncated"))


if __name__ == "__main__":
    unittest.main()
