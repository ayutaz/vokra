#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Causal tests for Dia's direct-entry compatibility admission boundary."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DUMPER_PATH = ROOT / "tools" / "parity" / "dia_1_6b_dump_reference.py"
WORKER_PATH = ROOT / "scripts" / "publish" / "vast-ai" / "run-dia-1-6b-validation.sh"
MARKER = "BLOCKED_UPSTREAM_PINNED_COMPATIBILITY"


def load_dumper():
    spec = importlib.util.spec_from_file_location("dia_dump_reference_admission_target", DUMPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke_main(module, *args: str):
    old_argv = sys.argv
    sys.argv = [str(DUMPER_PATH), *args]
    try:
        return module.main()
    finally:
        sys.argv = old_argv


def expect_blocked(callable_, *args):
    try:
        callable_(*args)
    except RuntimeError as error:
        assert MARKER in str(error), str(error)
    else:
        raise AssertionError("direct execution was not rejected")


def test_cli_malformed_input_is_blocked_before_path_validation() -> None:
    module = load_dumper()
    calls: list[tuple[object, ...]] = []

    def forbidden_path(*args):
        calls.append(args)
        raise AssertionError("cli_path was reached before compatibility admission")

    original = module.cli_path
    module.cli_path = forbidden_path
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                invoke_main(module, "--source", "relative/input")
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError("CLI did not terminate with parser error")
        assert MARKER in stderr.getvalue()
        assert "Traceback" not in stderr.getvalue()
    finally:
        module.cli_path = original
    assert calls == []


def test_script_entrypoint_reports_compatibility_block_before_path_error() -> None:
    result = subprocess.run(
        [sys.executable, str(DUMPER_PATH), "--source", "relative/input"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert MARKER in combined
    assert "Traceback" not in combined
    assert "output path must" not in combined
    assert "absolute" not in combined


def test_programmatic_run_is_blocked_before_any_side_effect() -> None:
    module = load_dumper()
    calls: list[tuple[object, ...]] = []

    def forbidden(*args, **kwargs):
        calls.append((*args, *kwargs.values()))
        raise AssertionError("execution side effect was reached before admission")

    original_binding = module.validate_binding
    original_identity = module.reference_project_identity
    original_import = module.importlib.import_module
    module.validate_binding = forbidden
    module.reference_project_identity = forbidden
    module.importlib.import_module = forbidden
    try:
        expect_blocked(
            module.run,
            *([object()] * 7),
            "fixed text",
            0,
            *(["0" * 40, "0" * 64, "1" * 64, "2" * 64, "VALIDATED", "NO_UPLOAD"]),
        )
    finally:
        module.validate_binding = original_binding
        module.reference_project_identity = original_identity
        module.importlib.import_module = original_import
    assert calls == []


def test_help_and_self_test_remain_model_free() -> None:
    module = load_dumper()
    original_import = module.importlib.import_module

    def forbidden_import(*args, **kwargs):
        raise AssertionError("target package import reached during non-execution mode")

    module.importlib.import_module = forbidden_import
    old_torchaudio = sys.modules.pop("torchaudio", None)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                invoke_main(module, "--help")
            except SystemExit as error:
                assert error.code == 0
            else:
                raise AssertionError("--help did not terminate normally")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert invoke_main(module, "--self-test") == 0
    finally:
        module.importlib.import_module = original_import
        sys.modules.pop("torchaudio", None)
        if old_torchaudio is not None:
            sys.modules["torchaudio"] = old_torchaudio


def test_worker_gate_remains_before_preparation_and_inputs() -> None:
    worker = WORKER_PATH.read_text(encoding="utf-8")
    assert f'SECURITY_REFRESH_STATUS="{MARKER}"' in worker
    assert "SECURITY_REFRESH_STATUS" in worker
    assert "check_project_identity\nsecurity_refresh_gate\nusage()" in worker


def parse_runner_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="count",
        default=0,
        help="run the offline stdlib admission suite (required exactly once)",
    )
    args = parser.parse_args(argv)
    if args.self_test != 1:
        parser.error("exactly one --self-test is required")
    return args


def test_runner_requires_explicit_self_test_mode() -> None:
    for invalid in ([], ["--unknown"], ["--self-test", "--self-test"]):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            try:
                parse_runner_args(invalid)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"runner accepted invalid arguments: {invalid}")
        assert "usage:" in stderr.getvalue()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            parse_runner_args(["--help"])
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError("runner --help did not terminate normally")


def main(argv: list[str] | None = None) -> int:
    parse_runner_args(argv)
    tests = (
        test_cli_malformed_input_is_blocked_before_path_validation,
        test_script_entrypoint_reports_compatibility_block_before_path_error,
        test_programmatic_run_is_blocked_before_any_side_effect,
        test_help_and_self_test_remain_model_free,
        test_worker_gate_remains_before_preparation_and_inputs,
        test_runner_requires_explicit_self_test_mode,
    )
    for test in tests:
        test()
    print("dia execution admission tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
