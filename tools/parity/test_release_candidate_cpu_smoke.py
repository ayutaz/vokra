#!/usr/bin/env python3
"""Offline tests for release_candidate_cpu_smoke.py.

These tests mock the CLI/network boundary and never create or read a model
file.  The actual functional smoke is intentionally hosted-runner only.
"""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("release_candidate_cpu_smoke.py")
SPEC = importlib.util.spec_from_file_location("release_candidate_cpu_smoke", SCRIPT)
assert SPEC and SPEC.loader
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def valid_manifest() -> dict[str, dict[str, object]]:
    return {
        "whisper-base": {
            "repo": "vokra/whisper-base",
            "revision": "a" * 40,
            "filename": "whisper-base.gguf",
            "sha256": "b" * 64,
            "size_bytes": 10,
        },
        "silero-vad-v5": {
            "repo": "vokra/silero-vad-v5",
            "revision": "c" * 40,
            "filename": "silero-vad-v5.gguf",
            "sha256": "d" * 64,
            "size_bytes": 11,
        },
    }


class FakePipe:
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.closed = False

    def fileno(self) -> int:
        return self.fd

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.pid = 4242
        self.stdout = FakePipe(1)
        self.stderr = FakePipe(2)
        self.returncode = returncode
        self.wait_calls: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout or 0)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeSelector:
    def __init__(self) -> None:
        self.keys: dict[FakePipe, object] = {}
        self.closed = False

    def register(self, stream: FakePipe, event: object, data: int) -> None:
        self.keys[stream] = type("Key", (), {"fileobj": stream, "data": data})()

    def unregister(self, stream: FakePipe) -> None:
        self.keys.pop(stream, None)

    def get_map(self) -> dict[FakePipe, object]:
        return self.keys

    def select(self, timeout: float) -> list[tuple[object, object]]:
        return [(key, None) for key in self.keys.values()]

    def close(self) -> None:
        self.closed = True


class ReleaseCandidateSmokeTests(unittest.TestCase):
    def _synthetic_inputs(self, directory: str) -> tuple[dict[str, smoke.ModelPin], Path, Path, Path]:
        root = Path(directory)
        audio = root / "tiny.wav"
        with wave.open(str(audio), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(b"\0\0" * 4)
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        audio.with_name("tiny.wav.sha256").write_text(f"{digest}  tiny.wav\n", encoding="utf-8")
        cli = root / "fake-cli"
        cli.write_bytes(b"offline test executable placeholder")
        cli.chmod(0o755)
        pins = smoke.parse_pins(valid_manifest())
        return pins, audio, cli, root / "receipt.json"

    def _run_synthetic_smoke(
        self,
        directory: str,
        outputs: list[object],
        *,
        download_error: Exception | None = None,
    ) -> tuple[dict[str, object], mock.Mock]:
        pins, audio, cli, receipt = self._synthetic_inputs(directory)
        calls: list[list[str]] = []

        def downloader(pin: smoke.ModelPin, destination: Path) -> tuple[str, int]:
            if download_error:
                raise download_error
            return pin.sha256, pin.size_bytes

        def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            output = outputs[len(calls) - 1]
            if isinstance(output, BaseException):
                raise output
            return output

        printer = mock.Mock()
        with mock.patch.object(smoke, "require_hosted_linux", return_value="a" * 40), mock.patch(
            "builtins.print", printer
        ):
            try:
                smoke.run_smoke(
                    pins=pins,
                    cli=cli,
                    audio=audio,
                    receipt=receipt,
                    expected_head="a" * 40,
                    repo_root=Path(directory),
                    downloader=downloader,
                    cli_runner=runner,
                    enforce_canonical_audio=False,
                )
            except smoke.SmokeError:
                pass
        return json.loads(receipt.read_text(encoding="utf-8")), printer

    def test_manifest_requires_exact_allowlisted_records(self) -> None:
        pins = smoke.parse_pins(valid_manifest())
        self.assertEqual(pins["whisper-base"].url, "https://huggingface.co/vokra/whisper-base/resolve/" + "a" * 40 + "/whisper-base.gguf")
        self.assertEqual(pins["silero-vad-v5"].size_bytes, 11)
        with self.assertRaises(smoke.SmokeError):
            smoke.parse_pins({})
        broken = valid_manifest()
        broken["whisper-base"]["repo"] = "openai/whisper-base"
        with self.assertRaises(smoke.SmokeError):
            smoke.parse_pins(broken)

    def test_manifest_rejects_unsafe_or_unbounded_values(self) -> None:
        for field, value in (
            ("revision", "not-a-revision"),
            ("filename", "../model.gguf"),
            ("sha256", "not-a-digest"),
            ("size_bytes", 0),
        ):
            manifest = valid_manifest()
            manifest["whisper-base"][field] = value
            with self.subTest(field=field), self.assertRaises(smoke.SmokeError):
                smoke.parse_pins(manifest)

    def test_stdout_contracts_are_semantic_and_strict(self) -> None:
        normalized = smoke._normalize_transcript(
            "asr: And so, my fellow Americans, ask not what your country can do for you.\n"
        )
        self.assertIn(smoke.ASR_KEYPHRASE, normalized)
        self.assertEqual(
            smoke._parse_vad("vad: 12 frames, speech_frames=3, mean_prob=0.2500\n", "test"),
            {"frames": 12, "speech_frames": 3, "mean_prob": 0.25},
        )
        with self.assertRaises(smoke.SmokeError):
            smoke._parse_vad("vad: 12 frames, speech_frames=13, mean_prob=0.2500\n", "test")
        with self.assertRaises(smoke.SmokeError):
            smoke._normalize_transcript("asr:\n")

    def test_cli_failure_is_never_a_success(self) -> None:
        def failed_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args[0], 17, "partial\n", "failure\n")

        records: list[smoke.CommandRecord] = []
        with self.assertRaises(smoke.SmokeError) as context:
            smoke._run_cli("mock-failure", ["vokra-cli", "run"], records, runner=failed_runner)
        self.assertIn("returned 17", str(context.exception))
        self.assertEqual(records[0].returncode, 17)
        self.assertNotIn("PASS", str(context.exception))

    def test_production_streaming_capture_cleans_up_timeout_overflow_and_read_error(self) -> None:
        def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
            return process

        process = FakeProcess()
        reads = {1: [b"ok", b""], 2: [b""]}
        terminate = mock.Mock(return_value=None)

        def read_ok(fd: int, size: int) -> bytes:
            return reads[fd].pop(0)

        result = smoke._run_cli_streaming(
            ["fake"],
            popen_factory=fake_popen,
            selector_factory=FakeSelector,
            read_fn=read_ok,
            set_blocking=lambda fd, enabled: None,
            clock=lambda: 0.0,
            terminate=terminate,
        )
        self.assertEqual(result[:3], (0, b"ok", b""))
        self.assertTrue(process.stdout.closed and process.stderr.closed)
        terminate.assert_not_called()

        class IdleSelector(FakeSelector):
            def select(self, timeout: float) -> list[tuple[object, object]]:
                return []

        for failure_kind, read_fn, process_state in (
            ("timeout", lambda fd, size: b"", FakeProcess(returncode=0)),
            ("overflow", lambda fd, size: b"x" * (smoke.MAX_COMMAND_OUTPUT_BYTES + 1), FakeProcess()),
            ("read", lambda fd, size: (_ for _ in ()).throw(OSError("read failed")), FakeProcess()),
        ):
            with self.subTest(failure_kind=failure_kind):
                terminate = mock.Mock(return_value=None)
                selector = IdleSelector() if failure_kind == "timeout" else FakeSelector()
                read_calls: list[int] = []

                def case_popen(*args: object, **kwargs: object) -> FakeProcess:
                    return process_state

                clock_values = iter((0.0, float(smoke.MAX_COMMAND_SECONDS + 1)))

                def clock() -> float:
                    return next(clock_values) if failure_kind == "timeout" else 0.0

                def read_with_record(fd: int, size: int) -> bytes:
                    read_calls.append(fd)
                    return read_fn(fd, size)

                result = smoke._run_cli_streaming(
                    ["fake"],
                    popen_factory=case_popen,
                    selector_factory=lambda: selector,
                    read_fn=read_with_record,
                    set_blocking=lambda fd, enabled: None,
                    clock=clock,
                    terminate=terminate,
                )
                expected_error = {
                    "timeout": "deadline",
                    "overflow": "bounded output",
                    "read": "read failed",
                }[failure_kind]
                self.assertIn(expected_error, result[3] or "")
                if failure_kind != "timeout":
                    self.assertTrue(read_calls)
                self.assertTrue(selector.closed)
                self.assertTrue(process_state.stdout.closed and process_state.stderr.closed)
                terminate.assert_called_once_with(process_state)

    def test_streaming_register_failure_cleans_up(self) -> None:
        process = FakeProcess()
        terminate = mock.Mock(return_value=None)

        class RegisterFailureSelector(FakeSelector):
            def register(self, stream: FakePipe, event: object, data: int) -> None:
                raise OSError("register failed")

        selector = RegisterFailureSelector()
        result = smoke._run_cli_streaming(
            ["fake"],
            popen_factory=lambda *args, **kwargs: process,
            selector_factory=lambda: selector,
            read_fn=lambda fd, size: b"",
            set_blocking=lambda fd, enabled: None,
            clock=lambda: 0.0,
            terminate=terminate,
        )
        self.assertIn("register failed", result[3] or "")
        self.assertTrue(selector.closed)
        self.assertTrue(process.stdout.closed and process.stderr.closed)
        terminate.assert_called_once_with(process)

    def test_leader_exit_still_cleans_owned_process_group_when_child_holds_pipe(self) -> None:
        process = FakeProcess(returncode=0)
        terminate = mock.Mock(return_value=None)
        clock_values = iter((0.0, float(smoke.MAX_COMMAND_SECONDS + 1)))
        selector = type(
            "ChildHoldingSelector",
            (FakeSelector,),
            {"select": lambda self, timeout: []},
        )()
        result = smoke._run_cli_streaming(
            ["fake"],
            popen_factory=lambda *args, **kwargs: process,
            selector_factory=lambda: selector,
            read_fn=lambda fd, size: b"",
            set_blocking=lambda fd, enabled: None,
            clock=lambda: next(clock_values),
            terminate=terminate,
        )
        self.assertIn("deadline", result[3] or "")
        self.assertTrue(selector.closed)
        self.assertTrue(process.stdout.closed and process.stderr.closed)
        terminate.assert_called_once_with(process)

    def test_owned_group_cleanup_reports_kill_diagnostic_after_reaped_leader(self) -> None:
        process = FakeProcess(returncode=0)
        with mock.patch.object(
            smoke.os,
            "killpg",
            side_effect=[None, PermissionError("kill denied")],
        ) as killpg:
            diagnostic = smoke._terminate_owned_process(process)
        self.assertIn("process-group kill failed", diagnostic or "")
        self.assertEqual(
            [call.args for call in killpg.call_args_list],
            [(process.pid, smoke.signal.SIGTERM), (process.pid, smoke.signal.SIGKILL)],
        )

    def test_semantic_parsers_see_full_bounded_capture_not_receipt_clip(self) -> None:
        long_prefix = "x" * (smoke.MAX_RECEIPT_TEXT_BYTES + 32)

        def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args[0], 0, f"asr: first\n{long_prefix}\nasr: second\n", ""
            )

        records: list[smoke.CommandRecord] = []
        with self.assertRaises(smoke.SmokeError):
            smoke._normalize_transcript(smoke._run_cli("second-asr", ["fake"], records, runner=runner))

        def vad_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args[0], 0, f"vad: 1 frames, speech_frames=0, mean_prob=0.0000\n{long_prefix}\n", ""
            )

        with self.assertRaises(smoke.SmokeError):
            smoke._parse_vad(
                smoke._run_cli("vad-junk", ["fake"], [], runner=vad_runner), "vad-junk"
            )

    def test_download_boundary_is_network_mocked_and_hash_checked(self) -> None:
        content = b"offline-test-payload"
        pin = smoke.ModelPin(
            "whisper-base",
            "vokra/whisper-base",
            "a" * 40,
            "fixture.gguf",
            hashlib.sha256(content).hexdigest(),
            len(content),
        )

        class FakeResponse(io.BytesIO):
            headers = {"Content-Length": str(len(content))}

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

            def geturl(self) -> str:
                return pin.url

        class FakeOpener:
            def open(self, request: object, timeout: int) -> FakeResponse:
                return FakeResponse(content)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "payload.bin"
            with mock.patch.object(smoke.urllib.request, "build_opener", return_value=FakeOpener()) as opener:
                digest, size = smoke._download(pin, destination)
            opener.assert_called_once()
            self.assertEqual((digest, size), (pin.sha256, pin.size_bytes))
            self.assertEqual(destination.read_bytes(), content)

    def test_redirect_handler_rejects_non_https(self) -> None:
        request = smoke.urllib.request.Request("https://huggingface.co/vokra/whisper-base")
        with self.assertRaises(smoke.SmokeError):
            smoke._HttpsOnlyRedirectHandler().redirect_request(
                request, None, 302, "redirect", {}, "http://example.invalid/model.gguf"
            )

    def test_remote_guards_fail_closed_offline(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(smoke.SmokeError):
                smoke.require_hosted_linux("a" * 40, Path.cwd())
        with mock.patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true", "VOKRA_RELEASE_CPU_SMOKE": "1"},
            clear=True,
        ), mock.patch.object(smoke, "_checked_out_head", return_value="b" * 40):
            with self.assertRaises(smoke.SmokeError):
                smoke.require_hosted_linux("a" * 40, Path.cwd())

    def test_mocked_success_receipt_binds_audio_silence_and_vad_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, printer = self._run_synthetic_smoke(
                directory,
                [
                    subprocess.CompletedProcess([], 0, "asr: ask not what your country can do for you\n", ""),
                    subprocess.CompletedProcess([], 0, "vad: 2 frames, speech_frames=1, mean_prob=0.5000\n", ""),
                    subprocess.CompletedProcess([], 0, "vad: 2 frames, speech_frames=0, mean_prob=0.0000\n", ""),
                ],
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["audio"]["geometry"]["frames"], 4)
            self.assertEqual(receipt["generated_silence"]["frames"], 4)
            self.assertEqual(receipt["semantic_checks"]["vad"]["jfk"]["frames"], 2)
            printer.assert_called_once_with("PASS release-candidate-cpu-smoke")

    def test_mocked_failures_write_fail_receipts_and_never_pass(self) -> None:
        cases = {
            "hash": ([], smoke.SmokeError("hash mismatch")),
            "phrase": ([subprocess.CompletedProcess([], 0, "asr: unrelated text\n", "")], None),
            "vad-silence": [
                subprocess.CompletedProcess([], 0, "asr: ask not what your country can do for you\n", ""),
                subprocess.CompletedProcess([], 0, "vad: 2 frames, speech_frames=1, mean_prob=0.5\n", ""),
                subprocess.CompletedProcess([], 0, "vad: 2 frames, speech_frames=1, mean_prob=0.5\n", ""),
            ],
            "vad-frame-mismatch": [
                subprocess.CompletedProcess([], 0, "asr: ask not what your country can do for you\n", ""),
                subprocess.CompletedProcess([], 0, "vad: 2 frames, speech_frames=1, mean_prob=0.5\n", ""),
                subprocess.CompletedProcess([], 0, "vad: 3 frames, speech_frames=0, mean_prob=0.0\n", ""),
            ],
            "cli-nonzero": [subprocess.CompletedProcess([], 9, "", "boom\n")],
            "cli-timeout": [subprocess.TimeoutExpired(["fake-cli"], 1)],
            "cli-overflow": [subprocess.CompletedProcess([], 0, "x" * (smoke.MAX_COMMAND_OUTPUT_BYTES + 1), "")],
        }
        for name, case in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                if isinstance(case, tuple):
                    outputs, download_error = case
                else:
                    outputs, download_error = case, None
                receipt, printer = self._run_synthetic_smoke(directory, outputs, download_error=download_error)
                self.assertEqual(receipt["status"], "FAIL")
                printer.assert_not_called()

    def test_pins_json_duplicate_and_size_bound_fail_before_download(self) -> None:
        duplicate = '{"whisper-base": {}, "whisper-base": {}}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pins.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(smoke.SmokeError):
                smoke._read_json(path)
            path.write_bytes(b"x" * (smoke.MAX_PINS_JSON_BYTES + 1))
            with self.assertRaises(smoke.SmokeError):
                smoke._read_json(path)

        class BoundedReader:
            def __init__(self) -> None:
                self.sizes: list[int] = []

            def __enter__(self) -> "BoundedReader":
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def read(self, size: int) -> bytes:
                self.sizes.append(size)
                return b"x" * size

        bounded_reader = BoundedReader()
        with mock.patch.object(Path, "open", return_value=bounded_reader):
            with self.assertRaises(smoke.SmokeError):
                smoke._read_json(Path("bounded.json"))
        self.assertEqual(bounded_reader.sizes, [smoke.MAX_PINS_JSON_BYTES + 1])

        manifest = valid_manifest()
        manifest["whisper-base"]["size_bytes"] = smoke.MAX_MODEL_BYTES - 1
        manifest["silero-vad-v5"]["size_bytes"] = 2
        with self.assertRaises(smoke.SmokeError):
            smoke.parse_pins(manifest)

    def test_wrong_environment_guard_prevents_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pins, audio, cli, receipt = self._synthetic_inputs(directory)
            downloader = mock.Mock(side_effect=AssertionError("download must not run"))
            with mock.patch.object(smoke, "require_hosted_linux", side_effect=smoke.SmokeError("wrong host")):
                with self.assertRaises(smoke.SmokeError):
                    smoke.run_smoke(
                        pins=pins,
                        cli=cli,
                        audio=audio,
                        receipt=receipt,
                        expected_head="a" * 40,
                        repo_root=Path(directory),
                        downloader=downloader,
                        enforce_canonical_audio=False,
                    )
            downloader.assert_not_called()

    def test_receipt_output_is_bounded(self) -> None:
        payload = {"status": "FAIL", "stdout": "x" * (smoke.MAX_RECEIPT_TEXT_BYTES * 2)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            smoke.write_receipt(path, payload)
            self.assertLessEqual(path.stat().st_size, smoke.MAX_RECEIPT_BYTES)

        exact_payload = {"status": "FAIL"}
        encoded_size = len(json.dumps(exact_payload, indent=2, sort_keys=True).encode("utf-8"))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            smoke, "MAX_RECEIPT_BYTES", encoded_size
        ):
            with self.assertRaises(smoke.SmokeError):
                smoke.write_receipt(Path(directory) / "too-large.json", exact_payload)


if __name__ == "__main__":
    unittest.main()
