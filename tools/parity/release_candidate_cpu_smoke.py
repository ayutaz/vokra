#!/usr/bin/env python3
"""Fail-closed hosted-Linux CPU smoke for the two release-candidate GGUFs.

This is a functional smoke, not a numerical-parity, Metal, or performance
measurement.  The model manifest is deliberately supplied by the release
owner at invocation time: this file does not contain (or infer) model pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlparse
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


EXPECTED_MODELS = {
    "whisper-base": "vokra/whisper-base",
    "silero-vad-v5": "vokra/silero-vad-v5",
}
EXPECTED_AUDIO = Path("tests/fixtures/audio/jfk-30s.wav")
MAX_MODEL_BYTES = 2_000_000_000
MAX_RECEIPT_TEXT_BYTES = 16 * 1024
MAX_RECEIPT_BYTES = 128 * 1024
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
MAX_PINS_JSON_BYTES = 32 * 1024
MAX_DOWNLOAD_SECONDS = 20 * 60
DOWNLOAD_SOCKET_TIMEOUT = 30
MAX_COMMAND_SECONDS = 10 * 60
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
SAFE_GGUF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$")
ASR_KEYPHRASE = "ask not what your country can do for you"
VAD_LINE = re.compile(
    r"^vad: (?P<frames>[0-9]+) frames, speech_frames=(?P<speech>[0-9]+), "
    r"mean_prob=(?P<mean>[0-9]+(?:\.[0-9]+)?)$"
)


class SmokeError(RuntimeError):
    """A fail-closed smoke precondition or assertion failed."""


@dataclass(frozen=True)
class ModelPin:
    name: str
    repo: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int

    @property
    def url(self) -> str:
        # `repo` has already been checked against the exact allowlist.
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


@dataclass
class CommandRecord:
    label: str
    argv: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Permit CDN redirects only when every hop remains HTTPS."""

    def redirect_request(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        target = urlparse(urljoin(request.full_url, newurl))
        if target.scheme.lower() != "https":
            raise SmokeError(f"download redirect left HTTPS: {target.geturl()!r}")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _bounded_text(value: bytes | str) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_RECEIPT_TEXT_BYTES:
        return text
    clipped = encoded[:MAX_RECEIPT_TEXT_BYTES].decode("utf-8", errors="ignore")
    return clipped + "\n[output truncated by release smoke]"


def _read_json(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_PINS_JSON_BYTES + 1)
    except OSError as exc:
        raise SmokeError(f"pins JSON is unreadable: {path}: {exc}") from exc
    if len(raw) > MAX_PINS_JSON_BYTES:
        raise SmokeError(f"pins JSON exceeds {MAX_PINS_JSON_BYTES} bytes")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise SmokeError(f"pins JSON is unreadable or invalid: {path}: {exc}") from exc


def parse_pins(value: Any) -> dict[str, ModelPin]:
    """Validate the owner-reviewed two-model manifest without network access."""
    if not isinstance(value, dict) or set(value) != set(EXPECTED_MODELS):
        raise SmokeError(
            "pins JSON must contain exactly the keys whisper-base and silero-vad-v5"
        )
    result: dict[str, ModelPin] = {}
    total = 0
    for name, expected_repo in EXPECTED_MODELS.items():
        raw = value[name]
        if not isinstance(raw, dict):
            raise SmokeError(f"pin {name}: record must be an object")
        required = {"repo", "revision", "filename", "sha256", "size_bytes"}
        if set(raw) != required:
            raise SmokeError(f"pin {name}: record keys must be exactly {sorted(required)}")
        repo = raw["repo"]
        revision = raw["revision"]
        filename = raw["filename"]
        digest = raw["sha256"]
        size = raw["size_bytes"]
        if repo != expected_repo:
            raise SmokeError(f"pin {name}: repo must be exactly {expected_repo!r}")
        if not isinstance(revision, str) or not HEX40.fullmatch(revision):
            raise SmokeError(f"pin {name}: revision must be exactly 40 hexadecimal characters")
        if not isinstance(filename, str) or not SAFE_GGUF.fullmatch(filename):
            raise SmokeError(f"pin {name}: filename must be a safe .gguf basename")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise SmokeError(f"pin {name}: sha256 must be exactly 64 hexadecimal characters")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SmokeError(f"pin {name}: size_bytes must be a positive integer")
        if size >= MAX_MODEL_BYTES:
            raise SmokeError(f"pin {name}: size_bytes must be below 2,000,000,000")
        total += size
        result[name] = ModelPin(name, repo, revision, filename, digest.lower(), size)
    if total >= MAX_MODEL_BYTES:
        raise SmokeError("combined model size must be below 2,000,000,000 bytes")
    if len({pin.filename for pin in result.values()}) != len(result):
        raise SmokeError("model filenames must be distinct")
    return result


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise SmokeError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest(), size


def _checked_out_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeError(f"cannot bind smoke to checked-out HEAD: {exc}") from exc
    head = result.stdout.strip()
    if not HEX40.fullmatch(head):
        raise SmokeError("git rev-parse HEAD did not return one 40-hex commit")
    return head.lower()


def require_hosted_linux(expected_head: str, repo_root: Path) -> str:
    if sys.platform != "linux":
        raise SmokeError("real release smoke is Linux-hosted only")
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise SmokeError("real release smoke requires GITHUB_ACTIONS=true")
    if os.environ.get("VOKRA_RELEASE_CPU_SMOKE") != "1":
        raise SmokeError("real release smoke requires VOKRA_RELEASE_CPU_SMOKE=1")
    if not HEX40.fullmatch(expected_head):
        raise SmokeError("expected HEAD must be exactly 40 hexadecimal characters")
    actual_head = _checked_out_head(repo_root)
    if actual_head != expected_head.lower():
        raise SmokeError(f"checked-out HEAD {actual_head} != expected {expected_head.lower()}")
    return actual_head


def _download(pin: ModelPin, destination: Path) -> tuple[str, int]:
    parsed = urlparse(pin.url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise SmokeError(f"download {pin.name}: constructed URL is not allowlisted HTTPS")
    request = urllib.request.Request(pin.url, headers={"Accept": "application/octet-stream"})
    deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
    try:
        opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler())
        response = opener.open(request, timeout=DOWNLOAD_SOCKET_TIMEOUT)
    except SmokeError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise SmokeError(f"download failed for {pin.name}: {exc}") from exc
    with response:
        final_url = response.geturl()
        if urlparse(final_url).scheme.lower() != "https":
            raise SmokeError(f"download {pin.name}: final URL is not HTTPS")
        header = response.headers.get("Content-Length")
        if header is not None:
            try:
                if int(header) != pin.size_bytes:
                    raise SmokeError(
                        f"download {pin.name}: Content-Length {header} != reviewed size {pin.size_bytes}"
                    )
            except ValueError as exc:
                raise SmokeError(f"download {pin.name}: invalid Content-Length") from exc
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("wb") as handle:
                while True:
                    if time.monotonic() >= deadline:
                        raise SmokeError(f"download {pin.name}: total deadline exceeded")
                    # urllib's blocking read can consume one socket timeout
                    # beyond this check; the deadline is enforced between reads.
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > pin.size_bytes or size >= MAX_MODEL_BYTES:
                        raise SmokeError(f"download {pin.name}: exceeded reviewed size bound")
                    digest.update(chunk)
                    handle.write(chunk)
        except OSError as exc:
            raise SmokeError(f"download {pin.name}: read/write failed: {exc}") from exc
    actual = digest.hexdigest()
    if size != pin.size_bytes:
        raise SmokeError(f"download {pin.name}: {size} bytes != reviewed size {pin.size_bytes}")
    if actual != pin.sha256:
        raise SmokeError(f"download {pin.name}: SHA-256 {actual} != reviewed {pin.sha256}")
    return actual, size


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> str | None:
    cleanup_errors: list[str] = []
    group_missing = False
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        group_missing = os.name == "posix"
    except OSError as exc:
        cleanup_errors.append(f"process-group termination failed: {exc}")
        if os.name == "posix" and process.poll() is None:
            try:
                process.terminate()
            except OSError as terminate_error:
                cleanup_errors.append(f"leader termination failed: {terminate_error}")
    leader_wait_timed_out = False
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        leader_wait_timed_out = True
    except OSError as exc:
        cleanup_errors.append(f"owned process wait failed: {exc}")
    if os.name == "posix":
        if not group_missing:
            try:
                # The leader may already be reaped while a descendant still
                # holds a pipe open; always make one bounded group-kill pass.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                cleanup_errors.append(f"process-group kill failed: {exc}")
    elif leader_wait_timed_out:
        try:
            process.kill()
            process.wait(timeout=1)
        except OSError as exc:
            cleanup_errors.append(f"owned process kill failed: {exc}")
        except subprocess.TimeoutExpired:
            cleanup_errors.append("owned process did not exit after kill")
    if leader_wait_timed_out and os.name == "posix":
        try:
            process.wait(timeout=1)
        except OSError as exc:
            cleanup_errors.append(f"owned process wait after group kill failed: {exc}")
        except subprocess.TimeoutExpired:
            cleanup_errors.append("owned process did not exit after group kill")
    return "; ".join(cleanup_errors) if cleanup_errors else None


def _run_cli_streaming(
    argv: list[str],
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    selector_factory: Callable[[], Any] = selectors.DefaultSelector,
    clock: Callable[[], float] = time.monotonic,
    read_fn: Callable[[int, int], bytes] = os.read,
    set_blocking: Callable[[int, bool], None] = os.set_blocking,
    terminate: Callable[[Any], str | None] = _terminate_owned_process,
) -> tuple[int | None, bytes, bytes, str | None]:
    """Run one owned CLI process with bounded, deadline-driven pipe reads."""
    if os.name != "posix":
        raise SmokeError("bounded release smoke capture requires a POSIX hosted runner")
    try:
        process = popen_factory(
            argv,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise SmokeError(f"CLI invocation failed: {exc}") from exc
    if process.stdout is None or process.stderr is None:
        cleanup_error = _terminate_owned_process(process)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        detail = f"CLI pipes were not created{f'; {cleanup_error}' if cleanup_error else ''}"
        raise SmokeError(detail)
    streams = (process.stdout, process.stderr)
    selector = None
    buffers = [bytearray(), bytearray()]
    deadline = clock() + MAX_COMMAND_SECONDS
    failure: str | None = None
    total = 0
    try:
        selector = selector_factory()
        for index, stream in enumerate(streams):
            set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, index)
        while selector.get_map():
            remaining = deadline - clock()
            if remaining <= 0:
                failure = "CLI wall deadline exceeded"
                break
            for key, _ in selector.select(timeout=min(remaining, 0.25)):
                try:
                    chunk = read_fn(key.fileobj.fileno(), 8192)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > MAX_COMMAND_OUTPUT_BYTES:
                    failure = "CLI output exceeded the bounded output limit"
                    break
                buffers[key.data].extend(chunk)
            if failure:
                break
        if not failure:
            try:
                process.wait(timeout=max(0.1, deadline - clock()))
            except subprocess.TimeoutExpired:
                failure = "CLI wall deadline exceeded"
    except Exception as exc:
        failure = failure or f"CLI capture failed: {exc}"
    finally:
        if selector is not None:
            try:
                selector.close()
            except Exception as exc:
                failure = failure or f"CLI selector cleanup failed: {exc}"
        for stream in streams:
            try:
                stream.close()
            except Exception as exc:
                failure = failure or f"CLI pipe cleanup failed: {exc}"
        if failure:
            cleanup_error = terminate(process)
            if cleanup_error:
                failure = f"{failure}; {cleanup_error}"
    return process.returncode, bytes(buffers[0]), bytes(buffers[1]), failure


def _run_cli(
    label: str,
    argv: list[str],
    records: list[CommandRecord],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> str:
    record = CommandRecord(label=label, argv=list(argv))
    records.append(record)
    if runner is None:
        returncode, stdout, stderr, failure = _run_cli_streaming(argv)
        record.returncode = returncode
        stdout_text = stdout.decode("utf-8", errors="replace")
        record.stdout = _bounded_text(stdout_text)
        record.stderr = _bounded_text(stderr)
        if failure:
            record.error = failure
            raise SmokeError(f"{label}: {failure}")
    else:
        try:
            completed = runner(
                argv,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=MAX_COMMAND_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            record.error = str(exc)
            raise SmokeError(f"{label}: CLI invocation failed: {exc}") from exc
        record.returncode = completed.returncode
        stdout_text = completed.stdout
        record.stdout = _bounded_text(stdout_text)
        record.stderr = _bounded_text(completed.stderr)
        if sum(
            len(stream.encode("utf-8", errors="replace"))
            for stream in (completed.stdout, completed.stderr)
        ) > MAX_COMMAND_OUTPUT_BYTES:
            record.error = "CLI output exceeded the bounded output limit"
            raise SmokeError(f"{label}: CLI output exceeded the bounded output limit")
    if record.returncode != 0:
        record.error = f"CLI returned {record.returncode}"
        raise SmokeError(f"{label}: CLI returned {record.returncode}")
    return stdout_text


def _normalize_transcript(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    asr_lines = [line[4:].strip() for line in lines if line.startswith("asr:")]
    if len(asr_lines) != 1 or not asr_lines[0]:
        raise SmokeError("ASR stdout must contain exactly one non-empty `asr:` line")
    normalized = re.sub(r"[^a-z0-9]+", " ", asr_lines[0].lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _parse_vad(stdout: str, label: str) -> dict[str, int | float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SmokeError(f"{label}: VAD stdout must contain exactly one summary line")
    match = VAD_LINE.fullmatch(lines[0])
    if not match:
        raise SmokeError(f"{label}: unexpected VAD stdout contract")
    frames = int(match["frames"])
    speech = int(match["speech"])
    mean = float(match["mean"])
    if frames <= 0 or speech < 0 or speech > frames or not math.isfinite(mean) or not 0 <= mean <= 1:
        raise SmokeError(f"{label}: invalid VAD frame/probability bounds")
    return {"frames": frames, "speech_frames": speech, "mean_prob": mean}


def _wav_geometry(path: Path) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            return {
                "channels": reader.getnchannels(),
                "sample_rate": reader.getframerate(),
                "sample_width_bytes": reader.getsampwidth(),
                "frames": reader.getnframes(),
            }
    except (OSError, wave.Error) as exc:
        raise SmokeError(f"cannot read WAV geometry: {path}: {exc}") from exc


def _write_silence_wav(source: Path, destination: Path) -> dict[str, Any]:
    geometry = _wav_geometry(source)
    channels = geometry["channels"]
    rate = geometry["sample_rate"]
    width = geometry["sample_width_bytes"]
    frames = geometry["frames"]
    if (channels, rate, width) != (1, 16_000, 2) or frames <= 0:
        raise SmokeError("fixture WAV must be non-empty 16 kHz mono PCM16")
    try:
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            remaining = frames
            block = b"\0" * (16_000 * 2)
            while remaining:
                count = min(remaining, 16_000)
                writer.writeframes(block[: count * 2])
                remaining -= count
    except (OSError, wave.Error) as exc:
        raise SmokeError(f"cannot create temporary silence WAV: {exc}") from exc
    digest, size = _sha256_file(destination)
    return {**geometry, "sha256": digest, "bytes": size}


def _receipt_payload(
    *,
    status: str,
    head: str | None,
    expected_head: str,
    audio_path: Path,
    audio_digest: str | None,
    audio_bytes: int | None,
    audio_geometry: dict[str, int] | None,
    silence: dict[str, Any] | None,
    pins: dict[str, ModelPin] | None,
    models: list[dict[str, Any]],
    commands: list[CommandRecord],
    checks: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "status": status,
        "functional_smoke_only": True,
        "not_numerical_parity": True,
        "not_metal_or_performance": True,
        "runtime": {
            "platform": sys.platform,
            "github_actions": os.environ.get("GITHUB_ACTIONS"),
            "marker": os.environ.get("VOKRA_RELEASE_CPU_SMOKE"),
            "head": head,
            "expected_head": expected_head,
        },
        "audio": {
            "path": str(audio_path),
            "sha256": audio_digest,
            "bytes": audio_bytes,
            "geometry": audio_geometry,
        },
        "generated_silence": silence,
        "reviewed_pins": {
            name: {
                "repo": pin.repo,
                "revision": pin.revision,
                "filename": pin.filename,
                "sha256": pin.sha256,
                "size_bytes": pin.size_bytes,
                "url": pin.url,
            }
            for name, pin in (pins or {}).items()
        },
        "models": models,
        "commands": [
            {
                "label": command.label,
                "argv": command.argv,
                "returncode": command.returncode,
                "stdout": command.stdout,
                "stderr": command.stderr,
                **({"error": command.error} if command.error else {}),
            }
            for command in commands
        ],
        "semantic_checks": checks,
        **({"error": error} if error else {}),
    }


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    if len(encoded) + 1 > MAX_RECEIPT_BYTES:
        raise SmokeError(f"receipt exceeds {MAX_RECEIPT_BYTES} bytes")
    path.write_bytes(encoded + b"\n")


def run_smoke(
    *,
    pins: dict[str, ModelPin],
    cli: Path,
    audio: Path,
    receipt: Path,
    expected_head: str,
    repo_root: Path,
    downloader: Callable[[ModelPin, Path], tuple[str, int]] = _download,
    cli_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    enforce_canonical_audio: bool = True,
) -> None:
    records: list[CommandRecord] = []
    model_receipts: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    head: str | None = None
    audio_digest: str | None = None
    audio_bytes: int | None = None
    audio_geometry: dict[str, int] | None = None
    silence_info: dict[str, Any] | None = None
    status = "FAIL"
    error: str | None = None
    try:
        head = require_hosted_linux(expected_head, repo_root)
        if not cli.is_file() or not os.access(cli, os.X_OK):
            raise SmokeError(f"CLI is not an executable file: {cli}")
        if enforce_canonical_audio and audio != repo_root / EXPECTED_AUDIO:
            raise SmokeError(f"audio must be the committed fixture {EXPECTED_AUDIO}")
        sidecar = audio.with_name(audio.name + ".sha256")
        if not sidecar.is_file():
            raise SmokeError(f"missing audio SHA sidecar: {sidecar}")
        try:
            sidecar_digest = sidecar.read_text(encoding="utf-8").split()[0].lower()
        except (OSError, IndexError) as exc:
            raise SmokeError("audio SHA sidecar is unreadable or empty") from exc
        if not HEX64.fullmatch(sidecar_digest):
            raise SmokeError("audio SHA sidecar is malformed")
        audio_digest, audio_bytes = _sha256_file(audio)
        if audio_digest != sidecar_digest:
            raise SmokeError("audio SHA sidecar does not match the committed fixture")
        audio_geometry = _wav_geometry(audio)
        with tempfile.TemporaryDirectory(prefix="vokra-release-cpu-smoke-") as temp_dir:
            temp = Path(temp_dir)
            silence = temp / "silence.wav"
            silence_info = _write_silence_wav(audio, silence)
            model_paths: dict[str, Path] = {}
            for name in EXPECTED_MODELS:
                pin = pins[name]
                path = temp / pin.filename
                actual_digest, actual_bytes = downloader(pin, path)
                model_paths[name] = path
                model_receipts.append(
                    {
                        "name": name,
                        "repo": pin.repo,
                        "revision": pin.revision,
                        "filename": pin.filename,
                        "url": pin.url,
                        "expected_sha256": pin.sha256,
                        "actual_sha256": actual_digest,
                        "expected_bytes": pin.size_bytes,
                        "actual_bytes": actual_bytes,
                    }
                )
            whisper = model_paths["whisper-base"]
            asr_argv = [str(cli), "run", "--model", str(whisper), "--input", str(audio), "--backend", "cpu"]
            transcript = _normalize_transcript(
                _run_cli("asr-jfk", asr_argv, records, runner=cli_runner)
            )
            if ASR_KEYPHRASE not in transcript:
                raise SmokeError(f"ASR transcript missed the JFK keyphrase: {transcript!r}")
            checks["asr_jfk_keyphrase"] = {"passed": True, "normalized_text": transcript}

            vad_model = model_paths["silero-vad-v5"]
            vad_audio_argv = [
                str(cli),
                "run",
                "--model",
                str(vad_model),
                "--input",
                str(audio),
                "--backend",
                "cpu",
            ]
            vad_audio = _parse_vad(
                _run_cli("vad-jfk", vad_audio_argv, records, runner=cli_runner), "vad-jfk"
            )
            if vad_audio["speech_frames"] <= 0 or vad_audio["speech_frames"] >= vad_audio["frames"]:
                raise SmokeError("VAD JFK run did not observe both speech and non-speech frames")
            vad_silence_argv = [
                str(cli),
                "run",
                "--model",
                str(vad_model),
                "--input",
                str(silence),
                "--backend",
                "cpu",
            ]
            vad_silence = _parse_vad(
                _run_cli("vad-silence", vad_silence_argv, records, runner=cli_runner),
                "vad-silence",
            )
            if vad_silence["speech_frames"] != 0:
                raise SmokeError("VAD silence run reported speech frames")
            if vad_silence["frames"] != vad_audio["frames"]:
                raise SmokeError("VAD silence/JFK frame counts differ for same-duration WAVs")
            checks["vad"] = {
                "passed": True,
                "jfk": vad_audio,
                "silence": vad_silence,
            }
        status = "PASS"
    except SmokeError as exc:
        error = str(exc)
    payload = _receipt_payload(
        status=status,
        head=head,
        expected_head=expected_head,
        audio_path=audio,
        audio_digest=audio_digest,
        audio_bytes=audio_bytes,
        audio_geometry=audio_geometry,
        silence=silence_info,
        pins=pins,
        models=model_receipts,
        commands=records,
        checks=checks,
        error=error,
    )
    write_receipt(receipt, payload)
    if error:
        raise SmokeError(error)
    print("PASS release-candidate-cpu-smoke")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins-json", type=Path, required=True, help="reviewed two-model pin manifest")
    parser.add_argument("--validate-pins-only", action="store_true")
    parser.add_argument("--cli", type=Path, help="built first-party vokra-cli executable")
    parser.add_argument("--audio", type=Path, default=EXPECTED_AUDIO)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-head")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path.cwd().resolve()
    try:
        pins = parse_pins(_read_json(args.pins_json))
        if args.validate_pins_only:
            print("reviewed pins valid")
            return 0
        if args.cli is None or args.receipt is None or args.expected_head is None:
            raise SmokeError("real smoke requires --cli, --receipt, and --expected-head")
        run_smoke(
            pins=pins,
            cli=args.cli.resolve(),
            audio=args.audio.resolve(),
            receipt=args.receipt.resolve(),
            expected_head=args.expected_head,
            repo_root=repo_root,
        )
    except SmokeError as exc:
        print(f"FAIL release-candidate-cpu-smoke: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
