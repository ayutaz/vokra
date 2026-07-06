# SPDX-License-Identifier: Apache-2.0
"""T13 — binding-vs-CLI byte-identical parity gate.

The Python binding is required to reproduce the ``vokra-cli run`` output
**byte-for-byte** for the same GGUF + input pair. That is the whole point
of the pure-``ctypes`` (D2) design: the Python wrapper is a marshalling
seam over ``include/vokra.h``, so the runtime code path it exercises is
identical to what the CLI drives, and any user-visible difference must
be either a wrapper bug or an ABI drift.

This test hashes the stdout of both paths and asserts the SHA-256 digests
match. We do NOT compare stdout as strings: (1) SHA-256 avoids embedding
model-specific transcripts in the test source (which would drift the
moment we point at a new GGUF), and (2) any stray difference — even a
trailing newline, a locale-shifted number, a decoder non-determinism —
turns into a distinct digest, so the failure signal is unambiguous.

Gating (per spec)
-----------------
The GGUF fixture required to run this test is not distributed with the
repo (BR-model-zoo Apache-2.0/MIT only, and even those are large). The
test is therefore ``PARITY_GATE_MODEL``-env-gated:

* unset (default, incl. CI without a mounted fixture): ``pytest.skip``,
  ``exit 0``. This is the CI contract — the file must import and the
  test must be collectable without a fixture, so a fresh contributor
  clone stays green.
* set to a path: run the full parity comparison. Missing file / wrong
  ABI / non-matching digests all raise (never silently pass).

Companion env vars keep the test task-agnostic:

* ``PARITY_GATE_INPUT``: WAV for VAD/ASR models. Required when the GGUF
  is an ASR or VAD model; ignored for TTS.
* ``PARITY_GATE_TEXT``: text for TTS models. Required when the GGUF is
  a TTS model; ignored for VAD/ASR.

Exactly one of ``PARITY_GATE_INPUT`` / ``PARITY_GATE_TEXT`` must be set
when ``PARITY_GATE_MODEL`` is set — the test refuses to guess.

Design notes
------------
* We invoke ``vokra-cli`` as a subprocess (``python -m`` is not used;
  ``vokra-cli`` is a Rust binary). The binary is located via
  ``VOKRA_CLI_BIN`` env, else ``cargo run -p vokra-cli --release --``
  fallback, else the CLI-side path is skipped (with a diagnostic in the
  skip message) rather than silently green.
* The binding-side "stdout" is synthesized from the Python API call —
  we format the result in the **exact same wire format** the CLI's
  ``println!`` emits (see ``crates/vokra-cli/src/run.rs``). This keeps
  the parity guarantee anchored to human-readable output, not to
  internal representations. If the CLI's output format ever changes,
  this test breaks loudly and the wrapper's format string is updated
  in the same PR — that is the desired coupling.
* stdout hashing uses ``hashlib.sha256(bytes)``. We never touch
  ``str.encode`` with a locale-shifted codec (NFR-RL-01): everything is
  UTF-8 explicit.
* Determinism caveat: ASR beam search and TTS flow sampling may include
  non-deterministic elements. The CLI seeds them the same way this
  binding does (via the shared runtime code path), so byte-identity
  holds. If a future model introduces a stochastic sampler with no
  fixed seed, this test will flake — the correct response is to fix the
  seed in the runtime, not to loosen the digest comparison here.

Zero-dep: only stdlib + pytest.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# path shim (matches test_smoke.py / test_session.py)
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# env-gated fixture resolution
# ---------------------------------------------------------------------------

_ENV_MODEL = "PARITY_GATE_MODEL"
_ENV_INPUT = "PARITY_GATE_INPUT"
_ENV_TEXT = "PARITY_GATE_TEXT"
_ENV_CLI_BIN = "VOKRA_CLI_BIN"


def _resolve_gate() -> tuple[Path, Optional[Path], Optional[str]]:
    """Return ``(model, input_wav, text)`` or trigger ``pytest.skip``.

    The gate is a triangle:

    * ``PARITY_GATE_MODEL`` unset → skip (CI default, contributor default).
    * set but file missing → hard fail: this is a misconfiguration, not
      a legitimate "no fixture" state, and silently skipping would hide
      test regressions in the CD pipeline that mounts a real fixture.
    * neither ``PARITY_GATE_INPUT`` nor ``PARITY_GATE_TEXT`` set → hard
      fail: we refuse to guess the model's task from the file name.
    * both set → hard fail: a single invocation cannot be both ASR and
      TTS, and picking one silently would hide a config mistake.
    """
    raw = os.environ.get(_ENV_MODEL, "")
    if not raw:
        pytest.skip(
            f"{_ENV_MODEL} unset — parity test requires a GGUF fixture "
            "(not distributed with the repo)."
        )
    model_path = Path(raw).expanduser()
    if not model_path.exists():
        # Hard error, not skip: the operator asked for the gate to run.
        raise FileNotFoundError(
            f"{_ENV_MODEL}={raw!r} does not exist. Set to a real GGUF "
            "path or unset to skip the parity test."
        )

    inp = os.environ.get(_ENV_INPUT, "")
    txt = os.environ.get(_ENV_TEXT, "")
    if inp and txt:
        raise RuntimeError(
            f"Both {_ENV_INPUT} and {_ENV_TEXT} are set. Set exactly one "
            "so the parity test knows whether to run ASR/VAD (--input) or "
            "TTS (--text)."
        )
    if not inp and not txt:
        raise RuntimeError(
            f"{_ENV_MODEL} is set but neither {_ENV_INPUT} nor "
            f"{_ENV_TEXT} is set. Set one of them so the parity test "
            "knows which task to drive."
        )

    input_path: Optional[Path] = None
    if inp:
        input_path = Path(inp).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(
                f"{_ENV_INPUT}={inp!r} does not exist."
            )

    return model_path, input_path, (txt or None)


def _resolve_cli_bin() -> list[str]:
    """Locate ``vokra-cli`` for the CLI-side of the comparison.

    Priority:
    1. ``VOKRA_CLI_BIN`` env — an absolute path to the built binary. This
       is the CI-friendly path: the build job puts the artifact somewhere
       stable and exports the env.
    2. ``vokra-cli`` on ``PATH`` (``shutil.which``) — useful for local
       development after ``cargo install --path crates/vokra-cli``.
    3. ``cargo`` on ``PATH`` — falls back to
       ``cargo run -p vokra-cli --release --``. This is slow (cold cargo
       resolve on every subprocess) but correct.

    If none of the three are available, we skip with a diagnostic. We
    do NOT silently pass: the test would then always appear green even
    though the CLI leg never ran.
    """
    env_bin = os.environ.get(_ENV_CLI_BIN, "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"{_ENV_CLI_BIN}={env_bin!r} does not exist."
            )
        return [str(p)]

    on_path = shutil.which("vokra-cli")
    if on_path:
        return [on_path]

    cargo = shutil.which("cargo")
    if cargo:
        # Locate the workspace root so ``cargo run`` finds the manifest
        # regardless of cwd. Structure: bindings/python/tests/ →
        # ../../.. is the repo root that owns the workspace Cargo.toml.
        workspace_root = Path(__file__).resolve().parents[3]
        return [
            cargo,
            "run",
            "--quiet",
            "--manifest-path",
            str(workspace_root / "Cargo.toml"),
            "-p",
            "vokra-cli",
            "--release",
            "--",
        ]

    pytest.skip(
        f"vokra-cli not found: set {_ENV_CLI_BIN}, install vokra-cli, "
        "or ensure `cargo` is on PATH."
    )
    # Unreachable — pytest.skip raises. Kept for typechecker's sake.
    raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# CLI + binding invocations, hashed
# ---------------------------------------------------------------------------


def _read_wav_mono_f32(path: Path) -> tuple[list[float], int]:
    """Read a mono PCM16 WAV into ``([-1, 1] f32, sample_rate)``.

    Uses stdlib :mod:`wave` (no numpy) so the test stays zero-dep. Only
    PCM16 mono is supported — that's what the vokra fixtures ship as,
    and any other shape means someone pointed the gate at the wrong file.
    """
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        if n_channels != 1:
            raise ValueError(
                f"parity gate WAV must be mono; got {n_channels} channels "
                f"in {path}"
            )
        if sampwidth != 2:
            raise ValueError(
                f"parity gate WAV must be PCM16; got sampwidth={sampwidth} "
                f"in {path}"
            )
        raw = w.readframes(n_frames)

    # Little-endian signed 16-bit → float in [-1, 1]. Manual loop keeps
    # us numpy-free; n_frames is bounded by the fixture (typically <1M).
    import struct

    ints = struct.unpack(f"<{n_frames}h", raw)
    scale = 1.0 / 32768.0
    return [i * scale for i in ints], sr


def _run_cli(argv_prefix: list[str], args: list[str]) -> bytes:
    """Invoke ``vokra-cli run`` and return raw stdout bytes.

    We capture bytes, not text, so the SHA-256 digest reflects the wire
    output verbatim — no ``.decode()`` normalisation, no newline
    translation. ``check=True`` turns a non-zero exit into a
    ``CalledProcessError`` with the stderr attached, which pytest prints
    on failure.
    """
    proc = subprocess.run(
        argv_prefix + args,
        check=True,
        capture_output=True,
        # Do NOT pass ``text=True`` — bytes only. The CLI writes UTF-8 on
        # all supported platforms; keeping bytes preserves any BOM /
        # trailing-newline nuance that would otherwise be silently
        # stripped by universal newlines translation on Windows.
    )
    return proc.stdout


def _format_asr_line(text: str) -> bytes:
    """Reproduce ``println!("asr: {text}")`` from ``run.rs::main``."""
    return f"asr: {text}\n".encode("utf-8")


def _format_tts_line(n_samples: int, sample_rate: int) -> bytes:
    """Reproduce the ``no-output`` TTS branch of ``run.rs::main``.

    The CLI prints
    ``tts: {n} samples, {secs:.3}s @ {sr} Hz (no --output; audio discarded)``
    when ``--output`` is omitted. We use the same format string here so
    the digest matches without needing a temp WAV write on either side.
    """
    secs = n_samples / float(sample_rate) if sample_rate else 0.0
    return (
        f"tts: {n_samples} samples, {secs:.3f}s @ {sample_rate} Hz "
        "(no --output; audio discarded)\n"
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------


def test_binding_matches_cli_sha256() -> None:
    """The Python binding and ``vokra-cli`` produce byte-identical stdout.

    Both legs run against the same GGUF + input, and their stdout is
    reduced to a SHA-256 digest. Byte-identical digests are the only
    accepted outcome. Even a single differing byte (newline, whitespace,
    number formatting, transcript token) causes the digests to diverge,
    which is the whole point.
    """
    model, input_wav, text = _resolve_gate()
    cli_prefix = _resolve_cli_bin()

    # --- CLI leg -------------------------------------------------------
    if text is not None:
        cli_args = ["run", "--model", str(model), "--text", text]
    else:
        assert input_wav is not None  # invariant from _resolve_gate
        cli_args = ["run", "--model", str(model), "--input", str(input_wav)]
    cli_stdout = _run_cli(cli_prefix, cli_args)
    cli_digest = hashlib.sha256(cli_stdout).hexdigest()

    # --- Binding leg ---------------------------------------------------
    # Import here (not at module top) so a failing ctypes load only skips
    # this test rather than breaking collection of the whole file.
    from vokra._bindings import bind
    from vokra._native import load_library
    from vokra.errors import VokraError
    from vokra.session import Session

    try:
        lib = bind(load_library())
    except (VokraError, OSError) as exc:
        pytest.skip(f"native libvokra not loadable: {exc}")

    binding_bytes: bytes
    with Session.open(str(model), lib=lib) as sess:
        if text is not None:
            pcm, sr = sess.synthesize(text)
            binding_bytes = _format_tts_line(len(pcm), sr)
        else:
            assert input_wav is not None
            pcm, sr = _read_wav_mono_f32(input_wav)
            transcript = sess.transcribe(pcm, sr)
            binding_bytes = _format_asr_line(transcript)

    binding_digest = hashlib.sha256(binding_bytes).hexdigest()

    # --- assert byte-identity -----------------------------------------
    # We compare the digests (short, comparable in a test report) but
    # attach the raw bytes to the failure message so a diverging pair
    # is diagnosable without re-running the test. Truncate to keep the
    # pytest output readable for pathological cases.
    def _preview(b: bytes) -> str:
        s = b.decode("utf-8", "replace")
        return s if len(s) <= 200 else s[:200] + f"... (+{len(s) - 200} bytes)"

    assert binding_digest == cli_digest, (
        "binding stdout and vokra-cli stdout diverged for the same GGUF "
        "and input.\n"
        f"  binding sha256 = {binding_digest}\n"
        f"  cli     sha256 = {cli_digest}\n"
        f"  binding output = {_preview(binding_bytes)!r}\n"
        f"  cli     output = {_preview(cli_stdout)!r}\n"
    )


# ---------------------------------------------------------------------------
# gate-collection sanity check — runs even without a fixture
# ---------------------------------------------------------------------------


def test_gate_skips_cleanly_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI without a fixture must skip, not fail.

    This test verifies the CI contract explicitly: when
    ``PARITY_GATE_MODEL`` is unset, ``_resolve_gate`` calls
    ``pytest.skip``, which raises :class:`pytest.skip.Exception`. We
    catch it via ``pytest.raises`` so the enclosing test passes rather
    than being reported as skipped — the whole point is to prove the
    skip logic fires on the unset path, which is the CI default.
    """
    # Neutralise every gate env var so this check is hermetic even when
    # the operator ran ``PARITY_GATE_MODEL=... pytest`` on their box.
    monkeypatch.delenv(_ENV_MODEL, raising=False)
    monkeypatch.delenv(_ENV_INPUT, raising=False)
    monkeypatch.delenv(_ENV_TEXT, raising=False)

    with pytest.raises(pytest.skip.Exception):
        _resolve_gate()
