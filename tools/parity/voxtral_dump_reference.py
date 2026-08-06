#!/usr/bin/env python3
"""Voxtral reference-dump CLI shim (FQ-07, 2026-07-31).

This is the CLI entry point that `.github/workflows/parity-voxtral-real.yml`
invokes. It follows the same naming convention as the newer parity dumpers
(`deberta_v3_dump_reference.py`, `dfn3_dump_reference.py`) and delegates the
heavy fp32 tower + streamed text decoder dump to the pre-existing
`tools/parity/dump_voxtral_reference.py` (which the M3-10-T19/T20 workflow
authored and the M3 parity_voxtral harness already binds).

The shim exists for three reasons:

1. **CLI shape parity** with the other new-generation parity dumpers so the
   workflow YAML reads uniformly (`voxtral_dump_reference.py --do-dump
   --checkpoint-dir X --output-dir Y`). The pre-existing dumper uses a
   different argparse contract (positional `out_dir`, no `--do-dump` toggle);
   translating here keeps the pre-existing dumper unchanged (FQ-07's
   constraint: never modify files not in the proposed_fix_files list).

2. **`--self-test` mode** (offline, network-free) that the
   `python-parity-oracles` job in `.github/workflows/ci-quality.yml` and the
   `tools/parity/test_parity_voxtral_workflow.py` oracle exercise. This
   verifies (a) argparse contract, (b) the underlying dumper file is
   reachable, and (c) the delegation path resolves — WITHOUT touching HF or
   torch. Zero-dep (NFR-DS-02): stdlib-only in `--self-test` mode.

3. **Fabricated-pass 禁止** boundary: the shim NEVER silently no-ops the
   real dump. `--do-dump` calls the delegated dumper via subprocess and
   propagates its exit code. Any missing checkpoint field, tokenizer, etc.
   surfaces as the exact stderr message the delegated dumper emits.

# Delegated dumper (`dump_voxtral_reference.py`)

Reproduces the upstream `transformers.models.voxtral` reference tensors for
`mistralai/Voxtral-Mini-3B-2507` (Apache-2.0). See its module docstring for
the RSS-discipline two-stage design (~6.8 GiB peak on M1 iMac) and the
mandatory self-check that proves streamed decode is bitwise-identical to
`LlamaModel`. This shim does NOT re-implement any of that; it just plumbs
the CLI.

# Usage

    # Real dump (delegates to dump_voxtral_reference.py):
    python3 tools/parity/voxtral_dump_reference.py --do-dump \\
        --checkpoint-dir /path/to/hf-snapshot/mistralai-voxtral-mini-3b \\
        --output-dir tests/parity/voxtral

    # Offline self-test (no network, no torch):
    python3 tools/parity/voxtral_dump_reference.py --self-test

Exit codes: 0 = success, 1 = usage / delegated dumper failure, 2 = argparse
error, 3 = self-test failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DELEGATED_DUMPER = Path(__file__).resolve().parent / "dump_voxtral_reference.py"


def _self_test() -> int:
    """Verify the shim contract without any network or torch calls.

    Checks:
      1. Delegated dumper (`dump_voxtral_reference.py`) is on disk.
      2. The delegated dumper carries an `argparse.ArgumentParser` +
         `--checkpoint-dir` argument (the surface this shim delegates to).
      3. Our own argparse rejects unknown flags (fabricated-pass guard: a
         typo like `--do-domp` must exit non-zero rather than skipping).
      4. `--do-dump` without `--checkpoint-dir` exits non-zero (missing
         required input must be a hard error, not a silent skip).

    Zero-dep — no torch / no transformers / no huggingface_hub imports.
    """
    fail = 0

    # (1) delegated dumper on disk.
    if not DELEGATED_DUMPER.is_file():
        print(f"self-test FAIL (1): missing delegated dumper {DELEGATED_DUMPER}",
              file=sys.stderr)
        fail = 1
    else:
        print(f"self-test OK (1): delegated dumper present at {DELEGATED_DUMPER}")

    # (2) delegated dumper accepts --checkpoint-dir. Substring check keeps
    # the self-test stdlib-only; running the delegated dumper's `--help`
    # would require its full torch+transformers venv.
    if DELEGATED_DUMPER.is_file():
        src = DELEGATED_DUMPER.read_text(encoding="utf-8")
        if "argparse" not in src or "--checkpoint-dir" not in src:
            print("self-test FAIL (2): delegated dumper missing argparse or "
                  "--checkpoint-dir surface", file=sys.stderr)
            fail = 1
        else:
            print("self-test OK (2): delegated dumper exposes --checkpoint-dir")

    # (3) unknown flag must be rejected by our own argparse. We recurse
    # into ourselves via subprocess to keep the check hermetic.
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--not-a-real-flag"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        print("self-test FAIL (3): unknown flag was accepted "
              "(fabricated-pass guard failed)", file=sys.stderr)
        fail = 1
    else:
        print(f"self-test OK (3): unknown flag rejected (rc={proc.returncode})")

    # (4) --do-dump without --checkpoint-dir must be a hard error.
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--do-dump"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        print("self-test FAIL (4): --do-dump with no --checkpoint-dir was "
              "accepted (missing required input must hard-fail)",
              file=sys.stderr)
        fail = 1
    else:
        print(f"self-test OK (4): --do-dump without --checkpoint-dir rejected "
              f"(rc={proc.returncode})")

    if fail:
        return 3
    print("self-test OK: all checks passed")
    return 0


def _do_dump(checkpoint_dir: Path, output_dir: Path,
             audio: Path | None) -> int:
    """Delegate to `dump_voxtral_reference.py` with the translated CLI."""
    if not DELEGATED_DUMPER.is_file():
        print(f"error: delegated dumper missing at {DELEGATED_DUMPER} — this "
              f"shim cannot run without it", file=sys.stderr)
        return 1
    if not checkpoint_dir.is_dir():
        print(f"error: --checkpoint-dir {checkpoint_dir} does not exist or is "
              f"not a directory (FR-EX-08: missing input is a hard error, "
              f"never a silent skip)", file=sys.stderr)
        return 1

    # Translate our CLI to the delegated dumper's shape:
    #   dump_voxtral_reference.py --checkpoint-dir X [--audio A] [out_dir]
    cmd = [sys.executable, str(DELEGATED_DUMPER),
           "--checkpoint-dir", str(checkpoint_dir)]
    if audio is not None:
        cmd.extend(["--audio", str(audio)])
    cmd.append(str(output_dir))

    print(f"delegating to: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Voxtral reference-dump CLI shim. Delegates the real fp32 "
            "tower + streamed text decoder dump to "
            "`dump_voxtral_reference.py`; provides a network-free "
            "`--self-test` mode for CI oracle coverage."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run the offline self-test (no network, no torch imports). "
            "Verifies the shim contract + delegated dumper presence. "
            "Exit 0 = pass, 3 = self-test failure."
        ),
    )
    parser.add_argument(
        "--do-dump",
        action="store_true",
        help=(
            "Actually run the reference dump (requires --checkpoint-dir + "
            "--output-dir + the delegated dumper's full torch + transformers "
            "+ mistral_common venv). Delegates to dump_voxtral_reference.py."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "LOCAL HF snapshot of mistralai/Voxtral-Mini-3B-2507 "
            "(config.json + model-*.safetensors + tekken.json + "
            "preprocessor_config.json + generation_config.json). Required "
            "when --do-dump is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write the reference dump. Defaults to "
            "tests/parity/voxtral/ (the committed fixture path)."
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help=(
            "Optional 16 kHz mono WAV override (PCM16 or IEEE_FLOAT32). "
            "Defaults to tests/fixtures/audio/jfk-30s.wav (the committed "
            "public-domain JFK 1961 recording)."
        ),
    )
    args = parser.parse_args(argv)

    # Self-test is mutually exclusive with --do-dump: prevent the two modes
    # from being conflated (a self-test that inadvertently triggers a real
    # dump would blow the network + torch envelope and defeat its purpose).
    if args.self_test and args.do_dump:
        print("error: --self-test and --do-dump are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.self_test:
        return _self_test()

    if not args.do_dump:
        print("error: pass --do-dump to actually run the reference dump, or "
              "--self-test for the offline self-test", file=sys.stderr)
        return 2

    if args.checkpoint_dir is None:
        print("error: --do-dump requires --checkpoint-dir (FR-EX-08: missing "
              "input is a hard error, never a silent skip)", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "tests" / "parity" / "voxtral"
    output_dir.mkdir(parents=True, exist_ok=True)

    return _do_dump(
        checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
        output_dir=output_dir,
        audio=args.audio.expanduser().resolve() if args.audio else None,
    )


if __name__ == "__main__":
    sys.exit(main())
