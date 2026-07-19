#!/usr/bin/env python3
"""Unit tests for the ``vocab_resource`` gate in ``dump_whisper_reference.py``
(M4-14-T03).

Pure-stdlib and network-free: the dumper's heavy third-party imports
(``numpy`` / ``torch`` / ``transformers``) are stubbed via ``sys.modules``
before import, so this test runs anywhere Python 3 runs — no venv required
(zero-dep NFR-DS-02: nothing is installed to run it). The stubs only need to
satisfy import-time attribute lookups; ``vocab_resource_for`` itself is a pure
function of the size string.

Why this gate matters (see docs/adr/M4-14-whisper-family.md §D4): turbo
(``openai/whisper-large-v3-turbo``) shares the exact 51866-token tokenizer
with large-v3 (in-repo anchor: ``crates/vokra-convert/src/models/whisper.rs``
``all_whisper_sizes_metadata_are_consistent`` rows — n_vocab 51866 for both).
The pre-M4-14 gate ``size == "whisper-large-v3"`` sent turbo down the
tokenizer-derived path while large-v3 used the bundled
``whisper_multilingual_text_vocab.bin`` resource — two code paths for one
vocabulary, i.e. drift risk, contradicting the dumper's own "standardise on
large-v3 to avoid drift" note. M4-14-T03 standardises turbo onto the bundled
resource; these tests pin the gate so a future edit cannot silently regress
turbo back to the second path.

Run:  python3 -m unittest discover tools/parity -p "test_*.py"
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

# Stub the dumper's heavy imports BEFORE importing it. `setdefault` keeps any
# real installed module (e.g. inside the CI venv) authoritative; the stubs are
# only used where torch/transformers/numpy are absent. Import-time code in the
# dumper touches none of these modules, so empty attribute surfaces suffice
# (the `transformers` names are bound at import, hence the three attributes).
for _name in ("numpy", "torch"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_tf = types.ModuleType("transformers")
_tf.WhisperForConditionalGeneration = object
_tf.WhisperProcessor = object
_tf.WhisperTokenizer = object
sys.modules.setdefault("transformers", _tf)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dump_whisper_reference as dumper  # noqa: E402  (stubs must precede)


class VocabResourceGate(unittest.TestCase):
    """Pins `vocab_resource_for` — the bundled-resource standardisation gate."""

    def test_large_v3_uses_bundled_resource(self):
        self.assertEqual(
            dumper.vocab_resource_for("whisper-large-v3"),
            dumper.VOCAB_RESOURCE,
            "large-v3 must regenerate the bundled text-vocab resource",
        )

    def test_turbo_uses_bundled_resource_shared_with_large_v3(self):
        # The M4-14-T03 fix: turbo shares large-v3's 51866 tokenizer, so it
        # must use the SAME bundled resource path (single source, no drift).
        self.assertEqual(
            dumper.vocab_resource_for("whisper-turbo"),
            dumper.VOCAB_RESOURCE,
            "turbo shares the large-v3 51866 tokenizer and must standardise "
            "on the bundled resource (drift-risk fix, ADR M4-14 §D4)",
        )

    def test_80_mel_family_stays_on_tokenizer_derived_path(self):
        # base/small/medium (51865-token vocab) never touch the bundled
        # resource — their tokenizer.bin is dumped per-size as before.
        for size in ("whisper-base", "whisper-small", "whisper-medium"):
            self.assertIsNone(
                dumper.vocab_resource_for(size),
                f"{size}: must not regenerate the bundled resource",
            )

    def test_every_supported_size_has_an_explicit_gate_answer(self):
        # FR-EX-08 shaped: the gate answers for exactly the supported sizes;
        # every SUPPORTED_MODELS key resolves without raising.
        for size in dumper.SUPPORTED_MODELS:
            dumper.vocab_resource_for(size)  # must not raise

    def test_shared_vocab_count_constant_is_51866(self):
        # The fabricated-fixture guard's expected count: the large-v3/turbo
        # family tokenizer length (in-repo anchor: converter test rows).
        self.assertEqual(dumper.LARGE_V3_FAMILY_VOCAB_N, 51866)


if __name__ == "__main__":
    unittest.main()
