#!/usr/bin/env python3
# vokra-misaki-g2p / python bridge — text -> phoneme string (misaki, upstream).
#
# The Rust wrapper (../src/lib.rs) shells out to this script. We take the text
# on argv (not stdin, so the terminal quoting story matches the CLI at large)
# and write a single JSON object to stdout:
#
#     {"phonemes": "hɜˈloʊ wɜɹld", "lang": "en"}
#
# The Rust side then maps every character of `phonemes` through the Kokoro
# GGUF's `vokra.kokoro.phoneme_symbols` table to obtain the phoneme id
# sequence `synthesize_phonemes` consumes. Unknown symbols become a loud
# error on the Rust side (FR-EX-08: never a silent skip).
#
# Fail-closed: unknown language, missing misaki, or a misaki call that raises
# is written to stderr and the script exits non-zero. The Rust side then
# surfaces the message to the user without adding its own interpretation.
#
# misaki API summary (upstream https://github.com/hexgrad/misaki):
#
#     from misaki import en
#     g2p = en.G2P(trf=False, british=False)
#     phonemes, tokens = g2p(text)   # phonemes: str (IPA-like, with stress)
#
# The English lexicon covers stress markers `ˈ` and `ˌ`, both of which
# Kokoro's `phoneme_symbols` table includes for the af/am voice family.

from __future__ import annotations

import argparse
import json
import sys


def die(msg: str) -> "None":
    """Fail-closed: no attempt at recovery, no fallback lexicon."""
    print(f"misaki_bridge: {msg}", file=sys.stderr)
    raise SystemExit(2)


def phonemize(lang: str, text: str) -> str:
    """Return the IPA-like phoneme string upstream Kokoro trains against.

    Language dispatch mirrors the misaki module layout: `en`, `ja`, `zh`, `ko`.
    Any other language is out of scope for this bridge (the four Kokoro
    upstream ships trained voices for).
    """
    try:
        # Localised imports so a missing sub-module names the offender clearly.
        if lang == "en":
            from misaki import en  # noqa: WPS433
            g2p = en.G2P(trf=False, british=False)
        elif lang == "en-gb":
            from misaki import en  # noqa: WPS433
            g2p = en.G2P(trf=False, british=True)
        elif lang == "ja":
            from misaki import ja  # noqa: WPS433
            g2p = ja.G2P()
        elif lang == "zh":
            from misaki import zh  # noqa: WPS433
            g2p = zh.G2P()
        elif lang == "ko":
            from misaki import ko  # noqa: WPS433
            g2p = ko.G2P()
        else:
            die(
                f"unsupported language: {lang!r} "
                "(supported: en / en-gb / ja / zh / ko — the Kokoro training set)"
            )
    except ImportError as e:
        die(
            f"cannot import misaki for lang={lang!r}: {e}. "
            "Install with `pip install misaki[en,ja,zh,ko]` "
            "(see integrations/vokra-misaki-g2p/README.md)."
        )

    try:
        result = g2p(text)
    except Exception as e:  # noqa: BLE001 — surface the upstream message
        die(f"misaki.{lang}.G2P raised: {e}")

    # misaki normally returns (phonemes: str, tokens: list). Older builds may
    # return the string alone; handle both without inventing behaviour.
    if isinstance(result, tuple) and result and isinstance(result[0], str):
        return result[0]
    if isinstance(result, str):
        return result
    die(f"misaki.{lang}.G2P returned unexpected type: {type(result).__name__}")


def main() -> "None":
    ap = argparse.ArgumentParser(
        description="text -> misaki IPA phoneme string, one JSON line on stdout"
    )
    ap.add_argument(
        "--lang",
        required=True,
        choices=["en", "en-gb", "ja", "zh", "ko"],
        help="misaki language module to load (matches upstream module names)",
    )
    ap.add_argument(
        "--text",
        required=True,
        help="the input text (quote appropriately at the shell)",
    )
    args = ap.parse_args()

    phonemes = phonemize(args.lang, args.text)
    # A single JSON line so the Rust side can parse without a streaming
    # decoder; `ensure_ascii=False` keeps the IPA characters legible in logs.
    print(json.dumps({"lang": args.lang, "phonemes": phonemes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
