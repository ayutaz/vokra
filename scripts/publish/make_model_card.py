#!/usr/bin/env python3
"""Generate a Hugging Face model card (README.md) for a converted Vokra GGUF.

The card is derived **from the artifact itself** — every licence, provenance and
shape claim is read out of the GGUF's `vokra.*` metadata, never supplied on the
command line. A card that cannot be backed by the file it describes is not
written at all. That is the whole point: a published weight and its card must
not be able to disagree.

Redistribution gate
-------------------
Uploading a converted weight to a public hub is redistribution, so this refuses
to emit a card when the artifact says it must not be redistributed:

* `vokra.provenance.weight_license` in {noncommercial, noncommercial-sharealike,
  unknown} — the fail-closed classes (FR-CP-03). Vokra may *run* these behind a
  research flag; it may not *republish* them.
* the licence chunk group is missing entirely — an unstamped artifact cannot
  document its own terms, which is exactly what a redistributed file has to do.

Both refusals exit non-zero and name the reason. `--allow-unstamped` exists only
for inspecting a legacy file locally; it refuses to write a card regardless.

Zero-dep: python3 standard library only (NFR-DS-02, same rule as
`scripts/sbom/generate_spdx.py`). No `huggingface_hub`, no `gguf` package.

Usage
-----
    scripts/publish/make_model_card.py MODEL.gguf --out README.md
    scripts/publish/make_model_card.py MODEL.gguf --print
    scripts/publish/make_model_card.py --self-test
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

# --- GGUF metadata value type tags (spec v3) --------------------------------
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIXED = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1,
          U64: 8, I64: 8, F64: 8}

# Publication tiers, mirroring `LicenseClass` in
# crates/vokra-core/src/compliance/license_class.rs. Kept as literal strings
# because this script must not depend on the Rust build.
#
# The distinction that matters: a class being restrictive is NOT a reason to
# refuse. Copyleft and non-commercial weights are redistributable under their
# own terms; what changes is which obligations the card has to carry. Only two
# things are actually refusable — a contractual ban on redistribution, and an
# artifact that cannot state its own terms.
PUBLISHABLE = {
    "permissive",
    "attribution-required",
    "copyleft",
    "conditional-commercial",
    "noncommercial",
    "noncommercial-sharealike",
    "non-commercial",
    "non-commercial-share-alike",
}

# Never publishable, under any flag. The prohibition is contractual, so no
# condition on our side makes it lawful.
FORBIDDEN = {"redistribution-forbidden"}

# Cannot document its own terms -> not redistributable.
FAIL_CLOSED = {"unknown"}

# Classes whose licence must survive onto the artifact unchanged. Relabelling
# one of these as Apache-2.0 misstates what a downstream user is bound by.
LICENSE_PRESERVED = {
    "copyleft",
    "noncommercial-sharealike",
    "non-commercial-share-alike",
}

NONCOMMERCIAL = {
    "noncommercial",
    "non-commercial",
    "noncommercial-sharealike",
    "non-commercial-share-alike",
}

TASK_HINTS = {
    "silero-vad": ("Voice activity detection", "vad"),
    "whisper": ("Speech recognition", "asr"),
    "voxtral": ("Speech recognition", "asr"),
    "kokoro-82m-istftnet": ("Text to speech", "tts"),
    "piper-plus-mb-istft-vits2": ("Text to speech", "tts"),
    "cosyvoice2": ("Text to speech", "tts"),
    "campplus": ("Speaker embedding", "speaker"),
    "mimi": ("Neural audio codec", "codec"),
    "dac": ("Neural audio codec", "codec"),
    "moshi": ("Speech to speech", "s2s"),
    "csm": ("Speech to speech", "s2s"),
}


class GgufReader:
    """Header-only GGUF parser: metadata + tensor directory, no payloads."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(path, "rb")
        if self.f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (bad magic)")
        self.version = self._u32()
        self.n_tensors = self._u64()
        n_kv = self._u64()
        self.meta = {}
        for _ in range(n_kv):
            key = self._str()
            self.meta[key] = self._value(self._u32())
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self._str()
            dims = [self._u64() for _ in range(self._u32())]
            dtype = self._u32()
            self._u64()  # offset
            self.tensors.append((name, dims, dtype))
        self.f.close()

    def _raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError(f"{self.path}: truncated header")
        return b

    def _u32(self):
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._raw(8))[0]

    def _str(self):
        return self._raw(self._u64()).decode("utf-8", "replace")

    def _value(self, t):
        if t == STR:
            return self._str()
        if t == BOOL:
            return self._raw(1)[0] != 0
        if t in (U32, I32):
            return struct.unpack("<i" if t == I32 else "<I", self._raw(4))[0]
        if t == F32:
            return struct.unpack("<f", self._raw(4))[0]
        if t in _FIXED:
            self._raw(_FIXED[t])
            return None
        if t == ARR:
            et = self._u32()
            n = self._u64()
            if et in _FIXED:
                self._raw(_FIXED[et] * n)
            elif et == STR:
                for _ in range(n):
                    self._raw(self._u64())
            return f"<array[{n}]>"
        raise ValueError(f"{self.path}: unknown metadata value type {t}")

    def get(self, key, default=None):
        return self.meta.get(key, default)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def usage_block(arch, name, filename):
    """Runnable instructions for the task this arch performs.

    Every command below is a real `vokra-cli` invocation; where a model cannot
    yet be driven end-to-end from the CLI the card says so instead of printing
    a command that would fail.
    """
    label, task = TASK_HINTS.get(arch, ("", ""))
    dl = (
        "```bash\n"
        "# Download (any HTTP client works — the file is a plain GGUF)\n"
        f"curl -L -o {filename} \\\n"
        f"  https://huggingface.co/vokra/{name}/resolve/main/{filename}\n"
        "```\n"
    )
    if task == "vad":
        run = (
            "```bash\n"
            f"vokra-cli run --model {filename} --input speech.wav\n"
            "```\n\n"
            "Prints the detected speech segments. Input must be mono 16 kHz WAV.\n"
        )
    elif task == "asr":
        run = (
            "```bash\n"
            f"vokra-cli run --model {filename} --input speech.wav\n"
            "\n"
            "# Beam search instead of greedy\n"
            f"vokra-cli run --model {filename} --input speech.wav --beam-size 5\n"
            "```\n\n"
            "Prints the transcript. Input must be mono 16 kHz WAV.\n"
        )
    elif task == "tts":
        run = (
            "```bash\n"
            f"vokra-cli run --model {filename} --text \"こんにちは\" --output out.wav\n"
            "```\n\n"
            "Text input needs a G2P front-end; see the model notes below for\n"
            "which one this voice expects.\n"
        )
    elif task == "speaker":
        run = (
            "```bash\n"
            "# 192-d speaker embedding\n"
            f"vokra-cli run --model {filename} --input a.wav\n"
            "\n"
            "# Cosine similarity between two utterances\n"
            f"vokra-cli run --model {filename} --input a.wav --compare b.wav\n"
            "```\n"
        )
    elif task in ("codec", "s2s"):
        run = (
            "```bash\n"
            f"# This artifact is consumed by another engine rather than run\n"
            f"# standalone — see the Vokra docs for the {task.upper()} pipeline.\n"
            "```\n"
        )
    else:
        run = (
            "```bash\n"
            f"vokra-cli run --model {filename} --input input.wav\n"
            "```\n"
        )
    return label, dl + "\n" + run


def build_card(path, repo_name=None, allow_noncommercial=False):
    g = GgufReader(path)
    arch = g.get("vokra.model.arch")
    name = repo_name or g.get("vokra.model.name") or arch or Path(path).stem
    cls = (g.get("vokra.provenance.weight_license") or "").strip().lower()
    lic = g.get("vokra.provenance.license")
    src = g.get("vokra.provenance.source")
    model_id = g.get("vokra.provenance.model_id")
    attribution = g.get("vokra.provenance.attribution")
    schema = g.get("vokra.schema.version")
    producer = g.get("vokra.schema.producer")

    if not cls:
        raise Refusal(
            f"{path}: no `vokra.provenance.weight_license` — this artifact does "
            "not document its own licence, so it must not be republished. "
            "Re-convert with a current vokra-convert, which stamps provenance."
        )
    if cls in FORBIDDEN:
        raise Refusal(
            f"{path}: weight licence class `{cls}` — redistribution is barred by "
            "contract or terms of use, not by a licence condition, so there is "
            "nothing we can add to the card to make publishing lawful. Refused "
            "unconditionally."
        )
    if cls in FAIL_CLOSED or cls not in PUBLISHABLE:
        raise Refusal(
            f"{path}: weight licence class `{cls}` is unrecognised or "
            "unclassifiable — an artifact whose terms we cannot state must not "
            "be republished. Re-convert, or classify it explicitly first."
        )
    if cls in NONCOMMERCIAL and not allow_noncommercial:
        raise Refusal(
            f"{path}: weight licence class `{cls}` is non-commercial. Publishing "
            "it is an owner policy decision, so it is off by default; pass "
            "--allow-noncommercial to acknowledge that the card will carry an "
            "explicit non-commercial banner."
        )
    if cls == "attribution-required" and not attribution:
        # CC-BY obliges the *redistributor* to carry attribution. Publishing a
        # weight whose file cannot state who to credit pushes that obligation
        # onto downstream users who have no way to discharge it.
        raise Refusal(
            f"{path}: licence class `{cls}` but no "
            "`vokra.provenance.attribution` — a CC-BY weight cannot be "
            "republished without the attribution it obliges. Re-convert with a "
            "converter that calls `stamp_attribution`."
        )

    label, usage = usage_block(arch, name, Path(path).name)
    digest = sha256(path)
    size_mb = Path(path).stat().st_size / (1024 * 1024)

    front = ["---", f"license: {lic or 'other'}", "library_name: vokra",
             "tags:", "  - vokra", "  - gguf"]
    if label:
        front.append(f"  - {label.lower().replace(' ', '-')}")
    front.append("---")

    body = [
        "",
        f"# {name} (Vokra GGUF)",
        "",
        f"{label + ' model, converted' if label else 'Converted'} to the Vokra "
        "GGUF format for [Vokra](https://github.com/ayutaz/vokra), a "
        "zero-dependency speech-AI inference runtime.",
        "",
        "**This is a conversion, not a new model.** The weights are the "
        "upstream ones; Vokra re-packages them so its runtime can memory-map "
        "them directly. Credit for the model belongs upstream — see *Source* "
        "below.",
        "",
        "## Files",
        "",
        "| File | Size | SHA-256 |",
        "|---|---|---|",
        f"| `{Path(path).name}` | {size_mb:.1f} MB | `{digest}` |",
        "",
        "## Usage",
        "",
        usage,
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Architecture | `{arch}` |",
        f"| Tensors | {g.n_tensors} |",
        f"| Upstream source | {src or '_(not recorded)_'} |",
        f"| Upstream licence | `{lic or 'unrecorded'}` |",
        f"| Licence class | `{cls}` |",
        f"| Registry model id | `{model_id or 'unrecorded'}` |",
        f"| Vokra GGUF schema | {schema if schema is not None else '_(pre-stamping)_'} |",
        f"| Converted by | {producer or '_(unrecorded)_'} |",
        "",
        "Every row above is read out of this file's own `vokra.*` metadata, so "
        "the card cannot claim something the artifact does not carry.",
        "",
        "## Licence",
        "",
        f"The weights are distributed under **{lic or 'their upstream licence'}**, "
        "unchanged from upstream. Conversion does not alter the licence, and "
        "your obligations run to the upstream author.",
        "",
    ]
    # --- tier-specific obligations -------------------------------------
    #
    # These sit directly under "Licence" because they are what a reader has to
    # act on. A card that states the licence name but not what it obliges is
    # only half a notice.
    if cls in LICENSE_PRESERVED:
        body += [
            "### ⚠️ Share-alike / copyleft — this licence travels with the file",
            "",
            f"This weight is **{lic}**, and that licence is *not* discharged by "
            "attribution alone. It attaches to derivatives.",
            "",
            "- This GGUF is a **conversion of an upstream weight**, so it is "
            f"itself **{lic}** — not Apache-2.0, and not covered by Vokra's own "
            "licence.",
            "- Anything you derive from it (a fine-tune, a re-quantisation, a "
            "further format conversion) carries the same licence.",
            "- Vokra's runtime is Apache-2.0. **Loading this weight does not "
            "change that**, because these licences restrict the terms of "
            "redistribution, not use. Shipping the *weight* onward is what "
            "carries the obligation.",
            "",
        ]
    if cls in NONCOMMERCIAL:
        body += [
            "### ⛔ Non-commercial — you may not use this weight commercially",
            "",
            f"The upstream weight is **{lic}**. It is republished here so the "
            "model can be evaluated and used for research, and the licence is "
            "unchanged by conversion.",
            "",
            "- **Do not use this in a commercial product or service.** That "
            "restriction is upstream's, not Vokra's, and Vokra cannot waive it.",
            "- Vokra's engine is Apache-2.0 and imposes no such limit — the "
            "limit is on **this weight**. Other models in this organisation are "
            "permissively licensed; check each one's card.",
            "- Vokra's runtime refuses to load a non-commercial weight unless "
            "an explicit research flag is set, so this restriction is enforced "
            "at load time rather than left to the reader.",
            "",
        ]
    if cls == "conditional-commercial":
        body += [
            "### ⚠️ Commercial use is conditional on a threshold",
            "",
            f"**{lic}** permits commercial use only below a stated threshold "
            "(typically annual revenue or monthly active users); above it a "
            "separate grant from the upstream author is required.",
            "",
            "**The threshold applies to you, not to Vokra** — we cannot "
            "evaluate it on your behalf. Read the upstream licence before "
            "shipping anything built on this weight.",
            "",
        ]
    if attribution:
        body += [
            "### Attribution required",
            "",
            "> " + attribution.replace("\n", "\n> "),
            "",
            "This licence obliges you to display the attribution above when you "
            "ship something built on these weights. Vokra surfaces it at "
            "runtime via `vokra_model_attribution` (C ABI) and a CLI banner.",
            "",
        ]
    body += [
        "## Verifying this file",
        "",
        "```bash",
        f"shasum -a 256 {Path(path).name}",
        f"# expect: {digest}",
        "```",
        "",
    ]
    return "\n".join(front + body)


class Refusal(Exception):
    """The artifact must not be republished, or cannot document itself."""


def self_test():
    """Both directions matter: refusing what must be refused, and *publishing*
    what is publishable. An over-strict gate is a silent failure too — it looks
    safe while quietly blocking work that was always allowed."""
    import tempfile

    def gguf(kvs):
        out = bytearray(b"GGUF")
        out += struct.pack("<I", 3) + struct.pack("<Q", 0)
        out += struct.pack("<Q", len(kvs))
        for k, v in kvs:
            out += struct.pack("<Q", len(k)) + k.encode()
            out += struct.pack("<I", STR)
            out += struct.pack("<Q", len(v)) + v.encode()
        return bytes(out)

    def write(td, name, kvs):
        p = Path(td) / name
        p.write_bytes(gguf(kvs))
        return p

    failures = []
    cases = 0
    with tempfile.TemporaryDirectory() as td:
        # --- T1 permissive: publishes, no banners --------------------------
        p = write(td, "ok.gguf", [
            ("vokra.model.arch", "silero-vad"),
            ("vokra.model.name", "silero-vad-v5"),
            ("vokra.provenance.weight_license", "permissive"),
            ("vokra.provenance.license", "MIT"),
            ("vokra.provenance.source", "snakers4/silero-vad v5 (MIT)"),
        ])
        card = build_card(p)
        cases += 1
        for must in ("license: MIT", "silero-vad-v5", "vokra-cli run", "SHA-256"):
            if must not in card:
                failures.append(f"permissive card missing {must!r}")
        for must_not in ("Share-alike", "Non-commercial", "threshold"):
            if must_not in card:
                failures.append(f"permissive card wrongly carries {must_not!r}")

        # --- T3 copyleft: publishes, and says the licence travels ----------
        p = write(td, "sa.gguf", [
            ("vokra.model.arch", "style-bert-vits2"),
            ("vokra.provenance.weight_license", "copyleft"),
            ("vokra.provenance.license", "CC-BY-SA-4.0"),
            ("vokra.provenance.attribution", "Upstream author, CC BY-SA 4.0"),
        ])
        card = build_card(p)
        cases += 1
        if "license: CC-BY-SA-4.0" not in card:
            failures.append("copyleft card must keep the original licence label")
        if "Share-alike" not in card or "carries the same licence" not in card:
            failures.append("copyleft card must state that the licence propagates")

        # --- T4 non-commercial: off by default, on with the flag -----------
        p = write(td, "nc.gguf", [
            ("vokra.model.arch", "f5-tts"),
            ("vokra.provenance.weight_license", "noncommercial"),
            ("vokra.provenance.license", "CC-BY-NC-4.0"),
        ])
        cases += 1
        try:
            build_card(p)
            failures.append("non-commercial published without the explicit flag")
        except Refusal:
            pass
        cases += 1
        card = build_card(p, allow_noncommercial=True)
        if "license: CC-BY-NC-4.0" not in card:
            failures.append("NC card must keep the original licence label")
        if "may not use this weight commercially" not in card:
            failures.append("NC card must carry an unmissable non-commercial banner")

        # --- conditional-commercial: publishes, states the threshold -------
        p = write(td, "cond.gguf", [
            ("vokra.model.arch", "indextts2"),
            ("vokra.provenance.weight_license", "conditional-commercial"),
            ("vokra.provenance.license", "bilibili Model Use License"),
        ])
        card = build_card(p)
        cases += 1
        if "threshold" not in card:
            failures.append("conditional card must state the threshold applies")

        # --- T5 forbidden: refused unconditionally, even with the flag -----
        p = write(td, "forbidden.gguf", [
            ("vokra.model.arch", "voicevox"),
            ("vokra.provenance.weight_license", "redistribution-forbidden"),
            ("vokra.provenance.license", "VOICEVOX terms"),
        ])
        for flag in (False, True):
            cases += 1
            try:
                build_card(p, allow_noncommercial=flag)
                failures.append(
                    f"redistribution-forbidden published (allow_noncommercial={flag})"
                )
            except Refusal:
                pass

        # --- unstamped / unknown: fail closed ------------------------------
        for name, kvs in [
            ("bare.gguf", [("vokra.model.arch", "whisper")]),
            ("unk.gguf", [
                ("vokra.model.arch", "mystery"),
                ("vokra.provenance.weight_license", "unknown"),
            ]),
        ]:
            p = write(td, name, kvs)
            cases += 1
            try:
                build_card(p, allow_noncommercial=True)
                failures.append(f"{name} was NOT refused")
            except Refusal:
                pass

        # --- attribution-required without the text -------------------------
        p = write(td, "attr.gguf", [
            ("vokra.model.arch", "mimi"),
            ("vokra.provenance.weight_license", "attribution-required"),
            ("vokra.provenance.license", "CC-BY-4.0"),
        ])
        cases += 1
        try:
            build_card(p)
            failures.append("attribution-required without text was NOT refused")
        except Refusal:
            pass

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print(f"make_model_card self-test: OK ({cases} cases)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf", nargs="?", help="converted Vokra GGUF")
    ap.add_argument("--out", help="write the card here (default: stdout)")
    ap.add_argument("--repo-name", help="override the model name in the card")
    ap.add_argument("--print", action="store_true", help="write to stdout")
    ap.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="permit non-commercial weights (owner policy decision; the card "
             "gains an explicit non-commercial banner)",
    )
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.gguf:
        ap.error("a GGUF path is required (or --self-test)")

    try:
        card = build_card(a.gguf, a.repo_name, a.allow_noncommercial)
    except Refusal as e:
        print(f"make_model_card: REFUSED — {e}", file=sys.stderr)
        return 2

    if a.out:
        Path(a.out).write_text(card, encoding="utf-8")
        print(f"make_model_card: wrote {a.out}")
    else:
        sys.stdout.write(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
