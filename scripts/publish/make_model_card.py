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

# Licence classes that must never be republished by Vokra. Mirrors
# `LicenseClass::requires_research_flag` in
# crates/vokra-core/src/compliance/license_class.rs — kept as literal strings
# because this script must not depend on the Rust build.
NO_REDISTRIBUTE = {"noncommercial", "noncommercial-sharealike", "unknown"}

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


def build_card(path, repo_name=None):
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
    if cls in NO_REDISTRIBUTE:
        raise Refusal(
            f"{path}: weight licence class `{cls}` — Vokra may run this behind a "
            "research flag but must not redistribute it. Publishing refused "
            "(FR-CP-03 / NFR-LC-04)."
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
    """Builds throwaway GGUFs and checks the gate both permits and refuses."""
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

    failures = []
    with tempfile.TemporaryDirectory() as td:
        permissive = Path(td) / "ok.gguf"
        permissive.write_bytes(gguf([
            ("vokra.model.arch", "silero-vad"),
            ("vokra.model.name", "silero-vad-v5"),
            ("vokra.provenance.weight_license", "permissive"),
            ("vokra.provenance.license", "MIT"),
            ("vokra.provenance.source", "snakers4/silero-vad v5 (MIT)"),
        ]))
        card = build_card(permissive)
        for must in ("license: MIT", "silero-vad-v5", "vokra-cli run",
                     "snakers4/silero-vad", "SHA-256"):
            if must not in card:
                failures.append(f"permissive card missing {must!r}")

        for cls in sorted(NO_REDISTRIBUTE):
            p = Path(td) / f"{cls}.gguf"
            p.write_bytes(gguf([
                ("vokra.model.arch", "f5-tts"),
                ("vokra.provenance.weight_license", cls),
                ("vokra.provenance.license", "CC-BY-NC-4.0"),
            ]))
            try:
                build_card(p)
                failures.append(f"class {cls!r} was NOT refused")
            except Refusal:
                pass

        attr_missing = Path(td) / "attr.gguf"
        attr_missing.write_bytes(gguf([
            ("vokra.model.arch", "mimi"),
            ("vokra.provenance.weight_license", "attribution-required"),
            ("vokra.provenance.license", "CC-BY-4.0"),
        ]))
        try:
            build_card(attr_missing)
            failures.append("attribution-required without text was NOT refused")
        except Refusal:
            pass

        bare = Path(td) / "bare.gguf"
        bare.write_bytes(gguf([("vokra.model.arch", "whisper")]))
        try:
            build_card(bare)
            failures.append("an unstamped artifact was NOT refused")
        except Refusal:
            pass

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print(f"make_model_card self-test: OK "
          f"({1 + len(NO_REDISTRIBUTE) + 2} cases)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf", nargs="?", help="converted Vokra GGUF")
    ap.add_argument("--out", help="write the card here (default: stdout)")
    ap.add_argument("--repo-name", help="override the model name in the card")
    ap.add_argument("--print", action="store_true", help="write to stdout")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.gguf:
        ap.error("a GGUF path is required (or --self-test)")

    try:
        card = build_card(a.gguf, a.repo_name)
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
