#!/usr/bin/env python3
"""Report which models can be published to a public hub today, and what blocks
the rest.

Publishing a converted weight is redistribution. Three independent conditions
must ALL hold, and they fail for different reasons and are fixed by different
people:

1. **Implemented** — a converter exists, so an artifact can be produced at all.
2. **Licence permits redistribution** — class is `permissive` or
   `attribution-required`; the fail-closed classes are run-only.
3. **Signed off** — the `docs/license-audit.md` §3.1 row is filled in. Blank is
   deliberate and means "not cleared for official distribution"; filling it is
   an owner legal judgement, not a code change.

Conflating these is how a project ends up either over-publishing (shipping
something it had no clearance for) or under-publishing (sitting on models that
were cleared months ago). This prints them separately so the owner's decision
list is exactly the rows that need a human.

Reads only tracked docs and the source tree — no network, no GGUF required, so
it answers "what could we publish" before anything is converted.

Zero-dep: python3 standard library only (NFR-DS-02).

Usage:
    scripts/publish/publishability-report.py
    scripts/publish/publishability-report.py --markdown
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "docs" / "license-audit.md"

# Shared §3.1 parser + explicit alias maps. Importing (rather than
# re-implementing the loop below) keeps this report and upload.sh's
# refusal in agreement — a row that upload.sh treats as blank cannot
# show up as "publishable today" here, which is exactly the class of
# drift that caused the original substring bug.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import signoff_match  # noqa: E402

# Catalog name -> converter module stem under crates/vokra-convert/src/models/.
# Only models with a converter can produce an artifact to publish at all.
CONVERTERS = {
    "Silero VAD v5": "silero",
    "Whisper base/small/medium/large-v3/turbo": "whisper",
    "piper-plus (ayutaz) 全モデル": "piper_plus",
    "Kokoro-82M": "kokoro",
    "CosyVoice / CosyVoice2 / CosyVoice3": "cosyvoice2",
    "Sesame CSM-1B": "csm",
    "Moshi (Helium + Mimi)": "moshi",
    "Voxtral (Mistral)": "voxtral",
    "DAC (Descript)": "dac",
    "Mimi codec (Kyutai)": "mimi",
    "CAM++ Speaker Embedding": "campplus",
    "DeepFilterNet3": "denoise",
    "UTMOS22-strong (SaruLab)": "utmos",
}

REDISTRIBUTABLE = {"permissive", "attribution-required"}


def license_class_for(stem):
    """Reads the class the converter stamps, from the converter source itself.

    Parsing the source rather than hard-coding a table keeps this honest: if a
    converter's stamp changes, this report changes with it.
    """
    p = REPO / "crates" / "vokra-convert" / "src" / "models" / f"{stem}.rs"
    if not p.is_file():
        return None, None
    src = p.read_text(encoding="utf-8")
    m = re.search(r"stamp_provenance\(\s*&mut b,\s*(?:[\w:]*::)?LicenseClass::(\w+),\s*\"([^\"]+)\"", src)
    if not m:
        return None, None
    rust_class = m.group(1)
    wire = {
        "Permissive": "permissive",
        "AttributionRequired": "attribution-required",
        "NonCommercial": "noncommercial",
        "NonCommercialShareAlike": "noncommercial-sharealike",
        "Unknown": "unknown",
    }.get(rust_class, rust_class.lower())
    return wire, m.group(2)


def signoff_rows(audit_path=None):
    """Parses the §3.1 sign-off table -> {model-name: approved?}.

    Delegated to signoff_match.parse_signoff_rows so this report and
    upload.sh's refusal read the audit the exact same way — a row that
    upload.sh treats as blank cannot show up as "publishable today"
    here, which is exactly the class of drift that caused the original
    substring bug. Column layout (verified against the live table):

        | model | weight licence | audit date | approver | decision | notes |

    A row counts as approved only when the **approver** cell holds a real
    name (the template writes `______________`) AND the **decision** cell
    has a ticked box. An unticked
    `☐ Commercial / ☐ Research-only / ☐ Rejected` is the blank template,
    not a decision.

    Erring toward "not approved" is the whole point: reporting a model as
    publishable when its row is blank would defeat the fail-closed design
    the blank row exists to implement.
    """
    return signoff_match.parse_signoff_rows(Path(audit_path or AUDIT))


# Sign-off rows are keyed by release-specific names ("DAC 24khz (Descript)")
# while the catalog uses family names ("DAC (Descript)"). Map explicitly rather
# than fuzzy-matching: a near-miss here would silently report an unsigned model
# as publishable.
SIGNOFF_ALIASES = {
    "DAC (Descript)": ["DAC 24khz (Descript)"],
    "Mimi codec (Kyutai)": ["Mimi codec (Kyutai)"],
    "Sesame CSM-1B": ["Sesame CSM-1B"],
    "Moshi (Helium + Mimi)": ["Moshi (Helium + Mimi)"],
    "UTMOS22-strong (SaruLab)": ["UTMOS22-strong (SaruLab)"],
    "CosyVoice / CosyVoice2 / CosyVoice3": ["CosyVoice2-0.5B"],
    "Voxtral (Mistral)": ["Voxtral-Mini-3B-2507", "Voxtral-Small-24B-2507"],
    "CAM++ Speaker Embedding": ["CAM++"],
    "DeepFilterNet3": ["DeepFilterNet3"],
    "Whisper base/small/medium/large-v3/turbo": [],
    "Silero VAD v5": [],
    "piper-plus (ayutaz) 全モデル": [],
    "Kokoro-82M": [],
}


def _self_test():
    """Hermetic self-test — no network, no real audit read.

    Drives a synthetic §3.1 fixture through the shared parser (via
    signoff_match) and asserts the three partitions this report emits:
    ready / blocked_signoff / blocked_license. Also confirms
    signoff_match's own self-test passes so the shared invariant we
    depend on is exercised here too.
    """
    import subprocess as sp
    import tempfile

    fixture = (
        "### Owner sign-off template\n\n"
        "| Model | Weight License | CC-verified date | Owner sign-off (YYYY-MM-DD) | Approval | Notes |\n"
        "|---|---|---|---|---|---|\n"
        "| **APPROVED-Model** | MIT | 2026-01-01 | 2026-01-02 yousan | ☑ Commercial / ☐ Research-only / ☐ Rejected | approved |\n"
        "| **PENDING-Model** | MIT | 2026-01-01 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | blank |\n"
    )

    failures = []

    # 1. Shared parser must agree with the fixture.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "audit.md"
        p.write_text(fixture, encoding="utf-8")
        rows = signoff_rows(p)
        if rows.get("APPROVED-Model") is not True:
            failures.append(f"APPROVED-Model should be True, got {rows.get('APPROVED-Model')!r}")
        if rows.get("PENDING-Model") is not False:
            failures.append(f"PENDING-Model should be False, got {rows.get('PENDING-Model')!r}")

    # 2. signoff_match's own self-test must pass — the parser we import
    #    from there is the single source of truth for the audit format.
    #    Same-interpreter import (already loaded); shell out to preserve
    #    the same env that CI will use.
    matcher = Path(__file__).resolve().parent / "signoff_match.py"
    rc = sp.call(
        ["python3", str(matcher), "--self-test"],
        stdout=sp.DEVNULL, stderr=sp.DEVNULL,
    )
    if rc != 0:
        failures.append(f"signoff_match --self-test exited {rc}")

    # 3. license_class_for regex must recognise a synthetic converter
    #    stamp (this is the piece publishability-report parses itself).
    with tempfile.TemporaryDirectory() as tmp:
        # Build a shim converter that stamps the exact form the regex expects.
        crate_dir = Path(tmp) / "crates" / "vokra-convert" / "src" / "models"
        crate_dir.mkdir(parents=True)
        (crate_dir / "shim.rs").write_text(
            'pub fn f() { stamp_provenance(&mut b, LicenseClass::Permissive, "mit"); }\n',
            encoding="utf-8",
        )
        # Point the module's REPO at the fixture root for this call only.
        global REPO
        real_repo = REPO
        REPO = Path(tmp)
        try:
            cls, spdx = license_class_for("shim")
            if cls != "permissive":
                failures.append(f"license_class_for('shim').cls = {cls!r}, want 'permissive'")
            if spdx != "mit":
                failures.append(f"license_class_for('shim').spdx = {spdx!r}, want 'mit'")
        finally:
            REPO = real_repo

    if failures:
        print("publishability-report self-test: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("publishability-report self-test: OK (3 cases: parser fixture + signoff_match delegated + license_class regex)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="hermetic self-test (no network, no real audit)")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    signoffs = signoff_rows()
    ready, blocked_license, blocked_signoff, no_converter = [], [], [], []

    for name, stem in sorted(CONVERTERS.items()):
        cls, spdx = license_class_for(stem)
        if cls is None:
            no_converter.append((name, stem, "converter does not stamp provenance"))
            continue
        if cls not in REDISTRIBUTABLE:
            blocked_license.append((name, cls, spdx))
            continue
        # A model is cleared only when EVERY sign-off row that covers it is
        # approved. No row at all means the audit never placed a hold on it.
        aliases = SIGNOFF_ALIASES.get(name, [name])
        relevant = [signoffs[a] for a in aliases if a in signoffs]
        if relevant and not all(relevant):
            pending = [a for a in aliases if a in signoffs and not signoffs[a]]
            blocked_signoff.append((name, cls, spdx, ", ".join(pending)))
        else:
            ready.append((name, cls, spdx))

    out = []
    w = out.append
    if a.markdown:
        w("## Publishable today (no owner action needed)\n")
        w("| Model | Licence | Class |")
        w("|---|---|---|")
        for n, c, s in ready:
            w(f"| {n} | `{s}` | `{c}` |")
        w("\n## Blocked on an owner sign-off (§3.1 row is blank)\n")
        w("| Model | Licence | Class | Pending row |")
        w("|---|---|---|---|")
        for n, c, s, rows in blocked_signoff:
            w(f"| {n} | `{s}` | `{c}` | §3.1 row: {rows} |")
        if blocked_license:
            w("\n## Not redistributable (run-only)\n")
            w("| Model | Class |")
            w("|---|---|")
            for n, c, _ in blocked_license:
                w(f"| {n} | `{c}` |")
        if no_converter:
            w("\n## Cannot produce an artifact\n")
            for n, stem, why in no_converter:
                w(f"- {n} (`{stem}.rs`): {why}")
    else:
        w(f"PUBLISHABLE TODAY ({len(ready)}):")
        for n, c, s in ready:
            w(f"  + {n:52s} {s:14s} [{c}]")
        w(f"\nBLOCKED — owner sign-off required ({len(blocked_signoff)}):")
        for n, c, s, rows in blocked_signoff:
            w(f"  ! {n:52s} {s:14s} [{c}]  <- §3.1: {rows}")
        if blocked_license:
            w(f"\nNOT REDISTRIBUTABLE — run-only ({len(blocked_license)}):")
            for n, c, _ in blocked_license:
                w(f"  x {n:52s} [{c}]")
        if no_converter:
            w(f"\nNO ARTIFACT POSSIBLE ({len(no_converter)}):")
            for n, stem, why in no_converter:
                w(f"  ? {n:52s} {why}")
        w("")
        w("Sign-off rows live in docs/license-audit.md §3.1. A blank row is")
        w("deliberate (fail-closed) and filling it is a legal judgement.")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
