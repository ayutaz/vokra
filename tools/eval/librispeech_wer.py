#!/usr/bin/env python3
"""X-06 nightly ASR-WER leg — score vokra Whisper transcripts against the
LibriSpeech references that ship inside the corpus itself.

`docs/milestones.md` §10 X-06 splits the nightly quality-verification work
into "CI job definition + threshold-breach revert/block operation = CC" and
"Tier 2 device lab = owner". This script is the measurement half of the CC
side; `.github/workflows/nightly-asr-wer.yml` is the job that drives it.

Scope — the ASR-WER leg ONLY. X-06 also names an LJSpeech TTS UTMOS leg
(deferred to M5-15, which owns the UTMOS un-defer) and a Tier-2 device
matrix (owner's self-hosted lab). Neither is in this file.

Why the references come out of the corpus
-----------------------------------------
LibriSpeech ships each chapter's ground truth in a sibling
`<speaker>-<chapter>.trans.txt`. Reading those is what makes this a real
measurement: a hand-copied reference list in the repo could silently drift
from the audio it claims to describe, and the resulting WER would look fine
while measuring nothing. The corpus is also never committed — the workflow
downloads it, exactly like every other real-asset job in this tree.

Honest-measurement rules enforced here (FR-EX-08 / NFR-QL-04)
------------------------------------------------------------
* Every pinned utterance MUST be present, with a non-empty hypothesis.
  A missing file is a hard error, never a quietly smaller corpus — scoring
  a subset would move the WER off the calibrated baseline while still
  reporting a number that looks comparable.
* A threshold is only meaningful for the utterance set it was measured on.
  Each supported slice therefore carries its own baseline and threshold
  (see CALIBRATIONS); a slice with no calibration is reported as a plain
  measurement with no verdict, rather than being judged against a number
  that came from a different corpus. The two calibrated slices differ by
  1.8pp precisely because corpus choice moves WER that much.
* No fabricated scores anywhere: if the scoring dependencies are missing,
  this exits non-zero with the install line rather than degrading to some
  cheaper string comparison.

Usage
-----
    python3 tools/eval/librispeech_wer.py \
        --librispeech-root <dir containing dev-clean/> \
        --transcripts <dir of {utterance_id}.txt> \
        [--threshold 0.06] [--json-out report.json] [--summary-out sum.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

# --------------------------------------------------------------------------
# Calibrations.
#
# A threshold is only meaningful for the exact corpus slice it was measured
# on, so each supported slice carries its own measured baseline and its own
# derived threshold. Scoring a slice with no calibration reports the number
# but does NOT invent a pass/fail for it.
# --------------------------------------------------------------------------

#: Speaker/chapter used from LibriSpeech dev-clean.
CAMPAIGN_SPEAKER = "1272"
CAMPAIGN_CHAPTER = "128104"

#: The exact utterances the 2026-07-16 real-weight campaign scored — every
#: other utterance of the chapter (which itself holds 0000..0014). Pinned as
#: an explicit list rather than a stride so that a corpus revision adding or
#: dropping an utterance surfaces as a missing-file error instead of quietly
#: redefining what "the subset" means.
CAMPAIGN_UTTERANCES = (
    "1272-128104-0000",
    "1272-128104-0002",
    "1272-128104-0003",
    "1272-128104-0005",
    "1272-128104-0007",
    "1272-128104-0009",
    "1272-128104-0011",
    "1272-128104-0013",
)

#: Every utterance of the pinned chapter, in trans-file order.
CHAPTER_UTTERANCES = tuple(f"1272-128104-{i:04d}" for i in range(15))

#: Headroom added to a measured baseline to get its pass threshold, in
#: percentage points, then rounded up to the nearest 0.2pp.
#:
#: Why headroom exists at all — it is NOT a noise budget. Run-to-run variance
#: is zero: Whisper greedy decode is deterministic, and flac→WAV decoding was
#: verified bit-identical (int16, sample-for-sample) to the WAVs the campaign
#: measured. The budget absorbs exactly two things:
#:   (a) upstream re-uploads of openai/whisper-base — it is fetched by tag,
#:       not by revision sha;
#:   (b) benign word-level churn from refactors, e.g. the suppress-list
#:       punctuation difference the campaign already observed on
#:       whisper-small.
#: 1.6pp buys ~3 extra word errors on the 206-word subset and ~5 on the
#: 340-word chapter — it scales with corpus size, which is the right shape
#: for "a couple of words may legitimately differ".
#:
#: The regressions this leg exists to catch are nowhere near this fine. The
#: defect class the campaign found (Silero's missing rolling context,
#: Kokoro's upstream-fidelity gap) lands WER at 0.3–1.0, i.e. 5–20x above
#: these lines. Separation is wide in the direction that matters.
HEADROOM_PP = 1.6


def derive_threshold(baseline_wer: float) -> float:
    """baseline + HEADROOM_PP, rounded UP to the nearest 0.2pp.

    Rounding up (never to nearest) keeps the threshold from landing below
    baseline+headroom through floating-point rounding.
    """
    import math

    return math.ceil((baseline_wer + HEADROOM_PP / 100) * 500) / 500


class Calibration(NamedTuple):
    """A corpus slice with a measured baseline and a derived threshold."""

    name: str
    utterances: tuple[str, ...]
    baseline_wer: float
    provenance: str

    @property
    def threshold(self) -> float:
        return derive_threshold(self.baseline_wer)


#: whisper-base over the 8 campaign utterances: 9 word edits (7 sub + 1 del
#: + 1 ins) over 206 reference words.
#:
#: NOTE this is NOT the 3.947% headline in the campaign report — that is the
#: 9-file aggregate which also includes jfk-30s.wav, an utterance this leg
#: does not score. Taking the report figure at face value would have set the
#: baseline ~0.4pp low. It was recomputed over the LibriSpeech utterances
#: alone from the campaign's committed per-file transcripts.
CAMPAIGN_CALIBRATION = Calibration(
    name="campaign-8",
    utterances=CAMPAIGN_UTTERANCES,
    baseline_wer=9 / 206,
    provenance=(
        "recomputed from the 2026-07-16 campaign's committed transcripts "
        "(docs/bench-baselines/m1-real-weight-eval-2026-07-16/), LibriSpeech "
        "utterances only; reproduced end-to-end from the OpenSLR tarball "
        "while authoring this leg (delta +0.0000%)"
    ),
)

#: whisper-base over the whole chapter: 21 word edits (15 sub + 5 del + 1
#: ins) over 340 reference words. Measured while authoring this leg, in the
#: same run that reproduced CAMPAIGN_CALIBRATION exactly — which is what
#: makes it trustworthy as an independent baseline.
#:
#: It is materially worse than the 8-utterance subset (6.18% vs 4.37%): the
#: 7 utterances the campaign left out are harder. That is precisely why a
#: single shared threshold would be wrong here.
CHAPTER_CALIBRATION = Calibration(
    name="chapter-15",
    utterances=CHAPTER_UTTERANCES,
    baseline_wer=21 / 340,
    provenance=(
        "measured 2026-07-19 on whisper-base (fp32 GGUF) in the same run "
        "that reproduced the campaign-8 baseline exactly"
    ),
)

CALIBRATIONS = (CAMPAIGN_CALIBRATION, CHAPTER_CALIBRATION)

#: The nightly default — mirrored by WER_THRESHOLD in
#: .github/workflows/nightly-asr-wer.yml, with the oracle asserting they
#: stay equal.
#:
#: Tighten only against a fresh measurement — never to turn a red run green
#: (the Kokoro PROSODY_F0_ATOL precedent: a tolerance moves only when a
#: measurement and a written rationale move with it).
DEFAULT_THRESHOLD = CAMPAIGN_CALIBRATION.threshold

#: Back-compat alias for the headline baseline.
CAMPAIGN_BASELINE_WER = CAMPAIGN_CALIBRATION.baseline_wer

#: sha256 of LibriSpeech dev-clean.tar.gz (337,926,286 bytes) as downloaded
#: from OpenSLR for the 2026-07-16 campaign, computed locally from that
#: exact file. This is a corpus-swap tripwire, not an upstream-published
#: figure: OpenSLR publishes an MD5 which is NOT reproduced here because it
#: was not independently verified while authoring this leg.
DEV_CLEAN_SHA256 = "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"


class ScoringError(RuntimeError):
    """An input is missing or unusable. Always fatal — see module docstring."""


# --------------------------------------------------------------------------
# Corpus / hypothesis loading (stdlib only, so the TDD oracle can drive it
# without installing the scoring stack).
# --------------------------------------------------------------------------


def parse_trans_file(path: Path) -> dict[str, str]:
    """Parse one LibriSpeech ``*.trans.txt`` into ``{utterance_id: text}``.

    Format is ``<utterance_id> <SPACE> <upper-cased transcript>`` per line.
    A line without a space separator is a corrupt corpus, not something to
    skip past.
    """
    out: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise ScoringError(f"{path}:{lineno}: malformed LibriSpeech transcript line: {line!r}")
        out[parts[0]] = parts[1].strip()
    return out


def utterance_flac_path(root: Path, utterance_id: str) -> Path:
    """Map ``1272-128104-0000`` → ``<root>/1272/128104/1272-128104-0000.flac``."""
    speaker, chapter, _index = utterance_id.split("-", 2)
    return root / speaker / chapter / f"{utterance_id}.flac"


def collect_references(dev_clean_root: Path, utterance_ids: list[str]) -> dict[str, str]:
    """Read references for `utterance_ids` from their chapters' trans files.

    Raises if any requested utterance is absent — a short corpus would make
    the reported WER incomparable to the calibrated baseline.
    """
    by_chapter: dict[Path, dict[str, str]] = {}
    refs: dict[str, str] = {}
    missing: list[str] = []
    for uid in utterance_ids:
        speaker, chapter, _index = uid.split("-", 2)
        trans = dev_clean_root / speaker / chapter / f"{speaker}-{chapter}.trans.txt"
        if trans not in by_chapter:
            if not trans.is_file():
                raise ScoringError(f"LibriSpeech transcript file not found: {trans}")
            by_chapter[trans] = parse_trans_file(trans)
        if uid not in by_chapter[trans]:
            missing.append(uid)
        else:
            refs[uid] = by_chapter[trans][uid]
    if missing:
        raise ScoringError(
            "corpus is missing pinned utterance(s): "
            + ", ".join(missing)
            + " — refusing to score a smaller corpus, which would silently "
            "invalidate the calibrated threshold"
        )
    return refs


def read_hypotheses(transcripts_dir: Path, utterance_ids: list[str]) -> dict[str, str]:
    """Read ``{transcripts_dir}/{utterance_id}.txt`` for each id.

    An absent or whitespace-only transcript is an error: an empty hypothesis
    scores as 100% deletions, which would read as a catastrophic model
    regression when the real fault was a failed transcription step.
    """
    hyps: dict[str, str] = {}
    problems: list[str] = []
    for uid in utterance_ids:
        path = transcripts_dir / f"{uid}.txt"
        if not path.is_file():
            problems.append(f"{uid}: no transcript at {path}")
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            problems.append(f"{uid}: transcript is empty ({path})")
            continue
        hyps[uid] = text
    if problems:
        raise ScoringError(
            "transcription step did not produce usable output:\n  " + "\n  ".join(problems)
        )
    return hyps


def verdict(measured_wer: float, threshold: float) -> tuple[bool, str]:
    """Pass/fail plus a one-line human verdict."""
    passed = measured_wer <= threshold
    state = "PASS" if passed else "FAIL"
    rel = "<=" if passed else ">"
    return passed, f"{state}: WER {measured_wer:.4%} {rel} threshold {threshold:.4%}"


def calibration_for(utterance_ids: list[str]) -> Calibration | None:
    """The calibration measured on exactly this slice, or None.

    Matching is on the utterance *set*, not the order they were requested
    in: corpus-level WER is total-edits over total-reference-words, which is
    order-invariant, so a reordered list is the same measurement. The length
    is compared too, so a duplicated id cannot sneak past as a match.
    """
    for cal in CALIBRATIONS:
        if len(utterance_ids) == len(cal.utterances) and set(utterance_ids) == set(cal.utterances):
            return cal
    return None


def resolve_utterances(args: argparse.Namespace) -> tuple[list[str], Calibration | None]:
    """Return ``(utterance_ids, calibration_or_None)``.

    A `None` calibration means this slice has no measured baseline, so no
    threshold is applied to it — reporting a pass/fail against a number
    measured on a different corpus would be a category error.
    """
    if args.utterances:
        ids = [u.strip() for u in args.utterances.split(",") if u.strip()]
        if not ids:
            raise ScoringError("--utterances was given but parsed to an empty list")
        # References are collected into a dict keyed by utterance id, so a
        # duplicate would silently collapse and quietly shrink the corpus.
        dupes = sorted({u for u in ids if ids.count(u) > 1})
        if dupes:
            raise ScoringError("--utterances lists duplicate id(s): " + ", ".join(dupes))
        return ids, calibration_for(ids)
    if args.all_in_chapter:
        root = Path(args.librispeech_root) / "dev-clean"
        trans = (
            root
            / CAMPAIGN_SPEAKER
            / CAMPAIGN_CHAPTER
            / f"{CAMPAIGN_SPEAKER}-{CAMPAIGN_CHAPTER}.trans.txt"
        )
        if not trans.is_file():
            raise ScoringError(f"LibriSpeech transcript file not found: {trans}")
        ids = sorted(parse_trans_file(trans))
        return ids, calibration_for(ids)
    return list(CAMPAIGN_UTTERANCES), CAMPAIGN_CALIBRATION


# --------------------------------------------------------------------------
# Scoring (needs the jiwer + normalizer stack).
# --------------------------------------------------------------------------


def load_normalizer(normalizer_json: Path):
    """Build whisper's EnglishTextNormalizer from the checkpoint's own map.

    Uses the `transformers` port rather than the `openai-whisper` package.
    The two were compared while authoring this leg — over these utterances
    they produce byte-identical normalizations (0 differing utterances) and
    the identical corpus WER 0.043689320388349516 — so the campaign's
    calibration transfers exactly, and the nightly avoids pulling torch for
    a pure-Python string transform.

    The spelling map is the checkpoint's `normalizer.json`, so it stays
    version-matched to the weights being scored instead of drifting with a
    separately-pinned package.
    """
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    except ImportError as exc:  # pragma: no cover - exercised only in a broken env
        raise ScoringError(
            f"cannot import the whisper English normalizer ({exc}). "
            "Install with: pip install 'transformers>=4.44,<4.60'"
        ) from exc
    if not normalizer_json.is_file():
        raise ScoringError(
            f"normalizer.json not found at {normalizer_json} — it ships with the "
            "HF whisper checkpoint and carries the English spelling map; scoring "
            "without it would apply a different normalization than the baseline"
        )
    return EnglishTextNormalizer(json.loads(normalizer_json.read_text(encoding="utf-8")))


def score(refs: dict[str, str], hyps: dict[str, str], normalizer) -> dict:
    """Corpus-level WER/CER plus per-utterance rows.

    Corpus-level means total edits over total reference words (jiwer's
    list-input behaviour), NOT the mean of per-utterance rates — the latter
    over-weights short utterances and is not what the baseline measured.
    """
    try:
        import jiwer
    except ImportError as exc:  # pragma: no cover - exercised only in a broken env
        raise ScoringError(
            f"cannot import jiwer ({exc}). Install with: pip install 'jiwer>=3,<4'"
        ) from exc

    ids = list(refs)
    norm_refs = [normalizer(refs[u]) for u in ids]
    norm_hyps = [normalizer(hyps[u]) for u in ids]

    blank = [u for u, r in zip(ids, norm_refs) if not r.strip()]
    if blank:
        raise ScoringError(
            "reference normalized to empty for: "
            + ", ".join(blank)
            + " — WER is undefined with no reference words"
        )

    out = jiwer.process_words(norm_refs, norm_hyps)
    per_utterance = [
        {
            "utterance": u,
            "wer": jiwer.wer(r, h),
            "cer": jiwer.cer(r, h),
            "ref_words": len(r.split()),
        }
        for u, r, h in zip(ids, norm_refs, norm_hyps)
    ]
    return {
        "wer": jiwer.wer(norm_refs, norm_hyps),
        "cer": jiwer.cer(norm_refs, norm_hyps),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "edits": out.substitutions + out.deletions + out.insertions,
        "ref_words": sum(len(r) for r in out.references),
        "n_utterances": len(ids),
        "per_utterance": per_utterance,
    }


def render_summary(report: dict) -> str:
    """GitHub step-summary markdown."""
    m, lines = report["measurement"], []
    lines.append("### X-06 nightly ASR-WER (LibriSpeech dev-clean)")
    lines.append("")
    lines.append(f"**{report['verdict']}**")
    lines.append("")
    lines.append(f"model: `{report['model']}`  ")
    lines.append(f"utterances: {m['n_utterances']} ({report['utterance_set']})  ")
    lines.append(f"reference words: {m['ref_words']}")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| WER | {m['wer']:.4%} |")
    lines.append(f"| CER | {m['cer']:.4%} |")
    lines.append(
        f"| word edits | {m['edits']} "
        f"(sub {m['substitutions']} / del {m['deletions']} / ins {m['insertions']}) |"
    )
    if report["threshold"] is not None:
        lines.append(f"| threshold | {report['threshold']:.4%} |")
    else:
        lines.append("| threshold | not applied (uncalibrated slice) |")
    if report["baseline_wer"] is not None:
        delta = m["wer"] - report["baseline_wer"]
        lines.append(f"| baseline (`{report['calibration']}`) | {report['baseline_wer']:.4%} |")
        lines.append(f"| delta vs baseline | {delta:+.4%} |")
    else:
        lines.append("| baseline | n/a — no calibration for this utterance set |")
    lines.append("")
    lines.append("<details><summary>per-utterance</summary>")
    lines.append("")
    lines.append("| utterance | WER | CER | ref words |")
    lines.append("| --- | --- | --- | --- |")
    for row in m["per_utterance"]:
        lines.append(
            f"| `{row['utterance']}` | {row['wer']:.4%} | {row['cer']:.4%} | {row['ref_words']} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append(
        "_Advisory leg (X-06). Not a required check; a breach is an "
        "investigate/revert signal per NFR-MT-07, not a merge blocker._"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--librispeech-root", required=True, help="dir containing dev-clean/")
    p.add_argument("--transcripts", required=True, help="dir of {utterance_id}.txt hypotheses")
    p.add_argument("--normalizer-json", required=True, help="normalizer.json from the HF checkpoint")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the pass threshold. Default: the calibration's own "
        "threshold; an uncalibrated slice is reported without a gate.",
    )
    p.add_argument("--model", default="whisper-base", help="label for the report only")
    p.add_argument("--utterances", default="", help="comma-separated ids (overrides the pinned set)")
    p.add_argument(
        "--all-in-chapter",
        action="store_true",
        help=f"score every utterance of {CAMPAIGN_SPEAKER}/{CAMPAIGN_CHAPTER} "
        "(wider sweep; drops the baseline comparison)",
    )
    p.add_argument("--json-out", default="", help="write the full report as JSON")
    p.add_argument("--summary-out", default="", help="write markdown for $GITHUB_STEP_SUMMARY")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ids, cal = resolve_utterances(args)
        dev_clean = Path(args.librispeech_root) / "dev-clean"
        refs = collect_references(dev_clean, ids)
        hyps = read_hypotheses(Path(args.transcripts), ids)
        normalizer = load_normalizer(Path(args.normalizer_json))
        measurement = score(refs, hyps, normalizer)
    except ScoringError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    # An explicit --threshold always wins; otherwise only a calibrated slice
    # gets gated. Scoring an uncalibrated slice against some other slice's
    # threshold would manufacture a verdict out of an unrelated measurement.
    threshold = args.threshold if args.threshold is not None else (cal.threshold if cal else None)

    if threshold is None:
        passed = True
        line = (
            f"MEASURED (not gated): WER {measurement['wer']:.4%} — no calibration "
            f"for this {len(ids)}-utterance set, so no threshold applies"
        )
    else:
        passed, line = verdict(measurement["wer"], threshold)

    report = {
        "model": args.model,
        "calibration": cal.name if cal else None,
        "utterance_set": cal.name if cal else f"custom ({len(ids)} utterances, uncalibrated)",
        "threshold": threshold,
        # Only claim a baseline when the scored slice is the one it was
        # measured on; otherwise the delta would compare two different corpora.
        "baseline_wer": cal.baseline_wer if cal else None,
        "baseline_provenance": cal.provenance if cal else None,
        "verdict": line,
        "passed": passed,
        "measurement": measurement,
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = render_summary(report)
    if args.summary_out:
        Path(args.summary_out).write_text(summary + "\n", encoding="utf-8")
    print(summary)

    if not passed:
        base = f" (baseline {cal.baseline_wer:.4%}, `{cal.name}`)" if cal else ""
        print(
            f"::error::ASR-WER regression: {measurement['wer']:.4%} exceeds "
            f"threshold {threshold:.4%}{base}",
            file=sys.stderr,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
