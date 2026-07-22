# X-06 requirement revision — proposal (owner sign-off required)

**Status**: PROPOSAL. CC does not edit the X-06 completion condition itself
(owner decision, X-06-T20). This document states the mismatch and proposes a
revision; the milestone text changes only after owner sign-off.

## The mismatch

The X-06 completion condition (milestones.md §10) requires a nightly quality
verification over **three corpora — LibriSpeech, LJSpeech, and VCTK**. The
workflow that landed (`nightly-asr-wer.yml`) implements only the **LibriSpeech
ASR-WER** leg and explicitly scopes the other two out:

- **LJSpeech TTS UTMOS/MOS** — deferred to **M5-15**, which owns the UTMOS
  un-defer (weights + license sign-off). Wiring a UTMOS job into the ASR-WER
  leg today would need weights that do not exist there yet, or a fabricated
  score — both barred (NFR-QL-04, FR-EX-08).
- **VCTK VC-similarity** — voice-conversion similarity is out of scope for the
  core repo. This is **consistent with CLAUDE.md design decision 8**: voice
  cloning / VC is fully separated into `vokra-voiceclone-experimental`. A VC
  quality corpus therefore does not belong in the core repo's nightly.

Read literally, the three-corpus condition is permanently unmet — not because
of a gap in the work, but because two of its three corpora were reassigned by
later decisions. That is a documentation-vs-decision drift, not a defect.

## Proposed revision (owner to ratify or amend)

Replace the three-corpus clause with:

> Nightly quality verification runs and results are published. The core-repo
> nightly measures **LibriSpeech ASR-WER** (`nightly-asr-wer.yml`) against a
> per-slice calibrated threshold; a breach drives revert/block (NFR-MT-07,
> see `docs/handoff/x-06-breach-response.md`). **LJSpeech TTS UTMOS/MOS** is
> owned by **M5-15** (UTMOS un-defer) and referenced from there rather than
> duplicated here. **VCTK VC-similarity is removed** from the core-repo X-06
> scope per design decision 8 (voice conversion lives in
> `vokra-voiceclone-experimental`); if a VC-quality nightly is wanted, it
> belongs to that repo's CI, not this one.

## VCTK: two options (owner picks)

1. **Delete** the VCTK clause outright. Cleanest; matches design decision 8
   (VC is not a core-repo concern at all). *(CC recommendation.)*
2. **Migrate-by-reference**: keep a one-line pointer that VC-quality nightly,
   if built, is a `vokra-voiceclone-experimental` CI task. Preserves the
   intent without importing VC into the core repo.

Both keep the core repo free of a VC corpus; option 2 only leaves a
breadcrumb.

## What is already true (no revision needed)

- The **LibriSpeech ASR-WER** leg is implemented, has a per-slice calibrated
  threshold, a real corpus checksum pin, and a breach → red-nightly posture.
- Its pre-verify oracle (`tools/eval/test_librispeech_wer.py`) is green.
- The model-level RTF companions (Silero PR-time; whisper-base + piper
  nightly, X-06-T11/T12) and the Tier 2 device matrix
  (`nightly-tier2-device.yml`, X-06-T21) extend the same posture to the
  performance thresholds (NFR-PF-09/10) the condition also names.

## Owner action

1. Choose VCTK option 1 or 2.
2. Edit the X-06 completion condition in `docs/milestones.md` §10 to the
   revised wording above (CC does not touch this line — X-06-T20).
3. Confirm the M5-15 UTMOS leg is the agreed home for LJSpeech quality.
