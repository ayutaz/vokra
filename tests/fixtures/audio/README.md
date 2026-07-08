# Vokra parity audio fixtures

This directory holds real-audio input used by the Whisper parity dumper
(`tools/parity/dump_whisper_reference.py`) and the M2-06 §3 CI workflow
(`.github/workflows/parity-whisper-real.yml`). The audio must be **committed
as plain binary** (no Git LFS — see `.gitattributes` `-filter` pin) so both
the workflow and the owner-side local regen (`docs/m2-owner-verification-
checklist.md` §3) can reach it without an extra fetch step.

## Files

| Path | Provenance | Rate / channels / codec | Size |
|------|------------|-------------------------|------|
| `jfk-30s.wav` | openai-whisper `tests/jfk.flac` — Public Domain LibriVox recording of JFK's 1961 inaugural address ("Ask not what your country can do for you…") | 16 kHz mono PCM16 (WAV) | ~344 KB (11 s of actual speech × 16 000 × 2 bytes + RIFF header; the dumper `load_pcm()` zero-pads this to 30 s = 480 000 samples internally to match Whisper's `N_SAMPLES`) |
| `jfk-30s.wav.sha256` | `sha256sum tests/fixtures/audio/jfk-30s.wav` output | text sidecar (`<hash>  <path>` — repo-relative) | ~90 bytes |

**File-name vs actual length note**: the file name `jfk-30s.wav` reflects the *window Whisper processes* (`N_SAMPLES = 480 000` samples = 30 s), not the actual clip duration. The source `tests/jfk.flac` from `openai/whisper` is only ~11 s of speech; the dumper's `load_pcm(path, n_samples=480_000)` right-pads the shorter clip with zeros, matching Whisper's own internal front-end padding. Keeping the shorter file avoids committing ~600 KB of silence to the repo.

## Why real audio (not synthetic PCM)

The M0-06 dumper used deterministic synthetic PCM to keep the fixture
byte-reproducible without any owner-side asset commit. That produced
ASR-meaningless "parody" transcripts (`(whistles)`, `(siren wails)`, `eot
only`) — see the per-size table in `docs/m2-owner-verification-checklist.md`
§3 "CC 側実測状況". Real audio is what makes `greedy_transcript_parity`,
`asr_beam_width_one_equals_greedy_and_renders`, and the argmax invariants
in `decoder_logits_parity_full_and_cached` line up with what a downstream
consumer of `openai/whisper-{base,large-v3}` would actually observe. The
dumper was switched to `--audio` in commit `4665d74` (M2-06 §3 follow-up);
this WAV is the missing input.

## Recipe (owner)

The canonical clip is the openai-whisper repo `tests/jfk.flac`, which is
already the correct length (~11 s of speech, right-padded to 30 s by the
Whisper feature extractor). Convert with `ffmpeg`:

```bash
# 1. Fetch the source (Public Domain LibriVox, one-time only; you can also
#    grab tests/jfk.flac from the openai/whisper repo at v20240930).
curl -Lo /tmp/jfk.flac \
  https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac

# 2. Convert to 16 kHz mono PCM16 WAV. -ac 1 forces mono; -ar 16000 forces
#    16 kHz; -c:a pcm_s16le forces int16 (the dumper's `audio_format=1,
#    bits=16` branch in load_pcm() — anything else is an FR-EX-08 hard
#    error, not a silent resample). Do NOT loudness-normalize, do NOT
#    apply any noise reduction — Whisper's own front-end normalizes.
ffmpeg -y \
  -i /tmp/jfk.flac \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  tests/fixtures/audio/jfk-30s.wav

# 3. Write the SHA256 sidecar (repo-relative path, single-line format that
#    `sha256sum -c` and the CI verification step accept).
sha256sum tests/fixtures/audio/jfk-30s.wav > tests/fixtures/audio/jfk-30s.wav.sha256

# 4. Commit both files. LFS is NOT used (~960 KB is well below the LFS
#    threshold; `.gitattributes` explicitly pins `-filter` so a stray
#    `git lfs track "*.wav"` cannot silently attach LFS to this path).
git add tests/fixtures/audio/jfk-30s.wav tests/fixtures/audio/jfk-30s.wav.sha256
git commit -m "chore(parity/whisper): commit jfk-30s.wav real-audio fixture (M2-06 §3)"
```

## Why not Git LFS

- The whole file is ~960 KB — well below the 100 MB GitHub soft limit and
  the 50 MB LFS-recommended threshold.
- Git LFS pointers break `include_bytes!` / mmap workflows in downstream
  crates (none touches this file today, but future test binaries might).
- Owner-side `git clone` + parity regen must "just work" without an extra
  `git lfs install` / `git lfs pull` step, mirroring the "single-binary
  distribution" red-line in `CLAUDE.md` (NFR-DS-02).

## What happens if the WAV is absent

- Local: `python3 tools/parity/dump_whisper_reference.py --model whisper-base`
  exits with a loud FR-EX-08 error containing the ffmpeg recipe above.
- CI (`.github/workflows/parity-whisper-real.yml`): the `setup` job detects
  the missing WAV and skips the parity leg with a step-summary annotation
  telling the owner to run the recipe. It does NOT synthesize a fallback
  input (that's exactly the "silent PCM parody" failure mode that
  4665d74 fixed).

## Related

- `tools/parity/dump_whisper_reference.py` — reads `jfk-30s.wav` via
  `--audio`, writes `input_pcm.f32` / `logmel.f32` / `encoder.f32` /
  `logits_last.f32` / `tokenizer.bin` / `manifest.txt` / `samples.txt`
  under `tests/parity/whisper_{size}/`.
- `crates/vokra-models/tests/parity_whisper.rs` — doubly-gated harness that
  compares the runtime against these fixtures at FP32 `atol = 0.01`
  (NFR-QL-01).
- `.github/workflows/parity-whisper-real.yml` — on-demand + PR-path CI
  that HF-downloads → converts → dumps → parity-tests with the pinned WAV.
- `docs/m2-owner-verification-checklist.md` §3 — owner-side manual regen
  procedure (still valid; the CI workflow complements it, does not replace).
