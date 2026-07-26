# `jaywalnut310/vits` vendor (SBV2 v2 Task 4)

**Status: VENDORED (8 `.py` files at pinned commit
`2e561ba58618d021b5b8323d3765880f7e0ecfdb`, 2021-06-14).** This directory
now ships this `README.md`, the `LICENSE` (unmodified upstream MIT text),
and the 8 vendored Python modules listed under "What ships here" below.
Task 30 landed the scaffold (LICENSE + README + zero source); Task 4
landed the actual VITS core source.

## What this is

[`jaywalnut310/vits`](https://github.com/jaywalnut310/vits) is the original
author reference implementation of VITS (Kim et al. 2021, *Conditional
Variational Autoencoder with Adversarial Learning for End-to-End
Text-to-Speech*, arXiv:2106.06103) — **MIT licensed**. Style-Bert-VITS2's
own upstream architecture is a descendant of this codebase (VITS ->
VITS2 -> Bert-VITS2 -> Style-Bert-VITS2), which is exactly why it is on the
permissive-reference allowlist in
`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §6 ("参照可能
(permissive のみ)" table) and this project's Task 30 brief (line 11):
Vokra's clean-room SBV2 v2 port may read/derive-from this repository's code,
but must **never** read `litagin02/Style-Bert-VITS2` or
`fishaudio/Bert-VITS2` themselves (both AGPL-3.0).

- **Source**: <https://github.com/jaywalnut310/vits>
- **License**: MIT, `Copyright (c) 2021 Jaehyeon Kim` (full text in the
  sibling `LICENSE` file, fetched verbatim from
  `https://raw.githubusercontent.com/jaywalnut310/vits/main/LICENSE` and
  diffed byte-identical against that URL before being committed here —
  never retyped from memory).
- **Pinned commit**: `2e561ba58618d021b5b8323d3765880f7e0ecfdb`
  (`2021-06-14T07:47:15Z`, verified via the public GitHub API on
  `2026-07-26` as the tip of the upstream `main` branch at that time and
  re-fetched during Task 4 vendoring on `2026-07-27`). Pinning a
  specific commit SHA — rather than a moving `main` — is what makes a
  future upstream force-push or edit unable to silently change what
  Vokra's clean-room port was diffed against.

## Why vendored (not `pip install`)

There is no PyPI package published under this name that is this repository
— `pip install vits` does **not** resolve to `jaywalnut310/vits` (per this
task's own ambiguity-resolution notes: "the PyPI package name is NOT `vits`
(that's usually unrelated)"). The only distribution channel is the GitHub
repository itself, so the permissive-license-preserving way to depend on
specific pieces of it is to vendor the minimal needed source with clear
attribution, per `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §6
("tensor name mapping の獲得手順", step 4: "論文 + jaywalnut310/vits (MIT)
と突き合わせ").

## What ships here (Task 4 vendoring pass)

The 8 vendored `.py` files below cover exactly the inference-only surface
`tools/parity/sbv2_dump_reference.py` needs to import + call. Nothing
training-side is vendored — no `train.py`, no `data_utils.py`, no
`losses.py`, no `monotonic_align/` Cython training kernel, no
`preprocess.py`, no `mel_processing.py`, no `utils.py`. Also **not**
vendored: upstream `models.py` as a whole, because it (1) imports
`monotonic_align` at file scope (its `SynthesizerTrn.forward` uses it),
and (2) contains training-side classes (`StochasticDurationPredictor`
already lives in native Rust as `SbV2SDP`, `PosteriorEncoder` /
`DiscriminatorP` / `DiscriminatorS` / `MultiPeriodDiscriminator` /
`SynthesizerTrn` are all training-side or covered by native Rust). The
three inference-only classes we DO need from `models.py`
(`TextEncoder`, `ResidualCouplingBlock`, `Generator`) are each
extracted verbatim into their own target file below (`text_encoder.py`,
`flow.py`, `decoder.py`).

### Target files (README's original contract — one class per file)

| Target file           | Contains                | Upstream source                             | Feeds parity reference for                                              |
|-----------------------|-------------------------|---------------------------------------------|-------------------------------------------------------------------------|
| `text_encoder.py`     | `TextEncoder` (extracted)                | `models.py` lines 135-176                   | `crates/vokra-models/src/sbv2/text_encoder.rs` (`phoneme_embed` / `text_hidden`) |
| `coupling.py`         | `ResidualCouplingLayer`, `WN` (re-export from sibling `modules.py`) | `modules.py` `ResidualCouplingLayer` + `WN` (already in vendored `modules.py`) | affine-coupling step inside `flow.py` below                             |
| `flow.py`             | `ResidualCouplingBlock` (extracted)      | `models.py` lines 179-209                   | `crates/vokra-models/src/sbv2/flow.rs` (`z_latent`)                     |
| `decoder.py`          | `Generator` (extracted, HiFi-GAN vocoder) | `models.py` lines 244-296                   | `crates/vokra-models/src/sbv2/decoder.rs` (`waveform`) — the Rust side already reuses `vokra-ops::hifigan` at ~100% per design doc §7's "既存資産の流用度" table; this Python reference exists purely to dump tensors for diffing, not because Rust needs new logic |

### Transitive dependencies (verbatim upstream, needed to make the above import + run)

The 4 target files above depend on shared upstream utilities that
`jaywalnut310/vits` places in sibling files (`commons.py` for
`init_weights` / `get_padding` / `sequence_mask`; `modules.py` for
`WN` / `ResBlock1` / `ResBlock2` / `LRELU_SLOPE` / `LayerNorm` /
`ResidualCouplingLayer` / `Flip`; `attentions.py` for `Encoder`;
`transforms.py` for the piecewise-rational-quadratic spline used by
`modules.ConvFlow`). Rather than inlining these into the target files
above (which would duplicate ~400 lines and make future re-audits
against upstream harder), we ship them here as separate files, each a
verbatim copy of the corresponding upstream file except for
the 3 minimum imports each needs adapted from bare-top-level
(`import commons` etc.) to relative-in-namespace (`from . import
commons` etc.):

| Vendored file    | Upstream source          | Bytes | Import adaptations (only lines diverging from upstream)                                                                                             |
|------------------|--------------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `commons.py`     | `commons.py`             | 4778  | none (only stdlib + numpy + torch imports)                                                                                                          |
| `transforms.py`  | `transforms.py`          | 8490  | none (only torch + numpy imports)                                                                                                                    |
| `modules.py`     | `modules.py`             | 13166 | `import commons` → `from . import commons`; `from commons import init_weights, get_padding` → `from .commons import init_weights, get_padding`; `from transforms import piecewise_rational_quadratic_transform` → `from .transforms import ...`; **dropped** stale `import scipy` (unreferenced in upstream body; would hard-fail import in parity-venv which does not carry scipy — full rationale in `modules.py`'s own header) |
| `attentions.py`  | `attentions.py`          | 11780 | `import commons` → `from . import commons`; `import modules` → `from . import modules`; `from modules import LayerNorm` → `from .modules import LayerNorm` |

Every adapted import line is documented in each file's header comment
block, so a byte-level `diff` against upstream shows exactly what
diverged and why. Every file's header also carries the pinned commit
SHA + upstream URL + sha256 of the upstream body it was copied from —
that lets a future re-audit (or a re-fetch after upstream force-pushes)
verify at a glance whether anything drifted.

### What is NOT vendored (deliberate exclusions)

- **`models.py` as a whole file** — imports `monotonic_align` at module
  scope, and its non-target classes are either training-side
  (`DiscriminatorP` / `DiscriminatorS` / `MultiPeriodDiscriminator` /
  `SynthesizerTrn`), already covered by native Rust
  (`StochasticDurationPredictor` → `SbV2SDP`), or unused by the
  inference path (`PosteriorEncoder` / `DurationPredictor`). The three
  inference-only classes we do need are extracted verbatim (see target
  files above).
- **`monotonic_align/`** — Cython training kernel, used only in
  `SynthesizerTrn.forward`'s neg-cross-entropy pathfinding at training
  time. The inference reference path (G2P → TextEncoder → DeBERTa
  bridge → SDP → flow → HiFi-GAN) does not touch it.
- **`train.py`, `data_utils.py`, `losses.py`, `preprocess.py`,
  `mel_processing.py`, `utils.py`** — training-side and I/O plumbing,
  irrelevant to a reference forward-pass dumper.

### What this vendoring does + does not unlock

**Does unlock**: `tools/parity/sbv2_dump_reference.py --do-dump`'s
third import gate (line ~361, `from vendor.vits import text_encoder`)
now resolves successfully (in a torch + transformers-equipped
interpreter). The dumper still fails after that gate — see next
bullet.

**Does NOT unlock**: a passing `--do-dump` run. The design doc §7
forward pipeline (G2P → `SbV2TextEncoder` → DeBERTa-bridge → SDP →
flow → HiFi-GAN, writing 11 `reference_dump/*.bin` files) is currently
a documentation comment inside `sbv2_dump_reference.py` at line 378+
("Unreachable until the vendoring above lands. When it does, this is
where the real pipeline goes"). Writing that pipeline body is a
**separate follow-up task**, gated on a real SBV2 v2 checkpoint being
inspected first (design doc §12 owner step) — otherwise a self-
consistent mirror would validate nothing (NFR-QL-04 / FR-EX-08,
`utmos_dump_reference.py` module doc's own Kokoro `92dbc92` lesson).

## Import namespace note (PEP 420)

This directory ships no `__init__.py` (and neither does the parent
`tools/parity/vendor/`). That is deliberate: Python's implicit
namespace-package machinery (PEP 420, 3.3+) resolves `vendor.vits.*`
without one, and `sbv2_dump_reference.py`'s `sys.path.insert(0,
str(Path(__file__).resolve().parent))` puts `tools/parity/` on
`sys.path` so the top-level `vendor` name resolves. Relative imports
inside the vendored files (`from . import commons` etc.) work with
namespace packages exactly the same as they do with regular packages.

## PyTorch API drift note (upstream is from 2021-06-14)

Upstream uses `torch.nn.utils.weight_norm` (see `decoder.py`'s
`Generator` and `modules.py`'s `WN` / `ResBlock1` / `ResBlock2`),
which is `DeprecationWarning`-flagged in torch >= 2.1 in favor of
`torch.nn.utils.parametrizations.weight_norm`. The vendored code
still runs at inference time (verified in the parity venv with torch
2.12.1 per Task 30 report §6); the DeprecationWarning noise on
import + module construction is expected and documented in each
affected file's header — it is not a bug in the vendoring.

## Verification the vendored files remain what they claim

- Each file's header records the upstream URL, byte count, and sha256
  of the upstream body it was copied from. Re-fetch that URL and diff
  against the region below the `# ---8<--- upstream verbatim ...` (or
  `# ---8<--- upstream models.py lines N-M ...`) divider to prove the
  file has not drifted.
- The three files that adapted imports (`modules.py`, `attentions.py`,
  and Task 4 did not touch `commons.py` / `transforms.py`) each document
  the exact old-line → new-line mapping in the header. `diff` against
  upstream should show exactly and only those adapted lines diverging.
- The three files that extracted individual classes from upstream
  `models.py` (`text_encoder.py`, `flow.py`, `decoder.py`) each record
  the exact upstream line-range they were extracted from.
- `coupling.py` extracts no class — it re-exports from the already-
  vendored `modules.py` — so it stays trivially in sync with whatever
  `modules.py` says.

## NOT REFERENCED (clean-room, repeated for this directory)

- `github.com/litagin02/Style-Bert-VITS2` (AGPL-3.0) and all forks
- `github.com/fishaudio/Bert-VITS2` (AGPL-3.0) and all forks
- Any community fork/derivative of either of the above

Only `github.com/jaywalnut310/vits` (MIT, this directory) is in scope
here. Every vendored `.py` file carries the same NOT-REFERENCED header
comment for the same reason.
