# tools/parity/rmvpe

Offline sidecar for **yxlllc/RMVPE** (or **Dream-High/RMVPE** parent,
same architecture; both MIT) → Vokra parity fixtures. Companion to the
Vokra-side runtime port in
[`crates/vokra-models/src/f0/rmvpe.rs`](../../crates/vokra-models/src/f0/rmvpe.rs)
and the env-gated parity harness
[`crates/vokra-models/tests/parity_rmvpe.rs`](../../crates/vokra-models/tests/parity_rmvpe.rs).

## Status (2026-08-13)

The Vokra runtime port is **feature-complete** as of commit
[`e7b6810`](../../crates/vokra-models/src/f0/rmvpe.rs) (real U-Net +
BiGRU forward, inline `pool2d` / `conv_transpose2d` / `pytorch_gru`
implementation, no external op deps). Path A of the parity harness
(`VOKRA_RMVPE_REAL_GGUF`) already binds shape / finite / sigmoid-range
contract against a real GGUF. **This directory provides the Path B
reference dumper** (`VOKRA_RMVPE_REAL_HIDDEN` + `_ARGMAX` +
`_HIDDEN_FEATURE_DIM`) so the owner can bind the ≥ 99 % argmax-match-rate
gate.

The earlier `RMVPE::extract_real` "loud-partial" (`VokraError::UnsupportedOp`,
per the 2026-07-30 CLAUDE.md wave-3 residual defer) is **resolved** —
see `docs/handoff/vast-ai-publish-rmvpe.md` §1 for the evolution.

## What this directory contains

- **`dump_reference.py`** — the offline reference dumper. Loads the
  upstream `nn.Module` from an owner-supplied checkout, loads the
  released `.pt` pickle, runs the real mel + U-Net forward on a
  supplied PCM (or canned sine sweep), captures the post-CNN hidden
  state via a pre-forward hook on the GRU submodule, and dumps
  `hidden.f32` + `argmax.u32` + `meta.json` for
  `parity_rmvpe.rs` Path B.
- **`fetch_rmvpe_pt.sh`** — owner helper: curl + optional sha256
  verify against a pinned GitHub Releases URL.
- **`pyproject.toml`** — uv project spec (Python 3.12 pinned per
  `[[feedback-python-3-12]]`, deps = `torch` + `numpy` + `soundfile`).
- **`.python-version`** — `3.12` (auto-created by `uv python pin`).

## Honest boundary — what CC did / did not do

**CC-side (this wave):**

- Wrote `dump_reference.py`, `fetch_rmvpe_pt.sh`, and this README.
- Kept the runtime side (`crates/vokra-models/src/f0/rmvpe.rs` +
  `parity_rmvpe.rs`) unchanged in this wave (already landed in
  [`e7b6810`](../../crates/vokra-models/src/f0/rmvpe.rs)).
- Verified the dumper's structural contract via static review against
  the parity-harness env-var / raw-byte-format expectations in
  `crates/vokra-models/tests/parity_rmvpe.rs` L80-104, L342-373.

**Owner-side (per 依頼者ルール #3, ≥ 2 GB models — RMVPE is 180 MB so
local M1 iMac is fine, but sequence is the same):**

1. Fetch the upstream `.pt` (`bash fetch_rmvpe_pt.sh --output ...`).
2. Clone the upstream repo (`git clone https://github.com/yxlllc/RMVPE.git ...`).
3. Run `uv run python dump_reference.py --pt-path ... --upstream-src ... --canned --out-dir ...`.
4. Convert `.pt` → Vokra GGUF via the existing
   `tools/parity/nemo_pt_to_safetensors.py` + `vokra-cli convert
   --model rmvpe` chain (see `docs/handoff/vast-ai-publish-rmvpe.md` §2.1).
5. Export the four env vars (Path A GGUF + Path B hidden / feature_dim
   / argmax) and run `cargo test -p vokra-models --test parity_rmvpe
   -- --nocapture`.
6. Wire the CI variables (`VOKRA_RMVPE_ENABLE=1` +
   `VOKRA_RMVPE_REAL_GGUF_PATH` + Path B env vars) to flip the switch
   on `.github/workflows/parity-rmvpe-real.yml`.

CC never executed `dump_reference.py` — the raw-byte outputs live only
in owner space, ephemeral, per the same-shape workflow the Kokoro and
DFN3 dumpers established.

## Prerequisites

- **`uv`** ([[feedback-python-uses-uv]]) — sidecar toolchain manager.
  Install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or
  `brew install uv`.
- **Python 3.12** — pinned in `.python-version` (`uv sync` auto-installs).
- **A local checkout of the upstream repo** (yxlllc/RMVPE recommended
  for the maintained fork; Dream-High/RMVPE has the same architecture).
  The dumper imports the `nn.Module` from `<upstream-src>/src/model.py`.
- **The released `.pt` checkpoint** (see `fetch_rmvpe_pt.sh`).

## Owner walkthrough — end-to-end

1. **Sync deps** (once per checkout):

   ```
   cd tools/parity/rmvpe
   uv sync
   ```

2. **Fetch the upstream `.pt`** (~ 180 MB):

   ```
   bash ./fetch_rmvpe_pt.sh --output ~/rmvpe-fixtures/rmvpe.pt
   # optional: verify sha256 if the release page publishes one
   #   bash ./fetch_rmvpe_pt.sh --output ~/rmvpe-fixtures/rmvpe.pt --sha256 <hex>
   ```

3. **Clone the upstream repo** (for the `nn.Module` class):

   ```
   git clone https://github.com/yxlllc/RMVPE.git ~/rmvpe-upstream
   ```

4. **Dump reference fixtures** — canned deterministic PCM (fastest,
   fully offline, exactly what the parity harness needs):

   ```
   uv run python dump_reference.py \
       --pt-path      ~/rmvpe-fixtures/rmvpe.pt \
       --upstream-src ~/rmvpe-upstream \
       --canned \
       --out-dir      ~/rmvpe-fixtures/dump
   ```

   Produces:

   - `~/rmvpe-fixtures/dump/hidden.f32` — `[n_frames * feature_dim]` f32 raw le
   - `~/rmvpe-fixtures/dump/argmax.u32` — `[n_frames]` u32 raw le
   - `~/rmvpe-fixtures/dump/meta.json` — provenance + shape metadata

   Or dump against a real 16 kHz mono PCM16 WAV clip:

   ```
   uv run python dump_reference.py \
       --pt-path      ~/rmvpe-fixtures/rmvpe.pt \
       --upstream-src ~/rmvpe-upstream \
       --pcm          ~/my-clip.wav \
       --out-dir      ~/rmvpe-fixtures/dump
   ```

5. **Build the Vokra GGUF** (see
   `docs/handoff/vast-ai-publish-rmvpe.md` §2.1 for the full recipe;
   180 MB fits comfortably on M1 iMac):

   ```
   uv run --project ../.. python tools/parity/nemo_pt_to_safetensors.py \
       --input  ~/rmvpe-fixtures/rmvpe.pt \
       --output ~/rmvpe-fixtures/rmvpe.safetensors
   cd ../..
   cargo build --release -p vokra-cli
   ./target/release/vokra-cli convert \
       --model rmvpe \
       --input  ~/rmvpe-fixtures/rmvpe.safetensors \
       --output ~/rmvpe-fixtures/rmvpe.gguf
   ```

6. **Wire the parity harness** (Path A + Path B in one run):

   ```
   export VOKRA_RMVPE_REAL_GGUF=~/rmvpe-fixtures/rmvpe.gguf
   export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe-fixtures/dump/hidden.f32
   export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=$(python3 -c \
       'import json; print(json.load(open("'"$HOME"'/rmvpe-fixtures/dump/meta.json"))["feature_dim"])')
   export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe-fixtures/dump/argmax.u32
   cargo test -p vokra-models --test parity_rmvpe -- --nocapture
   ```

   Expected result: Path A shape / finite / sigmoid-range contract
   passes; Path B argmax-match rate reports at least 99 % (== mean
   pitch |Δ| well below a semitone, matching the ARGMAX_MATCH_RATE_MIN
   gate in `parity_rmvpe.rs` L104).

7. **Flip the CI switch** on
   `.github/workflows/parity-rmvpe-real.yml` (`VOKRA_RMVPE_ENABLE=1`
   + `VOKRA_RMVPE_REAL_GGUF_PATH` + Path B env vars in the repo
   settings) once the local run is green.

## What the dumper does NOT do

- **No fair-use verbatim of upstream code into this repo** — the
  dumper imports the upstream `nn.Module` at runtime from an
  owner-supplied path (`--upstream-src`). No upstream Python enters
  Vokra source. This matches the DFN3 / Kokoro / Kyutai-STT precedent.
- **No `.pt` bundled in this repo** — the released checkpoint is
  fetched offline by the owner (`fetch_rmvpe_pt.sh`) and consumed only
  by the dumper. The Vokra runtime never sees the `.pt`; it sees only
  the derived GGUF (which is what the parity harness Path A binds).
- **No topology drift compensation** — the argmax-match gate allows
  ± 1 class of drift (20 cents ≈ a fifth of a semitone) which absorbs
  local-centroid decoder rounding, but any drift beyond that is a
  loud parity failure (FR-EX-08).

## Related

- Runtime port: `crates/vokra-models/src/f0/rmvpe.rs` (real U-Net +
  BiGRU, landed [`e7b6810`](../../crates/vokra-models/src/f0/rmvpe.rs))
- Parity harness: `crates/vokra-models/tests/parity_rmvpe.rs`
- Converter: `crates/vokra-convert/src/models/rmvpe.rs`
- CI workflow: `.github/workflows/parity-rmvpe-real.yml`
  (owner-driven flip switch)
- Handoff: `docs/handoff/vast-ai-publish-rmvpe.md` (§1 status
  evolution; §2.1 local M1 iMac walkthrough for the GGUF bridge; §7
  CI flip switch)
- Upstream (yxlllc, maintenance fork, MIT):
  <https://github.com/yxlllc/RMVPE>
- Upstream (Dream-High, original, MIT):
  <https://github.com/Dream-High/RMVPE>
- Paper: Wei et al. 2023 — "RMVPE: A Robust Model for Vocal Pitch
  Estimation in Polyphonic Music" (INTERSPEECH 2023)
- Sister F0 dumper precedents: `tools/parity/dfn3_dump_reference.py`,
  `tools/parity/dump_kokoro_reference.py`
- Memory: [[feedback-python-uses-uv]] / [[feedback-python-3-12]] /
  [[feedback-license-signoff-primary-source]]
