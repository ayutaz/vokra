# SBV2 v2 real-checkpoint parity fixtures

This directory is the fixture set for **Task 28**'s real-checkpoint SBV2
(Style-Bert-VITS2 v2) parity test
(`crates/vokra-models/tests/parity_sbv2_real.rs`,
`parity_sbv2_real_waveform_matches_reference_dump`) and the loader smoke
tests that sit next to it (`crates/vokra-models/tests/sbv2_gguf_loader.rs`,
`crates/vokra-bert/tests/deberta_v2_loader.rs`). The end-to-end parity
consumer is no longer `#[ignore]`d: it clean-skips when its gitignored real
fixtures are absent and runs under plain `cargo test` once provisioned. This
directory ships no real checkpoint data, only:

- this `README.md` (the recipe below),
- four `*.gguf.sha256` **real committed SHA256 sidecars** (the original
  three committed in `6580061`, with the ZH sidecar added later),
- `reference_dump.manifest.json` (a **schema template**, not a real dump —
  see "About the committed manifest" below).

Everything else — the actual `.gguf` / `.safetensors` checkpoint files and
the `reference_dump/*.bin` tensor dump — is `.gitignore`d (repo root
`.gitignore`, "SBV2 v2 real fixtures" section) and must be produced locally
by an owner following the recipe below. This mirrors the existing
`tests/fixtures/audio/` (Whisper real-audio) convention: fixtures that are
either too large, or under a non-Apache-2.0 upstream license, are never
committed as binary blobs — only the scaffolding that makes them
reproducible is.

## The four checkpoints

| file (in this directory) | role in the manifest | upstream | upstream license | `LicenseClass` | `vokra-cli convert --model` |
|---|---|---|---|---|---|
| `sbv2-v2-multilingual-base.gguf` | `checkpoint.sbv2_main` | [`litagin02/style_bert_vits2`](https://huggingface.co/litagin02) family (v2, JA+EN multilingual base) | AGPL-3.0 | `Copyleft` | `sbv2` |
| `deberta-v2-large-japanese-char-wwm.gguf` | `checkpoint.bert_ja` | [`ku-nlp/deberta-v2-large-japanese-char-wwm`](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) | cc-by-sa-4.0 | `Copyleft` (ShareAlike variant) | `deberta-v2` |
| `deberta-v3-large.gguf` | `checkpoint.bert_en` | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) | MIT | `Permissive` | `deberta-v3` |
| `chinese-roberta-wwm-ext-large.gguf` | `checkpoint.bert_zh` (ZH leg only) | [`hfl/chinese-roberta-wwm-ext-large`](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) | Apache-2.0 | `Permissive` | `bert-base` |

Source: `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §9 "配布 gate
拡張 / SKU" table (gitignore-local design doc; the license/SKU facts are
reproduced here because this README ships in the public repo) and the
`DEFAULT_BERT_JA_REPO` / `DEFAULT_BERT_EN_REPO` constants in
`tools/parity/sbv2_dump_reference.py` (verified live against the HF Hub API
by that script's own module doc, 2026-07-26).

The optional ZH BERT is Apache-2.0; the other weight licenses are listed in
the table. Vokra's own *code* here is clean-room Apache-2.0 (see "Clean-room
reminder" below). Distribution and sign-off decisions remain governed by
`docs/license-audit.md` §3.1.

**Language coverage caveat**: current base ckpt is a JP-Extra fine-tune —
EN/ZH synthesis runs but produces degraded audio; genuine multilingual
synthesis requires switching to a non-JP-Extra multilingual base. See
`docs/adr/sbv2-cleanroom.md` and CLAUDE.md "documented ceilings".

## Directory layout (what's committed vs. gitignored)

```
tests/fixtures/sbv2/
├── README.md                                        # committed (this file)
├── reference_dump.manifest.json                     # committed (schema template)
├── sbv2-v2-multilingual-base.gguf.sha256             # committed (real SHA256, `6580061`)
├── deberta-v2-large-japanese-char-wwm.gguf.sha256    # committed (real SHA256, `6580061`)
├── deberta-v3-large.gguf.sha256                      # committed (real SHA256, `6580061`)
├── chinese-roberta-wwm-ext-large.gguf.sha256         # committed (real SHA256, ZH leg)
│
├── sbv2-v2-multilingual-base.gguf                    # gitignored — produced locally
├── deberta-v2-large-japanese-char-wwm.gguf           # gitignored — produced locally
├── deberta-v3-large.gguf                             # gitignored — produced locally
├── chinese-roberta-wwm-ext-large.gguf                # gitignored — produced on VAST
├── *.safetensors                                     # gitignored — intermediate downloads, if staged here
└── reference_dump/                                   # gitignored — Task 30 dumper output
    ├── phoneme_embed.bin
    ├── text_hidden.bin
    ├── ... (11 tensors for JA/EN, 12 for ZH)
    └── waveform.bin
```

`.gitignore` (repo root) carries the three patterns that keep the middle
and bottom groups out of git:

```
tests/fixtures/sbv2/*.gguf
tests/fixtures/sbv2/*.safetensors
tests/fixtures/sbv2/reference_dump/
```

## Owner recipe

All real-weight steps below are **owner-only** (design doc §12
"依頼者残タスク") and must run on a **vast.ai instance**: they download,
load, convert, and verify large real weights. The canonical four-file ZH path
is the VAST-only worker:

```bash
bash scripts/publish/vast-ai/run-sbv2-zh-parity.sh
```

It reproduces the complete leg without an HF credential or upload. The
workflow mirrors it for an explicit `include_zh=true` dispatch. The manual
steps below remain useful for diagnosis.

### 1. Download the checkpoints

**SBV2 v2 base** — use the dedicated prep tool, which downloads via
`huggingface_hub.snapshot_download` and best-effort maps the upstream
`config.json` onto the `vokra.sbv2.*` hparam schema:

```bash
uv run --project tools/parity/sbv2 --frozen python \
    tools/parity/sbv2_prepare_checkpoint.py \
    --hf-repo litagin/Style-Bert-VITS2-2.0-base-JP-Extra \
    --revision a731761009f3c96d104487be6ad332bf1bb5a3a5 \
    --output-dir /tmp/sbv2-checkpoint --clean-room-defaults
```

This prints a `RESOLVED`/`UNRESOLVED` report for each of the 22 required
`vokra.sbv2.*` fields and writes `/tmp/sbv2-checkpoint/vokra-sbv2-config.json`
— read that report before continuing; an incomplete config makes
`vokra-cli convert --model sbv2` fail loudly at the first missing key
rather than silently guessing (see that tool's own module doc, "CONFIDENCE"
section, for exactly which fields are read/derived/defaulted/left
unresolved). **Multi-shard checkpoints are reported, not merged** — pick
the right `.safetensors` shard yourself if the download has more than one.

**DeBERTa v2 (JA) / v3 (EN)** — JA ships native safetensors; EN ships
`pytorch_model.bin` and must use the same uv-managed safe bridge as CI:

```bash
uv run --project tools/parity/sbv2 --frozen python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ku-nlp/deberta-v2-large-japanese-char-wwm",
    revision="547b0e8b044fba3f9b84d0ab9f990440bd130c8b",
    repo_type="model",
    local_dir="/tmp/deberta-v2-ja",
    allow_patterns=["*.safetensors", "*.json", "vocab.txt"],
)
PY

uv run --project tools/parity/sbv2 --frozen python \
    tools/parity/bin_to_safetensors.py \
    --hf-repo microsoft/deberta-v3-large \
    --revision 64a8c8eab3e352a784c658aef62be1662607476f \
    --output-dir /tmp/deberta-v3-en
```

No `token=` argument is needed for these public checkpoints. Never pass a
token via argv, which can leak through `ps` or shell history.

**Plain BERT (ZH)** — the pinned HFL release ships
`pytorch_model.bin` + `vocab.txt`. Convert the safe `weights_only=True`
pickle through the tracked uv-managed bridge on VAST:

```bash
uv run --project tools/parity/sbv2 --frozen python \
    tools/parity/bin_to_safetensors.py \
    --hf-repo hfl/chinese-roberta-wwm-ext-large \
    --revision a25cc9e05974bd9687e528edd516f2cfdb3f5db9 \
    --output-dir /tmp/bert-zh
```

### 2. Generate the reference dump (Task 30, `sbv2_dump_reference.py`)

```bash
uv run --project tools/parity/sbv2 --frozen python \
    tools/parity/sbv2_dump_reference.py \
    --checkpoint /tmp/sbv2-checkpoint \
    --output-dir tests/fixtures/sbv2 \
    --text "こんにちは。" --language ja \
    --do-dump
```

`--do-dump` is now wired to the tracked MIT VITS vendor modules. It still
fails loudly — and correctly — unless the VAST `tools/parity` uv environment
has its dependencies, the staged checkpoint contains every required tensor,
and `MinimalG2P` has a row for the exact `--text` / `--language` request.
Those gates prevent fabricated reference tensors; do not add heuristic G2P
fallbacks or hand-write a dump.

For the four-file ZH leg, use all pinned local BERT directories:

```bash
uv run --project tools/parity/sbv2 --frozen python \
    tools/parity/sbv2_dump_reference.py \
    --checkpoint /tmp/sbv2-checkpoint \
    --bert-ja-repo /tmp/deberta-v2-ja \
    --bert-en-repo /tmp/deberta-v3-en \
    --bert-zh-repo /tmp/bert-zh \
    --output-dir tests/fixtures/sbv2 --language zh --do-dump
```

The JA/EN DeBERTa encoders do **not** share the VITS-vendor gate — they are
loaded directly via HF `transformers`' `AutoModel` in the same uv-managed
environment:

```bash
uv run --project tools/parity python tools/parity/deberta_v2_dump_reference.py \
    --hf-repo ku-nlp/deberta-v2-large-japanese-char-wwm \
    --output-dir /tmp/deberta-v2-dump --do-dump
uv run --project tools/parity python tools/parity/deberta_v3_dump_reference.py \
    --hf-repo microsoft/deberta-v3-large \
    --output-dir /tmp/deberta-v3-dump --do-dump
```

These two are useful for isolating BERT-only numeric parity independently
of the combined SBV2 pipeline, but their output is **not**
what `reference_dump.manifest.json` / `parity_sbv2_real.rs` consume (that
file only reads `sbv2_dump_reference.py`'s combined 11-tensor dump).

### 3. Convert to GGUF

```bash
vokra-cli convert --model sbv2 \
    --input /tmp/sbv2-checkpoint/<the-resolved>.safetensors \
    --config /tmp/sbv2-checkpoint/vokra-sbv2-config.json \
    --output tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf

vokra-cli convert --model deberta-v2 \
    --input /tmp/deberta-v2-ja/<downloaded>.safetensors \
    --tokenizer /tmp/deberta-v2-ja/vocab.txt \
    --output tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf

uv run --project tools/parity/sbv2 --frozen python \
    tools/parity/extract_spm_metadata.py \
    --input /tmp/deberta-v3-en/spm.model \
    --output /tmp/deberta-v3-en/tokenizer_spm.json
vokra-cli convert --model deberta-v3 \
    --input /tmp/deberta-v3-en/<downloaded>.safetensors \
    --tokenizer /tmp/deberta-v3-en/tokenizer_spm.json \
    --output tests/fixtures/sbv2/deberta-v3-large.gguf

vokra-cli convert --model bert-base \
    --input /tmp/bert-zh/model.safetensors \
    --tokenizer /tmp/bert-zh/vocab.txt \
    --output tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf
```

(`--model sbv2` / `deberta-v2` / `deberta-v3` / `bert-base` are the canonical
`ModelKind::as_arg()` spellings — `crates/vokra-convert/src/lib.rs`
`ModelKind::from_arg` also accepts a few alias spellings, e.g. `sbv2-v2`,
`deberta_v2` with an underscore, etc. The DeBERTa converters need no
`--config`: `n_layers` / `vocab_size` / `d_model` are inferred directly
from the checkpoint's own tensor shapes.)

### 4. Hash and commit

```bash
sha256sum tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf > \
    tests/fixtures/sbv2/sbv2-v2-multilingual-base.gguf.sha256
sha256sum tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf > \
    tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf.sha256
sha256sum tests/fixtures/sbv2/deberta-v3-large.gguf > \
    tests/fixtures/sbv2/deberta-v3-large.gguf.sha256
sha256sum tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf > \
    tests/fixtures/sbv2/chinese-roberta-wwm-ext-large.gguf.sha256
```

Each command **overwrites** that file's current contents with a fresh
`sha256sum` line — a single `<hash>  <path>` line. Only these four
`.sha256` files (plus this README and the manifest) get committed; the
`.gguf` / `.safetensors` themselves stay local (`.gitignore`d). The
sidecars currently on disk (post-`6580061`) already carry real hashes;
this step is only needed after regenerating a GGUF (e.g. after a
converter bump changes what tensors get emitted — see Wave-4
CONVERTER-EMIT-EXPLICIT-ZEROS).

### 5. Run the gated tests

```bash
# Run this heavyweight fixture test only on the VAST instance.
cargo test -p vokra-models --test sbv2_gguf_loader -- --ignored
cargo test -p vokra-bert --test deberta_v2_loader -- --ignored
cargo test -p vokra-models --test parity_sbv2_real \
  parity_sbv2_real_waveform_matches_reference_dump -- --exact --nocapture
```

A missing fixture at this point is a **loud, actionable panic** naming
exactly what's absent — never a silent skip-and-pass (FR-EX-08).

### 6. Generate the SDP-body fixture (Blocker 2c residual)

This is separate from the end-to-end `reference_dump/sdp_sample.bin`: it
isolates the deterministic `pre -> +cond(g) -> DDSConv -> proj` body from
the random latent and flow inverse. The canonical path is the fail-closed VAST
worker, which pins all three upstream revisions, stages and converts all three
GGUFs, checks their committed sidecar hashes, records the execution
environment, generates the independent fixture, and runs the explicit ignored
test:

```bash
cd ~/vokra
bash scripts/publish/vast-ai/run-sbv2-sdp-parity.sh
```

See `docs/handoff/sbv2-sdp-vast-parity.md` for the rent → provision → work →
collect logs → destroy lifecycle. Its `--self-test` is checkpoint-free and may
run locally; the actual path refuses macOS and an unmarked Linux host.

The tool uses the vendored MIT upstream `StochasticDurationPredictor`, not
the Rust implementation, and writes two raw input fixtures plus the
channel-major output and JSON provenance. These, like the real GGUFs, are
derived from the AGPL checkpoint and are gitignored; do not commit or copy
them into the Apache-2.0 source tree. Do not generate or load them on the Mac.
For debugging the worker, its underlying fixture command is:

```bash
cd ~/vokra/tools/parity
uv run python sbv2_sdp_body_dump.py \
    --checkpoint <VAST-staged-sbv2-dir> \
    --output-dir ~/vokra/tests/fixtures/sbv2 \
    --seed 0 --T 50
```

Do not treat that isolated command as a complete gate: the worker's GGUF hash,
environment-provenance, and Rust-test steps are also required.

The first recorded run (2026-08-18, public JP-Extra v2 checkpoint) passed
with `max |Δ| = 9.536743164e-6` at channel 96 / time 31, below the strict
`1e-5` candidate bound. The tracked VAST worker's final end-to-end run at
commit `cdfb3e2` pinned all three upstream revisions, reproduced the three
committed GGUF hashes, recorded Xeon E5-2699 v4 / AVX2 / torch 2.13.0, and
passed in 79.90 s with `max |Δ| = 8.583068848e-6` at channel 118 / time 48.
The observed environment spread is still based on one checkpoint/input pair;
repeat the gate before altering the bound or its explicit `#[ignore]` posture.

## About the committed `reference_dump.manifest.json`

The committed manifest is **not a real dump** — it is the exact,
byte-for-byte stdout of `sbv2_dump_reference.py`'s **schema-preview mode**
(the default when `--do-dump` is omitted), which needs no checkpoint, no
`torch`, and no `transformers` to run:

```bash
uv run --no-project python tools/parity/sbv2_dump_reference.py \
    --checkpoint /tmp/unused --output-dir /tmp/unused --language ja
```

That mode prints the manifest this tool *would* write, with every
`request.*` / `checkpoint.*` field fully resolved from the CLI defaults,
and all 11 JA/EN `tensors[].shape` entries (12 for ZH) left as
**symbolic placeholders**
(`"T_text"` / `"T_bert"` / `"T_mel"` / `"samples"`) since real integer
dimensions only exist after a real forward pass runs. This is the same
honesty discipline the tool applies to a real dump: printing shapes that
*look* real but are not would itself be a fabricated artifact
(NFR-QL-04 / FR-EX-08).

Because the shapes are non-numeric strings, if the fixture set is ever
partially populated (real `.gguf` files dropped in locally without
re-running `--do-dump` to regenerate this manifest), `parity_sbv2_real.rs`'s
`find_tensor` helper fails loudly with `"... shape element is not a
non-negative integer"` the moment it tries to parse one — it never silently
treats the template as real data. **Once a real dump exists, this whole
file must be replaced** with the `--do-dump` run's own
`reference_dump.manifest.json` output (real integer shapes, real
`checkpoint.*`/`request.*` values) — do not hand-edit the shapes in place.

The 11 base dumped tensors plus the optional ZH BERT tensor (design doc §10 /
`parity_sbv2_real.rs` module doc). The Rust test iterates and compares every
tensor listed in the real manifest:

| tensor | shape | purpose |
|---|---|---|
| `phoneme_embed` | `[T_text, 192]` | text encoder input |
| `text_hidden` | `[T_text, 192]` | text encoder output |
| `bert_hidden_ja` | `[T_bert, 1024]` | DeBERTa v2 output (JA) |
| `bert_hidden_en` | `[T_bert, 1024]` | DeBERTa v3 output (EN) |
| `bert_hidden_zh` | `[T_bert, 1024]` | plain-BERT output (ZH leg only) |
| `bert_bridge_out` | `[T_text, 192]` | BERT bridge conv output |
| `speaker_embed` | `[1, 512]` | speaker embedding |
| `style_projected` | `[1, 192]` | style vector projection |
| `sdp_sample` | `[T_text]` | SDP duration sample |
| `mel_hidden` | `[T_mel, 192]` | length-regulated hidden |
| `z_latent` | `[T_mel, 192]` | normalizing-flow output |
| `waveform` | `[1, samples]` | final PCM |

## Sidecar format (the four `.gguf.sha256` files)

**Status (updated 2026-08-18)**: all four sidecars carry **real
`sha256sum`-format** lines. The original three were committed in `6580061`;
the ZH sidecar belongs to the explicit four-file leg. The
gate `[ -s <sidecar> ] && ! grep -q "placeholder" <sidecar>` sketched in
`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10 ("fixture 管理"
/ "CI workflow") now evaluates TRUE for every sidecar, so
`parity-sbv2-real.yml` proceeds to its real dumper + convert + parity
steps rather than skipping.

Historical (pre-`6580061`) format was a **comment-only** placeholder
containing the word `placeholder`, mirroring the convention
`tests/fixtures/audio/jfk-30s.wav.sha256` used before its real audio
fixture landed. That format is preserved in the workflow's own
"Detect ... presence" step (`grep -vE '^\s*(#|$)' <sidecar> | grep -q .`),
so a future SKU whose real hash is not yet computed can be added under
the same placeholder scheme and cleanly gate off until Step 4 replaces
the body with a real `sha256sum` line.

Step 4 above ("Hash and commit") is the canonical way to keep these
sidecars in sync when an owner regenerates the GGUFs — do not
`sha256sum` a stale local file and mix comment lines with a real hash
line.

## Known limitations (read before assuming the recipe "just works" end to end)

These are pre-existing, already-documented facts in the files this recipe
touches — not something this README's task introduces, and not something
this task fixes (each is out of scope for fixture scaffolding; flagged here
so an owner running the recipe isn't surprised):

1. **`sbv2_dump_reference.py --do-dump` needs a staged checkpoint and an
   exact G2P row.** The VITS vendor modules have landed; a missing
   `MinimalG2P` entry or required checkpoint tensor remains a deliberate
   loud failure. Generate the fixture only on VAST through `uv`.

2. **Fixture hashes are converter-version-specific.** The SBV2 v2 and
   DeBERTa v2/v3 converters now map upstream names into the runtime schemas;
   DeBERTa additionally duplicates shared projections and normalizes relative
   embeddings as required by the upstream config. A GGUF produced by an old
   staging-only converter must not be mixed with a newly generated sidecar.
   Regenerate the artifact and hash together, then run the matching real
   loader/parity gate before treating the fixture as current.

3. **Production Mandarin G2P remains separate from numerical parity.** The
   four-file ZH parity consumer composes the real WordPiece/BERT encoder with
   a fixed `PhonemizeFixture` replay. This proves the selected input and all
   downstream native tensors, but it does not turn the fixture row into a
   production Mandarin text frontend. The default loaders remain fail-closed
   when no production phonemizer is supplied.

4. **Several existing test files in this fixture family use a bare
   literal path instead of a `CARGO_MANIFEST_DIR`-relative one**, so they
   resolve incorrectly once `cargo test` puts them at their *crate* root
   rather than the repo root:
   - `crates/vokra-models/tests/sbv2_gguf_loader.rs` opens
     `"tests/fixtures/sbv2/main.gguf"` — both the **base path** (bare
     literal; resolves against `crates/vokra-models/`, not the repo root
     this directory actually lives at) and the **filename** (`main.gguf`,
     not `sbv2-v2-multilingual-base.gguf`) disagree with this fixture set.
   - `crates/vokra-bert/tests/deberta_v2_loader.rs` opens
     `"tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf"` — the
     filename matches this fixture set, but the same bare-literal-path
     issue means it resolves against `crates/vokra-bert/`, not the repo
     root.
   - `crates/vokra-convert/tests/deberta_convert.rs` and
     `crates/vokra-convert/tests/sbv2_convert.rs` (real-checkpoint
     `#[ignore]`d round-trip tests, gated on `*.safetensors` fixtures
     rather than `*.gguf`) have the same bare-literal-path issue, resolving
     against `crates/vokra-convert/`.

   Only `crates/vokra-models/tests/parity_sbv2_real.rs` (Task 28) gets this
   right today, via
   `Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("tests").join("fixtures").join("sbv2")`.
   This directory is deliberately placed at the **repo root**
   (`tests/fixtures/sbv2/`, matching `parity_sbv2_real.rs`'s own resolution
   and this task's brief) — a `CARGO_MANIFEST_DIR`-relative helper for the
   other four call sites (and fixing `sbv2_gguf_loader.rs`'s
   `main.gguf`/`bert_ja.gguf`/`bert_en.gguf` filenames to match the table
   above) is a known follow-up, out of scope for this fixture-management
   task (it touches other crates' test files, not this directory).

5. **A likely-miscoded `ModelKind::from_arg` alias.** *(Fixed 2026-07-27,
   SBV2 v2 plan Task 8.)* `crates/vokra-convert/src/lib.rs`'s `from_arg`
   used to accept `"ku-nlp/deberta-v3-large-japanese-char-wwm"` as an
   alias for `ModelKind::DebertaV3` (the **EN** encoder) — that
   HF-repo-shaped alias string had been copy-pasted from the JA
   (`deberta-v2`) arm above it. The real EN upstream is
   `microsoft/deberta-v3-large` (per the SKU table above), and that is
   now the HF-repo-shaped alias `from_arg` accepts for v3. The old
   nonexistent string is covered as a negative case in
   `modelkind_alias_and_roundtrip_tests::unknown_model_arg_returns_none`
   so it cannot silently return `Some(DebertaV3)` again. This README's
   recipe was unaffected — it always used the canonical `--model
   deberta-v3` spelling.

## Clean-room reminder

This README, and every tool it points to
(`tools/parity/sbv2_prepare_checkpoint.py`,
`tools/parity/sbv2_dump_reference.py`,
`tools/parity/deberta_v2_dump_reference.py`,
`tools/parity/deberta_v3_dump_reference.py`, the converters under
`crates/vokra-convert/src/models/`), is built without reading
`github.com/litagin02/Style-Bert-VITS2` (AGPL-3.0), `github.com/
fishaudio/Bert-VITS2` (AGPL-3.0), or any fork/derivative/blog-post
excerpt of either. The only code references authorized for this fixture
family are the VITS / VITS2 papers, `jaywalnut310/vits` (MIT),
`p0p4k/vits2_pytorch` (MIT), the DeBERTa v2 paper, `microsoft/DeBERTa`
(MIT), and HuggingFace `transformers` (Apache-2.0) — see
`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §6 for the full
allow-list this task inherits. Upstream `config.json` / safetensors
tensor-name **metadata** (structural facts only — names, shapes, dtypes,
license tags) is fair game; the AGPL Python source that produced those
weights is not, and this README never links to it beyond naming the HF
*model card* (weights + license + config, not code) as the download
source in the table above.

## What this README does NOT do

- It does not fix the path-convention / filename mismatches in
  `sbv2_gguf_loader.rs` / `deberta_v2_loader.rs` / `deberta_convert.rs` /
  `sbv2_convert.rs` (see "Known limitations" #4) — those are edits to
  other crates' test files, out of scope for this fixture-management task.
- It does not commit a real checkpoint or raw reference tensor. The tracked
  worker has run the four-file conversion/parity on VAST, but all generated
  binaries remain gitignored.
- It does not replace `docs/license-audit.md` §3.1. That file remains the
  source of truth for each checkpoint's redistribution decision.
