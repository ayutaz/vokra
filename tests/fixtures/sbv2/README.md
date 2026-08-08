# SBV2 v2 real-checkpoint parity fixtures

This directory is the fixture set for **Task 28**'s real-checkpoint SBV2
(Style-Bert-VITS2 v2) parity test
(`crates/vokra-models/tests/parity_sbv2_real.rs`,
`parity_sbv2_real_waveform_matches_reference_dump`) and the loader smoke
tests that sit next to it (`crates/vokra-models/tests/sbv2_gguf_loader.rs`,
`crates/vokra-bert/tests/deberta_v2_loader.rs`). All of those tests are
`#[ignore]`d — `cargo test` skips them by default because **this directory
ships no real checkpoint data**, only:

- this `README.md` (the recipe below),
- three `*.gguf.sha256` **real committed SHA256 sidecars** (committed
  `6580061`, 2026-08-06 — CI gate now OPEN, see "Sidecar format" below),
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

## The three checkpoints

| file (in this directory) | role in the manifest | upstream | upstream license | `LicenseClass` | `vokra-cli convert --model` |
|---|---|---|---|---|---|
| `sbv2-v2-multilingual-base.gguf` | `checkpoint.sbv2_main` | [`litagin02/style_bert_vits2`](https://huggingface.co/litagin02) family (v2, JA+EN multilingual base) | AGPL-3.0 | `Copyleft` | `sbv2` |
| `deberta-v2-large-japanese-char-wwm.gguf` | `checkpoint.bert_ja` | [`ku-nlp/deberta-v2-large-japanese-char-wwm`](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm) | cc-by-sa-4.0 | `Copyleft` (ShareAlike variant) | `deberta-v2` |
| `deberta-v3-large.gguf` | `checkpoint.bert_en` | [`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large) | MIT | `Permissive` | `deberta-v3` |

Source: `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §9 "配布 gate
拡張 / SKU" table (gitignore-local design doc; the license/SKU facts are
reproduced here because this README ships in the public repo) and the
`DEFAULT_BERT_JA_REPO` / `DEFAULT_BERT_EN_REPO` constants in
`tools/parity/sbv2_dump_reference.py` (verified live against the HF Hub API
by that script's own module doc, 2026-07-26).

**None of these three checkpoints is Apache-2.0.** Vokra's own *code* here
is clean-room Apache-2.0 (see "Clean-room reminder" below) — what varies is
the *weight* license, redistributed as-is with attribution, per this
project's `LicenseClass::Copyleft` distribution path (`docs/license-audit.md`
§3.1; the sign-off rows for these three SKUs are intentionally still blank —
see "What this README does NOT do" below).

## Directory layout (what's committed vs. gitignored)

```
tests/fixtures/sbv2/
├── README.md                                        # committed (this file)
├── reference_dump.manifest.json                     # committed (schema template)
├── sbv2-v2-multilingual-base.gguf.sha256             # committed (real SHA256, `6580061`)
├── deberta-v2-large-japanese-char-wwm.gguf.sha256    # committed (real SHA256, `6580061`)
├── deberta-v3-large.gguf.sha256                      # committed (real SHA256, `6580061`)
│
├── sbv2-v2-multilingual-base.gguf                    # gitignored — produced locally
├── deberta-v2-large-japanese-char-wwm.gguf           # gitignored — produced locally
├── deberta-v3-large.gguf                             # gitignored — produced locally
├── *.safetensors                                     # gitignored — intermediate downloads, if staged here
└── reference_dump/                                   # gitignored — Task 30 dumper output
    ├── phoneme_embed.bin
    ├── text_hidden.bin
    ├── ... (11 tensors total, see "About the committed manifest")
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

All three steps below are **owner-only** (design doc §12 "依頼者残タスク"):
they need real network access to gated/large upstream repos, and — for the
SBV2 main model's reference dump specifically — a vendoring step
(`tools/parity/vendor/vits/`) that has not landed yet (see "Known
limitations" below). Nothing here can be run unattended in CI without a
real checkpoint already staged.

### 1. Download the checkpoints

**SBV2 v2 base** — use the dedicated prep tool, which downloads via
`huggingface_hub.snapshot_download` and best-effort maps the upstream
`config.json` onto the `vokra.sbv2.*` hparam schema:

```bash
python3 tools/parity/sbv2_prepare_checkpoint.py \
    --hf-repo litagin02/style_bert_vits2 \
    --output-dir /tmp/sbv2-checkpoint
```

This prints a `RESOLVED`/`UNRESOLVED` report for each of the 22 required
`vokra.sbv2.*` fields and writes `/tmp/sbv2-checkpoint/vokra-sbv2-config.json`
— read that report before continuing; an incomplete config makes
`vokra-cli convert --model sbv2` fail loudly at the first missing key
rather than silently guessing (see that tool's own module doc, "CONFIDENCE"
section, for exactly which fields are read/derived/defaulted/left
unresolved). **Multi-shard checkpoints are reported, not merged** — pick
the right `.safetensors` shard yourself if the download has more than one.

**DeBERTa v2 (JA) / v3 (EN)** — these are plain HF `transformers`-style
repos (flat `.safetensors`, no `.pth` to flatten), so a direct
`huggingface_hub.snapshot_download` call is enough; there is no dedicated
prep tool for them yet. For example:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ku-nlp/deberta-v2-large-japanese-char-wwm",
    repo_type="model",
    local_dir="/tmp/deberta-v2-ja",
    allow_patterns=["*.safetensors", "*.json"],
)
# and the EN encoder:
snapshot_download(
    repo_id="microsoft/deberta-v3-large",
    repo_type="model",
    local_dir="/tmp/deberta-v3-en",
    allow_patterns=["*.safetensors", "*.json"],
)
```

(This mirrors `tools/parity/sbv2_prepare_checkpoint.py`'s own
`download_checkpoint()` — same function, same `allow_patterns`. No
`token=` argument: `huggingface_hub` resolves `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` from the environment or a cached login on its
own — never pass a token via argv, which can leak through `ps` / shell
history.)

### 2. Generate the reference dump (Task 30, `sbv2_dump_reference.py`)

```bash
python3 tools/parity/sbv2_dump_reference.py \
    --checkpoint /tmp/sbv2-checkpoint \
    --output-dir tests/fixtures/sbv2 \
    --text "こんにちは。" --language ja \
    --do-dump
```

**As of this commit, `--do-dump` always fails loudly** at one of three
tiers (see that script's own module doc for the full explanation):
missing `torch`, missing `transformers`, or — the tier every environment
with both installed will actually hit — `tools/parity/vendor/vits/`
shipping only a `LICENSE` + scaffold `README.md`, no vendored
`jaywalnut310/vits` (MIT) module yet. **A follow-up must vendor that module
before this step can produce a real dump** (see that README's "What a
follow-up vendoring pass should add here" table). This is deliberate
scaffolding, not a bug: fabricating a tensor dump without the real
permissive reference implementation would validate nothing (see
`tools/parity/utmos_dump_reference.py`'s own module doc, and memory
`feedback-honest-parity-atol`).

The JA/EN DeBERTa encoders do **not** share this gate — they are loaded
directly via HF `transformers`' `AutoModel`, a real, `pip install`-able
dependency, with no vendoring involved:

```bash
python3 tools/parity/deberta_v2_dump_reference.py \
    --hf-repo ku-nlp/deberta-v2-large-japanese-char-wwm \
    --output-dir /tmp/deberta-v2-dump --do-dump
python3 tools/parity/deberta_v3_dump_reference.py \
    --hf-repo microsoft/deberta-v3-large \
    --output-dir /tmp/deberta-v3-dump --do-dump
```

These two are useful for isolating BERT-only numeric parity independently
of the still-vendoring-blocked SBV2 pipeline, but their output is **not**
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
    --output tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf

vokra-cli convert --model deberta-v3 \
    --input /tmp/deberta-v3-en/<downloaded>.safetensors \
    --output tests/fixtures/sbv2/deberta-v3-large.gguf
```

(`--model sbv2` / `deberta-v2` / `deberta-v3` are the canonical
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
```

Each command **overwrites** that file's current contents with a fresh
`sha256sum` line — a single `<hash>  <path>` line. Only these three
`.sha256` files (plus this README and the manifest) get committed; the
`.gguf` / `.safetensors` themselves stay local (`.gitignore`d). The
sidecars currently on disk (post-`6580061`) already carry real hashes;
this step is only needed after regenerating a GGUF (e.g. after a
converter bump changes what tensors get emitted — see Wave-4
CONVERTER-EMIT-EXPLICIT-ZEROS).

### 5. Run the gated tests

```bash
cargo test -p vokra-models --test sbv2_gguf_loader -- --ignored
cargo test -p vokra-bert --test deberta_v2_loader -- --ignored
cargo test -p vokra-models --test parity_sbv2_real -- --ignored
```

A missing fixture at this point is a **loud, actionable panic** naming
exactly what's absent — never a silent skip-and-pass (FR-EX-08).

## About the committed `reference_dump.manifest.json`

The committed manifest is **not a real dump** — it is the exact,
byte-for-byte stdout of `sbv2_dump_reference.py`'s **schema-preview mode**
(the default when `--do-dump` is omitted), which needs no checkpoint, no
`torch`, and no `transformers` to run:

```bash
python3 tools/parity/sbv2_dump_reference.py \
    --checkpoint /tmp/unused --output-dir /tmp/unused --language ja
```

That mode prints the manifest this tool *would* write, with every
`request.*` / `checkpoint.*` field fully resolved from the CLI defaults,
and all 11 `tensors[].shape` entries left as **symbolic placeholders**
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

The 11 dumped tensors (design doc §10 / `parity_sbv2_real.rs` module doc —
today only `waveform` is actually diffed by the Rust test; the rest are
carried in the manifest/dump for a documented Task 28.x follow-up that adds
per-stage `SbV2Model` accessors):

| tensor | shape | purpose |
|---|---|---|
| `phoneme_embed` | `[T_text, 192]` | text encoder input |
| `text_hidden` | `[T_text, 192]` | text encoder output |
| `bert_hidden_ja` | `[T_bert, 1024]` | DeBERTa v2 output (JA) |
| `bert_hidden_en` | `[T_bert, 1024]` | DeBERTa v3 output (EN) |
| `bert_bridge_out` | `[T_text, 192]` | BERT bridge conv output |
| `speaker_embed` | `[1, 512]` | speaker embedding |
| `style_projected` | `[1, 192]` | style vector projection |
| `sdp_sample` | `[T_text]` | SDP duration sample |
| `mel_hidden` | `[T_mel, 192]` | length-regulated hidden |
| `z_latent` | `[T_mel, 192]` | normalizing-flow output |
| `waveform` | `[1, samples]` | final PCM (the tensor `parity_sbv2_real.rs` compares) |

## Sidecar format (the three `.gguf.sha256` files)

**Status (2026-08-09)**: all three sidecars now carry **real
`sha256sum`-format** lines (committed `6580061`, "feat(fixtures/sbv2):
commit real sha256 sidecars — opens parity-sbv2-real CI gate"). The
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

1. **`sbv2_dump_reference.py --do-dump` is gated on VITS vendoring that has
   not landed.** See step 2 above — the reference dump cannot actually
   produce `reference_dump/*.bin` today; `tools/parity/vendor/vits/`
   currently ships only a scaffold (`LICENSE` + `README.md`), no code.

2. **The SBV2 v2 / DeBERTa v2 / DeBERTa v3 converters do not yet rename
   tensors.** `crates/vokra-convert/src/models/{sbv2,deberta_v2,deberta_v3}.rs`
   each carry a `TODO(owner): tensor name mapping` module-doc section
   stating that every tensor is emitted under its **upstream checkpoint
   name, verbatim** — none of the three converters yet renames e.g. an
   upstream `dec.ups.0.weight` to whatever `SbV2Model::from_gguf` /
   `DebertaV2Encoder::from_gguf` / `DebertaV3Encoder::from_gguf` actually
   look up. Those module docs describe a GGUF produced today as a
   "provenance-correct, byte-faithful **staging artifact**, not yet
   loadable by `from_gguf`". Building the real rename table needs a real
   checkpoint's tensor names in hand (which is exactly what this fixture
   set, once populated, would supply) — it is a distinct follow-up task,
   not something Task 34's fixture scaffolding does.

3. **Even with real, correctly-named fixtures, `SbV2Model::synthesize`
   cannot succeed from this crate today.** `SbV2Model::from_gguf`'s own doc
   ("G2P is not loaded here") is explicit: the 3-file loader signature has
   no piper-plus G2P GGUF, so every `synthesize` call on a `from_gguf`
   -loaded model returns `VokraError::NotImplemented` at the G2P step. A
   real, working G2P needs `vokra_piper_plus::Phonemizer`, which lives
   outside `vokra-models`' zero-dependency root workspace
   (`integrations/vokra-piper-g2p`) — `crates/vokra-models` cannot depend
   on it (NFR-DS-02). `parity_sbv2_real.rs`'s own test already treats this
   specific, already-documented `NotImplemented` as an honestly-logged
   non-failure, not a fabricated pass — see that file's module doc.

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
- It does not populate a real checkpoint, run a real conversion, or
  compute a real hash — everything here is the reproducible scaffold an
  owner needs to do that themselves (design doc §12 "依頼者残タスク").
- It does not sign off on `docs/license-audit.md` §3.1 for these three
  SKUs — that stays blank (fail-closed default, memory
  `feedback-license-signoff-primary-source`) until an owner confirms the
  license facts against a real, in-hand checkpoint.
