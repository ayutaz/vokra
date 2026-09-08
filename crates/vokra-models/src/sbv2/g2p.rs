//! SBV2 G2P wrapper: piper-plus 8-language G2P を SBV2 phoneme table に mapping.
//! (Clean-room comment: see mod.rs)

use crate::strict_checkpoint::sha256_bytes;
use std::collections::HashMap;
use vokra_core::{Result, VokraError};
use vokra_piper_plus::Phonemizer;

/// Fixed SBV2 v2 language-row and global-tone facts from the authenticated
/// upstream `text/symbols.py`/`text/__init__.py` sources. Piper mappings carry
/// language-local raw tone ids; production bridges convert them to the global
/// `tone_embed` rows used by the checkpoint.
/// First global tone row reserved for ZH (raw tones 0 through 5).
pub const SBV2_ZH_TONE_START: u8 = 0;
/// First global tone row reserved for JP/JA (raw tones 0 through 1).
pub const SBV2_JA_TONE_START: u8 = 6;
/// First global tone row reserved for EN (raw tones 0 through 3).
pub const SBV2_EN_TONE_START: u8 = 8;
/// Number of global tone rows reserved for ZH.
pub const SBV2_ZH_TONE_COUNT: u8 = 6;
/// Number of global tone rows reserved for JP/JA.
pub const SBV2_JA_TONE_COUNT: u8 = 2;
/// Number of global tone rows reserved for EN.
pub const SBV2_EN_TONE_COUNT: u8 = 4;
/// Total number of global tone rows in the authenticated SBV2 checkpoint.
pub const SBV2_N_TONES: usize = 12;
/// Exact phone-vocabulary size in the authenticated JP-Extra checkpoint.
pub const SBV2_JP_EXTRA_N_VOCAB: usize = 112;
/// SHA-256 of the ordered JP-Extra symbols, canonicalized as each UTF-8
/// symbol followed by one NUL byte. This was computed from the exact ordered
/// `phoneme_symbols` array in the exact-head VAST evidence recorded in
/// `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` (SBV2
/// evidence SHA-256
/// `770e5481d0335a22b52ca511adcab2fa8a9231a74c6db49d8f60e0c6a3c4ea8f`).
pub const SBV2_JP_EXTRA_SYMBOLS_CANONICAL_SHA256: [u8; 32] = [
    0x7e, 0x1f, 0x45, 0x66, 0x31, 0x0f, 0x17, 0xb6, 0x19, 0x6d, 0x4b, 0x51, 0x30, 0x0c, 0x7e, 0x76,
    0x0d, 0x5c, 0x5f, 0xe6, 0x0e, 0xf4, 0xdd, 0x49, 0xe0, 0xb7, 0xe1, 0xd4, 0x77, 0x74, 0x09, 0x60,
];

/// Authenticated live model identity for the production Japanese route.
pub const SBV2_JP_EXTRA_MODEL_NAME: &str = "sbv2-v2-jp-extra-base";
/// Authenticated upstream repository for the production Japanese route.
pub const SBV2_JP_EXTRA_UPSTREAM_REPOSITORY: &str = "litagin/Style-Bert-VITS2-2.0-base-JP-Extra";
/// Immutable HF revision used by the production JP-Extra contract.
pub const SBV2_JP_EXTRA_HF_REVISION: &str = "a731761009f3c96d104487be6ad332bf1bb5a3a5";
/// Immutable upstream source commit used by the production JP-Extra contract.
pub const SBV2_JP_EXTRA_SOURCE_COMMIT: &str = "ef93f388fc1ddf0dc0f598126c1964923f1df94f";
/// Git blob identity for the upstream phoneme/tone symbol table.
pub const SBV2_JP_EXTRA_SYMBOLS_BLOB: &str = "846de64584e9ba4b8d96aab36d4efbcefb1a11e7";
/// Git blob identity for the upstream Japanese G2P implementation.
pub const SBV2_JP_EXTRA_JAPANESE_BLOB: &str = "5c055875626c16bd7d3489d02b4952ec90a3bbf6";
/// Git blob identity for the upstream Japanese mora table.
pub const SBV2_JP_EXTRA_MORA_BLOB: &str = "b43e54d8d8297cf1eac0e3e3f0eef6b4f1c24fa3";
/// Git blob identity for the upstream shared sequence mapper.
pub const SBV2_JP_EXTRA_SEQUENCE_BLOB: &str = "495e57b50d87a4ca3e8fe8dbaf003b4888581927";

/// SBV2 input language selector — drives which char-level mapping table
/// (and which tone convention) `SbV2Phonemizer::phonemize` uses, and
/// selects which row of
/// [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s
/// `language_embed` table
/// ([`super::text_encoder::N_LANGUAGES`] = 3: ZH/JA/EN) is broadcast-added
/// to every position.
///
/// `Hash` derives are required so [`Language`] can key a
/// [`PhonemizeFixture`]'s internal `HashMap<(Language, String), _>` (Task 7);
/// the other derives predate that use.
///
/// # `ZH` scope note (M6, 2026-08-06)
///
/// The real SBV2 v2 base checkpoint
/// (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`) ships a 3-row
/// `enc_p.language_emb.weight` table (ZH/JP/EN), so this enum must expose
/// all three variants for [`super::text_encoder::SbV2TextEncoder::forward`]'s
/// `language_id` argument to be constructible for a real ZH request. The
/// authenticated upstream row order is ZH=0, JP=1, EN=2; the Rust `JA` name
/// denotes that upstream JP row.
/// **A production ZH G2P is not implemented in this crate** — Vokra's ZH
/// G2P is out of scope for the M6 SBV2 v2 land. Selecting `ZH` at
/// [`SbV2Phonemizer::phonemize`] currently returns a loud
/// [`VokraError::NotImplemented`] (never a silent JA fallback — FR-EX-08);
/// the ZH variant exists so future ZH G2P work can plug in without a
/// second breaking enum change, and so a caller who has ZH phoneme ids
/// from another source can still hit the `language_id = 0` code path via
/// [`SbV2Phonemizer::from_fixture`] or by constructing a
/// [`PhonemizeResult`] directly.
///
/// # ZH code-side status (WP-21 doc sweep, 2026-08-10)
///
/// **Forward pointer — additive to the M6 scope note above (which still
/// describes the runtime `phonemize()` ZH arm's fail-closed state)**. The
/// SBV2 v2 language-embed table's row-0 dispatch (`Language::ZH` →
/// `language_id() = 0` → [`super::text_encoder::SbV2TextEncoder`]'s
/// `language_embed` row 0) is downstream-ready as of WP-16 (three-row
/// table landed) and exercised in synthetic-parity tests. What is still
/// open is the phonemizer wiring itself.
///
/// **Owner decision (2026-08-09)**: **ZH G2P = piper-plus reuse**. The
/// future ZH G2P route bridges the existing 8-language piper-plus G2P
/// via the excluded-workspace `integrations/vokra-piper-g2p` crate
/// (which already covers `zh` via `PassthroughPhonemizer` — see that
/// crate's `README.md`), so [`SbV2Phonemizer::from_piper_g2p`] can be
/// extended to accept a ZH-capable phonemizer without an enum change.
/// The pairing ZH BERT decision the same day is
/// `hfl/chinese-roberta-wwm-ext-large` (Apache-2.0, standard
/// `BertForMaskedLM` — NOT DeBERTa; see [`crate::sbv2`]'s module-level
/// "ZH code-side status" doc for the encoder-side gap-fill plan).
///
/// **Publish is deferred** per "モデルは公開しない" — §3.1 sign-off
/// blank remains fail-closed default until owner CI fixture regeneration
/// (WP-20) and license sign-off. WP-21 is docs-only sweep, not gap-fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// Japanese input (SBV2 raw pitch-accent tones, 0-1; global rows 6-7).
    /// Maps to
    /// [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s
    /// `language_embed` row 1.
    JA,
    /// English input (global tone band 8-11; SBV2 has no EN pitch accent).
    /// Maps to row 2.
    EN,
    /// Simplified Chinese input (SBV2 v2 language embed table id = 0 —
    /// downstream-ready per WP-17). Owner decision 2026-08-09: the ZH G2P
    /// path reuses piper-plus's own 8-language phonemizer via the
    /// excluded-workspace `integrations/vokra-piper-g2p` crate (which
    /// itself carries `piper-plus-g2p` — currently a passthrough for ZH
    /// per that crate's own README; a real ZH G2P dict is a piper-plus-side
    /// upgrade). WP-18 lands only the trait boundary + delegation inside
    /// `vokra-models`, gated by
    /// [`SbV2Phonemizer::with_zh_g2p`](SbV2Phonemizer::with_zh_g2p); WP-19
    /// wires the concrete bridge. **Fail-closed** at the G2P dispatch:
    /// calling `phonemize(_, Language::ZH)` on a phonemizer without a ZH
    /// G2P returns [`VokraError::NotImplemented`] (never a silent
    /// fall-through to a synthetic char-map — ZH deliberately has none, so
    /// a wiring miss cannot be hidden by "looks like it worked" output).
    /// Raw tones are Mandarin lexical tones 0-5 and already occupy global
    /// tone rows 0-5.
    ZH,
}

impl Language {
    /// Returns the row index into
    /// [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s
    /// `language_embed` table that this language selects. The fixed upstream
    /// `language_id_map` is `{ZH: 0, JP: 1, EN: 2}`; `JA` is Vokra's spelling
    /// for the upstream JP row.
    ///
    /// The type is `u8` because the downstream text-encoder forward takes
    /// a `u8` `language_id`; 3 fits comfortably.
    pub fn language_id(self) -> u8 {
        match self {
            Language::ZH => 0,
            Language::JA => 1,
            Language::EN => 2,
        }
    }
}

/// Policy for out-of-vocabulary (OOV) inputs the char-level maps or the
/// piper→SBV2 id maps do not cover (WP-14).
///
/// The four OOV lookup sites in this module —
/// [`SbV2Phonemizer::phonemize_ja_via_piper`],
/// [`SbV2Phonemizer::phonemize_ja_char_mapping`],
/// [`SbV2Phonemizer::phonemize_en_via_piper`] and
/// [`SbV2Phonemizer::phonemize_en_char_mapping`] — historically fell back
/// silently to `sbv2_default_phoneme_id` (the language's global default
/// tone) for any input the
/// active mapping did not cover, producing byte-valid but wrong TTS audio
/// with no signal to the caller. FR-EX-08 forbids that class of silent
/// failure, so every [`SbV2Phonemizer`] construction path now defaults to
/// [`OovPolicy::Strict`] and callers that genuinely want the pre-WP-14
/// behavior (test fixtures, throwaway experiments) must opt in explicitly
/// via [`SbV2Phonemizer::with_oov_policy`].
///
/// The [`Default`] impl is [`OovPolicy::Strict`] — a drift here would
/// silently flip every future constructor that reads the default, so the
/// `wp14_default_policy_is_strict` test in
/// `crates/vokra-models/tests/sbv2_g2p.rs` pins it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OovPolicy {
    /// Any OOV input aborts phonemization with
    /// [`VokraError::InvalidArgument`] naming the offending char (or piper
    /// phoneme id) and its position in the input sequence — the FR-EX-08
    /// fail-closed default.
    #[default]
    Strict,
    /// Any OOV input is silently replaced by `sbv2_default_phoneme_id`
    /// (global tone row `6` on the real JA piper path; synthetic test paths
    /// retain their local fixture row `0`). Matches the pre-WP-14 behavior. Opt in
    /// explicitly with [`SbV2Phonemizer::with_oov_policy`]; never the
    /// default, per FR-EX-08.
    Lenient,
}

/// The G2P output for one input string: phoneme ids in SBV2 vocabulary
/// space, per-phoneme pitch-accent tones, per-phoneme word-boundary flags,
/// and the original text (fed separately to the BERT bridge).
///
/// `Clone` (added for Task 7) is required so a [`PhonemizeFixture`] can hand
/// out an owned copy of the pre-computed result at every
/// [`SbV2Phonemizer::phonemize`] call without moving out of the fixture's
/// internal map — the fixture must outlive the single [`SbV2Phonemizer`]
/// instance it was built into (so successive lookups against the same key
/// remain valid), which forbids returning a `&PhonemizeResult` from
/// [`SbV2Phonemizer::phonemize`]'s existing `-> Result<PhonemizeResult>`
/// signature. Every field is a small owned-`Vec`/`String`, so `.clone()` is
/// a straightforward per-element copy; the piper- and synthetic-mapping paths
/// already build fresh `Vec`s per call, so this is a peer of that per-call
/// cost, not a new one.
#[derive(Debug, Clone)]
pub struct PhonemizeResult {
    /// Phoneme ids in SBV2 phoneme-table space (one per output phoneme).
    pub phoneme_ids: Vec<u16>,
    /// Global tone-embedding row per phoneme. Production G2P paths emit
    /// JP=6 through 7, ZH=0 through 5, and EN=8 through 11; synthetic test mappings may use their
    /// own deliberately small local fixture vocabulary.
    pub tones: Vec<u8>,
    /// Word-boundary flag per phoneme (true = first phoneme of a word).
    pub word_boundaries: Vec<bool>,
    /// Text passed to the BERT bridge (Task 16+). Native JP providers supply
    /// their returned normalized text; other paths preserve the request text.
    pub bert_input_text: String,
}

/// The authenticated source identities required by the JP-Extra production
/// G2P route.  These are intentionally data-only: the AGPL upstream
/// implementation and its dictionary never cross into the runtime crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbV2JapaneseG2pSource {
    /// Authenticated JP-Extra model identifier.
    pub model_name: String,
    /// Upstream repository URL that owns the model and source contract.
    pub upstream_repository: String,
    /// Immutable Hugging Face revision for the model artifact.
    pub hf_revision: String,
    /// Immutable source commit for the native G2P implementation.
    pub source_commit: String,
    /// Git blob identity for the ordered phone-symbol table.
    pub symbols_blob: String,
    /// Git blob identity for the Japanese lexicon resource.
    pub japanese_blob: String,
    /// Git blob identity for the mora resource.
    pub mora_blob: String,
    /// Git blob identity for the sequence resource.
    pub sequence_blob: String,
}

impl SbV2JapaneseG2pSource {
    /// Returns the source identity pinned by the JP-Extra audit.
    pub fn authenticated() -> Self {
        Self {
            model_name: SBV2_JP_EXTRA_MODEL_NAME.to_owned(),
            upstream_repository: SBV2_JP_EXTRA_UPSTREAM_REPOSITORY.to_owned(),
            hf_revision: SBV2_JP_EXTRA_HF_REVISION.to_owned(),
            source_commit: SBV2_JP_EXTRA_SOURCE_COMMIT.to_owned(),
            symbols_blob: SBV2_JP_EXTRA_SYMBOLS_BLOB.to_owned(),
            japanese_blob: SBV2_JP_EXTRA_JAPANESE_BLOB.to_owned(),
            mora_blob: SBV2_JP_EXTRA_MORA_BLOB.to_owned(),
            sequence_blob: SBV2_JP_EXTRA_SEQUENCE_BLOB.to_owned(),
        }
    }

    fn validate(&self) -> Result<()> {
        let expected = Self::authenticated();
        if self != &expected {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra Japanese G2P source identity is not the authenticated \
                 model/source contract; refusing to bind an unauthenticated frontend"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// A validated JP-Extra phone vocabulary and source contract.
///
/// The symbol vector is supplied by the independently audited sidecar.  The
/// runtime checks its exact dimensions, uniqueness and source identities, but
/// never embeds a copied upstream table or reconstructs one from a guessed
/// constant. The constructor authenticates the ordered symbol bytes against
/// the fixed digest recovered from VAST evidence. This keeps the mapping
/// usable by a native provider while retaining a fail-closed boundary for
/// stale or forged sidecars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbV2JapaneseG2pContract {
    source: SbV2JapaneseG2pSource,
    symbols: Vec<String>,
}

impl SbV2JapaneseG2pContract {
    /// Validates and binds the source identity and phone vocabulary.
    pub fn new(source: SbV2JapaneseG2pSource, symbols: Vec<String>) -> Result<Self> {
        source.validate()?;
        if symbols.len() != SBV2_JP_EXTRA_N_VOCAB {
            return Err(VokraError::InvalidArgument(format!(
                "SBV2 JP-Extra phone vocabulary has {} symbols; authenticated contract requires {}",
                symbols.len(),
                SBV2_JP_EXTRA_N_VOCAB
            )));
        }
        if symbols.iter().any(String::is_empty) {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra phone vocabulary contains an empty symbol".to_owned(),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(symbols.len());
        if symbols.iter().any(|symbol| !seen.insert(symbol)) {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra phone vocabulary contains duplicate symbols".to_owned(),
            ));
        }
        let canonical = canonical_symbol_bytes(&symbols);
        let actual_digest = sha256_bytes(&canonical);
        if actual_digest != SBV2_JP_EXTRA_SYMBOLS_CANONICAL_SHA256 {
            return Err(VokraError::InvalidArgument(format!(
                "SBV2 JP-Extra phone vocabulary digest mismatch: got {}, expected authenticated evidence digest {}",
                hex_digest(&actual_digest),
                hex_digest(&SBV2_JP_EXTRA_SYMBOLS_CANONICAL_SHA256)
            )));
        }
        Ok(Self { source, symbols })
    }

    /// Returns the bound source identities for diagnostics and evidence.
    pub fn source(&self) -> &SbV2JapaneseG2pSource {
        &self.source
    }

    /// Returns the source-ordered SBV2 phone vocabulary.
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }
}

/// Output of a native Japanese G2P provider before SBV2 vocabulary mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbV2JapaneseG2pOutput {
    /// Providerが返す正規化text。algorithm/parityはintegrationの責任で、この構造
    /// 検証は意味論を認証しない。runtimeはnon-emptyだけを確認してBERTへ渡す。
    pub normalized_text: String,
    /// Phones in the exact source vocabulary spelling.
    pub phones: Vec<String>,
    /// Japanese-local pitch-accent values (only 0 and 1 are authenticated).
    pub raw_tones: Vec<u8>,
    /// Source-aligned phone counts for the normalized utterance, including
    /// the boundary slots used by the authenticated JP-Extra frontend.
    pub word2ph: Vec<usize>,
}

/// Native provider seam for production Japanese G2P.
///
/// The implementation belongs to a dedicated native integration that proves
/// parity with the authenticated SBV2 JP-Extra source contract. The existing
/// piper-plus integration is intentionally not such an implementation: its
/// A1/A2/A3 prosody and token framing are a different model contract and it
/// does not expose SBV2 `word2ph`. This crate owns only the zero-dependency
/// SBV2 contract binding and conversion. Python, eSpeak, pyopenjtalk and
/// dictionary downloads cannot implement this trait inside the runtime.
pub trait SbV2JapaneseG2pProvider: Send + Sync {
    /// Phonemizes one normalized utterance into source-aligned JP-Extra data.
    fn phonemize(&self, text: &str) -> Result<SbV2JapaneseG2pOutput>;
}

/// A pre-computed G2P output table for [`SbV2Phonemizer::from_fixture`]
/// (Task 7) — the `(language, text)`-keyed lookup that lets a real,
/// [`SbV2Model::from_gguf`](crate::sbv2::SbV2Model::from_gguf_with_phonemizer)-loaded
/// model be exercised for a fixed set of test sentences without depending on
/// a real 8-language piper-plus G2P instance.
///
/// # Scope: fixture-driven parity, not a G2P
///
/// A [`PhonemizeFixture`]-backed phonemizer is **not** a G2P — every
/// [`SbV2Phonemizer::phonemize`] call is a plain `(language, text)` lookup,
/// so any call whose `(language, text)` pair is absent from `entries`
/// fails loudly with [`VokraError::InvalidArgument`] (FR-EX-08), never
/// silently falls through to
/// [`SbV2Phonemizer::synthetic_for_test`]'s toy char-mapping. This is
/// deliberate: the whole point of the fixture path is to reproduce the
/// exact ids a permissive Python reference dumper (Task 30's
/// `tools/parity/sbv2_dump_reference.py`) fed the reference forward pass,
/// so `SbV2Model::synthesize` compares against that dumper's own
/// intermediate tensors down the pipeline (Task 28's
/// `crates/vokra-models/tests/parity_sbv2_real.rs`) — falling back to a
/// different G2P for a miss would validate nothing.
///
/// Populating a real production JP-Extra G2P is instead the job of a dedicated
/// implementation of [`SbV2JapaneseG2pProvider`], attached through
/// [`SbV2Phonemizer::from_authenticated_japanese_g2p`]. The generic
/// [`SbV2Phonemizer::from_piper_g2p`] route is a different voice contract and
/// must not be treated as a JP-Extra adapter.
#[derive(Debug, Clone, Default)]
pub struct PhonemizeFixture {
    // (language, text) -> the pre-computed [`PhonemizeResult`] the Python
    // reference dumper fed the reference forward pass for that exact input.
    // Owned `String` (not a borrow) because the fixture outlives every input
    // string it was populated from (the dumper's transient argv), and because
    // the [`SbV2Phonemizer::phonemize`] signature takes `&str`, forcing a
    // lookup against an owned-`String` map key rather than a `&str`-keyed one.
    entries: HashMap<(Language, String), PhonemizeResult>,
}

impl PhonemizeFixture {
    /// Constructs an empty fixture. Populate it with [`insert`](Self::insert)
    /// before handing it to [`SbV2Phonemizer::from_fixture`]; an empty
    /// fixture is valid but every lookup will fail (FR-EX-08).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one pre-computed [`PhonemizeResult`] to the fixture, keyed on
    /// `(language, text)`. A later [`SbV2Phonemizer::phonemize`] call with
    /// exactly the same `(language, text)` pair returns a `Clone` of this
    /// result.
    ///
    /// Overwrites any prior entry for the same `(language, text)` key
    /// (returns the replaced value if present, `None` otherwise) — the same
    /// convention `HashMap::insert` uses; the fixture is a plain wrapper.
    pub fn insert(
        &mut self,
        language: Language,
        text: impl Into<String>,
        result: PhonemizeResult,
    ) -> Option<PhonemizeResult> {
        self.entries.insert((language, text.into()), result)
    }

    /// Number of `(language, text)` entries populated. A fixture with `0`
    /// entries is valid but every lookup will fail — see [`Self::new`].
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff no `(language, text)` entries have been inserted. See
    /// [`Self::len`].
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // Internal lookup consumed by [`SbV2Phonemizer::phonemize`]'s fixture
    // arm. Not `pub`: the only supported way to reach a fixture entry is
    // through the [`SbV2Phonemizer::phonemize`] surface (which then hits this
    // fn), so a call site that already has a [`PhonemizeFixture`] handle
    // cannot bypass the phonemizer.
    fn lookup(&self, language: Language, text: &str) -> Result<PhonemizeResult> {
        // Two-step probe: `HashMap::get` needs an owned key when the map's
        // key is `(K1, K2)` and we have `(K1, &K2)`, so borrow via a
        // language-scoped iterator instead of allocating an owned
        // `(Language, String)` per lookup. `entries.len()` is O(1); the scan
        // is O(n_entries_for_language), which is the intended contract
        // (fixtures are keyed on exact text; a full G2P would be
        // [`SbV2Phonemizer::from_piper_g2p`] instead).
        for ((lang, txt), result) in &self.entries {
            if *lang == language && txt == text {
                return Ok(result.clone());
            }
        }
        Err(VokraError::InvalidArgument(format!(
            "SbV2Phonemizer::from_fixture: no fixture entry for (language={language:?}, \
             text={text:?}). The fixture is a fixed-set lookup (not a G2P) and every miss is \
             a loud failure per FR-EX-08 — populate the fixture via PhonemizeFixture::insert \
             at construction. For a generic Piper voice, use \
             SbV2Phonemizer::from_piper_g2p; JP-Extra requires the audited \
             native provider seam, which is not bundled in this crate."
        )))
    }
}

/// SBV2 grapheme-to-phoneme wrapper: maps input text to the SBV2 phoneme
/// vocabulary (ids, tones, word boundaries) for the JA, EN and ZH language
/// families.
///
/// Four construction paths select the routing strategy — checked in
/// priority order by [`phonemize`](Self::phonemize):
///
/// 1. [`from_fixture`](Self::from_fixture) (Task 7) is checked **first**:
///    every lookup goes through a pre-computed `(language, text)` table
///    ([`PhonemizeFixture`]) that reproduces the exact ids a Python
///    reference dumper fed the reference forward pass; the piper-plus and
///    synthetic paths below are never consulted while a fixture is
///    installed. A miss is a loud [`VokraError::InvalidArgument`], never a
///    silent fall-through to the other paths (FR-EX-08). Covers all three
///    languages (ZH / JP / EN).
/// 2. [`from_authenticated_japanese_g2p`](Self::from_authenticated_japanese_g2p)
///    wires an isolated native Japanese provider to the audited JP-Extra
///    source vocabulary. Its output is checked and mapped here; the provider
///    must perform normalization and Japanese G2P outside this zero-dependency
///    runtime crate. This route is checked before the generic piper-id bridge.
/// 3. [`from_piper_g2p`](Self::from_piper_g2p) (Task 15) wires the real
///    piper-plus [`Phonemizer`] reuse boundary (M1-01-A,
///    `docs/piper-plus-integration.md` §7): input text is phonemized by
///    the injected `ja_g2p` / `en_g2p`, and the resulting piper-plus
///    phoneme id sequence is routed into SBV2 phoneme-table space through
///    `ja_mapping` / `en_mapping`. The ZH G2P is an *optional* attachment
///    on top of this (see [`with_zh_g2p`](Self::with_zh_g2p), WP-18) —
///    kept off the 2-language `from_piper_g2p` signature for backward
///    compatibility with pre-WP-18 call sites.
/// 4. [`synthetic_for_test`](Self::synthetic_for_test) (Task 14) uses a
///    deterministic char-level mapping instead, so this crate's own tests
///    can prove the module wiring without depending on a real G2P
///    instance or a real SBV2 phoneme table. Only JA and EN have a
///    synthetic char-map; ZH deliberately does not (see
///    [`Language::ZH`]'s doc's fail-closed rationale).
pub struct SbV2Phonemizer {
    // Task 7 fixture path — see [`from_fixture`](Self::from_fixture) and the
    // struct doc's "priority order" note. Checked FIRST by
    // [`phonemize`](Self::phonemize); a `Some(_)` here disables the
    // piper-plus and synthetic paths entirely (a miss inside the fixture is
    // a loud FR-EX-08 error, not a fall-through).
    fixtures: Option<PhonemizeFixture>,
    // Authenticated native JP route.  This is checked before the generic
    // piper-id bridge so a production caller cannot accidentally run a voice
    // vocabulary mapping against the SBV2 table.
    ja_native_g2p: Option<Box<dyn SbV2JapaneseG2pProvider>>,
    ja_native_symbols: HashMap<String, u16>,
    ja_native_contract: Option<SbV2JapaneseG2pContract>,
    // Real piper-plus G2P (M1-01-A reuse boundary), when wired via
    // `from_piper_g2p`. `None` for `synthetic_for_test()` builds, where
    // `phonemize_ja`/`phonemize_en` fall back to the `*_char_mapping` tables
    // below instead.
    ja_g2p: Option<Box<dyn Phonemizer>>,
    en_g2p: Option<Box<dyn Phonemizer>>,
    // WP-18: ZH real piper-plus G2P, wired via
    // [`with_zh_g2p`](Self::with_zh_g2p) (builder-style setter, chainable on
    // any of the three constructors below). `None` means ZH is not wired;
    // `phonemize_zh` then returns [`VokraError::NotImplemented`] rather than
    // falling through to a synthetic char-map (there is no ZH char-map on
    // purpose — see [`Language::ZH`]'s doc for the fail-closed rationale).
    zh_g2p: Option<Box<dyn Phonemizer>>,
    // Piper-plus phoneme id -> (SBV2 phoneme id, tone) / SBV2 phoneme id,
    // for the real-G2P path above. Keyed by `i64` (the piper-plus voice's
    // OWN phoneme id) rather than a phoneme *symbol* string:
    // `Phonemizer::phonemize` (`crates/vokra-piper-plus/src/phonemizer.rs`)
    // returns only an already BOS/PAD/EOS-framed `Vec<i64>` id sequence —
    // the trait boundary never hands the caller a phoneme symbol string to
    // key a mapping on. A piper-plus id absent from these maps falls back
    // to `sbv2_default_phoneme_id` (a documented mapping fallback, not a
    // silent no-op — FR-EX-08).
    ja_mapping: HashMap<i64, (u16, u8)>,
    // Piper-plus id -> (SBV2 phoneme id, English language-local raw tone).
    en_mapping: HashMap<i64, (u16, u8)>,
    // WP-18: ZH real-G2P mapping — same `(SBV2 phoneme id, tone)` shape as
    // `ja_mapping`, since Mandarin carries lexical tones (raw 0 through 5). Missing ids
    // fall back to `(sbv2_default_phoneme_id, 0)` — the same documented
    // mapping fallback JA uses.
    zh_mapping: HashMap<i64, (u16, u8)>,
    // Fallback SBV2 phoneme id for any input (piper-plus id or char) absent
    // from the active mapping table.
    sbv2_default_phoneme_id: u16,
    // Char -> (SBV2 phoneme id, tone) / SBV2 phoneme id, for the synthetic
    // path (`synthetic_for_test`). Renamed from `ja_mapping`/`en_mapping`
    // (Task 14) when Task 15 added the id-keyed real-G2P mapping above.
    // No `zh_char_mapping` on purpose — see [`Language::ZH`]'s doc.
    ja_char_mapping: HashMap<char, (u16, u8)>,
    en_char_mapping: HashMap<char, u16>,
    // WP-14: how the four OOV lookup sites behave when the active mapping
    // does not cover an input. Every constructor initialises this to
    // [`OovPolicy::Strict`] (FR-EX-08 fail-closed default); the legacy
    // silent-fallback path is only reachable via
    // [`SbV2Phonemizer::with_oov_policy`].
    oov_policy: OovPolicy,
}

impl SbV2Phonemizer {
    #[doc(hidden)]
    pub fn synthetic_for_test() -> Self {
        // 決定的な最小 mapping: JA hiragana → id 100+, EN letter → id 200+
        let mut ja = HashMap::new();
        for (i, c) in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんこんにちは"
            .chars()
            .enumerate()
        {
            ja.insert(c, (100 + i as u16, (i % 3) as u8));
        }
        let mut en = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz ".chars().enumerate() {
            en.insert(c, 200 + i as u16);
        }
        Self {
            fixtures: None,
            ja_native_g2p: None,
            ja_native_symbols: HashMap::new(),
            ja_native_contract: None,
            ja_g2p: None,
            en_g2p: None,
            zh_g2p: None,
            ja_mapping: HashMap::new(),
            en_mapping: HashMap::new(),
            zh_mapping: HashMap::new(),
            sbv2_default_phoneme_id: 0,
            ja_char_mapping: ja,
            en_char_mapping: en,
            oov_policy: OovPolicy::Strict, // WP-14 FR-EX-08 default
        }
    }

    /// Wires the real piper-plus G2P (M1-01-A reuse boundary,
    /// `docs/piper-plus-integration.md` §7): `ja_g2p` / `en_g2p` phonemize
    /// the input text, and their (already BOS/PAD/EOS-framed) piper-plus
    /// phoneme id sequence is routed into SBV2 phoneme-table space through
    /// `ja_mapping` / `en_mapping`.
    ///
    /// Each `ja_mapping` tone is a Japanese language-local raw value in
    /// raw values 0 through 1, and this bridge converts them to the
    /// authenticated global JP tone rows 6 through 7.
    /// Each `en_mapping` tone is an English language-local raw value in
    /// raw values 0 through 3, converted to global EN rows 8 through 11, because
    /// upstream English stress refinement emits raw tones 1/2/3, raw 3 for
    /// unstressed input, and raw 0 for punctuation/specials.
    ///
    /// A piper-plus id produced by `ja_g2p` / `en_g2p` that is absent from
    /// the corresponding mapping falls back to the default SBV2 phoneme id
    /// (`0`) and that language's global default tone — a documented mapping
    /// fallback, not a silent no-op. A `text`
    /// that `ja_g2p` / `en_g2p` itself cannot phonemize instead propagates
    /// that `Err` out of [`phonemize`](SbV2Phonemizer::phonemize): the
    /// real-G2P path never falls through to the synthetic char mapping
    /// (FR-EX-08).
    ///
    /// **ZH is not wired here**: this constructor's dispatch surface remains
    /// two-language (JA and EN; ZH is optional), while its mapping values
    /// carry the raw tone needed by the authenticated global-tone conversion.
    /// Chain [`with_zh_g2p`](Self::with_zh_g2p) on the returned value to
    /// enable [`Language::ZH`] dispatch; without that chained call
    /// `phonemize(_, Language::ZH)` fails loudly with
    /// [`VokraError::NotImplemented`].
    pub fn from_piper_g2p(
        ja_g2p: Box<dyn Phonemizer>,
        en_g2p: Box<dyn Phonemizer>,
        ja_mapping: HashMap<i64, (u16, u8)>,
        en_mapping: HashMap<i64, (u16, u8)>,
    ) -> Self {
        Self {
            fixtures: None,
            ja_native_g2p: None,
            ja_native_symbols: HashMap::new(),
            ja_native_contract: None,
            ja_g2p: Some(ja_g2p),
            en_g2p: Some(en_g2p),
            zh_g2p: None,
            ja_mapping,
            en_mapping,
            zh_mapping: HashMap::new(),
            sbv2_default_phoneme_id: 0,
            ja_char_mapping: HashMap::new(),
            en_char_mapping: HashMap::new(),
            oov_policy: OovPolicy::Strict, // WP-14 FR-EX-08 default
        }
    }

    /// Constructs a phonemizer with the authenticated native Japanese G2P
    /// seam and JP-Extra phone contract. This does not bundle a provider;
    /// end-to-end production G2P remains the integration layer's responsibility.
    ///
    /// The provider returns phones rather than pre-mapped ids.  This is
    /// deliberate: ids are owned by the SBV2 source vocabulary and are bound
    /// here, so a piper voice table or an arbitrary caller mapping cannot be
    /// silently applied to the JP-Extra checkpoint.  The contract constructor
    /// rejects stale source identities, wrong vocabulary dimensions and
    /// duplicate/empty symbols before any request is accepted.
    pub fn from_authenticated_japanese_g2p(
        provider: Box<dyn SbV2JapaneseG2pProvider>,
        contract: SbV2JapaneseG2pContract,
    ) -> Result<Self> {
        let mut symbol_to_id = HashMap::with_capacity(contract.symbols.len());
        for (id, symbol) in contract.symbols.iter().enumerate() {
            let id = u16::try_from(id).map_err(|_| {
                VokraError::InvalidArgument(
                    "SBV2 JP-Extra phone vocabulary id does not fit u16".to_owned(),
                )
            })?;
            symbol_to_id.insert(symbol.clone(), id);
        }
        let phonemizer = Self {
            fixtures: None,
            ja_native_g2p: Some(provider),
            ja_native_symbols: symbol_to_id,
            ja_native_contract: Some(contract),
            ja_g2p: None,
            en_g2p: None,
            zh_g2p: None,
            ja_mapping: HashMap::new(),
            en_mapping: HashMap::new(),
            zh_mapping: HashMap::new(),
            sbv2_default_phoneme_id: 0,
            ja_char_mapping: HashMap::new(),
            en_char_mapping: HashMap::new(),
            oov_policy: OovPolicy::Strict,
        };
        // Keep the complete contract alive, including source identities, so a
        // configured phonemizer remains auditable after construction.
        if !phonemizer.ja_native_symbols.contains_key("_") {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra phone vocabulary is missing the authenticated boundary symbol `_"
                    .to_owned(),
            ));
        }
        Ok(phonemizer)
    }

    /// Attaches the authenticated native Japanese route to an existing
    /// phonemizer (for example one also carrying an English provider).
    pub fn with_authenticated_japanese_g2p(
        mut self,
        provider: Box<dyn SbV2JapaneseG2pProvider>,
        contract: SbV2JapaneseG2pContract,
    ) -> Result<Self> {
        let mut symbol_to_id = HashMap::with_capacity(contract.symbols.len());
        for (id, symbol) in contract.symbols.iter().enumerate() {
            let id = u16::try_from(id).map_err(|_| {
                VokraError::InvalidArgument(
                    "SBV2 JP-Extra phone vocabulary id does not fit u16".to_owned(),
                )
            })?;
            symbol_to_id.insert(symbol.clone(), id);
        }
        if !symbol_to_id.contains_key("_") {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra phone vocabulary is missing the authenticated boundary symbol `_"
                    .to_owned(),
            ));
        }
        self.ja_native_g2p = Some(provider);
        self.ja_native_symbols = symbol_to_id;
        self.ja_native_contract = Some(contract);
        Ok(self)
    }

    /// Returns the authenticated native JP contract when this phonemizer has
    /// one wired. This is intended for diagnostics and evidence reporting;
    /// callers cannot replace the contract without rebuilding the route.
    pub fn authenticated_japanese_contract(&self) -> Option<&SbV2JapaneseG2pContract> {
        self.ja_native_contract.as_ref()
    }

    /// WP-18: attaches a real piper-plus `Phonemizer` for the [`Language::ZH`]
    /// dispatch path, plus the piper-plus-id-to-SBV2-phoneme-id mapping
    /// (Mandarin carries language-local raw tones 0 through 5, converted to
    /// global ZH rows 0 through 5; the mapping's value is
    /// `(sbv2_phoneme_id, tone)` — same shape as `ja_mapping`).
    ///
    /// Builder-style: returns `self` so it composes with any of the three
    /// non-fixture constructors —
    /// [`from_piper_g2p`](Self::from_piper_g2p) (production wiring; the
    /// pair the excluded-workspace `integrations/vokra-piper-g2p` bridge
    /// will call at construction, WP-19) or
    /// [`synthetic_for_test`](Self::synthetic_for_test) (for tests that
    /// need JA/EN char-map coverage AND a wired ZH G2P at the same
    /// [`SbV2Phonemizer`] instance). Attaching to a
    /// [`from_fixture`](Self::from_fixture)-built phonemizer is a no-op at
    /// dispatch: the fixture path is checked first (see
    /// [`phonemize`](Self::phonemize)'s priority-order doc), so the
    /// fixture's `(Language::ZH, text)` entries always win over `zh_g2p`.
    ///
    /// A piper-plus id produced by `zh_g2p` that is absent from
    /// `zh_mapping` falls back to `(sbv2_default_phoneme_id, 0)` — the
    /// documented global ZH start row (not a silent no-op, FR-EX-08).
    /// A `text` that `zh_g2p` itself cannot phonemize propagates that `Err`
    /// out of [`phonemize`](Self::phonemize).
    ///
    /// Overwrites any prior `with_zh_g2p`-installed pair (last call wins) —
    /// convention mirroring `HashMap::insert`.
    pub fn with_zh_g2p(
        mut self,
        zh_g2p: Box<dyn Phonemizer>,
        zh_mapping: HashMap<i64, (u16, u8)>,
    ) -> Self {
        self.zh_g2p = Some(zh_g2p);
        self.zh_mapping = zh_mapping;
        self
    }

    /// Wires a pre-computed [`PhonemizeFixture`] (Task 7) — a fixed-set
    /// `(language, text)` lookup that lets a real,
    /// [`SbV2Model::from_gguf`](crate::sbv2::SbV2Model::from_gguf) /
    /// [`from_gguf_with_phonemizer`](crate::sbv2::SbV2Model::from_gguf_with_phonemizer)-loaded
    /// model be exercised for a known-set of test sentences without needing
    /// a real 8-language piper-plus G2P instance.
    ///
    /// Every [`phonemize`](Self::phonemize) call goes through
    /// [`PhonemizeFixture`]'s `(language, text)` map (see the fixture's own
    /// doc); the piper-plus and synthetic-mapping construction paths
    /// ([`from_piper_g2p`](Self::from_piper_g2p) /
    /// [`synthetic_for_test`](Self::synthetic_for_test)) are never consulted
    /// while a fixture is installed. A `(language, text)` pair absent from
    /// the fixture is a loud [`VokraError::InvalidArgument`], never a
    /// silent fall-through to a different path (FR-EX-08). See the
    /// fixture's own doc for the "not a G2P, a fixed-set parity lookup"
    /// scope; see [`SbV2Model::from_gguf_with_phonemizer`](crate::sbv2::SbV2Model::from_gguf_with_phonemizer)
    /// for the concrete caller.
    #[doc(hidden)] // test/fixture-only, not a production G2P — see struct doc's priority-order note
    pub fn from_fixture(fixture: PhonemizeFixture) -> Self {
        Self {
            fixtures: Some(fixture),
            ja_native_g2p: None,
            ja_native_symbols: HashMap::new(),
            ja_native_contract: None,
            ja_g2p: None,
            en_g2p: None,
            zh_g2p: None,
            ja_mapping: HashMap::new(),
            en_mapping: HashMap::new(),
            zh_mapping: HashMap::new(),
            sbv2_default_phoneme_id: 0,
            ja_char_mapping: HashMap::new(),
            en_char_mapping: HashMap::new(),
            // The fixture path itself is already fail-closed on a miss
            // ([`PhonemizeFixture::lookup`]'s own loud
            // [`VokraError::InvalidArgument`]), so `oov_policy` is only
            // consulted if a caller also does `.with_oov_policy(Lenient)`
            // AND swaps the phonemizer out of fixture mode — an
            // unreachable combination on this construction path. Kept
            // Strict here to preserve the "every path defaults Strict"
            // FR-EX-08 invariant.
            oov_policy: OovPolicy::Strict,
        }
    }

    /// Overrides this phonemizer's [`OovPolicy`] (WP-14). Consumes and
    /// returns `self` so it composes with the three
    /// `synthetic_for_test`/`from_piper_g2p`/`from_fixture` construction
    /// paths without extra plumbing:
    ///
    /// ```no_run
    /// # use vokra_models::sbv2::{OovPolicy, SbV2Phonemizer};
    /// let lenient = SbV2Phonemizer::synthetic_for_test()
    ///     .with_oov_policy(OovPolicy::Lenient);
    /// ```
    ///
    /// The default is [`OovPolicy::Strict`] (FR-EX-08 fail-closed — see
    /// [`OovPolicy`]'s own doc for why); the only reason to call this with
    /// [`OovPolicy::Lenient`] is a test fixture or throwaway experiment
    /// that has already accepted the silent-fallback risk.
    pub fn with_oov_policy(mut self, policy: OovPolicy) -> Self {
        self.oov_policy = policy;
        self
    }

    /// Phonemize `text` under the given `language`, producing SBV2 phoneme
    /// ids, tones, word boundaries and the pass-through BERT input text.
    ///
    /// Dispatch priority (see the struct doc's "priority order" note):
    ///
    /// 1. If [`from_fixture`](Self::from_fixture) was used, `(language, text)`
    ///    must be present in the fixture; a miss is a loud
    ///    [`VokraError::InvalidArgument`] (FR-EX-08), not a fall-through.
    /// 2. Otherwise, if [`from_piper_g2p`](Self::from_piper_g2p) was used —
    ///    or, for [`Language::ZH`] specifically, if
    ///    [`with_zh_g2p`](Self::with_zh_g2p) was chained on top of any
    ///    non-fixture constructor — the injected [`Phonemizer`] runs and
    ///    its output is mapped into SBV2 phoneme-table space.
    /// 3. Otherwise (i.e.
    ///    [`synthetic_for_test`](Self::synthetic_for_test)), the deterministic
    ///    char-level mapping runs for JA / EN; [`Language::ZH`] instead
    ///    returns [`VokraError::NotImplemented`] (no synthetic ZH char-map,
    ///    fail-closed per [`Language::ZH`]'s doc).
    ///
    /// # Errors
    ///
    /// When wired via [`from_fixture`](Self::from_fixture), returns
    /// [`VokraError::InvalidArgument`] for any `(language, text)` pair the
    /// fixture does not contain. When wired via
    /// [`from_piper_g2p`](Self::from_piper_g2p) /
    /// [`with_zh_g2p`](Self::with_zh_g2p), propagates any error the
    /// injected piper-plus [`Phonemizer`] returns. On the
    /// [`synthetic_for_test`](Self::synthetic_for_test) path,
    /// [`Language::ZH`] returns [`VokraError::NotImplemented`]; JA and EN
    /// never fail.
    ///
    /// Under the default [`OovPolicy::Strict`] (WP-14, FR-EX-08 fail-closed
    /// — see [`OovPolicy`]'s own doc), any OOV input the active char /
    /// piper→SBV2 mapping does not cover is also
    /// [`VokraError::InvalidArgument`] naming the offending char (or piper
    /// phoneme id) and its position; under
    /// [`OovPolicy::Lenient`](OovPolicy::Lenient) (opt-in via
    /// [`with_oov_policy`](Self::with_oov_policy)) the same OOV input is
    /// silently replaced by `sbv2_default_phoneme_id` (tone `0`) and this
    /// path never fails.
    pub fn phonemize(&self, text: &str, language: Language) -> Result<PhonemizeResult> {
        if let Some(fixture) = &self.fixtures {
            return fixture.lookup(language, text);
        }
        match language {
            Language::JA => self.phonemize_ja(text),
            Language::EN => self.phonemize_en(text),
            Language::ZH => self.phonemize_zh(text),
        }
    }

    fn phonemize_ja(&self, text: &str) -> Result<PhonemizeResult> {
        if let Some(g2p) = &self.ja_native_g2p {
            return self.phonemize_ja_via_native(g2p.as_ref(), text);
        }
        match &self.ja_g2p {
            Some(g2p) => self.phonemize_ja_via_piper(g2p.as_ref(), text),
            None => self.phonemize_ja_char_mapping(text),
        }
    }

    fn phonemize_ja_via_native(
        &self,
        provider: &dyn SbV2JapaneseG2pProvider,
        text: &str,
    ) -> Result<PhonemizeResult> {
        let output = provider.phonemize(text)?;
        if output.normalized_text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra native Japanese G2P returned empty normalized text".to_owned(),
            ));
        }
        if output.phones.is_empty() {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra native Japanese G2P returned no phones".to_owned(),
            ));
        }
        if output.raw_tones.len() != output.phones.len() {
            return Err(VokraError::InvalidArgument(format!(
                "SBV2 JP-Extra native Japanese G2P returned {} tones for {} phones",
                output.raw_tones.len(),
                output.phones.len()
            )));
        }
        if output.word2ph.is_empty() || output.word2ph.iter().any(|&width| width == 0) {
            return Err(VokraError::InvalidArgument(
                "SBV2 JP-Extra native Japanese G2P returned an invalid word2ph sequence".to_owned(),
            ));
        }
        let phone_count: usize = output
            .word2ph
            .iter()
            .try_fold(0usize, |sum, &width| sum.checked_add(width))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "SBV2 JP-Extra native Japanese G2P word2ph length overflow".to_owned(),
                )
            })?;
        if phone_count != output.phones.len() {
            return Err(VokraError::InvalidArgument(format!(
                "SBV2 JP-Extra native Japanese G2P word2ph covers {phone_count} phones, expected {}",
                output.phones.len()
            )));
        }

        let mut ids = Vec::with_capacity(output.phones.len());
        let mut tones = Vec::with_capacity(output.phones.len());
        for (position, (phone, &raw_tone)) in output
            .phones
            .iter()
            .zip(output.raw_tones.iter())
            .enumerate()
        {
            let id = self.ja_native_symbols.get(phone).copied().ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "SBV2 JP-Extra native Japanese G2P phone {phone:?} at position {position} is absent from the authenticated source vocabulary"
                ))
            })?;
            ids.push(id);
            tones.push(global_tone(Language::JA, raw_tone)?);
        }
        let mut word_boundaries = vec![false; ids.len()];
        let mut position = 0usize;
        for width in output.word2ph {
            word_boundaries[position] = true;
            position += width;
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries,
            bert_input_text: output.normalized_text,
        })
    }

    /// Real-G2P JA path: routes `g2p`'s piper-plus phoneme id sequence
    /// through `ja_mapping` into SBV2 phoneme-table space.
    fn phonemize_ja_via_piper(&self, g2p: &dyn Phonemizer, text: &str) -> Result<PhonemizeResult> {
        let piper_ids = g2p.phonemize(text)?;
        let mut ids = Vec::with_capacity(piper_ids.len());
        let mut tones = Vec::with_capacity(piper_ids.len());
        let mut wb = Vec::with_capacity(piper_ids.len());
        for (i, piper_id) in piper_ids.iter().enumerate() {
            let (id, tone) = match self.ja_mapping.get(piper_id).copied() {
                Some((id, raw_tone)) => (id, global_tone(Language::JA, raw_tone)?),
                None => match self.oov_policy {
                    OovPolicy::Lenient => (self.sbv2_default_phoneme_id, SBV2_JA_TONE_START),
                    OovPolicy::Strict => {
                        return Err(oov_error_piper(Language::JA, *piper_id, i, "ja_mapping"));
                    }
                },
            };
            ids.push(id);
            tones.push(tone);
            // The piper-plus id sequence carries only BOS/PAD/EOS framing
            // (`PhonemeTable::frame`), no word segmentation of its own.
            // Conservatively mark only the first emitted phoneme as a word
            // start.
            //
            // COSMETIC-BUNDLE (2026-08-09): the pre-fix `TODO(Task 17-19):
            // tighten word-boundary detection when text encoder lands` is
            // now moot — the SBV2 v2 real-checkpoint text encoder consumes
            // a `language_embed` [3, d_model] table (ZH/JP/EN one-hot),
            // not the design-doc-guessed `wb_embed [2, d_model]` table
            // (see `sbv2::text_encoder::N_LANGUAGES`'s "Formerly the SBV2
            // v2 design doc §7 assumed a `word_boundary_emb` table" note
            // for the M6 primary-source verification). `word_boundaries`
            // is retained on the `PhonemizeResult` API only because it is
            // honest linguistic output of the G2P stage and the
            // parity-fixture format (`word_boundaries.bin` in
            // `tests/parity_sbv2_real.rs`'s manifest schema) already
            // documents it; no consumer inside this crate reads it, so
            // the conservative first-only marking here is the correct
            // steady-state behaviour, not a placeholder.
            wb.push(i == 0);
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    /// Synthetic-mapping JA path (Task 14; test-only, see
    /// [`synthetic_for_test`](SbV2Phonemizer::synthetic_for_test)).
    fn phonemize_ja_char_mapping(&self, text: &str) -> Result<PhonemizeResult> {
        let mut ids = Vec::new();
        let mut tones = Vec::new();
        let mut wb = Vec::new();
        for (i, c) in text.chars().enumerate() {
            let (id, tone) = match self.ja_char_mapping.get(&c).copied() {
                Some(pair) => pair,
                None => match self.oov_policy {
                    OovPolicy::Lenient => (self.sbv2_default_phoneme_id, 0),
                    OovPolicy::Strict => {
                        return Err(oov_error_char(Language::JA, c, i, "ja_char_mapping"));
                    }
                },
            };
            ids.push(id);
            tones.push(tone);
            wb.push(false);
        }
        // Mark first phoneme as word boundary start.
        if !wb.is_empty() {
            wb[0] = true;
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    fn phonemize_en(&self, text: &str) -> Result<PhonemizeResult> {
        match &self.en_g2p {
            Some(g2p) => self.phonemize_en_via_piper(g2p.as_ref(), text),
            None => self.phonemize_en_char_mapping(text),
        }
    }

    /// Real-G2P EN path: routes `g2p`'s piper-plus phoneme id sequence
    /// through `en_mapping` into SBV2 phoneme-table space. English raw tones
    /// are converted to the authenticated global EN band 8..11.
    fn phonemize_en_via_piper(&self, g2p: &dyn Phonemizer, text: &str) -> Result<PhonemizeResult> {
        let piper_ids = g2p.phonemize(text)?;
        let mut ids = Vec::with_capacity(piper_ids.len());
        let mut tones = Vec::with_capacity(piper_ids.len());
        let mut wb = Vec::with_capacity(piper_ids.len());
        for (i, piper_id) in piper_ids.iter().enumerate() {
            let (id, tone) = match self.en_mapping.get(piper_id).copied() {
                Some((id, raw_tone)) => (id, global_tone(Language::EN, raw_tone)?),
                None => match self.oov_policy {
                    OovPolicy::Lenient => (self.sbv2_default_phoneme_id, SBV2_EN_TONE_START),
                    OovPolicy::Strict => {
                        return Err(oov_error_piper(Language::EN, *piper_id, i, "en_mapping"));
                    }
                },
            };
            ids.push(id);
            tones.push(tone);
            // COSMETIC-BUNDLE (2026-08-09): the pre-fix
            // `TODO(Task 17-19): tighten word-boundary detection` is moot
            // for the same M6 reason documented on `phonemize_ja_via_piper`
            // — the SBV2 v2 text encoder consumes `language_embed`, not
            // `wb_embed`. `word_boundaries` is API-retained solely for the
            // parity-fixture format (`word_boundaries.bin`); no in-crate
            // consumer reads it here.
            wb.push(i == 0);
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    /// Synthetic-mapping EN path (Task 14; test-only, see
    /// [`synthetic_for_test`](SbV2Phonemizer::synthetic_for_test)).
    fn phonemize_en_char_mapping(&self, text: &str) -> Result<PhonemizeResult> {
        let mut ids = Vec::new();
        let mut wb = Vec::new();
        // Tracks whether the next non-space char starts a new word. Starts
        // `true` so the first character of the input begins word 1.
        //
        // Space characters are consumed without pushing any `wb` entry —
        // the flag is carried forward and applied to the next real
        // character instead. This keeps `wb` and `ids` the same length by
        // construction (one entry per emitted phoneme), avoiding the
        // previous approach's per-space phantom push + tail-truncation
        // reconciliation, which misaligned boundaries for 3+ word inputs
        // (see `en_phonemize_multiword_word_boundaries_aligned` regression
        // test).
        let mut next_is_word_start = true;
        // Position reports the input-char index (including spaces) so the
        // caller can locate the offending char in their original text.
        for (i, c) in text.to_ascii_lowercase().chars().enumerate() {
            if c == ' ' {
                next_is_word_start = true;
                continue;
            }
            let id = match self.en_char_mapping.get(&c).copied() {
                Some(id) => id,
                None => match self.oov_policy {
                    OovPolicy::Lenient => self.sbv2_default_phoneme_id,
                    OovPolicy::Strict => {
                        return Err(oov_error_char(Language::EN, c, i, "en_char_mapping"));
                    }
                },
            };
            ids.push(id);
            wb.push(next_is_word_start);
            next_is_word_start = false;
        }
        let tones = vec![0u8; ids.len()];
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    // WP-18: ZH dispatch — fail-closed if no ZH G2P is wired (never a
    // silent char-map fallback like JA/EN have; see [`Language::ZH`]'s
    // doc for the rationale).
    fn phonemize_zh(&self, text: &str) -> Result<PhonemizeResult> {
        match &self.zh_g2p {
            Some(g2p) => self.phonemize_zh_via_piper(g2p.as_ref(), text),
            None => Err(VokraError::NotImplemented(
                "SbV2Phonemizer::phonemize: Language::ZH requires a ZH G2P wired via \
                 SbV2Phonemizer::with_zh_g2p (chain on top of SbV2Phonemizer::from_piper_g2p \
                 to install one). WP-18 landed the trait boundary + delegation only; the \
                 concrete piper-plus ZH bridge is WP-19 in \
                 integrations/vokra-piper-g2p (excluded workspace — preserves NFR-DS-02). \
                 No synthetic ZH char-map exists on purpose (fail-closed FR-EX-08).",
            )),
        }
    }

    /// Real-G2P ZH path: routes `g2p`'s piper-plus phoneme id sequence
    /// through `zh_mapping` into SBV2 phoneme-table space. ZH carries
    /// Mandarin lexical tones (raw 0 through 5), so `zh_mapping`'s value is
    /// `(sbv2_phoneme_id, tone)` — same shape as `ja_mapping`; a piper id
    /// missing from the mapping falls back to
    /// `(sbv2_default_phoneme_id, 0)`.
    fn phonemize_zh_via_piper(&self, g2p: &dyn Phonemizer, text: &str) -> Result<PhonemizeResult> {
        let piper_ids = g2p.phonemize(text)?;
        let mut ids = Vec::with_capacity(piper_ids.len());
        let mut tones = Vec::with_capacity(piper_ids.len());
        let mut wb = Vec::with_capacity(piper_ids.len());
        for (i, piper_id) in piper_ids.iter().enumerate() {
            let (id, tone) = match self.zh_mapping.get(piper_id).copied() {
                Some((id, raw_tone)) => (id, global_tone(Language::ZH, raw_tone)?),
                None => (self.sbv2_default_phoneme_id, SBV2_ZH_TONE_START),
            };
            ids.push(id);
            tones.push(tone);
            // TODO(WP-19+): tighten word-boundary detection when the
            // concrete piper-plus ZH bridge lands its own segmentation
            // (currently the piper-plus ZH phonemizer is a passthrough per
            // `integrations/vokra-piper-g2p/README.md`'s "zh ... passthrough"
            // note, so any pre-word-segmentation done in the bridge is
            // pass-through of the input text's own tokenization). Follows
            // the same conservative rule as JA/EN's real-G2P paths.
            wb.push(i == 0);
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }
}

/// Converts a language-local raw tone id from a piper mapping into the fixed
/// global `tone_embed` row. Rejecting out-of-band values here prevents a
/// caller-supplied mapping from accidentally indexing another language's
/// tone rows.
fn global_tone(language: Language, raw_tone: u8) -> Result<u8> {
    let (start, count) = match language {
        Language::ZH => (SBV2_ZH_TONE_START, SBV2_ZH_TONE_COUNT),
        Language::JA => (SBV2_JA_TONE_START, SBV2_JA_TONE_COUNT),
        Language::EN => (SBV2_EN_TONE_START, SBV2_EN_TONE_COUNT),
    };
    if raw_tone >= count {
        return Err(VokraError::InvalidArgument(format!(
            "SBV2 {language:?} raw tone {raw_tone} is outside the authenticated local range 0..{count}"
        )));
    }
    start.checked_add(raw_tone).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "SBV2 {language:?} tone offset overflow for raw tone {raw_tone}"
        ))
    })
}

fn canonical_symbol_bytes(symbols: &[String]) -> Vec<u8> {
    let capacity = symbols.iter().map(|symbol| symbol.len() + 1).sum();
    let mut canonical = Vec::with_capacity(capacity);
    for symbol in symbols {
        canonical.extend_from_slice(symbol.as_bytes());
        canonical.push(0);
    }
    canonical
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from(DIGITS[(byte >> 4) as usize]));
        output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MockJapaneseG2p {
        output: SbV2JapaneseG2pOutput,
    }

    impl SbV2JapaneseG2pProvider for MockJapaneseG2p {
        fn phonemize(&self, _text: &str) -> Result<SbV2JapaneseG2pOutput> {
            Ok(self.output.clone())
        }
    }

    #[test]
    fn native_route_mapping_mechanics_are_independent_of_contract_validation() {
        let provider = MockJapaneseG2p {
            output: SbV2JapaneseG2pOutput {
                normalized_text: "正規化".to_owned(),
                phones: vec!["_".to_owned(), "test-symbol-1".to_owned()],
                raw_tones: vec![0, 1],
                word2ph: vec![1, 1],
            },
        };
        let mut phonemizer = SbV2Phonemizer::synthetic_for_test();
        phonemizer.ja_native_symbols =
            HashMap::from([("_".to_owned(), 0), ("test-symbol-1".to_owned(), 1)]);
        let result = phonemizer
            .phonemize_ja_via_native(&provider, "入力")
            .expect("native route mechanics");
        assert_eq!(result.phoneme_ids, vec![0, 1]);
        assert_eq!(result.tones, vec![6, 7]);
        assert_eq!(result.word_boundaries, vec![true, true]);
        assert_eq!(result.bert_input_text, "正規化");
    }
}

// WP-14 OOV error constructors, factored out so all four Strict-arm sites
// (`phonemize_{ja,en}_via_piper` + `phonemize_{ja,en}_char_mapping`) share
// the same message shape — a caller that greps error text for one form
// will find every path. Both messages carry the concrete offending value
// (char or piper phoneme id) AND its 0-based position AND the mapping
// name AND the WP-14 opt-out affordance, so no field the WP-14 tests
// (`crates/vokra-models/tests/sbv2_g2p.rs` `wp14_*`) assert on drifts
// silently between paths.

fn oov_error_char(language: Language, c: char, position: usize, mapping_name: &str) -> VokraError {
    VokraError::InvalidArgument(format!(
        "SbV2Phonemizer::phonemize ({language:?} char path): char {c:?} \
         (U+{codepoint:04X}) at position {position} is absent from {mapping_name} — \
         OovPolicy::Strict rejects this per FR-EX-08. Add a mapping entry, or opt into \
         legacy silent fallback via SbV2Phonemizer::with_oov_policy(OovPolicy::Lenient).",
        codepoint = c as u32,
    ))
}

fn oov_error_piper(
    language: Language,
    piper_id: i64,
    position: usize,
    mapping_name: &str,
) -> VokraError {
    VokraError::InvalidArgument(format!(
        "SbV2Phonemizer::phonemize ({language:?} piper path): piper phoneme id {piper_id} at \
         position {position} is absent from {mapping_name} (SBV2 id) — OovPolicy::Strict \
         rejects this per FR-EX-08. Add a mapping entry, or opt into legacy silent fallback \
         via SbV2Phonemizer::with_oov_policy(OovPolicy::Lenient)."
    ))
}
