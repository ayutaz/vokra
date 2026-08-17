//! **WeTextProcessing** (`wenet-e2e/WeTextProcessing`, **Apache-2.0**) —
//! the GGUF binder for inverse text normalization / text normalization grammar
//! bundles (Wave D 2026-08-15, a brand-new `text-normalization` category).
//!
//! # Primary sources
//!
//! - Repository: <https://github.com/wenet-e2e/WeTextProcessing> — license
//!   `Apache-2.0` per the GitHub repository API (`license.spdx_id`, verified
//!   2026-08-15; 809 stars).
//! - `runtime/processor/wetext_processor.cc` — the `Tag` → `Verbalize` →
//!   `Normalize` pipeline (`StdVectorFst::Read`, `StringCompiler`,
//!   `fst::Compose`, `fst::ShortestPath`, `StringPrinter`).
//! - `runtime/processor/wetext_token_parser.cc` — the tagged-token grammar and
//!   the per-language field-order tables.
//!
//! # What this binder is for
//!
//! Every ASR model in the Vokra catalogue emits normalized, unpunctuated text.
//! An utterance spoken as *"one hundred fourteen thousand five"* comes back as
//! those words; a production transcript needs `114005`. This is the missing
//! back half of the ASR pipeline. The same machine run over the *other* pair of
//! grammars is TN — the front half of a TTS pipeline.
//!
//! # Layering
//!
//! This crate is the **GGUF-aware** layer (`vokra-core` = GGUF reader,
//! `vokra-models` = GGUF binder, `vokra-ops` = GGUF-unaware operators,
//! `vokra-convert` = GGUF writer). So all this module does is:
//!
//! 1. verify `vokra.model.arch == "wetextprocessing"` strictly;
//! 2. read the `vokra.itn.*` chunk group (language, direction, the two grammar
//!    blobs, their sizes, their OpenFST header flags, the readability verdict);
//! 3. hand the bytes to [`vokra_ops::itn::ItnGrammarSet`], which owns every
//!    behaviour.
//!
//! The pipeline itself ([`vokra_ops::itn::ItnPipeline`]) reuses
//! `vokra_core::decode::wfst` — the M5-06 from-scratch OpenFST decode port.
//!
//! # Implementation-status matrix
//!
//! - **Real**: [`WeTextProcessing::from_gguf`] (strict arch check, strict
//!   `vokra.itn.*` chunk validation with a size cross-check against the
//!   embedded blobs, license-class surfacing), [`WeTextProcessing::grammars`],
//!   [`WeTextProcessing::reader_gap`], and the whole tagged-token /
//!   field-order / `Reorder` stage reachable through
//!   [`WeTextProcessing::reorder_tagged`].
//! - **Loud-partial**: [`WeTextProcessing::pipeline`] (and therefore
//!   [`WeTextProcessing::normalize`]) returns
//!   [`VokraError::UnsupportedOp`] when `vokra-ops` was built without the
//!   `vokra-wfst` feature, or when the stored grammars are outside the shape
//!   Vokra's byte-verified `read_openfst_vector` accepts (most commonly:
//!   pynini attached byte symbol tables, so the OpenFST header flags are
//!   non-zero). The message names the exact field, the exact value, and the
//!   exact developer-side command that closes it. **No fabricated normalised
//!   text is ever returned** (FR-EX-08).
//!
//! # Arch distinctness
//!
//! [`ARCH`] = `"wetextprocessing"` is the only FST-grammar arch in the tree, so
//! there is no near-neighbour to be confused with — but the check is strict
//! anyway. A GGUF from any other converter carries tensors and no
//! `vokra.itn.*` group; failing with a named arch mismatch is far more useful
//! than failing later on a missing metadata key.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::itn::{ItnGrammarSet, ItnParseType, ItnPipeline};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/wetextprocessing.rs`.
// Duplicated (not imported) so `vokra-models` gains no dependency edge onto
// `vokra-convert`; the same rule every sibling binder follows.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch`.
pub const ARCH: &str = "wetextprocessing";

/// Expected `vokra.model.category`.
pub const CATEGORY: &str = "text-normalization";

/// Upstream repository (echoed in loud-partial diagnostics).
pub const UPSTREAM_URL: &str = "github.com/wenet-e2e/WeTextProcessing";

/// GGUF metadata key: ISO-639-1 language (`zh` / `en` / `ja`).
pub const KEY_ITN_LANGUAGE: &str = "vokra.itn.language";
/// GGUF metadata key: direction (`itn` / `tn`).
pub const KEY_ITN_DIRECTION: &str = "vokra.itn.direction";
/// GGUF metadata key: the compiled `tagger.fst`, as a `U8` array.
pub const KEY_ITN_TAGGER_FST: &str = "vokra.itn.tagger_fst";
/// GGUF metadata key: the compiled `verbalizer.fst`, as a `U8` array.
pub const KEY_ITN_VERBALIZER_FST: &str = "vokra.itn.verbalizer_fst";
/// GGUF metadata key: `tagger.fst` byte length.
pub const KEY_ITN_TAGGER_BYTES: &str = "vokra.itn.tagger_bytes";
/// GGUF metadata key: `verbalizer.fst` byte length.
pub const KEY_ITN_VERBALIZER_BYTES: &str = "vokra.itn.verbalizer_bytes";
/// GGUF metadata key: whether Vokra's OpenFST reader can parse both grammars.
pub const KEY_ITN_VOKRA_READABLE: &str = "vokra.itn.vokra_readable";

/// A bound WeTextProcessing grammar bundle.
#[derive(Debug)]
pub struct WeTextProcessing {
    grammars: ItnGrammarSet,
    weight_license: LicenseClass,
    /// The converter's readability verdict, if it stamped one. `None` means the
    /// key was absent (an older artifact); the binder then falls back to its
    /// own header probe rather than assuming either answer.
    stamped_readable: Option<bool>,
}

impl WeTextProcessing {
    /// Binds a WeTextProcessing GGUF.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the missing
    /// or wrong key, so a mis-produced artifact has exactly one place to walk
    /// (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is not
    ///   `"wetextprocessing"`;
    /// - [`VokraError::ModelLoad`] when the language / direction pair is absent
    ///   or unrecognised (a bundle whose direction cannot be established would
    ///   silently pick the wrong field-order table at verbalize time);
    /// - [`VokraError::ModelLoad`] when either grammar blob is absent, is not a
    ///   `U8` array, holds an out-of-range element, or disagrees with its
    ///   stamped byte length;
    /// - [`VokraError::ModelLoad`] when a blob is not a readable OpenFST header
    ///   (via [`ItnGrammarSet::new`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, so a foreign GGUF fails with a specific message
        //    rather than a downstream missing-key error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "wetextprocessing: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model wetextprocessing`?). This is \
                     the only FST-GRAMMAR arch in the tree: every other converter emits weight \
                     tensors, while a WeTextProcessing bundle carries zero tensors and a \
                     `vokra.itn.*` chunk group holding two compiled OpenFST transducers. \
                     Silently accepting a foreign arch would mean handing arbitrary bytes to \
                     the OpenFST reader (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "wetextprocessing: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native WeTextProcessing GGUF (was it produced by `vokra-cli convert \
                     --model wetextprocessing`?)"
                        .to_owned(),
                ));
            }
        }

        // 2. Language + direction. Both must be present AND recognised: the
        //    direction selects the field-order table, and picking the wrong one
        //    silently reorders every verbalizer input.
        let language = read_str(file, KEY_ITN_LANGUAGE)?;
        let direction = read_str(file, KEY_ITN_DIRECTION)?;
        let parse_type =
            ItnParseType::from_language_direction(&language, &direction).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "wetextprocessing: unrecognised bundle `{language}_{direction}` — the \
                     upstream grammars are zh/en/ja x tn/itn (`zh_tn`, `zh_itn`, `en_tn`, \
                     `en_itn`, `ja_tn`, `ja_itn`). The direction selects the field-order table \
                     the verbalizer stage was compiled against, so guessing it would silently \
                     reorder every tagged token (FR-EX-08)."
                ))
            })?;

        // 3. The two grammar blobs, cross-checked against their stamped sizes.
        let tagger = read_blob(file, KEY_ITN_TAGGER_FST, KEY_ITN_TAGGER_BYTES, "tagger")?;
        let verbalizer = read_blob(
            file,
            KEY_ITN_VERBALIZER_FST,
            KEY_ITN_VERBALIZER_BYTES,
            "verbalizer",
        )?;

        // 4. Hand off to the op layer, which probes both OpenFST headers.
        let grammars = ItnGrammarSet::new(parse_type, tagger, verbalizer)?;

        // 5. Provenance surfacing. A GGUF missing the stamp reads back as
        //    `Unknown` (fail-closed at the M2-13 compliance gate).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        let stamped_readable = file.get(KEY_ITN_VOKRA_READABLE).and_then(|v| v.as_bool());

        Ok(Self {
            grammars,
            weight_license,
            stamped_readable,
        })
    }

    /// The bound grammar bundle.
    #[must_use]
    pub const fn grammars(&self) -> &ItnGrammarSet {
        &self.grammars
    }

    /// The language + direction this bundle was compiled for.
    #[must_use]
    pub const fn parse_type(&self) -> ItnParseType {
        self.grammars.parse_type()
    }

    /// The stamped weight-license class. The converter stamps `Permissive`
    /// (apache-2.0); a GGUF missing the stamp reads back as `Unknown`
    /// (fail-closed).
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The converter's `vokra.itn.vokra_readable` verdict, when it stamped one.
    #[must_use]
    pub const fn stamped_readable(&self) -> Option<bool> {
        self.stamped_readable
    }

    /// The reason Vokra's OpenFST reader cannot consume this bundle, or [`None`]
    /// when it can.
    ///
    /// Computed from the binder's own header probe rather than from the stamped
    /// verdict, so an artifact produced by an older converter (no stamp) still
    /// gets an honest answer.
    #[must_use]
    pub fn reader_gap(&self) -> Option<String> {
        self.grammars.compatibility().err().map(|e| format!("{e}"))
    }

    /// The tagger→verbalizer middle stage on its own: parse a tagged-token
    /// stream and rewrite its fields into the bundle's field order.
    ///
    /// **Real in every build** — it needs no FST. Useful for inspecting a
    /// tagger's output, and it is what a feature-off build can still exercise
    /// end to end.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for a malformed tagged-token stream.
    pub fn reorder_tagged(&self, tagged: &str) -> Result<String> {
        vokra_ops::itn::reorder(tagged, self.parse_type())
    }

    /// Builds a runnable pipeline over the bound grammars.
    ///
    /// This is the **loud-partial point**. Hold the returned [`ItnPipeline`] and
    /// reuse it: building one parses both compiled FSTs, which is the expensive
    /// step (a real `zh_itn` tagger is on the order of 10^5–10^6 states).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] when `vokra-ops` was built without the
    /// `vokra-wfst` feature, or when the stored grammars carry embedded symbol
    /// tables / an unverified version / a non-vector or non-tropical type. The
    /// message names the field, its value, and the developer-side fix.
    pub fn pipeline(&self) -> Result<ItnPipeline> {
        ItnPipeline::new(self.grammars.clone())
    }

    /// Convenience: run the whole `Verbalize(Tag(input))` pipeline once.
    ///
    /// **Builds a pipeline per call**, so it re-parses both grammars every
    /// time. Fine for a one-shot; for repeated use hold an [`ItnPipeline`] from
    /// [`Self::pipeline`] instead.
    ///
    /// # Errors
    ///
    /// As [`Self::pipeline`], plus the composition errors from the pipeline
    /// itself.
    pub fn normalize(&self, input: &str) -> Result<String> {
        self.pipeline()?.normalize(input)
    }
}

/// Reads a required string metadata key.
fn read_str(file: &GgufFile, key: &str) -> Result<String> {
    file.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "wetextprocessing: GGUF is missing the `{key}` string chunk. A bundle produced \
                 by `vokra-cli convert --model wetextprocessing` always stamps the whole \
                 `vokra.itn.*` group; a GGUF without it cannot say which grammar it holds."
            ))
        })
}

/// Reads a `U8` metadata array blob and cross-checks it against its stamped
/// byte length.
fn read_blob(file: &GgufFile, key: &str, len_key: &str, which: &str) -> Result<Vec<u8>> {
    let array = file.get(key).and_then(|v| v.as_array()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "wetextprocessing: GGUF is missing the `{key}` chunk (the compiled {which} \
                 grammar, embedded as a `U8` metadata array — the `vokra.tokenizer.model` \
                 precedent). Without it there is nothing to compose against. Upstream \
                 reference: https://{UPSTREAM_URL}"
        ))
    })?;
    let mut bytes = Vec::with_capacity(array.values.len());
    for (i, v) in array.values.iter().enumerate() {
        let raw = v.as_u64().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "wetextprocessing: `{key}` element {i} is not an integer — the {which} grammar \
                 must be a `U8` array of raw OpenFST bytes"
            ))
        })?;
        let byte = u8::try_from(raw).map_err(|_| {
            VokraError::ModelLoad(format!(
                "wetextprocessing: `{key}` element {i} is {raw}, out of `U8` range — the \
                 {which} grammar blob is corrupt"
            ))
        })?;
        bytes.push(byte);
    }
    // A length disagreement means the array was truncated or re-written without
    // updating the stamp: refuse rather than parse a half grammar (FR-EX-08).
    if let Some(stamped) = file.get(len_key).and_then(|v| v.as_u64())
        && stamped != bytes.len() as u64
    {
        return Err(VokraError::ModelLoad(format!(
            "wetextprocessing: the {which} grammar is {} bytes but `{len_key}` says {stamped} — \
             the blob and its stamp disagree, so the GGUF was truncated or edited. Refusing to \
             parse a partial FST (FR-EX-08).",
            bytes.len()
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};

    /// A minimal, well-formed OpenFST `VectorFst<StdArc>` binary (header plus a
    /// one-state, zero-arc, final body) in the byte layout already verified
    /// against real OpenFST 1.8.4 in `vokra-core/src/decode/wfst/reader.rs`.
    ///
    /// A scaffold, NOT a format oracle — the positive proof that the reader
    /// handles a real producer's output lives in
    /// `vokra-core/tests/parity_wfst.rs`.
    fn fst_bytes(flags: i32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&0x7EB2_FDD6u32.to_le_bytes());
        v.extend_from_slice(&6i32.to_le_bytes());
        v.extend_from_slice(b"vector");
        v.extend_from_slice(&8i32.to_le_bytes());
        v.extend_from_slice(b"standard");
        v.extend_from_slice(&2i32.to_le_bytes());
        v.extend_from_slice(&flags.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&0i64.to_le_bytes());
        v.extend_from_slice(&1i64.to_le_bytes());
        v.extend_from_slice(&0i64.to_le_bytes());
        v.extend_from_slice(&0.0f32.to_le_bytes());
        v.extend_from_slice(&0i64.to_le_bytes());
        v
    }

    fn u8_array(bytes: &[u8]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().copied().map(GgufMetadataValue::U8).collect(),
        })
    }

    struct Synth {
        arch: Option<&'static str>,
        language: Option<&'static str>,
        direction: Option<&'static str>,
        tagger: Option<Vec<u8>>,
        verbalizer: Option<Vec<u8>>,
        tagger_len_override: Option<u64>,
        license_class: Option<&'static str>,
    }

    impl Default for Synth {
        fn default() -> Self {
            Self {
                arch: Some(ARCH),
                language: Some("zh"),
                direction: Some("itn"),
                tagger: Some(fst_bytes(0)),
                verbalizer: Some(fst_bytes(0)),
                tagger_len_override: None,
                license_class: Some("permissive"),
            }
        }
    }

    impl Synth {
        fn build(self) -> GgufFile {
            let mut b = GgufBuilder::new();
            if let Some(a) = self.arch {
                b.add_string(chunks::KEY_MODEL_ARCH, a);
            }
            if let Some(l) = self.language {
                b.add_string(KEY_ITN_LANGUAGE, l);
            }
            if let Some(d) = self.direction {
                b.add_string(KEY_ITN_DIRECTION, d);
            }
            if let Some(c) = self.license_class {
                b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, c);
            }
            if let Some(t) = &self.tagger {
                b.add_metadata(KEY_ITN_TAGGER_FST, u8_array(t));
                b.add_metadata(
                    KEY_ITN_TAGGER_BYTES,
                    GgufMetadataValue::U64(self.tagger_len_override.unwrap_or(t.len() as u64)),
                );
            }
            if let Some(v) = &self.verbalizer {
                b.add_metadata(KEY_ITN_VERBALIZER_FST, u8_array(v));
                b.add_metadata(
                    KEY_ITN_VERBALIZER_BYTES,
                    GgufMetadataValue::U64(v.len() as u64),
                );
            }
            b.add_bool(KEY_ITN_VOKRA_READABLE, true);
            GgufFile::parse(b.to_bytes().expect("synthetic GGUF assembles")).expect("parses")
        }
    }

    #[test]
    fn binds_a_synthetic_bundle() {
        let m = WeTextProcessing::from_gguf(&Synth::default().build()).unwrap();
        assert_eq!(m.parse_type(), ItnParseType::ZhItn);
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
        assert_eq!(m.stamped_readable(), Some(true));
        assert_eq!(m.grammars().tagger_bytes(), fst_bytes(0).as_slice());
        assert_eq!(m.reader_gap(), None);
    }

    #[test]
    fn every_language_direction_pair_binds_to_its_own_parse_type() {
        for pt in ItnParseType::all() {
            let mut b = GgufBuilder::new();
            b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
            b.add_string(KEY_ITN_LANGUAGE, pt.language());
            b.add_string(KEY_ITN_DIRECTION, pt.direction());
            b.add_metadata(KEY_ITN_TAGGER_FST, u8_array(&fst_bytes(0)));
            b.add_metadata(KEY_ITN_VERBALIZER_FST, u8_array(&fst_bytes(0)));
            let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
            let m = WeTextProcessing::from_gguf(&file).unwrap();
            assert_eq!(m.parse_type(), pt, "{}", pt.prefix());
        }
    }

    #[test]
    fn missing_arch_is_loud() {
        let file = Synth {
            arch: None,
            ..Default::default()
        }
        .build();
        let Err(err) = WeTextProcessing::from_gguf(&file) else {
            panic!("expected an error when vokra.model.arch is absent");
        };
        let msg = format!("{err}");
        assert!(msg.contains("vokra.model.arch"), "{msg}");
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    #[test]
    fn foreign_arch_names_both_expected_and_actual() {
        let file = Synth {
            arch: Some("whisper"),
            ..Default::default()
        }
        .build();
        let Err(err) = WeTextProcessing::from_gguf(&file) else {
            panic!("expected an error for a foreign arch");
        };
        let msg = format!("{err}");
        assert!(msg.contains("whisper"), "actual arch must be named: {msg}");
        assert!(
            msg.contains("wetextprocessing"),
            "expected arch must be named: {msg}"
        );
    }

    #[test]
    fn missing_language_or_direction_is_loud_and_names_the_key() {
        for (synth, key) in [
            (
                Synth {
                    language: None,
                    ..Default::default()
                },
                KEY_ITN_LANGUAGE,
            ),
            (
                Synth {
                    direction: None,
                    ..Default::default()
                },
                KEY_ITN_DIRECTION,
            ),
        ] {
            let Err(err) = WeTextProcessing::from_gguf(&synth.build()) else {
                panic!("expected an error when `{key}` is absent");
            };
            assert!(format!("{err}").contains(key), "{err}");
        }
    }

    #[test]
    fn an_unknown_direction_is_refused_rather_than_defaulted() {
        let file = Synth {
            direction: Some("sideways"),
            ..Default::default()
        }
        .build();
        let Err(err) = WeTextProcessing::from_gguf(&file) else {
            panic!("expected an error for an unrecognised direction");
        };
        let msg = format!("{err}");
        assert!(msg.contains("zh_sideways"), "{msg}");
        assert!(msg.contains("field-order table"), "{msg}");
    }

    #[test]
    fn missing_grammar_blob_names_the_tensor_key() {
        let file = Synth {
            verbalizer: None,
            ..Default::default()
        }
        .build();
        let Err(err) = WeTextProcessing::from_gguf(&file) else {
            panic!("expected an error when the verbalizer blob is absent");
        };
        let msg = format!("{err}");
        assert!(msg.contains(KEY_ITN_VERBALIZER_FST), "{msg}");
        assert!(msg.contains("verbalizer"), "{msg}");
    }

    #[test]
    fn a_size_stamp_disagreement_is_refused() {
        let file = Synth {
            tagger_len_override: Some(999_999),
            ..Default::default()
        }
        .build();
        let Err(err) = WeTextProcessing::from_gguf(&file) else {
            panic!("expected an error when the blob and its stamp disagree");
        };
        let msg = format!("{err}");
        assert!(msg.contains("999999"), "{msg}");
        assert!(msg.contains("disagree"), "{msg}");
    }

    #[test]
    fn a_non_openfst_blob_is_refused() {
        let file = Synth {
            tagger: Some(b"definitely not an fst".to_vec()),
            ..Default::default()
        }
        .build();
        let Err(err) = WeTextProcessing::from_gguf(&file) else {
            panic!("expected an error for a non-OpenFST tagger blob");
        };
        assert!(format!("{err}").contains("tagger"), "{err}");
    }

    #[test]
    fn a_missing_license_stamp_falls_back_to_unknown() {
        let file = Synth {
            license_class: None,
            ..Default::default()
        }
        .build();
        let m = WeTextProcessing::from_gguf(&file).unwrap();
        assert_eq!(m.weight_license(), LicenseClass::Unknown);
    }

    #[test]
    fn the_reorder_stage_is_real_in_every_build() {
        let m = WeTextProcessing::from_gguf(&Synth::default().build()).unwrap();
        let out = m
            .reorder_tagged(r#"date { month: "01" day: "28" year: "2002" }"#)
            .unwrap();
        assert_eq!(out, r#"date { year: "2002" month: "01" day: "28" }"#);
    }

    #[test]
    fn a_symbol_table_bundle_reports_its_gap_and_refuses_to_build_a_pipeline() {
        // pynini attaches byte symbol tables by default, so this is the shape a
        // real upstream bundle is most likely to arrive in. The binder must
        // still BIND (the bytes are faithfully carried) and must report the gap
        // rather than pretending the bundle is usable.
        let file = Synth {
            tagger: Some(fst_bytes(0x3)),
            ..Default::default()
        }
        .build();
        let m = WeTextProcessing::from_gguf(&file).expect("binding must still succeed");
        let gap = m
            .reader_gap()
            .expect("a symbol-table bundle must report a gap");
        assert!(gap.contains("fstsymbols --clear_isymbols"), "{gap}");
        assert!(gap.contains("WeTextProcessing"), "{gap}");

        let Err(err) = m.pipeline() else {
            panic!("a bundle Vokra cannot read must not build a pipeline");
        };
        assert!(matches!(err, VokraError::UnsupportedOp(_)), "{err}");
    }

    /// The loud-partial contract: `normalize` must never return fabricated
    /// text. On this build it either runs the real grammars or errors.
    #[test]
    fn normalize_never_fabricates() {
        let m = WeTextProcessing::from_gguf(&Synth::default().build()).unwrap();
        // The scaffold grammar is a 1-state acceptor with no arcs: it accepts
        // only the empty string, so any real input must fail.
        match m.normalize("一百一十四万零五") {
            Ok(out) => panic!("expected a loud failure, got a normalised string: {out:?}"),
            Err(err) => {
                let msg = format!("{err}");
                assert!(
                    msg.contains("itn"),
                    "the error must come from the ITN pipeline: {msg}"
                );
            }
        }
        // The empty-input short-circuit is upstream behaviour, not a fallback.
        assert_eq!(m.normalize("").unwrap(), "");
    }

    #[cfg(not(feature = "vokra-wfst"))]
    #[test]
    fn without_the_feature_the_pipeline_stage_names_the_flag() {
        let m = WeTextProcessing::from_gguf(&Synth::default().build()).unwrap();
        let p = m
            .pipeline()
            .expect("a readable bundle still builds a pipeline");
        let Err(err) = p.tag("一百") else {
            panic!("expected a loud-partial without the wfst feature");
        };
        let msg = format!("{err}");
        assert!(msg.contains("vokra-wfst"), "{msg}");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
