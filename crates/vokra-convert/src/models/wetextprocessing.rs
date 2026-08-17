//! **WeTextProcessing** (`wenet-e2e/WeTextProcessing`, **Apache-2.0**) —
//! inverse text normalization / text normalization grammar bundles:
//! compiled OpenFST `.fst` → GGUF (Wave D 2026-08-15, brand-new category).
//!
//! # Model class — a grammar bundle, not a neural network
//!
//! This converter is unlike every other one in this tree: WeTextProcessing has
//! **no weights**. A bundle is two compiled OpenFST transducers plus the
//! language/direction that selects a field-order table:
//!
//! ```text
//! <prefix>/tagger.fst       classification + raw-field tagging
//! <prefix>/verbalizer.fst   non-standard-word verbalization
//! <prefix> ∈ { zh_tn, zh_itn, en_tn, en_itn, ja_tn, ja_itn }
//! ```
//!
//! At runtime the pipeline is
//! `Verbalize(Tag(input))` — compose the input string with the tagger, take the
//! shortest path, reorder the resulting tagged fields, compose *that* with the
//! verbalizer, take the shortest path again. The runtime side lives in
//! `vokra-ops::itn`; this converter only packages the two grammars so the
//! artifact is self-describing.
//!
//! # Why ITN matters here
//!
//! Every ASR model in the Vokra catalogue emits normalized, unpunctuated text.
//! An utterance spoken as *"one hundred fourteen thousand five"* comes back as
//! those words; a production transcript needs `114005`. ITN is the missing back
//! half of the ASR pipeline, and WeTextProcessing is the reference
//! implementation for zh / en / ja.
//!
//! # Grammars are embedded as `U8` metadata arrays, not tensors
//!
//! [`GgmlType`](vokra_core::gguf::GgmlType) has no byte dtype — it carries
//! `F32` / `F16` / `BF16` / `Q4_K` / `Q5_K` / `Q6_K` only, all of which would
//! reinterpret the FST bytes as numbers. The established way this repository
//! embeds an opaque binary blob in a GGUF is a `U8` **metadata array**: the
//! Whisper detokenizer rides in `vokra.tokenizer.model` exactly this way
//! (`models/whisper.rs::embed_tokenizer`), as do the Voxtral / CSM / Moshi
//! tokenizers. The two grammars follow that precedent under
//! [`KEY_ITN_TAGGER_FST`] / [`KEY_ITN_VERBALIZER_FST`].
//!
//! The on-disk cost is one byte per element, but the *builder* holds one
//! `GgufMetadataValue::U8` enum per byte while assembling, so conversion peaks
//! at roughly an order of magnitude above the grammar size. Grammars above
//! [`MAX_GRAMMAR_BYTES`] are refused with a message that says so rather than
//! being allowed to OOM the machine.
//!
//! # Distinct arch tag
//!
//! [`ARCH`] = `"wetextprocessing"` is the first entry in a brand-new
//! `text-normalization` category — there is no sibling arch it could be
//! confused with, because no other converter in the tree emits FST grammars.
//! The runtime binder verifies it strictly anyway (FR-EX-08): a GGUF from any
//! other converter handed to `WeTextProcessing::from_gguf` fails with a named
//! arch mismatch rather than a downstream missing-key error.
//!
//! # License — Apache-2.0 (primary source: GitHub API)
//!
//! `github.com/wenet-e2e/WeTextProcessing` reports
//! `license.spdx_id = "Apache-2.0"` via the GitHub repository API (verified
//! 2026-08-15; the repo also ships a top-level `LICENSE`, and every runtime
//! source file carries the Apache-2.0 header). Apache-2.0 maps to
//! [`LicenseClass::Permissive`] — the same commercial verdict as MIT, with no
//! runtime-side attribution obligation.
//!
//! The compiled `.fst` files a user supplies are **built from** those
//! Apache-2.0 rules, but they are produced on the user's machine by pynini;
//! this converter does not redistribute any upstream grammar. The §3.1
//! sign-off column in `docs/license-audit.md` is **BLANK** (fail-closed —
//! CC MUST NOT sign a license row, that is owner-only per memory
//! `[[feedback-license-signoff-primary-source]]`).
//!
//! # No pynini, no OpenFST, no ONNX (permanent)
//!
//! The grammars are *built* by pynini developer-side and *consumed* here as
//! finished bytes. Neither this converter nor the runtime links any OpenFST or
//! pynini code (NFR-DS-02): the reader is the from-scratch Rust port in
//! `vokra_core::decode::wfst`.

use std::path::{Path, PathBuf};

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks};
use vokra_ops::itn::{ItnParseType, OpenFstHeader};

use crate::ConvertError;

/// `vokra.model.arch` = `wetextprocessing`.
pub const ARCH: &str = "wetextprocessing";

/// `vokra.model.name` prefix. The full name appends the bundle prefix, e.g.
/// `wetextprocessing-zh-itn`.
pub const NAME: &str = "wetextprocessing";

/// `vokra.model.category` = `text-normalization` — a brand-new category. Not
/// `asr` and not `tts`: ITN sits *after* an ASR decode and TN sits *before* a
/// TTS front-end, so advertising it as either would mis-tier it in the model
/// card generator and the zoo manifest.
pub const CATEGORY: &str = "text-normalization";

/// Upstream GitHub tree. WeTextProcessing is GitHub-native (the PyPI package
/// `WeTextProcessing` builds the grammars locally; no HuggingFace mirror), so
/// provenance uses `upstream_url` rather than `upstream_hf`.
pub const UPSTREAM_URL: &str = "github.com/wenet-e2e/WeTextProcessing";

/// Default SPDX for the bundle: `apache-2.0`, per the GitHub repository API
/// (`license.spdx_id = "Apache-2.0"`, verified 2026-08-15) and the Apache-2.0
/// header on every upstream source file.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Largest single grammar this converter will embed, in bytes.
///
/// The `U8` metadata-array encoding costs one byte per element on disk but one
/// `GgufMetadataValue` enum per byte in the builder, so a grammar this size
/// already peaks in the low gigabytes while assembling. Real ITN/TN grammars
/// are single-digit megabytes; anything approaching this cap is far more likely
/// to be a wrong file than a real grammar, so it is refused with an explanation
/// instead of being allowed to exhaust memory.
pub const MAX_GRAMMAR_BYTES: usize = 64 * 1024 * 1024;

/// Ad-hoc metadata key for the model category (converter-side constant, mirror
/// of the sibling `gtcrn` / `nsnet2` / `emotion2vec` posture).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for a non-HuggingFace upstream (GitHub here).
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// GGUF metadata key: ISO-639-1 language (`zh` / `en` / `ja`).
pub const KEY_ITN_LANGUAGE: &str = "vokra.itn.language";
/// GGUF metadata key: direction (`itn` = spoken→written, `tn` = written→spoken).
pub const KEY_ITN_DIRECTION: &str = "vokra.itn.direction";
/// GGUF metadata key: the upstream bundle prefix (`zh_itn`, `en_tn`, …).
pub const KEY_ITN_PREFIX: &str = "vokra.itn.prefix";
/// GGUF metadata key: the compiled `tagger.fst`, as a `U8` array.
pub const KEY_ITN_TAGGER_FST: &str = "vokra.itn.tagger_fst";
/// GGUF metadata key: the compiled `verbalizer.fst`, as a `U8` array.
pub const KEY_ITN_VERBALIZER_FST: &str = "vokra.itn.verbalizer_fst";
/// GGUF metadata key: `tagger.fst` byte length (cross-check for the array).
pub const KEY_ITN_TAGGER_BYTES: &str = "vokra.itn.tagger_bytes";
/// GGUF metadata key: `verbalizer.fst` byte length (cross-check for the array).
pub const KEY_ITN_VERBALIZER_BYTES: &str = "vokra.itn.verbalizer_bytes";
/// GGUF metadata key: `tagger.fst` OpenFST header flags (0 = no symbol tables).
pub const KEY_ITN_TAGGER_FLAGS: &str = "vokra.itn.tagger_flags";
/// GGUF metadata key: `verbalizer.fst` OpenFST header flags.
pub const KEY_ITN_VERBALIZER_FLAGS: &str = "vokra.itn.verbalizer_flags";
/// GGUF metadata key: `tagger.fst` state count (from its header).
pub const KEY_ITN_TAGGER_STATES: &str = "vokra.itn.tagger_num_states";
/// GGUF metadata key: `verbalizer.fst` state count (from its header).
pub const KEY_ITN_VERBALIZER_STATES: &str = "vokra.itn.verbalizer_num_states";
/// GGUF metadata key: whether Vokra's `read_openfst_vector` can parse **both**
/// grammars as stored.
///
/// `false` means the artifact is still a faithful, complete package of the
/// grammars, but the runtime will loud-partial on use until the grammars are
/// re-emitted without symbol tables (or the reader is extended). Stamping it
/// lets a consumer discover that from metadata alone, without parsing bodies.
pub const KEY_ITN_VOKRA_READABLE: &str = "vokra.itn.vokra_readable";

const UPSTREAM_SOURCE: &str = "wenet-e2e/WeTextProcessing (inverse text normalization / text normalization, \
     pynini-built OpenFST tagger + verbalizer grammars for zh / en / ja, Apache-2.0)";

/// Outcome of a WeTextProcessing bundle conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeTextProcessingReport {
    /// The bundle prefix that was resolved (`zh_itn`, `en_tn`, …).
    pub prefix: &'static str,
    /// Resolved path of the `tagger.fst` that was read.
    pub tagger_path: PathBuf,
    /// Resolved path of the `verbalizer.fst` that was read.
    pub verbalizer_path: PathBuf,
    /// Byte length of the tagger grammar.
    pub tagger_bytes: usize,
    /// Byte length of the verbalizer grammar.
    pub verbalizer_bytes: usize,
    /// State count from the tagger header.
    pub tagger_states: i64,
    /// State count from the verbalizer header.
    pub verbalizer_states: i64,
    /// `true` when Vokra's OpenFST reader can parse both grammars as stored.
    pub vokra_readable: bool,
    /// When `vokra_readable` is `false`, the first gap found, verbatim. Kept so
    /// the CLI can print the actionable message instead of a bare boolean.
    pub reader_gap: Option<String>,
}

/// Resolves the bundle prefix from a path, mirroring the upstream C++ runtime.
///
/// `wetext_processor.cc` derives its `ParseType` by probing the tagger path for
/// the substrings `zh_tn_` / `zh_itn_` / `en_tn_` / `en_itn_` / `ja_tn_` /
/// `ja_itn_`, and `LOG(FATAL)`s when none matches. This does the same over the
/// whole path (so both the flat `zh_itn_tagger.fst` layout and the bundle
/// directory layout `.../zh_itn/<hash>/tagger.fst` resolve), with two
/// improvements: an ambiguous path that contains two different prefixes is
/// refused instead of silently taking the first, and the caller can override.
///
/// The match requires a non-alphanumeric boundary on both sides of the prefix,
/// so a longer token that merely *contains* one of them (`.../frozen_tn_v2/`)
/// does not resolve. `-` is folded to `_` first, so both separator styles work.
fn resolve_prefix(path: &Path) -> Result<ItnParseType, ConvertError> {
    let text = path.to_string_lossy().to_lowercase().replace('-', "_");
    let mut found: Vec<ItnParseType> = Vec::new();
    for pt in ItnParseType::all() {
        let needle = pt.prefix();
        // Require a separator (or a string boundary) on both sides so `en_tn`
        // cannot match inside an unrelated longer token.
        let mut start = 0usize;
        while let Some(idx) = text[start..].find(needle) {
            let at = start + idx;
            let before_ok = at == 0 || !text.as_bytes()[at - 1].is_ascii_alphanumeric();
            let after = at + needle.len();
            let after_ok = after == text.len() || !text.as_bytes()[after].is_ascii_alphanumeric();
            if before_ok && after_ok {
                found.push(pt);
                break;
            }
            start = at + 1;
        }
    }
    match found.len() {
        1 => Ok(found[0]),
        0 => Err(ConvertError::Usage(format!(
            "wetextprocessing: cannot tell which grammar bundle `{}` is — the path contains \
             none of the six upstream prefixes (zh_tn, zh_itn, en_tn, en_itn, ja_tn, ja_itn). \
             The upstream C++ runtime resolves this the same way (see \
             runtime/processor/wetext_processor.cc, which probes the tagger path and \
             LOG(FATAL)s otherwise). Either point --input at a bundle whose path contains \
             the prefix (e.g. `.../zh_itn/tagger.fst` or `zh_itn_tagger.fst`), or call \
             `convert_wetextprocessing_file_with_type` with an explicit ItnParseType.",
            path.display()
        ))),
        _ => Err(ConvertError::Usage(format!(
            "wetextprocessing: the path `{}` is ambiguous — it contains {} different bundle \
             prefixes ({}). Refusing to guess which grammar this is, because picking wrong \
             would silently apply the wrong field-order table at verbalize time (FR-EX-08). \
             Pass an unambiguous path or an explicit ItnParseType.",
            path.display(),
            found.len(),
            found
                .iter()
                .map(|p| p.prefix())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// Resolves the pair of grammar files from `input`.
///
/// Accepts both real upstream layouts:
/// - a **bundle directory** containing `tagger.fst` + `verbalizer.fst` (the
///   layout `tn/cache.py::_BUNDLE_FILENAMES` writes);
/// - a **flat `<prefix>_tagger.fst` file**, whose sibling
///   `<prefix>_verbalizer.fst` is derived (the layout the C++ runtime's
///   filename probe expects);
/// - a plain `tagger.fst` file, whose sibling `verbalizer.fst` is derived.
fn resolve_pair(input: &Path) -> Result<(PathBuf, PathBuf), ConvertError> {
    if input.is_dir() {
        return Ok((input.join("tagger.fst"), input.join("verbalizer.fst")));
    }
    let name = input
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            ConvertError::Usage(format!(
                "wetextprocessing: --input `{}` has no usable file name",
                input.display()
            ))
        })?
        .to_owned();
    let dir = input.parent().unwrap_or_else(|| Path::new("."));
    if let Some(stem) = name.strip_suffix("tagger.fst") {
        // `stem` is "" for a plain `tagger.fst`, or "zh_itn_" for the flat layout.
        return Ok((
            input.to_path_buf(),
            dir.join(format!("{stem}verbalizer.fst")),
        ));
    }
    Err(ConvertError::Usage(format!(
        "wetextprocessing: --input `{}` is neither a bundle directory (containing \
         `tagger.fst` and `verbalizer.fst`) nor a `*tagger.fst` file. Point it at the \
         bundle directory the WeTextProcessing builder wrote, or at the tagger grammar \
         whose `*verbalizer.fst` sibling sits next to it.",
        input.display()
    )))
}

/// Reads one grammar, enforcing the size cap and probing its header.
fn read_grammar(path: &Path, which: &str) -> Result<(Vec<u8>, OpenFstHeader), ConvertError> {
    let meta = std::fs::metadata(path).map_err(|e| {
        ConvertError::Usage(format!(
            "wetextprocessing: cannot stat the {which} grammar `{}`: {e}. A WeTextProcessing \
             bundle is `tagger.fst` + `verbalizer.fst` side by side; both must be present.",
            path.display()
        ))
    })?;
    let len = usize::try_from(meta.len()).unwrap_or(usize::MAX);
    if len > MAX_GRAMMAR_BYTES {
        return Err(ConvertError::Usage(format!(
            "wetextprocessing: the {which} grammar `{}` is {len} bytes, over the \
             {MAX_GRAMMAR_BYTES}-byte cap. Grammars are embedded as GGUF `U8` metadata \
             arrays (the `vokra.tokenizer.model` precedent), which costs one \
             `GgufMetadataValue` enum per byte while the builder assembles — a file this \
             size would peak in the gigabytes. Real ITN/TN grammars are single-digit \
             megabytes, so this is far more likely to be the wrong file.",
            path.display()
        )));
    }
    let bytes = std::fs::read(path).map_err(ConvertError::Io)?;
    let header = OpenFstHeader::probe(&bytes).map_err(|e| {
        ConvertError::Parse(format!(
            "wetextprocessing: the {which} grammar `{}` is not a readable OpenFST binary: {e}",
            path.display()
        ))
    })?;
    Ok((bytes, header))
}

/// Embeds a grammar blob as a `U8` metadata array — the `vokra.tokenizer.model`
/// precedent (`models/whisper.rs::embed_tokenizer`).
fn embed_blob(b: &mut GgufBuilder, key: &str, bytes: &[u8]) {
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().copied().map(GgufMetadataValue::U8).collect(),
        }),
    );
}

/// Converts a WeTextProcessing grammar bundle at `input` into a GGUF at
/// `output`, resolving the language/direction from the path.
///
/// # Errors
///
/// [`ConvertError::Usage`] when the bundle layout or the prefix cannot be
/// resolved, or a grammar exceeds [`MAX_GRAMMAR_BYTES`];
/// [`ConvertError::Io`] on read/write failure; [`ConvertError::Parse`] when a
/// grammar is not an OpenFST binary; [`ConvertError::Gguf`] on assembly
/// failure.
pub fn convert_wetextprocessing_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<WeTextProcessingReport, ConvertError> {
    let parse_type = resolve_prefix(input)?;
    convert_wetextprocessing_file_with_type(input, output, license, parse_type)
}

/// [`convert_wetextprocessing_file`] with the language/direction supplied
/// explicitly instead of inferred from the path.
///
/// # Errors
///
/// As [`convert_wetextprocessing_file`], minus the prefix-resolution failures.
pub fn convert_wetextprocessing_file_with_type(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    parse_type: ItnParseType,
) -> Result<WeTextProcessingReport, ConvertError> {
    let (tagger_path, verbalizer_path) = resolve_pair(input)?;
    let (tagger, tagger_header) = read_grammar(&tagger_path, "tagger")?;
    let (verbalizer, verbalizer_header) = read_grammar(&verbalizer_path, "verbalizer")?;

    let reader_gap = tagger_header
        .vokra_reader_gap()
        .map(|g| format!("tagger: {g}"))
        .or_else(|| {
            verbalizer_header
                .vokra_reader_gap()
                .map(|g| format!("verbalizer: {g}"))
        });
    let vokra_readable = reader_gap.is_none();

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(
        chunks::KEY_MODEL_NAME,
        &format!("{NAME}-{}", parse_type.prefix().replace('_', "-")),
    );
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let effective_license = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    b.add_string(KEY_ITN_LANGUAGE, parse_type.language());
    b.add_string(KEY_ITN_DIRECTION, parse_type.direction());
    b.add_string(KEY_ITN_PREFIX, parse_type.prefix());
    b.add_metadata(
        KEY_ITN_TAGGER_BYTES,
        GgufMetadataValue::U64(tagger.len() as u64),
    );
    b.add_metadata(
        KEY_ITN_VERBALIZER_BYTES,
        GgufMetadataValue::U64(verbalizer.len() as u64),
    );
    b.add_u32(KEY_ITN_TAGGER_FLAGS, tagger_header.flags as u32);
    b.add_u32(KEY_ITN_VERBALIZER_FLAGS, verbalizer_header.flags as u32);
    b.add_metadata(
        KEY_ITN_TAGGER_STATES,
        GgufMetadataValue::I64(tagger_header.num_states),
    );
    b.add_metadata(
        KEY_ITN_VERBALIZER_STATES,
        GgufMetadataValue::I64(verbalizer_header.num_states),
    );
    b.add_bool(KEY_ITN_VOKRA_READABLE, vokra_readable);

    embed_blob(&mut b, KEY_ITN_TAGGER_FST, &tagger);
    embed_blob(&mut b, KEY_ITN_VERBALIZER_FST, &verbalizer);

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;

    Ok(WeTextProcessingReport {
        prefix: parse_type.prefix(),
        tagger_path,
        verbalizer_path,
        tagger_bytes: tagger.len(),
        verbalizer_bytes: verbalizer.len(),
        tagger_states: tagger_header.num_states,
        verbalizer_states: verbalizer_header.num_states,
        vokra_readable,
        reader_gap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch directory (PID + sequence — the sibling
    /// `gtcrn` / `sepformer` pattern; no external `tempfile` dep, so
    /// zero-dep NFR-DS-02 is preserved).
    fn scratch_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-wetext-{tag}-{}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).expect("scratch dir");
        p
    }

    /// A minimal, well-formed OpenFST `VectorFst<StdArc>` binary: header plus a
    /// one-state, zero-arc, final body.
    ///
    /// A rejection/round-trip scaffold, NOT a format oracle — the byte layout
    /// it emits is the one already byte-verified against real OpenFST 1.8.4 in
    /// `vokra-core/src/decode/wfst/reader.rs`, and the positive proof that the
    /// reader handles a real producer's output lives in
    /// `vokra-core/tests/parity_wfst.rs`.
    fn fst_bytes(flags: i32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&0x7EB2_FDD6u32.to_le_bytes()); // magic
        v.extend_from_slice(&6i32.to_le_bytes());
        v.extend_from_slice(b"vector");
        v.extend_from_slice(&8i32.to_le_bytes());
        v.extend_from_slice(b"standard");
        v.extend_from_slice(&2i32.to_le_bytes()); // version
        v.extend_from_slice(&flags.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // properties
        v.extend_from_slice(&0i64.to_le_bytes()); // start
        v.extend_from_slice(&1i64.to_le_bytes()); // num_states
        v.extend_from_slice(&0i64.to_le_bytes()); // num_arcs (header)
        v.extend_from_slice(&0.0f32.to_le_bytes()); // final weight
        v.extend_from_slice(&0i64.to_le_bytes()); // narcs
        v
    }

    /// Writes a bundle directory `<scratch>/<prefix>/{tagger,verbalizer}.fst`.
    fn bundle(tag: &str, prefix: &str, flags: i32) -> PathBuf {
        let root = scratch_dir(tag);
        let dir = root.join(prefix);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tagger.fst"), fst_bytes(flags)).unwrap();
        std::fs::write(dir.join("verbalizer.fst"), fst_bytes(0)).unwrap();
        dir
    }

    fn blob_from_gguf(file: &GgufFile, key: &str) -> Vec<u8> {
        file.get(key)
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("`{key}` present as an array"))
            .values
            .iter()
            .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
            .collect()
    }

    #[test]
    fn converts_a_bundle_directory_and_roundtrips_both_grammars() {
        let dir = bundle("roundtrip", "zh_itn", 0);
        let out = dir.join("zh_itn.gguf");
        let report = convert_wetextprocessing_file(&dir, &out, None).unwrap();

        assert_eq!(report.prefix, "zh_itn");
        assert!(report.vokra_readable);
        assert_eq!(report.reader_gap, None);
        assert_eq!(report.tagger_states, 1);

        let file = GgufFile::parse(std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("wetextprocessing-zh-itn")
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_ITN_LANGUAGE).and_then(|v| v.as_str()),
            Some("zh")
        );
        assert_eq!(
            file.get(KEY_ITN_DIRECTION).and_then(|v| v.as_str()),
            Some("itn")
        );
        assert_eq!(
            file.get(KEY_ITN_VOKRA_READABLE).and_then(|v| v.as_bool()),
            Some(true)
        );

        // The grammars must come back BYTE-IDENTICAL — a lossy embed would
        // silently corrupt an FST body.
        assert_eq!(blob_from_gguf(&file, KEY_ITN_TAGGER_FST), fst_bytes(0));
        assert_eq!(blob_from_gguf(&file, KEY_ITN_VERBALIZER_FST), fst_bytes(0));
        assert_eq!(
            file.get(KEY_ITN_TAGGER_BYTES).and_then(|v| v.as_u64()),
            Some(fst_bytes(0).len() as u64)
        );
    }

    #[test]
    fn stamps_provenance_and_the_permissive_license_class() {
        let dir = bundle("provenance", "en_itn", 0);
        let out = dir.join("o.gguf");
        convert_wetextprocessing_file(&dir, &out, None).unwrap();
        let file = GgufFile::parse(std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
        );
        assert!(
            file.get(chunks::KEY_PROVENANCE_SOURCE)
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains("WeTextProcessing"))
        );
    }

    #[test]
    fn license_override_is_honoured() {
        let dir = bundle("license-override", "ja_itn", 0);
        let out = dir.join("o.gguf");
        convert_wetextprocessing_file(&dir, &out, Some("mit")).unwrap();
        let file = GgufFile::parse(std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
    }

    #[test]
    fn a_symbol_table_grammar_still_converts_but_is_flagged_not_readable() {
        // Faithfully packaging the bytes is right; PRETENDING they are usable
        // is not. The artifact carries the truth either way.
        let dir = bundle("symtab", "zh_tn", 0x3);
        let out = dir.join("o.gguf");
        let report = convert_wetextprocessing_file(&dir, &out, None).unwrap();
        assert!(!report.vokra_readable);
        let gap = report.reader_gap.expect("a gap must be recorded");
        assert!(gap.starts_with("tagger:"), "{gap}");
        assert!(gap.contains("fstsymbols --clear_isymbols"), "{gap}");

        let file = GgufFile::parse(std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_ITN_VOKRA_READABLE).and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(
            file.get(KEY_ITN_TAGGER_FLAGS).and_then(|v| v.as_u64()),
            Some(3)
        );
    }

    #[test]
    fn flat_prefixed_tagger_file_resolves_its_verbalizer_sibling() {
        let root = scratch_dir("flat");
        std::fs::write(root.join("en_tn_tagger.fst"), fst_bytes(0)).unwrap();
        std::fs::write(root.join("en_tn_verbalizer.fst"), fst_bytes(0)).unwrap();
        let out = root.join("o.gguf");
        let report =
            convert_wetextprocessing_file(&root.join("en_tn_tagger.fst"), &out, None).unwrap();
        assert_eq!(report.prefix, "en_tn");
        assert!(report.verbalizer_path.ends_with("en_tn_verbalizer.fst"));
    }

    #[test]
    fn a_non_fst_input_is_refused_loudly() {
        let root = scratch_dir("not-fst");
        let dir = root.join("zh_itn");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tagger.fst"), b"this is not an fst").unwrap();
        std::fs::write(dir.join("verbalizer.fst"), fst_bytes(0)).unwrap();
        let Err(err) = convert_wetextprocessing_file(&dir, &root.join("o.gguf"), None) else {
            panic!("expected an error for a non-OpenFST tagger");
        };
        let msg = format!("{err}");
        assert!(msg.contains("tagger"), "{msg}");
        assert!(msg.contains("OpenFST"), "{msg}");
    }

    #[test]
    fn a_missing_verbalizer_sibling_is_refused_loudly() {
        let root = scratch_dir("missing-sibling");
        let dir = root.join("ja_tn");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tagger.fst"), fst_bytes(0)).unwrap();
        let Err(err) = convert_wetextprocessing_file(&dir, &root.join("o.gguf"), None) else {
            panic!("expected an error when verbalizer.fst is absent");
        };
        assert!(format!("{err}").contains("verbalizer"), "{err}");
    }

    #[test]
    fn an_unresolvable_prefix_is_refused_loudly() {
        let root = scratch_dir("no-prefix");
        let dir = root.join("grammars");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tagger.fst"), fst_bytes(0)).unwrap();
        std::fs::write(dir.join("verbalizer.fst"), fst_bytes(0)).unwrap();
        let Err(err) = convert_wetextprocessing_file(&dir, &root.join("o.gguf"), None) else {
            panic!("expected an error when the prefix cannot be resolved");
        };
        let msg = format!("{err}");
        assert!(msg.contains("zh_itn"), "{msg}");
        assert!(msg.contains("wetext_processor.cc"), "{msg}");
    }

    #[test]
    fn an_explicit_parse_type_bypasses_prefix_resolution() {
        let root = scratch_dir("explicit-type");
        let dir = root.join("grammars");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tagger.fst"), fst_bytes(0)).unwrap();
        std::fs::write(dir.join("verbalizer.fst"), fst_bytes(0)).unwrap();
        let report = convert_wetextprocessing_file_with_type(
            &dir,
            &root.join("o.gguf"),
            None,
            ItnParseType::JaItn,
        )
        .unwrap();
        assert_eq!(report.prefix, "ja_itn");
    }

    #[test]
    fn resolve_prefix_covers_every_upstream_layout() {
        for pt in ItnParseType::all() {
            let flat = PathBuf::from(format!("/models/{}_tagger.fst", pt.prefix()));
            assert_eq!(resolve_prefix(&flat).unwrap(), pt, "{}", flat.display());
            let bundled = PathBuf::from(format!("/cache/{}/abc123/tagger.fst", pt.prefix()));
            assert_eq!(
                resolve_prefix(&bundled).unwrap(),
                pt,
                "{}",
                bundled.display()
            );
        }
    }

    #[test]
    fn an_ambiguous_path_is_refused_rather_than_guessed() {
        let p = PathBuf::from("/cache/zh_itn/en_tn/tagger.fst");
        let Err(err) = resolve_prefix(&p) else {
            panic!("expected an error for a path with two prefixes");
        };
        assert!(format!("{err}").contains("ambiguous"), "{err}");
    }
}
