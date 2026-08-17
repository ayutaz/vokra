//! **Voila** (`maitrix-org/Voila`, **MIT**, 2025) — Maitrix's full-duplex
//! speech-to-speech dialog family: safetensors → GGUF conversion.
//!
//! # Why this file exists
//!
//! The runtime binder `vokra-models::voila` landed in the Wave 9
//! 2026-08-14 audit follow-up with a strict `vokra.model.arch == "voila"`
//! gate, a `vokra-cli convert --model voila` repro command baked into two
//! of its error messages, and a `crates/vokra-cli/src/engine.rs`
//! `BOUND_ARCHES` row telling operators the model is bound. **No
//! converter shipped with it** — nothing in the tree could produce a GGUF
//! that binder accepts, so every one of those statements pointed at a
//! command that did not exist. This converter closes that gap.
//!
//! # Scope: BF16 pass-through skeleton, no topology axes
//!
//! Mirror of the sibling pass-through fleet (`llama_omni2` — the closest
//! S2S sibling — plus `facebook_denoiser` / `beats` / `eat` / `atst` /
//! `m2d` / `clap` / `emotion2vec` / `moonshine_*`): every F32 / F16 /
//! BF16 tensor is emitted **verbatim** under its upstream `state_dict`
//! name, and the only metadata written is the standard
//! `vokra.model.{arch,name,category}` +
//! `vokra.provenance.{upstream_url,weight_license,license,model_id,source}`
//! group.
//!
//! **No `vokra.voila.*` topology chunk is stamped, deliberately.** The
//! per-release axes the runtime forward will eventually need (speech
//! encoder backbone / hidden dim / layer count, LLM backbone family /
//! depth / width) are not transcribable from the primary sources the
//! binder cites, and they shift across the `Voila-base` / `Voila-chat` /
//! `Voila-audio-alpha` / `Voila-autonomous-preview` releases. Inventing
//! them here would be a fabricated axis in a redistributed artifact
//! (CLAUDE.md 「ハルシネーション厳禁」); omitting them is honest and costs
//! nothing today, because the binder does not read them.
//!
//! # Handshake with the runtime binder
//!
//! `vokra-models::voila::Voila::from_gguf` validates exactly two things:
//! `vokra.model.arch == "voila"` and a **non-empty** tensor manifest. It
//! walks no specific tensor name. A pass-through GGUF from this converter
//! therefore binds end to end, and the only thing still missing is the
//! forward itself — `Voila::converse` remains the documented loud-partial
//! (`VokraError::UnsupportedOp` naming the four deferred pieces). That is
//! now an accurate description of the tree: the model converts and binds;
//! the forward is deferred.
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] are duplicated on
//! both sides of the crate boundary (`vokra-models` cannot depend on
//! `vokra-convert` without reversing the layer stack `vokra-ops → vokra-core
//! → vokra-models → vokra-convert`). The duplication is pinned by
//! `tests::arch_name_category_and_upstream_pins_match_the_runtime_binder`
//! here and by `voila::tests::arch_name_category_and_source_pins_are_stable`
//! on the binder side — a drift on either side fails a test in the same
//! commit.
//!
//! # Provenance: `upstream_url`, not `upstream_hf`
//!
//! The verified primary source is the **GitHub reference-code repo**
//! `github.com/maitrix-org/Voila` (MIT), which is what the binder's module
//! doc and `vokra-models/src/lib.rs` both cite. The weights live on the
//! `huggingface.co/maitrix-org` org under per-release repo ids that the
//! binder explicitly flags as owner-verified-at-bind-time and liable to
//! drift, so **no single HF repo id is stamped** — writing one would
//! assert a repo id this converter has not read.
//!
//! Provenance therefore rides `vokra.provenance.upstream_url`, the
//! established GitHub-native
//! posture of `facebook_denoiser` / `nkf_aec` / `rnnoise` / `nsnet2` /
//! `beats` / `eat` / `atst` / `m2d` / `beat_this` / `mt3`. The
//! `vokra-convert` binary's `verify()` reads that key for this
//! [`crate::ModelKind`] arm, so the URL is visible in the verify line
//! rather than surfacing as `<none>` under an `upstream_hf`-only readback.
//!
//! # License and the §3.1 gate
//!
//! [`DEFAULT_LICENSE_SPDX`] is `"mit"`, mirroring what the runtime binder
//! documents (`Voila::weight_license` rustdoc: "the Voila converter stamps
//! `Permissive` (MIT — end-to-end per the `maitrix-org/Voila` repo
//! LICENSE)") and what the `vokra-models/src/lib.rs` Wave 9 marker
//! records. A caller holding the checkpoint under a different attestation
//! overrides at the `--license <spdx>` boundary.
//!
//! **Redistribution stays fail-closed regardless**: `docs/license-audit.md`
//! §3.1 carries **no Voila row at all** as of this landing, so the publish
//! chain refuses the artifact on the sign-off gate. Adding that row — and
//! signing it — is owner-only work (memory
//! `[[feedback-license-signoff-primary-source]]`); CC does not pre-fill it.
//!
//! # Input shape, and what does not exist yet
//!
//! This converter consumes a **single safetensors file**. Upstream ships a
//! PyTorch checkpoint driven by a Python pipeline, and the releases are
//! large enough that sharded safetensors are likely; a
//! `tools/parity/voila_prepare_checkpoint.py` sidecar (uv-managed Python
//! 3.12 per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`, mirror of the `llama_omni2` /
//! `firered_asr_llm_l` sidecars) would front it for shard-merge / tied-
//! tensor dedup / pickle flattening. **That sidecar does not exist yet** —
//! it is named here as the follow-up, not as a shipped tool.
//!
//! # No ONNX, no pickle (permanent)
//!
//! The runtime never touches ONNX or pickle (FR-LD-05 / NFR-DS-02); any
//! `.pt` / `.pth` flattening happens offline in the sidecar above, never
//! in this crate.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

// ---------------------------------------------------------------------------
// Contract constants — mirrored verbatim by `vokra-models::voila`. See the
// module doc "Handshake with the runtime binder" section.
// ---------------------------------------------------------------------------

/// `vokra.model.arch` value written for every Voila GGUF.
///
/// Mirrors `vokra_models::voila::ARCH`. Deliberately distinct from every
/// sibling S2S arch tag — `moshi` (Kyutai full-duplex, Mimi codec + inner
/// monologue), `csm` (Sesame CSM-1B full-duplex, Mimi codec + depth
/// transformer), `llama_omni2` (ICTNLP streaming half-duplex, Qwen2.5
/// backbone + Whisper encoder). All four sit in the S2S neighbourhood but
/// differ in duplex discipline, codec / vocoder choice and backbone
/// family, so a shared tag would misroute the runtime dispatch onto a
/// differently-shaped session manager (FR-EX-08).
pub const ARCH: &str = "voila";

/// `vokra.model.name` value written for every Voila GGUF.
///
/// Mirrors `vokra_models::voila::NAME`. A single tag covers the whole
/// family: this converter does **not** discriminate `Voila-base` /
/// `Voila-chat` / `Voila-audio-alpha` / `Voila-autonomous-preview`,
/// because it stamps no per-release axes that would differ between them.
/// A future variant-aware landing adds a `vokra.voila.variant` chunk and
/// a `NAME_PREFIX` split, the way `llama_omni2` does.
pub const NAME: &str = "voila";

/// `vokra.model.category` value — the S2S dialog family neighbourhood,
/// same tier as the sibling `moshi` / `csm` / `llama_omni2` releases.
/// Consumed by the model-card generator + zoo manifest tier gate so a
/// full-duplex dialog release is never advertised as an ASR / TTS release.
///
/// Mirrors `vokra_models::voila::CATEGORY`.
pub const CATEGORY: &str = "s2s";

/// Upstream reference-code `org/repo` slug.
///
/// Mirrors `vokra_models::voila::UPSTREAM_HF`, whose name is historical:
/// the value is the **GitHub** reference-code slug, not a HuggingFace
/// model repo id. It is not stamped as `vokra.provenance.upstream_hf` for
/// exactly that reason — see [`UPSTREAM_URL`] and the module doc
/// "Provenance" section.
pub const UPSTREAM_HF: &str = "maitrix-org/Voila";

/// Primary redistribution source, written under the
/// `vokra.provenance.upstream_url` key. The GitHub tree carrying the MIT
/// LICENSE and the reference pipeline; the per-release HF weight repo ids
/// are owner-verified at bind time and are deliberately not stamped.
///
/// Pinned as `"github.com/"` followed by [`UPSTREAM_HF`] in
/// `tests::upstream_url_is_the_github_form_of_the_upstream_slug`.
pub const UPSTREAM_URL: &str = "github.com/maitrix-org/Voila";

/// Default upstream weight licence (SPDX), resolving to
/// [`LicenseClass::Permissive`].
///
/// Mirrors what the runtime binder documents for a converter-produced
/// GGUF. Overridable at the `--license <spdx>` boundary for a caller who
/// obtained the checkpoint under a different attestation. Redistribution
/// remains gated on the `docs/license-audit.md` §3.1 sign-off, which has
/// no Voila row yet (owner follow-up — CC does not sign).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// `vokra.model.category` metadata key. Kept as a converter-local constant
/// per the established `facebook_denoiser` / `llama_omni2` / `clap`
/// convention (not yet centralized in `vokra_core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source URL
/// for GitHub-native releases, parallel to the HF-hosted
/// `vokra.provenance.upstream_hf` key. Same convention as
/// `facebook_denoiser` / `nkf_aec` / `rnnoise` / `nsnet2` / `beats`.
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Human-readable `vokra.provenance.source` string. Names the upstream,
/// the model class, the default SPDX, and the fact that the §3.1 row is
/// still missing — so an operator inspecting a stray GGUF sees the
/// redistribution posture without leaving the artifact.
const UPSTREAM_SOURCE: &str = "github.com/maitrix-org/Voila (Maitrix Voila full-duplex \
     speech-to-speech dialog family, 2025, mit — BF16 pass-through skeleton, no \
     vokra.voila.* topology axes stamped; docs/license-audit.md §3.1 row absent, \
     owner sign-off required before redistribution)";

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Outcome of a Voila conversion.
///
/// Mirrors the sibling BF16 pass-through counter shape, so the invariant
/// `read == written + skipped_non_float` is auditable at the report level
/// without re-opening the artifact.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VoilaReport {
    /// Total tensor entries observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all ride the same
    /// byte-copy pass-through arm).
    pub written: usize,
    /// Non-float tensors skipped. Defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time, so a
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for parity with the sibling converters.
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16 →
    /// f32 losslessly at load through the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
    /// (`bits << 16` is exact). A silent widen / downcast regression
    /// surfaces as this counter drifting from the input BF16 count.
    pub bf16_passthrough: usize,
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

/// Converts a Voila safetensors checkpoint at `input` into a Vokra-native
/// GGUF at `output`, returning a [`VoilaReport`].
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// `state_dict` name. The `vokra.model.{arch,name,category}` and
/// `vokra.provenance.{upstream_url,weight_license,license,model_id,source}`
/// chunks are stamped for the M2-13 runtime compliance gate; no
/// `vokra.voila.*` topology chunk is written (see the module doc — the
/// per-release axes are not transcribable and the binder does not read
/// them).
///
/// `license` optionally overrides the stamped weight licence (raw SPDX
/// string; the [`LicenseClass`] is re-derived through
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"mit"`), which resolves to
/// [`LicenseClass::Permissive`].
///
/// # Errors
///
/// - [`ConvertError::Io`] when reading `input` or writing `output` fails.
/// - [`ConvertError::Parse`] when `input` is not valid safetensors.
/// - [`ConvertError::Gguf`] when GGUF assembly fails.
///
/// # Memory footprint
///
/// The whole checkpoint is buffered (`std::fs::read`) and the GGUF is
/// assembled in memory before the write, the same posture as the sibling
/// pass-through converters. Per memory `[[feedback-large-models-on-vast-ai]]`
/// a multi-GB Voila release is converted on a rented vast.ai box rather
/// than the 16 GB M1 iMac; a streaming pass (the Voxtral / Moshi
/// `GgufStreamWriter` posture) is the follow-up if a smaller instance
/// class becomes the constraint.
pub fn convert_voila_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<VoilaReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );

    let mut report = VoilaReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Cross-crate constant pins + metadata round-trip + BF16 byte-identity
    //! + loud-failure negative space for the Voila converter.

    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanos + caller tag). Mirror of
    /// the sibling `facebook_denoiser` / `llama_omni2` fixture posture —
    /// no external `tempfile` dependency, preserving zero-dep NFR-DS-02.
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-voila-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    /// RAII cleanup so a failing test does not leak scratch files.
    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Wraps a safetensors header + payload into the on-disk framing.
    fn safetensors_blob(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// One F32 tensor, shape `[2, 3]` (6 elements x 4 bytes = 24 bytes).
    fn one_f32_checkpoint() -> Vec<u8> {
        safetensors_blob(
            r#"{"backbone.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &[0u8; 24],
        )
    }

    /// BF16 payload with distinct non-zero bit patterns, so a byte-identity
    /// assert catches any silent widen / downcast attempt.
    fn bf16_payload() -> Vec<u8> {
        [1.0_f32, -2.5, 0.15625, 3.5, -0.5, 42.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Reads a string metadata key out of an emitted GGUF, borrowing from
    /// the file image. `<none>` stands in for an absent key so a failing
    /// assert reports the absence instead of panicking on an unwrap.
    fn read_str<'a>(file: &'a GgufFile, key: &str) -> &'a str {
        file.get(key).and_then(|v| v.as_str()).unwrap_or("<none>")
    }

    // -----------------------------------------------------------------------
    // 1 — Cross-crate constant pins
    // -----------------------------------------------------------------------

    /// The four contract constants must match `vokra_models::voila`'s
    /// mirrored copies byte for byte. The two crates share only
    /// `vokra-core`, so these literals are the entire handshake: a drift on
    /// either side fails here or in
    /// `voila::tests::arch_name_category_and_source_pins_are_stable`.
    #[test]
    fn arch_name_category_and_upstream_pins_match_the_runtime_binder() {
        assert_eq!(ARCH, "voila");
        assert_eq!(NAME, "voila");
        assert_eq!(CATEGORY, "s2s");
        assert_eq!(UPSTREAM_HF, "maitrix-org/Voila");
        // Distinct from every sibling S2S arch tag — silent aliasing would
        // misroute the runtime dispatch onto a differently-shaped session
        // manager (FR-EX-08).
        for sibling in ["moshi", "csm", "llama_omni2"] {
            assert_ne!(
                ARCH, sibling,
                "voila arch must stay distinct from sibling S2S arch `{sibling}`"
            );
        }
    }

    /// [`UPSTREAM_URL`] is the GitHub form of [`UPSTREAM_HF`]. Pins the
    /// module doc's claim that the mirrored `UPSTREAM_HF` constant is a
    /// reference-code slug rather than a HuggingFace repo id — if a future
    /// wave repoints one constant at a real HF release, this fires and
    /// forces the other to be reconsidered in the same commit.
    #[test]
    fn upstream_url_is_the_github_form_of_the_upstream_slug() {
        assert_eq!(
            UPSTREAM_URL.strip_prefix("github.com/"),
            Some(UPSTREAM_HF),
            "UPSTREAM_URL must be the GitHub form of the mirrored UPSTREAM_HF slug"
        );
    }

    // -----------------------------------------------------------------------
    // 2 — Metadata round-trip
    // -----------------------------------------------------------------------

    /// A converted GGUF carries arch / name / category / upstream_url plus
    /// the default `mit` + `Permissive` provenance stamp, and does **not**
    /// carry an `upstream_hf` key (the module doc's honesty invariant: no
    /// HF repo id is asserted).
    #[test]
    fn round_trip_carries_arch_category_and_github_provenance() {
        let input = scratch_path("meta-in", "safetensors");
        let output = scratch_path("meta-out", "gguf");
        std::fs::write(&input, one_f32_checkpoint()).expect("write input");
        let _in_guard = TempFileGuard(input.clone());
        let _out_guard = TempFileGuard(output.clone());

        let report = convert_voila_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse");
        assert_eq!(read_str(&file, chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(&file, chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(&file, KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(&file, KEY_PROVENANCE_UPSTREAM_URL), UPSTREAM_URL);
        assert_eq!(
            read_str(&file, chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(&file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Permissive.as_str()
        );
        // No HF repo id is asserted anywhere in the artifact — the weights'
        // per-release HF repo is owner-verified at bind time.
        assert!(
            file.get("vokra.provenance.upstream_hf").is_none(),
            "voila must not stamp an unverified upstream_hf repo id"
        );
    }

    /// The emitted artifact satisfies both gates the runtime binder
    /// enforces: `vokra.model.arch == "voila"` and a non-empty tensor
    /// manifest. This is the property that makes the `BOUND_ARCHES` row and
    /// the binder's `vokra-cli convert --model voila` repro text true; it
    /// is asserted structurally here because `vokra-convert` cannot depend
    /// on `vokra-models` (that edge would reverse the layer stack).
    #[test]
    fn emitted_gguf_satisfies_both_runtime_binder_gates() {
        let input = scratch_path("binder-in", "safetensors");
        let output = scratch_path("binder-out", "gguf");
        std::fs::write(&input, one_f32_checkpoint()).expect("write input");
        let _in_guard = TempFileGuard(input.clone());
        let _out_guard = TempFileGuard(output.clone());

        convert_voila_file(&input, &output, None).expect("convert");
        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse");
        // Gate 1: strict arch match.
        assert_eq!(read_str(&file, chunks::KEY_MODEL_ARCH), "voila");
        // Gate 2: non-empty tensor manifest (an all-zero forward is refused
        // by `VoilaWeights::from_gguf`).
        assert!(
            !file.tensors().is_empty(),
            "the binder refuses a zero-tensor voila GGUF (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 3 — BF16 byte identity
    // -----------------------------------------------------------------------

    /// BF16 reaches the pass-through arm and survives byte for byte as GGUF
    /// type 30. Regression guard for the standing "no convert-time widening"
    /// invariant (mirror of `llama_omni2` / `facebook_denoiser`).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let payload = bf16_payload();
        assert_eq!(payload.len(), 12, "6 elements x 2 bytes BF16");
        let blob = safetensors_blob(
            r#"{"backbone.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#,
            &payload,
        );

        let input = scratch_path("bf16-in", "safetensors");
        let output = scratch_path("bf16-out", "gguf");
        std::fs::write(&input, blob).expect("write input");
        let _in_guard = TempFileGuard(input.clone());
        let _out_guard = TempFileGuard(output.clone());

        let report = convert_voila_file(&input, &output, None).expect("convert BF16");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse");
        let info = file
            .tensor_info("backbone.embed_tokens.weight")
            .expect("tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays GGUF type 30"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "BF16 payload must be byte-identical to the input"
        );
    }

    // -----------------------------------------------------------------------
    // 4 — Licence override
    // -----------------------------------------------------------------------

    /// An explicit `--license` override threads all the way into the
    /// provenance stamp, and the class is re-derived rather than kept at
    /// the default. `cc-by-nc-4.0` is used precisely because it resolves to
    /// a *stricter* class than the default, so a silent downgrade back to
    /// `Permissive` would fail here.
    #[test]
    fn license_override_threads_into_provenance_and_reclassifies() {
        let input = scratch_path("license-in", "safetensors");
        let output = scratch_path("license-out", "gguf");
        std::fs::write(&input, one_f32_checkpoint()).expect("write input");
        let _in_guard = TempFileGuard(input.clone());
        let _out_guard = TempFileGuard(output.clone());

        convert_voila_file(&input, &output, Some("cc-by-nc-4.0")).expect("convert");
        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse");
        assert_eq!(
            read_str(&file, chunks::KEY_PROVENANCE_LICENSE),
            "cc-by-nc-4.0"
        );
        assert_eq!(
            read_str(&file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::from_license_str("cc-by-nc-4.0").as_str(),
            "the class must be re-derived from the override, never left at the default"
        );
        assert_ne!(
            read_str(&file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Permissive.as_str(),
            "a stricter override must not silently keep the Permissive default"
        );
    }

    // -----------------------------------------------------------------------
    // 5 — Loud failure negative space
    // -----------------------------------------------------------------------

    /// Malformed input surfaces as `ConvertError::Parse`, never as a
    /// silently-empty successful conversion (FR-EX-08).
    #[test]
    fn malformed_input_returns_parse_error() {
        let output = scratch_path("malformed-out", "gguf");
        let _out_guard = TempFileGuard(output.clone());

        // Empty buffer.
        let empty = scratch_path("malformed-empty", "safetensors");
        std::fs::write(&empty, Vec::<u8>::new()).expect("write empty");
        let _empty_guard = TempFileGuard(empty.clone());
        let err = convert_voila_file(&empty, &output, None).expect_err("empty must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Header length claims far more bytes than the file carries.
        let truncated = scratch_path("malformed-trunc", "safetensors");
        let mut buf = Vec::new();
        buf.extend_from_slice(&1024u64.to_le_bytes());
        buf.extend_from_slice(b"{}");
        std::fs::write(&truncated, buf).expect("write truncated");
        let _trunc_guard = TempFileGuard(truncated.clone());
        let err =
            convert_voila_file(&truncated, &output, None).expect_err("truncated must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }

    /// A missing input file is `ConvertError::Io`, not a panic and not a
    /// zero-tensor success.
    #[test]
    fn missing_input_returns_io_error() {
        let missing = scratch_path("absent-in", "safetensors");
        let output = scratch_path("absent-out", "gguf");
        let _out_guard = TempFileGuard(output.clone());
        let err =
            convert_voila_file(&missing, &output, None).expect_err("absent input must be rejected");
        assert!(
            matches!(err, ConvertError::Io(_)),
            "expected ConvertError::Io, got {err:?}"
        );
    }

    /// A zero-tensor checkpoint still produces a fully stamped GGUF. The
    /// converter does not second-guess the input here — the refusal belongs
    /// to the runtime binder, whose `VoilaWeights::from_gguf` rejects a
    /// zero-tensor manifest so an all-zero forward can never bind. Pinning
    /// this keeps the division of labour explicit: the converter stamps
    /// provenance even on a degenerate artifact, so the binder's refusal is
    /// a *license-classified* refusal rather than a bare parse failure.
    #[test]
    fn zero_tensor_input_still_stamps_provenance_for_the_binder_to_refuse() {
        let input = scratch_path("zero-in", "safetensors");
        let output = scratch_path("zero-out", "gguf");
        std::fs::write(&input, safetensors_blob("{}", &[])).expect("write input");
        let _in_guard = TempFileGuard(input.clone());
        let _out_guard = TempFileGuard(output.clone());

        let report = convert_voila_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);

        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse");
        assert_eq!(read_str(&file, chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(
            read_str(&file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Permissive.as_str()
        );
        assert!(
            file.tensors().is_empty(),
            "the degenerate artifact carries no tensors — the binder refuses it at load"
        );
    }

    // -----------------------------------------------------------------------
    // 6 — Topology-axis abstention
    // -----------------------------------------------------------------------

    /// No `vokra.voila.*` chunk is written. The per-release encoder /
    /// backbone axes are not transcribable from the primary sources, so
    /// stamping a guessed value into a redistributed artifact would be a
    /// fabrication (CLAUDE.md 「ハルシネーション厳禁」). If a future wave
    /// lands real axes it must delete this test deliberately, not drift
    /// past it.
    #[test]
    fn no_voila_topology_axes_are_fabricated() {
        let input = scratch_path("axes-in", "safetensors");
        let output = scratch_path("axes-out", "gguf");
        std::fs::write(&input, one_f32_checkpoint()).expect("write input");
        let _in_guard = TempFileGuard(input.clone());
        let _out_guard = TempFileGuard(output.clone());

        convert_voila_file(&input, &output, None).expect("convert");
        let file = GgufFile::parse(std::fs::read(&output).expect("read gguf")).expect("parse");
        let fabricated: Vec<&str> = file
            .metadata()
            .iter()
            .map(|(key, _)| key.as_str())
            .filter(|key| key.starts_with("vokra.voila."))
            .collect();
        assert!(
            fabricated.is_empty(),
            "no vokra.voila.* axis may be stamped without a primary source, found {fabricated:?}"
        );
    }
}
