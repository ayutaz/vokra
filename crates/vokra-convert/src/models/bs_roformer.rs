#![allow(clippy::doc_lazy_continuation)]
//! **BS-Roformer / Mel-Band Roformer** (upstream unclear — Lucidrains
//! MIT code + third-party trainer checkpoints of mixed provenance):
//! safetensors → GGUF conversion (Wave 5 candidate, 2026-08-01).
//!
//! Input: a BS-Roformer or Mel-Band Roformer safetensors checkpoint
//! (Lu et al. 2023, arXiv:2310.01809 — "Music Source Separation with
//! Band-Split RoPE Transformer"). Band-Split RoPE Transformer for
//! **music source separation**: split the STFT magnitude of a mixture
//! into frequency bands, run a dual-path Transformer stack that
//! alternates time-axis and band-axis attention (each with RoPE
//! position encoding), and mask the STFT to isolate a target stem
//! (vocals is the most common publication target; drums / bass /
//! other are also possible).
//!
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream state-dict name, plus the `vokra.model.*` +
//! `vokra.provenance.*` metadata chunks a future
//! `crates/vokra-models/src/bs_roformer/` binder will read.
//!
//! # Vokra scope — music source separation (per 2026-07-30 scope expansion)
//!
//! BS-Roformer is a music source separator. Music source separation
//! was pinned in-scope by the 2026-07-30 依頼者指示「asr,tts,音楽系,
//! 音声分離など全てのモデルに対応したい」(memory
//! `[[project-scope-expansion-2026-07-30]]`). This converter stamps
//! `category = "separation"` — the tag every sibling separator (the
//! SepFormer speech-separation family, TIGER speech separator) uses,
//! so a downstream model-zoo / catalog surface sees BS-Roformer as
//! part of the separation family regardless of source domain (music
//! vocals stem vs speech speaker isolation).
//!
//! # ⚠️  Publish blocked — provenance unclear (fail-closed default)
//!
//! **Weight redistribution default is
//! [`LicenseClass::RedistributionForbidden`]**. This is a
//! defensively-strict default (`redistributable() = false`,
//! `commercial_ok() = false`) because the BS-Roformer weight
//! ecosystem is a mix that a converter cannot machine-check:
//!
//! - **Architecture / reference code** = MIT
//!   (`github.com/lucidrains/BS-RoFormer`, Phil Wang's clean-room
//!   PyTorch implementation of Lu et al. 2023 — the paper's authors
//!   never released reference weights, so every checkpoint in the
//!   wild is a downstream retraining).
//! - **Individual checkpoints** = mixed provenance. Third-party
//!   mirror `huggingface.co/chenmozhijin/BSRoformer-GGUF` aggregates
//!   converted GGUFs from multiple trainers; the underlying source
//!   `.ckpt` files ship variously under **GPL-3.0** (some
//!   Ultimate-Vocal-Remover / MDX-Net-community derivatives),
//!   **CC-BY-NC-4.0** (some MoisesDB / MusDB-fine-tune releases), or
//!   **no explicit license** (the majority — hobbyist releases). No
//!   uniform license clause covers the family.
//!
//! The `RedistributionForbidden` default is the sibling
//! [`super::vits_ja`] pattern applied to a different failure mode:
//! `vits_ja` refuses to redistribute because the training corpus
//! forbids it (JSUT / JVS terms); `bs_roformer` refuses because a
//! converter cannot know which checkpoint the caller has and thus
//! cannot know which license applies. The user overrides at the
//! outer `vokra-convert --license <spdx>` boundary when they know
//! the specific SPDX id for their checkpoint (the same Whisper /
//! kokoro / vits-ja override pattern — see `convert_file_licensed`
//! in `crates/vokra-convert/src/lib.rs`).
//!
//! **Publish is blocked at
//! `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`** —
//! there is no entry for `bs-roformer` (unlisted slug fails closed
//! as `UNKNOWN_REPO` at `publish-one.sh` gate time). An owner
//! decision selecting a specific checkpoint (and thus a specific
//! license) is the prerequisite to a first publish.
//!
//! # Scale — vast.ai handoff (~4.68 GB)
//!
//! BS-Roformer checkpoints range from ~150 MB (Mel-Band variants) to
//! ~4-5 GB (full BS-Roformer + high-band-count variants) per HF
//! mirror `chenmozhijin/BSRoformer-GGUF` file listing (2026-08-01).
//! The 4.68 GB flagship class sits just under the M1 iMac 16 GB
//! comfortable-local-convert threshold but is close enough to the
//! `[[feedback-large-models-on-vast-ai]]` ≥8 GB threshold that a
//! sharded safetensors + BF16 pass buffer could push peak resident
//! past the swap-death curve. Conversion is safer on vast.ai per
//! `docs/handoff/vast-ai-large-model-publish.md` for the top-of-
//! range variants (16 GB box sufficient); Mel-Band ~150 MB variants
//! convert locally without concern.
//!
//! # BF16 pass-through (mirror of musicgen_medium / xcodec2 / vits_ja)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm
//! — no convert-time widening. BF16 stays GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). The
//! observability counter [`BsRoformerReport::bf16_passthrough`]
//! records how many BF16 tensors landed on this arm so a silent
//! widen / downcast cannot slip in undetected. Community mixed-
//! precision retrainings routinely ship BF16, so this counter is
//! more than defensive here.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS /
//! VoxCPM / VibeVoice / vits_ja / musicgen contract). Real-weight
//! binding + Lu et al. 2023 parity is a follow-up wave gated on
//! §3.1 sign-off + a checkpoint-specific tensor-name manifest fetch
//! (multiple training pipelines apply distinct prefix
//! transformations — Lucidrains reference uses
//! `bs_roformer.transformer_blocks.*.*`, some UVR-community
//! derivatives use `model.*.transformer_blocks.*.*`). This
//! converter passes every F32 / F16 / BF16 tensor through unchanged
//! so a future `BsRoformer::from_gguf` can walk whichever names the
//! caller's checkpoint carries.
//!
//! # I64 refusal (integer tensor policy)
//!
//! The Vokra safetensors reader admits only F32 / F16 / BF16 at
//! parse time (the same policy every sibling BF16 pass-through
//! converter follows). Any I64 / I32 / I8 tensor in an upstream
//! checkpoint (position indices, buffer counters, etc.) must be
//! filtered offline via `tools/parity/bs_roformer_prepare_
//! checkpoint.py` before this converter runs — the prep script
//! drops the training-artefact integers with a warn (mirror of
//! `sepformer_prepare_checkpoint.py` / `bin_to_safetensors.py` INT
//! filter).
//!
//! # No ONNX (permanent)
//!
//! BS-Roformer trainers ship PyTorch pickle (`.ckpt`) or safetensors;
//! this converter **never** touches ONNX (FR-LD-05). The pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/bs_roformer/` module (whisper.cpp 型
//! self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Loud-partial precedent
//!
//! Real-weight forward binding is deferred: the runtime consumer
//! will walk the emitted tensor names and either succeed or fail
//! loudly per FR-EX-08. Today's converter surface is byte-exact
//! provenance + tensor-name preservation only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for BS-Roformer / Mel-Band Roformer GGUFs.
///
/// Distinct from every sibling separator arch tag
/// (`sepformer` / `tiger_separator` / `mp_senet`) because
/// BS-Roformer is a **dual-path frequency-band Transformer** over
/// STFT bands with RoPE position encoding, structurally distinct
/// from every sibling: SepFormer is a dual-path Transformer over
/// time-domain latents (not STFT bands), TIGER is a time-frequency
/// interleaved gain network, MP-SENet is a Mag-Phase parallel
/// speech-enhancement network. Silently sharing an arch tag would
/// mis-route runtime dispatch to a wrong-shape forward.
///
/// Distinct from every music-generation arch tag (`musicgen` /
/// `audioldm2`) because BS-Roformer separates a mixture into stems
/// (analysis) rather than generating audio (synthesis) — different
/// task family, different forward-pass surface.
pub const ARCH: &str = "bs_roformer";

/// `vokra.model.name` value written for the canonical BS-Roformer
/// GGUF. The specific size / band-count variant is not encoded here
/// today (the enum bloat waits for a second variant to justify it —
/// same `xcodec2` / `wavtokenizer` / `musicgen_medium` posture).
pub const NAME: &str = "bs-roformer";

/// `vokra.model.category` value — BS-Roformer is a **music source
/// separator** (mixture → vocals / drums / bass / other stems). The
/// category chunk is a taxonomy tag orthogonal to `arch`; the
/// runtime does not dispatch on it (arch does), but it is
/// machine-readable for model-zoo / catalog surfaces (see
/// `docs/license-audit.md`). Same category tag every sibling
/// separator uses (SepFormer speech-separation variants
/// `Wsj02mix` / `Libri2Mix` / `Libri3Mix`).
pub const CATEGORY: &str = "separation";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the
/// free-text `vokra.provenance.source`.
///
/// The default `chenmozhijin/BSRoformer-GGUF` is a **third-party
/// GGUF mirror** — not an official BS-Roformer release, and not
/// the upstream trainer's original checkpoint repo. A publish
/// path that ever unblocks would either (a) point at the specific
/// underlying trainer's HF repo, or (b) keep this mirror and
/// document the aggregation in `docs/license-audit.md`. Today's
/// stamp names the mirror the CLI slug refers to; the license
/// class default (`RedistributionForbidden`) is what actually
/// blocks the publish path.
pub const UPSTREAM_HF: &str = "chenmozhijin/BSRoformer-GGUF";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source` (`KEY_PROVENANCE_SOURCE`). Kept short
/// — the license machine class is carried separately in the
/// `vokra.provenance.weight_license` chunk. Mentions the
/// clean-room MIT reference (Lucidrains) that a downstream can
/// use if they want to re-train under a permissive corpus.
const UPSTREAM_SOURCE: &str = "BS-Roformer / Mel-Band Roformer (Lu et al. 2023 arXiv:2310.01809; \
     reference code `github.com/lucidrains/BS-RoFormer` MIT; weight \
     provenance mixed — third-party mirror `chenmozhijin/BSRoformer-GGUF` \
     aggregates trainer releases under GPL-3.0 / CC-BY-NC-4.0 / unclear \
     terms. Override with `--license <spdx>` if you own the checkpoint.)";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling BF16 pass-through converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a BS-Roformer conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::musicgen_large::MusicGenLargeReport`],
/// [`super::xcodec2::XCodec2Report`],
/// [`super::vits_ja::VitsJaReport`]) adapted to the file-oriented
/// `convert_bs_roformer_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BsRoformerReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting. Community mixed-precision retrainings
    /// ship BF16 routinely, so this counter is more than defensive
    /// here.
    pub bf16_passthrough: usize,
}

/// Converts a BS-Roformer / Mel-Band Roformer safetensors checkpoint
/// at `input` into a Vokra-native GGUF at `output`, returning a
/// [`BsRoformerReport`].
///
/// The upstream release ships as a torch pickle (`.ckpt`) in most
/// cases (both the Lucidrains reference training loop and
/// UVR-community derivatives). Callers pre-flatten offline via
/// `tools/parity/bs_roformer_prepare_checkpoint.py` (a thin wrapper
/// over `tools/parity/bin_to_safetensors.py` — the SBV2 v2 /
/// SpeechT5-HiFi-GAN / DeBERTa v3 large / VoxCPM-0.5B /
/// Fun-CosyVoice3 / MusicGen-Medium / MusicGen-Large precedent).
/// This function accepts safetensors only — no pickle parser enters
/// the Vokra tree (NFR-DS-02 zero-dep + FR-LD-05 no pickle in
/// runtime).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source
/// / upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`LicenseClass::RedistributionForbidden`] with the SPDX marker
/// `"weight-provenance-unclear"` — the fail-closed publish default
/// (sibling [`super::vits_ja`] uses `"corpus-restricted"` for a
/// different failure mode; here the specific failure is that a
/// converter cannot know which SPDX id applies to the caller's
/// checkpoint). A caller who knows the specific SPDX id for their
/// checkpoint (e.g. they trained it themselves on a permissive
/// corpus, or they have a specific trainer's GPL-3.0 release in
/// hand) overrides here — the same Whisper / kokoro / vits-ja /
/// xcodec2 / musicgen override pattern.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the
/// GGUF cannot be assembled.
pub fn convert_bs_roformer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<BsRoformerReport, ConvertError> {
    // NB: BS-Roformer checkpoints range from ~150 MB (Mel-Band variants)
    // to ~4-5 GB (full BS-Roformer). `std::fs::read` peaks at ~2x file
    // size in the worst case (input buffer + parsed safetensors view =
    // additive). The 4-5 GB class sits close to but not past the M1 iMac
    // 16 GB comfortable-local-convert threshold, so simple eager-read is
    // acceptable — no streaming reader needed for the current family.
    // Moshi (14 GB) remains the streaming-mandated tier.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Built-in stamp = RedistributionForbidden with a self-describing
    // SPDX marker (`weight-provenance-unclear`) that names the reason
    // this converter cannot silently label the artifact permissive.
    // The `license` argument (Some(non-empty spdx)) overrides these
    // three chunks — the empty-string case is explicitly filtered so an
    // accidental `Some("")` cannot silently downgrade the classification
    // (mirror of xcodec2 / musicgen_large / musicgen_medium empty-string
    // guard).
    //
    // `LicenseClass::RedistributionForbidden.redistributable() = false`
    // means the artifact is refused by the publish gate at
    // `LicenseClass::redistributable()` check time — the fail-closed
    // publish default is what actually blocks upload, distinct from the
    // `LicenseClass::NonCommercial.requires_research_flag = true` gate
    // that governs load-time (this class is not gated for load — the
    // JSUT / JVS derivation would let a permissively-trained checkpoint
    // load unimpeded once the license override lands, and refusing to
    // load an untrusted checkpoint the operator holds legitimately would
    // be counterproductive — the correct gate is the publish gate).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (
            "weight-provenance-unclear".to_owned(),
            LicenseClass::RedistributionForbidden,
        ),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = BsRoformerReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR (mirror of
    // musicgen_large / xcodec2 / vits_ja / neucodec / wavtokenizer);
    // the runtime widens BF16 → f32 exactly at load via the single
    // choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// A unique temp path — per-process id **plus** a monotonic counter
    /// so two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-bs-roformer-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// Encodes an f32 array as little-endian BF16 bytes (top 16 bits of
    /// the f32 pattern — the exact inverse of the runtime's
    /// `decode_bf16 : bits << 16`).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-declared dtype and raw payload.
    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Builds a two-tensor safetensors buffer (F32 first, then F16)
    /// with caller-supplied payloads.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    /// The BF16 pass-through arm must emit GGUF type 30
    /// (`GgmlType::BF16`) with byte-identical payload — mirror of the
    /// musicgen_large / xcodec2 / wavtokenizer / neucodec pin.
    /// Community mixed-precision retrainings routinely ship BF16, so
    /// this is a live-path test here rather than defensive.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity
        // assert catches any silent widen / downcast (zeroed payloads
        // would round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16 = bf16_bytes(&values);
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic BS-Roformer state-dict name — the
        // Lucidrains reference exposes the dual-path Transformer
        // block stack as `bs_roformer.transformer_blocks.*.*`.
        let input_bytes = safetensors_one(
            "bs_roformer.transformer_blocks.0.attend_time.to_qkv.weight",
            "BF16",
            &[2, 3],
            &bf16,
        );
        let input = tmp_path("bf16-in");
        let output = tmp_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_bs_roformer_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let file = GgufFile::open(&output).expect("load output gguf");
        let info = file
            .tensor_info("bs_roformer.transformer_blocks.0.attend_time.to_qkv.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// F32 + F16 mixed-dtype pass-through with the additive-default
    /// invariant on `bf16_passthrough` and all arch / provenance /
    /// category stamps — including the **critical** default
    /// RedistributionForbidden stamp (the whole point of the fail-
    /// closed publish posture vs. sibling permissive converters).
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_redistribution_forbidden() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable half-values: 1.0=0x3C00, -2.0=0xC000,
        // -0.5=0xB800, 3.0=0x4200, 0.15625=0x3100, 42.0=0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);

        // Mirror realistic BS-Roformer state-dict tensor names — the
        // Lucidrains reference exposes the band-split front-end as
        // `band_split.*` and the band-axis attention body under the
        // same transformer_blocks tree used above.
        let input_bytes = safetensors_f32_then_f16(
            "band_split.to_features.0.weight",
            &[1, 2],
            &f32_bytes,
            "bs_roformer.transformer_blocks.0.attend_freq.to_out.0.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = tmp_path("mixed-in");
        let output = tmp_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_bs_roformer_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16-only input must leave the BF16 counter at the Default 0 (additive-default invariant)"
        );

        let file = GgufFile::open(&output).expect("load output gguf");

        let f32_info = file
            .tensor_info("band_split.to_features.0.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("bs_roformer.transformer_blocks.0.attend_freq.to_out.0.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Arch / name / category / provenance chunks land with the
        // built-in fail-closed RedistributionForbidden stamp.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `separation` (music source separation, sibling of SepFormer speech-separation family)"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        // The default license path must stamp the fail-closed
        // RedistributionForbidden triple — the whole point of the
        // provenance-unclear posture. Silently defaulting to Permissive
        // or NonCommercial would be a misrepresentation because a
        // converter cannot know which specific SPDX id covers the
        // caller's checkpoint.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("weight-provenance-unclear"),
            "SPDX marker must self-describe the reason for the fail-closed default"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::RedistributionForbidden.as_str()),
            "weight_license class must default to RedistributionForbidden (unclear-provenance publish block)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// A caller-supplied `license` (e.g. the caller owns a specific
    /// trainer's release under a known SPDX id) overrides the built-in
    /// RedistributionForbidden stamp. Same override pattern as
    /// `convert_file_licensed` — the model_id / arch / category /
    /// upstream stamps survive but the license triple flips. This test
    /// pins the escape hatch a legitimate publisher takes.
    #[test]
    fn caller_license_override_swaps_the_stamp() {
        // Non-zero payloads that are NOT approximations of π/e —
        // clippy::approx_constant would flag 3.14/2.71 as a naked
        // approximation of the standard constants.
        let f32_vals: [f32; 2] = [11.5, -6.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("final_norm.weight", "F32", &[1, 2], &f32_bytes);
        let input = tmp_path("override-in");
        let output = tmp_path("override-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        // Override to MIT (Permissive) — the caller trained on a
        // permissive corpus and owns the checkpoint.
        let report = convert_bs_roformer_file(&input, &output, Some("mit")).expect("convert");
        assert_eq!(report.written, 1);

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override class must be re-derived from the SPDX id (MIT → Permissive)"
        );
        // Model id / arch / category / upstream_hf remain the built-in
        // values — the override changes only the license triple.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category (separation) must not flip when license overrides"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// fail-closed stamp — that would be a silent publish-gate
    /// downgrade (from RedistributionForbidden to whatever
    /// `from_license_str("")` decides, likely `Unknown`). The
    /// `Some(s) if !s.is_empty()` guard keeps the default
    /// RedistributionForbidden stamp (mirror of xcodec2 /
    /// wavtokenizer / musicgen_medium / musicgen_large empty-string
    /// guard test).
    #[test]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_vals: [f32; 2] = [0.5, -0.5];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("mask_estimator.weight", "F32", &[1, 2], &f32_bytes);
        let input = tmp_path("empty-in");
        let output = tmp_path("empty-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let _ = convert_bs_roformer_file(&input, &output, Some("")).expect("convert");

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("weight-provenance-unclear"),
            "empty string must NOT downgrade the license stamp — the whole point of the fail-closed default"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::RedistributionForbidden.as_str()),
            "empty string must NOT downgrade the class"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// An empty safetensors buffer produces zero reads / zero writes
    /// but still lands the full metadata stamp — the provenance stamp
    /// is independent of the tensor walk (mirror of xcodec2 /
    /// musicgen_medium / musicgen_large empty-input sanity pin).
    /// Especially load-bearing here: the fail-closed provenance stamp
    /// must land even when the input carries nothing usable.
    #[test]
    fn empty_safetensors_still_stamps_metadata() {
        // Constructing a valid empty safetensors requires the header
        // `{}` (2 bytes) prefixed by its little-endian u64 length (8
        // bytes) = 10 bytes total.
        let empty_header = "{}";
        let mut empty_safetensors = Vec::new();
        empty_safetensors.extend_from_slice(&(empty_header.len() as u64).to_le_bytes());
        empty_safetensors.extend_from_slice(empty_header.as_bytes());

        let input = tmp_path("empty-st-in");
        let output = tmp_path("empty-st-out");
        std::fs::write(&input, &empty_safetensors).expect("write empty safetensors");

        let report = convert_bs_roformer_file(&input, &output, None)
            .expect("empty safetensors must be accepted");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        // Even with zero tensors the metadata chunks still land — the
        // provenance stamp is independent of the tensor walk.
        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::RedistributionForbidden.as_str()),
            "stamp must land even with no tensors — fail-closed license posture applies unconditionally"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// Arch tag must be **distinct** from every sibling separator arch
    /// (SepFormer / TIGER / MP-SENet) — silently sharing would
    /// mis-route runtime dispatch to a wrong-shape forward
    /// (SepFormer = dual-path time-domain; TIGER = interleaved
    /// time-frequency gain; MP-SENet = magnitude-phase parallel).
    #[test]
    fn arch_is_distinct_from_sibling_separators() {
        assert_eq!(ARCH, "bs_roformer");
        assert_ne!(ARCH, "sepformer");
        assert_ne!(ARCH, "tiger_separator");
        assert_ne!(ARCH, "mp_senet");
        assert_ne!(ARCH, "metricgan_plus");
        // Distinct from music-generation arch tags (BS-Roformer
        // analyzes, MusicGen / AudioLDM 2 synthesize).
        assert_ne!(ARCH, "musicgen");
        assert_ne!(ARCH, "audioldm2");
    }
}
