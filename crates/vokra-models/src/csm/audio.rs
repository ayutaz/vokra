//! CSM audio decode chain — frame RVQ codes → `mimi_rvq` features →
//! Mimi neural decoder → PCM (M4-05-T15, gate G4 consumer).
//!
//! # What this glues (ADR M4-05 §D1-(c))
//!
//! - `vokra_ops::mimi_rvq::{mimi_rvq_decode, mimi_rvq_decode_paged,
//!   mimi_rvq_read_summed}` (M3-06 / M4-04, **RVQ parity CI green** =
//!   gate G4): codes → `[time, d_model]` **features — not PCM** (the op's
//!   own rustdoc). The streaming path uses the paged variant
//!   ([`vokra_core::cache::paged::BlockSize::Two`], 12.5 Hz audio-native
//!   — FR-OP-30) so the per-codebook feature streams live in the M3-03
//!   `[time, stream, codebook]` layout.
//! - [`MimiNeuralDecoder`] (T31〜T33): features → 24 kHz PCM.
//!
//! # Bounds (FR-EX-08)
//!
//! Every code must be `< bins` **before** decode: CSM's `audio_vocab_size`
//! may exceed the Mimi table size (special ids above the bins — ADR §D2),
//! and a special id must never silently gather row 0. The all-zero EOS
//! frame never reaches this chain (generation stops without appending).
//!
//! # GPU posture
//!
//! `HotOp::MimiRvq` is CPU-only through the Compute seam today (Metal /
//! CUDA arms = explicit `UnsupportedOp`; GPU kernels are the M4-04 /
//! follow-up lane, not taken early here). The coverage-gate consistency is
//! pinned by a test below.

use vokra_core::cache::paged::{BlockSize, PagedKvCache};
use vokra_core::{Result, VokraError};
use vokra_ops::mimi_rvq::{CodebookTable, MimiRvqAttrs, mimi_paged_dims, mimi_rvq_decode_paged};

use crate::mimi::{MimiDecoderState, MimiNeuralDecoder};

/// The assembled codes→PCM chain (RVQ lookup + neural decoder) plus the
/// paged per-codebook feature store for the streaming path.
pub struct CsmAudioDecodeChain {
    tables: Vec<CodebookTable>,
    attrs: MimiRvqAttrs,
    neural: MimiNeuralDecoder,
}

impl std::fmt::Debug for CsmAudioDecodeChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmAudioDecodeChain")
            .field("attrs", &self.attrs)
            .field("neural", &self.neural)
            .finish()
    }
}

/// Streaming state: paged feature store + neural-decoder state + the
/// pre-allocated per-frame feature row (frame loop = zero allocation,
/// FR-EX-05).
pub struct CsmAudioDecodeState {
    paged: PagedKvCache<f32>,
    neural: MimiDecoderState,
    features: Vec<f32>,
    next_t: usize,
    max_time: usize,
}

impl std::fmt::Debug for CsmAudioDecodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmAudioDecodeState")
            .field("next_t", &self.next_t)
            .field("max_time", &self.max_time)
            .finish()
    }
}

impl CsmAudioDecodeChain {
    /// Assembles the chain, cross-checking every shape seam: table count
    /// vs `attrs.n_codebooks`, table width vs the neural decoder's
    /// expected feature width.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the mismatched seam.
    pub fn new(
        tables: Vec<CodebookTable>,
        attrs: MimiRvqAttrs,
        neural: MimiNeuralDecoder,
    ) -> Result<Self> {
        if tables.len() != attrs.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio chain: {} tables != attrs.n_codebooks {}",
                tables.len(),
                attrs.n_codebooks
            )));
        }
        for (i, t) in tables.iter().enumerate() {
            if t.codebook_size != attrs.codebook_size || t.d_model != attrs.d_model {
                return Err(VokraError::InvalidArgument(format!(
                    "csm audio chain: table[{i}] is [{}, {}], attrs say [{}, {}]",
                    t.codebook_size, t.d_model, attrs.codebook_size, attrs.d_model
                )));
            }
        }
        if neural.expected_feature_dim() != attrs.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio chain: neural decoder expects {}-wide features but the RVQ \
                 tables produce {} (raw-table vs effective-table path mismatch — \
                 crate::mimi::decoder module docs)",
                neural.expected_feature_dim(),
                attrs.d_model
            )));
        }
        Ok(Self {
            tables,
            attrs,
            neural,
        })
    }

    /// The RVQ shape.
    #[must_use]
    pub fn attrs(&self) -> &MimiRvqAttrs {
        &self.attrs
    }

    /// PCM samples per frame.
    ///
    /// # Errors
    ///
    /// Propagates the neural config rate check.
    pub fn frame_hop(&self) -> Result<usize> {
        self.neural.frame_hop()
    }

    /// Fresh streaming state hosting up to `max_time` frames of paged
    /// per-codebook features ([`BlockSize::Two`] — 12.5 Hz audio-native).
    ///
    /// # Errors
    ///
    /// Propagates paged-arena allocation / config errors.
    pub fn state(&self, max_time: usize) -> Result<CsmAudioDecodeState> {
        let dims = mimi_paged_dims(&self.attrs, 1, max_time);
        Ok(CsmAudioDecodeState {
            paged: PagedKvCache::pre_allocate(dims, BlockSize::Two)?,
            neural: self.neural.state(1)?,
            features: vec![0.0; self.attrs.d_model],
            next_t: 0,
            max_time,
        })
    }

    /// Decodes one generated frame (`codes = [n_codebooks]`) into
    /// `pcm_out = [frame_hop]`: paged per-codebook write → summed read →
    /// neural decode. Allocation-free.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on out-of-range codes (checked
    /// before any table gather — module docs), shape mismatch, or a
    /// `max_time` overflow; [`VokraError::KvCacheExhausted`] surfaces
    /// verbatim from the paged arena.
    pub fn decode_frame_into(
        &self,
        state: &mut CsmAudioDecodeState,
        codes: &[u32],
        pcm_out: &mut [f32],
    ) -> Result<()> {
        if codes.len() != self.attrs.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio chain: codes len {} != n_codebooks {}",
                codes.len(),
                self.attrs.n_codebooks
            )));
        }
        // Bound check every code before any gather (special ids above the
        // Mimi bins must fail here, loudly).
        for (cb, &c) in codes.iter().enumerate() {
            if c as usize >= self.attrs.codebook_size {
                return Err(VokraError::InvalidArgument(format!(
                    "csm audio chain: code {c} (codebook {cb}) >= Mimi bins {} — a CSM \
                     special id reached the codec (FR-EX-08, no silent row-0 gather)",
                    self.attrs.codebook_size
                )));
            }
        }
        if state.next_t >= state.max_time {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio chain: frame {} >= state max_time {} (FR-EX-08 — no silent \
                 wrap-around)",
                state.next_t, state.max_time
            )));
        }
        // Paged per-codebook write (streaming variant — FR-OP-30) …
        mimi_rvq_decode_paged(
            codes,
            1,
            &self.tables,
            &self.attrs,
            0,
            &mut state.paged,
            state.next_t,
        )?;
        // … then the residual-summed read-back into the pre-allocated
        // feature row. This is the **allocation-free mirror** of
        // `mimi_rvq_read_summed` (which heap-returns its sum — fine for
        // batch, not for the frame loop); the equivalence is pinned by the
        // `alloc_free_fold_matches_read_summed` test below.
        state.features.iter_mut().for_each(|v| *v = 0.0);
        for cb in 0..self.attrs.n_codebooks {
            let (k_row, _v_row) =
                state
                    .paged
                    .read_step(0, state.next_t, 0, cb)
                    .ok_or_else(|| {
                        VokraError::InvalidArgument(format!(
                            "csm audio chain: paged feature hole at t {} codebook {cb} \
                         (decode_paged just wrote it — state corrupted?)",
                            state.next_t
                        ))
                    })?;
            for (dst, src) in state.features.iter_mut().zip(k_row.iter()) {
                *dst += *src;
            }
        }
        state.paged.advance(1);
        state.next_t += 1;
        // Neural decode: features → PCM.
        self.neural
            .decode_into(&mut state.neural, &state.features, pcm_out)
    }

    /// Convenience: fresh-state decode of `[time, n_codebooks]` codes into
    /// `time * frame_hop` PCM samples.
    ///
    /// # Errors
    ///
    /// See [`Self::decode_frame_into`].
    pub fn decode_all(&self, codes: &[u32], time: usize) -> Result<Vec<f32>> {
        if time == 0 || codes.len() != time * self.attrs.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio chain: codes len {} != time {} * n_codebooks {}",
                codes.len(),
                time,
                self.attrs.n_codebooks
            )));
        }
        let hop = self.frame_hop()?;
        let mut state = self.state(time)?;
        let mut pcm = vec![0.0f32; time * hop];
        for t in 0..time {
            let frame = &codes[t * self.attrs.n_codebooks..(t + 1) * self.attrs.n_codebooks];
            let out = &mut pcm[t * hop..(t + 1) * hop];
            self.decode_frame_into(&mut state, frame, out)?;
        }
        Ok(pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{Compute, HotOp};
    use crate::mimi::{MimiEncoder, MimiNeuralConfig};
    use vokra_core::BackendKind;
    use vokra_ops::mimi_rvq::mimi_rvq_read_summed;

    /// Chain assembled from the synthesized encoder's shared tables (the
    /// raw-table path) + a synthesized neural decoder.
    fn chain() -> CsmAudioDecodeChain {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let enc = MimiEncoder::synthesized(&cfg, 5).unwrap();
        let neural = crate::mimi::MimiNeuralDecoder::synthesized(&cfg, 9, true).unwrap();
        let attrs = MimiRvqAttrs {
            n_codebooks: cfg.quantizer.n_q,
            codebook_size: cfg.quantizer.bins,
            d_model: cfg.quantizer.dimension,
        };
        CsmAudioDecodeChain::new(enc.tables().to_vec(), attrs, neural).unwrap()
    }

    fn frame_codes(seed: u32, n_cb: usize, bins: usize) -> Vec<u32> {
        (0..n_cb)
            .map(|cb| ((seed as usize + cb * 5) % bins) as u32)
            .collect()
    }

    #[test]
    fn frame_to_pcm_path_is_green_and_deterministic() {
        let ch = chain();
        let hop = ch.frame_hop().unwrap();
        let n_cb = ch.attrs().n_codebooks;
        let bins = ch.attrs().codebook_size;
        let codes: Vec<u32> = (0..3).flat_map(|t| frame_codes(t, n_cb, bins)).collect();
        let a = ch.decode_all(&codes, 3).unwrap();
        let b = ch.decode_all(&codes, 3).unwrap();
        assert_eq!(a.len(), 3 * hop);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn streaming_frames_equal_the_batch_path() {
        let ch = chain();
        let hop = ch.frame_hop().unwrap();
        let n_cb = ch.attrs().n_codebooks;
        let bins = ch.attrs().codebook_size;
        let codes: Vec<u32> = (0..4).flat_map(|t| frame_codes(t, n_cb, bins)).collect();
        let batch = ch.decode_all(&codes, 4).unwrap();
        let mut state = ch.state(4).unwrap();
        let mut streamed = Vec::new();
        for t in 0..4 {
            let mut pcm = vec![0.0f32; hop];
            ch.decode_frame_into(&mut state, &codes[t * n_cb..(t + 1) * n_cb], &mut pcm)
                .unwrap();
            streamed.extend_from_slice(&pcm);
        }
        assert_eq!(batch, streamed, "paged streaming path == batch path");
    }

    #[test]
    fn out_of_range_code_is_rejected_before_any_gather() {
        let ch = chain();
        let hop = ch.frame_hop().unwrap();
        let mut state = ch.state(1).unwrap();
        let mut pcm = vec![0.0f32; hop];
        let mut codes = frame_codes(0, ch.attrs().n_codebooks, ch.attrs().codebook_size);
        codes[1] = ch.attrs().codebook_size as u32; // a "special id"
        let err = ch
            .decode_frame_into(&mut state, &codes, &mut pcm)
            .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(err.to_string().contains("special id"), "actionable message");
    }

    #[test]
    fn shape_seam_mismatches_are_loud_at_assembly() {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let enc = MimiEncoder::synthesized(&cfg, 5).unwrap();
        // Effective-table-path decoder (expects seanet-width features)
        // against raw quantizer-width tables must be rejected.
        let neural = crate::mimi::MimiNeuralDecoder::synthesized(&cfg, 9, false).unwrap();
        let attrs = MimiRvqAttrs {
            n_codebooks: cfg.quantizer.n_q,
            codebook_size: cfg.quantizer.bins,
            d_model: cfg.quantizer.dimension,
        };
        assert!(matches!(
            CsmAudioDecodeChain::new(enc.tables().to_vec(), attrs, neural),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn alloc_free_fold_matches_read_summed() {
        // The hot-loop fold in decode_frame_into must equal the official
        // mimi_rvq_read_summed op semantics.
        let ch = chain();
        let n_cb = ch.attrs().n_codebooks;
        let bins = ch.attrs().codebook_size;
        let codes = frame_codes(3, n_cb, bins);
        let mut state = ch.state(2).unwrap();
        mimi_rvq_decode_paged(&codes, 1, &ch.tables, ch.attrs(), 0, &mut state.paged, 0).unwrap();
        let official = mimi_rvq_read_summed(&state.paged, ch.attrs(), 0, 0).unwrap();
        let mut fold = vec![0.0f32; ch.attrs().d_model];
        for cb in 0..n_cb {
            let (k, _) = state.paged.read_step(0, 0, 0, cb).unwrap();
            for (dst, src) in fold.iter_mut().zip(k.iter()) {
                *dst += *src;
            }
        }
        assert_eq!(official, fold);
    }

    #[test]
    fn mimi_rvq_hot_op_coverage_gate_is_consistent_with_this_chain() {
        // The chain's RVQ stage is CPU-side (vokra-ops runtime fn); the
        // Compute seam must therefore accept a MimiRvq listing on CPU and
        // reject it on every GPU backend — either the coverage gate's
        // `UnsupportedOp` (feature-on builds) or `BackendUnavailable`
        // (feature-off builds). Both are explicit errors; a silent CPU
        // fallback is impossible (FR-EX-08).
        assert!(Compute::for_backend(BackendKind::Cpu, &[HotOp::MimiRvq]).is_ok());
        for backend in [BackendKind::Metal, BackendKind::Cuda, BackendKind::Vulkan] {
            assert!(
                Compute::for_backend(backend, &[HotOp::MimiRvq]).is_err(),
                "{backend:?} must not accept MimiRvq through the seam today"
            );
        }
    }
}
