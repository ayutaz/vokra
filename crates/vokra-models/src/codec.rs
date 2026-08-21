//! Standalone codec GGUF binders (M4-04 T10/T11) — dumb, validation-heavy
//! bridges from a codec GGUF to the `vokra-ops` RVQ decode inputs.
//!
//! The converter is the offline-math home (weight-norm folding, Mimi
//! `embedding_sum / clamp(cluster_usage)` + pre-projection — ADR M4-04 §D-f);
//! this module only **binds** the derived tensors:
//!
//! - [`MimiCodecGguf::from_gguf`] — `vokra.mimi.*` metadata +
//!   `vokra.mimi.codebook_tables` → `Vec<CodebookTable>` + [`MimiRvqAttrs`];
//! - [`DacCodecGguf::from_gguf`] — `vokra.dac.*` metadata +
//!   `vokra.dac.quantizer.{i}.*` → low-dim tables + [`DacOutProj`]s +
//!   [`DacRvqAttrs`].
//!
//! Living in `vokra-models` keeps the dependency direction intact:
//! `vokra-ops` never learns about GGUF (its ops take plain slices), and the
//! GGUF reader lives in `vokra-core` (ADR M4-04 §D-f; the same reasoning as
//! M3-06's "keep the helper in vokra-ops so the crate edge does not
//! reverse", now one level up).
//!
//! Every missing key / tensor / dtype / shape mismatch is an explicit
//! [`VokraError::ModelLoad`] (FR-EX-08 — a codec GGUF that half-loads would
//! corrupt the feature stream plausibly).
//!
//! Note: this binder does **not** run the M2-13 weight-license gate itself —
//! callers loading untrusted GGUFs go through the usual
//! `vokra_core::check_weight_license` path first (Mimi is
//! `AttributionRequired` = admitted with attribution; DAC is `Permissive`).

use std::sync::Arc;
use vokra_core::gguf::{GgmlType, GgufFile};

use vokra_core::{CodecDecoderEngine, CodecDecoderHandle, Result, VokraError};
use vokra_ops::{CodebookTable, DacOutProj, DacRvqAttrs, MimiRvqAttrs};

use crate::mimi::{MimiDecoderState, MimiNeuralConfig, MimiNeuralDecoder};

/// Reads a `u32` metadata key or fails loudly.
fn get_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(v) => v.as_u64().map(|x| x as u32).ok_or_else(|| {
            VokraError::ModelLoad(format!("codec GGUF: metadata `{key}` is not an integer"))
        }),
        None => Err(VokraError::ModelLoad(format!(
            "codec GGUF: required metadata `{key}` missing (was this GGUF produced by \
             `vokra-cli convert --model mimi|dac`?)"
        ))),
    }
}

/// Reads an F32 tensor's raw data + dimensions or fails loudly.
fn f32_tensor<'a>(file: &'a GgufFile, name: &str) -> Result<(Vec<u64>, &'a [u8])> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("codec GGUF: required tensor `{name}` missing"))
    })?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "codec GGUF: tensor `{name}` must be F32, got {:?}",
            info.dtype
        )));
    }
    let data = file
        .tensor_data(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("codec GGUF: tensor `{name}` has no data")))?;
    Ok((info.dimensions.clone(), data))
}

fn le_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Mimi
// ---------------------------------------------------------------------------

/// A standalone Mimi codec GGUF bound to its RVQ decode inputs.
#[derive(Debug, Clone)]
pub struct MimiCodecGguf {
    /// Shape attributes (from `vokra.mimi.*` metadata).
    pub attrs: MimiRvqAttrs,
    /// One effective (pre-projected) table per codebook, semantic first.
    pub tables: Vec<CodebookTable>,
}

impl MimiCodecGguf {
    /// Binds `vokra.mimi.*` + the derived `vokra.mimi.codebook_tables`
    /// tensor. Zero math — the converter already derived everything.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_codebooks = get_u32(file, "vokra.mimi.n_codebooks")? as usize;
        let codebook_size = get_u32(file, "vokra.mimi.codebook_size")? as usize;
        let d_model = get_u32(file, "vokra.mimi.d_model")? as usize;
        let attrs = MimiRvqAttrs {
            n_codebooks,
            codebook_size,
            d_model,
        };
        if n_codebooks == 0 || codebook_size == 0 || d_model == 0 {
            return Err(VokraError::ModelLoad(
                "mimi codec GGUF: vokra.mimi.* metadata has a zero axis".to_owned(),
            ));
        }

        let (dims, raw) = f32_tensor(file, "vokra.mimi.codebook_tables")?;
        let want = vec![n_codebooks as u64, codebook_size as u64, d_model as u64];
        if dims != want {
            return Err(VokraError::ModelLoad(format!(
                "mimi codec GGUF: vokra.mimi.codebook_tables dims {dims:?} != metadata {want:?}"
            )));
        }
        let vals = le_f32s(raw);
        let per_table = codebook_size * d_model;
        if vals.len() != n_codebooks * per_table {
            return Err(VokraError::ModelLoad(format!(
                "mimi codec GGUF: codebook_tables has {} f32s, expected {}",
                vals.len(),
                n_codebooks * per_table
            )));
        }
        let mut tables = Vec::with_capacity(n_codebooks);
        for cb in 0..n_codebooks {
            let slice = vals[cb * per_table..(cb + 1) * per_table].to_vec();
            tables.push(CodebookTable::new(codebook_size, d_model, slice)?);
        }
        Ok(Self { attrs, tables })
    }
}

/// Complete standalone Mimi token-to-PCM engine used by the generic streaming
/// codec surface. The immutable tables and neural weights are shared by every
/// opened handle; causal state and scratch buffers are handle-local.
#[derive(Clone)]
pub struct MimiStreamingCodec {
    tables: Arc<Vec<CodebookTable>>,
    attrs: MimiRvqAttrs,
    decoder: Arc<MimiNeuralDecoder>,
    sample_rate: u32,
    frame_hop: usize,
}

impl std::fmt::Debug for MimiStreamingCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimiStreamingCodec")
            .field("attrs", &self.attrs)
            .field("sample_rate", &self.sample_rate)
            .field("frame_hop", &self.frame_hop)
            .finish_non_exhaustive()
    }
}

impl MimiStreamingCodec {
    /// Binds a standalone Mimi GGUF into a complete streaming decoder.
    ///
    /// The standalone converter emits effective (already output-projected)
    /// codebook tables, so their width must equal the neural decoder's input
    /// width. A mixed or partial GGUF is rejected before a handle is exposed.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = MimiNeuralConfig::from_gguf(file)?;
        config.validate()?;
        let codec = MimiCodecGguf::from_gguf(file)?;
        let decoder = MimiNeuralDecoder::from_gguf(file, &config)?;
        Self::new(codec, decoder, config.sample_rate)
    }

    /// Builds the engine from already-bound components. Public to support
    /// deterministic synthesized-weight tests without fabricating source
    /// checkpoint provenance.
    pub fn new(codec: MimiCodecGguf, decoder: MimiNeuralDecoder, sample_rate: u32) -> Result<Self> {
        if codec.attrs.n_codebooks == 0 || codec.tables.len() != codec.attrs.n_codebooks {
            return Err(VokraError::ModelLoad(format!(
                "mimi streaming codec: {} tables != n_codebooks {}",
                codec.tables.len(),
                codec.attrs.n_codebooks
            )));
        }
        if codec.attrs.d_model != decoder.expected_feature_dim() {
            return Err(VokraError::ModelLoad(format!(
                "mimi streaming codec: effective codebook width {} != decoder input width {}",
                codec.attrs.d_model,
                decoder.expected_feature_dim()
            )));
        }
        if decoder.config().quantizer.n_q != codec.attrs.n_codebooks {
            return Err(VokraError::ModelLoad(format!(
                "mimi streaming codec: decoder n_q {} != codebook count {}",
                decoder.config().quantizer.n_q,
                codec.attrs.n_codebooks
            )));
        }
        if decoder.config().quantizer.bins != codec.attrs.codebook_size {
            return Err(VokraError::ModelLoad(format!(
                "mimi streaming codec: decoder bins {} != codebook size {}",
                decoder.config().quantizer.bins,
                codec.attrs.codebook_size
            )));
        }
        if sample_rate == 0 || sample_rate != decoder.config().sample_rate {
            return Err(VokraError::ModelLoad(format!(
                "mimi streaming codec: sample rate {sample_rate} != decoder sample rate {}",
                decoder.config().sample_rate
            )));
        }
        let frame_hop = decoder.frame_hop()?;
        Ok(Self {
            tables: Arc::new(codec.tables),
            attrs: codec.attrs,
            decoder: Arc::new(decoder),
            sample_rate,
            frame_hop,
        })
    }
}

impl CodecDecoderEngine for MimiStreamingCodec {
    fn open_decoder(&self) -> Result<Box<dyn CodecDecoderHandle + Send>> {
        Ok(Box::new(MimiStreamingDecoder {
            state: self.decoder.state(1)?,
            features: vec![0.0; self.attrs.d_model],
            pcm: vec![0.0; self.frame_hop],
            pending: false,
            tables: Arc::clone(&self.tables),
            attrs: self.attrs,
            decoder: Arc::clone(&self.decoder),
            sample_rate: self.sample_rate,
            frame_hop: self.frame_hop,
        }))
    }
}

/// One independently stateful Mimi decoder handle.
struct MimiStreamingDecoder {
    // Drop state/scratch before the shared immutable model fields below.
    state: MimiDecoderState,
    features: Vec<f32>,
    pcm: Vec<f32>,
    pending: bool,
    tables: Arc<Vec<CodebookTable>>,
    attrs: MimiRvqAttrs,
    decoder: Arc<MimiNeuralDecoder>,
    sample_rate: u32,
    frame_hop: usize,
}

impl CodecDecoderHandle for MimiStreamingDecoder {
    fn frame_hop(&self) -> usize {
        self.frame_hop
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn n_codebooks(&self) -> usize {
        self.attrs.n_codebooks
    }

    // ZERO-ALLOC-BEGIN — warmed successful code-frame decode; scratch and
    // causal state are allocated by open/reset. Guarded by
    // scripts/check-hot-path-allocs.sh and the counting-allocator test below.
    fn push_codes(&mut self, codes: &[u32]) -> Result<usize> {
        if codes.len() != self.attrs.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "mimi codec push: n_codebooks {} != checkpoint {}",
                codes.len(),
                self.attrs.n_codebooks
            )));
        }
        if self.pending {
            return Err(VokraError::InvalidArgument(
                "mimi codec push: pull the pending PCM frame before pushing another code frame"
                    .into(),
            ));
        }

        self.features.fill(0.0);
        for (cb, &index) in codes.iter().enumerate() {
            let row = self.tables[cb].row(index)?;
            for (dst, src) in self.features.iter_mut().zip(row) {
                *dst += *src;
            }
        }
        self.decoder
            .decode_into(&mut self.state, &self.features, &mut self.pcm)?;
        self.pending = true;
        Ok(1)
    }

    fn pull_pcm(&mut self, out: &mut [f32]) -> Result<usize> {
        if !self.pending {
            return Ok(0);
        }
        if out.len() < self.frame_hop {
            return Err(VokraError::InvalidArgument(format!(
                "mimi codec pull: output capacity {} < frame hop {}",
                out.len(),
                self.frame_hop
            )));
        }
        out[..self.frame_hop].copy_from_slice(&self.pcm);
        self.pending = false;
        Ok(self.frame_hop)
    }
    // ZERO-ALLOC-END

    fn reset(&mut self) -> Result<()> {
        self.state = self.decoder.state(1)?;
        self.features.fill(0.0);
        self.pcm.fill(0.0);
        self.pending = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DAC
// ---------------------------------------------------------------------------

/// A standalone DAC codec GGUF bound to its factorized RVQ decode inputs.
#[derive(Debug, Clone)]
pub struct DacCodecGguf {
    /// Shape attributes (from `vokra.dac.*` metadata).
    pub attrs: DacRvqAttrs,
    /// One low-dim codebook per quantizer (`[codebook_size, codebook_dim]`).
    pub tables: Vec<CodebookTable>,
    /// One weight-norm-folded output projection per quantizer.
    pub out_projs: Vec<DacOutProj>,
    /// Model sample rate (`vokra.dac.sample_rate`).
    pub sample_rate: u32,
    /// Encoder hop length (`vokra.dac.hop_length`) — frame rate =
    /// `sample_rate / hop_length` (24 kHz variant: 24000/320 = 75 Hz).
    pub hop_length: u32,
}

impl DacCodecGguf {
    /// Binds `vokra.dac.*` + the derived per-quantizer decode tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_codebooks = get_u32(file, "vokra.dac.n_codebooks")? as usize;
        let codebook_size = get_u32(file, "vokra.dac.codebook_size")? as usize;
        let codebook_dim = get_u32(file, "vokra.dac.codebook_dim")? as usize;
        let d_model = get_u32(file, "vokra.dac.d_model")? as usize;
        let sample_rate = get_u32(file, "vokra.dac.sample_rate")?;
        let hop_length = get_u32(file, "vokra.dac.hop_length")?;
        let attrs = DacRvqAttrs {
            n_codebooks,
            codebook_size,
            codebook_dim,
            d_model,
        };
        if n_codebooks == 0 || codebook_size == 0 || codebook_dim == 0 || d_model == 0 {
            return Err(VokraError::ModelLoad(
                "dac codec GGUF: vokra.dac.* metadata has a zero axis".to_owned(),
            ));
        }

        let mut tables = Vec::with_capacity(n_codebooks);
        let mut out_projs = Vec::with_capacity(n_codebooks);
        for i in 0..n_codebooks {
            let (cb_dims, cb_raw) = f32_tensor(file, &format!("vokra.dac.quantizer.{i}.codebook"))?;
            if cb_dims != vec![codebook_size as u64, codebook_dim as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "dac codec GGUF: quantizer {i} codebook dims {cb_dims:?} != \
                     [{codebook_size}, {codebook_dim}]"
                )));
            }
            tables.push(CodebookTable::new(
                codebook_size,
                codebook_dim,
                le_f32s(cb_raw),
            )?);

            let (w_dims, w_raw) =
                f32_tensor(file, &format!("vokra.dac.quantizer.{i}.out_proj_weight"))?;
            if w_dims != vec![d_model as u64, codebook_dim as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "dac codec GGUF: quantizer {i} out_proj_weight dims {w_dims:?} != \
                     [{d_model}, {codebook_dim}]"
                )));
            }
            let (b_dims, b_raw) =
                f32_tensor(file, &format!("vokra.dac.quantizer.{i}.out_proj_bias"))?;
            if b_dims != vec![d_model as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "dac codec GGUF: quantizer {i} out_proj_bias dims {b_dims:?} != [{d_model}]"
                )));
            }
            out_projs.push(DacOutProj::new(
                d_model,
                codebook_dim,
                le_f32s(w_raw),
                le_f32s(b_raw),
            )?);
        }
        Ok(Self {
            attrs,
            tables,
            out_projs,
            sample_rate,
            hop_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_ops::mimi_rvq_decode;

    /// A hand-assembled Mimi codec GGUF (bypassing the converter — the
    /// converter e2e lives in tests/codec_gguf_roundtrip.rs).
    fn mimi_gguf(n_cb: u32, cb_size: u32, d_model: u32) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.mimi.n_codebooks", n_cb);
        b.add_u32("vokra.mimi.codebook_size", cb_size);
        b.add_u32("vokra.mimi.d_model", d_model);
        let n = (n_cb * cb_size * d_model) as usize;
        let vals: Vec<u8> = (0..n)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        b.add_tensor(
            "vokra.mimi.codebook_tables",
            GgmlType::F32,
            vec![n_cb as u64, cb_size as u64, d_model as u64],
            vals,
        )
        .unwrap();
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn mimi_binder_splits_tables_per_codebook() {
        let file = mimi_gguf(2, 3, 4);
        let codec = MimiCodecGguf::from_gguf(&file).expect("bind");
        assert_eq!(codec.attrs.n_codebooks, 2);
        assert_eq!(codec.tables.len(), 2);
        assert_eq!(codec.tables[0].codebook_size, 3);
        assert_eq!(codec.tables[0].d_model, 4);
        // Table 1's first element continues the ramp where table 0 ended.
        assert_eq!(codec.tables[1].data[0], (3 * 4) as f32 * 0.5);
    }

    #[test]
    fn mimi_binder_rejects_missing_or_mismatched_pieces() {
        // Missing metadata key.
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.mimi.n_codebooks", 2);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            MimiCodecGguf::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));

        // Dims mismatch between metadata and tensor.
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.mimi.n_codebooks", 2);
        b.add_u32("vokra.mimi.codebook_size", 3);
        b.add_u32("vokra.mimi.d_model", 4);
        b.add_tensor(
            "vokra.mimi.codebook_tables",
            GgmlType::F32,
            vec![1, 3, 4],
            vec![0u8; 3 * 4 * 4],
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            MimiCodecGguf::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    fn synthesized_streaming_codec(seed: u64) -> (MimiStreamingCodec, MimiCodecGguf) {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let tables = (0..cfg.quantizer.n_q)
            .map(|cb| {
                let values = (0..cfg.quantizer.bins * cfg.seanet.dimension)
                    .map(|i| ((cb * 97 + i) as f32 - 40.0) * 0.002)
                    .collect();
                CodebookTable::new(cfg.quantizer.bins, cfg.seanet.dimension, values).unwrap()
            })
            .collect::<Vec<_>>();
        let codec = MimiCodecGguf {
            attrs: MimiRvqAttrs {
                n_codebooks: cfg.quantizer.n_q,
                codebook_size: cfg.quantizer.bins,
                d_model: cfg.seanet.dimension,
            },
            tables,
        };
        let neural = MimiNeuralDecoder::synthesized(&cfg, seed, false).unwrap();
        (
            MimiStreamingCodec::new(codec.clone(), neural, cfg.sample_rate).unwrap(),
            codec,
        )
    }

    #[test]
    fn mimi_successive_pushes_are_bit_identical_to_whole_decode() {
        let seed = 0x48;
        let (engine, codec) = synthesized_streaming_codec(seed);
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let codes = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
        let frames = codes.len() / codec.attrs.n_codebooks;
        let features = mimi_rvq_decode(&codes, frames, &codec.tables, &codec.attrs).unwrap();
        let reference = MimiNeuralDecoder::synthesized(&cfg, seed, false)
            .unwrap()
            .decode_all(&features)
            .unwrap();

        let mut stream = engine.open_decoder().unwrap();
        let mut got = Vec::with_capacity(reference.len());
        let mut frame = vec![0.0; stream.frame_hop()];
        for code_frame in codes.chunks_exact(stream.n_codebooks()) {
            assert_eq!(stream.push_codes(code_frame).unwrap(), 1);
            let written = stream.pull_pcm(&mut frame).unwrap();
            assert_eq!(written, stream.frame_hop());
            got.extend_from_slice(&frame[..written]);
        }
        assert_eq!(got, reference);

        stream.reset().unwrap();
        let n_codebooks = stream.n_codebooks();
        assert_eq!(stream.push_codes(&codes[..n_codebooks]).unwrap(), 1);
        stream.pull_pcm(&mut frame).unwrap();
        let frame_hop = stream.frame_hop();
        assert_eq!(&frame[..], &reference[..frame_hop]);
    }

    #[test]
    fn mimi_streaming_shape_and_backpressure_fail_loudly() {
        let (engine, _) = synthesized_streaming_codec(9);
        let mut stream = engine.open_decoder().unwrap();
        assert!(matches!(
            stream.push_codes(&[0, 1]),
            Err(VokraError::InvalidArgument(_))
        ));
        assert_eq!(stream.push_codes(&[0, 1, 2]).unwrap(), 1);
        assert!(matches!(
            stream.push_codes(&[0, 1, 2]),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut short = vec![0.0; stream.frame_hop() - 1];
        assert!(matches!(
            stream.pull_pcm(&mut short),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut exact = vec![0.0; stream.frame_hop()];
        assert_eq!(stream.pull_pcm(&mut exact).unwrap(), exact.len());
        assert_eq!(stream.pull_pcm(&mut exact).unwrap(), 0);
    }

    #[test]
    fn dac_binder_rejects_missing_quantizer_tensor() {
        let mut b = GgufBuilder::new();
        b.add_u32("vokra.dac.n_codebooks", 1);
        b.add_u32("vokra.dac.codebook_size", 2);
        b.add_u32("vokra.dac.codebook_dim", 2);
        b.add_u32("vokra.dac.d_model", 3);
        b.add_u32("vokra.dac.sample_rate", 24000);
        b.add_u32("vokra.dac.hop_length", 320);
        // No quantizer tensors at all.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = DacCodecGguf::from_gguf(&file).expect_err("must fail");
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }
}
