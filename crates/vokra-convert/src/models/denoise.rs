//! DeepFilterNet `denoise` → `vokra.denoise.*` GGUF conversion (M4-20 T12).
//!
//! The GGUF schema (config keys + flat F32 tensors) and the write core
//! ([`DenoiseModel::to_gguf_bytes`]) live in `vokra-ops`; this module is the
//! `vokra-convert` offline entry.
//!
//! # Honest scope (ADR M4-20 §D-5, ticket T12/T17)
//!
//! Parsing a **real** DeepFilterNet `.ckpt` into a [`DenoiseModel`] is the
//! **owner** side-car (T17): the real DeepFilterNet encoder is a conv+GRU stack,
//! whereas the runtime [`DenoiseWeights`] here is a shape-faithful scaffold, so
//! the tensor mapping needs the real checkpoint (which the owner holds). This
//! module therefore exposes the GGUF *writer* — [`convert_denoise_synthetic`]
//! for the offline round-trip / demo, and [`convert_denoise_from_model`] for an
//! already-built [`DenoiseModel`] — and leaves the checkpoint parser to the
//! owner. Nothing here fabricates a "converted real model".

use vokra_ops::denoise::{DeepFilterNetConfig, DenoiseModel, DenoiseWeights};

/// Writes the `vokra.denoise.*` GGUF for an already-built [`DenoiseModel`].
///
/// # Errors
///
/// Propagates [`DenoiseModel::to_gguf_bytes`].
pub fn convert_denoise_from_model(model: &DenoiseModel) -> vokra_core::Result<Vec<u8>> {
    model.to_gguf_bytes()
}

/// Builds a **synthetic** DeepFilterNet-shaped model for the given config and
/// writes its `vokra.denoise.*` GGUF (offline round-trip / demo path — NOT a
/// trained network; real weights are the owner checkpoint, T17).
///
/// # Errors
///
/// Propagates model construction / GGUF write errors.
pub fn convert_denoise_synthetic(
    cfg: DeepFilterNetConfig,
    seed: u64,
) -> vokra_core::Result<Vec<u8>> {
    let weights = DenoiseWeights::synthesized(&cfg, seed);
    let model = DenoiseModel::new(cfg, weights)?;
    convert_denoise_from_model(&model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_conversion_round_trips_through_gguf() {
        // Offline path: config → synthetic model → GGUF bytes → parse →
        // bind → forward runs. Proves the vokra-convert denoise offline path
        // (the real-checkpoint parse is owner, T17).
        let cfg = DeepFilterNetConfig {
            n_fft: 64,
            hop: 16,
            sample_rate: 16000,
            n_erb: 8,
            hidden: 16,
            df_bins: 6,
            df_order: 3,
        };
        let bytes = convert_denoise_synthetic(cfg, 11).unwrap();
        let gguf = vokra_core::gguf::GgufFile::parse(bytes).unwrap();
        let model = DenoiseModel::from_gguf(&gguf).unwrap();
        assert_eq!(model.config(), &cfg);
        let noisy: Vec<f32> = (0..1024).map(|i| 0.1 * (i as f32 * 0.07).sin()).collect();
        let enhanced = model.forward(&noisy).unwrap();
        assert_eq!(enhanced.len(), noisy.len());
    }
}
