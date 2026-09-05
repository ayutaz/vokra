//! CosyVoice2 → HiFTNet vocoder chain (SoTA plan §1(a) 訂正, Phase 1-3).
//!
//! # Correct upstream pipeline (arXiv:2412.10117 + cosyvoice/hifigan/generator.py)
//!
//! ```text
//! FSQ tokens → Qwen2.5-0.5B AR decoder → chunk-aware CFM → mel → HiFTNet → PCM
//! ```
//!
//! `HiFTGenerator` (upstream `cosyvoice/hifigan/generator.py:378`,
//! `class HiFTGenerator` — docstring: `"HiFTNet Generator: Neural Source
//! Filter + ISTFTNet"`) is the terminal vocoder that consumes the mel
//! spectrogram produced by the chunk-aware CFM and emits 24 kHz PCM. Vokra's
//! port lives in [`vokra_ops::hiftnet`] (Waves 3c-2/3c-3 + Wave 4 harness).
//!
//! # Why this seam (and not the `super::mimi_bridge` module)
//!
//! The 2026-07-22 SoTA-plan §1(a) 訂正 identified that the previous
//! CosyVoice2 → Mimi wiring was built on a wrong premise: **CosyVoice2 does
//! not use the Mimi codec** — that is exclusive to Moshi and CSM. The
//! upstream `cosyvoice/hifigan/generator.py` file makes no reference to
//! Mimi at any point; instead it composes `SourceModuleHnNSF` (NSF) with a
//! `torch.istft` post-conv (ISTFTNet) and a `Snake` activation stack (see
//! `:320 SourceModuleHnNSF`, `:503 torch.istft`, `:102 Snake`). The
//! `super::mimi_bridge` module is therefore `#[deprecated]`
//! and kept only to avoid breaking existing test imports and the
//! `super::chunk_pipeline` scaffold; new callers use [`HiFTChain`].
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! [`HiFTChain`] holds an owned [`vokra_ops::hiftnet::HiFTGenerator`] — the
//! op crate is a first-party `vokra-*` workspace member, so the root
//! `Cargo.lock` stays `vokra-*` only. No external crate is added by this
//! module.
//!
//! # Fail-loud contract (FR-EX-08)
//!
//! Every shape mismatch is caught inside [`vokra_ops::hiftnet::HiFTGenerator::new`]
//! (config-side: `upsample_kernel_sizes.len() != upsample_rates.len()`,
//! `resblock_dilation_sizes.len() != resblock_kernel_sizes.len()`, F0
//! predictor / ResBlock branch counts, conv layout mismatches). This module
//! propagates those errors verbatim rather than swallowing or re-wrapping
//! them, so a mis-supplied weight bundle surfaces the exact upstream failure.

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use vokra_ops::hiftnet::HiFTResidentOps;
use vokra_ops::hiftnet::{
    F0PredictorWeights, HiFTGenerator, HiFTGeneratorConfig, HiFTGeneratorWeights, ResBlockWeights,
};

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use super::hift_chain_metal::MetalHiFTResidentOps;

/// Configuration for the HiFTNet vocoder chain.
///
/// A newtype re-alias of [`vokra_ops::hiftnet::HiFTGeneratorConfig`] so the
/// public surface stays namespaced under the CosyVoice2 module (matching the
/// `text_encoder::CosyVoice2Tokenizer` / `llm::LlmBackboneConfig` pattern in
/// this crate) without introducing a shape-drift wrapper that would have to
/// be kept in sync with the op-crate config.
pub type HiFTChainConfig = HiFTGeneratorConfig;

/// Learned parameters for the HiFTNet vocoder chain.
///
/// A newtype re-alias of [`vokra_ops::hiftnet::HiFTGeneratorWeights`] for the
/// same reason as [`HiFTChainConfig`]. Every tensor layout (conv_pre,
/// per-stage ConvTranspose1d ups, per-stage source_downs, per-stage
/// source_resblocks, row-major `[num_upsamples * num_kernels]` resblocks,
/// conv_post, m_source_linear, F0 predictor) is documented on the op-crate
/// type.
pub type HiFTChainWeights = HiFTGeneratorWeights;

/// Immutable identity of the standalone CosyVoice2 HiFT companion artifact.
///
/// These values deliberately mirror the offline converter instead of making
/// the runtime depend on `vokra-convert` (the dependency direction is
/// converter → runtime, never the reverse).  The companion is a separate
/// vocoder artifact; the composite `cosyvoice2` loader remains
/// inspection-only.
const HIFT_ARCH: &str = "cosyvoice2_hift";
const HIFT_MODEL_NAME: &str = "cosyvoice2-0.5b-hift";
const HIFT_CATEGORY: &str = "vocoder";
const HIFT_UPSTREAM_HF: &str = "FunAudioLLM/CosyVoice2-0.5B";
const HIFT_UPSTREAM_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const HIFT_CHECKPOINT_FILE: &str = "hift.pt";
const HIFT_CHECKPOINT_BYTES: u32 = 83_390_254;
const HIFT_CHECKPOINT_SHA256: &str =
    "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879";
const HIFT_CONFIG_BYTES: u32 = 7_330;
const HIFT_CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
const HIFT_CONFIG_GIT_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
const HIFT_SOURCE_REPO: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
const HIFT_SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
const HIFT_SOURCE_GENERATOR_SHA256: &str =
    "f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626";
const HIFT_SOURCE_GENERATOR_GIT_BLOB_SHA1: &str = "326a1a70ae7707662939c20493b3a8e4b0906216";
const HIFT_TENSOR_COUNT: usize = 328;
const HIFT_WEIGHT_NORM_PAIRS: u32 = 82;
const HIFT_MANIFEST_SHA256: &str =
    "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c";

const HIFT_CONFIG_PREFIX: &str = "vokra.cosyvoice2_hift.config.";
const HIFT_KEY_CATEGORY: &str = "vokra.model.category";
const HIFT_KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const HIFT_KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const HIFT_KEY_PROVENANCE_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const HIFT_KEY_UPSTREAM_REVISION: &str = "vokra.cosyvoice2_hift.upstream_revision";
const HIFT_KEY_CHECKPOINT_FILE: &str = "vokra.cosyvoice2_hift.checkpoint_file";
const HIFT_KEY_CHECKPOINT_BYTES: &str = "vokra.cosyvoice2_hift.checkpoint_bytes";
const HIFT_KEY_CHECKPOINT_SHA256: &str = "vokra.cosyvoice2_hift.checkpoint_sha256";
const HIFT_KEY_CONFIG_BYTES: &str = "vokra.cosyvoice2_hift.config_bytes";
const HIFT_KEY_CONFIG_SHA256: &str = "vokra.cosyvoice2_hift.config_sha256";
const HIFT_KEY_CONFIG_GIT_BLOB_SHA1: &str = "vokra.cosyvoice2_hift.config_git_blob_sha1";
const HIFT_KEY_SOURCE_REPO: &str = "vokra.cosyvoice2_hift.source_repo";
const HIFT_KEY_SOURCE_REVISION: &str = "vokra.cosyvoice2_hift.source_revision";
const HIFT_KEY_SOURCE_GENERATOR_SHA256: &str = "vokra.cosyvoice2_hift.source_generator_sha256";
const HIFT_KEY_SOURCE_GENERATOR_GIT_BLOB_SHA1: &str =
    "vokra.cosyvoice2_hift.source_generator_git_blob_sha1";
const HIFT_KEY_MANIFEST_SHA256: &str = "vokra.cosyvoice2_hift.tensor_manifest_sha256";
const HIFT_KEY_WEIGHT_NORM_PAIRS: &str = "vokra.cosyvoice2_hift.weight_norm_pairs";

const HIFT_MANIFEST_DIGEST: [u8; 32] = [
    0xce, 0xcb, 0xb2, 0xd6, 0x8f, 0x91, 0x33, 0x7f, 0x26, 0x3d, 0xb0, 0xf0, 0x33, 0x3c, 0x75, 0x57,
    0x35, 0x16, 0xe7, 0x08, 0x7b, 0x6e, 0x75, 0xd6, 0xea, 0x64, 0x7b, 0x3f, 0x86, 0xaf, 0xec, 0x7c,
];

/// CosyVoice2 → HiFTNet vocoder chain — the mel → PCM seam.
///
/// Owns a single [`HiFTGenerator`] produced from a caller-supplied
/// [`HiFTChainConfig`] + [`HiFTChainWeights`] bundle. The forward path
/// delegates to [`HiFTGenerator::forward`] verbatim; this wrapper exists so
/// [`super::CosyVoice2Tts`] can carry an `Option<HiFTChain>` field with a
/// stable name across the T24 codec-migration work — the top-level engine
/// does not have to know which vocoder is bound, and a future caller who
/// wires the CFM head to this chain gets a single `hift_chain` seam.
///
/// # Construction
///
/// [`HiFTChain::new`] validates the config/weights bundle and builds an
/// internal [`HiFTGenerator`]. Every shape check surfaces the op-crate's
/// error verbatim (see the module docstring's fail-loud contract note).
///
/// # Forward
///
/// [`HiFTChain::forward`] takes a mel spectrogram `[in_channels, t_mel]`
/// row-major and returns a `Vec<f32>` PCM waveform. The sample rate of that
/// waveform is [`HiFTChainConfig::sampling_rate`] and the length is exactly
/// `t_mel * total_upsample_factor()` (upstream contract, spelled out in the
/// op-crate rustdoc for [`HiFTGenerator::forward`]).
#[derive(Debug, Clone)]
pub struct HiFTChain {
    generator: HiFTGenerator,
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key).and_then(GgufMetadataValue::as_str) {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: metadata `{key}`={value:?}, expected {expected:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: missing/non-string metadata `{key}`"
        ))),
    }
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: metadata `{key}`={value:?}, expected U32({expected})"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: missing/non-u32 metadata `{key}`"
        ))),
    }
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) if value.to_bits() == expected.to_bits() => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: metadata `{key}`={value:?}, expected F32({expected})"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: missing/non-f32 metadata `{key}`"
        ))),
    }
}

fn validate_config_metadata(file: &GgufFile) -> Result<()> {
    for (key, value) in [
        ("in_channels", 80),
        ("base_channels", 512),
        ("nb_harmonics", 8),
        ("sampling_rate", 24_000),
        ("istft_n_fft", 16),
        ("istft_hop_len", 4),
    ] {
        require_u32(file, &format!("{HIFT_CONFIG_PREFIX}{key}"), value)?;
    }
    for (key, value) in [
        ("nsf_alpha", 0.1),
        ("nsf_sigma", 0.003),
        ("nsf_voiced_threshold", 10.0),
        ("lrelu_slope", 0.1),
        ("audio_limit", 0.99),
    ] {
        require_f32(file, &format!("{HIFT_CONFIG_PREFIX}{key}"), value)?;
    }
    for (key, values) in [
        ("upsample_rates", [8, 5, 3]),
        ("upsample_kernel_sizes", [16, 11, 7]),
        ("resblock_kernel_sizes", [3, 7, 11]),
        ("source_resblock_kernel_sizes", [7, 7, 11]),
    ] {
        require_u32(file, &format!("{HIFT_CONFIG_PREFIX}{key}.len"), 3)?;
        for (index, value) in values.into_iter().enumerate() {
            require_u32(file, &format!("{HIFT_CONFIG_PREFIX}{key}.{index}"), value)?;
        }
    }
    for key in ["resblock_dilation_sizes", "source_resblock_dilation_sizes"] {
        for i in 0..3 {
            for (j, value) in [1, 3, 5].into_iter().enumerate() {
                require_u32(file, &format!("{HIFT_CONFIG_PREFIX}{key}.{i}.{j}"), value)?;
            }
        }
    }
    for (key, value) in [
        ("f0.class_channels", 1),
        ("f0.input_channels", 80),
        ("f0.cond_channels", 512),
        ("f0.num_layers", 5),
        ("f0.kernel_size", 3),
    ] {
        require_u32(file, &format!("{HIFT_CONFIG_PREFIX}{key}"), value)?;
    }
    Ok(())
}

fn load_f32(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "cosyvoice2_hift: required tensor `{name}` is missing"
        ))
    })?;
    if info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: tensor `{name}` is {:?}, expected F32",
            info.dtype
        )));
    }
    let actual: Vec<usize> = info.dimensions.iter().map(|&v| v as usize).collect();
    if actual != shape {
        return Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: tensor `{name}` shape {actual:?}, expected {shape:?}"
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "cosyvoice2_hift: tensor `{name}` decode failed: {error}"
        ))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "cosyvoice2_hift: tensor `{name}` contains non-finite values"
        )));
    }
    Ok(values)
}

fn load_weight_norm(
    file: &GgufFile,
    prefix: &str,
    rows: usize,
    cols: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let g = load_f32(
        file,
        &format!("{prefix}.parametrizations.weight.original0"),
        &[rows, 1, 1],
    )?;
    let v = load_f32(
        file,
        &format!("{prefix}.parametrizations.weight.original1"),
        &[rows, cols, kernel],
    )?;
    fold_weight_norm(&g, &v, rows, cols, kernel, prefix)
}

fn fold_weight_norm(
    g: &[f32],
    v: &[f32],
    rows: usize,
    cols: usize,
    kernel: usize,
    label: &str,
) -> Result<Vec<f32>> {
    let row_size = cols.checked_mul(kernel).ok_or_else(|| {
        VokraError::InvalidArgument(format!("cosyvoice2_hift: `{label}` shape overflow"))
    })?;
    let expected_v_len = rows.checked_mul(row_size).ok_or_else(|| {
        VokraError::InvalidArgument(format!("cosyvoice2_hift: `{label}` shape overflow"))
    })?;
    if g.len() != rows || v.len() != expected_v_len {
        return Err(VokraError::InvalidArgument(format!(
            "cosyvoice2_hift: `{label}` weight_norm shape mismatch: g={}, v={}, expected g={}, v={}",
            g.len(),
            v.len(),
            rows,
            expected_v_len
        )));
    }
    let mut output = vec![0.0; v.len()];
    for row in 0..rows {
        let g_value = g[row];
        if !g_value.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_hift: `{label}` original0 contains a non-finite value at row {row}"
            )));
        }
        let source = &v[row * row_size..(row + 1) * row_size];
        let mut norm_sq = 0.0f32;
        for &value in source {
            if !value.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "cosyvoice2_hift: `{label}` original1 contains a non-finite value at row {row}"
                )));
            }
            norm_sq += value * value;
        }
        let norm = norm_sq.sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_hift: `{label}` original1 row {row} has zero/non-finite norm"
            )));
        }
        for (dst, &value) in output[row * row_size..(row + 1) * row_size]
            .iter_mut()
            .zip(source)
        {
            *dst = g_value * value / norm;
            if !dst.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "cosyvoice2_hift: `{label}` folded row {row} is non-finite"
                )));
            }
        }
    }
    Ok(output)
}

fn load_resblock(
    file: &GgufFile,
    prefix: &str,
    channels: usize,
    kernel: usize,
) -> Result<ResBlockWeights> {
    let mut weights = ResBlockWeights {
        convs1_w: Vec::with_capacity(3),
        convs1_b: Vec::with_capacity(3),
        convs2_w: Vec::with_capacity(3),
        convs2_b: Vec::with_capacity(3),
        activations1_alpha: Vec::with_capacity(3),
        activations2_alpha: Vec::with_capacity(3),
    };
    for branch in 0..3 {
        let p = format!("{prefix}.convs1.{branch}");
        weights.activations1_alpha.push(load_f32(
            file,
            &format!("{prefix}.activations1.{branch}.alpha"),
            &[channels],
        )?);
        weights
            .convs1_w
            .push(load_weight_norm(file, &p, channels, channels, kernel)?);
        weights
            .convs1_b
            .push(load_f32(file, &format!("{p}.bias"), &[channels])?);

        let p = format!("{prefix}.convs2.{branch}");
        weights.activations2_alpha.push(load_f32(
            file,
            &format!("{prefix}.activations2.{branch}.alpha"),
            &[channels],
        )?);
        weights
            .convs2_w
            .push(load_weight_norm(file, &p, channels, channels, kernel)?);
        weights
            .convs2_b
            .push(load_f32(file, &format!("{p}.bias"), &[channels])?);
    }
    Ok(weights)
}

fn expected_hift_manifest() -> Vec<(String, Vec<usize>)> {
    let mut manifest = Vec::with_capacity(HIFT_TENSOR_COUNT);
    fn wn(
        manifest: &mut Vec<(String, Vec<usize>)>,
        prefix: &str,
        rows: usize,
        cols: usize,
        k: usize,
    ) {
        manifest.push((
            format!("{prefix}.parametrizations.weight.original0"),
            vec![rows, 1, 1],
        ));
        manifest.push((
            format!("{prefix}.parametrizations.weight.original1"),
            vec![rows, cols, k],
        ));
    }
    fn conv(
        manifest: &mut Vec<(String, Vec<usize>)>,
        prefix: &str,
        rows: usize,
        cols: usize,
        k: usize,
    ) {
        manifest.push((format!("{prefix}.bias"), vec![rows]));
        wn(manifest, prefix, rows, cols, k);
    }
    conv(&mut manifest, "conv_pre", 512, 80, 7);
    conv(&mut manifest, "conv_post", 18, 64, 7);
    for i in 0..5 {
        conv(
            &mut manifest,
            &format!("f0_predictor.condnet.{}", i * 2),
            512,
            if i == 0 { 80 } else { 512 },
            3,
        );
    }
    manifest.push(("f0_predictor.classifier.bias".to_owned(), vec![1]));
    manifest.push(("f0_predictor.classifier.weight".to_owned(), vec![1, 512]));
    manifest.push(("m_source.l_linear.bias".to_owned(), vec![1]));
    manifest.push(("m_source.l_linear.weight".to_owned(), vec![1, 9]));
    fn block(manifest: &mut Vec<(String, Vec<usize>)>, prefix: &str, channels: usize, k: usize) {
        for branch in 0..3 {
            manifest.push((
                format!("{prefix}.activations1.{branch}.alpha"),
                vec![channels],
            ));
            manifest.push((
                format!("{prefix}.activations2.{branch}.alpha"),
                vec![channels],
            ));
            conv(
                manifest,
                &format!("{prefix}.convs1.{branch}"),
                channels,
                channels,
                k,
            );
            conv(
                manifest,
                &format!("{prefix}.convs2.{branch}"),
                channels,
                channels,
                k,
            );
        }
    }
    for (i, (channels, k)) in [(256, 7), (128, 7), (64, 11)].into_iter().enumerate() {
        manifest.push((format!("source_downs.{i}.bias"), vec![channels]));
        manifest.push((
            format!("source_downs.{i}.weight"),
            vec![channels, 18, [30, 6, 1][i]],
        ));
        block(&mut manifest, &format!("source_resblocks.{i}"), channels, k);
    }
    for (i, (input, output, k)) in [(512, 256, 16), (256, 128, 11), (128, 64, 7)]
        .into_iter()
        .enumerate()
    {
        manifest.push((format!("ups.{i}.bias"), vec![output]));
        wn(&mut manifest, &format!("ups.{i}"), input, output, k);
    }
    for (i, channels) in [256, 128, 64].into_iter().enumerate() {
        for branch in 0..3 {
            block(
                &mut manifest,
                &format!("resblocks.{}", i * 3 + branch),
                channels,
                [3, 7, 11][branch],
            );
        }
    }
    manifest
}

impl HiFTChain {
    /// Builds a [`HiFTChain`] from its config + weights bundle.
    ///
    /// # Errors
    ///
    /// Propagates every [`HiFTGenerator::new`] validation error verbatim.
    /// The op-crate rustdoc enumerates them; the common ones are:
    ///
    /// - [`vokra_core::VokraError::InvalidArgument`] on empty `upsample_rates`, mismatched
    ///   `upsample_kernel_sizes` / `resblock_dilation_sizes` lengths, or a
    ///   conv weight vector whose length does not match the expected
    ///   `[out_ch, in_ch, kernel]` layout.
    ///
    /// # Zero-argument sanity check
    ///
    /// A caller who accidentally builds a [`HiFTChainWeights`] whose
    /// `ups_w` length disagrees with `upsample_rates` gets a loud error at
    /// construction time — never a mid-forward panic (FR-EX-08).
    pub fn new(cfg: HiFTChainConfig, weights: HiFTChainWeights) -> Result<Self> {
        let generator = HiFTGenerator::new(cfg, weights)?;
        Ok(Self { generator })
    }

    /// Binds the authenticated standalone CosyVoice2 HiFT companion GGUF.
    ///
    /// The loader intentionally performs all cheap identity and topology
    /// checks before decoding any tensor payload.  The GGUF carries the
    /// un-folded PyTorch `weight_norm` pairs; folding is performed here at
    /// bind time using `dim=0` semantics, which is also the correct first
    /// axis for both regular Conv1d and ConvTranspose1d weights.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, HIFT_ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, HIFT_MODEL_NAME)?;
        require_string(file, HIFT_KEY_CATEGORY, HIFT_CATEGORY)?;

        // Require the full standard provenance tuple, not merely the
        // resolver's best-effort fallback.  This keeps a hand-assembled file
        // from becoming indistinguishable from the audited artifact.
        require_string(
            file,
            vokra_core::gguf::chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            "permissive",
        )?;
        require_string(
            file,
            vokra_core::gguf::chunks::KEY_PROVENANCE_LICENSE,
            "apache-2.0",
        )?;
        require_string(
            file,
            vokra_core::gguf::chunks::KEY_PROVENANCE_MODEL_ID,
            HIFT_MODEL_NAME,
        )?;
        require_string(
            file,
            vokra_core::gguf::chunks::KEY_PROVENANCE_SOURCE,
            HIFT_SOURCE_REPO,
        )?;
        let license = vokra_core::resolve_license_class(file);
        if license.class != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_hift: provenance resolves to {:?}, expected permissive",
                license.class
            )));
        }

        for (key, expected) in [
            (HIFT_KEY_UPSTREAM_HF, HIFT_UPSTREAM_HF),
            (
                HIFT_KEY_PROVENANCE_UPSTREAM_REVISION,
                HIFT_UPSTREAM_REVISION,
            ),
            (
                HIFT_KEY_PROVENANCE_CHECKPOINT_SHA256,
                HIFT_CHECKPOINT_SHA256,
            ),
            (HIFT_KEY_UPSTREAM_REVISION, HIFT_UPSTREAM_REVISION),
            (HIFT_KEY_CHECKPOINT_FILE, HIFT_CHECKPOINT_FILE),
            (HIFT_KEY_CHECKPOINT_SHA256, HIFT_CHECKPOINT_SHA256),
            (HIFT_KEY_CONFIG_SHA256, HIFT_CONFIG_SHA256),
            (HIFT_KEY_CONFIG_GIT_BLOB_SHA1, HIFT_CONFIG_GIT_BLOB_SHA1),
            (HIFT_KEY_SOURCE_REPO, HIFT_SOURCE_REPO),
            (HIFT_KEY_SOURCE_REVISION, HIFT_SOURCE_REVISION),
            (
                HIFT_KEY_SOURCE_GENERATOR_SHA256,
                HIFT_SOURCE_GENERATOR_SHA256,
            ),
            (
                HIFT_KEY_SOURCE_GENERATOR_GIT_BLOB_SHA1,
                HIFT_SOURCE_GENERATOR_GIT_BLOB_SHA1,
            ),
            (HIFT_KEY_MANIFEST_SHA256, HIFT_MANIFEST_SHA256),
        ] {
            require_string(file, key, expected)?;
        }
        require_u32(file, HIFT_KEY_CHECKPOINT_BYTES, HIFT_CHECKPOINT_BYTES)?;
        require_u32(file, HIFT_KEY_CONFIG_BYTES, HIFT_CONFIG_BYTES)?;
        require_u32(file, HIFT_KEY_WEIGHT_NORM_PAIRS, HIFT_WEIGHT_NORM_PAIRS)?;
        validate_config_metadata(file)?;

        // The manifest hash authenticates names and dimensions.  Dtype is
        // checked separately because it is not part of the canonical digest.
        crate::strict_checkpoint::verify_tensor_manifest(
            file,
            "cosyvoice2_hift",
            HIFT_TENSOR_COUNT,
            HIFT_MANIFEST_DIGEST,
            HIFT_MODEL_NAME,
        )?;
        let expected = expected_hift_manifest();
        for (name, shape) in &expected {
            let info = file.tensor_info(name).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "cosyvoice2_hift: required tensor `{name}` is missing"
                ))
            })?;
            if info.dtype != GgmlType::F32 {
                return Err(VokraError::ModelLoad(format!(
                    "cosyvoice2_hift: tensor `{name}` is {:?}, expected F32",
                    info.dtype
                )));
            }
            let actual: Vec<usize> = info.dimensions.iter().map(|&v| v as usize).collect();
            if actual.as_slice() != shape.as_slice() {
                return Err(VokraError::ModelLoad(format!(
                    "cosyvoice2_hift: tensor `{name}` shape {actual:?}, expected {shape:?}"
                )));
            }
        }

        let conv_pre_w = load_weight_norm(file, "conv_pre", 512, 80, 7)?;
        let conv_pre_b = load_f32(file, "conv_pre.bias", &[512])?;
        let conv_post_w = load_weight_norm(file, "conv_post", 18, 64, 7)?;
        let conv_post_b = load_f32(file, "conv_post.bias", &[18])?;
        let mut ups_w = Vec::with_capacity(3);
        let mut ups_b = Vec::with_capacity(3);
        for (i, (input, output, kernel)) in [(512, 256, 16), (256, 128, 11), (128, 64, 7)]
            .into_iter()
            .enumerate()
        {
            ups_w.push(load_weight_norm(
                file,
                &format!("ups.{i}"),
                input,
                output,
                kernel,
            )?);
            ups_b.push(load_f32(file, &format!("ups.{i}.bias"), &[output])?);
        }

        let mut source_downs_w = Vec::with_capacity(3);
        let mut source_downs_b = Vec::with_capacity(3);
        for (i, (output, kernel)) in [(256, 30), (128, 6), (64, 1)].into_iter().enumerate() {
            source_downs_w.push(load_f32(
                file,
                &format!("source_downs.{i}.weight"),
                &[output, 18, kernel],
            )?);
            source_downs_b.push(load_f32(
                file,
                &format!("source_downs.{i}.bias"),
                &[output],
            )?);
        }

        let mut source_resblock_weights = Vec::with_capacity(3);
        for (i, (channels, kernel)) in [(256, 7), (128, 7), (64, 11)].into_iter().enumerate() {
            source_resblock_weights.push(load_resblock(
                file,
                &format!("source_resblocks.{i}"),
                channels,
                kernel,
            )?);
        }
        let mut resblock_weights = Vec::with_capacity(9);
        for (i, channels) in [256, 128, 64].into_iter().enumerate() {
            for j in 0..3 {
                resblock_weights.push(load_resblock(
                    file,
                    &format!("resblocks.{}", i * 3 + j),
                    channels,
                    [3, 7, 11][j],
                )?);
            }
        }

        let mut f0_conv_weights = Vec::with_capacity(5);
        let mut f0_conv_biases = Vec::with_capacity(5);
        for i in 0..5 {
            let input = if i == 0 { 80 } else { 512 };
            let prefix = format!("f0_predictor.condnet.{}", i * 2);
            f0_conv_weights.push(load_weight_norm(file, &prefix, 512, input, 3)?);
            f0_conv_biases.push(load_f32(file, &format!("{prefix}.bias"), &[512])?);
        }
        let f0_predictor_weights = F0PredictorWeights {
            conv_weights: f0_conv_weights,
            conv_biases: f0_conv_biases,
            linear_w: load_f32(file, "f0_predictor.classifier.weight", &[1, 512])?,
            linear_b: load_f32(file, "f0_predictor.classifier.bias", &[1])?,
        };

        let weights = HiFTChainWeights {
            conv_pre_w,
            conv_pre_b,
            ups_w,
            ups_b,
            source_downs_w,
            source_downs_b,
            source_resblock_weights,
            resblock_weights,
            conv_post_w,
            conv_post_b,
            m_source_linear_w: load_f32(file, "m_source.l_linear.weight", &[1, 9])?,
            m_source_linear_b: load_f32(file, "m_source.l_linear.bias", &[1])?[0],
            f0_predictor_weights,
        };
        let config = HiFTChainConfig {
            in_channels: 80,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: 24_000,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: vec![8, 5, 3],
            upsample_kernel_sizes: vec![16, 11, 7],
            istft_n_fft: 16,
            istft_hop_len: 4,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            source_resblock_kernel_sizes: vec![7, 7, 11],
            source_resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        };
        Self::new(config, weights)
    }

    /// Immutable access to the generator config the chain was built with —
    /// convenient for the caller to read the sample rate off the same
    /// source the vocoder used, without holding a duplicate copy.
    #[must_use]
    pub fn config(&self) -> &HiFTChainConfig {
        self.generator.config()
    }

    /// Output PCM sample rate in Hz — mirror of `config().sampling_rate`,
    /// kept as a first-class accessor so a caller wiring the result into a
    /// [`vokra_core::SynthesizedAudio`] does not have to walk into the
    /// config.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.generator.config().sampling_rate
    }

    /// Runs the standalone F0 predictor on a mel spectrogram without
    /// executing the vocoder decoder.  This is an independently comparable
    /// F0/reference seam: callers can compare the checkpoint's F0 sequence
    /// independently from source synthesis and ISTFT output.
    pub fn f0_predictor_forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        self.generator.f0_predictor_forward(mel, t_mel)
    }

    /// Runs the HiFTNet vocoder forward on a mel spectrogram.
    ///
    /// `mel` is row-major `[in_channels, t_mel]`. Returns the reconstructed
    /// PCM as a `Vec<f32>` of length `t_mel * total_upsample_factor()`.
    ///
    /// # Errors
    ///
    /// Propagates [`HiFTGenerator::forward`] errors verbatim:
    ///
    /// - [`vokra_core::VokraError::InvalidArgument`] on `t_mel == 0` or
    ///   `mel.len() != in_channels * t_mel`.
    /// - Any downstream op error surfaces with the op-crate's original
    ///   message.
    pub fn forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        self.generator.forward(mel, t_mel)
    }

    /// Runs HiFTNet using the selected backend.
    ///
    /// CPU retains the established scalar implementation.  Metal uses the
    /// complete context-owned resident graph on Apple targets when the
    /// optional `metal` feature is enabled, and performs exactly one final
    /// device-to-host download.  Other backends are rejected explicitly; no
    /// backend silently falls back to CPU.
    pub fn forward_with_backend(
        &self,
        mel: &[f32],
        t_mel: usize,
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        match backend {
            BackendKind::Cpu => self.forward(mel, t_mel),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            BackendKind::Metal => {
                let context = vokra_backend_metal::MetalContext::new()?;
                let mut ops = MetalHiFTResidentOps::new(&context);
                let before = context.readback_count();
                let output_len = t_mel
                    .checked_mul(self.config().total_upsample_factor() as usize)
                    .ok_or_else(|| {
                        VokraError::InvalidArgument(
                            "CosyVoice2 HiFT output length overflow".to_owned(),
                        )
                    })?;
                let result = self
                    .generator
                    .forward_with_resident_ops(&mut ops, mel, t_mel);
                let tensor = match result {
                    Ok(value) => value,
                    Err(error) => {
                        let delta = context.readback_count().saturating_sub(before);
                        if delta != 0 {
                            return Err(VokraError::BackendUnavailable(format!(
                                "CosyVoice2 HiFT Metal resident graph failed after {delta} readbacks; expected zero on failure: {error}"
                            )));
                        }
                        return Err(error);
                    }
                };
                let audio = ops.download(&tensor, 1, output_len)?;
                let delta = context.readback_count().saturating_sub(before);
                if delta != 1 {
                    return Err(VokraError::BackendUnavailable(format!(
                        "CosyVoice2 HiFT Metal resident forward performed {delta} readbacks; expected exactly one final readback"
                    )));
                }
                Ok(audio)
            }
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            BackendKind::Metal => Err(VokraError::BackendUnavailable(
                "CosyVoice2 HiFT Metal execution requires the `metal` feature on Apple targets"
                    .to_owned(),
            )),
            _ => Err(VokraError::UnsupportedOp(
                "CosyVoice2 HiFT supports only the CPU path and the Apple Metal resident path"
                    .to_owned(),
            )),
        }
    }
}

// -----------------------------------------------------------------------------
// SAFETY / posture notes for consumers
// -----------------------------------------------------------------------------
// This module contains ZERO `unsafe`. The generator forward is pure-safe Rust
// living in `vokra_ops::hiftnet`. `HiFTChain` is `Debug + Clone` because
// `HiFTGenerator` is (all owned f32 weight vectors, no interior mutability).

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::VokraError;
    use vokra_ops::hiftnet::{F0PredictorWeights, ResBlockWeights};

    /// Wave-4 harness pattern: a small-shape config so the synthesized-weight
    /// build path is exercised without a real HiFTNet checkpoint. The
    /// numbers here mirror `small_hift_config()` in the op-crate parity
    /// harness; keeping them in sync with a Wave 3c-3 helper is left to the
    /// respective owner (Wave 4 tests already pin the shapes, and any change
    /// there would surface as a build error here).
    fn small_hift_chain_bundle() -> (HiFTChainConfig, HiFTChainWeights) {
        let cfg = HiFTChainConfig {
            in_channels: 4,
            base_channels: 8,
            nb_harmonics: 2,
            sampling_rate: 16_000,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            istft_n_fft: 8,
            istft_hop_len: 2,
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            source_resblock_kernel_sizes: vec![3, 3],
            source_resblock_dilation_sizes: vec![vec![1], vec![1]],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        };

        // F0Predictor (cond_channels = base_channels = 8, num_layers = 5).
        let mut f0_conv_weights: Vec<Vec<f32>> = vec![vec![0.0; 8 * 4 * 3]];
        for _ in 1..5 {
            f0_conv_weights.push(vec![0.0; 8 * 8 * 3]);
        }
        let f0_weights = F0PredictorWeights {
            conv_weights: f0_conv_weights,
            conv_biases: vec![vec![0.0; 8]; 5],
            linear_w: vec![0.0; 8],
            linear_b: vec![0.0; 1],
        };

        // ups: stage 0 [in=8, out=4, k=4], stage 1 [in=4, out=2, k=4]
        let ups_w = vec![vec![0.0; 8 * 4 * 4], vec![0.0; 4 * 2 * 4]];
        let ups_b = vec![vec![0.0; 4], vec![0.0; 2]];

        // source_downs: n_fft + 2 = 10; upstream downsample_us for
        // upsample_rates=[2, 2] resolves to [2, 1]. Stage 0: k=4/stride=2/pad=1;
        // stage 1: k=1/stride=1/pad=0 → 2*10*1 = 20.
        let n_fft_plus_2 = 10;
        let source_downs_w = vec![
            vec![0.0; 4 * n_fft_plus_2 * 4], // stage 0
            vec![0.0; 2 * n_fft_plus_2],     // stage 1
        ];
        let source_downs_b = vec![vec![0.0; 4], vec![0.0; 2]];

        let make_res_zero = |ch: usize, k: usize, n_branches: usize| ResBlockWeights {
            convs1_w: vec![vec![0.0; ch * ch * k]; n_branches],
            convs1_b: vec![vec![0.0; ch]; n_branches],
            convs2_w: vec![vec![0.0; ch * ch * k]; n_branches],
            convs2_b: vec![vec![0.0; ch]; n_branches],
            activations1_alpha: vec![vec![0.0; ch]; n_branches],
            activations2_alpha: vec![vec![0.0; ch]; n_branches],
        };

        let source_resblock_weights = vec![
            make_res_zero(4, 3, 1), // stage 0
            make_res_zero(2, 3, 1), // stage 1
        ];
        // resblocks: row-major [num_ups * num_kernels], num_kernels = 1.
        let resblock_weights = vec![make_res_zero(4, 3, 1), make_res_zero(2, 3, 1)];

        let weights = HiFTChainWeights {
            conv_pre_w: vec![0.0; 8 * 4 * 7],
            conv_pre_b: vec![0.0; 8],
            ups_w,
            ups_b,
            source_downs_w,
            source_downs_b,
            source_resblock_weights,
            resblock_weights,
            conv_post_w: vec![0.0; n_fft_plus_2 * 2 * 7],
            conv_post_b: vec![0.0; n_fft_plus_2],
            m_source_linear_w: vec![0.0; 3], // nb_harmonics + 1
            m_source_linear_b: 0.0,
            f0_predictor_weights: f0_weights,
        };

        (cfg, weights)
    }

    /// The chain builds from a valid synthesized-weight bundle and its
    /// config/sample-rate accessors surface the same values the caller
    /// supplied.
    #[test]
    fn hift_chain_new_accepts_small_synthesized_bundle() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg, weights).expect("small bundle must build");
        assert_eq!(chain.config().in_channels, 4);
        assert_eq!(chain.config().base_channels, 8);
        assert_eq!(chain.sample_rate(), 16_000);
    }

    /// The forward pass produces the exact upstream length contract:
    /// `t_mel * total_upsample_factor()` PCM samples. This is the shape
    /// invariant a caller needs when packing the result into a
    /// `SynthesizedAudio` — a silent shift here would be undetectable
    /// downstream.
    #[test]
    fn hift_chain_forward_output_length_matches_upstream_contract() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        for &t_mel in &[1usize, 2, 3, 5] {
            let mel = vec![0.0f32; cfg.in_channels as usize * t_mel];
            let audio = chain.forward(&mel, t_mel).expect("forward must succeed");
            assert_eq!(
                audio.len(),
                t_mel * cfg.total_upsample_factor() as usize,
                "t_mel = {t_mel}"
            );
        }
    }

    /// A mis-shaped mel must surface as an explicit InvalidArgument — not
    /// a silent shorter output or a panic. This mirrors the op-crate's
    /// `hift_generator_forward_rejects_wrong_mel_shape` pin at the
    /// integration boundary.
    #[test]
    fn hift_chain_forward_rejects_wrong_mel_shape() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        let bogus = vec![0.0f32; cfg.in_channels as usize * 4 - 1];
        let err = chain
            .forward(&bogus, 4)
            .expect_err("wrong-length must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    /// `t_mel = 0` is rejected up front (zero-frame synthesis is not a
    /// valid HiFTNet input — the op-crate's own contract). The chain must
    /// propagate that verbatim, not swallow it.
    #[test]
    fn hift_chain_forward_rejects_zero_t_mel() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg, weights).expect("build");
        let err = chain.forward(&[], 0).expect_err("t_mel=0 must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    /// Same input → same output, twice. The chain does not introduce
    /// hidden RNG; upstream's `NsfEntropy::Deterministic` posture holds
    /// through the wrapper.
    #[test]
    fn hift_chain_forward_is_deterministic_on_same_input() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        let t_mel = 4;
        let mel: Vec<f32> = (0..(cfg.in_channels as usize * t_mel))
            .map(|i| ((i % 7) as f32) * 0.03 - 0.05)
            .collect();
        let a = chain.forward(&mel, t_mel).expect("forward 1");
        let b = chain.forward(&mel, t_mel).expect("forward 2");
        assert_eq!(a, b, "wrapper must not introduce hidden state");
    }

    #[test]
    fn hift_chain_backend_cpu_preserves_scalar_route() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        let mel = vec![0.0; cfg.in_channels as usize * 2];
        assert_eq!(
            chain
                .forward_with_backend(&mel, 2, BackendKind::Cpu)
                .unwrap(),
            chain.forward(&mel, 2).unwrap()
        );
    }

    #[test]
    fn hift_chain_backend_cpu_nonzero_fixture_produces_pcm() {
        let (cfg, mut weights) = small_hift_chain_bundle();
        // Keep the fixture tiny, but give the terminal logits a finite
        // non-zero bias so this is a structural synthesis check rather than
        // another all-zero shape-only assertion.
        weights.conv_post_b.fill(0.25);
        let chain = HiFTChain::new(cfg.clone(), weights).expect("build");
        let mel = vec![0.1; cfg.in_channels as usize * 2];
        let audio = chain
            .forward_with_backend(&mel, 2, BackendKind::Cpu)
            .expect("non-zero fixture must synthesize");
        assert_eq!(audio.len(), 2 * cfg.total_upsample_factor() as usize);
        assert!(
            audio.iter().any(|sample| sample.abs() > 1.0e-7),
            "non-zero terminal logits must produce non-zero PCM"
        );
    }

    #[test]
    fn hift_chain_non_cpu_non_metal_is_explicitly_unsupported() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg, weights).expect("build");
        let err = chain
            .forward_with_backend(&[0.0; 4 * 2], 2, BackendKind::Cuda)
            .unwrap_err();
        assert!(matches!(err, VokraError::UnsupportedOp(_)), "{err:?}");
    }

    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn hift_chain_metal_without_apple_feature_is_backend_unavailable() {
        let (cfg, weights) = small_hift_chain_bundle();
        let chain = HiFTChain::new(cfg, weights).expect("build");
        let err = chain
            .forward_with_backend(&[0.0; 4 * 2], 2, BackendKind::Metal)
            .unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)), "{err:?}");
    }

    /// A weight bundle whose `ups_w` length disagrees with `upsample_rates`
    /// must be caught at `new` time. This is the fail-loud construction
    /// contract; the op-crate's own new() surfaces the error, and the
    /// wrapper propagates it verbatim.
    #[test]
    fn hift_chain_new_rejects_ups_weight_count_mismatch() {
        let (cfg, mut weights) = small_hift_chain_bundle();
        // Drop one ups weight — the count no longer matches
        // upsample_rates.len() == 2.
        weights.ups_w.pop();
        weights.ups_b.pop();
        let err = HiFTChain::new(cfg, weights).expect_err("mismatched ups must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn weight_norm_fold_uses_rows_for_convtranspose_layout() {
        // ConvTranspose1d stores [input_channel, output_channel, kernel],
        // and PyTorch weight_norm(dim=0) normalizes the input-channel rows.
        let g = [2.0, 3.0];
        let v = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0];
        let folded = fold_weight_norm(&g, &v, 2, 2, 2, "ups.0").expect("finite rows");
        assert_eq!(folded, vec![1.2, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn weight_norm_fold_rejects_zero_and_nonfinite_rows() {
        assert!(fold_weight_norm(&[1.0], &[0.0, 0.0], 1, 1, 2, "test").is_err());
        assert!(fold_weight_norm(&[f32::NAN], &[1.0], 1, 1, 1, "test").is_err());
        assert!(fold_weight_norm(&[1.0], &[f32::INFINITY], 1, 1, 1, "test").is_err());
    }

    #[test]
    fn hift_manifest_mapping_pins_count_and_weight_norm_pairs() {
        let manifest = expected_hift_manifest();
        assert_eq!(manifest.len(), HIFT_TENSOR_COUNT);
        assert_eq!(
            manifest
                .iter()
                .filter(|(name, _)| name.ends_with(".parametrizations.weight.original0"))
                .count(),
            HIFT_WEIGHT_NORM_PAIRS as usize
        );
        assert!(manifest.iter().any(|(name, shape)| name
            == "ups.0.parametrizations.weight.original1"
            && shape == &[512, 256, 16]));
    }

    #[test]
    fn hift_manifest_digest_array_matches_pinned_string() {
        let bytes = HIFT_MANIFEST_SHA256.as_bytes();
        assert_eq!(bytes.len(), HIFT_MANIFEST_DIGEST.len() * 2);
        let mut decoded = [0u8; 32];
        for (index, pair) in bytes.chunks_exact(2).enumerate() {
            let high = match pair[0] {
                b'0'..=b'9' => pair[0] - b'0',
                b'a'..=b'f' => pair[0] - b'a' + 10,
                _ => panic!("invalid manifest digest hex"),
            };
            let low = match pair[1] {
                b'0'..=b'9' => pair[1] - b'0',
                b'a'..=b'f' => pair[1] - b'a' + 10,
                _ => panic!("invalid manifest digest hex"),
            };
            decoded[index] = (high << 4) | low;
        }
        assert_eq!(decoded, HIFT_MANIFEST_DIGEST);
    }

    #[test]
    fn hift_from_gguf_rejects_arch_before_tensor_manifest() {
        let mut builder = vokra_core::gguf::GgufBuilder::new();
        builder
            .add_string(chunks::KEY_MODEL_ARCH, "cosyvoice2")
            .add_string(chunks::KEY_MODEL_NAME, HIFT_MODEL_NAME);
        let file = GgufFile::parse(builder.to_bytes().expect("serialize"))
            .expect("metadata-only GGUF parses");
        let error = HiFTChain::from_gguf(&file).expect_err("wrong arch must fail first");
        assert!(matches!(error, VokraError::ModelLoad(_)));
        assert!(error.to_string().contains("vokra.model.arch"));
    }

    /// A cloned `HiFTChain` must produce byte-identical PCM for the same
    /// mel as the original. `#[derive(Clone)]` on line 89 is a documented
    /// public contract — [`super::CosyVoice2Tts`] is intended to carry an
    /// `Option<HiFTChain>` field (see the type docstring on lines 66-74),
    /// so the wrapper *will* be cloned. This pins Clone as a deep,
    /// independent copy: a future refactor that turned `generator` into
    /// `Arc<Mutex<HiFTGenerator>>` (or any other shared-state form) would
    /// still typecheck and pass every existing test, but the original and
    /// its clone would silently share generator state and diverge under
    /// concurrent forward calls. Comparing element-wise PCM equality
    /// catches any such regression on a deterministic input.
    #[test]
    fn hift_chain_clone_produces_forward_equivalent_output() {
        let (cfg, weights) = small_hift_chain_bundle();
        let original = HiFTChain::new(cfg.clone(), weights).expect("build");
        let cloned = original.clone();
        // Accessors on the clone must mirror the original (config is
        // owned by the internal generator; a shallow-clone regression
        // that swapped in a fresh default config would trip this).
        assert_eq!(cloned.config().in_channels, original.config().in_channels);
        assert_eq!(cloned.sample_rate(), original.sample_rate());
        let t_mel = 4;
        let mel: Vec<f32> = (0..(cfg.in_channels as usize * t_mel))
            .map(|i| ((i % 7) as f32) * 0.03 - 0.05)
            .collect();
        let orig_pcm = original.forward(&mel, t_mel).expect("orig forward");
        let cloned_pcm = cloned.forward(&mel, t_mel).expect("cloned forward");
        assert_eq!(
            orig_pcm, cloned_pcm,
            "Clone must be a deep, independent copy of the generator",
        );
    }
}
