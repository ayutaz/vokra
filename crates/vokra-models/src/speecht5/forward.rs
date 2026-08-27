//! Native SpeechT5 text-to-mel generation and optional HiFi-GAN synthesis.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use vokra_core::gguf::GgufFile;
use vokra_core::{
    BackendKind, LicenseClass, Result, SynthesisRequest, SynthesizedAudio, TtsEngine, VokraError,
};

use crate::compute::{Compute, HotOp};
use crate::hifigan::HiFiGan;

use super::weights::{Attention, BatchNormConv, FeedForward, Linear, SpeechT5Weights};
use super::{
    DECODER_LAYERS, ENCODER_MAX_RELATIVE_POSITION, HIDDEN_SIZE, MAX_SPEECH_POSITIONS,
    MAX_TEXT_POSITIONS, NUM_MEL_BINS, PAD_TOKEN_ID, REDUCTION_FACTOR, SPEAKER_EMBEDDING_DIM,
    SPEECH_DECODER_POSTNET_KERNEL, SpeechT5Checkpoint, SpeechT5Config, SpeechT5Tokenizer,
};

const LABEL: &str = "SpeechT5-TTS";
const DETERMINISTIC_SEED: u64 = 0x5350_4545_4348_5435;
const BATCH_NORM_EPS: f32 = 1.0e-5;

/// Every backend-dispatched operation in the SpeechT5 text-to-mel path.
///
/// Host work is limited to shape/layout transforms, embedding lookup,
/// residual addition, fixed eval-mode BatchNorm affine application, dropout
/// mask generation, and the scalar stop decision. Learned reductions and all
/// postnet convolutions/activations dispatch through one selected backend.
pub const SPEECHT5_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Relu,
    HotOp::Tanh,
    HotOp::Conv1d,
];

/// Deterministic controls for the autoregressive text-to-mel loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechT5GenerationOptions {
    /// Sum-of-two-stop-probabilities threshold used by the official generator.
    pub threshold: f32,
    /// Minimum generated length as an encoder-length ratio.
    pub minlen_ratio: f32,
    /// Maximum generated length as an encoder-length ratio.
    pub maxlen_ratio: f32,
    /// Seed for the always-on decoder-prenet dropout masks.
    pub dropout_seed: u64,
}

impl Default for SpeechT5GenerationOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            minlen_ratio: 0.0,
            maxlen_ratio: 20.0,
            dropout_seed: DETERMINISTIC_SEED,
        }
    }
}

impl SpeechT5GenerationOptions {
    fn validate(self) -> Result<Self> {
        if !self.threshold.is_finite() || !(0.0..1.0).contains(&self.threshold) {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: stop threshold {} must be finite and inside (0, 1)",
                self.threshold
            )));
        }
        if !self.minlen_ratio.is_finite()
            || !self.maxlen_ratio.is_finite()
            || self.minlen_ratio < 0.0
            || self.maxlen_ratio <= 0.0
            || self.minlen_ratio > self.maxlen_ratio
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: invalid generation ratios min={} max={}",
                self.minlen_ratio, self.maxlen_ratio
            )));
        }
        Ok(self)
    }
}

/// Generated SpeechT5 log-mel spectrogram.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechT5Mel {
    before_postnet: Vec<f32>,
    values: Vec<f32>,
    frames: usize,
    decoder_steps: usize,
}

impl SpeechT5Mel {
    /// Postnet-refined frame-major `[frames, 80]` values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Raw frame-major `[frames, 80]` values before the convolutional postnet.
    #[must_use]
    pub fn before_postnet(&self) -> &[f32] {
        &self.before_postnet
    }

    /// Returns the number of generated mel frames.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Returns the fixed number of mel-frequency bins per frame.
    #[must_use]
    pub const fn bins(&self) -> usize {
        NUM_MEL_BINS
    }

    /// Returns the number of autoregressive decoder iterations.
    #[must_use]
    pub const fn decoder_steps(&self) -> usize {
        self.decoder_steps
    }

    fn channel_major(&self) -> Vec<f32> {
        frame_to_channel_major(&self.values, self.frames, NUM_MEL_BINS)
    }
}

/// Complete native SpeechT5 text-to-speech runtime.
pub struct SpeechT5Tts {
    config: SpeechT5Config,
    tokenizer: SpeechT5Tokenizer,
    weight_license: LicenseClass,
    weights: Box<SpeechT5Weights>,
    backend: BackendKind,
    vocoder: Option<HiFiGan>,
}

impl SpeechT5Tts {
    /// Strictly binds the canonical 393-tensor text-to-mel artifact.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = SpeechT5Checkpoint::from_gguf(file)?;
        let config = checkpoint.config().clone();
        let tokenizer = checkpoint.tokenizer().clone();
        let weight_license = checkpoint.weight_license();
        let weights = Box::new(SpeechT5Weights::load(file)?);
        Ok(Self {
            config,
            tokenizer,
            weight_license,
            weights,
            backend: BackendKind::Cpu,
            vocoder: None,
        })
    }

    /// Opens and binds a strict SpeechT5 GGUF on the CPU backend.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Loads the text-to-mel and companion SpeechT5 HiFi-GAN artifacts.
    pub fn from_gguf_with_vocoder(model: &GgufFile, vocoder: &GgufFile) -> Result<Self> {
        Self::from_gguf(model)?.with_vocoder(HiFiGan::from_gguf(vocoder)?)
    }

    /// Selects one backend for SpeechT5 and its attached vocoder.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        if let Some(vocoder) = self.vocoder.take() {
            self.vocoder = Some(vocoder.with_backend(backend));
        }
        self
    }

    /// Attaches the 80-bin, 16 kHz SpeechT5 HiFi-GAN companion.
    pub fn with_vocoder(mut self, vocoder: HiFiGan) -> Result<Self> {
        if vocoder.attrs().n_mels != NUM_MEL_BINS || vocoder.sample_rate() != 16_000 {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: vocoder must accept {NUM_MEL_BINS} mel bins at 16000 Hz; got {} bins at {} Hz",
                vocoder.attrs().n_mels,
                vocoder.sample_rate()
            )));
        }
        self.vocoder = Some(vocoder.with_backend(self.backend));
        Ok(self)
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the authenticated SpeechT5 topology.
    #[must_use]
    pub const fn config(&self) -> &SpeechT5Config {
        &self.config
    }

    /// Returns the tokenizer embedded in the model checkpoint.
    #[must_use]
    pub const fn tokenizer(&self) -> &SpeechT5Tokenizer {
        &self.tokenizer
    }

    /// Returns the checkpoint weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Reports whether a compatible HiFi-GAN vocoder is attached.
    #[must_use]
    pub const fn has_vocoder(&self) -> bool {
        self.vocoder.is_some()
    }

    /// Tokenizes text and generates the postnet-refined mel spectrogram.
    pub fn generate_text_mel(
        &self,
        text: &str,
        speaker_embedding: &[f32],
        options: SpeechT5GenerationOptions,
    ) -> Result<SpeechT5Mel> {
        let tokens = self.tokenizer.encode(text)?;
        self.generate_tokens_mel(&tokens, speaker_embedding, options)
    }

    /// Generates a mel spectrogram from exact SpeechT5 token ids.
    pub fn generate_tokens_mel(
        &self,
        tokens: &[u32],
        speaker_embedding: &[f32],
        options: SpeechT5GenerationOptions,
    ) -> Result<SpeechT5Mel> {
        let options = options.validate()?;
        validate_tokens(tokens)?;
        let speaker = normalize_speaker(speaker_embedding)?;
        let compute = Compute::for_backend(self.backend, SPEECHT5_HOT_OPS)?;
        let encoder_mask: Vec<bool> = tokens.iter().map(|token| *token != PAD_TOKEN_ID).collect();
        let encoder = encode(
            tokens,
            &encoder_mask,
            &self.weights,
            &compute,
            self.config.layer_norm_eps,
        )?;
        generate(
            &encoder,
            &encoder_mask,
            &speaker,
            &self.weights,
            &compute,
            self.config.layer_norm_eps,
            options,
        )
    }
}

impl TtsEngine for SpeechT5Tts {
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        validate_request(request)?;
        let speaker = request.speaker_embedding.as_deref().ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: request.speaker_embedding is required and must contain {SPEAKER_EMBEDDING_DIM} values"
            ))
        })?;
        let vocoder = self.vocoder.as_ref().ok_or_else(|| {
            VokraError::UnsupportedOp(format!(
                "{LABEL}: text-to-mel is available, but waveform synthesis requires the strict microsoft/speecht5_hifigan companion GGUF"
            ))
        })?;
        let dropout_seed = if request.deterministic {
            DETERMINISTIC_SEED
        } else {
            next_runtime_seed()
        };
        let mel = self.generate_text_mel(
            &request.text,
            speaker,
            SpeechT5GenerationOptions {
                dropout_seed,
                ..SpeechT5GenerationOptions::default()
            },
        )?;
        let channel_major = mel.channel_major();
        let samples = vocoder.decode(&channel_major, mel.frames())?;
        Ok(SynthesizedAudio::new(samples, vocoder.sample_rate()))
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

#[derive(Debug)]
struct ProjectedKv {
    key: Vec<f32>,
    value: Vec<f32>,
    rows: usize,
}

#[derive(Debug)]
struct DecoderLayerCache {
    self_attention: ProjectedKv,
    cross: ProjectedKv,
}

fn validate_tokens(tokens: &[u32]) -> Result<()> {
    if tokens.is_empty() || tokens.len() > MAX_TEXT_POSITIONS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: token count {} must be inside 1..={MAX_TEXT_POSITIONS}",
            tokens.len()
        )));
    }
    if let Some((index, token)) = tokens
        .iter()
        .enumerate()
        .find(|(_, token)| (**token as usize) >= super::VOCAB_SIZE)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: token id {token} at index {index} is outside vocabulary 0..{}",
            super::VOCAB_SIZE
        )));
    }
    if tokens.iter().all(|token| *token == PAD_TOKEN_ID) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: an all-padding token sequence has no attention source"
        )));
    }
    Ok(())
}

fn normalize_speaker(speaker: &[f32]) -> Result<Vec<f32>> {
    if speaker.len() != SPEAKER_EMBEDDING_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: speaker embedding length {} != {SPEAKER_EMBEDDING_DIM}",
            speaker.len()
        )));
    }
    if speaker.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: speaker embedding must contain only finite values"
        )));
    }
    let norm = speaker
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-12);
    Ok(speaker.iter().map(|value| value / norm).collect())
}

fn validate_request(request: &SynthesisRequest) -> Result<()> {
    if let Some(language) = request.language.as_deref() {
        if !matches!(language, "en" | "en-US" | "en_US") {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: language {language:?} is unsupported; the pinned checkpoint is English"
            )));
        }
    }
    if request.prosody_features.is_some() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prosody_features are not consumed by SpeechT5"
        )));
    }
    if request.style_vec.is_some() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: style_vec is not consumed; pass the 512-value x-vector through speaker_embedding"
        )));
    }
    if request.speaker_id.is_some() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: speaker_id is not consumed; pass an explicit x-vector through speaker_embedding"
        )));
    }
    Ok(())
}

fn encode(
    tokens: &[u32],
    mask: &[bool],
    weights: &SpeechT5Weights,
    compute: &Compute,
    eps: f32,
) -> Result<Vec<f32>> {
    let rows = tokens.len();
    let mut hidden = vec![0.0f32; rows * HIDDEN_SIZE];
    for (row, &token) in tokens.iter().enumerate() {
        let source = token as usize * HIDDEN_SIZE;
        let target = row * HIDDEN_SIZE;
        hidden[target..target + HIDDEN_SIZE]
            .copy_from_slice(&weights.encoder.token_embedding[source..source + HIDDEN_SIZE]);
        for column in 0..HIDDEN_SIZE {
            hidden[target + column] += weights.encoder.position_alpha
                * weights.encoder.positions[row * HIDDEN_SIZE + column];
        }
    }
    hidden = weights
        .encoder
        .initial_norm
        .forward(compute, &hidden, rows, HIDDEN_SIZE, eps)?;

    for layer in &weights.encoder.layers {
        let projected = project_kv(&layer.attention, compute, &hidden, rows)?;
        let attention = attention(
            &layer.attention,
            compute,
            &hidden,
            rows,
            &projected,
            Some(mask),
            Some(&weights.encoder.relative_positions),
        )?;
        add_in_place(&mut hidden, &attention)?;
        hidden = layer
            .attention_norm
            .forward(compute, &hidden, rows, HIDDEN_SIZE, eps)?;
        let feed_forward = feed_forward(&layer.feed_forward, compute, &hidden, rows)?;
        add_in_place(&mut hidden, &feed_forward)?;
        hidden = layer
            .final_norm
            .forward(compute, &hidden, rows, HIDDEN_SIZE, eps)?;
    }
    Ok(hidden)
}

fn generate(
    encoder: &[f32],
    encoder_mask: &[bool],
    speaker: &[f32],
    weights: &SpeechT5Weights,
    compute: &Compute,
    eps: f32,
    options: SpeechT5GenerationOptions,
) -> Result<SpeechT5Mel> {
    let encoder_rows = encoder_mask.len();
    if encoder.len() != encoder_rows * HIDDEN_SIZE {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: encoder shape mismatch"
        )));
    }
    let min_steps =
        ((encoder_rows as f32 * options.minlen_ratio) / REDUCTION_FACTOR as f32).floor() as usize;
    let max_steps =
        ((encoder_rows as f32 * options.maxlen_ratio) / REDUCTION_FACTOR as f32).floor() as usize;
    if max_steps == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: generation max length resolved to zero"
        )));
    }

    let mut caches = Vec::with_capacity(DECODER_LAYERS);
    for layer in &weights.decoder.layers {
        caches.push(DecoderLayerCache {
            self_attention: ProjectedKv {
                key: Vec::with_capacity(max_steps.min(MAX_SPEECH_POSITIONS) * HIDDEN_SIZE),
                value: Vec::with_capacity(max_steps.min(MAX_SPEECH_POSITIONS) * HIDDEN_SIZE),
                rows: 0,
            },
            cross: project_kv(&layer.cross_attention, compute, encoder, encoder_rows)?,
        });
    }

    let mut rng = SplitMix64(options.dropout_seed);
    let mut last_mel = vec![0.0f32; NUM_MEL_BINS];
    let mut before_postnet = Vec::new();
    let mut steps = 0usize;

    loop {
        if steps >= MAX_SPEECH_POSITIONS {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: generation did not stop within the checkpoint's {MAX_SPEECH_POSITIONS} decoder positions"
            )));
        }
        let mut hidden = decoder_prenet(
            &last_mel,
            steps,
            speaker,
            &weights.decoder.prenet,
            &weights.decoder.final_layer,
            weights.decoder.position_alpha,
            &weights.decoder.positions,
            &weights.decoder.speaker_projection,
            compute,
            &mut rng,
        )?;

        for (layer, cache) in weights.decoder.layers.iter().zip(caches.iter_mut()) {
            let new_key = layer.self_attention.k.forward(compute, &hidden, 1)?;
            let new_value = layer.self_attention.v.forward(compute, &hidden, 1)?;
            cache.self_attention.key.extend_from_slice(&new_key);
            cache.self_attention.value.extend_from_slice(&new_value);
            cache.self_attention.rows += 1;
            let self_attention = attention(
                &layer.self_attention,
                compute,
                &hidden,
                1,
                &cache.self_attention,
                None,
                None,
            )?;
            add_in_place(&mut hidden, &self_attention)?;
            hidden = layer
                .self_attention_norm
                .forward(compute, &hidden, 1, HIDDEN_SIZE, eps)?;

            let cross_attention = attention(
                &layer.cross_attention,
                compute,
                &hidden,
                1,
                &cache.cross,
                Some(encoder_mask),
                None,
            )?;
            add_in_place(&mut hidden, &cross_attention)?;
            hidden = layer
                .cross_attention_norm
                .forward(compute, &hidden, 1, HIDDEN_SIZE, eps)?;

            let feed_forward = feed_forward(&layer.feed_forward, compute, &hidden, 1)?;
            add_in_place(&mut hidden, &feed_forward)?;
            hidden = layer
                .final_norm
                .forward(compute, &hidden, 1, HIDDEN_SIZE, eps)?;
        }

        let spectrum = weights.postnet.feat_out.forward(compute, &hidden, 1)?;
        before_postnet.extend_from_slice(&spectrum);
        last_mel.copy_from_slice(&spectrum[NUM_MEL_BINS..NUM_MEL_BINS * REDUCTION_FACTOR]);
        let stop_logits = weights.postnet.prob_out.forward(compute, &hidden, 1)?;
        let stop_probability_sum: f32 = stop_logits.iter().copied().map(sigmoid).sum();
        steps += 1;

        if steps >= min_steps && (stop_probability_sum >= options.threshold || steps >= max_steps) {
            break;
        }
    }

    let frames = steps * REDUCTION_FACTOR;
    let values = postnet(&before_postnet, frames, &weights.postnet.layers, compute)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: generated non-finite postnet values"
        )));
    }
    Ok(SpeechT5Mel {
        before_postnet,
        values,
        frames,
        decoder_steps: steps,
    })
}

#[allow(clippy::too_many_arguments)]
fn decoder_prenet(
    mel: &[f32],
    position: usize,
    speaker: &[f32],
    layers: &[Linear],
    final_layer: &Linear,
    position_alpha: f32,
    positions: &[f32],
    speaker_projection: &Linear,
    compute: &Compute,
    rng: &mut SplitMix64,
) -> Result<Vec<f32>> {
    if mel.len() != NUM_MEL_BINS || position >= MAX_SPEECH_POSITIONS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: invalid decoder prenet input"
        )));
    }
    let mut hidden = mel.to_vec();
    for layer in layers {
        hidden = layer.forward(compute, &hidden, 1)?;
        let mut activated = vec![0.0f32; hidden.len()];
        compute.relu_f32(&hidden, &mut activated)?;
        apply_consistent_dropout(&mut activated, 0.5, rng);
        hidden = activated;
    }
    hidden = final_layer.forward(compute, &hidden, 1)?;
    for column in 0..HIDDEN_SIZE {
        hidden[column] += position_alpha * positions[position * HIDDEN_SIZE + column];
    }
    hidden.extend_from_slice(speaker);
    hidden = speaker_projection.forward(compute, &hidden, 1)?;
    let mut activated = vec![0.0f32; hidden.len()];
    compute.relu_f32(&hidden, &mut activated)?;
    Ok(activated)
}

fn project_kv(
    attention: &Attention,
    compute: &Compute,
    hidden: &[f32],
    rows: usize,
) -> Result<ProjectedKv> {
    Ok(ProjectedKv {
        key: attention.k.forward(compute, hidden, rows)?,
        value: attention.v.forward(compute, hidden, rows)?,
        rows,
    })
}

#[allow(clippy::too_many_arguments)]
fn attention(
    weights: &Attention,
    compute: &Compute,
    query_hidden: &[f32],
    query_rows: usize,
    projected: &ProjectedKv,
    source_mask: Option<&[bool]>,
    relative_positions: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if query_hidden.len() != query_rows * HIDDEN_SIZE
        || projected.key.len() != projected.rows * HIDDEN_SIZE
        || projected.value.len() != projected.rows * HIDDEN_SIZE
        || source_mask.is_some_and(|mask| mask.len() != projected.rows)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: attention shape mismatch"
        )));
    }
    let head_dim = HIDDEN_SIZE / weights.heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut query = weights.q.forward(compute, query_hidden, query_rows)?;
    for value in &mut query {
        *value *= scale;
    }
    let mut merged = vec![0.0f32; query_rows * HIDDEN_SIZE];
    let relative_bias = if let Some(table) = relative_positions {
        if query_rows != projected.rows
            || table.len() != 2 * ENCODER_MAX_RELATIVE_POSITION * head_dim
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: relative-position attention shape mismatch"
            )));
        }
        let mut all_bias = vec![0.0f32; weights.heads * query_rows * projected.rows];
        let mut position_transposed = vec![0.0f32; head_dim * projected.rows];
        let mut query_bias = vec![0.0f32; weights.heads * projected.rows];
        for query_row in 0..query_rows {
            for key_row in 0..projected.rows {
                let index = relative_position_index(query_row, key_row);
                for inner in 0..head_dim {
                    position_transposed[inner * projected.rows + key_row] =
                        table[index * head_dim + inner];
                }
            }
            compute.gemm_f32(
                weights.heads,
                projected.rows,
                head_dim,
                &query[query_row * HIDDEN_SIZE..(query_row + 1) * HIDDEN_SIZE],
                &position_transposed,
                None,
                &mut query_bias,
            )?;
            for head in 0..weights.heads {
                let target = (head * query_rows + query_row) * projected.rows;
                all_bias[target..target + projected.rows].copy_from_slice(
                    &query_bias[head * projected.rows..(head + 1) * projected.rows],
                );
            }
        }
        Some(all_bias)
    } else {
        None
    };

    for head in 0..weights.heads {
        let mut query_head = vec![0.0f32; query_rows * head_dim];
        let mut key_transposed = vec![0.0f32; head_dim * projected.rows];
        let mut value_head = vec![0.0f32; projected.rows * head_dim];
        for row in 0..query_rows {
            let source = row * HIDDEN_SIZE + head * head_dim;
            query_head[row * head_dim..(row + 1) * head_dim]
                .copy_from_slice(&query[source..source + head_dim]);
        }
        for row in 0..projected.rows {
            let source = row * HIDDEN_SIZE + head * head_dim;
            value_head[row * head_dim..(row + 1) * head_dim]
                .copy_from_slice(&projected.value[source..source + head_dim]);
            for inner in 0..head_dim {
                key_transposed[inner * projected.rows + row] = projected.key[source + inner];
            }
        }
        let mut scores = vec![0.0f32; query_rows * projected.rows];
        compute.gemm_f32(
            query_rows,
            projected.rows,
            head_dim,
            &query_head,
            &key_transposed,
            None,
            &mut scores,
        )?;

        if let Some(position_bias) = relative_bias.as_ref() {
            for query_row in 0..query_rows {
                let bias = (head * query_rows + query_row) * projected.rows;
                for key_row in 0..projected.rows {
                    scores[query_row * projected.rows + key_row] += position_bias[bias + key_row];
                }
            }
        }

        if let Some(mask) = source_mask {
            for row in 0..query_rows {
                for (column, keep) in mask.iter().enumerate() {
                    if !keep {
                        scores[row * projected.rows + column] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, query_rows, projected.rows)?;
        let mut context = vec![0.0f32; query_rows * head_dim];
        compute.gemm_f32(
            query_rows,
            head_dim,
            projected.rows,
            &probabilities,
            &value_head,
            None,
            &mut context,
        )?;
        for row in 0..query_rows {
            let target = row * HIDDEN_SIZE + head * head_dim;
            merged[target..target + head_dim]
                .copy_from_slice(&context[row * head_dim..(row + 1) * head_dim]);
        }
    }
    weights.out.forward(compute, &merged, query_rows)
}

fn feed_forward(
    weights: &FeedForward,
    compute: &Compute,
    hidden: &[f32],
    rows: usize,
) -> Result<Vec<f32>> {
    let intermediate = weights.intermediate.forward(compute, hidden, rows)?;
    let mut activated = vec![0.0f32; intermediate.len()];
    compute.gelu_f32(&intermediate, &mut activated)?;
    weights.output.forward(compute, &activated, rows)
}

fn postnet(
    frame_major: &[f32],
    frames: usize,
    layers: &[BatchNormConv],
    compute: &Compute,
) -> Result<Vec<f32>> {
    if frames == 0 || frame_major.len() != frames * NUM_MEL_BINS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: postnet input shape mismatch"
        )));
    }
    let mut hidden = frame_to_channel_major(frame_major, frames, NUM_MEL_BINS);
    for layer in layers {
        let mut convolved = vec![0.0f32; layer.output_channels * frames];
        compute.conv1d_f32(
            &hidden,
            layer.input_channels,
            frames,
            &layer.conv_weight,
            layer.output_channels,
            SPEECH_DECODER_POSTNET_KERNEL,
            None,
            1,
            (SPEECH_DECODER_POSTNET_KERNEL - 1) / 2,
            &mut convolved,
        )?;
        apply_batch_norm(layer, &mut convolved, frames)?;
        if layer.activation {
            let mut activated = vec![0.0f32; convolved.len()];
            compute.tanh_f32(&convolved, &mut activated)?;
            hidden = activated;
        } else {
            hidden = convolved;
        }
    }
    let residual = channel_to_frame_major(&hidden, frames, NUM_MEL_BINS);
    let mut output = frame_major.to_vec();
    add_in_place(&mut output, &residual)?;
    Ok(output)
}

fn apply_batch_norm(layer: &BatchNormConv, values: &mut [f32], frames: usize) -> Result<()> {
    if values.len() != layer.output_channels * frames
        || layer.norm_weight.len() != layer.output_channels
        || layer.norm_bias.len() != layer.output_channels
        || layer.running_mean.len() != layer.output_channels
        || layer.running_var.len() != layer.output_channels
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: postnet BatchNorm shape mismatch"
        )));
    }
    for channel in 0..layer.output_channels {
        let variance = layer.running_var[channel];
        if !variance.is_finite() || variance < 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: postnet BatchNorm channel {channel} has invalid variance {variance}"
            )));
        }
        let scale = layer.norm_weight[channel] / (variance + BATCH_NORM_EPS).sqrt();
        let shift = layer.norm_bias[channel] - layer.running_mean[channel] * scale;
        for value in &mut values[channel * frames..(channel + 1) * frames] {
            *value = *value * scale + shift;
        }
    }
    Ok(())
}

fn relative_position_index(query: usize, key: usize) -> usize {
    let difference = query as isize - key as isize;
    (difference.clamp(
        -(ENCODER_MAX_RELATIVE_POSITION as isize),
        ENCODER_MAX_RELATIVE_POSITION as isize - 1,
    ) + ENCODER_MAX_RELATIVE_POSITION as isize) as usize
}

fn frame_to_channel_major(values: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; values.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            transposed[channel * frames + frame] = values[frame * channels + channel];
        }
    }
    transposed
}

fn channel_to_frame_major(values: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; values.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            transposed[frame * channels + channel] = values[channel * frames + frame];
        }
    }
    transposed
}

fn add_in_place(target: &mut [f32], source: &[f32]) -> Result<()> {
    if target.len() != source.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: residual shape mismatch {} != {}",
            target.len(),
            source.len()
        )));
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target += *source;
    }
    Ok(())
}

fn apply_consistent_dropout(values: &mut [f32], probability: f32, rng: &mut SplitMix64) {
    let scale = (1.0 - probability).recip();
    for value in values {
        if rng.next_unit_f32() < probability {
            *value *= scale;
        } else {
            *value = 0.0;
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn next_runtime_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64);
    time ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_position_indices_match_official_clamp() {
        assert_eq!(relative_position_index(0, 0), 160);
        assert_eq!(relative_position_index(1, 0), 161);
        assert_eq!(relative_position_index(0, 1), 159);
        assert_eq!(relative_position_index(0, 1_000), 0);
        assert_eq!(relative_position_index(1_000, 0), 319);
    }

    #[test]
    fn layout_transpose_round_trips() {
        let frame_major = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channel_major = frame_to_channel_major(&frame_major, 2, 3);
        assert_eq!(channel_major, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(channel_to_frame_major(&channel_major, 2, 3), frame_major);
    }

    #[test]
    fn dropout_is_seeded_and_scaled() {
        let mut first = vec![1.0f32; 32];
        let mut second = first.clone();
        apply_consistent_dropout(&mut first, 0.5, &mut SplitMix64(7));
        apply_consistent_dropout(&mut second, 0.5, &mut SplitMix64(7));
        assert_eq!(first, second);
        assert!(first.iter().all(|value| *value == 0.0 || *value == 2.0));
        assert!(first.contains(&0.0));
        assert!(first.contains(&2.0));
    }

    #[test]
    fn stable_sigmoid_handles_large_logits() {
        assert_eq!(sigmoid(1_000.0), 1.0);
        assert_eq!(sigmoid(-1_000.0), 0.0);
        assert_eq!(sigmoid(0.0), 0.5);
    }
}
