//! Internal, source-shaped CosyVoice2 batch-one route.
//!
//! This module mirrors the *ordering and shape contract* of the fixed
//! upstream `Qwen2LM` → causal flow → HiFT pipeline without pretending that a
//! partial GGUF is a production model.  The real checkpoint is a composite
//! (`llm.pt`, `flow.pt`, `hift.pt`, and the speech tokenizer); the public
//! loader therefore remains fail-closed.  Callers of this internal harness
//! inject the three reviewed numeric components and an AR random source.  The
//! fixed-flow noise packet is currently structural-only; a manifest-bound
//! binder must be added before this route can be enabled.
//!
//! The source contract is intentionally explicit:
//!
//! ```text
//! [SOS, prompt text embeddings, target text embeddings, TASK,
//!  prompt speech embeddings]
//!   -> speech token AR (yielded IDs are strictly `< 6561`)
//!   -> causal flow mel (the official flow returns generated-only frames)
//!   -> HiFTNet 24 kHz PCM
//! ```
//!
//! No fallback component is constructed here.  In particular, a missing
//! component is an error and not a zero tensor, identity codec, fresh flow
//! noise, or hidden RNG.

use vokra_core::{Result, VokraError};

/// CosyVoice2 speech-token vocabulary from the fixed upstream config.
pub const SPEECH_TOKEN_VOCAB_SIZE: usize = 6_561;
pub const NATIVE_TERMINAL_EOS: u32 = SPEECH_TOKEN_VOCAB_SIZE as u32;
pub const VLLM_STOP_TOKEN_IDS: [u32; 3] = [
    SPEECH_TOKEN_VOCAB_SIZE as u32,
    SPEECH_TOKEN_VOCAB_SIZE as u32 + 1,
    SPEECH_TOKEN_VOCAB_SIZE as u32 + 2,
];
/// Source config token/mel ratio.  Final generated frame count is taken from
/// the same-execution `h.shape[1] - prompt_feat.shape[1]` relation, not
/// inferred universally from this configuration value.
pub const TOKEN_MEL_RATIO: usize = 2;
/// The upstream HiFT chain emits 24 kHz audio.
pub const SAMPLE_RATE: u32 = 24_000;
/// The fixed upstream CFM buffer has 50*300 frames per channel.
pub const FIXED_FLOW_NOISE_FRAMES: usize = 15_000;
pub const OFFICIAL_REFERENCE_FORMAT: &str = "vokra-cosyvoice2-official-reference-v2";
pub const OFFICIAL_SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
pub const OFFICIAL_MODEL_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";

/// Fixed flow/CFM axes authenticated from the pinned `cosyvoice2.yaml`.
/// LLM axes are intentionally absent here: those come from the bundled Qwen
/// config/checkpoint and must be authenticated by the composite binder.
pub const FLOW_INPUT_SIZE: usize = 512;
pub const FLOW_OUTPUT_SIZE: usize = 80;
pub const FLOW_SPEAKER_DIM: usize = 192;
pub const FLOW_PRELOOKAHEAD: usize = 3;
pub const FLOW_ENCODER_LAYERS: usize = 6;
pub const FLOW_ENCODER_HEADS: usize = 8;
pub const FLOW_ENCODER_FFN: usize = 2_048;
pub const CFM_IN_CHANNELS: usize = 240;
pub const CFM_ESTIMATOR_IN_CHANNELS: usize = 320;
pub const CFM_ESTIMATOR_CHANNELS: usize = 256;
pub const CFM_ESTIMATOR_BLOCKS: usize = 4;
pub const CFM_ESTIMATOR_MID_BLOCKS: usize = 12;
pub const CFM_ESTIMATOR_HEADS: usize = 8;
pub const CFM_INFERENCE_STEPS: usize = 10;
pub const CFM_INFERENCE_CFG_RATE: f32 = 0.7;

/// Fixed HiFTNet axes authenticated from the pinned `cosyvoice2.yaml`.
pub const HIFT_BASE_CHANNELS: usize = 512;
pub const HIFT_HARMONICS: usize = 8;
pub const HIFT_UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
pub const HIFT_UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
pub const HIFT_ISTFT_NFFT: usize = 16;
pub const HIFT_ISTFT_HOP: usize = 4;

/// The upstream `ras_sampling` defaults in `cosyvoice2.yaml`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub top_p: f32,
    pub top_k: usize,
    pub win_size: usize,
    pub tau_r: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            top_p: 0.8,
            top_k: 25,
            win_size: 10,
            tau_r: 0.1,
        }
    }
}

/// A caller-owned random source for autoregressive speech-token sampling.
/// Flow noise is not drawn through this trait: the fixed upstream
/// `CausalConditionalCFM` creates `rand_noise = torch.randn([1, 80, 50 * 300])`
/// after seeding with 0 and slices its prefix on every inference call.
pub trait RandomSource {
    /// Returns the next uniformly distributed value in `[0, 1)`.
    fn next_f32(&mut self) -> f32;
}

/// A captured prefix of upstream CausalConditionalCFM's fixed noise packet,
/// stored in `[80, frames]` row-major layout. Arbitrary caller noise is not a
/// CosyVoice2 parity oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedFlowNoise {
    data: Vec<f32>,
    frames: usize,
    authenticated: bool,
}

impl FixedFlowNoise {
    /// Builds a packet corresponding to the source buffer's batch-one slice.
    /// Constructs a structural test packet. It is deliberately not accepted
    /// as a parity oracle by the route.  There is intentionally no public
    /// promotion constructor: a digest calculated by a caller is not
    /// authentication.  Until the fixed source/model/artifact manifest
    /// binder exists, the native route remains structurally blocked.
    pub fn new(data: Vec<f32>, frames: usize) -> Result<Self> {
        Self::validate(data, frames).map(|(data, frames)| Self {
            data,
            frames,
            authenticated: false,
        })
    }

    fn validate(data: Vec<f32>, frames: usize) -> Result<(Vec<f32>, usize)> {
        let expected = 80usize.checked_mul(frames).ok_or_else(|| {
            VokraError::InvalidArgument(
                "cosyvoice2 native: flow-noise frame count overflows".into(),
            )
        })?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 native: fixed flow noise has {} values, expected 80 * {frames} = {expected}",
                data.len()
            )));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: fixed flow noise contains a non-finite value".into(),
            ));
        }
        if data.iter().all(|&value| value == 0.0) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: fixed flow noise cannot be an all-zero placeholder".into(),
            ));
        }
        Ok((data, frames))
    }

    /// Returns the exact prefix used by upstream `rand_noise[:, :, :T]`.
    pub fn prefix(&self, frames: usize) -> Result<Self> {
        if frames > self.frames {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 native: fixed flow noise has {} frames, cannot slice {frames}",
                self.frames
            )));
        }
        let mut prefix = Self::new(
            (0..80)
                .flat_map(|channel| {
                    let start = channel * self.frames;
                    self.data[start..start + frames].iter().copied()
                })
                .collect(),
            frames,
        )?;
        prefix.authenticated = self.authenticated;
        Ok(prefix)
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    pub fn frames(&self) -> usize {
        self.frames
    }

    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.authenticated
    }
}

// Small, dependency-free SHA-256 retained for the future manifest binder. It
// is never treated as authentication by itself. The byte order matches the
// portable binary artifacts emitted by the official reference adapter
// (`float32` little endian).
fn sha256_f32_le(values: &[f32]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut bytes = Vec::with_capacity(values.len() * 4 + 72);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let bit_len = (bytes.len() as u64) * 8;
    bytes.push(0x80);
    while bytes.len() % 64 != 56 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&bit_len.to_be_bytes());
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    for block in bytes.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in w[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        ) = (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (state, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *state = state.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

/// One row in the exact Qwen2LM prompt sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Sos,
    PromptText,
    TargetText,
    Task,
    PromptSpeech,
}

/// Flattened, row-major `inputs_embeds` passed to the official Qwen wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmInputSequence {
    embeddings: Vec<f32>,
    kinds: Vec<InputKind>,
    row_width: usize,
}

impl LlmInputSequence {
    /// Builds `[sos, prompt text, target text, task, prompt speech]`.
    ///
    /// The LLM width is deliberately supplied by the authenticated checkpoint
    /// rather than guessed from a generic Qwen config.
    pub fn new(
        row_width: usize,
        sos: &[f32],
        prompt_text: &[f32],
        target_text: &[f32],
        task: &[f32],
        prompt_speech: &[f32],
    ) -> Result<Self> {
        if row_width == 0 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: authenticated Qwen input width must be non-zero".into(),
            ));
        }
        let parts = [
            (InputKind::Sos, sos),
            (InputKind::PromptText, prompt_text),
            (InputKind::TargetText, target_text),
            (InputKind::Task, task),
            (InputKind::PromptSpeech, prompt_speech),
        ];
        if sos.len() != row_width || task.len() != row_width {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 native: SOS and task embeddings must each contain exactly one row of width {row_width}"
            )));
        }
        let mut embeddings = Vec::new();
        let mut kinds = Vec::new();
        for (kind, rows) in parts {
            if rows.len() % row_width != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 native: {kind:?} embeddings have {} values, not a multiple of authenticated width {row_width}",
                    rows.len()
                )));
            }
            let n_rows = rows.len() / row_width;
            if rows.iter().any(|v| !v.is_finite()) {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 native: {kind:?} embeddings contain a non-finite value"
                )));
            }
            embeddings.extend_from_slice(rows);
            kinds.extend(std::iter::repeat_n(kind, n_rows));
        }
        if kinds.first() != Some(&InputKind::Sos)
            || kinds.get(1 + prompt_text.len() / row_width + target_text.len() / row_width)
                != Some(&InputKind::Task)
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: prompt sequence ordering is malformed".into(),
            ));
        }
        Ok(Self {
            embeddings,
            kinds,
            row_width,
        })
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.kinds.len()
    }

    #[must_use]
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    #[must_use]
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    #[must_use]
    pub fn kinds(&self) -> &[InputKind] {
        &self.kinds
    }
}

/// Batch-one conditioning passed to the internal route.
#[derive(Debug, Clone)]
pub struct BatchOneConditioning {
    /// Authenticated Qwen input width (`inputs_embeds` last dimension).
    pub llm_input_width: usize,
    pub sos_embedding: Vec<f32>,
    pub prompt_text_embeddings: Vec<f32>,
    pub target_text_embeddings: Vec<f32>,
    pub prompt_speech_embeddings: Vec<f32>,
    pub task_embedding: Vec<f32>,
    /// Prompt mel in row-major `[80, prompt_mel_frames]` layout.
    pub prompt_mel: Vec<f32>,
    pub prompt_mel_frames: usize,
    /// Keep the upstream streaming/finalize decision visible to the flow
    /// component.  The official flow performs prompt removal itself when
    /// `finalize=true`; the route validates that returned relation and does
    /// not slice the prompt a second time.
    pub streaming: bool,
    pub finalize: bool,
}

impl BatchOneConditioning {
    fn sequence(&self) -> Result<LlmInputSequence> {
        LlmInputSequence::new(
            self.llm_input_width,
            &self.sos_embedding,
            &self.prompt_text_embeddings,
            &self.target_text_embeddings,
            &self.task_embedding,
            &self.prompt_speech_embeddings,
        )
    }

    fn target_text_len(&self) -> Result<usize> {
        if self.llm_input_width == 0
            || self.target_text_embeddings.len() % self.llm_input_width != 0
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: target text embedding rows are malformed".into(),
            ));
        }
        let n = self.target_text_embeddings.len() / self.llm_input_width;
        if n == 0 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: target text must contain at least one authenticated token"
                    .into(),
            ));
        }
        Ok(n)
    }

    fn validate_prompt_mel(&self) -> Result<()> {
        let expected = 80usize.checked_mul(self.prompt_mel_frames).ok_or_else(|| {
            VokraError::InvalidArgument(
                "cosyvoice2 native: prompt mel frame count overflows".into(),
            )
        })?;
        if self.prompt_mel.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 native: prompt mel has {} values, expected 80 * {} = {expected}",
                self.prompt_mel.len(),
                self.prompt_mel_frames
            )));
        }
        if self.prompt_mel.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: prompt mel contains a non-finite value".into(),
            ));
        }
        Ok(())
    }
}

/// Output of the official causal flow encoder/CFM after its prompt slice.
#[derive(Debug, Clone)]
pub struct GeneratedMel {
    /// Row-major `[80, generated_frames]`. The upstream
    /// `CausalMaskedDiffWithXvec.inference` returns `feat[:, :, mel_len1:]`;
    /// callers must not remove prompt frames a second time.
    pub data: Vec<f32>,
    pub frames: usize,
    /// Number of prompt frames removed by the official flow implementation.
    pub prompt_frames_removed: usize,
    /// The full frame count immediately before the source's prompt slice.
    pub full_frames_before_prompt_slice: usize,
}

/// The official Qwen2LM wrapper yields only speech IDs `< 6561`. EOS (6561)
/// terminates without being yielded; the two IDs above EOS are special
/// control outputs and are continued over at the wrapper position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    Eos,
    MaxTokens,
}

/// One authenticated sampling observation from the official wrapper.
///
/// The upstream sampler may call its sampling function repeatedly while EOS
/// is disallowed.  Therefore this is an attempt-level record, not merely one
/// record per yielded token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingCallEvidence {
    pub call_index: usize,
    pub generation_step: usize,
    pub attempt_index: usize,
    pub selected_token: u32,
    pub ignore_eos: bool,
    pub decoded_count: usize,
    pub yielded: bool,
    pub skipped: bool,
    /// An EOS selected while `ignore_eos=true`; the official sampler retries
    /// inside this same outer generation step and does not terminate.
    pub ignored_eos: bool,
    /// A terminal EOS selected with `ignore_eos=false`.
    pub stop: bool,
}

/// One exact `Qwen2LM.forward_one_step` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmCallEvidence {
    pub call_index: usize,
    pub input_rows: usize,
    pub output_rows: usize,
}

/// Output boundary of the official `Qwen2LM.inference_wrapper`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechGeneration {
    pub yielded_tokens: Vec<u32>,
    pub termination: TerminationReason,
    pub min_tokens: usize,
    pub max_tokens: usize,
    /// Configured vLLM stops (all three upstream control IDs).
    pub configured_vllm_stop_token_ids: Vec<u32>,
    /// Effective native non-vLLM terminal: EOS only.
    pub native_terminal_eos: u32,
    pub sampled_tokens: Vec<u32>,
    pub llm_calls: Vec<LlmCallEvidence>,
    pub sampling_calls: Vec<SamplingCallEvidence>,
}

impl SpeechGeneration {
    pub fn validate(self) -> Result<Self> {
        let eos = SPEECH_TOKEN_VOCAB_SIZE as u32;
        if self.max_tokens == 0 || self.min_tokens > self.max_tokens {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: invalid authenticated generation bounds".into(),
            ));
        }
        if self.configured_vllm_stop_token_ids != VLLM_STOP_TOKEN_IDS
            || self.native_terminal_eos != eos
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: vLLM stops and native terminal EOS do not match the official contract".into(),
            ));
        }
        if self.sampled_tokens.len() != self.sampling_calls.len() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: sampled-token and sampling-call counts differ".into(),
            ));
        }
        if self.llm_calls.is_empty() || self.llm_calls.len() > self.max_tokens {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: invalid authenticated LLM call count".into(),
            ));
        }
        if self.llm_calls.iter().enumerate().any(|(expected, call)| {
            call.call_index != expected
                || call.input_rows == 0
                || call.output_rows != call.input_rows
        }) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: LLM call evidence is not exact and ordered".into(),
            ));
        }
        let mut expected_yielded = Vec::new();
        let mut calls_per_step = vec![0usize; self.llm_calls.len()];
        let mut accepted_step = vec![false; self.llm_calls.len()];
        let mut last_step = None;
        let mut final_step_tokens = vec![None; self.llm_calls.len()];
        for (expected, call) in self.sampling_calls.iter().enumerate() {
            if call.call_index != expected
                || call.generation_step >= self.llm_calls.len()
                || call.attempt_index != calls_per_step[call.generation_step]
                || call.decoded_count != expected_yielded.len()
            {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: sampling evidence is not call-order aligned".into(),
                ));
            }
            match last_step {
                None => {
                    if call.generation_step != 0 {
                        return Err(VokraError::InvalidArgument(
                            "cosyvoice2 native: sampling trace does not start at outer step zero"
                                .into(),
                        ));
                    }
                }
                Some(previous) if call.generation_step > previous + 1 => {
                    return Err(VokraError::InvalidArgument(
                        "cosyvoice2 native: sampling trace skips an outer generation step".into(),
                    ));
                }
                _ => {}
            }
            last_step = Some(call.generation_step);
            if accepted_step[call.generation_step] {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: sampling continued after an accepted outer-step result"
                        .into(),
                ));
            }
            calls_per_step[call.generation_step] += 1;
            let eos_selected = call.selected_token == eos;
            if call.selected_token > eos + 2
                || call.yielded != (call.selected_token < eos)
                || call.skipped != (call.selected_token > eos)
                || call.ignored_eos != (eos_selected && call.ignore_eos)
                || call.stop != (eos_selected && !call.ignore_eos)
                || call.ignore_eos != (call.generation_step < self.min_tokens)
                || call.selected_token != self.sampled_tokens[expected]
            {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: sampled/control/EOS flags do not match source semantics"
                        .into(),
                ));
            }
            if call.yielded {
                expected_yielded.push(call.selected_token);
            }
            if calls_per_step[call.generation_step] > 1
                && final_step_tokens[call.generation_step] != Some(eos)
            {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: a same-step retry must follow an ignored EOS".into(),
                ));
            }
            if !call.ignored_eos {
                accepted_step[call.generation_step] = true;
            }
            final_step_tokens[call.generation_step] = Some(call.selected_token);
        }
        if calls_per_step
            .iter()
            .zip(accepted_step.iter())
            .any(|(count, accepted)| *count == 0 || !accepted)
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: each outer step must end with one accepted sampling result"
                    .into(),
            ));
        }
        if expected_yielded != self.yielded_tokens {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: yielded tokens do not equal the filtered sampled stream".into(),
            ));
        }
        if self.sampling_calls.is_empty() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: speech generation lacks authenticated sampling call evidence"
                    .into(),
            ));
        }
        let last = self.sampling_calls.last().map(|call| call.selected_token);
        let last_step = self.sampling_calls.last().map(|call| call.generation_step);
        match self.termination {
            TerminationReason::Eos
                if last != Some(eos)
                    || self.sampling_calls.last().map_or(true, |call| !call.stop)
                    || last_step.map_or(true, |step| step < self.min_tokens) =>
            {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: EOS termination lacks a final EOS sampling observation"
                        .into(),
                ));
            }
            TerminationReason::MaxTokens
                if last == Some(eos) || self.llm_calls.len() != self.max_tokens =>
            {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: max-token termination does not consume max_tokens steps"
                        .into(),
                ));
            }
            _ => {}
        }
        for step in 1..self.llm_calls.len() {
            let previous = final_step_tokens[step - 1].ok_or_else(|| {
                VokraError::InvalidArgument(
                    "cosyvoice2 native: missing previous outer-step sample".into(),
                )
            })?;
            let expected_rows = if previous < eos {
                1
            } else if previous > eos {
                self.llm_calls[step - 1].input_rows
            } else {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: LLM call follows terminal EOS".into(),
                ));
            };
            if self.llm_calls[step].input_rows != expected_rows {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice2 native: lm_input row transition does not match source state".into(),
                ));
            }
        }
        Ok(self)
    }
}

/// Numeric components injected by a future authenticated composite binder.
pub trait BatchOneModel {
    /// Runs the official Qwen2LM AR path over `inputs_embeds`.
    fn generate_speech_tokens(
        &self,
        input: &LlmInputSequence,
        min_tokens: usize,
        max_tokens: usize,
        sampling: SamplingConfig,
        random: &mut dyn RandomSource,
    ) -> Result<SpeechGeneration>;

    /// Runs causal flow with prompt conditioning and the upstream streaming /
    /// finalize choice. The official flow returns generated-only frames after
    /// removing `prompt_mel_frames`; a pre-slice tap belongs in reference
    /// evidence and must never be sliced again by this route.
    fn generate_generated_mel(
        &self,
        speech_tokens: &[u32],
        prompt_mel: &[f32],
        prompt_mel_frames: usize,
        streaming: bool,
        finalize: bool,
        flow_noise: &FixedFlowNoise,
    ) -> Result<GeneratedMel>;
}

/// Terminal mel-to-PCM component.  The implementation for [`HiFTChain`] is
/// provided in this module; a binder must still supply real checkpoint
/// weights before it can be used by production code.
pub trait MelToPcm {
    fn sample_rate(&self) -> u32;
    fn decode(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>>;
}

/// Result of one source-shaped batch-one synthesis.
#[derive(Debug, Clone)]
pub struct BatchOneOutput {
    pub speech_tokens: Vec<u32>,
    /// Generated-only mel, after removing the prompt conditioning frames.
    pub mel: Vec<f32>,
    pub mel_frames: usize,
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

/// Internal batch-one route.  This is not a public production loader.
pub struct BatchOneRoute<'a, M: BatchOneModel, D: MelToPcm> {
    model: &'a M,
    decoder: &'a D,
    sampling: SamplingConfig,
}

impl<'a, M: BatchOneModel, D: MelToPcm> BatchOneRoute<'a, M, D> {
    pub fn new(model: &'a M, decoder: &'a D) -> Result<Self> {
        if decoder.sample_rate() != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 native: terminal decoder is {} Hz, expected authenticated {} Hz",
                decoder.sample_rate(),
                SAMPLE_RATE
            )));
        }
        Ok(Self {
            model,
            decoder,
            sampling: SamplingConfig::default(),
        })
    }

    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }

    /// Runs one request while preserving the upstream sequence and slicing
    /// contracts.  No random draw occurs in this function itself.
    pub fn synthesize(
        &self,
        conditioning: &BatchOneConditioning,
        flow_noise: &FixedFlowNoise,
        random: &mut dyn RandomSource,
    ) -> Result<BatchOneOutput> {
        if !self.sampling.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.sampling.top_p)
            || self.sampling.top_p == 0.0
            || self.sampling.top_k == 0
            || self.sampling.win_size == 0
            || !self.sampling.tau_r.is_finite()
            || self.sampling.tau_r <= 0.0
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: invalid ras_sampling parameters".into(),
            ));
        }
        if conditioning.streaming || !conditioning.finalize {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: only official non-streaming finalize=true flow evidence is supported"
                    .into(),
            ));
        }
        conditioning.validate_prompt_mel()?;
        let target_text_len = conditioning.target_text_len()?;
        let min_tokens = target_text_len.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument("cosyvoice2 native: min token bound overflows".into())
        })?;
        let max_tokens = target_text_len.checked_mul(20).ok_or_else(|| {
            VokraError::InvalidArgument("cosyvoice2 native: max token bound overflows".into())
        })?;
        let input = conditioning.sequence()?;
        if !flow_noise.is_authenticated() || flow_noise.frames() != FIXED_FLOW_NOISE_FRAMES {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: parity requires the authenticated full 1x80x15000 source rand_noise artifact"
                    .into(),
            ));
        }
        let generation = self
            .model
            .generate_speech_tokens(&input, min_tokens, max_tokens, self.sampling, random)?
            .validate()?;
        let speech_tokens = generation.yielded_tokens;
        let generated = self.model.generate_generated_mel(
            &speech_tokens,
            &conditioning.prompt_mel,
            conditioning.prompt_mel_frames,
            conditioning.streaming,
            conditioning.finalize,
            flow_noise,
        )?;
        if generated.prompt_frames_removed != conditioning.prompt_mel_frames
            || generated.full_frames_before_prompt_slice
                != generated.frames + conditioning.prompt_mel_frames
            || generated.data.len() != 80 * generated.frames
        {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 native: flow returned generated={} removed={} full={} values={}, expected removed={} full={}",
                generated.frames,
                generated.prompt_frames_removed,
                generated.full_frames_before_prompt_slice,
                generated.data.len(),
                conditioning.prompt_mel_frames,
                conditioning.prompt_mel_frames + generated.frames
            )));
        }
        if generated.data.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: causal flow returned non-finite mel".into(),
            ));
        }
        let generated_frames = generated.frames;
        let generated_mel = generated.data;
        let pcm = self.decoder.decode(&generated_mel, generated_frames)?;
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 native: HiFT decoder returned non-finite PCM".into(),
            ));
        }
        Ok(BatchOneOutput {
            speech_tokens,
            mel: generated_mel,
            mel_frames: generated_frames,
            pcm,
            sample_rate: SAMPLE_RATE,
        })
    }
}

impl MelToPcm for super::HiFTChain {
    fn sample_rate(&self) -> u32 {
        super::HiFTChain::sample_rate(self)
    }

    fn decode(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        super::HiFTChain::forward(self, mel, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Random;
    impl RandomSource for Random {
        fn next_f32(&mut self) -> f32 {
            panic!("route must not draw randomness itself")
        }
    }

    struct Decoder;
    impl MelToPcm for Decoder {
        fn sample_rate(&self) -> u32 {
            SAMPLE_RATE
        }
        fn decode(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
            assert_eq!(mel.len(), 80 * frames);
            Ok(vec![0.0; frames * 960])
        }
    }

    struct Model {
        termination: TerminationReason,
        bad_frames: bool,
        seen: std::cell::RefCell<Vec<InputKind>>,
    }
    impl BatchOneModel for Model {
        fn generate_speech_tokens(
            &self,
            input: &LlmInputSequence,
            min_tokens: usize,
            _max_tokens: usize,
            _sampling: SamplingConfig,
            _random: &mut dyn RandomSource,
        ) -> Result<SpeechGeneration> {
            *self.seen.borrow_mut() = input.kinds().to_vec();
            let mut out = vec![7; min_tokens];
            Ok(SpeechGeneration {
                yielded_tokens: std::mem::take(&mut out),
                termination: self.termination,
                min_tokens,
                max_tokens: min_tokens,
                configured_vllm_stop_token_ids: VLLM_STOP_TOKEN_IDS.to_vec(),
                native_terminal_eos: NATIVE_TERMINAL_EOS,
                sampled_tokens: {
                    let mut tokens = vec![7; min_tokens];
                    if self.termination == TerminationReason::Eos {
                        tokens.push(SPEECH_TOKEN_VOCAB_SIZE as u32);
                    }
                    tokens
                },
                llm_calls: vec![LlmCallEvidence {
                    call_index: 0,
                    input_rows: input.rows(),
                    output_rows: input.rows(),
                }],
                sampling_calls: vec![SamplingCallEvidence {
                    call_index: 0,
                    generation_step: 0,
                    attempt_index: 0,
                    selected_token: if self.termination == TerminationReason::Eos {
                        SPEECH_TOKEN_VOCAB_SIZE as u32
                    } else {
                        7
                    },
                    ignore_eos: true,
                    decoded_count: 0,
                    yielded: self.termination != TerminationReason::Eos,
                    skipped: false,
                    ignored_eos: false,
                    stop: self.termination == TerminationReason::Eos,
                }],
            })
        }
        fn generate_generated_mel(
            &self,
            tokens: &[u32],
            _prompt_mel: &[f32],
            prompt_frames: usize,
            _streaming: bool,
            _finalize: bool,
            flow_noise: &FixedFlowNoise,
        ) -> Result<GeneratedMel> {
            assert_eq!(flow_noise.frames(), FIXED_FLOW_NOISE_FRAMES);
            let frames = if self.bad_frames {
                tokens.len() * TOKEN_MEL_RATIO + 1
            } else {
                tokens.len() * TOKEN_MEL_RATIO
            };
            Ok(GeneratedMel {
                data: vec![0.0; 80 * frames],
                frames,
                prompt_frames_removed: prompt_frames,
                full_frames_before_prompt_slice: prompt_frames
                    + frames
                    + usize::from(self.bad_frames),
            })
        }
    }

    fn conditioning() -> BatchOneConditioning {
        BatchOneConditioning {
            llm_input_width: 2,
            sos_embedding: vec![0.0, 1.0],
            prompt_text_embeddings: vec![2.0, 3.0],
            target_text_embeddings: vec![4.0, 5.0, 6.0, 7.0],
            prompt_speech_embeddings: vec![8.0, 9.0],
            task_embedding: vec![10.0, 11.0],
            prompt_mel: vec![0.0; 80],
            prompt_mel_frames: 1,
            streaming: false,
            finalize: true,
        }
    }

    #[test]
    fn native_route_remains_blocked_without_manifest_binder() {
        let model = Model {
            termination: TerminationReason::Eos,
            bad_frames: false,
            seen: std::cell::RefCell::new(Vec::new()),
        };
        let decoder = Decoder;
        let route = BatchOneRoute::new(&model, &decoder).unwrap();
        let flow_data = vec![0.001; 80 * FIXED_FLOW_NOISE_FRAMES];
        let flow_noise = FixedFlowNoise::new(flow_data, FIXED_FLOW_NOISE_FRAMES).unwrap();
        let error = route
            .synthesize(&conditioning(), &flow_noise, &mut Random)
            .unwrap_err();
        assert!(matches!(error, VokraError::InvalidArgument(_)));
        assert!(model.seen.borrow().is_empty());
    }

    #[test]
    fn local_digest_is_not_flow_noise_authentication() {
        let data = vec![0.001; 80 * FIXED_FLOW_NOISE_FRAMES];
        let locally_computed_digest = sha256_f32_le(&data);
        assert_eq!(locally_computed_digest.len(), 64);
        let packet = FixedFlowNoise::new(data, FIXED_FLOW_NOISE_FRAMES).unwrap();
        assert!(!packet.is_authenticated());
        // There is deliberately no API accepting `locally_computed_digest`.
        // Only a future private, manifest-bound binder may set this bit.
    }

    #[test]
    fn fixed_flow_noise_prefix_preserves_channel_major_source_slice() {
        let mut packet = Vec::new();
        for channel in 0..80 {
            for frame in 0..4 {
                packet.push((channel * 10 + frame) as f32);
            }
        }
        let packet = FixedFlowNoise::new(packet, 4).unwrap();
        let prefix = packet.prefix(2).unwrap();
        assert_eq!(prefix.frames(), 2);
        assert_eq!(prefix.data()[0..2], [0.0, 1.0]);
        assert_eq!(prefix.data()[2..4], [10.0, 11.0]);
        assert!(packet.prefix(5).is_err());
    }

    #[test]
    fn wrapper_boundary_rejects_special_yields_and_keeps_eos_out_of_tokens() {
        let invalid = SpeechGeneration {
            yielded_tokens: vec![SPEECH_TOKEN_VOCAB_SIZE as u32 + 1],
            termination: TerminationReason::Eos,
            min_tokens: 0,
            max_tokens: 1,
            configured_vllm_stop_token_ids: VLLM_STOP_TOKEN_IDS.to_vec(),
            native_terminal_eos: NATIVE_TERMINAL_EOS,
            sampled_tokens: vec![SPEECH_TOKEN_VOCAB_SIZE as u32 + 1],
            llm_calls: vec![LlmCallEvidence {
                call_index: 0,
                input_rows: 1,
                output_rows: 1,
            }],
            sampling_calls: vec![SamplingCallEvidence {
                call_index: 0,
                generation_step: 0,
                attempt_index: 0,
                selected_token: SPEECH_TOKEN_VOCAB_SIZE as u32 + 1,
                ignore_eos: false,
                decoded_count: 0,
                yielded: false,
                skipped: true,
                ignored_eos: false,
                stop: false,
            }],
        };
        let error = invalid.validate().unwrap_err();
        assert!(matches!(error, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn wrapper_evidence_models_retry_and_outer_step_state() {
        let generation = SpeechGeneration {
            yielded_tokens: vec![7, 8],
            termination: TerminationReason::Eos,
            min_tokens: 1,
            max_tokens: 4,
            configured_vllm_stop_token_ids: VLLM_STOP_TOKEN_IDS.to_vec(),
            native_terminal_eos: NATIVE_TERMINAL_EOS,
            sampled_tokens: vec![
                SPEECH_TOKEN_VOCAB_SIZE as u32,
                SPEECH_TOKEN_VOCAB_SIZE as u32 + 2,
                7,
                8,
                SPEECH_TOKEN_VOCAB_SIZE as u32,
            ],
            llm_calls: vec![
                LlmCallEvidence {
                    call_index: 0,
                    input_rows: 4,
                    output_rows: 4,
                },
                LlmCallEvidence {
                    call_index: 1,
                    input_rows: 4,
                    output_rows: 4,
                },
                LlmCallEvidence {
                    call_index: 2,
                    input_rows: 1,
                    output_rows: 1,
                },
                LlmCallEvidence {
                    call_index: 3,
                    input_rows: 1,
                    output_rows: 1,
                },
            ],
            sampling_calls: vec![
                SamplingCallEvidence {
                    call_index: 0,
                    generation_step: 0,
                    attempt_index: 0,
                    selected_token: SPEECH_TOKEN_VOCAB_SIZE as u32,
                    ignore_eos: true,
                    decoded_count: 0,
                    yielded: false,
                    skipped: false,
                    ignored_eos: true,
                    stop: false,
                },
                SamplingCallEvidence {
                    selected_token: SPEECH_TOKEN_VOCAB_SIZE as u32 + 2,
                    call_index: 1,
                    generation_step: 0,
                    attempt_index: 1,
                    ignore_eos: true,
                    decoded_count: 0,
                    yielded: false,
                    skipped: true,
                    ignored_eos: false,
                    stop: false,
                },
                SamplingCallEvidence {
                    call_index: 2,
                    generation_step: 1,
                    attempt_index: 0,
                    selected_token: 7,
                    ignore_eos: false,
                    decoded_count: 0,
                    yielded: true,
                    skipped: false,
                    ignored_eos: false,
                    stop: false,
                },
                SamplingCallEvidence {
                    call_index: 3,
                    generation_step: 2,
                    attempt_index: 0,
                    selected_token: 8,
                    ignore_eos: false,
                    decoded_count: 1,
                    yielded: true,
                    skipped: false,
                    ignored_eos: false,
                    stop: false,
                },
                SamplingCallEvidence {
                    call_index: 4,
                    generation_step: 3,
                    attempt_index: 0,
                    selected_token: SPEECH_TOKEN_VOCAB_SIZE as u32,
                    ignore_eos: false,
                    decoded_count: 2,
                    yielded: false,
                    skipped: false,
                    ignored_eos: false,
                    stop: true,
                },
            ],
        };
        let validated = generation.validate().unwrap();
        assert_eq!(validated.yielded_tokens, [7, 8]);
        assert_eq!(validated.sampling_calls[0].selected_token, 6563);
    }

    #[test]
    fn wrapper_evidence_accepts_ignored_eos_then_valid_token_same_outer_step() {
        let eos = SPEECH_TOKEN_VOCAB_SIZE as u32;
        let generation = SpeechGeneration {
            yielded_tokens: vec![7],
            termination: TerminationReason::Eos,
            min_tokens: 1,
            max_tokens: 3,
            configured_vllm_stop_token_ids: VLLM_STOP_TOKEN_IDS.to_vec(),
            native_terminal_eos: NATIVE_TERMINAL_EOS,
            sampled_tokens: vec![eos, 7, eos],
            llm_calls: vec![
                LlmCallEvidence {
                    call_index: 0,
                    input_rows: 4,
                    output_rows: 4,
                },
                LlmCallEvidence {
                    call_index: 1,
                    input_rows: 1,
                    output_rows: 1,
                },
            ],
            sampling_calls: vec![
                SamplingCallEvidence {
                    call_index: 0,
                    generation_step: 0,
                    attempt_index: 0,
                    selected_token: eos,
                    ignore_eos: true,
                    decoded_count: 0,
                    yielded: false,
                    skipped: false,
                    ignored_eos: true,
                    stop: false,
                },
                SamplingCallEvidence {
                    call_index: 1,
                    generation_step: 0,
                    attempt_index: 1,
                    selected_token: 7,
                    ignore_eos: true,
                    decoded_count: 0,
                    yielded: true,
                    skipped: false,
                    ignored_eos: false,
                    stop: false,
                },
                SamplingCallEvidence {
                    call_index: 2,
                    generation_step: 1,
                    attempt_index: 0,
                    selected_token: eos,
                    ignore_eos: false,
                    decoded_count: 1,
                    yielded: false,
                    skipped: false,
                    ignored_eos: false,
                    stop: true,
                },
            ],
        };
        let validated = generation.validate().unwrap();
        assert_eq!(validated.yielded_tokens, [7]);
        assert_eq!(validated.sampling_calls[1].attempt_index, 1);
    }

    #[test]
    fn flow_must_return_generated_only_frames_without_double_slice() {
        let model = Model {
            termination: TerminationReason::Eos,
            bad_frames: true,
            seen: std::cell::RefCell::new(Vec::new()),
        };
        let decoder = Decoder;
        let route = BatchOneRoute::new(&model, &decoder).unwrap();
        let flow_data = vec![0.001; 80 * FIXED_FLOW_NOISE_FRAMES];
        let flow_noise = FixedFlowNoise::new(flow_data, FIXED_FLOW_NOISE_FRAMES).unwrap();
        let error = route
            .synthesize(&conditioning(), &flow_noise, &mut Random)
            .unwrap_err();
        assert!(matches!(error, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn ras_sampling_rejects_zero_top_p() {
        let model = Model {
            termination: TerminationReason::Eos,
            bad_frames: false,
            seen: std::cell::RefCell::new(Vec::new()),
        };
        let route = BatchOneRoute::new(&model, &Decoder)
            .unwrap()
            .with_sampling(SamplingConfig {
                top_p: 0.0,
                ..SamplingConfig::default()
            });
        let flow_data = vec![0.001; 80 * FIXED_FLOW_NOISE_FRAMES];
        let noise = FixedFlowNoise::new(flow_data, FIXED_FLOW_NOISE_FRAMES).unwrap();
        assert!(
            route
                .synthesize(&conditioning(), &noise, &mut Random)
                .is_err()
        );
        let route = BatchOneRoute::new(&model, &Decoder)
            .unwrap()
            .with_sampling(SamplingConfig {
                tau_r: 0.0,
                ..SamplingConfig::default()
            });
        assert!(
            route
                .synthesize(&conditioning(), &noise, &mut Random)
                .is_err()
        );
    }
}
