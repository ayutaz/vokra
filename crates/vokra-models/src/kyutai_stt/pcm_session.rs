//! Crate-private Kyutai STT PCM composite session.
//!
//! This is an authenticated component seam, not a public ASR loader.  The
//! constructor requires external whole-file digests for the decoder,
//! tokenizer, and converted Mimi GGUFs, and separately verifies the fixed raw
//! Mimi safetensors sidecar.  It does not infer provenance from filenames or
//! from the caller's paths.

use std::path::Path;

use vokra_core::gguf::{AsBytes, GgufFile, GgufMetadataValue};
use vokra_core::{BackendKind, Result, VokraError};

use crate::compute::Compute;
use crate::mimi::{MimiEncoder, MimiEncoderState, MimiNeuralConfig};
use crate::strict_checkpoint::sha256_bytes;

use super::{
    KYUTAI_STT_MIMI_BYTES, KYUTAI_STT_MIMI_SHA256, KyutaiSttAsr, KyutaiSttStreamingContract,
    KyutaiSttStreamingLm, KyutaiSttTextLogits, KyutaiSttTokenizer,
};

const KEY_MIMI_CHECKPOINT_BYTES: &str = "vokra.provenance.checkpoint_bytes";
const KEY_MIMI_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";

/// A whole-file digest supplied by an authenticated external transfer packet.
///
/// The type deliberately has no path or filename field.  A digest copied from
/// an untrusted caller is not a provenance decision; callers must obtain these
/// values from the owner-approved packet that accompanies the exact artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KyutaiSttPcmArtifactDigest {
    bytes: u64,
    sha256: [u8; 32],
}

impl KyutaiSttPcmArtifactDigest {
    /// Parses a lowercase hexadecimal SHA-256 and its independently recorded
    /// file length.  No source identity is inferred from this value.
    pub(crate) fn new(sha256: &str, bytes: u64) -> Result<Self> {
        if sha256.len() != 64 {
            return Err(VokraError::InvalidArgument(
                "kyutai STT PCM artifact SHA-256 must contain 64 hex characters".to_owned(),
            ));
        }
        let mut digest = [0u8; 32];
        for (index, pair) in sha256.as_bytes().chunks_exact(2).enumerate() {
            let high = hex_nibble(pair[0]).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "kyutai STT PCM artifact SHA-256 must be lowercase hexadecimal".to_owned(),
                )
            })?;
            let low = hex_nibble(pair[1]).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "kyutai STT PCM artifact SHA-256 must be lowercase hexadecimal".to_owned(),
                )
            })?;
            digest[index] = (high << 4) | low;
        }
        Ok(Self {
            bytes,
            sha256: digest,
        })
    }
}

/// External whole-file identities required before component construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KyutaiSttPcmArtifactDigests {
    pub(crate) decoder: KyutaiSttPcmArtifactDigest,
    pub(crate) tokenizer: KyutaiSttPcmArtifactDigest,
    pub(crate) mimi_gguf: KyutaiSttPcmArtifactDigest,
}

/// Authenticated, immutable native components for a PCM session.
#[derive(Debug)]
pub(crate) struct KyutaiSttPcmEngine {
    decoder: KyutaiSttAsr,
    mimi: MimiEncoder,
    tokenizer: KyutaiSttTokenizer,
    contract: KyutaiSttStreamingContract,
    backend: BackendKind,
}

impl KyutaiSttPcmEngine {
    /// Opens and authenticates all executable artifacts through read-only
    /// mappings, then binds the existing Mimi, decoder, and tokenizer seams.
    ///
    /// The expected GGUF digests must come from an authenticated external
    /// packet.  The raw Mimi sidecar is stronger than a caller-supplied
    /// expectation: its fixed filename-independent bytes and SHA-256 are
    /// checked against the pinned Kyutai identity, and the converted GGUF must
    /// carry the same measured checkpoint metadata before it is bound.
    pub(crate) fn from_paths(
        decoder_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        mimi_gguf_path: impl AsRef<Path>,
        raw_mimi_path: impl AsRef<Path>,
        digests: &KyutaiSttPcmArtifactDigests,
        backend: BackendKind,
    ) -> Result<Self> {
        let decoder = open_authenticated_gguf(decoder_path, "decoder", &digests.decoder)?;
        let tokenizer = open_authenticated_gguf(tokenizer_path, "tokenizer", &digests.tokenizer)?;
        let mimi_gguf = open_authenticated_gguf(mimi_gguf_path, "Mimi GGUF", &digests.mimi_gguf)?;
        authenticate_raw_mimi(raw_mimi_path)?;

        let asr = KyutaiSttAsr::from_component_gguf(&decoder)?;
        let contract = KyutaiSttStreamingContract::from_config(asr.config())?;
        let tokenizer = KyutaiSttTokenizer::from_gguf(&tokenizer)?;
        let mimi_config = MimiNeuralConfig::from_gguf(&mimi_gguf).map_err(|error| {
            VokraError::ModelLoad(format!(
                "kyutai STT PCM composite: Mimi config is not authenticated: {error}"
            ))
        })?;
        contract.validate_mimi_config(&mimi_config)?;
        validate_mimi_conversion_identity(&mimi_gguf)?;
        let mimi = MimiEncoder::from_gguf(&mimi_gguf, &mimi_config)?.with_backend(backend);

        // Preflight the full learned capability sets before constructing a
        // session.  Neither component can silently move unsupported work to
        // CPU after this check.
        Compute::for_mimi_backend(backend, crate::mimi::encoder::MIMI_HOT_OPS)?;
        let _ = asr.streaming_lm(backend)?;

        Ok(Self {
            decoder: asr,
            mimi,
            tokenizer,
            contract,
            backend,
        })
    }

    /// Starts a bounded PCM session.  The cap includes the exact left/right
    /// padding frames, not only caller-provided audio frames.
    pub(crate) fn session(&self, max_frames: usize) -> Result<KyutaiSttPcmSession<'_>> {
        if max_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kyutai STT PCM session: max_frames must be > 0".to_owned(),
            ));
        }
        let minimum = self.contract.padded_frame_count(0)?;
        if max_frames < minimum {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai STT PCM session: max_frames={max_frames} cannot hold the required {minimum} padding frames"
            )));
        }
        let lm = self.decoder.streaming_lm(self.backend)?;
        let mimi_state = self.mimi.state(1)?;
        Ok(KyutaiSttPcmSession {
            engine: self,
            mimi_state,
            lm,
            control: PcmSessionControl::new(&self.contract, max_frames)?,
        })
    }

    pub(crate) fn contract(&self) -> KyutaiSttStreamingContract {
        self.contract
    }

    pub(crate) fn tokenizer(&self) -> &KyutaiSttTokenizer {
        &self.tokenizer
    }
}

/// Stateful PCM-to-text session.  The session is crate-private until real
/// weight CPU parity, backend coverage, and provenance/legal review complete.
#[derive(Debug)]
pub(crate) struct KyutaiSttPcmSession<'a> {
    engine: &'a KyutaiSttPcmEngine,
    mimi_state: MimiEncoderState,
    lm: KyutaiSttStreamingLm<'a>,
    control: PcmSessionControl,
}

impl KyutaiSttPcmSession<'_> {
    /// Accepts PCM samples, encoding complete 1,920-sample frames while
    /// retaining at most one partial frame as carry.  Samples must be finite.
    pub(crate) fn push_pcm(&mut self, samples: &[f32]) -> Result<()> {
        self.push_pcm_with_callbacks(samples)
    }

    /// Appends exactly the prescribed right padding once, drains complete
    /// frames, and drops any final partial frame.  EOS is retained as a raw
    /// sampled token; it never stops this schedule early.
    pub(crate) fn finish(&mut self) -> Result<()> {
        self.finish_with_callbacks()
    }

    /// Clears PCM carry, Mimi causal state, LM KV state, schedule, and output
    /// bookkeeping together.  The next push starts the exact left-prefix path.
    pub(crate) fn reset(&mut self) {
        let control = &mut self.control;
        let mimi_state = &mut self.mimi_state;
        let lm = &mut self.lm;
        control.reset(&self.engine.contract, || mimi_state.reset(), || lm.reset());
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.control.poisoned
    }

    pub(crate) fn raw_text_tokens(&self) -> &[u32] {
        &self.control.raw_text_tokens
    }

    pub(crate) fn emitted_text_tokens(&self) -> &[u32] {
        &self.control.emitted_text_tokens
    }

    pub(crate) fn transcript(&self) -> Result<String> {
        self.ensure_live("transcript")?;
        self.engine
            .tokenizer
            .decode_text_tokens(&self.control.emitted_text_tokens)
    }

    fn ensure_live(&self, operation: &str) -> Result<()> {
        self.control.ensure_live(operation)
    }

    fn push_pcm_with_callbacks(&mut self, samples: &[f32]) -> Result<()> {
        let contract = self.engine.contract;
        let mimi = &self.engine.mimi;
        let mimi_state = &mut self.mimi_state;
        let lm = &mut self.lm;
        self.control.push(
            contract,
            samples,
            |frame, codes| mimi.encode_into(mimi_state, frame, codes),
            |previous, codes| {
                let step = lm.step_frame(previous, codes)?;
                greedy_token(step.logits(), contract.text_card())
            },
        )
    }

    fn finish_with_callbacks(&mut self) -> Result<()> {
        let contract = self.engine.contract;
        let mimi = &self.engine.mimi;
        let mimi_state = &mut self.mimi_state;
        let lm = &mut self.lm;
        self.control.finish(
            contract,
            |frame, codes| mimi.encode_into(mimi_state, frame, codes),
            |previous, codes| {
                let step = lm.step_frame(previous, codes)?;
                greedy_token(step.logits(), contract.text_card())
            },
        )
    }
}

/// The exact PCM carry/padding state machine is kept separate from the model
/// callbacks so its transactional behavior can be tested without weights.
#[derive(Debug)]
struct PcmFrameBuffer {
    pcm: Vec<f32>,
    frame: Vec<f32>,
    codes: Vec<u32>,
    max_frames: usize,
    encoded_frames: usize,
    finished: bool,
}

impl PcmFrameBuffer {
    fn new(contract: &KyutaiSttStreamingContract, max_frames: usize) -> Result<Self> {
        let minimum = contract.padded_frame_count(0)?;
        if max_frames < minimum {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai STT PCM frame buffer: max_frames={max_frames} cannot hold {minimum} padding frames"
            )));
        }
        Ok(Self {
            pcm: vec![0.0; contract.silence_prefix_samples()],
            frame: vec![0.0; contract.frame_hop_samples()],
            codes: vec![0; contract.n_q()],
            max_frames,
            encoded_frames: 0,
            finished: false,
        })
    }

    fn push<Encode, Decode>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        samples: &[f32],
        mut encode: Encode,
        mut decode: Decode,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Decode: FnMut(&[u32]) -> Result<()>,
    {
        if self.finished {
            return Err(VokraError::InvalidArgument(
                "kyutai STT PCM frame buffer: push after finish".to_owned(),
            ));
        }
        if !samples.iter().all(|sample| sample.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "kyutai STT PCM frame buffer: input PCM contains a non-finite sample".to_owned(),
            ));
        }
        if samples.is_empty() {
            return Ok(());
        }
        self.check_capacity(contract, samples.len())?;
        let hop = contract.frame_hop_samples();
        self.drain(contract, &mut encode, &mut decode)?;
        let mut offset = 0;
        while offset < samples.len() {
            let take = (hop - self.pcm.len()).min(samples.len() - offset);
            self.pcm.extend_from_slice(&samples[offset..offset + take]);
            offset += take;
            if self.pcm.len() == hop {
                self.drain(contract, &mut encode, &mut decode)?;
            }
        }
        Ok(())
    }

    fn finish<Encode, Decode>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        mut encode: Encode,
        mut decode: Decode,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Decode: FnMut(&[u32]) -> Result<()>,
    {
        if self.finished {
            return Err(VokraError::InvalidArgument(
                "kyutai STT PCM frame buffer: finish may be called only once".to_owned(),
            ));
        }
        let right = contract.right_padding_samples();
        self.check_capacity(contract, right)?;
        self.finished = true;
        let hop = contract.frame_hop_samples();
        self.drain(contract, &mut encode, &mut decode)?;
        let mut remaining = right;
        while remaining > 0 {
            let take = (hop - self.pcm.len()).min(remaining);
            self.pcm.extend(std::iter::repeat_n(0.0, take));
            remaining -= take;
            if self.pcm.len() == hop {
                self.drain(contract, &mut encode, &mut decode)?;
            }
        }
        self.pcm.truncate((self.pcm.len() / hop) * hop);
        self.drain(contract, &mut encode, &mut decode)
    }

    fn reset(&mut self, contract: &KyutaiSttStreamingContract) {
        self.pcm.clear();
        self.pcm.resize(contract.silence_prefix_samples(), 0.0);
        self.frame.fill(0.0);
        self.codes.fill(0);
        self.encoded_frames = 0;
        self.finished = false;
    }

    fn check_capacity(
        &self,
        contract: KyutaiSttStreamingContract,
        additional_samples: usize,
    ) -> Result<()> {
        checked_frame_capacity(
            self.pcm.len(),
            self.encoded_frames,
            additional_samples,
            contract.frame_hop_samples(),
            self.max_frames,
        )
    }

    fn drain<Encode, Decode>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        encode: &mut Encode,
        decode: &mut Decode,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Decode: FnMut(&[u32]) -> Result<()>,
    {
        let hop = contract.frame_hop_samples();
        while self.pcm.len() >= hop {
            self.frame.copy_from_slice(&self.pcm[..hop]);
            let remaining = self.pcm.len() - hop;
            self.pcm.copy_within(hop.., 0);
            self.pcm.truncate(remaining);
            encode(&self.frame, &mut self.codes)?;
            contract.validate_mimi_codes(&self.codes)?;
            decode(&self.codes)?;
            self.encoded_frames = self.encoded_frames.checked_add(1).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "kyutai STT PCM frame buffer: encoded frame count overflow".to_owned(),
                )
            })?;
        }
        Ok(())
    }
}

/// Shared lifecycle state for the real session and model-free control tests.
#[derive(Debug)]
struct PcmSessionControl {
    buffer: PcmFrameBuffer,
    schedule: TextSchedule,
    raw_text_tokens: Vec<u32>,
    emitted_text_tokens: Vec<u32>,
    finished: bool,
    poisoned: bool,
}

impl PcmSessionControl {
    fn new(contract: &KyutaiSttStreamingContract, max_frames: usize) -> Result<Self> {
        Ok(Self {
            buffer: PcmFrameBuffer::new(contract, max_frames)?,
            schedule: TextSchedule::default(),
            raw_text_tokens: Vec::new(),
            emitted_text_tokens: Vec::new(),
            finished: false,
            poisoned: false,
        })
    }

    fn ensure_live(&self, operation: &str) -> Result<()> {
        if self.poisoned {
            return Err(VokraError::InvalidArgument(format!(
                "kyutai STT PCM session is poisoned; call reset before {operation}"
            )));
        }
        if self.finished && operation == "push_pcm" {
            return Err(VokraError::InvalidArgument(
                "kyutai STT PCM session: PCM cannot be pushed after finish".to_owned(),
            ));
        }
        Ok(())
    }

    fn reset<ResetMimi, ResetLm>(
        &mut self,
        contract: &KyutaiSttStreamingContract,
        reset_mimi: ResetMimi,
        reset_lm: ResetLm,
    ) where
        ResetMimi: FnOnce(),
        ResetLm: FnOnce(),
    {
        reset_mimi();
        reset_lm();
        self.buffer.reset(contract);
        self.schedule = TextSchedule::default();
        self.raw_text_tokens.clear();
        self.emitted_text_tokens.clear();
        self.finished = false;
        self.poisoned = false;
    }

    fn begin_finish(&mut self) -> Result<()> {
        self.ensure_live("finish")?;
        if self.finished {
            return self.poison(VokraError::InvalidArgument(
                "kyutai STT PCM session: finish may be called only once".to_owned(),
            ));
        }
        self.finished = true;
        Ok(())
    }

    fn push<Encode, Step>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        samples: &[f32],
        encode: Encode,
        step: Step,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Step: FnMut(Option<u32>, &[u32]) -> Result<u32>,
    {
        self.ensure_live("push_pcm")?;
        let result = self.push_inner(contract, samples, encode, step);
        match result {
            Ok(()) => Ok(()),
            Err(error) => self.poison(error),
        }
    }

    fn push_inner<Encode, Step>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        samples: &[f32],
        mut encode: Encode,
        mut step: Step,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Step: FnMut(Option<u32>, &[u32]) -> Result<u32>,
    {
        let buffer = &mut self.buffer;
        let schedule = &mut self.schedule;
        let raw = &mut self.raw_text_tokens;
        let emitted = &mut self.emitted_text_tokens;
        buffer.push(contract, samples, &mut encode, |codes| {
            drive_sampled_code_row(contract, codes, schedule, raw, emitted, &mut step)
        })
    }

    fn finish<Encode, Step>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        encode: Encode,
        step: Step,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Step: FnMut(Option<u32>, &[u32]) -> Result<u32>,
    {
        self.begin_finish()?;
        let result = self.finish_inner(contract, encode, step);
        match result {
            Ok(()) => Ok(()),
            Err(error) => self.poison(error),
        }
    }

    fn finish_inner<Encode, Step>(
        &mut self,
        contract: KyutaiSttStreamingContract,
        mut encode: Encode,
        mut step: Step,
    ) -> Result<()>
    where
        Encode: FnMut(&[f32], &mut [u32]) -> Result<()>,
        Step: FnMut(Option<u32>, &[u32]) -> Result<u32>,
    {
        let buffer = &mut self.buffer;
        let schedule = &mut self.schedule;
        let raw = &mut self.raw_text_tokens;
        let emitted = &mut self.emitted_text_tokens;
        buffer.finish(contract, &mut encode, |codes| {
            drive_sampled_code_row(contract, codes, schedule, raw, emitted, &mut step)
        })
    }

    fn poison<T>(&mut self, error: VokraError) -> Result<T> {
        self.poisoned = true;
        self.raw_text_tokens.clear();
        self.emitted_text_tokens.clear();
        Err(error)
    }
}

fn checked_frame_capacity(
    pcm_len: usize,
    encoded_frames: usize,
    additional_samples: usize,
    hop: usize,
    max_frames: usize,
) -> Result<()> {
    let total = pcm_len.checked_add(additional_samples).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai STT PCM sample count overflow".to_owned())
    })?;
    let possible = total / hop;
    let projected = encoded_frames.checked_add(possible).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai STT PCM frame count overflow".to_owned())
    })?;
    if projected > max_frames {
        return Err(VokraError::InvalidArgument(format!(
            "kyutai STT PCM frame buffer: cap {max_frames} exceeded by projected {projected} frames"
        )));
    }
    Ok(())
}

fn drive_sampled_code_row<Step>(
    contract: KyutaiSttStreamingContract,
    codes: &[u32],
    schedule: &mut TextSchedule,
    raw: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    mut step: Step,
) -> Result<()>
where
    Step: FnMut(Option<u32>, &[u32]) -> Result<u32>,
{
    let old_schedule = *schedule;
    let old_raw = raw.len();
    let old_emitted = emitted.len();
    let result = (|| {
        if schedule.frames == 0 {
            let first_token = step(None, codes)?;
            record_sampled_token(contract, raw, emitted, first_token, false)?;
            schedule.accept(first_token);
            let second_token = step(schedule.previous, codes)?;
            record_sampled_token(contract, raw, emitted, second_token, true)?;
            schedule.accept(second_token);
        } else {
            let token = step(schedule.previous, codes)?;
            record_sampled_token(contract, raw, emitted, token, true)?;
            schedule.accept(token);
        }
        schedule.finish_frame()
    })();
    if result.is_err() {
        *schedule = old_schedule;
        raw.truncate(old_raw);
        emitted.truncate(old_emitted);
    }
    result
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct TextSchedule {
    frames: usize,
    previous: Option<u32>,
}

impl TextSchedule {
    fn accept(&mut self, token: u32) {
        self.previous = Some(token);
    }

    fn finish_frame(&mut self) -> Result<()> {
        self.frames = self.frames.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("kyutai STT text frame count overflow".to_owned())
        })?;
        Ok(())
    }
}

fn record_sampled_token(
    contract: KyutaiSttStreamingContract,
    raw: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    token: u32,
    expose: bool,
) -> Result<()> {
    if token as usize >= contract.text_card() {
        return Err(VokraError::InvalidArgument(format!(
            "kyutai STT PCM session: sampled token {token} outside [0, {})",
            contract.text_card()
        )));
    }
    raw.push(token);
    if expose && contract.emits_text_token(token) {
        emitted.push(token);
    }
    Ok(())
}

fn greedy_token(logits: &KyutaiSttTextLogits, text_card: usize) -> Result<u32> {
    if logits.frames() != 1 || logits.vocab() != text_card || logits.as_slice().is_empty() {
        return Err(VokraError::InvalidArgument(
            "kyutai STT PCM session: expected one finite text-logit row".to_owned(),
        ));
    }
    let values = logits.as_slice();
    if !values.iter().all(|value| value.is_finite()) {
        return Err(VokraError::ModelLoad(
            "kyutai STT PCM session: sampled logits contain a non-finite value".to_owned(),
        ));
    }
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|_| {
        VokraError::InvalidArgument("kyutai STT PCM session: token index overflows u32".to_owned())
    })
}

fn open_authenticated_gguf(
    path: impl AsRef<Path>,
    label: &str,
    expected: &KyutaiSttPcmArtifactDigest,
) -> Result<GgufFile> {
    let mapping = vokra_mmap::Mmap::open(path.as_ref()).map_err(VokraError::Io)?;
    if mapping.bytes().len() as u64 != expected.bytes {
        return Err(VokraError::ModelLoad(format!(
            "kyutai STT PCM composite: {label} byte count mismatch: expected {}, got {}",
            expected.bytes,
            mapping.bytes().len()
        )));
    }
    let actual = sha256_bytes(mapping.bytes());
    if actual != expected.sha256 {
        return Err(VokraError::ModelLoad(format!(
            "kyutai STT PCM composite: {label} whole-file SHA-256 {} does not match authenticated packet {}",
            hex_digest(&actual),
            hex_digest(&expected.sha256)
        )));
    }
    GgufFile::from_external(Box::new(mapping)).map_err(|error| {
        VokraError::ModelLoad(format!(
            "kyutai STT PCM composite: {label} GGUF parse failed: {error}"
        ))
    })
}

fn authenticate_raw_mimi(path: impl AsRef<Path>) -> Result<()> {
    let mapping = vokra_mmap::Mmap::open(path.as_ref()).map_err(VokraError::Io)?;
    if mapping.bytes().len() != KYUTAI_STT_MIMI_BYTES {
        return Err(VokraError::ModelLoad(format!(
            "kyutai STT PCM composite: raw Mimi sidecar has {} bytes, expected {}",
            mapping.bytes().len(),
            KYUTAI_STT_MIMI_BYTES
        )));
    }
    let actual = hex_digest(&sha256_bytes(mapping.bytes()));
    if actual != KYUTAI_STT_MIMI_SHA256 {
        return Err(VokraError::ModelLoad(format!(
            "kyutai STT PCM composite: raw Mimi sidecar SHA-256 {actual} does not match the fixed Kyutai identity"
        )));
    }
    Ok(())
}

fn validate_mimi_conversion_identity(file: &GgufFile) -> Result<()> {
    match file.get(KEY_MIMI_CHECKPOINT_SHA256) {
        Some(GgufMetadataValue::String(value)) if value == KYUTAI_STT_MIMI_SHA256 => {}
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "kyutai STT PCM composite: Mimi converted GGUF checkpoint SHA metadata is {other:?}, expected fixed raw identity {KYUTAI_STT_MIMI_SHA256}"
            )));
        }
        None => {
            return Err(VokraError::ModelLoad(
                "kyutai STT PCM composite: Mimi converted GGUF lacks measured checkpoint SHA metadata"
                    .to_owned(),
            ));
        }
    }
    match file.get(KEY_MIMI_CHECKPOINT_BYTES) {
        Some(GgufMetadataValue::U64(value)) if *value == KYUTAI_STT_MIMI_BYTES as u64 => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "kyutai STT PCM composite: Mimi converted GGUF checkpoint byte metadata is {other:?}, expected {KYUTAI_STT_MIMI_BYTES}"
        ))),
        None => Err(VokraError::ModelLoad(
            "kyutai STT PCM composite: Mimi converted GGUF lacks measured checkpoint byte metadata"
                .to_owned(),
        )),
    }
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in bytes {
        text.push(char::from(HEX[(byte >> 4) as usize]));
        text.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    text
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn contract() -> KyutaiSttStreamingContract {
        KyutaiSttStreamingContract::from_config(&super::super::KyutaiSttConfig::stt_2_6b_en())
            .unwrap()
    }

    #[test]
    fn artifact_digest_requires_lowercase_sha256() {
        assert!(KyutaiSttPcmArtifactDigest::new(&"a".repeat(64), 1).is_ok());
        assert!(KyutaiSttPcmArtifactDigest::new(&"A".repeat(64), 1).is_err());
        assert!(KyutaiSttPcmArtifactDigest::new("short", 1).is_err());
    }

    #[test]
    fn greedy_sampling_uses_first_index_on_ties() {
        let logits = KyutaiSttTextLogits::new(1, 4, vec![2.0, 3.0, 3.0, 1.0]).unwrap();
        assert_eq!(greedy_token(&logits, 4).unwrap(), 1);
    }

    #[test]
    fn greedy_sampling_rejects_nonfinite_logits() {
        assert!(KyutaiSttTextLogits::new(1, 2, vec![f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn shared_schedule_matches_first_double_step_and_current_rows() {
        let contract = contract();
        let mut schedule = TextSchedule::default();
        let mut raw = Vec::new();
        let mut emitted = Vec::new();
        let mut calls = Vec::new();
        let mut next = [11, 2, 3].into_iter();
        for row in 0..2 {
            let codes = [row as u32; 32];
            drive_sampled_code_row(
                contract,
                &codes,
                &mut schedule,
                &mut raw,
                &mut emitted,
                |previous, codes| {
                    calls.push((previous, codes[0]));
                    Ok(next.next().unwrap())
                },
            )
            .unwrap();
        }
        assert_eq!(calls, [(None, 0), (Some(11), 0), (Some(2), 1)]);
        assert_eq!(raw, [11, 2, 3]);
        assert_eq!(emitted, [2]);
        assert_eq!(schedule.frames, 2);
        assert_eq!(schedule.previous, Some(3));

        schedule = TextSchedule::default();
        raw.clear();
        emitted.clear();
        calls.clear();
        let codes = [7; 32];
        drive_sampled_code_row(
            contract,
            &codes,
            &mut schedule,
            &mut raw,
            &mut emitted,
            |previous, codes| {
                calls.push((previous, codes[0]));
                Ok(19)
            },
        )
        .unwrap();
        assert_eq!(calls, [(None, 7), (Some(19), 7)]);
    }

    #[test]
    fn shared_schedule_rolls_back_on_encoder_or_lm_error() {
        let contract = contract();
        let codes = [0; 32];
        let mut schedule = TextSchedule::default();
        let mut raw = Vec::new();
        let mut emitted = Vec::new();
        let mut calls = 0;
        let result = drive_sampled_code_row(
            contract,
            &codes,
            &mut schedule,
            &mut raw,
            &mut emitted,
            |_previous, _codes| {
                calls += 1;
                if calls == 2 {
                    Err(VokraError::ModelLoad("injected LM failure".to_owned()))
                } else {
                    Ok(11)
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(calls, 2);
        assert_eq!(schedule, TextSchedule::default());
        assert!(raw.is_empty() && emitted.is_empty());

        let mut buffer =
            PcmFrameBuffer::new(&contract, contract.padded_frame_count(0).unwrap()).unwrap();
        let result = buffer.push(
            contract,
            &[0.0; 1],
            |_frame, _codes| Err(VokraError::ModelLoad("injected encoder failure".to_owned())),
            |_codes| Ok(()),
        );
        assert!(result.is_err());
        assert!(!buffer.finished);
    }

    #[test]
    fn text_schedule_rejects_frame_counter_overflow() {
        let mut schedule = TextSchedule {
            frames: usize::MAX,
            previous: None,
        };
        assert!(schedule.finish_frame().is_err());
        assert_eq!(schedule.frames, usize::MAX);
    }

    #[test]
    fn lifecycle_control_poison_finish_retry_and_reset_are_transactional() {
        let contract = contract();
        let mut control =
            PcmSessionControl::new(&contract, contract.padded_frame_count(0).unwrap()).unwrap();
        let encoder_error = control.push(
            contract,
            &[0.0; 1],
            |_frame, _codes| Err(VokraError::ModelLoad("injected encoder failure".to_owned())),
            |_previous, _codes| Ok(11),
        );
        assert!(encoder_error.is_err());
        assert!(control.poisoned);
        assert!(control.raw_text_tokens.is_empty() && control.emitted_text_tokens.is_empty());
        assert!(control.ensure_live("push_pcm").is_err());

        let mut mimi_resets = 0;
        let mut lm_resets = 0;
        control.reset(&contract, || mimi_resets += 1, || lm_resets += 1);
        assert_eq!((mimi_resets, lm_resets), (1, 1));
        assert!(!control.poisoned && !control.finished);
        assert_eq!(control.schedule, TextSchedule::default());
        assert_eq!(control.buffer.pcm.len(), contract.silence_prefix_samples());
        assert!(control.ensure_live("push_pcm").is_ok());

        let lm_error = control.push(
            contract,
            &[0.0; 1],
            |_frame, codes| {
                codes.fill(0);
                Ok(())
            },
            |_previous, _codes| Err(VokraError::ModelLoad("injected LM failure".to_owned())),
        );
        assert!(lm_error.is_err());
        assert!(control.poisoned);
        assert!(control.raw_text_tokens.is_empty() && control.emitted_text_tokens.is_empty());

        control.reset(&contract, || mimi_resets += 1, || lm_resets += 1);
        let mut calls = 0;
        control
            .push(
                contract,
                &[0.0; 1],
                |_frame, codes| {
                    codes.fill(0);
                    Ok(())
                },
                |previous, _codes| {
                    calls += 1;
                    Ok(if previous.is_none() { 11 } else { 2 })
                },
            )
            .unwrap();
        assert!(calls > 2);
        assert!(!control.raw_text_tokens.is_empty());
        control
            .finish(
                contract,
                |_frame, codes| {
                    codes.fill(0);
                    Ok(())
                },
                |_previous, _codes| Ok(2),
            )
            .unwrap();
        assert!(
            control
                .finish(contract, |_frame, _codes| Ok(()), |_previous, _codes| Ok(2))
                .is_err()
        );
        assert!(control.poisoned);
        assert!(control.raw_text_tokens.is_empty() && control.emitted_text_tokens.is_empty());
        assert!(control.ensure_live("push_pcm").is_err());
        control.reset(&contract, || mimi_resets += 1, || lm_resets += 1);
        assert_eq!((mimi_resets, lm_resets), (3, 3));
        assert!(control.ensure_live("push_pcm").is_ok());
        assert_eq!(control.schedule, TextSchedule::default());
        assert_eq!(control.buffer.pcm.len(), contract.silence_prefix_samples());
    }

    #[test]
    fn control_push_nonfinite_cap_and_overflow_poison_automatically() {
        let contract = contract();
        let cap = contract.padded_frame_count(0).unwrap();
        let mut control = PcmSessionControl::new(&contract, cap).unwrap();
        control.raw_text_tokens.push(2);
        control.emitted_text_tokens.push(2);
        let nonfinite = control.push(
            contract,
            &[f32::NAN],
            |_frame, _codes| Ok(()),
            |_previous, _codes| Ok(2),
        );
        assert!(nonfinite.is_err() && control.poisoned);
        assert!(control.raw_text_tokens.is_empty() && control.emitted_text_tokens.is_empty());

        control.reset(&contract, || {}, || {});
        let cap_error = control.push(
            contract,
            &vec![0.0; contract.right_padding_samples() + contract.frame_hop_samples()],
            |_frame, codes| {
                codes.fill(0);
                Ok(())
            },
            |_previous, _codes| Ok(2),
        );
        assert!(cap_error.is_err() && control.poisoned);

        control.reset(&contract, || {}, || {});
        control.buffer.encoded_frames = usize::MAX;
        control.buffer.max_frames = usize::MAX;
        let overflow = control.push(
            contract,
            &[0.0],
            |_frame, codes| {
                codes.fill(0);
                Ok(())
            },
            |_previous, _codes| Ok(2),
        );
        assert!(overflow.is_err() && control.poisoned);
        assert!(control.raw_text_tokens.is_empty() && control.emitted_text_tokens.is_empty());
    }

    #[test]
    fn control_finish_without_push_drains_prefix_and_right_padding_once() {
        let contract = contract();
        let mut control =
            PcmSessionControl::new(&contract, contract.padded_frame_count(0).unwrap()).unwrap();
        let mut rows = 0;
        control
            .push(
                contract,
                &[],
                |_frame, _codes| panic!("empty PCM must not encode"),
                |_previous, _codes| panic!("empty PCM must not sample"),
            )
            .unwrap();
        assert_eq!(control.buffer.encoded_frames, 0);
        control
            .finish(
                contract,
                |_frame, codes| {
                    codes.fill(0);
                    Ok(())
                },
                |previous, _codes| {
                    rows += 1;
                    Ok(if previous.is_none() { 11 } else { 2 })
                },
            )
            .unwrap();
        assert_eq!(rows, contract.padded_frame_count(0).unwrap() + 1);
        assert_eq!(control.raw_text_tokens.len(), rows);
        assert!(
            control
                .push(
                    contract,
                    &[0.0],
                    |_frame, _codes| panic!("push after finish must not encode"),
                    |_previous, _codes| panic!("push after finish must not sample"),
                )
                .is_err()
        );
        assert!(
            control
                .finish(contract, |_frame, _codes| Ok(()), |_previous, _codes| Ok(2))
                .is_err()
        );
        assert!(control.poisoned);
    }

    #[test]
    fn pcm_buffer_uses_production_chunking_padding_and_reset() {
        let contract = contract();
        let input = vec![0.25; 1_921];
        let mut buffer =
            PcmFrameBuffer::new(&contract, contract.padded_frame_count(1_921).unwrap()).unwrap();
        let mut decoded = 0;
        for chunk in [&input[..1], &[][..], &input[1..961], &input[961..]] {
            buffer
                .push(
                    contract,
                    chunk,
                    |_frame, codes| {
                        codes.fill(0);
                        Ok(())
                    },
                    |_codes| {
                        decoded += 1;
                        Ok(())
                    },
                )
                .unwrap();
        }
        assert_eq!(decoded, 13);
        assert_eq!(buffer.encoded_frames, 13);
        buffer
            .finish(
                contract,
                |_frame, codes| {
                    codes.fill(0);
                    Ok(())
                },
                |_codes| {
                    decoded += 1;
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(decoded, contract.padded_frame_count(1_921).unwrap());
        assert!(
            buffer
                .finish(contract, |_frame, _codes| Ok(()), |_codes| Ok(()))
                .is_err()
        );
        buffer.reset(&contract);
        assert_eq!(buffer.encoded_frames, 0);
        assert_eq!(buffer.pcm.len(), contract.silence_prefix_samples());
        assert!(!buffer.finished);
    }

    #[test]
    fn pcm_buffer_rejects_cap_and_overflow_without_partial_append() {
        let contract = contract();
        let mut buffer =
            PcmFrameBuffer::new(&contract, contract.padded_frame_count(0).unwrap()).unwrap();
        let before = buffer.pcm.len();
        assert!(
            buffer
                .push(
                    contract,
                    &vec![0.0; contract.right_padding_samples() + 1_920],
                    |_frame, codes| {
                        codes.fill(0);
                        Ok(())
                    },
                    |_codes| Ok(()),
                )
                .is_err()
        );
        assert_eq!(buffer.pcm.len(), before);
        assert!(
            checked_frame_capacity(usize::MAX, 0, 1, contract.frame_hop_samples(), usize::MAX)
                .is_err()
        );
        assert!(
            checked_frame_capacity(
                0,
                usize::MAX,
                usize::MAX,
                contract.frame_hop_samples(),
                usize::MAX,
            )
            .is_err()
        );
    }

    #[test]
    fn pcm_buffer_partitioning_preserves_exact_frames_and_bounded_carry() {
        let contract = contract();
        let input: Vec<f32> = (0..contract.frame_hop_samples() * 2 + 17)
            .map(|index| index as f32 * 0.001)
            .collect();
        let cap = contract.padded_frame_count(input.len()).unwrap();
        let mut partitioned = PcmFrameBuffer::new(&contract, cap).unwrap();
        let mut partitioned_rows = Vec::new();
        let mut partitioned_codes = Vec::new();
        for chunk in [&input[..1], &input[1..193], &input[193..]] {
            partitioned
                .push(
                    contract,
                    chunk,
                    |frame, codes| {
                        partitioned_rows.push(frame.to_vec());
                        codes.fill(0);
                        partitioned_codes.push(codes.to_vec());
                        Ok(())
                    },
                    |_codes| Ok(()),
                )
                .unwrap();
        }
        assert!(partitioned.pcm.len() < contract.frame_hop_samples());

        let mut one_chunk = PcmFrameBuffer::new(&contract, cap).unwrap();
        let mut one_rows = Vec::new();
        let mut one_codes = Vec::new();
        one_chunk
            .push(
                contract,
                &input,
                |frame, codes| {
                    one_rows.push(frame.to_vec());
                    codes.fill(0);
                    one_codes.push(codes.to_vec());
                    Ok(())
                },
                |_codes| Ok(()),
            )
            .unwrap();
        partitioned
            .finish(
                contract,
                |frame, codes| {
                    partitioned_rows.push(frame.to_vec());
                    codes.fill(0);
                    partitioned_codes.push(codes.to_vec());
                    Ok(())
                },
                |_codes| Ok(()),
            )
            .unwrap();
        one_chunk
            .finish(
                contract,
                |frame, codes| {
                    one_rows.push(frame.to_vec());
                    codes.fill(0);
                    one_codes.push(codes.to_vec());
                    Ok(())
                },
                |_codes| Ok(()),
            )
            .unwrap();
        assert_eq!(partitioned_rows, one_rows);
        assert_eq!(partitioned_codes, one_codes);
        assert_eq!(partitioned.pcm, one_chunk.pcm);
    }

    #[test]
    fn conversion_metadata_requires_measured_sha_and_bytes() {
        let mut builder = GgufBuilder::new();
        builder
            .add_string(KEY_MIMI_CHECKPOINT_SHA256, KYUTAI_STT_MIMI_SHA256)
            .add_metadata(
                KEY_MIMI_CHECKPOINT_BYTES,
                GgufMetadataValue::U64(KYUTAI_STT_MIMI_BYTES as u64),
            );
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert!(validate_mimi_conversion_identity(&file).is_ok());

        for (sha, bytes) in [
            (Some("00".repeat(32)), Some(KYUTAI_STT_MIMI_BYTES as u64)),
            (Some(KYUTAI_STT_MIMI_SHA256.to_owned()), Some(1)),
            (None, Some(KYUTAI_STT_MIMI_BYTES as u64)),
            (Some(KYUTAI_STT_MIMI_SHA256.to_owned()), None),
        ] {
            let mut invalid = GgufBuilder::new();
            if let Some(value) = sha {
                invalid.add_string(KEY_MIMI_CHECKPOINT_SHA256, &value);
            }
            if let Some(value) = bytes {
                invalid.add_metadata(KEY_MIMI_CHECKPOINT_BYTES, GgufMetadataValue::U64(value));
            }
            let invalid_file = GgufFile::parse(invalid.to_bytes().unwrap()).unwrap();
            assert!(validate_mimi_conversion_identity(&invalid_file).is_err());
        }
        let mut wrong_types = GgufBuilder::new();
        wrong_types.add_metadata(
            KEY_MIMI_CHECKPOINT_SHA256,
            GgufMetadataValue::U64(KYUTAI_STT_MIMI_BYTES as u64),
        );
        wrong_types.add_string(KEY_MIMI_CHECKPOINT_BYTES, KYUTAI_STT_MIMI_SHA256);
        let wrong_types_file = GgufFile::parse(wrong_types.to_bytes().unwrap()).unwrap();
        assert!(validate_mimi_conversion_identity(&wrong_types_file).is_err());
        let mut wrong_byte_type = GgufBuilder::new();
        wrong_byte_type.add_string(KEY_MIMI_CHECKPOINT_SHA256, KYUTAI_STT_MIMI_SHA256);
        wrong_byte_type.add_string(KEY_MIMI_CHECKPOINT_BYTES, KYUTAI_STT_MIMI_SHA256);
        let wrong_byte_type_file = GgufFile::parse(wrong_byte_type.to_bytes().unwrap()).unwrap();
        assert!(validate_mimi_conversion_identity(&wrong_byte_type_file).is_err());
    }

    #[test]
    fn authenticated_mmap_rejects_length_and_hash_mismatch() {
        let path = std::env::temp_dir().join(format!(
            "vokra-kyutai-pcm-auth-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut builder = GgufBuilder::new();
        builder.add_string("test.key", "synthetic");
        let bytes = builder.to_bytes().unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let actual_digest =
            KyutaiSttPcmArtifactDigest::new(&hex_digest(&sha256_bytes(&bytes)), bytes.len() as u64)
                .unwrap();
        assert!(open_authenticated_gguf(&path, "synthetic", &actual_digest).is_ok());
        let wrong_length = KyutaiSttPcmArtifactDigest::new(
            &hex_digest(&sha256_bytes(&bytes)),
            (bytes.len() as u64) + 1,
        )
        .unwrap();
        let length_error = open_authenticated_gguf(&path, "synthetic", &wrong_length).unwrap_err();
        assert!(format!("{length_error:?}").contains("byte count mismatch"));
        let wrong_hash =
            KyutaiSttPcmArtifactDigest::new(&"0".repeat(64), bytes.len() as u64).unwrap();
        let hash_error = open_authenticated_gguf(&path, "synthetic", &wrong_hash).unwrap_err();
        assert!(format!("{hash_error:?}").contains("SHA-256"));
        assert!(authenticate_raw_mimi(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn unsupported_backend_is_rejected_before_model_execution() {
        assert!(
            Compute::for_mimi_backend(BackendKind::Vulkan, crate::mimi::encoder::MIMI_HOT_OPS)
                .is_err()
        );
    }

    #[test]
    fn suppression_keeps_raw_eos_but_hides_zero_and_padding() {
        let contract = contract();
        let mut raw = Vec::new();
        let mut emitted = Vec::new();
        record_sampled_token(contract, &mut raw, &mut emitted, 0, true).unwrap();
        record_sampled_token(contract, &mut raw, &mut emitted, 2, true).unwrap();
        record_sampled_token(contract, &mut raw, &mut emitted, 3, true).unwrap();
        assert_eq!(raw, [0, 2, 3]);
        assert_eq!(emitted, [2]);
    }

    #[test]
    fn sampled_token_bounds_are_fail_closed() {
        let contract = contract();
        let mut raw = Vec::new();
        let mut emitted = Vec::new();
        assert!(
            record_sampled_token(
                contract,
                &mut raw,
                &mut emitted,
                contract.text_card() as u32,
                true,
            )
            .is_err()
        );
        assert!(raw.is_empty() && emitted.is_empty());
    }

    #[test]
    fn padding_arithmetic_is_exact_and_drops_partial_tail() {
        let contract =
            KyutaiSttStreamingContract::from_config(&super::super::KyutaiSttConfig::stt_2_6b_en())
                .unwrap();
        assert_eq!(contract.padded_frame_count(0).unwrap(), 56);
        assert_eq!(contract.padded_frame_count(1_920).unwrap(), 57);
        assert_eq!(contract.padded_frame_count(1_921).unwrap(), 57);
    }
}
