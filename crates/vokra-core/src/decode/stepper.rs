//! Cache-based incremental ASR decode stepper (FR-ST-01 cache pattern, M1-08f).
//!
//! [`DecodeStepper`] is the third first-class stepping pattern: a stepper whose
//! hidden state is a KV cache. It advances an autoregressive decoder one token
//! per [`step`](DecodeStepper::step), emitting each drawn token as a
//! [`StreamEvent::Token`] — the KV cache stays hidden inside the underlying
//! [`LogitsSource`], so no tensor name is ever exposed (FR-ST-05).
//!
//! It is deliberately **model-independent**: it drives any
//! [`LogitsSource`] through the same [`Sampler`] pipeline as
//! [`sample_sequence`](super::sample_sequence), so incremental stepping yields
//! the *identical* token stream to a one-shot decode (the invariant its test
//! oracle pins). The concrete wiring — wrapping the M1-04 owned
//! `WhisperLogitsSource` (which owns its [`KvCache`](crate::KvCache) via
//! `DecoderState`) and exposing it as a [`Session`](crate::Session) stream — is a
//! `vokra-models` follow-up; this is the reusable core it builds on.
//!
//! This is incremental **decode** stepping, not full audio-in streaming ASR
//! (chunked-mel front-end + sliding-window encoder + LocalAgreement), which is a
//! larger model-sync effort tracked separately.

use crate::error::{Result, VokraError};
use crate::stream::{EventSink, StreamEvent};

use super::LogitsSource;
use super::sampler::{Sampler, SamplerConfig};

/// [`StreamEvent::Token::flags`] bit set on the terminal end-of-transcript token.
pub const TOKEN_FLAG_EOT: u32 = 1;

/// An incremental decode stepper over a [`LogitsSource`].
///
/// Construct with [`new`](Self::new) (a forced `prefix` such as Whisper's
/// `[SOT, lang, task]`), then call [`step`](Self::step) to advance one token, or
/// [`run`](Self::run) to advance until end-of-transcript or a cap. `Send`
/// (the source is boxed `+ Send`), so the stepper can be moved onto a worker
/// thread just like any [`StreamStep`](crate::stream::StreamStep).
pub struct DecodeStepper {
    src: Box<dyn LogitsSource + Send>,
    sampler: Sampler,
    /// The full running sequence (forced prefix + generated tokens).
    tokens: Vec<u32>,
    eot: u32,
    finished: bool,
}

impl DecodeStepper {
    /// Builds a stepper that starts from the forced `prefix` and stops when it
    /// draws `eot`. `cfg` is the same [`SamplerConfig`] a one-shot decode would
    /// use (`temperature == 0` ⇒ greedy, matching the Whisper greedy decoder).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `prefix` is empty.
    pub fn new(
        src: Box<dyn LogitsSource + Send>,
        prefix: &[u32],
        eot: u32,
        cfg: SamplerConfig,
    ) -> Result<Self> {
        if prefix.is_empty() {
            return Err(VokraError::InvalidArgument(
                "DecodeStepper: prefix must not be empty".into(),
            ));
        }
        Ok(Self {
            src,
            sampler: Sampler::new(cfg),
            tokens: prefix.to_vec(),
            eot,
            finished: false,
        })
    }

    /// Whether the terminal `eot` token has been drawn.
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Advances the decoder by one token: queries the source for the next
    /// logits, draws a token through the [`Sampler`], appends it to the running
    /// sequence and emits it as a [`StreamEvent::Token`]. Returns the drawn token
    /// id, or `None` once finished (idempotent tail).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the source returns a logits vector
    /// whose length disagrees with [`LogitsSource::vocab_size`], or any error
    /// from the source.
    pub fn step(&mut self, sink: &mut dyn EventSink) -> Result<Option<u32>> {
        if self.finished {
            return Ok(None);
        }
        let mut logits = self.src.logits(&self.tokens)?;
        if logits.len() != self.src.vocab_size() {
            return Err(VokraError::InvalidArgument(format!(
                "DecodeStepper: source returned {} logits, expected vocab_size {}",
                logits.len(),
                self.src.vocab_size()
            )));
        }
        let tok = self.sampler.sample(&mut logits);
        self.tokens.push(tok);
        let is_eot = tok == self.eot;
        let flags = if is_eot { TOKEN_FLAG_EOT } else { 0 };
        let _accepted = sink.emit(StreamEvent::Token { id: tok, flags });
        if is_eot {
            self.finished = true;
        }
        Ok(Some(tok))
    }

    /// Advances until `eot` is drawn or `max_new` tokens are produced, emitting
    /// each into `sink`. Returns the number of tokens produced (the terminal
    /// `eot` is counted). This is the streaming counterpart to
    /// [`sample_sequence`](super::sample_sequence) and yields the identical token
    /// stream for the same source / prefix / config.
    pub fn run(&mut self, sink: &mut dyn EventSink, max_new: usize) -> Result<usize> {
        let mut produced = 0;
        while produced < max_new {
            match self.step(sink)? {
                Some(_) => {
                    produced += 1;
                    if self.finished {
                        break;
                    }
                }
                None => break,
            }
        }
        Ok(produced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::sample_sequence;

    /// A deterministic stateful source: the argmax walks `0, 1, 2, …` (mod
    /// vocab) as calls accumulate, so decoding produces a *varied* token stream
    /// (not a constant), a stronger oracle than a fixed row. Each decode path
    /// gets its own fresh instance (the source is stateful). Internal oracle, no
    /// reference data.
    struct RampSource {
        calls: usize,
        vocab: usize,
    }
    impl RampSource {
        fn new(vocab: usize) -> Self {
            Self { calls: 0, vocab }
        }
    }
    impl LogitsSource for RampSource {
        fn logits(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            let mut row = vec![0.0f32; self.vocab];
            row[self.calls % self.vocab] = 1.0;
            self.calls += 1;
            Ok(row)
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
    }

    fn tokens_from_stepper(cfg: SamplerConfig, eot: u32, max_new: usize) -> Vec<u32> {
        let mut stepper =
            DecodeStepper::new(Box::new(RampSource::new(8)), &[100], eot, cfg).expect("stepper");
        let mut sink: Vec<StreamEvent> = Vec::new();
        stepper.run(&mut sink, max_new).expect("run");
        sink.into_iter()
            .map(|e| match e {
                StreamEvent::Token { id, .. } => id,
                other => panic!("unexpected event {other:?}"),
            })
            .collect()
    }

    #[test]
    fn incremental_stepping_equals_one_shot_decode() {
        // The full-vs-incremental invariant: DecodeStepper::run must reproduce
        // sample_sequence token-for-token for the same source/prefix/config.
        let cfg = SamplerConfig::greedy();
        let eot = 5;
        let batch =
            sample_sequence(&mut RampSource::new(8), &[100], eot, &cfg, 16).expect("batch decode");
        let incremental = tokens_from_stepper(cfg, eot, 16);
        assert_eq!(incremental, batch, "incremental decode == one-shot decode");
        // And it is the varied ramp that hits eot: [0,1,2,3,4,5].
        assert_eq!(incremental, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn replay_is_deterministic() {
        let a = tokens_from_stepper(SamplerConfig::greedy(), 5, 16);
        let b = tokens_from_stepper(SamplerConfig::greedy(), 5, 16);
        assert_eq!(a, b, "same source/prefix/config ⇒ identical token stream");
    }

    #[test]
    fn eot_flag_marks_the_terminal_token_and_finishes() {
        let mut stepper = DecodeStepper::new(
            Box::new(RampSource::new(8)),
            &[100],
            2,
            SamplerConfig::greedy(),
        )
        .expect("stepper");
        let mut sink: Vec<StreamEvent> = Vec::new();
        // Tokens 0, 1 are non-eot; token 2 is eot.
        for _ in 0..2 {
            stepper.step(&mut sink).expect("step");
            assert!(!stepper.finished());
        }
        stepper.step(&mut sink).expect("eot step");
        assert!(stepper.finished());
        // Further steps are idempotent no-ops.
        assert_eq!(stepper.step(&mut sink).expect("post-eot"), None);

        let flags: Vec<u32> = sink
            .iter()
            .map(|e| match e {
                StreamEvent::Token { flags, .. } => *flags,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(flags, vec![0, 0, TOKEN_FLAG_EOT], "only the last is eot");
    }

    #[test]
    fn rejects_empty_prefix() {
        assert!(matches!(
            DecodeStepper::new(
                Box::new(RampSource::new(4)),
                &[],
                0,
                SamplerConfig::greedy()
            ),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_vocab_mismatch() {
        struct BadSource;
        impl LogitsSource for BadSource {
            fn logits(&mut self, _t: &[u32]) -> Result<Vec<f32>> {
                Ok(vec![0.0, 0.0])
            }
            fn vocab_size(&self) -> usize {
                3
            }
        }
        let mut stepper =
            DecodeStepper::new(Box::new(BadSource), &[1], 0, SamplerConfig::greedy()).expect("new");
        let mut sink: Vec<StreamEvent> = Vec::new();
        assert!(matches!(
            stepper.step(&mut sink),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn stepper_is_send() {
        const fn assert_send<T: Send>() {}
        assert_send::<DecodeStepper>();
    }
}
