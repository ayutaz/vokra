//! Stochastic token sampler (temperature / top-k / top-p / repetition penalty)
//! over a model-independent [`LogitsSource`].
//!
//! Like [`beam_search`](super::beam_search), sampling is a **host-side runtime
//! function** (FR-OP-40 family), never a graph op: it drives any model through
//! the [`LogitsSource`] abstraction and knows nothing model-specific. It is the
//! sampled counterpart to greedy / beam decoding — the path a caller picks when
//! they want diversity rather than the single most-likely sequence.
//!
//! # Pipeline
//!
//! Each [`Sampler::sample`] call transforms the raw logits in a fixed order —
//! **temperature → repetition penalty → top-k → top-p → draw** — then draws one
//! token by inverse-CDF from the surviving nucleus. `temperature == 0` is an
//! exact short-circuit to greedy [`argmax`], **bit-identical** to the Whisper
//! greedy decoder, so a zero-temperature sampler reproduces greedy decoding.
//!
//! # Determinism
//!
//! The draw uses a seeded [`SplitMix64`]; a fixed [`SamplerConfig::seed`] plus a
//! fixed logits stream yields a fixed token stream (the shared-RNG reproducibility
//! guarantee).

use std::cmp::Ordering;

use crate::error::{Result, VokraError};
use crate::rng::SplitMix64;

use super::LogitsSource;

/// Sampling hyper-parameters.
///
/// Every knob past `temperature` is optional; `None` disables that stage, so the
/// default-ish "just temperature" config is the plain softmax sampler and
/// `temperature == 0` is greedy.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Softmax temperature. `> 0` sharpens (`< 1`) or flattens (`> 1`) the
    /// distribution; **exactly `0`** selects greedy [`argmax`] (no draw).
    pub temperature: f32,
    /// Keep only the `top_k` highest-logit tokens before drawing (`None` = keep
    /// all). `Some(1)` is argmax for any positive temperature.
    pub top_k: Option<usize>,
    /// Nucleus threshold: keep the smallest set of highest-probability tokens
    /// whose cumulative probability reaches `top_p` (`None` = keep all). The
    /// top token is always kept.
    pub top_p: Option<f32>,
    /// Divide (logit `> 0`) or multiply (logit `<= 0`) the logit of every
    /// already-emitted token by this factor (`> 1` discourages repeats), the
    /// CTRL repetition penalty. `None` disables it.
    pub repetition_penalty: Option<f32>,
    /// Seed for the draw RNG — fixes the sampled sequence for a given logits
    /// stream.
    pub seed: u64,
}

impl SamplerConfig {
    /// A greedy config (`temperature == 0`): every [`Sampler::sample`] returns
    /// [`argmax`], matching the Whisper greedy decoder token-for-token.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 0,
        }
    }
}

/// A stateful sampler: a seeded RNG, a [`SamplerConfig`], and the running set of
/// emitted tokens (needed for [`SamplerConfig::repetition_penalty`]).
#[derive(Debug, Clone)]
pub struct Sampler {
    rng: SplitMix64,
    cfg: SamplerConfig,
    /// Tokens this sampler has drawn so far (the repetition-penalty context).
    emitted: Vec<u32>,
}

impl Sampler {
    /// Builds a sampler for `cfg`, seeding the draw RNG from
    /// [`SamplerConfig::seed`].
    #[must_use]
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = SplitMix64::new(cfg.seed);
        Self {
            rng,
            cfg,
            emitted: Vec::new(),
        }
    }

    /// Draws one token from `logits`, recording it in the repetition-penalty
    /// context. `logits` is mutated in place by the temperature and
    /// repetition-penalty stages (the top-k / top-p filtering happens on a
    /// working copy).
    ///
    /// With `temperature == 0` this is exactly [`argmax`] over the untouched
    /// `logits` (no RNG draw), so a zero-temperature sampler is greedy.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        // Greedy short-circuit: bit-identical to the Whisper greedy decoder, and
        // it consumes no RNG so switching temperature never perturbs a later
        // stochastic draw's seed position.
        if self.cfg.temperature == 0.0 {
            let tok = argmax(logits);
            self.emitted.push(tok);
            return tok;
        }

        // 1. Temperature: scale logits by 1/T (T > 0 preserves order).
        let inv_t = 1.0 / self.cfg.temperature;
        for l in logits.iter_mut() {
            *l *= inv_t;
        }

        // 2. Repetition penalty over already-emitted tokens (CTRL form).
        if let Some(p) = self.cfg.repetition_penalty {
            for &t in &self.emitted {
                let idx = t as usize;
                if idx < logits.len() {
                    let v = logits[idx];
                    logits[idx] = if v > 0.0 { v / p } else { v * p };
                }
            }
        }

        let tok = self.draw(logits);
        self.emitted.push(tok);
        tok
    }

    /// Top-k / top-p filter and inverse-CDF draw over `logits` (already
    /// temperature- and penalty-adjusted). Reads only; never mutates `logits`.
    fn draw(&mut self, logits: &[f32]) -> u32 {
        // Candidate indices, highest logit first (ties broken by index, so the
        // top-1 candidate is the argmax first-on-ties — matching `argmax`).
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });

        // top-k: keep the k highest-logit candidates (at least one).
        if let Some(k) = self.cfg.top_k {
            idx.truncate(k.max(1).min(idx.len()));
        }

        // Softmax over the survivors, stabilized against the max (= idx[0]).
        let max = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| (logits[i] - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for pr in &mut probs {
            *pr /= sum;
        }

        // top-p (nucleus): shortest descending prefix whose cumulative
        // probability reaches `top_p`; the top token is always kept.
        let mut n = probs.len();
        if let Some(top_p) = self.cfg.top_p {
            let mut cum = 0.0f32;
            let mut cut = probs.len();
            for (j, &pr) in probs.iter().enumerate() {
                cum += pr;
                if cum >= top_p {
                    cut = j + 1;
                    break;
                }
            }
            n = cut.max(1);
        }

        // Inverse-CDF draw over the (renormalized) nucleus.
        let nucleus_sum: f32 = probs[..n].iter().sum();
        let u = self.rng.next_unit_f32() * nucleus_sum;
        let mut cum = 0.0f32;
        for (j, &pr) in probs[..n].iter().enumerate() {
            cum += pr;
            if u <= cum {
                return idx[j] as u32;
            }
        }
        // Float-rounding fallback: the last surviving candidate.
        idx[n - 1] as u32
    }
}

/// Index of the maximum element (first on ties) — the greedy token.
///
/// Bit-identical to the Whisper greedy decoder's arg-max: one pass keeping the
/// first strictly-greater element, initialized at `f32::NEG_INFINITY`. This is
/// the single shared arg-max used by both greedy decoding and the
/// zero-temperature [`Sampler`].
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Autoregressively samples tokens from `src`, starting from the forced
/// `prefix`, until `eot` is drawn or `max_new` tokens are produced.
///
/// The sampled counterpart to greedy / beam decoding: returns the **generated**
/// tokens only (the `prefix` is excluded; the terminal `eot` **is** included
/// when drawn), matching the greedy decoder's contract.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] for an empty `prefix`, or if `src` returns a
///   logits vector whose length disagrees with [`LogitsSource::vocab_size`];
/// - any error surfaced by the [`LogitsSource`].
pub fn sample_sequence(
    src: &mut dyn LogitsSource,
    prefix: &[u32],
    eot: u32,
    cfg: &SamplerConfig,
    max_new: usize,
) -> Result<Vec<u32>> {
    if prefix.is_empty() {
        return Err(VokraError::InvalidArgument(
            "sample_sequence: prefix must not be empty".into(),
        ));
    }
    let mut sampler = Sampler::new(cfg.clone());
    let mut tokens = prefix.to_vec();
    let mut generated = Vec::new();
    for _ in 0..max_new {
        let mut logits = src.logits(&tokens)?;
        if logits.len() != src.vocab_size() {
            return Err(VokraError::InvalidArgument(format!(
                "sample_sequence: source returned {} logits, expected vocab_size {}",
                logits.len(),
                src.vocab_size()
            )));
        }
        let next = sampler.sample(&mut logits);
        generated.push(next);
        tokens.push(next);
        if next == eot {
            break;
        }
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(temperature: f32) -> SamplerConfig {
        SamplerConfig {
            temperature,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: 0x0123_4567_89AB_CDEF,
        }
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let mut s = Sampler::new(cfg(0.0));
        // Tie at 0.5 → first (index 1), exactly like the greedy decoder.
        let logits = [0.1f32, 0.5, 0.5, 0.2];
        let mut work = logits;
        assert_eq!(s.sample(&mut work), 1);
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn top_k_one_is_argmax_at_any_temperature() {
        let logits = [0.3f32, 2.1, -1.0, 2.0, 0.5];
        let expected = argmax(&logits); // index 1
        for &t in &[0.25f32, 1.0, 3.0, 25.0] {
            let mut c = cfg(t);
            c.top_k = Some(1);
            let mut s = Sampler::new(c);
            let mut work = logits;
            assert_eq!(
                s.sample(&mut work),
                expected,
                "top_k=1 must equal argmax at temperature {t}"
            );
        }
    }

    #[test]
    fn same_seed_yields_identical_draws() {
        let base = [0.5f32, 1.0, 0.8, 1.2, 0.3];
        let mut a = Sampler::new(cfg(1.0));
        let mut b = Sampler::new(cfg(1.0));
        let mut sa = Vec::new();
        let mut sb = Vec::new();
        for _ in 0..48 {
            let mut la = base;
            let mut lb = base;
            sa.push(a.sample(&mut la));
            sb.push(b.sample(&mut lb));
        }
        assert_eq!(sa, sb, "same seed + same logits ⇒ identical draws");
        // The RNG is actually exercised (not a degenerate argmax collapse).
        assert!(
            sa.iter().any(|&t| t != sa[0]),
            "expected varied draws, got {sa:?}"
        );
    }

    #[test]
    fn different_seed_can_diverge() {
        let base = [0.5f32, 1.0, 0.8, 1.2, 0.3];
        let mut c2 = cfg(1.0);
        c2.seed = 0xFFFF_0000_FFFF_0000;
        let mut a = Sampler::new(cfg(1.0));
        let mut b = Sampler::new(c2);
        let sa: Vec<u32> = (0..48)
            .map(|_| {
                let mut l = base;
                a.sample(&mut l)
            })
            .collect();
        let sb: Vec<u32> = (0..48)
            .map(|_| {
                let mut l = base;
                b.sample(&mut l)
            })
            .collect();
        assert_ne!(sa, sb, "different seeds should not lockstep");
    }

    #[test]
    fn repetition_penalty_strictly_lowers_an_emitted_logit() {
        let mut c = cfg(1.0); // T = 1 ⇒ temperature scaling is a no-op
        c.repetition_penalty = Some(2.0);
        let mut s = Sampler::new(c);

        let base = [3.0f32, 1.0, 0.5, 0.2]; // all positive
        // First draw records some token `t`.
        let mut first = base;
        let t = s.sample(&mut first) as usize;

        // Second draw applies the penalty to `t`; observe the mutated logit.
        let before = base[t];
        let mut second = base;
        let _ = s.sample(&mut second);
        assert!(
            second[t] < before,
            "penalized logit {} must be strictly below the original {before}",
            second[t]
        );
        // The exact CTRL form for a positive logit: divided by the penalty.
        assert!((second[t] - before / 2.0).abs() < 1e-6);
    }

    #[test]
    fn small_top_p_draws_from_the_nucleus() {
        let mut c = cfg(1.0);
        c.top_p = Some(0.5);
        let mut s = Sampler::new(c);
        // token 0 dominates ⇒ its probability alone exceeds 0.5 ⇒ nucleus = {0}.
        let base = [10.0f32, 0.0, 0.0, 0.0];
        for _ in 0..24 {
            let mut work = base;
            assert_eq!(
                s.sample(&mut work),
                0,
                "a tiny nucleus must only ever draw its single dominant token"
            );
        }
    }

    /// A [`LogitsSource`] returning a fixed logits row every step (a stand-in
    /// model for the sequence-level tests; internal oracle, no reference data).
    struct FixedSource {
        row: Vec<f32>,
    }
    impl LogitsSource for FixedSource {
        fn logits(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            Ok(self.row.clone())
        }
        fn vocab_size(&self) -> usize {
            self.row.len()
        }
    }

    #[test]
    fn sample_sequence_greedy_stops_at_eot() {
        // argmax is token 2; with temperature 0 the sampler is greedy, so the
        // very first draw is the eot and generation stops after one token.
        let mut src = FixedSource {
            row: vec![0.1, 0.2, 5.0, 0.0],
        };
        let out = sample_sequence(&mut src, &[9], /*eot*/ 2, &SamplerConfig::greedy(), 16).unwrap();
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn sample_sequence_respects_max_new_without_eot() {
        // argmax is token 1; eot is out of the produced set, so generation runs
        // to the cap and yields exactly `max_new` copies of the argmax token.
        let mut src = FixedSource {
            row: vec![0.1, 5.0, 0.2],
        };
        let out = sample_sequence(&mut src, &[7], /*eot*/ 99, &SamplerConfig::greedy(), 4).unwrap();
        assert_eq!(out, vec![1, 1, 1, 1]);
    }

    #[test]
    fn sample_sequence_rejects_empty_prefix() {
        let mut src = FixedSource {
            row: vec![1.0, 2.0],
        };
        assert!(matches!(
            sample_sequence(&mut src, &[], 0, &SamplerConfig::greedy(), 4),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn sample_sequence_rejects_vocab_mismatch() {
        // vocab_size() reports 3 but the row is length 2.
        struct BadSource;
        impl LogitsSource for BadSource {
            fn logits(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
                Ok(vec![0.0, 0.0])
            }
            fn vocab_size(&self) -> usize {
                3
            }
        }
        let mut src = BadSource;
        assert!(matches!(
            sample_sequence(&mut src, &[1], 0, &SamplerConfig::greedy(), 4),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
