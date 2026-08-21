//! Continuous Mimi speech features exposed through the model-independent
//! session engine seam (#49).

use std::sync::Arc;

use vokra_core::engines::{SpeechFeatureEngine, SpeechFeatureStream};
use vokra_core::{Result, VokraError};

use super::MoshiEngine;
use crate::mimi::MimiEncoderState;

/// Bounded pending-output budget. At the released 25 Hz / 512-d geometry this
/// is 10.24 seconds and 512 KiB, all allocated at stream construction.
const DEFAULT_PENDING_FEATURE_FRAMES: usize = 256;

struct MoshiFeatureStream {
    engine: Arc<MoshiEngine>,
    encoder_state: MimiEncoderState,
    token_hop: usize,
    feature_hop: usize,
    feature_dim: usize,
    features_per_token_frame: usize,
    frame_rate_millihz: u32,
    partial_pcm: Vec<f32>,
    partial_len: usize,
    feature_scratch: Vec<f32>,
    pending: Vec<f32>,
    pending_capacity_frames: usize,
    pending_head: usize,
    pending_len: usize,
    next_output_sample: i64,
}

impl MoshiFeatureStream {
    fn new(engine: Arc<MoshiEngine>) -> Result<Self> {
        let encoder = engine.encoder();
        let token_hop = encoder.frame_hop()?;
        let feature_hop = encoder.feature_frame_hop()?;
        let feature_dim = encoder.feature_dim();
        if feature_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "moshi feature stream: feature dimension must be > 0".into(),
            ));
        }
        if token_hop % feature_hop != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "moshi feature stream: token hop {token_hop} is not divisible by feature hop \
                 {feature_hop}"
            )));
        }
        let features_per_token_frame = token_hop / feature_hop;
        let rate_num = u64::from(encoder.config().sample_rate) * 1000;
        if rate_num % feature_hop as u64 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "moshi feature stream: sample rate {} does not form an exact milli-Hz rate at \
                 feature hop {feature_hop}",
                encoder.config().sample_rate
            )));
        }
        let frame_rate_millihz = u32::try_from(rate_num / feature_hop as u64).map_err(|_| {
            VokraError::InvalidArgument(
                "moshi feature stream: frame rate does not fit u32 milli-Hz".into(),
            )
        })?;
        let scratch_len = features_per_token_frame
            .checked_mul(feature_dim)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "moshi feature stream: scratch geometry overflows usize".into(),
                )
            })?;
        let pending_len = DEFAULT_PENDING_FEATURE_FRAMES
            .checked_mul(feature_dim)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "moshi feature stream: pending geometry overflows usize".into(),
                )
            })?;
        let encoder_state = encoder.state(1)?;
        Ok(Self {
            engine,
            encoder_state,
            token_hop,
            feature_hop,
            feature_dim,
            features_per_token_frame,
            frame_rate_millihz,
            partial_pcm: vec![0.0; token_hop],
            partial_len: 0,
            feature_scratch: vec![0.0; scratch_len],
            pending: vec![0.0; pending_len],
            pending_capacity_frames: DEFAULT_PENDING_FEATURE_FRAMES,
            pending_head: 0,
            pending_len: 0,
            next_output_sample: 0,
        })
    }

    fn encode_frame(&mut self, pcm: &[f32]) -> Result<()> {
        self.engine.encoder().encode_features_into(
            &mut self.encoder_state,
            pcm,
            &mut self.feature_scratch,
        )?;
        for frame in 0..self.features_per_token_frame {
            let tail = (self.pending_head + self.pending_len) % self.pending_capacity_frames;
            let src =
                &self.feature_scratch[frame * self.feature_dim..(frame + 1) * self.feature_dim];
            let dst = &mut self.pending[tail * self.feature_dim..(tail + 1) * self.feature_dim];
            dst.copy_from_slice(src);
            self.pending_len += 1;
        }
        Ok(())
    }

    fn encode_partial_frame(&mut self) -> Result<()> {
        // Borrow the disjoint state/scratch/queue fields directly so the full
        // `partial_pcm` frame never needs a temporary copy.
        self.engine.encoder().encode_features_into(
            &mut self.encoder_state,
            &self.partial_pcm,
            &mut self.feature_scratch,
        )?;
        for frame in 0..self.features_per_token_frame {
            let tail = (self.pending_head + self.pending_len) % self.pending_capacity_frames;
            let src =
                &self.feature_scratch[frame * self.feature_dim..(frame + 1) * self.feature_dim];
            let dst = &mut self.pending[tail * self.feature_dim..(tail + 1) * self.feature_dim];
            dst.copy_from_slice(src);
            self.pending_len += 1;
        }
        Ok(())
    }
}

impl SpeechFeatureStream for MoshiFeatureStream {
    fn sample_rate(&self) -> u32 {
        self.engine.encoder().config().sample_rate
    }

    fn frame_rate_millihz(&self) -> u32 {
        self.frame_rate_millihz
    }

    fn feature_frame_hop(&self) -> usize {
        self.feature_hop
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    // ZERO-ALLOC-BEGIN (#49: C-ABI-facing streaming feature push/pull)
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<()> {
        let combined = self.partial_len.checked_add(pcm.len()).ok_or_else(|| {
            VokraError::InvalidArgument(
                "moshi feature stream: cumulative push length overflows usize".into(),
            )
        })?;
        let completed_token_frames = combined / self.token_hop;
        let produced = completed_token_frames
            .checked_mul(self.features_per_token_frame)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "moshi feature stream: produced feature count overflows usize".into(),
                )
            })?;
        let free = self.pending_capacity_frames - self.pending_len;
        if produced > free {
            return Err(VokraError::InvalidArgument(format!(
                "moshi feature stream: push would produce {produced} frames but bounded queue \
                 has {free} free; pull pending features before retrying (state unchanged)"
            )));
        }

        let mut rest = pcm;
        if self.partial_len > 0 {
            let take = (self.token_hop - self.partial_len).min(rest.len());
            self.partial_pcm[self.partial_len..self.partial_len + take]
                .copy_from_slice(&rest[..take]);
            self.partial_len += take;
            rest = &rest[take..];
            if self.partial_len == self.token_hop {
                self.encode_partial_frame()?;
                self.partial_len = 0;
            } else {
                return Ok(());
            }
        }

        while rest.len() >= self.token_hop {
            self.encode_frame(&rest[..self.token_hop])?;
            rest = &rest[self.token_hop..];
        }
        self.partial_pcm[..rest.len()].copy_from_slice(rest);
        self.partial_len = rest.len();
        Ok(())
    }

    fn pull_into(&mut self, out: &mut [f32]) -> Result<(usize, i64)> {
        if !out.is_empty() && out.len() < self.feature_dim {
            return Err(VokraError::InvalidArgument(format!(
                "moshi feature stream: output capacity {} floats is smaller than one \
                 feature row of {} floats (state unchanged)",
                out.len(),
                self.feature_dim
            )));
        }
        let frames = (out.len() / self.feature_dim).min(self.pending_len);
        let sample_delta = frames.checked_mul(self.feature_hop).ok_or_else(|| {
            VokraError::InvalidArgument(
                "moshi feature stream: timestamp delta overflows usize".into(),
            )
        })?;
        let sample_delta = i64::try_from(sample_delta).map_err(|_| {
            VokraError::InvalidArgument(
                "moshi feature stream: timestamp delta does not fit i64".into(),
            )
        })?;
        let new_timestamp = self
            .next_output_sample
            .checked_add(sample_delta)
            .ok_or_else(|| {
                VokraError::InvalidArgument("moshi feature stream: timestamp overflows i64".into())
            })?;
        let start = self.next_output_sample;
        for frame in 0..frames {
            let slot = (self.pending_head + frame) % self.pending_capacity_frames;
            let src = &self.pending[slot * self.feature_dim..(slot + 1) * self.feature_dim];
            let dst = &mut out[frame * self.feature_dim..(frame + 1) * self.feature_dim];
            dst.copy_from_slice(src);
        }
        self.pending_head = (self.pending_head + frames) % self.pending_capacity_frames;
        self.pending_len -= frames;
        self.next_output_sample = new_timestamp;
        Ok((frames, start))
    }
    // ZERO-ALLOC-END

    fn reset(&mut self) {
        self.encoder_state.reset();
        self.partial_len = 0;
        self.pending_head = 0;
        self.pending_len = 0;
        self.next_output_sample = 0;
    }
}

impl SpeechFeatureEngine for MoshiEngine {
    fn open_feature_stream(self: Arc<Self>) -> Result<Box<dyn SpeechFeatureStream + Send>> {
        Ok(Box::new(MoshiFeatureStream::new(self)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vokra_core::engines::SpeechFeatureEngine;

    use super::{DEFAULT_PENDING_FEATURE_FRAMES, MoshiEngine};

    fn pcm(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.031).cos()) * 0.25)
            .collect()
    }

    #[test]
    fn arbitrary_pcm_chunks_match_whole_buffer_continuous_forward_exactly() {
        let engine = Arc::new(MoshiEngine::synthesized_fixture(49).unwrap());
        let token_hop = engine.encoder().frame_hop().unwrap();
        let input = pcm(token_hop * 5);
        let want = engine.encoder().encode_features_all(&input).unwrap();
        let mut stream = engine.clone().open_feature_stream().unwrap();

        let chunk_sizes = [1usize, 17, 511, token_hop + 3, 29, 777];
        let mut cursor = 0usize;
        let mut chunk = 0usize;
        while cursor < input.len() {
            let take = chunk_sizes[chunk % chunk_sizes.len()].min(input.len() - cursor);
            stream.push_pcm(&input[cursor..cursor + take]).unwrap();
            cursor += take;
            chunk += 1;
        }

        let mut got = vec![0.0f32; want.len()];
        let (frames, start_sample) = stream.pull_into(&mut got).unwrap();
        assert_eq!(start_sample, 0);
        assert_eq!(frames * stream.feature_dim(), want.len());
        assert_eq!(got, want);
    }

    #[test]
    fn metadata_and_partial_pulls_keep_sample_accurate_timestamps() {
        let engine = Arc::new(MoshiEngine::synthesized_fixture(50).unwrap());
        let token_hop = engine.encoder().frame_hop().unwrap();
        let mut stream = engine.open_feature_stream().unwrap();
        assert_eq!(
            u64::from(stream.frame_rate_millihz()) * stream.feature_frame_hop() as u64,
            u64::from(stream.sample_rate()) * 1000,
        );

        stream.push_pcm(&pcm(token_hop * 2)).unwrap();
        let mut one = vec![0.0f32; stream.feature_dim()];
        for expected_frame in 0..4i64 {
            let (frames, start) = stream.pull_into(&mut one).unwrap();
            assert_eq!(frames, 1);
            assert_eq!(start, expected_frame * stream.feature_frame_hop() as i64,);
        }
        assert_eq!(stream.pull_into(&mut one).unwrap().0, 0);
    }

    #[test]
    fn reset_discards_tail_and_pending_features_and_restarts_timestamp_zero() {
        let engine = Arc::new(MoshiEngine::synthesized_fixture(51).unwrap());
        let token_hop = engine.encoder().frame_hop().unwrap();
        let input = pcm(token_hop);
        let mut stream = engine.open_feature_stream().unwrap();
        stream.push_pcm(&input[..token_hop / 2]).unwrap();
        stream.push_pcm(&input[token_hop / 2..]).unwrap();
        let mut first = vec![0.0f32; stream.feature_dim() * 2];
        assert_eq!(stream.pull_into(&mut first).unwrap(), (2, 0));

        stream.push_pcm(&input).unwrap();
        stream.reset();
        let mut empty = vec![0.0f32; stream.feature_dim()];
        assert_eq!(stream.pull_into(&mut empty).unwrap().0, 0);
        stream.push_pcm(&input).unwrap();
        let mut second = vec![0.0f32; first.len()];
        assert_eq!(stream.pull_into(&mut second).unwrap(), (2, 0));
        assert_eq!(second, first);
    }

    #[test]
    fn non_frame_sized_pull_is_rejected_without_consuming_output() {
        let engine = Arc::new(MoshiEngine::synthesized_fixture(52).unwrap());
        let token_hop = engine.encoder().frame_hop().unwrap();
        let mut stream = engine.open_feature_stream().unwrap();
        stream.push_pcm(&pcm(token_hop)).unwrap();
        let mut short = vec![0.0f32; stream.feature_dim() - 1];
        assert!(stream.pull_into(&mut short).is_err());
        let mut full = vec![0.0f32; stream.feature_dim() * 2];
        assert_eq!(stream.pull_into(&mut full).unwrap(), (2, 0));
    }

    #[test]
    fn bounded_queue_backpressure_rejects_before_consuming_pcm() {
        let engine = Arc::new(MoshiEngine::synthesized_fixture(53).unwrap());
        let token_hop = engine.encoder().frame_hop().unwrap();
        let mut stream = engine.clone().open_feature_stream().unwrap();
        let features_per_token_frame = token_hop / stream.feature_frame_hop();
        let too_many_token_frames = DEFAULT_PENDING_FEATURE_FRAMES / features_per_token_frame + 1;
        assert!(
            stream
                .push_pcm(&pcm(token_hop * too_many_token_frames))
                .is_err(),
        );

        let one_frame = pcm(token_hop);
        stream.push_pcm(&one_frame).unwrap();
        let mut got = vec![0.0f32; stream.feature_dim() * 2];
        assert_eq!(stream.pull_into(&mut got).unwrap(), (2, 0));

        let mut fresh = engine.open_feature_stream().unwrap();
        fresh.push_pcm(&one_frame).unwrap();
        let mut want = vec![0.0f32; got.len()];
        assert_eq!(fresh.pull_into(&mut want).unwrap(), (2, 0));
        assert_eq!(got, want, "rejected push must not advance recurrent state");
    }
}
