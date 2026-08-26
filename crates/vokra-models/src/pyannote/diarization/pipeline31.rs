//! Exact native execution of `pyannote/speaker-diarization-3.1`.
//!
//! This module follows pyannote.audio 3.1.1 at source revision
//! `6a972c0c4e95de04637d7221208736c64c8b972a`: five-second segmentation
//! windows with a 0.1-duration step, hard powerset-to-multilabel conversion,
//! overlap-aware masked WeSpeaker embeddings, cosine/centroid clustering,
//! speaker counting, and discrete-diarization reconstruction.

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::{BackendKind, Result, VokraError};
use vokra_ops::clustering::{AgglomerativeClustering, DistanceMetric, LinkageMethod};

use crate::wespeaker::{EMBED_DIM, WeSpeaker};

use super::{PIPELINE_NAME, PyannoteSpeakerDiarization31Config};
use crate::pyannote::rttm::{DiarizationSegment, write_rttm};
use crate::pyannote::{POWERSET_MAPPING_3SPK_2OVERLAP, PyanNet};

/// Required input sample rate of both exact dependency models.
pub const SAMPLE_RATE: u32 = 16_000;
/// PyanNet training/inference window: five seconds at 16 kHz.
pub const SEGMENTATION_WINDOW_SAMPLES: usize = 80_000;
/// Public pipeline `segmentation_step=0.1`: half a second at 16 kHz.
pub const SEGMENTATION_STEP_SAMPLES: usize = 8_000;
/// Exact PyanNet output rows for one five-second window.
pub const SEGMENTATION_FRAMES: usize = 293;
/// Local speaker lanes of the segmentation-3.0 powerset model.
pub const LOCAL_SPEAKERS: usize = 3;
/// Smallest waveform accepted by the pinned WeSpeaker Kaldi frontend.
const WESPEAKER_MIN_SAMPLES: usize = 400;

/// Exact three-GGUF native handle for speaker-diarization 3.1.
///
/// The pipeline GGUF is a strict weightless orchestration contract. Learned
/// weights remain in the separately audited PyanNet and WeSpeaker GGUFs so
/// each artifact keeps its own provenance and license gate.
#[derive(Debug)]
pub struct PyannoteSpeakerDiarization31 {
    config: PyannoteSpeakerDiarization31Config,
    segmentation: PyanNet,
    embedding: WeSpeaker,
    backend: BackendKind,
}

impl PyannoteSpeakerDiarization31 {
    /// Opens the exact pipeline, segmentation and embedding artifacts.
    ///
    /// CPU and Metal are the only accepted backends. PyanNet performs eager
    /// backend preflight; WeSpeaker uses the same selected backend for every
    /// learned Conv2D/projection and returns an explicit error if that backend
    /// is unavailable. No dependency model silently falls back to CPU.
    pub fn open(
        pipeline_path: impl AsRef<Path>,
        segmentation_path: impl AsRef<Path>,
        embedding_path: impl AsRef<Path>,
        backend: BackendKind,
    ) -> Result<Self> {
        let pipeline = GgufFile::open(pipeline_path.as_ref())?;
        let config = PyannoteSpeakerDiarization31Config::from_gguf(&pipeline)?;
        let segmentation = PyanNet::from_gguf_with_backend(segmentation_path.as_ref(), backend)?;
        let actual_frames = segmentation.num_frames(SEGMENTATION_WINDOW_SAMPLES);
        if actual_frames != SEGMENTATION_FRAMES {
            return Err(VokraError::ModelLoad(format!(
                "{PIPELINE_NAME}: segmentation dependency emits {actual_frames} frames for a five-second window, expected {SEGMENTATION_FRAMES}"
            )));
        }
        let embedding = WeSpeaker::from_path_with_backend(embedding_path.as_ref(), backend)?;
        Ok(Self {
            config,
            segmentation,
            embedding,
            backend,
        })
    }

    /// Strictly bound public pipeline configuration.
    #[must_use]
    pub const fn config(&self) -> &PyannoteSpeakerDiarization31Config {
        &self.config
    }

    /// Backend shared by both learned dependency models.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Runs the complete default speaker-diarization 3.1 pipeline.
    pub fn diarize(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<DiarizationSegment>> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "{PIPELINE_NAME}: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resample offline first"
            )));
        }
        if let Some((index, value)) = pcm
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(format!(
                "{PIPELINE_NAME}: pcm[{index}] is non-finite ({value})"
            )));
        }
        if pcm.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        for start_sample in chunk_starts(pcm.len()) {
            let window = padded_window(pcm, start_sample);
            let probabilities = self.segmentation.segment(&window)?;
            let activity = decode_chunk_activity(&probabilities)?;
            chunks.push(SegmentationChunk {
                start_sample,
                activity,
            });
        }

        let speaker_count = aggregate_speaker_count(&chunks);
        if speaker_count.iter().all(|count| *count == 0) {
            return Ok(Vec::new());
        }

        // pyannote computes an embedding for every (chunk, local-speaker)
        // pair, then filters inactive rows only for clustering. It later
        // assigns all rows to the nearest learned centroid before marking
        // inactive lanes as -2 for reconstruction.
        let min_clean_frames =
            (SEGMENTATION_FRAMES * WESPEAKER_MIN_SAMPLES).div_ceil(SEGMENTATION_WINDOW_SAMPLES);
        let mut all_embeddings = Vec::with_capacity(chunks.len() * LOCAL_SPEAKERS);
        let mut training_indices = Vec::new();
        for chunk in &chunks {
            let window = padded_window(pcm, chunk.start_sample);
            let clean_frames: Vec<bool> = chunk
                .activity
                .iter()
                .map(|frame| frame.iter().filter(|active| **active > 0.0).count() < 2)
                .collect();
            for speaker in 0..LOCAL_SPEAKERS {
                let mask: Vec<f32> = chunk.activity.iter().map(|frame| frame[speaker]).collect();
                let clean_mask: Vec<f32> = mask
                    .iter()
                    .zip(clean_frames.iter())
                    .map(|(active, clean)| if *clean { *active } else { 0.0 })
                    .collect();
                let clean_count = clean_mask.iter().filter(|value| **value > 0.0).count();
                let used_mask =
                    if self.config.embedding_exclude_overlap && clean_count > min_clean_frames {
                        clean_mask.as_slice()
                    } else {
                        mask.as_slice()
                    };
                let embedding = self
                    .embedding
                    .embed_pcm_masked(&window, SAMPLE_RATE, used_mask)?;
                validate_embedding(&embedding, all_embeddings.len())?;
                if mask.iter().any(|value| *value > 0.0) {
                    training_indices.push(all_embeddings.len());
                }
                all_embeddings.push(embedding);
            }
        }

        if training_indices.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "{PIPELINE_NAME}: non-zero aggregated speaker count produced no active embedding rows"
            )));
        }
        let training_embeddings: Vec<Vec<f32>> = training_indices
            .iter()
            .map(|index| all_embeddings[*index].clone())
            .collect();
        let clusterer = AgglomerativeClustering {
            threshold: self.config.clustering_threshold,
            metric: DistanceMetric::Cosine,
            linkage: LinkageMethod::Centroid,
        };
        let training_labels = clusterer.cluster_with_min_cluster_size(
            &training_embeddings,
            self.config.clustering_min_cluster_size as usize,
        );
        let centroids = cluster_centroids(&training_embeddings, &training_labels)?;
        let all_labels = assign_to_centroids(&all_embeddings, &centroids)?;

        let mut hard_clusters = Vec::with_capacity(chunks.len() * LOCAL_SPEAKERS);
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            for speaker in 0..LOCAL_SPEAKERS {
                let active = chunk.activity.iter().any(|frame| frame[speaker] > 0.0);
                hard_clusters
                    .push(active.then_some(all_labels[chunk_index * LOCAL_SPEAKERS + speaker]));
            }
        }

        reconstruct(&chunks, &hard_clusters, &speaker_count, centroids.len())
    }

    /// Runs [`Self::diarize`] and renders a standard RTTM body.
    pub fn diarize_to_rttm(&self, pcm: &[f32], sample_rate: u32, file_id: &str) -> Result<String> {
        Ok(write_rttm(file_id, &self.diarize(pcm, sample_rate)?))
    }
}

#[derive(Debug, Clone)]
struct SegmentationChunk {
    start_sample: usize,
    activity: Vec<[f32; LOCAL_SPEAKERS]>,
}

fn chunk_starts(num_samples: usize) -> Vec<usize> {
    if num_samples < SEGMENTATION_WINDOW_SAMPLES {
        return vec![0];
    }
    let complete = (num_samples - SEGMENTATION_WINDOW_SAMPLES) / SEGMENTATION_STEP_SAMPLES + 1;
    let mut starts: Vec<usize> = (0..complete)
        .map(|chunk| chunk * SEGMENTATION_STEP_SAMPLES)
        .collect();
    if (num_samples - SEGMENTATION_WINDOW_SAMPLES) % SEGMENTATION_STEP_SAMPLES > 0 {
        starts.push(complete * SEGMENTATION_STEP_SAMPLES);
    }
    starts
}

fn padded_window(pcm: &[f32], start_sample: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; SEGMENTATION_WINDOW_SAMPLES];
    if start_sample < pcm.len() {
        let available = (pcm.len() - start_sample).min(SEGMENTATION_WINDOW_SAMPLES);
        window[..available].copy_from_slice(&pcm[start_sample..start_sample + available]);
    }
    window
}

fn decode_chunk_activity(probabilities: &[Vec<f32>]) -> Result<Vec<[f32; LOCAL_SPEAKERS]>> {
    if probabilities.len() != SEGMENTATION_FRAMES {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: PyanNet returned {} rows for a five-second window, expected {SEGMENTATION_FRAMES}",
            probabilities.len()
        )));
    }
    let mut activity = Vec::with_capacity(SEGMENTATION_FRAMES);
    for (frame, row) in probabilities.iter().enumerate() {
        if row.len() != POWERSET_MAPPING_3SPK_2OVERLAP.len() {
            return Err(VokraError::InvalidArgument(format!(
                "{PIPELINE_NAME}: PyanNet frame {frame} has {} powerset classes, expected {}",
                row.len(),
                POWERSET_MAPPING_3SPK_2OVERLAP.len()
            )));
        }
        if let Some((class, value)) = row
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(format!(
                "{PIPELINE_NAME}: PyanNet frame {frame} class {class} is non-finite ({value})"
            )));
        }
        // `torch.argmax` selects the first occurrence on equal values.
        let mut class = 0usize;
        for (index, value) in row.iter().copied().enumerate().skip(1) {
            if value > row[class] {
                class = index;
            }
        }
        let mapping = POWERSET_MAPPING_3SPK_2OVERLAP[class];
        activity.push(mapping.map(|value| f32::from(value)));
    }
    Ok(activity)
}

fn validate_embedding(embedding: &[f32], index: usize) -> Result<()> {
    if embedding.len() != EMBED_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: embedding row {index} has {} values, expected {EMBED_DIM}",
            embedding.len()
        )));
    }
    if let Some((dimension, value)) = embedding
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: embedding row {index} dimension {dimension} is non-finite ({value})"
        )));
    }
    Ok(())
}

fn cluster_centroids(embeddings: &[Vec<f32>], labels: &[usize]) -> Result<Vec<Vec<f32>>> {
    if embeddings.len() != labels.len() || embeddings.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: clustering returned {} labels for {} active embeddings",
            labels.len(),
            embeddings.len()
        )));
    }
    let num_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut sums = vec![vec![0.0f64; EMBED_DIM]; num_clusters];
    let mut counts = vec![0usize; num_clusters];
    for (embedding, label) in embeddings.iter().zip(labels.iter().copied()) {
        counts[label] += 1;
        for (sum, value) in sums[label].iter_mut().zip(embedding.iter()) {
            *sum += f64::from(*value);
        }
    }
    let centroids = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            let scale = (count as f64).recip();
            sum.into_iter()
                .map(|value| (value * scale) as f32)
                .collect()
        })
        .collect();
    Ok(centroids)
}

fn assign_to_centroids(embeddings: &[Vec<f32>], centroids: &[Vec<f32>]) -> Result<Vec<usize>> {
    if centroids.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: cannot assign embeddings without a cluster centroid"
        )));
    }
    let mut labels = Vec::with_capacity(embeddings.len());
    for embedding in embeddings {
        let mut best_label = 0usize;
        let mut best_distance = f32::INFINITY;
        for (label, centroid) in centroids.iter().enumerate() {
            let distance = cosine_distance(embedding, centroid)?;
            if distance < best_distance {
                best_distance = distance;
                best_label = label;
            }
        }
        labels.push(best_label);
    }
    Ok(labels)
}

fn cosine_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: cosine dimensions differ ({} vs {})",
            left.len(),
            right.len()
        )));
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (left, right) in left.iter().zip(right.iter()) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: cosine assignment received a zero-norm embedding"
        )));
    }
    Ok((1.0 - (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0)) as f32)
}

fn aggregate_speaker_count(chunks: &[SegmentationChunk]) -> Vec<usize> {
    let Some(last) = chunks.last() else {
        return Vec::new();
    };
    let global_frames = closest_frame(last.start_sample + SEGMENTATION_WINDOW_SAMPLES) + 1;
    let mut sums = vec![0usize; global_frames];
    let mut contributions = vec![0usize; global_frames];
    for chunk in chunks {
        let start = closest_frame(chunk.start_sample);
        for (frame, activity) in chunk.activity.iter().enumerate() {
            let global = start + frame;
            sums[global] += activity.iter().filter(|value| **value > 0.0).count();
            contributions[global] += 1;
        }
    }
    sums.into_iter()
        .zip(contributions)
        .map(|(sum, count)| {
            if count == 0 {
                0
            } else {
                round_ratio_ties_even(sum as i128, count as i128) as usize
            }
        })
        .collect()
}

fn reconstruct(
    chunks: &[SegmentationChunk],
    hard_clusters: &[Option<usize>],
    speaker_count: &[usize],
    num_clusters: usize,
) -> Result<Vec<DiarizationSegment>> {
    if hard_clusters.len() != chunks.len() * LOCAL_SPEAKERS {
        return Err(VokraError::InvalidArgument(format!(
            "{PIPELINE_NAME}: reconstruction has {} cluster rows for {} chunk-speaker pairs",
            hard_clusters.len(),
            chunks.len() * LOCAL_SPEAKERS
        )));
    }
    let output_speakers = num_clusters.max(speaker_count.iter().copied().max().unwrap_or(0));
    if output_speakers == 0 || speaker_count.is_empty() {
        return Ok(Vec::new());
    }
    let mut activations = vec![0.0f32; speaker_count.len() * output_speakers];
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let mut clustered = vec![0.0f32; SEGMENTATION_FRAMES * num_clusters];
        for speaker in 0..LOCAL_SPEAKERS {
            let Some(cluster) = hard_clusters[chunk_index * LOCAL_SPEAKERS + speaker] else {
                continue;
            };
            for (frame, activity) in chunk.activity.iter().enumerate() {
                let slot = frame * num_clusters + cluster;
                clustered[slot] = clustered[slot].max(activity[speaker]);
            }
        }
        let start = closest_frame(chunk.start_sample);
        for frame in 0..SEGMENTATION_FRAMES {
            for cluster in 0..num_clusters {
                activations[(start + frame) * output_speakers + cluster] +=
                    clustered[frame * num_clusters + cluster];
            }
        }
    }

    let mut binary = vec![false; speaker_count.len() * output_speakers];
    for (frame, count) in speaker_count.iter().copied().enumerate() {
        let mut speakers: Vec<usize> = (0..output_speakers).collect();
        speakers.sort_by(|left, right| {
            activations[frame * output_speakers + *right]
                .total_cmp(&activations[frame * output_speakers + *left])
                .then_with(|| left.cmp(right))
        });
        for speaker in speakers.into_iter().take(count.min(output_speakers)) {
            binary[frame * output_speakers + speaker] = true;
        }
    }
    Ok(binary_to_segments(
        &binary,
        speaker_count.len(),
        output_speakers,
    ))
}

fn binary_to_segments(binary: &[bool], frames: usize, speakers: usize) -> Vec<DiarizationSegment> {
    if frames < 2 || speakers == 0 {
        return Vec::new();
    }
    let frame_step = 5.0f32 / SEGMENTATION_FRAMES as f32;
    let midpoint = |frame: usize| (frame as f32 + 0.5) * frame_step;
    let mut segments = Vec::new();
    for speaker in 0..speakers {
        let mut active = binary[speaker];
        let mut start = midpoint(0);
        for frame in 1..frames {
            let value = binary[frame * speakers + speaker];
            if active && !value {
                let end = midpoint(frame);
                if end > start {
                    segments.push(DiarizationSegment {
                        start_s: start,
                        duration_s: end - start,
                        speaker_id: speaker,
                    });
                }
                start = end;
                active = false;
            } else if !active && value {
                start = midpoint(frame);
                active = true;
            }
        }
        if active {
            let end = midpoint(frames - 1);
            if end > start {
                segments.push(DiarizationSegment {
                    start_s: start,
                    duration_s: end - start,
                    speaker_id: speaker,
                });
            }
        }
    }
    segments.sort_by(|left, right| {
        left.start_s
            .total_cmp(&right.start_s)
            .then_with(|| left.speaker_id.cmp(&right.speaker_id))
    });
    segments
}

/// `SlidingWindow.closest_frame` at the exact PyanNet output resolution.
fn closest_frame(sample: usize) -> usize {
    let numerator =
        2i128 * sample as i128 * SEGMENTATION_FRAMES as i128 - SEGMENTATION_WINDOW_SAMPLES as i128;
    let denominator = 2i128 * SEGMENTATION_WINDOW_SAMPLES as i128;
    round_ratio_ties_even(numerator, denominator).max(0) as usize
}

fn round_ratio_ties_even(numerator: i128, denominator: i128) -> i128 {
    debug_assert!(denominator > 0);
    if numerator < 0 {
        return -round_ratio_ties_even(-numerator, denominator);
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    match (2 * remainder).cmp(&denominator) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal if quotient % 2 == 0 => quotient,
        std::cmp::Ordering::Equal => quotient + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_chunk(start_sample: usize) -> SegmentationChunk {
        SegmentationChunk {
            start_sample,
            activity: vec![[0.0; LOCAL_SPEAKERS]; SEGMENTATION_FRAMES],
        }
    }

    #[test]
    fn chunk_starts_match_inference_slide_padding_contract() {
        assert_eq!(chunk_starts(1), vec![0]);
        assert_eq!(chunk_starts(SEGMENTATION_WINDOW_SAMPLES), vec![0]);
        assert_eq!(
            chunk_starts(SEGMENTATION_WINDOW_SAMPLES + 1),
            vec![0, SEGMENTATION_STEP_SAMPLES]
        );
        assert_eq!(
            chunk_starts(SEGMENTATION_WINDOW_SAMPLES + SEGMENTATION_STEP_SAMPLES),
            vec![0, SEGMENTATION_STEP_SAMPLES]
        );
    }

    #[test]
    fn closest_frame_matches_pyannote_sliding_window_geometry() {
        assert_eq!(closest_frame(0), 0);
        assert_eq!(closest_frame(SEGMENTATION_STEP_SAMPLES), 29);
        assert_eq!(closest_frame(2 * SEGMENTATION_STEP_SAMPLES), 58);
        assert_eq!(closest_frame(SEGMENTATION_WINDOW_SAMPLES), 292);
    }

    #[test]
    fn powerset_argmax_uses_first_class_on_equal_logits() {
        let mut probabilities =
            vec![vec![0.0f32; POWERSET_MAPPING_3SPK_2OVERLAP.len()]; SEGMENTATION_FRAMES];
        probabilities[0][0] = 1.0;
        probabilities[0][1] = 1.0;
        let activity = decode_chunk_activity(&probabilities).unwrap();
        assert_eq!(activity[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn one_chunk_count_has_exact_293_frame_extent() {
        let mut chunk = silent_chunk(0);
        chunk.activity[10] = [1.0, 1.0, 0.0];
        let count = aggregate_speaker_count(&[chunk]);
        assert_eq!(count.len(), SEGMENTATION_FRAMES);
        assert_eq!(count[10], 2);
        assert_eq!(count.iter().sum::<usize>(), 2);
    }

    #[test]
    fn overlapping_chunk_counts_are_averaged_then_ties_even_rounded() {
        let mut first = silent_chunk(0);
        let mut second = silent_chunk(SEGMENTATION_STEP_SAMPLES);
        first.activity[29] = [1.0, 0.0, 0.0];
        second.activity[0] = [1.0, 1.0, 0.0];
        let count = aggregate_speaker_count(&[first, second]);
        assert_eq!(count.len(), 323);
        // (1 + 2) / 2 = 1.5 -> Python/NumPy ties-to-even = 2.
        assert_eq!(count[29], 2);
    }

    #[test]
    fn centroid_assignment_is_cosine_and_first_wins_ties() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            assign_to_centroids(&embeddings, &centroids).unwrap(),
            vec![0, 1, 0]
        );
    }

    #[test]
    fn binary_conversion_uses_frame_midpoints_like_upstream_binarize() {
        let mut binary = vec![false; 5];
        binary[1] = true;
        binary[2] = true;
        let segments = binary_to_segments(&binary, 5, 1);
        assert_eq!(segments.len(), 1);
        let step = 5.0 / SEGMENTATION_FRAMES as f32;
        assert!((segments[0].start_s - 1.5 * step).abs() < 1.0e-7);
        assert!((segments[0].duration_s - 2.0 * step).abs() < 1.0e-7);
    }

    #[test]
    fn reconstruction_takes_top_count_cluster_activations() {
        let mut chunk = silent_chunk(0);
        chunk.activity[10] = [1.0, 1.0, 0.0];
        chunk.activity[11] = [1.0, 0.0, 0.0];
        let mut count = vec![0usize; SEGMENTATION_FRAMES];
        count[10] = 2;
        count[11] = 1;
        let clusters = [Some(0), Some(1), None];
        let segments = reconstruct(&[chunk], &clusters, &count, 2).unwrap();
        assert!(segments.iter().any(|segment| segment.speaker_id == 0));
        assert!(segments.iter().any(|segment| segment.speaker_id == 1));
    }
}
