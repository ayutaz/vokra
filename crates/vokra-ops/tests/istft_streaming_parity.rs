//! Streaming iSTFT parity: `istft_streaming` == batch `istft` (M2-05-T07).
//!
//! The WP completion condition (milestones.md §6 M2-05): "the streaming output
//! and the one-shot `istft` output agree". This drives that check through the
//! **public** `vokra-ops` API (`IstftStreamingState` / `istft_streaming_oneshot`
//! / IR dispatch), across every chunk split and the full attribute space, and
//! reports the observed `max|Δ|`.
//!
//! Oracle policy (numerical-parity skill): the trusted reference is the batch
//! `istft`, itself parity-tested against `torch.istft` in the M0-04 fixtures
//! (`tests/parity/fixtures/m0-04`). Streaming inherits that PyTorch parity
//! transitively by matching the batch op **bit-for-bit** (`max|Δ| == 0`), a
//! stronger bar than the FP32 `atol = 0.01` budget (NFR-QL-01). No fabricated
//! numbers: the reference is computed in-process by the batch op, not hardcoded.

use vokra_core::OpKind;
use vokra_ops::attrs::{IstftAttrs, IstftStreamingAttrs, Normalization, StftAttrs, Window};
use vokra_ops::{
    IstftStreamingState, OpValue, Spectrogram, dispatch, istft, istft_streaming_oneshot,
};

/// Analyses a synthetic multi-tone signal into a test spectrogram.
fn analyse(
    n_fft: usize,
    hop: usize,
    real_input: bool,
    window: Window,
    center: bool,
) -> Spectrogram {
    let signal: Vec<f32> = (0..5000)
        .map(|t| {
            let t = t as f32;
            (t * 0.017).sin() + 0.4 * (t * 0.101).cos() + 0.08 * (t * 0.29).sin()
        })
        .collect();
    let mut sa = StftAttrs::new(n_fft, hop);
    sa.real_input = real_input;
    sa.window = window;
    sa.center = center;
    vokra_ops::stft(&signal, &sa).unwrap()
}

fn slice_frames(spec: &Spectrogram, start: usize, count: usize) -> Spectrogram {
    let b = spec.bins;
    Spectrogram {
        frames: count,
        bins: b,
        re: spec.re[start * b..(start + count) * b].to_vec(),
        im: spec.im[start * b..(start + count) * b].to_vec(),
    }
}

/// Streams `spec` through `chunk_sizes` (cycled), concatenating push outputs and
/// the final flush.
fn stream(spec: &Spectrogram, attrs: &IstftStreamingAttrs, chunk_sizes: &[usize]) -> Vec<f32> {
    let mut state = IstftStreamingState::new(attrs).unwrap();
    let mut out = Vec::new();
    let (mut f, mut ci) = (0usize, 0usize);
    while f < spec.frames {
        let take = chunk_sizes[ci % chunk_sizes.len()]
            .max(1)
            .min(spec.frames - f);
        ci += 1;
        out.extend(state.push(&slice_frames(spec, f, take)).unwrap());
        f += take;
    }
    out.extend(state.finish());
    out
}

/// `(max|Δ|, argmax index)` between two equal-length signals.
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > max {
            max = d;
            at = i;
        }
    }
    (max, at)
}

#[test]
fn streaming_matches_batch_over_full_attribute_space_and_chunk_splits() {
    let windows = [
        Window::Hann,
        Window::Hamming,
        Window::BlackmanHarris,
        Window::Kaiser { beta: 6.0 },
    ];
    let patterns: [&[usize]; 5] = [&[1], &[2, 3, 5, 7], &[4], &[16, 1, 9], &[usize::MAX]];
    let mut worst = 0.0f32;

    for &window in &windows {
        for &center in &[true, false] {
            for &real_input in &[true, false] {
                for &norm in &[
                    Normalization::Backward,
                    Normalization::Forward,
                    Normalization::Ortho,
                ] {
                    let spec = analyse(256, 64, real_input, window, center);
                    let mut ia = IstftAttrs::new(256, 64);
                    ia.window = window;
                    ia.center = center;
                    ia.real_input = real_input;
                    ia.normalization = norm;
                    let attrs = IstftStreamingAttrs::from_istft(ia.clone());
                    let expect = istft(&spec, &ia).unwrap();

                    for pattern in patterns {
                        let got = stream(&spec, &attrs, pattern);
                        let (d, at) = max_abs_diff(&got, &expect);
                        worst = worst.max(d);
                        assert_eq!(
                            d, 0.0,
                            "window={window:?} center={center} real={real_input} norm={norm:?} \
                             chunks={pattern:?}: max|Δ|={d} at sample {at}"
                        );
                    }
                }
            }
        }
    }
    // Bit-exact everywhere.
    assert_eq!(worst, 0.0, "overall streaming-vs-batch max|Δ| = {worst}");
}

#[test]
fn oneshot_and_graph_dispatch_agree_with_batch() {
    // The public one-shot helper and the IR dispatch path (a graph node) both
    // reproduce the batch istft — the tail state never crosses the op boundary.
    let spec = analyse(400, 100, true, Window::Hann, true);
    let mut ia = IstftAttrs::new(400, 100);
    ia.length = Some(5000);
    let attrs = IstftStreamingAttrs::from_istft(ia.clone());
    let expect = istft(&spec, &ia).unwrap();

    let oneshot = istft_streaming_oneshot(&spec, &attrs).unwrap();
    assert_eq!(max_abs_diff(&oneshot, &expect).0, 0.0, "oneshot vs batch");

    let out = dispatch(
        &OpKind::IstftStreaming(attrs),
        &[OpValue::Complex {
            shape: vec![spec.frames, spec.bins],
            re: spec.re.clone(),
            im: spec.im.clone(),
        }],
    )
    .unwrap();
    let (_, via_graph) = out[0].as_real().unwrap();
    assert_eq!(
        max_abs_diff(via_graph, &expect).0,
        0.0,
        "graph dispatch vs batch"
    );
}

#[test]
fn realistic_vocoder_hop_matches_batch() {
    // A vocoder-like configuration (n_fft=1024, hop=256, real half-spectrum —
    // the iSTFTNet / Vocos head layout) streamed frame-by-frame.
    let spec = analyse(1024, 256, true, Window::Hann, true);
    let ia = IstftAttrs::new(1024, 256);
    let attrs = IstftStreamingAttrs::from_istft(ia.clone());
    let expect = istft(&spec, &ia).unwrap();
    let got = stream(&spec, &attrs, &[1]);
    let (d, at) = max_abs_diff(&got, &expect);
    assert_eq!(d, 0.0, "vocoder-hop streaming max|Δ|={d} at {at}");
}
