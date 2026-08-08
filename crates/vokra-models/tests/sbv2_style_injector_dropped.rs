//! STYLE-INJECTOR fix (2026-08-09) regression pin.
//!
//! Verifies that `SbV2Model::synthesize` no longer mixes `req.style_vec`
//! into the text-hidden path even when the model carries a
//! **non-identity** `StyleVectorInjector` — the Python reference
//! (`sbv2_dump_reference.py` step 9) explicitly does NOT mix style
//! into `text_hidden` on the base-checkpoint path. Pre-fix Vokra called
//! `self.style_injector.inject(&mut hidden_for_flow, phoneme_count,
//! &req.style_vec)` unconditionally, a latent parity risk that
//! base-ckpt tests missed because the base ckpt ships all-zero style
//! projections (making `inject` an arithmetic identity).
//!
//! # Oracle
//!
//! Build a synthetic SBV2 model whose `StyleVectorInjector` has
//! **non-zero** `proj_scale` + `proj_bias` weights (unlike
//! `SbV2Model::synthetic_for_test`'s zero projections). Then run
//! `synthesize` twice with the SAME request except for `style_vec`:
//! one call with the identity all-zero `style_vec`, one call with a
//! distinctive non-zero `style_vec`.
//!
//! * Post-fix (this repo): outputs are **byte-identical** — style_vec
//!   is not consumed by the pipeline.
//! * Pre-fix: outputs would diverge — the injector's non-zero
//!   projections × non-zero style_vec produce a non-identity mix into
//!   the flow input.

use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::deberta_v3::DebertaV3Encoder;
use vokra_bert::tokenizer::SbertTokenizer;
use vokra_core::ir::graph::{HifiGanAttrs, ResBlockType};
use vokra_models::sbv2::{
    BertBridge, Language, RngMode, SbV2BertContainer, SbV2Decoder, SbV2Flow, SbV2Model,
    SbV2Phonemizer, SbV2SDP, SbV2SynthRequest, SbV2TextEncoder, SpeakerEmbedding,
    StyleVectorInjector,
};
use vokra_ops::{
    HifiGanConfig, HifiGanWeights, MrfBranchWeights, ResBlockLayer, UpsampleStageWeights,
};

/// Build a model whose `StyleVectorInjector` has **non-zero**
/// projections. Every other component mirrors
/// `SbV2Model::synthetic_for_test` except for tiny shape tweaks so the
/// custom model still runs end-to-end. `style_vec` non-identity ⇒
/// pre-fix `inject(..)` observably mutates `hidden_for_flow`; post-fix
/// the same `synthesize` bypasses `inject(..)` so output is invariant
/// under `style_vec`.
fn build_model_with_nonzero_style_injector() -> SbV2Model {
    const D_MODEL: usize = 8;
    const D_BERT: usize = 8;
    const D_STYLE: usize = 4;
    const N_VOCAB: usize = 256;
    const N_TONES: usize = 3;
    const N_SPEAKERS: usize = 2;

    let phonemizer = SbV2Phonemizer::synthetic_for_test();

    let text_encoder = SbV2TextEncoder::from_weights(
        (0..N_VOCAB * D_MODEL)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect(),
        (0..N_TONES * D_MODEL)
            .map(|i| ((i as f32) * 0.01).cos() * 0.02)
            .collect(),
        vec![0.0; 3 * D_MODEL], // language_embed [N_LANGUAGES=3, D_MODEL] all-zero (identity)
        Vec::new(),             // empty transformer stack
        D_MODEL,
        N_VOCAB,
        N_TONES,
    );

    let tokenizer_pieces = vec![
        ("<pad>".to_string(), 0.0),
        ("<unk>".to_string(), 0.0),
        ("<s>".to_string(), 0.0),
        ("</s>".to_string(), 0.0),
    ];
    let bert = SbV2BertContainer {
        ja_tokenizer: SbertTokenizer::from_pieces_for_test(tokenizer_pieces.clone()),
        en_tokenizer: SbertTokenizer::from_pieces_for_test(tokenizer_pieces),
        ja: DebertaV2Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
        en: DebertaV3Encoder::synthetic_for_test(2, D_BERT, 2, 16, 512),
    };

    let bert_bridge = BertBridge::from_conv(
        (0..D_MODEL * D_BERT)
            .map(|i| ((i as f32) * 0.02).sin() * 0.03)
            .collect(),
        vec![0.0; D_MODEL],
        D_BERT,
        D_MODEL,
    );

    let speaker_embed = SpeakerEmbedding::from_table(
        (0..N_SPEAKERS * D_MODEL)
            .map(|i| ((i as f32) * 0.05).cos() * 0.1)
            .collect(),
        N_SPEAKERS,
        D_MODEL,
    );

    // NON-ZERO style projections — the key point of this factory.
    // `synthetic_for_test` uses zero projections (identity injector);
    // here we deliberately supply meaningfully-non-zero weights so any
    // pre-fix `inject(&mut hidden_for_flow, ..)` call would observably
    // perturb the flow input.
    let style_injector = StyleVectorInjector::from_projections(
        (0..D_MODEL * D_STYLE)
            .map(|i| 0.1 + (i as f32) * 0.03)
            .collect(),
        (0..D_MODEL * D_STYLE)
            .map(|i| -0.05 + (i as f32) * 0.02)
            .collect(),
        D_STYLE,
        D_MODEL,
    );

    let sdp = SbV2SDP::empty(D_MODEL, D_MODEL);
    let flow = SbV2Flow::from_layers(Vec::new(), D_MODEL);

    let attrs = HifiGanAttrs {
        n_mels: D_MODEL,
        initial_channel: 6,
        upsample_rates: vec![2, 2],
        upsample_kernel_sizes: vec![4, 4],
        resblock_kernel_sizes: vec![3],
        resblock_dilation_sizes: vec![vec![1]],
        sample_rate: 44_100,
        leaky_relu_slope: 0.1,
        res_block_type: ResBlockType::V2,
    };
    let weights = tiny_hifigan_weights(&attrs);
    let sample_rate = attrs.sample_rate;
    let decoder = SbV2Decoder::new(weights, attrs, HifiGanConfig::fp32(), sample_rate);

    SbV2Model::new(
        phonemizer,
        text_encoder,
        bert,
        bert_bridge,
        speaker_embed,
        style_injector,
        sdp,
        flow,
        decoder,
    )
}

fn tiny_hifigan_weights(attrs: &HifiGanAttrs) -> HifiGanWeights {
    let conv_pre_kernel = 3;
    let conv_post_kernel = 3;
    let mut conv_pre_weight = Vec::new();
    for oc in 0..attrs.initial_channel {
        for ic in 0..attrs.n_mels {
            for k in 0..conv_pre_kernel {
                conv_pre_weight.push(((oc + ic + k) as f32 * 0.017).sin() * 0.05);
            }
        }
    }
    let conv_pre_bias: Vec<f32> = (0..attrs.initial_channel)
        .map(|i| (i as f32 * 0.05).cos() * 0.01)
        .collect();

    let mut in_ch = attrs.initial_channel;
    let mut upsample_weights = Vec::new();
    let mut mrf_stage_weights = Vec::new();
    for stage in 0..attrs.n_upsample_stages() {
        let out_ch = (in_ch / 2).max(3);
        let kernel = attrs.upsample_kernel_sizes[stage];
        let stride = attrs.upsample_rates[stage];
        let mut weight = Vec::new();
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for k in 0..kernel {
                    weight.push(((ic + oc + k + stage) as f32 * 0.023).sin() * 0.05);
                }
            }
        }
        let bias: Vec<f32> = (0..out_ch)
            .map(|i| ((i + stage) as f32 * 0.07).cos() * 0.01)
            .collect();
        upsample_weights.push(UpsampleStageWeights {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
        });
        let mut branches = Vec::new();
        for b in 0..attrs.n_mrf_branches() {
            let layers = attrs.resblock_dilation_sizes[b]
                .iter()
                .map(|dilation| {
                    let kernel = attrs.resblock_kernel_sizes[b];
                    let mut weight = Vec::new();
                    for oc in 0..out_ch {
                        for ic in 0..out_ch {
                            for k in 0..kernel {
                                weight.push(((oc + ic + k + dilation) as f32 * 0.031).sin() * 0.05);
                            }
                        }
                    }
                    let bias: Vec<f32> = (0..out_ch)
                        .map(|i| ((i + *dilation + b) as f32 * 0.11).cos() * 0.01)
                        .collect();
                    ResBlockLayer {
                        weight,
                        bias,
                        weight_c2: None,
                        bias_c2: None,
                        dilation: *dilation,
                        kernel,
                        channels: out_ch,
                    }
                })
                .collect();
            branches.push(MrfBranchWeights { layers });
        }
        mrf_stage_weights.push(branches);
        in_ch = out_ch;
    }
    let mut conv_post_weight = Vec::new();
    for ic in 0..in_ch {
        for k in 0..conv_post_kernel {
            conv_post_weight.push(((ic + k) as f32 * 0.019).sin() * 0.05);
        }
    }
    HifiGanWeights {
        conv_pre_weight,
        conv_pre_bias,
        conv_pre_kernel,
        upsample_weights,
        mrf_stage_weights,
        conv_post_weight,
        conv_post_bias: vec![0.0],
        conv_post_kernel,
    }
}

fn base_request(style_vec: Vec<f32>) -> SbV2SynthRequest {
    SbV2SynthRequest {
        text: "test".to_string(),
        language: Language::EN,
        speaker_id: 0,
        speaker_embedding: None,
        style_vec,
        speed: 1.0,
        noise_scale: 0.0,
        noise_scale_w: 0.0,
        seed: 42,
        rng_mode: RngMode::GaussianSplitMix64Legacy,
    }
}

#[test]
fn synthesize_output_is_invariant_under_style_vec_post_fix() {
    let model = build_model_with_nonzero_style_injector();

    // Identity style (all zeros): every style-projection call is a
    // structural no-op regardless of whether `inject(..)` is called
    // or dropped.
    let zero_style = vec![0.0_f32; 4];
    // Distinctive non-zero style. With the model's non-zero
    // `proj_scale` + `proj_bias`, pre-fix `inject(..)` would produce
    // a MEANINGFUL mutation of `hidden_for_flow` (`h * (1 + scale) +
    // bias` with both `scale` and `bias` non-zero). Post-fix the
    // same `synthesize` bypasses `inject(..)` so the output is
    // insensitive to `style_vec`.
    let nonzero_style = vec![0.7_f32, -0.3, 0.5, 0.1];

    let out_zero = model
        .synthesize(&base_request(zero_style))
        .expect("synthesize(zero style) should succeed");
    let out_nonzero = model
        .synthesize(&base_request(nonzero_style))
        .expect("synthesize(nonzero style) should succeed");

    assert_eq!(
        out_zero.sample_rate, out_nonzero.sample_rate,
        "sample rate must not depend on style_vec"
    );
    assert_eq!(
        out_zero.samples.len(),
        out_nonzero.samples.len(),
        "sample count must not depend on style_vec (durations are text-driven)"
    );
    // Byte-identical: post-fix `req.style_vec` does not enter the
    // forward math at all, so the two runs are literally the same
    // pipeline output.
    let max_delta = out_zero
        .samples
        .iter()
        .zip(out_nonzero.samples.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_delta == 0.0,
        "STYLE-INJECTOR fix: `synthesize` must not mix `req.style_vec` into the pipeline. \
         Under this fixture's non-zero style projections, pre-fix output would diverge on \
         `style_vec = nonzero_style`; post-fix outputs are byte-identical. Observed \
         max|Δ| = {max_delta}"
    );
}
