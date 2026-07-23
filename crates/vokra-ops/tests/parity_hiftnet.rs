//! HiFTNet generator parity harness (SoTA plan Phase 1-2, Wave 4).
//!
//! # Reference oracle
//!
//! Real CosyVoice2 / Chatterbox HiFTNet checkpoints require owner sourcing
//! (the M3-06 / M3-09 consumer WPs coordinate that). Until they land the
//! harness runs the same **synthesized-weight**, deterministic pin the
//! upstream Wave 3c-3 in-crate tests use, but from the *integration* boundary
//! (`tests/` — no access to `#[cfg(test)]` helpers, so the small-shape
//! generator is re-declared here). What that proves:
//!
//! 1. **The generator pipeline is deterministic.** Same weights + same mel
//!    → same audio, bit-for-bit. Any hidden RNG on a `NsfEntropy::Deterministic`
//!    call surfaces immediately.
//! 2. **No shape drift between wired components.** Every weight tensor's
//!    length is a pure function of the config, and the audio length is a
//!    pure function of `t_mel × total_upsample_factor()`. A refactor that
//!    silently drops a component or mis-orders a Conv1d weight would blow
//!    up the shape or the value here.
//! 3. **Regression detection on the config surface.** The three tests below
//!    exercise (a) determinism, (b) input-sensitivity across mel seeds, and
//!    (c) length stability across `t_mel`. A silent hyperparameter change
//!    that alters the total-upsample factor, the source-fusion contract, or
//!    the ReflectionPad size trips the shape pin.
//!
//! # Flip-the-switch external reference
//!
//! The last test (`hift_gen_matches_external_reference_when_available`) is
//! the future-facing hook. When an owner produces a real checkpoint dump it
//! points the env-var [`VOKRA_HIFTNET_REFERENCE_DIR`] at a directory containing:
//!
//! - `weights.bin` — every f32 weight the Vokra port needs, concatenated in
//!   little-endian order using the exact layout [`load_weights_from_bytes`]
//!   consumes (documented at that function's docstring — same ordering as
//!   [`build_deterministic_hift_generator`]).
//! - `mel.bin` — the input mel, row-major `[in_channels, t_mel]` as raw f32
//!   LE. `t_mel` is derived from `mel.bin.len() / (4 * in_channels)`.
//! - `expected_audio.bin` — the upstream reference waveform, raw f32 LE.
//! - `config.env` (optional) — a text file with a single `expected_len=<N>`
//!   line to sanity-check the reference file's length before comparison.
//!
//! When the env var is unset the test emits a GitHub Actions `::warning::`
//! annotation and returns cleanly — the harness is deliberately no-op when
//! there is nothing to compare against, so CI stays green until an owner
//! flips the switch. Under-supplied fixtures (missing file, wrong length)
//! fail loudly. The atol is documented at the call site.
//!
//! [`VOKRA_HIFTNET_REFERENCE_DIR`]: #environment-variable-vokra_hiftnet_reference_dir

use vokra_ops::hiftnet::{
    F0PredictorWeights, HiFTGenerator, HiFTGeneratorConfig, HiFTGeneratorWeights, ResBlockWeights,
};

// ---------------------------------------------------------------------------
// Inline SplitMix64 — mirror of `crates/vokra-ops/src/nsf.rs::splitmix64`.
// ---------------------------------------------------------------------------
//
// The upstream helper is `pub(super)`-scoped inside `nsf.rs` so widening its
// visibility just for a test is over-reach; inlining the (well-known, fixed-
// constant) Vigna 2015 splitmix64 next-state avoids a public-API tremor.
// The two implementations must stay bit-identical: any change to one must be
// mirrored in the other, and the `splitmix64_is_deterministic_regression_pin`
// test in `nsf.rs` catches drift on that side.

/// SplitMix64 next-state (Vigna 2015). Fixed-width unsigned arithmetic makes
/// this reproducible across hosts.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw a single bounded f32 from the SplitMix64 stream.
///
/// The Wave 4 spec calls for `(splitmix64() & 0xFFFF) as f32 * 1e-5`, i.e.
/// values in `[0, 0.65535)`. The upper bound keeps every conv layer's output
/// bounded (weights × in_ch × kernel × input magnitude compounds, but the
/// terminal `exp(...).min(1e2)` on the magnitude spectrum plus the
/// `audio_limit = 0.99` clamp bound the audio regardless).
#[inline]
fn synth_f32(state: &mut u64) -> f32 {
    (splitmix64(state) & 0xFFFF) as f32 * 1e-5
}

/// Fill a length-`n` `Vec<f32>` from the stream.
fn synth_vec(state: &mut u64, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(synth_f32(state));
    }
    out
}

// ---------------------------------------------------------------------------
// Small-shape config — the same knobs Wave 3c-3's `small_hift_generator_bundle`
// uses. Re-declared here because the in-crate helper is `#[cfg(test)]`-scoped
// and integration tests do not see it. Kept in a single builder so the four
// tests below share one source of truth.
// ---------------------------------------------------------------------------

/// Shape crib (all row-major, derived from the config):
///
/// * `in_channels = 4, base_channels = 8, nb_harmonics = 2`
/// * `upsample_rates = [2, 2], upsample_kernel_sizes = [4, 4]`
/// * `istft_n_fft = 8, istft_hop_len = 2` — so `n_fft + 2 = 10` and
///   `total_upsample_factor = 2 * 2 * 2 = 8`
/// * `output_channels_at(0) = 4`, `output_channels_at(1) = 2`
/// * `downsample_us = [2, 1]` → source_downs stage 0: `k=4 stride=2 pad=1`,
///   stage 1: `k=1 stride=1 pad=0`
/// * `resblock_kernel_sizes = [3]`, `resblock_dilation_sizes = [[1]]`
/// * `source_resblock_kernel_sizes = [3, 3]`,
///   `source_resblock_dilation_sizes = [[1], [1]]`
fn small_hift_config() -> HiFTGeneratorConfig {
    HiFTGeneratorConfig {
        in_channels: 4,
        base_channels: 8,
        nb_harmonics: 2,
        sampling_rate: 16000,
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
    }
}

/// Synthesize one `ResBlockWeights` bundle for a given `(channels, kernel,
/// n_branches)` layout. Every weight cell is drawn from `state` in strict
/// per-branch order (`convs1_w`, `convs1_b`, `convs2_w`, `convs2_b`,
/// `activations1_alpha`, `activations2_alpha`) so the flip-the-switch
/// external-reference layout can reproduce the same walk.
fn synth_res_block(
    state: &mut u64,
    channels: usize,
    kernel: usize,
    n_branches: usize,
) -> ResBlockWeights {
    let mut convs1_w = Vec::with_capacity(n_branches);
    let mut convs1_b = Vec::with_capacity(n_branches);
    let mut convs2_w = Vec::with_capacity(n_branches);
    let mut convs2_b = Vec::with_capacity(n_branches);
    let mut activations1_alpha = Vec::with_capacity(n_branches);
    let mut activations2_alpha = Vec::with_capacity(n_branches);
    for _ in 0..n_branches {
        convs1_w.push(synth_vec(state, channels * channels * kernel));
        convs1_b.push(synth_vec(state, channels));
        convs2_w.push(synth_vec(state, channels * channels * kernel));
        convs2_b.push(synth_vec(state, channels));
        activations1_alpha.push(synth_vec(state, channels));
        activations2_alpha.push(synth_vec(state, channels));
    }
    ResBlockWeights {
        convs1_w,
        convs1_b,
        convs2_w,
        convs2_b,
        activations1_alpha,
        activations2_alpha,
    }
}

/// Build a fully-synthesized `HiFTGeneratorWeights` bundle from a running
/// SplitMix64 state, in the strict order documented below. Used by both the
/// deterministic harness ([`build_deterministic_hift_generator`]) and the
/// flip-the-switch fixture loader ([`load_weights_from_bytes`]) so a future
/// owner-produced `weights.bin` can pack the tensors in the same walk.
///
/// Walk (each entry is a single `synth_vec` / `synth_f32` call):
///
/// 1. `conv_pre_w`, `conv_pre_b`
/// 2. Per stage `i in 0..num_upsamples`:
///    a. `ups_w[i]`, `ups_b[i]`
/// 3. Per stage `i in 0..num_upsamples`:
///    a. `source_downs_w[i]`, `source_downs_b[i]`
/// 4. Per stage `i in 0..num_upsamples`:
///    a. `source_resblock_weights[i]` (nested per-branch walk — see
///    [`synth_res_block`])
/// 5. Row-major `resblock_weights[i * num_kernels + j]` for
///    `i in 0..num_upsamples, j in 0..num_kernels`
/// 6. `conv_post_w`, `conv_post_b`
/// 7. `m_source_linear_w` (`nb_harmonics + 1` scalars), `m_source_linear_b`
///    (single scalar)
/// 8. F0 predictor:
///    a. Layer 0 conv weight `[base_channels, in_channels, 3]`, bias
///    `[base_channels]`
///    b. Layers 1..5 conv weight `[base_channels, base_channels, 3]`, bias
///    `[base_channels]`
///    c. `linear_w[base_channels]`, `linear_b[1]`
fn synth_generator_weights(state: &mut u64, cfg: &HiFTGeneratorConfig) -> HiFTGeneratorWeights {
    let n_ups = cfg.num_upsamples();
    let n_kernels = cfg.num_kernels();
    let bc = cfg.base_channels as usize;
    let inc = cfg.in_channels as usize;
    let n_fft_plus_2 = cfg.istft_n_fft as usize + 2;

    // ---- (1) conv_pre --------------------------------------------------
    let conv_pre_w = synth_vec(state, bc * inc * 7);
    let conv_pre_b = synth_vec(state, bc);

    // ---- (2) ups (ConvTranspose1d in-channel leading) ------------------
    let mut ups_w = Vec::with_capacity(n_ups);
    let mut ups_b = Vec::with_capacity(n_ups);
    for i in 0..n_ups {
        let in_ch = bc >> i;
        let out_ch = bc >> (i + 1);
        let k = cfg.upsample_kernel_sizes[i] as usize;
        ups_w.push(synth_vec(state, in_ch * out_ch * k));
        ups_b.push(synth_vec(state, out_ch));
    }

    // ---- (3) source_downs ---------------------------------------------
    //
    // downsample_us mirrors the [1] + reverse(upsample_rates[:-1]) → cumsum
    // → reverse pipeline the port derives in `HiFTGenerator::new` — writing
    // it out here keeps the flip-the-switch layout self-explanatory rather
    // than making the reader chase the derivation.
    let mut downsample_rates: Vec<u32> = Vec::with_capacity(n_ups);
    downsample_rates.push(1);
    for i in (0..n_ups - 1).rev() {
        downsample_rates.push(cfg.upsample_rates[i]);
    }
    let mut downsample_cum: Vec<u32> = Vec::with_capacity(n_ups);
    let mut acc: u32 = 1;
    for &r in &downsample_rates {
        acc = acc.saturating_mul(r);
        downsample_cum.push(acc);
    }
    let downsample_us: Vec<u32> = downsample_cum.iter().rev().copied().collect();
    let mut source_downs_w = Vec::with_capacity(n_ups);
    let mut source_downs_b = Vec::with_capacity(n_ups);
    for (i, &u) in downsample_us.iter().enumerate() {
        let out_ch = cfg.output_channels_at(i) as usize;
        let k = if u == 1 { 1 } else { (u * 2) as usize };
        source_downs_w.push(synth_vec(state, out_ch * n_fft_plus_2 * k));
        source_downs_b.push(synth_vec(state, out_ch));
    }

    // ---- (4) source_resblocks -----------------------------------------
    let mut source_resblock_weights = Vec::with_capacity(n_ups);
    for i in 0..n_ups {
        let ch = cfg.output_channels_at(i) as usize;
        let k = cfg.source_resblock_kernel_sizes[i] as usize;
        let n_branches = cfg.source_resblock_dilation_sizes[i].len();
        source_resblock_weights.push(synth_res_block(state, ch, k, n_branches));
    }

    // ---- (5) resblocks (row-major over [i * num_kernels + j]) ---------
    let mut resblock_weights = Vec::with_capacity(n_ups * n_kernels);
    for i in 0..n_ups {
        let ch = cfg.output_channels_at(i) as usize;
        for j in 0..n_kernels {
            let k = cfg.resblock_kernel_sizes[j] as usize;
            let n_branches = cfg.resblock_dilation_sizes[j].len();
            resblock_weights.push(synth_res_block(state, ch, k, n_branches));
        }
    }

    // ---- (6) conv_post ------------------------------------------------
    let final_ch = cfg.output_channels_at(n_ups - 1) as usize;
    let conv_post_w = synth_vec(state, n_fft_plus_2 * final_ch * 7);
    let conv_post_b = synth_vec(state, n_fft_plus_2);

    // ---- (7) source module linear head --------------------------------
    let h1 = (cfg.nb_harmonics + 1) as usize;
    let m_source_linear_w = synth_vec(state, h1);
    let m_source_linear_b = synth_f32(state);

    // ---- (8) F0 predictor (fixed 5-layer 3-kernel stack) --------------
    // Wave 3c-2 wires the F0 predictor to `cond_channels = base_channels`
    // and `num_layers = 5` regardless of the outer config, so the walk is
    // hard-pinned to those constants here.
    let mut f0_conv_weights = Vec::with_capacity(5);
    let mut f0_conv_biases = Vec::with_capacity(5);
    f0_conv_weights.push(synth_vec(state, bc * inc * 3));
    f0_conv_biases.push(synth_vec(state, bc));
    for _ in 1..5 {
        f0_conv_weights.push(synth_vec(state, bc * bc * 3));
        f0_conv_biases.push(synth_vec(state, bc));
    }
    let f0_linear_w = synth_vec(state, bc);
    let f0_linear_b = synth_vec(state, 1);
    let f0_predictor_weights = F0PredictorWeights {
        conv_weights: f0_conv_weights,
        conv_biases: f0_conv_biases,
        linear_w: f0_linear_w,
        linear_b: f0_linear_b,
    };

    HiFTGeneratorWeights {
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
        m_source_linear_w,
        m_source_linear_b,
        f0_predictor_weights,
    }
}

/// Public builder used by every parity test: deterministic small-shape
/// `HiFTGenerator` seeded off `seed`.
fn build_deterministic_hift_generator(seed: u64) -> HiFTGenerator {
    let cfg = small_hift_config();
    let mut state = seed;
    let weights = synth_generator_weights(&mut state, &cfg);
    HiFTGenerator::new(cfg, weights).expect("deterministic HiFT generator must build")
}

/// Deterministic small-shape mel. Row-major `[in_ch, t_mel]`, `synth_f32`
/// values (see [`synth_f32`]) — bounded in `[0, 0.65535)` so the audio_limit
/// clamp is not exercised by the mel itself.
fn build_deterministic_mel(in_ch: usize, t_mel: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    synth_vec(&mut state, in_ch * t_mel)
}

// ---------------------------------------------------------------------------
// Wave 4 synthesized-weight determinism / shape pins
// ---------------------------------------------------------------------------

#[test]
fn hift_gen_pipeline_deterministic_end_to_end() {
    // Two forward passes on the same generator + mel must be bit-identical.
    // This pins the `NsfEntropy::Deterministic` contract inside forward and
    // rules out hidden RNG in every helper on the chain.
    let generator = build_deterministic_hift_generator(0x1234_5678);
    let t_mel = 8;
    let cfg = small_hift_config();
    let in_ch = cfg.in_channels as usize;
    let mel = build_deterministic_mel(in_ch, t_mel, 0xABCD_EF00);
    let audio1 = generator
        .forward(&mel, t_mel)
        .expect("forward 1 must succeed");
    let audio2 = generator
        .forward(&mel, t_mel)
        .expect("forward 2 must succeed");

    // Bit-identical output.
    assert_eq!(audio1, audio2, "forward is not deterministic");

    // Finiteness (no Inf / NaN slipped through the exp / iSTFT path).
    assert!(
        audio1.iter().all(|s| s.is_finite()),
        "forward produced a non-finite sample"
    );

    // Amplitude bound: the terminal `audio.clamp(-audio_limit, audio_limit)`
    // guarantees this on the last line of `decode` — checking here pins that
    // the clamp actually ran (a refactor that lifts it above the return
    // would fail this assertion). `audio_limit = 0.99` in `small_hift_config`.
    assert!(
        audio1.iter().all(|s| s.abs() <= 0.99),
        "forward produced a sample outside audio_limit"
    );

    // Exact length pin: `audio.len() == t_mel * total_upsample_factor()`
    // (upstream contract, spelled out in Wave 3c-2 tests). With the small
    // config `total_upsample_factor() = 2 * 2 * 2 = 8`, so t_mel = 8 → 64.
    assert_eq!(
        audio1.len(),
        t_mel * 8,
        "audio length must equal t_mel * total_upsample_factor()"
    );
}

#[test]
fn hift_gen_different_seeds_produce_different_output() {
    // Same weights, different mels → the pipeline must be sensitive to the
    // mel. A silent input-mask (e.g. a converter that drops the mel and
    // returns a phase-only synthesis from the source module) would collapse
    // both outputs to the same waveform and trip here.
    let generator = build_deterministic_hift_generator(0x1111);
    let cfg = small_hift_config();
    let in_ch = cfg.in_channels as usize;
    let t_mel = 8;
    let mel_a = build_deterministic_mel(in_ch, t_mel, 0xAAAA);
    let mel_b = build_deterministic_mel(in_ch, t_mel, 0xBBBB);

    // The seed pair is intentionally far apart in the SplitMix64 stream —
    // adjacent seeds could correlate on the first draws.
    assert_ne!(mel_a, mel_b, "mel synthesizer collapsed distinct seeds");

    let audio_a = generator.forward(&mel_a, t_mel).unwrap();
    let audio_b = generator.forward(&mel_b, t_mel).unwrap();
    assert_ne!(
        audio_a, audio_b,
        "different mel must produce different audio"
    );

    // Sanity: both still bounded and finite.
    assert!(audio_a.iter().all(|s| s.is_finite() && s.abs() <= 0.99));
    assert!(audio_b.iter().all(|s| s.is_finite() && s.abs() <= 0.99));
}

#[test]
fn hift_gen_shape_stability_across_t_mel() {
    // `audio.len()` is a pure function of `t_mel` at a fixed config:
    // `t_mel * total_upsample_factor()`. Pinning the exact formula here means
    // any accidental change to the reflection-pad size, the terminal iSTFT
    // hop, or the upsample cascade breaks this test.
    let generator = build_deterministic_hift_generator(0x2222);
    let cfg = small_hift_config();
    let in_ch = cfg.in_channels as usize;
    let factor = cfg.total_upsample_factor() as usize;
    // Sanity that the derived factor really is what we hard-coded in the
    // determinism test above.
    assert_eq!(
        factor, 8,
        "total_upsample_factor changed — audit before adjusting the test"
    );
    for &t_mel in &[4usize, 8, 16] {
        let mel = build_deterministic_mel(in_ch, t_mel, 0xCCCC);
        let audio = generator
            .forward(&mel, t_mel)
            .expect("forward must succeed");
        assert_eq!(
            audio.len(),
            t_mel * factor,
            "shape drifted at t_mel = {t_mel}"
        );
        assert!(
            audio.iter().all(|s| s.is_finite()),
            "non-finite sample at t_mel = {t_mel}"
        );
        assert!(
            audio.iter().all(|s| s.abs() <= 0.99),
            "sample outside audio_limit at t_mel = {t_mel}"
        );
    }
}

// ---------------------------------------------------------------------------
// Flip-the-switch external reference — owner-provided checkpoint parity
// ---------------------------------------------------------------------------

/// Read `n` f32 values off `floats` at `*cursor`, advancing the cursor.
/// Returns `Err` if the stream cannot supply that many entries.
fn take_from(floats: &[f32], cursor: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    if *cursor + n > floats.len() {
        return Err(format!(
            "weights.bin truncated at offset {cursor} — needed {n} floats, have {}",
            floats.len() - *cursor
        ));
    }
    let out = floats[*cursor..*cursor + n].to_vec();
    *cursor += n;
    Ok(out)
}

/// Load one [`ResBlockWeights`] bundle from the running f32 stream, in the
/// same per-branch order (`convs1_w`, `convs1_b`, `convs2_w`, `convs2_b`,
/// `activations1_alpha`, `activations2_alpha`) [`synth_res_block`] emits.
fn load_res_block(
    floats: &[f32],
    cursor: &mut usize,
    channels: usize,
    kernel: usize,
    n_branches: usize,
) -> Result<ResBlockWeights, String> {
    let mut convs1_w = Vec::with_capacity(n_branches);
    let mut convs1_b = Vec::with_capacity(n_branches);
    let mut convs2_w = Vec::with_capacity(n_branches);
    let mut convs2_b = Vec::with_capacity(n_branches);
    let mut activations1_alpha = Vec::with_capacity(n_branches);
    let mut activations2_alpha = Vec::with_capacity(n_branches);
    for _ in 0..n_branches {
        convs1_w.push(take_from(floats, cursor, channels * channels * kernel)?);
        convs1_b.push(take_from(floats, cursor, channels)?);
        convs2_w.push(take_from(floats, cursor, channels * channels * kernel)?);
        convs2_b.push(take_from(floats, cursor, channels)?);
        activations1_alpha.push(take_from(floats, cursor, channels)?);
        activations2_alpha.push(take_from(floats, cursor, channels)?);
    }
    Ok(ResBlockWeights {
        convs1_w,
        convs1_b,
        convs2_w,
        convs2_b,
        activations1_alpha,
        activations2_alpha,
    })
}

/// Populate a [`HiFTGeneratorWeights`] bundle from a raw f32 LE stream using
/// the same walk documented on [`synth_generator_weights`]. Returns `Err`
/// with a human-readable message if the stream is under-populated — a
/// too-short `weights.bin` is a caller mistake and must surface loudly
/// rather than silently produce a bad model.
///
/// The layout intentionally matches the synthesized-weight walk so a fixture
/// generator can seed the same tensors with real data by writing the
/// concatenated f32 stream in that order. The `docs/tickets/sota-phase1-2/`
/// side of the M3-09 hand-off documents the Python-side dumper contract.
fn load_weights_from_bytes(
    cfg: &HiFTGeneratorConfig,
    bytes: &[u8],
) -> Result<HiFTGeneratorWeights, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "weights.bin length {} is not a multiple of 4 bytes",
            bytes.len()
        ));
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // A plain cursor + free [`take_from`] helper rather than a closure —
    // both the top-level walk and the nested [`load_res_block`] need to
    // advance the same offset, and closures that mutably borrow `cursor`
    // twice do not compose (E0499).
    let mut cursor: usize = 0usize;

    let n_ups = cfg.num_upsamples();
    let n_kernels = cfg.num_kernels();
    let bc = cfg.base_channels as usize;
    let inc = cfg.in_channels as usize;
    let n_fft_plus_2 = cfg.istft_n_fft as usize + 2;

    let conv_pre_w = take_from(&floats, &mut cursor, bc * inc * 7)?;
    let conv_pre_b = take_from(&floats, &mut cursor, bc)?;

    let mut ups_w = Vec::with_capacity(n_ups);
    let mut ups_b = Vec::with_capacity(n_ups);
    for i in 0..n_ups {
        let in_ch = bc >> i;
        let out_ch = bc >> (i + 1);
        let k = cfg.upsample_kernel_sizes[i] as usize;
        ups_w.push(take_from(&floats, &mut cursor, in_ch * out_ch * k)?);
        ups_b.push(take_from(&floats, &mut cursor, out_ch)?);
    }

    // Mirror `synth_generator_weights` — see there for the `downsample_us`
    // derivation.
    let mut downsample_rates: Vec<u32> = Vec::with_capacity(n_ups);
    downsample_rates.push(1);
    for i in (0..n_ups - 1).rev() {
        downsample_rates.push(cfg.upsample_rates[i]);
    }
    let mut downsample_cum: Vec<u32> = Vec::with_capacity(n_ups);
    let mut acc: u32 = 1;
    for &r in &downsample_rates {
        acc = acc.saturating_mul(r);
        downsample_cum.push(acc);
    }
    let downsample_us: Vec<u32> = downsample_cum.iter().rev().copied().collect();
    let mut source_downs_w = Vec::with_capacity(n_ups);
    let mut source_downs_b = Vec::with_capacity(n_ups);
    for (i, &u) in downsample_us.iter().enumerate() {
        let out_ch = cfg.output_channels_at(i) as usize;
        let k = if u == 1 { 1 } else { (u * 2) as usize };
        source_downs_w.push(take_from(&floats, &mut cursor, out_ch * n_fft_plus_2 * k)?);
        source_downs_b.push(take_from(&floats, &mut cursor, out_ch)?);
    }

    let mut source_resblock_weights = Vec::with_capacity(n_ups);
    for i in 0..n_ups {
        let ch = cfg.output_channels_at(i) as usize;
        let k = cfg.source_resblock_kernel_sizes[i] as usize;
        let n_branches = cfg.source_resblock_dilation_sizes[i].len();
        source_resblock_weights.push(load_res_block(&floats, &mut cursor, ch, k, n_branches)?);
    }

    let mut resblock_weights = Vec::with_capacity(n_ups * n_kernels);
    for i in 0..n_ups {
        let ch = cfg.output_channels_at(i) as usize;
        for j in 0..n_kernels {
            let k = cfg.resblock_kernel_sizes[j] as usize;
            let n_branches = cfg.resblock_dilation_sizes[j].len();
            resblock_weights.push(load_res_block(&floats, &mut cursor, ch, k, n_branches)?);
        }
    }

    let final_ch = cfg.output_channels_at(n_ups - 1) as usize;
    let conv_post_w = take_from(&floats, &mut cursor, n_fft_plus_2 * final_ch * 7)?;
    let conv_post_b = take_from(&floats, &mut cursor, n_fft_plus_2)?;

    let h1 = (cfg.nb_harmonics + 1) as usize;
    let m_source_linear_w = take_from(&floats, &mut cursor, h1)?;
    let m_source_linear_b_vec = take_from(&floats, &mut cursor, 1)?;
    let m_source_linear_b = m_source_linear_b_vec[0];

    let mut f0_conv_weights = Vec::with_capacity(5);
    let mut f0_conv_biases = Vec::with_capacity(5);
    f0_conv_weights.push(take_from(&floats, &mut cursor, bc * inc * 3)?);
    f0_conv_biases.push(take_from(&floats, &mut cursor, bc)?);
    for _ in 1..5 {
        f0_conv_weights.push(take_from(&floats, &mut cursor, bc * bc * 3)?);
        f0_conv_biases.push(take_from(&floats, &mut cursor, bc)?);
    }
    let f0_linear_w = take_from(&floats, &mut cursor, bc)?;
    let f0_linear_b = take_from(&floats, &mut cursor, 1)?;
    let f0_predictor_weights = F0PredictorWeights {
        conv_weights: f0_conv_weights,
        conv_biases: f0_conv_biases,
        linear_w: f0_linear_w,
        linear_b: f0_linear_b,
    };

    if cursor != floats.len() {
        return Err(format!(
            "weights.bin has {} trailing floats after populating the model — layout mismatch",
            floats.len() - cursor
        ));
    }

    Ok(HiFTGeneratorWeights {
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
        m_source_linear_w,
        m_source_linear_b,
        f0_predictor_weights,
    })
}

/// Parse a raw f32 LE byte stream into a `Vec<f32>`.
fn f32_stream(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "raw f32 stream length {} is not a multiple of 4 bytes",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Comparison atol for the flip-the-switch harness. Documented (rather than
/// buried) so a future PR that widens it has to explicitly justify why.
///
/// - `NFR-QL-01` design-wide FP32 envelope: `atol = 0.01`.
/// - HiFTNet output is bounded by `audio_limit = 0.99`, so `1e-3` is ~0.1% of
///   the amplitude range — tight enough to catch a real regression but loose
///   enough to survive minor reduction-order differences vs an upstream
///   PyTorch dump on a small config.
const EXTERNAL_ATOL: f32 = 1e-3;

#[test]
fn hift_gen_matches_external_reference_when_available() {
    let dir = match std::env::var_os("VOKRA_HIFTNET_REFERENCE_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            // GitHub Actions annotation — visible in the workflow log but not
            // a failure. Stdout is fine here: `cargo test` swallows it under
            // pass, and the CI annotation harness reads the exact prefix.
            println!(
                "::warning::hift_gen_matches_external_reference_when_available skipped — \
                 VOKRA_HIFTNET_REFERENCE_DIR unset (owner-provided reference material not \
                 available yet)"
            );
            return;
        }
    };
    if !dir.is_dir() {
        panic!(
            "VOKRA_HIFTNET_REFERENCE_DIR = {dir:?} is not a directory",
            dir = dir.display()
        );
    }

    let cfg = small_hift_config();
    let in_ch = cfg.in_channels as usize;

    let weights_bytes =
        std::fs::read(dir.join("weights.bin")).unwrap_or_else(|e| panic!("read weights.bin: {e}"));
    let mel_bytes =
        std::fs::read(dir.join("mel.bin")).unwrap_or_else(|e| panic!("read mel.bin: {e}"));
    let expected_bytes = std::fs::read(dir.join("expected_audio.bin"))
        .unwrap_or_else(|e| panic!("read expected_audio.bin: {e}"));

    let weights = load_weights_from_bytes(&cfg, &weights_bytes)
        .unwrap_or_else(|e| panic!("weights.bin layout error: {e}"));
    let mel = f32_stream(&mel_bytes).unwrap_or_else(|e| panic!("mel.bin: {e}"));
    let expected =
        f32_stream(&expected_bytes).unwrap_or_else(|e| panic!("expected_audio.bin: {e}"));

    if mel.len() % in_ch != 0 {
        panic!(
            "mel.bin length {} is not a multiple of in_channels = {in_ch}",
            mel.len()
        );
    }
    let t_mel = mel.len() / in_ch;
    let expected_len = t_mel * cfg.total_upsample_factor() as usize;
    if expected.len() != expected_len {
        panic!(
            "expected_audio.bin length {} does not match t_mel * total_upsample_factor = {}",
            expected.len(),
            expected_len
        );
    }

    // Optional shape sanity via `config.env` — if the file exists it must
    // contain a single `expected_len=<N>` line, and `N` must match the
    // computed length. Anything else fails loudly rather than silently.
    let cfg_env = dir.join("config.env");
    if cfg_env.is_file() {
        let text =
            std::fs::read_to_string(&cfg_env).unwrap_or_else(|e| panic!("read config.env: {e}"));
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let (k, v) = trimmed
                .split_once('=')
                .unwrap_or_else(|| panic!("config.env: malformed line {trimmed:?}"));
            match k.trim() {
                "expected_len" => {
                    let claimed: usize = v
                        .trim()
                        .parse()
                        .unwrap_or_else(|e| panic!("config.env expected_len: {e}"));
                    assert_eq!(
                        claimed, expected_len,
                        "config.env expected_len disagrees with derived length"
                    );
                }
                other => panic!("config.env: unknown key {other:?}"),
            }
        }
    }

    let generator =
        HiFTGenerator::new(cfg, weights).expect("real-checkpoint HiFT generator must build");
    let audio = generator
        .forward(&mel, t_mel)
        .expect("real-checkpoint forward");
    assert_eq!(
        audio.len(),
        expected.len(),
        "Vokra output length disagrees with the reference"
    );

    let mut max_abs_delta = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (v, r)) in audio.iter().zip(expected.iter()).enumerate() {
        let d = (v - r).abs();
        if d > max_abs_delta {
            max_abs_delta = d;
            worst_idx = i;
        }
    }
    // Emit the observed delta on both pass and fail — a stable-but-close
    // reference should be visible in CI logs when the switch flips green.
    println!(
        "hift_gen external reference: max |Δ| = {max_abs_delta:.3e} at sample {worst_idx} \
         (atol = {EXTERNAL_ATOL:.3e})"
    );
    assert!(
        max_abs_delta <= EXTERNAL_ATOL,
        "reference max |Δ| = {max_abs_delta:.6} exceeds atol = {EXTERNAL_ATOL:.6}"
    );
}

// ---------------------------------------------------------------------------
// SoTA plan Phase 1-2 audit follow-up (2026-07-24): gap-fill tests.
//
// The upstream harness pins determinism, mel-input sensitivity, and shape
// stability, plus a flip-the-switch external-reference hook. The audit
// flagged three residual holes:
//
//   1. The CENTRAL invariant of the flip-the-switch design — that
//      `synth_generator_weights` and `load_weights_from_bytes` walk
//      identical tensor sequences — was not pinned. Silent drift on either
//      side would only surface when an owner produced a fixture, and
//      would look like an unexplained reference-parity failure. The two
//      round-trip tests below pin the invariant end-to-end (layout ==
//      then forward ==).
//   2. All four fixture-loader error branches (2 non-multiple-of-4 gates
//      + `take_from` truncation propagation + trailing-floats layout
//      mismatch) were unreachable from the existing tests, so a refactor
//      that weakened any guardrail would go unnoticed. The four
//      loader-error tests below feed shaped-to-fail byte streams to
//      pin each `Err(_)` branch.
//   3. Weight-influence sensitivity was untested — every existing test
//      either fixed the weights or varied the mel, so a silent
//      weights-are-ignored regression (converter that always uses
//      defaults, or a forward that drops the weight tensors on the
//      source-fusion chain) would pass every assertion. The final test
//      below pins that distinct weights on the same mel produce distinct
//      audio.
// ---------------------------------------------------------------------------

/// Push every `f32` in `src` to `out` as little-endian bytes. Mirrors the
/// `f32::from_le_bytes` walk in [`load_weights_from_bytes`].
fn push_f32s(out: &mut Vec<u8>, src: &[f32]) {
    for &v in src {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Serialize a [`ResBlockWeights`] bundle in the same per-branch order
/// [`synth_res_block`] emits.
fn push_res_block(out: &mut Vec<u8>, rb: &ResBlockWeights) {
    let n = rb.convs1_w.len();
    for i in 0..n {
        push_f32s(out, &rb.convs1_w[i]);
        push_f32s(out, &rb.convs1_b[i]);
        push_f32s(out, &rb.convs2_w[i]);
        push_f32s(out, &rb.convs2_b[i]);
        push_f32s(out, &rb.activations1_alpha[i]);
        push_f32s(out, &rb.activations2_alpha[i]);
    }
}

/// Serialize a [`HiFTGeneratorWeights`] bundle to raw little-endian f32
/// bytes in the exact walk [`load_weights_from_bytes`] consumes and
/// [`synth_generator_weights`] emits.
///
/// The three walks (synth, load, this serializer) must stay in lockstep;
/// the round-trip tests below use this helper to prove they still do.
/// A reorder on any of the three trips [`hift_gen_serialize_load_round_trip_bit_identical_layout`]
/// on the very next `cargo test` run.
fn serialize_generator_weights(w: &HiFTGeneratorWeights) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    // (1) conv_pre
    push_f32s(&mut out, &w.conv_pre_w);
    push_f32s(&mut out, &w.conv_pre_b);

    // (2) ups (per stage)
    for i in 0..w.ups_w.len() {
        push_f32s(&mut out, &w.ups_w[i]);
        push_f32s(&mut out, &w.ups_b[i]);
    }

    // (3) source_downs (per stage)
    for i in 0..w.source_downs_w.len() {
        push_f32s(&mut out, &w.source_downs_w[i]);
        push_f32s(&mut out, &w.source_downs_b[i]);
    }

    // (4) source_resblocks (per stage)
    for rb in &w.source_resblock_weights {
        push_res_block(&mut out, rb);
    }

    // (5) resblocks (row-major over [i * num_kernels + j])
    for rb in &w.resblock_weights {
        push_res_block(&mut out, rb);
    }

    // (6) conv_post
    push_f32s(&mut out, &w.conv_post_w);
    push_f32s(&mut out, &w.conv_post_b);

    // (7) source module linear head — vector then single scalar
    push_f32s(&mut out, &w.m_source_linear_w);
    out.extend_from_slice(&w.m_source_linear_b.to_le_bytes());

    // (8) F0 predictor: 5 conv weight+bias pairs, then linear_w, linear_b
    let f0 = &w.f0_predictor_weights;
    for i in 0..f0.conv_weights.len() {
        push_f32s(&mut out, &f0.conv_weights[i]);
        push_f32s(&mut out, &f0.conv_biases[i]);
    }
    push_f32s(&mut out, &f0.linear_w);
    push_f32s(&mut out, &f0.linear_b);

    out
}

/// Per-field bit-identical assertion on two [`ResBlockWeights`] bundles.
/// Split out so the top-level round-trip test can point at the exact
/// branch that drifted (the `Debug` printout of a whole
/// `HiFTGeneratorWeights` bundle is unreadable).
fn assert_res_block_bit_identical(a: &ResBlockWeights, b: &ResBlockWeights, tag: &str) {
    assert_eq!(a.convs1_w, b.convs1_w, "{tag}: convs1_w drift");
    assert_eq!(a.convs1_b, b.convs1_b, "{tag}: convs1_b drift");
    assert_eq!(a.convs2_w, b.convs2_w, "{tag}: convs2_w drift");
    assert_eq!(a.convs2_b, b.convs2_b, "{tag}: convs2_b drift");
    assert_eq!(
        a.activations1_alpha, b.activations1_alpha,
        "{tag}: activations1_alpha drift"
    );
    assert_eq!(
        a.activations2_alpha, b.activations2_alpha,
        "{tag}: activations2_alpha drift"
    );
}

/// Per-field bit-identical assertion on two [`HiFTGeneratorWeights`]
/// bundles. Walks every leaf so a silent drop or reorder trips the
/// message that points at the exact tensor.
fn assert_generator_weights_bit_identical(a: &HiFTGeneratorWeights, b: &HiFTGeneratorWeights) {
    assert_eq!(a.conv_pre_w, b.conv_pre_w, "conv_pre_w drift");
    assert_eq!(a.conv_pre_b, b.conv_pre_b, "conv_pre_b drift");
    assert_eq!(a.ups_w, b.ups_w, "ups_w drift");
    assert_eq!(a.ups_b, b.ups_b, "ups_b drift");
    assert_eq!(a.source_downs_w, b.source_downs_w, "source_downs_w drift");
    assert_eq!(a.source_downs_b, b.source_downs_b, "source_downs_b drift");
    assert_eq!(
        a.source_resblock_weights.len(),
        b.source_resblock_weights.len(),
        "source_resblock_weights count drift"
    );
    for (i, (x, y)) in a
        .source_resblock_weights
        .iter()
        .zip(b.source_resblock_weights.iter())
        .enumerate()
    {
        assert_res_block_bit_identical(x, y, &format!("source_resblock[{i}]"));
    }
    assert_eq!(
        a.resblock_weights.len(),
        b.resblock_weights.len(),
        "resblock_weights count drift"
    );
    for (i, (x, y)) in a
        .resblock_weights
        .iter()
        .zip(b.resblock_weights.iter())
        .enumerate()
    {
        assert_res_block_bit_identical(x, y, &format!("resblock[{i}]"));
    }
    assert_eq!(a.conv_post_w, b.conv_post_w, "conv_post_w drift");
    assert_eq!(a.conv_post_b, b.conv_post_b, "conv_post_b drift");
    assert_eq!(
        a.m_source_linear_w, b.m_source_linear_w,
        "m_source_linear_w drift"
    );
    // Bit-pattern compare on the scalar catches ±0 / signalling-NaN drift
    // that `==` would silently accept.
    assert_eq!(
        a.m_source_linear_b.to_bits(),
        b.m_source_linear_b.to_bits(),
        "m_source_linear_b drift"
    );
    assert_eq!(
        a.f0_predictor_weights.conv_weights, b.f0_predictor_weights.conv_weights,
        "f0.conv_weights drift"
    );
    assert_eq!(
        a.f0_predictor_weights.conv_biases, b.f0_predictor_weights.conv_biases,
        "f0.conv_biases drift"
    );
    assert_eq!(
        a.f0_predictor_weights.linear_w, b.f0_predictor_weights.linear_w,
        "f0.linear_w drift"
    );
    assert_eq!(
        a.f0_predictor_weights.linear_b, b.f0_predictor_weights.linear_b,
        "f0.linear_b drift"
    );
}

#[test]
fn hift_gen_serialize_load_round_trip_bit_identical_layout() {
    // Central invariant of the flip-the-switch design:
    // `synth_generator_weights` and `load_weights_from_bytes` must walk
    // identical tensor sequences. Serialize the synthesized bundle in the
    // documented walk order, feed those bytes through the loader, and
    // assert every leaf tensor is bit-identical. A reorder or missed
    // tensor on either the synth or the load side trips here.
    let cfg = small_hift_config();
    let mut state = 0xC0FF_EE00_DEAD_BEEFu64;
    let original = synth_generator_weights(&mut state, &cfg);

    let bytes = serialize_generator_weights(&original);
    let reconstructed = load_weights_from_bytes(&cfg, &bytes)
        .expect("round-trip load must succeed on well-formed bytes");

    assert_generator_weights_bit_identical(&original, &reconstructed);
}

#[test]
fn hift_gen_serialize_load_round_trip_forward_bit_identical() {
    // Ultimate contract of the flip-the-switch harness: a well-formed
    // fixture reconstructs the exact model. Even if the layout matches,
    // this test proves the reconstructed generator forwards to
    // bit-identical audio versus the direct-synth generator — the two
    // must be functionally interchangeable, not merely field-equal
    // (e.g. a hypothetical serializer that lossily rounded a subset of
    // weights would pass a partial layout check but fail here).
    let cfg = small_hift_config();
    let mut state = 0x1122_3344_5566_7788u64;
    let original = synth_generator_weights(&mut state, &cfg);
    let bytes = serialize_generator_weights(&original);
    let reconstructed =
        load_weights_from_bytes(&cfg, &bytes).expect("round-trip load must succeed");

    let gen_direct =
        HiFTGenerator::new(cfg.clone(), original).expect("direct-synth generator must build");
    let gen_reconstructed =
        HiFTGenerator::new(cfg.clone(), reconstructed).expect("reconstructed generator must build");

    let in_ch = cfg.in_channels as usize;
    let t_mel = 8;
    let mel = build_deterministic_mel(in_ch, t_mel, 0x9988_7766_5544_3322u64);
    let audio_direct = gen_direct
        .forward(&mel, t_mel)
        .expect("direct generator forward");
    let audio_reconstructed = gen_reconstructed
        .forward(&mel, t_mel)
        .expect("reconstructed generator forward");

    assert_eq!(
        audio_direct, audio_reconstructed,
        "reconstructed generator must forward to bit-identical audio \
         (round-trip contract of the flip-the-switch harness)"
    );
}

#[test]
fn load_weights_from_bytes_rejects_non_multiple_of_4() {
    // A byte stream whose length is not a multiple of 4 cannot be
    // interpreted as an f32 LE sequence — the loader's very first gate
    // has to reject it loudly (FR-EX-08: no silent truncation of the
    // fixture stream).
    let cfg = small_hift_config();
    let err = load_weights_from_bytes(&cfg, &[0u8; 3])
        .expect_err("non-multiple-of-4 byte length must surface an error");
    assert!(
        err.contains("is not a multiple of 4 bytes"),
        "expected non-multiple-of-4 error message, got: {err}"
    );
}

#[test]
fn load_weights_from_bytes_rejects_truncated_stream() {
    // A stream that is a valid f32-multiple length but too short to
    // populate every weight must surface the `take_from` truncation
    // message rather than silently producing a partial model. 4 bytes =
    // 1 f32, way short of the ~2000 floats the small config needs, so
    // the first `take_from` call runs out on `conv_pre_w`.
    let cfg = small_hift_config();
    let err = load_weights_from_bytes(&cfg, &[0u8; 4])
        .expect_err("undersized-but-4-aligned stream must surface an error");
    assert!(
        err.contains("truncated at offset"),
        "expected truncation error message, got: {err}"
    );
}

#[test]
fn load_weights_from_bytes_rejects_trailing_floats() {
    // Serialize a valid weights bundle, then append one extra f32 worth
    // of zero bytes. The loader's terminal `cursor != floats.len()` gate
    // must reject that layout mismatch — otherwise a fixture produced
    // against a different config could silently populate a partial model
    // and leak the extra parameters, defeating the whole point of the
    // walk-order contract.
    let cfg = small_hift_config();
    let mut state = 0xFEED_FACE_CAFE_BABEu64;
    let weights = synth_generator_weights(&mut state, &cfg);
    let mut bytes = serialize_generator_weights(&weights);
    bytes.extend_from_slice(&0.0_f32.to_le_bytes());
    let err = load_weights_from_bytes(&cfg, &bytes)
        .expect_err("oversized-but-4-aligned stream must surface an error");
    assert!(
        err.contains("trailing floats after populating the model"),
        "expected trailing-floats error message, got: {err}"
    );
}

#[test]
fn f32_stream_rejects_non_multiple_of_4() {
    // Same guardrail as `load_weights_from_bytes` but on the raw
    // mel / expected-audio byte parser. 5 bytes cannot decode into a
    // valid f32 LE sequence, so the length gate must surface loudly
    // (FR-EX-08). Without this test the loader could be quietly relaxed
    // to `chunks_exact` + drop-the-remainder and every existing
    // parity_hiftnet test would still pass.
    let err =
        f32_stream(&[0u8; 5]).expect_err("non-multiple-of-4 byte length must surface an error");
    assert!(
        err.contains("raw f32 stream length"),
        "expected non-multiple-of-4 error message, got: {err}"
    );
}

#[test]
fn hift_gen_weight_influence_sensitivity() {
    // Complement of `hift_gen_different_seeds_produce_different_output`
    // (same weights + different mels → different audio). Here we fix
    // the mel and vary the generator seed. A silent weights-are-ignored
    // regression (e.g. a converter that always installs the default
    // bundle, or a forward that drops the ResBlock outputs on the
    // source-fusion chain) would collapse both outputs to the same
    // waveform and trip here even though every current always-on test
    // (which either fixes the weights or the mel) would still pass.
    let gen_a = build_deterministic_hift_generator(0x0A0A_0A0A_0A0A_0A0Au64);
    let gen_b = build_deterministic_hift_generator(0xB0B0_B0B0_B0B0_B0B0u64);
    let cfg = small_hift_config();
    let in_ch = cfg.in_channels as usize;
    let t_mel = 8;
    let mel = build_deterministic_mel(in_ch, t_mel, 0xF00D_D00F_F00D_D00Fu64);

    let audio_a = gen_a
        .forward(&mel, t_mel)
        .expect("weight-A forward must succeed");
    let audio_b = gen_b
        .forward(&mel, t_mel)
        .expect("weight-B forward must succeed");

    assert_ne!(
        audio_a, audio_b,
        "distinct weights on the same mel must produce distinct audio — \
         a silent weights-are-ignored regression would collapse them"
    );
    assert!(
        audio_a.iter().all(|s| s.is_finite() && s.abs() <= 0.99),
        "audio_a violated finiteness or audio_limit"
    );
    assert!(
        audio_b.iter().all(|s| s.is_finite() && s.abs() <= 0.99),
        "audio_b violated finiteness or audio_limit"
    );
}
