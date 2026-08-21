//! Real-artifact gates for the runtime-gap TTS checkpoint wave.
//!
//! These tests are ignored by default because the fixed-revision GGUFs are
//! large. CI/VAST supplies the paths explicitly; absence never becomes a
//! success-shaped skip.

// Keep the independently dumped PyTorch f32 decimal spellings intact. Rust
// rounds each literal to f32 before comparison; shortening hundreds of
// reference values would make provenance review needlessly harder.
#![allow(clippy::excessive_precision)]

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_models::chatterbox::ChatterboxCheckpoint;
use vokra_models::chatterbox_nano::ChatterboxNanoCheckpoint;
use vokra_models::chatterbox_turbo::ChatterboxTurboCheckpoint;
use vokra_models::cosyvoice3::CosyVoice3Checkpoint;
use vokra_models::dia::DiaCheckpoint;
use vokra_models::vibevoice::VibeVoiceCheckpoint;
use vokra_models::voxcpm2::VoxCpm2Checkpoint;
use vokra_models::zonos::ZonosCheckpoint;

fn open(env: &str) -> GgufFile {
    let path = std::env::var(env).unwrap_or_else(|_| panic!("{env} must name the pinned GGUF"));
    vokra_mmap::open_gguf(Path::new(&path))
        .unwrap_or_else(|error| panic!("mmap {env}={path}: {error}"))
}

fn affine_input(dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|index| ((index as f32 + 0.25) * 0.071).sin() * 0.2)
        .collect()
}

fn assert_real_output(output: &[f32], expected: usize) {
    assert_eq!(output.len(), expected);
    assert!(output.iter().all(|value| value.is_finite()));
    assert!(output.iter().any(|value| value.abs() > 1.0e-7));
    let first = output[0];
    assert!(output.iter().any(|value| (*value - first).abs() > 1.0e-7));
}

fn assert_reference_prefix(label: &str, output: &[f32], reference: &[f32], bound: f32) {
    let max_error = output
        .iter()
        .zip(reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{label} prefix max_abs={max_error:e}");
    assert!(
        max_error <= bound,
        "{label} max_abs={max_error:e} exceeds measured PyTorch bound {bound:e}"
    );
}

#[test]
#[ignore = "requires fixed-revision 2.14 GB Chatterbox multilingual GGUF"]
fn chatterbox_multilingual_strict_bind_and_real_projection() {
    let file = open("VOKRA_CHATTERBOX_GGUF");
    let checkpoint = ChatterboxCheckpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 292);
    let projection = checkpoint
        .load_speaker_projection(&file)
        .expect("decode speaker projection");
    let output = projection.forward(&affine_input(256)).expect("forward");
    assert_real_output(&output, 1_024);
    assert_reference_prefix(
        "chatterbox speaker projection",
        &output,
        &[
            -0.0213574786,
            -0.0504276901,
            0.0373493284,
            0.0366551541,
            0.00544776302,
            0.0594436377,
            0.0578831285,
            -0.0649379864,
            0.0182958152,
            0.0172990002,
            0.091600433,
            0.0604639649,
            0.0645693168,
            -0.0320279859,
            -0.0226085782,
            0.0391765088,
            0.0514399521,
            0.076956965,
            -0.07904201,
            -0.0705867484,
            -0.103328578,
            0.00338842347,
            0.0660204142,
            -0.0192097258,
            -0.0281144083,
            -0.0260800924,
            0.0763431638,
            0.0385765359,
            0.0151573988,
            0.02630179,
            -0.0412452072,
            0.00397570524,
        ],
        5.0e-7,
    );
}

#[test]
#[ignore = "requires fixed-revision 870 MB Chatterbox Nano GGUF"]
fn chatterbox_nano_strict_bind_and_real_projection() {
    let file = open("VOKRA_CHATTERBOX_NANO_GGUF");
    let checkpoint = ChatterboxNanoCheckpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 155);
    let projection = checkpoint
        .load_speaker_projection(&file)
        .expect("decode speaker projection");
    let output = projection.forward(&affine_input(256)).expect("forward");
    assert_real_output(&output, 768);
    assert_reference_prefix(
        "chatterbox_nano speaker projection",
        &output,
        &[
            0.0398503989,
            0.06499286,
            -0.00922413915,
            -0.120761,
            -0.0494162776,
            -0.161250249,
            -0.108551748,
            0.0392697416,
            0.0968077108,
            0.145942941,
            0.0383821204,
            -0.150413454,
            -0.291820407,
            0.229225904,
            -0.0297782663,
            0.163138837,
            -0.293165386,
            -0.179192439,
            0.00893304497,
            -0.0771709383,
            -0.299783796,
            0.0606808625,
            -0.267308354,
            0.0901086628,
            0.214719474,
            0.0578269474,
            -0.139007643,
            0.0377126522,
            -0.239998654,
            0.288421214,
            0.0618908368,
            0.0214220621,
        ],
        5.0e-7,
    );
}

#[test]
#[ignore = "requires fixed-revision 1.92 GB Chatterbox Turbo GGUF"]
fn chatterbox_turbo_strict_bind_and_real_projection() {
    let file = open("VOKRA_CHATTERBOX_TURBO_GGUF");
    let checkpoint = ChatterboxTurboCheckpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 299);
    let projection = checkpoint
        .load_speaker_projection(&file)
        .expect("decode speaker projection");
    let output = projection.forward(&affine_input(256)).expect("forward");
    assert_real_output(&output, 1_024);
    assert_reference_prefix(
        "chatterbox_turbo speaker projection",
        &output,
        &[
            0.0630918071,
            -0.0231171958,
            0.215112984,
            0.0763546377,
            -0.228449464,
            0.154496059,
            -0.012388533,
            -0.199277624,
            0.00992986839,
            -0.358590126,
            0.276085228,
            -0.05257532,
            -0.0274771526,
            -0.23591426,
            -0.215029597,
            -0.0767177418,
            0.132538378,
            0.0850787982,
            0.179538444,
            -0.0981844291,
            -0.243113518,
            -0.0527831241,
            -0.128546461,
            -0.0140209123,
            -0.00594576634,
            -0.0231657326,
            -0.0937021375,
            0.0725198984,
            0.191654637,
            -0.123037875,
            -0.0208409857,
            0.0876440033,
        ],
        5.0e-7,
    );
}

#[test]
#[ignore = "requires fixed-revision 2.58 GB Fun-CosyVoice3 GGUF"]
fn cosyvoice3_strict_bind_and_real_q_projection() {
    let file = open("VOKRA_COSYVOICE3_GGUF");
    let checkpoint = CosyVoice3Checkpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 293);
    let projection = checkpoint
        .load_layer0_q_projection(&file)
        .expect("decode q projection");
    let output = projection.forward(&affine_input(896)).expect("forward");
    assert_real_output(&output, 896);
    assert_reference_prefix(
        "cosyvoice3 q projection",
        &output,
        &[
            0.00731309503,
            -0.00574579462,
            0.0828151926,
            -0.0930662826,
            0.0272626989,
            -0.00376475602,
            -7.83040285,
            -0.00697000325,
            9.0211401,
            -1.99742758,
            0.067421481,
            10.9700918,
            3.33760619,
            0.0172688067,
            -0.293850243,
            3.3996172,
            -3.1218152,
            6.4358902,
            -7.17156982,
            -1.10693765,
            -0.713756382,
            -1.23177493,
            -3.40016651,
            4.01742125,
            -3.61779284,
            1.9913435,
            -5.461761,
            -0.479076564,
            1.65221691,
            1.34833884,
            -0.6209203,
            0.251909554,
        ],
        1.0e-5,
    );
}

#[test]
#[ignore = "requires fixed-revision 6.44 GB Dia-1.6B GGUF"]
fn dia_strict_bind_and_real_text_embedding() {
    let file = open("VOKRA_DIA_GGUF");
    let checkpoint = DiaCheckpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 343);
    let embedding = checkpoint
        .load_text_embedding(&file)
        .expect("decode text embedding");
    let output = embedding.forward(&[1, 42, 255]).expect("forward");
    assert_real_output(&output, 3 * 1_024);
    assert_reference_prefix(
        "dia text embedding",
        &output,
        &[
            0.0453768894,
            -0.00585064664,
            -0.0163512137,
            -0.0240538325,
            0.0315013379,
            -0.0211965907,
            0.00304786558,
            0.0158774536,
            0.00254370342,
            -0.0174281895,
            -0.0111410916,
            -0.0669849217,
            0.00796367973,
            -0.000546370225,
            0.0310824998,
            0.00993846077,
            -0.0184143316,
            -0.0535238348,
            -0.0309143774,
            -0.0014393609,
            0.00973356143,
            -0.0276208725,
            -0.0408787467,
            0.00876725838,
            0.000812127721,
            0.0387268364,
            -0.0123188952,
            -0.0260790456,
            0.0257592183,
            -0.0241086092,
            -0.0116080735,
            -0.0325664394,
        ],
        1.0e-8,
    );
}

#[test]
#[ignore = "requires fixed-revision 5.41 GB VibeVoice-1.5B GGUF"]
fn vibevoice_strict_bind_and_real_acoustic_projection() {
    let file = open("VOKRA_VIBEVOICE_GGUF");
    let checkpoint = VibeVoiceCheckpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 1_204);
    let projection = checkpoint
        .load_acoustic_projection(&file)
        .expect("decode acoustic projection");
    let output = projection.forward(&affine_input(64)).expect("forward");
    assert_real_output(&output, 1_536);
    assert_reference_prefix(
        "vibevoice acoustic projection",
        &output,
        &[
            0.0124888122,
            -0.0719040185,
            0.0111484081,
            -0.00343785249,
            0.00104200095,
            -0.0360164791,
            0.00271556247,
            0.0834426805,
            0.0357392319,
            -0.0258298796,
            -0.038799502,
            0.0743041039,
            -0.0773309916,
            0.0136516597,
            0.0736978054,
            -0.0140699483,
            -0.00457623973,
            0.016677903,
            0.0652018487,
            -0.0167379007,
            -0.0284110438,
            0.0616351888,
            -0.0295339562,
            0.0196545348,
            -0.0187148973,
            0.0823164135,
            -0.0168883204,
            0.0332447588,
            -0.020168962,
            0.0153077934,
            0.00274825655,
            -0.072176002,
        ],
        1.0e-7,
    );
}

#[test]
#[ignore = "requires fixed-revision 1.30 GB VoxCPM-0.5B GGUF"]
fn voxcpm2_strict_bind_and_real_stop_projection() {
    let file = open("VOKRA_VOXCPM2_GGUF");
    let checkpoint = VoxCpm2Checkpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 377);
    let projection = checkpoint
        .load_stop_projection(&file)
        .expect("decode stop projection");
    let output = projection.forward(&affine_input(1_024)).expect("forward");
    assert_real_output(&output, 1_024);
    assert_reference_prefix(
        "voxcpm2 stop projection",
        &output,
        &[
            0.0289057195,
            -0.107121706,
            0.756561875,
            0.612757921,
            0.0638550073,
            0.0119663216,
            -0.0563777015,
            0.0534603074,
            -0.125252023,
            0.15408127,
            -0.00341626815,
            -0.0104399761,
            -0.187445283,
            -0.299942672,
            -0.354093611,
            -0.00753260124,
            0.0170237571,
            0.0341097936,
            0.00786537305,
            -0.125270367,
            -0.362320691,
            0.468626112,
            0.0371146798,
            0.516944289,
            0.440824807,
            0.541262031,
            0.154367015,
            -0.0118931131,
            0.03902556,
            -0.302830696,
            0.0322887562,
            -0.376780212,
        ],
        1.0e-6,
    );
}

#[test]
#[ignore = "requires fixed-revision 3.25 GB Zonos-v0.1-transformer GGUF"]
fn zonos_strict_bind_and_real_speaker_projection() {
    let file = open("VOKRA_ZONOS_GGUF");
    let checkpoint = ZonosCheckpoint::from_gguf(&file).expect("strict bind");
    assert_eq!(checkpoint.tensor_count(), 246);
    let projection = checkpoint
        .load_speaker_projection(&file)
        .expect("decode speaker projection");
    let output = projection.forward(&affine_input(128)).expect("forward");
    assert_real_output(&output, 2_048);
    assert_reference_prefix(
        "zonos speaker projection",
        &output,
        &[
            0.0492350608,
            -0.0449244902,
            -0.106137052,
            0.183792472,
            -0.170996606,
            -0.112742789,
            0.138072014,
            -0.021395579,
            -0.0121398829,
            -0.0125007182,
            0.00347883999,
            0.0510306433,
            0.162948474,
            0.104476281,
            0.00917741656,
            -0.124461994,
            -0.0478815362,
            0.0937883034,
            0.120375335,
            -0.0445312224,
            0.198787346,
            0.0899419338,
            -0.065318279,
            0.0913152397,
            0.0872428417,
            -0.0174940042,
            0.0719530284,
            0.00958740152,
            -0.230998576,
            0.113072708,
            0.0695867091,
            0.0422019586,
        ],
        1.0e-7,
    );
}
