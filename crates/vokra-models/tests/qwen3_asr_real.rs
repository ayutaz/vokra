//! VAST-only strict binding smoke for the released Qwen3-ASR GGUFs.
//!
//! No fixture is downloaded by this test.  The remote validation worker sets
//! either environment variable after preparing the corresponding public file.

use vokra_models::qwen3_asr::{Qwen3AsrCheckpoint, Qwen3AsrVariant};

fn bind_from_env(variable: &str, expected: Qwen3AsrVariant) {
    let Ok(path) = std::env::var(variable) else {
        return;
    };
    let file = vokra_mmap::open_gguf(&path).expect("open Qwen3-ASR GGUF through mmap");
    let checkpoint = Qwen3AsrCheckpoint::from_gguf(&file).expect("strict Qwen3-ASR bind");
    assert_eq!(checkpoint.variant(), expected);
    assert_eq!(checkpoint.tensor_count(), expected.tensor_count());
    assert_eq!(checkpoint.model_name(), expected.model_name());
}

#[test]
fn qwen3_asr_0_6b_strict_public_contract() {
    bind_from_env("VOKRA_QWEN3_ASR_0_6B_GGUF", Qwen3AsrVariant::B06);
}

#[test]
fn qwen3_asr_1_7b_strict_public_contract() {
    bind_from_env("VOKRA_QWEN3_ASR_1_7B_GGUF", Qwen3AsrVariant::B17);
}
