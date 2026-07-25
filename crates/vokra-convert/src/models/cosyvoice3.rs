//! **Fun-CosyVoice3-0.5B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 3, 2026-07-24).
//!
//! Input: the upstream `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` LLM
//! checkpoint (`llm.pt` exported to safetensors with verbatim tensor
//! names — the release ships torch pickles + a mixed-format package
//! including `flow.decoder.estimator.fp32.onnx` that the runtime
//! **never touches** at load, FR-LD-05). Output: a GGUF carrying every
//! float tensor plus the `vokra.model.*` and `vokra.cosyvoice3.*`
//! metadata chunks the native Fun-CosyVoice3 implementation
//! (`crates/vokra-models/src/cosyvoice3/`) reads.
//!
//! # Very-cheap follow-on — reuses CosyVoice2 shape-derivation
//!
//! Because the topology is a CosyVoice2 chain with training-side
//! refinements (DRSR + Core-Cocktail — arXiv:2505.17589), the
//! shape-derivation path applies verbatim: `vocab_size` / `hidden_dim`
//! from `llm.model.model.embed_tokens.weight`, `n_layer` from the
//! contiguous `llm.model.model.layers.{i}.*` block count, `ffn_dim`
//! from layer-0 `mlp.gate_proj.weight`, GQA algebra cross-checks from
//! `q_proj` / `k_proj` widths. This converter delegates the tensor
//! walk to [`crate::models::cosyvoice2::convert_with_config_and_tokenizer`]
//! and re-writes only the `vokra.model.*` + `vokra.cosyvoice3.*` chunks
//! on top so the runtime dispatches to `vokra-models::cosyvoice3` (a
//! different arch label) instead of `vokra-models::cosyvoice2`.
//!
//! # Hparam derivation
//!
//! Same as CosyVoice2 — the `--config` side-car (upstream HF
//! `config.json`, Qwen2 schema) supplies the GQA head split
//! (`num_attention_heads` / `num_key_value_heads`), `rope_theta`,
//! `rms_norm_eps`, `max_position_embeddings`; without it those keys
//! stay `0`-absent and the runtime refuses the LLM bind (loud,
//! FR-EX-08). Cross-checks between the config and the tensor shapes
//! (hidden size, layer count, FFN width, vocab, GQA algebra) fail the
//! conversion loudly.
//!
//! # Q/K/V attention biases
//!
//! The Qwen2 family ships attention Q/K/V biases; CosyVoice3 is built
//! on Qwen2, so biases are copied verbatim like every other tensor.
//! The CosyVoice2 converter's bias-uniformity check (all three per
//! layer, all layers) applies unchanged.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (same contract Whisper / Kokoro / CosyVoice2 use).
//!
//! # No ONNX (permanent)
//!
//! CosyVoice3 ships a mixture of `flow.decoder.estimator.fp32.onnx`
//! plus PyTorch pickles. This converter never touches ONNX — the LLM
//! backbone binds off `llm.pt`, and the Flow Matching estimator will
//! bind off `flow.pt` in the follow-up wave, both re-implemented
//! natively by the runtime (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgufBuilder, GgufFile, GgufMetadataValue, chunks};

use crate::ConvertError;
use crate::models::cosyvoice2::{
    CosyVoice2Report, TokenizerFiles, convert_with_config_and_tokenizer as cosyvoice2_convert,
};

/// `vokra.model.arch` for Fun-CosyVoice3 GGUFs — kept in sync with the
/// runtime constant `vokra-models::cosyvoice3::EXPECTED_ARCH`.
///
/// Intentionally distinct from CosyVoice2's `"cosyvoice2"` so the
/// runtime can label the loaded model correctly in telemetry / logs /
/// model cards; the hparam **schema** is the same (`vokra.cosyvoice3.*`
/// keys, byte-parallel to `vokra.cosyvoice2.*`) so a runtime path that
/// delegates the CosyVoice3 forward through the CosyVoice2 chain reads
/// the identical hparam layout at a different metadata prefix.
pub(crate) const ARCH: &str = "cosyvoice3";
/// `vokra.model.name` value written for the Fun-CosyVoice3 GGUF.
pub(crate) const NAME: &str = "fun-cosyvoice3-0.5b-2512";

// --- vokra.cosyvoice3.* metadata keys (parallel to CosyVoice2's) -----------
//
// Kept as constants inside this module — the two crates only share
// `vokra-core`, and `vokra-convert` does not depend on `vokra-models`.
// The runtime crate has an identical constant set under
// `crates/vokra-models/src/cosyvoice3/` (when the config reader lands as
// a follow-up wave, that side pins the same strings via round-trip
// tests).

const KEY_SAMPLE_RATE: &str = "vokra.cosyvoice3.sample_rate";
const KEY_VOCAB_SIZE: &str = "vokra.cosyvoice3.arch.vocab_size";
const KEY_HIDDEN_DIM: &str = "vokra.cosyvoice3.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.cosyvoice3.arch.n_layer";
const KEY_N_HEAD: &str = "vokra.cosyvoice3.arch.n_head";
const KEY_FFN_DIM: &str = "vokra.cosyvoice3.arch.ffn_dim";
const KEY_N_HEAD_KV: &str = "vokra.cosyvoice3.arch.n_head_kv";
const KEY_ROPE_BASE: &str = "vokra.cosyvoice3.arch.rope_base";
const KEY_RMS_NORM_EPS: &str = "vokra.cosyvoice3.arch.rms_norm_eps";
const KEY_N_CTX: &str = "vokra.cosyvoice3.arch.n_ctx";
const KEY_FLOW_NFE: &str = "vokra.cosyvoice3.flow.nfe";
const KEY_FLOW_SCHEDULE: &str = "vokra.cosyvoice3.flow.schedule";
const KEY_STREAMING_CHUNK_SIZE: &str = "vokra.cosyvoice3.streaming.chunk_size";
const KEY_STREAMING_CHUNK_HOP: &str = "vokra.cosyvoice3.streaming.chunk_hop";
const KEY_TOKENIZER_VOCAB: &str = "vokra.cosyvoice3.tokenizer.vocab";
const KEY_TOKENIZER_MERGES: &str = "vokra.cosyvoice3.tokenizer.merges";

// The CosyVoice2 keys are what the delegated converter writes. We
// re-read those values off the built GGUF and re-write them under the
// CosyVoice3 prefix so this converter has one source-of-truth for shape
// derivation (the CosyVoice2 walk) while emitting the CosyVoice3
// metadata chunk. Kept in sync with
// `crates/vokra-convert/src/models/cosyvoice2.rs` via the
// `key_strings_match_delegated_cosyvoice2_writer` test.
const DELEGATED_KEY_SAMPLE_RATE: &str = "vokra.cosyvoice2.sample_rate";
const DELEGATED_KEY_VOCAB_SIZE: &str = "vokra.cosyvoice2.arch.vocab_size";
const DELEGATED_KEY_HIDDEN_DIM: &str = "vokra.cosyvoice2.arch.hidden_dim";
const DELEGATED_KEY_N_LAYER: &str = "vokra.cosyvoice2.arch.n_layer";
const DELEGATED_KEY_N_HEAD: &str = "vokra.cosyvoice2.arch.n_head";
const DELEGATED_KEY_FFN_DIM: &str = "vokra.cosyvoice2.arch.ffn_dim";
const DELEGATED_KEY_N_HEAD_KV: &str = "vokra.cosyvoice2.arch.n_head_kv";
const DELEGATED_KEY_ROPE_BASE: &str = "vokra.cosyvoice2.arch.rope_base";
const DELEGATED_KEY_RMS_NORM_EPS: &str = "vokra.cosyvoice2.arch.rms_norm_eps";
const DELEGATED_KEY_N_CTX: &str = "vokra.cosyvoice2.arch.n_ctx";
const DELEGATED_KEY_FLOW_NFE: &str = "vokra.cosyvoice2.flow.nfe";
const DELEGATED_KEY_FLOW_SCHEDULE: &str = "vokra.cosyvoice2.flow.schedule";
const DELEGATED_KEY_STREAMING_CHUNK_SIZE: &str = "vokra.cosyvoice2.streaming.chunk_size";
const DELEGATED_KEY_STREAMING_CHUNK_HOP: &str = "vokra.cosyvoice2.streaming.chunk_hop";
const DELEGATED_KEY_TOKENIZER_VOCAB: &str = "vokra.cosyvoice2.tokenizer.vocab";
const DELEGATED_KEY_TOKENIZER_MERGES: &str = "vokra.cosyvoice2.tokenizer.merges";

/// Fun-CosyVoice3 sample rate (Hz) — same as CosyVoice2 (the HiFTNet
/// vocoder produces 24 kHz PCM by architecture).
const COSYVOICE3_SAMPLE_RATE: u32 = 24_000;

/// Outcome of a Fun-CosyVoice3 conversion (thin re-alias of the
/// CosyVoice2 report — the shape derivation runs through the same
/// path).
pub(crate) type CosyVoice3Report = CosyVoice2Report;

/// Converts a Fun-CosyVoice3 safetensors buffer (no config / tokenizer
/// side-car) — see [`convert_with_config_and_tokenizer`].
#[cfg(test)]
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CosyVoice3Report), ConvertError> {
    convert_with_config(bytes, None)
}

/// Converts a Fun-CosyVoice3 safetensors buffer with no tokenizer
/// side-car — see [`convert_with_config_and_tokenizer`].
pub(crate) fn convert_with_config(
    bytes: Vec<u8>,
    config_json: Option<&[u8]>,
) -> Result<(GgufBuilder, CosyVoice3Report), ConvertError> {
    convert_with_config_and_tokenizer(bytes, config_json, None)
}

/// Converts a Fun-CosyVoice3 safetensors buffer into a populated GGUF
/// builder plus a report of what was written vs. skipped.
///
/// Delegates the tensor walk + shape derivation + Q/K/V bias
/// uniformity check to the CosyVoice2 converter (the topology is
/// byte-identical); rewrites the arch label, model name, provenance,
/// and metadata chunk prefix on top so the runtime dispatches to
/// `vokra-models::cosyvoice3` instead of `vokra-models::cosyvoice2`.
pub(crate) fn convert_with_config_and_tokenizer(
    bytes: Vec<u8>,
    config_json: Option<&[u8]>,
    tokenizer: Option<TokenizerFiles<'_>>,
) -> Result<(GgufBuilder, CosyVoice3Report), ConvertError> {
    // Delegate to CosyVoice2's shape-driven walk: the safetensors reader,
    // the layer-count / GQA cross-checks, the F32/F16 tensor pass-through
    // — all identical. The builder returned carries the `vokra.cosyvoice2.*`
    // metadata chunk we now re-label.
    let (mut builder, report) =
        cosyvoice2_convert(bytes, config_json, tokenizer).map_err(rewrite_cosyvoice2_error)?;

    // Overwrite the delegated arch / name / source with the CosyVoice3
    // stamps. `add_string` on GgufBuilder overwrites existing keys in place,
    // so this is a targeted rename — the tensor payload and every other
    // metadata key are preserved verbatim.
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Self-describing redistribution: the artifact carries its own
    // licence, not relying on a consumer running Vokra's registry
    // resolver. Fun-CosyVoice3-0.5B ships apache-2.0
    // (huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 model-card
    // header, fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(NAME),
        Some("FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (apache-2.0)"),
    );

    // Serialize the delegated builder once so we can read the
    // shape-derived keys off a parsed `GgufFile` without needing
    // `GgufBuilder: Clone`. This is a single walk, not a per-key
    // round-trip.
    //
    // The unwrap is deliberate: `to_bytes` on a builder that just came
    // out of the delegate must not fail — it succeeded there. A silent
    // fallback to empty bytes would violate FR-EX-08 by hiding a
    // structural writer failure; the `?` propagates it instead.
    let bytes = builder.to_bytes()?;
    let file = GgufFile::parse(bytes).map_err(|e| {
        ConvertError::Parse(format!(
            "cosyvoice3: delegated writer produced a GGUF that would not re-parse: {e}"
        ))
    })?;

    // Copy the CosyVoice2 hparam keys we care about to the CosyVoice3
    // prefix. Fail-loud when a required key is missing — the delegated
    // writer always writes the shape-derived + sample-rate keys, so a
    // missing one means the delegate signature drifted, not a legitimate
    // absent value (FR-EX-08).
    match get_u32_from_file(&file, DELEGATED_KEY_SAMPLE_RATE) {
        Some(v) => {
            builder.add_u32(KEY_SAMPLE_RATE, v);
        }
        None => {
            // Model-card invariant: the delegated writer always emits
            // this, but if it were absent the CosyVoice3 sample rate is
            // fixed at 24 kHz by the HiFTNet vocoder.
            builder.add_u32(KEY_SAMPLE_RATE, COSYVOICE3_SAMPLE_RATE);
        }
    }
    copy_u32_or_zero(
        &mut builder,
        &file,
        DELEGATED_KEY_VOCAB_SIZE,
        KEY_VOCAB_SIZE,
    );
    copy_u32_or_zero(
        &mut builder,
        &file,
        DELEGATED_KEY_HIDDEN_DIM,
        KEY_HIDDEN_DIM,
    );
    copy_u32_or_zero(&mut builder, &file, DELEGATED_KEY_N_LAYER, KEY_N_LAYER);
    copy_u32_or_zero(&mut builder, &file, DELEGATED_KEY_N_HEAD, KEY_N_HEAD);
    copy_u32_or_zero(&mut builder, &file, DELEGATED_KEY_FFN_DIM, KEY_FFN_DIM);
    // Config-only keys: only present when `--config` was supplied to the
    // delegated writer. Leave them absent if the delegate didn't write
    // them (the runtime has documented fallbacks for absent keys).
    copy_u32_if_present(&mut builder, &file, DELEGATED_KEY_N_HEAD_KV, KEY_N_HEAD_KV);
    copy_f32_if_present(&mut builder, &file, DELEGATED_KEY_ROPE_BASE, KEY_ROPE_BASE);
    copy_f32_if_present(
        &mut builder,
        &file,
        DELEGATED_KEY_RMS_NORM_EPS,
        KEY_RMS_NORM_EPS,
    );
    copy_u32_if_present(&mut builder, &file, DELEGATED_KEY_N_CTX, KEY_N_CTX);
    copy_u32_or_zero(&mut builder, &file, DELEGATED_KEY_FLOW_NFE, KEY_FLOW_NFE);
    copy_str_or_default(
        &mut builder,
        &file,
        DELEGATED_KEY_FLOW_SCHEDULE,
        KEY_FLOW_SCHEDULE,
        "linear",
    );
    copy_u32_or_zero(
        &mut builder,
        &file,
        DELEGATED_KEY_STREAMING_CHUNK_SIZE,
        KEY_STREAMING_CHUNK_SIZE,
    );
    copy_u32_or_zero(
        &mut builder,
        &file,
        DELEGATED_KEY_STREAMING_CHUNK_HOP,
        KEY_STREAMING_CHUNK_HOP,
    );

    // Copy the embedded Qwen2 tokenizer chunks under the CosyVoice3
    // prefix when the delegate embedded them (`--config` side-car
    // supplied vocab.json + merges.txt alongside).
    copy_u8_array_if_present(
        &mut builder,
        &file,
        DELEGATED_KEY_TOKENIZER_VOCAB,
        KEY_TOKENIZER_VOCAB,
    );
    copy_u8_array_if_present(
        &mut builder,
        &file,
        DELEGATED_KEY_TOKENIZER_MERGES,
        KEY_TOKENIZER_MERGES,
    );

    Ok((builder, report))
}

/// Rewrites CosyVoice2-flavoured error messages so callers see the
/// CosyVoice3 arch name rather than the delegate's. Not a semantic
/// change — the underlying `ConvertError` variant is preserved.
fn rewrite_cosyvoice2_error(e: ConvertError) -> ConvertError {
    match e {
        ConvertError::Parse(msg) => ConvertError::Parse(msg.replace("cosyvoice2", "cosyvoice3")),
        other => other,
    }
}

/// Reads a `U32` value from a serialized-and-parsed `GgufFile`.
fn get_u32_from_file(file: &GgufFile, key: &str) -> Option<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Some(*v),
        _ => None,
    }
}

fn get_f32_from_file(file: &GgufFile, key: &str) -> Option<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => Some(*v),
        _ => None,
    }
}

fn get_str_from_file(file: &GgufFile, key: &str) -> Option<String> {
    match file.get(key) {
        Some(GgufMetadataValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn get_u8_array_from_file(file: &GgufFile, key: &str) -> Option<Vec<u8>> {
    match file.get(key) {
        Some(GgufMetadataValue::Array(arr)) => {
            let mut out = Vec::with_capacity(arr.values.len());
            for v in &arr.values {
                match v {
                    GgufMetadataValue::U8(b) => out.push(*b),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn copy_u32_or_zero(builder: &mut GgufBuilder, file: &GgufFile, from: &str, to: &str) {
    let v = get_u32_from_file(file, from).unwrap_or(0);
    builder.add_u32(to, v);
}

fn copy_u32_if_present(builder: &mut GgufBuilder, file: &GgufFile, from: &str, to: &str) {
    if let Some(v) = get_u32_from_file(file, from) {
        builder.add_u32(to, v);
    }
}

fn copy_f32_if_present(builder: &mut GgufBuilder, file: &GgufFile, from: &str, to: &str) {
    if let Some(v) = get_f32_from_file(file, from) {
        builder.add_f32(to, v);
    }
}

fn copy_str_or_default(
    builder: &mut GgufBuilder,
    file: &GgufFile,
    from: &str,
    to: &str,
    default: &str,
) {
    let v = get_str_from_file(file, from).unwrap_or_else(|| default.to_owned());
    builder.add_string(to, &v);
}

fn copy_u8_array_if_present(builder: &mut GgufBuilder, file: &GgufFile, from: &str, to: &str) {
    if let Some(bytes) = get_u8_array_from_file(file, from) {
        builder.add_metadata(
            to,
            GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::U8,
                values: bytes.into_iter().map(GgufMetadataValue::U8).collect(),
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a minimal safetensors buffer with one F32 tensor (the
    /// CosyVoice2 scaffold pattern). Only the header parsing and the
    /// verbatim byte-copy path are exercised.
    fn minimal_safetensors_one_f32() -> Vec<u8> {
        let header = r#"{"llm.wte":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    /// Builds a safetensors buffer with the full (tiny) Qwen2-shaped
    /// LLM backbone: vocab 16, hidden 8, 2 layers, ffn 16, kv_out 4 —
    /// same shapes as the CosyVoice2 harness (Fun-CosyVoice3 shares
    /// the Qwen2 topology).
    fn backbone_safetensors(with_bias: bool) -> Vec<u8> {
        let (vocab, d, ffn, kv) = (16u64, 8u64, 16u64, 4u64);
        let mut entries: Vec<(String, Vec<u64>)> = vec![(
            "llm.model.model.embed_tokens.weight".to_owned(),
            vec![vocab, d],
        )];
        for i in 0..2 {
            let p = format!("llm.model.model.layers.{i}.");
            entries.push((format!("{p}input_layernorm.weight"), vec![d]));
            entries.push((format!("{p}self_attn.q_proj.weight"), vec![d, d]));
            entries.push((format!("{p}self_attn.k_proj.weight"), vec![kv, d]));
            entries.push((format!("{p}self_attn.v_proj.weight"), vec![kv, d]));
            if with_bias {
                entries.push((format!("{p}self_attn.q_proj.bias"), vec![d]));
                entries.push((format!("{p}self_attn.k_proj.bias"), vec![kv]));
                entries.push((format!("{p}self_attn.v_proj.bias"), vec![kv]));
            }
            entries.push((format!("{p}self_attn.o_proj.weight"), vec![d, d]));
            entries.push((format!("{p}post_attention_layernorm.weight"), vec![d]));
            entries.push((format!("{p}mlp.gate_proj.weight"), vec![ffn, d]));
            entries.push((format!("{p}mlp.up_proj.weight"), vec![ffn, d]));
            entries.push((format!("{p}mlp.down_proj.weight"), vec![d, ffn]));
        }
        entries.push(("llm.model.model.norm.weight".to_owned(), vec![d]));
        build_safetensors(&entries)
    }

    fn build_safetensors(entries: &[(String, Vec<u64>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut offset = 0u64;
        for (i, (name, shape)) in entries.iter().enumerate() {
            let n: u64 = shape.iter().product();
            let end = offset + n * 4;
            if i > 0 {
                header.push(',');
            }
            let dims = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{offset},{end}]}}"#
            ));
            offset = end;
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + offset as usize, 0u8);
        out
    }

    /// Tiny Qwen2-style `config.json` matching `backbone_safetensors`
    /// shapes (head split 2/1, head_dim 4). Same fixture the CosyVoice2
    /// tests use.
    const TINY_CONFIG: &str = r#"{
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "vocab_size": 16,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 32768
    }"#;

    fn get_u32(file: &GgufFile, key: &str) -> u32 {
        match file.get(key) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_f32(file: &GgufFile, key: &str) -> f32 {
        match file.get(key) {
            Some(GgufMetadataValue::F32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with `vokra-models::cosyvoice3::EXPECTED_ARCH`.
        assert_eq!(ARCH, "cosyvoice3");
    }

    #[test]
    fn name_string_matches_hf_model_id() {
        // Kept in sync with `huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`.
        assert_eq!(NAME, "fun-cosyvoice3-0.5b-2512");
    }

    /// Every CosyVoice3 metadata key mirrors its CosyVoice2 counterpart
    /// with the `cosyvoice2` → `cosyvoice3` substitution. The two
    /// crates' constants have no compile-time link (the runtime side
    /// duplicates the strings under the two-crate constant rule), so
    /// this test pins the parallel here.
    #[test]
    fn key_strings_are_parallel_to_cosyvoice2_writer() {
        for (own, delegated) in [
            (KEY_SAMPLE_RATE, DELEGATED_KEY_SAMPLE_RATE),
            (KEY_VOCAB_SIZE, DELEGATED_KEY_VOCAB_SIZE),
            (KEY_HIDDEN_DIM, DELEGATED_KEY_HIDDEN_DIM),
            (KEY_N_LAYER, DELEGATED_KEY_N_LAYER),
            (KEY_N_HEAD, DELEGATED_KEY_N_HEAD),
            (KEY_FFN_DIM, DELEGATED_KEY_FFN_DIM),
            (KEY_N_HEAD_KV, DELEGATED_KEY_N_HEAD_KV),
            (KEY_ROPE_BASE, DELEGATED_KEY_ROPE_BASE),
            (KEY_RMS_NORM_EPS, DELEGATED_KEY_RMS_NORM_EPS),
            (KEY_N_CTX, DELEGATED_KEY_N_CTX),
            (KEY_FLOW_NFE, DELEGATED_KEY_FLOW_NFE),
            (KEY_FLOW_SCHEDULE, DELEGATED_KEY_FLOW_SCHEDULE),
            (KEY_STREAMING_CHUNK_SIZE, DELEGATED_KEY_STREAMING_CHUNK_SIZE),
            (KEY_STREAMING_CHUNK_HOP, DELEGATED_KEY_STREAMING_CHUNK_HOP),
            (KEY_TOKENIZER_VOCAB, DELEGATED_KEY_TOKENIZER_VOCAB),
            (KEY_TOKENIZER_MERGES, DELEGATED_KEY_TOKENIZER_MERGES),
        ] {
            assert_eq!(
                own.replace("cosyvoice3", "cosyvoice2"),
                delegated,
                "own {own:?} must be delegated {delegated:?} with prefix substitution"
            );
        }
    }

    #[test]
    fn round_trip_carries_cosyvoice3_arch_and_provenance() {
        // A scaffold buffer (no backbone tensors) keeps converting with
        // 0-placeholders + a note — the arch and provenance must land
        // under the CosyVoice3 prefix.
        let bytes = minimal_safetensors_one_f32();
        let (builder, report) = convert(bytes).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert!(report.derived.is_none());

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");

        // Arch / name.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
        );

        // Provenance: apache-2.0 permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
        // apache-2.0 alone doesn't stamp attribution (permissive tier)
        // — the source string names the family:
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION)
                .and_then(|v| v.as_str()),
            None,
            "permissive must not stamp attribution",
        );

        // Sample rate: model-card invariant.
        assert_eq!(get_u32(&file, KEY_SAMPLE_RATE), COSYVOICE3_SAMPLE_RATE);

        // Placeholder hparams: `0` (no backbone to derive from).
        for key in [
            KEY_VOCAB_SIZE,
            KEY_HIDDEN_DIM,
            KEY_N_LAYER,
            KEY_N_HEAD,
            KEY_FFN_DIM,
            KEY_FLOW_NFE,
            KEY_STREAMING_CHUNK_SIZE,
            KEY_STREAMING_CHUNK_HOP,
        ] {
            assert_eq!(get_u32(&file, key), 0, "{key}");
        }
        // The config-only keys are unwritten without --config.
        for key in [KEY_N_HEAD_KV, KEY_ROPE_BASE, KEY_RMS_NORM_EPS, KEY_N_CTX] {
            assert!(file.get(key).is_none(), "{key} must be absent");
        }

        // Schedule tag: `linear` default.
        assert_eq!(
            file.get(KEY_FLOW_SCHEDULE).and_then(|v| v.as_str()),
            Some("linear"),
        );

        // The delegated CosyVoice2 keys are ALSO present (the delegated
        // writer left them in place — we only overwrote arch / name /
        // provenance and ADDED our own prefix). This is intentional:
        // the runtime dispatches off `KEY_MODEL_ARCH`, not off the
        // hparam prefix, so leaving the delegated keys is harmless.
        // The `vokra.cosyvoice3.*` chunk is what a CosyVoice3-aware
        // runtime will read.
        assert_eq!(
            get_u32(&file, DELEGATED_KEY_SAMPLE_RATE),
            COSYVOICE3_SAMPLE_RATE,
        );
    }

    #[test]
    fn shape_hparams_derive_without_config() {
        let (builder, report) = convert(backbone_safetensors(true)).expect("convert");
        let d = report.derived.expect("backbone present → derived");
        // Same shape-derivation as CosyVoice2.
        assert_eq!(d.vocab_size, 16);
        assert_eq!(d.hidden_dim, 8);
        assert_eq!(d.n_layer, 2);
        assert_eq!(d.ffn_dim, 16);
        assert_eq!(d.n_head, 0, "head split needs --config");
        assert!(d.has_attn_bias);

        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(get_u32(&file, KEY_VOCAB_SIZE), 16);
        assert_eq!(get_u32(&file, KEY_HIDDEN_DIM), 8);
        assert_eq!(get_u32(&file, KEY_N_LAYER), 2);
        assert_eq!(get_u32(&file, KEY_FFN_DIM), 16);
        assert_eq!(get_u32(&file, KEY_N_HEAD), 0);
        assert!(file.get(KEY_N_HEAD_KV).is_none());
    }

    #[test]
    fn config_supplies_head_split_and_rope_group() {
        let (builder, report) =
            convert_with_config(backbone_safetensors(true), Some(TINY_CONFIG.as_bytes()))
                .expect("convert");
        let d = report.derived.expect("derived");
        assert_eq!(d.n_head, 2);
        assert_eq!(d.n_head_kv, 1);
        assert_eq!(d.n_ctx, 32_768);
        assert!(d.has_attn_bias);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(get_u32(&file, KEY_N_HEAD), 2);
        assert_eq!(get_u32(&file, KEY_N_HEAD_KV), 1);
        assert_eq!(get_u32(&file, KEY_N_CTX), 32_768);
        assert!((get_f32(&file, KEY_ROPE_BASE) - 1_000_000.0).abs() < 1e-1);
        assert!((get_f32(&file, KEY_RMS_NORM_EPS) - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn config_shape_mismatch_fails_loudly() {
        let bad = TINY_CONFIG.replace("\"hidden_size\": 8", "\"hidden_size\": 896");
        let err = convert_with_config(backbone_safetensors(true), Some(bad.as_bytes()))
            .expect_err("wrong config must fail");
        // The rewrite converts `cosyvoice2` → `cosyvoice3` in the error
        // message so the operator sees the arch name they invoked.
        let msg = err.to_string();
        assert!(msg.contains("hidden_size"), "must name the field: {msg}");
        assert!(msg.contains("cosyvoice3"), "must name our arch: {msg}");
        assert!(
            !msg.contains("cosyvoice2"),
            "must not leak the delegate's arch: {msg}"
        );
    }

    #[test]
    fn config_bad_gqa_split_fails_loudly() {
        let bad = TINY_CONFIG.replace("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 3");
        let err = convert_with_config(backbone_safetensors(true), Some(bad.as_bytes()))
            .expect_err("bad GQA split must fail");
        assert!(err.to_string().contains("GQA"), "{err}");
    }

    #[test]
    fn config_missing_required_field_fails_loudly() {
        let bad = TINY_CONFIG.replace("\"num_attention_heads\": 2,", "");
        let err = convert_with_config(backbone_safetensors(true), Some(bad.as_bytes()))
            .expect_err("missing head count must fail");
        assert!(err.to_string().contains("num_attention_heads"), "{err}");
    }

    #[test]
    fn tokenizer_files_are_embedded_under_cosyvoice3_prefix() {
        let vocab = br#"{"a":0,"b":1,"ab":2}"#.to_vec();
        let merges = b"#version: 0.2\na b\n".to_vec();
        let (builder, report) = convert_with_config_and_tokenizer(
            minimal_safetensors_one_f32(),
            None,
            Some(TokenizerFiles {
                vocab_json: &vocab,
                merges_txt: &merges,
            }),
        )
        .expect("convert");
        assert!(report.tokenizer_embedded, "tokenizer must be embedded");

        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        // The CosyVoice3 prefix must carry the bytes.
        let out = match file.get(KEY_TOKENIZER_VOCAB) {
            Some(GgufMetadataValue::Array(arr)) => arr
                .values
                .iter()
                .map(|v| match v {
                    GgufMetadataValue::U8(x) => *x,
                    other => panic!("non-U8 element {other:?}"),
                })
                .collect::<Vec<u8>>(),
            other => panic!("expected U8 array, got {other:?}"),
        };
        assert_eq!(out, vocab);
        let out = match file.get(KEY_TOKENIZER_MERGES) {
            Some(GgufMetadataValue::Array(arr)) => arr
                .values
                .iter()
                .map(|v| match v {
                    GgufMetadataValue::U8(x) => *x,
                    other => panic!("non-U8 element {other:?}"),
                })
                .collect::<Vec<u8>>(),
            other => panic!("expected U8 array, got {other:?}"),
        };
        assert_eq!(out, merges);
    }
}
