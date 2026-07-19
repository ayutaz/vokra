//! M4-06 T22 exit criterion — 合成 checkpoint の変換 → load → gate 通過が
//! e2e green: a synthetic BF16 Moshi checkpoint runs through the offline
//! `vokra-convert` library, and the produced GGUF loads through
//! `MoshiEngine::from_gguf_with_policy` under the **strict** compliance
//! policy (CC-BY 4.0 = `AttributionRequired` passes commercially, no
//! research flag), binds the real LM weights, surfaces the FR-MD-09
//! attribution, and decodes the embedded SentencePiece tokenizer.
//!
//! (Test-only edge vokra-models → vokra-convert: the M4-04 codec binder
//! roundtrip precedent; runtime dependency direction untouched.)

use vokra_core::CompliancePolicy;
use vokra_models::moshi::MoshiEngine;

/// Hand-encodes a minimal SentencePiece `ModelProto` with `n` pieces
/// (the tokenizer.rs test wire-writer — the runtime only ever reads).
fn spm_blob(n: usize) -> Vec<u8> {
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            out.push(b);
            if v == 0 {
                break;
            }
        }
    }
    let mut blob = Vec::new();
    for i in 0..n {
        let piece = format!("\u{2581}p{i}");
        let mut msg = Vec::new();
        msg.push(0x0a);
        varint(piece.len() as u64, &mut msg);
        msg.extend_from_slice(piece.as_bytes());
        msg.push(0x18); // type = NORMAL(1)
        msg.push(0x01);
        blob.push(0x0a);
        varint(msg.len() as u64, &mut blob);
        blob.extend_from_slice(&msg);
    }
    blob
}

/// A tiny synthetic Moshi checkpoint (BF16, upstream names — the
/// `MoshiConfig::tiny_for_tests` shape). Mirrors the converter's own
/// unit fixture; duplicated here because that helper is `#[cfg(test)]`
/// inside vokra-convert.
fn synthetic_checkpoint() -> Vec<u8> {
    let mut entries: Vec<(String, Vec<u64>)> = Vec::new();
    let (d, text, card) = (16u64, 13u64, 9u64);
    let (h_tm, d_dt, h_dt) = (8u64, 8u64, 6u64);
    entries.push(("text_emb.weight".into(), vec![text + 1, d]));
    entries.push(("text_linear.weight".into(), vec![text, d]));
    entries.push(("out_norm.alpha".into(), vec![1, 1, d]));
    for k in 0..4 {
        entries.push((format!("emb.{k}.weight"), vec![card + 1, d]));
    }
    for i in 0..2 {
        let p = format!("transformer.layers.{i}");
        entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d]));
        entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d]));
        entries.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * d, d]));
        entries.push((format!("{p}.self_attn.out_proj.weight"), vec![d, d]));
        entries.push((format!("{p}.gating.linear_in.weight"), vec![2 * h_tm, d]));
        entries.push((format!("{p}.gating.linear_out.weight"), vec![d, h_tm]));
    }
    for cb in 0..2 {
        entries.push((format!("depformer_in.{cb}.weight"), vec![d_dt, d]));
        entries.push((format!("linears.{cb}.weight"), vec![card, d_dt]));
    }
    entries.push(("depformer_text_emb.weight".into(), vec![text + 1, d_dt]));
    entries.push(("depformer_emb.0.weight".into(), vec![card + 1, d_dt]));
    for i in 0..2 {
        let p = format!("depformer.layers.{i}");
        entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d_dt]));
        entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d_dt]));
        entries.push((
            format!("{p}.self_attn.in_proj_weight"),
            vec![2 * 3 * d_dt, d_dt],
        ));
        entries.push((
            format!("{p}.self_attn.out_proj.weight"),
            vec![2 * d_dt, d_dt],
        ));
        for s in 0..2 {
            entries.push((
                format!("{p}.gating.{s}.linear_in.weight"),
                vec![2 * h_dt, d_dt],
            ));
            entries.push((
                format!("{p}.gating.{s}.linear_out.weight"),
                vec![d_dt, h_dt],
            ));
        }
    }

    // Deterministic non-constant BF16 payloads (an LCG over the bf16
    // pattern space keeps weights varied but finite: exponent clamped by
    // masking to small magnitudes).
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    let mut lcg = 0x1234_5678u32;
    for (i, (name, shape)) in entries.iter().enumerate() {
        let n: u64 = shape.iter().product();
        let start = data.len();
        for _ in 0..n {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            // Small-magnitude bf16: sign + fixed 0.0-0.5-ish exponent band.
            let frac = (lcg >> 16) as u16 & 0x007F;
            let sign = ((lcg >> 8) as u16) & 0x8000;
            let bf16 = sign | 0x3E00 | frac; // ±[0.125, 0.25) band
            data.extend_from_slice(&bf16.to_le_bytes());
        }
        let end = data.len();
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
            shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    header.push('}');
    let mut blob = Vec::new();
    blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
    blob.extend_from_slice(header.as_bytes());
    blob.extend_from_slice(&data);
    blob
}

#[test]
fn converted_gguf_loads_under_strict_policy_with_attribution_and_real_lm_weights() {
    let dir = std::env::temp_dir().join(format!("vokra-moshi-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ckpt = dir.join("model.safetensors");
    let tok = dir.join("tokenizer.model");
    let out = dir.join("moshi.gguf");
    std::fs::write(&ckpt, synthetic_checkpoint()).unwrap();
    std::fs::write(&tok, spm_blob(13)).unwrap();

    let summary = vokra_convert::convert_moshi_file(&ckpt, Some(&tok), &out).expect("convert");
    assert!(summary.tensor_count > 0);

    let bytes = std::fs::read(&out).unwrap();
    // The strict (fail-closed commercial) policy passes: CC-BY 4.0 is
    // AttributionRequired, NOT research-gated (T22 gate criterion).
    let engine = MoshiEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect("converted GGUF loads under the strict policy");

    // Real LM weights bound (not the synthesized fixture path).
    assert!(!engine.config().delays.is_empty());
    assert_eq!(engine.config().dep_q, 2);
    assert_eq!(engine.config().n_q_in, 4);
    assert_eq!(engine.config().delays, vec![0, 0, 1, 0, 1]);

    // FR-MD-09: the attribution surface is populated from the GGUF.
    let attribution = engine.attribution().expect("attribution present");
    assert!(attribution.text.contains("Kyutai"), "{}", attribution.text);
    assert!(attribution.text.contains("CC-BY 4.0"));

    // The embedded SentencePiece table decodes (13 pieces == text_card).
    // A short deterministic batch dialog exercises the full pipeline over
    // the converted weights (Mimi ends = documented synthesized bridge).
    use vokra_core::S2sEngine;
    use vokra_models::csm::EchoPath;
    let engine = engine.with_echo_path(EchoPath::BypassRecordedInput);
    let hop = engine.mimi_config().frame_hop_samples().unwrap();
    let input: Vec<f32> = (0..hop * 2)
        .map(|i| ((i as f32) * 0.05).sin() * 0.2)
        .collect();
    let request = vokra_core::DialogRequest::new("")
        .with_input_audio(input)
        .deterministic();
    let turn = engine
        .dialog(&request)
        .expect("dialog over converted weights");
    assert!(turn.audio.is_some());

    // ---- M4 cc-06: the mmap + mapped-lazy `from_path` load must be
    // observationally identical to the resident bytes load — same
    // attribution surface, same config, and a BIT-IDENTICAL deterministic
    // dialog turn (the mapped per-layer widening reproduces the resident
    // f32 values exactly; MappedTemporalBlocks docs).
    let mapped = MoshiEngine::from_path(&out)
        .expect("converted GGUF loads through the mmap + mapped-lazy path");
    let m_attr = mapped.attribution().expect("attribution via mmap load");
    assert!(m_attr.text.contains("Kyutai"), "{}", m_attr.text);
    assert_eq!(mapped.config().delays, vec![0, 0, 1, 0, 1]);
    let mapped = mapped.with_echo_path(EchoPath::BypassRecordedInput);
    let mapped_turn = mapped
        .dialog(&request)
        .expect("dialog over mapped-lazy weights");
    assert_eq!(
        mapped_turn.text, turn.text,
        "mapped vs resident monologue must be bit-identical"
    );
    assert_eq!(
        mapped_turn.audio.as_ref().map(|a| &a.samples),
        turn.audio.as_ref().map(|a| &a.samples),
        "mapped vs resident PCM must be bit-identical"
    );

    std::fs::remove_dir_all(&dir).ok();
}
