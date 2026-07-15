//! M4-04 T10/T11 — codec GGUF converter ⇄ runtime binder roundtrip.
//!
//! Builds a synthetic (but structurally faithful) upstream checkpoint, runs
//! it through the **public** `vokra-convert` entry points
//! (`convert_file(ModelKind::Mimi, …)` / `convert_dac_file(…)`), loads the
//! GGUF back, binds it with the `vokra_models::codec` binders and asserts:
//!
//! 1. the bound tables/projections are **bit-identical** to the same
//!    derivation computed independently here (converter math check);
//! 2. `mimi_rvq_decode` / `dac_rvq_decode` over the bound inputs match a
//!    hand fold (end-to-end decode through converter-produced data).
//!
//! Real-checkpoint parity (full 32-codebook tables) is the
//! `parity-rvq-real.yml` workflow (T16, owner-dispatched); PR CI runs this
//! synthetic roundtrip + the committed sliced-fixture reference tests.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_dac_file, convert_file};
use vokra_core::gguf::GgufFile;
use vokra_models::codec::{DacCodecGguf, MimiCodecGguf};
use vokra_ops::{dac_rvq_decode, mimi_rvq_decode};

fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vokra-codec-rt-{tag}-{}", std::process::id()));
    p
}

fn build_safetensors(entries: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut data = Vec::<u8>::new();
    for (i, (name, shape, vals)) in entries.iter().enumerate() {
        let start = data.len();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        if i > 0 {
            header.push(',');
        }
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{start},{end}]}}"#
        ));
    }
    header.push('}');
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

// ---------------------------------------------------------------------------
// Mimi
// ---------------------------------------------------------------------------

/// Synthetic moshi-native Mimi: 1 semantic + 2 acoustic, cb_size 4, dim 2,
/// d_model 3. Returns (safetensors bytes, per-split proj, per-cb (sum, usage)).
#[allow(clippy::type_complexity)]
fn synthetic_mimi() -> (Vec<u8>, [Vec<f32>; 2], Vec<(Vec<f32>, Vec<f32>)>) {
    let first_proj: Vec<f32> = (0..3)
        .flat_map(|o| (0..2).map(move |c| (o + 1) as f32 * 0.25 - c as f32 * 0.5))
        .collect();
    let rest_proj: Vec<f32> = first_proj.iter().map(|x| x * 2.0).collect();

    let mut per_cb = Vec::new();
    let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
        (
            "quantizer.rvq_first.output_proj.weight".into(),
            vec![3, 2, 1],
            first_proj.clone(),
        ),
        (
            "quantizer.rvq_rest.output_proj.weight".into(),
            vec![3, 2, 1],
            rest_proj.clone(),
        ),
    ];
    for (split, layer, salt) in [
        ("rvq_first", 0usize, 1.0f32),
        ("rvq_rest", 0, 2.0),
        ("rvq_rest", 1, 3.0),
    ] {
        let base = format!("quantizer.{split}.vq.layers.{layer}._codebook");
        let sum: Vec<f32> = (0..4 * 2).map(|i| (i as f32 - 3.0) * salt).collect();
        let usage: Vec<f32> = vec![1.0, 0.5, 0.0, 2.0];
        entries.push((format!("{base}.embedding_sum"), vec![4, 2], sum.clone()));
        entries.push((format!("{base}.cluster_usage"), vec![4], usage.clone()));
        per_cb.push((sum, usage));
    }
    (build_safetensors(&entries), [first_proj, rest_proj], per_cb)
}

#[test]
#[allow(clippy::needless_range_loop)] // index-form hand fold — mirrors the op's math 1:1
fn mimi_convert_load_bind_roundtrip_is_bit_identical() {
    let (st_bytes, projs, per_cb) = synthetic_mimi();
    let input = tmp_path("mimi-in");
    let output = tmp_path("mimi-out");
    std::fs::write(&input, st_bytes).expect("write input");

    let summary = convert_file(ModelKind::Mimi, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::Mimi);

    let file = GgufFile::open(&output).expect("open gguf");
    let codec = MimiCodecGguf::from_gguf(&file).expect("bind");
    assert_eq!(codec.attrs.n_codebooks, 3);
    assert_eq!(codec.attrs.codebook_size, 4);
    assert_eq!(codec.attrs.d_model, 3);

    // Independent re-derivation (same math, written separately):
    // table[cb][i,o] = Σ_c W[o,c] * (sum[i,c] / max(usage[i], 1e-5)).
    for cb in 0..3 {
        let w = if cb == 0 { &projs[0] } else { &projs[1] };
        let (sum, usage) = &per_cb[cb];
        for i in 0..4 {
            let denom = usage[i].max(1e-5);
            for o in 0..3 {
                let mut acc = 0.0_f32;
                for c in 0..2 {
                    acc += w[o * 2 + c] * (sum[i * 2 + c] / denom);
                }
                let got = codec.tables[cb].data[i * 3 + o];
                assert_eq!(got, acc, "table[{cb}][{i},{o}] must be bit-identical");
            }
        }
    }

    // End-to-end decode through the bound tables matches a hand fold.
    let codes = vec![0u32, 1, 2, 3, 2, 1];
    let out = mimi_rvq_decode(&codes, 2, &codec.tables, &codec.attrs).expect("decode");
    let mut want = vec![0.0_f32; 2 * 3];
    for t in 0..2 {
        for cb in 0..3 {
            let idx = codes[t * 3 + cb] as usize;
            for o in 0..3 {
                want[t * 3 + o] += codec.tables[cb].data[idx * 3 + o];
            }
        }
    }
    assert_eq!(out, want);

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

// ---------------------------------------------------------------------------
// DAC
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::needless_range_loop)] // index-form hand fold — mirrors the op's math 1:1
fn dac_convert_load_bind_roundtrip_is_bit_identical() {
    // 2 quantizers, cb_size 3, codebook_dim 2, d_model 4.
    let mut entries: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut expected_folded: Vec<Vec<f32>> = Vec::new();
    for i in 0..2usize {
        let prefix = format!("quantizer.quantizers.{i}");
        let cb: Vec<f32> = (0..3 * 2).map(|k| k as f32 * 0.5 - i as f32).collect();
        entries.push((format!("{prefix}.codebook.weight"), vec![3, 2], cb));

        // v rows [o]: [1+o, 2+o]; g[o] = 2.0 → W[o,:] = 2*[1+o,2+o]/||·||₂.
        let mut v = Vec::new();
        let mut folded = Vec::new();
        for o in 0..4 {
            let row = [1.0 + o as f32, 2.0 + o as f32];
            v.extend_from_slice(&row);
            let norm = (row[0] * row[0] + row[1] * row[1]).sqrt();
            let scale = 2.0 / norm;
            folded.push(row[0] * scale);
            folded.push(row[1] * scale);
        }
        expected_folded.push(folded);
        entries.push((format!("{prefix}.out_proj.weight_v"), vec![4, 2, 1], v));
        entries.push((
            format!("{prefix}.out_proj.weight_g"),
            vec![4, 1, 1],
            vec![2.0; 4],
        ));
        entries.push((
            format!("{prefix}.out_proj.bias"),
            vec![4],
            (0..4).map(|o| o as f32 * 0.125 + i as f32).collect(),
        ));
    }

    let input = tmp_path("dac-in");
    let config = tmp_path("dac-cfg");
    let output = tmp_path("dac-out");
    std::fs::write(&input, build_safetensors(&entries)).expect("write input");
    std::fs::write(
        &config,
        br#"{"n_codebooks":2,"codebook_size":3,"codebook_dim":2,"d_model":4,"sample_rate":24000,"hop_length":320}"#,
    )
    .expect("write config");

    let summary = convert_dac_file(&input, &config, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::Dac);

    let file = GgufFile::open(&output).expect("open gguf");
    let codec = DacCodecGguf::from_gguf(&file).expect("bind");
    assert_eq!(codec.attrs.n_codebooks, 2);
    assert_eq!(codec.attrs.codebook_dim, 2);
    assert_eq!(codec.attrs.d_model, 4);
    assert_eq!(codec.sample_rate, 24000);
    assert_eq!(codec.hop_length, 320);

    // Folded weights bit-identical to the independent fold above.
    for i in 0..2 {
        assert_eq!(
            codec.out_projs[i].weight, expected_folded[i],
            "quantizer {i} folded weight"
        );
    }

    // End-to-end factorized decode through the bound inputs vs hand fold.
    let codes = vec![0u32, 2, 1, 0];
    let out =
        dac_rvq_decode(&codes, 2, &codec.tables, &codec.out_projs, &codec.attrs).expect("decode");
    let mut want = vec![0.0_f32; 2 * 4];
    for t in 0..2 {
        for cb in 0..2 {
            let idx = codes[t * 2 + cb] as usize;
            let low = &codec.tables[cb].data[idx * 2..(idx + 1) * 2];
            for o in 0..4 {
                let mut y = codec.out_projs[cb].bias[o];
                for c in 0..2 {
                    y += codec.out_projs[cb].weight[o * 2 + c] * low[c];
                }
                want[t * 4 + o] += y;
            }
        }
    }
    assert_eq!(out, want);

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}
