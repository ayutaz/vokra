//! Fixture-free FireRedVAD converter → native binder handshake.

use std::collections::BTreeMap;
use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file};
use vokra_models::firered_vad::{FireredVad, NATIVE_VARIANT};

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vokra-firered-vad-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ))
}

fn shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::from([
        ("firered_vad.cmvn.mean".to_owned(), vec![80]),
        ("firered_vad.cmvn.inverse_std".to_owned(), vec![80]),
        ("firered_vad.dfsmn.fc1.weight".to_owned(), vec![80, 256]),
        ("firered_vad.dfsmn.fc1.bias".to_owned(), vec![256]),
        ("firered_vad.dfsmn.fc2.weight".to_owned(), vec![256, 128]),
        ("firered_vad.dfsmn.fc2.bias".to_owned(), vec![128]),
        ("firered_vad.dfsmn.dnn.0.weight".to_owned(), vec![128, 256]),
        ("firered_vad.dfsmn.dnn.0.bias".to_owned(), vec![256]),
        ("firered_vad.output.weight".to_owned(), vec![256, 1]),
        ("firered_vad.output.bias".to_owned(), vec![1]),
    ]);
    for index in 0..8 {
        shapes.insert(
            format!("firered_vad.dfsmn.memory.{index}.weight"),
            vec![128, 1, 20],
        );
    }
    for index in 0..7 {
        shapes.insert(
            format!("firered_vad.dfsmn.block.{index}.fc1.weight"),
            vec![128, 256],
        );
        shapes.insert(
            format!("firered_vad.dfsmn.block.{index}.fc1.bias"),
            vec![256],
        );
        shapes.insert(
            format!("firered_vad.dfsmn.block.{index}.fc2.weight"),
            vec![256, 128],
        );
    }
    assert_eq!(shapes.len(), 39);
    shapes
}

fn canonical_safetensors() -> Vec<u8> {
    let mut entries = Vec::new();
    let mut data = Vec::new();
    for (name, shape) in shapes() {
        let start = data.len();
        let elements = shape.iter().product::<usize>();
        let value = if name == "firered_vad.cmvn.inverse_std" {
            1.0f32
        } else {
            0.0
        };
        for _ in 0..elements {
            data.extend_from_slice(&value.to_le_bytes());
        }
        entries.push(format!(
            "{name:?}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            data.len()
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut output = Vec::with_capacity(8 + header.len() + data.len());
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(&data);
    output
}

#[test]
fn converter_output_binds_and_runs_native_forward() {
    let input = temp_path("input.safetensors");
    let output = temp_path("output.gguf");
    std::fs::write(&input, canonical_safetensors()).expect("write fixture");
    let summary = convert_file(ModelKind::FireredVad, &input, &output).expect("convert");
    assert_eq!(summary.tensor_count, 39);

    let model = FireredVad::from_path(&output).expect("bind converted GGUF");
    let config = model.native_config().expect("native variant");
    assert_eq!(config.sample_rate, 16_000);
    assert_eq!(config.dfsmn.n_blocks, 8);
    assert_eq!(NATIVE_VARIANT, "stream-vad-dfsmn-v1");
    let probabilities = model.forward_features(&[0.0; 80]).expect("native forward");
    assert_eq!(probabilities, vec![0.5]);

    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}
