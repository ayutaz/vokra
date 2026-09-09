use std::path::{Component, Path, PathBuf};

use vokra_core::json::{self, JsonValue};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Conv1d,
    ConvTranspose1d,
}

pub struct Fixture {
    pub kind: Kind,
    pub input: Vec<f32>,
    pub input_shape: [usize; 3],
    pub weight: Vec<f32>,
    pub weight_shape: [usize; 3],
    pub bias: Vec<f32>,
    pub output: Vec<f32>,
    pub output_shape: [usize; 3],
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub output_padding: usize,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("vocoder_conv")
}

fn object<'a>(value: &'a JsonValue, context: &str) -> &'a JsonValue {
    assert!(value.as_object().is_some(), "{context} must be an object");
    value
}

fn exact_keys(value: &JsonValue, expected: &[&str], context: &str) {
    let entries = value
        .as_object()
        .unwrap_or_else(|| panic!("{context} must be an object"));
    assert_eq!(
        entries.len(),
        expected.len(),
        "{context} has unexpected key count"
    );
    for key in expected {
        assert!(
            entries.iter().any(|(actual, _)| actual == key),
            "{context} is missing or has no exact key {key:?}"
        );
    }
}

fn string(value: &JsonValue, key: &str) -> String {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .unwrap_or_else(|| panic!("manifest field {key:?} must be a string"))
        .to_owned()
}

fn number(value: &JsonValue, key: &str) -> usize {
    value
        .get(key)
        .and_then(JsonValue::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or_else(|| panic!("manifest field {key:?} must be a usize"))
}

fn shape(value: &JsonValue, key: &str) -> Vec<usize> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("manifest field {key:?} must be a 3-element array"));
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or_else(|| panic!("manifest field {key:?}[{index}] must be usize"))
        })
        .collect()
}

fn shape3(values: Vec<usize>, key: &str) -> [usize; 3] {
    assert_eq!(values.len(), 3, "manifest field {key:?} must have rank 3");
    values.try_into().unwrap_or_else(|_| unreachable!())
}

fn sha256_hex(data: &[u8]) -> String {
    // FIPS-180-4 SHA-256, kept here so fixture integrity does not depend on a
    // test-only external digest crate (the workspace is zero-dependency).
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut padded = data.to_vec();
    let bit_len = (padded.len() as u64) * 8;
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for block in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, bytes) in block.chunks_exact(4).take(16).enumerate() {
            w[index] = u32::from_be_bytes(bytes.try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            (hh, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = (*slot).wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

fn read_tensor(root: &Path, record: &JsonValue, role: &str) -> (Vec<f32>, Vec<usize>) {
    object(record, role);
    exact_keys(record, &["bytes", "dtype", "path", "sha256", "shape"], role);
    let relative = string(record, "path");
    let relative_path = Path::new(&relative);
    let mut components = relative_path.components();
    assert!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "fixture path {relative:?} must be one normal filename component"
    );
    assert_eq!(string(record, "dtype"), "float32", "{role} dtype");
    let shape = shape(record, "shape");
    let path = root.join(relative_path);
    let metadata = std::fs::symlink_metadata(&path)
        .unwrap_or_else(|error| panic!("stat {role} fixture {relative:?}: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "fixture {relative:?} must be a regular non-symlink file"
    );
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read {role} fixture {relative:?}: {error}"));
    assert_eq!(number(record, "bytes"), bytes.len(), "{role} byte count");
    assert_eq!(
        string(record, "sha256"),
        sha256_hex(&bytes),
        "{role} SHA-256"
    );
    let elements = shape.iter().product::<usize>();
    assert_eq!(bytes.len(), elements * 4, "{role} shape/byte count");
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (values, shape)
}

pub fn load(name: &str) -> Fixture {
    let root = fixture_root();
    let expected_files = [
        "README.md",
        "manifest.json",
        "conv1d_d2_s2_p2_bias.f32",
        "conv1d_d2_s2_p2_input.f32",
        "conv1d_d2_s2_p2_output.f32",
        "conv1d_d2_s2_p2_weight.f32",
        "conv_transpose1d_s3_p1_op2_bias.f32",
        "conv_transpose1d_s3_p1_op2_input.f32",
        "conv_transpose1d_s3_p1_op2_output.f32",
        "conv_transpose1d_s3_p1_op2_weight.f32",
    ];
    let mut actual_files = Vec::new();
    for entry in std::fs::read_dir(&root).expect("read vocoder Conv fixture directory") {
        let entry = entry.expect("read vocoder Conv fixture directory entry");
        let metadata =
            std::fs::symlink_metadata(entry.path()).expect("stat fixture directory entry");
        assert!(
            metadata.file_type().is_file(),
            "fixture directory entry {:?} must be a regular non-symlink file",
            entry.file_name()
        );
        actual_files.push(entry.file_name().to_string_lossy().into_owned());
    }
    actual_files.sort_unstable();
    let mut expected_files = expected_files
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    expected_files.sort_unstable();
    assert_eq!(actual_files, expected_files, "fixture directory file set");

    let manifest_path = root.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path).unwrap_or_else(|error| {
        panic!("read vocoder Conv fixture manifest {manifest_path:?}: {error}")
    });
    assert_eq!(
        sha256_hex(&manifest_bytes),
        "b438a28b6bc64754dc119d186080749775a78ad2b9345a5f20f480cdbcaa0c07",
        "outer manifest SHA-256"
    );
    let manifest = json::parse(&manifest_bytes).expect("vocoder Conv manifest must be JSON");
    exact_keys(&manifest, &["cases", "provenance", "schema"], "manifest");
    assert_eq!(string(&manifest, "schema"), "vokra-vocoder-conv-parity-v1");
    let provenance = object(
        manifest.get("provenance").expect("manifest provenance"),
        "provenance",
    );
    exact_keys(
        provenance,
        &[
            "byte_order",
            "dtype",
            "generator",
            "oracle",
            "randomness",
            "torch_version",
            "value_policy",
        ],
        "provenance",
    );
    assert_eq!(string(provenance, "oracle"), "PyTorch torch.nn.functional");
    assert_eq!(
        string(provenance, "generator"),
        "tools/parity/vocoder_conv_dump_reference.py"
    );
    assert_eq!(string(provenance, "randomness"), "none");
    assert_eq!(string(provenance, "value_policy"), "signed powers of two");
    assert_eq!(string(provenance, "dtype"), "float32");
    assert_eq!(string(provenance, "byte_order"), "little-endian");
    assert_eq!(string(provenance, "torch_version"), "2.13.0+cu130");

    let cases = object(manifest.get("cases").expect("manifest cases"), "cases");
    exact_keys(
        cases,
        &["conv1d_d2_s2_p2", "conv_transpose1d_s3_p1_op2"],
        "cases",
    );
    let case = object(
        cases
            .get(name)
            .unwrap_or_else(|| panic!("missing fixture case {name}")),
        name,
    );
    exact_keys(case, &["attrs", "oracle", "tensors"], name);
    let kind = match string(case, "oracle").as_str() {
        "torch.nn.functional.conv1d" => Kind::Conv1d,
        "torch.nn.functional.conv_transpose1d" => Kind::ConvTranspose1d,
        other => panic!("unsupported fixture oracle {other:?}"),
    };
    let attrs = object(case.get("attrs").expect("case attrs"), "attrs");
    exact_keys(
        attrs,
        &[
            "dilation",
            "in_channels",
            "kernel",
            "out_channels",
            "output_padding",
            "padding",
            "stride",
        ],
        "attrs",
    );
    let tensors = object(case.get("tensors").expect("case tensors"), "tensors");
    exact_keys(tensors, &["bias", "input", "output", "weight"], "tensors");
    let (input, input_shape) =
        read_tensor(&root, tensors.get("input").expect("input tensor"), "input");
    let (weight, weight_shape) = read_tensor(
        &root,
        tensors.get("weight").expect("weight tensor"),
        "weight",
    );
    let (bias, bias_shape) = read_tensor(&root, tensors.get("bias").expect("bias tensor"), "bias");
    let (output, output_shape) = read_tensor(
        &root,
        tensors.get("output").expect("output tensor"),
        "output",
    );
    assert_eq!(bias_shape.len(), 1, "bias must be rank 1");
    assert_eq!(bias_shape[0], number(attrs, "out_channels"));
    Fixture {
        kind,
        input,
        input_shape: shape3(input_shape, "input shape"),
        weight,
        weight_shape: shape3(weight_shape, "weight shape"),
        bias,
        output,
        output_shape: shape3(output_shape, "output shape"),
        in_channels: number(attrs, "in_channels"),
        out_channels: number(attrs, "out_channels"),
        kernel: number(attrs, "kernel"),
        stride: number(attrs, "stride"),
        dilation: number(attrs, "dilation"),
        padding: number(attrs, "padding"),
        output_padding: number(attrs, "output_padding"),
    }
}
