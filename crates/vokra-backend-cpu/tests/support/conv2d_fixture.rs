use std::path::{Component, Path, PathBuf};

use vokra_core::json::{self, JsonValue};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Conv2d,
    ConvTranspose2d,
}

pub struct Fixture {
    pub name: String,
    pub kind: Kind,
    pub input: Vec<f32>,
    pub input_shape: [usize; 4],
    pub weight: Vec<f32>,
    pub weight_shape: [usize; 4],
    pub bias: Vec<f32>,
    pub output: Vec<f32>,
    pub output_shape: [usize; 4],
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub output_padding: [usize; 2],
    pub groups: usize,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("conv2d")
}

fn object<'a>(value: &'a JsonValue, context: &str) -> &'a JsonValue {
    assert!(value.as_object().is_some(), "{context} must be an object");
    value
}

fn exact_keys(value: &JsonValue, expected: &[&str], context: &str) {
    let entries = value
        .as_object()
        .unwrap_or_else(|| panic!("{context} must be an object"));
    assert_eq!(entries.len(), expected.len(), "{context} key count");
    for key in expected {
        assert!(
            entries.iter().any(|(actual, _)| actual == key),
            "{context} is missing exact key {key:?}"
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
        .and_then(|number| usize::try_from(number).ok())
        .unwrap_or_else(|| panic!("manifest field {key:?} must be a usize"))
}

fn array(value: &JsonValue, key: &str) -> Vec<usize> {
    value
        .get(key)
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("manifest field {key:?} must be an integer array"))
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .unwrap_or_else(|| panic!("manifest field {key:?}[{index}] must be usize"))
        })
        .collect()
}

fn array2(value: &JsonValue, key: &str) -> [usize; 2] {
    array(value, key)
        .try_into()
        .unwrap_or_else(|_| panic!("manifest field {key:?} must have length 2"))
}

fn shape4(values: Vec<usize>, key: &str) -> [usize; 4] {
    values
        .try_into()
        .unwrap_or_else(|_| panic!("manifest field {key:?} must have rank 4"))
}

fn sha256_hex(data: &[u8]) -> String {
    // FIPS-180-4 SHA-256, kept local so this test adds no digest dependency.
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
    let mut h = [
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
        for (index, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[index] = u32::from_be_bytes(chunk.try_into().unwrap());
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
    let shape = array(record, "shape");
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
    let digest = string(record, "sha256");
    assert_eq!(digest.len(), 64, "{role} digest length");
    assert!(
        digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && digest.chars().any(|character| character != '0'),
        "{role} digest must be a non-placeholder SHA-256"
    );
    assert_eq!(digest, sha256_hex(&bytes), "{role} SHA-256");
    let elements = shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value));
    assert_eq!(elements, Some(bytes.len() / 4), "{role} shape/byte count");
    assert_eq!(bytes.len() % 4, 0, "{role} must contain whole f32 values");
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (values, shape)
}

fn expected_files() -> Vec<String> {
    let cases = [
        "conv2d_grouped_d2_s21_p12",
        "conv_transpose2d_grouped_d21_s23_p12_op12",
        "conv_transpose2d_op1_lt_dilation",
    ];
    let mut files = vec!["README.md", "manifest.json", "manifest.sha256"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for case in cases {
        for role in ["input", "weight", "bias", "output"] {
            files.push(format!("{case}_{role}.f32"));
        }
    }
    files.sort_unstable();
    files
}

pub fn load_all() -> Vec<Fixture> {
    let root = fixture_root();
    let mut actual = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read Conv2d fixture directory {root:?}: {error}"))
        .map(|entry| {
            let entry = entry.expect("read Conv2d fixture directory entry");
            let metadata = std::fs::symlink_metadata(entry.path()).expect("stat fixture entry");
            assert!(
                metadata.file_type().is_file(),
                "fixture entry {:?} must be a regular non-symlink file",
                entry.file_name()
            );
            entry.file_name().to_string_lossy().into_owned()
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(
        actual,
        expected_files(),
        "Conv2d fixture directory file set"
    );

    let manifest_path = root.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|error| panic!("read Conv2d fixture manifest: {error}"));
    let pin = std::fs::read_to_string(root.join("manifest.sha256"))
        .expect("read Conv2d manifest SHA-256 pin");
    let mut pin_parts = pin.split_whitespace();
    let pinned_digest = pin_parts.next().expect("manifest SHA-256 digest");
    assert_eq!(
        pin_parts.next(),
        Some("manifest.json"),
        "manifest pin filename"
    );
    assert!(pin_parts.next().is_none(), "manifest pin has extra fields");
    assert_eq!(pinned_digest.len(), 64, "manifest pin digest length");
    assert!(
        pinned_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && pinned_digest.chars().any(|character| character != '0'),
        "manifest pin must be a non-placeholder SHA-256"
    );
    assert_eq!(
        pinned_digest,
        sha256_hex(&manifest_bytes),
        "manifest SHA-256"
    );
    let manifest = json::parse(&manifest_bytes).expect("Conv2d manifest must be JSON");
    exact_keys(&manifest, &["cases", "provenance", "schema"], "manifest");
    assert_eq!(string(&manifest, "schema"), "vokra-conv2d-parity-v1");
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
        "tools/parity/conv2d_dump_reference.py"
    );
    assert_eq!(string(provenance, "randomness"), "none");
    assert_eq!(string(provenance, "value_policy"), "signed powers of two");
    assert_eq!(string(provenance, "dtype"), "float32");
    assert_eq!(string(provenance, "byte_order"), "little-endian");
    assert!(!string(provenance, "torch_version").is_empty());

    let cases = object(manifest.get("cases").expect("manifest cases"), "cases");
    let names = [
        "conv2d_grouped_d2_s21_p12",
        "conv_transpose2d_grouped_d21_s23_p12_op12",
        "conv_transpose2d_op1_lt_dilation",
    ];
    names
        .iter()
        .map(|name| load_case(&root, cases, name))
        .collect()
}

fn load_case(root: &Path, cases: &JsonValue, name: &str) -> Fixture {
    let case = object(
        cases
            .get(name)
            .unwrap_or_else(|| panic!("missing Conv2d fixture case {name}")),
        name,
    );
    exact_keys(case, &["attrs", "oracle", "tensors"], name);
    let kind = match string(case, "oracle").as_str() {
        "torch.nn.functional.conv2d" => Kind::Conv2d,
        "torch.nn.functional.conv_transpose2d" => Kind::ConvTranspose2d,
        other => panic!("unsupported Conv2d fixture oracle {other:?}"),
    };
    let attrs = object(case.get("attrs").expect("case attrs"), "attrs");
    exact_keys(
        attrs,
        &[
            "dilation",
            "groups",
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
    let (input, input_shape) = read_tensor(root, tensors.get("input").unwrap(), "input");
    let (weight, weight_shape) = read_tensor(root, tensors.get("weight").unwrap(), "weight");
    let (bias, bias_shape) = read_tensor(root, tensors.get("bias").unwrap(), "bias");
    let (output, output_shape) = read_tensor(root, tensors.get("output").unwrap(), "output");
    let input_shape = shape4(input_shape, "input shape");
    let weight_shape = shape4(weight_shape, "weight shape");
    let output_shape = shape4(output_shape, "output shape");
    assert_eq!(input_shape[0], 1, "{name} input batch");
    assert_eq!(output_shape[0], 1, "{name} output batch");
    assert_eq!(bias_shape.len(), 1, "{name} bias must be rank 1");
    assert_eq!(
        bias_shape[0],
        number(attrs, "out_channels"),
        "{name} bias shape"
    );
    Fixture {
        name: name.to_owned(),
        kind,
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        output,
        output_shape,
        in_channels: number(attrs, "in_channels"),
        out_channels: number(attrs, "out_channels"),
        kernel: array2(attrs, "kernel"),
        stride: array2(attrs, "stride"),
        padding: array2(attrs, "padding"),
        dilation: array2(attrs, "dilation"),
        output_padding: array2(attrs, "output_padding"),
        groups: number(attrs, "groups"),
    }
}
