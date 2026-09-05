use std::path::{Path, PathBuf};

use vokra_core::json::{self, JsonValue};

pub struct Fixture {
    pub name: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub output: Vec<f32>,
    pub atol: f32,
    pub rtol: f32,
}

const CASES: [(&str, usize, usize, usize); 3] = [
    ("full_k32_m8_n64", 8, 64, 96),
    ("tails_m3_n35_k65", 3, 35, 65),
    ("tails_m9_n33_k31", 9, 33, 31),
];

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("bf16_gemm")
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

fn real(value: &JsonValue, key: &str) -> f32 {
    let number = value
        .get(key)
        .and_then(|value| match value {
            JsonValue::Int(number) => Some(*number as f64),
            JsonValue::Float(number) => Some(*number),
            _ => None,
        })
        .unwrap_or_else(|| panic!("manifest field {key:?} must be a JSON number"));
    assert!(number.is_finite(), "manifest field {key:?} must be finite");
    let narrowed = number as f32;
    assert!(narrowed.is_finite(), "manifest field {key:?} overflows f32");
    narrowed
}

fn shape(value: &JsonValue, context: &str) -> (usize, usize, usize) {
    let array = value
        .get("shape")
        .and_then(JsonValue::as_object)
        .unwrap_or_else(|| panic!("{context}.shape must be an object"));
    let shape = JsonValue::Object(array.to_vec());
    exact_keys(&shape, &["k", "m", "n"], &format!("{context}.shape"));
    (
        number(&shape, "m"),
        number(&shape, "n"),
        number(&shape, "k"),
    )
}

fn sha256_hex(data: &[u8]) -> String {
    // FIPS-180-4 SHA-256, kept local so this integration test remains
    // dependency-free (the workspace runtime has no third-party crates).
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
        0x6a09_e667u32,
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
            *slot = slot.wrapping_add(value);
        }
    }
    let mut output = String::with_capacity(64);
    for word in h {
        output.push_str(&format!("{word:08x}"));
    }
    output
}

fn read_f32(
    root: &Path,
    record: &JsonValue,
    context: &str,
    expected_path: &str,
    shape: &[usize],
) -> Vec<f32> {
    exact_keys(
        record,
        &["bytes", "dtype", "path", "sha256", "shape"],
        context,
    );
    assert_eq!(string(record, "dtype"), "float32", "{context} dtype");
    let relative = string(record, "path");
    assert_eq!(relative, expected_path, "{context} path");
    assert!(!relative.is_empty(), "{context} path must not be empty");
    assert!(!relative.contains('/'), "{context} path must be a basename");
    let path = root.join(&relative);
    let metadata = std::fs::symlink_metadata(&path)
        .unwrap_or_else(|error| panic!("read {context} fixture {path:?}: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "{context} must be a regular file"
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {context}: {error}"));
    assert_eq!(number(record, "bytes"), bytes.len(), "{context} byte count");
    let digest = string(record, "sha256");
    assert_eq!(digest.len(), 64, "{context} digest length");
    assert!(
        digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && digest.chars().any(|character| character != '0'),
        "{context} digest must be a non-placeholder SHA-256"
    );
    assert_eq!(digest, sha256_hex(&bytes), "{context} SHA-256");
    let declared_shape = record
        .get("shape")
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("{context} shape must be an integer array"));
    let declared_shape = declared_shape
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .unwrap_or_else(|| panic!("{context} shape[{index}] must be a usize"))
        })
        .collect::<Vec<_>>();
    assert_eq!(declared_shape, shape, "{context} declared shape");
    let expected_elements = shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{context} must contain whole f32 values"
    );
    assert_eq!(
        expected_elements,
        Some(bytes.len() / 4),
        "{context} shape/byte count"
    );
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{context} has non-finite f32"
    );
    values
}

pub fn load_all() -> Vec<Fixture> {
    let root = fixture_root();
    let mut expected_files = vec!["README.md", "manifest.json", "manifest.sha256"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for (name, _, _, _) in CASES {
        expected_files.extend([
            format!("{name}_a.f32"),
            format!("{name}_b.f32"),
            format!("{name}_output.f32"),
        ]);
    }
    expected_files.sort_unstable();
    let mut actual = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("read BF16 fixture directory {root:?}: {error}"))
        .map(|entry| {
            let entry = entry.expect("read BF16 fixture directory entry");
            let metadata =
                std::fs::symlink_metadata(entry.path()).expect("stat BF16 fixture entry");
            assert!(
                metadata.file_type().is_file(),
                "fixture entry must be a regular file"
            );
            entry.file_name().to_string_lossy().into_owned()
        })
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, expected_files, "BF16 fixture directory file set");

    let manifest_bytes = std::fs::read(root.join("manifest.json")).expect("read BF16 manifest");
    let pin =
        std::fs::read_to_string(root.join("manifest.sha256")).expect("read BF16 manifest pin");
    let mut parts = pin.split_whitespace();
    let pinned = parts.next().expect("manifest SHA-256 digest");
    assert_eq!(parts.next(), Some("manifest.json"), "manifest pin filename");
    assert!(parts.next().is_none(), "manifest pin has extra fields");
    assert_eq!(pinned.len(), 64, "manifest pin digest length");
    assert!(
        pinned.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "manifest pin format"
    );
    assert_eq!(pinned, sha256_hex(&manifest_bytes), "manifest SHA-256");
    let manifest = json::parse(&manifest_bytes).expect("BF16 manifest must be JSON");
    exact_keys(
        &manifest,
        &["cases", "comparison", "provenance", "schema"],
        "manifest",
    );
    assert_eq!(string(&manifest, "schema"), "vokra-bf16-gemm-parity-v1");

    let provenance = object(
        manifest.get("provenance").expect("manifest provenance"),
        "provenance",
    );
    exact_keys(
        provenance,
        &[
            "byte_order",
            "device",
            "dtype",
            "generator",
            "generator_identity",
            "oracle",
            "randomness",
            "torch_version",
        ],
        "provenance",
    );
    assert_eq!(string(provenance, "byte_order"), "little-endian");
    assert_eq!(string(provenance, "device"), "cpu");
    assert_eq!(string(provenance, "dtype"), "float32");
    assert_eq!(
        string(provenance, "generator"),
        "tools/parity/bf16_gemm/dump_reference.py"
    );
    assert_eq!(
        string(provenance, "generator_identity"),
        "deterministic torch.matmul BF16-widened oracle"
    );
    assert_eq!(
        string(provenance, "oracle"),
        "torch.matmul(a.to(torch.bfloat16).to(torch.float32), b.to(torch.bfloat16).to(torch.float32))"
    );
    assert_eq!(string(provenance, "randomness"), "none");
    let torch_version = string(provenance, "torch_version");
    assert_eq!(
        torch_version.split('+').next(),
        Some("2.13.0"),
        "BF16 fixture must use the pinned Torch 2.13.0 series"
    );

    let comparison = object(
        manifest.get("comparison").expect("manifest comparison"),
        "comparison",
    );
    exact_keys(comparison, &["atol", "rtol"], "comparison");
    let atol = real(comparison, "atol");
    let rtol = real(comparison, "rtol");
    assert_eq!(atol, 1e-3, "BF16 fixture must use the pre-registered atol");
    assert_eq!(rtol, 0.0, "BF16 fixture must use the pre-registered rtol");

    let cases = object(manifest.get("cases").expect("manifest cases"), "cases");
    assert_eq!(cases.as_object().unwrap().len(), CASES.len(), "case count");
    CASES
        .into_iter()
        .map(|(name, expected_m, expected_n, expected_k)| {
            let case = object(
                cases
                    .get(name)
                    .unwrap_or_else(|| panic!("missing BF16 case {name}")),
                name,
            );
            exact_keys(case, &["shape", "tensors"], name);
            let (m, n, k) = shape(case, name);
            assert_eq!(
                (m, n, k),
                (expected_m, expected_n, expected_k),
                "{name} shape"
            );
            let tensors = object(case.get("tensors").expect("case tensors"), "tensors");
            exact_keys(tensors, &["a", "b", "output"], &format!("{name}.tensors"));
            let a = read_f32(
                &root,
                tensors.get("a").unwrap(),
                &format!("{name}.a"),
                &format!("{name}_a.f32"),
                &[m, k],
            );
            let b = read_f32(
                &root,
                tensors.get("b").unwrap(),
                &format!("{name}.b"),
                &format!("{name}_b.f32"),
                &[k, n],
            );
            let output = read_f32(
                &root,
                tensors.get("output").unwrap(),
                &format!("{name}.output"),
                &format!("{name}_output.f32"),
                &[m, n],
            );
            Fixture {
                name: name.to_owned(),
                m,
                n,
                k,
                a,
                b,
                output,
                atol,
                rtol,
            }
        })
        .collect()
}
