//! Shared fail-closed binding for published pass-through checkpoints.
//!
//! Several model converters preserve upstream tensor names and shapes
//! verbatim.  A count-only loader would accept an unrelated or truncated
//! checkpoint with the same number of tensors, so these binders pin a
//! canonical SHA-256 over the complete sorted `(name, dimensions)` manifest.
//! Tensor payload bytes are deliberately excluded: quantization may change
//! those without changing the architecture contract.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

/// Immutable identity and complete tensor-manifest contract for one release.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StrictCheckpointSpec {
    pub(crate) label: &'static str,
    pub(crate) arch: &'static str,
    pub(crate) model_name: &'static str,
    pub(crate) model_name_alias: Option<&'static str>,
    pub(crate) tensor_count: usize,
    pub(crate) manifest_sha256: [u8; 32],
}

/// Cheap handle proving that a GGUF matches a pinned release manifest.
#[derive(Debug, Clone)]
pub(crate) struct StrictCheckpoint {
    model_name: String,
    weight_license: LicenseClass,
    tensor_count: usize,
}

impl StrictCheckpoint {
    pub(crate) fn bind(file: &GgufFile, spec: StrictCheckpointSpec) -> Result<Self> {
        let arch = required_string(file, chunks::KEY_MODEL_ARCH, spec.label)?;
        if arch != spec.arch {
            return Err(VokraError::ModelLoad(format!(
                "{}: unsupported `{}`={arch:?}; expected {:?}",
                spec.label,
                chunks::KEY_MODEL_ARCH,
                spec.arch
            )));
        }
        let model_name = required_string(file, chunks::KEY_MODEL_NAME, spec.label)?;
        if model_name != spec.model_name && spec.model_name_alias != Some(model_name) {
            return Err(VokraError::ModelLoad(format!(
                "{}: unsupported `{}`={model_name:?}; expected {:?} or {:?}",
                spec.label,
                chunks::KEY_MODEL_NAME,
                spec.model_name,
                spec.model_name_alias
            )));
        }
        verify_tensor_manifest(
            file,
            spec.label,
            spec.tensor_count,
            spec.manifest_sha256,
            model_name,
        )?;

        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(GgufMetadataValue::as_str)
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            model_name: model_name.to_owned(),
            weight_license,
            tensor_count: file.tensors().len(),
        })
    }

    pub(crate) fn model_name(&self) -> &str {
        &self.model_name
    }

    pub(crate) const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    pub(crate) const fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

/// Verify a complete sorted `(tensor name, dimensions)` manifest without
/// imposing an arch/name metadata policy.
///
/// Composite families such as MusicGen have release-specific metadata and
/// may need a narrowly authenticated legacy arch alias, but must still share
/// the exact same fail-closed tensor contract as [`StrictCheckpoint`].
pub(crate) fn verify_tensor_manifest(
    file: &GgufFile,
    label: &str,
    tensor_count: usize,
    expected_sha256: [u8; 32],
    model_name: &str,
) -> Result<()> {
    if file.tensors().len() != tensor_count {
        return Err(VokraError::ModelLoad(format!(
            "{label}: tensor count {}, expected {tensor_count} for {model_name:?}",
            file.tensors().len()
        )));
    }
    let actual = manifest_sha256(file);
    if actual != expected_sha256 {
        return Err(VokraError::ModelLoad(format!(
            "{label}: complete tensor name/shape manifest SHA-256 {}, expected {} for {model_name:?}",
            hex(&actual),
            hex(&expected_sha256)
        )));
    }
    Ok(())
}

pub(crate) fn load_tensor(
    file: &GgufFile,
    label: &str,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>> {
    require_tensor_shape(file, label, name, expected)?;
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("{label}: tensor `{name}` decode failed: {error}"))
    })
}

pub(crate) fn require_tensor_shape(
    file: &GgufFile,
    label: &str,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("{label}: required tensor `{name}` is missing"))
    })?;
    let actual: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect();
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{label}: tensor `{name}` shape {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

/// Row-major PyTorch `Linear(in_features, out_features)` for one or more rows.
pub(crate) fn linear_rows(
    label: &str,
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    if in_features == 0
        || out_features == 0
        || input.is_empty()
        || input.len() % in_features != 0
        || weight.len() != in_features * out_features
        || bias.is_some_and(|value| value.len() != out_features)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: affine shape mismatch: input={}, weight={}, bias={:?}, in_features={in_features}, out_features={out_features}",
            input.len(),
            weight.len(),
            bias.map(<[f32]>::len)
        )));
    }
    let rows = input.len() / in_features;
    let mut output = vec![0.0f32; rows * out_features];
    for row in 0..rows {
        for out in 0..out_features {
            let mut value = bias.map_or(0.0, |values| values[out]);
            for inner in 0..in_features {
                value += input[row * in_features + inner] * weight[out * in_features + inner];
            }
            output[row * out_features + out] = value;
        }
    }
    Ok(output)
}

/// Row-major PyTorch embedding lookup over an upstream `[vocab, dim]` table.
pub(crate) fn embedding_rows(
    label: &str,
    token_ids: &[u32],
    weight: &[f32],
    vocab_size: usize,
    dimension: usize,
) -> Result<Vec<f32>> {
    if vocab_size == 0
        || dimension == 0
        || token_ids.is_empty()
        || weight.len() != vocab_size * dimension
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: embedding shape mismatch: ids={}, weight={}, vocab_size={vocab_size}, dimension={dimension}",
            token_ids.len(),
            weight.len()
        )));
    }
    let mut output = Vec::with_capacity(token_ids.len() * dimension);
    for (position, &token_id) in token_ids.iter().enumerate() {
        let token = token_id as usize;
        if token >= vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: token_ids[{position}]={token_id} is outside 0..{vocab_size}"
            )));
        }
        output.extend_from_slice(&weight[token * dimension..(token + 1) * dimension]);
    }
    Ok(output)
}

fn required_string<'a>(file: &'a GgufFile, key: &str, label: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{label}: missing/non-string `{key}`")))
}

fn manifest_sha256(file: &GgufFile) -> [u8; 32] {
    let mut tensors: Vec<_> = file.tensors().iter().collect();
    tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    let capacity = tensors
        .iter()
        .map(|tensor| tensor.name.len() + 1 + 8 + tensor.dimensions.len() * 8)
        .sum();
    let mut canonical = Vec::with_capacity(capacity);
    for tensor in tensors {
        canonical.extend_from_slice(tensor.name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(&(tensor.dimensions.len() as u64).to_le_bytes());
        for dimension in &tensor.dimensions {
            canonical.extend_from_slice(&dimension.to_le_bytes());
        }
    }
    sha256(&canonical)
}

fn hex(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from(DIGITS[(byte >> 4) as usize]));
        output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    output
}

// Zero-dependency SHA-256 (FIPS-180-4 section 6.2), shared only inside
// `vokra-models` so the runtime dependency graph remains first-party-only.
const SHA256_K: [u32; 64] = [
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

const SHA256_H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = SHA256_H0;
    let bit_len = (data.len() as u64) * 8;
    let mut buffer = Vec::with_capacity(data.len() + 72);
    buffer.extend_from_slice(data);
    buffer.push(0x80);
    while buffer.len() % 64 != 56 {
        buffer.push(0);
    }
    buffer.extend_from_slice(&bit_len.to_be_bytes());
    for block in buffer.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in block.chunks_exact(4).take(16).enumerate() {
            words[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for (index, word) in words.iter().enumerate() {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let first = hh
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(*word);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = s0.wrapping_add(majority);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut digest = [0u8; 32];
    for (index, word) in h.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_abc() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn linear_rows_matches_hand_calculation_and_rejects_shapes() {
        let output = linear_rows(
            "test",
            &[2.0, -1.0, 0.5],
            &[1.0, 2.0, 3.0, -2.0, 0.0, 4.0],
            Some(&[0.25, -0.5]),
            3,
            2,
        )
        .unwrap();
        assert_eq!(output, vec![1.75, -2.5]);
        assert!(linear_rows("test", &[1.0], &[1.0], None, 2, 1).is_err());
    }

    #[test]
    fn embedding_rows_is_row_major_and_checks_ids() {
        let weights = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        assert_eq!(
            embedding_rows("test", &[1, 0], &weights, 2, 3).unwrap(),
            vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
        );
        assert!(embedding_rows("test", &[2], &weights, 2, 3).is_err());
    }

    #[test]
    #[ignore = "diagnostic: requires VOKRA_MANIFEST_GGUF pointing at a real artifact"]
    fn dump_real_manifest_sha256() {
        let path = std::env::var("VOKRA_MANIFEST_GGUF")
            .expect("VOKRA_MANIFEST_GGUF must name the real GGUF");
        let file = GgufFile::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        eprintln!(
            "path={path} tensors={} manifest_sha256={}",
            file.tensors().len(),
            hex(&manifest_sha256(&file))
        );
        let filter = std::env::var("VOKRA_MANIFEST_FILTER").ok();
        let mut tensors: Vec<_> = file
            .tensors()
            .iter()
            .filter(|tensor| {
                filter
                    .as_ref()
                    .is_none_or(|needle| tensor.name.contains(needle))
            })
            .collect();
        tensors.sort_unstable_by_key(|tensor| tensor.element_count().unwrap_or(u64::MAX));
        for tensor in tensors.into_iter().take(96) {
            eprintln!("{} {:?} {:?}", tensor.name, tensor.dimensions, tensor.dtype);
        }
    }

    #[test]
    #[ignore = "diagnostic: requires VOKRA_MANIFEST_GGUF/TENSOR_NAME/TENSOR_OUT"]
    fn dump_real_tensor_as_f32_le() {
        let path = std::env::var("VOKRA_MANIFEST_GGUF")
            .expect("VOKRA_MANIFEST_GGUF must name the real GGUF");
        let name = std::env::var("VOKRA_TENSOR_NAME").expect("VOKRA_TENSOR_NAME is required");
        let output = std::env::var("VOKRA_TENSOR_OUT").expect("VOKRA_TENSOR_OUT is required");
        let file = GgufFile::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        let values = file
            .tensor_f32(&name)
            .unwrap_or_else(|error| panic!("decode {name}: {error}"));
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&output, bytes)
            .unwrap_or_else(|error| panic!("write tensor dump {output}: {error}"));
    }
}
