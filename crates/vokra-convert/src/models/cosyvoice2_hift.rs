//! Strict CosyVoice2 HiFTNet companion safetensors -> GGUF conversion.
//! The upstream g/v pairs and Snake alphas are preserved verbatim; folding is
//! exclusively a runtime bind-time operation.

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

pub const ARCH: &str = "cosyvoice2_hift";
pub const NAME: &str = "cosyvoice2-0.5b-hift";
pub const CATEGORY: &str = "vocoder";
pub const UPSTREAM_HF: &str = "FunAudioLLM/CosyVoice2-0.5B";
pub const UPSTREAM_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
pub const CHECKPOINT_FILE: &str = "hift.pt";
pub const CHECKPOINT_BYTES: u64 = 83_390_254;
pub const CHECKPOINT_SHA256: &str =
    "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879";
pub const CONFIG_BYTES: u64 = 7_330;
pub const CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
pub const CONFIG_GIT_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
pub const SOURCE_REPO: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
pub const SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
pub const SOURCE_GENERATOR_SHA256: &str =
    "f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626";
pub const SOURCE_GENERATOR_GIT_BLOB_SHA1: &str = "326a1a70ae7707662939c20493b3a8e4b0906216";
pub const TENSOR_COUNT: usize = 328;
pub const MANIFEST_SHA256: &str =
    "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c";
pub const KEY_CATEGORY: &str = "vokra.model.category";
pub const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
pub const KEY_PROVENANCE_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
pub const KEY_UPSTREAM_REVISION: &str = "vokra.cosyvoice2_hift.upstream_revision";
pub const KEY_CHECKPOINT_FILE: &str = "vokra.cosyvoice2_hift.checkpoint_file";
pub const KEY_CHECKPOINT_BYTES: &str = "vokra.cosyvoice2_hift.checkpoint_bytes";
pub const KEY_CHECKPOINT_SHA256: &str = "vokra.cosyvoice2_hift.checkpoint_sha256";
pub const KEY_CONFIG_BYTES: &str = "vokra.cosyvoice2_hift.config_bytes";
pub const KEY_CONFIG_SHA256: &str = "vokra.cosyvoice2_hift.config_sha256";
pub const KEY_CONFIG_GIT_BLOB_SHA1: &str = "vokra.cosyvoice2_hift.config_git_blob_sha1";
pub const KEY_SOURCE_REPO: &str = "vokra.cosyvoice2_hift.source_repo";
pub const KEY_SOURCE_REVISION: &str = "vokra.cosyvoice2_hift.source_revision";
pub const KEY_SOURCE_GENERATOR_SHA256: &str = "vokra.cosyvoice2_hift.source_generator_sha256";
pub const KEY_SOURCE_GENERATOR_GIT_BLOB_SHA1: &str =
    "vokra.cosyvoice2_hift.source_generator_git_blob_sha1";
pub const KEY_MANIFEST_SHA256: &str = "vokra.cosyvoice2_hift.tensor_manifest_sha256";
pub const KEY_WEIGHT_NORM_PAIRS: &str = "vokra.cosyvoice2_hift.weight_norm_pairs";
const CONFIG_PREFIX: &str = "vokra.cosyvoice2_hift.config.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Summary of a successful strict HiFT companion conversion.
pub struct ConvertHiftReport {
    /// Number of tensors emitted.
    pub written: usize,
    /// Number of metadata entries emitted.
    pub metadata_count: usize,
    /// Serialized GGUF size in bytes.
    pub output_bytes: u64,
}

/// Converts the strictly bound CosyVoice2 HiFT safetensors and YAML sidecar.
pub fn convert_cosyvoice2_hift_file(
    input: &Path,
    config: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertHiftReport, ConvertError> {
    require_explicit_license(license)?;
    let config_bytes = std::fs::read(config).map_err(ConvertError::Io)?;
    validate_config(&config_bytes)?;
    let st = SafetensorsFile::parse(std::fs::read(input).map_err(ConvertError::Io)?)?;
    validate_manifest(&st)?;
    let mut builder = GgufBuilder::new();
    builder
        .add_string(chunks::KEY_MODEL_ARCH, ARCH)
        .add_string(chunks::KEY_MODEL_NAME, NAME)
        .add_string(KEY_CATEGORY, CATEGORY);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(NAME),
        Some(SOURCE_REPO),
    );
    for (k, v) in [
        (KEY_UPSTREAM_HF, UPSTREAM_HF),
        (KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_PROVENANCE_CHECKPOINT_SHA256, CHECKPOINT_SHA256),
        (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_CHECKPOINT_FILE, CHECKPOINT_FILE),
        (KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256),
        (KEY_CONFIG_SHA256, CONFIG_SHA256),
        (KEY_CONFIG_GIT_BLOB_SHA1, CONFIG_GIT_BLOB_SHA1),
        (KEY_SOURCE_REPO, SOURCE_REPO),
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_SOURCE_GENERATOR_SHA256, SOURCE_GENERATOR_SHA256),
        (
            KEY_SOURCE_GENERATOR_GIT_BLOB_SHA1,
            SOURCE_GENERATOR_GIT_BLOB_SHA1,
        ),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
    ] {
        builder.add_string(k, v);
    }
    builder
        .add_u32(KEY_CHECKPOINT_BYTES, CHECKPOINT_BYTES as u32)
        .add_u32(KEY_CONFIG_BYTES, CONFIG_BYTES as u32)
        .add_u32(KEY_WEIGHT_NORM_PAIRS, 82);
    add_config_metadata(&mut builder);
    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            GgmlType::F32,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    let out = builder.to_bytes()?;
    write_no_replace(output, &out).map_err(ConvertError::Io)?;
    Ok(ConvertHiftReport {
        written: TENSOR_COUNT,
        metadata_count: builder.metadata_count(),
        output_bytes: out.len() as u64,
    })
}

fn require_explicit_license(license: Option<&str>) -> Result<(), ConvertError> {
    match license {
        Some(value) if value.eq_ignore_ascii_case("apache-2.0") => Ok(()),
        Some(value) => Err(ConvertError::Parse(format!(
            "{ARCH}: explicit license attestation must exactly match Apache-2.0; got `{value}`"
        ))),
        None => Err(ConvertError::Parse(format!(
            "{ARCH}: explicit Apache-2.0 license attestation is required"
        ))),
    }
}

/// Atomically publish only to an absent output path. A same-filesystem hard
/// link is the std-only no-replace primitive: concurrent writers and stale
/// outputs are rejected. Cleanup is attempted on both paths; after publication
/// a cleanup failure does not turn success into an error.
fn write_no_replace(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "output has no file name")
    })?;
    let mut temporary = None;
    for attempt in 0..100u32 {
        let candidate = parent.join(format!(
            ".{}.vokra-cosyvoice2-hift-{}-{}",
            name.to_string_lossy(),
            std::process::id(),
            attempt
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let Some((temporary_path, mut file)) = temporary else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate temporary output",
        ));
    };
    let operation = (|| {
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        std::fs::hard_link(&temporary_path, output)
    })();
    let cleanup = std::fs::remove_file(&temporary_path);
    match operation {
        Err(error) => {
            let _ = cleanup;
            Err(error)
        }
        Ok(()) => {
            let _ = cleanup;
            Ok(())
        }
    }
}

fn add_config_metadata(b: &mut GgufBuilder) {
    for (k, v) in [
        ("in_channels", 80),
        ("base_channels", 512),
        ("nb_harmonics", 8),
        ("sampling_rate", 24_000),
        ("istft_n_fft", 16),
        ("istft_hop_len", 4),
    ] {
        b.add_u32(&format!("{CONFIG_PREFIX}{k}"), v);
    }
    for (k, v) in [
        ("nsf_alpha", 0.1),
        ("nsf_sigma", 0.003),
        ("nsf_voiced_threshold", 10.0),
        ("lrelu_slope", 0.1),
        ("audio_limit", 0.99),
    ] {
        b.add_f32(&format!("{CONFIG_PREFIX}{k}"), v);
    }
    for (name, values) in [
        ("upsample_rates", [8, 5, 3]),
        ("upsample_kernel_sizes", [16, 11, 7]),
        ("resblock_kernel_sizes", [3, 7, 11]),
        ("source_resblock_kernel_sizes", [7, 7, 11]),
    ] {
        for (i, v) in values.into_iter().enumerate() {
            b.add_u32(&format!("{CONFIG_PREFIX}{name}.{i}"), v);
        }
        b.add_u32(&format!("{CONFIG_PREFIX}{name}.len"), 3);
    }
    for name in ["resblock_dilation_sizes", "source_resblock_dilation_sizes"] {
        for i in 0..3 {
            for (j, v) in [1, 3, 5].into_iter().enumerate() {
                b.add_u32(&format!("{CONFIG_PREFIX}{name}.{i}.{j}"), v);
            }
        }
    }
    for (k, v) in [
        ("f0.class_channels", 1),
        ("f0.input_channels", 80),
        ("f0.cond_channels", 512),
        ("f0.num_layers", 5),
        ("f0.kernel_size", 3),
    ] {
        b.add_u32(&format!("{CONFIG_PREFIX}{k}"), v);
    }
}

fn validate_config(bytes: &[u8]) -> Result<(), ConvertError> {
    let sha = crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(bytes));
    let mut blob = format!("blob {}\0", bytes.len()).into_bytes();
    blob.extend_from_slice(bytes);
    let blob_sha = hex_bytes(&sha1(&blob));
    if bytes.len() as u64 != CONFIG_BYTES
        || sha != CONFIG_SHA256
        || blob_sha != CONFIG_GIT_BLOB_SHA1
    {
        return Err(ConvertError::Parse(format!(
            "{ARCH}: exact cosyvoice2.yaml identity mismatch (bytes={}, sha256={sha}, blob_sha1={blob_sha})",
            bytes.len()
        )));
    }
    Ok(())
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected: BTreeMap<_, _> = expected_manifest().into_iter().collect();
    if st.tensors().len() != TENSOR_COUNT || expected.len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "{ARCH}: expected exactly {TENSOR_COUNT} tensors"
        )));
    }
    let mut actual = BTreeMap::new();
    for t in st.tensors() {
        if t.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "{ARCH}: `{}` must be F32",
                t.name
            )));
        }
        actual.insert(t.name.clone(), t.shape.clone());
    }
    if actual != expected {
        return Err(ConvertError::Parse(format!(
            "{ARCH}: complete name/shape manifest mismatch"
        )));
    }
    let digest = crate::models::canary_1b_flash::hex(
        &crate::models::canary_1b_flash::manifest_sha256(&actual),
    );
    if digest != MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "{ARCH}: manifest digest {digest} != pinned {MANIFEST_SHA256}"
        )));
    }
    Ok(())
}

pub(crate) fn expected_manifest() -> Vec<(String, Vec<u64>)> {
    let mut m = Vec::with_capacity(TENSOR_COUNT);
    fn wn(m: &mut Vec<(String, Vec<u64>)>, p: &str, r: u64, c: u64, k: u64) {
        m.push((
            format!("{p}.parametrizations.weight.original0"),
            vec![r, 1, 1],
        ));
        m.push((
            format!("{p}.parametrizations.weight.original1"),
            vec![r, c, k],
        ));
    }
    fn conv(m: &mut Vec<(String, Vec<u64>)>, p: &str, r: u64, c: u64, k: u64) {
        m.push((format!("{p}.bias"), vec![r]));
        wn(m, p, r, c, k);
    }
    conv(&mut m, "conv_pre", 512, 80, 7);
    conv(&mut m, "conv_post", 18, 64, 7);
    for i in 0..5 {
        conv(
            &mut m,
            &format!("f0_predictor.condnet.{}", i * 2),
            512,
            if i == 0 { 80 } else { 512 },
            3,
        );
    }
    m.push(("f0_predictor.classifier.bias".into(), vec![1]));
    m.push(("f0_predictor.classifier.weight".into(), vec![1, 512]));
    m.push(("m_source.l_linear.bias".into(), vec![1]));
    m.push(("m_source.l_linear.weight".into(), vec![1, 9]));
    fn block(m: &mut Vec<(String, Vec<u64>)>, p: &str, ch: u64, k: u64) {
        for b in 0..3 {
            m.push((format!("{p}.activations1.{b}.alpha"), vec![ch]));
            m.push((format!("{p}.activations2.{b}.alpha"), vec![ch]));
            conv(m, &format!("{p}.convs1.{b}"), ch, ch, k);
            conv(m, &format!("{p}.convs2.{b}"), ch, ch, k);
        }
    }
    for (i, ch) in [256, 128, 64].into_iter().enumerate() {
        m.push((format!("source_downs.{i}.bias"), vec![ch]));
        m.push((
            format!("source_downs.{i}.weight"),
            vec![ch, 18, [30, 6, 1][i]],
        ));
        block(&mut m, &format!("source_resblocks.{i}"), ch, [7, 7, 11][i]);
    }
    for (i, (out, inn, k)) in [(256, 512, 16), (128, 256, 11), (64, 128, 7)]
        .into_iter()
        .enumerate()
    {
        m.push((format!("ups.{i}.bias"), vec![out]));
        wn(&mut m, &format!("ups.{i}"), inn, out, k);
    }
    for (i, ch) in [256, 128, 64].into_iter().enumerate() {
        for j in 0..3 {
            block(
                &mut m,
                &format!("resblocks.{}", i * 3 + j),
                ch,
                [3, 7, 11][j],
            );
        }
    }
    m
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h = [
        0x67452301u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];
    let bit = (data.len() as u64) * 8;
    let mut p = data.to_vec();
    p.push(0x80);
    while p.len() % 64 != 56 {
        p.push(0)
    }
    p.extend_from_slice(&bit.to_be_bytes());
    for c in p.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(c[i * 4..i * 4 + 4].try_into().unwrap())
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1)
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5a827999),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e)
    }
    let mut o = [0u8; 20];
    for (i, v) in h.into_iter().enumerate() {
        o[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes())
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manifest_is_pinned() {
        let m: BTreeMap<_, _> = expected_manifest().into_iter().collect();
        assert_eq!(m.len(), TENSOR_COUNT);
        assert_eq!(
            crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::manifest_sha256(
                &m
            )),
            MANIFEST_SHA256
        )
    }

    #[test]
    fn explicit_license_attestation_is_required_and_fixed() {
        let missing = require_explicit_license(None).expect_err("missing license must fail closed");
        assert!(missing.to_string().contains("attestation is required"));

        let wrong = require_explicit_license(Some("mit")).expect_err("wrong license must fail");
        assert!(wrong.to_string().contains("must exactly match Apache-2.0"));

        require_explicit_license(Some("Apache-2.0"))
            .expect("case-insensitive Apache-2.0 attestation must be accepted");
    }

    #[test]
    fn sha1_vector() {
        assert_eq!(
            hex_bytes(&sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        )
    }

    #[test]
    fn wrong_or_missing_config_is_rejected_before_output() {
        let root = std::env::temp_dir().join(format!("vokra-hift-test-{}", std::process::id()));
        let input = root.join("input.safetensors");
        let config = root.join("cosyvoice2.yaml");
        let output = root.join("out.gguf");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        std::fs::write(&input, b"not a checkpoint").unwrap();
        std::fs::write(&config, b"wrong").unwrap();
        std::fs::write(&output, b"original").unwrap();
        let error = convert_cosyvoice2_hift_file(&input, &config, &output, Some("apache-2.0"))
            .expect_err("wrong config must fail closed");
        assert!(error.to_string().contains("identity mismatch"));
        assert_eq!(std::fs::read(&output).unwrap(), b"original");
        std::fs::remove_file(&output).unwrap();
        let _ = convert_cosyvoice2_hift_file(&input, &config, &output, Some("apache-2.0"));
        assert!(
            !output.exists(),
            "failure must not leave an output artifact"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn missing_license_is_rejected_before_input_io() {
        let error = convert_cosyvoice2_hift_file(
            Path::new("/definitely/nonexistent/cosyvoice2-hift.safetensors"),
            Path::new("/definitely/nonexistent/cosyvoice2.yaml"),
            Path::new("/definitely/nonexistent/out.gguf"),
            None,
        )
        .expect_err("missing license must fail before input IO");
        assert!(
            error
                .to_string()
                .contains("explicit Apache-2.0 license attestation")
        );
    }

    #[test]
    fn partial_manifest_is_rejected() {
        let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        let st = SafetensorsFile::parse(bytes).unwrap();
        let error = validate_manifest(&st).expect_err("partial manifest must fail closed");
        assert!(
            error.to_string().contains("expected exactly 328")
                || error.to_string().contains("manifest")
        );
    }

    #[test]
    fn output_is_no_replace_and_temporary_is_cleaned() {
        let root = std::env::temp_dir().join(format!("vokra-hift-atomic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).unwrap();
        let output = root.join("model.gguf");
        std::fs::write(&output, b"original").unwrap();
        let error = write_no_replace(&output, b"replacement").expect_err("must reject overwrite");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&output).unwrap(), b"original");
        let fresh = root.join("fresh.gguf");
        write_no_replace(&fresh, b"complete").unwrap();
        assert_eq!(std::fs::read(&fresh).unwrap(), b"complete");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(root);
    }
}
