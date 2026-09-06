//! Staged CosyVoice2 flow-component safetensors -> GGUF conversion.
//!
//! This is a crate-private, inspection-only component path.  It accepts only
//! a prepared F32 safetensors file; the original `flow.pt` is never read or
//! reconstructed here.  Its fixed raw identity and the actual prepared input
//! digest are stamped separately so the latter is not mistaken for the former.
//! No CLI or public converter registration is provided until the prepared
//! artifact has been reviewed on VAST.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufMetadataValue, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

const ARCH: &str = "cosyvoice2";
const NAME: &str = "cosyvoice2-0.5b-flow";
const COMPONENT: &str = "flow";
const COMPOSITE_STATUS: &str = "INSPECTION_ONLY";
const UPSTREAM_HF: &str = "FunAudioLLM/CosyVoice2-0.5B";
const UPSTREAM_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const CHECKPOINT_FILE: &str = "flow.pt";
const CHECKPOINT_BYTES: u32 = 450_575_567;
const CHECKPOINT_SHA256: &str = "ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541";
const CONFIG_FILE: &str = "cosyvoice2.yaml";
const CONFIG_BYTES: u32 = 7_330;
const CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
const CONFIG_GIT_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
const SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
const SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
const SOURCE_LICENSE_FILE: &str = "LICENSE";
const SOURCE_LICENSE_BYTES: u32 = 11_357;
const SOURCE_LICENSE_SHA256: &str =
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
const SOURCE_LICENSE_GIT_BLOB_SHA1: &str = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64";
const SOURCE_LICENSE_DECLARED: &str = "Apache-2.0";
const TENSOR_COUNT: usize = 1_121;
const MANIFEST_SHA256: &str = "68abcdcdc961091b9c9e6456146ac9da06cda9f86e45a8faaac22a3b0b26217d";
const DATA_PICKLE_SHA256: &str = "d8977bfb852a57439b8e6ca0a656b69bf171c6f86e434bce2210d2c05f0a2448";
const STORAGE_MANIFEST_SHA256: &str =
    "e2ec1a5009a0bf4f63eaebe82d4a1bc037f1cb26e82360f7d40c4c9700fc68ee";
const SOURCE_CLOSURE_MANIFEST_SHA256: &str =
    "cc391a4a63e95b9cdf6373b5b41ac94239a89d6594a235bc142a17c542532673";

const KEY_COMPONENT: &str = "vokra.cosyvoice2_flow.component";
const KEY_COMPOSITE_STATUS: &str = "vokra.cosyvoice2.composite_status";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const KEY_COMPONENT_UPSTREAM_REVISION: &str = "vokra.cosyvoice2_flow.upstream_revision";
const KEY_CHECKPOINT_FILE: &str = "vokra.cosyvoice2_flow.checkpoint_file";
const KEY_CHECKPOINT_BYTES: &str = "vokra.cosyvoice2_flow.checkpoint_bytes";
const KEY_CHECKPOINT_COMPONENT_SHA256: &str = "vokra.cosyvoice2_flow.checkpoint_sha256";
const KEY_CONFIG_FILE: &str = "vokra.cosyvoice2_flow.config_file";
const KEY_CONFIG_BYTES: &str = "vokra.cosyvoice2_flow.config_bytes";
const KEY_CONFIG_SHA256: &str = "vokra.cosyvoice2_flow.config_sha256";
const KEY_CONFIG_GIT_BLOB_SHA1: &str = "vokra.cosyvoice2_flow.config_git_blob_sha1";
const KEY_SOURCE_REPOSITORY: &str = "vokra.cosyvoice2_flow.source_repo";
const KEY_SOURCE_REVISION: &str = "vokra.cosyvoice2_flow.source_revision";
const KEY_SOURCE_LICENSE_FILE: &str = "vokra.cosyvoice2_flow.source_license_file";
const KEY_SOURCE_LICENSE_BYTES: &str = "vokra.cosyvoice2_flow.source_license_bytes";
const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.cosyvoice2_flow.source_license_sha256";
const KEY_SOURCE_LICENSE_GIT_BLOB_SHA1: &str = "vokra.cosyvoice2_flow.source_license_git_blob_sha1";
const KEY_SOURCE_LICENSE_DECLARED: &str = "vokra.cosyvoice2_flow.source_license_declared";
const KEY_MANIFEST_SHA256: &str = "vokra.cosyvoice2_flow.tensor_manifest_sha256";
const KEY_DATA_PICKLE_SHA256: &str = "vokra.cosyvoice2_flow.data_pickle_sha256";
const KEY_STORAGE_MANIFEST_SHA256: &str = "vokra.cosyvoice2_flow.storage_manifest_sha256";
const KEY_SOURCE_CLOSURE_MANIFEST_SHA256: &str =
    "vokra.cosyvoice2_flow.source_closure_manifest_sha256";
const KEY_PREPARED_BYTES: &str = "vokra.cosyvoice2_flow.prepared_input.bytes";
const KEY_PREPARED_SHA256: &str = "vokra.cosyvoice2_flow.prepared_input.sha256";
const KEY_PREPARED_STATUS: &str = "vokra.cosyvoice2_flow.prepared_input.authentication_status";

const SOURCE_ROLES: &[(&str, &str, &str)] = &[
    (
        "cosyvoice/cli/cosyvoice.py",
        "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859",
        "cc443bed44c651a47492fc7e2142e3a88fb47627",
    ),
    (
        "cosyvoice/flow/decoder.py",
        "ef5eceb9db7f63ddda1d5bca6bfa6b28b8ea11656c4b1f9c109d28f656cbbf29",
        "97768a459fbb89a2c99f98de302628d8ccafda67",
    ),
    (
        "cosyvoice/flow/flow.py",
        "a8497feb58336e7566b1f085d11acff9cb4f1a24949abd2c244fbf97c76f9b6d",
        "a068288f889aff4079b0c54c612897d31d08882a",
    ),
    (
        "cosyvoice/flow/flow_matching.py",
        "b1ad671fe37f872c034bde8f75cc19c1b88758d54e375fa2b54e14a088addfe6",
        "7f92df5d24690fe89fc548ab60f37483f91b03a6",
    ),
    (
        "cosyvoice/transformer/upsample_encoder.py",
        "a8003c212ce64697ce43001f776902ee60696a3b7e373935479029dccaf7d569",
        "6ffda6acad25cc0cfcf1bc07b9211c326ca8d49f",
    ),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConvertCosyVoice2FlowReport {
    pub(crate) written: usize,
    pub(crate) metadata_count: usize,
    pub(crate) output_bytes: u64,
    pub(crate) prepared_bytes: u64,
}

/// Converts a prepared flow safetensors file and the exact YAML sidecar.
///
/// The caller must provide an explicit Apache-2.0 attestation.  This function
/// is private and remains inspection-only until the prepared digest is pinned
/// by a reviewed VAST run.
pub(crate) fn convert_cosyvoice2_flow_file(
    input: &Path,
    config: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertCosyVoice2FlowReport, ConvertError> {
    require_explicit_license(license)?;
    require_regular_file(input, "prepared flow safetensors")?;
    require_regular_file(config, CONFIG_FILE)?;
    require_absent_output(output)?;
    let input_bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let config_bytes = std::fs::read(config).map_err(ConvertError::Io)?;
    let (builder, prepared_bytes) = convert_component(input_bytes, &config_bytes)?;
    let output_bytes = builder.to_bytes()?;
    write_no_replace(output, &output_bytes).map_err(ConvertError::Io)?;
    Ok(ConvertCosyVoice2FlowReport {
        written: builder.tensor_count(),
        metadata_count: builder.metadata_count(),
        output_bytes: output_bytes.len() as u64,
        prepared_bytes,
    })
}

/// Validate the VAST handoff paths before reading any payload. Symlink paths
/// are rejected at this handoff boundary; callers must keep the reviewed
/// regular files in place for the conversion operation.
fn require_regular_file(path: &Path, label: &str) -> Result<(), ConvertError> {
    if !path.is_absolute() {
        return Err(ConvertError::Usage(format!(
            "{ARCH} flow: {label} path must be absolute"
        )));
    }
    let metadata = std::fs::symlink_metadata(path).map_err(ConvertError::Io)?;
    let file_type = metadata.file_type();
    if !file_type.is_file() || file_type.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "{ARCH} flow: {label} must be a regular non-symlink file"
        )));
    }
    Ok(())
}

fn require_absent_output(path: &Path) -> Result<(), ConvertError> {
    if !path.is_absolute() {
        return Err(ConvertError::Usage(
            "cosyvoice2 flow: output path must be absolute".to_owned(),
        ));
    }
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(ConvertError::Usage(
            "cosyvoice2 flow: output path must be absent (no replacement)".to_owned(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ConvertError::Io(error)),
    }
}

fn require_explicit_license(license: Option<&str>) -> Result<(), ConvertError> {
    match license {
        Some(value) if value.eq_ignore_ascii_case(SOURCE_LICENSE_DECLARED) => Ok(()),
        Some(value) => Err(ConvertError::Usage(format!(
            "{ARCH} flow: explicit Apache-2.0 license attestation required; got `{value}`"
        ))),
        None => Err(ConvertError::Usage(format!(
            "{ARCH} flow: explicit Apache-2.0 license attestation is required"
        ))),
    }
}

fn convert_component(
    input_bytes: Vec<u8>,
    config_bytes: &[u8],
) -> Result<(GgufBuilder, u64), ConvertError> {
    validate_config(config_bytes)?;
    let prepared_bytes = input_bytes.len() as u64;
    let prepared_sha256 = hex(&crate::models::canary_1b_flash::sha256(&input_bytes));
    let st = SafetensorsFile::parse(input_bytes)?;
    validate_manifest(&st)?;

    let mut builder = GgufBuilder::new();
    stamp_metadata(&mut builder, prepared_bytes, &prepared_sha256);
    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            GgmlType::F32,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    Ok((builder, prepared_bytes))
}

fn stamp_metadata(builder: &mut GgufBuilder, prepared_bytes: u64, prepared_sha256: &str) {
    builder
        .add_string(chunks::KEY_MODEL_ARCH, ARCH)
        .add_string(chunks::KEY_MODEL_NAME, NAME)
        .add_string(KEY_COMPONENT, COMPONENT)
        .add_string(KEY_COMPOSITE_STATUS, COMPOSITE_STATUS)
        .add_string(KEY_UPSTREAM_HF, UPSTREAM_HF)
        .add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)
        .add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)
        .add_string(KEY_COMPONENT_UPSTREAM_REVISION, UPSTREAM_REVISION)
        .add_string(KEY_CHECKPOINT_FILE, CHECKPOINT_FILE)
        .add_u32(KEY_CHECKPOINT_BYTES, CHECKPOINT_BYTES)
        .add_string(KEY_CHECKPOINT_COMPONENT_SHA256, CHECKPOINT_SHA256)
        .add_string(KEY_CONFIG_FILE, CONFIG_FILE)
        .add_u32(KEY_CONFIG_BYTES, CONFIG_BYTES)
        .add_string(KEY_CONFIG_SHA256, CONFIG_SHA256)
        .add_string(KEY_CONFIG_GIT_BLOB_SHA1, CONFIG_GIT_BLOB_SHA1)
        .add_string(KEY_SOURCE_REPOSITORY, SOURCE_REPOSITORY)
        .add_string(KEY_SOURCE_REVISION, SOURCE_REVISION)
        .add_string(KEY_SOURCE_LICENSE_FILE, SOURCE_LICENSE_FILE)
        .add_u32(KEY_SOURCE_LICENSE_BYTES, SOURCE_LICENSE_BYTES)
        .add_string(KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256)
        .add_string(
            KEY_SOURCE_LICENSE_GIT_BLOB_SHA1,
            SOURCE_LICENSE_GIT_BLOB_SHA1,
        )
        .add_string(KEY_SOURCE_LICENSE_DECLARED, SOURCE_LICENSE_DECLARED)
        .add_string(KEY_MANIFEST_SHA256, MANIFEST_SHA256)
        .add_string(KEY_DATA_PICKLE_SHA256, DATA_PICKLE_SHA256)
        .add_string(KEY_STORAGE_MANIFEST_SHA256, STORAGE_MANIFEST_SHA256)
        .add_string(
            KEY_SOURCE_CLOSURE_MANIFEST_SHA256,
            SOURCE_CLOSURE_MANIFEST_SHA256,
        )
        .add_metadata(KEY_PREPARED_BYTES, GgufMetadataValue::U64(prepared_bytes))
        .add_string(KEY_PREPARED_SHA256, prepared_sha256)
        .add_string(
            KEY_PREPARED_STATUS,
            "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED",
        );
    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        SOURCE_LICENSE_DECLARED,
        Some(NAME),
        Some(SOURCE_REPOSITORY),
    );
    for (path, sha256, blob) in SOURCE_ROLES {
        builder
            .add_string(
                &format!("vokra.cosyvoice2_flow.source.{path}.sha256"),
                sha256,
            )
            .add_string(
                &format!("vokra.cosyvoice2_flow.source.{path}.git_blob_sha1"),
                blob,
            );
    }
}

fn validate_config(bytes: &[u8]) -> Result<(), ConvertError> {
    let sha256 = hex(&crate::models::canary_1b_flash::sha256(bytes));
    let mut blob = format!("blob {}\0", bytes.len()).into_bytes();
    blob.extend_from_slice(bytes);
    let blob_sha1 = hex_bytes(&sha1(&blob));
    if bytes.len() as u64 != CONFIG_BYTES as u64
        || sha256 != CONFIG_SHA256
        || blob_sha1 != CONFIG_GIT_BLOB_SHA1
    {
        return Err(ConvertError::Parse(format!(
            "{ARCH} flow: exact cosyvoice2.yaml identity mismatch (bytes={}, sha256={sha256}, blob_sha1={blob_sha1})",
            bytes.len()
        )));
    }
    Ok(())
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    if st.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "{ARCH} flow: expected exactly {TENSOR_COUNT} tensors"
        )));
    }
    let mut actual = BTreeMap::new();
    for tensor in st.tensors() {
        validate_tensor_descriptor(&tensor.name, tensor.dtype)?;
        if actual
            .insert(tensor.name.clone(), tensor.shape.clone())
            .is_some()
        {
            return Err(ConvertError::Parse(format!(
                "{ARCH} flow: duplicate tensor `{}`",
                tensor.name
            )));
        }
    }
    validate_manifest_map(&actual)
}

fn validate_tensor_descriptor(name: &str, dtype: GgmlType) -> Result<(), ConvertError> {
    if !is_flow_tensor_name(name) {
        return Err(ConvertError::Parse(format!(
            "{ARCH} flow: tensor `{name}` is outside the staged flow prefixes"
        )));
    }
    if dtype != GgmlType::F32 {
        return Err(ConvertError::Parse(format!(
            "{ARCH} flow: `{name}` must be F32"
        )));
    }
    Ok(())
}

fn validate_manifest_map(actual: &BTreeMap<String, Vec<u64>>) -> Result<(), ConvertError> {
    if actual.len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "{ARCH} flow: manifest has {}, expected {TENSOR_COUNT} tensors",
            actual.len()
        )));
    }
    if actual.keys().any(|name| !is_flow_tensor_name(name)) {
        return Err(ConvertError::Parse(
            "cosyvoice2 flow: manifest contains a name outside staged prefixes".to_owned(),
        ));
    }
    let digest = hex(&crate::models::canary_1b_flash::manifest_sha256(actual));
    if digest != MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "{ARCH} flow: manifest digest {digest} != pinned {MANIFEST_SHA256}"
        )));
    }
    Ok(())
}

fn is_flow_tensor_name(name: &str) -> bool {
    [
        "decoder.",
        "encoder.",
        "encoder_proj.",
        "input_embedding.",
        "spk_embed_affine_layer.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn write_no_replace(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "output has no file name")
    })?;
    let mut temporary = None;
    for attempt in 0..100u32 {
        let candidate = parent.join(format!(
            ".{}.vokra-cosyvoice2-flow-{}-{}",
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

fn hex(bytes: &[u8; 32]) -> String {
    crate::models::canary_1b_flash::hex(bytes)
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h = [
        0x67452301u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 80];
        for (index, word) in words[..16].iter_mut().enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes(chunk[start..start + 4].try_into().unwrap());
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (index, &word) in words.iter().enumerate() {
            let (f, k) = match index {
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
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut output = [0u8; 20];
    for (index, word) in h.into_iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    #[test]
    fn explicit_apache_attestation_is_required() {
        assert!(require_explicit_license(None).is_err());
        assert!(require_explicit_license(Some("mit")).is_err());
        require_explicit_license(Some("apache-2.0")).expect("Apache-2.0 is accepted");
    }

    #[test]
    fn wrong_config_identity_fails_closed() {
        let error = validate_config(b"wrong").expect_err("wrong config must fail");
        assert!(error.to_string().contains("identity mismatch"));
    }

    #[test]
    fn manifest_count_name_and_digest_are_fail_closed() {
        let empty = BTreeMap::new();
        assert!(validate_manifest_map(&empty).is_err());
        let mut outside = BTreeMap::new();
        outside.insert("conv_pre.weight".to_owned(), vec![1]);
        assert!(validate_manifest_map(&outside).is_err());
        assert!(validate_tensor_descriptor("decoder.weight", GgmlType::F16).is_err());
        assert!(validate_tensor_descriptor("conv_pre.weight", GgmlType::F32).is_err());
        let mut one = BTreeMap::new();
        one.insert("decoder.weight".to_owned(), vec![1]);
        assert_ne!(
            hex(&crate::models::canary_1b_flash::manifest_sha256(&one)),
            MANIFEST_SHA256
        );
    }

    #[test]
    fn metadata_stamps_raw_identity_and_unpinned_prepared_digest() {
        let mut builder = GgufBuilder::new();
        let prepared_sha = "a".repeat(64);
        stamp_metadata(&mut builder, 123, &prepared_sha);
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_CHECKPOINT_FILE)
                .and_then(|value| value.as_str()),
            Some(CHECKPOINT_FILE)
        );
        assert_eq!(
            file.get(KEY_CHECKPOINT_SHA256)
                .and_then(|value| value.as_str()),
            Some(CHECKPOINT_SHA256)
        );
        assert_eq!(
            file.get(KEY_PREPARED_BYTES)
                .and_then(|value| value.as_u64()),
            Some(123)
        );
        assert_eq!(
            file.get(KEY_PREPARED_SHA256)
                .and_then(|value| value.as_str()),
            Some(prepared_sha.as_str())
        );
        assert_eq!(
            file.get(KEY_PREPARED_STATUS)
                .and_then(|value| value.as_str()),
            Some("PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED")
        );
    }

    #[test]
    fn sha1_and_no_replace_contracts_hold() {
        assert_eq!(
            hex_bytes(&sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        let root =
            std::env::temp_dir().join(format!("vokra-cosyvoice2-flow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).expect("test directory");
        let output = root.join("model.gguf");
        std::fs::write(&output, b"original").expect("existing output");
        let error = write_no_replace(&output, b"replacement").expect_err("must not replace");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&output).expect("read output"), b"original");
        let fresh = root.join("fresh.gguf");
        write_no_replace(&fresh, b"complete").expect("publish absent output");
        assert_eq!(std::fs::read(&fresh).expect("read fresh"), b"complete");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn vast_handoff_paths_fail_closed_for_relative_missing_symlink_and_output() {
        let root = std::env::temp_dir().join(format!(
            "vokra-cosyvoice2-flow-paths-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).expect("temporary test directory");
        let regular = root.join("prepared.safetensors");
        std::fs::write(&regular, b"placeholder").expect("regular input");
        assert!(require_regular_file(Path::new("relative.safetensors"), "input").is_err());
        assert!(require_regular_file(&root.join("missing"), "input").is_err());
        #[cfg(unix)]
        {
            let link = root.join("link.safetensors");
            std::os::unix::fs::symlink(&regular, &link).expect("symlink");
            assert!(require_regular_file(&link, "input").is_err());
        }
        let existing = root.join("existing.gguf");
        std::fs::write(&existing, b"keep").expect("existing output");
        assert!(require_absent_output(&existing).is_err());
        assert!(require_absent_output(Path::new("relative.gguf")).is_err());
        assert!(require_absent_output(&root.join("new.gguf")).is_ok());
        let _ = std::fs::remove_dir_all(root);
    }

    fn required_vast_path(name: &str) -> std::path::PathBuf {
        let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
        let path = std::path::PathBuf::from(value);
        require_regular_file(&path, name).unwrap_or_else(|error| panic!("{name}: {error}"));
        path
    }

    fn required_vast_license(name: &str) -> String {
        let license = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
        require_explicit_license(Some(&license)).unwrap_or_else(|error| panic!("{name}: {error}"));
        license
    }

    fn validate_vast_prepared_manifest(path: &Path, component: &str, input: &Path) {
        let bytes = std::fs::read(path).expect("prepared manifest");
        let root = vokra_core::json::parse(&bytes).expect("prepared manifest JSON");
        let string = |object: &vokra_core::json::JsonValue, key: &str| {
            object
                .get(key)
                .and_then(vokra_core::json::JsonValue::as_str)
        };
        assert_eq!(
            string(&root, "format"),
            Some("vokra-cosyvoice2-component-prepared-safetensors-v1")
        );
        assert_eq!(string(&root, "status"), Some("PREPARED_SAFETENSORS_READY"));
        assert_eq!(string(&root, "component"), Some(component));
        let output_record = root.get("output").expect("prepared output record");
        assert_eq!(string(output_record, "path"), input.to_str());
        assert_eq!(
            output_record
                .get("bytes")
                .and_then(vokra_core::json::JsonValue::as_u64),
            Some(std::fs::metadata(input).unwrap().len())
        );
        let sha = hex(&crate::models::canary_1b_flash::sha256(
            &std::fs::read(input).expect("prepared input"),
        ));
        assert_eq!(string(output_record, "sha256"), Some(sha.as_str()));
        let execution = root.get("execution").expect("execution contract");
        assert_eq!(string(execution, "model_execution"), Some("NOT_RUN"));
        assert_eq!(string(execution, "torch_import"), Some("NOT_RUN"));
        assert_eq!(string(execution, "publication"), Some("NO_UPLOAD"));
    }

    /// VAST-only: converts one authenticated prepared component and leaves
    /// bind-only verification to the matching model-crate ignored test.
    #[test]
    #[ignore = "requires the authenticated VAST prepared CosyVoice2 flow artifact"]
    fn vast_real_prepared_flow_conversion() {
        let input = required_vast_path("VOKRA_COSYVOICE2_FLOW_PREPARED");
        let config = required_vast_path("VOKRA_COSYVOICE2_FLOW_CONFIG");
        let manifest = required_vast_path("VOKRA_COSYVOICE2_FLOW_PREPARED_MANIFEST");
        validate_vast_prepared_manifest(&manifest, COMPONENT, &input);
        let license = required_vast_license("VOKRA_COSYVOICE2_FLOW_LICENSE");
        let output = std::path::PathBuf::from(
            std::env::var("VOKRA_COSYVOICE2_FLOW_OUTPUT")
                .expect("VOKRA_COSYVOICE2_FLOW_OUTPUT is required"),
        );
        require_absent_output(&output).expect("flow output must be an absent absolute path");
        let report = convert_cosyvoice2_flow_file(&input, &config, &output, Some(&license))
            .expect("VAST prepared flow conversion");
        let file = GgufFile::open(&output).expect("converted flow GGUF");
        assert_eq!(file.tensors().len(), TENSOR_COUNT);
        assert_eq!(report.written, TENSOR_COUNT);
        assert_eq!(
            report.output_bytes,
            std::fs::metadata(&output).unwrap().len()
        );
        assert_eq!(
            file.get(KEY_COMPONENT).and_then(GgufMetadataValue::as_str),
            Some(COMPONENT)
        );
        assert_eq!(
            file.get(KEY_COMPOSITE_STATUS)
                .and_then(GgufMetadataValue::as_str),
            Some(COMPOSITE_STATUS)
        );
        assert!(
            manifest.is_file(),
            "manifest was validated as a regular file"
        );
    }
}
