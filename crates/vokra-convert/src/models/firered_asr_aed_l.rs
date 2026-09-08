//! FireRedASR-AED-L prepared-safetensors → GGUF converter.
//!
//! The only accepted input is the exact safetensors artifact emitted by the
//! VAST-only preparation sidecar. Every authenticated float tensor is copied
//! under its verbatim prepared name, and the complete name list is stamped as
//! a required-tensor manifest for the runtime binder. The exact inspected CMVN
//! and dictionary sidecars plus the official AED search defaults are stamped
//! as authenticated metadata; native beam execution and an independent oracle
//! remain parity-gated.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufStreamWriter, GgufTensorDecl,
    GgufValueType, chunks,
};

use crate::ConvertError;

/// Compatibility report retained by the converter dispatch API.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FireredAsrAedLReport {
    /// Source tensors observed by a successful conversion.
    pub read: usize,
    /// Tensors written by a successful conversion.
    pub written: usize,
    /// Non-floating tensors skipped by a successful conversion.
    pub skipped_non_float: usize,
    /// BF16 tensors passed through by a successful conversion.
    pub bf16_passthrough: usize,
}

/// Canonical architecture identifier retained for alias and dispatch tests.
#[allow(dead_code)] // Retained as inspection-only dispatch metadata until native parity is authenticated.
pub const ARCH: &str = "firered_asr_aed_l";

/// Canonical model name retained for metadata consumers.
#[allow(dead_code)] // Retained as inspection-only model metadata until native parity is authenticated.
pub const NAME: &str = "firered-asr-aed-l";

/// Canonical upstream repository.
#[allow(dead_code)] // Retained as inspection-only provenance until the checkpoint is authenticated.
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedASR-AED-L";

/// Fixed identity of the prepared artifact produced by the VAST bridge.
pub const UPSTREAM_REVISION: &str = "e57f5960d03cff1071ff7acbb409314d1e70ed3d";
pub const SOURCE_REVISION: &str = "834635e4cf277ed8ca92049fc375b17c3dc20748";
pub const CHECKPOINT_BYTES: u64 = 4_678_597_714;
pub const CHECKPOINT_SHA256: &str =
    "12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3";
pub const PREPARED_BYTES: u64 = 4_678_403_512;
pub const PREPARED_SHA256: &str =
    "5e8608d5a23af0761cb6bb52d08ee19a6476b8c324799eff3c63c9785cef583e";
pub const TENSOR_COUNT: usize = 940;

// Values observed and hash-bound by the authenticated VAST inspection.
pub const SAMPLE_RATE: u32 = 16_000;
pub const N_MELS: u32 = 80;
pub const VOCAB_SIZE: u32 = 7_832;
pub const ENCODER_LAYERS: u32 = 16;
pub const DECODER_LAYERS: u32 = 16;
pub const D_MODEL: u32 = 1_280;
pub const N_HEAD: u32 = 20;
pub const FFN_DIM: u32 = 5_120;
pub const KERNEL_SIZE: u32 = 33;
pub const BLANK_ID: u32 = 0;
pub const SOS_ID: u32 = 3;
pub const EOS_ID: u32 = 4;
pub const PAD_ID: u32 = 2;

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut words = [0u32; 64];
    for (index, chunk) in block.chunks_exact(4).take(16).enumerate() {
        words[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
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
    let mut work = *state;
    for index in 0..64 {
        let sum1 = work[4].rotate_right(6) ^ work[4].rotate_right(11) ^ work[4].rotate_right(25);
        let choice = (work[4] & work[5]) ^ (!work[4] & work[6]);
        let temp1 = work[7]
            .wrapping_add(sum1)
            .wrapping_add(choice)
            .wrapping_add(SHA256_K[index])
            .wrapping_add(words[index]);
        let sum0 = work[0].rotate_right(2) ^ work[0].rotate_right(13) ^ work[0].rotate_right(22);
        let majority = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
        let temp2 = sum0.wrapping_add(majority);
        work = [
            temp1.wrapping_add(temp2),
            work[0],
            work[1],
            work[2],
            work[3].wrapping_add(temp1),
            work[4],
            work[5],
            work[6],
        ];
    }
    for (value, delta) in state.iter_mut().zip(work) {
        *value = value.wrapping_add(delta);
    }
}

fn sha256_file(path: &Path) -> Result<String, ConvertError> {
    let mut file = std::fs::File::open(path)?;
    let length = file.metadata()?.len();
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut block = [0u8; 64];
    let mut buffered = 0usize;
    let mut chunk = [0u8; 1 << 20];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        let mut offset = 0;
        while offset < read {
            let count = (64 - buffered).min(read - offset);
            block[buffered..buffered + count].copy_from_slice(&chunk[offset..offset + count]);
            buffered += count;
            offset += count;
            if buffered == 64 {
                sha256_block(&mut state, &block);
                buffered = 0;
            }
        }
    }
    block[buffered] = 0x80;
    buffered += 1;
    if buffered > 56 {
        block[buffered..].fill(0);
        sha256_block(&mut state, &block);
        buffered = 0;
    }
    block[buffered..56].fill(0);
    block[56..].copy_from_slice(&(length * 8).to_be_bytes());
    sha256_block(&mut state, &block);
    let mut out = String::with_capacity(64);
    for value in state {
        out.push_str(&format!("{value:08x}"));
    }
    Ok(out)
}

fn read_authenticated_sidecar(
    path: &Path,
    label: &str,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<Vec<u8>, ConvertError> {
    let metadata = std::fs::symlink_metadata(path).map_err(ConvertError::Io)?;
    if metadata.file_type().is_symlink() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L {label} sidecar must not be a symlink"
        )));
    }
    if !metadata.file_type().is_file() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L {label} sidecar must be a regular file"
        )));
    }
    if metadata.len() != expected_bytes {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L {label} sidecar byte count mismatch: expected {expected_bytes}, got {}",
            metadata.len()
        )));
    }
    let bytes = std::fs::read(path).map_err(ConvertError::Io)?;
    if bytes.len() as u64 != expected_bytes {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L {label} sidecar byte count mismatch: expected {expected_bytes}, got {}",
            bytes.len()
        )));
    }
    let digest = sha256_bytes(&bytes);
    if digest != expected_sha256 {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L {label} sidecar SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )));
    }
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut block = [0u8; 64];
    let mut offset = 0usize;
    while offset + 64 <= bytes.len() {
        block.copy_from_slice(&bytes[offset..offset + 64]);
        sha256_block(&mut state, &block);
        offset += 64;
    }
    let tail = &bytes[offset..];
    block.fill(0);
    block[..tail.len()].copy_from_slice(tail);
    block[tail.len()] = 0x80;
    if tail.len() >= 56 {
        sha256_block(&mut state, &block);
        block.fill(0);
    }
    block[56..].copy_from_slice(&((bytes.len() as u64) * 8).to_be_bytes());
    sha256_block(&mut state, &block);
    state.iter().map(|value| format!("{value:08x}")).collect()
}

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_PROVENANCE_SOURCE_REVISION: &str = "vokra.provenance.source_revision";
const KEY_PROVENANCE_CHECKPOINT_BYTES: &str = "vokra.provenance.checkpoint_bytes";
const KEY_PROVENANCE_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const KEY_PROVENANCE_PREPARED_BYTES: &str = "vokra.provenance.prepared_bytes";
const KEY_PROVENANCE_PREPARED_SHA256: &str = "vokra.provenance.prepared_sha256";
const KEY_REQUIRED_TENSORS: &str = "vokra.firered_asr_aed_l.required_tensors";
const KEY_TENSOR_MANIFEST: &str = "vokra.firered_asr_aed_l.tensor_manifest";
pub const KEY_CMVN_TEXT: &str = "vokra.firered_asr_aed_l.cmvn_txt";
pub const KEY_CMVN_TEXT_SHA256: &str = "vokra.firered_asr_aed_l.cmvn_txt_sha256";
pub const KEY_DICT_TEXT: &str = "vokra.firered_asr_aed_l.dict_txt";
pub const KEY_DICT_TEXT_SHA256: &str = "vokra.firered_asr_aed_l.dict_txt_sha256";
pub const KEY_DECODE_POLICY_NAME: &str = "vokra.firered_asr_aed_l.search.name";
pub const KEY_DECODE_POLICY_BEAM_SIZE: &str = "vokra.firered_asr_aed_l.search.beam_size";
pub const KEY_DECODE_POLICY_NBEST: &str = "vokra.firered_asr_aed_l.search.nbest";
pub const KEY_DECODE_POLICY_MAX_LEN: &str = "vokra.firered_asr_aed_l.search.decode_max_len";
pub const KEY_DECODE_POLICY_SOFTMAX_SMOOTHING: &str =
    "vokra.firered_asr_aed_l.search.softmax_smoothing";
pub const KEY_DECODE_POLICY_LENGTH_PENALTY: &str = "vokra.firered_asr_aed_l.search.length_penalty";
pub const KEY_DECODE_POLICY_EOS_PENALTY: &str = "vokra.firered_asr_aed_l.search.eos_penalty";
pub const DECODE_POLICY_NAME: &str = "batch_beam_search";
pub const SEARCH_BEAM_SIZE: u32 = 3;
pub const SEARCH_NBEST: u32 = 1;
pub const SEARCH_DECODE_MAX_LEN: u32 = 0;
pub const SEARCH_SOFTMAX_SMOOTHING: f32 = 1.25;
pub const SEARCH_LENGTH_PENALTY: f32 = 0.6;
pub const SEARCH_EOS_PENALTY: f32 = 1.0;
pub const CMVN_TEXT_BYTES: u64 = 2_985;
pub const CMVN_TEXT_SHA256: &str =
    "11816db612b43318ab01f9cfd05ee121dd3900b7a39d893f59d0104a06c199d2";
pub const DICT_TEXT_BYTES: u64 = 71_448;
pub const DICT_TEXT_SHA256: &str =
    "6907215aeb034f6926b26bf8abfd650f756781622480a2342ec1f29b2072cafe";
const SPEC_KEYS: [(&str, u32); 16] = [
    ("vokra.firered_asr_aed_l.sample_rate", SAMPLE_RATE),
    ("vokra.firered_asr_aed_l.n_mels", N_MELS),
    ("vokra.firered_asr_aed_l.vocab_size", VOCAB_SIZE),
    ("vokra.firered_asr_aed_l.encoder.n_layer", ENCODER_LAYERS),
    ("vokra.firered_asr_aed_l.encoder.d_model", D_MODEL),
    ("vokra.firered_asr_aed_l.encoder.n_head", N_HEAD),
    ("vokra.firered_asr_aed_l.encoder.ffn_dim", FFN_DIM),
    ("vokra.firered_asr_aed_l.decoder.n_layer", DECODER_LAYERS),
    ("vokra.firered_asr_aed_l.decoder.d_model", D_MODEL),
    ("vokra.firered_asr_aed_l.decoder.n_head", N_HEAD),
    ("vokra.firered_asr_aed_l.decoder.ffn_dim", FFN_DIM),
    ("vokra.firered_asr_aed_l.encoder.kernel_size", KERNEL_SIZE),
    ("vokra.firered_asr_aed_l.blank_id", BLANK_ID),
    ("vokra.firered_asr_aed_l.sos_id", SOS_ID),
    ("vokra.firered_asr_aed_l.eos_id", EOS_ID),
    ("vokra.firered_asr_aed_l.pad_id", PAD_ID),
];
static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Owns a sibling temporary and removes it if conversion fails before the
/// final publish.  Keeping this guard local to this converter prevents a
/// failed 4.7 GB stream from leaving a file that looks like a valid GGUF.
struct AtomicOutputGuard {
    path: PathBuf,
    published: bool,
}

impl Drop for AtomicOutputGuard {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Publishes a completed sibling temporary without replacing a destination
/// that appeared after the initial path validation.  `hard_link` is an
/// atomic same-directory create and therefore gives this converter a
/// no-clobber finalization primitive on the Linux VAST worker.
fn publish_no_clobber(temp: &Path, destination: &Path) -> Result<(), ConvertError> {
    validate_publish_paths(temp, destination)?;
    std::fs::hard_link(temp, destination).map_err(ConvertError::Io)?;
    // The destination is now the published artifact.  Retiring our private
    // sibling is best-effort: a cleanup error must not turn a successful
    // publication into a false failure or remove the final artifact.
    let _ = std::fs::remove_file(temp);
    Ok(())
}

fn validate_publish_paths(temp: &Path, destination: &Path) -> Result<(), ConvertError> {
    reject_unsafe_path(temp, "temporary output")?;
    reject_unsafe_path(destination, "output")?;
    if temp.is_symlink() || !temp.is_file() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L temporary output must be a regular non-symlink file: {}",
            temp.display()
        )));
    }
    let parent = destination.parent().ok_or_else(|| {
        ConvertError::Usage("FireRedASR-AED-L output must have a parent directory".to_owned())
    })?;
    if !parent.is_dir() || parent.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L output parent must be an existing regular non-symlink directory: {}",
            parent.display()
        )));
    }
    // Re-resolve the canonical parent immediately before publication.  The
    // lexical walk above rejects unexpected symlink ancestry while allowing
    // macOS's conventional `/var` link.
    parent.canonicalize().map_err(ConvertError::Io)?;
    if destination.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L output must not be a symlink: {}",
            destination.display()
        )));
    }
    Ok(())
}

fn temporary_output_candidate(
    parent: &Path,
    output_name: &std::ffi::OsStr,
    sequence: u64,
) -> PathBuf {
    parent.join(format!(
        ".{}.tmp-{}-{}",
        output_name.to_string_lossy(),
        std::process::id(),
        sequence
    ))
}

fn create_temporary_output(
    parent: &Path,
    output_name: &std::ffi::OsStr,
    sequence: &AtomicU64,
) -> Result<(PathBuf, std::fs::File), ConvertError> {
    for _ in 0..32 {
        let candidate = temporary_output_candidate(
            parent,
            output_name,
            sequence.fetch_add(1, Ordering::Relaxed),
        );
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(ConvertError::Io(error)),
        }
    }
    Err(ConvertError::Usage(
        "FireRedASR-AED-L exhausted temporary output candidates".to_owned(),
    ))
}

/// Explicitly refuses the legacy generic-dispatch route: FireRed conversion
/// cannot proceed without caller-supplied authenticated `cmvn.txt` and
/// `dict.txt` sidecars. Use [`convert_firered_asr_aed_l_file_with_sidecars`].
/// The optional license override is also rejected by the sidecar-aware route
/// unless it matches the fixed upstream license contract.
pub fn convert_firered_asr_aed_l_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<FireredAsrAedLReport, ConvertError> {
    Err(ConvertError::Usage(
        "FireRedASR-AED-L conversion requires explicit cmvn.txt and dict.txt sidecars; use the sidecar-aware entrypoint".to_owned(),
    ))
}

/// Converts the prepared checkpoint and embeds the exact raw release
/// sidecars. The sidecars are explicit inputs so a VAST worker cannot silently
/// discover a replacement from another model directory.
pub fn convert_firered_asr_aed_l_file_with_sidecars(
    input: &Path,
    cmvn_path: &Path,
    dict_path: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FireredAsrAedLReport, ConvertError> {
    if license.is_some() {
        return Err(ConvertError::Usage(
            "FireRedASR-AED-L conversion has a fixed Apache-2.0 weight license; arbitrary --license overrides are refused".to_owned(),
        ));
    }
    validate_conversion_paths(input, cmvn_path, dict_path, output)?;
    // This operation is intentionally VAST-only: the prepared artifact is
    // ~4.7 GB and is never acquired or executed on the maintainer machine.
    // Hashing is streamed, and only one tensor payload is buffered below.
    if input.is_symlink() || output.is_symlink() {
        return Err(ConvertError::Usage(
            "FireRedASR-AED-L input/output must not be symlinks".to_owned(),
        ));
    }
    let input_path = input.canonicalize().map_err(ConvertError::Io)?;
    let output_path = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .map_err(ConvertError::Io)?
        .join(output.file_name().ok_or_else(|| {
            ConvertError::Usage("FireRedASR-AED-L output must name a file".to_owned())
        })?);
    if input_path == output_path {
        return Err(ConvertError::Usage(
            "FireRedASR-AED-L output must not alias the prepared input".to_owned(),
        ));
    }
    if output.exists() && !output.is_file() {
        return Err(ConvertError::Usage(
            "FireRedASR-AED-L output must be a regular file".to_owned(),
        ));
    }
    if output.exists() {
        return Err(ConvertError::Usage(
            "FireRedASR-AED-L output already exists; refusing to clobber an authenticated artifact"
                .to_owned(),
        ));
    }
    let output_name = output_path.file_name().ok_or_else(|| {
        ConvertError::Usage("FireRedASR-AED-L output must name a file".to_owned())
    })?;
    let input_bytes = std::fs::metadata(input)?.len();
    if input_bytes != PREPARED_BYTES {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L converter accepts only the VAST prepared safetensors artifact ({PREPARED_BYTES} bytes); got {} bytes",
            input_bytes
        )));
    }
    let digest = sha256_file(input)?;
    if digest != PREPARED_SHA256 {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L prepared safetensors SHA-256 mismatch: expected {PREPARED_SHA256}, got {digest}"
        )));
    }
    let mut st = crate::safetensors::SafetensorsFileReader::open(input)?;
    if st.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L prepared tensor-count mismatch: expected {TENSOR_COUNT}, got {}",
            st.tensors().len()
        )));
    }
    if st
        .tensors()
        .iter()
        .any(|tensor| tensor.dtype != GgmlType::F32)
    {
        return Err(ConvertError::Usage(
            "FireRedASR-AED-L prepared artifact contains a non-F32 tensor; regenerate with the audited VAST bridge".to_owned(),
        ));
    }
    let cmvn =
        read_authenticated_sidecar(cmvn_path, "cmvn.txt", CMVN_TEXT_BYTES, CMVN_TEXT_SHA256)?;
    let dict =
        read_authenticated_sidecar(dict_path, "dict.txt", DICT_TEXT_BYTES, DICT_TEXT_SHA256)?;

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, "asr");
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_PROVENANCE_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_PROVENANCE_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
    builder.add_metadata(
        KEY_PROVENANCE_CHECKPOINT_BYTES,
        GgufMetadataValue::U64(CHECKPOINT_BYTES),
    );
    builder.add_metadata(
        KEY_PROVENANCE_PREPARED_BYTES,
        GgufMetadataValue::U64(PREPARED_BYTES),
    );
    builder.add_string(KEY_PROVENANCE_PREPARED_SHA256, PREPARED_SHA256);
    let spdx = "apache-2.0";
    let class = LicenseClass::Permissive;
    vokra_core::stamp_provenance(
        &mut builder,
        class,
        spdx,
        Some(NAME),
        Some("FireRedTeam/FireRedASR-AED-L prepared F32 checkpoint"),
    );
    for (key, value) in SPEC_KEYS {
        builder.add_u32(key, value);
    }
    builder.add_string(KEY_DECODE_POLICY_NAME, DECODE_POLICY_NAME);
    builder.add_u32(KEY_DECODE_POLICY_BEAM_SIZE, SEARCH_BEAM_SIZE);
    builder.add_u32(KEY_DECODE_POLICY_NBEST, SEARCH_NBEST);
    builder.add_u32(KEY_DECODE_POLICY_MAX_LEN, SEARCH_DECODE_MAX_LEN);
    builder.add_f32(
        KEY_DECODE_POLICY_SOFTMAX_SMOOTHING,
        SEARCH_SOFTMAX_SMOOTHING,
    );
    builder.add_f32(KEY_DECODE_POLICY_LENGTH_PENALTY, SEARCH_LENGTH_PENALTY);
    builder.add_f32(KEY_DECODE_POLICY_EOS_PENALTY, SEARCH_EOS_PENALTY);
    for (key, payload, digest) in [
        (KEY_CMVN_TEXT, cmvn.as_slice(), CMVN_TEXT_SHA256),
        (KEY_DICT_TEXT, dict.as_slice(), DICT_TEXT_SHA256),
    ] {
        builder.add_metadata(
            key,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: payload.iter().copied().map(GgufMetadataValue::U8).collect(),
            }),
        );
        let digest_key = if key == KEY_CMVN_TEXT {
            KEY_CMVN_TEXT_SHA256
        } else {
            KEY_DICT_TEXT_SHA256
        };
        builder.add_string(digest_key, digest);
    }
    builder.add_metadata(
        KEY_REQUIRED_TENSORS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: st
                .tensors()
                .iter()
                .map(|tensor| GgufMetadataValue::String(tensor.name.clone()))
                .collect(),
        }),
    );
    // The name-only declaration catches truncation; this parallel field
    // carries the exact dtype and GGUF dimension contract.  It is encoded as
    // `name|dtype-tag|dim,dim,...` to stay within GGUF Array<String> and is
    // checked against the actual tensor descriptors by the native binder.
    builder.add_metadata(
        KEY_TENSOR_MANIFEST,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: st
                .tensors()
                .iter()
                .map(|tensor| {
                    let dims = tensor
                        .shape
                        .iter()
                        .map(|dim| dim.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    GgufMetadataValue::String(format!(
                        "{}|{}|{}",
                        tensor.name,
                        tensor.dtype.tag(),
                        dims
                    ))
                })
                .collect(),
        }),
    );
    let declarations: Vec<GgufTensorDecl> = st
        .tensors()
        .iter()
        .map(|tensor| GgufTensorDecl {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            dimensions: tensor.shape.clone(),
        })
        .collect();
    let (temporary_path, output_file) = create_temporary_output(
        output_path
            .parent()
            .expect("canonical output path has a parent"),
        output_name,
        &OUTPUT_SEQUENCE,
    )?;
    let mut temporary = AtomicOutputGuard {
        path: temporary_path,
        published: false,
    };
    let mut writer = GgufStreamWriter::begin(
        std::io::BufWriter::new(output_file),
        &builder,
        &declarations,
    )
    .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    let mut payload = Vec::new();
    let mut report = FireredAsrAedLReport::default();
    for tensor in &declarations {
        report.read += 1;
        st.read_tensor_into(&tensor.name, &mut payload)?;
        writer
            .write_tensor(&tensor.name, &payload)
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
        report.written += 1;
    }
    let output_file = writer
        .finish()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?
        .into_inner()
        .map_err(|error| ConvertError::Io(error.into_error()))?;
    output_file.sync_all().map_err(ConvertError::Io)?;
    // The destination was validated absent above and the temporary lives in
    // the same canonical directory.  A hard-link create is atomic and
    // no-clobber, so a concurrent creator wins rather than being overwritten.
    publish_no_clobber(&temporary.path, &output_path)?;
    temporary.published = true;
    Ok(report)
}

fn validate_conversion_paths(
    input: &Path,
    cmvn_path: &Path,
    dict_path: &Path,
    output: &Path,
) -> Result<(), ConvertError> {
    for (path, label) in [
        (input, "checkpoint"),
        (cmvn_path, "cmvn.txt"),
        (dict_path, "dict.txt"),
        (output, "output"),
    ] {
        reject_unsafe_path(path, label)?;
    }
    if output.exists() || output.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L output already exists or is symlinked: {}",
            output.display()
        )));
    }
    for (path, label) in [
        (input, "checkpoint"),
        (cmvn_path, "cmvn.txt"),
        (dict_path, "dict.txt"),
    ] {
        if path.is_symlink() || !path.is_file() {
            return Err(ConvertError::Usage(format!(
                "FireRedASR-AED-L {label} must be a regular non-symlink file: {}",
                path.display()
            )));
        }
    }
    let parent = output.parent().ok_or_else(|| {
        ConvertError::Usage("FireRedASR-AED-L output must have a parent directory".to_owned())
    })?;
    if !parent.is_dir() || parent.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L output parent must be an existing regular non-symlink directory: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn reject_unsafe_path(path: &Path, label: &str) -> Result<(), ConvertError> {
    let raw = path.to_string_lossy();
    #[cfg(windows)]
    let has_lexical_dot = raw
        .split(['/', '\\'])
        .any(|component| matches!(component, "." | ".."));
    #[cfg(not(windows))]
    let has_lexical_dot = raw
        .split('/')
        .any(|component| matches!(component, "." | ".."));
    if has_lexical_dot
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(ConvertError::Usage(format!(
            "FireRedASR-AED-L {label} must not contain lexical dot components"
        )));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(ConvertError::Io)?
            .join(path)
    };
    let mut current = absolute.as_path();
    loop {
        if current.is_symlink() && current != Path::new("/var") {
            return Err(ConvertError::Usage(format!(
                "FireRedASR-AED-L {label} has symlink ancestry: {}",
                current.display()
            )));
        }
        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_inputs_are_refused_without_output() {
        let root =
            std::env::temp_dir().join(format!("vokra-firered-refusal-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("temp directory");
        let input = root.join("arbitrary.safetensors");
        let output = root.join("must-not-exist.gguf");
        std::fs::write(&input, b"arbitrary").expect("input");
        let error = convert_firered_asr_aed_l_file(&input, &output, None)
            .expect_err("non-authenticated prepared input must refuse");
        assert!(error.to_string().contains("explicit cmvn.txt and dict.txt"));
        assert!(!output.exists());
        let error = convert_firered_asr_aed_l_file_with_sidecars(
            &input,
            &root.join("cmvn.txt"),
            &root.join("dict.txt"),
            &output,
            Some("mit"),
        )
        .expect_err("license override must be refused");
        assert!(error.to_string().contains("fixed Apache-2.0"));
        assert!(!output.exists());
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn existing_output_is_never_clobbered() {
        let root =
            std::env::temp_dir().join(format!("vokra-firered-no-clobber-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("temp directory");
        let input = root.join("arbitrary.safetensors");
        let output = root.join("existing.gguf");
        std::fs::write(&input, b"arbitrary").expect("input");
        std::fs::write(&output, b"sentinel").expect("output");
        let error = convert_firered_asr_aed_l_file_with_sidecars(
            &input,
            &root.join("cmvn.txt"),
            &root.join("dict.txt"),
            &output,
            None,
        )
        .expect_err("existing output must be rejected before any stream");
        assert!(error.to_string().contains("already exists"));
        assert_eq!(std::fs::read(&output).expect("sentinel"), b"sentinel");
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn conversion_paths_reject_dot_components_and_symlink_ancestors() {
        let root = std::env::temp_dir().join(format!(
            "vokra-firered-path-boundary-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temp directory");
        let input = root.join("prepared.safetensors");
        let cmvn = root.join("cmvn.txt");
        let dict = root.join("dict.txt");
        let output = root.join("output.gguf");
        for path in [&input, &cmvn, &dict] {
            std::fs::write(path, b"fixture").expect("fixture file");
        }
        validate_conversion_paths(&input, &cmvn, &dict, &output)
            .expect("regular conversion paths accepted");
        let dotted = root.join(".").join("output.gguf");
        assert!(reject_unsafe_path(&dotted, "output").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let real_parent = root.join("real-parent");
            let linked_parent = root.join("linked-parent");
            std::fs::create_dir(&real_parent).expect("real parent");
            symlink(&real_parent, &linked_parent).expect("linked parent");
            let linked_input = linked_parent.join("prepared.safetensors");
            let linked_cmvn = linked_parent.join("cmvn.txt");
            let linked_dict = linked_parent.join("dict.txt");
            let linked_output = linked_parent.join("output.gguf");
            assert!(
                validate_conversion_paths(
                    &linked_input,
                    &linked_cmvn,
                    &linked_dict,
                    &linked_output
                )
                .expect_err("symlink ancestry accepted")
                .to_string()
                .contains("symlink ancestry")
            );
        }
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn authenticated_sidecar_helper_accepts_exact_bytes_and_rejects_drift() {
        let root = std::env::temp_dir().join(format!(
            "vokra-firered-sidecar-helper-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temp directory");
        let path = root.join("cmvn.txt");
        std::fs::write(&path, b"abc").expect("sidecar");
        let digest = sha256_bytes(b"abc");
        assert_eq!(
            read_authenticated_sidecar(&path, "cmvn.txt", 3, &digest).expect("exact sidecar"),
            b"abc"
        );
        let wrong_size = read_authenticated_sidecar(&path, "cmvn.txt", 4, &digest)
            .expect_err("wrong byte count must fail");
        assert!(wrong_size.to_string().contains("byte count mismatch"));
        let wrong_hash = read_authenticated_sidecar(&path, "cmvn.txt", 3, "0".repeat(64).as_str())
            .expect_err("wrong digest must fail");
        assert!(wrong_hash.to_string().contains("SHA-256 mismatch"));
        let missing = read_authenticated_sidecar(&root.join("missing.txt"), "cmvn.txt", 3, &digest)
            .expect_err("missing sidecar must fail");
        assert!(matches!(missing, ConvertError::Io(_)));
        #[cfg(unix)]
        {
            let link = root.join("link.txt");
            std::os::unix::fs::symlink(&path, &link).expect("symlink");
            let error = read_authenticated_sidecar(&link, "cmvn.txt", 3, &digest)
                .expect_err("symlink sidecar must fail");
            assert!(error.to_string().contains("must not be a symlink"));
        }
        let directory = root.join("directory");
        std::fs::create_dir(&directory).expect("directory sidecar");
        let error = read_authenticated_sidecar(&directory, "cmvn.txt", 3, &digest)
            .expect_err("directory sidecar must fail before read");
        assert!(error.to_string().contains("regular file"));
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn failed_atomic_output_guard_removes_only_its_temp() {
        let root = std::env::temp_dir().join(format!(
            "vokra-firered-atomic-cleanup-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temp directory");
        let temporary = root.join(".output.gguf.tmp-test");
        std::fs::write(&temporary, b"partial").expect("partial temp");
        {
            let _guard = AtomicOutputGuard {
                path: temporary.clone(),
                published: false,
            };
        }
        assert!(!temporary.exists(), "failed stream temp must be cleaned");
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn concurrent_destination_is_rejected_without_clobbering() {
        let root =
            std::env::temp_dir().join(format!("vokra-firered-publish-race-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("temp directory");
        let temporary = root.join(".output.gguf.tmp");
        let destination = root.join("output.gguf");
        std::fs::write(&temporary, b"complete").expect("temporary");
        std::fs::write(&destination, b"racing-writer").expect("destination");
        let error = publish_no_clobber(&temporary, &destination)
            .expect_err("a destination created after validation must win");
        assert!(matches!(
            error,
            ConvertError::Io(ref io_error)
                if io_error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(
            std::fs::read(&destination).expect("destination"),
            b"racing-writer"
        );
        assert!(
            temporary.exists(),
            "failed publication retains its own temp for guard cleanup"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn temporary_output_retries_after_candidate_collision() {
        let root = std::env::temp_dir().join(format!(
            "vokra-firered-temp-sequence-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temp directory");
        let output_name = std::ffi::OsStr::new("output.gguf");
        let sequence = AtomicU64::new(0);
        let occupied = temporary_output_candidate(&root, output_name, 0);
        std::fs::write(&occupied, b"occupied").expect("occupied candidate");
        let (candidate, file) = create_temporary_output(&root, output_name, &sequence)
            .expect("next candidate should be created");
        drop(file);
        assert_ne!(candidate, occupied);
        assert!(candidate.exists());
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn streaming_sha256_matches_standard_vectors() {
        let root = std::env::temp_dir().join(format!("vokra-firered-sha-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("temp directory");
        let path = root.join("payload");

        std::fs::write(&path, []).expect("empty input");
        assert_eq!(
            sha256_file(&path).expect("empty hash"),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        std::fs::write(&path, b"abc").expect("abc input");
        assert_eq!(
            sha256_file(&path).expect("hash"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        // This crosses the 64-byte compression-block boundary many times and
        // is the standard SHA-256 million-'a' test vector.
        std::fs::write(&path, vec![b'a'; 1_000_000]).expect("multi-block input");
        assert_eq!(
            sha256_file(&path).expect("multi-block hash"),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
