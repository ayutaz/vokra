//! Strict loader for the official VibeVoice Realtime v1 CPU noise tape.
//!
//! This consumes the producer's existing reference.json plus cpu/*.npy
//! records. It authenticates the reference input and its complete two-row
//! noise files, but it does not authorize model or audio execution, validate a
//! native GGUF, bind an owner/license/preset/tokenizer, or prove parity.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use vokra_core::json::{JsonValue, parse as parse_json};

use super::sha256;

const FORMAT: &str = "vokra-vibevoice-realtime-streaming-reference-v1";
const OPEN_STATUS: &str = "REFERENCE_RUN_OPEN_NOT_RUST_PARITY";
const PUBLICATION: &str = "NO_UPLOAD";
const SOURCE_REPOSITORY: &str = "microsoft/VibeVoice";
const SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
const SOURCE_ORIGIN: &str = "https://github.com/microsoft/VibeVoice.git";
const CHECKPOINT_BYTES: u64 = 2_035_332_888;
const CHECKPOINT_SHA256: &str = "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514";
const PRESET_PAYLOAD_SHA256: &str =
    "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897";
const MAX_NOISE_DRAWS: usize = 64;
const NOISE_ROWS: usize = 2;
const NOISE_WIDTH: usize = 64;
const NOISE_VALUES: usize = NOISE_ROWS * NOISE_WIDTH;
// Engineering input limits only: oversized reference packets/noise files are
// loud unsupported inputs. These are not model shapes, parity bounds, or an
// authorization decision.
const MAX_REFERENCE_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_NOISE_FILE_BYTES: u64 = 64 * 1024;

/// Authenticated native input rows from the producer's [2, 64] tape.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VibeVoiceRealtimeNoiseTape {
    /// First row of each official record, in packet order.
    pub(crate) draws: Vec<Vec<f32>>,
}

impl VibeVoiceRealtimeNoiseTape {
    pub(crate) fn len(&self) -> usize {
        self.draws.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }
}

/// Loads and authenticates the existing official v1 CPU noise tape.
pub(crate) fn load_vibevoice_realtime_noise_tape(
    reference_dir: &Path,
    expected_reference_sha256: &str,
) -> Result<VibeVoiceRealtimeNoiseTape, String> {
    let expected_packet_sha =
        parse_sha256(expected_reference_sha256, "expected reference SHA-256")?;
    ensure_directory(reference_dir, "reference directory")?;
    let cpu_dir = reference_dir.join("cpu");
    ensure_directory(&cpu_dir, "reference cpu directory")?;
    let packet_path = reference_dir.join("reference.json");
    ensure_regular_file(&packet_path, "reference.json")?;

    let packet_bytes = read_regular_file(&packet_path, "reference.json", MAX_REFERENCE_JSON_BYTES)?;
    let actual_packet_sha = sha256(&packet_bytes);
    if actual_packet_sha != expected_packet_sha {
        return Err(format!(
            "reference.json SHA-256 mismatch: expected {}, got {}",
            hex_digest(&expected_packet_sha),
            hex_digest(&actual_packet_sha)
        ));
    }
    let packet = parse_json(&packet_bytes).map_err(|error| format!("reference.json: {error}"))?;
    reject_duplicate_keys(&packet, "reference")?;
    validate_packet_identity(&packet)?;

    let packets = as_object(
        object_field(&packet, "packets", "reference")?,
        "reference.packets",
    )?;
    let cpu = object_value(packets, "cpu", "reference.packets")?;
    if string_field(cpu, "device", "reference.packets.cpu")? != "cpu" {
        return Err("reference.packets.cpu.device must be cpu".to_owned());
    }
    if !bool_field(cpu, "noise_matching", "reference.packets.cpu")? {
        return Err("reference.packets.cpu.noise_matching must be true".to_owned());
    }

    let records = array_field(cpu, "diffusion_initial_noise", "reference.packets.cpu")?;
    let draw_count = usize_field(cpu, "noise_draws", "reference.packets.cpu")?;
    if draw_count == 0 || draw_count > MAX_NOISE_DRAWS {
        return Err(format!(
            "reference.packets.cpu.noise_draws must be in 1..={MAX_NOISE_DRAWS}"
        ));
    }
    if draw_count != records.len() {
        return Err(format!(
            "reference.packets.cpu.noise_draws ({draw_count}) does not match diffusion_initial_noise ({})",
            records.len()
        ));
    }

    let hashes = array_field(cpu, "noise_hashes", "reference.packets.cpu")?;
    if hashes.len() != records.len() {
        return Err(format!(
            "reference.packets.cpu.noise_hashes ({}) does not match diffusion_initial_noise ({})",
            hashes.len(),
            records.len()
        ));
    }

    let mut filenames = BTreeSet::new();
    let mut draws = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let tensor = parse_tensor_record(record, index)?;
        validate_basename(&tensor.file, index)?;
        let expected_filename = format!("diffusion_initial_noise_{index}.npy");
        if tensor.file != expected_filename {
            return Err(format!(
                "diffusion_initial_noise[{index}] file must be {expected_filename}"
            ));
        }
        if !filenames.insert(tensor.file.clone()) {
            return Err(format!(
                "diffusion_initial_noise[{index}] repeats file {}",
                tensor.file
            ));
        }
        if tensor.shape != [NOISE_ROWS, NOISE_WIDTH] {
            return Err(format!(
                "diffusion_initial_noise[{index}] shape must be [2, 64], got {:?}",
                tensor.shape
            ));
        }
        if tensor.dtype != "float32" || !tensor.finite {
            return Err(format!(
                "diffusion_initial_noise[{index}] must declare finite float32"
            ));
        }
        let payload_sha = parse_sha256(
            string_value(
                &hashes[index],
                &format!("reference.packets.cpu.noise_hashes[{index}]"),
            )?,
            &format!("noise payload hash {index}"),
        )?;
        let path = cpu_dir.join(&tensor.file);
        ensure_regular_file(&path, &format!("noise file {index}"))?;
        let bytes = read_regular_file(&path, &format!("noise file {index}"), MAX_NOISE_FILE_BYTES)?;
        let actual_file_sha = sha256(&bytes);
        if actual_file_sha != tensor.sha256 {
            return Err(format!(
                "noise file {index} SHA-256 mismatch: expected {}, got {}",
                hex_digest(&tensor.sha256),
                hex_digest(&actual_file_sha)
            ));
        }
        let npy = parse_npy_f32(&bytes, index)?;
        let actual_payload_sha = sha256(npy.payload);
        if actual_payload_sha != payload_sha {
            return Err(format!(
                "noise payload {index} SHA-256 mismatch: expected {}, got {}",
                hex_digest(&payload_sha),
                hex_digest(&actual_payload_sha)
            ));
        }
        draws.push(npy.values[..NOISE_WIDTH].to_vec());
    }

    Ok(VibeVoiceRealtimeNoiseTape { draws })
}

fn validate_packet_identity(packet: &JsonValue) -> Result<(), String> {
    if string_field(packet, "format", "reference")? != FORMAT {
        return Err(format!("reference.format must be {FORMAT}"));
    }
    if string_field(packet, "status", "reference")? != OPEN_STATUS {
        return Err(format!("reference.status must be {OPEN_STATUS}"));
    }
    if string_field(packet, "publication", "reference")? != PUBLICATION {
        return Err("reference.publication must be NO_UPLOAD".to_owned());
    }
    let source = object_field(packet, "source", "reference")?;
    if string_field(source, "repository", "reference.source")? != SOURCE_REPOSITORY
        || string_field(source, "revision", "reference.source")? != SOURCE_REVISION
        || string_field(source, "origin", "reference.source")? != SOURCE_ORIGIN
    {
        return Err("reference.source does not match pinned Microsoft identity".to_owned());
    }
    let checkpoint = object_field(packet, "checkpoint", "reference")?;
    if u64_field(checkpoint, "bytes", "reference.checkpoint")? != CHECKPOINT_BYTES
        || string_field(checkpoint, "sha256", "reference.checkpoint")? != CHECKPOINT_SHA256
    {
        return Err("reference.checkpoint does not match pinned identity".to_owned());
    }
    let preset = object_field(packet, "preset", "reference")?;
    if string_field(preset, "payload_sha256", "reference.preset")? != PRESET_PAYLOAD_SHA256 {
        return Err("reference.preset does not match pinned Carter identity".to_owned());
    }
    Ok(())
}

#[derive(Debug)]
struct TensorRecord {
    file: String,
    shape: Vec<usize>,
    dtype: String,
    sha256: [u8; 32],
    finite: bool,
}

fn parse_tensor_record(value: &JsonValue, index: usize) -> Result<TensorRecord, String> {
    let label = format!("diffusion_initial_noise[{index}]");
    let object = as_object(value, &label)?;
    const REQUIRED: &[&str] = &["file", "shape", "dtype", "sha256", "finite"];
    for (key, _) in object {
        if !REQUIRED.contains(&key.as_str()) {
            return Err(format!("{label} has unsupported field {key}"));
        }
    }
    for key in REQUIRED {
        if !object.iter().any(|(name, _)| name == key) {
            return Err(format!("{label} is missing {key}"));
        }
    }
    let shape_values = array_field(value, "shape", &label)?;
    if shape_values.len() != 2 {
        return Err(format!("{label}.shape must have exactly two dimensions"));
    }
    let mut shape = Vec::with_capacity(2);
    for (dimension, value) in shape_values.iter().enumerate() {
        shape.push(usize_value(value, &format!("{label}.shape[{dimension}]"))?);
    }
    Ok(TensorRecord {
        file: string_field(value, "file", &label)?.to_owned(),
        shape,
        dtype: string_field(value, "dtype", &label)?.to_owned(),
        sha256: parse_sha256(
            string_field(value, "sha256", &label)?,
            &format!("{label}.sha256"),
        )?,
        finite: bool_field(value, "finite", &label)?,
    })
}

#[derive(Debug)]
struct NpyF32<'a> {
    values: Vec<f32>,
    payload: &'a [u8],
}

fn parse_npy_f32<'a>(bytes: &'a [u8], index: usize) -> Result<NpyF32<'a>, String> {
    let label = format!("noise file {index}");
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{label} is not an NPY file"));
    }
    let (header_start, header_len) = match (bytes[6], bytes[7]) {
        (1, 0) => (
            10usize,
            usize::from(u16::from_le_bytes([bytes[8], bytes[9]])),
        ),
        (2, 0) | (3, 0) => {
            if bytes.len() < 12 {
                return Err(format!("{label} has a truncated NPY header"));
            }
            (
                12usize,
                usize::try_from(u32::from_le_bytes(bytes[8..12].try_into().unwrap()))
                    .map_err(|_| format!("{label} header length overflows usize"))?,
            )
        }
        _ => return Err(format!("{label} uses an unsupported NPY version")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| format!("{label} header overflows"))?;
    if header_end > bytes.len() {
        return Err(format!("{label} has a truncated NPY header"));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| format!("{label} NPY header is not UTF-8"))?;
    validate_npy_header(header, &label)?;
    let payload_len = NOISE_VALUES
        .checked_mul(4)
        .ok_or_else(|| format!("{label} payload length overflows"))?;
    let payload_end = header_end
        .checked_add(payload_len)
        .ok_or_else(|| format!("{label} payload length overflows"))?;
    if payload_end != bytes.len() {
        return Err(format!("{label} has trailing or truncated NPY bytes"));
    }
    let payload = &bytes[header_end..payload_end];
    let values = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if !values.iter().all(|value| value.is_finite()) {
        return Err(format!("{label} contains non-finite values"));
    }
    Ok(NpyF32 { values, payload })
}

struct NpyHeaderParser<'a> {
    input: &'a [u8],
    position: usize,
    label: &'a str,
}

impl NpyHeaderParser<'_> {
    fn error(&self, reason: &str) -> String {
        format!(
            "{} NPY header at byte {}: {reason}",
            self.label, self.position
        )
    }

    fn whitespace(&mut self) {
        while matches!(
            self.input.get(self.position),
            Some(b' ' | b'\t' | b'\r' | b'\n')
        ) {
            self.position += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.input.get(self.position) == Some(&byte) {
            self.position += 1;
            Ok(())
        } else {
            Err(self.error("unexpected token"))
        }
    }

    fn quoted(&mut self) -> Result<&str, String> {
        let quote = *self
            .input
            .get(self.position)
            .ok_or_else(|| self.error("missing quoted string"))?;
        if quote != b'\'' && quote != b'"' {
            return Err(self.error("expected quoted string"));
        }
        self.position += 1;
        let start = self.position;
        while let Some(&byte) = self.input.get(self.position) {
            if byte == b'\\' {
                return Err(self.error("escaped header strings are unsupported"));
            }
            if byte == quote {
                let value = std::str::from_utf8(&self.input[start..self.position])
                    .map_err(|_| self.error("quoted string is not UTF-8"))?;
                self.position += 1;
                return Ok(value);
            }
            self.position += 1;
        }
        Err(self.error("unterminated quoted string"))
    }

    fn identifier(&mut self) -> Result<&str, String> {
        let start = self.position;
        while matches!(
            self.input.get(self.position),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'_')
        ) {
            self.position += 1;
        }
        if start == self.position {
            return Err(self.error("expected identifier"));
        }
        std::str::from_utf8(&self.input[start..self.position])
            .map_err(|_| self.error("identifier is not UTF-8"))
    }

    fn integer(&mut self) -> Result<usize, String> {
        let start = self.position;
        while matches!(self.input.get(self.position), Some(b'0'..=b'9')) {
            self.position += 1;
        }
        if start == self.position {
            return Err(self.error("expected non-negative integer"));
        }
        std::str::from_utf8(&self.input[start..self.position])
            .map_err(|_| self.error("shape integer is not UTF-8"))?
            .parse()
            .map_err(|_| self.error("shape integer overflows usize"))
    }
}

fn validate_npy_header(header: &str, label: &str) -> Result<(), String> {
    let mut parser = NpyHeaderParser {
        input: header.as_bytes(),
        position: 0,
        label,
    };
    parser.whitespace();
    parser.expect(b'{')?;
    let mut seen_descr = false;
    let mut seen_fortran = false;
    let mut seen_shape = false;
    loop {
        parser.whitespace();
        if parser.input.get(parser.position) == Some(&b'}') {
            parser.position += 1;
            break;
        }
        let key = parser.quoted()?.to_owned();
        parser.whitespace();
        parser.expect(b':')?;
        parser.whitespace();
        match key.as_str() {
            "descr" => {
                if seen_descr {
                    return Err(parser.error("duplicate descr key"));
                }
                seen_descr = true;
                if parser.quoted()? != "<f4" {
                    return Err(parser.error("descr must be little-endian float32"));
                }
            }
            "fortran_order" => {
                if seen_fortran {
                    return Err(parser.error("duplicate fortran_order key"));
                }
                seen_fortran = true;
                if parser.identifier()? != "False" {
                    return Err(parser.error("fortran_order must be False"));
                }
            }
            "shape" => {
                if seen_shape {
                    return Err(parser.error("duplicate shape key"));
                }
                seen_shape = true;
                parser.expect(b'(')?;
                parser.whitespace();
                let rows = parser.integer()?;
                parser.whitespace();
                parser.expect(b',')?;
                parser.whitespace();
                let columns = parser.integer()?;
                parser.whitespace();
                if parser.input.get(parser.position) == Some(&b',') {
                    parser.position += 1;
                    parser.whitespace();
                }
                parser.expect(b')')?;
                if rows != NOISE_ROWS || columns != NOISE_WIDTH {
                    return Err(parser.error("shape must be (2, 64)"));
                }
            }
            _ => return Err(parser.error("unsupported NPY header key")),
        }
        parser.whitespace();
        if parser.input.get(parser.position) == Some(&b',') {
            parser.position += 1;
            continue;
        }
        parser.expect(b'}')?;
        break;
    }
    parser.whitespace();
    if parser.position != parser.input.len() {
        return Err(parser.error("trailing header tokens"));
    }
    if !seen_descr || !seen_fortran || !seen_shape {
        return Err(parser.error("header is missing a required key"));
    }
    Ok(())
}

fn ensure_directory(path: &Path, label: &str) -> Result<(), String> {
    ensure_no_symlink_ancestors(path, label)?;
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !metadata.file_type().is_dir() {
        return Err(format!("{label} must be a regular directory"));
    }
    Ok(())
}

pub(super) fn ensure_regular_file(path: &Path, label: &str) -> Result<(), String> {
    ensure_no_symlink_ancestors(path, label)?;
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    Ok(())
}

fn ensure_no_symlink_ancestors(path: &Path, label: &str) -> Result<(), String> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("{label} current directory failed: {error}"))?
            .join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(std::path::MAIN_SEPARATOR.to_string()),
            Component::CurDir => {}
            Component::ParentDir => current.push(".."),
            Component::Normal(part) => current.push(part),
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "{label} has a symlink ancestor: {}",
                    current.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "{label} metadata failed at {}: {error}",
                    current.display()
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn read_regular_file(
    path: &Path,
    label: &str,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    let mut file = fs::File::open(path).map_err(|error| format!("{label} open failed: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err(format!("{label} is not a regular file"));
    }
    let expected_len = metadata.len();
    if expected_len > max_bytes {
        return Err(format!(
            "{label} exceeds the {max_bytes}-byte resource limit"
        ));
    }
    let capacity = usize::try_from(expected_len)
        .map_err(|_| format!("{label} is too large for this platform"))?;
    let read_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| format!("{label} read limit overflows"))?;
    let mut bytes = Vec::with_capacity(capacity);
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{label} read failed: {error}"))?;
    if bytes.len() as u64 > max_bytes || bytes.len() as u64 != expected_len {
        return Err(format!("{label} changed while being read"));
    }
    let final_len = file
        .metadata()
        .map_err(|error| format!("{label} final metadata failed: {error}"))?
        .len();
    if final_len != expected_len {
        return Err(format!("{label} changed while being read"));
    }
    Ok(bytes)
}

fn validate_basename(file: &str, index: usize) -> Result<(), String> {
    if file.is_empty() || file.contains('\\') {
        return Err(format!(
            "diffusion_initial_noise[{index}] file is not a safe basename"
        ));
    }
    let mut components = Path::new(file).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(format!(
            "diffusion_initial_noise[{index}] file is not a safe basename"
        )),
    }
}

pub(super) fn reject_duplicate_keys(value: &JsonValue, path: &str) -> Result<(), String> {
    match value {
        JsonValue::Object(entries) => {
            let mut keys = BTreeSet::new();
            for (key, child) in entries {
                if !keys.insert(key.as_str()) {
                    return Err(format!("duplicate JSON key {path}.{key}"));
                }
                reject_duplicate_keys(child, &format!("{path}.{key}"))?;
            }
        }
        JsonValue::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_duplicate_keys(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn as_object<'a>(value: &'a JsonValue, label: &str) -> Result<&'a [(String, JsonValue)], String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn object_field<'a>(
    value: &'a JsonValue,
    name: &str,
    label: &str,
) -> Result<&'a JsonValue, String> {
    let value = field(value, name, label)?;
    as_object(value, &format!("{label}.{name}"))?;
    Ok(value)
}

fn object_value<'a>(
    object: &'a [(String, JsonValue)],
    name: &str,
    label: &str,
) -> Result<&'a JsonValue, String> {
    let value = object
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .ok_or_else(|| format!("{label} is missing {name}"))?;
    as_object(value, &format!("{label}.{name}"))?;
    Ok(value)
}

fn field<'a>(value: &'a JsonValue, name: &str, label: &str) -> Result<&'a JsonValue, String> {
    value
        .as_object()
        .and_then(|entries| entries.iter().find(|(key, _)| key == name))
        .map(|(_, value)| value)
        .ok_or_else(|| format!("{label} is missing {name}"))
}

fn string_field<'a>(value: &'a JsonValue, name: &str, label: &str) -> Result<&'a str, String> {
    field(value, name, label)?
        .as_str()
        .ok_or_else(|| format!("{label}.{name} must be a string"))
}

fn string_value<'a>(value: &'a JsonValue, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{label} must be a string"))
}

fn usize_field(value: &JsonValue, name: &str, label: &str) -> Result<usize, String> {
    usize_value(field(value, name, label)?, &format!("{label}.{name}"))
}

fn u64_field(value: &JsonValue, name: &str, label: &str) -> Result<u64, String> {
    field(value, name, label)?
        .as_u64()
        .ok_or_else(|| format!("{label}.{name} must be a non-negative integer"))
}

fn usize_value(value: &JsonValue, label: &str) -> Result<usize, String> {
    let number = value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))?;
    usize::try_from(number).map_err(|_| format!("{label} overflows usize"))
}

fn bool_field(value: &JsonValue, name: &str, label: &str) -> Result<bool, String> {
    match field(value, name, label)? {
        JsonValue::Bool(value) => Ok(*value),
        _ => Err(format!("{label}.{name} must be a boolean")),
    }
}

fn array_field<'a>(
    value: &'a JsonValue,
    name: &str,
    label: &str,
) -> Result<&'a [JsonValue], String> {
    field(value, name, label)?
        .as_array()
        .ok_or_else(|| format!("{label}.{name} must be an array"))
}

pub(super) fn parse_sha256(value: &str, label: &str) -> Result<[u8; 32], String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        ));
    }
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("{label} contains invalid hexadecimal"))?;
    }
    Ok(digest)
}

pub(super) fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 0x0f) as usize] as char);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_bytes(values: &[f32]) -> Vec<u8> {
        assert_eq!(values.len(), NOISE_VALUES);
        let mut header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 64), }".to_vec();
        let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
        header.extend(std::iter::repeat_n(b' ', padding));
        header.push(b'\n');
        assert_eq!((10 + header.len()) % 16, 0);
        let mut bytes = Vec::with_capacity(10 + header.len() + values.len() * 4);
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn npy_header_range(bytes: &[u8]) -> std::ops::Range<usize> {
        let header_start = match (bytes[6], bytes[7]) {
            (1, 0) => 10,
            (2, 0) | (3, 0) => 12,
            _ => panic!("test fixture uses unsupported NPY version"),
        };
        let header_len = match (bytes[6], bytes[7]) {
            (1, 0) => usize::from(u16::from_le_bytes([bytes[8], bytes[9]])),
            (2, 0) | (3, 0) => {
                usize::try_from(u32::from_le_bytes(bytes[8..12].try_into().unwrap())).unwrap()
            }
            _ => unreachable!(),
        };
        header_start..header_start + header_len
    }

    fn npy_payload_offset(bytes: &[u8]) -> usize {
        npy_header_range(bytes).end
    }

    fn packet_many(records: &[(String, String, String)]) -> String {
        assert!(!records.is_empty());
        let hashes = records
            .iter()
            .map(|(_, _, payload_sha)| format!("\"{payload_sha}\""))
            .collect::<Vec<_>>()
            .join(",");
        let records_json = records
            .iter()
            .map(|(noise_file, file_sha, _)| {
                format!(
                    r#"{{"file":"{noise_file}","shape":[2,64],"dtype":"float32","sha256":"{file_sha}","finite":true}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"format":"{FORMAT}","status":"{OPEN_STATUS}","publication":"NO_UPLOAD","source":{{"repository":"{SOURCE_REPOSITORY}","revision":"{SOURCE_REVISION}","origin":"{SOURCE_ORIGIN}"}},"checkpoint":{{"bytes":{CHECKPOINT_BYTES},"sha256":"{CHECKPOINT_SHA256}"}},"preset":{{"payload_sha256":"{PRESET_PAYLOAD_SHA256}"}},"packets":{{"cpu":{{"device":"cpu","noise_matching":true,"noise_draws":{count},"noise_hashes":[{hashes}],"diffusion_initial_noise":[{records_json}]}}}}}}"#,
            count = records.len(),
        )
    }

    fn packet(noise_file: &str, file_sha: &str, payload_sha: &str) -> String {
        packet_many(&[(
            noise_file.to_owned(),
            file_sha.to_owned(),
            payload_sha.to_owned(),
        )])
    }

    fn fixture_many(packet: &str, files: &[(&str, &[u8])]) -> (PathBuf, String) {
        let temp_root = fs::canonicalize(std::env::temp_dir()).unwrap();
        let stem = format!(
            "vokra-realtime-noise-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("case")
        );
        let root = (0..32)
            .map(|attempt| temp_root.join(format!("{stem}-{attempt}")))
            .find(|candidate| match fs::create_dir(candidate) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => panic!("create fixture root {}: {error}", candidate.display()),
            })
            .expect("could not allocate a unique fixture root");
        fs::create_dir_all(root.join("cpu")).unwrap();
        for (filename, bytes) in files {
            fs::write(root.join("cpu").join(filename), bytes).unwrap();
        }
        fs::write(root.join("reference.json"), packet).unwrap();
        let expected = hex_digest(&sha256(packet.as_bytes()));
        (root, expected)
    }

    fn fixture(packet: &str, npy: &[u8], filename: &str) -> (PathBuf, String) {
        fixture_many(packet, &[(filename, npy)])
    }

    #[test]
    fn loads_official_shape_and_authenticates_both_noise_rows() {
        let values: Vec<f32> = (0..NOISE_VALUES).map(|value| value as f32).collect();
        let npy = npy_bytes(&values);
        let payload_offset = npy_payload_offset(&npy);
        let packet = packet(
            "diffusion_initial_noise_0.npy",
            &hex_digest(&sha256(&npy)),
            &hex_digest(&sha256(&npy[payload_offset..])),
        );
        let (root, expected) = fixture(&packet, &npy, "diffusion_initial_noise_0.npy");
        let tape = load_vibevoice_realtime_noise_tape(&root, &expected).unwrap();
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.draws[0], values[..NOISE_WIDTH]);
        assert!(!tape.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accepts_sixty_four_source_named_draws_in_packet_order() {
        let values: Vec<f32> = (0..NOISE_VALUES).map(|value| value as f32).collect();
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));
        let records: Vec<(String, String, String)> = (0..64)
            .map(|index| {
                (
                    format!("diffusion_initial_noise_{index}.npy"),
                    file_sha.clone(),
                    payload_sha.clone(),
                )
            })
            .collect();
        let packet = packet_many(&records);
        let files: Vec<(&str, &[u8])> = records
            .iter()
            .map(|(filename, _, _)| (filename.as_str(), npy.as_slice()))
            .collect();
        let (root, expected) = fixture_many(&packet, &files);
        let tape = load_vibevoice_realtime_noise_tape(&root, &expected).unwrap();
        assert_eq!(tape.len(), 64);
        assert!(tape.draws.iter().all(|draw| draw == &values[..NOISE_WIDTH]));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preserves_two_contiguous_source_record_ordinals() {
        let values_a = vec![0.25; NOISE_VALUES];
        let values_b = vec![0.5; NOISE_VALUES];
        let npy_a = npy_bytes(&values_a);
        let npy_b = npy_bytes(&values_b);
        let records = vec![
            (
                "diffusion_initial_noise_0.npy".to_owned(),
                hex_digest(&sha256(&npy_a)),
                hex_digest(&sha256(&npy_a[npy_payload_offset(&npy_a)..])),
            ),
            (
                "diffusion_initial_noise_1.npy".to_owned(),
                hex_digest(&sha256(&npy_b)),
                hex_digest(&sha256(&npy_b[npy_payload_offset(&npy_b)..])),
            ),
        ];
        let packet = packet_many(&records);
        let files = vec![
            ("diffusion_initial_noise_0.npy", npy_a.as_slice()),
            ("diffusion_initial_noise_1.npy", npy_b.as_slice()),
        ];
        let (root, expected) = fixture_many(&packet, &files);
        let tape = load_vibevoice_realtime_noise_tape(&root, &expected).unwrap();
        assert_eq!(
            tape.draws,
            vec![
                values_a[..NOISE_WIDTH].to_vec(),
                values_b[..NOISE_WIDTH].to_vec()
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_row_one_tamper_with_rebound_file_but_stale_payload_hash() {
        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));
        let mut tampered = npy.clone();
        let payload_offset = npy_payload_offset(&tampered);
        tampered[payload_offset + NOISE_WIDTH * 4..payload_offset + NOISE_WIDTH * 4 + 4]
            .copy_from_slice(&0.5f32.to_le_bytes());
        let packet = packet(
            "diffusion_initial_noise_0.npy",
            &hex_digest(&sha256(&tampered)),
            &payload_sha,
        );
        let (root, expected) = fixture(&packet, &tampered, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("payload 0 SHA-256 mismatch")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_duplicate_keys_bad_shape_and_file_trailing_bytes() {
        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));

        let duplicate = packet("diffusion_initial_noise_0.npy", &file_sha, &payload_sha).replacen(
            "\"device\":\"cpu\"",
            "\"device\":\"cpu\",\"device\":\"cpu\"",
            1,
        );
        let (root, expected) = fixture(&duplicate, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("duplicate JSON key")
        );
        fs::remove_dir_all(&root).unwrap();

        let bad_shape = packet("diffusion_initial_noise_0.npy", &file_sha, &payload_sha)
            .replace("\"shape\":[2,64]", "\"shape\":[128]");
        let (root, expected) = fixture(&bad_shape, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("exactly two dimensions")
        );
        fs::remove_dir_all(&root).unwrap();

        let mut trailing = npy.clone();
        trailing.push(0);
        let packet = packet(
            "diffusion_initial_noise_0.npy",
            &hex_digest(&sha256(&trailing)),
            &payload_sha,
        );
        let (root, expected) = fixture(&packet, &trailing, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("trailing or truncated")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_unsafe_filename_and_uppercase_hash() {
        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));

        let unsafe_packet = packet("../noise.npy", &file_sha, &payload_sha);
        let (root, expected) = fixture(&unsafe_packet, &npy, "noise.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("safe basename")
        );
        fs::remove_dir_all(&root).unwrap();

        let uppercase_packet = packet(
            "diffusion_initial_noise_0.npy",
            &file_sha.to_uppercase(),
            &payload_sha,
        );
        let (root, expected) = fixture(&uppercase_packet, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("lowercase hexadecimal")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_reference_hash_count_matching_and_npy_encoding_failures() {
        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));
        let valid_packet = packet("diffusion_initial_noise_0.npy", &file_sha, &payload_sha);
        let (root, _expected) = fixture(&valid_packet, &npy, "diffusion_initial_noise_0.npy");
        assert!(load_vibevoice_realtime_noise_tape(&root, &"0".repeat(64)).is_err());
        fs::remove_dir_all(&root).unwrap();

        let count_packet = valid_packet.replace("\"noise_draws\":1", "\"noise_draws\":2");
        let (root, expected) = fixture(&count_packet, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("does not match diffusion_initial_noise")
        );
        fs::remove_dir_all(&root).unwrap();

        let zero_packet = valid_packet.replace("\"noise_draws\":1", "\"noise_draws\":0");
        let (root, expected) = fixture(&zero_packet, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("must be in 1..=64")
        );
        fs::remove_dir_all(&root).unwrap();

        let over_packet = valid_packet.replace("\"noise_draws\":1", "\"noise_draws\":65");
        let (root, expected) = fixture(&over_packet, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("must be in 1..=64")
        );
        fs::remove_dir_all(&root).unwrap();

        let no_match_packet =
            valid_packet.replace("\"noise_matching\":true", "\"noise_matching\":false");
        let (root, expected) = fixture(&no_match_packet, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("noise_matching must be true")
        );
        fs::remove_dir_all(&root).unwrap();

        let mut fortran = npy.clone();
        let header_range = npy_header_range(&fortran);
        let header = std::str::from_utf8(&fortran[header_range.clone()])
            .unwrap()
            .replace("False", "True ");
        fortran[header_range].copy_from_slice(header.as_bytes());
        let fortran_packet = packet(
            "diffusion_initial_noise_0.npy",
            &hex_digest(&sha256(&fortran)),
            &payload_sha,
        );
        let (root, expected) = fixture(&fortran_packet, &fortran, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("fortran_order must be False")
        );
        fs::remove_dir_all(&root).unwrap();

        let mut big_endian = npy.clone();
        let header_range = npy_header_range(&big_endian);
        let header = std::str::from_utf8(&big_endian[header_range.clone()])
            .unwrap()
            .replace("<f4", ">f4");
        big_endian[header_range].copy_from_slice(header.as_bytes());
        let big_endian_packet = packet(
            "diffusion_initial_noise_0.npy",
            &hex_digest(&sha256(&big_endian)),
            &payload_sha,
        );
        let (root, expected) = fixture(
            &big_endian_packet,
            &big_endian,
            "diffusion_initial_noise_0.npy",
        );
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("little-endian")
        );
        fs::remove_dir_all(&root).unwrap();

        let mut nonfinite = npy.clone();
        let nonfinite_offset = npy_payload_offset(&nonfinite);
        nonfinite[nonfinite_offset..nonfinite_offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let nonfinite_payload_sha = hex_digest(&sha256(&nonfinite[nonfinite_offset..]));
        let nonfinite_packet = packet(
            "diffusion_initial_noise_0.npy",
            &hex_digest(&sha256(&nonfinite)),
            &nonfinite_payload_sha,
        );
        let (root, expected) = fixture(
            &nonfinite_packet,
            &nonfinite,
            "diffusion_initial_noise_0.npy",
        );
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("non-finite")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_noise_file_before_reading_target() {
        use std::os::unix::fs::symlink;

        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));
        let packet = packet("diffusion_initial_noise_0.npy", &file_sha, &payload_sha);
        let (root, expected) = fixture(&packet, &npy, "diffusion_initial_noise_0.npy");
        fs::write(root.join("cpu").join("real.npy"), &npy).unwrap();
        fs::remove_file(root.join("cpu").join("diffusion_initial_noise_0.npy")).unwrap();
        symlink(
            "real.npy",
            root.join("cpu").join("diffusion_initial_noise_0.npy"),
        )
        .unwrap();
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("has a symlink ancestor")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_contradictory_or_noncanonical_npy_header_fields() {
        let duplicate_descr =
            "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (2, 64), }";
        assert!(validate_npy_header(duplicate_descr, "duplicate").is_err());
        let duplicate_shape =
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 64), 'shape': (2, 64), }";
        assert!(validate_npy_header(duplicate_shape, "duplicate").is_err());
        let altered_case = "{'descr': '<f4', 'fortran_order': false, 'shape': (2, 64), }";
        assert!(validate_npy_header(altered_case, "case").is_err());
        let extra_shape = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 64, 1), }";
        assert!(validate_npy_header(extra_shape, "shape").is_err());
        let unknown_key = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 64), 'x': 1, }";
        assert!(validate_npy_header(unknown_key, "unknown").is_err());
    }

    #[test]
    fn rejects_wrong_identity_missing_hashes_unknown_version_and_truncated_header() {
        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));
        let valid_packet = packet("diffusion_initial_noise_0.npy", &file_sha, &payload_sha);

        let wrong_identity = valid_packet.replace(FORMAT, "wrong-format");
        let (root, expected) = fixture(&wrong_identity, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("reference.format")
        );
        fs::remove_dir_all(&root).unwrap();

        let missing_hashes = valid_packet.replace(
            &format!("\"noise_hashes\":[\"{payload_sha}\"]"),
            "\"noise_hashes\":[]",
        );
        let (root, expected) = fixture(&missing_hashes, &npy, "diffusion_initial_noise_0.npy");
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("noise_hashes")
        );
        fs::remove_dir_all(&root).unwrap();

        let mut unknown_version = npy.clone();
        unknown_version[6] = 9;
        assert!(
            parse_npy_f32(&unknown_version, 0)
                .unwrap_err()
                .contains("unsupported NPY version")
        );
        assert!(
            parse_npy_f32(&npy[..20], 0)
                .unwrap_err()
                .contains("truncated NPY header")
        );
    }

    #[test]
    fn rejects_oversized_inputs_before_allocating_payload_buffers() {
        let values = vec![0.25; NOISE_VALUES];
        let npy = npy_bytes(&values);
        let file_sha = hex_digest(&sha256(&npy));
        let payload_sha = hex_digest(&sha256(&npy[npy_payload_offset(&npy)..]));
        let packet = packet("diffusion_initial_noise_0.npy", &file_sha, &payload_sha);

        let (root, expected) = fixture(&packet, &npy, "diffusion_initial_noise_0.npy");
        fs::OpenOptions::new()
            .write(true)
            .open(root.join("reference.json"))
            .unwrap()
            .set_len(MAX_REFERENCE_JSON_BYTES + 1)
            .unwrap();
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("resource limit")
        );
        fs::remove_dir_all(&root).unwrap();

        let (root, expected) = fixture(&packet, &npy, "diffusion_initial_noise_0.npy");
        fs::OpenOptions::new()
            .write(true)
            .open(root.join("cpu").join("diffusion_initial_noise_0.npy"))
            .unwrap()
            .set_len(MAX_NOISE_FILE_BYTES + 1)
            .unwrap();
        assert!(
            load_vibevoice_realtime_noise_tape(&root, &expected)
                .unwrap_err()
                .contains("resource limit")
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn validates_expected_reference_hash_before_filesystem_access() {
        let error = load_vibevoice_realtime_noise_tape(
            Path::new("/vokra/nonexistent/reference"),
            "not-a-sha256",
        )
        .unwrap_err();
        assert!(error.contains("64 lowercase hexadecimal"));
    }
}
