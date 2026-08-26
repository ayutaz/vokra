//! Stable, dependency-free file contracts used by `vokra-cli run`.
//!
//! These formats are deliberately versioned. CT-Punc needs caller-supplied
//! token strings paired with the exact token ids passed to the model, while
//! Mimi needs a portable representation of its time-major RVQ codes. Neither
//! contract can be represented honestly by printing a Rust debug array.

/// Version marker required as the first line of a CT-Punc token file.
pub(crate) const CT_PUNC_TSV_V1: &str = "vokra-ct-punc-tsv-v1";

/// Paired token strings and token ids parsed from CT-Punc TSV v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CtPuncInput {
    pub(crate) tokens: Vec<String>,
    pub(crate) token_ids: Vec<u32>,
}

/// Parses the CT-Punc v1 side-car.
///
/// The first line is exactly [`CT_PUNC_TSV_V1`]. Every following line is
/// `<u32 token id>\t<escaped UTF-8 token>`. The token field accepts literal
/// Unicode scalar values and the escapes `\\`, `\t`, `\n`, `\r`, and
/// `\u{HEX}`. Empty tokens, raw control characters, malformed escapes, extra
/// tab-separated columns, and an empty record set fail loudly.
pub(crate) fn parse_ct_punc_tsv(text: &str) -> Result<CtPuncInput, String> {
    let mut lines = text.lines();
    let raw_header = lines
        .next()
        .ok_or_else(|| format!("CT-Punc token file is empty; expected `{CT_PUNC_TSV_V1}`"))?;
    let header = raw_header.strip_suffix('\r').unwrap_or(raw_header);
    if header != CT_PUNC_TSV_V1 {
        return Err(format!(
            "CT-Punc token file has header `{header}`, expected `{CT_PUNC_TSV_V1}`"
        ));
    }

    let mut tokens = Vec::new();
    let mut token_ids = Vec::new();
    for (index, raw_line) in lines.enumerate() {
        let line_no = index + 2;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            return Err(format!(
                "CT-Punc token file line {line_no} is empty; every record must pair an id and token"
            ));
        }
        let (id, escaped) = line.split_once('\t').ok_or_else(|| {
            format!(
                "CT-Punc token file line {line_no} has no tab; expected `<u32 id>\\t<escaped token>`"
            )
        })?;
        if escaped.contains('\t') {
            return Err(format!(
                "CT-Punc token file line {line_no} has more than two TSV fields; encode a token tab as `\\t`"
            ));
        }
        let id = id.parse::<u32>().map_err(|e| {
            format!("CT-Punc token file line {line_no}: `{id}` is not a u32 token id: {e}")
        })?;
        let token = unescape_ct_punc_token(escaped, line_no)?;
        if token.is_empty() {
            return Err(format!(
                "CT-Punc token file line {line_no} has an empty token; token/id alignment must be explicit"
            ));
        }
        token_ids.push(id);
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err("CT-Punc token file contains no token records".to_owned());
    }
    Ok(CtPuncInput { tokens, token_ids })
}

fn unescape_ct_punc_token(input: &str, line_no: usize) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            if ch.is_control() {
                return Err(format!(
                    "CT-Punc token file line {line_no} contains raw control U+{:04X}; use an escape",
                    ch as u32
                ));
            }
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                if chars.next() != Some('{') {
                    return Err(format!(
                        "CT-Punc token file line {line_no} has malformed Unicode escape; expected `\\u{{HEX}}`"
                    ));
                }
                let mut hex = String::new();
                let mut closed = false;
                for c in chars.by_ref() {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    if !c.is_ascii_hexdigit() || hex.len() == 6 {
                        return Err(format!(
                            "CT-Punc token file line {line_no} has invalid Unicode escape `\\u{{{hex}{c}`"
                        ));
                    }
                    hex.push(c);
                }
                if !closed || hex.is_empty() {
                    return Err(format!(
                        "CT-Punc token file line {line_no} has unterminated or empty Unicode escape"
                    ));
                }
                let scalar = u32::from_str_radix(&hex, 16).map_err(|e| {
                    format!(
                        "CT-Punc token file line {line_no}: invalid Unicode scalar `{hex}`: {e}"
                    )
                })?;
                let scalar = char::from_u32(scalar).ok_or_else(|| {
                    format!(
                        "CT-Punc token file line {line_no}: U+{scalar:04X} is not a Unicode scalar"
                    )
                })?;
                out.push(scalar);
            }
            Some(other) => {
                return Err(format!(
                    "CT-Punc token file line {line_no} has unknown escape `\\{other}`; supported: `\\\\`, `\\t`, `\\n`, `\\r`, `\\u{{HEX}}`"
                ));
            }
            None => {
                return Err(format!(
                    "CT-Punc token file line {line_no} ends with an incomplete escape"
                ));
            }
        }
    }
    Ok(out)
}

/// MeloTTS precomputed-feature container magic (`VKRMELO1`) and version.
const MELOTTS_MAGIC: &[u8; 8] = b"VKRMELO1";
const MELOTTS_VERSION: u16 = 1;
const MELOTTS_HEADER_LEN: usize = 64;
const MELOTTS_BERT_DIMENSION: usize = 1_024;
const MELOTTS_JA_BERT_DIMENSION: usize = 768;

/// Language identity pinned by a MeloTTS feature container.
///
/// The numeric values are part of the `VKRMELO1` v1 file contract. A feature
/// bundle is deliberately tied to one release language: feeding Chinese phone
/// and tone ids to the English embedding tables can stay numerically in range
/// while producing plausible-looking but invalid speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeloFeatureVariantV1 {
    English = 0,
    Chinese = 1,
    Korean = 2,
    Spanish = 3,
    Japanese = 4,
}

impl MeloFeatureVariantV1 {
    fn from_u32(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::English),
            1 => Ok(Self::Chinese),
            2 => Ok(Self::Korean),
            3 => Ok(Self::Spanish),
            4 => Ok(Self::Japanese),
            other => Err(format!(
                "MeloTTS feature container has unknown variant id {other}; v1 supports 0=english, 1=chinese, 2=korean, 3=spanish, 4=japanese"
            )),
        }
    }

    pub(crate) const fn tag(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Chinese => "chinese",
            Self::Korean => "korean",
            Self::Spanish => "spanish",
            Self::Japanese => "japanese",
        }
    }
}

/// Portable MeloTTS acoustic-input feature container, version 1.
///
/// The payload is little-endian and position-major:
///
/// 1. `sequence_len` phoneme `u32` ids;
/// 2. `sequence_len` tone `u32` ids;
/// 3. `sequence_len` language `u32` ids;
/// 4. `[sequence_len, 1024]` BERT `f32` values;
/// 5. `[sequence_len, 768]` JA-BERT `f32` values.
///
/// Raw text is intentionally not accepted here. The five official acoustic
/// GGUFs do not contain the language-specific G2P/tokenizer/BERT frontends,
/// so pretending to derive these arrays in the runtime would silently change
/// the model input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeloTextFeaturesV1 {
    pub(crate) variant: MeloFeatureVariantV1,
    pub(crate) speaker_id: u32,
    pub(crate) phoneme_ids: Vec<u32>,
    pub(crate) tones: Vec<u32>,
    pub(crate) language_ids: Vec<u32>,
    pub(crate) bert: Vec<f32>,
    pub(crate) ja_bert: Vec<f32>,
}

impl MeloTextFeaturesV1 {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < MELOTTS_HEADER_LEN {
            return Err(format!(
                "MeloTTS feature container is truncated: {} bytes, need at least {MELOTTS_HEADER_LEN}",
                bytes.len()
            ));
        }
        if &bytes[..8] != MELOTTS_MAGIC {
            return Err(
                "MeloTTS feature container has wrong magic; expected `VKRMELO1`".to_owned(),
            );
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != MELOTTS_VERSION {
            return Err(format!(
                "MeloTTS feature container version {version} is unsupported; this build reads version {MELOTTS_VERSION}"
            ));
        }
        let header_len = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
        if header_len != MELOTTS_HEADER_LEN {
            return Err(format!(
                "MeloTTS feature container v1 header length {header_len} != {MELOTTS_HEADER_LEN}"
            ));
        }
        if read_u32(bytes, 12)? != 0 || bytes[36..64].iter().any(|&byte| byte != 0) {
            return Err("MeloTTS feature container v1 has non-zero reserved fields".to_owned());
        }
        let variant = MeloFeatureVariantV1::from_u32(read_u32(bytes, 16)?)?;
        let sequence_len = usize::try_from(read_u32(bytes, 20)?)
            .map_err(|_| "MeloTTS sequence length does not fit this host".to_owned())?;
        if sequence_len == 0 {
            return Err("MeloTTS feature container has an empty sequence".to_owned());
        }
        let speaker_id = read_u32(bytes, 24)?;
        let bert_dimension = usize::try_from(read_u32(bytes, 28)?)
            .map_err(|_| "MeloTTS BERT dimension does not fit this host".to_owned())?;
        let ja_bert_dimension = usize::try_from(read_u32(bytes, 32)?)
            .map_err(|_| "MeloTTS JA-BERT dimension does not fit this host".to_owned())?;
        if bert_dimension != MELOTTS_BERT_DIMENSION
            || ja_bert_dimension != MELOTTS_JA_BERT_DIMENSION
        {
            return Err(format!(
                "MeloTTS feature container dimensions are BERT={bert_dimension}, JA-BERT={ja_bert_dimension}; v1 requires {MELOTTS_BERT_DIMENSION} and {MELOTTS_JA_BERT_DIMENSION}"
            ));
        }

        let values_per_position = 3usize
            .checked_add(bert_dimension)
            .and_then(|value| value.checked_add(ja_bert_dimension))
            .ok_or("MeloTTS feature width overflow")?;
        let payload_values = sequence_len
            .checked_mul(values_per_position)
            .ok_or("MeloTTS feature payload size overflow")?;
        let expected = MELOTTS_HEADER_LEN
            .checked_add(
                payload_values
                    .checked_mul(4)
                    .ok_or("MeloTTS feature payload size overflow")?,
            )
            .ok_or("MeloTTS feature container size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "MeloTTS feature container length {} != header {MELOTTS_HEADER_LEN} + {payload_values} four-byte values ({expected})",
                bytes.len()
            ));
        }

        let mut offset = MELOTTS_HEADER_LEN;
        let phoneme_ids = take_u32_values(bytes, &mut offset, sequence_len);
        let tones = take_u32_values(bytes, &mut offset, sequence_len);
        let language_ids = take_u32_values(bytes, &mut offset, sequence_len);
        let bert = take_f32_values(bytes, &mut offset, sequence_len * MELOTTS_BERT_DIMENSION);
        let ja_bert = take_f32_values(bytes, &mut offset, sequence_len * MELOTTS_JA_BERT_DIMENSION);
        debug_assert_eq!(offset, bytes.len());
        if let Some((label, index, value)) =
            [("bert", bert.as_slice()), ("ja_bert", ja_bert.as_slice())]
                .into_iter()
                .find_map(|(label, values)| {
                    values
                        .iter()
                        .copied()
                        .enumerate()
                        .find(|(_, value)| !value.is_finite())
                        .map(|(index, value)| (label, index, value))
                })
        {
            return Err(format!(
                "MeloTTS feature container {label}[{index}] is non-finite ({value})"
            ));
        }
        Ok(Self {
            variant,
            speaker_id,
            phoneme_ids,
            tones,
            language_ids,
            bert,
            ja_bert,
        })
    }

    #[cfg(test)]
    fn to_bytes(&self) -> Vec<u8> {
        let sequence_len = u32::try_from(self.phoneme_ids.len()).expect("test sequence fits u32");
        assert_eq!(self.tones.len(), sequence_len as usize);
        assert_eq!(self.language_ids.len(), sequence_len as usize);
        assert_eq!(
            self.bert.len(),
            sequence_len as usize * MELOTTS_BERT_DIMENSION
        );
        assert_eq!(
            self.ja_bert.len(),
            sequence_len as usize * MELOTTS_JA_BERT_DIMENSION
        );
        let payload_values =
            sequence_len as usize * (3 + MELOTTS_BERT_DIMENSION + MELOTTS_JA_BERT_DIMENSION);
        let mut out = Vec::with_capacity(MELOTTS_HEADER_LEN + payload_values * 4);
        out.extend_from_slice(MELOTTS_MAGIC);
        out.extend_from_slice(&MELOTTS_VERSION.to_le_bytes());
        out.extend_from_slice(&(MELOTTS_HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(self.variant as u32).to_le_bytes());
        out.extend_from_slice(&sequence_len.to_le_bytes());
        out.extend_from_slice(&self.speaker_id.to_le_bytes());
        out.extend_from_slice(&(MELOTTS_BERT_DIMENSION as u32).to_le_bytes());
        out.extend_from_slice(&(MELOTTS_JA_BERT_DIMENSION as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 28]);
        for values in [&self.phoneme_ids, &self.tones, &self.language_ids] {
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        for values in [&self.bert, &self.ja_bert] {
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }
}

fn take_u32_values(bytes: &[u8], offset: &mut usize, count: usize) -> Vec<u32> {
    let end = *offset + count * 4;
    let values = bytes[*offset..end]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    *offset = end;
    values
}

fn take_f32_values(bytes: &[u8], offset: &mut usize, count: usize) -> Vec<f32> {
    let end = *offset + count * 4;
    let values = bytes[*offset..end]
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    *offset = end;
    values
}

/// Mimi code-container magic (`VKRMCODE`) and current format version.
const MIMI_MAGIC: &[u8; 8] = b"VKRMCODE";
const MIMI_VERSION: u16 = 1;
const MIMI_HEADER_LEN: usize = 96;
const MIMI_CHANNELS: u16 = 1;
const MIMI_CODE_WIDTH_BITS: u16 = 32;

/// Portable Mimi RVQ code container, version 1.
///
/// Codes are `[n_frames, n_codebooks]`, time-major, unsigned 32-bit little
/// endian. The SHA-256 is over the GGUF's effective
/// `vokra.mimi.codebook_tables` tensor bytes; decode refuses a different
/// table even when all topology axes happen to match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MimiCodesV1 {
    pub(crate) sample_rate: u32,
    pub(crate) frame_rate_mhz: u32,
    pub(crate) n_codebooks: u32,
    pub(crate) codebook_size: u32,
    pub(crate) feature_dimension: u32,
    pub(crate) n_frames: u64,
    pub(crate) pcm_samples: u64,
    pub(crate) model_sha256: [u8; 32],
    pub(crate) codes: Vec<u32>,
}

impl MimiCodesV1 {
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let payload_len = self
            .codes
            .len()
            .checked_mul(4)
            .ok_or("Mimi code payload size overflow")?;
        let mut out = Vec::with_capacity(MIMI_HEADER_LEN + payload_len);
        out.extend_from_slice(MIMI_MAGIC);
        out.extend_from_slice(&MIMI_VERSION.to_le_bytes());
        out.extend_from_slice(&(MIMI_HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.frame_rate_mhz.to_le_bytes());
        out.extend_from_slice(&MIMI_CHANNELS.to_le_bytes());
        out.extend_from_slice(&MIMI_CODE_WIDTH_BITS.to_le_bytes());
        out.extend_from_slice(&self.n_codebooks.to_le_bytes());
        out.extend_from_slice(&self.codebook_size.to_le_bytes());
        out.extend_from_slice(&self.feature_dimension.to_le_bytes());
        out.extend_from_slice(&self.n_frames.to_le_bytes());
        out.extend_from_slice(&self.pcm_samples.to_le_bytes());
        out.extend_from_slice(&self.model_sha256);
        out.extend_from_slice(&[0u8; 8]); // reserved for additive v1 fields
        debug_assert_eq!(out.len(), MIMI_HEADER_LEN);
        for code in &self.codes {
            out.extend_from_slice(&code.to_le_bytes());
        }
        Ok(out)
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < MIMI_HEADER_LEN {
            return Err(format!(
                "Mimi code container is truncated: {} bytes, need at least {MIMI_HEADER_LEN}",
                bytes.len()
            ));
        }
        if &bytes[..8] != MIMI_MAGIC {
            return Err("Mimi code container has wrong magic; expected `VKRMCODE`".to_owned());
        }
        let version = read_u16(bytes, 8)?;
        if version != MIMI_VERSION {
            return Err(format!(
                "Mimi code container version {version} is unsupported; this build reads version {MIMI_VERSION}"
            ));
        }
        let header_len = read_u16(bytes, 10)? as usize;
        if header_len != MIMI_HEADER_LEN {
            return Err(format!(
                "Mimi code container v1 header length {header_len} != {MIMI_HEADER_LEN}"
            ));
        }
        if read_u32(bytes, 12)? != 0 || bytes[88..96].iter().any(|&b| b != 0) {
            return Err("Mimi code container v1 has non-zero reserved fields".to_owned());
        }
        let channels = read_u16(bytes, 24)?;
        let code_width = read_u16(bytes, 26)?;
        if channels != MIMI_CHANNELS {
            return Err(format!(
                "Mimi code container has {channels} channels; v1 is mono-only"
            ));
        }
        if code_width != MIMI_CODE_WIDTH_BITS {
            return Err(format!(
                "Mimi code container uses {code_width}-bit codes; v1 requires unsigned 32-bit little-endian codes"
            ));
        }
        let n_codebooks = read_u32(bytes, 28)?;
        let n_frames = read_u64(bytes, 40)?;
        let count_u64 = n_frames
            .checked_mul(u64::from(n_codebooks))
            .ok_or("Mimi code count overflow")?;
        let count = usize::try_from(count_u64)
            .map_err(|_| "Mimi code count does not fit this host".to_owned())?;
        let expected = MIMI_HEADER_LEN
            .checked_add(count.checked_mul(4).ok_or("Mimi payload size overflow")?)
            .ok_or("Mimi container size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "Mimi code container length {} != header {MIMI_HEADER_LEN} + {count} u32 codes ({expected})",
                bytes.len()
            ));
        }
        let mut model_sha256 = [0u8; 32];
        model_sha256.copy_from_slice(&bytes[56..88]);
        let codes = bytes[MIMI_HEADER_LEN..]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let parsed = Self {
            sample_rate: read_u32(bytes, 16)?,
            frame_rate_mhz: read_u32(bytes, 20)?,
            n_codebooks,
            codebook_size: read_u32(bytes, 32)?,
            feature_dimension: read_u32(bytes, 36)?,
            n_frames,
            pcm_samples: read_u64(bytes, 48)?,
            model_sha256,
            codes,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0
            || self.frame_rate_mhz == 0
            || self.n_codebooks == 0
            || self.codebook_size == 0
            || self.feature_dimension == 0
            || self.n_frames == 0
            || self.pcm_samples == 0
        {
            return Err(
                "Mimi code container v1 has a zero rate, axis, frame, or PCM length".to_owned(),
            );
        }
        let n_codebooks = usize::try_from(self.n_codebooks)
            .map_err(|_| "Mimi codebook count does not fit this host")?;
        let expected = usize::try_from(self.n_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(n_codebooks))
            .ok_or("Mimi code count does not fit this host")?;
        if self.codes.len() != expected {
            return Err(format!(
                "Mimi code vector has {} entries, expected n_frames*n_codebooks = {expected}",
                self.codes.len()
            ));
        }
        if let Some((index, code)) = self
            .codes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| *code >= self.codebook_size)
        {
            return Err(format!(
                "Mimi code {code} at payload index {index} is outside codebook_size {}",
                self.codebook_size
            ));
        }
        Ok(())
    }
}

/// FocalCodec single-codebook container magic (`VKRFOC01`) and version.
const FOCALCODEC_MAGIC: &[u8; 8] = b"VKRFOC01";
const FOCALCODEC_VERSION: u16 = 1;
const FOCALCODEC_HEADER_LEN: usize = 96;

/// Portable FocalCodec BSQ token container, version 1.
///
/// Tokens are one unsigned little-endian u32 per time step. The topology
/// fields distinguish the released 50 / 25 / 12.5 Hz variants, while the
/// tensor fingerprint prevents a sibling or modified checkpoint from decoding
/// a stream whose integer range happens to be compatible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FocalCodecCodesV1 {
    pub(crate) sample_rate: u32,
    pub(crate) token_hz_times_two: u32,
    pub(crate) frame_hop: u32,
    pub(crate) codebook_size: u32,
    pub(crate) code_dimension: u32,
    pub(crate) token_count: u64,
    pub(crate) pcm_samples: u64,
    pub(crate) model_sha256: [u8; 32],
    pub(crate) tokens: Vec<u32>,
}

impl FocalCodecCodesV1 {
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let payload_len = self
            .tokens
            .len()
            .checked_mul(4)
            .ok_or("FocalCodec token payload size overflow")?;
        let mut out = Vec::with_capacity(FOCALCODEC_HEADER_LEN + payload_len);
        out.extend_from_slice(FOCALCODEC_MAGIC);
        out.extend_from_slice(&FOCALCODEC_VERSION.to_le_bytes());
        out.extend_from_slice(&(FOCALCODEC_HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.token_hz_times_two.to_le_bytes());
        out.extend_from_slice(&self.frame_hop.to_le_bytes());
        out.extend_from_slice(&self.codebook_size.to_le_bytes());
        out.extend_from_slice(&self.code_dimension.to_le_bytes());
        out.extend_from_slice(&self.token_count.to_le_bytes());
        out.extend_from_slice(&self.pcm_samples.to_le_bytes());
        out.extend_from_slice(&self.model_sha256);
        out.extend_from_slice(&[0u8; 12]); // reserved for additive v1 fields
        debug_assert_eq!(out.len(), FOCALCODEC_HEADER_LEN);
        for token in &self.tokens {
            out.extend_from_slice(&token.to_le_bytes());
        }
        Ok(out)
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < FOCALCODEC_HEADER_LEN {
            return Err(format!(
                "FocalCodec code container is truncated: {} bytes, need at least {FOCALCODEC_HEADER_LEN}",
                bytes.len()
            ));
        }
        if &bytes[..8] != FOCALCODEC_MAGIC {
            return Err(
                "FocalCodec code container has wrong magic; expected `VKRFOC01`".to_owned(),
            );
        }
        let version = read_u16(bytes, 8)?;
        if version != FOCALCODEC_VERSION {
            return Err(format!(
                "FocalCodec code container version {version} is unsupported; this build reads version {FOCALCODEC_VERSION}"
            ));
        }
        let header_len = read_u16(bytes, 10)? as usize;
        if header_len != FOCALCODEC_HEADER_LEN {
            return Err(format!(
                "FocalCodec code container v1 header length {header_len} != {FOCALCODEC_HEADER_LEN}"
            ));
        }
        if read_u32(bytes, 12)? != 0 || bytes[84..96].iter().any(|&byte| byte != 0) {
            return Err("FocalCodec code container v1 has non-zero reserved fields".to_owned());
        }
        let token_count = read_u64(bytes, 36)?;
        let token_count_host = usize::try_from(token_count)
            .map_err(|_| "FocalCodec token count does not fit this host".to_owned())?;
        let expected = FOCALCODEC_HEADER_LEN
            .checked_add(
                token_count_host
                    .checked_mul(4)
                    .ok_or("FocalCodec payload size overflow")?,
            )
            .ok_or("FocalCodec container size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "FocalCodec code container length {} != header {FOCALCODEC_HEADER_LEN} + {token_count_host} u32 tokens ({expected})",
                bytes.len()
            ));
        }
        let mut model_sha256 = [0u8; 32];
        model_sha256.copy_from_slice(&bytes[52..84]);
        let tokens = bytes[FOCALCODEC_HEADER_LEN..]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        let parsed = Self {
            sample_rate: read_u32(bytes, 16)?,
            token_hz_times_two: read_u32(bytes, 20)?,
            frame_hop: read_u32(bytes, 24)?,
            codebook_size: read_u32(bytes, 28)?,
            code_dimension: read_u32(bytes, 32)?,
            token_count,
            pcm_samples: read_u64(bytes, 44)?,
            model_sha256,
            tokens,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0
            || self.token_hz_times_two == 0
            || self.frame_hop == 0
            || self.codebook_size == 0
            || self.code_dimension == 0
            || self.token_count == 0
            || self.pcm_samples == 0
        {
            return Err(
                "FocalCodec code container v1 has a zero rate, axis, token, or PCM length"
                    .to_owned(),
            );
        }
        let expected = usize::try_from(self.token_count)
            .map_err(|_| "FocalCodec token count does not fit this host")?;
        if self.tokens.len() != expected {
            return Err(format!(
                "FocalCodec token vector has {} entries, expected token_count = {expected}",
                self.tokens.len()
            ));
        }
        if let Some((index, token)) = self
            .tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token >= self.codebook_size)
        {
            return Err(format!(
                "FocalCodec token {token} at payload index {index} is outside codebook_size {}",
                self.codebook_size
            ));
        }
        Ok(())
    }
}

/// NaturalSpeech 3 FACodec V2 container magic (`VKRFA2V1`) and version.
const FACODEC_MAGIC: &[u8; 8] = b"VKRFA2V1";
const FACODEC_VERSION: u16 = 1;
const FACODEC_HEADER_LEN: usize = 112;

/// Portable NaturalSpeech 3 FACodec V2 packet, version 1.
///
/// The payload first stores the decoder-required speaker embedding as
/// little-endian f32 values, followed by frame-major little-endian u32 codes.
/// The exact GGUF tensor fingerprint prevents decoding with a sibling or
/// modified checkpoint that merely has compatible axes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FacodecCodesV1 {
    pub(crate) sample_rate: u32,
    pub(crate) frame_hop: u32,
    pub(crate) n_codebooks: u32,
    pub(crate) codebook_size: u32,
    pub(crate) embedding_dim: u32,
    pub(crate) frames: u64,
    pub(crate) pcm_samples: u64,
    pub(crate) model_sha256: [u8; 32],
    pub(crate) speaker_embedding: Vec<f32>,
    pub(crate) codes: Vec<u32>,
}

impl FacodecCodesV1 {
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let embedding_values = u64::try_from(self.speaker_embedding.len())
            .map_err(|_| "FACodec speaker embedding length exceeds u64")?;
        let code_count =
            u64::try_from(self.codes.len()).map_err(|_| "FACodec code count exceeds u64")?;
        let payload_values = self
            .speaker_embedding
            .len()
            .checked_add(self.codes.len())
            .ok_or("FACodec payload value count overflow")?;
        let payload_len = payload_values
            .checked_mul(4)
            .ok_or("FACodec payload size overflow")?;
        let mut out = Vec::with_capacity(FACODEC_HEADER_LEN + payload_len);
        out.extend_from_slice(FACODEC_MAGIC);
        out.extend_from_slice(&FACODEC_VERSION.to_le_bytes());
        out.extend_from_slice(&(FACODEC_HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.frame_hop.to_le_bytes());
        out.extend_from_slice(&self.n_codebooks.to_le_bytes());
        out.extend_from_slice(&self.codebook_size.to_le_bytes());
        out.extend_from_slice(&self.embedding_dim.to_le_bytes());
        out.extend_from_slice(&self.frames.to_le_bytes());
        out.extend_from_slice(&self.pcm_samples.to_le_bytes());
        out.extend_from_slice(&self.model_sha256);
        out.extend_from_slice(&code_count.to_le_bytes());
        out.extend_from_slice(&embedding_values.to_le_bytes());
        out.extend_from_slice(&[0u8; 12]); // reserved for additive v1 fields
        debug_assert_eq!(out.len(), FACODEC_HEADER_LEN);
        for &value in &self.speaker_embedding {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &code in &self.codes {
            out.extend_from_slice(&code.to_le_bytes());
        }
        Ok(out)
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < FACODEC_HEADER_LEN {
            return Err(format!(
                "FACodec code container is truncated: {} bytes, need at least {FACODEC_HEADER_LEN}",
                bytes.len()
            ));
        }
        if &bytes[..8] != FACODEC_MAGIC {
            return Err("FACodec code container has wrong magic; expected `VKRFA2V1`".to_owned());
        }
        let version = read_u16(bytes, 8)?;
        if version != FACODEC_VERSION {
            return Err(format!(
                "FACodec code container version {version} is unsupported; this build reads version {FACODEC_VERSION}"
            ));
        }
        let header_len = read_u16(bytes, 10)? as usize;
        if header_len != FACODEC_HEADER_LEN {
            return Err(format!(
                "FACodec code container v1 header length {header_len} != {FACODEC_HEADER_LEN}"
            ));
        }
        if read_u32(bytes, 12)? != 0 || bytes[100..112].iter().any(|&byte| byte != 0) {
            return Err("FACodec code container v1 has non-zero reserved fields".to_owned());
        }

        let code_count = read_u64(bytes, 84)?;
        let embedding_values = read_u64(bytes, 92)?;
        let code_count_host = usize::try_from(code_count)
            .map_err(|_| "FACodec code count does not fit this host".to_owned())?;
        let embedding_values_host = usize::try_from(embedding_values)
            .map_err(|_| "FACodec embedding length does not fit this host".to_owned())?;
        let payload_values = embedding_values_host
            .checked_add(code_count_host)
            .ok_or("FACodec payload value count overflow")?;
        let expected = FACODEC_HEADER_LEN
            .checked_add(
                payload_values
                    .checked_mul(4)
                    .ok_or("FACodec payload size overflow")?,
            )
            .ok_or("FACodec container size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "FACodec code container length {} != header {FACODEC_HEADER_LEN} + {embedding_values_host} f32 embedding values + {code_count_host} u32 codes ({expected})",
                bytes.len()
            ));
        }

        let mut model_sha256 = [0u8; 32];
        model_sha256.copy_from_slice(&bytes[52..84]);
        let embedding_end = FACODEC_HEADER_LEN + embedding_values_host * 4;
        let speaker_embedding = bytes[FACODEC_HEADER_LEN..embedding_end]
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        let codes = bytes[embedding_end..]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        let parsed = Self {
            sample_rate: read_u32(bytes, 16)?,
            frame_hop: read_u32(bytes, 20)?,
            n_codebooks: read_u32(bytes, 24)?,
            codebook_size: read_u32(bytes, 28)?,
            embedding_dim: read_u32(bytes, 32)?,
            frames: read_u64(bytes, 36)?,
            pcm_samples: read_u64(bytes, 44)?,
            model_sha256,
            speaker_embedding,
            codes,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0
            || self.frame_hop == 0
            || self.n_codebooks == 0
            || self.codebook_size == 0
            || self.embedding_dim == 0
            || self.frames == 0
            || self.pcm_samples == 0
        {
            return Err(
                "FACodec code container v1 has a zero rate, axis, frame, or PCM length".to_owned(),
            );
        }
        let expected_embedding = usize::try_from(self.embedding_dim)
            .map_err(|_| "FACodec embedding dimension does not fit this host")?;
        if self.speaker_embedding.len() != expected_embedding {
            return Err(format!(
                "FACodec speaker embedding has {} values, expected embedding_dim = {expected_embedding}",
                self.speaker_embedding.len()
            ));
        }
        if let Some((index, value)) = self
            .speaker_embedding
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "FACodec speaker embedding value {index} is non-finite ({value})"
            ));
        }
        let expected_codes = self
            .frames
            .checked_mul(u64::from(self.n_codebooks))
            .ok_or("FACodec code count overflow")?;
        let expected_codes = usize::try_from(expected_codes)
            .map_err(|_| "FACodec code count does not fit this host")?;
        if self.codes.len() != expected_codes {
            return Err(format!(
                "FACodec code vector has {} entries, expected frames * n_codebooks = {expected_codes}",
                self.codes.len()
            ));
        }
        let decoded_extent = self
            .frames
            .checked_mul(u64::from(self.frame_hop))
            .ok_or("FACodec decoded PCM extent overflow")?;
        if self.pcm_samples < decoded_extent
            || self.pcm_samples - decoded_extent >= u64::from(self.frame_hop)
        {
            return Err(format!(
                "FACodec original PCM length {} is inconsistent with {} frames at hop {} (decoded extent {decoded_extent})",
                self.pcm_samples, self.frames, self.frame_hop
            ));
        }
        if let Some((index, code)) = self
            .codes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| *code >= self.codebook_size)
        {
            return Err(format!(
                "FACodec code {code} at payload index {index} is outside codebook_size {}",
                self.codebook_size
            ));
        }
        Ok(())
    }
}

/// SNAC hierarchical-code container magic (`VKRSNAC1`) and current version.
const SNAC_MAGIC: &[u8; 8] = b"VKRSNAC1";
const SNAC_VERSION: u16 = 1;
const SNAC_HEADER_LEN: usize = 144;
const SNAC_MAX_STAGES: usize = 4;

/// Portable SNAC hierarchical-code container, version 1.
///
/// Unlike flat RVQ codecs, each SNAC stage has its own temporal stride and
/// therefore its own code count. The payload is stage-major unsigned u32
/// little endian; `stage_lengths` and `vq_strides` make those extents
/// explicit. The model fingerprint covers every GGUF tensor name, dtype,
/// shape, and payload so decode cannot silently use a different checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnacCodesV1 {
    pub(crate) sample_rate: u32,
    pub(crate) n_stages: u32,
    pub(crate) codebook_size: u32,
    pub(crate) latent_dim: u32,
    pub(crate) hop_length: u32,
    pub(crate) base_frames: u64,
    pub(crate) pcm_samples: u64,
    pub(crate) model_sha256: [u8; 32],
    pub(crate) vq_strides: [u32; SNAC_MAX_STAGES],
    pub(crate) stage_lengths: [u64; SNAC_MAX_STAGES],
    pub(crate) codes: Vec<Vec<u32>>,
}

impl SnacCodesV1 {
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let code_count = self.codes.iter().try_fold(0usize, |total, stage| {
            total
                .checked_add(stage.len())
                .ok_or("SNAC code count overflow")
        })?;
        let payload_len = code_count
            .checked_mul(4)
            .ok_or("SNAC code payload size overflow")?;
        let capacity = SNAC_HEADER_LEN
            .checked_add(payload_len)
            .ok_or("SNAC container size overflow")?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(SNAC_MAGIC);
        out.extend_from_slice(&SNAC_VERSION.to_le_bytes());
        out.extend_from_slice(&(SNAC_HEADER_LEN as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // flags, reserved
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.n_stages.to_le_bytes());
        out.extend_from_slice(&self.codebook_size.to_le_bytes());
        out.extend_from_slice(&self.latent_dim.to_le_bytes());
        out.extend_from_slice(&self.hop_length.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
        out.extend_from_slice(&self.base_frames.to_le_bytes());
        out.extend_from_slice(&self.pcm_samples.to_le_bytes());
        out.extend_from_slice(&self.model_sha256);
        for stride in self.vq_strides {
            out.extend_from_slice(&stride.to_le_bytes());
        }
        for length in self.stage_lengths {
            out.extend_from_slice(&length.to_le_bytes());
        }
        out.extend_from_slice(&[0u8; 8]); // reserved for additive v1 fields
        debug_assert_eq!(out.len(), SNAC_HEADER_LEN);
        for stage in &self.codes {
            for code in stage {
                out.extend_from_slice(&code.to_le_bytes());
            }
        }
        Ok(out)
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < SNAC_HEADER_LEN {
            return Err(format!(
                "SNAC code container is truncated: {} bytes, need at least {SNAC_HEADER_LEN}",
                bytes.len()
            ));
        }
        if &bytes[..8] != SNAC_MAGIC {
            return Err("SNAC code container has wrong magic; expected `VKRSNAC1`".to_owned());
        }
        let version = read_u16(bytes, 8)?;
        if version != SNAC_VERSION {
            return Err(format!(
                "SNAC code container version {version} is unsupported; this build reads version {SNAC_VERSION}"
            ));
        }
        let header_len = read_u16(bytes, 10)? as usize;
        if header_len != SNAC_HEADER_LEN {
            return Err(format!(
                "SNAC code container v1 header length {header_len} != {SNAC_HEADER_LEN}"
            ));
        }
        if read_u32(bytes, 12)? != 0
            || read_u32(bytes, 36)? != 0
            || bytes[136..144].iter().any(|&byte| byte != 0)
        {
            return Err("SNAC code container v1 has non-zero reserved fields".to_owned());
        }

        let n_stages = read_u32(bytes, 20)?;
        let mut model_sha256 = [0u8; 32];
        model_sha256.copy_from_slice(&bytes[56..88]);
        let mut vq_strides = [0u32; SNAC_MAX_STAGES];
        for (stage, stride) in vq_strides.iter_mut().enumerate() {
            *stride = read_u32(bytes, 88 + stage * 4)?;
        }
        let mut stage_lengths = [0u64; SNAC_MAX_STAGES];
        for (stage, length) in stage_lengths.iter_mut().enumerate() {
            *length = read_u64(bytes, 104 + stage * 8)?;
        }

        let active_stages = usize::try_from(n_stages)
            .map_err(|_| "SNAC stage count does not fit this host".to_owned())?;
        if active_stages > SNAC_MAX_STAGES {
            return Err(format!(
                "SNAC code container has {active_stages} stages; v1 supports at most {SNAC_MAX_STAGES}"
            ));
        }
        let code_count = stage_lengths[..active_stages]
            .iter()
            .try_fold(0u64, |total, &length| {
                total.checked_add(length).ok_or("SNAC code count overflow")
            })?;
        let code_count = usize::try_from(code_count)
            .map_err(|_| "SNAC code count does not fit this host".to_owned())?;
        let expected = SNAC_HEADER_LEN
            .checked_add(
                code_count
                    .checked_mul(4)
                    .ok_or("SNAC payload size overflow")?,
            )
            .ok_or("SNAC container size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "SNAC code container length {} != header {SNAC_HEADER_LEN} + {code_count} u32 codes ({expected})",
                bytes.len()
            ));
        }

        let mut payload = bytes[SNAC_HEADER_LEN..].chunks_exact(4);
        let mut codes = Vec::with_capacity(active_stages);
        for &length in &stage_lengths[..active_stages] {
            let length = usize::try_from(length)
                .map_err(|_| "SNAC stage length does not fit this host".to_owned())?;
            let mut stage_codes = Vec::with_capacity(length);
            for _ in 0..length {
                let word = payload
                    .next()
                    .ok_or("SNAC code container payload is truncated")?;
                stage_codes.push(u32::from_le_bytes([word[0], word[1], word[2], word[3]]));
            }
            codes.push(stage_codes);
        }
        debug_assert!(payload.next().is_none());
        let parsed = Self {
            sample_rate: read_u32(bytes, 16)?,
            n_stages,
            codebook_size: read_u32(bytes, 24)?,
            latent_dim: read_u32(bytes, 28)?,
            hop_length: read_u32(bytes, 32)?,
            base_frames: read_u64(bytes, 40)?,
            pcm_samples: read_u64(bytes, 48)?,
            model_sha256,
            vq_strides,
            stage_lengths,
            codes,
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0
            || self.codebook_size == 0
            || self.latent_dim == 0
            || self.hop_length == 0
            || self.base_frames == 0
            || self.pcm_samples == 0
        {
            return Err(
                "SNAC code container v1 has a zero rate, axis, frame, hop, or PCM length"
                    .to_owned(),
            );
        }
        let n_stages = usize::try_from(self.n_stages)
            .map_err(|_| "SNAC stage count does not fit this host")?;
        if !(1..=SNAC_MAX_STAGES).contains(&n_stages) {
            return Err(format!(
                "SNAC code container v1 stage count {n_stages} is outside 1..={SNAC_MAX_STAGES}"
            ));
        }
        if self.codes.len() != n_stages {
            return Err(format!(
                "SNAC code container has {} stage vectors, expected {n_stages}",
                self.codes.len()
            ));
        }
        for stage in 0..n_stages {
            let stride = u64::from(self.vq_strides[stage]);
            if stride == 0 || !self.base_frames.is_multiple_of(stride) {
                return Err(format!(
                    "SNAC stage {stage} stride {stride} does not divide base_frames {}",
                    self.base_frames
                ));
            }
            let expected = self.base_frames / stride;
            if self.stage_lengths[stage] != expected {
                return Err(format!(
                    "SNAC stage {stage} length {} != base_frames/stride {expected}",
                    self.stage_lengths[stage]
                ));
            }
            let actual = u64::try_from(self.codes[stage].len())
                .map_err(|_| "SNAC stage code count exceeds u64")?;
            if actual != expected {
                return Err(format!(
                    "SNAC stage {stage} code vector has {actual} entries, expected {expected}"
                ));
            }
            if let Some((index, code)) = self.codes[stage]
                .iter()
                .copied()
                .enumerate()
                .find(|(_, code)| *code >= self.codebook_size)
            {
                return Err(format!(
                    "SNAC stage {stage} code {code} at index {index} is outside codebook_size {}",
                    self.codebook_size
                ));
            }
        }
        for stage in n_stages..SNAC_MAX_STAGES {
            if self.vq_strides[stage] != 0 || self.stage_lengths[stage] != 0 {
                return Err(format!(
                    "SNAC inactive stage {stage} must have zero stride and length"
                ));
            }
        }
        let padded_samples = self
            .base_frames
            .checked_mul(u64::from(self.hop_length))
            .ok_or("SNAC padded PCM length overflow")?;
        if self.pcm_samples > padded_samples {
            return Err(format!(
                "SNAC original PCM length {} exceeds padded decode extent {padded_samples}",
                self.pcm_samples
            ));
        }
        Ok(())
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let b = bytes
        .get(offset..offset + 2)
        .ok_or("Mimi code container truncated")?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let b = bytes
        .get(offset..offset + 4)
        .ok_or("Mimi code container truncated")?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let b = bytes
        .get(offset..offset + 8)
        .ok_or("Mimi code container truncated")?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

// Zero-dependency SHA-256 (FIPS-180-4 §6.2), matching the first-party
// WebGPU/Vulkan shader-pin implementation. It is local to the CLI to avoid a
// cross-backend dependency for a portable model-compatibility fingerprint.
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

#[must_use]
pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = SHA256_H0;
    let bit_len = (data.len() as u64) * 8;
    let mut buf = Vec::with_capacity(data.len() + 72);
    buf.extend_from_slice(data);
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0);
    }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    for block in buf.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
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
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_punc_tsv_pairs_ids_and_escaped_unicode_tokens() {
        let parsed = parse_ct_punc_tsv(
            "vokra-ct-punc-tsv-v1\r\n7\thello\r\n42\t世\\u{754C}\r\n9\ta\\tb\\\\c\n",
        )
        .unwrap();
        assert_eq!(parsed.token_ids, vec![7, 42, 9]);
        assert_eq!(parsed.tokens, vec!["hello", "世界", "a\tb\\c"]);
    }

    #[test]
    fn ct_punc_tsv_rejects_unversioned_unpaired_and_bad_escape_inputs() {
        assert!(
            parse_ct_punc_tsv("7\thello\n")
                .unwrap_err()
                .contains("header")
        );
        assert!(
            parse_ct_punc_tsv("vokra-ct-punc-tsv-v1\n7\n")
                .unwrap_err()
                .contains("no tab")
        );
        assert!(
            parse_ct_punc_tsv("vokra-ct-punc-tsv-v1\n7\ta\\x\n")
                .unwrap_err()
                .contains("unknown escape")
        );
    }

    fn sample_melotts_features() -> MeloTextFeaturesV1 {
        MeloTextFeaturesV1 {
            variant: MeloFeatureVariantV1::Japanese,
            speaker_id: 0,
            phoneme_ids: vec![0, 7],
            tones: vec![0, 3],
            language_ids: vec![0, 1],
            bert: vec![0.25; 2 * MELOTTS_BERT_DIMENSION],
            ja_bert: vec![-0.5; 2 * MELOTTS_JA_BERT_DIMENSION],
        }
    }

    #[test]
    fn melotts_v1_round_trips_language_pinned_position_major_features() {
        let value = sample_melotts_features();
        let bytes = value.to_bytes();
        assert_eq!(&bytes[..8], b"VKRMELO1");
        assert_eq!(&bytes[16..20], &4u32.to_le_bytes());
        assert_eq!(MeloTextFeaturesV1::from_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn melotts_v1_rejects_wrong_language_shape_truncation_and_nonfinite_values() {
        let mut bytes = sample_melotts_features().to_bytes();
        bytes[16..20].copy_from_slice(&99u32.to_le_bytes());
        assert!(
            MeloTextFeaturesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("unknown variant")
        );

        let mut bytes = sample_melotts_features().to_bytes();
        bytes.pop();
        assert!(
            MeloTextFeaturesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("length")
        );

        let mut bytes = sample_melotts_features().to_bytes();
        let first_bert = MELOTTS_HEADER_LEN + 2 * 3 * 4;
        bytes[first_bert..first_bert + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(
            MeloTextFeaturesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("bert[0] is non-finite")
        );

        let mut bytes = sample_melotts_features().to_bytes();
        bytes[28..32].copy_from_slice(&768u32.to_le_bytes());
        assert!(
            MeloTextFeaturesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("requires 1024 and 768")
        );
    }

    fn sample_codes() -> MimiCodesV1 {
        MimiCodesV1 {
            sample_rate: 24_000,
            frame_rate_mhz: 12_500,
            n_codebooks: 2,
            codebook_size: 2048,
            feature_dimension: 512,
            n_frames: 3,
            pcm_samples: 5_760,
            model_sha256: sha256(b"codebook"),
            codes: vec![1, 2, 3, 4, 5, 6],
        }
    }

    #[test]
    fn mimi_v1_round_trips_and_pins_little_endian_u32_codes() {
        let value = sample_codes();
        let bytes = value.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"VKRMCODE");
        assert_eq!(&bytes[96..100], &1u32.to_le_bytes());
        assert_eq!(MimiCodesV1::from_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn mimi_v1_rejects_payload_length_and_code_range_mismatch() {
        let mut bytes = sample_codes().to_bytes().unwrap();
        bytes.pop();
        assert!(
            MimiCodesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("length")
        );

        let mut value = sample_codes();
        value.codes[2] = value.codebook_size;
        assert!(
            value
                .to_bytes()
                .unwrap_err()
                .contains("outside codebook_size")
        );
    }

    fn sample_focalcodec_codes() -> FocalCodecCodesV1 {
        FocalCodecCodesV1 {
            sample_rate: 16_000,
            token_hz_times_two: 25,
            frame_hop: 1_280,
            codebook_size: 8_192,
            code_dimension: 13,
            token_count: 3,
            pcm_samples: 3_200,
            model_sha256: sha256(b"focalcodec-model"),
            tokens: vec![1, 4_096, 8_191],
        }
    }

    #[test]
    fn focalcodec_v1_round_trips_single_codebook_tokens() {
        let value = sample_focalcodec_codes();
        let bytes = value.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"VKRFOC01");
        assert_eq!(&bytes[96..100], &1u32.to_le_bytes());
        assert_eq!(FocalCodecCodesV1::from_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn focalcodec_v1_rejects_truncation_reserved_bits_and_token_range() {
        let mut bytes = sample_focalcodec_codes().to_bytes().unwrap();
        bytes.pop();
        assert!(
            FocalCodecCodesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("length")
        );

        let mut bytes = sample_focalcodec_codes().to_bytes().unwrap();
        bytes[84] = 1;
        assert!(
            FocalCodecCodesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("reserved")
        );

        let mut value = sample_focalcodec_codes();
        value.tokens[1] = value.codebook_size;
        assert!(
            value
                .to_bytes()
                .unwrap_err()
                .contains("outside codebook_size")
        );
    }

    fn sample_facodec_codes() -> FacodecCodesV1 {
        FacodecCodesV1 {
            sample_rate: 16_000,
            frame_hop: 200,
            n_codebooks: 6,
            codebook_size: 1_024,
            embedding_dim: 3,
            frames: 2,
            pcm_samples: 437,
            model_sha256: sha256(b"facodec-v2-model"),
            speaker_embedding: vec![0.25, -0.5, 1.0],
            codes: vec![0, 1, 2, 3, 4, 5, 1_023, 11, 12, 13, 14, 15],
        }
    }

    #[test]
    fn facodec_v1_round_trips_embedding_then_frame_major_codes() {
        let value = sample_facodec_codes();
        let bytes = value.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"VKRFA2V1");
        assert_eq!(&bytes[112..116], &0.25f32.to_le_bytes());
        assert_eq!(&bytes[124..128], &0u32.to_le_bytes());
        assert_eq!(FacodecCodesV1::from_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn facodec_v1_rejects_truncation_reserved_range_and_non_finite_style() {
        let mut bytes = sample_facodec_codes().to_bytes().unwrap();
        bytes.pop();
        assert!(
            FacodecCodesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("length")
        );

        let mut bytes = sample_facodec_codes().to_bytes().unwrap();
        bytes[100] = 1;
        assert!(
            FacodecCodesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("reserved")
        );

        let mut value = sample_facodec_codes();
        value.codes[7] = value.codebook_size;
        assert!(
            value
                .to_bytes()
                .unwrap_err()
                .contains("outside codebook_size")
        );

        let mut value = sample_facodec_codes();
        value.speaker_embedding[1] = f32::NAN;
        assert!(value.to_bytes().unwrap_err().contains("non-finite"));

        let mut value = sample_facodec_codes();
        value.pcm_samples = 600;
        assert!(value.to_bytes().unwrap_err().contains("inconsistent"));
    }

    fn sample_snac_codes() -> SnacCodesV1 {
        SnacCodesV1 {
            sample_rate: 24_000,
            n_stages: 3,
            codebook_size: 4096,
            latent_dim: 768,
            hop_length: 2_048,
            base_frames: 8,
            pcm_samples: 15_000,
            model_sha256: sha256(b"snac-model"),
            vq_strides: [4, 2, 1, 0],
            stage_lengths: [2, 4, 8, 0],
            codes: vec![vec![1, 2], vec![3, 4, 5, 6], vec![7; 8]],
        }
    }

    #[test]
    fn snac_v1_round_trips_hierarchical_stage_major_codes() {
        let value = sample_snac_codes();
        let bytes = value.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"VKRSNAC1");
        assert_eq!(&bytes[144..148], &1u32.to_le_bytes());
        assert_eq!(&bytes[152..156], &3u32.to_le_bytes());
        assert_eq!(SnacCodesV1::from_bytes(&bytes).unwrap(), value);
    }

    #[test]
    fn snac_v1_rejects_truncation_code_range_and_stage_extent_mismatch() {
        let mut bytes = sample_snac_codes().to_bytes().unwrap();
        bytes.pop();
        assert!(
            SnacCodesV1::from_bytes(&bytes)
                .unwrap_err()
                .contains("length")
        );

        let mut value = sample_snac_codes();
        value.codes[1][2] = value.codebook_size;
        assert!(
            value
                .to_bytes()
                .unwrap_err()
                .contains("outside codebook_size")
        );

        let mut value = sample_snac_codes();
        value.stage_lengths[1] = 3;
        assert!(value.to_bytes().unwrap_err().contains("base_frames/stride"));
    }

    #[test]
    fn snac_v1_rejects_nonzero_inactive_stage_and_impossible_pcm_length() {
        let mut value = sample_snac_codes();
        value.vq_strides[3] = 1;
        assert!(value.to_bytes().unwrap_err().contains("inactive stage 3"));

        let mut value = sample_snac_codes();
        value.pcm_samples = value.base_frames * u64::from(value.hop_length) + 1;
        assert!(
            value
                .to_bytes()
                .unwrap_err()
                .contains("exceeds padded decode extent")
        );
    }

    #[test]
    fn sha256_matches_nist_abc_vector() {
        let got = sha256(b"abc");
        let want = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(got, want);
    }
}
