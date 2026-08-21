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
