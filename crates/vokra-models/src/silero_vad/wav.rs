//! Minimal, dependency-free WAV reader for mono PCM (M0-05-T10).
//!
//! Just enough of RIFF/WAVE to feed the VAD demo and the parity tests: mono,
//! either 32-bit IEEE float (`fmt` tag 3) or 16-bit signed int (`fmt` tag 1).
//! No external crate (NFR-DS-02, and no GPL — NFR-LC-02). Anything else
//! (stereo, other bit depths, compressed) is an explicit error.

use vokra_core::{Result, VokraError};

/// Decoded mono PCM plus its sample rate.
pub struct WavData {
    /// Sample rate in Hz (as declared by the `fmt ` chunk).
    pub sample_rate: u32,
    /// Mono samples in `[-1.0, 1.0]` (int16 is scaled by `1/32768`).
    pub samples: Vec<f32>,
}

/// Reads a mono WAV file (float32 or int16 PCM) into [`WavData`].
pub fn read_wav_f32(path: impl AsRef<std::path::Path>) -> Result<WavData> {
    let bytes = std::fs::read(path)?;
    parse(&bytes)
}

fn err(msg: impl Into<String>) -> VokraError {
    VokraError::InvalidArgument(msg.into())
}

fn le_u16(b: &[u8], off: usize) -> Result<u16> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| err("WAV truncated"))
}

fn le_u32(b: &[u8], off: usize) -> Result<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| err("WAV truncated"))
}

/// Parses a WAV byte buffer (exposed for the round-trip test).
pub(super) fn parse(b: &[u8]) -> Result<WavData> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(err("not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (audio_format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = le_u32(b, pos + 4)? as usize;
        let body_start = pos + 8;
        let body_end = body_start
            .checked_add(size)
            .filter(|&e| e <= b.len())
            .ok_or_else(|| err("WAV chunk size out of range"))?;
        let body = &b[body_start..body_end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err(err("WAV fmt chunk too small"));
            }
            fmt = Some((
                le_u16(body, 0)?,
                le_u16(body, 2)?,
                le_u32(body, 4)?,
                le_u16(body, 14)?,
            ));
        } else if id == b"data" {
            data = Some(body);
        }
        // Chunks are word-aligned: skip a pad byte after odd-sized bodies.
        pos = body_end + (size & 1);
    }

    let (audio_format, channels, rate, bits) = fmt.ok_or_else(|| err("WAV has no fmt chunk"))?;
    let data = data.ok_or_else(|| err("WAV has no data chunk"))?;
    if channels != 1 {
        return Err(err(format!(
            "only mono WAV supported, got {channels} channels"
        )));
    }
    let samples = match (audio_format, bits) {
        // IEEE float32.
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // Signed int16.
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect(),
        _ => {
            return Err(err(format!(
                "unsupported WAV format (audio_format={audio_format}, bits={bits}); \
                 use mono float32 or int16"
            )));
        }
    };
    Ok(WavData {
        sample_rate: rate,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_wav(fmt_tag: u16, bits: u16, rate: u32, body: &[u8]) -> Vec<u8> {
        let block_align = bits / 8;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&fmt_tag.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes()); // channels
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        let mut chunks = Vec::new();
        chunks.extend_from_slice(b"fmt ");
        chunks.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        chunks.extend_from_slice(&fmt);
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chunks.extend_from_slice(body);
        let mut riff = Vec::new();
        riff.extend_from_slice(b"RIFF");
        riff.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        riff.extend_from_slice(b"WAVE");
        riff.extend_from_slice(&chunks);
        riff
    }

    #[test]
    fn reads_float32_mono() {
        let body: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let w = parse(&build_wav(3, 32, 16000, &body)).unwrap();
        assert_eq!(w.sample_rate, 16000);
        assert_eq!(w.samples, vec![0.5, -0.25]);
    }

    #[test]
    fn reads_int16_mono_scaled() {
        let body: Vec<u8> = [16384i16, -32768]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = parse(&build_wav(1, 16, 8000, &body)).unwrap();
        assert_eq!(w.sample_rate, 8000);
        assert_eq!(w.samples, vec![0.5, -1.0]);
    }

    #[test]
    fn rejects_stereo_and_bad_header() {
        assert!(parse(b"nope").is_err());
        // channels = 2
        let mut wav = build_wav(3, 32, 16000, &[0, 0, 0, 0]);
        wav[22] = 2;
        assert!(parse(&wav).is_err());
    }

    /// A 16-byte `fmt ` body for mono `rate`/`bits` with format tag `fmt_tag`.
    fn fmt_body(fmt_tag: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block_align = bits / 8;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&fmt_tag.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes()); // channels
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt
    }

    /// Emits one RIFF sub-chunk (id, u32 LE size, body) plus the word-alignment
    /// pad byte the spec requires after an odd-sized body.
    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(id);
        c.extend_from_slice(&(body.len() as u32).to_le_bytes());
        c.extend_from_slice(body);
        if body.len() % 2 == 1 {
            c.push(0);
        }
        c
    }

    /// Wraps concatenated sub-chunks in a `RIFF....WAVE` container.
    fn riff(chunks: &[u8]) -> Vec<u8> {
        let mut r = Vec::new();
        r.extend_from_slice(b"RIFF");
        r.extend_from_slice(&((4 + chunks.len()) as u32).to_le_bytes());
        r.extend_from_slice(b"WAVE");
        r.extend_from_slice(chunks);
        r
    }

    /// Unwraps the message of an expected `InvalidArgument` (WavData has no Debug).
    fn err_msg(r: Result<WavData>) -> String {
        match r {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(VokraError::InvalidArgument(m)) => m,
            Err(_) => panic!("expected InvalidArgument"),
        }
    }

    #[test]
    fn skips_odd_sized_chunk_before_data() {
        // A 3-byte (odd) LIST chunk between `fmt ` and `data` forces the
        // word-alignment pad-byte skip (`pos = body_end + (size & 1)`); the
        // float samples after it must still be located and decoded correctly.
        let body: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut chunks = Vec::new();
        chunks.extend_from_slice(&chunk(b"fmt ", &fmt_body(3, 16000, 32)));
        chunks.extend_from_slice(&chunk(b"LIST", b"INF")); // odd (3-byte) body
        chunks.extend_from_slice(&chunk(b"data", &body));
        let w = parse(&riff(&chunks)).unwrap();
        assert_eq!(w.sample_rate, 16000);
        assert_eq!(w.samples, vec![0.5, -0.25]);
    }

    #[test]
    fn rejects_unsupported_format_and_bit_depth() {
        // PCM tag (1) but 24-bit is unsupported.
        assert!(
            err_msg(parse(&build_wav(1, 24, 16000, &[0u8; 6]))).contains("unsupported WAV format")
        );
        // A-law tag (6) at 16-bit is unsupported.
        assert!(
            err_msg(parse(&build_wav(6, 16, 16000, &[0u8; 4]))).contains("unsupported WAV format")
        );
    }

    #[test]
    fn rejects_missing_data_and_missing_fmt() {
        // `fmt ` present but no `data` chunk.
        let only_fmt = riff(&chunk(b"fmt ", &fmt_body(3, 16000, 32)));
        assert!(err_msg(parse(&only_fmt)).contains("no data chunk"));
        // `data` present but no `fmt ` chunk (fmt is checked first).
        let only_data = riff(&chunk(b"data", &0.5f32.to_le_bytes()));
        assert!(err_msg(parse(&only_data)).contains("no fmt chunk"));
    }

    #[test]
    fn rejects_data_chunk_size_out_of_range() {
        // A `data` header declaring 1000 bytes with none present must be
        // rejected, not read out of bounds.
        let mut chunks = Vec::new();
        chunks.extend_from_slice(&chunk(b"fmt ", &fmt_body(3, 16000, 32)));
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&1000u32.to_le_bytes());
        assert!(err_msg(parse(&riff(&chunks))).contains("chunk size out of range"));
    }
}
