//! Minimal, dependency-free WAV reader for mono float32 / int16 PCM.
//!
//! Just enough of RIFF/WAVE to read evaluation clips for the audio metrics; no
//! external crate and no GPL (NFR-DS-02 / NFR-LC-02). Stereo, other bit depths
//! and compressed formats are explicit errors. (This mirrors the reader in
//! `vokra-models`; it is intentionally duplicated so `vokra-eval` stays a lean
//! leaf crate that does not pull the model crate into its build.)

/// Decoded mono PCM plus its declared sample rate.
#[derive(Debug, Clone)]
pub struct Wav {
    /// Sample rate in Hz, from the `fmt ` chunk.
    pub sample_rate: u32,
    /// Mono samples in `[-1.0, 1.0]` (int16 is scaled by `1/32768`).
    pub samples: Vec<f32>,
}

/// Reads a mono WAV file (float32 or int16 PCM) from `path`.
///
/// # Errors
///
/// Returns a human-readable message if the file cannot be read or is not a
/// supported mono float32/int16 WAV.
pub fn read_wav(path: impl AsRef<std::path::Path>) -> Result<Wav, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    parse(&bytes)
}

fn le_u16(b: &[u8], off: usize) -> Result<u16, String> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| "WAV truncated".to_owned())
}

fn le_u32(b: &[u8], off: usize) -> Result<u32, String> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "WAV truncated".to_owned())
}

/// Parses a WAV byte buffer (RIFF/WAVE chunk walk).
fn parse(b: &[u8]) -> Result<Wav, String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_owned());
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = le_u32(b, pos + 4)? as usize;
        let body_start = pos + 8;
        let body_end = body_start
            .checked_add(size)
            .filter(|&e| e <= b.len())
            .ok_or_else(|| "WAV chunk size out of range".to_owned())?;
        let body = &b[body_start..body_end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err("WAV fmt chunk too small".to_owned());
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
        // Chunks are word-aligned: skip a pad byte after an odd-sized body.
        pos = body_end + (size & 1);
    }

    let (audio_format, channels, rate, bits) =
        fmt.ok_or_else(|| "WAV has no fmt chunk".to_owned())?;
    let data = data.ok_or_else(|| "WAV has no data chunk".to_owned())?;
    if channels != 1 {
        return Err(format!("only mono WAV supported, got {channels} channels"));
    }
    let samples = match (audio_format, bits) {
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect(),
        _ => {
            return Err(format!(
                "unsupported WAV format (audio_format={audio_format}, bits={bits}); \
                 use mono float32 or int16"
            ));
        }
    };
    Ok(Wav {
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
        let w = parse(&build_wav(3, 32, 16_000, &body)).unwrap();
        assert_eq!(w.sample_rate, 16_000);
        assert_eq!(w.samples, vec![0.5, -0.25]);
    }

    #[test]
    fn reads_int16_mono_scaled() {
        let body: Vec<u8> = [16_384i16, -32_768]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = parse(&build_wav(1, 16, 8_000, &body)).unwrap();
        assert_eq!(w.sample_rate, 8_000);
        assert_eq!(w.samples, vec![0.5, -1.0]);
    }

    #[test]
    fn rejects_stereo_and_bad_header() {
        assert!(parse(b"nope").is_err());
        let mut wav = build_wav(3, 32, 16_000, &[0, 0, 0, 0]);
        wav[22] = 2; // channels = 2
        assert!(parse(&wav).is_err());
    }

    #[test]
    fn rejects_unsupported_bit_depth() {
        assert!(parse(&build_wav(1, 24, 16_000, &[0u8; 6])).is_err());
    }
}
