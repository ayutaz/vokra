//! Minimal, dependency-free mono WAV reader + writer (M1-10a).
//!
//! Just enough RIFF/WAVE to read a clip for `run`/`bench` input and to write the
//! `run tts` output. No external crate and no GPL (NFR-DS-02 / NFR-LC-02);
//! stereo, other bit depths and compressed formats are explicit errors. The
//! reader mirrors the one in `vokra-eval` / `vokra-models` (kept duplicated so
//! `vokra-cli` stays a lean leaf crate).

/// Decoded mono PCM plus its declared sample rate.
#[derive(Debug, Clone)]
pub(crate) struct Wav {
    /// Sample rate in Hz, from the `fmt ` chunk.
    pub(crate) sample_rate: u32,
    /// Mono samples in `[-1.0, 1.0]` (int16 is scaled by `1/32768`).
    pub(crate) samples: Vec<f32>,
}

/// Reads a mono WAV file (float32 or int16 PCM) from `path`.
pub(crate) fn read_wav(path: impl AsRef<std::path::Path>) -> Result<Wav, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    parse(&bytes)
}

/// Writes `samples` as a mono IEEE-float32 WAV at `sample_rate` Hz to `path`.
pub(crate) fn write_wav(
    path: impl AsRef<std::path::Path>,
    samples: &[f32],
    sample_rate: u32,
) -> Result<(), String> {
    let bits: u16 = 32;
    let channels: u16 = 1;
    let block_align: u16 = channels * (bits / 8);
    let byte_rate: u32 = sample_rate * u32::from(block_align);
    let data_len: u32 = (samples.len() * 4) as u32;

    let mut out = Vec::with_capacity(44 + samples.len() * 4);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, &out).map_err(|e| e.to_string())
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

    #[test]
    fn write_then_read_round_trips_float32() {
        let samples = vec![0.5f32, -0.25, 0.0, 1.0, -1.0];
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-wav-{}.wav", std::process::id()));
        write_wav(&path, &samples, 16_000).expect("write");
        let w = read_wav(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert_eq!(w.sample_rate, 16_000);
        assert_eq!(w.samples, samples);
    }

    #[test]
    fn rejects_non_riff_and_stereo() {
        assert!(parse(b"nope").is_err());
        // A valid mono file, then flip the channel count to 2.
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-wav-stereo-{}.wav", std::process::id()));
        write_wav(&path, &[0.0f32; 4], 8_000).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes[22] = 2; // channels field
        assert!(parse(&bytes).is_err());
    }
}
