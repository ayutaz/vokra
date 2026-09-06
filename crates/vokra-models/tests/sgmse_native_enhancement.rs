//! VAST-only end-to-end SGMSE enhancement consumer.
//!
//! The wrapper verifies the official reference packet and GGUF before this
//! ignored test starts. This test consumes the exact PCM and captured
//! prior/corrector/predictor noise tensors; it never creates a random source
//! or compares against a Python reimplementation.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use vokra_models::compute::Compute;
use vokra_models::sgmse::{SgmseModel, SgmseNoise};

const GGUF_ENV: &str = "VOKRA_SGMSE_GGUF";
const REFERENCE_ENV: &str = "VOKRA_SGMSE_REFERENCE_DIR";
const NATIVE_ENV: &str = "VOKRA_SGMSE_NATIVE_OUTPUT_DIR";
const VAST_ENV: &str = "VOKRA_PUBLISH_ON_VAST";
const EXPECTED_NOISE_CALLS: usize = 61;

fn required_env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| panic!("{name} is required for SGMSE enhancement parity"))
}

fn reject_symlink_ancestry(path: &Path, label: &str) {
    let mut current = path;
    loop {
        assert!(
            !current.is_symlink(),
            "{label} has symlink ancestry: {}",
            path.display()
        );
        if current == Path::new("/") {
            break;
        }
        current = current
            .parent()
            .unwrap_or_else(|| panic!("{label} has no absolute parent: {}", path.display()));
    }
}

fn read_f32(path: &Path, label: &str) -> Vec<f32> {
    assert!(!path.is_symlink(), "{label} must not be a symlink");
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {label}: {error}"));
    assert!(
        !bytes.is_empty() && bytes.len() % 4 == 0,
        "{label} has invalid bytes"
    );
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
        .collect();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains non-finite values"
    );
    values
}

fn write_new_f32(path: &Path, values: &[f32]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    for value in values {
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "non-finite enhancement output",
            ));
        }
        file.write_all(&value.to_le_bytes())?;
    }
    file.sync_all()
}

#[derive(Debug)]
struct NoiseCall {
    index: usize,
    kind: u8,
    step: i64,
    corrector: bool,
    count: usize,
    offset: usize,
}

struct CapturedNoise {
    bytes: Vec<u8>,
    calls: Vec<NoiseCall>,
    cursor: usize,
}

impl CapturedNoise {
    fn load(reference: &Path) -> Self {
        let bytes = fs::read(reference.join("noise.f32")).expect("read captured noise");
        let text = fs::read_to_string(reference.join("noise_calls.txt")).expect("read noise index");
        let calls = text
            .lines()
            .map(|line| {
                let fields: Vec<_> = line.split_whitespace().collect();
                assert_eq!(
                    fields.len(),
                    6,
                    "noise index row has the wrong column count"
                );
                NoiseCall {
                    index: fields[0].parse().expect("noise index"),
                    kind: fields[1].parse().expect("noise kind"),
                    step: fields[2].parse().expect("noise step"),
                    corrector: fields[3].parse::<u8>().expect("noise corrector") != 0,
                    count: fields[4].parse().expect("noise count"),
                    offset: fields[5].parse().expect("noise offset"),
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            calls.len(),
            EXPECTED_NOISE_CALLS,
            "captured noise call count"
        );
        Self {
            bytes,
            calls,
            cursor: 0,
        }
    }

    fn fill_next(&mut self, step: i64, corrector: bool, out: &mut [f32]) {
        let call = self
            .calls
            .get(self.cursor)
            .expect("captured noise exhausted");
        let expected_kind = if self.cursor == 0 { 0 } else { 1 };
        assert_eq!(call.index, self.cursor, "captured noise order");
        assert_eq!(call.kind, expected_kind, "captured noise kind");
        assert_eq!(call.step, step, "captured noise step");
        assert_eq!(call.corrector, corrector, "captured noise corrector flag");
        assert_eq!(call.count, out.len(), "captured noise shape");
        let byte_end = call
            .offset
            .checked_add(call.count * 4)
            .expect("noise offset overflow");
        assert!(
            byte_end <= self.bytes.len(),
            "captured noise payload bounds"
        );
        for (output, chunk) in out
            .iter_mut()
            .zip(self.bytes[call.offset..byte_end].chunks_exact(4))
        {
            *output = f32::from_le_bytes(chunk.try_into().expect("noise f32 bytes"));
            assert!(output.is_finite(), "captured noise is non-finite");
        }
        self.cursor += 1;
    }
}

impl SgmseNoise for CapturedNoise {
    fn fill_prior(&mut self, out: &mut [f32]) -> vokra_core::Result<()> {
        self.fill_next(-1, false, out);
        Ok(())
    }

    fn fill(&mut self, step: usize, corrector: bool, out: &mut [f32]) -> vokra_core::Result<()> {
        self.fill_next(step as i64, corrector, out);
        Ok(())
    }
}

#[test]
#[ignore = "real SGMSE enhancement/reference packet runs on VAST/Linux only"]
fn sgmse_native_enhancement_matches_official_reference() {
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "SGMSE enhancement is VAST/Linux-only"
    );
    assert_eq!(
        std::env::var(VAST_ENV).as_deref(),
        Ok("1"),
        "set VOKRA_PUBLISH_ON_VAST=1"
    );
    let gguf = required_env(GGUF_ENV);
    let reference = required_env(REFERENCE_ENV);
    let native = required_env(NATIVE_ENV);
    assert!(gguf.is_absolute() && reference.is_absolute() && native.is_absolute());
    reject_symlink_ancestry(&gguf, GGUF_ENV);
    reject_symlink_ancestry(&reference, REFERENCE_ENV);
    reject_symlink_ancestry(&native, NATIVE_ENV);
    assert!(gguf.is_file(), "GGUF is missing");
    assert!(reference.is_dir(), "reference packet is missing");
    assert!(
        !native.exists() && !native.is_symlink(),
        "native output must be absent"
    );
    let native_parent = native.parent().expect("native output parent");
    assert!(native_parent.is_dir(), "native output parent is missing");
    reject_symlink_ancestry(native_parent, "native output parent");

    let pcm = read_f32(&reference.join("input_pcm.f32"), "reference input PCM");
    let mut noise = CapturedNoise::load(&reference);
    let file = vokra_mmap::open_gguf(&gguf).expect("open authenticated SGMSE GGUF");
    let mut model = SgmseModel::from_gguf(&file).expect("bind authenticated SGMSE GGUF");
    let enhanced = model
        .enhance(&Compute::cpu(), &pcm, &mut noise)
        .expect("run native SGMSE CPU enhancement");
    assert_eq!(enhanced.len(), pcm.len(), "native enhancement length");
    assert_eq!(
        noise.cursor, EXPECTED_NOISE_CALLS,
        "native did not consume all captured noise"
    );
    fs::create_dir(&native).expect("create absent native output directory");
    write_new_f32(&native.join("enhanced_pcm.f32"), &enhanced).expect("write native enhancement");
    let entries: Vec<_> = fs::read_dir(&native)
        .expect("read native output")
        .map(|entry| entry.expect("read output entry").file_name())
        .collect();
    assert_eq!(entries, vec![std::ffi::OsString::from("enhanced_pcm.f32")]);
}
