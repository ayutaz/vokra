//! VAST-only real-weight SGMSE score parity consumer.
//!
//! The independent reference packet is authenticated by the VAST wrapper
//! before this ignored test is launched.  This test only loads the separately
//! authenticated GGUF, consumes the four fixed reference input planes, calls
//! the public [`SgmseModel::score`] API, and writes the two score planes to a
//! newly-created disjoint directory.  It intentionally does not generate or
//! copy expected score values.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use vokra_models::compute::Compute;
use vokra_models::sgmse::SgmseModel;

const HEIGHT: usize = 256;
const WIDTH: usize = 64;
const PLANE_COUNT: usize = HEIGHT * WIDTH;
const PLANE_BYTES: usize = PLANE_COUNT * std::mem::size_of::<f32>();
const GGUF_ENV: &str = "VOKRA_SGMSE_GGUF";
const GGUF_SHA256_ENV: &str = "VOKRA_SGMSE_GGUF_SHA256";
const REFERENCE_ENV: &str = "VOKRA_SGMSE_REFERENCE_DIR";
const NATIVE_ENV: &str = "VOKRA_SGMSE_NATIVE_OUTPUT_DIR";
const VAST_ENV: &str = "VOKRA_PUBLISH_ON_VAST";

fn required_env(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| panic!("{name} is required for VAST SGMSE native parity"))
}

fn require_absolute(path: &Path, label: &str) {
    assert!(
        path.is_absolute(),
        "{label} must be absolute: {}",
        path.display()
    );
}

fn reject_symlink_ancestry(path: &Path, label: &str) {
    let mut current = path;
    loop {
        if current.is_symlink() {
            panic!("{label} has symlink ancestry: {}", current.display());
        }
        if current == Path::new("/") {
            break;
        }
        current = current
            .parent()
            .unwrap_or_else(|| panic!("{label} has no absolute parent: {}", path.display()));
    }
}

fn canonical_existing(path: &Path, label: &str) -> PathBuf {
    path.canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {label} {}: {error}", path.display()))
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
}

fn read_plane(path: &Path, label: &str) -> Vec<f32> {
    assert!(!path.is_symlink(), "{label} must not be a symlink");
    let bytes =
        fs::read(path).unwrap_or_else(|error| panic!("read {label} {}: {error}", path.display()));
    assert_eq!(bytes.len(), PLANE_BYTES, "{label} has the wrong byte count");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    assert_eq!(
        values.len(),
        PLANE_COUNT,
        "{label} has the wrong element count"
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
    values
}

fn write_new_f32(path: &Path, values: &[f32], label: &str) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{label} contains a non-finite value"),
            ));
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn assert_exact_native_output(native: &Path) {
    let entries: Vec<_> = fs::read_dir(native)
        .expect("read newly-created native output directory")
        .map(|entry| entry.expect("read native output entry").path())
        .collect();
    let mut names: Vec<_> = entries
        .iter()
        .map(|path| path.file_name().expect("native entry name").to_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["score_imag.f32".into(), "score_real.f32".into()],
        "native output must contain exactly the two score planes"
    );
    for path in entries {
        assert!(!path.is_symlink(), "native output entry is a symlink");
        assert!(path.is_file(), "native output entry is not regular");
        assert_eq!(
            path.metadata().expect("native output metadata").len() as usize,
            PLANE_BYTES,
            "native output has the wrong byte count: {}",
            path.display()
        );
    }
}

#[test]
#[ignore = "real SGMSE GGUF/reference packet runs on VAST/Linux only"]
fn sgmse_native_score_matches_independent_reference() {
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "SGMSE native parity is VAST/Linux-only"
    );
    assert_eq!(
        std::env::var(VAST_ENV).as_deref(),
        Ok("1"),
        "set VOKRA_PUBLISH_ON_VAST=1 on the disposable VAST worker"
    );

    let gguf_path = required_env(GGUF_ENV);
    let gguf_sha256 = std::env::var(GGUF_SHA256_ENV).unwrap_or_else(|_| {
        panic!("{GGUF_SHA256_ENV} is required; wrapper must verify the GGUF hash")
    });
    assert!(
        gguf_sha256.len() == 64
            && gguf_sha256
                .chars()
                .all(|character| character.is_ascii_hexdigit()),
        "{GGUF_SHA256_ENV} must be a 64-character hexadecimal digest"
    );
    require_absolute(&gguf_path, GGUF_ENV);
    reject_symlink_ancestry(&gguf_path, GGUF_ENV);
    assert!(gguf_path.is_file(), "{GGUF_ENV} is not a regular file");

    let reference_dir = required_env(REFERENCE_ENV);
    require_absolute(&reference_dir, REFERENCE_ENV);
    reject_symlink_ancestry(&reference_dir, REFERENCE_ENV);
    assert!(reference_dir.is_dir(), "{REFERENCE_ENV} is not a directory");

    let native_dir = required_env(NATIVE_ENV);
    require_absolute(&native_dir, NATIVE_ENV);
    assert!(
        !native_dir.exists(),
        "native output must be absent before the test"
    );
    assert!(
        !native_dir.is_symlink(),
        "native output must not be a symlink"
    );
    let native_parent = native_dir.parent().expect("absolute native output parent");
    assert!(native_parent.is_dir(), "native output parent must exist");
    reject_symlink_ancestry(native_parent, "native output parent");

    let gguf_real = canonical_existing(&gguf_path, GGUF_ENV);
    let reference_real = canonical_existing(&reference_dir, REFERENCE_ENV);
    let native_parent_real = canonical_existing(native_parent, "native output parent");
    let native_candidate =
        native_parent_real.join(native_dir.file_name().expect("native output name"));
    assert!(
        !paths_overlap(&gguf_real, &reference_real),
        "GGUF and reference paths overlap"
    );
    assert!(
        !paths_overlap(&gguf_real, &native_candidate),
        "GGUF and native paths overlap"
    );
    assert!(
        !paths_overlap(&reference_real, &native_candidate),
        "reference and native paths overlap"
    );

    let noisy_real = read_plane(
        &reference_dir.join("input_noisy_real.f32"),
        "input noisy real",
    );
    let noisy_imag = read_plane(
        &reference_dir.join("input_noisy_imag.f32"),
        "input noisy imag",
    );
    let condition_real = read_plane(
        &reference_dir.join("input_condition_real.f32"),
        "input condition real",
    );
    let condition_imag = read_plane(
        &reference_dir.join("input_condition_imag.f32"),
        "input condition imag",
    );

    let mut state = Vec::with_capacity(2 * PLANE_COUNT);
    state.extend_from_slice(&noisy_real);
    state.extend_from_slice(&noisy_imag);
    let mut condition = Vec::with_capacity(2 * PLANE_COUNT);
    condition.extend_from_slice(&condition_real);
    condition.extend_from_slice(&condition_imag);
    let mut output = vec![0.0f32; 2 * PLANE_COUNT];

    // `SgmseModel::from_gguf` is the authenticated role/shape/source gate;
    // the wrapper's GGUF SHA-256 check authenticates the file identity before
    // this call is reached.
    let file = vokra_mmap::open_gguf(&gguf_path).expect("open authenticated SGMSE GGUF");
    let model = SgmseModel::from_gguf(&file).expect("bind authenticated SGMSE GGUF");
    model
        .score(&Compute::cpu(), &state, &condition, 0.5, &mut output)
        .expect("run native SGMSE CPU score graph");
    assert!(
        output.iter().all(|value| value.is_finite()),
        "native SGMSE score contains a non-finite value"
    );

    fs::create_dir(&native_dir).expect("create absent native output directory");
    let native_real = native_dir.join("score_real.f32");
    let native_imag = native_dir.join("score_imag.f32");
    write_new_f32(&native_real, &output[..PLANE_COUNT], "native score real")
        .expect("write native score real");
    write_new_f32(&native_imag, &output[PLANE_COUNT..], "native score imag")
        .expect("write native score imag");
    assert_exact_native_output(&native_dir);
    eprintln!(
        "SGMSE native CPU score produced two 256x64 planes at {} (GGUF sha256={gguf_sha256})",
        native_dir.display()
    );
}
