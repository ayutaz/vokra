//! **CLAP** (LAION contrastive language-audio pretraining).
//!
//! Conversion is intentionally inspection-only. The current model-free
//! reference tree proves the official processor/config contract, but it does
//! not authenticate a real checkpoint's state-dict roles, shapes, or native
//! preprocessing/forward semantics. Emitting a GGUF from arbitrary
//! safetensors would therefore stamp provenance that has not been proved.
//! Keep this boundary fail-closed until the owner-reviewed VAST evidence
//! supplies the complete manifest and dependency/license closure.

#[cfg(test)]
use std::io::Write;
#[cfg(test)]
use std::path::Component;
use std::path::Path;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ConvertError;

/// Stable converter architecture tag for the CLAP two-tower model family.
#[allow(dead_code)]
pub const ARCH: &str = "clap";
/// Canonical upstream model name used by the inspection contract.
#[allow(dead_code)]
pub const NAME: &str = "clap-htsat-fused";
/// Classification category for the audio-text embedding model surface.
#[allow(dead_code)]
pub const CATEGORY: &str = "classification";
/// Canonical upstream Hugging Face repository used by the inspection tools.
#[allow(dead_code)]
pub const UPSTREAM_HF: &str = "laion/clap-htsat-fused";
/// Immutable upstream revision used by the model-free reference contract.
#[allow(dead_code)]
pub const UPSTREAM_REVISION: &str = "365dea6ef167def6676140ed93bbc43f84dabb28";
/// Upstream-declared SPDX candidate. This is not owner approval and is not
/// stamped into an artifact while conversion remains inspection-only.
#[allow(dead_code)]
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

pub const INSPECTION_ONLY_REASON: &str = concat!(
    "CLAP HTSAT-fused conversion is INSPECTION_ONLY: the authenticated VAST ",
    "checkpoint tensor manifest, native preprocessing/forward contract, and ",
    "dependency/license closure are unavailable; refusing to emit GGUF from ",
    "arbitrary safetensors"
);
#[cfg(test)]
static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClapReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_clap_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ClapReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(INSPECTION_ONLY_REASON.to_owned()))
}

#[cfg(test)]
fn reject_unsafe_path(path: &Path, label: &str) -> Result<(), ConvertError> {
    let raw = path.to_string_lossy();
    #[cfg(windows)]
    let has_lexical_dot = raw
        .split(['/', '\\'])
        .any(|part| matches!(part, "." | ".."));
    #[cfg(not(windows))]
    let has_lexical_dot = raw.split('/').any(|part| matches!(part, "." | ".."));
    if has_lexical_dot
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(ConvertError::Usage(format!(
            "CLAP {label} must not contain lexical dot components"
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
                "CLAP {label} has symlink ancestry: {}",
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
fn write_no_clobber(destination: &Path, payload: &[u8]) -> Result<(), ConvertError> {
    validate_publish_path(destination)?;
    ensure_file_identity_support()?;
    let parent = destination.parent().expect("validated output parent");
    let name = destination
        .file_name()
        .ok_or_else(|| ConvertError::Usage("CLAP output must name a file".to_owned()))?;
    for _ in 0..32 {
        let sequence = OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{}.tmp-{}-{}",
            name.to_string_lossy(),
            std::process::id(),
            sequence
        ));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(ConvertError::Io(error)),
        };
        if let Err(error) = file.write_all(payload).and_then(|_| file.sync_all()) {
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Io(error));
        }
        if let Err(error) = validate_publish_path(destination) {
            remove_owned_temp(&file, &temporary);
            return Err(error);
        }
        if !same_file_identity(&file, &temporary) {
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Usage(
                "CLAP temporary output was replaced before publication".to_owned(),
            ));
        }
        if let Err(error) = std::fs::hard_link(&temporary, destination) {
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Io(error));
        }
        if !same_file_identity(&file, destination) {
            // Do not remove the destination: it may be an unowned file
            // linked by a replacement race in the verify→link window.
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Usage(
                "CLAP published output failed regular-file identity verification".to_owned(),
            ));
        }
        // The final link is already published.  Cleanup is best-effort and
        // identity-guarded so a replacement temp is never unlinked and a
        // cleanup failure cannot turn a successful publication into Err.
        remove_owned_temp(&file, &temporary);
        return Ok(());
    }
    Err(ConvertError::Usage(
        "CLAP exhausted temporary output candidates".to_owned(),
    ))
}

#[cfg(test)]
fn ensure_file_identity_support() -> Result<(), ConvertError> {
    #[cfg(unix)]
    {
        Ok(())
    }
    #[cfg(not(unix))]
    {
        Err(ConvertError::Usage(
            "CLAP atomic publication requires a supported file identity API on this platform"
                .to_owned(),
        ))
    }
}

#[cfg(test)]
fn same_file_identity(file: &std::fs::File, path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let Ok(expected) = file.metadata() else {
            return false;
        };
        let Ok(actual) = std::fs::symlink_metadata(path) else {
            return false;
        };
        actual.file_type().is_file()
            && expected.dev() == actual.dev()
            && expected.ino() == actual.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = (file, path);
        false
    }
}

#[cfg(test)]
fn remove_owned_temp(file: &std::fs::File, path: &Path) {
    if same_file_identity(file, path) {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
fn validate_publish_path(destination: &Path) -> Result<(), ConvertError> {
    reject_unsafe_path(destination, "output")?;
    let parent = destination.parent().ok_or_else(|| {
        ConvertError::Usage("CLAP output must have a parent directory".to_owned())
    })?;
    if !parent.is_dir() || parent.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "CLAP output parent must be an existing regular non-symlink directory: {}",
            parent.display()
        )));
    }
    if destination.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "CLAP output must not be a symlink: {}",
            destination.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexical_dot_paths_are_rejected_on_every_platform() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-dot-paths-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        assert!(reject_unsafe_path(&root.join(".").join("artifact.gguf"), "output").is_err());
        assert!(reject_unsafe_path(&root.join("..").join("artifact.gguf"), "output").is_err());
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn conversion_stays_inspection_only_without_authenticated_manifest() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-inspection-only-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let input = root.join("arbitrary.safetensors");
        let output = root.join("artifact.gguf");
        std::fs::write(&input, b"arbitrary unverified payload").expect("input");

        let error = convert_clap_file(&input, &output, Some("not-an-approved-license"))
            .expect_err("unverified CLAP conversion must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("INSPECTION_ONLY"),
            "unexpected gate: {message}"
        );
        assert!(
            message.contains("tensor manifest"),
            "unexpected gate: {message}"
        );
        assert!(
            !output.exists(),
            "inspection-only conversion created output"
        );

        std::fs::write(&output, b"existing output").expect("existing output");
        let error = convert_clap_file(&input, &output, None)
            .expect_err("inspection-only conversion must remain rejected");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert_eq!(
            std::fs::read(&output).expect("read preserved output"),
            b"existing output"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn output_publication_is_no_clobber_and_path_bounded() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-output-boundary-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let output = root.join("artifact.gguf");
        write_no_clobber(&output, b"first").expect("first publication");
        let second = write_no_clobber(&output, b"second").expect_err("clobber accepted");
        assert!(matches!(
            second,
            ConvertError::Io(ref error)
                if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(
            std::fs::read(&output).expect("published artifact"),
            b"first"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let real = root.join("real");
            let linked = root.join("linked");
            std::fs::create_dir(&real).expect("real");
            symlink(&real, &linked).expect("symlink");
            assert!(validate_publish_path(&linked.join("artifact.gguf")).is_err());
        }
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(not(unix))]
    #[test]
    fn non_unix_publication_fails_closed_without_touching_output() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-unsupported-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let output = root.join("artifact.gguf");

        let input = root.join("input.safetensors");
        let error = convert_clap_file(&input, &output, None).expect_err("unsupported conversion");
        assert!(matches!(error, ConvertError::Usage(_)));
        assert!(!output.exists(), "unsupported publication created output");

        std::fs::write(&input, b"not a safetensors file").expect("input");
        std::fs::write(&output, b"existing").expect("existing output");
        let error = convert_clap_file(&input, &output, None).expect_err("unsupported conversion");
        assert!(matches!(error, ConvertError::Usage(_)));
        assert_eq!(
            std::fs::read(&output).expect("read preserved output"),
            b"existing"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn replacement_temp_is_not_removed_by_owned_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-temp-race-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let temporary = root.join("artifact.tmp");
        std::fs::write(&temporary, b"owned").expect("temporary");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&temporary)
            .expect("open owned temp");
        let destination = root.join("destination");
        std::fs::hard_link(&temporary, &destination).expect("publish owned temp");
        assert!(same_file_identity(&file, &destination));
        std::fs::remove_file(&destination).expect("remove destination");
        std::fs::write(&destination, b"destination replacement").expect("destination replacement");
        assert!(!same_file_identity(&file, &destination));
        std::fs::remove_file(&temporary).expect("remove owned path");
        std::fs::write(&temporary, b"replacement").expect("replacement");
        assert!(!same_file_identity(&file, &temporary));
        remove_owned_temp(&file, &temporary);
        assert_eq!(
            std::fs::read(&temporary).expect("replacement remains"),
            b"replacement"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
