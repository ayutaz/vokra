//! CoreML compiled-artifact contract for declared delegate submodels.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use vokra_core::{Result, VokraError};

/// Fixed CoreML feature name for Whisper log-mel input.
pub const WHISPER_ENCODER_INPUT: &str = "log_mel";
/// Fixed CoreML feature name for Whisper encoder hidden-state output.
pub const WHISPER_ENCODER_OUTPUT: &str = "encoder_hidden";

const MANIFEST_FORMAT: &str = "vokra-coreml-sidecar-v1";
const COMPILED_MODEL_NAME: &str = "whisper-encoder.mlmodelc";
const MANIFEST_NAME: &str = "manifest.txt";
const MINIMUM_DEPLOYMENT_TARGET: &str = "macOS14";
const COREMLTOOLS_VERSION: &str = "9.0";
const ACCEPTED_ARCHS: &[&str] = &[
    "whisper",
    "crisper-whisper",
    "distil-whisper",
    "kotoba-whisper",
];

/// Internal arithmetic precision requested when the CoreML model was built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMlComputePrecision {
    /// FP16 arithmetic, the production ANE path.
    Float16,
    /// FP32 arithmetic, retained as a diagnostic/parity build option.
    Float32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactBinding {
    model_arch: String,
    compute_precision: CoreMlComputePrecision,
    source_gguf_sha256: String,
    compiled_tree_sha256: String,
}

/// A compiled CoreML artifact plus its shape contract.
///
/// Vokra loads only `.mlmodelc` directories at runtime. Portable
/// `.mlpackage` generation and compilation belong to the offline converter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMlArtifact {
    compiled_model: PathBuf,
    input_shape: [usize; 3],
    output_shape: [usize; 3],
    binding: Option<ArtifactBinding>,
}

impl CoreMlArtifact {
    /// Declares a complete Whisper encoder artifact.
    ///
    /// Shapes include the batch dimension and must be `[1, n_mels, n_frames]`
    /// for the input and `[1, n_audio_ctx, d_model]` for the output.
    pub fn whisper_encoder(
        compiled_model: impl Into<PathBuf>,
        input_shape: [usize; 3],
        output_shape: [usize; 3],
    ) -> Result<Self> {
        let compiled_model = compiled_model.into();
        if compiled_model.extension().and_then(|v| v.to_str()) != Some("mlmodelc") {
            return Err(VokraError::InvalidArgument(format!(
                "CoreML runtime artifact `{}` must be a compiled .mlmodelc directory; generate \
                 the portable .mlpackage offline and compile it with coremlcompiler",
                compiled_model.display()
            )));
        }
        if input_shape.into_iter().any(|axis| axis == 0)
            || output_shape.into_iter().any(|axis| axis == 0)
        {
            return Err(VokraError::InvalidArgument(format!(
                "CoreML Whisper encoder shapes must have no zero axis (input {input_shape:?}, \
                 output {output_shape:?})"
            )));
        }
        if input_shape[0] != 1 || output_shape[0] != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "CoreML Whisper encoder currently requires batch 1 (input {input_shape:?}, \
                 output {output_shape:?})"
            )));
        }
        Ok(Self {
            compiled_model,
            input_shape,
            output_shape,
            binding: None,
        })
    }

    /// Resolves and verifies `<source.gguf>.coreml/manifest.txt`.
    ///
    /// The manifest binds the sidecar to the complete source GGUF and compiled
    /// `.mlmodelc` tree by SHA-256, and pins feature names, shapes, converter
    /// version, deployment target, and arithmetic precision. Any drift fails
    /// before CoreML is loaded; this function never substitutes a CPU path.
    pub fn from_whisper_sidecar(
        source_gguf: impl AsRef<Path>,
        expected_model_arch: &str,
        expected_input_shape: [usize; 3],
        expected_output_shape: [usize; 3],
    ) -> Result<Self> {
        let source_gguf = source_gguf.as_ref();
        if !source_gguf.is_file() {
            return Err(load_error(format!(
                "source GGUF is missing or is not a regular file: {}",
                source_gguf.display()
            )));
        }
        let sidecar = sidecar_path(source_gguf);
        let manifest_path = sidecar.join(MANIFEST_NAME);
        let metadata = std::fs::metadata(&manifest_path).map_err(|error| {
            load_error(format!(
                "cannot stat manifest `{}`: {error}; generate it with tools/coreml/build_whisper_encoder.sh",
                manifest_path.display()
            ))
        })?;
        if !metadata.is_file() || metadata.len() > 16 * 1024 {
            return Err(load_error(format!(
                "manifest `{}` must be a regular UTF-8 file no larger than 16 KiB",
                manifest_path.display()
            )));
        }
        let text = std::fs::read_to_string(&manifest_path).map_err(|error| {
            load_error(format!(
                "cannot read manifest `{}`: {error}",
                manifest_path.display()
            ))
        })?;
        let values = parse_manifest(&text)?;

        require(&values, "format", MANIFEST_FORMAT)?;
        require(&values, "submodel", "whisper_encoder")?;
        require(&values, "compiled_model", COMPILED_MODEL_NAME)?;
        require(&values, "input_name", WHISPER_ENCODER_INPUT)?;
        require(&values, "output_name", WHISPER_ENCODER_OUTPUT)?;
        require(
            &values,
            "minimum_deployment_target",
            MINIMUM_DEPLOYMENT_TARGET,
        )?;
        require(&values, "coremltools_version", COREMLTOOLS_VERSION)?;

        let model_arch = value(&values, "model_arch")?;
        if !ACCEPTED_ARCHS.contains(&model_arch) {
            return Err(load_error(format!(
                "manifest model_arch `{model_arch}` is not one of {ACCEPTED_ARCHS:?}"
            )));
        }
        if model_arch != expected_model_arch {
            return Err(load_error(format!(
                "manifest model_arch `{model_arch}` != source GGUF binder arch `{expected_model_arch}`"
            )));
        }
        let input_shape = parse_shape(value(&values, "input_shape")?, "input_shape")?;
        let output_shape = parse_shape(value(&values, "output_shape")?, "output_shape")?;
        if input_shape != expected_input_shape {
            return Err(load_error(format!(
                "manifest input_shape {input_shape:?} != runtime Whisper contract {expected_input_shape:?}"
            )));
        }
        if output_shape != expected_output_shape {
            return Err(load_error(format!(
                "manifest output_shape {output_shape:?} != runtime Whisper contract {expected_output_shape:?}"
            )));
        }
        let compute_precision = match value(&values, "compute_precision")? {
            "float16" => CoreMlComputePrecision::Float16,
            "float32" => CoreMlComputePrecision::Float32,
            other => {
                return Err(load_error(format!(
                    "manifest compute_precision `{other}` is unsupported"
                )));
            }
        };

        let expected_source = canonical_digest(value(&values, "source_gguf_sha256")?)?;
        let actual_source = crate::digest::file_sha256(source_gguf).map_err(|error| {
            load_error(format!(
                "cannot hash source GGUF `{}`: {error}",
                source_gguf.display()
            ))
        })?;
        if actual_source != expected_source {
            return Err(load_error(format!(
                "source_gguf_sha256 mismatch for `{}`: manifest {expected_source}, actual {actual_source}",
                source_gguf.display()
            )));
        }

        let compiled_model = sidecar.join(COMPILED_MODEL_NAME);
        let expected_tree = canonical_digest(value(&values, "compiled_tree_sha256")?)?;
        let actual_tree = crate::digest::tree_sha256(&compiled_model).map_err(|error| {
            load_error(format!(
                "cannot hash compiled CoreML tree `{}`: {error}",
                compiled_model.display()
            ))
        })?;
        if actual_tree != expected_tree {
            return Err(load_error(format!(
                "compiled_tree_sha256 mismatch for `{}`: manifest {expected_tree}, actual {actual_tree}",
                compiled_model.display()
            )));
        }

        let mut artifact = Self::whisper_encoder(compiled_model, input_shape, output_shape)?;
        artifact.binding = Some(ArtifactBinding {
            model_arch: model_arch.to_owned(),
            compute_precision,
            source_gguf_sha256: actual_source,
            compiled_tree_sha256: actual_tree,
        });
        Ok(artifact)
    }

    /// Path to the compiled `.mlmodelc` directory.
    pub fn compiled_model(&self) -> &Path {
        &self.compiled_model
    }

    /// Required CoreML input shape `[1, n_mels, n_frames]`.
    pub fn input_shape(&self) -> [usize; 3] {
        self.input_shape
    }

    /// Required CoreML output shape `[1, n_audio_ctx, d_model]`.
    pub fn output_shape(&self) -> [usize; 3] {
        self.output_shape
    }

    /// Model architecture recorded by a verified sidecar manifest.
    pub fn model_arch(&self) -> Option<&str> {
        self.binding
            .as_ref()
            .map(|binding| binding.model_arch.as_str())
    }

    /// Arithmetic precision recorded by a verified sidecar manifest.
    pub fn compute_precision(&self) -> Option<CoreMlComputePrecision> {
        self.binding
            .as_ref()
            .map(|binding| binding.compute_precision)
    }

    /// Rejects artifact precision modes that cannot be a production ANE path.
    ///
    /// FP32 remains supported by the low-level backend for numerical diagnosis,
    /// but the 2026-08-24 M1 placement probe put its entire estimated cost on
    /// CPU. Production ASR therefore accepts only a manifest-verified FP16
    /// artifact. This is a necessary, not sufficient, condition: release
    /// qualification must still run the device-specific 90% placement probe.
    pub fn require_production_ane_precision(&self) -> Result<()> {
        match self.compute_precision() {
            Some(CoreMlComputePrecision::Float16) => Ok(()),
            Some(CoreMlComputePrecision::Float32) => Err(VokraError::UnsupportedOp(
                "CoreML float32 sidecars are diagnostic-only: the M1 probe placed 100% of \
                 estimated compute cost on CPU; production ASR requires a float16 artifact plus \
                 a separate >=90% ANE placement result"
                    .to_owned(),
            )),
            None => Err(load_error(
                "production CoreML ASR requires a manifest-verified precision binding".to_owned(),
            )),
        }
    }

    /// Source GGUF SHA-256 recorded by a verified sidecar manifest.
    pub fn source_gguf_sha256(&self) -> Option<&str> {
        self.binding
            .as_ref()
            .map(|binding| binding.source_gguf_sha256.as_str())
    }

    /// Compiled `.mlmodelc` tree SHA-256 recorded by a verified manifest.
    pub fn compiled_tree_sha256(&self) -> Option<&str> {
        self.binding
            .as_ref()
            .map(|binding| binding.compiled_tree_sha256.as_str())
    }
}

fn sidecar_path(source_gguf: &Path) -> PathBuf {
    let mut value = OsString::from(source_gguf.as_os_str());
    value.push(".coreml");
    PathBuf::from(value)
}

fn load_error(message: String) -> VokraError {
    VokraError::ModelLoad(format!("CoreML Whisper sidecar: {message}"))
}

fn parse_manifest(text: &str) -> Result<BTreeMap<&str, &str>> {
    const KEYS: &[&str] = &[
        "format",
        "submodel",
        "model_arch",
        "source_gguf_sha256",
        "compiled_model",
        "compiled_tree_sha256",
        "input_name",
        "output_name",
        "input_shape",
        "output_shape",
        "compute_precision",
        "minimum_deployment_target",
        "coremltools_version",
    ];
    let mut values = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            return Err(load_error(format!("manifest line {} is empty", index + 1)));
        }
        let (key, value) = line.split_once('=').ok_or_else(|| {
            load_error(format!("manifest line {} has no `=` separator", index + 1))
        })?;
        if !KEYS.contains(&key) {
            return Err(load_error(format!("manifest contains unknown key `{key}`")));
        }
        if value.is_empty() {
            return Err(load_error(format!(
                "manifest key `{key}` has an empty value"
            )));
        }
        if values.insert(key, value).is_some() {
            return Err(load_error(format!("manifest key `{key}` is duplicated")));
        }
    }
    for key in KEYS {
        if !values.contains_key(key) {
            return Err(load_error(format!("manifest is missing key `{key}`")));
        }
    }
    Ok(values)
}

fn value<'a>(values: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str> {
    values
        .get(key)
        .copied()
        .ok_or_else(|| load_error(format!("manifest is missing key `{key}`")))
}

fn require(values: &BTreeMap<&str, &str>, key: &str, expected: &str) -> Result<()> {
    let actual = value(values, key)?;
    if actual != expected {
        return Err(load_error(format!(
            "manifest {key} `{actual}` != required `{expected}`"
        )));
    }
    Ok(())
}

fn parse_shape(raw: &str, key: &str) -> Result<[usize; 3]> {
    let mut axes = raw.split(',');
    let mut shape = [0usize; 3];
    for axis in &mut shape {
        let raw_axis = axes
            .next()
            .ok_or_else(|| load_error(format!("manifest {key} `{raw}` is not rank 3")))?;
        *axis = raw_axis
            .parse::<usize>()
            .map_err(|_| load_error(format!("manifest {key} `{raw}` contains a non-usize axis")))?;
        if *axis == 0 {
            return Err(load_error(format!(
                "manifest {key} `{raw}` contains a zero axis"
            )));
        }
    }
    if axes.next().is_some() {
        return Err(load_error(format!("manifest {key} `{raw}` is not rank 3")));
    }
    Ok(shape)
}

fn canonical_digest(raw: &str) -> Result<&str> {
    if raw.len() != 64
        || !raw
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(load_error(format!(
            "manifest SHA-256 `{raw}` is not canonical lowercase hex"
        )));
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct FixtureDir(PathBuf);

    impl FixtureDir {
        fn new() -> Self {
            let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "vokra-coreml-artifact-test-{}-{serial}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn valid_sidecar() -> (FixtureDir, PathBuf) {
        let fixture = FixtureDir::new();
        let gguf = fixture.0.join("model.gguf");
        std::fs::write(&gguf, b"source-gguf").unwrap();
        let sidecar = PathBuf::from(format!("{}.coreml", gguf.display()));
        let model = sidecar.join("whisper-encoder.mlmodelc");
        std::fs::create_dir_all(model.join("nested")).unwrap();
        std::fs::write(model.join("a.txt"), b"alpha").unwrap();
        std::fs::write(model.join("nested/b.bin"), [0, 1, 2]).unwrap();
        // Both digests were generated independently with Python 3.12's
        // hashlib; the tree framing is documented in tools/coreml.
        std::fs::write(
            sidecar.join("manifest.txt"),
            "format=vokra-coreml-sidecar-v1\n\
             submodel=whisper_encoder\n\
             model_arch=whisper\n\
             source_gguf_sha256=7b6a4380117024649b02f21dca93303cf5b1bff9d00eb42c22543337b4f874c5\n\
             compiled_model=whisper-encoder.mlmodelc\n\
             compiled_tree_sha256=9c137a2d9ebce664d79409abfe05ebfc782a66e6dd091aee17187db9dce53317\n\
             input_name=log_mel\n\
             output_name=encoder_hidden\n\
             input_shape=1,2,4\n\
             output_shape=1,2,4\n\
             compute_precision=float16\n\
             minimum_deployment_target=macOS14\n\
             coremltools_version=9.0\n",
        )
        .unwrap();
        (fixture, gguf)
    }

    #[test]
    fn sidecar_binds_source_tree_names_and_shapes() {
        let (_fixture, gguf) = valid_sidecar();
        let artifact = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect("independently hashed valid sidecar");
        assert_eq!(
            artifact.compiled_model().file_name().unwrap(),
            "whisper-encoder.mlmodelc"
        );
        assert_eq!(artifact.model_arch(), Some("whisper"));
        assert_eq!(
            artifact.compute_precision(),
            Some(CoreMlComputePrecision::Float16)
        );
        assert_eq!(
            artifact.source_gguf_sha256(),
            Some("7b6a4380117024649b02f21dca93303cf5b1bff9d00eb42c22543337b4f874c5")
        );
        assert_eq!(
            artifact.compiled_tree_sha256(),
            Some("9c137a2d9ebce664d79409abfe05ebfc782a66e6dd091aee17187db9dce53317")
        );
        artifact
            .require_production_ane_precision()
            .expect("verified FP16 is eligible for the separately probed ANE path");
    }

    #[test]
    fn production_binding_rejects_fp32_diagnostic_sidecar() {
        let (_fixture, gguf) = valid_sidecar();
        let sidecar = PathBuf::from(format!("{}.coreml", gguf.display()));
        let manifest_path = sidecar.join("manifest.txt");
        let manifest = std::fs::read_to_string(&manifest_path)
            .unwrap()
            .replace("compute_precision=float16", "compute_precision=float32");
        std::fs::write(manifest_path, manifest).unwrap();

        let artifact = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect("FP32 remains loadable by the low-level diagnostic backend");
        assert_eq!(
            artifact.compute_precision(),
            Some(CoreMlComputePrecision::Float32)
        );
        let error = artifact
            .require_production_ane_precision()
            .expect_err("production ASR must not disguise a CPU CoreML run as an ANE delegate");
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(format!("{error}").contains("diagnostic-only"));
    }

    #[test]
    fn sidecar_rejects_source_or_tree_drift_and_shape_mismatch() {
        let (_fixture, gguf) = valid_sidecar();
        let err = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 6], [1, 2, 4])
            .expect_err("shape drift must fail before loading CoreML");
        assert!(matches!(err, VokraError::ModelLoad(_)));
        assert!(format!("{err}").contains("input_shape"));

        std::fs::write(&gguf, b"different-source").unwrap();
        let err = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect_err("source GGUF drift must fail closed");
        assert!(format!("{err}").contains("source_gguf_sha256"));

        std::fs::write(&gguf, b"source-gguf").unwrap();
        let sidecar = PathBuf::from(format!("{}.coreml", gguf.display()));
        std::fs::write(sidecar.join("whisper-encoder.mlmodelc/a.txt"), b"tampered").unwrap();
        let err = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect_err("compiled model tree drift must fail closed");
        assert!(format!("{err}").contains("compiled_tree_sha256"));
    }

    #[test]
    fn verified_artifact_identity_changes_when_bound_source_changes() {
        let (_fixture, gguf) = valid_sidecar();
        let before = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect("initial sidecar");

        std::fs::write(&gguf, b"different-source").unwrap();
        let sidecar = PathBuf::from(format!("{}.coreml", gguf.display()));
        let manifest_path = sidecar.join("manifest.txt");
        let manifest = std::fs::read_to_string(&manifest_path).unwrap().replace(
            "7b6a4380117024649b02f21dca93303cf5b1bff9d00eb42c22543337b4f874c5",
            "82909483ea74f3634a5d50f46d7b5dfbbb2ee25c6c8608cb2467d306e59c6fec",
        );
        std::fs::write(manifest_path, manifest).unwrap();
        let after = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect("regenerated sidecar");

        assert_ne!(
            before, after,
            "the TLS cache key must include manifest content identities"
        );
    }
}
