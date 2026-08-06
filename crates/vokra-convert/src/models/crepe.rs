//! CREPE (Kim et al. 2018) checkpoint → GGUF conversion (M5 gap follow-up).
//!
//! # Input
//!
//! A **prepared** safetensors checkpoint + a JSON config side-car, both
//! produced offline by `tools/parity/keras_h5_to_safetensors.py` from the
//! upstream `crepe/model-{tiny,small,medium,large,full}.h5` release
//! (Keras / TensorFlow never enters the converter — zero-dep, NFR-DS-02;
//! `dac_prepare_checkpoint.py` / `kokoro_prepare_checkpoint.py`
//! precedent):
//!
//! - safetensors: 6 conv blocks × `{weight, bias, bn.gamma, bn.beta,
//!   bn.moving_mean, bn.moving_variance}` + final `{classifier.weight,
//!   classifier.bias}` — **44 tensors total**, all F32 (Keras stores
//!   weights in f32 native; BF16 is not a Keras dtype).
//!   - Conv weights arrive as `[c_out, c_in, kh, 1]` **already permuted**
//!     from Keras' `[kh, 1, c_in, c_out]` layout (the prepare script
//!     owns that permute — the Rust converter does not resort to
//!     numeric axis assumptions).
//! - config JSON: `{"capacity": "<tiny|small|medium|large|full>",
//!   "hop": <u32>, "fmin": <f32>, "fmax": <f32>}` — the two f32 fields
//!   are informational (search-grid bounds; the classifier has fixed
//!   360 bins spanning `1997.379...` cents to `1997.379 + 7180` cents).
//!
//! # What is written
//!
//! 1. **All 44 upstream tensors pass-through verbatim** (F32 → F32 —
//!    exact-shape checked against the capacity-derived per-block filter
//!    counts, mismatch = hard `ConvertError::Parse` per FR-EX-08).
//! 2. `vokra.f0.crepe.{capacity,hop,fmin,fmax}` metadata — the runtime
//!    [`CREPE::from_gguf`](vokra_models::f0::crepe::CREPE::from_gguf)
//!    reads these into [`CrepeConfig`](vokra_models::f0::crepe::CrepeConfig).
//! 3. `vokra.provenance.*`: `model_id = "crepe"` → `Permissive` (MIT).
//!
//! # License
//!
//! Both code and weights ship **MIT** end-to-end (upstream
//! `github.com/marl/crepe/main/LICENSE.txt`, "MIT License / Copyright (c)
//! 2018 Jong Wook Kim et al.", fetched 2026-07-30 — CLAUDE.md
//! 「ハルシネーション厳禁」). MIT is a `Permissive` license class — same
//! commercial verdict as apache-2.0 (no runtime-side attribution
//! obligation).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value.
pub const ARCH: &str = "crepe";
/// `vokra.model.name` prefix (the full name gets the capacity tag appended:
/// `"crepe (tiny)"` / `"crepe (full)"` etc).
const NAME_PREFIX: &str = "crepe";
/// `vokra.model.category` value.
const CATEGORY: &str = "f0";
/// Ad-hoc metadata key for the model category (mirror of `emotion2vec.rs`).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// `vokra.provenance.upstream_hf` value (source repo — GitHub, not HF).
pub const UPSTREAM_SOURCE: &str = "marl/crepe (github)";
/// Canonical weight license SPDX.
pub const DEFAULT_LICENSE: &str = "mit";

/// GGUF metadata keys (mirror of the runtime module — kept private to the
/// converter so the schema evolves through the runtime's own const
/// declarations, not by racing).
const KEY_CAPACITY: &str = "vokra.f0.crepe.capacity";
const KEY_HOP: &str = "vokra.f0.crepe.hop";
const KEY_FMIN: &str = "vokra.f0.crepe.fmin";
const KEY_FMAX: &str = "vokra.f0.crepe.fmax";

/// Per-block filter multipliers (upstream `[32, 4, 4, 4, 8, 16]`).
const FILTER_MULT: [usize; 6] = [32, 4, 4, 4, 8, 16];
/// Per-block kernel widths along the freq axis (upstream `[512, 64, …]`).
const KERNEL_WIDTH: [usize; 6] = [512, 64, 64, 64, 64, 64];
/// Number of pitch classes at the classifier output (upstream fixed 360).
const N_BINS: usize = 360;

/// The capacity discriminator (mirrors the runtime's `CapacityFactor` — kept
/// private here so a schema evolution goes through the runtime's own
/// const declarations, not a duplicated public enum in the converter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Capacity {
    Tiny,
    Small,
    Medium,
    Large,
    Full,
}

impl Capacity {
    fn multiplier(self) -> usize {
        match self {
            Self::Tiny => 4,
            Self::Small => 8,
            Self::Medium => 16,
            Self::Large => 24,
            Self::Full => 32,
        }
    }
    fn as_tag(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Full => "full",
        }
    }
    fn from_tag(tag: &str) -> Option<Self> {
        match tag.trim().to_ascii_lowercase().as_str() {
            "tiny" => Some(Self::Tiny),
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

/// Parsed CREPE config side-car.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CrepeConvertConfig {
    capacity: Capacity,
    hop: u32,
    fmin: f32,
    fmax: f32,
}

impl CrepeConvertConfig {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
        let capacity_str = root
            .get("capacity")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                ConvertError::Parse(
                    "crepe config: required string field `capacity` missing (one of \
                 tiny/small/medium/large/full; the keras_h5_to_safetensors.py side-car \
                 emits it from the .h5 filename)"
                        .to_owned(),
                )
            })?;
        let capacity = Capacity::from_tag(capacity_str).ok_or_else(|| {
            ConvertError::Parse(format!(
                "crepe config: `capacity = \"{capacity_str}\"` is not one of tiny/small/medium/large/full"
            ))
        })?;
        // Optional integer field: default = 160 (upstream `step_size=10` @ 16 kHz).
        let hop = match root.get("hop") {
            Some(v) => u32::try_from(v.as_u64().ok_or_else(|| {
                ConvertError::Parse("crepe config: `hop` must be a non-negative integer".to_owned())
            })?)
            .map_err(|_| ConvertError::Parse("crepe config: `hop` overflows u32".to_owned()))?,
            None => 160,
        };
        // Optional f32 fields: defaults match the runtime's `DEFAULT_FMIN` /
        // `DEFAULT_FMAX`. Encoded as JSON numbers → parsed as u64 (integer)
        // is rejected loudly — the informational Hz bounds are meant to be
        // written as floats, not truncated.
        let fmin = read_opt_f32(&root, "fmin")?.unwrap_or(50.0);
        let fmax = read_opt_f32(&root, "fmax")?.unwrap_or(1100.0);
        Ok(Self {
            capacity,
            hop,
            fmin,
            fmax,
        })
    }

    /// Per-block filter counts (`FILTER_MULT[i] * capacity.multiplier()`).
    fn filters(&self) -> [usize; 6] {
        let m = self.capacity.multiplier();
        [
            FILTER_MULT[0] * m,
            FILTER_MULT[1] * m,
            FILTER_MULT[2] * m,
            FILTER_MULT[3] * m,
            FILTER_MULT[4] * m,
            FILTER_MULT[5] * m,
        ]
    }

    /// Flat vector length going into the final Dense (mirror of
    /// `vokra_models::f0::crepe::CrepeConfig::flat_len`).
    fn flat_len(&self) -> usize {
        4 * self.filters()[5]
    }
}

/// Conversion report.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct CrepeReport {
    /// F32 tensors written verbatim (BF16 is unreachable here because the
    /// Keras export is always F32).
    pub(crate) written: usize,
    /// Total tensors read from the safetensors header (`written` +
    /// anything that failed a shape check — none reach here in the happy
    /// path since a shape mismatch is a hard error).
    pub(crate) read: usize,
    /// Capacity chosen at conversion time (embedded in the summary note
    /// so a CLI operator sees which size they built without opening the
    /// GGUF).
    pub(crate) capacity: &'static str,
}

/// Converts a prepared CREPE safetensors buffer + config side-car into a
/// populated GGUF builder.
///
/// The 44-tensor mapping is total: every declared tensor must exist with
/// exactly the dims the config implies, and no upstream tensor may be
/// left over at the end (a stray tensor is a hard error rather than a
/// silent drop — the same FR-EX-08 posture as `models::utmos::convert`).
pub(crate) fn convert(
    bytes: Vec<u8>,
    config: &CrepeConvertConfig,
) -> Result<(GgufBuilder, CrepeReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let filters = config.filters();
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let name = format!("{NAME_PREFIX} ({})", config.capacity.as_tag());
    b.add_string(chunks::KEY_MODEL_NAME, &name);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_CAPACITY, config.capacity.as_tag());
    b.add_u32(KEY_HOP, config.hop);
    b.add_f32(KEY_FMIN, config.fmin);
    b.add_f32(KEY_FMAX, config.fmax);

    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE,
        Some("crepe"),
        Some(UPSTREAM_SOURCE),
    );

    let mut report = CrepeReport {
        capacity: config.capacity.as_tag(),
        ..Default::default()
    };

    // Track which upstream tensors we consumed so a stray is a hard error
    // (the "no silent drop" contract — mirrors `models::utmos::convert`).
    let mut consumed: Vec<bool> = vec![false; st.tensors().len()];
    let name_to_idx: std::collections::HashMap<&str, usize> = st
        .tensors()
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    report.read = st.tensors().len();

    let mut c_in_running = 1usize;
    for (bi, (&filt, &kh)) in filters.iter().zip(KERNEL_WIDTH.iter()).enumerate() {
        let idx = bi + 1;
        let expect_shape_w: Vec<u64> = vec![filt as u64, c_in_running as u64, kh as u64, 1];
        let expect_shape_1d: Vec<u64> = vec![filt as u64];
        for (name, expect) in [
            (format!("conv{idx}.weight"), &expect_shape_w),
            (format!("conv{idx}.bias"), &expect_shape_1d),
            (format!("conv{idx}.bn.gamma"), &expect_shape_1d),
            (format!("conv{idx}.bn.beta"), &expect_shape_1d),
            (format!("conv{idx}.bn.moving_mean"), &expect_shape_1d),
            (format!("conv{idx}.bn.moving_variance"), &expect_shape_1d),
        ] {
            let i = name_to_idx.get(name.as_str()).copied().ok_or_else(|| {
                ConvertError::Parse(format!(
                    "crepe: required tensor `{name}` not found — not a prepared CREPE checkpoint \
                     (run tools/parity/keras_h5_to_safetensors.py first)"
                ))
            })?;
            let t = &st.tensors()[i];
            if t.dtype != GgmlType::F32 {
                return Err(ConvertError::Parse(format!(
                    "crepe: tensor `{name}` must be F32, got {:?}",
                    t.dtype
                )));
            }
            if &t.shape != expect {
                return Err(ConvertError::Parse(format!(
                    "crepe: tensor `{name}` shape {:?} != expected {:?} (capacity={}, block={})",
                    t.shape,
                    expect,
                    config.capacity.as_tag(),
                    idx
                )));
            }
            b.add_tensor(&name, t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec())?;
            report.written += 1;
            consumed[i] = true;
        }
        c_in_running = filt;
    }

    // Final Dense classifier (`Dense(360, activation='sigmoid')`).
    let flat = config.flat_len();
    let expect_cw: Vec<u64> = vec![N_BINS as u64, flat as u64];
    let expect_cb: Vec<u64> = vec![N_BINS as u64];
    for (name, expect) in [
        ("classifier.weight", &expect_cw),
        ("classifier.bias", &expect_cb),
    ] {
        let i = name_to_idx.get(name).copied().ok_or_else(|| {
            ConvertError::Parse(format!(
                "crepe: required tensor `{name}` not found — the .h5 export must include the \
                 final Dense(360) classifier"
            ))
        })?;
        let t = &st.tensors()[i];
        if t.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "crepe: tensor `{name}` must be F32, got {:?}",
                t.dtype
            )));
        }
        if &t.shape != expect {
            return Err(ConvertError::Parse(format!(
                "crepe: tensor `{name}` shape {:?} != expected {:?}",
                t.shape, expect
            )));
        }
        b.add_tensor(name, t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec())?;
        report.written += 1;
        consumed[i] = true;
    }

    // Any leftover upstream tensor is a hard error (no silent drop —
    // FR-EX-08).
    let strays: Vec<&str> = consumed
        .iter()
        .enumerate()
        .filter(|&(_, &c)| !c)
        .map(|(i, _)| st.tensors()[i].name.as_str())
        .collect();
    if !strays.is_empty() {
        return Err(ConvertError::Parse(format!(
            "crepe: {} upstream tensor(s) not consumed by the converter: {}",
            strays.len(),
            strays.join(", ")
        )));
    }

    Ok((b, report))
}

fn read_opt_f32(root: &JsonValue, key: &str) -> Result<Option<f32>, ConvertError> {
    match root.get(key) {
        Some(JsonValue::Int(i)) => Ok(Some(*i as f32)),
        Some(JsonValue::Float(f)) => Ok(Some(*f as f32)),
        Some(other) => Err(ConvertError::Parse(format!(
            "crepe config: `{key}` must be a number, got {other:?}"
        ))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Encode an f32 slice into little-endian bytes (mirror of the
    /// runtime-side helper).
    fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Build a synthetic safetensors buffer carrying every tensor CREPE
    /// expects at the given capacity (identity BN + zero classifier
    /// weights + a small biased classifier bias so a downstream forward
    /// still evaluates to a defined argmax).
    fn synthetic_crepe_safetensors(capacity: Capacity) -> Vec<u8> {
        let filters = {
            let m = capacity.multiplier();
            [
                FILTER_MULT[0] * m,
                FILTER_MULT[1] * m,
                FILTER_MULT[2] * m,
                FILTER_MULT[3] * m,
                FILTER_MULT[4] * m,
                FILTER_MULT[5] * m,
            ]
        };
        let flat = 4 * filters[5];

        // Emit tensors + build header JSON in the order safetensors likes:
        // one map from name → {dtype, shape, data_offsets}.
        let mut c_in = 1usize;
        let mut entries: Vec<(String, String, Vec<u64>, Vec<u8>)> = Vec::new();
        for (bi, (&filt, &kh)) in filters.iter().zip(KERNEL_WIDTH.iter()).enumerate() {
            let idx = bi + 1;
            let w_len = filt * c_in * kh;
            let w = vec![0.0f32; w_len];
            entries.push((
                format!("conv{idx}.weight"),
                "F32".to_owned(),
                vec![filt as u64, c_in as u64, kh as u64, 1],
                f32_to_le_bytes(&w),
            ));
            entries.push((
                format!("conv{idx}.bias"),
                "F32".to_owned(),
                vec![filt as u64],
                f32_to_le_bytes(&vec![0.0; filt]),
            ));
            entries.push((
                format!("conv{idx}.bn.gamma"),
                "F32".to_owned(),
                vec![filt as u64],
                f32_to_le_bytes(&vec![1.0; filt]),
            ));
            entries.push((
                format!("conv{idx}.bn.beta"),
                "F32".to_owned(),
                vec![filt as u64],
                f32_to_le_bytes(&vec![0.0; filt]),
            ));
            entries.push((
                format!("conv{idx}.bn.moving_mean"),
                "F32".to_owned(),
                vec![filt as u64],
                f32_to_le_bytes(&vec![0.0; filt]),
            ));
            entries.push((
                format!("conv{idx}.bn.moving_variance"),
                "F32".to_owned(),
                vec![filt as u64],
                f32_to_le_bytes(&vec![1.0; filt]),
            ));
            c_in = filt;
        }
        entries.push((
            "classifier.weight".to_owned(),
            "F32".to_owned(),
            vec![N_BINS as u64, flat as u64],
            f32_to_le_bytes(&vec![0.0; N_BINS * flat]),
        ));
        let mut cbias = vec![0.0f32; N_BINS];
        cbias[42] = 1.0; // biased argmax for downstream determinism
        entries.push((
            "classifier.bias".to_owned(),
            "F32".to_owned(),
            vec![N_BINS as u64],
            f32_to_le_bytes(&cbias),
        ));

        let mut cursor: usize = 0;
        let mut header_parts: Vec<String> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in &entries {
            let start = cursor;
            let end = start + bytes.len();
            cursor = end;
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header_parts.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#,
            ));
            payload.extend_from_slice(bytes);
        }
        let header = format!("{{{}}}", header_parts.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn parse_config_defaults() {
        let json = br#"{"capacity":"tiny"}"#.to_vec();
        let cfg = CrepeConvertConfig::parse(&json).expect("parse");
        assert_eq!(cfg.capacity, Capacity::Tiny);
        assert_eq!(cfg.hop, 160);
        assert!((cfg.fmin - 50.0).abs() < 1e-6);
        assert!((cfg.fmax - 1100.0).abs() < 1e-6);
    }

    #[test]
    fn parse_config_all_fields() {
        let json = br#"{"capacity":"full","hop":160,"fmin":40.5,"fmax":1200.0}"#.to_vec();
        let cfg = CrepeConvertConfig::parse(&json).expect("parse");
        assert_eq!(cfg.capacity, Capacity::Full);
        assert_eq!(cfg.hop, 160);
        assert!((cfg.fmin - 40.5).abs() < 1e-6);
        assert!((cfg.fmax - 1200.0).abs() < 1e-6);
    }

    #[test]
    fn parse_config_rejects_bad_capacity() {
        let json = br#"{"capacity":"colossal"}"#.to_vec();
        let err = CrepeConvertConfig::parse(&json).expect_err("bad tag");
        assert!(matches!(err, ConvertError::Parse(_)));
    }

    #[test]
    fn convert_tiny_round_trips_shape_pins() {
        let cfg = CrepeConvertConfig {
            capacity: Capacity::Tiny,
            hop: 160,
            fmin: 50.0,
            fmax: 1100.0,
        };
        let st_bytes = synthetic_crepe_safetensors(Capacity::Tiny);
        let (builder, report) = convert(st_bytes, &cfg).expect("convert");
        // 6 blocks × 6 tensors + 2 classifier tensors = 38 tensors.
        assert_eq!(report.written, 6 * 6 + 2);
        assert_eq!(report.read, 6 * 6 + 2);
        assert_eq!(report.capacity, "tiny");

        // Round-trip through the GGUF byte layer to catch any dtype /
        // shape mismatch (mirror of DAC's converter self-test).
        let out = builder.to_bytes().expect("serialize");
        let path = std::env::temp_dir().join(format!(
            "vokra-crepe-convert-roundtrip-{}.gguf",
            std::process::id(),
        ));
        std::fs::write(&path, &out).expect("write");
        let gguf = GgufFile::open(&path).expect("read back");
        // Metadata keys are present + typed as declared.
        assert_eq!(
            gguf.get(KEY_CAPACITY).and_then(|v| v.as_str()),
            Some("tiny")
        );
        assert_eq!(gguf.get(KEY_HOP).and_then(|v| v.as_u64()), Some(160));
        // One block-1 weight tensor: `[c_out=128, c_in=1, kh=512, 1]`.
        let t = gguf.tensor_info("conv1.weight").expect("conv1.weight");
        assert_eq!(t.dtype, GgmlType::F32);
        assert_eq!(t.dimensions, vec![128, 1, 512, 1]);
        // Classifier bias: `[360]`.
        let cb = gguf
            .tensor_info("classifier.bias")
            .expect("classifier.bias");
        assert_eq!(cb.dimensions, vec![360]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn convert_rejects_missing_tensor() {
        let cfg = CrepeConvertConfig {
            capacity: Capacity::Tiny,
            hop: 160,
            fmin: 50.0,
            fmax: 1100.0,
        };
        // Header with only the metadata but no tensors.
        let header = "{}";
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        let err = convert(buf, &cfg).expect_err("must reject empty");
        match err {
            ConvertError::Parse(msg) => assert!(msg.contains("required tensor")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn convert_rejects_stray_tensor() {
        let cfg = CrepeConvertConfig {
            capacity: Capacity::Tiny,
            hop: 160,
            fmin: 50.0,
            fmax: 1100.0,
        };
        let mut st_bytes = synthetic_crepe_safetensors(Capacity::Tiny);
        // Splice an extra tensor into the header by shifting the payload
        // is fragile; simplest: build a fresh safetensors with one extra
        // stray. Instead, re-decode + rewrite one extra key into the
        // header before the closing brace.
        let hdr_len = u64::from_le_bytes(st_bytes[..8].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&st_bytes[8..8 + hdr_len])
            .unwrap()
            .to_owned();
        assert!(hdr.ends_with('}'));
        // Compute the new stray tensor's payload placement: append 4 zero
        // bytes at the very end (a 1-element f32) and rewrite offsets.
        let old_payload_start = 8 + hdr_len;
        let old_payload_len = st_bytes.len() - old_payload_start;
        let stray_offset = old_payload_len;
        let stray_end = stray_offset + 4;
        let stray_entry = format!(
            r#","stray_tensor":{{"dtype":"F32","shape":[1],"data_offsets":[{stray_offset},{stray_end}]}}"#
        );
        let new_hdr = format!("{}{}}}", &hdr[..hdr.len() - 1], stray_entry);
        // Rebuild buffer.
        let mut out = Vec::new();
        out.extend_from_slice(&(new_hdr.len() as u64).to_le_bytes());
        out.extend_from_slice(new_hdr.as_bytes());
        out.extend_from_slice(&st_bytes[old_payload_start..]);
        out.extend_from_slice(&[0u8; 4]);
        st_bytes = out;

        let err = convert(st_bytes, &cfg).expect_err("stray must fail");
        match err {
            ConvertError::Parse(msg) => {
                assert!(msg.contains("not consumed"), "unexpected msg: {msg}")
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }
}
