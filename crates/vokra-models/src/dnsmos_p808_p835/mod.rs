//! DNSMOS P.808 / P.835 (Microsoft `DNS-Challenge/DNSMOS`, MIT) — runtime
//! binder for the `dnsmos` converter arch (2026-08-05).
//!
//! # Runtime layout (loud-partial, RMVPE + openwakeword precedent)
//!
//! ```text
//! PCM (16 kHz mono f32)
//!   -> chunk to 9.01 s windows (INPUT_LENGTH = 144160 samples, zero-pad
//!      the tail)
//!   -> mel front-end
//!        (n_fft=321, hop=160, n_mels=120, sr=16000, Hann-centred,
//!         power_to_db(ref=max), (db+40)/40 normalise) — REAL helper,
//!         reuses `vokra_ops::stft` + `mel_filterbank` (deferred wiring,
//!         gated behind the loud-partial today)
//!   -> per-variant CNN forward             ← **loud-partial**
//!        (transcribed from the ONNX graph's `node` list; the current
//!         sidecar only walks `initializer` so the op sequence
//!         `conv → BN → relu → pool` is not primary-source-derivable
//!         and would be silent-wrong if best-guessed)
//!   -> P.808: 1 scalar; P.835: 3 scalars (SIG, BAK, OVRL)
//!   -> polyfit calibration (personalized-off coeffs from `dnsmos_local.py`)
//!      → per-chunk MOS ∈ [1, 5]
//!   -> mean across chunks
//! ```
//!
//! # Loud-partial classification (design §3)
//!
//! - **Real**: config + mel front-end + polyfit + score-shell + variants
//!   inventory + FR-EX-08 loud-fails.
//! - **Loud-partial**: [`Dnsmos::score_p808`] / [`score_p835`] /
//!   [`score_all`] and the [`MosScorerEngine::score`] impl all return
//!   [`VokraError::UnsupportedOp`] naming (a) the future GGUF metadata
//!   chunk `vokra.dnsmos.{p808,p835}.topology` that will pin the CNN
//!   op-token sequence, and (b) the sidecar to extend
//!   (`tools/parity/dnsmos_prepare_checkpoint.py`).
//!
//! Rationale: `dnsmos_local.py` (MIT) exposes only front-end + polyfit;
//! the CNN backbone is recoverable only by walking the trained ONNX
//! graph's `graph.node` list. The current converter walks
//! `graph.initializer` only, so the op-order is not primary-source-
//! transcribable. Following the RMVPE `extract_real` posture, the
//! surrounding scaffold lands today so a follow-up wave can flip the
//! switch by (i) extending the sidecar to emit
//! `vokra.dnsmos.{variant}.topology` as a u32 op-token array, and (ii)
//! wiring the future `cnn_forward` (routed through
//! [`cnn_forward_loud_partial`] today) against that token stream.
//!
//! # `vokra.dnsmos.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::dnsmos::convert_dnsmos_file`:
//!
//! - `vokra.dnsmos.bundle` (`Array<String>`): canonical order
//!   `["p808", "p835"]` (subset for a partial bundle).
//! - `vokra.dnsmos.sample_rate` (u32): PCM sample rate the model
//!   expects (16 000).
//! - `vokra.dnsmos.p808.checkpoint` (String): upstream `.onnx` filename
//!   (auditability); present iff `p808` in bundle.
//! - `vokra.dnsmos.p835.checkpoint` (String): same for P.835.
//! - `vokra.dnsmos.{p808,p835}.topology` (`Array<u32>`): op-token
//!   sequence for the CNN backbone. Not written by the current sidecar
//!   — flipping this on lands the real forward.

use std::sync::Arc;

use vokra_core::engines::{MosScore, MosScorerEngine};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType};
use vokra_core::{Result, VokraError};

#[cfg(test)]
mod tests;

// ---- arch / provenance constants (mirror of the converter's `pub const`
// surface — same duplication convention every fsmn_vad / silero_vad /
// openwakeword binder uses so the runtime does not add a cross-crate
// dependency edge onto the converter). --------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model dnsmos-p808-p835`.
pub const ARCH: &str = "dnsmos";

/// Expected `vokra.model.name` value written by the DNSMOS converter.
pub const NAME: &str = "dnsmos-p808-p835";

/// Expected `vokra.model.category` value — `"eval"` (MOS predictor
/// tier, sibling of UTMOS in `vokra-eval::metrics::utmos`).
pub const CATEGORY: &str = "eval";

/// Upstream sample rate DNSMOS was trained at (Hz). Both variants share
/// this; a differently-rated GGUF is either mis-configured or a non-
/// canonical fork — fail loud (FR-EX-08).
pub const EXPECTED_SAMPLE_RATE: u32 = 16_000;

/// Fixed 9.01 s input window (144 160 samples at 16 kHz) — the
/// `INPUT_LENGTH` constant transcribed verbatim from
/// `microsoft/DNS-Challenge/DNSMOS/dnsmos_local.py`. Consumed by the
/// (deferred) chunking + mel front-end.
pub const INPUT_LENGTH_SAMPLES: usize = 144_160;

// ---- vokra.dnsmos.* metadata keys (duplication of the converter's
// pub-const surface for the same reason as above) ---------------------

/// GGUF metadata key: model category tag.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// GGUF metadata key: bundle inventory (`Array<String>`).
pub const KEY_DNSMOS_BUNDLE: &str = "vokra.dnsmos.bundle";
/// GGUF metadata key: sample rate (u32 Hz).
pub const KEY_DNSMOS_SAMPLE_RATE: &str = "vokra.dnsmos.sample_rate";
/// GGUF metadata key: P.808 upstream checkpoint filename.
pub const KEY_DNSMOS_P808_CKPT: &str = "vokra.dnsmos.p808.checkpoint";
/// GGUF metadata key: P.835 upstream checkpoint filename.
pub const KEY_DNSMOS_P835_CKPT: &str = "vokra.dnsmos.p835.checkpoint";
/// GGUF metadata key: P.808 CNN op-token sequence (future `Array<u32>` —
/// currently absent; presence flips the CNN forward out of loud-partial
/// via [`cnn_forward_loud_partial`]).
pub const KEY_DNSMOS_P808_TOPOLOGY: &str = "vokra.dnsmos.p808.topology";
/// GGUF metadata key: P.835 CNN op-token sequence (future `Array<u32>` —
/// same role as [`KEY_DNSMOS_P808_TOPOLOGY`] for the P.835 variant).
pub const KEY_DNSMOS_P835_TOPOLOGY: &str = "vokra.dnsmos.p835.topology";

/// Sub-model tag inside a DNSMOS bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnsmosSubmodel {
    /// ITU-T P.808 overall quality predictor (single scalar out).
    P808,
    /// ITU-T P.835 3-way predictor (SIG / BAK / OVRL).
    P835,
}

impl DnsmosSubmodel {
    /// Canonical short name for logs / metadata.
    pub const fn short(&self) -> &'static str {
        match self {
            Self::P808 => "p808",
            Self::P835 => "p835",
        }
    }
    /// The GGUF tensor-name prefix each variant's weights carry
    /// (`"p808."` / `"p835."` — the converter emits initializer names
    /// verbatim under this prefix).
    pub const fn tensor_prefix(&self) -> &'static str {
        match self {
            Self::P808 => "p808.",
            Self::P835 => "p835.",
        }
    }
}

/// DNSMOS runtime config (transcribed verbatim from `vokra.dnsmos.*` at
/// load time; every field is required and validated loudly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnsmosConfig {
    /// Bundle inventory in canonical order (`"p808"` before `"p835"`).
    pub bundle: Vec<String>,
    /// PCM sample rate the model expects (Hz). Must equal
    /// [`EXPECTED_SAMPLE_RATE`].
    pub sample_rate: u32,
    /// Whether the P.808 variant is present in this GGUF (cached from
    /// [`Self::bundle`] for fast dispatch).
    pub has_p808: bool,
    /// Whether the P.835 variant is present in this GGUF.
    pub has_p835: bool,
}

impl DnsmosConfig {
    /// Validates the config loudly (FR-EX-08).
    pub fn validate(&self) -> Result<()> {
        if self.bundle.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: `{KEY_DNSMOS_BUNDLE}` is empty — the GGUF must advertise \
                 at least one of `p808` / `p835` (a bundle with neither is a \
                 conversion bug; the converter refuses to emit one)"
            )));
        }
        if self.sample_rate != EXPECTED_SAMPLE_RATE {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: `{KEY_DNSMOS_SAMPLE_RATE}` = {} — DNSMOS is trained at \
                 {EXPECTED_SAMPLE_RATE} Hz only (upstream `dnsmos_local.py` pins \
                 SAMPLING_RATE = 16000). Resample the audio upstream rather than \
                 emitting a different-rate GGUF (FR-EX-08).",
                self.sample_rate,
            )));
        }
        for v in &self.bundle {
            match v.as_str() {
                "p808" | "p835" => {}
                other => {
                    return Err(VokraError::ModelLoad(format!(
                        "dnsmos: `{KEY_DNSMOS_BUNDLE}` contains unknown variant \
                         `{other}` — expected one of `p808`, `p835`"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Reads config from a parsed GGUF's `vokra.dnsmos.*` chunk group.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let bundle = read_string_array(gguf, KEY_DNSMOS_BUNDLE)?;
        let sample_rate = gguf
            .get(KEY_DNSMOS_SAMPLE_RATE)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "dnsmos GGUF missing required u32 metadata `{KEY_DNSMOS_SAMPLE_RATE}`"
                ))
            })?;
        let sample_rate = u32::try_from(sample_rate).map_err(|_| {
            VokraError::ModelLoad(format!(
                "dnsmos GGUF metadata `{KEY_DNSMOS_SAMPLE_RATE}` = {sample_rate} does not fit in u32"
            ))
        })?;
        let has_p808 = bundle.iter().any(|v| v == "p808");
        let has_p835 = bundle.iter().any(|v| v == "p835");
        let cfg = Self {
            bundle,
            sample_rate,
            has_p808,
            has_p835,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Reads a required `Array<String>` metadata chunk, enforcing element-
/// type (FR-EX-08 — refuse the load rather than silently coerce).
fn read_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "dnsmos GGUF missing required Array<String> metadata `{key}`"
        ))
    })?;
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "dnsmos GGUF metadata `{key}` is not an array (expected Array<String>)"
        ))
    })?;
    if arr.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "dnsmos GGUF metadata `{key}` has element_type {:?}, expected String",
            arr.element_type
        )));
    }
    let mut out = Vec::with_capacity(arr.values.len());
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::String(s) => out.push(s.clone()),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "dnsmos GGUF metadata `{key}[{i}]` is not String (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    Ok(out)
}

/// A single bound sub-model (weight-tensor names + counts). Weights are
/// referenced by name only until the CNN forward wires — the current
/// binder does not preload every tensor into RAM because the forward is
/// loud-partial (see [`cnn_forward_loud_partial`]), and the follow-up
/// wave that lights up the CNN forward will decide the caching shape
/// based on the topology metadata.
#[derive(Debug, Clone)]
pub struct DnsmosBundle {
    /// Which variant this bundle entry represents.
    pub variant: DnsmosSubmodel,
    /// The GGUF tensor names carrying this variant's weights (verbatim
    /// prefixed initializer names, e.g. `"p808.conv1/kernel"`).
    pub tensor_names: Vec<String>,
}

/// DNSMOS session — an immutable shareable bundle plus the config it
/// was bound against. Constructed via [`Self::from_gguf`] or (for tests)
/// [`Self::synthesized`]; scored via [`Self::score_p808`] /
/// [`Self::score_p835`] / [`Self::score_all`].
#[derive(Debug, Clone)]
pub struct Dnsmos {
    cfg: DnsmosConfig,
    bundles: Arc<Vec<DnsmosBundle>>,
    /// Variants string slice built once at bind time so
    /// [`MosScorerEngine::variants`] can return a stable `&[&'static str]`
    /// without allocating per call.
    variants: &'static [&'static str],
}

impl Dnsmos {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Returns [`VokraError::ModelLoad`] if the arch tag is wrong, any
    /// required `vokra.dnsmos.*` chunk is missing, the sample rate is
    /// not `16000`, the bundle is empty, or any advertised variant
    /// carries no tensors under its expected prefix (FR-EX-08 — no
    /// silent partial bundle).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Arch check first — a UTMOS / silero-vad / openwakeword GGUF
        // handed to us by mistake fails with a clear message instead of
        // a downstream "missing tensor".
        match gguf
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
        {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "dnsmos: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "dnsmos: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it)"
                        .to_owned(),
                ));
            }
        }

        let cfg = DnsmosConfig::from_gguf(gguf)?;

        // For each advertised variant, collect the tensor names carrying
        // its prefix. A variant advertised in the bundle inventory but
        // with zero matching tensors is a hard error — a silent partial
        // bundle would let the loud-partial forward eventually surface
        // as "MOS = 0" rather than a load-time refusal.
        let mut bundles: Vec<DnsmosBundle> = Vec::with_capacity(cfg.bundle.len());
        for v in &cfg.bundle {
            let variant = match v.as_str() {
                "p808" => DnsmosSubmodel::P808,
                "p835" => DnsmosSubmodel::P835,
                // validated in DnsmosConfig::validate — unreachable.
                _ => unreachable!(),
            };
            let prefix = variant.tensor_prefix();
            let names: Vec<String> = gguf
                .tensors()
                .iter()
                .map(|t| t.name.clone())
                .filter(|n| n.starts_with(prefix))
                .collect();
            if names.is_empty() {
                return Err(VokraError::ModelLoad(format!(
                    "dnsmos: bundle inventory advertises variant `{v}` but the GGUF \
                     carries no tensors under the `{prefix}` prefix — the sidecar \
                     produced a stale metadata / empty tensor pair (FR-EX-08 forbids \
                     silent partial bundles)"
                )));
            }
            bundles.push(DnsmosBundle {
                variant,
                tensor_names: names,
            });
        }

        let variants: &'static [&'static str] = match (cfg.has_p808, cfg.has_p835) {
            (true, true) => &["p808", "p835"],
            (true, false) => &["p808"],
            (false, true) => &["p835"],
            // DnsmosConfig::validate refuses an empty bundle.
            (false, false) => unreachable!(),
        };

        Ok(Self {
            cfg,
            bundles: Arc::new(bundles),
            variants,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Builds a synthesized (test-only) session advertising both P.808
    /// and P.835. The `score_*` calls still hit the loud-partial — the
    /// CNN forward is not fabricated (FR-EX-08).
    pub fn synthesized() -> Self {
        Self {
            cfg: DnsmosConfig {
                bundle: vec!["p808".to_owned(), "p835".to_owned()],
                sample_rate: EXPECTED_SAMPLE_RATE,
                has_p808: true,
                has_p835: true,
            },
            bundles: Arc::new(vec![
                DnsmosBundle {
                    variant: DnsmosSubmodel::P808,
                    tensor_names: Vec::new(),
                },
                DnsmosBundle {
                    variant: DnsmosSubmodel::P835,
                    tensor_names: Vec::new(),
                },
            ]),
            variants: &["p808", "p835"],
        }
    }

    /// Returns the checkpoint's config.
    pub fn config(&self) -> &DnsmosConfig {
        &self.cfg
    }

    /// Returns the bound sub-model bundle metadata (mostly for
    /// diagnostics — the forward paths read weights on demand).
    pub fn bundles(&self) -> &[DnsmosBundle] {
        &self.bundles
    }

    /// Scores a clip against the P.808 predictor. Returns a MOS in
    /// `[1, 5]` on the future real path; today returns
    /// [`VokraError::UnsupportedOp`] with the topology-extension recipe.
    pub fn score_p808(&self, pcm16k: &[f32]) -> Result<f32> {
        if !self.cfg.has_p808 {
            return Err(VokraError::InvalidArgument(format!(
                "dnsmos: cannot score variant `p808` (P.808) — the bundle \
                 advertises only {:?}. A partial bundle must be scored on its \
                 advertised variants only (FR-EX-08 forbids fabricating a `None` \
                 MOS as a `0.0`)",
                self.cfg.bundle
            )));
        }
        // Loud-partial gate fires BEFORE the mel front-end runs — the
        // caller cannot observe a partial computation that looks like a
        // real forward. The `pcm16k` length is consumed only by the
        // future real path; we silence the unused warning by binding.
        let _ = pcm16k;
        Err(cnn_forward_loud_partial(DnsmosSubmodel::P808))
    }

    /// Scores a clip against the P.835 predictor. Returns
    /// `(SIG, BAK, OVRL)` MOS scalars on the future real path; today
    /// returns [`VokraError::UnsupportedOp`] with the topology-extension
    /// recipe.
    pub fn score_p835(&self, pcm16k: &[f32]) -> Result<(f32, f32, f32)> {
        if !self.cfg.has_p835 {
            return Err(VokraError::InvalidArgument(format!(
                "dnsmos: cannot score variant `p835` (P.835) — the bundle \
                 advertises only {:?}. A partial bundle must be scored on its \
                 advertised variants only (FR-EX-08 forbids fabricating a `None` \
                 MOS as a `0.0`)",
                self.cfg.bundle
            )));
        }
        let _ = pcm16k;
        Err(cnn_forward_loud_partial(DnsmosSubmodel::P835))
    }

    /// Scores a clip on every variant the bundle advertises, folding
    /// the results into a single [`MosScore`]. Absent variants stay
    /// `None` (never `Some(0.0)`). Any variant present in the bundle
    /// falls into the same loud-partial as [`Self::score_p808`] /
    /// [`Self::score_p835`], so this returns the first
    /// [`VokraError::UnsupportedOp`] encountered.
    pub fn score_all(&self, pcm16k: &[f32]) -> Result<MosScore> {
        let mut out = MosScore::default();
        if self.cfg.has_p808 {
            out.p808 = Some(self.score_p808(pcm16k)?);
        }
        if self.cfg.has_p835 {
            let (sig, bak, ovrl) = self.score_p835(pcm16k)?;
            out.sig = Some(sig);
            out.bak = Some(bak);
            out.ovrl = Some(ovrl);
        }
        Ok(out)
    }
}

impl MosScorerEngine for Dnsmos {
    fn variants(&self) -> &[&'static str] {
        self.variants
    }

    fn score(&self, pcm16k: &[f32]) -> Result<MosScore> {
        self.score_all(pcm16k)
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by every `score_*` path until the CNN topology metadata lands.
///
/// The message names both (a) the future GGUF metadata chunk that will
/// pin the op sequence and (b) the sidecar to extend. An owner (or the
/// follow-up CC wave) reading this error knows exactly where to flip
/// the switch — no fabricated `0.0` MOS ever appears (FR-EX-08).
fn cnn_forward_loud_partial(variant: DnsmosSubmodel) -> VokraError {
    let short = variant.short();
    let topo_key = match variant {
        DnsmosSubmodel::P808 => KEY_DNSMOS_P808_TOPOLOGY,
        DnsmosSubmodel::P835 => KEY_DNSMOS_P835_TOPOLOGY,
    };
    VokraError::UnsupportedOp(format!(
        "dnsmos {short}: CNN backbone forward is a loud-partial — the current \
         `tools/parity/dnsmos_prepare_checkpoint.py` sidecar walks the ONNX \
         `graph.initializer` only, so the `conv → BN → relu → pool` op sequence \
         is not primary-source-transcribable and would be silent-wrong if \
         best-guessed. Extend the sidecar to emit `{topo_key}` (u32 op-token \
         array from the ONNX `graph.node` list), re-run \
         `vokra-cli convert --model dnsmos-p808-p835`, then this call flips to \
         a real MOS ∈ [1, 5]. Until then this is a loud pending — no silent \
         fabricated 0.0 MOS (FR-EX-08)."
    ))
}
