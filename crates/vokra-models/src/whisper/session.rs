//! Whisper session construction with a resolved [`QuantPolicy`] (M2-08 T07).
//!
//! Layering: [`WhisperSession`] wraps a [`WhisperModel`] plus the [`QuantPolicy`]
//! the model was loaded under, and gates op instantiation against the policy's
//! per-op activation dtype (FR-EX-08). This is the c06 slice of M2-08:
//!
//! - **Weights unchanged** — K-quants still dequantize to F32 via `tensor_f32`
//!   at load, so the actual weight storage is orthogonal to activation gating.
//! - **FP16 is metadata-only in M2-08** — every kernel still runs FP32; a
//!   scheme resolving to [`ActivationDtype::F16`] is recorded on the session
//!   for the future FP16 kernel path but does not change dispatch today.
//! - **INT8 is rejected** — no INT8 activation kernel exists in any backend
//!   in M2-08. Every op resolving to [`ActivationDtype::Int8`] surfaces
//!   [`VokraError::UnsupportedQuantPath`] at session ctor, per FR-EX-08.
//!
//! The T05 (`vokra.quant.*` chunk reader) and T09 (`MinDtypeRegistry`
//! validate) attach later; this WP pins the T07 activation-dtype gate.

use std::sync::Arc;

use vokra_core::gguf::GgufFile;
use vokra_core::quant::{ActivationDtype, QuantPolicy};
use vokra_core::{BackendKind, Result, VokraError};

use super::WhisperModel;

/// The set of op-kind identifiers a Whisper session instantiates.
///
/// These are the strings T09 (`MinDtypeRegistry`) keys against — and the
/// strings [`VokraError::UnsupportedQuantPath`] carries in its `op` field so
/// callers can surface *which* op tripped the gate. The order matches
/// [`crate::whisper::WHISPER_HOT_OPS`]; layer-norm / conv1d are included
/// because Whisper routes them through the backend too (and INT8 kernels
/// would need to cover them together, not piecemeal — FR-EX-08 uniformity).
const WHISPER_OP_NAMES: &[&str] = &[
    "whisper::gemm",
    "whisper::gemv",
    "whisper::softmax",
    "whisper::layer_norm",
    "whisper::gelu",
    "whisper::conv1d",
];

/// A Whisper model bound to a [`QuantPolicy`] and a target [`BackendKind`].
///
/// Constructed by [`Self::from_gguf`] / [`Self::from_gguf_on`]. The wrapped
/// [`WhisperModel`] and the resolved [`QuantPolicy`] are both accessible so
/// downstream drivers (greedy / beam / streaming) can thread them through
/// without re-loading.
pub struct WhisperSession {
    model: Arc<WhisperModel>,
    policy: QuantPolicy,
    backend: BackendKind,
}

impl WhisperSession {
    /// Loads a Whisper model from `file`, reads its [`QuantPolicy`] and
    /// validates that every op the session will instantiate has an
    /// activation dtype the backend can execute.
    ///
    /// Defaults to [`BackendKind::Cpu`]; use [`Self::from_gguf_on`] for
    /// another backend.
    ///
    /// # Arch verification (FR-EX-08)
    ///
    /// `vokra.model.arch` is checked against
    /// [`ACCEPTED_ARCHS`](super::ACCEPTED_ARCHS) **before** any tensor is
    /// bound. Whisper is one of the binders that legitimately serves more
    /// than one arch tag (`whisper` / `crisper-whisper` / `distil-whisper`
    /// / `kotoba-whisper` all share the topology verbatim) — the whole set
    /// is accepted and everything else refused; the per-tag rationale is
    /// recorded on [`ACCEPTED_ARCHS`](super::ACCEPTED_ARCHS).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is
    ///   not one of [`ACCEPTED_ARCHS`](super::ACCEPTED_ARCHS);
    /// - [`VokraError::ModelLoad`] / [`VokraError::FrontendMismatch`] from
    ///   [`WhisperModel::from_gguf`];
    /// - [`VokraError::UnsupportedQuantPath`] if the resolved policy asks any
    ///   Whisper op to run at [`ActivationDtype::Int8`] (no INT8 kernel in
    ///   M2-08 on any backend, FR-EX-08).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_on(file, BackendKind::Cpu)
    }

    /// [`Self::from_gguf`] on an explicit [`BackendKind`]. The activation
    /// gate applies uniformly across backends: an INT8 activation is
    /// rejected on CPU, Metal, and CUDA alike (FR-EX-08 uniformity). The
    /// arch gate documented on [`Self::from_gguf`] applies here too — this
    /// is the shared body both entry points run.
    pub fn from_gguf_on(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "vokra.whisper.n_mels missing" or, worse, a partial bind.
        super::verify_arch(file)?;
        let model = WhisperModel::from_gguf(file)?;
        // T05 lands the real chunk reader inside `QuantPolicy::from_gguf`;
        // until then the reader returns `default_vocoder_safe`, which is
        // exactly the "chunk absent → safe default" fall-through c06 pins.
        let policy = load_quant_policy(file)?;
        gate_ops_against_policy(&policy, WHISPER_OP_NAMES, backend)?;
        Ok(Self {
            model: Arc::new(model),
            policy,
            backend,
        })
    }

    /// The wrapped [`WhisperModel`] behind the [`Arc`] the decoder drivers
    /// already consume via [`WhisperModel::decoder`].
    pub fn model(&self) -> &Arc<WhisperModel> {
        &self.model
    }

    /// The [`QuantPolicy`] this session was loaded under.
    pub fn policy(&self) -> &QuantPolicy {
        &self.policy
    }

    /// The [`BackendKind`] the session gated its ops against.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }
}

/// Read a [`QuantPolicy`] from a GGUF file, falling back to the
/// vocoder-safe default when the `vokra.quant.*` chunk is absent.
///
/// The T05 chunk reader lands as `QuantPolicy::from_gguf` on the type
/// itself; c06 keeps the fall-through inline here so this file is
/// self-contained under the change-scope rule. When T05 attaches the real
/// reader, this helper collapses to a single delegation.
fn load_quant_policy(_file: &GgufFile) -> Result<QuantPolicy> {
    // Chunk-absent path — every GGUF today. Vocoder-safe = all-fp16, no rules
    // (T04 preset in `crate::quant::resolve`).
    Ok(vokra_core::quant::resolve::default_vocoder_safe())
}

/// For each op-kind the model will instantiate, resolve its scheme under the
/// policy and reject if the activation dtype has no kernel path on `backend`.
///
/// M2-08 T07 policy:
/// - [`ActivationDtype::F32`] and [`ActivationDtype::F16`] pass — F16 is
///   metadata-only today (kernels still run F32), but the gate documents
///   which ops opt into the fp16 path.
/// - [`ActivationDtype::Int8`] fails on every backend (FR-EX-08). No silent
///   downgrade to F32 / F16.
fn gate_ops_against_policy(
    policy: &QuantPolicy,
    op_names: &[&str],
    backend: BackendKind,
) -> Result<()> {
    for op in op_names {
        // T04 resolver: policy-carried rule walk with fall-through to the
        // policy's default. c06 relies on `resolve` never failing.
        let scheme = vokra_core::quant::resolve(policy, op);
        match scheme.activation_dtype() {
            ActivationDtype::F32 | ActivationDtype::F16 => continue,
            ActivationDtype::Int8 => {
                return Err(VokraError::UnsupportedQuantPath {
                    op: (*op).to_owned(),
                    scheme: scheme.as_str().to_owned(),
                    backend: backend_display(backend).to_owned(),
                });
            }
        }
    }
    Ok(())
}

/// Human-readable [`BackendKind`] name for [`VokraError::UnsupportedQuantPath`].
fn backend_display(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Metal => "metal",
        BackendKind::Cuda => "cuda",
        _ => "unknown",
    }
}

#[cfg(test)]
mod quant_load {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_core::quant::QuantScheme;
    use vokra_core::quant::policy::LayerPattern;

    /// A GGUF with no `vokra.quant.*` chunk and no whisper config — used to
    /// exercise `load_quant_policy` in isolation from the model load (which
    /// would fail on missing config).
    fn empty_gguf() -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_u32("unrelated.key", 1);
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    /// Builds a metadata-only GGUF carrying `arch` (or none when `arch` is
    /// `None`) so the arch gate can be exercised without a weight manifest.
    fn arch_only_gguf(arch: Option<&str>) -> GgufFile {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, a);
        }
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn from_gguf_rejects_gguf_without_arch_stamp() {
        // FR-EX-08: an unstamped GGUF is not a Vokra-native Whisper
        // artifact. `WhisperSession` is `!Debug`, so `unwrap_err()` would
        // not compile on the `Result` — use let-else on the `Err` arm.
        let file = arch_only_gguf(None);
        let Err(err) = WhisperSession::from_gguf(&file) else {
            panic!("an unstamped GGUF must not build a Whisper session");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains(vokra_core::gguf::chunks::KEY_MODEL_ARCH),
                "message must name the missing key: {msg}",
            ),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_expected_and_actual() {
        // A Kokoro GGUF is the canonical mis-route from the audit note.
        let file = arch_only_gguf(Some("kokoro"));
        let Err(err) = WhisperSession::from_gguf_on(&file, BackendKind::Cpu) else {
            panic!("a foreign-arch GGUF must not build a Whisper session");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("kokoro"),
                    "message must name the actual arch: {msg}",
                );
                assert!(
                    msg.contains("whisper"),
                    "message must name the expected arch set: {msg}",
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_accepts_every_whisper_family_arch_past_the_arch_gate() {
        // Whisper legitimately serves four arch tags. Each must clear the
        // arch gate and fail *later* (on the absent `vokra.whisper.*`
        // config), never at the arch gate itself — otherwise a working
        // distil-whisper / kotoba-whisper / CrisperWhisper checkpoint would
        // regress to unloadable.
        for &arch in super::super::ACCEPTED_ARCHS {
            let file = arch_only_gguf(Some(arch));
            let Err(err) = WhisperSession::from_gguf(&file) else {
                panic!("{arch}: a weightless GGUF must still fail");
            };
            match err {
                VokraError::ModelLoad(msg) => assert!(
                    !msg.contains("GGUF arch is") && !msg.contains("is missing `vokra.model.arch`"),
                    "{arch} must clear the arch gate, got: {msg}",
                ),
                other => panic!("{arch}: expected ModelLoad, got {other:?}"),
            }
        }
    }

    #[test]
    fn whisper_medusa_is_not_accepted_by_the_arch_gate() {
        // `whisper-medusa-v1` shares the base topology but adds Medusa
        // speculative-decoding heads this binder has no module for.
        // Admitting it would silently drop the heads.
        let file = arch_only_gguf(Some("whisper-medusa-v1"));
        let Err(err) = WhisperSession::from_gguf(&file) else {
            panic!("whisper-medusa-v1 must not build a Whisper session");
        };
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains("whisper-medusa-v1"),
                "message must name the actual arch: {msg}",
            ),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn load_quant_policy_defaults_to_vocoder_safe_when_chunk_absent() {
        // c06 contract: chunk-absent → default_vocoder_safe (all-fp16).
        let file = empty_gguf();
        let policy = load_quant_policy(&file).unwrap();
        assert_eq!(policy.default_scheme(), QuantScheme::Fp16);
        assert!(policy.rules().is_empty());
    }

    #[test]
    fn gate_passes_for_fp16_default_policy_on_cpu() {
        // Every Whisper op under the vocoder-safe default resolves to Fp16 →
        // activation F16 → passes today's gate (F16 kernels run as F32 in
        // M2-08; the gate is dtype-only).
        let policy = vokra_core::quant::resolve::default_vocoder_safe();
        gate_ops_against_policy(&policy, WHISPER_OP_NAMES, BackendKind::Cpu).unwrap();
    }

    #[test]
    fn gate_passes_for_fp32_default_policy() {
        let policy = QuantPolicy::new(QuantScheme::Fp32);
        gate_ops_against_policy(&policy, WHISPER_OP_NAMES, BackendKind::Cpu).unwrap();
        gate_ops_against_policy(&policy, WHISPER_OP_NAMES, BackendKind::Metal).unwrap();
        gate_ops_against_policy(&policy, WHISPER_OP_NAMES, BackendKind::Cuda).unwrap();
    }

    #[test]
    fn gate_rejects_int8_default_policy_on_every_backend() {
        // FR-EX-08 uniformity: INT8 must fail on CPU, Metal, and CUDA alike
        // (no INT8 activation kernel exists anywhere in M2-08). The error
        // must carry the op, scheme, and backend so the caller can surface a
        // policy edit hint without string-matching the message body.
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
        for kind in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let err = gate_ops_against_policy(&policy, WHISPER_OP_NAMES, kind).unwrap_err();
            let expected_backend = backend_display(kind);
            match err {
                VokraError::UnsupportedQuantPath {
                    op,
                    scheme,
                    backend,
                } => {
                    // Whisper op names use the `whisper::` prefix.
                    assert!(op.starts_with("whisper::"), "op field was `{op}`");
                    assert_eq!(scheme, "w8a8");
                    assert_eq!(backend, expected_backend);
                }
                other => panic!("expected UnsupportedQuantPath, got {other:?}"),
            }
        }
    }

    #[test]
    fn gate_rejects_int8_when_targeted_by_a_specific_rule() {
        // Even a targeted rule that only hits one op must trip the gate — no
        // silent per-op downgrade, no partial success.
        let policy = QuantPolicy::new(QuantScheme::Fp16).with_rule(
            LayerPattern::Exact("whisper::gemm".to_owned()),
            QuantScheme::W8A8Int8,
        );
        let err = gate_ops_against_policy(&policy, WHISPER_OP_NAMES, BackendKind::Cpu).unwrap_err();
        match err {
            VokraError::UnsupportedQuantPath {
                op,
                scheme,
                backend,
            } => {
                assert_eq!(op, "whisper::gemm");
                assert_eq!(scheme, "w8a8");
                assert_eq!(backend, "cpu");
            }
            other => panic!("expected UnsupportedQuantPath, got {other:?}"),
        }
    }

    #[test]
    fn error_display_mentions_no_silent_fallback() {
        // FR-EX-08 audit trail: the error's `Display` names FR-EX-08 so an
        // operator reading logs can trace the policy back to the requirement.
        let err = VokraError::UnsupportedQuantPath {
            op: "whisper::gemm".to_owned(),
            scheme: "w8a8".to_owned(),
            backend: "cpu".to_owned(),
        };
        let msg = err.to_string();
        assert!(msg.contains("FR-EX-08"), "message was: {msg}");
        assert!(msg.contains("whisper::gemm"));
        assert!(msg.contains("w8a8"));
        assert!(msg.contains("cpu"));
    }

    #[test]
    fn backend_display_covers_known_kinds() {
        assert_eq!(backend_display(BackendKind::Cpu), "cpu");
        assert_eq!(backend_display(BackendKind::Metal), "metal");
        assert_eq!(backend_display(BackendKind::Cuda), "cuda");
    }
}
