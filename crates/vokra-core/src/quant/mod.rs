//! Quantization policy — mechanism-first, kernel-agnostic (M2-08).
//!
//! This module carries the *policy* half of the Vokra quantization design
//! and exists so the offline converter can decide *per tensor* what dtype
//! to emit, the runtime can read that decision back deterministically, and
//! both sides can reject quant paths that a model's ops forbid — **before**
//! hitting a backend (FR-EX-08: no silent CPU fallback, generalized here to
//! "no silent widen"). The INT8 GEMM kernels themselves are follow-up WPs;
//! M2-08 stops at policy validation errors when a scheme resolves to an
//! unsupported activation dtype.
//!
//! # WxAy terminology (pinned)
//!
//! Throughout this crate and the `docs/design/` design docs, quantization
//! schemes are named in the **`WxAy`** convention:
//!
//! - `W` is the **weight** bit width — the dtype the weight is stored in on
//!   disk / in the [`GgufFile`](crate::gguf::GgufFile).
//! - `A` is the **activation** bit width — the dtype activations flow in at
//!   runtime, i.e. what the GEMM's LHS/output are.
//!
//! Concretely:
//!
//! - **`W4A16`** — 4-bit weight (K-quant Q4_K/Q5_K, or an eventual pure 4-bit
//!   layout) + 16-bit activation (F16). "Weight-only quant" in the literature.
//! - **`W8A8`** — 8-bit weight (INT8) + 8-bit activation (INT8). Requires
//!   real INT8 GEMM kernels (AVX-VNNI / SDOT / i8mm). M2-08 defines the
//!   policy shape and reserves the alias; the kernels are a separate WP.
//! - **`FP32` / `FP16`** — the un-quantized tiers, kept as first-class
//!   [`scheme::QuantScheme`] variants so a policy can pin a specific op or
//!   model region (e.g. vocoders, biases) to full precision.
//!
//! When a scheme is called "K-quant" (`W4A16Q4K`, `W4A16Q5K`, `W4A16Q6K`) the
//! weight is stored in one of the [`GgmlType::Q4_K`](crate::gguf::GgmlType) /
//! `Q5_K` / `Q6_K` layouts (SSOT: `crates/vokra-core/src/gguf/tensor.rs`)
//! and dequantizes to F32 at load via
//! [`gguf::quant::dequantize`](crate::gguf); the activation dtype the policy
//! records is **F16** by contract, even though the actual GEMM in M2-08 still
//! runs F32 (see "Activation dtype in M2-08" below).
//!
//! # Requirements traceability
//!
//! - **FR-QT-02** — per-tensor / per-layer quantization policy: the
//!   [`policy::QuantPolicy`] builder + [`resolve::resolve`] pair, and the
//!   `vokra.quant.*` GGUF chunk in [`chunk`] so the converter's decisions
//!   round-trip to the runtime.
//! - **FR-QT-03** — quantization scheme constraints per op: the
//!   [`registry::MinDtypeRegistry`] anchors the FR-OP-10 (hifigan_generator),
//!   FR-OP-11 (bigvgan_generator), FR-OP-12 (vocos_head), FR-OP-13
//!   (snake_activation) minimum-dtype rules; [`validate`] rejects a policy
//!   that would drive any of those ops below their minimum activation dtype.
//! - **FR-QT-04** — HiFi-GAN INT8 opt-in: the sole atomic constructor path
//!   [`policy::QuantPolicy::with_hifigan_int8_opt_in`] flips the opt-in bool
//!   *and* attaches a required [`policy::CalibrationRef`] in one call, so
//!   "opt-in without calibration" is unrepresentable by construction. The
//!   MEL-loss / UTMOS verify (NFR-QL-02) is wired in T12 via `vokra-eval`,
//!   kept out of this crate to preserve the zero-dep leaf shape.
//!
//! # Existing-asset hooks (T01)
//!
//! Nothing here re-invents a storage layout, a dequantizer, a metric, or a
//! policy error type — M2-08 reuses what M0/M1 already delivered:
//!
//! - [`gguf::GgmlType`](crate::gguf::GgmlType) — the on-disk weight dtype
//!   tag. [`scheme::QuantScheme::weight_dtype`] returns one of these (or the
//!   `Int8Reserved` marker) so T05/T06 can drive tensor emission without a
//!   second dtype vocabulary.
//! - [`gguf::quant::dequantize`](crate::gguf) — the sole K-quant decode
//!   path (called from [`GgufFile::tensor_f32`](crate::gguf::GgufFile));
//!   T07 reload goes through it unchanged. This module never rolls its own
//!   dequant.
//! - `vokra_eval::MelLoss` (`crates/vokra-eval/src/metrics/mel_loss.rs`) —
//!   the mel-domain L1 metric the T11 helper `check_degradation` will drive
//!   the NFR-QL-02 5% gate through. Wired from `vokra-cli`, not from here,
//!   so `vokra-core` stays a zero-dep leaf.
//! - [`gguf::FrontendPolicy`](crate::gguf::FrontendPolicy) — the precedent
//!   for a policy that is *constructed in Rust*, persisted through a
//!   `vokra.*` GGUF chunk, and defaults to `Fail`. [`policy::QuantPolicy`]
//!   is shaped the same way (no serde, no TOML — NFR-DS-02 zero-dep).
//!
//! # Activation dtype in M2-08
//!
//! The `activation_dtype()` a [`scheme::QuantScheme`] reports is a
//! **metadata contract**: it says what the runtime *would* run activations
//! in if the backend had that kernel. In M2-08 the CPU / Metal / CUDA GEMMs
//! are all F32-only (`crates/vokra-backend-cpu/src/kernels/mod.rs` header);
//! the practical effect of `W4A16Q*K` on the hot path is that weights are
//! stored quantized and dequantized to F32 at load — exactly today's
//! behaviour. `W8A8` is a policy variant with no kernel: any policy that
//! resolves to it errors before hitting a backend (FR-EX-08). FP16 activation
//! kernels and INT8 activation kernels are both explicitly out of scope;
//! they are tracked as follow-up WPs.
//!
//! # Zero-dep invariant
//!
//! This module is std-only: no serde, no toml, no regex. Pattern matching in
//! [`policy::LayerPattern::Glob`] uses a hand-rolled matcher restricted to
//! `*` (any run of characters) and literal segments — the same idiom used
//! elsewhere in the codebase (e.g. `crates/vokra-core/src/frontend.rs`
//! string→enum helpers, `crates/vokra-core/src/compliance/level.rs`
//! `as_str`/`from_class_str` round-trip). Persistence flows through the
//! `vokra.quant.*` GGUF chunk (T05), never through a config file.
//!
//! # Out-of-scope for M2-08 (follow-ups)
//!
//! - INT8 GEMM kernels (AVX-VNNI / SDOT / i8mm) on the three backends.
//! - The HiFi-GAN / BigVGAN / Vocos kernel bodies themselves (FR-OP-10 /
//!   FR-OP-11 / FR-OP-12) — this module registers the *constraints* so the
//!   ops light up correctly when they land, but does not deliver the ops.
//! - KV cache quantization (FR-QT-05, v1.0 scope).
//! - Extending the converter policy pass to piper / CAM++ / Silero
//!   converters — M2-08 keeps the same converter scope as M1-02 (whisper).
//!
//! # WP completion checklist (T16, for PR body)
//!
//! When the M2-08 PR lands, its description must certify each of the
//! following against evidence surfaced by the CI run and the tests in this
//! crate + `vokra-eval` + `vokra-cli`:
//!
//! - **FR-QT-02** — per-tensor / per-layer policy delivered end-to-end:
//!   [`policy::QuantPolicy`] builder + [`resolve::resolve`] first-match
//!   dispatch + [`chunk`] `vokra.quant.*` GGUF round-trip. Covered by
//!   `quant::policy`, `quant::resolve`, `quant::chunk` unit tests (T02/T03/
//!   T04/T05).
//! - **FR-QT-03** — per-op minimum-dtype constraints delivered as an audit
//!   trail: [`registry::MinDtypeRegistry`] pre-populates FR-OP-10
//!   (`hifigan_generator`), FR-OP-11 (`bigvgan_generator`), FR-OP-12
//!   (`vocos_head`), FR-OP-13 (`snake_activation`) with `fr_ref` strings;
//!   [`validate::validate_policy_against_model`] rejects a policy that would
//!   drive any of those ops below their minimum activation dtype. Covered by
//!   `quant::registry` and `quant::validate` tests (T08/T09/T10).
//! - **NFR-DS-02** — zero-dep invariant preserved: `cargo deny check`
//!   green, `./scripts/check-zero-deps.sh` green, root `Cargo.lock` still
//!   `vokra-*` only. No new external crate dep introduced by this WP;
//!   `vokra-eval` remains `vokra-core` + `vokra-ops` only.
//! - **NFR-QL-01** — FP16 tier parity: `atol = 0.01` verified by
//!   `crates/vokra-core/tests/quant_parity.rs` (T13) driving hand-generated
//!   K-quant fixtures through `dequantize → GEMM → F32`.
//! - **NFR-QL-02** — degradation gate: `5%` relative MEL-loss threshold
//!   enforced by [`verify::DEGRADATION_THRESHOLD`] and the T11 helper
//!   `check_degradation`, wired through the T14 e2e test in
//!   `crates/vokra-cli/tests/policy_e2e.rs`.
//!
//! # Follow-up issues to open on merge (T16)
//!
//! The PR body must also open, and link, one tracking issue per item below.
//! These are intentional M2-08 exits, not oversights:
//!
//! 1. **W8A8 INT8 GEMM kernels per backend** (`vokra-backend-cpu` /
//!    `vokra-backend-metal` / `vokra-backend-cuda`) — AVX-VNNI / SDOT / i8mm
//!    ISA-gated paths; unblocks [`scheme::QuantScheme::W8A8Int8`] from
//!    always erroring at validate.
//! 2. **Vocos / BigVGAN / HiFi-GAN kernel bodies** (FR-OP-10 / FR-OP-11 /
//!    FR-OP-12) — delivered model-synced with their consumer models. The
//!    registry entries in [`registry`] already audit-anchor the constraints.
//! 3. **KV cache quantization** (FR-QT-05) — v1.0 scope.
//! 4. **Piper / CAM++ / Silero converter policy wiring** — extend the T06
//!    per-tensor `resolve` loop from the whisper converter to the other
//!    three model converters (currently widen-to-F32).
//! 5. **UTMOS / DNSMOS weight delivery** (M1-09b) — flips
//!    `DegradationReport::mel_loss_only` off and closes the perceptual
//!    half of NFR-QL-02.

pub mod chunk;
pub mod policy;
pub mod registry;
pub mod resolve;
pub mod scheme;
pub mod validate;
pub mod verify;

pub use policy::{CalibrationRef, LayerPattern, QuantPolicy, QuantRule};
pub use registry::{
    BIGVGAN_GENERATOR_OP, DowngradePolicy, HIFIGAN_GENERATOR_OP, KOKORO_ISTFT_HEAD_OP, MinDtype,
    MinDtypeEntry, MinDtypeRegistry, SNAKE_ACTIVATION_OP, VOCOS_HEAD_OP,
};
pub use resolve::{default_vocoder_safe, resolve, whisper_q4_k};
pub use scheme::{ActivationDtype, QuantScheme, WeightDtype};
pub use validate::validate_policy_against_model;
pub use verify::{DEGRADATION_THRESHOLD, DegradationReport, verify_hifigan_int8};
