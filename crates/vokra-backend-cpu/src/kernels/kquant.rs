//! K-quants SIMD dequant fusion + the specialized INT8 / BF16 / FP16
//! dispatch surface (M4-17-T10..T17).
//!
//! This module fulfils the `vokra-core` placement note
//! (`gguf/quant/mod.rs`): the safe scalar dequant in `vokra-core` is the
//! **oracle**, and the SIMD-accelerated path lives here in the
//! `unsafe`-allowed backend crate and "MUST stay bit-identical to this
//! reference". Three surfaces (ADR M4-17 §(b)-2/(e)):
//!
//! 1. **Bit-identical dequant → f32** ([`kquant_dequant_on`]): per-format
//!    SIMD kernels that replay the exact core arithmetic (`d1 * q - m1` =
//!    mul→sub, two roundings — **never** fused into an FMA, which would
//!    change the bits) lane-parallel. AVX2 and NEON implementations; the
//!    AVX-512-family tiers delegate to the AVX2 implementation (dequant is
//!    memory-bound; a zmm-width port is a perf follow-up and perf here is
//!    advisory/owner-measured).
//! 2. **INT8 fusion** ([`kquant_gemv_i8`] / [`kquant_gemv_i8_on`] /
//!    [`kquant_gemv2_i8_on`]): weights stay in their K-quant super-blocks,
//!    unpacked to a format-independent `(q_u8[256], per-16-element-group
//!    d_eff/m_eff)` representation; activations are Q8-quantized per
//!    16-element group. The per-group integer sums `isum = Σ q·q8` /
//!    `asum = Σ q8` are exact i32 on every path (scalar, AVX-512 VNNI,
//!    AVX-VNNI-256, NEON dotprod, i8mm), and the float combine is **one
//!    shared scalar function** — so all INT8 ISA paths are bit-identical to
//!    the scalar-int8 reference, and the only approximation vs the
//!    f32-dequant GEMV is the activation quantization (the honest,
//!    input-derived atol bound — see [`int8_error_bound`]).
//! 3. **Reduced-precision matmuls** ([`gemm_bf16_on`] / [`gemm_fp16_on`]):
//!    opt-in tiers; the `Scalar` arm is the emulation oracle (bf16/fp16
//!    input rounding + reference accumulation), the SIMD arms are
//!    `vdpbf16ps` / BFMMLA / fp16 FMLA. Bands are architectural
//!    (mantissa-derived, input-dependent) — see [`bf16_dot_bound`].
//!
//! Block layouts are transcribed from the `vokra-core` decoders (themselves
//! transcribed from ggml `k_quants.h`, MIT — a data-format specification);
//! the unit tests pin every transcription against the core oracle.

use vokra_core::gguf::quant as core_quant;
use vokra_core::gguf::tensor::QK_K;
use vokra_core::{Result, VokraError};

use crate::features::{CpuFeatures, IsaPath};

/// Elements per K-quant scale group in the unified INT8 representation.
/// All three formats hold their sub-scale constant over 16 contiguous
/// elements (Q4_K/Q5_K use 32-element sub-scales = two identical groups;
/// Q6_K is natively per-16).
pub const KQUANT_GROUP: usize = 16;
/// Scale groups per 256-element super-block.
const GROUPS: usize = QK_K / KQUANT_GROUP;

/// The K-quant weight formats served by the specialized INT8 / dequant
/// kernels (M1-02 formats; mirrors `vokra_core::gguf::GgmlType`'s K-quant
/// subset without pulling the full dtype enum into kernel signatures).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KQuantDtype {
    /// `block_q4_K`: 144 bytes / 256 elements.
    Q4K,
    /// `block_q5_K`: 176 bytes / 256 elements.
    Q5K,
    /// `block_q6_K`: 210 bytes / 256 elements.
    Q6K,
}

impl KQuantDtype {
    /// On-disk super-block size in bytes.
    pub const fn block_bytes(self) -> usize {
        match self {
            KQuantDtype::Q4K => 144,
            KQuantDtype::Q5K => 176,
            KQuantDtype::Q6K => 210,
        }
    }

    /// The K-quant [`GgmlType`](vokra_core::gguf::GgmlType) mapped into this
    /// kernel-side enum, or `None` for any non-K-quant dtype (`F32` / `F16` /
    /// `BF16`). Lets a model loader decide whether an on-disk tensor can feed
    /// the fused INT8 kernels without duplicating the mapping (M5-15-T26).
    pub const fn from_ggml(t: vokra_core::gguf::GgmlType) -> Option<Self> {
        match t {
            vokra_core::gguf::GgmlType::Q4K => Some(KQuantDtype::Q4K),
            vokra_core::gguf::GgmlType::Q5K => Some(KQuantDtype::Q5K),
            vokra_core::gguf::GgmlType::Q6K => Some(KQuantDtype::Q6K),
            _ => None,
        }
    }

    fn ggml(self) -> vokra_core::gguf::GgmlType {
        match self {
            KQuantDtype::Q4K => vokra_core::gguf::GgmlType::Q4K,
            KQuantDtype::Q5K => vokra_core::gguf::GgmlType::Q5K,
            KQuantDtype::Q6K => vokra_core::gguf::GgmlType::Q6K,
        }
    }
}

// ---------------------------------------------------------------------
// f16 / bf16 conversions (shared by the fp16/bf16 kernels and their
// emulation oracles).
// ---------------------------------------------------------------------

/// Rounds an exact `f64` value to IEEE-754 binary16 (round-to-nearest-even).
///
/// Implemented over exact dyadic arithmetic (`v / 2^e` is exact for powers
/// of two, `f64::round_ties_even` is the RNE primitive), so there is no
/// double rounding through `f32`. Overflow saturates to ±inf, NaN maps to a
/// quiet NaN.
pub fn f64_to_f16_rne(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    // Half-precision boundaries: max finite 65504, min normal 2^-14.
    if a >= 65520.0 {
        // 65504 + ulp/2 rounds to inf.
        return sign | 0x7C00;
    }
    if a < 2f64.powi(-14) {
        // Subnormal: quantum 2^-24.
        let q = (a / 2f64.powi(-24)).round_ties_even();
        let qi = q as u16; // 0..=1024
        if qi >= 1024 {
            return sign | 0x0400; // rounded up to the smallest normal
        }
        return sign | qi;
    }
    // Normal: find the exponent via bit inspection of the f64.
    let bits = a.to_bits();
    let e = ((bits >> 52) & 0x7FF) as i32 - 1023; // floor(log2(a))
    let ulp = 2f64.powi(e - 10);
    let q = (a / ulp).round_ties_even(); // in [1024, 2048]
    let (qi, e) = if q >= 2048.0 {
        (1024u32, e + 1)
    } else {
        (q as u32, e)
    };
    if e + 15 >= 31 {
        return sign | 0x7C00;
    }
    sign | (((e + 15) as u16) << 10) | ((qi - 1024) as u16)
}

/// Rounds `f32` → binary16 RNE (exact: `f32 → f64` is lossless).
pub fn f32_to_f16_rne(x: f32) -> u16 {
    f64_to_f16_rne(f64::from(x))
}

/// binary16 → f32 (exact) — re-exported from the `vokra-core` oracle so both
/// sides of every fp16 parity comparison decode identically.
pub fn f16_to_f32(h: u16) -> f32 {
    core_quant::f16_to_f32(h)
}

/// Rounds `f32` → bfloat16 RNE (the standard bias-and-truncate bit trick;
/// overflow saturates to inf naturally, NaN keeps a set mantissa bit).
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    (((b as u64 + 0x7FFF + ((b as u64 >> 16) & 1)) >> 16) & 0xFFFF) as u16
}

/// bfloat16 → f32 (exact).
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Emulated fp16 fused multiply-add: `round_f16(a * b + c)` with the product
/// and sum carried in `f64` (fp16 products are ≤ 22-bit dyadics = exact in
/// f64; the sum is exact except for astronomically wide exponent spans, and
/// the parity band is ±2 fp16 ulp precisely to absorb such residuals — ADR
/// M4-17 §(f)).
pub fn fp16_fma_emu(a: u16, b: u16, c: u16) -> u16 {
    let av = f64::from(f16_to_f32(a));
    let bv = f64::from(f16_to_f32(b));
    let cv = f64::from(f16_to_f32(c));
    f64_to_f16_rne(av * bv + cv)
}

// ---------------------------------------------------------------------
// Bit-identical SIMD dequant (surface 1).
// ---------------------------------------------------------------------

/// Validates a K-quant payload against `n_elements` (mirrors the core
/// dispatch: whole super-blocks, exact byte length).
fn validate_kquant(dtype: KQuantDtype, bytes: &[u8], n_elements: usize) -> Result<usize> {
    if n_elements == 0 || !n_elements.is_multiple_of(QK_K) {
        return Err(VokraError::InvalidArgument(format!(
            "K-quant element count {n_elements} is not a positive multiple of {QK_K}"
        )));
    }
    let nb = n_elements / QK_K;
    let want = nb * dtype.block_bytes();
    if bytes.len() != want {
        return Err(VokraError::InvalidArgument(format!(
            "{dtype:?} payload is {} bytes, want {want} for {n_elements} elements",
            bytes.len()
        )));
    }
    Ok(nb)
}

/// Dequantizes a K-quant payload to `f32` on a forced ISA path
/// (M4-17-T12). **Bit-identical contract**: every path returns exactly the
/// bytes the `vokra-core` scalar reference returns.
///
/// - `Scalar` runs the core reference itself (the oracle).
/// - The x86-64 tiers (`Avx2` and the AVX-512 family / `AvxVnni256`, whose
///   `supports` gates all include AVX2) run the AVX2 lane-parallel replay.
/// - The AArch64 tiers run the NEON replay.
/// - `Rvv` / `Rvv071` / `WasmSimd128` have no SIMD dequant in this WP — an
///   explicit [`VokraError::UnsupportedOp`], never a silent scalar switch
///   (FR-EX-08 principle).
///
/// # Errors
/// [`VokraError::BackendUnavailable`] when this host cannot run `isa`;
/// [`VokraError::InvalidArgument`] on a malformed payload.
pub fn kquant_dequant_on(
    isa: IsaPath,
    dtype: KQuantDtype,
    bytes: &[u8],
    n_elements: usize,
) -> Result<Vec<f32>> {
    if !CpuFeatures::detect().supports(isa) {
        return Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )));
    }
    validate_kquant(dtype, bytes, n_elements)?;
    match isa {
        IsaPath::Scalar => core_quant::dequantize(dtype.ggml(), bytes, n_elements)
            .map_err(|e| VokraError::InvalidArgument(format!("K-quant payload rejected: {e}"))),
        IsaPath::Avx2
        | IsaPath::Avx512
        | IsaPath::Avx512Vnni
        | IsaPath::Avx512Bf16
        | IsaPath::AvxVnni256 => {
            #[cfg(target_arch = "x86_64")]
            {
                Ok(x86::dequant(dtype, bytes, n_elements))
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("x86-64 tier passed `supports` off x86-64")
            }
        }
        IsaPath::Neon
        | IsaPath::NeonFp16
        | IsaPath::NeonDotprod
        | IsaPath::NeonI8mm
        | IsaPath::NeonBf16 => {
            #[cfg(target_arch = "aarch64")]
            {
                Ok(arm::dequant(dtype, bytes, n_elements))
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                unreachable!("aarch64 tier passed `supports` off aarch64")
            }
        }
        other => Err(VokraError::UnsupportedOp(format!(
            "no SIMD K-quant dequant kernel on the {other} path (M4-17 covers the AVX2/AVX-512 and NEON families)"
        ))),
    }
}

/// Unpacks the 6-bit sub-scale/min pair for sub-block `j` (0..8) —
/// transcription of ggml `get_scale_min_k4` (pinned against the core oracle
/// by the dequant bit-identity tests).
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Per-sub-block scale pairs `(d1, m1)` for a Q4_K/Q5_K block — computed
/// with the exact core expressions (`d * f32::from(sc)`), shared by the
/// SIMD dequant replays and the INT8 unpack.
fn q45_sub_scales(block: &[u8]) -> [(f32, f32); 8] {
    let d = core_quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = core_quant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let mut out = [(0.0f32, 0.0f32); 8];
    for (j, slot) in out.iter_mut().enumerate() {
        let (sc, mn) = get_scale_min_k4(j, scales);
        *slot = (d * f32::from(sc), dmin * f32::from(mn));
    }
    out
}

/// Q6_K per-16-element effective scales `d * f32::from(sc[g] as i8)` in
/// output-element order (`g = element / 16`), matching the core decoder's
/// `8*half + l/16 + 2*quarter` schedule.
fn q6_group_scales(block: &[u8]) -> [f32; GROUPS] {
    let d = core_quant::f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let sc = &block[192..208];
    let mut out = [0.0f32; GROUPS];
    for half in 0..2 {
        for quarter in 0..4 {
            for sub in 0..2 {
                // Elements 128*half + 32*quarter + 16*sub .. +16 use scale
                // index 8*half + sub + 2*quarter.
                let g = (128 * half + 32 * quarter + 16 * sub) / 16;
                let idx = 8 * half + sub + 2 * quarter;
                out[g] = d * f32::from(sc[idx] as i8);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// INT8 fusion seam (surface 2).
// ---------------------------------------------------------------------

/// A super-block unpacked to the format-independent INT8 representation:
/// `y[t] ≈ d_eff[t/16] * q[t] - m_eff[t/16]` (exactly the core dequant for
/// Q4_K/Q5_K; for Q6_K `q` is the biased value `q6 + 32` and
/// `m_eff = d_eff * 32`, an algebraic identity that differs from the core
/// value only in f32 association — the INT8 surface is bounded by the
/// activation-quantization band, not bit identity).
pub(crate) struct UnpackedBlock {
    q: [u8; QK_K],
    d_eff: [f32; GROUPS],
    m_eff: [f32; GROUPS],
}

impl UnpackedBlock {
    fn zeroed() -> Self {
        UnpackedBlock {
            q: [0u8; QK_K],
            d_eff: [0.0; GROUPS],
            m_eff: [0.0; GROUPS],
        }
    }
}

/// Unpacks one super-block into the unified INT8 representation
/// (M4-17-T12's "dequant → int8 seam").
#[allow(clippy::needless_range_loop)] // explicit sub-block index math is clearer
fn unpack_block_i8(dtype: KQuantDtype, block: &[u8], out: &mut UnpackedBlock) {
    debug_assert_eq!(block.len(), dtype.block_bytes());
    match dtype {
        KQuantDtype::Q4K | KQuantDtype::Q5K => {
            let subs = q45_sub_scales(block);
            let qs = match dtype {
                KQuantDtype::Q4K => &block[16..144],
                KQuantDtype::Q5K => &block[48..176],
                KQuantDtype::Q6K => unreachable!(),
            };
            let qh = (dtype == KQuantDtype::Q5K).then(|| &block[16..48]);
            for j in 0..8 {
                let (d1, m1) = subs[j];
                out.d_eff[2 * j] = d1;
                out.d_eff[2 * j + 1] = d1;
                out.m_eff[2 * j] = m1;
                out.m_eff[2 * j + 1] = m1;
                let chunk = j / 2;
                let hi_nibble = j % 2 == 1;
                for t in 0..32 {
                    let byte = qs[32 * chunk + t];
                    let mut q = if hi_nibble { byte >> 4 } else { byte & 0xF };
                    if let Some(qh) = qh {
                        // The 5th bit for sub-block j is qh bit j (low half
                        // of chunk k uses bit 2k, high half bit 2k+1 = j).
                        if qh[t] & (1u8 << j) != 0 {
                            q += 16;
                        }
                    }
                    out.q[32 * j + t] = q;
                }
            }
        }
        KQuantDtype::Q6K => {
            let dg = q6_group_scales(block);
            for (g, &d_eff) in dg.iter().enumerate() {
                out.d_eff[g] = d_eff;
                // y = d_eff * (q6 - 32)  ⇔  d_eff * q_u8 - (d_eff * 32);
                // 32 is a power of two so `d_eff * 32.0` is exact.
                out.m_eff[g] = d_eff * 32.0;
            }
            let ql_all = &block[0..128];
            let qh_all = &block[128..192];
            for half in 0..2 {
                let ql = &ql_all[half * 64..half * 64 + 64];
                let qh = &qh_all[half * 32..half * 32 + 32];
                let y = &mut out.q[half * 128..half * 128 + 128];
                for l in 0..32 {
                    y[l] = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
                    y[l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                    y[l + 64] = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                    y[l + 96] = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                }
            }
        }
    }
}

/// Q8-quantized activations: per-16-element-group symmetric quantization
/// (`s = max|x| / 127`, `q = round(x / s)` clamped to ±127) plus the
/// precomputed per-group sums `Σ q` the combine consumes.
pub(crate) struct Q8Activations {
    q: Vec<i8>,
    scales: Vec<f32>,
    gsums: Vec<i32>,
}

fn quantize_activations(x: &[f32]) -> Q8Activations {
    debug_assert!(x.len().is_multiple_of(KQUANT_GROUP));
    let ng = x.len() / KQUANT_GROUP;
    let mut q = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; ng];
    let mut gsums = vec![0i32; ng];
    for g in 0..ng {
        let xs = &x[KQUANT_GROUP * g..KQUANT_GROUP * (g + 1)];
        let amax = xs.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            continue; // scale 0.0, all-zero quants, gsum 0.
        }
        let s = amax / 127.0;
        let mut gsum = 0i32;
        for (t, &v) in xs.iter().enumerate() {
            let r = (v / s).round().clamp(-127.0, 127.0) as i32;
            q[KQUANT_GROUP * g + t] = r as i8;
            gsum += r;
        }
        scales[g] = s;
        gsums[g] = gsum;
    }
    Q8Activations { q, scales, gsums }
}

/// The shared float combine (ADR M4-17 §(e)): one fixed expression and
/// group order, so every INT8 ISA path (whose integer sums are exact) is
/// bit-identical to the scalar-int8 reference.
#[inline]
fn combine_group(acc: &mut f32, s_act: f32, d_eff: f32, m_eff: f32, isum: i32, asum: i32) {
    *acc += s_act * (d_eff * isum as f32 - m_eff * asum as f32);
}

/// Scalar per-group integer sums — the reference the SIMD group-sum cores
/// are exactly equal to (integer arithmetic).
fn scalar_group_sums(q: &[u8], x: &[i8], sums: &mut [i32]) {
    debug_assert_eq!(q.len(), x.len());
    debug_assert_eq!(sums.len() * KQUANT_GROUP, q.len());
    for (g, s) in sums.iter_mut().enumerate() {
        let mut acc = 0i32;
        for t in 0..KQUANT_GROUP {
            acc += i32::from(q[KQUANT_GROUP * g + t]) * i32::from(x[KQUANT_GROUP * g + t]);
        }
        *s = acc;
    }
}

/// The theoretical INT8-vs-f32 error bound for one GEMV row (honest atol
/// derivation, feedback memory "parity atol は architectural bound 由来"):
/// activation rounding is ≤ `s_g / 2` per element, so
/// `|int8 − f32| ≤ Σ_g (s_g / 2) · Σ_{t∈g} |w_t|` plus FP32 accumulation
/// noise. The differential tests assert against `2 ×` this input-derived
/// bound rather than a fixed tolerance.
pub fn int8_error_bound(dtype: KQuantDtype, w_row: &[u8], x: &[f32]) -> f32 {
    let nb = w_row.len() / dtype.block_bytes();
    let mut ub = UnpackedBlock::zeroed();
    let mut bound = 0.0f32;
    for blk in 0..nb {
        unpack_block_i8(
            dtype,
            &w_row[blk * dtype.block_bytes()..(blk + 1) * dtype.block_bytes()],
            &mut ub,
        );
        for g in 0..GROUPS {
            let xs = &x[blk * QK_K + KQUANT_GROUP * g..blk * QK_K + KQUANT_GROUP * (g + 1)];
            let amax = xs.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let s = amax / 127.0;
            let wsum: f32 = (0..KQUANT_GROUP)
                .map(|t| {
                    (ub.d_eff[g] * f32::from(ub.q[QK_K.min(KQUANT_GROUP * g + t)]) - ub.m_eff[g])
                        .abs()
                })
                .sum();
            bound += 0.5 * s * wsum;
        }
    }
    bound
}

/// Which group-sum core a forced INT8 path uses.
/// Signature of a per-group INT8 dot-sum core.
type GroupSumsFn = fn(&[u8], &[i8], &mut [i32]);

fn int8_group_sums_for(isa: IsaPath) -> Result<GroupSumsFn> {
    if !CpuFeatures::detect().supports(isa) {
        return Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )));
    }
    match isa {
        IsaPath::Scalar => Ok(scalar_group_sums),
        IsaPath::Avx512Vnni => {
            #[cfg(target_arch = "x86_64")]
            {
                Ok(super::avx512::vnni_group_sums)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("Avx512Vnni passed `supports` off x86-64")
            }
        }
        IsaPath::AvxVnni256 => {
            #[cfg(target_arch = "x86_64")]
            {
                Ok(super::avxvnni256::vnni256_group_sums)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("AvxVnni256 passed `supports` off x86-64")
            }
        }
        IsaPath::NeonDotprod => {
            #[cfg(target_arch = "aarch64")]
            {
                Ok(super::neon_dotprod::dotprod_group_sums)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                unreachable!("NeonDotprod passed `supports` off aarch64")
            }
        }
        other => Err(VokraError::UnsupportedOp(format!(
            "no K-quant INT8 GEMV kernel on the {other} path (int8 tiers: avx512vnni | avxvnni256 | neon-dotprod | scalar)"
        ))),
    }
}

fn validate_gemv_i8(
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    w: &[u8],
    x_len: usize,
    out_len: usize,
    n_x: usize,
) -> Result<usize> {
    if k == 0 || !k.is_multiple_of(QK_K) {
        return Err(VokraError::InvalidArgument(format!(
            "K-quant INT8 GEMV needs k to be a positive multiple of {QK_K}, got {k}"
        )));
    }
    let nb = k / QK_K;
    let row_bytes = nb * dtype.block_bytes();
    if w.len() != m * row_bytes {
        return Err(VokraError::InvalidArgument(format!(
            "K-quant weight payload is {} bytes, want {} (m={m} rows x {row_bytes})",
            w.len(),
            m * row_bytes
        )));
    }
    if x_len != n_x * k {
        return Err(VokraError::InvalidArgument(format!(
            "activation length {x_len} != {n_x} vector(s) x k={k}"
        )));
    }
    if out_len != n_x * m {
        return Err(VokraError::InvalidArgument(format!(
            "output length {out_len} != {n_x} column(s) x m={m}"
        )));
    }
    Ok(nb)
}

/// Row-major K-quant INT8 GEMV on a forced ISA path (M4-17-T10/T13/T15):
/// `out[i] = Σ_l dequant(w[i, l]) · x[l]` computed through the Q8 activation
/// quantization + exact integer group sums. `w` holds `m` rows of
/// `k / 256` super-blocks each; `k` must be a multiple of 256.
///
/// Every accepted path returns bit-identical results (shared combine, exact
/// integer sums); the difference vs the f32 dequant GEMV is bounded by
/// [`int8_error_bound`].
#[allow(clippy::needless_range_loop)] // group index drives four parallel arrays
pub fn kquant_gemv_i8_on(
    isa: IsaPath,
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    w: &[u8],
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let group_sums = int8_group_sums_for(isa)?;
    let nb = validate_gemv_i8(dtype, m, k, w, x.len(), out.len(), 1)?;
    let acts = quantize_activations(x);
    let row_bytes = nb * dtype.block_bytes();
    let mut ub = UnpackedBlock::zeroed();
    let mut isums = [0i32; GROUPS];
    for i in 0..m {
        let row = &w[i * row_bytes..(i + 1) * row_bytes];
        let mut acc = 0.0f32;
        for blk in 0..nb {
            unpack_block_i8(
                dtype,
                &row[blk * dtype.block_bytes()..(blk + 1) * dtype.block_bytes()],
                &mut ub,
            );
            group_sums(&ub.q, &acts.q[blk * QK_K..(blk + 1) * QK_K], &mut isums);
            for g in 0..GROUPS {
                combine_group(
                    &mut acc,
                    acts.scales[blk * GROUPS + g],
                    ub.d_eff[g],
                    ub.m_eff[g],
                    isums[g],
                    acts.gsums[blk * GROUPS + g],
                );
            }
        }
        out[i] = acc;
    }
    Ok(())
}

/// [`kquant_gemv_i8_on`] on the host's best INT8 tier
/// ([`CpuFeatures::best_int8_isa`]), falling back to the scalar-int8
/// reference when no SIMD INT8 tier exists (a within-CPU-backend tier
/// choice — every path computes bit-identical results here, so this is not
/// the FR-EX-08 cross-backend fallback).
pub fn kquant_gemv_i8(
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    w: &[u8],
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let isa = CpuFeatures::detect()
        .best_int8_isa()
        .unwrap_or(IsaPath::Scalar);
    kquant_gemv_i8_on(isa, dtype, m, k, w, x, out)
}

/// Two-activation K-quant INT8 GEMV (`x2 = [x0 | x1]`, `out[2i + c] =
/// row_i · x_c`) on a forced path — the SMMLA-native shape (M4-17-T16).
///
/// Accepted paths: `NeonI8mm` (2x2 SMMLA tiles), `NeonDotprod` and `Scalar`
/// (two single-x passes). All paths share the combine and exact integer
/// sums, so they are bit-identical to each other.
#[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))] // `nb` feeds the i8mm arm
pub fn kquant_gemv2_i8_on(
    isa: IsaPath,
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    w: &[u8],
    x2: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let nb = validate_gemv_i8(dtype, m, k, w, x2.len(), out.len(), 2)?;
    match isa {
        IsaPath::Scalar | IsaPath::NeonDotprod | IsaPath::Avx512Vnni | IsaPath::AvxVnni256 => {
            // Column-by-column through the single-x kernel (same combine and
            // integer sums per (row, column) as the tile kernel).
            let mut col = vec![0.0f32; m];
            for c in 0..2 {
                kquant_gemv_i8_on(isa, dtype, m, k, w, &x2[c * k..(c + 1) * k], &mut col)?;
                for i in 0..m {
                    out[2 * i + c] = col[i];
                }
            }
            Ok(())
        }
        IsaPath::NeonI8mm => {
            if !CpuFeatures::detect().supports(IsaPath::NeonI8mm) {
                return Err(VokraError::BackendUnavailable(format!(
                    "the {isa} kernel path is not available on this host CPU"
                )));
            }
            #[cfg(target_arch = "aarch64")]
            {
                gemv2_i8mm(dtype, m, k, nb, w, x2, out);
                Ok(())
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                unreachable!("NeonI8mm passed `supports` off aarch64")
            }
        }
        other => Err(VokraError::UnsupportedOp(format!(
            "no K-quant INT8 GEMV2 kernel on the {other} path (tiers: neon-i8mm | neon-dotprod | avx512vnni | avxvnni256 | scalar)"
        ))),
    }
}

/// Whether `isa` has the 2-activation SMMLA tile (the only path where
/// batching activations changes the instruction mix rather than just the
/// loop structure).
fn has_i8mm_tile(isa: IsaPath) -> bool {
    isa == IsaPath::NeonI8mm
}

/// Activation vectors consumed per SMMLA tile pass.
const I8MM_TILE_ACTS: usize = 2;

/// The single-vector tier an i8mm batch runs its **odd tail** on.
///
/// SMMLA consumes two activation vectors per pass, so the unpaired tail of an
/// odd batch has no 2x2 tile to fill and must run through the single-vector
/// kernel — which has no i8mm form at all: [`int8_group_sums_for`] has no
/// `NeonI8mm` arm, and [`CpuFeatures::best_int8_isa`] deliberately never
/// reports it ("i8mm serves the 2-activation matmul shape ... not this
/// selector"). Routing the tail on the requested `isa` therefore made every
/// odd `n_act` a hard `UnsupportedOp` on i8mm hosts.
///
/// Every accepted INT8 tier is bit-identical here (shared combine + exact
/// integer sums), so which dot-product tier the tail lands on cannot move a
/// result bit: this is a within-CPU-backend tier choice, not the FR-EX-08
/// cross-backend fallback. Kept separate from [`int8_tail_isa`] so the
/// invariant it encodes — an i8mm tail always maps to a tier the
/// single-vector kernel accepts — stays checkable on hosts without i8mm.
fn int8_tail_tier(isa: IsaPath) -> IsaPath {
    if isa == IsaPath::NeonI8mm {
        CpuFeatures::detect()
            .best_int8_isa()
            .unwrap_or(IsaPath::Scalar)
    } else {
        isa
    }
}

/// [`int8_tail_tier`] with the host gate on the **requested** path kept
/// intact.
///
/// A single-activation i8mm request (`n_act == 1`, or any batch after its
/// last pair) never reaches [`kquant_gemv2_i8_on`]'s own `supports` check, so
/// without this gate an i8mm request on a host that has no i8mm would be
/// quietly answered by another tier instead of erroring — a silent
/// capability substitution FR-EX-08 forbids.
fn int8_tail_isa(isa: IsaPath) -> Result<IsaPath> {
    if isa == IsaPath::NeonI8mm && !CpuFeatures::detect().supports(IsaPath::NeonI8mm) {
        return Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )));
    }
    Ok(int8_tail_tier(isa))
}

/// `n_act`-activation K-quant INT8 GEMV on a forced path (M5-15-T32) — the
/// generalization of [`kquant_gemv2_i8_on`] past its 2-activation ceiling.
///
/// `xn` holds `n_act` activation vectors of length `k` back to back
/// (`xn[c * k .. (c + 1) * k]`); `out[n_act * i + c] = row_i · x_c`, i.e. the
/// same `[m][n_act]` interleaved layout the 2-activation kernel already used.
///
/// **Activations are quantized per vector**: each `x_c` derives its own Q8
/// group scales, so no scale is ever shared across activation vectors and the
/// result is bit-for-bit `n_act` separate [`kquant_gemv_i8_on`] calls (ADR
/// `M5-15-quant.md` §D2 — the alternative, one scale set for the whole batch,
/// would change the numbers with the batch size). The SMMLA tier consumes two
/// vectors per pass and the odd tail through the single-vector kernel on
/// [`int8_tail_tier`] (SMMLA has no 1-activation form); since every accepted
/// path is bit-identical, tiling is a speed choice only.
///
/// # Errors
/// [`VokraError::InvalidArgument`] on `n_act == 0` or a shape mismatch;
/// [`VokraError::BackendUnavailable`] when `isa` is a path this host lacks;
/// whatever the underlying single/two-activation kernel returns for an
/// unsupported `isa`.
#[allow(clippy::too_many_arguments)] // GEMV operands + the forced isa + the batch width
pub fn kquant_gemvn_i8_on(
    isa: IsaPath,
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    n_act: usize,
    w: &[u8],
    xn: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_batch(dtype, m, k, n_act, w, xn.len(), out.len())?;
    let tail_isa = int8_tail_isa(isa)?;
    let mut pair: Vec<f32> = Vec::new();
    let mut col: Vec<f32> = Vec::new();
    let mut c = 0usize;
    while c < n_act {
        if has_i8mm_tile(isa) && n_act - c >= I8MM_TILE_ACTS {
            pair.resize(I8MM_TILE_ACTS * m, 0.0);
            kquant_gemv2_i8_on(
                isa,
                dtype,
                m,
                k,
                w,
                &xn[c * k..(c + I8MM_TILE_ACTS) * k],
                &mut pair,
            )?;
            for i in 0..m {
                out[n_act * i + c] = pair[I8MM_TILE_ACTS * i];
                out[n_act * i + c + 1] = pair[I8MM_TILE_ACTS * i + 1];
            }
            c += I8MM_TILE_ACTS;
        } else {
            col.resize(m, 0.0);
            kquant_gemv_i8_on(tail_isa, dtype, m, k, w, &xn[c * k..(c + 1) * k], &mut col)?;
            for i in 0..m {
                out[n_act * i + c] = col[i];
            }
            c += 1;
        }
    }
    Ok(())
}

/// Row-major K-quant INT8 **GEMM** on a forced path (M5-15-T31): the
/// `nn.Linear` shape, `out[t, i] = Σ_l dequant(w[i, l]) · x[t, l]`.
///
/// - `w` is the **untransposed** `[m, k]` weight exactly as GGUF stores it
///   (`m` = output features, `k` = input features, `k % 256 == 0`), so the
///   model layer keeps the quantized super-blocks verbatim and never pays the
///   `[out, in] → [in, out]` transpose the f32 path needs;
/// - `x` is `[n_act, k]` row-major (activations / tokens);
/// - `out` is `[n_act, m]` row-major — the layout `Compute::gemm_f32`
///   produces, so this is a drop-in for a quantized `nn.Linear`.
///
/// Equal, element for element, to `n_act` separate [`kquant_gemv_i8_on`]
/// calls (see [`kquant_gemvn_i8_on`] on the per-vector activation scales);
/// bounded against the f32 dequant GEMM by [`int8_error_bound`], **not**
/// bit-identical to it (`UnpackedBlock`'s doc: "the INT8 surface is bounded by
/// the activation-quantization band, not bit identity").
///
/// # Errors
/// [`VokraError::InvalidArgument`] on `n_act == 0` or a shape mismatch;
/// [`VokraError::BackendUnavailable`] when `isa` is a path this host lacks;
/// whatever the underlying kernel returns for an unrunnable `isa`. An odd
/// activation count is **not** an error on the SMMLA tier: the unpaired tail
/// runs on [`int8_tail_tier`], bit-identically.
#[allow(clippy::too_many_arguments)] // GEMM operands + the forced isa + the batch width
pub fn kquant_gemm_i8_on(
    isa: IsaPath,
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    n_act: usize,
    w: &[u8],
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_batch(dtype, m, k, n_act, w, x.len(), out.len())?;
    let tail_isa = int8_tail_isa(isa)?;
    let mut pair: Vec<f32> = Vec::new();
    let mut c = 0usize;
    while c < n_act {
        if has_i8mm_tile(isa) && n_act - c >= I8MM_TILE_ACTS {
            // The tile kernel writes `[m][2]`; scatter it into the two
            // (contiguous, disjoint) output rows.
            pair.resize(I8MM_TILE_ACTS * m, 0.0);
            kquant_gemv2_i8_on(
                isa,
                dtype,
                m,
                k,
                w,
                &x[c * k..(c + I8MM_TILE_ACTS) * k],
                &mut pair,
            )?;
            let (row0, row1) = out[c * m..(c + I8MM_TILE_ACTS) * m].split_at_mut(m);
            for i in 0..m {
                row0[i] = pair[I8MM_TILE_ACTS * i];
                row1[i] = pair[I8MM_TILE_ACTS * i + 1];
            }
            c += I8MM_TILE_ACTS;
        } else {
            // One activation writes a contiguous output row — no scratch and
            // no scatter at all on this path.
            kquant_gemv_i8_on(
                tail_isa,
                dtype,
                m,
                k,
                w,
                &x[c * k..(c + 1) * k],
                &mut out[c * m..(c + 1) * m],
            )?;
            c += 1;
        }
    }
    Ok(())
}

/// [`kquant_gemm_i8_on`] on the host's best INT8 tier.
///
/// Prefers the SMMLA tile when the host has i8mm **and** there are at least
/// two activation vectors to fill a 2x2 tile; otherwise the dot-product tier
/// [`CpuFeatures::best_int8_isa`] picks (which never reports `NeonI8mm`,
/// because the 1-activation GEMV cannot use a 2-row tile). Every accepted
/// tier is bit-identical, so this is a within-CPU-backend speed choice, not
/// the FR-EX-08 cross-backend fallback.
#[allow(clippy::too_many_arguments)] // GEMM operands + the batch width
pub fn kquant_gemm_i8(
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    n_act: usize,
    w: &[u8],
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let feats = CpuFeatures::detect();
    let isa = if n_act >= I8MM_TILE_ACTS && feats.supports(IsaPath::NeonI8mm) {
        IsaPath::NeonI8mm
    } else {
        feats.best_int8_isa().unwrap_or(IsaPath::Scalar)
    };
    kquant_gemm_i8_on(isa, dtype, m, k, n_act, w, x, out)
}

/// Shared shape validation for the batched entries. `n_act == 0` is rejected
/// explicitly: [`validate_gemv_i8`] would otherwise accept it (`0 * k == 0`
/// lengths line up) and the caller would get a silent no-op instead of an
/// error (FR-EX-08).
fn validate_batch(
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    n_act: usize,
    w: &[u8],
    x_len: usize,
    out_len: usize,
) -> Result<()> {
    if n_act == 0 {
        return Err(VokraError::InvalidArgument(
            "batched K-quant INT8 GEMV/GEMM needs at least one activation vector, got n_act = 0"
                .to_owned(),
        ));
    }
    validate_gemv_i8(dtype, m, k, w, x_len, out_len, n_act)?;
    Ok(())
}

/// SMMLA 2x2-tile implementation of the two-activation GEMV.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // group index drives four parallel arrays
fn gemv2_i8mm(
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    nb: usize,
    w: &[u8],
    x2: &[f32],
    out: &mut [f32],
) {
    let acts0 = quantize_activations(&x2[..k]);
    let acts1 = quantize_activations(&x2[k..]);
    let row_bytes = nb * dtype.block_bytes();
    let mut ub0 = UnpackedBlock::zeroed();
    let mut ub1 = UnpackedBlock::zeroed();
    let mut i = 0;
    // Row pairs through SMMLA.
    while i + 2 <= m {
        let r0 = &w[i * row_bytes..(i + 1) * row_bytes];
        let r1 = &w[(i + 1) * row_bytes..(i + 2) * row_bytes];
        let mut acc = [0.0f32; 4]; // [r0·x0, r0·x1, r1·x0, r1·x1]
        for blk in 0..nb {
            unpack_block_i8(
                dtype,
                &r0[blk * dtype.block_bytes()..(blk + 1) * dtype.block_bytes()],
                &mut ub0,
            );
            unpack_block_i8(
                dtype,
                &r1[blk * dtype.block_bytes()..(blk + 1) * dtype.block_bytes()],
                &mut ub1,
            );
            for g in 0..GROUPS {
                let e = blk * QK_K + KQUANT_GROUP * g;
                let a0: &[u8; 16] = ub0.q[KQUANT_GROUP * g..KQUANT_GROUP * (g + 1)]
                    .try_into()
                    .expect("group slice is 16 bytes");
                let a1: &[u8; 16] = ub1.q[KQUANT_GROUP * g..KQUANT_GROUP * (g + 1)]
                    .try_into()
                    .expect("group slice is 16 bytes");
                let x0: &[i8; 16] = acts0.q[e..e + 16].try_into().expect("16");
                let x1: &[i8; 16] = acts1.q[e..e + 16].try_into().expect("16");
                let mut tile = [0i32; 4];
                super::neon_i8mm::smmla_group_tile(a0, a1, x0, x1, &mut tile);
                let ga = blk * GROUPS + g;
                combine_group(
                    &mut acc[0],
                    acts0.scales[ga],
                    ub0.d_eff[g],
                    ub0.m_eff[g],
                    tile[0],
                    acts0.gsums[ga],
                );
                combine_group(
                    &mut acc[1],
                    acts1.scales[ga],
                    ub0.d_eff[g],
                    ub0.m_eff[g],
                    tile[1],
                    acts1.gsums[ga],
                );
                combine_group(
                    &mut acc[2],
                    acts0.scales[ga],
                    ub1.d_eff[g],
                    ub1.m_eff[g],
                    tile[2],
                    acts0.gsums[ga],
                );
                combine_group(
                    &mut acc[3],
                    acts1.scales[ga],
                    ub1.d_eff[g],
                    ub1.m_eff[g],
                    tile[3],
                    acts1.gsums[ga],
                );
            }
        }
        out[2 * i] = acc[0];
        out[2 * i + 1] = acc[1];
        out[2 * i + 2] = acc[2];
        out[2 * i + 3] = acc[3];
        i += 2;
    }
    // Odd tail row: scalar integer sums (exact, so still bit-identical).
    if i < m {
        let row = &w[i * row_bytes..(i + 1) * row_bytes];
        let mut isums = [0i32; GROUPS];
        for (c, acts) in [&acts0, &acts1].into_iter().enumerate() {
            let mut acc = 0.0f32;
            for blk in 0..nb {
                unpack_block_i8(
                    dtype,
                    &row[blk * dtype.block_bytes()..(blk + 1) * dtype.block_bytes()],
                    &mut ub0,
                );
                scalar_group_sums(&ub0.q, &acts.q[blk * QK_K..(blk + 1) * QK_K], &mut isums);
                for g in 0..GROUPS {
                    combine_group(
                        &mut acc,
                        acts.scales[blk * GROUPS + g],
                        ub0.d_eff[g],
                        ub0.m_eff[g],
                        isums[g],
                        acts.gsums[blk * GROUPS + g],
                    );
                }
            }
            out[2 * i + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------
// Reduced-precision matmuls (surface 3).
// ---------------------------------------------------------------------

fn validate_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], out: &[f32]) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(VokraError::InvalidArgument(format!(
            "reduced-precision GEMM shape mismatch: a={} (want {}), b={} (want {}), out={} (want {})",
            a.len(),
            m * k,
            b.len(),
            k * n,
            out.len(),
            m * n
        )));
    }
    Ok(())
}

/// Input-derived architectural error bound for a bf16 dot product of length
/// `k`: each input carries ≤ `2^-9` relative rounding (8-bit mantissa RNE),
/// so `|bf16_dot − f32_dot| ≤ Σ |a_l · b_l| · (2 · 2^-9 + 2^-9²) + ε_acc`.
/// Tests assert against `2 ×` this bound (ADR M4-17 §(f)); the same shape
/// serves fp16 with `2^-11` (10-bit mantissa + RNE).
pub fn dot_precision_bound(a_row: &[f32], b_col: &[f32], rel: f32) -> f32 {
    let mut s = 0.0f32;
    for (&x, &y) in a_row.iter().zip(b_col) {
        s += (x * y).abs();
    }
    s * (2.0 * rel + rel * rel) + f32::EPSILON * s * a_row.len() as f32
}

/// Relative input-rounding bound for bf16: 8 significand bits (7 explicit
/// mantissa + implicit leading 1), so RNE rounding is <= half of
/// `ulp = 2^-7`, i.e. `2^-8` relative.
pub const BF16_REL: f32 = 1.0 / 256.0; // 2^-8
/// Relative input-rounding bound for fp16: 11 significand bits, so RNE
/// rounding is <= half of `ulp = 2^-10`, i.e. `2^-11` relative.
pub const FP16_REL: f32 = 1.0 / 2048.0; // 2^-11

/// BF16 matmul `out[m,n] = bf16(a[m,k]) · bf16(b[k,n])` with f32
/// accumulation, on a forced path (M4-17-T11/T17). **Opt-in tier** — never
/// selected implicitly for f32-precision ops (ADR M4-17 §(b)-2).
///
/// - `Scalar` = the emulation oracle: inputs rounded to bf16 (RNE), products
///   exact in f32, sequential accumulation.
/// - `Avx512Bf16` = `vdpbf16ps`; `NeonBf16` = BFMMLA 2x2 tiles. Their exact
///   internal pair-rounding is NOT asserted (no local silicon — ADR M4-17
///   §(f)); parity uses the architectural band [`dot_precision_bound`].
pub fn gemm_bf16_on(
    isa: IsaPath,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_gemm(m, n, k, a, b, out)?;
    if !CpuFeatures::detect().supports(isa) {
        return Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )));
    }
    match isa {
        IsaPath::Scalar => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        // bf16 products are exact in f32 (8-bit x 8-bit
                        // significands); the sequential f32 accumulation is
                        // the reference semantics.
                        acc += bf16_to_f32(f32_to_bf16_rne(a[i * k + l]))
                            * bf16_to_f32(f32_to_bf16_rne(b[l * n + j]));
                    }
                    out[i * n + j] = acc;
                }
            }
            Ok(())
        }
        IsaPath::Avx512Bf16 => {
            #[cfg(target_arch = "x86_64")]
            {
                // Prepack: bf16 rows of `a` and columns of `b` (transposed),
                // zero-padded to whole 32-element zmm blocks (bf16 zero
                // products are exact zeros).
                let kp = k.next_multiple_of(32).max(32);
                let mut ap = vec![0u16; m * kp];
                for i in 0..m {
                    for l in 0..k {
                        ap[i * kp + l] = f32_to_bf16_rne(a[i * k + l]);
                    }
                }
                let mut btp = vec![0u16; n * kp];
                for j in 0..n {
                    for l in 0..k {
                        btp[j * kp + l] = f32_to_bf16_rne(b[l * n + j]);
                    }
                }
                for i in 0..m {
                    for j in 0..n {
                        out[i * n + j] = super::avx512::bf16_dot(
                            &ap[i * kp..(i + 1) * kp],
                            &btp[j * kp..(j + 1) * kp],
                        );
                    }
                }
                Ok(())
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!("Avx512Bf16 passed `supports` off x86-64")
            }
        }
        IsaPath::NeonBf16 => {
            #[cfg(target_arch = "aarch64")]
            {
                // Prepack 2x4 BFMMLA tiles over even-padded rows/cols and
                // 4-padded k (zero rows/cols/chunks contribute exact zeros).
                let kp = k.next_multiple_of(4).max(4);
                let k4 = kp / 4;
                let mp = m.next_multiple_of(2);
                let np = n.next_multiple_of(2);
                let bf = |v: f32| f32_to_bf16_rne(v);
                // a_tiles[pair][chunk]: [row0[4], row1[4]] interleaved.
                let mut a_tiles = vec![0u16; (mp / 2) * k4 * 8];
                for p in 0..mp / 2 {
                    for c in 0..k4 {
                        for h in 0..2 {
                            let i = 2 * p + h;
                            for t in 0..4 {
                                let l = 4 * c + t;
                                let v = if i < m && l < k { bf(a[i * k + l]) } else { 0 };
                                a_tiles[(p * k4 + c) * 8 + 4 * h + t] = v;
                            }
                        }
                    }
                }
                let mut b_tiles = vec![0u16; (np / 2) * k4 * 8];
                for p in 0..np / 2 {
                    for c in 0..k4 {
                        for h in 0..2 {
                            let j = 2 * p + h;
                            for t in 0..4 {
                                let l = 4 * c + t;
                                let v = if j < n && l < k { bf(b[l * n + j]) } else { 0 };
                                b_tiles[(p * k4 + c) * 8 + 4 * h + t] = v;
                            }
                        }
                    }
                }
                for pi in 0..mp / 2 {
                    for pj in 0..np / 2 {
                        let mut ctile = [0.0f32; 4];
                        super::neon_bf16::bfmmla_tile(
                            &mut ctile,
                            &a_tiles[pi * k4 * 8..(pi + 1) * k4 * 8],
                            &b_tiles[pj * k4 * 8..(pj + 1) * k4 * 8],
                            k4,
                        );
                        for h in 0..2 {
                            for c in 0..2 {
                                let (i, j) = (2 * pi + h, 2 * pj + c);
                                if i < m && j < n {
                                    out[i * n + j] = ctile[2 * h + c];
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                unreachable!("NeonBf16 passed `supports` off aarch64")
            }
        }
        other => Err(VokraError::UnsupportedOp(format!(
            "no BF16 matmul kernel on the {other} path (bf16 tiers: avx512bf16 | neon-bf16 | scalar)"
        ))),
    }
}

/// fp16 GEMM `out[m,n] = fp16(a) · fp16(b)` with a **genuine fp16 FMA
/// accumulator chain** (8-lane `fmla .8h` per column strip), on a forced
/// path (M4-17-T14). **Opt-in tier.**
///
/// - `Scalar` = the structurally identical emulation oracle
///   ([`fp16_fma_emu`] per lane, same order) — parity vs the NEON kernel is
///   a ±2 fp16-ulp band (rounding-mode / subnormal defensive margin, ADR
///   M4-17 §(f)).
/// - `NeonFp16` = the inline-asm FMLA kernel (executes on this Apple M1).
pub fn gemm_fp16_on(
    isa: IsaPath,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) -> Result<()> {
    validate_gemm(m, n, k, a, b, out)?;
    if !CpuFeatures::detect().supports(isa) {
        return Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )));
    }
    if !matches!(isa, IsaPath::Scalar | IsaPath::NeonFp16) {
        return Err(VokraError::UnsupportedOp(format!(
            "no fp16 GEMM kernel on the {isa} path (fp16 tiers: neon-fp16 | scalar; x86-64 fp16 compute is AMX-FP16 = v1.5+, out of M4-17 scope)"
        )));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        out.fill(0.0);
        return Ok(());
    }
    // Shared prep: fp16 `a` (m x k) and column-zero-padded fp16 `b` (k x np).
    let np = n.next_multiple_of(8);
    let af: Vec<u16> = a.iter().map(|&v| f32_to_f16_rne(v)).collect();
    let mut bf = vec![0u16; k * np];
    for l in 0..k {
        for j in 0..n {
            bf[l * np + j] = f32_to_f16_rne(b[l * n + j]);
        }
    }
    for i in 0..m {
        let a_col = &af[i * k..(i + 1) * k];
        let mut j = 0;
        while j < n {
            let mut acc = [0u16; 8];
            match isa {
                IsaPath::NeonFp16 => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        super::neon_fp16::fp16_fma_row_strip(
                            &mut acc,
                            a_col,
                            bf[j..].as_ptr(),
                            np,
                            k,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        unreachable!("NeonFp16 passed `supports` off aarch64")
                    }
                }
                IsaPath::Scalar => {
                    // Emulation oracle: identical lane/step order to the
                    // FMLA kernel.
                    for (l, &av) in a_col.iter().enumerate() {
                        for (lane, slot) in acc.iter_mut().enumerate() {
                            *slot = fp16_fma_emu(av, bf[l * np + j + lane], *slot);
                        }
                    }
                }
                _ => unreachable!("gated above"),
            }
            for (lane, &h) in acc.iter().enumerate() {
                if j + lane < n {
                    out[i * n + j + lane] = f16_to_f32(h);
                }
            }
            j += 8;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// SIMD dequant replays (bit-identical to the core scalar reference).
// ---------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{KQuantDtype, QK_K, q6_group_scales, q45_sub_scales};
    use core::arch::x86_64::*;

    pub(super) fn dequant(dtype: KQuantDtype, bytes: &[u8], n_elements: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n_elements];
        let bb = dtype.block_bytes();
        for (i, block) in bytes.chunks_exact(bb).enumerate() {
            let y = &mut out[i * QK_K..(i + 1) * QK_K];
            // SAFETY: every x86-64 tier's `supports` gate includes AVX2
            // (`kquant_dequant_on` checked it before dispatching here).
            unsafe {
                match dtype {
                    KQuantDtype::Q4K => dequant_q45_block(block, None, y),
                    KQuantDtype::Q5K => dequant_q45_block(block, Some(()), y),
                    KQuantDtype::Q6K => dequant_q6_block(block, y),
                }
            }
        }
        out
    }

    /// Widens the low 8 bytes of `q` and stores `d * q - m` (mul then sub —
    /// the exact core rounding sequence; never an FMA).
    ///
    /// # Safety
    /// Requires `avx2`; `out` must have 8 writable f32.
    #[target_feature(enable = "avx2")]
    unsafe fn dq8(q8: __m128i, dv: __m256, mv: __m256, out: *mut f32) {
        // SAFETY: caller contract (8 writable f32 at `out`; avx2 on).
        unsafe {
            let qf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q8));
            _mm256_storeu_ps(out, _mm256_sub_ps(_mm256_mul_ps(dv, qf), mv));
        }
    }

    /// Q4_K / Q5_K block replay (Q5 adds the `qh` 5th bit when `q5` is Some).
    ///
    /// # Safety
    /// Requires `avx2`; `block` is a whole 144/176-byte block, `y` 256 f32.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::needless_range_loop)] // sub-block index drives scale + layout math
    unsafe fn dequant_q45_block(block: &[u8], q5: Option<()>, y: &mut [f32]) {
        // SAFETY: all slice windows below are within the fixed block layout
        // (validated by the caller); stores target `y[32j..32j+32]`.
        unsafe {
            let subs = q45_sub_scales(block);
            let qs = if q5.is_some() {
                &block[48..176]
            } else {
                &block[16..144]
            };
            let qh = q5.map(|()| &block[16..48]);
            let low_mask = _mm_set1_epi8(0x0F);
            for j in 0..8 {
                let (d1, m1) = subs[j];
                let dv = _mm256_set1_ps(d1);
                let mv = _mm256_set1_ps(m1);
                let chunk = j / 2;
                let hi_nibble = j % 2 == 1;
                for half in 0..2 {
                    // 16 quant bytes -> 16 elements of sub-block j.
                    let raw = _mm_loadu_si128(qs[32 * chunk + 16 * half..].as_ptr() as *const _);
                    let mut q16 = if hi_nibble {
                        _mm_and_si128(_mm_srli_epi16(raw, 4), low_mask)
                    } else {
                        _mm_and_si128(raw, low_mask)
                    };
                    if let Some(qh) = qh {
                        let hb = _mm_loadu_si128(qh[16 * half..].as_ptr() as *const _);
                        let bit = _mm_and_si128(hb, _mm_set1_epi8(1 << j));
                        let is_zero = _mm_cmpeq_epi8(bit, _mm_setzero_si128());
                        let add16 = _mm_andnot_si128(is_zero, _mm_set1_epi8(16));
                        q16 = _mm_add_epi8(q16, add16);
                    }
                    let base = 32 * j + 16 * half;
                    dq8(q16, dv, mv, y[base..].as_mut_ptr());
                    dq8(_mm_srli_si128(q16, 8), dv, mv, y[base + 8..].as_mut_ptr());
                }
            }
        }
    }

    /// Widens the low 8 bytes of `q` (biased 0..=63), subtracts 32 in i32,
    /// and stores `d * q` (single multiply — the exact core sequence for
    /// Q6_K's `d_eff * q_signed`).
    ///
    /// # Safety
    /// Requires `avx2`; `out` must have 8 writable f32.
    #[target_feature(enable = "avx2")]
    unsafe fn dq8_q6(q8: __m128i, dv: __m256, out: *mut f32) {
        // SAFETY: caller contract (8 writable f32 at `out`; avx2 on).
        unsafe {
            let qi = _mm256_sub_epi32(_mm256_cvtepu8_epi32(q8), _mm256_set1_epi32(32));
            _mm256_storeu_ps(out, _mm256_mul_ps(dv, _mm256_cvtepi32_ps(qi)));
        }
    }

    /// Q6_K block replay.
    ///
    /// # Safety
    /// Requires `avx2`; `block` is a whole 210-byte block, `y` 256 f32.
    #[target_feature(enable = "avx2")]
    unsafe fn dequant_q6_block(block: &[u8], y: &mut [f32]) {
        // SAFETY: fixed-layout slice windows; stores per 16-element group.
        unsafe {
            let dg = q6_group_scales(block);
            let ql_all = &block[0..128];
            let qh_all = &block[128..192];
            let low_mask = _mm_set1_epi8(0x0F);
            let two_mask = _mm_set1_epi8(0x30u8 as i8);
            for half in 0..2 {
                let ql = &ql_all[half * 64..];
                let qh = &qh_all[half * 32..];
                for sub in 0..2 {
                    // 16-byte window `l = 16*sub .. 16*sub+16` of the half.
                    let l0 = 16 * sub;
                    let qla = _mm_loadu_si128(ql[l0..].as_ptr() as *const _);
                    let qlb = _mm_loadu_si128(ql[l0 + 32..].as_ptr() as *const _);
                    let qhv = _mm_loadu_si128(qh[l0..].as_ptr() as *const _);
                    // Quarter 0: (ql & 0xF) | ((qh & 3) << 4)
                    let q0 = _mm_or_si128(
                        _mm_and_si128(qla, low_mask),
                        _mm_and_si128(_mm_slli_epi16(qhv, 4), two_mask),
                    );
                    // Quarter 1: (ql[l+32] & 0xF) | (((qh >> 2) & 3) << 4)
                    let q1 = _mm_or_si128(
                        _mm_and_si128(qlb, low_mask),
                        _mm_and_si128(_mm_slli_epi16(qhv, 2), two_mask),
                    );
                    // Quarter 2: (ql >> 4) | (((qh >> 4) & 3) << 4)
                    let q2 = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(qla, 4), low_mask),
                        _mm_and_si128(qhv, two_mask),
                    );
                    // Quarter 3: (ql[l+32] >> 4) | (((qh >> 6) & 3) << 4)
                    let q3 = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(qlb, 4), low_mask),
                        _mm_and_si128(_mm_srli_epi16(qhv, 2), two_mask),
                    );
                    for (quarter, qv) in [q0, q1, q2, q3].into_iter().enumerate() {
                        let base = 128 * half + 32 * quarter + l0;
                        let dv = _mm256_set1_ps(dg[base / 16]);
                        dq8_q6(qv, dv, y[base..].as_mut_ptr());
                        let dv2 = _mm256_set1_ps(dg[(base + 8) / 16]);
                        dq8_q6(_mm_srli_si128(qv, 8), dv2, y[base + 8..].as_mut_ptr());
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{KQuantDtype, QK_K, q6_group_scales, q45_sub_scales};
    use core::arch::aarch64::*;

    pub(super) fn dequant(dtype: KQuantDtype, bytes: &[u8], n_elements: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n_elements];
        let bb = dtype.block_bytes();
        for (i, block) in bytes.chunks_exact(bb).enumerate() {
            let y = &mut out[i * QK_K..(i + 1) * QK_K];
            // SAFETY: NEON is the AArch64 baseline (every aarch64 tier's
            // `supports` gate includes `neon`).
            unsafe {
                match dtype {
                    KQuantDtype::Q4K => dequant_q45_block(block, false, y),
                    KQuantDtype::Q5K => dequant_q45_block(block, true, y),
                    KQuantDtype::Q6K => dequant_q6_block(block, y),
                }
            }
        }
        out
    }

    /// Widens 16 u8 lanes and stores `d * q - m` per lane (mul then sub —
    /// the exact core rounding sequence; never an FMA).
    ///
    /// # Safety
    /// Requires `neon`; `out` must have 16 writable f32.
    #[target_feature(enable = "neon")]
    unsafe fn dq16(q: uint8x16_t, d: f32, m: f32, out: *mut f32) {
        // SAFETY: caller contract (16 writable f32 at `out`; NEON baseline).
        unsafe {
            let dv = vdupq_n_f32(d);
            let mv = vdupq_n_f32(m);
            let lo = vmovl_u8(vget_low_u8(q));
            let hi = vmovl_u8(vget_high_u8(q));
            let f = [
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))),
            ];
            for (t, fv) in f.into_iter().enumerate() {
                vst1q_f32(out.add(4 * t), vsubq_f32(vmulq_f32(dv, fv), mv));
            }
        }
    }

    /// # Safety
    /// Requires `neon`; `block` is a whole 144/176-byte block, `y` 256 f32.
    #[target_feature(enable = "neon")]
    unsafe fn dequant_q45_block(block: &[u8], q5: bool, y: &mut [f32]) {
        // SAFETY: fixed-layout slice windows; stores per 16-element run.
        unsafe {
            let subs = q45_sub_scales(block);
            let qs = if q5 { &block[48..176] } else { &block[16..144] };
            let low_mask = vdupq_n_u8(0x0F);
            for j in 0..8 {
                let (d1, m1) = subs[j];
                let chunk = j / 2;
                let hi_nibble = j % 2 == 1;
                for half in 0..2 {
                    let raw = vld1q_u8(qs[32 * chunk + 16 * half..].as_ptr());
                    let mut q16 = if hi_nibble {
                        vshrq_n_u8(raw, 4)
                    } else {
                        vandq_u8(raw, low_mask)
                    };
                    if q5 {
                        let qh = vld1q_u8(block[16 + 16 * half..].as_ptr());
                        let bit = vtstq_u8(qh, vdupq_n_u8(1 << j));
                        q16 = vaddq_u8(q16, vandq_u8(bit, vdupq_n_u8(16)));
                    }
                    dq16(q16, d1, m1, y[32 * j + 16 * half..].as_mut_ptr());
                }
            }
        }
    }

    /// Widens 16 biased u8 lanes (0..=63), subtracts 32 in i16, and stores
    /// `d * q` per lane (single multiply — the exact core sequence).
    ///
    /// # Safety
    /// Requires `neon`; `out` must have 16 writable f32; the two `d` values
    /// cover lanes 0..8 and 8..16 — callers pass the same value twice when
    /// the group spans a single scale.
    #[target_feature(enable = "neon")]
    unsafe fn dq16_q6(q: uint8x16_t, d: f32, out: *mut f32) {
        // SAFETY: caller contract (16 writable f32 at `out`).
        unsafe {
            let dv = vdupq_n_f32(d);
            let s16lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(q)));
            let s16hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(q)));
            let bias = vdupq_n_s16(32);
            let alo = vsubq_s16(s16lo, bias);
            let ahi = vsubq_s16(s16hi, bias);
            let f = [
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(alo))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(alo))),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(ahi))),
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(ahi))),
            ];
            for (t, fv) in f.into_iter().enumerate() {
                vst1q_f32(out.add(4 * t), vmulq_f32(dv, fv));
            }
        }
    }

    /// # Safety
    /// Requires `neon`; `block` is a whole 210-byte block, `y` 256 f32.
    #[target_feature(enable = "neon")]
    unsafe fn dequant_q6_block(block: &[u8], y: &mut [f32]) {
        // SAFETY: fixed-layout slice windows; stores per 16-element group.
        unsafe {
            let dg = q6_group_scales(block);
            let low_mask = vdupq_n_u8(0x0F);
            let two = vdupq_n_u8(3);
            for half in 0..2 {
                let ql = &block[half * 64..];
                let qh = &block[128 + half * 32..];
                for sub in 0..2 {
                    let l0 = 16 * sub;
                    let qla = vld1q_u8(ql[l0..].as_ptr());
                    let qlb = vld1q_u8(ql[l0 + 32..].as_ptr());
                    let qhv = vld1q_u8(qh[l0..].as_ptr());
                    let q0 = vorrq_u8(vandq_u8(qla, low_mask), vshlq_n_u8(vandq_u8(qhv, two), 4));
                    let q1 = vorrq_u8(
                        vandq_u8(qlb, low_mask),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), two), 4),
                    );
                    let q2 = vorrq_u8(
                        vshrq_n_u8(qla, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), two), 4),
                    );
                    let q3 = vorrq_u8(vshrq_n_u8(qlb, 4), vshlq_n_u8(vshrq_n_u8(qhv, 6), 4));
                    for (quarter, qv) in [q0, q1, q2, q3].into_iter().enumerate() {
                        let base = 128 * half + 32 * quarter + l0;
                        // A 16-byte run spans exactly one 16-element scale
                        // group (base is 16-aligned).
                        dq16_q6(qv, dg[base / 16], y[base..].as_mut_ptr());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::IsaPath;

    /// Deterministic PRNG (xorshift64*), no external crate (NFR-DS-02).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_f32(&mut self) -> f32 {
            ((self.next_u64() >> 40) as u32) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| (self.next_u64() >> 32) as u8).collect()
        }
        fn vecf(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    /// Random-but-structurally-sane K-quant payloads: random bytes ARE valid
    /// payloads for all three formats (every bit pattern decodes), except we
    /// cap the f16 scales to keep magnitudes finite and away from f16
    /// inf/NaN patterns.
    fn random_blocks(rng: &mut Rng, dtype: KQuantDtype, nb: usize) -> Vec<u8> {
        let mut bytes = rng.bytes(nb * dtype.block_bytes());
        for b in 0..nb {
            let base = b * dtype.block_bytes();
            let (d_off, dmin_off) = match dtype {
                KQuantDtype::Q4K | KQuantDtype::Q5K => (base, Some(base + 2)),
                KQuantDtype::Q6K => (base + 208, None),
            };
            // Small positive normal f16 scales (0x2C00 = ~0.0625 with random
            // low mantissa bits) — finite, well-conditioned.
            bytes[d_off] = bytes[d_off].wrapping_mul(31);
            bytes[d_off + 1] = 0x2C;
            if let Some(o) = dmin_off {
                bytes[o] = bytes[o].wrapping_mul(17);
                bytes[o + 1] = 0x24;
            }
        }
        bytes
    }

    #[test]
    fn f16_roundtrip_is_exhaustive_identity() {
        // Every finite/inf f16 bit pattern must survive f16 -> f32 -> f16.
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(
                    f16_to_f32(f32_to_f16_rne(f)).is_nan(),
                    "NaN class lost for {h:#06x}"
                );
                continue;
            }
            assert_eq!(
                f32_to_f16_rne(f),
                h,
                "roundtrip failed for {h:#06x} (value {f})"
            );
        }
    }

    #[test]
    fn f16_rounding_picks_nearest_even() {
        // 1.0 + 2^-11 sits exactly between 1.0 (0x3C00) and 1.0009766
        // (0x3C01): RNE picks the even mantissa (0x3C00).
        assert_eq!(f64_to_f16_rne(1.0 + 2f64.powi(-11)), 0x3C00);
        // 1.0 + 3*2^-11 ties between 0x3C01 and 0x3C02 -> even 0x3C02.
        assert_eq!(f64_to_f16_rne(1.0 + 3.0 * 2f64.powi(-11)), 0x3C02);
        // Overflow saturates to inf; halfway-to-inf rounds to inf.
        assert_eq!(f64_to_f16_rne(65520.0), 0x7C00);
        assert_eq!(f64_to_f16_rne(65519.9), 0x7BFF);
        // Subnormal quantum.
        assert_eq!(f64_to_f16_rne(2f64.powi(-24)), 0x0001);
        assert_eq!(f64_to_f16_rne(2f64.powi(-25)), 0x0000); // tie to even 0
    }

    #[test]
    fn bf16_conversion_matches_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(f32_to_bf16_rne(1.0), 0x3F80);
        assert_eq!(f32_to_bf16_rne(-2.0), 0xC000);
        // bf16 ulp(1.0) = 2^-7 (8 significand bits). 1 + 2^-8 is an exact
        // tie -> RNE picks the even mantissa (1.0); 1 + 3*2^-8 ties between
        // odd 0x3F81 and even 0x3F82 -> 0x3F82.
        assert_eq!(f32_to_bf16_rne(1.0 + 2f32.powi(-8)), 0x3F80);
        assert_eq!(f32_to_bf16_rne(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        // A 0.75-ulp offset is NOT a tie: it rounds to the nearer 0x3F81.
        assert_eq!(f32_to_bf16_rne(1.0 + 3.0 * 2f32.powi(-9)), 0x3F81);
        assert!(bf16_to_f32(f32_to_bf16_rne(f32::NAN)).is_nan());
    }

    #[test]
    fn unified_unpack_reconstructs_core_dequant() {
        let mut rng = Rng::new(0xD00D_F00D);
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
            let bytes = random_blocks(&mut rng, dtype, 2);
            let oracle = core_quant::dequantize(dtype.ggml(), &bytes, 512).unwrap();
            let mut ub = UnpackedBlock::zeroed();
            for blk in 0..2 {
                unpack_block_i8(
                    dtype,
                    &bytes[blk * dtype.block_bytes()..(blk + 1) * dtype.block_bytes()],
                    &mut ub,
                );
                for t in 0..QK_K {
                    let g = t / KQUANT_GROUP;
                    let got = ub.d_eff[g] * f32::from(ub.q[t]) - ub.m_eff[g];
                    let want = oracle[blk * QK_K + t];
                    let tol = match dtype {
                        // Q4/Q5: identical expression -> bit-exact.
                        KQuantDtype::Q4K | KQuantDtype::Q5K => 0.0,
                        // Q6: d*(q+32) - d*32 vs d*q — one extra f32
                        // rounding; bound by an ulp of the subtrahend scale.
                        KQuantDtype::Q6K => (ub.d_eff[g] * 32.0).abs() * f32::EPSILON * 2.0,
                    };
                    assert!(
                        (got - want).abs() <= tol,
                        "{dtype:?} unified unpack mismatch at {t}: got {got}, want {want} (tol {tol})"
                    );
                }
            }
        }
    }

    #[test]
    fn q8_quantization_error_is_within_half_scale() {
        let mut rng = Rng::new(0xACED_5EED);
        let x = rng.vecf(64);
        let acts = quantize_activations(&x);
        for g in 0..4 {
            let s = acts.scales[g];
            let mut gsum = 0i32;
            for t in 0..16 {
                let q = acts.q[16 * g + t];
                gsum += i32::from(q);
                let rec = s * f32::from(q);
                assert!(
                    (rec - x[16 * g + t]).abs() <= s * 0.5 + 1e-7,
                    "q8 rounding beyond half-scale at ({g},{t})"
                );
            }
            assert_eq!(gsum, acts.gsums[g], "gsum bookkeeping");
        }
        // All-zero group stays zero with zero scale (no NaN path).
        let z = quantize_activations(&[0.0f32; 16]);
        assert_eq!(z.scales[0], 0.0);
        assert!(z.q.iter().all(|&q| q == 0));
    }

    #[test]
    fn scalar_int8_gemv_tracks_f32_dequant_within_derived_bound() {
        let mut rng = Rng::new(0xBEEF_CAFE);
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
            let (m, k) = (3usize, 512usize);
            let nb = k / QK_K;
            let w: Vec<u8> = (0..m)
                .flat_map(|_| random_blocks(&mut rng, dtype, nb))
                .collect();
            let x = rng.vecf(k);
            let mut got = vec![0.0f32; m];
            kquant_gemv_i8_on(IsaPath::Scalar, dtype, m, k, &w, &x, &mut got).unwrap();

            let row_bytes = nb * dtype.block_bytes();
            for i in 0..m {
                let row = &w[i * row_bytes..(i + 1) * row_bytes];
                let wf = core_quant::dequantize(dtype.ggml(), row, k).unwrap();
                let want: f32 = wf.iter().zip(&x).map(|(&a, &b)| a * b).sum();
                let bound = int8_error_bound(dtype, row, &x);
                assert!(
                    (got[i] - want).abs() <= 2.0 * bound.max(1e-6),
                    "{dtype:?} row {i}: int8 {} vs f32 {} exceeds 2x derived bound {bound}",
                    got[i],
                    want
                );
            }
        }
    }

    #[test]
    fn int8_gemv_rejects_malformed_shapes_and_foreign_paths() {
        let w = vec![0u8; KQuantDtype::Q4K.block_bytes()];
        let x = vec![0.0f32; QK_K];
        let mut out = vec![0.0f32; 1];
        // k not a multiple of 256.
        assert!(matches!(
            kquant_gemv_i8_on(IsaPath::Scalar, KQuantDtype::Q4K, 1, 100, &w, &x, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        // A path with no INT8 kernel is an explicit UnsupportedOp (FR-EX-08
        // spirit — never a silent reroute).
        let host_f32 = CpuFeatures::detect().best_isa();
        if matches!(host_f32, IsaPath::Avx2 | IsaPath::Neon) {
            assert!(matches!(
                kquant_gemv_i8_on(host_f32, KQuantDtype::Q4K, 1, QK_K, &w, &x, &mut out),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
        // A path this host cannot run is BackendUnavailable.
        let feats = CpuFeatures::detect();
        for isa in [
            IsaPath::Avx512Vnni,
            IsaPath::AvxVnni256,
            IsaPath::NeonDotprod,
        ] {
            if !feats.supports(isa) {
                assert!(matches!(
                    kquant_gemv_i8_on(isa, KQuantDtype::Q4K, 1, QK_K, &w, &x, &mut out),
                    Err(VokraError::BackendUnavailable(_))
                ));
            }
        }
    }

    /// M5-15 regression: the **odd-activation tail** of an i8mm batch must
    /// land on a tier the single-vector kernel actually implements.
    ///
    /// `int8_group_sums_for` has no `NeonI8mm` arm (SMMLA has no 1-activation
    /// form), so passing the caller's `isa` straight through made every odd
    /// `n_act` on an i8mm host a hard `UnsupportedOp`. This invariant is
    /// host-independent — it fails on *any* CPU if the mapping is identity —
    /// which is what lets a machine without i8mm pin the fix.
    #[test]
    fn i8mm_tail_tier_is_runnable_by_the_single_vector_kernel() {
        let tail = int8_tail_tier(IsaPath::NeonI8mm);
        assert_ne!(
            tail,
            IsaPath::NeonI8mm,
            "SMMLA has no single-vector form; the tail must move to a dot tier"
        );
        assert!(
            int8_group_sums_for(tail).is_ok(),
            "an i8mm tail maps to {tail}, which the single-vector kernel rejects"
        );
        // Every other tier passes through untouched (no reroute at all).
        for isa in [
            IsaPath::Scalar,
            IsaPath::Avx512Vnni,
            IsaPath::AvxVnni256,
            IsaPath::NeonDotprod,
        ] {
            assert_eq!(int8_tail_tier(isa), isa, "{isa} must not be rerouted");
        }
    }

    /// FR-EX-08: the tail reroute must not paper over a host that has no i8mm
    /// at all. `n_act == 1` never reaches the tile kernel's own `supports`
    /// check, so the batched entries gate the requested path themselves —
    /// otherwise an i8mm request would be silently served by another tier.
    #[test]
    fn i8mm_batch_on_a_host_without_i8mm_is_backend_unavailable() {
        if CpuFeatures::detect().supports(IsaPath::NeonI8mm) {
            return; // i8mm host: the odd-tail e2e lives in kquant_gemm_parity.rs
        }
        let dtype = KQuantDtype::Q4K;
        let w = vec![0u8; dtype.block_bytes()];
        // n_act = 1 has no tile pass at all; 2 and 3 also exercise the gate
        // before any work is done.
        for n_act in [1usize, 2, 3] {
            let x = vec![0.0f32; n_act * QK_K];
            let mut out = vec![0.0f32; n_act];
            assert!(
                matches!(
                    kquant_gemvn_i8_on(IsaPath::NeonI8mm, dtype, 1, QK_K, n_act, &w, &x, &mut out),
                    Err(VokraError::BackendUnavailable(_))
                ),
                "gemvn(n_act={n_act}) on an i8mm-less host must be BackendUnavailable"
            );
            assert!(
                matches!(
                    kquant_gemm_i8_on(IsaPath::NeonI8mm, dtype, 1, QK_K, n_act, &w, &x, &mut out),
                    Err(VokraError::BackendUnavailable(_))
                ),
                "gemm(n_act={n_act}) on an i8mm-less host must be BackendUnavailable"
            );
        }
    }

    #[test]
    fn bf16_scalar_emulation_tracks_f32_within_architectural_bound() {
        let mut rng = Rng::new(0x0B16_0B16);
        let (m, n, k) = (3usize, 5usize, 64usize);
        let a = rng.vecf(m * k);
        let b = rng.vecf(k * n);
        let mut got = vec![0.0f32; m * n];
        gemm_bf16_on(IsaPath::Scalar, m, n, k, &a, &b, &mut got).unwrap();
        for i in 0..m {
            for j in 0..n {
                let a_row: Vec<f32> = (0..k).map(|l| a[i * k + l]).collect();
                let b_col: Vec<f32> = (0..k).map(|l| b[l * n + j]).collect();
                let want: f32 = a_row.iter().zip(&b_col).map(|(&x, &y)| x * y).sum();
                let bound = dot_precision_bound(&a_row, &b_col, BF16_REL);
                assert!(
                    (got[i * n + j] - want).abs() <= 2.0 * bound,
                    "bf16 emulation ({i},{j}): {} vs {} exceeds 2x bound {bound}",
                    got[i * n + j],
                    want
                );
            }
        }
    }

    #[test]
    fn fp16_scalar_emulation_tracks_f32_within_architectural_bound() {
        let mut rng = Rng::new(0xF9_16F9);
        let (m, n, k) = (2usize, 9usize, 24usize); // n=9 exercises the lane tail
        let a = rng.vecf(m * k);
        let b = rng.vecf(k * n);
        let mut got = vec![0.0f32; m * n];
        gemm_fp16_on(IsaPath::Scalar, m, n, k, &a, &b, &mut got).unwrap();
        for i in 0..m {
            for j in 0..n {
                let a_row: Vec<f32> = (0..k).map(|l| a[i * k + l]).collect();
                let b_col: Vec<f32> = (0..k).map(|l| b[l * n + j]).collect();
                let want: f32 = a_row.iter().zip(&b_col).map(|(&x, &y)| x * y).sum();
                // fp16 additionally rounds the running accumulator each
                // step: bound by input rounding + k * ulp(acc) growth.
                let scale = a_row
                    .iter()
                    .zip(&b_col)
                    .map(|(&x, &y)| (x * y).abs())
                    .sum::<f32>();
                let bound = dot_precision_bound(&a_row, &b_col, FP16_REL)
                    + scale * FP16_REL * 2.0 * k as f32 / 8.0;
                assert!(
                    (got[i * n + j] - want).abs() <= 2.0 * bound,
                    "fp16 emulation ({i},{j}): {} vs {} exceeds 2x bound {bound}",
                    got[i * n + j],
                    want
                );
            }
        }
    }

    #[test]
    fn reduced_precision_gemms_reject_foreign_paths_explicitly() {
        let a = [1.0f32; 4];
        let b = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        // f32 tiers have no bf16/fp16 kernel -> explicit UnsupportedOp.
        let host_f32 = CpuFeatures::detect().best_isa();
        if matches!(host_f32, IsaPath::Avx2 | IsaPath::Neon) {
            assert!(matches!(
                gemm_bf16_on(host_f32, 2, 2, 2, &a, &b, &mut out),
                Err(VokraError::UnsupportedOp(_))
            ));
            assert!(matches!(
                gemm_fp16_on(host_f32, 2, 2, 2, &a, &b, &mut out),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
        // Unsupported-on-host tiers are BackendUnavailable.
        let feats = CpuFeatures::detect();
        for isa in [IsaPath::Avx512Bf16, IsaPath::NeonBf16] {
            if !feats.supports(isa) {
                assert!(matches!(
                    gemm_bf16_on(isa, 2, 2, 2, &a, &b, &mut out),
                    Err(VokraError::BackendUnavailable(_))
                ));
            }
        }
        if !feats.supports(IsaPath::NeonFp16) {
            assert!(matches!(
                gemm_fp16_on(IsaPath::NeonFp16, 2, 2, 2, &a, &b, &mut out),
                Err(VokraError::BackendUnavailable(_))
            ));
        }
    }

    #[test]
    fn dequant_on_scalar_equals_core_reference() {
        let mut rng = Rng::new(0xDEAD_10CC);
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
            let bytes = random_blocks(&mut rng, dtype, 3);
            let got = kquant_dequant_on(IsaPath::Scalar, dtype, &bytes, 768).unwrap();
            let want = core_quant::dequantize(dtype.ggml(), &bytes, 768).unwrap();
            assert_eq!(got, want, "{dtype:?} scalar dequant must be the oracle");
        }
    }

    #[test]
    fn dequant_on_host_simd_is_bit_identical_to_core() {
        // The T12 contract: on whatever SIMD family this host supports, the
        // fused dequant must be bit-identical (atol = 0.0) to vokra-core.
        let mut rng = Rng::new(0x51D0_51D0);
        let isa = CpuFeatures::detect().best_isa();
        if isa == IsaPath::Scalar || matches!(isa, IsaPath::Rvv | IsaPath::WasmSimd128) {
            eprintln!("skip: no M4-17 SIMD dequant family on this host ({isa})");
            return;
        }
        for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
            let bytes = random_blocks(&mut rng, dtype, 4);
            let got = kquant_dequant_on(isa, dtype, &bytes, 1024).unwrap();
            let want = core_quant::dequantize(dtype.ggml(), &bytes, 1024).unwrap();
            assert_eq!(
                got, want,
                "{dtype:?} SIMD dequant on {isa} must be bit-identical to the core reference"
            );
        }
    }

    #[test]
    fn dequant_rejects_bad_payload_and_rvv_wasm_paths() {
        let bytes = vec![0u8; 10];
        assert!(matches!(
            kquant_dequant_on(IsaPath::Scalar, KQuantDtype::Q4K, &bytes, 256),
            Err(VokraError::InvalidArgument(_))
        ));
        let ok_bytes = vec![0u8; 144];
        assert!(matches!(
            kquant_dequant_on(IsaPath::Rvv, KQuantDtype::Q4K, &ok_bytes, 256),
            // Rvv is not runnable on this host at all -> BackendUnavailable
            // (host gate fires before the kernel-coverage gate).
            Err(VokraError::BackendUnavailable(_)) | Err(VokraError::UnsupportedOp(_))
        ));
    }
}
