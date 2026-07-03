//! Recursive mixed-radix complex FFT core (M0-04-T03, T04).
//!
//! This is a from-scratch Rust reimplementation of the mixed-radix
//! Cooley-Tukey algorithm used by pocketfft's `cfftp` (M. Reinecke,
//! Max-Planck-Society, BSD-3-Clause — see
//! `THIRD_PARTY_LICENSES/pocketfft-LICENSE.txt`). It is *not* a line-by-line
//! transliteration of pocketfft's iterative `pass{2,3,4,5}` / `passg` kernels:
//! it uses the equivalent recursive decimation-in-time formulation, which is
//! shorter and easier to verify while computing the identical transform.
//!
//! ## Radix coverage (maps onto the M0-04 tickets)
//!
//! The length is factorized (see [`factorize`]) preferring radix `4` for
//! power-of-two lengths, exactly like pocketfft's `cfftp_factorize`. A single
//! combine step ([`combine`]) then acts as the radix-`p` butterfly for whatever
//! factor `p` the level carries:
//!
//! - `p ∈ {4, 2}` — the power-of-two path (M0-04-T03);
//! - `p ∈ {3, 5, …}` and other small primes — composite-length support
//!   (M0-04-T04);
//! - a large prime factor is handled one level up by Bluestein
//!   ([`super::bluestein`], M0-04-T05) so this core never runs an `O(p²)`
//!   combine for a big `p`.
//!
//! Specialized straight-line radix-3/5 butterflies (pocketfft `pass3`/`pass5`)
//! are a performance follow-up; the generic [`combine`] already produces the
//! correct result for those factors and is validated against Bluestein for
//! prime lengths in the test suite.
//!
//! Hot-path allocation: the combine step allocates one scratch buffer per
//! recursion node. Removing per-call allocation is FR-EX-05 (M1); the plan /
//! recursion structure is arranged so that optimization does not require an
//! API change.

use vokra_core::Complex32;

/// Read-only context threaded through the recursion so the recursive function
/// stays within the clippy argument budget.
pub(crate) struct FftCtx<'a> {
    /// The transform input (length equals the top transform length).
    pub input: &'a [Complex32],
    /// Forward root table of the top length (`e^{-2πi t / big_n}`).
    pub tw: &'a [Complex32],
    /// Top transform length; every recursion sub-length divides it.
    pub big_n: usize,
    /// When `true`, conjugate the roots to compute the inverse (unnormalized)
    /// transform.
    pub inverse: bool,
}

/// Factorizes `n` into radices, preferring `4` then `2` then increasing odd
/// primes — the ordering used by pocketfft `cfftp_factorize`.
///
/// The product of the returned factors equals `n` (empty for `n == 1`).
pub(crate) fn factorize(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n % 4 == 0 {
        factors.push(4);
        n /= 4;
    }
    if n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }
    let mut d = 3;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// The largest factor produced by [`factorize`] (`1` for `n <= 1`).
pub(crate) fn largest_factor(n: usize) -> usize {
    factorize(n).into_iter().max().unwrap_or(1)
}

/// Recursively evaluates the unnormalized length-`output.len()` DFT of the
/// sub-sequence `ctx.input[offset], ctx.input[offset+stride], …`, writing the
/// result into `output`.
///
/// `factors` is the factorization of `output.len()` (see [`factorize`]), peeled
/// from the front one radix per level.
pub(crate) fn fft_rec(
    ctx: &FftCtx<'_>,
    offset: usize,
    stride: usize,
    output: &mut [Complex32],
    factors: &[usize],
) {
    let n = output.len();
    if n == 1 {
        output[0] = ctx.input[offset];
        return;
    }

    let p = factors[0];
    let m = n / p;
    let rest = &factors[1..];

    // p sub-transforms of length m, decimating the input by p.
    for j in 0..p {
        let sub = &mut output[j * m..(j + 1) * m];
        fft_rec(ctx, offset + j * stride, stride * p, sub, rest);
    }

    combine(ctx, output, n, p, m);
}

/// Radix-`p` butterfly combine: turns the `p` length-`m` sub-DFTs stored
/// contiguously in `output` into the length-`n = p·m` DFT.
///
/// Implements `X[k] = Σ_{j<p} S_j[k mod m] · e^{∓2πi·jk/n}`, the general
/// mixed-radix recombination (the `p = 2` case is the classic butterfly). The
/// twiddle `e^{∓2πi·jk/n}` is read from the top-level table with the level
/// stride `big_n / n`; the inverse conjugates it.
fn combine(ctx: &FftCtx<'_>, output: &mut [Complex32], n: usize, p: usize, m: usize) {
    let level_stride = ctx.big_n / n;
    let mut scratch = vec![Complex32::ZERO; n];
    for (k, out_k) in scratch.iter_mut().enumerate() {
        let r = k % m;
        let mut acc = Complex32::ZERO;
        for j in 0..p {
            let s = output[j * m + r];
            let idx = ((j * k) % n) * level_stride;
            let w = if ctx.inverse {
                ctx.tw[idx].conj()
            } else {
                ctx.tw[idx]
            };
            acc = acc + s * w;
        }
        *out_k = acc;
    }
    output.copy_from_slice(&scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorize_prefers_radix4_then_odd_primes() {
        assert_eq!(factorize(1), Vec::<usize>::new());
        assert_eq!(factorize(8), vec![4, 2]);
        assert_eq!(factorize(64), vec![4, 4, 4]);
        assert_eq!(factorize(60), vec![4, 3, 5]);
        assert_eq!(factorize(2), vec![2]);
        assert_eq!(factorize(97), vec![97]);
        assert_eq!(factorize(1000), vec![4, 2, 5, 5, 5]);
    }

    #[test]
    fn largest_factor_reports_big_prime() {
        assert_eq!(largest_factor(1024), 4);
        assert_eq!(largest_factor(97), 97);
        assert_eq!(largest_factor(60), 5);
    }
}
