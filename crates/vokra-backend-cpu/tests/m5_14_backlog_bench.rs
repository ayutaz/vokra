//! M5-14-BACKLOG break-even microbench (T02 / T06), `#[ignore]`d so it never
//! runs in the normal suite. Run in release, pinned:
//!
//! ```text
//! cargo test -p vokra-backend-cpu --release --test m5_14_backlog_bench \
//!     -- --ignored --nocapture
//! ```
//!
//! It measures, on the host's active ISA:
//!
//! **T06** — at the whisper-small decoder projection shapes, the current route
//! for `m < PACK_MR` (`gemm_f32`, which gates to the legacy kernel) vs. the
//! *forced* packed route (`gemm_test_probe::packed_forced`), the route a widened
//! gate would take. This is the break-even the batched-beam premise rests on
//! (fold `beam_width` m=1 forwards into one m=beam_width GEMM); if packed is not
//! faster at m ∈ 2..7 the premise collapses. `t06b` sweeps the thin-m MACs
//! break-even; `t06c` compares packed m=N against N separate m=1 forwards.
//!
//! **T02** — at the CAM++ conv-as-GEMM shapes (`m = c_out` RowTiles families +
//! the FCM thin-m front), full `gemm_f32` vs the forced packed route, the
//! evidence for whether a weight-keyed pack cache can bite (Findings 1/2).
//!
//! Every measured pair is also asserted **bit-identical** (`to_bits`), so the
//! bench doubles as a correctness check of `packed_forced` at these shapes.

use std::time::Instant;

use vokra_backend_cpu::gemm_test_probe as probe;
use vokra_backend_cpu::{active_isa, kernels};

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Rng(s | 1)
    }
    fn f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Median wall-ns per call over `reps` timed passes after `warm` warm-ups.
fn bench<F: FnMut()>(warm: usize, reps: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warm {
        for _ in 0..iters {
            f();
        }
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        samples.push(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[reps / 2]
}

fn assert_bits(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx} len");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{ctx}: bit diff at {i}: {x} vs {y}"
        );
    }
}

#[test]
#[ignore = "microbench; run with --release --ignored --nocapture"]
fn t06_batched_beam_breakeven() {
    println!(
        "\n=== T06 batched-beam break-even (ISA {:?}) ===",
        active_isa()
    );
    if !probe::active_gemm_has_packed() {
        println!("active ISA has no packed kernels — nothing to measure");
        return;
    }
    // whisper-small decoder projection shapes (d_model 768, ffn 3072). The
    // GEMM the batched forward would issue is m = beam_width over these (n, k).
    let shapes: &[(usize, usize, &str)] = &[
        (768, 768, "attn proj  n=768 k=768"),
        (3072, 768, "fc1        n=3072 k=768"),
        (768, 3072, "fc2        n=768 k=3072"),
    ];
    for &(n, k, tag) in shapes {
        println!("\n-- {tag} --");
        println!("  m  legacy(gemm_f32)  packed_forced   packed/legacy");
        for m in 1..=8usize {
            let mut rng = Rng::new(0xB14 ^ ((m as u64) << 8) ^ ((n as u64) << 20));
            let a = rng.vec(m * k);
            let b = rng.vec(k * n);
            let bias = rng.vec(n);
            let mut cur = vec![0.0f32; m * n];
            let mut pk = vec![0.0f32; m * n];
            // Correctness: current route vs forced packed must be bit-identical.
            kernels::gemm_f32(m, n, k, &a, &b, Some(&bias), &mut cur).unwrap();
            let fired = probe::packed_forced(m, n, k, &a, &b, Some(&bias), &mut pk);
            assert!(fired, "packed kernels present but packed_forced no-op");
            assert_bits(&cur, &pk, &format!("{tag} m={m}"));

            let iters = if m * n * k < 2_000_000 { 2000 } else { 400 };
            let legacy_ns = bench(2, 5, iters, || {
                kernels::gemm_f32(m, n, k, &a, &b, Some(&bias), &mut cur).unwrap();
            });
            let packed_ns = bench(2, 5, iters, || {
                probe::packed_forced(m, n, k, &a, &b, Some(&bias), &mut pk);
            });
            let ratio = packed_ns / legacy_ns;
            let verdict = if m >= probe::PACK_MR {
                "(already routed)"
            } else if ratio < 0.97 {
                "packed WINS"
            } else if ratio > 1.03 {
                "legacy wins"
            } else {
                "~tie"
            };
            println!("  {m}  {legacy_ns:>12.0}  {packed_ns:>12.0}   {ratio:>6.3}  {verdict}");
        }
    }
}

#[test]
#[ignore = "microbench; run with --release --ignored --nocapture"]
fn t06b_thin_m_macs_breakeven() {
    // The `m >= PACK_MR` gate widening (Finding 4) must not regress any
    // *single* m ∈ 2..7 GEMM: find, per m, the MACs at which packed overtakes
    // legacy. `PACKED_MIN_MACS` was calibrated at m >= 8; small m pads more
    // rows to MR = 8, so the thin-m break-even can sit higher. k fixed at the
    // whisper d_model (768); n swept from tiny to the current gate's regime.
    println!(
        "\n=== T06b thin-m packed-vs-legacy MACs break-even (ISA {:?}) ===",
        active_isa()
    );
    if !probe::active_gemm_has_packed() {
        println!("no packed kernels");
        return;
    }
    let k = 768usize;
    println!("  m   n     MACs    packed/legacy   (>1 = legacy still wins)");
    for m in 2..=7usize {
        for &n in &[8usize, 16, 24, 48, 96, 160, 256, 384, 512] {
            let macs = m * n * k;
            let mut rng = Rng::new(0xB16 ^ ((m as u64) << 8) ^ ((n as u64) << 16));
            let a = rng.vec(m * k);
            let b = rng.vec(k * n);
            let mut leg = vec![0.0f32; m * n];
            let mut pk = vec![0.0f32; m * n];
            kernels::gemm_f32_on(active_isa(), m, n, k, &a, &b, None, &mut leg).unwrap();
            let fired = probe::packed_forced(m, n, k, &a, &b, None, &mut pk);
            if !fired {
                continue;
            }
            assert_bits(&leg, &pk, &format!("m={m} n={n}"));
            let iters = 4000;
            let leg_ns = bench(3, 7, iters, || {
                kernels::gemm_f32_on(active_isa(), m, n, k, &a, &b, None, &mut leg).unwrap();
            });
            let pk_ns = bench(3, 7, iters, || {
                probe::packed_forced(m, n, k, &a, &b, None, &mut pk);
            });
            let gate = if probe::would_use_packed(m, n, k) {
                "GATE-on"
            } else {
                "gate-off"
            };
            println!(
                "  {m}  {n:>4}  {macs:>7}    {:>6.3}        {gate}",
                pk_ns / leg_ns
            );
        }
    }
    println!("  (PACKED_MIN_MACS = {} MACs)", 1usize << 18);
}

#[test]
#[ignore = "microbench; run with --release --ignored --nocapture"]
fn t06c_batched_vs_perbeam() {
    // The real batched-beam A/B: one m = beam_width packed forward vs
    // beam_width separate m = 1 forwards (the current per-beam route through
    // the m1 axpy path). This is what T09 folds; the routing gate (T06/T06b)
    // only enables the packed m = N side.
    println!(
        "\n=== T06c batched (packed m=N) vs per-beam (N × m1) (ISA {:?}) ===",
        active_isa()
    );
    if !probe::active_gemm_has_packed() {
        println!("no packed kernels");
        return;
    }
    let shapes: &[(usize, usize, &str)] = &[
        (768, 768, "attn proj n=768 k=768"),
        (3072, 768, "fc1       n=3072 k=768"),
        (768, 3072, "fc2       n=768 k=3072"),
    ];
    for &(n, k, tag) in shapes {
        println!("\n-- {tag} --  (m1×N = N separate m=1 forwards)");
        println!("  N   packed(m=N)   N × m1     batched/perbeam");
        // Per-m1 cost (constant across N).
        let mut rng = Rng::new(0xB1C ^ ((n as u64) << 16));
        let a1 = rng.vec(k);
        let b = rng.vec(k * n);
        let bias = rng.vec(n);
        let mut o1 = vec![0.0f32; n];
        let m1_ns = bench(3, 7, 3000, || {
            kernels::gemm_f32(1, n, k, &a1, &b, Some(&bias), &mut o1).unwrap();
        });
        for beam in 2..=6usize {
            let a = rng.vec(beam * k);
            let mut pk = vec![0.0f32; beam * n];
            let iters = if beam * n * k < 2_000_000 { 2000 } else { 500 };
            let packed_ns = bench(3, 7, iters, || {
                probe::packed_forced(beam, n, k, &a, &b, Some(&bias), &mut pk);
            });
            let perbeam_ns = m1_ns * beam as f64;
            println!(
                "  {beam}  {packed_ns:>10.0}  {perbeam_ns:>9.0}      {:>6.3}",
                packed_ns / perbeam_ns
            );
        }
    }
}

#[test]
#[ignore = "microbench; run with --release --ignored --nocapture"]
fn t02_camplus_pack_shapes() {
    println!(
        "\n=== T02 CAM++ conv-as-GEMM shapes (ISA {:?}) ===",
        active_isa()
    );
    if !probe::active_gemm_has_packed() {
        println!("active ISA has no packed kernels — nothing to measure");
        return;
    }
    // (m=c_out, n≈t_frames, k=c_in·kk) families traced from camplus.rs. n is
    // reduced from the Wave-0 43920 so the debug-free bench stays quick while
    // keeping the routing branch (RowTiles for m>MC, thin-m for m<=MC).
    let shapes: &[(usize, usize, usize, &str)] = &[
        (32, 4096, 288, "FCM conv2 thin-m  m=32  (m<=MC → thin-m)"),
        (128, 2048, 320, "tdnn      m=128 (m>MC → RowTiles)"),
        (256, 1024, 512, "block1 transit m=256"),
        (512, 512, 1024, "block2/3 transit m=512"),
        (192, 512, 1024, "dense head m=192"),
    ];
    println!("  shape                                 legacy    packed  packed/legacy  MACs");
    for &(m, n, k, tag) in shapes {
        let mut rng = Rng::new(0xCA3 ^ ((m as u64) << 8) ^ ((k as u64) << 24));
        let a = rng.vec(m * k); // weight (operand A)
        let b = rng.vec(k * n); // activation / im2col (operand B)
        let mut leg = vec![0.0f32; m * n];
        let mut pk = vec![0.0f32; m * n];
        kernels::gemm_f32_on(active_isa(), m, n, k, &a, &b, None, &mut leg).unwrap();
        probe::packed_forced(m, n, k, &a, &b, None, &mut pk);
        assert_bits(&leg, &pk, tag);
        let iters = 200;
        let legacy_ns = bench(2, 5, iters, || {
            kernels::gemm_f32_on(active_isa(), m, n, k, &a, &b, None, &mut leg).unwrap();
        });
        let packed_ns = bench(2, 5, iters, || {
            probe::packed_forced(m, n, k, &a, &b, None, &mut pk);
        });
        let macs = m * n * k;
        println!(
            "  {tag:<36} {legacy_ns:>8.0}  {packed_ns:>8.0}   {:>6.3}      {macs}",
            packed_ns / legacy_ns
        );
    }
    println!(
        "\n  note: operand A = weight (reusable across calls), operand B = \
         activation (changes). A pack traffic per element ~ 1 copy amortised \
         over n MACs; large n ⇒ A-pack fraction small."
    );
}
