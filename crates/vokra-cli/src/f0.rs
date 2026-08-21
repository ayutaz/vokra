//! `vokra-cli f0` — checkpoint-free pitch extraction (YIN / PyIN).
//!
//! # Why this is its own subcommand rather than a `run --task`
//!
//! Every `run` task loads a GGUF: `RunArgs::model` is a required `String`, and
//! the F0 routes there (`ModelTask::F0Rmvpe`, `F0Fcpe`, `F0Crepe`) reach their
//! extractors through the supplied GGUF. YIN and PyIN have no weights at all — they are
//! pure DSP over the input samples, with no checkpoint, no license class and
//! no `docs/license-audit.md` §3.1 row. Threading them through a
//! model-loading path would mean inventing a `--model` a caller cannot
//! supply, so they get an entry point shaped like what they actually are.
//!
//! They were unreachable from any binary until 2026-08-16:
//! `pub use f0::{pyin, yin};` in `crates/vokra-ops/src/lib.rs` was the only
//! reference to either symbol anywhere outside that crate. This is the same
//! shape the 2026-08-15 audit found for the ITN capability — a landed
//! implementation that stops one layer short of a user — except that these
//! two need no external input whatsoever to be useful.
//!
//! # Output
//!
//! Byte-identical in shape to `run`'s RMVPE route
//! (`crates/vokra-cli/src/run.rs`, `run_f0_rmvpe`): a summary line on stderr's
//! sibling stdout, then one tab-separated row per frame,
//!
//! ```text
//! time_sec<TAB>hz<TAB>voiced<TAB>confidence
//! ```
//!
//! so a caller can swap `--algo` (or swap in the RMVPE route) without
//! touching whatever parses the rows.
//!
//! Both ops return `Vec<f32>` of Hz with `0.0` marking an unvoiced frame, so
//! `voiced` is derived as `hz > 0.0` and `confidence` is reported as `1.0` /
//! `0.0` to match. Neither op exposes a per-frame confidence today; emitting a
//! fabricated one would be worse than emitting the binary the op actually
//! supports (FR-EX-08), and the column exists so the row shape stays shared.
//!
//! # Sample rate
//!
//! Not fixed. `yin` / `pyin` take `sample_rate` and derive their lag search
//! from it, so any rate the WAV carries is honored as-is and nothing is
//! silently resampled. `sample_rate == 0`, or an `fmin`/`fmax` pair the op
//! rejects, fails loudly through the op's own `InvalidArgument`.

use std::process::ExitCode;

use crate::wav;

pub(crate) const USAGE: &str = "\
vokra-cli f0 — checkpoint-free pitch extraction (YIN / PyIN)

USAGE:
    vokra-cli f0 --input <in.wav> [--algo yin|pyin] [--fmin <hz>] [--fmax <hz>]

OPTIONS:
    --input <path>   mono WAV to analyse (required). Any sample rate: the
                     lag search is derived from it, and nothing is resampled.
    --algo <name>    yin (default) or pyin. PyIN marginalises YIN's absolute
                     threshold over a prior instead of fixing one, so it
                     tracks through octave errors YIN can commit; it costs
                     more per frame.
    --fmin <hz>      pitch search floor [default 65.0, ~C2]
    --fmax <hz>      pitch search ceiling [default 2093.0, ~C7]
    -h, --help       this text

OUTPUT:
    A summary line, then one row per frame:
        time_sec<TAB>hz<TAB>voiced<TAB>confidence
    identical in shape to `vokra-cli run` on an RMVPE GGUF, so downstream
    parsing is unchanged. An unvoiced frame is hz=0.000, voiced=false.

NOTE:
    These two extractors carry no weights, so unlike the RMVPE / FCPE / CREPE
    members of the same family they need no `--model` and no checkpoint.
";

/// Pitch floor in Hz — roughly C2, the bottom of a low male speaking range.
const DEFAULT_FMIN: f32 = 65.0;
/// Pitch ceiling in Hz — roughly C7, above soprano and well above speech.
const DEFAULT_FMAX: f32 = 2093.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Algo {
    Yin,
    Pyin,
}

impl Algo {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "yin" => Ok(Self::Yin),
            "pyin" => Ok(Self::Pyin),
            other => Err(format!(
                "f0: unknown --algo `{other}` (expected `yin` or `pyin`); \
                 the neural members of this family (rmvpe / fcpe / crepe) load \
                 a checkpoint and run through `vokra-cli run --model <gguf>` \
                 instead"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Yin => "yin",
            Self::Pyin => "pyin",
        }
    }
}

#[derive(Debug)]
struct F0Args {
    input: Option<String>,
    algo: Algo,
    fmin: f32,
    fmax: f32,
}

fn parse_args(args: &[String]) -> Result<Option<F0Args>, String> {
    let mut out = F0Args {
        input: None,
        algo: Algo::Yin,
        fmin: DEFAULT_FMIN,
        fmax: DEFAULT_FMAX,
    };
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        // Every value-taking flag reads `args[i + 1]` through this helper, so
        // a trailing `--fmin` with nothing after it is a named error rather
        // than a panic on an out-of-range index.
        let mut value = |flag: &str| -> Result<String, String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("f0: {flag} needs a value"))
        };
        match a {
            "-h" | "--help" => return Ok(None),
            "--input" => out.input = Some(value("--input")?),
            "--algo" => out.algo = Algo::parse(&value("--algo")?)?,
            "--fmin" => {
                let v = value("--fmin")?;
                out.fmin = v
                    .parse()
                    .map_err(|_| format!("f0: --fmin `{v}` is not a number"))?;
            }
            "--fmax" => {
                let v = value("--fmax")?;
                out.fmax = v
                    .parse()
                    .map_err(|_| format!("f0: --fmax `{v}` is not a number"))?;
            }
            other => return Err(format!("f0: unknown option `{other}`")),
        }
        i += 1;
    }
    Ok(Some(out))
}

/// Runs the `f0` subcommand.
///
/// # Errors
///
/// A bad flag, a missing `--input`, an unreadable or non-mono WAV, or an
/// argument the underlying op rejects (a zero sample rate, an inverted
/// `fmin`/`fmax`).
pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    let Some(a) = parse_args(args)? else {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    };
    let path = a
        .input
        .as_deref()
        .ok_or("f0: --input <in.wav> is required")?;

    let clip = wav::read_wav(path)?;
    let hz = match a.algo {
        Algo::Yin => vokra_ops::yin(&clip.samples, clip.sample_rate, a.fmin, a.fmax),
        Algo::Pyin => vokra_ops::pyin(&clip.samples, clip.sample_rate, a.fmin, a.fmax),
    }
    .map_err(|e| format!("f0 ({}): {e}", a.algo.name()))?;

    for line in render(&hz, clip.sample_rate, a.algo) {
        println!("{line}");
    }
    Ok(ExitCode::SUCCESS)
}

/// Formats the summary line and one row per frame.
///
/// Split out of [`main`] so the OUTPUT can be tested, not just the argument
/// parsing. An earlier revision of this module had eight tests that all
/// passed while `voiced` was hard-coded to `true` — they exercised
/// `parse_args` and nothing else, so the column a caller actually reads was
/// unguarded. `render` takes the Hz slice directly, which is what the ops
/// return, so a test can hand it a known track and check every field.
fn render(hz: &[f32], sample_rate: u32, algo: Algo) -> Vec<String> {
    let voiced = hz.iter().filter(|v| **v > 0.0).count();
    let mut out = Vec::with_capacity(hz.len() + 1);
    out.push(format!(
        "f0: {} frames, voiced_frames={voiced}, algo={} @ {sample_rate} Hz",
        hz.len(),
        algo.name(),
    ));
    // The op's frame index times its hop, both fixed by the op — see
    // `vokra_ops::f0::DEFAULT_HOP`. Computed here rather than returned by the
    // op because the op's contract is "one Hz value per frame".
    let hop_sec = vokra_ops::f0::DEFAULT_HOP as f32 / sample_rate as f32;
    for (i, v) in hz.iter().enumerate() {
        let is_voiced = *v > 0.0;
        out.push(format!(
            "{:.4}\t{:.3}\t{}\t{:.4}",
            i as f32 * hop_sec,
            v,
            is_voiced,
            if is_voiced { 1.0_f32 } else { 0.0 }
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn defaults_are_yin_over_a_speech_pitch_range() {
        let a = parse_args(&args(&["--input", "x.wav"]))
            .expect("parse")
            .expect("not help");
        assert_eq!(a.algo, Algo::Yin);
        assert_eq!(a.fmin, DEFAULT_FMIN);
        assert_eq!(a.fmax, DEFAULT_FMAX);
        assert_eq!(a.input.as_deref(), Some("x.wav"));
    }

    #[test]
    fn both_algorithms_parse() {
        for (flag, want) in [("yin", Algo::Yin), ("pyin", Algo::Pyin)] {
            let a = parse_args(&args(&["--input", "x.wav", "--algo", flag]))
                .expect("parse")
                .expect("not help");
            assert_eq!(a.algo, want, "--algo {flag}");
        }
    }

    #[test]
    fn an_unknown_algo_names_the_neural_route_instead_of_just_refusing() {
        // The neural F0 members DO exist, behind `run --model`. A bare
        // "unknown algo" would leave a caller who typed `--algo rmvpe`
        // believing Vokra has no RMVPE at all.
        let err = parse_args(&args(&["--algo", "rmvpe"])).expect_err("must refuse");
        assert!(err.contains("rmvpe"), "must echo what was typed: {err}");
        assert!(
            err.contains("vokra-cli run"),
            "must point at the route that does load a checkpoint: {err}"
        );
    }

    #[test]
    fn a_value_flag_with_nothing_after_it_is_named_not_a_panic() {
        for flag in ["--input", "--algo", "--fmin", "--fmax"] {
            let err = parse_args(&args(&[flag])).expect_err("must refuse");
            assert!(err.contains(flag), "error must name {flag}: {err}");
        }
    }

    #[test]
    fn a_non_numeric_bound_is_refused() {
        let err = parse_args(&args(&["--fmin", "low"])).expect_err("must refuse");
        assert!(err.contains("--fmin"), "{err}");
    }

    #[test]
    fn missing_input_is_refused_by_main() {
        let err = main(&args(&["--algo", "pyin"])).expect_err("must refuse");
        assert!(err.contains("--input"), "{err}");
    }

    #[test]
    fn help_short_circuits_before_any_input_requirement() {
        for flag in ["-h", "--help"] {
            assert!(
                parse_args(&args(&[flag])).expect("parse").is_none(),
                "{flag} must request the usage text"
            );
        }
    }

    /// A KNOWN track through `render`, checked field by field.
    ///
    /// This is the test the first revision of this module lacked: its eight
    /// tests all passed while `voiced` was hard-coded to `true`, because
    /// every one of them stopped at `parse_args`. Verified adversarially —
    /// forcing `is_voiced = true` makes this fail and the parsing tests do
    /// not notice.
    #[test]
    fn render_marks_voiced_frames_by_hz_and_times_them_by_the_op_hop() {
        // 16 kHz, hop 256 -> 16 ms per frame.
        let hz = [0.0_f32, 220.0, 0.0, 440.5];
        let rows = render(&hz, 16_000, Algo::Yin);

        assert_eq!(rows.len(), 5, "one summary line plus one row per frame");
        assert_eq!(
            rows[0],
            "f0: 4 frames, voiced_frames=2, algo=yin @ 16000 Hz"
        );

        // time<TAB>hz<TAB>voiced<TAB>confidence, hop 256 / 16000 = 0.016 s.
        assert_eq!(rows[1], "0.0000\t0.000\tfalse\t0.0000");
        assert_eq!(rows[2], "0.0160\t220.000\ttrue\t1.0000");
        assert_eq!(rows[3], "0.0320\t0.000\tfalse\t0.0000");
        assert_eq!(rows[4], "0.0480\t440.500\ttrue\t1.0000");
    }

    /// The confidence column tracks `voiced`, which tracks `hz > 0.0`.
    ///
    /// Pinned separately because the three are derived from one another: a
    /// change that decoupled them would still produce plausible-looking rows.
    #[test]
    fn confidence_never_disagrees_with_the_voiced_flag() {
        let hz = [0.0_f32, 1.0, 0.0, 99.0, 0.0];
        for row in render(&hz, 16_000, Algo::Pyin).iter().skip(1) {
            let f: Vec<&str> = row.split('\t').collect();
            let hz_v: f32 = f[1].parse().expect("hz column");
            let voiced: bool = f[2].parse().expect("voiced column");
            let conf: f32 = f[3].parse().expect("confidence column");
            assert_eq!(voiced, hz_v > 0.0, "voiced must follow hz: {row}");
            assert!(
                (conf - if voiced { 1.0 } else { 0.0 }).abs() < f32::EPSILON,
                "confidence must follow voiced: {row}"
            );
        }
    }

    /// The row shape must stay swappable with `run`'s RMVPE route, which is
    /// the entire reason it was copied rather than invented.
    #[test]
    fn every_row_has_the_four_shared_columns() {
        let rows = render(&[0.0_f32, 120.0], 22_050, Algo::Yin);
        for row in rows.iter().skip(1) {
            assert_eq!(
                row.split('\t').count(),
                4,
                "row must stay time/hz/voiced/confidence: {row}"
            );
        }
        assert!(
            !rows[0].contains('\t'),
            "the summary line must not look like a data row: {}",
            rows[0]
        );
    }

    /// The hop is per-second, so a different rate must retime the rows.
    #[test]
    fn a_different_sample_rate_retimes_the_frames() {
        let hz = [0.0_f32, 0.0];
        let at_16k = render(&hz, 16_000, Algo::Yin);
        let at_8k = render(&hz, 8_000, Algo::Yin);
        // Same hop in SAMPLES, half the rate -> twice the wall-clock spacing.
        assert_eq!(at_16k[2].split('\t').next(), Some("0.0160"));
        assert_eq!(at_8k[2].split('\t').next(), Some("0.0320"));
    }

    /// An empty track is a summary line and nothing else — not a panic and
    /// not a fabricated frame.
    #[test]
    fn an_empty_track_renders_only_the_summary() {
        let rows = render(&[], 16_000, Algo::Yin);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("0 frames"), "{}", rows[0]);
        assert!(rows[0].contains("voiced_frames=0"), "{}", rows[0]);
    }

    #[test]
    fn unknown_options_are_refused() {
        let err = parse_args(&args(&["--model", "x.gguf"])).expect_err("must refuse");
        assert!(err.contains("--model"), "{err}");
    }
}
