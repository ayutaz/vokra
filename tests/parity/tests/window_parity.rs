//! Window-function parity against numpy (M0-04-T09).
//!
//! numpy.{hanning,hamming,kaiser} are the symmetric-window references. numpy
//! has no 4-term Blackman-Harris, so that window is covered analytically in
//! `vokra-ops` unit tests instead. Tolerance is NFR-QL-01 FP32 `atol = 0.01`.

use vokra_core::ir::graph::{Window, WindowSymmetry};
use vokra_ops::window::window;
use vokra_parity::{assert_close, load_dir};

#[test]
fn window_parity_against_numpy() {
    let fixtures = load_dir("window");
    assert!(
        !fixtures.is_empty(),
        "no window fixtures; run: python3 tests/parity/gen_parity_fixtures.py window"
    );
    for f in &fixtures {
        let length = f.usize("length");
        let symmetry = match f.get("symmetry") {
            "symmetric" => WindowSymmetry::Symmetric,
            "periodic" => WindowSymmetry::Periodic,
            other => panic!("unknown symmetry {other:?}"),
        };
        let kind = match f.get("window") {
            "hann" => Window::Hann,
            "hamming" => Window::Hamming,
            "kaiser" => Window::Kaiser {
                beta: f.f64("beta") as f32,
            },
            other => panic!("unknown window {other:?}"),
        };
        let got = window(kind, length, symmetry);
        assert_close(&got, &f.floats("values"), &format!("{:?}", f.path));
    }
}
