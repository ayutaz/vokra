//! Frame-synchronous token-passing WFST decoder (M5-06 T07–T10).
//!
//! Given a decode graph [`Fst`] and a per-frame acoustic **emission** matrix,
//! the decoder runs the classic token-passing / Viterbi sweep and returns the
//! best output (word) sequence (and, on request, a lattice / n-best). It is the
//! reachable portion of the composition `emission ⊗ graph`, so it agrees with
//! OpenFST's `shortestpath(compose(E, G))` — the parity oracle
//! (`tests/parity_wfst.rs`).
//!
//! # Emission contract (T07)
//!
//! `emission[frame][ilabel]` is the acoustic **cost** (negative log-likelihood;
//! **lower is better**) of consuming input label `ilabel` at `frame`. Labels
//! index the row directly (`ilabel = 0` is epsilon and is never consumed from
//! emission). Each row must be longer than the largest `ilabel` any arc uses,
//! or it is an explicit [`VokraError::InvalidArgument`] — the dimension /
//! label-space mismatch is never silently clamped (FR-EX-08).
//!
//! # Feeder honesty (the M5-06 constraint, do not paper over it)
//!
//! The emission matrix is what a CTC / RNN-T acoustic head would produce, but
//! those decoders (`ctc_decode` / `rnnt_decode`, FR-OP-41/42) are **reserved
//! and unimplemented** ([`crate::m5_residual_ops`]). Vokra's only live acoustic
//! decoder is attention-based `beam_search`, which returns *token sequences*,
//! not a frame-synchronous emission matrix. So M5-06 lands `wfst_decode` as a
//! **decode-only** primitive verified against offline reference emissions; a
//! classic HCLG frame-synchronous e2e ASR waits on the CTC/RNN-T feeder (ADR
//! M5-06). The API here does not pretend a live ASR path exists.
//!
//! # What this is *not*
//!
//! No `compose` / `determinize` / `minimize` (ADR M5-06 §1). The HCLG graph is
//! composed offline by the developer-side OpenFST toolchain; Vokra reads the
//! finished graph and decodes over it.

use std::collections::BTreeMap;

use crate::error::{Result, VokraError};

use super::fst::{Fst, StateId};
use super::lattice::{LatArc, RawLattice, WfstHypothesis, WfstLattice};
use super::semiring::TropicalWeight;

/// Sentinel FST state for the lattice super-final sink (see
/// [`super::lattice`]). Kept in sync with the lattice module's constant.
const SUPER_FINAL_STATE: i64 = -1;

/// Token-passing decoder configuration.
#[derive(Debug, Clone)]
pub struct WfstDecodeConfig {
    /// Beam width for pruning: at each frame, a token whose best cost exceeds
    /// `frame_min + beam` is not expanded. `None` disables pruning (the exact
    /// full search — used by the parity fixtures so the decoder provably finds
    /// the same optimum as OpenFST). A pruned token stays in the lattice but
    /// becomes a dead-end and is trimmed away.
    pub beam: Option<f32>,
    /// Number of hypotheses [`WfstDecoder::decode_nbest`] returns, best-first.
    /// Mirrors [`crate::decode::BeamSearchConfig::n_best`].
    pub n_best: usize,
}

impl Default for WfstDecodeConfig {
    fn default() -> Self {
        Self {
            beam: None,
            n_best: 1,
        }
    }
}

/// A token-passing decoder over a decode graph.
///
/// Borrows the [`Fst`]; construct with [`WfstDecoder::new`] and optionally set a
/// beam / n-best via [`WfstDecoder::with_config`] or the builder helpers.
#[derive(Debug, Clone)]
pub struct WfstDecoder<'f> {
    fst: &'f Fst<TropicalWeight>,
    config: WfstDecodeConfig,
}

impl<'f> WfstDecoder<'f> {
    /// A decoder over `fst` with the default config (no pruning, 1-best).
    pub fn new(fst: &'f Fst<TropicalWeight>) -> Self {
        Self {
            fst,
            config: WfstDecodeConfig::default(),
        }
    }

    /// A decoder over `fst` with an explicit config.
    pub fn with_config(fst: &'f Fst<TropicalWeight>, config: WfstDecodeConfig) -> Self {
        Self { fst, config }
    }

    /// Sets the beam width (builder style).
    #[must_use]
    pub fn beam(mut self, beam: f32) -> Self {
        self.config.beam = Some(beam);
        self
    }

    /// Sets the n-best count (builder style).
    #[must_use]
    pub fn n_best(mut self, n: usize) -> Self {
        self.config.n_best = n;
        self
    }

    /// Builds the trimmed search lattice for `emission` (T11).
    ///
    /// # Errors
    /// - [`VokraError::InvalidArgument`] — invalid FST (see [`Fst::validate`]),
    ///   an emission row too short for the FST's labels, a NaN emission, or an
    ///   epsilon cycle in the graph (the T09 cycle guard);
    pub fn lattice(&self, emission: &[Vec<f32>]) -> Result<WfstLattice> {
        let raw = build_raw_lattice(self.fst, emission, self.config.beam)?;
        Ok(WfstLattice::from_raw(raw))
    }

    /// Decodes the best (minimum-cost) output (word) sequence for `emission`.
    /// `Ok(None)` when no complete path exists (no token reaches a final state
    /// after consuming all frames). See [`Self::lattice`] for the error cases.
    pub fn decode(&self, emission: &[Vec<f32>]) -> Result<Option<WfstHypothesis>> {
        Ok(self.lattice(emission)?.best_path())
    }

    /// Decodes up to [`WfstDecodeConfig::n_best`] hypotheses, best-first,
    /// de-duplicated by output sequence (T12).
    pub fn decode_nbest(&self, emission: &[Vec<f32>]) -> Result<Vec<WfstHypothesis>> {
        Ok(self.lattice(emission)?.nbest(self.config.n_best))
    }
}

/// Runs the forward token-passing sweep and returns the raw (untrimmed)
/// lattice. Split out so the unit tests can inspect the pre-trim graph, and so
/// [`WfstDecoder::lattice`] owns the single trim step.
fn build_raw_lattice(
    fst: &Fst<TropicalWeight>,
    emission: &[Vec<f32>],
    beam: Option<f32>,
) -> Result<RawLattice> {
    fst.validate()?;
    let eps_topo = epsilon_topo_order(fst)?; // T09 cycle guard
    check_emission(fst, emission)?;

    let num_frames = emission.len();
    let start_state = fst
        .start()
        .expect("validated FST has a start state (Fst::validate)");

    let mut b = Builder::new(num_frames);
    let start_node = b.get_or_create(0, start_state);
    // cost_to is used only for beam pruning; distances for output come from the
    // trimmed lattice (authoritative).
    b.cost_to[start_node] = 0.0;

    // Epsilon closure at frame 0 (before consuming any emission).
    b.epsilon_close(fst, 0, &eps_topo);

    // `f` is the frame index used to key `per_frame(f)`, `get_or_create(f+1, …)`
    // AND `emission[f]` — not a simple slice walk, so index it directly.
    #[allow(clippy::needless_range_loop)]
    for f in 0..num_frames {
        // Determine the prune threshold for frame f (if beaming).
        let threshold = beam.map(|w| {
            let min_f = b
                .per_frame(f)
                .values()
                .map(|&nd| b.cost_to[nd])
                .filter(|c| c.is_finite())
                .fold(f32::INFINITY, f32::min);
            min_f + w
        });

        // Emitting expansion frame f -> f+1. Iterate present states ascending
        // (BTreeMap) for determinism. Snapshot the (state,node) pairs first so
        // the borrow of `per_frame` ends before we mutate the builder.
        let entries: Vec<(StateId, usize)> =
            b.per_frame(f).iter().map(|(&s, &nd)| (s, nd)).collect();
        for (state, node) in entries {
            if let Some(t) = threshold
                && b.cost_to[node] > t
            {
                continue; // pruned — do not expand
            }
            let base = b.cost_to[node];
            for (ai, arc) in fst.arcs_of(state)?.iter().enumerate() {
                if arc.is_epsilon() {
                    continue; // handled by epsilon closure
                }
                let e = emission[f][arc.ilabel as usize];
                let cost = e + arc.weight.value();
                let to = b.get_or_create(f + 1, arc.nextstate);
                b.push_arc(LatArc {
                    from: node,
                    to,
                    ilabel: arc.ilabel,
                    olabel: arc.olabel,
                    weight: cost,
                    key: (state, ai as u32),
                });
                if base.is_finite() {
                    b.cost_to[to] = b.cost_to[to].min(base + cost);
                }
            }
        }

        // Epsilon closure at the new frame f+1.
        b.epsilon_close(fst, f + 1, &eps_topo);
    }

    // Super-final: connect frame-`num_frames` final states.
    let final_node = b.new_super_final();
    let last: Vec<(StateId, usize)> = b
        .per_frame(num_frames)
        .iter()
        .map(|(&s, &nd)| (s, nd))
        .collect();
    for (state, node) in last {
        if fst.is_final(state)? {
            let fw = fst.final_weight(state)?.value();
            b.push_arc(LatArc {
                from: node,
                to: final_node,
                ilabel: 0,
                olabel: 0,
                weight: fw,
                key: (state, u32::MAX),
            });
        }
    }

    Ok(RawLattice {
        frames: b.frames,
        states: b.states,
        arcs: b.arcs,
        start: start_node,
        final_node,
        num_frames,
    })
}

/// The lattice under construction during the forward pass.
struct Builder {
    frames: Vec<usize>,
    states: Vec<i64>,
    arcs: Vec<LatArc>,
    cost_to: Vec<f32>,
    /// `index[frame]` = FST-state → lattice-node for that frame. Size
    /// `num_frames + 1` (frame `num_frames` holds the last-frame states that
    /// feed the super-final).
    index: Vec<BTreeMap<StateId, usize>>,
}

impl Builder {
    fn new(num_frames: usize) -> Self {
        Self {
            frames: Vec::new(),
            states: Vec::new(),
            arcs: Vec::new(),
            cost_to: Vec::new(),
            index: vec![BTreeMap::new(); num_frames + 1],
        }
    }

    fn per_frame(&self, frame: usize) -> &BTreeMap<StateId, usize> {
        &self.index[frame]
    }

    /// Returns the node for `(frame, state)`, creating it (cost `+∞`) if new.
    fn get_or_create(&mut self, frame: usize, state: StateId) -> usize {
        if let Some(&nd) = self.index[frame].get(&state) {
            return nd;
        }
        let nd = self.frames.len();
        self.frames.push(frame);
        self.states.push(i64::from(state));
        self.cost_to.push(f32::INFINITY);
        self.index[frame].insert(state, nd);
        nd
    }

    /// Creates the single super-final sink node.
    fn new_super_final(&mut self) -> usize {
        let nd = self.frames.len();
        self.frames.push(self.index.len().saturating_sub(1)); // = num_frames
        self.states.push(SUPER_FINAL_STATE);
        self.cost_to.push(f32::INFINITY);
        nd
    }

    fn push_arc(&mut self, arc: LatArc) {
        self.arcs.push(arc);
    }

    /// Epsilon closure within `frame`: propagate `ilabel == 0` arcs, relaxing
    /// `cost_to` in epsilon-topological order so each state's cost is final
    /// before its epsilon out-arcs fire. The graph's epsilon subgraph is known
    /// acyclic ([`epsilon_topo_order`]), so this terminates and creates every
    /// epsilon-reachable node + arc for the frame.
    fn epsilon_close(&mut self, fst: &Fst<TropicalWeight>, frame: usize, eps_topo: &[StateId]) {
        for &state in eps_topo {
            let Some(&node) = self.index[frame].get(&state) else {
                continue; // this state is not (yet) present in the frame
            };
            let base = self.cost_to[node];
            // arcs_of cannot fail for an in-range state; the FST validated.
            let arcs: Vec<(u32, StateId, f32, super::fst::Label)> = fst
                .arcs_of(state)
                .expect("validated state in range")
                .iter()
                .enumerate()
                .filter(|(_, a)| a.is_epsilon())
                .map(|(ai, a)| (ai as u32, a.nextstate, a.weight.value(), a.olabel))
                .collect();
            for (ai, next, w, olabel) in arcs {
                let to = self.get_or_create(frame, next);
                self.push_arc(LatArc {
                    from: node,
                    to,
                    ilabel: 0,
                    olabel,
                    weight: w,
                    key: (state, ai),
                });
                if base.is_finite() {
                    self.cost_to[to] = self.cost_to[to].min(base + w);
                }
            }
        }
    }
}

/// A topological order of the FST states under **epsilon** arcs (Kahn, ties by
/// ascending state id). Doubles as the T09 **epsilon-cycle guard**: an epsilon
/// cycle leaves some states with non-zero residual in-degree, so the order is
/// shorter than the state count and we return an explicit error instead of
/// looping forever.
///
/// # Errors
/// [`VokraError::InvalidArgument`] if the epsilon subgraph has a cycle.
fn epsilon_topo_order(fst: &Fst<TropicalWeight>) -> Result<Vec<StateId>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = fst.num_states();
    let mut indeg = vec![0usize; n];
    let mut eps_out: Vec<Vec<StateId>> = vec![Vec::new(); n];
    // `s` is both the arc-source (StateId for `arcs_of`) and the `eps_out`
    // index, so a range loop is the clear form here.
    #[allow(clippy::needless_range_loop)]
    for s in 0..n {
        for arc in fst.arcs_of(s as StateId)? {
            if arc.is_epsilon() {
                eps_out[s].push(arc.nextstate);
                indeg[arc.nextstate as usize] += 1;
            }
        }
    }
    let mut heap: BinaryHeap<Reverse<StateId>> = BinaryHeap::new();
    for (s, &d) in indeg.iter().enumerate() {
        if d == 0 {
            heap.push(Reverse(s as StateId));
        }
    }
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(s)) = heap.pop() {
        order.push(s);
        for &t in &eps_out[s as usize] {
            let ti = t as usize;
            indeg[ti] -= 1;
            if indeg[ti] == 0 {
                heap.push(Reverse(t));
            }
        }
    }
    if order.len() != n {
        return Err(VokraError::InvalidArgument(
            "wfst: epsilon cycle detected in the FST — epsilon closure would not terminate \
             (FR-EX-08)"
                .into(),
        ));
    }
    Ok(order)
}

/// Validates the emission matrix against the FST's label space (T07): every row
/// must be longer than the largest `ilabel` any arc uses, and no value may be
/// NaN.
///
/// # Errors
/// [`VokraError::InvalidArgument`] on a short row or a NaN emission.
fn check_emission(fst: &Fst<TropicalWeight>, emission: &[Vec<f32>]) -> Result<()> {
    let mut max_ilabel = 0u32;
    for s in 0..fst.num_states() {
        for arc in fst.arcs_of(s as StateId)? {
            if !arc.is_epsilon() {
                max_ilabel = max_ilabel.max(arc.ilabel);
            }
        }
    }
    for (f, row) in emission.iter().enumerate() {
        if (row.len() as u32) <= max_ilabel {
            return Err(VokraError::InvalidArgument(format!(
                "wfst: emission frame {f} has length {} but the FST uses ilabel up to \
                 {max_ilabel} (need length > {max_ilabel}) — FR-EX-08",
                row.len()
            )));
        }
        if let Some(i) = row.iter().position(|v| v.is_nan()) {
            return Err(VokraError::InvalidArgument(format!(
                "wfst: emission frame {f} has a NaN at index {i}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::fst::{Arc, Fst, Label, StateId};
    use super::super::semiring::{Semiring, TropicalWeight};
    use super::*;

    type W = TropicalWeight;

    fn arc(il: Label, ol: Label, w: f32, next: StateId) -> Arc<W> {
        Arc {
            ilabel: il,
            olabel: ol,
            weight: TropicalWeight::new(w),
            nextstate: next,
        }
    }

    /// Linear graph: 0 -(1:10/0.5)-> 1 -(2:20/0.25)-> 2(F/0.125).
    fn linear() -> Fst<W> {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        let s2 = f.add_state(TropicalWeight::new(0.125));
        f.set_start(s0);
        f.add_arc(s0, arc(1, 10, 0.5, s1)).unwrap();
        f.add_arc(s1, arc(2, 20, 0.25, s2)).unwrap();
        f
    }

    /// Emission for the linear/epsilon graphs: 2 frames, labels {1,2}, index 0
    /// unused. frame0 prefers label 1, frame1 prefers label 2.
    fn emission_lin() -> Vec<Vec<f32>> {
        vec![
            vec![f32::INFINITY, 0.1, 5.0], // frame 0: label1=0.1, label2=5.0
            vec![f32::INFINITY, 5.0, 0.2], // frame 1: label1=5.0, label2=0.2
        ]
    }

    // ---- T08: emitting expansion, single/best path -------------------------

    #[test]
    fn linear_best_path_matches_hand_computation() {
        // 0.1+0.5 (frame0,label1) + 0.2+0.25 (frame1,label2) + 0.125 (final)
        // = 1.175, words [10, 20].
        let f = linear();
        let hyp = WfstDecoder::new(&f)
            .decode(&emission_lin())
            .unwrap()
            .expect("a complete path exists");
        assert_eq!(hyp.words, vec![10, 20]);
        assert!((hyp.score - 1.175).abs() < 1e-5, "score={}", hyp.score);
    }

    #[test]
    fn single_frame_emit_cost() {
        // One state, one final state, one emitting arc; 1 frame.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0)); // final, free
        f.set_start(s0);
        f.add_arc(s0, arc(1, 7, 0.3, s1)).unwrap();
        let em = vec![vec![f32::INFINITY, 0.4]]; // frame0: label1 = 0.4
        let hyp = WfstDecoder::new(&f).decode(&em).unwrap().unwrap();
        assert_eq!(hyp.words, vec![7]);
        assert!((hyp.score - 0.7).abs() < 1e-6, "{}", hyp.score);
    }

    // ---- T09: epsilon closure + cycle guard --------------------------------

    #[test]
    fn epsilon_arc_is_free_within_a_frame() {
        // 0 -(1:10/0.5)-> 1 -(eps:30/0.05)-> 2 -(2:20/0.25)-> 3(F/0.125)
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        let s2 = f.add_state(TropicalWeight::zero());
        let s3 = f.add_state(TropicalWeight::new(0.125));
        f.set_start(s0);
        f.add_arc(s0, arc(1, 10, 0.5, s1)).unwrap();
        f.add_arc(s1, arc(0, 30, 0.05, s2)).unwrap(); // epsilon
        f.add_arc(s2, arc(2, 20, 0.25, s3)).unwrap();
        let hyp = WfstDecoder::new(&f)
            .decode(&emission_lin())
            .unwrap()
            .unwrap();
        // 0.6 + 0.05 (eps) + 0.45 + 0.125 = 1.225, words [10, 30, 20].
        assert_eq!(hyp.words, vec![10, 30, 20]);
        assert!((hyp.score - 1.225).abs() < 1e-5, "{}", hyp.score);
    }

    #[test]
    fn epsilon_cycle_is_explicit_error() {
        // 0 -(eps/0)-> 1 -(eps/0)-> 0 : an epsilon cycle.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::new(0.0));
        let s1 = f.add_state(TropicalWeight::zero());
        f.set_start(s0);
        f.add_arc(s0, arc(0, 0, 0.0, s1)).unwrap();
        f.add_arc(s1, arc(0, 0, 0.0, s0)).unwrap();
        let em = vec![vec![f32::INFINITY, 0.0]];
        match WfstDecoder::new(&f).decode(&em) {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("epsilon cycle"), "{m}"),
            other => panic!("expected epsilon-cycle error, got {other:?}"),
        }
    }

    // ---- T10: backtrace / epsilon-removed output ---------------------------

    #[test]
    fn no_complete_path_returns_none() {
        // Graph needs 1 emitting arc but we give it 2 frames — no path consumes
        // exactly 2 frames and ends final.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(1, 5, 0.1, s1)).unwrap();
        let em = vec![vec![f32::INFINITY, 0.0], vec![f32::INFINITY, 0.0]];
        assert!(WfstDecoder::new(&f).decode(&em).unwrap().is_none());
    }

    // ---- emission contract (T07) -------------------------------------------

    #[test]
    fn short_emission_row_is_explicit_error() {
        let f = linear(); // uses ilabel up to 2 → rows need length >= 3
        let em = vec![vec![0.0, 0.1], vec![0.0, 0.1]]; // length 2 → too short
        match WfstDecoder::new(&f).decode(&em) {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("emission frame"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn nan_emission_is_explicit_error() {
        let f = linear();
        let em = vec![
            vec![f32::INFINITY, f32::NAN, 5.0],
            vec![f32::INFINITY, 5.0, 0.2],
        ];
        match WfstDecoder::new(&f).decode(&em) {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("NaN"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn invalid_fst_surfaces_from_decode() {
        // No final state → validate() rejects it through decode().
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        f.set_start(s0);
        f.add_arc(s0, arc(1, 1, 0.0, s0)).unwrap();
        let em = vec![vec![f32::INFINITY, 0.0]];
        assert!(matches!(
            WfstDecoder::new(&f).decode(&em),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T11: lattice well-formedness --------------------------------------

    #[test]
    fn lattice_is_well_formed() {
        let f = linear();
        let lat = WfstDecoder::new(&f).lattice(&emission_lin()).unwrap();
        assert!(lat.has_path());
        assert!(lat.is_acyclic());
        assert!(lat.is_well_formed());
        // start frame 0, final is the super-final sink (state None).
        let (sf, ss) = lat.node(lat.start());
        assert_eq!(sf, 0);
        assert_eq!(ss, Some(0));
        let (_, fs) = lat.node(lat.final_node());
        assert_eq!(fs, None);
    }

    // ---- T12: n-best -------------------------------------------------------

    /// Two parallel paths with distinct words + distinct cost (fixture design:
    /// no ties). A: [10,20]@1.175, B: [11,21]@1.225.
    fn nbest_graph() -> Fst<W> {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        let s2 = f.add_state(TropicalWeight::zero());
        let s4 = f.add_state(TropicalWeight::new(0.125));
        f.set_start(s0);
        f.add_arc(s0, arc(1, 10, 0.5, s1)).unwrap();
        f.add_arc(s1, arc(2, 20, 0.25, s4)).unwrap();
        f.add_arc(s0, arc(1, 11, 0.5, s2)).unwrap();
        f.add_arc(s2, arc(2, 21, 0.30, s4)).unwrap();
        f
    }

    #[test]
    fn nbest_is_sorted_unique_and_capped() {
        let f = nbest_graph();
        let hyps = WfstDecoder::new(&f)
            .n_best(5)
            .decode_nbest(&emission_lin())
            .unwrap();
        assert_eq!(hyps.len(), 2, "exactly two distinct paths exist");
        assert_eq!(hyps[0].words, vec![10, 20]);
        assert_eq!(hyps[1].words, vec![11, 21]);
        assert!(hyps[0].score <= hyps[1].score, "best-first by cost");
        assert!((hyps[0].score - 1.175).abs() < 1e-5);
        assert!((hyps[1].score - 1.225).abs() < 1e-5);
        // Unique word sequences.
        assert_ne!(hyps[0].words, hyps[1].words);
    }

    #[test]
    fn nbest_one_equals_best_path() {
        let f = nbest_graph();
        let d = WfstDecoder::new(&f);
        let best = d.decode(&emission_lin()).unwrap().unwrap();
        let nb = d.n_best(1).decode_nbest(&emission_lin()).unwrap();
        assert_eq!(nb.len(), 1);
        assert_eq!(nb[0].words, best.words);
        assert!((nb[0].score - best.score).abs() < 1e-6);
    }

    #[test]
    fn nbest_zero_is_empty() {
        let f = linear();
        let hyps = WfstDecoder::new(&f)
            .n_best(0)
            .decode_nbest(&emission_lin())
            .unwrap();
        assert!(hyps.is_empty());
    }

    // ---- beam pruning ------------------------------------------------------

    #[test]
    fn wide_beam_keeps_the_optimum() {
        // A large beam must not change the best path vs. no pruning.
        let f = nbest_graph();
        let no_prune = WfstDecoder::new(&f)
            .decode(&emission_lin())
            .unwrap()
            .unwrap();
        let wide = WfstDecoder::new(&f)
            .beam(100.0)
            .decode(&emission_lin())
            .unwrap()
            .unwrap();
        assert_eq!(no_prune.words, wide.words);
        assert!((no_prune.score - wide.score).abs() < 1e-6);
    }
}
