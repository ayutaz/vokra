//! Weighted finite-state transducer data structure (M5-06 T04).
//!
//! A minimal, decode-only [`Fst`]: states, per-state out-arcs, per-state final
//! weights, and a single start state. This is **not** a general FST algebra
//! library — there is no `compose` / `determinize` / `minimize` (ADR M5-06 §1:
//! the HCLG graph is composed offline by the developer-side OpenFST toolchain
//! and Vokra reads the finished graph). It holds exactly what the
//! token-passing decoder ([`super::decoder`]) needs.
//!
//! Labels follow the OpenFST convention: `0` is **epsilon** (a non-emitting
//! transition), positive labels are real symbols. `ilabel` is the acoustic /
//! input symbol the decoder consumes a frame for; `olabel` is the output
//! (word) symbol emitted on the best path.

use crate::error::{Result, VokraError};

use super::semiring::Semiring;

/// A state index into [`Fst::states`]. OpenFST serialises this as `int32`; we
/// hold the non-negative index as `u32` and represent OpenFST's `kNoStateId`
/// (`-1`, "no start") as [`Option::None`] on [`Fst::start`].
pub type StateId = u32;

/// An input/output symbol. `0` is epsilon (OpenFST convention). OpenFST
/// serialises labels as signed `int32`; the reader rejects negative labels
/// (out of the decode-only scope) so the in-memory type is `u32`.
pub type Label = u32;

/// One transition: consume `ilabel`, emit `olabel`, pay `weight`, move to
/// `nextstate`. `ilabel == 0` is a non-emitting **epsilon** arc (traversed
/// within a frame, no acoustic cost).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc<W: Semiring> {
    /// Input (acoustic) label; `0` = epsilon.
    pub ilabel: Label,
    /// Output (word) label; `0` = epsilon (nothing emitted).
    pub olabel: Label,
    /// Arc weight (graph cost) in the semiring `W`.
    pub weight: W,
    /// Destination state index.
    pub nextstate: StateId,
}

impl<W: Semiring> Arc<W> {
    /// `true` if this is a non-emitting epsilon arc (`ilabel == 0`).
    #[inline]
    pub fn is_epsilon(&self) -> bool {
        self.ilabel == 0
    }
}

/// One FST state: its out-arcs and its final weight. A **non-final** state has
/// final weight [`Semiring::zero`] (`+∞` for tropical, "cannot terminate here").
#[derive(Debug, Clone)]
struct State<W: Semiring> {
    arcs: Vec<Arc<W>>,
    final_weight: W,
}

/// A decode-only weighted finite-state transducer.
///
/// Construct it either by hand ([`Fst::new`] + [`Fst::add_state`] +
/// [`Fst::add_arc`] + [`Fst::set_start`]) or by reading an OpenFST binary
/// (`super::reader::read_openfst_vector`). Always call [`Fst::validate`] before
/// decoding — the reader does not validate structural invariants that only
/// matter to the decoder.
#[derive(Debug, Clone)]
pub struct Fst<W: Semiring> {
    start: Option<StateId>,
    states: Vec<State<W>>,
}

impl<W: Semiring> Default for Fst<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Semiring> Fst<W> {
    /// A new, empty FST with no start and no states.
    pub fn new() -> Self {
        Self {
            start: None,
            states: Vec::new(),
        }
    }

    /// Adds a state with the given `final_weight` ([`Semiring::zero`] for a
    /// non-final state) and returns its [`StateId`].
    pub fn add_state(&mut self, final_weight: W) -> StateId {
        let id = self.states.len() as StateId;
        self.states.push(State {
            arcs: Vec::new(),
            final_weight,
        });
        id
    }

    /// Sets the start state. Overwrites any previous start.
    pub fn set_start(&mut self, s: StateId) {
        self.start = Some(s);
    }

    /// Sets (overwrites) the final weight of an existing state. Used by the
    /// reader, which creates every state first (so forward `nextstate`
    /// references resolve) and then fills in final weight + arcs per state.
    ///
    /// # Errors
    /// [`VokraError::InvalidArgument`] if `s` is out of range.
    pub fn set_final(&mut self, s: StateId, final_weight: W) -> Result<()> {
        let n = self.states.len();
        self.states
            .get_mut(s as usize)
            .map(|st| st.final_weight = final_weight)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "wfst: set_final on out-of-range state {s} (num_states={n})"
                ))
            })
    }

    /// Appends an out-arc to state `from`. Returns an error if `from` is out of
    /// range (the arc's `nextstate` is validated later by [`Fst::validate`], so
    /// arcs can be added before their target state exists).
    pub fn add_arc(&mut self, from: StateId, arc: Arc<W>) -> Result<()> {
        let n = self.states.len();
        let s = self.states.get_mut(from as usize).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "wfst: add_arc from out-of-range state {from} (num_states={n})"
            ))
        })?;
        s.arcs.push(arc);
        Ok(())
    }

    /// The start state, or `None` for an FST with no start (OpenFST
    /// `kNoStateId`).
    #[inline]
    pub fn start(&self) -> Option<StateId> {
        self.start
    }

    /// The number of states.
    #[inline]
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// The total number of arcs across all states.
    pub fn num_arcs(&self) -> usize {
        self.states.iter().map(|s| s.arcs.len()).sum()
    }

    /// The out-arcs of state `s`, in insertion / file order.
    ///
    /// # Errors
    /// [`VokraError::InvalidArgument`] if `s` is out of range.
    pub fn arcs_of(&self, s: StateId) -> Result<&[Arc<W>]> {
        self.states
            .get(s as usize)
            .map(|st| st.arcs.as_slice())
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("wfst: arcs_of out-of-range state {s}"))
            })
    }

    /// The final weight of state `s` ([`Semiring::zero`] if non-final).
    ///
    /// # Errors
    /// [`VokraError::InvalidArgument`] if `s` is out of range.
    pub fn final_weight(&self, s: StateId) -> Result<W> {
        self.states
            .get(s as usize)
            .map(|st| st.final_weight)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("wfst: final_weight out-of-range state {s}"))
            })
    }

    /// `true` if state `s` is final (final weight is not [`Semiring::zero`]).
    ///
    /// # Errors
    /// [`VokraError::InvalidArgument`] if `s` is out of range.
    pub fn is_final(&self, s: StateId) -> Result<bool> {
        Ok(!self.final_weight(s)?.is_zero())
    }

    /// Validates the structural invariants the decoder relies on (FR-EX-08 —
    /// every violation is an explicit error, never a silent skip):
    ///
    /// - a **start state** is set and in range (`start 未設定` is rejected);
    /// - every arc's `nextstate` is in range (no **dangling** transition);
    /// - at least one state is final (otherwise no path can ever complete);
    /// - no weight is `NaN` (a NaN would corrupt every `min`/`+` downstream —
    ///   the `不正 final` / invalid-arc-weight case).
    ///
    /// # Errors
    /// [`VokraError::InvalidArgument`] naming the first violation found.
    pub fn validate(&self) -> Result<()> {
        let n = self.states.len() as StateId;
        // Start present + in range.
        match self.start {
            None => {
                return Err(VokraError::InvalidArgument(
                    "wfst: FST has no start state (kNoStateId) — cannot decode".into(),
                ));
            }
            Some(s) if s >= n => {
                return Err(VokraError::InvalidArgument(format!(
                    "wfst: start state {s} out of range (num_states={n})"
                )));
            }
            Some(_) => {}
        }
        let mut any_final = false;
        for (idx, st) in self.states.iter().enumerate() {
            // Final weight sanity.
            if weight_is_nan(st.final_weight) {
                return Err(VokraError::InvalidArgument(format!(
                    "wfst: state {idx} has a NaN final weight"
                )));
            }
            if !st.final_weight.is_zero() {
                any_final = true;
            }
            // Arc targets in range + finite weights.
            for (a, arc) in st.arcs.iter().enumerate() {
                if arc.nextstate >= n {
                    return Err(VokraError::InvalidArgument(format!(
                        "wfst: state {idx} arc {a} has dangling nextstate {} (num_states={n})",
                        arc.nextstate
                    )));
                }
                if weight_is_nan(arc.weight) {
                    return Err(VokraError::InvalidArgument(format!(
                        "wfst: state {idx} arc {a} has a NaN weight"
                    )));
                }
            }
        }
        if !any_final {
            return Err(VokraError::InvalidArgument(
                "wfst: FST has no final state — no path can ever complete".into(),
            ));
        }
        Ok(())
    }
}

/// `true` if a weight is NaN. Semiring is generic, so we route through the
/// `approx_eq` contract: a NaN is not `approx_eq` to itself at any tolerance
/// (IEEE-754), whereas every non-NaN weight is `approx_eq` to itself.
fn weight_is_nan<W: Semiring>(w: W) -> bool {
    !w.approx_eq(w, f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::super::semiring::TropicalWeight;
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

    /// Builds the linear fixture graph: 0 -(1:10/0.5)-> 1 -(2:20/0.25)-> 2(F/0.125).
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

    #[test]
    fn accessors_report_structure() {
        let f = linear();
        assert_eq!(f.num_states(), 3);
        assert_eq!(f.num_arcs(), 2);
        assert_eq!(f.start(), Some(0));
        assert_eq!(f.arcs_of(0).unwrap().len(), 1);
        assert_eq!(f.arcs_of(2).unwrap().len(), 0);
        assert!(!f.is_final(0).unwrap());
        assert!(f.is_final(2).unwrap());
        assert_eq!(f.final_weight(2).unwrap(), TropicalWeight::new(0.125));
        assert!(f.arcs_of(0).unwrap()[0].olabel == 10);
    }

    #[test]
    fn epsilon_arc_is_flagged() {
        assert!(arc(0, 30, 0.05, 1).is_epsilon());
        assert!(!arc(1, 10, 0.5, 1).is_epsilon());
    }

    #[test]
    fn valid_fst_passes_validation() {
        assert!(linear().validate().is_ok());
    }

    #[test]
    fn missing_start_is_rejected() {
        let mut f = linear();
        f.start = None;
        match f.validate() {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("no start"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_start_is_rejected() {
        let mut f = linear();
        f.set_start(99);
        match f.validate() {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("out of range"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn dangling_nextstate_is_rejected() {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(1, 1, 0.5, 42)).unwrap(); // 42 does not exist
        match f.validate() {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("dangling nextstate"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn no_final_state_is_rejected() {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        f.set_start(s0);
        f.add_arc(s0, arc(1, 1, 0.5, s1)).unwrap();
        match f.validate() {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("no final state"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn nan_final_weight_is_rejected() {
        let mut f = linear();
        // Poison state 2's final weight with NaN.
        f.states[2].final_weight = TropicalWeight::new(f32::NAN);
        match f.validate() {
            Err(VokraError::InvalidArgument(m)) => assert!(m.contains("NaN"), "{m}"),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn add_arc_from_out_of_range_state_errs() {
        let mut f = Fst::<W>::new();
        f.add_state(TropicalWeight::zero());
        assert!(matches!(
            f.add_arc(7, arc(1, 1, 0.0, 0)),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn out_of_range_accessors_err() {
        let f = linear();
        assert!(f.arcs_of(9).is_err());
        assert!(f.final_weight(9).is_err());
        assert!(f.is_final(9).is_err());
    }
}
