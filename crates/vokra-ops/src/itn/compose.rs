//! `shortestpath(compose(<linear input>, G))` over Vokra's existing WFST types.
//!
//! **Opt-in**: compiled only under the `vokra-wfst` feature, because it borrows
//! `vokra_core::decode::wfst::{Fst, Arc, TropicalWeight}`, which are themselves
//! behind that feature. Everything else in [`super`] — the tagged-token parser,
//! the field-order tables, the header probe, the grammar container — compiles
//! unconditionally.
//!
//! # What upstream does, and what this is
//!
//! `runtime/processor/wetext_processor.cc` (WeTextProcessing, Apache-2.0):
//!
//! ```cpp
//! std::string Processor::Compose(const std::string& input, const StdVectorFst* fst) {
//!   StdVectorFst input_fst;
//!   compiler_->operator()(input, &input_fst);      // StringCompiler<StdArc>, BYTE tokens
//!   StdVectorFst lattice;
//!   fst::Compose(input_fst, *fst, &lattice);
//!   return ShortestPath(lattice);                  // ShortestPath(..., 1, true) + StringPrinter
//! }
//! ```
//!
//! Composing a **linear acceptor** with a transducer and taking the best path
//! does not need a general `compose` (which Vokra deliberately does not have —
//! ADR M5-06 §1). The reachable part of that composition is exactly the product
//! graph over `(input position, FST state)`:
//!
//! - an arc with `ilabel == 0` (epsilon) moves within a position — it consumes
//!   no input;
//! - an arc with `ilabel == b` fires only when the input byte at the current
//!   position is `b`, and moves to the next position;
//! - a path completes at `(input.len(), s)` for any final `s`, paying that
//!   state's final weight;
//! - the path cost is the tropical product (`⊗` = `+`) of the arc weights, and
//!   the best path is the tropical sum (`⊕` = `min`) over all of them.
//!
//! So this module walks that product graph directly. It never materialises the
//! lattice, which matters: a WeTextProcessing tagger has on the order of
//! 10^5–10^6 states and the composition only ever touches the reachable
//! frontier.
//!
//! # Byte labels, not characters
//!
//! `StringCompiler<StdArc>` and `StringPrinter<StdArc>` default to
//! `TokenType::BYTE`, so upstream grammars are transducers over **UTF-8 bytes**
//! with labels `1..=255` (`0` is reserved for epsilon). This module therefore
//! consumes `&[u8]` and emits `Vec<u8>`. A NUL byte in the input would be
//! indistinguishable from epsilon and is refused loudly.
//!
//! # Why Dijkstra, and the non-negative-weight requirement
//!
//! Epsilon arcs can form cycles inside a `cdrewrite`-derived grammar, so the
//! frame-synchronous decoder's topological epsilon closure (which rejects
//! epsilon cycles outright) is not usable here. Dijkstra handles cycles fine —
//! but only for **non-negative** weights. Every WeTextProcessing weight comes
//! from `pynutil.add_weight` with a positive cost, so this holds in practice;
//! a negative weight is nonetheless refused loudly at the moment it is
//! encountered rather than silently producing a wrong "best" path (FR-EX-08).
//!
//! # Tie-breaking is deterministic but not OpenFST-identical
//!
//! When two paths have exactly equal cost, this picks the one that relaxes
//! first under a state-ascending sweep. OpenFST's `ShortestPath` tie-breaking
//! is an implementation detail of its queue discipline, so equal-cost ties may
//! resolve differently there. Costs always agree; the *string* may differ only
//! among exactly-equal-cost alternatives. Said plainly rather than papered over.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use vokra_core::decode::wfst::{Fst, Label, StateId, TropicalWeight};
use vokra_core::error::{Result, VokraError};

/// The result of composing an input string with a grammar and taking the best
/// path.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposeOutcome {
    /// The concatenated non-epsilon output labels of the best path, as bytes.
    pub output: Vec<u8>,
    /// The tropical cost of the best path (arc weights plus the final weight).
    pub cost: f32,
    /// How many `(position, state)` product nodes were actually visited. A
    /// diagnostic for the composition's real cost on a large grammar.
    pub visited_nodes: usize,
}

/// One reached `(position, state)` product node, with its best-known cost and
/// the back-pointer that produced it.
#[derive(Debug, Clone)]
struct Node {
    cost: f32,
    /// `(predecessor node index, output label emitted on the incoming arc)`.
    prev: Option<(usize, Label)>,
}

/// Min-heap entry keyed on the IEEE-754 bit pattern of a **non-negative** f32.
///
/// For non-negative, non-NaN floats the unsigned bit pattern orders exactly as
/// the numeric value does, so this is a total order with no float comparison in
/// the heap. Non-negativity is guaranteed by the negative-weight rejection in
/// [`compose_shortest_path`].
#[derive(Debug, PartialEq, Eq)]
struct HeapItem {
    cost_bits: u32,
    state: StateId,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: `BinaryHeap` is a max-heap, we want the cheapest first.
        other
            .cost_bits
            .cmp(&self.cost_bits)
            .then_with(|| other.state.cmp(&self.state))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Composes `input` (as a linear byte acceptor) with `fst` and returns the
/// best path's output labels — the equivalent of upstream
/// `ShortestPath(Compose(StringCompiler(input), fst))`.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] — the FST fails [`Fst::validate`]; the
///   input contains a NUL byte (indistinguishable from epsilon in OpenFST's
///   byte mode); an arc or final weight is negative (breaks Dijkstra); or the
///   grammar has no path accepting the input, with the byte offset at which it
///   died.
/// - [`VokraError::UnsupportedOp`] — the best path emits an output label above
///   255, which cannot be a byte in OpenFST's `TokenType::BYTE` mode (the
///   grammar was compiled for a symbol-table token type Vokra does not carry).
pub fn compose_shortest_path(input: &[u8], fst: &Fst<TropicalWeight>) -> Result<ComposeOutcome> {
    fst.validate()?;
    if let Some(pos) = input.iter().position(|&b| b == 0) {
        return Err(VokraError::InvalidArgument(format!(
            "itn: input contains a NUL byte at offset {pos} — in OpenFST's byte token mode \
             label 0 is epsilon, so a NUL is not representable in a linear input acceptor"
        )));
    }

    let start = fst
        .start()
        .expect("validated FST has a start state (Fst::validate)");

    let mut arena: Vec<Node> = vec![Node {
        cost: 0.0,
        prev: None,
    }];
    // The live frontier for the current input position: state -> arena index.
    let mut layer: BTreeMap<StateId, usize> = BTreeMap::new();
    layer.insert(start, 0);

    let n = input.len();
    for pos in 0..=n {
        epsilon_close(fst, &mut arena, &mut layer)?;
        if pos == n {
            break;
        }
        let byte = Label::from(input[pos]);
        let mut next: BTreeMap<StateId, usize> = BTreeMap::new();
        for (&state, &node) in &layer {
            let base = arena[node].cost;
            for arc in fst.arcs_of(state)? {
                if arc.ilabel != byte {
                    continue;
                }
                let w = arc.weight.value();
                check_weight(w, state, "arc")?;
                if w.is_infinite() {
                    continue; // tropical zero = "no path"
                }
                relax(
                    &mut arena,
                    &mut next,
                    arc.nextstate,
                    base + w,
                    node,
                    arc.olabel,
                );
            }
        }
        if next.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "itn: the grammar accepts no path for this input — composition died at byte \
                 offset {pos} (of {n}). A WeTextProcessing grammar normally has a catch-all \
                 `char` rule, so an empty composition means the grammar and the input \
                 disagree (wrong language bundle, or a tagged stream fed to the tagger \
                 instead of the verbalizer)."
            )));
        }
        layer = next;
    }

    // Terminate: the cheapest final state in the last layer, plus its final weight.
    let mut best: Option<(f32, usize)> = None;
    for (&state, &node) in &layer {
        if !fst.is_final(state)? {
            continue;
        }
        let fw = fst.final_weight(state)?.value();
        check_weight(fw, state, "final")?;
        if fw.is_infinite() {
            continue;
        }
        let total = arena[node].cost + fw;
        if best.is_none_or(|(b, _)| total < b) {
            best = Some((total, node));
        }
    }
    let Some((cost, end)) = best else {
        return Err(VokraError::InvalidArgument(format!(
            "itn: the grammar consumed all {n} input bytes but ended in no final state — the \
             composition is non-accepting. Check that the grammar and the input agree \
             (tagger grammars accept raw text; verbalizer grammars accept a reordered \
             tagged-token stream)."
        )));
    };

    // Walk back-pointers, collecting non-epsilon output labels in reverse.
    let mut out_rev: Vec<u8> = Vec::new();
    let mut cursor = end;
    while let Some((prev, olabel)) = arena[cursor].prev {
        if olabel != 0 {
            let byte = u8::try_from(olabel).map_err(|_| {
                VokraError::UnsupportedOp(format!(
                    "itn: the best path emits output label {olabel}, which is not a byte \
                     (1..=255). Upstream WeTextProcessing prints grammar output with \
                     `StringPrinter<StdArc>` in `TokenType::BYTE` mode, so every output \
                     label must be a UTF-8 byte. A grammar compiled against a symbol-table \
                     token type is out of scope — re-emit it byte-tokenised. Upstream \
                     reference: https://github.com/wenet-e2e/WeTextProcessing \
                     (runtime/processor/wetext_processor.cc)."
                ))
            })?;
            out_rev.push(byte);
        }
        cursor = prev;
    }
    out_rev.reverse();

    Ok(ComposeOutcome {
        output: out_rev,
        cost,
        visited_nodes: arena.len(),
    })
}

/// Rejects a negative weight loudly: Dijkstra's correctness (and the bit-pattern
/// heap key) both require non-negative costs.
fn check_weight(w: f32, state: StateId, which: &str) -> Result<()> {
    if w < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "itn: state {state} has a negative {which} weight {w} — the composition uses \
             Dijkstra (so that epsilon CYCLES are supported, unlike the frame-synchronous \
             decoder's topological closure), and Dijkstra requires non-negative weights. \
             WeTextProcessing grammars only ever use positive `pynutil.add_weight` costs; \
             a negative weight is refused rather than silently yielding a wrong best path \
             (FR-EX-08)."
        )));
    }
    Ok(())
}

/// Relaxes `state` in `layer` to `cost`, creating its arena node on first reach.
/// Returns `true` when the cost improved (or the node is new).
fn relax(
    arena: &mut Vec<Node>,
    layer: &mut BTreeMap<StateId, usize>,
    state: StateId,
    cost: f32,
    prev_node: usize,
    olabel: Label,
) -> bool {
    match layer.get(&state) {
        Some(&idx) => {
            if cost < arena[idx].cost {
                arena[idx].cost = cost;
                arena[idx].prev = Some((prev_node, olabel));
                true
            } else {
                false
            }
        }
        None => {
            let idx = arena.len();
            arena.push(Node {
                cost,
                prev: Some((prev_node, olabel)),
            });
            layer.insert(state, idx);
            true
        }
    }
}

/// Epsilon closure of the current layer: Dijkstra restricted to `ilabel == 0`
/// arcs, seeded with every state already in the layer.
///
/// Unlike the frame-synchronous decoder, this tolerates epsilon **cycles** —
/// with non-negative weights, revisiting a state can never lower its settled
/// cost, so the standard settled-set argument applies.
fn epsilon_close(
    fst: &Fst<TropicalWeight>,
    arena: &mut Vec<Node>,
    layer: &mut BTreeMap<StateId, usize>,
) -> Result<()> {
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for (&state, &idx) in layer.iter() {
        heap.push(HeapItem {
            cost_bits: arena[idx].cost.to_bits(),
            state,
        });
    }
    // A state is settled the first time it is popped (Dijkstra invariant).
    let mut settled: BTreeSet<StateId> = BTreeSet::new();
    while let Some(item) = heap.pop() {
        if !settled.insert(item.state) {
            continue; // stale duplicate
        }
        let node = layer[&item.state];
        let base = arena[node].cost;
        for arc in fst.arcs_of(item.state)? {
            if arc.ilabel != 0 {
                continue;
            }
            let w = arc.weight.value();
            check_weight(w, item.state, "arc")?;
            if w.is_infinite() {
                continue;
            }
            if settled.contains(&arc.nextstate) {
                continue;
            }
            let cost = base + w;
            if relax(arena, layer, arc.nextstate, cost, node, arc.olabel) {
                heap.push(HeapItem {
                    cost_bits: cost.to_bits(),
                    state: arc.nextstate,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use vokra_core::decode::wfst::Arc;
    use vokra_core::decode::wfst::Semiring;

    fn arc(il: u32, ol: u32, w: f32, next: StateId) -> Arc<TropicalWeight> {
        Arc {
            ilabel: il,
            olabel: ol,
            weight: TropicalWeight::new(w),
            nextstate: next,
        }
    }

    /// `"ab"` -> `"XY"`, cost 0.5 + 0.25 + final 0.125.
    fn linear_transducer() -> Fst<TropicalWeight> {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        let s2 = f.add_state(TropicalWeight::new(0.125));
        f.set_start(s0);
        f.add_arc(s0, arc(u32::from(b'a'), u32::from(b'X'), 0.5, s1))
            .unwrap();
        f.add_arc(s1, arc(u32::from(b'b'), u32::from(b'Y'), 0.25, s2))
            .unwrap();
        f
    }

    #[test]
    fn transduces_a_linear_path_and_reports_its_cost() {
        let f = linear_transducer();
        let out = compose_shortest_path(b"ab", &f).unwrap();
        assert_eq!(out.output, b"XY");
        assert!(
            (out.cost - 0.875).abs() < 1e-6,
            "cost {} != 0.5+0.25+0.125",
            out.cost
        );
    }

    #[test]
    fn picks_the_cheaper_of_two_alternatives() {
        // Two parallel routes for "a": cheap emits "L", expensive emits "H".
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(u32::from(b'a'), u32::from(b'H'), 5.0, s1))
            .unwrap();
        f.add_arc(s0, arc(u32::from(b'a'), u32::from(b'L'), 1.0, s1))
            .unwrap();
        let out = compose_shortest_path(b"a", &f).unwrap();
        assert_eq!(out.output, b"L");
        assert!((out.cost - 1.0).abs() < 1e-6);
    }

    #[test]
    fn epsilon_arcs_emit_without_consuming_input() {
        // s0 -eps:'<'-> s1 -a:'a'-> s2 (final), so "a" -> "<a".
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        let s2 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(0, u32::from(b'<'), 0.1, s1)).unwrap();
        f.add_arc(s1, arc(u32::from(b'a'), u32::from(b'a'), 0.2, s2))
            .unwrap();
        let out = compose_shortest_path(b"a", &f).unwrap();
        assert_eq!(out.output, b"<a");
    }

    #[test]
    fn trailing_epsilons_can_reach_the_final_state() {
        // Consuming "a" lands on a non-final state; an epsilon then reaches final.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::zero());
        let s2 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(u32::from(b'a'), u32::from(b'A'), 0.0, s1))
            .unwrap();
        f.add_arc(s1, arc(0, u32::from(b'!'), 0.0, s2)).unwrap();
        assert_eq!(compose_shortest_path(b"a", &f).unwrap().output, b"A!");
    }

    #[test]
    fn epsilon_cycles_terminate_and_do_not_loop_forever() {
        // s0 -eps/1.0-> s0 (self-loop) plus the real path. The topological
        // closure used by the frame-synchronous decoder would REJECT this graph;
        // Dijkstra settles s0 once and moves on.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(0, 0, 1.0, s0)).unwrap();
        f.add_arc(s0, arc(u32::from(b'a'), u32::from(b'A'), 0.0, s1))
            .unwrap();
        let out = compose_shortest_path(b"a", &f).unwrap();
        assert_eq!(out.output, b"A");
        assert!((out.cost - 0.0).abs() < 1e-6);
    }

    #[test]
    fn deleting_transducer_emits_nothing() {
        // ilabel 'a', olabel 0 (epsilon) — a pynutil.delete equivalent.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(u32::from(b'a'), 0, 0.0, s1)).unwrap();
        assert!(compose_shortest_path(b"a", &f).unwrap().output.is_empty());
    }

    #[test]
    fn multibyte_utf8_passes_through_byte_by_byte() {
        // '人' is E4 BA BA; an identity transducer over those three bytes.
        let bytes = "人".as_bytes().to_vec();
        let mut f = Fst::new();
        let mut prev = f.add_state(TropicalWeight::zero());
        f.set_start(prev);
        for (i, &b) in bytes.iter().enumerate() {
            let final_w = if i + 1 == bytes.len() {
                TropicalWeight::new(0.0)
            } else {
                TropicalWeight::zero()
            };
            let next = f.add_state(final_w);
            f.add_arc(prev, arc(u32::from(b), u32::from(b), 0.0, next))
                .unwrap();
            prev = next;
        }
        let out = compose_shortest_path(&bytes, &f).unwrap();
        assert_eq!(String::from_utf8(out.output).unwrap(), "人");
    }

    #[test]
    fn unaccepted_input_is_loud_with_the_byte_offset() {
        let f = linear_transducer();
        let Err(err) = compose_shortest_path(b"aq", &f) else {
            panic!("expected an error when the grammar rejects the input");
        };
        let msg = format!("{err}");
        assert!(msg.contains("accepts no path"), "{msg}");
        assert!(msg.contains("offset 1"), "{msg}");
    }

    #[test]
    fn ending_on_a_non_final_state_is_loud() {
        let f = linear_transducer();
        let Err(err) = compose_shortest_path(b"a", &f) else {
            panic!("expected an error when the path ends on a non-final state");
        };
        assert!(format!("{err}").contains("no final state"), "{err}");
    }

    #[test]
    fn negative_weights_are_refused_loudly() {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(u32::from(b'a'), u32::from(b'A'), -1.0, s1))
            .unwrap();
        let Err(err) = compose_shortest_path(b"a", &f) else {
            panic!("expected an error for a negative arc weight");
        };
        let msg = format!("{err}");
        assert!(msg.contains("negative"), "{msg}");
        assert!(msg.contains("Dijkstra"), "{msg}");
    }

    #[test]
    fn nul_byte_input_is_refused_loudly() {
        let f = linear_transducer();
        let Err(err) = compose_shortest_path(b"a\0b", &f) else {
            panic!("expected an error for a NUL byte in the input");
        };
        assert!(format!("{err}").contains("NUL"), "{err}");
    }

    #[test]
    fn non_byte_output_label_is_loud() {
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        let s1 = f.add_state(TropicalWeight::new(0.0));
        f.set_start(s0);
        f.add_arc(s0, arc(u32::from(b'a'), 4096, 0.0, s1)).unwrap();
        let Err(err) = compose_shortest_path(b"a", &f) else {
            panic!("expected an error for an out-of-byte-range output label");
        };
        let msg = format!("{err}");
        assert!(msg.contains("4096"), "{msg}");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    #[test]
    fn empty_input_still_needs_a_final_state() {
        // Start is final: the empty string composes to the empty output.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::new(0.25));
        f.set_start(s0);
        let out = compose_shortest_path(b"", &f).unwrap();
        assert!(out.output.is_empty());
        assert!((out.cost - 0.25).abs() < 1e-6);
    }

    #[test]
    fn invalid_fst_is_rejected_before_any_work() {
        // No final state at all — Fst::validate catches this.
        let mut f = Fst::new();
        let s0 = f.add_state(TropicalWeight::zero());
        f.set_start(s0);
        assert!(compose_shortest_path(b"a", &f).is_err());
    }

    #[test]
    fn visited_node_count_stays_proportional_to_the_reachable_frontier() {
        // 3 nodes for "ab" over the 3-state linear transducer: one per position.
        let f = linear_transducer();
        let out = compose_shortest_path(b"ab", &f).unwrap();
        assert_eq!(out.visited_nodes, 3);
    }
}
