//! Search lattice + best-path + n-best (M5-06 T11 + T12).
//!
//! The token-passing forward pass ([`super::decoder`]) produces a
//! [`WfstLattice`]: the reachable, acyclic, weighted subgraph of the
//! composition `emission ⊗ graph`. A lattice **node** is a `(frame, state)`
//! pair (plus one super-final sink); a lattice **arc** is a surviving graph
//! transition (emitting = advance one frame, or epsilon = stay in the frame),
//! carrying the accumulated cost `emission[frame][ilabel] ⊗ arc.weight` and the
//! output (word) label.
//!
//! This is a **decode output** structure, not a general FST-algebra object:
//! best-path is a single-source shortest path over the DAG, and n-best is
//! k-shortest-paths over the same DAG. There is no `compose` / `determinize`
//! here (ADR M5-06 §1) — that would be the general algebra explicitly out of
//! the M5-06 scope.
//!
//! # Determinism / tie-breaking
//!
//! Best-path and n-best are deterministic. When two paths have **equal** total
//! cost, the tie is broken by the lexicographic order of their lattice-arc
//! index sequence (arcs are created in a fixed frame → state(ascending) →
//! file-order during the forward pass, and every node's out-arcs are sorted by
//! `(source_state, fst_arc_index)`). Parity fixtures are engineered so the
//! optimum is *strictly* unique (ADR M5-06 §6), so parity never actually
//! depends on the tie order — the rule exists only to make the API
//! deterministic, mirroring the honesty discipline of
//! `feedback-honest-parity-atol`.

use super::fst::{Label, StateId};

/// A super-final sink node has this sentinel FST state (it is not a real FST
/// state — it collects the per-state final weights).
const SUPER_FINAL_STATE: i64 = -1;

/// One lattice arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatArc {
    /// Source lattice node index.
    pub from: usize,
    /// Destination lattice node index.
    pub to: usize,
    /// Input (acoustic) label consumed; `0` for epsilon and for the super-final
    /// arc.
    pub ilabel: Label,
    /// Output (word) label; `0` = nothing emitted.
    pub olabel: Label,
    /// Accumulated cost on this arc (`emission ⊗ graph`, or the final weight for
    /// a super-final arc).
    pub weight: f32,
    /// Deterministic ordering key `(source_fst_state, fst_arc_index)` used to
    /// sort out-arcs and to break path ties. `u32::MAX` arc-index marks the
    /// super-final arc.
    pub key: (u32, u32),
}

/// A weighted, acyclic search lattice (decode output).
#[derive(Debug, Clone)]
pub struct WfstLattice {
    /// `frames[i]` = the frame index of node `i` (`num_frames` for super-final).
    frames: Vec<usize>,
    /// `states[i]` = the FST state of node `i` ([`SUPER_FINAL_STATE`] for the
    /// super-final sink).
    states: Vec<i64>,
    /// All lattice arcs.
    arcs: Vec<LatArc>,
    /// `out_arcs[i]` = indices into [`Self::arcs`] leaving node `i`, sorted by
    /// arc `key`.
    out_arcs: Vec<Vec<usize>>,
    /// `in_arcs[i]` = indices into [`Self::arcs`] entering node `i`.
    in_arcs: Vec<Vec<usize>>,
    start: usize,
    final_node: usize,
    num_frames: usize,
}

/// Raw lattice pieces handed over by the forward pass, before trimming.
pub(super) struct RawLattice {
    pub frames: Vec<usize>,
    pub states: Vec<i64>,
    pub arcs: Vec<LatArc>,
    pub start: usize,
    pub final_node: usize,
    pub num_frames: usize,
}

impl WfstLattice {
    /// Builds a trimmed lattice from the forward pass's raw pieces.
    ///
    /// **Trimming** keeps only nodes on at least one start → super-final path
    /// (accessible *and* co-accessible), so the returned lattice contains no
    /// dead-ends — the well-formedness the n-best search and the T11 test
    /// expect. Nodes are re-indexed; arc order (hence tie-breaking) is
    /// preserved.
    pub(super) fn from_raw(raw: RawLattice) -> Self {
        let n = raw.frames.len();
        // Accessibility (reachable from start) over the raw arcs.
        let mut fwd = vec![false; n];
        {
            let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
            for a in &raw.arcs {
                out[a.from].push(a.to);
            }
            let mut stack = vec![raw.start];
            fwd[raw.start] = true;
            while let Some(u) = stack.pop() {
                for &v in &out[u] {
                    if !fwd[v] {
                        fwd[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        // Co-accessibility (can reach the super-final) over reversed arcs.
        let mut bwd = vec![false; n];
        {
            let mut rin: Vec<Vec<usize>> = vec![Vec::new(); n];
            for a in &raw.arcs {
                rin[a.to].push(a.from);
            }
            let mut stack = vec![raw.final_node];
            bwd[raw.final_node] = true;
            while let Some(u) = stack.pop() {
                for &v in &rin[u] {
                    if !bwd[v] {
                        bwd[v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        let keep: Vec<bool> = (0..n).map(|i| fwd[i] && bwd[i]).collect();
        // Old → new index map (stable order).
        let mut remap = vec![usize::MAX; n];
        let mut frames = Vec::new();
        let mut states = Vec::new();
        for i in 0..n {
            if keep[i] {
                remap[i] = frames.len();
                frames.push(raw.frames[i]);
                states.push(raw.states[i]);
            }
        }
        let new_n = frames.len();
        let mut arcs: Vec<LatArc> = Vec::new();
        for a in &raw.arcs {
            if keep[a.from] && keep[a.to] {
                arcs.push(LatArc {
                    from: remap[a.from],
                    to: remap[a.to],
                    ..*a
                });
            }
        }
        let mut out_arcs: Vec<Vec<usize>> = vec![Vec::new(); new_n];
        let mut in_arcs: Vec<Vec<usize>> = vec![Vec::new(); new_n];
        for (ai, a) in arcs.iter().enumerate() {
            out_arcs[a.from].push(ai);
            in_arcs[a.to].push(ai);
        }
        // Deterministic out-arc order: by arc key, then arc index.
        for outs in &mut out_arcs {
            outs.sort_by(|&x, &y| arcs[x].key.cmp(&arcs[y].key).then(x.cmp(&y)));
        }
        WfstLattice {
            frames,
            states,
            arcs,
            out_arcs,
            in_arcs,
            start: remap[raw.start],
            final_node: remap[raw.final_node],
            num_frames: raw.num_frames,
        }
    }

    /// Number of lattice nodes.
    pub fn num_nodes(&self) -> usize {
        self.frames.len()
    }

    /// Number of lattice arcs.
    pub fn num_arcs(&self) -> usize {
        self.arcs.len()
    }

    /// The start node index.
    pub fn start(&self) -> usize {
        self.start
    }

    /// The super-final node index.
    pub fn final_node(&self) -> usize {
        self.final_node
    }

    /// The number of acoustic frames the lattice spans.
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// All lattice arcs (read-only).
    pub fn arcs(&self) -> &[LatArc] {
        &self.arcs
    }

    /// `true` if the start node can reach the super-final (a path exists).
    pub fn has_path(&self) -> bool {
        self.num_nodes() > 0 && self.start != usize::MAX && self.final_node != usize::MAX
    }

    /// A topological order of the nodes (Kahn's algorithm, ties broken by
    /// ascending node index for determinism). The lattice is a DAG (emitting
    /// arcs advance the frame; within-frame epsilon arcs are acyclic — enforced
    /// by the decoder's epsilon-cycle guard), so this always succeeds; a
    /// residual in-degree would indicate a cycle bug and is reported by
    /// [`Self::is_acyclic`].
    fn topo_order(&self) -> Vec<usize> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let n = self.num_nodes();
        let mut indeg: Vec<usize> = self.in_arcs.iter().map(|v| v.len()).collect();
        let mut heap: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        for (i, &d) in indeg.iter().enumerate() {
            if d == 0 {
                heap.push(Reverse(i));
            }
        }
        let mut order = Vec::with_capacity(n);
        while let Some(Reverse(u)) = heap.pop() {
            order.push(u);
            for &ai in &self.out_arcs[u] {
                let v = self.arcs[ai].to;
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    heap.push(Reverse(v));
                }
            }
        }
        order
    }

    /// `true` if the lattice is acyclic (Kahn processed every node).
    pub fn is_acyclic(&self) -> bool {
        self.topo_order().len() == self.num_nodes()
    }

    /// Well-formedness for the T11 test: acyclic, has a path, and every node is
    /// both accessible (from start) and co-accessible (to super-final). The
    /// last holds by construction because [`Self::from_raw`] trims dead-ends.
    pub fn is_well_formed(&self) -> bool {
        if !self.is_acyclic() || !self.has_path() {
            return false;
        }
        let fwd = self.forward_distances();
        let bwd = self.backward_distances();
        (0..self.num_nodes()).all(|i| fwd[i].is_finite() && bwd[i].is_finite())
    }

    /// Shortest cost from the start node to every node (`+∞` if unreachable),
    /// computed by DAG relaxation in topological order.
    pub fn forward_distances(&self) -> Vec<f32> {
        let n = self.num_nodes();
        let mut dist = vec![f32::INFINITY; n];
        if self.start >= n {
            return dist; // no-path (fully trimmed) lattice
        }
        dist[self.start] = 0.0;
        for u in self.topo_order() {
            if !dist[u].is_finite() {
                continue;
            }
            for &ai in &self.out_arcs[u] {
                let a = &self.arcs[ai];
                let cand = dist[u] + a.weight;
                if cand < dist[a.to] {
                    dist[a.to] = cand;
                }
            }
        }
        dist
    }

    /// Shortest cost from every node to the super-final (`+∞` if it cannot be
    /// reached), computed by relaxing in reverse topological order.
    pub fn backward_distances(&self) -> Vec<f32> {
        let n = self.num_nodes();
        let mut dist = vec![f32::INFINITY; n];
        if self.final_node >= n {
            return dist; // no-path (fully trimmed) lattice
        }
        dist[self.final_node] = 0.0;
        let mut order = self.topo_order();
        order.reverse();
        for u in order {
            for &ai in &self.out_arcs[u] {
                let a = &self.arcs[ai];
                if !dist[a.to].is_finite() {
                    continue;
                }
                let cand = dist[a.to] + a.weight;
                if cand < dist[u] {
                    dist[u] = cand;
                }
            }
        }
        dist
    }

    /// The single best (minimum-cost) start → super-final path, as its output
    /// (word) label sequence (epsilon removed) and total cost. `None` if no
    /// path exists.
    ///
    /// Ties are broken deterministically: among incoming arcs achieving the
    /// same minimum, the one with the smallest `key` (then smallest arc index)
    /// wins — a Viterbi backtrace over the DAG.
    pub fn best_path(&self) -> Option<WfstHypothesis> {
        if !self.has_path() {
            return None;
        }
        let n = self.num_nodes();
        let mut dist = vec![f32::INFINITY; n];
        let mut back: Vec<Option<usize>> = vec![None; n];
        dist[self.start] = 0.0;
        for u in self.topo_order() {
            if !dist[u].is_finite() {
                continue;
            }
            for &ai in &self.out_arcs[u] {
                let a = &self.arcs[ai];
                let cand = dist[u] + a.weight;
                let improve = cand < dist[a.to];
                let tie = cand == dist[a.to]
                    && back[a.to].is_some_and(|prev| {
                        let pk = self.arcs[prev].key;
                        (a.key, ai) < (pk, prev)
                    });
                if improve || tie {
                    dist[a.to] = cand;
                    back[a.to] = Some(ai);
                }
            }
        }
        if !dist[self.final_node].is_finite() {
            return None;
        }
        // Backtrace.
        let mut words = Vec::new();
        let mut cur = self.final_node;
        while cur != self.start {
            let ai = back[cur]?;
            let a = &self.arcs[ai];
            if a.olabel != 0 {
                words.push(a.olabel);
            }
            cur = a.from;
        }
        words.reverse();
        Some(WfstHypothesis {
            words,
            score: dist[self.final_node],
        })
    }

    /// Up to `n` best paths, best-first, de-duplicated by output (word)
    /// sequence.
    ///
    /// A* over the DAG with the exact heuristic [`Self::backward_distances`],
    /// so complete paths pop in strictly non-decreasing total cost. Equal-cost
    /// ties are broken by the lattice-arc index sequence (see the module docs).
    /// Returns `[]` if no path exists or `n == 0`.
    ///
    /// A defensive cap bounds the number of queue pops; the small,
    /// distinct-path parity fixtures never approach it. General k-shortest on a
    /// large lattice (Eppstein) is future work — out of the M5-06 decode-only
    /// scope.
    pub fn nbest(&self, n: usize) -> Vec<WfstHypothesis> {
        if n == 0 || !self.has_path() {
            return Vec::new();
        }
        let h = self.backward_distances();
        if !h[self.start].is_finite() {
            return Vec::new();
        }

        // Min-heap of partial paths keyed by (estimated total, arc-index path).
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<Candidate>> =
            std::collections::BinaryHeap::new();
        heap.push(std::cmp::Reverse(Candidate {
            est: OrdF32(h[self.start]),
            cost: 0.0,
            node: self.start,
            path: Vec::new(),
        }));

        let mut out: Vec<WfstHypothesis> = Vec::new();
        let mut seen: Vec<Vec<Label>> = Vec::new();
        // Cap: generous relative to fixture size; documented as a scope bound.
        let cap = 1_000_000usize;
        let mut pops = 0usize;

        while let Some(std::cmp::Reverse(c)) = heap.pop() {
            pops += 1;
            if pops > cap {
                break;
            }
            if c.node == self.final_node {
                let words = self.words_of(&c.path);
                if !seen.contains(&words) {
                    seen.push(words.clone());
                    out.push(WfstHypothesis {
                        words,
                        score: c.cost,
                    });
                    if out.len() == n {
                        break;
                    }
                }
                continue;
            }
            for &ai in &self.out_arcs[c.node] {
                let a = &self.arcs[ai];
                if !h[a.to].is_finite() {
                    continue; // dead-end (should not happen post-trim, but safe)
                }
                let cost = c.cost + a.weight;
                let mut path = c.path.clone();
                path.push(ai);
                heap.push(std::cmp::Reverse(Candidate {
                    est: OrdF32(cost + h[a.to]),
                    cost,
                    node: a.to,
                    path,
                }));
            }
        }
        out
    }

    /// The output (word) labels of an arc-index path, epsilon removed.
    fn words_of(&self, path: &[usize]) -> Vec<Label> {
        path.iter()
            .filter_map(|&ai| {
                let ol = self.arcs[ai].olabel;
                (ol != 0).then_some(ol)
            })
            .collect()
    }

    /// The `(frame, state)` of a node, for tests / inspection. `state` is
    /// `None` for the super-final sink.
    pub fn node(&self, i: usize) -> (usize, Option<StateId>) {
        let s = self.states[i];
        (
            self.frames[i],
            if s == SUPER_FINAL_STATE {
                None
            } else {
                Some(s as StateId)
            },
        )
    }
}

/// One n-best result: the output (word) label sequence and its total path cost.
/// Mirrors [`crate::decode::BeamHypothesis`] (best-first, cost/score field) so
/// the two search primitives feel the same (ADR M5-06 §6).
#[derive(Debug, Clone, PartialEq)]
pub struct WfstHypothesis {
    /// Output (word) labels along the path, epsilon (`0`) removed.
    pub words: Vec<Label>,
    /// Total path cost (tropical `⊗` accumulation of emission + graph weights +
    /// the final weight).
    pub score: f32,
}

/// A* search candidate. `Ord` compares the estimated total first, then the
/// arc-index path lexicographically (the deterministic tie-break).
#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    est: OrdF32,
    cost: f32,
    node: usize,
    path: Vec<usize>,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.est
            .cmp(&other.est)
            .then_with(|| self.path.cmp(&other.path))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A total-ordered `f32` wrapper (IEEE-754 `total_cmp`) so costs can key a
/// `BinaryHeap`. Path costs here are always finite and non-NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
