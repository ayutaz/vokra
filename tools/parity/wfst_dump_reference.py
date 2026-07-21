#!/usr/bin/env python3
"""M5-06 wfst_decode parity reference dumper (T13a + T13b).

Generates the committed OpenFST **binary graph fixtures** and the **reference**
best-path / score / n-best that `crates/vokra-core/tests/parity_wfst.rs` checks
Vokra's from-scratch WFST decoder against.

Independent oracle (numerical-parity discipline)
------------------------------------------------
The reference is produced by **real OpenFST** — the same Apache-2.0 codebase
Vokra deliberately does NOT link at runtime. We use the OpenFST **command-line
tools** (`fstcompile`, `fstcompose`, `fstshortestpath`, `fstprint`, `fstinfo`),
not the `pywrapfst` Python extension, because for Python 3.9 / macOS-arm64 there
is no installable `pywrapfst` / `openfst_python` / `pynini` wheel and building
them from source is impractical (probed 2026-07-20). The CLI tools are an
equally independent oracle — the T13a未確定事項 #4 explicitly left the oracle
choice open ("pywrapfst ... か Kaldi のデコーダを reference にするか"). This
choice is recorded in ADR M5-06.

If the OpenFST CLI is not found this script **aborts loudly** (it never
falls back to a self-computed / self-mirror reference — a self-mirror verifies
nothing, per the numerical-parity skill). Install with `brew install openfst`
(Apache-2.0). The generated fixtures are committed, so the Rust parity test
needs no OpenFST at all — CI reads the committed reference.

The token-passing decoder computes, over T acoustic frames, the min-cost path
of the composition `E ⊗ G` where:
  * G = the decode graph (this fixture's `.fst`), and
  * E = a per-frame emission "sausage": from frame-state t to t+1, one arc per
        acoustic label k with cost `emission[t][k]`.
OpenFST's `shortestpath(compose(E, G))` is exactly that min-cost path, so it is
the reference for `WfstDecoder::decode`; `--nshortest=N` gives the n-best.

Usage
-----
    python3 tools/parity/wfst_dump_reference.py
        [--out crates/vokra-core/tests/parity/fixtures/m5-06]

Emission convention (matches the Rust decoder, T07): `emission[frame][ilabel]`
is the acoustic **cost** (lower better); index 0 (epsilon) is never consumed and
is written as a large sentinel.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# --- OpenFST CLI discovery (loud abort if absent) --------------------------

FST_TOOLS = ["fstcompile", "fstcompose", "fstshortestpath", "fstprint", "fstinfo"]


def require_openfst():
    # Homebrew installs to /opt/homebrew/bin (arm) or /usr/local/bin (intel).
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.isdir(extra) and extra not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
    missing = [t for t in FST_TOOLS if shutil.which(t) is None]
    if missing:
        sys.exit(
            "FATAL: OpenFST CLI tools not found: {}\n"
            "  The parity reference MUST come from real OpenFST (independent oracle;\n"
            "  a self-computed reference would verify nothing — numerical-parity skill).\n"
            "  Install: brew install openfst   (Apache-2.0, dev-time only, not a runtime dep).\n"
            "  This script never falls back to a self-mirror.".format(", ".join(missing))
        )
    ver = subprocess.run(["fstinfo", "--help"], capture_output=True, text=True)
    return ver.returncode == 0


# --- fixture definitions ----------------------------------------------------
#
# Each graph is a list of arc rows "from to ilabel olabel weight" and final rows
# "state weight". Emissions are T rows; index 0 = epsilon sentinel (unused).
# Fixtures are engineered so the optimum (and each n-best rank) is STRICTLY
# unique — parity never depends on tie-breaking (ADR M5-06 §6).

SENTINEL = 1000000000.0  # emission[*][0]: epsilon, never consumed.


def linear_fixture():
    # 0 -(1:10/0.5)-> 1 -(2:20/0.25)-> 2(F/0.125).  best = [10,20] @ 1.175
    graph = [
        "0\t1\t1\t10\t0.5",
        "1\t2\t2\t20\t0.25",
        "2\t0.125",
    ]
    emission = [
        [SENTINEL, 0.1, 5.0],
        [SENTINEL, 5.0, 0.2],
    ]
    return "linear", graph, emission, 2  # nbest count


def epsilon_fixture():
    # start-eps + mid-eps:
    # 0 -(eps:40/0.02)-> 1 -(1:10/0.5)-> 2 -(eps:30/0.05)-> 3 -(2:20/0.25)-> 4(F/0.125)
    # best = [40,10,30,20] @ 1.245
    graph = [
        "0\t1\t0\t40\t0.02",
        "1\t2\t1\t10\t0.5",
        "2\t3\t0\t30\t0.05",
        "3\t4\t2\t20\t0.25",
        "4\t0.125",
    ]
    emission = [
        [SENTINEL, 0.1, 5.0],
        [SENTINEL, 5.0, 0.2],
    ]
    return "epsilon", graph, emission, 1


def nbest_fixture():
    # Two parallel, distinct-word, distinct-cost paths sharing the frame count.
    # A: [10,20] @ 1.175   B: [11,21] @ 1.225
    graph = [
        "0\t1\t1\t10\t0.5",
        "1\t4\t2\t20\t0.25",
        "0\t2\t1\t11\t0.5",
        "2\t4\t2\t21\t0.30",
        "4\t0.125",
    ]
    emission = [
        [SENTINEL, 0.1, 5.0],
        [SENTINEL, 5.0, 0.2],
    ]
    return "nbest", graph, emission, 3


FIXTURES = [linear_fixture, epsilon_fixture, nbest_fixture]


# --- OpenFST helpers --------------------------------------------------------


def compile_fst(text_rows, path, tmp):
    txt = os.path.join(tmp, "in.txt")
    with open(txt, "w") as f:
        f.write("\n".join(text_rows) + "\n")
    subprocess.run(
        ["fstcompile", "--arc_type=standard", "--fst_type=vector", txt, path],
        check=True,
    )


def emission_fst(emission, path, tmp):
    """Build the per-frame emission acceptor E (labels 1..L over T frames)."""
    rows = []
    T = len(emission)
    for t, row in enumerate(emission):
        for k in range(1, len(row)):  # skip index 0 (epsilon sentinel)
            rows.append(f"{t}\t{t + 1}\t{k}\t{k}\t{repr(row[k])}")
    rows.append(f"{T}\t0.0")
    compile_fst(rows, path, tmp)


def fstprint(path):
    return subprocess.run(["fstprint", path], check=True, capture_output=True, text=True).stdout


def initial_state(path):
    out = subprocess.run(["fstinfo", path], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("initial state"):
            return int(line.split()[-1])
    raise RuntimeError("no initial state in fstinfo")


def parse_printed(text):
    """Parse fstprint output → (arcs, finals). arc = (src,dst,il,ol,w); final=(state,w)."""
    arcs, finals = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) >= 4:
            src, dst, il, ol = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            w = float(parts[4]) if len(parts) >= 5 else 0.0
            arcs.append((src, dst, il, ol, w))
        else:
            state = int(parts[0])
            w = float(parts[1]) if len(parts) >= 2 else 0.0
            finals.append((state, w))
    return arcs, finals


def enumerate_paths(path):
    """DFS-enumerate every start→final path of an acyclic FST (the nshortest
    output). Returns list of (olabels, total_cost), each path's cost = sum of arc
    weights + final weight. OpenFST did the n-best selection; we only serialise
    the resulting paths."""
    text = fstprint(path)
    arcs, finals = parse_printed(text)
    start = initial_state(path)
    final_w = {s: w for s, w in finals}
    out_arcs = {}
    for (src, dst, il, ol, w) in arcs:
        out_arcs.setdefault(src, []).append((dst, il, ol, w))

    results = []

    def dfs(node, words, cost):
        if node in final_w:
            results.append((list(words), cost + final_w[node]))
        for (dst, il, ol, w) in out_arcs.get(node, []):
            if ol != 0:
                words.append(ol)
            dfs(dst, words, cost + w)
            if ol != 0:
                words.pop()

    dfs(start, [], 0.0)
    return results


def reference_nbest(graph_path, emission, n, tmp):
    e_path = os.path.join(tmp, "e.fst")
    emission_fst(emission, e_path, tmp)
    comp = os.path.join(tmp, "comp.fst")
    with open(comp, "wb") as f:
        subprocess.run(["fstcompose", e_path, graph_path], check=True, stdout=f)
    nsh = os.path.join(tmp, "nsh.fst")
    with open(nsh, "wb") as f:
        subprocess.run(
            ["fstshortestpath", f"--nshortest={n}", comp], check=True, stdout=f
        )
    paths = enumerate_paths(nsh)
    # Best-first by cost, then dedup by word sequence (keep the min = first).
    paths.sort(key=lambda p: p[1])
    seen, out = set(), []
    for words, cost in paths:
        key = tuple(words)
        if key not in seen:
            seen.add(key)
            out.append((words, cost))
    return out


# --- expected-file writer ---------------------------------------------------


def write_expected(out_dir, name, graph_path, emission, nbest):
    arcs, finals = parse_printed(fstprint(graph_path))
    start = initial_state(graph_path)
    num_states = max(
        [a[0] for a in arcs] + [a[1] for a in arcs] + [f[0] for f in finals] + [0]
    ) + 1

    lines = []
    lines.append(f"# M5-06 wfst parity fixture `{name}` — generated by")
    lines.append("#   tools/parity/wfst_dump_reference.py (real OpenFST CLI oracle).")
    lines.append("# Do not edit by hand; regenerate. See ADR M5-06 + fixtures README.")
    lines.append("VERSION 1")
    lines.append(f"GRAPH {name}.fst")
    # --- structure (T06 reader verification), canonicalised by sorting ---
    lines.append(f"START {start}")
    lines.append(f"NUMSTATES {num_states}")
    for state, w in sorted(finals):
        lines.append(f"FINAL {state} {repr(w)}")
    for (src, dst, il, ol, w) in sorted(arcs):
        lines.append(f"ARC {src} {il} {ol} {repr(w)} {dst}")
    # --- emission (fed to the Rust decoder) ---
    T = len(emission)
    L = len(emission[0]) if emission else 0
    lines.append(f"FRAMES {T} {L}")
    for t, row in enumerate(emission):
        lines.append("EMIT " + str(t) + " " + " ".join(repr(v) for v in row))
    # --- reference best-path + n-best (from OpenFST) ---
    ref = reference_nbest(graph_path, emission, nbest, tempfile.mkdtemp())
    best_words, best_cost = ref[0]
    lines.append("BEST " + repr(best_cost) + " " + " ".join(str(w) for w in best_words))
    for words, cost in ref:
        lines.append("NBEST " + repr(cost) + " " + " ".join(str(w) for w in words))

    path = os.path.join(out_dir, f"{name}.expected")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path, ref


def main():
    ap = argparse.ArgumentParser(description="M5-06 wfst parity reference dumper")
    # repo root = .../tools/parity/<this> → up 3 levels.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    default_out = os.path.join(
        repo_root,
        "crates",
        "vokra-core",
        "tests",
        "parity",
        "fixtures",
        "m5-06",
    )
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    require_openfst()
    os.makedirs(args.out, exist_ok=True)

    for fixture in FIXTURES:
        name, graph, emission, nbest = fixture()
        tmp = tempfile.mkdtemp()
        graph_path = os.path.join(args.out, f"{name}.fst")
        compile_fst(graph, graph_path, tmp)
        expected_path, ref = write_expected(args.out, name, graph_path, emission, nbest)
        print(f"[{name}] wrote {graph_path} ({os.path.getsize(graph_path)} bytes)")
        print(f"[{name}] wrote {expected_path}")
        for words, cost in ref:
            print(f"    ref: {words} @ {cost}")

    print("\nOK — fixtures generated with the real OpenFST oracle.")


if __name__ == "__main__":
    main()
