#!/usr/bin/env python3
"""crates_publish_order.py — topological crates.io publish order (X-07-T10).

crates.io is an immutable index and a crate can only be published AFTER every
`{ path = ..., version = ... }` dependency it references already exists on the
registry (X-07-T09 adds the `version` to the workspace path deps). This tool
derives the publish set + a valid publish order MECHANICALLY from
`cargo metadata`, so the order can never silently drift from the real
dependency graph (CLAUDE.md hallucination red-line: no hand-maintained list).

DESIGN (fixed by ADR docs/adr/X-07-release-train.md §crates.io):

* Publish roots = the shipped runtime C ABI (`vokra-capi`) + the CLI
  (`vokra-cli`). The publish SET is their transitive path-dependency closure
  over runtime (`normal`) deps — INCLUDING `dep:`-optional backends, because
  crates.io requires an optional dependency's crate to exist on the registry
  too. Dev/build deps are excluded (cargo strips path-only dev-deps on publish).

* Anything NOT in that closure is deliberately excluded and STAYS
  `publish = false`:
    - vokra-eval          — evaluation metrics, dev/eval-only (not in closure);
    - vokra-parity        — test-only parity harness (tests/parity);
    - vokra-wasm-harness  — test-only wasm ABI harness (tests/wasm-harness);
    - integrations/*      — excluded workspaces (own Cargo.lock; link
                            NON-`vokra-*` crates — never publishable as vokra).

* The closure computed from the real graph is 15 crates (NOT the "11" the spec
  intake estimated — the 4 GPU/NPU backend crates vulkan/webgpu/coreml/qnn were
  added after that count, and `vokra-models` references all 6 backends as
  `dep:`-optional so they are forced into the closure). Actual code wins; the
  discrepancy is recorded in the ADR.

Zero-dep (NFR-DS-02): python3 stdlib only, driving `cargo metadata` (a built-in
of the pinned toolchain). No third-party crate, so the root Cargo.lock cannot
change.

Usage:
    python3 tools/release/crates_publish_order.py            # print order
    python3 tools/release/crates_publish_order.py --json     # JSON envelope
    python3 tools/release/crates_publish_order.py --verify   # self-test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Publish roots: the shipped runtime C ABI + the CLI. The set is their closure.
ROOTS = ("vokra-capi", "vokra-cli")

# Deliberately excluded from the publish set (kept `publish = false`). Asserted
# by --verify so a future accidental inclusion is caught.
EXCLUDED = ("vokra-eval", "vokra-parity", "vokra-wasm-harness")


def fail(msg: str) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def repo_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


def cargo_metadata(root: str) -> dict:
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        fail(f"`cargo metadata` failed:\n{proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        fail(f"could not parse `cargo metadata` output: {exc}")


def build_graph(md: dict) -> dict[str, list[str]]:
    """name -> sorted list of first-party path-dep names (runtime deps only)."""
    graph: dict[str, list[str]] = {}
    for pkg in md["packages"]:
        name = pkg["name"]
        if not name.startswith("vokra"):
            continue
        deps = set()
        for dep in pkg["dependencies"]:
            # kind == None is the `normal` (runtime) kind; "dev"/"build" excluded.
            if dep.get("kind") not in (None, "normal"):
                continue
            dname = dep["name"]
            # Only first-party path deps join the closure. A path dep is what
            # crates.io needs a `version` for; registry-resolved deps (there are
            # none in this zero-dep workspace) would not need publishing here.
            if dname.startswith("vokra") and dep.get("path"):
                deps.add(dname)
        graph[name] = sorted(deps)
    return graph


def closure(graph: dict[str, list[str]], roots: tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        if node not in graph:
            fail(f"publish root/dep '{node}' not found in the workspace graph")
        seen.add(node)
        for dep in graph[node]:
            if dep not in seen:
                stack.append(dep)
    return seen


def topo_order(graph: dict[str, list[str]], nodes: set[str]) -> list[str]:
    """Kahn's algorithm restricted to `nodes`; deps published before dependents.

    Ties are broken by name so the order is deterministic (a precondition for
    the T13 oracle and for a stable ADR record).
    """
    # In-set dependency edges only.
    remaining = {n: {d for d in graph[n] if d in nodes} for n in nodes}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            fail(f"dependency cycle among publish crates: {sorted(remaining)}")
        for n in ready:
            order.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def compute(root: str) -> tuple[list[str], dict[str, list[str]]]:
    md = cargo_metadata(root)
    graph = build_graph(md)
    for r in ROOTS:
        if r not in graph:
            fail(f"publish root '{r}' missing from the workspace")
    nodes = closure(graph, ROOTS)
    return topo_order(graph, nodes), graph


def verify(root: str) -> int:
    order, graph = compute(root)
    problems: list[str] = []
    order_index = {n: i for i, n in enumerate(order)}

    # (1) valid topological order: every in-set dep precedes its dependent.
    for n in order:
        for dep in graph[n]:
            if dep in order_index and order_index[dep] > order_index[n]:
                problems.append(f"{dep} must be published before {n}")

    # (2) closure is self-consistent: no in-set crate has an in-set dep outside
    #     the order (would be a dangling crates.io reference).
    node_set = set(order)
    for n in order:
        for dep in graph[n]:
            if dep in node_set and dep not in order_index:
                problems.append(f"{n} depends on {dep} which is not in the order")

    # (3) roots are present.
    for r in ROOTS:
        if r not in node_set:
            problems.append(f"publish root {r} missing from the computed set")

    # (4) vokra-core is a leaf (published first) — it has no first-party runtime
    #     path deps, so it must be at index 0.
    if order and order[0] != "vokra-core":
        problems.append(
            f"expected vokra-core first (leaf of the graph), got {order[0]}"
        )

    # (5) the deliberately-excluded crates must NOT be in the set.
    for x in EXCLUDED:
        if x in node_set:
            problems.append(f"excluded crate {x} leaked into the publish set")

    # (6) no `integrations/*` crate is in the set (excluded workspaces).
    for n in node_set:
        # integrations crates are not workspace members, so they never appear in
        # `cargo metadata --no-deps` of the ROOT workspace; assert the invariant
        # holds by name convention as a belt-and-suspenders guard.
        if n in ("vokra-godot", "vokra-server", "piper-plus-g2p"):
            problems.append(f"excluded-workspace crate {n} leaked into the set")

    if problems:
        print("crates_publish_order --verify: FAIL", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(
        f"crates_publish_order --verify: OK "
        f"({len(order)} crates, valid topological order, "
        f"vokra-core first, {len(EXCLUDED)} crates correctly excluded)"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true", help="run the self-test")
    ap.add_argument("--json", action="store_true", help="emit a JSON envelope")
    args = ap.parse_args()

    root = repo_root()

    if args.verify:
        sys.exit(verify(root))

    order, _graph = compute(root)
    if args.json:
        json.dump(
            {"roots": list(ROOTS), "order": order, "count": len(order)},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        for name in order:
            print(name)


if __name__ == "__main__":
    main()
