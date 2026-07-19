# Good first tasks

**English** | [日本語](good-first-tasks.ja.md)

Self-contained starting points for a first contribution to Vokra. Each entry
has a file/line anchor or a reproduction command, acceptance criteria you can
check yourself, and a rough size — so you can decide before starting whether
it is worth your time.

**Last reviewed: 2026-07-20.**

## How to use this list

- **No claiming ritual.** Open a PR when you have something. If you want to
  avoid duplicate work on a larger item, open a *Question* issue first.
- **These are not tracked as GitHub Issues.** Work items live in the
  maintainer's ticket tree, so there is no issue number to reference — see
  [CONTRIBUTING.md](../CONTRIBUTING.md) §1.
- **Start from a green build.** `cargo test --workspace` should pass before
  you change anything; CONTRIBUTING has measured timings for what "normal"
  looks like.
- **Sizes** are rough: **XS** under an hour, **S** an hour or two, **M** half
  a day, **L** more than that.

## What is deliberately *not* here

This list is short on purpose. Entries are added when a genuinely
self-contained task exists — not to reach a number. Two categories are
excluded by policy, so please do not expect to find them:

- **Anything touching a numerical parity tolerance.** Tolerances are derived
  architectural bounds, not knobs; changing one is never a first task.
- **Anything requiring hardware the contributor cannot verify on.** For
  example the CUDA driver-version TODO at
  `crates/vokra-backend-cuda/src/sys.rs:284` needs a real NVIDIA GPU to
  confirm, so it stays with the maintainer rather than becoming a task
  somebody cannot finish.

Also out of scope for a first task: the zero-dependency invariant, the C ABI
surface, and GPU kernels.

---

## 1. Fix a stale documentation pointer in `compute.rs` — XS

**Where**: `crates/vokra-models/src/compute.rs:2`

The module documentation of the `Compute` seam points readers at a design
note that is not in the repository:

```rust
//! Imperative compute dispatcher for the native models (Phase 3 of the GPU
//! execution architecture; see `scratchpad/graph-engine-plan.md` §3).
```

`scratchpad/` is not tracked — `git ls-files scratchpad/` returns nothing —
so anybody following that reference finds nothing. The design it refers to is
now described publicly in [architecture.md](architecture.md) §2.

**What to do**: replace the dangling reference with a pointer to the public
description. It is the only such reference in the tree
(`grep -rn "scratchpad/" crates/ --include="*.rs"` returns exactly one line).

**Acceptance criteria**

- `grep -rn "scratchpad/" crates/ --include="*.rs"` returns nothing
- `cargo doc -p vokra-models` still builds
- `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` pass

**Why it is worth doing**: it is a two-line change that makes the most
architecturally important module in the model layer readable by someone who
does not have the private planning tree — which is the whole reason
`architecture.md` exists.

---

## 2. Add `--help` to the check scripts that lack it — S

**Where**: `scripts/check-*.sh`

The repository has an established convention: a check script supports
`--help` and prints its purpose, modes and exit codes. `check-zero-deps.sh`,
`check-forbidden-symbols.sh` and eleven others do not follow it yet.

Reproduce the list:

```bash
for s in scripts/check-*.sh; do grep -q -- '--help' "$s" || echo "$s"; done
```

At the time of writing that prints **13** scripts.

**What to do**: follow the shape already used by
`scripts/check-platform-support.sh` and `scripts/check-doc-references.sh` — a
`usage()` that re-prints the header comment block, and a flag case at the
bottom. **Do not change any check's behaviour or exit codes**; this is purely
about discoverability.

Doing two or three scripts is a perfectly good PR — you do not need to do all
thirteen.

**Acceptance criteria**

- each script you touched prints useful text for `--help` and exits 0
- running each script with no arguments behaves exactly as before, same exit
  code (verify against `git stash`)
- an unknown flag exits non-zero rather than silently running the check

---

## 3. Translate `README-swift-package.md` into Japanese — S

**Where**: `README-swift-package.md` (about 1.8 KB) → new `README-swift-package.ja.md`

Vokra maintains English and Japanese versions of its user-facing docs, but
the Swift Package README is English-only.

**What to do**: add the Japanese twin and cross-link both files using the
existing convention — see the third line of
[getting-started.md](getting-started.md) and
[getting-started.ja.md](getting-started.ja.md).

**Acceptance criteria**

- `README-swift-package.ja.md` exists and links back to the English page
- the English page links to it
- the two files have the same heading structure

**Note**: this is the smallest of the translation tasks and a good way to get
familiar with the PR process before taking a larger one.

---

## 4. Translate the Python binding README into Japanese — M

**Where**: `bindings/python/README.md` (about 5.9 KB) → new `bindings/python/README.ja.md`

Same convention as task 3. Larger, and contains code examples.

**What to do**: translate the prose; **leave code blocks, API names and
command lines exactly as they are**. Verify any command you translate the
description of still does what the description says.

**Acceptance criteria**

- the Japanese twin exists, both pages cross-link
- code blocks are byte-identical to the English page
- the two files have the same heading structure

---

## 5. Translate the Unity binding READMEs into Japanese — M

**Where**:

- `bindings/unity/com.vokra.unity/README.md` (about 4.4 KB)
- `bindings/unity/com.vokra.unity/Samples~/VadAsrTts/README.md` (about 4.1 KB)

Two related files; doing both in one PR is natural, but either alone is fine.

**Acceptance criteria**: as task 4, for each file translated.

---

## 6. Translate `CONTRIBUTING.md` into Japanese — L

**Where**: `CONTRIBUTING.md` (about 11 KB) → new `CONTRIBUTING.ja.md`

The largest translation gap, and deliberately left for a contributor rather
than done by the maintainer: translating the contribution guide is a real way
to find the parts of it that do not make sense, and reporting those is as
valuable as the translation.

**Please read this one before starting**: the English page was substantially
rewritten on 2026-07-20 (§1, §2, §6 and a new quick-start section). Work from
the current `main`, not from an older copy.

**What to do**: translate, cross-link, and **open a Question issue for
anything that reads as unclear or wrong** rather than smoothing it over in
the translation.

**Acceptance criteria**

- `CONTRIBUTING.ja.md` exists, both pages cross-link
- the two files have the same heading structure
- requirement IDs, command lines and file paths are left as-is
- links resolve — `bash scripts/check-community-docs.sh` reports no new
  broken links

---

## Maintaining this list

- **Owner**: the maintainer, reviewed at each quarterly Go/No-go review
  (see [governance/quarterly-reviews/](governance/quarterly-reviews/)).
- **When an entry is completed**: delete it in the same PR that completes it,
  and update *Last reviewed* above. Completed entries are not kept as a
  history — the git log is the history, and a list of already-done items
  makes the page harder to use.
- **When adding an entry**: it must satisfy every criterion in
  [CONTRIBUTING.md](../CONTRIBUTING.md) plus a file/line anchor or
  reproduction command, explicit acceptance criteria, and a size. An entry
  that cannot state its acceptance criteria is not ready to be listed.
- **If this list is empty**, that is a valid state and means no
  self-contained task currently exists — not that the page needs filling.
- The Japanese twin, [good-first-tasks.ja.md](good-first-tasks.ja.md), must
  be updated in the same PR; `scripts/check-community-docs.sh` fails if the
  two drift apart.
