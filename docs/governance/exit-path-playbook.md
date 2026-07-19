# Exit path playbook

What to do if a quarterly Go/No-go review returns **No-go**.

This exists because the failure mode it guards against is well documented in
this field: a project stops being maintained without saying so, users discover
it months later from a stale dependency, and the work that was worth keeping
is lost with the rest. Writing the wind-down procedure while the project is
healthy costs an afternoon; writing it during a wind-down does not happen.

Nothing here is a prediction. Selecting a path is a maintainer decision, taken
per [quarterly-review-runbook.md](quarterly-review-runbook.md) §6.2 and
recorded in the review record.

## Principles

1. **Announce before you stop.** The worst outcome is silence — an
   unmaintained runtime that still looks alive is actively harmful to anyone
   who adopts it.
2. **Preserve what is separable.** Vokra's speech operators, the GGUF
   metadata conventions and the parity harness have value independent of the
   runtime as a whole. Land them somewhere they can survive.
3. **Leave users a route out.** A migration note beats a deprecation notice.
4. **Do not delete.** Archive the repository read-only; do not remove it.
   Downstream forks and citations depend on it resolving.

## Path selection

Which path fits depends on *why* the switch fired.

| Firing switch | Most plausible path | Why it follows |
|---|---|---|
| **J** — HA Voice does not adopt Vokra for Wyoming | **A. Wyoming / Home Assistant integration** | The Wyoming server implementation already exists. If the obstacle is adoption rather than capability, contributing it as a Wyoming implementation preserves the work in the ecosystem it was built for. |
| **F** — Candle covers the model set natively / **E** — a major vendor ships a speech runtime under Apache 2.0 | **B. Merge into Candle** or **C. Acquisition by HF / ggml-org** | If the ecosystem has absorbed the capability, the differentiated remainder is the speech-specific operator set, not the runtime. Land the operators where they will be maintained. |
| **C / D / K** — engagement, committers, or addressable market | **D. Post-mortem and archive** | These say the project did not find users, not that the technology failed. There is nothing to merge; there is something to write down. |
| **L** — burnout / funding | **D**, optionally after **A/B/C** | Handover requires energy. If there is none left, archive first and revisit later rather than starting a negotiation that cannot be finished. |

A caution that predates this document: **"we will merge upstream" is not a
plan by itself.** The receiving project has to want it. Paths A–C are only
real once someone on the other side has agreed; until then the honest default
is D.

---

## Path A — Wyoming Protocol / Home Assistant integration

**Preconditions**: the Wyoming server passes protocol-level end-to-end tests
against a real client; licensing is clear (Apache-2.0, no GPL contamination).

1. Open a conversation with the Home Assistant / Wyoming maintainers *before*
   announcing anything — establish whether an implementation contribution is
   wanted, and in what form.
2. Split the server out into a standalone, independently buildable artefact so
   adopting it does not require adopting the whole runtime.
3. Document the model conversion route: whatever they adopt must be usable
   without the private planning tree.
4. If accepted, redirect the repository README at the new home and archive.
5. If declined, fall through to D and say so in the post-mortem — a declined
   handover is information the next person will want.

## Path B — Merge into Candle as an audio extension

**Preconditions**: the operators build without the rest of Vokra; parity tests
travel with them; the licence is compatible.

1. Identify the separable set — the speech operators, STFT/iSTFT/mel
   frontend, codec decode paths, and the parity fixtures that prove them
   correct. Operators without their parity tests are not worth donating; the
   tests are most of the value.
2. Confirm the receiving project wants them before doing porting work.
3. Port incrementally, upstream-first, so partial completion still leaves
   value behind.
4. Keep attribution and `NOTICE` obligations intact through the move.

## Path C — Acquisition by HuggingFace / ggml-org

**Preconditions**: someone on the other side has expressed interest. Do not
build a data room speculatively.

1. Prepare the inventory: what exists, what is proven, what is not, and the
   honest state of each backend. **Overstating maturity here is worse than
   walking away** — it converts a wind-down into a reputation problem.
2. Licensing and attribution audit — every third-party term the codebase
   carries.
3. Clarify what transfers: repository, name, package namespaces, and what
   the maintainer will and will not continue to do afterwards.
4. Whatever the outcome, publish the post-mortem (path D) anyway.

## Path D — Post-mortem and orderly archive

Always available, and the honest default when A–C have no counterparty. This
path is a success condition, not a failure one: a clearly-ended project that
documented what it learned is more useful to the field than one that faded.

1. **Write the post-mortem.** What was attempted, what worked, what did not,
   and what the numbers actually were. Specifically worth recording:
   - which parts of the ONNX-alternative thesis held up and which did not;
   - the real cost of the zero-dependency invariant, and whether it paid;
   - measured performance against the reference implementations, including
     where Vokra lost;
   - the adoption numbers, unrounded.
2. **Publish it** — repository README and wherever the project has an
   audience. Do not bury it in a changelog.
3. **Give users a migration route.** Name the alternatives honestly, including
   the ones this project was positioned against.
4. **Freeze deliberately**: final release, `CHANGELOG` entry stating the
   project is no longer maintained, README banner at the top, and an
   invitation for a maintainer to step forward if anyone wants to.
5. **Archive the repository read-only.** Do not delete it, do not rename it,
   do not release the package namespaces to someone else.
6. Record the decision and the date in the review record.

---

## What is *not* an exit path

- **Silent abandonment.** Commits stop, issues go unanswered, the README still
  says the project is active. This is the outcome every step above exists to
  prevent.
- **Deleting the repository.** Forks, citations and lockfiles break, and the
  post-mortem's value is destroyed with it.
- **A deprecation notice with no migration route.** Tells users they have a
  problem without helping them solve it.
- **Handing the namespace to an unvetted party.** A package name with existing
  install base is a supply-chain asset; transferring it carelessly creates a
  risk for users who trusted the project.
