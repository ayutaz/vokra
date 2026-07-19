<!--
  日本語で書いていただいて構いません。このテンプレートが英語なのは、GitHub の
  PR テンプレートが言語切り替えを持たないためです（言語別に複製すると
  レビュー面が二重になります）。
  Japanese is welcome. This template is English-only because GitHub PR
  templates have no language switch; duplicating it would double the review
  surface. See CONTRIBUTING.md for the rules behind each checkbox.
-->

## What this changes

<!-- One or two sentences. What is different after this PR, and why. -->

## How it was verified

<!--
  Commands you actually ran, with their results. Please do not list commands
  you did not run — an unverified claim costs a reviewer more than an honest
  "not tested on X".
-->

```
```

---

## Checklist

Tick what applies. If something does not apply, say so rather than deleting
the line — "N/A: no new dependency" is a useful review signal.

### Always

- [ ] `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` pass
- [ ] `cargo test --workspace` passes
- [ ] **Zero external dependencies** — `bash scripts/check-zero-deps.sh` passes.
      The root `Cargo.lock` still resolves to `vokra-*` crates only
      (`NFR-DS-02`). If this PR needed functionality from outside, it uses one
      of the two sanctioned routes in [CONTRIBUTING.md](https://github.com/ayutaz/vokra/blob/main/CONTRIBUTING.md) §7
      (a first-party optional feature, or an isolated `integrations/`
      workspace) rather than adding a crate.
- [ ] **No silent fallback** — any path this PR cannot handle raises an
      explicit error rather than quietly degrading (`FR-EX-08`).
- [ ] **No parity tolerance was widened to make a check pass.** Tolerances are
      architectural bounds, not knobs. If one genuinely had to move, the PR
      explains the bound it is derived from.

### If this PR adds or changes model support

See [CONTRIBUTING.md](https://github.com/ayutaz/vokra/blob/main/CONTRIBUTING.md) §4 — all four are required in **this
same PR**:

- [ ] [docs/license-audit.md](https://github.com/ayutaz/vokra/blob/main/docs/license-audit.md) updated — licence of
      code *and* weights, commercial usability, training-data provenance
- [ ] Model-zoo policy respected — weights that are non-commercial or of
      unclear training-data rights stay out of the official zoo and are
      reachable only behind an explicit research flag
- [ ] For TTS / VC: the
      [docs/legal-compliance.md](https://github.com/ayutaz/vokra/blob/main/docs/legal-compliance.md) checklist was
      followed
- [ ] [NOTICE](https://github.com/ayutaz/vokra/blob/main/NOTICE) updated if the addition carries attribution or
      distribution terms

### Design red lines

Confirm this PR crosses none of them
([CONTRIBUTING.md](https://github.com/ayutaz/vokra/blob/main/CONTRIBUTING.md) §5, reasoning in
[docs/architecture.md](https://github.com/ayutaz/vokra/blob/main/docs/architecture.md) §3):

- [ ] No ONNX graph loading in the runtime — ONNX stays in the offline
      converter only (`FR-LD-05`)
- [ ] No onnxruntime in the piper-plus inference path
- [ ] No eSpeak-NG (GPL-3.0)
- [ ] No NNAPI backend
- [ ] No soxr / rubberband (GPL)

### If this PR touches `unsafe`

- [ ] Every `unsafe` block carries a `// SAFETY:` comment
- [ ] The crate is already on the opt-out list in the workspace manifest —
      this PR does not add a new one (`NFR-RL-07`)

---

## Anything reviewers should look at closely

<!--
  Optional. Known rough edges, a decision you were unsure about, or a part
  you would like a second opinion on. Saying "I am not sure this is the right
  layer for this" is welcome and saves review cycles.
-->
