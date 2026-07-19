# SPDX SBOM reproducibility evidence (2026-07-19)

**cc-33** from the M4-residual audit — pre-material for **M4-15-T10**, whose
formal close is owner ratification. This is the evidence, not the ratification.

The audit corrected the original framing: the SBOM is **not committed**, so
"regenerate and `cmp` against the repo copy" is not a possible test. The
meaningful question is whether the same commit produces a byte-identical SBOM
from a **different checkout at a different absolute path** — which is what the
handoff (`docs/handoff/m4-15.md` §(a)) actually asks an owner to confirm on
their release machine.

## Result — byte-identical, and demonstrably not vacuous

All runs, both checkouts detached at `53d216c`:

```sh
python3 scripts/sbom/generate_spdx.py --package vokra-capi \
  --no-default-features --features vulkan \
  --doc-name vokra-capi-cpu-vulkan-only --output <path>
```

`--output` is required; the generator writes no document without it. The script
resolves the repo it describes from its **own** location (`repo_root()` walks up
from `__file__`), not from the working directory — which is what makes check 2
below a genuine cross-checkout comparison rather than one checkout measured
twice.

| # | comparison | result |
|---|---|---|
| 1 | same path, two consecutive runs | **identical** |
| 2 | **different absolute path, same commit** (`git worktree` under `/private/tmp/...` vs the repo worktree under `/Users/...`) | **identical** — sha256 `c0a2fb3c...` |
| 3 | `TZ=Asia/Tokyo LC_ALL=de_DE.UTF-8 LANG=de_DE.UTF-8` | **identical** — no wall clock, no locale-formatted numbers |
| 4 | `SOURCE_DATE_EPOCH=1000000000` | **differs**, as designed: `created` moves `2026-07-19T03:49:29Z` -> `2001-09-09T01:46:40Z` |
| 5 | default features instead of vulkan-only | **differs**: 7 packages vs 8 (`vokra-backend-vulkan` present only in the vulkan build) |
| 6 | local absolute paths in the output | **0 occurrences** in either checkout's SBOM |

Every row above was reproduced a second time, in a separate session against a
freshly created `git worktree`, and came out identical — including the
`c0a2fb3c…` digest, the `2026-07-19T03:49:29Z` / `2001-09-09T01:46:40Z` pair in
check 4, and the 8-vs-7 package split in check 5. The committed
`vokra-capi-cpu-vulkan-only.spdx.json` `cmp`s equal to both runs.

Checks 4-6 exist so check 2 cannot be passed by a document that is simply
constant. The SBOM does track its inputs (feature resolution is live, check 5),
does honour the reproducible-build epoch (check 4), and does not embed the
producing machine's paths (check 6).

Determinism mechanism, confirmed against the generated document:

- `created` = `2026-07-19T03:49:29Z`, exactly the HEAD commit timestamp of
  `53d216c` (`1784432969` = `2026-07-19T12:49:29+09:00`) — not the wall clock.
- `documentNamespace` =
  `https://github.com/ayutaz/vokra/spdx/vokra-capi-cpu-vulkan-only/53d216c3089f2ca9c6cdfd64e44cb8bedcc8fbe0`
  — full git SHA, no UUID.
- `creators` = `Tool: vokra-generate-spdx` — no hostname or user.

The in-repo self-test `scripts/sbom/test-generate-spdx.sh` also still passes
(**9 passed, 0 failed**), including its own b4/b5 same-runner determinism and
b8 zero-dep isolation checks.

## Docker leg — skipped

The audit made the Docker cross-image leg conditional on the daemon happening to
be up. `docker info` failed on this machine (daemon down), so that leg was
**not run**. It is not claimed as passing.

## What this does and does not establish

Establishes: at a fixed commit, the generator is insensitive to checkout path,
timezone, locale, and repeat invocation, and is sensitive to the inputs it
should be sensitive to.

Does **not** establish: reproducibility across a different *toolchain* version
or a different OS/architecture. `cargo tree` resolution is the input to this
document, and only one toolchain on one machine (macOS arm64) was exercised.
That is precisely the residue M4-15-T10 leaves to the owner's release
environment, and it remains open.

The generated document is committed alongside this report as
`vokra-capi-cpu-vulkan-only.spdx.json` so a future run has something concrete to
diff against — which also removes the gap that made the original cc-33 framing
impossible.

Environment: Apple M1 iMac, macOS arm64; python3 (stdlib only) + `cargo tree`.
