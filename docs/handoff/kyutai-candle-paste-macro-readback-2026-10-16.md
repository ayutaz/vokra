# Candle paste replacement: static macro frontier — 2026-10-16 JST

## Verdict and scope

**SOURCE_READBACK_ONLY_COMPATIBILITY_NOT_RUN.** This read-only investigation
follows the [accepted fixed-database advisory result](kyutai-candle-applied-advisory-result-2026-10-16.md).
It identifies concrete compatibility cases rather than repeating that audit
or promoting source inspection into compiler/model acceptance. Reviewed local
baseline is `d47617ecfe2bd326edaafc709cd47023b0687d46`; published implementation
remains `04574e71bf9f63c40ad62bf3c351bf65db8e6c51`. PR #255 stays draft.

Root rehashes the unchanged retained parent receipt
`234a530ba183472daf0a34444703b934df741a50520f4c48cf421be25ec3ff36`
and pastey receipt
`d4ca85e59f16d9ae6dc36f406473b6c534fe86895c353c3aca18930bce6edba2`.
The parent receipt inventories **157 Rust files** across sixteen archives,
but retains only ten literal paste-line contexts, in four archive identities:
the two gemm-common versions and the two Metal versions. This is neither a
complete retained Rust text corpus nor a complete macro-reachability proof.

## Why zero direct hits cannot remove a parent

The exact gemm-common source exports `gemm_cplx_def!`, which invokes exported
`__inject_mod_cplx!`; the latter calls the ordinary dependency path
`paste::paste!` and generates the module name using `[<$module _cplx>]`.

The fixed gemm-c32 and gemm-c64 sources at **both versions** demonstrate the
other half of that call chain. Their `src/lib.rs` imports gemm-common macros;
their complete 142-byte `src/gemm.rs` calls `gemm_common::gemm_cplx_def!`.
Those four callers have zero direct paste contexts in the retained scan.
Removing their dependency aliases on that basis would ignore the exported
macro expansion boundary. The reviewed sixteen-parent manifest plan therefore
stays intact; no dependency is removed as "unused" by this investigation.

The two Metal versions use a different boundary: `#[macro_use] pub extern
crate paste`, followed by `paste!` in type positions in `foreign_obj_type!`.
The generated `...Ref` names appear in `Deref`, associated target types,
Objective-C `Message` and `Debug` implementations. A test of only a free
function/item identifier would not cover those usages.

Primary fixed source locations:

- [gemm-common 0.17.1 export/expansion](https://github.com/sarah-quinones/gemm/blob/8f52cc1ae3569000d753369a7e61c97bc1f4ae2a/gemm-common/src/gemm.rs#L845),
  [0.18.2 export/expansion](https://github.com/sarah-quinones/gemm/blob/a9d8ed36e51ba039ffe8cb2512925202612e19fd/gemm-common/src/gemm.rs#L921).
- [gemm-c32 0.17.1 caller](https://github.com/sarah-quinones/gemm/blob/8f52cc1ae3569000d753369a7e61c97bc1f4ae2a/gemm-c32/src/gemm.rs),
  [gemm-c64 0.17.1 caller](https://github.com/sarah-quinones/gemm/blob/8f52cc1ae3569000d753369a7e61c97bc1f4ae2a/gemm-c64/src/gemm.rs),
  [gemm-c32 0.18.2 caller](https://github.com/sarah-quinones/gemm/blob/a9d8ed36e51ba039ffe8cb2512925202612e19fd/gemm-c32/src/gemm.rs),
  [gemm-c64 0.18.2 caller](https://github.com/sarah-quinones/gemm/blob/a9d8ed36e51ba039ffe8cb2512925202612e19fd/gemm-c64/src/gemm.rs).
- [Metal 0.27.0 source](https://github.com/gfx-rs/metal-rs/blob/f507da4686234e658f31de54d2aa0dfa8abd236b/src/lib.rs#L139),
  [Metal 0.29.0 source](https://github.com/gfx-rs/metal-rs/blob/b8a9e8cae6c7cea3264ffe790c3c1f2294f8464f/src/lib.rs#L129).

## Independent byte binding and transport distinction

The connector cannot resolve the original `sarah-ek/gemm` raw resources and
returns "file metadata without a file SHA". A public GitHub read of the
repository URL establishes its redirect to `sarah-quinones/gemm`; this is not
an arbitrary mirror. Root reads the **same fixed commits** there and streams
the exact source bytes through SHA-256 without saving/executing source files.
All twelve selected files agree with their original archive member hashes.

| Exact source member(s) | Files | Bytes each | SHA-256 |
|---|---:|---:|---|
| gemm-common 0.17.1 `src/gemm.rs` | 1 | 36262 | `0d27e51ddfd903f2f31bf8185401e45c79151430b7fd2c5bd59bd95a603e9911` |
| gemm-common 0.18.2 `src/gemm.rs` | 1 | 38552 | `f78a5fe1c112190e1216f7291864540aa02fb5c1fe6b27007b9549de2d3e1768` |
| gemm-c32/c64, both versions, `src/lib.rs` | 4 | 279 | `818b591e8f04ba4c9c41fb33bbeb7af2fa047a96f59b258b69674a37734d53ca` |
| gemm-c32, both versions, `src/gemm.rs` | 2 | 142 | `943af3ead1c597967f6ac38d6af912e902e24deab254955c1b9caf2998381239` |
| gemm-c64, both versions, `src/gemm.rs` | 2 | 142 | `6ec6e0a700096d415183b6f68b5f2f1bf4992a0be2fa1b38911ba925afad1b5d` |
| Metal 0.27.0 `src/lib.rs` | 1 | 17522 | `e5491ebe9c38e10a28ca5f3f5626bebd4c3901eb55400849c3a0921772f23afe` |
| Metal 0.29.0 `src/lib.rs` | 1 | 17413 | `2a60e138ddae705485ea04f40fe16bab6fc1b087c08eb9ccaea11fa3af4e3b5c` |

The selected files total **111,433 bytes**. This proves byte identity for
those members, not every upstream file, whole Git tree or complete macro
expansion. Existing receipt URLs/VCS declarations and historical evidence are
not rewritten because the current transport uses the canonical repository.

## Pastey declaration surface, not an equivalence result

Root independently rehashes all **five retained pastey Rust bodies**, totaling
**63,790 bytes**, against their recorded lengths/hashes. In retained
`src/lib.rs` (27,518 bytes, SHA-256
`768089e3d34148892706cca3e3fdf102b8c6a174f7fde96df1156ebe56423251`),
the public proc-macro entry points `paste`, `item` and `expr` are present;
the latter two delegate to `paste`. Thus no missing entry-point-name mismatch
is demonstrated by these specific callers. The ordinary dependency key must
still remain `paste` when the package name changes to pastey.

The retained segment parser handles identifier concatenation; the known
gemm-common/Metal snippets use that form. It also has an environment-variable
branch for input containing `env!`. The reviewed snippets do not request
that branch, but source inspection does not establish all reachable inputs,
compiler behavior, span/hygiene equivalence or an inert executable closure.
Only selected declaration/expansion code is read here; rehashing the other
bodies is not a claim to have reviewed every source line or run their tests.

## Next executable acceptance boundary

Before approving actual adoption, define and review compatibility cases for:

1. The unchanged dependency key with the real pastey 0.2.3 proc-macro, not a
   renamed fake paste implementation.
2. Exported cross-crate macro expansion reaching the caller's `paste` alias,
   including gemm-c32 and gemm-c64 at both fixed versions.
3. Identifier generation inside Metal's type/trait/associated-type positions,
   keeping both versions and the genuine `macro_use` import route.
4. The real pinned Linux CPU source/feature closure and build/probe side
   effects, after the corresponding source/license/native/operator gates.
   A small syntax smoke test does not replace full package/API compilation.

These are test requirements, **not execution authorization or passed tests**.
The existing audit-only GitHub permission does not approve building/executing
upstream proc-macros or build scripts. No compile, package installation,
model/config/tokenizer acquisition, paid worker, Apple provisioning or
publication occurs in this investigation. A read-only search naming
`prepare_paste_overlay.py` is refused by the local-model guard; it is not
retried through a different reader or used to change that rule. Independent
JSON/official-source investigation continues without accessing that refused
file through another route.

Source/native/license/operator decisions, independent full-PCM reference,
authorized real-weight CPU and final reviewed clean-HEAD/fresh Apple packets
still belong to the original step-4 objective for all unresolved rows. No
model count/status, owner signature, protected manifest or numerical bound
changes. Scaleway remains last.
