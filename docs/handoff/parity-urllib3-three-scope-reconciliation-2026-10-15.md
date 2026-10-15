# Three isolated parity urllib3 scopes — 2026-10-15

Prepared from accepted main `5ceb432545cc1d9bdf2ff5c5ff289d683276539a`,
separately from the DFN3/DAC/OWSM changes. This updates Baichuan inspection,
ChatTTS reference and VibeVoice 1.5B reference to urllib3 2.8.0 without
changing other dependencies, markers, source/model identities or approval
states. Canary's immutable cross-bound evidence scope is excluded rather
than restamped. The protected CosyVoice2 and Silero test trees are not
materialized in the private sparse clone.

## Exact record and current factual bindings

All three locks use the exact reviewed DFN3/PyPI 2.8.0 record. Manager
independently compares the ordered complete package lists and all root lock
fields, preserving duplicate platform variants rather than collapsing names.
Every other record, dependency edge, artifact and marker remains exact.

| Tree | Full records | Dependency edges | New lock SHA-256 |
|---|---:|---:|---|
| Baichuan inspection | 14 | 13 | `c88a3d8ef8488f84105d4530b56e522396eea9d9b931ec2916202f952aeb177d` |
| ChatTTS | 53 | 101 | `6492180bc5917fce1b17c7b806e3aa6116ac19bd6c3095792387719576f95551` |
| VibeVoice 1.5B | 44 | 70 | `c53133843ca81d3854e2676566b1f3b61a6b5d80a84b31aecf90f8362c3f5dd8` |

Baichuan package rows bind
`e8989e69b0e9563d8578ae40f7d35caacf84559296eea50009a0f4db7572caad`;
its resolution-marker digest is unchanged. ChatTTS inventory/rows/factual
license-record hashes are respectively
`0cd1d7fb4f0a961a10374d6b24a9427097732d97191583426e1c1e603c7f4308`,
`21437ac499cf61bfd0e5c390f9e6c0c6b5d141a179d83811db8284086f873e32`
and `ab305d332c9fcac73252f30e89c3929fb294cc343badf2a3c2324b889df3d124`.
VibeVoice package rows/factual license rows bind
`e74bb016e7bd12d09a37fc6b1be959a3bd9f3d9ee6872950d7c6c493537468b0`
and `91034b0a115c6c7cebede9f89375980e6541f33f59229d22cd178e3874c599f3`.

The corresponding current-source validators/worker scripts bind those new
facts. Only digest literals and the urllib3 identity/primary URL change;
execution/refusal logic, owners, signatures, shapes and numerical tolerances
do not. Old installed/native/reference evidence is not edited or promoted.
The urllib3 MIT finding reuses the exact full wheel/sdist license evidence in
[the accepted release review](../license-audit.md#shared-parity-urllib3-release-review-2026-10-05-pr-194),
not an inference from an unrelated project's policy exception or just a
classifier. It does not approve any complete model dependency closure.

## Independent focused replay

Manager independently passes the three project-owned Python self-tests and
all six affected shell self-tests using offline UV/Python 3.12 or Bash. These
are stdlib/source-contract tests, not upstream imports or model/Apple runs.

Actual current metadata paths also validate the new bindings:

- Baichuan `lock_dependency_audit()` authenticates the new lock and 14 rows,
  then reports `BLOCKED_UNREVIEWED_TRANSITIVE`.
- ChatTTS `validate_dependency_gate()` passes current inventory/rows/factual
  audit bindings and refuses at the unchanged affirmative-approval condition.
- VibeVoice `reference_environment_identity()` authenticates the new lock and
  rows while retaining `BLOCKED_UNREVIEWED_TRANSITIVE` and
  `BLOCKED_UNVERIFIED_API_SMOKE`.

Manager separately supplies the unchanged main's old 2.7.0 inputs to each
current validator. Baichuan and VibeVoice reject the exact old lock hash;
ChatTTS rejects the old inventory digest. These are intended stale-input
failures, not model or numerical results. Current pyproject hashes are
`6022d8b1d833333eff3d73a35c1b3da46b5675391de09c9068b78b6d4c084712`,
`8a6d4a25bdbb9641e59707813c4187c15d9ec53a2acbabad8c507b6b9f38b68f`
and `76c220b9c0f14731b38bd1e09e26b9d4591335d007f3fa44ee6a0293512ab4f1`
in table order.

The private three-scope source commit `a32af1a` was made in a narrow sparse
clone without materialized Git hooks. After integrating it as `da1efe8`,
manager explicitly ran the unchanged trusted `.githooks/pre-commit` from the
full integration clone: formatting and all four static checks passed.
This management record is committed there with the actual hook installed.
No claim is made that the earlier private commit itself ran the hook.

The normal pre-push entry passes all eight compliance regressions, then
refuses the deep Cargo path on the maintainer Mac before compilation.
Manager independently proves the complete `crates`, `tests`, Cargo root,
`.cargo`, `.config` and toolchain trees unchanged from the recorded green
VAST source replay `4263f41631bd15288f06d3cd5c6b83ea4a8fdf60` through
accepted main `5ceb4325` to this candidate. That replay recorded 8,223 passed,
zero failed and 105 ignored tests, Clippy and static gates; its owned worker
and storage were destroyed. The documented `VOKRA_SKIP_HOOKS=1` source-push
path may therefore reuse this exact Rust identity, not claim a new remote
run or bypass the source-contract tests and fresh hosted CI. Guards and their
allowlists are unchanged. The final six locks are independently compared
again as ordered full records, including all unrelated root fields/markers.

No resolver/sync/install, upstream/model/config/tokenizer acquisition or
execution, Cargo compilation, provider/credential/SSH operation, owner/legal
decision, artifact replacement or upload is performed. ChatTTS's AGPL/NC,
other native/license/owner holds and every `NO_UPLOAD` boundary remain.
Source acceptance and fresh integrated-head CI remain separate from the
campaign's real-weight CPU and final clean-HEAD Apple packet exit gates.
