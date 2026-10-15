# Dia urllib3 minimal reconciliation — 2026-10-15 JST

This source-only review concerns Dependabot [PR #253](https://github.com/ayutaz/vokra/pull/253)
at `554c1d3ffc7ffeb49cdda08cc17172060237b6bb`, against accepted main
`b5ae1b3cff45383bc00339606cc2f0a5d245e1c6`. Bot-head CI does not certify
the corrected candidate or the model reference environment.

## Minimal lock update and coordinated consumers

The bot updates urllib3 2.7.0 to 2.8.0, but also removes 14 dependency markers
from the two platform-specific Torch 2.13.0 records. Luna restores all 14
entries exactly to base. Manager independently parses base, bot and corrected
locks with offline Python 3.12 stdlib TOML: all 30 non-urllib3 complete package
records and root fields equal base; urllib3 equals the bot record exactly.
All three locks have 31 package records. The final lock delta is three
insertions and three deletions, not an environment resolution or installation.

| Input | SHA-256 |
|---|---|
| Base dedicated lock | `06d1f30607934c822c12fdef1db62369f2af0a72372e19d2ba782ffb95583449` |
| Bot dedicated lock | `df4de47954bea86939467388dd8b121afc8d25d614f536763cc3196d3fdc3612` |
| Corrected dedicated lock | `8356b43556bb3c94a75556b57b839cce15d4b4e899d8df033404c6b2ccf1c5de` |
| Unchanged dedicated project | `4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb` |

Leaving only the lock changed would break current factual consumers. Luna
therefore updates seven current-lock hash literals across six files: the
dumper and its self-test, evidence validator, dependency auditor, lock
contract, unsigned owner-scope producer and upstream compatibility probe.
Manager compares each entire source body with base and proves that these
seven literals are the only source differences. The README adds a dated
supersession rather than overwriting historical inputs or measurements.

The exact urllib3 2.8.0 artifact URLs, sizes and digests match the previously
reviewed [FACodec update](facodec-urllib3-minimal-reconciliation-2026-10-15.md),
which records the [official PyPI metadata](https://pypi.org/pypi/urllib3/2.8.0/json)
and [fixed-release MIT primary license](https://github.com/urllib3/urllib3/blob/b1d30ab61fe0db8f11092805e8c5ac43e091064a/LICENSE.txt).
Reusing these identical artifact facts does not validate the complete Dia
Python/native environment or approve its execution.

## Independent checks and remaining boundary

Manager independently passes all five existing offline self-tests: dependency
audit, unsigned owner-scope, upstream compatibility probe, dumper and evidence
validator. No target package or upstream module is imported. The current
`build_lock_audit` reads actual corrected project bytes and emits 31 all-lock
rows / 27 active Linux rows, including urllib3 2.8.0, while preserving
`BLOCKED_UNREVIEWED_TRANSITIVE` and `NOT_PERFORMED` license classification.
An old-lock evidence mutation is rejected for the intended exact reason:
`dependency audit project bytes are stale`. `git diff --check` passes.

The ordinary pre-commit passes all five checks. Ordinary pre-push passes all
eight compliance regressions, then refuses its deep Cargo path on the
maintainer Mac because the nested lock is outside the existing neutral-path
allowlist. No local Cargo compile/test starts and no hook is changed. Manager
independently compares the integrated candidate with recorded green VAST
source `4263f41631bd15288f06d3cd5c6b83ea4a8fdf60`: `crates`, `tests`, root
Cargo manifests/lock, `.cargo`, `.config` and root Rust/build configuration
have no differences. This is source identity, not a new workspace replay or
an installed Python compatibility result. For this corresponding source push,
the documented `VOKRA_SKIP_HOOKS=1` path uses that remote evidence and the
focused offline tests above; corrected-head hosted CI remains mandatory.

No historical installed/native/API evidence or owner signature is relabeled
for this new lock. In particular, the earlier `f9e095f7` audit, its 147
publisher-license file records and 164 native files are not new measurements.
The historical license manifest, pending owner decisions and unconditional
upstream compatibility refusal are unchanged. Real-weight CPU/reference,
final Apple packets and CPU/Metal/no-fallback evidence remain unfinished.

No model/config/tokenizer acquisition or execution, package sync/install,
heavy local Cargo, new paid worker or publication occurs. Protected CosyVoice2
owner material and the Silero model fixture are not materialized or inspected.
Fresh corrected-head CI, required context/application pairs and normal merge
protection remain separate acceptance gates. No public inventory row advances.
