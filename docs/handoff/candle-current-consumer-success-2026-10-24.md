# Candle current consumer success — 2026-10-24

The current producer/consumer integration for PR
[#278](https://github.com/ayutaz/vokra/pull/278) succeeded at source head
`7b28175be8c1e608779f160b6d1beccd67d6b157`, which includes accepted main
`8bc58adbd010fb071295f3e44166745a480efc8c`.
This record establishes the bounded source-only integration, not PR merge,
model execution, numerical parity, license approval or Apple completion.

## Run, artifacts and independent readback

[Run 40065385239](https://github.com/ayutaz/vokra/actions/runs/40065385239),
attempt 1, completed both producer
[job 120787434925](https://github.com/ayutaz/vokra/actions/runs/40065385239/job/120787434925)
and consumer
[job 120798539752](https://github.com/ayutaz/vokra/actions/runs/40065385239/job/120798539752)
successfully. The GitHub run API and producer envelope both identify reusable
workflow commit `f0c4ef25a43ecfd42d7b502b8dd178d99ebde1fe`.

| Artifact | ID | ZIP bytes | GitHub API / downloaded ZIP SHA-256 |
|---|---:|---:|---|
| Application | 12549713896 | 259327 | `bdf12a083c2cb59d14666bef36178bb5c4e1991fa7ab9bfb3e7bf15b8c3d0942` |
| Envelope | 12549669205 | 1115 | `98f6c07b068df5c5a65cc2cfd3b874c23762916dbbcbd45ec5b40c5a9f62e926` |
| Consumer report | 12551995087 | 2847 | `5cce7763b69a6fae04acd140eb9c25e2a693aaa71ca51859d0de9c6cc8f5c44a` |

The manager downloaded only these small source-only ZIPs and independently
matched each archive's bytes and digest to the API. The application has exactly
eight authenticated members totaling 4,333,048 extracted bytes. Every member
matches the envelope size/hash and remains within 2 MiB; the compressed archive
remains within 4 MiB. The existing envelope schema validator succeeds offline.
This verifies the real size case that previously failed the extracted-byte cap.

Unaltered JSON members retained beyond Actions artifact expiry:

- [Envelope](evidence/candle-current-envelope-40065385239-2026-10-24.json):
  2444 bytes, SHA-256
  `36d2c94becc7653dc98bf7a85f8bd7e9e3bee8784b811a3623c9cfe340fb6871`.
- [Advisory report](evidence/candle-current-advisory-report-40065385239-2026-10-24.json):
  13853 bytes, SHA-256
  `ee4b04b5fe368709b3fd97820fbf05ba6c1e1c19ddaacba85b20009ff82a2ff1`.

## Executed tests and report scope

The consumer's controller suite ran 14 tests successfully and its envelope suite
ran four successfully. The initial artifact suite reported 18 tests with 14
fixture-dependent skips before download. After downloading the genuine current
application and envelope, all 18 artifact tests passed with no skips.

The actual source-only audit command exited 0 and the report records `PASS` for
`candle_current_final_lock`. It binds final-lock SHA-256
`21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6`,
154 lock records and the fixed RustSec database commit
`edb58f5974b83eb8302332821cb6d389777aa351`. The database was clean before and
after the command. The retained command output reports zero vulnerabilities
and zero warnings against that fixed database; the 17 source-absent packages
have the separately labeled fixed-database absence check.

This is not a current generic security scan. The report explicitly preserves
`root_runtime_audit=NOT_RUN`, `license_review=NOT_PERFORMED`,
`model_execution=NOT_AUTHORIZED`, `approval=null`, `upload=NO_UPLOAD`, and
`yanked_check=NOT_CHECKED`.

## Acceptance and history

The PR's remaining required CI and normal protected merge were still pending
when this record was written. PR #271 must receive its own fresh integration
after the reviewed action updates; this receipt is not that later result.
The [earlier producer readback](candle-current-producer-readback-2026-10-24.md)
and its old source identity remain dated evidence. Its cancelled consumer is
not counted as success, and the original extracted-cap failure is preserved.
