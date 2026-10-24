# Candle current producer evidence readback — 2026-10-24

This record covers the source-only producer at PR [#278](https://github.com/ayutaz/vokra/pull/278)
head `a4a9c5b09baba4951b3b702066e47e966850abc3`. It does not establish
consumer success, model execution, numerical parity, publication eligibility,
or Apple CPU/Metal completion.

## Observed producer and authenticated inputs

[Run 40060353787](https://github.com/ayutaz/vokra/actions/runs/40060353787)
attempt 1 has a successful producer
[job 120772293257](https://github.com/ayutaz/vokra/actions/runs/40060353787/job/120772293257).
The GitHub run API reports the PR source head above and the reusable producer
workflow at merge commit `69aef2357d15f6ed443e4e5965435427f14c6c66`.
Both match the [unaltered producer envelope](evidence/candle-current-producer-envelope-2026-10-24.json).

| Artifact | ID | ZIP bytes | GitHub API / downloaded ZIP SHA-256 |
|---|---:|---:|---|
| Application evidence | 12548826112 | 259315 | `d716b3820fdddf95002ad0cc174e9bc0a9404335bbc409260092f15ee51e902d` |
| Current-run envelope | 12548343978 | 1111 | `69c58bedbab22041602620af316eb2155cbc6329a91a0afe088155f8af2073f6` |

The envelope JSON is 2444 bytes with SHA-256
`047edbe644727aa287cccd0f97f76f8f94dbc747df83f1b15e128a8097337fba`.
The application ZIP has the exact eight envelope member names, no duplicate or
nested paths, and 4,333,048 uncompressed bytes. Every member is within the
existing 2 MiB member bound and matches its authenticated size and SHA-256.
The compressed ZIP remains within the existing 4 MiB bound. The existing
`current_artifact.validate_envelope` schema validation also succeeds offline.

This is genuine current-run evidence of the size case addressed by the PR:
the extracted total exceeds 4 MiB while the compressed artifact does not.
It is not a substitute for the real consumer's complete validation.

The application receipt explicitly retains `source_only=true`,
`status=SOURCE_APPLICATION_REVIEW_PENDING`, `approval=null`,
`model_execution=NOT_AUTHORIZED`, and `upload=NO_UPLOAD`.
Its source receipt identifies official Moshi source
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` with fixed Git blob authentication.
These source facts do not approve the registry pastey identity or model/legal
scope.

## Remaining acceptance work

At this readback, consumer
[job 120783585702](https://github.com/ayutaz/vokra/actions/runs/40060353787/job/120783585702)
is queued. Its success is not yet established. The full current consumer,
required checks, current base integration, and normal protected merge remain
necessary. The old extracted-cap failure remains a separate historical result.

Only the small source-only evidence ZIPs were downloaded. Verification used
Python 3.12 through `uv run`, standard-library ZIP/JSON/hash inspection, and the
reviewed envelope validator. No model artifact, dependency installation,
workspace Cargo command, model inference, or cloud instance was used.
