# MeloTTS minimal urllib3 lock reconciliation — 2026-10-15 JST

> **Accepted-main integration at 19:49 UTC:** PR #249 is normally accepted
> as main `6f135822075f9b98de08dfb7329a88d786d36531`, and the MeloTTS
> correction normally merges that main at private `69b4a0a9`. Fresh complete
> CI is still required for the corrected published head. The original
> preparation snapshot below is retained; it is not a new model verdict.

At the 19:37 UTC preparation snapshot:

Baseline main is `731592f97251f52f0e80ea81d211d9dde6b52d71`.
Dependabot PR #248 starts at `58766652c1b97c4e9230713899d4e63ae77b45e5`;
the separately reviewed correction is private commit
`f4813f02db4ee48f7d7f2f7eb7ddb01b7962a9e0`. It is not yet a published
or CI-accepted replacement head.

The original 29-added/29-removed lock diff contains three urllib3 record
changes and 26 unrelated dependency-edge marker removals. The correction
restores those 26 Linux x86_64 markers. The resulting diff against baseline
changes only urllib3's version, sdist record and wheel record, retaining
Dependabot's exact 2.8.0 URLs, hashes, sizes and upload times.

Manager independently parses both locks with stdlib tomllib and verifies all
23 package names, all dependency edges, resolution/supported markers and
the unchanged project environment restriction. Only the urllib3 package
record differs; its complete record equals the Dependabot input. Candidate
lock SHA-256 is
`4d92aa517c35fe968dc35a2edf4564787cb1384ee2b6f5c9c08dc108b508a39a`.
Diff checking and all five ordinary pre-commit gates pass.

There is no MeloTTS current lock-hash audit/API-smoke/owner manifest to
restamp. Existing fixture evidence is not regenerated or promoted. No
dependency synchronization, package import, model acquisition/load/forward,
Cargo compile/test, credential/provider access or publication occurs.

The original PR's 70 completed checks (69 successful, one intentional
full-history gitleaks skip) cover its original head only. Integrate the latest
accepted main and require fresh complete CI for the corrected published
head before ordinary merge. This update does not complete real-weight CPU,
owner/legal or Apple gates and does not generate a final transfer packet;
the [full plan](next-steps-2026-10-14.md) remains in force.
