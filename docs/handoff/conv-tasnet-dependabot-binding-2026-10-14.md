# Conv-TasNet Dependabot lock binding — 2026-10-14

The review starts at PR #232 head
`a42598158e76168d813a6727447fd41fb889308b`, based on accepted source main
`f8804b7429691bd90520aec801869262ba53573b`. The bot updates multidict
6.7.1 to 6.9.1 and normalizes redundant Linux dependency markers. Both
top-level lock marker arrays and the unchanged project still restrict the
environment to Linux x86_64 / Python 3.12. No other platform is approved.

Manager's bounded [official PyPI JSON](https://pypi.org/pypi/multidict/6.9.1/json)
readback matches the sdist and all three locked wheels' URLs, SHA-256 values
and sizes. None is yanked. This is metadata verification, not acquisition,
native-code execution or a package/license review. The closure remains blocked.

The bot also omitted the existing Torch wheel size required by the strict
artifact schema. A HEAD-only request to its unchanged official wheel URL
returned HTTP 200 and Content-Length 184,378,187. Restore that field instead
of weakening the artifact-size gate; no wheel body was downloaded.

The corrected lock SHA-256 is
`d74403e35eb490f4dd25c4a1cd41e0fe6327aeeb752d887006e43bb2ebec104f`;
canonical package-row SHA-256 is
`6451d90b72c2093cf720ec789c649ce80b09dd3336745a1c58973ed6feb78612`.
Bind the checker and manifest to these new inputs, not the former scope.
Keep package reviews empty, license contradictions unresolved, signer/digest
null, and publication `NO_UPLOAD`. Torch 2.9.1 and the unresolved model/data
terms are not repaired or approved by this dependency update.

Luna and manager independently pass the actual stdlib-only gate self-test.
Current inputs reach the unresolved package-review stop after binding/schema
checks; stale lock, old manifest, old approval evidence, both widened marker
arrays and malformed artifact metadata are rejected for the intended reasons.
Whitespace checks pass. Rust/Cargo inputs are unchanged from the recorded
green VAST `4263f416` source replay; this is not a fresh Rust replay at this
Python/lock correction. Fresh exact-head PR CI is still required before merge.
No local model processing, dependency installation, cloud allocation or upload
is performed. No model row or operator approval is promoted.
