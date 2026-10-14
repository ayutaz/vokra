# PR #230 primary-source link transport — 2026-10-14

The source replay remains bound to `4263f416`; PR #230 initially used
documentation-recording head `8f18a6a0`. No source, model, owner decision or
numerical gate changes in this documentation-only follow-up.

The original documentation-links job `115694375101` failed on two GitHub
HTML blob pages with HTTP 503 (ANTLR setup and VibeVoice guidance). Bounded
credential-free GET requests to those exact URLs returned 200 at 05:24 UTC.
After exact-head/failure revalidation, only that failed job was rerun once.
The new job `115696532722` passed the original two links but failed on three
different GitHub HTML pages with 503: Moshi MLX inference, Demucs README and
MUSDB README. Neither result is relabeled green. The moving error set supports
an external HTML-response problem, not a claim that the source files are absent;
the specific server-side cause is not established.

The fix uses the official raw bodies for the same source paths. ANTLR's tag
link is strengthened to the already authenticated tag commit. The six-row
Kyutai table consistently uses the raw endpoints whose exact bodies were
already measured. No citation is deleted, no checker is excluded, no HTTP
status is accepted as success, and no retry/timeout or workflow is weakened.
New-head required CI must prove that the correction passes.

Fresh official bodies at review time:

| Source | Bytes | SHA-256 | Git blob SHA-1 |
|---|---:|---|---|
| ANTLR setup at `e4c1a74c66bd5290364ea2b36c97cd724b247357` | 527 | `00ce3bb1670e8e4961d9ef2414dfa6ac467b52465382841b432578f91733ec67` | `75d0dc5f55c117e03801da4fe0f8a6552ef3f62a` |
| VibeVoice guidance at `94da20d98b2fa7688e9cbfaf7692ddb4954f7600` | 8095 | `5bdb2d52a0677e32074fa03b25ddd4d758a8856499113e5dd7d42d13acd886fe` | `6ca2deb24edf31fa342bd7c407e66143056f59ed` |
| Demucs README at `e976d93ecc3865e5757426930257e200846a520a` | 18462 | `b2156a62247dae102c6ca3b000022d67f512743bd919523df320780120a329be` | `fe7e77b992b7918eba9491119d313abf5bc3eff1` |
| MUSDB README at the existing `master` alias | 9567 | `83c68379d6e7f0f135c63e06c4e465a303b46f1baddf16a53c8d275b615c8d74` | `880ffe2b42bd3eeebcf10a056eedcc04359c7ded` |

MUSDB `master` remains mutable; this readback is not an immutable source pin
or a new dataset/model license decision. All historical terms, packet
denominators, scope hashes, signer fields and execution/publication holds
remain unchanged. The raw Kyutai bodies retain every size/hash in the
[source reconciliation](kyutai-stt-pcm-contract-reconciliation-2026-10-14.md).

This follow-up does not claim global GitHub recovery or guarantee future
network availability. The normal link checker remains authoritative for the
new HEAD. No VAST/Scaleway instance or local model/Cargo run is needed.

## New-head failure and checker investigation (05:48 UTC)

The raw-link documentation correction was committed as `a362a48a`. Its
documentation-links job `115698658569` still failed: four other GitHub HTML
blob pages returned 503 (the PR template's legal-compliance and NOTICE
links, ANTLR LICENSE at tag `4.9.3`, and the pinned VibeVoice tokenizer line
anchor). The report recorded 1,079 total links, 1,062 successes, four errors
and zero timeouts. This is a failed new-head check, not a successful raw-link
fix or grounds for another unchanged rerun.

At 05:48 UTC, the exact-head check-runs API returned all 86 checks:
79 success, five skipped, one in progress and this one failure. Neither an
older head's green checks nor the source-only VAST replay clears that failure.

The pinned action already supplies `github.token`; a missing token was not
established. In lychee v0.24.2, a failed public GitHub URL is retried through
a repository-existence API check, but a public repository URL with a file
endpoint is rejected rather than verifying that file. Thus that built-in
route cannot verify the failed public blob links. This explains the lack of
a useful file fallback, not the origin of GitHub's HTTP 503 responses.

Primary implementation sources inspected by the manager:

- [Pinned action inputs and default version](https://raw.githubusercontent.com/lycheeverse/lychee-action/e7477775783ea5526144ba13e8db5eec57747ce8/action.yml)
- [Token export and actual exit-code output](https://raw.githubusercontent.com/lycheeverse/lychee-action/e7477775783ea5526144ba13e8db5eec57747ce8/entrypoint.sh)
- [Public GitHub fallback](https://raw.githubusercontent.com/lycheeverse/lychee/lychee-v0.24.2/lychee-lib/src/checker/website.rs)
- [JSON formatter and upstream example](https://raw.githubusercontent.com/lycheeverse/lychee/lychee-v0.24.2/lychee-bin/src/formatters/stats/json.rs)
- [Counters and response-set semantics](https://raw.githubusercontent.com/lycheeverse/lychee/lychee-v0.24.2/lychee-bin/src/formatters/stats/response.rs)
- [Optional response fields](https://raw.githubusercontent.com/lycheeverse/lychee/lychee-v0.24.2/lychee-lib/src/types/response.rs)

A bounded exact-file resolver is under review, not yet accepted or committed
at this readback. Ordinary lychee must still scan the full existing scope;
the proposed mandatory final gate must reject missing/malformed reports,
runtime errors, timeouts and every unhandled link failure. An original 503
is not itself success: any permitted alternative must independently verify
the same official repository, ref, path, content identity and line range.
No original failed job is relabeled, and model/owner/Apple gates are unchanged.

## Reviewed resolver and offline acceptance

The manager reviewed Luna's three-file implementation and independently
reran its 15 offline stdlib tests and actionlint/ShellCheck: all passed.
Earlier incomplete test assertions are not used as acceptance evidence.
The final tests prove that API fault cases actually reach their intended
validation, rather than stopping first at a missing-token error.

The ordinary lychee inputs, exclusions, retries and timeouts are unchanged.
Its action is a JSON/exit-code collector (`fail: false`), followed by an
unconditional mandatory gate. Collector success is not job success. An
absent, empty, malformed or inconsistent report, panic/unknown exit code,
timeout, non-503 failure, unsupported host/URL, or failed API verification
fails the final gate. The new tests also run in workflow-security.

Only an original 503 for a canonical GitHub blob URL may use the alternative
file check. Contents API 200, exact repository/ref/path/name and link
identities, strict base64, decoded size, recomputed Git blob SHA-1, UTF-8 and
LF-based line-anchor bounds must agree. Symlink/submodule identity fields
are rejected. Non-commit refs also require complete
[matching-reference evidence](https://docs.github.com/en/rest/git/refs#list-matching-references)
with exactly one branch/tag match and no ambiguous slash descendant;
pagination is rejected. This is a live-link verification, not an immutable
pin or license approval for mutable branch/tag URLs.

Limits are ten distinct failed URLs, 1 MiB per API response and a shared
60-second deadline. Credentials come only from the job's `GITHUB_TOKEN`
environment and are sent only to constructed `api.github.com` URLs; redirects
are forbidden. Logs retain the original JSON failures and identify successful
exact-file alternatives without token/header/body/error representations.

Final reviewed SHA-256 values:

- checker: `8a3ed2c406e19c984187af8da76211ce7781dc17e769cae06aa7136e94b4889f`
- tests: `eea9bca68aeb49bc4c145fe1958e05ce8eb34d1379ef07e1832ed5209113295b`
- workflow: `6ca948567f31e5a8d5c69d1085970d444930d79551e6cc6f76cb69b614d91904`

A separate credential-free bounded GET readback returned Contents API 200
for all four links from the failed `a362a48a` job, with the expected response
identity fields. It did not execute the authenticated final gate or clear
that job. The final implementation still requires fresh exact-head CI.
Rust source, Cargo inputs and native build configuration are unchanged from
the recorded `4263f416` VAST replay; its green result is retained only for
that unchanged source, not restamped as a new-head model or CI verdict.
