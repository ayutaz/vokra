# Vokra 0.3.0 release-preparation record — 2026-09-17

This record separates the source release from the ongoing public-model and
Apple-hardware campaign. Version `0.3.0` is a pre-1.0 release with known gaps;
it is not a declaration that the public model catalog, the six-platform
matrix, the C ABI or v1.0 GA is complete.

## Scope and publication channel

The release baseline is GitHub Releases only. The repository variables
`CRATES_IO_PUBLISH_ENABLED`, `NPM_PUBLISH_ENABLED`,
`OPENUPM_PUBLISH_ENABLED`, `PYPI_PUBLISH_ENABLED` and
`DESKTOP_AAR_ENABLED` remain explicitly `false`, and the repository has no
Actions secrets. crates.io, npm, PyPI and OpenUPM did not contain a Vokra
package at review time. Enabling any external registry requires a separate
owner decision, namespace and credential provisioning, and post-publication
verification; the `v0.3.0` tag must not imply that authorization.

The starting public `main` was
`2b08b7f738575d46c50dd9e2dbfb8d137b27beae` (PR #109). The live read-only
model-hub audit reported 194 public repositories, 193 GGUF-bearing
repositories and 198 GGUF files: Mac CPU `full=136`, `partial=43`,
`no-runtime-binder=14`, `not-artifact=1`; Metal `full=136`,
`blocked-by-cpu=57`, `not-artifact=1`. There were therefore 58 unresolved
public rows.

PR #109's exact clean VAST candidate
`772ba5a30d3b793bc79ab7281acdb0823e751244` completed strict official-weight
reload, converted the four public Qwen3-TTS variants plus their shared 12 Hz
decoder, passed independent CPU parity 4/4, and closed 38/38 recovered packet
checksums. It performed no upload and did not run a model on the maintainer
Mac. Apple CPU/reference, Metal/reference and Metal/CPU no-fallback remain
pending, so that evidence does not reduce the 58-row unresolved total.

## Security triage

The 2026-09-17 GitHub snapshot contained 275 open Dependabot alerts:
6 critical, 53 high, 101 medium and 115 low.

| Scope | Count | Release classification |
|---|---:|---|
| `tools/parity/**` isolated Python locks | 274 | Not shipped or imported by the zero-dependency runtime. These packages execute only inside explicit offline reference/parity environments. 243 alerts have a published patched version and must be updated per locked oracle so numerical provenance is revalidated; 31 have no published patched release (27 Torch, 3 Accelerate, 1 NLTK). They remain visible rather than being bulk-dismissed. |
| `integrations/vokra-misaki-g2p/uv.lock` | 1 | High, transitive NLTK `CVE-2026-81726`; no patched release exists. This is an opt-in out-of-workspace developer bridge, not a distributed runtime dependency. Vokra invokes `misaki.*.G2P` and does not call the affected NLTK model-artifact save/load APIs. Keep the alert open for the upstream fix. |

The root runtime zero-dependency gate remains green. The three isolated Rust
lockfiles using vulnerable `rustls 0.23.41` were updated to `0.23.45`;
`rustls-webpki` moved to `0.103.15`, the yanked `spin 0.9.8` moved to `0.9.9`,
and the server-bench lock incorporates `ureq 3.4.1`.

Code-scanning alerts #59–#64 were reviewed individually. Each dereference is
the expected unsafe Godot GDExtension boundary: the matching create callback
allocates a `Box<SessionInstance>` or `Box<StreamInstance>`, installs that
exact opaque pointer with Godot, and the paired free callback runs after method
callbacks. CodeQL cannot model that external ABI precondition, so all six were
dismissed as `won't fix` with the same provenance rationale. The distinct risk
that callbacks for one Godot object might overlap across threads is retained in
[issue #110](https://github.com/ayutaz/vokra/issues/110); the dismissal does
not classify that concurrency question as safe.

The five remaining code-scanning alerts are OpenSSF Scorecard process signals
(`CIIBestPracticesID`, `MaintainedID`, `CodeReviewID`, `SASTID` and
`VulnerabilitiesID`) with no source location. They remain open as repository
governance work rather than being represented as code vulnerabilities. Open
secret-scanning alerts were zero.

## Release gates

The release candidate must still satisfy all of the following at one exact
clean head before the tag is created:

1. the 19-crate ordered publish graph and the `0.3.0` version contract;
2. all local release oracles and static documentation/workflow gates;
3. fresh pull-request CI, including the Unity package and server-compat jobs;
4. `release.yml` with `dry_run=true` and `enforce_publish_config=true`;
5. inspection of the XCFramework, Unity, Godot, Python wheel, npm, SBOM and
   checksum artifacts;
6. a non-empty `[0.3.0]` root changelog section and Unity release link;
7. the final clean commit, followed by `v0.3.0` only if every gate above is
   green.

Scaleway Apple execution is not a gate for this limited pre-1.0 source release
because the release makes no Apple-completion claim. It remains a separate
required gate before the affected Qwen3-TTS public rows can be promoted or
their artifacts replaced.
