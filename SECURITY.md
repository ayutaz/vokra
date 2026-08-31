# Security Policy

**English** | [日本語](SECURITY.ja.md)

## Reporting a vulnerability

Report vulnerabilities only through
[GitHub Private Vulnerability Reporting](https://github.com/ayutaz/vokra/security/advisories/new).
This is the project's sole private reporting channel. Vokra does not publish
an email address or another project-specific contact point.

Do not open a public issue for a suspected vulnerability. The private advisory
should include the affected commit or version, platform and backend, impact,
and a minimal reproduction when it is safe to share. Link to restricted model
weights rather than attaching or redistributing them.

## Response and disclosure

Vokra is maintained on a best-effort basis without an SLA. Reports are
triaged according to reproducibility and impact, with memory corruption,
remote code execution, cross-session disclosure, and release-gate bypasses
receiving priority.

Please keep the report private while a fix or mitigation is prepared. Public
disclosure and reporter credit are coordinated in the advisory; anonymity is
respected when requested.

## Supported versions

The workspace is `0.3.0` development; no Git tag or published release exists
yet. Security fixes are made on the current development line. Older commits,
private builds, and unreleased snapshots do not receive backports.

## In scope

- Memory-safety or validation failures caused by malformed GGUF,
  safetensors, audio, codec, or request data.
- Safe Rust or C API calls that can trigger undefined behavior, invalid
  lifetimes, or cross-session data exposure.
- The mmap loader, raw GPU FFI backends, offline converter, and repository
  release or provenance gates.
- Request parsing, isolation, and resource-control vulnerabilities in
  `vokra-server` and repository-owned integrations.

## Out of scope

- The quality, bias, licensing, or intended behavior of upstream model
  weights.
- Vulnerabilities in third-party drivers, operating systems, hosted services,
  or projects maintained in another repository.
- Unsupported configurations, operator-supplied secrets committed to a fork,
  or scanner output without a reproducible security impact.
- Performance limits or resource consumption inherent to running a model,
  unless untrusted input can bypass an enforced limit or affect another
  session.

## Dependency posture

The root runtime has no third-party Cargo dependencies; CI enforces a
first-party-only `Cargo.lock`. This reduces dependency-chain exposure but does
not replace review of Vokra's own parsers, unsafe boundaries, FFI, and
generated artifacts.
