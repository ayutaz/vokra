# Release workflow first dry-run evidence — 2026-08-22

## Result

The first real `release.yml` `workflow_dispatch` exercise is complete. The
final run was green with `dry_run=true` on branch
`chore/release-dry-run-2026-08-22` at
`36b48f3a9943f8bb3c4b07e7c0d7a991fc2dd3a6`:

- release dry-run: [run 32564192022](https://github.com/ayutaz/vokra/actions/runs/32564192022)
- same-SHA CI preflight: [run 32563835960](https://github.com/ayutaz/vokra/actions/runs/32563835960)
- same-SHA Godot preflight: [run 32563836277](https://github.com/ayutaz/vokra/actions/runs/32563836277)

All 17 release jobs completed successfully. The final run started at
2026-08-22 09:09:43 UTC and completed at 09:17:33 UTC.

## Defects found by the first dispatch

The initial default-branch dry-run, [run 32562714991](https://github.com/ayutaz/vokra/actions/runs/32562714991),
failed and exposed two defects that the static oracles had not exercised:

1. `unity-package-release` invoked `scripts/build-android.sh` without first
   setting up the Android SDK/NDK.
2. The four-wheel manifest job asked `setup-uv` to resolve the moving latest
   version and timed out while fetching the remote version manifest.

The branch fixes add the pinned `setup-android` action to the Unity release
job, preserve the action-exported `ANDROID_NDK_HOME`, and pin the manifest
job's uv binary to `0.12.5` (the repository VAST provisioning pin). A manual
`workflow_dispatch` entry was also added to `ci.yml` so a release dry-run can
produce exact-ref/SHA preflight artifacts without creating a PR or updating
`main`.

An intermediate run, [run 32563504731](https://github.com/ayutaz/vokra/actions/runs/32563504731),
confirmed the wheel fix but exposed that an expression-time empty
`ANDROID_NDK_ROOT` value was overriding the NDK environment exported at run
time. Removing that override produced the final green run.

## What the green run exercised

- release-note extraction and non-tag semver announced-skip
- crates.io topological package checks, leaf `cargo publish --dry-run`, and
  zero-dependency lockfile tripwire
- iOS XCFramework build, ABI verification, packaging, checksum, and SPDX SBOM
- Unity Linux, Android arm64, WebGL, macOS, Windows, and iOS slices; NVIDIA
  non-bundle scan; six-slice assertion; deterministic UPM pack; checksum; SBOM
- four native Python wheels, exact-target manifest, clean-install tests,
  binding tests, `twine check`, and Python release artifact assembly
- Godot five-target artifact reuse, package compliance, deterministic pack,
  checksum, and SPDX SBOM
- npm WebAssembly assembly, lockfile tripwire, deterministic pack, checksum,
  SPDX SBOM, and `npm publish --dry-run`

The desktop library/CLI and standalone Android AAR jobs remained deliberately
T32-gated by `DESKTOP_AAR_ENABLED`; their effectful build/upload steps were
announced-skips, as documented in `docs/handoff/x-07.md` section 4.

## Final workflow artifacts

GitHub reported nine unexpired artifacts, all attached to run `32564192022`
and the exact final SHA:

| artifact | GitHub artifact digest |
|----------|------------------------|
| `Vokra-unity-package-release` | `sha256:867876b2bcd7fb2ffc500c578313f8397ab821c1b90ef99661878f03e16a558c` |
| `Vokra-xcframework-release` | `sha256:b73a4c861c1a40b2b3736dc649bd4f65c72f097ab654532493ba63ad0b979c07` |
| `vokra-python-release` | `sha256:6b68ebd335db23d54e3dc74de5c1eff17a2cd5f26946583ab36e30645ac110b3` |
| `vokra-python-wheels` | `sha256:8320dcde599c31dc95c2b701cf10cffd6371e046961da79616e9fa51e57ae9e4` |
| `vokra-python-wheel-linux-x86_64` | `sha256:25916d17084357ff180eecd91b408ff1467a81021478d43d74a1c144f156b124` |
| `vokra-python-wheel-macos-arm64` | `sha256:e4c4cbd41ca5f6681d66f445cc348007eb270d6f05878745b1e406d74f405700` |
| `vokra-python-wheel-macos-x86_64` | `sha256:8532941f45e75319cdffbd485fbd8e3b4890e16f3d1a03d70e54323337f8bacf` |
| `vokra-python-wheel-windows-x86_64` | `sha256:540f0bf46c423e81ddc4efe7f54cea58ad08fd3dcdc9367ed1a0533421411e58` |
| `Vokra-godot-assetlib-release` | `sha256:b97d275373a30d1a67704dc0cb65e7aba591fa42b2ab16e5361e7b673b72121c` |

## No-publish proof and remaining boundary

After completion, both GitHub API queries still returned empty arrays:

- `GET /repos/ayutaz/vokra/releases` -> `[]`
- `GET /repos/ayutaz/vokra/tags` -> `[]`

The run log also records all GitHub Release uploads, attestations, the
`Package.swift` patch, PyPI publish, crates.io publish, npm real publish, and
OpenUPM publish as skipped by the dry-run/non-tag gates. Only workflow
artifacts were uploaded.

This closes the missing first `release.yml` dry-run evidence. It does not
authorize or prove a real tag-triggered publication, registry credentials,
T32-gated desktop/AAR outputs, or the separate first
`release-cadence.yml` dispatch.

## Reproduction commands

```sh
gh workflow run ci.yml --ref chore/release-dry-run-2026-08-22
gh workflow run godot-crossbuild.yml --ref chore/release-dry-run-2026-08-22
gh workflow run release.yml --ref chore/release-dry-run-2026-08-22 --field dry_run=true
```
