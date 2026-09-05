# SpeechT5 TTS parity closure

The VAST worker uses the pinned `uv.lock` project and the official
`SpeechT5ForTextToSpeech.generate_speech` route from the isolated
`transformers==5.10.4` pin. The prior isolated reference pin was 5.5.0; no
upstream requirement is asserted for that version. Compatibility of the
security-remediated pin is `BLOCKED_UNVERIFIED_API_SMOKE` until an authorized
VAST model smoke run.
The lock and project bytes, all canonical dependency rows, fixed TTS and
HiFi-GAN revisions/artifact hashes, and the historical public GGUF identity
are bound by `license_gate_manifest.json`.

`preflight_gate.py` is standard-library-only and runs with
`uv run --no-project --offline` before scratch creation, synchronization,
source/model download, conversion, or Cargo. It binds the exact Linux lock,
including uv's build-constraint manifest for the isolated NumPy source build.
The compact `dependency_audit_evidence.json` records the full historical VAST
audit digest for the prior isolated pin without committing the 1.3 MB report;
its old input hashes intentionally fail the active-lock gate until a fresh
audit is produced. `patchelf` is GPL build-only and
is not installed in or redistributed with the final environment; its operator
approval remains an explicit gate.

`post_sync_audit.py` runs immediately after `uv sync` and before any source or
model acquisition. It independently checks the synchronized package closure,
absence of build-only dependencies, source-built NumPy native libraries and
ELF NEEDED allowlists, and the exact reviewed torch `libgomp` identity. The
five model/source factual review rows are complete; acquisition and validation
remain fail-closed solely until operator approval supplies the separate
authenticated evidence file.

The source dumper is now pinned to the active 5.10.4 route but refuses
Transformers import and model acquisition while
`BLOCKED_UNVERIFIED_API_SMOKE` remains set. CPU and Apple workers require the named Cargo
test to report exactly one passed, zero failed/ignored, and require each
model-level parity sentinel exactly once. No upload path is present.

## Fresh dependency audit

Before operator preflight is closed, an explicitly authorized VAST job may
run `scripts/publish/vast-ai/audit-speecht5-tts-dependencies.sh`. It requires
`VOKRA_PUBLISH_ON_VAST=1`, Linux x86_64, a clean checkout, and an absent
output directory. The worker performs frozen `uv sync`, then runs the
model-free auditor with `--no-sync` to record the exact installed closure,
publisher license metadata/files, native and bundled payloads, and build-only
facts in full and compact JSON. The dependency-only frozen `uv sync` may use
the locked package indexes. If an installed wheel has no publisher file, the
auditor may make a bounded license-only request for that package's exact
locked PyPI sdist, validating the host, path, redirects, size, and SHA256
before recording archive license bytes/hashes. The `auditor_network_requests`
field covers only these fallback requests, including failed fetch attempts; it
does not count dependency sync traffic. It does not acquire model weights or
any non-license source files,
import Torch/Transformers, run Cargo, upload, or update the manifest/reviews.
In the evidence, `publisher_license_files_missing` means the installed wheel
did not contain a publisher file; `publisher_license_evidence_missing` means
neither that wheel nor the locked-sdist fallback yielded license evidence.
The latter is the fail-closed blocker, so a successful fallback is not reported
as missing overall evidence. Compact evidence retains the network count and
scope without credentials or response-body URLs beyond the locked artifact
facts.
The existing compact evidence and preflight gate remain unchanged until the
fresh evidence is reviewed by the owner.

## API smoke prerequisite

Before changing the reviewed compatibility status, run the dedicated
`scripts/publish/vast-ai/run-speecht5-tts-api-smoke.sh` worker on a disposable
Linux x86_64 VAST instance. It requires authenticated preflight approval
evidence accepted by the existing license gate, a clean checkout, and two
absent absolute output paths. The worker uses the frozen
Python 3.12 project, the exact SpeechT5 checkpoint above, and the official
`SpeechT5ForTextToSpeech.generate_speech` route with the deterministic `Hi.`
input. It emits only hashed API evidence and `NO_UPLOAD`; it does not run
Vokra or upload/publish any artifact. Keep the status
`BLOCKED_UNVERIFIED_API_SMOKE` until that evidence is reviewed.

The approval file is the authenticated evidence consumed by the existing
`preflight_gate.py` against `license_gate_manifest.json`. It must match that
manifest's reviewed scope and operator signer. This worker does not introduce
a parallel approval schema; the current manifest remains blocked while its
existing dependency/model/operator reviews are unresolved. The Python worker
also invokes this same gate in-process, so direct invocation cannot bypass it.
