# MOSS-Audio model-free API smoke

This directory is a VAST-only, model-free compatibility probe for the pinned
OpenMOSS source revision and the 4B/8B metadata snapshots. The project lock is
Python 3.12-only and pins the repository security-floor route
`transformers==5.10.4` and the complete official-reference dependency closure;
it is deliberately separate from the active MOSS project, which remains
blocked on its historical `transformers==5.5.0` contract.

The probe downloads no checkpoint shards. It authenticates the source checkout,
the complete metadata-only snapshot, and the locked package artifacts, then
imports the official configuration, model, and processor classes. It constructs
only the configuration and processor; model instantiation and checkpoint
loading are explicitly not performed. An incompatible API produces
`BLOCKED_INCOMPATIBLE_API` evidence and never becomes a compatibility PASS.

Before any project environment can be installed, the VAST worker runs the
stdlib-only `--closure-only` preflight. That gate requires an owner approval
bound to the exact Vokra HEAD, project/lock bytes, package identities, and
source/4B/8B license-review evidence. The normal VAST result writes a digest
bound evidence file and summary. The metadata/reference evidence is transferred
VAST-to-Apple directly; only small logs are recovered locally. There is no
upload path.

Examples (on the approved VAST host):

```text
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --self-test
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --closure-only \
  --variant all --approval-evidence /path/approval.json --expected-head <HEAD>
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --variant all \
  --approval-evidence /path/approval.json --expected-head <HEAD>
```

The pinned identities in `api_smoke.py` must be regenerated from authenticated
primary-source evidence if upstream metadata changes; no topology or parity
bound is inferred by this smoke.
