# Conv-TasNet Libri1Mix validation

This is the dedicated Python 3.12, Linux x86_64 CPU-only project for the
VAST oracle run. It is pinned to `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`
at revision `bb8a876bc157b5cf3c405994accb798c49146016`, with the authenticated
`pytorch_model.bin` identity recorded in `license_gate_manifest.json`.

The gate is deliberately fail-closed. The upstream YAML says CC-BY-SA-4.0,
the checked-in license body says CC-BY-SA-3.0, and WHAM data is marked
CC-BY-NC-4.0 Research-only. The Python closure review is also pending. Until
an external owner approval is added without changing the fixed identities,
the worker exits 2 before creating work, cache, or evidence. Any conversion
and parity execution is VAST-only; publication remains `NO_UPLOAD`.

The reference dumper is an independent Asteroid 0.7.0 oracle and records
stream-hashed artifacts and runtime versions. The Apple verifier accepts only
the exact VAST-produced manifest/artifact set and keeps Metal measurements at
`MEASURED_NOT_GATED`.
