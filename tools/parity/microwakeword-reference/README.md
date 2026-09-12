# microWakeWord reference dependency evidence

`audit_closure.py` is a stdlib-only collector for the pinned Linux x86_64
Python 3.12 dependency closure used by the independent LiteRT reference. It
first inventories every `.dist-info` under site-packages and requires the
inventory to exactly match the seven external lock packages (the virtual
project is excluded). It then records installed distribution metadata, the
complete raw `RECORD` evidence, bounded case-insensitive
`LICENSE`/`LICENCE`/`COPYING`/`NOTICE`/`COPYRIGHT` candidates, and native
payload hashes with `readelf` `NEEDED` facts when available. Every declared and
actual file hash/size is checked. uv's two standardized installer rows
(`<dist-info>/INSTALLER` and `REQUESTED`) are retained verbatim under
`installer_generated_rows` and checked against the current uv contract
(`INSTALLER` is the two-byte `uv` marker and `REQUESTED` is empty). Only those
two rows, plus the self-referential actual digest/size of `RECORD`, are omitted
from `normalized_entries_*`; all other rows remain in the normalized identity.
Unknown extra rows therefore remain fail-closed rather than being hidden as
installer metadata. The four known uv console scripts are recorded separately
with their complete wrapper bytes and a portable identity: only the absolute
venv Python path in the first shebang line is normalized. The wrapper body,
target/import contract, and shebang's `bin/python*` environment-relative target
are verified against the reviewed per-entrypoint body size/SHA, import target,
and `sys.exit(main())` contract; unknown or additional scripts remain part of
the normalized identity and fail closed. The absolute shebang must remain an
environment `.venv/bin/python*` target, while its root may vary between VAST
work directories.
The worker passes both the synchronized venv root and the exact
`sysconfig.get_path("purelib")` site-packages path. RECORD entries may point to
venv-owned files such as `../../../bin/...`; lexical traversal and symlink
checks reject only paths outside that environment root.

The collector does not import LiteRT or NumPy, inspect a model, classify a
license, or grant publication permission. Reports are
`EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED` when collection succeeds and always
set `fixture_generation_permitted=false` and `publication_permitted=false`.
The downstream Inspector may promote only the exact reviewed packet to an
effective `PASS`; the raw collector flags remain unchanged in the manifest.
Any missing/unknown package, duplicate row, symlink, path escape, oversize
file, or absence of both bounded METADATA license declarations and wheel-local
license candidates is recorded as a fail-closed collection failure. A missing
wheel-local candidate alone is a collected fact, not a collector failure.

Run the real collection only on a clean Linux x86_64 VAST checkout after the
worker's frozen sync:

```text
VOKRA_PUBLISH_ON_VAST=1 \
  scripts/publish/vast-ai/run-microwakeword-reference-audit.sh \
  --work-dir /workspace/vokra-microwakeword-reference-audit \
  --evidence-dir /workspace/vokra-microwakeword-reference-evidence
```

The paths must be newly absent, canonical, and outside the checkout. The
worker performs no model acquisition, inference, Cargo, Git push, or upload;
the post-sync collector is launched with the venv interpreter's isolated `-I` flag and
receives explicit `--environment-root` and `--site-packages` arguments.
Use `--self-test` for the offline fake `dist-info` test; it performs no sync or
network operation.
