# microWakeWord reference dependency evidence

`audit_closure.py` is a stdlib-only collector for the pinned Linux x86_64
Python 3.12 dependency closure used by the independent LiteRT reference. It
first inventories every `.dist-info` under site-packages and requires the
inventory to exactly match the eight external lock packages (the virtual
project is excluded). It then records installed distribution metadata, the exact `RECORD` identity, bounded
case-insensitive `LICENSE`/`LICENCE`/`COPYING`/`NOTICE`/`COPYRIGHT` candidates,
and native payload hashes with `readelf` `NEEDED` facts when available.
The worker passes both the synchronized venv root and the exact
`sysconfig.get_path("purelib")` site-packages path. RECORD entries may point to
venv-owned files such as `../../../bin/...`; lexical traversal and symlink
checks reject only paths outside that environment root.

The collector does not import LiteRT or NumPy, inspect a model, classify a
license, or grant publication permission. Reports are
`EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED` when collection succeeds and always
set `fixture_generation_permitted=false` and `publication_permitted=false`.
Any missing/unknown package, duplicate row, symlink, path escape, oversize
file, or missing license candidate is recorded as a fail-closed collection
failure.

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
