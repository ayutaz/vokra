# MMS-1B-All dedicated staging closure

This directory is intentionally not a copy of the broad `tools/parity`
project.  It contains a dedicated Python 3.12/Linux x86_64 CPU-only
`pyproject.toml` and `uv.lock`, resolved on VAST with `uv add --no-sync` and
`uv lock --refresh`.  The lock is restricted to the PyPI and official
PyTorch CPU indexes; CUDA, NVIDIA, and Triton dependencies are forbidden.

`dependency_audit_evidence.json` is the model-free VAST inventory for the
exact lock.  It records every lock artifact URL/hash/size tuple, publisher
license metadata, bundled license-file hashes, native payload hashes, and
the one PyTorch CPU wheel whose upstream lock record omitted `size` (the
archive bytes/hash are recovered separately).  This is factual evidence,
not a legal approval: all package/native rows remain
`PENDING_OWNER_APPROVAL`, so the closure gate stays fail-closed.  The audit
performed no model checkpoint download, model instantiation, execution, or
upload.
The report also binds a normalized `name==version` installed-distribution
multiset for every non-virtual lock row; version mismatches, missing,
unexpected, and duplicate distributions make the audit fail closed.

`api_model_free_evidence.json` records the same VAST environment's official
Transformers API inspection: `AutoProcessor.from_pretrained(**kwargs)`,
`Wav2Vec2ForCTC.from_pretrained(**kwargs)`, and the explicit
`Wav2Vec2ForCTC.load_adapter(target_lang=language)` surface.  The class was
imported for signature/source inspection, but no model was instantiated, no
weights were loaded, and no forward pass was executed.  It is API evidence
only; it does not authorize checkpoint loading or parity.

`api_model_free_inspector.py` is the reproducible generator.  It validates the
exact project/lock, requires a clean checkout at the supplied 40-character
HEAD, binds its own SHA-256, and writes evidence without replacement.
`dependency_audit.py` applies the same clean-HEAD and self-hash binding to its
VAST package inventory.  Existing evidence in this tree is provisional until
both generators are rerun after the first clean implementation commit.
Run both generators with output paths outside the checkout (or in a separate
clean worktree): the generated evidence itself must not make the checkout
dirty before the second generator runs.  Only after both outputs are
independently reviewed should they be copied into this directory and bound by
the pending manifest.
On VAST, pass `--repo-root` pointing at the clean Vokra checkout and the exact
transferred `--expected-head`; keep `--output` outside that checkout until both
commands succeed.  The dependency audit additionally requires
`VOKRA_PUBLISH_ON_VAST=1` and the frozen environment/archive.

The eventual manifest must cover the complete backbone (`model.safetensors`),
exactly one explicit `adapter.<language>.safetensors`, and its matching
`vocabs/<language>.txt`; the public ~8.9 MB adapter must never be represented as
the 1B backbone.  Package license/native-bundled review rows and independent
owner approval are mandatory.  Publication remains `NO_UPLOAD`, and all
reference/runtime statuses remain `BLOCKED_PENDING_AUTHENTICATED_MANIFEST`
until separate real evidence is reviewed. The VAST and Apple entry points
also require the exact clean Vokra HEAD to be supplied and bound into the
external approval evidence; they do not acquire or execute a checkpoint when
the closure is absent.
