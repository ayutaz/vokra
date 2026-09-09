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
The pending manifest records that evidence source HEAD separately.  The
pending gate requires both reports to agree with that source HEAD, while the
runner still verifies the current checkout HEAD independently; committing the
tracked evidence therefore does not create a self-referential HEAD contract.
The eventual complete manifest carries the same `evidence_source_head` field;
only the separate owner approval record is bound to the runner's current
checkout HEAD.
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

The complete-manifest gate additionally requires the output of the
model-free metadata audit (`--metadata-evidence`). It binds the manifest's
backbone, one explicitly selected adapter, and matching language vocabulary to
the audit's exact LFS payload SHA-256 and byte size. A regular Git blob without
an authenticated payload digest is rejected; the gate never guesses a digest
or a tensor shape. This identity preflight is independent of and prior to any
checkpoint import, runtime binder, or parity claim.

## No-weight server-metadata audit

`hf_metadata_audit.py` is the smaller model-free route for authenticating the
exact eight-file upstream snapshot closure before the owner-review closure is
complete: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`,
`vocab.json`, `special_tokens_map.json`, the full `model.safetensors` backbone,
exactly `adapter.<language>.safetensors`, and exactly
`vocabs/<language>.txt`. It uses one bounded standard-library HTTPS request to
the fixed Hugging Face model-info endpoint with `blobs=true`; the raw response
is duplicate-key checked and its final URL, content type, byte count, and
SHA-256 are bound in the report. For Git-LFS files it also checks the server
LFS-pointer Git blob identity against the reported payload digest. It never
resolves or downloads a file, imports Transformers, constructs a model, or
executes inference, and sends no `Authorization` header or ambient token.

The language is intentionally mandatory; no English/default adapter is
selected:

```bash
scripts/publish/vast-ai/run-mms-1b-all-validation.sh \
  --metadata-only --language <official-code> \
  --expected-head <clean-vokra-40-hex-head> \
  --output /absolute/path/mms-metadata.json
```

This VAST-only route verifies the Vokra checkout is clean and exactly at the
supplied expected head before any metadata network request. The runner invokes
the stdlib-only auditor with `uv run --no-cache --no-project --offline
--python 3.12`; it performs no project synchronization at all. It exits 2
after writing a no-clobber report whose
`vokra_checkout` object contains the exact `expected_head`, `actual_head`, and
`clean=true`, together with `status=BLOCKED_PENDING_OWNER_REVIEW` and
`publication=NO_UPLOAD`. The report is factual server metadata only; it is not
an owner approval, does not make a non-commercial weight publishable, and does
not authorize runtime or CPU/Metal parity. `--self-test` exercises a synthetic
pass, missing/drifted/dirty-head rejection, eight-file sidecar omission and
tamper rejection, raw duplicate-key rejection, bounded-response rejection,
and output no-clobber behavior without network access or model artifacts.

The checkout binding has this exact shape (the two values are the supplied
and observed 40-character commit, not a branch name):

```json
"vokra_checkout": {
  "expected_head": "<40-lowercase-hex>",
  "actual_head": "<same-40-lowercase-hex>",
  "clean": true
}
```
