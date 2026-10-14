# FireRed checkpoint configuration source readback — 2026-10-14

> **11:01:31 UTC supersession:** PR #236 was normally accepted after its
> exact head `afbaf0f9` completed 69 successful checks, one skip, no pending
> or failure, and all 16 required context/application pairs. Accepted main
> is `99bfa77a7daceee719012a595badf08b370da390`. The configuration finding
> below was privately recorded at `3b1ae62e`; it and the inert-reader
> preparation are a separate follow-up, not new model/approval evidence.

Read-only source investigation against accepted main
`ab653152b6d41a04fb9adacc6e7d89b694fc0188`. This record is prepared separately
from source-observer PR #236 and is not an accepted change to main. No model,
checkpoint, tokenizer or archive was acquired, loaded or executed. No gate,
approval, source/model identity or numerical bound was changed.

## Empty sidecar does not prove absent checkpoint configuration

The fixed official FireRed source is
`834635e4cf277ed8ca92049fc375b17c3dc20748`. Its
[inference loader](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/fireredasr/models/fireredasr.py)
(Git blob `9cb4e3333adccda9c5d8b914fce92e52ca0e1975`) reads
`model.pth.tar`, takes `package["args"]`, and loads
`package["model_state_dict"]` with `strict=True`. It does not read
`config.yaml`. The zero-byte sidecar at model revision
`e57f5960d03cff1071ff7acbb409314d1e70ed3d` proves only that the sidecar is
empty, not that configuration is absent from the checkpoint.

The official
[AED constructor](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/fireredasr/models/fireredasr_aed.py)
(Git blob `4938c2c3c9271b4b65ef38a1fdce464bddef2c02`) obtains these fields
from that Namespace:

- Encoder: `idim`, `n_layers_enc`, `n_head`, `d_model`,
  `residual_dropout`, `dropout_rate`, `kernel_size`, `pe_maxlen`.
- Decoder: `sos_id`, `eos_id`, `pad_id`, `odim`, `n_layers_dec`,
  `n_head`, `d_model`, `residual_dropout`, `pe_maxlen`.

Those source statements establish a required-field contract, not observed
values in the released checkpoint. Existing `EXPECTED_ARGS` checks are not
a new inspection receipt; the constructor also reads dropout and positional
length fields. Record the actual full Namespace separately before claiming
complete configuration authentication. Decode search policy is another
runtime input and is not a substitute for checkpoint configuration.

## Existing restricted loading route, not arbitrary pickle permission

The official loader has a `torch.load` call without an explicit
`weights_only` argument. This is a source observation, not permission to
execute that deserialization path or a claim about a framework-version
default.

At the accepted main, Vokra's
[independent reference loader](https://github.com/ayutaz/vokra/blob/ab653152b6d41a04fb9adacc6e7d89b694fc0188/tools/parity/firered_asr_aed_l_reference.py#L236)
instead requires the unsafe-global list to equal exactly
`["argparse.Namespace"]`, permits only that class in a scoped `safe_globals`
context, and explicitly uses `weights_only=True` on CPU. It validates the
exact `args`/`model_state_dict` envelope, types and expected arguments, then
uses the fixed upstream AED class with a strict state load. The
[checkpoint inspector](https://github.com/ayutaz/vokra/blob/ab653152b6d41a04fb9adacc6e7d89b694fc0188/tools/parity/firered_asr_aed_l_inspect.py#L367)
uses the same restricted global policy and explicit weights-only loading.
These are existing source paths, not successful executions in this readback.

Both routes still materialize tensors. The declared checkpoint is
4,678,597,714 bytes and remains remote-only; the maintainer Mac must not
acquire or inspect it through either route. There is no FireRed-specific
data-pickle-only envelope extraction receipt in this investigation. A future
bounded metadata-only adapter would be a separately reviewed implementation,
not authorization for a new checkpoint or cloud task.

## Disposition and next prerequisite

Keep the existing `BLOCKED_EMPTY_CONFIG` machine label and exact scope hash
unchanged. Interpret it as an unresolved checkpoint-configuration boundary,
not proof that the official inference route needs a nonexistent YAML file.
Do not rename the gate, infer model dimensions, fabricate observed arguments,
or carry approval across a changed scope.

First resolve exact dependency/native/training provenance and owner/operator
decisions. Only an approved remote scope may acquire the fixed checkpoint and
sidecars and produce actual argument, state-dict and sidecar byte receipts.
Independent frontend/encoder/decoder/beam/reference and real-weight CPU
parity must then pass. Existing source-only Rust results are not those
measurements; no row advances, upload remains disabled, and the final clean
HEAD/Apple packet goal remains open.

## Inert structural reader preparation, not an actual checkpoint receipt

The first-party [metadata reader](../../tools/audit/firered_checkpoint_metadata.py)
now prepares a bounded path for already extracted `data.pkl` metadata. It
does not open a checkpoint ZIP, tensor payload, network connection or model
package. CLI use requires Linux and an explicit remote marker before input
read; this is an execution-location guard, not source/model/operator
approval. Real input acquisition/extraction remains a separately admitted
remote scope. The maintainer Mac is used only for tiny synthetic unit tests.

The restricted pickle global set resolves `argparse.Namespace`,
`collections.OrderedDict`, the fixed tensor reducer and supported storage
labels to inert local objects. Descriptor mutation, unknown globals or
opcodes, unsupported metadata, trailing bytes, sparse memo indices, cycles,
missing constructor fields, boolean integers, nonfinite numbers and invalid
storage spans fail closed. There is no replacement numeric forward or
independent-reference claim.

Input is capped at 512 KiB before read. Streaming preflight caps opcodes at
32,768 and memo entries/indices at 8,192. Shared normalization budgets cap
nodes at 8,192 and output at 2 MiB, with per-occurrence string/descriptor
charges before expansion and a final exact serialized-size check before
write. These are new tool security limits, not relaxed model parity bounds.
The real metadata's compatibility and size are still unverified.

Review caught and corrected exponential shared-graph expansion, post-reducer
descriptor mutation, recursively nested credential-bearing keys, unpinned
parent-path races, premature FIFO blocking and an inner-fd cleanup failure.
The corrected reader uses non-following, fd-pinned parent traversal and
exclusive output creation. It never overwrites or automatically deletes a
partial output or another writer's file. Ordinary OrderedDict hook/state
metadata is explicitly handled rather than silently ignored.

Luna and manager independently pass 27 offline/no-project Python 3.12
stdlib tests. Controls prove oversized input is refused before read, sparse
memo/trailing input before unpickler construction, Darwin/marker refusal
before input read, and expansion failures are caused by the intended budget.
One invocation is added to the existing workflow-security job; actionlint
passes and no action version, permission or job is added.

The single stdlib Pickler fixture has 940 **synthetic**, distinct storage
descriptors: 93,568 input bytes, 28,286 opcodes, 6,610 memo writes and maximum
memo index 6,609. Manager measurement renders 245,487 bytes and records
2,166,553 bytes of Python-traced peak allocation. This is not RSS, a real
checkpoint measurement, actual model configuration or a 940-tensor source
inventory. No tensor values or model framework are loaded.

Captured source/model labels remain `DECLARED_NOT_AUTHENTICATED`; the full
checkpoint hash is null, approval is null, publication is `NO_UPLOAD`, and
real-checkpoint/model/parity gates are `NOT_RUN`. The existing gate/scope
hash, machine status, expected model facts and owner decisions are not
changed. There is still no actual FireRed argument receipt, released-weight
CPU/reference comparison or final Apple packet.
