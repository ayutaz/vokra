# FireRed checkpoint configuration source readback — 2026-10-14

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
