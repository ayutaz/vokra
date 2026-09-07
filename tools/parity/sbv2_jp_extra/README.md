# SBV2 JP-Extra source-contract boundary

This directory is a plan and landing point for the production Japanese G2P
contract. It deliberately contains no copied Style-Bert-VITS2 implementation,
phoneme table, tone algorithm, or production runtime wiring.

## Authenticated inputs

The contract generator must execute the official reference in an isolated VAST
worker, using these immutable identities:

- HF model revision: `a731761009f3c96d104487be6ad332bf1bb5a3a5`
- Official Style-Bert-VITS2 tag: `2.0`
- Official source commit: `ef93f388fc1ddf0dc0f598126c1964923f1df94f`
- `text/symbols.py` blob: `846de64584e9ba4b8d96aab36d4efbcefb1a11e7`
- `text/japanese.py` blob: `5c055875626c16bd7d3489d02b4952ec90a3bbf6`
- `text/japanese_mora_list.py` blob: `b43e54d8d8297cf1eac0e3e3f0eef6b4f1c24fa3`
- `common/log.py` blob: `51dca5f3f39047ed9fb58f59d765ca0a332bc49f`
- `common/stdout_wrapper.py` blob: `23c6e76462753190d77a1b58dfe022012c90a028`
- `text/__init__.py` blob: `495e57b50d87a4ca3e8fe8dbaf003b4888581927`
- License blob: `0ad25db4bd1d86c452db3f9602ccdbe172438f52`

The worker must verify every identity before importing the reference. A
different revision or missing blob is a hard failure, not a best-effort
substitution.

## Required generated artifacts

The VAST worker should emit a hash-bound contract sidecar containing, as data:

1. the exact sorted symbol table and row count;
2. language-id mapping and per-language tone offsets;
3. source-authenticated vocabulary/tone dimensions;
4. the reference source/checkpoint identities above; and
5. deterministic reference outputs for a small Japanese sentence corpus,
   including phones, tones, word boundaries, and normalized input text.

The sidecar and fixtures must be reviewed as independent reference output,
then committed with SHA-256 manifests. The converter may consume the sidecar
only after validating its source hashes, model identity, source dimensions, and
internal lengths; malformed or mismatched data must fail closed. It must embed
the validated contract data in GGUF rather than recreate it from constants.
The runtime must read and validate that GGUF data and must not hardcode a
replacement mapping.

`validate_contract.py` is the model-free schema gate for that sidecar. It is
stdlib-only and has two safe modes:

```sh
uv run --no-project --offline --python 3.12 \
  tools/parity/sbv2_jp_extra/validate_contract.py --self-test
uv run --no-project --offline --python 3.12 \
  tools/parity/sbv2_jp_extra/validate_contract.py --contract <sidecar.json>
```

The VAST-only producer is `generate_contract.py`, driven by
`scripts/publish/vast-ai/run-sbv2-jp-extra-g2p-contract.sh`. The worker clones
the official repository at the fixed commit, checks the commit and the
authenticated Japanese/symbol/sequence/license Git blob identities before
loading only the upstream Japanese module and shared sequence mapper, then
invokes the official `text_normalize` + `g2p(..., use_jp_extra=True)` path for
the fixed corpus below. It never imports the package initializer as a package
(which could eagerly load English), and never contains a Japanese G2P mirror or
fallback. The producer derives `n_vocab=len(symbols)` and `n_tones=num_tones`
from authenticated `text/symbols.py`, and refuses any source-table drift from
the fixed `178`/`12` boundary. This is deliberately a source-table proof, not
a model/checkpoint claim; model compatibility remains the responsibility of
the existing strict GGUF/checkpoint binder and VAST parity worker.

Example VAST invocation (the source checkout is created inside the disposable
worker directory):

```sh
scripts/publish/vast-ai/run-sbv2-jp-extra-g2p-contract.sh \
  --expected-head <clean-vokra-head> \
  --work-dir /vast/scratch/sbv2-jp-extra-contract-<run-id> \
  --output /vast/evidence/sbv2-jp-extra-g2p-contract.json
```

The output JSON and its `.sha256` sidecar are created with no-clobber atomic
writes. The output/work paths must be absolute, canonical, absent before the
run, and disjoint from the clean Vokra checkout. Self-tests use only fake
upstream modules and fixtures; they do not sync dependencies, download source
or models, import the real upstream package, or run Cargo.

The SBV2 reference lock retains only the Japanese frontend dependency
`pyopenjtalk==0.4.1` and `loguru==0.7.3`; the English frontend and its
`g2p-en`/`distance` GPL closure are deliberately absent. The upstream Japanese
module's `num2words` import is satisfied only by an in-process sentinel that
raises on numeric normalization; `num2words` itself is not installed or
executed. Therefore numeric-text G2P is unsupported and blocked. This worker
does not auto-approve any frontend/helper license: the owner/license audit
must explicitly review pyopenjtalk, loguru, and the authenticated upstream
AGPL execution before treating a generated contract as releasable. Publication
remains `NO_UPLOAD`.

The Rust production route is intentionally not present until this sidecar has
an independently reviewed complete hash. The existing generic
`from_piper_g2p` route is not an authentication boundary for JP-Extra.

The fixed upstream `text/symbols.py` contract authenticates the exact schema
values required by `validate_contract.py`: `language_id_map` is
`{"ZH": 0, "JP": 1, "EN": 2}`, tone counts are `ZH=6`, `JP=2`, `EN=4`,
the tone starts are `ZH=0`, `JP=6`, `EN=8`, and the checkpoint boundary is
`n_vocab=178`, `n_tones=12`. The JP-only worker uses the official shared
mapper from `text/__init__.py` to convert raw JP tones into the global `6..7`
band; no English frontend is loaded.

## Primary-source identity checks

These read only immutable commit/tree metadata; they do not fetch or execute
AGPL source. Run them in the VAST worker before generating the sidecar:

```sh
gh api repos/litagin02/Style-Bert-VITS2/commits/ef93f388fc1ddf0dc0f598126c1964923f1df94f \
  --jq .sha
gh api 'repos/litagin02/Style-Bert-VITS2/git/trees/ef93f388fc1ddf0dc0f598126c1964923f1df94f?recursive=1' \
  --jq '.tree[] | select(.path == "text/symbols.py" or .path == "text/japanese.py" or .path == "text/japanese_mora_list.py" or .path == "common/log.py" or .path == "common/stdout_wrapper.py" or .path == "text/__init__.py" or .path == "LICENSE") | [.path,.sha] | @tsv'
curl --fail --silent --show-error \
  https://huggingface.co/api/models/litagin/Style-Bert-VITS2-2.0-base-JP-Extra/revision/a731761009f3c96d104487be6ad332bf1bb5a3a5 \
  | jq -e --arg rev a731761009f3c96d104487be6ad332bf1bb5a3a5 '.sha == $rev'
```

The corresponding immutable primary-source URLs are the [HF revision](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra/commit/a731761009f3c96d104487be6ad332bf1bb5a3a5), the [official source commit](https://github.com/litagin02/Style-Bert-VITS2/commit/ef93f388fc1ddf0dc0f598126c1964923f1df94f), and the authenticated `symbols.py`, `japanese.py`, `japanese_mora_list.py`, `common/log.py`, `common/stdout_wrapper.py`, `text/__init__.py`, and `LICENSE` blobs at that commit.

The fixed `text/__init__.py` path applies the language start offset to raw
G2P tones. Therefore JP raw tones `0/1` become global SBV2 tone ids `6/7`
(`tone_offsets["JP"] = 6`); a JP fixture containing ZH-band `0..5` or EN-band
`8..11` is invalid and rejected by the validator.

## License and execution boundary

The official implementation is AGPL-3.0. It may be executed only as an
independent reference in the VAST parity worker; no AGPL source may be copied
into Apache-licensed runtime, converter, or integration code. No model weights
are acquired or executed locally. Until the VAST-generated contract and
committed fixtures exist, `sbv2-v2-jp-extra-base` Japanese production G2P is
intentionally unavailable and callers must receive an explicit failure.
