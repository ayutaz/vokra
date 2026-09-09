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
- `pyopenjtalk/__init__.py` blob:
  `656c5089f529150828b5b6fe512b0ca942d9a3a8`
- `bert/deberta-v2-large-japanese-char-wwm/config.json` blob:
  `9fb6b0ac2ec49b6556e58b5ed9492eb33166714d`
- `bert/deberta-v2-large-japanese-char-wwm/special_tokens_map.json` blob:
  `a8b3208c2884c4efb86e49300fdd3dc877220cdf`
- `bert/deberta-v2-large-japanese-char-wwm/tokenizer_config.json` blob:
  `8ab2175580e45760875557201e5543019ca3039b`
- `bert/deberta-v2-large-japanese-char-wwm/vocab.txt` blob:
  `ef3652a1877f4c898e6fcb3e605c432c7bcc56b1`

The worker must verify every identity before importing the reference. A
different revision or missing blob is a hard failure, not a best-effort
substitution.

The Japanese frontend's only native G2P dependency is the sdist-only
`pyopenjtalk==0.4.1`. Its PyPI sdist is pinned in `tools/parity/sbv2/uv.lock`
to SHA-256
`d5ada46f7fc2b52c1c79c273eb9668ff6ad7ab276a8db9d8be119ef93440f0dc`.
The VAST wrapper clones and audits the corresponding upstream commit
`0f0fc44e782a8134cd9a51d80b57b48a7c95bb80` before any Python import. The
authenticated license chain is:

- fixed build declaration: `pyproject.toml` blob
  `9de1588afb8603b1ca9f13c3faca1f658057ba33`; its source-declared
  `setuptools>=64`, `setuptools_scm>=8`, `cython>=0.29.16`, `cmake`, and
  Python-3.12-applicable `numpy>=1.25.0` requirements are checked against the
  project-selected build constraints;

- pyopenjtalk MIT: `LICENSE.md` blob
  `d66bbcca2d9f4d1f9244ea80ec5acda93dbb469b`;
- bundled `mei_normal.htsvoice` CC-BY-3.0: license blob
  `753611721aea5b6ab7f713229c04cdbf8e63dff5`;
- modified Open JTalk BSD: submodule commit
  `462fc38e7520aa89e4d32b2611749208528c901e`, with `src/COPYING` blob
  `495268369d51f7794083769e3305ef108593ab94`, dictionary `COPYING` blobs
  `05d9789fde8883f09b0b9a814a53e6000346e964` and
  `8c50c6c47472d3b190177ce754c5227091040856`;
- modified HTS Engine BSD: submodule commit
  `214e26dfb7f728ff9db39c14a59db709abcc121d`, with `src/COPYING` blob
  `55081f59b6f2e3ec7be3e32e72cca9ebea099671`;
- `loguru==0.7.3` MIT, pinned directly in the SBV2 reference project.

`audit_pyopenjtalk.py` has two fail-closed phases. The static phase checks the
sdist lock identity, the authenticated source/tag, two native submodules, all
license blobs, and the upstream build-system declaration before project
dependency resolution/build. If static checks fail, the worker stops before
pyopenjtalk can be built or imported. The post-install phase checks installed
package metadata, `RECORD`, bundled voice files, native binary inventory and
license payloads before `generate_contract.py`; if it fails, the generator is
not reached. Both phases explicitly reject GPL/LGPL or unexpected payloads.
The Open JTalk dictionary is fixed runtime G2P data, not model weight: the
worker authenticates its archive URL, size, SHA-256, exact tar members, and
license, then authenticates the extracted payload again before and after the
generator. Automatic dictionary download is disabled by supplying the verified
`OPEN_JTALK_DICT_DIR` payload to the generator.
The build-only constraints in the SBV2
project pin `setuptools==80.9.0`, `setuptools-scm==9.2.0`, `cython==3.1.4`,
`cmake==4.1.0`, and `numpy==2.5.2` via uv's supported
`build-constraint-dependencies` setting. These constraints do not add runtime
dependencies.

The lock records the application sdist/wheel hashes for pyopenjtalk and
loguru, but uv's build-only constraint packages are not project packages and
therefore have no archive hashes in this lock. The audit intentionally does
not claim those archives are hash-authenticated; the VAST run must retain its
download/archive evidence and native build payload inspection as a residual
release blocker.

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
authenticated Japanese/symbol/sequence/license/tokenizer Git blob identities before
loading only the upstream Japanese module and shared sequence mapper, then
invokes the official `text_normalize` + `g2p(..., use_jp_extra=True)` path for
the fixed corpus below. It never imports the package initializer as a package
(which could eagerly load English), and never contains a Japanese G2P mirror or
fallback. The producer derives `n_vocab=len(symbols)` and `n_tones=num_tones`
from authenticated `text/symbols.py`, and refuses any source-table drift from
the fixed `112`/`12` boundary. This is deliberately a source-table proof, not
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

The Rust runtime now exposes the narrow production seam
`SbV2JapaneseG2pProvider` + `SbV2JapaneseG2pContract`. The caller must supply
an isolated, first-party native provider and the exact symbol vector from an
independently reviewed sidecar; the runtime validates the source identities
and the `112`-symbol digest, then performs only structural checks on provider
output (non-empty normalized text, tone range/length, and `word2ph` coverage)
before mapping phones to SBV2 ids. Provider algorithm semantics and parity
remain unproven. It does not embed or recreate the upstream table, execute
Python/eSpeak/pyopenjtalk, or accept the generic
`from_piper_g2p` id mapping as a JP-Extra authentication boundary.

This is deliberately a fail-closed integration boundary: until a separately
audited native provider is wired by the integration layer, end-to-end Japanese
production G2P remains unavailable. The contract API itself is model-free and
safe to exercise with test providers.

The existing `integrations/vokra-piper-g2p` bridge cannot be used as that
provider. The pinned `piper-plus-g2p` Japanese source exposes
`phonemize_with_prosody` as `(Vec<String>, Vec<Option<ProsodyInfo>>)` and its
implementation inserts Piper framing/prosody tokens (`^`, `$`, `?`, `_`, `[`,
`]`, `#`) and PUA encodings. Its `ProsodyInfo` fields are A1/A2/A3 accent-label
values, not SBV2's binary raw tone vector, and the API has no `word2ph` output.
The bridge consequently exposes Piper voice ids/prosody/language ids only.
Converting those values into SBV2 phones, tones, or boundaries would be an
unverified semantic guess, so no adapter is provided.

The direct MIT `jpreprocess 0.9.1` route was checked as well. Its public
`JPreprocess::extract_fullcontext()` returns `Vec<jlabel::Label>`; the labels
carry the current phone and A/F accent context, but not the original symbol
kind/count needed by JP-Extra. The upstream SBV2 maintainer documents this
boundary explicitly: full-context extraction loses punctuation/symbol
distinctions, while the production algorithm combines an accent-aware
phoneme path with a separate frontend path that preserves those symbols. The
official explanation also defines SBV2 tones as a separate low/high `0/1`
value per emitted phone. Therefore `jpreprocess` labels alone cannot prove
the authenticated `phones + raw_tones + word2ph` tuple; deriving it would be
an unverified clean-room guess. `jpreprocess::run_frontend()` exposes only
serialized NJD features and its own API documents that original strings are
dropped, so it does not supply the missing production tuple either.

Primary source: the [immutable upstream Japanese-processing explanation](https://github.com/litagin02/Style-Bert-VITS2/blob/ef93f388fc1ddf0dc0f598126c1964923f1df94f/docs/Style-Bert-VITS2_en.md#japanese-language-processing).

The fixed upstream `text/symbols.py` contract authenticates the exact schema
values required by `validate_contract.py`: `language_id_map` is
`{"ZH": 0, "JP": 1, "EN": 2}`, tone counts are `ZH=6`, `JP=2`, `EN=4`,
the tone starts are `ZH=0`, `JP=6`, `EN=8`, and the authenticated source boundary is
`n_vocab=112`, `n_tones=12`. This vocabulary count is fixed by the authenticated
source table rather than inferred from model weights: the official model
implementation uses `len(symbols)` for its embedding size in
`models_jp_extra.py` (blob `1bb2dd2e3b76a3d1978cd591eb4d4415c4c8a9ba`), and
the training entry point passes `len(symbols)` to `SynthesizerTrn` in
`train_ms_jp_extra.py` (blob `4ac102adf1b264c7bd0488b76a5e0174bc988ac2`).
The JP-only worker uses the official shared
mapper from `text/__init__.py` to convert raw JP tones into the global `6..7` band;
no English frontend is loaded. These source dimensions do not establish
real-weight compatibility: the strict GGUF weight binder and numerical parity
gate remain separate.

The runtime's ordered-symbol digest is
`7e1f4566310f17b6196d4b51300c7e760d5c5fe60ef4dd49e0b7e1d477740960`. It is the
SHA-256 of the exact `phoneme_symbols` array in the exact-head VAST evidence
recorded in `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md`,
using the canonical byte stream `UTF-8(symbol) || 0x00` for every symbol in
source order. The archive
`/private/tmp/vokra-pre-scaleway-final-evidence-86efd5cf.tar.gz` is retained
only as the recovered path from that maintainer session. This digest
authenticates the table without copying the AGPL source table into the runtime.

## Primary-source identity checks

These read only immutable commit/tree metadata; they do not fetch or execute
AGPL source. Run them in the VAST worker before generating the sidecar:

```sh
gh api repos/litagin02/Style-Bert-VITS2/commits/ef93f388fc1ddf0dc0f598126c1964923f1df94f \
  --jq .sha
gh api 'repos/litagin02/Style-Bert-VITS2/git/trees/ef93f388fc1ddf0dc0f598126c1964923f1df94f?recursive=1' \
  --jq '.tree[] | select(.path == "text/symbols.py" or .path == "text/japanese.py" or .path == "text/japanese_mora_list.py" or .path == "common/log.py" or .path == "common/stdout_wrapper.py" or .path == "text/__init__.py" or .path == "LICENSE" or .path == "bert/deberta-v2-large-japanese-char-wwm/config.json" or .path == "bert/deberta-v2-large-japanese-char-wwm/special_tokens_map.json" or .path == "bert/deberta-v2-large-japanese-char-wwm/tokenizer_config.json" or .path == "bert/deberta-v2-large-japanese-char-wwm/vocab.txt") | [.path,.sha] | @tsv'
curl --fail --silent --show-error \
  https://huggingface.co/api/models/litagin/Style-Bert-VITS2-2.0-base-JP-Extra/revision/a731761009f3c96d104487be6ad332bf1bb5a3a5 \
  | jq -e --arg rev a731761009f3c96d104487be6ad332bf1bb5a3a5 '.sha == $rev'
```

The corresponding immutable primary-source URLs are the [HF revision](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra/commit/a731761009f3c96d104487be6ad332bf1bb5a3a5), the [official source commit](https://github.com/litagin02/Style-Bert-VITS2/commit/ef93f388fc1ddf0dc0f598126c1964923f1df94f), and the authenticated `symbols.py`, `japanese.py`, `japanese_mora_list.py`, `common/log.py`, `common/stdout_wrapper.py`, `text/__init__.py`, `LICENSE`, and DeBERTa tokenizer files at that commit.

The fixed `text/__init__.py` path applies the language start offset to raw
G2P tones. Therefore JP raw tones `0/1` become global SBV2 tone ids `6/7`
(`tone_offsets["JP"] = 6`); a JP fixture containing ZH-band `0..5` or EN-band
`8..11` is invalid and rejected by the validator.

## License and execution boundary

The official implementation is AGPL-3.0. It may be executed only as an
independent reference in the VAST parity worker; no AGPL source may be copied
into Apache-licensed runtime, converter, or integration code. No model weights
are acquired or executed locally. Callers without the audited sidecar and a
wired native provider must receive an explicit failure.
