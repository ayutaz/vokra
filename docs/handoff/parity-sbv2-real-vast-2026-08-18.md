# SBV2 JA/EN + four-file ZH full-manifest real parity on VAST (2026-08-18)

## Outcome

The current three-checkpoint SBV2 JA/EN path completed on VAST instance
`47955178`. The independent Python reference produced all 11 manifest tensors,
four phonemizer side files, four acoustic-flow taps, and a 27,136-sample
waveform. The named non-ignored Rust consumer then passed every intermediate,
duration, waveform-length, waveform, and mel-loss assertion.

This closed the old implementation/runtime uncertainty for the current JA/EN
path. PR #33 later supplied its GitHub Actions verdict, and the four-file ZH
follow-up recorded below supplied the missing ZH numerical result. Neither run
constitutes an enabled UTMOS quality result.

## Reproducible inputs

The VAST run resolved and pinned these upstream revisions before downloading:

- SBV2 JP-Extra: `a731761009f3c96d104487be6ad332bf1bb5a3a5`
- DeBERTa v2 JA: `547b0e8b044fba3f9b84d0ab9f990440bd130c8b`
- DeBERTa v3 EN: `64a8c8eab3e352a784c658aef62be1662607476f`

The regenerated GGUF hashes were:

- `sbv2-v2-multilingual-base.gguf`:
  `19319c864ce210f630c021dece532d71d22a45b2573340601104f67cb3b26628`
- `deberta-v2-large-japanese-char-wwm.gguf`:
  `592f0a6b3f538b1b069562785205c8268f8351d654c7cdda53a695db947bb42a`
- `deberta-v3-large.gguf`:
  `858c07bf3909b1b8d25d7ca7c1e0fed0e7c2bd9889a0068c6ebcfd689d0eb579`

The first two matched their committed sidecars. The EN sidecar still carried
the pre-DeBERTa-fix artifact hash; it was updated to the already independently
validated `f43567b` converter output above.

## Environment recorded before the numerical run

- candidate commit: `4af6082549224044d52944675c5b21f4be5014f1`
- Linux x86_64, Xeon E5-2699 v4, AVX2
- memory: `528241436 KiB`
- Rust `1.97.1`, Cargo `1.97.1`
- uv `0.12.5`, Python `3.12.14`
- locked oracle: torch `2.6.0+cpu`, transformers `4.57.6`,
  huggingface-hub `0.36.0`, safetensors `0.5.3`, sentencepiece `0.2.2`,
  protobuf `5.29.5`

The dedicated Linux-x86_64 project is `tools/parity/sbv2/`. `uv sync
--project tools/parity/sbv2 --frozen` and its import/version probe passed on
VAST before any reference generation.

## Numerical result

Command shape:

```text
cargo test --locked --release -p vokra-models \
  --test parity_sbv2_real \
  parity_sbv2_real_waveform_matches_reference_dump -- --nocapture
```

The test passed in 72.22 seconds after the release test binary was built.
No tolerance was widened.

| Stage | max abs diff | Existing bound | Verdict |
|---|---:|---:|---|
| `phoneme_embed` | `0` | `0.01` | PASS |
| `text_hidden` | `4.395843e-7` | `0.01` | PASS |
| `bert_hidden_ja` | `2.956390e-5` | `0.05` | PASS |
| `bert_bridge_out` | `2.974272e-5` | `0.07` | PASS |
| `speaker_embed` | `0` | `0.01` | PASS |
| `style_projected` | `0` | `0.01` | PASS |
| `sdp_sample` | `0` | `0.05` | PASS |
| `mel_hidden` | `2.974272e-5` | `0.07` | PASS |
| `z_latent` | `3.314018e-5` | `0.08` | PASS |
| waveform length ratio | `1.0000` | `±10%` | PASS |
| `waveform` | `4.521498e-2` | `1.5` | PASS |
| mel-loss RMS | `1.655322e-1` | `0.3` | PASS |

The UTMOS tail gate was explicitly skipped because
`VOKRA_SBV2_UTMOS_ENABLE` was not set.

## Four-file ZH follow-up

Commit `e5641864e916a6f5a493aca14a0003bd1e23e6da` was run on disposable VAST
instance `47977839` using the tracked
`scripts/publish/vast-ai/run-sbv2-zh-parity.sh` worker. The worker refused
local execution, used only the locked `tools/parity/sbv2` uv project for
Python, and performed every download, conversion, reference forward, and
`vokra-models` Cargo invocation on VAST. No HF credential or upload was used.

The fourth pinned input was
`hfl/chinese-roberta-wwm-ext-large@a25cc9e05974bd9687e528edd516f2cfdb3f5db9`.
Its safe torch-pickle bridge wrote 399 safetensors entries, including an
explicit clone of the tied MLM decoder/word-embedding storage. The
`bert-base` converter emitted 389 runtime tensors, skipped ten documented
pooler/MLM-head tensors, embedded the 21,128-piece WordPiece vocabulary, and
reproduced the committed GGUF hash:

```text
a1a1df298fedb585b5278a2c048c5a11515968e2fdf43b856354f964c3e89b59  chinese-roberta-wwm-ext-large.gguf
```

The other three regenerated GGUFs also matched their committed sidecars. The
real upstream `transformers` reference loaded JA DeBERTa v2, EN DeBERTa v3,
and the ZH WordPiece/plain-BERT model; the selected ZH request produced
`T_text=5`, `bert_hidden_zh=[5,1024]`, `T_mel=75`, and a 38,400-sample
waveform. The Rust consumer passed `1 passed / 0 failed / 0 ignored` in
1026.70 seconds on Linux x86_64, Xeon E5-2673 v4, 125 GiB RAM, Rust 1.97.1,
Python 3.12.14, torch 2.6.0+cpu, and transformers 4.57.6.

| Stage | max abs diff | Existing bound | Verdict |
|---|---:|---:|---|
| `phoneme_embed` | `0` | `0.01` | PASS |
| `text_hidden` | `7.152557e-7` | `0.01` | PASS |
| `bert_hidden_zh` | `1.907349e-5` | `0.01` | PASS |
| `bert_bridge_out` | `5.960464e-6` | `0.07` | PASS |
| `speaker_embed` | `0` | `0.01` | PASS |
| `style_projected` | `0` | `0.01` | PASS |
| `sdp_sample` | `0` | `0.05` | PASS |
| `mel_hidden` | `5.960464e-6` | `0.07` | PASS |
| `z_latent` | `1.096725e-5` | `0.08` | PASS |
| waveform length ratio | `1.0000` | `±10%` | PASS |
| `waveform` | `1.031446e-1` | `1.5` | PASS |
| mel-loss RMS | `1.820711e-1` | `0.3` | PASS |

No tolerance changed. UTMOS was again an explicit skip. The ZH
`MinimalG2P` row is a fixed, byte-replayed parity input, not a claim that
production Mandarin G2P is complete. The base checkpoint is JP-Extra, so this
numerical implementation result also does not remove the documented degraded
ZH audio-quality caveat. After the text evidence was recorded, disposable
instance `47977839` was destroyed; the account was verified with no running
instance. The unrelated retained Voxtral volume remains stopped (`exited`).

## Workflow defects found and corrected

The VAST run exposed four stale CI assumptions:

1. ad-hoc `venv`/`pip` setup instead of a frozen uv project;
2. floating HF revisions and a dumper that reopened repo IDs instead of the
   already downloaded pinned snapshot directories;
3. no comparison between regenerated GGUFs and the committed sidecar hashes;
4. `cargo test ... -- --ignored`, even though WP-06 removed `#[ignore]` from
   the real numerical consumer on 2026-08-11. That option filtered out the
   intended test.

The workflow now uses the dedicated uv lock, pins all four possible revisions,
passes the selected local BERT snapshot paths into the dumper, verifies the
complete selected bundle, and explicitly runs the named non-ignored test.
`include_zh=false` remains the default so recurring JA cost is unchanged;
`include_zh=true` selects the proven four-file ZH leg.

## Retained VAST artifacts

The instance remains the owner-prescribed retained volume until this wave is
committed and final verification finishes. Current evidence is under:

- `/root/scratchpad/sbv2-family10-4af6082/checkpoints/`
- `/root/scratchpad/sbv2-family10-4af6082/logs/environment-full.txt`
- `/root/scratchpad/sbv2-family10-4af6082/logs/full-dump.log`
- `/root/scratchpad/sbv2-family10-4af6082/logs/full-parity.log`

The real GGUF and raw reference files remain gitignored. No Hugging Face upload
was attempted.
