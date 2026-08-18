# vast.ai publish runbook — VoxCPM2-2B (openbmb/VoxCPM2)

## 0. 2026-08-18 VAST execution result

Conversion and real-weight structural verification are complete on VAST instance
`47955178` at commit `5bc62ae`. No model payload was loaded on the Mac.

- Immutable upstream revision:
  `bffb3df5a29440629464e5e839f4d214c8714c3d`.
- The upstream release is not a complete single safetensors checkpoint:
  `model.safetensors` has 577 BF16 tensors while `audiovae.pth` has 311 FP32
  tensors plus one pinned int32 sample-rate-boundary buffer. The official loader
  prefixes the VAE state with `audio_vae.` before loading it.
- `tools/parity/voxcpm2_prepare_checkpoint.py` verified all input hashes and
  emitted 888 tensors (577 BF16 + 311 FP32), 4,956,973,816 bytes, SHA-256
  `f8c8ed28b98e38378c5cf368933b0a7fef9df2cc6becee915d1ecca073035ffa`.
- `vokra-cli convert --model voxcpm2 ... --tokenizer tokenizer.json` emitted
  GGUF v3 with 888 tensors / 60 metadata keys / 4,960,621,760 bytes, SHA-256
  `1cdea939d265b9f64fafcd51470bf008a99d40659cf6545db9335f3d08509aa6`.
  Header verification confirmed 577 BF16, 311 F32, all AudioVAE sentinels,
  tokenizer length 3,676,772 bytes, and exact upstream repo/revision metadata.
- The raw main-only checkpoint is now refused (`577/888`, no output file), so
  the former success-shaped but non-decodable conversion path is closed.
- The real GGUF then exercised the Rust `parity_voxcpm2` structural leg. Its
  first run found that the harness incorrectly read the BOOL
  `vokra.voxcpm2.residual_lm.no_rope` as an integer; commit `e8d016f` fixed the
  test contract. The rerun passed with 888 tensors and the 2B runtime config.
  `VOKRA_VOXCPM2_REFDIR` was unset and native forward remains unimplemented, so
  this is explicitly not a numerical-output parity claim.
- `publish-one.sh` was run without `--push`. It stopped fail-closed at owner
  sign-off mapping with `UNKNOWN_REPO` (exit 5): the license audit row is signed,
  but the public slug `vokra/voxcpm2-2b` is intentionally not registered while
  the model's explicit voice-cloning positioning and the repository's
  voice-clone separation policy still need an owner destination/legal decision.
- The instance was stopped, not destroyed. Staged VoxCPM2 and corrected Voxtral
  artifacts remain on its volume pending explicit upload authorization.

The commands below are the current reproducible conversion procedure. Do not
use the old generic `run-one.sh` main-weight-only route, and do not add `--push`
until the destination decision is recorded.

**Related**:
- 本 runbook は `docs/handoff/vast-ai-large-model-publish.md`（総論 = §2 recipe /
  §3 provision.sh gotcha / §4 lifecycle）を **前提** とする。共通手順は総論を参照
  し、本 doc は VoxCPM2-2B に固有の差分のみを記述する。
- 設計仕様: `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`
  （Wave 0-4 実装計画、§5 owner Option A/B/C 判断待ち）
- 既存 published sibling: `huggingface.co/vokra/voxcpm-0.5b`

## 1. モデル情報

| 項目 | 値 |
|---|---|
| Upstream HF repo | `openbmb/VoxCPM2` |
| Upstream HF URL | https://huggingface.co/openbmb/VoxCPM2 |
| License (upstream code) | apache-2.0 |
| License (upstream weight) | apache-2.0 |
| SPDX (Vokra 判定) | `apache-2.0`（LicenseClass = Permissive） |
| Weight bundle | `model.safetensors` 4,580,080,592 bytes + `audiovae.pth` 376,951,122 bytes |
| Execution policy | **VAST-only** for download, preparation, conversion, verification, and any future upload |
| Vokra ModelKind | `VoxCpm2`（既存、CLI `--model voxcpm2`） |
| Variant marker | `vokra.model.name = "voxcpm2-2b"`（Option C hybrid、実GGUFで確認済み） |
| Arch tag | `vokra.model.arch = "voxcpm2"`（0.5B と同一、upstream `architecture` tag と一致） |
| Candidate HF slug | `vokra/voxcpm2-2b`（未承認・未登録。別リポdestinationの可能性あり） |
| Attribution 要求 | apache-2.0 標準の LICENSE + NOTICE 同梱のみ、runtime-side 追加なし |
| Non-commercial 制限 | なし |
| Share-alike | なし |

### Primary source verify command（本 doc land 時点で实行済）

```bash
# License verification
curl -sL https://huggingface.co/api/models/openbmb/VoxCPM2 | \
  uv run --no-project python -c "import json,sys; d=json.load(sys.stdin); \
    print('license:', d.get('cardData',{}).get('license')); \
    print('license_name:', d.get('cardData',{}).get('license_name')); \
    print('gated:', d.get('gated'))"
# expected:
#   license: apache-2.0
#   license_name: None
#   gated: False

# Size verification
./scripts/publish/check-model-size.sh openbmb/VoxCPM2
# inventory only; the active execution policy is VAST-only regardless of the
# helper's historical size bands
```

## 2. vast.ai instance recipe

**共通仕様**は `docs/handoff/vast-ai-large-model-publish.md` §2.2 を参照。VoxCPM2-2B
固有の緩和:

| 項目 | 推奨値 | 備考 |
|---|---|---|
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` or `nvidia/cuda:13.0.0-*` | 総論 §2.2 と同じ。`nvidia/cuda:13.0.0` 系は provision.sh gotcha §3 参照 |
| RAM | **32 GB 以上**（総論の 64 GB より緩和可） | 4.96 GB weight + convert working set + upload buffer。総論の 48 GB Voxtral 級 shards ではない |
| Disk | **80 GB 以上**（総論の 200 GB より緩和可） | 上流 DL 4.96 GB + GGUF 4.96 GB + HF cache buffer + 余裕 |
| GPU | Convert には不要（converter は CPU only） | vast.ai は GPU 前提販売ゆえ最安 GPU を選ぶ |
| Network | 非従量課金 or inclusive band | 上下 ~10 GB out-bound（DL 4.96 GB + upload 4.96 GB） |
| 課金見込 | ~1 hour × $0.3-0.5/hr = **$0.3-0.5** | Voxtral-Small-24B（$0.6-1.0）より安い |

## 3. provision.sh gotcha（総論 §3 を参照、以下は該当箇所）

`scripts/publish/vast-ai/provision.sh` は下記 4 件を idempotent に修正済（Wave 12
harden_vast_docker_image、2026-08-03 land）。VoxCPM2-2B convert 前に一度だけ実行:

| Gotcha | 起因 | provision.sh の対応 |
|---|---|---|
| **hf_config.pth shim** | `nvidia/cuda:13.0.0` image が仕込む Python startup shim が `HF_ENDPOINT` を malicious mirror `117.175.104.83:8081` に上書き | shim 除去 + certifi CA 再植え付け（memory [[reference-vast-ai-hf-config-pth-shim]]） |
| **huggingface_hub < 0.30 pin** | 1.x xet-token routing が mirror 404 を投げ、`HF_HUB_DISABLE_XET` も一部 bypass、0.30+ `resume_download` deprecated で flaky egress を落とす | vast.ai 上のみ `huggingface_hub < 0.30` に pin。local machine は pin 不要（memory [[reference-huggingface-hub-lt-030-vast-ai]]） |
| **certifi CA bundle** | 空 or 古い CA bundle で HTTPS 検証失敗 | `certifi` 再 install + `SSL_CERT_FILE` export |
| **stack tool install（torch/numpy/safetensors）** | VAST 用の `uv pip --system` compatibility layer に必要。実行・変換・検証はすべて uv-managed Python で行う | provision.sh Wave 12 で pre-install |

```bash
# Public upstream download and dry-run need no HF credential. Transfer a token
# only after an explicit upload approval and destination decision.

# 1 コマンドで Rust + uv + Python 3.12 + hf-transfer + repo + vokra-cli build まで完了
curl -sSL https://raw.githubusercontent.com/ayutaz/vokra/main/scripts/publish/vast-ai/provision.sh | bash
source ~/.bashrc  # VOKRA_PUBLISH_ON_VAST=1 marker を pick up
```

## 4. Complete-checkpoint conversion command

### 4.1 Generic pipeline status

`run-one.sh` cannot prepare the separately shipped `audiovae.pth`; using it
directly would hand the incomplete 577-tensor main checkpoint to the converter.
The converter now rejects that input. Use §4.2 until a dedicated wrapper lands.

### 4.2 手動 fallback（総論 §2.5 に準拠）

自動化 pipeline が事故った場合の手順:

```bash
mkdir -p ~/scratchpad/hf-cache ~/scratchpad/staging/voxcpm2-2b

cd ~/vokra
uv sync --project tools/parity   # pyproject.toml + uv.lock から依存 install

# HF から固定 revision の完全 bundle を DL
uv run --no-project --python 3.12 \
  --with 'huggingface_hub<0.30' --with hf-transfer python <<'PY'
import os
os.environ.setdefault('HF_HUB_CACHE', '/root/scratchpad/hf-cache')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='openbmb/VoxCPM2',
    revision='bffb3df5a29440629464e5e839f4d214c8714c3d',
    cache_dir='/root/scratchpad/hf-cache',
    allow_patterns=['model.safetensors', 'audiovae.pth', 'config.json',
                    'tokenizer.json', 'tokenizer_config.json',
                    'special_tokens_map.json'],
    token=os.environ.get('HF_TOKEN'),
)
print('DONE:', path)
PY

# Prepare + convert (VAST only)
SNAP=/root/scratchpad/hf-cache/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d
uv run --project tools/parity python \
  tools/parity/voxcpm2_prepare_checkpoint.py \
  --snapshot-dir "$SNAP" \
  --output /root/scratchpad/staging/voxcpm2-2b/complete.safetensors \
  --manifest /root/scratchpad/staging/voxcpm2-2b/prepare-manifest.json

./target/release/vokra-cli convert \
  --model voxcpm2 \
  --input /root/scratchpad/staging/voxcpm2-2b/complete.safetensors \
  --tokenizer "$SNAP/tokenizer.json" \
  --output /root/scratchpad/staging/voxcpm2-2b/model.gguf

# Publish gate inspection only. This currently exits 5 (UNKNOWN_REPO) by
# design until the voice-clone destination/legal decision is recorded.
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/voxcpm2-2b/model.gguf \
  --repo vokra/voxcpm2-2b \
  --license-spdx apache-2.0
```

### 4.3 Upstream layout

The pinned revision has one main safetensors file plus a PyTorch AudioVAE
sidecar. It is not sharded. `audiovae.pth` must be loaded only by the UV-managed
preparer with `torch.load(..., weights_only=True)`; Python never enters the
runtime or converter.

## 5. §3.1 sign-off status

**License status: signed Commercial on 2026-07-28. Publication destination:
unresolved and fail-closed.**

`docs/license-audit.md` §3.1 row 296 (`openbmb/VoxCPM2`) is signed Apache-2.0
Commercial. That answers redistribution permission only. It does not resolve
the separate repository policy for a model whose official positioning includes
voice cloning. Accordingly `scripts/publish/signoff_match.py` has no
`voxcpm2-2b` public-repo mapping and `publish-one.sh` stops at `UNKNOWN_REPO`.

**Owner action**:

1. Decide whether this release's stated voice-cloning purpose makes it a
   `vokra-voiceclone-experimental` artifact or whether a documented exception
   permits the main `vokra/` model namespace.
2. Ratify the corresponding M5-05 legal/consent/watermark posture.
3. Only after that decision, add the exact destination slug to
   `REPO_TO_SIGNOFF_ROWS` and re-run `publish-one.sh` without `--push`.
4. Transfer an HF token and add `--push` only after a fresh explicit upload
   authorization. The current conversation has not granted it.

**publish-one.sh の 5 gate**（総論 §2.5 と同じ）:

1. Catalog reality — 未実装の ★ 公式 zoo 宣言拒否
2. Redistributable — `LicenseClass::redistributable()` false 拒否（apache-2.0 は
   Permissive で pass）
3. Provenance chunk 刻印 — `vokra.provenance.*` chunk 群が missing なら拒否
4. §3.1 exact slug-to-row mapping — currently `UNKNOWN_REPO` by design even
   though the underlying Apache-2.0 row is signed
5. T4 (NonCommercial) は `--allow-noncommercial` 明示必須 — VoxCPM2-2B は T1
   ゆえ非該当

## 6. 期待される artifacts

Approved destinationへのpublish成功後に含めるartifact:

| ファイル | 内容 |
|---|---|
| `model.gguf` | 4,960,621,760 bytes; 888 tensors (577 BF16 main + 311 F32 AudioVAE) |
| `README.md` | `make_model_card.py` 自動生成、tier T1 obligation + apache-2.0 表記 + upstream 情報 |
| `LICENSE` | apache-2.0 canonical text（`fetch_license.sh --spdx apache-2.0` で取得、`https://huggingface.co/openbmb/VoxCPM2/raw/main/LICENSE` を pin） |
| `NOTICE` | apache-2.0 標準 NOTICE（attribution required、Copyright 表記あり） |
| `SOURCE.md` | 上流 URL + 再変換手順 + Vokra converter バージョン + commit SHA |

### GGUF metadata（vokra.* chunk 群）

`vokra-cli convert --model voxcpm2` が実artifactへ刻んだchunk（2026-08-18確認済み）:

| Key | 型 | 値 |
|---|---|---|
| `vokra.schema.version` | string | `"1"`（writer choke point で自動刻印） |
| `vokra.schema.producer` | string | `"vokra-cli-<version>"` |
| `vokra.model.arch` | string | `"voxcpm2"`（0.5B と同一） |
| `vokra.model.name` | string | `"voxcpm2-2b"`（0.5B と区別する variant marker） |
| `vokra.provenance.upstream_hf` | string | `"openbmb/VoxCPM2"` |
| `vokra.provenance.upstream_revision` | string | `"bffb3df5a29440629464e5e839f4d214c8714c3d"`（pinned SHA） |
| `vokra.provenance.upstream_license` | string | `"apache-2.0"` |
| `vokra.voxcpm2.lm.hidden_dim` | u32 | `2048`（0.5B は 1024） |
| `vokra.voxcpm2.lm.n_layer` | u32 | `28`（0.5B は 24） |
| `vokra.voxcpm2.lm.n_head` | u32 | `16` |
| `vokra.voxcpm2.lm.n_head_kv` | u32 | `2` |
| `vokra.voxcpm2.lm.kv_channels` | u32 | **`128`（新設、0.5B は非明示 = 64 derived）** |
| `vokra.voxcpm2.lm.ffn_dim` | u32 | `6144`（0.5B は 4096） |
| `vokra.voxcpm2.encoder.n_layer` | u32 | `12`（0.5B は 4） |
| `vokra.voxcpm2.encoder.kv_channels` | u32 | **`128`（新設）** |
| `vokra.voxcpm2.dit.n_layer` | u32 | `12`（0.5B は 4） |
| `vokra.voxcpm2.dit.kv_channels` | u32 | **`128`（新設）** |
| `vokra.voxcpm2.dit.mean_mode` | bool | **`false`（新設）** |
| `vokra.voxcpm2.residual_lm.n_layer` | u32 | `8`（0.5B は 6） |
| `vokra.voxcpm2.residual_lm.no_rope` | bool | **`true`（新設、0.5B は false）** |
| `vokra.voxcpm2.patch_size` | u32 | `4`（0.5B は 2） |
| `vokra.voxcpm2.max_length` | u32 | `8192`（0.5B は 4096） |
| `vokra.voxcpm2.scalar_quantization_latent_dim` | u32 | `512`（0.5B は 256） |
| `vokra.vae_continuous.sr_bin_boundaries` | u32[] | **`[20000, 30000, 40000]`（新設、0.5B は不在）** |

**設計仕様 = `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md` §1「5 新
metadata key（converter で追加）」を参照**（正確には 6 key = LM 側 3 + residual 1
+ dit 1 + vae 1）。

## 7. Gate 発火状態（parity CI）

`.github/workflows/parity-tts-continuous-vae-real.yml` は **既に VoxCPM2-2B の
pinned SHA で待機中**:

```yaml
env:
  VOXCPM2_REPO: openbmb/VoxCPM2
  VOXCPM2_REVISION: bffb3df5a29440629464e5e839f4d214c8714c3d
```

Runtime/converter factories are already landed. Remaining CI activation work:

1. **CI variable**: `VOKRA_TTS_CONT_VAE_ENABLE=1` を GitHub repo settings で set
2. **Fixture GGUF**: approved destinationからGGUFをCIがpull
   （`VOKRA_VOXCPM2_GGUF` env で pointing、workflow YAML 側で HF から fetch）
3. **PyTorch reference dump**: `VOKRA_VOXCPM2_REFDIR` 環境変数が pointing する
   directory に PyTorch reference の中間 tensor dump を配置（owner が生成、CI
   runner の uv-managed environment に `openbmb-voxcpm2` を install 済）

## 8. Owner critical path

1. Record the voice-clone destination/legal decision described in §5.
2. Obtain explicit authorization for the chosen remote write/upload.
3. Resume instance `47955178`; verify the two staged SHA-256 values from §0.
4. Add the approved slug mapping and run the dry-run again.
5. Only then run `publish-one.sh --push`, live-verify the remote artifact, and
   destroy the instance after both VoxCPM2 and retained Voxtral artifacts no
   longer need the volume.
6. Activate the real parity workflow with an independently generated upstream
   reference. Structural real-weight verification is complete, but numerical
   output parity has not been claimed. A weight-byte mirror is not an adequate
   independent reference; the future fixture must tap an actual upstream
   forward stage and compare it with a landed native Rust forward.

## 9. Notes

- **VoxCPM 0.5B との共存**: 既存 `vokra/voxcpm-0.5b`（0.5B、published 済）は
  この作業で touch しない。2Bのdestinationは§5の判断待ち。
- **Variant decision is implemented**: the landed Option C hybrid path records
  `vokra.model.arch=voxcpm2` and `vokra.model.name=voxcpm2-2b`; the real header
  was verified on VAST.
- Any future restamp or upload remains VAST-only under the active user rule,
  even if a metadata-only rewrite would theoretically fit on the Mac.

## See also

- **Priority ordering (2026-08-14)**: `docs/handoff/vast-ai-execution-priority.md`
  — conversion portion of Priority 1 is complete; publish is policy-blocked

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- 設計仕様: `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`
- Sibling: `huggingface.co/vokra/voxcpm-0.5b`（0.5B、published 済）
- CI workflow: `.github/workflows/parity-tts-continuous-vae-real.yml`（既 2B pin
  待機中）
- Memory: [[feedback-large-models-on-vast-ai]] / [[project-restamp-provenance]] /
  [[feedback-license-signoff-primary-source]] / [[reference-vast-ai-hf-config-pth-shim]] /
  [[reference-huggingface-hub-lt-030-vast-ai]]
