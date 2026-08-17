# vast.ai publish runbook — VoxCPM2-2B (openbmb/VoxCPM2)

**Owner-triggered.** CC は本 doc 作成のみ。実 vast.ai instance の起動・convert・
publish は owner が本 runbook を追いながら実行する。

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
| Total safetensors size | **4.96 GB**（BF16） |
| 判定 (`check-model-size.sh`) | `LOCAL_OK`（4-8 GiB total、single-tenant なら local OK / 依頼者ルール #1 で ≥2GB は vast.ai 推奨） |
| Vokra ModelKind | `VoxCpm2`（既存、CLI `--model voxcpm2`） |
| Variant marker | `vokra.model.name = "voxcpm2-2b"`（0.5B との区別、Option C = Hybrid pattern を Wave 0 で owner 確定予定） |
| Arch tag | `vokra.model.arch = "voxcpm2"`（0.5B と同一、upstream `architecture` tag と一致） |
| Vokra HF slug | `vokra/voxcpm2-2b` |
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
# expected verdict: LOCAL_OK or LOCAL_BORDERLINE (4.96 GB BF16)
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
# SSH 接続後、まず HF token を export（instance destroy で消える）
export HF_TOKEN='hf_xxxxxx'   # 本機 .env の HF= 値をここに貼る

# 1 コマンドで Rust + uv + Python 3.12 + hf-transfer + repo + vokra-cli build まで完了
curl -sSL https://raw.githubusercontent.com/ayutaz/vokra/main/scripts/publish/vast-ai/provision.sh | bash
source ~/.bashrc  # VOKRA_PUBLISH_ON_VAST=1 marker を pick up
```

## 4. Convert + publish command

### 4.1 自動化 pipeline（推奨、Phase B）

```bash
# provision.sh 完了後、以下 1 コマンド:
~/vokra/scripts/publish/vast-ai/run-one.sh \
  --hf-repo openbmb/VoxCPM2 \
  --vokra-slug voxcpm2-2b \
  --model-kind voxcpm2 \
  --license-spdx apache-2.0 \
  --push
```

**注意**: `--push` を外せば dry-run stage のみ（5-gate verify のみ）。本番 upload
前に必ず dry-run で全 gate 通過を確認すること（総論 §2.5 と同じ規律）。

### 4.2 手動 fallback（総論 §2.5 に準拠）

自動化 pipeline が事故った場合の手順:

```bash
mkdir -p ~/scratchpad/hf-cache ~/scratchpad/staging/voxcpm2-2b

cd ~/vokra/tools/parity
uv sync   # pyproject.toml + uv.lock から依存 install

# HF から weight + config DL
uv run --with huggingface_hub python <<'PY'
import os
os.environ.setdefault('HF_HUB_CACHE', '/root/scratchpad/hf-cache')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='openbmb/VoxCPM2',
    cache_dir='/root/scratchpad/hf-cache',
    allow_patterns=['*.safetensors', 'model.safetensors.index.json',
                    'config.json', 'generation_config.json',
                    'tokenizer*.json', '*.md', 'LICENSE*'],
    token=os.environ['HF_TOKEN'],
)
print('DONE:', path)
PY

# Convert
SNAP=$(ls -d /root/scratchpad/hf-cache/models--openbmb--VoxCPM2/snapshots/*/ | head -1)
cd ~/vokra

# 単一 safetensors か multi-shard かを確認
if [ -f "$SNAP/model.safetensors.index.json" ]; then
    INPUT="$SNAP/model.safetensors.index.json"
else
    INPUT="$SNAP/model.safetensors"
fi

./target/release/vokra-cli convert \
  --model voxcpm2 \
  --input "$INPUT" \
  --config "$SNAP/config.json" \
  --output /root/scratchpad/staging/voxcpm2-2b/model.gguf

# Publish 5-gate（dry-run → --push で本番）
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/voxcpm2-2b/model.gguf \
  --repo vokra/voxcpm2-2b \
  --license-spdx apache-2.0
# ↑ dry-run 全 gate 通過を確認してから ↓
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/voxcpm2-2b/model.gguf \
  --repo vokra/voxcpm2-2b \
  --license-spdx apache-2.0 --push

# 検証
curl -sI https://huggingface.co/vokra/voxcpm2-2b | head -1
# HTTP/2 200 が返れば live
```

### 4.3 Sharded safetensors の場合

`openbmb/VoxCPM2` は BF16 4.96 GB ゆえ **単一 safetensors か 2 shard 程度** の可能
性が高い（HF snapshot で要確認）。もし `model.safetensors.index.json` が存在する
multi-shard であれば、上記 script の `INPUT` 変数は自動で index.json を指す。既存
の `vokra-cli convert --model voxcpm2` が `MappedSafetensors` 経路で index.json を
消費するため、事前 merge は不要（Voxtral の `convert_voxtral_file_streaming` と
異なり VoxCPM2 は size が中規模ゆえ streaming path 未実装、mmap で足りる想定）。

**もし convert が OOM で fail した場合**: 依頼者ルール #1（≥2GB は vast.ai）に従っ
て `--config` 側車で明示 variant 指定 or Option B の兄弟 file `voxcpm2_2b.rs` へ
Wave 0 ADR で切り替え（`docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md` §5
参照）。

## 5. §3.1 sign-off status

**現状: blank（fail-closed default）**。

`docs/license-audit.md` §3.1 に `vokra/voxcpm2-2b` 行を **追加待ち**（本 doc land
時点では未追加、Wave 1 land 時に追加予定）。追加後の状態:

| 列 | 予定値 |
|---|---|
| Vokra slug | `vokra/voxcpm2-2b` |
| Upstream HF | `openbmb/VoxCPM2` |
| Category | Multilingual TTS (30 languages) |
| SPDX | apache-2.0 |
| Vokra tier | T1 (Permissive Commercial) |
| Commercial sign-off | **☐（空欄）** |
| Sign-off date | **☐（空欄）** |
| Signer | **☐（空欄）** |

**Owner action**:

1. **Primary source を直接照合** — https://huggingface.co/openbmb/VoxCPM2 の HF
   model card + LICENSE + config.json で apache-2.0 表記を確認
2. yousan として **☑ Commercial** sign-off（`docs/license-audit.md` §3.1 template
   を使用、CC の primary-source-transcribable pattern で埋める、memory
   [[feedback-license-signoff-primary-source]] の rule 準拠）
3. Sign-off 後に §4 の `publish-one.sh --push` が gate 通過（5 gate 目 = §3.1
   sign-off blank refuse が unblock）

**publish-one.sh の 5 gate**（総論 §2.5 と同じ）:

1. Catalog reality — 未実装の ★ 公式 zoo 宣言拒否
2. Redistributable — `LicenseClass::redistributable()` false 拒否（apache-2.0 は
   Permissive で pass）
3. Provenance chunk 刻印 — `vokra.provenance.*` chunk 群が missing なら拒否
4. §3.1 sign-off 欄 blank 拒否 — **fail-closed default、上記 owner action 必須**
5. T4 (NonCommercial) は `--allow-noncommercial` 明示必須 — VoxCPM2-2B は T1
   ゆえ非該当

## 6. 期待される artifacts

Publish 成功後の `huggingface.co/vokra/voxcpm2-2b` repo に含まれる:

| ファイル | 内容 |
|---|---|
| `model.gguf` | ~4.96 GB（BF16 pass-through 変換、tensor 数は上流 safetensors index に依存） |
| `README.md` | `make_model_card.py` 自動生成、tier T1 obligation + apache-2.0 表記 + upstream 情報 |
| `LICENSE` | apache-2.0 canonical text（`fetch_license.sh --spdx apache-2.0` で取得、`https://huggingface.co/openbmb/VoxCPM2/raw/main/LICENSE` を pin） |
| `NOTICE` | apache-2.0 標準 NOTICE（attribution required、Copyright 表記あり） |
| `SOURCE.md` | 上流 URL + 再変換手順 + Vokra converter バージョン + commit SHA |

### GGUF metadata（vokra.* chunk 群）

`vokra-cli convert --model voxcpm2` が刻む chunk（Wave 1 land で完成予定）:

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

Owner が下記を全て満たすと **flip the switch で発火**（新規 workflow 不要）:

1. **Runtime 側**: `VoxCpm2Config::voxcpm2_2b()` factory の追加（Wave 1、
   `crates/vokra-models/src/voxcpm2/mod.rs`）
2. **Converter 側**: `crates/vokra-convert/src/models/voxcpm2.rs` の variant-aware
   dispatch（Wave 0 ADR 確定 → Option C hybrid の場合 auto-detect by
   `base_lm.embed_tokens.weight` の hidden_size shape）
3. **CI variable**: `VOKRA_TTS_CONT_VAE_ENABLE=1` を GitHub repo settings で set
4. **Fixture GGUF**: 上記 §4 で publish した `vokra/voxcpm2-2b` を CI が pull
   （`VOKRA_VOXCPM2_GGUF` env で pointing、workflow YAML 側で HF から fetch）
5. **PyTorch reference dump**: `VOKRA_VOXCPM2_REFDIR` 環境変数が pointing する
   directory に PyTorch reference の中間 tensor dump を配置（owner が生成、CI
   runner に `pip install openbmb-voxcpm2` の Python 依存を install 済）

## 8. Owner critical path

**依頼者ルール #3（publish は §3.1 sign-off 完了後 owner が判断）** に従い、以下
順序で:

1. **CC 側実装完了確認** — Wave 1 (Runtime `VoxCpm2Config::voxcpm2_2b()` +
   Converter variant-aware) が land されたことを確認
2. **Wave 0 ADR 確定** — Option A / B / C（`docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`
   §5）から Option C（Hybrid）を推奨、owner の最終判断
3. **HF primary source 直接照合** — https://huggingface.co/openbmb/VoxCPM2 の
   LICENSE / README / config.json / cardData を目視確認
4. **§3.1 sign-off** — `docs/license-audit.md` §3.1 に yousan として ☑ Commercial
   2026-XX-XX sign
5. **vast.ai instance 起動** — §2 recipe に従って rent（~$0.3-0.5、~1 hour）
6. **§3 provision.sh 実行** — 1 コマンド
7. **§4.1 run-one.sh 実行** — dry-run → `--push`
8. **§7 CI flip the switch** — variable + fixture 配置 → workflow_dispatch

## 9. Notes

- **VoxCPM 0.5B との共存**: 既存 `vokra/voxcpm-0.5b`（0.5B、published 済）は
  この作業で touch しない。sibling として `vokra/voxcpm2-2b` を新設する。
- **`vokra.model.name` 値の Wave 0 ADR 依存**: Option A / B / C のいずれを選ぶか
  で `vokra.model.name` の実値と `vokra.model.arch` の実値が変わる。上記 §6 の
  metadata 表は Option C（Hybrid、推奨）前提。Option B（別 arch tag）を選ぶと
  arch が `"voxcpm2_2b"` に変わる。
- **restamp_provenance で低メモリ再刻印可能**: 一度 publish した後 LICENSE /
  NOTICE / SOURCE.md を差し替えたい場合、`restamp_provenance` で tensor コピー
  無しで刻印可能（memory [[project-restamp-provenance]]、Voxtral 8.7 GB を M1
  iMac 16 GB で peak footprint 6.4 MB 実測）。VoxCPM2-2B の 4.96 GB は Voxtral
  より小さいゆえ同手法で余裕を持って再刻印可能 = vast.ai 再起動不要。

## See also

- **Priority ordering (2026-08-14)**: `docs/handoff/vast-ai-execution-priority.md`
  — 本 job は **Priority 1** (最初に実行)、local first 試行推奨

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- 設計仕様: `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`
- Sibling: `huggingface.co/vokra/voxcpm-0.5b`（0.5B、published 済）
- CI workflow: `.github/workflows/parity-tts-continuous-vae-real.yml`（既 2B pin
  待機中）
- Memory: [[feedback-large-models-on-vast-ai]] / [[project-restamp-provenance]] /
  [[feedback-license-signoff-primary-source]] / [[reference-vast-ai-hf-config-pth-shim]] /
  [[reference-huggingface-hub-lt-030-vast-ai]]
