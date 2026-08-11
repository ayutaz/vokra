---
name: vast-ai-workflow
description: 大モデル（>8 GB safetensors）を vast.ai で変換 / 検証 / publish するときに使う。M1 iMac（16 GB）で mmap すると swap で死ぬのを避け、`provision.sh` の 4 gotcha（hf_config.pth shim / huggingface_hub<0.30 pin / certifi / stack tool install）+ rent→provision→work→destroy lifecycle + Voxtral streaming reader パターン + FA v3 Hopper bakeoff の runbook を示す。
---

# vast.ai で大モデルを扱う

**単一事実源**: `scripts/publish/vast-ai/provision.sh` + `docs/handoff/vast-ai-large-model-publish.md`。本 skill はそれを skill 表現に翻訳したもの。

**大原則**: 依頼者機（M1 iMac 16 GB）で >8 GB safetensors を mmap すると swap が急伸して macOS が強制終了する。Voxtral-Small-24B (48 GB) で swap 40 GB 到達実証済。→ **vast.ai へ escalate**。

## 0. いつ vast.ai を使うか

**要 vast.ai**:
- >8 GB safetensors の convert（converter が全 tensor を Python 側で touch する場合）
- >8 GB GGUF の publish（30 Mbps outbound で数時間、vast.ai の 2.6 Gbps で数分）
- H100 / A100 が必要な bakeoff（FA v3 Hopper、CoreML/ANE は別 = 実機 Mac / iPhone）

**M1 iMac で OK（実測）**:
- whisper-* (最大 2.9 GB)
- csm-1b (6.21 GB tight OK)
- kyutai-stt (5.23 GB)
- parakeet-* (2.5-4.25 GB)
- **restamp_provenance 経路**: **8.7 GB Voxtral を peak 6.4 MB で publish 実績あり**（mmap 読取のみ、tensor コピーなし。→ skill `publish-model-to-hf` §7）

**要 vast.ai 実例**:
- Voxtral-Small-24B (48 GB)
- Kimi-Audio-7B 系
- 30B+ 全般

[[feedback-large-models-on-vast-ai]]。

## 1. Lifecycle（4 phase）

1. **rent**: vast.ai 上で GPU instance を借りる（cheapest でも RAM ≥64 GB / disk ≥200 GB は必須、convert 用途なら GPU は 4090 で十分、H100 は FA v3 bench 用）
2. **provision**: `scripts/publish/vast-ai/provision.sh` を SSH 上で実行（4 gotcha を pre-handle）
3. **work**: `run-one.sh` per model or 直接 cargo コマンド
4. **destroy**: **必ず `vastai destroy` で auto-destroy**（走らせっぱなしは $/h で課金継続、ADR §D6）

## 2. Rent phase（vast.ai 側）

```bash
# 手元 CLI（vastai）で GPU offer 検索
vastai search offers 'gpu_name=RTX_4090 rentable=true' --order 'dph_total'
# ↑ 一番安いのを選ぶ

vastai create instance <offer-id> \
  --image nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --disk 200 \
  --ssh
```

**image 選定注意**: `nvidia/cuda:13.0.0` の stock image は 4 個の gotcha を含む（下記 §3）→ `12.4.1-devel-ubuntu22.04` が実績 stable。H100 bakeoff で 13.0.0 が必要な場合も `provision.sh` の `harden_vast_docker_image` が pre-handle する。

## 3. Provision phase — 4 gotcha を pre-handle

vast.ai の stock image が持つ 4 個の trap を `provision.sh` が **`install_uv` より前**に潰す。個別に対処するのは実績上 1 day 溶かす:

| # | Gotcha | 症状 | Fix |
|---|--------|------|-----|
| A | **hf_config.pth site-packages shim** | Python startup shim が `HF_ENDPOINT` を malicious mirror `117.175.104.83:8081` に書換、全 large DL が 404 | `/usr/local/lib/python3.10/dist-packages/hf_config.pth` + `/usr/lib/python3/dist-packages/pip/_vendor/hf_config.pth` を削除 |
| B | **huggingface_hub >= 0.30 non-xet regression** | xet-token routing が mirror 404 を投げる、`HF_HUB_DISABLE_XET` も一部 bypass | `pip install 'huggingface_hub<0.30'` で pin（**vast.ai 上のみ**、local は pin 不要） |
| C | **certifi CA bundle 空/stale** | HTTPS 全般 fail | `certifi` を再植え付け |
| D | **torch/numpy/safetensors が system layer に無い** | `python3 -c` fallback や `uv_cmd` fallback が missing modules で死ぬ | pre-install で torch + numpy + safetensors を system-level に置く |

**呼び方**:
```bash
# vast.ai instance に SSH した後
export HF_TOKEN='hf_xxxxxxxx'
git clone https://github.com/ayutaz/vokra.git ~/vokra
cd ~/vokra
bash scripts/publish/vast-ai/provision.sh  # idempotent, rerun-safe
```

**idempotent**: rerun-safe。各 step が artifact を probe して skip する。`git pull` 後にも安全に再実行できる。

[[reference-vast-ai-hf-config-pth-shim]] / [[reference-huggingface-hub-lt-030-vast-ai]]。

## 4. Work phase

### 4.1 Convert + publish 1 モデル

```bash
# provision 済 instance 上
export VOKRA_PUBLISH_ON_VAST=1  # publish-one.sh gate 7 の implicit bypass
scripts/publish/vast-ai/run-one.sh <slug>  # convert → stage → push chain
```

### 4.2 Voxtral streaming reader パターン（sharded safetensors、mmap 節約）

- 通常は sharded safetensors を `tools/parity/<slug>_prepare_checkpoint.py` で事前 merge（→ skill `add-speech-model` §2.1）
- **Voxtral だけは例外**: TextDecoder の `Vec<f32>` eager binding が ~15 GB 要求で M1 を殺していた root cause → `MappedTextBlocks` / `MappedHeads`（mmap + tiled transpose、lm_head streaming）で **peak 15 GB → 3.55 GB**。同じ pattern を他 sharded モデルに横展開する場合は先例として参照
- 実装: `crates/vokra-models/src/voxtral/mapped_lazy.rs` 系。streaming 適用可能なモデルは事前 merge 不要（converter が sharded を直接読む）

### 4.3 大モデルの publish 直行（H100 レンタル + 即 push）

```bash
# 大 GGUF を convert する必要がなく、既存 HF から DL → restamp → repush だけの場合
# ローカル (M1 iMac) の restamp_provenance で peak 6.4 MB で完結する
# → vast.ai を借りずに済む（Voxtral 8.7 GB で実証）
```

**vast.ai を借りる vs 借りない判断**: convert が要る = 借りる / provenance だけ = ローカル restamp。

## 5. Destroy phase（**必ず**）

```bash
# 手元 CLI から
vastai destroy instance <instance-id>
```

**auto-destroy を仕込む**: measure が終わったら trap で自動 destroy する pattern（ADR §D6）:
```bash
# vast.ai instance 上
trap 'vastai destroy instance $VAST_CONTAINERLABEL' EXIT
# ↑ SSH セッション終了時に自動 destroy
```

**destroy 忘れは $/h で継続課金**。H100 は $1.7-2.5/h、8h 忘れると $15+ 溶ける。

## 6. FA v3 Hopper bakeoff runbook（M4-07 実績）

- **hardware**: H100 PCIe (SM 9.0, 80 GB VRAM)
- **image**: nvidia/cuda:12.4.1-devel-ubuntu22.04
- **session cost 実績**: $1.73 / 60 min（2026-08-10）
- **手順**: `docs/m4-07-hopper-bench-handover.md` §3 / `provision.sh` + `cargo build --release -p vokra-cli -p vokra-backend-cuda` + `tools/parity/cuda_rtf_variance.sh --iters 10 --fa-mode {decomposed,v2,v3}`
- **結果**: FA v3 で **5.7% e2e speedup** vs FA v2、causal parity 1.206e-2 / non-causal 1.026e-2（atol 0.02 内 60% / 51%）
- **注意**: `provision.sh` の `crates/vokra-cli` sanity check が glob-based workspace で misfire するので、直接 `cargo build --release -p vokra-cli` を叩く

## 7. 費用 vs 手間の判断

| Task | vast.ai 費用目安 | 判断 |
|------|--------------|------|
| Voxtral-Small-24B convert + publish | RTX 4090 8h × $0.30/h = $2.4 | 妥当（M1 で試すと mac 強制終了 = 復旧に時間 loss） |
| H100 FA v3 bakeoff | H100 60 min × $1.73/h = $1.73 | 妥当（M4-07 実績） |
| provenance だけの差替 | $0（ローカル restamp） | **借りない** |
| whisper-large-v3 convert | $0（M1 で 2.9 GB は余裕） | **借りない** |

## 8. 出禁パターン

- **vast.ai を借りっぱなしで放置**: $/h 課金継続、trap `vastai destroy` を必ず仕込む
- **provision.sh を skip して pip 手打ちで頑張る**: 4 gotcha に順番にハマる（実績 1 day 溶かす）
- **`huggingface_hub` を local と同じ最新版で使う**: vast.ai 上では <0.30 pin 必須（xet-token regression）、local との差分を明示的に持つ
- **`HF_TOKEN` を CLI 引数で渡す**: shell history + `ps` に残る → 環境変数経由で
- **borrowed instance で `.env` を書く**: destroy で消える + credential 漏洩リスク、環境変数を SSH セッション内で export

## 9. Cross-reference

- skill `publish-model-to-hf` — publish gate 5 段の詳細
- skill `add-speech-model` — sharded safetensors 事前 merge / GGUF 5D limit
- `scripts/publish/vast-ai/provision.sh` — 実装 (single source of truth)
- `docs/handoff/vast-ai-large-model-publish.md` — 実運用 runbook
- `docs/m4-07-hopper-bench-handover.md` — H100 bakeoff 実績
- `docs/bench-baselines/vast-2026-08-10-h100/` — H100 measurement data
