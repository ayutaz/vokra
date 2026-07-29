# vast.ai runbook — 大きめ音声モデルの convert + publish

Tracked / public。**2026-07-28** に本 M1 iMac (16GB RAM) 上で Voxtral-Small-24B-2507 (48GB BF16、11 shards) を convert 試行中、`SafetensorsFile::open` が全 shard を mmap した結果 swap 40GB used に到達 = OS レベル force shutdown リスクが実証されたことを受け、**8GB 超のモデル weight は本機で処理せず vast.ai の GPU box (副次的にメモリの多い host) で処理する** ことに 2026-07-28 決定。本 runbook は同 決定に基づく手順集。

memory [[feedback-large-models-on-vast-ai]] の運用側詳細版。

## 0. TL;DR — 自動化 pipeline (Phase B, 2026-07-28)

**判定**: `scripts/publish/check-model-size.sh <hf-repo>` を local で走らせて `LOCAL_SAFE / LOCAL_OK / LOCAL_BORDERLINE / VAST_AI_REQUIRED` の verdict を確認。

**vast.ai 側で最初の 1 回だけ**:
```bash
# SSH 接続後、まず HF token を export (instance destroy で消える)
export HF_TOKEN='hf_xxxxxx'

# 1 コマンドで Rust + uv + Python 3.12 + hf-transfer + repo + vokra-cli build まで完了
curl -sSL https://raw.githubusercontent.com/ayutaz/vokra/main/scripts/publish/vast-ai/provision.sh | bash
source ~/.bashrc  # VOKRA_PUBLISH_ON_VAST=1 marker を pick up
```

**モデル per に 1 コマンド**:
```bash
# 例: Voxtral-Small-24B (48 GB、必ず vast.ai)
~/vokra/scripts/publish/vast-ai/run-one.sh \
  --hf-repo mistralai/Voxtral-Small-24B-2507 \
  --vokra-slug voxtral-small-24b-2507 \
  --model-kind voxtral \
  --license-spdx apache-2.0 \
  --push
```

`run-one.sh` は `HF snapshot_download (hf-transfer で 40x) → autodetect input → vokra-cli convert → publish-one.sh` を chain。`--push` を外せば dry-run stage のみ。T4 (非商用) は `--allow-noncommercial`、T3 (Copyleft) は `--acknowledge-copyleft`。詳細は `run-one.sh --help`。

**local から誤って大モデルを publish する事故防止**: `publish-one.sh` に **gate 7 (>8 GiB fail-closed)** を追加済 (2026-07-28)。`VOKRA_PUBLISH_ON_VAST=1` 環境変数 (provision.sh が自動 set) がある instance では auto-bypass。owner 明示の `--allow-large` でも bypass 可 (自分の upload 帯域を分かってる時のみ)。

自動化 pipeline が事故ったときの手動 fallback として §2 を残す。

## 1. どのモデルが vast.ai 必須か

**Absolutely vast.ai (本機不可)**:
- Voxtral-Small-24B-2507 (48GB BF16、11 shards、mistralai/Voxtral-Small-24B-2507)
- Kimi-Audio-7B-Instruct (~14GB BF16 見込、moonshotai/Kimi-Audio-7B-Instruct、BF16 fleet)
- Voxtral-Mini-3B-2507 は既 published (2026-07-23) だが再変換なら borderline (~9GB BF16 → tight)
- 将来の 30B+ モデル (Qwen3-Audio-30B、Baichuan-Audio 系)

**Voxtral streaming path (2026-07-29 追加)**: `convert_voxtral_file_streaming`
API を追加した (M5 gap A-3、crates/vokra-convert/src/lib.rs)。header-only
mmap per shard + 1-tensor-at-a-time payload streaming で、Voxtral-Small-24B
(48GB) を M1 iMac (16GB) 上で peak `max(shard_header) + max(tensor_payload)`
のフットプリントで変換可能。**K-quant はスコープ外** (widen-then-quantize が
in-memory 必須ゆえ)。owner が local で dry-run するときは `convert_voxtral_file`
の代わりにこれを使う。vast.ai は引き続き quantize 系や 30B+ の base case として
必要。

**Borderline (single-tenant なら本機可、他作業と競合させない)**:
- 5-8GB の safetensors: kyutai-stt (5.23GB BF16、2026-07-28 実績あり)、csm-1b (6.21GB single-file、実績あり)、moshiko-7b (~14GB BF16、既 published)
- 判定基準: `curl -sL "https://huggingface.co/api/models/<repo>?blobs=true"` で最大 shard サイズ + shards 数から推定。**11 shards × 4GB = mmap 44GB → vast.ai**、単一 shard 6GB → local single-tenant で OK。

**Safe locally (問題なく処理可)**:
- ≤4GB の safetensors全般 (whisper-* / kokoro / silero / piper / dfn3 / utmos / dac / mimi / parakeet-tdt / parakeet-ctc / distil-whisper / kotoba-whisper / dia / qwen3-tts / vibevoice-1.5b / voxcpm-0.5b / cosyvoice2-0.5b / chatterbox-* / irodori / zonos / xcodec2)

## 2. vast.ai 手順 (Voxtral-Small-24B-2507 を例に)

### 2.1 事前準備 (本機 CC で実施可)

- **§3.1 sign-off 確認**: `docs/license-audit.md` §3.1 で該当 row が signed。Voxtral-Small-24B は row 250 = ☑ Commercial 2026-07-23 yousan 済。
- **HF token 確認**: `HF_TOKEN` (or `HF`) を本機の `.env` から取得 → vast.ai インスタンスに `export` で渡す。
- **branch 確認**: vast.ai には `main` (or 現行 branch) を clone。scratch 系変更が残る場合は事前 commit + push。

### 2.2 vast.ai インスタンス起動

- **Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (Ubuntu 22.04 + CUDA、Python 3.10 前提)。または `nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04` + apt で Rust/Python install。
- **RAM**: 最低 **64GB** (48GB shards mmap + convert working set + upload buffer)。vast.ai の "RAM" fetch フィルタで `>= 64` を指定。
- **Disk**: 最低 **200GB** (upstream DL 48GB + GGUF 48GB + HF cache buffer 40GB)。vast.ai の "Disk Space" で `>= 200` を指定。
- **GPU**: GPU は convert には**不要** (converter は CPU only、ただし vast.ai は GPU box を売っている都合上 GPU 付きが安いことが多い)。安さ最優先で "cheapest with 64GB RAM" を選ぶ。
- **ネットワーク**: HF DL + HF upload で数十 GB out-bound → **ネットワーク非従量課金** or **inclusive** の box を選ぶ (vast.ai の bandwidth 課金は host 依存)。
- **課金**: 見込 wall clock = DL 20 分 + convert 10-30 分 + upload 30-60 分 = **~2 hours × $0.3-0.5/hr = $0.6-1.0**。auto-destroy 後 cleanup 済確認。

### 2.3 SSH 接続後の初期セットアップ (~5 min)

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# uv (Python 管理は uv、[[feedback-python-uses-uv]] + [[feedback-python-3-12]])
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.12

# repo clone (public repo だが .env は含まれない — HF_TOKEN は下記 export で渡す)
git clone https://github.com/ayutaz/vokra.git ~/vokra
cd ~/vokra

# HF token 手動 export (公開しない、instance destroy で消える)
export HF_TOKEN='hf_xxxxxxxxxxxxxx'   # 本機 .env の HF= 値をここに貼る
export HF='$HF_TOKEN'
```

### 2.4 vokra-cli release build (~5 min)

```bash
cd ~/vokra
cargo build --release -p vokra-cli
# バイナリ = target/release/vokra-cli
```

### 2.5 upstream weight DL + convert + publish

Voxtral-Small-24B-2507 の場合:

```bash
mkdir -p ~/scratchpad/hf-cache ~/scratchpad/staging/voxtral-small-24b-2507

cd ~/vokra/tools/parity
uv sync   # pyproject.toml + uv.lock から依存 install

# HF から shards のみ DL (consolidated.safetensors 除外 = 48GB DL、96GB DL 回避)
uv run --with huggingface_hub python <<'PY'
import os
os.environ.setdefault('HF_HUB_CACHE', '/root/scratchpad/hf-cache')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='mistralai/Voxtral-Small-24B-2507',
    cache_dir='/root/scratchpad/hf-cache',
    allow_patterns=['model-*.safetensors', 'model.safetensors.index.json',
                    'config.json', 'generation_config.json', 'tekken.json',
                    'params.json', '*.md', 'special_tokens_map.json',
                    'tokenizer*.json', 'preprocessor_config.json'],
    token=os.environ['HF_TOKEN'],
)
print('DONE:', path)
PY

# Convert (multi-shard 経由)
SNAP=$(ls -d /root/scratchpad/hf-cache/models--mistralai--Voxtral-Small-24B-2507/snapshots/*/ | head -1)
cd ~/vokra
./target/release/vokra-cli convert \
  --model voxtral \
  --input "$SNAP/model.safetensors.index.json" \
  --config "$SNAP/config.json" \
  --output /root/scratchpad/staging/voxtral-small-24b-2507/model.gguf

# Publish 5-gate (dry-run で verify → --push で本番)
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/voxtral-small-24b-2507/model.gguf \
  --repo vokra/voxtral-small-24b-2507 \
  --license-spdx apache-2.0
# ↑ dry-run 全 gate 通過を確認してから ↓ --push で本番
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/voxtral-small-24b-2507/model.gguf \
  --repo vokra/voxtral-small-24b-2507 \
  --license-spdx apache-2.0 --push

# 検証
curl -sI https://huggingface.co/vokra/voxtral-small-24b-2507 | head -1
# HTTP/2 200 が返れば live
```

### 2.6 instance destroy (billing 抑制)

vast.ai UI から即 destroy、または `vastai destroy <instance-id>` (CLI 使用時)。**upload 完了 → live 確認 → destroy** の順で、GGUF は remote に残らない。次回同モデル再 publish 時は本機で `restamp` するか、vast.ai を再起動。

## 3. int tensor 対応 (parakeet 系で発生した pattern)

一部 checkpoint に `num_batches_tracked` (BatchNorm training-only int64 counter) 等の inference-inert int tensor が入っている。Vokra converter は F32/F16/BF16 のみ受け付けるため、**convert 前に strip する**:

```bash
# tools/parity/strip_int_tensors.py で inference-inert int tensor を除去
uv run python ~/vokra/tools/parity/strip_int_tensors.py \
  --input  "$SNAP/model.safetensors" \
  --output /root/scratchpad/staging/<model>/model.stripped.safetensors
# manifest sidecar (.stripped-manifest.json) が dropped tensor 一覧を記録
```

Voxtral-Small-24B は全 tensor が BF16 = strip 不要 (2026-07-28 事前確認済)。

## 4. 事前サイズ確認 command

vast.ai へ移送するか本機で処理するかの判定は事前に:

**推奨 (2026-07-28〜)**: `scripts/publish/check-model-size.sh` を使う。HF API を叩いて上表の threshold で機械判定 + rationale + 誘導先を human-readable で出力。exit code は `VAST_AI_REQUIRED` = 1、それ以外 = 0 ゆえ script chain にも使える。

```bash
# 人間向け表示
./scripts/publish/check-model-size.sh mistralai/Voxtral-Small-24B-2507

# JSON (jq / 他 script 消費用)
./scripts/publish/check-model-size.sh --json openai/whisper-base
```

**手動 fallback** (自動化 script が使えない環境):

```bash
# HF API で合計 safetensors サイズ確認
curl -sL "https://huggingface.co/api/models/<repo>?blobs=true" | python3 -c "
import json, sys
d = json.load(sys.stdin)
total = 0
for s in d.get('siblings', []):
    if s.get('rfilename', '').endswith('.safetensors'):
        sz = s.get('size', 0) or 0
        total += sz
        print(f'  {sz:>12} {s[\"rfilename\"]}')
print(f'TOTAL: {total:,} bytes = {total/1024**3:.2f} GiB')
"
```

**判定 threshold** (check-model-size.sh と runbook 側で同期):

- **≤4 GiB total**: `LOCAL_SAFE` — 本機で OK。
- **4-8 GiB total, max shard ≤6 GiB**: `LOCAL_OK` — 本機で single-tenant。
- **8-16 GiB or shards ≥5**: `LOCAL_BORDERLINE` — 本機は single-tenant で慎重に (他ビルド/テスト全部止める)。可能なら vast.ai。
- **>16 GiB**: `VAST_AI_REQUIRED` — **vast.ai 必須**。M1 iMac (16GB RAM) で mmap すると swap thrash → Mac 強制終了リスク。

## 5. Owner action

- **HF_TOKEN の vast.ai instance への手渡し**: 本機 `.env` の `HF=hf_xxx` 値を SSH セッションで export。**instance 破棄で消える** ので secret 漏洩なし。
- **課金承認**: 1 モデル publish あたり ~$0.6-1.0。事前予算目安。
- **live 確認**: `curl -sI https://huggingface.co/vokra/<name>` が HTTP/2 200 を返せば live、destroy 進めて OK。

## 6. 現時点の owner queue (2026-07-28)

vast.ai 必須の implementation-ready モデル (§3.1 signed Commercial):
- **Voxtral-Small-24B-2507** — Apache-2.0、row 250 signed 2026-07-23 yousan、converter 完備 (multi-shard)

vast.ai 必須の implementation-pending モデル (§3.1 sign-off + wiring 両方要):
- Kimi-Audio-7B (BF16 fleet skeleton、CLI dispatch wiring 要 + §3.1 blank)

owner 判断待ちの vast.ai-scale モデル: なし (§3.1 で fail-closed 済分)。

## 7. Phase B 自動化 script 一覧 (2026-07-28)

| Path | 実行場所 | 役割 |
|---|---|---|
| `scripts/publish/check-model-size.sh` | local | HF API で size 判定、`LOCAL_SAFE / LOCAL_OK / LOCAL_BORDERLINE / VAST_AI_REQUIRED` verdict |
| `scripts/publish/publish-one.sh` (**gate 7 追加**) | どこでも | GGUF publish の 5-gate chain。8 GiB 超で fail-closed、`VOKRA_PUBLISH_ON_VAST=1` or `--allow-large` で bypass |
| `scripts/publish/vast-ai/provision.sh` | vast.ai | Rust/uv/Python 3.12/hf-transfer/repo/vokra-cli を idempotent に install、shell rc に marker export |
| `scripts/publish/vast-ai/run-one.sh` | vast.ai | 1 モデル分の DL + convert + publish chain (`--push` で本番) |

各 script は `--self-test` を持つ (pure、network fetch 無し)。CI で回すのは `check-model-size.sh` と `publish-one.sh` の self-test (vast.ai script は vast.ai 前提ゆえ CI 側の自動化対象外)。

**Phase C (将来 candidate)**: local から `vastai` CLI (~/.local/bin/vastai、1.1.3 install 済) 経由で instance lifecycle まで自動化する orchestrator。owner が 1 command で instance rent → provision → run-one → destroy まで完結する形。現時点では owner が instance lifecycle を握る Phase B 止まり。

## 関連

- memory [[feedback-large-models-on-vast-ai]] (方針)
- memory [[project-huggingface-vokra-publication]] (5-gate publish)
- memory [[project-restamp-provenance]] (低メモリ再刻印は本機で可、convert 本体は別)
- `docs/m5-owner-verification-checklist.md` §6.9 (Wave 3 publish sign-off queue)
- `docs/license-audit.md` §3.1 (row-per-model sign-off state)
