# vast.ai runbook — 大きめ音声モデルの convert + publish

Tracked / public。**2026-07-28** に本 M1 iMac (16GB RAM) 上で Voxtral-Small-24B-2507 (48GB BF16、11 shards) を convert 試行中、`SafetensorsFile::open` が全 shard を mmap した結果 swap 40GB used に到達 = OS レベル force shutdown リスクが実証された。初期の 8GB heuristic は、2026-08-16 の owner 指示で **合計 2GB 以上の model artefact は VAST**へ引き下げられた。workspace 全体と `vokra-models` の Cargo も VAST-only。本 runbook は現在の厳しい運用境界を正本とし、後段の旧 size classifier は履歴/コスト見積りとしてのみ残す。

**2026-08-17 operational override**: model conversion, real-weight verification, and upload work that can materially consume memory run on **vast.ai only**. Do not download a checkpoint, convert it, verify it against real weights, or upload its result from the M1 iMac merely because the historical size heuristic below returns `LOCAL_SAFE` / `LOCAL_OK`. The Mac may perform checkpoint-free work only: source/doc review, static tests, and the HF-metadata size preflight. This override supersedes older local-conversion suggestions in this and per-model handoffs.

**2026-08-18 Voxtral-Small-24B corrected dry-run + parity evidence**: VAST instance `47955178` converted pinned upstream commit `da5b42409f279fdd92febee0511a6c32828569c1` through `run-voxtral-small-24b.sh` without an HF credential. A first provenance-only artifact was rejected before upload because its adapter metadata was absent. The corrected tracked side-car run produced 852 tensors / 54 metadata keys / 851 BF16 exact passthrough / 0 skipped / tokenizer embedded / active `frame_stack_mlp` adapter / 48,542,409,248 bytes / SHA-256 `91f2733492dd49b8e8f810192c77538d7d6d2f4c1c568098e11c3ad91f752c87`; converter peak RSS = 1,780.18 MiB. Header source is `mistralai/Voxtral-Small-24B-2507 (Apache-2.0)`, and model-card, owner sign-off, LICENSE, NOTICE, SOURCE and dry-run gates all pass. Independent upstream reference generation peaked at 130.43 GiB. Rust real-runtime results with unchanged bounds: mel `1.311e-5`, encoder `2.956e-5`, projector `1.812e-5`, logits `6.356e-4`, plus exact 27-id greedy/EOS match in 5,292.21 s. Bounded fixtures live in `tests/parity/voxtral-small-24b-2507/`; their VAST fixture-only smoke passed 4/4 after commit `7640a02`. The instance is stopped (CLI status `exited`, disk retained) with the corrected GGUF staged. Resume only for explicitly authorized credential transfer + `--push`, then live-verify and destroy.

memory [[feedback-large-models-on-vast-ai]] の運用側詳細版。

## 0. TL;DR — 自動化 pipeline (Phase B, 2026-07-28)

**判定**: `scripts/publish/check-model-size.sh <hf-repo>` を local で走らせて `LOCAL_SAFE / LOCAL_OK / LOCAL_BORDERLINE / VAST_AI_REQUIRED` の verdict を確認する。この preflight は checkpoint を取得しないメタデータ照会だけに限る。convert / real-weight verify / upload は verdict にかかわらず上記 override に従い vast.ai で行う。

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
# Voxtral-Small-24B (48 GB、必ず vast.ai)。revision / shard-only include /
# tokenizer / expected provenance は専用 wrapper で固定される。
~/vokra/scripts/publish/vast-ai/run-voxtral-small-24b.sh
```

上記 dry-run が green になっても upload 権限は自動では生じない。依頼者がこの artifact/repo の upload を明示承認した場合だけ、同じ command に `--push` を追加する。

`run-one.sh` は `HF snapshot_download (hf-transfer) → autodetect input → vokra-cli convert → GGUF header verification → publish-one.sh` を chain。`--push` を外せば dry-run stage のみ。T4 (非商用) は `--allow-noncommercial`、T3 (Copyleft) は `--acknowledge-copyleft`。詳細は `run-one.sh --help`。Voxtral-Small-24B は `run-voxtral-small-24b.sh` を使い、upstream commit `da5b42409f279fdd92febee0511a6c32828569c1` と 11 shard のみを固定する（同 repo の duplicate `consolidated.safetensors` は取得しない）。

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
(48GB) でも peak `max(shard_header) + max(tensor_payload)` のフットプリントに
抑える。2026-08-18 に adapter side-car も同じ streaming path へ統合し、VASTで
peak 1,780.18 MiB を実測した。**この技術的な低メモリ性はlocal実行の許可ではない**。
上記operational overrideどおり、実weight変換・検証・uploadはサイズを問わずVASTで
行う。**K-quant はスコープ外** (widen-then-quantizeがin-memory必須)。

**Historical borderline classifier (現在は VAST、local 実行許可ではない)**:
- 5-8GB の safetensors: kyutai-stt (5.23GB BF16、2026-07-28 実績あり)、csm-1b (6.21GB single-file、実績あり)、moshiko-7b (~14GB BF16、既 published)
- 判定基準: `curl -sL "https://huggingface.co/api/models/<repo>?blobs=true"` で最大 shard サイズ + shards 数から推定していた。現在は合計 2GB 以上なら単一 shard でも VAST。

**Historical safe-size list (現在は preflight 参考のみ)**:
- ≤4GB の safetensors全般 (whisper-* / kokoro / silero / piper / dfn3 / utmos / dac / mimi / parakeet-tdt / parakeet-ctc / distil-whisper / kotoba-whisper / dia / qwen3-tts / vibevoice-1.5b / voxcpm-0.5b / cosyvoice2-0.5b / chatterbox-* / irodori / zonos / xcodec2)

## 2. vast.ai 手順 (Voxtral-Small-24B-2507 を例に)

### 2.1 事前準備 (本機 agent で実施可)

- **§3.1 sign-off 確認**: `docs/license-audit.md` §3.1 で該当 row が signed。Voxtral-Small-24B は row 250 = ☑ Commercial 2026-07-23 yousan 済。
- **HF token 確認**: `HF_TOKEN` (or `HF`) を本機の `.env` から取得 → vast.ai インスタンスに `export` で渡す。
- **branch 確認**: vast.ai には current `main` を clone。未 push commit が必要なら remote branch を増やさず `git bundle` で渡し、手元と VAST の HEAD 一致を確認する。

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
export HF="$HF_TOKEN"
```

### 2.4 vokra-cli release build (~5 min)

```bash
cd ~/vokra
cargo build --release -p vokra-cli
# バイナリ = target/release/vokra-cli
```

### 2.5 upstream weight DL + convert + publish

Voxtral-Small-24B-2507 の推奨経路:

```bash
cd ~/vokra

# dry-run: convert 後に model/source/tokenizer/tensor count と全 publish gate を検証
./scripts/publish/vast-ai/run-voxtral-small-24b.sh

# dry-run 後、依頼者がこの upload を明示承認した場合だけ
./scripts/publish/vast-ai/run-voxtral-small-24b.sh --push
```

専用 wrapper は exact upstream revision、`model-*.safetensors` + index、`config.json`、`tekken.json` を固定する。`consolidated.safetensors` は取得しないため、upstream weight の 48 GB 重複 download を避ける。`--push` invocation 自体も publish 前に必ず dry-run gate を通る。

以下は自動化が利用できない場合だけの手動 fallback:

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
# ↑ dry-run 全 gate 通過 + exact upload の依頼者明示承認後だけ ↓
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/voxtral-small-24b-2507/model.gguf \
  --repo vokra/voxtral-small-24b-2507 \
  --license-spdx apache-2.0 --push

# 検証
curl -sI https://huggingface.co/vokra/voxtral-small-24b-2507 | head -1
# HTTP/2 200 が返れば live
```

### 2.6 instance destroy (billing 抑制)

vast.ai UI から即 destroy、または `scripts/publish/vast-ai/vastai-safe.sh destroy instance <instance-id>` (CLI 使用時)。dry-run/evidence だけなら完了後すぐ destroy。upload が明示承認された場合は **upload 完了 → live 確認 → destroy** の順で、GGUF は remote に残さない。ローカルから Vast CLI を呼ぶ場合は必ず `vastai-safe.sh` を経由すること。stdout/stderr に誤って出る URL クエリの `api_key` 等を `[REDACTED]` に置換し、CLI の終了コードは保持する。

## 3. int tensor 対応 (parakeet 系で発生した pattern)

一部 checkpoint に `num_batches_tracked` (BatchNorm training-only int64 counter) 等の inference-inert int tensor が入っている。Vokra converter は F32/F16/BF16 のみ受け付けるため、**convert 前に strip する**:

```bash
# tools/parity/strip_int_tensors.py で inference-inert int tensor を除去
uv run --no-project --python 3.12 python ~/vokra/tools/parity/strip_int_tensors.py \
  --input  "$SNAP/model.safetensors" \
  --output /root/scratchpad/staging/<model>/model.stripped.safetensors
# manifest sidecar (.stripped-manifest.json) が dropped tensor 一覧を記録
```

Voxtral-Small-24B は全 tensor が BF16 = strip 不要 (2026-07-28 事前確認済)。

## 4. 事前サイズ確認 command

artefact の規模と必要 disk/RAM を見積もる preflight は事前に行う。現在の実作業 routing は合計 2GB 以上で VAST であり、旧 label は local 実行許可ではない:

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
curl -sL "https://huggingface.co/api/models/<repo>?blobs=true" | uv run --no-project --python 3.12 python -c "
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

**Historical script verdict** (`check-model-size.sh` の出力互換。現在の 2GB VAST 運用を上書きしない):

- **≤4 GiB total**: script label `LOCAL_SAFE` — size metadata label only。
- **4-8 GiB total, max shard ≤6 GiB**: script label `LOCAL_OK` — size metadata label only。
- **8-16 GiB or shards ≥5**: script label `LOCAL_BORDERLINE`。
- **>16 GiB**: script label `VAST_AI_REQUIRED`。

Operational verdict: **合計 2 GiB 以上は全 label で VAST**。将来 script threshold を変更するまでは、Codex/Claude memory guard と本 runbook がより厳しい上位規則になる。

## 5. Owner action

- **HF_TOKEN の vast.ai instance への手渡し**: 本機 `.env` の値を SSH セッションの環境変数で渡す。CLI 引数や output に表示しない。instance 破棄は永続化を減らすだけで、既に terminal/log に出た secret を無効化しないため、その場合は即 rotate する。
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
| `scripts/publish/vast-ai/vastai-safe.sh` | local | Vast CLI の stdout/stderr を redaction し、終了コードを保持する wrapper |
| `scripts/publish/vast-ai/test-vastai-safe.sh` | local | ネットワークなしの wrapper 契約テスト |

各 script は `--self-test` を持つ (pure、network fetch 無し)。CI で回すのは `check-model-size.sh` と `publish-one.sh` の self-test (vast.ai script は vast.ai 前提ゆえ CI 側の自動化対象外)。

**Phase C (将来 candidate)**: local から `vastai` CLI (~/.local/bin/vastai、1.1.3 install 済) 経由で instance lifecycle まで自動化する orchestrator。owner が 1 command で instance rent → provision → run-one → destroy まで完結する形。現時点では owner が instance lifecycle を握る Phase B 止まり。

## 5. TIER 1+2 audio-gap defer markers (2026-07-30 追加)

依頼者 2026-07-30 指示「大きいモデルは vast.ai で変換, アップロード」を受
けた TIER 1+2 impl workflow (`wf_022575ce-077`) で defer marker として
`ModelKind` に登録した 3 モデルの vast.ai publish runbook。CLI wiring は
`2556b4a` で land 済、`vokra-cli convert --model <name>` は callable だが
実 publish は vast.ai 経由 owner。

**2026-07-30 status update**:

| Model | Status | Notes |
|---|---|---|
| **Nemotron-3.5-ASR-Streaming-0.6B** | **✅ Published** | `openmdw-1.1` = Permissive (CC ADR 2026-07-30 primary-source 照合)。GGUF 2.55 GB を local M1 で convert + push、`fetch_license.sh` に openmdw-1.1 inline text 追加 (canonical URL 無しゆえ)。Live at https://huggingface.co/vokra/nemotron-3.5-asr-streaming-0.6b |
| **Voxtral-Mini-4B-Realtime-2602** | **📦 vast.ai queue** | 8.25 GB safetensors × 2 = 16.5 GB total、gate 7 refuse。§5.1 runbook 参照 |
| **Cohere-Transcribe-03-2026** | **🚪 owner HF gate accept 待ち** | `gated=auto`、owner が HF UI で "Access repository" を要クリック。§5.2 runbook 参照 |


### 5.1 `mistralai/Voxtral-Mini-4B-Realtime-2602` (~8 GB BF16、apache-2.0)

**Verdict**: historical label は BORDERLINE だが、現在の 2GB rule では
**VAST required**。M1 iMac 16 GB では実行しない。

**Vokra ModelKind**: `VoxtralMiniRealtime` (`--model voxtral-mini-realtime`)。
converter は既 land、Voxtral (Mistral) 家族と同じ `models::voxtral::convert`
呼び出し (streaming 経路対応)。

**License**: apache-2.0 (HF cardData primary source 2026-07-30 CC 直接照合)。
`docs/license-audit.md` §3.1 で 2026-07-30 yousan (依頼者委任 = CC 判断)
☑ Commercial sign 対象。

**Runbook**:
```bash
# vast.ai instance (§2.2 と同じ specs = 64 GB RAM / 200 GB disk 推奨)
# provision.sh 完了後、以下 1 コマンド:
~/vokra/scripts/publish/vast-ai/run-one.sh \
  --hf-repo mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --vokra-slug voxtral-mini-4b-realtime-2602 \
  --model-kind voxtral \
  --license-spdx apache-2.0
```

dry-run 後、依頼者が exact artifact/repo の upload を明示承認した場合だけ `--push` を追加する。

**推定コスト**: ~1-1.5h wall-clock × $0.3-0.5/hr = **$0.3-0.75**。

### 5.2 `CohereLabs/cohere-transcribe-03-2026` (~1 GB but gated=auto、apache-2.0)

**Verdict**: historical size label は SAFE (~1 GB) だが **HF gate accept 要
(gated=auto)**。現在の model-work override に従って変換/検証/upload は VAST。
owner の HF UI accept は 1 回のみで、以降 authenticated fetch 可。

**Vokra ModelKind**: `CohereTranscribe` (`--model cohere-transcribe` /
`cohere-transcribe-03-2026`)。

**License**: apache-2.0 (HF cardData primary source 2026-07-30、gated=auto
は access control のみで追加条項なし)。

**Runbook**:
```bash
# 1. owner が HF UI で https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 の
#    "Access repository" ボタンを一度クリック (非拘束 advisory の accept)
# 2. HF_TOKEN を export
export HF_TOKEN='hf_xxxxxx'
# 3. current model-work override に従い vast.ai instance で dry-run:
~/vokra/scripts/publish/vast-ai/run-one.sh \
  --hf-repo CohereLabs/cohere-transcribe-03-2026 \
  --vokra-slug cohere-transcribe-03-2026 \
  --model-kind cohere-transcribe \
  --license-spdx apache-2.0
```

upload は runtime readiness と dry-run を再確認し、依頼者が exact artifact/repo を明示承認した場合だけ `--push` を追加する。

**注意**: `cohere-transcribe` は新規 ModelKind (2026-07-30 CLI wiring commit
`2556b4a` で追加)。converter dispatch は library-callable だが実 forward /
runtime は未実装 (converter skeleton = BF16 pass-through のみ)。**owner が
publish しても消費側 runtime forward が未実装ゆえ、実 ASR には使えない**
= 実 publish は「converter 存在の公表」以上の value は現時点で薄い、runtime
forward 実装完了後に publish 推奨。

### 5.3 `nvidia/nemotron-3.5-asr-streaming-0.6b` — ✅ Published 2026-07-30

**Verdict**: **CC ADR 完了 (2026-07-30)**。HF cardData primary source =
`license: "other"` / `license_name: "openmdw-1.1"` /
`license_link: https://openmdw.ai/license/1-1/`。openmdw.ai/license/1-1/ を
CC 直接照合、**OpenMDW-1.1 = Permissive MIT-analog for ML weights** と判定
(commercial 可 / redistribution 可 = 要 existing notice 保持 / no share-
alike / no non-commercial restriction / attribution = notice 保持のみ =
Apache-2.0 と同 tier)。

**判定反映済**:

- `crates/vokra-core/src/compliance/license_class.rs`: `PERMISSIVE_TOKENS`
  に `openmdw` token 追加 (8→9)、`registry_lookup` に
  `_ if id.starts_with("nemotron-asr")` → `LicenseClass::Permissive` walk
  追加
- `crates/vokra-convert/src/models/nemotron_asr.rs`: 新規 converter
  (`convert_nemotron_asr_file`、BF16 pass-through、`wespeaker.rs` mirror、
  ARCH="nemotron_asr_streaming"、CATEGORY="asr")
- `crates/vokra-convert/src/lib.rs`: dispatch を defer marker error から
  実 converter 呼び出しに flip、`convert_file_with_slug` から call
- `scripts/publish/fetch_license.sh`: `openmdw-1.1` は canonical plain-text
  URL が無い (openmdw.ai は HTML only、Linux Foundation の SPDX list 未登
  録) ゆえ `inline_license_text()` fn を新設、verbatim OpenMDW-1.1 text を
  script に埋め込み。`--spdx openmdw-1.1` で inline text を LICENSE として
  write。redistribution §D 要 "a copy of this agreement" 保持 = inline は
  最短経路 (canonical URL が出れば `canonical_url()` に移設)
- `docs/license-audit.md`: Nemotron row → ☑ Commercial 2026-07-30 yousan
  (CC 判断)

**Published GGUF**: 2.55 GB (BF16 pass-through、549 tensor)、live at
https://huggingface.co/vokra/nemotron-3.5-asr-streaming-0.6b (2026-07-30
push 完了、README + LICENSE + NOTICE + SOURCE.md 同梱、`vokra.model.arch =
"nemotron_asr_streaming"` / `vokra.provenance.upstream_hf =
"nvidia/nemotron-3.5-asr-streaming-0.6b"`)。

**Runtime forward = 未実装**: converter skeleton は BF16 pass-through で
tensor 名は upstream verbatim 保持。実 ASR forward は Nemotron-3.5 arch
(NVIDIA 独自 streaming Conformer variant、FastConformer sibling ではない)
を `crates/vokra-models/src/nemotron_asr/` に future wave で native 実装、
tensor manifest は published GGUF から読める。ゆえに現状 publish は
"converter 存在の公表 + weight 二次配布 + license 判定" の value のみ、
runtime forward 実装後に消費可能。

## 6. 残 2 model 共通の事前 owner タスク

Nemotron は 2026-07-30 に completed (§5.3)、残 2 モデル用:

1. **HF token** の export (`export HF_TOKEN=hf_xxx`) — vast.ai instance
   destroy で消えるため接続毎に再 export
2. **§3.1 sign-off** — Voxtral-Realtime = ☑ Commercial 2026-07-30 yousan
   (2556b4a と同 wave の row 追加、docs/handoff で follow up)、
   Cohere-transcribe = ☑ Commercial 同、Nemotron-ASR = ☑ Commercial
   2026-07-30 yousan (agent ADR 完了、§5.3)
3. **branch 確認** — vast.ai instance は current `main` を clone。未 push delta が必要なら `git bundle` で渡し、旧 long-lived branch は clone/push/merge しない
4. **cost 見込確認** — 残 2 モデル合計 ~$0.7-1.3 (Voxtral 8GB × 1 +
   Cohere 1GB × 1、~2-3h wall-clock)
5. **Cohere HF gate accept** — owner が browser で
   https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 を開き
   "Access repository" をクリック。以降 fetch 可能 (`gated=auto` は非拘束
   advisory accept のみ)。agent 側では accept できないため必須 owner task

## 関連

- memory [[feedback-large-models-on-vast-ai]] (方針)
- memory [[project-huggingface-vokra-publication]] (5-gate publish)
- memory [[project-restamp-provenance]] (低メモリ再刻印は本機で可、convert 本体は別)
- `docs/m5-owner-verification-checklist.md` §6.9 (Wave 3 publish sign-off queue)
- `docs/license-audit.md` §3.1 (row-per-model sign-off state)
- `docs/handoff/tier1-tier2-audio-impl-2026-07-30.md` (TIER 1+2 land、defer markers)
