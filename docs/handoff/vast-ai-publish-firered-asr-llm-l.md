# vast.ai publish runbook — FireRedASR-LLM-L (FireRedTeam/FireRedASR-LLM-L)

**Owner-triggered.** CC は本 doc 作成のみ。実 vast.ai instance の起動・convert・
publish は owner が本 runbook を追いながら実行する。

**Related**:
- 本 runbook は `docs/handoff/vast-ai-large-model-publish.md`（総論 = §2 recipe /
  §3 provision.sh gotcha / §4 lifecycle）を **前提** とする。共通手順は総論を参照
  し、本 doc は FireRedASR-LLM-L に固有の差分のみを記述する。
- Converter code: `crates/vokra-convert/src/models/firered_asr_llm_l.rs`
  (Wave B fast-track land, coverage-audit-2026-08-03、commit `cae8fcd`)。
- Sidecar: `tools/parity/firered_asr_llm_l/prepare_checkpoint.py` (uv-managed
  Python 3.12) — **§0.2 参照: 現状 sharded safetensors 前提、upstream 実際は
  `.pth.tar` ゆえ owner 側で bridge 追加が必要**。
- Sibling published: `huggingface.co/vokra/nemotron-3.5-asr-streaming-0.6b`
  (2026-07-30 published、openmdw-1.1 Permissive)。
- Sibling code-only: `crates/vokra-convert/src/models/canary_qwen.rs`
  (Canary FastConformer + Voxtral-style Qwen decoder、同 "encoder + adapter + LLM
  decoder" mold の precedent)。
- Sibling code-only: `crates/vokra-convert/src/models/firered_asr_aed_l.rs`
  (Whisper-topology AED、FireRedASR family の別 variant、~2.2 GB)。

## 0. Primary source verify + 実 upstream 差分の記録 (2026-08-13、CC 直接照合)

### 0.1 License — Apache-2.0 確認済 ✓

```bash
curl -sL https://huggingface.co/api/models/FireRedTeam/FireRedASR-LLM-L | \
  python3 -c "import json,sys; d=json.load(sys.stdin); cd=d.get('cardData',{}); \
    print('license:', cd.get('license')); \
    print('language:', cd.get('language')); \
    print('tags:', cd.get('tags')); \
    print('gated:', d.get('gated'))"
```

出力:
```
license: apache-2.0
language: ['en', 'zh']
tags: ['audio', 'automatic-speech-recognition', 'asr']
gated: False
```

README frontmatter (`https://huggingface.co/FireRedTeam/FireRedASR-LLM-L/raw/main/README.md`
先頭):
```yaml
---
license: apache-2.0
language:
  - en
  - zh
tags:
- audio
- automatic-speech-recognition
- asr
---
```

**判定**: `LicenseClass::Permissive`、`redistributable() = true`、gate 2 pass。
`docs/license-audit.md` §3.1 owner sign-off 対象 (☑ Commercial 見込)。

### 0.2 Upstream file format — `.pth.tar` (NOT sharded safetensors) — sidecar 側修正必要

**現在の HF listing** (2026-08-13、CC 直接 fetch):

```bash
curl -sL "https://huggingface.co/api/models/FireRedTeam/FireRedASR-LLM-L?blobs=true" | \
  python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d.get('siblings', []):
    print(f'  {s.get(\"size\", 0):>14,} {s.get(\"rfilename\", \"\")}')"
```

出力:
```
           1,519 .gitattributes
           6,770 README.md
           1,408 asr_encoder.pth.tar
           1,311 cmvn.ark
           2,985 cmvn.txt
               0 config.yaml
   3,627,720,250 model.pth.tar
```

**発見された差分** (audit ticket / converter code の想定と upstream 実際の差):

| 項目 | audit ticket / converter code の想定 | upstream 実際 (2026-08-13) |
|---|---|---|
| Weight format | **sharded safetensors** (`model-*.safetensors` + `model.safetensors.index.json`) | **`model.pth.tar` 単一** (PyTorch tar archive、3.38 GiB) |
| Total size | ~16.6 GB BF16 | 3.38 GiB `model.pth.tar` 単体 (asr_encoder は 1.4 KB marker のみ) |
| Config | `config.json` (Hugging Face 標準) | `config.yaml` **0 bytes 空ファイル** — 実 config は GitHub 側 (§0.3 参照) |
| Feature 前処理 | (converter 側 assumption なし) | Kaldi CMVN (`cmvn.ark` 1311 bytes / `cmvn.txt` 2985 bytes) — 実 forward で必要 |

**"~16.6 GB" の内訳**: README §Usage で明示されている通り、**FireRedASR-LLM-L は
Qwen2-7B-Instruct を別途 download 前提**:

> If you want to use `FireRedASR-LLM-L`, you also need to download
> [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and place
> it in the folder `pretrained_models`. Then, go to folder `FireRedASR-LLM-L`
> and run `$ ln -s ../Qwen2-7B-Instruct`

FireRedASR-LLM-L の 8.3B params の内訳 = **Conformer encoder (~1.3B、`model.pth.tar`
に in) + adapter (small、in) + Qwen2-7B-Instruct LM decoder (~7B、別 repo)**。
実 runtime footprint = 3.38 GiB (asr side) + ~15 GB (Qwen2-7B BF16) = ~18-19 GB。
task 指示の "~16.6 GB" は combined 概算。

### 0.3 Config の入手先 — GitHub 側

`config.yaml` が 0 bytes ゆえ、実 config は `github.com/FireRedTeam/FireRedASR`
リポジトリを clone して取得する必要がある。converter が `--config` パラメータで
参照する schema は Hugging Face 標準 `config.json` を assume していない可能性 —
converter code header の `config` 期待値を確認 (audit ticket 起票時点で決めた
schema があるはず):

```bash
# converter が期待する config schema を確認 (owner が sanity check)
grep -A 20 "fn convert_firered_asr_llm_l_file" \
  ~/vokra/crates/vokra-convert/src/models/firered_asr_llm_l.rs | head -40
```

owner が upstream `github.com/FireRedTeam/FireRedASR` の `pretrained_models/FireRedASR-LLM-L/`
配下の実 config (yaml / json / python module) を目視で確認し、Vokra converter の
`--config` に渡す JSON を手で組む工程が現時点で必要になる。

### 0.4 Sidecar `.pth.tar` → safetensors bridge が現状未実装

現状の `tools/parity/firered_asr_llm_l/prepare_checkpoint.py` は **sharded
safetensors 前提の merge/dedup/strip 骨組み** で、upstream `.pth.tar` の
extraction には対応していない (converter code header docstring の
"sharded safetensors" 記述と一致している)。owner 側で:

1. **`.pth.tar` → 個別 tensor extraction** (Python `tarfile` + `torch.load` で
   `state_dict` を復元、mirror of `tools/parity/nemo_pt_to_safetensors.py` = M4-20
   T17 DFN3 Phase B の bridging pattern precedent)
2. **`safetensors.torch.save_file` で flat safetensors 化** (dedup + strip は
   既存 sidecar の logic を流用可能)
3. Vokra converter は既存の flat safetensors 経路で consume

の 3 step が必要。上記 (1)(2) を既存 sidecar に統合する PR (owner が local で
書く、CC は本 doc の handoff まで) が事前に land されるまで、vast.ai 上の
`run-one.sh` は途中 fail する可能性が高い。

**Precedent bridge**: `tools/parity/nemo_pt_to_safetensors.py` (NeMo `.nemo` archive
→ safetensors、M4-20 T17 DFN3 pattern) — 同様の tarball extraction pattern を
mirror できる。実装は owner Python 側 (uv-managed 3.12、`safetensors[torch]` +
`torch` の pin は既存 `tools/parity/firered_asr_llm_l/pyproject.toml` に in)。

## 1. モデル情報 (primary source 照合後の実値)

| 項目 | 値 |
|---|---|
| Upstream HF repo | `FireRedTeam/FireRedASR-LLM-L` |
| Upstream HF URL | https://huggingface.co/FireRedTeam/FireRedASR-LLM-L |
| Upstream sha | `9837461f78d15ee66565d00aaec0bc5497d7fbc1` (2026-08-13 API 取得、`lastModified: 2025-03-05T11:44:58.000Z`) |
| License (upstream code) | apache-2.0 (README frontmatter + HF cardData 両方確認) |
| License (upstream weight) | apache-2.0 (同上) |
| SPDX (Vokra 判定) | `apache-2.0` (`LicenseClass::Permissive`) |
| Total footprint | **3.38 GiB `model.pth.tar`** (asr side) **+ ~15 GB Qwen2-7B-Instruct** (別途 DL、README §Usage 参照) = 実 runtime ~18-19 GB |
| 判定 (`check-model-size.sh`) — asr side only | `LOCAL_OK` (asr 単体 3.38 GiB は 4-8 GiB range 下端) |
| 判定 — Qwen2-7B-Instruct 含めた full runtime | `LOCAL_BORDERLINE` — combined 18-19 GB は vast.ai 推奨 |
| Vokra ModelKind | `FireredAsrLlmL` (`--model firered-asr-llm-l` / `firered_asr_llm_l` / `fireredasr-llm-l` 等 15 alias) |
| Arch tag | `vokra.model.arch = "firered_asr_llm_l"` (sibling `firered_asr_aed_l` (AED variant) と区別) |
| Category tag | `vokra.model.category = "asr"` (sibling Canary-Qwen / Voxtral / Whisper family と同 tier) |
| Vokra HF slug | `vokra/firered-asr-llm-l` |
| Attribution 要求 | apache-2.0 標準の LICENSE + NOTICE 同梱のみ、runtime-side 追加なし |
| Non-commercial 制限 | なし |
| Share-alike | なし |
| Language coverage | 英語 + 中国語 (README §Evaluation aishell1 / aishell2 / WenetSpeech / KeSpeech / LibriSpeech 実績) |
| Architecture | Conformer encoder + linear/MLP audio-to-text adapter + Qwen2-7B LM decoder ("Encoder-Adapter-LLM framework" per README §Method) |
| Sibling variants | FireRedASR-AED-L (`firered_asr_aed_l`、Whisper-topology AED、~2.2 GB、`fireredteam/FireRedASR-AED-L`) |

### Primary source verify command (本 doc land 時点で実行済)

```bash
# License verification (§0.1 参照)
curl -sL https://huggingface.co/api/models/FireRedTeam/FireRedASR-LLM-L | \
  python3 -c "import json,sys; d=json.load(sys.stdin); cd=d.get('cardData',{}); \
    print('license:', cd.get('license')); print('language:', cd.get('language'))"
# → license: apache-2.0 / language: ['en', 'zh']

# File manifest (§0.2 参照)
curl -sL "https://huggingface.co/api/models/FireRedTeam/FireRedASR-LLM-L?blobs=true" | \
  python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d.get('siblings', []):
    print(f'  {s.get(\"size\", 0):>14,} {s.get(\"rfilename\", \"\")}')"

# README §Usage の Qwen2-7B-Instruct 依存確認
curl -sL "https://huggingface.co/FireRedTeam/FireRedASR-LLM-L/raw/main/README.md" | \
  grep -A 3 "Qwen2-7B-Instruct"

# Size verification (asr side only)
./scripts/publish/check-model-size.sh FireRedTeam/FireRedASR-LLM-L
# expected verdict: LOCAL_OK (3.38 GiB)
```

## 2. vast.ai instance recipe

**共通仕様**は `docs/handoff/vast-ai-large-model-publish.md` §2.2 を参照。
FireRedASR-LLM-L 固有の値 (Qwen2-7B-Instruct 統合込み):

| 項目 | 推奨値 | 備考 |
|---|---|---|
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` or `nvidia/cuda:13.0.0-*` | 総論 §2.2 と同じ。`nvidia/cuda:13.0.0` 系は provision.sh gotcha §3 参照 |
| RAM | **64 GB 以上** (総論と同じ) | asr side 3.38 GB + Qwen2-7B-Instruct BF16 ~15 GB を mmap し合流して single GGUF 化する working set + upload buffer |
| Disk | **100 GB 以上** | 上流 DL 3.38 GB (asr) + 15 GB (Qwen2-7B) + prepare_checkpoint 中間 safetensors (extracted from .pth.tar) 3.38 GB + GGUF 18-19 GB + HF cache buffer + 余裕 |
| GPU | Convert には不要 (converter は CPU only) | vast.ai は GPU 前提販売ゆえ最安 GPU を選ぶ |
| Network | 非従量課金 or inclusive band | 上下 ~40 GB out-bound (DL 3.38 + 15 GB = 18 GB in + upload 18-19 GB) |
| 課金見込 | ~2-3h × $0.3-0.5/hr = **$0.6-1.5** | Voxtral-Small-24B ($0.6-1.0) 級、Qwen2-7B DL に時間を要する |

## 3. provision.sh gotcha (総論 §3 を参照)

`scripts/publish/vast-ai/provision.sh` は下記 4 件を idempotent に修正済。
FireRedASR-LLM-L convert 前に一度だけ実行:

| Gotcha | 起因 | provision.sh の対応 |
|---|---|---|
| **hf_config.pth shim** | `nvidia/cuda:13.0.0` image が仕込む Python startup shim が `HF_ENDPOINT` を malicious mirror `117.175.104.83:8081` に上書き | shim 除去 + certifi CA 再植え付け (memory `[[reference-vast-ai-hf-config-pth-shim]]`) |
| **huggingface_hub < 0.30 pin** | 1.x xet-token routing が mirror 404 を投げ、`HF_HUB_DISABLE_XET` も一部 bypass、0.30+ `resume_download` deprecated で flaky egress を落とす | vast.ai 上のみ `huggingface_hub < 0.30` に pin (memory `[[reference-huggingface-hub-lt-030-vast-ai]]`) |
| **certifi CA bundle** | 空 or 古い CA bundle で HTTPS 検証失敗 | `certifi` 再 install + `SSL_CERT_FILE` export |
| **stack tool install (torch/numpy/safetensors)** | resilient_batch.sh の uv fallback / ad-hoc `python3 -c` が消費するが system 層に無い、加えて `.pth.tar` extraction では `torch` が必須 (未 install だと fail) | provision.sh Wave 12 で pre-install |

```bash
# SSH 接続後、まず HF token を export (instance destroy で消える)
export HF_TOKEN='hf_xxxxxx'   # 本機 .env の HF= 値をここに貼る

# 1 コマンドで Rust + uv + Python 3.12 + hf-transfer + repo + vokra-cli build まで完了
curl -sSL https://raw.githubusercontent.com/ayutaz/vokra/main/scripts/publish/vast-ai/provision.sh | bash
source ~/.bashrc  # VOKRA_PUBLISH_ON_VAST=1 marker を pick up
```

## 4. Convert + publish command

### 4.0 前提: `.pth.tar` → safetensors bridge を sidecar に追加

**現状 (2026-08-13)**: §0.4 の通り `tools/parity/firered_asr_llm_l/prepare_checkpoint.py`
は sharded safetensors 前提。upstream 実際は `model.pth.tar` (PyTorch tar
archive) ゆえ、owner 側で extraction step を追加する必要がある。

**Owner action** (§4 実行の事前 dependency):

1. `tools/parity/nemo_pt_to_safetensors.py` の tarball extraction pattern を
   参考に、`tools/parity/firered_asr_llm_l/prepare_checkpoint.py` に
   `.pth.tar` 入力対応 subcommand or `--input-format pth-tar` 分岐を追加
2. Extraction logic 概略 (Python):
   ```python
   import tarfile, io, torch
   with tarfile.open(pth_tar_path, 'r') as tf:
       for member in tf.getmembers():
           if member.name.endswith('.pth') or member.name.endswith('.pt'):
               f = tf.extractfile(member)
               state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu',
                                       weights_only=True)  # PyTorch 2.4+ safety
               # dedup / strip / bf16 cast は既存 sidecar logic を流用
   ```
3. Qwen2-7B-Instruct 側は upstream 標準の sharded safetensors ゆえ既存 merge
   logic がそのまま使える
4. asr side (extracted) + qwen2 side (merged) を single flat safetensors に合流
   (fireredasr の adapter が両者を橋渡しする arch ゆえ tensor namespace 衝突
   なし、README §Method の "Encoder-Adapter-LLM framework" 記述と一致)

**Bridge が land されるまで**: §4.1 の run-one.sh は sidecar 段で fail する。
Bridge land 後に再開。

### 4.1 自動化 pipeline (推奨、Phase B、**bridge land 後**)

```bash
# provision.sh 完了後、以下 1 コマンド (bridge land 後に有効)
~/vokra/scripts/publish/vast-ai/run-one.sh \
  --hf-repo FireRedTeam/FireRedASR-LLM-L \
  --vokra-slug firered-asr-llm-l \
  --model-kind firered-asr-llm-l \
  --license-spdx apache-2.0 \
  --push
```

**注意**: `--push` を外せば dry-run stage のみ (5-gate verify + gate 7 大サイズ
check)。本番 upload 前に必ず dry-run で全 gate 通過を確認すること (総論 §2.5 と
同じ規律)。gate 7 (>8 GiB fail-closed) は combined GGUF ~18-19 GB で hit するが
`VOKRA_PUBLISH_ON_VAST=1` (provision.sh が set) で auto-bypass。

### 4.2 手動 fallback (総論 §2.5 に準拠、bridge land 後)

```bash
mkdir -p ~/scratchpad/hf-cache ~/scratchpad/staging/firered-asr-llm-l

cd ~/vokra/tools/parity
uv sync   # pyproject.toml + uv.lock から依存 install (firered_asr_llm_l/ は sibling)

# HF から FireRedASR-LLM-L 側 DL (.pth.tar + cmvn + README)
uv run --with huggingface_hub python <<'PY'
import os
os.environ.setdefault('HF_HUB_CACHE', '/root/scratchpad/hf-cache')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='FireRedTeam/FireRedASR-LLM-L',
    cache_dir='/root/scratchpad/hf-cache',
    allow_patterns=['*.pth.tar', 'cmvn.*', 'config.yaml', '*.md'],
    token=os.environ['HF_TOKEN'],
)
print('DONE asr:', path)
PY

# HF から Qwen2-7B-Instruct 側 DL (別 repo、~15 GB BF16 sharded safetensors)
uv run --with huggingface_hub python <<'PY'
import os
os.environ.setdefault('HF_HUB_CACHE', '/root/scratchpad/hf-cache')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='Qwen/Qwen2-7B-Instruct',
    cache_dir='/root/scratchpad/hf-cache',
    allow_patterns=['*.safetensors', 'model.safetensors.index.json',
                    'config.json', 'generation_config.json',
                    'tokenizer*.json', '*.md', 'LICENSE'],
    token=os.environ['HF_TOKEN'],
)
print('DONE qwen2:', path)
PY

# Prepare checkpoint (§4.0 bridge land 後、.pth.tar → safetensors extraction + merge)
FIRERED_SNAP=$(ls -d /root/scratchpad/hf-cache/models--FireRedTeam--FireRedASR-LLM-L/snapshots/*/ | head -1)
QWEN_SNAP=$(ls -d /root/scratchpad/hf-cache/models--Qwen--Qwen2-7B-Instruct/snapshots/*/ | head -1)

cd ~/vokra
uv run --project tools/parity/firered_asr_llm_l python \
  tools/parity/firered_asr_llm_l/prepare_checkpoint.py \
  --input-format pth-tar \
  --asr-input   "$FIRERED_SNAP/model.pth.tar" \
  --qwen2-dir   "$QWEN_SNAP" \
  --output      /root/scratchpad/staging/firered-asr-llm-l/model.merged.safetensors

# GitHub 側から実 config を取得 (§0.3 参照、config.yaml が空ゆえ)
git clone --depth 1 https://github.com/FireRedTeam/FireRedASR /root/scratchpad/firered-github
CONFIG_PATH="/root/scratchpad/firered-github/pretrained_models/FireRedASR-LLM-L/config.yaml"
# ↑ 実 upstream の config path は github repo 側で owner が確認、上記は推定 path

# Convert
./target/release/vokra-cli convert \
  --model firered-asr-llm-l \
  --input /root/scratchpad/staging/firered-asr-llm-l/model.merged.safetensors \
  --config "$CONFIG_PATH" \
  --output /root/scratchpad/staging/firered-asr-llm-l/model.gguf

# Publish 5-gate + gate 7 (dry-run → --push で本番)
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/firered-asr-llm-l/model.gguf \
  --repo vokra/firered-asr-llm-l \
  --license-spdx apache-2.0
# ↑ dry-run 全 gate 通過を確認してから ↓
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/firered-asr-llm-l/model.gguf \
  --repo vokra/firered-asr-llm-l \
  --license-spdx apache-2.0 --push

# 検証
curl -sI https://huggingface.co/vokra/firered-asr-llm-l | head -1
# HTTP/2 200 が返れば live
```

### 4.3 CMVN 前処理の runtime forward での取り扱い

`cmvn.ark` (1311 bytes) / `cmvn.txt` (2985 bytes) は Kaldi format の Cepstral
Mean-Variance Normalization 統計。実 forward で feature (mel-fbank) を CMVN で
正規化する必要がある — 現在の Vokra runtime feature extractor (`crates/vokra-ops`
系) には Kaldi CMVN 直接読取り path がない可能性、GGUF metadata に CMVN 統計を
埋める方式が明示的統合パスになる:

- converter 側で `cmvn.ark` を parse し `vokra.firered_asr_llm.cmvn.means`
  / `vokra.firered_asr_llm.cmvn.vars` (u8 array or f32 array) に刻む (future
  wave の runtime forward 実装で consume)
- 現時点の Wave B fast-track converter は cmvn 未 emit の可能性 — owner が converter
  code (`crates/vokra-convert/src/models/firered_asr_llm_l.rs`) を確認し、
  未 emit ならば sidecar bridge を書く時に cmvn tensor を synthetic として
  safetensors に inject する option を追加、converter が pass-through する

### 4.4 SGLang / vLLM sampler 前提 (Qwen2-7B LM decoder 側)

FireRedASR-LLM-L の LM decoder は Qwen2-7B-Instruct ゆえ、upstream で SGLang /
vLLM sampler を使用している可能性がある (Qwen2 系の標準 inference stack)。Vokra
runtime は SGLang / vLLM に依存しない — 標準 Sampler primitive (`crates/vokra-core/src/decode/sampler.rs`)
に置換する:

- **converter 側 impact**: なし (BF16 pass-through は sampler 独立)
- **runtime forward 側 impact**: `crates/vokra-models/src/firered_asr_llm_l/`
  (未実装、future wave) で Qwen2 tokenizer + Vokra Sampler primitive の組み合
  わせを実装 — sibling `canary_qwen` (Canary FastConformer + Voxtral-style
  Qwen decoder) precedent を mirror。既 `vokra-ops::qwen2` (voxtral /
  kyutai_stt / canary_qwen precedent) が Qwen2 forward path を共有ゆえ流用可
- **beam search / n-best**: Whisper / Voxtral と同 API surface で
  `transcribe_beam` (m3-15 で land 済の trait method) を実装可

## 5. §3.1 sign-off status

**現状: blank (fail-closed default)**。

`docs/license-audit.md` §3.1 に `vokra/firered-asr-llm-l` 行を **追加待ち** (本
doc land 時点では未追加、Wave 1 land 時に追加予定)。追加後の状態:

| 列 | 予定値 |
|---|---|
| Vokra slug | `vokra/firered-asr-llm-l` |
| Upstream HF | `FireRedTeam/FireRedASR-LLM-L` |
| Category | ASR (English + Chinese、Conformer + adapter + Qwen2-7B LM decoder) |
| SPDX | apache-2.0 |
| Vokra tier | T1 (Permissive Commercial) |
| Commercial sign-off | **☐ (空欄)** |
| Sign-off date | **☐ (空欄)** |
| Signer | **☐ (空欄)** |

**Owner action** (優先順):

1. **Primary source を直接照合** — https://huggingface.co/FireRedTeam/FireRedASR-LLM-L
   の HF model card + README frontmatter + `github.com/FireRedTeam/FireRedASR`
   LICENSE で apache-2.0 表記を確認 (§0.1 の cardData / README 引用と一致)
2. **Training data audit** — Chinese ASR training-corpus の commercial-use
   audit (WenetSpeech / KeSpeech / aishell1 / aishell2 混成疑義 — README §Evaluation
   に列挙、README §Method には training corpus 明記なし)。可能性:
   - WenetSpeech = CC-BY 4.0 (attribution required、商用可)
   - AISHELL-1 = Apache 2.0 (商用可)
   - AISHELL-2 = 非商用 (要注意、Ali 独自 license)
   - KeSpeech = MIT (商用可、per README frontmatter)
   - 混成の場合は最も restrictive な条件が全体に伝播する可能性
3. **yousan として ☑ Commercial** sign-off (`docs/license-audit.md` §3.1 template
   を使用、CC の primary-source-transcribable pattern で埋める、memory
   `[[feedback-license-signoff-primary-source]]` の rule 準拠) — training data
   audit が完了して問題なしと判定した場合のみ
4. **Qwen2-7B-Instruct 側の license 確認** — FireRedASR-LLM-L publish は Qwen2
   weight を含める形態のため、Qwen2-7B-Instruct の Qwen License Agreement (Apache-2.0-analog、
   月間 100M active user 未満で商用可) の inheritance が働く。`vokra.provenance.inherited_license`
   chunk を converter が刻む必要がある (owner が converter code で確認、
   `crates/vokra-convert/src/models/firered_asr_llm_l.rs` の provenance emission
   に inherited license enum が入っているか)

**publish-one.sh の 5-gate + gate 7** (総論 §2.5 と同じ):

1. Catalog reality — 未実装の ★ 公式 zoo 宣言拒否 (runtime forward 未実装ゆえ
   ★ 宣言しない前提、pass)
2. Redistributable — `LicenseClass::redistributable()` false 拒否 (apache-2.0 は
   Permissive で pass)
3. Provenance chunk 刻印 — `vokra.provenance.*` chunk 群が missing なら拒否
4. §3.1 sign-off 欄 blank 拒否 — **fail-closed default、上記 owner action 必須**
5. T4 (NonCommercial) は `--allow-noncommercial` 明示必須 — 本モデルは T1 ゆえ
   非該当
6. (欠番)
7. **>8 GiB fail-closed (combined GGUF ~18-19 GB → hit)** — `VOKRA_PUBLISH_ON_VAST=1`
   (provision.sh が set) or `--allow-large` で bypass。vast.ai 上では自動 bypass

## 6. 期待される artifacts

Publish 成功後の `huggingface.co/vokra/firered-asr-llm-l` repo に含まれる:

| ファイル | 内容 |
|---|---|
| `model.gguf` | ~18-19 GB (BF16 pass-through、asr side + Qwen2-7B LM decoder 合流、tensor 数は §4.2 の bridge 実装で決まる) |
| `README.md` | `make_model_card.py` 自動生成、tier T1 obligation + apache-2.0 表記 + upstream 情報 + Qwen2 inheritance 表記 |
| `LICENSE` | apache-2.0 canonical text (`fetch_license.sh --spdx apache-2.0`) |
| `NOTICE` | apache-2.0 標準 NOTICE + Qwen2 attribution (upstream `Qwen/Qwen2-7B-Instruct` の LICENSE / NOTICE の inheritance) |
| `SOURCE.md` | 上流 URL (FireRedASR-LLM-L + Qwen2-7B-Instruct 両方) + 再変換手順 + Vokra converter バージョン + commit SHA + `.pth.tar` extraction bridge script pointer |

### GGUF metadata (vokra.* chunk 群、converter が刻む)

| Key | 型 | 値 |
|---|---|---|
| `vokra.schema.version` | string | `"1"` (writer choke point で自動刻印) |
| `vokra.schema.producer` | string | `"vokra-cli-<version>"` |
| `vokra.model.arch` | string | `"firered_asr_llm_l"` (sibling `firered_asr_aed_l` (AED variant) と区別) |
| `vokra.model.category` | string | `"asr"` (sibling Canary-Qwen / Voxtral / Whisper family と同 tier) |
| `vokra.provenance.upstream_hf` | string | `"FireRedTeam/FireRedASR-LLM-L"` |
| `vokra.provenance.upstream_revision` | string | `"9837461f78d15ee66565d00aaec0bc5497d7fbc1"` (pinned SHA、§0.2 API 取得) |
| `vokra.provenance.upstream_license` | string | `"apache-2.0"` |
| `vokra.provenance.inherited_from` | string | `"Qwen/Qwen2-7B-Instruct"` (LM decoder side、owner が converter code で emission 確認) |
| `vokra.provenance.inherited_license` | string | `"tongyi-qianwen-license"` (Qwen2-7B-Instruct の実 SPDX、要 owner 確認) |
| `vokra.firered_asr_llm.cmvn.means` | array<f32> | Kaldi CMVN mean 統計 (`cmvn.ark` から converter が parse、future wave の runtime forward で consume) |
| `vokra.firered_asr_llm.cmvn.vars` | array<f32> | Kaldi CMVN variance 統計 |

## 7. Gate 発火状態 (parity CI)

**現状**: parity CI は未設定 (converter code は既 land、runtime forward は未
実装、`crates/vokra-models/src/firered_asr_llm_l/` は future wave)。

Runtime forward 実装 (future wave) 後の flip-the-switch は:

1. `crates/vokra-models/src/firered_asr_llm_l/` に native forward 実装 (Conformer
   encoder + adapter + Qwen2 LM decoder、sibling `canary_qwen` precedent mirror)
2. `.github/workflows/parity-asr-firered-asr-llm-l-real.yml` scaffold 追加
3. Owner が §5 の sign-off + §4 の publish を完了、fixture GGUF が
   `vokra/firered-asr-llm-l` にある状態
4. `VOKRA_ASR_FIRERED_LLM_ENABLE=1` を GitHub repo settings で set
5. PyTorch reference dump は upstream `github.com/FireRedTeam/FireRedASR` の
   inference script + Qwen2-7B-Instruct を separately pin して runner に install
6. 日本語 fixture (JFK 30s / aishell1 sample など) で WER 計測、Vokra native
   forward vs upstream WER の差分を assert

## 8. Owner critical path (優先順)

**依頼者ルール #3 (publish は §3.1 sign-off 完了後 owner が判断) に従い、以下
順序で**:

1. **Primary source 目視確認** — 本 doc §0.1 (license) + §0.2 (file manifest)
   + §0.3 (GitHub config location) の内容が 2026-08-13 CC 記述時点と一致して
   いることを確認
2. **§4.0 bridge PR** — `.pth.tar` → safetensors extraction を
   `tools/parity/firered_asr_llm_l/prepare_checkpoint.py` に追加 (owner Python
   side、precedent = `tools/parity/nemo_pt_to_safetensors.py`)。CI に不足 test
   は追加、bridge 実装 land 後に §4.1 の run-one.sh が有効になる
3. **Training data audit** — §5 (2) 参照、AISHELL-2 非商用条項が混入していないか
   確認 (混成の場合 restrictive 条件が全体に伝播する法的 risk)
4. **§3.1 sign-off** — `docs/license-audit.md` §3.1 に yousan として ☑ Commercial
   2026-XX-XX sign (training data audit がクリアな場合のみ)、Qwen2 inheritance
   の記載も明示
5. **vast.ai instance 起動** — §2 recipe に従って rent (~$0.6-1.5、~2-3 hour)
6. **§3 provision.sh 実行** — 1 コマンド
7. **§4.1 run-one.sh 実行** — dry-run → `--push` (gate 7 は VOKRA_PUBLISH_ON_VAST=1
   で auto-bypass)
8. **§7 CI flip the switch** (runtime forward 実装後の future wave)

## 9. Notes

- **converter code は既 land**: `crates/vokra-convert/src/models/firered_asr_llm_l.rs`
  (Wave B fast-track、commit `cae8fcd`) と `tools/parity/firered_asr_llm_l/`
  sidecar は既存。ただし sidecar は現状 sharded safetensors 前提、実 upstream は
  `.pth.tar` ゆえ §4.0 bridge が事前 dependency
- **Qwen2-7B-Instruct 依存**: FireRedASR-LLM-L 単体では forward できない (adapter
  は Qwen2 LM decoder を接続する前提)。`vokra/firered-asr-llm-l` に published GGUF
  は Qwen2 weight を含める必要があり、Qwen License Agreement の商用 threshold
  (100M active user 未満は Apache-2.0-analog) の inheritance 表記が必須
- **AED sibling との重複**: FireRedASR-AED-L (`firered_asr_aed_l`、Whisper-topology
  AED、~2.2 GB) は同じ FireRedTeam の別 variant で、`vokra/firered-asr-aed-l`
  として並行 publish 対象 (audit ticket ではあるが本 doc scope 外、別 handoff)
- **restamp_provenance で低メモリ再刻印可能**: publish 後 LICENSE / NOTICE /
  SOURCE.md を差し替えたい場合、`restamp_provenance` で tensor コピー無しで
  刻印可能 (memory `[[project-restamp-provenance]]`、8.7 GB Voxtral を M1 iMac
  16 GB で peak footprint 6.4 MB 実測)。FireRedASR-LLM-L combined GGUF ~18-19 GB
  は M1 iMac 16 GB では mmap tight (依頼者ルール #1 で vast.ai 推奨) だが、
  再刻印の provenance-only update は mmap 経路で M1 iMac でも走る (Voxtral 8.7 GB
  実測の scale-up)
- **BF16 pass-through**: converter は BF16 → BF16 の pass-through (K-quants
  スコープ外)。Runtime forward 側で BF16 → F32 は `decode_bf16` が losslessly
  widen
- **task 指示の "~16.6GB" 表記との差分**: §0.2 参照。asr side 単体 3.38 GiB +
  Qwen2-7B ~15 GB = ~18-19 GB combined。task 指示の 16.6 GB は概算値、実 disk
  footprint は 18-19 GB を見込む
- **task 指示の "safetensors 経路" 前提との差分**: §0.2 / §4.0 参照。upstream
  は `.pth.tar` ゆえ owner side bridge が事前 dependency

## See also

- **Priority ordering (2026-08-14)**: `docs/handoff/vast-ai-execution-priority.md`
  — 本 job は **Priority 2** (bridge PR land + training data audit 後に vast.ai、
  Priority 1 と wall-clock 並行で owner が bridge PR 準備を推奨)

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- Converter: `crates/vokra-convert/src/models/firered_asr_llm_l.rs`
- Sidecar: `tools/parity/firered_asr_llm_l/prepare_checkpoint.py`
- Bridge precedent: `tools/parity/nemo_pt_to_safetensors.py` (M4-20 T17 DFN3
  Phase B の tarball extraction pattern)
- Sibling code-only: `crates/vokra-convert/src/models/firered_asr_aed_l.rs`
  (AED variant、Whisper-topology、~2.2 GB) / `canary_qwen.rs` (Canary FastConformer
  + Voxtral-style Qwen decoder、"encoder + adapter + LLM decoder" mold precedent)
- Sibling published: `huggingface.co/vokra/nemotron-3.5-asr-streaming-0.6b`
  (2026-07-30 published、openmdw-1.1 Permissive) — sibling ASR publish precedent
- Qwen2-7B-Instruct: https://huggingface.co/Qwen/Qwen2-7B-Instruct (LM decoder
  side、Qwen License Agreement)
- 5-gate publish: memory `[[project-huggingface-vokra-publication]]`
- Primary source rule: memory `[[feedback-license-signoff-primary-source]]`
- vast.ai routing: memory `[[feedback-large-models-on-vast-ai]]`
- .pth.tar extraction pattern: memory `[[reference-safetensors-shared-tensor-dedup]]`
  (dedup logic 流用) / `nemo_pt_to_safetensors.py` (tarball extraction precedent)
- Sharded safetensors dedup precedent: memory `[[project-vokra-cli-sharded-safetensors]]`
  (model.safetensors.index.json 直渡し不可の理由と bridge pattern)
