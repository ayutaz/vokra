# Coverage-audit 2026-08-03 Wave A handoff

> 依頼者「ultracodeで計画を立てて全てを対応を進めてください」に基づき launch した coverage-audit 2026-08-03 Wave A top-5 の実装 + publish 進捗。CC 側の converter code 全 landed、publish は 1/5 completed = **nkf-aec** のみ、他 4 model は upstream fetch friction で **owner critical path**。

## 1. 概要

### CC land 済 (2 commit)
- **`cc66fd3`** feat(convert/nkf-aec): Wave A ticket = aec/NEC (MIT, 5.3 KB)
- **`e8b8c21`** feat(convert): Wave A top-4 + 5 §3.1 signoff (coverage-audit 2026-08-03)
  - rnnoise / nsnet2 / dnsmos / frcrn converter Rust + prep script (uv Python 3.12) + tests
  - docs/license-audit.md §3.1 に 5 rows sign-off (all Permissive, CC 判断)
  - scripts/publish/signoff_match.py に 5 slug entry (APPROVED gate 通過)

### verify (main tree HEAD e8b8c21)
- `cargo test -p vokra-convert`: **733 passed / 0 failed / 3 ignored** (11 suites)
- `cargo fmt --check` / `cargo clippy -p vokra-convert -- -D warnings`: OK
- `scripts/check-zero-deps.sh`: OK (root Cargo.lock は vokra-* のみ、NFR-DS-02 preserved)
- `scripts/check-abi-changelog.sh`: OK (v1.0-rc baseline 33 fn + 11 typedef 不変、**新規 C ABI ゼロ**)
- `scripts/publish/signoff_match.py --self-test`: OK (7 approval cases + 1 converter case)

### publish status (huggingface.co/vokra)

| # | Slug | Size | Status | Notes |
|---|------|------|--------|-------|
| 1 | **nkf-aec** | 23.7 KB GGUF | ✅ **HTTP 200** published | upstream = `github.com/fjiang9/NKF-AEC/src/nkf_epoch70.pt` (README `pretrained/nkf.pt` 記述と異なる = 実 file は `src/`)、prep script 動作確認済 |
| 2 | rnnoise-v0.2 | TBD | ⏸ upstream barrier | v0.2 release asset は source tarball のみ、`weights_blob_9.bin` は **build 必須** (`autogen.sh && ./configure && make`) or main branch checkout の別 path。**owner or 別 phase 対応** |
| 3 | nsnet2 | TBD | ⏸ upstream barrier | DNS-Challenge master にも interspeech2020/master にも `NSNet2-baseline` dir 不在。`download-dns-challenge-5-baseline.sh` 経由の **1.4 GB Baseline.zip** DL 要 (Azure blob URL、authorization 不要だが time cost 大)。**owner or 別 phase 対応** |
| 4 | dnsmos-p808-p835 | (blocked) | ⏸ prep script FR-EX-08 loud-fail | ONNX 4 files (`model_v8.onnx` + `sig_bak_ovr.onnx` + `sig.onnx` + `bak_ovr.onnx`) 全 fetch OK。prep script は p808 16 initializers extract 成功、p835 で `mos_estimator_logpow/truediv/y:0` initializer が **empty shape scalar** で FR-EX-08 posture により refuse (fabrication 防止の loud-fail 正しい)。**修正候補**: (a) prep script に "TF-export truediv scalar constant" skip logic 追加 (`shape==[]` かつ integer/float scalar は drop、CLAUDE.md `denoise` prep pattern 準拠)、または (b) 実行時に別 op = ONNX Runtime 経由の DNSMOS session 実装で bypass (Vokra M5 posture と合わない)。**推奨 = (a)** owner が prep script に skip whitelist 追加後 land。 |
| 5 | frcrn | TBD | ⏸ upstream barrier | `github.com/alibabasglab/FRCRN` は README のみで pretrained checkpoint 不在。ModelScope 経由 (`damo/speech_frcrn_ans_cirm_16k`) で `pytorch_model.bin` を DL 必要。HF mirror (`alibabasglab/FRCRN`) = 401 (不在)。**owner ModelScope authentication + `uv add modelscope`** |

## 2. Owner critical path (Wave A 完了までの最短経路)

### 2a. dnsmos-p808-p835 (即時可能、prep script re-run のみ)
```bash
cd /Users/inamotoyuuta/Desktop/Otonx
source tools/parity/.venv/bin/activate  # or `uv sync --project tools/parity`
python tools/parity/dnsmos_prepare_checkpoint.py \
  --p808 ~/checkpoints/dns-challenge/DNSMOS/DNSMOS/model_v8.onnx \
  --p835 ~/checkpoints/dns-challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx \
  --output ~/checkpoints/dnsmos/model.safetensors
./target/release/vokra-cli convert --model dnsmos-p808-p835 \
  --input ~/checkpoints/dnsmos/model.safetensors \
  --output ~/gguf/dnsmos-p808-p835.gguf
export HF_TOKEN=$(grep '^HF=' .env | cut -d'=' -f2-)
bash scripts/publish/publish-one.sh --gguf ~/gguf/dnsmos-p808-p835.gguf \
  --repo vokra/dnsmos-p808-p835 --license-spdx MIT --push
```

### 2b. rnnoise-v0.2 (C build 経由、~5 min)
```bash
cd ~/checkpoints/rnnoise
tar -xzf rnnoise-0.2.tar.gz
cd rnnoise-0.2
./autogen.sh && ./configure && make
# build 完了後、weights_blob_9.bin が root or .libs/ に生成
find . -name "weights_blob_9.bin"
# 次に prep + convert + publish (nkf-aec pattern)
```

### 2c. nsnet2 (DNS5 Baseline.zip 1.4 GB DL、~10 min)
```bash
cd ~/checkpoints/nsnet2
curl -o Baseline.zip https://dnschallengepublic.blob.core.windows.net/dns5archive/Baseline.zip
unzip Baseline.zip
find . -name "nsnet2-20ms-baseline.onnx"
# 見つかったら prep + convert + publish
```

### 2d. frcrn (ModelScope 経由、~5 min)
```bash
cd tools/parity && uv add modelscope
uv run --project tools/parity python -c "
from modelscope import snapshot_download
p = snapshot_download('damo/speech_frcrn_ans_cirm_16k', cache_dir='/tmp/modelscope')
print(p)
"
# .pt 見つかったら prep + convert + publish
```

## 3. 次 phase 判断

### Wave B/C/D の owner sign-off queue
- Wave B fast-track local (13 items) の中の hibiki-2b / sber-gigaam-v3 / reazonspeech-nemo-v2 / NVIDIA family (canary-1b-flash / parakeet-tdt-1.1b / sortformer-diar / magpietts / parakeet-unified-en) は **primary source 精度確認さえ済めば CC 実装可能** (`docs/tickets/coverage-audit-2026-08-03/wave-b/{slug}.md` per-model ticket 完備)
- Wave B vast.ai (14 items) は **vast.ai provisioning + owner sign-off 必須** = 各 model の 独自 license audit と 5-30 GB DL
- Wave C MoE (qwen3-omni / zonos2) は **`moe_dispatch` / `moe_expert_gemm` 新規 op 起票必要** = runtime crate 側の RFC + implementation
- Wave D T4 non-commercial (13 items) は **X-Codec-2 precedent 踏襲** で `--allow-noncommercial` gate 通過、既 pattern

## 4. Deep-research 全体 summary の再引用

- **200 rows 候補** (P0=54 / P1=71 / P2=44 / P3=31)
- **9 分野**: ASR / TTS / Music-gen / Sep / Audio-LLM / Enhancement / Codec+VAD+KWS+Speaker / Eval+SSL+WM / 非HF
- **License 分布 P0**: T1 Commercial 64% / T4 NC 24% / 独自 audit 要 12%
- **Non-HF source**: 25% (ModelScope / NGC / GitHub Release / Zenodo / Sber ai-sage mirror)

詳細:
- `/private/tmp/claude-501/-Users-inamotoyuuta-Desktop-Otonx/aa94d709-.../scratchpad/vokra-audio-model-coverage-audit-2026-08-03.md`
- `docs/tickets/coverage-audit-2026-08-03/INDEX.md` + `OWNER-CRITICAL-PATH.md` + `IMPL-PLAN.md` + 72 per-model MD

## 5. 教訓

- worktree agent の base (d05ab7d) と main tree HEAD (58629ab) が 20+ commits 乖離 = cherry-pick は 11+ conflicts 発生 → **independent files copy + shared file additive edit を 1 agent に統合委任** の pattern が有効
- upstream fetch は README 記述と実 file 配置がズレる (nkf-aec: `pretrained/nkf.pt` 想定だが実 `src/nkf_epoch70.pt`) → prep script は **file name-agnostic** (torch.load で拡張子のみ判定) で robust
- `uv run --project tools/parity` は project directory 存在 warning + cache init hang 事例あり = owner は venv activate 直接 or `uv sync --project tools/parity` 先行推奨
- HF mirror が存在しない ModelScope-only モデル (frcrn) = `uv add modelscope` 経路標準化推奨 (今 wave 前は不要だった、Wave A 以降で FunASR / ClearerVoice-Studio 系が増える見込み = 汎用 pattern 化)

---

**Session 総括**: nkf-aec = **HF vokra org 166 model** に land (前 165 + 1)。他 4 model = converter code land 済 + owner critical path 化。全 §3.1 sign-off 完了 = publish gate 通過準備完了。
