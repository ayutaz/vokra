# post-audit 2026-08-13 wave — summary

**Date**: 2026-08-13
**Branch**: `feat/post-audit-cc-gap-2026-08-13`（main HEAD `40558f5` から作成）
**Author**: Claude Code（本 doc は audit → plan → implement workflow の最終
handoff summary）
**Scope**: PR #28 merged（2026-08-12）後の post-audit CC-gap wave。Plan phase の
reality-check で `already_landed` / `true_gaps`（Utility / Music-und / SSL 各
wave 実装） / `vast_ai_handoff_only`（本 handoff docs 3 件）/ `out_of_scope`
（Non-goals 該当）に分類。

---

## 1. Plan phase reality-check 結果

Audit findings に対し「実 HEAD `40558f5` で既 landed か / true gap か / vast.ai
必要か / Non-goals か」を再判定。

| Bucket | 件数 | 対応 |
|---|---|---|
| `already_landed` | 判定内訳は per-audit item、本 wave では **counting は Plan phase 側で完結** | 対応不要（reality-check で消化） |
| `true_gaps` | 14 件 | Utility (2) + Music-und (6) + SSL (5) + Refactor (1)、本 branch で implement wave land 済 |
| `vast_ai_handoff_only` | 3 件 | 本 doc + 3 handoff docs で owner-hand-off |
| `out_of_scope` (Non-goals) | 判定内訳は per-audit item | Non-goals 該当（Matcha-TTS / RVC 系 / AudioSeal embed 等）は再開放禁止、実装しない |

---

## 2. Land 状況（各 wave commit SHA 一覧）

**本 branch 上の全 commit（main から 14 commits ahead、後続 verify wave が填める
予定）**:

### 2.1 Refactor（1 commit）

| SHA | 内容 |
|---|---|
| `64490d3` | `refactor(piper_plus): rename synthesize_full -> synthesize_pseudo_streaming (FR-ST-04)` |

### 2.2 Utility wave（2 commits、MoE primitives）

| SHA | 内容 |
|---|---|
| `d74cbac` | `feat(vokra-ops): add MoE dispatch primitive (top-k routing + capacity gate)` |
| `a9036fb` | `feat(vokra-ops): add MoE expert GEMM primitive (per-expert reduction)` |

### 2.3 Music-und wave（6 commits、music-understanding converters）

| SHA | 内容 |
|---|---|
| `9414536` | `feat(convert): YAMNet 521-class AudioSet edge classifier (music-und wave)` |
| `d6eb842` | `feat(convert): MERT-v1-330M music-understanding embedding (music-und wave)` |
| `083f531` | `feat(convert): MuQ Mel-RVQ + BEATs teacher music encoder (music-und wave)` |
| `2510c6c` | `feat(convert): Dasheng universal audio encoder (music-und wave)` |
| `d20c8d3` | `feat(convert): PANNs Cnn14 527-class AudioSet tagging (music-und wave)` |
| `87a6c2e` | `feat(convert): Basic-Pitch polyphonic audio-to-MIDI (music-und wave)` |

### 2.4 SSL-encoder wave（5 commits、self-supervised audio encoders）

| SHA | 内容 |
|---|---|
| `defe26f` | `feat(convert): BEATs foundational SSL audio encoder (SSL-encoder wave)` |
| `ca04c1b` | `feat(convert): EAT Effective Audio Transformer SSL (SSL-encoder wave)` |
| `a8867cf` | `feat(convert): ATST Audio Teacher-Student Transformer SSL (SSL-encoder wave)` |
| `79c3691` | `feat(convert): MAEST Discogs AST music-tagger SSL (SSL-encoder wave)` |
| `bdce8c3` | `feat(convert): M2D Masked Modeling Duo SSL (SSL-encoder wave close)` |

**HEAD**: `bdce8c3`（SSL-encoder wave close 時点）

**Verify status**: 後続 verify wave が本 branch 全体で `cargo test --workspace` /
fmt / clippy `-D warnings` / zero-dep / abi-changelog / gen-c-abi drift check を
再走予定。個々の converter commit は wave 内で verify したが、branch tip での
integrated verify は wave close 後の verify commit 待ち（本 handoff summary land
と同 commit で verify 結果を残す想定、または後続 commit で追加）。

**依頼者ルール #3 の遵守**: 上記 14 converter は **すべて converter + test + docs
まで**。実 publish（HF upload）は **§3.1 sign-off 完了後 owner が判断**。CC は
publish action を実行していない。

---

## 3. Vast.ai handoff docs 一覧

**Plan phase の `vast_ai_handoff_only` 3 対象について、owner-triggered runbook を
作成した**（本 wave が「実装漏れではなく別 WP」判断で honest scope boundary を
維持したもの）。

| # | Model / Work item | Handoff doc | Size / License | Owner trigger 理由 |
|---|---|---|---|---|
| 1 | **VoxCPM2-2B**（openbmb/VoxCPM2） | [`docs/handoff/vast-ai-publish-voxcpm2-2b.md`](vast-ai-publish-voxcpm2-2b.md) | 4.96 GB BF16 / apache-2.0 | 依頼者ルール #1（≥2GB は vast.ai）。設計 spec `2026-07-28-voxcpm2-2b-design.md` §5 の Wave 0 ADR Option A/B/C 収束が gate、runtime + converter variant-aware が Wave 1 で land すれば CI 側は既 pinned SHA で待機中 |
| 2 | **RMVPE**（Dream-High/RMVPE MIT） | [`docs/handoff/vast-ai-publish-rmvpe.md`](vast-ai-publish-rmvpe.md) | 180 MB / mit | サイズ的には local M1 iMac 可（依頼者ルール #2）だが、`.pt` pickle security + 内部 U-Net + GRU forward が real weight 到着まで defer（loud-partial `VokraError::UnsupportedOp`）。CC は defer marker 維持、topology 実装は owner-provisioned real checkpoint bundle が parity harness で verify されるまで pending（CLAUDE.md wave 3 "under-specified in primary source" 判断） |
| 3 | **Vocoder / codec GPU kernel wave**（HiFTNet / BigVGAN / SNAC / Qwen3-TTS-codec 等の Metal MSL + CUDA NVRTC） | [`docs/handoff/vast-ai-vocoder-gpu-kernels.md`](vast-ai-vocoder-gpu-kernels.md) | N/A（kernel work item） | Metal 半分は M1 iMac 上で CC 可 / CUDA 半分は vast.ai owner 必須で非対称、既存 M4 spec 明示的スコープ外 follow-up = 別 WP。CPU arm は既 real で機能完結、GPU 化 = 性能最適化ゆえ v1.0 GA blocking ではない（M5-13 C ABI 凍結 precondition 外） |

**共通**: すべて `docs/handoff/vast-ai-large-model-publish.md`（総論 = §2 recipe
/ §3 provision.sh gotcha / §4 lifecycle）を前提とし、各 handoff は該当モデル /
work item に固有の差分のみを記述。

---

## 4. Owner critical path

本 wave 完了後の owner-triggered work item リスト:

### 4.1 短期（本 branch の PR 作成 + merge 前後）

1. **Verify wave 実行** — 本 branch の 14 commit を `cargo test --workspace` / fmt
   / clippy / zero-dep / abi-changelog / gen-c-abi drift check で integrated verify
   （CC が後続 commit で追加予定 or PR review 時に verify）
2. **本 branch → PR 作成** — `feat/post-audit-cc-gap-2026-08-13` から main へ、
   14 commit を bundle merge

### 4.2 中期（本 handoff docs の owner action）

3. **VoxCPM2-2B publish**（handoff #1）:
   - Wave 0 ADR 確定（Option A / B / C）
   - Wave 1 runtime + converter variant-aware land 確認
   - HF primary source 直接照合（apache-2.0）
   - §3.1 sign-off（yousan として ☑ Commercial）
   - vast.ai instance 起動（~$0.3-0.5、~1 hour）
   - `run-one.sh --push` で publish
   - CI variable `VOKRA_TTS_CONT_VAE_ENABLE=1` set、parity CI flip the switch
4. **RMVPE publish**（handoff #2）:
   - GitHub primary source 直接照合（MIT）
   - §3.1 sign-off
   - Local M1 iMac 上で `.pt` → safetensors → GGUF bridge（vast.ai 起動不要）
   - `publish-one.sh --push` で publish
   - CI variable `VOKRA_RMVPE_ENABLE=1` + `VOKRA_RMVPE_REAL_GGUF_PATH` set
   - **後続 CC wave** への request: real weight bundle が parity harness に届いた
     と report → CC が U-Net + GRU forward kernel 実装 wave を起動（silent-wrong
     risk 回避、real weight で iterate 必須）
5. **Vocoder / codec GPU kernel wave**（handoff #3）:
   - 優先 op を owner が指定（推奨: `mimi_rvq` + `hiftnet`）
   - CC 側 Metal 半分実装（M1 iMac local、bit-identical parity）
   - vast.ai instance 起動（~$1-4、~2-4 hours）
   - CC 側 CUDA kernel source string 下書き
   - vast.ai 上で NVRTC compile + real GPU bakeoff（bit-identical vs CPU）
   - commit + push、integrated verify

### 4.3 長期（本 wave と独立、既存 owner critical path 6 系統）

CLAUDE.md「現在のフェーズ状態」に列挙された 6 系統は本 wave で touch していない:

6. NPU bakeoff（M5-01 CoreML/ANE + M5-02 QNN/Hexagon、NFR-PF-12 2× gate、M5-13
   C ABI 凍結の precondition）
7. EU 認証（EU AI Act Article 50、2026-08-02 applies）
8. 資金調達（seed $500K-$1M 級、Cloudflare AI Gateway on-device 版 positioning）
9. NDA（M5-04 console static-link gate、実運用）
10. voice-clone 別リポ（`vokra-voiceclone-experimental` publish、ELVIS Act 分離
    ポリシー）
11. v1.0 GA タグ（M5-13 T17、上記 6 + 10 完了後）

---

## 5. Next actionable

**Owner triggered**:

1. Verify wave (`cargo test --workspace` + gate check) を本 branch tip `bdce8c3`
   に対して実行、結果を報告
2. 本 branch から main への PR 作成 → merge
3. 上記 §4.2 の handoff #1-3 のうち **最優先 handoff を選択** し、該当 runbook を
   実行:
   - **推奨最優先 = handoff #1 VoxCPM2-2B**（CI 側 pinned SHA で既 waiting、Wave
     0 ADR + Wave 1 land + sign-off ですぐ flip the switch 可能、publish 実績で
     org 総計 195+ モデルへ）
   - 次点 = handoff #2 RMVPE（local M1 iMac で完結可、vast.ai 費用ゼロ、F0
     tier 3 姉妹の trio 最後を close）
   - 低優先 = handoff #3 GPU kernel wave（correctness ではなく性能最適化、v1.0
     GA blocking でない、後回し可）

**CC triggered**（本 wave で完了）:

- 本 handoff summary + 3 handoff docs の commit + push（本 wave の最終
  deliverable）

---

## 6. 教訓 / 規律

本 wave で確認・維持した規律:

1. **依頼者ルール #1 (≥2GB は vast.ai)** — VoxCPM2-2B (4.96 GB) を local convert
   attempt せず、runbook 作成のみ。実 instance 起動は owner。
2. **依頼者ルール #3 (publish は §3.1 sign-off 完了後 owner が判断)** — CC 側で
   converter + test + docs まで land、実 HF upload は絶対に行わない。
3. **honest scope boundary** — Vocoder GPU kernel wave の Metal 半分 (M1 iMac 可)
   と CUDA 半分 (vast.ai 必須) の非対称性を「実装漏れではなく別 WP」として明示
   （fake-complete より honest、CLAUDE.md M4 節末尾の判断継承）。
4. **loud-partial は fake-complete より honest** — RMVPE の `extract_real` =
   `VokraError::UnsupportedOp` を維持、best-guess topology を書いて silent-wrong
   を犯すより pending signal で loud fail（FR-EX-08）。
5. **Non-goals 該当は絶対に手を出さない** — Matcha-TTS / RVC v2・GPT-SoVITS in
   `ayutaz/vokra` / AudioSeal 強制-embedding / NNAPI / Piper (piper1-gpl) /
   ONNX グラフ受け / Bark 2 / watermark 埋め込み engine の 8 系統は再開放禁止、
   handoff docs も作らない。
6. **数字を捏造しない** — 本 handoff docs 3 件はいずれも実 vast.ai instance を
   起動していないゆえ RTF / speedup / cost の数字は書かない（size / license /
   §3.1 sign-off status は primary source 照合済のみ記載、vast.ai 実行時の実測
   は owner run 時に埋める placeholder として残した）。

---

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- 本 wave の handoff docs:
  - `docs/handoff/vast-ai-publish-voxcpm2-2b.md`
  - `docs/handoff/vast-ai-publish-rmvpe.md`
  - `docs/handoff/vast-ai-vocoder-gpu-kernels.md`
- CLAUDE.md 「現在のタスク状態」= 前回 wave（PR #28 merged 2026-08-12）+ 本 wave
  との継承関係
- 設計 spec:
  - `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`（VoxCPM2-2B）
- CI workflows:
  - `.github/workflows/parity-tts-continuous-vae-real.yml`（VoxCPM2-2B、既 2B
    pin 待機中）
  - `.github/workflows/parity-rmvpe-real.yml`（RMVPE、owner-driven flip switch
    待機中）
- Memory: [[feedback-large-models-on-vast-ai]] / [[feedback-license-signoff-primary-source]] /
  [[project-m4-implementation]] / [[project-huggingface-vokra-publication]] /
  [[reference-vast-ai-hf-config-pth-shim]] / [[reference-huggingface-hub-lt-030-vast-ai]]
