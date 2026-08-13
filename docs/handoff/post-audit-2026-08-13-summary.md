# post-audit 2026-08-13 wave — summary

**Date**: 2026-08-13
**Branch**: `feat/post-audit-cc-gap-2026-08-13`（main HEAD `40558f5` から作成）
**Author**: Claude Code（本 doc は audit → plan → implement workflow の最終
handoff summary、後続 WF1 wave land を含めて追記済）
**Scope**: PR #28 merged（2026-08-12）後の post-audit CC-gap wave。Plan phase の
reality-check で `already_landed` / `true_gaps`（Utility / Music-und / SSL 各
wave 実装） / `vast_ai_handoff_only`（本 handoff docs 3 件）/ `out_of_scope`
（Non-goals 該当）に分類。**後続に WF1 wave（RMVPE 実装 + KWS Phase 1 +
Vocoder Metal 初 op + SPDX 拡張、8 commits）が land、依頼者 M1 iMac 16GB OOM
発火事象を受けて memory-safe workflow 規律（§7）を確立**。

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

**本 branch 上の全 commit（main から 25 commits ahead = 14 converter + 3 handoff
docs + 8 WF1）**:

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

### 2.5 Handoff docs 初版（3 commits、本 summary + vast.ai runbook 3 件）

| SHA | 内容 |
|---|---|
| `0682d01` | `docs(handoff): vast.ai publish runbook for VoxCPM2-2B / RMVPE / vocoder GPU kernel wave` |
| `ce5dcd4` | `docs(handoff): post-audit 2026-08-13 wave summary`（本 doc の genesis） |
| `ee7dbfb` | `chore(lockfiles): sync excluded workspace Cargo.lock after wave commits` |

### 2.6 WF1 wave（8 commits、RMVPE 実装 + KWS Phase 1 + Vocoder Metal 初 op + SPDX 拡張）

**依頼者 M1 iMac 16GB がパンクした事象**（詳細 = §7）を受け、以降の workflow
は memory-safe workflow 規律（`CARGO_BUILD_JOBS=1` / per-crate / no `--workspace`
/ GPU feature 同時 compile 禁止）で進める前提で発火した最初の wave。

| SHA | 内容 |
|---|---|
| `e7b6810` | `feat(rmvpe): real U-Net + BiGRU forward with fixture-gated parity (loud-partial resolved)` |
| `7db02be` | `chore(publish): fetch_license.sh SPDX 拡張 (gpl-3.0 / lgpl-3.0 / mpl-2.0 / isc / unlicense / epl-2.0)` |
| `5343731` | `feat(tools/parity): microwakeword prepare_checkpoint.py (TFLite→ vokra.kws GGUF, uv Python 3.12)` |
| `c21cb14` | `feat(vokra-kws-micro): 40-band log-mel feature extraction + scalar transcendentals (no_std, M5-03b Phase 1)` |
| `66d0077` | `feat(mimi-rvq): Metal MSL gather+fold kernel + bit-identical vs CPU parity` |
| `cca69ba` | `feat(tools/parity): rmvpe reference dumper (yxlllc verbatim, MIT, uv Python 3.12)` |
| `0f39478` | `docs(handoff): rmvpe topology fully-specified per e7b6810, loud-partial resolved` |
| `e972f70` | `chore(lockfiles): sync tools/parity/uv.lock for rmvpe-parity workspace member` |

**HEAD**: `e972f70`（WF1 wave close 時点、main から 25 commits ahead）

**WF1 land 内訳** — 5 系統の deliverable:
1. **RMVPE loud-partial resolved**（`e7b6810` + `cca69ba` + `0f39478` + `e972f70`）: 上流 `yxlllc/RMVPE`（MIT）の primary source 再精査で U-Net + BiGRU + head topology が primary-source-transcribable と判明 → real forward を land（inline `pool2d` + `conv_transpose2d` + `pytorch_gru`、外部 op 依存なし = NFR-DS-02 保存）。`extract_real()` は `VokraError::UnsupportedOp` を返さなくなった。Path A（`VOKRA_RMVPE_REAL_GGUF`）+ Path B（`VOKRA_RMVPE_REAL_HIDDEN` + `_ARGMAX` + `_HIDDEN_FEATURE_DIM`、`tools/parity/rmvpe/dump_reference.py` 発、argmax-match-rate ≥ 99 % gate）の両 fixture-gated parity leg も land。2026-07-30 の「under-specified」判定は REVERSED。
2. **microWakeWord KWS Phase 1**（`5343731` + `c21cb14`）: (a) 上流 kahrendt/microWakeWord canonical TFLite → `vokra.kws.*` GGUF の offline sidecar（TFLite Interpreter walk + INT8 dequant + provenance/frontend metadata group）+ (b) `vokra-kws-micro` crate に 40-band log-mel front-end + 自前 scalar transcendentals（`#![no_std]` + alloc、512-pt radix-2 FFT + HTK triangular mel）。`detect()` は scaffold のまま = Phase 2 で real classifier を配線予定（M5-03 IoT Tier-3 KWS 側、ADR M5-03b Proposed）。
3. **Vocoder Metal 初 op**（`66d0077`）: `HotOp::MimiRvq.covered_by_metal()` を `false → true` に flip、`Compute::mimi_rvq_f32` の Metal arm を `VokraError::UnsupportedOp` から real `vokra_mimi_rvq_gather_fold_f32` MSL kernel dispatch に変更。CPU `rvq_fold_core` と bit-identical（max |Δ| = 0）を tiny/canonical 両 shape で M1 iMac 上検証済（P2 sub-wave 1/11 of the Vocoder Metal 半分 wave、M3-06 T14）。残 10 op は WF2 で land 予定。
4. **公式 publish パイプ SPDX 拡張**（`7db02be`）: `fetch_license.sh` に GPL-3.0 / LGPL-3.0 / MPL-2.0 / ISC / Unlicense / EPL-2.0 の canonical LICENSE URL を追加（gnu.org / unlicense.org 直、MPL/ISC/EPL は SPDX license-list-data raw）。`--self-test` を全 canonical_url() branch 網羅の 18-suite coverage loop に書き換え。
5. **lockfile drift sync**（`e972f70`）: `uv init tools/parity/rmvpe/` に伴う `tools/parity/pyproject.toml` `[tool.uv.workspace] members` の workspace lockfile 追随。

**Verify status**: WF1 wave の 8 commit は **memory-safe workflow 規律**（§7）に
従い per-crate（`cargo test -p <single-crate> --lib` + `CARGO_BUILD_JOBS=1`）で
verify 済。branch tip での integrated `cargo test --workspace` は禁止事項ゆえ
未実行（M1 iMac 16GB OOM 回避）、CI 側 workflow で verify。

**依頼者ルール #3 の遵守**: 上記 14 converter + WF1 の RMVPE / KWS / Vocoder
Metal 実装は **すべて converter + test + docs まで**。実 publish（HF upload）は
**§3.1 sign-off 完了後 owner が判断**。CC は publish action を実行していない。

---

## 3. Vast.ai handoff docs 一覧

**Plan phase の `vast_ai_handoff_only` 3 対象について、owner-triggered runbook を
作成した**（本 wave が「実装漏れではなく別 WP」判断で honest scope boundary を
維持したもの）。

| # | Model / Work item | Handoff doc | Size / License | Owner trigger 理由 |
|---|---|---|---|---|
| 1 | **VoxCPM2-2B**（openbmb/VoxCPM2） | [`docs/handoff/vast-ai-publish-voxcpm2-2b.md`](vast-ai-publish-voxcpm2-2b.md) | 4.96 GB BF16 / apache-2.0 | 依頼者ルール #1（≥2GB は vast.ai）。設計 spec `2026-07-28-voxcpm2-2b-design.md` §5 の Wave 0 ADR Option A/B/C 収束が gate、runtime + converter variant-aware が Wave 1 で land すれば CI 側は既 pinned SHA で待機中 |
| 2 | **RMVPE**（Dream-High/RMVPE MIT） | [`docs/handoff/vast-ai-publish-rmvpe.md`](vast-ai-publish-rmvpe.md) | 180 MB / mit | **✅ 2026-08-13 WF1 update: loud-partial resolved**（`e7b6810`）。上流 `yxlllc/RMVPE`（MIT）の primary source 再精査で U-Net + BiGRU + head topology が primary-source-transcribable と判明 → real forward を land。CLAUDE.md wave 3 "under-specified in primary source" 判定は REVERSED。**owner critical path 圧縮**: real verify に vast.ai 不要（local M1 iMac で完結）、`fetch_rmvpe_pt.sh` の curl ~5 分 + `tools/parity/rmvpe/dump_reference.py` の `uv run` ~30 秒で Path B fixture が揃い、`cargo test -p vokra-models parity_rmvpe` を per-crate で発火可能（memory-safe rule 準拠、§7） |
| 3 | **Vocoder / codec GPU kernel wave**（HiFTNet / BigVGAN / SNAC / Qwen3-TTS-codec 等の Metal MSL + CUDA NVRTC） | [`docs/handoff/vast-ai-vocoder-gpu-kernels.md`](vast-ai-vocoder-gpu-kernels.md) | N/A（kernel work item） | **2026-08-13 WF1 partial land**（`66d0077`）: `mimi_rvq` Metal MSL 初 op（sub-wave 1/11）が bit-identical vs CPU で land。残 10 op は WF2（別 workflow）へ。Metal 半分は M1 iMac 上で CC 可 / CUDA 半分は vast.ai owner 必須で非対称、既存 M4 spec 明示的スコープ外 follow-up = 別 WP。CPU arm は既 real で機能完結、GPU 化 = 性能最適化ゆえ v1.0 GA blocking ではない（M5-13 C ABI 凍結 precondition 外） |

**共通**: すべて `docs/handoff/vast-ai-large-model-publish.md`（総論 = §2 recipe
/ §3 provision.sh gotcha / §4 lifecycle）を前提とし、各 handoff は該当モデル /
work item に固有の差分のみを記述。

---

## 4. Owner critical path

本 wave 完了後の owner-triggered work item リスト:

### 4.1 短期（本 branch の PR 作成 + merge 前後）

1. **Verify wave 実行** — 本 branch の 25 commit（14 converter + 3 handoff + 8 WF1）
   を integrated verify。**memory-safe workflow 規律（§7）** の制約下で:
   - Local M1 iMac 16GB 上では `cargo test --workspace` / `--all-features` は
     絶対に使わない（前回 OOM 発火経路）→ per-crate `cargo test -p <crate> --lib`
     + `CARGO_BUILD_JOBS=1` で個別走査、または CI 側 workflow で verify
   - `cargo fmt --check` / `scripts/check-zero-deps.sh` / `scripts/check-abi-changelog.sh`
     / `scripts/gen-c-abi.sh --check` はゼロメモリで local OK
2. **本 branch → PR 作成** — `feat/post-audit-cc-gap-2026-08-13` から main へ、
   25 commit を bundle merge

### 4.2 中期（本 handoff docs の owner action）

3. **VoxCPM2-2B publish**（handoff #1）:
   - Wave 0 ADR 確定（Option A / B / C）
   - Wave 1 runtime + converter variant-aware land 確認
   - HF primary source 直接照合（apache-2.0）
   - §3.1 sign-off（yousan として ☑ Commercial）
   - vast.ai instance 起動（~$0.3-0.5、~1 hour）
   - `run-one.sh --push` で publish
   - CI variable `VOKRA_TTS_CONT_VAE_ENABLE=1` set、parity CI flip the switch
4. **RMVPE publish**（handoff #2）— **2026-08-13 WF1 update: real forward 実装済**:
   - GitHub primary source 直接照合（MIT、`yxlllc/RMVPE` = 上流 fork の
     primary-source-transcribable topology、`Dream-High/RMVPE` = paper origin）
   - §3.1 sign-off
   - **Local M1 iMac のみで real verify 完結**（vast.ai 起動不要、memory-safe
     rule 準拠 = §7）: `tools/parity/rmvpe/fetch_rmvpe_pt.sh` の curl ~5 分 +
     `tools/parity/rmvpe/dump_reference.py` の `uv run` ~30 秒で Path B fixture
     （`hidden.f32` + `argmax.u32` + `meta.json`）が揃う → `VOKRA_RMVPE_REAL_HIDDEN`
     + `_ARGMAX` + `_HIDDEN_FEATURE_DIM` を env で set → `cargo test -p vokra-models
     parity_rmvpe`（per-crate、`CARGO_BUILD_JOBS=1`）で ≥ 99 % argmax-match-rate
     gate 発火
   - Local M1 iMac 上で `.pt` → safetensors → GGUF bridge（vast.ai 起動不要、
     180 MB は依頼者ルール #1 の ≥2GB 閾値以下）
   - `publish-one.sh --push` で publish
   - CI variable `VOKRA_RMVPE_ENABLE=1` + `VOKRA_RMVPE_REAL_GGUF_PATH` set
   - **後続 CC wave 依頼不要**: WF1 `e7b6810` で U-Net + BiGRU forward kernel
     は real 実装済（silent-wrong risk 回避、fixture-gated parity で bind）
5. **Vocoder / codec GPU kernel wave**（handoff #3）— **2026-08-13 WF1 partial land**:
   - WF1 `66d0077` で `mimi_rvq` Metal MSL 初 op 発火（sub-wave 1/11）、CPU
     `rvq_fold_core` と bit-identical（max |Δ| = 0）
   - **WF2（別 workflow）で Vocoder Metal 残 10 op**: HiFTNet / BigVGAN / SNAC
     / Qwen3-TTS-codec 等の Metal MSL kernel、M1 iMac local で per-op bit-identical
     parity で land 予定（memory-safe rule 準拠、`CARGO_BUILD_JOBS=1` + per-crate）
   - CUDA 半分は vast.ai owner 必須（~$1-4、~2-4 hours、NVRTC compile + real
     GPU bakeoff）
   - commit + push、integrated verify（CI 側 workflow）

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

1. Verify wave — 本 branch tip `e972f70` に対して **memory-safe rule（§7）** の
   制約下で per-crate 走査 or CI 側 workflow で integrated verify（**`cargo test
   --workspace` は絶対に使わない**、前回 OOM 発火経路）
2. 本 branch から main への PR 作成 → merge（25 commits bundle）
3. 上記 §4.2 の handoff #1-3 のうち **最優先 handoff を選択** し、該当 runbook を
   実行:
   - **推奨最優先 = handoff #2 RMVPE**（WF1 で real forward + Path B dumper が
     land 済 → **owner curl ~5 分 + `uv run` ~30 秒で local M1 iMac 上 real
     verify 完結**、vast.ai 費用ゼロ、F0 tier 3 姉妹の trio 最後を close。
     `publish-one.sh --push` で org 総計 +1）
   - 次点 = handoff #1 VoxCPM2-2B（CI 側 pinned SHA で既 waiting、Wave 0 ADR
     + Wave 1 land + sign-off ですぐ flip the switch 可能、publish 実績で org
     総計 195+ モデルへ、vast.ai ~$0.3-0.5 / ~1 hour）
   - 低優先 = handoff #3 GPU kernel wave（correctness ではなく性能最適化、v1.0
     GA blocking でない、後回し可。WF2 で残 10 Metal op が CC 側 land 予定）

**CC triggered**（後続 workflow）:

- **WF2**: Vocoder Metal 残 10 op（M3-06 T14 sub-wave 2/11 - 11/11、`mimi_rvq`
  以外の HiFTNet / BigVGAN / SNAC / Qwen3-TTS-codec 等）を per-op bit-identical
  parity で land。memory-safe rule 準拠（`CARGO_BUILD_JOBS=1` + per-crate、
  GPU feature 同時 compile 禁止）で M1 iMac 上完結。
- **WF3**: (a) microWakeWord Phase 2/3 = WF1 で land した `vokra-kws-micro` の
  40-band log-mel front-end に real classifier を配線（`detect()` の scaffold を
  real forward に置換、no_std 保存）+ (b) MoE coverage-audit fast-track subset
  = MoE routing/dispatch primitives（`vokra-ops` に既 land）を実消費する converter
  wave の残（MoE-based TTS/ASR モデルの subset）。

- 本 handoff summary の update commit（本 land）

---

## 6. 教訓 / 規律

本 wave で確認・維持した規律:

1. **依頼者ルール #1 (≥2GB は vast.ai)** — VoxCPM2-2B (4.96 GB) を local convert
   attempt せず、runbook 作成のみ。実 instance 起動は owner。
2. **依頼者ルール #3 (publish は §3.1 sign-off 完了後 owner が判断)** — CC 側で
   converter + test + docs まで land、実 HF upload は絶対に行わない。
3. **honest scope boundary** — Vocoder GPU kernel wave の Metal 半分 (M1 iMac 可)
   と CUDA 半分 (vast.ai 必須) の非対称性を「実装漏れではなく別 WP」として明示
   （fake-complete より honest、CLAUDE.md M4 節末尾の判断継承）。**WF1 で mimi_rvq
   Metal 初 op が sub-wave 1/11 として land**、残 10 op は WF2 へ = pattern の
   実践。
4. **loud-partial は fake-complete より honest — かつ primary source 再精査で
   REVERSED しうる** — 2026-07-30 CLAUDE.md wave 3 判断で RMVPE の `extract_real`
   = `VokraError::UnsupportedOp` を loud-partial 維持していたが、2026-08-13 の
   feasibility 調査（`wf_7062f2d5`）で上流 `yxlllc/RMVPE`（MIT）の primary source
   を再精査したところ **fully-specified** と判明 → WF1 `e7b6810` で real forward
   を land、defer 判断は REVERSED。「loud-partial 判定は上流を再精査したうえで
   下すのが望ましい」という後続 pattern を確立。
5. **Non-goals 該当は絶対に手を出さない** — Matcha-TTS / RVC v2・GPT-SoVITS in
   `ayutaz/vokra` / AudioSeal 強制-embedding / NNAPI / Piper (piper1-gpl) /
   ONNX グラフ受け / Bark 2 / watermark 埋め込み engine の 8 系統は再開放禁止、
   handoff docs も作らない。
6. **数字を捏造しない** — 本 handoff docs 3 件はいずれも実 vast.ai instance を
   起動していないゆえ RTF / speedup / cost の数字は書かない（size / license /
   §3.1 sign-off status は primary source 照合済のみ記載、vast.ai 実行時の実測
   は owner run 時に埋める placeholder として残した）。**WF1 の Metal parity は
   実測 max |Δ| = 0（bit-identical）** を M1 iMac 上で走らせて記録、こちらは
   実測ゆえ数字を残した。
7. **Memory constraint を workflow 規律に格上げ**（§7 新規） — 依頼者 M1 iMac
   16GB OOM 発火を受けて `CARGO_BUILD_JOBS=1` + per-crate + GPU feature 排他 の
   3 点セットを規律化。integrated verify は CI 側に委譲する分業で local 開発の
   continuity を維持。今後の全 workflow に適用。

---

## 7. Memory constraint（M1 iMac 16GB OOM 発火 → memory-safe workflow 規律）

**発火事象**（2026-08-13）: 依頼者 M1 iMac 16GB 上で `cargo test --workspace` を
走らせた際、metal / cuda / vulkan feature の compile を並列 rustc job で同時に
発火させたことで **OS が out-of-memory kill を発火 → セッションがパンク**。前回
wave（Utility + Music-und + SSL 全 14 converter）を landed した直後の verify 経路
で顕在化。

**根本原因**:

- `cargo test --workspace` は依存 crate を全て並列 compile → RSS が数 GB 台に
  即到達
- `--all-features` は `metal` / `cuda` / `vulkan` / `webgpu` / `coreml` / `qnn`
  を同時に有効化 → 生 FFI + shader source string の compile 単位が同時発火し
  workspace-wide の compile working set が 16GB を超える
- `CARGO_BUILD_JOBS` の default（=論理コア数、M1 iMac で 8）が rustc 並列度を
  8 に上げる → 各 rustc の peak RSS × 8 で M1 iMac 16GB を突破

**memory-safe workflow 規律**（本 WF1 以降、依頼者 M1 iMac 上の CC 作業全般に
適用）:

| # | ルール | 理由 |
|---|---|---|
| 1 | 全 cargo command で `CARGO_BUILD_JOBS=1` を必ず設定 | parallel rustc 禁止 = peak RSS × 1 |
| 2 | `cargo test --workspace` / `--all-features` / `--all-targets` は禁止 | OOM 発火経路の直接的 root cause |
| 3 | 使う command は `cargo test -p <single-crate> --lib` のみ、1 crate at a time | 依存 crate compile 単位を最小化 |
| 4 | Clippy は `cargo clippy -p <single-crate> -- -D warnings` のみ、対象 crate だけ | 同上 |
| 5 | Fmt は `cargo fmt --check`（ゼロメモリ、OK） | rustc invocation なし = safe |
| 6 | GPU feature（`metal` / `cuda` / `vulkan`）を同時 compile 禁止 | 生 FFI + MSL/PTX/GLSL source string の compile 単位が同時発火 = 16GB 超 |
| 7 | ≥2GB モデルの convert / parity は一切 local で実行しない | vast.ai handoff、既存 `docs/handoff/` で owner-triggered（依頼者ルール #1） |
| 8 | `scripts/check-zero-deps.sh` / `scripts/check-abi-changelog.sh` / `scripts/gen-c-abi.sh --check` は local OK | shell/Python のみ、rustc 発火なし |
| 9 | integrated `cargo test --workspace` は CI 側 workflow で verify | GitHub Actions runner は 7GB RAM だが並列 job で分散可能 |

**適用対象**:

- 本 branch `feat/post-audit-cc-gap-2026-08-13` の WF1 wave 以降の全 CC 作業
- 今後の workflow（WF2 Vocoder Metal 残 10 op / WF3 microWakeWord Phase 2/3 +
  MoE subset）も **同 pattern**
- WF1 wave の 8 commit は本 規律に従い per-crate verify で land 済、integrated
  verify は本 branch の PR merge 前に CI 側 workflow で発火予定

**Python サブツリー**（`tools/parity/*/`）は uv-managed venv で per-tree 隔離
（[[feedback-python-uses-uv]] + [[feedback-python-3-12]]）ゆえ runtime `Cargo.lock`
に影響しない = 本 規律の対象外（runtime memory footprint は Python プロセスの
別集計、Rust rustc 並列とは orthogonal）。

**教訓**: `cargo test --workspace` は「便利」な習慣だが、16GB 機では GPU
feature を含む monorepo では即 OOM 発火経路。**per-crate 走査 + `CARGO_BUILD_JOBS=1`
+ GPU feature 排他** の 3 点セットを規律化することで、local 開発の continuity
を維持しつつ integrated verify は CI 側に委譲する分業が成立する。

---

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- 本 wave の handoff docs:
  - `docs/handoff/vast-ai-publish-voxcpm2-2b.md`
  - `docs/handoff/vast-ai-publish-rmvpe.md`（WF1 で real forward 実装済 + Path B
    dumper 追加 = §4.2 handoff #2 参照）
  - `docs/handoff/vast-ai-vocoder-gpu-kernels.md`（WF1 で mimi_rvq Metal 初 op
    land = §4.2 handoff #3 参照）
- CLAUDE.md 「現在のタスク状態」= 前回 wave（PR #28 merged 2026-08-12）+ 本 wave
  との継承関係
- 設計 spec:
  - `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`（VoxCPM2-2B）
- CI workflows:
  - `.github/workflows/parity-tts-continuous-vae-real.yml`（VoxCPM2-2B、既 2B
    pin 待機中）
  - `.github/workflows/parity-rmvpe-real.yml`（RMVPE、owner-driven flip switch
    待機中、WF1 で real forward + Path B dumper が land 済のため flip 準備完了）
- Memory: [[feedback-large-models-on-vast-ai]] / [[feedback-license-signoff-primary-source]] /
  [[project-m4-implementation]] / [[project-huggingface-vokra-publication]] /
  [[reference-vast-ai-hf-config-pth-shim]] / [[reference-huggingface-hub-lt-030-vast-ai]] /
  [[feedback-python-uses-uv]] / [[feedback-python-3-12]]
