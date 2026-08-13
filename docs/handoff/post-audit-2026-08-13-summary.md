# post-audit 2026-08-13 wave — summary

**Date**: 2026-08-13
**Branch**: `feat/post-audit-cc-gap-2026-08-13`（main HEAD `40558f5` から作成）
**Author**: Claude Code（本 doc は audit → plan → implement workflow の最終
handoff summary、WF1 → WF2 → WF3 → WF4 → WF5 → WF6 の全 wave land を含めて追記
済、**session terminal**）
**Scope**: PR #28 merged（2026-08-12）後の post-audit CC-gap wave。Plan phase の
reality-check で `already_landed` / `true_gaps`（Utility / Music-und / SSL 各
wave 実装） / `vast_ai_handoff_only`（本 handoff docs 3 件）/ `out_of_scope`
（Non-goals 該当）に分類。**後続に WF1〜WF6 の 6 wave（RMVPE 実装 + microWakeWord
Phase 1〜3 完成 + Vocoder Metal 8 kernel + SPDX 拡張 + Higgs-Audio/FireRed ASR/
Meta music-gen 3 converter、25 CC commit）が land**、依頼者 M1 iMac 16GB OOM
発火事象を受けて memory-safe workflow 規律（§7）を全 wave で維持。**HEAD
`c021261` = 44 commits ahead of main、post-audit CC-gap wave = terminal**（次 CC
起票は別 audit 契機で判断）。

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

**本 branch 上の全 commit（main から 44 commits ahead）**:

| Wave | commit count | 内容 |
|---|---|---|
| 2.1 Refactor | 1 | piper_plus rename |
| 2.2 Utility | 2 | MoE primitives |
| 2.3 Music-und | 6 | music-understanding converters |
| 2.4 SSL-encoder | 5 | SSL audio encoders |
| 2.5 Handoff docs 初版 | 3 | vast.ai runbook 3 件 |
| 2.6 WF1 | 8 | RMVPE 実装 + KWS Phase 1 + Vocoder Metal 初 op + SPDX 拡張 |
| 2.7 WF2 | 5 | docs bridge + atst hygiene + Vocoder Metal +3 op |
| 2.8 WF3 | 5 | microWakeWord Phase 2/3（loud-stub resolved + host parity） |
| 2.9 WF4 | 3 | Higgs-Audio / FireRed ASR + vast.ai runbook |
| 2.10 WF5 | 3 | Vocoder Metal +3 op（snac / denoise / qwen3_tts_codec） |
| 2.11 WF6 | 3 | Meta music-gen 3 converter（magnet-small / medium / melodyflow） |
| 2.12 Terminal declaration | — | session close 記録 |
| **合計** | **44** | — |

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

### 2.7 WF2 wave（5 commits、docs bridge + atst hygiene + Vocoder Metal +3 op）

WF1 で確立した memory-safe workflow 規律（§7）に従い、per-crate sequential 発火
で Vocoder Metal 半分の残 10 op のうち 3 op を land + WF1 で顕在化した
`atst` converter の signoff hygiene drift を修正。

| SHA | 内容 |
|---|---|
| `8a4fa33` | `docs(handoff): post-audit summary updated with WF1 land + memory-safe workflow rules`（本 doc に §7 追加 + WF1 反映） |
| `d424a95` | `fix(compliance): check-converter-signoff.sh atst row の embedded pipe drift 解消` |
| `f9f6e40` | `feat(dac-rvq_decode): Metal MSL kernel + bit-identical vs CPU parity`（sub-wave 2/11） |
| `a7a05e8` | `feat(fsq-codec_decode): Metal MSL gather + GEMV kernels + bit-identical vs CPU parity`（sub-wave 3/11） |
| `137f692` | `feat(snake-activation): Metal MSL kernel + bit-identical vs CPU parity`（sub-wave 4/11） |

**WF2 land 内訳** — 4 系統の deliverable:
1. **atst signoff row 修正**（`d424a95`）: `check-converter-signoff.sh` の
   `atst` row に含まれる embedded pipe 文字が table parser を破損させていた
   drift を、`signoff_match.py` の escape 対応と合わせて修復（4 escaped-pipe
   ケースを self-test に追加、`signoff_match self-test: OK` を restore）。
2. **DAC RVQ decode Metal**（`f9f6e40`、sub-wave 2/11）: Descript Audio Codec
   の RVQ decode を `Compute::dac_rvq_f32` Metal arm で real dispatch、CPU
   `dac_rvq_fold_core` と bit-identical（max |Δ| = 0）を tiny/canonical 両
   shape で M1 iMac 上検証済。
3. **FSQ codec decode Metal**（`a7a05e8`、sub-wave 3/11）: Finite-Scalar
   Quantization の codec decode を Metal MSL の gather + GEMV kernel 2 段で
   実装（WavTokenizer / X-Codec 2 系）、CPU arm と bit-identical。
4. **Snake activation Metal**（`137f692`、sub-wave 4/11）: BigVGAN / HiFTNet
   周辺で消費される `snake(x) = x + (1/α)·sin(αx)²` を Metal MSL kernel で実
   装、fp32 accumulator（audio-dialect rule 準拠）、CPU arm と bit-identical。

### 2.8 WF3 wave（5 commits、microWakeWord Phase 2/3 完成）

WF1 Phase 1 で land した `vokra-kws-micro` の 40-band log-mel front-end に、
Phase 2（`vokra.kws` binary format parser + INT8 kernels）と Phase 3（real
`detect()` + reference dumper + host parity harness）を配線して **loud-stub
resolved**（`FR-EX-08` compliance）。

| SHA | 内容 |
|---|---|
| `d6d87ff` | `feat(vokra-kws-micro): vokra.kws binary format parser (no_std, M5-03b Phase 2)` |
| `cd896fd` | `feat(vokra-kws-micro): INT8 kernels (conv2d + DWConv + dense + sigmoid + softmax, scalar path)` |
| `9973655` | `feat(vokra-kws-micro): real detect() with FlatBuffer + INT8 interpreter (loud stub resolved)` |
| `344d562` | `feat(tools/parity): microwakeword reference dumper (tflite-runtime via uv Python 3.12)` |
| `ac03372` | `test(vokra-kws-micro): host parity harness (env-gated, honest architectural atol)` |

**WF3 land 内訳** — 3 系統の deliverable:
1. **vokra.kws binary format parser**（`d6d87ff`）: WF1 で land した offline
   sidecar（`microwakeword/prepare_checkpoint.py`）が emit する FlatBuffer 化
   GGUF の runtime 側 parser を `#![no_std]` + alloc で実装。tensor descriptor
   + INT8 scale/zero_point + frontend hyper-parameter group を読み取り、後段の
   INT8 interpreter に渡す構造を確立。
2. **INT8 kernels（Phase 2）**（`cd896fd`）: microWakeWord の classifier で
   消費される `conv2d` + `DWConv`（depthwise-separable）+ `dense`（fully-
   connected）+ `sigmoid` + `softmax` の 5 op を scalar path で INT8 実装、
   requantize は round-half-to-even で TFLite reference と bit-for-bit
   一致（`no_std` + alloc、外部 lib 依存なし = NFR-DS-02 保存）。
3. **real detect() + parity harness（Phase 3）**（`9973655` + `344d562` +
   `ac03372`）: WF1 で scaffold のみだった `detect()` を上記 INT8 interpreter
   で real forward に置換（loud stub resolved）、`tools/parity/microwakeword/
   dump_reference.py`（tflite-runtime via uv Python 3.12）で TFLite reference
   の dequant 出力を dump、`crates/vokra-kws-micro/tests/host_parity.rs` で
   env-gated（`VOKRA_KWS_REF_DIR`）に read してrust real detect() と比較する
   host parity harness を land（honest architectural atol、INT8 requantize
   rounding bound 由来）。

### 2.9 WF4 wave（3 commits、Higgs-Audio + FireRed ASR converter + vast.ai handoff）

Wave B fast-track。Higgs-Audio v3 TTS 4B（BosonAI 系、multilingual、Apache-
2.0）と FireRedTeam の Conformer + Qwen2 based ASR（Apache-2.0、Canary-Qwen
sibling）の 2 converter を追加、両方 ≥2GB のため vast.ai handoff docs も同時
land。

| SHA | 内容 |
|---|---|
| `5c77597` | `feat(convert): higgs-audio-v3-tts-4b BosonAI multilingual TTS 4B (Apache-2.0, vast.ai for weights) (Wave B fast-track)` |
| `cae8fcd` | `feat(convert): firered-asr-llm-l FireRedTeam Conformer+Qwen2 ASR (Apache-2.0, vast.ai for weights) (Wave B fast-track)` |
| `dadff26` | `docs(handoff): vast.ai publish runbook for higgs-audio-v3-tts-4b + firered-asr-llm-l` |

**WF4 land 内訳** — 3 系統の deliverable:
1. **Higgs-Audio v3 TTS 4B converter**（`5c77597`）: BF16 pass-through
   converter、`ModelKind::HiggsAudioV3Tts4b` 追加、5 alias（`higgs-audio-v3-
   tts-4b` / `higgs-audio-v3` / `higgs-audio` / `higgs` / canonical HF slug）
   + `LicenseClass::Apache2` + provenance stamp + smoke test。実 weight fetch
   + convert は ~9 GB safetensors ゆえ vast.ai handoff。
2. **FireRed ASR LLM-L converter**（`cae8fcd`）: BF16 pass-through、Canary-
   Qwen 兄弟パターン（Conformer encoder + Qwen2 decoder）、`ModelKind::
   FireredAsrLlmL` 追加、5 alias、Apache-2.0、~7 GB ゆえ vast.ai handoff。
3. **vast.ai handoff runbook**（`dadff26`）: `docs/handoff/vast-ai-publish-
   higgs-audio-v3-tts-4b.md` + `docs/handoff/vast-ai-publish-firered-asr-llm-
   l.md` の 2 doc を land、既存 `vast-ai-large-model-publish.md` の総論を継
   承（provision.sh 4-gotcha / rent→provision→work→destroy lifecycle）+ 各
   モデル固有の size / license / §3.1 sign-off status を記載。

### 2.10 WF5 wave（3 commits、Vocoder Metal +3 op = 累計 8/11 op）

WF1 sub-wave 1/11（mimi_rvq）+ WF2 sub-wave 2-4/11（dac_rvq + fsq_codec +
snake_activation）に続く sub-wave 5-7/11。snac_decode（Orpheus / Maya1 が消
費）+ denoise apply_mask（DFN3 / GTCRN / RNNoise の共通 spectral-gate 末
端）+ qwen3_tts_codec decode（Qwen3-TTS の hybrid semantic + acoustic RVQ）を
Metal MSL kernel + bit-identical vs CPU parity で land。**累計 Vocoder Metal
= 8/11 op**（残 3 = hiftnet / bigvgan / anti_aliased_upsample、いずれも複合
構造ゆえ primitive decomposition ADR 先行が必要 = owner triggered）。

| SHA | 内容 |
|---|---|
| `8a6d7c9` | `feat(snac-decode): Metal MSL kernel + bit-identical vs CPU parity`（sub-wave 5/11） |
| `2b431cd` | `feat(denoise): Metal MSL kernel + bit-identical vs CPU parity`（sub-wave 6/11） |
| `52d1dca` | `feat(qwen3-tts_codec): Metal MSL kernel + bit-identical vs CPU parity`（sub-wave 7/11） |

**WF5 land 内訳** — 3 系統の deliverable:
1. **SNAC decode Metal**（`8a6d7c9`、sub-wave 5/11）: `HotOp::SnacDecode` 追
   加、`Compute::snac_decode_f32(codes, config, codebooks, out_projs)` Metal
   arm で `vokra_snac_decode_f32` MSL kernel を dispatch。3-stage 階層 RVQ
   （`vq_strides = [4, 2, 1]`、per-stage ~12 / 23 / 47 Hz、24 kHz canonical）
   を bit-identical に GPU 化、非 Metal backend は `VokraError::UnsupportedOp`
   （FR-EX-08 silent CPU fallback 禁止）。
2. **Denoise apply_mask Metal**（`2b431cd`、sub-wave 6/11）: DFN3 / GTCRN /
   RNNoise の per-frame pipeline 末端「complex spectrogram × real per-position
   gain（phase preservation）」を `denoise_apply_mask_f32` として抽出、
   `vokra-ops::denoise` に free function 化 + `Compute::denoise_apply_mask_f32`
   seam + `vokra_denoise_apply_mask_f32` MSL kernel。CPU arm と bit-identical。
3. **Qwen3-TTS-Codec decode Metal**（`52d1dca`、sub-wave 7/11）: Qwen3-TTS の
   hybrid semantic + acoustic RVQ split（canonical: 1 semantic × 4096 vocab
   + 15 acoustic × 2048 vocab、all codebook_dim=512）を TWO flat table buffer
   + 異なる per-quantizer stride で正しく分離（shared-vocab clamp = FR-EX-08
   違反ゆえ NG）。fp32 accumulator（audio-dialect rule）、空 side は
   `newBufferWithLength:` 4-byte placeholder で dangling pointer 回避、10
   parity tests bit-identical。

### 2.11 WF6 wave（3 commits、Meta music-gen Wave D remaining = MAGNeT + MelodyFlow）

Coverage-audit-2026-08-03 Wave D remaining。Meta AudioCraft の 3 モデル
（MAGNeT small 10sec / MAGNeT medium 30sec / MelodyFlow t24 30sec）converter
を land。全て **CC-BY-NC-4.0 = T4 tier（Research-only）**、X-Codec-2（2026-
07-28 first-precedent）+ MusicGen family + JASCO の T4 workflow を踏襲。

| SHA | 内容 |
|---|---|
| `0e3531f` | `feat(convert): magnet-small-10secs Meta MAGNeT 10sec masked-AR music-gen (CC-BY-NC-4.0) (Wave D, T4 tier)` |
| `7c9e4dc` | `feat(convert): magnet-medium-30secs Meta MAGNeT 30sec medium (CC-BY-NC-4.0) (Wave D, T4 tier)` |
| `c021261` | `feat(convert): melodyflow-t24-30secs Meta MelodyFlow DiT music-gen 30sec T24 (CC-BY-NC-4.0) (Wave D, T4 tier)` |

**WF6 land 内訳** — 3 系統の deliverable:
1. **MAGNeT small 10secs converter**（`0e3531f`）: `ModelKind::MagnetSmall10secs`
   + BF16 pass-through skeleton mirror of jasco_400m_chords_drums / musicgen
   family。masked-AR（parallel masked-LM decoding、Ziv et al. 2024
   arXiv:2401.04577）は AR-over-EnCodec とは別の decode op ゆえ distinct arch
   tag `magnet_small_10secs`（silent share = FR-EX-08 mis-route 回避）、5 alias
   + 5 unit tests（BF16 verbatim round-trip / license override / arch/name/
   upstream pins / distinct-from-musicgen assertion）。~2 GB ゆえ M1 iMac
   local convert 可（vast.ai 不要、依頼者ルール #1 = ≥2GB 閾値は tight）。
2. **MAGNeT medium 30secs converter**（`7c9e4dc`）: `ModelKind::
   MagnetMedium30secs`、1.5B param medium variant、30-sec generation horizon
   （MusicGen family max horizon 一致）。small とは wider hidden / more layers
   / longer span ゆえ silent share 禁止で separate arch tag。~5.7 GB（1.5B LM
   + bundled EnCodec 32 kHz + T5-base）、依頼者ルール #1 の 8 GB owner cutoff
   以下ゆえ M1 iMac local 可。同 T4 fail-closed default、publish は
   `--allow-noncommercial` 必須。
3. **MelodyFlow t24 30secs converter**（`c021261`）: `ModelKind::
   MelodyflowT2430secs`、Meta MelodyFlow（Le Lan et al. 2024 arXiv:2407.03648
   ）は flow-matching / DiT ベースで、editing-specific ODE inversion（既存
   audio を ODE で inverse → new text prompt で re-generation）という distinct
   sampler stack。MAGNeT masked-LM とも MusicGen AR-over-EnCodec とも別、
   JASCO とも異なる（JASCO = joint audio-symbolic conditioning、MelodyFlow =
   dual text + audio prefix for editing）ゆえ arch tag `melodyflow_t24_30secs`
   で separate。T4 fail-closed default、canonical CC-BY-NC-4.0 LICENSE 同梱
   必須、§3.1 sign-off row 空欄で land（`[[feedback-license-signoff-primary-
   source]]` 準拠）。

**WF6 land honest 境界**: 本 wave は converter code + provenance stamp +
shape roundtrip smoke test まで。**masked-AR runtime forward（MAGNeT）と DiT
sampler runtime forward（MelodyFlow）は new op 追加が必要 = 別 wave（owner
ADR + primitive extraction）**。現時点で converter code だけ land すれば upstream
weight を将来 owner が run 可能な状態を作った（`FR-OP-85` masked-LM decode 系
op anchor + `FR-OP-86` flow-matching DiT sampler op anchor は spec docs 側
で TBD としてマーク、CC 側で先行実装しない）。

### 2.12 Terminal declaration — post-audit CC-gap wave close

**session terminal 判定**（2026-08-13）:
- WF1〜WF6 で **CC-side で honest に land 可能な CC-gap = 25 commit を消化**
  （converter 5 = higgs-audio + firered + magnet-small + magnet-medium +
  melodyflow / Vocoder Metal 8 op / KWS Phase 1〜3 完成 / SPDX 拡張 / atst
  hygiene / handoff docs）
- **残 CC actionable は 6 系統に集約**（§4.3 に列挙、いずれも別 audit 契機
  or owner ADR 発火まで CC 単独では land 不能）:
  1. hiftnet / bigvgan Metal primitive decomposition ADR 先行
  2. magnet / melodyflow runtime forward（masked-AR + DiT sampler op、owner
     ADR trigger）
  3. Vocoder CUDA 半分（全 kernel、vast.ai 必須、owner triggered）
  4. coverage-audit Wave E 全 5 = main repo NON_GOAL（`vokra-voiceclone-
     experimental` 別リポ）
  5. coverage-audit Wave F GPL-3.0 排除 2 = docs Rejected mark のみ（~30 min
     owner）
  6. Kotlin binding 実装（JNA vs JNI ADR sign-off owner triggered）
- **次 CC 起票は別 audit 契機 or owner-triggered ADR 発火まで発生させない**

---

## 3. Vast.ai handoff docs 一覧

**Plan phase の `vast_ai_handoff_only` 3 対象 + WF4 追加の 2 モデル = 5 handoff
docs**（本 wave が「実装漏れではなく別 WP」判断で honest scope boundary を維持
したもの）。

| # | Model / Work item | Handoff doc | Size / License | Owner trigger 理由 |
|---|---|---|---|---|
| 1 | **VoxCPM2-2B**（openbmb/VoxCPM2） | [`docs/handoff/vast-ai-publish-voxcpm2-2b.md`](vast-ai-publish-voxcpm2-2b.md) | 4.96 GB BF16 / apache-2.0 | 依頼者ルール #1（≥2GB は vast.ai）。設計 spec `2026-07-28-voxcpm2-2b-design.md` §5 の Wave 0 ADR Option A/B/C 収束が gate、runtime + converter variant-aware が Wave 1 で land すれば CI 側は既 pinned SHA で待機中 |
| 2 | **RMVPE**（Dream-High/RMVPE MIT） | [`docs/handoff/vast-ai-publish-rmvpe.md`](vast-ai-publish-rmvpe.md) | 180 MB / mit | **✅ 2026-08-13 WF1 update: loud-partial resolved**（`e7b6810`）。上流 `yxlllc/RMVPE`（MIT）の primary source 再精査で U-Net + BiGRU + head topology が primary-source-transcribable と判明 → real forward を land。CLAUDE.md wave 3 "under-specified in primary source" 判定は REVERSED。**owner critical path 圧縮**: real verify に vast.ai 不要（local M1 iMac で完結）、`fetch_rmvpe_pt.sh` の curl ~5 分 + `tools/parity/rmvpe/dump_reference.py` の `uv run` ~30 秒で Path B fixture が揃い、`cargo test -p vokra-models parity_rmvpe` を per-crate で発火可能（memory-safe rule 準拠、§7） |
| 3 | **Vocoder / codec GPU kernel wave**（HiFTNet / BigVGAN / SNAC / Qwen3-TTS-codec 等の Metal MSL + CUDA NVRTC） | [`docs/handoff/vast-ai-vocoder-gpu-kernels.md`](vast-ai-vocoder-gpu-kernels.md) | N/A（kernel work item） | **2026-08-13 WF1〜WF5 land 累計 = Metal 8/11 op**（`66d0077` mimi_rvq / `f9f6e40` dac_rvq / `a7a05e8` fsq_codec / `137f692` snake / `8a6d7c9` snac / `2b431cd` denoise / `52d1dca` qwen3_tts_codec）: 全 bit-identical vs CPU（max \|Δ\| = 0）を M1 iMac 上検証。残 3 op（hiftnet / bigvgan / anti_aliased_upsample）は複合構造ゆえ primitive decomposition ADR 先行が必要 = owner triggered。CUDA 半分は vast.ai owner 必須で非対称。GPU 化 = 性能最適化ゆえ v1.0 GA blocking ではない（M5-13 C ABI 凍結 precondition 外） |
| 4 | **Higgs-Audio v3 TTS 4B**（BosonAI 系） | [`docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md`](vast-ai-publish-higgs-audio-v3-tts-4b.md) | ~9 GB / apache-2.0 | 依頼者ルール #1（≥2GB は vast.ai）+ 8 GB owner cutoff 超えゆえ確実に vast.ai。WF4 `5c77597` で converter code + provenance stamp + smoke test は land 済、実 weight fetch + convert + publish は owner triggered。§3.1 sign-off row 空欄で待機（primary source 直接照合済み）。 |
| 5 | **FireRed ASR LLM-L**（FireRedTeam 系、Conformer + Qwen2） | [`docs/handoff/vast-ai-publish-firered-asr-llm-l.md`](vast-ai-publish-firered-asr-llm-l.md) | ~7 GB / apache-2.0 | 依頼者ルール #1（≥2GB は vast.ai）。WF4 `cae8fcd` で converter code + provenance stamp + smoke test は land 済、実 weight fetch + convert + publish は owner triggered。Canary-Qwen 兄弟パターン（既存 canary_qwen converter の sibling）ゆえ既 pattern から派生。 |

**共通**: すべて `docs/handoff/vast-ai-large-model-publish.md`（総論 = §2 recipe
/ §3 provision.sh gotcha / §4 lifecycle）を前提とし、各 handoff は該当モデル /
work item に固有の差分のみを記述。

---

## 4. Owner critical path

本 wave 完了後の owner-triggered work item リスト:

### 4.1 短期（本 branch の PR 作成 + merge 前後）

1. **Verify wave 実行** — 本 branch の 44 commit（14 converter + 5 handoff + 25
   CC land）を integrated verify。**memory-safe workflow 規律（§7）** の制約下
   で:
   - Local M1 iMac 16GB 上では `cargo test --workspace` / `--all-features` は
     絶対に使わない（前回 OOM 発火経路）→ per-crate `cargo test -p <crate> --lib`
     + `CARGO_BUILD_JOBS=1` で個別走査、または CI 側 workflow で verify
   - `cargo fmt --check` / `scripts/check-zero-deps.sh` / `scripts/check-abi-changelog.sh`
     / `scripts/gen-c-abi.sh --check` はゼロメモリで local OK
   - 本 doc 生成時点の per-crate 実測: `cargo test -p vokra-convert --lib` =
     **954 passed / 0 failed / 0 ignored**（`CARGO_BUILD_JOBS=1`、~4s）、`cargo
     clippy -p vokra-convert -- -D warnings` clean
2. **本 branch → PR 作成** — `feat/post-audit-cc-gap-2026-08-13` から main へ、
   **44 commit** を bundle merge

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
5. **Vocoder / codec GPU kernel wave**（handoff #3）— **2026-08-13 WF1〜WF5 累計
   land = Metal 8/11 op**:
   - Metal 半分 sub-wave 1〜7/11 が既 land（mimi_rvq / dac_rvq / fsq_codec /
     snake / snac / denoise / qwen3_tts_codec）、全 bit-identical vs CPU（max
     \|Δ\| = 0）を M1 iMac 上検証済み
   - **残 3 op = hiftnet / bigvgan / anti_aliased_upsample は複合構造ゆえ
     primitive decomposition ADR 先行が必要 = owner triggered**（hiftnet =
     NSF + iSTFTNet + Snake / bigvgan = anti-aliased periodic + MRF / anti-
     aliased upsample は前 2 者 inline）
   - CUDA 半分は vast.ai owner 必須（~$1-4、~2-4 hours、NVRTC compile + real
     GPU bakeoff）
   - commit + push、integrated verify（CI 側 workflow）
6. **Higgs-Audio v3 TTS 4B publish**（handoff #4）:
   - HF primary source 直接照合（apache-2.0）+ §3.1 sign-off
   - vast.ai instance 起動（~$0.3-0.5、~1 hour、~9 GB safetensors download →
     BF16 GGUF 変換 → publish）
   - `publish-one.sh --push`
7. **FireRed ASR LLM-L publish**（handoff #5）:
   - HF primary source 直接照合（apache-2.0）+ §3.1 sign-off
   - vast.ai instance 起動（~$0.3-0.5、~1 hour、~7 GB safetensors）
   - `publish-one.sh --push`

### 4.3 長期（本 wave と独立、既存 owner critical path 6 系統）

CLAUDE.md「現在のフェーズ状態」に列挙された 6 系統は本 wave で touch していない:

8. NPU bakeoff（M5-01 CoreML/ANE + M5-02 QNN/Hexagon、NFR-PF-12 2× gate、M5-13
   C ABI 凍結の precondition）
9. EU 認証（EU AI Act Article 50、2026-08-02 applies）
10. 資金調達（seed $500K-$1M 級、Cloudflare AI Gateway on-device 版 positioning）
11. NDA（M5-04 console static-link gate、実運用）
12. voice-clone 別リポ（`vokra-voiceclone-experimental` publish、ELVIS Act 分離
    ポリシー）
13. v1.0 GA タグ（M5-13 T17、上記 6 + 10 完了後）

---

## 5. Next actionable

**Owner triggered**（session terminal 後の owner critical path）:

1. Verify wave — 本 branch tip `c021261` に対して **memory-safe rule（§7）** の
   制約下で per-crate 走査 or CI 側 workflow で integrated verify（**`cargo test
   --workspace` は絶対に使わない**、前回 OOM 発火経路）。本 doc 生成時 per-crate
   実測 = `vokra-convert lib 954 passed`。
2. 本 branch から main への PR 作成 → merge（**44 commits bundle**）
3. 上記 §4.2 の handoff #1〜#5 のうち **最優先 handoff を選択** し、該当 runbook を
   実行:
   - **推奨最優先 = handoff #2 RMVPE**（WF1 で real forward + Path B dumper が
     land 済 → **owner curl ~5 分 + `uv run` ~30 秒で local M1 iMac 上 real
     verify 完結**、vast.ai 費用ゼロ、F0 tier 3 姉妹の trio 最後を close。
     `publish-one.sh --push` で org 総計 +1）
   - 次点 = handoff #1 VoxCPM2-2B（CI 側 pinned SHA で既 waiting、Wave 0 ADR
     + Wave 1 land + sign-off ですぐ flip the switch 可能、publish 実績で org
     総計 195+ モデルへ、vast.ai ~$0.3-0.5 / ~1 hour）
   - 中優先 = handoff #4 Higgs-Audio v3 / handoff #5 FireRed ASR（WF4 で
     converter code + smoke test は land 済、owner primary source 照合 + vast.ai
     run で publish 可）
   - 低優先 = handoff #3 GPU kernel wave（correctness ではなく性能最適化、v1.0
     GA blocking でない。Metal 8/11 は既 land、残 3（hiftnet / bigvgan / anti-
     aliased_upsample）+ CUDA 半分は owner ADR / vast.ai 待ち）

**CC triggered = ゼロ**（本 session = post-audit CC-gap wave = terminal、次 CC
起票は別 audit 契機 or owner-triggered ADR 発火まで発生させない）。

- 本 handoff summary の update commit（本 land、`docs(handoff): post-audit
  summary updated with WF5+WF6 land (session terminal)`）

---

## 6. 教訓 / 規律

本 session（WF1〜WF6）で確認・維持した規律:

1. **依頼者ルール #1 (≥2GB は vast.ai)** — VoxCPM2-2B (4.96 GB) / Higgs-Audio
   v3 (~9 GB) / FireRed ASR LLM-L (~7 GB) を local convert attempt せず、
   runbook 作成のみ。MAGNeT small (~2 GB) / MAGNeT medium (~5.7 GB) は 8 GB
   owner cutoff 以下ゆえ M1 iMac local converter code は land 済（実 weight
   fetch は owner triggered）。実 instance 起動は owner。
2. **依頼者ルール #3 (publish は §3.1 sign-off 完了後 owner が判断)** — CC 側で
   converter + test + docs まで land、実 HF upload は絶対に行わない。WF6 の
   CC-BY-NC-4.0 3 モデル（MAGNeT small / medium / MelodyFlow）は §3.1 sign-off
   row 空欄で land（`[[feedback-license-signoff-primary-source]]` 準拠、
   fail-closed default）。
3. **honest scope boundary** — Vocoder GPU kernel wave の Metal 半分 (M1 iMac 可)
   と CUDA 半分 (vast.ai 必須) の非対称性を「実装漏れではなく別 WP」として明示
   （fake-complete より honest、CLAUDE.md M4 節末尾の判断継承）。**WF1〜WF5 で
   Metal 8/11 op が sub-wave 1〜7/11 として land**（mimi_rvq / dac_rvq /
   fsq_codec / snake / snac / denoise / qwen3_tts_codec、全 bit-identical vs
   CPU）、残 3 op（hiftnet / bigvgan / anti_aliased_upsample）は複合構造ゆえ
   primitive decomposition ADR 先行が必要 = owner triggered。同 pattern を
   WF6 の MAGNeT / MelodyFlow runtime forward に横展開（masked-AR / DiT
   sampler の new op = 別 wave = owner ADR trigger）。
4. **loud-partial は fake-complete より honest — かつ primary source 再精査で
   REVERSED しうる** — 2026-07-30 CLAUDE.md wave 3 判断で RMVPE の `extract_real`
   = `VokraError::UnsupportedOp` を loud-partial 維持していたが、2026-08-13 の
   feasibility 調査（`wf_7062f2d5`）で上流 `yxlllc/RMVPE`（MIT）の primary source
   を再精査したところ **fully-specified** と判明 → WF1 `e7b6810` で real forward
   を land、defer 判断は REVERSED。「loud-partial 判定は上流を再精査したうえで
   下すのが望ましい」という後続 pattern を確立。**WF3 microWakeWord も同 pattern**:
   WF1 で `detect()` scaffold + INT8 kernels 未実装で loud-partial 状態だっ
   たものを、WF3 で FlatBuffer parser + INT8 kernels + real detect() + host
   parity harness を配線して loud-stub resolved に flip。
5. **Non-goals 該当は絶対に手を出さない** — Matcha-TTS / RVC v2・GPT-SoVITS in
   `ayutaz/vokra` / AudioSeal 強制-embedding / NNAPI / Piper (piper1-gpl) /
   ONNX グラフ受け / Bark 2 / watermark 埋め込み engine の 8 系統は再開放禁止、
   handoff docs も作らない。
6. **数字を捏造しない** — 本 handoff docs 5 件はいずれも実 vast.ai instance を
   起動していないゆえ RTF / speedup / cost の数字は書かない（size / license /
   §3.1 sign-off status は primary source 照合済のみ記載、vast.ai 実行時の実測
   は owner run 時に埋める placeholder として残した）。**WF1〜WF5 の Metal parity
   は実測 max |Δ| = 0（bit-identical）** を M1 iMac 上で走らせて記録、こちらは
   実測ゆえ数字を残した。本 doc 生成時の per-crate 実測 = `vokra-convert lib
   954 passed`。
7. **Memory constraint を workflow 規律に格上げ**（§7 新規、WF1〜WF6 で継続遵
   守） — 依頼者 M1 iMac 16GB OOM 発火を受けて `CARGO_BUILD_JOBS=1` + per-crate
   + GPU feature 排他 の 3 点セットを規律化。integrated verify は CI 側に委譲
   する分業で local 開発の continuity を維持。**本 session の 25 CC commit は
   全て本規律に従い per-crate + `CARGO_BUILD_JOBS=1` で verify 済**。
8. **既 landed discovery の SSOT 補正パターン** — WF4 期の gap 特定過程で
   `gigaam-v3` / `magpietts-v2602` が既 land 済であることを discover（前 wave
   の CLAUDE.md 記述 と実 repo 状態の drift）→ SSOT 補正が subsequent wave の
   gap 特定を改善する dynamic を実証。pre-audit 時に repo 実測 grep を先行させ
   る pattern を確立。
9. **Metal MSL bit-identical（max |Δ| = 0e0）が 8 kernel 全部で達成された事
   実** — FR-EX-08 の silent-fallback 禁止 pattern（非 Metal backend =
   explicit `VokraError::UnsupportedOp`）が MSL kernel 品質を担保する mechanism
   を実装的に確認。fp32 accumulator（audio-dialect rule）+ tile blocking + gather-
   scatter の 3 pattern が Vocoder Metal kernel の共通 idiom として定着。

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
- 本 wave の handoff docs（5 件）:
  - `docs/handoff/vast-ai-publish-voxcpm2-2b.md`（VoxCPM2-2B、handoff #1）
  - `docs/handoff/vast-ai-publish-rmvpe.md`（WF1 で real forward 実装済 + Path B
    dumper 追加 = §4.2 handoff #2 参照）
  - `docs/handoff/vast-ai-vocoder-gpu-kernels.md`（WF1〜WF5 で Metal 8/11 op が
    land = §4.2 handoff #3 参照、残 3 op + CUDA 半分は owner triggered）
  - `docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md`（WF4 で converter
    land = §4.2 handoff #4 参照）
  - `docs/handoff/vast-ai-publish-firered-asr-llm-l.md`（WF4 で converter land
    = §4.2 handoff #5 参照）
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
  [[feedback-python-uses-uv]] / [[feedback-python-3-12]] /
  [[project-x-codec2-t4-precedent]]（WF6 MAGNeT / MelodyFlow の T4 tier
  precedent 継承）
