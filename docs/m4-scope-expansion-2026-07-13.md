# M4 (v1.0-rc — 旧 v1.0 GA) scope expansion candidate list — 2026-07-13 ultracode workflow 由来

**Status**: **判定確定（2026-07-14 依頼者判定）**。GO 8 件は `docs/milestones.md` §8 の WP 一覧に **M4-13〜M4-20** として正式昇格済、見送り 2 件（BIG-4 / BIG-8）は M5 残置。確定内容・事実訂正は下記「判定記録（2026-07-14）」および milestones.md §8「スコープ拡張判定の記録」を参照。本文書の候補記述（§候補一覧 以降）は**判定前の分析記録**としてそのまま保存する。

> **2026-07-14 追記（v-label 再割当 #2）**: 本判定確定後の同日、依頼者決定により **M4 = v1.0-rc / M5 = v1.0 GA（旧 v2.0）** へ再割当（旧 v2.0 までの全スコープを v1.0 とする）。C ABI 凍結（IF-01）は M4-12 → **M5-13**（v1.0 GA タグ = M5 close）へ移動し、v1.0-rc は semver prerelease として Pre-1.0 ABI 政策を継続する。本文書中の「v1.0 GA（= M4 の意味）」は「v1.0-rc」、「M4-12 凍結」は「M5-13 凍結」、「v1.0.x patch」は「v1.0 GA（M5 close）後の patch」と読み替える。判定内容（GO 8 / 見送り 2 / hard gate 6 結論）は本再割当で変更されない。詳細: `docs/handoff/m4-12.md` §(f)、`docs/milestones.md` §2.2「v-label 再割当 #2」。

## 判定記録（2026-07-14 依頼者確定）

| BIG ID | 判定 | 昇格先 / 残置先 | hard gate の結論 |
|--------|------|----------------|------------------|
| BIG-1  | **GO** | **M4-13** | —（gate 対象外、recommend） |
| BIG-2  | **GO** | **M4-14** | —（gate 対象外、recommend） |
| BIG-3  | **GO（分離案）** | **M4-15** | (f) build target 化のみ M4、critical-safe market claim は M5-08/M5-11 残置 |
| BIG-4  | **見送り** | M5-07 残置 | (b) sherpa-onnx 既対応で差別化にならず、owner モデル検収 backlog 保護（Kill switch L 予防） |
| BIG-5  | **GO（縮小）** | **M4-16** | —（gate 対象外、adjust どおり wfst_decode + SynthID は M5-06 残置） |
| BIG-6  | **GO** | **M4-17** | (e) standing runner は買わない — cloud VM per-run（M2-14 と同一 owner オペ）で advisory 開始、M2-14 standup 時に AVX-512 対応 instance を選べば 1 台兼務 |
| BIG-7  | **GO（UTMOS 先行）** | **M4-18** | (d) weight + license が M4 kickoff 週までに揃わなければ v1.0.x patch へ自動 defer、DNSMOS は license fail-closed |
| BIG-8  | **見送り** | v1.0.x patch or M5-06 | (a) drop reversal はしない — `docs/handoff/m4-12.md` §(e)-4 の patch-release 条項 land 済で ABI freeze regret が解消済、コストは conditional 中最大 |
| BIG-9  | **GO（scope-guard 付き）** | **M4-19** | (c) scope-guard 3 点で GO: DoD は protocol-level テストまで（M5Stack 実機 optional・HA 採用は exit criteria 外）／community engagement は CC land 後に own pace／faster-whisper 互換は behavioral parity まで（bit-exact 非目標） |
| BIG-10 | **GO** | **M4-20** | —（gate 対象外、adjust どおり trigger-backed subset のみ） |

**事実訂正（ratification 時の SoT 照合で発見、いずれもコスト減方向で判定に影響なし）**:

1. **BIG-2「small/medium = unplanned」は誤り** — `docs/milestones.md` §6 M2-06 行は当初から Whisper **small/medium/large-v3/turbo** の 4 サイズを WP スコープに含み、large-v3 のみ完了の部分完了状態。M4-14 は新規 WP ではなく **M2-06 carry-over の完了 WP**。
2. **BIG-9「未 scope」は誤り** — **FR-SV-05 Wyoming Protocol サーバは M2-09 で基本実装済**（event loop + info reply の unit-level 契約テスト 120+16 green、milestones.md §6 M2-15 追記 / commit `0bb73bb`）。M4-19 は新規実装ではなく **completion + barge-in 配線 + protocol e2e** で、wave 見積は 2-3 → **1-2** に減少。
3. **BIG-8「EU AI Act Article 50 enforcement 既経過」は誤り** — 適用開始は **2026-08-02** で本文書作成時点（2026-07-13）では未到来。M4 成果物はいずれも適用開始日より後に出るため判断への影響なし（§BIG-8 本文も訂正済）。

**確定後の総見積**: GO 8 件 = **12-18.5 waves**（BIG-9 訂正込み）。次のアクション = `docs/tickets/m4/` に M4-13〜M4-20 の 30 分単位 tickets を起票（M4-01〜M4-12 は M4 正式 kickoff 時に rolling wave 起票）。

**Position**: Vokra の M3 (v0.9) CC 実装が terminal 到達（`docs/m3-owner-verification-checklist.md` / PR #4 merged 2026-07-11、main HEAD `1f934da`）した直後の中間 review。実測 CC velocity（M0=2 日 / M1=1 日 / M2=4 日 for 11/15 WP / M3=~5-9 日 for 17 WP）と critical path が完全に owner ボトルネックに移行した事実を踏まえ、M5 の pull-forward 可能項目と unallocated 音声特化 op 完成を M4 スコープに追加するかを整理した。

---

## Meta

| field | value |
|-------|-------|
| Report date (UTC)     | `2026-07-13` |
| Owner                 | `ayutaz`（判定担当） |
| Trigger               | 依頼者依頼「開発速度が思ったよりも早いので v1 までに入れられるものを増やしたい。他にどのようなものがあるのかを徹底的に調査して大きな項目にまとめてほしい」 |
| Workflow              | ultracode `wf_53683532-11c`（22 agents / 8 discovery + synthesize + 12 verify + present、~34 分） |
| Vokra baseline commit | `b1606b6`（feature branch `docs/abi-changelog-name-fix`、post-PR-#4 sync） |
| Related SoT           | `docs/milestones.md` §8 (M4)・§9 (M5)、`docs/deliverables.md`、`docs/tickets/m3/README.md`、`docs/handoff/m4-12.md`、`CLAUDE.md`（gitignore local SSOT） |

---

## 判定に用いた設計制約 red-line（不変条件）

- **FA v3 前倒し禁止**（設計制約 §5-(7)、既に M4-07 内スコープ）。Hopper WGMMA/TMA 特化ゆえ Ampere/Ada 顧客層で恩恵薄、v1.5+ 押し出しは Kill switch 判断ではなく設計制約
- **NNAPI 恒久非対応**（Google 公式が Android 15 で deprecated、cargo-deny で恒久 ban）
- **CC-BY-NC / CC-BY-NC-SA weight の公式 zoo 除外**（F5-TTS / Fish-Speech / EnCodec pretrained）、engine op のみ対応で research flag 分離
- **GPL/LGPL 依存禁止**（Unity/Godot 配布経路）、eSpeak-NG / Piper (OHF-Voice/piper1-gpl) 恒久非対応
- **zero-dep NFR-DS-02 保存**（root Cargo.lock は `vokra-*` のみ、生 FFI / dlopen / 除外 workspace）
- **FR-EX-08 no silent CPU fallback**（明示エラー）
- **M4-12 C ABI 凍結後の追加は semver major bump のみ許容** → v1.0 で焼き込む API surface は選別必須。凍結前の追加は backward compat additive 前提

---

## 候補一覧

| BIG ID | 名称 | カテゴリ | 現在 | 提案 | Verdict | 必要 wave | 起源 |
|--------|------|---------|------|------|---------|-----------|------|
| BIG-1  | Vulkan バックエンド完成（M3-02 partial 消化） | backend | M3 partial ~70% | M4 早期 | **recommend** | 2-3 | M3-02 未達分の M4 完成 |
| BIG-2  | Whisper family（turbo + small + medium） | model | M2-06 残 + unplanned | M4 | **recommend** | 1（+0.5 atol） | shape-driven converter 拡張 |
| BIG-3  | Vulkan-only ビルド target 化（M5-08 subset） | infra | M5-08 | M4 縮小 + M5 残置 | **adjust** | 1-2 | M5-08 の CC 完結部分 |
| BIG-4  | Matcha-TTS（M5-07 subset、clean MIT のみ） | model | M5-07 | M4 | **conditional** | 2-3 | M5-07 3 モデル中 MIT 1 件 |
| BIG-5  | FSQ codec family（M5-06 subset、wfst_decode + SynthID は M5） | op | M5-06 | M4 | **adjust** | 1-2 | M5-06 の RVQ 延長分 |
| BIG-6  | CPU ISA サーバ層拡張（AMX 除外版） | backend | 未着手 | M4 | **adjust** | 3 | Q3 決定サーバ市場 pitch material |
| BIG-7  | UTMOS + DNSMOS 評価解除 | op | M1-09b BLOCKED | M4 | **conditional** | 1-2 | Kill switch I 判定必要 |
| BIG-8  | Watermark 復活（AudioSeal + C2PA、SynthID は評価のみ） | governance | 全期間 deferred（2026-07-04 drop） | M4 | **conditional** | 3-4 | EU AI Act Article 50 遵守復元 |
| BIG-9  | Wyoming Protocol Server（HA Voice / Rhasspy 後継） | integration | 未 scope | M4 | **conditional** | 2-3 | Kill switch J exit path 実体化 |
| BIG-10 | Audio dialect op subset（trigger-backed のみ） | op | unallocated | M4 | **adjust** | 2-3 | catalogue 埋め合わせ、C ABI 凍結前 anchor |

---

## 詳細

### BIG-1 Vulkan バックエンド完成（recommend、2-3 waves）

M3-02 未達分（T14-T22 実 kernel の glslc precompile、T27-T29 graph-executor 他 op arm、T33/T34 Whisper base parity CI、T37-T40 Android real device）を M4 早期に押し込む。BIG-3 critical-safe SKU の直列前提。

- **CC タスク**: T27-T29 graph-executor Vulkan arm 拡張（gemm/layernorm/gelu/softmax）、SPIR-V dispatch chain 拡張、coop-matrix ext 無し時の FR-EX-08 明示エラーパス、gpu-vulkan-parity.yml lavapipe 経由の advisory 運用継続
- **依頼者タスク**: LunarG SDK / glslang install → T14-T22 の precompiled `.spv` commit、Snapdragon 8 Gen 3（Adreno 750）+ Dimensity 9300（Mali G720）実機 soak（M3-18 と併走）、gpu-vulkan-parity.yml 初回 workflow_dispatch
- **リスク**: coop-matrix 無し時のスカラー fallback で Whisper RTF 2x 劣化なら subgroup INT8/FP16 kernel 追加 wave 必須、lavapipe（CPU-side icd）は driver-level GLSL→SPIR-V コンパイラバグを catch できないため実機 soak は M4 exit hard gate、gpu-vulkan-parity.yml は数週の連続 green 経過後にのみ required-check へ promote

### BIG-2 Whisper family（turbo + small + medium）（recommend、1 wave + 0.5 atol）

M2-06 の shape-driven converter + tokenizer vocab 埋込機構が全サイズ共通、追加コスト極小。Kill switch A（Unity Sentis）+ E/F（HF/Candle）に対する予防的拡幅、v1.0 GA で「サイズ × backend × platform × on-device」タプルの網羅性を確保。

- **CC タスク**: `vokra-cli convert` を small/medium/turbo で 3 回実行、parity-whisper-real.yml matrix に 3 サイズ追加、`docs/license-audit.md` を 1 行拡張（3 サイズ共通の MIT/MIT）
- **依頼者タスク**: 3 サイズの weight license 一括 sign-off、各サイズの実 checkpoint 変換 owner test（openai-whisper `tests/jfk.flac` 流用可）、workflow_dispatch 起動
- **リスク**: turbo は distilled 809M（decoder 4 層）で数値挙動が base/large-v3 と異なり per-tensor atol 個別 calibration 必要の可能性（Kokoro PROSODY_F0_ATOL 型 architectural bound を rustdoc + ADR + CI YAML に redundantly 記録するのは必須、fabricated pass 禁止）、small/medium は n_state 中間サイズ（768/1024）で shape-driven converter が未 exercise の bug surface を露出する可能性

### BIG-3 Vulkan-only ビルド target 化（M5-08 subset）（adjust、1-2 waves）

`--no-default-features --features vulkan` の CI matrix 追加 + LICENSE/NOTICE 分岐 + SBOM (SPDX) 生成。`vokra-critical-safe` の市場ポジショニング（医療/車載/軍事 SKU claim）は M5-08 / M5-11 に残置（B2B から SLA / ISO 26262 / IEC 62304 相当の追加要件流入回避）。BIG-1 完成の直列依存。

- **CC タスク**: `--no-default-features --features vulkan` target の CI matrix 追加、差分 LICENSE/NOTICE テンプレ、`cargo-cyclonedx` を `integrations/vokra-sbom-tool` 除外 workspace または `scripts/` 隔離で SBOM 生成、v1.0 GA release notes は「CPU + Vulkan-only build target」表記（critical-safe の語は排除）
- **依頼者タスク**: Vulkan（BIG-1）の owner 側 T14-T22 / T33-T34 / T37 完了、SBOM の reproducible-build verify、市場ポジショニング（医療/車載/軍事 SKU claim）は M5-08 に残置し M5-11 商用契約と bundle
- **リスク**: Vulkan 未完成のまま「Vulkan-only SKU」提供は動かない SKU リリース、SKU 検出 API（`vokra_sku_kind` 等）を M4-12 C ABI 凍結に含めず v1.1 additive で温存（ABI freeze regret 回避）、market claim を先行させると B2B から追加要件が M4 期間中に流入

### BIG-4 Matcha-TTS（M5-07 subset、clean MIT のみ）（conditional、2-3 waves）

M3-05 flow_sampler + M3-07 hifigan_generator + M3-08 length_conditioning + M3-17 prosody_control API が全て land 済で追加コスト model integration + parity のみ。M5-07 の Bark / StyleTTS 2 は license blocker で M5 継続。

- **CC タスク**: `matcha::text_encoder` + `matcha::flow_matching_decoder`（M3-05 再利用）+ `matcha::vocoder`（M3-07 再利用）+ GGUF converter + `vokra.matcha.*` metadata + parity-matcha-real.yml scaffold、phoneme adapter が要る場合 +0.5-1 wave
- **依頼者タスク**: MIT/MIT license sign-off、official checkpoint 変換 owner test、UTMOS/MEL loss 実測（BIG-7 依存）、Matcha reference eSpeak-NG G2P と piper-plus G2P の phoneme set 互換性事前 verify
- **conditional 3 条件**: (1) 差別化 rationale の owner 再確認 = sherpa-onnx が既に native 対応済ゆえ 4th TTS の必要性、(2) owner backlog 上限管理、(3) phoneme set 互換性 offline verify
- **リスク**: Matcha text encoder（Transformer + LSTM 混合）が Kokoro BiLstm1d 型 SGEMM byte-order 依存 architectural bound を再発すれば per-tensor atol calibration が必要、`vokra.matcha.*` metadata schema を M4-12 で frozen 化するか EXPERIMENTAL 化するかの status marker が要る

### BIG-5 FSQ codec family（M5-06 subset）（adjust、1-2 waves）

M4-04 で RVQ 完成の自然な延長で FSQ family を「音声 codec 完全対応」landmark 化。wfst_decode は decoder subsystem で独立 2-3 waves 必要、domain adaptation 用途は M5 の critical-safe SKU 実装期に spec 成熟後 freeze が regret 少ない。SynthID は Google 契約可否ゆえ切り出し。

- **CC タスク**: wavtokenizer_vq（65k+ vocab 単段 GEMV bound）+ xcodec2_fsq（RVQ とは別サブグラフ）+ `vokra.wavtokenizer.*` / `vokra.xcodec2.*` metadata + model zoo entry 追加
- **依頼者タスク**: WavTokenizer (MIT) + X-Codec 2 (MIT + Apache 2.0 dual-license、per-file か combined か明確化) の license sign-off、CosyVoice2 / Voxtral / Mimi の M3 sign-off queue と統合
- **リスク**: FSQ 65k+ vocab embedding API + FSQ vocabulary encoding が M4-12 で semver lock される（wfst_decode を切ることで API surface を 1 系統に集約）、SynthID を切り離した場合 landmark claim が partial になる可能性

### BIG-6 CPU ISA サーバ層拡張（AMX 除外版）（adjust、3 waves）

Q3 決定（2026-07-02、サーバサイド市場明示化）以降、AVX2 tier のみで Cascade Lake+ Xeon / EPYC 7003+ の INT8/BF16 主力パスが空白 = 理論性能の 20-40% しか出せず、Cartesia/Deepgram on-prem 差別化 pitch material の実弾を欠く。`std::arch` のみで zero-dep 保存。

- **CC タスク**: `CpuFeatures` probe 拡張（cpuid EBX/ECX + AT_HWCAP/HWCAP2）、AVX-512F/DQ/BW/VL + VNNI + BF16 kernel、AVX-VNNI 256-bit（Alder Lake+ E-core）、ARM64 fp16 arithmetic + dotprod + i8mm + bf16、K-quants dequant fusion、`selftest.rs` による SIGILL 予防、`IsaPath` enum に `#[non_exhaustive]` + C header に予約範囲宣言
- **依頼者タスク**: Skylake-X / Ice Lake Xeon（AVX-512 VNNI/BF16）self-hosted runner を M2-14（CUDA RTX 4090）と同一 owner オペレーションで 1 台追加、Apple M2+ / Pi 5 Cortex-A76 / Graviton3 実機での bench 検証
- **なぜ AMX 除外**: AMX-TILE/INT8/BF16 は Rust stable での intrinsic 供給未確定 + Sapphire Rapids+ の real-hardware soak 期間確保困難で M5 に切り出し。AMX-FP16 / AVX10.1/10.2 は v1.5+ anchor 済ゆえ本 BIG-6 とは別
- **リスク**: AVX-VNNI 256-bit の tier 選択ロジックが Alder Lake+ hybrid CPU で E-core / P-core / Xeon / Zen4 の 4 分岐まで膨らむ可能性、`IsaPath` enum を M4-12 で凍結する前に予約範囲を `docs/abi-changelog.md` M3-16 preparation section に predetermined 記録し M5 SME / RVV Zvfh 拡張 / AMX を backward compat additive にできる状態にする

### BIG-7 UTMOS + DNSMOS 評価解除（conditional、1-2 waves）

M1-09b は AudioMosMetric trait のみ landed で weight 待ち BLOCKED。Kill switch I red-line 自体が「MEL loss / UTMOS が 5% 以上劣化」を要求しているのに、M1-13 は mel_loss 単騎で通過している。v1.0 GA で CosyVoice2 + CSM + Moshi + Matcha-TTS の 4 モデルを mel_loss 単騎 gate で出荷するのは honest-negative。

- **CC タスク**: UTMOS（wav2vec2 SSL + regression head）+ DNSMOS P.835（small CNN）の native inference、`vokra.utmos.*` / `vokra.dnsmos.*` metadata、AudioMosMetric trait 実装 + degradation.rs 配線、CI 統合、per-tensor atol は Kokoro T17-fixup precedent 準拠
- **依頼者タスク**: UTMOS weight source URL + license sign-off（`docs/license-audit.md` に CosyVoice2/Voxtral row template で追加）、DNSMOS Microsoft P.835 license 検証（research-only 制約なら UTMOS-only fallback へ scope 縮小）
- **conditional 条件**: M4 kickoff 週までに weight + license 揃わなければ v1.0.x patch へ defer、UTMOS/DNSMOS 各々を個別に scope reduce 可能
- **リスク**: UTMOS reference は SaruLab MOS Challenge 2022（合成音声）domain で、Mimi codec + streaming（CSM/Moshi）に対する score calibration が miscalibrated の可能性 → 実 sample での相関検証で advisory-only 判定になる場合あり（Kokoro PROSODY_F0_ATOL と同じ honest engineering）、DNSMOS license blocker なら UTMOS 単独 gate に scope 縮小

### BIG-8 Watermark 復活（AudioSeal + C2PA、SynthID は評価のみ）（conditional、3-4 waves、owner reversal 必須）

2026-07-04 依頼者 drop（M1-07）で FR-CP-01 / FR-CP-02 config-only 実装 → M4 で runtime 埋め込みを復活。EU AI Act Article 50（**2026-08-02 適用開始 — 初版の「enforcement 既経過」表記は誤り、本文書作成時点では未到来（2026-07-14 訂正）**）+ California SB 942（2026-01-01 施行済）の marking obligation を engine layer で担う landmark。ただし M4 成果物はいずれも適用開始日（2026-08-02）より後に出るため timing は判断に影響せず、`docs/legal-compliance.md` §1.4 deployer-side visible UI disclosure MUST での暫定運用が既成立。

- **CC タスク**: AudioSeal（Meta MIT reference）Rust port を root workspace（`crates/vokra-ops` + `crates/vokra-models/src/watermark/`）に、C2PA は subset 自前実装 or `integrations/vokra-compliance-c2pa/` 除外 workspace（`c2pa-rs` の openssl/asn1-rs 依存で NFR-DS-02 破綻回避）
- **依頼者タスク**: **2026-07-04 のドロップ判断 reversal**（最大 blocker、CLAUDE.md「M3 確定判断」欄を「M4 で FR-CP-01/02 revive」に書換）、Google SynthID 契約可否、`docs/legal-compliance.md` §1.4 の deployer-side MUST を engine-side MUST に格上げ判断
- **conditional 最重要条件**: owner reversal、代替解 = M4-12 changelog に「WatermarkConfig backend_status は Deferred→Active への遷移を patch release で許容」条項を documentation-only で追加し runtime 実装は M5-06 まで temporize
- **リスク**: AudioSeal PyTorch reference との bit-exact parity は Kokoro T17-fixup 級 architectural bound 発見リスク（fabricated pass 禁止、per-tensor atol + ADR + step summary 3 面 redundant）、M4-05 CSM / M4-06 Moshi の TTS 系実装と bundle しないと cross-cutting change が M4 close を遅延、依頼者が 2 度目 drop をした場合は scope thrash 確定で Kill switch L 加速、SynthID Google 契約は owner-only の商用交渉 = M4 期間内に決着しない可能性

### BIG-9 Wyoming Protocol Server（HA Voice / Rhasspy 後継統合）（conditional、2-3 waves）

現状 Wyoming Server 実装は faster-whisper + piper1-gpl（GPL-3.0 + eSpeak-NG GPL-3.0 二重汚染）主流で、MIT/Apache 2.0 の Rust native は不在 = Vokra が最初の一級選択肢になる差別化 landmark、Kill switch J（HA Voice が Vokra を採用しない意思決定）の pass 材料、exit path 実体化。

- **CC タスク**: Wyoming Protocol Server（TCP + JSON header + binary payload、handcrafted parser で zero-dep 保存）、faster-whisper API 互換 endpoint、piper HTTP API 互換 endpoint、`Stream::interrupt()` の Wyoming barge-in 対応、`integrations/vokra-server` 内に閉じる
- **依頼者タスク**: HA Voice / Wyoming コミュニティ調整 + wyoming-vokra PR 提出（X-05 community engagement）、HA Voice Satellite（M5Stack）実機確保、`docs/deliverables.md` に「Wyoming Server public endpoint は v1.0 semver stability から明示除外（protocol-tracking / experimental tier）」明記、faster-whisper 互換は word-level timestamp + language detection の behavioral parity までとし float32 CE precision の bit-exact は非目標と ticket spec に事前明記
- **conditional 4 hard gates**: (1) X-05 community engagement 1 wave 相当の owner time commit、(2) HA Voice Satellite 実機 or emulator 確保、(3) Wyoming public endpoint の semver 除外 documentation、(4) faster-whisper bit-exact 追求の scope reduce 事前明記
- **リスク**: Wyoming Protocol は HA コミュニティ主導で後方互換保証なし = v1.0 GA 後の protocol drift 追随が indefinite に伸びる（Kill switch L modest risk）、HA コミュニティ採用は political timing で M4 window で確実に決着する保証なし、CC 完成と実 value 実現の gap が reputation risk

### BIG-10 Audio dialect op subset（trigger-backed のみ）（adjust、2-3 waves）

M4-12 で C ABI 凍結すると audio op 追加は semver major bump にしか許容されないが、trigger model 無しに全 catalogue を landing すると未使用 op が semver 責任範囲に半永久残存 = 「機構先行・実体後追い」規律違反。BigVGAN（Kokoro=iSTFTNet、CosyVoice2=Mimi、piper-plus=MB-iSTFT で trigger model 無し）+ CTC/RNN-T（Whisper=beam_search で trigger 無し）+ ECAPA-TDNN/WeSpeaker/TitaNet（CAM++ で covered）+ diarize（trigger + license 二重 blocker）はすべて speculative。

- **CC タスク**:
  - (a) `beam_search` word-level timestamps 拡張（Whisper trigger、v0.1 MVP 以来 in-scope、~0.5 wave）
  - (b) `speaker_verify`（CAM++ trigger、既存 base の minimal API、~0.5 wave）
  - (c) 音声強化 subset = `denoise`（DeepFilterNet MIT）+ `agc` + `hpf` + `loudness_norm`（Discord bot / コールセンター trigger、~1-2 waves）
  - M4-12 ABI changelog に「MechanismRegistry」anchor を残の op に対して先行記録
- **依頼者タスク**: DeepFilterNet MIT + parity reference の owner test、TitaNet（NVIDIA NeMo 家系で NC 制約可能性、BigVGAN と同型 license audit を M5 移行時に）、GTCRN license 事前確認、pyannote diarize の HF gated + 商用条項精査
- **M5 据え置き対象**: BigVGAN chain + CTC/RNN-T + ECAPA-TDNN/WeSpeaker/TitaNet + diarize（M4-12 changelog で mechanism anchor のみ先行記録して backward compat additive を可能にする）
- **リスク**: BigVGAN は Kokoro T17-fixup 級 architectural bound 発見時に 3-4 waves 単独消費のリスク（5-7 waves 全体では最大 8-10 waves upper bound）、trigger 無し op を M4 で landing すると未使用 C ABI symbol が semver 責任範囲に半永久残存で M4-12 discipline 違反、TitaNet の NVIDIA NC restriction 未確認、owner backlog 圧迫で Kill switch L 加速

---

## 前倒しを推奨しない項目（M5 keep or drop）

### Reject（scope creep / mislabeled M5）

- **NPU delegates（CoreML + QNN、M5-01 + M5-02）**: effort 見積 2-3 waves は seam+probe plumbing のみで真の delivery は 8-12 waves、Kill switch A 防衛の正当化が弱く（Metal + Vulkan で競合パリティ既成立）、M4-12 で delegate 選択 API を C ABI に露出させると real-hardware bakeoff なしの API 面確定 = regret 発生、owner backlog Kill switch L 加速。M5-01 / M5-02 に据え置き
- **サーバサイド productization full stack（K8s Helm + OTEL + Grafana + STT→LLM→TTS pipeline）**: OpenAI Realtime API + vLLM 互換は M2-09 で shipped 済 = double-count、K8s Helm + OTEL + Grafana は enterprise 商用 GA（M5-11）と特性 align で M4 OSS DoD と mismatch、STT→LLM→TTS pipeline は M4-05 Moshi / M3-09 CosyVoice2 real HF 完成 downstream。M5-11 商用契約 bundle 継続

### Owner calendar-fixed（M5 keep 恒久）

- **M5-04 Console SDK**（Nintendo/Sony/Microsoft）: NDA プロセス各 1-3 ヶ月 × 3 社の owner-only、CC 圧縮不能
- **M5-05 vokra-voiceclone-experimental full integration**: ELVIS Act / NO FAKES Act tool-distributor liability の owner 法務判断 + repo 名称確定が gate
- **M5-10 EU AI Act 認証取得**: Regulator 3-6 ヶ月固定期間の owner-only
- **M5-11 商用化・資金調達**（seed $500K-$1M）: 3-9 ヶ月 owner-only、GA 見込み成立後
- **M5-12 GA 宣言と DoD 充足確認**: 定義上 M5 末尾、DoD 全 gating 通過後

### Ecosystem / license 恒久 blocker（M5 keep or drop）

- **M5-07 Bark**: Suno 方針で voice cloning 目的の再学習禁止、owner legal + community 判断 v2.0+ 継続
- **M5-07 StyleTTS 2**: weight license 不明（yl4579 issue #117 未決）、fail-closed research flag
- **F5-TTS / Fish-Speech / EnCodec pretrained の公式 model zoo 追加**: CC-BY-NC 系で恒久除外、engine op のみ対応
- **NNAPI**: Google 公式 Android 15 で deprecated、cargo-deny で恒久 ban
- **Piper（OHF-Voice/piper1-gpl）/ eSpeak-NG**: GPL-3.0 二重汚染、piper-plus で代替済
- **Discord bot demo / Discord community 復活**: 2026-07-04/06 owner drop 済、GitHub Issues/Discussions に完全置換

### 設計制約 red-line（不変）

- **M4-07 FA v3 の M4 内早期実装**: 既に M4 スコープ内、v1.5+ 前倒し禁止の設計制約 §5-(7) は Ampere/Ada 支配的な顧客層根拠で不変
- **M5-03 IoT Tier 3**（Cortex-M55/M85 + ESP32-S3 + HA Voice PE）: HA Voice PE は remote 前提で実機確保困難、addressable market 縮小
- **AMX-FP16 / AVX10.1/10.2**: v1.5+ anchor 済、silicon 入手困難で BIG-6 CPU ISA 拡張からも意図的除外
- **Cortex-M0-M4 / ARMv6 / Xtensa LX6 / RP2040**（Tier 4）: 恒久非対応
- **CSM（M4-05）+ Moshi（M4-06）の M3 逆前倒し**: M4-03 AEC 先行必須の直列依存

### Owner ADR 待ち（M4 前倒し不可、M4-09 / M4-10 と同扱い）

- **M4-08 RVV 0.7.1 追加リソース投入**: LicheePi 4A / Milk-V Duo owner 機材確保が gate
- **M4-09 piper-plus G2P 方針再評価**: M0-07 分析 + 運用実績投入待ち
- **M4-10 MLIR audio dialect + StableHLO 採否再評価**: レビュアー B 指摘 #13 で却下寄り
- **F0 extraction ops**（RMVPE/FCPE/CREPE/PyIN/Harvest）単独前倒し: M5-05 vokra-voiceclone-experimental と同時扱いが clean

---

## 推奨アクション（依頼者判断待ち、優先順位順）

1. **[CC 即時着手可能]** BIG-1 Vulkan 完成の CC-side 2 waves + BIG-2 Whisper family 変換 + parity CI matrix を M4 kickoff 前に前倒し着手（placeholder-then-swap 戦略で owner glslc 完了に blocking されない）
2. **[owner decision 6 件、M4 kickoff 前 hard gate]**:
   - (a) BIG-8 Watermark 復活の 2026-07-04 drop reversal 判断（reject なら M4-12 changelog に patch-release-allowed 条項の documentation-only 追加）
   - (b) BIG-4 Matcha-TTS の差別化必要性再確認（sherpa-onnx が既に native 対応済）
   - (c) BIG-9 Wyoming Protocol の X-05 community engagement 1 wave time budget commit + HA Voice Satellite 実機確保
   - (d) BIG-7 UTMOS/DNSMOS の weight sourcing ETA と DNSMOS license 検証
   - (e) BIG-6 CPU ISA 拡張の self-hosted Skylake-X / Ice Lake runner 追加判断
   - (f) BIG-3 Vulkan-only build target 化 vs M5-08 critical-safe SKU 分離判断
3. **[owner unblock 継続]** 積み残し owner backlog 消化を M4 準備期間で進行: M3-18 Android/Godot 実機 RTF、M3-19 Kill switch D 判定、M3-11 T19（実 Godot 4.3+ editor verify）+ T20（WP-close PR）、M3-15 real network 75ms 実測、iOS 実機 RTF、M2-14 self-hosted CUDA runner standup、M3-09/M3-10 real HF checkpoint parity。BIG-6 CPU ISA 拡張の Skylake-X runner を M2-14 と同一 owner オペで束ねる
4. **[docs 昇格]** 依頼者判定通過項目のみを本文書から `docs/milestones.md` §8 M4 WP 一覧に `M4-13`〜 の連番で正式昇格。ratify 後は `docs/tickets/m4/` を新設し 30 分単位 tickets を M2/M3 と同型規律で authoring
5. **[docs 更新]** `docs/handoff/m4-12.md` に semver anchor 拡張の documentation を先行 land: (a) HTTP/gRPC/WebSocket API（vokra-server + Wyoming Server）= 「protocol-tracking / experimental tier」明記、(b) `IsaPath` enum + FSQ codec API + Matcha metadata の予約範囲 / EXPERIMENTAL marker、(c) delegate 選択 API（NPU 系）は post-v1.0 additive で M4-12 semver anchor から意図的除外、(d) WatermarkConfig backend_status の Deferred→Active 遷移を patch release で許容

---

## 総見積と honest note

依頼者判定通過項目の全採用時: recommend 2 件（4-5 waves）+ adjust 4 件（7-11 waves）+ conditional 4 件（8-13 waves）= **総 19-29 waves** が M4 pull-forward 候補全体。M4 既存 12 WP + BIG 8-10 件で honest 見積は現実的に選別必須で、依頼者判断で conditional 4 件のうち少なくとも 2 件を reject / defer するのが M4 close date と Kill switch L 予防の観点で妥当。

実測 velocity（M0=2 日 / M1=1 日 / M2=4 日 for 11/15 WP / M3=~5-9 日 for 17 WP）は precedent + Wave 並列化が効いた case であり、backend layer + governance layer + real hardware validation を同時 stacking すると extrapolation 精度が落ちる不確実性を明示的に保持する。特に BIG-8 Watermark 復活は 3-4 waves 見積のうち AudioSeal PyTorch reference との parity が Kokoro T17-fixup 級 architectural bound に相当した場合 +2-3 waves の上振れ余地あり。

---

## 追跡

- **起源 workflow**: `wf_53683532-11c`（2026-07-13、22 agents / 8 discovery + synthesize + 12 verify + present、~34 分、workflow output は local session transcript のみ）
- **本文書の位置付け**: 実装対象ではない Draft。依頼者判定を経て個別項目が `docs/milestones.md` §8 M4 WP 一覧に昇格するまで、`docs/tickets/m4/` の authoring は開始しない
- **関連文書**:
  - `docs/milestones.md` §8（M4 現行スコープ 12 WP）→ 本文書の候補が ratify されると本節に追加される
  - `docs/milestones.md` §9（M5 現行スコープ 12 WP）→ 本文書の候補のうち M5 subset を pull-forward するものは M5 側で「M4 に subset 移動」と annotate される
  - `docs/handoff/m4-12.md`（M4-12 C ABI 凍結準備 handoff）→ 本文書 §推奨アクション 5 の semver anchor 拡張 documentation が先行 land 対象
  - `CLAUDE.md` 現在のタスク状態 → 本文書への短い pointer を追加
