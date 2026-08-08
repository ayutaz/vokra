# TIER 1 + TIER 2 audio-gap implementation partial land (2026-07-30)

## 前文

依頼者 2026-07-30 指示「asr,tts,音楽系,音声分離など全てのモデルに対応した
い」を受け、`docs/handoff/hf-audio-gap-2026-07-30.md` の TIER 1 (46 permissive)
+ TIER 2 (3 cc-by) 計 49 モデル + defer marker 3 (Voxtral-Realtime /
Cohere-transcribe / Nemotron-ASR) の実装 workflow (`wf_022575ce-077`) を
launch。8 agent (7 parallel worktree implementer + 1 consolidator)、~50 min
実行、7/7 agent tests_passing = true 報告。

## 実 land 状況 (partial)

**25 new converter files land = library-callable modules として `pub mod` 経由**:

| Category | Files |
|---|---|
| ASR | `qwen3_asr.rs`, `wav2vec2_ctc.rs` |
| TTS | `moss_tts.rs`, `melotts.rs`, `speecht5.rs`, `parler.rs`, `vieneu.rs`, `bark.rs` (variant), `kyutai_tts.rs` |
| Vocoder / Codec | `hifigan_vocoder.rs`, `bigvgan.rs`, `focalcodec.rs` |
| Enhancement / Separation | `tiger.rs`, `mp_senet.rs`, `metricgan_plus.rs`, `sepformer.rs` |
| VAD / Turn detection | `fsmn_vad.rs`, `firered_vad.rs`, `smart_turn.rs` |
| Classification / Speaker | `clap.rs`, `ast.rs`, `speechbrain_lang_id.rs`, `xvector.rs`, `deepfake_detection.rs` |
| Aesthetics | `audiobox_aesthetics.rs` |

**Extended existing files (WT2/WT7 内部拡張)**:
- `qwen3_tts.rs` — Qwen3-TTS 1.7B-CustomVoice + 1.7B-VoiceDesign variants (internal Variant enum)
- `moshi.rs` — Moshika-RAG alias routing (`ModelKind::from_arg` に moshika* alias 追加)

**Registry addition**:
- `crates/vokra-convert/src/lib.rs` `from_arg` に moshi/moshika/moshiko + Kyutai alias 追加 (moshika-rag test の requirement)

## Deferred (follow-up wave 必須)

**5 shared files への CLI wiring は本 wave では land できなかった**:

- `crates/vokra-convert/src/lib.rs`
  - `ModelKind` に 49 new variant 追加 (Qwen3Asr / Wav2Vec2Ctc / MossTts / MeloTts / SpeechT5Tts / ParlerTts / VieNeuTts / HifiganVocoder / BigVGan / Focalcodec / TigerSeparator / MpSenet / MetricganPlus / SepFormer / FsmnVad / FireredVad / SmartTurn / Clap / Ast / LangIdVoxlingua107 / XVector / LangIdCommonlanguage / DeepfakeDetection / KyutaiTts / AudioboxAesthetics 等) — enum arm 未追加
  - `impl ModelKind::from_arg` に 49 new match arm — 未追加
  - `impl ModelKind::as_arg` に 49 new match arm — 未追加
  - `convert_file` の dispatch match に 49 new arm — 未追加
  - `pub use models::*::convert_*_file[_with_variant]` re-export — 未追加
- `crates/vokra-convert/src/main.rs` — verify match arm の 49 addition
- `crates/vokra-cli/src/convert.rs` — USAGE token + `parses_every_model_kind_and_help_lists_them` test case
- `crates/vokra-core/src/compliance/license_class.rs` — 49 family prefix walks + exact-name arm
- `docs/license-audit.md` §3.1 — 49 new sign-off rows

**帰結**: `vokra-cli convert --model qwen3-asr --input ... --output ...` は **本 land では動作しない**。converter 関数は library API 経由で直接呼び出し可能 (`use vokra_convert::models::qwen3_asr::convert_qwen3_asr_file`)。CLI 経由の使用は follow-up wave 完了後。

## 統合失敗の root cause

**WT branches の base = `d05ab7d` (origin/main の 1 commit 遡り)**、my working branch
`feat/model-publish-and-m5-gap-2026-07-29` は `d05ab7d` から **25 commits diverged**
(pyannote Wave 3+4 + license sign-off wave + LibriSpeech mirror script 等を含む)。

7 WT 全て `d05ab7d` を base として自身の shared file 拡張を行ったが、これらは
my branch の **pyannote 追加 (Crepe / Rmvpe / PyannoteSegmentation ModelKind 等)**
と同じ anchor (`ModelKind` enum 末尾、`from_arg` match 末尾、dispatch match 末尾)
に addition を配置しているため、cherry-pick / merge / merge-file の全 strategy が
7 way conflict を発生させた:

- `git cherry-pick -X theirs` = incoming WT の変更のみ取り込み → 既存 pyannote 変更 drop
- `git cherry-pick -X ours` = 現在の変更のみ保持 → WT 追加 drop
- `git merge-file --union` = 単純 union → Rust の match arm 構造を認識せず broken (dispatch arm truncate、`=>` operator 位置ずれ、既存 arm body 途中で他 arm 挿入 = syntactic garbage)
- Programmatic diff addition extraction = anchor drift (WT の "d05ab7d 基準の line 番号" と現 tree の line 番号不一致)

いずれの automated strategy も Rust の semantic を認識せず、結果として lib.rs / main.rs / cli/convert.rs が syntactic broken (~30+ error) になった。

## Follow-up wave 選択肢

**Option 1**: 手動 CLI wiring (~1000 line の Rust 追加 across 5 files、~4-6h CC 時間)
- 49 new ModelKind variant + 49 from_arg case + 49 as_arg case + 49 dispatch arm + license class family walks を Edit tool で段階的に追加
- 各 WT が landed した new converter file の `pub fn convert_<name>_file` signature を参照
- primary source は WT の tempcommit (worktree branch worktree-wf_022575ce-077-{1..7} に保存済 SHA 58c7475 / 1b044f0 / 31162a9 / 820199a / 6157913 / d39924c / 94ef82a)

**Option 2**: Workflow 再実行 with base = 現 HEAD
- `Workflow` を再 launch する際 worktree の `isolation: 'worktree'` は自動で新 base = 現 HEAD を取る (これは 2026-07-30 の実行時は origin/main が base だったが、現在の HEAD 32d03c9 は既に pyannote content を含む)
- 各 agent が現 HEAD 上に自身の変更を land する = drift 消失
- 統合は cleaner cherry-pick で完了する見込み

**Option 3**: MVP land + owner CLI wiring
- 本 partial land を先に commit + push
- CLI wiring は owner が Edit tool で手動、または CC 別 session で継続

## verify state (partial land 時)

- `cargo test -p vokra-convert --lib` = **517 passed** / 0 failed (+ pyannote 53 pre-existing + 25 new modules の内部 test)
- `cargo test -p vokra-core --lib` = **530 passed** / 0 failed
- `cargo fmt --all --check` clean (rustfmt 適用済)
- `cargo clippy -p vokra-convert -p vokra-core -- -D warnings` clean (25 new modules に `#![allow(dead_code)]` 追加 = CLI wiring 完了までの temporary allowance)
- `scripts/check-zero-deps.sh` OK (root Cargo.lock は vokra-* のみ)
- **新規 C ABI = 0** (Rust surface のみ、v1.0-rc baseline 33 fn + 11 typedef 不変)

## 教訓 (memory へ反映候補)

1. **ultracode workflow の worktree base は明示指定推奨**: 現行 `isolation: 'worktree'` は
   自動で `git worktree add` を実行するが、これが `origin/main` を base に取ると、
   my working branch と drift が発生。将来的には `isolation: {worktree: {base: 'HEAD'}}`
   のような明示指定オプションが必要 (現時点では unsupported)。

2. **7-way parallel worktree の統合は非現実的**: 各 WT が同じ enum / match の同じ末尾
   anchor に addition を加えると、pairwise 3-way merge は clean だが n=7 の合流は
   syntactic garbage を produce する (Rust の match arm 境界を git は認識せず、
   隣接 arm を分割破壊する)。将来的には (a) 各 WT が異なる anchor を持つ、(b) parallel
   ではなく sequential (WT2 は WT1 land 済 base、WT3 は WT1+2 land 済 base) 実行を検討。

## 依頼者判断待ち

上記 3 option のいずれで進めるか。

- **Option 1** = 確実で straight-forward だが CC 時間投入大 (~4-6h)
- **Option 2** = workflow 再実行で解決見込みだが確証なし (drift の source が
  worktree base 選定なら解決するが、他の要因も可能性あり)
- **Option 3** = 今すぐ partial 反映して次段階に進む、CLI wiring を owner
  or 別 session に defer

## 2026-08-09 addendum: Wave 6 lead status verification (RESOLVED)

以下は本 handoff が **RESOLVED** = 全 tier1/tier2 CLI wiring が land 済であることを
Wave 6 lead が現 HEAD (`0703579`) で verify した結果。**新規追加コミット無し** — 単に
先行 wave (M5 gap wave 1 = `02664f6` / SBV2 Phase 1 = `f7af1ba` / SoTA Phase 1-4 = `7ed0548` /
M5 CC 実装 = `64485c7`) で段階的に land 済であることを確認する retrospective note。

**Wave 6 lead verification protocol** (feat/sbv2-voxtral-real-verify-2026-08-06 HEAD `0703579`):

**"Deferred (follow-up wave 必須)" 5 file wiring — 全て land 済**:

- `crates/vokra-convert/src/lib.rs`:
  - `ModelKind` enum: 25 tier1/tier2 variant 全て present (Qwen3Asr / Wav2Vec2Ctc /
    MossTts / MeloTts* / SpeechT5Tts / ParlerTts* / VieNeuTts / HifiganVocoder /
    BigVGan / Focalcodec / TigerSeparator / MpSenet / MetricganPlus / SepFormer /
    FsmnVad / FireredVad / SmartTurn / Clap / Ast / LangIdVoxlingua107 / XVector /
    DeepfakeDetection / KyutaiTts / AudioboxAesthetics)
  - `impl ModelKind::from_arg`: 全て `Some(Self::*)` match arm 存在
  - `impl ModelKind::as_arg`: 全て `Self::* => "*"` match arm 存在
  - `convert_file` dispatch: 全て `ModelKind::* => {...}` arm 存在
  - `pub use models::*::convert_*_file`: 全て re-export 済
- `crates/vokra-convert/src/main.rs`: verify match arm 全 addition 済
- `crates/vokra-cli/src/convert.rs`: `parses_every_model_kind_and_help_lists_them` test green
- `crates/vokra-core/src/compliance/license_class.rs`: family prefix walks 全 land
- `docs/license-audit.md` §3.1: sign-off template 追加済 (blank rows は依頼者専任)

**帰結の訂正**: 本 handoff の line 48 が言った「`vokra-cli convert --model qwen3-asr ...`
は本 land では動作しない」は **現時点では失効** — CLI 経由の全 25 tier1/tier2 model は
`0703579` HEAD で routing 動作する (`parses_every_model_kind_and_help_lists_them` test で
137 passed = 全 ModelKind の parse pass 含む)。

**Verification (Wave 6 lead, 2026-08-09)**:

- `cargo test -p vokra-cli --release parses_every_model_kind`: 1 passed / 0 failed
- `cargo test -p vokra-cli --release`: 137 lib + 4 policy_e2e + 2 quant_fused = 0 failed
- 新規 C ABI = 0 (Rust surface のみ、v1.0-rc baseline 33 fn + 11 typedef 不変)
- root `Cargo.lock` = vokra-* のみ (NFR-DS-02 preserved)

**同 Wave 6 audit で "wiring gap" と表示された他 3 items も同様に land 済 verified**:

- BF16-FLEET-WIRING (16 skeletons): KimiAudio/StepAudio2Mini/BaichuanAudio/Speechtokenizer/
  Funcodec/XyTokenizer/Bicodec/Neucodec/EcapaTdnn/Wespeaker/Speaker3d/Emotion2vec (12 non-VC)
  全て full quad (enum + from_arg + as_arg + dispatch) 揃い。voice-clone 4 (openvoice_v2/
  knn_vc/freevc/meanvc) も ModelKind 内に配線済 (ELVIS Act 別リポ move は依然 owner 決定)。
- WAVE3-CHERRY-PICKS (FCPE / Silero v6.2.1 / FSMN-VAD): `fcpe.rs` / `fsmn_vad.rs` present
  in `crates/vokra-convert/src/models/`; `SileroVariant::{V5, V6_2_1}` in `silero.rs`;
  `ModelKind::{Fcpe, FsmnVad}` fully wired.
- M5-14-BACKLOG (12 tickets/6h spec): CC 側 6 landed (T01=ADR, T02/T06/T07/T08 実装,
  T10 changelog) + 2 honest defer with ADR-documented mechanism analysis (T03-T05
  pack-once-share = mechanism mismatch on CAM++; T09 batched forward override =
  beam≥5 gain-limited, foundation landed for future override). T11/T12 = owner.
  Detailed reasoning: `docs/adr/M5-14-BACKLOG-pack-cache-batched-beam.md` (ACCEPTED).

**教訓 (Wave 6 audit ↔ 実 HEAD の乖離)**: 本 handoff が生成された 2026-07-30 時点では
partial land だったが、その後の follow-up wave で全 wiring が land 済 → Wave 6 audit の
"actionable gap" 判定は state-stale。今後は audit 前に per-file verification (from_arg /
as_arg / dispatch triple grep) を回すか、または `cargo test -p vokra-cli
parses_every_model_kind` の 1 shot check で state を confirm する。
