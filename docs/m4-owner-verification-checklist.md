# M4 (v1.0-rc) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — 実機テスト・実 weight sourcing・法務 sign-off・外部契約 / インフラ provisioning・ADR 判断を担当。
**CC-side status**: **M4 CC 実装 terminal 到達（2026-07-15、全 20 WP = M4-01〜M4-20）**。investigation 3 round のうち round 2 / round 3 = **2 連続 0 CC ticket** で terminal 判定（M3 と同じ規律）。verify = **default 2340 passed / all-features 2364 passed / 0 failed / 4 ignored**、cargo fmt / clippy `-D warnings` / `scripts/check-zero-deps.sh`（root Cargo.lock = `vokra-*` のみ、NFR-DS-02）/ `scripts/check-abi-changelog.sh` / `scripts/check-platform-support.sh`（anchors 30→50）全ゲート green。branch `feat/m4-plan-and-wave1`。

**位置付け（v-label 規律）**: M4 = **v1.0-rc**（旧 v1.5 → v1.0 GA → 2026-07-14 再割当 #2 で v1.0-rc）。本チェックリストの owner タスク消化 = **OSS 機能完成（rc）判定**（v1.0-rc close、暦月目安 2026-12〜2027-03）の入力であり、**商用 GA 宣言でも C ABI 凍結でもない**（凍結は M5-13 / v1.0 GA タグで発火 = `docs/handoff/m4-12.md` §(f)）。

**owner タスク規模**: `docs/tickets/m4/README.md` §2 の実測 = **依頼者 32 件 + 両方 2 件（M4-01-T02 WebGPU ADR 承認 / M4-18-T01 UTMOS kickoff gate）= 計 34 owner-touch チケット**、16 WP に分布。owner タスク 0 件の WP = M4-02 / M4-03 / M4-12 / M4-19（うち M4-02 / M4-19 は「申し送り」owner follow-up あり = §4 / §carry-over）。

**「CC 完了分」と「owner 残」の分離（honest）**: 実 weight を要する parity は **CC 側で flip-the-switch harness まで完成**しており（synthesized weight で shape / 決定性 / 有限性を機械検証済、upstream 数値一致は未主張）、owner が実 checkpoint を投入した瞬間に発火する。GPU 実 kernel（§6）は **M4 実装漏れではなく各 spec が意図的に別 WP 化した follow-up**（CPU arm は real で機能完結）。

**参照文書の tracking**: 本ファイル（`docs/m4-owner-verification-checklist.md`）は **tracked（gitignore 対象外、`git check-ignore` で確認済）** = `docs/m3-owner-verification-checklist.md` と同運用。`docs/handoff/m4-*.md` / `docs/platform-support/*.md` / `docs/license-audit.md` / `docs/legal-compliance.md` / `docs/m4-07-hopper-bench-handover.md` / `docs/m4-scope-expansion-2026-07-13.md` / `docs/benchmarks/*.md` は **tracked（public）**。`docs/tickets/m4/*.md`（spec）/ `docs/adr/M4-*.md`（ADR）/ `docs/milestones.md` は **gitignore ローカル内部文書**（本書からは ID で参照）。

各タスクは **(a) 何をするか / (b) なぜ owner 専任か / (c) 参照（handoff or spec ticket ID）/ (d) 完了条件** の 4 項で記述する。

---

## 1. 実モデル weight parity（flip-the-switch harness の発火）

CC 側は各モデルの **flip-the-switch parity harness**（実 checkpoint 到着で発火する `assert_vs_hf_reference` 系 / real-checkpoint CI）を完成済み。owner は実 weight を sourcing して発火させる。**fabricated pass 禁止**（実測前に atol を緩めない、synthesized weight での pass を「upstream 一致」と偽らない）は全項共通の red-line。

### 1.1 Sesame CSM-1B（M4-05）

- **(a)**: 公式 checkpoint（`sesame/csm-1b` + `meta-llama/Llama-3.2-1B`、いずれも HF gated repo）を入手し CC に手渡す。tensor manifest の flip-the-switch（T02/T06/T23/T24）が実 checkpoint で発火する。
- **(b)**: HF gated repo のアカウント承諾が必要、weight は CC 機体に無い。
- **(c)**: spec `M4-05-T29`（owner）。CC 到達分 = Llama-3.2-flavor backbone + depth transformer + Mimi neural chain + `assert_vs_hf_reference` harness。
- **(d)**: checkpoint 手渡し済 + flip-the-switch テストが実 checkpoint で実行可能。

### 1.2 Moshi（M4-06）

- **(a)**: 上流配布（`kyutai-labs/moshi` / HF `kyutai/moshiko-pytorch-bf16`、~15GB BF16、NOTICE §5 記載 source）から checkpoint を取得し pin、T25 workflow / T07・T15 fixture / T26 デモに供給。
- **(b)**: ~15GB weight の sourcing、CC 機体に無い。
- **(c)**: spec `M4-06-T29`（owner）。
- **(d)**: fixture / workflow へ weight 供給済、flip-the-switch 発火。

### 1.3 Whisper small / medium / turbo（M4-14 = M2-06 carry-over）

- **(a)**: `openai/whisper-small` / `-medium` / `-large-v3-turbo` を HF 取得 → `vokra-cli convert --model whisper`（shape-driven、自動 size 検出、fp16 passthrough）→ `tools/parity/dump_whisper_reference.py` で **real-audio fixture（jfk-30s.wav 由来）を再生成 → owner レビュー後に手動 commit**（auto-commit は red-line）。その後 `parity-whisper-real.yml` を 3-size opt-in で初回 workflow_dispatch し、turbo が atol 0.01 を超過した場合のみ実測 max|Δ| を CC に渡して honest calibrate。
- **(b)**: 実 checkpoint 変換 + real-audio fixture の owner レビュー commit + HF DL を要する CI 起動。
- **(c)**: spec `M4-14-T09`（変換 + fixture 再生成）/ `M4-14-T10`（初回 dispatch + turbo atol 判断）。CC 到達分 = shape-driven converter（5 サイズ）+ per-size atol lookup 基盤（default 0.01 維持）+ 3-size matrix CI + dumper turbo `vocab_resource` 標準化（turbo は large-v3 と同一 51866 vocab）。
- **(d)**: small/medium/turbo の `tests/parity/whisper_{size}/` が real-audio 由来に更新・commit 済、3-size parity leg が実 checkpoint で完走、turbo max|Δ| が記録（0.01 超過なら honest calibrate 値が確定）。

### 1.4 DeepFilterNet（M4-20）

- **(a)**: MIT checkpoint を取得し `vokra-cli convert --model denoise`（T12）で GGUF 化 → Vokra native forward（T11）と upstream DeepFilterNet（PyTorch reference）の enhanced 出力を実データ照合（atol / SNR 改善量は upstream 参照で確定）。
- **(b)**: 実 checkpoint / PyTorch reference 実行が必須。
- **(c)**: spec `M4-20-T17`（owner）。CC 到達分 = `crates/vokra-ops/src/denoise.rs`（STFT → ERB gain → deep-filter → iSTFT topology）+ 合成 weight での shape / GGUF round-trip 検証。
- **(d)**: 実 checkpoint で native denoise が reference と許容誤差内一致（or honest negative の記録）。

### 1.5 UTMOS / DNSMOS（M4-18）

- **(a)**: UTMOS weight source URL 確定（§3.5 の license と同時）→ `docs/adr/M4-18-utmos-gate.md` recipe（offline reference dump → `score.json` + clip commit → `source.env` commit → CC の T05 converter 実装依頼 → `parity-utmos` workflow_dispatch）で flip 実行。DNSMOS は license 次第（§3.5）。
- **(b)**: weight source + license が未着（kickoff 週 gate = **現状 NO-GO-defer**、`docs/handoff/m4-18.md` §(a)）。
- **(c)**: handoff `docs/handoff/m4-18.md` §(d) 依頼者 queue + §(e) 制約 / spec `M4-18-T02`（weight+license）`T05`（converter mapping、CC だが実 mapping は checkpoint 到着後）。CC 到達分 = `crates/vokra-eval/src/metrics/utmos.rs` skeleton + `AudioMosMetric` trait + `parity_utmos.rs` harness（synthesized 拒否 + env/fixture gated clean skip、fixture 未 commit = 実 reference 無しでの expected_score 捏造禁止）。
- **(d)**: weight URL 確定 + license sign-off 記入 + `parity-utmos` 初回 workflow_dispatch 完走（or Rejected で defer 根拠が記録）。

### 1.6 Mimi / DAC real-checkpoint parity（M4-04 / M4-05）

- **(a)**: (i) `parity-rvq-real.yml`（mimi / dac / encodec）を初回 workflow_dispatch し per-tensor max|Δ| + verdict 表を確認（M4-04-T21）。(ii) Mimi encode/decode parity（`M4-05-T14`/`T34` = **CC harness、実 checkpoint は M4-05-T29 の owner sourcing で発火**）。
- **(b)**: 実 checkpoint 変換の CI 起動 + HF flakiness、CC 機体では未実行。
- **(c)**: spec `M4-04-T21`（owner 初回 dispatch）/ `M4-05-T14`・`T34`（CC parity、owner checkpoint 依存）。CC 到達分 = `mimi_rvq` / `dac_rvq` op family + converter + `parity-rvq-real.yml` scaffold + upstream reference fixture 契約。
- **(d)**: 初回 dispatch が success（差し戻し fix 後を含む）。required check への promotion はしない前提を維持（HF flakiness、Kokoro/Whisper real CI と同運用）。

### 1.7 WavTokenizer / X-Codec 2（M4-16、FSQ）

- **(a)**: CC 側 parity fixture は **合成 weight のみ（pretrained 未 DL・未使用）**。実 WavTokenizer / X-Codec 2 model の e2e parity は **後続の実モデル統合 WP**（M4 スコープ外）で発火。本フェーズの owner タスクは license sign-off（§3.4）のみ。
- **(b)**: 実 weight は license 確定後（特に X-Codec 2 の CC-BY-NC 4.0 疑義、§3.4）。
- **(c)**: spec `M4-16`（op のみ）+ 消費者 WP（converter metadata `documented`→`persisted` + 実モデル e2e）。
- **(d)**: op parity（合成）green は CC 完了済 = 本カテゴリでの owner 残は license（§3.4）に集約。

---

## 2. 実機テスト（CC 機体に無い hardware / real device）

### 2.1 Hopper H100 で FA v3 有効化 + FA v2 比計測 + ダッシュボード登録（M4-07）

- **(a)**: vast.ai H100（SM 9.0）を起動 → probe が SM 9.0 報告 → NVRTC feasibility（`fa_v3_nvrtc_feasibility`）green → FA v3 parity 3 面（causal / non-causal / validation）green + worst |Δ| 記録 → e2e RTF `--fa-mode {decomposed,v2,v3}` × N=10 → kernel-level FA v2 比（研究 §10「2-3x」照合はここのみ）→ `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` の TBD を実測で fill → **ダッシュボード（X-06 nightly 結果公開面）に FA v2 比の行を追加 = WP close 発火**。
- **(b)**: H100（Hopper）は CC 機体（M1 iMac / vast.ai RTX 4090 = SM 8.9）に無い。kernel は CUDA-less 機体で blind 転記された **OWNER-VERIFY hotspot**（wgmma descriptor LBO/SBO 割当・d-fragment map・compute_90a NVRTC 通過が実機未検証）で、§1-§2 が最初の実証。
- **(c)**: handover `docs/m4-07-hopper-bench-handover.md`（全手順 + §6 差し戻し条件 + §7 チェックリスト）/ spec `M4-07-T17`（有効化確認）`T18`（計測 + 登録）。**H100 数値を RTX 4090 gate 用 baseline に混ぜない**。届かなくても honest 登録で WP close（2-3x は受け入れ基準ではない）。**FA v3 の v1.0 以前前倒しは設計制約違反 = red-line**（Hopper 専用ゆえ本 WP に閉じる）。
- **(d)**: FA v3 parity 3 面 green + worst |Δ| 記録（完了条件前半）+ ダッシュボード登録（完了条件後半）+ `vastai destroy`。

### 2.2 RISC-V LicheePi 4A / Milk-V Duo 実機（M4-08、RVV 0.7.1）

- **(a)**: (i) 両 board 調達 + bring-up + `/proc/cpuinfo` 全文 + `uname -a` dump 収集（`isa` の `v` 有無 / `mvendorid` 露出 = 検出前提の ground-truth backfill）。(ii) LicheePi 4A（Tier 1）= `active_isa()` が 0.7.1 tier 報告 + selftest green + Silero VAD / Whisper base smoke。(iii) Milk-V Duo（Tier 2）= Silero VAD only の動作確認。
- **(b)**: RISC-V 実機は CC 機体に無い（CLAUDE.md「開発環境」）。
- **(c)**: spec `M4-08-T14`（調達 + dump）`T15`（LicheePi 4A）`T16`（Milk-V Duo）。CC 到達分 = probe / dispatch / kernel + cross-build CI。dump が CC の source 調査と食い違えば T05 parser fixup を CC に差し戻す。
- **(d)**: 両 board の dump 取得 + LicheePi 4A で selftest green + smoke 動作、Milk-V Duo で Silero VAD 動作。**RTF 数値目標は両 board に未定義 = 参考記録のみ（発明しない）**。

### 2.3 Android 実機 Vulkan soak（M4-13、exit hard gate）

- **(a)**: Snapdragon 8 Gen 3（Adreno 750）+ Dimensity 9300（Mali G720）実機で `cargo build --target aarch64-linux-android --features vulkan --release` → adb push → JNI wrapper で 30s 音声 transcribe → 転写 vs reference 一致 + **median RTF < 0.7（NFR-PF-06）**。
- **(b)**: lavapipe（CPU-side ICD、CI で検証済）は **driver-level GLSL→SPIR-V コンパイラ bug を catch できない** ため実機 soak が M4-13 の exit hard gate。Android GPU 実機は CC に無い。
- **(c)**: spec `M4-13-T17`（owner、M3-18 併走）+ handover `docs/m3-18-android-rtf-handover.md`。**依存 = M4-13-T16 の glslc `.spv` commit（§4.3）が先**。coop-matrix 非対応チップで RTF 2x 劣化なら subgroup INT8/FP16 kernel 追加 wave（別 WP へ flow）。
- **(d)**: Adreno 750 + Mali G720 で Whisper base が Vulkan 経由で動作 + 転写実用一致 + median RTF < 0.7。未達は SPIR-V shader 最適化 issue へ flow。

### 2.4 Web ブラウザ実機テスト（M4-01 spot check → M4-11 正式判定）

- **(a)**: M4-01 の Web 配布物（WASM + WebGPU、`scripts/build-wasm.sh pkg` / `web-wasm.yml`）を実ブラウザ（提案: Chrome / Edge / Safari / Firefox × 各 OS）で動かし **Whisper base 動作（NFR-PF-08）** を確認・記録。WebGPU 非対応ブラウザで **silent に壊れず明示エラー or WASM-SIMD fallback**（FR-EX-08）を確認。M4-02 Unity WebGL デモも同一セッションで確認。
- **(b)**: 実ブラウザ / 実 OS の GPU 挙動 + Safari WebGPU 対応状況は実測でのみ確定（発明禁止）。base HEAD は integrated `0691ae9` で Web は CC-side landed（M4-11 handoff §(0) RESOLVED）。
- **(c)**: spec `M4-01-T28`（owner spot check、M4-11 への先行入力）→ `M4-11-T11`（owner 正式判定）+ handoff `docs/handoff/m4-11.md` §(b) T11 / `docs/handoff/m4-02.md` §4-2（web delivery 要件 = MIME / 圧縮 / COOP-COEP 不要 / モデル HTTP fetch）。記録先 = `docs/platform-support/v1.0-rc-support-matrix.md` Part 10。
- **(d)**: owner 確定ブラウザ組合せで Whisper base 動作 + Unity WebGL デモ動作が matrix に記録、不動作は理由付き honest 記録。**NFR-PF-08 後段の CSM streaming は M4-05/06 検収に合流（二重確認しない）**。

### 2.5 CPU ISA サーバ / ARM64 上位 tier perf 実測（M4-17、advisory）

- **(a)**: (i) cloud VM per-run（Cascade Lake+ Xeon / EPYC Zen4）で **AVX-512F/DQ/BW/VL + VNNI + BF16 + AVX-VNNI-256** の perf を `VOKRA_CPU_ISA=...` override 経由で AVX2 tier 比実測。(ii) **Apple M2+**（i8mm + bf16、M1 非対応ゆえ M2+ 必須）+ **Raspberry Pi 5 Cortex-A76**（fp16 + dotprod）+ **AWS Graviton3** で ARM64 tier 実測。
- **(b)**: AVX-512 系 instance / Apple M2+ / Pi 5 / Graviton3 は CC 機体（M1）に無い。**i8mm/bf16 は M1 非対応ゆえ本実機実測が初の実 silicon 検証**。
- **(c)**: spec `M4-17-T23`（x86-64）`T24`（ARM64）。**計測は advisory**（CI required check に数値 assert を入れない = M3-01 CUDA RTF gate と同 posture、standing runner 買わない = 2026-07-14 判定）。**M2-14 self-hosted CUDA runner standup の backlog と束ねる**（AVX-512 対応 instance を選べば 1 台兼務）。
- **(d)**: 各 ISA tier の perf が実測され、AVX2 / NEON baseline 比の改善が `docs/bench-baselines/` 相当に advisory 記録（数値目標は gate ではない）。

### 2.6 CSM / Moshi full-duplex デモ実機検収（M4-05 / M4-06）

- **(a)**: real mic/speaker 環境で T26 ハーネス（real weights 投入済）を駆動し、CSM = streaming 動作 + barge-in + **AEC 実音響**（スピーカー再生→mic 収音の実 echo 経路で自己エコー崩壊なし）。Moshi = full-duplex（AEC 有効で対話継続）+ barge-in で発話即 flush + inner monologue text 整合。`parity-{moshi}-real.yml` 初回 workflow_dispatch も同時消化。
- **(b)**: スピーカ→マイクの実音響 echo 経路の AEC 実効性は **実機でのみ検証可能**（CC の合成 echo は近似 = T16/T26 で明示切り離し）。
- **(c)**: spec `M4-05-T30`（CSM streaming + AEC 実音響 + legal-compliance）/ `M4-06-T30`（full-duplex 検収 + parity dispatch）。CC 到達分 = M3-14 barge-in / M4-03 aec op（CPU）/ file-driven CLI demo。real mic/speaker demo アプリ（OS audio API 依存）は不足時に excluded workspace で follow-up 起票判断。
- **(d)**: CSM 受け入れ基準（streaming / barge-in / AEC）+ Moshi (a)〜(c) の検収 pass が記録され、各 WP 完了条件が owner 確認済。

---

## 3. license sign-off（`docs/license-audit.md` §3.1 Owner sign-off template）

**運用**: §3.1 template の各行 "Owner sign-off (YYYY-MM-DD)" + "Approval"（☐ Commercial / ☐ Research-only / ☐ Rejected）を記入。**空欄 = 未サインオフ = 公式配布不可（fail-closed）**。CC は upstream LICENSE の事実確認（CC-verified 節）まで、distribute-or-not の法務判断は依頼者専権。**M3 queue（CosyVoice2 / Voxtral / Mimi）と一括処理可**（wave 計画 owner track「license sign-off queue 一括」）。

### 3.1 CSM / Moshi（M4-05 / M4-06）

- **(a)**: CSM = Apache 2.0 / Apache 2.0（§3 L80）+ Mimi encoder/neural decoder（Kyutai CC-BY 4.0）attribution 確認。Moshi = code Apache 2.0 / weight CC-BY 4.0（§3 L81、要 credit）+ attribution 文言の法務的十分性（焼き込み文 + banner 表示形 + CLI `--quiet` 抑止可否）確定。両者 legal-compliance checklist（FR-MD-13）通過。
- **(b)**: 商用配布 + training data 疑義なしの法務判断。
- **(c)**: spec `M4-05-T29` / `M4-06-T29` + `docs/license-audit.md` §3.1（CSM / Moshi 行）+ `docs/legal-compliance.md` checklist。
- **(d)**: §3.1 に sign-off 記録 + checklist 通過。zoo 公開は sign-off 通過が前提。

### 3.2 DAC / Mimi standalone zoo 配布（M4-04）

- **(a)**: DAC（MIT、§3.1 L205）の商用配布 sign-off + Mimi standalone codec GGUF の zoo 配布判断（既存 Mimi 行 L204 の sign-off に含める）。attribution（NOTICE §5 + 配布物同梱）で CC-BY 4.0 義務充足を確認。
- **(b)**: fail-closed = sign-off まで DAC / Mimi standalone GGUF は release に載せない。
- **(c)**: spec `M4-04-T20` + `docs/license-audit.md` §3.1 L204/L205。
- **(d)**: DAC + Mimi standalone の配布可否が記録され zoo 配布可否が確定。

### 3.3 Whisper small / medium / turbo（M4-14）

- **(a)**: §3.1 owner sign-off 表（L178-182）の small/medium/turbo 行を記入。5 サイズ共通 MIT/MIT（OpenAI 公式）ゆえ一括 Commercial + 日付 + 署名。**turbo は別 checkpoint（`openai/whisper-large-v3-turbo`）だが同一 MIT + 同一 51866 vocab**（混同防止）。
- **(b)**: FR-MD-13 の weight 法務 sign-off。
- **(c)**: spec `M4-14-T11` + `docs/license-audit.md` §3.1 L178-182 + M4-14 activation note（L149）。
- **(d)**: 3 サイズ行に Commercial 判定 + 日付 + 署名、FR-MD-13 完了。

### 3.4 WavTokenizer / X-Codec 2（M4-16、dual-license 齟齬解消）

- **(a)**: WavTokenizer（MIT）sign-off + **X-Codec 2 の license 3 系統齟齬を確定**: (i) code = MIT（GitHub `zhenye234/X-Codec-2.0` + PyPI `xcodec2` 0.1.5）/ (ii) **weight 配布 repo HF `HKUSTAudio/xcodec2` README = `license: cc-by-nc-4.0`**（2026-07-15 CC fetch）/ (iii) milestones §8 + deliverables §3.5 =「MIT+Apache 2.0 dual」= 三者不一致。upstream `LICENSE` 実体で per-file か combined か + weight license を確認し license-audit.md を正に統一。**NC 確定なら weight を公式 zoo 除外 + `crates/vokra-core/src/compliance/license_class.rs` の `"x-codec-2" | "xcodec2" => Permissive` 分類変更を CC に差し戻す**（現分類だと NC weight を素通し）。
- **(b)**: dual-license 実体の法務確定 + zoo 搭載可否。CC は事実 surface まで（§CC-verified M4-16 flag 節）、確定は owner。
- **(c)**: spec `M4-16-T14` + `docs/license-audit.md` §3.1 L207 + §CC-verified M4-16 flag 節（L151-155）。**sign-off 空欄 = 配布不可（fail-closed）が既に効いており判定完了まで zoo 配布は発生しない**。
- **(d)**: 両者の商用配布判断が sign-off + X-Codec 2 dual-license 表記が SoT 間で統一。

### 3.5 UTMOS / DNSMOS（M4-18）

- **(a)**: UTMOS = SaruLab UTMOS22 系の weight source URL 確定 + §3.1 に UTMOS 行を追加し sign-off（学習データ由来の商用配布可否 = SaruLab MOS Challenge 2022 音声由来 weight の再配布条項確認、research-only なら評価用途限定 + zoo 除外）。DNSMOS = Microsoft P.835（`microsoft/DNS-Challenge` 系）の license 一次資料検証 → 商用 OK なら T11 GO / research-only なら T11 skip で UTMOS 単独 scope 縮小。
- **(b)**: weight source + license 一次資料の確認（本 spec に推測 URL / license を書かない = ハルシネーション厳禁）。**kickoff 週 gate = 現状 NO-GO-defer**（`docs/handoff/m4-18.md` §(a)）で v1.0.x patch へ自動 defer 済 = **owner が defer 追認 or M4 内 flip を確定**。
- **(c)**: handoff `docs/handoff/m4-18.md` §(d) queue / spec `M4-18-T02`（UTMOS）`T03`（DNSMOS）。
- **(d)**: UTMOS weight URL 確定 + §3.1 sign-off 記入（or Rejected 記録）+ DNSMOS license 判定が一次資料引用付きで記録され T11 GO/skip 確定。

### 3.6 DeepFilterNet + GTCRN（M4-20）

- **(a)**: DeepFilterNet3 の MIT（code/weight とも）を精査し公式 zoo 収録可否を sign-off + GTCRN license 事前確認（RNNoise BSD は candidate）。TitaNet（NeMo NC 制約可能性）/ pyannote `diarize`（HF gated + 商用条項）は「M5 で確認」の記録のみ。
- **(b)**: model zoo 収録の法務判断。
- **(c)**: spec `M4-20-T18` + `docs/license-audit.md` §3.1（DeepFilterNet3 行 L91、★ 要 owner sign-off T18）。
- **(d)**: DeepFilterNet の zoo 収録可否 sign-off + GTCRN license 判定記録。CC-BY-NC / 学習権利不明 weight の zoo 除外規律を維持。

### 3.7 既存 M3 sign-off queue（carry-over）

- **(a)**: CosyVoice2（Apache 2.0）/ Voxtral（Apache 2.0）/ Mimi（CC-BY 4.0）の §3.1 sign-off（Wave 14 で docs 側 land 済、依頼者記入待ち）。上記 3.1〜3.6 と一括処理可。
- **(c)**: `docs/license-audit.md` §3.1（該当行）+ M3 owner checklist §3。

---

## 4. 外部契約 / インフラ provisioning

### 4.1 npm org / scope + NPM_TOKEN（M4-01）

- **(a)**: npm org / scope（提案 `@vokra` = GitHub org 命名と揃える）を確保し publish 用 `NPM_TOKEN` を repo secret 登録。初回 publish の是非・時期を判断（rc タグで CD 自動発行 / preview prerelease / dry-run 継続の 3 択）。
- **(b)**: npm アカウント + secret provisioning は owner 権限。
- **(c)**: spec `M4-01-T27`。CC 到達分 = `npm-web-release`（release.yml）CD job を dry-run で検証済。
- **(d)**: `NPM_TOKEN` 登録 + publish 時期判断が記録（実 publish or rc タグ時実行の確定）。

### 4.2 CDN 選定（M4-11）

- **(a)**: `docs/platform-support/cdn-selection-material.md`（R1-R6 checklist）で Web 配布（WASM / GGUF モデルファイル）の CDN を選定。**COOP/COEP レスポンスヘッダ制御可否（WASM Threads 成立条件）を必須 gate**。ドメイン（vokra.dev / .io / .ai、CLAUDE.md 取得推奨）接続構成も記録。
- **(b)**: CDN 契約 + ドメイン取得は owner。
- **(c)**: spec `M4-11-T12` + handoff `docs/handoff/m4-11.md` §(b) T12 / `docs/handoff/m4-02.md` §2（圧縮 / MIME / COOP-COEP 要件）。記録先 = matrix Part 9。
- **(d)**: 採用 CDN + 根拠 + COOP/COEP 制御方法 + 費用が記録。

### 4.3 glslc precompiled `.spv` commit（M4-13）

- **(a)**: LunarG SDK / glslang install → `scripts/compile-vulkan-shaders.sh --update` で 12 `.comp`（gemm_subgroup / gemm_coopmat / gemv / softmax / softmax_causal / layer_norm / gelu / conv1d / elementwise / activation / transpose / gather）を `kernels/precompiled/*.spv` に compile + git commit + 各 blob の sha256 を `spirv.rs::SHADERS` の `expected_sha256_hex` に paste + `load_spv` arm を `include_bytes!` に置換。
- **(b)**: glslc toolchain の developer-side install（現在 `kernels/precompiled/` は README のみ、handcrafted `copy_f32`/`add_f32` の 2 本のみ commit 済）。
- **(c)**: spec `M4-13-T16` + `docs/adr/M3-02-spirv-generation.md` §4-(a) + `scripts/install-vulkan-toolchain.md` + handoff `docs/handoff/m4-15.md` §(b)。**§2.3 Android soak の前提**。
- **(d)**: 12 `.spv` commit + SHA-256 pin が `verify_pinned_hashes` green + `load_spv` が該当 op で `Some(bytes)` 返却（op が lit up）。

### 4.4 secrets.UNITY_LICENSE provisioning（M4-02、M2-11 carry-over）

- **(a)**: `nightly-webgl.yml` の Unity leg + `nightly-il2cpp.yml` を license-gated skip から発火させる Unity license secret を provision。
- **(b)**: Unity ライセンス契約 = owner。CC 側は `.a` staticlib build / thread-free gate / wasm-harness leg（license 不要、node で実 VAD 完走）まで完成。
- **(c)**: handoff `docs/handoff/m4-02.md` §4-1 + spec `M4-02`（依頼者 0 件 = 申し送りのみ、`nightly-il2cpp.yml` 冒頭コメントが一次ソース）。
- **(d)**: `nightly-webgl.yml` 初回 workflow_dispatch で wasm-harness leg 即 green + Unity leg が license 後 green。

### 4.5 CI 初回 workflow_dispatch 群（owner 引き渡し前例）

- **(a)**: CC 機体で未実行の workflow を GitHub Actions 上で初回起動: `parity-rvq-real.yml`（M4-04-T21）/ `parity-whisper-real.yml` 3-size（M4-14-T10）/ `parity-moshi-real.yml`（M4-06-T30）/ `parity-utmos.yml`（M4-18、weight 到着後）/ `gpu-vulkan-parity.yml` lavapipe（M4-13-T18）/ `web-wasm.yml` + `npm-web-release` dry-run（M4-01-T28）。
- **(b)**: 初回 run は owner 引き渡し（プロジェクト前例、M2/M3 と同）。green / honest skip 理由の可視を確認。
- **(c)**: 各 spec の owner ticket + 各 workflow YAML の "initial workflow_dispatch is owner handoff" コメント。
- **(d)**: 各初回 run が green（or skip 理由が honest に可視）。**required check への promotion は数週連続 green 後の owner 判断**（HF flakiness の PR blocking 回避、Kokoro/Whisper real CI と同運用）。

---

## 5. owner ADR 判断（M4-12 v1.0-rc baseline 前 = hard gate G7）

CC は評価材料 + ADR 草案を **Status: Proposed** で止め、判断記録欄を**空欄**で渡す（fabricated pass と同型の honest 規律 = CC は Accepted にしない）。

### 5.1 M4-09 — piper-plus G2P 方針（3 択）

- **(a)**: `docs/adr/M4-09-g2p-policy.md` §8 判断記録欄に ① 流用継続（git rev `41f3696` = v0.5.0 pin）/ ②a Rust 移植 = in-tree port / ②b Rust 移植 = 除外 ws vendor / ③ 辞書ベース独自実装 を選択 + `docs/piper-plus-integration.md` §8 の B-7/B-8/B-10（+ 可能なら B-9/B-11）を遡及確定 + 後続指示（① = M5-09 スキップ / ② = M5-09 スケジュール化 / ③ = M5-09 re-scope or 新 WP）。
- **(b)**: 判断・承認そのものが milestones §8 M4-09 行の owner 担当。CC 推奨 = **(提案) ① 流用継続**（忠実度最高 + 追加工数ゼロ、③ misaki は ES/FR/PT/SV 欠 + Python 実装 + eSpeak-NG fallback が red-line）だが拘束しない。
- **(c)**: ADR `docs/adr/M4-09-g2p-policy.md`（判断材料 3 pack + wasm32 probe exit 0 + misaki 一次確認）/ spec `M4-09-T05`。
- **(d)**: §8 記入 → Status Accepted + T06 文書伝播 + T07 build script 齟齬（§6 latent bug）の解消方向確定。**M4-12 前必須（G7）**。

### 5.2 M4-10 — MLIR audio dialect + StableHLO 採否（FR-EX-01 再評価条項の消化）

- **(a)**: `docs/adr/M4-10-mlir-stablehlo.md` §13 判断記録欄に Option A（現状維持 = flat op enum）/ B（offline 限定）/ C（runtime 統合）を選択 + §5.3（graph-first 実行への転換計画の有無）回答 + 却下時の次回再評価トリガ確定 or 採用時の §11 手続き発動。
- **(b)**: 採否判断 = owner。CC 推奨 = **(提案) Option A 確定 + Option B 保留 + Option C 非採用**（graph IR 経由の実行が現状ほぼ無く MLIR 便益の分母が空、flat enum は 7 wave / 13 variant で backend 破壊 0、競合 ggml/Candle も MLIR 不採用）。**Option C は NFR-DS-02 / SRS §5 改訂 + CLAUDE.md 改訂を伴う**。
- **(c)**: ADR `docs/adr/M4-10-mlir-stablehlo.md`（OpKind 17 variant + backend coverage 実測 + supersede 系譜）/ spec `M4-10-T08`。
- **(d)**: §13 記入 → Accepted。採用時は CLAUDE.md + SRS 改訂 + 新規実装 WP 起票を **M4-12 前（G7）**に完了。

### 5.3 M4-18 G2 — UTMOS defer 中の M4-05/M4-06 品質判定方針の追認

- **(a)**: ratified 完了条件「MEL loss / UTMOS 劣化 5% 未満」は UTMOS defer 中 honest に判定不能ゆえ、**「mel_loss 5% gate + UTMOS は advisory（判定不能を明示）」への切替を追認・記録**（ratified 条件の変更 = owner 専権）。
- **(b)**: ratified 完了条件の変更判断。機械面は `DegradationReport::mel_loss_only = true` + `KvQuantVerifyReport::utmos_unavailable = true` が「UTMOS gate: 未達」を明示（mel_loss 単騎で「UTMOS 通過」と偽らない）。
- **(c)**: handoff `docs/handoff/m4-18.md` §(d)-4 / spec `M4-18-T01`（両方 = kickoff gate）。
- **(d)**: 判定方針切替が記録され M4-05/06 の品質判定 posture が確定。

### 5.4 M4-01 T02 — WebGPU × zero-dep ADR 承認（両方）

- **(a)**: wgpu × NFR-DS-02 衝突の解決構成（(A) 生 extern-import shim 提案 / (B) 除外 workspace + wgpu / (C) dlopen 不成立）を CC 改訂案で確認・判定。
- **(b)**: zero-dep 中核 selling point に関わる設計判定は owner 承認。
- **(c)**: spec `M4-01-T02`（両方、CC が改訂案 / owner 判定）+ ADR `docs/adr/M4-01-webgpu-wasm.md`。
- **(d)**: ADR 構成が owner 承認。

---

## 6. M4 follow-up（別 WP 扱い = 実装漏れではない）

**位置付け（honest）**: 以下は **各 spec が明示的にスコープ外化した follow-up** であり M4 の実装漏れではない。**CPU arm は real で機能完結**しており、GPU 化は性能最適化。**Metal 半分は M1 iMac で CC 検証可だが CUDA 半分は vast.ai owner 必須で非対称ゆえ別 WP**（M4 kernel-fusion 系 follow-up ticket 群 or M5）。

### 6.1 codec op の GPU 実 kernel（Metal MSL / CUDA NVRTC）

- **(a)**: RVQ 系（`mimi_rvq` / `dac_rvq` / `encodec_rvq`）= 3 op 共通の naive「embedding gather + FP32 fold、shared-memory tile 不要」layout で差替。FSQ 系（`wavtokenizer_vq` / `xcodec2_fsq`）= 単段 GEMV bound ゆえ既存 gemv/gather kernel 再利用。現状は **seam-awareness + 明示 `UnsupportedOp`（FR-EX-08、silent CPU fallback 禁止）** まで。
- **(c)**: spec `M4-04`（L45 / L230 / L367 = GPU 実 kernel はスコープ外の明記）/ `M4-16`（L206 / L208 / L264 / L287）。naive layout メモ = `mimi_rvq.rs` L104-106。
- **(d)**: follow-up ticket で発火（本フェーズの完了条件外）。

### 6.2 S2S full device-residency / per-step 1 command buffer kernel fusion

- **(a)**: CSM / Moshi / Voxtral の per-step 1 command buffer 化（Mistral RMSNorm + RoPE + GQA-attn + SwiGLU の融合 Metal MSL / CUDA PTX kernel）。現状は M3-10 Wave 10 と同じ **thin Compute-seam wrapper + `ResidencyMode` enum + plumbing seam** まで（per-op 実 GPU dispatch は既に real）。
- **(c)**: spec `M4-05`（L41 / L369）/ `M4-06`（L73 / L82 / L458-(c)）+ M3-10 Wave 10 の残置整理。
- **(d)**: M4 kernel-fusion 系 follow-up ticket で発火。**FA v3 は使わない（M4-07 red-line）**。

### 6.3 参考: AEC / 音声強化 op は CPU-by-design（GPU follow-up ではない）

- AEC（M4-03）= 低次元・毎フレーム streaming filter で GPU offload 転送コストが支配的ゆえ **Compute seam に追加しない（CPU only）**、GPU seam は「必要性が実測で示された場合のみ」（M4-03 L50 / L323 / L355-(c)）。音声強化 op（denoise / agc / hpf / loudness_norm、M4-20）も同様に GPU seam は「必要に応じ後続 WP」（M4-20 L264-(d)）。**これらは GPU 化前提の follow-up ではない**（誤った暗黙の期待を作らないため module doc に明記済）。

---

## 7. doc の owner 1 語修正（軽微）

- **(a)**: `docs/milestones.md` §8 M4-11 行末の「（依頼者タスク一覧 v1.0 GA）」は 2026-07-08 mid-form（M4 = v1.0 GA）の stale reference。v-label 再割当 #2 後の正 = §11「2026-12〜2027-03（改訂: M4/v1.0-rc close 目安）」行。
- **(b)**: `docs/milestones.md` は gitignore ローカル（owner 領分）ゆえ owner が編集。
- **(c)**: handoff `docs/handoff/m4-11.md` §(d) / spec `M4-11-T13`（承認時に適用可否を判断）。
- **(d)**: §8 M4-11 行の参照が §11 行に修正（or「軽微ゆえ不要」の判断記録）。

---

## Carry-over（M2 / M3 owner backlog、M4 と同一 owner オペで束ねる）

| 項目 | 要件 | 参照 | M4 束ね先 |
|------|------|------|-----------|
| iOS 実機 RTF（Whisper base RTF < 0.5） | NFR-PF-03 | `docs/m2-14-ios-rtf-handover.md` | 単独（M2 carry-over） |
| M2-14 self-hosted CUDA runner standup | NFR-PF-04（CUDA large-v3 RTF<0.1 formal gate） | M3 checklist §6 / `docs/m2-cuda-rtf-variance-2026-07-08.md` | §2.5 M4-17 cloud VM と 1 台兼務 |
| M3-18 Android/Godot 実機 | NFR-PF-06 / FR-API-05 | `docs/m3-18-android-rtf-handover.md` | §2.3 M4-13 Android soak + §2.5 M4-17 ARM64 と併走 |
| M3-11 T19/T20（実 Godot editor verify + WP-close PR） | FR-API-05 | `docs/m3-11-godot-demo-handover.md` | 単独（M3 carry-over） |
| M3-15 real network 75ms 実測 | NFR-PF-05 | `docs/m3-15-server-latency-handover.md` | 単独（M3 carry-over） |
| M3-19 Kill switch D 判定（暦月 2027-03〜05） | NFR-MT-05 | M3 checklist §4 | 四半期 Go/No-go review |

**M4-11 T13 gap-flow 判断対象（owning WP 不在の残存 gap、3 択 = M4 follow-up / M5 送り / 要件改訂）**: (a) **Android AAR**（standalone AAR を出す CD job 不在、arm64 `.so` は Unity UPM / Godot addon 同梱のみ）/ (b) **desktop 共有ライブラリ + CLI の release 自動発行 job**（`release.yml` は ios/unity/pypi/godot のみ、NFR-MT-08 対照 gap）/ (c) **NFR-MT-02 Tier 2 実機 nightly**（現在 per-PR cross-build のみ、owner lab standup = X-06 後の継続タスク）。詳細 = `docs/handoff/m4-11.md` §(c) / matrix Part 11.4。

**M4-19 Wyoming 申し送り（本 WP 完了条件外 = scope-guard）**: M5Stack / HA Voice Satellite soak（optional）/ Wyoming・HA community engagement（X-05、own pace）/ faster-whisper real-WER 検証（`m4_19_asr_real_gguf_round_trip_gated` を real GGUF で実行）/ CLI model wiring（M2-09-T04 carry-over = `--asr-base`/`--tts-piper` を `InferenceService::build` + `spawn_server_with_service` に配線）。詳細 = `docs/handoff/m4-19.md` §Owner tasks。

**M4-15 SBOM reproducible-build verify（owner 1 件）**: 同一入力から SPDX SBOM が別マシン / 別 checkout で byte-identical に再生成されることを確認（`docs/handoff/m4-15.md` §(a)、spec `M4-15-T10`）。差分は環境依存フィールドを記録して CC に fixup 依頼（fabricated pass 禁止）。

**worktree 産 ADR の main checkout への sync（owner）**: worktree sandbox が main checkout への直接 Write を禁止するため、`docs/adr/M4-15-*.md` / `docs/adr/M4-18-utmos-{gate,arch}.md` 等は worktree 内作成 = **worktree cleanup 前に main checkout の `docs/adr/` へコピー保全**（`docs/handoff/m4-15.md` §(f) / `docs/handoff/m4-18.md` §(d)-6）。

---

## WP 別 owner タスク一覧（cross-reference）

| WP | owner チケット | カテゴリ | CC-side status |
|----|--------------|---------|----------------|
| M4-01 | T27 npm / T28 browser spot check / **T02 WebGPU ADR（両方）** | §4.1 / §2.4 / §5.4 | ✅ WebGPU backend + WASM + npm CD dry-run |
| M4-02 | （依頼者 0、申し送り = UNITY_LICENSE） | §4.4 | ✅ staticlib + wasm-harness leg green |
| M4-03 | （依頼者 0、acoustic 検収は M4-05/06 吸収） | §6.3 | ✅ aec op（CPU、GPU seam なし） |
| M4-04 | T20 DAC/Mimi sign-off / T21 parity-rvq dispatch | §3.2 / §1.6 / §4.5 | ✅ mimi_rvq + dac_rvq + parity scaffold |
| M4-05 | T29 checkpoint+license / T30 legal+streaming demo | §1.1 / §3.1 / §2.6 | ✅ CSM native + flip-the-switch harness |
| M4-06 | T29 weight+license / T30 full-duplex demo+dispatch | §1.2 / §3.1 / §2.6 | ✅ Moshi native + attribution 3 面 |
| M4-07 | T17 Hopper 有効化 / T18 FA v2 比+dashboard | §2.1 | ✅ FA v3 kernel + 3-way dispatch + gated tests |
| M4-08 | T14 board+dump / T15 LicheePi 4A / T16 Milk-V Duo | §2.2 | ✅ RVV 0.7.1 probe/dispatch/kernel + cross-build CI |
| M4-09 | T05 G2P 方針 3 択（ADR §8） | §5.1 | ✅ ADR 材料 3 pack + Proposed 草案 |
| M4-10 | T08 MLIR 採否（ADR §13） | §5.2 | ✅ ADR 材料 + supersede 系譜 + Proposed 草案 |
| M4-11 | T11 Web 実機 / T12 CDN / T13 matrix sign-off | §2.4 / §4.2 / §7 | ✅ support-matrix + drift check + CDN 材料 |
| M4-12 | （依頼者 0、records-only、rc タグで実行済） | — | ✅ rc baseline snapshot + anchor rotation（凍結非発火） |
| M4-13 | T16 glslc .spv / T17 Android soak / T18 dispatch | §4.3 / §2.3 / §4.5 | ✅ Vulkan 完成（CC-side、`.spv` placeholder-then-swap） |
| M4-14 | T09 convert / T10 dispatch+atol / T11 weight sign-off | §1.3 / §3.3 | ✅ 3-size CI matrix + per-size atol lookup |
| M4-15 | T10 SBOM reproducible verify | carry-over | ✅ vulkan-only build target + SPDX SBOM + scanner |
| M4-16 | T14 WavTokenizer/X-Codec 2 sign-off（dual-license） | §3.4 / §1.7 | ✅ fsq_codec op（合成 fixture）+ EXPERIMENTAL 記録 |
| M4-17 | T23 x86 cloud VM perf / T24 ARM64 実機 perf | §2.5 | ✅ CPU ISA server tier kernel + probe + selftest |
| M4-18 | T02 UTMOS weight+license / T03 DNSMOS / **T01 gate（両方）** | §1.5 / §3.5 / §5.3 | ✅ UTMOS harness（NO-GO-defer、weight 非依存分完成） |
| M4-19 | （依頼者 0、申し送り = M5Stack/community/WER/CLI wiring） | carry-over | ✅ Wyoming completion（accept loop / synthesize / barge-in） |
| M4-20 | T17 DeepFilterNet parity / T18 sign-off+GTCRN | §1.4 / §3.6 | ✅ denoise/agc/hpf/loudness/speaker_verify + word-ts interface |

**カテゴリ別内訳（distinct owner チケット）**: §1 weight parity = 8（M4-05-T29 / M4-06-T29 / M4-14-T09,T10 / M4-20-T17 / M4-18-T02 / M4-04-T21 / M4-05-T14+T34=CC harness）/ §2 実機 = 12（M4-07-T17,T18 / M4-08-T14,T15,T16 / M4-13-T17 / M4-01-T28 / M4-11-T11 / M4-17-T23,T24 / M4-05-T30 / M4-06-T30）/ §3 license = 8（M4-05-T29 / M4-06-T29 / M4-14-T11 / M4-04-T20 / M4-16-T14 / M4-18-T02,T03 / M4-20-T18 + M3 queue）/ §4 infra = 6（M4-01-T27 / M4-11-T12 / M4-13-T16 / M4-02 UNITY / M4-13-T18 + 初回 dispatch 群）/ §5 ADR = 4（M4-09-T05 / M4-10-T08 / M4-18-T01 / M4-01-T02）。**一部チケットは複数カテゴリに跨る**（例: M4-05-T29 = weight sourcing §1 + license §3、M4-14-T10 = dispatch §4 + atol §1）。

---

## Contact / Escalation

- owner 検証で CC 側 fixup が必要になった場合（parity 差し戻し / license 分類変更 / build script 齟齬 / turbo atol calibrate 等）は本チェックリストの該当項に追記して依頼者から CC に振る。
- **v1.0-rc close 判定**は上記 §1〜§5 の owner タスク消化 + `docs/milestones.md` §8 各 WP Exit criteria + `docs/platform-support/v1.0-rc-support-matrix.md`（M4-11-T13 承認）を根拠に依頼者が最終判断（OSS 機能完成 rc 判定、商用 GA ではない）。
- **hard gate G7**（§5.1 M4-09 / §5.2 M4-10）は **M4-12 v1.0-rc ABI baseline 記録前に確定必須**。M4-12 は records-only で rc タグ実行済（`docs/handoff/m4-12.md` §(g)）だが、G2P / MLIR 判断が API 面 / 配布物構成に影響する場合は baseline を再スナップショットする。
- **参照 SoT**: `docs/milestones.md` §8（WP 一覧・Exit criteria・hard gate G1〜G8）/ `docs/tickets/m4/README.md`（wave 計画 + owner track）/ `docs/handoff/m4-*.md`（各 WP 引き渡し）/ `docs/platform-support/v1.0-rc-support-matrix.md`（全プラットフォーム確認）/ `docs/license-audit.md` §3.1（sign-off template）/ CLAUDE.md「M4（v1.0-rc）🚧 実装開始」節。
