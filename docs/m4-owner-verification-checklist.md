# M4 (v1.0-rc) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — 実機テスト・実 weight sourcing・法務 sign-off・外部契約 / インフラ provisioning・ADR 判断を担当。
**CC-side status**: **M4 CC 実装 terminal 到達（2026-07-15、全 20 WP = M4-01〜M4-20）**。investigation 3 round のうち round 2 / round 3 = **2 連続 0 CC ticket** で terminal 判定（M3 と同じ規律）。terminal 時 verify = default 2340 / all-features 2364 passed。**その後 2026-07-16 に依頼者指示で post-terminal CC-gap 追加実装 campaign を実施**（terminal 後の追加洗い出し = ultracode 32 候補中 17 land、既存 WP 内の完成度向上 = P0 wasm ビルド破損修正・converter alignment_heads/word-timestamp・vokra-server 本番 startup 配線・CSM/Mimi from_gguf・実 M1 Metal parity + Llama MSL kernel 4種・RingKVCache・agc/hpf streaming 等）。**merge 状況（2026-07-19 更新）**: **M4 は PR #8 として main に merge 済**（merge commit `ff12104`、2026-07-19、branch `feat/m4-plan-and-wave1` → main）。本チェックリスト起草時点の「PR 未作成」前提は失効しており、**「default branch に workflow ファイルが無いので workflow_dispatch できない」というブロッカーも解消済**（§4.5）。以降 main は `13a2a6e` まで進んでいる。

**verify の数値について（honest）**: **default 2418 / all-features 2443 passed / 0 failed / 4 ignored** は **2026-07-16 の post-terminal campaign 時点で記録された snapshot** であり、その後 land した実 weight 評価 campaign 1/2 と M5-14 でテストが追加されているため **現 HEAD の値ではない**。正確な現在値は full suite（`cargo test --workspace`）を実行した時に **実行条件（debug/release・doctests 有無）と併記して re-pin** すること — 条件が違うと値が変わるため、条件抜きの数字は比較できない。C ABI は新規追加なし（rc baseline 33 fn 不変）。cargo fmt / clippy `-D warnings` / `scripts/check-zero-deps.sh`（root Cargo.lock = `vokra-*` のみ、NFR-DS-02）/ `scripts/check-abi-changelog.sh` / `scripts/check-platform-support.sh`（anchors 50）は各 wave で green を確認している。**本チェックリストの owner タスクは、後続 campaign で CC 到達分が伸びた項目（§1.2 / §1.3 / §1.4 / §1.6 / §1.7 / §4.3）で縮小した**が、実機 / license sign-off / 外部インフラ / owner ADR / CI 初回 dispatch は**いずれも消えていない**。

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
- **(b)**: HF gated repo のアカウント承諾が必要、weight は CC 機体に無い。**blocker は 2026-07-17 campaign-2 の probe で 2 点に特定済** = repo metadata は 200 で到達する（`gated: auto` / apache-2.0）が **file resolve が 401** → 残るのは **(i) gate 受諾 + (ii) fresh HF token** のみ。CC 側 harness / binding は完成済ゆえ、受諾後ただちに発火する。
  <!-- claim-evidence: docs/bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md#CSM probe -->
- **(c)**: spec `M4-05-T29`（owner）。CC 到達分 = Llama-3.2-flavor backbone + depth transformer + Mimi neural chain + `assert_vs_hf_reference` harness。
- **(d)**: checkpoint 手渡し済 + flip-the-switch テストが実 checkpoint で実行可能。

### 1.2 Moshi（M4-06）

- **(a)**: **weight sourcing は完了**（2026-07-17 campaign-2）。HF `kyutai/moshiko-pytorch-bf16` = 14.32 GiB を取得済（NOTICE §5 記載 source、license を live で cc-by-4.0 確認）。切詰めモデル（temporal 32→2 / depformer 6→2、95 tensor、実 tensor bytes verbatim）で **backbone hidden max\|Δ\| 8.249e-5 / text logits 2.360e-5 / emitted frame 11/11 bit-exact**（text + 8 codebook、delay ring 込み full greedy）を実測、env-gated `parity_moshi.rs` が `is_synthesized=false` assert 付きで実発火。attribution 2 面（CLI banner + C ABI `vokra_model_attribution`）も実 GGUF から正しい CC-BY 4.0 文言を返す。
  <!-- claim-evidence: docs/bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md#moshiko-pytorch-bf16 -->
- **(b)**: **残っているのは weight の有無ではなく full-7B の実行環境**。16 GB 級機では converter ~97 GiB / torch fp32 dump ~43 GiB / `MoshiEngine::from_path` ~60 GiB と**全段 BLOCKED**（mmap 未配線）。したがって owner の判断が要るのは (i) large-RAM runner を用意するか streaming converter + mmap 配線を CC に出すか の戦略選択、(ii) `parity-moshi-real.yml` の再設計（現状の ubuntu-latest 記述のままでは完走不能 = **この dispatch は先に (i) を解いてからでないと確実に赤**）、(iii) license sign-off（§3.1）。
- **(c)**: spec `M4-06-T29`（owner）。CC 到達分 = 実 weight での切詰め parity + duplex 実走 + attribution 3 面。
- **(d)**: full-7B 戦略の確定 + `parity-moshi-real.yml` の再設計 + flip-the-switch 発火。**PCM quality gate は実 Mimi weight mapping 待ちの by-design refusal のまま**（捏造せず verbatim 保存）。

### 1.3 Whisper small / medium / turbo（M4-14 = M2-06 carry-over）

- **(a)**: **変換 + real-audio fixture 再生成 + commit は完了済**（`9d3eaae` → PR #8 `ff12104`）。`tests/parity/whisper_{base,small,medium,turbo}/manifest.txt` は 4 サイズとも `pcm_source = tests/fixtures/audio/jfk-30s.wav` / `pcm_len = 480000` / `pcm_sha256 = 58adb4ea…` を持つ = **synthetic 1 秒ではなく実 JFK 音声由来**。owner レビュー後の手動 commit という red-line も守られている（auto-commit していない）。残る owner 作業は **`parity-whisper-real.yml` の 3-size 初回 workflow_dispatch**（M4-14-T10）。
  <!-- claim-evidence: tests/parity/whisper_small/manifest.txt#jfk-30s.wav -->
  <!-- claim-evidence: tests/parity/whisper_turbo/manifest.txt#jfk-30s.wav -->
- **(b)**: HF DL を要する CI 起動は owner 引き渡し（§4.5）。**turbo atol の判断材料は取得済** = 2026-07-16 の実 weight 評価で turbo の M4-14 P1 懸念は非発現（gated leg 8/8 pass、atol 0.01 に対し余裕 ~3 桁）= **現時点で calibrate は不要**。dispatch が想定外の max\|Δ\| を出した場合のみ実測値を CC に渡して honest calibrate（実測前に atol を緩めない）。
  <!-- claim-evidence: docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md#atol calibrate -->
- **(c)**: spec `M4-14-T09`（変換 + fixture 再生成 = **完了**）/ `M4-14-T10`（初回 dispatch + turbo atol 判断 = **残**）。CC 到達分 = shape-driven converter（5 サイズ）+ per-size atol lookup 基盤（default 0.01 維持）+ 3-size matrix CI + dumper turbo `vocab_resource` 標準化（turbo は large-v3 と同一 51866 vocab）+ 4 サイズの転写 byte 一致実測。
- **(d)**: 3-size parity leg が実 checkpoint で完走 + weight sign-off（§3.3）。**honest 残置**: `tests/parity/whisper_large_v3/` だけは M0-06 期の合成 1 秒 fixture（`pcm_len = 16000`、`pcm_source` なし）のままで、M4-14 の 3-size スコープ外。「Whisper family の parity fixture は全て実音声由来」と読み替えないこと。

### 1.4 DeepFilterNet（M4-20）

- **(a)**: **T17 の実 checkpoint parity は達成済**（2026-07-17、`9b718d1`）。libDF topology を転写した real 実装（Vorbis-window STFT/ERB frontend + conv/GRU encoder + ERB/DF decoder + lookahead deep filtering、**115 個の upstream 名 tensor**）で、**SI-SNR 14.768399 dB vs upstream 14.768398 dB（gap 2.0e-7 dB）/ enhanced 波形 max\|Δ\| 4.17e-7 / 21 stage tap 全 PASS**。license も dual MIT/Apache-2.0 として一次確認済。
  <!-- claim-evidence: crates/vokra-ops/tests/parity_denoise_dfn3.rs -->
  <!-- claim-evidence: docs/license-audit.md#14.768399 -->
- **(b)**: **残 owner は T18 の最終 license sign-off のみ**（§3.6）。`docs/license-audit.md` §3.1 の DeepFilterNet3 行は sign-off 欄が空 = fail-closed が稼働中で、記入まで公式 zoo 配布は発生しない。**CC は sign-off 欄を pre-fill しない**。
- **(c)**: spec `M4-20-T17`（**達成済**）/ `M4-20-T18`（owner sign-off = **残**）。CC 到達分 = `crates/vokra-ops/src/denoise.rs` の real topology 実装 + env-gated `parity_denoise_dfn3` + `vokra.denoise.*` schema v2 + `convert --model denoise`。
- **(d)**: §3.6 の sign-off 記入で完了（parity 側の完了条件は充足済）。

### 1.5 UTMOS / DNSMOS（M4-18）

- **(a)**: **kickoff 週 gate の自動 defer は 2026-07-18 の依頼者承認で解除済（un-defer）**。campaign-2 の utmos-probe が weight の匿名取得可・license 全鎖 permissive・BVCC に academic 限定条項なしを一次資料で確認したため、defer の根拠（weight/license 未着）が消滅した。**UTMOS 実装一式の owner は M5-15（T14–T22）へ移管**され、M4 残渣ではない。
- **(b)**: 本項で owner に残るのは **(i) DNSMOS の採否**（M5-15-T23。Microsoft P.835 系の license 一次資料検証、商用不可なら UTMOS 単独 scope に縮小 = **fail-closed を継続**）と **(ii) 評価用 weight の公式 zoo 掲載判断**（M5-15-T24）の 2 点のみ。**加えて §3.5 側に残る owner entry**（weight source URL の確定 = `tests/parity/utmos/source.env` の commit、§3.1 UTMOS 行の license sign-off、`parity-utmos` 初回 workflow_dispatch）は消えていない — 下記 (c)/(d) の通り、CC 側が到達した分だけ残作業が縮んだのであって置き換わってはいない。
- **(c)**: 実装 spec は M5-15（gitignore ローカル）。handoff `docs/handoff/m4-18.md` §(d) queue は un-defer 前の記述である点に注意（`docs/adr/M4-18-utmos-gate.md` に supersede 注記を追記済）。**CC 到達分（2026-07-20 時点、起草時から前進）** = `crates/vokra-eval/src/metrics/utmos.rs`（v0 skeleton に加え **v1 = UTMOS22-strong 実 topology**）+ `AudioMosMetric` trait + `parity_utmos.rs`（final score）+ `parity_utmos_stages.rs`（stage 別）の 2 harness + **converter `vokra-convert --model utmos`（M5-15 T14、実 land 済）** + **reference fixture の commit 済み**（`tests/parity/utmos/score.json` + `ref-clip.wav`）。**「fixture 未 commit」という起草時の記述は失効** — 実 upstream 実装を import して生成した reference が入っており（mirror 禁止 = Kokoro `92dbc92` の教訓）、M1 iMac 実測で全 stage + score が upstream 一致（score max\|Δ\| 1.192e-7）。捏造禁止の原則自体は不変（atol は測定由来、`score.json` の `provenance` に導出を記録）。
  <!-- claim-evidence: tests/parity/utmos/score.json -->
  <!-- claim-evidence: crates/vokra-convert/src/models/utmos.rs -->
- **(d)**: M5-15 側で `parity-utmos` 初回 workflow_dispatch 完走（初回起動は owner）+ DNSMOS 判定記録。**dispatch の残ブロッカーは 1 点に特定済** = `tests/parity/utmos/source.env`（weight URL + sha256）が未 commit で、これは §3.1 の UTMOS license sign-off 待ち。checkpoint 自体は永久に非 commit（Vokra は weight を配布しない）ゆえ、source.env が入るまで workflow は明示 annotation 付きで clean skip する（捏造 pass しない）。**§5.3 の G2（UTMOS defer 中の暫定判定方針）は un-defer により前提が変わった** — 追認対象は「defer 継続の是非」ではなく「M5-15 着地までの暫定 posture」になる。

### 1.6 Mimi / DAC real-checkpoint parity（M4-04 / M4-05）

- **(a)**: **`parity-rvq-real.yml` の recipe はローカルで完全 first-fire 済**（2026-07-16）= checkpoint sha256 が workflow の pin と MATCH、venv pin 一致、**106/106 tensor PASS**、`real_codec_parity` 2/2。**Mimi は PCM roundtrip まで完通**（encode code 4384/4384 = 100% 一致、decode max\|Δ\| 3.67e-6、`ebe1cc5`）。したがって「CC 機体では未実行」は失効。
  <!-- claim-evidence: docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md#106/106 -->
  <!-- claim-evidence: docs/bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md#4384/4384 -->
- **(b)**: **owner に残るのは 2 点**: (i) **GitHub Actions 上の正式な初回 workflow_dispatch**（ローカル first-fire は Actions 実行の代替にならない — runner 環境 / HF DL 経路は未検証）、(ii) **encodec leg は今回未実行**（EnCodec weight は CC-BY-NC ゆえ取得可否自体が owner の判断 = §3 の fail-closed 姿勢を維持するか研究用途で取得するか）。
- **(c)**: spec `M4-04-T21`（owner 初回 dispatch）/ `M4-05-T14`・`T34`（CC parity = ローカル発火済）。CC 到達分 = `mimi_rvq` / `dac_rvq` op family + converter + `parity-rvq-real.yml` scaffold + upstream reference fixture 契約 + 実 checkpoint での first-fire。
- **(d)**: 初回 dispatch が success（差し戻し fix 後を含む）+ encodec leg の可否判断。required check への promotion はしない前提を維持（HF flakiness、Kokoro/Whisper real CI と同運用）。

### 1.7 WavTokenizer / X-Codec 2（M4-16、FSQ）

- **(a)**: **WavTokenizer は実 pretrained weight で検証済**（2026-07-16）= 実 codebook `[4096, 512]` + jfk 由来の実 codes 440 個で **max Δ = 0.0（bit-identical、atol 1e-6）**。「合成 weight のみ・pretrained 未 DL」は失効。**X-Codec 2 は honest skip** = HF `HKUSTAudio/xcodec2` の `cc-by-nc-4.0` を live 確認して fail-closed で取得を止めた（released shape + 合成 projection での max Δ = 0.0 までは検証）。
  <!-- claim-evidence: docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md#cc-by-nc-4.0 -->
- **(b)**: 実 weight を止めているのは能力ではなく **license 判断**（§3.4 の X-Codec 2 ruling）。**ruling が出るまで `crates/vokra-core/src/compliance/license_class.rs` が X-Codec 2 を Permissive と分類しており、NC weight を素通しする穴が開いたまま** = ruling 後の 1 行 flip は CC 側の待機タスク（owner の判定が先）。
  <!-- claim-evidence: crates/vokra-core/src/compliance/license_class.rs#x-codec-2 -->
- **(c)**: spec `M4-16`（op のみ）+ 消費者 WP（converter metadata `documented`→`persisted` + 実モデル e2e）。CC 到達分 = fsq op family + 実 codebook parity + license 事実の一次確認。
- **(d)**: §3.4 の sign-off + X-Codec 2 ruling。**op parity 側は実 weight で完了済**ゆえ、本カテゴリの owner 残は license に集約される（この点は従来どおり）。

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
- **(c)**: spec `M4-13-T17`（owner、M3-18 併走）+ handover `docs/m3-18-android-rtf-handover.md`。**依存だった M4-13-T16 の glslc `.spv` commit は完了済（§4.3）= 本 soak の前提は解除済、着手可**（12 `.spv` が `crates/vokra-backend-vulkan/kernels/precompiled/` に commit 済 + `spirv::` 24 tests green = §4.3 (d) の記録どおり）。coop-matrix 非対応チップで RTF 2x 劣化なら subgroup INT8/FP16 kernel 追加 wave（別 WP へ flow）。
  <!-- claim-evidence: crates/vokra-backend-vulkan/kernels/precompiled/SHA256SUMS#gemm_coopmat.spv -->
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
- **(b)**: weight source + license 一次資料の確認（本 spec に推測 URL / license を書かない = ハルシネーション厳禁）。**kickoff 週 gate の自動 defer は 2026-07-18 の依頼者承認で解除済（un-defer）** — campaign-2 の probe が weight 匿名取得可 + license 全鎖 permissive + BVCC に academic 限定条項なしを一次資料で確認したため。UTMOS 実装は M5-15（T14–T22）が owner。`docs/handoff/m4-18.md` §(a) の NO-GO-defer 記述は un-defer 前のもの。**DNSMOS は依頼者の最終確認まで fail-closed 継続**。
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

- **(a)**: **完了（owner 作業は残っていない）**。12 `.spv`（gemm_subgroup / gemm_coopmat / gemv / softmax / softmax_causal / layer_norm / gelu / conv1d / elementwise / activation / transpose / gather）が `crates/vokra-backend-vulkan/kernels/precompiled/` に commit 済（PR #8 `ff12104`）。`PROVENANCE` が compiler を **glslangValidator 11:16.4.0** に pin、`SHA256SUMS` + `spirv.rs::SHADERS` の `expected_sha256_hex` + `include_bytes!` arm も揃っている。
  <!-- claim-evidence: crates/vokra-backend-vulkan/kernels/precompiled/PROVENANCE#glslangValidator -->
  <!-- claim-evidence: crates/vokra-backend-vulkan/src/spirv.rs#include_bytes! -->
- **(b)**: 「glslc toolchain が developer 機に無い」という前提が失効した（glslangValidator 16.4.0 導入済）ため CC 側で消化済。**この完了により §2.3 Android soak の前提（M4-13-T16 が先）が解除され、owner は Android soak に着手できる**。
- **(c)**: spec `M4-13-T16`（**完了**）+ ADR M3-02 §4-(a) + `scripts/install-vulkan-toolchain.md` + handoff `docs/handoff/m4-15.md` §(b)。
- **(d)**: **達成済** — `cargo test -p vokra-backend-vulkan --lib spirv::` = **24 passed / 0 failed**（`verify_pinned_hashes_is_ok_for_committed_blobs` / `sha256sums_file_matches_manifest_pins` 含む、2026-07-19 実行）。残る Vulkan の owner タスクは §2.3 実機 soak と §4.5 の lavapipe 初回 dispatch のみ。

### 4.4 secrets.UNITY_LICENSE provisioning（M4-02、M2-11 carry-over）

- **(a)**: `nightly-webgl.yml` の Unity leg + `nightly-il2cpp.yml` を license-gated skip から発火させる Unity license secret を provision。
- **(b)**: Unity ライセンス契約 = owner。CC 側は `.a` staticlib build / thread-free gate / wasm-harness leg（license 不要、node で実 VAD 完走）まで完成。
- **(c)**: handoff `docs/handoff/m4-02.md` §4-1 + spec `M4-02`（依頼者 0 件 = 申し送りのみ、`nightly-il2cpp.yml` 冒頭コメントが一次ソース）。
- **(d)**: `nightly-webgl.yml` 初回 workflow_dispatch で wasm-harness leg 即 green + Unity leg が license 後 green。

### 4.5 CI 初回 workflow_dispatch 群（owner 引き渡し前例）

- **(a)**: CC 機体で未実行の workflow を GitHub Actions 上で初回起動: `parity-rvq-real.yml`（M4-04-T21）/ `parity-whisper-real.yml` 3-size（M4-14-T10）/ `parity-moshi-real.yml`（M4-06-T30）/ `parity-utmos.yml`（M5-15 の flip 後）/ `gpu-vulkan-parity.yml` lavapipe（M4-13-T18）/ `web-wasm.yml` + `npm-web-release` dry-run（M4-01-T28）/ `parity-kokoro-real.yml` の再 dispatch（`92dbc92` で reference 側を修正したため）。
- **(b)**: 初回 run は owner 引き渡し（プロジェクト前例、M2/M3 と同）。green / honest skip 理由の可視を確認。**PR #8 が merge 済（`ff12104`）なので「default branch に workflow が無い」という以前の前提ブロッカーは解消済**。
- **(c)**: 各 spec の owner ticket + 各 workflow YAML の "initial workflow_dispatch is owner handoff" コメント。
- **(d)**: 各初回 run が green（or skip 理由が honest に可視）。**required check への promotion は数週連続 green 後の owner 判断**（HF flakiness の PR blocking 回避、Kokoro/Whisper real CI と同運用）。
- **(e) sequencing 拘束（順序を守らないと確実に赤くなる — 起動前に確認）**:
  - `parity-moshi-real.yml` は **§1.2 (b)-(i) の full-7B 戦略が解けてから**。現状 16 GB 級では converter/dump/load が全段 BLOCKED ゆえ、先に起動しても環境要因で赤になるだけで情報が得られない。
  - `godot-crossbuild.yml` は **Godot compliance scanner の bash 修正が land してから**（`scripts/compliance/check-godot-package-no-nvidia.sh` の配列長展開が bash>=4.4 でのみ fail する既知欠陥。ローカル bash 3.2 では再現しない = CI でのみ露見）。**本 doc-hygiene WP ではこの修正を実装していない**（owning WP 側の作業）。
  - `parity-csm-real` 系は **§1.1 の HF gate 受諾 + fresh token 後**。
  - Android soak（§2.3）は §4.3 完了により **前提解除済 = 即着手可**。

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

### 5.3 M4-18 G2 — M4-05/M4-06 品質判定方針の追認（UTMOS un-defer 済）

**前提更新（2026-07-18 依頼者承認の un-defer）**: 本項起草時の前提「UTMOS は defer 中で判定不能」は失効した（§1.5）。**追認対象は「defer 継続の是非」ではなく「M5-15 で UTMOS が着地するまでの暫定 posture」**に変わる。**owner の追認タスク自体は残る**（下記 (a)〜(d)）— UTMOS 着地までは mel_loss 単騎で品質判定する期間が実在し、ratified 条件からの一時的な乖離を owner が記録する必要は un-defer 後も変わらないため。

- **(a)**: ratified 完了条件「MEL loss / UTMOS 劣化 5% 未満」は **UTMOS 着地前（M5-15 まで）** honest に判定不能ゆえ、**「mel_loss 5% gate + UTMOS は advisory（判定不能を明示）」への切替を追認・記録**（ratified 条件の変更 = owner 専権）。
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

**M4-19 Wyoming 申し送り（本 WP 完了条件外 = scope-guard）**: M5Stack / HA Voice Satellite soak（optional）/ Wyoming・HA community engagement（X-05、own pace）/ **faster-whisper real-WER 検証**（`m4_19_asr_real_gguf_round_trip_gated` を `VOKRA_WHISPER_BASE_GGUF` + `VOKRA_PIPER_GGUF` に実 GGUF を与えて実行）。詳細 = `docs/handoff/m4-19.md` §Owner tasks。

> **CLI model wiring（旧 M2-09-T04 carry-over）は完了済につき本 queue から除去**。production 起動経路は `integrations/vokra-server/src/server.rs` の `run_with_config`（:65）→ `build_service`（:77、定義 :348）→ `InferenceService::build`（:353）→ `spawn_server_wired`（:84、定義 :293）で、**listener を bind する前に registry を同期構築する**。model path が未設定なら health-only + Wyoming discovery-only で起動し、設定されているのに GGUF が欠損 / 破損していれば **hard startup error**（FR-EX-08 = 半配線のまま port を開いて全 request を 404 / no-op する状態を作らない）。実 flag 名は `integrations/vokra-server/src/config.rs` の `--whisper-base`（:462）/ `--piper-plus`（:518）/ `--piper-g2p`（:537）で、`integrations/vokra-server/README.md` の quickstart と一致する。**`spawn_server_with_service`（`server.rs`:130）は Wyoming 統合テスト専用パス**（実 Whisper GGUF なしで mock backend を駆動するために HTTP listener を health-only のまま残す）であり、production の配線先ではない。
> <!-- claim-evidence: integrations/vokra-server/src/config.rs#--whisper-base -->
> <!-- claim-evidence: integrations/vokra-server/src/config.rs#--piper-plus -->
> <!-- claim-evidence: integrations/vokra-server/src/server.rs#spawn_server_wired -->
>
> **実行時の落とし穴**: piper voice GGUF は `spk_proj.0.weight` を持つものを渡すこと。`crates/vokra-models/src/piper_plus/config.rs` の `Dims::derive`（:136）がこのテンソルを無条件に要求するため、持たない voice は load 時点で `InvalidArgument` で loudly に落ちる（FR-EX-08 としては正しい挙動だが、原因が分からないと詰まる）。M4-residual 監査 注記 3 は公開 voice の一部がこれに該当したと記録している。
> <!-- claim-evidence: crates/vokra-models/src/piper_plus/config.rs#spk_proj.0.weight -->

**M4-15 SBOM reproducible-build verify（owner 1 件）**: 同一入力から SPDX SBOM が別マシン / 別 checkout で byte-identical に再生成されることを確認（`docs/handoff/m4-15.md` §(a)、spec `M4-15-T10`）。差分は環境依存フィールドを記録して CC に fixup 依頼（fabricated pass 禁止）。

**worktree 産 ADR の main checkout への sync（owner）— 完了・close**: 対象だった M4-14 / M4-15 / M4-18（gate・arch）の 4 本はいずれも main checkout の `docs/adr/`（gitignore ローカル）に実在することを確認済。**新たな sync 作業は残っていない**。以後 worktree で ADR を起こした場合の保全手順としてのみ `docs/handoff/m4-15.md` §(f) / `docs/handoff/m4-18.md` §(d)-6 を参照する。

---

## WP 別 owner タスク一覧（cross-reference）

| WP | owner チケット | カテゴリ | CC-side status |
|----|--------------|---------|----------------|
| M4-01 | T27 npm / T28 browser spot check / **T02 WebGPU ADR（両方）** | §4.1 / §2.4 / §5.4 | ✅ WebGPU backend + WASM + npm CD dry-run |
| M4-02 | （依頼者 0、申し送り = UNITY_LICENSE） | §4.4 | ✅ staticlib + wasm-harness leg green |
| M4-03 | （依頼者 0、acoustic 検収は M4-05/06 吸収） | §6.3 | ✅ aec op（CPU、GPU seam なし） |
| M4-04 | T20 DAC/Mimi sign-off / T21 parity-rvq dispatch | §3.2 / §1.6 / §4.5 | ✅ mimi_rvq + dac_rvq + parity scaffold + **実 checkpoint ローカル first-fire 106/106**（残 = Actions 初回 dispatch + encodec leg） |
| M4-05 | T29 checkpoint+license / T30 legal+streaming demo | §1.1 / §3.1 / §2.6 | ✅ CSM native + flip-the-switch harness（blocker は HF gate 受諾 + fresh token の 2 点に特定済） |
| M4-06 | T29 weight+license / T30 full-duplex demo+dispatch | §1.2 / §3.1 / §2.6 | ✅ Moshi native + attribution 3 面 + **実 weight 14.32 GiB で切詰め parity 11/11 bit-exact**（残 = full-7B RAM 戦略 + dispatch 再設計） |
| M4-07 | T17 Hopper 有効化 / T18 FA v2 比+dashboard | §2.1 | ✅ FA v3 kernel + 3-way dispatch + gated tests |
| M4-08 | T14 board+dump / T15 LicheePi 4A / T16 Milk-V Duo | §2.2 | ✅ RVV 0.7.1 probe/dispatch/kernel + cross-build CI |
| M4-09 | T05 G2P 方針 3 択（ADR §8） | §5.1 | ✅ ADR 材料 3 pack + Proposed 草案 |
| M4-10 | T08 MLIR 採否（ADR §13） | §5.2 | ✅ ADR 材料 + supersede 系譜 + Proposed 草案 |
| M4-11 | T11 Web 実機 / T12 CDN / T13 matrix sign-off | §2.4 / §4.2 / §7 | ✅ support-matrix + drift check + CDN 材料 |
| M4-12 | （依頼者 0、records-only、rc タグで実行済） | — | ✅ rc baseline snapshot + anchor rotation（凍結非発火） |
| M4-13 | ~~T16 glslc .spv~~（完了）/ T17 Android soak / T18 dispatch | §4.3 / §2.3 / §4.5 | ✅ Vulkan 完成 + **12 `.spv` commit 済（`spirv::` 24 tests green）= T17 の前提解除** |
| M4-14 | ~~T09 convert~~（完了）/ T10 dispatch+atol / T11 weight sign-off | §1.3 / §3.3 | ✅ 3-size CI matrix + per-size atol lookup + **4 サイズ fixture が実 jfk 音声由来**（turbo calibrate 不要の材料込み） |
| M4-15 | T10 SBOM reproducible verify | carry-over | ✅ vulkan-only build target + SPDX SBOM + scanner |
| M4-16 | T14 WavTokenizer/X-Codec 2 sign-off（dual-license） | §3.4 / §1.7 | ✅ fsq_codec op + **WavTokenizer 実 codebook で Δ 0.0 bit-identical** / X-Codec 2 は NC を live 確認し honest skip |
| M4-17 | T23 x86 cloud VM perf / T24 ARM64 実機 perf | §2.5 | ✅ CPU ISA server tier kernel + probe + selftest |
| M4-18 | **un-defer 済（2026-07-18）→ 実装 owner は M5-15 T14–T22** / 残 = DNSMOS 採否 + zoo 掲載 **+ §3.5 の license sign-off + weight URL（`source.env`）確定 + `parity-utmos` 初回 dispatch** | §1.5 / §3.5 / §5.3 | ✅ UTMOS harness 2 本（final + stage 別）+ converter + **実 upstream 由来 reference fixture commit 済** |
| M4-19 | （依頼者 0、申し送り = M5Stack / community / real-WER。**CLI wiring は完了につき除去**） | carry-over | ✅ Wyoming completion（accept loop / synthesize / barge-in）+ 起動時 registry 構築 |
| M4-20 | ~~T17 DeepFilterNet parity~~（達成済）/ T18 sign-off+GTCRN | §1.4 / §3.6 | ✅ denoise/agc/hpf/loudness/speaker_verify + word-ts interface + **DFN3 実 checkpoint parity（SI-SNR gap 2.0e-7 dB）** |

**カテゴリ別内訳（distinct owner チケット）**: 件数は **open（owner 未消化）のみ**を数え、上の WP 別表で取り消し線を入れたチケットは「完了」として併記する（表と tally が食い違わないための規律）。

- **§1 weight parity = open 4**（M4-05-T29 / M4-06-T29 / M4-14-T10 / M4-04-T21）＋ **完了 2**（M4-14-T09 = 変換 + fixture 再生成 / M4-20-T17 = DFN3 実 checkpoint parity）＋ **M5-15 移管 1**（M4-18-T02 = UTMOS 実装面、§1.5。license sign-off 面は §3 に残置）＋ M4-05-T14+T34 は CC harness（ローカル発火済 = owner 作業なし）
- **§2 実機 = open 12**（M4-07-T17,T18 / M4-08-T14,T15,T16 / M4-13-T17 / M4-01-T28 / M4-11-T11 / M4-17-T23,T24 / M4-05-T30 / M4-06-T30）
- **§3 license = open 8**（M4-05-T29 / M4-06-T29 / M4-14-T11 / M4-04-T20 / M4-16-T14 / M4-18-T02,T03 / M4-20-T18）＋ **M3 carry-over queue**（§3.7 = CosyVoice2 / Voxtral / Mimi の sign-off、M4 の 8 件には数えないが owner 残としては存置）
- **§4 infra = open 5**（M4-01-T27 / M4-11-T12 / M4-02 UNITY / M4-13-T18 + 初回 dispatch 群）＋ **完了 1**（M4-13-T16 = glslc `.spv` commit、§4.3 = §2.3 Android soak の前提解除）
- **§5 ADR = open 4**（M4-09-T05 / M4-10-T08 / M4-18-T01 / M4-01-T02）

**一部チケットは複数カテゴリに跨る**（例: M4-05-T29 = weight sourcing §1 + license §3、M4-14-T10 = dispatch §4 + atol §1、M4-18-T02 = §1 は M5-15 移管 / §3 license 面は残置）。

---

## Contact / Escalation

- owner 検証で CC 側 fixup が必要になった場合（parity 差し戻し / license 分類変更 / build script 齟齬 / turbo atol calibrate 等）は本チェックリストの該当項に追記して依頼者から CC に振る。
- **v1.0-rc close 判定**は上記 §1〜§5 の owner タスク消化 + `docs/milestones.md` §8 各 WP Exit criteria + `docs/platform-support/v1.0-rc-support-matrix.md`（M4-11-T13 承認）を根拠に依頼者が最終判断（OSS 機能完成 rc 判定、商用 GA ではない）。
- **hard gate G7**（§5.1 M4-09 / §5.2 M4-10）は **M4-12 v1.0-rc ABI baseline 記録前に確定必須**。M4-12 は records-only で rc タグ実行済（`docs/handoff/m4-12.md` §(g)）だが、G2P / MLIR 判断が API 面 / 配布物構成に影響する場合は baseline を再スナップショットする。
- **参照 SoT**: `docs/milestones.md` §8（WP 一覧・Exit criteria・hard gate G1〜G8）/ `docs/tickets/m4/README.md`（wave 計画 + owner track）/ `docs/handoff/m4-*.md`（各 WP 引き渡し）/ `docs/platform-support/v1.0-rc-support-matrix.md`（全プラットフォーム確認）/ `docs/license-audit.md` §3.1（sign-off template）/ CLAUDE.md「M4（v1.0-rc）🚧 CC 実装 terminal 到達（2026-07-15）」節。
