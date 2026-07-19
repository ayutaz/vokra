# Vokra M4-residual 監査 統合レポート

- 監査対象: M0〜M4 (v1.0-rc) 残タスク全域 @ HEAD `6186135` / branch `feat/m4-plan-and-wave1` / 2026-07-19
- 入力: 4 ドメイン sweep（M4 tickets+owner checklist / models+parity+実weight 残渣 / GPU backends+kernels / server+CI+docs-currency）+ 敵対的検証 63 verdicts
- 採用規律: CC 行は **CONFIRMED-CC のみ**。REJECTED は訂正先バケットへ、ALREADY-DONE はドキュメント修正リスト化。ランキングは P1 優先 → effort 昇順。

---

## (a) エグゼクティブサマリー

- 重複統合後の総タスク数は **88 行**: **CC 対応可能 42**（P1 7 / P2 15 / P3 20）、**owner 専任 19**（バンドル行含む）、**already-done（ドキュメント修正のみ）14**、**deliberately-deferred 13**。敵対的検証 63 件の内訳は CONFIRMED-CC 62 / REJECTED 1（word-timestamps の perf 半分は M5-14 `3d71d0d` で解消済 → already-done へ帰着）。
- 最大の発見は **owner 前提の陳腐化**: (i) glslangValidator 16.4.0 が本 M1 に導入済で監査中に 12/12 の .comp コンパイルに成功 → Vulkan .spv 生成（M4-13-T16）は CC 化（spec L47 自身が「CC/owner どちらでも可」と明記）、(ii) 実 weight campaign（07-16/17）により「CC 機体に weight が無い」系の checklist 前提（Moshi 14.3GiB / Whisper 3 サイズ / DFN3 / Mimi・DAC・WavTokenizer / Voxtral adapter.json）が全て失効。
- 即効の P1 CC は 7 件: bash 1 行修正（godot compliance scanner — release.yml も同時に塞ぐ）/ CHANGELOG 15 commit 分（DoS 修正の Security 記載含む）/ abi-changelog GGUF 行（M5-13 で未文書 schema を凍結するリスク）/ Vulkan 12 .spv / Voxtral encoder 32 層 + frame-stacking adapter（実 Voxtral ASR e2e の最後のブロッカー）/ Moshi streaming converter + mmap。
- owner critical path: **M4 PR 作成 + merge が全 workflow_dispatch の前提**（workflow は default branch 必須）→ CSM HF gated 受諾（約 10 分で 1 WP 分の実 weight 検証を解錠）→ license sign-off queue（fail-closed 稼働中）→ ADR 4 本（M4-09 G2P / M4-10 MLIR / M4-01 WebGPU / M4-11 gap-flow）+ CosyVoice2 codec-path 決定。

---

## (b) CC 対応可能 backlog（42 行、P1 → effort 昇順）

### P1（7 件）

| ID | タイトル | P | 工数 | 依存 / 根拠（最良証跡） |
|---|---|---|---|---|
| cc-01 | T18 Godot compliance scanner の bash bad-substitution 修正（`${#libs[@]:-0}` → `${#libs[@]}`） | P1 | S | run 29239384239 が line 145 で実 fail（5 crossbuild target は全緑）。release.yml:732 も同スクリプト。bash>=4.4 でのみ再現（ローカル /bin/bash 3.2 は非再現 — 検証は brew bash 5 or CI で） |
| cc-02 | CHANGELOG.md へ `a3a53ff..6186135` の 15 commit 反映（P1 修正 wave / campaign-2 / M5-14。`b06e0d6` の unbounded-alloc DoS は Security 節） | P1 | S | `git log -- CHANGELOG.md` が当該範囲で空。1.0.0-rc.1 に巻き込まれる公開文書。ctx576/DFN3/packed/word-timestamp の grep = 0 hits |
| cc-03 | docs/abi-changelog.md GGUF metadata 行の追補（voxtral head_dim・n_head_kv / cosyvoice2 arch.* / denoise 7→20 keys + 実 115-tensor / Mimi neural-chain 出力 / silero sr8k・sr16k） | P1 | S | 記録政策 :53-58（「every such change lands with an entry」）違反状態。M5-13 凍結前の schema 文書化リスク。全 emit 箇所コード確認済 |
| cc-04 | Vulkan 12 .spv compile + commit + SHA-256 pin + include_bytes! swap（M4-13-T16） | P1 | M | owner 前提「glslc 不在」失効: glslang 16.4.0 導入済・監査中 12/12 コンパイル成功（scratchpad 証跡）。spec L47「CC/owner どちらでも可」、ADR M3-02 §4(a) が brew glslang を裁可。blob-gated テスト群（kernel_dispatch 14 / parity 10+2）点灯、**o-01 Android soak の前提**。CI drift-gate の compiler provenance pin を同 PR で。shader が compiler-blind 執筆のため構文修正で L 寄りに伸びる可能性 |
| cc-05 | Voxtral projector に x4 frame-stacking AdapterKind 追加（[1500,1280]→[375,5120] concat → MLP） | P1 | M | adapter.json in_dim 5120 = 4×1280 で必須と証明。既存 4 種（DownsampleLinear = avg-pool）では表現不能。adapter.json + 実 tensor bind は campaign-1 検証済 |
| cc-06 | Moshi full-7B 有効化: parity-moshi-real truncated leg（先行・非先取り）+ streaming converter + vokra-mmap 配線 | P1 | L | engine.rs:260 `fs::read` + converter Vec 全材料化を実測（~97GiB / ~60GiB、16GB 級で全段 BLOCKED）。owner memo (e)-4 が option (ii) を CC に**事前裁可**（「両方でも可」）。vokra-mmap は既に vokra-models の依存。truncated GGUF（4.4GB、11/11 bit-exact）で回帰保証。**o-16 の moshi dispatch はこれが前提（現状 dispatch は確実に赤）** |
| cc-07 | Voxtral audio encoder の 32 層 pre-norm transformer stack 実装（現状 conv-stem-only stub） | P1 | L | stub 自身の deferral 条件（real checkpoint parity dump）が充足可能に: weights 2 shards + merged f16 + config ローカル、venv で `VoxtralForConditionalGeneration` import 成功（transformers 4.57.6 / torch 2.8.0）。config = Whisper-large-v3 級（32L/1280/20H/128mel）で M2-06 native 実装を再利用。実 Voxtral ASR e2e（M3-10 DoD）最後のブロッカー |

### P2（15 件）

| ID | タイトル | P | 工数 | 依存 / 根拠 |
|---|---|---|---|---|
| cc-08 | Whisper small/medium/turbo の real-audio (jfk) parity fixture 再生成 + small/medium alignment_heads **値**検証 | P2 | S | 現 fixtures は real-weight × synthetic 1-s（manifest: seed=1234 / pcm_len=16000、`9d3eaae` のみ）。checkpoints + venv + dumper + jfk-30s.wav 全てローカル。ハードコード表（whisper.rs:637-646）は独立クロスチェック未実施。turbo atol calibrate 不要の実測材料あり。**red-line 維持: commit は owner レビュー後手動、T10 dispatch は owner** |
| cc-09 | vllm_compat loopback flake の根治（+ 必要ならチケット化） | P2 | S–M | pristine HEAD で 2/3 再現（ConnectionReset os 54 @ vllm_compat.rs:384）。raw-TCP helper（retry 無し・Connection: close）が 4 テストファイルにコピペ → 共有修正。並列 suite 実行依存（単発 binary 6/6 green）。CI server-compat job の赤ノイズ源 |
| cc-10 | CLI に voxtral arch route 追加 | P2 | S | run.rs:298-306 が follow-up と明記、engine.rs ARCH 定数に voxtral 無し。**cc-05/cc-07 後に land、または stub 中は transcribe を hard-gate（FR-EX-08 — stub 出力の露出禁止）**。9.4GB GGUF の 16GB 機ロード smoke 含む |
| cc-11 | vokra-server README の flag 表記修正（`--asr-base` 等 → 実装済 `--whisper-base` 系 + `--piper-g2p` 追記） | P2 | S | fc8feec 後未更新。quickstart 通りに実行すると unknown-flag error になる公開文書 |
| cc-12 | README / README.ja の perf・実証 claims 更新（CPU が ORT 超え: base 2.5x / turbo 2.7x / Silero 2.3x + 実 weight 検証: Whisper 4 サイズ byte 一致・codec parity・DFN3 SI-SNR gap 2.0e-7 dB） | P2 | S | committed bench reports（m5-14-final / m1-real-weight-eval）引用のみ。**M1 rig-scoped vs onnxruntime 1.19.2 CPU の限定表記必須** |
| cc-13 | platform-support matrix のテスト数更新（2418 → 2551 chain、provenance 2418→2479→2532→2551） | P2 | S | 書く前に fresh debug `--workspace` 実測で pin（2551 は release + doctests） |
| cc-14 | Kokoro voicepack の voice-name lookup（M2-07-T02 TBD 解消） | P2 | M | mod.rs:283-292 NotImplemented。voicepack tensor + voice_names metadata は GGUF に有り。schema 注意: upstream ref_s は token 数 index（converter schema 拡張の可能性、再変換は数分） |
| cc-15 | CosyVoice2 T06: text tokenizer vocab の GGUF 埋め込み + Rust encode | P2 | M | codec 決定（o-11）と独立。vocab.json + merges.txt ローカル、whisper U8-embed が前例。encode 側 byte-level BPE が M の半分（in-repo tokenizer は decode-only） |
| cc-16 | Moshi duplex へ real Mimi GGUF side-car bind → PCM quality gate un-refuse（+ limiter/clamp） | P2 | M | blocker（converter mapping）は `ebe1cc5` で解消済（codes 4384/4384 = 100%、decode Δ 3.67e-6）。engine.rs:232-240 は依然 synthesized 構築。gate test 自身が置換手順を記述。duplex peak 1.412 の実測化も兼ねる |
| cc-17 | Voxtral parity fixtures + dumper + fixture-leg CI（M3-10-T19/T20、CC 担当列） | P2 | M | parity_voxtral.rs:6-8 が不足 artifact（dump_voxtral_reference.py）を自己指定。tests/parity/ に voxtral 無し = 出荷 ASR で唯一 fixture ゼロ。**cc-05 + cc-07 後**。bf16 RSS 規律（fp32 ~18.7GB 不可）、Moshi 式 truncated dump が fallback |
| cc-18 | server hardening bundle: spawn_blocking（openai transcriptions + piper TTS）+ GET /v1/models + TTS language-id 到達性 | P2 | M | openai.rs は spawn_blocking ゼロ（wyoming.rs:843 が copy pattern）、/v1/models は doc comment のみ、SynthesisRequest.language は server から設定不能。/v1/models は vllm 側 route-closure test（no_extra_v1_routes）を**意図的に**調整 |
| cc-19 | Word-timestamps ユーザー表面: verbose_json + timestamp_granularities=word（501 解除）+ beam HTTP 露出 + CLI flag | P2 | M | 501 の自己 deferral 前提（core 未着）が `da13bfd`（Δ 5-50ms、sanity 6/6）+ `eb41648`（alignment_heads EXACT）で充足。openai.rs:417-419 の T07/T08 followup コメントが owning anchor。**注記 1 の検証 split あり — 着地時に M4-11-T13 gap-flow への owner nod を 1 行記録推奨** |
| cc-20 | m4 / m3 owner-verification-checklist の全面 refresh（6 節 + m3 Voxtral (a)） | P2 | M | **(c) 節が入力リスト**。stale な owner 前提を全て証跡付きで書換え。owner queue は縮小のみ（削除禁止）: CI dispatch / license / CSM gate / 非 Mac 実機 / ADR は残す。honest 残置（full-7B RAM block 等）必須 |
| cc-21 | piper-plus loader の可変 decoder upsample geometry（mera-multilingual dec.ups.2 [64,32,8] 対応） | P2 | M | **唯一の Apache-2.0 公開 voice** がロード不能（P2 #10）。decoder.rs:9-11 固定 geometry。config.json 駆動で数値発明不要。css10 near-bit-exact（mel-L1 0.0033-0.0035）を regression guard に |
| cc-22 | WASM デモの macOS Chrome pre-flight（owner T11 matrix の pre-material） | P2 | M | web/dist ビルド済 + serve.mjs + whisper-base.gguf + chrome-devtools MCP 全てローカル。Part 10 は現在空 = 実ブラウザ証跡ゼロ。**pre-flight 記録であり owner sign-off ではない**（M4-01-T28 / M4-11-T11 正式判定は owner）。Safari leg は観察のみ |

### P3（20 件）

| ID | タイトル | P | 工数 | 依存 / 根拠 |
|---|---|---|---|---|
| cc-23 | docs/deliverables.md §3.3 へ M4-19 安定性ティア注記を verbatim 適用 | P3 | S | handoff m4-19:75-87 に全文。blocker は worktree の locational のみ — 本 checkout に file 有り、§3.3 anchor（L81）確認済 |
| cc-24 | CLI kokoro phoneme-id route（piper raw path のミラー） | P3 | S | engine.rs に kokoro ARCH 無し（convert 側のみ）。G2P 不要。cc-14 と対で voice-name UX 完成 |
| cc-25 | Silero VAD 8 kHz (ctx288) real-speech 評価（最後の未検証象限） | P3 | S | {16k,8k}×{synthetic,real} の残り 1 象限。bothrate GGUF + ORT venv + 実クリップ全部ローカル。ctx576 修正（`7639dc0`）は 16 kHz のみ検証済 |
| cc-26 | whisper-medium Metal 転写 parity + **turbo Metal 補完** | P3 | S | 監査で turbo Metal も未検証と判明（driver.log が途中終端 — CLAUDE.md「small/turbo Metal byte 一致」は過大 → ad-14）。medium.gguf ローカル、CLI --backend は c68038f で稼働 |
| cc-27 | Metal graph-arm Mul（+ 任意で Copy）MSL kernel — CUDA/Vulkan/WebGPU と対称化 | P3 | S | d37c3d7 の「捏造拒否」= honesty stance であり spec deferral ではない（spec line 不存在を確認）。19 MSL kernel が template。+ doc 修正（Copy 過大記載） |
| cc-28 | stale eval-cache GGUF 再生成（cosyvoice2 pre-fix / voxtral f16 pair の破棄マーク） | P3 | S | cosyvoice2 GGUF は bias 修正 `7336079` 前の産物で post-fix loader が `llm=None` で**黙って**通る。罠: top-level config.json は 2-byte stub、実 config は CosyVoice-BlankEN/config.json |
| cc-29 | `--max-concurrent-sessions` CLI knob（hardcoded 4 の解消） | P3 | S | server.rs:41-45 自己申告 follow-up。registry と scheduler の両 `::minimum()` に単一値（invariant pin 済）。overload error path 実装済 |
| cc-30 | per-model backend override（T03 follow-up） | P3 | S | service.rs:139-141 自己申告。要 Cargo feature passthrough（server は現状 metal 非ビルド）。Metal==CPU byte 一致実証（c68038f — sweep の c68accf は typo）が動機 |
| cc-31 | license-audit DFN3 行の precision 更新（dual MIT/Apache-2.0 + 実 parity 済状態） | P3 | S | campaign-2 一次証跡（sha256 49c52edc）。**T18 sign-off checkbox は owner — pre-fill 禁止** |
| cc-32 | cron 衝突解消（Mon 06:00 ×5 + 05:00 ×2 の stagger） | P3 | S | parity-rvq-real / parity-utmos / gpu-vulkan-parity の in-file コメントが掲げる offset 規律に自ら違反 |
| cc-33 | SBOM 再現性検証（fresh checkout + CI artifact 比較。Docker leg は daemon 起動が条件） | P3 | S | SBOM は未 commit → 「repo copy と cmp」は不可能（訂正済）: fresh-checkout regen vs main regen、or CI artifact `vokra-cpu-vulkan-only-linux` と比較。**正式 close（M4-15-T10）は owner ratify**、diff は checklist:291 により CC へ還流 |
| cc-34 | m4-19 real-GGUF Wyoming round-trip の記録化 | P3 | S | 監査 + 検証で 2 回 green（38s、SKIP 無し）。**-neutralspk GGUF 必須を pin**（6lang 多話者は spk_proj.0.weight missing で hard error — 注記 3）。multilang WER leg を bundle すると M |
| cc-35 | campaign P2/P3 残のチケット転記 | P3 | S | 本報告の backlog 表が実質カバー — docs/tickets/ への転記 + convert --config delegate/standalone 非対称の追補のみ |
| cc-36 | campaign P3 cleanup bundle: Mimi GGUF 906→519MB dedup / CLI FR-EX-08 gates（--backend silent-CPU on VAD/TTS/S2S、convert --config silent drop）/ duplex limiter・SPDX banner・--backend selector | P3 | M | run.rs:86-90 の documented silent-CPU は crate 自身の FR-EX-08 姿勢と矛盾。mimi 二重 emit をコード確認済。「見送り」は leg scope-guard であり spec red-line ではない。limiter は cc-16 と同時 land が自然 |
| cc-37 | PagedKvCache の release_stream API（zero-fill O(pages) → O(1)） | P3 | M | session.rs:418-426 自己申告 follow-up（M3-15 T04 discussion）。C ABI 無関係（Rust surface のみ）。multi_session の state-leak テストが regression oracle |
| cc-38 | POST /v1/audio/speech (TTS) endpoint | P3 | M | README 自身の「v1.0+」期限が到来（旧 v1.0 = M3 済）。FR 義務なし = README-declared surface の完成。TTS service + real G2P（`581758a`）+ voice GGUF 全て有り |
| cc-39 | server の whisper small/medium/turbo slots | P3 | M | M4-14 で model 側完了 + 4 サイズ byte 一致検証済 + GGUF ローカル。large-v3 pattern 踏襲、silent substitution 禁止維持 |
| cc-40 | server real-GGUF slot 検証（voxtral / silero / kokoro-advertise。large-v3 leg は任意 ~3GB ungated DL） | P3 | M | campaign-2 §(f) が honest に未実施と記録。slots は fc8feec で配線済。kokoro は advertise + 明示 501 の検証のみ（合成は dd-06 のまま） |
| cc-41 | Godot T19 headless 検証 leg（**load(path) trampoline 実装込み**） | P3 | M | ba33bd0 で 5 trampolines real dispatch 済 = handover §3.b stale。ただし監査で registry.rs:197（inner session 常に None）+ load 未登録を発見 — demo は load_model を呼ぶため実装無しでは到達不能。Godot は無償 DL。**editor GUI 確認 + T20 close PR は owner** |
| cc-42 | X-06 nightly LibriSpeech ASR-WER leg（advisory、CC 担当分） | P3 | L | milestones §10 X-06 の CC 半分が M1 以来 open。campaign が同 pipeline をローカル実証済（dev-clean + jiwer + normalizer、資産 ~/.cache に有り）。required-check にしない。UTMOS leg は M5-15（dd-08）、Tier-2 実機は owner |

---

## (c) already-done — ドキュメント修正リスト（14 件）

タスク自体は完了済で**記載だけが stale**。cc-20（checklist refresh）の入力リストを兼ねる。

1. **ad-01** M4-20-T17 DFN3 実 checkpoint parity: `9b718d1` で完了（SI-SNR 14.768399 vs 14.768398 = gap 2.0e-7 dB、115 tensors、21 stage taps、波形 max Δ 4.17e-7）→ checklist §1.4 (:43-48) + M4-20 spec :57 を完了化。残 owner は T18 license sign-off のみ。
2. **ad-02** Moshi weight sourcing + truncated parity: 14.32GiB ローカル + 11/11 frame bit-exact → checklist §1.2(b)「CC 機体に無い」を書換え。honest 残置: full-7B convert の RAM block（→ cc-06）+ torch reference dump（→ o-06/o-14）。
3. **ad-03** Mimi / DAC / WavTokenizer の real parity ローカル first-fire: 106/106、WavTokenizer Δ 0.0 bit-identical、Mimi PCM roundtrip codes 100%（`ebe1cc5`）→ checklist §1.6 :60「CC 機体では未実行」/ §1.6(ii)「owner sourcing で発火」/ §1.7 :66「pretrained 未 DL」全て失効。残 = parity-rvq-real dispatch（owner）+ license。
4. **ad-04** Whisper 3 サイズの変換 + fixture commit（`9d3eaae`/`babd7e8`、gated 8/8 pass、転写 4 サイズ byte 一致）→ checklist §1.3(a)「HF 取得 + 変換 + fixture 生成」は完了。**ただし fixtures は synthetic 1-s のままで real-audio 再生成 delta は cc-08 に残存**。
5. **ad-05** UTMOS un-defer 判断: 2026-07-18 依頼者承認 → M5-15（T14-T22）→ checklist §1.5/§3.5/:318 の「NO-GO-defer」「owner 確認待ち」表記を M5-15 参照へ。
6. **ad-06** server CLI model wiring（M2-09-T04 carry-over）: `fc8feec`（Config + InferenceService::build + missing-GGUF hard error）+ `581758a`（--piper-g2p + /api/tts）で完了、real GGUF e2e 検証済 → handoff m4-19 item 4 + checklist :289 を strike（README flag 修正は cc-11）。
7. **ad-07** worktree 産 ADR の main sync: M4-15/M4-18×2/M4-14 の 4 ファイルが docs/adr/ に存在 → checklist :293 close（ファイル内の stale 注記掃除のみ）。
8. **ad-08** Voxtral adapter.json side-car（m3-checklist :34 (a)「依頼者」）: `12e574e` で 762/762 tensor byte-identical 変換 + 実 GGUF ロード + adapter bind 済 → 完了化。残 = WER 計測（cc-17 系で CC 化）+ license（owner）。
9. **ad-09** Kokoro prosody f0 fixup follow-up は moot: `92dbc92` で PROSODY_F0_ATOL=0.05 は flawed reference の artifact と判明・撤去、真 upstream 比 f0 = 3.01e-3 @ default atol 0.01 → CLAUDE.md M2 節残 list の当該行 retire（SSOT sweep は owner 承認時）。
10. **ad-10** Vulkan graph-arm「~28% 残 T27-T29」記載は stale: coverage は CUDA arm と同等（machine-checked、graph_arm_coverage.rs）、残は blob gate のみ（→ cc-04）→ CLAUDE.md M3 節を更新。
11. **ad-11** word-timestamps beam overhead +9.7〜93.1% 記載は `3d71d0d`（beam incremental KV、beam-1 +3.9% bit-identical）で supersede → campaign report errata。**REJECTED claim の perf 半分の帰着先**。
12. **ad-12** musl static build（M2-09-T19）は CI 済（ci.yml:638-716、静的リンク assert 付き）→ campaign-2 §(f)「unverified」表記に注記。残 = musl バイナリ上の real-GGUF smoke（Linux host = owner or CI 拡張）。
13. **ad-13** mel_frontend_baseline.json は**意図的 non-regen**（M1 値で上書きすると ubuntu gate が falsely redden — m5-14-final report :74 に決定記録済）→ アクション不要、上書き禁止ガードとして記録。
14. **ad-14** 過大記載 2 点の訂正: CLAUDE.md「whisper-small/turbo Metal == CPU byte 一致」は turbo 未検証（→ cc-26 で補完）/ `d37c3d7` message + CLAUDE.md「Metal graph-executor Add+Softmax+Copy 配線」は Copy 未配線（→ cc-27）。

---

## (d) owner 専任リスト（19 行）

### 実機・インフラ（6）
- **o-01 [P1]** M4-13-T17 Android Vulkan soak（Adreno 750 + Mali G720、Whisper base RTF < 0.7 = WP exit hard gate）— **cc-04 land 後に即実行可能**。lavapipe では driver 級バグを捕捉不能。
- **o-02 [P2]** M4-05-T30 / M4-06-T30 S2S 実マイク/スピーカー検収（実音響エコー AEC path。CSM 側は o-07 依存）。
- **o-03 [P2]** M4-07-T17/T18 H100 FA v3（vast.ai 有償、SM 9.0。kernel は OWNER-VERIFY hotspot、baseline JSON の TBD 埋め + dashboard 登録）。
- **o-04 [P2]** M2-14 self-hosted CUDA runner standup + RTF<0.10 formal gate + required-check promotion（M4-17 cloud VM と bundle 可）。
- **o-05 [P2]** carry-over 実機 bundle: iOS 実機 RTF / M3-18 Android+Godot / M3-15 real-network 75ms（NFR-PF-05）/ M4-17 CPU ISA cloud VM + ARM64 tier perf（**M5-14 後 baseline で計測**）/ M4-08 RISC-V boards（cpuinfo dump は T05 parser fixup として CC へ還流可）/ M5Stack soak（optional）/ ブラウザ multi-OS 正式 matrix（M4-01-T28 / M4-11-T11 — cc-22 が pre-material）。
- **o-06 [P3]** Moshi full-7B torch reference dump 用 large-RAM 環境（fp32 ~43GiB。cc-06 とは独立の残渣）。

### weight・gated アクセス（2）
- **o-07 [P1]** CSM: sesame/csm-1b + meta-llama/Llama-3.2-1B の HF gated 受諾 + fresh token（file resolve 401 の一次証跡 csm-probe.txt。**唯一の blocker、~10 分で WP 全体の実 weight 検証を解錠** — CC 側 harness/binding は完成済）。
- **o-08 [P3]** EnCodec（CC-BY-NC）weight の research 用取得可否判断（parity leg 未実行の唯一理由。fail-closed 姿勢の維持判断込み）。

### 判断・法務（6）
- **o-09 [P1]** license sign-off queue（docs/license-audit.md §3.1、fail-closed 稼働中）: Whisper 3 サイズ / CSM・Moshi attribution / DAC+Mimi zoo / WavTokenizer + X-Codec 2（NC live 一次確認済）/ DFN3 T18（dual MIT/Apache-2.0）/ CosyVoice2・Voxtral M3 carry-over / piper css10 'other' + private v7 canonical repo。事実材料は全て CC 収集済。
- **o-10 [P1]** ADR 判断 4 本: M4-09 G2P 3 択（dd-06 kokoro 合成の gate）/ M4-10 MLIR 採否 / M4-01-T02 WebGPU shim の Accepted 化 / M4-02 Accepted 化。遅延着地で API/配布が変わる場合は M4-12 baseline re-snapshot（checklist:330）。
- **o-11 [P1]** CosyVoice2 codec-path 決定（SSOT の Mimi bridge vs upstream 実配布 = FSQ speech_tokenizer_v2 + flow.pt + HiFT、Mimi 非含有）。flow.pt/hift.pt は匿名 DL 可 — **決定当日から CC 実装可（L 規模）**。
- **o-12 [P1/P2]** M4-11 T11/T13 matrix 承認 + gap-flow 3 択（Android AAR / desktop release job = NFR-MT-08 gap / Tier-2 nightly）。決定後の CD job 実装は CC。
- **o-13 [P2]** X-Codec 2 T14 ruling（code MIT vs weight cc-by-nc-4.0 の 3 系統齟齬。**ruling まで registry が Permissive = NC gate の fail-open 穴**）。ruling 後の license_class.rs 1 行 flip は CC 待機タスク（checklist:146）。
- **o-14 [P3]** 判断残: M4-15「動かない SKU 禁止」gate（o-01 連鎖）+ M4-15-T10 SBOM 正式 ratify（cc-33 が pre-material）/ parity-moshi-real 再設計の full-7B 戦略（option (i)/(ii)/両方 — cc-06 は非先取り）/ M3-19 Kill switch D（2027-03〜05 calendar-fixed）。

### GitHub 操作（5）
- **o-15 [P1]** **M4 PR 作成 + merge**（branch `feat/m4-plan-and-wave1`、PR 未作成）— workflow_dispatch は default branch 上のファイル必須のため**全 dispatch の前提**。
- **o-16 [P2]** first/re-dispatch bundle: parity-kokoro-real（92dbc92 後の再 dispatch）/ parity-whisper-real 3-size（T10 — turbo atol 判断材料は既取得）/ parity-rvq-real / gpu-vulkan-parity（T18）/ web-wasm + npm dry-run（ローカル pre-verify 全緑済）/ godot-crossbuild（**cc-01 後**）/ parity-moshi-real（**cc-06 後のみ — 現状は確実に赤**）/ parity-csm-real（o-07 後）/ parity-utmos（M5-15 flip 後）/ nightly-webgl（o-17 後）。
- **o-17 [P2]** secrets / provisioning: NPM_TOKEN + npm org（@vokra）/ OPENUPM_TOKEN / PyPI / UNITY_LICENSE（il2cpp + webgl leg の perma-skip 解除）/ CDN 選定 + vokra.dev/.io/.ai ドメイン取得（COOP/COEP は現行 thread-free baseline では不要 — handoff m4-02 §2）。
- **o-18 [P3]** required-check promotion 判断（gpu-vulkan-parity 連続緑後 / gpu-cuda-rtf は runner standup 後）。
- **o-19 [P3]** Godot M3-11 T20 WP-close PR + editor 最終確認（cc-41 の headless evidence を参照）。

---

## (e) deliberately-deferred（13 件、spec 引用付き — 格上げしない）

1. **dd-01** codec op GPU kernels 全 backend（mimi/dac/encodec RVQ + wavtokenizer/xcodec2 FSQ）: M4-04 spec「GPU 実 kernel（Metal MSL / CUDA NVRTC / Vulkan SPIR-V）はスコープ外 = M3-06 T14/T15 と同じ deferred posture」/ M4-16「seam-awareness + 明示 UnsupportedOp まで、実 dispatch は follow-up ticket」。M5-14 で Mimi encode/decode CPU target MET → **perf trigger 非発火を再確認済**。owner の WP-scoping 決定なしに昇格禁止。
2. **dd-02** S2S kernel fusion / per-step 1-command-buffer full residency + fused GQA driver: checklist §6.2「別 WP（M4 kernel-fusion 系 follow-up or M5）」、campaign「honest 境界: driver は deferred（partial）」。**FA v3 は M4-07 外使用禁止（red-line）**。primitive 4 kernel（rms_norm/rope/silu/swiglu）は `d37c3d7` で land 済。
3. **dd-03** AEC / enhancement ops = CPU-by-design: M4-03「Compute seam に追加しない（CPU only）」（L50/L323/L355-(c)）。DFN3 RTF 0.0411（target ≤0.06 MET）→ **profiling trigger 非発火を再確認済**。
4. **dd-04** DAC feature→PCM（SEANet）decoder: M4-04 spec :44「consumer 未定…スコープ外」+ ADR M4-04:97-99 mechanism anchor。GGUF は full tensors pass-through 済で将来 land 時も再変換不要。
5. **dd-05** WavTokenizer / X-Codec 2 converter wiring: ADR M4-16 §D-d「converter-side emission…は実 model 統合 WP」。real-codebook parity は Δ 0.0 bit-identical で先行検証済 = 将来 WP は検証済 numerics から開始。
6. **dd-06** Kokoro text→PCM G2P bridge + server synthesize 501: mod.rs:743-745「needs a G2P bridge (out of scope M2-07)」。M4-09 owner ADR（o-10）が gate — 決定前の昇格禁止。
7. **dd-07** Stft（front-end ops）の GPU graph arm: kernels/README.md:92「putting Stft on a GPU graph arm is a separate M4+ decision」= ADR-level 判断、CC 単独昇格不可。
8. **dd-08** UTMOS 実装一式（wav2vec2_regression.v1 arch bump + 8 modules + LSTM/GroupNorm kernels + license/abi 行）: **M5-15 T14-T22 が正式 owner**（2026-07-18 un-defer 承認済）。M4 残渣ではない。
9. **dd-09** CPU speed vs ORT の残 2/11 target: M5-14 backlog（batched-beam / pack-once-share）が所有 — campaign P2 #9 は supersede 済と読む。
10. **dd-10** Wyoming streaming-synthesis mid-stop: handoff m4-19「follow-up when SynthesizeService gains a streaming form (M4/M5 kernel-fusion class)」— M4-19 DoD 外を明記。
11. **dd-11** Unity WebGL の IndexedDB model cache: handoff m4-02:50-52「follow-up (not implemented — the demo re-fetches per load)」。CDN 決定（o-17）依存。
12. **dd-12** C ABI freeze bundle（STABILITY 書換え / Non-C-ABI 節 / EXPERIMENTAL markers / watermark patch 条項 / gate promotion）: v-label 再割当 #2 により **M5-13（v1.0 GA タグ）**。監査で `gen-c-abi.sh --check` OK + rc baseline（33 fn + 11 typedef）不変 = campaign は C ABI 無追加を確認済。
13. **dd-13** DFN3 の THIRD_PARTY_LICENSES ファイル複製: NOTICE:163-167 が「real checkpoint が model zoo に追加される時（owner T18 sign-off）」を条件化 — 先行 land は T18 の front-run になるため待機。

---

## (f) 注記

1. **REJECTED 1 件の帰着（word-ts-greedy-path-and-surface）**: perf 半分 = ad-11（M5-14 `3d71d0d` beam incremental KV、beam-1 +3.9% = claim 自身の受容帯域内）。表面半分は verifier 間で split — 1 件 REJECTED（「M4-11-T13 gap-flow の owner scope nod 要」）vs 2 件 CONFIRMED（openai.rs:417-419 の T07/T08 followup コメント + 501 の自己 deferral 前提充足 = owning anchor 有り、ADR M4-20 の制限は C ABI 面のみ）。統合判断: cc-19 として CC backlog に採用しつつ、**着地時に gap-flow への owner nod を 1 行記録することを推奨**（安価な保険、非ブロッカー）。
2. **検証過程で判明した sweep 側の訂正（本報告に反映済）**: c68accf → `c68038f` / vllm flake は並列 suite 実行依存（単発 binary 6/6 green、再現率 1/3〜2/3）/ cc-01 は bash 3.2 で非再現（bash>=4.4 挙動）/ m4-19 gated test は `-neutralspk` GGUF 必須 / voxtral adapter.json に active key は無い / CHANGELOG 欠落は 15 commits（14 でなく）・grep hits は 5（0 でなく、ただし全て既存節）/ SBOM は未 commit（「repo copy と cmp」不成立 → 比較先を訂正）。
3. **未検証 side-finding（新規、要チケット）**: 多話者 `piper-plus-css10-ja-6lang.gguf` が `spk_proj.0.weight` missing で load hard-error（FR-EX-08 として正しい挙動だが、multi-speaker loader 対応は cc-21 と同じ piper loader wave への相乗りを推奨）。
4. **sequencing 拘束（要遵守）**: o-16 の parity-moshi-real dispatch は **cc-06 後のみ** / godot-crossbuild dispatch は **cc-01 後** / o-01 Android soak は **cc-04 後** / parity-csm-real は **o-07 後** / cc-10・cc-17 は **cc-05 + cc-07 後**（stub 露出は FR-EX-08 違反）/ cc-40 の kokoro は advertise + 明示 501 の検証まで（合成は dd-06）/ 全 workflow_dispatch は **o-15（M4 PR merge）が前提**。
5. **監査 ground rules の堅持**: PR 作成・merge / workflow_dispatch / ADR Accept / license 判断 / parity fixture の auto-commit は owner 専権。CC 行はいずれもこの線を越えない（cc-08 は「生成 + 検証 + diff staging」まで、commit は owner）。
6. **スコープ境界**: 本監査は HEAD `6186135`（M5-14 Wave 0-3 込み）時点の M0〜M4 残渣が対象。M5-14/15 スコープ（CPU perf 残 / quant 実測 / UTMOS 実装）は dd-08/dd-09 に境界のみ記録し、本 backlog へは混入させていない。