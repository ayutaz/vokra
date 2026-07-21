# license-audit.md — Vokra 依存ライセンス総覧

**最終更新**: 2026-07-04（M2-13: research flag enforcement 機構の実装を §3 末尾に追記 / M1-02: GGUF K-quant + safetensors runtime direct-load のフォーマット参照を追記 / M0-08: §3 CAM++ 行を Vokra が実際に変換対応した具体ソース `ayousanz/campplus-onnx`（上流 `iic/speech_campplus`）に更新 / 2026-07-06: §2・§6 の backend binding crate（cudarc/metal-rs）を「実装は不採用＝手書き生 FFI」に、§2/§5 G2P を実装済 `integrations/vokra-piper-g2p` に、§6 CUDA を no-silent-fallback（explicit `BackendUnavailable`）に現物実装と整合 / 2026-07-15: §4 音声処理ライブラリ表の Vokra 採用時期ラベルを §3 + M4 実装状況と整合（denoise=v1.0-rc/M4-20、wfst=v1.0 GA/M5、C2PA embedding は deferred）、§3 Bark 行を post-v1.0 GA に更新、§3 Matcha-TTS 行の配布ラベルを「★ post-v1.0 GA」に更新（Bark と同 tier の齟齬解消、FACT SHEET MODELS 整合）、§2 wgpu 行を「実装は不採用」（M4-01 raw WebGPU extern-import shim、NFR-DS-02 zero-dep）に更新し stale label `v0.1-v1.5` を除去）
**目的**: Vokra が依存するすべての Rust crate、モデル weight、音声 codec、vocoder、辞書、G2P、audio 前処理ライブラリのライセンスを列挙し、Apache 2.0 core との互換性、Unity/Godot Asset Store 配布可否、商用ゲーム組込可否を明示する。

**運用**:
- 新規依存追加は PR に本ドキュメント更新を必須
- CC-BY-NC / CC-BY-NC-SA / 学習権利不明の weight は **公式 model zoo から除外**、research flag で分離
- GPL/LGPL 依存が unavoidable な場合は Wyoming Protocol 経由の分離プロセスモデルで実装（Vokra binary に static link しない）

---

## 1. Vokra Core ライセンス方針

- **Vokra Core**: **Apache 2.0** (特許 grant 条項が Unity/Godot エコシステムで推奨されるため MIT より優先)
- **Vokra 生成コード**: LICENSE ファイル、NOTICE ファイル、SPDX-License-Identifier ヘッダを全ソースファイル冒頭に付与
- **NOTICE 記載必須項目**:
  - Meta AudioSeal (MIT) — 論文 attribution
  - piper-plus (MIT) — 依頼者作 TTS 統合
  - Mimi codec (CC-BY 4.0) — Kyutai attribution
  - BigVGAN 論文 attribution (実装はスクラッチ、NVIDIA reference は使用しない)
  - Snake activation 論文 attribution (Ziyin 2020)
  - FlashAttention v2 論文 attribution (Dao 2023)

---

## 2. Rust Crate 依存表

| Crate | ライセンス | 用途 | Vokra 互換 | 備考 |
|-------|----------|-----|------------|-----|
| **std** | Apache 2.0 / MIT | Rust 標準 | ○ | |
| **rayon** | Apache 2.0 / MIT | work-stealing thread pool | ○ | libomp/OpenMP は使わない |
| **wgpu** | Apache 2.0 / MIT | WebGPU/Vulkan/Metal/DX12 backend | ○ | **実装は不採用** — WebGPU backend は wgpu を使わず、raw WebGPU extern-import shim を手書き実装（M4-01、NFR-DS-02 zero-dep、cudarc/metal-rs と同じ生 FFI 方針） |
| **cudarc** | Apache 2.0 / MIT | CUDA driver API bindings | ○ | **実装は不採用** — CUDA backend は binding crate（cudarc/cust/rustacuda）を使わず、Driver API + NVRTC を手書き生 FFI で `dlopen`（`libcuda`/`libnvrtc`）実装（NFR-DS-02 zero-dep、`third_party/NVIDIA-EULA.md` Binding-crate note）。cudart/cudnn/cublas は bundle せず system install 検出のみ |
| **metal-rs** | MIT | Metal API bindings | ○ | **実装は不採用** — Metal backend は binding crate（metal-rs/objc2/objc/core-foundation）を使わず、Obj-C/Metal を手書き生 FFI（`#[link(kind="framework")]`）実装（NFR-DS-02 zero-dep） |
| **ash** | Apache 2.0 / MIT | Vulkan bindings | ○ | Android/Linux backend |
| **safetensors** | Apache 2.0 | tensor loader (HF) | ○ | pickle 回避 |
| **realfft** | MIT / Apache 2.0 | Rust FFT (pocketfft port) | ○ | FFTW3 (GPL) 排除 |
| **rustfft** | MIT / Apache 2.0 | Rust FFT | ○ | realfft の依存として |
| **hound** | Apache 2.0 | WAV I/O | ○ | |
| **symphonia** | **MPL-2.0** | audio decode (mp3/aac/opus 等) | **要注意** | 修正部分のみ MPL 保持、Vokra 修正なしで使う場合 Apache 2.0 core と両立可能。Unity Asset Store 配布時に NOTICE で明示 |
| **sleef-rs** | Apache 2.0 or BSL-1.0 (要確認) | SIMD 数学関数 | **要確認** | 代替: `pulp` (Apache 2.0/MIT) or `std::simd` |
| **pulp** | Apache 2.0 / MIT | portable SIMD | ○ | sleef-rs 代替候補 |
| **simde** (C, WASM 経由) | MIT | SIMD emulation for WASM | ○ | AVX 等を WASM SIMD にエミュ |
| **c2pa-rs** | Apache 2.0 | C2PA manifest embedder (Adobe) | ○ | EU AI Act 対応 |
| **serde** / **serde_json** | Apache 2.0 / MIT | シリアライズ | ○ | |
| **tokio** (server 用途) | MIT | async runtime | ○ | サーバサイド API 用 |
| **axum** or **actix-web** (server 用途) | MIT / Apache 2.0 | HTTP server | ○ | vLLM 互換 API 用 |
| **candle-core** / **candle-transformers** (参考用) | Apache 2.0 / MIT | Rust ML | ○ | Whisper reference 実装として参照可、Vokra は自前実装だが kernel の参考にする |
| **cbindgen** | MPL-2.0 | C ABI ヘッダ生成 | ○ (build-only) | ビルド時のみ、成果物には含まれない |
| **SBOM generator（first-party、`scripts/sbom/generate_spdx.py`）** | Apache 2.0（Vokra 本体） | SBOM (SPDX 2.3) 生成 | ○ (build-only) | M4-15。第三者 SBOM crate（cargo-sbom / cargo-cyclonedx 等）は不採用 — `cargo tree` + python3 標準ライブラリのみで生成し root Cargo.lock 不変（NFR-DS-02、ADR M4-15 §(b)）。成果物に入るのは生成された SPDX JSON のみ |

**未確認 / 要検討**:
- **G2P（実装済、M0）**: 実 8 言語 G2P は依頼者作 MIT crate `piper-plus-g2p`（piper-plus repo 内）を **out-of-workspace の opt-in 統合 crate `integrations/vokra-piper-g2p`**（root workspace 非 member・独自 `Cargo.lock`・git `rev` pin）から呼び出して提供。`piper-plus-g2p` の非 `vokra-*` 推移依存（jpreprocess/regex/serde 等）は zero-dep runtime（NFR-DS-02）に入らず、`vokra_piper_plus::Phonemizer` trait 境界で注入。text→音声 JA/EN 実動、**eSpeak-NG (GPL-3.0) 不使用**。in-runtime Rust 化・`phonemizer-rs`/`phonikud` は将来検討
- **RVC 系 F0 抽出**: RMVPE (MIT) / FCPE (MIT) / CREPE (MIT) の Rust port が未成熟 → 自前実装 or Python 呼び出し (server 用途のみ)
- **soxr resampler (LGPL)**: **不採用**、代替 `speexdsp resampler` (BSD) 相当を Rust で独自実装

**明示的排除**:
- **FFTW3 (GPL-2.0-or-later)**: 排除、pocketfft (BSD-3) Rust 移植で代替
- **libespeak-ng (GPL-3.0-or-later)**: 排除、piper-plus 独自 G2P で代替
- **libopenmp (LGPL)**: 排除、rayon で代替
- **soxr (LGPL)**: 排除、speexdsp AEC/resampler の Rust port で代替
- **rubberband (GPL)**: 排除

**オンディスク・フォーマット参照（コードコピーではなくデータ仕様の transcribe）**:
- **GGUF / ggml K-quant ブロックレイアウト** (`Q4_K`/`Q5_K`/`Q6_K` の `block_q*_K` 構造・`get_scale_min_k4` パッキング): ggml / llama.cpp (**MIT**) の `k_quants.h` / `dequantize_row_q*_K` から **フォーマット仕様のみ** を参照し、Vokra 独自の scalar・`unsafe`-free 実装を新規記述 (M1-02、`crates/vokra-core/src/gguf/quant/`)。バイトレイアウトはフォーマットそのものであり著作物のコピーではない (whisper.cpp 型 native 再実装、CLAUDE.md 方針)。正当性は外部ファイルに依存しない in-repo の analytic oracle (closed-form super-block + quantize→dequant roundtrip) で pin。外部 crate 依存はゼロ (NFR-DS-02)。
- **safetensors ヘッダ (JSON) フォーマット**: Hugging Face の on-disk 仕様 (**Apache-2.0**) を参照し、reader / JSON パーサとも Vokra 独自記述 (std-only、外部 crate なし)。ONNX/protobuf は runtime に入らない。

### vokra-server の crate-scoped 例外（2026-07-19、依頼者承認）

**対象は `integrations/vokra-server` の除外 workspace のみ**。root `Cargo.lock` は `vokra-*` のみで不変（NFR-DS-02 は無関係）。すべて **opt-in の `--piper-g2p`**（既定 OFF）が引き込む単一の依存鎖から来る:

```
vokra-server -> vokra-piper-g2p -> piper-plus-g2p (ayutaz/piper-plus, rev 41f3696)
  -> jpreprocess / lindera-dictionary / encoding / quick-xml
```

| 種別 | 対象 | 判断 |
|---|---|---|
| **CC0-1.0** | `encoding-index-{japanese,korean,simpchinese,singlebyte,tradchinese}` + `encoding_index_tests` | **許可**（依頼者承認 2026-07-19、理由「ライブラリに制約が出ない」）。実体は Unicode↔レガシー CJK の**符号位置対応表 = データ**。`allow` への一括追加ではなく crate 限定にしたのは、CC0 が著作権は放棄しても**特許権を waive しない**ため（Vokra が Apache-2.0 を選んだ理由は特許 grant）。将来の CC0 **コード**依存は改めて審査を要する |
| **CDLA-Permissive-2.0** | `webpki-roots`（0.26.11 / 1.0.8） | **許可**（同承認）。Mozilla の CA 証明書バンドル = 同じく許諾型データライセンスで下流に義務を課さない。0.26.11 は `ureq` ← `jpreprocess-naist-jdic` の **build-dependency**、1.0.8 は `reqwest` の **dev-dependency** — **いずれも製品バイナリには入らない** |

**RUSTSEC ignore 4 件**（`integrations/vokra-server/deny.toml` に根拠を全文記載）:

| ID | 内容 | 到達性 |
|---|---|---|
| RUSTSEC-2026-0194 | quick-xml 0.37.5 — 重複属性名チェックの二次時間（remote DoS） | **到達不能**。quick-xml の唯一の利用箇所は piper-plus-g2p の SSML パーサだが、`vokra-piper-g2p` / `vokra-server` はどちらも `ssml` / `SsmlParser` を一切参照しない（TTS は plain text に対し `phonemize` を呼ぶのみ）|
| RUSTSEC-2026-0195 | quick-xml 0.37.5 — namespace 宣言の無制限確保（memory DoS） | **二重に到達不能**。`NsReader` 固有だが、piper-plus-g2p は plain `Reader` のみ使用 |
| RUSTSEC-2025-0141 | bincode unmaintained（← jpreprocess-core） | 脆弱性ではなく保守状態。同梱 NAIST JDIC 辞書の読込に使用（バイト列は crate 同梱でリクエスト由来ではない）|
| RUSTSEC-2021-0153 | `encoding` unmaintained（← lindera-dictionary） | 同上。上記 CC0 例外と同じ crate 群 |

**恒久的な修正は upstream 側**: quick-xml >= 0.41 への更新は `piper-plus-g2p` の `^0.37` に対し semver 非互換のため、本リポジトリからは到達できない（`cargo update` 不可）。`ayutaz/piper-plus` の rev を進める際に 4 件とも再評価すること。

**再評価トリガー（重要）**: SSML を受け付ける実装（`/api/tts` や `/v1/audio/speech` での `<speak>` パススルー等）を入れた瞬間に RUSTSEC-2026-0194/0195 は**実際に live になる**（リクエスト本文が攻撃者制御になるため）。その場合は先に piper-plus を更新し、ignore を外すこと。

---

## 3. モデル Weight ライセンス表

| モデル | Code License | Weight License | 商用可 | Vokra 公式配布 | 備考 |
|-------|------------|-------------|-----|-------------|-----|
| **Silero VAD v5** | MIT | MIT | ○ | ★ 公式 zoo | v5 で 3x faster、size ~2MB (v4 は 1.7MB) |
| **Whisper base/small/medium/large-v3/turbo** | MIT | MIT | ○ | ★ 公式 zoo | OpenAI 公式 |
| **piper-plus (ayutaz) 全モデル** | MIT | MIT | ○ | ★ 公式 zoo | 依頼者作、8 言語、eSpeak-NG 依存なし |
| **Kokoro-82M** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | hexgrad、iSTFTNet 系 vocoder |
| **CosyVoice / CosyVoice2 / CosyVoice3** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Alibaba FunAudioLLM。**M3-09 で自前実装 scaffold**（`crates/vokra-models/src/cosyvoice2/`、text encoder + Flow Matching stub + Mimi bridge + GGUF converter。実 checkpoint parity は依頼者 HF アクセス前提の follow-up）。 |
| **Sesame CSM-1B** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Sesame AI Labs。**M4-05 で native 自前実装**（`crates/vokra-models/src/csm/` = Llama-3.2-flavor backbone + depth transformer + Mimi neural chain（`crates/vokra-models/src/mimi/`、M4-06 Moshi と共有）、`vokra-cli convert --model csm`、`registry_lookup("sesame-csm"/"csm-1b") == Permissive`）。実 checkpoint / tokenizer は **HF gated repo**（sesame/csm-1b + meta-llama/Llama-3.2-1B）= T29 依頼者入手 + §3.1 sign-off 前提、**zoo 公開は sign-off 通過が前提**（sign-off 前に配布 URL を公開しない）。Mimi weight（Kyutai CC-BY 4.0）の attribution は NOTICE §5 が encoder / neural decoder 消費分まで cover。 |
| **Moshi (Helium + Mimi)** | Apache 2.0 | **CC-BY 4.0** | ○ (要 credit) | ★ 公式 zoo | Kyutai、attribution 表示義務 → docs/legal-compliance.md 参照。**M4-06 で native 自前実装**（`crates/vokra-models/src/moshi/` = Helium temporal transformer + per-step-weight depformer + inner monologue + full-duplex session、Mimi neural chain は M4-05 共有 module を consume、`vokra-cli convert --model moshi`、`registry_lookup("moshi") == AttributionRequired`）。**FR-MD-09 attribution 表示機能を実装**（converter が `vokra.provenance.attribution` を焼き込み → `Session::attribution` Rust API + C ABI `vokra_model_attribution` + `vokra-cli` 起動 banner の 3 面、chunk 不在時は registry fallback で AttributionRequired が常に非空 — NOTICE §5 が LM weight 消費分まで cover）。実 checkpoint（`kyutai/moshiko-pytorch-bf16`、~15GB BF16）の sourcing + §3.1 sign-off は T29 依頼者、**zoo 公開は sign-off 通過が前提**。CLI banner の `--quiet` 抑止可否は sign-off 判定事項として flag 済（M4-06-T24）。 |
| **Voxtral (Mistral)** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Mistral、2025-07 リリース。**M3-10 で自前実装 scaffold**（`crates/vokra-models/src/voxtral/`、Whisper 派生 audio encoder + Mistral GQA/RoPE/SwiGLU/RMSNorm text decoder + ASR/S2S heads + config-aware converter `convert_voxtral_file`。実 multilang WER は follow-up）。 |
| **DAC (Descript)** | MIT | MIT | ○ | ★ 公式 zoo | Descript 公式。**M4-04 で `dac_rvq` op + converter + standalone zoo GGUF を実装**（`crates/vokra-ops/src/dac_rvq.rs` factorized decode、`vokra-cli convert --model dac`（要 `dac_prepare_checkpoint.py` side-car）、zoo primary = 24 kHz / 8 kbps variant（tag 0.0.4、75 Hz）。配布解禁は §3.1 の owner sign-off 待ち = fail-closed）。 |
| **Mimi codec (Kyutai)** | Apache 2.0 | CC-BY 4.0 | ○ (要 credit) | ★ 公式 zoo | Moshi パッケージの一部。**M3-06 で `mimi_rvq` op を実装**（`crates/vokra-ops/src/mimi_rvq.rs`、CC-BY 4.0 attribution は `NOTICE` §5 に記載、`registry_lookup("mimi") == AttributionRequired`）。**M4-04 で standalone codec GGUF（`vokra.mimi.*` persisted、`vokra-cli convert --model mimi`）+ CSM/Moshi 向け multi-stream streaming 完成**（配布物同梱の NOTICE §5 で attribution 充足、standalone zoo 配布判断は §3.1 Mimi 行の sign-off に含める）。 |
| **WavTokenizer** | MIT | MIT | ○ | ★ 公式 zoo | 中山大 (repo owner jishengpeng)。**M4-16 で `wavtokenizer_vq` op を実装**（`crates/vokra-ops/src/fsq_codec.rs`、FSQ family = RVQ と別サブグラフ、parity fixture は合成 weight のみ = pretrained 未使用）。GitHub `jishengpeng/WavTokenizer` LICENSE = MIT を 2026-07-15 に CC 再確認。配布解禁は §3.1 owner sign-off 待ち = fail-closed（M4-16-T14） |
| **X-Codec 2 (Llasa)** | MIT（GitHub `zhenye234/X-Codec-2.0` + PyPI `xcodec2` 0.1.5 metadata、2026-07-15 CC 確認） | **⚠ 齟齬 — T14 確定待ち**: 本表旧値 MIT ↔ milestones.md §8 / deliverables.md §3.5「MIT+Apache 2.0 dual」↔ **HF `HKUSTAudio/xcodec2`（weight 配布 repo）README front-matter `license: cc-by-nc-4.0`（2026-07-15 CC fetch）** | **⚠ T14 判定待ち**（weight が CC-BY-NC 4.0 確定なら ✕ 非商用） | **⚠ 保留**（sign-off 空欄 = 配布不可の fail-closed 運用が既に効いている） | HKUST。**M4-16 で `xcodec2_fsq` op を実装**（`crates/vokra-ops/src/fsq_codec.rs`、engine op のみ = EnCodec FR-OP-32 と同じ「op は対応・weight は別判定」姿勢が可能。parity fixture は合成 projection のみ + reference code は vector-quantize-pytorch 1.17.8 (MIT) = **pretrained weight 未 DL・未使用**）。license 表記の 3 系統齟齬は §3.1 flag 節参照、確定は T14 owner sign-off。NC 確定時は `license_class.rs` の `xcodec2` 分類（現 Permissive）の変更差し戻しが必要 |
| **UTMOS22-strong (SaruLab)** | MIT（GitHub `sarulab-speech/UTMOS22` LICENSE = MIT "Copyright (c) 2022 Saruwatari&Koyama laboratory, The University of Tokyo"、GitHub API spdx `MIT`。HF space `sarulab-speech/UTMOS-demo` cardData `license: mit` + LICENSE ファイル同文。2026-07-17 campaign-2 utmos-probe で一次確認） | 同左 MIT（ckpt `epoch=3-step=7459.ckpt` は上記 space の配布物、sha256 `44c57e3e4135a243b43d2c82b6a693fcd56f15f9ad0e1eb2a8b31fdecd3a49b8`。同梱の SSL `wav2vec_small.pt` sha256 `c66c39eaed1b79a61ea8573f71e08f6641ff156b6a8f458cfaab53877dfa4a26` = fairseq リポ **MIT** / HF `facebook/wav2vec2-base` タグ **apache-2.0**、どちらでも permissive） | ○ | **要 owner sign-off**（§3.1、fail-closed） | **評価用メトリクス**（zoo の TTS/ASR モデルではなく `vokra-eval` の品質ゲート実装、NFR-QL-02 / FR-OP-93）。**M5-15 で native 自前実装 + upstream parity 達成**（`crates/vokra-eval/src/metrics/utmos.rs` = `wav2vec2_regression.v1`、`vokra-cli convert --model utmos`（要 `utmos_prepare_checkpoint.py` side-car）、224 upstream tensor → 223 GGUF tensor（pos_conv weight-norm を offline fold））。**実測: 全 9 stage tap + 最終 score が upstream python 一致**（score \|Δ\| 1.19e-7、6 clip 横断で ≤ 9.3e-7、reference は **upstream 実装を import** する dumper = `tools/parity/utmos_dump_reference.py`）。**残 owner 判断**: fine-tune データ **BVCC / VoiceMOS Challenge 2022**（Zenodo record 10691660、license id `other-open`、access `open`）は 「Blizzard Challenge 由来サンプルの**再配布**禁止」条項を持つ = **データ**の再配布制限であり、**academic-only / model-restriction 条項は record 内に見当たらない**（CC 事実確認、2026-07-17）。ただし weight が Blizzard/VCC の聴取実験音声・評点に由来する「training-data 疑い」クラスであることの最終判断は owner（§3.1 UTMOS 行 Notes）。**Vokra は weight を同梱せず**、owner が取得した GGUF を `--utmos-gguf` で渡す運用（weight 不在時は明示エラー = FR-EX-08）。 |
| **openWakeWord** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | dscripka |
| **CAM++ Speaker Embedding** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Alibaba 3D-Speaker `iic/speech_campplus`。`vokra-convert`（`ModelKind::CamPlus`）で GGUF 変換対応済、変換元 ONNX は `ayousanz/campplus-onnx`（約 27MB、Apache-2.0）。6.91M params、fbank80→192-d embedding（native forward） |
| **ECAPA-TDNN (SpeechBrain)** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | SpeechBrain |
| **WeSpeaker** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Duke Kunshan |
| **TitaNet (NVIDIA NeMo)** | 未確認（一次資料未照合、T13） | **⚠ NVIDIA NC 制約 未確認**（`TITANET_SPEAKER_ENCODE_OP`（`crates/vokra-core/src/m5_residual_ops.rs`）の blocker が一次記述） | **⚠ T13 判定待ち** | **⚠ 保留**（sign-off 空欄 = 配布不可の fail-closed） | FR-OP-80 変種（`speaker_encode`）。**op 未実装 = residual anchor のみ**（`titanet_speaker_encode`、`m5_residual_ops.rs`、`docs/abi-changelog.md:471` 予約）。CAM++ が speaker embedding を既にカバーするため trigger 差別化材料は薄い（M5-ORPHAN-SCOPE T04）。weight license は NVIDIA NC 制約の有無が **未確認** = owner 一次資料照合（T13）。**推定 license を書かない**（fail-closed）。op landing 側 WP まで `license_class.rs` へ分類を足さない。 |
| **pyannote (speaker diarization)** | 未確認（一次資料未照合、T13） | **⚠ HF-gated 未確認**（`DIARIZE_OP`（`crates/vokra-core/src/m5_residual_ops.rs`）= trigger + license 二重 blocker） | **⚠ T13 判定待ち** | **⚠ 保留**（sign-off 空欄 = 配布不可の fail-closed） | FR-OP-82（`diarize`、optional feature flag）。**op 未実装 = residual anchor のみ**。HF-gated model の商用配布可否（利用規約承諾要）は **未確認** = owner 一次資料照合（T13、M5-ORPHAN-SCOPE T05）。**推定 license を書かない**（fail-closed）。op landing 側 WP まで `license_class.rs` へ分類を足さない。 |
| **DeepFilterNet3** | **MIT / Apache-2.0 dual**（upstream Rikorose/DeepFilterNet の LICENSE-MIT + LICENSE-APACHE、2026-07-17 campaign-2 で一次確認） | 同左（checkpoint は同リポジトリ配布物として同一 dual、release sha256 `49c52edc…`） | ○ | ★ 公式 zoo（要 owner sign-off T18） | Rikorose、Speech Enhancement。**M4-20 (c) で `denoise` op を native 自前実装** → **2026-07-17 に実 checkpoint での完全実装・parity 達成**（`9b718d1`: libDF topology 転写 = Vorbis-window STFT/ERB frontend + conv/GRU encoder + ERB/DF decoder + lookahead deep filtering、115 verbatim upstream-named tensors、`vokra.denoise.*` schema v2、`convert --model denoise`）。**実測: enhanced 波形 max \|Δ\| 4.17e-7、SI-SNR 14.768399 dB vs upstream 14.768398 dB（gap 2.0e-7 dB）、21 stage tap 全 PASS**（env-gated `parity_denoise_dfn3`）。**owner 残は T18 の最終 license sign-off のみ**（T17 の実 checkpoint parity は達成済）。attribution は NOTICE §8（DeepFilterNet MIT）記載。 |
| **RNNoise** | BSD | BSD | ○ | ★ 公式 zoo | Xiph。denoise 代替候補（M4-20 (c)、DeepFilterNet が第一候補）。 |
| **GTCRN** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo（要 owner license 事前確認 T18） | 2023。denoise 代替候補。license 事前確認は owner T18（`docs/m4-scope-expansion-2026-07-13.md` §BIG-10 依頼者タスク）。 |
| **AudioSeal (Meta)** | MIT | MIT | ○ | ★ 公式 zoo | 推奨デフォルト watermark |
| **F5-TTS (SWivid)** | MIT | **CC-BY-NC 4.0** | ✕ 非商用 | ✕ research flag | エンジンは対応、weight 別途取得 |
| **E2-TTS** | MIT | 要確認 | △ | ✕ audit 後判断 | 論文実装のみ |
| **Fish-Speech v1.4/v1.5** | Apache 2.0 | **CC-BY-NC-SA 4.0** | ✕ 非商用+ShareAlike | ✕ research flag | |
| **Bark (Suno)** | MIT | MIT (元 CC-BY-NC → 変更) | △ | ✕ (Suno voice cloning 方針で禁止) | post-v1.0 GA 検討、research flag |
| **StyleTTS 2** | MIT | 要確認 | △ | ✕ audit 後判断 | |
| **Matcha-TTS** | MIT | MIT | ○ | ★ post-v1.0 GA | |
| **RVC v2** | MIT | **不明 / 学習権利疑い** | △ | ✕ **vokra-voiceclone-experimental** に分離 | training data copyright laundering の Reddit/GitHub Issue で複数指摘 |
| **GPT-SoVITS** | MIT | 不明 | △ | ✕ voiceclone-experimental 分離 | |
| **EnCodec (Meta)** | MIT | **CC-BY-NC 4.0** | ✕ 非商用 | ✕ research flag | 商用は DAC/Mimi/WavTokenizer 推奨。**FR-OP-32 恒久制約**により公式 model zoo 非搭載を維持（M2-13 runtime gate + release CI 側の `scripts/compliance/check-encodec-exclusion.sh` 二重防御、M3-06 ADR §D2）。 |
| **BigVGAN reference (NVIDIA)** | **NVIDIA Source Code License-NC** | 非商用 | ✕ | ✕ | Vokra は **論文からスクラッチ再実装**、reference 未使用 (NOTICE 明記) |
| **HiFi-GAN reference** | MIT | MIT (公式) | ○ | ○ | Kaggle 系派生は要確認 |
| **Vocos** | MIT | MIT | ○ | ★ 公式 zoo | Charactr AI |
| **StyleTTS 2** | MIT | 要確認 (yl4579) | △ | ✕ audit 後 | issue #117 で ONNX 化議論継続中 |

**Voice cloning モデル分離** (レビュアー D 指摘 D4):
- 上記表の RVC v2、GPT-SoVITS、その他話者クローン用途モデルは **`vokra-voiceclone-experimental` リポジトリに完全分離**
- Vokra core は VAD/ASR/TTS のみを公式パッケージに含める
- Voice cloning 機能ドキュメントには **explicit consent workflow** のサンプル (WAV に signed manifest を要求) を含める
- Tennessee **ELVIS Act (2024-07-01)** と連邦 **NO FAKES Act** (2025-04 再導入) の tool-distributor liability 対策

**Speaker embedding は core に残す**: 話者クローンではなく話者特徴抽出 (現代 zero-shot TTS の必須入力)、ELVIS Act の "primary purpose" 判定を避けるため機能を限定 (embedding 抽出のみ、任意音声への転写機能は含めない)。

### research flag enforcement 機構（M2-13 実装、2026-07-04）

上表の weight license 分類を **機構として強制**する research-flag gate を実装した（FR-CP-03 / FR-MD-10 / FR-OP-32、`crates/vokra-core/src/compliance/`）。

- **本表が SoT**: `LicenseClass`（`Permissive` / `AttributionRequired` / `NonCommercial` / `NonCommercialShareAlike` / `Unknown`）の built-in registry は本 §3 の各行の機械写像であり、独自のライセンス判定は導入しない。**F5-TTS = `NonCommercial`（CC-BY-NC 4.0）/ Fish-Speech v1.4/v1.5 = `NonCommercialShareAlike`（CC-BY-NC-SA 4.0）/ EnCodec = `NonCommercial`（CC-BY-NC 4.0）** が gate 要に分類され、**research flag なしでロード拒否**（`VokraError::ResearchLicenseRequired`、明示エラー・silent load 禁止）。
- **fail-closed**: provenance（`vokra.provenance.*` GGUF chunk）も registry も引けない weight は `Unknown` に倒し、gate 要（研究フラグ必須）とする。分類不能を「商用可」と誤判定しない。
- **weight license ≠ crate license（別機構）**: 本 gate は **モデル weight** の非商用ライセンスを対象とする。依存 crate の GPL/LGPL 排除は別機構 = `cargo-deny`（§2、NFR-LC-02/04、CI required check）が担い、本機構は変更しない。
- **公式 zoo 非搭載の維持**: 公式 model zoo は Apache 2.0 / MIT / CC-BY（attribution）weight のみ（§「Vokra 公式配布」列の ★）。CC-BY-NC 系（F5-TTS / Fish-Speech / EnCodec）は `✕ research flag` のまま非搭載を維持する。
- 解錠経路（研究/評価用途限定）: `CompliancePolicy::with_research_license(true)` / 環境変数 `VOKRA_ALLOW_RESEARCH_LICENSE=1` / `ComplianceLevel::Research`。EnCodec の商用代替は DAC / Mimi / WavTokenizer / X-Codec 2（§3）。警告・免責文言の法務的十分性は FR-MD-13 / X-03（依頼者判断）に従属。

### CC-verified 事実確認（2026-07-07 M2 対応 / 2026-07-10 M3-09 + M3-10 追記）

**位置付け（重要）**: 本節は Claude Code（CC）が **一次公表資料の写し** として実施した事実確認の記録である。**法務的な配布判断（"配布して良い" の意思決定）は依頼者（`ayutaz`）に帰属**し、本節では実施しない。Owner sign-off は各行末尾の空欄に依頼者が別途記入する。判断の下敷きとなる事実（upstream の license 表記 / 発行元 / 公表 URL）だけをここに固定する。

事実確認の方法:
- **一次資料の URL 引用**: 各モデルの Hugging Face model card / GitHub リポジトリ LICENSE ファイル / 公式論文の license 節など、upstream の権威ある公表資料のみを引用する。
- **機構との突合**: `crates/vokra-core/src/compliance/license_class.rs::registry_lookup()` の built-in registry に本節の分類が反映されていることを、同ファイル内のユニットテスト (`registry_lookup_permissive`, `registry_lookup_non_commercial`, `registry_lookup_share_alike` 相当) で pin。研究フラグなしの CC-BY-NC 系 load 拒否は
`crates/vokra-core/src/compliance/mod.rs` の統合テストで検証済み（`cargo test -p vokra-core compliance` all green）。
- **grep 検証**: 公式 model zoo publish 経路（`vokra-convert` の permissive/attribution 分岐のみ）に F5-TTS / Fish-Speech / EnCodec の model_id が混入していないことを機械 grep（`grep -rn "f5-tts\|fish-speech\|encodec" crates/vokra-convert/src/`）で確認。

| モデル | Weight License | 一次資料（CC 引用） | Registry 分類（機械） | 公式 zoo 適格 | Owner sign-off |
|---|---|---|---|---|---|
| **Whisper base/small/medium/large-v3/turbo** | **MIT** | `openai/whisper` GitHub リポジトリ LICENSE ファイル（MIT）；Hugging Face `openai/whisper-base` 〜 `openai/whisper-large-v3-turbo` model cards の license: mit タグ | `Permissive` | ✓ | ______________ |
| **Kokoro-82M** | **Apache-2.0** | Hugging Face `hexgrad/Kokoro-82M` model card の license: apache-2.0 タグ；同リポジトリ LICENSE | `Permissive` | ✓ | ______________ |
| **piper-plus (依頼者作) 全モデル** | **MIT** | `ayutaz/piper-plus` GitHub リポジトリ LICENSE (MIT)；ONNX voice model の LICENSE も MIT | `Permissive` | ✓ | ______________ |
| **CAM++ Speaker Embedding** | **Apache-2.0** | ModelScope `iic/speech_campplus` の license: apache-2.0 タグ；変換元 `ayousanz/campplus-onnx` の LICENSE（Apache-2.0） | `Permissive` | ______________ |
| **CosyVoice2-0.5B** | **Apache-2.0** | Hugging Face `FunAudioLLM/CosyVoice2-0.5B` model card の license: apache-2.0 タグ；`FunAudioLLM/CosyVoice` GitHub リポジトリ LICENSE ファイル（Apache License Version 2.0） | `Permissive` | ✓ | ______________ |
| **Voxtral-Mini-3B-2507** | **Apache-2.0** | Hugging Face `mistralai/Voxtral-Mini-3B-2507` model card の license: apache-2.0 タグ | `Permissive` | ✓ | ______________ |
| **Voxtral-Small-24B-2507** | **Apache-2.0** | Hugging Face `mistralai/Voxtral-Small-24B-2507` model card の license: apache-2.0 タグ | `Permissive` | ✓ | ______________ |
| **DAC (Descript)** | **MIT** | `descriptinc/descript-audio-codec` GitHub リポジトリ LICENSE ファイル（MIT、GitHub API license.spdx_id = MIT を 2026-07-15 に CC 確認）；weights は同リポジトリの GitHub releases 配布物（`weights_24khz.pth` 等、`dac/utils/__init__.py` L18-39 の pinned URL 表）で **別段の weight license ファイルは同梱されていない**（リポジトリ LICENSE の下で公表） | `Permissive` | ✓ | ______________ |
| **WavTokenizer** | **MIT** | `jishengpeng/WavTokenizer` GitHub リポジトリ LICENSE ファイル（MIT License、copyright jishengpeng 2024 — 2026-07-15 CC fetch）；released checkpoints（`WavTokenizer-{small,medium}-*-24k-4096`）は同リポジトリ README から配布・別段の weight license 記載なし | `Permissive` | ✓ | ______________ |

**M4-14 activation note（Whisper family 完成 = M2-06 carry-over、2026-07-15）**: Whisper **small / medium / turbo** は M4-14 で `parity-whisper-real.yml` の parity CI matrix に昇格（workflow_dispatch opt-in leg。base / large-v3 と同一の HF DL → `vokra-cli convert` → dumper → `cargo test parity_whisper` 経路）。5 サイズ共通 MIT/MIT ゆえ**本表への新規行追加は無し**（上記 Whisper 行と §3 model zoo 行が M2-06 時点から 5 サイズをカバー済、FR-MD-13 の追記は本 note）。**provenance 注意（owner sign-off 時の混同防止）**: turbo の HF checkpoint は `openai/whisper-large-v3-turbo` — large-v3（`openai/whisper-large-v3`）とは**別 checkpoint**（distilled、decoder 4 層）だが、tokenizer は large-v3 と同一の 51866-token vocabulary を共有する（in-repo anchor: `vokra-convert` whisper converter の n_vocab 行、dumper の `vocab_resource_for` は両サイズを同一 bundled resource に標準化 = M4-14-T03）。実 sign-off（下記 template の Whisper small/medium/turbo 行の空欄記入）は M4-14-T11（依頼者）。

**M4-16 activation note + X-Codec 2 license 齟齬 flag（FSQ codec family、2026-07-15）**: M4-16 で `wavtokenizer_vq` / `xcodec2_fsq` op を実装（`crates/vokra-ops/src/fsq_codec.rs`、engine op のみ・parity fixture は合成 weight のみ = **pretrained weight は未 DL・未使用**）。**X-Codec 2 の license 表記は 3 系統で不一致**（CC は事実の surface まで、確定は M4-16-T14 依頼者 sign-off）:

1. **code**: GitHub `zhenye234/X-Codec-2.0` = MIT badge、PyPI `xcodec2==0.1.5` package metadata = MIT（いずれも 2026-07-15 CC fetch）。
2. **weight 配布 repo**: HF `HKUSTAudio/xcodec2`（`model.safetensors` 3.29 GB の在処）README YAML front-matter = **`license: cc-by-nc-4.0`**（2026-07-15 CC fetch）— 本表旧値「MIT/MIT」とも milestones.md §8 / deliverables.md §3.5「MIT+Apache 2.0 dual」とも一致しない。
3. **T14 判定事項**: (a) weight license の実体確定（HF タグが正なら weight は CC-BY-NC 4.0 = 公式 zoo 非搭載・research flag 系へ、EnCodec FR-OP-32 と同型の「op は対応・weight 除外」運用）、(b) per-file か combined か（milestones/deliverables の「dual」表記の出所確認・SoT 表記統一）、(c) NC 確定時は `crates/vokra-core/src/compliance/license_class.rs` の `"x-codec-2" | "xcodec2" => Permissive` 分類の変更を CC に差し戻す（現分類のままだと weight-load gate が NC weight を素通しするため）。**sign-off 空欄 = 配布不可（fail-closed）が既に効いており、判定完了まで zoo 配布は発生しない**。

**Attribution 要（CC-BY 4.0、公式 zoo 搭載可、NOTICE 記載必須）**:

| モデル | Weight License | 一次資料（CC 引用） | Registry 分類 | 公式 zoo | 補足 |
|---|---|---|---|---|---|
| **Mimi codec (Kyutai)** | **CC-BY 4.0** | Hugging Face `kyutai/mimi` model card の license: cc-by-4.0 タグ；同 model card 提供の BibTeX citation `@techreport{kyutai2024moshi, ..., institution = {Kyutai}, year={2024}, url={http://kyutai.org/Moshi.pdf}}` | `AttributionRequired` | ✓ | 商用 OK ただし attribution 要。`NOTICE` §5 に BibTeX citation 記載済（M3-06 で反映、`crates/vokra-ops/src/mimi_rvq.rs` の `registry_lookup("mimi") == AttributionRequired` gate で機構強制） |

**Non-commercial 系（研究フラグ必須、公式 zoo 非搭載を維持）**:

| モデル | Weight License | 一次資料（CC 引用） | Registry 分類 | 公式 zoo | 補足 |
|---|---|---|---|---|---|
| **F5-TTS** | **CC-BY-NC 4.0** | `SWivid/F5-TTS` GitHub `LICENSE`（CC-BY-NC 4.0）；HF model card の license タグ | `NonCommercial` | ✗ 非搭載（維持） | 商用は DAC/Mimi/WavTokenizer 推奨（§3） |
| **Fish-Speech v1.4/v1.5** | **CC-BY-NC-SA 4.0** | `fishaudio/fish-speech` GitHub のライセンス条項；HF model card | `NonCommercialShareAlike` | ✗ 非搭載（維持） | ShareAlike で派生物も同ライセンス |
| **EnCodec (Meta)** | **CC-BY-NC 4.0** | `facebookresearch/encodec` GitHub 内 `LICENSE_WEIGHTS`（CC-BY-NC 4.0） | `NonCommercial` | ✗ 非搭載（維持） | code は MIT だが weight のみ CC-BY-NC |

**残るリスク**（CC が判断できず、依頼者法務判断に委ねる項目）:
- **RVC v2 / GPT-SoVITS**: 学習データの権利関係が公表資料で確定できない（Reddit/GitHub issue で複数指摘）。CC は本節に事実（一次資料での不明確さ）だけを記す。判断は依頼者の `vokra-voiceclone-experimental` 別リポジトリ運用方針に従属。
- **Bark (Suno)**: 元 CC-BY-NC → MIT へ変更された経緯が Suno 公式のライセンス方針（voice cloning 再学習禁止）と整合するかの判定は法務案件。CC は本節では判定しない。
- **StyleTTS 2 (yl4579)**: weight license が公表資料で明示されていない（code は MIT）。CC の分類は `Unknown` → fail-closed（研究フラグ必須）に落ちるため、公式 zoo 非搭載側の安全側に倒れる。商用配布判断は依頼者。

**Article 50 checklist（`docs/legal-compliance.md`）の現状**:
- ✅ Machine-readable marking の設計面: `vokra.provenance.*` GGUF chunk（`crates/vokra-core/src/compliance/`）と `LicenseClass` 分類。
- ⚠ **Deferred（残タスク）**: AudioSeal watermark の runtime embedding（FR-CP-01）と C2PA manifest sign/verify（FR-CP-02）は 2026-07-04 依頼者ドロップにより **config 面（`WatermarkConfig`）のみ実装**、実埋め込みは deferred。**NFR-LG-01（EU AI Act Article 50）と NFR-LG-02（California SB 942）の runtime marking 要件を完全には満たしていない**。運用側で TTS 出力に AI 生成表示（disclosure text）を必ず加える必要がある（M2-13 `WatermarkConfig::require_disclosure` の設定面はある）。
- 詳細な checklist 通過は依頼者判断（sign-off はここでは行わない）。

### Owner sign-off template（依頼者記入）

**位置付け（重要）**: 前節 §CC-verified 事実確認は Claude Code（CC）が一次公表資料（upstream の LICENSE ファイル / model card / GitHub 等）を写した **事実の記録** である。本節は、その事実を踏まえて **依頼者（`ayutaz`）が下す配布可否の法務的意思決定** を記録する場である。CC はライセンス facts の verification（upstream の license 表記の引用）のみを担い、"このモデルを商用配布して良いか / research-only で扱うか / 拒否するか" という distribute-or-not の法務判断は本節で依頼者が明示的にサインオフする。

依頼者は各行の "Owner sign-off (YYYY-MM-DD)" 欄に日付、"Approval" 欄で該当箱にチェック（`☑`）、必要に応じて "Notes" 欄に判断根拠を記入する。空欄のままの行は「未サインオフ＝公式配布不可」の運用とする（fail-closed）。

| Model | Weight License | CC-verified date | Owner sign-off (YYYY-MM-DD) | Approval | Notes |
|---|---|---|---|---|---|
| **Whisper base** | MIT | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Whisper small** | MIT | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Whisper medium** | MIT | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Whisper large-v3** | MIT | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Whisper turbo** | MIT | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Kokoro-82M** | Apache-2.0 | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **piper-plus** | MIT | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **CAM++** | Apache-2.0 | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **F5-TTS** | CC-BY-NC 4.0 | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Fish-Speech v1.4** | CC-BY-NC-SA 4.0 | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **Fish-Speech v1.5** | CC-BY-NC-SA 4.0 | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **EnCodec** | CC-BY-NC 4.0 | 2026-07-07 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | |
| **CosyVoice2-0.5B** | Apache-2.0 | 2026-07-10 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M3-09 対応 |
| **Voxtral-Mini-3B-2507** | Apache-2.0 | 2026-07-10 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M3-10 対応 |
| **Voxtral-Small-24B-2507** | Apache-2.0 | 2026-07-10 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M3-10 対応 |
| **Mimi codec (Kyutai)** | CC-BY 4.0 | 2026-07-10 | ______________ | ☐ Commercial (attribution 込) / ☐ Research-only / ☐ Rejected | M3-06 で NOTICE §5 反映済、機構 gate 済。**M4-04 standalone zoo 対応** — standalone codec GGUF（`--model mimi`、kyutai/moshiko-pytorch-bf16 tokenizer safetensors 由来）の zoo 配布判断も本行の sign-off で一括（M4-04-T20） |
| **DAC 24khz (Descript)** | MIT | 2026-07-15 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M4-04 対応（`dac_rvq` op + `--model dac` converter、zoo primary = 24 kHz/8 kbps tag 0.0.4）。§3 表は ★ 公式 zoo 指定済だが本行 sign-off まで配布不可（fail-closed、M4-04-T20） |
| **WavTokenizer** | MIT | 2026-07-15 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M4-16 対応（`wavtokenizer_vq` op、fixture は合成のみ）。released 構成 vocab 4096 / d_model 512（ADR M4-16 §D-c） |
| **X-Codec 2 (Llasa)** | **齟齬 — code MIT / HF weight タグ CC-BY-NC 4.0** | 2026-07-15 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M4-16 対応（`xcodec2_fsq` op、fixture は合成のみ）。**T14 判定 3 点**: weight license 実体（HF `HKUSTAudio/xcodec2` タグ cc-by-nc-4.0 vs 本表旧値 MIT vs milestones/deliverables「MIT+Apache dual」）／per-file か combined か／NC 確定時の `license_class.rs` 分類変更差し戻し（§CC-verified の M4-16 flag 節参照） |
| **Sesame CSM-1B** | Apache-2.0 | 2026-07-15 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M4-05 対応（§3 表は ★ 公式 zoo 指定）。実 checkpoint / tokenizer は **HF gated repo**（`sesame/csm-1b` + `meta-llama/Llama-3.2-1B`）= T29 依頼者入手。共有 Mimi neural chain の weight（Kyutai **CC-BY 4.0**）attribution は NOTICE §5 が cover。本行 sign-off まで配布不可（fail-closed） |
| **Moshi (Helium + Mimi)** | CC-BY 4.0 | 2026-07-15 | ______________ | ☐ Commercial (attribution 込) / ☐ Research-only / ☐ Rejected | M4-06 対応（§3 表は ★ 公式 zoo 指定、attribution 表示義務）。実 checkpoint `kyutai/moshiko-pytorch-bf16`（~15GB BF16）= T29 依頼者入手。FR-MD-09 attribution 表示（`vokra_model_attribution` 他 3 面）実装済、NOTICE §5 が LM weight 消費分まで cover。本行 sign-off まで配布不可（fail-closed）。CLI banner `--quiet` 抑止可否も本 sign-off 判定事項（M4-06-T24） |
| **UTMOS22-strong (SaruLab)** | MIT（SSL `wav2vec_small.pt` は fairseq MIT / HF apache-2.0） | 2026-07-17（campaign-2 utmos-probe） | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M5-15 対応（**評価メトリクス**であって配布モデルではない — `vokra-eval` の NFR-QL-02 5% ゲート実装）。**CC 側は完了**（upstream parity 達成、§3 表参照）。**owner 判断 2 点**: (1) fine-tune データ BVCC が Blizzard/VCC 聴取実験由来である 「training-data 疑い」クラスの受容可否（Zenodo record 10691660 に academic-only 条項は**無い**が、サンプル再配布禁止条項は有る = CC 事実確認）、(2) weight を Vokra が**同梱しない**現行運用（owner 取得 → `--utmos-gguf` で注入）を維持するか、zoo 配布に格上げするか。本行 sign-off まで zoo 配布不可（fail-closed）。**CC は本欄を pre-fill しない** |
| **DeepFilterNet3** | **MIT / Apache-2.0 dual**（2026-07-17 一次確認） | 2026-07-15（license 精査 2026-07-17 更新） | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M4-20 (c) 対応（`denoise` op、§3 表は ★ 公式 zoo（要 owner sign-off T18））。**T17 実 checkpoint parity は 2026-07-17 に達成済**（`9b718d1`、SI-SNR gap 2.0e-7 dB / 波形 max |Δ| 4.17e-7 / 21 tap PASS）→ **残は T18 の owner sign-off のみ**（本欄の署名・判定は owner 記入、CC は pre-fill しない）。attribution は NOTICE §8（DeepFilterNet MIT）記載。本行 sign-off まで配布不可（fail-closed） |
| **TitaNet (NVIDIA NeMo)** | ⚠ NVIDIA NC 未確認 | 2026-07-20（staging のみ、license 未照合） | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M5-ORPHAN-SCOPE-T14 staging（FR-OP-80 変種 `titanet_speaker_encode`、**op 未実装 = residual anchor**、`docs/abi-changelog.md:471`）。**T13 判定 2 点**: (1) TitaNet pretrained checkpoint の NVIDIA NC 制約有無の一次資料照合、(2) NC 確定時は非商用 = zoo 除外。CAM++ が既に speaker embedding をカバー（差別化材料は薄い、T04）。**本欄の署名・判定は owner 記入、CC は pre-fill しない**（`weight license` 列は「未確認」= 推定を書かない）。本行 sign-off まで配布不可（fail-closed） |
| **pyannote (speaker diarization)** | ⚠ HF-gated 未確認 | 2026-07-20（staging のみ、license 未照合） | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | M5-ORPHAN-SCOPE-T14 staging（FR-OP-82 `diarize`、optional feature flag、**op 未実装 = residual anchor**）。**T13 判断**: HF-gated model の利用規約承諾 + 商用配布可否の一次資料照合（trigger + license 二重 blocker、`DIARIZE_OP`（`m5_residual_ops.rs`））。**本欄の署名・判定は owner 記入、CC は pre-fill しない**（`weight license` 列は「未確認」= 推定を書かない）。本行 sign-off まで配布不可（fail-closed） |

---

## 4. Codec / Vocoder / 音声処理ライブラリ

| ライブラリ | ライセンス | 用途 | Vokra 採用 | 代替 |
|---------|----------|-----|------------|-----|
| **pocketfft** (C++) | BSD-3-Clause | FFT | 参考（Rust 移植） | FFTW3 (GPL) 排除 |
| **realfft** (Rust) | MIT/Apache 2.0 | RFFT | ★ 採用 | pocketfft の実質後継 |
| **speexdsp resampler** (C) | BSD | polyphase sinc interpolation | 参考（Rust 移植） | soxr (LGPL) 排除 |
| **speexdsp AEC** (C) | BSD | Acoustic Echo Cancellation | ★ **採用**（M4-03 で Rust port 実施 = `vokra-ops::aec`、mdf.c AUMDF float build、upstream pin `7a158783df74`。attribution は NOTICE §7 + `THIRD_PARTY_LICENSES/speexdsp-LICENSE.txt`、ADR M4-03 §D-(a)） | WebRTC AEC3 と比較の上で採用 |
| **WebRTC AEC3** | BSD | AEC | ✕ **M4-03 で不採用**（ADR M4-03 §D-(a) 参照: 実装規模が数十ファイル級で 30 分チケット列に収まらず、delay estimator が queue 設計と絡む。license 上の問題ではない — 将来の再評価は妨げない） | speexdsp AEC を採用 |
| **RNNoise** | BSD | Noise Suppression | ★ v1.0-rc（M4-20 (c) 代替候補） | |
| **DeepFilterNet3** (Rust) | MIT | Noise Suppression | ★ v1.0-rc（M4-20 (c) 第一候補、owner sign-off T18） | Rikorose 公式 |
| **GTCRN** | Apache 2.0 | Noise Suppression | ★ v1.0-rc 検討（M4-20 (c) 代替候補） | |
| **AudioSeal** (Meta) | MIT | Watermark | ★ 推奨デフォルト | |
| **SynthID audio** (Google DeepMind) | Google 個別契約要 | Watermark | 検討中 | 代替: SilentCipher / WaveGuard (OSS) |
| **C2PA (c2pa-rs)** | Apache 2.0 | Content provenance manifest | △ config 面のみ（embedding deferred、§3 Article 50 checklist 参照） | Adobe |
| **libsamplerate** | BSD | resample | 検討中 | speexdsp と比較 |
| **libsoxr** | LGPL | resample | ✕ 排除 | speexdsp で代替 |
| **rubberband** | GPL | pitch shift / time stretch | ✕ 排除 | 自前実装 or 除外 |
| **libespeak-ng** | GPL-3.0-or-later | G2P | ✕ 排除 | piper-plus 独自 G2P、または misaki / IPA 辞書 |
| **OpenFST** | Apache 2.0 | WFST decoder | ★ 実装確定 (M5-06、from-scratch Rust port、runtime は OpenFST 非依存) | dev 時 parity fixture 生成のみ OpenFST CLI を使用 (成果物の依存にならない)。owner clean-room sign-off = M5-06-T17 |
| **kenlm** | LGPL | n-gram LM | ✕ 検討中止 | 独自 Rust 実装 or `lm-rs` |
| **librosa** (Python 参考) | ISC | Mel filter bank 参考 | 参考のみ | Slaney/HTK 両対応の Rust 実装を独自 |
| **torchaudio** (Python 参考) | BSD | 参考 | 参考のみ | |

---

## 5. G2P (Grapheme-to-Phoneme) 戦略 — レビュアー C/D 指摘 #5 対応

**eSpeak-NG (GPL-3.0) を Vokra core から完全排除**する。**実装状況（M0）**: 実 8 言語 G2P は out-of-workspace の opt-in crate `integrations/vokra-piper-g2p`（`piper-plus-g2p` を git 依存、zero-dep runtime 非干渉、§2 参照）で提供済。以下は言語別の対応方針:

| 言語 | 戦略 | ライセンス |
|-----|-----|----------|
| **日本語** | piper-plus 独自 G2P (`pyopenjtalk` 相当を独自 Rust 実装) | MIT |
| **英語** | piper-plus 独自 G2P (`g2p-en` Apache 2.0 相当) | MIT / Apache 2.0 |
| **中国語** | piper-plus 独自 G2P (pinyin 辞書ベース) | MIT |
| **スペイン語** | piper-plus 独自 G2P | MIT |
| **フランス語** | piper-plus 独自 G2P | MIT |
| **ポルトガル語** | piper-plus 独自 G2P | MIT |
| **スウェーデン語** | piper-plus 独自 G2P (コード対応、モデル未) | MIT |
| **韓国語** | piper-plus 独自 G2P (v1.13.0+) | MIT |
| **その他言語** | phoneme 直接入力 (ホストアプリ側責任)、Vokra API `synthesize_phonemes(ipa_string)` を提供 | N/A |

**IPA (International Phonetic Alphabet) 中間表現**: 全モデルで phoneme 直接入力に対応。ホストアプリ側で外部 G2P (許諾ライセンスの `phonemizer-rs`、商用 SDK 等) を使う場合の pass-through 経路を提供。

---

## 6. NVIDIA CUDA / cuDNN EULA 準拠 (レビュアー D 指摘 D8)

**Vokra の CUDA 対応方針**:

1. **cudart / cudnn / cublas / cufft を Vokra が bundle しない**
   - NVIDIA EULA: "installed only in a private (non-shared) directory location that is used only by the application"
   - Unity Asset Store の `Assets/Plugins/x64/*.dll` は "shared plugins directory" 解釈可能 → EULA 違反リスク

2. **開発者側 install モデル**（M2-03 実装済、`crates/vokra-backend-cuda`）:
   - `dlopen("libcuda.so.1")` / `LoadLibrary("nvcuda.dll")`（+ NVRTC `libnvrtc`）で system install CUDA を実行時検出
   - 検出失敗時は **explicit `VokraError::BackendUnavailable`**（silent CPU fallback しない、FR-EX-08）。CPU 選択は呼び出し側の明示的判断
   - **binding crate（cudarc/cust/rustacuda）は不採用**、Driver API + NVRTC を手書き生 FFI で dlopen（NFR-DS-02、`third_party/NVIDIA-EULA.md`）。CUDA runtime そのものは NVIDIA proprietary

3. **医療 / 車載 / 軍事向け SKU 分離**:
   - NVIDIA EULA: "not tested or certified by NVIDIA for use in critical applications such as avionics, navigation, autonomous vehicle applications, military, medical, life support"
   - **Vokra-critical-safe** SKU (CPU/Vulkan-only ビルド) を別配布
   - README に "Vokra with CUDA is not certified for medical/automotive/aviation/military use per NVIDIA CUDA EULA" と明記

4. **cuDNN 必須にしない**:
   - 音声モデルは cuDNN 依存を排除できる op 選定 (畳み込みは自前 CUDA kernel、attention は FlashAttention v2 port)
   - cuDNN 検出時は Otonx が動的に GEMM/conv kernel を委譲、非検出時は自前 kernel

5. **medical / avionics / autonomous vehicle 顧客の相談ポイント**:
   - CUDA 経由推論は NVIDIA EULA でサポート外 → CPU/Vulkan/CoreML のみ
   - Unity Asset Store distributes に CUDA を含めない (Vokra-cuda-desktop は separate package)

---

## 7. モバイル OS 生成 AI 表示義務

### Apple iOS/macOS App Store Guideline 5.5 (2025-)
- Otonx を組み込んだ Unity/Swift/native アプリは App Review Metadata で "AI voice included" を declare 必要
- Vokra API に `is_ai_generated: bool = true` metadata の pass-through 経路

### Google Play Generative AI Content (2024-06 施行)
- generative AI apps に user report mechanism + guardrails 義務化
- Vokra API に `report_generated_content(content_hash, reason)` 経路
- `docs/legal-compliance.md` の user-report workflow 参照

### 各 App Store の実装は依頼者ドキュメント責任

---

## 8. 商標 / ブランディング

- **Vokra** (Vox ラテン voice + -kra suffix) — 2026-07-02 時点で商標 clear (USPTO/EUIPO/JPO 公開検索 + WebSearch 確認、音声 AI 分野で既存プロジェクト・企業なし)
- **旧候補の却下理由**:
  - Otonx: ONNX Foundation 誤認、"Oh-tonks" 誤読、Ottex.ai 衝突
  - Kotohane (言羽): AV 女優「雫ことはね」(Wikidata Q26046050) + pixivFANBOX R-18 illustrator「コトハネ」との名前衝突、SEO / 企業法務調査 / App Store 審査リスク
- **近接名の識別**: VOKKA (voice models 企業) とは phonetic 差 (Vok-ka vs Vok-ra) で識別可能。Voka (VOKA AI Voice Receptionist、商標登録済) とは 1 字違いだが末尾 -kra で発音・表記とも distinct
- **推奨ドメイン確保**: `vokra.dev` / `vokra.io` / `vokra.ai` / `vokra.rs` (Rust コミュニティ準拠、v0 spike 前に取得)
- **推奨 GitHub org**: `vokra-org` or `vokra-audio`
- **NOTICE 記載**: "Vokra is not affiliated with the ONNX Project or Linux Foundation."
- Otonx / Kotohane (旧名) の利用は非推奨、各ドキュメントの旧名表記は歴史的経緯として残す (rebrand ノート付き)

---

## 9. ライセンス conflict resolution フロー

新規依存 (crate / モデル / library) 追加時:

1. **ライセンス確認** → SPDX ID を本ドキュメントに追記
2. **GPL/LGPL 判定** → GPL/LGPL は原則排除、unavoidable な場合は Wyoming Protocol 経由の分離プロセス
3. **CC-BY-NC / CC-BY-NC-SA 判定** → weight は公式配布から除外、research flag で分離
4. **NVIDIA / Apple / Google プロプライエタリ SDK 判定** → bundle しない、system install 検出のみ
5. **Attribution 要件** (CC-BY 4.0, Apache 2.0 NOTICE) → NOTICE ファイルに追記
6. **特許 grant** (Apache 2.0) → 該当なしなら OK、MIT/BSD は特許 grant なし
7. **PR reviewer 確認** → 依頼者が最終承認、Claude Code は audit ドキュメント更新を必ず含める

---

## 10. 定期監査

- **四半期ごと**: 全依存の最新版でライセンス変更がないかチェック (Piper が MIT → GPL-3.0 化した precedent)
- **モデル追加時**: HF Hub / GitHub の LICENSE / MODEL_CARD を確認、training data の legality も可能な範囲で verify
- **年次**: 弁護士による外部 audit (依頼者裁量)

## 11. 参考出典

- [OHF-Voice/piper1-gpl README](https://github.com/OHF-Voice/piper1-gpl) — GPL-3.0 + eSpeak-NG 二重汚染確認
- [ayutaz/piper-plus README](https://github.com/ayutaz/piper-plus/blob/dev/README_EN.md) — MIT ライセンス、eSpeak-NG 独立
- [Kokoro-82M License](https://huggingface.co/hexgrad/Kokoro-82M) — Apache 2.0
- [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
- [NVIDIA cuDNN EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)
- [ELVIS Act (Tennessee)](https://en.wikipedia.org/wiki/ELVIS_Act)
- [EU AI Act Article 50](https://artificialintelligenceact.eu/article/50/)
- [Mimi codec attribution (Kyutai)](https://huggingface.co/kyutai/mimi)
- [Meta AudioSeal](https://github.com/facebookresearch/audioseal)
- [c2pa-rs (Adobe)](https://github.com/contentauth/c2pa-rs)
- [NVIDIA BigVGAN activations.py (NC ライセンス reference)](https://github.com/NVIDIA/BigVGAN/blob/main/activations.py)
- [F5-TTS SWivid (weight CC-BY-NC 4.0)](https://github.com/SWivid/F5-TTS)
- [Fish-Speech LICENSE](https://github.com/fishaudio/fish-speech)
