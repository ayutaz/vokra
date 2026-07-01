# license-audit.md — Vokra 依存ライセンス総覧

**最終更新**: 2026-07-02
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
| **wgpu** | Apache 2.0 / MIT | WebGPU/Vulkan/Metal/DX12 backend | ○ | v0.1-v1.5 |
| **cudarc** | Apache 2.0 / MIT | CUDA driver API bindings | ○ | cudart bundle しない、system install 検出のみ |
| **metal-rs** | MIT | Metal API bindings | ○ | macOS/iOS backend |
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

**未確認 / 要検討**:
- **G2P Rust crate 選定**: piper-plus 独自 G2P を Vokra からも呼べる形にするか、あるいは `phonemizer-rs` (要ライセンス確認)、`phonikud` (要確認) を検討。**eSpeak-NG (GPL-3.0) は排除**
- **RVC 系 F0 抽出**: RMVPE (MIT) / FCPE (MIT) / CREPE (MIT) の Rust port が未成熟 → 自前実装 or Python 呼び出し (server 用途のみ)
- **soxr resampler (LGPL)**: **不採用**、代替 `speexdsp resampler` (BSD) 相当を Rust で独自実装

**明示的排除**:
- **FFTW3 (GPL-2.0-or-later)**: 排除、pocketfft (BSD-3) Rust 移植で代替
- **libespeak-ng (GPL-3.0-or-later)**: 排除、piper-plus 独自 G2P で代替
- **libopenmp (LGPL)**: 排除、rayon で代替
- **soxr (LGPL)**: 排除、speexdsp AEC/resampler の Rust port で代替
- **rubberband (GPL)**: 排除

---

## 3. モデル Weight ライセンス表

| モデル | Code License | Weight License | 商用可 | Vokra 公式配布 | 備考 |
|-------|------------|-------------|-----|-------------|-----|
| **Silero VAD v5** | MIT | MIT | ○ | ★ 公式 zoo | v5 で 3x faster、size ~2MB (v4 は 1.7MB) |
| **Whisper base/small/medium/large-v3/turbo** | MIT | MIT | ○ | ★ 公式 zoo | OpenAI 公式 |
| **piper-plus (ayutaz) 全モデル** | MIT | MIT | ○ | ★ 公式 zoo | 依頼者作、8 言語、eSpeak-NG 依存なし |
| **Kokoro-82M** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | hexgrad、iSTFTNet 系 vocoder |
| **CosyVoice / CosyVoice2 / CosyVoice3** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Alibaba FunAudioLLM |
| **Sesame CSM-1B** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Sesame AI Labs |
| **Moshi (Helium + Mimi)** | Apache 2.0 | **CC-BY 4.0** | ○ (要 credit) | ★ 公式 zoo | Kyutai、attribution 表示義務 → docs/legal-compliance.md 参照 |
| **Voxtral (Mistral)** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Mistral、2025-07 リリース |
| **DAC (Descript)** | MIT | MIT | ○ | ★ 公式 zoo | Descript 公式 |
| **Mimi codec (Kyutai)** | Apache 2.0 | CC-BY 4.0 | ○ (要 credit) | ★ 公式 zoo | Moshi パッケージの一部 |
| **WavTokenizer** | MIT | MIT | ○ | ★ 公式 zoo | 中山大 |
| **X-Codec 2 (Llasa)** | MIT | MIT | ○ | ★ 公式 zoo | HKUST |
| **openWakeWord** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | dscripka |
| **CAM++ Speaker Embedding** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Alibaba |
| **ECAPA-TDNN (SpeechBrain)** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | SpeechBrain |
| **WeSpeaker** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | Duke Kunshan |
| **DeepFilterNet3** | MIT | MIT | ○ | ★ 公式 zoo | Rikorose、Speech Enhancement |
| **RNNoise** | BSD | BSD | ○ | ★ 公式 zoo | Xiph |
| **GTCRN** | Apache 2.0 | Apache 2.0 | ○ | ★ 公式 zoo | 2023 |
| **AudioSeal (Meta)** | MIT | MIT | ○ | ★ 公式 zoo | 推奨デフォルト watermark |
| **F5-TTS (SWivid)** | MIT | **CC-BY-NC 4.0** | ✕ 非商用 | ✕ research flag | エンジンは対応、weight 別途取得 |
| **E2-TTS** | MIT | 要確認 | △ | ✕ audit 後判断 | 論文実装のみ |
| **Fish-Speech v1.4/v1.5** | Apache 2.0 | **CC-BY-NC-SA 4.0** | ✕ 非商用+ShareAlike | ✕ research flag | |
| **Bark (Suno)** | MIT | MIT (元 CC-BY-NC → 変更) | △ | ✕ (Suno voice cloning 方針で禁止) | v2.0+ 検討、research flag |
| **StyleTTS 2** | MIT | 要確認 | △ | ✕ audit 後判断 | |
| **Matcha-TTS** | MIT | MIT | ○ | ★ v2.0+ | |
| **RVC v2** | MIT | **不明 / 学習権利疑い** | △ | ✕ **vokra-voiceclone-experimental** に分離 | training data copyright laundering の Reddit/GitHub Issue で複数指摘 |
| **GPT-SoVITS** | MIT | 不明 | △ | ✕ voiceclone-experimental 分離 | |
| **EnCodec (Meta)** | MIT | **CC-BY-NC 4.0** | ✕ 非商用 | ✕ research flag | 商用は DAC/Mimi/WavTokenizer 推奨 |
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

---

## 4. Codec / Vocoder / 音声処理ライブラリ

| ライブラリ | ライセンス | 用途 | Vokra 採用 | 代替 |
|---------|----------|-----|------------|-----|
| **pocketfft** (C++) | BSD-3-Clause | FFT | 参考（Rust 移植） | FFTW3 (GPL) 排除 |
| **realfft** (Rust) | MIT/Apache 2.0 | RFFT | ★ 採用 | pocketfft の実質後継 |
| **speexdsp resampler** (C) | BSD | polyphase sinc interpolation | 参考（Rust 移植） | soxr (LGPL) 排除 |
| **speexdsp AEC** (C) | BSD | Acoustic Echo Cancellation | ★ v1.5 採用予定 (Rust port) | WebRTC AEC3 と選択 |
| **WebRTC AEC3** | BSD | AEC | ★ 候補 (Rust port 検討) | speexdsp と選択 |
| **RNNoise** | BSD | Noise Suppression | ★ v0.5 採用予定 | |
| **DeepFilterNet3** (Rust) | MIT | Noise Suppression | ★ v0.5 採用予定 | Rikorose 公式 |
| **GTCRN** | Apache 2.0 | Noise Suppression | ★ v1.0 検討 | |
| **AudioSeal** (Meta) | MIT | Watermark | ★ 推奨デフォルト | |
| **SynthID audio** (Google DeepMind) | Google 個別契約要 | Watermark | 検討中 | 代替: SilentCipher / WaveGuard (OSS) |
| **C2PA (c2pa-rs)** | Apache 2.0 | Content provenance manifest | ★ v0.5 採用 | Adobe |
| **libsamplerate** | BSD | resample | 検討中 | speexdsp と比較 |
| **libsoxr** | LGPL | resample | ✕ 排除 | speexdsp で代替 |
| **rubberband** | GPL | pitch shift / time stretch | ✕ 排除 | 自前実装 or 除外 |
| **libespeak-ng** | GPL-3.0-or-later | G2P | ✕ 排除 | piper-plus 独自 G2P、または misaki / IPA 辞書 |
| **OpenFST** | Apache 2.0 | WFST decoder | ★ v1.0 検討 (Rust port) | |
| **kenlm** | LGPL | n-gram LM | ✕ 検討中止 | 独自 Rust 実装 or `lm-rs` |
| **librosa** (Python 参考) | ISC | Mel filter bank 参考 | 参考のみ | Slaney/HTK 両対応の Rust 実装を独自 |
| **torchaudio** (Python 参考) | BSD | 参考 | 参考のみ | |

---

## 5. G2P (Grapheme-to-Phoneme) 戦略 — レビュアー C/D 指摘 #5 対応

**eSpeak-NG (GPL-3.0) を Vokra core から完全排除**する。以下の戦略で対応:

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

2. **開発者側 install モデル**:
   - `dlopen("libcuda.so")` / `LoadLibrary("nvcuda.dll")` で system install CUDA を実行時検出
   - 検出失敗時は CPU/Vulkan にフォールバック
   - `cudarc` は Rust binding として MIT/Apache 2.0、CUDA runtime そのものは NVIDIA proprietary

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
