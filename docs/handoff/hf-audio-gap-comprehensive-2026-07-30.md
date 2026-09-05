# HuggingFace 音声・音楽 AI 総合 gap 調査 (2026-07-30 拡張版)

> **2026-08-30 legal supersession boundary:** This is a historical research
> inventory. Current legal analysis is `docs/legal-compliance.md`: automatic
> watermark/C2PA integration is **Deferred**, the NO FAKES Act is not
> established/enacted, Apple Guideline 5.5 is **Mobile Device Management
> (MDM)**, and California SB 942 is operative. The legal conclusions and
> “Article 50対応”/“NO FAKES” wording below are design-era context only and
> must not be used as current compliance guidance; applicable role, use,
> jurisdiction, disclosure/marking, consent, and rights require owner/deployer
> review.

## 1. 前文

**依頼者 2026-07-30 指示**: 「音楽関連も対応する。前回の hf-audio-gap-2026-07-30.md より広い調査範囲を再度洗い直せ」= [[project-scope-expansion-2026-07-30]] anchor に基づく scope 拡張版。

**前回 (hf-audio-gap-2026-07-30.md) との差分**:

- **音楽生成 (Music-Gen) を in-scope 化**: ACE-Step / MusicGen / Stable Audio / Magenta / Riffusion / Mustango / MOSS-SoundEffect 等を候補追加。前回 doc の「音声以外 out-of-scope (2026-07-22 依頼者 [[project-deep-not-wide-goal]])」から本 doc は「音楽も含める」に更新。
- **音楽理解 (Music-Understanding) を追加**: CLAP / AST / MERT / MuQ / Dasheng / PANNs / YAMNet 等の music-embedding / music-tagging を独立カテゴリ化。
- **カテゴリ別 wider sweep**: 前回 top-150 一律だったところを、per-category (ASR/TTS/VAD/Music-Gen/Music-Understanding/Enhancement/Codec/Audio-LLM/S2S/Classification) 独立 top-150 に拡張。6 agent parallel search で raw 713 unique hits → 455 candidates → 本 doc 記載 289 unique models (dedup 済)。
- **fine-tune family の代表化**: jonatasgrosman/wav2vec2-XLSR-53 15+ 言語 fine-tune や facebook/mms-tts-*** 1000+ 言語族は代表 1 行 + 展開注記で集約。community mirror (mlx-community/*, onnx-community/*, aufklarer/*, ilintar/*) は代表 1 行に集約。
- **T4 tier precedent 適用済**: 2026-07-28 に X-Codec-2 で確立した Research-only tier (cc-by-nc-4.0 + `--allow-noncommercial` 明示 + LicenseClass::NonCommercial + fetch_license SPDX 追加) を tier3 全数に拡張適用可能と判定。

## 2. 総括サマリ

### 2.1 Tier 別合計 (dedup 後、289 unique)

| Tier | 数 | 説明 |
|---|---|---|
| **TIER 1** permissive (apache-2.0 / MIT / BSD / CC0) | **175** | 商用配布制約なし、そのまま publish 可 |
| **TIER 2** cc-by (attribution 要) | **17** | NOTICE §7 attribution、Mimi/Kyutai 系 handling 流用 |
| **TIER 3** NC / NC-SA / Research-only | **60** | X-Codec-2 T4 precedent 経路 (`--allow-noncommercial`) |
| **TIER 4** other / gated / RAIL / bespoke | **30** | Owner ADR 必須 (Coqui CPML, Qwen license, LFM Open, Stability Community, NVIDIA, Llama-3.2, OpenRAIL, AGPL 等) |
| **TIER 5** voice-clone (別リポ管轄) | **7** | ELVIS Act / NO FAKES → `vokra-voiceclone-experimental` へ隔離 |

### 2.2 カテゴリ別合計 (primary category assignment)

| Category | 数 | 特筆 |
|---|---|---|
| ASR | 50 | wav2vec2/HuBERT/WavLM family + Qwen3-ASR + Cohere-Transcribe + Granite + Moonshine + Whisper missing sizes |
| TTS | 60 | MOSS-TTS + Orpheus + Parler + MeloTTS + Zonos v2 variants + Kokoro voice packs + Indic families |
| Audio-LLM / S2S | 42 | Qwen2-Audio + Qwen2.5-Omni + MiniCPM-o + VITA + LFM2-Audio + Higgs v3 + Moshi/Moshika/Moshiko variants + Hibiki |
| VAD / Diarization / Alignment | 28 | pyannote pipeline layer + smart-turn + Namo + FireRedVAD + sortformer + diarizen + MMS forced-aligner + wav2vec2 phoneme |
| Music-Gen | 24 | ACE-Step family (MIT flagship) + MusicGen family + Stable Audio 3 + Magenta Realtime + Riffusion + Mustango + MOSS-SoundEffect + Jukebox |
| Music-Understanding | 22 | CLAP (5) + AST + MERT + MuQ + Dasheng + PANNs + YAMNet + Basic Pitch + Stanford CRFM Anticipation + MIDI-LLM |
| Enhancement / Separation | 30 | MP-SENet + TIGER + SepFormer + MossFormer2 + Miipher-2 + DeepVQE-AEC + Wave-U-Net + VoiceRestore + LavaSR + HTDemucs + BS-Roformer + UVR-MDX |
| Codec / Vocoder | 20 | NeuCodec (+distill) + FocalCodec family + MioCodec + Ming tokenizer + Voila + Dasheng + TaDiCodec + LinaCodec + jhcodec + BigVGAN v2 |
| Speaker / Emotion / Classification | 13 | WeSpeaker ReDimNet2 + LID (VoxLingua107, MMS-LID) + SER family (SUPERB, IEMOCAP, XLSR, audeering) + gender + accent + deepfake detection + AASIST3 |

### 2.3 特に高需要 (dl > 500k) TOP 30

| # | dl | HF ID | Tier | Category |
|---|---|---|---|---|
| 1 | 9,259,986 | coqui/XTTS-v2 | 4 | TTS |
| 2 | 8,853,435 | pyannote/speaker-diarization-3.1 | 1 | Diarization |
| 3 | 8,119,864 | laion/clap-htsat-fused | 1 | Music-Understanding |
| 4 | 7,236,790 | pyannote/wespeaker-voxceleb-resnet34-LM | 2 | Speaker |
| 5 | 3,457,704 | pyannote/voice-activity-detection | 1 | VAD |
| 6 | 3,086,453 | pyannote/segmentation | 1 | VAD |
| 7 | 2,439,246 | MahmoudAshraf/mms-300m-1130-forced-aligner | 3 | Alignment |
| 8 | 2,386,590 | Qwen/Qwen3-ASR-0.6B | 1 | ASR |
| 9 | 1,965,146 | facebook/musicgen-medium | 3 | Music-Gen |
| 10 | 1,875,500 | openai/whisper-tiny | 1 | ASR |
| 11 | 1,872,508 | Qwen/Qwen3-ASR-1.7B | 1 | ASR |
| 12 | 1,736,690 | facebook/wav2vec2-base-960h | 1 | ASR |
| 13 | 1,719,977 | Qwen/Qwen2.5-Omni-3B | 4 | Audio-LLM |
| 14 | 1,702,440 | jonatasgrosman/wav2vec2-large-xlsr-53-japanese | 1 | ASR |
| 15 | 1,664,488 | nvidia/bigvgan_v2_22khz_80band_256x | 1 | Vocoder |
| 16 | 1,611,283 | facebook/w2v-bert-2.0 | 1 | ASR |
| 17 | 1,591,138 | indonesian-nlp/wav2vec2-indonesian-javanese-sundanese | 1 | ASR |
| 18 | 1,498,165 | nvidia/parakeet-ctc-1.1b | 2 | ASR |
| 19 | 1,135,307 | CohereLabs/cohere-transcribe-03-2026 | 1 | ASR |
| 20 | 1,051,034 | microsoft/wavlm-base-plus | 4 | Speaker/ASR base |
| 21 | 970,489 | airesearch/wav2vec2-large-xlsr-53-th | 2 | ASR |
| 22 | 960,500 | nvidia/nemotron-3.5-asr-streaming-0.6b | 4 | ASR |
| 23 | 958,373 | nvidia/bigvgan_v2_44khz_128band_512x | 1 | Vocoder |
| 24 | 950,367 | KBLab/wav2vec2-large-voxrex-swedish | 1 | ASR |
| 25 | 910,143 | alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech | 1 | Classification |
| 26 | 887,707 | facebook/hubert-base-ls960 | 1 | ASR base |
| 27 | 804,704 | kresnik/wav2vec2-large-xlsr-korean | 1 | ASR |
| 28 | 768,049 | pyannote/embedding | 1 | Speaker |
| 29 | 713,736 | Qwen/Qwen2-Audio-7B-Instruct | 1 | Audio-LLM |
| 30 | 695,510 | microsoft/VibeVoice-ASR | 1 | ASR |

## 3. TIER 1 (permissive、最優先)

### 3.1 ASR

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | Qwen/Qwen3-ASR-0.6B | 2.39M | Qwen3 encoder-decoder ASR 11 lang、Voxtral 直接競合 |
| ★★★ | Qwen/Qwen3-ASR-1.7B | 1.87M | Qwen3-ASR larger variant |
| ★★★ | openai/whisper-tiny | 1.88M | Whisper family gap 埋め、edge IoT tier1/2 |
| ★★★ | facebook/wav2vec2-base-960h | 1.74M | wav2vec2 CTC canonical baseline、family 全体 parent |
| ★★★ | jonatasgrosman/wav2vec2-large-xlsr-53-japanese | 1.7M | XLSR-53 JA (family 15+ 言語 fine-tune = ru 3.65M/pt 4M/pl 2.66M/nl 1.87M 等) |
| ★★★ | facebook/w2v-bert-2.0 | 1.61M | Seamless-M4T speech encoder base、143 langs |
| ★★★ | indonesian-nlp/wav2vec2-indonesian-javanese-sundanese | 1.59M | id/jv/su 3 言語 1 モデル、東南アジア市場 |
| ★★★ | CohereLabs/cohere-transcribe-03-2026 | 1.14M | Cohere Labs 新 ASR (2026-03)、enterprise 100+ langs |
| ★★★ | KBLab/wav2vec2-large-voxrex-swedish | 950k | Swedish VoxRex (cc0-1.0 = public domain) |
| ★★★ | facebook/hubert-base-ls960 | 887k | HuBERT base、RVC/VC feature extractor foundation |
| ★★★ | kresnik/wav2vec2-large-xlsr-korean | 805k | 韓国語 XLSR、ko voice gap 解消 |
| ★★★ | microsoft/VibeVoice-ASR | 695k | Microsoft VibeVoice ASR variant、conversation focused |
| ★★★ | NbAiLab/nb-wav2vec2-1b-nynorsk | 586k | Norwegian Nynorsk (兄弟 bokmaal-v2 = 503k) |
| ★★★ | microsoft/Phi-4-multimodal-instruct | 554k | Phi-4 Speech adapter (ADR 要判断: multimodal 部分 in-scope 判定) |
| ★★★ | ibm-granite/granite-speech-4.1-2b | 544k | IBM Granite Speech 4.1 (Q-former + Granite LLM、enterprise ASR) |
| ★★★ | ibm-granite/granite-speech-3.3-2b | 525k | Granite Speech 3.3 (前世代 backbone) |
| ★★★ | microsoft/wavlm-large | 524k | WavLM large (license 未宣言 = owner adjudication) |
| ★★ | facebook/wav2vec2-large-960h-lv60-self | 628k | wav2vec2-large + LibriVox 60k + self-training |
| ★★ | ibm-granite/granite-speech-4.1-2b-plus | 129k | Granite 4.1 enhanced (instruction following) |
| ★★ | ai-sage/GigaAM-v3 | 351k | Sber GigaAM v3 (RU SoTA)、Voxtral RU 上回る |
| ★★ | facebook/wav2vec2-large-xlsr-53 | 397k | XLSR-53 multilingual pretrained base (53 langs) |
| ★★ | pyannote/voice-activity-detection | 3.46M | pyannote VAD pipeline (gated=auto、MIT)、Silero と別実装 |
| ★★ | UsefulSensors/moonshine-tiny | 200k | Moonshine tiny (Whisper 5-15x 高速、Pi 5 RTF<0.05) |
| ★★ | openai/whisper-tiny.en | 178k | Whisper tiny EN-only |
| ★★ | facebook/wav2vec2-conformer-rope-large-960h-ft | 180k | wav2vec2-Conformer + RoPE |
| ★★ | zai-org/GLM-ASR-Nano-2512 | 166k | Zhipu GLM-ASR Nano (GLM-4 base、ZH SoTA) |
| ★★ | facebook/hubert-large-ls960-ft | 142k | HuBERT-large + LibriSpeech CTC |
| ★★ | MediaTek-Research/Breeze-ASR-25 | 134k | Traditional Chinese (台湾繁体) |
| ★★ | primeline/whisper-large-v3-turbo-german | 115k | Whisper-large-v3-turbo 独語 fine-tune |
| ★★ | tarteel-ai/whisper-base-ar-quran | 284k | Whisper-base Arabic Quran 特化 |
| ★★ | facebook/wav2vec2-xlsr-53-espeak-cv-ft | 437k | Phoneme recognition (eSpeak IPA)、TTS 学習 alignment |
| ★★ | facebook/wav2vec2-lv-60-espeak-cv-ft | 314k | wav2vec2-LV60 phoneme CTC (eSpeak IPA vocab) |
| ★★ | UsefulSensors/moonshine-base | 17k | Moonshine base (Pi 5 real-time、Whisper-small 相当) |
| ★★ | UsefulSensors/moonshine-streaming-tiny | 10k | Moonshine tiny streaming (real-time UI) |
| ★★ | Qwen/Qwen3-ForcedAligner-0.6B | 421k | Qwen3 base forced aligner (Charsiu 上位互換) |
| ★★ | Qwen/Qwen3-ASR-0.6B-hf | 144k | Qwen3-ASR 0.6B transformers-native port |
| ★ | facebook/hubert-xlarge-ls960-ft | 8k | HuBERT-xlarge (SoTA CTC 研究用途) |
| ★ | alvanlii/wav2vec2-BERT-cantonese | 327k | Cantonese wav2vec2-BERT CTC |
| ★ | TencentGameMate/chinese-hubert-large | 10k | ZH HuBERT (RVC/So-VITS ZH pipeline) |
| ★ | TencentGameMate/chinese-hubert-base | 6k | ZH HuBERT base |
| ★ | yky-h/japanese-hubert-large | 11k | JA HuBERT (prj-beatrice VC の base) |
| ★ | prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 | 2k | JA phoneme CTC (Beatrice VC framework) |
| ★ | distil-whisper/distil-large-v3.5 | 4k | distil-whisper v3.5 (v3 精度回復 + 高速化) |
| ★ | distil-whisper/distil-medium.en | 6k | distil-whisper medium EN |
| ★ | distil-whisper/distil-small.en | 8k | distil-whisper small EN |
| ★ | distil-whisper/distil-large-v2 | 6k | distil-whisper v2 (516 likes、後方互換) |

### 3.2 TTS

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | OpenMOSS-Team/MOSS-TTS | 375k | MOSS-TTS base (Fudan、EN/ZH、AR LM style TTS) family total 700k+ |
| ★★★ | ai4bharat/indic-parler-tts | 376k | Parler-TTS Indic 22 言語 |
| ★★★ | pnnbao-ump/VieNeu-TTS-v3-Turbo | 347k | Vietnamese TTS (#16 by dl in TTS) |
| ★★★ | kenpath/svara-tts-v1 | 222k | Indic multilingual TTS |
| ★★★ | myshell-ai/MeloTTS-English | 152k | MyShell MeloTTS EN (VITS-based multilingual family) |
| ★★★ | suno/bark-small | 126k | Bark small edge variant |
| ★★★ | OpenMOSS-Team/MOSS-TTS-v1.5 | 111k | MOSS-TTS v1.5 improved |
| ★★★ | microsoft/speecht5_tts | 64k | Microsoft SpeechT5 TTS (837 likes) |
| ★★★ | myshell-ai/MeloTTS-Korean | 71k | 韓国語 TTS gap 埋め (permissive KO TTS 唯一) |
| ★★★ | parler-tts/parler-tts-mini-multilingual-v1.1 | 70k | Parler multilingual 8 EU 言語 |
| ★★★ | OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 | 58k | MOSS on-device optimized |
| ★★★ | OpenMOSS-Team/MOSS-TTS-Nano-100M | 57k | MOSS-TTS Nano 100M (edge、Kokoro 相当サイズ) |
| ★★ | myshell-ai/MeloTTS-Chinese | 49k | MeloTTS ZH |
| ★★ | myshell-ai/MeloTTS-Japanese | 38k | MeloTTS JA (SBV2/Irodori と並ぶ MIT option) |
| ★★ | contextboxai/Kokoro-Vietnamese | 36k | Community Kokoro Vietnamese fine-tune |
| ★★ | OpenMOSS-Team/MOSS-TTS-Realtime | 30k | Streaming variant (Wyoming/vokra-server 適合) |
| ★★ | canopylabs/orpheus-3b-0.1-ft | 23k | Orpheus 3B LLM-based zero-shot TTS + emotion (auto-gated) |
| ★★ | canopylabs/3b-fr-ft-research_release | 30k | Orpheus 3B French fine-tune |
| ★★ | canopylabs/3b-de-ft-research_release | 24k | Orpheus 3B German fine-tune |
| ★★ | ai4bharat/IndicF5 | 21k | F5-TTS Indic 11 langs fine-tune |
| ★★ | Marvis-AI/marvis-tts-250m-v0.1 | 2k (v0.2: 405) | Marvis compact voice cloning 250M |
| ★★ | Marvis-AI/marvis-tts-100m-v0.2 | 406 | Marvis 100M (Cortex-A55 tier2 IoT) |
| ★★ | kakao-enterprise/vits-ljs | 20k | Kakao VITS LJSpeech (educational reference) |
| ★★ | parler-tts/parler-tts-large-v1 | 11k | Parler-TTS Large、natural-language voice prompt |
| ★★ | parler-tts/parler-tts-mini-v1 | 11k | Parler-TTS Mini |
| ★★ | OpenMOSS-Team/MOSS-VoiceGenerator | 11k | MOSS vocoder companion |
| ★★ | vibevoice/VibeVoice-1.5B | 11k | VibeVoice 1.5B community mirror (Rejected policy 継承) |
| ★★ | vibevoice/VibeVoice-7B | 14k | VibeVoice 7B mirror (同じ Rejected) |
| ★★ | nari-labs/Dia-1.6B-0626 | 10k | Dia v1 refresh (既存 Dia loader) |
| ★★ | neuphonic/neutts-air | 9k | NeuTTS Air realtime cloning (879 likes MIT flagship) |
| ★★ | hexgrad/Kokoro-82M-v1.1-zh | 9k | Kokoro v1.1 Chinese variant (voice pack) |
| ★★ | YatharthS/LuxTTS | 8k | Luxembourgish TTS (ISO 639-1 lb) |
| ★★ | OpenMOSS-Team/MOSS-TTSD-v1.0 | 7k | MOSS Dialogue variant (multi-speaker conversation) |
| ★★ | SPRINGLab/Indic-Mio | 7k | SPRINGLab Indic 追加 option |
| ★★ | inclusionAI/Ming-omni-tts-0.5B | 6k | Ming-Omni TTS 0.5B (Ant Group ZH/EN) |
| ★★ | dots-studio/dots.tts-mf | 6k | dots.tts multi-feature |
| ★★ | nari-labs/Dia2-2B | 5k | Dia 2 (2B、新世代) |
| ★★ | maya-research/maya1 | 5k | Maya Research natural-language voice design (889 likes) |
| ★★ | maya-research/Veena | 4k | Maya Veena Hindi TTS |
| ★★ | prathoshap/vagdhenu | 4k | Vagdhenu Sanskrit/Indic |
| ★★ | ekwek/Soprano-1.1-80M | 4k | Soprano 80M (213 likes、operatic/singing) |
| ★★ | ByteDance/MegaTTS3 | 310 | MegaTTS 3 zero-shot voice cloning (418 likes、latent diffusion) |
| ★★ | Zyphra/Zonos-v0.1-hybrid | 1k | Zonos Mamba-Transformer hybrid (1106 likes) |
| ★★ | metavoiceio/metavoice-1B-v0.1 | 172 | MetaVoice 1B (789 likes、zero-shot + emotional) |
| ★ | hareeshbabu82/TeluguIndicF5 | 3k | Telugu F5-TTS fine-tune |
| ★ | zai-org/GLM-TTS | 999 | GLM-TTS (346 likes、Zhipu MIT) |
| ★ | gpt-omni/mini-omni | 1 | Mini-Omni compact S2S (443 likes、MIT recent) |
| ★ | nineninesix/gepard-1.0 | 9k | Gepard 1.0 (127 likes、fast/tiny) |
| ★ | aoi-ot/VibeVoice-Large | 4k | 別 VibeVoice-Large mirror (Rejected 継承) |

### 3.3 Audio-LLM / S2S

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | Qwen/Qwen2-Audio-7B-Instruct | 714k | Qwen2-Audio 7B (Whisper-large-v3 encoder + Qwen2-7B LM、~15GB) primary Apache-2.0 audio-LLM |
| ★★★ | openbmb/MiniCPM-o-4_5 | 612k | MiniCPM-o 4.5 omnimodal (audio+vision+text 会話) |
| ★★★ | openbmb/MiniCPM-o-2_6 | 365k | MiniCPM-o 2.6 (8B、SigLIP + Whisper + Qwen2.5) |
| ★★ | Qwen/Qwen2-Audio-7B | 44k | Qwen2-Audio 7B base (非 instruct) |
| ★★ | bosonai/higgs-audio-v3-stt | 7k | Higgs Audio v3 STT (Boson AI open) |
| ★★ | tsinghua-ee/video-SALMONN-2_plus_3B | 120 | SALMONN-2 3B (audio-visual LLM smaller) |
| ★★ | tsinghua-ee/video-SALMONN-2 | 22 | SALMONN-2 (audio+video multimodal Q&A) |
| ★★ | bosonai/higgs-audio-v3-8b-stt-v2 | 935 | Higgs v3 8B STT v2 |
| ★★ | HeartMuLa/HeartMuLa-oss-3B | 805 | HeartMuLa 3B audio LLM (256 likes) |
| ★★ | HeartMuLa/HeartMuLa-oss-3B-happy-new-year | 2k | HeartMuLa 3B FT variant |
| ★ | VITA-MLLM/VITA-Audio-Boost | 47 | VITA-Audio Boost (streaming low-latency) |
| ★ | VITA-MLLM/VITA-Audio-Plus-Vanilla | 22 | VITA-Audio Plus Vanilla (quality) |
| ★ | VITA-MLLM/Freeze-Omni | 0 | Freeze-Omni (frozen-backbone omnimodal、20 likes) |
| ★ | gpt-omni/mini-omni2 | 63 | Mini-Omni2 (visual+audio+text real-time、286 likes) |
| ★ | YatharthS/LavaSR | 719 | LavaSR audio LLM (Speech+Reasoning、84 likes) |
| ★ | Marvis-AI/marvis-tts-250m-v0.2 | 405 | Marvis streaming TTS 250M v0.2 |

### 3.4 VAD / Diarization / Alignment

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | pyannote/speaker-diarization-3.1 | 8.85M | Full end-to-end diarization pipeline (segmentation + wespeaker embedding + clustering)、Wave 3+4 で model は landed だが pipeline 層は gap |
| ★★★ | pyannote/voice-activity-detection | 3.46M | Dedicated VAD pipeline (WhisperX / Rev.ai / HA Voice で使用) |
| ★★★ | pyannote/segmentation | 3.09M | v2 segmentation (overlap 検出も同時)、Vokra は 3.0 のみ |
| ★★★ | pyannote/embedding | 768k | X-vector speaker embedding (SincNet + TDNN)、pyannote pipeline default |
| ★★ | pyannote/overlapped-speech-detection | 24k | 重複発話検出 (Silero 未対応)、meeting/full-duplex 必須 |
| ★★ | speechbrain/vad-crdnn-libriparty | 17k | SpeechBrain CRDNN VAD (meetings scenarios 適) |
| ★★ | videosdk-live/Namo-Turn-Detector-v1-Multilingual | 6k | Turn-detection 23 langs、Moshi/CSM full-duplex + M4-19 Wyoming に必須 |
| ★★ | pipecat-ai/smart-turn-v2 | 4k | pipecat wav2vec2-BERT turn-detection (<50ms) |
| ★★ | pipecat-ai/smart-turn-v3 | 0 (185 lk) | pipecat v3 latest (weight 到着後 land) |
| ★★ | vitouphy/wav2vec2-xls-r-300m-timit-phoneme | 6k | EN phoneme CTC (Charsiu 補完) |
| ★★ | vitouphy/wav2vec2-xls-r-300m-phoneme | 4k | Multilingual IPA phoneme (eSpeak-NG 依存排除) |
| ★★ | pyannote/speech-separation-ami-1.0 | 6k | PixIT joint separation + diarization (meeting) |
| ★★ | FireRedTeam/FireRedVAD | 1k | Xiaohongshu VAD (ZH/multi 頑健性) |
| ★★ | mago-ai/ultra_diar_streaming_sortformer_8spk_v1 | 705 | 8-speaker streaming diarization (pyannote 4-speaker cap 超) |
| ★★ | TEN-framework/TEN_Turn_Detection | 445 | TEN Agent 全 duplex stack turn-detection (ZH ecosystem) |
| ★★ | videosdk-live/Namo-Turn-Detector-v1-English | 414 | Namo EN 特化 (mobile 用) |
| ★ | RobroKools/vad-bert | 912 | BERT-based VAD (SBV2 と BERT sub-graph 共有可能性 test) |
| ★ | CuriousMonkey7/HumAware-VAD | 718 | 音楽/humming false positive 少ない Silero variant |
| ★ | BUT-FIT/diarizen-meeting-base | 49 | BUT-FIT diarizen MIT (meeting scenarios) |

### 3.5 Music-Gen

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | ACE-Step/Ace-Step1.5 | 60k (808 lk) | MIT text-to-music (vocal 付き song generation)、permissive-license 最強 music-gen |
| ★★ | ACE-Step/acestep-captioner | 27k | ACE-Step audio-to-caption (music description) |
| ★★ | ACE-Step/acestep-5Hz-lm-4B | 22k | ACE-Step 5Hz music-token AR LM (4B、musical bar scale) |
| ★★ | ACE-Step/acestep-v15-base | 14k | ACE-Step base baseline |
| ★★ | ACE-Step/acestep-5Hz-lm-0.6B | 7k | ACE-Step 5Hz LM 0.6B (edge tier) |
| ★★ | ACE-Step/acestep-v15-xl-sft | 6k | ACE-Step XL SFT (large + prompt following) |
| ★★ | ACE-Step/acestep-v15-xl-turbo | 4k | ACE-Step XL turbo (distilled few-step、real-time) |
| ★★ | declare-lab/mustango | 4k | Mustango music-theory aware text-to-music (chord/key/tempo control) |
| ★★ | OpenMOSS-Team/MOSS-SoundEffect-v2.0 | 2k | MOSS text-to-SFX (game/film audio) |
| ★★ | OpenMOSS-Team/MOSS-SoundEffect | 1k | MOSS SFX v1 |
| ★★ | mispeech/Dasheng-AudioGen | 358 | Dasheng general text-to-audio (music + SFX、permissive alt to AudioGen) |
| ★★ | magenta-community/magenta-realtime-2 | 133 | Apache-2.0 community mirror of Magenta Realtime 2 (attribution 回避) |
| ★★ | magenta-community/magenta-realtime-2-small | 176 | Magenta Realtime 2 small variant |
| ★ | HeartMuLa/HeartMuLa-oss-3B-happy-new-year | 2k | HeartMuLa music FT variant |
| ★ | mradermacher/zen-musician-i1-GGUF | 496 | Zen-Musician GGUF (Apache-2.0 small music LLM) |
| ★ | calcuis/ace-gguf | 3k | ACE-Step GGUF pre-converted (parity target) |

### 3.6 Music-Understanding

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | laion/clap-htsat-fused | 8.12M | CLAP HTSAT-fused、text-audio joint embedding、8.1M dl SoTA music-search foundation |
| ★★★ | laion/larger_clap_music_and_speech | 675k | Larger CLAP music+speech dual-domain |
| ★★★ | MIT/ast-finetuned-audioset-10-10-0.4593 | 620k | AST AudioSet-527 (music instrument/genre/sound-event tagging) |
| ★★★ | laion/clap-htsat-unfused | 534k | CLAP HTSAT-unfused (separate projections) |
| ★★ | laion/larger_clap_general | 229k | Larger CLAP general audio (broader corpus) |
| ★★ | laion/larger_clap_music | 34k | Larger CLAP music-only |
| ★★ | MIT/ast-finetuned-audioset-14-14-0.443 | 26k | AST 14x14 patch coarse variant (faster real-time) |
| ★★ | mispeech/ced-base | 9k | CED AudioSet SoTA (mAP 49.6+、Xiaomi ViT distillation) |
| ★★ | mispeech/dasheng-base | 1k | Dasheng audio encoder base (speech+music+environmental) |
| ★★ | mispeech/dasheng-0.6B | 226 | Dasheng 0.6B |
| ★★ | mispeech/dasheng-1.2B | 86 | Dasheng 1.2B |
| ★★ | nicofarr/panns_Cnn14 | 60 | PANNs CNN14 (527-class AudioSet、classic) |
| ★★ | thelou1s/yamnet | 5k | YAMNet MobileNetV1 (521-class edge classifier) |
| ★★ | mispeech/ced-tiny | 7k | CED tiny variant (edge efficient) |
| ★★ | OpenMOSS-Team/MOSS-Music-8B-Instruct | 2k | MOSS-Music 8B Instruct (music-theory QA、genre/key/tempo/mood) |
| ★★ | OpenMOSS-Team/MOSS-Music-8B-Thinking | 216 | MOSS-Music 8B Thinking (CoT music analysis) |
| ★★ | stanford-crfm/music-medium-800k | 10k | Stanford Anticipation symbolic MIDI-token AR LM |
| ★★ | stanford-crfm/music-large-800k | 1k | Stanford Anticipation large |
| ★★ | stanford-crfm/music-small-800k | 4k | Stanford Anticipation small (edge) |
| ★★ | spotify/basic-pitch | 0 (28 lk) | Spotify Basic Pitch audio-to-MIDI (polyphonic pitch)、industry-standard AMT |
| ★★ | schism-audio/basic-pitch-mlx | 34 | Basic Pitch MLX (Apple Silicon parity target) |
| ★★ | dima806/music_genres_classification | 3k | GTZAN-style 10-class music genre |
| ★★ | dima806/musical_instrument_detection | 765 | Instrument classifier (drums/guitar/piano/vocals) |
| ★★ | StanislavKo28/music_moods_classification | 4k | Music mood (happy/sad/energetic/calm) |
| ★★ | MelodyMachine/Deepfake-audio-detection-V2 | 5k | Deepfake / AI-generated music detection (法務・marking review の技術候補。自動的な法令対応は主張しない) |
| ★★ | awsaf49/sonics-spectttra-gamma-5s | 41k | SONICS SpecTTTra Gamma-5s (synthetic music discrimination SoTA) |
| ★★ | awsaf49/sonics-spectttra-beta-5s | 33k | SONICS Beta-5s variant |
| ★★ | awsaf49/sonics-spectttra-alpha-120s | 12k | SONICS Alpha-120s (long-clip full-song detection) |
| ★★ | mudler/ced-gguf | 8k | CED GGUF (AST-alternative edge tagger) |
| ★★ | nicofarr/panns_MobileNetV2 | 8 | PANNs MobileNetV2 (edge classification) |
| ★★ | laion/sound-effect-captioning-whisper | 31 | Whisper sound-effect captioning (Vokra Whisper backbone 流用) |
| ★★ | slseanwu/beats-conformer-bart-audio-captioner | 9 | BEATs + Conformer + BART audio captioning |

### 3.7 Enhancement / Separation

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | JusperLee/TIGER-DnR | 178k | TIGER Dialog-and-Reverb separation、a2a #1 outside vocoders |
| ★★★ | JacobLinCool/MP-SENet-DNS | 59k | MP-SENet magnitude+phase enhancement (DFN3 orthogonal 補完) |
| ★★ | JusperLee/TIGER-speech | 20k | TIGER speech-only variant (arch 共有) |
| ★★ | speechbrain/sepformer-wsj02mix | 3k | SepFormer 2-speaker separation canonical baseline、8+ checkpoint 1 arch |
| ★★ | speechbrain/sepformer-wham16k-enhancement | 3k | SepFormer WHAM 16k enhancement |
| ★★ | speechbrain/sepformer-whamr16k | 2k | SepFormer WHAMR 16k joint dereverb + separation |
| ★★ | speechbrain/metricgan-plus-voicebank | 9k | MetricGAN+ perceptual-metric-optimized denoiser (STOI/PESQ) |
| ★★ | alibabasglab/mossformer2-librimix-2spk | 6k | MossFormer2 speaker separation (WSJ0-2mix SoTA) |
| ★★ | starkdmi/MossFormer2-SE | 1k | MossFormer2 speech enhancement 16 kHz |
| ★★ | LocalAI-io/LocalVQE | 6k | LocalVQE realtime enhancement + AEC pipeline (Moshi/CSM AEC gap 解消) |
| ★★ | richiejp/deepvqe-aec-gguf | 197 | DeepVQE AEC GGUF (Microsoft、Moshi/CSM 必須) |
| ★★ | line-corporation/open-universe | 137 | LINE Universe universal restoration (denoise+dereverb+BWE+declip) |
| ★★ | jadechoghari/VoiceRestore | 42 | VoiceRestore flow-matching restoration (F5-TTS-style、M3-05 flow_sampler 実践) |
| ★★ | YatharthS/LavaSR | 719 | LavaSR speech super-resolution (8→48 kHz、AudioSR alternative) |
| ★★ | YatharthS/FlashSR | 0 (64 lk) | FlashSR fast one-step SR (distilled、MLX port 596 dl) |
| ★★ | wrice/waveunet-vctk-24khz | 120 | Wave-U-Net raw-waveform enhancement (time-domain、no STFT) |
| ★★ | Xiaobin-Rong/unipase | 315 | UniPASE unified pre-processing (denoise+dereverb+AEC 共有) |
| ★★ | Ceva-IP/DPDFNet | 331 | CEVA DSP low-compute denoise (M5-03 Cortex-M55 tier) |
| ★★ | weya-ai/hush | 36 (40 lk) | hush realtime denoise (framework-agnostic) |
| ★★ | mispeech/dasheng-denoiser | 743 | Dasheng-based denoiser (backbone-shared with tokenizer + AudioGen) |
| ★★ | speechbrain/sepformer-dns4-16k-enhancement | 299 | SepFormer DNS4 challenge trained |
| ★★ | speechbrain/sgmse-voicebank | 63 | SGMSE diffusion-based enhancement (flow/ODE sampler on real weights) |
| ★★ | AlayaLab/AudioSep-hive | 99 | AudioSep language-queried universal separation (text-prompt driven) |
| ★★ | rocca/lyra-v2-soundstream | 61 | Google Lyra V2 / SoundStream port (3.2 kbps super-low-bitrate) |
| ★★ | JusperLee/TIGER-speech-tiny | 140 | TIGER speech tiny (arch 共有) |
| ★ | speechbrain/sepformer-wham | 127 | SepFormer WHAM (arch 共有) |
| ★ | speechbrain/sepformer-whamr | 612 | SepFormer WHAMR |
| ★ | speechbrain/sepformer-whamr-enhancement | 242 | SepFormer WHAMR enhancement |
| ★ | speechbrain/sepformer-wham-enhancement | 135 | SepFormer WHAM enhancement |
| ★ | speechbrain/sepformer-libri2mix | 430 | SepFormer LibriMix 2-speaker |
| ★ | speechbrain/sepformer-libri3mix | 283 | SepFormer LibriMix 3-speaker |
| ★ | speechbrain/sepformer-wsj03mix | 298 | SepFormer WSJ0-3mix |
| ★ | cisco-ai/pase | 56 | PASE Cisco AI self-supervised backbone (enhancement/separation heads) |
| ★ | cstr/htdemucs-GGUF | 333 | HTDemucs v4 GGUF (Meta MIT music source separation、4-stem) |
| ★ | HiDolen/Mini-BS-RoFormer-18M | 129 | Mini-BS-Roformer 18M music source separation |
| ★ | mlx-community/mel-roformer-kim-vocal-2-mlx | 164 | Kim Vocal 2 MLX (UVR/MSST community vocal removal 標準) |
| ★ | gyoom-sa/UVR-MDX-LiteRT | 104 | UVR-MDX-Net LiteRT (vocal/instrumental split) |
| ★ | NextFire/tsurumeso-vocal-remover | 68 | tsurumeso JA vocal remover (lightweight) |
| ★ | Intel/demucs-openvino | 0 | Intel Demucs OpenVINO (parity target) |
| ★ | schism-audio/htdemucs-6s-coreml | 30 | HTDemucs 6-stem CoreML (professional production) |

### 3.8 Codec / Vocoder

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | nvidia/bigvgan_v2_22khz_80band_256x | 1.66M | BigVGAN v2 22kHz vocoder (MIT confirmed 2026-07-22)、a2a #1 |
| ★★★ | nvidia/bigvgan_v2_44khz_128band_512x | 958k | BigVGAN v2 44.1 kHz high-fidelity |
| ★★★ | speechbrain/tts-hifigan-libritts-22050Hz | 282k | HiFi-GAN LibriTTS 22050Hz (281k dl standalone vocoder) |
| ★★★ | neuphonic/neucodec | 109k | NeuCodec 50Hz single-codebook speech codec (128 tok/s、HuBERT + HiFi-GAN) |
| ★★ | microsoft/speecht5_hifigan | 46k | SpeechT5 HiFi-GAN vocoder companion |
| ★★ | neuphonic/distill-neucodec | 42k | Distilled NeuCodec (smaller/faster edge) |
| ★★ | nvidia/bigvgan_v2_24khz_100band_256x | 28k | BigVGAN v2 24 kHz (Kokoro/CosyVoice sampling 適合) |
| ★★ | Aratako/MioCodec-25Hz-44.1kHz-v2 | 22k | JA-focused MioCodec (Kokoro/piper-plus JA stack 補完) |
| ★★ | Aratako/MioCodec-25Hz-24kHz | 12k | MioCodec 24 kHz |
| ★★ | ACE-Step/acestep-transcriber | 16k | ACE-Step transcriber (music → symbolic) |
| ★★ | lucadellalib/focalcodec_50hz | 1k | FocalCodec 50Hz single-codebook (family 6 checkpoints 1 converter) |
| ★★ | mispeech/dashengtokenizer | 4k | Dasheng general audio tokenizer (backbone shared) |
| ★★ | amphion/TaDiCodec | 27 | Amphion diffusion-based codec (unique arch) |
| ★★ | jhcodec/jhcodec | 853 | jhcodec RVQ-based (codec breadth) |
| ★★ | inclusionAI/Ming-omni-tts-tokenizer-12Hz | 27 | Ming-Omni 12Hz (最低 rate speech tokenizer、LLM S2S bandwidth 最小化) |
| ★★ | maitrix-org/Voila-Tokenizer | 305 | Voila-Tokenizer (S2S voice tokenizer) |
| ★★ | patriotyk/vocos-mel-hifigan-compat-44100khz | 418 | Vocos 44.1 kHz mel-conditioned (fp16 必須) |
| ★★ | nvidia/bigvgan_v2_22khz_80band_fmax8k_256x | 4k | BigVGAN v2 fmax=8k narrow-band (telephony) |
| ★ | ktvoice/Codec | 142 | ktvoice Codec (codec breadth 補完) |
| ★ | lucadellalib/focalcodec_50hz_2k_causal | 431 | FocalCodec 50Hz 2k causal |
| ★ | lucadellalib/focalcodec_50hz_4k_causal | 325 | FocalCodec 50Hz 4k causal |
| ★ | lucadellalib/focalcodec_25hz | 227 | FocalCodec 25Hz |

### 3.9 Speaker / Emotion / Classification

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech | 910k | Gender recognition (910k dl 最高) |
| ★★★ | xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned | 259k | HuBERT-large RU SER (Dusha) |
| ★★★ | JaesungHuh/voice-gender-classifier | 230k | Voice gender ECAPA-TDNN (MIT permissive、最人気) |
| ★★★ | prithivMLmods/Common-Voice-Gender-Detection | 176k | Gender detection Common Voice multi-ling |
| ★★★ | speechbrain/lang-id-voxlingua107-ecapa | 116k | Spoken LID 107 langs、ECAPA-TDNN (commercial-friendly、MMS-LID alternative) |
| ★★ | microsoft/speecht5_asr | 91k | SpeechT5 ASR head |
| ★★ | superb/wav2vec2-base-superb-er | 70k | SUPERB wav2vec2 emotion (IEMOCAP 4-class) |
| ★★ | speechbrain/emotion-recognition-wav2vec2-IEMOCAP | 55k | SpeechBrain wav2vec2 SER (IEMOCAP) |
| ★★ | superb/hubert-large-superb-er | 19k | HuBERT-Large emotion (larger alternative) |
| ★★ | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition | 14k | wav2vec2-XLSR emotion (RAVDESS 8-class、251 likes 人気) |
| ★★ | speechbrain/spkrec-resnet-voxceleb | 11k | ResNet speaker embedding (ECAPA alternative) |
| ★★ | speechbrain/spkrec-xvect-voxceleb | 10k | x-vector speaker embedding (classic TDNN reference) |
| ★★ | firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3 | 11k | Whisper-large-v3 encoder SER (Vokra Whisper backbone) |
| ★★ | Jzuluaga/accent-id-commonaccent_ecapa | 9k | ECAPA-TDNN accent-ID (British/American/Indian/Aussie) |
| ★★ | superb/wav2vec2-base-superb-ks | 9k | SUPERB wav2vec2 KWS (Google Speech Commands 35-class) |
| ★★ | speechbrain/lang-id-commonlanguage_ecapa | 5k | Common-Language LID (45 langs) |
| ★★ | MIT/ast-finetuned-speech-commands-v2 | 3k | AST KWS Google Speech Commands v2 |
| ★★ | bookbot/distil-wav2vec2-adult-child-cls-37m | 4k | Adult/child voice classifier (age-group) |
| ★★ | dima806/english_accents_classification | 2k | EN accent classifier alternative |
| ★★ | 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes | 1k | WavLM multi-attribute SER (valence/arousal/dominance + categorical) |
| ★★ | superb/wav2vec2-base-superb-sid | 977 | SUPERB speaker ID (VoxCeleb1) |
| ★★ | Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM | 572 | ReDimNet2-B6 SoTA VoxCeleb EER |
| ★★ | Jzuluaga/accent-id-commonaccent_xlsr-en-english | 1k | wav2vec2-XLSR accent-ID EN |
| ★★ | Jzuluaga/accent-id-commonaccent_xlsr-es-spanish | 151 | Spanish accent-ID |
| ★ | Wespeaker/wespeaker-cnceleb-resnet34-LM | 68 | WeSpeaker CN-Celeb (Mandarin speaker verification) |
| ★ | mo-thecreator/Deepfake-audio-detection | 1k | Deepfake detector alternative |
| ★ | Gustking/wav2vec2-large-xlsr-deepfake-audio-classification | 3k | wav2vec2-XLSR deepfake classifier |
| ★ | abhishtagatya/hubert-base-960h-asv19-deepfake | 810 | HuBERT ASVspoof2019 deepfake detector |
| ★ | lab260/Spectra-AASIST3 | 309 | AASIST3 anti-spoofing (Spectra、SOTA ASVspoof) |
| ★ | lab260/Spectra-AASIST | 31 | AASIST v1 (Spectra、classic implementation) |
| ★ | speechbrain/google_speech_command_xvector | 112 | x-vector KWS baseline |

### 3.10 Watermark

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | facebook/audioseal | 12k | AudioSeal proactive watermark（法務・marking review の技術候補。automatic integration は Deferred、EU AI Act/SB 942 の適合は主張しない）。M5-05 T04 ADR ratify で first-class op 化を検討 |

## 4. TIER 2 (cc-by、attribution 要)

### 4.1 ASR / Speaker

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | pyannote/wespeaker-voxceleb-resnet34-LM | 7.24M | WeSpeaker ResNet34 + LM (pyannote 3.1 使用の実 embedding、attribution required) |
| ★★★ | pyannote/speaker-diarization-community-1 | 4.97M | 新 (2025) community-driven diarization pipeline (3.1 後継) |
| ★★★ | nvidia/parakeet-ctc-1.1b | 1.5M | Parakeet-CTC 1.1B (NeMo FastConformer、cc-by-4.0 attribution) |
| ★★ | airesearch/wav2vec2-large-xlsr-53-th | 970k | Thai XLSR-53 (cc-by-sa-4.0 = ShareAlike、LicenseClass::Copyleft 経路) |
| ★★ | nvidia/canary-1b-v2 | 62k | Canary v2 (2025 release、精度+速度改善) |
| ★★ | nvidia/canary-1b-flash | 5k | Canary Flash 高速化 |
| ★★ | nvidia/canary-180m-flash | 2k | Canary 180M Flash (edge/mobile) |
| ★★ | nvidia/diar_streaming_sortformer_4spk-v2 | 44k | NVIDIA Streaming Sortformer 4-speaker real-time diarization |
| ★★ | onnx-community/wespeaker-voxceleb-resnet34-LM | 971 | WeSpeaker ResNet34 VoxCeleb ONNX variant |
| ★★ | espnet/voxcelebs12_rawnet3 | 6k | RawNet3 speaker embedding (raw-waveform alternative to ECAPA) |

### 4.2 TTS / Enhancement / Music

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | facebook/audiobox-aesthetics | 321k | Meta audio-quality-aesthetics scorer (music/audio quality)、UTMOS22 補完 |
| ★★ | kyutai/tts-1.6b-en_fr | 52k | Kyutai TTS 1.6B en+fr (378 likes、Kyutai 同 team = Moshi/Mimi) |
| ★★ | kyutai/tts-0.75b-en-public | 5k | Kyutai TTS 0.75B EN public |
| ★★ | google/magenta-realtime-2 | 7k | Google Magenta Realtime 2 (237 likes、canonical CC-BY-4.0 real-time music) |
| ★★ | google/magenta-realtime | 35 | Google Magenta RT-1 (555 likes、prompting) |
| ★★ | wyz/tfgridnet_for_urgent24 | 102 | TF-GridNet URGENT 2024 enhancement (SoTA time-frequency) |
| ★★ | mpariente/DPRNNTasNet-ks2_WHAM_sepclean | 5k | DPRNN-TasNet WHAM 2-speaker separation (academic standard) |
| ★★ | JorisCos/ConvTasNet_Libri2Mix_sepclean_16k | 3k | ConvTasNet Libri2Mix (lightweight separation) |
| ★★ | YatharthS/LinaCodec | 31 | LinaCodec speech codec (LavaSR family companion) |
| ★★ | laion/vocalburst-captioning-whisper | 436 | Whisper vocal-burst captioning (laughter/sighs) |
| ★ | syvai/plapre-nano | 37k | PlaPre Nano tiny TTS (gated=auto) |
| ★ | gaunernst/vit_base_patch16_1024_128.audiomae_as2m | 858 | AudioMAE ViT-Base AudioSet-2M (timm format) |
| ★ | cstr/tabcnn-GGUF | 428 | TabCNN audio-to-guitar-tab notation (Basic Pitch 補完) |
| ★ | mradermacher/CiSiMi-GGUF | 468 | CiSiMi cc-by-sa-4.0 text-to-audio LLM (Copyleft variant) |

### 4.3 S2S (Kyutai family)

| ★ | HF ID | dl | 用途 |
|---|---|---|---|
| ★★★ | kyutai/moshiko-pytorch-bf16 | 268k | Moshiko male-voice Moshi (highest-dl variant、247 likes) |
| ★★ | kyutai/moshika-pytorch-bf16 | 14k | Moshika female-voice Moshi variant (alternate voice/persona) |
| ★★ | kyutai/moshika-rag-pytorch-bf16 | 2k | Moshika RAG-tuned (grounded conversation) |
| ★★ | kyutai/hibiki-1b-pytorch-bf16 | 6k | Hibiki 1B real-time S2S 翻訳 (EN↔FR、Moshi と別 task) |
| ★★ | kyutai/hibiki-2b-pytorch-bf16 | 448 | Hibiki 2B larger S2S 翻訳 |

## 5. TIER 3 (NC / NC-SA / Research-only、T4 経路)

### 5.1 ASR / Alignment

| ★ | HF ID | dl | 用途 / T4 経路 |
|---|---|---|---|
| ★★★ | MahmoudAshraf/mms-300m-1130-forced-aligner | 2.44M | MMS 1130 lang forced aligner (WhisperX backend)、highest-dl aligner。X-Codec-2 precedent 適用 |
| ★★★ | facebook/mms-1b-all | 268k | MMS 1B ASR 1000+ langs (Meta multilingual)、African/low-resource key |
| ★★ | facebook/mms-300m | 81k | MMS 300M base (aligner の fine-tune 元) |
| ★★ | facebook/seamless-m4t-v2-large | 241k | SeamlessM4T v2 large (100+ langs unified S2T/T2T/T2S/S2S、Voxtral 直接競合) |
| ★★ | facebook/hf-seamless-m4t-medium | 99k | SeamlessM4T Medium (100 langs input / 35 output) |
| ★★ | facebook/hf-seamless-m4t-large | 6k | SeamlessM4T Large |
| ★★ | facebook/seamless-m4t-medium | 0 | Seamless-M4T medium (v1 世代) |
| ★★ | nvidia/canary-1b | 2k | Canary 1B original (NC version、cc-by-4.0 flash/v2 と依頼者 ADR で選択) |

### 5.2 TTS / S2S

| ★ | HF ID | dl | 用途 / T4 経路 |
|---|---|---|---|
| ★★★ | 2Noise/ChatTTS | 2k (1663 lk) | ChatTTS conversational TTS (laugh/emotion tokens、very-high community interest) |
| ★★ | SWivid/E2-TTS | 114k | E2-TTS flow-matching TTS (SWivid F5-TTS authors、Vokra F5 scaffold 隣) |
| ★★ | HKUSTAudio/Llasa-3B | 685 (528 lk) | Llasa 3B HKUST LLaMA-based zero-shot TTS + XCodec2 (Vokra 済) |
| ★★ | mistralai/Voxtral-4B-TTS-2603 | 4k (888 lk) | Voxtral 4B TTS (Voxtral ASR TTS companion、fresh release) |
| ★★ | SparkAudio/Spark-TTS-0.5B | 917 (745 lk) | Spark-TTS BiCodec (Vokra 済)、NC-SA + strictest tier3 |
| ★★ | fishaudio/s1-mini | 3k (675 lk) | Fish-Speech s1-mini (auto-gated) |
| ★★ | facebook/mms-tts-eng | 109k | MMS-TTS 1000+ langs anchor (family 数百 = mms-tts-{hin,ara,khm,kor,tam,tel,vie,urd,fra 等}) |
| ★★ | facebook/mms-tts-ara | 11k | MMS Arabic TTS (family 例) |
| ★★ | Misha24-10/F5-TTS_RUSSIAN | 87k | F5-TTS Russian community fine-tune (210 likes、Vokra F5 scaffold voice-pack) |
| ★ | khmerttsopensource/khmer-tts | 1k | Khmer TTS (km 言語 gap) |
| ★ | KRAFTON/Raon-Speech-9B | 5k | KRAFTON Raon-Speech 9B (Korean-first) |
| ★ | KRAFTON/Raon-SpeechChat-9B | 846 | Raon-SpeechChat 9B (chat-tuned) |
| ★ | KRAFTON/Raon-OpenTTS-1B | 153 | Raon-OpenTTS 1B (Korean) |
| ★ | Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1 | 7 (15 lk) | Miipher-2 universal speech restoration (Google Miipher reproduction) |
| ★ | kyutai/moshika-rl-seamless | 172 | Moshika RL-tuned + Seamless-conditioned |

### 5.3 Music-Gen (最重要 T4 slot)

| ★ | HF ID | dl | 用途 / T4 経路 |
|---|---|---|---|
| ★★★ | facebook/musicgen-medium | 1.97M | MusicGen medium (Meta text-to-music 参照、highest-dl music family)。X-Codec-2 T4 pattern |
| ★★★ | facebook/musicgen-small | 357k (500 lk) | MusicGen small (500 likes 最人気 variant) |
| ★★ | facebook/musicgen-large | 73k (535 lk) | MusicGen large (3.3B、535 likes flagship) |
| ★★ | facebook/musicgen-melody | 13k (255 lk) | MusicGen with melody conditioning (chromagram-guided) |
| ★★ | facebook/musicgen-stereo-large | 4k (98 lk) | MusicGen stereo large (dual-EnCodec channels、rare stereo music-gen) |
| ★★ | facebook/musicgen-stereo-medium | 2k | MusicGen stereo medium |
| ★★ | facebook/musicgen-stereo-small | 2k | MusicGen stereo small (real-time stereo) |
| ★★ | facebook/musicgen-melody-large | 325 | MusicGen melody-large highest quality |
| ★★ | facebook/audiogen-medium | 28k (147 lk) | AudioGen medium (text-to-audio SFX generation、MusicGen companion) |
| ★★ | facebook/audio-magnet-medium | 63 (35 lk) | MAGNeT non-AR text-to-audio (fast parallel、novel arch) |
| ★★ | cvssp/audioldm2 | 20k (70 lk) | AudioLDM 2 latent diffusion text-to-audio (music+sound+speech) |
| ★★ | cvssp/audioldm2-music | 820 | AudioLDM 2 music-specialized |
| ★★ | cvssp/audioldm-l-full | 795 | AudioLDM v1 large-full (early text-to-audio LDM reference) |
| ★★ | declare-lab/tango | 30 (46 lk) | Tango text-to-audio LDM (TangoFlux/Mustango 予備) |
| ★★ | auffusion/auffusion | 0 (12 lk) | Auffusion spectrogram-based diffusion (SD backbone repurposed) |
| ★★ | HKUSTAudio/AudioX-MAF | 509 | AudioX-MAF multimodal (Vokra X-Codec-2 team 続) |
| ★★ | HiDolen/Mini-BS-RoFormer-V2-46.8M | 464 | Mini-BS-Roformer V2 (larger vocal-separation variant) |
| ★★ | MansfieldPlumbing/Demucs_v4_TRT | 338 | Demucs v4 TensorRT (GPU-accelerated inference reference) |
| ★★ | aufklarer/FlashSR-MLX-4bit | 596 | FlashSR MLX 4-bit (Apple Silicon parity target) |
| ★★ | cstr/btc-chords-GGUF | 673 | BTC bi-directional Transformer chord recognition |
| ★★ | PearlLeeStudio/TheArtist-MusicTransformer-pop-baseline | 477 | Music Transformer pop-baseline (symbolic MIDI generation) |
| ★★ | NandemoGHS/Anime-XCodec2-44.1kHz-v2 | 744 | Anime-XCodec2 anime-voice 44.1 kHz (Vokra X-Codec-2 派生) |
| ★★ | shuoyang-zheng/jaspers-rave-models | 93 | RAVE latent-audio manipulation (IRCAM VAE audio effects) |

### 5.4 Music-Understanding

| ★ | HF ID | dl | 用途 / T4 経路 |
|---|---|---|---|
| ★★★ | m-a-p/MERT-v1-330M | 581k | MERT v1 330M music-understanding embedding (BERT for music、reference model) |
| ★★★ | OpenMuQ/MuQ-large-msd-iter | 304k | MuQ Music Understanding via Quantization (MERT alternative) |
| ★★ | m-a-p/MERT-v1-95M | 84k | MERT v1 95M smaller (edge tier) |
| ★★ | OpenMuQ/MuQ-MuLan-large | 48k | MuQ-MuLan music-text CLAP-alternative (music-text 対応) |
| ★★ | m-a-p/music2vec-v1 | 2k | Music2vec v1 (MERT predecessor) |
| ★★ | mtg-upf/discogs-maest-30s-pw-129e | 3k | MAEST music-tagger Discogs (genre+subgenre+mood+instruments、broadest music tagger) |

### 5.5 Classification / LID

| ★ | HF ID | dl | 用途 / T4 経路 |
|---|---|---|---|
| ★★★ | facebook/mms-lid-126 | 253k | Meta MMS spoken LID 126 langs (SpeechBrain LID 補完) |
| ★★ | facebook/mms-lid-256 | 249k | MMS LID 256 langs |
| ★★ | facebook/mms-lid-1024 | 92k | MMS LID 1024 langs |
| ★★ | facebook/mms-lid-4017 | 49k | MMS LID 4017 langs (largest) |
| ★★★ | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | 654k | audeering SER (MSP-Podcast dimensional VAD、highest-dl SER) |
| ★★★ | audeering/wav2vec2-large-robust-24-ft-age-gender | 268k | audeering age + binary gender regression |
| ★★★ | onecxi/open-vakgyata | 271k | Open Vakgyata Indic-language SER + understanding (271k dl covers Indic gap) |
| ★★ | audeering/wav2small | 120 | audeering wav2small edge SER (distilled age/gender/emotion) |
| ★ | lab260/AASIST3 | 146 | AASIST3 anti-spoofing (canonical NC checkpoint、Spectra-AASIST3 preferred) |
| ★ | davidscripka/openwakeword | 0 (5 lk) | openWakeWord OSS wake-word framework (Home Assistant で popular、NC-SA) |
| ★ | MU-NLPC/whisper-small-audio-captioning | 2k | Whisper-small audio captioning (Vokra Whisper backbone) |
| ★ | MU-NLPC/whisper-large-v2-audio-captioning | 76 (12 lk) | Whisper-large-v2 audio captioning |

### 5.6 Diarization

| ★ | HF ID | dl | 用途 / T4 経路 |
|---|---|---|---|
| ★★★ | nvidia/diar_sortformer_4spk-v1 | 11k (150 lk) | Original NVIDIA Sortformer offline diarization (token-classification style) |
| ★★ | BUT-FIT/diarizen-wavlm-large-s80-md-v2 | 3k | BUT DiariZen v2 WavLM-Large EEND 80-speaker (DIHARD/CALLHOME SoTA) |
| ★★ | pyannote/brouhaha | 19k | Joint VAD + SNR + C50 estimation (openrail、triple-task unique) |

## 6. TIER 4 (other / gated / RAIL / 要精査)

各行に **owner ADR 確認事項** を記載。

### 6.1 ASR / Audio-LLM

| ★ | HF ID | dl | License | Owner ADR で確認 |
|---|---|---|---|---|
| ★★★ | Qwen/Qwen2.5-Omni-3B | 1.72M | other (Qwen license) | Tongyi Qianwen 商用条件 (<100M MAU) 適用可否 |
| ★★★ | Qwen/Qwen2.5-Omni-7B | 415k (1922 lk) | other (Qwen license) | 同上、多 language + real-time TTS-like |
| ★★★ | microsoft/wavlm-base-plus | 1.05M | no-declared-license | Microsoft research 通常 MIT だが cardData 未宣言 = 要問合せ / 事前 fail-closed |
| ★★★ | nvidia/nemotron-3.5-asr-streaming-0.6b | 960k (963 lk) | other (NVIDIA Open Model License) | 研究/商用 別 tier、Vokra サーバ用途 (M3-15) と完璧マッチ |
| ★★★ | microsoft/wavlm-large | 524k | no-declared-license | 同 wavlm-base-plus |
| ★★ | Revai/reverb-diarization-v1 | 400k | other (Rev.ai custom) | 商用 STT ベンダ Rev.ai on-prem 移行需要 |
| ★★ | google/medasr | 249k | other (Google 独自) + gated=auto | 医療 vertical (CPU/Vulkan-only SKU 適合) |
| ★★ | slplab/wav2vec2-large-robust-L2-english-phoneme-recognition | 227k | UNKNOWN | cardData license 欠、slplab publication + repo LICENSE 要参照 |
| ★★ | nvidia/nemotron-speech-streaming-en-0.6b | 153k (597 lk) | other | 同 nemotron-3.5-asr |
| ★★ | microsoft/wavlm-base | 38k | no-declared-license | 同上 |
| ★★★ | LiquidAI/LFM2.5-Audio-1.5B | 1k (436 lk) | other (LFM Open License) | Liquid AI 独自 bespoke |
| ★★★ | LiquidAI/LFM2-Audio-1.5B | 243 (358 lk) | other | 同 LFM Open License、efficient hybrid state-space |
| ★★ | LiquidAI/LFM2.5-Audio-1.5B-JP | 178 (68 lk) | other | LFM2.5 Japanese-tuned |
| ★★ | LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF | 2k | other | JP GGUF variant |
| ★★★ | zai-org/glm-4-voice-9b | 10k (119 lk) | other | GLM-4-Voice license 未宣言 = Zhipu 独自要参照 |
| ★★ | zai-org/glm-4-voice-tokenizer | 34k | other | GLM-4-Voice speech tokenizer companion |
| ★★ | zai-org/glm-4-voice-decoder | 182 | other | GLM-4-Voice vocoder/decoder |
| ★★ | tencent/Covo-Audio-Chat | 2k (100 lk) | other | Tencent 独自 license 要参照 |

### 6.2 TTS

| ★ | HF ID | dl | License | Owner ADR で確認 |
|---|---|---|---|---|
| ★★★ | coqui/XTTS-v2 | 9.26M (3695 lk) | other (Coqui CPML) | Coqui Public Model License = 非商用 research-only。**LicenseClass::CoquiPublic 新設要検討**。16 langs + zero-shot cloning、huge community |
| ★★★ | bosonai/higgs-tts-3-4b | 376k (684 lk) | other (Boson custom) | 100+ 言語カバー最広、要 Boson 独自条件確認 |
| ★★★ | bosonai/higgs-tts-2-3b-base | 194k (694 lk) | other (Boson custom) | 同上 v2 base |
| ★★★ | nvidia/personaplex-7b-v1 | 261k (2634 lk) | other (NVIDIA custom) + gated=auto | NVIDIA-specific 条件受諾要 |
| ★★★ | IndexTeam/IndexTTS-2 | 16k (764 lk) | unknown | Repo LICENSE 実文書要確認、high like count |
| ★★★ | Supertone/supertonic-3 | 32k (896 lk) | openrail | RAIL clauses (use-based restrictions) confirm |
| ★★ | Supertone/supertonic-2 | 1k (393 lk) | openrail | 同 openrail |
| ★★ | Supertone/supertonic | 577 (484 lk) | openrail | 同 openrail |
| ★★ | HumeAI/tada-1b | 15k (240 lk) | llama3.2 | Meta Llama-3.2 use-based restriction (Vokra 初 Llama license precedent 要判定) |
| ★★ | HumeAI/tada-3b-ml | 9k (159 lk) | llama3.2 | 同上 multilingual |
| ★★ | CAMB-AI/MARS5-TTS | 63 (480 lk) | agpl-3.0 | **AGPL-3.0 は strong copyleft (network use trigger)、Unity/Godot 使用不能 = design constraint 違反**、fails core use case |
| ★★ | eustlb/higgs-audio-v2-generation-3B-base | 4k | unknown | 権限継承なし、fail-closed default で Rejected 相当 |
| ★★ | multimodalart/higgs-audio-v3-tts-4b-transformers | 43k (17 lk) | other | Higgs Audio TTS license 'other' vs bosonai (Apache-2.0) の差異確認 |
| ★★ | bosonai/higgs-audio-v2-tokenizer | 33k (56 lk) | other | Higgs v2 独自条件 vs v3 STT (Apache-2.0) の乖離 |
| ★★ | kyutai/hibiki-zero-3b-pytorch-bf16 | 2k (56 lk) | unknown | 通常 Kyutai は CC-BY-4.0 mirror、要確認で昇格可能 |
| ★★ | coqui/XTTS-v1 | 661 (369 lk) | other (Coqui CPML) | v2 落選なら v1 も落選 (同 license) |

### 6.3 Music-Gen (Stability + Riffusion + Jukebox)

| ★ | HF ID | dl | License | Owner ADR で確認 |
|---|---|---|---|---|
| ★★★ | stabilityai/stable-audio-open-1.0 | 18k (1541 lk) | other (Stability Community) + gated=auto | 商用 <$1M revenue、prominent redistribution restrictions |
| ★★★ | stabilityai/stable-audio-3-medium | 40k (246 lk) | other + gated=auto | 同 Stability Community License |
| ★★ | stabilityai/stable-audio-open-small | 2k (270 lk) | other + gated=auto | 同上 |
| ★★ | stabilityai/stable-audio-3-small-music | 15k (103 lk) | other + gated=auto | Music-specialized |
| ★★ | stabilityai/stable-audio-3-small-sfx | 13k (72 lk) | other + gated=auto | SFX-specialized |
| ★★ | stabilityai/stable-audio-3-medium-base | 14k (24 lk) | other + gated=auto | pre-instruct-tuned |
| ★★ | stabilityai/stable-audio-3-optimized | 2k (26 lk) | other + gated=auto | Inference-optimized |
| ★★ | stabilityai/stable-codec-speech-16k | 746 (24 lk) | other + gated=auto | Stability codec 16kHz speech |
| ★★★ | nvidia/music-flamingo-2601-hf | 80k (108 lk) | other (NVIDIA Non-Commercial) | Music Q&A / captioning / theory analysis |
| ★★ | nvidia/music-flamingo-hf | 16k (99 lk) | other | Music-Flamingo base HF format |
| ★★ | nvidia/music-flamingo-think-2601-hf | 1k (41 lk) | other | Reasoning-tuned variant |
| ★★★ | riffusion/riffusion-model-v1 | 971 (650 lk) | creativeml-openrail-m | SD-1.5 spectrogram-image music-gen、CreativeML OpenRAIL-M has use-based restrictions |
| ★★ | slseanwu/MIDI-LLM_Llama-3.2-1B | 2k (34 lk) | llama3.2 | Llama 3.2 Community License、MIDI-LLM 唯一 (Stanford CRFM は permissive) |
| ★★ | openai/jukebox-1b-lyrics | 134 (21 lk) | other | ambiguous license (歴史的 OpenAI weights)、要確認 |
| ★★ | openai/jukebox-5b-lyrics | 91 (42 lk) | other | 同上 |
| ★★ | declare-lab/TangoFlux | 2k (112 lk) | other/unknown | License field 空、upstream README 参照要 |
| ★★ | nvidia/RE-USE | 17k (87 lk) | other (NVIDIA Open Model License) | 同 nemotron |
| ★★ | ilintar/thinksound-gguf | 4k | other | ThinkSound repo license 要確認 (NC likely) |

### 6.4 Enhancement / Codec / Watermark / Bio-Acoustic

| ★ | HF ID | dl | License | Owner ADR で確認 |
|---|---|---|---|---|
| ★★ | Sony/SilentCipher | 12k (7 lk) | unspecified | Sony repo/paper 要確認、AudioSeal alternative |
| ★★ | joaogante/dummy_synthid_detector | 108 | unspecified | SynthID audio 真 Google 契約要 (dummy placeholder) |
| ★★ | anvuew/BS-RoFormer | 130 (8 lk) | gpl-3.0 | Copyleft (LicenseClass::Copyleft T3 route)、Vokra core linking の GPL 適合要確認 |
| ★★ | anvuew/dereverb_bs_roformer | 374 (23 lk) | gpl-3.0 | 同上 dereverb-focused variant |
| ★★ | MigoXV/mossformer2-se-48k | 29 | unspecified | Upstream 要確認 |
| ★★ | chenmozhijin/BSRoformer-GGUF | 7k | unspecified | Original BS-RoFormer paper (MIT-ish) だが weights 第三者再配布 = 要確認 |
| ★★ | JusperLee/Dolphin | 3k (16 lk) | other | Dolphin speech separation (TIGER 作者 follow-up)、Upstream license 要確認 |
| ★★ | nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 | 4k (46 lk) | other | NeMo family (通常 CC-BY-4.0 だが repo 未宣言)、ONNX weights 済 |
| ★★ | qualcomm/YamNet | (mirror) | other | YAMNet mirror license 独自 (Qualcomm) |
| ★★ | Speech-Arena-2025/DF_Arena_500M_V_1 | 6k | other | Speech-Arena 2025 consortium terms 要確認 |
| ★★ | Speech-Arena-2025/DF_Arena_1B_V_1 | 2k (9 lk) | other | 同上 1B version |
| ★★ | tiantiaf/whisper-large-v3-msp-podcast-emotion-dim | 3k | openrail | RAIL use-based restrictions |
| ★★ | tiantiaf/wavlm-large-age-sex | 2k (11 lk) | openrail | 同上 |
| ★★ | google/DiarizationLM-8b-Fisher-v2 | 3k (37 lk) | llama3 | Llama-3 Community License (name-attribution required)、LLM-based post-processor for diarization |
| ★★ | WillHeld/DiVA-llama-3-v0-8b | 37 (35 lk) | mpl-2.0 | MPL-2.0 file-level weak-copyleft、DiVA (Distilled Voice Assistant) on Llama-3 |
| ★★ | DBD-research-group/ConvNeXT-Base-BirdSet-XCL | 1k | other | Bio-acoustic bird call classifier |
| ★★ | DBD-research-group/Bird-MAE-Base | 1k | other | Bird-MAE self-supervised encoder |
| ★★ | orcasound/orcahello-srkw-detector-v1 | 2k | other | Marine bio-acoustic Southern Resident Killer Whale detector |

## 7. TIER 5 (voice-cloning、別リポ管轄)

**Historical design judgment 8 anchor (superseded as current legal guidance)**: the
2026-07 research record described Tennessee ELVIS Act / a proposed federal NO
FAKES Act risk for voice-cloning tools and proposed a separate
`vokra-voiceclone-experimental` repository. Current legal applicability,
disclosure/marking, consent, and rights must be checked by the owner/deployer
under `docs/legal-compliance.md` §§3–4 and FR-CP-04; this row does not claim
automatic compliance.

**該当モデルは main repo (ayutaz/vokra) には publish しない**、`staging/vokra-voiceclone-experimental` scaffold (M5-05、`6dc9f86` land) から別リポへ移送。tier は "license" ではなく "primary use = voice-clone" で判定。license 自体は permissive (MIT) を含む。

| ★ | HF ID | dl | License | 位置付け |
|---|---|---|---|---|
| ★★★ | lj1995/VoiceConversionWebUI | 0 (1207 lk) | mit | RVC WebUI = de facto RVC assets hub、data hub なので dl=0 |
| ★★★ | myshell-ai/OpenVoiceV2 | 0 (497 lk) | mit | OpenVoice V2 zero-shot voice-clone (Vokra openvoice_v2 wire 済、publish 別リポ) |
| ★★ | microsoft/speecht5_vc | 3k (112 lk) | mit | SpeechT5 VC any-to-any conversion (MIT permissive だが primary = voice clone) |
| ★★ | SPRINGLab/EZ-VC | 211 (7 lk) | cc-by-nc-4.0 + gated=auto | EZ-VC "Easy Zero-shot Any-to-Any"、NC + voice-clone primary = tier5 明確 |

**注**: Vokra 対応済の以下 4 モデルも既に別リポ計画: openvoice_v2 / knn_vc / freevc / meanvc (2026-07-30 residual wave3 で `5f7cb15` wire 済、code side 消化 = ModelKind + CLI dispatch、publish は voiceclone repo 側 owner)。

## 8. 実装優先度提案

以下 4 案から依頼者選択、または hybrid で組合わせ。

### Option A: TIER 1 全数 一括 land (~30 wave)

- 175 unique tier1 models を wave-per-family 単位で land (ultracode 20 agents parallel × 30 wave)
- 見積り: CC 実装 ~3-4 週間、per-wave verify で 4 gate green (test/fmt/clippy/zero-dep)
- **メリット**: 一気に compat matrix 拡大、model zoo 3.5x 拡張 (現 31 → ~200 models)
- **デメリット**: owner sign-off queue が 175 行に膨らむ、review 負担大、per-family §3.1 sign-off の primary source 確認 owner 実行の bottleneck
- **リスク**: 音楽関連は EnCodec 依存 (MusicGen 系) が多く、Vokra は EnCodec 除外 policy 遵守で MusicGen 対応時 codec 差し替えが技術的 blocker

### Option B: 高需要 TOP 20 先行 (~5 wave)

- §2.3 の dl>500k を優先実装、tier 混合
- Wave 1: pyannote pipeline layer 4 items (speaker-diarization-3.1 + VAD + segmentation + embedding) + wespeaker-voxceleb-resnet34-LM
- Wave 2: wav2vec2 CTC family (base-960h + large-960h-lv60-self + xlsr-53 + jonatasgrosman-JA)
- Wave 3: Qwen3-ASR (0.6B + 1.7B + ForcedAligner) + Cohere-Transcribe + Granite Speech
- Wave 4: BigVGAN v2 (3 variants) + HiFi-GAN LibriTTS + NeuCodec
- Wave 5: CLAP (5 variants) + AST (2) + MERT + audiobox-aesthetics
- 見積り: CC 実装 ~5-7 日、owner sign-off queue ~30 行
- **メリット**: ROI 最大、compat matrix の重要ノードを埋める
- **デメリット**: coverage 断片的、深さ (family 全体) より広さ寄り

### Option C: カテゴリ縦割り (Music family / Enhancement family / ...)

- 依頼者音楽 in-scope 化を最活用:
  - **Wave M**: 音楽関連一括 (ACE-Step 10 + BigVGAN 4 + CLAP 5 + AST 2 + MERT 2 + Basic Pitch + Stanford Anticipation 3 + HTDemucs + BS-Roformer + audiobox-aesthetics + MOSS-Music/SoundEffect + Mustango + Magenta community = ~35 models)
- **Wave A**: ASR wav2vec2/HuBERT foundation 一括 (base + large + xlsr + hubert + w2v-BERT + jonatasgrosman family 代表 5 lang + KO/SV/TH + eSpeak phoneme = ~15 models)
- **Wave T**: TTS MOSS + Parler + MeloTTS family 一括 (MOSS 8 + Parler 3 + MeloTTS 4 + Orpheus 3 + Marvis 2 + Neu = ~22 models)
- **Wave V**: VAD / turn-detection / pyannote pipeline 一括 (pyannote 5 + smart-turn 2 + Namo 2 + FireRedVAD + phoneme aligner 2 = ~14 models)
- **Wave E**: Enhancement / Separation / Codec 一括 (TIGER 3 + SepFormer 5 + MP-SENet + MetricGAN+ + LocalVQE + NeuCodec + FocalCodec + MioCodec = ~15 models)
- **Wave L**: Audio-LLM 一括 (Qwen2-Audio + MiniCPM-o + VITA + Higgs v3 STT + HeartMuLa + SALMONN-2 = ~10 models)
- 見積り: 6 wave × 5-10 日 = 6-8 週間、model zoo ~110 追加
- **メリット**: family 全体を deep 対応、Vokra 「深さで勝つ」戦略と一致
- **デメリット**: wave 間で dependency (Vulkan/CUDA/Metal kernel 依存等) が複雑化

### Option D: Foundational 優先 (wav2vec2 / CLAP / BigVGAN / MMS / Demucs)

- Vokra が「downstream compound」できる base model のみ集中
- Wave 1: wav2vec2 base + large-960h-lv60-self + xlsr-53 + w2v-BERT-2.0 + hubert-base + hubert-large (~6 models)
- Wave 2: CLAP HTSAT (fused + unfused + larger 3 variants) + AST-AudioSet (2) (~7 models)
- Wave 3: BigVGAN v2 (3) + HiFi-GAN LibriTTS + NeuCodec + distill (~6 models)
- Wave 4: MMS-1B / MMS-300M / MMS forced-aligner (T4 経路) (~3 models)
- Wave 5: HTDemucs + BS-Roformer + Kim Vocal 2 (~3 models、music-in-scope 追加)
- Total ~25 models、CC 実装 ~7-10 日
- **メリット**: これら base 対応後、jonatasgrosman family 15+ 言語 / MMS-TTS 1000 langs / MMS-LID 4017 langs / Cantonese w2v-BERT / RVC 系 (別リポ) / TIGER family 等が「converter 差替のみ」で連鎖 land 可能。実質 model zoo は base 対応の 10-20x 倍化
- **デメリット**: end-user 直感的なモデル (Whisper tiny / MusicGen 等) は Wave 6 以降に押し出し

**推奨**: **Option D + Option C の Wave M (音楽) + Wave V (turn-detection) を並走**。foundational で multiplier 効果を確保しつつ、依頼者 in-scope 化した音楽 + full-duplex agent 需要 (Wyoming completion M4-19 dependency) を並行 land。

## 9. 本 doc の限界

1. **HF Hub 全数ではない**: 各カテゴリで top-150 (dl+likes) から抽出。long-tail (dl<100) の低知名度モデルは未網羅。特に research-only の学会論文 companion (ICASSP/Interspeech 直近) は反映されていない可能性。
2. **fine-tune family は代表 1 行**: jonatasgrosman/wav2vec2-XLSR-53 は 15+ 言語 fine-tune が単一 arch を共有するため、日本語 (1.7M dl) 1 行で他 14+ 言語 (合計 dl 15M+) を暗示。同様に facebook/mms-tts-* 1000+ 言語族は eng + ara の 2 行で family 全体を示唆。
3. **community mirror 集約**: `mlx-community/*` / `onnx-community/*` / `aufklarer/*` (CoreML) / `ilintar/*` (GGUF) / `Xenova/*` / `Systran/*` / `FluidInference/*` / `handy-computer/*` 等の format-port mirror は元 model 1 行に集約。redistribution は元 model の license を継承する必要。
4. **license 検証は cardData のみ**: HF `/api/models/{id}` cardData の license field のみ参照。実際の LICENSE ファイルとの一致は publish 前 §3.1 sign-off で primary source 確認必要。特に `no-declared-license` / `unknown` / `other` は owner ADR が最優先 gate。
5. **音楽以外の scope 拡張は本 doc に反映せず**: 依頼者 2026-07-30 で明示 in-scope 化されたのは音楽のみ。汎用 LLM (Llama / Qwen text / GPT系) / vision / video / multimodal LLM (Vision 部分) は out-of-scope 維持。ただし Phi-4-multimodal-instruct のように speech adapter だけ切り出せるものは owner ADR で per-model 判断。
6. **T4 tier 適用可否は各 tier3 モデルで independent 判断**: X-Codec-2 precedent (`--allow-noncommercial` + LicenseClass::NonCommercial + fetch_license SPDX) は cc-by-nc-4.0 に applies だが、cc-by-nc-sa-4.0 (share-alike 追加) / OpenRAIL (use-based restrictions) / Coqui CPML (bespoke) / Llama-3.2 / Stability Community (revenue-based) は別 tier / 別 owner ADR。
7. **voice-clone 判定は "primary purpose"**: license permissive (MIT/Apache) でも primary purpose = zero-shot voice cloning of arbitrary target speakers ならば tier5 (別リポ) 扱い (ELVIS Act / NO FAKES / design 判断 8)。「speaker adaptation」「voice style transfer」との境界は各モデル card + paper primary use case で判定要、SpeechT5-VC 等は境界事例で ADR 要 (main repo 残置か別リポ移送か)。
8. **依頼者音楽 scope 拡張は本 doc 内 permissive のみ実務化可能**: MusicGen family (cc-by-nc + EnCodec) は Vokra EnCodec 除外 policy と衝突、代替 codec (DAC / SNAC / SpeechTokenizer) への re-training or 変換が必要 = 実装 blocker。ACE-Step (MIT) / Mustango (Apache) / MOSS-SoundEffect (Apache) / Magenta community (Apache) / Basic Pitch (Apache) / Stanford Anticipation (Apache) / HTDemucs (MIT) 等の permissive 選択が優先。
