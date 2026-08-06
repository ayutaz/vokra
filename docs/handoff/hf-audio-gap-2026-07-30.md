# HuggingFace 音声モデル gap 調査 (2026-07-30)

依頼者「他に音声系のモデルで対応していないものを huggingface から探してく
ださい。なるべく多くのモデルを対応するようにしたいです」を受けた system
体系調査。

## 調査方法

- HF API `pipeline_tag={automatic-speech-recognition, text-to-speech,
  voice-activity-detection, audio-to-audio, audio-classification,
  text-to-audio}` の各 top-50 by downloads を取得 (計 ~300 model)
- 現行 Vokra coverage (`ModelKind` 54 variants + F0 fcpe + Wave 3+4
  pyannote 統合) と cross-reference
- 未 covered モデルは per-model `api/models/<id>` で
  license / gated / pipeline / language / downloads を確認
- **設計判断 8**: voice-cloning target (`personaplex`, `speecht5_vc` 等)
  は main repo 対象外 = `vokra-voiceclone-experimental` 別リポ管轄
- **memory `project-goal-depth-not-breadth`**: 音楽生成 (MusicGen,
  Stable Audio, ACE-Step) は音声 (speech) scope 外 = **out-of-scope 維持**
- **memory `feedback-license-signoff-primary-source`**: 実 publish 前
  の owner sign-off は primary source (HF cardData) 直接照合が必要

## 現行 Vokra coverage (2026-07-30 時点、baseline)

**Converter ModelKind 54 variants** + F0 fcpe: Whisper family (base/small/
medium/large-v3/turbo) / Kokoro / piper-plus / CAM++ / Silero VAD /
CosyVoice2 / CosyVoice3 / Voxtral (Mini/Small) / Mimi / DAC /
WavTokenizer / SpeechTokenizer / FunCodec / XY-Tokenizer / Bicodec /
Neucodec / X-Codec 2 / CSM-1B / Moshi / F5-TTS / Fish-Speech / EnCodec
(NC gated) / DeepFilterNet3 / UTMOS22 / Bark / StyleTTS 2 (Rejected) /
Matcha-TTS / TitaNet / ECAPA-TDNN / WeSpeaker / speaker_3d /
emotion2vec / pyannote-segmentation (Wave 3+4 landed) / Canary /
Canary-Qwen / Parakeet-TDT / Parakeet-CTC / omniASR-CTC / distil-
whisper / kotoba-whisper / Zonos / Dia / VoxCPM / VoxCPM2 / VibeVoice /
Chatterbox (base/turbo/nano) / Qwen3-TTS-0.6B/1.7B / Irodori-TTS /
kimi_audio / step_audio2_mini / baichuan_audio / kyutai_stt / rmvpe /
crepe / Charsiu (wav2vec2 CTC align) / SBV2 / deberta-v2 / deberta-v3 /
reazonspeech-k2 (Zipformer) / OWSM (E-Branchformer) 他。

**Voice-clone 別リポ (main repo out-of-scope、設計判断 8)**:
openvoice_v2 / knn_vc / freevc / meanvc / RVC v2 / GPT-SoVITS。

## Gap 一覧 (優先度付き)

## TIER 1: apache-2.0 / MIT (商用可、attribution 不要) — **最優先**

### ASR

| 優先 | Model | dl | 用途 |
|---|---|---:|---|
| ★★★ | `Qwen/Qwen3-ASR-0.6B` | 2.4M | Alibaba Qwen3-ASR 0.6B (multilingual) |
| ★★★ | `Qwen/Qwen3-ASR-1.7B` | 1.9M | sibling 1.7B |
| ★★★ | `facebook/wav2vec2-base-960h` | 1.7M | wav2vec2 base LibriSpeech 960h (foundational en) |
| ★★★ | `facebook/wav2vec2-large-xlsr-53` | 397k | XLSR-53 base (multilingual pretrained) |
| ★★ | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` | 1.7M | JA fine-tune of XLSR-53 |
| ★★ | `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | 1.3M | ZH-CN fine-tune |
| ★★ | `mistralai/Voxtral-Mini-4B-Realtime-2602` | 2M | Voxtral 4B realtime variant |
| ★ | `distil-whisper/distil-large-v3` | 1.6M | MIT、sibling of distil-large-v3.5 (既 covered) |
| ★ | `CohereLabs/cohere-transcribe-03-2026` | 1.1M | Cohere transcribe (gated=auto) |

### TTS

| 優先 | Model | dl | 用途 |
|---|---|---:|---|
| ★★★ | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 2.5M | Qwen3-TTS 1.7B CustomVoice variant |
| ★★★ | `OpenMOSS-Team/MOSS-TTS` | 375k | Fudan MOSS TTS (30+ lang) |
| ★★ | `OpenMOSS-Team/MOSS-TTS-v1.5` | 110k | sibling |
| ★★ | `OpenMOSS-Team/MOSS-TTS-Nano-100M` | 57k | Nano variant |
| ★★ | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | 58k | Local transformer variant |
| ★★ | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 686k | VoiceDesign variant |
| ★★ | `ai4bharat/indic-parler-tts` | 376k | Indic TTS (gated=auto) |
| ★★ | `speechbrain/tts-hifigan-libritts-22050Hz` | 281k | HiFi-GAN vocoder (**Vokra は BigVGAN skeleton あり、HiFiGAN 未実装**) |
| ★★ | `myshell-ai/MeloTTS-English` | 152k | MeloTTS EN (MIT) |
| ★★ | `myshell-ai/MeloTTS-Chinese` | 49k | MeloTTS ZH (MIT) |
| ★★ | `myshell-ai/MeloTTS-Korean` | 71k | MeloTTS KO (MIT) |
| ★★ | `microsoft/speecht5_tts` | 64k | SpeechT5 TTS (MIT) |
| ★ | `pnnbao-ump/VieNeu-TTS-v3-Turbo` | 347k | Vietnamese TTS |
| ★ | `parler-tts/parler-tts-mini-multilingual-v1.1` | 70k | Parler TTS mini |
| ★ | `suno/bark-small` | 126k | Bark small (main Bark 既 covered) |

### VAD / diarization

| 優先 | Model | dl | 用途 |
|---|---|---:|---|
| ★★ | `funasr/fsmn-vad` | 2.9k | FSMN VAD (**residual-wave3 で cherry-pick 失敗、retry 対象**) |
| ★★ | `FunAudioLLM/fsmn-vad-GGUF` | 1.9k | 上流 GGUF 版 |
| ★ | `pipecat-ai/smart-turn-v2` | 4.1k | Turn detection (bsd-2-clause) |
| ★ | `FireRedTeam/FireRedVAD` | 1.4k | FireRed VAD (apache-2.0) |

### Codec / vocoder / enhancement

| 優先 | Model | dl | 用途 |
|---|---|---:|---|
| ★★★ | `nvidia/bigvgan_v2_22khz_80band_256x` | 1.7M | **BigVGAN v2 vocoder** (MIT、Vokra は skeleton のみ = generator op 未実装) |
| ★★★ | `nvidia/bigvgan_v2_44khz_128band_512x` | 958k | BigVGAN v2 44kHz |
| ★★ | `nvidia/bigvgan_v2_24khz_100band_256x` | 28k | BigVGAN v2 24kHz |
| ★ | `nvidia/bigvgan_base_24khz_100band` | 1.6k | BigVGAN base |
| ★★ | `JusperLee/TIGER-DnR` | 178k | TIGER dnr source separation (apache-2.0) |
| ★ | `JusperLee/TIGER-speech` | 20k | TIGER speech separation |
| ★ | `JacobLinCool/MP-SENet-DNS` | 58k | MP-SENet noise suppression (MIT) |
| ★ | `speechbrain/metricgan-plus-voicebank` | 9k | MetricGAN+ enhancement |
| ★ | `speechbrain/sepformer-wsj02mix` | 3k | SepFormer source separation |
| ★ | `speechbrain/sepformer-wham16k-enhancement` | 2.6k | SepFormer enhancement |
| ★ | `lucadellalib/focalcodec_50hz` | 1.4k | Focal codec |

### Audio classification / speaker embedding

| 優先 | Model | dl | 用途 |
|---|---|---:|---|
| ★★★ | `laion/clap-htsat-fused` | **8.1M** | CLAP audio-text alignment (apache-2.0) |
| ★★ | `MIT/ast-finetuned-audioset-10-10-0.4593` | 620k | AST (Audio Spectrogram Transformer、bsd-3-clause) |
| ★★ | `speechbrain/lang-id-voxlingua107-ecapa` | 116k | 107-lang identification |
| ★ | `speechbrain/spkrec-xvect-voxceleb` | 10k | x-vector speaker embedding (CAM++ で covered 済でも代替として) |
| ★ | `MelodyMachine/Deepfake-audio-detection-V2` | 4.7k | Deepfake detection |
| ★ | `speechbrain/lang-id-commonlanguage_ecapa` | 4.7k | CommonLanguage lang ID |

## TIER 2: cc-by-4.0 (商用可、attribution 要)

| 優先 | Model | dl | 用途 |
|---|---|---:|---|
| ★★ | `kyutai/tts-1.6b-en_fr` | 52k | Kyutai TTS (Moshi 家族の attribution 継承) |
| ★ | `kyutai/moshika-rag-pytorch-bf16` | 2k | Moshika RAG |
| ★★ | `facebook/audiobox-aesthetics` | 321k | Audio aesthetics classifier |

## TIER 3: cc-by-nc / cc-by-nc-sa (非商用、Research-only)

**判定**: T4 (Research-only) 経路で converter + zoo publish 可能 (X-Codec 2
precedent = 2026-07-28 land)。`--allow-noncommercial` 明示 flag 必須。

| Model | dl | 用途 |
|---|---:|---|
| `MahmoudAshraf/mms-300m-1130-forced-aligner` | 2.4M | MMS forced aligner (**Charsiu の代替**、多言語 1000+ 対応) |
| `SWivid/E2-TTS` | 114k | E2 TTS (Emilia dataset) |
| `facebook/hf-seamless-m4t-medium` | 99k | SeamlessM4T medium (多言語 S2ST) |
| `2Noise/ChatTTS` | 2.3k | ChatTTS (人気 1663 likes) |
| `facebook/mms-tts-eng` + 全 MMS TTS | 108k | Facebook MMS TTS (1000+ 言語) |
| `facebook/mms-lid-{126,256,1024,4017}` | 250k+ | MMS language identification |
| `m-a-p/MERT-v1-330M / v1-95M` | 665k+ | MERT music understanding |
| `OpenMuQ/MuQ-large-msd-iter` | 304k | MuQ music understanding |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` | 268k | age+gender classifier |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | 654k | emotion (**emotion2vec covered、代替**) |
| `BUT-FIT/diarizen-wavlm-large-s80-md` | 668 | diarization WavLM |
| `notmax123/Zonos-Hebrew` | 177k | Zonos Hebrew fork |

## TIER 4: "other" / gated / unknown (要精査、per-model owner 判断)

| Model | dl | license | notes |
|---|---:|---|---|
| `coqui/XTTS-v2` | **9.3M** | other (CPML 1.0) | Coqui Public Model License = non-commercial 個人 free / 商用は subscription。**T4 経路** |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | 960k | other | NVIDIA custom license (要 primary source 精査) |
| `bosonai/higgs-tts-3-4b` | 376k | other | Higgs Audio v3 (要精査) |
| `bosonai/higgs-tts-2-3b-base` | 194k | other | Higgs Audio v2 |
| `fishaudio/s2-pro` | 264k | other | Fish S2 Pro (要精査) |
| `stabilityai/stable-codec-speech-16k` | 746 | other (gated=auto) | Stability speech codec |
| `LiquidAI/LFM2.5-Audio-1.5B` | 1.3k | other | LFM audio LLM |
| `LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF` | 2.2k | other | JA variant |
| `k2-fsa/OmniVoice` | 848k | ? | 1200+ lang TTS (license 未宣言) |
| `tencent/Covo-Audio-Chat` | 2.1k | ? | Tencent audio chat |
| `kyutai/hibiki-zero-3b-pytorch-bf16` | 2k | ? | Hibiki translation |

## voice-cloning targets (別リポ管轄、main repo 対象外)

**設計判断 8** により `vokra-voiceclone-experimental` へ:

- `microsoft/speecht5_vc` (2.8k dl, MIT) — SpeechT5 voice conversion
- `nvidia/personaplex-7b-v1` (261k dl, other) — persona-based voice conversion 7B
- 既 wire 済 (main repo 側 converter code のみ): `myshell-ai/OpenVoiceV2`, `bshall/knn-vc`, `OlaWod/FreeVC`, `ASLP-lab/MeanVC`

## Out-of-scope (音声 scope 外、深さで勝つ方針)

**memory `project-goal-depth-not-breadth`**「音声モデルで深さで勝つ。音声以外
は非対応」に照らして **音楽生成 (music generation)** は out-of-scope 維持:

- `facebook/musicgen-*` (small/medium/large/melody/stereo, ~2.4M combined dl)
- `stabilityai/stable-audio-*` (open, 3-medium, 3-small, sfx variants, ~90k combined)
- `ACE-Step/*` (music generation family, ~100k combined)
- `google/magenta-realtime-2` (7.1k)
- `declare-lab/mustango`, `declare-lab/TangoFlux` (音楽 + audio effects)

これらは speech (音声) ではなく music (音楽) 生成なので、Vokra の
core scope 外。将来別 project or 拡張として検討可能だが本 gap 対象外。

## 総括: 実装優先度サマリ

**TIER 1 (permissive、実装優先)**:
- ASR 9 家族 (Qwen3-ASR 2 サイズ + wav2vec2 + XLSR + XLSR-JA/ZH-CN + Voxtral realtime + distil-large-v3 + Cohere)
- TTS 15 モデル (Qwen3-TTS 1.7B x2 + MOSS 4 variants + HiFiGAN vocoder + MeloTTS 3 lang + SpeechT5 + parler + indic + VieNeu + bark-small)
- VAD 4 (FSMN + FireRed + smart-turn + Namo)
- Codec/vocoder 12 (**BigVGAN v2 x4 = 高需要**, TIGER, MP-SENet, SepFormer, MetricGAN, focalcodec)
- Audio classification 6 (**CLAP 8.1M dl = 最高需要**, AST, voxlingua107, xvect, deepfake)
- **計 ~46 モデル**

**TIER 2 (cc-by、attribution 込 = **推奨**)**:
- Kyutai family 2 (TTS + Moshika RAG)
- audiobox-aesthetics
- **計 3**

**TIER 3 (NC、Research-only、T4 経路)**:
- MMS family (aligner + TTS + LID 4 サイズ) = **音声処理では非常に重要な foundational**
- MERT / MuQ music understanding
- SeamlessM4T / E2-TTS / ChatTTS
- audeering emotion / age-gender / BUT-FIT diarizen / Zonos-Hebrew
- **計 ~12**

**TIER 4 (要精査、owner ADR 判断)**:
- coqui/XTTS-v2 (9.3M dl = massive、CPML の商用 subscription 判断)
- Boson Higgs / Fish s2-pro / LiquidAI LFM / Tencent Covo / k2-fsa OmniVoice
- **計 ~11**

**voice-cloning (別リポ)**: microsoft/speecht5_vc + nvidia/personaplex-7b-v1

**out-of-scope (音楽)**: MusicGen / Stable Audio / ACE-Step family

## 次アクション候補

依頼者判断で以下から:

1. **TIER 1 一括 land**: 46 モデル (converter skeleton + 主要 6-7 モデルの
   runtime scaffold) を Wave 5 として ultracode で実装
2. **BigVGAN v2 vocoder 単独**: 需要 1.7M dl / 高価値 / Vokra は skeleton
   のみ ゆえ generator op + weight loader を先行実装
3. **CLAP + AST + AudioSet classifier**: 音声 embedding / classification
   系を一括
4. **MMS family (TIER 3)**: 1000+ 言語対応の foundational、Research-only
   でも T4 publish 経路で価値大
5. **XTTS-v2 (TIER 4)**: 9.3M dl の巨大需要、license 商用判定は要 ADR
   だが実装価値は極大
6. **wav2vec2 family (TIER 1)**: XLSR-53 + 日本語/中国語/その他多言語 fine-
   tune 群、音声認識の foundational

**メモリ規律**: 実装時は 全 gate green + primary source 直接引用 + 設計
判断 8 (voice-clone 分離) 遵守 + NFR-DS-02 (zero-dep) 維持 + 実 sign-off
は owner (`feedback-license-signoff-primary-source`)。
