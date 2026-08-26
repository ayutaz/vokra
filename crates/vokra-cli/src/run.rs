//! `vokra-cli run` — load a GGUF and run its task on an input (M1-10a).
//!
//! ```text
//! vokra-cli run --model vad.gguf     --input speech.wav
//! vokra-cli run --model whisper.gguf --input speech.wav
//! vokra-cli run --model voice.gguf   --text "hello vokra" [--output out.wav]
//! ```
//!
//! The task is detected from the model architecture (see [`crate::engine`]);
//! VAD prints per-frame speech-probability summary, ASR prints the transcript,
//! and TTS writes a WAV (or reports the sample count when `--output` is absent).

use std::process::ExitCode;

use vokra_core::engines::{KwsEngine, SeparationEngine};
use vokra_core::{AecEngine, Session};

use crate::engine::{self, ModelTask};
use crate::runtime_contracts::{
    FocalCodecCodesV1, MeloTextFeaturesV1, MimiCodesV1, SnacCodesV1, parse_ct_punc_tsv, sha256,
};
use crate::wav;

pub(crate) const USAGE: &str = "\
vokra-cli run — load a GGUF and run VAD / ASR / TTS / speaker / language ID / watermark

USAGE:
    vokra-cli run --model <model.gguf> [--input <in.wav>] [--text <string>] [--output <out.wav>]
                  [--backend cpu|metal|cuda] [--beam-size <N>] [--length-penalty <α>]
                  [--no-repeat-ngram <N>] [--fixture-tokenizer] [--interrupt-after <N>]
                  [--deterministic] [--far-end <reference.wav>]
    vokra-cli run --model <whisper.gguf> --input <in.wav> --word-timestamps
    vokra-cli run --model <parakeet-tdt.gguf> --input <16k-mono.wav>
    vokra-cli run --model <nemotron-asr.gguf> --input <16k-mono.wav> \
                  [--tokenizer <tokenizer.json>] [--language <code|auto>]
    vokra-cli run --model <wav2vec2.gguf> --input <16k-mono.wav> [--output <features.f32>]
    vokra-cli run --model <voxtral.gguf> --input <in.wav> [--language <code>] [--bare-prompt]
    vokra-cli run --model <speaker.gguf> --input <a.wav> [--compare <b.wav>] [--output <embedding.f32>]
    vokra-cli run --model <lang-id.gguf> --input <16k-mono.wav> [--output <scores.f32>]
    vokra-cli run --model <audiobox-aesthetics.gguf> --input <16k-mono.wav> \
                  [--output <ce-cu-pc-pq.f32>]
    vokra-cli run --model <audioseal.gguf> --input <16k-mono.wav> \
                  [--watermark-mode detect] [--watermark-variant base|streaming]
    vokra-cli run --model <audioseal.gguf> --input <16k-mono.wav> --watermark-mode embed \
                  --watermark-message <16-bits> [--watermark-alpha <gain>] [--output <out.wav>]
    vokra-cli run --model <kokoro.gguf> --text <phonemes> --style <s.f32> [--output <out.wav>]
    vokra-cli run --model <sbv2.gguf> --bert-ja <bert_ja.gguf> --bert-en <bert_en.gguf> \
                  --text <string> [--language ja|en] [--output <out.wav>]
    vokra-cli run --model <melotts.gguf> --input <features.vmf> [--length-scale <s>] \
                  [--output <out.wav>]
    vokra-cli run --model <fsmn-vad.gguf> --input <in.wav>
    vokra-cli run --model <smart-turn-v2.gguf> --input <16k-mono.wav>
    vokra-cli run --model <utmos22-strong.gguf> --input <16k-mono.wav>
    vokra-cli run --model <openwakeword.gguf> --input <16k-mono.wav>
    vokra-cli run --model <nsnet2.gguf> --input <noisy.wav> [--output <clean.wav>]
    vokra-cli run --model <metricgan-plus.gguf> --input <16k-noisy.wav> [--output <clean.wav>]
    vokra-cli run --model <deepfilternet3.gguf> --input <48k-noisy.wav> [--output <clean.wav>]
    vokra-cli run --model <rnnoise.gguf> --input <48k-noisy.wav> [--output <clean.wav>]
    vokra-cli run --model <separator.gguf> --input <mixture.wav> [--output <separated.wav>]
    vokra-cli run --model <conv-tasnet.gguf> --input <noisy.wav> [--output <enhanced.wav>]
    vokra-cli run --model <pyannote-segmentation.gguf> --input <in.wav>
    vokra-cli run --model <pyannote-diarization.gguf> \
                  --segmentation-model <pyannote-segmentation.gguf> \
                  --embedding-model <pyannote-wespeaker.gguf> --input <16k-mono.wav> \
                  [--output <turns.rttm>]
    vokra-cli run --model <rmvpe.gguf> --input <in.wav>
    vokra-cli run --model <fcpe.gguf> --input <in.wav>
    vokra-cli run --model <crepe.gguf> --input <in.wav>
    vokra-cli run --model <charsiu.gguf> --input <in.wav> --text \"P AE T\" \
                  [--output <alignment.tsv>]
    vokra-cli run --model <wetextprocessing.gguf> --text <string>
    vokra-cli run --model <nkf-aec.gguf> --input <mic.wav> --far-end <reference.wav> \
                  [--output <clean.wav>]
    vokra-cli run --model <ct-punc.gguf> --tokens <tokens.tsv> [--output <restored.txt>]
    vokra-cli run --model <bert-family.gguf> --token-ids <u32,u32,...> [--output <hidden.f32>]
    vokra-cli run --model <mimi.gguf> --codec-mode encode --input <in.wav> --output <codes.vmc>
    vokra-cli run --model <mimi.gguf> --codec-mode decode --input <codes.vmc> --output <out.wav>
    vokra-cli run --model <dac.gguf> --codec-mode decode --input <codes.u32le> --output <out.wav>
    vokra-cli run --model <wavtokenizer.gguf> --codec-mode decode --input <codes.u32le> \
                  [--bandwidth-id <0..3>] --output <out.wav>
    vokra-cli run --model <neucodec.gguf> --codec-mode decode --input <codes.u32le> \
                  --output <out.wav>
    vokra-cli run --model <xcodec2.gguf> --codec-mode decode --input <codes.u32le> \
                  --output <out.wav>
    vokra-cli run --model <funcodec.gguf> --codec-mode decode --num-quantizers <1..32> \
                  --input <codes.u32le> --output <out.wav>
    vokra-cli run --model <speechtokenizer.gguf> --codec-mode decode \
                  --num-quantizers <1..8> --input <codes.u32le> --output <out.wav>
    vokra-cli run --model <miocodec.gguf> --codec-mode decode --input <tokens.vmi> \
                  --output <out.wav>
    vokra-cli run --model <yue-upsampler.gguf> --input <1024ch-features.f32> \
                  [--output <44.1k-out.wav>]
    vokra-cli run --model <snac.gguf> --codec-mode encode --input <in.wav> --output <codes.vsc>
    vokra-cli run --model <snac.gguf> --codec-mode decode --input <codes.vsc> --output <out.wav>
    vokra-cli run --model <focalcodec.gguf> --codec-mode encode --input <16k.wav> --output <codes.vfc>
    vokra-cli run --model <focalcodec.gguf> --codec-mode decode --input <codes.vfc> --output <out.wav>
    vokra-cli run --model <moss-audio-tokenizer-nano.gguf> --codec-mode decode \
                  --num-quantizers <1..16> --input <codes.u32le> --output <stereo.wav>
    vokra-cli run --model <moss-tts-nano.gguf> \
                  --audio-tokenizer <moss-audio-tokenizer-nano.gguf> \
                  --max-new-frames <N> --input <prompt-rows.u32le> --output <stereo.wav>
    vokra-cli run --model <moss-tts-v1.5.gguf> \
                  --audio-tokenizer <moss-audio-tokenizer-full.gguf> \
                  --max-new-frames <N> --input <prompt-rows.u32le> --output <mono.wav>
    vokra-cli run --model <moss-voice-generator.gguf> \
                  --audio-tokenizer <moss-audio-tokenizer-full.gguf> \
                  --max-new-frames <N> --input <prompt-rows.u32le> --output <mono.wav>
    vokra-cli run --model <bigvgan.gguf> --input <mel.f32> [--output <out.wav>]
    vokra-cli run --model <hifigan-vocoder.gguf> --input <mel.f32> [--output <out.wav>]
    vokra-cli run --model <speecht5-hifigan.gguf> --input <mel.f32> [--output <out.wav>]
    vokra-cli run --model <vocos.gguf> --input <features.f32> [--bandwidth-id <0..3>] [--output <out.wav>]

OPTIONS:
    --model <path>              GGUF model file (arch selects VAD / ASR / TTS / S2S /
                                speaker / language ID / denoise / separation /
                                segmentation / diarization / F0).
                                An arch vokra-models binds but this CLI has no
                                task for is refused with the binding module and
                                the library entry point to call — never a bare
                                `unsupported model arch` (FR-EX-08).
    --segmentation-model <path> pyannote-speaker-diarization only, REQUIRED:
                                exact pyannote/segmentation-3.0 PyanNet GGUF.
                                The weightless pipeline GGUF cannot infer or
                                download this dependency.
    --embedding-model <path>    pyannote-speaker-diarization only, REQUIRED:
                                exact pyannote WeSpeaker ResNet34-LM GGUF.
                                It keeps its independent CC-BY attribution and
                                strict manifest gate.
    --audio-tokenizer <path>    MOSS-TTS only, REQUIRED: exact companion GGUF.
                                Nano requires Audio Tokenizer Nano; Base/v1.5
                                require Audio Tokenizer Full. Both stages must
                                support the same selected backend.
    --tokenizer <path>          Nemotron 3.5 ASR only: authenticated official
                                tokenizer.json sidecar. Required for the
                                published legacy GGUF; newly converted files
                                embed the same bytes and do not need this flag.
    --input <path>              mono WAV input (required for VAD, ASR, speaker,
                                language ID, denoise, separation, segmentation and F0; optional recorded
                                context audio for S2S — the explicit AEC bypass
                                path, FR-OP-60). For NKF-AEC this is the mic
                                signal and must be paired with --far-end.
                                For Mimi encode it is a mono WAV; for Mimi
                                decode it is a `VKRMCODE` v1 code container.
                                For DAC decode it is raw time-major
                                `[frames,n_codebooks]` little-endian u32.
                                For WavTokenizer decode it is one raw
                                little-endian u32 code per 75 Hz frame.
                                For NeuCodec base/distill decode it is one raw
                                little-endian u32 code per 50 Hz frame.
                                For X-Codec2 decode it is one raw little-endian
                                u32 code per 50 Hz frame.
                                For FunCodec decode it is raw frame-major
                                `[frames,num_quantizers]` u32le at 50 Hz;
                                `--num-quantizers` declares the residual-VQ
                                prefix width.
                                For SpeechTokenizer decode it is the same raw
                                frame-major residual-VQ matrix at 50 Hz with
                                a release maximum of eight codebooks.
                                For MOSS Audio Tokenizer decode it is raw
                                frame-major `[frames,num_quantizers]` u32le;
                                `--num-quantizers` declares the row width.
                                For MOSS-TTS it is a raw frame-major u32le
                                prompt matrix: `[rows,17]` for Nano or
                                `[rows,33]` for Base/v1.5, with text id first
                                and audio ids/pad sentinels after it. The model
                                GGUF does not bundle its tokenizer/template
                                assets, so raw text is never guessed here.
                                For SNAC encode it is a mono WAV; for decode it
                                is a `VKRSNAC1` v1 hierarchical code container.
                                For FocalCodec encode it is a 16 kHz mono WAV;
                                for decode it is a `VKRFOC01` v1 BSQ token
                                container pinned to the exact checkpoint.
                                For MeloTTS it is a `VKRMELO1` v1 bundle of
                                release-pinned phoneme/tone/language ids and
                                position-major BERT/JA-BERT features; raw text
                                is not inferred from the acoustic GGUF.
                                For BigVGAN, both HiFi-GAN variants, and Vocos
                                it is raw little-endian f32 feature data in
                                channel-major `[channels, frames]` order;
                                frames are derived exactly from length.
    --far-end <path>            nkf_aec only: sample-aligned far-end/reference
                                mono WAV. It must have exactly the same sample
                                rate and sample count as --input; no trim,
                                repeat, channel mixing, or resampling is done.
    --backend <name>            cpu | metal | cuda — backend for the model's hot
                                ops [default cpu]. Mirrors `bench --backend`:
                                honored only by architectures whose complete
                                learned-op set has an explicit backend route.
                                CPU-only or routed-partial engines reject an
                                unsupported selection loudly rather than
                                silently running on CPU (FR-EX-08).
                                For the honoring paths, metal/cuda need the CLI
                                built with that feature — an unavailable backend
                                fails loudly at inference, never silently on CPU.
    --compare <path>            speaker (CAM++ / X-vector / ECAPA-TDNN / WeSpeaker / TitaNet-L):
                                second WAV; prints the
                                cosine similarity of the two embeddings
                                (speaker_verify, FR-OP-81)
    --text <string>             text to synthesize (TTS) / normalize
                                (wetextprocessing) / the reply text CSM
                                speaks (S2S — caller-supplied, the model does
                                not generate text). For `kokoro` this is
                                PHONEME content, not graphemes: either a misaki
                                IPA string (each char looked up in the GGUF's
                                vokra.kokoro.phoneme_symbols table) or the
                                piper raw-id form (1 2 3 / 1,2,3 — content
                                ids only, sentinels are added). Kokoro has no
                                G2P bridge in-tree, so unmappable input is an
                                error rather than a silent drop.
                                For Charsiu this is an exact, whitespace-
                                delimited sequence from the GGUF's official
                                phone vocabulary (for example `SH IY`); no G2P,
                                case folding, or unknown-token substitution is
                                inferred.
    --voice <name>              kokoro only: voice name from the GGUF's
                                vokra.kokoro.voice_names. The name resolves,
                                but mapping it to a style row is NOT
                                implemented yet (M2-07-T02), so this cannot
                                synthesize on any GGUF — use --style.
    --style <path>              kokoro only: raw little-endian f32 style
                                vector, style_dim or 2*style_dim floats. The
                                2*style_dim form is upstream's full ref_s row
                                ([:style_dim] conditions the decoder,
                                [style_dim:] the prosody predictor). Takes
                                precedence over --voice.
    --length-scale <s>          kokoro or melotts: duration multiplier
                                (reciprocal of upstream `speed`) [default 1.0]
    --watermark-mode <mode>     audioseal only: `detect` (default) or `embed`.
    --watermark-variant <name>  audioseal only: `base` (default, non-causal)
                                or `streaming` (causal checkpoint evaluated as
                                one complete buffer; no chunk state implied).
    --watermark-message <bits>  audioseal embed only: exactly 16 ASCII 0/1
                                characters. Detect recovers and prints all 16.
    --watermark-alpha <gain>    audioseal embed only: finite watermark mix
                                gain in `pcm + gain * watermark` [default 1.0].
    --output <path>             WAV file for audio-producing tasks. An encoder-
                                only Wav2Vec2 GGUF writes time-major raw f32
                                `[frames, hidden]` features. Charsiu
                                writes alignment TSV; CT-Punc
                                writes exact UTF-8 restored text when present.
                                Speaker encoders write their native embedding
                                as raw little-endian f32 (`[dim]`).
                                Mimi requires it: code container for encode,
                                WAV for decode. DAC decode also requires a WAV
                                output. SNAC requires a `.vsc` code container
                                for encode and a WAV for decode. FocalCodec
                                requires a `.vfc` token container for encode and
                                a WAV for decode. Vocoders write
                                waveform WAV.
                                Multi-speaker SepFormer models derive
                                `<stem>.sourceN.wav` paths from this value.
    --tokens <path>             ct_punc only: `vokra-ct-punc-tsv-v1` UTF-8
                                side-car. Each record is `<u32 id><TAB><token>`;
                                tokens allow literal Unicode plus `\\`, `\t`,
                                `\n`, `\r`, and `\\u{HEX}` escapes. This keeps
                                caller token strings paired with the exact ids
                                passed to the model; no tokenizer is inferred.
    --token-ids <ids>           standalone bert_base/deberta_v2/deberta_v3, or
                                MusicGen Small/Melody conditional prompt:
                                comma-separated u32 ids (whitespace allowed).
                                No tokenizer or unknown-token substitution is
                                inferred. BERT --output writes row-major `[T,D]`
                                little-endian f32.
    --music-unconditional-token-ids <ids>
                                MusicGen Small/Melody only, REQUIRED: exact T5
                                token ids for the classifier-free null prompt.
                                The CLI never guesses how an empty prompt was
                                tokenized.
    --music-frames <N>          MusicGen Small/Melody only, REQUIRED: positive
                                number of 50 Hz EnCodec frames to generate.
    --max-new-frames <N>        MOSS-TTS only, REQUIRED: positive generation
                                cap (Nano audio frames; Base/v1.5 delayed rows).
    --music-seed <u64>          MusicGen Small/Melody sampling seed [default 0].
    --codec-mode <mode>         mimi: `encode` (mono WAV -> portable code
                                container) or `decode` (container -> WAV).
                                The v1 container pins time-major `[frame,cb]`
                                order, u32 little-endian codes, mono rate,
                                frame rate, topology, and codebook SHA-256.
                                dac: `decode` only, from raw time-major u32le
                                codes; unsupported encode is an explicit error.
                                wavtokenizer: `decode` only, from one raw u32le
                                code per frame. CPU and Metal cover the full
                                released token-to-waveform graph; encode is an
                                explicit error until its Encodec parity wave.
                                neucodec: `decode` only, from one raw u32le
                                code per frame. CPU and Metal cover the shared
                                base/distill token-to-waveform decoder; encode
                                remains an explicit error.
                                xcodec2: `decode` only, from one raw u32le code
                                per frame. CPU and Metal cover the complete
                                token-to-waveform decoder; encode remains an
                                explicit error.
                                funcodec: `decode` only, from a frame-major
                                residual-VQ u32le matrix. CPU and Metal cover
                                the complete token-to-waveform graph; encode
                                remains an explicit error.
                                speechtokenizer: `decode` only, from a
                                frame-major residual-VQ u32le matrix. CPU and
                                Metal cover the complete token-to-waveform
                                graph; encode remains an explicit error.
                                miocodec: `decode` only, from a VKRMIO01
                                container carrying FSQ codes, target samples,
                                and the required 128-d global embedding. CPU
                                and Metal cover the complete waveform decoder;
                                encode remains an explicit error.
                                snac: `encode` (CPU mono WAV -> versioned
                                stage-major container) or `decode` (container ->
                                WAV on CPU/Metal). Metal encode is an explicit
                                unsupported-operation error; it never falls
                                back to CPU.
                                focalcodec: `encode` (16 kHz mono WAV ->
                                versioned BSQ tokens) or `decode` (tokens ->
                                16 kHz WAV). CPU and Metal both run the complete
                                learned-op set; uncovered backends fail before
                                inference and never fall back to CPU.
                                moss_audio_tokenizer: `decode` only. Nano and
                                Full run their distinct complete decoders on
                                CPU or Metal; encode fails explicitly.
    --num-quantizers <N>       FunCodec, SpeechTokenizer or MOSS Audio
                                Tokenizer: positive row width of the raw
                                frame-major code matrix. Defaults to the
                                authenticated release maximum (FunCodec/Full
                                32, Nano 16, SpeechTokenizer 8).
    --bandwidth-id <0..3>       vocos-encodec-24khz or WavTokenizer only:
                                AdaLayerNorm condition. WavTokenizer defaults
                                to upstream's documented inference id 0.
    --beam-size <N>             ASR beam-search width (default 1 = greedy).
                                Honored for `voxtral` (n-best beam) and, with
                                --word-timestamps, for `whisper`. An arch whose
                                dispatch does not honor it errors out rather
                                than silently ignoring the flag (FR-EX-08).
    --length-penalty <α>        GNMT length-penalty exponent for beam search
                                (default 0.6). See `voxtral::BeamConfig`.
    --no-repeat-ngram <N>       Block repeated n-grams of length N during
                                beam search (default 0 = disabled).
    --word-timestamps           whisper only: emit per-word start/end times
                                (cross-attention DTW alignment, M4-20) after
                                the transcript, as `word<TAB>start<TAB>end`
                                lines. Requires the GGUF to carry
                                `vokra.whisper.alignment_heads`; a model
                                without them is an explicit error, never a
                                silent empty list.
    --language <code>           voxtral: transcription language for the
                                trained prompt's `lang:<code>` segment
                                (lowercase ISO 639, default `en`). Pass
                                `auto` to omit the segment and let the model
                                infer the language.
                                sbv2: which phonemizer + BERT encoder path
                                (JA DeBERTa v2 / EN DeBERTa v3) synthesizes
                                `--text` — `ja` (default when omitted) or
                                `en`; any other value is a loud error rather
                                than the silent any-non-`en`-is-JA default
                                the underlying TtsEngine adapter otherwise
                                applies (FR-EX-08).
    --bare-prompt               voxtral only: decode from the bare
                                soft-prefix + BOS layout instead of the
                                trained transcription prompt. Honest LM
                                continuation conditioned on the audio — NOT
                                a transcript (see AsrPromptLayout).
    --fixture-tokenizer         S2S only: swap the (T29-gated) embedded
                                tokenizer for the explicit fixture byte
                                tokenizer — host-only smoke, linguistically
                                meaningless output (never inferred, FR-EX-08)
    --interrupt-after <N>       S2S only: stream frames and barge-in
                                (M3-14 semantics) after N frames — the T19
                                interrupt demo path
    --deterministic             S2S only: temperature-0 sampling
                                (reproducible smoke / parity anchor)
    --duplex                    Moshi only: continuous full-duplex demo —
                                push mic frames from --input, pull model
                                frames, print the inner monologue (M4-06)
    --echo-sim <gain>           Moshi duplex only: mix the previous model
                                frame into the next mic frame at <gain>
                                (0..1) — the synthetic echo path the AEC
                                cancels (T26; without it the session runs
                                the explicit recorded-input AEC opt-out)
    --mimi <path>               Moshi only: standalone Mimi codec GGUF
                                (from `vokra-cli convert --model mimi`) —
                                binds the REAL codec on both duplex ends
                                instead of the synthesized bridge; a bind
                                failure is a hard error (FR-EX-08)
    --bert-ja <path>            sbv2 only, REQUIRED: the JA-path DeBERTa v2
                                BERT GGUF (`vokra-cli convert --model
                                deberta-v2`). `SbV2Model::from_gguf` needs
                                this alongside --model and --bert-en; a
                                missing value is a hard error, not a silent
                                skip (FR-EX-08).
    --bert-en <path>            sbv2 only, REQUIRED: the EN-path DeBERTa v3
                                BERT GGUF (`vokra-cli convert --model
                                deberta-v3`). See --bert-ja.
    --speaker-embedding <path>  sbv2 only, OPTIONAL: raw little-endian f32
                                external zero-shot speaker embedding
                                (Blocker 3). Length must equal the loaded
                                model's projection d_in (real ckpt: 512);
                                a wrong length is a loud error (FR-EX-08),
                                never a silent zero-pad/truncate. When
                                absent, the model's projection (if any)
                                is fed the deterministic all-zero
                                `[d_speaker]` default; on a legacy model
                                with no projection loaded, `speaker_id 0`
                                is used instead. Rejected loudly on every
                                non-sbv2 arch (FR-EX-08 — other archs
                                have their own speaker paths, e.g.
                                kokoro `--voice` / `--style`).
    -h, --help                  print this help
";

/// Parsed `run` arguments.
struct RunArgs {
    model: String,
    /// pyannote-diarization-only PyanNet dependency GGUF.
    segmentation_model: Option<String>,
    /// pyannote-diarization-only WeSpeaker dependency GGUF.
    embedding_model: Option<String>,
    /// MOSS-TTS-Nano-only codec sidecar.
    audio_tokenizer: Option<String>,
    /// Nemotron-ASR-only official tokenizer.json sidecar.
    tokenizer: Option<String>,
    input: Option<String>,
    text: Option<String>,
    output: Option<String>,
    /// CT-Punc-only versioned TSV pairing token ids and escaped UTF-8 tokens.
    tokens: Option<String>,
    /// Standalone BERT-family or MusicGen conditional comma-separated ids.
    token_ids: Option<String>,
    /// MusicGen-only classifier-free null-prompt T5 ids.
    music_unconditional_token_ids: Option<String>,
    /// MusicGen-only number of 50 Hz codec frames to generate.
    music_frames: Option<usize>,
    /// MOSS-TTS-Nano-only generation cap.
    max_new_frames: Option<usize>,
    /// MusicGen-only deterministic sampler seed. `None` means zero.
    music_seed: Option<u64>,
    /// Standalone codec direction. Required on Mimi/SNAC and on DAC decode;
    /// rejected for every non-codec architecture.
    codec_mode: Option<CodecMode>,
    /// MOSS Audio Tokenizer raw code-matrix width.
    num_quantizers: Option<usize>,
    /// Vocos Encodec AdaLayerNorm condition (`0..4`).
    bandwidth_id: Option<usize>,
    /// Backend the model's hot ops run on (mirrors `bench --backend`).
    /// Honored only by concrete engines whose dispatch binds
    /// `.with_backend(...)` for the complete declared learned-op set. Engines
    /// without that seam are rejected loudly in `main` rather than run
    /// silently on CPU (FR-EX-08). For honoring paths an *unavailable* backend
    /// still fails loudly at inference, never silently on CPU.
    backend: vokra_core::BackendKind,
    /// Speaker (CAM++ / X-vector) only: second WAV for cosine similarity.
    /// comparison. Any other task rejects the flag loudly (FR-EX-08).
    compare: Option<String>,
    /// NKF-AEC only: the far-end/reference WAV paired sample-for-sample with
    /// `input`. Rejected on every other task.
    far_end: Option<String>,
    /// Beam-search width (default 1 = greedy). Only honored for `voxtral`
    /// arch — other archs error out on `> 1` rather than silently ignoring
    /// (FR-EX-08).
    beam_size: usize,
    /// GNMT length-penalty exponent (default 0.6, per `BeamConfig`).
    length_penalty: f32,
    /// Block repeated n-grams of this length during beam search
    /// (default 0 = disabled).
    no_repeat_ngram: usize,
    /// Whisper only (cc-19): emit per-word timestamps after the transcript.
    /// Any other arch rejects the flag loudly (FR-EX-08).
    word_timestamps: bool,
    /// Voxtral only: the raw `--language` value. `None` = flag absent (keep
    /// the engine default, `en`); `Some("auto")` = omit the `lang:` segment
    /// entirely; `Some(code)` = that code.
    language: Option<String>,
    /// Voxtral only: opt into the bare soft-prefix + BOS layout.
    bare_prompt: bool,
    /// S2S: explicit fixture-tokenizer opt-in (host-only smoke).
    fixture_tokenizer: bool,
    /// S2S: barge-in after N streamed frames (T19 demo).
    interrupt_after: Option<usize>,
    /// S2S: deterministic (temperature-0) sampling.
    deterministic: bool,
    /// Moshi (M4-06): continuous full-duplex push/pull demo (T26).
    duplex: bool,
    /// Moshi duplex: synthetic echo attenuation — the previous model
    /// frame is mixed into the next mic frame at this gain, exercising
    /// the AEC path end to end (T26 合成 echo 経路).
    echo_sim: Option<f32>,
    /// Moshi only: standalone Mimi codec side-car GGUF — binds the real
    /// codec ends instead of the synthesized bridge (hard error on any
    /// bind failure; rejected loudly on every other arch — FR-EX-08).
    mimi: Option<String>,
    /// Kokoro only (cc-24): voice name from `vokra.kokoro.voice_names`.
    /// Rejected loudly on every other arch (FR-EX-08).
    voice: Option<String>,
    /// Kokoro only (cc-24): path to a raw little-endian f32 style vector
    /// (`style_dim` or `2·style_dim` floats). Takes precedence over
    /// `--voice`, matching `KokoroTts::synthesize_phonemes`.
    style: Option<String>,
    /// Kokoro only (cc-24): duration multiplier, the reciprocal of
    /// upstream's `speed`. Defaults to 1.0 = upstream default.
    length_scale: f32,
    /// SBV2 only (Task 38), REQUIRED for that arch: path to the JA-path
    /// DeBERTa v2 BERT GGUF `SbV2Model::from_gguf` needs alongside `--model`
    /// and `--bert-en`. Rejected loudly on every other arch (FR-EX-08).
    bert_ja: Option<String>,
    /// SBV2 only (Task 38), REQUIRED for that arch: path to the EN-path
    /// DeBERTa v3 BERT GGUF. See `bert_ja`.
    bert_en: Option<String>,
    /// SBV2 only (Blocker 3), OPTIONAL: path to a raw little-endian f32
    /// external zero-shot speaker embedding, forwarded to
    /// `SynthesisRequest::speaker_embedding` (its `Option<Vec<f32>>` shape).
    /// Rejected loudly on every non-sbv2 arch (FR-EX-08 — other archs
    /// have their own speaker paths, e.g. Kokoro's `--voice`/`--style`).
    speaker_embedding: Option<String>,
    /// AudioSeal task direction. Absent means detect on that arch; any
    /// presence is rejected on non-AudioSeal models.
    watermark_mode: Option<WatermarkMode>,
    /// AudioSeal official checkpoint variant. Absent means base.
    watermark_variant: Option<vokra_models::audioseal::AudiosealVariant>,
    /// Exact 16-bit AudioSeal message, required for embed and invalid for
    /// detect.
    watermark_message: Option<[u8; vokra_models::audioseal::NBITS]>,
    /// AudioSeal watermark mixing gain. Absent means 1.0.
    watermark_alpha: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodecMode {
    Encode,
    Decode,
}

impl CodecMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "encode" => Ok(Self::Encode),
            "decode" => Ok(Self::Decode),
            other => Err(format!(
                "unknown --codec-mode `{other}` (expected encode or decode)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WatermarkMode {
    Detect,
    Embed,
}

impl WatermarkMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "detect" => Ok(Self::Detect),
            "embed" => Ok(Self::Embed),
            other => Err(format!(
                "unknown --watermark-mode `{other}` (expected detect or embed)"
            )),
        }
    }
}

fn parse_watermark_message(value: &str) -> Result<[u8; vokra_models::audioseal::NBITS], String> {
    if value.len() != vokra_models::audioseal::NBITS
        || !value.bytes().all(|byte| matches!(byte, b'0' | b'1'))
    {
        return Err(format!(
            "--watermark-message must be exactly {} ASCII 0/1 characters (got `{value}`)",
            vokra_models::audioseal::NBITS
        ));
    }
    let mut message = [0u8; vokra_models::audioseal::NBITS];
    for (destination, byte) in message.iter_mut().zip(value.bytes()) {
        *destination = byte - b'0';
    }
    Ok(message)
}

fn parse_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model: Option<String> = None;
    let mut segmentation_model: Option<String> = None;
    let mut embedding_model: Option<String> = None;
    let mut audio_tokenizer: Option<String> = None;
    let mut tokenizer: Option<String> = None;
    let mut input: Option<String> = None;
    let mut text: Option<String> = None;
    let mut output: Option<String> = None;
    let mut tokens: Option<String> = None;
    let mut token_ids: Option<String> = None;
    let mut music_unconditional_token_ids: Option<String> = None;
    let mut music_frames: Option<usize> = None;
    let mut max_new_frames: Option<usize> = None;
    let mut music_seed: Option<u64> = None;
    let mut codec_mode: Option<CodecMode> = None;
    let mut num_quantizers: Option<usize> = None;
    let mut bandwidth_id: Option<usize> = None;
    let mut backend = vokra_core::BackendKind::Cpu;
    let mut compare: Option<String> = None;
    let mut far_end: Option<String> = None;
    // Beam-search defaults: greedy (beam_size = 1). Length-penalty 0.6 is
    // only meaningful when beam_size > 1; the default is arbitrary but
    // matches `voxtral::BeamConfig::with_beam_size` so the same value flows
    // through if the user only passes `--beam-size`.
    let mut beam_size: usize = 1;
    let mut length_penalty: f32 = 0.6;
    let mut no_repeat_ngram: usize = 0;
    let mut word_timestamps = false;
    let mut language: Option<String> = None;
    let mut bare_prompt = false;
    let mut fixture_tokenizer = false;
    let mut interrupt_after: Option<usize> = None;
    let mut deterministic = false;
    let mut duplex = false;
    let mut echo_sim: Option<f32> = None;
    let mut mimi: Option<String> = None;
    let mut voice: Option<String> = None;
    let mut style: Option<String> = None;
    let mut length_scale: f32 = 1.0;
    let mut bert_ja: Option<String> = None;
    let mut bert_en: Option<String> = None;
    let mut speaker_embedding: Option<String> = None;
    let mut watermark_mode: Option<WatermarkMode> = None;
    let mut watermark_variant: Option<vokra_models::audioseal::AudiosealVariant> = None;
    let mut watermark_message: Option<[u8; vokra_models::audioseal::NBITS]> = None;
    let mut watermark_alpha: Option<f32> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(args.get(i + 1).ok_or("--model requires a value")?.clone());
                i += 2;
            }
            "--segmentation-model" => {
                segmentation_model = Some(
                    args.get(i + 1)
                        .ok_or("--segmentation-model requires a GGUF path")?
                        .clone(),
                );
                i += 2;
            }
            "--embedding-model" => {
                embedding_model = Some(
                    args.get(i + 1)
                        .ok_or("--embedding-model requires a GGUF path")?
                        .clone(),
                );
                i += 2;
            }
            "--audio-tokenizer" => {
                audio_tokenizer = Some(
                    args.get(i + 1)
                        .ok_or("--audio-tokenizer requires a GGUF path")?
                        .clone(),
                );
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(
                    args.get(i + 1)
                        .ok_or("--tokenizer requires a tokenizer.json path")?
                        .clone(),
                );
                i += 2;
            }
            "--input" => {
                input = Some(args.get(i + 1).ok_or("--input requires a value")?.clone());
                i += 2;
            }
            "--text" => {
                text = Some(args.get(i + 1).ok_or("--text requires a value")?.clone());
                i += 2;
            }
            "--output" => {
                output = Some(args.get(i + 1).ok_or("--output requires a value")?.clone());
                i += 2;
            }
            "--tokens" => {
                tokens = Some(args.get(i + 1).ok_or("--tokens requires a path")?.clone());
                i += 2;
            }
            "--token-ids" => {
                token_ids = Some(
                    args.get(i + 1)
                        .ok_or("--token-ids requires a comma-separated u32 list")?
                        .clone(),
                );
                i += 2;
            }
            "--music-unconditional-token-ids" => {
                music_unconditional_token_ids = Some(
                    args.get(i + 1)
                        .ok_or(
                            "--music-unconditional-token-ids requires a comma-separated u32 list",
                        )?
                        .clone(),
                );
                i += 2;
            }
            "--music-frames" => {
                let value = args.get(i + 1).ok_or("--music-frames requires a value")?;
                let frames = value
                    .parse::<usize>()
                    .map_err(|error| format!("--music-frames must be an integer: {error}"))?;
                if frames == 0 {
                    return Err("--music-frames must be positive".to_owned());
                }
                music_frames = Some(frames);
                i += 2;
            }
            "--max-new-frames" => {
                let value = args.get(i + 1).ok_or("--max-new-frames requires a value")?;
                let frames = value
                    .parse::<usize>()
                    .map_err(|error| format!("--max-new-frames must be an integer: {error}"))?;
                if frames == 0 {
                    return Err("--max-new-frames must be positive".to_owned());
                }
                max_new_frames = Some(frames);
                i += 2;
            }
            "--music-seed" => {
                let value = args.get(i + 1).ok_or("--music-seed requires a value")?;
                music_seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|error| format!("--music-seed must be u64: {error}"))?,
                );
                i += 2;
            }
            "--codec-mode" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--codec-mode requires encode or decode")?;
                codec_mode = Some(CodecMode::parse(value)?);
                i += 2;
            }
            "--num-quantizers" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--num-quantizers requires a positive integer")?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|error| format!("--num-quantizers must be an integer: {error}"))?;
                if parsed == 0 {
                    return Err("--num-quantizers must be positive".to_owned());
                }
                num_quantizers = Some(parsed);
                i += 2;
            }
            "--bandwidth-id" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--bandwidth-id requires an integer in 0..3")?;
                let parsed: usize = value
                    .parse()
                    .map_err(|error| format!("--bandwidth-id must be an integer: {error}"))?;
                if parsed >= 4 {
                    return Err(format!("--bandwidth-id must be in 0..3 (got {parsed})"));
                }
                bandwidth_id = Some(parsed);
                i += 2;
            }
            "--backend" => {
                let v = args.get(i + 1).ok_or("--backend requires a value")?;
                backend = crate::bench::parse_backend(v)?;
                i += 2;
            }
            "--compare" => {
                compare = Some(args.get(i + 1).ok_or("--compare requires a value")?.clone());
                i += 2;
            }
            "--far-end" => {
                far_end = Some(args.get(i + 1).ok_or("--far-end requires a value")?.clone());
                i += 2;
            }
            "--beam-size" => {
                let v = args.get(i + 1).ok_or("--beam-size requires a value")?;
                beam_size = v
                    .parse()
                    .map_err(|e| format!("--beam-size must be an unsigned integer: {e}"))?;
                if beam_size == 0 {
                    return Err("--beam-size must be >= 1".to_owned());
                }
                i += 2;
            }
            "--length-penalty" => {
                let v = args.get(i + 1).ok_or("--length-penalty requires a value")?;
                length_penalty = v
                    .parse()
                    .map_err(|e| format!("--length-penalty must be a float: {e}"))?;
                if !length_penalty.is_finite() || length_penalty < 0.0 {
                    return Err(format!(
                        "--length-penalty must be a non-negative finite float (got {length_penalty})"
                    ));
                }
                i += 2;
            }
            "--no-repeat-ngram" => {
                let v = args
                    .get(i + 1)
                    .ok_or("--no-repeat-ngram requires a value")?;
                no_repeat_ngram = v
                    .parse()
                    .map_err(|e| format!("--no-repeat-ngram must be an unsigned integer: {e}"))?;
                i += 2;
            }
            "--word-timestamps" => {
                word_timestamps = true;
                i += 1;
            }
            "--language" => {
                let v = args.get(i + 1).ok_or("--language requires a value")?;
                if v.is_empty() {
                    return Err("--language must not be empty (use `auto` to omit the \
                                prompt's lang: segment)"
                        .to_owned());
                }
                language = Some(v.clone());
                i += 2;
            }
            "--bare-prompt" => {
                bare_prompt = true;
                i += 1;
            }
            "--fixture-tokenizer" => {
                fixture_tokenizer = true;
                i += 1;
            }
            "--interrupt-after" => {
                let v = args
                    .get(i + 1)
                    .ok_or("--interrupt-after requires a value")?;
                interrupt_after =
                    Some(v.parse().map_err(|e| {
                        format!("--interrupt-after must be an unsigned integer: {e}")
                    })?);
                i += 2;
            }
            "--deterministic" => {
                deterministic = true;
                i += 1;
            }
            "--duplex" => {
                duplex = true;
                i += 1;
            }
            "--echo-sim" => {
                let v = args.get(i + 1).ok_or("--echo-sim requires a value")?;
                let g: f32 = v
                    .parse()
                    .map_err(|e| format!("--echo-sim must be a float gain: {e}"))?;
                if !g.is_finite() || !(0.0..=1.0).contains(&g) {
                    return Err(format!("--echo-sim gain must be in [0, 1] (got {g})"));
                }
                echo_sim = Some(g);
                i += 2;
            }
            "--mimi" => {
                let v = args.get(i + 1).ok_or("--mimi requires a GGUF path")?;
                mimi = Some(v.clone());
                i += 2;
            }
            "--voice" => {
                let v = args.get(i + 1).ok_or("--voice requires a name")?;
                if v.is_empty() {
                    return Err("--voice must not be empty".to_owned());
                }
                voice = Some(v.clone());
                i += 2;
            }
            "--style" => {
                let v = args.get(i + 1).ok_or("--style requires a path")?;
                style = Some(v.clone());
                i += 2;
            }
            "--length-scale" => {
                let v = args.get(i + 1).ok_or("--length-scale requires a value")?;
                length_scale = v
                    .parse()
                    .map_err(|e| format!("--length-scale must be a float: {e}"))?;
                if !length_scale.is_finite() || length_scale <= 0.0 {
                    return Err(format!(
                        "--length-scale must be a positive finite float (got {length_scale})"
                    ));
                }
                i += 2;
            }
            "--bert-ja" => {
                let v = args.get(i + 1).ok_or("--bert-ja requires a path")?;
                bert_ja = Some(v.clone());
                i += 2;
            }
            "--bert-en" => {
                let v = args.get(i + 1).ok_or("--bert-en requires a path")?;
                bert_en = Some(v.clone());
                i += 2;
            }
            "--speaker-embedding" => {
                let v = args
                    .get(i + 1)
                    .ok_or("--speaker-embedding requires a path")?;
                speaker_embedding = Some(v.clone());
                i += 2;
            }
            "--watermark-mode" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--watermark-mode requires detect or embed")?;
                watermark_mode = Some(WatermarkMode::parse(value)?);
                i += 2;
            }
            "--watermark-variant" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--watermark-variant requires base or streaming")?;
                watermark_variant = Some(match value.as_str() {
                    "base" => vokra_models::audioseal::AudiosealVariant::Base,
                    "streaming" => vokra_models::audioseal::AudiosealVariant::Streaming,
                    other => {
                        return Err(format!(
                            "unknown --watermark-variant `{other}` (expected base or streaming)"
                        ));
                    }
                });
                i += 2;
            }
            "--watermark-message" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--watermark-message requires exactly 16 bits")?;
                watermark_message = Some(parse_watermark_message(value)?);
                i += 2;
            }
            "--watermark-alpha" => {
                let value = args
                    .get(i + 1)
                    .ok_or("--watermark-alpha requires a finite float")?;
                let alpha = value
                    .parse::<f32>()
                    .map_err(|error| format!("--watermark-alpha must be a float: {error}"))?;
                if !alpha.is_finite() {
                    return Err(format!("--watermark-alpha must be finite (got {alpha})"));
                }
                watermark_alpha = Some(alpha);
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    Ok(RunArgs {
        model: model.ok_or("--model is required")?,
        segmentation_model,
        embedding_model,
        audio_tokenizer,
        tokenizer,
        input,
        text,
        output,
        tokens,
        token_ids,
        music_unconditional_token_ids,
        music_frames,
        max_new_frames,
        music_seed,
        codec_mode,
        num_quantizers,
        bandwidth_id,
        backend,
        compare,
        far_end,
        beam_size,
        length_penalty,
        no_repeat_ngram,
        word_timestamps,
        language,
        bare_prompt,
        fixture_tokenizer,
        interrupt_after,
        deterministic,
        duplex,
        echo_sim,
        mimi,
        voice,
        style,
        length_scale,
        bert_ja,
        bert_en,
        speaker_embedding,
        watermark_mode,
        watermark_variant,
        watermark_message,
        watermark_alpha,
    })
}

/// Human name for a resolved `--backend`, matching the lowercase spellings
/// [`crate::bench::parse_backend`] accepts (used only in diagnostics).
fn backend_flag_name(backend: vokra_core::BackendKind) -> String {
    use vokra_core::BackendKind;
    match backend {
        BackendKind::Cpu => "cpu".to_owned(),
        BackendKind::Metal => "metal".to_owned(),
        BackendKind::Cuda => "cuda".to_owned(),
        BackendKind::Vulkan => "vulkan".to_owned(),
        // `BackendKind` is `#[non_exhaustive]`; the CLI parser only yields the
        // four above today, so fall back to the debug spelling for any future
        // variant rather than failing to name it.
        other => format!("{other:?}").to_lowercase(),
    }
}

/// For a task whose `run` dispatch does **not** thread `--backend` into its
/// engine, the human label of that engine; `None` for the backend-honoring
/// paths.
///
/// Engines listed here bind without `.with_backend(...)`, so a non-CPU
/// selection would be a silent CPU no-op. [`main`] rejects that combination
/// loudly (FR-EX-08). The match is exhaustive; keep it in lock-step with the
/// concrete binders.
fn cpu_only_engine_label(task: ModelTask) -> Option<&'static str> {
    match task {
        // Wave G (2026-08-15): these routed arches bind concrete models
        // without a `.with_backend(...)` seam, so a non-CPU selection would
        // be a silent CPU run. NSNet2 later left this group and is classified
        // in the mixed denoise arm below.
        ModelTask::KwsOpenwakeword => Some("openWakeWord keyword spotting"),
        // Both routed denoise arches bind the selected backend themselves.
        ModelTask::Denoise => None,
        ModelTask::F0Crepe => Some("CREPE F0 (pitch) extraction"),
        ModelTask::AlignCharsiu => Some("Charsiu forced alignment"),
        ModelTask::TextNormalize => Some("WeTextProcessing normalization"),
        ModelTask::CtPunc => Some("CT-Punc punctuation restoration"),
        ModelTask::TextEncoder => None,
        ModelTask::VocoderBigVgan => None,
        ModelTask::VocoderHifiGan => None,
        ModelTask::S2s => Some("CSM speech-to-speech"),
        // SBV2 (Task 38): `SbV2Model` / `DebertaV2Encoder` / `DebertaV3Encoder`
        // have no `.with_backend(...)` seam yet (no Metal/CUDA Compute-seam
        // wiring for this arch today) — a non-CPU `--backend` would silently
        // run on the CPU, same class of gap as VAD/CSM/Moshi above.
        ModelTask::Sbv2 => Some("SBV2 (Style-Bert-VITS2 v2) TTS"),
        // Backend-honoring: the concrete engine binds `.with_backend(...)`, so
        // a non-CPU backend reaches the hot ops (and an unavailable one fails
        // loudly at inference — the existing FR-EX-08 posture). The guard must
        // NOT fire for these.
        ModelTask::Asr
        | ModelTask::AsrVoxtral
        | ModelTask::AsrNemotron
        | ModelTask::SpeechFeaturesWav2Vec2
        | ModelTask::Vad
        | ModelTask::VadFirered
        | ModelTask::VadTen
        | ModelTask::F0Rmvpe
        | ModelTask::F0Fcpe
        | ModelTask::SmartTurn
        | ModelTask::AudioClassificationAst
        | ModelTask::Utmos
        | ModelTask::Dnsmos
        | ModelTask::Nisqa
        | ModelTask::Segment
        | ModelTask::DiarizationPyannote
        | ModelTask::Separation
        | ModelTask::Tts
        | ModelTask::TtsKokoro
        | ModelTask::TtsMelo
        | ModelTask::Speaker
        | ModelTask::LangId
        | ModelTask::AudioQualityAudiobox
        | ModelTask::EmotionClassification
        | ModelTask::DeepfakeClassification
        | ModelTask::WatermarkAudioseal
        | ModelTask::MimiCodec
        | ModelTask::DacCodec
        | ModelTask::WavTokenizerCodec
        | ModelTask::NeuCodec
        | ModelTask::XCodec2
        | ModelTask::FunCodec
        | ModelTask::SpeechTokenizer
        | ModelTask::MioCodec
        | ModelTask::SnacCodec
        | ModelTask::FocalCodec
        | ModelTask::MossAudioTokenizerCodec
        | ModelTask::TtsMossNano
        | ModelTask::TtsMossDelay
        | ModelTask::TtsMossVoiceGenerator
        | ModelTask::S2sDuplex
        | ModelTask::VadFsmn
        | ModelTask::VocoderVocos
        | ModelTask::AecNkf
        | ModelTask::MusicGeneration => None,
        // Bench-only tasks — unreachable from `run` (each hits its own explicit
        // rejection in `main`'s `match`). Returning `None` lets that more
        // specific error fire instead of a backend complaint.
        ModelTask::MelFrontend | ModelTask::Cosyvoice2Synthetic => None,
    }
}

/// Entry point for `vokra-cli run`.
pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let a = parse_args(args)?;
    let hint = a
        .fixture_tokenizer
        .then_some(engine::TaskHint::CsmFixtureTokenizer);
    let (session, task) =
        engine::load_session_with_backend_and_mimi(&a.model, a.backend, hint, a.mimi.as_deref())?;

    if a.tokenizer.is_some() && task != ModelTask::AsrNemotron {
        return Err(
            "run: --tokenizer is only supported for the nemotron_asr_streaming arch; other model tokenizers must be embedded by their audited converter"
                .to_owned(),
        );
    }

    if (a.segmentation_model.is_some() || a.embedding_model.is_some())
        && task != ModelTask::DiarizationPyannote
    {
        return Err(
            "run: --segmentation-model / --embedding-model are only supported for the \
             pyannote-speaker-diarization arch"
                .to_owned(),
        );
    }
    if task == ModelTask::DiarizationPyannote {
        if a.segmentation_model.is_none() || a.embedding_model.is_none() {
            return Err(
                "run (pyannote diarization): both --segmentation-model <pyannote-segmentation.gguf> \
                 and --embedding-model <pyannote-wespeaker.gguf> are required; the weightless \
                 pipeline GGUF never downloads or guesses dependencies"
                    .to_owned(),
            );
        }
    }
    if a.audio_tokenizer.is_some()
        && !matches!(
            task,
            ModelTask::TtsMossNano | ModelTask::TtsMossDelay | ModelTask::TtsMossVoiceGenerator
        )
    {
        return Err(
            "run: --audio-tokenizer is only supported for authenticated MOSS-TTS releases"
                .to_owned(),
        );
    }
    if a.max_new_frames.is_some()
        && !matches!(
            task,
            ModelTask::TtsMossNano | ModelTask::TtsMossDelay | ModelTask::TtsMossVoiceGenerator
        )
    {
        return Err(
            "run: --max-new-frames is only supported for authenticated MOSS-TTS releases"
                .to_owned(),
        );
    }

    // `--compare` belongs to the speaker task only. Reject it on
    // every other arch rather than silently ignoring the flag (FR-EX-08).
    if a.compare.is_some() && task != ModelTask::Speaker {
        return Err(
            "run: --compare is only supported for speaker-embedding arches — \
             it compares two speaker embeddings (FR-OP-81)"
                .to_owned(),
        );
    }
    if a.far_end.is_some() && task != ModelTask::AecNkf {
        return Err(
            "run: --far-end is only supported for the nkf_aec arch — it is the \
             sample-aligned far-end/reference stream paired with the --input mic WAV"
                .to_owned(),
        );
    }
    if a.tokens.is_some() && task != ModelTask::CtPunc {
        return Err(
            "run: --tokens is only supported for the ct_punc arch — it supplies the \
             versioned, paired token-string/token-id input to CtPunc::restore"
                .to_owned(),
        );
    }
    if a.token_ids.is_some() && task != ModelTask::TextEncoder && task != ModelTask::MusicGeneration
    {
        return Err(
            "run: --token-ids is only supported for standalone bert_base/deberta_v2/deberta_v3 or musicgen arches"
                .to_owned(),
        );
    }
    if (a.music_unconditional_token_ids.is_some()
        || a.music_frames.is_some()
        || a.music_seed.is_some())
        && task != ModelTask::MusicGeneration
    {
        return Err(
            "run: --music-unconditional-token-ids / --music-frames / --music-seed are only supported for the musicgen arch"
                .to_owned(),
        );
    }
    if a.codec_mode.is_some()
        && task != ModelTask::MimiCodec
        && task != ModelTask::DacCodec
        && task != ModelTask::WavTokenizerCodec
        && task != ModelTask::NeuCodec
        && task != ModelTask::XCodec2
        && task != ModelTask::FunCodec
        && task != ModelTask::SpeechTokenizer
        && task != ModelTask::MioCodec
        && task != ModelTask::SnacCodec
        && task != ModelTask::FocalCodec
        && task != ModelTask::MossAudioTokenizerCodec
    {
        return Err(
            "run: --codec-mode is only supported for standalone mimi/dac/wavtokenizer/neucodec/xcodec2/funcodec/speechtokenizer/miocodec/snac/focalcodec/moss_audio_tokenizer arches"
                .to_owned(),
        );
    }
    if a.num_quantizers.is_some()
        && task != ModelTask::FunCodec
        && task != ModelTask::SpeechTokenizer
        && task != ModelTask::MossAudioTokenizerCodec
    {
        return Err(
            "run: --num-quantizers is only supported for funcodec, speechtokenizer or moss_audio_tokenizer arches"
                .to_owned(),
        );
    }
    if a.bandwidth_id.is_some()
        && task != ModelTask::VocoderVocos
        && task != ModelTask::WavTokenizerCodec
    {
        return Err(
            "run: --bandwidth-id is only supported for the vocos encodec_24khz variant or wavtokenizer"
                .to_owned(),
        );
    }
    if (a.watermark_mode.is_some()
        || a.watermark_variant.is_some()
        || a.watermark_message.is_some()
        || a.watermark_alpha.is_some())
        && task != ModelTask::WatermarkAudioseal
    {
        return Err(
            "run: --watermark-mode / --watermark-variant / --watermark-message / \
             --watermark-alpha are only supported for the audioseal_real_weight arch"
                .to_owned(),
        );
    }
    // `--word-timestamps` is a Whisper-only surface (cross-attention DTW,
    // M4-20); `--bare-prompt` is a Voxtral-only prompt-layout knob. Each is
    // rejected off its own arch rather than silently ignored (FR-EX-08).
    if a.word_timestamps && task != ModelTask::Asr {
        return Err(
            "run: --word-timestamps is only supported for the whisper arches (`whisper`, \
             `distil-whisper`, `kotoba-whisper` — the distilled checkpoints share the \
             identical decoder topology and route to this same ASR task) — it needs the \
             cross-attention alignment heads (M4-20). Voxtral has no such alignment."
                .to_owned(),
        );
    }
    if a.bare_prompt && task != ModelTask::AsrVoxtral {
        return Err(
            "run: --bare-prompt is only supported for the voxtral arch — it selects the bare \
             soft-prefix + BOS layout instead of the trained transcription prompt"
                .to_owned(),
        );
    }
    // `--language` is shared by two archs with distinct meanings: Voxtral's
    // transcription-prompt `lang:` segment, and SBV2's JA/EN phonemizer +
    // BERT-encoder routing (Task 38). Rejected off every other arch rather
    // than silently ignored (FR-EX-08).
    if a.language.is_some()
        && task != ModelTask::AsrVoxtral
        && task != ModelTask::AsrNemotron
        && task != ModelTask::Sbv2
    {
        return Err(
            "run: --language is only supported for the voxtral arch (the trained \
             transcription prompt's `lang:` segment), nemotron_asr_streaming (the released \
             prompt-id map), or the sbv2 arch (JA/EN phonemizer + BERT-encoder routing)"
                .to_owned(),
        );
    }
    // `--bert-ja` / `--bert-en` are SBV2-only side-car GGUFs (Task 38): the
    // DeBERTa v2 (JA) / v3 (EN) BERT encoders `SbV2Model::from_gguf` requires
    // alongside the main model file. Rejected off every other arch rather
    // than silently ignored (FR-EX-08).
    if (a.bert_ja.is_some() || a.bert_en.is_some()) && task != ModelTask::Sbv2 {
        return Err(
            "run: --bert-ja / --bert-en are only supported for the sbv2 arch — they load the \
             DeBERTa v2 (JA) / v3 (EN) BERT side-car GGUFs SbV2Model::from_gguf requires"
                .to_owned(),
        );
    }
    // `--speaker-embedding` is Blocker 3's SBV2-only external zero-shot
    // speaker path. Rejected off every other arch rather than silently
    // ignored — other archs have their own speaker paths
    // (Kokoro `--voice`/`--style`, CAM++ single-input embedding, ...),
    // and silently dropping caller-supplied data on the floor here would
    // produce plausible-looking-but-wrong-speaker audio (FR-EX-08).
    if a.speaker_embedding.is_some() && task != ModelTask::Sbv2 {
        return Err(
            "run: --speaker-embedding is only supported for the sbv2 arch — it is Blocker 3's \
             external zero-shot speaker input (SBV2's `enc_p.encoder.spk_emb_linear` projects \
             the caller-supplied 512-d vector into the text-encoder hidden width). Other \
             archs use their own speaker paths (e.g. kokoro --voice / --style)"
                .to_owned(),
        );
    }
    // `--voice` / `--style` / `--length-scale` are Kokoro style-conditioning
    // knobs (cc-24). Rejected off that arch rather than silently ignored
    // (FR-EX-08) — a dropped style would change the speaker without saying so.
    if (a.voice.is_some() || a.style.is_some()) && task != ModelTask::TtsKokoro {
        return Err(
            "run: --voice / --style are only supported for the kokoro arch — they select the \
             style vector that conditions its decoder and prosody predictor"
                .to_owned(),
        );
    }
    if a.length_scale != 1.0 && task != ModelTask::TtsKokoro && task != ModelTask::TtsMelo {
        return Err(
            "run: --length-scale is only supported for the kokoro or melotts arch \
             (piper-plus exposes its own scales through the engine API, not the CLI)"
                .to_owned(),
        );
    }
    // `--backend metal|cuda|vulkan` reaches the model's hot ops only on the
    // arches whose `run` dispatch binds the concrete engine
    // `.with_backend(...)`. CPU-only engines bind without a backend and run on
    // CPU regardless. A non-CPU `--backend` for one of them would be a silent
    // no-op, so reject it loudly rather than misreport where the model ran
    // (FR-EX-08). Honoring arches keep their existing posture: an unavailable
    // backend fails loudly at inference, never on CPU.
    if a.backend != vokra_core::BackendKind::Cpu {
        if let Some(engine_label) = cpu_only_engine_label(task) {
            return Err(format!(
                "run: --backend {backend} is not supported for this model — the \
                 {engine_label} engine runs on the CPU regardless (its dispatch is not \
                 backend-parameterised). This architecture has no complete \
                 --backend route; a non-CPU backend here would \
                 silently run on the CPU (FR-EX-08). Re-run with --backend cpu (or omit it).",
                backend = backend_flag_name(a.backend),
            ));
        }
    }

    match task {
        // FSMN-VAD (Wave G) shares this arm verbatim: its binder implements
        // the same `VadEngine` trait Silero does and the dispatch injects it
        // into the same session slot, so the Silero path stays byte-identical.
        ModelTask::Vad | ModelTask::VadFsmn | ModelTask::VadFirered | ModelTask::VadTen => {
            let path = a
                .input
                .as_deref()
                .ok_or("run (VAD): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let probs = run_vad(&session, &clip.samples, clip.sample_rate)?;
            let n = probs.len();
            let speech = probs.iter().filter(|&&p| p >= 0.5).count();
            let mean = if n == 0 {
                0.0
            } else {
                probs.iter().sum::<f32>() / n as f32
            };
            println!("vad: {n} frames, speech_frames={speech}, mean_prob={mean:.4}");
        }
        ModelTask::Asr => {
            let path = a
                .input
                .as_deref()
                .ok_or("run (ASR): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let arch = session
                .gguf()
                .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str());
            if matches!(
                arch,
                Some(vokra_models::wav2vec2_ctc::ARCH)
                    | Some(vokra_models::hubert::ARCH)
                    | Some(vokra_models::data2vec_audio::ARCH)
            ) && clip.sample_rate != 16_000
            {
                return Err(format!(
                    "run ({arch:?} ASR): {path} is {} Hz, expected 16000 Hz — resample offline first (FR-EX-08: never a silent resample)",
                    clip.sample_rate
                ));
            }
            // Whisper's beam-search entry point lives on the concrete
            // `WhisperAsr` (n-best + alignment), not on the shared
            // `AsrEngine` trait the session injects. `--word-timestamps`
            // therefore routes through the beam surface (cc-19); without
            // it, a beam-only flag on the Whisper path would be silently
            // dropped, so it is a hard error instead (FR-EX-08).
            if a.word_timestamps {
                run_whisper_word_timestamps(&a.model, a.backend, &clip.samples, &a)?;
                return Ok(ExitCode::SUCCESS);
            }
            if a.beam_size > 1 || a.no_repeat_ngram > 0 {
                return Err(
                    "run (ASR): --beam-size > 1 / --no-repeat-ngram are only honored on the \
                     whisper path together with --word-timestamps (which routes through the \
                     beam/alignment surface), or on the `voxtral` arch. Add \
                     --word-timestamps, or run a voxtral GGUF."
                        .to_owned(),
                );
            }
            // Length-penalty defaults to 0.6 (matching `BeamConfig`). A
            // user who explicitly set --length-penalty AND beam_size = 1
            // is passing a flag that has no effect (greedy ignores the
            // penalty); we detect that combination by comparing to the
            // parser default. Rather than surfacing that as a hard error
            // (which would trip normal users who explored the flag), we
            // print an informational note and continue.
            #[allow(clippy::float_cmp)]
            if a.beam_size == 1 && a.length_penalty != 0.6 {
                eprintln!(
                    "run (ASR): note — --length-penalty is only applied when --beam-size > 1 \
                     (greedy ignores the length penalty)."
                );
            }
            let text = run_asr(&session, &clip.samples)?;
            println!("asr: {text}");
        }
        ModelTask::AsrNemotron => {
            run_nemotron_asr(&session, &a)?;
        }
        ModelTask::SpeechFeaturesWav2Vec2 => {
            let path = a
                .input
                .as_deref()
                .ok_or("run (Wav2Vec2 features): --input <16k-mono.wav> is required")?;
            if a.beam_size != 1
                || a.no_repeat_ngram != 0
                || a.length_penalty.to_bits() != 0.6f32.to_bits()
            {
                return Err(
                    "run (Wav2Vec2 features): beam-search flags are not applicable to the encoder-only checkpoint"
                        .to_owned(),
                );
            }
            let clip = wav::read_wav(path)?;
            if clip.sample_rate != 16_000 {
                return Err(format!(
                    "run (Wav2Vec2 features): {path} is {} Hz, expected 16000 Hz — resample offline first (FR-EX-08: never a silent resample)",
                    clip.sample_rate
                ));
            }
            let model = vokra_models::wav2vec2_ctc::Wav2Vec2Ctc::from_file(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let (features, frames) = model
                .encode_features(&clip.samples)
                .map_err(|error| error.to_string())?;
            let hidden = model.config().hidden_size;
            if let Some(output) = a.output.as_deref() {
                let bytes = features
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                std::fs::write(output, bytes).map_err(|error| {
                    format!("run (Wav2Vec2 features): --output {output}: {error}")
                })?;
                println!("wav2vec2-features: {frames} frames x {hidden} f32 -> {output}");
            } else {
                println!(
                    "wav2vec2-features: {frames} frames x {hidden} f32 (no --output; features discarded)"
                );
            }
        }
        ModelTask::AsrVoxtral => {
            run_voxtral(&session, &a)?;
        }
        ModelTask::Tts => {
            let text = a
                .text
                .as_deref()
                .ok_or("run (TTS): --text <string> is required")?;
            let audio = session.tts().synthesize(text).map_err(|e| e.to_string())?;
            emit_audio(
                "tts",
                &audio.samples,
                audio.sample_rate,
                a.output.as_deref(),
            )?;
        }
        ModelTask::MusicGeneration => {
            run_musicgen(&a)?;
        }
        ModelTask::TtsMossNano => {
            run_moss_tts_nano(&session, &a)?;
        }
        ModelTask::TtsMossDelay => {
            run_moss_tts_delay(&session, &a)?;
        }
        ModelTask::TtsMossVoiceGenerator => {
            run_moss_voice_generator(&session, &a)?;
        }
        ModelTask::TtsKokoro => {
            run_kokoro(&a)?;
        }
        ModelTask::S2sDuplex => {
            run_s2s_duplex(&session, &a)?;
        }
        ModelTask::S2s => {
            run_s2s(&session, &a)?;
        }
        ModelTask::Speaker => {
            run_speaker(&session, &a)?;
        }
        ModelTask::LangId => {
            run_lang_id(&session, &a)?;
        }
        ModelTask::AudioQualityAudiobox => {
            run_audiobox_aesthetics(&session, &a)?;
        }
        ModelTask::EmotionClassification => {
            run_emotion2vec(&session, &a)?;
        }
        ModelTask::DeepfakeClassification => {
            run_deepfake_detection(&session, &a)?;
        }
        ModelTask::WatermarkAudioseal => {
            run_audioseal(&session, &a)?;
        }
        ModelTask::Sbv2 => {
            run_sbv2(&session, &a)?;
        }
        ModelTask::TtsMelo => {
            run_melotts(&session, &a)?;
        }
        // ---- Wave G (2026-08-15) — newly routed real forwards ------------
        ModelTask::Denoise => {
            run_denoise(&session, &a)?;
        }
        ModelTask::Separation => {
            run_separation(&session, &a)?;
        }
        ModelTask::Segment => {
            run_segment(&a)?;
        }
        ModelTask::DiarizationPyannote => {
            run_pyannote_diarization(&a)?;
        }
        ModelTask::F0Rmvpe => {
            run_f0_rmvpe(&a)?;
        }
        ModelTask::F0Fcpe => {
            run_f0_fcpe(&a)?;
        }
        ModelTask::F0Crepe => {
            run_f0_crepe(&a)?;
        }
        ModelTask::AlignCharsiu => {
            run_charsiu_align(&session, &a)?;
        }
        ModelTask::TextNormalize => {
            run_text_normalize(&session, &a)?;
        }
        ModelTask::KwsOpenwakeword => {
            run_openwakeword(&session, &a)?;
        }
        ModelTask::SmartTurn => {
            run_smart_turn(&session, &a)?;
        }
        ModelTask::AudioClassificationAst => {
            run_ast_classification(&session, &a)?;
        }
        ModelTask::Utmos => {
            run_utmos(&session, &a)?;
        }
        ModelTask::Dnsmos => {
            run_dnsmos(&session, &a)?;
        }
        ModelTask::Nisqa => {
            run_nisqa(&session, &a)?;
        }
        ModelTask::AecNkf => {
            run_nkf_aec(&session, &a)?;
        }
        ModelTask::CtPunc => {
            run_ct_punc(&session, &a)?;
        }
        ModelTask::TextEncoder => {
            run_bert_encoder(&session, &a)?;
        }
        ModelTask::MimiCodec => {
            run_mimi_codec(&session, &a)?;
        }
        ModelTask::DacCodec => {
            run_dac_codec(&session, &a)?;
        }
        ModelTask::WavTokenizerCodec => {
            run_wavtokenizer_codec(&session, &a)?;
        }
        ModelTask::NeuCodec => {
            run_neucodec_codec(&session, &a)?;
        }
        ModelTask::XCodec2 => {
            run_xcodec2_codec(&session, &a)?;
        }
        ModelTask::FunCodec => {
            run_funcodec_codec(&session, &a)?;
        }
        ModelTask::SpeechTokenizer => {
            run_speechtokenizer_codec(&session, &a)?;
        }
        ModelTask::MioCodec => {
            run_miocodec(&session, &a)?;
        }
        ModelTask::SnacCodec => {
            run_snac_codec(&session, &a)?;
        }
        ModelTask::FocalCodec => {
            run_focalcodec(&session, &a)?;
        }
        ModelTask::MossAudioTokenizerCodec => {
            run_moss_audio_tokenizer_codec(&session, &a)?;
        }
        ModelTask::VocoderBigVgan => {
            run_bigvgan(&session, &a)?;
        }
        ModelTask::VocoderHifiGan => {
            run_hifigan(&session, &a)?;
        }
        ModelTask::VocoderVocos => {
            run_vocos(&session, &a)?;
        }
        // `mel-frontend` is a bench-only task (M2-04-T11) — it isolates the
        // Whisper log-mel path so the fused / unfused RTF isn't polluted by
        // encoder / decoder time. `vokra-cli run` has no analogous end-user
        // output, so reject rather than silently print something (FR-EX-08).
        ModelTask::MelFrontend => {
            return Err(
                "run: task `mel-frontend` is not supported (bench-only, see `vokra-cli bench --task mel-frontend`)"
                    .to_owned(),
            );
        }
        // Same posture for `cosyvoice2-synthetic` (M3-09-T24): bench-only
        // scaffold task. A real CosyVoice2 checkpoint's TTS run lands with
        // T07/T08 (LLM backbone forward) + T14/T15 (streaming pipeline
        // wired to a user-facing API) — that follow-on adds a
        // `ModelTask::Cosyvoice2` arm alongside `Tts` for the arch dispatch.
        ModelTask::Cosyvoice2Synthetic => {
            return Err(
                "run: task `cosyvoice2-synthetic` is not supported (bench-only, see \
                 `vokra-cli bench --task cosyvoice2-synthetic`)"
                    .to_owned(),
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Runs the strict UTMOS22-strong scorer and prints its single MOS value.
fn run_utmos(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (UTMOS22-strong): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let runtime = crate::utmos_runtime::UtmosRuntime::from_gguf(session.gguf(), args.backend)
        .map_err(|error| format!("run (UTMOS22-strong): {error}"))?;
    let score = runtime
        .score(&clip.samples, clip.sample_rate)
        .map_err(|error| format!("run (UTMOS22-strong): {error}"))?;
    println!("utmos: score={score:.9}");
    Ok(())
}

/// Runs the strict Microsoft DNSMOS P.808 + P.835 bundle.
fn run_dnsmos(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (DNSMOS): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let rate = vokra_models::dnsmos_p808_p835::SAMPLE_RATE;
    if clip.sample_rate != rate {
        return Err(format!(
            "run (DNSMOS): {path} is {} Hz, expected {rate} Hz — resample explicitly before scoring (FR-EX-08)",
            clip.sample_rate
        ));
    }
    let model = vokra_models::dnsmos_p808_p835::Dnsmos::from_gguf_with_backend(
        session.gguf(),
        args.backend,
    )
    .map_err(|error| format!("run (DNSMOS): {error}"))?;
    let score = model
        .score_all(&clip.samples)
        .map_err(|error| format!("run (DNSMOS): {error}"))?;
    println!(
        "dnsmos: p808={:.9} sig={:.9} bak={:.9} ovrl={:.9}",
        score.p808.expect("strict DNSMOS bundle always has P.808"),
        score.sig.expect("strict DNSMOS bundle always has SIG"),
        score.bak.expect("strict DNSMOS bundle always has BAK"),
        score.ovrl.expect("strict DNSMOS bundle always has OVRL")
    );
    Ok(())
}

/// Runs the strict NISQA v2 five-dimension quality scorer.
fn run_nisqa(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (NISQA): --input <mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let model = vokra_models::nisqa::Nisqa::from_gguf_with_backend(session.gguf(), args.backend)
        .map_err(|error| format!("run (NISQA): {error}"))?;
    let score = model
        .score_at_sample_rate(&clip.samples, clip.sample_rate)
        .map_err(|error| format!("run (NISQA): {error}"))?;
    println!(
        "nisqa: mos={:.9} noisiness={:.9} discontinuity={:.9} coloration={:.9} loudness={:.9}",
        score.mos, score.noisiness, score.discontinuity, score.coloration, score.loudness
    );
    Ok(())
}

/// Runs the strict SpeechBrain Lang-ID frontend/backbone/classifier and prints
/// the five highest official labels. `--output` writes the complete score
/// vector as little-endian f32 in label order.
fn run_lang_id(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (Lang-ID): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let model = vokra_models::lang_id::LangIdEcapa::from_gguf(session.gguf())
        .map_err(|error| format!("run (Lang-ID): {error}"))?
        .with_backend(args.backend);
    let scores = model
        .identify_pcm(&clip.samples, clip.sample_rate)
        .map_err(|error| format!("run (Lang-ID): {error}"))?;
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (rank, (index, score)) in ranked.into_iter().take(5).enumerate() {
        println!(
            "lang-id[{}]: index={index} label={} score={score:.9}",
            rank + 1,
            model.labels()[index]
        );
    }
    if let Some(output) = args.output.as_deref() {
        let bytes = scores
            .iter()
            .flat_map(|score| score.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (Lang-ID): --output {output}: {error}"))?;
        println!(
            "lang-id: {} scores in official label order -> {output}",
            scores.len()
        );
    }
    Ok(())
}

/// Runs the strict WavLM Audiobox Aesthetics scorer. `--output` writes four
/// little-endian f32 values in the upstream CE / CU / PC / PQ order.
fn run_audiobox_aesthetics(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (Audiobox Aesthetics): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != vokra_models::audiobox_aesthetics::SAMPLE_RATE {
        return Err(format!(
            "run (Audiobox Aesthetics): {path} is {} Hz, expected {} Hz — resample explicitly before scoring (FR-EX-08)",
            clip.sample_rate,
            vokra_models::audiobox_aesthetics::SAMPLE_RATE
        ));
    }
    let model = vokra_models::audiobox_aesthetics::AudioboxAesthetics::from_file(session.gguf())
        .map_err(|error| format!("run (Audiobox Aesthetics): {error}"))?
        .with_backend(args.backend);
    let scores = model
        .score_pcm(&clip.samples, clip.sample_rate)
        .map_err(|error| format!("run (Audiobox Aesthetics): {error}"))?;
    let values = scores.as_array();
    println!(
        "audiobox-aesthetics: CE={:.9} CU={:.9} PC={:.9} PQ={:.9}",
        values[0], values[1], values[2], values[3]
    );
    if let Some(output) = args.output.as_deref() {
        let bytes = values
            .iter()
            .flat_map(|score| score.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (Audiobox Aesthetics): --output {output}: {error}"))?;
        println!("audiobox-aesthetics: CE/CU/PC/PQ f32 -> {output}");
    }
    Ok(())
}

/// Runs the strict emotion2vec+ Large classifier. `--output` writes all nine
/// softmax scores as little-endian f32 in the official bilingual label order.
fn run_emotion2vec(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (emotion2vec): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let model = vokra_models::emotion2vec::Emotion2Vec::from_gguf(session.gguf())
        .map_err(|error| format!("run (emotion2vec): {error}"))?
        .with_backend(args.backend);
    let scores = model
        .classify_scores(&clip.samples, clip.sample_rate)
        .map_err(|error| format!("run (emotion2vec): {error}"))?;
    let labels = vokra_models::emotion2vec::Emotion2Vec::class_labels();
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (rank, (index, score)) in ranked.into_iter().enumerate() {
        println!(
            "emotion2vec[{}]: index={index} label={} score={score:.9}",
            rank + 1,
            labels[index]
        );
    }
    if let Some(output) = args.output.as_deref() {
        let bytes = scores
            .iter()
            .flat_map(|score| score.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (emotion2vec): --output {output}: {error}"))?;
        println!(
            "emotion2vec: {} scores in official label order -> {output}",
            scores.len()
        );
    }
    Ok(())
}

/// Runs the canonical Wav2Vec2 deepfake classifier. `--output` writes the
/// two softmax scores as little-endian f32 in `[fake, real]` order. No verdict
/// threshold is selected by the CLI.
fn run_deepfake_detection(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (deepfake detection): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let model = vokra_models::deepfake_detection::DeepfakeDetection::from_gguf_with_backend(
        session.gguf(),
        args.backend,
    )
    .map_err(|error| format!("run (deepfake detection): {error}"))?;
    let result = model
        .score_pcm(&clip.samples, clip.sample_rate)
        .map_err(|error| format!("run (deepfake detection): {error}"))?;
    let logits = result.logits();
    let scores = result.probabilities();
    let labels = vokra_models::deepfake_detection::DeepfakeDetection::class_labels();
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (rank, (index, score)) in ranked.into_iter().enumerate() {
        println!(
            "deepfake-detection[{}]: index={index} label={} logit={:.9} score={score:.9}",
            rank + 1,
            labels[index],
            logits[index]
        );
    }
    if let Some(output) = args.output.as_deref() {
        let bytes = scores
            .iter()
            .flat_map(|score| score.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (deepfake detection): --output {output}: {error}"))?;
        println!(
            "deepfake-detection: {} scores in [fake, real] order -> {output}",
            scores.len()
        );
    }
    Ok(())
}

/// Runs the strict four-checkpoint AudioSeal generator/detector. Explicit use
/// here does not change the separate global TTS watermark policy, whose
/// `WatermarkConfig::backend_status()` remains Deferred.
fn run_audioseal(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (AudioSeal): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != vokra_models::audioseal::SAMPLE_RATE {
        return Err(format!(
            "run (AudioSeal): {path} is {} Hz, expected {} Hz — resample explicitly before watermarking (FR-EX-08)",
            clip.sample_rate,
            vokra_models::audioseal::SAMPLE_RATE
        ));
    }
    let model = vokra_models::audioseal::Audioseal::from_file(session.gguf())
        .map_err(|error| format!("run (AudioSeal): {error}"))?
        .with_backend(args.backend);
    let mode = args.watermark_mode.unwrap_or(WatermarkMode::Detect);
    let variant = args
        .watermark_variant
        .unwrap_or(vokra_models::audioseal::AudiosealVariant::Base);
    let variant_name = match variant {
        vokra_models::audioseal::AudiosealVariant::Base => "base",
        vokra_models::audioseal::AudiosealVariant::Streaming => "streaming",
    };

    match mode {
        WatermarkMode::Detect => {
            if args.watermark_message.is_some() || args.watermark_alpha.is_some() {
                return Err(
                    "run (AudioSeal detect): --watermark-message and --watermark-alpha are \
                     embed-only; refusing to ignore them"
                        .to_owned(),
                );
            }
            if args.output.is_some() {
                return Err(
                    "run (AudioSeal detect): --output is not defined for detection; results \
                     are printed, so refusing to ignore the path"
                        .to_owned(),
                );
            }
            let detection = model
                .detect_pcm(&clip.samples, clip.sample_rate, variant)
                .map_err(|error| format!("run (AudioSeal detect): {error}"))?;
            let message = detection
                .message
                .iter()
                .map(|&bit| char::from(b'0' + bit))
                .collect::<String>();
            let message_probabilities = detection
                .message_probabilities
                .iter()
                .map(|probability| format!("{probability:.9}"))
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "audioseal-detect: variant={variant_name} probability={:.9} samples={} message={message} message_probabilities={message_probabilities}",
                detection.detection_probability,
                detection.positive_probabilities.len()
            );
        }
        WatermarkMode::Embed => {
            let message = args.watermark_message.as_ref().ok_or(
                "run (AudioSeal embed): --watermark-message <exactly-16-bits> is required",
            )?;
            let alpha = args.watermark_alpha.unwrap_or(1.0);
            let output = model
                .embed_pcm(&clip.samples, clip.sample_rate, message, alpha, variant)
                .map_err(|error| format!("run (AudioSeal embed): {error}"))?;
            emit_audio(
                &format!("audioseal-embed-{variant_name}"),
                &output,
                clip.sample_rate,
                args.output.as_deref(),
            )?;
        }
    }
    Ok(())
}

/// Runs the strict public AST AudioSet classifier and prints the ten highest
/// raw logits by class index. The current public GGUF does not carry the 527
/// human-readable AudioSet labels, so inventing names here would be unsafe;
/// callers can request the complete little-endian f32 vector via `--output`.
fn run_ast_classification(session: &vokra_core::Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (AST): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != vokra_models::ast::SAMPLE_RATE {
        return Err(format!(
            "run (AST): {path} is {} Hz, expected {} Hz — resample explicitly before inference (FR-EX-08)",
            clip.sample_rate,
            vokra_models::ast::SAMPLE_RATE
        ));
    }
    let model = vokra_models::ast::AstAudioSet::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(args.backend);
    let logits = model
        .classify_pcm(&clip.samples, clip.sample_rate)
        .map_err(|error| error.to_string())?;

    if let Some(output) = args.output.as_deref() {
        let bytes = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (AST): --output {output}: {error}"))?;
    }

    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    ranked.sort_unstable_by(|&left, &right| logits[right].total_cmp(&logits[left]));
    let top = ranked
        .into_iter()
        .take(10)
        .map(|index| format!("{index}:{:.6}", logits[index]))
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "ast: {} logits; top10(class_index:logit) {top}",
        logits.len()
    );
    if let Some(output) = args.output.as_deref() {
        println!("ast: raw logits -> {output}");
    } else {
        eprintln!(
            "vokra: AST note: this GGUF has no AudioSet label-name table; class indices are printed and no names are fabricated"
        );
    }
    Ok(())
}

/// Writes synthesized PCM to `--output`, or reports its duration when the flag
/// is absent. Shared by the piper-plus and Kokoro TTS arms so both report the
/// same way. `label` prefixes the line (`tts` / `kokoro`).
fn emit_audio(
    label: &str,
    samples: &[f32],
    sample_rate: u32,
    output: Option<&str>,
) -> Result<(), String> {
    match output {
        Some(out) => {
            wav::write_wav(out, samples, sample_rate)?;
            println!(
                "{label}: wrote {} samples @ {sample_rate} Hz -> {out}",
                samples.len()
            );
        }
        None => {
            let secs = samples.len() as f64 / f64::from(sample_rate);
            println!(
                "{label}: {} samples, {secs:.3}s @ {sample_rate} Hz (no --output; audio discarded)",
                samples.len()
            );
        }
    }
    Ok(())
}

/// Maps `--text` to Kokoro phoneme ids, wrapped in the id-0 sentinels
/// upstream's tokenizer adds (`input_ids = [0, *content, 0]`, `kokoro==0.9.4`
/// `pipeline.py`).
///
/// Two input forms, mirroring `PassthroughPhonemizer`'s content/framing split
/// on the piper-plus side (`vokra-piper-plus::phonemizer`):
///
/// - **symbol form** (`"həlˈO wˈɜːld"`) — each `char` is looked up in the
///   GGUF's `vokra.kokoro.phoneme_symbols` table (index = id). Every symbol in
///   the shipped 178-entry table is a single `char`, so a per-`char` lookup is
///   exact.
/// - **raw-id form** (`"1 2 3"` or `"1,2,3"`) — the piper raw-id syntax,
///   whitespace- or comma-separated. Reproduces an exact upstream tokenization
///   (e.g. replaying a parity dump) without routing IPA through a shell. Ids
///   are **content only**; the sentinels are added here, as in piper's
///   `parse_content` / `phonemize` split.
///
/// # Disambiguation is verified, not assumed
///
/// The raw-id form is selected when every token is ASCII digits. That is only
/// unambiguous while no phoneme symbol is itself a digit — true of the shipped
/// misaki table, but checked against the actual table at run time rather than
/// trusted, so a future table that adds a digit symbol is a loud error instead
/// of a silent misreading of the caller's input.
///
/// # Unmappable input is an error
///
/// Both upstream Kokoro and this crate's `PiperPlusTts::tokenize` silently drop
/// symbols they cannot map. This route does not: dropping a phoneme changes the
/// utterance with no signal to the caller, which is the silent-fallback shape
/// FR-EX-08 forbids. The message names every offending character so a caller
/// can see whether they passed graphemes by mistake — the most likely error,
/// since there is no G2P bridge in-tree to convert them.
pub(crate) fn kokoro_phoneme_ids(text: &str, symbols: &[String]) -> Result<Vec<i64>, String> {
    let content = if is_id_sequence(text) {
        kokoro_content_from_ids(text, symbols)?
    } else {
        kokoro_content_from_symbols(text, symbols)?
    };
    if content.is_empty() {
        return Err("run (kokoro): --text produced no phonemes".to_owned());
    }
    // Upstream wraps the content in id 0. Index 0 of the table is the
    // empty/pad entry, so the sentinels are pushed positionally rather than
    // looked up.
    let mut ids = Vec::with_capacity(content.len() + 2);
    ids.push(0);
    ids.extend_from_slice(&content);
    ids.push(0);
    Ok(ids)
}

/// Whether `text` is the piper raw-id form: at least one token, every token
/// non-empty ASCII digits, split on whitespace or `,`.
fn is_id_sequence(text: &str) -> bool {
    let mut any = false;
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        any = true;
        if !tok.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
    }
    any
}

/// Parses the raw-id form into **content** ids (no sentinels).
fn kokoro_content_from_ids(text: &str, symbols: &[String]) -> Result<Vec<i64>, String> {
    // The digit heuristic is only sound while no symbol is a bare digit; verify
    // against this GGUF's actual table instead of assuming (FR-EX-08).
    let digit_symbols: Vec<&String> = symbols
        .iter()
        .filter(|s| s.len() == 1 && s.as_bytes()[0].is_ascii_digit())
        .collect();
    if !digit_symbols.is_empty() {
        return Err(format!(
            "run (kokoro): --text looks like a raw id sequence, but this voice's \
             phoneme_symbols table contains digit symbol(s) {digit_symbols:?}, so the \
             raw-id and symbol forms are ambiguous for this model — the input cannot be \
             interpreted without guessing (FR-EX-08)"
        ));
    }
    let mut ids = Vec::new();
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let id: i64 = tok
            .parse()
            .map_err(|_| format!("run (kokoro): `{tok}` is not a phoneme id"))?;
        // Bound against the real table so an out-of-range id fails here rather
        // than indexing past the embedding rows downstream.
        if id <= 0 || id as usize >= symbols.len() {
            return Err(format!(
                "run (kokoro): phoneme id {id} out of range — --text takes CONTENT ids in \
                 1..{} (id 0 is the pad sentinel and is added automatically, as in the \
                 piper raw-id path)",
                symbols.len()
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

/// Parses the symbol form into **content** ids (no sentinels).
fn kokoro_content_from_symbols(text: &str, symbols: &[String]) -> Result<Vec<i64>, String> {
    let mut ids = Vec::with_capacity(text.chars().count());
    let mut unknown: Vec<char> = Vec::new();
    let mut buf = [0u8; 4];
    for ch in text.chars() {
        let needle = ch.encode_utf8(&mut buf);
        match symbols.iter().position(|s| s == needle) {
            // Index 0 is the pad sentinel; a table whose entry 0 is empty can
            // never match here, but guard anyway so a fixture that puts a real
            // symbol at 0 cannot inject an extra sentinel mid-sequence.
            Some(0) | None => unknown.push(ch),
            Some(id) => ids.push(id as i64),
        }
    }
    if !unknown.is_empty() {
        unknown.sort_unstable();
        unknown.dedup();
        return Err(format!(
            "run (kokoro): --text contains {} character(s) absent from this voice's \
             vokra.kokoro.phoneme_symbols table: {unknown:?}. --text takes misaki IPA \
             PHONEMES, not graphemes (there is no G2P bridge in-tree); dropping them \
             silently would change the utterance (FR-EX-08)",
            unknown.len()
        ));
    }
    Ok(ids)
}

/// Reads a raw little-endian f32 style vector from `path`.
///
/// The file must be a whole number of f32s and match either `style_dim` or
/// `2·style_dim` — the two lengths `KokoroTts::synthesize_phonemes` accepts.
/// Both checks happen here so a truncated dump is named as such rather than
/// surfacing as a shape error from inside the prosody predictor.
pub(crate) fn read_style_vector(path: &str, style_dim: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("--style {path}: {e}"))?;
    // `%` rather than `usize::is_multiple_of`: this crate inherits the
    // workspace MSRV (1.85) and that method is stable only since 1.87.
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "--style {path}: {} bytes is not a whole number of f32s",
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    if n != style_dim && n != 2 * style_dim {
        return Err(format!(
            "--style {path}: {n} floats — expected style_dim ({style_dim}) or 2*style_dim \
             ({}) for a full upstream ref_s row",
            2 * style_dim
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The Kokoro-82M synthesis path (cc-24).
///
/// `--text` is a misaki phoneme string (see [`kokoro_phoneme_ids`]); the style
/// comes from `--style` (a raw f32 dump of an upstream voicepack row) or
/// `--voice` (a name from the GGUF's table).
///
/// # Why the engine is rebuilt here
///
/// The reachable synthesis surface is the concrete
/// [`vokra_models::kokoro::KokoroTts::synthesize_phonemes`], not the
/// [`vokra_core::TtsEngine`] trait — Kokoro's `synthesize` is a hard
/// `NotImplemented` pending a misaki G2P bridge. The arch dispatch therefore
/// hands back a bare session and the concrete engine binds once, here, from the
/// model path (the `ModelTask::Speaker` / `ModelTask::AsrVoxtral` pattern).
fn run_kokoro(a: &RunArgs) -> Result<(), String> {
    use vokra_models::kokoro::KokoroTts;

    let text = a
        .text
        .as_deref()
        .ok_or("run (kokoro): --text <phonemes> is required")?;
    let tts = KokoroTts::from_path(&a.model)
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    let config = tts.config();
    let ids = kokoro_phoneme_ids(text, &config.phoneme_symbols)?;

    // Style resolution mirrors `synthesize_phonemes`: an explicit override wins
    // over a name. Neither present is an error — there is no neutral default
    // style, and silently substituting zeros would synthesize in a voice the
    // caller never asked for (FR-EX-08).
    let style = match a.style.as_deref() {
        Some(path) => Some(read_style_vector(path, config.style_dim)?),
        None => None,
    };
    if style.is_none() && a.voice.is_none() {
        return Err(format!(
            "run (kokoro): a style is required — pass --style <f32 dump> (style_dim {} or \
             2*style_dim {}), or --voice <name> from {:?}",
            config.style_dim,
            2 * config.style_dim,
            config.voice_names,
        ));
    }

    let audio = tts
        .synthesize_phonemes(
            &ids,
            a.voice.as_deref(),
            style.as_deref(),
            0.0,
            a.length_scale,
        )
        .map_err(|e| {
            // `--voice` hits a hard `NotImplemented` in the model layer
            // (`synthesize_phonemes` — the voice → style-row lookup is
            // M2-07-T02). Append the actionable workaround rather than just
            // propagating the bare "not implemented".
            if a.voice.is_some() && style.is_none() {
                format!(
                    "{e}\nnote: the voice name resolves against \
                     `vokra.kokoro.voice_names`, but mapping it to a style row is not \
                     implemented yet, so `--voice` cannot synthesize on ANY Kokoro GGUF \
                     — including one whose voicepack rows were stacked in at conversion \
                     time (`tools/parity/kokoro_prepare_checkpoint.py --stack-voicepack`, \
                     off by default; upstream ships the rows as separate `voices/*.pt`). \
                     Until then use --style: export the row upstream would have picked, \
                     `voicepack[len(phonemes) - 1]` ({} f32, little-endian), and pass \
                     the file.",
                    2 * config.style_dim
                )
            } else {
                e.to_string()
            }
        })?;

    println!(
        "kokoro: {} phoneme ids (incl. 2 sentinels), style {}",
        ids.len(),
        match (&style, a.voice.as_deref()) {
            (Some(s), _) => format!("override ({} f32)", s.len()),
            (None, Some(v)) => format!("voice `{v}`"),
            (None, None) => unreachable!("checked above"),
        }
    );
    emit_audio(
        "kokoro",
        &audio.samples,
        audio.sample_rate,
        a.output.as_deref(),
    )
}

/// The speaker-embedding demo path (CAM++ / X-vector / ECAPA-TDNN / WeSpeaker /
/// TitaNet, FR-OP-81). Each model
/// owns its exact PCM frontend and returns its native embedding width. With
/// `--compare <b.wav>` the cosine similarity of the two embeddings via
/// [`vokra_models::speaker::speaker_verify`] (threshold-free: the operating
/// point is the caller's, ADR M4-20 §D-4).
///
/// The concrete encoder binds here from the already-open session GGUF and
/// honors `--backend`; this keeps model-specific diagnostics available without
/// loading the weights twice. Unknown/malformed layouts and unsupported
/// backends are loud errors (FR-EX-08).
fn run_speaker(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::SpeakerEngine;
    use vokra_models::ecapa_tdnn::EcapaTdnn;
    use vokra_models::speaker::{SpeakerEncoder, speaker_verify};
    use vokra_models::titanet::TitaNet;
    use vokra_models::wespeaker::WeSpeaker;
    use vokra_models::xvector::XVector;

    let input = a
        .input
        .as_deref()
        .ok_or("run (speaker): --input <a.wav> is required")?;
    let arch = session
        .gguf()
        .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or("run (speaker): GGUF is missing `vokra.model.arch`")?;
    let encoder: Box<dyn SpeakerEngine> = match arch {
        "campplus" => Box::new(
            SpeakerEncoder::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(a.backend),
        ),
        "xvector" => Box::new(
            XVector::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(a.backend),
        ),
        "ecapa_tdnn" => Box::new(
            EcapaTdnn::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(a.backend),
        ),
        "wespeaker" => Box::new(
            WeSpeaker::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(a.backend),
        ),
        "titanet-large" => Box::new(
            TitaNet::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(a.backend),
        ),
        other => {
            return Err(format!(
                "run (speaker): unsupported speaker architecture `{other}`"
            ));
        }
    };

    // WAV → model-owned frontend → embedding. A mismatched rate is an
    // explicit error inside the model; neither frontend silently resamples.
    let embed_clip = |path: &str| -> Result<Vec<f32>, String> {
        let clip = wav::read_wav(path)?;
        let emb = encoder
            .embed(&clip.samples, clip.sample_rate)
            .map_err(|e| format!("run (speaker): {path}: {e}"))?;
        let l2 = emb
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        println!(
            "speaker: {path}: samples={} dim={} l2_norm={l2:.6}",
            clip.samples.len(),
            emb.len()
        );
        Ok(emb)
    };

    let emb_a = embed_clip(input)?;
    if let Some(output) = a.output.as_deref() {
        let bytes = emb_a
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (speaker): --output {output}: {error}"))?;
        println!(
            "speaker: embedding dim={} raw little-endian f32 -> {output}",
            emb_a.len()
        );
    }
    if let Some(compare) = a.compare.as_deref() {
        let emb_b = embed_clip(compare)?;
        let result = speaker_verify(&emb_a, &emb_b, None).map_err(|e| e.to_string())?;
        println!("speaker: cosine_similarity={:.6}", result.similarity);
    }
    Ok(())
}

/// The native NSNet2 / RNNoise / DeepFilterNet3 / MetricGAN+ enhancement path.
///
/// `--input` is a mono WAV at the model's trained rate; the denoised PCM goes
/// to `--output` (or its duration is reported when the flag is absent, the
/// shared [`emit_audio`] contract).
///
/// The forward is real — [`vokra_models::nsnet2::Nsnet2V1::denoise_pcm`] runs
/// the STFT → `fc_in` → GRU ×2 → `fc_1..3` → mask → iSTFT chain. Like
/// [`run_speaker`], the concrete model binds here from the session's own GGUF
/// (the [`Session`] facade has no denoise engine slot), and its strict
/// `vokra.model.arch` check refuses a foreign artifact loudly (FR-EX-08).
///
/// A rate mismatch is an explicit error rather than a silent resample: NSNet2's
/// STFT geometry (`n_fft` / `hop` / `win_length`) is fixed against the trained
/// rate, so feeding a 44.1/48 kHz clip through it would emit plausible-sounding
/// but wrong audio with no diagnostic.
fn run_denoise(session: &Session, a: &RunArgs) -> Result<(), String> {
    let path = a
        .input
        .as_deref()
        .ok_or("run (denoise): --input <in.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let arch = session
        .gguf()
        .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or("run (denoise): GGUF is missing `vokra.model.arch`")?;
    let (rate, out) = match arch {
        "nsnet2" => {
            let model = vokra_models::nsnet2::Nsnet2V1::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let rate = model.config().sample_rate;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .denoise_pcm(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        "rnnoise" => {
            let model = vokra_models::rnnoise::RnnoiseV02::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let rate = vokra_models::rnnoise::SAMPLE_RATE;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .denoise_pcm(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        "denoise" => {
            let model = vokra_models::deepfilternet3::DeepFilterNet3::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let rate = model.config().sample_rate;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .enhance(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        "metricgan_plus" => {
            let model = vokra_models::metricgan_plus::MetricGanPlus::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let rate = vokra_models::metricgan_plus::SAMPLE_RATE;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .enhance(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        "mp_senet" => {
            let model = vokra_models::mp_senet::MpSenet::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let rate = vokra_models::mp_senet::SAMPLE_RATE;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .enhance(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        "facebook_denoiser" => {
            let model = vokra_models::facebook_denoiser::FbDenoiser::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(a.backend);
            let rate = vokra_models::facebook_denoiser::SAMPLE_RATE;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .denoise(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        "frcrn" => {
            let model =
                vokra_models::frcrn::Frcrn::from_gguf_with_backend(session.gguf(), a.backend)
                    .map_err(|error| error.to_string())?;
            let rate = vokra_models::frcrn::SAMPLE_RATE;
            if clip.sample_rate != rate {
                return Err(denoise_rate_error(path, arch, rate, clip.sample_rate));
            }
            let output = model
                .enhance(&clip.samples)
                .map_err(|error| error.to_string())?;
            (rate, output)
        }
        other => {
            return Err(format!(
                "run (denoise): internal dispatch error: arch `{other}` is not nsnet2, rnnoise, denoise, metricgan_plus, mp_senet, facebook_denoiser, or frcrn"
            ));
        }
    };
    println!(
        "denoise: {} in -> {} out samples @ {rate} Hz",
        clip.samples.len(),
        out.len()
    );
    emit_audio("denoise", &out, rate, a.output.as_deref())
}

fn denoise_rate_error(path: &str, arch: &str, expected: u32, actual: u32) -> String {
    format!(
        "run (denoise): {path}: arch `{arch}` expects a {expected} Hz mono WAV, got {actual} Hz — resample offline first (FR-EX-08: never a silent resample)"
    )
}

/// Runs one complete source-separation or enhancement utterance.
fn run_separation(session: &Session, a: &RunArgs) -> Result<(), String> {
    let path = a
        .input
        .as_deref()
        .ok_or("run (separation): --input <mixture.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let arch = session
        .gguf()
        .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or("run (separation): GGUF is missing `vokra.model.arch`")?;
    let (label, model): (&str, Box<dyn SeparationEngine>) = match arch {
        "sepformer" => (
            "sepformer",
            Box::new(
                vokra_models::sepformer::SepFormer::from_gguf(session.gguf())
                    .map_err(|error| error.to_string())?
                    .with_backend(a.backend),
            ),
        ),
        "conv_tasnet" => (
            "conv-tasnet",
            Box::new(
                vokra_models::conv_tasnet::ConvTasnet::from_gguf(session.gguf())
                    .map_err(|error| error.to_string())?
                    .with_backend(a.backend),
            ),
        ),
        "tiger_separator" => (
            "tiger",
            Box::new(
                vokra_models::tiger::TigerSeparator::from_gguf(session.gguf())
                    .map_err(|error| error.to_string())?
                    .with_backend(a.backend),
            ),
        ),
        "mossformer2_ss_16k" => (
            "mossformer2-ss-16k",
            Box::new(
                vokra_models::mossformer2_ss_16k::Mossformer2Ss16k::from_gguf(session.gguf())
                    .map_err(|error| error.to_string())?
                    .with_backend(a.backend),
            ),
        ),
        other => {
            return Err(format!(
                "run (separation): internal dispatch error: arch `{other}` is not sepformer, conv_tasnet, tiger_separator, or mossformer2_ss_16k"
            ));
        }
    };
    let rate = model.sample_rate();
    if clip.sample_rate != rate {
        return Err(format!(
            "run (separation): {path} is {} Hz, but arch `{arch}` requires {rate} Hz — resample offline first (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let outputs = model
        .separate(&clip.samples)
        .map_err(|error| error.to_string())?;
    if outputs.len() != model.output_streams() {
        return Err(format!(
            "run (separation): internal output-count mismatch: model declares {}, forward returned {}",
            model.output_streams(),
            outputs.len()
        ));
    }

    match a.output.as_deref() {
        None => println!(
            "{label}: {} input samples -> {} streams x {} samples @ {rate} Hz (no --output; audio discarded)",
            clip.samples.len(),
            outputs.len(),
            outputs.first().map_or(0, Vec::len),
        ),
        Some(output) if outputs.len() == 1 => {
            wav::write_wav(output, &outputs[0], rate)?;
            println!(
                "{label}: wrote {} samples @ {rate} Hz -> {output}",
                outputs[0].len()
            );
        }
        Some(output) => {
            for (index, pcm) in outputs.iter().enumerate() {
                let stream_path = separation_stream_path(output, index + 1);
                wav::write_wav(&stream_path, pcm, rate)?;
                println!(
                    "{label}: source {} wrote {} samples @ {rate} Hz -> {}",
                    index + 1,
                    pcm.len(),
                    stream_path.display(),
                );
            }
        }
    }
    Ok(())
}

fn separation_stream_path(base: &str, source: usize) -> std::path::PathBuf {
    let path = std::path::Path::new(base);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("separated");
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("wav");
    path.with_file_name(format!("{stem}.source{source}.{extension}"))
}

/// Exact pyannote speaker-diarization 3.1 three-GGUF route.
fn run_pyannote_diarization(a: &RunArgs) -> Result<(), String> {
    use vokra_models::pyannote::diarization::{PyannoteSpeakerDiarization31, SAMPLE_RATE};

    let input = a
        .input
        .as_deref()
        .ok_or("run (pyannote diarization): --input <16k-mono.wav> is required")?;
    let segmentation = a.segmentation_model.as_deref().ok_or(
        "run (pyannote diarization): --segmentation-model <pyannote-segmentation.gguf> is required",
    )?;
    let embedding = a.embedding_model.as_deref().ok_or(
        "run (pyannote diarization): --embedding-model <pyannote-wespeaker.gguf> is required",
    )?;
    let clip = wav::read_wav(input)?;
    if clip.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "run (pyannote diarization): {input} is {} Hz, expected {SAMPLE_RATE} Hz — resample offline first (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let file_id = std::path::Path::new(input)
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            format!("run (pyannote diarization): cannot derive a UTF-8 RTTM file id from `{input}`")
        })?;
    if file_id.chars().any(char::is_whitespace) {
        return Err(format!(
            "run (pyannote diarization): input stem `{file_id}` contains whitespace and cannot be an RTTM file id; rename the WAV explicitly"
        ));
    }
    let model = PyannoteSpeakerDiarization31::open(&a.model, segmentation, embedding, a.backend)
        .map_err(|error| error.to_string())?;
    let segments = model
        .diarize(&clip.samples, clip.sample_rate)
        .map_err(|error| error.to_string())?;
    let rttm = vokra_models::pyannote::rttm::write_rttm(file_id, &segments);
    if let Some(output) = a.output.as_deref() {
        std::fs::write(output, rttm.as_bytes())
            .map_err(|error| format!("run (pyannote diarization): --output {output}: {error}"))?;
        println!(
            "pyannote-diarization: {} turn(s), backend={:?} -> {output}",
            segments.len(),
            model.backend()
        );
    } else {
        print!("{rttm}");
    }
    Ok(())
}

/// The pyannote `segmentation-3.0` speaker-segmentation path (Wave G).
///
/// Prints one summary line: total frames, frames carrying any speaker, frames
/// carrying two or more (overlapped speech), and the set of speaker indices
/// seen. The per-frame powerset decode itself is
/// [`vokra_models::pyannote::PyanNet::segment_powerset`].
///
/// The selected CPU or Metal backend is preflighted before the WAV is read.
/// Unsupported or unavailable selections fail explicitly; the model never
/// falls back to CPU per operation.
fn run_segment(a: &RunArgs) -> Result<(), String> {
    use vokra_models::pyannote::PyanNet;

    let path = a
        .input
        .as_deref()
        .ok_or("run (segment): --input <in.wav> is required")?;
    let model = PyanNet::from_gguf_with_backend(std::path::Path::new(&a.model), a.backend)
        .map_err(|e| e.to_string())?;
    let rate = model.config().sample_rate;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != rate {
        return Err(format!(
            "run (segment): {path}: expected a {rate} Hz mono WAV (PyanNet's SincNet \
             front-end learns its filter cutoffs against that rate), got {} Hz — \
             resample offline first (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let activity = model
        .segment_powerset(&clip.samples)
        .map_err(|e| e.to_string())?;
    let frames = activity.len();
    let speech = activity
        .iter()
        .filter(|f| !f.active_speakers.is_empty())
        .count();
    let overlap = activity
        .iter()
        .filter(|f| f.active_speakers.len() > 1)
        .count();
    let mut speakers: Vec<usize> = activity
        .iter()
        .flat_map(|f| f.active_speakers.iter().copied())
        .collect();
    speakers.sort_unstable();
    speakers.dedup();
    println!(
        "segment: {frames} frames, speech_frames={speech}, overlap_frames={overlap}, \
         speakers_seen={speakers:?}"
    );
    Ok(())
}

/// The RMVPE F0 (pitch) extraction path (Wave G, FR-OP-83).
///
/// Prints a summary line, then one `time_sec<TAB>hz<TAB>voiced<TAB>confidence`
/// row per analysis hop — the same tab-separated shape `--word-timestamps`
/// uses, so the track pipes straight into a downstream tool.
///
/// Only RMVPE is routed here. Its
/// [`extract_real`](vokra_models::f0::rmvpe::RMVPE::extract_real) returns a
/// `Result` and has no silent all-zero branch — the timebase-only accessor is
/// [`frame_times`](vokra_models::f0::rmvpe::RMVPE::frame_times), which returns
/// bare timestamps and is never what this path calls.
///
/// The sibling extractors FCPE and CREPE stay in the CLI's bound-arch
/// registry, but as of 2026-08-15 neither is held back by a fabricated track:
/// both gained the same `extract` / `extract_real` / `frame_times` shape and
/// now refuse a weightless or wrong-rate artifact with a named error. What
/// they lack is the CLI wiring — see their rows in `BOUND_ARCHES`.
fn run_f0_rmvpe(a: &RunArgs) -> Result<(), String> {
    use vokra_models::f0::rmvpe::RMVPE;

    let path = a
        .input
        .as_deref()
        .ok_or("run (f0): --input <in.wav> is required")?;
    let model = RMVPE::open(&a.model)
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    let rate = model.config().sample_rate;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != rate {
        return Err(format!(
            "run (f0): {path}: expected a {rate} Hz mono WAV (the RMVPE mel front-end is \
             fixed at the rate in `vokra.rmvpe.*`), got {} Hz — resample offline first \
             (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let track = model
        .extract_real(&clip.samples, clip.sample_rate)
        .map_err(|e| e.to_string())?;
    emit_f0_track(&track, model.config().hop, rate);
    Ok(())
}

fn run_f0_fcpe(a: &RunArgs) -> Result<(), String> {
    use vokra_models::f0::fcpe::FCPE;

    let path = a
        .input
        .as_deref()
        .ok_or("run (f0): --input <in.wav> is required")?;
    let model = FCPE::from_gguf(std::path::Path::new(&a.model))
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    let rate = model.config().sample_rate;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != rate {
        return Err(format!(
            "run (f0): {path}: expected a {rate} Hz mono WAV (the FCPE mel front-end is \
             fixed at the rate in `vokra.f0.fcpe.*`), got {} Hz — resample offline first \
             (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let track = model
        .extract_real(&clip.samples, clip.sample_rate)
        .map_err(|e| e.to_string())?;
    emit_f0_track(&track, model.config().hop, rate);
    Ok(())
}

fn run_f0_crepe(a: &RunArgs) -> Result<(), String> {
    use vokra_models::f0::crepe::{CREPE, NATIVE_SAMPLE_RATE};

    let path = a
        .input
        .as_deref()
        .ok_or("run (f0): --input <in.wav> is required")?;
    let model = CREPE::from_gguf(std::path::Path::new(&a.model)).map_err(|e| e.to_string())?;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != NATIVE_SAMPLE_RATE {
        return Err(format!(
            "run (f0): {path}: expected a {NATIVE_SAMPLE_RATE} Hz mono WAV (CREPE's \
             1024-sample frame and cent grid are anchored to that rate), got {} Hz — \
             resample offline first (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let track = model
        .extract_real(&clip.samples, clip.sample_rate)
        .map_err(|e| e.to_string())?;
    emit_f0_track(&track, model.config().hop, NATIVE_SAMPLE_RATE);
    Ok(())
}

/// Charsiu's paired audio/phone-sequence route.
///
/// The phone transcript is deliberately accepted only as exact, whitespace-
/// delimited labels from the GGUF vocabulary.  G2P, case folding and unknown
/// substitution would all change the alignment contract, so they remain
/// explicit caller-side steps.
fn run_charsiu_align(session: &Session, a: &RunArgs) -> Result<(), String> {
    let path = a
        .input
        .as_deref()
        .ok_or("run (charsiu): --input <in.wav> is required")?;
    let phone_text = a
        .text
        .as_deref()
        .ok_or("run (charsiu): --text \"P AE T\" is required")?;
    let phones: Vec<String> = phone_text
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect();
    if phones.is_empty() {
        return Err("run (charsiu): --text must contain at least one phone label".to_owned());
    }

    let model = vokra_models::align::charsiu::Charsiu::from_file(session.gguf())
        .map_err(|e| e.to_string())?;
    let clip = wav::read_wav(path)?;
    let expected_rate = model.config().sample_rate;
    if clip.sample_rate != expected_rate {
        return Err(format!(
            "run (charsiu): {path}: expected a {expected_rate} Hz mono WAV, got {} Hz — \
             resample offline first (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    let aligned = model
        .align(&clip.samples, clip.sample_rate, &phones)
        .map_err(|e| e.to_string())?;

    let mut tsv =
        String::from("vokra-charsiu-alignment-tsv-v1\nphone\tstart_sec\tend_sec\tconfidence\n");
    for token in &aligned {
        use std::fmt::Write as _;
        writeln!(
            tsv,
            "{}\t{:.6}\t{:.6}\t{:.6}",
            token.text, token.start_sec, token.end_sec, token.confidence
        )
        .expect("writing to a String cannot fail");
    }
    match a.output.as_deref() {
        Some(out) => {
            std::fs::write(out, tsv)
                .map_err(|e| format!("run (charsiu): cannot write {out}: {e}"))?;
            println!("charsiu: wrote {} aligned phone(s) -> {out}", aligned.len());
        }
        None => print!("{tsv}"),
    }
    Ok(())
}

/// Shared renderer for all three checkpoint-backed F0 estimators.
fn emit_f0_track(track: &[vokra_models::f0::F0Frame], hop: u32, rate: u32) {
    let voiced = track.iter().filter(|f| f.voiced).count();
    println!(
        "f0: {} frames, voiced_frames={voiced}, hop={} @ {rate} Hz",
        track.len(),
        hop
    );
    for frame in track {
        println!(
            "{:.4}\t{:.3}\t{}\t{:.4}",
            frame.time_sec, frame.hz, frame.voiced, frame.confidence
        );
    }
}

fn run_text_normalize(session: &Session, a: &RunArgs) -> Result<(), String> {
    let text = a
        .text
        .as_deref()
        .ok_or("run (wetextprocessing): --text <string> is required")?;
    let model = vokra_models::wetextprocessing::WeTextProcessing::from_gguf(session.gguf())
        .map_err(|e| e.to_string())?;
    let normalized = model.normalize(text).map_err(|e| e.to_string())?;
    println!("normalize: {normalized}");
    Ok(())
}

/// CT-Punc's versioned paired-input route.
///
/// Token strings and ids are read from one TSV record stream so they cannot
/// acquire different lengths through two independently parsed flags. The
/// binder still validates vocabulary range and emits exactly one label per
/// record. `--output` is exact UTF-8 restored text with no diagnostic prefix;
/// stdout uses a labelled human-readable line when no file is requested.
fn run_ct_punc(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.input.is_some() || a.text.is_some() {
        return Err(
            "run (ct_punc): use --tokens <tokens.tsv>; --input/--text cannot preserve the \
             required caller-supplied token-string/token-id pairing"
                .to_owned(),
        );
    }
    let path = a
        .tokens
        .as_deref()
        .ok_or("run (ct_punc): --tokens <tokens.tsv> is required")?;
    let input = std::fs::read_to_string(path)
        .map_err(|e| format!("run (ct_punc): --tokens {path}: {e}"))?;
    let paired = parse_ct_punc_tsv(&input)?;
    debug_assert_eq!(paired.tokens.len(), paired.token_ids.len());
    let tokens: Vec<&str> = paired.tokens.iter().map(String::as_str).collect();
    let model =
        vokra_models::ct_punc::CtPunc::from_gguf(session.gguf()).map_err(|e| e.to_string())?;
    let restored = model
        .restore(&tokens, &paired.token_ids)
        .map_err(|e| e.to_string())?;
    if let Some(output) = a.output.as_deref() {
        std::fs::write(output, restored.as_bytes())
            .map_err(|e| format!("run (ct_punc): --output {output}: {e}"))?;
        eprintln!(
            "ct_punc: restored {} paired tokens -> {output}",
            paired.tokens.len()
        );
    } else {
        println!("ct_punc: {restored}");
    }
    Ok(())
}

fn parse_comma_u32_ids(raw: &str, surface: &str, flag: &str) -> Result<Vec<u32>, String> {
    if raw.trim().is_empty() {
        return Err(format!("run ({surface}): {flag} must not be empty"));
    }
    raw.split(',')
        .enumerate()
        .map(|(index, field)| {
            let field = field.trim();
            if field.is_empty() {
                return Err(format!("run ({surface}): {flag} field {index} is empty"));
            }
            field.parse::<u32>().map_err(|error| {
                format!("run ({surface}): {flag} field {index} `{field}` is not u32: {error}")
            })
        })
        .collect()
}

fn parse_bert_token_ids(raw: &str) -> Result<Vec<u32>, String> {
    parse_comma_u32_ids(raw, "BERT encoder", "--token-ids")
}

/// Standalone BERT-family final-hidden-state execution.
///
/// The GGUF may carry a tokenizer for an SBV2 parent, but this generic route
/// intentionally accepts exact ids only: the three public sidecars use three
/// different tokenisation schemes. Inferring the wrong one would produce a
/// valid-shaped but semantically wrong hidden state.
fn run_bert_encoder(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.input.is_some() || a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (BERT encoder): use --token-ids <u32,u32,...>; --input/--text/--tokens are not standalone BERT inputs"
                .to_owned(),
        );
    }
    let raw = a
        .token_ids
        .as_deref()
        .ok_or("run (BERT encoder): --token-ids <u32,u32,...> is required")?;
    let token_ids = parse_bert_token_ids(raw)?;
    let model = vokra_models::bert_runtime::BertRuntime::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?;
    let hidden = model
        .encode(&token_ids, a.backend)
        .map_err(|error| error.to_string())?;

    if let Some(output) = a.output.as_deref() {
        let mut bytes = Vec::with_capacity(hidden.len() * std::mem::size_of::<f32>());
        for value in &hidden {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(output, bytes)
            .map_err(|error| format!("run (BERT encoder): --output {output}: {error}"))?;
        println!(
            "bert-hidden: arch={} tokens={} hidden={} f32 -> {output}",
            model.kind().arch(),
            token_ids.len(),
            model.d_model(),
        );
    } else {
        let min = hidden.iter().copied().fold(f32::INFINITY, f32::min);
        let max = hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = hidden.iter().map(|value| f64::from(*value)).sum::<f64>() / hidden.len() as f64;
        println!(
            "bert-hidden: arch={} tokens={} hidden={} min={min:.6e} max={max:.6e} mean={mean:.6e} (no --output; features discarded)",
            model.kind().arch(),
            token_ids.len(),
            model.d_model(),
        );
    }
    Ok(())
}

/// Public MusicGen Small/Melody explicit token-id generation.
///
/// The public composite GGUFs contain T5-base, the autoregressive LM and the
/// 32 kHz EnCodec decoder, but no tokenizer. Both the conditional and CFG-null
/// token sequences are therefore required inputs rather than guessed from raw
/// text. Medium/Large share the arch tag and fail from the model binder with
/// their explicit missing-companion diagnostic.
fn run_musicgen(a: &RunArgs) -> Result<(), String> {
    if a.input.is_some() || a.text.is_some() || a.tokens.is_some() {
        return Err("run (MusicGen): use explicit --token-ids and \
             --music-unconditional-token-ids; --input/--text/--tokens are not accepted because \
             the public GGUF has no tokenizer"
            .to_owned());
    }
    let conditional = parse_comma_u32_ids(
        a.token_ids
            .as_deref()
            .ok_or("run (MusicGen): --token-ids <u32,u32,...> is required")?,
        "MusicGen",
        "--token-ids",
    )?;
    let unconditional = parse_comma_u32_ids(
        a.music_unconditional_token_ids
            .as_deref()
            .ok_or("run (MusicGen): --music-unconditional-token-ids <u32,u32,...> is required")?,
        "MusicGen",
        "--music-unconditional-token-ids",
    )?;
    let frames = a
        .music_frames
        .ok_or("run (MusicGen): --music-frames <positive-50Hz-frame-count> is required")?;
    let generation = vokra_models::audiocraft_lm::AudioCraftGenerationConfig::sampled(
        frames,
        a.music_seed.unwrap_or(0),
    );
    let policy = vokra_core::CompliancePolicy::from_env();
    let model = vokra_models::musicgen::MusicGen::from_path_with_policy_and_backend(
        &a.model, &policy, a.backend,
    )
    .map_err(|error| format!("run (MusicGen bind): {error}"))?;
    let pcm = model
        .generate_from_token_ids(&conditional, None, &unconditional, None, &generation)
        .map_err(|error| format!("run (MusicGen generate): {error}"))?;
    emit_audio(
        "musicgen",
        &pcm,
        model.config().sample_rate_hz,
        a.output.as_deref(),
    )
}

/// Standalone Mimi encode/decode using the portable `VKRMCODE` v1 contract.
fn run_mimi_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_models::codec::MimiCodecGguf;
    use vokra_models::mimi::{MimiEncoder, MimiNeuralConfig, MimiNeuralDecoder};
    use vokra_ops::{MimiRvqAttrs, mimi_rvq_decode};

    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (mimi): --text/--tokens are not codec inputs; use --input plus \
             --codec-mode encode|decode"
                .to_owned(),
        );
    }
    let mode = a
        .codec_mode
        .ok_or("run (mimi): --codec-mode encode|decode is required")?;
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (mimi): --input <wav|codes.vmc> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (mimi): --output <codes.vmc|wav> is required")?;

    // The standalone route is a model-load boundary just like Moshi's
    // `with_mimi_gguf`: enforce the provenance gate and surface CC-BY
    // attribution before binding any learned tensors.
    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|e| e.to_string())?;
    if let Some(info) = vokra_core::resolve_attribution(session.gguf()) {
        eprintln!("vokra: ATTRIBUTION ({}) {}", info.license, info.text);
    }

    let cfg = MimiNeuralConfig::from_gguf(session.gguf()).map_err(|e| e.to_string())?;
    cfg.validate().map_err(|e| e.to_string())?;
    let codec = MimiCodecGguf::from_gguf(session.gguf()).map_err(|e| e.to_string())?;
    let n_q = cfg.quantizer.n_q;
    if codec.attrs.n_codebooks < n_q {
        return Err(format!(
            "run (mimi): effective table GGUF has {} codebooks but the neural config requires {n_q}",
            codec.attrs.n_codebooks
        ));
    }
    if codec.attrs.codebook_size != cfg.quantizer.bins {
        return Err(format!(
            "run (mimi): effective table codebook_size {} != neural quantizer bins {}",
            codec.attrs.codebook_size, cfg.quantizer.bins
        ));
    }
    let table_bytes = session
        .gguf()
        .tensor_data("vokra.mimi.codebook_tables")
        .ok_or("run (mimi): GGUF tensor `vokra.mimi.codebook_tables` has no data")?;
    let model_sha256 = sha256(table_bytes);
    let hop = cfg.frame_hop_samples().map_err(|e| e.to_string())?;

    match mode {
        CodecMode::Encode => {
            let clip = wav::read_wav(input_path)
                .map_err(|e| format!("run (mimi encode): {input_path}: {e}"))?;
            if clip.sample_rate != cfg.sample_rate {
                return Err(format!(
                    "run (mimi encode): {input_path} is {} Hz, model requires {} Hz — \
                     resample offline first (FR-EX-08: never a silent resample)",
                    clip.sample_rate, cfg.sample_rate
                ));
            }
            if clip.samples.is_empty() || clip.samples.len() % hop != 0 {
                return Err(format!(
                    "run (mimi encode): input has {} samples; v1 requires a positive exact \
                     multiple of the model frame hop {hop} (no implicit pad/trim)",
                    clip.samples.len()
                ));
            }
            let encoder = MimiEncoder::from_gguf(session.gguf(), &cfg)
                .map_err(|e| e.to_string())?
                .with_backend(a.backend);
            let codes = encoder
                .encode_all(&clip.samples)
                .map_err(|e| e.to_string())?;
            let n_frames = codes.len() / n_q;
            let container = MimiCodesV1 {
                sample_rate: cfg.sample_rate,
                frame_rate_mhz: cfg.frame_rate_mhz,
                n_codebooks: u32::try_from(n_q)
                    .map_err(|_| "run (mimi encode): n_codebooks exceeds u32")?,
                codebook_size: u32::try_from(cfg.quantizer.bins)
                    .map_err(|_| "run (mimi encode): codebook_size exceeds u32")?,
                feature_dimension: u32::try_from(codec.attrs.d_model)
                    .map_err(|_| "run (mimi encode): feature dimension exceeds u32")?,
                n_frames: u64::try_from(n_frames)
                    .map_err(|_| "run (mimi encode): frame count exceeds u64")?,
                pcm_samples: u64::try_from(clip.samples.len())
                    .map_err(|_| "run (mimi encode): sample count exceeds u64")?,
                model_sha256,
                codes,
            };
            let bytes = container.to_bytes()?;
            std::fs::write(output_path, bytes)
                .map_err(|e| format!("run (mimi encode): --output {output_path}: {e}"))?;
            println!(
                "mimi encode: {} samples -> {} frames x {} codebooks -> {output_path}",
                clip.samples.len(),
                n_frames,
                n_q
            );
        }
        CodecMode::Decode => {
            let decoder = MimiNeuralDecoder::from_gguf(session.gguf(), &cfg)
                .map_err(|e| e.to_string())?
                .with_backend(a.backend);
            if codec.attrs.d_model != decoder.expected_feature_dim() {
                return Err(format!(
                    "run (mimi decode): effective table feature dimension {} != neural decoder input {}",
                    codec.attrs.d_model,
                    decoder.expected_feature_dim()
                ));
            }
            let bytes = std::fs::read(input_path)
                .map_err(|e| format!("run (mimi decode): {input_path}: {e}"))?;
            let container = MimiCodesV1::from_bytes(&bytes)?;
            validate_mimi_codes_for_model(&container, &cfg, &codec, model_sha256, hop)?;
            let n_frames = usize::try_from(container.n_frames)
                .map_err(|_| "run (mimi decode): frame count does not fit this host")?;
            let attrs = MimiRvqAttrs {
                n_codebooks: n_q,
                codebook_size: codec.attrs.codebook_size,
                d_model: codec.attrs.d_model,
            };
            let features =
                mimi_rvq_decode(&container.codes, n_frames, &codec.tables[..n_q], &attrs)
                    .map_err(|e| e.to_string())?;
            let pcm = decoder.decode_all(&features).map_err(|e| e.to_string())?;
            let expected_samples = usize::try_from(container.pcm_samples)
                .map_err(|_| "run (mimi decode): PCM sample count does not fit this host")?;
            if pcm.len() != expected_samples {
                return Err(format!(
                    "run (mimi decode): decoder emitted {} samples, container declares {expected_samples}",
                    pcm.len()
                ));
            }
            wav::write_wav(output_path, &pcm, cfg.sample_rate)
                .map_err(|e| format!("run (mimi decode): --output {output_path}: {e}"))?;
            println!(
                "mimi decode: {} frames x {} codebooks -> {} samples @ {} Hz -> {output_path}",
                n_frames,
                n_q,
                pcm.len(),
                cfg.sample_rate
            );
        }
    }
    Ok(())
}

/// Descript DAC's exact non-causal token-to-waveform path.
///
/// Input is headerless row-major `[frames, n_codebooks]` little-endian u32.
/// A raw contract is intentional: DAC's upstream `.dac` container is a NumPy
/// pickle object and cannot enter the zero-dependency runtime. Code range,
/// matrix width, model topology, and output extent are still validated before
/// a WAV is written.
fn run_dac_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (dac): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (dac): encode is not implemented; this public runtime surface is the complete released token-to-PCM decoder used by DAC-backed TTS models"
                    .to_owned(),
            );
        }
        None => {
            return Err("run (dac): --codec-mode decode is required".to_owned());
        }
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (dac): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (dac): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (dac decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (dac decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let model = vokra_models::dac::Dac::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let frames = codes.len() / model.n_codebooks();
    let pcm = model
        .decode_codes(&codes)
        .map_err(|error| error.to_string())?;
    let expected = model
        .output_samples(frames)
        .map_err(|error| error.to_string())?;
    if pcm.len() != expected {
        return Err(format!(
            "run (dac decode): decoder emitted {} samples, topology predicts {expected}",
            pcm.len()
        ));
    }
    wav::write_wav(output_path, &pcm, model.sample_rate())
        .map_err(|error| format!("run (dac decode): --output {output_path}: {error}"))?;
    println!(
        "dac decode: {frames} frames x {} codebooks -> {} samples @ {} Hz -> {output_path}",
        model.n_codebooks(),
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// WavTokenizer's released single-codebook token-to-waveform path.
///
/// Input is one headerless little-endian u32 code per 75 Hz frame. The model
/// validates every index against the 4096-entry public codebook. Upstream's
/// documented inference condition id is zero; callers may select another
/// released AdaLayerNorm row explicitly with `--bandwidth-id`.
fn run_wavtokenizer_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (wavtokenizer): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (wavtokenizer): encode is not implemented yet; the public runtime currently exposes the complete released token-to-PCM path, and never substitutes a different encoder"
                    .to_owned(),
            );
        }
        None => {
            return Err("run (wavtokenizer): --codec-mode decode is required".to_owned());
        }
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (wavtokenizer): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (wavtokenizer): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (wavtokenizer decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (wavtokenizer decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let condition_id = a.bandwidth_id.unwrap_or(0);
    let model = vokra_models::wavtokenizer::WavTokenizer::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let pcm = model
        .decode_codes_with_condition(&codes, condition_id)
        .map_err(|error| error.to_string())?;
    let expected = codes
        .len()
        .checked_mul(model.hop_length())
        .ok_or("run (wavtokenizer decode): output sample count overflow")?;
    if pcm.len() != expected {
        return Err(format!(
            "run (wavtokenizer decode): decoder emitted {} samples, expected {expected} from {} frames x hop {}",
            pcm.len(),
            codes.len(),
            model.hop_length()
        ));
    }
    wav::write_wav(output_path, &pcm, model.sample_rate())
        .map_err(|error| format!("run (wavtokenizer decode): --output {output_path}: {error}"))?;
    println!(
        "wavtokenizer decode: {} codes / condition {condition_id} -> {} samples @ {} Hz -> {output_path}",
        codes.len(),
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// NeuCodec base/distill shared 50 Hz FSQ token-to-waveform path.
///
/// Input is one headerless little-endian u32 code per frame. The strict
/// binder selects the exact published variant from metadata and refuses any
/// encoder/decoder manifest drift before decoding.
fn run_neucodec_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (neucodec): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (neucodec): encode is not implemented yet; the public runtime exposes the complete official token-to-PCM decoder and never substitutes another encoder"
                    .to_owned(),
            );
        }
        None => return Err("run (neucodec): --codec-mode decode is required".to_owned()),
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (neucodec): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (neucodec): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (neucodec decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (neucodec decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let model = vokra_models::neucodec::NeuCodec::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let pcm = model
        .decode_codes(&codes)
        .map_err(|error| error.to_string())?;
    let expected = codes
        .len()
        .checked_mul(model.hop_length())
        .ok_or("run (neucodec decode): output sample count overflow")?;
    if pcm.len() != expected {
        return Err(format!(
            "run (neucodec decode): decoder emitted {} samples, expected {expected} from {} frames x hop {}",
            pcm.len(),
            codes.len(),
            model.hop_length()
        ));
    }
    wav::write_wav(output_path, &pcm, model.sample_rate())
        .map_err(|error| format!("run (neucodec decode): --output {output_path}: {error}"))?;
    println!(
        "neucodec {:?} decode: {} codes -> {} samples @ {} Hz -> {output_path}",
        model.variant(),
        codes.len(),
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// X-Codec2 50 Hz FSQ token-to-waveform path.
///
/// Input is one headerless little-endian u32 code per frame. The binder pins
/// the exact public 1,153-tensor GGUF and its CC-BY-NC-4.0 provenance before
/// the shared decoder is allowed to read any weight.
fn run_xcodec2_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (xcodec2): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (xcodec2): encode is not implemented yet; the public runtime exposes the complete official token-to-PCM decoder and never substitutes another encoder"
                    .to_owned(),
            );
        }
        None => return Err("run (xcodec2): --codec-mode decode is required".to_owned()),
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (xcodec2): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (xcodec2): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (xcodec2 decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (xcodec2 decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let model = vokra_models::xcodec2::XCodec2::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let pcm = model
        .decode_codes(&codes)
        .map_err(|error| error.to_string())?;
    let expected = codes
        .len()
        .checked_mul(model.hop_length())
        .ok_or("run (xcodec2 decode): output sample count overflow")?;
    if pcm.len() != expected {
        return Err(format!(
            "run (xcodec2 decode): decoder emitted {} samples, expected {expected} from {} frames x hop {}",
            pcm.len(),
            codes.len(),
            model.hop_length()
        ));
    }
    wav::write_wav(output_path, &pcm, model.sample_rate())
        .map_err(|error| format!("run (xcodec2 decode): --output {output_path}: {error}"))?;
    println!(
        "xcodec2 decode: {} codes -> {} samples @ {} Hz -> {output_path}",
        codes.len(),
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// Alibaba DAMO FunCodec frame-major residual-VQ token-to-PCM path.
fn run_funcodec_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (funcodec): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (funcodec): PCM-to-token encode is not implemented for the audited release; no substitute encoder or CPU fallback is used"
                    .to_owned(),
            );
        }
        None => return Err("run (funcodec): --codec-mode decode is required".to_owned()),
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (funcodec): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (funcodec): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let model = vokra_models::funcodec::FunCodec::from_gguf_with_backend(session.gguf(), a.backend)
        .map_err(|error| error.to_string())?;
    let num_quantizers = a.num_quantizers.unwrap_or(model.max_quantizers());
    if !(1..=model.max_quantizers()).contains(&num_quantizers) {
        return Err(format!(
            "run (funcodec): --num-quantizers {num_quantizers} is outside release range 1..={}",
            model.max_quantizers()
        ));
    }
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (funcodec decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (funcodec decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    if !codes.len().is_multiple_of(num_quantizers) {
        return Err(format!(
            "run (funcodec decode): {} codes are not divisible by --num-quantizers {num_quantizers}; input must be frame-major [frames,num_quantizers]",
            codes.len()
        ));
    }
    let frames = codes.len() / num_quantizers;
    let pcm = model
        .decode_frame_major(&codes, frames, num_quantizers)
        .map_err(|error| error.to_string())?;
    wav::write_wav(output_path, &pcm, model.sample_rate())
        .map_err(|error| format!("run (funcodec decode): --output {output_path}: {error}"))?;
    println!(
        "funcodec decode: {frames} frames x {num_quantizers} quantizers -> {} samples @ {} Hz -> {output_path}",
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// Fudan/OpenMOSS SpeechTokenizer frame-major residual-VQ token-to-PCM path.
fn run_speechtokenizer_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (speechtokenizer): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (speechtokenizer): PCM-to-token encode is not implemented for the audited release; no substitute encoder or CPU fallback is used"
                    .to_owned(),
            );
        }
        None => {
            return Err("run (speechtokenizer): --codec-mode decode is required".to_owned());
        }
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (speechtokenizer): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (speechtokenizer): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let model = vokra_models::speechtokenizer::SpeechTokenizer::from_gguf_with_backend(
        session.gguf(),
        a.backend,
    )
    .map_err(|error| error.to_string())?;
    let num_quantizers = a.num_quantizers.unwrap_or(model.max_quantizers());
    if !(1..=model.max_quantizers()).contains(&num_quantizers) {
        return Err(format!(
            "run (speechtokenizer): --num-quantizers {num_quantizers} is outside release range 1..={}",
            model.max_quantizers()
        ));
    }
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (speechtokenizer decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (speechtokenizer decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    if !codes.len().is_multiple_of(num_quantizers) {
        return Err(format!(
            "run (speechtokenizer decode): {} codes are not divisible by --num-quantizers {num_quantizers}; input must be frame-major [frames,num_quantizers]",
            codes.len()
        ));
    }
    let frames = codes.len() / num_quantizers;
    let pcm = model
        .decode_frame_major(&codes, frames, num_quantizers)
        .map_err(|error| error.to_string())?;
    wav::write_wav(output_path, &pcm, model.sample_rate()).map_err(|error| {
        format!("run (speechtokenizer decode): --output {output_path}: {error}")
    })?;
    println!(
        "speechtokenizer decode: {frames} frames x {num_quantizers} quantizers -> {} samples @ {} Hz -> {output_path}",
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// OpenMOSS Audio Tokenizer raw code-matrix to PCM path.
///
/// Nano accepts a caller-declared residual-LFQ width and emits standard
/// 48 kHz stereo-interleaved float WAV. Full retains the session's mmap and
/// decodes its separate 24 kHz mono / 32-codebook topology one layer at a
/// time; the two release contracts are never interchanged.
fn run_moss_audio_tokenizer_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (moss_audio_tokenizer): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (moss_audio_tokenizer): PCM-to-token encode is not implemented for either audited release; no substitute encoder or CPU fallback is used"
                    .to_owned(),
            );
        }
        None => {
            return Err("run (moss_audio_tokenizer): --codec-mode decode is required".to_owned());
        }
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (moss_audio_tokenizer): --input <codes.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (moss_audio_tokenizer): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let model =
        vokra_models::moss_audio_tokenizer::MossAudioTokenizer::from_gguf_mapped_with_backend(
            session.gguf_arc(),
            a.backend,
        )
        .map_err(|error| error.to_string())?;
    if model.requires_metadata_repair() {
        eprintln!(
            "vokra: WARNING: this exact public Nano tensor manifest carries the historical Full metadata stamp; routing is authenticated by the complete manifest, but replace the artifact with a correctly stamped publication when authorized"
        );
    }
    let num_quantizers = a.num_quantizers.unwrap_or(model.max_quantizers());
    if num_quantizers > model.max_quantizers() {
        return Err(format!(
            "run (moss_audio_tokenizer): --num-quantizers {num_quantizers} exceeds {:?} maximum {}",
            model.variant(),
            model.max_quantizers()
        ));
    }
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (moss_audio_tokenizer decode): {input_path}: {error}"))?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "run (moss_audio_tokenizer decode): {input_path} has {} bytes; expected a positive multiple of four for u32le codes",
            bytes.len()
        ));
    }
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    if !codes.len().is_multiple_of(num_quantizers) {
        return Err(format!(
            "run (moss_audio_tokenizer decode): {} codes are not divisible by --num-quantizers {num_quantizers}; input must be frame-major [frames,num_quantizers]",
            codes.len()
        ));
    }
    let frames = codes.len() / num_quantizers;
    let audio = model
        .decode_frame_major(&codes, frames, num_quantizers)
        .map_err(|error| error.to_string())?;
    wav::write_wav_channels(output_path, &audio.pcm, audio.sample_rate, audio.channels).map_err(
        |error| format!("run (moss_audio_tokenizer decode): --output {output_path}: {error}"),
    )?;
    println!(
        "moss_audio_tokenizer {:?} decode: {frames} frames x {num_quantizers} quantizers -> {} samples/channel x {} channels @ {} Hz -> {output_path}",
        model.variant(),
        audio.samples_per_channel,
        audio.channels,
        audio.sample_rate
    );
    Ok(())
}

/// MOSS-TTS Nano explicit prompt-matrix to native codec decode.
///
/// The text SentencePiece model is not present in the public GGUF, so the CLI
/// accepts only the fully assembled upstream-compatible 17-column rows. Both
/// independently authenticated GGUFs select the same backend before inference.
fn run_moss_tts_nano(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() {
        return Err(
            "run (moss_tts Nano): --text is unavailable because the public GGUF does not bundle the pinned tokenizer.model; pass explicit [rows,17] u32le prompt ids with --input"
                .to_owned(),
        );
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (moss_tts Nano): --input <prompt-rows.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (moss_tts Nano): --output <stereo.wav> is required")?;
    let codec_path = a.audio_tokenizer.as_deref().ok_or(
        "run (moss_tts Nano): --audio-tokenizer <moss-audio-tokenizer-nano.gguf> is required",
    )?;
    let max_new_frames = a
        .max_new_frames
        .ok_or("run (moss_tts Nano): --max-new-frames <N> is required")?;

    let policy = vokra_core::CompliancePolicy::from_env();
    vokra_core::check_weight_license(session.gguf(), &policy).map_err(|error| error.to_string())?;
    let codec_file = std::sync::Arc::new(
        vokra_mmap::open_gguf(codec_path)
            .map_err(|error| format!("run (moss_tts Nano): codec {codec_path}: {error}"))?,
    );
    vokra_core::check_weight_license(&codec_file, &policy).map_err(|error| error.to_string())?;

    let model =
        vokra_models::moss_tts::MossTtsNano::from_gguf_with_backend(session.gguf(), a.backend)
            .map_err(|error| error.to_string())?;
    let codec =
        vokra_models::moss_audio_tokenizer::MossAudioTokenizer::from_gguf_mapped_with_backend(
            codec_file, a.backend,
        )
        .map_err(|error| error.to_string())?;
    if model.requires_metadata_repair() {
        eprintln!(
            "vokra: WARNING: this exact MOSS-TTS Nano manifest has the historical rope_base=0 header; runtime routing uses the authenticated 194-tensor contract, but the artifact still needs an authorized metadata replacement"
        );
    }
    if codec.requires_metadata_repair() {
        eprintln!(
            "vokra: WARNING: this exact MOSS Audio Tokenizer Nano manifest has the historical Full metadata stamp; runtime routing is manifest-authenticated, but the artifact still needs an authorized metadata replacement"
        );
    }

    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (moss_tts Nano): {input_path}: {error}"))?;
    const ROW_BYTES: usize = 17 * 4;
    if bytes.is_empty() || !bytes.len().is_multiple_of(ROW_BYTES) {
        return Err(format!(
            "run (moss_tts Nano): {input_path} has {} bytes; expected a positive multiple of {ROW_BYTES} for frame-major [rows,17] u32le prompt ids",
            bytes.len()
        ));
    }
    let prompt_rows: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let synthesis = model
        .synthesize_prompt_rows(&codec, &prompt_rows, max_new_frames)
        .map_err(|error| error.to_string())?;
    wav::write_wav_channels(
        output_path,
        &synthesis.audio.pcm,
        synthesis.audio.sample_rate,
        synthesis.audio.channels,
    )
    .map_err(|error| format!("run (moss_tts Nano): --output {output_path}: {error}"))?;
    println!(
        "moss_tts Nano: {} prompt rows -> {} generated frames x {} codebooks -> {} samples/channel x {} channels @ {} Hz -> {output_path}",
        prompt_rows.len() / 17,
        synthesis.generated.frames,
        synthesis.generated.num_codebooks,
        synthesis.audio.samples_per_channel,
        synthesis.audio.channels,
        synthesis.audio.sample_rate
    );
    Ok(())
}

/// MOSS-TTS Base/v1.5 explicit 33-column prompt to Full codec decode.
///
/// The official tokenizer/template remains an explicit caller companion. The
/// mapped 8B LLM and 7 GB codec retain their mappings and execute every learned
/// reduction on the same selected CPU/Metal backend.
fn run_moss_tts_delay(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() {
        return Err(
            "run (moss_tts Base/v1.5): --text is unavailable because the GGUF does not bundle the pinned tokenizer/template; pass explicit [rows,33] u32le prompt ids with --input"
                .to_owned(),
        );
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (moss_tts Base/v1.5): --input <prompt-rows.u32le> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (moss_tts Base/v1.5): --output <mono.wav> is required")?;
    let codec_path = a.audio_tokenizer.as_deref().ok_or(
        "run (moss_tts Base/v1.5): --audio-tokenizer <moss-audio-tokenizer-full.gguf> is required",
    )?;
    let max_new_tokens = a
        .max_new_frames
        .ok_or("run (moss_tts Base/v1.5): --max-new-frames <N> is required")?;

    let policy = vokra_core::CompliancePolicy::from_env();
    vokra_core::check_weight_license(session.gguf(), &policy).map_err(|error| error.to_string())?;
    let codec_file = std::sync::Arc::new(
        vokra_mmap::open_gguf(codec_path)
            .map_err(|error| format!("run (moss_tts Base/v1.5): codec {codec_path}: {error}"))?,
    );
    vokra_core::check_weight_license(&codec_file, &policy).map_err(|error| error.to_string())?;

    let checkpoint =
        vokra_models::moss_tts::MossTtsDelayCheckpoint::from_gguf_mapped(session.gguf_arc())
            .map_err(|error| error.to_string())?;
    let model = vokra_models::moss_tts::MossTtsDelay::from_checkpoint(checkpoint, a.backend)
        .map_err(|error| error.to_string())?;
    let codec =
        vokra_models::moss_audio_tokenizer::MossAudioTokenizer::from_gguf_mapped_with_backend(
            codec_file, a.backend,
        )
        .map_err(|error| error.to_string())?;

    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (moss_tts Base/v1.5): {input_path}: {error}"))?;
    const ROW_BYTES: usize = 33 * 4;
    if bytes.is_empty() || !bytes.len().is_multiple_of(ROW_BYTES) {
        return Err(format!(
            "run (moss_tts Base/v1.5): {input_path} has {} bytes; expected a positive multiple of {ROW_BYTES} for frame-major [rows,33] u32le prompt ids",
            bytes.len()
        ));
    }
    let prompt_rows: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let options = vokra_models::moss_tts::MossTtsDelayGenerationOptions {
        max_new_tokens,
        ..Default::default()
    };
    let release = model.checkpoint().release();
    let synthesis = model
        .synthesize_prompt_rows(&codec, &prompt_rows, &options)
        .map_err(|error| error.to_string())?;
    let [segment] = synthesis.segments.as_slice() else {
        return Err(format!(
            "run (moss_tts Base/v1.5): generation produced {} audio segments; the single-WAV CLI requires exactly one (the library API preserves zero/multiple official segments explicitly)",
            synthesis.segments.len()
        ));
    };
    wav::write_wav_channels(
        output_path,
        &segment.audio.pcm,
        segment.audio.sample_rate,
        segment.audio.channels,
    )
    .map_err(|error| format!("run (moss_tts Base/v1.5): --output {output_path}: {error}"))?;
    println!(
        "moss_tts {release:?}: {} prompt rows -> {} generated delayed rows -> {} codec frames (trimmed {} prefix samples) -> {} mono samples @ {} Hz -> {output_path}",
        prompt_rows.len() / 33,
        synthesis.generated.row_count(),
        segment.frames,
        segment.trimmed_prefix_samples,
        segment.audio.samples_per_channel,
        segment.audio.sample_rate,
    );
    Ok(())
}

/// MOSS-VoiceGenerator explicit 17-column prompt to Full codec decode.
fn run_moss_voice_generator(session: &Session, a: &RunArgs) -> Result<(), String> {
    const RUN_LABEL: &str = "run (moss-voice-generator)";
    if a.text.is_some() {
        return Err(format!(
            "{RUN_LABEL}: --text is unavailable because the GGUF does not bundle the pinned tokenizer/template; pass explicit [rows,17] u32le prompt ids with --input"
        ));
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or_else(|| format!("{RUN_LABEL}: --input <prompt-rows.u32le> is required"))?;
    let output_path = a
        .output
        .as_deref()
        .ok_or_else(|| format!("{RUN_LABEL}: --output <mono.wav> is required"))?;
    let codec_path = a.audio_tokenizer.as_deref().ok_or_else(|| {
        format!("{RUN_LABEL}: --audio-tokenizer <moss-audio-tokenizer-full.gguf> is required")
    })?;
    let max_new_tokens = a
        .max_new_frames
        .ok_or_else(|| format!("{RUN_LABEL}: --max-new-frames <N> is required"))?;

    let policy = vokra_core::CompliancePolicy::from_env();
    vokra_core::check_weight_license(session.gguf(), &policy).map_err(|error| error.to_string())?;
    let codec_file = std::sync::Arc::new(
        vokra_mmap::open_gguf(codec_path)
            .map_err(|error| format!("{RUN_LABEL}: codec {codec_path}: {error}"))?,
    );
    vokra_core::check_weight_license(&codec_file, &policy).map_err(|error| error.to_string())?;

    let checkpoint =
        vokra_models::moss_tts::MossVoiceGeneratorCheckpoint::from_gguf_mapped(session.gguf_arc())
            .map_err(|error| error.to_string())?;
    let requires_metadata_repair = checkpoint.requires_metadata_repair();
    let model = vokra_models::moss_tts::MossVoiceGenerator::from_checkpoint(checkpoint, a.backend)
        .map_err(|error| error.to_string())?;
    let codec =
        vokra_models::moss_audio_tokenizer::MossAudioTokenizer::from_gguf_mapped_with_backend(
            codec_file, a.backend,
        )
        .map_err(|error| error.to_string())?;

    let bytes =
        std::fs::read(input_path).map_err(|error| format!("{RUN_LABEL}: {input_path}: {error}"))?;
    const ROW_BYTES: usize = 17 * 4;
    if bytes.is_empty() || !bytes.len().is_multiple_of(ROW_BYTES) {
        return Err(format!(
            "{RUN_LABEL}: {input_path} has {} bytes; expected a positive multiple of {ROW_BYTES} for frame-major [rows,17] u32le prompt ids",
            bytes.len()
        ));
    }
    let prompt_rows: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    let mut options = model.default_generation_options();
    options.max_new_tokens = max_new_tokens;
    let synthesis = model
        .synthesize_prompt_rows(&codec, &prompt_rows, &options)
        .map_err(|error| error.to_string())?;
    let [segment] = synthesis.segments.as_slice() else {
        return Err(format!(
            "{RUN_LABEL}: generation produced {} audio segments; the single-WAV CLI requires exactly one (the library API preserves zero/multiple official segments explicitly)",
            synthesis.segments.len()
        ));
    };
    wav::write_wav_channels(
        output_path,
        &segment.audio.pcm,
        segment.audio.sample_rate,
        segment.audio.channels,
    )
    .map_err(|error| format!("{RUN_LABEL}: --output {output_path}: {error}"))?;
    if requires_metadata_repair {
        eprintln!(
            "warning: {RUN_LABEL}: the exact historical public GGUF has the authenticated VoiceGenerator tensor manifest but a stale MOSS-TTS 8B header; execution used the strict 343-tensor VoiceGenerator contract"
        );
    }
    println!(
        "moss-voice-generator: {} prompt rows -> {} generated delayed rows -> {} codec frames (trimmed {} prefix samples) -> {} mono samples @ {} Hz -> {output_path}",
        prompt_rows.len() / 17,
        synthesis.generated.row_count(),
        segment.frames,
        segment.trimmed_prefix_samples,
        segment.audio.samples_per_channel,
        segment.audio.sample_rate,
    );
    Ok(())
}

/// MioCodec 25 Hz FSQ + global-embedding to 44.1 kHz waveform path.
///
/// VKRMIO01 is versioned because raw codes alone are insufficient: the
/// official decoder also requires a 128-dimensional global embedding and an
/// explicit target sample length for its interpolation/iSTFT geometry.
fn run_miocodec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (miocodec): --text/--tokens are not codec inputs; use --input plus --codec-mode decode"
                .to_owned(),
        );
    }
    match a.codec_mode {
        Some(CodecMode::Decode) => {}
        Some(CodecMode::Encode) => {
            return Err(
                "run (miocodec): encode is not implemented; the official token + global-embedding to waveform decoder is available, and Vokra never substitutes another encoder"
                    .to_owned(),
            );
        }
        None => return Err("run (miocodec): --codec-mode decode is required".to_owned()),
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (miocodec): --input <tokens.vmi> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (miocodec): --output <out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (miocodec decode): {input_path}: {error}"))?;
    let input = vokra_models::miocodec::MioCodecDecodeInput::from_bytes(&bytes)
        .map_err(|error| format!("run (miocodec decode): {input_path}: {error}"))?;
    let model = vokra_models::miocodec::MioCodec::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let pcm = model
        .decode_input(&input)
        .map_err(|error| error.to_string())?;
    wav::write_wav(output_path, &pcm, model.sample_rate())
        .map_err(|error| format!("run (miocodec decode): --output {output_path}: {error}"))?;
    println!(
        "miocodec decode: {} codes / target {} -> {} samples @ {} Hz -> {output_path}",
        input.codes.len(),
        input.target_samples,
        pcm.len(),
        model.sample_rate()
    );
    Ok(())
}

/// Fingerprints the exact SNAC tensor contract without hashing unrelated GGUF
/// metadata. The strict binder separately validates architecture, provenance,
/// license, and the complete manifest. This ledger pins every tensor's name,
/// dtype, shape, and payload so a code container cannot be decoded with a
/// sibling or modified checkpoint that happens to share the same topology.
fn codec_model_fingerprint(
    file: &vokra_core::gguf::GgufFile,
    domain: &[u8],
    label: &str,
) -> Result<[u8; 32], String> {
    let mut ledger = Vec::new();
    ledger.extend_from_slice(domain);
    ledger.extend_from_slice(
        &u64::try_from(file.tensors().len())
            .map_err(|_| format!("run ({label}): tensor count exceeds u64"))?
            .to_le_bytes(),
    );
    for info in file.tensors() {
        let name = info.name.as_bytes();
        ledger.extend_from_slice(
            &u64::try_from(name.len())
                .map_err(|_| format!("run ({label}): tensor name length exceeds u64"))?
                .to_le_bytes(),
        );
        ledger.extend_from_slice(name);
        ledger.extend_from_slice(&info.dtype.tag().to_le_bytes());
        ledger.extend_from_slice(
            &u32::try_from(info.dimensions.len())
                .map_err(|_| format!("run ({label}): tensor rank exceeds u32"))?
                .to_le_bytes(),
        );
        for &dimension in &info.dimensions {
            ledger.extend_from_slice(&dimension.to_le_bytes());
        }
        let payload = file.tensor_bytes(info);
        ledger.extend_from_slice(
            &u64::try_from(payload.len())
                .map_err(|_| format!("run ({label}): tensor byte length exceeds u64"))?
                .to_le_bytes(),
        );
        ledger.extend_from_slice(&sha256(payload));
    }
    Ok(sha256(&ledger))
}

fn snac_model_fingerprint(file: &vokra_core::gguf::GgufFile) -> Result<[u8; 32], String> {
    codec_model_fingerprint(file, b"vokra-snac-gguf-tensor-fingerprint-v1", "snac")
}

fn focalcodec_model_fingerprint(file: &vokra_core::gguf::GgufFile) -> Result<[u8; 32], String> {
    codec_model_fingerprint(
        file,
        b"vokra-focalcodec-gguf-tensor-fingerprint-v1",
        "focalcodec",
    )
}

fn validate_snac_codes_for_model(
    codes: &SnacCodesV1,
    model: &vokra_models::snac::Snac,
    model_sha256: [u8; 32],
) -> Result<(), String> {
    let config = model.config();
    let n_stages = u32::try_from(config.n_stages)
        .map_err(|_| "run (snac decode): model stage count exceeds u32")?;
    let codebook_size = u32::try_from(model.codebook_size())
        .map_err(|_| "run (snac decode): model codebook size exceeds u32")?;
    let latent_dim = u32::try_from(model.latent_dim())
        .map_err(|_| "run (snac decode): model latent dimension exceeds u32")?;
    let hop_length = u32::try_from(model.hop_length())
        .map_err(|_| "run (snac decode): model hop length exceeds u32")?;
    if codes.sample_rate != model.sample_rate()
        || codes.n_stages != n_stages
        || codes.codebook_size != codebook_size
        || codes.latent_dim != latent_dim
        || codes.hop_length != hop_length
        || codes.vq_strides != config.vq_strides
    {
        return Err(format!(
            "run (snac decode): container/model contract mismatch: container \
             rate={} stages={} bins={} latent={} hop={} strides={:?}, model \
             rate={} stages={} bins={} latent={} hop={} strides={:?}",
            codes.sample_rate,
            codes.n_stages,
            codes.codebook_size,
            codes.latent_dim,
            codes.hop_length,
            codes.vq_strides,
            model.sample_rate(),
            n_stages,
            codebook_size,
            latent_dim,
            hop_length,
            config.vq_strides
        ));
    }
    if codes.model_sha256 != model_sha256 {
        return Err(format!(
            "run (snac decode): model SHA-256 mismatch (container {}, model {}); \
             hierarchical codes must be decoded by the exact GGUF tensors that encoded them",
            hex_digest(&codes.model_sha256),
            hex_digest(&model_sha256)
        ));
    }
    Ok(())
}

/// SNAC 24/44 kHz waveform encode and hierarchical-code decode.
///
/// The encoder pads exactly as the upstream runtime requires and records the
/// caller's original sample count. Decode validates the padded topology,
/// requires the exact tensor fingerprint, then trims only that recorded tail.
/// Metal encode reaches the model's explicit unsupported-operation error;
/// there is no host-side codebook-search fallback.
fn run_snac_codec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (snac): --text/--tokens are not codec inputs; use --input plus \
             --codec-mode encode|decode"
                .to_owned(),
        );
    }
    let mode = a
        .codec_mode
        .ok_or("run (snac): --codec-mode encode|decode is required")?;
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (snac): --input <wav|codes.vsc> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (snac): --output <codes.vsc|wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    if let Some(info) = vokra_core::resolve_attribution(session.gguf()) {
        eprintln!("vokra: ATTRIBUTION ({}) {}", info.license, info.text);
    }
    let model = vokra_models::snac::Snac::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let model_sha256 = snac_model_fingerprint(session.gguf())?;

    match mode {
        CodecMode::Encode => {
            let clip = wav::read_wav(input_path)
                .map_err(|error| format!("run (snac encode): {input_path}: {error}"))?;
            if clip.sample_rate != model.sample_rate() {
                return Err(format!(
                    "run (snac encode): {input_path} is {} Hz, model requires {} Hz — \
                     resample offline first (FR-EX-08: never a silent resample)",
                    clip.sample_rate,
                    model.sample_rate()
                ));
            }
            let codes = model
                .encode(&clip.samples, clip.sample_rate)
                .map_err(|error| error.to_string())?;
            let config = model.config();
            if codes.len() != config.n_stages {
                return Err(format!(
                    "run (snac encode): encoder returned {} stages, model declares {}",
                    codes.len(),
                    config.n_stages
                ));
            }
            let mut base_frames = None;
            let mut stage_lengths = [0u64; 4];
            for (stage, (stage_codes, &stride)) in
                codes.iter().zip(config.active_vq_strides()).enumerate()
            {
                let frames = stage_codes
                    .len()
                    .checked_mul(stride as usize)
                    .ok_or("run (snac encode): base frame count overflow")?;
                if let Some(expected) = base_frames {
                    if frames != expected {
                        return Err(format!(
                            "run (snac encode): stage {stage} implies {frames} base frames, expected {expected}"
                        ));
                    }
                } else {
                    base_frames = Some(frames);
                }
                stage_lengths[stage] = u64::try_from(stage_codes.len())
                    .map_err(|_| "run (snac encode): stage length exceeds u64")?;
            }
            let base_frames = base_frames.ok_or("run (snac encode): encoder returned no stages")?;
            let container = SnacCodesV1 {
                sample_rate: model.sample_rate(),
                n_stages: u32::try_from(config.n_stages)
                    .map_err(|_| "run (snac encode): stage count exceeds u32")?,
                codebook_size: u32::try_from(model.codebook_size())
                    .map_err(|_| "run (snac encode): codebook size exceeds u32")?,
                latent_dim: u32::try_from(model.latent_dim())
                    .map_err(|_| "run (snac encode): latent dimension exceeds u32")?,
                hop_length: u32::try_from(model.hop_length())
                    .map_err(|_| "run (snac encode): hop length exceeds u32")?,
                base_frames: u64::try_from(base_frames)
                    .map_err(|_| "run (snac encode): base frame count exceeds u64")?,
                pcm_samples: u64::try_from(clip.samples.len())
                    .map_err(|_| "run (snac encode): PCM sample count exceeds u64")?,
                model_sha256,
                vq_strides: config.vq_strides,
                stage_lengths,
                codes,
            };
            std::fs::write(output_path, container.to_bytes()?)
                .map_err(|error| format!("run (snac encode): --output {output_path}: {error}"))?;
            println!(
                "snac encode: {} samples -> {} base frames / {} hierarchical stages -> {output_path}",
                clip.samples.len(),
                base_frames,
                config.n_stages
            );
        }
        CodecMode::Decode => {
            let bytes = std::fs::read(input_path)
                .map_err(|error| format!("run (snac decode): {input_path}: {error}"))?;
            let container = SnacCodesV1::from_bytes(&bytes)?;
            validate_snac_codes_for_model(&container, &model, model_sha256)?;
            let mut pcm = model
                .decode(&container.codes)
                .map_err(|error| error.to_string())?;
            let expected_padded = usize::try_from(container.base_frames)
                .ok()
                .and_then(|frames| frames.checked_mul(model.hop_length()))
                .ok_or("run (snac decode): padded PCM sample count does not fit this host")?;
            if pcm.len() != expected_padded {
                return Err(format!(
                    "run (snac decode): decoder emitted {} samples, container topology predicts {expected_padded}",
                    pcm.len()
                ));
            }
            let original_samples = usize::try_from(container.pcm_samples).map_err(
                |_| "run (snac decode): original PCM sample count does not fit this host",
            )?;
            pcm.truncate(original_samples);
            wav::write_wav(output_path, &pcm, model.sample_rate())
                .map_err(|error| format!("run (snac decode): --output {output_path}: {error}"))?;
            println!(
                "snac decode: {} base frames / {} hierarchical stages -> {} samples @ {} Hz -> {output_path}",
                container.base_frames,
                container.n_stages,
                pcm.len(),
                model.sample_rate()
            );
        }
    }
    Ok(())
}

fn validate_focalcodec_codes_for_model(
    codes: &FocalCodecCodesV1,
    model: &vokra_models::focalcodec::FocalCodec,
    model_sha256: [u8; 32],
) -> Result<(), String> {
    let frame_hop = u32::try_from(model.frame_hop())
        .map_err(|_| "run (focalcodec decode): frame hop exceeds u32")?;
    let codebook_size = u32::try_from(model.codebook_size())
        .map_err(|_| "run (focalcodec decode): codebook size exceeds u32")?;
    let code_dimension = u32::try_from(model.code_dimension())
        .map_err(|_| "run (focalcodec decode): code dimension exceeds u32")?;
    if codes.sample_rate != model.sample_rate()
        || codes.token_hz_times_two != model.variant().token_hz_times_two()
        || codes.frame_hop != frame_hop
        || codes.codebook_size != codebook_size
        || codes.code_dimension != code_dimension
    {
        return Err(format!(
            "run (focalcodec decode): container/model contract mismatch: container \
             rate={} token_hz_times_two={} hop={} bins={} dim={}, model \
             rate={} token_hz_times_two={} hop={} bins={} dim={}",
            codes.sample_rate,
            codes.token_hz_times_two,
            codes.frame_hop,
            codes.codebook_size,
            codes.code_dimension,
            model.sample_rate(),
            model.variant().token_hz_times_two(),
            frame_hop,
            codebook_size,
            code_dimension
        ));
    }
    if codes.model_sha256 != model_sha256 {
        return Err(format!(
            "run (focalcodec decode): model SHA-256 mismatch (container {}, model {}); \
             BSQ tokens must be decoded by the exact GGUF tensors that encoded them",
            hex_digest(&codes.model_sha256),
            hex_digest(&model_sha256)
        ));
    }
    Ok(())
}

/// FocalCodec 50 / 25 / 12.5 Hz waveform encode and BSQ-token decode.
///
/// The versioned container pins the complete tensor fingerprint and the
/// variant timebase. Decode reproduces upstream's explicit `output_length`
/// contract: excess decoder samples are truncated and a short decoder output
/// is extended by repeating its final sample. This is deterministic host glue;
/// all learned encoder, compressor, decompressor, and Vocos operations dispatch
/// through the selected `Compute` backend.
fn run_focalcodec(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() {
        return Err(
            "run (focalcodec): --text/--tokens are not codec inputs; use --input plus \
             --codec-mode encode|decode"
                .to_owned(),
        );
    }
    let mode = a
        .codec_mode
        .ok_or("run (focalcodec): --codec-mode encode|decode is required")?;
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (focalcodec): --input <16k-mono.wav|codes.vfc> is required")?;
    let output_path = a
        .output
        .as_deref()
        .ok_or("run (focalcodec): --output <codes.vfc|out.wav> is required")?;

    vokra_core::check_weight_license(session.gguf(), &vokra_core::CompliancePolicy::from_env())
        .map_err(|error| error.to_string())?;
    if let Some(info) = vokra_core::resolve_attribution(session.gguf()) {
        eprintln!("vokra: ATTRIBUTION ({}) {}", info.license, info.text);
    }
    let model = vokra_models::focalcodec::FocalCodec::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let model_sha256 = focalcodec_model_fingerprint(session.gguf())?;

    match mode {
        CodecMode::Encode => {
            let clip = wav::read_wav(input_path)
                .map_err(|error| format!("run (focalcodec encode): {input_path}: {error}"))?;
            if clip.sample_rate != model.sample_rate() {
                return Err(format!(
                    "run (focalcodec encode): {input_path} is {} Hz, model requires {} Hz — \
                     resample offline first (FR-EX-08: never a silent resample)",
                    clip.sample_rate,
                    model.sample_rate()
                ));
            }
            let tokens = model
                .encode(&clip.samples)
                .map_err(|error| error.to_string())?;
            let container = FocalCodecCodesV1 {
                sample_rate: model.sample_rate(),
                token_hz_times_two: model.variant().token_hz_times_two(),
                frame_hop: u32::try_from(model.frame_hop())
                    .map_err(|_| "run (focalcodec encode): frame hop exceeds u32")?,
                codebook_size: u32::try_from(model.codebook_size())
                    .map_err(|_| "run (focalcodec encode): codebook size exceeds u32")?,
                code_dimension: u32::try_from(model.code_dimension())
                    .map_err(|_| "run (focalcodec encode): code dimension exceeds u32")?,
                token_count: u64::try_from(tokens.len())
                    .map_err(|_| "run (focalcodec encode): token count exceeds u64")?,
                pcm_samples: u64::try_from(clip.samples.len())
                    .map_err(|_| "run (focalcodec encode): PCM sample count exceeds u64")?,
                model_sha256,
                tokens,
            };
            let token_count = container.token_count;
            std::fs::write(output_path, container.to_bytes()?).map_err(|error| {
                format!("run (focalcodec encode): --output {output_path}: {error}")
            })?;
            println!(
                "focalcodec {} encode: {} samples -> {token_count} BSQ tokens -> {output_path}",
                model.variant().tag(),
                clip.samples.len()
            );
        }
        CodecMode::Decode => {
            let bytes = std::fs::read(input_path)
                .map_err(|error| format!("run (focalcodec decode): {input_path}: {error}"))?;
            let container = FocalCodecCodesV1::from_bytes(&bytes)?;
            validate_focalcodec_codes_for_model(&container, &model, model_sha256)?;
            let mut pcm = model
                .decode(&container.tokens)
                .map_err(|error| error.to_string())?;
            let raw_samples = pcm.len();
            let original_samples = usize::try_from(container.pcm_samples)
                .map_err(|_| "run (focalcodec decode): PCM sample count does not fit this host")?;
            if pcm.len() > original_samples {
                pcm.truncate(original_samples);
            } else if pcm.len() < original_samples {
                let tail = *pcm
                    .last()
                    .ok_or("run (focalcodec decode): decoder returned empty PCM")?;
                pcm.resize(original_samples, tail);
            }
            wav::write_wav(output_path, &pcm, model.sample_rate()).map_err(|error| {
                format!("run (focalcodec decode): --output {output_path}: {error}")
            })?;
            println!(
                "focalcodec {} decode: {} BSQ tokens / {raw_samples} raw samples -> {} samples @ {} Hz -> {output_path}",
                model.variant().tag(),
                container.token_count,
                pcm.len(),
                model.sample_rate()
            );
        }
    }
    Ok(())
}

/// BigVGAN's explicit mel-file CLI contract. The raw input is channel-major
/// `[n_mels, frames]` little-endian f32; no header, transpose, padding, or
/// inferred normalization is applied.
fn run_bigvgan(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() || a.codec_mode.is_some() {
        return Err(
            "run (bigvgan): --text/--tokens/--codec-mode are not vocoder inputs; pass a raw channel-major mel file with --input"
                .to_owned(),
        );
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (bigvgan): --input <mel.f32> is required")?;
    let model = vokra_models::bigvgan::BigVGan::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let n_mels = model.config().in_channels as usize;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (bigvgan): --input {input_path}: {error}"))?;
    let (mel, frames) = parse_vocoder_feature_bytes(&bytes, n_mels, input_path, "bigvgan")?;
    let pcm = model
        .decode(&mel, frames)
        .map_err(|error| error.to_string())?;
    emit_audio(
        "bigvgan",
        &pcm,
        model.variant().sample_rate(),
        a.output.as_deref(),
    )
}

/// Both standalone HiFi-GAN variants use the same explicit file layout as
/// BigVGAN. The strict model binder applies variant-owned preprocessing
/// (SpeechT5 mean/scale normalization or SpeechBrain replicate padding); the
/// CLI does not infer or duplicate it.
fn run_hifigan(session: &Session, a: &RunArgs) -> Result<(), String> {
    const TASK: &str = "hifigan";
    if a.text.is_some() || a.tokens.is_some() || a.codec_mode.is_some() {
        return Err(
            "run (hifigan): --text/--tokens/--codec-mode are not vocoder inputs; pass a raw channel-major mel file with --input"
                .to_owned(),
        );
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (hifigan): --input <mel.f32> is required")?;
    let model = vokra_models::hifigan::HiFiGan::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let n_mels = model.attrs().n_mels;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (hifigan): --input {input_path}: {error}"))?;
    let (mel, frames) = parse_vocoder_feature_bytes(&bytes, n_mels, input_path, TASK)?;
    let pcm = model
        .decode(&mel, frames)
        .map_err(|error| error.to_string())?;
    let variant = match model.sample_rate() {
        16_000 => "speecht5_hifigan",
        22_050 => "hifigan_vocoder",
        _ => TASK,
    };
    emit_audio(variant, &pcm, model.sample_rate(), a.output.as_deref())
}

/// Vocos-family explicit raw-feature contract. Encodec and YuE features are
/// assumed to have been produced by their matching codec frontend; this
/// runtime does not bundle or silently substitute a neural frontend.
fn run_vocos(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.text.is_some() || a.tokens.is_some() || a.codec_mode.is_some() {
        return Err(
            "run (vocos): --text/--tokens/--codec-mode are not vocoder inputs; pass raw channel-major features with --input"
                .to_owned(),
        );
    }
    let arch = session
        .gguf()
        .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or("run (vocos): missing model architecture")?;
    if arch == vokra_models::yue_upsampler::ARCH {
        return run_yue_upsampler(session, a);
    }
    if arch != vokra_models::vocos::ARCH {
        return Err(format!(
            "run (vocos): internal dispatch error for architecture {arch:?}"
        ));
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (vocos): --input <features.f32> is required")?;
    let model = vokra_models::vocos::Vocos::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(a.backend);
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (vocos): --input {input_path}: {error}"))?;
    let (features, frames) =
        parse_vocoder_feature_bytes(&bytes, model.config().n_input, input_path, "vocos")?;
    let pcm = match model.variant() {
        vokra_models::vocos::VocosVariant::Mel24khz => {
            if a.bandwidth_id.is_some() {
                return Err(
                    "run (vocos): --bandwidth-id is invalid for mel_24khz (plain LayerNorm)"
                        .to_owned(),
                );
            }
            model.decode(&features, frames)
        }
        vokra_models::vocos::VocosVariant::Encodec24khz => {
            let bandwidth_id = a.bandwidth_id.ok_or(
                "run (vocos): encodec_24khz requires --bandwidth-id <0..3> (1.5/3.0/6.0/12.0 kbps)",
            )?;
            model.decode_with_bandwidth(&features, frames, bandwidth_id)
        }
    }
    .map_err(|error| error.to_string())?;
    emit_audio("vocos", &pcm, model.sample_rate(), a.output.as_deref())
}

fn run_yue_upsampler(session: &Session, a: &RunArgs) -> Result<(), String> {
    if a.bandwidth_id.is_some() {
        return Err(
            "run (yue-upsampler): --bandwidth-id is invalid for the plain-LayerNorm YuE release"
                .to_owned(),
        );
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (yue-upsampler): --input <1024ch-features.f32> is required")?;
    let model = vokra_models::yue_upsampler::YueUpsampler::from_gguf_with_backend(
        session.gguf(),
        a.backend,
    )
    .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (yue-upsampler): --input {input_path}: {error}"))?;
    let (features, frames) =
        parse_vocoder_feature_bytes(&bytes, model.input_channels(), input_path, "yue-upsampler")?;
    let pcm = model
        .decode(&features, frames)
        .map_err(|error| error.to_string())?;
    emit_audio(
        "yue-upsampler",
        &pcm,
        model.sample_rate(),
        a.output.as_deref(),
    )
}

fn parse_vocoder_feature_bytes(
    bytes: &[u8],
    n_mels: usize,
    input_path: &str,
    task: &str,
) -> Result<(Vec<f32>, usize), String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "run ({task}): {input_path} has {} bytes, not a whole number of little-endian f32 values",
            bytes.len()
        ));
    }
    let mel: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if mel.is_empty() || mel.len() % n_mels != 0 {
        return Err(format!(
            "run ({task}): {input_path} contains {} floats; expected a positive exact multiple of channels={n_mels} in channel-major [channels, frames] order",
            mel.len(),
        ));
    }
    if let Some((index, value)) = mel
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "run ({task}): {input_path} mel[{index}] is non-finite ({value})"
        ));
    }
    let frames = mel.len() / n_mels;
    Ok((mel, frames))
}

fn validate_mimi_codes_for_model(
    codes: &MimiCodesV1,
    cfg: &vokra_models::mimi::MimiNeuralConfig,
    codec: &vokra_models::codec::MimiCodecGguf,
    model_sha256: [u8; 32],
    hop: usize,
) -> Result<(), String> {
    let expected_n_q = u32::try_from(cfg.quantizer.n_q)
        .map_err(|_| "run (mimi decode): model n_codebooks exceeds u32")?;
    let expected_bins = u32::try_from(cfg.quantizer.bins)
        .map_err(|_| "run (mimi decode): model codebook_size exceeds u32")?;
    let expected_dim = u32::try_from(codec.attrs.d_model)
        .map_err(|_| "run (mimi decode): model feature dimension exceeds u32")?;
    if codes.sample_rate != cfg.sample_rate
        || codes.frame_rate_mhz != cfg.frame_rate_mhz
        || codes.n_codebooks != expected_n_q
        || codes.codebook_size != expected_bins
        || codes.feature_dimension != expected_dim
    {
        return Err(format!(
            "run (mimi decode): container/model contract mismatch: container \
             rate={} frame_rate_mhz={} n_q={} bins={} dim={}, model \
             rate={} frame_rate_mhz={} n_q={} bins={} dim={}",
            codes.sample_rate,
            codes.frame_rate_mhz,
            codes.n_codebooks,
            codes.codebook_size,
            codes.feature_dimension,
            cfg.sample_rate,
            cfg.frame_rate_mhz,
            expected_n_q,
            expected_bins,
            expected_dim
        ));
    }
    if codes.model_sha256 != model_sha256 {
        return Err(format!(
            "run (mimi decode): codebook SHA-256 mismatch (container {}, model {}); \
             codes must be decoded by the exact effective codebook tables that encoded them",
            hex_digest(&codes.model_sha256),
            hex_digest(&model_sha256)
        ));
    }
    let frames = usize::try_from(codes.n_frames)
        .map_err(|_| "run (mimi decode): frame count does not fit this host")?;
    let expected_pcm = frames
        .checked_mul(hop)
        .ok_or("run (mimi decode): frame count overflows PCM length")?;
    let expected_pcm_u64 = u64::try_from(expected_pcm)
        .map_err(|_| "run (mimi decode): PCM sample count exceeds u64")?;
    if codes.pcm_samples != expected_pcm_u64 {
        return Err(format!(
            "run (mimi decode): container pcm_samples {} != n_frames {} * model hop {hop} = {expected_pcm}",
            codes.pcm_samples, frames
        ));
    }
    Ok(())
}

fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn run_nkf_aec(session: &Session, a: &RunArgs) -> Result<(), String> {
    let mic_path = a
        .input
        .as_deref()
        .ok_or("run (nkf_aec): --input <mic.wav> is required")?;
    let far_path = a
        .far_end
        .as_deref()
        .ok_or("run (nkf_aec): --far-end <reference.wav> is required")?;
    let mic = wav::read_wav(mic_path).map_err(|e| format!("{mic_path}: {e}"))?;
    let far = wav::read_wav(far_path).map_err(|e| format!("{far_path}: {e}"))?;
    validate_aec_pair(mic_path, &mic, far_path, &far)?;

    let model = vokra_models::aec::nkf_aec::NkfAec::from_gguf(session.gguf())
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    let cfg = model.config();
    if mic.sample_rate != cfg.sample_rate {
        return Err(format!(
            "run (nkf_aec): input pair is {} Hz, but this checkpoint is fixed at {} Hz — \
             resample both WAVs offline first (FR-EX-08: never a silent resample)",
            mic.sample_rate, cfg.sample_rate
        ));
    }

    let mut stream = model
        .open_stream(mic.sample_rate)
        .map_err(|e| e.to_string())?;
    let mut cleaned = stream
        .push_paired(&mic.samples, &far.samples)
        .map_err(|e| e.to_string())?;

    // Offline `run` promises one output sample per paired input sample. The
    // streaming engine commits only samples no future center=false STFT frame
    // can change, so zero-extend both streams just far enough to flush the OLA
    // tail, then discard the synthetic extension from the returned waveform.
    if !mic.samples.is_empty() {
        let frames = mic.samples.len().div_ceil(cfg.hop);
        let required_total = (frames - 1)
            .checked_mul(cfg.hop)
            .and_then(|n| n.checked_add(cfg.n_fft))
            .ok_or("run (nkf_aec): input length overflows the offline flush geometry")?;
        let pad = required_total.saturating_sub(mic.samples.len());
        if pad > 0 {
            let zeros = vec![0.0f32; pad];
            cleaned.extend(
                stream
                    .push_paired(&zeros, &zeros)
                    .map_err(|e| e.to_string())?,
            );
        }
        if cleaned.len() < mic.samples.len() {
            return Err(format!(
                "run (nkf_aec): offline flush committed only {} of {} samples — refusing \
                 to pad a model output silently",
                cleaned.len(),
                mic.samples.len()
            ));
        }
        cleaned.truncate(mic.samples.len());
    }

    emit_audio("aec", &cleaned, mic.sample_rate, a.output.as_deref())
}

fn validate_aec_pair(
    mic_path: &str,
    mic: &wav::Wav,
    far_path: &str,
    far: &wav::Wav,
) -> Result<(), String> {
    if mic.sample_rate != far.sample_rate {
        return Err(format!(
            "run (nkf_aec): sample-rate mismatch: --input {mic_path} is {} Hz, but \
             --far-end {far_path} is {} Hz; the streams must be sample-aligned and Vokra \
             never resamples silently",
            mic.sample_rate, far.sample_rate
        ));
    }
    if mic.samples.len() != far.samples.len() {
        return Err(format!(
            "run (nkf_aec): sample-count mismatch: --input {mic_path} has {} samples, but \
             --far-end {far_path} has {}; no trim/repeat is allowed for an AEC reference",
            mic.samples.len(),
            far.samples.len()
        ));
    }
    Ok(())
}

/// The five-release MeloTTS acoustic synthesis path.
///
/// `--input` is a `VKRMELO1` v1 bundle containing the exact already-expanded
/// phoneme/tone/language ids and BERT matrices the official acoustic graph
/// consumes. The public acoustic GGUFs do not carry language-specific raw-text
/// frontends; `--text` is therefore rejected rather than ignored or mapped by
/// a guessed tokenizer. The bundle also pins its release language, preventing
/// in-range Chinese ids from being accepted by an English embedding table.
fn run_melotts(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::rng::GaussianSplitMix64;
    use vokra_models::melotts::{MeloSynthesisOptions, MeloTextFeatures, MeloTtsCheckpoint};

    if a.text.is_some() {
        return Err(
            "run (melotts): --text is not accepted because the public acoustic GGUF does not \
             contain its language-specific G2P/tokenizer/BERT frontend; pass a versioned \
             `VKRMELO1` feature bundle through --input instead"
                .to_owned(),
        );
    }
    let input_path = a
        .input
        .as_deref()
        .ok_or("run (melotts): --input <features.vmf> is required (`VKRMELO1` v1)")?;
    let bytes = std::fs::read(input_path)
        .map_err(|error| format!("run (melotts): --input {input_path}: {error}"))?;
    let packed = MeloTextFeaturesV1::from_bytes(&bytes)
        .map_err(|error| format!("run (melotts): --input {input_path}: {error}"))?;

    let checkpoint = MeloTtsCheckpoint::from_gguf(session.gguf()).map_err(|e| e.to_string())?;
    let expected_variant = checkpoint.variant().tag();
    if packed.variant.tag() != expected_variant {
        return Err(format!(
            "run (melotts): feature bundle variant `{}` does not match model variant `{expected_variant}`; cross-language feature reuse is refused",
            packed.variant.tag()
        ));
    }
    let model = checkpoint
        .load_model(session.gguf())
        .map_err(|e| e.to_string())?;
    let features = MeloTextFeatures {
        phoneme_ids: &packed.phoneme_ids,
        tones: &packed.tones,
        language_ids: &packed.language_ids,
        bert: &packed.bert,
        ja_bert: &packed.ja_bert,
        speaker_id: packed.speaker_id,
    };
    // A fixed first-party seed makes CPU/Metal comparisons reproducible. The
    // synthesis controls remain the official defaults except for the explicit
    // caller-selected duration scale.
    let mut rng = GaussianSplitMix64::new(0x4d45_4c4f_5454_5331);
    let output = model
        .synthesize(
            features,
            MeloSynthesisOptions {
                length_scale: a.length_scale,
                ..MeloSynthesisOptions::default()
            },
            &mut rng,
            a.backend,
        )
        .map_err(|e| e.to_string())?;
    emit_audio(
        &format!("melotts-{expected_variant}"),
        &output.pcm,
        output.sample_rate,
        a.output.as_deref(),
    )
}

/// The SBV2 (Style-Bert-VITS2 v2) synthesis path (Task 38).
///
/// `SbV2Model::from_gguf` needs THREE GGUFs: this model's own weights
/// (`--model`, already opened by the generic dispatch — reused here via
/// `session.gguf()`, mirroring [`run_speaker`]'s reuse of the session's own
/// file) plus the two BERT side-cars `--bert-ja` / `--bert-en` (each
/// `vokra-convert`'s DeBERTa v2 / v3 converter output — read back at
/// runtime by `vokra-bert`'s `DebertaV2Encoder`/`DebertaV3Encoder`, opened
/// fresh here). Like
/// Kokoro / Voxtral / speaker, the dispatch hands back a bare session and
/// the concrete engine binds here rather than through the [`Session`]
/// facade, since the facade's `TtsEngine` slot has no way to carry the two
/// extra side-car paths.
///
/// `--language` selects the phonemizer + BERT path (`ja` when absent, or
/// `en`); anything else is a loud error here rather than the
/// [`vokra_core::TtsEngine`] adapter's own silent "any string not starting
/// with `en` is JA" default — see [`SbV2Model`]'s `TtsEngine` impl. This
/// pre-validation lets `run` still delegate the actual `Language` selection
/// and the correctly-`d_style`-sized default style vector to that adapter
/// (both live behind `SbV2Model`'s private fields, unreachable from this
/// crate).
///
/// # Honest scope (Task 38 does not paper over Task 24 / Task 30)
///
/// `SbV2Model::from_gguf`'s loaded phonemizer is its own documented
/// `UnwiredPhonemizer` placeholder (no G2P GGUF in that loader's 3-file
/// signature) — every synthesize call therefore fails with an explicit
/// [`vokra_core::VokraError::NotImplemented`] today. Separately,
/// `convert_sbv2_file`'s converter output is not yet loadable by
/// `from_gguf` at all (its tensor names are the verbatim upstream
/// safetensors names, not yet renamed to the `sbv2.*` hierarchy
/// `from_gguf` reads — that rename table is Task 30). Both are pre-existing,
/// documented limitations this CLI wiring surfaces honestly rather than
/// hides.
///
/// [`SbV2Model`]: vokra_models::sbv2::SbV2Model
fn run_sbv2(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::{SynthesisRequest, TtsEngine};
    use vokra_models::sbv2::SbV2Model;

    let text = a
        .text
        .as_deref()
        .ok_or("run (sbv2): --text <string> is required")?;
    let bert_ja_path = a.bert_ja.as_deref().ok_or(
        "run (sbv2): --bert-ja <bert_ja.gguf> is required — the DeBERTa v2 (JA) BERT GGUF \
         from `vokra-cli convert --model deberta-v2`",
    )?;
    let bert_en_path = a.bert_en.as_deref().ok_or(
        "run (sbv2): --bert-en <bert_en.gguf> is required — the DeBERTa v3 (EN) BERT GGUF \
         from `vokra-cli convert --model deberta-v3`",
    )?;
    // Reject anything but ja/en/zh up front (see this fn's doc — the
    // TtsEngine adapter's own default would otherwise silently swallow a
    // typo like `--language jp`). ZH is accepted at this validation step
    // because the SBV2 v2 base checkpoint has 3 language embedding rows
    // (M6 refactor 2026-08-06 — see `SbV2TextEncoder`'s
    // `language_embed` doc), but note that `synthesize` still
    // loud-refuses ZH at the BERT tokenizer step until ZH BERT + G2P are
    // wired (FR-EX-08 — see `SbV2Model::synthesize`'s ZH arm).
    if let Some(lang) = a.language.as_deref() {
        if !lang.eq_ignore_ascii_case("ja")
            && !lang.eq_ignore_ascii_case("en")
            && !lang.eq_ignore_ascii_case("zh")
        {
            return Err(format!(
                "run (sbv2): --language must be `ja`, `en`, or `zh`, got `{lang}`"
            ));
        }
    }

    let bert_ja_gguf = vokra_mmap::open_gguf(bert_ja_path)
        .map_err(|e| format!("--bert-ja {bert_ja_path}: {e}"))?;
    let bert_en_gguf = vokra_mmap::open_gguf(bert_en_path)
        .map_err(|e| format!("--bert-en {bert_en_path}: {e}"))?;

    let model = SbV2Model::from_gguf(session.gguf(), &bert_ja_gguf, &bert_en_gguf)
        .map_err(|e| e.to_string())?;

    let mut request = SynthesisRequest::new(text);
    if let Some(lang) = a.language.as_deref() {
        request = request.with_language(lang);
    }
    // Blocker 3: forward the raw little-endian f32 external speaker
    // embedding into `SynthesisRequest::speaker_embedding`. The file
    // length must be a whole multiple of 4 (one f32 per element); any
    // remainder is a caller error, not silently truncated. The
    // element-count-vs-projection-d_in check happens downstream in
    // `SbV2Model::synthesize` step 5 (loud FR-EX-08 error naming both
    // sides), where the model's own projection knows its `d_in`.
    if let Some(path) = a.speaker_embedding.as_deref() {
        let bytes = std::fs::read(path).map_err(|e| format!("--speaker-embedding {path}: {e}"))?;
        if bytes.len() % 4 != 0 {
            return Err(format!(
                "--speaker-embedding {path}: file length {} is not a multiple of 4 (one f32 \
                 per element) — corrupt file or wrong endianness?",
                bytes.len()
            ));
        }
        let embedding: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        request = request.with_speaker_embedding(embedding);
    }
    // Fully-qualified: `SbV2Model` also has an inherent `synthesize(&self,
    // req: &SbV2SynthRequest)` (a different request type) — plain
    // `model.synthesize(&request)` would resolve to that inherent method
    // first (Rust picks inherent methods by name before trait methods,
    // regardless of argument type) and fail to type-check, rather than
    // falling back to this trait impl.
    let audio = TtsEngine::synthesize(&model, &request).map_err(|e| e.to_string())?;
    emit_audio(
        "sbv2",
        &audio.samples,
        audio.sample_rate,
        a.output.as_deref(),
    )
}

/// Pushes the whole clip through a fresh VAD stream and returns the per-frame
/// speech probabilities.
fn run_vad(session: &Session, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>, String> {
    let mut handle = session.open_vad_stream().map_err(|e| e.to_string())?;
    handle.push_pcm(pcm, sample_rate).map_err(|e| e.to_string())
}

fn run_openwakeword(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (KWS): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != 16_000 {
        return Err(format!(
            "run (KWS): {path}: expected 16000 Hz mono WAV, got {} Hz — resample offline first",
            clip.sample_rate
        ));
    }
    let mut model = vokra_models::kws::openwakeword::OpenwakewordSession::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?;
    let predictions = model
        .push_pcm16k(&clip.samples)
        .map_err(|error| error.to_string())?;
    let mut detections = 0usize;
    for (name, probability) in &predictions {
        if *probability >= 0.5 {
            println!("kws: wakeword={name} probability={probability:.6}");
            detections += 1;
        }
    }
    let heads = model.wakeword_names().len();
    let chunks = predictions.len().checked_div(heads).unwrap_or_default();
    println!("kws: {chunks} chunks, detections={detections}, threshold=0.500000");
    Ok(())
}

/// Scores one complete utterance with SmartTurn v2. The single scalar output
/// is intentionally not broadcast into the VAD frame contract.
fn run_smart_turn(session: &Session, args: &RunArgs) -> Result<(), String> {
    let path = args
        .input
        .as_deref()
        .ok_or("run (smart-turn): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    let model = vokra_models::smart_turn::SmartTurn::from_gguf(session.gguf())
        .map_err(|error| error.to_string())?
        .with_backend(args.backend);
    let prediction = model
        .predict_endpoint(&clip.samples, clip.sample_rate)
        .map_err(|error| error.to_string())?;
    let probability = prediction.completion_probability();
    println!(
        "smart-turn: completion_probability={probability:.6} is_complete={} threshold={:.6}",
        prediction.is_complete(vokra_models::smart_turn::DEFAULT_COMPLETION_THRESHOLD),
        vokra_models::smart_turn::DEFAULT_COMPLETION_THRESHOLD,
    );
    Ok(())
}

/// Transcribes the clip and returns the recognized text.
fn run_asr(session: &Session, pcm: &[f32]) -> Result<String, String> {
    Ok(session
        .asr()
        .transcribe(pcm)
        .map_err(|e| e.to_string())?
        .text)
}

/// Runs the released Nemotron 3.5 offline causal path with an explicit
/// language prompt. The public July GGUF predates tokenizer embedding, so an
/// authenticated sidecar is accepted without requiring an irreversible model
/// re-upload; newly converted GGUFs carry the same bytes internally.
fn run_nemotron_asr(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_models::nemotron_asr_streaming::{NemotronAsr, SAMPLE_RATE, prompt_id_for_language};

    let path = a
        .input
        .as_deref()
        .ok_or("run (Nemotron ASR): --input <16k-mono.wav> is required")?;
    let clip = wav::read_wav(path)?;
    if clip.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "run (Nemotron ASR): {path} is {} Hz, expected {SAMPLE_RATE} Hz — resample offline first (FR-EX-08: never a silent resample)",
            clip.sample_rate
        ));
    }
    if a.beam_size != 1 || a.no_repeat_ngram != 0 || a.length_penalty.to_bits() != 0.6f32.to_bits()
    {
        return Err(
            "run (Nemotron ASR): only the released greedy RNN-T generation contract is implemented; --beam-size / --no-repeat-ngram / --length-penalty are not accepted"
                .to_owned(),
        );
    }

    let model = match a.tokenizer.as_deref() {
        Some(tokenizer_path) => {
            let bytes = std::fs::read(tokenizer_path)
                .map_err(|error| format!("--tokenizer {tokenizer_path}: {error}"))?;
            NemotronAsr::from_gguf_with_tokenizer_bytes(session.gguf(), &bytes)
        }
        None => NemotronAsr::from_gguf(session.gguf()),
    }
    .map_err(|error| error.to_string())?
    .with_backend(a.backend);
    if !model.has_tokenizer() {
        return Err(
            "run (Nemotron ASR): this legacy GGUF has no embedded official tokenizer.json; pass --tokenizer <authenticated tokenizer.json>, or reconvert with `vokra-cli convert --model nemotron-asr-streaming --tokenizer tokenizer.json`"
                .to_owned(),
        );
    }
    let language = a.language.as_deref().unwrap_or("auto");
    let prompt_id = prompt_id_for_language(language).ok_or_else(|| {
        format!(
            "run (Nemotron ASR): unsupported --language {language:?}; use a released processor language tag such as en-US, ja-JP, fr-FR, or auto"
        )
    })?;
    let transcription = model
        .transcribe_with_prompt(&clip.samples, prompt_id)
        .map_err(|error| error.to_string())?;
    println!("asr: {}", transcription.text);
    Ok(())
}

/// The Voxtral ASR path (P2 cc-10). Binds the concrete
/// [`vokra_models::voxtral::VoxtralAsr`] from the session's (mmap-backed)
/// GGUF exactly once — see [`ModelTask::AsrVoxtral`] for why the engine is
/// not injected by the dispatch — then greedy- or beam-decodes.
///
/// Prompt layout: the trained transcription wrapper by default (built at
/// runtime from the GGUF's embedded tekken vocab), with `--language` picking
/// the `lang:<code>` segment (`auto` omits it) and `--bare-prompt` opting
/// into the honest LM-continuation layout.
fn run_voxtral(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::AsrEngine;
    use vokra_models::voxtral::{AsrPromptLayout, VoxtralAsr};

    let path = a
        .input
        .as_deref()
        .ok_or("run (voxtral ASR): --input <in.wav> is required")?;
    let clip = wav::read_wav(path)?;

    // Bind through the BOUNDED-MEMORY store. `Session` already holds a parse of
    // this file, but `Session::open` reads it into an owned buffer, and
    // `VoxtralAsr::from_gguf` would then widen the whole `language_model` group
    // to f32 on top of that — 14.95 GiB at the real 3B shape, which simply does
    // not fit on a 16 GiB host. Re-opening through the mapping and binding the
    // decoder blocks lazily keeps the values bit-identical (pinned by
    // `mapped_blocks_match_resident_bitwise`) while the blocks stay in the map.
    //
    // A quantized GGUF cannot be addressed per element, so the mapped bind
    // refuses it; fall back to the resident path rather than failing the run,
    // and say so (FR-EX-08: the change of route is visible, not silent).
    let mapped = std::sync::Arc::new(vokra_mmap::open_gguf(&a.model).map_err(|e| e.to_string())?);
    let mut asr = match VoxtralAsr::from_gguf_mapped(std::sync::Arc::clone(&mapped)) {
        Ok(asr) => asr,
        Err(e) => {
            eprintln!(
                "vokra: NOTE voxtral mapped (bounded-memory) bind unavailable — {e}\n\
                 vokra: falling back to the resident loader (this widens the whole \
                 decoder to f32; expect a much larger footprint)"
            );
            VoxtralAsr::from_gguf(session.gguf()).map_err(|e| e.to_string())?
        }
    }
    .with_backend(a.backend);
    if a.bare_prompt {
        asr = asr.with_prompt_layout(AsrPromptLayout::BareSoftPrefix);
    }
    // `--language auto` = omit the prompt's `lang:` segment (upstream
    // `TranscriptionRequest.language = None`); an absent flag keeps the
    // engine default (`en`).
    match a.language.as_deref() {
        None => {}
        Some("auto") => asr = asr.with_language(None),
        Some(code) => asr = asr.with_language(Some(code.to_owned())),
    }

    if a.beam_size > 1 {
        let beams = asr
            .transcribe_beam_with_config_overrides(
                &clip.samples,
                a.beam_size,
                a.length_penalty,
                a.no_repeat_ngram,
                /*max_new_tokens*/ 0,
            )
            .map_err(|e| e.to_string())?;
        let best = beams
            .first()
            .ok_or("run (voxtral ASR): beam search produced no hypothesis")?;
        println!("asr: {}", best.text);
        for (i, b) in beams.iter().enumerate() {
            println!(
                "asr-alt[{i}]: score={:.4} logp={:.4} {}",
                b.result.length_normalized_score, b.result.log_prob, b.text
            );
        }
        return Ok(());
    }
    // `--no-repeat-ngram` only bites inside beam search; saying so beats
    // silently dropping it (FR-EX-08 spirit — the flag parses, so the user
    // gets a diagnostic rather than a wrong assumption).
    if a.no_repeat_ngram > 0 {
        eprintln!(
            "run (voxtral ASR): note — --no-repeat-ngram applies to beam search only \
             (greedy ignores it); pass --beam-size > 1 to use it."
        );
    }
    let text = asr
        .transcribe(&clip.samples)
        .map_err(|e| e.to_string())?
        .text;
    println!("asr: {text}");
    Ok(())
}

/// Whisper `--word-timestamps` (cc-19 CLI half): routes through the concrete
/// [`vokra_models::whisper::WhisperAsr`] beam/alignment surface (word
/// timestamps come from cross-attention DTW over the best hypothesis, M4-20
/// — the greedy `AsrEngine::transcribe` produces no alignment), prints the
/// transcript then one `word<TAB>start<TAB>end` line per word.
///
/// A GGUF without `vokra.whisper.alignment_heads` is an explicit error
/// raised inside `beam_search` — never an empty word list (FR-EX-08).
fn run_whisper_word_timestamps(
    model_path: &str,
    backend: vokra_core::BackendKind,
    pcm: &[f32],
    a: &RunArgs,
) -> Result<(), String> {
    use vokra_core::decode::BeamSearchConfig;
    use vokra_models::whisper::WhisperAsr;

    // Re-open the GGUF for the concrete engine: `Session` lends its file by
    // reference and `WhisperAsr::from_gguf` takes one, so this binds against
    // the same mmap-backed parse the dispatch already validated.
    let gguf = vokra_mmap::open_gguf(model_path).map_err(|e| e.to_string())?;
    let asr = WhisperAsr::from_gguf(&gguf)
        .map_err(|e| e.to_string())?
        .with_backend(backend);
    if !asr.has_tokenizer() {
        return Err(
            "run (whisper --word-timestamps): the GGUF embeds no tokenizer \
             (`vokra.tokenizer.model`), so word spans cannot be rendered to text. \
             Re-convert with the tokenizer chunk."
                .to_owned(),
        );
    }

    let mut cfg = BeamSearchConfig::greedy(vokra_models::whisper::greedy::DEFAULT_MAX_NEW_TOKENS);
    cfg.word_timestamps = true;
    // `--beam-size` (and its companions) ride the same surface: width 1 is
    // the greedy-equivalent alignment run.
    cfg.beam_width = a.beam_size.max(1);
    if a.beam_size > 1 {
        cfg.length_normalization = a.length_penalty;
        cfg.no_repeat_ngram_size = a.no_repeat_ngram;
    }
    let hyps = asr
        .transcribe_tokens_beam_nbest(pcm, &cfg)
        .map_err(|e| e.to_string())?;
    let best = hyps
        .first()
        .ok_or("run (whisper --word-timestamps): beam search produced no hypothesis")?;
    let text = asr.render_ids(&best.tokens).map_err(|e| e.to_string())?;
    println!("asr: {text}");

    // `beam_search` raises an explicit error when word timestamps were
    // requested on a model without alignment heads, so reaching here with
    // `None` would mean the driver silently skipped the alignment — surface
    // that rather than printing zero words as if the clip had none.
    let timings = best.word_timestamps.as_ref().ok_or(
        "run (whisper --word-timestamps): the decoder returned no alignment for the best \
         hypothesis (expected cross-attention word timings)",
    )?;
    for w in timings {
        let span = best.tokens.get(w.token_start..w.token_end).ok_or(
            "run (whisper --word-timestamps): word span out of range for the \
                    hypothesis tokens",
        )?;
        let word = asr.render_ids(span).map_err(|e| e.to_string())?;
        println!("word\t{word}\t{:.3}\t{:.3}", w.start, w.end);
    }
    Ok(())
}

/// The Moshi full-duplex demo path (M4-06-T26): file-driven mic frames
/// pushed through the facade duplex handle, model frames pulled per push,
/// with an optional synthetic echo path (`--echo-sim <gain>` — the
/// previous model frame mixes into the next mic frame; the session runs
/// its AEC against the pull-stamped reference queue) and the barge-in
/// demo (`--interrupt-after <frames>` flushes via the cross-thread
/// handle). The machine-checkable asserts (T26 (a)〜(d)) run inline:
/// full-length processing without underrun/panic, monotone reference
/// tags (a violating push would error loudly), flush-on-interrupt, and
/// deterministic reproduction under `--deterministic`. 知覚品質 / 実機音響
/// は T30 owner 検収 (合成 echo は近似 — spec 明記の切り離し).
fn run_s2s_duplex(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::DuplexSessionConfig;

    if a.text.is_some() {
        return Err(
            "run (Moshi duplex): --text is not accepted — Moshi GENERATES its own \
             reply (inner monologue); the transcript prints at the end"
                .to_owned(),
        );
    }
    let input_path = a.input.as_deref().ok_or(
        "run (Moshi duplex): --input <user.wav> is required (mic-side audio, mono, \
         at the model sample rate)",
    )?;
    let clip = wav::read_wav(input_path)?;

    if !a.duplex {
        // Batch turn through the facade (the run_s2s analog): push the
        // whole utterance, collect the reply + monologue transcript.
        let mut request = vokra_core::DialogRequest::new("").with_input_audio(clip.samples);
        if a.deterministic {
            request = request.deterministic();
        }
        let turn = session
            .s2s()
            .dialog_request(&request)
            .map_err(|e| e.to_string())?;
        let audio = turn
            .audio
            .ok_or("run (Moshi): the engine returned no audio")?;
        println!(
            "s2s (moshi): {} samples @ {} Hz, monologue: \"{}\" (use --duplex for \
             the continuous push/pull demo)",
            audio.samples.len(),
            audio.sample_rate,
            turn.text
        );
        if let Some(out) = a.output.as_deref() {
            wav::write_wav(out, &audio.samples, audio.sample_rate)?;
            println!("s2s (moshi): wrote -> {out}");
        }
        return Ok(());
    }

    let mut cfg = DuplexSessionConfig::new();
    if a.deterministic {
        cfg = cfg.deterministic();
    }
    if a.echo_sim.is_none() {
        // Recorded-file input with no echo path: the explicit opt-out
        // (never silent — the session records the citable warning).
        cfg = cfg.with_aec_disabled_explicitly();
    }
    let mut handle = session.s2s().duplex_with(&cfg).map_err(|e| e.to_string())?;
    let hop = handle.frame_hop();
    let sr = handle.sample_rate();
    // The synthetic echo arrives one pull→push cycle late; compensate the
    // reference clock accordingly (the T17 playback-offset knob).
    if a.echo_sim.is_some() {
        let mut cfg2 = cfg.clone().with_playback_offset_samples(hop as u64);
        if a.deterministic {
            cfg2 = cfg2.deterministic();
        }
        handle = session
            .s2s()
            .duplex_with(&cfg2)
            .map_err(|e| e.to_string())?;
    }

    let n_frames = clip.samples.len() / hop;
    if n_frames == 0 {
        return Err(format!(
            "run (Moshi duplex): input shorter than one frame ({} samples < hop {hop})",
            clip.samples.len()
        ));
    }
    let interrupt_handle = handle.interrupt_handle();
    let mut mic = vec![0.0f32; hop];
    let mut echo: Vec<f32> = vec![0.0; hop];
    let mut pcm: Vec<f32> = Vec::with_capacity(n_frames * hop);
    let mut emitted = 0usize;
    let mut interrupted_at: Option<usize> = None;
    for f in 0..n_frames {
        mic.copy_from_slice(&clip.samples[f * hop..(f + 1) * hop]);
        if let Some(gain) = a.echo_sim {
            for (m, e) in mic.iter_mut().zip(echo.iter()) {
                *m += gain * *e;
            }
        }
        // (a) the pipeline processes the whole input without underrun.
        handle.push_mic_frame(&mic).map_err(|e| e.to_string())?;
        // (b) pulls stamp monotone reference tags — a violation errors.
        while let Some(frame) = handle.pull_model_frame().map_err(|e| e.to_string())? {
            if a.echo_sim.is_some() {
                echo.copy_from_slice(&frame);
            }
            pcm.extend_from_slice(&frame);
            emitted += 1;
        }
        if let Some(after) = a.interrupt_after {
            if f + 1 == after && interrupted_at.is_none() {
                // (c) cross-thread barge-in: pending output flushes.
                interrupt_handle.interrupt();
                let flushed = handle.pull_model_frame().map_err(|e| e.to_string())?;
                if flushed.is_some() {
                    return Err("duplex barge-in: pending output survived the \
                         interrupt (flush contract violated)"
                        .to_owned());
                }
                interrupted_at = Some(f + 1);
                echo.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }
    let text = handle.monologue_text().map_err(|e| e.to_string())?;
    for w in handle.warnings() {
        eprintln!("duplex warning: {w}");
    }
    println!(
        "s2s-duplex: {n_frames} mic frames -> {emitted} model frames ({} samples) @ {sr} Hz{}{}",
        pcm.len(),
        interrupted_at
            .map(|f| format!(", barge-in after {f}"))
            .unwrap_or_default(),
        a.echo_sim
            .map(|g| format!(", echo-sim gain {g} (AEC active)"))
            .unwrap_or_default(),
    );
    println!("s2s-duplex monologue: \"{text}\"");
    if let Some(out) = a.output.as_deref() {
        wav::write_wav(out, &pcm, sr)?;
        println!("s2s-duplex: wrote {} samples @ {sr} Hz -> {out}", pcm.len());
    }
    Ok(())
}

/// The S2S (Sesame CSM) demo path — T20: recorded-file dialog turn through
/// the injected `S2sEngine` (batch) or, with `--interrupt-after`, the
/// streaming loop + M3-14-contract barge-in demo (T19).
fn run_s2s(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::DialogRequest;

    let text = a.text.as_deref().ok_or(
        "run (S2S): --text <reply text> is required — CSM speaks caller-supplied \
                text (it does not generate a reply; ADR M4-05 §D1-(b))",
    )?;
    let mut request = DialogRequest::new(text);
    if a.deterministic {
        request = request.deterministic();
    }
    if let Some(path) = a.input.as_deref() {
        let clip = wav::read_wav(path)?;
        request = request.with_input_audio(clip.samples);
    }

    if let Some(after) = a.interrupt_after {
        // Streaming + barge-in demo. The engine handle is only reachable
        // through the facade for batch dialog; the streaming surface is a
        // Rust API on the concrete engine, so this arm rebuilds it from
        // the model path (same GGUF, same synthesized bridge).
        use vokra_models::csm::{CsmEngine, CsmStreamConfig, EchoPath, FixtureByteTokenizer};
        let engine = CsmEngine::from_path(&a.model).map_err(|e| e.to_string())?;
        let engine = if a.fixture_tokenizer {
            let vocab = engine.config().text_vocab_size;
            engine
                .with_tokenizer(std::sync::Arc::new(
                    FixtureByteTokenizer::new(vocab).map_err(|e| e.to_string())?,
                ))
                .map_err(|e| e.to_string())?
        } else {
            engine
        };
        let engine = engine.with_echo_path(EchoPath::BypassRecordedInput);
        let mut stream = engine
            .open_stream(
                &request,
                Some(CsmStreamConfig {
                    max_frames: after * 4 + 8,
                }),
            )
            .map_err(|e| e.to_string())?;
        let handle = stream.interrupt_handle();
        let mut sink: Vec<vokra_core::StreamEvent> = Vec::new();
        let mut pcm = Vec::new();
        let mut frames = 0usize;
        while let Some(chunk) = stream.next_frame(&mut sink).map_err(|e| e.to_string())? {
            pcm.extend_from_slice(chunk);
            frames += 1;
            if frames == after {
                handle.interrupt();
            }
        }
        println!(
            "s2s: streamed {frames} frames ({} samples), stopped = {:?} (barge-in after {after})",
            pcm.len(),
            stream.stopped()
        );
        if let Some(out) = a.output.as_deref() {
            let sr = engine.config().sample_rate;
            wav::write_wav(out, &pcm, sr)?;
            println!("s2s: wrote {} samples @ {sr} Hz -> {out}", pcm.len());
        }
        return Ok(());
    }

    let turn = session
        .s2s()
        .dialog_request(&request)
        .map_err(|e| e.to_string())?;
    let audio = turn
        .audio
        .ok_or("run (S2S): the engine returned no audio")?;
    match a.output.as_deref() {
        Some(out) => {
            wav::write_wav(out, &audio.samples, audio.sample_rate)?;
            println!(
                "s2s: \"{}\" -> {} samples @ {} Hz -> {out}",
                turn.text,
                audio.samples.len(),
                audio.sample_rate
            );
        }
        None => {
            let secs = audio.samples.len() as f64 / f64::from(audio.sample_rate);
            println!(
                "s2s: \"{}\" -> {} samples, {secs:.3}s @ {} Hz (no --output; audio discarded)",
                turn.text,
                audio.samples.len(),
                audio.sample_rate
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    fn silero_fixture() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
            .to_string_lossy()
            .into_owned()
    }

    /// `--mimi <path>` parses into `RunArgs::mimi` (Moshi real-codec
    /// side-car); a bare `--mimi` is a loud parse error.
    #[test]
    fn parse_accepts_mimi_sidecar_path() {
        let a = parse_args(&args(&[
            "--model",
            "m.gguf",
            "--input",
            "mic.wav",
            "--duplex",
            "--mimi",
            "codec.gguf",
        ]))
        .expect("parses");
        assert_eq!(a.mimi.as_deref(), Some("codec.gguf"));
        assert!(a.duplex);
        let err = match parse_args(&args(&["--model", "m.gguf", "--mimi"])) {
            Err(e) => e,
            Ok(_) => panic!("bare --mimi must be rejected"),
        };
        assert!(err.contains("--mimi requires a GGUF path"), "got: {err}");
    }

    #[test]
    fn parse_accepts_nemotron_tokenizer_and_language() {
        let parsed = parse_args(&args(&[
            "--model",
            "nemotron.gguf",
            "--input",
            "speech.wav",
            "--tokenizer",
            "tokenizer.json",
            "--language",
            "ja-JP",
        ]))
        .expect("Nemotron sidecar flags parse");
        assert_eq!(parsed.tokenizer.as_deref(), Some("tokenizer.json"));
        assert_eq!(parsed.language.as_deref(), Some("ja-JP"));
        let error = match parse_args(&args(&["--model", "nemotron.gguf", "--tokenizer"])) {
            Err(error) => error,
            Ok(_) => panic!("bare --tokenizer must be rejected"),
        };
        assert_eq!(error, "--tokenizer requires a tokenizer.json path");
    }

    #[test]
    fn parse_accepts_explicit_musicgen_token_and_generation_contract() {
        let parsed = parse_args(&args(&[
            "--model",
            "musicgen-small.gguf",
            "--token-ids",
            "71,1234,1",
            "--music-unconditional-token-ids",
            "1",
            "--music-frames",
            "250",
            "--music-seed",
            "42",
            "--output",
            "music.wav",
        ]))
        .expect("MusicGen explicit input flags parse");
        assert_eq!(parsed.token_ids.as_deref(), Some("71,1234,1"));
        assert_eq!(parsed.music_unconditional_token_ids.as_deref(), Some("1"));
        assert_eq!(parsed.music_frames, Some(250));
        assert_eq!(parsed.music_seed, Some(42));
        assert!(
            parse_args(&args(&[
                "--model",
                "musicgen-small.gguf",
                "--music-frames",
                "0",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parse_accepts_explicit_moss_tts_companion_contract() {
        let parsed = parse_args(&args(&[
            "--model",
            "moss-tts-nano.gguf",
            "--audio-tokenizer",
            "moss-audio-tokenizer-nano.gguf",
            "--max-new-frames",
            "300",
            "--input",
            "prompt.u32le",
            "--output",
            "speech.wav",
        ]))
        .expect("MOSS-TTS explicit companion flags parse");
        assert_eq!(
            parsed.audio_tokenizer.as_deref(),
            Some("moss-audio-tokenizer-nano.gguf")
        );
        assert_eq!(parsed.max_new_frames, Some(300));
        assert!(
            parse_args(&args(&[
                "--model",
                "moss-tts-nano.gguf",
                "--max-new-frames",
                "0",
            ]))
            .is_err()
        );
    }

    #[test]
    fn parse_accepts_pyannote_dependency_models() {
        let parsed = parse_args(&args(&[
            "--model",
            "pipeline.gguf",
            "--segmentation-model",
            "segmentation.gguf",
            "--embedding-model",
            "wespeaker.gguf",
            "--input",
            "meeting.wav",
            "--output",
            "meeting.rttm",
        ]))
        .expect("pyannote dependency flags parse");
        assert_eq!(
            parsed.segmentation_model.as_deref(),
            Some("segmentation.gguf")
        );
        assert_eq!(parsed.embedding_model.as_deref(), Some("wespeaker.gguf"));

        let error = match parse_args(&args(&["--model", "pipeline.gguf", "--embedding-model"])) {
            Err(error) => error,
            Ok(_) => panic!("missing dependency path must be explicit"),
        };
        assert!(error.contains("--embedding-model requires a GGUF path"));
    }

    #[test]
    fn parses_wave2_ct_punc_and_mimi_contract_flags() {
        let ct = parse_args(&args(&[
            "--model",
            "ct.gguf",
            "--tokens",
            "tokens.tsv",
            "--output",
            "restored.txt",
        ]))
        .expect("CT-Punc flags parse");
        assert_eq!(ct.tokens.as_deref(), Some("tokens.tsv"));
        assert_eq!(ct.codec_mode, None);

        for (value, expected) in [("encode", CodecMode::Encode), ("decode", CodecMode::Decode)] {
            let mimi = parse_args(&args(&[
                "--model",
                "mimi.gguf",
                "--codec-mode",
                value,
                "--input",
                "input.bin",
                "--output",
                "output.bin",
            ]))
            .expect("Mimi mode parses");
            assert_eq!(mimi.codec_mode, Some(expected));
        }
        assert!(
            parse_args(&args(&[
                "--model",
                "mimi.gguf",
                "--codec-mode",
                "roundtrip"
            ]))
            .err()
            .unwrap()
            .contains("expected encode or decode")
        );
        assert!(
            parse_args(&args(&["--model", "ct.gguf", "--tokens"]))
                .err()
                .unwrap()
                .contains("--tokens requires a path")
        );
    }

    #[test]
    fn parses_moss_quantizer_width_and_rejects_zero() {
        let parsed = parse_args(&args(&[
            "--model",
            "moss.gguf",
            "--codec-mode",
            "decode",
            "--num-quantizers",
            "12",
            "--input",
            "codes.u32le",
            "--output",
            "audio.wav",
        ]))
        .expect("MOSS raw-code width parses");
        assert_eq!(parsed.num_quantizers, Some(12));

        let error = parse_args(&args(&["--model", "moss.gguf", "--num-quantizers", "0"]))
            .err()
            .expect("zero quantizers must fail");
        assert!(error.contains("must be positive"));
    }

    #[test]
    fn mimi_container_is_bound_to_exact_model_contract() {
        use vokra_models::codec::MimiCodecGguf;
        use vokra_models::mimi::MimiNeuralConfig;
        use vokra_ops::MimiRvqAttrs;

        let cfg = MimiNeuralConfig::tiny_for_tests();
        let hop = cfg.frame_hop_samples().unwrap();
        let codec = MimiCodecGguf {
            attrs: MimiRvqAttrs {
                n_codebooks: cfg.quantizer.n_q,
                codebook_size: cfg.quantizer.bins,
                d_model: cfg.transformer.d_model,
            },
            // Contract validation reads topology only; RVQ math is exercised
            // by vokra-ops and the real-weight VAST lane.
            tables: Vec::new(),
        };
        let digest = sha256(b"exact effective tables");
        let mut codes = MimiCodesV1 {
            sample_rate: cfg.sample_rate,
            frame_rate_mhz: cfg.frame_rate_mhz,
            n_codebooks: u32::try_from(cfg.quantizer.n_q).unwrap(),
            codebook_size: u32::try_from(cfg.quantizer.bins).unwrap(),
            feature_dimension: u32::try_from(codec.attrs.d_model).unwrap(),
            n_frames: 2,
            pcm_samples: u64::try_from(2 * hop).unwrap(),
            model_sha256: digest,
            codes: vec![0; 2 * cfg.quantizer.n_q],
        };

        validate_mimi_codes_for_model(&codes, &cfg, &codec, digest, hop).unwrap();

        codes.model_sha256[0] ^= 1;
        assert!(
            validate_mimi_codes_for_model(&codes, &cfg, &codec, digest, hop)
                .unwrap_err()
                .contains("SHA-256 mismatch")
        );
        codes.model_sha256 = digest;
        codes.n_codebooks += 1;
        assert!(
            validate_mimi_codes_for_model(&codes, &cfg, &codec, digest, hop)
                .unwrap_err()
                .contains("contract mismatch")
        );
        codes.n_codebooks -= 1;
        codes.pcm_samples += 1;
        assert!(
            validate_mimi_codes_for_model(&codes, &cfg, &codec, digest, hop)
                .unwrap_err()
                .contains("pcm_samples")
        );
    }

    /// Task 38: `--bert-ja` / `--bert-en` parse into `RunArgs`, are absent by
    /// default, and each requires a value.
    #[test]
    fn parses_bert_ja_and_bert_en_flags() {
        let a = parse_args(&args(&[
            "--model",
            "sbv2.gguf",
            "--text",
            "こんにちは",
            "--bert-ja",
            "bert_ja.gguf",
            "--bert-en",
            "bert_en.gguf",
        ]))
        .expect("parses");
        assert_eq!(a.bert_ja.as_deref(), Some("bert_ja.gguf"));
        assert_eq!(a.bert_en.as_deref(), Some("bert_en.gguf"));

        let a = parse_args(&args(&["--model", "m.gguf"])).expect("parses");
        assert!(a.bert_ja.is_none());
        assert!(a.bert_en.is_none());

        let err = match parse_args(&args(&["--model", "m.gguf", "--bert-ja"])) {
            Err(e) => e,
            Ok(_) => panic!("bare --bert-ja must be rejected"),
        };
        assert!(err.contains("--bert-ja requires a path"), "got: {err}");

        let err = match parse_args(&args(&["--model", "m.gguf", "--bert-en"])) {
            Err(e) => e,
            Ok(_) => panic!("bare --bert-en must be rejected"),
        };
        assert!(err.contains("--bert-en requires a path"), "got: {err}");
    }

    /// Writes a synthesized-fixture CSM GGUF (tiny shape config + mimi
    /// chunk + provenance + a placeholder tokenizer blob) into a temp file
    /// and returns its path — the M4-05-T20 host-only smoke input.
    fn csm_fixture_gguf(tag: &str) -> std::path::PathBuf {
        use vokra_core::gguf::{GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};
        use vokra_models::csm::CsmConfig;
        use vokra_models::mimi::MimiNeuralConfig;
        let cfg = CsmConfig::tiny_for_tests();
        let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
        mimi_cfg.quantizer.n_q = cfg.n_codebooks;
        mimi_cfg.quantizer.bins = cfg.audio_vocab_size;
        let mut fixed = cfg.clone();
        fixed.sample_rate = mimi_cfg.sample_rate;
        fixed.frame_rate_mhz = mimi_cfg.frame_rate_mhz;
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        vokra_core::stamp_provenance(
            &mut b,
            vokra_core::LicenseClass::Permissive,
            "Apache-2.0",
            Some("sesame/csm-1b"),
            None,
        );
        fixed.write_gguf_metadata(&mut b);
        mimi_cfg.write_gguf_metadata(&mut b);
        b.add_metadata(
            "vokra.tokenizer.model",
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: vec![GgufMetadataValue::U8(1)],
            }),
        );
        let path = std::env::temp_dir().join(format!(
            "vokra-cli-csm-smoke-{tag}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, b.to_bytes().expect("serialize")).expect("write fixture");
        path
    }

    #[test]
    fn parses_a_full_run_invocation() {
        let a = parse_args(&args(&[
            "--model", "m.gguf", "--input", "in.wav", "--output", "o.wav",
        ]))
        .expect("valid");
        assert_eq!(a.model, "m.gguf");
        assert_eq!(a.input.as_deref(), Some("in.wav"));
        assert_eq!(a.output.as_deref(), Some("o.wav"));
        assert_eq!(a.text, None);
        // Defaults for beam-search flags.
        assert_eq!(a.beam_size, 1);
        assert!((a.length_penalty - 0.6).abs() < 1e-6);
        assert_eq!(a.no_repeat_ngram, 0);
    }

    #[test]
    fn parses_beam_search_flags() {
        let a = parse_args(&args(&[
            "--model",
            "m.gguf",
            "--input",
            "in.wav",
            "--beam-size",
            "5",
            "--length-penalty",
            "1.2",
            "--no-repeat-ngram",
            "3",
        ]))
        .expect("valid");
        assert_eq!(a.beam_size, 5);
        assert!((a.length_penalty - 1.2).abs() < 1e-6);
        assert_eq!(a.no_repeat_ngram, 3);
    }

    #[test]
    fn rejects_bad_beam_size_and_length_penalty() {
        // --beam-size = 0 is rejected (matches BeamConfig invariant).
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--beam-size", "0"]))
                .err()
                .unwrap()
                .contains("beam-size must be >= 1")
        );
        // --beam-size non-integer.
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--beam-size", "nope"]))
                .err()
                .unwrap()
                .contains("--beam-size")
        );
        // --length-penalty negative.
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--length-penalty", "-1"]))
                .err()
                .unwrap()
                .contains("--length-penalty")
        );
        // --length-penalty NaN.
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--length-penalty", "nan"]))
                .err()
                .unwrap()
                .contains("--length-penalty")
        );
        // dangling values.
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--beam-size"]))
                .err()
                .unwrap(),
            "--beam-size requires a value"
        );
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--length-penalty"]))
                .err()
                .unwrap(),
            "--length-penalty requires a value"
        );
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--no-repeat-ngram"]))
                .err()
                .unwrap(),
            "--no-repeat-ngram requires a value"
        );
    }

    // ---- --backend (bench-surface mirror) + --compare (speaker) ----------

    /// `--backend` parses exactly like `bench --backend` (shared
    /// `parse_backend`): default cpu, metal/cuda/vulkan/coreml/qnn accepted at
    /// parse time (availability is an inference-time explicit error, FR-EX-08).
    #[test]
    fn parses_backend_flag_with_cpu_default() {
        use vokra_core::BackendKind;
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.backend, BackendKind::Cpu);
        for (name, kind) in [
            ("cpu", BackendKind::Cpu),
            ("metal", BackendKind::Metal),
            ("cuda", BackendKind::Cuda),
            ("vulkan", BackendKind::Vulkan),
            // QNN delegate (Qualcomm Hexagon NPU, M5-02): parses to the variant;
            // an actual run is an explicit UnsupportedOp / BackendUnavailable
            // unless built with the `qnn` feature on Android/Linux/Windows.
            ("qnn", BackendKind::Qnn),
        ] {
            let a = parse_args(&args(&["--model", "m.gguf", "--backend", name]))
                .unwrap_or_else(|e| panic!("--backend {name} should parse: {e}"));
            assert_eq!(a.backend, kind, "--backend {name}");
        }
    }

    #[test]
    fn rejects_unknown_backend_and_dangling_backend() {
        let err = parse_args(&args(&["--model", "m.gguf", "--backend", "npu"]))
            .err()
            .unwrap();
        assert!(err.contains("unknown --backend"), "got: {err}");
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--backend"]))
                .err()
                .unwrap(),
            "--backend requires a value"
        );
    }

    #[test]
    fn parses_compare_flag_and_rejects_dangling_compare() {
        let a = parse_args(&args(&[
            "--model",
            "spk.gguf",
            "--input",
            "a.wav",
            "--compare",
            "b.wav",
        ]))
        .expect("valid");
        assert_eq!(a.compare.as_deref(), Some("b.wav"));
        // Default: no compare.
        let a = parse_args(&args(&["--model", "spk.gguf"])).expect("valid");
        assert_eq!(a.compare, None);
        assert_eq!(
            parse_args(&args(&["--model", "spk.gguf", "--compare"]))
                .err()
                .unwrap(),
            "--compare requires a value"
        );
    }

    #[test]
    fn parses_far_end_flag_and_rejects_dangling_far_end() {
        let a = parse_args(&args(&[
            "--model",
            "nkf.gguf",
            "--input",
            "mic.wav",
            "--far-end",
            "ref.wav",
        ]))
        .expect("valid");
        assert_eq!(a.far_end.as_deref(), Some("ref.wav"));
        assert_eq!(
            parse_args(&args(&["--model", "nkf.gguf", "--far-end"]))
                .err()
                .unwrap(),
            "--far-end requires a value"
        );
    }

    #[test]
    fn aec_pair_contract_rejects_rate_and_length_mismatch() {
        let mic = wav::Wav {
            sample_rate: 16_000,
            samples: vec![0.0; 320],
        };
        let same = mic.clone();
        validate_aec_pair("mic.wav", &mic, "ref.wav", &same).expect("aligned pair");

        let wrong_rate = wav::Wav {
            sample_rate: 48_000,
            samples: vec![0.0; 320],
        };
        let err = validate_aec_pair("mic.wav", &mic, "ref.wav", &wrong_rate).unwrap_err();
        assert!(err.contains("sample-rate mismatch"), "{err}");
        assert!(err.contains("16000") && err.contains("48000"), "{err}");

        let wrong_len = wav::Wav {
            sample_rate: 16_000,
            samples: vec![0.0; 319],
        };
        let err = validate_aec_pair("mic.wav", &mic, "ref.wav", &wrong_len).unwrap_err();
        assert!(err.contains("sample-count mismatch"), "{err}");
        assert!(err.contains("320") && err.contains("319"), "{err}");
    }

    // ---- P2 cc-10 / cc-19: voxtral route + whisper word timestamps -------

    #[test]
    fn parses_word_timestamps_language_and_bare_prompt_flags() {
        let a = parse_args(&args(&["--model", "m.gguf", "--input", "in.wav"])).expect("valid");
        assert!(!a.word_timestamps);
        assert_eq!(a.language, None);
        assert!(!a.bare_prompt);

        let a = parse_args(&args(&[
            "--model",
            "m.gguf",
            "--input",
            "in.wav",
            "--word-timestamps",
        ]))
        .expect("valid");
        assert!(a.word_timestamps);

        let a = parse_args(&args(&[
            "--model",
            "v.gguf",
            "--input",
            "in.wav",
            "--language",
            "fr",
            "--bare-prompt",
        ]))
        .expect("valid");
        assert_eq!(a.language.as_deref(), Some("fr"));
        assert!(a.bare_prompt);

        // `auto` is carried verbatim; the run arm maps it to "omit the
        // lang: segment".
        let a = parse_args(&args(&["--model", "v.gguf", "--language", "auto"])).expect("valid");
        assert_eq!(a.language.as_deref(), Some("auto"));
    }

    #[test]
    fn rejects_dangling_and_empty_language() {
        assert_eq!(
            parse_args(&args(&["--model", "v.gguf", "--language"]))
                .err()
                .unwrap(),
            "--language requires a value"
        );
        assert!(
            parse_args(&args(&["--model", "v.gguf", "--language", ""]))
                .err()
                .unwrap()
                .contains("must not be empty")
        );
    }

    /// `--word-timestamps` off the whisper arch is an explicit contract
    /// error (FR-EX-08 — Voxtral has no cross-attention alignment).
    #[test]
    fn word_timestamps_on_non_whisper_arch_is_rejected() {
        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--word-timestamps",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--word-timestamps is only supported for the whisper"),
            "got: {err}"
        );
    }

    /// `--language` / `--bare-prompt` off the voxtral arch likewise.
    #[test]
    fn voxtral_prompt_flags_on_other_arch_are_rejected() {
        // `--language` is now shared by voxtral AND sbv2 (Task 38), so its
        // rejection message differs from `--bare-prompt`'s (voxtral-only) —
        // check each flag against its own message rather than one shared
        // substring.
        let mut argv = args(&["--model", &silero_fixture(), "--input", "unused.wav"]);
        argv.extend(vec!["--language".to_owned(), "fr".to_owned()]);
        let err = main(&argv).unwrap_err();
        assert!(
            err.contains("--language is only supported for the voxtral arch")
                && err.contains("sbv2 arch"),
            "got: {err}"
        );

        let mut argv = args(&["--model", &silero_fixture(), "--input", "unused.wav"]);
        argv.extend(vec!["--bare-prompt".to_owned()]);
        let err = main(&argv).unwrap_err();
        assert!(
            err.contains("--bare-prompt is only supported for the voxtral arch"),
            "got: {err}"
        );
    }

    /// Task 38: `--bert-ja` / `--bert-en` off the sbv2 arch are rejected
    /// loudly rather than silently ignored (FR-EX-08) — mirrors
    /// `voxtral_prompt_flags_on_other_arch_are_rejected`.
    #[test]
    fn sbv2_side_cars_are_rejected_on_other_models() {
        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--bert-ja",
            "bert_ja.gguf",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--bert-ja / --bert-en are only supported for the sbv2 arch"),
            "got: {err}"
        );

        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--bert-en",
            "bert_en.gguf",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--bert-ja / --bert-en are only supported for the sbv2 arch"),
            "got: {err}"
        );
    }

    /// Builds a metadata-only `sbv2`-arch GGUF (no real tensors) at a fresh
    /// temp path — enough for the dispatch to select [`ModelTask::Sbv2`]
    /// (bare session) and reach `run_sbv2`'s own argument checks, mirroring
    /// `voxtral_metadata_only_gguf_fails_loudly_at_engine_bind`'s fixture
    /// style.
    fn sbv2_metadata_only_gguf(tag: &str) -> std::path::PathBuf {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "sbv2");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model = std::env::temp_dir().join(format!(
            "vokra-cli-sbv2-meta-{tag}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&model, &bytes).unwrap();
        model
    }

    fn melotts_metadata_only_gguf(tag: &str) -> std::path::PathBuf {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "melotts");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model = std::env::temp_dir().join(format!(
            "vokra-cli-melotts-meta-{tag}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&model, &bytes).unwrap();
        model
    }

    #[test]
    fn melotts_requires_versioned_features_and_refuses_raw_text() {
        let model = melotts_metadata_only_gguf("input-contract");
        let err = main(&args(&["--model", model.to_str().unwrap()])).unwrap_err();
        assert!(
            err.contains("--input <features.vmf> is required") && err.contains("VKRMELO1"),
            "got: {err}"
        );

        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "hello",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--text is not accepted") && err.contains("G2P/tokenizer/BERT"),
            "got: {err}"
        );

        let _ = std::fs::remove_file(&model);
    }

    #[test]
    fn melotts_backend_and_length_scale_reach_the_feature_contract() {
        let model = melotts_metadata_only_gguf("backend-contract");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "metal",
            "--length-scale",
            "0.5",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(err.contains("--input <features.vmf> is required"), "{err}");
        assert!(!err.contains("runs on the CPU regardless"), "{err}");
        assert!(!err.contains("only supported for the kokoro"), "{err}");
    }

    #[test]
    fn melotts_invalid_feature_container_fails_before_tensor_binding() {
        let model = melotts_metadata_only_gguf("bad-container");
        let features =
            std::env::temp_dir().join(format!("vokra-cli-melotts-bad-{}.vmf", std::process::id()));
        std::fs::write(&features, b"not a MeloTTS feature bundle").unwrap();
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--input",
            features.to_str().unwrap(),
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&features);
        assert!(err.contains("feature container is truncated"), "{err}");
        assert!(!err.contains("missing/non-string"), "{err}");
    }

    /// `run_sbv2`'s three required-argument checks fire in order: `--text`,
    /// then `--bert-ja`, then `--bert-en` — each names the missing flag
    /// rather than a generic failure (FR-EX-08).
    #[test]
    fn sbv2_requires_text_then_bert_ja_then_bert_en() {
        let model = sbv2_metadata_only_gguf("order");

        let err = main(&args(&["--model", model.to_str().unwrap()])).unwrap_err();
        assert!(err.contains("--text <string> is required"), "got: {err}");

        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "こんにちは",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--bert-ja <bert_ja.gguf> is required"),
            "got: {err}"
        );

        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "こんにちは",
            "--bert-ja",
            "/no/such/bert_ja.gguf",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--bert-en <bert_en.gguf> is required"),
            "got: {err}"
        );

        let _ = std::fs::remove_file(&model);
    }

    /// With `--text` / `--bert-ja` / `--bert-en` all present, an invalid
    /// `--language` value is rejected before any file I/O on the (here,
    /// nonexistent) side-car paths — the CLI's own pre-validation, distinct
    /// from the `TtsEngine` adapter's silent any-non-`en`-is-JA default (see
    /// `run_sbv2`'s doc).
    #[test]
    fn sbv2_language_rejects_anything_but_ja_en_or_zh() {
        let model = sbv2_metadata_only_gguf("lang");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "こんにちは",
            "--bert-ja",
            "/no/such/bert_ja.gguf",
            "--bert-en",
            "/no/such/bert_en.gguf",
            "--language",
            "fr",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(
            err.contains("--language must be `ja`, `en`, or `zh`, got `fr`"),
            "got: {err}"
        );
    }

    /// `ja` / `en` (any case) both pass the `--language` pre-validation and
    /// reach the (here, failing) `--bert-ja` GGUF open — proving the check
    /// does not reject the two values it is meant to accept.
    #[test]
    fn sbv2_language_accepts_ja_and_en_case_insensitively() {
        let model = sbv2_metadata_only_gguf("lang-ok");
        for lang in ["ja", "JA", "en", "EN"] {
            let err = main(&args(&[
                "--model",
                model.to_str().unwrap(),
                "--text",
                "hello",
                "--bert-ja",
                "/no/such/bert_ja.gguf",
                "--bert-en",
                "/no/such/bert_en.gguf",
                "--language",
                lang,
            ]))
            .unwrap_err();
            assert!(
                !err.contains("--language must be"),
                "{lang}: rejected as an invalid language: {err}"
            );
            assert!(
                err.contains("--bert-ja"),
                "{lang}: expected the fixture --bert-ja path to fail to open: {err}"
            );
        }
        let _ = std::fs::remove_file(&model);
    }

    /// A metadata-only `voxtral` GGUF reaches the run arm (dispatch is
    /// bare by design) and then fails loudly when the concrete engine binds
    /// — never a silent success (FR-EX-08).
    #[test]
    fn voxtral_metadata_only_gguf_fails_loudly_at_engine_bind() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let dir = std::env::temp_dir();
        let model = dir.join(format!("vokra-cli-vox-meta-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let in_wav = dir.join(format!("vokra-cli-vox-meta-{}.wav", std::process::id()));
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 16_000).unwrap();
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--input",
            in_wav.to_str().unwrap(),
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&in_wav);
        assert!(!err.is_empty(), "loud bind error expected");
    }

    /// Voxtral without `--input` is a contract error naming the flag.
    #[test]
    fn voxtral_without_input_is_a_contract_error() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model =
            std::env::temp_dir().join(format!("vokra-cli-vox-noinput-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let err = main(&args(&["--model", model.to_str().unwrap()])).unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(err.contains("--input"), "actionable: {err}");
    }

    /// Real-GGUF gated e2e for the voxtral CLI route (P2 cc-10): set
    /// `VOKRA_VOXTRAL_GGUF` (+ optional `VOKRA_VOXTRAL_WAV`) to run; skips
    /// clean when unset. Prints the transcript to stdout via the run arm —
    /// the numeric/text assertion rides the models-crate e2e test
    /// (`voxtral_transcription_prompt.rs`), this one proves the CLI wiring.
    #[test]
    fn voxtral_real_gguf_cli_route_gated() {
        let Ok(model) = std::env::var("VOKRA_VOXTRAL_GGUF") else {
            eprintln!("skipping voxtral CLI e2e: set VOKRA_VOXTRAL_GGUF to run");
            return;
        };
        let wav = std::env::var("VOKRA_VOXTRAL_WAV").unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/audio/jfk-30s.wav")
                .to_string_lossy()
                .into_owned()
        });
        let code = main(&args(&["--model", &model, "--input", &wav])).expect("voxtral CLI runs");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    /// Real-GGUF gated check for `--word-timestamps` (cc-19): set
    /// `VOKRA_WHISPER_GGUF` (+ optional `VOKRA_WHISPER_WAV`).
    #[test]
    fn whisper_word_timestamps_cli_route_gated() {
        let Ok(model) = std::env::var("VOKRA_WHISPER_GGUF") else {
            eprintln!("skipping whisper word-timestamps CLI e2e: set VOKRA_WHISPER_GGUF to run");
            return;
        };
        let wav = std::env::var("VOKRA_WHISPER_WAV").unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/audio/jfk-30s.wav")
                .to_string_lossy()
                .into_owned()
        });
        let code = main(&args(&[
            "--model",
            &model,
            "--input",
            &wav,
            "--word-timestamps",
        ]))
        .expect("whisper --word-timestamps runs");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    /// The help text documents the new surface (Fix A + Fix C of the
    /// campaign-2 cli-enablers leg).
    #[test]
    fn help_text_documents_backend_compare_and_speaker() {
        assert!(USAGE.contains("--backend"), "USAGE lists --backend");
        assert!(
            USAGE.contains("cpu | metal | cuda"),
            "USAGE lists the backend names"
        );
        assert!(USAGE.contains("--compare"), "USAGE lists --compare");
        assert!(USAGE.contains("--far-end"), "USAGE lists --far-end");
        assert!(USAGE.contains("speaker"), "USAGE mentions the speaker task");
        assert!(USAGE.contains("CAM++"), "USAGE names the CAM++ family");
        assert!(
            USAGE.contains("X-vector"),
            "USAGE names the X-vector family"
        );
        // P2 cc-10 / cc-19 surface.
        assert!(
            USAGE.contains("--word-timestamps"),
            "USAGE lists --word-timestamps"
        );
        assert!(USAGE.contains("--language"), "USAGE lists --language");
        assert!(USAGE.contains("--bare-prompt"), "USAGE lists --bare-prompt");
        assert!(USAGE.contains("voxtral"), "USAGE names the voxtral arch");
        assert!(USAGE.contains("wetextprocessing"));
        assert!(USAGE.contains("nkf_aec"));
        assert!(USAGE.contains("fcpe.gguf"));
        assert!(USAGE.contains("crepe.gguf"));
        assert!(USAGE.contains("ct-punc.gguf"));
        assert!(USAGE.contains("vokra-ct-punc-tsv-v1"));
        assert!(USAGE.contains("--codec-mode encode"));
        assert!(USAGE.contains("VKRMCODE"));
        assert!(USAGE.contains("snac.gguf"));
        assert!(USAGE.contains("VKRSNAC1"));
        assert!(USAGE.contains("miocodec.gguf"));
        assert!(USAGE.contains("VKRMIO01"));
    }

    #[test]
    fn snac_model_fingerprint_pins_tensor_name_shape_dtype_and_payload() {
        use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

        fn fixture(name: &str, dimensions: Vec<u64>, values: &[f32]) -> GgufFile {
            let mut builder = GgufBuilder::new();
            let payload = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            builder
                .add_tensor(name, GgmlType::F32, dimensions, payload)
                .expect("add fingerprint tensor");
            GgufFile::parse(builder.to_bytes().expect("serialize fingerprint fixture"))
                .expect("parse fingerprint fixture")
        }

        let original = fixture("weight", vec![2], &[1.0, 2.0]);
        let same = fixture("weight", vec![2], &[1.0, 2.0]);
        let renamed = fixture("other", vec![2], &[1.0, 2.0]);
        let reshaped = fixture("weight", vec![1, 2], &[1.0, 2.0]);
        let changed = fixture("weight", vec![2], &[1.0, 3.0]);
        let digest = snac_model_fingerprint(&original).unwrap();
        assert_eq!(digest, snac_model_fingerprint(&same).unwrap());
        assert_ne!(digest, snac_model_fingerprint(&renamed).unwrap());
        assert_ne!(digest, snac_model_fingerprint(&reshaped).unwrap());
        assert_ne!(digest, snac_model_fingerprint(&changed).unwrap());
    }

    /// `--compare` on a non-speaker arch is an explicit contract error
    /// (FR-EX-08: never silently ignore a user flag).
    #[test]
    fn compare_on_non_speaker_arch_is_rejected() {
        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--compare",
            "b.wav",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--compare is only supported for speaker-embedding arches"),
            "got: {err}"
        );
    }

    #[test]
    fn far_end_on_non_aec_arch_is_rejected() {
        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--far-end",
            "reference.wav",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--far-end is only supported for the nkf_aec arch"),
            "got: {err}"
        );
    }

    #[test]
    fn wave2_contract_flags_are_rejected_off_their_arches() {
        let model = silero_fixture();
        let err = main(&args(&["--model", &model, "--tokens", "tokens.tsv"])).unwrap_err();
        assert!(err.contains("--tokens is only supported for the ct_punc arch"));

        let err = main(&args(&["--model", &model, "--codec-mode", "encode"])).unwrap_err();
        assert!(err.contains(
            "--codec-mode is only supported for standalone \
             mimi/dac/wavtokenizer/neucodec/xcodec2/miocodec/snac/focalcodec arches"
        ));
    }

    /// A campplus-arch GGUF whose tensors do not bind fails loudly at the
    /// encoder bind inside the Speaker arm (the engine dispatch itself
    /// returns a bare session — see `engine::tests`).
    #[test]
    fn speaker_metadata_only_gguf_fails_loudly_at_encoder_bind() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "campplus");
        let bytes = b.to_bytes().expect("serialize gguf");
        let dir = std::env::temp_dir();
        let model = dir.join(format!("vokra-cli-spk-meta-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let in_wav = dir.join(format!("vokra-cli-spk-meta-{}.wav", std::process::id()));
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 16_000).unwrap();
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--input",
            in_wav.to_str().unwrap(),
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&in_wav);
        // The bind error names the missing tensor / weight, not a panic.
        assert!(!err.is_empty(), "loud bind error expected");
    }

    /// Speaker task without `--input` is a contract error.
    #[test]
    fn speaker_without_input_is_a_contract_error() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "campplus");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model =
            std::env::temp_dir().join(format!("vokra-cli-spk-noinput-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let err = main(&args(&["--model", model.to_str().unwrap()])).unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(err.contains("--input"), "actionable: {err}");
    }

    /// Real-GGUF gated e2e (mirrors the model parity gates).  Each available
    /// public speaker artifact must run through the same CLI surface; an unset
    /// model variable skips only that leg.
    #[test]
    fn speaker_real_gguf_e2e_identical_inputs_gated() {
        let dir = std::env::temp_dir();
        let in_wav = dir.join(format!("vokra-cli-spk-e2e-{}.wav", std::process::id()));
        // 1 s of deterministic pseudo-speech at 16 kHz (multi-tone, enough
        // frames for the CAM++ receptive field).
        let samples: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.3 * (t * std::f32::consts::TAU * 220.0).sin()
                    + 0.2 * (t * std::f32::consts::TAU * 660.0).sin()
            })
            .collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 16_000).unwrap();
        let wav8k = dir.join(format!("vokra-cli-spk-e2e8k-{}.wav", std::process::id()));
        wav::write_wav(wav8k.to_str().unwrap(), &samples[..8000], 8_000).unwrap();
        for (variable, label, dimension) in [
            ("VOKRA_CAMPLUS_GGUF", "CAM++", 192),
            ("VOKRA_XVECTOR_GGUF", "X-vector", 512),
            ("VOKRA_ECAPA_GGUF", "ECAPA-TDNN", 192),
            ("VOKRA_WESPEAKER_GGUF", "WeSpeaker", 256),
            ("VOKRA_TITANET_GGUF", "TitaNet-L", 192),
        ] {
            let Ok(model) = std::env::var(variable) else {
                eprintln!("skipping {label} CLI e2e: set {variable} to run");
                continue;
            };
            // Identical inputs → success and cosine 1.0 on stdout.
            let output = dir.join(format!(
                "vokra-cli-spk-e2e-{label}-{}.f32",
                std::process::id()
            ));
            let code = main(&args(&[
                "--model",
                &model,
                "--input",
                in_wav.to_str().unwrap(),
                "--compare",
                in_wav.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
            ]))
            .unwrap_or_else(|error| panic!("{label} speaker e2e failed: {error}"));
            assert_eq!(code, ExitCode::SUCCESS);
            assert_eq!(std::fs::metadata(&output).unwrap().len(), dimension * 4);
            let _ = std::fs::remove_file(&output);

            // A non-16 kHz clip is an explicit error (no silent resample).
            let err = main(&args(&[
                "--model",
                &model,
                "--input",
                wav8k.to_str().unwrap(),
            ]))
            .unwrap_err();
            assert!(
                err.contains("16000 Hz") || err.contains("16 kHz"),
                "{label}: got {err}"
            );
        }
        let _ = std::fs::remove_file(&in_wav);
        let _ = std::fs::remove_file(&wav8k);
    }

    #[test]
    fn rejects_missing_model_and_dangling_flag_and_stray_arg() {
        assert_eq!(
            parse_args(&args(&["--input", "x.wav"])).err().unwrap(),
            "--model is required"
        );
        assert_eq!(
            parse_args(&args(&["--model"])).err().unwrap(),
            "--model requires a value"
        );
        assert!(
            parse_args(&args(&["--bogus"]))
                .err()
                .unwrap()
                .contains("unexpected argument")
        );
    }

    /// M4-06-T26: the full duplex demo path over a converted synthetic
    /// checkpoint — echo-sim (AEC active), barge-in flush, deterministic
    /// reproduction, and the attribution banner side of the load (the
    /// engine dispatch prints it; here we assert the session carries the
    /// resolved info).
    #[test]
    fn moshi_duplex_demo_smoke_with_echo_sim_and_barge_in() {
        let dir = std::env::temp_dir().join(format!("vokra-cli-moshi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        let tok = dir.join("tok.model");
        let gguf = dir.join("moshi.gguf");
        let in_wav = dir.join("user.wav");
        let out_wav = dir.join("model.wav");
        std::fs::write(&ckpt, moshi_fixture::synthetic_checkpoint()).unwrap();
        std::fs::write(&tok, moshi_fixture::spm_blob(13)).unwrap();
        vokra_convert::convert_moshi_file(&ckpt, Some(tok.as_path()), &gguf).expect("convert");

        // 4 frames of pseudo-speech at the converted (real-constant) rates.
        let (session, task) =
            engine::load_session(gguf.to_str().unwrap()).expect("moshi session loads");
        assert_eq!(task, ModelTask::S2sDuplex);
        assert!(
            session.attribution().is_some(),
            "FR-MD-09: the loader resolves the attribution surface"
        );
        let hop = 1920usize; // 24 kHz / 12.5 Hz (converter constants)
        let samples: Vec<f32> = (0..hop * 4)
            .map(|i| ((i as f32) * 0.01).sin() * 0.2)
            .collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 24_000).unwrap();

        let run_once = || {
            let args: Vec<String> = [
                "--model",
                gguf.to_str().unwrap(),
                "--input",
                in_wav.to_str().unwrap(),
                "--output",
                out_wav.to_str().unwrap(),
                "--duplex",
                "--echo-sim",
                "0.5",
                "--interrupt-after",
                "2",
                "--deterministic",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            main(&args).expect("duplex demo runs");
            wav::read_wav(out_wav.to_str().unwrap()).expect("output wav")
        };
        let a = run_once();
        let b = run_once();
        assert!(!a.samples.is_empty(), "model frames were pulled");
        assert_eq!(a.samples, b.samples, "T26 (d): deterministic reproduction");

        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            use vokra_core::{BackendKind, VokraError};

            match vokra_models::moshi::gpu_backend_probe(BackendKind::Metal) {
                Ok(()) => {
                    let metal_wav = dir.join("model-metal.wav");
                    let metal_args: Vec<String> = [
                        "--model",
                        gguf.to_str().unwrap(),
                        "--input",
                        in_wav.to_str().unwrap(),
                        "--output",
                        metal_wav.to_str().unwrap(),
                        "--backend",
                        "metal",
                        "--duplex",
                        "--echo-sim",
                        "0.5",
                        "--interrupt-after",
                        "2",
                        "--deterministic",
                    ]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                    main(&metal_args).expect("Moshi Metal duplex demo runs");
                    let metal =
                        wav::read_wav(metal_wav.to_str().unwrap()).expect("Metal output wav");
                    assert_eq!(metal.samples.len(), a.samples.len());
                    let max_abs = metal
                        .samples
                        .iter()
                        .zip(&a.samples)
                        .map(|(&gpu, &cpu)| (gpu - cpu).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!("Moshi CLI CPU/Metal PCM max_abs={max_abs:.9e}");
                    assert!(
                        max_abs <= 0.01,
                        "Moshi CLI CPU/Metal PCM max_abs={max_abs} exceeds 0.01"
                    );
                }
                Err(VokraError::BackendUnavailable(error)) => {
                    eprintln!("skip Moshi CLI Metal parity: {error}");
                }
                Err(error) => panic!("unexpected Moshi Metal probe error: {error}"),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Moshi duplex argument contract: --text is rejected (the model
    /// generates its reply), --input is required.
    #[test]
    fn moshi_duplex_rejects_text_and_requires_input() {
        let dir = std::env::temp_dir().join(format!("vokra-cli-moshi-neg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        let tok = dir.join("tok.model");
        let gguf = dir.join("moshi.gguf");
        std::fs::write(&ckpt, moshi_fixture::synthetic_checkpoint()).unwrap();
        std::fs::write(&tok, moshi_fixture::spm_blob(13)).unwrap();
        vokra_convert::convert_moshi_file(&ckpt, Some(tok.as_path()), &gguf).expect("convert");
        // A tokenizer-less conversion fails loudly at LOAD (monologue
        // decode is load-bearing) — pin that posture too.
        let bare = dir.join("bare.gguf");
        vokra_convert::convert_moshi_file(&ckpt, None, &bare).expect("convert bare");
        let err = engine::load_session(bare.to_str().unwrap()).unwrap_err();
        assert!(err.contains("vokra.tokenizer.model"), "loud: {err}");
        let base = ["--model".to_string(), gguf.to_str().unwrap().to_string()];
        let mut with_text: Vec<String> = base.to_vec();
        with_text.extend(["--text".into(), "scripted".into()]);
        let err = main(&with_text).unwrap_err();
        assert!(err.contains("GENERATES"), "contract: {err}");
        let err = main(&base).unwrap_err();
        assert!(
            err.contains("--input") || err.contains("--duplex") || err.contains("required"),
            "actionable: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Shared synthetic Moshi checkpoint fixtures (the converter/e2e
    /// wire shapes — MoshiConfig::tiny_for_tests).
    mod moshi_fixture {
        pub(super) fn spm_blob(n: usize) -> Vec<u8> {
            fn varint(mut v: u64, out: &mut Vec<u8>) {
                loop {
                    let mut b = (v & 0x7f) as u8;
                    v >>= 7;
                    if v != 0 {
                        b |= 0x80;
                    }
                    out.push(b);
                    if v == 0 {
                        break;
                    }
                }
            }
            let mut blob = Vec::new();
            for i in 0..n {
                let piece = format!("\u{2581}p{i}");
                let mut msg = Vec::new();
                msg.push(0x0a);
                varint(piece.len() as u64, &mut msg);
                msg.extend_from_slice(piece.as_bytes());
                msg.push(0x18);
                msg.push(0x01);
                blob.push(0x0a);
                varint(msg.len() as u64, &mut blob);
                blob.extend_from_slice(&msg);
            }
            blob
        }

        pub(super) fn synthetic_checkpoint() -> Vec<u8> {
            let mut entries: Vec<(String, Vec<u64>)> = Vec::new();
            let (d, text, card) = (16u64, 13u64, 9u64);
            let (h_tm, d_dt, h_dt) = (8u64, 8u64, 6u64);
            entries.push(("text_emb.weight".into(), vec![text + 1, d]));
            entries.push(("text_linear.weight".into(), vec![text, d]));
            entries.push(("out_norm.alpha".into(), vec![1, 1, d]));
            for k in 0..4 {
                entries.push((format!("emb.{k}.weight"), vec![card + 1, d]));
            }
            for i in 0..2 {
                let p = format!("transformer.layers.{i}");
                entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d]));
                entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d]));
                entries.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * d, d]));
                entries.push((format!("{p}.self_attn.out_proj.weight"), vec![d, d]));
                entries.push((format!("{p}.gating.linear_in.weight"), vec![2 * h_tm, d]));
                entries.push((format!("{p}.gating.linear_out.weight"), vec![d, h_tm]));
            }
            for cb in 0..2 {
                entries.push((format!("depformer_in.{cb}.weight"), vec![d_dt, d]));
                entries.push((format!("linears.{cb}.weight"), vec![card, d_dt]));
            }
            entries.push(("depformer_text_emb.weight".into(), vec![text + 1, d_dt]));
            entries.push(("depformer_emb.0.weight".into(), vec![card + 1, d_dt]));
            for i in 0..2 {
                let p = format!("depformer.layers.{i}");
                entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d_dt]));
                entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d_dt]));
                entries.push((
                    format!("{p}.self_attn.in_proj_weight"),
                    vec![2 * 3 * d_dt, d_dt],
                ));
                entries.push((
                    format!("{p}.self_attn.out_proj.weight"),
                    vec![2 * d_dt, d_dt],
                ));
                for s in 0..2 {
                    entries.push((
                        format!("{p}.gating.{s}.linear_in.weight"),
                        vec![2 * h_dt, d_dt],
                    ));
                    entries.push((
                        format!("{p}.gating.{s}.linear_out.weight"),
                        vec![d_dt, h_dt],
                    ));
                }
            }
            let mut header = String::from("{");
            let mut data: Vec<u8> = Vec::new();
            let mut lcg = 0x9876_5432u32;
            for (i, (name, shape)) in entries.iter().enumerate() {
                let n: u64 = shape.iter().product();
                let start = data.len();
                for _ in 0..n {
                    lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
                    let frac = (lcg >> 16) as u16 & 0x007F;
                    let sign = ((lcg >> 8) as u16) & 0x8000;
                    data.extend_from_slice(&(sign | 0x3E00 | frac).to_le_bytes());
                }
                let end = data.len();
                if i > 0 {
                    header.push(',');
                }
                header.push_str(&format!(
                    "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
                    shape
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            header.push('}');
            let mut blob = Vec::new();
            blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
            blob.extend_from_slice(header.as_bytes());
            blob.extend_from_slice(&data);
            blob
        }
    }

    #[test]
    fn run_vad_over_committed_fixture_yields_frames() {
        let (session, task) = engine::load_session(&silero_fixture()).expect("silero loads");
        assert_eq!(task, ModelTask::Vad);
        // 1 s of silence at 16 kHz completes several fixed-size frames.
        let pcm = vec![0.0f32; 16_000];
        let probs = run_vad(&session, &pcm, 16_000).expect("vad runs");
        assert!(!probs.is_empty(), "1 s of audio should complete >= 1 frame");
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn s2s_host_only_smoke_batch_dialog_writes_a_wav() {
        // T20: explicit CPU backend, synthesized-fixture GGUF, explicit
        // fixture tokenizer (opt-in flag) → e2e run + WAV out.
        let model = csm_fixture_gguf("batch");
        let out = std::env::temp_dir().join(format!(
            "vokra-cli-csm-smoke-out-{}.wav",
            std::process::id()
        ));
        let code = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "host only smoke",
            "--fixture-tokenizer",
            "--deterministic",
            "--output",
            out.to_str().unwrap(),
        ]))
        .expect("s2s smoke runs");
        assert_eq!(code, ExitCode::SUCCESS);
        let clip = wav::read_wav(out.to_str().unwrap()).expect("output WAV parses");
        assert!(!clip.samples.is_empty());
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn s2s_streaming_barge_in_demo_stops_after_n_frames() {
        let model = csm_fixture_gguf("interrupt");
        let code = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "interrupt me",
            "--fixture-tokenizer",
            "--deterministic",
            "--interrupt-after",
            "2",
        ]))
        .expect("s2s barge-in demo runs");
        assert_eq!(code, ExitCode::SUCCESS);
        let _ = std::fs::remove_file(&model);
    }

    #[test]
    fn s2s_without_text_is_a_contract_error() {
        let model = csm_fixture_gguf("no-text");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--fixture-tokenizer",
        ]))
        .unwrap_err();
        assert!(err.contains("--text"), "actionable: {err}");
        let _ = std::fs::remove_file(&model);
    }

    #[test]
    fn s2s_gguf_tokenizer_without_fixture_flag_fails_loudly() {
        // Without --fixture-tokenizer the embedded (T29-gated) tokenizer is
        // honest: encode = NotImplemented — never a silent byte fallback.
        let model = csm_fixture_gguf("honest");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "should fail loudly",
        ]))
        .unwrap_err();
        assert!(
            err.contains("not implemented") || err.contains("T29"),
            "{err}"
        );
        let _ = std::fs::remove_file(&model);
    }

    // ---- cc-24: kokoro phoneme-id route -----------------------------------

    /// The misaki table shipped in a real Kokoro GGUF: index = id, entry 0 is
    /// the (unaddressable) pad sentinel. Trimmed to the symbols these tests
    /// need; the real voice carries 178.
    fn kokoro_symbols() -> Vec<String> {
        ["", "ð", "ə", " ", "k", "w", "ɪ"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    #[test]
    fn kokoro_tokenizer_wraps_ids_in_upstream_sentinels() {
        // Upstream builds `input_ids = [0, *ids, 0]` (kokoro==0.9.4).
        let ids = kokoro_phoneme_ids("ðə", &kokoro_symbols()).expect("all symbols known");
        assert_eq!(ids, vec![0, 1, 2, 0]);
    }

    #[test]
    fn kokoro_tokenizer_maps_every_char_by_table_position() {
        let ids = kokoro_phoneme_ids("ðə kwɪ", &kokoro_symbols()).expect("all symbols known");
        // ð=1 ə=2 space=3 k=4 w=5 ɪ=6, wrapped in the sentinels.
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 0]);
    }

    #[test]
    fn kokoro_tokenizer_rejects_unknown_characters_instead_of_dropping_them() {
        // FR-EX-08: silently dropping an unmappable phoneme would change the
        // utterance with no signal. The message must name the offenders and
        // point at the missing G2P bridge.
        let err = kokoro_phoneme_ids("ðəZ🎵", &kokoro_symbols()).unwrap_err();
        assert!(err.contains('Z') && err.contains('🎵'), "names them: {err}");
        assert!(
            err.contains("PHONEMES"),
            "explains the input contract: {err}"
        );
    }

    #[test]
    fn kokoro_tokenizer_rejects_empty_phoneme_text() {
        let err = kokoro_phoneme_ids("", &kokoro_symbols()).unwrap_err();
        assert!(err.contains("no phonemes"), "{err}");
    }

    #[test]
    fn kokoro_tokenizer_accepts_the_piper_raw_id_form() {
        // Content ids only; the sentinels are added here, as in piper's
        // `parse_content` / `phonemize` split. Whitespace and comma separated
        // forms are equivalent.
        let want = vec![0, 1, 2, 3, 0];
        assert_eq!(
            kokoro_phoneme_ids("1 2 3", &kokoro_symbols()).unwrap(),
            want
        );
        assert_eq!(
            kokoro_phoneme_ids("1,2,3", &kokoro_symbols()).unwrap(),
            want
        );
        assert_eq!(
            kokoro_phoneme_ids(" 1,  2 ,3 ", &kokoro_symbols()).unwrap(),
            want
        );
    }

    #[test]
    fn kokoro_raw_id_form_agrees_with_the_symbol_form() {
        // The two spellings of the same utterance must tokenize identically —
        // otherwise one of them is silently synthesizing something else.
        let syms = kokoro_symbols();
        assert_eq!(
            kokoro_phoneme_ids("ðə kwɪ", &syms).unwrap(),
            kokoro_phoneme_ids("1 2 3 4 5 6", &syms).unwrap()
        );
    }

    #[test]
    fn kokoro_raw_id_form_rejects_out_of_range_and_the_pad_sentinel() {
        let syms = kokoro_symbols(); // 7 entries → content ids are 1..7
        for bad in ["0", "7", "99"] {
            let err = kokoro_phoneme_ids(bad, &syms).unwrap_err();
            assert!(err.contains("out of range"), "{bad}: {err}");
        }
    }

    #[test]
    fn kokoro_raw_id_form_is_refused_when_a_digit_is_itself_a_symbol() {
        // The digit heuristic is only sound while no symbol is a bare digit.
        // A table that breaks that must produce a loud ambiguity error rather
        // than silently picking one reading (FR-EX-08).
        let mut syms = kokoro_symbols();
        syms.push("2".to_owned());
        let err = kokoro_phoneme_ids("1 2 3", &syms).unwrap_err();
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn kokoro_style_vector_accepts_both_upstream_widths() {
        let dir = std::env::temp_dir();
        for n in [4usize, 8] {
            let p = dir.join(format!("vokra-cli-style-{n}-{}.f32", std::process::id()));
            let bytes: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
            std::fs::write(&p, &bytes).unwrap();
            // style_dim = 4 → accepts 4 (single) and 8 (full ref_s row).
            let v = read_style_vector(p.to_str().unwrap(), 4).expect("width accepted");
            assert_eq!(v.len(), n);
            assert_eq!(v[0], 0.0);
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn kokoro_style_vector_rejects_wrong_width_and_ragged_files() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("vokra-cli-style-bad-{}.f32", std::process::id()));

        // Wrong float count (5 is neither style_dim nor 2*style_dim).
        std::fs::write(
            &p,
            (0..5)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let err = read_style_vector(p.to_str().unwrap(), 4).unwrap_err();
        assert!(err.contains("5 floats"), "{err}");

        // Not a whole number of f32s.
        std::fs::write(&p, [0u8; 6]).unwrap();
        let err = read_style_vector(p.to_str().unwrap(), 4).unwrap_err();
        assert!(err.contains("whole number of f32s"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn kokoro_style_flags_are_rejected_off_the_kokoro_arch() {
        // FR-EX-08: a style knob silently ignored on another arch would change
        // nothing about the output while implying it had.
        let model = silero_fixture();
        // `--input` is any readable path here: the arch guard fires before the
        // task itself runs, so the fixture doubles as the input.
        for flag in [["--voice", "af_heart"], ["--style", "/nonexistent.f32"]] {
            let err = main(&args(&[
                "--model", &model, "--input", &model, flag[0], flag[1],
            ]))
            .unwrap_err();
            assert!(
                err.contains("only supported for the kokoro arch"),
                "{flag:?}: {err}"
            );
        }
    }

    #[test]
    fn kokoro_length_scale_is_rejected_off_the_kokoro_arch() {
        let model = silero_fixture();
        let err = main(&args(&[
            "--model",
            &model,
            "--input",
            &model,
            "--length-scale",
            "1.5",
        ]))
        .unwrap_err();
        assert!(
            err.contains("only supported for the kokoro or melotts arch"),
            "{err}"
        );
    }

    #[test]
    fn kokoro_length_scale_rejects_non_positive_values() {
        for bad in ["0", "-1.0", "nan"] {
            let err = parse_args(&args(&["--model", "m.gguf", "--length-scale", bad]))
                .map(|_| ())
                .unwrap_err();
            assert!(err.contains("positive finite"), "{bad}: {err}");
        }
    }

    #[test]
    fn audioseal_cli_parses_exact_message_and_variant() {
        let parsed = parse_args(&args(&[
            "--model",
            "audioseal.gguf",
            "--watermark-mode",
            "embed",
            "--watermark-variant",
            "streaming",
            "--watermark-message",
            "1010101010101010",
            "--watermark-alpha",
            "0.75",
        ]))
        .expect("valid AudioSeal flags");
        assert_eq!(parsed.watermark_mode, Some(WatermarkMode::Embed));
        assert_eq!(
            parsed.watermark_variant,
            Some(vokra_models::audioseal::AudiosealVariant::Streaming)
        );
        assert_eq!(
            parsed.watermark_message,
            Some([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        );
        assert_eq!(parsed.watermark_alpha, Some(0.75));
    }

    #[test]
    fn audioseal_cli_rejects_ambiguous_messages_and_modes() {
        for bad in [
            "",
            "1010",
            "101010101010101x",
            "１０１０１０１０１０１０１０１０",
        ] {
            let error = parse_watermark_message(bad).unwrap_err();
            assert!(error.contains("exactly 16 ASCII 0/1"), "{bad:?}: {error}");
        }
        let error = WatermarkMode::parse("watermark").unwrap_err();
        assert!(error.contains("expected detect or embed"), "{error}");
    }

    // ---- cc-36: FR-EX-08 guard for a non-CPU --backend on a CPU-only arch --

    /// Every `ModelTask` is classified: the CPU-only engines (whose `run`
    /// dispatch never threads `--backend`) return a label so the guard fires,
    /// and the backend-honoring paths return `None` so it does NOT — the
    /// no-regression contract for whisper / voxtral / kokoro / speaker.
    #[test]
    fn cpu_only_engine_label_classifies_every_task() {
        // Silent-ignore engines → guard fires (named label).
        assert_eq!(
            cpu_only_engine_label(ModelTask::S2s),
            Some("CSM speech-to-speech")
        );
        assert_eq!(
            cpu_only_engine_label(ModelTask::Sbv2),
            Some("SBV2 (Style-Bert-VITS2 v2) TTS")
        );
        assert_eq!(
            cpu_only_engine_label(ModelTask::CtPunc),
            Some("CT-Punc punctuation restoration")
        );
        assert_eq!(cpu_only_engine_label(ModelTask::TextEncoder), None);
        // Backend-honoring arches bind `.with_backend(...)` → guard must NOT
        // fire (no regression). This is the piece that covers whisper.
        assert_eq!(cpu_only_engine_label(ModelTask::Asr), None);
        assert_eq!(cpu_only_engine_label(ModelTask::AsrVoxtral), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Vad), None);
        assert_eq!(cpu_only_engine_label(ModelTask::VadFirered), None);
        assert_eq!(cpu_only_engine_label(ModelTask::VadTen), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Tts), None);
        assert_eq!(cpu_only_engine_label(ModelTask::TtsKokoro), None);
        assert_eq!(cpu_only_engine_label(ModelTask::TtsMelo), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Speaker), None);
        assert_eq!(cpu_only_engine_label(ModelTask::LangId), None);
        assert_eq!(cpu_only_engine_label(ModelTask::AudioQualityAudiobox), None);
        assert_eq!(cpu_only_engine_label(ModelTask::WatermarkAudioseal), None);
        assert_eq!(cpu_only_engine_label(ModelTask::MimiCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::DacCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::WavTokenizerCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::NeuCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::XCodec2), None);
        assert_eq!(cpu_only_engine_label(ModelTask::FunCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::SpeechTokenizer), None);
        assert_eq!(cpu_only_engine_label(ModelTask::MioCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::SnacCodec), None);
        assert_eq!(cpu_only_engine_label(ModelTask::FocalCodec), None);
        assert_eq!(
            cpu_only_engine_label(ModelTask::MossAudioTokenizerCodec),
            None
        );
        assert_eq!(cpu_only_engine_label(ModelTask::S2sDuplex), None);
        assert_eq!(cpu_only_engine_label(ModelTask::VadFsmn), None);
        assert_eq!(cpu_only_engine_label(ModelTask::VocoderVocos), None);
        assert_eq!(cpu_only_engine_label(ModelTask::VocoderBigVgan), None);
        assert_eq!(cpu_only_engine_label(ModelTask::VocoderHifiGan), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Denoise), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Separation), None);
        assert_eq!(cpu_only_engine_label(ModelTask::AecNkf), None);
        assert_eq!(cpu_only_engine_label(ModelTask::F0Rmvpe), None);
        assert_eq!(cpu_only_engine_label(ModelTask::F0Fcpe), None);
        assert_eq!(cpu_only_engine_label(ModelTask::SmartTurn), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Segment), None);
        assert_eq!(cpu_only_engine_label(ModelTask::DiarizationPyannote), None);
        // Bench-only tasks — unreachable from `run`; defer to their own error.
        assert_eq!(cpu_only_engine_label(ModelTask::MelFrontend), None);
        assert_eq!(cpu_only_engine_label(ModelTask::Cosyvoice2Synthetic), None);
    }

    #[test]
    fn separation_multi_stream_paths_are_deterministic() {
        assert_eq!(
            separation_stream_path("/tmp/mix.wav", 1),
            std::path::PathBuf::from("/tmp/mix.source1.wav")
        );
        assert_eq!(
            separation_stream_path("relative/output", 3),
            std::path::PathBuf::from("relative/output.source3.wav")
        );
    }

    /// Real-public-artifact CPU/Metal/official-FP64 parity for every released
    /// SepFormer checkpoint. This Apple-only family sweep proves that every
    /// released weight set and every 1/2/3-stream head reaches the selected
    /// Metal backend through the complete separation forward.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn sepformer_all_public_cpu_metal_waveform_parity() {
        let Some(directory) = std::env::var_os("VOKRA_SEPFORMER_GGUF_DIR") else {
            eprintln!(
                "skip: set VOKRA_SEPFORMER_GGUF_DIR to the seven fixed-revision public GGUFs"
            );
            return;
        };
        let pcm = include_bytes!("../../vokra-models/tests/fixtures/sepformer/pcm.f32.bin")
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32 fixture chunk")))
            .collect::<Vec<_>>();
        assert_eq!(pcm.len(), 4_096);
        let metrics = |cpu: &[f32], metal: &[f32]| {
            assert_eq!(metal.len(), cpu.len());
            let mut max_abs = 0.0f32;
            let mut sum_abs = 0.0f64;
            let mut cpu_peak = 0.0f32;
            for (&cpu_value, &metal_value) in cpu.iter().zip(metal) {
                let delta = (cpu_value - metal_value).abs();
                max_abs = max_abs.max(delta);
                sum_abs += f64::from(delta);
                cpu_peak = cpu_peak.max(cpu_value.abs());
            }
            (max_abs, sum_abs / cpu.len() as f64, cpu_peak)
        };

        let read_f32 = |bytes: &[u8]| {
            assert_eq!(bytes.len() % 4, 0, "reference contains whole f32 values");
            bytes
                .chunks_exact(4)
                .map(|chunk| {
                    f32::from_le_bytes(chunk.try_into().expect("four-byte f32 reference chunk"))
                })
                .collect::<Vec<_>>()
        };
        #[allow(clippy::type_complexity)]
        let rows: [(&str, usize, u32, &[u8], &[u8], f32, f64); 7] = [
            (
                "sepformer-wsj02mix.gguf",
                2,
                8_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/wsj02mix/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/wsj02mix/separated.f32.bin"
                ),
                0.01,
                0.001,
            ),
            (
                "sepformer-libri2mix.gguf",
                2,
                8_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/libri2mix/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/libri2mix/separated.f32.bin"
                ),
                0.01,
                0.001,
            ),
            (
                "sepformer-libri3mix.gguf",
                3,
                8_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/libri3mix/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/libri3mix/separated.f32.bin"
                ),
                0.01,
                0.001,
            ),
            (
                "sepformer-wham16k-enhancement.gguf",
                1,
                16_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/wham16k-enhancement/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/wham16k-enhancement/separated.f32.bin"
                ),
                0.01,
                0.001,
            ),
            (
                "sepformer-whamr16k.gguf",
                2,
                16_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/whamr16k/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/whamr16k/separated.f32.bin"
                ),
                0.01,
                0.001,
            ),
            (
                "sepformer-whamr-8khz.gguf",
                2,
                8_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/whamr-8khz/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/whamr-8khz/separated.f32.bin"
                ),
                0.01,
                0.001,
            ),
            (
                "sepformer-dns4.gguf",
                1,
                16_000,
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/dns4-16k-enhancement/encoder.f32.bin"
                ),
                include_bytes!(
                    "../../vokra-models/tests/fixtures/sepformer/official-fp64/dns4-16k-enhancement/separated.f32.bin"
                ),
                0.1513,
                0.00515,
            ),
        ];
        let only = std::env::var("VOKRA_SEPFORMER_ONLY").ok();
        let mut failures = Vec::new();
        let mut ran = 0usize;
        for (
            file_name,
            expected_streams,
            expected_rate,
            encoder_reference,
            separated_reference,
            waveform_max_bound,
            waveform_mean_bound,
        ) in rows
        {
            if only.as_deref().is_some_and(|only| only != file_name) {
                continue;
            }
            ran += 1;
            let path = std::path::Path::new(&directory).join(file_name);
            let expected_encoder = read_f32(encoder_reference);
            let expected_separated = read_f32(separated_reference);
            assert_eq!(expected_encoder.len(), 256 * 511, "{file_name}");
            assert_eq!(
                expected_separated.len(),
                pcm.len() * expected_streams,
                "{file_name} official separated shape"
            );
            let file = vokra_core::gguf::GgufFile::open(&path)
                .unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
            let (cpu_encoder, cpu_frames, cpu_output) = {
                let cpu = vokra_models::sepformer::SepFormer::from_gguf(&file)
                    .unwrap_or_else(|error| panic!("strict CPU bind {file_name}: {error}"));
                assert_eq!(
                    usize::try_from(cpu.n_out()).expect("SepFormer n_out fits usize"),
                    expected_streams,
                    "{file_name}"
                );
                assert_eq!(cpu.sample_rate(), expected_rate, "{file_name}");
                let (encoder, frames) = cpu
                    .encode_features(&pcm)
                    .unwrap_or_else(|error| panic!("CPU encoder {file_name}: {error}"));
                let output = cpu
                    .separate(&pcm)
                    .unwrap_or_else(|error| panic!("CPU separation {file_name}: {error}"));
                (encoder, frames, output)
            };
            let (metal_encoder, metal_frames, metal_output) = {
                let metal = vokra_models::sepformer::SepFormer::from_gguf(&file)
                    .unwrap_or_else(|error| panic!("strict Metal bind {file_name}: {error}"))
                    .with_backend(vokra_core::BackendKind::Metal);
                let (encoder, frames) = metal
                    .encode_features(&pcm)
                    .unwrap_or_else(|error| panic!("Metal encoder {file_name}: {error}"));
                let output = metal
                    .separate(&pcm)
                    .unwrap_or_else(|error| panic!("Metal separation {file_name}: {error}"));
                (encoder, frames, output)
            };
            assert_eq!(metal_frames, cpu_frames, "{file_name} encoder frames");
            let (encoder_max, encoder_mean, encoder_cpu_peak) =
                metrics(&cpu_encoder, &metal_encoder);
            let (cpu_encoder_max, cpu_encoder_mean, _) = metrics(&cpu_encoder, &expected_encoder);
            let (metal_encoder_max, metal_encoder_mean, _) =
                metrics(&metal_encoder, &expected_encoder);
            eprintln!(
                "SepFormer public CPU/Metal {file_name} encoder: frames={cpu_frames} \
                 values={} cpu_peak={encoder_cpu_peak:.9e} max_abs={encoder_max:.9e} \
                 mean_abs={encoder_mean:.9e} cpu_official_max={cpu_encoder_max:.9e} \
                 cpu_official_mean={cpu_encoder_mean:.9e} \
                 metal_official_max={metal_encoder_max:.9e} \
                 metal_official_mean={metal_encoder_mean:.9e}",
                cpu_encoder.len()
            );
            if encoder_max > 0.01
                || encoder_mean > 0.001
                || cpu_encoder_max > 0.01
                || cpu_encoder_mean > 0.001
                || metal_encoder_max > 0.01
                || metal_encoder_mean > 0.001
            {
                failures.push(format!(
                    "{file_name} encoder CPU/Metal max={encoder_max:.9e} mean={encoder_mean:.9e}; \
                     CPU/official max={cpu_encoder_max:.9e} mean={cpu_encoder_mean:.9e}; \
                     Metal/official max={metal_encoder_max:.9e} mean={metal_encoder_mean:.9e}"
                ));
            }
            assert_eq!(cpu_output.len(), expected_streams, "CPU {file_name}");
            assert_eq!(metal_output.len(), expected_streams, "Metal {file_name}");

            for (stream, (cpu, metal)) in cpu_output.iter().zip(&metal_output).enumerate() {
                assert_eq!(metal.len(), cpu.len(), "{file_name} stream {stream}");
                assert!(
                    cpu.iter().chain(metal).all(|value| value.is_finite()),
                    "{file_name} stream {stream} produced non-finite PCM"
                );
                let (max_abs, mean_abs, cpu_peak) = metrics(cpu, metal);
                let expected = expected_separated
                    .chunks_exact(expected_streams)
                    .map(|sample| sample[stream])
                    .collect::<Vec<_>>();
                let (cpu_official_max, cpu_official_mean, _) = metrics(cpu, &expected);
                let (metal_official_max, metal_official_mean, _) = metrics(metal, &expected);
                eprintln!(
                    "SepFormer public CPU/Metal {file_name} stream={stream}: \
                     samples={} cpu_peak={cpu_peak:.9e} max_abs={max_abs:.9e} \
                     mean_abs={mean_abs:.9e} cpu_official_max={cpu_official_max:.9e} \
                     cpu_official_mean={cpu_official_mean:.9e} \
                     metal_official_max={metal_official_max:.9e} \
                     metal_official_mean={metal_official_mean:.9e}",
                    cpu.len()
                );
                // Six variants retain the existing 0.01 / 0.001 boundary.
                // DNS4 uses the separately documented official-FP32 floor in
                // fixtures/sepformer/README.md; the encoder gate above stays
                // at the strict family boundary for every variant.
                if max_abs > waveform_max_bound
                    || mean_abs > waveform_mean_bound
                    || cpu_official_max > waveform_max_bound
                    || cpu_official_mean > waveform_mean_bound
                    || metal_official_max > waveform_max_bound
                    || metal_official_mean > waveform_mean_bound
                {
                    failures.push(format!(
                        "{file_name} stream {stream} CPU/Metal max={max_abs:.9e} \
                         mean={mean_abs:.9e}; CPU/official max={cpu_official_max:.9e} \
                         mean={cpu_official_mean:.9e}; Metal/official \
                         max={metal_official_max:.9e} mean={metal_official_mean:.9e}; \
                         bounds max={waveform_max_bound:.9e} \
                         mean={waveform_mean_bound:.9e}"
                    ));
                }
            }
        }
        assert!(ran > 0, "VOKRA_SEPFORMER_ONLY did not name a public GGUF");
        assert!(
            failures.is_empty(),
            "SepFormer CPU/Metal/official drift exceeds its documented variant boundary:\n{}",
            failures.join("\n")
        );
    }

    /// Real-weight full-posterior CPU/Metal parity for the FSMN learned-op
    /// seam. The CPU path remains independently pinned to FunASR in
    /// `parity_fsmn_vad_real`; this leg proves that every released projection
    /// and causal depthwise-memory block reaches the selected Apple backend.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fsmn_vad_real_cpu_metal_full_posterior_parity() {
        let Some(path) = std::env::var_os("VOKRA_FSMN_VAD_REAL_GGUF") else {
            eprintln!("skip: set VOKRA_FSMN_VAD_REAL_GGUF to the strict canonical GGUF");
            return;
        };
        let cpu =
            vokra_models::fsmn_vad::FsmnVadV1::open(&path).expect("bind strict FSMN-VAD for CPU");
        let metal = vokra_models::fsmn_vad::FsmnVadV1::open(&path)
            .expect("bind strict FSMN-VAD for Metal")
            .with_backend(vokra_core::BackendKind::Metal);
        let width = cpu.config().encoder.input_dim;
        let features = (0..13 * width)
            .map(|index| {
                let phase = index as f32;
                (phase * 0.013).sin() * 0.25 + (phase * 0.007).cos() * 0.1
            })
            .collect::<Vec<_>>();
        let cpu_probabilities = cpu
            .forward_features(&features)
            .expect("CPU FSMN feature forward");
        let metal_probabilities = metal
            .forward_features(&features)
            .expect("Metal FSMN feature forward");
        assert_eq!(metal_probabilities.len(), cpu_probabilities.len());
        let max_abs = metal_probabilities
            .iter()
            .zip(&cpu_probabilities)
            .map(|(metal, cpu)| (metal - cpu).abs())
            .fold(0.0f32, f32::max);
        eprintln!("FSMN-VAD real CPU/Metal full-posterior max_abs={max_abs:.9e}");
        assert!(
            max_abs <= 0.01,
            "FSMN-VAD CPU/Metal posterior max |delta| {max_abs:.9e} exceeds the fixed FP32 GPU bound"
        );
    }

    /// Real public-checkpoint CPU/Metal parity for FCPE's complete learned
    /// path. The independent CPU-vs-torchfcpe oracle lives in the models
    /// crate; this Apple-only leg proves the selected Metal backend preserves
    /// the final unrounded F0/confidence track and voiced decisions.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fcpe_real_cpu_metal_track_parity() {
        let Some(model_path) =
            std::env::var_os("VOKRA_FCPE_REAL_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("skip: set VOKRA_FCPE_REAL_GGUF to the strict FCPE v001 GGUF");
            return;
        };
        let wav_path = std::env::var_os("VOKRA_FCPE_REAL_WAV").map_or_else(
            || {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/parity/fcpe/input.wav")
            },
            std::path::PathBuf::from,
        );
        let clip = wav::read_wav(&wav_path).expect("read committed FCPE parity WAV");
        assert_eq!(clip.sample_rate, 16_000);

        let cpu =
            vokra_models::f0::fcpe::FCPE::from_gguf(&model_path).expect("bind strict FCPE for CPU");
        let metal = vokra_models::f0::fcpe::FCPE::from_gguf(&model_path)
            .expect("bind strict FCPE for Metal")
            .with_backend(vokra_core::BackendKind::Metal);
        let cpu_track = cpu
            .extract_real(&clip.samples, clip.sample_rate)
            .expect("FCPE CPU track");
        let metal_track = metal
            .extract_real(&clip.samples, clip.sample_rate)
            .expect("FCPE Metal track");
        assert_eq!(metal_track.len(), cpu_track.len());
        assert!(!cpu_track.is_empty());

        let mut voiced_mismatches = 0usize;
        let mut max_f0_abs = 0.0f32;
        let mut sum_f0_abs = 0.0f64;
        let mut max_confidence_abs = 0.0f32;
        let mut sum_confidence_abs = 0.0f64;
        for (cpu_frame, metal_frame) in cpu_track.iter().zip(&metal_track) {
            assert_eq!(
                metal_frame.time_sec.to_bits(),
                cpu_frame.time_sec.to_bits(),
                "CPU/Metal FCPE timebase diverged"
            );
            voiced_mismatches += usize::from(cpu_frame.voiced != metal_frame.voiced);
            let f0_delta = (cpu_frame.hz - metal_frame.hz).abs();
            max_f0_abs = max_f0_abs.max(f0_delta);
            sum_f0_abs += f64::from(f0_delta);
            let confidence_delta = (cpu_frame.confidence - metal_frame.confidence).abs();
            max_confidence_abs = max_confidence_abs.max(confidence_delta);
            sum_confidence_abs += f64::from(confidence_delta);
        }
        let count = cpu_track.len() as f64;
        let mean_f0_abs = sum_f0_abs / count;
        let mean_confidence_abs = sum_confidence_abs / count;
        eprintln!(
            "FCPE real CPU/Metal: frames={} voiced_mismatches={} \
             f0_max_abs={max_f0_abs:.9e} f0_mean_abs={mean_f0_abs:.9e} \
             confidence_max_abs={max_confidence_abs:.9e} \
             confidence_mean_abs={mean_confidence_abs:.9e}",
            cpu_track.len(),
            voiced_mismatches,
        );
        assert_eq!(voiced_mismatches, 0, "CPU/Metal voiced decisions diverged");
        // M1 8-core GPU measurement on the committed 33-frame official input:
        // f0 max/mean = 1.526e-4 / 3.560e-5 Hz; confidence max/mean =
        // 1.013e-6 / 2.254e-7. The fixed bounds leave modest device/compiler
        // headroom while remaining far below a perceptible or decision-level
        // difference. A Metal-unavailable host errors before reaching here;
        // there is no CPU fallback.
        assert!(
            max_f0_abs <= 2.5e-4 && mean_f0_abs <= 5.0e-5,
            "FCPE CPU/Metal F0 drift max={max_f0_abs:.9e} mean={mean_f0_abs:.9e} Hz \
             exceeds 2.5e-4 / 5e-5 Hz"
        );
        assert!(
            max_confidence_abs <= 2.0e-6 && mean_confidence_abs <= 3.0e-7,
            "FCPE CPU/Metal confidence drift max={max_confidence_abs:.9e} \
             mean={mean_confidence_abs:.9e} exceeds 2e-6 / 3e-7"
        );
    }

    #[test]
    fn vocoder_feature_bytes_enforce_shape_and_finite_contract() {
        let bytes: Vec<u8> = [0.25_f32, -0.5, 1.0, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let (mel, frames) =
            parse_vocoder_feature_bytes(&bytes, 2, "mel.f32", "test-vocoder").unwrap();
        assert_eq!(mel, vec![0.25, -0.5, 1.0, 2.0]);
        assert_eq!(frames, 2);

        let err =
            parse_vocoder_feature_bytes(&bytes[..bytes.len() - 1], 2, "short.f32", "test-vocoder")
                .unwrap_err();
        assert!(err.contains("whole number"), "{err}");

        let nan = f32::NAN.to_le_bytes();
        let err = parse_vocoder_feature_bytes(&nan, 1, "nan.f32", "test-vocoder").unwrap_err();
        assert!(err.contains("non-finite"), "{err}");

        let err = parse_vocoder_feature_bytes(&bytes, 3, "shape.f32", "test-vocoder").unwrap_err();
        assert!(err.contains("exact multiple of channels=3"), "{err}");
    }

    /// Silero is backend-honoring: Metal, explicit CPU and the default CPU all
    /// pass the former guard and reach the VAD task's `--input` contract.
    #[test]
    fn backend_selection_on_silero_reaches_vad_input_contract() {
        let model = silero_fixture();
        let err = main(&args(&["--model", &model, "--backend", "metal"])).unwrap_err();
        assert!(
            !err.contains("is not supported for this model"),
            "Metal must pass the former CPU-only guard: {err}"
        );
        assert!(err.contains("--input"), "Metal reaches the VAD task: {err}");

        let err_cpu = main(&args(&["--model", &model, "--backend", "cpu"])).unwrap_err();
        assert!(
            !err_cpu.contains("is not supported for this model"),
            "cpu passes the guard: {err_cpu}"
        );
        assert!(
            err_cpu.contains("--input"),
            "cpu reaches the VAD task: {err_cpu}"
        );

        let err_unset = main(&args(&["--model", &model])).unwrap_err();
        assert!(
            !err_unset.contains("is not supported for this model"),
            "default backend passes the guard: {err_unset}"
        );
        assert!(
            err_unset.contains("--input"),
            "default reaches the VAD task: {err_unset}"
        );
    }

    /// (a) `--backend cuda` on the CSM (S2S) arch is a loud FR-EX-08 error.
    /// The guard fires before the `--text` contract check, so no --text is
    /// needed to trip it.
    #[test]
    fn non_cpu_backend_on_csm_s2s_is_rejected_loudly() {
        let model = csm_fixture_gguf("backend-guard");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "cuda",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(
            err.contains("--backend cuda is not supported"),
            "names the backend: {err}"
        );
        assert!(
            err.contains("CSM speech-to-speech"),
            "names the engine: {err}"
        );
    }

    /// Moshi is backend-honoring: a Metal selection passes the former
    /// CPU-only guard and reaches the task's ordinary input contract. Uses the
    /// same synthetic checkpoint the duplex smoke test converts.
    #[test]
    fn non_cpu_backend_on_moshi_duplex_reaches_the_engine() {
        let dir = std::env::temp_dir().join(format!("vokra-cli-moshi-be-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        let tok = dir.join("tok.model");
        let gguf = dir.join("moshi.gguf");
        std::fs::write(&ckpt, moshi_fixture::synthetic_checkpoint()).unwrap();
        std::fs::write(&tok, moshi_fixture::spm_blob(13)).unwrap();
        vokra_convert::convert_moshi_file(&ckpt, Some(tok.as_path()), &gguf).expect("convert");
        let err = main(&args(&[
            "--model",
            gguf.to_str().unwrap(),
            "--backend",
            "metal",
        ]))
        .unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            err.contains("--input"),
            "Moshi should pass the backend guard and reach its input contract: {err}"
        );
        assert!(
            !err.contains("--backend metal is not supported"),
            "Moshi must not be classified CPU-only: {err}"
        );
    }

    /// SBV2 has no `.with_backend(...)` seam (Task 38) — a non-CPU
    /// `--backend` is rejected before `run_sbv2` runs, so the metadata-only
    /// fixture (no real tensors) is enough to trigger it.
    #[test]
    fn non_cpu_backend_on_sbv2_is_rejected_loudly() {
        let model = sbv2_metadata_only_gguf("backend");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "metal",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(
            err.contains("--backend metal is not supported"),
            "names the backend: {err}"
        );
        assert!(
            err.contains("SBV2 (Style-Bert-VITS2 v2) TTS"),
            "names the engine: {err}"
        );
    }

    #[test]
    fn bert_token_id_parser_is_strict_and_whitespace_tolerant() {
        assert_eq!(parse_bert_token_ids("101, 42,102").unwrap(), [101, 42, 102]);
        for invalid in ["", "101,,102", "-1", "101,text"] {
            assert!(
                parse_bert_token_ids(invalid).is_err(),
                "invalid input `{invalid}` must fail"
            );
        }
    }

    #[test]
    fn metal_backend_on_standalone_bert_passes_former_cpu_only_guard() {
        let mut builder = vokra_core::gguf::GgufBuilder::new();
        builder.add_string("vokra.model.arch", "deberta_v3");
        vokra_core::stamp_provenance(
            &mut builder,
            vokra_core::LicenseClass::Permissive,
            "Apache-2.0",
            Some("microsoft/deberta-v3-large"),
            None,
        );
        let model = std::env::temp_dir().join(format!(
            "vokra-cli-bert-backend-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&model, builder.to_bytes().expect("serialize GGUF")).unwrap();
        let error = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "metal",
            "--token-ids",
            "1,2",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(model);
        assert!(
            !error.contains("--backend metal is not supported"),
            "Metal must pass the former CPU-only guard: {error}"
        );
        assert!(
            error.contains("missing required GGUF metadata"),
            "metadata-only fixture should reach the concrete binder: {error}"
        );
    }

    /// Exact public-artifact Apple CPU/Metal sweep for the three standalone
    /// BERT-family encoders. The 5e-4 boundary is the repository's established
    /// FP32 device-parity envelope; it was not widened for these models.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn standalone_bert_all_public_cpu_metal_hidden_parity() {
        let Some(directory) = std::env::var_os("VOKRA_BERT_PUBLIC_GGUF_DIR") else {
            eprintln!(
                "skip: set VOKRA_BERT_PUBLIC_GGUF_DIR to the three fixed-revision public GGUFs"
            );
            return;
        };
        let cases: [(&str, &[u32]); 3] = [
            ("chinese-roberta-wwm-ext-large.gguf", &[101, 102]),
            ("deberta-v2-large-japanese-char-wwm.gguf", &[1, 2]),
            ("deberta-v3-large.gguf", &[1, 2]),
        ];
        for (filename, token_ids) in cases {
            let path = std::path::Path::new(&directory).join(filename);
            let file = vokra_core::gguf::GgufFile::open(&path)
                .unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
            let model = vokra_models::bert_runtime::BertRuntime::from_gguf(&file)
                .unwrap_or_else(|error| panic!("bind {}: {error}", path.display()));
            let cpu = model
                .encode(token_ids, vokra_core::BackendKind::Cpu)
                .unwrap_or_else(|error| panic!("CPU {}: {error}", path.display()));
            let metal = model
                .encode(token_ids, vokra_core::BackendKind::Metal)
                .unwrap_or_else(|error| panic!("Metal {}: {error}", path.display()));
            assert_eq!(metal.len(), cpu.len());
            let mut max_abs = 0.0_f32;
            let mut sum_abs = 0.0_f64;
            let mut cpu_peak = 0.0_f32;
            for (&expected, &actual) in cpu.iter().zip(&metal) {
                let delta = (expected - actual).abs();
                max_abs = max_abs.max(delta);
                sum_abs += f64::from(delta);
                cpu_peak = cpu_peak.max(expected.abs());
            }
            let mean_abs = sum_abs / cpu.len() as f64;
            eprintln!(
                "bert-public-metal: {filename} values={} max_abs={max_abs:.9e} mean_abs={mean_abs:.9e}",
                cpu.len()
            );
            assert!(
                cpu_peak > 1e-4,
                "{filename}: CPU negative control is degenerate (peak={cpu_peak:.9e})"
            );
            assert!(
                max_abs <= 5e-4 && mean_abs <= 1e-4,
                "{filename}: CPU/Metal parity exceeded 5e-4/1e-4 (max={max_abs:.9e}, mean={mean_abs:.9e})"
            );
        }
    }

    /// (c) No regression on a backend-honoring arch: `--backend metal` on a
    /// voxtral GGUF passes the guard (voxtral binds `.with_backend(...)` in
    /// the run arm), so the run fails for an UNRELATED reason (missing
    /// --input), never with the guard's "runs on the CPU regardless" message.
    #[test]
    fn non_cpu_backend_on_backend_honoring_voxtral_passes_the_guard() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model =
            std::env::temp_dir().join(format!("vokra-cli-vox-be-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "metal",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(
            !err.contains("runs on the CPU regardless")
                && !err.contains("is not supported for this model"),
            "guard must not fire on voxtral: {err}"
        );
        assert!(
            err.contains("--input"),
            "voxtral reaches its own task: {err}"
        );
    }

    /// (c) No regression on the kokoro arch either: `--backend metal` passes
    /// the guard (kokoro TTS binds `.with_backend(...)`), reaching `run_kokoro`
    /// which requires `--text` before it can synthesize.
    #[test]
    fn non_cpu_backend_on_kokoro_passes_the_guard() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "kokoro-82m-istftnet");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model =
            std::env::temp_dir().join(format!("vokra-cli-kok-be-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "metal",
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(
            !err.contains("runs on the CPU regardless")
                && !err.contains("is not supported for this model"),
            "guard must not fire on kokoro: {err}"
        );
        assert!(err.contains("--text"), "kokoro reaches its own task: {err}");
    }
}
