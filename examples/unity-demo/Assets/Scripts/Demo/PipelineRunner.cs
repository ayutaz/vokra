// PipelineRunner.cs — VAD → ASR → TTS driver on a single worker thread (M0-10-T07).
//
// The whole C ABI call sequence (session/stream create → use → destroy) runs on
// ONE background thread. This is deliberate: FR-API-03 (Session Send+Sync, atomic
// ref count) is v0.1 MVP scope, so the M0 demo does not rely on runtime thread
// safety; and vokra_last_error is a thread-local errno, so producing and reading
// an error on the same thread is required (T04). Results are handed to the Unity
// main thread only through a ConcurrentQueue<PipelineEvent> (drained in DemoUi.Update).
//
// This type is intentionally free of any UnityEngine dependency so the sequencing
// is host-agnostic and side-effect-free apart from the queue and optional file out.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using Vokra;

namespace Vokra.Demo
{
    public enum PipelineStage
    {
        Info,
        Vad,
        Asr,
        Tts,
        Error,
        Done,
    }

    /// <summary>One message from the worker thread to the UI.</summary>
    public sealed class PipelineEvent
    {
        public PipelineStage Stage;
        public string Message;

        // Set only on the Tts stage: the synthesized mono f32 PCM and its rate.
        public float[] Pcm;
        public int SampleRate;

        public PipelineEvent(PipelineStage stage, string message)
        {
            Stage = stage;
            Message = message;
        }
    }

    public sealed class PipelineConfig
    {
        public string ModelsDir;
        public string InputWavPath;

        // Optional: when set (headless mode), the TTS output is written here.
        public string OutputWavPath;

        // Demo model filenames placed by scripts/fetch-demo-models.sh (T09).
        public string SileroFile = "silero-vad-v5.gguf";
        public string WhisperFile = "whisper-base.gguf";
        public string PiperFile = "voice.gguf";

        // Sample rates required by the models (Vokra does not resample in M0).
        public int VadSampleRate = 16000;
        public int AsrSampleRate = 16000;

        // Silero speech-probability threshold and 16 kHz frame size (512 samples).
        public float VadThreshold = 0.5f;
        public int VadFrameSamples = 512;

        // Spoken when ASR produced no real text yet (M0 Whisper emits bracketed
        // token ids until the tokenizer is embedded in the GGUF — M0-09 followup).
        public string TtsFallbackText = "Hello from Vokra.";
    }

    public sealed class PipelineRunner
    {
        private readonly PipelineConfig _config;
        private readonly ConcurrentQueue<PipelineEvent> _events = new ConcurrentQueue<PipelineEvent>();
        private Thread _thread;
        private volatile bool _finished;

        public PipelineRunner(PipelineConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
        }

        public bool Finished => _finished;

        /// <summary>Drains one queued event, if any. Call from the main thread.</summary>
        public bool TryDequeue(out PipelineEvent evt) => _events.TryDequeue(out evt);

        /// <summary>Starts the pipeline on a background worker thread (idempotent).</summary>
        public void Start()
        {
            if (_thread != null)
            {
                return;
            }

            _thread = new Thread(Run) { IsBackground = true, Name = "vokra-pipeline" };
            _thread.Start();
        }

        /// <summary>Runs the pipeline synchronously on the calling thread (headless mode).</summary>
        public void RunBlocking() => Run();

        private void Emit(PipelineStage stage, string message) => _events.Enqueue(new PipelineEvent(stage, message));

        private void Run()
        {
            try
            {
                Emit(PipelineStage.Info, $"Vokra runtime {VokraSession.RuntimeVersion}");

                float[] pcm = WavIo.ReadMonoExpectingRate(_config.InputWavPath, _config.VadSampleRate);
                Emit(PipelineStage.Info, $"input: {Path.GetFileName(_config.InputWavPath)} " +
                    $"({pcm.Length} samples @ {_config.VadSampleRate} Hz)");

                string asrText = RunVadThenAsr(pcm);
                RunTts(asrText);
            }
            catch (Exception ex)
            {
                Emit(PipelineStage.Error, ex.Message);
            }
            finally
            {
                _finished = true;
                Emit(PipelineStage.Done, "pipeline finished");
            }
        }

        private string RunVadThenAsr(float[] pcm)
        {
            // --- VAD (always available: the Silero fixture is committed) ---
            string sileroPath = Path.Combine(_config.ModelsDir, _config.SileroFile);
            if (File.Exists(sileroPath))
            {
                try
                {
                    using VokraSession vad = VokraSession.CreateFromFile(sileroPath);
                    using VokraStream stream = vad.OpenVadStream(_config.VadSampleRate);

                    int speech = 0, total = 0;
                    const int chunk = 2048;
                    for (int off = 0; off < pcm.Length; off += chunk)
                    {
                        int len = Math.Min(chunk, pcm.Length - off);
                        var slice = new float[len];
                        Array.Copy(pcm, off, slice, 0, len);
                        stream.PushPcm(slice);
                        foreach (float p in stream.PollAll())
                        {
                            total++;
                            if (p >= _config.VadThreshold)
                            {
                                speech++;
                            }
                        }
                    }

                    float frameMs = 1000f * _config.VadFrameSamples / _config.VadSampleRate;
                    Emit(PipelineStage.Vad,
                        $"VAD: {speech}/{total} frames above {_config.VadThreshold:0.00} " +
                        $"(~{speech * frameMs / 1000f:0.00}s speech of {total * frameMs / 1000f:0.00}s)");
                }
                catch (Exception ex)
                {
                    Emit(PipelineStage.Error, $"VAD: {ex.Message}");
                }
            }
            else
            {
                Emit(PipelineStage.Vad, $"VAD skipped: {sileroPath} not found (run fetch-demo-models.sh)");
            }

            // --- ASR (needs the uncommitted Whisper base GGUF) ---
            string whisperPath = Path.Combine(_config.ModelsDir, _config.WhisperFile);
            if (!File.Exists(whisperPath))
            {
                Emit(PipelineStage.Asr, $"ASR skipped: {whisperPath} not found (place Whisper base GGUF — T09)");
                return null;
            }

            try
            {
                using VokraSession asr = VokraSession.CreateFromFile(whisperPath);
                string text = asr.Transcribe(pcm, _config.AsrSampleRate);
                Emit(PipelineStage.Asr, $"ASR: {text}");
                return text;
            }
            catch (Exception ex)
            {
                Emit(PipelineStage.Error, $"ASR: {ex.Message}");
                return null;
            }
        }

        private void RunTts(string asrText)
        {
            string piperPath = Path.Combine(_config.ModelsDir, _config.PiperFile);
            if (!File.Exists(piperPath))
            {
                Emit(PipelineStage.Tts, $"TTS skipped: {piperPath} not found (place piper-plus voice GGUF — T09)");
                return;
            }

            // In M0, Whisper emits bracketed token ids (no embedded tokenizer yet),
            // which are not speakable text — fall back to a fixed demo sentence.
            string text = LooksLikeText(asrText) ? asrText : _config.TtsFallbackText;

            try
            {
                using VokraSession tts = VokraSession.CreateFromFile(piperPath);
                (float[] outPcm, int rate) = tts.Synthesize(text);

                var evt = new PipelineEvent(PipelineStage.Tts,
                    $"TTS: {outPcm.Length} samples @ {rate} Hz for \"{text}\"")
                {
                    Pcm = outPcm,
                    SampleRate = rate,
                };
                _events.Enqueue(evt);

                if (!string.IsNullOrEmpty(_config.OutputWavPath))
                {
                    WavIo.WritePcm16(_config.OutputWavPath, outPcm, rate);
                    Emit(PipelineStage.Tts, $"TTS: wrote {_config.OutputWavPath}");
                }
            }
            catch (Exception ex)
            {
                Emit(PipelineStage.Error, $"TTS: {ex.Message}");
            }
        }

        private static bool LooksLikeText(string s)
        {
            if (string.IsNullOrWhiteSpace(s))
            {
                return false;
            }

            // The M0 no-tokenizer fallback renders as "[no tokenizer; token ids: ...]".
            return !s.TrimStart().StartsWith("[", StringComparison.Ordinal);
        }
    }
}
