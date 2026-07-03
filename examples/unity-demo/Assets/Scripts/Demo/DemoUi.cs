// DemoUi.cs — the single MonoBehaviour that drives the demo scene (M0-10-T08).
//
// Two modes:
//   * Interactive (Editor / windowed player): a minimal IMGUI (OnGUI) panel with a
//     Run button, a scrolling status log, and a Play-TTS button.
//   * Headless (-batchmode, or when -vokraInput is passed): runs the pipeline once,
//     logs every event, writes the TTS WAV (-vokraOutput), and quits with an exit
//     code (0 = ok, 1 = a stage errored). This is the machine-checkable path used
//     for the Linux verification (T11): `-batchmode -nographics` needs no audio
//     device or display.
//
// UI choice: the ticket proposes uGUI, but this demo uses IMGUI so the whole UI is
// code-only — no Canvas/EventSystem/Font asset/TextMeshPro setup that varies across
// Unity versions and would need scene authoring. The v0.5 official plugin
// (FR-API-04) can ship a richer uGUI/UI-Toolkit front end.
//
// Threading: PipelineRunner does every C ABI call on one worker thread; DemoUi only
// drains its ConcurrentQueue on the Unity main thread (Update) — no native call is
// ever made from the render/UI code (T07).

using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace Vokra.Demo
{
    [DisallowMultipleComponent]
    public sealed class DemoUi : MonoBehaviour
    {
        private PipelineRunner _runner;
        private AudioSource _audio;

        private readonly StringBuilder _log = new StringBuilder();
        private Vector2 _scroll;
        private bool _running;
        private bool _hadError;

        private float[] _ttsPcm;
        private int _ttsRate;

        private PipelineConfig _config;

        private void Awake()
        {
            _audio = gameObject.AddComponent<AudioSource>();
            _config = BuildConfig();
        }

        private void Start()
        {
            Append($"Vokra Unity demo — models: {_config.ModelsDir}");
            Append($"input: {_config.InputWavPath}");

            if (Application.isBatchMode)
            {
                RunHeadless();
            }
        }

        // ---- configuration from command line / StreamingAssets defaults ----

        private PipelineConfig BuildConfig()
        {
            string modelsDir = GetArg("-vokraModelsDir", Path.Combine(Application.streamingAssetsPath, "models"));
            string input = GetArg("-vokraInput", Path.Combine(Application.streamingAssetsPath, "test_16k.wav"));
            string output = GetArg("-vokraOutput", null);
            string text = GetArg("-vokraText", null);

            var cfg = new PipelineConfig
            {
                ModelsDir = modelsDir,
                InputWavPath = input,
                OutputWavPath = output,
            };
            if (!string.IsNullOrEmpty(text))
            {
                cfg.TtsFallbackText = text;
            }

            return cfg;
        }

        private static string GetArg(string name, string fallback)
        {
            string[] argv = Environment.GetCommandLineArgs();
            for (int i = 0; i < argv.Length - 1; i++)
            {
                if (string.Equals(argv[i], name, StringComparison.Ordinal))
                {
                    return argv[i + 1];
                }
            }

            return fallback;
        }

        // ---- interactive path ----

        private void StartPipeline()
        {
            if (_running)
            {
                return;
            }

            _ttsPcm = null;
            _running = true;
            _runner = new PipelineRunner(_config);
            _runner.Start();
        }

        private void Update()
        {
            if (_runner == null)
            {
                return;
            }

            while (_runner.TryDequeue(out PipelineEvent evt))
            {
                HandleEvent(evt);
            }
        }

        private void HandleEvent(PipelineEvent evt)
        {
            if (evt.Stage == PipelineStage.Tts && evt.Pcm != null && evt.Pcm.Length > 0)
            {
                _ttsPcm = evt.Pcm;
                _ttsRate = evt.SampleRate;
            }

            if (evt.Stage == PipelineStage.Error)
            {
                _hadError = true;
            }

            if (evt.Stage == PipelineStage.Done)
            {
                _running = false;
            }

            Append($"[{evt.Stage}] {evt.Message}");
        }

        private void PlayTts()
        {
            if (_ttsPcm == null)
            {
                return;
            }

            AudioClip clip = AudioClipUtil.ToMonoClip("vokra-tts", _ttsPcm, _ttsRate);
            if (clip != null)
            {
                _audio.PlayOneShot(clip);
            }
        }

        private void OnGUI()
        {
            const int pad = 10;
            var area = new Rect(pad, pad, Screen.width - 2 * pad, Screen.height - 2 * pad);
            GUILayout.BeginArea(area);

            GUILayout.Label("Vokra — VAD → ASR → TTS demo", GUI.skin.box);

            GUILayout.BeginHorizontal();
            GUI.enabled = !_running;
            if (GUILayout.Button("Run pipeline", GUILayout.Height(32)))
            {
                StartPipeline();
            }

            GUI.enabled = _ttsPcm != null && !_running;
            if (GUILayout.Button("Play TTS", GUILayout.Height(32)))
            {
                PlayTts();
            }

            GUI.enabled = true;
            GUILayout.EndHorizontal();

            _scroll = GUILayout.BeginScrollView(_scroll, GUI.skin.box);
            GUILayout.Label(_log.ToString());
            GUILayout.EndScrollView();

            GUILayout.EndArea();
        }

        // ---- headless path (T11) ----

        private void RunHeadless()
        {
            Append("headless: running pipeline once");
            try
            {
                var runner = new PipelineRunner(_config);
                runner.RunBlocking();
                while (runner.TryDequeue(out PipelineEvent evt))
                {
                    if (evt.Stage == PipelineStage.Error)
                    {
                        _hadError = true;
                    }

                    Debug.Log($"[vokra-demo][{evt.Stage}] {evt.Message}");
                }
            }
            catch (Exception ex)
            {
                _hadError = true;
                Debug.LogError($"[vokra-demo] {ex}");
            }

            int code = _hadError ? 1 : 0;
            Debug.Log($"[vokra-demo] exit code {code}");
            Application.Quit(code);
        }

        private void Append(string line)
        {
            _log.AppendLine(line);
            _scroll.y = float.MaxValue; // stick to bottom
        }
    }
}
