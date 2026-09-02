# Qwen3-TTS independent real-weight reference

This project imports the official `QwenLM/Qwen3-TTS` source tree at
revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`. The VAST worker checks
out that exact Git commit and passes it through `PYTHONPATH`; it does not
install the upstream project metadata or its optional extras. It does not mirror
the talker, code predictor, prompt builder, or 12-Hz decoder. The dumper calls
the official `Qwen3TTSModel` wrapper and captures the exact generated
`audio_codes` packet before the official tokenizer decodes it to PCM.

The four main model snapshots are separately pinned by immutable Hugging Face
revision. Base variants use the official speaker encoder in x-vector-only mode;
its fixed embedding is written as `speaker_embedding.f32le` for native Vokra
input. CustomVoice variants use the official fixed `Serena` speaker table.

The generated manifest records source/model revisions, official package
revision, nested-vs-standalone decoder identity, prompt/generation settings,
and hashes for every output. Missing official imports, non-finite output,
revision drift, or malformed codes aborts. Run with `--decoder-dir` pointing at
the separately staged `Qwen/Qwen3-TTS-Tokenizer-12Hz` snapshot; the nested
`speech_tokenizer/model.safetensors` must have the exact same authenticated
SHA-256 as that standalone checkpoint.
The minimal lock contains only the inference closure. `librosa`, `soundfile`,
and their `soxr` native-audio path remain because the official wrapper imports
them unconditionally for voice-clone audio normalization; `gradio`,
`onnxruntime`, `protobuf`, and `sox` demo/ONNX/SoX paths are excluded. The tool
is offline after source and model snapshots are staged and never uploads or
publishes.

The isolated reference previously used `transformers==4.57.3`, which is
affected by `GHSA-xrqw-3rrv-vx5w` (<5.10.0). The reviewed dependency is now
`transformers==5.10.4`; source/API compatibility remains
`BLOCKED_UNVERIFIED_API_SMOKE` until an authorized VAST model smoke test is
completed. This dependency remediation does not claim API parity.
