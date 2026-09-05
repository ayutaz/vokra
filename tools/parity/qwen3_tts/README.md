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

The bounded API smoke is `scripts/publish/vast-ai/run-qwen3-tts-api-smoke.sh`.
It is VAST/Linux x86_64-only, requires `VOKRA_PUBLISH_ON_VAST=1`, and stages
only the fixed 0.6B-Base release plus the authenticated 12-Hz decoder. The
worker is currently fail-closed at the existing unresolved license manifest
(the first reported blocker is `accelerate==1.12.0`), so it cannot sync,
download, import, or run a model until legitimate dependency/component reviews
and authenticated owner evidence are recorded. After all gates pass it calls
the official `Qwen3TTSModel.from_pretrained` wrapper
with `local_files_only=True`, `dtype=float32`, and `device_map="cpu"`, then
emits `api-smoke.json` under the disposable work directory. The evidence is a
strict `vokra-qwen3-tts-api-smoke-v1` JSON document containing the exact source,
model, decoder, lock, approval-evidence SHA-256 plus the existing license gate
manifest digest/approval scope/owner sign-offs, Vokra checkout HEAD/clean
status, package-version, input-hash, and call-checkpoint records; its
publication value is always `NO_UPLOAD`. The Python worker repeats the VAST
platform gate and rejects direct output paths that overlap any authenticated
input or have symlink ancestry. Run
`scripts/publish/vast-ai/run-qwen3-tts-api-smoke.sh --self-test` locally for
the no-model/no-network contract checks. The full four-variant validation
remains blocked until this API smoke has an authenticated VAST result.
