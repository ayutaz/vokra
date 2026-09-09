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
Torch declares `setuptools` transitively, but this fixed route never imports
it. The project therefore applies the strict impossible-marker override
`setuptools ; python_version < '0'`; uv omits the package from every platform
closure, and the gate rejects any lock that reintroduces it because the pinned
release bundles an LGPLv3 `autocommand` payload.

The isolated model-free inspection pins `torch==2.7.1` and `torchaudio==2.7.1`; both
resolve from the explicit `https://download.pytorch.org/whl/cpu` index (Linux
uses the corresponding `+cpu` lock rows). PyPI torchaudio and CUDA/NVIDIA
runtime packages are rejected by the lock and smoke gates. The isolated
reference previously used `transformers==4.57.3`, which is
affected by `GHSA-xrqw-3rrv-vx5w` (<5.10.0). The reviewed dependency is now
`transformers==5.10.4`; source/API compatibility remains
`BLOCKED_UNVERIFIED_API_SMOKE` until an authorized VAST model smoke test is
completed. This dependency remediation does not claim API parity.

The pinned upstream source contains one `@check_model_inputs()` decorator,
while Transformers 5.10.4 exposes `check_model_inputs(func)`. Both API smoke
phases therefore import the single shared bounded compatibility adapter from
`qwen_source_compat.py` and apply it only inside the disposable clean VAST
source checkout: target
`qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`, original
bytes `40519`, original SHA-256
`844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628`, exactly
one replacement of `@check_model_inputs()` with `@check_model_inputs`, yielding
patched bytes `40517` and patched SHA-256
`a9da44f2f6b7ff0beb4dd43e8c4c48138e51423e9bcc515a253ea088381d3b9c`. The
evidence status is `COMPATIBILITY_PATCH_APPLIED`; this is not raw upstream
compatibility, and any source/hash/count/path drift blocks before import.

The bounded API smoke is `scripts/publish/vast-ai/run-qwen3-tts-api-smoke.sh`.
It is VAST/Linux x86_64-only, requires `VOKRA_PUBLISH_ON_VAST=1`, and stages
only the fixed 0.6B-Base release plus the authenticated 12-Hz decoder. The
previous `accelerate==1.12.0` route was removed because of
`GHSA-4j2p-28q2-5m79`. Inspection of the pinned official source shows that the
wrapper does not import Accelerate and forwards loader kwargs to Transformers,
so the candidate route uses ordinary CPU `from_pretrained` with
`local_files_only=True`, `dtype=float32`, no `device_map`, and
`low_cpu_mem_usage=False`. The shell and Python gates reject any lock that
reintroduces Accelerate before synchronization or model download. This is a
Accelerate-free load design candidate, not a completed reference result: the
existing owner/license gate remains before dependency synchronization and model
acquisition, the VAST worker requires at least 60 GB RAM and 100 GB free
scratch space, and the real-weight load remains unverified until an authorized
VAST run. After those gates pass it calls the official
`Qwen3TTSModel.from_pretrained` wrapper and emits `api-smoke.json` under the
disposable work directory. The evidence is a
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

Before owner evidence is available, the same VAST runner exposes an independent
model-free API phase:

```text
scripts/publish/vast-ai/run-qwen3-tts-api-smoke.sh --model-free --variant all \
  --expected-head <HEAD>
```

This phase stages only the exact official source and the five metadata files for
each selected model revision (`config.json`, `generation_config.json`,
`merges.txt`, `tokenizer_config.json`, and `vocab.json`). It imports the
official `Qwen3TTSModel`, `Qwen3TTSConfig`, and `Qwen3TTSProcessor` APIs,
constructs only config/processor objects, and records `PASS_MODEL_FREE` with
source/model/operator approval fields still pending. It rejects any checkpoint
file, never calls `Qwen3TTSModel.from_pretrained`, and is not a parity or
publication result. Because the reviewed runtime intentionally excludes the
forbidden `sox` and `onnxruntime` packages, the inspection installs strict
import-only sentinels for both modules while importing the official source.
Only inert `__file__` and valid `__spec__` metadata are allowed; functional
attributes such as `sox.Transformer` and `onnxruntime.InferenceSession` fail
closed. Successful
evidence records each module independently under `optional_sentinels`, with
`accesses=0`, and the original `sys.modules` state is restored.
An import/API incompatibility is emitted as atomic `BLOCKED_INCOMPATIBLE_API`
evidence with `checkpoint_load=NOT_PERFORMED`, never as an unstructured
traceback.

The separate model-free dependency/license audit is
`scripts/publish/vast-ai/audit-qwen3-tts-dependencies.sh`. It is restricted to
an already synchronized Linux x86_64 VAST environment and records the active
Python 3.12 lock closure, installed publisher files, native ELF `NEEDED`
entries, and exact locked-sdist plus fixed source LICENSE path evidence. For
the five HF model revisions whose exact `LICENSE` path returns HTTP 404, it
additionally fetches only
`https://huggingface.co/api/models/{repo}/revision/{revision}` and
accepts the bounded `cardData.license` projection when the API-returned `id`
and `sha` match the pinned repository and revision, `private`/`gated`/`disabled`
are exactly false, and `siblings` is a non-empty safe, duplicate-free
`{rfilename}` tree with no LICENSE-like file. Its stable payload identity is
the SHA-256 and byte length of canonical JSON with
`schema=vokra-hf-model-info-canonical-v1` and only those accepted facts
(including sorted sibling filenames). The raw API body is size-bounded before
parsing but is not retained or hashed as evidence, so ignored API/card fields,
whitespace, and object ordering cannot create evidence drift.
README text and arbitrary metadata are never accepted as license evidence. It never acquires weights,
imports model code, invokes Cargo, or uploads anything. The dependency audit
evidence is currently `STALE_REQUIRES_VAST_AUDIT` because the security removal
of `accelerate==1.12.0` (and its `psutil` transitive dependency) invalidated the
prior installed-payload and native facts; an authorized Linux x86_64 VAST audit
must rerun before owner approval. This stale marker is intentional and does
not bypass dependency review.
The owner-approval scope intentionally excludes this volatile dependency-audit
reference to avoid a hash cycle; the compact bytes, full-report SHA-256, input
hashes, closure/facts, and approval state remain bound separately by the gate.
Every factual package/component record has a canonical full-fact digest bound
back to its manifest row; inactive rows remain pending and carry no installed
license/native claim. The compact evidence records factual installed metadata
only and is not an owner legal conclusion. Owner approval and checkpoint/full
API smoke remain blocked until a fresh exact-head VAST audit and legitimate
dependency/component reviews are recorded; dependency synchronization and
model-free API smoke are independent and may proceed.
Run its `--self-test` locally; do not run the production audit on the
maintainer machine. The production audit can optionally emit the compact
projection with `--compact-output <absent-path>` when a separately authorized
VAST job is collecting a fresh full report.
