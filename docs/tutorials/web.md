# Web (WASM / WebGPU) tutorial

**English** | [日本語](web.ja.md)

This tutorial covers the **Vokra Web runtime** (`web/pkg`, npm package
`@vokra/web`): Whisper base ASR in the browser over

- a **WASM CPU path** (SIMD128 or scalar), and
- a **WebGPU backend** (a raw wasm import shim + hand-written JS glue over
  `navigator.gpu` — no `wgpu` crate, no `wasm-bindgen`; zero runtime npm
  dependencies).

## 1. Install

```sh
npm install @vokra/web
```

> The npm scope is registered by the maintainer (M4-01-T27); until the
> first registry publish, build the package from the repo:
> `scripts/build-wasm.sh pkg` → `web/pkg/`.

## 2. Get a model

Models are **never bundled**. Convert a Whisper base checkpoint offline:

```sh
cargo build --release -p vokra-cli
./target/release/vokra-cli convert \
  --model whisper \
  --input /path/to/whisper-base/model.safetensors \
  --output whisper-base.gguf
```

Serve `whisper-base.gguf` as a static asset next to your app. The Web
runtime loads it fully in memory — there is **no mmap on WASM** (the
`vokra-mmap` zero-copy loader is native-only; the browser path is
fetch → `ArrayBuffer` → the in-memory GGUF parser).

## 3. Transcribe

```js
import { createSession } from "@vokra/web";

const model = await (await fetch("/models/whisper-base.gguf")).arrayBuffer();

// backend is an EXPLICIT choice — Vokra never silently falls back:
const session = await createSession(model, { backend: "cpu" }); // or "webgpu"

const wav = await (await fetch("/audio/jfk-30s.wav")).arrayBuffer(); // 16 kHz mono PCM16
const { text, rtf, wallMs } = await session.transcribe(wav);
console.log(text, `RTF ${rtf.toFixed(3)}`);

await session.close();
```

`transcribe` also accepts a raw `Float32Array` of 16 kHz mono PCM (e.g.
from `AudioContext.decodeAudioData` + resampling).

## 4. Backend selection is explicit (FR-EX-08)

| you pass | behaviour |
|----------|-----------|
| `{ backend: "cpu" }` (default) | WASM CPU path. Works everywhere, no special headers. |
| `{ backend: "webgpu" }` | Requires cross-origin isolation **and** a WebGPU adapter. Missing either **rejects with an explanatory error** — Vokra does not degrade to the CPU behind your back. Choosing the CPU is *your* explicit `"cpu"`. |

The error messages name the fix (COOP/COEP deployment below, or a
WebGPU-enabled browser).

## 5. COOP/COEP deployment (required for `webgpu`)

WebGPU readback (`mapAsync`) is async-only, while Vokra's inference loop is
synchronous — the runtime bridges the two by running inference in a
dedicated Web Worker that blocks on `Atomics.wait` over a
`SharedArrayBuffer` command channel to a main-thread GPU proxy. Browsers
enable `SharedArrayBuffer` only on **cross-origin-isolated** pages, so your
server must send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

For local testing the repo ships a dependency-free server that sets both:

```sh
scripts/build-wasm.sh pkg
node web/demo/serve.mjs          # http://localhost:8788/web/demo/
```

The **CPU path needs neither header** — the demo's cpu backend works on any
static host.

## 6. SIMD128: two artifacts, automatic selection

WASM has **no runtime CPU feature detection** — SIMD acceptance is decided
when the engine validates the module. The package therefore ships two
artifacts (`vokra_wasm_simd128.wasm` / `vokra_wasm_base.wasm`) and the
loader picks one with a `WebAssembly.validate` probe
(`session.meta.artifact` tells you which). **Relaxed SIMD is not used**
(Safari-partial per the quarterly ISA watch; its non-deterministic fma
conflicts with Vokra's parity discipline) — kernels use deterministic
mul + add.

## 7. Memory64 status (survey)

Whisper base (~74 M params) fits comfortably in wasm32 linear memory, so
the runtime targets wasm32. As of this WP's survey: Rust's
`wasm64-unknown-unknown` is Tier 3, browser Memory64 ships in the
Chromium/Firefox lines and not in Safari (the maintainer's browser spot
check re-verifies — support status is recorded, never invented). Large
models on the Web are a follow-up.

## 8. Performance notes (honest state)

The WebGPU backend currently runs **per-op** (upload → dispatch → readback
per kernel). At whisper-base scale this is expected to be *slower* than the
WASM CPU path — the same stage the Metal backend went through before
device-resident chains landed. Measure with the demo's RTF display or
`tools/wasm/parity.html`, and see
`docs/bench-baselines/web-2026-07-15/README.md` for recorded numbers (also
the Kill-switch-G comparison input). Whole-run device residency is the
follow-up.

## 9. Troubleshooting

| symptom | cause / fix |
|---------|-------------|
| `backend webgpu needs SharedArrayBuffer…` | COOP/COEP headers missing — §5. |
| `no WebGPU adapter…` | Browser/context without WebGPU — pick `"cpu"` explicitly or use a WebGPU-enabled browser. |
| `WAV must be 16 kHz mono PCM16` | Resample/convert offline (`ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav`) or pass a `Float32Array`. |
| Node: `backend: "webgpu"` fails | Expected — Node has no `navigator.gpu`; the failure is the explicit unavailability contract. |
