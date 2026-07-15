# @vokra/web

Vokra speech runtime for the browser (M4-01): Whisper base ASR over

- a **WASM CPU path** — two artifacts (`vokra_wasm_simd128.wasm` /
  `vokra_wasm_base.wasm`); the loader picks one with a
  `WebAssembly.validate` SIMD probe (WASM has no runtime CPU feature
  detection), and
- a **WebGPU backend** — a raw wasm import shim + hand-written glue driving
  `navigator.gpu` from a dedicated worker over a SharedArrayBuffer bridge
  (no `wgpu`, no `wasm-bindgen`; zero runtime npm dependencies).

```js
import { createSession } from "@vokra/web";

const model = await (await fetch("/models/whisper-base.gguf")).arrayBuffer();
const session = await createSession(model, { backend: "cpu" }); // or "webgpu"
const { text, rtf } = await session.transcribe(wavBytes); // 16 kHz mono PCM16 WAV or Float32Array
console.log(text, rtf);
await session.close();
```

## Backend selection is explicit (no silent fallback)

Vokra never downgrades your backend choice silently (FR-EX-08):

- `backend: "webgpu"` **requires** (1) cross-origin isolation and (2) a
  WebGPU adapter. Missing either rejects with an error that names the fix.
- `backend: "cpu"` (default) is the explicit CPU choice.

## COOP/COEP deployment (required for `webgpu`)

The WebGPU path blocks the compute worker on `Atomics.wait` over a
SharedArrayBuffer, which browsers only enable on cross-origin-isolated
pages. Serve your page with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

A dependency-free local server that sets both headers ships with the repo
demo (`web/demo/serve.mjs`). The CPU path needs neither header.

## Models

Models are **never bundled**. Convert a Whisper checkpoint offline with
`vokra-cli convert --model whisper …` (or see `scripts/fetch-demo-models.sh`
in the repo) and serve the `.gguf` next to your app. Sessions load it fully
in memory (no mmap on WASM).

## Node

The package targets browsers. Under Node the CPU path works (see the repo's
`tools/wasm/run-whisper-wasm.mjs` harness); `backend: "webgpu"` reports
"unavailable" — explicitly.

License: Apache-2.0 (see LICENSE / NOTICE).
