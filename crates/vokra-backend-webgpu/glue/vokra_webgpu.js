// vokra_webgpu.js — hand-written import-object glue for the Vokra WebGPU
// backend (M4-01-T09/T10; ADR M4-01-webgpu-wasm §3/§5).
//
// Zero npm dependencies (NFR-DS-02 red line: no wasm-bindgen/web-sys — the
// import surface is the hand-maintained mirror of
// crates/vokra-backend-webgpu/src/sys.rs; keep the two in lock-step).
//
// Three pieces:
//
//   createUnavailableImports(reason)
//     The dlopen-failure analogue: every import exists (instantiation
//     REQUIRES the full surface) but probe() reports 0 / ops fail with a
//     readable message. Used by Node harnesses and by browsers without a
//     WebGPU adapter. The Rust side maps this to an explicit
//     VokraError::BackendUnavailable (FR-EX-08 — never a silent CPU
//     fallback; the CPU is the caller's explicit BackendKind::Cpu choice).
//
//   createProxyImports({ctl, data, kick, getMemory})
//     The worker-side implementation: forwards every GPU call over a
//     SharedArrayBuffer command channel to the main-thread GPU proxy and
//     blocks on Atomics.wait (worker-legal; ADR M4-01 §3). Payloads are
//     chunked through the data SAB, so buffers larger than the channel
//     (model weights) stream in DATA_CAPACITY pieces.
//
//   initGpuProxy() / attachVokraGpuProxy(worker, {ctl, data, device})
//     The main-thread half: owns the GPUDevice, services requests on
//     message-kick, answers with Atomics.store + Atomics.notify (the main
//     thread never waits — only the worker does).
//
// Readback consolidation (M2-01 lesson): the Rust side calls buffer_read at
// run boundaries only; the bridge itself supports arbitrarily-sized reads
// via read_begin/read_chunk/read_end (one mapAsync per read, chunked copy).

export const VOKRA_WEBGPU_IMPORT_MODULE = "vokra_webgpu";

// ---- SAB channel layout -----------------------------------------------------
// ctl:  Int32Array(8) over a SharedArrayBuffer
//   [0] TURN   0 = worker owns / idle, 1 = request posted, 2 = response ready
//   [1] STATUS i32 result of the last command (>= 0 ok, < 0 failure)
//   [2] LEN    response payload length in `data`
// data: Uint8Array over a SharedArrayBuffer
//   [0..64)  request header: Int32Array view = [cmd, a0..a7]
//   [64.. )  request/response payload (chunked)

export const CTL_WORDS = 8;
export const HEADER_BYTES = 64;
export const DATA_CAPACITY = 16 * 1024 * 1024; // 16 MiB payload channel

const TURN = 0;
const STATUS = 1;
const LEN = 2;

// Command codes (worker → proxy).
const CMD_BUF_CREATE = 4;
const CMD_BUF_WRITE_CHUNK = 5;
const CMD_BUF_READ_BEGIN = 6;
const CMD_BUF_READ_CHUNK = 7;
const CMD_BUF_READ_END = 8;
const CMD_BUF_DESTROY = 9;
const CMD_SHADER_CREATE = 10;
const CMD_PIPELINE_CREATE = 11;
const CMD_DISPATCH = 12;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

// ---- mode 1: unavailable (no adapter / Node) --------------------------------

/**
 * Import object arm for hosts without WebGPU. Every import is present (the
 * module cannot instantiate otherwise); probe() reports 0 and every op
 * fails with `reason` readable through error_len/error_read — the exact
 * analogue of a dlopen failure (FR-EX-08: the Rust side surfaces an
 * explicit BackendUnavailable).
 */
export function createUnavailableImports(reason, getMemory) {
  let lastError = textEncoder.encode(String(reason ?? "WebGPU is not available in this host"));
  const fail = () => {
    return -1;
  };
  return {
    vokra_webgpu_probe: () => 0,
    vokra_webgpu_error_len: () => lastError.length,
    vokra_webgpu_error_read: (dst, cap) => {
      const n = Math.min(lastError.length, cap >>> 0);
      new Uint8Array(getMemory().buffer, dst >>> 0, n).set(lastError.subarray(0, n));
      return n;
    },
    vokra_webgpu_buffer_create: () => 0,
    vokra_webgpu_buffer_write: fail,
    vokra_webgpu_buffer_read: fail,
    vokra_webgpu_buffer_destroy: () => {},
    vokra_webgpu_shader_create: () => 0,
    vokra_webgpu_pipeline_create: () => 0,
    vokra_webgpu_dispatch: fail,
    vokra_webgpu_now_ms: () => (globalThis.performance ? performance.now() : Date.now()),
  };
}

// ---- mode 2: worker-side proxy imports --------------------------------------

/**
 * Import object arm used INSIDE the dedicated compute worker when the main
 * thread reported a live GPUDevice. `kick` posts a wake-up message to the
 * proxy; `getMemory` returns the instantiated module's WebAssembly.Memory
 * (resolved lazily — the imports are built before instantiation).
 */
export function createProxyImports({ ctl, data, kick, getMemory }) {
  const header = new Int32Array(data.buffer, 0, HEADER_BYTES / 4);
  const payload = () => new Uint8Array(data.buffer, HEADER_BYTES);
  let lastError = new Uint8Array(0);

  /** Round-trip one command synchronously (Atomics.wait — worker-only). */
  function call(cmd, args = [], payloadBytes = null) {
    header[0] = cmd;
    for (let i = 0; i < 8; i++) header[1 + i] = args[i] ?? 0;
    if (payloadBytes) payload().set(payloadBytes);
    Atomics.store(ctl, STATUS, 0);
    Atomics.store(ctl, TURN, 1);
    kick();
    // Block until the proxy flips TURN away from 1 (worker-legal wait).
    while (Atomics.load(ctl, TURN) === 1) {
      Atomics.wait(ctl, TURN, 1);
    }
    const status = Atomics.load(ctl, STATUS);
    const len = Atomics.load(ctl, LEN);
    let resp = null;
    if (len > 0) {
      resp = new Uint8Array(len);
      resp.set(payload().subarray(0, len));
    }
    if (status < 0 && resp) lastError = resp;
    Atomics.store(ctl, TURN, 0);
    return { status, resp };
  }

  const CHUNK = DATA_CAPACITY - HEADER_BYTES;

  return {
    // The proxy path only exists when the device is live.
    vokra_webgpu_probe: () => 1,
    vokra_webgpu_error_len: () => lastError.length,
    vokra_webgpu_error_read: (dst, cap) => {
      const n = Math.min(lastError.length, cap >>> 0);
      new Uint8Array(getMemory().buffer, dst >>> 0, n).set(lastError.subarray(0, n));
      return n;
    },
    vokra_webgpu_buffer_create: (size, usage) => {
      const { status } = call(CMD_BUF_CREATE, [size >>> 0, usage >>> 0]);
      return status > 0 ? status : 0;
    },
    vokra_webgpu_buffer_write: (buf, offset, src, len) => {
      const mem = new Uint8Array(getMemory().buffer);
      let done = 0;
      while (done < (len >>> 0)) {
        const n = Math.min(CHUNK, (len >>> 0) - done);
        const chunk = mem.subarray((src >>> 0) + done, (src >>> 0) + done + n);
        const { status } = call(CMD_BUF_WRITE_CHUNK, [buf >>> 0, (offset >>> 0) + done, n], chunk);
        if (status < 0) return -1;
        done += n;
      }
      return 0;
    },
    vokra_webgpu_buffer_read: (buf, offset, dst, len) => {
      // read_begin maps a staging copy on the proxy; chunks stream back.
      const begin = call(CMD_BUF_READ_BEGIN, [buf >>> 0, offset >>> 0, len >>> 0]);
      if (begin.status < 0) return -1;
      const mem = new Uint8Array(getMemory().buffer);
      let done = 0;
      let failed = false;
      while (done < (len >>> 0)) {
        const n = Math.min(CHUNK, (len >>> 0) - done);
        const { status, resp } = call(CMD_BUF_READ_CHUNK, [done, n]);
        if (status < 0 || !resp || resp.length !== n) {
          failed = true;
          break;
        }
        mem.set(resp, (dst >>> 0) + done);
        done += n;
      }
      const end = call(CMD_BUF_READ_END, []);
      return failed || end.status < 0 ? -1 : 0;
    },
    vokra_webgpu_buffer_destroy: (buf) => {
      call(CMD_BUF_DESTROY, [buf >>> 0]);
    },
    vokra_webgpu_shader_create: (namePtr, nameLen, srcPtr, srcLen) => {
      const mem = new Uint8Array(getMemory().buffer);
      const name = mem.subarray(namePtr >>> 0, (namePtr >>> 0) + (nameLen >>> 0));
      const src = mem.subarray(srcPtr >>> 0, (srcPtr >>> 0) + (srcLen >>> 0));
      if (name.length + src.length + 8 > CHUNK) return 0; // sources are ~KB; guard anyway
      const body = new Uint8Array(4 + name.length + src.length);
      new DataView(body.buffer).setUint32(0, name.length, true);
      body.set(name, 4);
      body.set(src, 4 + name.length);
      const { status } = call(CMD_SHADER_CREATE, [body.length], body);
      return status > 0 ? status : 0;
    },
    vokra_webgpu_pipeline_create: (shader, entryPtr, entryLen) => {
      const mem = new Uint8Array(getMemory().buffer);
      const entry = new Uint8Array(entryLen >>> 0);
      entry.set(mem.subarray(entryPtr >>> 0, (entryPtr >>> 0) + (entryLen >>> 0)));
      const { status } = call(CMD_PIPELINE_CREATE, [shader >>> 0, entry.length], entry);
      return status > 0 ? status : 0;
    },
    vokra_webgpu_dispatch: (pipeline, bufsPtr, bufsLen, uniformPtr, uniformLen, wgX, wgY, wgZ) => {
      const mem = getMemory();
      const ids = new Uint32Array(bufsLen >>> 0);
      // The wasm-side ids array may be unaligned relative to Uint32Array's
      // requirement in exotic layouts; copy via DataView (alignment-free).
      const dv = new DataView(mem.buffer);
      for (let i = 0; i < ids.length; i++) ids[i] = dv.getUint32((bufsPtr >>> 0) + 4 * i, true);
      const uni = new Uint8Array(uniformLen >>> 0);
      uni.set(new Uint8Array(mem.buffer, uniformPtr >>> 0, uniformLen >>> 0));
      const body = new Uint8Array(4 + ids.length * 4 + uni.length);
      const bdv = new DataView(body.buffer);
      bdv.setUint32(0, ids.length, true);
      for (let i = 0; i < ids.length; i++) bdv.setUint32(4 + 4 * i, ids[i], true);
      body.set(uni, 4 + ids.length * 4);
      const { status } = call(
        CMD_DISPATCH,
        [pipeline >>> 0, ids.length, uni.length, wgX >>> 0, wgY >>> 0, wgZ >>> 0],
        body,
      );
      return status < 0 ? -1 : 0;
    },
    vokra_webgpu_now_ms: () => performance.now(),
  };
}

// ---- main-thread GPU proxy ---------------------------------------------------

/**
 * Requests the adapter + device on the main thread (async — the proxy's
 * event loop stays free, which is what makes the worker-side Atomics.wait
 * bridge sound). Returns {ok:true, device} or {ok:false, error}.
 */
export async function initGpuProxy() {
  if (!("gpu" in navigator)) {
    return { ok: false, error: "navigator.gpu is absent (this browser has no WebGPU)" };
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, error: `requestAdapter() threw: ${e}` };
  }
  if (!adapter) {
    return { ok: false, error: "requestAdapter() returned null (no WebGPU adapter)" };
  }
  try {
    const device = await adapter.requestDevice();
    return { ok: true, device };
  } catch (e) {
    return { ok: false, error: `requestDevice() failed: ${e}` };
  }
}

/**
 * Attaches the command-servicing proxy for `worker` on the main thread.
 * Owns the glue-side handle tables (buffers / shaders / pipelines). The
 * worker posts {vokraKick:true} after writing a request; the proxy answers
 * with Atomics.store + Atomics.notify (the main thread never waits).
 */
export function attachVokraGpuProxy(worker, { ctl, data, device }) {
  const header = new Int32Array(data.buffer, 0, HEADER_BYTES / 4);
  const payloadView = () => new Uint8Array(data.buffer, HEADER_BYTES);

  const buffers = new Map();
  const shaders = new Map(); // id -> {module, name}
  const shaderByName = new Map(); // name -> id (pipeline cache key)
  const pipelines = new Map();
  let nextId = 1;
  let mapped = null; // {staging, bytes} during a read_begin..read_end window

  const USAGE_INPUT = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const USAGE_OUTPUT = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  function respond(status, respBytes = null) {
    let len = 0;
    if (respBytes) {
      len = respBytes.length;
      payloadView().set(respBytes.subarray(0, len));
    }
    Atomics.store(ctl, LEN, len);
    Atomics.store(ctl, STATUS, status);
    Atomics.store(ctl, TURN, 2);
    Atomics.notify(ctl, TURN);
  }

  function respondError(message) {
    respond(-1, textEncoder.encode(String(message)));
  }

  async function service() {
    const cmd = header[0];
    const a = [];
    for (let i = 0; i < 8; i++) a.push(header[1 + i] >>> 0);
    try {
      switch (cmd) {
        case CMD_BUF_CREATE: {
          const [size, usage] = a;
          const buffer = device.createBuffer({
            size: Math.max(4, Math.ceil(size / 4) * 4),
            usage: usage === 1 ? USAGE_OUTPUT : USAGE_INPUT,
          });
          const id = nextId++;
          buffers.set(id, buffer);
          respond(id);
          break;
        }
        case CMD_BUF_WRITE_CHUNK: {
          const [id, offset, len] = a;
          const buffer = buffers.get(id);
          if (!buffer) return respondError(`buffer_write: unknown buffer ${id}`);
          // Copy out of the SAB (writeBuffer requires a non-shared source in
          // some engines) then upload.
          const chunk = new Uint8Array(len);
          chunk.set(payloadView().subarray(0, len));
          device.queue.writeBuffer(buffer, offset, chunk);
          respond(0);
          break;
        }
        case CMD_BUF_READ_BEGIN: {
          const [id, offset, len] = a;
          const buffer = buffers.get(id);
          if (!buffer) return respondError(`buffer_read: unknown buffer ${id}`);
          if (mapped) return respondError("buffer_read: a read window is already open");
          const staging = device.createBuffer({
            size: Math.max(4, Math.ceil(len / 4) * 4),
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(buffer, offset, staging, 0, Math.ceil(len / 4) * 4);
          device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          mapped = { staging, bytes: new Uint8Array(staging.getMappedRange(), 0, len) };
          respond(0);
          break;
        }
        case CMD_BUF_READ_CHUNK: {
          const [chunkOff, len] = a;
          if (!mapped) return respondError("buffer_read: no open read window");
          respond(0, mapped.bytes.subarray(chunkOff, chunkOff + len));
          break;
        }
        case CMD_BUF_READ_END: {
          if (mapped) {
            mapped.staging.unmap();
            mapped.staging.destroy();
            mapped = null;
          }
          respond(0);
          break;
        }
        case CMD_BUF_DESTROY: {
          const [id] = a;
          const buffer = buffers.get(id);
          if (buffer) {
            buffer.destroy();
            buffers.delete(id);
          }
          respond(0);
          break;
        }
        case CMD_SHADER_CREATE: {
          const body = payloadView().subarray(0, a[0]);
          const nameLen = new DataView(body.buffer, body.byteOffset, 4).getUint32(0, true);
          const name = textDecoder.decode(body.subarray(4, 4 + nameLen));
          if (shaderByName.has(name)) return respond(shaderByName.get(name));
          const source = textDecoder.decode(body.subarray(4 + nameLen));
          // Compilation is the browser/driver's responsibility (NFR-RL-05).
          const module = device.createShaderModule({ label: `vokra:${name}`, code: source });
          const id = nextId++;
          shaders.set(id, { module, name });
          shaderByName.set(name, id);
          respond(id);
          break;
        }
        case CMD_PIPELINE_CREATE: {
          const [shaderId, entryLen] = a;
          const sh = shaders.get(shaderId);
          if (!sh) return respondError(`pipeline_create: unknown shader ${shaderId}`);
          const entryPoint = textDecoder.decode(payloadView().subarray(0, entryLen));
          const pipeline = device.createComputePipeline({
            label: `vokra:${sh.name}:${entryPoint}`,
            layout: "auto",
            compute: { module: sh.module, entryPoint },
          });
          const id = nextId++;
          pipelines.set(id, pipeline);
          respond(id);
          break;
        }
        case CMD_DISPATCH: {
          const [pipelineId, nBufs, uniformLen, wgX, wgY, wgZ] = a;
          const pipeline = pipelines.get(pipelineId);
          if (!pipeline) return respondError(`dispatch: unknown pipeline ${pipelineId}`);
          const body = payloadView();
          const bdv = new DataView(body.buffer, body.byteOffset);
          const count = bdv.getUint32(0, true);
          if (count !== nBufs) return respondError("dispatch: bufs header mismatch");
          const entries = [];
          for (let i = 0; i < count; i++) {
            const id = bdv.getUint32(4 + 4 * i, true);
            const buffer = buffers.get(id);
            if (!buffer) return respondError(`dispatch: unknown buffer ${id}`);
            entries.push({ binding: i, resource: { buffer } });
          }
          if (uniformLen > 0) {
            // Per-dispatch uniform buffer (pooling is the FR-EX-05
            // follow-up recorded in context.rs).
            const uni = new Uint8Array(uniformLen);
            uni.set(body.subarray(4 + count * 4, 4 + count * 4 + uniformLen));
            const ubuf = device.createBuffer({
              size: Math.max(16, Math.ceil(uniformLen / 16) * 16),
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(ubuf, 0, uni);
            entries.push({ binding: count, resource: { buffer: ubuf } });
          }
          const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries,
          });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(wgX, wgY, wgZ);
          pass.end();
          device.queue.submit([enc.finish()]);
          respond(0);
          break;
        }
        default:
          respondError(`unknown vokra_webgpu command ${cmd}`);
      }
    } catch (e) {
      respondError(`${e}`);
    }
  }

  worker.addEventListener("message", (ev) => {
    if (ev.data && ev.data.vokraKick) {
      // The worker is blocked on Atomics.wait; single-flight by construction.
      void service();
    }
  });
}
