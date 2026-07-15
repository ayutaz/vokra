// @vokra/web — hand-written type declarations (M4-01-T20; no wasm-bindgen /
// type generators — ADR M4-01-webgpu-wasm §5).

/** Which artifact this host will load (WebAssembly.validate SIMD probe). */
export function selectArtifact(): "simd128" | "base";

/** Minimal RIFF PCM16 (16 kHz mono) → Float32Array decoder. */
export function wavToF32(bytes: Uint8Array): Float32Array;

export interface VokraSessionOptions {
  /**
   * Compute backend — an EXPLICIT choice, never a silent fallback
   * (FR-EX-08):
   * - `"cpu"` (default): the WASM CPU path (SIMD128 artifact when the host
   *   validates it, scalar otherwise).
   * - `"webgpu"`: requires cross-origin isolation (COOP/COEP headers — the
   *   SharedArrayBuffer bridge) AND a WebGPU adapter; missing either
   *   REJECTS with an explanatory error.
   */
  backend?: "cpu" | "webgpu";
  /** Overrides where the .wasm / worker assets resolve from
   * (default: this module's own URL). */
  baseUrl?: string | URL;
}

export interface VokraTranscription {
  /** The transcript text. */
  text: string;
  /** Wall-clock time of the transcribe call (ms, performance.now). */
  wallMs: number;
  /** Audio duration (ms, samples / 16 000). */
  audioMs: number;
  /** Real-time factor = wallMs / audioMs. */
  rtf: number;
}

export interface VokraSessionMeta {
  /** Which .wasm artifact was loaded. */
  artifact: "simd128" | "base";
  /** The backend this session was created with. */
  backend: "cpu" | "webgpu";
  /** Whether the loaded artifact was compiled with SIMD128. */
  simd128: boolean;
}

export declare class VokraSession {
  readonly meta: VokraSessionMeta;
  /**
   * Transcribes 16 kHz mono audio: a Float32Array of PCM samples, or a WAV
   * (RIFF PCM16 16 kHz mono) buffer.
   */
  transcribe(audio: Float32Array | Uint8Array | ArrayBuffer): Promise<VokraTranscription>;
  /** Releases the model. */
  close(): Promise<void>;
}

/**
 * Creates a Vokra ASR session from an in-memory Vokra whisper-base `.gguf`
 * (fetch the model yourself — models are never bundled in this package).
 */
export function createSession(
  modelBytes: ArrayBuffer | Uint8Array,
  options?: VokraSessionOptions,
): Promise<VokraSession>;
