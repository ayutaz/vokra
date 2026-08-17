/*
 * VokraSession.kt — Kotlin/JVM entry point for the Vokra Android JNI binding.
 *
 * Copyright The Vokra Authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See docs/adr/M4-kotlin-binding-jni-vs-jna.md (Proposed) for the ADR that
 * gates the JNI vs JNA choice. This file implements the raw-JNI branch (B).
 *
 * SURFACE (scaffold, 2026-08-14):
 *   - VokraSession.create(path)  — wraps vokra_session_create_from_file
 *   - VokraSession.close()       — wraps vokra_session_destroy (AutoCloseable)
 *   - Vokra.lastError()          — wraps vokra_last_error (thread-local)
 *
 * ROLLING FOLLOW-UPS (out of scope for this landing wave, per ADR §7):
 *   - VokraSession.transcribe(pcm, sampleRate)  → vokra_asr_transcribe
 *   - VokraSession.synthesize(text)             → vokra_tts_synthesize
 *   - VokraSession.openStream(sampleRate)       → vokra_stream_open
 *   - Vokra.assetToFile(context, name)          → AssetManager → filesDir
 *     (NFR-RL-04 Android StreamingAssets helper)
 *   - Kotlin coroutines wrappers on Dispatchers.IO
 */
package com.vokra

/**
 * The `AutoCloseable` handle to a loaded Vokra model. Instances are created
 * with [Vokra.createSession] and MUST be `close()`d (or used in `use { ... }`)
 * to release the underlying native memory.
 *
 * Handles are backed by the atomic-refcounted `vokra_session_t` C ABI type —
 * safe to share across threads once created (`Session: Send + Sync`, see
 * `include/vokra.h` L124). Do NOT construct one directly; use
 * [Vokra.createSession].
 */
class VokraSession internal constructor(
    /**
     * Opaque native handle (a `*mut vokra_session_t` cast to `jlong`). Zero
     * indicates an already-freed handle; every method guards against that.
     */
    @Volatile private var handle: Long,
) : AutoCloseable {

    /**
     * The pointer value, exposed for the rolling-wave ASR / TTS / VAD wrappers
     * that will land in the follow-up WP (ADR §7 "後続実装 WP の起票"). The
     * value MUST NOT outlive `this`; treat it as a borrow, not an ownership
     * transfer.
     */
    val nativeHandle: Long
        get() = handle

    /**
     * Whether this handle has already been freed. `close()` is idempotent, so
     * this becomes `true` after the first close.
     */
    val isClosed: Boolean
        get() = handle == 0L

    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            nativeSessionFree(reservedContext, h)
        }
    }

    /**
     * Best-effort finalizer safety net — `close()` remains the required API.
     * The JVM may or may not call this before process exit; treating it as
     * a hard cleanup guarantee is unsafe.
     */
    @Suppress("removal", "deprecation")
    protected fun finalize() {
        close()
    }

    companion object {
        // Reserved context handle — see docs on `nativeContextNew`. Held here
        // (not per-session) because Vokra has one global runtime; if a future
        // ADR change makes contexts real, this promotion path opens without
        // an API break.
        private val reservedContext: Long

        init {
            // Loads `libvokra_android.so`. On Android, `System.loadLibrary`
            // searches the APK's per-ABI `jniLibs/` folder (populated by
            // `cargo ndk -t <abi> -o jniLibs build --release` — see README.md
            // §Build steps). On desktop JVM (dev), it searches
            // `java.library.path`; drop the built `.dylib` / `.so` / `.dll`
            // in a directory on the path or set `-Djava.library.path=...`.
            System.loadLibrary("vokra_android")
            reservedContext = nativeContextNew()
        }

        /**
         * Loads a Vokra GGUF model from the filesystem and returns a session
         * bound to the CPU backend (`vokra_session_create_from_file`).
         *
         * On Android, the `path` should be an absolute path inside
         * `Context.filesDir` (or `Context.cacheDir`) — the runtime does not
         * read from APK `AssetManager` URLs (NFR-RL-04). Use the (planned)
         * `Vokra.assetToFile(context, "model.gguf")` helper to expand a
         * StreamingAsset into a real file before this call.
         *
         * @throws VokraException on failure. The exception message is the
         * thread-local `vokra_last_error()` at the moment of failure.
         */
        @JvmStatic
        fun create(path: String): VokraSession {
            val handle = nativeSessionCreate(path)
            if (handle == 0L) {
                throw VokraException(
                    Vokra.lastError() ?: "vokra_session_create_from_file returned NULL"
                )
            }
            return VokraSession(handle)
        }

        // --- Native trampolines. Symbols are exported by the Rust cdylib at
        // Java_com_vokra_VokraSession_<method>; keep the method names in sync
        // with `integrations/vokra-android/src/lib.rs`. -----------------------

        @JvmStatic private external fun nativeContextNew(): Long
        @JvmStatic private external fun nativeContextFree(handle: Long)
        @JvmStatic private external fun nativeSessionCreate(path: String): Long
        @JvmStatic private external fun nativeSessionFree(context: Long, handle: Long)
        @JvmStatic internal external fun nativeGetLastError(): String?
    }
}

/**
 * Package-level convenience API for the Vokra runtime — currently exposes
 * the thread-local error read-back so callers can inspect `vokra_last_error`
 * without going through a session handle (matches the C ABI thread-local
 * contract in `include/vokra.h` L305).
 */
object Vokra {
    /**
     * The calling thread's last Vokra error message, or `null` if no error is
     * recorded on this thread. The value is a JVM-owned copy of the runtime's
     * thread-local buffer at the moment of the call; safe to hold onto after
     * the call returns.
     */
    @JvmStatic
    fun lastError(): String? = VokraSession.nativeGetLastError()

    /**
     * Convenience factory forwarded to [VokraSession.create]; mirrors the
     * ADR-00xx-language-binding-conventions.md §1 "handle" contract that
     * every binding language exposes a top-level `Vokra.createSession`.
     */
    @JvmStatic
    fun createSession(path: String): VokraSession = VokraSession.create(path)
}
