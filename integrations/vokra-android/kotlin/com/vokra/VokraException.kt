/*
 * VokraException.kt — Kotlin exception type for Vokra JNI call failures.
 *
 * Copyright The Vokra Authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See ADR-00xx-language-binding-conventions.md §2 (error variant surface)
 * and ADR docs/adr/M4-kotlin-binding-jni-vs-jna.md.
 */
package com.vokra

/**
 * Thrown by [VokraSession] and (future) ASR / TTS / VAD wrappers when the
 * underlying `vokra-capi` call returns a non-zero `vokra_status_t` or an
 * out-parameter that the C ABI documents as "NULL on error".
 *
 * The [message] is the thread-local `vokra_last_error()` string at the moment
 * of failure, so it may be a rich multi-line diagnostic (the runtime formats
 * `VokraError` with `Display`). If Vokra had no error recorded on the calling
 * thread (which should not happen but defends against `nativeGetLastError`
 * returning `null` after a panic — see `catch_panic` in
 * `integrations/vokra-android/src/lib.rs`), a synthetic fallback message is
 * used.
 *
 * # ELVIS Act / voice-clone (CLAUDE.md §8)
 *
 * VC / RVC / GPT-SoVITS APIs are deliberately absent from this Kotlin
 * binding surface — they live in the `vokra-voiceclone-experimental`
 * separate repo. Loading a voice-clone GGUF through [VokraSession.create]
 * therefore fails with `VOKRA_ERROR_UNSUPPORTED_OP` (or a similar
 * loud-error status), which surfaces as a [VokraException] here — never
 * as a silent fallback (FR-EX-08).
 */
class VokraException(message: String) : RuntimeException(message)
