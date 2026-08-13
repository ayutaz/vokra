# vokra-android — Kotlin/Android JNI binding (Proposed ADR scaffold)

Out-of-workspace integration crate that packages a Kotlin/Android JNI
binding for the Vokra speech-first runtime. Landed 2026-08-14 as the CC
implementation branch of `docs/adr/M4-kotlin-binding-jni-vs-jna.md`
(**Proposed** — owner sign-off queued on M4-11 T13 gap-flow).

**ADR status**: this scaffold implements branch (B) raw-JNI. If the owner
approves branch (A) JNA at ADR §7 sign-off, the Kotlin sources under
`kotlin/com/vokra/` become docs-only demonstrations of the raw-JNI
baseline the ADR compares against; the Kotlin runtime API changes to
declare a `com.sun.jna.Library` and calls the stock `libvokra.so`
directly instead of `libvokra_android.so`.

## Layout

```
integrations/vokra-android/
├── Cargo.toml                        # excluded workspace root (own Cargo.lock)
├── README.md                         # this file
├── src/
│   ├── lib.rs                        # 5 JNI trampolines + JNI_OnLoad + panic firewall
│   ├── capi.rs                       # extern "C" for vokra_* symbols (3 fn subset)
│   └── jni.rs                        # hand-declared JNI 1.6 ABI (no jni / jni-sys crate)
├── kotlin/
│   └── com/vokra/
│       ├── VokraSession.kt           # AutoCloseable handle + companion factory
│       └── VokraException.kt         # thread-local errno → Kotlin exception
├── templates/
│   ├── AndroidManifest.xml.template  # permissions + <application> block
│   └── build.gradle.kts.template     # Gradle + cargo-ndk wiring
└── (jniLibs/ — produced by `cargo ndk`, not tracked)
```

## Isolation from the Vokra root workspace

Same isolation pattern as `integrations/vokra-godot/`,
`integrations/vokra-piper-g2p/`, `integrations/vokra-server/` — this
directory is its **own** workspace (empty `[workspace]` table in
`Cargo.toml`) with its own `Cargo.lock`. The zero-dependency invariant
on the root `Cargo.lock` (NFR-DS-02, enforced by
`scripts/check-zero-deps.sh`) is untouched. The root workspace's
`[workspace.exclude] = ["integrations"]` glob already skips this
directory; no root `Cargo.toml` edit is needed to add new crates under
`integrations/`.

## Scope (this landing wave)

Minimal Session lifecycle surface only — 5 JNI trampolines:

| Java signature | C ABI call | Purpose |
|---|---|---|
| `nativeContextNew()` | (reserved) | Returns a synthetic non-zero handle for a future real-context ADR change. |
| `nativeContextFree(handle)` | (reserved) | No-op paired free. |
| `nativeSessionCreate(path)` | `vokra_session_create_from_file` | Loads a GGUF and returns a session handle. |
| `nativeSessionFree(ctx, handle)` | `vokra_session_destroy` | Releases the session; NULL-safe. |
| `nativeGetLastError()` | `vokra_last_error` | Thread-local errno read-back. |

**Rolling follow-ups** (deferred to a separate WP per ADR §7 "後続実装 WP
の起票"): ASR / TTS / VAD / streaming / AEC / S2S wrappers, an
`AssetManager` → `Context.filesDir` helper (NFR-RL-04), Kotlin
coroutine wrappers (`Dispatchers.IO`), and Maven Central publish CD.

## Build steps

Vokra Android JNI binding builds with `cargo-ndk`
(<https://github.com/bbqsrc/cargo-ndk>) — Google's official NDK
integration for Cargo.

### One-time setup

```bash
cargo install cargo-ndk --locked
rustup target add aarch64-linux-android armv7-linux-androideabi \
                  x86_64-linux-android i686-linux-android
# Install Android NDK r26+ (any recent SDK Manager entry works). Point
# ANDROID_NDK_HOME at it, or drop a line into ~/.cargo/config.toml.
```

### Cross-compile the JNI cdylib

```bash
cd integrations/vokra-android
cargo ndk -t arm64-v8a -t armeabi-v7a -o jniLibs build --release
```

- `-t arm64-v8a` — Android 64-bit ARM (99%+ market coverage in 2026).
- `-t armeabi-v7a` — legacy 32-bit ARM (drop if you don't ship to
  pre-Android 10 devices).
- `-o jniLibs` — writes `<abi>/libvokra_android.so` into a folder that
  Android Gradle Plugin picks up when placed at `src/main/jniLibs/`.
- `--release` — hits the size-optimised `[profile.release]` in the
  workspace root (LTO + strip + single codegen unit; see the root
  `Cargo.toml` §Build profiles). Runtime `panic = "unwind"` is
  preserved for the FFI panic firewall (see ADR-0003 §4).

Copy `jniLibs/<abi>/libvokra_android.so` and the `kotlin/com/vokra/*`
sources into your Android app project — see
`templates/build.gradle.kts.template` for the automated Gradle wiring.

### Off-device unit tests (host-side)

The `#[cfg(test)]` blocks in `src/lib.rs`, `src/jni.rs`, and
`src/capi.rs` are safe to run without a JVM — they exercise the
NULL-env branches and the reserved-context trampolines. To run them:

```bash
cd integrations/vokra-android
CARGO_BUILD_JOBS=1 cargo check
CARGO_BUILD_JOBS=1 cargo test -p vokra-android
```

Both commands stay inside the isolated workspace and do NOT touch the
root `Cargo.lock` (verified with `git status` — no root-tree diff).

## ABI selection guidance

| ABI | 2026 market share | Add to `abiFilters`? |
|---|---|---|
| `arm64-v8a` | ~99% of active devices | **Always** |
| `armeabi-v7a` | <1% (pre-Android 10 low-end) | Only if your app targets those SKUs |
| `x86_64` | Emulator | For CI on hosted emulators |
| `x86` | Very rare | Skip unless legacy |

Every ABI you enable roughly doubles the `.so` cost in your APK
(strip-only, no shrink): budget ~10 MB per ABI for the CPU-only Vokra
runtime. GGUF model files (Whisper base ~150 MB, Kokoro ~80 MB, ...)
dominate over binary size in real deployments.

## Not in this crate

- **Maven Central publishing** — owner-owned credentials (`MAVEN_CENTRAL_USERNAME`,
  `MAVEN_CENTRAL_PASSWORD`, GPG signing key). CI wiring lands after ADR
  sign-off (M4-11 T13 gap-flow decision).
- **AAR packaging** — the raw-JNI branch (B) does not itself produce an
  AAR; consumers integrate as source (`kotlin/com/vokra/`) + prebuilt
  `.so`. Branch (A) JNA is the AAR-natural path.
- **RVC / GPT-SoVITS / voice-cloning bindings** — deliberately absent per
  CLAUDE.md §8 (ELVIS Act separation); those live in the
  `vokra-voiceclone-experimental` separate repo, never in this main
  repo binding.
- **NNAPI delegate wiring** — Google-deprecated in Android 15 (2024-10);
  Android GPU work goes through Vulkan (M3-02) or CPU. See CLAUDE.md
  §6 "なぜ NNAPI に対応しないか".

## Related documents

- `docs/adr/M4-kotlin-binding-jni-vs-jna.md` — the Proposed ADR gating
  the JNI vs JNA decision (owner sign-off queued).
- `docs/adr/ADR-00xx-language-binding-conventions.md` — cross-language
  FFI contracts (handle ownership, error variant surface, thread rules,
  buffer ownership, locale-independence).
- `docs/adr/0003-c-abi-design.md` — Vokra C ABI (`include/vokra.h`,
  cbindgen; opaque handles + thread-local errno + poll-driven streams).
- `docs/m3-18-android-rtf-handover.md` — the one-shot Android RTF
  scaffold that this permanent binding supersedes.
- `docs/tickets/m2/M2-12-language-bindings.md` — the M2-12 ticket that
  originally deferred Kotlin/Swift/JS to a follow-up rolling wave (Python
  landed in M2-12).
- `integrations/vokra-godot/` — sister integration with the same
  isolated-workspace + hand-written FFI pattern for Godot 4.x.
