// NativeMethods.cs — P/Invoke declarations for the Vokra C ABI (M2-11-T09).
//
// Promoted from examples/unity-demo/Assets/Scripts/Vokra/NativeMethods.cs
// (M0-10-T03) into the com.vokra.unity UPM package Runtime asmdef.
//
// The single source of truth for every signature below is the cbindgen-generated
// header include/vokra.h. Do not invent signatures here; mirror the header
// exactly. Symbol ↔ C# mapping table:
//
//   vokra.h (C)                                                NativeMethods (C#)
//   --------------------------------------------------------   ----------------------
//   vokra_session_create_from_file(char*, session_t**)     ->  SessionCreateFromFile
//   vokra_session_create_from_bytes(uint8_t*,size_t,session_t**) -> SessionCreateFromBytes (M4-02)
//   vokra_session_retain(const session_t*, session_t**)    ->  SessionRetain          (M2-11-T09)
//   vokra_session_destroy(session_t*)                      ->  SessionDestroy
//   vokra_asr_transcribe(session*,float*,size_t,i32,char**) -> AsrTranscribe
//   vokra_string_free(char*)                               ->  StringFree
//   vokra_last_error() -> const char*                      ->  LastError
//   vokra_version()    -> const char*                      ->  Version
//   vokra_stream_open(session*,i32,stream_t**)             ->  StreamOpen
//   vokra_stream_push_pcm(stream*,float*,size_t)           ->  StreamPushPcm
//   vokra_stream_poll(stream*,float*,size_t,size_t*)       ->  StreamPoll
//   vokra_stream_poll_events(stream*,event_t*,size_t,size_t*) -> StreamPollEvents     (M2-11-T09)
//   vokra_stream_destroy(stream*)                          ->  StreamDestroy
//   vokra_tts_synthesize(session*,char*,float**,size_t*,i32*) -> TtsSynthesize
//   vokra_audio_free(float*,size_t)                        ->  AudioFree
//
// Marshalling policy:
//   - CallingConvention.Cdecl (the C ABI is `extern "C"`).
//   - size_t  <-> UIntPtr (correct on 32- and 64-bit; desktop targets are 64-bit).
//   - int32_t <-> int, the vokra_status_t enum <-> VokraStatus (int-sized).
//   - opaque handles (vokra_session_t* / vokra_stream_t*) <-> IntPtr.
//   - UTF-8 in (path/text): passed as an explicit NUL-terminated byte[] (see
//     Native.Utf8) so we never depend on the default `string` marshalling /
//     process locale (NFR-RL-01 locale trap).
//   - `const char*` returns (LastError/Version) are Vokra-owned or static, so we
//     receive them as IntPtr and never let the marshaller free them.
//   - blittable float[] parameters are pinned by the marshaller for the duration
//     of the call; [In]/[Out] documents direction.
//   - vokra_event_t is a fixed 12-byte POD ({int kind; uint32 a; float b}),
//     mapped to VokraEvent below (Sequential, Pack=4) — layout is verified by
//     the numeric-layout test in vokra-capi.
//
// Platform-conditional library name (D4, ADR-0007, NFR-RL-03 + M4-02):
//   - iOS device / TestFlight / App Store builds link the C ABI as a static
//     library (`libvokra.a`) and must resolve symbols via the special
//     "__Internal" name — dlopen of a custom dylib is forbidden.
//   - WebGL builds are the same shape (ADR M4-02 §1): the Emscripten-target
//     `Plugins/WebGL/libvokra.a` is statically linked into the single wasm
//     module by Unity's Emscripten, so symbols also resolve via "__Internal"
//     (dynamic library loading does not exist on the Web).
//   - Every other target (macOS/Linux/Windows/Android + Editor on any host)
//     resolves the shared library `vokra` (Unity strips the platform
//     prefix/suffix: libvokra.dylib / libvokra.so / vokra.dll / arm64-v8a
//     libvokra.so).
//   - Every [DllImport] below MUST use NativeMethods.Lib; no hardcoded
//     "vokra"/"__Internal" literal is allowed. Enforced by
//     scripts/check-native-methods.sh (3-state assert: iOS+WebGL / default).

using System;
using System.Runtime.InteropServices;

namespace Vokra
{
    /// <summary>
    /// Status codes returned by the fallible Vokra C functions. Numeric values
    /// mirror <c>vokra_status_t</c> in include/vokra.h (ADR-0003 §3-d) and are
    /// part of the (M0-unstable) ABI.
    /// </summary>
    public enum VokraStatus
    {
        Ok = 0,
        Io = 1,
        ModelLoad = 2,
        UnsupportedOp = 3,
        BackendUnavailable = 4,
        InvalidArgument = 5,
        GraphValidation = 6,
        NotImplemented = 7,
        Panic = 8,
        Other = 9,
    }

    /// <summary>
    /// Kind tag of a <see cref="VokraEvent"/>. Mirrors <c>vokra_event_kind_t</c>
    /// in include/vokra.h; numeric values are part of the (M0-unstable) ABI.
    /// </summary>
    public enum VokraEventKind
    {
        Unknown = 0,
        SpeechProb = 1,
        Token = 2,
    }

    /// <summary>
    /// A generalized streaming event drained by <c>vokra_stream_poll_events</c>.
    /// Mirrors <c>vokra_event_t</c> (a 12-byte POD) in include/vokra.h — the
    /// meaning of <see cref="A"/> and <see cref="B"/> depends on <see cref="Kind"/>.
    /// Sequential layout with Pack=4 pins the exact 12-byte footprint (int32 +
    /// uint32 + float32).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct VokraEvent
    {
        public VokraEventKind Kind;
        public uint A;
        public float B;
    }

    internal static class NativeMethods
    {
        // Platform-conditional native library name (D4, ADR-0007, NFR-RL-03,
        // M4-02).
        //
        // - iOS device builds statically link libvokra.a and resolve symbols via
        //   "__Internal" (dlopen of a custom dylib is forbidden by the App Store
        //   review guidelines).
        // - WebGL builds statically link the Emscripten-target libvokra.a into
        //   the single wasm module (there is no dynamic loading on the Web), so
        //   they use the same "__Internal" resolution (ADR M4-02 §1).
        // - Every other build (Editor on any host, desktop players, Android)
        //   resolves the shared library "vokra": Unity's PluginImporter strips
        //   the platform prefix/suffix (lib*.dylib / lib*.so / *.dll).
        //
        // Every [DllImport] below MUST use this constant so there is a single
        // switch point per platform; scripts/check-native-methods.sh greps for
        // any hardcoded "vokra" / "__Internal" literal in a DllImport and fails
        // the build if one appears.
#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        internal const string Lib = "__Internal";
#else
        internal const string Lib = "vokra";
#endif
        private const CallingConvention Cc = CallingConvention.Cdecl;

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_session_create_from_file")]
        internal static extern VokraStatus SessionCreateFromFile(
            [In] byte[] pathUtf8, out IntPtr outSession);

        // M4-02: bytes-based session create — the WebGL model path (and a
        // general-purpose alternative everywhere). The buffer is copied by
        // the native side before the call returns; the marshaller pins the
        // byte[] for the duration of the call only.
        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_session_create_from_bytes")]
        internal static extern VokraStatus SessionCreateFromBytes(
            [In] byte[] data, UIntPtr len, out IntPtr outSession);

        // FR-API-03: atomic ref count. Cheap clone of the inner Session (Arc bump);
        // the model is freed only when the last handle is destroyed. The new
        // handle is safe to move to another thread (Session: Send + Sync).
        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_session_retain")]
        internal static extern VokraStatus SessionRetain(
            IntPtr session, out IntPtr outSession);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_session_destroy")]
        internal static extern void SessionDestroy(IntPtr session);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_asr_transcribe")]
        internal static extern VokraStatus AsrTranscribe(
            IntPtr session,
            [In] float[] pcm,
            UIntPtr numSamples,
            int sampleRate,
            out IntPtr outTextUtf8);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_string_free")]
        internal static extern void StringFree(IntPtr s);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_last_error")]
        internal static extern IntPtr LastError();

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_version")]
        internal static extern IntPtr Version();

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_stream_open")]
        internal static extern VokraStatus StreamOpen(
            IntPtr session, int sampleRate, out IntPtr outStream);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_stream_push_pcm")]
        internal static extern VokraStatus StreamPushPcm(
            IntPtr stream, [In] float[] pcm, UIntPtr numSamples);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_stream_poll")]
        internal static extern VokraStatus StreamPoll(
            IntPtr stream,
            [Out] float[] outProbs,
            UIntPtr capacity,
            out UIntPtr outCount);

        // Generalized poll drained as typed VokraEvent structs (M1-08 event ring).
        // vokra_stream_poll (above) is the f32-probability fast path over the
        // same ring; both are non-blocking.
        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_stream_poll_events")]
        internal static extern VokraStatus StreamPollEvents(
            IntPtr stream,
            [Out] VokraEvent[] outEvents,
            UIntPtr capacity,
            out UIntPtr outCount);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_stream_destroy")]
        internal static extern void StreamDestroy(IntPtr stream);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_tts_synthesize")]
        internal static extern VokraStatus TtsSynthesize(
            IntPtr session,
            [In] byte[] textUtf8,
            out IntPtr outPcm,
            out UIntPtr outNumSamples,
            out int outSampleRate);

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_audio_free")]
        internal static extern void AudioFree(IntPtr pcm, UIntPtr numSamples);
    }
}
