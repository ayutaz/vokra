// NativeMethods.cs — P/Invoke declarations for the Vokra C ABI (M0-10-T03).
//
// The single source of truth for every signature below is the cbindgen-generated
// header include/vokra.h (M0-09). Do not invent signatures here; mirror the
// header exactly. Symbol ↔ C# mapping table:
//
//   vokra.h (C)                                              NativeMethods (C#)
//   ------------------------------------------------------   ----------------------
//   vokra_session_create_from_file(char*, session_t**)   ->  SessionCreateFromFile
//   vokra_session_destroy(session_t*)                    ->  SessionDestroy
//   vokra_asr_transcribe(session*,float*,size_t,i32,char**) -> AsrTranscribe
//   vokra_string_free(char*)                             ->  StringFree
//   vokra_last_error() -> const char*                    ->  LastError
//   vokra_version()    -> const char*                    ->  Version
//   vokra_stream_open(session*,i32,stream_t**)           ->  StreamOpen
//   vokra_stream_push_pcm(stream*,float*,size_t)         ->  StreamPushPcm
//   vokra_stream_poll(stream*,float*,size_t,size_t*)     ->  StreamPoll
//   vokra_stream_destroy(stream*)                        ->  StreamDestroy
//   vokra_tts_synthesize(session*,char*,float**,size_t*,i32*) -> TtsSynthesize
//   vokra_audio_free(float*,size_t)                      ->  AudioFree
//
// Marshalling policy (M0-10-T03):
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

    internal static class NativeMethods
    {
        // Unified cdylib name (crates/vokra-capi Cargo.toml `[lib] name = "vokra"`):
        // libvokra.dylib / libvokra.so / vokra.dll. iOS links the staticlib via
        // DllImport("__Internal") (NFR-RL-03) — that is v0.5 official-plugin scope
        // (FR-API-04); this desktop demo uses the shared-library name only.
        internal const string Lib = "vokra";
        private const CallingConvention Cc = CallingConvention.Cdecl;

        [DllImport(Lib, CallingConvention = Cc, EntryPoint = "vokra_session_create_from_file")]
        internal static extern VokraStatus SessionCreateFromFile(
            [In] byte[] pathUtf8, out IntPtr outSession);

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
