// VokraSession.cs — high-level wrapper over a Vokra C ABI session (M0-10-T04).
//
// A session is one loaded GGUF model with its matching native engine (ASR / TTS /
// VAD) injected by the C ABI from the `vokra.model.arch` metadata. This thin layer
// exposes just what the demo pipeline (T07) needs and keeps every raw IntPtr and
// status code behind SafeHandle + VokraException.
//
// Thread-safety: FR-API-03 (Session is Send + Sync, atomic ref count) is v0.1 MVP
// scope. This demo does NOT rely on it: all methods here are called from a single
// worker thread (T07), which also satisfies the thread-local vokra_last_error
// contract.

using System;
using System.Runtime.InteropServices;

namespace Vokra
{
    public sealed class VokraSession : IDisposable
    {
        private readonly VokraSessionHandle _handle;

        private VokraSession(VokraSessionHandle handle)
        {
            _handle = handle;
        }

        /// <summary>The Vokra runtime version string (<c>vokra_version</c>).</summary>
        public static string RuntimeVersion => Native.PtrToString(NativeMethods.Version()) ?? "unknown";

        /// <summary>
        /// Loads a GGUF model and creates a CPU-backed session. The task (ASR / TTS /
        /// VAD) is chosen from the model's <c>vokra.model.arch</c> metadata.
        /// </summary>
        public static VokraSession CreateFromFile(string ggufPath)
        {
            if (string.IsNullOrEmpty(ggufPath))
            {
                throw new ArgumentException("model path is empty", nameof(ggufPath));
            }

            VokraStatus status = NativeMethods.SessionCreateFromFile(Native.Utf8(ggufPath), out IntPtr raw);
            VokraException.ThrowIfError(status, $"load model '{ggufPath}'");
            if (raw == IntPtr.Zero)
            {
                throw new VokraException(VokraStatus.Other, "session handle was NULL on success");
            }

            return new VokraSession(VokraSessionHandle.FromRaw(raw));
        }

        /// <summary>
        /// Transcribes mono f32 PCM. <paramref name="sampleRate"/> must equal the
        /// model's front-end rate — Vokra does not resample in M0 (FR-OP-04 is M1).
        /// </summary>
        public string Transcribe(float[] pcm, int sampleRate)
        {
            pcm ??= Array.Empty<float>();
            // DangerousGetHandle is safe here: the handle outlives the synchronous
            // call and is only Disposed later on this same worker thread (T07).
            VokraStatus status = NativeMethods.AsrTranscribe(
                _handle.DangerousGetHandle(), pcm, (UIntPtr)(uint)pcm.Length, sampleRate, out IntPtr outText);
            VokraException.ThrowIfError(status, "ASR transcribe");
            try
            {
                return Native.PtrToString(outText) ?? string.Empty;
            }
            finally
            {
                NativeMethods.StringFree(outText);
            }
        }

        /// <summary>
        /// Synthesizes speech PCM from UTF-8 text. Returns the mono f32 samples (in
        /// [-1, 1]) and the model's output sample rate.
        /// </summary>
        public (float[] pcm, int sampleRate) Synthesize(string text)
        {
            VokraStatus status = NativeMethods.TtsSynthesize(
                _handle.DangerousGetHandle(), Native.Utf8(text ?? string.Empty),
                out IntPtr outPcm, out UIntPtr outNum, out int outRate);
            VokraException.ThrowIfError(status, "TTS synthesize");
            try
            {
                int n = checked((int)outNum.ToUInt64());
                var buf = new float[n];
                if (n > 0)
                {
                    Marshal.Copy(outPcm, buf, 0, n);
                }

                return (buf, outRate);
            }
            finally
            {
                NativeMethods.AudioFree(outPcm, outNum);
            }
        }

        /// <summary>
        /// Opens a Silero VAD stream at <paramref name="sampleRate"/> Hz (8000 or
        /// 16000). Fails with <see cref="VokraStatus.NotImplemented"/> if the model
        /// is not a VAD model.
        /// </summary>
        public VokraStream OpenVadStream(int sampleRate)
        {
            VokraStatus status = NativeMethods.StreamOpen(_handle.DangerousGetHandle(), sampleRate, out IntPtr raw);
            VokraException.ThrowIfError(status, "open VAD stream");
            if (raw == IntPtr.Zero)
            {
                throw new VokraException(VokraStatus.Other, "stream handle was NULL on success");
            }

            return new VokraStream(VokraStreamHandle.FromRaw(raw));
        }

        public void Dispose()
        {
            _handle.Dispose();
        }
    }
}
