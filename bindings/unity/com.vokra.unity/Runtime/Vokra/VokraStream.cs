// VokraStream.cs — high-level wrapper over a Vokra VAD stream (M2-11-T09).
//
// Promoted from examples/unity-demo/Assets/Scripts/Vokra/VokraStream.cs
// (M0-10-T04) into the com.vokra.unity UPM package Runtime asmdef.
//
// A stateful Silero VAD stream: push mono f32 PCM, drain per-frame speech
// probabilities (fast path) or typed events (VokraEvent[], M1-08 generalized
// ring). All recurrent state (LSTM h/c, framing) is hidden in the native handle
// (FR-LD-06).

using System;

namespace Vokra
{
    public sealed class VokraStream : IDisposable
    {
        private readonly VokraStreamHandle _handle;

        internal VokraStream(VokraStreamHandle handle)
        {
            _handle = handle;
        }

        /// <summary>Pushes mono f32 PCM; completed frames' probabilities are buffered for <see cref="Poll"/>.</summary>
        public void PushPcm(float[] pcm)
        {
            pcm ??= Array.Empty<float>();
            VokraStatus status = NativeMethods.StreamPushPcm(
                _handle.DangerousGetHandle(), pcm, (UIntPtr)(uint)pcm.Length);
            VokraException.ThrowIfError(status, "VAD push PCM");
        }

        /// <summary>
        /// Drains up to <paramref name="capacity"/> buffered speech probabilities.
        /// Returns an array sized to the number actually produced (may be empty).
        /// If it returns exactly <paramref name="capacity"/> items, call again —
        /// more may still be pending.
        /// </summary>
        public float[] Poll(int capacity = 256)
        {
            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity));
            }

            var buf = new float[capacity];
            VokraStatus status = NativeMethods.StreamPoll(
                _handle.DangerousGetHandle(), buf, (UIntPtr)(uint)capacity, out UIntPtr outCount);
            VokraException.ThrowIfError(status, "VAD poll");

            int n = checked((int)outCount.ToUInt64());
            if (n == buf.Length)
            {
                return buf;
            }

            var result = new float[n];
            Array.Copy(buf, result, n);
            return result;
        }

        /// <summary>Drains all currently-buffered probabilities by looping <see cref="Poll"/>.</summary>
        public System.Collections.Generic.List<float> PollAll(int chunk = 256)
        {
            var all = new System.Collections.Generic.List<float>();
            while (true)
            {
                float[] batch = Poll(chunk);
                all.AddRange(batch);
                if (batch.Length < chunk)
                {
                    break;
                }
            }

            return all;
        }

        /// <summary>
        /// Drains up to <paramref name="capacity"/> typed events (M1-08). The event
        /// ring is shared with <see cref="Poll"/>: prefer this overload when the
        /// caller needs frame indices or forward-compatibility with ASR token
        /// events on the same stream.
        /// </summary>
        public VokraEvent[] PollEvents(int capacity = 256)
        {
            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity));
            }

            var buf = new VokraEvent[capacity];
            VokraStatus status = NativeMethods.StreamPollEvents(
                _handle.DangerousGetHandle(), buf, (UIntPtr)(uint)capacity, out UIntPtr outCount);
            VokraException.ThrowIfError(status, "stream poll events");

            int n = checked((int)outCount.ToUInt64());
            if (n == buf.Length)
            {
                return buf;
            }

            var result = new VokraEvent[n];
            Array.Copy(buf, result, n);
            return result;
        }

        public void Dispose()
        {
            _handle.Dispose();
        }
    }
}
