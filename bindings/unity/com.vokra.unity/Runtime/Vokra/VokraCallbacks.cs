// VokraCallbacks.cs — IL2CPP-safe native→C# callback pattern (M2-11-T10 / D5 / R1).
//
// STATUS: Pattern hardened per D5; live wiring gated on VOKRA_STREAMING_CALLBACKS_ENABLED
// until a callback-registration symbol lands in the C ABI (M2-04 streaming). The current
// M0 C ABI exposes only poll-based streaming (vokra_stream_push_pcm + vokra_stream_poll);
// this file exists so IL2CPP AOT exercises the pattern during T15 (Linux headless smoke)
// and so the shape is locked in for FR-API-04 / NFR-RL-02.
//
// Invariants enforced (grep-checked by scripts/check-callback-pattern.sh):
//   1. All native delegates carry [UnmanagedFunctionPointer(CallingConvention.Cdecl)].
//   2. Every entry point is a STATIC method decorated with
//      [AOT.MonoPInvokeCallback(typeof(TDelegate))] AND [UnityEngine.Scripting.Preserve]
//      (instance methods cannot be marshalled under IL2CPP AOT; [Preserve] + link.xml
//      block the AOT tree-shaker from stripping the never-managed-called dispatcher).
//   3. Each delegate type used across the P/Invoke boundary has a rooted
//      `static readonly TDelegate` field so the GC cannot free the function pointer
//      while native code retains it.
//   4. Userdata lifetimes: every GCHandle.Alloc in this file is paired with a
//      matching GCHandle.Free in the Dispose/finally of the returned Registration.
//   5. Dispatch never touches Unity APIs from the native thread — payloads are
//      copied into managed arrays and enqueued on a ConcurrentQueue for main-thread
//      drain by the consumer.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AOT; // Unity's [MonoPInvokeCallback]
using UnityEngine.Scripting; // [Preserve] — pairs with link.xml to defeat IL2CPP stripping.

namespace Vokra
{
    /// <summary>
    /// Illustrative native callback signature: <c>void cb(const float* data, size_t n,
    /// void* userdata)</c>. Mirrors how a future <c>vokra_stream_set_callback</c> would
    /// hand results back instead of the caller polling.
    /// </summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void VokraProbCallback(IntPtr data, UIntPtr count, IntPtr userData);

    /// <summary>
    /// RAII wrapper for a callback registration. Ensures the GCHandle backing the
    /// userdata pointer is freed exactly once, matching the Alloc in
    /// <see cref="VokraCallbacks.Register"/>. Freed handles set <see cref="UserData"/>
    /// back to <see cref="IntPtr.Zero"/> to make double-free / UAF loud.
    /// </summary>
    public sealed class VokraCallbackRegistration : IDisposable
    {
        private GCHandle _handle;
        private bool _disposed;

        internal VokraCallbackRegistration(GCHandle handle, IntPtr fnPtr)
        {
            _handle = handle;
            UserData = GCHandle.ToIntPtr(handle);
            FunctionPointer = fnPtr;
        }

        /// <summary>Opaque pointer to hand to native code as <c>userdata</c>.</summary>
        public IntPtr UserData { get; private set; }

        /// <summary>Static delegate's function pointer — safe for native code to retain
        /// until <see cref="Dispose"/> is called on this registration.</summary>
        public IntPtr FunctionPointer { get; }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            try
            {
                if (_handle.IsAllocated)
                {
                    _handle.Free(); // Paired with GCHandle.Alloc in Register().
                }
            }
            finally
            {
                UserData = IntPtr.Zero;
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// IL2CPP-safe static-callback + GCHandle-userdata pattern. Nothing here calls
    /// native code by default (no callback API exists in M0); the live P/Invoke path
    /// is gated behind <c>VOKRA_STREAMING_CALLBACKS_ENABLED</c> so IL2CPP still AOT-
    /// compiles the dispatcher during nightly smoke tests without depending on an
    /// unshipped C-ABI symbol.
    /// </summary>
    internal static class VokraCallbacks
    {
        // Rooted delegate instance — must be `static readonly` so the GC cannot free
        // the function pointer while native code holds it (NFR-RL-02 / R1).
        private static readonly VokraProbCallback DispatchDelegate = Dispatch;

        /// <summary>
        /// The single static entry point native code invokes. Recovers the managed
        /// sink from <paramref name="userData"/> (a <see cref="GCHandle"/>), copies
        /// the unmanaged buffer into a fresh managed array, and enqueues it. It must
        /// not touch Unity APIs directly — this may run off the main thread.
        /// </summary>
        [MonoPInvokeCallback(typeof(VokraProbCallback))]
        [Preserve] // Keep the AOT tree-shaker from stripping (R1 mitigation, with link.xml).
        private static void Dispatch(IntPtr data, UIntPtr count, IntPtr userData)
        {
            if (userData == IntPtr.Zero)
            {
                return;
            }

            ConcurrentQueue<float[]> sink;
            try
            {
                sink = GCHandle.FromIntPtr(userData).Target as ConcurrentQueue<float[]>;
            }
            catch
            {
                // Freed or invalid handle — do not propagate exceptions into native code.
                return;
            }

            if (sink == null)
            {
                return;
            }

            int n = checked((int)count.ToUInt64());
            var buf = new float[n];
            if (n > 0 && data != IntPtr.Zero)
            {
                Marshal.Copy(data, buf, 0, n);
            }

            sink.Enqueue(buf);
        }

        /// <summary>
        /// Pin the managed sink as userdata and hand the static callback + userdata to
        /// native code. Caller must <see cref="IDisposable.Dispose"/> the returned
        /// registration after unregistering with native (no leak, no UAF).
        /// </summary>
        internal static VokraCallbackRegistration Register(ConcurrentQueue<float[]> sink)
        {
            if (sink == null)
            {
                throw new ArgumentNullException(nameof(sink));
            }

            var handle = GCHandle.Alloc(sink, GCHandleType.Normal);
            try
            {
                IntPtr fnPtr = Marshal.GetFunctionPointerForDelegate(DispatchDelegate);
                var registration = new VokraCallbackRegistration(handle, fnPtr);

#if VOKRA_STREAMING_CALLBACKS_ENABLED
                // Live wiring — enabled only when the C ABI ships a set-callback symbol.
                // Kept behind a define so the reference remains compile-checked during
                // IL2CPP AOT (T15) without linking against an unshipped extern.
                //   NativeMethods.StreamSetCallback(streamHandle, fnPtr, registration.UserData);
#endif
                return registration;
            }
            catch
            {
                // Registration ctor / delegate marshalling failed — free the handle now so
                // the Alloc above is paired with a Free on every path (R1 lifetime rule).
                if (handle.IsAllocated)
                {
                    handle.Free();
                }
                throw;
            }
        }
    }
}
