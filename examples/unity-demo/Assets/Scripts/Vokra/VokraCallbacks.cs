// VokraCallbacks.cs — IL2CPP-safe native→C# callback pattern (M0-10-T05).
//
// STATUS: TEMPLATE / NOT WIRED IN M0.
// The M0 C ABI (include/vokra.h) exposes NO callback-registration symbol — the
// streaming surface is POLL-based (vokra_stream_push_pcm + vokra_stream_poll,
// used by PipelineRunner). There is therefore nothing to bind a callback to yet.
//
// This file exists to lock in the *correct* pattern for any push/callback API a
// future C ABI adds, and to keep the demo honest about NFR-RL-02: under IL2CPP,
// C# CLOSURES cannot be marshalled to native code. The only safe shape is:
//   1. a `static` method decorated with [AOT.MonoPInvokeCallback(typeof(TDelegate))],
//   2. context passed as an opaque userdata pointer via GCHandle (never a captured
//      lambda / instance delegate),
//   3. the delegate instance kept alive in a static field so the GC cannot collect
//      it while native code holds the function pointer.
// When the C ABI grows such an API, wire it here and delete this notice. Until
// then this is a compile-checked reference, carried forward to the v0.5 official
// plugin (FR-API-04).

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using AOT; // Unity's [MonoPInvokeCallback]

namespace Vokra
{
    /// <summary>
    /// Illustrative native callback signature: <c>void cb(const float* data, size_t
    /// n, void* userdata)</c>. Mirrors how a future <c>vokra_stream_set_callback</c>
    /// would hand results back instead of the caller polling. Not referenced by any
    /// current DllImport (see the file header).
    /// </summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void VokraProbCallback(IntPtr data, UIntPtr count, IntPtr userData);

    /// <summary>
    /// Reference implementation of the IL2CPP-safe static-callback + GCHandle-userdata
    /// pattern. Nothing here calls native code (no callback API exists in M0); it is a
    /// compile-checked template.
    /// </summary>
    internal static class VokraCallbacks
    {
        // Keep the delegate instance rooted so the GC never frees the function
        // pointer handed to native code (NFR-RL-02).
        private static readonly VokraProbCallback DispatchDelegate = Dispatch;

        /// <summary>
        /// The single static entry point native code would call. It recovers the
        /// managed context from <paramref name="userData"/> (a GCHandle) and enqueues
        /// the payload — it must not touch Unity APIs directly (it may run off the
        /// main thread); the consumer drains the queue on the main thread.
        /// </summary>
        [MonoPInvokeCallback(typeof(VokraProbCallback))]
        private static void Dispatch(IntPtr data, UIntPtr count, IntPtr userData)
        {
            if (userData == IntPtr.Zero)
            {
                return;
            }

            var sink = GCHandle.FromIntPtr(userData).Target as ConcurrentQueue<float[]>;
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
        /// Demonstrates registration: pin a managed sink as userdata and hand the
        /// static callback + userdata to native code. Returns the GCHandle the caller
        /// must <see cref="GCHandle.Free"/> after unregistering (leak/UAF guard).
        /// </summary>
        /// <remarks>
        /// The commented call is the shape a real API would take; it is intentionally
        /// absent because no such symbol exists in the M0 vokra.h.
        /// </remarks>
        internal static GCHandle Register(ConcurrentQueue<float[]> sink)
        {
            var handle = GCHandle.Alloc(sink);
            IntPtr userData = GCHandle.ToIntPtr(handle);
            IntPtr fnPtr = Marshal.GetFunctionPointerForDelegate(DispatchDelegate);

            // v0.5: NativeMethods.StreamSetCallback(streamHandle, fnPtr, userData);
            _ = fnPtr;
            _ = userData;
            return handle;
        }
    }
}
