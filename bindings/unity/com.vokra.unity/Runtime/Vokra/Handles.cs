// Handles.cs — SafeHandle wrappers for the opaque C ABI handles (M2-11-T09).
//
// Promoted from examples/unity-demo/Assets/Scripts/Vokra/Handles.cs (M0-10-T04)
// into the com.vokra.unity UPM package Runtime asmdef.
//
// Wrapping vokra_session_t* / vokra_stream_t* in SafeHandle subclasses gives the
// runtime deterministic, finalizer-backed release (the matching vokra_*_destroy)
// and guards against double-free / leaks. Caller code only ever sees the
// high-level VokraSession / VokraStream — a raw IntPtr never escapes this layer.

using System;
using System.Runtime.InteropServices;

namespace Vokra
{
    /// <summary>Owns a <c>vokra_session_t*</c>; releases it with <c>vokra_session_destroy</c>.</summary>
    internal sealed class VokraSessionHandle : SafeHandle
    {
        private VokraSessionHandle() : base(IntPtr.Zero, ownsHandle: true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        internal static VokraSessionHandle FromRaw(IntPtr raw)
        {
            var h = new VokraSessionHandle();
            h.SetHandle(raw);
            return h;
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.SessionDestroy(handle); // NULL is a no-op per vokra.h
            return true;
        }
    }

    /// <summary>Owns a <c>vokra_stream_t*</c>; releases it with <c>vokra_stream_destroy</c>.</summary>
    internal sealed class VokraStreamHandle : SafeHandle
    {
        private VokraStreamHandle() : base(IntPtr.Zero, ownsHandle: true) { }

        public override bool IsInvalid => handle == IntPtr.Zero;

        internal static VokraStreamHandle FromRaw(IntPtr raw)
        {
            var h = new VokraStreamHandle();
            h.SetHandle(raw);
            return h;
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.StreamDestroy(handle); // NULL is a no-op per vokra.h
            return true;
        }
    }
}
