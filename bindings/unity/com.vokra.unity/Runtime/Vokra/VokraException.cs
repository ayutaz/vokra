// VokraException.cs — single error path for the C ABI (M2-11-T09).
//
// Promoted from examples/unity-demo/Assets/Scripts/Vokra/VokraException.cs
// (M0-10-T04) into the com.vokra.unity UPM package Runtime asmdef.
//
// Every C ABI call returns a vokra_status_t. On any non-OK status the wrappers
// read vokra_last_error() (thread-local errno, FR-API-01) and throw a
// VokraException carrying both the status code and the detail message. This
// funnels the whole surface into one exception type so caller code never
// inspects raw status codes.

using System;

namespace Vokra
{
    /// <summary>An error returned by the Vokra C ABI.</summary>
    public sealed class VokraException : Exception
    {
        /// <summary>The <c>vokra_status_t</c> code that produced this exception.</summary>
        public VokraStatus Status { get; }

        public VokraException(VokraStatus status, string message)
            : base(message)
        {
            Status = status;
        }

        /// <summary>
        /// Throws a <see cref="VokraException"/> if <paramref name="status"/> is not
        /// <see cref="VokraStatus.Ok"/>, attaching <c>vokra_last_error()</c>.
        /// Call on the same thread as the failing C ABI call (thread-local errno).
        /// </summary>
        internal static void ThrowIfError(VokraStatus status, string context)
        {
            if (status == VokraStatus.Ok)
            {
                return;
            }

            string detail = Native.LastError() ?? "(no error message)";
            throw new VokraException(status, $"{context} failed [{status}]: {detail}");
        }
    }
}
