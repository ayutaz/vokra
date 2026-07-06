// Native.cs — small marshalling helpers shared by the Vokra wrappers (M2-11-T09).
//
// Promoted from examples/unity-demo/Assets/Scripts/Vokra/Native.cs (M0-10-T03/T04)
// into the com.vokra.unity UPM package Runtime asmdef.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Vokra
{
    internal static class Native
    {
        /// <summary>
        /// Encodes <paramref name="s"/> as a NUL-terminated UTF-8 byte array for a
        /// C <c>const char*</c> parameter. Explicit encoding avoids any dependency
        /// on the default string marshalling or the process locale (NFR-RL-01).
        /// </summary>
        internal static byte[] Utf8(string s)
        {
            s ??= string.Empty;
            int len = Encoding.UTF8.GetByteCount(s);
            var buf = new byte[len + 1]; // + trailing NUL
            Encoding.UTF8.GetBytes(s, 0, s.Length, buf, 0);
            buf[len] = 0;
            return buf;
        }

        /// <summary>
        /// Reads a Vokra-owned NUL-terminated UTF-8 C string; never frees it.
        /// Decoded manually (not via Marshal.PtrToStringUTF8) so it works on every
        /// Unity Mono/IL2CPP profile regardless of BCL version.
        /// </summary>
        internal static string PtrToString(IntPtr utf8)
        {
            if (utf8 == IntPtr.Zero)
            {
                return null;
            }

            int len = 0;
            while (Marshal.ReadByte(utf8, len) != 0)
            {
                len++;
            }

            if (len == 0)
            {
                return string.Empty;
            }

            var bytes = new byte[len];
            Marshal.Copy(utf8, bytes, 0, len);
            return Encoding.UTF8.GetString(bytes);
        }

        /// <summary>
        /// The calling thread's last Vokra error message, or <c>null</c>. Must be
        /// read on the SAME thread that produced the error — <c>vokra_last_error</c>
        /// is a thread-local errno (FR-API-01).
        /// </summary>
        internal static string LastError()
        {
            return PtrToString(NativeMethods.LastError());
        }
    }
}
