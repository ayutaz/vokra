// WavIo.cs — minimal mono RIFF/WAVE reader + writer for the demo (M0-10-T06).
//
// Reads canonical RIFF/WAVE with a PCM16 (format 1) or IEEE-Float32 (format 3)
// `data` chunk into mono float[] + sample rate; writes float[] back as PCM16.
//
// Scope (M0): MONO only, and NO resampling — `resample` (FR-OP-04) is v0.1 MVP
// scope (M1-06), and soxr/rubberband are GPL and excluded as a design red line.
// Stereo input, unsupported formats/bit-depths, and a sample rate other than the
// caller's expected value are EXPLICIT errors, not silently coerced. Pure .NET
// (no UnityEngine) so the parse logic is host-agnostic; AudioClip conversion lives
// in AudioClipUtil.cs.

using System;
using System.IO;
using System.Text;

namespace Vokra.Demo
{
    public sealed class WavFormatException : Exception
    {
        public WavFormatException(string message) : base(message) { }
    }

    public static class WavIo
    {
        /// <summary>Reads a mono WAV file into f32 samples in [-1, 1] plus its sample rate.</summary>
        public static (float[] samples, int sampleRate) ReadMono(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            return ParseMono(bytes);
        }

        /// <summary>Parses a mono WAV from an in-memory buffer (see <see cref="ReadMono"/>).</summary>
        public static (float[] samples, int sampleRate) ParseMono(byte[] bytes)
        {
            if (bytes == null || bytes.Length < 12)
            {
                throw new WavFormatException("not a WAV: buffer too small");
            }

            if (Ascii(bytes, 0) != "RIFF" || Ascii(bytes, 8) != "WAVE")
            {
                throw new WavFormatException("not a RIFF/WAVE file");
            }

            int fmtTag = -1, channels = -1, sampleRate = -1, bitsPerSample = -1;
            int dataOffset = -1, dataLen = -1;

            int pos = 12; // after "RIFF" <size> "WAVE"
            while (pos + 8 <= bytes.Length)
            {
                string id = Ascii(bytes, pos);
                long size = ReadU32(bytes, pos + 4);
                int body = pos + 8;
                if (body + size > bytes.Length)
                {
                    throw new WavFormatException($"chunk '{id}' size {size} runs past end of file");
                }

                if (id == "fmt ")
                {
                    if (size < 16)
                    {
                        throw new WavFormatException("fmt chunk too small");
                    }

                    fmtTag = (int)ReadU16(bytes, body);
                    channels = (int)ReadU16(bytes, body + 2);
                    sampleRate = (int)ReadU32(bytes, body + 4);
                    bitsPerSample = (int)ReadU16(bytes, body + 14);
                }
                else if (id == "data")
                {
                    dataOffset = body;
                    dataLen = (int)size;
                }

                // Chunks are word-aligned: an odd size is followed by a pad byte.
                pos = body + (int)size + ((int)size & 1);
            }

            if (fmtTag < 0)
            {
                throw new WavFormatException("missing fmt chunk");
            }

            if (dataOffset < 0)
            {
                throw new WavFormatException("missing data chunk");
            }

            if (channels != 1)
            {
                throw new WavFormatException($"expected mono, got {channels} channels (demo does not down-mix)");
            }

            float[] samples;
            if (fmtTag == 1 && bitsPerSample == 16)
            {
                int n = dataLen / 2;
                samples = new float[n];
                for (int i = 0; i < n; i++)
                {
                    short s = (short)ReadU16(bytes, dataOffset + i * 2);
                    samples[i] = s / 32768f;
                }
            }
            else if (fmtTag == 3 && bitsPerSample == 32)
            {
                int n = dataLen / 4;
                samples = new float[n];
                for (int i = 0; i < n; i++)
                {
                    samples[i] = BitConverter.ToSingle(bytes, dataOffset + i * 4);
                }
            }
            else
            {
                throw new WavFormatException(
                    $"unsupported WAV format tag={fmtTag} bits={bitsPerSample} (want PCM16 or Float32)");
            }

            return (samples, sampleRate);
        }

        /// <summary>
        /// Reads a mono WAV and requires <paramref name="expectedRate"/> Hz — the demo
        /// never resamples (FR-OP-04 is M1). A mismatch is an explicit error.
        /// </summary>
        public static float[] ReadMonoExpectingRate(string path, int expectedRate)
        {
            (float[] samples, int rate) = ReadMono(path);
            if (rate != expectedRate)
            {
                throw new WavFormatException(
                    $"'{Path.GetFileName(path)}' is {rate} Hz but the model needs {expectedRate} Hz; " +
                    "the demo does not resample (FR-OP-04 is v0.1 MVP). Provide a matching-rate mono WAV.");
            }

            return samples;
        }

        /// <summary>Writes mono f32 samples (clamped to [-1, 1]) as a PCM16 WAV.</summary>
        public static void WritePcm16(string path, float[] samples, int sampleRate)
        {
            samples ??= Array.Empty<float>();
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var w = new BinaryWriter(fs);

            int dataBytes = samples.Length * 2;
            const int byteRate16 = 2; // bytes per mono sample
            w.Write(Encoding.ASCII.GetBytes("RIFF"));
            w.Write(36 + dataBytes);
            w.Write(Encoding.ASCII.GetBytes("WAVE"));

            w.Write(Encoding.ASCII.GetBytes("fmt "));
            w.Write(16);                                  // PCM fmt chunk size
            w.Write((short)1);                            // audio format = PCM
            w.Write((short)1);                            // channels = mono
            w.Write(sampleRate);                          // sample rate
            w.Write(sampleRate * byteRate16);             // byte rate
            w.Write((short)byteRate16);                   // block align
            w.Write((short)16);                           // bits per sample

            w.Write(Encoding.ASCII.GetBytes("data"));
            w.Write(dataBytes);
            foreach (float f in samples)
            {
                float c = f < -1f ? -1f : (f > 1f ? 1f : f);
                w.Write((short)Math.Round(c * 32767f));
            }
        }

        /// <summary>Reads raw little-endian f32 PCM (headerless, e.g. the committed .f32 fixtures).</summary>
        public static float[] ReadRawFloat32(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (bytes.Length % 4 != 0)
            {
                throw new WavFormatException("raw f32 length is not a multiple of 4");
            }

            var samples = new float[bytes.Length / 4];
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] = BitConverter.ToSingle(bytes, i * 4);
            }

            return samples;
        }

        private static string Ascii(byte[] b, int offset) => Encoding.ASCII.GetString(b, offset, 4);

        private static uint ReadU16(byte[] b, int o) => (uint)(b[o] | (b[o + 1] << 8));

        private static long ReadU32(byte[] b, int o) =>
            b[o] | ((long)b[o + 1] << 8) | ((long)b[o + 2] << 16) | ((long)b[o + 3] << 24);
    }
}
