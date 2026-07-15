// SPDX-License-Identifier: Apache-2.0
// Vokra - StreamingAssets expansion helper for Android and WebGL
// (NFR-RL-04 + its WebGL twin, M4-02-T07).
//
// On Android, files under Assets/StreamingAssets are packaged inside the APK
// (or split APKs). Their virtual URL is of the form
//   jar:file:///data/app/<pkg>/base.apk!/assets/<relative>
// which the C ABI (fopen/mmap) cannot open directly. This helper copies the
// asset out to Application.persistentDataPath on first access and returns a
// real filesystem path safe for native code.
//
// On WebGL, StreamingAssets are HTTP(S)-served next to the player build —
// there is no filesystem URL at all, and additionally the Vokra C ABI's
// file-path loader is not usable under Unity-bundled Emscripten (rust-std
// fs ABI skew, ADR M4-02 §2). The WebGL model path is therefore BYTES:
// fetch with ReadBytesAsync and load with VokraSession.CreateFromBytes.
// EnsureLocalCopyAsync still works on WebGL for MANAGED (C#) consumers — it
// materialises the asset into persistentDataPath (Emscripten virtual FS),
// which C# file IO reads fine — but the returned path must NOT be handed to
// VokraSession.CreateFromFile on WebGL. The synchronous EnsureLocalCopy
// throws NotSupportedException on WebGL: a synchronous busy-wait would
// deadlock the single-threaded browser main loop, so it fails loudly
// instead (FR-EX-08).
//
// On the remaining platforms (Windows/macOS/Linux/iOS + Editor) the file
// already lives on the real filesystem under Application.streamingAssetsPath,
// so the path helpers return that path directly.
//
// IL2CPP-safe: pure managed code, no delegate marshalling.

using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
#if (UNITY_ANDROID || UNITY_WEBGL) && !UNITY_EDITOR
using UnityEngine.Networking;
#endif

namespace Vokra
{
    /// <summary>
    /// Resolves StreamingAssets content for native consumption.
    /// On Android the asset is copied out of the APK jar into
    /// <see cref="Application.persistentDataPath"/> on first access; on WebGL
    /// assets are HTTP-fetched (<see cref="ReadBytesAsync"/> pairs with
    /// <c>VokraSession.CreateFromBytes</c>, and the synchronous
    /// <see cref="EnsureLocalCopy"/> is not supported); on all other
    /// platforms the direct StreamingAssets path is returned.
    /// </summary>
    public static class VokraAndroidAssets
    {
        private const string SubDirectory = "vokra";

        /// <summary>
        /// Synchronously ensures the asset at <paramref name="relativePath"/>
        /// exists on the real filesystem and returns its absolute path.
        /// Idempotent: on Android the copy is skipped when the destination
        /// file already exists with a matching byte count.
        /// </summary>
        /// <param name="relativePath">Path relative to StreamingAssets. Must not
        /// be null, empty, or absolute.</param>
        /// <returns>Absolute filesystem path safe for fopen / mmap.</returns>
        /// <exception cref="NotSupportedException">On WebGL: a synchronous
        /// wait on the browser main thread would deadlock the tab, so this
        /// throws instead of hanging (FR-EX-08). Use
        /// <see cref="ReadBytesAsync"/> + <c>VokraSession.CreateFromBytes</c>
        /// for models, or <see cref="EnsureLocalCopyAsync"/> for managed
        /// file IO.</exception>
        public static string EnsureLocalCopy(string relativePath)
        {
            ValidateRelativePath(relativePath);

#if UNITY_WEBGL && !UNITY_EDITOR
            throw new NotSupportedException(
                "Vokra: EnsureLocalCopy is synchronous and would deadlock the " +
                "WebGL main thread (StreamingAssets are HTTP-served). Use " +
                "VokraAndroidAssets.ReadBytesAsync + VokraSession.CreateFromBytes " +
                "for models, or EnsureLocalCopyAsync for managed file IO.");
#elif UNITY_ANDROID && !UNITY_EDITOR
            string sourceUrl = BuildStreamingAssetsUrl(relativePath);
            string destinationPath = BuildDestinationPath(relativePath);

            using (UnityWebRequest request = UnityWebRequest.Get(sourceUrl))
            {
                var op = request.SendWebRequest();
                // Synchronous wait: one-shot startup init only; do not call per
                // frame. Android-only — a busy-wait like this deadlocks WebGL
                // (see the NotSupportedException branch above).
                while (!op.isDone) { }

#if UNITY_2020_2_OR_NEWER
                if (request.result != UnityWebRequest.Result.Success)
#else
                if (request.isNetworkError || request.isHttpError)
#endif
                {
                    throw new IOException(
                        "Vokra: failed to read StreamingAssets '" + relativePath +
                        "' from APK jar (" + sourceUrl + "): " + request.error);
                }

                byte[] payload = request.downloadHandler.data;
                if (payload == null)
                {
                    throw new IOException(
                        "Vokra: empty payload for StreamingAssets '" + relativePath + "'.");
                }

                // Idempotent size check.
                if (File.Exists(destinationPath) &&
                    new FileInfo(destinationPath).Length == payload.LongLength)
                {
                    return destinationPath;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
                File.WriteAllBytes(destinationPath, payload);
            }

            return destinationPath;
#else
            // Editor / Desktop / iOS: streamingAssetsPath is a real filesystem path.
            return Path.Combine(Application.streamingAssetsPath, relativePath);
#endif
        }

        /// <summary>
        /// Asynchronous variant of <see cref="EnsureLocalCopy"/> for large
        /// models (e.g., Whisper large-v3, ~3 GB). Reports byte-level progress
        /// in the range [0.0, 1.0] via <paramref name="progress"/>.
        /// On WebGL the asset is fetched over HTTP and written into
        /// <see cref="Application.persistentDataPath"/> (Emscripten virtual
        /// filesystem): the returned path is valid for MANAGED (C#) file IO —
        /// do NOT pass it to <c>VokraSession.CreateFromFile</c> on WebGL
        /// (rust-std fs ABI skew under Unity-bundled Emscripten fails loudly,
        /// ADR M4-02 §2); use <see cref="ReadBytesAsync"/> +
        /// <c>VokraSession.CreateFromBytes</c> for models instead.
        /// In-session reads of the written file are immediate; persistence
        /// across browser sessions depends on the Unity version's IndexedDB
        /// sync behaviour (verify on the target Unity version — not asserted
        /// here).
        /// </summary>
        /// <param name="relativePath">Path relative to StreamingAssets.</param>
        /// <param name="progress">Optional progress reporter.</param>
        /// <returns>Task resolving to the absolute filesystem path.</returns>
        public static async Task<string> EnsureLocalCopyAsync(
            string relativePath,
            IProgress<float> progress = null)
        {
            ValidateRelativePath(relativePath);

#if (UNITY_ANDROID || UNITY_WEBGL) && !UNITY_EDITOR
            string destinationPath = BuildDestinationPath(relativePath);
            byte[] payload = await FetchStreamingAssetAsync(relativePath, progress);

            if (File.Exists(destinationPath) &&
                new FileInfo(destinationPath).Length == payload.LongLength)
            {
                progress?.Report(1f);
                return destinationPath;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
            File.WriteAllBytes(destinationPath, payload);
            progress?.Report(1f);
            return destinationPath;
#else
            progress?.Report(1f);
            await Task.CompletedTask;
            return Path.Combine(Application.streamingAssetsPath, relativePath);
#endif
        }

        /// <summary>
        /// Reads the whole StreamingAssets asset at
        /// <paramref name="relativePath"/> into memory. This is the model
        /// path on WebGL (M4-02): pair the returned bytes with
        /// <c>VokraSession.CreateFromBytes</c>, which never touches the
        /// (HTTP-served, ABI-skewed) filesystem. Works on every platform:
        /// Android/WebGL fetch via UnityWebRequest, everything else reads the
        /// file directly.
        /// </summary>
        /// <param name="relativePath">Path relative to StreamingAssets.</param>
        /// <param name="progress">Optional progress reporter.</param>
        /// <returns>Task resolving to the asset bytes.</returns>
        public static async Task<byte[]> ReadBytesAsync(
            string relativePath,
            IProgress<float> progress = null)
        {
            ValidateRelativePath(relativePath);

#if (UNITY_ANDROID || UNITY_WEBGL) && !UNITY_EDITOR
            byte[] payload = await FetchStreamingAssetAsync(relativePath, progress);
            progress?.Report(1f);
            return payload;
#else
            byte[] payload = File.ReadAllBytes(
                Path.Combine(Application.streamingAssetsPath, relativePath));
            progress?.Report(1f);
            await Task.CompletedTask;
            return payload;
#endif
        }

        private static void ValidateRelativePath(string relativePath)
        {
            if (string.IsNullOrEmpty(relativePath))
            {
                throw new ArgumentException(
                    "relativePath must be non-empty.", nameof(relativePath));
            }
            if (Path.IsPathRooted(relativePath))
            {
                throw new ArgumentException(
                    "relativePath must be relative, not absolute.",
                    nameof(relativePath));
            }
            if (relativePath.Contains(".."))
            {
                throw new ArgumentException(
                    "relativePath must not contain '..' segments.",
                    nameof(relativePath));
            }
        }

#if (UNITY_ANDROID || UNITY_WEBGL) && !UNITY_EDITOR
        /// <summary>
        /// Awaits a UnityWebRequest GET of the StreamingAssets asset and
        /// returns its payload. Shared by the Android jar expansion and the
        /// WebGL HTTP fetch; always polls asynchronously (Task.Yield) — never
        /// a busy-wait, which would deadlock the WebGL main thread.
        /// </summary>
        private static async Task<byte[]> FetchStreamingAssetAsync(
            string relativePath,
            IProgress<float> progress)
        {
            string sourceUrl = BuildStreamingAssetsUrl(relativePath);
            using (UnityWebRequest request = UnityWebRequest.Get(sourceUrl))
            {
                var op = request.SendWebRequest();
                float last = -1f;
                while (!op.isDone)
                {
                    if (progress != null)
                    {
                        float p = op.progress;
                        if (p != last)
                        {
                            progress.Report(p);
                            last = p;
                        }
                    }
                    await Task.Yield();
                }

#if UNITY_2020_2_OR_NEWER
                if (request.result != UnityWebRequest.Result.Success)
#else
                if (request.isNetworkError || request.isHttpError)
#endif
                {
                    throw new IOException(
                        "Vokra: failed to read StreamingAssets '" + relativePath +
                        "' (" + sourceUrl + "): " + request.error);
                }

                byte[] payload = request.downloadHandler.data;
                if (payload == null)
                {
                    throw new IOException(
                        "Vokra: empty payload for StreamingAssets '" + relativePath + "'.");
                }

                return payload;
            }
        }

        private static string BuildStreamingAssetsUrl(string relativePath)
        {
            // Android: Application.streamingAssetsPath is
            //   jar:file:///data/app/<pkg>/base.apk!/assets
            // WebGL: it is the HTTP(S) URL of the StreamingAssets folder next
            // to the player build. Both take a '/'-joined relative suffix.
            string normalized = relativePath.Replace('\\', '/');
            return Application.streamingAssetsPath + "/" + normalized;
        }

        private static string BuildDestinationPath(string relativePath)
        {
            return Path.Combine(
                Application.persistentDataPath, SubDirectory, relativePath);
        }
#endif
    }
}
