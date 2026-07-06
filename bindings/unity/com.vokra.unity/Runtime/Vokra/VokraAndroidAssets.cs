// SPDX-License-Identifier: Apache-2.0
// Vokra - Android StreamingAssets expansion helper (NFR-RL-04).
//
// On Android, files under Assets/StreamingAssets are packaged inside the APK
// (or split APKs). Their virtual URL is of the form
//   jar:file:///data/app/<pkg>/base.apk!/assets/<relative>
// which the C ABI (fopen/mmap) cannot open directly. This helper copies the
// asset out to Application.persistentDataPath on first access and returns a
// real filesystem path safe for native code.
//
// On non-Android platforms (Windows/macOS/Linux/iOS + Editor) the file already
// lives on the real filesystem under Application.streamingAssetsPath, so we
// return that path directly.
//
// IL2CPP-safe: pure managed code, no delegate marshalling.

using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Networking;
#endif

namespace Vokra
{
    /// <summary>
    /// Resolves a StreamingAssets-relative path into a real filesystem path
    /// suitable for native code (fopen/mmap via the Vokra C ABI).
    /// On Android the asset is copied out of the APK jar into
    /// <see cref="Application.persistentDataPath"/> on first access; on all
    /// other platforms the direct StreamingAssets path is returned.
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
        public static string EnsureLocalCopy(string relativePath)
        {
            ValidateRelativePath(relativePath);

#if UNITY_ANDROID && !UNITY_EDITOR
            string sourceUrl = BuildStreamingAssetsUrl(relativePath);
            string destinationPath = BuildDestinationPath(relativePath);

            using (UnityWebRequest request = UnityWebRequest.Get(sourceUrl))
            {
                var op = request.SendWebRequest();
                // Synchronous wait: one-shot startup init only; do not call per frame.
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
        /// </summary>
        /// <param name="relativePath">Path relative to StreamingAssets.</param>
        /// <param name="progress">Optional progress reporter.</param>
        /// <returns>Task resolving to the absolute filesystem path.</returns>
        public static async Task<string> EnsureLocalCopyAsync(
            string relativePath,
            IProgress<float> progress = null)
        {
            ValidateRelativePath(relativePath);

#if UNITY_ANDROID && !UNITY_EDITOR
            string sourceUrl = BuildStreamingAssetsUrl(relativePath);
            string destinationPath = BuildDestinationPath(relativePath);

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
                        "' from APK jar (" + sourceUrl + "): " + request.error);
                }

                byte[] payload = request.downloadHandler.data;
                if (payload == null)
                {
                    throw new IOException(
                        "Vokra: empty payload for StreamingAssets '" + relativePath + "'.");
                }

                if (File.Exists(destinationPath) &&
                    new FileInfo(destinationPath).Length == payload.LongLength)
                {
                    progress?.Report(1f);
                    return destinationPath;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
                File.WriteAllBytes(destinationPath, payload);
                progress?.Report(1f);
            }

            return destinationPath;
#else
            progress?.Report(1f);
            await Task.CompletedTask;
            return Path.Combine(Application.streamingAssetsPath, relativePath);
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

#if UNITY_ANDROID && !UNITY_EDITOR
        private static string BuildStreamingAssetsUrl(string relativePath)
        {
            // Application.streamingAssetsPath on Android is
            //   jar:file:///data/app/<pkg>/base.apk!/assets
            // We append the relative path with forward slashes.
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
