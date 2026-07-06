// AudioClipUtil.cs — float[] PCM → AudioClip for playback (M0-10-T06).

using UnityEngine;

namespace Vokra.Demo
{
    public static class AudioClipUtil
    {
        /// <summary>Wraps mono f32 PCM in an <see cref="AudioClip"/> for an AudioSource.</summary>
        public static AudioClip ToMonoClip(string name, float[] pcm, int sampleRate)
        {
            if (pcm == null || pcm.Length == 0)
            {
                return null;
            }

            var clip = AudioClip.Create(name, pcm.Length, 1, sampleRate, false);
            clip.SetData(pcm, 0);
            return clip;
        }
    }
}
