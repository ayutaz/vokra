/*
 * smoke_tts.c — C smoke test for the Vokra TTS API (M0-09-T11).
 *
 * create session (piper-plus voice GGUF) -> vokra_tts_synthesize -> non-empty
 * PCM + sample rate -> vokra_audio_free. Using only <vokra.h>. The synthesis
 * path is piper-plus native (MB-iSTFT-VITS2); no onnxruntime is involved
 * (FR-LD-05).
 *
 * ENV-GATED: the piper voice GGUF (~77 MB) is not committed, so the model path
 * comes from VOKRA_PIPER_GGUF. When unset the test cleanly SKIPs (exit 0).
 *
 * Usage:  VOKRA_PIPER_GGUF=voice.gguf smoke_tts [text]
 *   default text: "aiueo" (in-vocabulary for the piper voices; see tts_demo).
 *
 * Exit code: 0 = pass or skip, 1 = fail.
 */

#include <stdio.h>
#include <stdlib.h>

#include "vokra.h"

int main(int argc, char **argv) {
    const char *model = getenv("VOKRA_PIPER_GGUF");
    if (!model || model[0] == '\0') {
        printf("smoke_tts: SKIP (set VOKRA_PIPER_GGUF to a piper voice GGUF to run)\n");
        return 0;
    }
    const char *text = argc > 1 ? argv[1] : "aiueo";

    printf("smoke_tts: vokra %s\n", vokra_version());

    vokra_session_t *session = NULL;
    vokra_status_t st = vokra_session_create_from_file(model, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_tts: FAIL create session (%d): %s\n", (int)st,
                vokra_last_error());
        return 1;
    }

    float *pcm = NULL;
    size_t num_samples = 0;
    int32_t sample_rate = 0;
    st = vokra_tts_synthesize(session, text, &pcm, &num_samples, &sample_rate);
    if (st != VOKRA_OK || pcm == NULL) {
        fprintf(stderr, "smoke_tts: FAIL synthesize (%d): %s\n", (int)st,
                vokra_last_error());
        vokra_session_destroy(session);
        return 1;
    }
    if (num_samples == 0 || sample_rate <= 0) {
        fprintf(stderr, "smoke_tts: FAIL empty audio (%zu samples, %d Hz)\n",
                num_samples, (int)sample_rate);
        vokra_audio_free(pcm, num_samples);
        vokra_session_destroy(session);
        return 1;
    }
    printf("smoke_tts: \"%s\" -> %zu samples @ %d Hz (%.2f s)\n", text,
           num_samples, (int)sample_rate, (double)num_samples / sample_rate);

    vokra_audio_free(pcm, num_samples);
    vokra_session_destroy(session);
    printf("smoke_tts: PASS\n");
    return 0;
}
