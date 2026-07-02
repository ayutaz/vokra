/*
 * smoke_asr.c — C smoke test for the Vokra ASR API (M0-09-T11).
 *
 * create session (Whisper GGUF) -> vokra_asr_transcribe -> non-empty text ->
 * vokra_string_free. Using only <vokra.h>. Audio input is raw little-endian
 * float32 PCM read with fread (no WAV parser, no strtod — NFR-RL-01).
 *
 * ENV-GATED: the Whisper base GGUF (~290 MB) is not committed, so the model
 * path comes from VOKRA_WHISPER_GGUF. When unset the test cleanly SKIPs
 * (exit 0), matching the M0-05/06/07 parity gating.
 *
 * Usage:  VOKRA_WHISPER_GGUF=whisper-base.gguf smoke_asr [input.f32]
 *   default input (run from repo root): tests/capi/fixtures/asr_input_16k.f32
 *
 * Exit code: 0 = pass or skip, 1 = fail.
 */

#include <stdio.h>
#include <stdlib.h>

#include "vokra.h"

static const char *DEFAULT_INPUT = "tests/capi/fixtures/asr_input_16k.f32";

/* Reads a whole file of raw little-endian f32 samples. Caller frees. */
static float *read_f32(const char *path, size_t *out_n) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long bytes = ftell(f);
    if (bytes < 0 || bytes % (long)sizeof(float) != 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    size_t n = (size_t)bytes / sizeof(float);
    float *buf = (float *)malloc(n ? n * sizeof(float) : 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) {
        free(buf);
        return NULL;
    }
    *out_n = n;
    return buf;
}

int main(int argc, char **argv) {
    const char *model = getenv("VOKRA_WHISPER_GGUF");
    if (!model || model[0] == '\0') {
        printf("smoke_asr: SKIP (set VOKRA_WHISPER_GGUF to a whisper GGUF to run)\n");
        return 0;
    }
    const char *input = argc > 1 ? argv[1] : DEFAULT_INPUT;

    printf("smoke_asr: vokra %s\n", vokra_version());

    size_t n = 0;
    float *pcm = read_f32(input, &n);
    if (!pcm || n == 0) {
        fprintf(stderr, "smoke_asr: FAIL could not read fixture %s\n", input);
        free(pcm);
        return 1;
    }

    vokra_session_t *session = NULL;
    vokra_status_t st = vokra_session_create_from_file(model, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_asr: FAIL create session (%d): %s\n", (int)st,
                vokra_last_error());
        free(pcm);
        return 1;
    }

    char *text = NULL;
    st = vokra_asr_transcribe(session, pcm, n, 16000, &text);
    free(pcm);
    if (st != VOKRA_OK || text == NULL) {
        fprintf(stderr, "smoke_asr: FAIL transcribe (%d): %s\n", (int)st,
                vokra_last_error());
        vokra_session_destroy(session);
        return 1;
    }
    if (text[0] == '\0') {
        fprintf(stderr, "smoke_asr: FAIL transcript is empty\n");
        vokra_string_free(text);
        vokra_session_destroy(session);
        return 1;
    }
    printf("smoke_asr: transcript = \"%s\"\n", text);
    vokra_string_free(text);
    vokra_session_destroy(session);
    printf("smoke_asr: PASS\n");
    return 0;
}
