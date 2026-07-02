/*
 * smoke_vad.c — C smoke test for the Vokra VAD stream API (M0-09-T11).
 *
 * Exercises the C ABI end to end from C, using only <vokra.h>:
 *   create session (Silero VAD GGUF) -> open stream -> push PCM -> poll probs.
 * Plus one error case: a non-existent model must fail with a non-zero status
 * and a non-NULL vokra_last_error().
 *
 * Audio input is raw little-endian float32 PCM read with fread — no WAV parser
 * and no strtod / locale-dependent parsing (NFR-RL-01). The model is the
 * committed 2 MB Silero fixture (no env gate needed).
 *
 * Usage: smoke_vad [model.gguf] [input.f32]
 *   defaults (run from the repo root):
 *     tests/parity/silero_vad/silero-vad-v5.gguf
 *     tests/capi/fixtures/vad_input_16k.f32
 *
 * Exit code: 0 = pass, 1 = fail.
 */

#include <stdio.h>
#include <stdlib.h>

#include "vokra.h"

static const char *DEFAULT_MODEL = "tests/parity/silero_vad/silero-vad-v5.gguf";
static const char *DEFAULT_INPUT = "tests/capi/fixtures/vad_input_16k.f32";

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
    const char *model = argc > 1 ? argv[1] : DEFAULT_MODEL;
    const char *input = argc > 2 ? argv[2] : DEFAULT_INPUT;

    printf("smoke_vad: vokra %s\n", vokra_version());

    size_t n = 0;
    float *pcm = read_f32(input, &n);
    if (!pcm || n == 0) {
        fprintf(stderr, "smoke_vad: FAIL could not read fixture %s\n", input);
        free(pcm);
        return 1;
    }

    vokra_session_t *session = NULL;
    vokra_status_t st = vokra_session_create_from_file(model, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_vad: FAIL create session (%d): %s\n", (int)st,
                vokra_last_error());
        free(pcm);
        return 1;
    }

    vokra_stream_t *stream = NULL;
    st = vokra_stream_open(session, 16000, &stream);
    if (st != VOKRA_OK || stream == NULL) {
        fprintf(stderr, "smoke_vad: FAIL open stream (%d): %s\n", (int)st,
                vokra_last_error());
        vokra_session_destroy(session);
        free(pcm);
        return 1;
    }

    /* Push in chunks, polling after each to exercise buffered push/poll. */
    size_t total = 0;
    for (size_t off = 0; off < n;) {
        size_t chunk = (n - off < 2048) ? (n - off) : 2048;
        st = vokra_stream_push_pcm(stream, pcm + off, chunk);
        if (st != VOKRA_OK) {
            fprintf(stderr, "smoke_vad: FAIL push (%d): %s\n", (int)st,
                    vokra_last_error());
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
            free(pcm);
            return 1;
        }
        off += chunk;

        float probs[64];
        size_t count = 0;
        st = vokra_stream_poll(stream, probs, 64, &count);
        if (st != VOKRA_OK) {
            fprintf(stderr, "smoke_vad: FAIL poll (%d): %s\n", (int)st,
                    vokra_last_error());
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
            free(pcm);
            return 1;
        }
        for (size_t i = 0; i < count; i++) {
            if (!(probs[i] >= 0.0f && probs[i] <= 1.0f)) {
                fprintf(stderr, "smoke_vad: FAIL probability %f out of [0,1]\n",
                        probs[i]);
                vokra_stream_destroy(stream);
                vokra_session_destroy(session);
                free(pcm);
                return 1;
            }
        }
        total += count;
    }

    vokra_stream_destroy(stream);
    free(pcm);

    if (total == 0) {
        fprintf(stderr, "smoke_vad: FAIL no speech probabilities produced\n");
        vokra_session_destroy(session);
        return 1;
    }
    printf("smoke_vad: %zu frames, %zu probabilities\n", n / 512, total);

    /* Error path: a non-existent model must fail loudly, not crash. */
    vokra_session_t *bad = NULL;
    st = vokra_session_create_from_file("/vokra/no/such/model.gguf", &bad);
    if (st == VOKRA_OK || bad != NULL) {
        fprintf(stderr, "smoke_vad: FAIL missing model was not rejected\n");
        vokra_session_destroy(session);
        return 1;
    }
    if (vokra_last_error() == NULL) {
        fprintf(stderr, "smoke_vad: FAIL missing model set no error message\n");
        vokra_session_destroy(session);
        return 1;
    }
    printf("smoke_vad: missing-model error reported: %s\n", vokra_last_error());

    vokra_session_destroy(session);
    printf("smoke_vad: PASS\n");
    return 0;
}
