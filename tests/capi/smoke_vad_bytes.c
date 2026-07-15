/*
 * smoke_vad_bytes.c — C smoke test for the bytes-based session create
 * (M4-02, `vokra_session_create_from_bytes`).
 *
 * The bytes-based twin of smoke_vad.c: the C caller reads the whole GGUF
 * into memory (fopen/fread — the embedder-side IO that Unity C# performs on
 * WebGL) and hands the buffer to the C ABI, which must never touch the
 * filesystem for the model. Then the full VAD stream path runs: push PCM ->
 * poll speech probabilities.
 *
 * This file is ALSO the wasm32-unknown-emscripten verification harness body
 * (ADR M4-02 §2/§3): linked against the Unity WebGL staticlib with the
 * pinned emcc and executed under node by scripts/build-unity-webgl-lib.sh
 * --verify, it proves the C-ABI-side IO + bytes session + Silero VAD
 * inference all work under a Unity-era Emscripten runtime, where the
 * rust-std fs path (`vokra_session_create_from_file`) is ABI-skewed.
 *
 * The from_file behaviour is probed informationally (printed, not asserted):
 * the FR-EX-08 property asserted here is "from_file either works or fails
 * loudly with a non-OK status — it never hangs and never yields a broken
 * session silently".
 *
 * Usage: smoke_vad_bytes [model.gguf] [input.f32]
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

/* Reads a whole file into a malloc'd buffer. Caller frees. */
static unsigned char *read_all(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long bytes = ftell(f);
    if (bytes <= 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    unsigned char *buf = (unsigned char *)malloc((size_t)bytes);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)bytes, f);
    fclose(f);
    if (got != (size_t)bytes) {
        free(buf);
        return NULL;
    }
    *out_len = (size_t)bytes;
    return buf;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : DEFAULT_MODEL;
    const char *input = argc > 2 ? argv[2] : DEFAULT_INPUT;

    printf("smoke_vad_bytes: vokra %s\n", vokra_version());

    /* Embedder-side IO: C reads the model bytes (Unity C# does the same). */
    size_t model_len = 0;
    unsigned char *model_bytes = read_all(model, &model_len);
    if (!model_bytes) {
        fprintf(stderr, "smoke_vad_bytes: FAIL could not read model %s\n", model);
        return 1;
    }
    printf("smoke_vad_bytes: read %zu model bytes from %s\n", model_len, model);

    size_t n = 0;
    unsigned char *pcm_raw = read_all(input, &n);
    if (!pcm_raw || n % sizeof(float) != 0) {
        fprintf(stderr, "smoke_vad_bytes: FAIL could not read fixture %s\n", input);
        free(model_bytes);
        free(pcm_raw);
        return 1;
    }
    float *pcm = (float *)pcm_raw;
    n /= sizeof(float);

    /* Bytes-based session create: the primary Unity WebGL model path. */
    vokra_session_t *session = NULL;
    vokra_status_t st = vokra_session_create_from_bytes(model_bytes, model_len, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_vad_bytes: FAIL create from bytes (%d): %s\n", (int)st,
                vokra_last_error());
        free(model_bytes);
        free(pcm);
        return 1;
    }
    /* The buffer is copied by the call: freeing it immediately is legal. */
    free(model_bytes);

    vokra_stream_t *stream = NULL;
    st = vokra_stream_open(session, 16000, &stream);
    if (st != VOKRA_OK || stream == NULL) {
        fprintf(stderr, "smoke_vad_bytes: FAIL open stream (%d): %s\n", (int)st,
                vokra_last_error());
        vokra_session_destroy(session);
        free(pcm);
        return 1;
    }

    size_t total = 0;
    for (size_t off = 0; off < n;) {
        size_t chunk = (n - off < 2048) ? (n - off) : 2048;
        st = vokra_stream_push_pcm(stream, pcm + off, chunk);
        if (st != VOKRA_OK) {
            fprintf(stderr, "smoke_vad_bytes: FAIL push (%d): %s\n", (int)st,
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
            fprintf(stderr, "smoke_vad_bytes: FAIL poll (%d): %s\n", (int)st,
                    vokra_last_error());
            vokra_stream_destroy(stream);
            vokra_session_destroy(session);
            free(pcm);
            return 1;
        }
        for (size_t i = 0; i < count; i++) {
            if (!(probs[i] >= 0.0f && probs[i] <= 1.0f)) {
                fprintf(stderr, "smoke_vad_bytes: FAIL probability %f out of [0,1]\n",
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
    vokra_session_destroy(session);
    free(pcm);

    if (total == 0) {
        fprintf(stderr, "smoke_vad_bytes: FAIL no speech probabilities produced\n");
        return 1;
    }
    printf("smoke_vad_bytes: %zu frames, %zu probabilities\n", n / 512, total);

    /* Error path: NULL data / empty buffer must fail loudly, not crash. */
    vokra_session_t *bad = NULL;
    st = vokra_session_create_from_bytes(NULL, 16, &bad);
    if (st == VOKRA_OK || bad != NULL) {
        fprintf(stderr, "smoke_vad_bytes: FAIL NULL data was not rejected\n");
        return 1;
    }
    unsigned char one = 0;
    st = vokra_session_create_from_bytes(&one, 0, &bad);
    if (st == VOKRA_OK || bad != NULL) {
        fprintf(stderr, "smoke_vad_bytes: FAIL empty buffer was not rejected\n");
        return 1;
    }
    printf("smoke_vad_bytes: NULL/empty buffer rejected: %s\n", vokra_last_error());

    /* Informational probe (NOT an assertion of success): from_file must
     * either succeed or fail with a non-OK status — no hang, no silent
     * broken session (FR-EX-08). Under Unity-era Emscripten the rust-std
     * stat ABI skew makes this the loud-failure branch (ADR M4-02 §2). */
    vokra_session_t *file_session = NULL;
    st = vokra_session_create_from_file(model, &file_session);
    if (st == VOKRA_OK && file_session != NULL) {
        printf("smoke_vad_bytes: from_file also works on this runtime\n");
        vokra_session_destroy(file_session);
    } else if (st != VOKRA_OK && file_session == NULL) {
        printf("smoke_vad_bytes: from_file fails loudly on this runtime (%d): %s\n",
               (int)st, vokra_last_error());
    } else {
        fprintf(stderr, "smoke_vad_bytes: FAIL from_file returned an inconsistent "
                        "status/handle pair\n");
        return 1;
    }

    printf("smoke_vad_bytes: PASS\n");
    return 0;
}
