/*
 * smoke_backend_options.c — C smoke test for backend selection + speaker
 * verification (design 2026-08-14 §3, C-caller view of T1-T6 / T8).
 *
 * The Rust integration test under crates/vokra-capi/tests/ declares the new
 * symbols by hand; this one goes through <vokra.h> the way a real embedder
 * does, so it also proves the generated declarations compile as C11 with
 * -Wall -Wextra -Werror and that `VOKRA_BACKEND_*` are usable enum constants
 * rather than comments.
 *
 * Covers:
 *   - vokra_backend_available(VOKRA_BACKEND_CPU) is true;
 *   - options lifecycle: create -> set_backend(CPU) -> create session ->
 *     destroy, and destroy(NULL) as a no-op;
 *   - opts = NULL means "defaults" and loads the same model;
 *   - an unavailable backend is refused with VOKRA_ERROR_BACKEND_UNAVAILABLE
 *     and never yields a session (FR-EX-08: no silent CPU fall back);
 *   - an unknown enum value is VOKRA_ERROR_INVALID_ARGUMENT and leaves the
 *     options object usable;
 *   - vokra_speaker_verify of a vector with itself is 1.0, and the
 *     similarity-only mode (NULL decision slot) works.
 *
 * The model is the committed 2 MB Silero fixture (no env gate needed). No
 * speaker model is required: vokra_speaker_verify is pure arithmetic on two
 * vectors and takes no session.
 *
 * Usage: smoke_backend_options [model.gguf]
 *   default (run from the repo root):
 *     tests/parity/silero_vad/silero-vad-v5.gguf
 *
 * Exit code: 0 = pass, 1 = fail.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vokra.h"

static const char *DEFAULT_MODEL = "tests/parity/silero_vad/silero-vad-v5.gguf";

/* Every non-CPU backend in the exposed enum. */
static const enum vokra_backend_t GPU_BACKENDS[] = {
    VOKRA_BACKEND_METAL,
    VOKRA_BACKEND_CUDA,
    VOKRA_BACKEND_VULKAN,
    VOKRA_BACKEND_WEBGPU,
};
static const size_t N_GPU_BACKENDS = sizeof(GPU_BACKENDS) / sizeof(GPU_BACKENDS[0]);

static const char *backend_name(enum vokra_backend_t b) {
    switch (b) {
    case VOKRA_BACKEND_CPU:
        return "CPU";
    case VOKRA_BACKEND_METAL:
        return "METAL";
    case VOKRA_BACKEND_CUDA:
        return "CUDA";
    case VOKRA_BACKEND_VULKAN:
        return "VULKAN";
    case VOKRA_BACKEND_WEBGPU:
        return "WEBGPU";
    }
    return "<unknown>";
}

static const char *last_error_or(const char *fallback) {
    const char *msg = vokra_last_error();
    return msg ? msg : fallback;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : DEFAULT_MODEL;

    printf("smoke_backend_options: vokra %s\n", vokra_version());

    /* 1. The CPU backend is the always-present baseline (FR-BE-01). */
    if (!vokra_backend_available(VOKRA_BACKEND_CPU)) {
        fprintf(stderr, "smoke_backend_options: CPU reported unavailable\n");
        return 1;
    }
    for (size_t i = 0; i < N_GPU_BACKENDS; i++) {
        printf("smoke_backend_options: %s available = %s\n",
               backend_name(GPU_BACKENDS[i]),
               vokra_backend_available(GPU_BACKENDS[i]) ? "yes" : "no");
    }

    /* 2. Options lifecycle + an explicit CPU session. */
    struct vokra_session_options_t *opts = vokra_session_options_create();
    if (!opts) {
        fprintf(stderr, "smoke_backend_options: options create returned NULL\n");
        return 1;
    }
    enum vokra_status_t st = vokra_session_options_set_backend(opts, VOKRA_BACKEND_CPU);
    if (st != VOKRA_OK) {
        fprintf(stderr, "smoke_backend_options: set_backend(CPU) failed: %d (%s)\n", (int)st,
                last_error_or("no message"));
        vokra_session_options_destroy(opts);
        return 1;
    }

    struct vokra_session_t *session = NULL;
    st = vokra_session_create_from_file_with_options(model, opts, &session);
    if (st != VOKRA_OK || !session) {
        fprintf(stderr, "smoke_backend_options: CPU session create failed: %d (%s)\n", (int)st,
                last_error_or("no message"));
        vokra_session_options_destroy(opts);
        return 1;
    }
    vokra_session_destroy(session);
    session = NULL;

    /* 3. opts = NULL means "use the defaults". */
    st = vokra_session_create_from_file_with_options(model, NULL, &session);
    if (st != VOKRA_OK || !session) {
        fprintf(stderr, "smoke_backend_options: NULL-options create failed: %d (%s)\n", (int)st,
                last_error_or("no message"));
        vokra_session_options_destroy(opts);
        return 1;
    }
    vokra_session_destroy(session);
    session = NULL;

    /* 4. An unavailable backend must be refused, never downgraded to CPU. */
    for (size_t i = 0; i < N_GPU_BACKENDS; i++) {
        enum vokra_backend_t backend = GPU_BACKENDS[i];
        if (vokra_backend_available(backend)) {
            continue; /* present on this machine — covered by the Rust test */
        }
        struct vokra_session_options_t *gpu = vokra_session_options_create();
        if (!gpu) {
            fprintf(stderr, "smoke_backend_options: options create returned NULL\n");
            vokra_session_options_destroy(opts);
            return 1;
        }
        st = vokra_session_options_set_backend(gpu, backend);
        if (st != VOKRA_OK) {
            /* Eager rejection is allowed, but only as BACKEND_UNAVAILABLE. */
            if (st != VOKRA_ERROR_BACKEND_UNAVAILABLE) {
                fprintf(stderr, "smoke_backend_options: set_backend(%s) gave %d\n",
                        backend_name(backend), (int)st);
                vokra_session_options_destroy(gpu);
                vokra_session_options_destroy(opts);
                return 1;
            }
            vokra_session_options_destroy(gpu);
            continue;
        }
        struct vokra_session_t *gpu_session = NULL;
        st = vokra_session_create_from_file_with_options(model, gpu, &gpu_session);
        if (st == VOKRA_OK || gpu_session != NULL) {
            fprintf(stderr,
                    "smoke_backend_options: %s is unavailable but a session was created — "
                    "that is the silent CPU fall back FR-EX-08 forbids\n",
                    backend_name(backend));
            vokra_session_destroy(gpu_session);
            vokra_session_options_destroy(gpu);
            vokra_session_options_destroy(opts);
            return 1;
        }
        if (st != VOKRA_ERROR_BACKEND_UNAVAILABLE) {
            fprintf(stderr, "smoke_backend_options: %s create gave %d, expected %d\n",
                    backend_name(backend), (int)st, (int)VOKRA_ERROR_BACKEND_UNAVAILABLE);
            vokra_session_options_destroy(gpu);
            vokra_session_options_destroy(opts);
            return 1;
        }
        vokra_session_options_destroy(gpu);
    }
    printf("smoke_backend_options: unavailable backends refused loudly, no CPU fall back\n");

    /* 5. An unknown enum value is rejected and leaves the object usable. The
     *    5 / 6 slots a caller might guess for CoreML / QNN are included. */
    const int UNKNOWN[] = {5, 6, 7, 99, -1};
    for (size_t i = 0; i < sizeof(UNKNOWN) / sizeof(UNKNOWN[0]); i++) {
        st = vokra_session_options_set_backend(opts, UNKNOWN[i]);
        if (st != VOKRA_ERROR_INVALID_ARGUMENT) {
            fprintf(stderr, "smoke_backend_options: set_backend(%d) gave %d, expected %d\n",
                    UNKNOWN[i], (int)st, (int)VOKRA_ERROR_INVALID_ARGUMENT);
            vokra_session_options_destroy(opts);
            return 1;
        }
    }
    st = vokra_session_create_from_file_with_options(model, opts, &session);
    if (st != VOKRA_OK || !session) {
        fprintf(stderr, "smoke_backend_options: a rejected set_backend broke the object: %d\n",
                (int)st);
        vokra_session_options_destroy(opts);
        return 1;
    }
    vokra_session_destroy(session);
    printf("smoke_backend_options: unknown enum values rejected, object still usable\n");

    /* NULL handles are documented no-ops / invalid-argument, never crashes. */
    vokra_session_options_destroy(NULL);
    if (vokra_session_options_set_backend(NULL, VOKRA_BACKEND_CPU) !=
        VOKRA_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "smoke_backend_options: set_backend(NULL) was not rejected\n");
        vokra_session_options_destroy(opts);
        return 1;
    }
    vokra_session_options_destroy(opts);

    /* 6. Speaker verification: no session, no model — two vectors in. */
    float emb[192];
    for (size_t i = 0; i < sizeof(emb) / sizeof(emb[0]); i++) {
        emb[i] = (float)((i % 7)) - 3.0f;
    }
    float similarity = 0.0f;
    bool same = false;
    st = vokra_speaker_verify(emb, 192, emb, 192, 0.99f, &similarity, &same);
    if (st != VOKRA_OK || fabsf(similarity - 1.0f) > 1e-5f || !same) {
        fprintf(stderr, "smoke_backend_options: self-verify gave %d similarity %f same %d\n",
                (int)st, (double)similarity, (int)same);
        return 1;
    }
    /* Similarity-only mode: a NULL decision slot needs no threshold. */
    similarity = 0.0f;
    st = vokra_speaker_verify(emb, 192, emb, 192, 0.0f, &similarity, NULL);
    if (st != VOKRA_OK || fabsf(similarity - 1.0f) > 1e-5f) {
        fprintf(stderr, "smoke_backend_options: similarity-only mode gave %d (%f)\n", (int)st,
                (double)similarity);
        return 1;
    }
    /* A length mismatch is a loud argument error. */
    if (vokra_speaker_verify(emb, 192, emb, 191, 0.5f, &similarity, NULL) !=
        VOKRA_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "smoke_backend_options: length mismatch was not rejected\n");
        return 1;
    }
    printf("smoke_backend_options: speaker_verify self-similarity %f\n", (double)similarity);

    printf("smoke_backend_options: PASS\n");
    return 0;
}
