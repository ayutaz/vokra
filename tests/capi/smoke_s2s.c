/*
 * smoke_s2s.c — C smoke test for the Vokra full-duplex S2S + model
 * attribution surface (M4-06: vokra_s2s_duplex_* + vokra_model_attribution),
 * using only <vokra.h>.
 *
 * Two always-on legs plus one env-gated leg:
 *
 *  1. NULL-argument error paths for every entry point — no model needed. The
 *     FR-EX-08 contract: a bad handle is a loud VOKRA_ERROR_INVALID_ARGUMENT,
 *     never a crash and never a silently broken session.
 *
 *  2. The committed Silero VAD GGUF (a permissive-license, non-duplex model)
 *     drives the model-free-ish surfaces: vokra_model_attribution reports NO
 *     display obligation (*out_needed == 0) for permissive weights, and
 *     vokra_s2s_duplex_open on a model with no duplex engine fails loudly.
 *
 *  3. ENV-GATED full duplex round trip: when VOKRA_MOSHI_GGUF points at a
 *     Moshi GGUF (uncommitted — CC-BY 4.0 weights, ~GBs), open -> frame_hop
 *     -> sample_rate -> attribution -> push_mic/pull_audio -> text ->
 *     interrupt -> destroy. When unset this leg SKIPs cleanly (the test still
 *     passes on legs 1-2), matching smoke_tts / smoke_asr env gating.
 *
 * Usage: smoke_s2s [permissive_model.gguf]
 *   default (from the repo root): tests/parity/silero_vad/silero-vad-v5.gguf
 *   VOKRA_MOSHI_GGUF=moshi.gguf smoke_s2s   (also runs the duplex leg)
 *
 * Exit code: 0 = pass (incl. env-gated skip), 1 = fail.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vokra.h"

static const char *DEFAULT_MODEL = "tests/parity/silero_vad/silero-vad-v5.gguf";

/* Reports whether `st` is the expected INVALID_ARGUMENT loud error; returns 1
   on mismatch (so callers can OR failures together). */
static int expect_invalid(vokra_status_t st, const char *what) {
    if (st != VOKRA_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "smoke_s2s: FAIL %s: expected INVALID_ARGUMENT, got %d (%s)\n",
                what, (int)st, vokra_last_error());
        return 1;
    }
    return 0;
}

/* Leg 3: the env-gated full duplex round trip. Returns 0 on pass, 1 on fail. */
static int run_moshi_leg(const char *model) {
    printf("smoke_s2s: duplex leg — moshi GGUF %s\n", model);

    vokra_session_t *session = NULL;
    vokra_status_t st = vokra_session_create_from_file(model, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_s2s: FAIL create moshi session (%d): %s\n", (int)st,
                vokra_last_error());
        return 1;
    }

    /* Deterministic sampling; recorded-input opt-out (aec_disabled_explicitly)
       keeps the smoke self-contained, mirroring the Rust lifecycle test. */
    vokra_s2s_duplex_t *duplex = NULL;
    st = vokra_s2s_duplex_open(session, /*deterministic=*/1, /*seed=*/0,
                               /*aec_disabled_explicitly=*/1, /*playback_offset=*/0, &duplex);
    if (st != VOKRA_OK || duplex == NULL) {
        fprintf(stderr, "smoke_s2s: FAIL duplex_open (%d): %s\n", (int)st, vokra_last_error());
        vokra_session_destroy(session);
        return 1;
    }

    size_t hop = 0;
    uint32_t rate = 0;
    if (vokra_s2s_frame_hop(duplex, &hop) != VOKRA_OK || hop == 0 ||
        vokra_s2s_sample_rate(duplex, &rate) != VOKRA_OK || rate == 0) {
        fprintf(stderr, "smoke_s2s: FAIL frame_hop/sample_rate (hop %zu, rate %u): %s\n", hop,
                rate, vokra_last_error());
        vokra_s2s_duplex_destroy(duplex);
        vokra_session_destroy(session);
        return 1;
    }
    printf("smoke_s2s: duplex hop %zu samples @ %u Hz\n", hop, rate);

    /* Attribution on an attribution-required model is non-empty and
       NUL-terminated (two-call sizing discipline). */
    size_t needed = 0;
    if (vokra_model_attribution(session, NULL, 0, &needed) != VOKRA_OK || needed <= 1) {
        fprintf(stderr, "smoke_s2s: FAIL attribution sizing (needed %zu): %s\n", needed,
                vokra_last_error());
        vokra_s2s_duplex_destroy(duplex);
        vokra_session_destroy(session);
        return 1;
    }
    char *attr = (char *)malloc(needed);
    if (!attr || vokra_model_attribution(session, attr, needed, &needed) != VOKRA_OK ||
        attr[needed - 1] != '\0') {
        fprintf(stderr, "smoke_s2s: FAIL attribution fetch: %s\n", vokra_last_error());
        free(attr);
        vokra_s2s_duplex_destroy(duplex);
        vokra_session_destroy(session);
        return 1;
    }
    printf("smoke_s2s: attribution = \"%s\"\n", attr);
    free(attr);

    /* Push mic frames and pull model frames through the C ABI. */
    float *in = (float *)malloc(hop * sizeof(float));
    float *out = (float *)malloc(hop * sizeof(float));
    if (!in || !out) {
        fprintf(stderr, "smoke_s2s: FAIL out of memory\n");
        free(in);
        free(out);
        vokra_s2s_duplex_destroy(duplex);
        vokra_session_destroy(session);
        return 1;
    }
    for (size_t i = 0; i < hop; i++) {
        in[i] = 0.2f * (float)(((i / 32) % 2) ? 1 : -1); /* deterministic square wave */
    }
    int rc = 0;
    size_t pulled = 0;
    for (int step = 0; step < 3; step++) {
        int emitted = 0;
        if (vokra_s2s_push_mic(duplex, in, hop, &emitted) != VOKRA_OK) {
            fprintf(stderr, "smoke_s2s: FAIL push_mic step %d: %s\n", step, vokra_last_error());
            rc = 1;
            break;
        }
        size_t len = 0;
        if (vokra_s2s_pull_audio(duplex, out, hop, &len) != VOKRA_OK) {
            fprintf(stderr, "smoke_s2s: FAIL pull_audio step %d: %s\n", step, vokra_last_error());
            rc = 1;
            break;
        }
        pulled += len;
    }
    free(in);
    free(out);
    if (rc == 0) {
        printf("smoke_s2s: pushed 3 mic frames, pulled %zu model samples\n", pulled);

        /* Inner-monologue text (two-call sizing discipline). */
        size_t tneeded = 0;
        if (vokra_s2s_text(duplex, NULL, 0, &tneeded) != VOKRA_OK || tneeded < 1) {
            fprintf(stderr, "smoke_s2s: FAIL text sizing (needed %zu): %s\n", tneeded,
                    vokra_last_error());
            rc = 1;
        } else {
            char *buf = (char *)malloc(tneeded);
            if (!buf || vokra_s2s_text(duplex, buf, tneeded, &tneeded) != VOKRA_OK ||
                buf[tneeded - 1] != '\0') {
                fprintf(stderr, "smoke_s2s: FAIL text fetch: %s\n", vokra_last_error());
                rc = 1;
            }
            free(buf);
        }
    }

    /* Cross-thread barge-in handle: create, fire, destroy (same thread here —
       the concurrency contract is exercised Rust-side; this is a call smoke). */
    if (rc == 0) {
        vokra_s2s_interrupt_t *interrupt = NULL;
        if (vokra_s2s_interrupt_handle(duplex, &interrupt) != VOKRA_OK || interrupt == NULL ||
            vokra_s2s_interrupt(interrupt) != VOKRA_OK) {
            fprintf(stderr, "smoke_s2s: FAIL interrupt handle/fire: %s\n", vokra_last_error());
            rc = 1;
        }
        vokra_s2s_interrupt_destroy(interrupt);
    }

    vokra_s2s_duplex_destroy(duplex);
    vokra_session_destroy(session);
    if (rc == 0) {
        printf("smoke_s2s: duplex leg PASS\n");
    }
    return rc;
}

int main(int argc, char **argv) {
    const char *permissive = argc > 1 ? argv[1] : DEFAULT_MODEL;

    printf("smoke_s2s: vokra %s\n", vokra_version());

    int rc = 0;

    /* Leg 1: NULL-argument error paths — no model needed. Valid out-pointers
       are passed so only the NULL handle is under test. */
    {
        size_t hop = 0;
        uint32_t rate = 0;
        int emitted = 0;
        size_t len = 0;
        size_t needed = 0;
        float pcm[8] = {0};
        char buf[8] = {0};
        vokra_s2s_duplex_t *dx = NULL;
        vokra_s2s_interrupt_t *it = NULL;

        rc |= expect_invalid(vokra_s2s_duplex_open(NULL, 0, 0, 0, 0, &dx), "duplex_open NULL session");
        rc |= expect_invalid(vokra_s2s_frame_hop(NULL, &hop), "frame_hop NULL duplex");
        rc |= expect_invalid(vokra_s2s_sample_rate(NULL, &rate), "sample_rate NULL duplex");
        rc |= expect_invalid(vokra_s2s_push_mic(NULL, pcm, 8, &emitted), "push_mic NULL duplex");
        rc |= expect_invalid(vokra_s2s_pull_audio(NULL, pcm, 8, &len), "pull_audio NULL duplex");
        rc |= expect_invalid(vokra_s2s_text(NULL, buf, sizeof buf, &needed), "text NULL duplex");
        rc |= expect_invalid(vokra_s2s_interrupt_handle(NULL, &it), "interrupt_handle NULL duplex");
        rc |= expect_invalid(vokra_s2s_interrupt(NULL), "interrupt NULL handle");
        rc |= expect_invalid(vokra_model_attribution(NULL, NULL, 0, &needed),
                             "model_attribution NULL session");

        /* destroy is NULL-tolerant (documented no-op, must not crash). */
        vokra_s2s_duplex_destroy(NULL);
        vokra_s2s_interrupt_destroy(NULL);

        if (dx != NULL || it != NULL) {
            fprintf(stderr, "smoke_s2s: FAIL error path left an out-handle non-NULL\n");
            rc = 1;
        }
    }
    if (rc == 0) {
        printf("smoke_s2s: NULL-argument paths all rejected loudly\n");
    }

    /* Leg 2: permissive committed model (Silero VAD) — real surfaces. */
    {
        vokra_session_t *session = NULL;
        vokra_status_t st = vokra_session_create_from_file(permissive, &session);
        if (st != VOKRA_OK || session == NULL) {
            fprintf(stderr, "smoke_s2s: FAIL create permissive session %s (%d): %s\n", permissive,
                    (int)st, vokra_last_error());
            return 1;
        }

        /* A permissive-license model carries no display obligation. */
        size_t needed = 12345; /* sentinel — must be overwritten to 0 */
        st = vokra_model_attribution(session, NULL, 0, &needed);
        if (st != VOKRA_OK || needed != 0) {
            fprintf(stderr, "smoke_s2s: FAIL permissive attribution (%d, needed %zu): %s\n",
                    (int)st, needed, vokra_last_error());
            vokra_session_destroy(session);
            return 1;
        }
        printf("smoke_s2s: permissive model reports no attribution obligation\n");

        /* No duplex engine on this model — the open must fail loudly. */
        vokra_s2s_duplex_t *dx = NULL;
        st = vokra_s2s_duplex_open(session, 0, 0, 0, 0, &dx);
        if (st == VOKRA_OK || dx != NULL) {
            fprintf(stderr, "smoke_s2s: FAIL non-duplex model was not rejected (%d)\n", (int)st);
            vokra_s2s_duplex_destroy(dx);
            vokra_session_destroy(session);
            return 1;
        }
        if (vokra_last_error() == NULL) {
            fprintf(stderr, "smoke_s2s: FAIL duplex_open failure set no error message\n");
            vokra_session_destroy(session);
            return 1;
        }
        printf("smoke_s2s: duplex_open on non-duplex model rejected: %s\n", vokra_last_error());

        vokra_session_destroy(session);
    }

    /* Leg 3: env-gated full duplex round trip. */
    {
        const char *moshi = getenv("VOKRA_MOSHI_GGUF");
        if (!moshi || moshi[0] == '\0') {
            printf("smoke_s2s: duplex leg SKIP (set VOKRA_MOSHI_GGUF to a moshi GGUF to run)\n");
        } else if (run_moshi_leg(moshi) != 0) {
            return 1;
        }
    }

    if (rc != 0) {
        return 1;
    }
    printf("smoke_s2s: PASS\n");
    return 0;
}
