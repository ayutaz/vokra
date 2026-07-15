/*
 * smoke_aec.c — C smoke test for the Vokra AEC API (M4-03, FR-OP-60).
 *
 * Exercises the acoustic-echo-cancellation C ABI end to end from C, using
 * only <vokra.h>:
 *   create (canceller + far-end writer) -> ref_push -> process -> reset ->
 *   destroy.
 *
 * The AEC surface is **model-free** — it runs on raw little-endian float32
 * PCM with no GGUF — so this test needs no fixture and no env gate. The
 * synthetic far-end is a deterministic SplitMix64 stream and the near-end is
 * a pure two-tap echo of it (no near-end speech), mirroring the Rust
 * `round_trip_cancels_echo` fixture (crates/vokra-capi/src/aec.rs) so the
 * echo-energy shrink bound below holds on the same input. The check is an
 * end-to-end sanity of the C surface, not a numerical parity gate — those
 * live in tests/parity/aec.
 *
 * Also covers the FR-EX-08 loud-error contract across the boundary: NULL /
 * invalid arguments must return VOKRA_ERROR_INVALID_ARGUMENT with a message
 * on vokra_last_error(), never a crash or a silent degrade.
 *
 * Usage: smoke_aec        (no arguments; the AEC needs no model)
 * Exit code: 0 = pass, 1 = fail.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vokra.h"

#define SAMPLE_RATE 16000u
#define FRAME 64      /* frame_size (even, > 0) */
#define FILTER 256    /* filter_length (echo tail, >= frame_size) */
#define FRAMES 200    /* frames processed in the round trip */

/* SplitMix64 step (test-local), matching the Rust AEC round-trip fixture so
   the deterministic far-end stream — and therefore the echo-shrink bound —
   is the same one that test proves out. */
static float next_farend(uint64_t *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    double r = (double)z / (double)UINT64_MAX;
    return (float)((r - 0.5) * 0.5);
}

/* Reports whether `st` is the expected INVALID_ARGUMENT loud error; returns 1
   on mismatch (so callers can OR failures together). */
static int expect_invalid(vokra_status_t st, const char *what) {
    if (st != VOKRA_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "smoke_aec: FAIL %s: expected INVALID_ARGUMENT, got %d (%s)\n",
                what, (int)st, vokra_last_error());
        return 1;
    }
    return 0;
}

/* Reports whether the thread-local last error mentions `needle`. */
static int err_mentions(const char *needle) {
    const char *msg = vokra_last_error();
    if (!msg || !strstr(msg, needle)) {
        fprintf(stderr, "smoke_aec: FAIL last error \"%s\" must mention \"%s\"\n",
                msg ? msg : "(null)", needle);
        return 1;
    }
    return 0;
}

int main(void) {
    printf("smoke_aec: vokra %s\n", vokra_version());

    vokra_aec_config_t config = {
        .sample_rate = SAMPLE_RATE,
        .frame_size = FRAME,
        .filter_length = FILTER,
        .ref_queue_capacity_samples = 0, /* default = 8 * filter_length */
    };
    const size_t n = FRAME;
    const size_t total = (size_t)FRAMES * FRAME;

    float *farend = (float *)malloc(total * sizeof(float));
    float *mic = (float *)malloc(total * sizeof(float));
    float out[FRAME];
    if (!farend || !mic) {
        fprintf(stderr, "smoke_aec: FAIL out of memory\n");
        free(farend);
        free(mic);
        return 1;
    }

    /* Deterministic far-end noise; near-end mic = a pure two-tap echo of it. */
    uint64_t state = 0x123456789ABCDEF0ULL;
    for (size_t i = 0; i < total; i++) {
        farend[i] = next_farend(&state);
    }
    for (size_t i = 0; i < total; i++) {
        size_t j2 = i >= 2 ? i - 2 : 0;
        size_t j9 = i >= 9 ? i - 9 : 0;
        mic[i] = 0.5f * farend[j2] - 0.25f * farend[j9];
    }

    /* create: one canceller handle + one far-end writer handle. */
    vokra_aec_t *aec = NULL;
    vokra_aec_ref_writer_t *writer = NULL;
    vokra_status_t st = vokra_aec_create(&config, &aec, &writer);
    if (st != VOKRA_OK || aec == NULL || writer == NULL) {
        fprintf(stderr, "smoke_aec: FAIL create (%d): %s\n", (int)st, vokra_last_error());
        free(farend);
        free(mic);
        return 1;
    }

    /* Round trip: push far-end for frame f, then cancel mic frame f; the
       far-end window is fully covered so every frame reports CANCELLED. */
    double early = 0.0;
    double late = 0.0;
    for (size_t f = 0; f < FRAMES; f++) {
        uint64_t pos = (uint64_t)(f * n);

        size_t accepted = 0;
        st = vokra_aec_ref_push(writer, farend + f * n, n, pos, &accepted);
        if (st != VOKRA_OK || accepted != n) {
            fprintf(stderr, "smoke_aec: FAIL ref_push frame %zu (%d, accepted %zu): %s\n",
                    f, (int)st, accepted, vokra_last_error());
            goto fail;
        }

        vokra_aec_status_t frame_status = VOKRA_AEC_RESET;
        size_t missing = (size_t)-1;
        st = vokra_aec_process(aec, mic + f * n, pos, out, n, &frame_status, &missing);
        if (st != VOKRA_OK) {
            fprintf(stderr, "smoke_aec: FAIL process frame %zu (%d): %s\n", f, (int)st,
                    vokra_last_error());
            goto fail;
        }
        if (frame_status != VOKRA_AEC_CANCELLED || missing != 0) {
            fprintf(stderr, "smoke_aec: FAIL frame %zu status %d missing %zu "
                            "(expected CANCELLED, 0)\n",
                    f, (int)frame_status, missing);
            goto fail;
        }

        double energy = 0.0;
        for (size_t i = 0; i < n; i++) {
            energy += (double)out[i] * (double)out[i];
        }
        if (f >= 10 && f < 40) {
            early += energy;
        }
        if (f >= 160 && f < 200) {
            late += energy;
        }
    }
    if (!(late < 0.5 * early)) {
        fprintf(stderr, "smoke_aec: FAIL echo did not shrink: early %e late %e\n", early, late);
        goto fail;
    }
    printf("smoke_aec: %d frames cancelled, echo energy early %e -> late %e\n", FRAMES, early,
           late);

    /* reset: returns to the as-new state. */
    st = vokra_aec_reset(aec);
    if (st != VOKRA_OK) {
        fprintf(stderr, "smoke_aec: FAIL reset (%d): %s\n", (int)st, vokra_last_error());
        goto fail;
    }

    /* Error paths (FR-EX-08): every bad argument is a loud INVALID_ARGUMENT,
       never a crash or silent degrade. Failures are OR'd so all are reported. */
    int bad = 0;

    /* create: NULL config, and configs that fail validation. */
    vokra_aec_t *a2 = NULL;
    vokra_aec_ref_writer_t *w2 = NULL;
    bad |= expect_invalid(vokra_aec_create(NULL, &a2, &w2), "create NULL config");
    bad |= err_mentions("config");
    if (a2 != NULL || w2 != NULL) {
        fprintf(stderr, "smoke_aec: FAIL create failure left out-params non-NULL\n");
        bad = 1;
    }
    vokra_aec_config_t zero_rate = config;
    zero_rate.sample_rate = 0;
    bad |= expect_invalid(vokra_aec_create(&zero_rate, &a2, &w2), "create zero sample_rate");
    vokra_aec_config_t odd_frame = config;
    odd_frame.frame_size = FRAME - 1; /* odd */
    bad |= expect_invalid(vokra_aec_create(&odd_frame, &a2, &w2), "create odd frame_size");

    /* ref_push: NULL writer / NULL pcm with len>0 / NULL out_accepted /
       backward playback_pos. */
    size_t accepted = 0;
    bad |= expect_invalid(vokra_aec_ref_push(NULL, farend, n, 0, &accepted), "ref_push NULL writer");
    bad |= expect_invalid(vokra_aec_ref_push(writer, NULL, n, 0, &accepted), "ref_push NULL pcm");
    bad |= expect_invalid(vokra_aec_ref_push(writer, farend, n, 0, NULL), "ref_push NULL out_accepted");
    bad |= err_mentions("out_accepted");
    /* The writer already advanced past the round trip; a low pos is backward. */
    bad |= expect_invalid(vokra_aec_ref_push(writer, farend, n, 500, &accepted), "ref_push backward pos");

    /* process: NULL aec / NULL out / NULL out_status / wrong length / zero length. */
    vokra_aec_status_t frame_status = VOKRA_AEC_CANCELLED;
    size_t missing = 0;
    bad |= expect_invalid(vokra_aec_process(NULL, mic, 0, out, n, &frame_status, &missing),
                          "process NULL aec");
    bad |= expect_invalid(vokra_aec_process(aec, mic, 0, NULL, n, &frame_status, &missing),
                          "process NULL out");
    bad |= err_mentions("out");
    bad |= expect_invalid(vokra_aec_process(aec, mic, 0, out, n, NULL, &missing),
                          "process NULL out_status");
    bad |= err_mentions("out_status");
    bad |= expect_invalid(vokra_aec_process(aec, mic, 0, out, n / 2, &frame_status, NULL),
                          "process wrong length");
    bad |= expect_invalid(vokra_aec_process(aec, mic, 0, out, 0, &frame_status, NULL),
                          "process zero length");

    /* reset: NULL handle. */
    bad |= expect_invalid(vokra_aec_reset(NULL), "reset NULL aec");

    if (bad) {
        goto fail;
    }
    printf("smoke_aec: invalid-argument paths all rejected loudly\n");

    /* destroy: real handles, then NULL (documented no-op) — order is free. */
    vokra_aec_destroy(aec);
    vokra_aec_ref_writer_destroy(writer);
    vokra_aec_destroy(NULL);
    vokra_aec_ref_writer_destroy(NULL);

    free(farend);
    free(mic);
    printf("smoke_aec: PASS\n");
    return 0;

fail:
    vokra_aec_destroy(aec);
    vokra_aec_ref_writer_destroy(writer);
    free(farend);
    free(mic);
    return 1;
}
