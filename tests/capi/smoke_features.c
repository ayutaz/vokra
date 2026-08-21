/*
 * smoke_features.c — C smoke test for streaming continuous speech features
 * (#49: vokra_feat_*), using only <vokra.h>.
 *
 * The committed Silero model covers the fail-closed family mismatch. Set
 * VOKRA_MOSHI_GGUF to run the native Mimi encoder happy path as well.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vokra.h"

static const char *DEFAULT_MODEL = "tests/parity/silero_vad/silero-vad-v5.gguf";

static int expect_invalid(vokra_status_t status, const char *what) {
    if (status != VOKRA_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "smoke_features: FAIL %s: expected INVALID_ARGUMENT, got %d (%s)\n",
                what, (int)status, vokra_last_error());
        return 1;
    }
    return 0;
}

static int run_moshi_leg(const char *model) {
    vokra_session_t *session = NULL;
    vokra_status_t status = vokra_session_create_from_file(model, &session);
    if (status != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_features: FAIL create Moshi session (%d): %s\n", (int)status,
                vokra_last_error());
        return 1;
    }

    vokra_feat_t *feat = vokra_feat_open(session);
    if (feat == NULL) {
        fprintf(stderr, "smoke_features: FAIL feat_open: %s\n", vokra_last_error());
        vokra_session_destroy(session);
        return 1;
    }

    const int32_t rate_mhz = vokra_feat_frame_rate_mhz(feat);
    const int32_t dim = vokra_feat_dim(feat);
    if (rate_mhz != 25000 || dim <= 0) {
        fprintf(stderr, "smoke_features: FAIL geometry: rate_mhz=%d dim=%d (%s)\n",
                (int)rate_mhz, (int)dim, vokra_last_error());
        vokra_feat_destroy(feat);
        vokra_session_destroy(session);
        return 1;
    }

    /* Released Moshi/Mimi uses a 1,920-sample token frame at 24 kHz. One
       token frame emits two continuous 25 Hz feature rows. */
    float pcm[1920] = {0};
    if ((size_t)dim > SIZE_MAX / 2 / sizeof(float)) {
        fprintf(stderr, "smoke_features: FAIL feature dimension overflows allocation\n");
        vokra_feat_destroy(feat);
        vokra_session_destroy(session);
        return 1;
    }
    const size_t out_cap = (size_t)dim * 2;
    float *out = (float *)malloc(out_cap * sizeof(float));
    if (out == NULL) {
        fprintf(stderr, "smoke_features: FAIL out of memory\n");
        vokra_feat_destroy(feat);
        vokra_session_destroy(session);
        return 1;
    }

    size_t frames = SIZE_MAX;
    int64_t start_sample = -2;
    status = vokra_feat_push_pcm(feat, pcm, sizeof pcm / sizeof pcm[0]);
    if (status == VOKRA_OK) {
        status = vokra_feat_pull(feat, out, out_cap, &frames, &start_sample);
    }
    if (status != VOKRA_OK || frames != 2 || start_sample != 0) {
        fprintf(stderr,
                "smoke_features: FAIL first push/pull: status=%d frames=%zu start=%lld (%s)\n",
                (int)status, frames, (long long)start_sample, vokra_last_error());
        free(out);
        vokra_feat_destroy(feat);
        vokra_session_destroy(session);
        return 1;
    }

    vokra_feat_reset(feat);
    frames = SIZE_MAX;
    start_sample = -2;
    status = vokra_feat_push_pcm(feat, pcm, sizeof pcm / sizeof pcm[0]);
    if (status == VOKRA_OK) {
        status = vokra_feat_pull(feat, out, out_cap, &frames, &start_sample);
    }
    if (status != VOKRA_OK || frames != 2 || start_sample != 0) {
        fprintf(stderr,
                "smoke_features: FAIL reset timestamp: status=%d frames=%zu start=%lld (%s)\n",
                (int)status, frames, (long long)start_sample, vokra_last_error());
        free(out);
        vokra_feat_destroy(feat);
        vokra_session_destroy(session);
        return 1;
    }

    printf("smoke_features: Moshi leg PASS (%d mHz, dim %d)\n", (int)rate_mhz, (int)dim);
    free(out);
    vokra_feat_destroy(feat);
    vokra_session_destroy(session);
    return 0;
}

int main(int argc, char **argv) {
    const char *non_feature_model = argc > 1 ? argv[1] : DEFAULT_MODEL;
    int rc = 0;

    size_t frames = 77;
    int64_t start_sample = 88;
    float one = 0.0f;
    if (vokra_feat_open(NULL) != NULL || vokra_feat_frame_rate_mhz(NULL) != -1 ||
        vokra_feat_dim(NULL) != -1) {
        fprintf(stderr, "smoke_features: FAIL NULL metadata/open contract\n");
        rc = 1;
    }
    rc |= expect_invalid(vokra_feat_push_pcm(NULL, &one, 1), "push NULL handle");
    rc |= expect_invalid(vokra_feat_pull(NULL, &one, 1, &frames, &start_sample),
                         "pull NULL handle");
    if (frames != 77 || start_sample != 88) {
        fprintf(stderr, "smoke_features: FAIL rejected pull modified scalar outputs\n");
        rc = 1;
    }
    vokra_feat_reset(NULL);
    vokra_feat_destroy(NULL);

    vokra_session_t *session = NULL;
    vokra_status_t status = vokra_session_create_from_file(non_feature_model, &session);
    if (status != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_features: FAIL create non-feature session (%d): %s\n",
                (int)status, vokra_last_error());
        return 1;
    }
    vokra_feat_t *feat = vokra_feat_open(session);
    if (feat != NULL || vokra_last_error() == NULL) {
        fprintf(stderr, "smoke_features: FAIL non-feature model did not fail closed\n");
        vokra_feat_destroy(feat);
        vokra_session_destroy(session);
        return 1;
    }
    vokra_session_destroy(session);

    const char *moshi = getenv("VOKRA_MOSHI_GGUF");
    if (moshi == NULL || moshi[0] == '\0') {
        printf("smoke_features: Moshi leg SKIP (set VOKRA_MOSHI_GGUF to run)\n");
    } else if (run_moshi_leg(moshi) != 0) {
        return 1;
    }

    if (rc != 0) {
        return 1;
    }
    printf("smoke_features: PASS\n");
    return 0;
}
