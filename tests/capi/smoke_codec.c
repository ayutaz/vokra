/*
 * smoke_codec.c — C-caller coverage for the generic streaming codec decoder
 * (#48). The committed non-codec GGUF pins loud open failure and NULL safety;
 * VOKRA_MIMI_GGUF enables one real all-zero code-frame -> PCM round trip.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vokra.h"

static int non_codec_and_null_paths(const char *non_codec_model) {
    vokra_session_t *session = NULL;
    enum vokra_status_t st = vokra_session_create_from_file(non_codec_model, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_codec: non-codec fixture failed to load: %d (%s)\n", (int)st,
                vokra_last_error());
        return 1;
    }
    vokra_codec_decoder_t *decoder = vokra_codec_decoder_open(session);
    if (decoder != NULL) {
        fprintf(stderr, "smoke_codec: non-codec session unexpectedly opened decoder\n");
        vokra_codec_decoder_destroy(decoder);
        vokra_session_destroy(session);
        return 1;
    }
    vokra_session_destroy(session);

    if (vokra_codec_decoder_open(NULL) != NULL ||
        vokra_codec_decoder_frame_hop(NULL) != -1 ||
        vokra_codec_decoder_sample_rate(NULL) != -1 ||
        vokra_codec_decoder_n_codebooks(NULL) != -1) {
        fprintf(stderr, "smoke_codec: NULL constructor/query contract failed\n");
        return 1;
    }
    int32_t emitted = -1;
    size_t written = (size_t)-1;
    const uint32_t code = 0;
    float pcm = 0.0f;
    if (vokra_codec_decoder_push_codes(NULL, &code, 1, &emitted) !=
            VOKRA_ERROR_INVALID_ARGUMENT ||
        vokra_codec_decoder_pull_pcm(NULL, &pcm, 1, &written) !=
            VOKRA_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "smoke_codec: NULL push/pull did not fail invalid-argument\n");
        return 1;
    }
    vokra_codec_decoder_reset(NULL);
    vokra_codec_decoder_destroy(NULL);
    return 0;
}

static int real_mimi_roundtrip(const char *path) {
    vokra_session_t *session = NULL;
    enum vokra_status_t st = vokra_session_create_from_file(path, &session);
    if (st != VOKRA_OK || session == NULL) {
        fprintf(stderr, "smoke_codec: Mimi session failed to load: %d (%s)\n", (int)st,
                vokra_last_error());
        return 1;
    }
    vokra_codec_decoder_t *decoder = vokra_codec_decoder_open(session);
    /* The decoder retains the model independently. */
    vokra_session_destroy(session);
    if (decoder == NULL) {
        fprintf(stderr, "smoke_codec: Mimi decoder open failed: %s\n", vokra_last_error());
        return 1;
    }

    int32_t n_codebooks = vokra_codec_decoder_n_codebooks(decoder);
    int32_t frame_hop = vokra_codec_decoder_frame_hop(decoder);
    int32_t sample_rate = vokra_codec_decoder_sample_rate(decoder);
    if (n_codebooks <= 0 || frame_hop <= 0 || sample_rate <= 0) {
        fprintf(stderr, "smoke_codec: invalid model axes n=%d hop=%d rate=%d\n", n_codebooks,
                frame_hop, sample_rate);
        vokra_codec_decoder_destroy(decoder);
        return 1;
    }

    uint32_t *codes = calloc((size_t)n_codebooks, sizeof(*codes));
    float *pcm = calloc((size_t)frame_hop, sizeof(*pcm));
    if (codes == NULL || pcm == NULL) {
        fprintf(stderr, "smoke_codec: allocation failed\n");
        free(codes);
        free(pcm);
        vokra_codec_decoder_destroy(decoder);
        return 1;
    }

    int32_t emitted = 0;
    size_t written = 0;
    st = vokra_codec_decoder_push_codes(decoder, codes, (size_t)n_codebooks, &emitted);
    if (st != VOKRA_OK || emitted != 1) {
        fprintf(stderr, "smoke_codec: push failed: %d emitted=%d (%s)\n", (int)st, emitted,
                vokra_last_error());
        free(codes);
        free(pcm);
        vokra_codec_decoder_destroy(decoder);
        return 1;
    }
    st = vokra_codec_decoder_pull_pcm(decoder, pcm, (size_t)frame_hop, &written);
    if (st != VOKRA_OK || written != (size_t)frame_hop) {
        fprintf(stderr, "smoke_codec: pull failed: %d written=%zu (%s)\n", (int)st, written,
                vokra_last_error());
        free(codes);
        free(pcm);
        vokra_codec_decoder_destroy(decoder);
        return 1;
    }

    vokra_codec_decoder_reset(decoder);
    vokra_codec_decoder_destroy(decoder);
    free(codes);
    free(pcm);
    printf("smoke_codec: PASS (%d codebooks, %d samples/frame, %d Hz)\n", n_codebooks,
           frame_hop, sample_rate);
    return 0;
}

int main(int argc, char **argv) {
    const char *non_codec_model = argc > 1 ? argv[1] :
        "tests/parity/silero_vad/silero-vad-v5.gguf";
    if (non_codec_and_null_paths(non_codec_model) != 0) {
        return 1;
    }

    const char *mimi = getenv("VOKRA_MIMI_GGUF");
    if (mimi == NULL || *mimi == '\0') {
        printf("smoke_codec: SKIP real Mimi leg (set VOKRA_MIMI_GGUF)\n");
        return 0;
    }
    return real_mimi_roundtrip(mimi);
}
