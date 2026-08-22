/* Independent RNNoise v0.2 waveform oracle.
 *
 * Compile this file against the unmodified official v0.2 release sources. It
 * calls rnnoise_process_frame directly and therefore covers Xiph's own
 * high-pass, 32-band frontend, pitch search/remove_doubling, neural network,
 * delayed spectrum filter, and overlap-add synthesis in one pass.
 *
 *   cc -std=c99 -O0 -DDISABLE_NEON -I include -I src \
 *     /path/to/rnnoise_v02_waveform_reference.c \
 *     src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c src/celt_lpc.c \
 *     src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c \
 *     src/rnnoise_data.c src/rnnoise_tables.c -lm -o rnnoise-v02-waveform-reference
 *
 * DISABLE_NEON selects the architecture-independent portable Xiph kernels.
 * Each row is: frame index, VAD probability, then 16 deterministic PCM taps.
 * The input generator uses integer arithmetic plus exact power-of-two scaling
 * so the Rust parity test can reproduce it without a libm dependency.
 */

#include <stdint.h>
#include <stdio.h>

#include "rnnoise.h"

#define FRAME_SIZE 480
#define N_FRAMES 16
#define N_TAPS 16

static float sample_for(uint32_t *state, int frame, int index) {
  int global = frame * FRAME_SIZE + index;
  int phase = global % 240;
  int triangle = phase < 120 ? phase - 60 : 180 - phase;
  int carrier = global % 16 < 8 ? 10000 : -10000;
  int32_t noise;
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  noise = (int32_t)((*state >> 16) & UINT32_C(0xffff)) - 32768;
  if (frame == 0) return 0.f;
  if (frame < 10)
    return (float)(triangle * 320 + carrier + noise / 16) / 32768.f;
  return (float)(noise / 2) / 32768.f;
}

int main(void) {
  DenoiseState *state = rnnoise_create(NULL);
  uint32_t rng = UINT32_C(0x6d2b79f5);
  int frame;
  if (state == NULL) {
    fprintf(stderr, "rnnoise_create failed\n");
    return 2;
  }
  for (frame = 0; frame < N_FRAMES; ++frame) {
    float input[FRAME_SIZE];
    float output[FRAME_SIZE];
    float vad;
    int index;
    for (index = 0; index < FRAME_SIZE; ++index)
      input[index] = sample_for(&rng, frame, index);
    vad = rnnoise_process_frame(state, output, input);
    printf("%d,%.9g", frame, vad);
    for (index = 0; index < N_TAPS; ++index) {
      int tap = (index * 73 + 19) % FRAME_SIZE;
      printf(",%.9g", output[tap]);
    }
    printf("\n");
  }
  rnnoise_destroy(state);
  return 0;
}
