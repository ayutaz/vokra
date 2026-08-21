/* Independent RNNoise v0.2 network oracle.
 *
 * Compile this file against the unmodified official v0.2 release sources.
 * It calls Xiph's init_rnnoise/compute_rnn directly; it does not mirror the
 * Rust implementation.  Each CSV row is 65 input features, 32 gain outputs,
 * and one VAD output.  Nine significant digits round-trip every f32 input.
 *
 *   cc -std=c99 -O0 -Isrc \
 *     /path/to/rnnoise_v02_network_reference.c \
 *     src/rnn.c src/rnnoise_data.c src/nnet.c src/nnet_default.c \
 *     src/parse_lpcnet_weights.c -lm -o rnnoise-v02-network-reference
 */

#include <stdio.h>
#include <string.h>

#include "denoise.h"
#include "rnn.h"

int main(void) {
  RNNoise model;
  RNNState state;
  int frame;
  memset(&model, 0, sizeof(model));
  memset(&state, 0, sizeof(state));
  if (init_rnnoise(&model, rnnoise_arrays) != 0) {
    fprintf(stderr, "init_rnnoise failed\n");
    return 2;
  }

  for (frame = 0; frame < 4; ++frame) {
    float features[65];
    float gains[32];
    float vad;
    int index;
    for (index = 0; index < 65; ++index) {
      int raw = (index * 37 + frame * 17 + frame * index * 3) % 101;
      features[index] = frame == 0 ? 0.f : (raw - 50) / 25.f;
    }
    compute_rnn(&model, &state, gains, &vad, features, 0);
    for (index = 0; index < 65; ++index) printf("%.9g,", features[index]);
    for (index = 0; index < 32; ++index) printf("%.9g,", gains[index]);
    printf("%.9g\n", vad);
  }
  return 0;
}
