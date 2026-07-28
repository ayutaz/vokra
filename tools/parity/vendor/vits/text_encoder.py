"""VITS ``TextEncoder`` (upstream ``models.py`` lines 135-176), extracted.

This target module hosts one class only — ``TextEncoder`` — extracted
verbatim from ``jaywalnut310/vits/models.py`` at the pinned commit noted
below. See the sibling ``README.md`` mapping table for why only this
class (not upstream ``models.py`` as a whole) is vendored here: the
Vokra parity path only needs the inference-only text encoder to dump
``phoneme_embed`` / ``text_hidden`` for
``crates/vokra-models/src/sbv2/text_encoder.rs``, and the rest of
upstream ``models.py`` (``StochasticDurationPredictor``,
``PosteriorEncoder``, discriminators, ``SynthesizerTrn``) is training-
side or reaches into ``monotonic_align`` (Cython training kernel) which
this vendor deliberately does NOT ship.
"""

# Copied from jaywalnut310/vits @ 2e561ba58618d021b5b8323d3765880f7e0ecfdb, MIT
# License (see the sibling ``LICENSE`` file). Upstream source:
# https://raw.githubusercontent.com/jaywalnut310/vits/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py
# — 19375 bytes, sha256 de1f89cfed83b5b345a6b90c3f987b8e390b95041f0ab466e5bf76cef94a0875.
# The class body below (`class TextEncoder(nn.Module):` ... through
# `return x, m, logs, x_mask`) is byte-identical to upstream ``models.py``
# lines 135-176; only the imports at the top of THIS file are new (they
# replace upstream ``models.py``'s file-level imports, restricted to the
# subset that ``TextEncoder`` actually references — ``attentions.Encoder``
# and ``commons.sequence_mask``, both via the sibling relative-imports).
#
# NOT REFERENCED: github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0),
# github.com/fishaudio/Bert-VITS2 (AGPL-3.0). Only ``jaywalnut310/vits``
# MIT source has been read or copied. See
# ``tools/parity/vendor/vits/README.md`` for the full attribution + clean-
# room rationale.

import math

import torch
from torch import nn

from . import attentions, commons


# ---8<--- upstream models.py lines 135-176, byte-identical below ---8<---
class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask
