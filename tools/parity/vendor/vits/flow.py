"""VITS ``ResidualCouplingBlock`` (upstream ``models.py`` lines 179-209).

The stacked normalizing-flow head that VITS wraps around
``modules.ResidualCouplingLayer`` (that layer itself is re-exported from
the sibling ``coupling.py``). Extracted here — rather than importing the
whole upstream ``models.py`` — for the same reason ``text_encoder.py``
does: this vendor deliberately does NOT ship the training-side pieces of
``models.py`` (``StochasticDurationPredictor``, ``PosteriorEncoder``,
discriminators, ``SynthesizerTrn``) nor the ``monotonic_align`` Cython
kernel they reach into. See the sibling ``README.md`` mapping table for
the full rationale + which ``crates/vokra-models/src/sbv2/flow.rs``
tensor this feeds (``z_latent``).
"""

# Copied from jaywalnut310/vits @ 2e561ba58618d021b5b8323d3765880f7e0ecfdb, MIT
# License (see the sibling ``LICENSE`` file). Upstream source:
# https://raw.githubusercontent.com/jaywalnut310/vits/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py
# — 19375 bytes, sha256 de1f89cfed83b5b345a6b90c3f987b8e390b95041f0ab466e5bf76cef94a0875.
# The class body below (``class ResidualCouplingBlock(nn.Module):`` ...
# through ``return x``) is byte-identical to upstream ``models.py`` lines
# 179-209; only the imports at the top of THIS file are new (they replace
# upstream ``models.py``'s file-level imports, restricted to what
# ``ResidualCouplingBlock`` actually references — ``modules`` for
# ``ResidualCouplingLayer`` and ``Flip``).
#
# NOT REFERENCED: github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0),
# github.com/fishaudio/Bert-VITS2 (AGPL-3.0). Only ``jaywalnut310/vits``
# MIT source has been read or copied. See
# ``tools/parity/vendor/vits/README.md`` for the full attribution + clean-
# room rationale.

from torch import nn

from . import modules


# ---8<--- upstream models.py lines 179-209, byte-identical below ---8<---
class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x
