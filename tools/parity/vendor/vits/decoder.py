"""VITS ``Generator`` (HiFi-GAN, upstream ``models.py`` lines 244-296).

The HiFi-GAN generator VITS uses as its neural vocoder. Extracted here
— rather than importing the whole upstream ``models.py`` — for the same
reason ``text_encoder.py`` / ``flow.py`` do: this vendor deliberately
does NOT ship the training-side pieces of ``models.py``
(``StochasticDurationPredictor``, ``PosteriorEncoder``,
``DiscriminatorP`` / ``DiscriminatorS`` / ``MultiPeriodDiscriminator``,
``SynthesizerTrn``) nor the ``monotonic_align`` Cython kernel they
reach into. The Rust side (``crates/vokra-models/src/sbv2/decoder.rs``)
already reuses ``vokra-ops::hifigan`` at ~100% per design doc §7's
"既存資産の流用度" table — this vendored reference exists purely so
``sbv2_dump_reference.py`` can dump the intermediate/final tensors for
diffing (design doc §10 ``waveform`` row), not because Rust needs new
logic.
"""

# Copied from jaywalnut310/vits @ 2e561ba58618d021b5b8323d3765880f7e0ecfdb, MIT
# License (see the sibling ``LICENSE`` file). Upstream source:
# https://raw.githubusercontent.com/jaywalnut310/vits/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py
# — 19375 bytes, sha256 de1f89cfed83b5b345a6b90c3f987b8e390b95041f0ab466e5bf76cef94a0875.
# The class body below (``class Generator(torch.nn.Module):`` ... through
# ``remove_weight_norm`` method) is byte-identical to upstream ``models.py``
# lines 244-296; only the imports at the top of THIS file are new (they
# replace upstream ``models.py``'s file-level imports, restricted to what
# ``Generator`` actually references — ``Conv1d`` / ``ConvTranspose1d`` and
# ``weight_norm`` / ``remove_weight_norm`` from ``torch``, ``F.leaky_relu``
# / ``torch.tanh``, plus ``modules`` for ``ResBlock1`` / ``ResBlock2`` /
# ``LRELU_SLOPE`` and ``commons.init_weights`` via the sibling relative-
# imports).
#
# NOTE on torch API drift: upstream (2021-06-14) uses ``torch.nn.utils.
# weight_norm``, which is DeprecationWarning-flagged (though still
# functional at inference time) in torch >= 2.1 in favor of
# ``torch.nn.utils.parametrizations.weight_norm``. Left as-is so this
# stays a verbatim vendor — the DeprecationWarning noise at import time
# is expected and not a bug.
#
# NOT REFERENCED: github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0),
# github.com/fishaudio/Bert-VITS2 (AGPL-3.0). Only ``jaywalnut310/vits``
# MIT source has been read or copied. See
# ``tools/parity/vendor/vits/README.md`` for the full attribution + clean-
# room rationale.

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from . import modules
from .commons import init_weights


# ---8<--- upstream models.py lines 244-296, byte-identical below ---8<---
class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
