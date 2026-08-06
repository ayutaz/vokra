# Copied from jaywalnut310/vits @ 2e561ba58618d021b5b8323d3765880f7e0ecfdb, MIT
# License (see the sibling ``LICENSE`` file in this directory). Upstream source:
# https://raw.githubusercontent.com/jaywalnut310/vits/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py
# — extracted lines 17-93 (the ``StochasticDurationPredictor`` class only),
# 3214 bytes of that extract, sha256
# 119f793b02fda3b51db5f92e801d3d62c48e3af5d0ea0b0fa6ab74eacec2fd26.
#
# ---8<--- IMPORT ADAPTATIONS (only lines diverging from upstream) ---8<---
# Upstream ``models.py`` executes with the ``jaywalnut310/vits`` repo root on
# ``sys.path`` and therefore imports its siblings as bare top-level modules
# (``import commons`` / ``import modules``). That assumption does not hold once
# the class is extracted into ``tools/parity/vendor/vits/sdp.py``, so the two
# lines below are rewritten to the ``vendor.vits`` namespace, exactly like the
# already-vendored ``text_encoder.py`` / ``flow.py`` / ``decoder.py`` did with
# their own extracted classes:
#
#   upstream:  ``import commons``
#   vendored:  ``from . import commons``  (not referenced in the class body,
#              but kept for parity with upstream's top-level imports and to
#              match the sibling target files' pattern)
#
#   upstream:  ``import modules``
#   vendored:  ``from . import modules``
#
# Upstream also unconditionally imports ``attentions`` and ``monotonic_align`` at
# module scope. Neither is referenced by the ``StochasticDurationPredictor``
# class body extracted below; ``monotonic_align`` is a Cython training kernel
# that would hard-fail to import in the parity venv (which does not build it).
# Dropped as unused adaptations to keep this file's import phase clean:
#
#   upstream:  ``import attentions``
#   vendored:  (dropped; unreferenced in the class body)
#
#   upstream:  ``import monotonic_align``
#   vendored:  (dropped; unreferenced in the class body, would hard-fail import)
#
# Every other line below (indentation, docstrings, trailing whitespace, and the
# entire class body) is byte-identical to that upstream range.
# See tools/parity/vendor/vits/README.md for the full attribution + clean-room
# rationale.
#
# NOT REFERENCED: github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0),
# github.com/fishaudio/Bert-VITS2 (AGPL-3.0). Only ``jaywalnut310/vits``
# MIT source has been read or copied.
#
# ---8<--- upstream ``models.py`` lines 17-93 verbatim (except the 2 adapted +
# 2 dropped imports above) ---8<---
import math
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from . import commons
from . import modules
from .commons import init_weights, get_padding


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw
