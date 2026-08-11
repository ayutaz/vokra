"""SBV2 v2 VITS2 flow (clean-room composition, MIT).

Blocker 8 (2026-08-06). The sibling ``flow.py`` ships jaywalnut310/vits
(MIT) ``ResidualCouplingBlock`` — a WN-based coupling. That module does
NOT load the SBV2 v2 base checkpoint's ``flow.*`` state_dict (108
missing tensors on ``strict=False`` load: the SBV2 checkpoint carries
transformer-encoder weights, ``enc.attn_layers.*`` / ``enc.norm_layers_1.*``
/ ``enc.ffn_layers.*`` / ``enc.spk_emb_linear.*``, not WN's
``enc.in_layers.*`` / ``enc.res_skip_layers.*``).

This file adds one clean-room composition —
``Sbv2TransformerCouplingBlock`` — that DOES load those weights,
assembled from primitives already vendored under
``tools/parity/vendor/vits/`` from ``jaywalnut310/vits`` (MIT):

  - ``attentions.MultiHeadAttention`` (relative-position, window_size=4,
    heads_share=True) — provides ``attn_layers.<i>.conv_{q,k,v,o}.*``
    and ``attn_layers.<i>.emb_rel_{k,v}``.
  - ``attentions.FFN`` (non-causal, kernel_size=5) — provides
    ``ffn_layers.<i>.conv_{1,2}.*``.
  - ``modules.LayerNorm`` — provides ``norm_layers_{1,2}.<i>.{gamma,beta}``.
  - ``modules.Flip`` — provides the parameter-free ``flow.flows.1/3/5/7``
    slots interleaved between coupling layers.

Only the outer three classes below (``Sbv2FlowEncoder`` /
``Sbv2TransformerCouplingLayer`` / ``Sbv2TransformerCouplingBlock``) are
new; their internal per-block sub-modules resolve to the already-
vendored jaywalnut310 (MIT) primitives above, byte-identical to what the
sibling ``attentions.py`` / ``modules.py`` ship. No architectural
novelty is invented here — this is a re-composition of published MIT
primitives whose shapes were dictated by the real SBV2 v2 base
checkpoint's tensor-name manifest (``litagin/Style-Bert-VITS2-2.0-base-
JP-Extra`` — a public data artifact; the safetensors tensor-name +
shape listing, never the AGPL Python code).

Design pattern for the flat ``nn.ModuleList([layer, Flip]*n_flows)``
outer stack (walked forward for training, ``reversed(self.flows)`` for
inference) is common to both
``jaywalnut310/vits/models.ResidualCouplingBlock`` (already vendored as
``flow.py``) and ``p0p4k/vits2_pytorch/models.
ResidualCouplingTransformersBlock`` at pinned commit
``1f4f3790568180f8dec4419d5cad5d0877b034bb``, whose ``LICENSE`` is
byte-identical to jaywalnut310/vits (sha256
``3d8165162cef96f686f02146ac2e4ae80db5797296a99c658befa424ee64727b``,
verified via a curl fetch on 2026-08-06 — the ``LICENSE`` file this
directory already ships covers both authors). None of p0p4k's five
``TransformerCoupling*Layer`` variants is layout-compatible with SBV2
v2 base's checkpoint (verified 2026-08-06 by safetensors introspection
of ``/tmp/sbv2-fixtures/sbv2-prep/G_0.safetensors``) — so this file
does not vendor p0p4k source; it only shares the "flat stack" design
pattern that both p0p4k and vanilla VITS already use.

The exact one-time-per-block ``spk_emb_linear`` application point (Rust
``crates/vokra-models/src/sbv2/flow.rs``'s ``SbV2TransformerCouplingLayer``
``inverse`` step 2: ``h = h + self.spk_emb_linear(g.mT).mT``) was
derived directly from the tensor-name manifest — the checkpoint carries
a single ``enc.spk_emb_linear.weight [D_MODEL, D_SPEAKER]`` per
coupling layer with no per-position or per-transformer-layer variant,
which forces the one-projection-per-block application point.

NOT REFERENCED (repeat the clean-room contract):
- ``github.com/litagin02/Style-Bert-VITS2`` (AGPL-3.0)
- ``github.com/fishaudio/Bert-VITS2`` (AGPL-3.0)
- Any community fork/blog-post code excerpt of either of the above.

See ``tools/parity/vendor/vits/README.md`` for the full permissive-
reference allowlist + attribution table.
"""

# Copyright (c) 2021 Jaehyeon Kim (MIT — jaywalnut310/vits + p0p4k/vits2_pytorch
# share the same MIT LICENSE text, see the sibling ``LICENSE`` file). This
# composition file is Apache-2.0-compatible via the MIT primitives it
# imports; the composition itself is authored by Vokra under Apache-2.0.

import torch
from torch import nn

from .attentions import MultiHeadAttention, FFN
from .modules import LayerNorm, Flip


class Sbv2FlowEncoder(nn.Module):
    """The ``enc`` sub-tree of one SBV2 coupling layer — an Encoder-style
    transformer stack (mirrors jaywalnut310 ``attentions.Encoder``'s body,
    MIT) plus a per-block ``spk_emb_linear`` that adds the speaker
    projection ONCE before the stack.

    Tensor-name contract (matches ``flow.flows.<i>.enc.*`` in the SBV2
    v2 base checkpoint 1:1, verified 2026-08-06 by safetensors
    introspection):

      - ``spk_emb_linear.weight [hidden_channels, gin_channels]``
      - ``spk_emb_linear.bias   [hidden_channels]``
      - ``attn_layers.<0..n_layers-1>.conv_{q,k,v,o}.weight/bias``
      - ``attn_layers.<0..n_layers-1>.emb_rel_{k,v}``  (heads_share=True
        → first dim = 1, second dim = 2*window_size + 1 = 9 for
        window_size=4, third dim = k_channels = hidden/n_heads)
      - ``norm_layers_1.<0..n_layers-1>.{gamma,beta}``  (post-attn LN)
      - ``norm_layers_2.<0..n_layers-1>.{gamma,beta}``  (post-ffn LN)
      - ``ffn_layers.<0..n_layers-1>.conv_{1,2}.weight/bias``
    """

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size, p_dropout=0.0, window_size=4, gin_channels=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        # Per-block speaker conditioning — INSIDE the ``enc`` sub-tree per
        # the SBV2 base checkpoint's ``enc.spk_emb_linear.*`` naming.
        if gin_channels > 0:
            self.spk_emb_linear = nn.Linear(gin_channels, hidden_channels)

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(
                hidden_channels, hidden_channels, n_heads,
                p_dropout=p_dropout, window_size=window_size,
            ))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels, filter_channels,
                kernel_size, p_dropout=p_dropout,
            ))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        """``x``: ``[B, hidden_channels, T]``. ``g``:
        ``[B, gin_channels, 1]`` (per-utterance speaker vector).
        ``x_mask``: ``[B, 1, T]``.

        Applies spk_emb ONCE before the transformer stack (matches Rust
        ``crates/vokra-models/src/sbv2/flow.rs``
        ``SbV2TransformerCouplingLayer::inverse`` step 2:
        ``h = h + self.spk_emb_linear(g.mT).mT``), then runs the
        jaywalnut310 ``attentions.Encoder`` body verbatim.
        """
        if g is not None and self.gin_channels > 0:
            # g: [B, gin_channels, 1] -> transpose to [B, 1, gin_channels]
            # linear:               -> [B, 1, hidden_channels]
            # transpose back:       -> [B, hidden_channels, 1]
            # broadcast-add over T:  [B, hidden, T] + [B, hidden, 1] = [B, hidden, T]
            g_broadcast = self.spk_emb_linear(g.transpose(1, 2)).transpose(1, 2)
            x = x + g_broadcast

        # ---- jaywalnut310 attentions.Encoder body (MIT), verbatim ----
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class Sbv2TransformerCouplingLayer(nn.Module):
    """One SBV2 v2 flow coupling layer — ``pre / enc / post`` with an
    affine coupling (``mean_only=True`` on SBV2 v2 base:
    ``post.weight [half_channels, hidden_channels, 1]`` and no
    log-scale channel).

    Tensor-name contract (matches ``flow.flows.<i>.*`` in the SBV2 v2
    base checkpoint 1:1, verified 2026-08-06):

      - ``pre.weight [hidden_channels, half_channels, 1]``  (Conv1d
        ``half_channels -> hidden_channels`` kernel=1)
      - ``pre.bias  [hidden_channels]``
      - ``enc.*``   (see ``Sbv2FlowEncoder`` docstring)
      - ``post.weight [half_channels * (2 - mean_only), hidden_channels, 1]``
        (Conv1d ``hidden_channels -> half_channels`` for ``mean_only=True``)
      - ``post.bias   [half_channels * (2 - mean_only)]``
    """

    def __init__(self, channels, hidden_channels, kernel_size, n_heads,
                 n_layers, filter_channels=768, p_dropout=0.0,
                 window_size=4, gin_channels=0, mean_only=True):
        assert channels % 2 == 0, "channels must be even (affine coupling splits into halves)"
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = Sbv2FlowEncoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            window_size=window_size,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels,
                              self.half_channels * (2 - mean_only), 1)
        # Upstream convention (jaywalnut310 modules.ResidualCouplingLayer):
        # initialize post weight/bias to zero so the freshly-constructed
        # coupling is the identity. state_dict load overrides both.
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """``x``: ``[B, channels, T]``. ``g``: ``[B, gin_channels, 1]``.
        Forward direction returns ``(x, logdet)``; reverse returns just
        ``x`` — matches jaywalnut310 ``modules.ResidualCouplingLayer``'s
        convention 1:1 (and Rust ``SbV2Flow::inverse`` only ever calls
        the reverse direction, which is inference)."""
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class Sbv2TransformerCouplingBlock(nn.Module):
    """The SBV2 v2 flow's outer stack — ``[layer, Flip] * n_flows``,
    walked forward for training / ``reversed(self.flows)`` for
    inference. Matches Rust ``crates/vokra-models/src/sbv2/flow.rs``
    ``SbV2Flow``'s ``FlowLayer`` enum semantics + the SBV2 v2 base
    checkpoint's ``flow.flows.<i>`` module hierarchy 1:1 (real ckpt
    indices ``0, 2, 4, 6`` are ``Sbv2TransformerCouplingLayer``
    instances, indices ``1, 3, 5, 7`` are ``Flip`` slots with zero
    parameters — the same interleaving jaywalnut310
    ``ResidualCouplingBlock.__init__`` produces).
    """

    def __init__(self, channels, hidden_channels, kernel_size, n_heads,
                 n_layers, filter_channels=768, p_dropout=0.0,
                 window_size=4, n_flows=4, gin_channels=0, mean_only=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(Sbv2TransformerCouplingLayer(
                channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                n_heads=n_heads,
                n_layers=n_layers,
                filter_channels=filter_channels,
                p_dropout=p_dropout,
                window_size=window_size,
                gin_channels=gin_channels,
                mean_only=mean_only,
            ))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
