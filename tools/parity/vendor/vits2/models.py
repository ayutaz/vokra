"""Vendored from p0p4k/vits2_pytorch@1f4f3790568180f8dec4419d5cad5d0877b034bb/models.py.

MIT License. See sibling LICENSE.

Contains ONLY: ResidualCouplingTransformersLayer (as TransformerCouplingLayer),
ResidualCouplingTransformersBlock (as TransformerCouplingBlock).

Deliberately excluded: SynthesizerTrn, PosteriorEncoder, 
MultiPeriodDiscriminator, MultiScaleDiscriminator, TextEncoder,
DurationPredictor, StochasticDurationPredictor — those are either training-
side or covered elsewhere.

Also includes minimal helper classes (WN, Flip) vendored from modules.py
to enable self-contained inference without external dependencies.

Feeds Vokra parity reference for `sbv2.flow.layer.<i>.*` tensor
group (SBV2 v2 Blocker 2b).
"""

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from . import attentions


# Vendored from modules.py (WN and Flip helper classes)

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x



class WN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)



class TransformerCouplingLayer(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # vits2
        self.pre_transformer = attentions.Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        # vits2
        self.post_transformer = attentions.Encoder(
            self.hidden_channels,
            self.hidden_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
        x0_ = x0_ + x0  # vits2 residual connection
        h = self.pre(x0_) * x_mask  # changed from x0 to x0_ to retain x0 for the flow
        h = self.enc(h, x_mask, g=g)

        # vits2 - (experimental;uncomment the following 2 line to use)
        # h_ = self.post_transformer(h, x_mask)
        # h = h + h_ #vits2 residual connection

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



class TransformerCouplingBlock(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        use_transformer_flows=False,
        transformer_flow_type="pre_conv",
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        if use_transformer_flows:
            if transformer_flow_type == "pre_conv":
                for i in range(n_flows):
                    self.flows.append(
                        TransformerCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
            elif transformer_flow_type == "pre_conv2":
                for i in range(n_flows):
                    self.flows.append(
                        TransformerCouplingLayer2(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
            elif transformer_flow_type == "fft":
                for i in range(n_flows):
                    self.flows.append(
                        FFTransformerCouplingLayerAlias(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
            elif transformer_flow_type == "mono_layer_inter_residual":
                for i in range(n_flows):
                    self.flows.append(
                        modules.ResidualCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
                    self.flows.append(
                        MonoTransformerFlowLayer(
                            channels, hidden_channels, mean_only=True
                        )
                    )
            elif transformer_flow_type == "mono_layer_post_residual":
                for i in range(n_flows):
                    self.flows.append(
                        modules.ResidualCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
                    self.flows.append(
                        MonoTransformerFlowLayer(
                            channels,
                            hidden_channels,
                            mean_only=True,
                            residual_connection=True,
                        )
                    )
        else:
            for i in range(n_flows):
                self.flows.append(
                    modules.ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

