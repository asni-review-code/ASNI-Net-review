"""ASNI-Net components. Adapted portions: see LICENSE."""
import torch
import torch.nn.functional as F
from torch import nn
from .initialization import initialize_layer


class ConvStage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, features):
        return self.layers(features)


class TokenProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.normalization = nn.LayerNorm(out_channels)
        self.apply(initialize_layer)

    def forward(self, features):
        features = self.convolution(features)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return self.normalization(F.gelu(tokens)), height, width


def fractional_sample(features, offsets, axis):
    """Linear interpolation along one spatial axis with zero extension."""
    length = features.shape[axis]
    coordinate_shape = [1, 1, 1, 1]
    coordinate_shape[axis] = length
    positions = torch.arange(length, dtype=offsets.dtype, device=features.device)
    coordinates = positions.view(coordinate_shape) + offsets
    lower = coordinates.floor()
    fraction = coordinates - lower
    lower_index = lower.to(torch.long)
    upper_index = lower_index + 1
    low = torch.gather(features, axis, lower_index.clamp(0, length - 1))
    high = torch.gather(features, axis, upper_index.clamp(0, length - 1))
    low = low * ((lower_index >= 0) & (lower_index < length)).to(features.dtype)
    high = high * ((upper_index >= 0) & (upper_index < length)).to(features.dtype)
    return low * (1 - fraction) + high * fraction


class AxisSampler(nn.Module):
    def __init__(self, channels, axis):
        super().__init__()
        self.axis = axis
        self.offset_norm = nn.LayerNorm(channels)
        self.offset_projection = nn.Linear(channels, channels)
        nn.init.trunc_normal_(self.offset_projection.weight, std=0.02)
        nn.init.zeros_(self.offset_projection.bias)

    def forward(self, tokens, height, width):
        batch, _, channels = tokens.shape
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        with torch.autocast(device_type=features.device.type, enabled=False):
            full = tokens.float() if tokens.dtype in (torch.float16, torch.bfloat16) else tokens
            offsets = self.offset_projection(self.offset_norm(full))
            offsets = offsets.transpose(1, 2).reshape(batch, channels, height, width)
            sampled = fractional_sample(features.to(full.dtype), offsets, self.axis)
        return sampled.flatten(2).transpose(1, 2)


class SamplingInteractionPath(nn.Module):
    """SIP: sequential height/width sampling and weighted residual fusion."""
    def __init__(self, height_sampler, width_sampler, height_projection, width_projection, dropout):
        super().__init__()
        self.height_sampler = height_sampler
        self.width_sampler = width_sampler
        self.height_projection = height_projection
        self.width_projection = width_projection
        self.dropout = dropout

    def forward(self, tokens, height, width):
        first = self.height_sampler(tokens, height, width)
        first = self.dropout(F.gelu(self.height_projection(first)))
        second = self.width_sampler(first, height, width)
        second = self.dropout(self.width_projection(second))
        return tokens + 0.5 * first + 0.5 * second


class NodeInteraction(nn.Module):
    """NI: aggregate over positions, mix compact nodes, redistribute over nodes."""
    def __init__(self, channels, nodes=16):
        super().__init__()
        node_channels = channels // 4
        self.normalization = nn.LayerNorm(channels)
        self.assignment = nn.Linear(channels, nodes)
        self.value = nn.Linear(channels, node_channels)
        self.node_mix = nn.Linear(nodes, nodes)
        self.channel_mix = nn.Linear(node_channels, node_channels)
        self.restore = nn.Linear(node_channels, channels)
        for layer in (self.assignment, self.value, self.node_mix, self.channel_mix, self.restore):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens):
        normalized = self.normalization(tokens)
        scores, values = self.assignment(normalized), self.value(normalized)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            nodes = scores.float().softmax(1).transpose(1, 2) @ values.float()
        mixed = self.node_mix(nodes.transpose(1, 2)).transpose(1, 2)
        nodes = nodes + self.channel_mix(F.gelu(mixed))
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            broadcast = scores.float().softmax(-1) @ nodes.float()
        return tokens + self.restore(broadcast)
