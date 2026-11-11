"""Replay the frozen experiment's scratch initialization, including discarded draws.

Temporary tensors preserve the starting parameters for every seed; they are not
forward branches or learned weights. Provenance: ../LICENSE.
"""
import math
from torch import nn


def initialize_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.zeros_(layer.bias)
        nn.init.ones_(layer.weight)
    elif isinstance(layer, nn.Conv2d):
        fan_out = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        fan_out //= layer.groups
        nn.init.normal_(layer.weight, 0, math.sqrt(2.0 / fan_out))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class BlockInitialization(nn.Module):
    """Only two projections and a mapping slice survive this temporary layout.

    The discarded tensors account for draws made by the research constructor.
    Changing the sequence changes the reported seeds' starting parameters.
    """
    def __init__(self, channels):
        super().__init__()
        self.height_projection = nn.Linear(channels, channels)
        self.width_projection = nn.Linear(channels, channels)
        self.discarded_projection_a = nn.Linear(channels, channels)
        self.discarded_projection_b = nn.Linear(channels, channels)
        self.discarded_joint_projection = nn.Linear(2 * channels, channels)
        self.mapping_source = nn.Linear(2 * channels, channels)
        self.discarded_convolutions = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.discarded_token_norm = nn.LayerNorm(2 * channels)
        self.discarded_spatial_norm = nn.BatchNorm2d(channels)
        # The experiment initialized both the layout and its enclosing stage.
        self.apply(initialize_layer)
        self.apply(initialize_layer)
