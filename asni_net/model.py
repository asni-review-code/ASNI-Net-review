"""ASNI-Net: convolutional encoder, three SIP -> NI -> CM blocks, additive decoder.

Architecture and numerical behavior follow the frozen paper model. Source
attributions and license texts are provided in LICENSE.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .initialization import BlockInitialization
from .layers import (
    AxisSampler, ConvStage, NodeInteraction, SamplingInteractionPath,
    TokenProjection, fractional_sample,
)


class ASNIBlock(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.channels = channels
        # These 2*C parameters existed but were unused in the frozen model.
        # Keep them explicitly for checkpoint/count identity; never apply them.
        self.reserved_normalization = nn.LayerNorm(channels)
        self._initialization = BlockInitialization(channels)
        self.dropout = nn.Dropout(dropout)

    def finish_initialization(self, samplers, interaction):
        source = self._initialization
        with torch.random.fork_rng(devices=[]):
            self.channel_mapping = nn.Linear(self.channels, self.channels)
        with torch.no_grad():
            self.channel_mapping.weight.copy_(source.mapping_source.weight[:, :self.channels])
            self.channel_mapping.bias.copy_(source.mapping_source.bias)
        self.sampling = SamplingInteractionPath(
            samplers[0], samplers[1], source.height_projection,
            source.width_projection, self.dropout,
        )
        self.node_interaction = interaction
        del self._initialization

    def forward(self, tokens, height, width):
        sampled = self.sampling(tokens, height, width)
        interacted = self.node_interaction(sampled)
        return self.dropout(self.channel_mapping(interacted))


def _finish_blocks(blocks):
    # Preserve both the construction order and the external random state.
    # Two discarded samplers per stage are historical initialization draws.
    with torch.random.fork_rng(devices=[]):
        samplers = [
            [AxisSampler(block.channels, axis) for axis in (2, 3, 3, 2)]
            for block in blocks
        ]
    with torch.random.fork_rng(devices=[]):
        initial_seed = torch.initial_seed()
        interactions = []
        for index, block in enumerate(blocks):
            torch.manual_seed(initial_seed + 1000 + index)
            interactions.append(NodeInteraction(block.channels))
    for block, axial, interaction in zip(blocks, samplers, interactions):
        block.finish_initialization(axial, interaction)


def _spatial(tokens, height, width):
    return tokens.reshape(tokens.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()


def _upsample(features):
    return F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)


class ASNINet(nn.Module):
    """Formal 1,448,945-parameter model: RGB input, one lesion logit channel."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvStage(3, 16, 16), ConvStage(16, 32, 32), ConvStage(32, 64, 64),
        ])
        self.token_projections = nn.ModuleDict({
            'encoder': TokenProjection(64, 128),
            'bottleneck': TokenProjection(128, 256),
            'decoder': TokenProjection(256, 128),
        })
        self.asni_blocks = nn.ModuleDict({
            'encoder': ASNIBlock(128, 0.0),
            'bottleneck': ASNIBlock(256, 0.1),
            'decoder': ASNIBlock(128, 0.0),
        })
        self.output_norms = nn.ModuleDict({
            'encoder': nn.LayerNorm(128),
            'bottleneck': nn.LayerNorm(256),
            'decoder': nn.LayerNorm(128),
        })
        self.decoder_projection = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder_normalization = nn.BatchNorm2d(64)
        self.decoder = nn.ModuleList([
            ConvStage(64, 64, 32), ConvStage(32, 32, 16), ConvStage(16, 16, 8),
        ])
        self.segmentation_head = nn.Conv2d(8, 1, 1)
        _finish_blocks(list(self.asni_blocks.values()))

    def _token_stage(self, features, stage):
        tokens, height, width = self.token_projections[stage](features)
        tokens = self.asni_blocks[stage](tokens, height, width)
        return _spatial(self.output_norms[stage](tokens), height, width)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or any(s % 16 for s in images.shape[-2:]):
            raise ValueError('Expected NCHW RGB input with dimensions divisible by 16')
        skips = []
        features = images
        for stage in self.encoder:
            features = stage(features)
            skips.append(features)
            features = F.max_pool2d(features, 2)

        encoded = self._token_stage(features, 'encoder')
        features = self._token_stage(F.max_pool2d(encoded, 2), 'bottleneck')
        tokens, height, width = self.token_projections['decoder'](features)
        features = torch.add(_upsample(_spatial(tokens, height, width)), encoded)
        height, width = features.shape[-2:]
        tokens = self.asni_blocks['decoder'](features.flatten(2).transpose(1, 2), height, width)
        features = _spatial(self.output_norms['decoder'](tokens), height, width)
        features = _upsample(F.relu(self.decoder_normalization(self.decoder_projection(features))))

        for index, (stage, skip) in enumerate(zip(self.decoder, reversed(skips))):
            features = stage(torch.add(features, skip))
            if index < 2:
                features = _upsample(features)
        return {'logits': self.segmentation_head(features), 'aux_logits': []}


def build_model():
    return ASNINet()


def forward_model(model, image, valid=None):
    return model(image)
