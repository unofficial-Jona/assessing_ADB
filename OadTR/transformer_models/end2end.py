import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import flow_to_image

from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .Transformer import TransformerModel
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
from .ViT import VisionTransformer_v3

from ipdb import set_trace


class End2End(nn.Module):
    def __init__(self, rgb_model, flow_model, ViT):
        super(End2End).__init__()

        self.rgb_model = rgb_model
        self.flow_model = flow_model
        self.ViT = ViT

    def forward(self, frames):
        flow = self.flow_model(frames[:-1], frames[1:])
        flow = flow_to_image(flow)
        flow = self.rgb_model(flow)
        rgb = self.rgb_model(frames)

        out = ViT(rgb, flow)
        return out
