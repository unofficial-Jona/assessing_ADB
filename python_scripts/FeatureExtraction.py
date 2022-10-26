import torch.nn as nn
import torch
import numpy as np
import torchvision.models as models

# CNN Backbone to extract features of individual frames

def return_backbone(model = models.resnext101_32x8d(pretrained=True), cut_layers = 1):
    """load model and cut classification head from it, so it can be used as a CNN Backbone in the transformer architecture. 

    Args:
        model (pytorch model): must employ the .children() method. Defaults to ResNext101_32x8d(pretrained=True).
        cut_layers (int, optional): Number of layers to be cut from the end of the model. Defaults to 1.

    Returns:
        pytorch model: model without classification head
    """
    newmodel = nn.Sequential(*(list(model.children())[:-cut_layers]))
    return newmodel

