import torch.nn as nn
from abc import ABC, abstractmethod


class ImageEncoder(nn.Module):
    def __init__(self, config):
        pass

    @abstractmethod
    def forward(self, img):  # @todo special case: ViT
        pass
