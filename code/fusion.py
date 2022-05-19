import torch.nn as nn
from abc import ABC, abstractmethod


class FusionBase(nn.Module):
    def __init__(self, config):
        pass

    @abstractmethod
    def forward(self, txt, img):
        pass
