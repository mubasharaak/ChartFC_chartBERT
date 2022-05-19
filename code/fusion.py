import torch.nn as nn
from abc import ABC, abstractmethod


class FusionBase(nn.Module):
    def __init__(self, config):
        super(FusionBase, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, txt, img):
        pass


class ConcatFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, txt, img):
        pass


class ConcatBiGRUFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, txt, img):
        pass


class MultiplicationFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, txt, img):
        pass


class MCBFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, txt, img):
        pass


class TransformerFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, txt, img):
        pass
