import torch.nn as nn
from text_encoder import TextEncoder
from image_encoder import ImageEncoder
from fusion import FusionBase
from abc import ABC, abstractmethod


class Classifier(nn.Module):
    def __init__(self, config):
        pass

    def forward(self):
        pass


class ChartFCBaseline(nn.Module):
    def __init__(self, config):
        super(ChartFCBaseline, self).__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.fusion = FusionBase(config)
        self.classifier = Classifier(config)

    def forward(self, img, txt):
        # Unimodal encoding
        txt_features = self.text_encoder(txt)
        img_features = self.image_encoder(img)

        # Fusion
        mm_features = self.fusion(txt_features, img_features)
        # Classification
        out = self.classifier(mm_features)

        return out
