import torch.nn as nn
from text_encoder import TextEncoder
from image_encoder import ImageEncoder
from fusion import FusionBase
from abc import ABC, abstractmethod


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        pass

    def forward(self):
        pass


class ChartFCBaseline(nn.Module):
    def __init__(self, config):
        super(ChartFCBaseline, self).__init__()
        self.text_encoder = config.COMPONENTS[config.txt_encoder]
        self.image_encoder = config.COMPONENTS[config.img_encoder]
        self.fusion = config.COMPONENTS[config.fusion_method]
        self.classifier = Classifier(config)

    def forward(self, img, txt, txt_len):
        # Unimodal encoding
        txt_features = self.text_encoder(txt, txt_len)
        img_features = self.image_encoder(img)

        # Fusion
        mm_features = self.fusion(txt_features, img_features)
        # Classification
        out = self.classifier(mm_features)

        return out
