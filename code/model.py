import torch.nn as nn
from abc import ABC, abstractmethod


class TextEncoder(nn.Module):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class ImageEncoder(nn.Module):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class FusionMethod(nn.Module):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class Classifier(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class ChartFCBaseline(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
