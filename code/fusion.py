import torch
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
        config.fusion_out_dim = config.text_dim + config.img_dim
        self.fusion_dim = config.fusion_out_dim
        self.num_mmc_units = config.fusion_out_dim

        self.bn = nn.BatchNorm2d(self.fusion_dim)
        self.transform_convs = []
        self.transform_convs.append(nn.Conv2d(self.fusion_dim, self.num_mmc_units, kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(2):
            self.transform_convs.append(nn.Conv2d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)

    def forward(self, txt, img):
        _, _, nw, nh = img.shape  # @todo make sure all image features come in this shape
        _, tdim = txt.shape
        txt_tile = txt.repeat(1, 1, nw * nh)
        txt_tile = txt_tile.view(-1, tdim, nw, nh)

        mm_feat = self.bn(torch.cat([img, txt_tile], dim=1))
        mm_feat = self.transform_convs(mm_feat)

        return mm_feat


class ConcatBiGRUFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        config.fusion_out_dim = 0

    def forward(self, txt, img):
        pass


class MultiplicationFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        config.fusion_out_dim = config.img_dim
        self.config = config

        # optionally try bn and a few conv2d and relu layers
        self.transform_convs = []
        self.num_mmc_units = config.img_dim
        self.bn = nn.BatchNorm2d(self.num_mmc_units)

        self.transform_convs.append(nn.Conv2d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(2):
            self.transform_convs.append(nn.Conv2d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)

    def forward(self, txt, img):
        # reshape and tile txt_feat
        _, _, nw, nh = img.shape
        txt_feat = torch.unsqueeze(txt, -1)
        txt_feat = torch.unsqueeze(txt_feat, -1)
        txt_feat = torch.tile(txt_feat, (int(self.config.img_dim / self.config.text_dim), nh, nw))

        # multiply
        mm_feat = torch.matmul(txt_feat, img)

        # 1x1 conv and relu
        mm_feat = self.transform_convs(self.bn(mm_feat))
        return mm_feat


class MCBFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        config.fusion_out_dim = 0

    def forward(self, txt, img):
        pass


class TransformerFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        config.fusion_out_dim = 0

    def forward(self, txt, img):
        pass
