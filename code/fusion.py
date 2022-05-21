import copy
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from compact_bilinear_pooling_layer import CompactBilinearPooling
from layer import BertLayer, BertPooler


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

        self.relu = nn.ReLU()
        self.pool_size = 1
        self.bigru = nn.GRU(input_size=self.num_mmc_units,
                            hidden_size=self.num_mmc_units,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, txt, img):
        # concat fusion
        _, _, nw, nh = img.shape  # @todo make sure all image features come in this shape
        _, tdim = txt.shape
        txt_tile = txt.repeat(1, 1, nw * nh)
        txt_tile = txt_tile.view(-1, tdim, nw, nh)

        mm_feat = self.bn(torch.cat([img, txt_tile], dim=1))
        mm_feat = self.transform_convs(mm_feat)

        # recurrent fusion
        _, fs, nw, nh = mm_feat.shape
        mmc_feat = mm_feat.view(-1, fs, nw * nh)
        mmc_feat = torch.transpose(mmc_feat, 1, 2)
        output, h = self.bigru(mmc_feat)
        h_flattened = torch.flatten(torch.transpose(h, 0, 1), start_dim=1)

        return h_flattened


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
        mcb_out_dim = 16000
        config.fusion_out_dim = 2048
        self.config = config
        self.comp_layer1 = CompactBilinearPooling(config.fusion_out_dim, config.fusion_out_dim, mcb_out_dim,
                                                  sum_pool=False)
        self.conv1 = nn.Conv2d(mcb_out_dim, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        # weights
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.comp_layer2 = CompactBilinearPooling(config.fusion_out_dim, config.fusion_out_dim, mcb_out_dim,
                                                  sum_pool=False)

    def forward(self, txt, img):
        # L2 norm of img_feat
        img_feat = torch.nn.functional.normalize(img)

        # reshape and tile txt_feat
        bs, _, nw, nh = img_feat.shape
        txt_feat = torch.unsqueeze(txt, -1)
        txt_feat = torch.unsqueeze(txt_feat, -1)
        txt_tile = torch.tile(txt_feat, (int(self.config.img_dim / (self.config.text_dim * 2)), nh, nw))

        # 1st MCB
        out = self.comp_layer1(txt_tile, img_feat)
        out = out.permute(0, 3, 1, 2)
        out = torch.sqrt(F.relu(out)) - torch.sqrt(F.relu(-out))  # todo square root check initial code
        out = torch.nn.functional.normalize(out)

        # weights
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        out = out.reshape(-1, 1, nw * nh)
        weights = nn.functional.softmax(out, dim=2)
        weights = weights.reshape((-1, 1, nw, nh))  # or other way around nh, nw

        # apply weights to image vector
        bottom1_resh = img_feat.view(bs, nw * nh, -1)
        weights = weights.view(bs, -1, 1)

        res = torch.bmm(bottom1_resh.transpose(1, 2), weights)
        res = res.squeeze(2)

        # prepare data for 2nd MCB
        res_unsqueezed = res.unsqueeze(-1).unsqueeze(-1)

        # 2nd call of MCB
        final_out = self.comp_layer2(res_unsqueezed, txt_feat)
        final_out = final_out.squeeze()
        final_out = torch.sqrt(F.relu(final_out)) - torch.sqrt(F.relu(-final_out))  # square root
        final_out = torch.nn.functional.normalize(final_out)

        return final_out


class TransformerFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        config.fusion_out_dim = 768
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.fusion_transf_layers)])
        self.pooler = BertPooler(config)

    def forward(self, txt, img):
        output_all_encoded_layers = False
        mm_feat = torch.cat([img, txt], dim=1)
        attention_mask = torch.ones((txt.shape[0], txt.shape[1]), dtype=torch.long)
        attention_mask = torch.cat((attention_mask,
                                    torch.ones((attention_mask.shape[0], img.shape[1]), dtype=torch.long)), 1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        all_encoder_layers = []
        hidden_states = mm_feat
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            return hidden_states


