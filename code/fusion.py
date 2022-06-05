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
        if config.img_encoder == "vit":
            config.fusion_out_dim = config.text_dim

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

        self.avg_pool = nn.AvgPool2d((3, 3), stride=(15, 20), padding=(1, 1))
        self.avg_pool_output = nn.AvgPool2d((3, 3), stride=(100, 1), padding=(1, 1))

    def forward(self, txt, img, apply_pooling=True):
        img = self.avg_pool(img)
        _, _, nw, nh = img.shape
        bs, tdim1, tdim2 = txt.shape

        txt = txt.permute(0, 2, 1)
        txt = torch.unsqueeze(txt, -1)

        txt_tile = txt
        if nw == 1 and nh == 1:
            img = img.repeat(1, 1, tdim1, 1)
            mm_feat = torch.cat([img, txt_tile], dim=1)
        else:
            mm_feat = torch.cat([img, txt_tile], dim=2)

        mm_feat = self.bn(mm_feat)
        mm_feat = self.transform_convs(mm_feat)
        if len(mm_feat.shape) == 4 and apply_pooling:
            mm_feat = self.avg_pool_output(mm_feat).squeeze()

        return mm_feat


class ConcatBiGRUFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        if config.img_encoder == "vit":
            config.fusion_out_dim = 2*config.text_dim
            self.fusion_dim = config.text_dim
            self.num_mmc_units = config.text_dim
        else:
            config.fusion_out_dim = 2*(config.text_dim + config.img_dim)
            self.fusion_dim = config.text_dim + config.img_dim
            self.num_mmc_units = config.text_dim + config.img_dim

        self.avg_pool = nn.AvgPool2d((3, 3), stride=(15, 20), padding=(1, 1))
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
        img = self.avg_pool(img)
        _, _, nw, nh = img.shape
        bs, tdim1, tdim2 = txt.shape

        txt = txt.permute(0, 2, 1)
        txt = torch.unsqueeze(txt, -1)

        txt_tile = txt
        if nw == 1 and nh == 1:
            img = img.repeat(1, 1, tdim1, 1)
            mm_feat = torch.cat([img, txt_tile], dim=1)
        else:
            mm_feat = torch.cat([img, txt_tile], dim=2)

        mm_feat = self.bn(mm_feat)
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
        config.fusion_out_dim = config.img_dim + config.text_dim
        self.config = config

        self.avg_pool = nn.AvgPool2d((3, 3), stride=(15, 20), padding=(1, 1))
        # optionally try bn and a few conv2d and relu layers
        self.num_mmc_units = config.text_dim
        self.bn = nn.BatchNorm1d(self.num_mmc_units)

        self.transform_convs = []
        self.transform_convs.append(nn.Conv1d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(2):
            self.transform_convs.append(nn.Conv1d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)

        if config.img_encoder == "resnet":
            self.lin1 = nn.Linear(config.img_dim, int(config.img_dim/8))
            self.lin2 = nn.Linear(int(config.img_dim/8)*config.text_dim, config.img_dim + config.text_dim)
        else:
            self.lin1 = nn.Linear(config.img_dim, int(config.img_dim/2))
            self.lin2 = nn.Linear(int(config.img_dim/2)*config.text_dim, config.img_dim + config.text_dim)

    def forward(self, txt, img):
        # reshape and tile txt_feat
        bs, i_dim, nw, nh = img.shape

        if not (nw == 1 and nh == 1):
            img = self.avg_pool(img)

        img = img.squeeze(-1)
        if self.config.img_encoder != "vit":
            img = img.repeat(1, 1, txt.shape[1])
        else:
            img_dim = img.shape[2]
            img = img.repeat(1, 1, txt.shape[1])
            txt = txt.repeat(1, img_dim, 1)

        txt = txt.permute(0, 2, 1)
        img = img.permute(0, 2, 1)

        img = self.lin1(img)
        mm_feat = torch.matmul(txt, img)
        # 1x1 conv and relu
        mm_feat = self.bn(mm_feat)
        mm_feat = self.transform_convs(mm_feat)
        mm_feat = mm_feat.reshape(bs, -1)

        # dimensionality reduction
        mm_feat = self.lin2(mm_feat)

        return mm_feat


class MCBFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        mcb_out_dim = 16000
        config.fusion_out_dim = 3000
        self.config = config
        self.comp_layer1 = CompactBilinearPooling(config.img_dim, config.img_dim, mcb_out_dim,
                                                  sum_pool=False)
        self.conv1 = nn.Conv2d(mcb_out_dim, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        # weights
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.comp_layer2 = CompactBilinearPooling(config.img_dim, config.img_dim, mcb_out_dim,
                                                  sum_pool=False)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=(100, 1), padding=(1, 1))
        self.lin1 = nn.Linear(config.text_dim, config.img_dim)
        self.lin2 = nn.Linear(mcb_out_dim, config.fusion_out_dim)

    def forward(self, txt, img):
        _, img_dim, nw, nh = img.shape
        bs, tdim1, tdim2 = txt.shape

        # prepare txt input for first MCB
        txt = self.avg_pool(txt)
        txt = txt.reshape(bs, -1)
        # txt = self.lin1(txt)
        txt_tile = txt.repeat(1, 1, nw * nh)
        txt_tile = txt_tile.reshape(bs, -1, nw, nh)

        # 1st MCB
        img_feat = img
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
        bottom1_resh = img_feat.contiguous().view(bs, nw * nh, -1)
        weights = weights.view(bs, -1, 1)
        res = torch.bmm(bottom1_resh.transpose(1, 2), weights)
        res = res.squeeze(2)

        # prepare data for 2nd MCB
        res_unsqueezed = res.unsqueeze(-1).unsqueeze(-1)
        txt = txt.unsqueeze(-1).unsqueeze(-1)

        # 2nd call of MCB
        final_out = self.comp_layer2(txt, res_unsqueezed)
        final_out = final_out.squeeze()
        final_out = torch.sqrt(F.relu(final_out)) - torch.sqrt(F.relu(-final_out))  # square root
        final_out = torch.nn.functional.normalize(final_out)

        # final lin layer
        final_out = self.lin2(final_out)

        return final_out


class TransformerFusion(FusionBase):
    def __init__(self, config):
        super().__init__(config)
        config.fusion_out_dim = config.text_dim + config.img_dim
        if config.img_encoder == "vit":
            config.fusion_out_dim = config.text_dim

        self.pre_fusion = ConcatFusion(config)
        self.add_tensor = 0
        if (config.fusion_out_dim % config.num_attention_heads) != 0:
            self.add_tensor = config.num_attention_heads - (config.fusion_out_dim % config.num_attention_heads)
            config.fusion_out_dim = config.fusion_out_dim + self.add_tensor

        config.hidden_size = config.fusion_out_dim
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.fusion_transf_layers)])
        self.pooler = BertPooler(config)
        self.avg_pool_output = nn.AvgPool2d((3, 3), stride=(100, 1), padding=(1, 1))

    def forward(self, txt, img):
        output_all_encoded_layers = False
        mm_feat = self.pre_fusion(txt, img, apply_pooling=False)
        mm_feat = mm_feat.squeeze()
        attention_mask = torch.ones((mm_feat.shape[0], mm_feat.shape[2]), dtype=torch.long)

        # make sure that mm_feat%12 = 0 and extend attention with zeros
        if self.add_tensor != 0:
            bs, d1, d2 = mm_feat.shape
            tens_zeros = torch.zeros([bs, self.add_tensor, d2]).cuda()
            mm_feat = torch.cat([mm_feat, tens_zeros], dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        all_encoder_layers = []
        hidden_states = mm_feat
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
            hidden_states = hidden_states.permute(0, 2, 1)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        hidden_states = self.avg_pool_output(hidden_states.permute(0, 2, 1))
        hidden_states = hidden_states.squeeze()

        if not output_all_encoded_layers:
            return hidden_states
        else:
            return all_encoder_layers
