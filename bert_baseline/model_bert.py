from collections import OrderedDict

import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from layer import GELU
from transformers import BertTokenizer, BertModel
from simple_mcb_baseline.model_simple_fusion import ConcatFusion, MultiplicationFusion, RecurrentFusion
from image_encoder import DenseNet


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_encoder = BertModel.from_pretrained(model_name, output_hidden_states=True)
        # self.bert_encoder.encoder.layer = nn.ModuleList(list(self.bert_encoder.encoder.layer)[:3]) # use only subset of encoder layers

    def forward(self, txt):
        txt = list(txt)
        embeddings = self.tokenizer.batch_encode_plus(txt, padding='longest', return_tensors='pt',
                                                      return_attention_mask=True)
        embeddings = embeddings.to('cuda')
        out = self.bert_encoder(**embeddings)
        encoded_q = torch.mean(out[0], dim=1).squeeze()  # @todo check what output am I returning and how reshaping?
        return encoded_q


class ImageEncoder(nn.Module):
    def __init__(self, densenet_config):
        super(ImageEncoder, self).__init__()
        self.densenet = DenseNet(block_config=densenet_config).cuda()

    def forward(self, img):
        _, dense, final = self.densenet(img)
        return dense[0], dense[1], final


class Classifier(nn.Module):
    def __init__(self, num_classes, config):
        super(Classifier, self).__init__()
        self.config = config
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            GELU(),
            FusedLayerNorm(config.hidden_size * 2, eps=1e-12),
            nn.Linear(config.hidden_size * 2, num_classes)
        )

    def forward(self, mm_feat):
        out = self.classifier(mm_feat)
        return out


class ChartFCBaseline(nn.Module):
    def __init__(self, token_count, num_classes, config):
        super(ChartFCBaseline, self).__init__()
        # text encoder
        self.txt_encoder = BertEncoder(config)

        # image encoder
        self.img_encoder = ImageEncoder(config.densenet_config)
        img_dims = config.densenet_dim

        # fusion
        self.bimodal_low = ConcatFusion(config, img_dim=img_dims[0])
        self.bimodal_high = ConcatFusion(config, img_dim=img_dims[2])
        self.maxpool_low = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)  # RGB to L
        self.rf_high = RecurrentFusion(config.num_rf_out, config.num_bimodal_units)

        # classifier
        self.classifier = Classifier(num_classes, config.num_rf_out * 4, config)

    def forward(self, img, txt):
        txt_feat = self.txt_encoder(txt)
        feat_low, feat_mid, feat_high = self.img_encoder(img)

        feat_low = self.maxpool_low(feat_low)
        bimodal_feat_low = self.bimodal_low(feat_low, txt_feat)
        bimodal_feat_high = self.bimodal_high(feat_high, txt_feat)

        bimodal_feat_low = self.rf_low(bimodal_feat_low)
        bimodal_feat_high = self.rf_high(bimodal_feat_high)
        final_feat = torch.cat([bimodal_feat_low, bimodal_feat_high], dim=1)

        out = self.classifier(final_feat)
        return out
