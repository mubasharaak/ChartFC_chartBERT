import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from layer import GELU
from transformers import BertTokenizer, BertModel

from image_encoder import DenseNet
# from simple_mcb_baseline.model_simple_fusion import ConcatFusion, RecurrentFusion
from model_simple_fusion import ConcatFusion, RecurrentFusion


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


class MultiplicationFusion(nn.Module):
    def __init__(self, config=None, img_dim=None):
        super(MultiplicationFusion, self).__init__()
        self.config = config
        # optionally try bn and a few conv2d and relu layers
        self.transform_convs = []
        if img_dim < config.text_dim:
            self.num_mmc_units = config.text_dim
        else:
            self.num_mmc_units = img_dim*3

        self.bn = nn.BatchNorm2d(self.num_mmc_units)
        self.transform_convs.append(nn.Conv2d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(2):
            self.transform_convs.append(nn.Conv2d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)

    def forward(self, txt_feat, img_feat):
        # reshape and tile txt_feat
        _, idim, nw, nh = img_feat.shape
        txt_feat = torch.unsqueeze(txt_feat, -1)
        txt_feat = torch.unsqueeze(txt_feat, -1)
        if idim < self.config.text_dim:
            txt_feat = torch.tile(txt_feat, (1, nh, nw))
            img_feat = torch.tile(img_feat, (int(self.config.text_dim/idim), 1, 1))
        else:
            txt_feat = torch.tile(txt_feat, (4, nh, nw))
            img_feat = torch.tile(img_feat, (3, 1, 1))

        # multiply
        mm_feat = torch.matmul(txt_feat, img_feat)

        # 1x1 conv and relu
        mm_feat = self.transform_convs(self.bn(mm_feat))
        return mm_feat


class Classifier(nn.Module):
    def __init__(self, num_classes, config):
        super(Classifier, self).__init__()
        self.config = config
        self.classifier = nn.Sequential(
            # nn.Linear(config.hidden_size*4, config.hidden_size * 2), # @todo remove after experiments with ONLY concat (no biGRU)
            nn.Linear(int(6*6*config.hidden_size*2), config.hidden_size * 2), # @todo remove after experiments with ONLY concat (no biGRU)
            GELU(),
            FusedLayerNorm(config.hidden_size * 2, eps=1e-12),
            nn.Linear(config.hidden_size * 2, num_classes)
        )

    def forward(self, mm_feat):
        out = self.classifier(mm_feat)
        return out


class ChartFCBaseline(nn.Module):
    def __init__(self, token_count, num_classes, config, use_ocr=False):
        super(ChartFCBaseline, self).__init__()
        # text encoder
        self.txt_encoder_claim = BertEncoder(config)

        # image encoder
        self.img_encoder = ImageEncoder(config.densenet_config)
        img_dims = config.densenet_dim

        # fusion
        if use_ocr:
            self.use_ocr = True
            self.txt_encoder_ocr = BertEncoder(config)
            self.bimodal_low = ConcatFusion(config, img_dim=img_dims[0], ocr_dim=config.text_dim)
            self.bimodal_high = ConcatFusion(config, img_dim=img_dims[2], ocr_dim=config.text_dim)
        else:
            self.use_ocr = False
            # self.bimodal_low = ConcatFusion(config, img_dim=img_dims[0])
            # self.bimodal_high = ConcatFusion(config, img_dim=img_dims[2])
            self.bimodal_high = MultiplicationFusion(config, img_dim=img_dims[2])

        self.maxpool_low = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        # self.rf_low = RecurrentFusion(config.num_rf_out, config.num_rf_out) # @todo remove after experiments with ONLY concat (no biGRU)
        # self.rf_high = RecurrentFusion(config.num_rf_out, config.num_rf_out) # @todo remove after experiments with ONLY concat (no biGRU)

        # classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6)) # @todo remove after experiments with ONLY concat (no biGRU)
        self.classifier = Classifier(num_classes, config)

    def forward(self, img, txt, ocr=None):
        txt_feat = self.txt_encoder_claim(txt)
        feat_low, feat_mid, feat_high = self.img_encoder(img)
        feat_low = self.maxpool_low(feat_low)
        if self.use_ocr:
            ocr_feat = self.txt_encoder_ocr(ocr)
            bimodal_feat_low = self.bimodal_low(txt_feat, feat_low, ocr_feat)
            bimodal_feat_high = self.bimodal_high(txt_feat, feat_high, ocr_feat)
        else:
            # bimodal_feat_low = self.bimodal_low(txt_feat, feat_low)
            bimodal_feat_high = self.bimodal_high(txt_feat, feat_high)

        # bimodal_feat_low = self.rf_low(bimodal_feat_low) # @todo remove after experiments with ONLY concat (no biGRU)
        # bimodal_feat_high = self.rf_high(bimodal_feat_high) # @todo remove after experiments with ONLY concat (no biGRU)
        # final_feat = torch.cat([bimodal_feat_low, bimodal_feat_high], dim=1) # @todo remove after experiments with ONLY concat (no biGRU)

        final_feat = self.avg_pool(bimodal_feat_high) # @todo remove after experiments with ONLY concat (no biGRU)
        final_feat = final_feat.view(final_feat.shape[0], -1) # @todo remove after experiments with ONLY concat (no biGRU)

        out = self.classifier(final_feat)
        return out
