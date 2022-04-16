from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torchvision.models as models
from layer import BertLayer, BertPooler, GELU
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from apex.normalization.fused_layer_norm import FusedLayerNorm


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        self.word_embeddings = nn.Embedding(len(self.tokenizer), config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(len(self.tokenizer), config.hidden_size)

        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, txt, txt_len, is_ocr = False):
        embeddings = self.tokenizer.batch_encode_plus(list(txt), padding='longest', return_tensors='pt',
                                                      return_attention_mask=True)
        position_ids = torch.arange(0, embeddings["input_ids"].size(1), dtype=torch.long).unsqueeze(0).repeat(
            embeddings["input_ids"].size(0), 1).to("cuda")

        input_ids = embeddings["input_ids"].to("cuda")
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        attention_mask = embeddings["attention_mask"]
        if is_ocr:
            token_type_embeddings = self.token_type_embeddings(torch.zeros_like(input_ids).to("cuda"))
        else:
            token_type_embeddings = self.token_type_embeddings(embeddings["token_type_ids"].to("cuda"))

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, attention_mask


class ImageEncoder(nn.Module):
    def __init__(self, num_init_features=None):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=1, bias=False)),
            ('norm', nn.BatchNorm2d(num_init_features)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

    def forward(self, img):
        # @todo features of batches of image
        img_feat = self.conv(img)
        return img_feat


class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states) #@todo adjust instead of list, return tensor
        if not output_all_encoded_layers:
            return hidden_states

        return all_encoder_layers


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
        self.txt_encoder = BertEmbeddings(config)

        # image encoder
        self.img_encoder = models.alexnet()
        self.img_lin = nn.Linear(256*14*19, 768)
        # self.img_encoder = torch.nn.Sequential(*list(models.resnet152().children())[:8])
        # self.img_encoder = ImageEncoder(config.img_dim)
        # self.img_encoder = torch.nn.Sequential(*list(models.vgg16().children())[0])

        # fusion
        self.pooler = BertPooler(config)
        self.fusion = UniterEncoder(config)

        # classifier
        self.classifier = Classifier(num_classes, config)

    def forward(self, img, txt, txt_len, ocr, ocr_len):
        txt_feat, attention_mask = self.txt_encoder(txt, txt_len)
        ocr_feat, ocr_attention_mask = self.txt_encoder(ocr, ocr_len, is_ocr=True)
        img_feat = self.img_encoder.features(img)
        img_feat = img_feat.reshape(img_feat.shape[0], -1)
        img_feat = self.img_lin(img_feat)
        img_feat = img_feat.unsqueeze(1)

        mm_feat = torch.cat([img_feat, txt_feat, ocr_feat], dim=1)
        attention_mask = torch.cat(
            (attention_mask, ocr_attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long)), 1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # todo slice image tensor before sending to transformer layers
        mm_feat = self.fusion(mm_feat, extended_attention_mask, output_all_encoded_layers=False)
        mm_feat = self.pooler(mm_feat)

        out = self.classifier(mm_feat)
        return out
