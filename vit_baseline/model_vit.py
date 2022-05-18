import torch
import torch.nn as nn
from PIL import Image
from apex.normalization.fused_layer_norm import FusedLayerNorm
from layer import GELU
from transformers import BertTokenizer, BertModel
from transformers import ViTFeatureExtractor, ViTModel


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
        # encoded_q = torch.mean(out[0], dim=1).squeeze()
        return out[0]


class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, img_list):
        image_list = []
        for img in img_list:
            img = Image.open(img).convert('RGB') # set 'RGB' to 'L' if black-white image
            image_list.append(img)

        inputs = self.feature_extractor(image_list, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)

        img_feat = outputs.last_hidden_state.to("cuda")
        return img_feat


class RecurrentFusion(nn.Module):
    def __init__(self, num_bigru_units, feat_in):
        super(RecurrentFusion, self).__init__()
        self.bigru = nn.GRU(input_size=feat_in,
                            hidden_size=num_bigru_units,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, mmc_feat):
        mmc_feat = torch.transpose(mmc_feat, 1, 2)
        output, h = self.bigru(mmc_feat)
        h_flattened = torch.flatten(torch.transpose(h, 0, 1), start_dim=1)
        return h_flattened


class Classifier(nn.Module):
    def __init__(self, num_classes, config):
        super(Classifier, self).__init__()
        hidden_size = config.hidden_size*4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            GELU(),
            FusedLayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, mm_feat):
        # mm_feat = mm_feat[:, 0, :]
        out = self.classifier(mm_feat)
        return out


class ChartFCBaseline(nn.Module):
    def __init__(self, token_count, num_classes, config):
        super(ChartFCBaseline, self).__init__()
        # text encoder
        self.txt_encoder = BertEncoder(config)

        # image encoder = ViT
        self.img_encoder = ViTEncoder()

        # fusion
        self.bn = nn.BatchNorm1d(config.fusion_out_dim)
        self.transform_convs = []
        self.num_mmc_units = config.fusion_out_dim
        self.transform_convs.append(nn.Conv1d(config.fusion_out_dim, self.num_mmc_units, kernel_size=1))
        self.transform_convs.append(nn.ReLU())
        for i in range(2):
            self.transform_convs.append(nn.Conv1d(self.num_mmc_units, self.num_mmc_units, kernel_size=1))
            self.transform_convs.append(nn.ReLU())
        self.transform_convs = nn.Sequential(*self.transform_convs)
        self.rf = RecurrentFusion(config.fusion_out_dim, config.fusion_out_dim)

        # classifier
        self.classifier = Classifier(num_classes, config)

    def forward(self, img, txt):
        # encodingx
        txt_feat = self.txt_encoder(txt)
        img_feat = self.img_encoder(img)

        # fusion
        final_feat = torch.cat([img_feat, txt_feat], dim=1)
        final_feat = torch.transpose(final_feat, 1, 2)
        final_feat = self.bn(final_feat)
        final_feat = self.transform_convs(final_feat)
        final_feat = self.rf(final_feat)

        # classifier
        # final_feat = torch.transpose(final_feat, 1, 2)
        out = self.classifier(final_feat)
        return out
