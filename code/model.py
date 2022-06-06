import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_out_dim * 2, config.fusion_out_dim * 4),
            nn.GELU(),
            # nn.Dropout(),
            FusedLayerNorm(config.fusion_out_dim * 4, eps=1e-12),
            nn.Linear(config.fusion_out_dim * 4, config.num_classes)
        )

    def forward(self, mm_features):
        out = self.classifier(mm_features)
        return out


class ChartFCBaseline(nn.Module):
    def __init__(self, config):
        super(ChartFCBaseline, self).__init__()
        self.text_encoder = config.COMPONENTS[config.txt_encoder](config)
        self.image_encoder = config.COMPONENTS[config.img_encoder](config)
        self.fusion = config.COMPONENTS[config.fusion_method](config)
        self.classifier = Classifier(config)

    def forward(self, img, txt, txt_encode, txt_len, ocr, ocr_len):
        # Unimodal encoding
        txt_features = self.text_encoder(txt, txt_encode, txt_len)
        ocr_features = self.text_encoder(ocr, None, txt_len)
        img_features = self.image_encoder(img)

        # Fusion
        mm_features_1 = self.fusion(txt_features, img_features)
        mm_features_2 = self.fusion(ocr_features, img_features)
        mm_features = torch.cat([mm_features_1, mm_features_2], dim=1)

        # Classification
        out = self.classifier(mm_features)

        return out
