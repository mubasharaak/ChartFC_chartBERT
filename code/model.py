import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim * 2),
            nn.GELU(),
            # nn.Dropout(),
            FusedLayerNorm(config.fusion_dim * 2, eps=1e-12),
            nn.Linear(config.fusion_dim * 2, config.num_classes)
        )

    def forward(self, mm_features):
        out = self.classifier(mm_features)
        return out


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
