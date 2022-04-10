from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    def __init__(self, token_count=None, config=None):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(token_count, config.txt_embedding_dim)
        self.lstm = nn.LSTM(input_size=config.txt_embedding_dim, hidden_size=config.lstm_dim, num_layers=2)
        self.drop = nn.Dropout() # MCB

    def forward(self, txt, txt_len):
        embedding = self.embedding(txt)
        embedding = torch.tanh(embedding) # MCB specific

        packed = pack_padded_sequence(embedding, txt_len, batch_first=True, enforce_sorted=False)
        o, (h, c) = self.lstm(packed)
        txt_feat = torch.cat([c.squeeze(0)[0], c.squeeze(0)[1]], dim=1) # LSTM output layer 1 and layer 2 concat
        txt_feat = self.drop(txt_feat)
        return txt_feat


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


class Classifier(nn.Module):
    def __init__(self, num_classes, config):
        super(Classifier, self).__init__()
        self.config = config
        pooling_size = 6
        self.avg_pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_classifier),
            nn.Linear(config.fusion_out_dim*pooling_size*pooling_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_classifier),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # self.relu = nn.ReLU()
        # self.drop = nn.Dropout()
        # self.classifier = nn.Linear(mm_feat_dim, num_classes)

    def forward(self, mm_feat):
        mm_feat = self.avg_pool(mm_feat)
        mm_feat = mm_feat.view(mm_feat.shape[0], -1)

        # projection = self.relu(bimodal_emb)
        # projection = self.drop(projection)
        out = self.classifier(mm_feat)

        return out


class ChartFCBaseline(nn.Module):
    def __init__(self, token_count, num_classes, config):
        super(ChartFCBaseline, self).__init__()
        self.txt_encoder = TextEncoder(token_count, config)
        # self.img_encoder = ImageEncoder(config.img_dim)
        self.img_encoder = torch.nn.Sequential(*list(models.resnet152().children())[:8]) #resnet :7 ?
        # self.img_encoder = torch.nn.Sequential(*list(models.vgg16().children())[0])
        self.classifier = Classifier(num_classes, config)

    def forward(self, img, txt, txt_len):
        txt_feat = self.txt_encoder(txt, txt_len)
        img_feat = self.img_encoder(img) # remove .features for own Image encoder or resnet

        mm_feat = None
        out = self.classifier(mm_feat)
        return out
