from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

from compact_bilinear_pooling_layer import CompactBilinearPooling


class TextEncoder(nn.Module):
    def __init__(self, token_count=None, config=None):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(token_count, config.txt_embedding_dim)
        self.lstm = nn.LSTM(input_size=config.txt_embedding_dim, hidden_size=config.text_dim, num_layers=2)
        self.drop = nn.Dropout(0.3) # MCB

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


class MCBPooling(nn.Module):
    def __init__(self, config=None):
        super(MCBPooling, self).__init__()
        mcb_out_dim = 16000
        self.config = config
        self.comp_layer1 = CompactBilinearPooling(config.fusion_out_dim, config.fusion_out_dim, mcb_out_dim, sum_pool=False)
        self.conv1 = nn.Conv2d(mcb_out_dim, 512, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        # weights
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.comp_layer2 = CompactBilinearPooling(config.fusion_out_dim, config.fusion_out_dim, mcb_out_dim, sum_pool=False)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(mcb_out_dim, 1),
            # nn.Dropout(),
            # nn.ReLU(inplace=True),
        )

    def forward(self, txt_feat, img_feat):
        # L2 norm of img_feat
        img_feat = torch.nn.functional.normalize(img_feat)

        # reshape and tile txt_feat
        bs, _, nw, nh = img_feat.shape
        txt_feat = torch.unsqueeze(txt_feat, -1)
        txt_feat = torch.unsqueeze(txt_feat, -1)
        txt_tile = torch.tile(txt_feat, (int(self.config.img_dim/(self.config.text_dim*2)), nh, nw))

        # 1st MCB
        out = self.comp_layer1(txt_tile, img_feat)
        out = out.permute(0, 3, 1, 2)
        out = torch.sqrt(F.relu(out)) - torch.sqrt(F.relu(-out)) # todo square root
        out = torch.nn.functional.normalize(out)

        # weights
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        out = out.reshape(-1, 1, nw * nh)
        weights = nn.functional.softmax(out, dim=2)
        weights = weights.reshape((-1, 1, nw, nh)) # or other way around nh, nw

        # apply weights to image vector
        bottom1_resh = img_feat.view(bs, nw * nh, -1)
        weights = weights.view(bs, -1, 1)

        res = torch.bmm(bottom1_resh.transpose(1, 2), weights)
        res = res.squeeze(2)

        # prepare data for 2nd MCB
        res_unsqueezed = res.unsqueeze(-1).unsqueeze(-1)

        # 2nd call of MCB
        out2 = self.comp_layer2(res_unsqueezed, txt_feat)
        out2 = out2.squeeze()
        out2 = torch.sqrt(F.relu(out2)) - torch.sqrt(F.relu(-out2)) # square root
        out2 = torch.nn.functional.normalize(out2)

        # classifier
        final_out = self.classifier(out2)
        return final_out


class ChartFCBaseline(nn.Module):
    def __init__(self, token_count, num_classes, config):
        super(ChartFCBaseline, self).__init__()
        self.txt_encoder = TextEncoder(token_count, config)

        self.img_encoder = torch.nn.Sequential(*list(models.resnet152().children())[:8])
        # self.img_encoder = ImageEncoder(config.img_dim)
        # self.img_encoder = torch.nn.Sequential(*list(models.vgg16().children())[0])
        # self.img_encoder.freeze() # freeze image encoder weights

        self.mcb = MCBPooling(config)

    def forward(self, img, txt, txt_len):
        txt_feat = self.txt_encoder(txt, txt_len)
        # ocr_feat = self.txt_encoder(ocr, ocr_len)
        img_feat = self.img_encoder(img) # remove .features for own Image encoder or resnet
        out = self.mcb(txt_feat, img_feat)
        return out
