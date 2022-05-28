from abc import abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.densenet import _DenseBlock, _Transition
from transformers import ViTFeatureExtractor, ViTModel


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, img):  # @todo special case: ViT
        pass


class SimpleImageEncoder(ImageEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.img_dim = 768
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, config.img_dim, kernel_size=7, stride=2, padding=1, bias=False)),
            ('norm', nn.BatchNorm2d(config.img_dim)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

    def forward(self, img):
        img_feat = self.conv(img)
        return img_feat


class AlexNetEncoder(ImageEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.img_dim = 768
        self.alexnet = models.alexnet()
        self.img_lin = nn.Linear(256*14*19, config.img_dim)

    def forward(self, img):
        img_feat = self.alexnet.features(img)
        img_feat = img_feat.reshape(img_feat.shape[0], -1)
        img_feat = self.img_lin(img_feat)
        img_feat = img_feat.unsqueeze(-1).unsqueeze(-1)
        return img_feat


class ResNetEncoder(ImageEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.img_dim = 2048
        self.resnet = nn.Sequential(*list(models.resnet152().children())[:8])

    def forward(self, img):
        img_feat = self.resnet(img)
        return img_feat


class DenseNetEncoder(ImageEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.block_config = block_config=(6, 6, 6)
        num_init_features = 64

        self.first_conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU()),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.denseblock = [] # list of dense blocks
        num_features = num_init_features
        growth_rate = 32
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=4, growth_rate=growth_rate, drop_rate=0)

            self.denseblock.append(nn.Sequential(OrderedDict([
                (f'dblock{i}', block),
            ])))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.denseblock[i].add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)

        # Official init from torch repo.  # @todo initialise also other models, e.g. resnet and alexnet?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.denseblock = nn.ModuleList(self.denseblock)
        config.img_dim = config.densenet_dim[0] + num_features

    def forward(self, img):
        first_conv_feat = self.first_conv(img)
        denseblock_feat = [self.denseblock[0](first_conv_feat)]
        for i in range(len(self.block_config) - 1):
            denseblock_feat.append(self.denseblock[i + 1](denseblock_feat[i]))

        final_feat = self.final_bn(denseblock_feat[-1])
        final_feat = final_feat.repeat(1, 1, 2, 2)
        bs, f1, f2, f3 = final_feat.shape
        final_feat = torch.cat([final_feat, torch.zeros(bs, f1, 1, f3).cuda()], dim=2)
        final_feat = torch.cat([final_feat, torch.zeros(bs, f1, (f2+1), 1).cuda()], dim=3)

        out = torch.cat([denseblock_feat[0], final_feat], dim=1)
        return out


class ViTEncoder(ImageEncoder):
    def __init__(self, config):
        super().__init__(config)
        config.img_dim = 768
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, img):
        image_list = list(img.cpu()) # tensor to list of tensors
        inputs = self.feature_extractor(image_list, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.vit(**inputs)

        img_feat = outputs.last_hidden_state.to("cuda")
        img_feat = img_feat.unsqueeze(-1)
        img_feat = img_feat.permute(0, 2, 1, 3)
        return img_feat
