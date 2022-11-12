"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
__all__ = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(pretrained=False, num_classes=1000):
    model = VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def vgg13_bn(pretrained=False, num_classes=1000):
    model = VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model

def vgg16_bn(pretrained=False, num_classes=1000):
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

def vgg19_bn(pretrained=False, num_classes=1000):
    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model