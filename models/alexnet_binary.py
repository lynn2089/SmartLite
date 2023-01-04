import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
import time

__all__ = ['alexnet_binary']

class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl=1
        self.features0 = nn.Sequential(
            BinarizeConv2d(3, int(64*self.ratioInfl), kernel_size=11, stride=4, padding=2)
        )
        self.features1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(96)),
            nn.Hardtanh(inplace=True)
        )
        self.features2 = nn.Sequential(
            BinarizeConv2d(int(96), int(256), kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(256)),
            nn.Hardtanh(inplace=True)
        )
        self.features3 = nn.Sequential(
            BinarizeConv2d(int(256), int(384*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(384*self.ratioInfl)),
            nn.Hardtanh(inplace=True)
        )
        self.features4 = nn.Sequential(
            BinarizeConv2d(int(384*self.ratioInfl), int(384*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(384*self.ratioInfl)),
            nn.Hardtanh(inplace=True)
        )
        self.features5 = nn.Sequential(
            BinarizeConv2d(int(384*self.ratioInfl), 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True)
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(4096, num_classes),
            nn.BatchNorm1d(1000),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features_0(x)
        x = torch.sign(x)
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
