import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvBlock(nn.Module):
    def __init__(self, chin, chout, kernel_size=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU()
        )
        self.mp = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        res = self.seq(x)
        res = self.mp(res)
        return res


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 224x224
        self.conv_block1 = ConvBlock(3, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.conv_block4 = ConvBlock(128, 256)
        self.conv_block5 = ConvBlock(256, 256)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12544, 1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        return x


base_cnn_model = CNN().to(device)

transfer_model = models.densenet201(pretrained=True)
transfer_model.classifier = nn.Sequential(
    nn.Linear(in_features=1920, out_features=1, bias=True),
    nn.Sigmoid()
)

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1, bias=True),
    nn.Sigmoid()
)
resnet = resnet.to(device)

for p in transfer_model.features.parameters():
    p.requires_grad = False

transfer_model = transfer_model.to(device)

eff_model = EfficientNet.from_pretrained('efficientnet-b5')
eff_model._fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1, bias=True),
    nn.Sigmoid())
eff_model = eff_model.to(device)
