from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, chin, chout, kernel_size=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=chin, out_channels=chout, kernel_size=kernel_size, padding=0, stride=2),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        res = self.seq(x)
        return res


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 16)
        self.conv_block2 = ConvBlock(16, 32)
        self.conv_block3 = ConvBlock(32, 64)

        self.fc1 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
