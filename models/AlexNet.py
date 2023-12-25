import torch
import torch.nn as nn
from fft_conv_pytorch import FFTConv2d

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool=True, fourier=False):
        super(ConvBlock, self).__init__()

        if fourier:
            self.conv = FFTConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if pool:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        else:
            self.pool = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pool(self.relu(self.conv(x)))
        return out
    

class AlexNet(nn.Module):
    def __init__(self, fourier=False):
        super(AlexNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 11, 4, 2, pool=True, fourier=fourier)
        self.conv2 = ConvBlock(64, 192, 5, 1, 2, pool=True, fourier=fourier)
        self.conv3 = ConvBlock(192, 384, 3, 1, 1, pool=False, fourier=False)
        self.conv4 = ConvBlock(384, 256, 3, 1, 1, pool=False, fourier=False)
        self.conv5 = ConvBlock(256, 256, 3, 1, 1, pool=True, fourier=False)
        self.last_seq = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        return self.last_seq(x)    


if __name__=='__main__':
    x = torch.rand(10, 3, 227, 227)
    model = AlexNet(fourier=False)
    print(model(x).shape)

