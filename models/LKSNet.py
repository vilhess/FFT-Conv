import torch
import torch.nn as nn
from fft_conv_pytorch import FFTConv2d

# LKS stands for "Large Kernel Size"

KERNEL_SIZES = [39, 27, 15]
STRIDES = [1, 1, 1]
PADDINGS = [0, 0, 0]
CHANNELS = [3, 5, 7]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, fourier):
        super(ConvBlock, self).__init__()
        if fourier: 
            self.conv = FFTConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    


class LKSNet(nn.Module):
    def __init__(self, fourier=False, kernel_sizes = None, strides = None, paddings = None, channels = None):
        super(LKSNet, self).__init__()

        if kernel_sizes is not None:
            kernel_sizes = kernel_sizes
        else:
            kernel_sizes = KERNEL_SIZES

        if strides is not None:
            strides = strides
        else:
            strides = STRIDES

        if paddings is not None:
            paddings = paddings
        else:
            paddings = PADDINGS

        if channels is not None:
            channels = channels
        else:
            channels = CHANNELS

        channel = 3
        layers = []

        for i in range(len(kernel_sizes)):
            layers.append(ConvBlock(channel, channels[i], kernel_sizes[i], strides[i], paddings[i], fourier))
            channel = channels[i]
        self.features = nn.Sequential(*layers)
        self.classifier = Classifier(channel, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':

    x = torch.randn(1, 3, 224, 224)
    model = LKSNet()
    out = model(x)

    print(out.shape)


