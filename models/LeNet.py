import torch
import torch.nn as nn
from fft_conv_pytorch import FFTConv2d

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, fourier=False):
        super(ConvBlock, self).__init__()

        if fourier == True:
            self.conv = FFTConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.pool(self.conv(x)))
        return out
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 16*5*5)
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out
    

class LeNet(nn.Module):
    def __init__(self, fourier=False):
        super(LeNet, self).__init__()

        self.conv1 = ConvBlock(1, 6, kernel_size=5, stride=1, padding=2, fourier=fourier)
        self.conv2 = ConvBlock(6, 16, kernel_size=5, stride=1, padding=0, fourier=fourier)
        self.classifier = Classifier()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out
    
if __name__=='__main__':
    x = torch.randn(64, 1, 28, 28)
    model = LeNet(fourier=True)
    print(model(x).shape)