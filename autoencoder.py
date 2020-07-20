import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.densenet import DenseNet
# convert data to torch.FloatTensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import os


class OrigEncoder(nn.Module):

    def __init__(self):
        super(OrigEncoder, self).__init__()

        self.conv1_1 = nn.Conv2d(4, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.bn2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.bn3_1 = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.bn4_1 = nn.BatchNorm2d(512)

        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x = x.float().cuda()
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)


        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)

        x = self.pool2(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.bn3_1(x)

        x = self.pool3(x)

        x = self.conv4_1(x)
        x = F.relu(x)

        x = self.pool4(x)

        # print("---------------------------------")
        # print(x.size())
        return x


class OrigDecoder(nn.Module):

    def __init__(self):
        super(OrigDecoder, self).__init__()

        self.conv1_1 = nn.ConvTranspose2d(4, 64, 3)
        self.conv1_2 = nn.ConvTranspose2d(64, 64, 3)
        self.pool1 = nn.Upsample(2)

        self.conv2_1 = nn.ConvTranspose2d(64, 128, 3)
        self.bn2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.ConvTranspose2d(128, 128, 3)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.pool2 = nn.Upsample(2)

        self.conv3_1 = nn.ConvTranspose2d(512, 256, 3)
        self.bn3_1 = nn.BatchNorm2d(256)

        self.pool3 = nn.Upsample(2)

        self.conv4_1 = nn.ConvTranspose2d(512, 512, 3)

        self.conv5_1 = nn.ConvTranspose2d(512, 512, 3)

        self.up = nn.Upsample(2)

    def forward(self, x):
        x = x.float().cuda()

        x = self.up(x)
        x = self.conv5_1(x)
        x = F.relu(x)


        x = self.up(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = F.relu(x)


        x = self.up(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.bn3_1(x)


        x = self.up
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.bn2_2(x)


        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.bn2_1(x)
        x = F.relu(x)

        x = self.conv1_1(F.relu(x))
        x = F.relu(x)
        x = self.bn1_1(x)
        print(x.size())
        z = self.conv0_0(F.relu(x))
        # print("---------------------------------")
        # print(x.size())
        return x


class OrigAE(nn.Module):
    def __init__(self):
        super(OrigAE, self).__init__()
        self.encoder = OrigEncoder()
        self.decoder = OrigDecoder()


    def forward(self, x):
        x = x.float().cuda()
        z = self.encoder(x)
        print("----------")
        print(z.shape)
        print("----------")
        answer = self.decoder(z)

        z = F.sigmoid(z)

        return z


