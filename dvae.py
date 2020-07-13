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

class dVAE_Encoder(nn.Module):

    def __init__(self):
        super(dVAE_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, 6, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, dilation=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 6, stride=1, dilation=2, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 5, stride=3, dilation=2, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = x.float().cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        print(x.shape)
        x = self.conv2(F.leaky_relu(x))
        x = self.bn2(x)
        print(x.shape)
        x = self.conv3(F.leaky_relu(x))
        x = self.bn3(x)
        print(x.shape)
        x = self.conv4(F.leaky_relu(x))
        x = self.bn4(x)
        print(x.shape)
        # print("---------------------------------")
        # print(x.size())
        return x


class dVAE_Decoder(nn.Module):

    def __init__(self):
        super(dVAE_Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(8, 4, 6, stride=1, dilation=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, dilation=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 16, 6, stride=1, dilation=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 16, 5, stride=3, dilation=2, padding=1)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = x.float().cuda()
        x = self.conv4(x)
        x = self.bn4(x)
        print(x.shape)
        x = self.conv3(F.leaky_relu(x))
        x = self.bn3(x)
        print(x.shape)
        x = self.conv2(F.leaky_relu(x))
        x = self.bn2(x)
        print(x.shape)
        x = self.conv1(F.leaky_relu(x))
        x = self.bn1(x)
        print(x.shape)
        # print(x.size())
        return x


class dVAE(nn.Module):
    def __init__(self):
        super(dVAE, self).__init__()
        self.encoder = dVAE_Encoder()

        self.conv_mu = nn.Conv2d(16, 16, 3, stride=1, dilation=1)
        self.conv_logvar = nn.Conv2d(16, 16, 3, stride=1, dilation=1)

        self.decoder = dVAE_Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.float().cuda()
        latent = self.encoder(x)

        mu = self.conv_mu(latent)
        logvar = self.conv_logvar(latent)

        z = self.reparameterize(mu, logvar)
        print("----------")
        print(z.shape)
        print("----------")
        answer = self.decoder(z)
        
        return answer, mu, logvar

    import torch
    from torch.utils import data
    import random
    from torch.utils.data import dataloader, random_split

