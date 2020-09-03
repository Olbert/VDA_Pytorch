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


class CVAE_Encoder(nn.Module):

    def __init__(self):
        super(CVAE_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = x.float().cuda()
        x = self.conv1(x)
        x = self.bn1(x)
        # print(x.shape)
        x = self.conv2(F.leaky_relu(x))
        x = self.bn2(x)
        # print(x.shape)
        x = self.conv3(F.leaky_relu(x))
        x = self.bn3(x)
        # print(x.shape)
        x = self.conv4(F.leaky_relu(x))
        x = self.bn4(x)
        # print(x.shape)
        # print("---------------------------------")
        # print(x.size())
        return x


class CVAE_Decoder(nn.Module):

    def __init__(self):
        super(CVAE_Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)

        self.conv5 = nn.ConvTranspose2d(64, 64, 4, stride=2)  # , padding=1)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.bn5 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = x.float().cuda()

        x = self.conv5(x)
        x = self.bn5(x)
        # print(x.shape)
        x = F.pad(x, (-1, 0, -1, 0, 0, 0), mode='constant', value=0)
        x = self.conv4(F.leaky_relu(x))
        x = self.bn4(x)

        # print(x.shape)
        x = self.conv3(F.leaky_relu(x))
        x = self.bn3(x)
        # print(x.shape)
        x = self.conv2(F.leaky_relu(x))
        x = self.bn2(x)
        # print(x.shape)
        x = self.conv1(F.leaky_relu(x))
        x = self.bn1(x)
        # print(x.shape)
        # print(x.size())
        return x


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = CVAE_Encoder()

        self.conv_mu = nn.Conv2d(64, 64, 3, stride=2)
        self.conv_logvar = nn.Conv2d(64, 64, 3, stride=2)

        self.decoder = CVAE_Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.float().cuda()
        z = self.encoder(x)

        mu = self.conv_mu(z)
        logvar = self.conv_logvar(z)
        # mu_f = torch.flatten(mu, start_dim=1)
        # logvar_f = torch.flatten(logvar, start_dim=1)
        z = self.reparameterize(mu, logvar)
        # print("----------")
        # print(z.shape)
        # print("----------")
        # z = z.view(-1,64,7,7)
        answer = self.decoder(z)
        return answer, mu, logvar
