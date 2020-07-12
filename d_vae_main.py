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

torch.device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
transform = transforms.ToTensor()


# define the NN architecture

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
        answer = self.decoder(z)
        return answer, mu, logvar

    import torch
    from torch.utils import data
    import random
    from torch.utils.data import dataloader, random_split


if __name__ == '__main__':

    roots = (['E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\HGG\\',
              'E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\LGG\\'])

    patient_folders = [os.path.join(roots[0], p) for p in os.listdir(roots[0])] + \
                      [os.path.join(roots[1], p) for p in os.listdir(roots[1])]

    data_full = utils.DataLoader.download_data(patient_folders)

    # data_full = np.load("dataset_healthy.npy")
    train_data, valid_data, test_data = utils.PreProcessor.split(data_full, True)

    # train_data, valid_data, test_data = np.load("E:\Lab\Lab_VDA_Local\dataset_split.npy", allow_pickle=True)

    batch_size = 8

    train_dataloader = torch.utils.data.DataLoader(train_data[0:100], batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data[0:50], batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # initialize the NN
    model = dVAE()
    # print(model)
    model.cuda()
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = 1000

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data in train_dataloader:
            # _ stands in for labels, here
            # no need to flatten images
            images = data.float()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            images.cuda()
            outputs, _, _ = model(images)
            # calculate the loss
            outputs = F.pad(outputs, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)
            images.to(torch.device("cuda:0"))
            outputs.to(torch.device("cuda:0"))
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))
        if epoch % 10 == 0:
            with torch.no_grad():
                valid_loss = []
                current_valid_loss = 0.0
                for data in valid_dataloader:
                    images = data.float()
                    optimizer.zero_grad()
                    images.cuda()
                    outputs, _, _ = model(images)

                    # noinspection PyTypeChecker
                    outputs = F.pad(outputs, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)
                    images.to(torch.device("cuda:0"))
                    outputs.to(torch.device("cuda:0"))
                    loss = criterion(outputs, images)
                    loss.backward()
                    optimizer.step()
                    current_valid_loss += loss.item() * images.size(0)
                    valid_loss.append(current_valid_loss)
            # noinspection PyUnresolvedReferences
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_valid_loss
            }, "./Model_12_08.pt")

    # torch.save(model.state_dict(), "./Model_10_08.pt")

    # obtain one batch of test images
    dataiter = iter(train_dataloader)
    images = dataiter.next().float()

    # get sample outputs
    output, _, _ = model(images)
    output = F.pad(output, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)
    # prep images for display
    images = images.cpu().numpy()

    # output is resized into a batch of iages
    output = output.view(batch_size, 4, 240, 240)
    # use detach when it's an output that requires_grad
    output = output.cpu().detach().numpy()

    # plot the first ten input images and then reconstructed images
    # fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    f1 = plt.figure(figsize=(12, 12))
    ax1 = f1.add_subplot(221)
    ax2 = f1.add_subplot(222)
    ax3 = f1.add_subplot(223)
    ax4 = f1.add_subplot(224)

    ax1.imshow(images[1][0], cmap="gray")
    ax2.imshow(images[1][1], cmap="gray")
    ax3.imshow(images[1][2], cmap="gray")
    ax4.imshow(images[1][3], cmap="gray")
    plt.show()

    f2 = plt.figure(figsize=(12, 12))
    ax5 = f2.add_subplot(221)
    ax6 = f2.add_subplot(222)
    ax7 = f2.add_subplot(223)
    ax8 = f2.add_subplot(224)

    ax5.imshow(output[0][0], cmap="gray")
    ax6.imshow(output[0][1], cmap="gray")
    ax7.imshow(output[0][2], cmap="gray")
    ax8.imshow(output[0][3], cmap="gray")
    plt.show()

    f3 = plt.figure(figsize=(12, 12))
    ax11 = f3.add_subplot(221)
    ax12 = f3.add_subplot(222)
    ax13 = f3.add_subplot(223)
    ax14 = f3.add_subplot(224)

    ax11.imshow(output[0][0] - images[1][0], cmap="gray")
    ax12.imshow(output[0][1] - images[1][1], cmap="gray")
    ax13.imshow(output[0][2] - images[1][2], cmap="gray")
    ax14.imshow(output[0][3] - images[1][3], cmap="gray")
    plt.show()
