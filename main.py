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
from dvae import dVAE
from cvae import CVAE
from vae_base import VAE_Base
from autoencoder import OrigAE

torch.device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
transform = transforms.ToTensor()
import torch
from torch.utils import data
import random
import h5py
from torch.utils.data import dataloader, random_split
import scipy.misc

def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.cpu().detach().numpy()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def to_distribution(D):
    D_ = D.view(-1, D.shape[2], D.shape[3])
    D_ = D_.view(D_.shape[0], -1)
    D_norm = D.view(D.shape[0], D.shape[1], -1)
    D_norm = D_norm.sum(dim=2).view(-1)
    D = (D_/D_norm.unsqueeze(1)).view(*D.shape)
    return D

def elbo(recon_x, x, mu, logsig):

        N, C, iw, ih = recon_x.shape
        M=1
        x = x.contiguous().view([N*M,C,iw,ih])
        recon_x = recon_x.view([N,C,iw,ih])
        loss = nn.CrossEntropyLoss()
        BCE =  loss(recon_x, x)/ (N*M)
        KLD_element = (logsig - mu**2 - torch.exp(logsig) + 1 )
        KLD = - torch.mean(torch.sum(KLD_element* 0.5, dim=2) )

        return BCE + KLD

def loss_function(x,recon_x, mu, logsig):

        N, C, iw, ih = x.shape
        x_tile = x.repeat(8,1,1,1,1).permute(1,0,2,3,4)
        #J = - self.log_likelihood_estimate(recon_x, x_tile, Z, mu, logsig)
        J_low = elbo(recon_x, x, mu, logsig)
        return J_low


def loss_function_git(recons,input,mu,log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = 0.5 # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        try:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0).sum()
        except:
            kld_loss = 0

        loss = recons_loss + kld_weight * kld_loss
        return loss


if __name__ == '__main__':

    roots = (['E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\HGG\\',
              'E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\LGG\\'])

    patient_folders = [os.path.join(roots[0], p) for p in os.listdir(roots[0])] + \
                      [os.path.join(roots[1], p) for p in os.listdir(roots[1])]

    file = "h5py"
    if file == "npy":
        train_data, valid_data, test_data = np.load("E:\\Lab\\resources\\dataset_split.npy", allow_pickle=True)
        train_data_aug = np.load("E:\\Lab\\resources\\dataset_healthy_aug_new.npy", allow_pickle=True)

    elif file == "h5py":
        dataset = h5py.File("E:\\Lab\\resources\\dataset_healthy.h5", "r")

        train_data, valid_data, _ = dataset["train_imgs"], dataset["valid_imgs"], dataset["test_imgs"]
        dataset = h5py.File("E:\\Lab\\resources\\dataset_healthy_aug.h5", "r")

        train_data_aug = dataset["train_imgs"]

    train_data = np.array((train_data, train_data_aug))
    train_data = np.swapaxes(train_data, 0, 1)

    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(train_data[0:100], batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data[0:10], batch_size=batch_size, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # initialize the NN
    model = dVAE()
    # print(model)
    model.cuda()
    # specify loss function
    criterion = nn.MSELoss()

    # criterion = nn.CrossEntropyLoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # number of epochs to train the model
    n_epochs = 500
    train_loss_data = []
    valid_loss = []
    valid_loss1 =[]
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        if epoch == 300:
            for g in optimizer.param_groups:
                print("Lerning Rate decreased")
                g['lr'] = 0.01
        ###################
        # train the model #
        ###################
        for batch_id, xy in enumerate(train_dataloader):
            img = xy[:, 0, :, :]
            aug_img = xy[:, 1, :, :]
            optimizer.zero_grad()
            img = img.float().cuda()
            aug_img = aug_img.float().cuda()


            # forward pass: compute predicted outputs by passing inputs to the model
            outputs, mu, logvar = model(aug_img)

            #outputs = to_distribution(outputs)
            # x1 = to_distribution(x)

            loss = loss_function_git(outputs, img, mu, logvar)

            loss1 = criterion(img, outputs)

            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * aug_img.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_dataloader)
        train_loss_data.append(train_loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))
        if epoch % 10 == 0:
            with torch.no_grad():

                current_valid_loss = 0.0
                for batch_id, img in enumerate(valid_dataloader):
                    optimizer.zero_grad()
                    img = img.float().cuda()

                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs, mu, logvar = model(img)

                    # calculate the loss
                    # outputs = F.pad(outputs, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)
                    outputs.cuda()
                    outputs = to_distribution(outputs)
                    img = to_distribution(img)

                    loss = loss_function_git(outputs, img, mu, logvar)
                    loss1 = criterion(img, outputs)

                    current_valid_loss += loss1.item() * img.size(0)
                    valid_loss1.append(current_valid_loss)
                    valid_loss.append(loss.item() * img.size(0))
            # noinspection PyUnresolvedReferences
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_valid_loss
            }, "./Model_22_08.pt")

    # torch.save(model.state_dict(), "./Model_10_08.pt")
    plt.plot(train_loss_data)
    plt.show()
    plt.plot(valid_loss)
    plt.show()
    plt.plot(valid_loss1)
    plt.show()