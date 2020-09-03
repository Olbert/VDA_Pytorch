import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from sklearn.mixture import GaussianMixture
import main
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.morphology
import torch
from torch.utils import data
import random
import h5py

torch.device("cuda")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
transform = transforms.ToTensor()

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

def rescale_linear(array, new_min, new_max):
        """Rescale an arrary linearly."""
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m * array + b


def rescale_linear4(img, new_min, new_max):
    for i in range(0, 3):
        array = img[i]
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        img[i] = m * array + b
    return img

if __name__ == '__main__':

    roots = (['E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\HGG\\',
              'E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\LGG\\'])

    patient_folders = [os.path.join(roots[0], p) for p in os.listdir(roots[0])] + \
                      [os.path.join(roots[1], p) for p in os.listdir(roots[1])]


    model = main.OrigAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint = torch.load("E:\\Lab\\Models\\final\\Model_OrigAE_350.pt")
    # checkpoint = torch.load("./Model_22_08.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    model.eval()


    test_data = np.load('E:\\Lab\\resources\\dataset_unhealthy.npy', allow_pickle=True)
    test_data_mask = np.load('E:\\Lab\\resources\\dataset_mask.npy', allow_pickle=True)

    test_data = np.hstack((test_data, test_data_mask[:, np.newaxis, :, :]))
    batch_size = 1

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    sums_error = np.zeros(test_data.shape[0]+1)
    i = 0

    show = True

    for batch_id, xy in enumerate(test_dataloader):
        images = xy[:, 0:4, :, :]
        mask = xy[0, 4, :, :]

        images = images.float().cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        output, mu, logvar = model(images)
        # outputs = F.pad(outputs, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)

        output = output.cpu().detach().numpy()
        images = images.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()

        output[0] = rescale_linear4(output[0], 0, 1)
        images[0] = rescale_linear4(images[0], 0, 1)
        mask[mask>0] = 1



        new_image = output - images
        new_image[0] = rescale_linear4(new_image[0], 0, 1)

        # badIndices = (new_image[0] > 0.3)
        # new_image[0][badIndices] = 1

        n = 6

        # new_image = 0 # !!
        l = 240
        img = new_image[0, 0]

        classif = GaussianMixture(n_components=3, n_init=1)
        classif.fit(img.reshape((img.size, 1)))

        cluster = classif.predict(img.reshape((img.size, 1)))
        cluster = cluster.reshape(240, 240)

        sums = []
        for z in range(0, 3):
            sums.append((cluster == z).sum())
        tumor_id = np.argmin(sums)
        tumor = (cluster == tumor_id).astype(bool)


        #tumor = skimage.morphology.remove_small_objects(tumor, min_size=20)


        if (show):
            f1 = plt.figure(figsize=(12, 12))


            ax3 = f1.add_subplot(221)
            ax3.imshow(images[0][0], cmap="gray")

            ax1 = f1.add_subplot(222)
            ax1.imshow(new_image[0][0], cmap="gray")

            ax2 = f1.add_subplot(223)
            ax2.imshow(mask, cmap="gray")

            ax2 = f1.add_subplot(224)
            ax2.imshow(mask - tumor, cmap="gray")

            f2 = plt.figure(figsize=(12, 12))


            ax3 = f2.add_subplot(221)
            ax3.imshow(tumor, cmap="gray")

            ax1 = f2.add_subplot(222)
            ax1.imshow(cluster, cmap="gray")

            ax2 = f2.add_subplot(223)
            test = skimage.morphology.binary_closing(tumor)
            test = skimage.morphology.remove_small_objects(test, min_size=50)
            ax2.imshow(test, cmap="gray")


            plt.show()

        tumor = skimage.morphology.remove_small_objects(tumor, min_size=50)

        intersection = np.logical_and(mask, test)
        union = np.logical_or(mask, tumor)
        iou_score = np.sum(intersection) / np.sum(union)

        #sum = np.count_nonzero(np.abs(mask - tumor))
        sums_error[i] = iou_score
        i+=1

    print(sums_error.mean())
    np.save("iou_AE", sums_error)
    plt.plot(sums_error)
    plt.show()