
import nibabel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import seaborn as sns
import os
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.misc
import utils

import main

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def rescale_linear4(img, new_min, new_max):

    for i in range(0,3):
        array = img[i]
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        img[i] = m * array + b
    return img

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(0)
random.seed(0)


roots = (['E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\HGG\\',
          'E:\\Lab\\resources\\MICCAI_BraTS_2019_Data_Training\\MICCAI_BraTS_2019_Data_Training\\LGG\\'])

patient_folders = [os.path.join(roots[0], p) for p in os.listdir(roots[0])] + \
                  [os.path.join(roots[1], p) for p in os.listdir(roots[1])]

# data = DataLoader.download_data(patient_folders)


#data = np.load("./dataset.npy")
#train_data, vald_data, test_data = PreProcessor.split(data, save=True)

model = main.VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load("E:\\Lab\\Models\\Model_13_08_1250day.pt")
# checkpoint = torch.load("./Model_12_08.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# model = d_vae_main.dVAE()
# model.load_state_dict(torch.load("./Model_11_08.pt"))
# model.run(train_data, vald_data)


""" Just write model path here"""

# model = torch.load("./Models/Model_10_08.pt")
model.eval()

# _, _, test_data = np.load("E:\Lab\Lab_VDA_Local\dataset_split.npy", allow_pickle=True)
# test_data = np.load("E:\\Lab\\resources\\dataset_unhealthy.npy", allow_pickle=True)

test_data = np.load("./dataset_20unhealthy.npy", allow_pickle=True)
test_data_mask = np.load("./dataset_20mask.npy", allow_pickle=True)

test_data = np.hstack((test_data,test_data_mask[:,np.newaxis,:,:]))
batch_size = 1


test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

dataiter = iter(test_dataloader)
images = dataiter.next().float()


mask = images.data[0,4]
images = images.data[:,0:4]
mask = mask.cpu().detach().numpy()


output,_,_ = model(images)
output = F.pad(output, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)
# prep images for display
images = images.cpu().numpy()
output = output.view(batch_size, 4, 240, 240)
# use detach when it's an output that requires_grad
output = output.cpu().detach().numpy()

# plot the first ten input images and then reconstructed images
# fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

# print(output[0]-images[0])
# print(output[0]-output[0]-mask)


# badIndices = (output[0] < 0)
# output[0][badIndices]=0

# output[0] = rescale_linear4(output[0],0,1500)
# images[0] = rescale_linear4(images[0],0,1500)
mask = rescale_linear(mask,0,1500)

new_image = output - images
# new_image[0] = rescale_linear4(images[0],0,1500)

#
badIndices = (new_image[0] > 800)
new_image[0][badIndices]=1500
#
#
# badIndices = (images[0] == 0)
# images[0][badIndices]=1000

f1 = plt.figure(figsize=(12, 12))
ax1 = f1.add_subplot(221)
ax2 = f1.add_subplot(222)
ax3 = f1.add_subplot(223)
ax4 = f1.add_subplot(224)

ax1.imshow(images[0][0], cmap="gray")
ax2.imshow(images[0][1], cmap="gray")
ax3.imshow(images[0][2], cmap="gray")
ax4.imshow(images[0][3], cmap="gray")
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

ax11.imshow(new_image[0][0], cmap="gray")
ax12.imshow(new_image[0][1], cmap="gray")
ax13.imshow(new_image[0][2], cmap="gray")
ax14.imshow(new_image[0][3], cmap="gray")
plt.show()

f4 = plt.figure(figsize=(12, 12))
ax21 = f4.add_subplot(221)
ax22 = f4.add_subplot(222)
ax23 = f4.add_subplot(223)
ax24 = f4.add_subplot(224)

ax21.imshow(images[0][0] - mask, cmap="gray")
ax22.imshow(images[0][1] - mask, cmap="gray")
ax23.imshow(images[0][2] - mask, cmap="gray")
ax24.imshow(images[0][3] - mask, cmap="gray")

f5 = plt.figure(figsize=(12, 12))
ax51 = f5.add_subplot(221)

ax51.imshow( mask, cmap="gray")

plt.show()





np.random.seed(1)
for k in range(2,4):
    for n in range(6,10):
        # new_image = 0 # !!
        l = 240
        img = new_image[0,0]

        classif = GaussianMixture(n_components=k)
        classif.fit(img.reshape((img.size, 1)))

        cluster = classif.predict(img.reshape((img.size, 1)))
        cluster = cluster.reshape(240, 240)
        # threshold = np.mean(classif.means_)
        # binary_img = (img < threshold).astype(int)
        plt.savefig('')
        sums = []
        for z in range(0,k):
            sums.append((cluster == z).sum())
        tumor_id = np.argmin(sums)
        tumor = (cluster == tumor_id).astype(int)


        scipy.misc.imsave('E:\\Lab\\images\\gmm\\img_tum_' + str(k) + str(n) + '.jpg', tumor)
        scipy.misc.imsave('E:\\Lab\\images\\gmm\\img_'+str(k) + str(n) + '.jpg', cluster)

