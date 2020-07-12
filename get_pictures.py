
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

import utils

import d_vae_main


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

model = d_vae_main.dVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load("E:\\Lab\\Models\\Model_id_func.pt")
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
test_data = np.load("E:\\Lab\\resources\\dataset_unhealthy.npy", allow_pickle=True)

batch_size = 1


test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

dataiter = iter(test_dataloader)
images = dataiter.next().float()
output,_,_ = model(images)
output = F.pad(output, (8, 8, 8, 8, 0, 0, 0, 0), mode='constant', value=0)
# prep images for display
images = images.cpu().numpy()
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

ax11.imshow(output[0][0] - images[0][0], cmap="gray")
ax12.imshow(output[0][1] - images[0][1], cmap="gray")
ax13.imshow(output[0][2] - images[0][2], cmap="gray")
ax14.imshow(output[0][3] - images[0][3], cmap="gray")
plt.show()