
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2


for k in range(2,4):
    for n in range(6,10):

        tumor = cv2.imread('E:\\Lab\\images\\gmm\\img_tum_' + str(k) + str(n) + '.jpg', 0)
        tumor = cv2.resize(tumor,(480,480))
        kernel = np.ones((3, 3), np.uint8)

        tumor = cv2.erode(tumor, kernel, iterations=2)
       # img_dilation = cv2.dilate(tumor, kernel, iterations=1)

        cv2.imwrite('E:\\Lab\\images\\gmm\\img_tum_erdil_' + str(k) + str(n) + '.jpg', tumor)



