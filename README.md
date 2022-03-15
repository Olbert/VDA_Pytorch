# Unsupervised Anomaly Segmentation in Brain MR Images using Context Variational Autoencoder


This article proposes an unsupervised algorithm that
can outline a brain tumor by reconstructing model-internal representation of the brain without a tumor and comparing it to the
actual MRI scan. The main part of the algorithm is a deep learning-based variational autoencoder. It is trained with augmented
healthy brain MRI scans from BraTS 2019 dataset for which, the autoencoder learns to reconstruct the healthy images. Given
unhealthy brain MRI scan, i.e. image containing tumor, autoencoder returns a healthy version of it. By calculating residual
between two images and applying Gaussian Mixture Model we are able to outline the tumor on a scan.
 
 ![Alt text](results.png?raw=true "Title")
 
 
## Table of contents

-----

* [Dataset](#dataset)
* [Citation](#citation)

------

## Dataset

BRATS 2019 dataset was used for training and all experiments.
It can be downloaded here: [https://www.med.upenn.edu/cbica/brats2019/data.html]

Training parameters as well as the preprocessed data are availiable in the files.


## Citation

Please cite as 

```bibtex
@mastersthesis{domain_shift,
  author       = {Albert Gubaidullin}, 
  title        = {Unsupervised Anomaly Segmentation in Brain MR Images using Context Variational Autoencoder},
  school       = {University of Bonn},
  year         = 2020
}
```
