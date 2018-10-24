# Salt Detection in Seismic Images

### Goal
The purpose of this project is to detect salt in seismic images by classifying each pixel in
a 101x101 picture. The dataset contains 4,000 training images (with corresponding masks that identify
salt) and 18,000 test images. This project was completed as part of the Kaggle TGS Salt competition 
which closed in October 2018. https://www.kaggle.com/c/tgs-salt-identification-challenge. I teamed up
with @des137

### Approach
We used a CNN with U-Net architecture in our approach. A detailed description of the U-Net can be found
in the following publication by Olaf Ronneberger, Philipp Fischer and Thomas Brox:
https://pdfs.semanticscholar.org/0704/5f87709d0b7b998794e9fa912c0aba912281.pdf. A basic U-Net implementation
in Keras is provided by @yihui-he. Our best solution used filter sizes increasing from 32 to 1024 and image 
sizes decreasing from 128x128 to 4x4 during contracting. We used 4 convolutional layers per U-Net step, filter
sizes of 3x3 and 2x2, random normal weight initialization, adam optimizer with learning rate 1e-4, no dropout,
and a batch size of 64.

### Results
We achieved a mean intersection over union of correctly identified salt of 0.776 on the test set. Below are some 
examples on predicions on the validation set. Images in the first column are actual seismic images. The second and 
third columns show the corresponding labeled masks and the masks predicted by CNNs, respectively. White pixels in 
the masks represents identified salt.
<p align="center">
  <img src="https://github.com/roman807/TGS_Salt/blob/master/examples.png" width="350">
</p>
