# LKATGAN
- PyTorch implementation.

# Prerequisites
- python 3.7
- torch 1.13.1
- torchvision 0.14.1

# Network
![image](https://github.com/lifegoeso/t/blob/master/img/network.jpg)

## LK-Unet
![image](https://github.com/lifegoeso/t/blob/master/img/LK-Unet.jpg)

# Data Preparation
[KAIST-MS](https://github.com/SoonminHwang/rgbt-ped-detection/blob/master/data/README.md) and [IRVI](https://pan.baidu.com/s/1og7bcuVDModuBJhEQXWPxg?pwd=IRVI) dataset. 
The resolution of all images in the input network is 256*256.



# Colorization results
## KAIST-MS dataset
![image](https://github.com/lifegoeso/t/blob/master/img/Experiments.jpg)
(a) Thermal infrared iamges. (b) CycleGAN. (c) Pix2pix. (d) SCGAN. (e) TICCGAN. (f) PealGAN. (g) MUGAN. (h) LKAT-GAN. (i)RGB images.
## IRVI dataset
![image](https://github.com/lifegoeso/t/blob/master/img/Experiments2.jpg)
(a) Thermal infrared iamges. (b) I2VGAN. (c) TICCGAN. (d) SCGAN. (e) PealGAN. (f) MUGAN. (g) LKAT-GAN. (h) RGB images.
# Comparison methods
[CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
[SCGAN](https://github.com/zhaoyuzhi/Semantic-Colorization-GAN).
[TICCGAN](https://github.com/Kuangxd/TICCGAN).
[I2VGAN](https://github.com/BIT-DA/I2V-GAN).
[PealGAN](https://github.com/FuyaLuo/PearlGAN).
[MUGAN](https://github.com/HangyingLiao/MUGAN).

# Acknowledgments
This code borrows heavily from [TICCGAN](https://github.com/Kuangxd/TICCGAN).
