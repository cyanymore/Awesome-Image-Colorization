# MCL

#This code implements MCL model, described in the paper Multi-feature Contrastive Learning for Unpaired Image-to-Image Translation, Yao Gou, Min Li, Yu Song, Yujie He, LiTao Wang, CAIS, 2022.

#The code borrows heavily from the PyTorch implementation of CycleGAN and CUT:
#https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
#https://github.com/taesungp/contrastive-unpaired-translation

#Our MCL is inspired by ContraD
#https://github.com/jh-jeong/ContraD

#Train the MCL model: python train.py --dataroot ./datasets/yourdata --name yourname_MCL --IMCL_mode imcl

#Train the sinmcl model: python train.py --dataroot ./datasets/yourdata --name yourname_MCL --model sinmcl
