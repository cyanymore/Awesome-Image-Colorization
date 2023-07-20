import os
import numpy as np
import math
from PIL import Image
import numpy
import numpy as np
import math
import cv2
import torch
from torch.autograd import Variable
import time
import numpy
from scipy.ndimage import gaussian_filter
from sklearn import metrics
from skimage.measure import compare_ssim
from sewar.full_ref import uqi
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import ssim
from numpy.lib.stride_tricks import as_strided as ast
start = time.clock()





# def psnr(img1, img2):
#     mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
#     if mse < 1.0e-10:
#         return 100 * 1.0
#     return 10 * math.log10(255.0 * 255.0 / mse)
# def psnr(target, ref):
# 	#将图像格式转为float64
#     target_data = np.array(target, dtype=np.float64)
#     ref_data = np.array(ref,dtype=np.float64)
#     # 直接相减，求差值
#     diff = ref_data - target_data
#     # 按第三个通道顺序把三维矩阵拉平
#     diff = diff.flatten('C')
#     # 计算MSE值
#     rmse = math.sqrt(np.mean(diff ** 2.))
#     # 精度
#     eps = np.finfo(np.float64).eps
#     if(rmse == 0):
#         rmse = eps
#     return 20*math.log10(255.0/rmse)

# def mae(img1, img2):
# 	#将图像格式转为float64
#     target_data = np.array(img1, dtype=np.float64)
#     ref_data = np.array(img2,dtype=np.float64)
#     # 直接相减，求差值
#     diff = ref_data - target_data
#     # 按第三个通道顺序把三维矩阵拉平
#     # diff = diff.flatten('C')
#     # 计算MSE值
#     mae = (np.mean(abs(diff)))
#     # 精度
#     return mae
def mse(img1, img2):
    mse = np.mean( np.square(img1-img2) )
    return mse



def mae(img1, img2):

    mae = (abs(img1 - img2))/256*256
    return mae


# def ssim(y_true, y_pred):
#     u_true = np.mean(y_true)
#     u_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     A=y_true-u_true
#     B=y_pred-u_pred
#     std_true = np.sqrt(var_true)
#     std_pred = np.sqrt(var_pred)
#     c1 = np.square(0.01 * 7)
#     c2 = np.square(0.03 * 7)
#     ssim = (2 * u_true * u_pred + c1) * ( 2*std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim / denom
# def ssim(imageA, imageB):
#     # 为确保图像能被转为灰度图
#     imageA = np.array(imageA, dtype=np.uint8)
#     imageB = np.array(imageB, dtype=np.uint8)
#
#     # 通道分离，注意顺序BGR不是RGB
#     (B1, G1, R1) = cv2.split(imageA)
#     (B2, G2, R2) = cv2.split(imageB)
#
#     # convert the images to grayscale BGR2GRAY
#     grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#     grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#
#     # 方法一
#     # (grayScore, diff) = compare_ssim(grayA, grayB, full=True)
#     # diff = (diff * 255).astype("uint8")
#     # print("gray SSIM: {}".format(grayScore))
#
#     # 方法二
#     (score0, diffB) = compare_ssim(B1, B2, full=True)
#     (score1, diffG) = compare_ssim(G1, G2, full=True)
#     (score2, diffR) = compare_ssim(R1, R2, full=True)
#     aveScore = (score0 + score1 + score2) / 3
#     #print("BGR average SSIM: {}".format(aveScore))
#
#     return   aveScore#,grayScore


path1 = 'E:\\pytorch-CycleGAN-and-pix2pix\\results\\dualgan\\test_latest\\image2\\'  #真
path2 = 'E:\\pytorch-CycleGAN-and-pix2pix\\results\\github\\test_latest\\image1\\'  #假
f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
# list_mae = []
# list_mse = []
list_uqi=[]
list_msssim=[]
for i in range(1, 144):

    img_a = Image.open(path1 + str(i) +'_'+ 'real_B'+'.png')
    # img_b = Image.open(path2 + str(i)+'_'+  'fake_B'+ '.png')
    # img_a = Image.open(path1 + str(i) +'.png')
    img_b = Image.open(path2 + str(i) +'.png')



    img_a = np.array(img_a)
    img_b = np.array(img_b)

    psnr_num = psnr(img_a, img_b)
    ssim_num = ssim(img_a, img_b)
    # mae_num = mae(img_a, img_b)
    # mse_num = mse(img_a,img_b)
    uqi_num=uqi(img_a,img_b)
    msssim_num=msssim(img_a,img_b)
    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    # list_mae.append(mae_num)
    # list_mse.append(mse_num)
    list_msssim.append(msssim_num)
    list_uqi.append(uqi_num)
print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
print("平均SSIM:", np.mean(list_ssim))# ,list_ssim)
print("平均uqi:", np.mean(list_uqi))
print("平均msssim:", np.mean(list_msssim))
# print("平均MAE:", np.mean(list_mae))  # ,list_mae)
# print('平均MSE:',np.mean(list_mse))

elapsed = (time.clock() - start)
print("Time used:", elapsed)
