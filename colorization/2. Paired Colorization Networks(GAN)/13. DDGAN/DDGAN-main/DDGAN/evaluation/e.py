import cv2
import numpy as np
import os
from PIL import Image
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.measure import shannon_entropy as shannon_entropy
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import normalized_root_mse as compare_nrmse


def PSNR(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


path1 = r'/home/sys120-1/cy/color_n/results/m/test_latest/images/'
path2 = r'/home/sys120-1/cy/color_n/results/m/test_latest/images/'

f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
list_en = []
list_mse = []
list_nrmse = []

for i in range(0, 80):
    img_a = Image.open(path1 + format(str(i), '0>5s') + '_fake_B.png')
    img_b = Image.open(path2 + format(str(i), '0>5s') + '_real_B.png')
    img_a = np.array(img_a)
    img_b = np.array(img_b)
    img_ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    img_gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    # psnr_num = compare_psnr(img_ga, img_gb, data_range=255)
    psnr_num = PSNR(img_ga, img_gb)
    ssim_num = compare_ssim(img_ga, img_gb, data_range=255)
    en_num = shannon_entropy(img_ga, base=2)
    mse_num = compare_mse(img_ga, img_gb)
    nrse_num = compare_nrmse(img_ga, img_gb, normalization='Euclidean')

    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    list_en.append(en_num)
    list_mse.append(mse_num)
    list_nrmse.append(nrse_num)


print('平均PSNR:', np.mean(list_psnr))
print('平均SSIM:', np.mean(list_ssim))
print('平均en:', np.mean(list_en))
print('平均mse:', np.mean(list_mse))
print('平均nrmse:', np.mean(list_nrmse))

# SSIM,PSNR,entropy越大越好
# MSE,NRMSE越小越好
