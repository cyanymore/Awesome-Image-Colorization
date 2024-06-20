from skimage import io, transform, color,data_dir
import numpy as np


def convert_gray(f):
    rgb = io.imread(f)  # 依次读取rgb图片
    gray = color.rgb2gray(rgb)  # 将rgb图片转换成灰度图
    return gray
datapath='D:/cccc/cartoon1/test_label1/'  #图片所在的路径
str=datapath+'/*.jpg'   #识别.jpg的图像
coll = io.ImageCollection(str, load_func=convert_gray)
for i in range(len(coll)):
    io.imsave('D:/cccc/cartoon1/test_picture1/' + np.str(i) + '.png', coll[i])  # 循环保存图片resized_c2g_image