import os
 
import numpy as np
import tensorflow as tf
import cv2


#读取图片的函数，接收六个参数
#输入参数分别是图片名，图片路径，标签路径，图片格式，标签格式，需要调整的尺寸大小
def TrainImageReader(file_name, x_image_path, y_image_path, picture_format = ".png", label_format = ".jpg", size = 256):
    x_image_name =x_image_path + file_name + picture_format #得到图片名称和路径
    y_image_name = y_image_path + file_name + label_format #得到标签名称和路径
    x_image = cv2.imread(x_image_name, 0) #读取图片

    x_image = np.expand_dims(x_image, axis=2)
    x_image = np.concatenate((x_image, x_image, x_image), axis=-1)
    y_image = cv2.imread(y_image_name, 1) #读取标签

    x_image_resize_t = cv2.resize(x_image, (size, size)) #调整图片的尺寸，改变成网络输入的大小
    x_image_resize = x_image_resize_t / 127.5 - 1. #归一化图片
    y_image_resize_t = cv2.resize(y_image, (size, size)) #调整标签的尺寸，改变成网络输入的大小
    y_image_resize = y_image_resize_t / 127.5 - 1. #归一化标签
    return x_image_resize, y_image_resize #返回网络输入的图片，标签，还有原图片和标签的长宽
