import os
 
import numpy as np
import tensorflow as tf
import cv2


def TestImageReader(file_list, step, size): #训练数据读取接口
    file_length = len(file_list) #获取图片列表总长度
    line_idx = step % file_length #获取一张待读取图片的下标
    test_line_content = file_list[line_idx] #获取一张测试图片路径与名称
    test_image_name, _ = os.path.splitext(os.path.basename(test_line_content)) #获取该张测试图片名
    test_image = cv2.imread(test_line_content, 0) #读取一张测试图片
    # 灰度图变成三通道
    test_image = np.expand_dims(test_image, axis=2)
    test_image = np.concatenate((test_image, test_image, test_image), axis=-1)
    test_image_resize_t = cv2.resize(test_image, (size, size)) #改变读取的测试图片的大小
    test_image_resize = test_image_resize_t/127.5-1 #归一化测试图片
    return test_image_name, test_image_resize #返回读取并处理的一张测试图片与它的名称
