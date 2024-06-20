# -*- coding: utf-8 -*-

import argparse
import os

from os import listdir
from os.path import isfile, join
import random
import math
import pickle

from tqdm import tqdm

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#@title Méthodes utiles
def shift_right(x) :
  return torch.roll(x,1,2)

def shift_down(x) :
  return torch.roll(x,1,1)

def create_mask(batch_size, W) :
  """
  Causal masking is employed by setting all A_m,n = 0
  where n > m during self-attention
  """
  mask = np.tril(np.ones((batch_size,W,W)),k=0).astype("uint8")
  return torch.Tensor(mask).int()

def positionalencoding2d(d_model, height, width, batch_size):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :param batch_size: size of the batch
    :return: batch_size * height * width * d_model position matrix

    :source: https://github.com/wzlxjtu/PositionalEncoding2D
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.permute(1,2,0).repeat(batch_size,1,1,1)

def delete_gray_img(root) :
  list_img = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
  nb_image_del = 0

  for p in list_img :
    img = Image.open(p)
    if(len(np.array(img).shape) != 3):
      os.remove(p)
      nb_image_del += 1

  print(nb_image_del,"images noirs et blancs ont été supprimées")

def resize_img_dataset(root_src, root_dst, size) :
  list_img = [f for f in listdir(root_src) if isfile(join(root_src, f))]

  transform_reshape = transforms.Compose([             
      transforms.Resize(size), # interpolaition = BILINEAR
      #transforms.CenterCrop(size)
  ])

  for f in tqdm(list_img) :
    if not os.path.exists(join(root_dst, f)) and f.split(".")[1] == "jpg" :
        img = Image.open(join(root_src, f))
        img_reshape = transform_reshape(img)
        img_reshape.save(join(root_dst, f))
        
      # self.transform_x64_c = transforms.Compose([
    #     transforms.Resize(64), # interpolaition = BILINEAR
    #     transforms.CenterCrop(64),
    #     transforms.Lambda(lambda x : convertTo3bit(x,7)),
    # ])

#resize_img_dataset('/content/gdrive/My Drive/AMAL/Projet/dataset/tiny_imagenet','/content/gdrive/My Drive/AMAL/Projet/dataset/tiny_imagenet_256',256)
#resize_img_dataset('/content/gdrive/My Drive/AMAL/Projet/dataset/tiny_imagenet_256','/content/gdrive/My Drive/AMAL/Projet/dataset/tiny_imagenet_64',64)


def convertTo3bit(x,N) :
  """
  Color Quantization
  x : valeur allant de 0 à 255
  N : N+1 valeurs différentes (np.linspace(0,255,N+1))
  """
  return torch.round(torch.round(x*(N/255))*(255/N)).long()

def intTo3bit(value) :
  """
  value : int
  """
  tmp = value
  v1 = 64
  c1 = value // v1
  tmp -= c1 * v1
  v2 = 8
  c2 = tmp // v2
  tmp -= c2 * v2
  v3 = 1
  c3 = tmp
  return (c1,c2,c3)

def bitsToInt(channels):
    res = 0
    for k in range(3):
        res += torch.round(channels[k]*(7/255)) * 8**k 
    return res

def toOneChannel(x):
    n, r, c, _ = x.shape
    res = torch.zeros(n, r, c)
    for b in range(n):
        for i in range(r):
            for j in range(c): 
                res[b,i,j] = bitsToInt(x[b,i,j])
    return res

class ColTranDataset(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3 = None, max_img=8) :    
        self.root_path_1 = root_path_1
        self.root_path_2 = root_path_2
        self.root_path_3 = root_path_3
        self.images_1_path = [join(root_path_1, f) for f in listdir(root_path_1) if isfile(join(root_path_1, f))]
        self.images_2_path = [join(root_path_2, f) for f in listdir(root_path_2) if isfile(join(root_path_2, f))]
        if self.root_path_3 is not None:
            self.images_3_path = [join(root_path_3, f) for f in listdir(root_path_3) if isfile(join(root_path_3, f))]
   

    def __len__(self):
        return len(self.images_1_path)

    def __getitem__(self,index):
        images_1 = Image.open(self.images_1_path[index]) 
        images_1 = torch.Tensor(np.array(images_1)).long()
        images_2 = Image.open(self.images_2_path[index]) 
        images_2 = torch.Tensor(np.array(images_2)).long()
        if self.root_path_3 is not None:
            images_3 = Image.open(self.images_3_path[index]) 
            images_3 = torch.Tensor(np.array(images_3)).long()
            return images_1, images_2, images_3
        else: return images_1, images_2
        
class ColTranCoreDatasetToy(Dataset):

    def __init__(self, nb_images) :    
        self.toys = [torch.randint(0, 255, (4,4,3)).long() for i in range(nb_images)]

    def __len__(self):
        return len(self.toys)

    def __getitem__(self,index):

        return self.toys[index][:,:,random.randint(0, 2)], convertTo3bit(self.toys[index], 7)
    
class ColTranUpColorDatasetToy(Dataset):

    def __init__(self, nb_images) :    
        self.toys = [torch.randint(0, 255, (4,4,3)).long() for i in range(nb_images)]
        
    def __len__(self):
        return len(self.toys)

    def __getitem__(self,index):

        return self.toys[index][:,:,random.randint(0, 2)], convertTo3bit(self.toys[index], 7), self.toys[index]
    
class ColTranUpSpaceDatasetToy(Dataset):

    def __init__(self, nb_images) :    
        self.toys = [torch.randint(0, 255, (8,8,3)).long() for i in range(nb_images)]

    def __len__(self):
        return len(self.toys)

    def __getitem__(self,index):
        
        x_s = F.interpolate(self.toys[index].unsqueeze(0).permute(0,3,1,2).float(),(4,4),mode="bilinear").long()
        x_s = x_s.squeeze(0).permute(1,2,0)

        return self.toys[index][:,:,random.randint(0, 2)], x_s, self.toys[index]
    
    
class ColTranDatasetToy(Dataset):

    def __init__(self, nb_images) :    
        self.toys = [torch.randint(0, 255, (8,8,3)).long() for i in range(nb_images)]

    def __len__(self):
        return len(self.toys)

    def __getitem__(self,index):
        
        x_s = F.interpolate(self.toys[index].unsqueeze(0).permute(0,3,1,2).float(),(4,4),mode="bilinear").long()
        x_s = x_s.squeeze(0).permute(1,2,0)
        
        channel_random = random.randint(0, 2)

        return x_s[:,:,channel_random],(self.toys[index])[:,:,channel_random], convertTo3bit(x_s, 7)
    

def draw_result(lst_iter, lst_loss, title):
    plt.figure()
    plt.plot(lst_iter, lst_loss, '-b', label='train')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig("./plot/"+title+".png")  # should before show method
    plt.cla()
