import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

import cv2
import skimage
from skimage import data, exposure, img_as_float,io
import numpy as np

class TransCrop():
    def __init__(self,crop_size=256):
        self.crop_size=crop_size
    def __call__(self, image,label):
        # w=image.shape[0]
        # h=image.shape[1]
        image=np.array(image)
        w,h,c=image.shape
        h1=np.random.randint(0,h-self.crop_size)
        w1 = np.random.randint(0, w - self.crop_size)
        image=image[w1:(w1+self.crop_size),h1:(h1+self.crop_size)]
        label = label[w1:(w1 + self.crop_size), h1:(h1 + self.crop_size)]
        return image,label
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        # self.root = opt.dataroot
        self.root = '/home/admin1325/PycharmProjects/pythonProject'
        # self.dir_A=os.path.join(opt.nir_train_data_path2)
        # self.dir_B = os.path.join(opt.rgb_train_data_path2)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.A_paths,self.B_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        T=TransCrop()
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        #A=cv2.imread(A_path)
        #B=cv2.imread(B_path)
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        #A,B=T(A,B)
        # # AB = cv2.imread(AB_path)
        # w, h = A.size
        # if w!=256:
        #     A, B = T(A, B)
        #w2 = int(w / 2)
        #A = A.crop((0, 0, w2, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)
        #B = B.crop((0, 0, w2, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)

        w_offset = random.randint(0, max(0, self.opt.loadSize_w - self.opt.fineSize_w - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize_h - self.opt.fineSize_h - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize_h, w_offset:w_offset + self.opt.fineSize_w]
        B = B[:, h_offset:h_offset + self.opt.fineSize_h, w_offset:w_offset + self.opt.fineSize_w]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        B1= transforms.Resize(128)(B)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)


        #数据增强————————————————————————————————————————————————————————————————————————————————————

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
