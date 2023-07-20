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


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        # self.root = opt.dataroot
        self.root = ''
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)

        A = np.array(A)
        B = np.array(B)
        if self.opt.train_flip:
            is_flip = random.randint(0, 1)
            if is_flip:
                flip = random.randint(-1, 1)   #-1，0，1
                A = cv2.flip(A, flip)
                B = cv2.flip(B, flip)



        A = Image.fromarray(np.uint8(A)).convert('RGB')
        B = Image.fromarray(np.uint8(B)).convert('RGB')

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)



        w_offset = random.randint(0, max(0, self.opt.loadSize_w - self.opt.fineSize_w - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize_h - self.opt.fineSize_h - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize_h, w_offset:w_offset + self.opt.fineSize_w]
        B = B[:, h_offset:h_offset + self.opt.fineSize_h, w_offset:w_offset + self.opt.fineSize_w]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

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
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
