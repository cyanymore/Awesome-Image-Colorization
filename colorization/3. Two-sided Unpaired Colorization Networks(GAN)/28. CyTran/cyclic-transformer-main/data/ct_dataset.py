# This code is released under the CC BY-SA 4.0 license.

import pickle
import random

import numpy as np
import pydicom

from data.base_dataset import BaseDataset


class CTDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.raw_data = pickle.load(open(opt.dataroot, "rb"))
        self.labels = None
        self.Aclass = opt.Aclass
        self.Bclass = opt.Bclass
        self._make_dataset()

    def _make_dataset(self):
        dataA = []
        dataB = []

        for entity in self.raw_data:
            if entity == 'A':
                dataA += self.raw_data[entity]
            if entity == 'B':
                dataB += self.raw_data[entity]
            
            # print(entity)

        self.raw_dataA = dataA
        self.raw_dataB = dataB
        self.A_size = len(self.raw_dataA)
        self.B_size = len(self.raw_dataB)
        # print(self.raw_dataA[0])
        # print(self.raw_dataB[0])
        # raise


    def __getitem__(self, index):
        # Image from A
        # A_path = self.raw_dataA[index]
        # print(A_path)
        # A_image = pydicom.dcmread(A_path, force=True).pixel_array
        # A_image[A_image < 0] = 0
        # A_image = A_image / 1e3
        # A_image = A_image - 1
        # A_image = np.expand_dims(A_image, 0).astype(float)

        # Paired image from B
        # path = self.raw_data[index].replace(self.Aclass, self.Bclass)
        # B_index = random.randint(0, self.B_size - 1)
        # B_path = self.raw_dataB[index]
        # print(B_path)
        # B_image = pydicom.dcmread(B_path, force=True).pixel_array
        # B_image[B_image < 0] = 0
        # B_image = B_image / 1e3
        # B_image = B_image - 1
        # B_image = np.expand_dims(B_image, 0).astype(np.float)

        # return {'A': A_image, 'B': B_image, 'A_paths': A_path, 'B_paths': B_path}
        
        # Image from A
        A_path = self.raw_dataA[index]
        A_image = pydicom.dcmread(A_path, force=True).pixel_array
        # A_image[A_image < 0] = 0
        A_image = A_image / 1e3
        # A_image = A_image - 1
        A_image = np.expand_dims(A_image, 0).astype(np.float)

        # Paired image from B
        # path = self.raw_data[index].replace(self.Aclass, self.Bclass)
        if index >= self.B_size:
            index = random.randint(0, self.B_size - 1)
        
        B_path = self.raw_dataB[index]
        B_image = pydicom.dcmread(B_path, force=True).pixel_array
        # B_image[B_image < 0] = 0
        B_image = B_image / 1e3
        # B_image = B_image - 1
        B_image = np.expand_dims(B_image, 0).astype(np.float)
        
        # print(index)
        # print(A_path)
        # print(B_path)
        # raise
        
        return {'A': A_image, 'B': B_image}

    def __len__(self):
        return max(self.A_size, self.B_size)
