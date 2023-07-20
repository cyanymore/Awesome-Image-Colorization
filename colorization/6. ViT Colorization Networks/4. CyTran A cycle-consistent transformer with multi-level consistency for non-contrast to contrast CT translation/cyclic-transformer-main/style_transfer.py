import glob
import os
import numpy as np
import pandas as pd
import pydicom
import torch
import nibabel as nib

from skimage.metrics import structural_similarity as ssim
from models import create_model
from options.train_options import TrainOptions


def style_transfer(root_path, output_path, tagA='venous', device='cpu'):
    # root_path is path to directories to be transferred, save_path is where to save
    
    
    # get the right model
    opt = TrainOptions().parse()
    opt.load_iter = 22
    opt.isTrain = False
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    gen.eval()
    
    
    # loop over each scan and make the transferred scan
    for scan in sorted(glob.glob(os.path.join(root_path, tagA, '*'))):
        orig_img = pydicom.dcmread(scan).pixel_array
        
        # Scale original image, and transform
        # orig_img[orig_img < 0] = 0
        orig_img = orig_img / 1e3
        # orig_img = orig_img - 1

        orig_img_in = np.expand_dims(orig_img, 0).astype(float)
        orig_img_in = torch.from_numpy(orig_img_in).float().to(device)
        orig_img_in = orig_img_in.unsqueeze(0)

        native_fake = gen(orig_img_in)[0, 0].detach().cpu().numpy()


        # Scale the output image and save
        fake_arr = native_fake.copy()
        # fake_arr[fake_arr < 0] = 0
        fake_arr = fake_arr * 1e3
        # fake_arr = fake_arr + 1
        native_fake = pydicom.dcmread(scan)
        native_fake.PixelData = fake_arr.astype(np.uint16).tobytes()
        native_fake.save_as(os.path.join(output_path, os.path.basename(scan)))
        # nifti_data = nib.Nifti1Image(fake_arr, np.eye(4))
        # nib.save(nifti_data, os.path.join(output_path, os.path.splitext(os.path.basename(scan))[0] + '.nii.gz'))
        
        # fake_arr = native_fake.copy()
        # fake_arr[fake_arr < 0] = 0
        # fake_arr = fake_arr / 1e3
        # fake_arr = fake_arr + 1
        # fake_arr = fake_arr.astype(np.float32)
        # fake_arr = np.clip(fake_arr, -1, 1)  # clip the values to -1 and 1
        # fake_arr = (fake_arr + 1) / 2  # scale the values to 0 and 1
        # fake_arr = fake_arr * 4095  # scale the values to 12-bit range
        # fake_arr = fake_arr.astype(np.uint16)
        
        # fake_arr = native_fake.copy()
        # fake_arr[fake_arr < 0] = 0
        # fake_arr = fake_arr * (np.max(orig_img) / np.max(fake_arr))
        # fake_arr = fake_arr + 1        
                        
        # native_fake = pydicom.dcmread(scan)
        # native_fake.PixelData = fake_arr.tobytes()
        # native_fake.save_as(os.path.join(output_path, os.path.basename(scan)))
        
        
        
if __name__ == '__main__':
    style_transfer(
        'model_data/test/EMPTY_40/',
        'generated_data/EMPTY_40_epoch_22_preprocess'
    )
    
    
