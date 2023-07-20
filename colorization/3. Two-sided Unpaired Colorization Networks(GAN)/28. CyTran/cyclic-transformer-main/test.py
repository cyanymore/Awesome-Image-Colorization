# This code is released under the CC BY-SA 4.0 license.

import glob
import os
import numpy as np
import pandas as pd
import pydicom
import torch

from skimage.metrics import structural_similarity as ssim
from models import create_model
from options.train_options import TrainOptions


@torch.no_grad()
def compute_eval_metrics_gan(root_path, tagA='venous', tagB='arterial', device='cpu'):
    # root_path - is the path to the raw Coltea-Lung-CT-100W data set.
  
    opt = TrainOptions().parse()
    opt.load_iter = 40
    opt.isTrain = False
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    gen.eval()

    # eval_dirs = pd.read_csv(os.path.join(root_path, 'test.csv'))
    # eval_dirs = list(eval_dirs.iloc[:, 1])        

    mae_pre = []
    mae_post = []
    rmse_pre = []
    rmse_post = []
    ssim_pre = []
    ssim_post = []

    for path in glob.glob(os.path.join(root_path, 'validate/*')):
        # print(path)            
        # if not path.split('/')[-1] in eval_dirs:
            # print(path.split('/')[-1])                        
            # continue  
            
        # list for the tagB file paths    
        list_scan_paths = []
        
        # loop over each path and append 
        for scan in glob.glob(os.path.join(path, tagB, '*')):
            list_scan_paths.append(scan)
            
        index = 0

        for scan in glob.glob(os.path.join(path, tagA, '*')):
            
            orig_img = pydicom.dcmread(scan).pixel_array
            native_img = pydicom.dcmread(list_scan_paths[index]).pixel_array

            # Scale native image
            # native_img[native_img < 0] = 0
            native_img = native_img / 1e3
            # native_img = native_img - 1

            # Scale original image, which is transform
            # orig_img[orig_img < 0] = 0
            orig_img = orig_img / 1e3
            # orig_img = orig_img - 1

            orig_img_in = np.expand_dims(orig_img, 0).astype(float)
            orig_img_in = torch.from_numpy(orig_img_in).float().to(device)
            orig_img_in = orig_img_in.unsqueeze(0)

            native_fake = gen(orig_img_in)[0, 0].detach().cpu().numpy()

            mae_pre.append(np.mean(np.abs(orig_img - native_img)))
            mae_post.append(np.mean(np.abs(native_fake - native_img)))

            rmse_pre.append(np.sqrt(np.mean((orig_img - native_img)**2)))
            rmse_post.append(np.sqrt(np.mean((native_fake - native_img)**2)))

            ssim_pre.append(ssim(orig_img, native_img, data_range=orig_img.max()-orig_img.min()))
            ssim_post.append(ssim(native_fake, native_img, data_range=native_img.max()-native_img.min()))
            
            # set index to next
            index += 1

    mae_pre = np.mean(mae_pre)
    mae_post = np.mean(mae_post)
    rmse_pre = np.mean(rmse_pre)
    rmse_post = np.mean(rmse_post)
    ssim_pre = np.mean(ssim_pre)
    ssim_post = np.mean(ssim_post)

    print(f"MAE before {mae_pre}, after {mae_post}")
    print(f"RMSE before {rmse_pre}, after {rmse_post}")
    print(f"SSIM before {ssim_pre}, after {ssim_post}")
  

if __name__ == '__main__':
    compute_eval_metrics_gan(
        root_path='model_data_aligned/',
        device='cuda'
    )
