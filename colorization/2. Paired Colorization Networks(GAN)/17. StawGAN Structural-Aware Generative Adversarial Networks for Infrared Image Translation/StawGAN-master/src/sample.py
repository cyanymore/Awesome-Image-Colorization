from codecs import ignore_errors
import os
import random
import shutil
from collections import OrderedDict
import munch

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from src.dataloader import DefaultDataset, DroneVeichleDataset
from src.models_quat import StyleEncoder
from src.utils import get_style, getLabel, label2onehot, save_image, save_json
from PIL import Image

from src.models import LPIPS, InceptionV3
import numpy as np
import glob
import cv2
from scipy import linalg
import subprocess
from ignite.metrics import FID, InceptionScore, PSNR

def sample(args):
    mode="test"
    print('Calculating evaluation metrics...')
    device = "cpu"
    ep = str(args.epoch) if not args.preloaded_data else str(args.epoch*10)
    in_c = 1 if not args.color_images else 3
    in_c_gen = in_c+4 if args.wavelet_type != None else in_c
    in_c_gen = 1 if args.lab else in_c_gen
    in_c_gen = in_c_gen+6 if (args.classes[0] and not args.classes[1]) else in_c_gen
    from src.models_quat import Generator
    netG = Generator(in_c=in_c_gen + args.c_dim, mid_c=args.G_conv, layers=2, s_layers=3, affine=True, last_ac=True,
                     colored_input=args.color_images, wav=args.wavelet_type,real=args.real, qsn=args.qsn, phm=args.phm, 
                     lab=args.lab, classes= args.classes, groupnorm=args.groupnorm).to(device)
    netG.load_state_dict(torch.load(args.save_path+"/"+args.experiment_name+"/netG_use_"+ep+"_"+args.experiment_name+".pkl", map_location=device))
    domains = ["valimg","valimgr"]
    domains.sort()
    num_domains = len(domains)
    # calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)
    print('Number of domains: %d' % num_domains)
    eval_dataset_imgr = DefaultDataset(args.dataset_path+"/test/valimgr")
    eval_dataset_img = DefaultDataset(args.dataset_path+"/test/valimg",remove_dark=args.remove_dark_samples)
    loaders = {
        "valimgr_loader": DataLoader(eval_dataset_imgr, batch_size=args.eval_batch_size, drop_last=args.classes[0]),
        "valimg_loader": DataLoader(eval_dataset_img, batch_size=args.eval_batch_size, drop_last=args.classes[0]),
    }
    if args.classes[0]:
        dataloader_to_get_classes = iter(DataLoader(to_get_classes_dataset, batch_size=args.eval_batch_size, drop_last=True))
        domains.reverse()
        if args.classes[1]:
            netSE = StyleEncoder(img_size=args.img_size).to(device)
            ep = str(args.epoch) if not args.preloaded_data else str(args.epoch*10)
            netSE.load_state_dict(torch.load(args.save_path+"/"+args.experiment_name+"/netSE_"+ep+"_"+args.experiment_name+".pkl", map_location=device))

    mod = {"valimgr": 0, "valimg": 1}

    # loaders = (syneval_dataset, syneval_dataset2,syneval_dataset3)
    # loaders = (DataLoader(syneval_dataset,batch_size=4), DataLoader(syneval_dataset2,batch_size=4),DataLoader(syneval_dataset3,batch_size=4))
    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]
        loader_ref = loaders[trg_domain + "_loader"]
        # path_ref = os.path.join(args.dataset_path + '/val', trg_domain)
        # loader_ref = get_eval_loader(root=path_ref,
        #                                  img_size=args.image_size,
        #                                  batch_size=args.eval_batch_size,
        #                                  imagenet_normalize=False,
        #                                  drop_last=True)
        for src_idx, src_domain in enumerate(src_domains):
            loader_src = loaders[src_domain + "_loader"]
            # path_src = os.path.join(args.dataset_path + '/val', src_domain)
            # loader_src = get_eval_loader(root=path_src,
            #                              img_size=args.image_size,
            #                              batch_size=args.eval_batch_size,
            #                              imagenet_normalize=False)
            task = '%s_to_%s' % (src_domain, trg_domain) 
            path_fake = os.path.join(args.sample_dir, task)
            path_fake_gt = os.path.join(path_fake,"ground_truth")
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)
            os.makedirs(path_fake_gt)
            print('Generating images for %s...' % task)
            for i, (x_src) in enumerate(tqdm(loader_src, total=len(loader_src))):
                N = x_src.size(0)
                x_src = x_src.to(device)

                # y_trg = torch.tensor([trg_idx] * N).to(device)
                group_of_images = []
                try:
                    x_ref = next(iter_ref).to(device)
                except:
                    iter_ref = iter(loader_ref)
                    x_ref = next(iter_ref).to(device)

                if x_ref.size(0) > N:
                    x_ref = x_ref[:N]
                
                # idx = random.choice([0,1,2]) #parte da zoomare?
                # x_src_batch = x_src.unsqueeze(0)
                idx = mod[trg_domain]
                c = getLabel(x_src, device, idx, args.c_dim)
                # x_src = x_src[:, :1, :, :]
                c_classes_trg, style_trg = None, None
                if args.classes[0]:
                    try:
                        batch = next(dataloader_to_get_classes)
                    except:
                        dataloader_to_get_classes = iter(dataloader_to_get_classes)
                        batch = next(dataloader_to_get_classes)
                    (x_real, t_img, paired_img, mask, label_org, t_imgs_classes_org, classes_org) = batch
                    rand_idx_classes = torch.randperm(classes_org.size(0))
                    classes_trg = classes_org[rand_idx_classes]
                    c_classes_trg = (label2onehot(classes_trg, 6) - torch.tensor([1,0,0,0,0,0])).to(device)
                    # c_classes_org = label2onehot(classes_org, 6).to(device)
                    t_imgs_classes_org = t_imgs_classes_org.to(device)
                    t_imgs_classes_trg = t_imgs_classes_org[rand_idx_classes].to(device)
                    if c_classes_trg.size(0) > N:
                        c_classes_trg = c_classes_trg[:N]
                if args.classes[1]:
                    yy_trg, style_trg, x_segm = get_style(munch.Munch({"netSE": netSE}), 
                                                        y_trg=c_classes_trg, x_segm= t_imgs_classes_trg, x_real=x_real if args.classes_image else None)  if args.classes[1] else (None, None, None)                
                    if style_trg.size(0) > N:
                        style_trg = style_trg[:N]
                x_fake = netG(x_src, None, c, mode='test', wav_type=args.wavelet_type,
                                class_label= c_classes_trg , style=style_trg)
                group_of_images.append(x_fake)
                
                # save generated images to calculate FID later
                for k in range(N):
                    filename = os.path.join(path_fake,'%.4i.png' % (i * args.eval_batch_size + (k + 1)))
                    filename_src = os.path.join(path_fake_gt,'%.4i.png' % (i * args.eval_batch_size + (k + 1)))
                    save_image(x_fake[k], ncol=1, filename=filename, args=args)
                    save_image(x_src[k], ncol=1, filename=filename_src, args=args)

                # lpips_value = calculate_lpips_given_images(group_of_images)
                # lpips_values.append(lpips_value)
            # print(lpips_values)
            # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
            # lpips_mean = np.array(lpips_values).mean()
            # lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean

        # delete dataloaders
        del loader_src
        if mode == 'test':
            del loader_ref
            del iter_ref