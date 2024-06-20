from metrics import calculate_all_metrics
from models_quat import ContrastNet, DiscriminatorStyle, Generator, Discriminator, SSIM_Loss, ShapeUNet, StyleEncoder, TVLoss
from dataloader import *
from torch.utils.data import DataLoader
from utils import *
import time
import matplotlib.pyplot as plt
import datetime
import wandb
import copy
import torch.nn.functional as F
from tqdm import tqdm
import munch

import torchgeometry as tgm

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(args):
    glr = args.lr
    dlr = args.ttur


    # set_seed(args.random_seed)
    if not args.preloaded_data:
        # if not args.single_mod:
            syn_dataset = DroneVeichleDataset(
                path=args.dataset_path, split='train', img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim, lab=args.lab, 
                classes=args.classes[0], single_mod=(args.single_mod, 'ir' if 'ir' in args.experiment_name else 'rgb'),debug = "debug" in args.mode, remove_dark=args.remove_dark_samples)
            syn_loader = DataLoader(
                syn_dataset, batch_size=args.batch_size, shuffle=True,)
            syneval_dataset = DroneVeichleDataset(
                path=args.dataset_path, split='val', img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim, lab=args.lab,
                classes=args.classes[0], single_mod=(args.single_mod, 'ir' if 'ir' in args.experiment_name else 'rgb'), debug = "debug" in args.mode,remove_dark=args.remove_dark_samples)
        # else:
            # syn_dataset = DroneVeichleDataset(
            #     path=args.dataset_path, split='train', img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim, lab=args.lab, classes=args.classes[0], single_mod=(args.single_mod, 'ir'),
            #     debug = "debug" in args.mode)
            # syn_loader = DataLoader(
            #     syn_dataset, batch_size=args.batch_size, shuffle=True,)
            # syneval_dataset = DroneVeichleDataset(
            #     path=args.dataset_path, split='val', img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim, lab=args.lab, classes=args.classes[0], single_mod=(args.single_mod, 'ir'),
            #     debug = "debug" in args.mode)
            
            # syn_dataset2 = DroneVeichleDataset(
            #     path=args.dataset_path, split='train', img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim, lab=args.lab, classes=args.classes[0], single_mod=(args.single_mod, 'rgb'),
            #     debug = "debug" in args.mode)
            # syn_loader2 = DataLoader(
            #     syn_dataset, batch_size=args.batch_size, shuffle=True,)
            # syneval_dataset2 = DroneVeichleDataset(
            #     path=args.dataset_path, split='val', img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim, lab=args.lab, classes=args.classes[0], single_mod=(args.single_mod, 'rgb'),
            #     debug = "debug" in args.mode)
    else:
        idx = 0
        tensors_path = args.tensors_path #"/tensors/tensors_paired"
        args.epoch = 10*args.epoch
        args.sepoch = 10*args.sepoch

        args.save_every = 10*args.save_every
        args.eval_every = 10*args.eval_every
        syn_dataset = DroneVeichleDataset(to_be_loaded=True)
        syn_dataset.load_dataset(path=args.dataset_path+tensors_path, split="train",
                                 idx=str(idx), img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim,lab=args.lab,
                                 classes=args.classes[0], single_mod=(args.single_mod, 'ir' if 'ir' in args.experiment_name else 'rgb'),debug = "debug" in args.mode, remove_dark=args.remove_dark_samples)
        syn_loader = DataLoader(
            syn_dataset, batch_size=args.batch_size, shuffle=True)

        syneval_dataset = DroneVeichleDataset(to_be_loaded=True)
        syneval_dataset.load_dataset(path=args.dataset_path+tensors_path, split="val", idx=str(
            idx), img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim,lab=args.lab)

    in_c = 1 if not args.color_images else 3
    in_c_gen = in_c+4 if args.wavelet_type != None else in_c
    in_c_gen = 1 if args.lab else in_c_gen
    in_c_gen = in_c_gen+6 if (args.classes[0] and not args.classes[1]) else in_c_gen
    if args.qsn or args.phm:
        while (in_c_gen + args.c_dim) % 4 != 0: #3+2
            in_c_gen+=1
        while in_c % 4 !=0:
            in_c+=1
    netG = Generator(in_c=in_c_gen + args.c_dim, mid_c=args.G_conv, layers=args.layers_gen, s_layers=args.s_layers_gen, r_lay=args.r_lay, affine=True, last_ac=True,
                     colored_input=args.color_images, wav=args.wavelet_type,real=args.real, qsn=args.qsn, phm=args.phm, lab=args.lab, classes= args.classes, groupnorm=args.groupnorm)
    if args.pretrained_generator:
        netG.load_state_dict(torch.load(
            args.save_path+"/pretrained_gen_"+str(args.img_size)+".pt"))

    
    nets = munch.Munch({"netG": netG,
                        "netD_i": Discriminator(c_dim=args.c_dim * 2, image_size=args.img_size, colored_input=args.color_images,real=args.real, qsn=args.qsn, phm=args.phm, classes=args.classes, spectral=args.spectral),
                        "netD_t": Discriminator(c_dim=args.c_dim * 2, image_size=args.img_size, colored_input=args.color_images,real=args.real, qsn=args.qsn, phm=args.phm, classes=args.classes, spectral=args.spectral),
                        "netH":  ShapeUNet(img_ch=in_c, output_ch=1, mid=args.h_conv,real=args.real, qsn=args.qsn, phm=args.phm),
                        "netSE": StyleEncoder(img_size=args.img_size).to(device) if args.classes[1] else None,
                        "netD_style" : DiscriminatorStyle(img_size=args.img_size).to(device) if args.classes[1] else None,
                        "netContr": ContrastNet().to(device) if args.contrast_t else None
                        })
    
    nets.netG.to(device)
    nets.netD_i.to(device)
    nets.netD_t.to(device)
    nets.netH.to(device)

    optims = munch.Munch({"g_optimizier": torch.optim.Adam(nets.netG.parameters(), lr=glr, betas=(args.betas[0], args.betas[1])),
                          "di_optimizier": torch.optim.Adam(nets.netD_i.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1])),
                          "dt_optimizier": torch.optim.Adam(nets.netD_t.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1])),
                          "h_optimizier": torch.optim.Adam(nets.netH.parameters(), lr=glr, betas=(args.betas[0], args.betas[1])),
                          "se_optimizier": torch.optim.Adam(nets.netSE.parameters(), lr=glr, betas=(args.betas[0], args.betas[1])) if args.classes[1] else None,
                          "ds_optimizier": torch.optim.Adam(nets.netD_style.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1])) if args.classes[1] else None,
                          "con_optimizier": torch.optim.Adam(nets.netContr.parameters(), lr=1e-4) if args.contrast_t else None
                          }) 
    # if args.single_mod:
    #     nets.netG2 = copy.deepcopy(nets.netG).to(device)
    #     nets.netD_i2 = copy.deepcopy(nets.netD_i).to(device)
    #     nets.netD_t2 = copy.deepcopy(nets.netD_t).to(device)
    #     optims.g_optimizier_2= copy.deepcopy(nets.g_optimizier)
    #     optims.di_optimizier_2= copy.deepcopy(nets.di_optimizier)
    #     optims.dt_optimizier_2= copy.deepcopy(nets.dt_optimizier)

    if args.sepoch > 0:
        load_nets(args, nets, args.sepoch, optims)
    nets['netG_use'] = copy.deepcopy(netG)
    nets.netG_use.to(device)

    start_time = time.time()
    print('start training...')

    ii = args.sepoch * len(syn_loader)
    # ssim = tgm.losses.SSIM(args.img_size-1, reduction='mean')
    ssim = SSIM_Loss(win_size=3, data_range=1.0, size_average=True, channel=3)
    criterionL1 = torch.nn.L1Loss()
    criterionTV = TVLoss(TVLoss_weight=1)
    # logdir = "log/" + args.save_path
    # log_writer = LogWriter(logdir)
    with wandb.init(config=args, project="targan_drone") as run:
        wandb.run.name = args.experiment_name
        for epoch in tqdm(range(args.sepoch, args.epoch), initial=args.sepoch, total=args.epoch):
            #alternatively train target part
            mode='train' if epoch%2==0 else 'no_target'
            if not args.alternate_target:
                mode = 'train'
            for i, batch in tqdm(enumerate(syn_loader), total=len(syn_loader)):
                #mode='train' if random.random() > .5 else 'no_target'
                if args.classes[0]:
                    (x_real, t_img, paired_img, mask, label_org, t_imgs_classes_org, classes_org) = batch
                    dim_classes_label = 6
                    zero_class_tensor = torch.zeros(dim_classes_label)
                    zero_class_tensor[0] = 1
                else:
                    (x_real, t_img, paired_img, mask, label_org) = batch

                # 1. Preprocess input data
                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]
                c_org = label2onehot(label_org, args.c_dim)
                c_trg = label2onehot(label_trg, args.c_dim)
                d_false_org = label2onehot(
                    label_org + args.c_dim, args.c_dim * 2)
                d_org = label2onehot(label_org, args.c_dim * 2)
                g_trg = label2onehot(label_trg, args.c_dim * 2)

                if args.classes[0]:
                    rand_idx_classes = torch.randperm(classes_org.size(0))
                    classes_trg = classes_org[rand_idx_classes]
                    c_classes_trg = (label2onehot(classes_trg, dim_classes_label) - zero_class_tensor).to(device)
                    #classes_org torch.tensor([1,0,0,0,0,0]),([1,1,0,0,0,0])]
                    #  ho una macchina al t_imgs_classes[i][0], ho una macchina in t_imgs_classes[i][0] e un truck in t_imgs_classes[i][1] 
                    c_classes_org = label2onehot(classes_org, dim_classes_label).to(device)
                    t_imgs_classes_org = t_imgs_classes_org.to(device)
                    t_imgs_classes_trg = t_imgs_classes_org[rand_idx_classes].to(device)

                    if args.classes[0] and not args.classes[1]:
                        d_false_class_org = label2onehot(classes_org + args.c_dim, dim_classes_label * 2).to(device)
                        d_class_org = label2onehot(classes_org, dim_classes_label * 2).to(device)
                        zero_class_tensor_trg = torch.zeros(dim_classes_label*2)
                        zero_class_tensor_trg[0] = 1
                        g_class_trg = (label2onehot(classes_trg, dim_classes_label * 2) - zero_class_tensor_trg).to(device)

                # plt.subplot(231)
                # plt.imshow(  [x_real[2].cpu().detach().permute(1, 2, 0).numpy()  )
                x_real = x_real.to(device)  # Input images.

                c_org = c_org.to(device)  # Original domain labels.
                c_trg = c_trg.to(device)  # Target domain labels.
                # Labels for computing classification loss.
                d_org = d_org.to(device)
                # Labels for computing classification loss.
                g_trg = g_trg.to(device)
                # Labels for computing classification loss.
                d_false_org = d_false_org.to(device)
                mask = mask.to(device)
                paired_img = paired_img.to(device)

                t_img = t_img.to(device)
                # plt.subplot(232)
                # plt.imshow(t_img[2].cpu().detach().permute(1, 2, 0).numpy())

                index = loss_filter(mask)
                # 2. Train the discriminator
                # Compute loss with real whole images.
                out_src, out_cls, out_class_cls = nets.netD_i(x_real) if not args.single_mod else nets.netD_i(paired_img)
                
                # plt.subplot(233)
                # plt.imshow(  x_real[2].cpu().detach().permute(1, 2, 0).numpy(), cmap='gray')
                # plt.savefig('x_real greyscaled')
                # raise Exception
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = F.binary_cross_entropy_with_logits(
                    out_cls, d_org, reduction='sum') / out_cls.size(0)
                
                if args.classes[0] and not args.classes[1]:
                    d_class_loss_cls = F.binary_cross_entropy_with_logits(
                        out_class_cls, d_class_org, reduction='sum') / out_class_cls.size(0)
                
                
                # Compute loss with fake whole images.
                with torch.no_grad():
                    yy_trg, style_trg, x_segm = get_style(nets,y_trg=c_classes_trg, x_segm= t_imgs_classes_trg, x_real=x_real if args.classes_image else None)  if args.classes[1] else (None, None, None)
                    x_fake, t_fake = nets.netG(x_real, t_img, c_trg, mode=mode,wav_type=args.wavelet_type, style=style_trg,  class_label=c_classes_trg if (args.classes[0] and not args.classes[1]) else None)
                # plt.imshow(  x_fake[2].cpu().detach().permute(1, 2, 0).numpy(), cmap='gray')
                # plt.savefig('x-fake greyscaled')
                # raise Exception
                out_src, out_f_cls, out_class_f_cls = nets.netD_i(x_fake.detach())
                d_loss_fake = torch.mean(out_src)
                d_loss_f_cls = F.binary_cross_entropy_with_logits(
                    out_f_cls, d_false_org, reduction='sum') / out_f_cls.size(0)
                
                if args.classes[0] and not args.classes[1]:
                    d_class_loss_f_cls = F.binary_cross_entropy_with_logits(
                        out_class_f_cls, d_false_class_org, reduction='sum') / out_class_f_cls.size(0)
                
                
                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device) if not args.single_mod else torch.rand(paired_img.size(0), 1, 1, 1).to(device)
                x_hat = (alpha * x_real.data + (1 - alpha)
                         * x_fake.data).requires_grad_(True) if not args.single_mod else (alpha * paired_img.data + (1 - alpha)
                                                                                         * x_fake.data).requires_grad_(True)
                out_src, _, _ = nets.netD_i(x_hat)
                d_loss_gp = gradient_penalty(out_src, x_hat, device)

                # compute loss with target images
                if index.shape[0] != 0 and mode =='train':
                    out_src, out_cls, out_class_cls = nets.netD_t(
                        torch.index_select(t_img, dim=0, index=index)) if not args.single_mod else nets.netD_t(torch.index_select(paired_img * mask, dim=0, index=index)) 
                    d_org = torch.index_select(d_org, dim=0, index=index)
                    d_loss_real_t = -torch.mean(out_src)
                    d_loss_cls_t = F.binary_cross_entropy_with_logits(
                        out_cls, d_org, reduction='sum') / out_cls.size(0)

                    if args.classes[0] and not args.classes[1]:
                        d_class_org = torch.index_select(d_class_org, dim=0, index=index)
                        d_class_loss_cls_t = F.binary_cross_entropy_with_logits(
                            out_class_cls, d_class_org, reduction='sum') / out_class_cls.size(0)

                    out_src, out_f_cls, out_class_f_cls = nets.netD_t(
                        torch.index_select(t_fake.detach(), dim=0, index=index))
                    d_false_org = torch.index_select(
                        d_false_org, dim=0, index=index)
                    d_loss_fake_t = torch.mean(out_src)
                    d_loss_f_cls_t = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org,
                                                                        reduction='sum') / out_f_cls.size(0)
                    if args.classes[0] and not args.classes[1]:
                        d_false_class_org = torch.index_select(
                            d_false_class_org, dim=0, index=index)
                        d_class_loss_f_cls_t = F.binary_cross_entropy_with_logits(out_class_f_cls, d_false_class_org,
                                                                        reduction='sum') / out_class_f_cls.size(0)

                    x_hat = (alpha * t_img.data + (1 - alpha)
                             * t_fake.data).requires_grad_(True) if not args.single_mod else (alpha * (paired_img * mask).data + (1 - alpha)
                                                                                                * t_fake.data).requires_grad_(True)
                    x_hat = torch.index_select(x_hat, dim=0, index=index)
                    out_src, _, _ = nets.netD_t(x_hat)
                    d_loss_gp_t = gradient_penalty(out_src, x_hat, device)
                    
                    disc_contr_classes = d_class_loss_cls_t + d_class_loss_f_cls_t if args.classes[0] and not args.classes[1] else torch.tensor(0).to(device)
                    
                    dt_loss = d_loss_real_t + d_loss_fake_t + d_loss_cls_t + \
                        d_loss_gp_t * 10 + d_loss_f_cls_t * args.w_d_false_t_c +  disc_contr_classes    
                    w_dt = (-d_loss_real_t - d_loss_fake_t).item()
                else:
                    dt_loss = torch.FloatTensor([0]).to(device)
                    w_dt = 0
                    d_loss_f_cls_t = torch.FloatTensor([0]).to(device)
                # Backward and optimize.
                disc_contr_classes = d_class_loss_cls + d_class_loss_f_cls if args.classes[0] and not args.classes[1] else torch.tensor(0).to(device)
                di_loss = d_loss_real + d_loss_fake + d_loss_cls + \
                    d_loss_gp * 10 + d_loss_f_cls * args.w_d_false_c +  disc_contr_classes

                d_loss = di_loss + dt_loss
                w_di = (-d_loss_real - d_loss_fake).item()

                optims.g_optimizier.zero_grad()
                optims.di_optimizier.zero_grad()
                optims.dt_optimizier.zero_grad()
                d_loss.backward()
                optims.di_optimizier.step()
                optims.dt_optimizier.step()

                #  3. Train the generator
                # Original-to-target domain.
                x_fake, t_fake = nets.netG(
                    x_real, t_img, c_trg, mode=mode,wav_type=args.wavelet_type, style=style_trg, class_label=c_classes_trg if (args.classes[0] and not args.classes[1]) else None)
                out_src, out_cls, out_class_cls = nets.netD_i(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = F.binary_cross_entropy_with_logits(
                    out_cls, g_trg, reduction='sum') / out_cls.size(0)
                
                if args.classes[0] and not args.classes[1]:
                    g_class_loss_cls = F.binary_cross_entropy_with_logits(
                        out_class_cls, g_class_trg, reduction='sum') / out_class_cls.size(0)

                # Target-to-original domain.
                yy_org, style_org, x_segm = get_style(nets,y_trg=c_classes_org, x_segm= t_imgs_classes_org, x_real=x_real if args.classes_image else None) if args.classes[1] else ( None, None, None)

                x_reconst, t_reconst = nets.netG(x_fake, t_fake, c_org, mode=mode, wav_type=args.wavelet_type, style=style_org, class_label=c_classes_org if (args.classes[0] and not args.classes[1]) else None)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                if args.classes[1]:
                    g_loss_adva, g_l_sty = compute_g_loss(nets, args, x_real=x_real, t_img=t_img, y_trg=c_classes_trg, c_trg=c_trg, x_segm=t_imgs_classes_trg, c_classes_org=c_classes_org)
                    g_style_loss = g_loss_adva + g_l_sty  
                    # d_style_loss = d_loss+d_loss_fake
                    # optims.ds_optimizier.zero_grad()
                    # d_style_loss.backward()
                    # optims.ds_optimizier.step()
                if args.loss_ssim and ((epoch>10 and not args.preloaded_data) or (epoch>100 and args.preloaded_data)):
                    # x_fake: (N,3,H,W) a batch of normalized images (-1 ~ 1)
                    # paired_img: (N,3,H,W)  
                    #X = (X + 1) / 2  # [-1, 1] => [0, 1]
                    #Y = (Y + 1) / 2  
                    ssim_loss = ssim((x_fake + 1) / 2, (paired_img+1).to(device) /2)
                else:
                    ssim_loss = torch.tensor(0)
                if args.contrast_t:
                    coeff = nets.netContr(x_fake)
                    l1_loss = criterionL1(nets.netContr.apply_transforms(x_fake, coeff), paired_img.to(device)) *0.5 + criterionL1(x_fake, paired_img.to(device)) *0.5
                else:
                    l1_loss = torch.tensor(0)


                if args.tv_loss:
                    loss_tv = criterionTV(x_fake)
                else:
                    loss_tv = torch.tensor(0)

                if index.shape[0] != 0 and mode=='train':
                    out_src, out_cls, out_class_cls = nets.netD_t(
                        torch.index_select(t_fake, dim=0, index=index))
                    g_trg = torch.index_select(g_trg, dim=0, index=index)
                    g_loss_fake_t = -torch.mean(out_src)
                    g_loss_cls_t = F.binary_cross_entropy_with_logits(
                        out_cls, g_trg, reduction='sum') / out_cls.size(0)
                    if args.classes[0] and not args.classes[1]:
                        g_class_trg = torch.index_select(g_class_trg, dim=0, index=index)
                        g_class_loss_cls_t = F.binary_cross_entropy_with_logits(
                            out_class_cls, g_class_trg, reduction='sum') / out_class_cls.size(0)
                    gt_loss = g_loss_fake_t + g_loss_cls_t * args.w_g_t_c +     (g_class_loss_cls_t if args.classes[0] and not args.classes[1] else torch.tensor(0).to(device))
                else:
                    gt_loss = torch.FloatTensor([0]).to(device)
                    g_loss_cls_t = torch.FloatTensor([0]).to(device)

                # print(nets.netH(t_fake).shape, mask.shape)

                # mask.repeat(1, 3, 1, 1).float()
                if mode =='train':
                    shape_loss_t = F.mse_loss(nets.netH(t_fake), mask.float())
                    g_loss_rec_t = torch.mean(torch.abs(t_img - t_reconst))
                    cross_loss = torch.mean(
                        torch.abs((denorm(x_fake) if not args.lab else x_fake) * mask - (denorm(t_fake) if not args.lab else t_fake)))
                else:
                    shape_loss_t, g_loss_rec_t, cross_loss = torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device), torch.FloatTensor([0]).to(device)
                # Backward and optimize.
                gi_loss = g_loss_fake + args.w_cycle * g_loss_rec + \
                    g_loss_cls * args.w_g_c  +ssim_loss*args.w_ssim + loss_tv*args.w_tv + l1_loss*args.weight_l1 + (g_class_loss_cls if args.classes[0] and not args.classes[1] else torch.tensor(0).to(device)) # + shape_loss* args.w_shape
                gt_loss = gt_loss + args.w_cycle * g_loss_rec_t + \
                    shape_loss_t * args.w_shape + cross_loss * args.w_g_cross
                style_comp = 0 if not args.classes[1] else g_style_loss *args.w_shape
                g_loss = gi_loss + gt_loss + style_comp

                optims.g_optimizier.zero_grad()
                optims.di_optimizier.zero_grad()
                optims.dt_optimizier.zero_grad()
                optims.h_optimizier.zero_grad()
                if args.classes[1]:
                    optims.se_optimizier.zero_grad()
                    optims.ds_optimizier.zero_grad()
                if args.contrast_t:
                    optims.con_optimizier.zero_grad()

                g_loss.backward()
                optims.g_optimizier.step()
                optims.h_optimizier.step()
                if args.classes[1]:
                    optims.se_optimizier.step()
                    optims.ds_optimizier.step()
                if args.contrast_t:
                    optims.con_optimizier.step()

                moving_average(nets.netG, nets.netG_use, beta=0.999)


                if (i + 0) % args.logs_every == 0:
                    all_losses = dict()
                    all_losses["train/D/loss_total"] = d_loss.item()
                    all_losses["train/G/loss_total"] = g_loss.item()
                    all_losses["train/G/loss_image_total"] = gi_loss.item()
                    all_losses["train/G/loss_target_total"] = gt_loss.item()

                    all_losses["train/D/di"] = di_loss
                    all_losses["train/D/dt"] = dt_loss
                    all_losses["train/D/loss_f_cls"] = d_loss_f_cls.item()
                    all_losses["train/D/loss_f_cls_t"] = d_loss_f_cls_t.item()
                    all_losses["train/G/loss_cls"] = g_loss_cls.item()
                    all_losses["train/G/loss_cls_t"] = g_loss_cls_t.item()
                    all_losses["train/G/loss_ssim"] = ssim_loss.item()
                    all_losses["train/G/loss_shape_t"] = shape_loss_t.item()
                    all_losses["train/G/loss_cross"] = cross_loss.item()
                    all_losses["train/G/loss_tv"] = loss_tv.item()
                    all_losses["train/G/l1_loss"] = l1_loss.item()
                    if args.classes[1]:
                        all_losses["train/G/loss_style"] = g_l_sty.item()
                        all_losses["train/G/loss_adva"] = g_loss_adva.item()

                    wandb.log(all_losses, step=ii, commit=True)
                ii = ii + 1
                ###################################

            if (epoch + 1) % 1 == 0 and (epoch + 1) > 0:
                # show syn images after every epoch
                # try:
                x_real, x_infrared, x_rgb, trg_orig, trg_infra_fake, trg_rgb_fake, pair, class_seg = plot_images(
                    nets, syneval_dataset, device, args.c_dim, args.wavelet_type, args.lab, args.classes, "debug" in args.mode, args.single_mod, args.preloaded_data)
                wandb.log({"orig": wandb.Image(
                    x_real, caption="orig_" + str(epoch))}, commit=False)
                wandb.log(
                    {"ir": wandb.Image(x_infrared, caption="ir_" + str(epoch))}, commit=False)
                wandb.log(
                    {"img": wandb.Image(x_rgb, caption="img_" + str(epoch))}, commit=False)
                wandb.log({"orig_trg": wandb.Image(
                    trg_orig, caption="orig_trg_" + str(epoch))}, commit=False)
                wandb.log({"ir_trg": wandb.Image(trg_infra_fake,
                        caption="ir_trg_" + str(epoch))}, commit=False)
                wandb.log({"img_trg": wandb.Image(trg_rgb_fake,
                        caption="img_trg_" + str(epoch))}, commit=False)
                wandb.log({"pair": wandb.Image(pair,
                        caption="pair_" + str(epoch))}, commit=False)
                if args.classes[1]:
                    wandb.log({"class_seg": wandb.Image(class_seg,
                            caption="class_seg_" + str(epoch))}, commit=False)
                # except Exception as e:
                #     print(e)
                # raise Exception

            if (epoch + 1) % args.save_every == 0:
                args.net_name = 'netG'
                save_state_net(nets.netG, args, epoch + 1,
                               optims.g_optimizier, args.experiment_name)
                args.net_name = 'netG_use'
                save_state_net(nets.netG_use, args, epoch +
                               1, None, args.experiment_name)
                args.net_name = 'netDi'
                save_state_net(nets.netD_i, args, epoch + 1,
                               optims.di_optimizier, args.experiment_name)
                args.net_name = 'netDt'
                save_state_net(nets.netD_t, args, epoch + 1,
                               optims.dt_optimizier, args.experiment_name)
                args.net_name = 'netH'
                save_state_net(nets.netH, args, epoch + 1,
                               optims.h_optimizier, args.experiment_name)
                if args.classes[1]:
                    args.net_name = 'netDs'
                    save_state_net(nets.netD_style, args, epoch + 1,
                                optims.ds_optimizier, args.experiment_name)
                    args.net_name = 'netSE'
                    save_state_net(nets.netSE, args, epoch + 1,
                                optims.se_optimizier, args.experiment_name)
                if args.contrast_t:
                    args.net_name = 'netContr'
                    save_state_net(nets.netContr, args, epoch + 1,
                                optims.con_optimizier, args.experiment_name)
            if (epoch+1) % args.eval_every == 0:
                psnr_ignite, fid_dict, dice_dict, s_score_dict, iou_dict, IS_ignite_dict, fid_ignite_dict, mae_dict, ssim_dict = calculate_all_metrics(args, nets.netG)
                wandb.log(dict(psnr_ignite), step=ii + 1, commit=False)
                wandb.log(dict(fid_dict), step=ii + 1, commit=False)
                wandb.log(dict(IS_ignite_dict), step=ii + 1, commit=False)
                wandb.log(dict(fid_ignite_dict), step=ii + 1, commit=False)
                wandb.log(dict(ssim_dict), step=ii + 1, commit=False)

                wandb.log(dict(dice_dict), step=ii + 1, commit=False)
                wandb.log(dict(iou_dict), step=ii + 1, commit=False)
                wandb.log(dict(mae_dict), step=ii + 1, commit=False)
                wandb.log(dict(s_score_dict), step=ii + 1, commit=True)
                # formatt = args.experiment_name +"        & {:.6f} & {:.6f} & {:.6f}  & {:.6f}  & {:.6f}     & {:.6f}  & {:.6f}           \\ ".format(
                #     (fid_stargan["FID/mg_mean"]+fid_stargan["FID/gr_mean"])/2,
                #     (fid_ignite_dict["FID-ignite/valimg_mean"]+fid_ignite_dict["FID-ignite/valimgr_mean"])/2,
                #     (IS_ignite_dict["IS/valimg_mean"]+IS_ignite_dict["IS/valimgr_mean"])/2,
                #     (dice_dict["DICE/img"]+dice_dict["DICE/imgr"])/2,
                #     (s_score_dict["S-SCORE/img"]+s_score_dict["S-SCORE/imgr"])/2,
                #     (iou_dict["IoU/img"]+iou_dict["IoU/imgr"])/2,
                #     (mae_dict["mae/img"]+mae_dict["mae/imgr"])/2,
                # )
                # wandb.log({"latex_string":formatt},step=ii + 1, commit=True)
            if (epoch + 1) % 1 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Epoch [%i/%i], " % (
                    elapsed, epoch + 1, args.epoch)
                print(log)

            if args.preloaded_data:
                idx = idx+1 if (idx+1) < 18 else 0
                print("loading dataset ", idx)
                syn_dataset = DroneVeichleDataset(to_be_loaded=True)
                syn_dataset.load_dataset(path=args.dataset_path+tensors_path, split="train", idx=str(idx), 
                    img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim,lab=args.lab)
                syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)
            
            if args.decay and epoch >args.epoch//2:
                decay_frac = (epoch - args.epoch//2) / (args.epoch//2)
                for opt in optims:
                    new_lr = args.ttur * (1 - decay_frac) if opt.startswith('d') else args.lr * (1 - decay_frac)
                    if optims[opt] is not None:
                        for param_group in optims[opt].param_groups:
                            param_group['lr'] = new_lr
                    print(opt,' updated learning rate: %f' % new_lr)
                #test
                args.w_ssim = args.w_ssim * (1 - decay_frac)
                args.tv_loss = args.tv_loss * (1 - decay_frac)
                args.w_cycle = args.w_cycle * (1 - decay_frac)




def onehot2label(onehotvec):
    """Convert label indices to one-hot vectors."""
    batch_size = onehotvec.size(0)
    out = torch.zeros(batch_size, 1)
    # print(onehotvec.shape)
    for i in range(batch_size):
        item = (onehotvec.squeeze()[i] == 1).nonzero(as_tuple=True)[0].item()
        out[i] = item
         
    return out

def compute_g_loss(nets, args, x_real, t_img, c_trg, y_trg, x_segm=None, c_classes_org=None):
    # estrapolo lo style code dalla immagine segmentata/quella intera?
    # gli passo anche la label target (quindi se é car car?)

    yy_trg, s_trg, x_segm_valid = get_style(nets=nets, x_segm=x_segm, y_trg=y_trg, x_real=x_real if args.classes_image else None)
    #genero l immagine fake passandogli la segmentate/reale di un truck con lo stylecode della car
    x_fake, t_fake = nets.netG(x_real, t_img, c_trg, style = s_trg, wav_type=args.wavelet_type)
    
    #calcolo la loss passando al dsicriminatore la label target car e la fake
    out = nets.netD_style(x_fake, yy_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    #estgrapolo stile da fake img segm passando sempre stesso target car??
    s_pred = nets.netSE(t_fake, yy_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    # s_trg2 = nets.style_encoder(x_ref2, y_trg)
    # x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    # x_fake2 = x_fake2.detach()
    # loss_ds = torch.mean(torch.abs(x_fake - x_fake2))


    #USARE QUELLA CHE GIA CI STA
    # cycle-consistency loss
    # s_org = netSE(x_real, y_org)
    # x_rec = netG(x_fake, style=s_org, )
    # loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    #loss = loss_adv +loss_sty  #c*args.lambda_sty  #+ args.lambda_cyc * loss_cyc

    #d_loss_real, d_losses_fake = compute_d_loss(nets, args, x_real, c_classes_org, yy_trg, x_segm, c_trg, t_img)
    return loss_adv,  loss_sty#, d_loss_real, d_losses_fake
                    #loss_cyc.item()]

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# def compute_d_loss(nets, args, x_real, c_classes_org, yy_trg, x_segm, c_trg, t_img):
#     # with real images
#     x_real.requires_grad_()
#     yy_org, s_trg, x_segm_valid = get_style(nets=nets, x_segm=x_segm, y_trg=c_classes_org)
#     out = nets.netD_style(x_real, yy_org)
#     loss_real = adv_loss(out, 1)
#     # loss_reg = r1_reg(out, x_real)

#     # with fake images
#     with torch.no_grad():
#         s_trg = nets.netSE(x_segm_valid, yy_trg)
#         x_fake, t_fake =  nets.netG(x_real, t_img, c_trg, style = s_trg, wav_type=args.wavelet_type)
#     out = nets.netD_style(x_fake, yy_trg)
#     loss_fake = adv_loss(out, 0)

#     loss = loss_real + loss_fake #+  args.lambda_reg * loss_reg
#     return loss_real, \
#                        loss_fake,\
#                     #    loss_reg.item()

