from metrics import calculate_all_metrics
from models_quat import Generator, Discriminator, ShapeUNet
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


def train(args):
    glr = args.lr
    dlr = args.ttur
    print(glr, dlr)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set_seed(args.random_seed)
    if not args.preloaded_data:
        syn_dataset = KAISTDataset(
            path=args.dataset_path, split='train', colored_data=args.color_images, paired_image=args.loss_ssim)
        syn_loader = DataLoader(
            syn_dataset, batch_size=args.batch_size, shuffle=True)
        syneval_dataset = KAISTDataset(
            path=args.dataset_path, split='val', colored_data=args.color_images, paired_image=args.loss_ssim)
    else:
        idx = 0
        args.epoch = 10*args.epoch
        syn_dataset = KAISTDataset(to_be_loaded=True)
        syn_dataset.load_dataset(path=args.dataset_path+"/tensors/tensors_paired", split="train",
                                 idx=str(idx), img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim)
        syn_loader = DataLoader(
            syn_dataset, batch_size=args.batch_size, shuffle=True)

        syneval_dataset = KAISTDataset(to_be_loaded=True)
        syneval_dataset.load_dataset(path=args.dataset_path+"/tensors/tensors_paired", split="val", idx=str(
            idx), img_size=args.img_size, colored_data=args.color_images, paired_image=args.loss_ssim)

    in_c = 1 if not args.color_images else 3
    in_c_gen = in_c+4 if args.wavelet_type != None else in_c

    if args.qsn or args.phm:
        while (in_c_gen + args.c_dim) % 4 != 0: #3+2
            in_c_gen+=1
        while in_c % 4 !=0:
            in_c+=1
    netG = Generator(in_c=in_c_gen + args.c_dim, mid_c=args.G_conv, layers=2, s_layers=3, affine=True, last_ac=True,
                     colored_input=args.color_images, wav=args.wavelet_type,real=args.real, qsn=args.qsn, phm=args.phm)
    if args.pretrained_generator:
        netG.load_state_dict(torch.load(
            args.save_path+"/pretrained_gen_"+str(args.img_size)+".pt"))

    nets = munch.Munch({"netG": netG,
                        "netD_i": Discriminator(c_dim=args.c_dim * 2, image_size=args.img_size, colored_input=args.color_images,real=args.real, qsn=args.qsn, phm=args.phm),
                        "netD_t": Discriminator(c_dim=args.c_dim * 2, image_size=args.img_size, colored_input=args.color_images,real=args.real, qsn=args.qsn, phm=args.phm),
                        "netH":  ShapeUNet(img_ch=in_c, output_ch=1, mid=args.h_conv,real=args.real, qsn=args.qsn, phm=args.phm),
                        })
    nets.netG.to(device)
    nets.netD_i.to(device)
    nets.netD_t.to(device)
    nets.netH.to(device)

    optims = munch.Munch({"g_optimizier": torch.optim.Adam(nets.netG.parameters(), lr=glr, betas=(args.betas[0], args.betas[1])),
                          "di_optimizier": torch.optim.Adam(nets.netD_i.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1])),
                          "dt_optimizier": torch.optim.Adam(nets.netD_t.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1])),
                          "h_optimizier": torch.optim.Adam(nets.netH.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))})

    if args.sepoch > 0:
        load_nets(args, nets, args.sepoch, optims)
    nets['netG_use'] = copy.deepcopy(netG)
    nets.netG_use.to(device)

    start_time = time.time()
    print('start training...')

    ii = args.sepoch * len(syn_loader)
    ssim = tgm.losses.SSIM(3, reduction='mean')

    # logdir = "log/" + args.save_path
    # log_writer = LogWriter(logdir)
    with wandb.init(config=args, project="targan_drone") as run:
        wandb.run.name = args.experiment_name
        for epoch in tqdm(range(args.sepoch, args.epoch), initial=args.sepoch, total=args.epoch):
            for i, (x_real, shape_mask, label_org) in tqdm(enumerate(syn_loader), total=len(syn_loader)):
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
                shape_mask = shape_mask.to(device)

                # plt.subplot(232)
                # plt.imshow(t_img[2].cpu().detach().permute(1, 2, 0).numpy())

                # 2. Train the discriminator
                # Compute loss with real whole images.
                out_src, out_cls = nets.netD_i(x_real)
                # plt.subplot(233)
                # plt.imshow(  x_real[2].cpu().detach().permute(1, 2, 0).numpy(), cmap='gray')
                # plt.savefig('x_real greyscaled')
                # raise Exception
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = F.binary_cross_entropy_with_logits(
                    out_cls, d_org, reduction='sum') / out_cls.size(0)

                # Compute loss with fake whole images.
                with torch.no_grad():
                    x_fake = nets.netG(x_real, None, c_trg, mode="kaist", wav_type=args.wavelet_type)
                # plt.imshow(  x_fake[2].cpu().detach().permute(1, 2, 0).numpy(), cmap='gray')
                # plt.savefig('x-fake greyscaled')
                # raise Exception
                out_src, out_f_cls = nets.netD_i(x_fake.detach())
                d_loss_fake = torch.mean(out_src)
                d_loss_f_cls = F.binary_cross_entropy_with_logits(
                    out_f_cls, d_false_org, reduction='sum') / out_f_cls.size(0)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
                x_hat = (alpha * x_real.data + (1 - alpha)
                         * x_fake.data).requires_grad_(True)
                out_src, _ = nets.netD_i(x_hat)
                d_loss_gp = gradient_penalty(out_src, x_hat, device)

                # compute loss with target images
                
                dt_loss = torch.FloatTensor([0]).to(device)
                w_dt = 0
                d_loss_f_cls_t = torch.FloatTensor([0]).to(device)
                # Backward and optimize.
                di_loss = d_loss_real + d_loss_fake + d_loss_cls + \
                    d_loss_gp * 10 + d_loss_f_cls * args.w_d_false_c
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
                x_fake = nets.netG(
                    x_real, None, c_trg, mode="kaist", wav_type=args.wavelet_type)
                out_src, out_cls = nets.netD_i(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = F.binary_cross_entropy_with_logits(
                    out_cls, g_trg, reduction='sum') / out_cls.size(0)
                # mask = mask.repeat(1, 3, 1, 1)
                # shape_mask = shape_mask.repeat(1, 3, 1, 1)
                # print(shape_mask.shape,nets.netH(x_fake).shape )
                # shape_loss = F.mse_loss(nets.netH(x_fake), shape_mask.float())
                # Target-to-original domain.
                x_reconst = nets.netG(
                    x_fake, None, c=c_org, mode="kaist", wav_type=args.wavelet_type)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                if args.loss_ssim:
                    ssim_loss = ssim(x_fake, shape_mask.to(device))
                else:
                    ssim_loss = torch.tensor(0)
                
                gt_loss = torch.FloatTensor([0]).to(device)
                g_loss_cls_t = torch.FloatTensor([0]).to(device)

                # print(nets.netH(t_fake).shape, mask.shape)

                # mask.repeat(1, 3, 1, 1).float()
                
                # Backward and optimize.
                gi_loss = g_loss_fake + args.w_cycle * g_loss_rec + \
                    g_loss_cls * args.w_g_c  +ssim_loss*args.w_ssim # + shape_loss* args.w_shape
                g_loss = gi_loss + gt_loss

                optims.g_optimizier.zero_grad()
                optims.di_optimizier.zero_grad()
                optims.dt_optimizier.zero_grad()
                optims.h_optimizier.zero_grad()
                g_loss.backward()
                optims.g_optimizier.step()
                optims.h_optimizier.step()

                moving_average(nets.netG, nets.netG_use, beta=0.999)

                if (i + 0) % args.logs_every == 0:
                    all_losses = dict()
                    all_losses["train/D/loss_total"] = d_loss.item()
                    all_losses["train/G/loss_total"] = g_loss.item()
                    all_losses["train/D/w_di"] = w_di
                    all_losses["train/D/w_dt"] = w_dt
                    all_losses["train/D/loss_f_cls"] = d_loss_f_cls.item()
                    all_losses["train/D/loss_f_cls_t"] = d_loss_f_cls_t.item()
                    all_losses["train/G/loss_cls"] = g_loss_cls.item()
                    all_losses["train/G/loss_cls_t"] = g_loss_cls_t.item()
                    all_losses["train/G/loss_ssim"] = ssim_loss.item()
                    # all_losses["train/G/loss_shape_t"] = shape_loss_t.item()
                    # all_losses["train/G/loss_cross"] = cross_loss.item()
                    wandb.log(all_losses, step=ii, commit=True)

                ii = ii + 1
                ###################################

            if (epoch + 1) % 1 == 0 and (epoch + 1) > 0:
                # show syn images after every epoch
                x_real, x_infrared, x_rgb = plot_images(
                    nets.netG_use, syneval_dataset, device, args.c_dim, args.wavelet_type)
                # print(x.shape, y.shape, z.shape)
                # plt.subplot(231)
                # plt.imshow(  x.cpu().detach().permute(1, 2, 0).numpy()  )
                # plt.subplot(232)
                # plt.imshow(  y.cpu().detach().numpy()  )
                # plt.subplot(233)
                # plt.imshow(  z.cpu().detach().numpy()  )
                # plt.savefig('final')
                wandb.log({"orig": wandb.Image(
                    x_real, caption="orig_" + str(epoch))}, commit=False)
                wandb.log(
                    {"ir": wandb.Image(x_infrared, caption="ir_" + str(epoch))}, commit=False)
                wandb.log(
                    {"img": wandb.Image(x_rgb, caption="img_" + str(epoch))}, commit=False)
                # wandb.log({"orig_trg": wandb.Image(
                #     trg_orig, caption="orig_trg_" + str(epoch))}, commit=False)
                # wandb.log({"ir_trg": wandb.Image(trg_infra_fake,
                #           caption="ir_trg_" + str(epoch))}, commit=False)
                # wandb.log({"img_trg": wandb.Image(trg_rgb_fake,
                #           caption="img_trg_" + str(epoch))}, commit=False)

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
            if (epoch+1) % args.eval_every == 0:
                fid_stargan, fid_dict, dice_dict, s_score_dict, iou_dict, IS_ignite_dict, fid_ignite_dict, mae_dict = calculate_all_metrics(args, nets.net_G)
                wandb.log(dict(fid_stargan), step=ii + 1, commit=False)
                wandb.log(dict(fid_dict), step=ii + 1, commit=False)
                wandb.log(dict(IS_ignite_dict), step=ii + 1, commit=False)
                wandb.log(dict(fid_ignite_dict), step=ii + 1, commit=False)

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
                log = "Elapsed time [%s], Iteration [%i/%i], " % (
                    elapsed, epoch + 1, args.epoch)
                print(log)
                torch.cuda.empty_cache()
            if args.preloaded_data:
                idx = idx+1 if (idx+1) < 18 else 0
                print("loading dataset ", idx)
                syn_dataset = DroneVeichleDataset(to_be_loaded=True)
                syn_dataset.load_dataset("dataset/tensors", split="train", idx=str(
                    idx), img_size=args.img_size, colored_data=args.color_images)
                syn_loader = DataLoader(
                    syn_dataset, batch_size=args.batch_size, shuffle=True)
