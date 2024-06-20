# -*- coding: utf-8 -*-
import os
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from options.test_options import TestOptions
import csv
import numpy as np
import cv2
import math
import torch
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    setup_seed(20)
    opt = TrainOptions().parse()
    opt2 = TestOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    data_loader_test = CreateDataLoader(opt2)
    dataset_test = data_loader_test.load_data()

    dataset_size = len(data_loader)
    dataset_size_test = len(data_loader_test)
    print('#training images = %d' % dataset_size)
    print('#test images = %d' % dataset_size_test)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    best_psnr = 0
    if os.path.exists('/home/image1325/user/hyw/TIC/train_img') is not True:
        os.makedirs('/home/image1325/user/hyw/TIC/train_img')
    if os.path.exists('/home/image1325/user/hyw/TIC/test_img') is not True:
        os.makedirs('/home/image1325/user/hyw/TIC/test_img')
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        print(opt.epoch_count, opt.niter + opt.niter_decay + 1)
        ssim_sum = 0
        psnr_sum = 0
        img_num = 0
        with tqdm(total=math.ceil(len(dataset)/opt.batchSize), ascii=True) as tt:
            tt.set_description('epoch: {}/{}'.format(epoch, 41))
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            for i, data in enumerate(dataset):
                img_num = img_num + 1
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                model.optimize_parameters()

                fake = model.get_img_gen(data)
                result = np.clip(fake.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                label = model.get_img_label(data)
                labels = np.clip(label.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                nir = model.get_img_nir(data)
                nir = np.clip(nir.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                ssim = model.get_ssim(labels, result)
                psnr = model.get_psnr(labels, result)
                ssim_sum = ssim_sum + ssim
                psnr_sum = psnr_sum + psnr


                tt.update(1)
                ssim_avg = ssim_sum / img_num
                psnr_avg = psnr_sum / img_num
                # if total_steps % opt.display_freq == 0:
                #     save_result = total_steps % opt.update_html_freq == 0
                #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                if total_steps%1000==0:
                    color_img_sum = np.hstack([labels,result,nir])
                    color_img_sum = cv2.cvtColor(color_img_sum, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(r"/home/image1325/user/hyw/TIC/train_img/" + 'train_' + str(epoch)+ '_' + str(i) + ".png",
                                color_img_sum)
                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data,ssim_avg,psnr_avg)
                    f = open('result1101_gll.csv', 'a',newline='')

                    csv_writer = csv.writer(f)
                    message = ""
                    for k, v in losses.items():
                            message += '%s: %.3f ' % (k, v)
                    message += '  '
                    csv_writer.writerow([epoch,message,ssim_avg,psnr_avg])
                    f.close()
                iter_data_time = time.time()
            ssim_avg = ssim_sum / img_num
            psnr_avg = psnr_sum / img_num
            f2 = open('each_epoch_1101_gll.csv', 'a', newline='')
            csv_writer = csv.writer(f2)
            csv_writer.writerow([epoch, message, ssim_avg, psnr_avg])
            f2.close()

            if epoch <= opt.niter + opt.niter_decay:
                best_psnr = psnr_avg
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                # model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            model.update_learning_rate()

        # Save the results of the current training and testing
        ssim_test_sum = 0
        psnr_test_sum = 0
        img_test_num = 0
        if epoch % 1 == 0:
            with tqdm(total=len(dataset_test), ascii=True) as tt2:
                tt2.set_description('cal Test PSNR')
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0
                for i, data in enumerate(dataset_test):
                    if i>1000 and epoch<15:
                        break
                    img_test_num = img_test_num + 1
                    model.set_input(data)
                    model.test()
                    fake = model.get_img_gen(data)

                    result = np.clip(fake.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                    label = model.get_img_label(data)

                    labels = np.clip(label.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                    nir = model.get_img_nir(data)
                    nir = np.clip(nir.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)

                    ssim_test = model.get_ssim(labels, result)
                    psnr_test = model.get_psnr(labels, result)
                    ssim_test_sum = ssim_test_sum + ssim_test
                    psnr_test_sum = psnr_test_sum + psnr_test
                    if (i < 100):
                        color_img_sum2 = np.hstack([labels,result,nir])
                        color_img_sum2 = cv2.cvtColor(color_img_sum2, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(r"/home/image1325/user/hyw/TIC/test_img/" + 'test_' + str(epoch) + '_' + str(i) + ".png",
                                    color_img_sum2)
                    tt2.update(1)
                ssim_test_avg = ssim_test_sum / img_test_num
                psnr_test_avg = psnr_test_sum / img_test_num

                if psnr_test_avg > best_psnr:
                    best_psnr = psnr_test_avg
                    print('saving the model at the end of epoch %d, iters %d' %
                          (epoch, total_steps))
                    model.save_networks('latest')


                print("TestSSIM:")
                print(ssim_test_avg)
                print("TestPSNR:")
                print(psnr_test_avg)
                f3 = open('each_epoch_1101test_gll.csv', 'a', newline='')

                csv_writer = csv.writer(f3)
                csv_writer.writerow([epoch, ssim_test_avg, psnr_test_avg])
                f3.close()
                tt2.update(1)





