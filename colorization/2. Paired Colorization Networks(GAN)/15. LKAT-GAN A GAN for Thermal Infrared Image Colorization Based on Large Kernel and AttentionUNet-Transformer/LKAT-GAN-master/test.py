import csv

import cv2
import numpy as np
from tqdm import tqdm

from data import CreateDataLoader
from models import create_model
from options.test_options import TestOptions

ssim_test_sum = 0
psnr_test_sum = 0
img_test_num = 0

opt = TestOptions().parse()
data_loader_test = CreateDataLoader(opt)
dataset_test = data_loader_test.load_data()
model = create_model(opt)
model.setup(opt)
with tqdm(total=len(dataset_test), ascii=True) as tt2:
    tt2.set_description('cal Test PSNR')
    epoch_iter = 0
    for i, data in enumerate(dataset_test):
        if i > 5000:
            break
        img_test_num = img_test_num + 1
        model.set_input(data)
        model.test()
        fake = model.get_img_gen(data)
        # result=trans(fake.cpu().data.numpy())
        result = np.clip(fake.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
        label = model.get_img_label(data)
        # labels=trans(label.cpu().data.numpy())
        labels = np.clip(label.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
        nir = model.get_img_nir(data)
        nir = np.clip(nir.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
        # labels = np.clip(label.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
        ssim_test = model.get_ssim(labels, result)
        psnr_test = model.get_psnr(labels, result)
        ssim_test_sum = ssim_test_sum + ssim_test
        psnr_test_sum = psnr_test_sum + psnr_test

        color_img_sum2 = np.hstack([labels, result, nir])
        color_img_sum2 = cv2.cvtColor(color_img_sum2, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            r"/home/image1325/user/hyw/TIC/test_img_d3up/" + 'test_' + str(opt.which_epoch) + '_' + str(i) + ".png",
            color_img_sum2)
        tt2.update(1)

    ssim_test_avg = ssim_test_sum / img_test_num
    psnr_test_avg = psnr_test_sum / img_test_num
    print("TestSSIM:")
    print(ssim_test_avg)
    print("TestPSNR:")
    print(psnr_test_avg)
    f3 = open('each_epoch_1101test_gll.csv', 'a', newline='')
    csv_writer = csv.writer(f3)
    csv_writer.writerow([opt.which_epoch, ssim_test_avg, psnr_test_avg])
    f3.close()
