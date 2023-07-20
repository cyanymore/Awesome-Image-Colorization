# -*- coding: utf-8 -*-
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
import math




if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt2 = TestOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    data_loader_test = CreateDataLoader(opt2)
    dataset_test = data_loader_test.load_data()

    dataset_size = len(data_loader)
    dataset_size_test = len(data_loader_test)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):


        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            total_iters += opt.batchSize

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)


            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

        if epoch == opt.niter + opt.niter_decay:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))

            model.save_networks('latest')

            iter_data_time = time.time()

        if epoch <= opt.niter + opt.niter_decay:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()






