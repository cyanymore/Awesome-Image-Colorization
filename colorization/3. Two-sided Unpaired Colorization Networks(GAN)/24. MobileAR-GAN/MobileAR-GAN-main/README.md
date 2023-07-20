
## Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org/
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/GANGREEK/MobileAR-GAN.git
cd MobileARGAN
```

### MobileARGAN train/test

- Train a model:
```bash
#!
python3 train.py --dataroot ./datasets/omsiv  --name omsiv512 --model MobleAR --no_dropout --gpu_ids 0  --display_id 0 --dataset_mode aligned
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`
- Test the model:
```bash
#
python3 test.py --dataroot ./datasets/omsiv  --name omsiv --model MobileAR --no_dropout --gpu_ids 0  --display_id 0 --dataset_mode aligned
```
The test results will be saved to a html file here: `./results/$dataset_Name$/latest_test/index.html`.



- Training with the following parameters led to the best results;
```bash
python3 train.py --dataroot ./datasets/rainy_sunny --name rainy_sunny_cyclegan --model cycle_gan --no_dropout --batchSize 3 --display_id 0 --niter 200 --niter_decay 200 --lambda_A 10.0 --lambda_B 10.0 --lambda_feat 1.0
```


## Training/test Details
- See `options/train_options.py` and `options/base_options.py` for training flags; see `options/test_options.py` and `options/base_options.py` for test flags.
- CPU/GPU (default `--gpu_ids 0`): Set `--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batchSize 32`) to benefit from multiple gpus.  
- During training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will be shown on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have visdom installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id 0`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.
- Images can be resized and cropped in different ways using `--resize_or_crop` option. The default option `'resize_and_crop'` resizes the image to be of size `(opt.loadSize, opt.loadSize)` and does a random crop of size `(opt.fineSize, opt.fineSize)`. `'crop'` skips the resizing step and only performs random cropping. `'scale_width'` resizes the image to have width `opt.fineSize` while keeping the aspect ratio. `'scale_width_and_crop'` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`.


Cite the Paper : N. K. Yadav, S. K. Singh and S. R. Dubey, "MobileAR-GAN: MobileNet-Based Efficient Attentive Recurrent Generative Adversarial Network for Infrared-to-Visual Transformations," in IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1-9, 2022, Art no. 5009909, doi: 10.1109/TIM.2022.3166202.


## Acknowledgments
Code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
