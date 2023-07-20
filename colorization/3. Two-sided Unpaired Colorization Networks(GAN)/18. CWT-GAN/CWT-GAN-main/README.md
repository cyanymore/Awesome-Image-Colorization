## CWT-GAN

> **Unsupervised Generative Adversarial Networks with Cross-model Weight Transfer Mechanism for Image-to-image Translation**<br>
>
> **Abstract** *Image-to-image translation covers a variety of application scenarios in reality, and is one of the key research directions in computer vision. However, due to the defects of GAN, current translation frameworks may encounter model collapse and low quality of generated images. To solve the above problems, this paper proposes a new model CWT-GAN, which introduces the cross-model weight transfer mechanism. The discriminator of CWT-GAN has the same encoding module structure as the generator’s. In the training process, the discriminator will transmit the weight of its encoding module to the generator in a certain proportion after each weight update. CWT-GAN can generate diverse  and higher-quality images with the aid of the weight transfer mechanism, since features learned by discriminator tend to be more expressive than those learned by generator trained via maximum likelihood. Extensive experiments demonstrate that our CWT-GAN performs better than the state-of-the-art methods in a single translation direction for several datasets.*

### Prerequisites
* Python 3.6.9 
* Pytorch 1.1.0 and torchvision (https://pytorch.org/)
* TensorboardX
* Tensorflow (for tensorboard usage)
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 16.04.

### Train
```
> python main.py --dataset cat2dog
```
* If the memory of gpu is **not sufficient**, set `--light` to True

### Restoring from the previous checkpoint
```
> python main.py --dataset cat2dog --resume True
```
* Previous checkpoint:  **dataset**_params_latest.pt
* If the memory of gpu is **not sufficient**, set `--light` to True
* Trained models(set --light to True):
Our previous checkpoint on summer2winter can be downloaded from https://pan.baidu.com/s/1YptASRfb8lLyS52iqz0y9Q  (password: mvwu)

### Test
```
> python main.py --dataset cat2dog --phase test
```
### Metric
```
> python fid_kid.py testA fakeA --mmd-var 
```
* You can use gpu, set `--gpu` to **the index of gpu**, such as `--gpu 0`
