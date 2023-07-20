from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models_mae import MaskedAutoencoderViT
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
def check_img_data_range(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 1.0
def drop_path(x, drop_prob, training=False, scale_by_keep=True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
class AverageCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
class Atten(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, dim):
        super(Atten, self).__init__()
        self.Adp=nn.AdaptiveAvgPool2d((1))
        self.conv=nn.Sequential(
            nn.Conv2d(dim,dim//4,1,1),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, 1, 1),
            nn.Sigmoid()
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        input=x
        #C*H*W
        x=x*self.conv(self.Adp(x))
        #1*H*W
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out=input+x*self.sigmoid(max_out)
        return out
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
class RepLKBlock(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size,drop_path=0.):
        super(RepLKBlock,self).__init__()
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, dw_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(dw_channels),
            nn.GELU(),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(dw_channels,in_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(in_channels),
            nn.GELU(),
        )
        self.large_kernel = nn.Conv2d(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1,padding=block_lk_size // 2 ,groups=dw_channels,bias=True)
        self.lk_nonlinear = nn.GELU()
        self.prelkb_bn = nn.InstanceNorm2d(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #print('drop path:', self.drop_path)
    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
class ConvFFN(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels,drop_path=0.):
        super(ConvFFN,self).__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.InstanceNorm2d(in_channels)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(internal_channels),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(out_channels),
        )
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
class Decoder(nn.Module):
    def __init__(self, in_planes=1, out_planes=16):
        super(Decoder, self).__init__()
        self.in_planes=in_planes
        self.out_planes = out_planes
        self.dense=nn.Sequential(
            nn.Conv2d(out_planes * 4, out_planes * 2, 3, 1, 1),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

        )
        self.dense2=nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes , 3, 1, 1),
            RepLKBlock(out_planes , out_planes , 13),
            ConvFFN(out_planes , out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
        )
        self.dense3=nn.Sequential(
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 2, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 2, out_planes),
        )
        self.end=nn.Sequential(
            nn.Conv2d(out_planes,3,3,1,1),
        )
        self.start=nn.Sequential(
            nn.Conv2d(in_planes,out_planes,3,1,1),
            nn.InstanceNorm2d(out_planes),
            nn.GELU()
        )
        self.encoder = Encoder(in_planes, out_planes)
    def forward(self,nir):

        fu_en=self.encoder(nir)

        ee=self.dense(F.interpolate(fu_en[0], scale_factor=2, mode="nearest")) + fu_en[1]
        ee1 = self.dense2(F.interpolate(ee, scale_factor=2, mode="nearest")) + fu_en[2]
        ee2 = self.dense3(ee1)
        return ee2
class Decoder2(nn.Module):
    def __init__(self, in_planes=1, out_planes=16):
        super(Decoder2, self).__init__()
        self.in_planes=in_planes
        self.out_planes = out_planes
        self.d=nn.Sequential(
            nn.Conv2d(out_planes,out_planes*2,3,1,1),
            nn.InstanceNorm2d(out_planes*2 ),
            nn.GELU(),
        )
        self.up=nn.Sequential(

            nn.Conv2d(self.out_planes * 2, self.out_planes, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes),
            nn.GELU(),
        )
        self.d2=nn.Sequential(
            nn.Conv2d(out_planes* 2,out_planes*4,3,2,1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.GELU(),
        )
        self.up2=nn.Sequential(

            nn.Conv2d(self.out_planes * 4, self.out_planes*2, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes*2),

            nn.GELU(),

        )
        self.d3=nn.Sequential(
            nn.Conv2d(out_planes* 4,out_planes*8,3,2,1),
            nn.InstanceNorm2d(out_planes * 8),
            nn.GELU(),
        )
        self.up3=nn.Sequential(
            nn.Conv2d(self.out_planes * 8, self.out_planes*4, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes*4),
            nn.GELU(),

        )
        self.d4=nn.Sequential(
            nn.Conv2d(out_planes * 8, out_planes * 16, 3, 2, 1),
            nn.InstanceNorm2d(out_planes * 16),
            nn.GELU(),
        )
        self.up4=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.out_planes * 16, self.out_planes*8, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes*8),
            nn.GELU(),
        )
        self.out=nn.Sequential(
            nn.Conv2d(out_planes,3,1,1)
        )
        self.ct1=Atten(out_planes*2)
        self.ct2 = Atten(out_planes*4)
        self.ct3 = Atten(out_planes*8 )
    def forward(self,input):
        x1=self.ct1(self.d(input))
        x2=self.ct2(self.d2(x1))
        x3=self.ct3(self.d3(x2))
        x31=self.up3(F.interpolate(x3, scale_factor=2, mode="nearest"))+x2
        x21=self.up2(F.interpolate(x31, scale_factor=2, mode="nearest"))+x1
        x11=self.up(F.interpolate(x21, scale_factor=2, mode="nearest"))
        return x11
class Encoder(nn.Module):
    def __init__(self, in_planes=1, out_planes=16):
        super(Encoder, self).__init__()
        self.in_planes=in_planes
        self.out_planes = out_planes
        self.down1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_planes),
            nn.GELU(),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.GELU(),
        )
        self.encoder_layers1 = nn.Sequential(
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes * 2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_planes * 2),
            nn.GELU(),
            nn.Conv2d(out_planes*2, out_planes * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 2),
            nn.GELU(),
        )
        self.encoder_layers2 = nn.Sequential(
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

        )
        self.down3 = nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes * 4, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.GELU(),
            nn.Conv2d(out_planes * 4, out_planes * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.GELU(),
        )
        self.encoder_layers3 = nn.Sequential(
            RepLKBlock(out_planes * 4, out_planes * 4, 13),
            ConvFFN(out_planes * 4, out_planes * 16, out_planes * 4),
            RepLKBlock(out_planes * 4, out_planes * 4, 13),
            ConvFFN(out_planes * 4, out_planes * 16, out_planes * 4),
        )
    def forward(self,input):
        e=self.down1(input)
        e=self.encoder_layers1(e)
        e11 = self.down2(e)
        e11=self.encoder_layers2(e11)
        e22 = self.down3(e11)
        e22=self.encoder_layers3(e22)
        return e22,e11,e


# Defines the Generator.
class LKAT_network(nn.Module):
    def __init__(self,in_planes=3,out_planes=64):
        super(LKAT_network, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.decoder1=Decoder(in_planes,out_planes)
        self.decoder2=Decoder2(out_planes,out_planes)
        self.Ence=MaskedAutoencoderViT(img_size=128,
        patch_size=16, in_chans=out_planes,embed_dim=512, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.out=nn.Conv2d(out_planes*2,3,1,1)
    def forward(self,nir_image):
        x2=nir_image

        mid=self.decoder1(x2)

        d2 = self.decoder2(mid)

        f_rgb_ence=self.Ence(mid)
        f=torch.cat((d2,f_rgb_ence),dim=1)
        f_rgb = self.out(f)
        return f_rgb

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
    def forward(self, input):
        return self.model(input)

from models.vgg import Vgg16
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class lkatModel(BaseModel):
    def name(self):
        return 'lkat_model'
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'L1','D_real', 'D_fake']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G3', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G3']
        # use_gan
        self.use_gan = opt.use_GAN
        self.w_vgg = opt.w_vgg
        self.w_tv = opt.w_tv
        self.w_gan = opt.w_gan
        self.use_condition = opt.use_condition
        # load/define networks
        self.netG3 = LKAT_network().cuda()
        if self.isTrain:
            if self.use_condition == 1:
                self.netD =NLayerDiscriminator(6).cuda()
            else:
                self.netD = NLayerDiscriminator(3).cuda()
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(use_lsgan=opt.no_lsgan).to(self.device)
            # load vgg network
            self.vgg = Vgg16().type(torch.cuda.FloatTensor)
            self.optimizers=[]
            # initialize optimizers
            self.optimizer_G = torch.optim.AdamW(self.netG3.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.AdamW(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        self.fake_B= self.netG3(self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.use_condition == 1:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = self.real_B
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()
    def backward_G(self):
        if self.use_condition == 1:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
        # vgg loss
        self.real_B_features = self.vgg(self.real_B)
        self.fake_B_features = self.vgg(self.fake_B)
        self.loss_vgg = self.criterionL1(self.fake_B_features[1], self.real_B_features[1]) * 1 + self.criterionL1(
            self.fake_B_features[2], self.real_B_features[2]) * 1 + self.criterionL1(self.fake_B_features[3],self.real_B_features[
             3]) * 1 + self.criterionL1(self.fake_B_features[0], self.real_B_features[0]) * 1
        # TV loss
        diff_i = torch.sum(torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(self.fake_B[:, :, 1:, :] - self.fake_B[:, :, :-1, :]))
        self.tv_loss = (diff_i + diff_j) / (256 * 256)

        self.loss_G = self.loss_G_GAN * self.w_gan + self.loss_vgg * self.w_vgg +self.tv_loss*self.w_tv+self.loss_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    def get_img_gen(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.fake_B= self.netG3(self.real_A)
        return ((self.fake_B + 1) / 2)*255

    def get_img_label(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        return ((self.real_B + 1) / 2)*255

    def get_img_nir(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        return ((self.real_A + 1) / 2)*255

    def get_psnr(self,img1, img2):
        return peak_signal_noise_ratio(img1, img2, data_range=check_img_data_range(img1))
    def get_ssim(self,img1, img2):
        return structural_similarity(img1, img2, multichannel=(len(img1.shape)==3), data_range=check_img_data_range(img1))
