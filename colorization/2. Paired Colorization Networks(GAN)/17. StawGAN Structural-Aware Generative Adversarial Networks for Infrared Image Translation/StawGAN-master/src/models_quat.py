import random
import numpy as np
import torch
from torch import nn

from HLayers.QSN2 import Qspectral_norm
from HLayers.quaternion_layers import QuaternionConv, QuaternionTransposeConv, QuaternionInstanceNorm2d
from HLayers.PHC import PHMConv, PHMTransposeConv

import torch.nn.functional as F
from wavelet import wavelet_wrapper
from torch.nn.utils.parametrizations import spectral_norm
device = "cuda" if torch.cuda.is_available() else "cpu"
import torchvision.transforms.functional as TF

# def moving_average_update(statistic, curr_value, momentum):
#     term_1 = (1 - momentum) * statistic
#     term_2 = momentum * curr_value
#     new_value = term_1 + term_2
#     return  new_value.data

'''
BLOCKS
'''

bias = False


class conv_block(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True), downsample=False, upsample=False,
                 share_net_real=False, phm=False, phm_n=4, qsn=False, real=True, groupnorm=False):
        super(conv_block, self).__init__()
        if phm and not share_net_real:
            self.conv = nn.Sequential(
                PHMConv(phm_n, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
                PHMConv(phm_n, ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
            )
        elif qsn and not share_net_real:
            self.conv = nn.Sequential(
                QuaternionConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
                QuaternionConv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv
            )
        elif real or share_net_real:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.GroupNorm(num_channels = ch_out, num_groups=32, affine=affine) if upsample and groupnorm  else nn.InstanceNorm2d(ch_out, affine=affine) ,
                actv,
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.GroupNorm(num_channels = ch_out, num_groups=32, affine=affine) if upsample and groupnorm  else nn.InstanceNorm2d(ch_out, affine=affine) ,
                actv
            )
        self.downsample = downsample
        self.upsample = upsample
        if self.upsample:
            self.up = up_conv(ch_out, ch_out // 2, affine,phm=phm, qsn=qsn, real=real, groupnorm=groupnorm)

    def forward(self, x):
        x1 = self.conv(x)
        c = x1.shape[1]
        if self.downsample:
            x2 = F.avg_pool2d(x1, 2)
            # half of channels for skip
            return x1[:, :c // 2, :, :], x2
        # x1[:,:,:,:]
        if self.upsample:
            x2 = self.up(x1)
            return x2
        return x1


class up_conv(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True), phm=False, phm_n=4, qsn=False, real=True, groupnorm=False):
        super(up_conv, self).__init__()

        if phm:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                PHMConv(phm_n, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
            )
        elif qsn:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                QuaternionConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
            )
        elif real:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                #nn.InstanceNorm2d(ch_out, affine=affine),
                nn.GroupNorm(num_channels = ch_out, num_groups=32, affine=affine) if groupnorm  else nn.InstanceNorm2d(ch_out, affine=affine) ,
                actv
            )

    def forward(self, x):
        x = self.up(x)
        return x


'''
ENCODER/DECODER
'''


class Encoder(nn.Module):
    # the Encoder_x or Encoder_r of G
    def __init__(self, in_c, mid_c, layers, affine, phm=False, qsn=False, real=True):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(layers):
            encoder.append(conv_block(in_c, mid_c, affine, downsample=True, upsample=False, phm=phm, qsn=qsn, real=real))
            in_c = mid_c
            mid_c = mid_c * 2
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        res = []
        for layer in self.encoder:
            x1, x2 = layer(x)
            res.append([x1, x2])
            x = x2
        return res


class Decoder(nn.Module):
    # the Decoder_x or Decoder_r of G
    def __init__(self, in_c, mid_c, layers, affine, r, phm=False, qsn=False, real=True,groupnorm=False):
        super(Decoder, self).__init__()
        decoder = []
        for i in range(layers - 1):
            decoder.append(conv_block(in_c - r, mid_c, affine, downsample=False, upsample=True, phm=phm, qsn=qsn, real=real, groupnorm=groupnorm))
            in_c = mid_c
            mid_c = mid_c // 2
            r = r // 2
        decoder.append(conv_block(in_c - r, mid_c, affine, downsample=False, upsample=False, phm=phm, qsn=qsn, real=real,groupnorm=groupnorm))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, share_input, encoder_input):
        encoder_input.reverse()
        x = 0
        for i, layer in enumerate(self.decoder):
            x = torch.cat([share_input, encoder_input[i][0]], dim=1)
            # print(x.shape,share_input.shape, encoder_input[i][0].shape)
            x = layer(x)
            share_input = x
        return x


'''
SHARE NET
'''


class ShareNet(nn.Module):
    # the Share Block of G
    def __init__(self, in_c, out_c, layers, affine, r, share_net_real=True, phm=False, qsn=False, real=True, groupnorm=False):
        super(ShareNet, self).__init__()
        encoder = []
        decoder = []
        for i in range(layers - 1):
            encoder.append(conv_block(in_c, in_c * 2, affine, downsample=True, upsample=False,
                                      share_net_real=share_net_real, phm=phm, qsn=qsn, real=real))
            decoder.append(conv_block(out_c - r, out_c // 2, affine, downsample=False, upsample=True,
                                      share_net_real=share_net_real, phm=phm, qsn=qsn, real=real, groupnorm=groupnorm))
            in_c = in_c * 2
            out_c = out_c // 2
            r = r // 2
        self.bottom = conv_block(in_c, in_c * 2, affine, upsample=True, phm=phm, qsn=qsn, real=real, groupnorm=groupnorm)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.layers = layers

    def forward(self, x):
        encoder_output = []
        x = x[-1][1]
        for layer in self.encoder:
            x1, x2 = layer(x)
            encoder_output.append([x1, x2])
            x = x2
        bottom_output = self.bottom(x)
        if self.layers == 1:
            return bottom_output
        encoder_output.reverse()
        for i, layer in enumerate(self.decoder):
            #12,256,32,32
            #12, 128,128, 64
            x = torch.cat([bottom_output, encoder_output[i][0]], dim=1)
            x = layer(x)
            bottom_output = x
        return x


'''
DISCRIMINATOR
'''


class Discriminator(nn.Module):
    # the D_x or D_r of TarGAN ( backbone of PatchGAN )

    def __init__(self, image_size=256, conv_dim=64, c_dim=5, repeat_num=6, colored_input=True, classes = (False,False),
                 real=True, qsn=False, phm=False, phm_n=4, spectral=False, last_layer_gen_real=True):
        super(Discriminator, self).__init__()
        layers = []
        if phm:
            layers.append(PHMConv(phm_n, phm_n, conv_dim, kernel_size=4, stride=2, padding=1))
        elif qsn:
            if spectral:
                layers.append(Qspectral_norm(QuaternionConv(4, conv_dim, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(QuaternionConv(4, conv_dim, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        elif real:
            if spectral:
                layers.append(spectral_norm(nn.Conv2d(1 if not colored_input else 3, conv_dim, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(1 if not colored_input else 3, conv_dim, kernel_size=4, stride=2, padding=1))

        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if phm:
                layers.append(PHMConv(phm_n, curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            elif qsn:
                if spectral:
                    layers.append(
                        Qspectral_norm(QuaternionConv(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
                else:
                    layers.append(QuaternionConv(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            elif real:
                if spectral:
                    layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
                else:
                    layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))

            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)

        if phm:
            self.conv1 = PHMConv(phm_n, curr_dim, 4, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = PHMConv(phm_n, curr_dim, c_dim, kernel_size=kernel_size, bias=False)
            # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
            # print(curr_dim,c_dim,kernel_size)

        elif qsn:
            # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
            # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
            if spectral:
                self.conv1 = Qspectral_norm(
                    QuaternionConv(in_channels=curr_dim, out_channels=4, kernel_size=3, stride=1, padding=1,
                                   bias=False))
                self.conv2 = Qspectral_norm(
                    QuaternionConv(in_channels=curr_dim, out_channels=c_dim, kernel_size=kernel_size, stride=1,
                                   bias=False))
            else:
                self.conv1 = QuaternionConv(in_channels=curr_dim, out_channels=4, kernel_size=3, stride=1, padding=1,
                                            bias=False)
                self.conv2 = QuaternionConv(in_channels=curr_dim, out_channels=c_dim, kernel_size=kernel_size, stride=1,
                                            bias=False)
        elif real:
            if spectral:
                self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
                self.conv2 = spectral_norm(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))
            else:
                self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
                self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
                if classes[0] and not classes[1]:
                    self.conv3 = nn.Conv2d(curr_dim, 6*2, kernel_size=kernel_size, bias=False)

        if last_layer_gen_real:
            if spectral:
                self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
                self.conv2 = spectral_norm(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))

        self.real = real
        self.last_layer_gen_real = last_layer_gen_real
        self.qsn = qsn
        self.phm = phm
        self.in_c = 1 if not colored_input else 3 if real else 4
        self.classes= classes
    def forward(self, x_real):
        # print("discriminatore in entrata", x.shape) #torch.Size([4, 1, 128, 128])
        # rgb + aalpha
        # FOR WAVELET COMMENTED
        # if (not args.real and args.soup) and not args.last_layer_gen_real:
        #     x = x.repeat(1, 3, 1, 1)
        #     x = torch.cat([x, grayscale(x)], 1)
        # print("discriminatore dopo main",h.shape) #torch.Size([4, 2048, 2, 2])
        if self.qsn or self.phm:
            x_real = torch.cat((x_real, 
                                torch.zeros(x_real.shape[0], self.in_c-x_real.shape[1] ,x_real.shape[2],x_real.shape[3]).to(device)),
                                 dim=1).to(device)
        h = self.main(x_real)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        if self.classes[0] and not self.classes[1]:
            out_class_seg = self.conv3(h)
        # print("discriminatore out src",out_src.shape) #torch.Size([4, 1, 2, 2])
        # print("discriminatore out cls",out_cls.shape, "view",out_cls.view(out_cls.size(0), out_cls.size(1)).shape) #torch.Size([4, 6, 1, 1]) view torch.Size([4, 6])
        return out_src, \
               out_cls.view(out_cls.size(0), out_cls.size(1)), \
               out_class_seg.view(out_class_seg.size(0),out_class_seg.size(1)) if self.classes[0] and not self.classes[1] else out_cls



'''
Generator
'''
@torch.no_grad()
def create_wavelet_from_input_tensor(inputs, mods, wav_type ):
    modalities = ["ir" if mods[i][0].any()==1 else "rgb" if mods[i][1].any()==1 else "" for i in range(mods.size(0))]
    # .permute(1, 2,0)
    lst = [
        torch.from_numpy(wavelet_wrapper(wav_type, chunk.squeeze().cpu().detach(), chunk.size(2), modalities[i])).type(torch.FloatTensor)
        for i,chunk in enumerate(torch.split(inputs.detach(), 1, dim=0))
        ]
    return torch.stack(lst, dim=0).to(device)



class Generator(nn.Module):
    # the G of TarGAN

    def __init__(self, in_c, mid_c, layers, s_layers, affine, r_lay=256,last_ac=True, colored_input=True, wav=False, real=True, qsn=False, phm=False, phm_n=4, 
                        classes=(False, False),spectral=False, last_layer_gen_real=True, lab=False, groupnorm=False):
        super(Generator, self).__init__()
        self.img_encoder = Encoder(in_c, mid_c, layers, affine, phm=phm, qsn=qsn, real=real)
        self.in_c_targ = in_c-4 if wav is not None else in_c
        self.target_encoder = Encoder(self.in_c_targ, mid_c, layers, affine, phm=phm, qsn=qsn, real=real)
        if not classes[1]:
            self.target_decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, affine,64, groupnorm) 
            self.img_decoder = Decoder(in_c = mid_c * (2 ** layers),mid_c = mid_c * (2 ** (layers - 1)), layers = layers, affine = affine,r=64,groupnorm= groupnorm)
        else:
            self.target_style_decoder = DecoderStyle(in_c=mid_c * (2 ** layers), style_dim=64, img_size=256) 
            self.img_style_decoder = DecoderStyle(in_c=mid_c * (2 ** layers), style_dim=64, img_size=256) 
            

        self.share_net = ShareNet(in_c = mid_c * (2 ** (layers - 1)),
                                 out_c = mid_c * (2 ** (layers - 1 + s_layers)), 
                                layers = s_layers, affine = affine, r = r_lay, groupnorm=groupnorm)
        if phm and not last_layer_gen_real:
            self.out_img = PHMConv(phm_n, mid_c, 4, 1, bias=bias)
            self.out_tumor = PHMConv(phm_n, mid_c, 4, 1, bias=bias)
        elif qsn and not last_layer_gen_real:
            self.out_img = QuaternionConv(mid_c, 4, 1, stride=1, bias=bias)
            self.out_tumor = QuaternionConv(mid_c, 4, 1, stride=1, bias=bias)
        elif real or last_layer_gen_real:
            if classes[1]:
                self.to_rgb = nn.Sequential(
                    nn.InstanceNorm2d(mid_c, affine=True),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(mid_c, (1 if not colored_input else 2 if lab else 3), 1, 1, 0)
                    )
                self.to_tumor = nn.Sequential(
                    nn.InstanceNorm2d(mid_c, affine=True),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(mid_c, (1 if not colored_input else 2 if lab else 3), 1, 1, 0)
                    )
            else:
                self.out_img = nn.Conv2d(mid_c, 1 if not colored_input else 2 if lab else 3, 1, bias=bias)
                self.out_tumor = nn.Conv2d(mid_c, 1 if not colored_input else 2 if lab else 3, 1, bias=bias)
        self.last_ac = last_ac
        self.real = real
        self.qsn = qsn
        self.in_c = in_c
        self.phm = phm
        self.lab = lab
        self.classes = classes
        self.sep = False
    # G(image,target_image,target_modality) --> (out_image,output_target_area_image)

    def forward(self, img, tumor=None, c=None, mode="train", wav_type=None, style=None,class_label=None):
        # print("input img shape",img.shape, c.shape) torch.Size([4, 1, 128, 128]) torch.Size([4, 3])
        if self.sep:
            rgb=[i for i,mod in enumerate(c) if torch.all(mod == torch.tensor([0.,1.]).to(device)).item()]
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, img.size(2), img.size(3))
        if self.classes[0] and not self.classes[1]:
            class_label = class_label.view(class_label.size(0), class_label.size(1), 1, 1)
            class_label = class_label.repeat(1, 1, img.size(2), img.size(3))
        img_orig = img
        tumor_orig = tumor
        if wav_type != None:
            img = torch.cat([img, create_wavelet_from_input_tensor(img, c, wav_type)], dim=1)
        if self.lab:
            img = torch.cat([img[:,:1,:,:], c],dim=1)
        elif self.classes[0] and not self.classes[1]:
            img = torch.cat([img, c, class_label], dim=1)
        elif self.sep:
            img = torch.cat([img[[rgb]], c[[rgb]]], dim=1)
        else:
            img = torch.cat([img, c], dim=1)
        if self.qsn or self.phm:
            img = torch.cat((img, torch.zeros(img.shape[0], self.in_c-img.shape[1] ,img.shape[2],img.shape[3]).to(device)), dim=1).to(device)
        

        x_1 = self.img_encoder(img)
        s_1 = self.share_net(x_1)
        if self.classes[1]:
            dec_out = self.img_style_decoder(s_1,x_1, style)
        else:
            dec_out = self.img_decoder(s_1,x_1)

        res_img = self.out_img(dec_out) if not self.classes[1] else self.to_rgb(dec_out)
        if self.lab:
            res_img = torch.cat([img_orig[:,:1,:,:], res_img], dim=1).to(device)
        #print(res_img.shape)#torch.Size([4, 1, 128, 128])
        if self.last_ac:
            res_img = torch.tanh(res_img)
        if mode == "train" and not self.sep:          
            if self.lab:
                tumor = torch.cat([tumor[:,:1,:,:],c],dim=1)
            elif self.classes[0] and not self.classes[1]:
                tumor = torch.cat([tumor, c, class_label], dim=1)
            else:
                tumor = torch.cat([tumor, c], dim=1)
            if self.qsn or self.phm:
                tumor = torch.cat((tumor, torch.zeros(tumor.shape[0], self.in_c_targ-tumor.shape[1] ,tumor.shape[2],tumor.shape[3]).to(device)), dim=1).to(device)
            
            x_2 = self.target_encoder(tumor)
            s_2 = self.share_net(x_2)
            if self.classes[1]:
                dec_out = self.target_style_decoder(s_2, x_2, style)
            else:
                dec_out = self.target_decoder(s_2,x_2)
            res_tumor = self.out_tumor(dec_out) if not self.classes[1] else self.to_tumor(dec_out)
            if self.lab:
                res_tumor = torch.cat([tumor_orig[:,:1,:,:], res_tumor], dim=1).to(device)

            if self.last_ac:
                res_tumor = torch.tanh(res_tumor)


            return res_img, res_tumor
        
        if self.sep:
            ir = [i for i,mod in enumerate(c) if torch.all(mod == torch.tensor([1.,0.]).to(device)).item()]
            if len(ir) !=0:    
                tumor = torch.cat([tumor[[ir]], c[[ir]]], dim=1)
                x_2 = self.target_encoder(tumor)
                s_2 = self.share_net(x_2)
                if self.classes[1]:
                    dec_out = self.target_style_decoder(s_2, x_2, style)
                else:
                    dec_out = self.target_decoder(s_2,x_2)
                res_tumor = self.out_tumor(dec_out) if not self.classes[1] else self.to_tumor(dec_out)
                if self.lab:
                    res_tumor = torch.cat([tumor_orig[:,:1,:,:], res_tumor], dim=1).to(device)

                if self.last_ac:
                    res_tumor = torch.tanh(res_tumor)
            res = []
            rgb_it, ir_it= iter(res_img), iter(res_tumor)
            for i in range(c.size(0)):
                if rgb[i] == i:
                    res[i] == next(rgb_it)
                else:
                    res[i] == next(ir_it)
            return torch.stack(res, dim=0).to(device)

        # res_img = adjust_contrast(res_img, nets.netContr(res_img)[:,0])
        # res_img = adjust_sharpness(res_img, nets.netContr(res_img)[:,1])
        # res_img = adjust_gamma(res_img, nets.netContr(res_img)[:,2])
        if mode=='no_target':
            return res_img,res_img
        return res_img #,tumor_orig+torch.randn_like(img_orig).to(device) #ŧesting


'''
SHAPE U NET
'''


class ShapeUNet(nn.Module):
    # the S of TarGAN

    def __init__(self, img_ch=1, mid=32, output_ch=1, real=True, qsn=False, phm=False, phm_n=4, last_layer_gen_real=True):
        super(ShapeUNet, self).__init__()
        self.img_ch = img_ch
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=mid)

        
        self.Conv2 = conv_block(ch_in=mid, ch_out=mid * 2)
        self.Conv3 = conv_block(ch_in=mid * 2, ch_out=mid * 4)
        self.Conv4 = conv_block(ch_in=mid * 4, ch_out=mid * 8)
        self.Conv5 = conv_block(ch_in=mid * 8, ch_out=mid * 16)

        if phm:
            self.Up5 = PHMTransposeConv(phm_n, mid * 16, mid * 8, kernel_size=2, stride=2)
            self.Up4 = PHMTransposeConv(phm_n, mid * 8, mid * 4, kernel_size=2, stride=2)
            self.Up3 = PHMTransposeConv(phm_n, mid * 4, mid * 2, kernel_size=2, stride=2)
            self.Up2 = PHMTransposeConv(phm_n, mid * 2, mid * 1, kernel_size=2, stride=2)
            self.Conv_1x1 = PHMConv(phm_n, mid * 1, output_ch, kernel_size=1)
        elif qsn:
            self.Up5 = QuaternionTransposeConv(mid * 16, mid * 8, kernel_size=2, stride=2)
            self.Up4 = QuaternionTransposeConv(mid * 8, mid * 4, kernel_size=2, stride=2)
            self.Up3 = QuaternionTransposeConv(mid * 4, mid * 2, kernel_size=2, stride=2)
            self.Up2 = QuaternionTransposeConv(mid * 2, mid * 1, kernel_size=2, stride=2)
            self.Conv_1x1 = QuaternionConv(mid * 1, output_ch, kernel_size=1, stride=1)
            # self.Conv_1x1 = nn.Conv2d(mid * 1, output_ch, kernel_size=1)

        elif real:
            self.Up5 = nn.ConvTranspose2d(mid * 16, mid * 8, kernel_size=2, stride=2)
            self.Up4 = nn.ConvTranspose2d(mid * 8, mid * 4, kernel_size=2, stride=2)
            self.Up3 = nn.ConvTranspose2d(mid * 4, mid * 2, kernel_size=2, stride=2)
            self.Up2 = nn.ConvTranspose2d(mid * 2, mid * 1, kernel_size=2, stride=2)
            self.Conv_1x1 = nn.Conv2d(mid * 1, output_ch, kernel_size=1)
        if last_layer_gen_real:
            self.Conv_1x1 = nn.Conv2d(mid * 1, output_ch, kernel_size=1)

        self.Up_conv5 = conv_block(ch_in=mid * 16, ch_out=mid * 8)
        self.Up_conv4 = conv_block(ch_in=mid * 8, ch_out=mid * 4)
        self.Up_conv3 = conv_block(ch_in=mid * 4, ch_out=mid * 2)
        self.Up_conv2 = conv_block(ch_in=mid * 2, ch_out=mid * 1)
        self.phm = phm
        self.qsn = qsn
    def forward(self, x):
        # encoding path
        # if (not args.real and args.soup) and not args.last_layer_gen_real:
        #     x = x.repeat(1, 3, 1, 1)
        #     x = torch.cat([x, grayscale(x)], 1)
        # wavelet
        if self.qsn or self.phm:
            x = torch.cat((x, torch.zeros(x.shape[0], self.img_ch-x.shape[1] ,x.shape[2],x.shape[3]).to(device)), dim=1).to(device)
      
        x1 = self.Conv1(x)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)


        return torch.sigmoid(d1)


####################################################################
import math

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=6, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out 



class DecoderStyle(nn.Module):
    def __init__(self, in_c, style_dim, img_size, w_hpf=0, max_conv_dim=384):
        super(DecoderStyle, self).__init__()
        self.decode = nn.ModuleList()
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        
        # for _ in range(repeat_num):
        #     dim_out = min(in_c*2, max_conv_dim)
        #     self.decode.insert(0, AdainResBlk(dim_out, in_c, style_dim, w_hpf=w_hpf, upsample=True))  # stack-like
        #     in_c = dim_out
        
        self.decode.append(AdainResBlk(192, 192, style_dim, w_hpf=w_hpf, upsample=True))  # stack-like
        # bottleneck blocks
        # for _ in range(2):
        self.decode.append(AdainResBlk(224, 64, style_dim, w_hpf=w_hpf))
        #self.decode.insert(0, AdainResBlk(256, 256, style_dim, w_hpf=w_hpf))
        #self.decode.insert(0, AdainResBlk(192, 192, style_dim, w_hpf=w_hpf))
    
    
    def forward(self, share_input, encoder_input, style):
        # encoder_input.reverse()
        # x = 0
        # for i, layer in enumerate(self.decoder):
        #     x = torch.cat([share_input, encoder_input[i][0]], dim=1)
        #     # print(x.shape,share_input.shape, encoder_input[i][0].shape)
        #     x = layer(x)
        #     share_input = x
        # return x
        encoder_input.reverse()
        for i,block in enumerate(self.decode):
            x = torch.cat([share_input, encoder_input[i][0]], dim=1)
            x = block(x, style)
            share_input = x
        return x



class DiscriminatorStyle(nn.Module):
    def __init__(self, img_size=256, num_domains=6, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out



# Defines the total variation (TV) loss, which encourages spatial smoothness in the translated image.
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



from pytorch_msssim import SSIM

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return ( 1 - super(SSIM_Loss, self).forward(img1, img2) )



class ContrastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(pow(2,4)*pow(61,2), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def apply_transforms(self,img_b, coeff):
        x,y,z = coeff[:,0], coeff[:,1], coeff[:,2]
        imgs = [TF.adjust_contrast(img, abs(x)[i]) for i,img in enumerate(img_b)]
        img_b = torch.stack(imgs) 
        imgs = [TF.adjust_sharpness(img, abs(y)[i]) for i,img in enumerate(img_b)]
        img_b = torch.stack(imgs) 
        imgs = [TF.adjust_gamma(img, abs(z)[i]) for i,img in enumerate(img_b)]
        img_b = torch.stack(imgs) 
        return img_b
