import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'gll':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
                             n_blocks_local=3, norm_layer=norm_layer)
    elif which_model_netG == 'cascaded':
        netG = cascaded(input_nc, output_nc, ngf)
    elif which_model_netG == 'generator':
        netG = Generator(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'multi':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D=3, getIntermFeat=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class GANLoss_multi(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss_multi, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# gloal local generator
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####           
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample            
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock_gll(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock_gll(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock_gll(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_gll, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
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


class NLayerDiscriminator_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator_multi, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_multi(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


##cascaded network
class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class cascaded(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(cascaded, self).__init__()

        # Layer1 4*4---8*8
        self.conv1 = nn.Conv2d(input_nc, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay1 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv11 = nn.Conv2d(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay11 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
        self.relu11 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Layer2 8*8---16*16
        self.conv2 = nn.Conv2d(ngf * 16 + input_nc, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay2 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv22 = nn.Conv2d(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay22 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
        self.relu22 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # layer3 16*16---32*32
        self.conv3 = nn.Conv2d(ngf * 16 + input_nc, ngf * 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay3 = LayerNorm(ngf * 8, eps=1e-12, affine=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv33 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay33 = LayerNorm(ngf * 8, eps=1e-12, affine=True)
        self.relu33 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # layer4 32*32---64*64
        self.conv4 = nn.Conv2d(ngf * 8 + input_nc, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay4 = LayerNorm(ngf * 4, eps=1e-12, affine=True)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv44 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay44 = LayerNorm(ngf * 4, eps=1e-12, affine=True)
        self.relu44 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # layer5 64*64---128*128

        self.conv5 = nn.Conv2d(ngf * 4 + input_nc, ngf * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay5 = LayerNorm(ngf * 2, eps=1e-12, affine=True)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv55 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay55 = LayerNorm(ngf * 2, eps=1e-12, affine=True)
        self.relu55 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # layer6 128*128---256*256
        self.conv6 = nn.Conv2d(ngf * 2 + input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay6 = LayerNorm(ngf, eps=1e-12, affine=True)
        self.relu6 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv66 = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=True)
        self.lay66 = LayerNorm(ngf, eps=1e-12, affine=True)
        self.relu66 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Layer7 256*256
        self.conv7 = nn.Conv2d(ngf + input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True)

        # Layer_downsample
        self.downsample = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

    def forward(self, input):
        input_128 = self.downsample(input)
        input_64 = self.downsample(input_128)
        input_32 = self.downsample(input_64)
        input_16 = self.downsample(input_32)
        input_8 = self.downsample(input_16)
        input_4 = self.downsample(input_8)

        # Layer1 4*4---8*8
        out1 = self.conv1(input_4)
        L1 = self.lay1(out1)
        out2 = self.relu1(L1)

        out11 = self.conv11(out2)
        L11 = self.lay11(out11)
        out22 = self.relu11(L11)

        m = nn.Upsample(size=(input_4.size(3) * 2, input_4.size(3) * 2), mode='bilinear')

        img1 = torch.cat((m(out22), input_8), 1)

        # Layer2 8*8---16*16
        out3 = self.conv2(img1)
        L2 = self.lay2(out3)
        out4 = self.relu2(L2)

        out33 = self.conv22(out4)
        L22 = self.lay22(out33)
        out44 = self.relu22(L22)

        m = nn.Upsample(size=(input_8.size(3) * 2, input_8.size(3) * 2), mode='bilinear')

        img2 = torch.cat((m(out44), input_16), 1)

        # Layer3 16*16---32*32
        out5 = self.conv3(img2)
        L3 = self.lay3(out5)
        out6 = self.relu3(L3)

        out55 = self.conv33(out6)
        L33 = self.lay33(out55)
        out66 = self.relu33(L33)

        m = nn.Upsample(size=(input_16.size(3) * 2, input_16.size(3) * 2), mode='bilinear')

        img3 = torch.cat((m(out66), input_32), 1)

        # Layer4 32*32---64*64
        out7 = self.conv4(img3)
        L4 = self.lay4(out7)
        out8 = self.relu4(L4)

        out77 = self.conv44(out8)
        L44 = self.lay44(out77)
        out88 = self.relu44(L44)

        m = nn.Upsample(size=(input_32.size(3) * 2, input_32.size(3) * 2), mode='bilinear')

        img4 = torch.cat((m(out88), input_64), 1)

        # Layer5 64*64---128*128
        out9 = self.conv5(img4)
        L5 = self.lay5(out9)
        out10 = self.relu5(L5)

        out99 = self.conv55(out10)
        L55 = self.lay55(out99)
        out110 = self.relu55(L55)

        m = nn.Upsample(size=(input_64.size(3) * 2, input_64.size(3) * 2), mode='bilinear')

        img5 = torch.cat((m(out110), input_128), 1)

        # Layer6 128*128---256*256
        out11 = self.conv6(img5)
        L6 = self.lay6(out11)
        out12 = self.relu6(L6)

        out111 = self.conv66(out12)
        L66 = self.lay66(out111)
        out112 = self.relu66(L66)

        m = nn.Upsample(size=(input_128.size(3) * 2, input_128.size(3) * 2), mode='bilinear')

        img6 = torch.cat((m(out112), input), 1)

        # Layer7 256*256
        out13 = self.conv7(img6)

        return out13


class BNPReLU(nn.Module):
    def __init__(self, in_ch):
        super(BNPReLU, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.prelu = nn.PReLU(in_ch)

    def forward(self, x):
        out = self.bn(x)
        out = self.prelu(out)
        return out


class ConvBNPReLU(nn.Module):
    def __init__(self, in_ch, out_ch, KSize, stride, padding, dilation=(1, 1), groups=1, state=False, bias=False):
        super(ConvBNPReLU, self).__init__()
        self.state = state
        self.conv = nn.Conv2d(in_ch, out_ch, KSize, stride, padding, dilation, groups, bias)
        if self.state:
            self.bnprelu = BNPReLU(out_ch)

    def forward(self, x):
        out = self.conv(x)
        if self.state:
            out = self.bnprelu(out)

        return out


class BNReLU(nn.Module):
    def __init__(self, in_ch):
        super(BNReLU, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, KSize, stride, padding, dilation=(1, 1), groups=1, state=False, bias=False):
        super(ConvBNReLU, self).__init__()
        self.state = state
        self.conv = nn.Conv2d(in_ch, out_ch, KSize, stride, padding, dilation, groups, bias)
        if self.state:
            self.bnrelu = BNReLU(out_ch)

    def forward(self, x):
        out = self.conv(x)
        if self.state:
            out = self.bnrelu(out)

        return out


class DSAModule(nn.Module):
    def __init__(self,in_ch, kSize=1):
        super(DSAModule,self).__init__()

        self.conv1 = nn.Sequential(
            BNPReLU(in_ch),
            ConvBNPReLU(in_ch, in_ch, kSize, 1, padding=0, state=True),
        )
        self.dconv2 = ConvBNPReLU(in_ch, in_ch, 3, 1, padding=3, dilation=3)
        self.dconv3 = ConvBNPReLU(in_ch, in_ch, 3, 1, padding=3, dilation=3)
        self.conv4 = ConvBNPReLU(in_ch, in_ch, 1, 1, padding=0, state=True)
        self.conv5 = ConvBNPReLU(in_ch, in_ch, 1, 1, padding=0, state=True)
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.conv1(x)
        br1 = self.dconv2(x1)
        br2 = self.dconv3(x1)
        br3 = self.conv4(x1)
        br12 = torch.mul(br1, br2)
        br12 = self.sigmoid(br12)
        br123 = torch.mul(br12, br3)
        br = torch.add(br123, x1)
        br = self.conv5(br)
        out = torch.add(x, br)
        return out


class CAttention(nn.Module):
    def __init__(self, in_ch, r=2):
        super(CAttention, self).__init__()
        self.conv1 = ConvBNReLU(in_ch, in_ch // r, 1, 1, 0)
        self.conv2 = ConvBNReLU(in_ch // r, in_ch, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        b, c, h, w = x1.size()
        feat = F.adaptive_avg_pool2d(x1, (1, 1)).view(b, c)
        feat = feat.view(b, c, 1, 1)
        feat = feat.expand_as(x1).clone()
        feat = self.conv2(feat)
        out = torch.add(feat, x)
        return out


class DualAttention(nn.Module):
    def __init__(self, in_ch):
        super(DualAttention, self).__init__()
        self.spatial = DSAModule(in_ch)
        self.conv1 = ConvBNPReLU(in_ch, in_ch, 1, 1, 0, state=True)
        self.channel = CAttention(in_ch)
        self.conv2 = ConvBNPReLU(in_ch, in_ch, 1, 1, 0, state=True)

    def forward(self, x):
        sp = self.spatial(x)
        sp = self.conv1(sp)
        ch = self.channel(x)
        ch = self.conv2(ch)
        out = torch.add(sp, ch)
        return out


class NewModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(NewModule, self).__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch // 2, 1, 1, 0, state=True)
        self.conv2 = ConvBNReLU(out_ch // 2, out_ch // 4, 3, 1, 1, state=True)
        self.conv3 = ConvBNReLU(out_ch // 4, out_ch // 8, 3, 1, 1, state=True)
        self.conv4 = ConvBNReLU(out_ch // 8, out_ch // 8, 3, 1, 1, state=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf):
        super(Generator, self).__init__()
        self.nm0 = NewModule(in_ch, ngf)

        self.nm1 = NewModule(ngf, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.nm2 = NewModule(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.nm3 = NewModule(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.nm4 = NewModule(256, 512)
        self.trans1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.att1 = DualAttention(512)
        self.nm5 = NewModule(512 + 256, 256)
        self.trans2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.att2 = DualAttention(256)
        self.nm6 = NewModule(256 + 128, 128)
        self.trans3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.att3 = DualAttention(128)

        self.nm7 = NewModule(128 + 64, 128)
        self.conv1 = ConvBNReLU(128, ngf, 1, 1, 0)
        # self.conv2 = ConvBNReLU(8, out_ch, 1, 1, 0)
        self.conv2 = ConvBNReLU(ngf + ngf, out_ch, 1, 1, 0)

    def forward(self, x):
        x00 = self.nm0(x)

        x10 = self.nm1(x00)
        x11 = self.pool1(x10)
        x20 = self.nm2(x11)
        x21 = self.pool2(x20)
        x30 = self.nm3(x21)
        x31 = self.pool3(x30)

        x40 = self.nm4(x31)
        x41 = self.trans1(x40)
        x42 = self.att1(x41)
        x43 = torch.cat([x42, x30], dim=1)
        x50 = self.nm5(x43)
        x51 = self.trans2(x50)
        x52 = self.att2(x51)
        x53 = torch.cat([x52, x20], dim=1)
        x60 = self.nm6(x53)
        x61 = self.trans3(x60)
        x62 = self.att3(x61)
        x63 = torch.cat([x62, x10], dim=1)

        x70 = self.nm7(x63)
        x71 = self.conv1(x70)
        x72 = torch.cat([x71, x00], dim=1)
        # x72 = torch.add(x71, x00)
        x73 = self.conv2(x72)
        out = torch.tanh(x73)
        return out
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
#
#
# class BNReLU(nn.Module):
#     def __init__(self, in_ch):
#         super(BNReLU, self).__init__()
#         self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.bn(x)
#         out = self.relu(out)
#         return out
#
#
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, KSize, stride, padding, dilation=(1, 1), groups=1, state=False, bias=False):
#         super(ConvBNReLU, self).__init__()
#         self.state = state
#         self.conv = nn.Conv2d(in_ch, out_ch, KSize, stride, padding, dilation, groups, bias)
#         if self.state:
#             self.bnrelu = BNReLU(out_ch)
#
#     def forward(self, x):
#         out = self.conv(x)
#         if self.state:
#             out = self.bnrelu(out)
#
#         return out
#
#
# class Generator(nn.Module):
#     def __init__(self, in_ch, out_ch, ngf):
#         super(Generator, self).__init__()
#         self.nm0 = ConvBNReLU(in_ch, ngf, 3, 1, 1, state=True)
#
#         self.nm1 = ConvBNReLU(ngf, 64, 3, 1, 1, state=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.nm2 = ConvBNReLU(64, 128, 3, 1, 1)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.nm3 = ConvBNReLU(128, 256, 3, 1, 1)
#         self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.nm4 = ConvBNReLU(256, 512, 3, 1, 1, state=True)
#         self.trans1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
#         self.nm5 = ConvBNReLU(512 + 256, 256, 3, 1, 1, state=True)
#         self.trans2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.nm6 = ConvBNReLU(256 + 128, 128, 3, 1, 1, state=True)
#         self.trans3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#
#         self.nm7 = ConvBNReLU(128 + 64, 128, 3, 1, 1, state=True)
#         self.conv1 = ConvBNReLU(128, ngf, 1, 1, 0)
#         # self.conv2 = ConvBNReLU(8, out_ch, 1, 1, 0)
#         self.conv2 = ConvBNReLU(ngf + ngf, out_ch, 1, 1, 0)
#
#     def forward(self, x):
#         x00 = self.nm0(x)
#
#         x10 = self.nm1(x00)
#         x11 = self.pool1(x10)
#         x20 = self.nm2(x11)
#         x21 = self.pool2(x20)
#         x30 = self.nm3(x21)
#         x31 = self.pool3(x30)
#
#         x40 = self.nm4(x31)
#         x41 = self.trans1(x40)
#         x43 = torch.cat([x41, x30], dim=1)
#         x50 = self.nm5(x43)
#         x51 = self.trans2(x50)
#         x53 = torch.cat([x51, x20], dim=1)
#         x60 = self.nm6(x53)
#         x61 = self.trans3(x60)
#         x63 = torch.cat([x61, x10], dim=1)
#
#         x70 = self.nm7(x63)
#         x71 = self.conv1(x70)
#         x72 = torch.cat([x71, x00], dim=1)
#         # x72 = torch.add(x71, x00)
#         x73 = self.conv2(x72)
#         out = torch.tanh(x73)
#         return out
#
#
# if __name__ == "__main__":
#     input = torch.Tensor(1, 1, 256, 256).cuda()
#     model = Generator(1, 3, 32).cuda()
#     model.eval()
#     print(model)
#     output = model(input)
#     summary(model, (1, 256, 256))
#     print(output.shape)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
#
#
# # model
# class BNPReLU(nn.Module):
#     def __init__(self, in_ch):
#         super(BNPReLU, self).__init__()
#         self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
#         self.prelu = nn.PReLU(in_ch)
#
#     def forward(self, x):
#         out = self.bn(x)
#         out = self.prelu(out)
#         return out
#
#
# class ConvBNPReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, KSize, stride, padding, dilation=(1, 1), groups=1, state=False, bias=False):
#         super(ConvBNPReLU, self).__init__()
#         self.state = state
#         self.conv = nn.Conv2d(in_ch, out_ch, KSize, stride, padding, dilation, groups, bias)
#         if self.state:
#             self.bnprelu = BNPReLU(out_ch)
#
#     def forward(self, x):
#         out = self.conv(x)
#         if self.state:
#             out = self.bnprelu(out)
#
#         return out
#
#
# class BNReLU(nn.Module):
#     def __init__(self, in_ch):
#         super(BNReLU, self).__init__()
#         self.bn = nn.BatchNorm2d(in_ch, eps=1e-3)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.bn(x)
#         out = self.relu(out)
#         return out
#
#
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch, out_ch, KSize, stride, padding, dilation=(1, 1), groups=1, state=False, bias=False):
#         super(ConvBNReLU, self).__init__()
#         self.state = state
#         self.conv = nn.Conv2d(in_ch, out_ch, KSize, stride, padding, dilation, groups, bias)
#         if self.state:
#             self.bnrelu = BNReLU(out_ch)
#
#     def forward(self, x):
#         out = self.conv(x)
#         if self.state:
#             out = self.bnrelu(out)
#
#         return out
#
#
# class NewModule(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(NewModule, self).__init__()
#         self.conv1 = ConvBNReLU(in_ch, out_ch // 2, 1, 1, 0, state=True)
#         self.conv2 = ConvBNReLU(out_ch // 2, out_ch // 4, 3, 1, 1, state=True)
#         self.conv3 = ConvBNReLU(out_ch // 4, out_ch // 8, 3, 1, 1, state=True)
#         self.conv4 = ConvBNReLU(out_ch // 8, out_ch // 8, 3, 1, 1, state=True)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         out = torch.cat([x1, x2, x3, x4], dim=1)
#         return out
#
#
# class Generator(nn.Module):
#     def __init__(self, in_ch, out_ch, ngf):
#         super(Generator, self).__init__()
#         self.nm0 = NewModule(in_ch, ngf)
#
#         self.nm1 = NewModule(ngf, 64)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.nm2 = NewModule(64, 128)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.nm3 = NewModule(128, 256)
#         self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.nm4 = NewModule(256, 512)
#         self.trans1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
#         self.nm5 = NewModule(512 + 256, 256)
#         self.trans2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.nm6 = NewModule(256 + 128, 128)
#         self.trans3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
#
#         self.nm7 = NewModule(128 + 64, 128)
#         self.conv1 = ConvBNReLU(128, ngf, 1, 1, 0)
#         # self.conv2 = ConvBNReLU(8, out_ch, 1, 1, 0)
#         self.conv2 = ConvBNReLU(ngf + ngf, out_ch, 1, 1, 0)
#
#     def forward(self, x):
#         x00 = self.nm0(x)
#
#         x10 = self.nm1(x00)
#         x11 = self.pool1(x10)
#         x20 = self.nm2(x11)
#         x21 = self.pool2(x20)
#         x30 = self.nm3(x21)
#         x31 = self.pool3(x30)
#
#         x40 = self.nm4(x31)
#         x41 = self.trans1(x40)
#         x43 = torch.cat([x41, x30], dim=1)
#         x50 = self.nm5(x43)
#         x51 = self.trans2(x50)
#         x53 = torch.cat([x51, x20], dim=1)
#         x60 = self.nm6(x53)
#         x61 = self.trans3(x60)
#         x63 = torch.cat([x61, x10], dim=1)
#
#         x70 = self.nm7(x63)
#         x71 = self.conv1(x70)
#         x72 = torch.cat([x71, x00], dim=1)
#         # x72 = torch.add(x71, x00)
#         x73 = self.conv2(x72)
#         out = torch.tanh(x73)
#         return out
#
#
# if __name__ == "__main__":
#     input = torch.Tensor(1, 1, 256, 256).cuda()
#     model = Generator(1, 3, 32).cuda()
#     model.eval()
#     print(model)
#     output = model(input)
#     summary(model, (1, 256, 256))
#     print(output.shape)
