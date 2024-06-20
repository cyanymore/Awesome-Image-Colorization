import numpy as np
import torch
from PIL import Image
import os
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import cv2
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .losses import init_loss
import torch_optimizer as optimizer
from . import Color
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
use_amp = True
scaler = torch.cuda.amp.GradScaler()
import math
from ptflops import get_model_complexity_info
from torch.optim.optimizer import Optimizer

class Adam16(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.float().cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,fp32_group in zip(self.param_groups,self.fp32_param_groups):
            for p,fp32_p in zip(group['params'],fp32_group['params']):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()
def orb_input(image_orb):
    #print(image_orb.shape,type(image_orb))
    real_A = image_orb[0, :,:, :]
    orb = cv2.ORB_create(nfeatures=5000)
    img1=real_A.cpu().detach().numpy()#np.array(real_A)
    key, des1 = orb.detectAndCompute(img1.transpose(1,2,0),None)
    #if des1 is None:
       #return 1;
    return key

class MobileARModel(BaseModel):
    def name(self):
        return 'PerCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        
        
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        macs, params = get_model_complexity_info(self.netG_A, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFeat = init_loss(opt, self.Tensor)
            self.contentLoss = init_loss(opt, self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSyn = torch.nn.L1Loss()
            self.criterionCS = torch.nn.L1Loss()
            #self.ssim_module = SSIM(data_range=255, size_average=True, channel=3)
            #self.ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
            self.criterionCDGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            
            # initialize optimizers
            self.optimizer_G = optimizer.DiffGrad(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),eps=1e-04,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = optimizer.DiffGrad(self.netD_A.parameters(), eps=1e-04 ,lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = optimizer.DiffGrad(self.netD_B.parameters(), eps=1e-04, lr=opt.lr, betas=(opt.beta1, 0.999))
            
            #self.optimizer_G = Adam16(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())
            #self.optimizer_G = Adam16(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
            #self.optimizer_D_A = Adam16(self.netD_A.parameters())
            #self.optimizer_D_B = Adam16(self.netD_B.parameters())
            #self.netG_A, self.optimizer_G = torch.cuda.amp.initialize(self.netG_A, self.optimizer_G, opt_level="O1") 

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        #networks.print_network(self.netG_B)
        if self.isTrain:
            #networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
       

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        #self.real_A = Variable(self.input_A, volatile=True)
        #self.fake_B = self.netG_A.forward(self.real_A)
        #self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        #self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        with torch.cuda.amp.autocast():
        # Real
            pred_real = netD.forward(real)
            loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
            pred_fake = netD.forward(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        #loss_D1 = (torch.mean((pred_real - torch.mean(pred_fake) - 1) ** 2) + torch.mean((pred_fake - torch.mean(pred_real) + 1) ** 2))/2
            loss_D = scaler.scale((loss_D_real + loss_D_fake) * 0.5) #+loss_D1
            #print(loss_D)
             
        # backward
            #loss_D.backward()
            #return loss_D
            scaler.scale(loss_D).backward()
            return scaler.scale(loss_D)#loss_D

    def backward_D_A(self):
        with torch.cuda.amp.autocast(enabled=True):
             fake_B = self.fake_B_pool.query(self.fake_B)
             self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        with torch.cuda.amp.autocast(enabled=True):
             fake_A = self.fake_A_pool.query(self.fake_A)
             self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    def backward_D_basic1(self, netD, real, fake):
        with torch.cuda.amp.autocast(enabled=True):
        # Real
            pred_real = netD(real)
            loss_D_real = self.criterionCDGAN(pred_real, True)
        # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionCDGAN(pred_fake, False)
        # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        #loss_D.backward()
            scaler.scale(loss_D).backward()
            return scaler.scale(loss_D)#loss_D

    def backward_D_A1(self):
        rec_B = self.rec_B
        self.loss_D_A1 = self.backward_D_basic1(self.netD_A, self.real_B, rec_B)

    def backward_D_B1(self):
        rec_A = self.rec_A
        self.loss_D_B1 = self.backward_D_basic1(self.netD_B, self.real_A, rec_A)

    def backward_G(self):
        with torch.cuda.amp.autocast(enabled=True):
            lambda_idt = self.opt.identity
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B

            lambda_feat_AfA = self.opt.lambda_feat_AfA    
            lambda_feat_BfB = self.opt.lambda_feat_BfB

            lambda_feat_fArecA = self.opt.lambda_feat_fArecA
            lambda_feat_fBrecB = self.opt.lambda_feat_fBrecB

            lambda_feat_ArecA = self.opt.lambda_feat_ArecA
            lambda_feat_BrecB = self.opt.lambda_feat_BrecB
        
            lambda_syn_A = self.opt.lambda_syn_A
            lambda_syn_B = self.opt.lambda_syn_B
        
            lambda_CS_A = self.opt.lambda_CS_A
            lambda_CS_B = self.opt.lambda_CS_B
        

        # Identity loss
            if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
                self.idt_A = self.netG_A.forward(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
                self.idt_B = self.netG_B.forward(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0
        
        # GAN loss
        # D_A(G_A(A))
            self.fake_B = self.netG_A.forward(self.real_A)
            #print(self.fake_B.dtype())
            pred_fake = self.netD_A.forward(self.fake_B)
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
            self.fake_A = self.netG_B.forward(self.real_B)
            pred_fake = self.netD_B.forward(self.fake_A)
            self.loss_G_B = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
            self.rec_A = self.netG_B.forward(self.fake_B) 
        
        ########################Keypoint Loss############################
            real_A = np.squeeze(self.real_A)
            real_A= real_A.permute(1,2,0)
            real_A=real_A.cpu().detach().numpy()
            fake_A = np.squeeze(self.fake_A)
            fake_A= fake_A.permute(1,2,0)
            fake_A=fake_A.cpu().detach().numpy()
        ########self.color_lossA=Color.color_loss(real_A,fake_A)
            real_B = np.squeeze(self.real_B)
            real_B= real_B.permute(1,2,0)
            real_B=real_B.cpu().detach().numpy()
            fake_B = np.squeeze(self.fake_B)
            fake_B= fake_B.permute(1,2,0)
            fake_B=fake_B.cpu().detach().numpy()
       

        # Synthesized loss
            self.loss_SynB = self.criterionSyn(self.fake_B, self.real_B) * lambda_syn_B
            self.loss_SynA = self.criterionSyn(self.fake_A, self.real_A) * lambda_syn_A

            self.rec_A = self.netG_B(self.fake_B)
#        

            self.rec_B = self.netG_A(self.fake_A)
#        

     
        # CS loss
            self.loss_CSA = self.criterionCS(self.fake_A, self.rec_A.detach()) * lambda_CS_A
            self.loss_CSB = self.criterionCS(self.fake_B, self.rec_B.detach()) * lambda_CS_B

        
        
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        # Backward cycle loss
            self.rec_B = self.netG_A.forward(self.fake_A)

        
        
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B


        
            self.feat_loss_AfA = self.criterionFeat.get_loss(self.fake_A, self.real_A) * 1#lambda_feat_AfA    
            self.feat_loss_BfB = self.criterionFeat.get_loss(self.fake_B, self.real_B) *1# lambda_feat_BfB

            self.feat_loss_fArecA = self.criterionFeat.get_loss(self.fake_A, self.rec_A) *1# lambda_feat_fArecA
            self.feat_loss_fBrecB = self.criterionFeat.get_loss(self.fake_B, self.rec_B) *1# lambda_feat_fBrecB

            self.feat_loss_ArecA = self.criterionFeat.get_loss(self.rec_A, self.real_A) *1# lambda_feat_ArecA 
            self.feat_loss_BrecB = self.criterionFeat.get_loss(self.rec_B, self.real_B  ) *1# lambda_feat_BrecB

            self.feat_loss = self.feat_loss_AfA+ self.feat_loss_BfB+ self.feat_loss_fArecA+ self.feat_loss_fBrecB + self.feat_loss_ArecA + self.feat_loss_BrecB
            self.L1 = self.loss_SynA  +self.loss_SynB + self.loss_CSA + self.loss_CSB
        # combined loss
            self.loss_G = scaler.scale(self.loss_G_A + self.loss_G_B  + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.feat_loss +self.L1)
            #self.loss_G.backward()
            scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
       
        self.backward_G()
        scaler.step(self.optimizer_G)
        scaler.update()
        #self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        #self.optimizer_D_A.step()
        scaler.step(self.optimizer_D_A)
        scaler.update()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        #sself.optimizer_D_B.step()
        scaler.step(self.optimizer_D_B)
        scaler.update()

    def get_current_errors(self):
        D_A = self.loss_D_A.data.item()
        G_A = self.loss_G_A.data.item()
        Cyc_A = self.loss_cycle_A.data.item()
        D_B = self.loss_D_B.data.item()
        G_B = self.loss_G_B.data.item()
        Cyc_B = self.loss_cycle_B.data.item()
        CSA = self.loss_CSA.data.item()
        CSB = self.loss_CSB.data.item()
        SynA = self.loss_SynA.data.item()
        SynB = self.loss_SynB.data.item()
        
        
        
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data.item()
            idt_B = self.loss_idt_B.data.item()
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B), 
#                                ('D_A1', D_A1), ('G_A1', G_A1), ('D_B1', D_B1), ('G_B1', G_B1), 
                                ('CSA', CSA), ('CSB', CSB), ('SynA', SynA), ('SynB', SynB) ])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
#                                ('D_A1', D_A1), ('G_A1', G_A1), ('D_B1', D_B1), ('G_B1', G_B1), 
                                ('CSA', CSA), ('CSB', CSB), ('SynA', SynA), ('SynB', SynB) ])

    def get_current_visuals(self):
        #real_A = util.tensor2im(self.real_A.data)
        #fake_B = util.tensor2im(self.fake_B.data)
        #rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        #rec_B = util.tensor2im(self.rec_B.data)
        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([
                                ('real_B', real_B), ('fake_A', fake_A)])
        else:
            return OrderedDict([
                                ('real_B', real_B), ('fake_A', fake_A)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

