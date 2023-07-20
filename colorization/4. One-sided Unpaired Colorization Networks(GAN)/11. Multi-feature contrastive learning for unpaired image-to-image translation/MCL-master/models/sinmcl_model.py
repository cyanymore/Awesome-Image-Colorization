import torch
from .imcl_model import IMCLModel
from third_party.gather_layer import GatherLayer
import torch.nn.functional as F


class SinMCLModel(IMCLModel):
    """ This class implements the single image translation model (Fig 1) of
    Multi-feature Contrastive Learning for Unpaired Image-to-Image Translation,
    Yao Gou, Min Li, Yu Song, Yujie He, LiTao Wang, CAIS, 2022
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = IMCLModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_R1', type=float, default=1.0,
                            help='weight for the R1 gradient penalty')
        parser.add_argument('--lambda_identity', type=float, default=1.0,
                            help='the "identity preservation loss"')
        parser.set_defaults(nce_includes_all_negatives_from_minibatch=True,
                            dataset_mode="singleimage",
                            netG="stylegan2",
                            stylegan2_G_num_downsampling=1,
                            netD="stylegan2",
                            gan_mode="nonsaturating",
                            num_patches=1,
                            nce_layers="0,2,4",
                            lambda_NCE=4.0,
                            ngf=10,
                            ndf=8,
                            lr=0.002,
                            beta1=0.0,
                            beta2=0.99,
                            load_size=1024,
                            crop_size=64,
                            preprocess="zoom_and_patch",
        )

        if is_train:
            parser.set_defaults(preprocess="zoom_and_patch",
                                batch_size=16,
                                save_epoch_freq=1,
                                save_latest_freq=20000,
                                n_epochs=8,
                                n_epochs_decay=8,

            )
        else:
            parser.set_defaults(preprocess="none",  # load the whole image as it is
                                batch_size=1,
                                num_test=1,
            )
            
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if self.isTrain:
            if opt.lambda_R1 > 0.0:
                self.loss_names += ['D_R1']
            if opt.lambda_identity > 0.0:
                self.loss_names += ['idt']

    def compute_D_loss(self):
        self.real_B.requires_grad_()
        GAN_loss_D = super().compute_D_loss()
        self.loss_D_R1 = self.R1_loss(self.pred_real, self.real_B)
        self.loss_D = GAN_loss_D + self.loss_D_R1
        return self.loss_D

    def compute_G_loss(self):
        IMCL_loss_G = super().compute_G_loss()
        self.loss_idt = torch.nn.functional.l1_loss(self.idt_B, self.real_B) * self.opt.lambda_identity
        return IMCL_loss_G + self.loss_idt

    def R1_loss(self, real_pred, real_img):
        grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True, retain_graph=True)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.opt.lambda_R1 * 0.5)

    def mcl_fake(self, out1, out2, temperature, distributed=False):
        if distributed:
            out1 = torch.cat(GatherLayer.apply(out1), dim=0)
            out2 = torch.cat(GatherLayer.apply(out2), dim=0)
        N = out1.size(0)

        _out = [out1, out2]
        outputs = torch.cat(_out, dim=0)
        sim_matrix = outputs @ outputs.t()
        sim_matrix = sim_matrix / temperature
        sim_matrix.fill_diagonal_(-5e4)

        mask = torch.zeros_like(sim_matrix)
        mask[N:, N:] = 1
        mask.fill_diagonal_(0)

        sim_matrix = sim_matrix[N:]
        mask = mask[N:]
        mask = mask / mask.sum(1, keepdim=True)

        lsm = F.log_softmax(sim_matrix, dim=1)
        lsm = lsm * mask
        d_loss = -lsm.sum(1).mean()
        return d_loss
