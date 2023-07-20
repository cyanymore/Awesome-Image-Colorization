import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# loss
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


# model
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
        self.conv3 = ConvBNReLU(out_ch // 4, out_ch // 8, 1, 1, 0, state=True)
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


if __name__ == "__main__":
    input = torch.Tensor(1, 1, 256, 256).cuda()
    model = Generator(1, 3, 32).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (1, 256, 256))
    print(output.shape)
