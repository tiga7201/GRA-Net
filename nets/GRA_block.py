import torch.nn as nn
import torch
from nets.dcn_v2 import DeformConv2d


class BaConv(nn.Module):
    def __init__(self, in_c, out_c, k, p, d):
        super(BaConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, dilation=d)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LRM(nn.Module):
    def __init__(self, c):
        super(LRM, self).__init__()
        self.conv_r1_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r1_2 = BaConv(c // 4, c // 4, (1, 3), (0, 1), 1)
        self.conv_r1_3 = BaConv(c // 4, c // 4, (3, 1), (1, 0), 1)
        self.conv_r1_4 = BaConv(c // 2, c // 4, (3, 3), (1, 1), 1)

        self.conv_r2_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r2_2 = BaConv(c // 4, c // 4, (1, 3), (0, 1), 1)
        self.conv_r2_3 = BaConv(c // 4, c // 4, (3, 1), (1, 0), 1)
        self.conv_r2_4 = BaConv(c // 2, c // 4, (3, 3), (3, 3), 3)

        self.conv_r3_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r3_2 = BaConv(c // 4, c // 4, (3, 3), (1, 1), 1)
        self.conv_r3_3 = BaConv(c // 4, c // 4, (3, 3), (1, 1), 1)
        self.conv_r3_4 = BaConv(c // 4, c // 4, (3, 3), (4, 4), 4)

        self.conv_r4_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_l1 = BaConv(c // 4, c // 4, (15, 1), (7, 0), 1)
        self.conv_l2 = BaConv(c // 4, c // 4, (1, 15), (0, 7), 1)
        self.conv_h1 = BaConv(c // 4, c // 4, (1, 15), (0, 7), 1)
        self.conv_h2 = BaConv(c // 4, c // 4, (15, 1), (7, 0), 1)

    def forward(self, x):
        x1 = self.conv_r1_1(x)
        x1_2 = self.conv_r1_2(x1)
        x1_3 = self.conv_r1_3(x1)
        x1 = torch.cat([x1_2, x1_3], dim=1)
        x1 = self.conv_r1_4(x1)

        x2 = self.conv_r2_1(x)
        x2_2 = self.conv_r2_2(x2)
        x2_3 = self.conv_r2_3(x2)
        x2 = torch.cat([x2_2, x2_3], dim=1)
        x2 = self.conv_r2_4(x2)

        x3 = self.conv_r3_1(x)
        x3 = self.conv_r3_2(x3)
        x3 = self.conv_r3_3(x3)
        x3 = self.conv_r3_4(x3)

        x4 = self.conv_r4_1(x)
        x_l = self.conv_l1(x4)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_h1(x4)
        x_r = self.conv_h2(x_r)
        x4 = x_l + x_r

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class FIA(nn.Module):
    def __init__(self, c, h, w):
        super(FIA, self).__init__()
        self.conv = nn.Conv2d(c, c // 2, (1, 1))
        self.local_pool = nn.AdaptiveAvgPool2d((h // 4, w // 4))
        self.softmax = nn.Softmax(dim=-1)
        self.de_conv1 = nn.ConvTranspose2d(160, 160, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0))
        self.de_conv2 = nn.ConvTranspose2d(160, 160, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))

    def forward(self, x):
        p = self.conv(x)
        p1 = self.local_pool(p)
        n, c, h, w = p1.size()
        p1 = p1.view(n, c, h*w).permute(0, 2, 1)
        p2 = self.local_pool(p).view(n, c, h*w)
        p3 = self.local_pool(p).view(n, c, h*w)
        a0 = torch.bmm(p1, p2)
        a = self.softmax(a0)
        sf = torch.bmm(p3, a)
        s = sf.view(n, c, h, w)
        y0 = self.de_conv1(s)
        y = self.de_conv2(y0)  # (30, 30)
        z = torch.cat([y, p], dim=1)

        return z


class ABR(nn.Module):
    def __init__(self, c):
        super(ABR, self).__init__()
        self.dcn1 = DeformConv2d(c, c, kernel_size=3, padding=1, stride=1, bias=None, modulation=True)
        self.relu = nn.ReLU(inplace=True)
        self.dcn2 = DeformConv2d(c, c, kernel_size=3, padding=1, stride=1, bias=None, modulation=True)

    def forward(self, x):
        residual = x
        x = self.dcn1(x)
        x = self.relu(x)
        x = self.dcn2(x)
        out = x + residual

        return out


class LRMs(nn.Module):
    def __init__(self, c):
        super(LRMs, self).__init__()
        self.conv_r1_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r1_2 = BaConv(c // 4, c // 2, (3, 3), (1, 1), 1)
        self.conv_r1_4 = BaConv(c // 2, c // 4, (3, 3), (1, 1), 1)

        self.conv_r2_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r2_2 = BaConv(c // 4, c // 2, (3, 3), (1, 1), 1)
        self.conv_r2_4 = BaConv(c // 2, c // 4, (3, 3), (3, 3), 3)

        self.conv_r3_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r3_2 = BaConv(c // 4, c // 4, (3, 3), (1, 1), 1)
        self.conv_r3_3 = BaConv(c // 4, c // 4, (3, 3), (1, 1), 1)
        self.conv_r3_4 = BaConv(c // 4, c // 4, (3, 3), (4, 4), 4)

        self.conv_r4_1 = BaConv(c, c // 4, (1, 1), 0, 1)
        self.conv_r4_2 = BaConv(c // 4, c // 4, (15, 15), (7, 7), 1)

    def forward(self, x):
        x1 = self.conv_r1_1(x)
        x1 = self.conv_r1_2(x1)
        x1 = self.conv_r1_4(x1)

        x2 = self.conv_r2_1(x)
        x2 = self.conv_r2_2(x2)
        x2 = self.conv_r2_4(x2)

        x3 = self.conv_r3_1(x)
        x3 = self.conv_r3_2(x3)
        x3 = self.conv_r3_3(x3)
        x3 = self.conv_r3_4(x3)

        x4 = self.conv_r4_1(x)
        x4 = self.conv_r4_2(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x
