import torch
import torch.nn as nn
import torch.nn.functional as F
from cupy_layers.aggregation_zeropad import LocalConvolution
from models.layers import get_act_layer


class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size*self.kernel_size, qk_hh, qk_ww)
        
        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
        
        return out.contiguous()

class Cotlay(nn.Module):
    def __init__(self):
        super(CotNet18, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CotLayer()  
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)  

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x


model = Cotlay()
print(model)
