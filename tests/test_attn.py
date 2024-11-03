import editdistance
import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform

class ScaleNet(nn.Module):
    def __init__(self,in_channels, size=(4,15)):
        super(ScaleNet, self).__init__()
        conv_func = conv_with_kaiming_uniform(True, True, use_deformable=False, use_bn=False)
        self.h, self.w = size
        self.rescale = nn.Upsample(size=(self.h, self.w*2), mode='bilinear', align_corners=False)
        self.attn_conv = nn.Sequential(
            conv_func(in_channels, in_channels, 3, stride=(2, 1)),
            conv_func(in_channels, in_channels//2, 3, stride=(2, 1))
        )
        self.attn = nn.Sequential(
            nn.Linear(self.w*in_channels//2,512),
            nn.ReLU(),
            nn.Linear(512,self.w*self.w*2),
        )
        self.f_conv = nn.Sequential(
            conv_func(in_channels, in_channels, 3, stride=(2, 1)),
            conv_func(in_channels, in_channels, 3, stride=(2, 1))
        )
        # self.attention = conv_func(in_channels, len(neighbors), (len(neighbors),3), stride=(len(neighbors), 1),padding=(0, dilation * (3 - 1) // 2))
    def forward(self, x):
        b = x.size(0)
        af = self.attn_conv(x).view((b,-1))
        att = self.attn(af).view((b,self.w*2, self.w)).softmax(dim=-1)
        ff = torch.bmm(self.f_conv(self.rescale(x)).squeeze(dim=-2),att)
        return ff.unsqueeze(dim=-2)

if __name__ == '__main__':
    in_channels = 256
    x = torch.randn([4,in_channels,4,15])
    nga = ScaleNet(in_channels)
    o = nga(x)
    print(o.shape)
    # x = torch.randn([2,steps,128])
    # y = torch.randn([2,steps,128])
    # ms = dms(x,x)
    # print(ms)
    # print(dms.sim(x,x))
    # torch.mm(x[:,0,:],y[:,0,:].t())
