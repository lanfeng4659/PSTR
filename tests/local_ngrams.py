import editdistance
import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
x = torch.randn([4,15,128])
y = torch.randn([3,15,128])
class NGramsAtt(nn.Module):
    
    def __init__(self,in_channels, neighbors=[1,3,5,7]):
        super(NGramsAtt, self).__init__()
        conv_func = nn.Conv2d
        dilation=1
        self.convs = []
        for i, neigh in enumerate(neighbors):
            self.convs.append(
                nn.Sequential(conv_func(in_channels, in_channels, (1,neigh), stride=(1, 1), padding=(0,dilation * (neigh - 1) // 2)),
                              nn.BatchNorm2d(in_channels), nn.ReLU())
                )
        self.attention = conv_func(in_channels, len(neighbors), (len(neighbors),3), stride=(len(neighbors), 1),padding=(0, dilation * (3 - 1) // 2))
    def forward(self, x):
        x = x.permute((0,2,1))[:,:,None] # [b,c,1,T]
        fs = []
        for conv in self.convs:
            fs.append(conv(x))
        fm = torch.cat(fs, dim=2) # [b,c,len,T]
        att_v = F.softmax(self.attention(fm),dim=1) # [b,len,1,T]
        att_f = (fm*att_v.permute(0,2,1,3)).sum(dim=2) # [b,c,T]
        return att_f.permute(0,2,1)

if __name__ == '__main__':
    in_channels = 128
    nga = NGramsAtt(in_channels)
    o = nga(x)
    print(o.shape)
    # x = torch.randn([2,steps,128])
    # y = torch.randn([2,steps,128])
    # ms = dms(x,x)
    # print(ms)
    # print(dms.sim(x,x))
    # torch.mm(x[:,0,:],y[:,0,:].t())
