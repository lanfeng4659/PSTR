import editdistance
import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
x = torch.randn([4,15,128])
y = torch.randn([3,15,128])
class Verify(nn.Module):
    
    def __init__(self,in_channels, sim_thred=0.5):
        super(Verify, self).__init__()
        self.sim_thred = sim_thred
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def loss(self, preds, targets):
        targets = (targets > self.sim_thred).long()
        # loss = F.cross_entropy(preds.permute(0,2,1), targets, reduction='none')
        loss = F.cross_entropy(preds.permute(0,2,1), targets)
        return loss
    # def inference(self, preds):
    #     return 
    def forward(self, x, y, targets):
        x = x.view(x.size(0), 1, -1)
        y = y.view(1, y.size(0), -1)

        c = self.classifier(x-y)
        loss = self.loss(c, targets)
        return loss

if __name__ == '__main__':
    in_channels = 128
    targets = F.sigmoid(torch.randn(x.size(0), y.size(0)))
    # print(targets.shape)
    verify = Verify(x.size(1)*x.size(2))
    loss = verify(x, y, targets)
    print(loss)
    # x = torch.randn([2,steps,128])
    # y = torch.randn([2,steps,128])
    # ms = dms(x,x)
    # print(ms)
    # print(dms.sim(x,x))
    # torch.mm(x[:,0,:],y[:,0,:].t())
