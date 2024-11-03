import editdistance
import argparse
import time
import torch
from torch import nn
x = torch.randn([4,15,128])
y = torch.randn([3,15,128])
class CTCPredictor(nn.Module):
    def __init__(self, in_channels, class_num):
        super(CTCPredictor, self).__init__()
    def norm(self,x):
        
    def forward(self, x, embeddings, targets=None):
        x = self.clf(x)
        if self.training:
            x = F.log_softmax(x, dim=-1).permute(1,0,2)
            # print(targets.shape)
            input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
            target_lengths, targets_sum = self.prepare_targets(targets)
            # print(x.shape,targets.shape,target_lengths.shape)
            # loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True) / 10
            loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True)
            # loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True)/2
            return loss
        return x
    def prepare_targets(self, targets):
        target_lengths = (targets != self.class_num - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        sum_targets = torch.cat(sum_targets)
        return target_lengths, sum_targets

if __name__ == '__main__':
    steps = 5
    dms = DynamicMaxSimilarity(steps)
    x = torch.randn([2,steps,128])
    y = torch.randn([2,steps,128])
    ms = dms(x,x)
    print(ms)
    # print(dms.sim(x,x))
    # torch.mm(x[:,0,:],y[:,0,:].t())
