import torch
import torch.nn.functional as F
from torch import nn
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, embed_dim=512, num_heads=1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, q,rois):
        #q,k,v [roi_num_per_img, img_num, embed_dim]
        return self.multihead_attn(q,rois,rois)[0]


detr = DETR()
rois = torch.randn([64,1,512])
q = torch.randn([10,1,512])
out = detr(q,rois)
print(out.shape)

embedding = nn.Embedding(10, 3)
# weight = embedding.weight.clone().detach()
weight = embedding.weight
print(weight)

# pro = nn.Linear(256,128)
# x = torch.randn([15,100,256])
# print(pro(x).shape)