import torch
from maskrcnn_benchmark.modeling.one_stage_head.align.align import PyramidFeatures

net = PyramidFeatures([2,3])
feats = torch.randn([4,15,256])
net(feats)