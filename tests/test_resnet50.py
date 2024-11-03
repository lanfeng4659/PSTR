# import torch
from torchvision.models import resnet50
# from thop import profile
model = resnet50()
# input = torch.randn(1, 3, 125, 125)
# flops, params = profile(model, inputs=(input, ))
# print(flops/2)
# print(params)
from ptflops import get_model_complexity_info
flops, params = get_model_complexity_info(model, (3, 125, 125), as_strings=True, print_per_layer_stat=True)
print(flops,params)