
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import numpy as np
import argparse
import os
import cv2
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from tqdm import tqdm
from maskrcnn_benchmark.modeling.poolers import Pooler, PolyPooler, PolyPoolerTextLenSensitive
#   PIXEL_MEAN: [103.53, 116.28, 123.675]
#   PIXEL_STD: [57.375, 57.12, 58.395]

def denormalize(image):
    std_ = torch.tensor([[57.375, 57.12, 58.395]]).to(image.device)
    mean_ = torch.tensor([[103.53, 116.28, 123.675]]).to(image.device)
    image.mul_(std_).add_(mean_)
    return image
# cfg_path = "./configs/SiamRPN.yaml"
cfg_path = "yundao_configs/cn_r50_bilinear_fcoscount_contrastive/pretrain.yaml"
cfg.merge_from_file(cfg_path)
data_loaders = make_data_loader(cfg, is_train=True, is_distributed=False)
pooler = PolyPoolerTextLenSensitive(
                num_points=7,
                output_size_list=[(4,8),(4,12),(4,16),(4,20)],
                lens_area = [4,8,12,1000],
                scales=(0.25, 0.125, 0.0625),
                sampling_ratio=1,
                canonical_scale=1.0,
                mode='align')
for i,(image, boxlist, idx) in enumerate(tqdm(data_loaders)):
    image_tensor = image.tensors
    inputs = [image_tensor[:,:,::4*2**i,::4*2**i] for i in range(3)]
    results = pooler(inputs, boxlist)
    # print(len(results))
    for r in results:
        print(r.shape)
    # image = image_tensor.permute(0,2,3,1).float()
    # image_de = denormalize(image).data.cpu().numpy().astype(np.uint8)
    # for j in range(len(boxlist)):
    #     image_per = image_de[j].copy()
    #     # print(image_per.max(),image_per.min())
    #     boxes = boxlist[j]
    #     # if boxes.get_field("texts")==None:
    #     assert len(boxes.get_field("texts"))==boxes.bbox.size(0), (boxes.get_field("texts"),boxes)
