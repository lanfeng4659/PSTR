
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import numpy as np
import argparse
import os
import cv2
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from tqdm import tqdm
#   PIXEL_MEAN: [103.53, 116.28, 123.675]
#   PIXEL_STD: [57.375, 57.12, 58.395]
save_path = "./visul"
if os.path.exists(save_path)==False:
    os.makedirs(save_path)
def compute_centerness_targets(self, reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                    (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)
def visual_centerness():
    return None
def denormalize(image):
    std_ = torch.tensor([[57.375, 57.12, 58.395]]).to(image.device)
    mean_ = torch.tensor([[103.53, 116.28, 123.675]]).to(image.device)
    image.mul_(std_).add_(mean_)
    return image
cfg_path = "/home/jfwu/seg/projects/bezier_curve_text_spotting-master/configs/train.yaml"
cfg.merge_from_file(cfg_path)
data_loaders = make_data_loader(cfg, is_train=True, is_distributed=False)
for i,(image, boxlist, idx) in enumerate(tqdm(data_loaders)):
    # continue
    image_tensor = image.tensors.permute(0,2,3,1).float()
    print(image_tensor.shape)
    # image_de = image_tensor.data.cpu().numpy().astype(np.uint8)

    image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)
    for j in range(len(boxlist)):
        image_per = image_de[j].copy()
        print(image_per.max(),image_per.min())
        boxes = boxlist[j]
        bbox = boxes.bbox.data.cpu().numpy().astype(np.int32)
        polys = boxes.get_field("polys").data.cpu().numpy().astype(np.int32)
        # print(boxes.fields())
        bbox = bbox[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2])
        # print(len(polys), bbox.shape)
        assert len(polys) == len(bbox)
        image_per = cv2.drawContours(image_per, bbox, -1, color=(0,255,0), thickness=1)
        for poly in polys:
            poly = poly.reshape([1,-1,2]).astype(np.int32)
            # print(image_per.min())
            image_per = cv2.drawContours(image_per, poly, -1, color=(255,0,0), thickness=1)
        image_path = os.path.join(save_path, "image_{}.jpg".format(np.random.randint(0,10000)))
        cv2.imwrite(image_path, image_per)
        print(image_path,image_per.shape, bbox.shape,len(polys))
print("done")