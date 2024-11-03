import torch
from torch import nn
from torch.nn import functional as F
import cv2
from maskrcnn_benchmark.layers import ROIAlign, BezierAlign, ROIAlignAdaptive
import numpy as np
# def convert_rois(rois):
#     new_rois = []
#     for roi in rois:
#         if (roi[4] - roi[2])/(roi[3] - roi[1]) > 3:
#             new_rois.append(roi[(0,2,3,)])
#     return torch.cat([[] for roi in rois])
pooler = ROIAlignAdaptive([16,128], spatial_scale=1, sampling_ratio=1)
polys = [[1175,188,1214,370,1182,379,1143,197],
[1200,281,1236,274,1256,363,1220,371],
[1178,173,1212,165,1234,264,1200,272]]
img = cv2.imread("/workspace/wanghao/datasets/icdar2015/train_images/img_88.jpg")
h,w,c = img.shape
print(img.shape)
img = torch.tensor(img).permute((2,0,1))[None].cuda().float()

poly = polys[0]
minx,miny,maxx,maxy = min(poly[::2]), min(poly[1::2]), max(poly[::2]), max(poly[1::2])
roi = torch.tensor([0,minx,miny,maxx,maxy]).reshape([1,-1]).cuda().float()
print(img.shape, roi.shape)
crop = pooler(img, roi)
print(crop.shape)
# print(crop)
crop_np = crop[0].permute(1,2,0).data.cpu().numpy()
print(crop_np.shape)
cv2.imwrite("save.jpg", crop_np.astype(np.int32))
