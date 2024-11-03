import os
import cv2
import torch
import numpy as np
from maskrcnn_benchmark.layers.arbitrary_roi_align import ArbitraryROIAlign
size = (128,400)
def extract_poly(image, poly):
    num_point = poly.shape[1]//2
    pooler = ArbitraryROIAlign(size, 5*2, [0,0]).cuda()
    image = torch.tensor(image).permute((2,0,1))[None].float().cuda()
    b,c,h,w = image.shape
    wh=torch.FloatTensor([w,h])[None,None,:]
    poly = torch.tensor(poly).float()/wh
    poly = poly.cuda()
    idx = list(range(num_point)) + list(range(num_point, num_point*2))[::-1]
    index_in_which_img = torch.tensor([0]).float()
    # print(image, poly[:,idx], index_in_which_img)
    roi = pooler(image, poly[:,idx], index_in_which_img)
    return roi[0].permute((1,2,0)).data.cpu().numpy().astype(np.uint8)
def draw_attention(image, att):
    att = cv2.resize(att, (size[1], size[0]))
    att = (att*255).astype(np.uint8)
    att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    print(att.shape, image.shape)
    att_image = cv2.addWeighted(att, 0.4, image, 0.6, 0.)
    return att_image
folder = 'tools/boundary_images'
# poly = [[46,396],[1176,277],[1198,465],[39,553]]
poly = [[19,376],[414,172],[787,79],[1266,139],[1666,282],  [1445,492],[1160,323],[782,276],[464,336],[112,524]]
att = np.zeros([1,15])
att[0,10] = 1.0
# att[0,1] = 1.0
# np.array
poly = np.array(poly).reshape([1,-1,2]).astype(np.int32)



image = cv2.imread(os.path.join(folder,"en.png"))
roi = extract_poly(image, poly)
att_roi = draw_attention(roi, att)
cv2.drawContours(image, poly, -1, thickness=2, color=(255,0,0))
cv2.imwrite(os.path.join(folder, 'en_save.jpg'), image)
cv2.imwrite(os.path.join(folder, 'en_roi.jpg'), roi)
cv2.imwrite(os.path.join(folder, 'en_att_roi.jpg'), att_roi)