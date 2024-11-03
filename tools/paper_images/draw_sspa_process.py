import torch
import os
import cv2
import numpy as np

# image_name = '中国邮政储蓄银行/Google_0015.jpeg'
# query = '银行'
# folder = 'Log/cn_r50_bilinear_fcoscount_pt_b64/finetune/inference_100k/chinese_collect'
# predictions = torch.load(os.path.join(folder, 'chinese_collect_predictions.pth'))

image_name = 'train_ReCTS_002689.jpg'
query = '饺子'
folder = 'Log/cn_r50_bilinear_fcoscount_pt_b64/finetune/inference_100k/rects_test'
predictions = torch.load(os.path.join(folder, 'rects_test_predictions.pth'))

save_folder = 'tools/paper_images/sspa_process'
texts = predictions[0].get_field("texts").tolist()
words_embedding_nor = predictions[0].get_field("words_embedding_nor")[texts.index(query)].view(1,-1)

# colors = [(201,207,),(),(),()]
# import ipdb; ipdb.set_trace()
def draw_touming(img1,img2):
    alpha = 0.6
    beta = 1-alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha,img2, beta, gamma)
    return img_add
for image_id, prediction in enumerate(predictions):
    img_path = str(prediction.get_field("path"))
    if image_name not in img_path:
        continue
    image = cv2.imread(img_path)
    
    sr_num = prediction.get_field("scores").size(0)
    polys = prediction.get_field("polys")
    scale = prediction.get_field("scale")
    lens = prediction.get_field("query_lens").data.cpu().numpy().reshape([-1])
    polys[:,::2] *= scale[0]
    polys[:,1::2] *= scale[1]
    polys = polys.data.cpu().numpy().reshape([-1,14,2]).astype(np.int32)

    image_sp = image.copy()
    mask = np.zeros_like(image_sp)
    print(sr_num)
    # sr_num=1
    for poly in polys[:sr_num]:
        cv2.drawContours(mask, [poly], -1, thickness=-1, color=(0,0,255))
    image_sp = draw_touming(image_sp, mask)

    image_ssp = image_sp.copy()

    # color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    # temp = np.array([polys[0,0], polys[2,0], polys[2,-1], polys[0,-1]]).reshape([-1,2])
    # cv2.drawContours(image_ssp, [temp], -1, thickness=2, color=color)

    for poly, len in zip(polys[sr_num:], lens[sr_num:]):
        # if len > 2: continue
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        cv2.drawContours(image_ssp, [poly], -1, thickness=2, color=color)

    img_embedding = prediction.get_field("imgs_embedding_nor") #[:sr_num]
    similarity =  words_embedding_nor.mm(img_embedding.t())
    max_scores, max_idx = similarity.max(dim=1)
    max_idx = max_idx.data.cpu().numpy()[0]
    poly = polys[max_idx]

    image_retrieved = image_sp.copy()
    cv2.drawContours(image_ssp, [poly], -1, thickness=2, color=(0,255,0))
    cv2.drawContours(image_retrieved, [poly], -1, thickness=2, color=(0,255,0))

    

    cv2.imwrite(os.path.join(save_folder, 'image_sp.jpg'), image_sp)
    cv2.imwrite(os.path.join(save_folder, 'image_ssp.jpg'), image_ssp)
    cv2.imwrite(os.path.join(save_folder, 'image_retrieved.jpg'), image_retrieved)

    