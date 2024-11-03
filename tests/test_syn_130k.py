import os
import io
import json
from pycocotools.coco import COCO
import contextlib
import cv2
import numpy as np
# img_floder = '/home/ymk-wh/workspace/datasets/retrieval_chinese/syntext/syn_130k_images/'
# def show_sample(data, save_folder):
#     img_info, anns = data
#     file_name = img_info['file_name']
#     img_path = os.path.join(img_floder, file_name)
#     img = cv2.imread(img_path)
#     for ann in anns:
#         poly = ann['segmentation']
#         poly = np.array(poly).reshape([-1,4,2])
#         cv2.drawContours(img, poly, -1, thickness=2, color=(0,255,0))
#     cv2.imwrite(os.path.join(save_folder, file_name), img)

# json_file = "/home/ymk-wh/workspace/datasets/retrieval_chinese/syntext/annotations/syn130k_word_train.json"
# with contextlib.redirect_stdout(io.StringIO()):
#     coco_api = COCO(json_file)
# cat_ids = sorted(coco_api.getCatIds())
# cats = coco_api.loadCats(cat_ids)
# img_ids = sorted(coco_api.imgs.keys())
# imgs = coco_api.loadImgs(img_ids)
# anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
# ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
# assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(json_file)
# imgs_anns = list(zip(imgs, anns))

# show_sample(imgs_anns[0], 'temp')
# import ipdb; ipdb.set_trace()
# print(load_dict)
char_json_file = "/home/ymk-wh/workspace/datasets/retrieval_chinese/syntext/annotations/syn130k_char_train.json"
with open(char_json_file) as f:
    dict_ = json.load(f)
print(dict_)