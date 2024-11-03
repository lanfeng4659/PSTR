import torch
import os
try:
  import moxing as mox
  mox.file.shift('os', 'mox')
  run_on_remote = True
except:
  run_on_remote = False
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
# import unicode
from maskrcnn_benchmark.data.datasets.augs import PSSAugmentation,TestAugmentation,SythAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import io
import json
from pycocotools.coco import COCO
import contextlib
import cv2
from xml.etree.ElementTree import ElementTree
from maskrcnn_benchmark.data.datasets.chinese_utils import load_chars
from maskrcnn_benchmark.data.datasets.utils import generate_partial_proposals_labels
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
def load_ann(path, chars):
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(path)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    
    res = []
    idxs = []
    for img_info, anns in list(zip(imgs, anns)):
        # gt = unicode(gt, 'utf-8')#gt.decode('utf-8')
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['texts'] = []
        item['file_name'] = img_info['file_name']

        for ann in anns:
            label = ann['rec']
            label = filter_word(label,chars=chars)
            # print("after:",label)
            if len(label)<2:
                continue
            if label == '###':
                continue
            assert len(ann['segmentation'][0])==8, ann['segmentation'][0]
            x1, y1, x2, y2, x3, y3, x4, y4 =  ann['segmentation'][0]
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label)
        if len(item['polys'])==0:
            continue
        # print(len(res))
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    return res
class SynthtextChinese130K(object):
    def __init__(self, path, is_training = True):
        assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        # self.chars = np.load(os.path.join(path, 'chars.npy')).tolist()
        # self.chars = np.load("/workspace/wanghao/projects/Pytorch-yolo-phoc/selected_chars.npy").tolist()
        self.chars = load_chars()
        self.generate_information(path)
        self.path = path
        
    def generate_information(self, path):
        if self.is_training:
            # self.image_floder = os.path.join(path, 'images')
            self.gt_path = os.path.join(path, 'annotations/syn130k_word_train.json')
            self.samples = load_ann(self.gt_path,self.chars)
            print(len(self.samples))
    def len(self):
        return len(self.samples)
    def getitem(self,index):
        # print(self.len())
        sample = self.samples[index]
        img_path = os.path.join(self.path, 'syn_130k_images', sample["file_name"])
        # img_path = gt_path.replace("gts","images").replace(".txt",".jpg")
        return img_path, sample['polys'].copy(), sample['texts'].copy()
NUM_POINT=7
class SynthtextChinese130KDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = SynthtextChinese130K(data_dir, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            # print("get item")
            path, polys, texts = self.dataset.getitem(idx)
            img = imread(path)
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags = self.augment(img, polys, texts)
            boxes = []#[[np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])] for poly in polys]
            # # boxes = np.array(boxes).reshape([-1,4])
            # order_polys = []
            # boundarys = []
            for poly in polys:
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
                # boundarys.append(pts_expand)
                # order_polys.append(get_ordered_polys(poly))
                # cv2.drawContours(aug_img, pts_expand.reshape([1,-1,2]).astype(np.int32),-1,color=(255,0,0),thickness=1)
            # cv2.imwrite(os.path.join('vis',os.path.basename(path)), aug_img[:,:,(2,1,0)])
            boxes = np.array(boxes).reshape([-1,4])
            # order_polys = np.array(order_polys).reshape([-1,8])
            # boundarys = np.array(boundarys).reshape([-1,NUM_POINT*4])
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            # boxlist.add_field('polys',torch.tensor(order_polys))
            # boxlist.add_field('boundarys',torch.tensor(boundarys))
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            boxlist.add_field('texts',tags)
            polys = [self.expand_point(ps) for ps in polys]
            boxlist.add_field('polys',torch.as_tensor(polys, dtype=torch.float32))
            partial_polys = []
            partial_texts = []
            bag_ids = [len(polys)]

            for poly, text in zip(polys, tags):
                outs = generate_partial_proposals_labels(np.array(poly), text)
                bag_ids.append(bag_ids[-1]+len(outs))
                partial_polys.extend([v[0] for v in outs])
                partial_texts.extend([v[1] for v in outs])
            # visual_partial_text(aug_img, partial_polys, partial_texts)
            # print(bag_ids, tags, partial_texts)
            boxlist.add_field('bag_ids', bag_ids)
            boxlist.add_field('partial_texts',partial_texts)
            boxlist.add_field('partial_polys',torch.as_tensor(partial_polys, dtype=torch.float32).view([-1,28]))

            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
        else:
            path, _, _ = self.dataset.getitem(idx)
            img = imread(path)
            aug_img, _, _ = self.augment(img)
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist=None
            if self.transforms:
                image,_ = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, None, idx

    def __len__(self):
        return self.dataset.len()

    def expand_point(self, poly):
        poly = np.array(poly).reshape(-1, 2)
        up_x = np.linspace(poly[0, 0], poly[1, 0], NUM_POINT)
        up_y = np.linspace(poly[0, 1], poly[1, 1], NUM_POINT)
        up = np.stack((up_x, up_y), axis=1)
        do_x = np.linspace(poly[2, 0], poly[3, 0], NUM_POINT)
        do_y = np.linspace(poly[2, 1], poly[3, 1], NUM_POINT)
        do = np.stack((do_x, do_y), axis=1)
        poly_expand = np.concatenate((up, do), axis=0)
        return poly_expand.reshape(-1).tolist()

    def get_img_info(self, idx):
        # print("get info")
        if self.is_train:
            return {"path":"none", "height": 768, "width": 1280}
        path, _, _ = self.dataset.getitem(idx)
        size = Image.open(path).size
        # size = [1280,768]
        return {"path":path, "height": size[1], "width": size[0]}


if __name__ == "__main__":
    data_dir = "/home/ymk-wh/workspace/datasets/retrieval_chinese/syntext"
    dataset = SynthtextChinese130KDataset(data_dir, augment="PSSAugmentation")
    image, boxlist, idx = dataset[0]
    print(image, boxlist)
    # import ipdb; ipdb.set_trace()

