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
from .augs import PSSAugmentation,TestAugmentation,SythAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import numpy as np
import cv2
from xml.etree.ElementTree import ElementTree
from maskrcnn_benchmark.data.datasets.utils import generate_partial_proposals_labels
def load_ann(gt_paths):
    res = []
    for gt in gt_paths:
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['texts'] = []
        item['paths'] = gt
        reader = open(gt).readlines()
        for line in reader:
            parts = line.strip().split(',')
            label = parts[-1]
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label)
            if label == '###':
                item['tags'].append(True)
            else:
                item['tags'].append(False)
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['tags'] = np.array(item['tags'], dtype=np.bool)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    #     print('read',item['polys'])
    # exit()
    return res
class Synthtext90k(object):
    def __init__(self, path, is_training = True):
        assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        self.generate_information(path)
    def generate_information(self, path):
        if self.is_training:
            self.image_floder = os.path.join(path, 'images')
            self.gt_floder = os.path.join(path, 'annotations')
            # self.image_list = os.listdir(self.image_floder)
            self.gt_list    = os.listdir(self.gt_floder)
    def parse_xml_file(self,gt_path):
        texts = []
        polys = []
        tree = ElementTree()
        tree.parse(gt_path)
        for object_ in tree.findall("object"):
        # print(objects)
            text = object_.find("name").text
            xmin = int(object_.find("bndbox/xmin").text)
            ymin = int(object_.find("bndbox/ymin").text)
            xmax = int(object_.find("bndbox/xmax").text)
            ymax = int(object_.find("bndbox/ymax").text)
            texts.append(text)
            polys.append([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
        return np.array(polys, dtype=np.float32),np.array(texts, dtype=np.str)


    def len(self):
        return len(self.gt_list)
    def getitem(self,index):
        # print(self.len())
        gt_name = self.gt_list[index]
        gt_path = os.path.join(self.gt_floder, gt_name)
        img_path = os.path.join(self.image_floder, gt_name.replace('.xml','.jpg'))
        polys, texts = self.parse_xml_file(gt_path)
        # print(texts)
        # img = cv2.imread(img_path)
        # print(self.image_path_list[index])
        return img_path, polys, texts
NUM_POINT=7
class SynthText90kDateset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = Synthtext90k(data_dir, is_train)

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
            boxlist.add_field('polys',torch.as_tensor([self.expand_point(ps) for ps in polys], dtype=torch.float32))
            
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
    data_dir = "/root/datasets/ic15_end2end"
    ic15_dataset = IC15(data_dir)
    image, boxlist, idx = ic15_dataset[0]
    import ipdb; ipdb.set_trace()

