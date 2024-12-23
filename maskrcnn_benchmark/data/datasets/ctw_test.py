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
from .augs import PSSAugmentation,TestAugmentation,SythAugmentation,RetrievalAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import numpy as np
import cv2
import scipy.io as scio
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
class CTWRetrieval(object):
    def __init__(self, path, is_training = True):
        # assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        self.all_texts = []
        self.parse_data(path)

    def parse_data(self,path):
        # dataFile = os.path.join(path,"data.mat")
        gtPath = os.path.join(path,"gts")
        gtFiles = os.listdir(gtPath)
        str_queries = [word.strip() for word in open(os.path.join(path,'queries.txt'),'r').readlines()]
        y_trues = np.zeros([len(str_queries),len(gtFiles)])
        for i,gtFile in enumerate(gtFiles):
            gt = os.path.join(gtPath,gtFile)
            words = [word.strip() for word in open(gt,'r').readlines()]
            for j,word in enumerate(words):
                # print(word)
                if word not in str_queries:
                    continue
                y_trues[str_queries.index(word),i]=1
        # print(y_trues.sum())
        # print(str_queries)
        self.img_lists = [os.path.join(path,"images",imgName.replace('.txt','.jpg')) for imgName in gtFiles]
        # print(gtFiles[0],self.img_lists[0])
        # print(str_queries)
        self.str_queries = str_queries
        self.y_trues = y_trues
        # print(self.y_trues.shape,len(self.str_queries))


    def len(self):
        return len(self.img_lists)
    def getitem(self,index):
        return self.img_lists[index], self.str_queries, self.y_trues
NUM_POINT=7
class CTWRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True, augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = CTWRetrieval(data_dir, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            # print("get item")
            path, queries, trues = self.dataset.getitem(idx)
            img = imread(path)
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags = self.augment(img, None, None)
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist = BoxList([], image.size, mode="xyxy")
            boxlist.add_field('retrieval_trues',trues)
            boxlist.add_field('texts',queries)
            # boxlist.add_field("y_trues",trues)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
        else:
            path, queries, trues = self.dataset.getitem(idx)
            # print(polys)
            img = imread(path,mode="RGB")
            ori_h, ori_w, _ = img.shape
            aug_img, polys, tags = self.augment(img, None, None)
            test_h, test_w, _ = aug_img.shape
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList([[0,0,0,0]], image.size, mode="xyxy")
            boxlist.add_field('retrieval_trues',trues)
            boxlist.add_field('texts',np.array(queries))
            boxlist.add_field('scale',np.array([ori_w/test_w, ori_h/test_h]))
            boxlist.add_field('path',np.array(path))
            boxlist.add_field("y_trues",trues)
            # boxlist.add_field('test_texts',self.dataset.all_texts)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx

    def __len__(self):
        return self.dataset.len()

    def get_img_info(self, idx):
        # print("get info")
        if self.is_train:
            return {"path":"none", "height": 768, "width": 1280}
        path, _, _ = self.dataset.getitem(idx)
        size = Image.open(path).size
        # size = [1280,768]
        return {"path":path, "height": size[1], "width": size[0]}
