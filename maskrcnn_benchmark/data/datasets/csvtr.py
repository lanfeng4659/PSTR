import torch
import glob
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
from maskrcnn_benchmark.data.datasets.augs import PSSAugmentation,TestAugmentation,SythAugmentation,RetrievalAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import numpy as np
import cv2
import scipy.io as scio
from maskrcnn_benchmark.data.datasets.chinese_utils import load_chars
class CSVTR(object):
    def __init__(self, path, is_training = False):
        # assert is_training==True
        self.name = 'csvtr'
        self.is_training = is_training
        self.difficult_label = '###'
        self.all_texts = []
        self.parse_data(path)

    def parse_data(self,gt_path):
        self.str_queries_full = []
        self.img_lists = []
        # queries = [v for v in os.listdir(gt_path) if '.txt' not in v]
        queries = []
        img_num = 0
        for name in os.listdir(os.path.join(gt_path, 'v1')):
            if '.txt' in name: continue
            queries.append(name)
            img_num += len(os.listdir(os.path.join(gt_path, 'v1', name)))
        for name in  os.listdir(os.path.join(gt_path, 'v2')):
            if '.txt' in name: continue
            queries.append(name)
            img_num += len(os.listdir(os.path.join(gt_path, 'v2', name)))
        query_num = len(queries)

        self.y_trues_full = np.zeros([query_num,img_num])
        cur_idx = 0
        # print(img_num)
        for idx,query in enumerate(queries):
            self.str_queries_full.append(query)
            version = 'v1' if query in os.listdir(os.path.join(gt_path, 'v1')) else 'v2'
            self.img_lists.extend([os.path.join(gt_path, version, query, img) for img in os.listdir(os.path.join(gt_path, version, query))])
            query_img_num = len(os.listdir(os.path.join(gt_path, version, query)))
            self.y_trues_full[idx,cur_idx:cur_idx+query_img_num] = 1
            cur_idx+=query_img_num
        # print(len(self.img_lists))
        self.str_queries_partial = [v.strip() for v in open(os.path.join(gt_path, 'csvtr_partial_query_v1v2.txt')).readlines()]
        self.y_trues_partial = np.zeros([len(self.str_queries_partial),len(self.img_lists)])
        for idx,img_path in enumerate(self.img_lists):
            name = img_path.split('/')[-2]
            for query in self.str_queries_partial:
                if query in name:
                    self.y_trues_partial[self.str_queries_partial.index(query), idx]=1
        self.str_queries = self.str_queries_full + self.str_queries_partial
        self.y_trues = np.concatenate((self.y_trues_full, self.y_trues_partial))
    # def parse_data(self,gt_path):
    #     self.str_queries = [v.strip() for v in open(os.path.join(gt_path, 'v1_query_partial.txt')).readlines()]
    #     self.img_lists = []
    #     for name in os.listdir(gt_path):
    #         if '.txt' in name:
    #             continue
    #         self.img_lists.extend([os.path.join(gt_path, name, v) for v in os.listdir(os.path.join(gt_path, name))])
    #     query_num = len(self.str_queries)
    #     img_num = len(self.img_lists)
    #     self.y_trues = np.zeros([query_num,img_num])
    #     for idx,img_path in enumerate(self.img_lists):
    #         name = img_path.split('/')[-2]
    #         for query in self.str_queries:
    #             if query in name:
    #                 self.y_trues[self.str_queries.index(query), idx]=1
        
    def len(self):
        return len(self.img_lists)
    def getitem(self,index):
        return self.img_lists[index], self.str_queries, self.y_trues
NUM_POINT=7
class CSVTRDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True, augment=None):
        super().__init__()
        assert is_train==False
        self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = CSVTR(data_dir, is_train)

    def __getitem__(self, idx):

        path, queries, trues = self.dataset.getitem(idx)
        # print(path)
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
    # data_dir = "/root/datasets/ic15_end2end"
    # ic15_dataset = IC15(data_dir)
    # image, boxlist, idx = ic15_dataset[0]
    # import ipdb; ipdb.set_trace()
    filter = ['/home/ymk-wh/workspace/datasets/retrieval_chinese/chinese_collect/v2/派乐汉堡/Google_0073.jpeg']
    data = CSVTR('/home/ymk-wh/workspace/datasets/retrieval_chinese/chinese_collect', False) 
    for path in data.img_lists:
        if path in filter: continue
        Image.open(path)

