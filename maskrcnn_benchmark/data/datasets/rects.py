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
from maskrcnn_benchmark.data.datasets.augs import PSSAugmentation,TestAugmentation,SythAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import numpy as np
import cv2
from xml.etree.ElementTree import ElementTree
from maskrcnn_benchmark.data.datasets.chinese_utils import load_chars
from maskrcnn_benchmark.data.datasets.utils import generate_partial_proposals_labels
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    # print(chars)
    # exit()
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
def load_ann(gt_paths,chars):
    res = []
    idxs = []
    for gt in gt_paths:
        # if len(res)>9:
        #     continue
        # gt = unicode(gt, 'utf-8')#gt.decode('utf-8')
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['texts'] = []
        item['gt_path'] = gt
        # print(gt)
        reader = codecs.open(gt,encoding='utf-8').readlines()
        # reader = open(gt).readlines()
        for line in reader:
            parts = line.strip().split(';')
            label = parts[-1]
            # print(label)
            # print("before:",label)
            label = filter_word(label,chars=chars)
            # print("after:",label)
            if len(label)<2:
                continue
            if label == '###':
                continue
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[1:9]))
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label)
        if len(item['polys'])==0:
            continue
        # print(len(res))
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    return res
class ReCTS(object):
    def __init__(self, path, is_training = True):
        # assert is_training==True
        self.name = 'rects'
        self.is_training = is_training
        self.difficult_label = '###'
        # self.chars = np.load(os.path.join(path, 'chars.npy')).tolist()
        # self.chars = np.load("/workspace/wanghao/projects/Pytorch-yolo-phoc/selected_chars.npy").tolist()
        self.chars = load_chars()
        self.floder = path
        self.path = os.path.join(path, 'train') if self.is_training else os.path.join(path, 'test')
        if self.is_training:
            self.generate_information(self.path)
        else:
            self.parse_data(self.path)
    def generate_information(self, path):
        if self.is_training:
            self.gt_list = glob.glob(os.path.join(path, "labels", '*.txt'))
            self.samples = load_ann(self.gt_list,self.chars)
            print(len(self.samples))
    def parse_data(self,gt_path):
        def get_words(gt):
            words = []
            reader = codecs.open(gt,encoding='utf-8').readlines()
            for line in reader:
                parts = line.strip().split(';')
                label = parts[-1]
                label = filter_word(label,chars=self.chars)
                words.append(label)
            return words
        
        
        # self.str_queries = [v.strip() for v in open(os.path.join(self.floder, 'query_full.txt')).readlines()]
        # print(self.str_queries)
        self.img_lists = glob.glob(os.path.join(self.floder, 'test/images/*.jpg'))
        
        self.str_queries_full = [v.strip() for v in open(os.path.join(self.floder, 'query_fm.txt')).readlines()]
        self.str_queries_partial = [v.strip() for v in open(os.path.join(self.floder, 'query_ctp.txt')).readlines()]
        self.str_queries = self.str_queries_full + self.str_queries_partial
        # self.str_queries = self.str_queries_partial
        query_num = len(self.str_queries)
        img_num = len(self.img_lists)
        self.y_trues = np.zeros([query_num,img_num])
        cur_idx = 0
        for idx,img_path in enumerate(self.img_lists):
            words = get_words(img_path.replace('images', 'labels').replace('.jpg', '.txt'))
            for word in words:
                for qi, query in enumerate(self.str_queries):
                    if query in word:
                    # if query == word:
                        self.y_trues[qi, idx]=1
        # print(self.y_trues.sum(axis=1).min())
        self.samples = self.img_lists
    def len(self):
        return len(self.samples)
    def getitem(self,index):
        # print(self.len())
        if self.is_training:
            sample = self.samples[index]
            gt_path = sample["gt_path"]
            img_path = gt_path.replace('labels','images').replace(".txt",".jpg")
            return img_path, sample['polys'].copy(), sample['texts'].copy()
        else:
            return self.img_lists[index], self.str_queries, self.y_trues

NUM_POINT=7
class ReCTSDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = ReCTS(data_dir, is_train)

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
    data_dir = "/home/ymk-wh/workspace/datasets/retrieval_chinese/ReCTS"
    # dataset = ReCTSDataset(data_dir, augment="PSSAugmentation")
    dataset = ReCTS(data_dir, is_training=False)
    # text_dict = {}
    # for i in range(dataset.len()):
    #     for text in dataset.getitem(i)[-1]:
    #         textlen = len(text)
    #         if textlen not in text_dict.keys():
    #             text_dict[textlen] = 1
    #         else:
    #             text_dict[textlen] += 1
    # print(text_dict)
        

