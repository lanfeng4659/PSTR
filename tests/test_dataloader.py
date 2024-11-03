
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import numpy as np
import argparse
import os
import cv2
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from tqdm import tqdm
from maskrcnn_benchmark.utils.text_util import TextGenerator
text_generator = TextGenerator()
def get_augmented_words(texts):
    word_texts = texts.copy()
    words = [torch.tensor(text_generator.label_map(text.lower())).long() for text in word_texts]
    assert len(word_texts)==len(words), print(len(word_texts),len(words))
    return word_texts, words
def split_full_partial(proposals):
    max_nums = sum([p.get_field("bag_ids")[-1] for p in proposals])
    idxs = torch.tensor([-1]*max_nums)
    offset = 0
    for idx, proposals_per_im in enumerate(proposals):
        bag_ids = proposals_per_im.get_field("bag_ids")
        full_num, all_num = bag_ids[0], bag_ids[-1]
        print(full_num, all_num)
        idxs[offset:offset+full_num] = idx*2
        idxs[offset+full_num:offset+all_num] = idx*2+1
        offset += all_num
    return idxs
# cfg_path = "./configs/SiamRPN.yaml"
cfg_path = "yundao_configs/en_r50_mil/finetune.yaml"
cfg.merge_from_file(cfg_path)
data_loaders = make_data_loader(cfg, is_train=True, is_distributed=False)
# print(data_loaders)
for i,(image, boxlist, idx) in enumerate(tqdm(data_loaders)):
    texts = []
    new_proposals = []
    for proposals_per_im in boxlist:
        tm = len(proposals_per_im.get_field("texts").tolist())
        pm = proposals_per_im.get_field('polys').size(0)
        assert tm==pm, print(tm, pm)
        ptm = len(proposals_per_im.get_field("partial_texts"))
        ppm = proposals_per_im.get_field('partial_polys').size(0)
        assert ptm==ppm, print(ptm, ppm)
        texts.extend(proposals_per_im.get_field("texts").tolist())
        texts.extend(proposals_per_im.get_field("partial_texts"))
        
        print('here: ',tm, pm, tm+ptm, pm+ppm)
        all_polys = torch.cat([proposals_per_im.get_field('polys').float(), proposals_per_im.get_field('partial_polys').float()])
        minx,miny,maxx,maxy = torch.min(all_polys[:,::2], dim=1, keepdim=True)[0],\
                                torch.min(all_polys[:,1::2], dim=1, keepdim=True)[0],\
                                torch.max(all_polys[:,::2], dim=1, keepdim=True)[0],\
                                torch.max(all_polys[:,1::2], dim=1, keepdim=True)[0]
        bbox = torch.cat([minx,miny,maxx,maxy],dim=1)
        proposals_per_im.bbox = bbox
        proposals_per_im.add_field('polys', all_polys)
        new_proposals.append(proposals_per_im)
    all_polys_num = [p.get_field('polys').size(0) for p in new_proposals]
    print(all_polys_num)
    assert sum(all_polys_num)==len(texts), print(sum(all_polys_num),len(texts))
    imgs_texts = texts.copy()
    word_texts, words = get_augmented_words(texts)
    idxs = split_full_partial(boxlist)
    full_idxs = torch.nonzero(idxs%2==0).view(-1)
    part_idxs = torch.nonzero(idxs%2==1).view(-1)
    part_idxs = part_idxs
    part_idxs_list = part_idxs.data.cpu().numpy()

    part_img_texts = [imgs_texts[v] for v in range(len(imgs_texts)) if v in part_idxs_list]
    part_word_texts= [word_texts[v] for v in range(len(word_texts)) if v in part_idxs_list]
    assert part_idxs.max() < sum(all_polys_num), print(part_idxs.max(), all_polys_num)
    assert len(part_idxs) == len(part_img_texts), print(len(part_idxs) , len(part_img_texts))
