import os
import glob
import io
import json
from pycocotools.coco import COCO
import contextlib
lsvt_path = "/home/ymk-wh/workspace/datasets/retrieval_chinese/LSVT"
rects_path = "/home/ymk-wh/workspace/datasets/retrieval_chinese/ReCTS"
synth_chinese_path = "/home/ymk-wh/workspace/datasets/retrieval_chinese/SynthText_Chinese"
synth_chinese130k_path = "/home/ymk-wh/workspace/datasets/retrieval_chinese/syntext"
def isChinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False
def read_file_lsvt(path):
    words = []
    lines = open(path).readlines()
    for line in lines:
        parts = line.strip().split(';')
        if parts[-1] != '"###"':
            words.append(parts[-1])
    return words
def read_file_rects(path):
    words = []
    lines = open(path).readlines()
    for line in lines:
        # print(line)
        parts = line.strip().split(';')
        if parts[-1] != '"###"':
            words.append(parts[-1])
    return words
def read_file_synth_chinese(path):
    words = []
    lines = open(path).readlines()
    for line in lines:
        # print(line)
        parts = line.strip().split(',')
        if parts[-1] != '"###"':
            words.append(parts[-1])
    return words


def lsvt_stactic(path):
    all_words = {}
    for gt in glob.glob(os.path.join(path, '*.txt')):
        words = read_file_lsvt(gt)
        for word in words:
            if word in all_words.keys():
                all_words[word] += 1
            else:
                all_words[word] = 1
    all_words = sorted(all_words.items(), key=lambda item:item[1], reverse=True)
    # print(all_words[:500])
    return all_words
def rects_stactic(path):
    all_words = {}
    for gt in glob.glob(os.path.join(path, '*.txt')):
        words = read_file_rects(gt)
        for word in words:
            if word in all_words.keys():
                all_words[word] += 1
            else:
                all_words[word] = 1
    all_words = sorted(all_words.items(), key=lambda item:item[1], reverse=True)
    # print(all_words[:500])
    return all_words
def synth_chinese_stactic(path):
    all_words = {}
    for gt in glob.glob(os.path.join(path, '*.txt')):
        words = read_file_synth_chinese(gt)
        for word in words:
            if word in all_words.keys():
                all_words[word] += 1
            else:
                all_words[word] = 1
    all_words = sorted(all_words.items(), key=lambda item:item[1], reverse=True)
    # print(all_words[:500])
    return all_words

def count_lsvt():
    train_chars = []
    train_all_words = lsvt_stactic(os.path.join(lsvt_path, 'train'))
    test_all_words = lsvt_stactic(os.path.join(lsvt_path, 'test'))
    all_words = train_all_words
    all_words.extend(test_all_words)
    for word in all_words:
        word = word[0]
        for char in word.strip('"'):
            if char not in train_chars and isChinese(char):
                train_chars.append(char)
    return train_chars
def count_rects():
    train_chars = []
    train_all_words = rects_stactic(os.path.join(rects_path, 'train/labels'))
    test_all_words = rects_stactic(os.path.join(rects_path, 'test/labels'))
    all_words = train_all_words
    all_words.extend(test_all_words)
    for word in all_words:
        word = word[0]
        for char in word.strip('"'):
            if char not in train_chars and isChinese(char):
                train_chars.append(char)
    return train_chars
def count_synth_chinese():
    train_chars = []
    train_all_words = synth_chinese_stactic(os.path.join(synth_chinese_path, 'gts'))
    all_words = train_all_words
    for word in all_words:
        word = word[0]
        for char in word.strip('"'):
            if char not in train_chars and isChinese(char):
                train_chars.append(char)
    return train_chars
def count_synth130k():
    train_chars = []
    json_file = os.path.join(synth_chinese130k_path, 'annotations/syn130k_word_train.json')
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(json_file)
    imgs_anns = list(zip(imgs, anns))
    for texts in anns:
        for text in texts:
            word = text['rec']
            for char in word:
                if char not in train_chars and isChinese(char):
                    train_chars.append(char)
    return train_chars

def merge_chars(list_):
    chars = list_[0]
    for cl in list_[1:]:
        for char in cl:
            if char not in chars:
                chars.append(char)
    return chars
lsvt_chars = count_lsvt()
rects_chars= count_rects()
synth_chars= count_synth_chinese()
synth130k_chars = count_synth130k()
chars = merge_chars([lsvt_chars, rects_chars, synth_chars, synth130k_chars])
print(len(chars))
print(chars)
import numpy as np
np.save("./data_static/chars.npy", chars)