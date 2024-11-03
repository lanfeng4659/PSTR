import os, glob
import thulac
import numpy as np
from maskrcnn_benchmark.data.datasets.chinese_utils import load_chars
chars = load_chars()
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
thu_lac = thulac.thulac(seg_only=True)
def read_file_lsvt_and_rects(path):
    words = []
    lines = open(path).readlines()
    for line in lines:
        parts = line.strip().split(';')
        if parts[-1] != '"###"':
            words.append(parts[-1])
    return words
def stactic(path):
    all_words = {}
    for gt in glob.glob(os.path.join(path, '*.txt')):
        words = read_file_lsvt_and_rects(gt)
        for word in words:
            word = filter_word(word.strip("\""), chars)
            if len(word) < 2:
                continue 
            if word in all_words.keys():
                all_words[word] += 1
            else:
                all_words[word] = 1
    all_words = sorted(all_words.items(), key=lambda item:item[1], reverse=True)
    # print(all_words[:500])
    return all_words
def cut_word(dicts):
    new_dict = {}
    for k,v in dicts:
        segs = thu_lac.cut(k.strip("\""), text=True)
        for seg in segs.split(" "):
            if len(seg) ==1:
                continue
            if seg in new_dict.keys():
                new_dict[seg] += v
            else:
                new_dict[seg] = v
    new_dict = sorted(new_dict.items(), key=lambda item:item[1], reverse=True)
    # new_dict = sorted(new_dict.items(), key=lambda item:item[1], reverse=True)
    return new_dict
def write_to_text(lists, path):
    f = open(path, 'w')
    for k,v in lists:
        f.write(k+'\n')
def find_query(data_path, save_path):
    # all_words = lsvt_stactic(data_path)
    new_dict = stactic(data_path)
    # new_dict = cut_word(new_dict)
    write_to_text(new_dict[:600], save_path)
def find_partial_labels_csvtr(data_path, save_path):
    all_words = {}
    for query in os.listdir(data_path):
        all_words[query] = len(os.listdir(os.path.join(data_path, query)))
    new_dict = sorted(all_words.items(), key=lambda item:item[1], reverse=True)
    new_dict = cut_word(new_dict)
    write_to_text(new_dict, save_path)

def find_partial_labels_csvtrv1v2(path, save_path):
    all_words = {}
    data_path = os.path.join(path, 'v1')
    for query in os.listdir(data_path):
        if '.txt' in query:
            continue
        all_words[query] = len(os.listdir(os.path.join(data_path, query)))

    data_path = os.path.join(path, 'v2')
    for query in os.listdir(data_path):
        all_words[query] = len(os.listdir(os.path.join(data_path, query)))

    new_dict = sorted(all_words.items(), key=lambda item:item[1], reverse=True)
    new_dict = cut_word(new_dict)
    write_to_text(new_dict, save_path)

find_partial_labels_csvtrv1v2("/home/ymk-wh/workspace/datasets/retrieval_chinese/chinese_collect", "./dataset_labels/csvtr_partial_query_v1v2.txt")
# find_partial_labels_csvtr("/home/ymk-wh/workspace/datasets/retrieval_chinese/chinese_collect/v1", "./dataset_labels/csvtr_query.txt")
# find_query("/home/ymk-wh/workspace/datasets/retrieval_chinese/ReCTS/test/labels", save_path='./dataset_labels/rects_query_full.txt')
# find_query("/home/ymk-wh/workspace/datasets/retrieval_chinese/LSVT/test", save_path='./dataset_labels/lsvt_query_full.txt') # need to filter out some queries in person.

