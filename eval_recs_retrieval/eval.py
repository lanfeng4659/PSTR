from maskrcnn_benchmark.data.datasets.csvtr import CSVTR
from maskrcnn_benchmark.data.datasets.rects import ReCTS
from maskrcnn_benchmark.data.datasets.lsvt import LSVT
import os
import numpy as np
from eval_recs_retrieval.read_labels import get_jump_retrieval_y_trues, get_jump_retrieval_y_trues_rects_lsvt
import editdistance
from sklearn.metrics import average_precision_score
dataset_dict = {
    'CSVTR': ['/home/ymk-wh/workspace/datasets/retrieval_chinese/chinese_collect', "/home/ymk-wh/workspace/researches/wanghao/RetrievalHuaWei/csvtr_jump_query.txt"],
    # 'ReCTS': ['/home/ymk-wh/workspace/datasets/retrieval_chinese/ReCTS', "query/rects/query_nctp.txt"],
    # 'LSVT' : ['/home/ymk-wh/workspace/datasets/retrieval_chinese/LSVT', "query/lsvt/query_nctp.txt"]
}
function_dict = {
    'CSVTR': ['get_jump_retrieval_y_trues', "chinese_collect"],
    'ReCTS': ['get_jump_retrieval_y_trues_rects_lsvt', "ReCTS/test/images"],
    'LSVT' : ['get_jump_retrieval_y_trues_rects_lsvt', "LSVT/test"]
}
expName_floder = {
    'ABCNetv2': '/home/ymk-wh/workspace2/researches/wanghao/ABCNetv2/texts',
    # 'Ours-Rec': '/home/ymk-wh/workspace/researches/wanghao/RetrievalHuaWei/eval_recs_retrieval/Ours-Rec',
}
def get_queries(path):
    queries = []
    for line in open(path).readlines():
        queries.append(line.strip())
    return queries
def build_query_trues(dataset_name = 'CSVTR'):
    img_floder, query_path = dataset_dict[dataset_name]
    dataset = eval(dataset_name)(img_floder,is_training=False)
    queries = []
    queries += dataset.str_queries_full
    queries += dataset.str_queries_partial
    str_queries_jump = get_queries(query_path)
    queries = queries + str_queries_jump
    
    get_gt_func, pred_path = function_dict[dataset_name]
    y_trues = eval(get_gt_func)(dataset, queries)
    
    return pred_path, dataset,queries,y_trues, [len(dataset.str_queries_full), len(dataset.str_queries_partial), len(str_queries_jump)]
def read_texts(floder, img_path, is_csvtr=False):
    # print(floder, is_csvtr, img_path)
    if is_csvtr:
        pred_path = os.path.join(floder, "/".join(img_path.split('/')[-3:])+'.txt')
        # print()
    else:
        pred_path = os.path.join(floder, os.path.basename(img_path).replace('.jpg','.txt'))
    texts = []
    if os.path.exists(pred_path):
        texts = [text.strip() for text in open(pred_path).readlines()]
    else:
        print("File {} not exists.".format(pred_path))
    return texts
def calculate_similarity(query, texts):
    def nor_editdistance(word1, word2):
        return 1-editdistance.eval(word1,word2)/max(len(word1), len(word2))
    score = 0 if len(texts)==0 else max([nor_editdistance(query, text) for text in texts])
    # print(score)
    return score
def meanAP(preds, trues, queries=[]):
    APs = []
    for y_scores, y_trues in zip(preds, trues):
        
        AP = average_precision_score(y_trues, y_scores)
        APs.append(AP)
    return APs   
from tqdm import tqdm
def evaluate(floder, dataset,queries,y_trues):
    scores = np.zeros_like(y_trues)
    for iImg, img_path in enumerate(tqdm(dataset.img_lists)):
        texts = read_texts(floder, img_path, is_csvtr=dataset.name=='csvtr')
        for iQuery, query in enumerate(queries):
            scores[iQuery, iImg]=calculate_similarity(query, texts)
    APs = meanAP(scores,y_trues)
    return APs
def sort_all_map(APs, nums):
    fm_num, ctp_num, nctp_num = nums
    pm_num = ctp_num + nctp_num
    line = "FM ({}): {}, PM ({}): {}, CTP ({}): {}, NCTP ({}): {}".format(
        fm_num, sum(APs[:fm_num])/fm_num,
        pm_num, sum(APs[fm_num:fm_num+pm_num])/pm_num,
        ctp_num, sum(APs[fm_num:fm_num+ctp_num])/ctp_num,
        nctp_num, sum(APs[fm_num+ctp_num:fm_num+ctp_num+nctp_num])/nctp_num
    )
    print(line)
    # print_lines.append(line)
    # return APs   
import time
def evaluate_all_experiments():

    for expName, floder in expName_floder.items():
        print('Evaluate {}.'.format(expName))
        
        for dataset_name in dataset_dict.keys():

            print('Dataset {} ...'.format(dataset_name))
            # start = time.time()
            pred_path, dataset,queries,y_trues, query_nums = build_query_trues(dataset_name = dataset_name)
            floder = os.path.join(floder, pred_path)
            start = time.time()
            all_APs = evaluate(floder, dataset=dataset,queries=queries,y_trues=y_trues)
            sort_all_map(all_APs, query_nums)
            # np.savez(os.path.join("all_results", "{}_{}.npz".format(expName, dataset_name)), print_lines=print_lines, queries=queries, all_APs=all_APs, query_nums=query_nums)
            print("Speed: {} fps.".format(len(y_trues[0])/(time.time() - start)))
evaluate_all_experiments()