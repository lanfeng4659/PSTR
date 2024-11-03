import os
import torch
import numpy as np
import cv2
from scipy.misc import imread, imresize
import shutil
from PIL import Image,ImageDraw
from tqdm import tqdm
from sklearn.metrics import average_precision_score
def meanAP(preds, trues, queries=[]):
    APs = []
    if len(queries)==0:
        queries=['none']*len(trues)
    for y_scores, y_trues, query in zip(preds, trues, queries):
        
        AP = average_precision_score(y_trues, y_scores)
        temp = zip(y_scores, y_trues)
        sorted_paths = sorted(temp, key=lambda temp:temp[0], reverse=True)
        print(sorted_paths[:50], AP, query)
        APs.append(AP)
    return APs
def vis_multi_image(image_list, shape=[1,-1]):
    image_num = len(image_list)
    h, w,_ = np.array(image_list[0]).shape
    #print h,w
    num_w = int(image_num/shape[0])
    num_h = shape[0]
    new_im = Image.new('RGB', (num_w*w,num_h*h))
    for idx, image in enumerate(image_list):
        idx_w = idx%num_w
        idx_h = int(idx/num_w)
        new_im.paste(image, (int(idx_w*w),int(idx_h*h)))
    return new_im
def re_ranking(retrieval_texts_embedding,predictions,y_trues):
    y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):
        for image_id, prediction in enumerate(predictions):
            img_embedding = prediction.get_field("imgs_embedding_nor")
            if img_embedding.size(0)==0:
                score = 0
                box = [0,0,0,0]
            else:
                similarity = embedding.mm(img_embedding.t())
                score,box_idx =  similarity.max(dim=1)
                # print(similarity.min(dim=1))
                score = score.data.cpu().numpy()[0]
                # k = min(img_embedding.size(0),2)
                # print(similarity.shape)
                # score = torch.topk(similarity,k,dim=1)[0].mean().data.cpu().numpy()
                box_idx = box_idx.data.cpu().numpy()[0]
                box = prediction.get_field("polys")[box_idx].data.cpu().numpy()

            y_scores[idx,image_id] = score
    APs = meanAP(y_scores, y_trues)
    show_aps(APs)
    print(sum(APs)/len(APs))
    return y_scores

def get_y_scores_trues(predictions):
    
    y_scores = []
    max_idxs = []
    retrieval_image_embedding={}
    for image_id, prediction in enumerate(predictions):
        # print(prediction.fields())
        if "words_embedding_nor" in prediction.fields():
            words_embedding_nor = prediction.get_field("words_embedding_nor")
        if "y_trues" in prediction.fields():
            y_trues = prediction.get_field("y_trues")
        else:
            y_trues=None
        # boxes = prediction.bbox.data.cpu().numpy()
        polys = prediction.get_field("polys")
        scale = prediction.get_field("scale")
        polys[:,::2] *= scale[0]
        polys[:,1::2] *= scale[1]
        prediction.add_field("polys", polys)
        sr_num = prediction.get_field("scores").size(0)
        img_embedding = prediction.get_field("imgs_embedding_nor")[:sr_num]
        if img_embedding.size(0)==0:
            y_scores.append(torch.zeros([words_embedding_nor.size(0)]).to(img_embedding.device))
            max_idxs.append(torch.zeros([words_embedding_nor.size(0)]).to(img_embedding.device).long())
            continue
        similarity =  words_embedding_nor.mm(img_embedding.t())
        max_scores, max_idx = similarity.max(dim=1)
        y_scores.append(max_scores)
        max_idxs.append(max_idx)

    y_scores = torch.stack(y_scores,dim=1).data.cpu().numpy()
    max_idxs = torch.stack(max_idxs,dim=1).data.cpu().numpy()
    return y_scores, y_trues, max_idxs, words_embedding_nor

def evaluate_box_proposals(predictions, dataset=None):
    retrieval_texts_embedding = {}
    full_query_num = len(dataset.str_queries_full)
    y_scores, y_trues, max_idxs, words_embedding_nor = get_y_scores_trues(predictions)
    if y_trues==None:
        y_trues=dataset.y_trues
    print(max_idxs.shape, y_scores.shape)
    # texts = predictions[0].get_field("texts")
    retrieval_results = {}
    # for idx,(text, embedding) in enumerate(zip(texts, words_embedding_nor)):
    #     retrieval_results[text] = {}
    #     max_score = 0
    #     for image_id, prediction in enumerate(predictions):

    #         img_embedding = prediction.get_field("imgs_embedding_nor")
    #         # import ipdb; ipdb.set_trace()
    #         if img_embedding.size(0)==0:
    #             score = 0
    #             box = [0]*28
    #         else:
    #             score,box_idx =  y_scores[idx, image_id], max_idxs[idx, image_id]
    #             # print(box_idx)
                
    #             boxes = prediction.get_field("polys").data.cpu().numpy()
    #             all_box = boxes
    #             # print(boxes.shape, img_embedding.shape, box_idx)
    #             box = all_box[box_idx]

    #         # print(text,str(prediction.get_field("path")),idx,image_id)
    #         retrieval_results[text][str(prediction.get_field("path"))] = [score,box,y_trues[idx,image_id],all_box]
    #     retrieval_results[text]["y_trues"] = sum(y_trues[idx])
    #     retrieval_results[text]["ap"] = meanAP(y_scores[idx].reshape([1,-1]), y_trues[idx].reshape([1,-1]))
    # analysis_APs(retrieval_results, full_query_num)
    # APs = meanAP(y_scores, y_trues)
    # show_aps(APs)
    # print(sum(APs)/len(APs))
    mAPs = []

    APs = meanAP(y_scores[:full_query_num], y_trues[:full_query_num])
    mAPs.append(sum(APs)/len(APs)) 

    APs = meanAP(y_scores[full_query_num:], y_trues[full_query_num:])
    # print(APs)
    mAPs.append(sum(APs)/len(APs)) 
    print(mAPs)

    return retrieval_results
def analysis_APs(retrieval_results,full_query_num):
    full_ap_len_dict = {}
    partial_ap_len_dict = {}
    full_aps, partial_aps = [], []
    for idx, (text, info) in enumerate(retrieval_results.items()):
        textlen = len(text)
        if idx < full_query_num:
            if textlen in full_ap_len_dict.keys():
                full_ap_len_dict[textlen].append(info['ap'][0])
            else:
                full_ap_len_dict[textlen] = [info['ap'][0]]
            full_aps.append(info['ap'][0])
            print("full: {}: {} : {}".format(text, info['y_trues'], info['ap']))
        else:
            if textlen in partial_ap_len_dict.keys():
                partial_ap_len_dict[textlen].append(info['ap'][0])
            else:
                partial_ap_len_dict[textlen] = [info['ap'][0]]
            partial_aps.append(info['ap'][0])
            print("partial: {}: {} : {}".format(text, info['y_trues'], info['ap']))
    full_ap_len_dict = {k:sum(v)/len(v) for k,v in full_ap_len_dict.items()}
    partial_ap_len_dict = {k:sum(v)/len(v) for k,v in partial_ap_len_dict.items()}
    
    # print(sorted(full_ap_len_dict.items(), key=lambda x: x[0], reverse=False))
    # print(sorted(partial_ap_len_dict.items(), key=lambda x: x[0], reverse=False))
    # print(partial_aps)
    # print(sum(full_aps)/len(full_aps), sum(partial_aps)/len(partial_aps))
def show_aps(aps):
    import matplotlib.pyplot as plt
    aps = sorted(aps)
    plt.plot(aps)
    plt.plot([sum(aps)/len(aps)]*len(aps))
    plt.savefig("aps.png")
def draw_boxes(path,score,box,y,all_boxes):
    # print(box)
    # image = cv2.imread(path,cv2.IMREAD_COLOR)
    image = imread(path,mode="RGB")
    # print(path)
    # print(image.shape)
    # minx,miny,maxx,maxy = box
    # print(box)
    box = np.array(box).reshape([-1,14,2]).astype(np.int32)
    all_boxes = np.array(all_boxes).reshape([-1,14,2]).astype(np.int32)
    # box = np.array([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]]).reshape([-1,4,2]).astype(np.int32)
    # cv2.drawContours(image, all_boxes, -1, color=(0,0,255),thickness=1)
    # print(box)
    cv2.drawContours(image, box, -1, color=(0,255,0),thickness=2)
    cv2.putText(image,str(score),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,str(y),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,os.path.basename(path),(100,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    
    image = Image.fromarray(image).convert('RGB').resize((768, 640))
    return image
def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
# predictions = torch.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/predictions.pth")
def only_get_scores(predictions, words_embedding_nor=None, y_trues=None):
    y_scores, y_trues2, max_idxs, words_embedding_nor2 = get_y_scores_trues(predictions)
    
    if words_embedding_nor==None:
        words_embedding_nor = words_embedding_nor2
        y_trues = y_trues2
    return y_scores, y_trues, words_embedding_nor

if __name__ =='__main__':
    from maskrcnn_benchmark.data.datasets.lsvt import LSVT
    from maskrcnn_benchmark.data.datasets.csvtr import CSVTR
    from maskrcnn_benchmark.data.datasets.rects import ReCTS
    dataset = LSVT('/home/ymk-wh/workspace/datasets/retrieval_chinese/LSVT',is_training=False)
    # dataset = CSVTR('/home/ymk-wh/workspace/datasets/retrieval_chinese/chinese_collect',is_training=False)
    # dataset = ReCTS('/home/ymk-wh/workspace/datasets/retrieval_chinese/ReCTS',is_training=False)
    # folder = 'Log2/cn_r50_poly_cmsam_count_gsts_mil_ms/finetune/inference/rects_test'

    # predictions = torch.load(os.path.join(folder, 'csvtr_predictions0-1500.pth'))
    # # import ipdb; ipdb.set_trace()
    # # y_scores, y_trues, words_embedding_nor=only_get_scores(predictions)
    # y_scores1, y_trues1, max_idxs, words_embedding_nor = get_y_scores_trues(predictions,words_embedding_nor = None)
    # print("load first")
    # y_trues = dataset.y_trues
    # print(y_trues.shape)
    # del predictions
    # predictions1 = torch.load(os.path.join(folder, 'csvtr_predictions1500-.pth'))
    # y_scores2, y_trues2, max_idxs, words_embedding_nor = get_y_scores_trues(predictions1,words_embedding_nor = words_embedding_nor)
    # print(y_scores1.shape, y_scores2.shape)
    # y_scores = np.concatenate((y_scores1, y_scores2),axis=1)
    # full_query_num=len(dataset.str_queries_full)
    
    # partial_queries = dataset.str_queries_partial

    # mAPs = []
    # print('full')
    # APs = meanAP(y_scores[:full_query_num], y_trues[:full_query_num])
    # mAPs.append(sum(APs)/len(APs)) 
    # print('partial')
    # APs = meanAP(y_scores[full_query_num:], y_trues[full_query_num:], partial_queries)
    # # print(APs)
    # mAPs.append(sum(APs)/len(APs)) 
    # print(mAPs)

    # print("load done")
    # predictions.extend(predictions1)
    # print("extend done")
    # img_folder = os.path.join(folder, 'show')
    # mkdir(img_folder)
    predictions = torch.load("Log/cn_r50_bilinear_fcoscount_pt_b64/finetune/inference/lsvt_test/lsvt_test_predictions.pth")
    retrieval_results = evaluate_box_proposals(predictions,dataset)
    # del predictions
    # evaluate_box_proposals2(predictions,600)
    # np.save("temp.npy",retrieval_results)
    # retrieval_results = np.load("temp.npy",allow_pickle=True).item()
    # results = np.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/retrieval_results.npy",allow_pickle=True).item()
    # for text, datas in retrieval_results.items():
    #     # dict_ = {data.keys():data.values() for data in datas}
    #     y_trues = datas["y_trues"]
    #     ap = datas["ap"]
    #     del datas["y_trues"]
    #     del datas["ap"]
    #     sorted_paths = sorted(datas.items(), key=lambda datas:datas[1][0], reverse=True)
    #     image_list = [draw_boxes(x[0], x[1][0], x[1][1],x[1][2],x[1][3]) for x in sorted_paths[:20]]
    #     image = vis_multi_image(image_list, shape=[4,-1])
    #     image.save(os.path.join(img_folder,"{}_{}_{}.jpg".format(text,y_trues, ap[0])))