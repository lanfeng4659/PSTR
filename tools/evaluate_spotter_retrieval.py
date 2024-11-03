import os
import torch
import numpy as np
import cv2
from scipy.misc import imread, imresize
import shutil
from PIL import Image,ImageDraw
from tqdm import tqdm
from sklearn.metrics import average_precision_score
def meanAP(preds, trues):
    APs = []
    for y_scores, y_trues in zip(preds, trues):
        AP = average_precision_score(y_trues, y_scores)
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
import editdistance
def nor_editdistance(word1, word2):
    # return 1 if word1==word2 else 0
    return 1-editdistance.eval(word1,word2)/max(len(word1), len(word2))
def get_y_scores_trues(predictions):
    def get_scores_maxs(recognized_texts, texts):
        y_scores = np.zeros([len(texts), len(recognized_texts)])
        for i in range(len(texts)):
            for j in range(len(recognized_texts)):
                y_scores[i,j] = nor_editdistance(texts[i], recognized_texts[j])
        return y_scores

    words_embedding_nor = None
    y_scores = []
    max_idxs = []
    retrieval_image_embedding={}
    for image_id, prediction in enumerate(predictions):
        if "y_trues" in prediction.fields():
            y_trues = prediction.get_field("y_trues")
        # boxes = prediction.bbox.data.cpu().numpy()
        polys = prediction.get_field("polys")
        scale = prediction.get_field("scale")
        polys[:,::2] *= scale[0]
        polys[:,1::2] *= scale[1]
        prediction.add_field("polys", polys)
        texts = prediction.get_field("texts")
        similarity = get_scores_maxs(prediction.get_field("recognized_texts"), texts)
        if similarity.shape[1]==0:
            similarity = np.zeros([len(texts), 1])
        similarity = torch.tensor(similarity)
        

        max_scores, max_idx = similarity.max(dim=1)
        y_scores.append(max_scores)
        max_idxs.append(max_idx)

    y_scores = torch.stack(y_scores,dim=1).data.cpu().numpy()
    max_idxs = torch.stack(max_idxs,dim=1).data.cpu().numpy()
    return y_scores, y_trues, max_idxs

def evaluate_box_proposals(predictions, full_query_num=23):
    retrieval_texts_embedding = {}
    y_scores, y_trues, max_idxs = get_y_scores_trues(predictions)
    print(max_idxs.shape, y_scores.shape)
    texts = predictions[0].get_field("texts")
    retrieval_results = {}
    for idx,text in enumerate(texts):
        retrieval_results[text] = {}
        max_score = 0
        for image_id, prediction in enumerate(predictions):
            recognized_texts = prediction.get_field("recognized_texts")
            if len(recognized_texts)==0:
                score = 0
                box = [0]*28
            else:
                score,box_idx =  y_scores[idx, image_id], max_idxs[idx, image_id]
                # print(box_idx)
                
                boxes = prediction.get_field("polys").data.cpu().numpy()
                all_box = boxes
                # print(boxes.shape, img_embedding.shape, box_idx)
                box = all_box[box_idx]


            retrieval_results[text][str(prediction.get_field("path"))] = [score,box,y_trues[idx,image_id],all_box]
        retrieval_results[text]["y_trues"] = sum(y_trues[idx])
        retrieval_results[text]["ap"] = meanAP(y_scores[idx].reshape([1,-1]), y_trues[idx].reshape([1,-1]))
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
if __name__ =='__main__':
    folder = 'Log/spotter/finetune/inference/lsvt_test'
    predictions = torch.load(os.path.join(folder, 'lsvt_test_predictions.pth'))
    img_folder = os.path.join(folder, 'show')
    mkdir(img_folder)
    # predictions = torch.load("Log/cn_r50_bilinear_fcoscount/finetune/inference/rects_test/rects_test_best_predictions.pth")
    retrieval_results = evaluate_box_proposals(predictions,600)
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