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
def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
def show_basic_info(predictions, save_folder):
    mkdir(save_folder)
    retrieval_texts_embedding = {}
    if "words_embedding_nor" in predictions[0].fields():
        words_embedding = predictions[0].get_field("words_embedding_nor")
        texts = predictions[0].get_field("texts")
        for text, we in zip(texts, words_embedding):
            # print(text, we.size(1))
            retrieval_texts_embedding[text] = we.reshape(1,-1)
    for image_id, prediction in enumerate(predictions):
        img_path = str(prediction.get_field("path"))
        img_folder = os.path.join(save_folder, img_path.split('/')[-2])
        mkdir(img_folder)
        if "query_lens" not in prediction.fields():
            continue
        # print("this")
        lens = prediction.get_field("query_lens").reshape([-1]).data.cpu().numpy()
        polys = prediction.get_field("polys").reshape([-1,14,2]).data.cpu().numpy()
        scale = prediction.get_field("scale")
        polys[:,:,0] *= scale[0]
        polys[:,:,1] *= scale[1]
        # query = img_path.split('/')[-2]
        query = "银行"
        query_embedding = retrieval_texts_embedding[query]
        img_embeddings = prediction.get_field("imgs_embedding_nor")
        # rois = prediction.get_field("rois")
        # for v in img_embeddings:
        #     print(v.shape, query_embedding.shape)
        # img_embeddings = [v for v in img_embeddings if v.size(1)==query_embedding.size(1)]

        image = cv2.imread(img_path)
        max_score = 0
        max_score_poly = None
        full_num = len(prediction.get_field("lens"))
        for idx, (wordlen, poly, em) in enumerate(zip(lens, polys, img_embeddings)):
            poly = poly.astype(np.int32)
            
            if wordlen == len(query):
                ie = em.view(-1,1)
                # if idx < full_num:
                #     print(em.sum())
                    # print(rois)
                score = query_embedding.mm(ie).data.cpu().numpy()[0][0]
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) if idx >= full_num else (255,0,0)
                # cv2.drawContours(image, [poly], -1, color=color,thickness=2)
                # cv2.putText(image,"%.3f" % score,(poly[0][0]+20,poly[0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                # print(score)
                if score > max_score:
                    max_score_poly = poly
                    max_score = score
        if max_score == 0:
            continue
        # print(max_score)
        cv2.putText(image,str(max_score),(max_score_poly[0][0]+20,max_score_poly[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1)

        cv2.drawContours(image, [max_score_poly], -1, color=(0,255,0),thickness=2)
        cv2.putText(image,str(wordlen),(poly[0][0],poly[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
        cv2.imwrite(os.path.join(img_folder, img_path.split('/')[-1]), image)
    return
def evaluate_box_proposals(predictions, full_query_num=23):
    retrieval_texts_embedding = {}
    y_trues = None
    # full_query_num = len(dataset.dataset.str_queries_full)
    for image_id, prediction in tqdm(enumerate(predictions)):
        if "words_embedding_nor" in prediction.fields():
            words_embedding = prediction.get_field("words_embedding_nor")
            texts = prediction.get_field("texts")
            for text, we in zip(texts, words_embedding):
                # print(text, we.size(1))
                retrieval_texts_embedding[text] = we.reshape(1,-1)
            # import ipdb; ipdb.set_trace()
        if "y_trues" in prediction.fields():
            y_trues = prediction.get_field("y_trues")
            # import ipdb;ipdb.set_trace()
        # boxes = prediction.bbox
        polys = prediction.get_field('polys')
        scale = prediction.get_field("scale")
        polys[:,::2] *= scale[0]
        polys[:,1::2] *= scale[1]
        prediction.add_field("polys", polys)
    y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    # y_trues = prediction.get_field("y_trues")
    retrieval_results = {}
    retrieval_image_embedding={}
    for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):
        retrieval_results[text] = {}
        retrieval_image_embedding[text] = embedding
        max_score = 0
        we_len = embedding.size(1)
        for image_id, prediction in enumerate(predictions):
            img_embeddings = prediction.get_field("imgs_embedding_nor")
            # for v in img_embeddings:
            #     print(v.shape, v.size(1), we_len)
            img_embedding = [v for v in img_embeddings if v.size(1)==we_len]
            if len(img_embedding)==0:
                score = 0
                box = [0]*28
                all_box = prediction.get_field("polys").data.cpu().numpy()
            else:
                img_embedding = img_embedding[0]
                similarity = embedding.mm(img_embedding.t())
                score,box_idx =  similarity.max(dim=1)
                # print(similarity.min(dim=1))
                score = score.data.cpu().numpy()[0]
                box_idx = box_idx.data.cpu().numpy()[0]
                box = prediction.get_field("polys")[box_idx].data.cpu().numpy()
                all_box = prediction.get_field("polys").data.cpu().numpy()
            if score>max_score:
                max_score = score
                retrieval_image_embedding[text]=img_embedding[box_idx].view(1,-1)
            y_scores[idx,image_id] = score
            retrieval_results[text][str(prediction.get_field("path"))] = [score,box,y_trues[idx,image_id],all_box]
        retrieval_results[text]["y_trues"] = sum(y_trues[idx])
        retrieval_results[text]["ap"] = meanAP(y_scores[idx].reshape([1,-1]), y_trues[idx].reshape([1,-1]))
    # print(y_scores.max(axis=1))
    analysis_APs(retrieval_results, full_query_num)
    # APs = meanAP(y_scores, y_trues)
    # show_aps(APs)
    # print(sum(APs)/len(APs))
    mAPs = []

    APs = meanAP(y_scores[:full_query_num], y_trues[:full_query_num])
    mAPs.append(sum(APs)/len(APs)) 

    APs = meanAP(y_scores[full_query_num:], y_trues[full_query_num:])
    mAPs.append(sum(APs)/len(APs)) 
    print(mAPs)
    # y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)

    # APs = meanAP(y_scores+y_scores2, y_trues)
    # print(sum(APs)/len(APs))
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
    
    print(sorted(full_ap_len_dict.items(), key=lambda x: x[0], reverse=False))
    print(sorted(partial_ap_len_dict.items(), key=lambda x: x[0], reverse=False))
    print(sum(full_aps)/len(full_aps), sum(partial_aps)/len(partial_aps))
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
    cv2.drawContours(image, all_boxes, -1, color=(0,0,255),thickness=1)
    cv2.drawContours(image, box, -1, color=(0,255,0),thickness=2)
    cv2.putText(image,str(score),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,str(y),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,os.path.basename(path),(100,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    
    image = Image.fromarray(image).convert('RGB').resize((768, 640))
    return image
save_path = "show_rects_retrievals"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# predictions = torch.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/predictions.pth")
if __name__ =='__main__':
    folder = 'Log/cn_r50_bilinear_fcoscount_cutword_eg/finetune/inference/chinese_collect'
    predictions = torch.load(os.path.join(folder, 'chinese_collect_predictions.pth'))
    show_basic_info(predictions, save_folder=os.path.join(folder, 'show_basic_info'))
    # retrieval_results = evaluate_box_proposals(predictions,23)
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
    #     image.save(os.path.join(save_path,"{}_{}_{}.jpg".format(text,y_trues, ap[0])))