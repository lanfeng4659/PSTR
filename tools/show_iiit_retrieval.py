import os
import torch
from torch import nn
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
def precision(preds, trues):
    for n in [100,200,300,400,500,600,700,800,900,1000]:
        p = []
        n*=10
        for y_scores, y_trues in zip(preds, trues):
            # print(y_trues[:10], y_scores[:10])
            rank1 = np.argsort(y_scores)[::-1]
            top_y_trues = y_trues[rank1][:n]
            top_y_scores= y_scores[rank1][:n]
            # print(top_y_scores)
            AP = average_precision_score(top_y_trues, top_y_scores)
            # AP = sum(top_y_trues)/sum(y_trues)
            # print(AP)
            p.append(AP)
        print("P@{}: {}".format(n,sum(p)/len(p)))
    return p
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
                box = prediction.get_field("boxes")[box_idx].data.cpu().numpy()

            y_scores[idx,image_id] = score
    APs = meanAP(y_scores, y_trues)
    show_aps(APs)
    print(sum(APs)/len(APs))
    
    return y_scores
def evaluate_box_proposals(
    predictions, dataset, thresholds=0.23, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # img_floder = os.path.join(output_folder,'images')
    # txt_floder = os.path.join(output_folder,'texts')
    # for folder in [img_floder, txt_floder]:
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    y_trues = dataset.dataset.y_trues
    words_embedding_nor = None
    y_scores = []
    retrieval_image_embedding={}
    for image_id, prediction in enumerate(predictions):
        # print(prediction.fields())
        # import ipdb; ipdb.set_trace()
        path = str(prediction.get_field("path"))
        # image = cv2.imread(path)
        path = os.path.join('vis',os.path.basename(path))
        if "words_embedding_nor" in prediction.fields():
            words_embedding_nor = prediction.get_field("words_embedding_nor")
        boxes = prediction.bbox.data.cpu().numpy()
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        boxes = boxes[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        # cv2.drawContours(image, boxes, -1, color=(255,0,0), thickness=1)
        img_embedding = prediction.get_field("imgs_embedding_nor")
        if img_embedding.size(0)==0:
            y_scores.append(torch.zeros([words_embedding_nor.size(0)]).to(img_embedding.device))
            continue
        similarity =  words_embedding_nor.mm(img_embedding.t())
        # ver = prediction.get_field("ver").softmax(dim=-1)
        # print(ver)
        # similarity = similarity * ver[:,:,1]
        # similarity = ver[:,:,1]
        # print(ver.shape, similarity.shape)
        score, idx = similarity.max(dim=1)
        # print(score)
        y_scores.append(score)
        # cv2.drawContours(image, boxes[idx].reshape([-1,4,2]), -1, color=(0,255,0), thickness=2)
        # cv2.imwrite(path, image)
    # import ipdb;ipdb.set_trace()
    y_scores = torch.stack(y_scores,dim=1).data.cpu().numpy()
    APs = meanAP(y_scores, y_trues)
    mAP = sum(APs)/len(APs)
    print(mAP)

    # y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)
    # print(mAP)
    # np.save(os.path.join(output_folder,"retrieval_results.npy"), retrieval_results)
    return mAP
def local_score(a,b):
    a = nn.functional.normalize(a.reshape([15,-1]))
    b = nn.functional.normalize(b.reshape([15,-1]))
    c = (a*b).sum(dim=-1).mean()
    return c

def evaluate_box_proposals2(predictions,dataset):
    retrieval_texts_embedding = {}
    y_trues = dataset.dataset.y_trues
    for image_id, prediction in tqdm(enumerate(predictions)):
        
        # print(prediction)
        if "words_embedding_nor" in prediction.fields():
            words_embedding = prediction.get_field("words_embedding_nor")
        # if "y_trues" in prediction.fields():
        #     y_trues = prediction.get_field("y_trues")
            # import ipdb;ipdb.set_trace()
        boxes = prediction.bbox
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        prediction.add_field("boxes", boxes)
        if words_embedding.size(0) == 0:
            continue
        for idx,text in enumerate(prediction.get_field("texts")):
            text = text.lower()
            retrieval_texts_embedding[text] = words_embedding[idx,:].reshape(1,-1)
    y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    # y_trues = prediction.get_field("y_trues")
    retrieval_results = {}
    retrieval_image_embedding={}
    use_local_score = False
    for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):
        retrieval_results[text] = {}
        retrieval_image_embedding[text] = embedding
        max_score = 0
        for image_id, prediction in enumerate(predictions):
            img_embedding = prediction.get_field("imgs_embedding_nor")
            if img_embedding.size(0)==0:
                score = 0
                box = [0,0,0,0]
                boxes = [[0,0,0,0]]
            else:
                similarity = embedding.mm(img_embedding.t())
                # print(similarity.shape)
                score,box_idx =  similarity.max(dim=1)
                # print(score)
                # import ipdb; ipdb.set_trace()
                # print(box_idx,prediction.get_field("boxes").shape,img_embedding.shape)
                # print(similarity.min(dim=1))
                if use_local_score:
                    score = local_score(embedding, img_embedding[box_idx])
                else:
                    score = score.data.cpu().numpy()[0]
                # k = min(img_embedding.size(0),2)
                # print(similarity.shape)
                # score = torch.topk(similarity,k,dim=1)[0].mean().data.cpu().numpy()
                box_idx = box_idx.data.cpu().numpy()[0]
                box = prediction.get_field("boxes")[box_idx].data.cpu().numpy()
                boxes = prediction.get_field("boxes").data.cpu().numpy()
                # print(box)
            if score>max_score:
                max_score = score
                retrieval_image_embedding[text]=img_embedding[box_idx].view(1,-1)
            y_scores[idx,image_id] = score
            retrieval_results[text][str(prediction.get_field("path"))] = [score,box,y_trues[idx,image_id],boxes]
        retrieval_results[text]["y_trues"] = sum(y_trues[idx])
        retrieval_results[text]["ap"] = meanAP(y_scores[idx].reshape([1,-1]), y_trues[idx].reshape([1,-1]))
    # import ipdb; ipdb.set_trace()
    APs = meanAP(y_scores, y_trues)
    # show_aps(APs)
    print(sum(APs)/len(APs))
    precision(y_scores, y_trues)
    y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)

    APs = meanAP(y_scores+y_scores2, y_trues)
    print(sum(APs)/len(APs))
    return retrieval_results
def global_score(x,y):
    return nn.functional.normalize(x).mm(nn.functional.normalize(y).t())
# def evaluate_box_proposals2(predictions,dataset):
#     retrieval_texts_embedding = {}
#     y_trues = dataset.dataset.y_trues
#     for image_id, prediction in tqdm(enumerate(predictions)):
        
#         # print(prediction)
#         if "words_embedding_nor" in prediction.fields():
#             words_embedding_ = prediction.get_field("words_embedding_nor")
#             words_embedding = words_embedding_.tanh().view(words_embedding_.size(0),-1)
#         # if "y_trues" in prediction.fields():
#         #     y_trues = prediction.get_field("y_trues")
#             # import ipdb;ipdb.set_trace()
#         boxes = prediction.bbox
#         scale = prediction.get_field("scale")
#         boxes[:,::2] *= scale[0]
#         boxes[:,1::2] *= scale[1]
#         prediction.add_field("boxes", boxes)
#         if words_embedding.size(0) == 0:
#             continue
#         for idx,text in enumerate(prediction.get_field("texts")):
#             text = text.lower()
#             retrieval_texts_embedding[text] = words_embedding[idx,:].reshape(1,-1)
#     y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
#     # y_trues = prediction.get_field("y_trues")
#     retrieval_results = {}
#     retrieval_image_embedding={}
#     use_local_score = True
#     for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):
#         retrieval_results[text] = {}
#         retrieval_image_embedding[text] = embedding
#         max_score = 0
#         for image_id, prediction in enumerate(predictions):
#             img_embedding_ = prediction.get_field("imgs_embedding_nor")
            
#             if img_embedding_.size(0)==0:
#                 score = 0
#                 box = [0,0,0,0]
#             else:
#                 img_embedding = img_embedding_.tanh().view(img_embedding_.size(0),-1)
#                 # similarity = embedding.mm(img_embedding.t())
#                 similarity = global_score(embedding, img_embedding)
#                 # print(similarity.shape)
#                 score,box_idx =  similarity.max(dim=1)
                
#                 # import ipdb; ipdb.set_trace()
#                 # print(box_idx,prediction.get_field("boxes").shape,img_embedding.shape)
#                 # print(similarity.min(dim=1))
#                 if use_local_score:
#                     score = local_score(embedding, img_embedding[box_idx])
#                 else:
#                     score = score.data.cpu().numpy()[0]
#                 # k = min(img_embedding.size(0),2)
#                 # print(similarity.shape)
#                 # score = torch.topk(similarity,k,dim=1)[0].mean().data.cpu().numpy()
#                 box_idx = box_idx.data.cpu().numpy()[0]
#                 box = prediction.get_field("boxes")[box_idx].data.cpu().numpy()
#                 # print(box)
#             if score>max_score:
#                 max_score = score
#                 retrieval_image_embedding[text]=img_embedding[box_idx].view(1,-1)
#             y_scores[idx,image_id] = score
#             retrieval_results[text][str(prediction.get_field("path"))] = [score,box,y_trues[idx,image_id]]
#         retrieval_results[text]["y_trues"] = sum(y_trues[idx])
#         retrieval_results[text]["ap"] = meanAP(y_scores[idx].reshape([1,-1]), y_trues[idx].reshape([1,-1]))
#     APs = meanAP(y_scores, y_trues)
#     # show_aps(APs)
#     print(sum(APs)/len(APs))
#     precision(y_scores, y_trues)
#     y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)

#     APs = meanAP(y_scores+y_scores2, y_trues)
#     print(sum(APs)/len(APs))
#     return retrieval_results
def show_aps(aps):
    import matplotlib.pyplot as plt
    aps = sorted(aps)
    plt.plot(aps)
    plt.plot([sum(aps)/len(aps)]*len(aps))
    plt.savefig("aps.png")
def draw_boxes(path,score,box,y,boxes):
    # print(box)
    # image = cv2.imread(path,cv2.IMREAD_COLOR)
    image = imread(path,mode="RGB")
    # print(path)
    # print(image.shape)
    minx,miny,maxx,maxy = box
    # print(box)
    box = np.array([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]]).reshape([-1,4,2]).astype(np.int32)
    boxes = np.array([[[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]] for box in boxes]).reshape([-1,4,2]).astype(np.int32)
    cv2.drawContours(image, boxes, -1, color=(255,0,0),thickness=1)
    cv2.drawContours(image, box, -1, color=(0,255,0),thickness=2)
    
    cv2.putText(image,str(score),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,str(y),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,os.path.basename(path),(100,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    
    image = Image.fromarray(image).convert('RGB').resize((768, 640))
    return image
save_path = "show_iiit_retrievals"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# predictions = torch.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/predictions.pth")
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
if __name__ =='__main__':
    # predictions = torch.load("Log/retrival_scalenetnocc_pre_aug/inference/iiit_test/predictions.pth")
    # predictions = torch.load("predictions_iiit_7799.pth")
    # predictions = torch.load("predictions_iiit_7805.pth")
    # predictions = torch.load("predictions_iiit_7709.pth")
    # predictions = torch.load("predictions_iiit_7832.pth")
    predictions = torch.load("Log/scalenetnocc_r50bn_neckl4_norec_poly/finetune/inference/totaltext_test/predictions.pth")
    # predictions = torch.load("predictions_iiit_7916.pth")
    # predictions = torch.load("predictions_temp.pth")
    cfg.merge_from_file('configs/retrival.yaml')
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    # retrieval_results = evaluate_box_proposals(predictions,data_loaders_val[0].dataset)
    retrieval_results = evaluate_box_proposals2(predictions,data_loaders_val[0].dataset)
    np.save("temp.npy",retrieval_results)
    retrieval_results = np.load("temp.npy",allow_pickle=True).item()
    # results = np.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/retrieval_results.npy",allow_pickle=True).item()
    for text, datas in retrieval_results.items():
        # dict_ = {data.keys():data.values() for data in datas}
        y_trues = datas["y_trues"]
        ap = datas["ap"]
        del datas["y_trues"]
        del datas["ap"]
        sorted_paths = sorted(datas.items(), key=lambda datas:datas[1][0], reverse=True)
        image_list = [draw_boxes(x[0], x[1][0], x[1][1],x[1][2], x[1][3]) for x in sorted_paths[:48]]
        image = vis_multi_image(image_list, shape=[6,-1])
        image.save(os.path.join(save_path,"{}_{}_{}.jpg".format(text,y_trues, ap[0])))