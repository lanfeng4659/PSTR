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

def visualize_attention_image(image_filepath, prediction,boxes, vis_dir):
    vis_image_h, vis_image_w = 32, 128
    image = cv2.imread(image_filepath)
    if image is None:
        return None
    image_h, image_w, _ = image.shape
    image_copy = image.copy()
    if len(prediction) <= 0:
        return
    # polygons = [p.polygons[0].numpy().reshape(14, 2).astype(np.int32) for p in prediction.get_field("masks").polygons]
    image_copy = cv2.polylines(image_copy, boxes, True, (0, 255, 0), 1)
    ret_concat_images = [image_copy]

    bboxes = prediction.bbox.numpy().astype(np.int32)
    # rec_codes = prediction.get_field("rec_code")
    # rec_probs = prediction.get_field("rec_prob")
    rec_attentions = prediction.get_field("attention").numpy()
    n_boxes = bboxes.shape[0]
    for i in range(n_boxes):
        x1, y1, x2, y2 = bboxes[i]
        # rec_code = rec_codes[i]
        attention_images = rec_attentions[i]
        len_seq = attention_images.shape[0]
        # print(attention_images.shape)
        # len_seq = (rec_code > 0).sum().item() + 1  # 1 for EOS token
        roi_image = image[y1:y2, x1:x2, :].copy()
        roi_h, roi_w, _ = roi_image.shape
        if roi_h * roi_w < 20:  # remove tiny box
            continue
        roi_image = cv2.resize(roi_image, (vis_image_w, vis_image_h))
        all_step_attention_images = [roi_image]
        for step in range(len_seq):
            temp = attention_images[step]
            attention_heatmap = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp)) * 255).astype(np.uint8).reshape([1,-1])
            # print(attention_heatmap.shape)
            attention_heatmap = cv2.resize(attention_heatmap, (vis_image_w, vis_image_h))
            attention_heatmap = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)
            attention_image = cv2.addWeighted(attention_heatmap, 0.4, roi_image, 0.6, 0.)
            all_step_attention_images.append(attention_image)
        all_step_attention_images = np.vstack(all_step_attention_images)
        ret_concat_images.append(all_step_attention_images)
    
    # concat image list and save
    hs = [img.shape[0] for img in ret_concat_images]
    max_h = max(hs)
    ret_concat_images = [np.pad(img, ((0, max_h - h), (0, 0), (0, 0)), mode="constant") for h, img in zip(hs, ret_concat_images)]
    ret_image = np.hstack(ret_concat_images)
    save_filepath = os.path.join(vis_dir, os.path.basename(image_filepath))
    cv2.imwrite(save_filepath, ret_image)
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
    folder = 'show_attentions'
    if not os.path.exists(folder):
        os.makedirs(folder)
    y_trues = dataset.dataset.y_trues
    words_embedding_nor = None
    y_scores = []
    retrieval_image_embedding={}
    for image_id, prediction in enumerate(predictions):
        # print(prediction.fields())
        # import ipdb; ipdb.set_trace()
        path = str(prediction.get_field("path"))
        atts = prediction.get_field("attention")
        boxes = prediction.bbox.data.cpu().numpy()
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        boxes = boxes[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        visualize_attention_image(path,prediction,boxes,folder)
        # cv2.drawContours(image, boxes, -1, color=(255,0,0), thickness=1)

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
    predictions = torch.load("Log/finetune_on_norec_ic17_sn_r18_l4_norc/inference/iiit_test/predictions.pth")
    # predictions = torch.load("predictions_iiit_7916.pth")
    # predictions = torch.load("predictions_temp.pth")
    cfg.merge_from_file('configs/retrival.yaml')
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    # retrieval_results = evaluate_box_proposals(predictions,data_loaders_val[0].dataset)
    retrieval_results = evaluate_box_proposals(predictions,data_loaders_val[0].dataset)
    # np.save("temp.npy",retrieval_results)
    # retrieval_results = np.load("temp.npy",allow_pickle=True).item()
    # # results = np.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/retrieval_results.npy",allow_pickle=True).item()
    # for text, datas in retrieval_results.items():
    #     # dict_ = {data.keys():data.values() for data in datas}
    #     y_trues = datas["y_trues"]
    #     ap = datas["ap"]
    #     del datas["y_trues"]
    #     del datas["ap"]
    #     sorted_paths = sorted(datas.items(), key=lambda datas:datas[1][0], reverse=True)
    #     image_list = [draw_boxes(x[0], x[1][0], x[1][1],x[1][2], x[1][3]) for x in sorted_paths[:48]]
    #     image = vis_multi_image(image_list, shape=[6,-1])
    #     image.save(os.path.join(save_path,"{}_{}_{}.jpg".format(text,y_trues, ap[0])))