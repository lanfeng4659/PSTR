import torch
import os
import cv2
import numpy as np


# query = '黄焖鸡米饭'
query = '银行'
folder = 'Log/cn_r50_bilinear_fcoscount_pt_b64/finetune/inference_100k/chinese_collect'
predictions = torch.load(os.path.join(folder, 'chinese_collect_predictions.pth'))

# query = '泡面小食堂'
# query = '牛肉面'
# folder = 'Log/cn_r50_bilinear_fcoscount_pt_b64/finetune/inference_100k/rects_test'
# predictions = torch.load(os.path.join(folder, 'rects_test_predictions.pth'))

# query = '中国福利彩票'
# query = '便利'
# folder = 'Log/cn_r50_bilinear_fcoscount_pt_b64/finetune/inference/lsvt_test'
# predictions = torch.load(os.path.join(folder, 'lsvt_test_predictions_0-1500.pth'))
# predictions1 = torch.load(os.path.join(folder, 'lsvt_test_predictions_1500-.pth'))
# predictions.extend(predictions1)


save_folder = os.path.join('tools/paper_images/sub-sequence', query)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
texts = predictions[0].get_field("texts").tolist()
words_embedding_nor = predictions[0].get_field("words_embedding_nor")[texts.index(query)].view(1,-1)

def get_y_scores_trues(predictions):
    y_scores = []
    paths = []
    all_polys = []
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
        sr_num = prediction.get_field("scores").size(0)
        img_embedding = prediction.get_field("imgs_embedding_nor") #[:sr_num]
        paths.append(str(prediction.get_field("path"))) 
        if img_embedding.size(0)==0:
            y_scores.append(torch.zeros([words_embedding_nor.size(0)]).to(img_embedding.device))
            max_idxs.append(torch.zeros([words_embedding_nor.size(0)]).to(img_embedding.device).long())
            all_polys.append(np.array([0]*28))
            continue
        similarity =  words_embedding_nor.mm(img_embedding.t())
        max_scores, max_idx = similarity.max(dim=1)
        y_scores.append(max_scores)
        max_idxs.append(max_idx)
        all_polys.append(polys[max_idx].data.cpu().numpy())
        

    y_scores = torch.stack(y_scores,dim=1).data.cpu().numpy()
    max_idxs = torch.stack(max_idxs,dim=1).data.cpu().numpy()
    return y_scores, all_polys, max_idxs, paths
def save_to_paths(datas):
    for idx, data in enumerate(datas):
        score, poly, path = data
        image = cv2.imread(path)
        cv2.drawContours(image, poly.reshape([-1,14,2]).astype(np.int32), -1, thickness=2, color=(0,255,0))
        cv2.imwrite(os.path.join(save_folder, '{}_{}'.format(idx, os.path.basename(path))), image)
y_scores, all_polys, max_idxs, paths = get_y_scores_trues(predictions)
infos = [(x,y,z) for x,y,z in zip(y_scores[0], all_polys, paths)]
# print(infos)
sorted_paths = sorted(infos, key=lambda infos:infos[0], reverse=True)
save_to_paths(sorted_paths[:100])




    