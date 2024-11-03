import torch
import os
from maskrcnn_benchmark.data.datasets.lsvt import LSVTDataset
import cv2
import numpy as np
def draw_image(image, polys, polys_partial, save_path):
    polys = np.array(polys).reshape([-1,14,2]).astype(np.int32)
    # cv2.drawContours(image, polys, -1, thickness=2, color=(0,255,0))
    for poly in polys_partial:
        poly = poly.reshape([1,-1,2]).astype(np.int32)
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        cv2.drawContours(image, poly, -1, thickness=3, color=color)
    cv2.imwrite(save_path, image)
def sample_partial_polys(polys_partial, cover_len=2):
    polys = np.array(polys_partial).reshape([-1,2])
    # print(polys[(0,1,2,3),:])
    point_num = len(polys)//2
    cover_len = cover_len*2
    new_polys = [np.concatenate([polys[i_start:i_start+cover_len+1] ,polys[point_num*2 - i_start-cover_len-1:point_num*2 - i_start]])for i_start in range(len(polys)-cover_len)]
    # for poly in new_polys:
        # print(poly.shape)
    return new_polys
def generate_partial_proposals(poly, query_len, text_len=10, num_point=7):
    poly = poly.reshape([-1,2])
    point_num = len(poly)//2
    inter_up_x = (poly[point_num-1, 0] - poly[0, 0]) / (text_len*2)
    inter_up_y = (poly[point_num-1, 1] - poly[0, 1]) / (text_len*2)
    inter_do_x = (poly[point_num, 0] - poly[-1, 0]) / (text_len*2)
    inter_do_y = (poly[point_num, 1] - poly[-1, 1]) / (text_len*2)

    span_up_x = np.linspace(poly[0, 0], poly[0, 0]+inter_up_x*(query_len*2), num_point)
    span_up_y = np.linspace(poly[0, 1], poly[0, 1]+inter_up_y*(query_len*2), num_point)
    span_up = np.stack((span_up_x, span_up_y),axis=1)
    span_do_x = np.linspace(poly[-1, 0] + inter_do_x*(query_len*2), poly[-1, 0], num_point)
    span_do_y = np.linspace(poly[-1, 1] + inter_do_y*(query_len*2), poly[-1, 1], num_point)
    span_do = np.stack((span_do_x, span_do_y),axis=1)

    span = np.concatenate((span_up, span_do), axis=0)
    ret = [span]
    for i in range(1,(text_len-query_len)*2+1):
        if i%2 == 1:
            continue
        new_span = np.zeros_like(span)
        new_span[:num_point,0] = span[:num_point,0] + inter_up_x*i
        new_span[:num_point,1] = span[:num_point,1] + inter_up_y*i
        new_span[num_point:,0] = span[num_point:,0] + inter_do_x*i
        new_span[num_point:,1] = span[num_point:,1] + inter_do_y*i

        ret.append(new_span)

    return ret
def generate_partial_proposals_labels(poly, text, query_lens=[2,3], num_point=7):
    text_len = len(text)
    poly = poly.reshape([-1,2])
    point_num = len(poly)//2
    inter_up_x = (poly[point_num-1, 0] - poly[0, 0]) / (text_len*2)
    inter_up_y = (poly[point_num-1, 1] - poly[0, 1]) / (text_len*2)
    inter_do_x = (poly[point_num, 0] - poly[-1, 0]) / (text_len*2)
    inter_do_y = (poly[point_num, 1] - poly[-1, 1]) / (text_len*2)

    ret = []
    for query_len in query_lens:
        span_up_x = np.linspace(poly[0, 0], poly[0, 0]+inter_up_x*(query_len*2), num_point)
        span_up_y = np.linspace(poly[0, 1], poly[0, 1]+inter_up_y*(query_len*2), num_point)
        span_up = np.stack((span_up_x, span_up_y),axis=1)
        span_do_x = np.linspace(poly[-1, 0] + inter_do_x*(query_len*2), poly[-1, 0], num_point)
        span_do_y = np.linspace(poly[-1, 1] + inter_do_y*(query_len*2), poly[-1, 1], num_point)
        span_do = np.stack((span_do_x, span_do_y),axis=1)

        span = np.concatenate((span_up, span_do), axis=0)
        ret.append([span, text[:query_len]])
        for i in range(1,(text_len-query_len)*2+1):
            if i%2 == 1:
                continue
            new_span = np.zeros_like(span)
            new_span[:num_point,0] = span[:num_point,0] + inter_up_x*i
            new_span[:num_point,1] = span[:num_point,1] + inter_up_y*i
            new_span[num_point:,0] = span[num_point:,0] + inter_do_x*i
            new_span[num_point:,1] = span[num_point:,1] + inter_do_y*i
            print(i//2, query_len, text, text[(i//2):(i//2)+query_len])
            ret.append([new_span, text[(i//2):(i//2)+query_len]])

    return ret

dataset = LSVTDataset(os.path.join('./datasets', 'LSVT'), augment='PSSAugmentation')
for (image, boxlist, idx) in dataset:
    if idx !=8 :
        continue
    image = np.ascontiguousarray(np.array(image)[:,:,(2,1,0)]) 
    polys = boxlist.get_field('polys')[2]
    texts = boxlist.get_field('texts')[2]
    print(polys.shape)
    # polys_partial = boxlist.get_field('polys_partial')[1]
    # print(polys_partial)
    # new_polys = sample_partial_polys(polys_partial, cover_len=4)
    # new_polys = generate_partial_proposals(np.array(polys), query_len=4, text_len=12)
    new_polys_texts = generate_partial_proposals_labels(np.array(polys), query_lens=[2,3], text=texts)
    new_polys = [v[0] for v in new_polys_texts]
    new_texts = [v[1] for v in new_polys_texts]
    print(len(new_polys), new_texts)
    draw_image(image, polys,new_polys, 'temp.jpg')