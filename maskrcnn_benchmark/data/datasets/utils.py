import numpy as np
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
        if query_len >= text_len: continue
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
            # print(i//2, query_len, text, text[(i//2):(i//2)+query_len])
            ret.append([new_span, text[(i//2):(i//2)+query_len]])
    return ret