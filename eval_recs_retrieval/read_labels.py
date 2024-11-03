import numpy as np
import codecs
def get_words(gt):
    words = []
    reader = codecs.open(gt,encoding='utf-8').readlines()
    for line in reader:
        parts = line.strip().split(';')
        label = parts[-1]
        words.append(label)
    return words
def get_jump_retrieval_y_trues(dataset, queries):
    query_num = len(queries)
    img_num = dataset.len()
    y_trues = np.zeros([query_num,img_num])
    for i in range(img_num):
        name = dataset.img_lists[i].split('/')[-2]
        for j in range(query_num):
            y_trues[j,i] = isContained(name, queries[j])
    return y_trues
def get_jump_retrieval_y_trues_rects_lsvt(dataset, queries):
    query_num = len(queries)
    img_num = dataset.len()
    y_trues = np.zeros([query_num,img_num])
    for i in range(img_num):
        img_path = dataset.img_lists[i]
        words = get_words(img_path.replace('images', 'labels').replace('.jpg', '.txt'))
        for word in words:
            for j in range(query_num):
                if isContained(word, queries[j])==1:
                    y_trues[j,i]=1
    return y_trues 
            
def isContained(a,b):
    sa,sb = len(a), len(b)
    if sb > sa or sb==0: return 0
    dp = [0]*sb
    idx= [-1]*sb
    dp[0] = b[0] in a
    if dp[0]==0: return 0
    idx[0] = a.index(b[0])
    for i in range(1, sb):
        if dp[i-1] and (b[i] in a[idx[i-1]+1:]):
            dp[i] = 1 
        if b[i] not in a[idx[i-1]+1:]:
            return 0
        idx[i] = a[idx[i-1]+1:].index(b[i]) + idx[i-1] + 1
        # print(b[:i+1], dp[i], idx[i])
    return dp[-1]