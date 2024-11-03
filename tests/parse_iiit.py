import os
import scipy.io as scio
import numpy as np
dataFile = "/workspace/wh/projects/FCOSText/datasets/IIIT_STR_V1.0/data.mat"
imgPath = "/workspace/wh/projects/FCOSText/datasets/IIIT_STR_V1.0/imgDatabase"
data = scio.loadmat(dataFile)
images = [img for img in os.listdir(imgPath)]
str_queries = []
for i in range(data['data'].shape[1]):
    str_queries.append(str(data['data'][0,i][0][0][0][0]).lower())
y_trues = np.zeros([len(str_queries),len(images)])
for i in range(len(str_queries)):
    for j in range(len(data['data'][0,i][1])):
        imgName = data['data'][0,i][1][j][0][0]
        y_trues[i,images.index(imgName)] = 1
        if imgName.find("img_")==-1:
            print(imgName)

# print(str_queries)
# print(len(str_queries))