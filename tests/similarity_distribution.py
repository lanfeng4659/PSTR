import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from maskrcnn_benchmark.data.datasets.synthtext90k import Synthtext90k
# from maskrcnn_benchmark.utils.text_util import TextGenerator
# dataset = Synthtext90k("./datasets/SynthText_90KDict")
# generator = TextGenerator()
def save_excel(data):
    import xlwt
    book = xlwt.Workbook() 
    sheet = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
    
    sheet.write(0,0,'Series1')
    sheet.write(0,1,'Series2')
    print(len(data[0]))
    for i, (v1,v2) in enumerate(zip(data[0], data[1])):
        sheet.write(i,0,v1)
        sheet.write(i,1,v2)
    book.save('Excel_Workbook.xls')
similarity = []
similarity_wa = []
regions = 100
def similarity_region(x):
    nums = [0]*regions
    step = 1./regions
    # print(x)
    # x = np.array(x).reshape(-1)
    # print(x.shape)(x > i*step)*(x < (i+1)*step)
    for i in range(regions):
        nums[i] = sum([1 for y in x if (y>i*step and y<(i+1)*step)])
    return nums
def hard_ming(x):
    y = []
    m,n = x.shape
    # sorted(x,axis=1,reverse=False)
    for i in range(m):
        y.extend(sorted(x[i].tolist(),reverse=True)[:n//4])
    return y
# for i in range(1000):
#     texts = []
#     for j in range(10):
#         texts.extend(dataset.getitem(i)[2])
#     similarity.extend(generator.calculate_similarity_matric(texts,texts).reshape(-1).tolist())
#     texts_ag = texts
#     texts_ag.extend([generator(text) for text in texts])
#     similarity_wa.extend(hard_ming(generator.calculate_similarity_matric(texts,texts_ag)))
# np.save("similarity.npy",similarity)
# np.save("similarity_wa.npy",similarity_wa)
similarity = np.load("similarity.npy").tolist()
similarity_wa = np.load("similarity_wa.npy").tolist()
# print(plt.hist([x for x in similarity if x>0 and x<1], bins=20))
# plt.show()
# plt.savefig("similarity.png")
# plt.close()
# print(plt.hist([x for x in similarity_wa if x>0 and x<1], bins=20))

#48 151 82 43 80 160 161 162
# kwargs = dict(histtype='stepfilled', bins=20)
step = 1.0/10
excel_data = []
similarity = [x for x in similarity if x>0 and x<1]
# print(len([x for x in similarity if x<0.2])*1.0/len(similarity))
results, edges = np.histogram(similarity, normed=True, bins=np.arange(0,1,step))
binWidth = edges[1] - edges[0]
# print(results*binWidth)
plt.rc('font',family='Times New Roman')
fig = plt.figure(figsize=(7,4))
# excel_data.append(results*binWidth)
excel_data.append(similarity)
plt.bar(edges[:-1], results*binWidth, binWidth,alpha=0.4, color='lightcoral')

similarity_wa = [x for x in similarity_wa if x>0 and x<1]
# print(similarity_wa)
results, edges = np.histogram(similarity_wa, normed=True, bins=np.arange(0,1,step))
binWidth = edges[1] - edges[0]
# print(results*binWidth)
# excel_data.append(results*binWidth)
excel_data.append(similarity_wa)
# print(edges[0:],results*binWidth)
plt.bar(edges[:-1], results*binWidth, binWidth,alpha=0.4, color='green')
# plt.hist(similarity, color='lightcoral',weights=weights, **kwargs)
# plt.hist([x for x in similarity_wa if x>0 and x<1], color='green', **kwargs)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Similarity",fontsize=15)
# plt.ylabel("Probability",fontsize=15)
plt.ylabel("Proportion",fontsize=15)

plt.show()
plt.savefig("similarity_wa.png",bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
save_excel(excel_data)