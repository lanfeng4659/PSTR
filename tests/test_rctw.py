from maskrcnn_benchmark.data.datasets.rctw import RCTW
from maskrcnn_benchmark.data.datasets.ctw_test import CTWRetrieval
from maskrcnn_benchmark.data.datasets.ctw_train import CTWTrain
from maskrcnn_benchmark.data.datasets.synthtext_chinese import SynthtextChinese
# dataset = CTWTrain("./datasets/ctw_top100_retrieval/")
dataset = SynthtextChinese("./datasets/SynthText_Chinese/")
for i in range(dataset.len()):
    dataset.getitem(i)
print(dataset.len())