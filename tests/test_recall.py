from maskrcnn_benchmark.data.datasets.evaluation.retrieval.eval_det.Pascal_VOC import eval_result
import os
input_dir = "Log/use_textness_ic13_15_17_svt_add_textness_weight/inference/svt_test/texts"
allInputs = os.listdir(input_dir)
gt_floder = "./svts"
eval_result(input_dir,allInputs,gt_floder,ignore_difficult=True)