PRETRAIN_CFG="configs/retrival_tsr_r50bn.yaml"
FINETUNE_CFG="configs/retrival_finetune_r50bn.yaml"
folder="Log/scalenet_r50bn_neckl4_norec"
# PRETRAIN_CFG="configs/retrival_tsr_r18.yaml"
# FINETUNE_CFG="configs/retrival_finetune_r18.yaml"
# pretraining phase
export NGPUS=2
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20000 tools/train_net.py --config-file ${PRETRAIN_CFG} --OUTPUT_DIR "${folder}/pretrain" --WEIGHT "/home/ymk-wh/workspace/researches/wanghao/retrieval-models/resnet50-19c8e357.pth" --skip-test

# # delete parameters
# python tools/model_del.py --input-file "${folder}/pretrain/model_final.pth" --output-file "${folder}/model_pretrain.pth" 

# # finetuning phase
# export NGPUS=2
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20000 tools/train_net.py --config-file ${FINETUNE_CFG} --OUTPUT_DIR "${folder}/finetune" --WEIGHT "${folder}/model_pretrain.pth" --skip-test

# # testing phase
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file ${FINETUNE_CFG} --OUTPUT_DIR "${folder}/finetune" --dataset_id 0 # iiit
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file ${FINETUNE_CFG} --OUTPUT_DIR "${folder}/finetune" --dataset_id 1 # coco
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file ${FINETUNE_CFG} --OUTPUT_DIR "${folder}/finetune" --dataset_id 2 # svt
# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file ${FINETUNE_CFG} --OUTPUT_DIR "${folder}/finetune" --dataset_id 3 # totaltext

# CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/retrival_finetune_r50bn.yaml --OUTPUT_DIR "Log/scalenetnocc_r50bn_neckl4_norec_poly/finetune_add_tt" --dataset_id 3