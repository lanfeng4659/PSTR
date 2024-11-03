export NGPUS=2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20000 tools/train_net.py --config-file "configs/retrival_tsr.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20000 tools/train_net.py --config-file "configs/retrival_tsr_r18.yaml"
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20000 tools/train_net.py --config-file "configs/retrival_finetune2.yaml"
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20000 tools/train_net.py --config-file "configs/retrival_finetune_r18.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/retrival_finetune2.yaml"
sh tools/test.sh
