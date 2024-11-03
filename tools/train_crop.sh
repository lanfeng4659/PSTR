# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/retrival_finetune.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/retrival_syn_svt.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/SiamRPN.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/SiamRPNRetrieval.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/retrival_finetune_usetextness.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/detect.yaml"
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file "configs/retrival.yaml"
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20002 tools/train_net.py --config-file "configs/retrieval_only.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20002 tools/train_net.py --config-file "configs/retrival_attention.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20008 tools/train_net.py --config-file "configs/detect.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20008 tools/train_net.py --config-file "configs/retrival.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20002 tools/train_net.py --config-file "configs/retrival_chinese.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 20008 tools/train_net.py --config-file "configs/retrival_finetune2.yaml"

# sh tools/test.sh
