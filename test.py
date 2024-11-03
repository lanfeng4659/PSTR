from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import os
try:
  import moxing as mox
  mox.file.shift('os', 'mox')
  run_on_remote = True
except:
  run_on_remote = False
# print(run_on_remote, os.listdir("s3://bucket-ocr-beijing4/ymk-wh/datasets/retrieval"))
from PIL import Image
def copy_compiled_to_local():

  # Save compiled .whl to local.
  import glob
  s3_root_dir = 's3://bucket-ocr-beijing4/ymk-wh/codes/wanghao/RetrievalHuaWei/'
  remote_whl_dir = '/home/ma-user/modelarts/user-job-dir/RetrievalHuaWei/libs/apex/dist'
  # remote_whl_paths = ['*.whl']
  # for remote_whl_path in remote_whl_paths:
  #     remote_whl_path = glob.glob(os.path.join(remote_whl_dir, remote_whl_path))[0]
  #     remote_whl_name = os.path.basename(remote_whl_path)
  #     mox.file.copy(remote_whl_path, os.path.join(s3_root_dir, remote_whl_name))
  # Save compiled .so to local.
  import glob
  s3_root_dir = 's3://bucket-ocr-beijing4/ymk-wh/codes/wanghao/RetrievalHuaWei/'
  remote_root_dir = '/home/ma-user/modelarts/user-job-dir/RetrievalHuaWei/'
  # remote_so_paths = ['maskrcnn_benchmark/*.so']
  mox.file.copy_parallel(os.path.join(remote_root_dir, 'build'), os.path.join(s3_root_dir,'build'))
  mox.file.copy_parallel(os.path.join(remote_root_dir, 'maskrcnn_benchmark.egg-info'), os.path.join(s3_root_dir,'maskrcnn_benchmark.egg-info'))
  mox.file.copy_parallel(os.path.join(remote_root_dir, 'maskrcnn_benchmark'), os.path.join(s3_root_dir,'maskrcnn_benchmark'))
  # for remote_so_path in remote_so_paths:
  #     print("finding", glob.glob(os.path.join(remote_root_dir, remote_so_path)))
  #     remote_so_path = glob.glob(os.path.join(remote_root_dir, remote_so_path))[0]
  #     s3_so_path = os.path.join(s3_root_dir, remote_so_path.replace(remote_root_dir, ''))
  #     mox.file.copy(remote_so_path, s3_so_path)

if run_on_remote:
    # online compile detectron2
    def exist_delete(path):
        if os.path.exists(path):
            os.system('rm -r {0}'.format(path))
    work_dir = '/home/ma-user/modelarts/user-job-dir/RetrievalHuaWei/'
    os.environ['MAKEFLAGS'] = "-j16"
    cmd = 'cd {0} && python setup.py build_ext install'.format(os.path.join(work_dir, "libs/cocoapi/PythonAPI"))
    os.system(cmd)

    cmd = 'cd {0} && python setup.py install --cuda_ext --cpp_ext'.format(os.path.join(work_dir, "libs/apex"))
    # cmd = 'cd {0} && python setup.py bdist_wheel'.format(os.path.join(work_dir, "libs/apex"))
    os.system(cmd)

    # online compile AdelaiDet
    exist_delete(os.path.join(work_dir, 'build'))
    cmd = 'cd {0} && python setup.py build develop'.format(work_dir)
    os.system(cmd)

    # copy_compiled_to_local()

import argparse
import os.path as osp

def parse_args():
  parser = argparse.ArgumentParser(description='runner for ABCNetv2')
  parser.add_argument(
      "--pretrain-config",
      default="",
      metavar="FILE",
      help="path to config file",
      type=str,
  )
  parser.add_argument(
      "--finetune-config",
      default="",
      metavar="FILE",
      help="path to config file",
      type=str,
  )
  parser.add_argument(
      "--weight",
      default="",
      metavar="FILE",
      help="path to config file",
      type=str,
  )  
  parser.add_argument(
      "--data_url",
      default="",
      help="path to datasets",
      type=str,
  )
  parser.add_argument(
      "--output-dir",
      default="",
      help="path to logs",
      type=str,
  )
  parser.add_argument(
      "--num-gpus",
      help="number of gpus",
      type=int,
  )

  # args = parser.parse_args()
  # modelarts will transfer some extra parameters
  args, unknown = parser.parse_known_args()

  cur_path = os.path.abspath(__file__)
  cur_dir  = os.path.dirname(cur_path)
  run_file_path = os.path.join(cur_dir, 'tools/train_net.py')

  args.run_file_path = run_file_path
  return args

def main():
  args = parse_args()
  if not run_on_remote:
    print(args)
    cmd_pretrain = "CUDA_VISIBLE_DEVICES=3,2 python -m torch.distributed.launch --nproc_per_node={0} --master_port 20000 tools/train_net.py \
           --config-file {1} \
           --OUTPUT_DIR {2}/pretrain \
           --WEIGHT {3} \
           --skip-test".format( \
                              args.num_gpus,
                              args.pretrain_config,
                              args.output_dir,
                              args.weight)
    cmd_model_process = "python tools/model_del.py --input-file {0}/pretrain/model_final.pth --output-file {0}/model_pretrain.pth".format(args.output_dir)
    cmd_finetune = "CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node={0} --master_port 20000 tools/train_net.py \
          --config-file {1} \
          --OUTPUT_DIR {2}/finetune \
          --WEIGHT {2}/model_pretrain.pth \
           --skip-test".format( \
                              args.num_gpus,
                              args.finetune_config,
                              args.output_dir)
  else:
    # firstly, load all big files on S3 to local ssd /cache/xxx.

    # cache_dataset_path = '/cache/datasets'
    # textocr_folders = ["TextOCR/train_images", "TextOCR/train_gts", "TextOCR/val_gts", \
    #   "TextOCR/openimages/boxes"]
    # textocr_files = ["TextOCR/objects/visual_genome_categories.json"]
    # for textocr_folder in textocr_folders:
    #   mox.file.copy_parallel(osp.join(args.datasets_path, textocr_folder), osp.join(cache_dataset_path, textocr_folder))
    # for textocr_file in textocr_files:
    #   mox.file.copy(osp.join(args.datasets_path, textocr_file), osp.join(cache_dataset_path, textocr_file))
    
    # cache_dataset_path = '/cache/datasets/retrieval_chinese'
    # mox.file.copy_parallel(args.datasets_path, cache_dataset_path)
    # datasets = ['LSVT','ReCTS','chinese_collect']
    # datasets = ['totaltext']
    # for data in datasets:
        # mox.file.copy_parallel(os.path.join(args.datasets_path, data), os.path.join(cache_dataset_path, data))
    # args.datasets_path = cache_dataset_path


    cache_weight_path = '/cache/%s' % os.path.basename(args.weight) 
    mox.file.copy(args.weight, cache_weight_path)
    args.weight = cache_weight_path

    # prepare environment. Including one-time compile (.so, .whl or pip install).
    cur_path = os.path.abspath(__file__)
    cur_dir  = os.path.dirname(cur_path)
    run_file_path = os.path.join(cur_dir, 'tools/train_net.py')
    # run the codes with distributed training.
    cmd_pretrain = "python -m torch.distributed.launch --nproc_per_node={0} --master_port 20000 {1} \
           --config-file {2} \
           --OUTPUT_DIR {3}/pretrain \
           --WEIGHT {4} \
           --skip-test".format( \
                              args.num_gpus,
                              run_file_path,
                              args.pretrain_config,
                              args.output_dir,
                              args.weight)
    cmd_model_process = "python {0} --input-file {1}/pretrain/model_final.pth --output-file {1}/model_pretrain.pth".format(os.path.join(cur_dir, 'tools/model_del.py'), args.output_dir)
    cmd_finetune = "python -m torch.distributed.launch --nproc_per_node={0} --master_port 20000 {1} \
          --config-file {2} \
          --OUTPUT_DIR {3}/finetune \
          --WEIGHT {3}/model_pretrain.pth \
           --skip-test".format( \
                              args.num_gpus,
                              run_file_path,
                              args.finetune_config,
                              args.output_dir)
    # cmd_finetune = "python -m torch.distributed.launch --nproc_per_node={0} --master_port 20000 {1} \
    #       --config-file {2} \
    #       --OUTPUT_DIR {3}/finetune2 \
    #       --WEIGHT {4} \
    #        --skip-test".format( \
    #                           args.num_gpus,
    #                           run_file_path,
    #                           args.finetune_config,
    #                           args.output_dir,
    #                           args.weight)
  
  if run_on_remote:
    os.system('cd /home/ma-user/modelarts/user-job-dir/RetrievalHuaWei/')


  cur_path = os.path.abspath(__file__)
  cur_dir  = os.path.dirname(cur_path)
  run_file_path = os.path.join(cur_dir, 'tools/test_net.py')
  ours_test = "CUDA_VISIBLE_DEVICES=0 python {0} --config-file {1} \
   --OUTPUT_DIR {2}/finetune \
    --dataset_id {3}".format(run_file_path, args.finetune_config, args.output_dir, 0) # iiit
  lsvt_test = "CUDA_VISIBLE_DEVICES=0 python {0} --config-file {1} \
   --OUTPUT_DIR {2}/finetune \
    --dataset_id {3}".format(run_file_path, args.finetune_config, args.output_dir, 1) # iiit
  rects_test = "CUDA_VISIBLE_DEVICES=0 python {0} --config-file {1} \
   --OUTPUT_DIR {2}/finetune \
    --dataset_id {3}".format(run_file_path, args.finetune_config, args.output_dir, 5) # iiit
  # tt_test = "CUDA_VISIBLE_DEVICES=0 python {0} --config-file {1} \
  #  --OUTPUT_DIR {2}/finetune \
  #   --dataset_id {3}".format(run_file_path, args.finetune_config, args.output_dir, 3) # iiit
  # os.system(ours_test)
  os.system(rects_test)
  # os.system(lsvt_test)
  # os.system(tt_test)


if __name__ == '__main__':
  main()
