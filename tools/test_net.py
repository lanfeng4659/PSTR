# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import os
import argparse
try:
  import moxing as mox
  mox.file.shift('os', 'mox')
  run_on_remote = True
except:
  run_on_remote = False

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--OUTPUT_DIR",
        default="None",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )	# --config-file "configs/align/line_bezier0732.yaml" 
	# --skip-test \
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR if args.OUTPUT_DIR == 'None' else args.OUTPUT_DIR
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    print(cfg.DATASETS.TEST,args.dataset_id)
    cfg.DATASETS.TEST = (cfg.DATASETS.TEST[args.dataset_id],)
    cfg.MODEL.ALIGN.DET_SCORE = cfg.MODEL.ALIGN.DET_SCORE[args.dataset_id]
    
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON and not cfg.MODEL.KE_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.KE_ON:
        iou_types = iou_types + ("kes",)
    
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    # import ipdb; ipdb.set_trace()
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    rec_type = cfg.MODEL.ALIGN.PREDICTOR
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    best_mAP = 0
    all_lines = []
    if args.ckpt is None:
        assert len(dataset_names)==1, "only support testing one dataset for each time."
        f = open(os.path.join(output_dir,"inference","mAPs_{}.txt".format(dataset_names[0])),"w")
        for ckpt_name in sorted(os.listdir(output_dir),reverse=True):
            if ".pth" not in ckpt_name:
                continue
            # if "80000" in ckpt_name:
            #     continue
        # ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
            ckpt = os.path.join(output_dir, ckpt_name)
            _ = checkpointer.load(ckpt,use_latest=False)
            for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
                
                mAPs, fps, predictions = inference(
                                    model,
                                    data_loader_val,
                                    dataset_name=dataset_name,
                                    iou_types=iou_types,
                                    rec_type = rec_type,
                                    box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                                    device=cfg.MODEL.DEVICE,
                                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                                    output_folder=output_folder,
                                )
                synchronize()
                # mAPs = [mAPs, 0.0]
                # if mAPs[0] > best_mAP:
                #     torch.save(predictions, os.path.join(output_folder, '{}_best_predictions.pth'.format(dataset_name)))
                best_mAP = mAPs[0] if mAPs[0] > best_mAP else best_mAP
                line = "Model:{},full mAP:{},partial mAP:{}, best mAP:{}\n".format(ckpt,mAPs[0],mAPs[1],best_mAP)
                print(line)
                f.write(line)
        f.close()
        # for line in all_lines:
            # f.write(line)
        
    else:
        # ckpt = os.path.join(output_dir, ckpt_name)
        ckpt = args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            mAPs = inference(
                                model,
                                data_loader_val,
                                dataset_name=dataset_name,
                                iou_types=iou_types,
                                rec_type = rec_type,
                                box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                                device=cfg.MODEL.DEVICE,
                                expected_results=cfg.TEST.EXPECTED_RESULTS,
                                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                                output_folder=output_folder,
                            )
            best_mAP = mAPs[0] if mAPs[0] > best_mAP else best_mAP
            synchronize()
            line = "Model:{},full mAP:{},partial mAP:{}, best mAP:{}\n".format(ckpt,mAPs[0],mAPs[1],best_mAP)
            print(line)




if __name__ == "__main__":
    main()
