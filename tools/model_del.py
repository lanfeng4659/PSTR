import os
try:
  import moxing as mox
  mox.file.shift('os', 'mox')
  run_on_remote = True
except:
  run_on_remote = False
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--input-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--output-file",
        default="None",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    model = torch.load(args.input_file)
    del model['optimizer']
    del model['iteration']
    del model['scheduler']
    torch.save(model, args.output_file)



if __name__ == "__main__":
    main()