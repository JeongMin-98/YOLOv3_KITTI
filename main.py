import torch
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH argments")
    parser.add_argument("--gpus", type=int, nargs="+", default=[], \
                        help="List of GPU device id")
    parser.add_argument("--mode", type=str, help="mode : train / eval",
                        default=None)
    parser.add_argument("--cfg", type=str, help="model config path",
                        default=None)
    parser.add_argument("--checkpoint", type=str, help="model checkpoint path")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def train():
    print("train")


def eval():
    print("eval")


def demo():
    print("demo")


if __name__ == "__main__":
    print("main")

    args = parse_args()

    # cfg parser

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        eval()
    else:
        print("unknown mode")

    print("finished")
