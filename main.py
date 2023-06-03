import torch
import argparse
import sys
from utils import tools
from dataloader.yolodata import YoloData
from dataloader.data_transforms import get_transformations
from torch.utils.data.dataloader import DataLoader
from model.yolov3 import *


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


def train(cfg_param=None, using_gpus=None):
    print("train")
    my_transform = get_transformations(cfg_param=cfg_param, is_train=True)
    # data loader
    train_data = YoloData(is_train=True,
                          transform=my_transform,
                          cfg_param=cfg_param)
    train_loader = DataLoader(train_data,
                              batch_size=cfg_param['batch'],
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True,
                              )
    # for i, batch in enumerate(train_loader):
    #     img, targets, anno_path = batch
    #     print("iter {}, img {}, targets {}, anno_path {}".format(i, img.shape, targets.shape, anno_path))
    #
    #     tools.draw_box(img[0].detach().cpu())
    model = DarkNet53(args.cfg, cfg_param, is_train=True)
    for name, param in model.named_parameters():
        print(f"name {name} param {param.shape}")
    # training model
    model.train()
    for i, batch in enumerate(train_loader):
        img, targets, anno_path = batch

        output = model(img)
        print("output len : {}, shape {} {} {}".format(len(output), output[0].shape, output[1].shape, output[2].shape))
        sys.exit(1)


def eval():
    print("eval")


def demo():
    print("demo")


if __name__ == "__main__":
    print("main")

    args = parse_args()

    # cfg parser
    cfg_data = tools.parse_hyperparam_config(args.cfg)
    cfg_param = tools.get_hyperparam(cfg_data)

    print(cfg_data)

    if args.mode == "train":

        train(cfg_param=cfg_param)

    elif args.mode == "eval":
        eval()
    else:
        print("unknown mode")

    print("finished")
