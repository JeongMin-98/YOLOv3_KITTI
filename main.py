import torch
import argparse
import sys
from utils import tools
from dataloader.yolodata import YoloData
from dataloader.data_transforms import get_transformations
from torch.utils.data.dataloader import DataLoader
from model.yolov3 import *
from train.trainer import Trainer
from tensorboardX import SummaryWriter


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


def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    # skip invalid frames
    if len(batch) == 0:
        return

    imgs, targets, anno_path = list(zip(*batch))

    imgs = torch.stack([img for img in imgs])

    if targets[0] is None or anno_path[0] is None:
        return imgs, None, None

    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets, 0)

    return imgs, targets, anno_path


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
                              drop_last=False,
                              shuffle=False,
                              collate_fn=collate_fn
                              )
    torch_writer = SummaryWriter("./output")

    model = DarkNet53(args.cfg, cfg_param, is_train=True)
    # training model
    model.train()
    model.initialize_weight()

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU")
    else:
        device = torch.device("cpu")
        print("CPU")

    model = model.to(device)

    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=None, params=cfg_param, device=device, torch_writer=torch_writer)
    trainer.run()


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

    if args.mode == "train":

        train(cfg_param=cfg_param)

    elif args.mode == "eval":
        eval()
    else:
        print("unknown mode")

    print("finished")
