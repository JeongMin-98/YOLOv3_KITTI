import os, sys
import torch
import torch.optim as optim

from train.loss import YoloLoss
from utils.tools import *


class Trainer:
    def __init__(self, model, train_loader, eval_loader, params, device, torch_writer, checkpoint=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = params['max_batch']
        self.device = device
        self.torch_writer = torch_writer
        self.epoch = 0
        self.iter = 0
        self.yolo_loss = YoloLoss(self.model.n_classes, self.device)
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=params['lr'],
                                   momentum=params['momentum'],
                                   )
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iter = checkpoint['iteration']

        # 실제 학습이 진행될 수록 learning rate가 떨어지도록 해야 학습이 잘된다.
        # iter의 폭에서 어느 지점에서 줄일지를 정하는지를 milestones에서 정하고 비율도 정할 수 있다.
        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[10000, 20000, 30000],
                                                                  gamma=0.5)

    def run_iter(self):
        # drop the batch value when valid values
        for i, batch in enumerate(self.train_loader):
            if batch is None:
                continue
            input_img, targets, anno_path = batch

            input_img = input_img.to(self.device, non_blocking=True)

            output = self.model(input_img)

            # get loss between output and target
            loss, loss_list = self.yolo_loss.compute_loss(output, targets, self.model.yolo_layers)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler_multistep.step(self.iter)
            self.iter += 1

            loss_name = ['total loss', 'obj_loss', 'cls_loss', 'box_loss']

            if i % 10 == 0:
                print("epoch {} / iter {} lr {} loss {}".format(
                    self.epoch,
                    self.iter,
                    get_lr(self.optimizer),
                    loss.item(),
                ))
                self.torch_writer.add_scalar('lr', get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar('total loss', loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)

        return loss

    def run_eval(self):
        # all predictions on eval dataset
        pred = []
        # all ground truth on eval dataset
        gt_labels = []
        for idx, batch in enumerate(self.eval_loader):
            if batch is None:
                continue
            input_img, targets, _ = batch
            input_img = input_img.to(self.device, non_blocking=True)

            gt_labels += targets[..., 1].tolist()

            targets[..., 2:6] = cxcy2minmax(targets[..., 2:6])
            input_width = input_img.shape[3]
            input_height = input_img.shape[2]

            targets[..., 2] *= input_width
            targets[..., 4] *= input_width
            targets[..., 3] *= input_height
            targets[..., 5] *= input_height

            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppresion()

    def run(self):
        while True:
            self.model.train()
            # loss calculation
            loss = self.run_iter()
            self.epoch += 1
            if self.epoch % 50 == 0:
                checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iter,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss
                }, checkpoint_path)

            # evaluate
