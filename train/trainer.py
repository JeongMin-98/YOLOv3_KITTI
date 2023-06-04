import os, sys
import torch
import torch.optim as optim

from train.loss import YoloLoss
from utils.tools import *


class Trainer:
    def __init__(self, model, train_loader, eval_loader, params, device):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = params['max_batch']
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yolo_loss = YoloLoss(self.model.n_classes, self.device)
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=params['lr'],
                                   momentum=params['momentum'],
                                   )

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

            print(f"loss {loss} loss_list {loss_list[0]}")
            print("output - len: {} shape : {}".format(len(output), output[0].shape))

            sys.exit(1)

    def run(self):
        while True:
            self.model.train()
            # loss calculation
            self.run_iter()
            self.epoch += 1
            # if self.epoch % 50 == 0:
            #     checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
            #     torch.save({
            #         'epoch': self.epoch,
            #         'iteration': self.iter,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'loss': loss
            #     }, checkpoint_path)
