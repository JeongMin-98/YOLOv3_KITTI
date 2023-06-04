import pdb

import torch
import torch.nn as nn
from utils.tools import *
import os, sys


class YoloLoss(nn.Module):

    def __init__(self, num_class, device):
        super(YoloLoss, self).__init__()
        self.num_class = num_class
        self.device = device
        self.bcellogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device)).to(self.device)

    def compute_loss(self, pred, target, yolo_layer):
        """
        input resolution 608 * 608
        the number of boxes in each yolo layer is according to input resolution
        pout shape : [batch, anchor, grid_h, grid_w, box_attrib]
        the number of boxes in each yolo layer = anchor * grid_h * grid_w
        yolo0 -> 3 * 19 * 19, yolo1 -> 3 * 38 * 38, yolo2 -> 3 * 76 * 76
        total boxes : 22743.

        positive prediction vs negative prediction
        pos : neg = 0.01 : 0.99

        Only using positive prediction, get box_loss and class_loss.

        When using negative prediction, only cal obj_loss.

        """
        loss_class = torch.zeros(1, device=self.device)
        loss_object = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)

        # get positive targets
        target_class, target_boxes, target_indices, target_anchor = self.get_targets(pred, target, yolo_layer)
        # 3 yolo layer
        for pred_idx, pred_out in enumerate(pred):
            batch_id, anchor_id, gy, gx = target_indices[pred_idx]

            target_object = torch.zeros_like(pred_out[..., 0], device=self.device)
            num_targets = batch_id.shape[0]
            if num_targets:
                # box attrib
                ps = pred_out[batch_id, anchor_id, gy, gx]
                pxy = torch.sigmoid(ps[..., 0:2])
                pwh = torch.exp(ps[..., 2:4]) * target_anchor[pred_idx]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, target_boxes[pred_idx], x1y1x2y2=False, device=self.device)
                loss_box += (1 - iou).mean()
                target_object[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(target_object.dtype)

                if ps.size(1) - 5 >= 1:
                    t = torch.zeros_like(ps[..., 5:], device=self.device)
                    t[range(num_targets), target_class[pred_idx]] = 1
                    loss_class += self.bcellogloss(ps[:, 5:], t)

            loss_object += self.bcellogloss(pred_out[..., 4], target_object)

        loss_class *= 0.05
        loss_object *= 1.0
        loss_box *= 0.5

        loss = loss_class + loss_object + loss_box
        loss_list = [loss.item(), loss_object.item(), loss_class.item(), loss_box.item()]

        del loss_class
        del loss_object
        del loss_box

        return loss, loss_list

    def get_targets(self, pred, targets, yolo_layer):
        num_anc = 3
        num_target = targets.shape[0]
        target_class, target_boxes, target_indices, target_anchor = [], [], [], []

        gain = torch.ones(7, device=self.device)

        anchor_index = torch.arange(num_anc, device=targets.device).float().view(num_anc, 1).repeat(1, num_target)
        targets = torch.cat((targets.repeat(num_anc, 1, 1), anchor_index[:, :, None]), 2).to(self.device)
        del anchor_index
        for yi, yl in enumerate(yolo_layer):
            anchors = yl.anchor / yl.stride
            gain[2:6] = torch.tensor(pred[yi].shape)[[3, 2, 3, 2]]

            t = targets * gain

            if num_target:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4
                t = t[j]
            else:
                t = targets[0]

            b, c = t[:, :2].long().T

            gxy = t[:, 2:4]
            gwh = t[:, 4:6]

            gij = gxy.long()

            gi, gj = gij.T

            a = t[:, 6].long()

            target_indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
            target_boxes.append(torch.cat((gxy - gij, gwh), 1))
            target_anchor.append(anchors[a])
            target_class.append(c)

        del t
        del targets
        del gain

        return target_class, target_boxes, target_indices, target_anchor
