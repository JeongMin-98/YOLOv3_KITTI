import os, sys

import numpy as np
import torch
import torch.nn as nn
from utils.tools import parse_model_config


def add_conv2d_layer(layer_index, modules, layer_info, in_channel, batch_normalizations=True):
    filters = int(layer_info['filters'])
    size = int(layer_info['size'])
    stride = int(layer_info['stride'])
    pad = (size - 1) // 2
    # pad = int(layer_info['pad'])
    modules.add_module('layer_' + str(layer_index) + '_conv',
                       nn.Conv2d(in_channel,
                                 filters,
                                 size,
                                 stride,
                                 pad,
                                 ))
    if layer_info['batch_normalize'] == '1':
        modules.add_module('layer_' + str(layer_index) + '_batch_normalization',
                           nn.BatchNorm2d(filters))
    add_activation_layer(layer_index, modules, layer_info)
    return modules


def add_activation_layer(layer_index, modules, layer_info):
    if layer_info['activation'] == 'leaky':
        modules.add_module('layer_' + str(layer_index) + '_activation',
                           nn.LeakyReLU(0.1, inplace=True))
    elif layer_info['activation'] == 'relu':
        modules.add_module('layer_' + str(layer_index) + '_activation',
                           nn.ReLU())


class YoloLayer(nn.Module):
    def __init__(self, layer_info: dict, in_width: int, in_height: int, is_train: bool):
        super(YoloLayer, self).__init__()
        self.n_classes = int(layer_info['classes'])
        self.ignore_thresh = float(layer_info['ignore_thresh'])
        self.box_attr = self.n_classes + 5  # 5 = bbox[4] + objectness[1]
        mask_idxes = [int(x) for x in layer_info['mask'].split(',')]
        anchor_all = [int(x) for x in layer_info['anchors'].split(',')]
        anchor_all = [(anchor_all[i], anchor_all[i + 1]) for i in range(0, len(anchor_all), 2)]
        self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes])
        self.in_width = in_width
        self.in_height = in_height
        self.stride = None
        self.lw = None
        self.lh = None
        self.training = is_train

    # x is input. [N, C, H, W]
    def forward(self, x):

        self.lw = x.shape[3]
        self.lh = x.shape[2]
        self.anchor = self.anchor.to(x.device)
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode='floor'),
                                    torch.div(self.in_height, self.lh, rounding_mode='floor')])
        self.stride = self.stride.to(x.device)

        # if kitti data. n_classes is 8. C = (8+5) * 3 = 39
        # [batch, box_attrib * anchor, lh, lw] ex) [1, 39, 19, 19]

        # 4dim [batch, box_attrib * ancor, lh, lw] => 5dim [batch, anchor, box_attrib, lh, lw]
        # => [bathc, anchor,lh, lw, box_attrib]
        x = x.view(-1, self.anchor.shape[0], self.box_attr, self.lh, self.lw).permute(0, 1, 3, 4, 2).contiguous()

        return x


class DarkNet53(nn.Module):
    def __init__(self, cfg, param, is_train):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param['in_channels'])
        self.in_width = int(param['in_width'])
        self.in_height = int(param['in_height'])
        self.n_classes = int(param['class'])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YoloLayer)]
        self.training = is_train

    def set_layer(self, config):
        module_list = nn.ModuleList()

        # Channels of initial input
        in_channels = [self.in_channels]
        yolo_idx = 0

        for layer_index, info in enumerate(config):
            modules = nn.Sequential()
            if info['type'] == "convolutional":
                filters = int(info['filters'])
                modules = add_conv2d_layer(layer_index, modules, info, in_channels[-1], True)
                in_channels.append(filters)
            elif info['type'] == 'shortcut':
                modules.add_module('layer_' + str(layer_index) + '_shortcut', nn.Identity())
                in_channels.append(in_channels[-1])
            elif info['type'] == 'route':
                modules.add_module('layer_' + str(layer_index) + '_route', nn.Identity())
                layers = [int(y) for y in info["layers"].split(",")]
                if len(layers) == 1:
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2:
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]+1])
            elif info['type'] == 'upsample':
                modules.add_module('layer_' + str(layer_index) + '_upsample',
                                   nn.Upsample(scale_factor=int(info['stride']), mode='nearest'))
                in_channels.append(in_channels[-1])
            elif info['type'] == 'yolo':
                yolo_layer = YoloLayer(info, self.in_width, self.in_height, self.training)
                modules.add_module('layer_' + str(layer_index) + '_yolo', yolo_layer)
                in_channels.append(in_channels[-1])
            elif info['type'] == 'maxpool':
                pass
            module_list.append(modules)

        return module_list

    def initialize_weight(self):
        """ initialize weight """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        yolo_result = []
        layer_result = []
        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            if name['type'] == "convolutional":
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'shortcut':
                x = x + layer_result[int(name['from'])]
                layer_result.append(x)
            elif name['type'] == 'yolo':
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name['type'] == 'upsample':
                x = layer(x)
                layer_result.append(x)
            elif name['type'] == 'route':
                layers = [int(y) for y in name['layers'].split(',')]
                features = [layer_result[l] for l in layers]
                x = torch.cat(features, dim=1)
                layer_result.append(x)


        return yolo_result
