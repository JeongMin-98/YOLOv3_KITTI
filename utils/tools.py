"""
    강의 내용 참고하여 작성
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def read_config(path):
    """ read config files """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    return lines


def parse_model_config(path):
    """ Parses the yolo-v3 layer configuration file and returns module defines"""
    lines = read_config(path)
    module_defs = []

    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name == "net":
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name

            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name == "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_hyperparam_config(path):
    lines = read_config(path)

    module_defs = []

    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name != "net":
                continue
            module_defs.append({})
            module_defs[-1]['type'] = type_name

            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if type_name != "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def get_hyperparam(cfg):
    for c in cfg:
        if c['type'] == 'net':
            batch = int(c['batch'])
            subdivision = int(c['subdivisions'])
            momentum = float(c['momentum'])
            decay = float(c['decay'])
            saturation = float(c['saturation'])
            hue = float(c['hue'])
            exposure = float(c['exposure'])
            lr = float(c['learning_rate'])
            burn_in = int(c['burn_in'])
            max_batch = int(c['max_batches'])
            lr_policy = c['policy']
            steps = [int(x) for x in c['steps'].split(',')]
            scales = [float(x) for x in c['scales'].split(',')]
            in_width = int(c['width'])
            in_height = int(c['height'])
            in_channels = int(c['channels'])
            _class = int(c['class'])
            ignore_cls = int(c['ignore_cls'])

            return {'batch': batch,
                    'subdivision': subdivision,
                    'momentum': momentum,
                    'decay': decay,
                    'saturation': saturation,
                    'hue': hue,
                    'exposure': exposure,
                    'lr': lr,
                    'burn_in': burn_in,
                    'max_batch': max_batch,
                    'lr_policy': lr_policy,
                    'steps': steps,
                    'scales': scales,
                    'in_width': in_width,
                    'in_height': in_height,
                    'in_channels': in_channels,
                    'class': _class,
                    'ignore_cls': ignore_cls}
        else:
            continue


def xywh2xyxy_np(x):
    """
    Input value
    1. x[...,0] = center_x
    2. x[...,1] = center_y
    3. x[...,2] = weight
    4. x[...,3] = height

    Output Value
    minx, miny, maxx, maxy
    """
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # minx
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # miny
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # maxx
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # maxy

    return y


def draw_box(img):
    img = img * 255

    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1, 2, 0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)
    # elif img.ndim == 2:
    #     img_data = np.array(img, dtype=np.uint8)
    #     img_data = Image.fromarray(img_data, 'L')

    draw = ImageDraw.Draw(img_data)
    fontsize = 15
    # font = ImageFont.truetype("./arial.ttf", fontsize)
    plt.imshow(img_data)
    plt.show()
