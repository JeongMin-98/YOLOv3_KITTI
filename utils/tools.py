"""
    강의 내용 참고하여 작성
"""

def parse_hyperparam_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

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
