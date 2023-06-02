
def parse_hyperparm_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if l and not l.startswith('#')]
