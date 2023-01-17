import itertools
from argparse import Namespace


def dict_to_namespace(d: dict):
    return Namespace(**d)


def product(**kwargs):
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    cfgs = []
    for item in itertools.product(*values):
        item = list(item)
        assert len(item) == len(keys)
        d = dict(zip(keys, item))
        cfgs.append(Namespace(**d))
    return cfgs
