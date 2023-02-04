import itertools
from argparse import Namespace
from typing import Iterable, Sequence, Any, List, Dict


def dict_to_namespace(d: Dict[str, Any]):
    return Namespace(**d)


def product(**kwargs):
    keys: List[str] = list(kwargs.keys())
    values = kwargs.values()
    i = 1
    for item in itertools.product(*values):
        args = dict_to_namespace(dict(zip(keys, item)))
        args.id = i
        yield args
        i += 1
