import itertools
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence


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


class ConfigProduct:
    id = -1

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __iter__(self):
        yield from [self.__class__(**vars(item)) for item in product(**self._get_class_fields())]

    def __repr__(self) -> str:
        return self._get_instance_fields().__repr__()

    def _get_class_fields(self):
        return {k: list(v) for k, v in self.__class__.__dict__.items()
                if not k.startswith('_') and isinstance(v, Iterable)}

    def _get_instance_fields(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


if __name__ == '__main__':
    class Config(ConfigProduct):
        field1 = [1, 2]
        field2 = 'ab'
        field3 = (divmod, sum, max)

    for cfg in Config():
        print(cfg)
