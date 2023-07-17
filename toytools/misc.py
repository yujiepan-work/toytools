import hashlib
import time
from datetime import datetime, timedelta
from typing import Any
import jstyleson
from pathlib import Path

BASE32_CHARS = "0123456789abcdefghijkmnprstvwxyz"
BASE16_CHARS = "0123456789abcdef"


def get_time(timezone_delta: int = 8):
    return datetime.utcnow() + timedelta(hours=timezone_delta)


def convert_base(number, base_chars=BASE32_CHARS):
    base = len(base_chars)
    result = []
    while number != 0:
        number, i = divmod(number, base)
        result.append(base_chars[i])
    return "".join(result[::-1])


def today_cipher(extra_secs=0, add_date=True, num_chars=4, base_chars=BASE32_CHARS, timezone_delta: int = 8):
    now = get_time(timezone_delta)
    seconds = now.second + now.minute * 60 + now.hour * 3600
    max_seconds = 24 * 3600
    seconds = min(max_seconds, seconds + extra_secs)
    max_new_base = len(base_chars) ** (num_chars) - 1
    h = seconds / max_seconds * max_new_base
    result = convert_base(int(h), base_chars)
    result = result.zfill(num_chars)
    if add_date:
        result = f"{now.month:02d}{now.day:02d}_" + result
    return result


def get_hash(obj: Any) -> str:
    md5 = hashlib.md5(str(obj).encode())
    return md5.hexdigest()


def json_dump(obj, file_path=None, mode='w', temp_folder=None, **kwargs):
    if file_path is None:
        obj_str = jstyleson.dumps(obj)
        cipher = get_hash(obj_str)[-16:]
        if temp_folder is None:
            file_path = f'/tmp/{cipher}.json'
        else:
            file_path = Path(temp_folder, f'{cipher}.json')

    indent = kwargs.pop('indent', 2)

    with open(file_path, mode=mode, encoding='utf-8') as f:
        jstyleson.dump(
            obj, f, indent=indent, **kwargs
        )
    return Path(file_path).absolute()


def json_load(file_path, encoding='utf-8'):
    with open(Path(file_path), 'r', encoding=encoding) as f:
        return jstyleson.load(f)


def set_seed(seed: int = 42):
    """Set seed for `torch`, `numpy` and built-in `random`."""
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError as err:
        pass
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
