import hashlib
from datetime import datetime, timedelta
from typing import Any
import time

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


def today_cipher(num_chars=4, base_chars=BASE32_CHARS, timezone_delta: int = 8):
    now = get_time(timezone_delta)
    seconds = now.second + now.minute * 60 + now.hour * 3600
    max_seconds = 24 * 3600
    max_new_base = len(base_chars) ** (num_chars) - 1
    h = seconds / max_seconds * max_new_base
    return convert_base(int(h), base_chars)


def get_hash(obj: Any) -> str:
    md5 = hashlib.md5(str(obj).encode())
    return md5.hexdigest()
