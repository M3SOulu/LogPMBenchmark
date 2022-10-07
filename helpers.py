import string
from typing import Sequence


def split_str_idx(s: str, idx: Sequence[int]) -> Sequence[str]:
    idx = [-1] + list(idx) + [len(s)]
    return [s[i1 + 1: i2] for i1, i2 in zip(idx[:-1], idx[1:])]


def extract_keywords(message: str, mask: str, junk=None):
    if junk is None:
        junk = frozenset(string.whitespace + string.punctuation)
    return frozenset(''.join(cx if cy == '0' and cx not in junk else ' ' for cx, cy in zip(message, mask)).split())
