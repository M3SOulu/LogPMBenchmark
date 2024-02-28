import difflib
import string
from typing import Sequence, Hashable

from benchmark.base_classes import BaseBenchmark
from Brain import Brain
import pandas as pd

from helpers import fix_whitespace_problem

class BrainBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.cid = None
        self.temps = None
    def fit(self, x: Sequence[str]):
        df, _= Brain.parse(x, [], "", 4, [], 0, False, pd.DataFrame())
        self.cid = [eid for eid in df["EventId"]]
        self.temps = [tmp for tmp in df['EventTemplate']]

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return [self.predict_once(e, t) for e, t in zip(x, self.temps)]

    def predict_once(self, x: str, tmp: str) -> str:
        tokens = x.split()
        pattern = tmp.split()
        mask = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pattern, tokens, False).get_opcodes():
            if tag == 'equal':
                for t in tokens[j1:j2]:
                    mask.append('0' * len(t))
            else:
                for t in tokens[j1:j2]:
                    mask.append('1' * len(t))
        mask_str = '0'.join(mask)
        return fix_whitespace_problem(x, mask_str) 