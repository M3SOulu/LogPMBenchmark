import difflib
import re
from typing import Sequence, Hashable

from benchmark.base_classes import BaseBenchmark
from helpers import fix_whitespace_problem
from pyspell.spell import lcsmap, lcsobj


class SpellBenchmark(BaseBenchmark):
    def __init__(self):
        self.refmt = re.compile(r'\s+')
        self.spell = lcsmap(r'\s+')
        self.lcs_objects = None

    def fit(self, x: Sequence[str]):
        self.lcs_objects = [self.spell.insert(m) for m in x]

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return [self.predict_once(m, o) for m, o in zip(x, self.lcs_objects)]

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return [o.get_id() for o in self.lcs_objects]

    def predict_once(self, x: str, match_obj: lcsobj) -> str:
        tokens = self.refmt.split(x)
        pattern = match_obj._lcsseq
        self.spell.match(x)
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
