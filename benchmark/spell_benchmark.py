import difflib
import re
from typing import Sequence, Hashable

from benchmark.base_classes import BaseBenchmark
from pyspell.spell import lcsmap, lcsobj


class SpellBenchmark(BaseBenchmark):
    def __init__(self, refmt):
        self.refmt = re.compile(refmt)
        self.spell = lcsmap(refmt)
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
        mask = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pattern, tokens, False).get_opcodes():
            assert tag not in ('insert', 'remove')
            if tag == 'equal':
                for t in tokens[j1:j2]:
                    mask.append('0' * len(t))
            elif tag == 'replace':
                for t in tokens[j1:j2]:
                    mask.append('1' * len(t))
        return '0'.join(mask)
