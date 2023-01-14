import re
import string
from typing import Sequence, Hashable

import optuna

from benchmark.base_classes import BaseBenchmark
from lenma.lenma_template import LenmaTemplateManager


class LenmaBenchmark(BaseBenchmark):
    def __init__(self, threshold=0.9):
        self.template_manager = LenmaTemplateManager(threshold)
        self.inferred_templates = None

    def fit(self, x: Sequence[str]):
        self.inferred_templates = [self.template_manager.infer_template(self.words(t), i) for i, t in enumerate(x)]

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        assert self.inferred_templates and len(self.inferred_templates) == len(self.inferred_templates)
        masks = []
        for message, tmp in zip(x, self.inferred_templates):
            mask = ['1'] * len(message)

            # set all spaces to zero
            for idx in re.finditer(r'\s', message):
                mask[idx.start()] = '0'

            # set all words to zero
            for word in tmp.words:
                for match in re.finditer(f'{re.escape(word)}', message):
                    for idx in range(match.start(), match.end()):
                        mask[idx] = '0'
            masks.append(''.join(mask))

        return masks

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return [0]

    @staticmethod
    def from_trial(trial: optuna.Trial):
        th = trial.suggest_float('threshold', 0.05, 0.95)
        return LenmaBenchmark(th)

    @staticmethod
    def words(line: str):
        it = line.strip().split()
        it = (w for w in it)
        it = (w for w in it if not any(c in string.digits for c in w))
        lst = list(it)
        if lst:
            return lst
        return line.strip().split()
