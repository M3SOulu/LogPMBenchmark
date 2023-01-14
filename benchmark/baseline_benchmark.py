import random
from typing import Sequence, Hashable

from benchmark.base_classes import BaseBenchmark


class NoParameterBenchmark(BaseBenchmark):

    def fit(self, x: Sequence[str]):
        pass

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return ['0' * len(e) for e in x]

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return [0 for _ in x]


class AllParameterBenchmark(BaseBenchmark):

    def fit(self, x: Sequence[str]):
        pass

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return ['1' * len(e) for e in x]

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return [0 for _ in x]


class RandomParameterBenchmark(BaseBenchmark):

    def fit(self, x: Sequence[str]):
        pass

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return [self.random_mask(len(e)) for e in x]

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return [0 for _ in x]

    @staticmethod
    def random_mask(l):
        return ''.join(random.choice(('0', '1')) for _ in range(l))
