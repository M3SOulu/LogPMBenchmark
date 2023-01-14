from typing import Sequence, Hashable

import optuna
from drain3.drain import Drain

from benchmark.base_classes import BaseBenchmark
from helpers import fix_whitespace_problem


class DrainBenchmark(BaseBenchmark):
    def __init__(self, depth=3, sim_threshold=0.6, max_children=400, max_clusters=None):
        self.drain = Drain(depth=depth,
                           sim_th=sim_threshold,
                           max_children=max_children,
                           max_clusters=max_clusters)
        self.log_clusters = None

    @staticmethod
    def from_trial(trial: optuna.Trial):
        depth = trial.suggest_int('depth', 3, 10)
        sim_th = trial.suggest_float('sim_th', 0.1, 0.95)
        max_ch = trial.suggest_int('max_ch', 100, 1000)
        return DrainBenchmark(depth, sim_th, max_ch)

    def fit(self, x: Sequence[str]):
        for m in x:
            self.drain.add_log_message(m)

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return [self.predict_once(m) for m in x]

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return [self.drain.match(m).cluster_id for m in x]

    def predict_once(self, x: str) -> str:
        mask = '0'.join(('1' if template == '<*>' else '0') * len(token) for token, template in
                            zip(x.split(), self.drain.match(x).get_template().split()))
        if len(x) == len(mask):
            return mask
        mask = fix_whitespace_problem(x, mask)
        assert len(mask) == len(x)
        return mask
