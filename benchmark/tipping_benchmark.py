import string
from typing import Optional, Sequence, Hashable

from benchmark.base_classes import BaseBenchmark
from tipping import parse


class TippingBenchmark(BaseBenchmark):
    def __init__(self, sensitivity: Optional[float] = None):
        self.masks = None
        self.clusters = None
        self.theta = 0.9 if sensitivity is None else sensitivity

    def fit(self, x: Sequence[str]):
        self.clusters, self.masks, _ = parse(
            x,
            threshold=self.theta,
            symbols=string.punctuation,
            special_whites=[],
            special_blacks=[
                r"\w{2,}://\S+",
                r"[a-zA-Z]+(?:\.[a-zA-Z]+){2,}",
                r"(?<=^|\s)/(?:[^/\s]+/)+[^/\s]*(?=$|\s)",
                r"(?<=^|\s)(?:[^\.\s]+)(?:\.[^\.\s]+){2,}(?=$|\s)",
                r"\(\)",
            ],
            return_templates=False,
            return_masks=True,
        )

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return self.masks

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return self.clusters
