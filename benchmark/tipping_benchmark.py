import string
from typing import Sequence, Hashable

from benchmark.base_classes import BaseBenchmark
from tipping import token_independency_clusters


class TippingBenchmark(BaseBenchmark):
    def __init__(self):
        self.masks = None
        self.clusters = None

    def fit(self, x: Sequence[str]):
        self.clusters, self.masks, _ = token_independency_clusters(
            x,
            threshold=0.9,
            symbols=string.punctuation,
            special_whites=[],
            special_blacks=[
                r"\w{2,}://\S+",
                r"[a-zA-Z]+(?:\.[a-zA-Z]+){2,}",
                r"(?<=^|\s)/(?:[^/\s]+/)+[^/\s]*(?=$|\s)",
                r"(?<=^|\s)(?:[^\.\s]+)(?:\.[^\.\s]+){2,}(?=$|\s)",
                r"\(\)"
            ],
            return_templates=False,
            return_masks=True,
        )

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        return self.masks

    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        return self.clusters
