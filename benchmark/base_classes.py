import string
from abc import abstractmethod
from typing import Sequence, Hashable, Dict, Callable

import optuna
import pandas as pd
import scipy
from sklearn.metrics.cluster import adjusted_rand_score
import timeit

import numpy as np

from dataset import Dataset
from helpers import split_str_idx


class BaseBenchmark:

    def benchmark(self, dataset: Dataset) -> Dict[str, float]:
        # fit to dataset
        before_fit = timeit.default_timer()
        self.fit(dataset.x)
        after_fit = timeit.default_timer()

        # predict parsing masks
        before_predict_mask = timeit.default_timer()
        y_hat = self.predict_mask(dataset.x)
        after_predict_mask = timeit.default_timer()
        print(f"Training {self.__class__.__name__} finished")

        # # convert y and y_hat to cluster IDs
        # before_predict_cluster = timeit.default_timer()
        # y_hat_cluster = self.predict_cluster(dataset.x)
        # after_predict_cluster = timeit.default_timer()

        # compute accuracies
        # pre_temp_acc, rec_temp_acc = template_accuracy(y_hat, dataset.y, dataset.x)
        # p_acc_char = parsing_accuracy_character(y_hat, dataset.y)
        p_acc_token = parsing_accuracy_token(y_hat, dataset.y, dataset.x)
        print(f"Parsing Accuracy Finished {self.__class__.__name__} finished")
        mean_ma = mean_mask_agreement(y_hat, dataset.y)
        print(f"Mask Agreement Finished {self.__class__.__name__} finished")
        # r_score = rand_score(y_hat_cluster, dataset.c)
        # precision, recall, f1_score, group_accuracy = logpai_metrics(y_hat_cluster, dataset.c)

        return {
            'mean_mask_agreement': mean_ma,
            # 'character_parsing_accuracy': p_acc_char,
            'token_parsing_accuracy': p_acc_token,
            # 'precision_template_accuracy': pre_temp_acc,
            # 'recall_template_accuracy': rec_temp_acc,
            # 'rand_score': r_score,
            # 'precision': precision,
            # 'recall': recall,
            # 'f1 score': f1_score,
            # 'group_accuracy': group_accuracy,
            'fit_duration': after_fit - before_fit,
            'predict_mask_duration': after_predict_mask - before_predict_mask,
            'total_duration': (after_fit - before_fit) + (after_predict_mask - before_predict_mask),
            # 'predict_cluster_duration': after_predict_cluster - before_predict_cluster
        }

    def tune(self, dataset: Dataset, criterion='mean_mask_agreement'):
        study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        study.optimize(self.optim_target(dataset, criterion))

    @abstractmethod
    def fit(self, x: Sequence[str]):
        """

        :param x:
        :return:
        """

    @abstractmethod
    def predict_mask(self, x: Sequence[str]) -> Sequence[str]:
        """

        :param x:
        :return:
        """

    @abstractmethod
    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]:
        """

        :param x:
        :return:
        """

    @staticmethod
    @abstractmethod
    def from_trial(trial: optuna.Trial):
        """

        :param trial:
        :return:
        """

    @classmethod
    def optim_target(cls, ds: Dataset, criterion: str) -> Callable[[optuna.Trial], float]:
        def func(trial: optuna.Trial):
            b = cls.from_trial(trial)
            return b.benchmark(ds)[criterion]

        return func


def parsing_accuracy_character(y_hat: Sequence[str], y: Sequence[str]) -> float:
    assert len(y_hat) == len(y)
    return np.mean([mask_accuracy_character(a, b, hard=False) for a, b in zip(y_hat, y)])


def parsing_accuracy_token(y_hat: Sequence[str], y: Sequence[str], x: Sequence[str]) -> float:
    return np.mean([mask_accuracy_token(yp, yt, m, hard=True) for yp, yt, m in zip(y_hat, y, x)])


def mean_mask_agreement(y_hat: Sequence[str], y: Sequence[str]) -> float:
    return np.mean([mask_agreement(a, b) for a, b in zip(y_hat, y)])


def rand_score(y_hat: Sequence[Hashable], y: Sequence[Hashable]) -> float:
    return adjusted_rand_score(y, y_hat)


def template_accuracy(y_hat: Sequence[Hashable], y: Sequence[Hashable], x: Sequence[str]) -> float:
    apply_mask = lambda msg, msk: tuple(''.join(c for c, m in zip(msg, msk) if m == '0').split())
    temp_idx = {}
    for idx, (msg, msk) in enumerate(zip(x, y_hat)):
        const = apply_mask(msg, msk)
        if const not in temp_idx:
            temp_idx[const] = []
        temp_idx[const].append(idx)
    t_v = [set(apply_mask(x[idx], y[idx]) for idx in indices) == {const} for const, indices in temp_idx.items()]
    t_o = set(apply_mask(e_x, e_y) for e_x, e_y in zip(x, y))
    return sum(t_v) / len(t_v), sum(t_v) / len(t_o)


def logpai_metrics(y_hat: Sequence[Hashable], y: Sequence[Hashable]) -> float:
    series_groundtruth = pd.Series(y)
    series_parsedlog = pd.Series(y_hat)
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
            error = False
        # if error:
        #     print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy


def mask_agreement(x: str, y: str):
    assert len(x) == len(y)
    m = np.zeros((2, 2))
    for a, b in zip(x, y):
        m[int(a), int(b)] += 1
    y0 = m[:, 0].sum()
    y1 = m[:, 1].sum()
    if y1 == 0:
        return (m[0, 0] - m[1, 0]) / y0
    m0 = (m[0, 0] - m[1, 0]) / y0
    m1 = (m[1, 1] - m[0, 1]) / y1
    return np.mean([m0, m1])


def mask_accuracy_character(a: str, b: str, hard=False) -> float:
    assert len(a) == len(b)
    if hard:
        return 0 if any(ea != eb for ea, eb in zip(a, b)) else 1
    return sum(1 for ea, eb in zip(a, b) if ea == eb) / len(a)


def mask_accuracy_token(a: str, b: str, m: str, hard=False) -> float:
    assert len(a) == len(b)
    idx = [i for i, c in enumerate(m) if c in string.whitespace]
    a_tokenized = ['1' in s for s in split_str_idx(a, idx)]
    b_tokenized = ['1' in s for s in split_str_idx(b, idx)]
    assert len(a_tokenized) == len(b_tokenized)
    if hard:
        return 0 if any(t1 != t2 for t1, t2 in zip(a_tokenized, b_tokenized)) else 1
    return sum(1 for t1, t2 in zip(a_tokenized, b_tokenized) if t1 == t2) / len(a_tokenized)
