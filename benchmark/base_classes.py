import string
from abc import abstractmethod
from typing import Sequence, Hashable, Dict

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

        # convert y and y_hat to cluster IDs
        before_predict_cluster = timeit.default_timer()
        y_hat_cluster = self.predict_cluster(dataset.x)
        after_predict_cluster = timeit.default_timer()

        # compute accuracies
        p_acc_char = parsing_accuracy_character(y_hat, dataset.y)
        p_acc_token = parsing_accuracy_token(y_hat, dataset.y, dataset.x)
        r_score = rand_score(y_hat_cluster, dataset.c)
        precision, recall, f1_score, group_accuracy = logpai_metrics(y_hat_cluster, dataset.c)

        return {
            'character_parsing_accuracy': p_acc_char,
            'token_parsing_accuracy': p_acc_token,
            'rand_score': r_score,
            'precision': precision,
            'recall': recall,
            'f1 score': f1_score,
            'group_accuracy': group_accuracy,
            'fit_duration': after_fit - before_fit,
            'predict_mask_duration': after_predict_mask - before_predict_mask,
            'predict_cluster_duration': after_predict_cluster - before_predict_cluster
        }

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


def parsing_accuracy_character(y_hat: Sequence[str], y: Sequence[str]) -> float:
    assert len(y_hat) == len(y)
    return np.mean([mask_accuracy_character(a, b) for a, b in zip(y_hat, y)])


def parsing_accuracy_token(y_hat: Sequence[str], y: Sequence[str], x: Sequence[str]) -> float:
    return np.mean([mask_accuracy_token(yp, yt, m) for yp, yt, m in zip(y_hat, y, x)])


def rand_score(y_hat: Sequence[Hashable], y: Sequence[Hashable]) -> float:
    return adjusted_rand_score(y, y_hat)


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
        if error:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy


def mask_accuracy_character(a: str, b: str) -> float:
    assert len(a) == len(b)
    return sum(1 for ea, eb in zip(a, b) if ea == eb) / len(a)


def mask_accuracy_token(a: str, b: str, m: str) -> float:
    assert len(a) == len(b)
    idx = [i for i, c in enumerate(m) if c in string.whitespace]
    a_tokenized = ['1' in s for s in split_str_idx(a, idx)]
    b_tokenized = ['1' in s for s in split_str_idx(b, idx)]
    assert len(a_tokenized) == len(b_tokenized)
    return sum(1 for t1, t2 in zip(a_tokenized, b_tokenized) if t1 == t2) / len(a_tokenized)
