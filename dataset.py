import csv
from pathlib import Path

import numpy as np


class Dataset:
    def __init__(self, path, name=None, subsample=None, encoding='utf-8'):
        self.__x = []
        self.__y = []
        self.__c = []
        with open(path, encoding=encoding) as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row[0].strip()) == len(row[1].strip())
                self.__x.append(row[0].strip())
                self.__y.append(row[1].strip())
                self.__c.append(row[2].strip())
            if subsample is not None:
                assert type(subsample) is int
                indices = np.random.choice(np.arange(len(self.__x)), subsample)
                self.__x = [self.__x[idx] for idx in indices]
                self.__y = [self.__y[idx] for idx in indices]
                self.__c = [self.__c[idx] for idx in indices]
        self.__name = name if name is not None else Path(path).name

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, item):
        return self.__x[item], self.__y[item], self.__c[item]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def c(self):
        return self.__c
