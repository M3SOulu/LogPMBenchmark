import csv
import string
from pathlib import Path

from helpers import extract_keywords


class Dataset:
    def __init__(self, path, name=None, encoding='utf-8'):
        self.__x = []
        self.__y = []
        self.__c = []
        cluster_table = {}
        with open(path, encoding=encoding) as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row[0].strip()) == len(row[1].strip())
                self.__x.append(row[0].strip())
                self.__y.append(row[1].strip())
                cluster_str = extract_keywords(row[0], row[1])
                if cluster_str not in cluster_table:
                    cluster_table[cluster_str] = len(cluster_table)
                self.__c.append(cluster_table[cluster_str])
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
