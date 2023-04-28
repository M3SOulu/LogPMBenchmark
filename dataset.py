import csv
import multiprocessing
import os.path
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

DATASET_NAMES = ('hpc', 'zookeeper', 'android', 'apache', 'hadoop', 'hdfs', 'linux', 'openstack', 'proxifier', 'ssh')
download_lock = multiprocessing.Lock()


class Dataset:
    def __init__(self, name, subsample=None, encoding='utf-8'):
        self.__x = []
        self.__y = []
        self.__c = []
        assert name in DATASET_NAMES, f"Invalid dataset name: {name}, it should be chosen from {DATASET_NAMES}"
        path = f'data/{name}.csv'
        if not os.path.exists(path):
            download_dataset(name)
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


def check_data_directory():
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir()
    elif not data_dir.is_dir():
        print("data should be a directory but it is not. Consider removing it and running the benchmark again.")
        exit(1)


def download_dataset(name: str, block_size=2048):
    check_data_directory()
    url = f'https://zenodo.org/record/7875570/files/{name}.csv'
    temp_path = f'data/.{name}.csv'
    file_path = f'data/{name}.csv'
    if os.path.exists(file_path):
        return
    print(f"Dataset '{name}' were not found in '{file_path}' so downloading it from '{url}'")
    with download_lock:
        response = requests.get(url, stream=True)
        if not response:
            print(f"Unable to download {name}")
            exit(1)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(temp_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
            exit(1)
        else:
            os.rename(temp_path, file_path)
            print(f"{name} downloaded successfully")
