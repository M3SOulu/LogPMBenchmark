import itertools
import os
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Dict, Any, Type

import pandas as pd

from benchmark.base_classes import BaseBenchmark
from benchmark.baseline_benchmark import NoParameterBenchmark, AllParameterBenchmark, RandomParameterBenchmark
from benchmark.drain_benchmark import DrainBenchmark
from benchmark.lenma_benchmark import LenmaBenchmark
from benchmark.spell_benchmark import SpellBenchmark
from dataset import Dataset

DATASETS = ('hpc', 'zookeeper', 'android', 'apache', 'hadoop', 'hdfs', 'linux', 'openstack', 'proxifier', 'ssh')
BENCHMARKS = (SpellBenchmark, DrainBenchmark, LenmaBenchmark)
BASELINES = (NoParameterBenchmark, AllParameterBenchmark, RandomParameterBenchmark)
MULTI_PROCESS = True


def main():
    args = [arg for arg in itertools.product(DATASETS, BENCHMARKS)]
    if MULTI_PROCESS:
        with Pool(cpu_count()) as pool:
            results = pool.starmap(benchmark, args)
    else:
        results = [benchmark(*a) for a in args]

    df = pd.DataFrame.from_records(results)
    print(df.to_string())
    df.to_csv("benchmark_results.csv", index=False)


def benchmark(ds_name: str, parser: Type[BaseBenchmark]) -> Dict[str, Any]:
    print(f"[{os.getpid()}] <{parser.__name__}, {ds_name}>")
    ds = Dataset(f'data/{ds_name}.csv')
    b = parser()
    res = b.benchmark(ds)
    print(f"({parser.__name__}, {ds_name}) -> {', '.join(f'{k}: {v:.2f}' for k, v in res.items())}")
    return {'parser': parser.__name__, 'dataset': ds_name} | res


if __name__ == '__main__':
    main()
