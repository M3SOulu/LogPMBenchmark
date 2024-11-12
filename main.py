import os
from typing import Dict, Any, Optional, Sequence, Union, List
import fire
import pandas as pd
from dataset import Dataset, DATASET_NAMES, download_dataset
from benchmark import BENCHMARKS
from helpers import sensitivity_gen


def main():
    fire.Fire(
        {
            "benchmark": benchmark,
            "sensitivity": sensitivity,
            "download": download,
            "list": ls,
        }
    )


def benchmark(
    parser: str,
    datasets: Optional[Union[str, Sequence[str]]] = None,
    results_path: str = "results.csv",
):
    """
    Benchmark a parser on a given dataset. If the dataset is not passed, the parser will be benchmarked on all
    available datasets. If any dataset does not exist in the data directory, the dataset will be automatically
    downloaded.
    :param datasets: The name of the dataset or none for all dataset
    :param parser: The parser algorithm name
    :param results_path: The csv file path where the benchmark results will be stored
    :return:
    """
    assert parser in BENCHMARKS
    assert results_path
    if isinstance(datasets, str):
        assert datasets in DATASET_NAMES
    elif isinstance(datasets, list):
        assert all(ds in DATASET_NAMES for ds in datasets)

    if datasets is None:
        datasets = DATASET_NAMES
    elif isinstance(datasets, str):
        datasets = [datasets]
    assert datasets is not None
    print(f"Benchmarking {parser} on {datasets}")
    args = [(ds, parser) for ds in datasets]
    results = [__benchmark(*a) for a in args]
    df = pd.DataFrame.from_records(results)
    df.to_csv(results_path, index=False)
    print(f"Results stored in {results_path}")


def sensitivity(
    parser: str,
    datasets: Optional[Union[str, Sequence[str]]] = None,
    step: float = 0.05,
    results_path: str = "results_sensitivity.csv",
):
    """
    TODO
    :param datasets: The name of the dataset or none for all dataset
    :param parser: The parser algorithm name
    :param results_path: The csv file path where the benchmark results will be stored
    :return:
    """
    assert parser in BENCHMARKS
    assert results_path
    if isinstance(datasets, str):
        assert datasets in DATASET_NAMES
    elif isinstance(datasets, list):
        assert all(ds in DATASET_NAMES for ds in datasets)

    if datasets is None:
        datasets = DATASET_NAMES
    elif isinstance(datasets, str):
        datasets = [datasets]
    assert datasets is not None
    print(f"Benchmarking {parser} on {datasets}")
    args = [(ds, parser, sen) for ds in datasets for sen in sensitivity_gen(step)]
    results = [__benchmark(*a) for a in args]
    df = pd.DataFrame.from_records(results)
    df.to_csv(results_path, index=False)
    print(f"Results stored in {results_path}")


def download(dataset_name: str):
    """
    Download the benchmark csv file from Zenodo.org
    :param dataset_name: the name of the benchmark dataset all in lower case letters.
    :return:
    """
    assert dataset_name in DATASET_NAMES
    download_dataset(dataset_name)


def ls():
    """
    List available parsers and datasets
    :return:
    """
    print("\nParsers:")
    for p in BENCHMARKS:
        print(f"\t{p}")
    print("\nDatasets:")
    for d in DATASET_NAMES:
        print(f"\t{d}")


def __benchmark(
    ds_name: str, parser: str, sensitivity: Optional[float] = None
) -> Dict[str, Any]:
    if sensitivity is None:
        print(f"Process {os.getpid()} is benchmarking {parser} on {ds_name}")
    else:
        print(
            f"Process {os.getpid()} is benchmarking {parser} on {ds_name} with sensitivity {sensitivity:.2f}"
        )
    ds = Dataset(ds_name)
    b = BENCHMARKS[parser]() if sensitivity is None else BENCHMARKS[parser](sensitivity)
    res = b.benchmark(ds)
    print(
        f"({parser}, {ds_name}) -> {', '.join(f'{k}: {v:.2f}' for k, v in res.items())}"
    )
    if sensitivity is None:
        return {"parser": parser, "dataset": ds_name} | res
    else:
        return {"parser": parser, "dataset": ds_name, "sensitivity": sensitivity} | res


if __name__ == "__main__":
    main()
