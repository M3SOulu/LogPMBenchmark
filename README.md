# LogPM Benchmark

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Introduction

Log PM is a log parser benchmark emphasizing precise in-message parameter detection rather than template-based message clustering. The original paper is titled "LogPM: A new benchmark" and published at <Conference>.

Log PM introduces a new parsing output called parameter mask, a binary sequence with the same length as the log message where each element indicates if the corresponding message character is a parameter.
For instance, the log message:
``` log
User u_123 connected from 10.10.1.10
```
is supposed to produce the mask:
```log
000001111100000000000000001111111111
```

The traditional metrics, such as group accuracy, F1 score, and rand index, are also available in this benchmark as well. You may modify the metrics in `benchmark/baseline_benchmark.py`


## Usage

Please install python version >= 3.9 if you haven't installed it already.
Clone the repository:
```commandline
git clone https://github.com/M3SOulu/LogPMBenchmark.git
cd LogPMBenchmark
```

Prepare execution environment:
```commandline
conda env create -f environment.yaml
conda activate LogPMBenchmark
```

Run the benchmark:
```commandline
python main.py benchmark <parser_name> <dataset_name>
```
E.g.
```commandline
python main.py benchmark spell proxifier
```
benchmarks the SPELL parsing algorithm on proxifier dataset.

Run the benchmark for all datasets ignore the dataset argument,and just pass the parser:
```commandline
python main.py benchmark <parser_name>
```

Check the list of available datasets and parsers by:
```commandline
python main.py list
```

Download a dataset without any benchmark:
```commandline
python main.py download <dataset_name>
```

## Benchmark a new parser:
1. Create a new class and make it inherit from `BaseBnechmark`.
```python
from benchmark.base_classes import BaseBenchmark

class MyParserBenchmark(BaseBenchmark):
    pass
```
2. Implement `fit`, `predict_mask`, and `predict_cluster` methods.
```python
class MyParserBenchmark(BaseBenchmark):
    def __init__(self): # initialization (optional)
        ...

    def fit(self, x: Sequence[str]): # learn the latent patterns give the messages
        ...

    def predict_mask(self, x: Sequence[str]) -> Sequence[str]: # predict the parameter masks given the messages
        ...
        
    def predict_cluster(self, x: Sequence[str]) -> Sequence[Hashable]: # predict the cluster IDs given the message
        ...
        
```
3. Add the class to the `PARSER` dictionary object in `benchmark/__init__.py`
```python
BENCHMARKS = {
    'no_parameter': NoParameterBenchmark,
    'all_parameter': AllParameterBenchmark,
    'random_parameter': RandomParameterBenchmark,
    'drain': DrainBenchmark,
    'lenma': LenmaBenchmark,
    'spell': SpellBenchmark,
    'my_parser': MyParserBechmark
}
```
4. Check if your parser have been added successfully by `python main.py list`.
```commandline
❯ python .\main.py list

Parsers:
        no_parameter
        all_parameter
        random_parameter
        drain
        lenma
        spell
        my_parser

Datasets:
        hpc
        zookeeper
        android
        apache
        hadoop
        hdfs
        linux
        openstack
        proxifier
        ssh
```
5. Benchmark it by running `python main.py benchmark my_parser`.


Please consider citing the paper if you use the code. Bibtex:
```

```
