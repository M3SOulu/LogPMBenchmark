from benchmark.baseline_benchmark import NoParameterBenchmark, AllParameterBenchmark, RandomParameterBenchmark
from benchmark.drain_benchmark import DrainBenchmark
from benchmark.lenma_benchmark import LenmaBenchmark
from benchmark.spell_benchmark import SpellBenchmark
from benchmark.tipping_benchmark import TippingBenchmark

BENCHMARKS = {
    'no_parameter': NoParameterBenchmark,
    'all_parameter': AllParameterBenchmark,
    'random_parameter': RandomParameterBenchmark,
    'drain': DrainBenchmark,
    'lenma': LenmaBenchmark,
    'spell': SpellBenchmark,
    'tipping': TippingBenchmark
}
