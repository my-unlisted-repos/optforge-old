import pytest

import optforge as of

BENCHMARKS = [
    of.benchmarks.WasteTreatmentPlantDesign(),
    of.benchmarks.ChemicalEquilibriumProblem(),
    of.benchmarks.TransformerDesign(),
    #of.benchmarks.CataloguePlanning(),
    #of.benchmarks.ChemicalProcessControlProblem()
]

@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_benchmark(benchmark: of.benchmarks.Benchmark,):
    minima = benchmark.get_minima(); minima_params = benchmark.get_minima_params()
    if minima is None: raise ValueError(f'{benchmark.__class__.__name__}.get_minima() is None.')
    if minima_params is None: raise ValueError(f'{benchmark.__class__.__name__}.get_minima_params() is None.')
    value = benchmark.evaluate_params(minima_params)
    assert value == pytest.approx(minima, rel=0.001)