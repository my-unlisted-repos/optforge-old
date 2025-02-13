import numpy as np
import pytest

import optforge as of
from optforge.paramdict import ParamDict


def _get_categorical_array_param(*args, **kwargs):
    study = of.Study()
    study.objective = lambda trial: 1
    study.optimizer = of.optim.baseline.BaselineNoop()
    trial = of.Trial(ParamDict(), study, False)
    trial.suggest_categorical_array('p', *args, **kwargs)
    return trial.params['p']

@pytest.mark.parametrize("scale", [0.1, None, 100])
def test_onehot_param(scale):
    vals = []
    for i in range(1000):
        p = _get_categorical_array_param(1, ('a', 'b', 'c', 'd', 'e'), scale=scale, one_hot=True, init='uniform')
        vals.append(p()[0])

    for i in ('a', 'b', 'c', 'd', 'e'):
        assert 150 < vals.count(i) < 250

@pytest.mark.parametrize("scale", [0.1, None, 100])
def test_onehot_param_sample_random(scale):
    vals = []
    for i in range(1000):
        p = _get_categorical_array_param( 1, ('a', 'b', 'c', 'd', 'e'), scale=scale, one_hot=True, )
        p.data = p.sample_random()
        vals.append(p()[0])

    for i in ('a', 'b', 'c', 'd', 'e'):
        assert 150 < vals.count(i) < 250


@pytest.mark.parametrize("scale", [0.5, None, 100])
def test_onehot_param_sample_petrub(scale:float | None):
    vals = []
    scalez = scale if scale is not None else 1
    init_val: float = 0.1 / scalez
    for i in range(5000):
        p = _get_categorical_array_param( 1, ('a', 'b', 'c', 'd', 'e'), init = [[0, init_val, 0, 0, 0]], scale=scale, one_hot=True, ) # type:ignore
        p.data = p.sample_petrub(0.1)
        vals.append(p()[0])

    for i in ('a', 'b', 'c', 'd', 'e'):
        if i == 'b': assert 3400 < vals.count(i) < 3600
        else: assert 300 < vals.count(i) < 500


@pytest.mark.parametrize("value", [0., 1, 2., 3, 4.])
@pytest.mark.parametrize("scale", [0.5, None, 100])
def test_onehot_param_set_from_numeric_value(value, scale):
    p = _get_categorical_array_param( 1, ('0', '1', '2', '3', '4'), scale = scale, one_hot=True, )
    p.sampler.set_index_array(p, np.array([value])) # type:ignore
    assert float(p()[0]) == value
    assert float(p.sampler.get_index_array(p)[0]) == value # type:ignore
