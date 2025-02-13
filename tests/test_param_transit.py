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
@pytest.mark.parametrize("normalize", [True, False])
def test_transit_param(scale,normalize):
    vals = []
    for i in range(1000):
        p = _get_categorical_array_param( 1, ('a', 'b', 'c', 'd', 'e'), scale=scale, normalize=normalize, one_hot=False)
        vals.append(p()[0])

    for i in ('a', 'b', 'c', 'd', 'e'):
        assert 150 < vals.count(i) < 250

@pytest.mark.parametrize("scale", [0.1, None, 100])
@pytest.mark.parametrize("normalize", [True, False])
def test_transit_param_sample_random(scale,normalize):
    vals = []
    for i in range(1000):
        p = _get_categorical_array_param( 1, ('a', 'b', 'c', 'd', 'e'), scale=scale, normalize=normalize, one_hot=False)
        p.data = p.sample_random()
        vals.append(p()[0])

    for i in ('a', 'b', 'c', 'd', 'e'):
        assert 150 < vals.count(i) < 250


def test_transit_param_sample_petrub():
    vals = []
    for i in range(5000):
        p = _get_categorical_array_param( 1, ('a', 'b', 'c', 'd', 'e'), init = 1, one_hot=False)
        p.data = p.sample_petrub(0.1)
        vals.append(p()[0])

    for i in ('a', 'b', 'c', 'd', 'e'):
        if i == 'b': assert 4300 < vals.count(i) < 4750
        else: assert 70 < vals.count(i) < 130
        # assert 900 < vals.count(i) < 1100

# def test_transit_param_sample_increment():
#     vals = []
#     for i in range(10000):
#         p = _get_categorical_array_param( 10, ('a', 'b', 'c', 'd', 'e'), init = 1, transitional=True)
#         p.data = p.sample_increment(0.2, 3)
#         vals.append(p()[3])

#     for i in ('a', 'b', 'c', 'd', 'e'):
#         if i == 'b': assert 8000 < vals.count(i) < 9000
#         else: assert 330 < vals.count(i) < 470


@pytest.mark.parametrize("value", [0., 1, 2., 3, 4.])
@pytest.mark.parametrize("scale", [0.5, None, 100])
@pytest.mark.parametrize("normalize", [True, False])
def test_transit_param_set_from_numeric_value(value, scale, normalize):
    p = _get_categorical_array_param(1, ('0', '1', '2', '3', '4'), scale = scale, normalize=normalize, one_hot=False, )
    p.set_unscaled_array(np.array([value]))
    assert float(p()[0]) == value
    assert float(p.get_unscaled_array()[0]) == value