import numpy as np
import pytest

import optforge as of
from optforge.paramdict import ParamDict


def _get_array_param(*args, **kwargs):
    study = of.Study()
    study.objective = lambda trial: 1
    study.optimizer = of.optim.baseline.BaselineNoop()
    trial = of.Trial(ParamDict(), study, False)
    trial.suggest_array('p', *args, **kwargs)
    return trial.params['p']

@pytest.mark.parametrize("low", [-100, 1])
@pytest.mark.parametrize("high", [-50,5])
@pytest.mark.parametrize("scale", [0.1, None, 10])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("init", [0, -6, 6])
def test_int_param(low, high, scale, normalize, init):
    if low >= high: return
    if low > init: return
    if high < init: return

    p = _get_array_param(1, low, high, init = init, step=1, scale=scale, normalize=normalize, discrete=True)
    assert p() == init


def test_int_param_sample_petrub():
    l = []
    m = []
    h = []
    for i in range(2000):
        p = _get_array_param(1, 0, 10, init = 5, step=1, discrete=True)
        p.data = p.sample_petrub(0.1)
        value = p()
        if value == 4: l.append(i)
        if value == 5: m.append(i)
        if value == 6: h.append(i)
        else: assert value in (4,5,6)

    assert 200 < len(l) < 300
    assert 1400 < len(m) < 1600
    assert 200 < len(h) < 300

def test_int_param_sample_set():
    l = []
    h = []
    for i in range(2000):
        p = _get_array_param(1, 0, 10, init = 5, step=1, normalize=False, discrete=True)
        p.data = p.sample_set(np.array(4.2))
        value = p()
        if value == 4: l.append(i)
        if value == 5: h.append(i)
        else: assert value in (4,5)

    assert 0.2 < len(h) / len(l) < 0.3

def test_int_param_sample_set_norm():
    l = []
    h = []
    for i in range(5000):
        p = _get_array_param(1, 0, 10, init = 5, step=1, normalize=True, discrete=True)
        p.data = p.sample_set(np.array(-0.05))
        value = p()
        if value == 4: l.append(i)
        if value == 5: h.append(i)
        else: assert value in (4,5)

    assert 0.3 < len(l) / len(h) < 0.36


# def test_int_param_sample_increment():
#     l = []
#     h = []
#     for i in range(5000):
#         p = _get_array_param(1, 0, 10, init = 4, step=1, normalize=False, discrete=True)
#         p.data = p.sample_increment(0.1, 0)
#         value = p()
#         if value == 4: l.append(i)
#         if value == 5: h.append(i)
#         else: assert value in (4,5)

#     assert 0.09 < (len(h) / len(l)) < 0.13