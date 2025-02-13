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
def test_continous_param_init_uniform(low,high,scale,normalize):
    """Test that default uniform initialization has mean of (low + high) / 2"""
    if low > high: return
    values = []
    for i in range(2000):
        param = _get_array_param(10, low = low, high = high, scale=scale, normalize = normalize)
        values.append(param())

    assert float(np.mean(values)) == pytest.approx(
        (low + high) / 2, abs = (high - low) / 100
        )

@pytest.mark.parametrize("low", [-100, 1])
@pytest.mark.parametrize("high", [-50,5])
@pytest.mark.parametrize("scale", [0.1, None, 10])
@pytest.mark.parametrize("normalize", [True, False])
def test_continous_param_sample_random(low,high,scale,normalize):
    """Test that random sampling has mean of (low + high) / 2"""
    if low > high: return
    values = []
    for i in range(2000):
        param = _get_array_param(10, low = low, high = high, scale=scale, normalize = normalize)
        param.data = param.sample_random()
        values.append(param())

    assert pytest.approx(float(np.mean(values))) == pytest.approx(
        (low + high) / 2, abs = (high - low) / 100)

@pytest.mark.parametrize("low", [-100, 1])
@pytest.mark.parametrize("high", [-50,5])
@pytest.mark.parametrize("sigma", [0.1,0.4])
def test_continous_param_sample_generate_petrubation(low,high,sigma):
    """Test that random sampling has mean of (low + high) / 2"""
    if low > high: return
    values = []
    for i in range(1000):
        param = _get_array_param(10, low = low, high = high, init = high)
        param.data += param.sample_generate_petrubation(sigma)
        values.append(param())

    mean = float(np.mean(values))
    assert mean == pytest.approx(high - ((high - low) * sigma / 2), abs = (high - low) / 250)

@pytest.mark.parametrize("low", [-100, 1])
@pytest.mark.parametrize("high", [-50,5])
@pytest.mark.parametrize("sigma", [0.1,0.4])
def test_continous_param_sample_petrub(low,high,sigma):
    """Test that random sampling has mean of (low + high) / 2"""
    if low > high: return
    values = []
    for i in range(1000):
        param = _get_array_param(10, low = low, high = high, init = high)
        param.data = param.sample_petrub(sigma)
        values.append(param())

    mean = float(np.mean(values))
    assert mean == pytest.approx(high - ((high - low) * sigma / 2), abs = (high - low) / 250)


@pytest.mark.parametrize("low", [-100, 1])
@pytest.mark.parametrize("high", [-50,5])
@pytest.mark.parametrize("scale", [0.1, None, 10])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("init", [0, 0.3, 1])
def test_continous_param_general(low,high,scale,normalize,init):
    if low >= high: return

    norm_init = (high - low) * init + low
    param = _get_array_param((2,2), low = low, high = high, scale=scale, normalize = normalize, init=norm_init)

    # param() must be inits
    mean_param = float(np.mean(param()))
    mean_data = float(np.mean(param.data))
    assert mean_param == norm_init

    # test attrs
    if scale is None: scale = 1
    if normalize:
        # if normalize, low and high ar always -1 and 1
        assert param.low == -scale
        assert param.high == scale
        # test normalized data
        assert mean_data == pytest.approx(((init - 0.5) * 2) * scale)
    # if not normalize
    else:
        # low and high will be multiplied by scale, or None
        assert param.low == low * scale
        assert param.high == high * scale

        # data will be init * scale
        assert mean_data == norm_init * scale


@pytest.mark.parametrize("low", [-100, 1])
@pytest.mark.parametrize("high", [-50,5])
@pytest.mark.parametrize("scale", [0.1, None, 10])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("set", [0, 0.3, 1])
def test_continous_param_set_from_numeric_value(low,high,scale,normalize,set):
    if low >= high: return

    norm_set = (high - low) * set + low
    param = _get_array_param(1, low = low, high = high, scale=scale, normalize = normalize)
    param.set_unscaled_array(np.array(norm_set))
    assert float(param()[0]) == norm_set

    param = _get_array_param(1, low = low, high = high, scale=scale, normalize = normalize)
    param.set_value(norm_set)
    assert float(param()[0]) == norm_set