import pytest
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_almost_equal,
                           assert_equal)
from scipy.integrate import solve_ivp
from ivp_enhanced.rk import ImplicitMidpoint, ImplicitEuler
from test_common import fun_zero, func_spiral

@pytest.mark.parametrize('method', [ImplicitEuler, ImplicitMidpoint])
def test_rungekutta_fixed_func_spiral(method):
    T = 0.1
    result = solve_ivp(func_spiral, [0, 2 * np.pi], np.zeros(2), method=method, step=T)
    # same result with julia?
    assert_(result.success)
    assert_equal(result.status, 0)
    assert_equal(len(result.t), int(2 * np.pi / T) + 2)
    t = result.t
    x = t * np.cos(t)
    y = t * np.sin(t)
    y_true = np.vstack((x, y))
    if isinstance(method, ImplicitMidpoint):
        assert np.mean(np.square(y_true - result.y)) < 1e-3

if __name__ == '__main__':
    test_rungekutta_fixed_func_spiral(ImplicitEuler)