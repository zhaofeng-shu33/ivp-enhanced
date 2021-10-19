import pytest

import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_almost_equal,
                           assert_equal)
from scipy.integrate import solve_ivp
from scipy.stats import linregress

from ivp_enhanced.rk import Midpoint, RK4, BS3, RKF4, DOPRI5
from test_common import fun_zero, func_spiral


@pytest.mark.parametrize('method', [RK4, Midpoint, BS3, RKF4])
def test_rungekutta_fixed_fun_zero(method):
    result = solve_ivp(fun_zero, [0, 10], np.ones(3), method=method, step=1.0)
    assert_(result.success)
    assert_equal(result.status, 0)
    assert_allclose(result.y, 1.0, rtol=1e-15)

@pytest.mark.parametrize('method', [RK4, Midpoint, BS3, RKF4, DOPRI5])
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
    assert np.mean(np.square(y_true - result.y)) < 1e-3

@pytest.mark.parametrize('method', [RK4, Midpoint, BS3, RKF4])
def test_rungekutta_fixed_dense(method):
    T = 0.1
    result = solve_ivp(func_spiral, [0, 2 * np.pi], np.zeros(2), method=method, step=T)
    result_2 = solve_ivp(func_spiral, [0, 2 * np.pi], np.zeros(2), method=method, step=T/3, dense_output=True)

    y_true = result_2.sol(result.t)
    assert np.mean(np.square(y_true - result.y)) < 1e-3

@pytest.mark.parametrize('method', [Midpoint])
def test_rungekutta_fixed_interp(method):
    T = 1.0
    result = solve_ivp(func_spiral, [0, 1.0], np.zeros(2), method=method, step=T, dense_output=True)
    t_val = np.linspace(0, 1.0)
    y_val = result.sol(t_val)
    print(y_val.shape)
    result_1 = linregress(t_val, y_val[0, :])
    assert_almost_equal(result_1.rvalue, 1.0)
    result_2 = linregress(t_val, y_val[1, :])
    assert_almost_equal(result_2.rvalue, 1.0)

method_controller_list = [(Midpoint, 'IController'), (Midpoint, 'PIController'),
                          (BS3, 'IController'), (BS3, 'PIController'),
                          (RKF4, 'IController'), (RKF4, 'PIController'),
                          (DOPRI5, 'IController'), (DOPRI5, 'PIController')]

@pytest.mark.parametrize('method, controller', method_controller_list)
def test_rungekutta_adaptive_fun_zero(method, controller):
    result = solve_ivp(fun_zero, [0, 10], np.ones(3), method=method, controller=controller)
    assert_(result.success)
    assert_equal(result.status, 0)
    assert_allclose(result.y, 1.0, rtol=1e-15)

@pytest.mark.parametrize('method, controller', method_controller_list)
def test_rungekutta_adaptive_func_spiral(method, controller):
    result = solve_ivp(func_spiral, [0, 2 * np.pi], np.zeros(2), method=method, controller=controller)
    assert_(result.success)
    assert_equal(result.status, 0)
    t = result.t
    x = t * np.cos(t)
    y = t * np.sin(t)
    y_true = np.vstack((x, y))
    assert np.mean(np.square(y_true - result.y)) < 1e-3

def stiff_problem(t, y):
    y1 = -2000 * (np.cos(t) * y[0] + np.sin(t) * y[1] + 1)
    y2 = -2000 * (-np.sin(t) * y[0] + np.cos(t) * y[1] + 1)
    return [y1, y2]

def test_pi_objective_function_failed():
    sol = solve_ivp(stiff_problem,
                [0, np.pi / 2], [1, 0], method=BS3, controller='PIController',
                 beta_1=0.5, beta_2=0.5, max_nfev=6000)
    assert sol.status == -1

if __name__ == '__main__':
    test_rungekutta_fixed_func_spiral(BS3)
