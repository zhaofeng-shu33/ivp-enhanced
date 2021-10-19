import warnings

import numpy as np
from ivp_enhanced.ivp import solve_ivp
from ivp_enhanced.rk import BS3
from test_common import fun_zero, func_spiral

def test_collect_err():
    result = solve_ivp(fun_zero, [0, 10], np.ones(3), method=BS3, record_err=True, controller='PIController')
    assert hasattr(result, 'err')

def max_nfev_event(t, y):
    if t < 1:
        return 1
    return y[0]

max_nfev_event.terminal = True

def test_pre_mature_terminate():
    result = solve_ivp(func_spiral, [0, 2 * np.pi], np.zeros(2), method=BS3, events=[max_nfev_event])
    assert result.status == 1

def max_nfev_event_with_solver(t, y, solver):
    return solver.nfev - 54

max_nfev_event_with_solver.terminal = True

def test_pre_mature_terminate_with_event_solver():
    warnings.simplefilter("ignore")
    result = solve_ivp(func_spiral, [0, 2 * np.pi], np.zeros(2), method=BS3, events=[max_nfev_event_with_solver],
             event_solver=True)
    assert result.status == 1