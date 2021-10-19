import numpy as np
from ivp_enhanced.ivp import solve_ivp
from ivp_enhanced.rk import ImplicitMidpoint, Midpoint
import matplotlib.pyplot as plt
def simplectic_problem(t, y):
    return [-y[1], y[0]]
# use fixed stepsize
T = 100
h = 0.1
sol_midpoint = solve_ivp(simplectic_problem, [0, T], [0.0, 1],
                method=Midpoint, step=h)

energy_midpoint = np.linalg.norm(sol_midpoint.y, axis=0)
sol_implicit_midpoint = solve_ivp(simplectic_problem, [0, T], [0.0, 1],
                method=ImplicitMidpoint, step=h)
# using the true sol to estimate the error
sol_true = np.vstack((-np.sin(sol_implicit_midpoint.t), np.cos(sol_implicit_midpoint.t)))
n_dim = 2
err = np.sqrt(np.average(np.linalg.norm(sol_implicit_midpoint.y - sol_true, axis=0) ** 2 / n_dim))
print(err)

energy_implicit_midpoint = np.linalg.norm(sol_implicit_midpoint.y, axis=0)
# get the last -60 values
inspected_times = sol_implicit_midpoint.t[-70:]
plt.figure()
plt.plot(sol_implicit_midpoint.t, energy_midpoint, label='midpoint')
plt.plot(sol_implicit_midpoint.t, energy_implicit_midpoint, label='implicit midpoint')
plt.xlabel('time')
plt.ylabel('energy')
plt.legend()
plt.figure()
plt.plot(inspected_times, sol_implicit_midpoint.y[0, -70:], label='sol')
plt.plot(inspected_times, sol_true[0, -70:], label='true')
plt.xlabel('time')
plt.ylabel('p')
plt.legend()
plt.show()
# when we increase T without changing h for fixed stepsize RK23
# the error increases propotionally
