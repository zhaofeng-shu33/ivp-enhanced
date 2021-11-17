# implement fixed step Runge Kutta solver
import numpy as np
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step, EPS, num_jac)
from scipy.integrate._ivp.rk import rk_step, RkDenseOutput
from scipy.integrate._ivp.rk import SAFETY, MIN_FACTOR, MAX_FACTOR
from scipy.sparse import csc_matrix, issparse, eye
from scipy.optimize._numdiff import group_columns
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve
from scipy.integrate._ivp import dop853_coefficients



NEWTON_MAXITER = 6
def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex, solve_lu, A, C):
    """Solve the collocation system.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    h : float
        Step to try.
    Z0 : ndarray, shape (3, n)
        Initial guess for the solution. It determines new values of `y` at
        ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.
    scale : float
        Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.
    tol : float
        Tolerance to which solve the system. This value is compared with
        the normalized by `scale` error.
    LU_real, LU_complex
        LU decompositions of the system Jacobians.
    solve_lu : callable
        Callable which solves a linear system given a LU decomposition. The
        signature is ``solve_lu(LU, b)``.

    Returns
    -------
    converged : bool
        Whether iterations converged.
    n_iter : int
        Number of completed iterations.
    Z : ndarray, shape (3, n)
        Found solution.
    rate : float
        The rate of convergence.
    """
    n = y.shape[0]
    Z = Z0

    n_stages = 1
    F = np.empty((n_stages, n))
    ch = h * C

    dZ_norm_old = None
    # dW = np.empty_like(W)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        for i in range(n_stages):
            F[i] = fun(t + ch[i], y + Z[i])

        if not np.all(np.isfinite(F)):
            break

        F = - Z + h * A[0, 0] * F
        f_real = F[0] # .T.dot(TI_REAL) - M_real * W[0]
        
        # f_complex = F.T.dot(TI_COMPLEX) - M_complex * (W[1] + 1j * W[2])

        dZ_real = solve_lu(LU_real, f_real)
        # dW_complex = solve_lu(LU_complex, f_complex)

        # dW[0] = dW_real
        # dW[1] = dW_complex.real
        # dW[2] = dW_complex.imag

        dZ_norm = norm(dZ_real / scale)
        if dZ_norm_old is not None:
            rate = dZ_norm / dZ_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dZ_norm > tol)):
            break

        Z += dZ_real
        # Z = T.dot(W)

        if (dZ_norm == 0 or
                rate is not None and rate / (1 - rate) * dZ_norm < tol):
            converged = True
            break

        dZ_norm_old = dZ_norm

    return converged, k + 1, Z, rate


class RungeKuttaAdaptive(OdeSolver):
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    E: np.ndarray = NotImplemented
    P: np.ndarray = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented
    TOO_MANY_EVAL = "Function Evaluation exceeds the maximal time"
    def __init__(self, fun, t0, y0, t_bound, step=None, adaptive=True,
                 max_step=np.inf,
                 rtol=1e-3, atol=1e-6, first_step=None, beta_1=None,
                 beta_2=None, controller='IController', max_nfev = np.inf,
                 record_err=False, no_reject=False,
                 vectorized=False, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        self.y_old = None
        if step is not None:
            self.adaptive = False
        else:
            self.adaptive = adaptive
            self.max_nfev = max_nfev
        self.no_reject = no_reject
        if controller == 'PIController':
            self.is_pi_control = True
            if beta_1 is None:
                beta_1 = 0.7 / (self.error_estimator_order + 1)
            if beta_2 is None:
                beta_2 = 0.4 / (self.error_estimator_order + 1)
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.previous_error_norm = 1e-4
        else:
            self.is_pi_control = False
        self.f = self.fun(self.t, self.y)

        if self.adaptive:
            self.max_step = validate_max_step(max_step)
            self.rtol, self.atol = validate_tol(rtol, atol, self.n)
            if first_step is None:
                self.h_abs = select_initial_step(
                    self.fun, self.t, self.y, self.f, self.direction,
                    self.error_estimator_order, self.rtol, self.atol)
            else:
                self.h_abs = validate_first_step(first_step, t0, t_bound)
            self.error_exponent = -1 / (self.error_estimator_order + 1)
        else:
            self.h_abs = validate_max_step(step)
        self.record_err = record_err and self.adaptive
        if self.record_err:
            self.err = []
            if self.is_pi_control:
                self.err.append(self.previous_error_norm)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.h_previous = None
        if self.order >= 3: # construct P on the fly
            self.P = np.zeros([self.K.shape[0], 3])
            self.P[0, 0] = 1
            self.P[-1, 1] = -1
            self.P[-1, 2] = 1
            self.P[:-1, 1] = 3 * self.B
            self.P[:-1, 2] = -2 * self.B
            self.P[0, 1] -= 2
            self.P[0, 2] += 1

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

    def _step_impl(self):
        t = self.t
        y = self.y

        h_abs = self.h_abs

        if self.adaptive:
            max_step = self.max_step
            rtol = self.rtol
            atol = self.atol

            min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

            if self.h_abs > max_step:
                h_abs = max_step
            elif self.h_abs < min_step:
                h_abs = min_step
            else:
                h_abs = self.h_abs

        step_accepted = False

        while not step_accepted:
            if self.adaptive and h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            elif self.adaptive and self.nfev > self.max_nfev:
                return False, self.TOO_MANY_EVAL
            else:
                step_accepted = True
            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                    self.B, self.C, self.K)
            if self.adaptive:
                scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
                error_norm = self._estimate_error_norm(self.K, h, scale)
                if error_norm != 0:
                    if self.is_pi_control:
                        propose_q = error_norm ** (-self.beta_1)
                        if error_norm < 1 or self.no_reject:
                            propose_q *= self.previous_error_norm ** self.beta_2
                    else:
                        propose_q = error_norm ** self.error_exponent
                if error_norm < 1 or self.no_reject: # increase the step length
                    if error_norm == 0:
                        factor = MAX_FACTOR
                    else:
                        factor = min(MAX_FACTOR,
                                    max(MIN_FACTOR, SAFETY * propose_q))

                    # if step_rejected and factor > 1:
                    #    factor = min(1, factor)
                    if self.record_err:
                        self.err.append(error_norm)

                    h_abs *= factor

                    step_accepted = True
                    self.previous_error_norm = np.max([1e-4, error_norm])
                else: # decrease the step length
                    factor = min(MAX_FACTOR,
                                    max(MIN_FACTOR, SAFETY * propose_q))
                    h_abs *= factor
                    step_accepted = False

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)

class RK4(RungeKuttaAdaptive):
    """Explicit Runge-Kutta method of order 4.
       does not support adaptive step size
    """
    order = 4
    n_stages = 4
    C = np.array([0, 1/2, 1/2, 1])
    A = np.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 1/2, 0, 0],
        [0, 0, 1, 0]
    ])
    B = np.array([1/6, 1/3, 1/3, 1/6])
    # Hermite Cubic polynomial

class RKF4(RungeKuttaAdaptive):
    # mentioned in the following article
    # Low-order classical Runge-Kutta formulas with stepsize control and their application to some heat transfer problems
    order = 4
    n_stages = 5
    error_estimator_order = 3 # the order which is lower than itself
    C = np.array([0, 1/4, 4/9, 6/7, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0],
        [4/81, 32/81, 0, 0, 0],
        [57/98, -432/343, 1053/686, 0, 0],
        [1/6,0,27/52,49/156,0]
    ])
    B = np.array([43/288, 0, 243/416, 343/1872, 1/12])
    E = np.array([-5/288, 0, 27/416, -245/1872, 1/12, 0])

class Midpoint(RungeKuttaAdaptive):
    order = 2
    n_stages = 2
    error_estimator_order = 1
    C = np.array([0, 0.5])
    A = np.array([
        [0, 0],
        [1/2, 0]
    ])
    B = np.array([0, 1.0])
    # linear interpolation
    P = np.array([[0.0], [1.0], [0.0]])
    E = np.array([-1.0, 1.0, 0])
class BS3(RungeKuttaAdaptive):
    # Bogacki-Shampine method, also called ode23, RK23
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 0.5, 0.75])
    A = np.array([
        [0, 0, 0],
        [1/2, 0, 0],
        [0, 0.75, 0]
    ])
    B = np.array([2/9, 1/3, 4/9])
    E = np.array([5/72, -1/12, -1/9, 1/8])

class DOPRI5(RungeKuttaAdaptive):
    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])

class DOPRI853(RungeKuttaAdaptive):
    n_stages = dop853_coefficients.N_STAGES
    order = 8
    error_estimator_order = 7
    A = dop853_coefficients.A[:n_stages, :n_stages]
    B = dop853_coefficients.B
    C = dop853_coefficients.C[:n_stages]
    E3 = dop853_coefficients.E3
    E5 = dop853_coefficients.E5
    D = dop853_coefficients.D

    A_EXTRA = dop853_coefficients.A[n_stages + 1:]
    C_EXTRA = dop853_coefficients.C[n_stages + 1:]

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        super().__init__(fun, t0, y0, t_bound, max_step=max_step, rtol=rtol, atol=atol,
                         vectorized=vectorized, first_step=first_step, **extraneous)
        self.K_extended = np.empty((dop853_coefficients.N_STAGES_EXTENDED,
                                    self.n), dtype=self.y.dtype)
        self.K = self.K_extended[:self.n_stages + 1]

    def _estimate_error(self, K, h):  # Left for testing purposes.
        err5 = np.dot(K.T, self.E5)
        err3 = np.dot(K.T, self.E3)
        denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
        correction_factor = np.ones_like(err5)
        mask = denom > 0
        correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
        return h * err5 * correction_factor

    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self.E5) / scale
        err3 = np.dot(K.T, self.E3) / scale
        err5_norm_2 = np.linalg.norm(err5)**2
        err3_norm_2 = np.linalg.norm(err3)**2
        if err5_norm_2 == 0 and err3_norm_2 == 0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))

    def _dense_output_impl(self):
        K = self.K_extended
        h = self.h_previous
        for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA),
                                   start=self.n_stages + 1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        F = np.empty((dop853_coefficients.INTERPOLATOR_POWER, self.n),
                     dtype=self.y_old.dtype)

        f_old = K[0]
        delta_y = self.y - self.y_old

        F[0] = delta_y
        F[1] = h * f_old - delta_y
        F[2] = 2 * delta_y - h * (self.f + f_old)
        F[3:] = h * np.dot(self.D, K)

        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)

class ImplicitRungeKuttaAdaptive(OdeSolver):
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented
    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, step=None, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self.y_old = None
        if step is not None:
            self.adaptive = False
        else:
            self.adaptive = True
        self.f = self.fun(self.t, self.y)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        if self.adaptive:
            self.max_step = validate_max_step(max_step)            
            # Select initial step assuming the same order which is used to control
            # the error.
            if first_step is None:
                self.h_abs = select_initial_step(
                    self.fun, self.t, self.y, self.f, self.direction,
                    self.error_estimator_order, self.rtol, self.atol)
            else:
                self.h_abs = validate_first_step(first_step, t0, t_bound)
        else:
            self.h_abs = step
        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        self.sol = None

        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)
        if issparse(self.J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            I = eye(self.n, format='csc')
        else:
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I

        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y, f):
                self.njev += 1
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor,
                                             sparsity)
                return J
            J = jac_wrapped(t0, y0, self.f)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev = 1
            if issparse(J):
                J = csc_matrix(J)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=float)

            else:
                J = np.asarray(J, dtype=float)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
        else:
            if issparse(jac):
                J = csc_matrix(jac)
            else:
                J = np.asarray(jac, dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        t = self.t
        y = self.y
        f = self.f

        h_abs = self.h_abs
        atol = self.atol
        rtol = self.rtol

        if self.adaptive:
            max_step = self.max_step

            min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
            if self.h_abs > max_step:
                h_abs = max_step
                h_abs_old = None
                error_norm_old = None
            elif self.h_abs < min_step:
                h_abs = min_step
                h_abs_old = None
                error_norm_old = None
            else:
                h_abs = self.h_abs
                h_abs_old = self.h_abs_old
                error_norm_old = self.error_norm_old

        J = self.J
        LU_real = self.LU_real
        LU_complex = self.LU_complex

        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if self.adaptive and h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            step_accepted = True
            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            # if self.sol is None:
            Z0 = np.zeros((self.n_stages, y.shape[0]))
            # else:
            #    Z0 = self.sol(t + h * C).T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None:
                    # compute the LU decomposition of I - h A x J
                    LU_real = self.lu(self.I - h * self.A[0, 0] * J)
                    # LU_complex = self.lu(MU_COMPLEX / h * self.I - J)

                converged, n_iter, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Z0, scale, self.newton_tol,
                    LU_real, LU_complex, self.solve_lu, self.A, self.C)

                if not converged:
                    if current_jac:
                        break

                    J = self.jac(t, y, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            y_new = y + self.B[0] / self.A[0, 0] * Z[-1]
            # ZE = Z.T.dot(E) / h
            # error = self.solve_lu(LU_real, f + ZE)
            # scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # error_norm = norm(error / scale)
            # safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
            #                                            + n_iter)

            # if rejected and error_norm > 1:
            #     error = self.solve_lu(LU_real, self.fun(t, y + error) + ZE)
            #     error_norm = norm(error / scale)

            # if error_norm > 1:
            #     factor = predict_factor(h_abs, h_abs_old,
            #                             error_norm, error_norm_old)
            #     h_abs *= max(MIN_FACTOR, safety * factor)

            #     LU_real = None
            #     LU_complex = None
            #     rejected = True
            # else:
            #     step_accepted = True

        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = 1.0
        # factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        # factor = min(MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        # self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new

        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        # self.sol = self._compute_dense_output()

        return step_accepted, message

class ImplicitMidpoint(ImplicitRungeKuttaAdaptive):
    error_estimator_order = 1
    n_stages = 1
    C = np.array([1/2])
    A = np.array([
        [1/2]
    ])
    B = np.array([1.0])

class ImplicitEuler(ImplicitRungeKuttaAdaptive):
    error_estimator_order = 0
    n_stages = 1
    C = np.array([1.0])
    A = np.array([
        [1.0]
    ])
    B = np.array([1.0])