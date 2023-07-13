import math
import time
import warnings
import symengine as se

from gen.lib.base_model import Model
from gen.lib.sym_gen import JitParam

import collections
from typing import Union, Sequence
import random
from functools import lru_cache
from math import atan2, comb
import numpy as np
import scipy.linalg as spl
from jitcode import UnsuccessfulIntegration
from numpy.linalg import pinv, norm
from scipy.linalg import block_diag
import nlopt
from cnst.constant import constants
import pandas as pd
import matplotlib.pyplot as plt


def seed(a=None):
    random.seed(a)


@lru_cache(maxsize=None)
def discrete_set(N_coeff, N=500, ti=0, tf=1):
    assert ti < tf
    t = np.linspace(ti, tf, N)
    out = np.zeros((N_coeff, N))
    for i in range(N_coeff):
        out[i, :] = comb(N_coeff-1, i) * (t-ti)**i * (tf-t)**(N_coeff-1-i) / (tf-ti)**(N_coeff-1)
    return out


def discrete_traj(a, N_coeff, N=100, ti=0, tf=1):
    assert ti < tf
    if isinstance(a, np.ndarray):
        numel = a.size
    else:
        numel = len(a)
    return np.reshape(a, (numel//N_coeff, N_coeff)) @ discrete_set(N_coeff, N=N, ti=ti, tf=tf)


def bspline_deriv(ctrl_points, t, ti, tf, deriv, degree, N_dim):
    assert tf >= t >= ti
    assert tf > ti
    dt = tf - ti
    a = ctrl_points.reshape((N_dim, -1))
    point = np.zeros(N_dim)
    for k in range(deriv+1):
        for j in range(deriv - k , degree - k + 1):
            point[:] += math.factorial(degree) * comb(deriv, k) * (-1)**k * (t-ti)**(j-deriv+k) * (tf-t)**(degree-j-k) /\
                        (dt**degree * math.factorial(j-deriv+k) * math.factorial(degree-j-k)) * a[:, j]


    return point



def _spline_deriv_start(degree, N_deriv_max, dt):
    mat = np.zeros((N_deriv_max+1, N_deriv_max+1), dtype=np.object_)
    for N_deriv in range(N_deriv_max+1):
        for k in range(N_deriv+1):
            mat[N_deriv, N_deriv-k] = dt**(-N_deriv) * np.prod(np.arange(degree-N_deriv, degree)+1) * comb(N_deriv, k) * (-1)**k
    return mat


def _spline_deriv_end(degree, N_deriv_max, dt):
    mat = np.zeros((N_deriv_max+1, N_deriv_max+1), dtype=np.object_)
    for N_deriv in range(N_deriv_max+1):
        for k in range(N_deriv+1):
            mat[N_deriv, N_deriv_max-k] = dt**(-N_deriv) * np.prod(np.arange(degree-N_deriv, degree)+1) * comb(N_deriv, k) * (-1)**k
    return mat


@lru_cache()
def get_constraint_matrix_function(degree, N_deriv_max, N_dim):
    dt = se.Symbol("dt", real=True)
    beginning = _spline_deriv_start(degree, N_deriv_max, dt)
    end = _spline_deriv_end(degree, N_deriv_max, dt)
    M_part = spl.block_diag(beginning, end)
    M = spl.block_diag(*[M_part] * N_dim)
    M_func = se.lambdify(dt, M)
    return lambda dt: np.array(M_func(dt))


class PiecewiseSplineTrajectory:

    def __init__(self, time_vec, waypoints: Union[list, np.ndarray]):
        self._waypoints = np.array(waypoints)
        self._waypoints_t = np.array(time_vec)
        assert np.all(time_vec[1:] > time_vec[:-1])  # check if sorted
        self._N_waypoints = len(time_vec)
        tmp, self._N_jc, self._N_dim = self._waypoints.shape
        assert self._N_waypoints == tmp
        self._waypoints_mask = np.full(self.waypoints.shape, False, dtype=np.bool_)
        self._waypoints_t_mask = np.full(self._waypoints_t.shape, False)
        self.controller_gains = np.array([])
        self.controller_gains_along_traj = np.array([])
        self.traj_cost = np.array([])
        self.optimization_time = 0
        self.objective_fct_pre_opt = []
        self.N_coeffs = 2*self._N_jc
        self.time_points = []
        self.pos_all = []
        self.vel_all = []
        self.acc_all = []



    # This function calculates the control points a "shape of trajectory" between two waypoints
    def Calculate_Control_Point(self, wp_i, wp_f, DT):
        b = np.zeros([2 * self._N_jc * self._N_dim, 1])

        for ii in range(self._N_dim):
            for j in range(self._N_jc):
                b[2 * (self._N_jc * (ii) + j)] = wp_i[j, ii]
                b[2 * (self._N_jc * (ii) + j) + 1] = wp_f[j, ii]

        # constraints matrix

        M = np.block([
            [1, np.zeros(self.N_coeffs - 1)],
            [np.zeros(self.N_coeffs - 1), 1],
            [-self.N_coeffs / DT, self.N_coeffs / DT, np.zeros(self.N_coeffs - 2)],
            [np.zeros(self.N_coeffs - 2), -self.N_coeffs / DT, self.N_coeffs / DT],
            [self.N_coeffs * (self.N_coeffs - 1) / DT ** 2, -2 * self.N_coeffs * (self.N_coeffs - 1) / DT ** 2,
             self.N_coeffs * (self.N_coeffs - 1) / DT ** 2,
             np.zeros(self.N_coeffs - 3)],
            [np.zeros(self.N_coeffs - 3), self.N_coeffs * (self.N_coeffs - 1) / DT ** 2, -2 * self.N_coeffs * (self.N_coeffs - 1) / DT ** 2,
             self.N_coeffs * (self.N_coeffs - 1) / DT ** 2]
        ])


        M = block_diag(M, M, M, M)
        M_inv = pinv(M)

        return M_inv @ b  # Control points of the piece

    def Construct_trajectory(self): # argument is the sampling time of the trajectory
        c = constants() # class of constants
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        select = 0
        i = 0
        t = self._waypoints_t[0]
        pos_all = np.zeros([c.numTimestep, self._N_dim])
        vel_all = np.zeros([c.numTimestep, self._N_dim])
        acc_all = np.zeros([c.numTimestep, self._N_dim])

        while i <= c.numTimestep:
            # select the right piece of the trajectory (between 2 waypoints)

            if i == 0:  # We only enter this for the first iteration where select = 0 / i = 0
                wp_i = self._waypoints[select]
                wp_f = self._waypoints[select + 1]
                DT = self._waypoints_t[select + 1] - self._waypoints_t[select]
                a = self.Calculate_Control_Point(wp_i, wp_f, DT) # we call a function to calculate control points of each trajectory piece


            if self._waypoints_t[select + 1] <= t:  # We enter this whenever there is a change in the segment and thus we need to recalculate ctrl_pts [a]

                select = select + 1
                if select == self._N_waypoints -1:
                    break
                wp_i = self._waypoints[select]
                wp_f = self._waypoints[select + 1]
                DT = self._waypoints_t[select + 1] - self._waypoints_t[select]
                a = self.Calculate_Control_Point(wp_i, wp_f, DT)


            s = (t - self._waypoints_t[select])/DT  # s belongs between [0-->1] for each segment / starts at zero and ends at 1 at each segment

            # True all above

            bp_rd = np.zeros([self.N_coeffs, 1])
            bp_vd = np.zeros([self.N_coeffs, 1])
            bp_ad = np.zeros([self.N_coeffs, 1])
            traj_build = np.zeros([self._N_dim, self.N_coeffs])

            for ii in range(self._N_dim):
                traj_build[ii, :] = a[ii * self.N_coeffs: (ii + 1) * self.N_coeffs, 0]

            for ii in range(self.N_coeffs):

                bp_rd[ii] = (s ** ii) * ((1 - s) ** (self.N_coeffs - 1 - ii)) * math.comb(self.N_coeffs - 1, ii)
                bp_vd[ii] = (ii * (s ** (ii - 1)) * ((1 - s) ** (self.N_coeffs - 1 - ii)) - (self.N_coeffs - 1 - ii) * (
                            s ** ii) * ((1 - s) ** (self.N_coeffs - ii - 2))) * math.comb(self.N_coeffs - 1, ii) / DT

                bp_ad[ii] = (ii * (ii - 1) * (s ** (ii - 2)) * ((1 - s) ** (self.N_coeffs - 1 - ii)) - 2 * ii * (
                            self.N_coeffs - 1 - ii) * (s ** (ii - 1)) * ((1 - s) ** (self.N_coeffs - ii - 2)) + (self.N_coeffs - 1 - ii) * (self.N_coeffs - ii - 2) * (s ** ii) *
                                         (1 - s) ** (self.N_coeffs - ii - 3)) * math.comb(self.N_coeffs - 1, ii) / DT**2

            pos_all[i, :] = bp_rd.T @ traj_build.T
            vel_all[i, :] = bp_vd.T @ traj_build.T
            acc_all[i, :] = bp_ad.T @ traj_build.T
            # updating time and updating counts
            t += c.Ts
            i += 1

        vel_all[0, :] = 0 #just because the first iterations gives nan in vel and acc so we just replace it with zero for other functions
        acc_all[0, :] = 0 #just because the first iterations gives nan in vel and acc so we just replace it with zero for other functions

        if pos_all[-1, 0] or pos_all[-1, 1] == 0:
            self.pos_all = np.array(pos_all[:-1, :])
            self.vel_all = np.array(vel_all[:-1, :])
            self.acc_all = np.array(acc_all[:-1, :])

        else:
            self.pos_all = np.array(pos_all)
            self.vel_all = np.array(vel_all)
            self.acc_all = np.array(acc_all)





    def waypoints(self):
        return self._waypoints_t

    def return_obj_cost(self):
        return self.objective_fct_pre_opt

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Expected CompositeSplineTrajectory not {type(other)}")
        return (
            np.all(self._waypoints == other._waypoints)
            and np.all(self._waypoints_t == other._waypoints_t)
            and np.all(self._waypoints_mask == other._waypoints_mask)
            and self._N_waypoints == other._N_waypoints
            and np.all(self._waypoints_t_mask == other._waypoints_t_mask)
        )

    def __deepcopy__(self, memodict={}):
        new_obj = PiecewiseSplineTrajectory(self._waypoints_t.copy(), self._waypoints.copy())
        new_obj._waypoints_mask[:] = self._waypoints_mask[:]
        new_obj._waypoints_t_mask[:] = self._waypoints_t_mask[:]
        return new_obj

    def deepcopy(self):
        return self.__deepcopy__()


    def optimize(self, model, nlc, target_point, use_target=False, lower_bounds=None, upper_bounds=None, optim_time=10.0, N = 500, dx_init=0.1):

        model.set_default_state(self._waypoints[0])
        model.ODE.set_initial_value()
        self.objective_fct_pre_opt.clear()

        if nlc is None:
            return self, target_point
        N_eval = [0]

        def objective(x, grad):
            N_eval[0] += 1
            self.update_from_flat_params(x)
            model.ODE.set_initial_value(time=self.waypoints_t[0])
            ti, tf = self.waypoints_t[0], self.waypoints_t[-1]
            for ctrl_points, ti_part, tf_part in self.traj_iter():
                time_vector = np.linspace(ti_part, tf_part, max(2, round(N * (tf_part - ti_part) / (tf - ti))))
                model.ODE["a"] = ctrl_points
                model.ODE.integrate_on_time_vector(time_vector)
            states = model.ODE.last_result
            cost = _pre_opti_cost_from_states(model, states, model.ODE.time_points, target_point, use_target)

            self.objective_fct_pre_opt.append(cost)


            return cost

        def soft_bounds(x, time_vector, states, lb, ub):
            pos = states[:, model.output_indices()]
            lower = np.max(lb - pos, axis=0)
            upper = np.min(pos - ub, axis=0)
            return np.concatenate([lower, upper])

        def nlc_wrap(result, x, grad):
            states = model.ODE.last_result
            time_vector = model.ODE.time_points
            cond1 = nlc(grad, x, time_vector, states)
            result[:len(cond1)] = cond1
            result[len(cond1):] = soft_bounds(x, time_vector, states, lb2, ub2)
            # print(result)
            return result[:]

        def eq_constraints(result, x, grad):
            states = model.ODE.last_result
            times_vec = model.ODE.time_points
            result[:] = model.eq_constraints(grad, states, times_vec, target_point)
            return result

        # OPTIMIZE

        tic = time.time()
        x_ini = self.get_flat_free_params()
        opt = nlopt.opt(nlopt.LN_COBYLA, len(x_ini))  # choosing optimizer for pre-conditionning

        # input saturation and minimum curvature radius contraints, first pre-conditionning of the trajectory,
        # to make the trajectory dynamically feasible by the system

        opt.add_inequality_mconstraint(nlc_wrap, np.concatenate([model.nonlcon_tol(), 1e-5*np.ones(2*len(model.output_indices()))]))

        if model.eq_constraints_tol(target_point):
            self._waypoints_mask[-1, :, :] = True
            if use_target == True:  # for second pre-opt, with imperfect controller, target reach constraint
                opt.add_equality_mconstraint(eq_constraints, model.eq_constraints_tol(target_point))

        if lower_bounds is None:
            lower_bounds = np.full(np.size(x_ini), -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full(np.size(x_ini), np.inf)

        ub2 = upper_bounds[0, :len(model.output_indices())]
        lb2 = lower_bounds[0, :len(model.output_indices())]
        lb, ub = self.generate_bounds()
        if lower_bounds.shape == self.waypoints[0].shape:
            for i in range(lb.shape[0]):
                lb[i, 1:] = np.maximum(lower_bounds.flatten(), lb[i, 1:])
        elif lower_bounds.size == lb.size:
            lb = np.maximum(lower_bounds.flatten(), lb.flatten())
        if upper_bounds.shape == self.waypoints[0].shape:
            for i in range(ub.shape[0]):
                ub[i, 1:] = np.minimum(upper_bounds.flatten(), ub[i, 1:])
        elif upper_bounds.size == ub.size:
            ub = np.minimum(upper_bounds.flatten(), ub.flatten())
        opt.set_lower_bounds(lb.flatten())
        opt.set_upper_bounds(ub.flatten())
        x_ini = np.minimum(ub.flatten(), np.maximum(lb.flatten(), x_ini))
        opt.set_initial_step(dx_init)
        opt.set_ftol_rel(1e-6)
        opt.set_ftol_abs(1e-3)
        opt.set_xtol_rel(1e-3)


        opt.set_min_objective(objective)
        opt.set_maxtime(60 * optim_time)
        init_cost = objective(x_ini, None)
        x_final = opt.optimize(x_ini)
        self.update_from_flat_params(x_final)
        cost = np.array(self.objective_fct_pre_opt)
        print(
            f"Obj improvement: {init_cost:.2e} --> {opt.last_optimum_value():.2e} "
            f"in {time.time() - tic} with exit code {opt.last_optimize_result()} in {opt.get_numevals()} {N_eval} evals"
        )
        return cost



    @property
    def waypoints(self):
        return self._waypoints

    @property
    def waypoints_t(self):
        return self._waypoints_t

    @property
    def waypoints_mask(self):
        return self._waypoints_mask

    @property
    def N_waypoints(self):
        return self._N_waypoints

    @property
    def N_jc(self):
        return self._N_jc

    @property
    def N_dim(self):
        return self._N_dim

    @property
    def N_deriv(self):
        return self._N_jc -1

    @property
    def degree(self):
        return 2*self._N_jc - 1

    def free_waypoint(self, index):
        if index == 0:
            raise RuntimeError("cannot free 1st waypoint")
        self._waypoints_mask[index] = True

    def lock_waypoint(self, index):
        self._waypoints_mask[index] = False

    def generate_bounds(self):
        relative_margin = 0.01
        l_stack = np.full((self.N_waypoints, self.N_jc*self.N_dim + 1), -np.inf)
        u_stack = np.full((self.N_waypoints, self.N_jc*self.N_dim + 1), np.inf)
        # preventing points in the past
        ti = self._waypoints_t[0]
        tf = self.waypoints_t[-1]
        dt = (tf - ti) * relative_margin
        l_stack[1:, 0] = ti + dt
        u_stack[1:, 0] = tf
        #flat non zero returns the indices of non zero values
        indices = np.flatnonzero(np.diff(self._waypoints_t_mask))
        for start, end in zip(indices[::2], indices[1::2] + 1):
            ti = self._waypoints_t[start]
            tf = self.waypoints_t[end]
            dt = (tf - ti)*relative_margin
            l_stack[start+1:end, 0] = ti + dt
            u_stack[start+1:end, 0] = tf - dt
        ctrl_mask = np.reshape(self._waypoints_mask, newshape=(self.N_waypoints, self.N_jc*self.N_dim))
        waypoints_tmp = np.reshape(self.waypoints, newshape=(self.N_waypoints, self.N_jc*self.N_dim))
        l_stack[:, 1:][ctrl_mask == False] = u_stack[:, 1:][ctrl_mask == False] = waypoints_tmp[ctrl_mask == False]
        l_stack[:, 0][self._waypoints_t_mask == False] = u_stack[:, 0][self._waypoints_t_mask == False] = self.waypoints_t[self._waypoints_t_mask == False]
        return l_stack, u_stack

    def nlc(self, flat_params, grad):
        stack = np.reshape(flat_params, (self.N_waypoints, -1))
        if grad is not None and grad.size > 0:
            raise NotImplementedError("")
        dt = stack[:, 0]
        indices = np.flatnonzero(np.diff(self._waypoints_t_mask))
        condition = np.zeros(len(indices)//2)
        if self._waypoints_t_mask[indices[0]]:
            raise RuntimeError()
        i = 0
        for start, end in zip(indices[::2], indices[1::2] + 1):
            ti = self._waypoints_t[start]
            tf = self.waypoints_t[end]
            plage_dt = np.sum(dt[start+1:end])
            condition[i] = plage_dt - (tf-ti)
            i += 1
        return condition

    def get_flat_free_params(self):
        stack = np.zeros((self.N_waypoints, self.N_jc*self.N_dim + 1))
        stack[:, 1:] = np.reshape(self._waypoints, (self.N_waypoints, -1))
        stack[:, 0] = self._waypoints_t
        return stack.flatten()

    # done for controller gain opt
    def update_controller_gains(self, params):
        self.controller_gains = params

    def save_controller_gains_along_trajectory(self, params):
        self.controller_gains_along_traj = params

    def save_cost_along_trajectory(self, cost):
        self.traj_cost = cost

    def set_opt_time(self, opt_t):
        self.optimization_time = opt_t

    def get_opt_time(self):
        return self.optimization_time
    # done for controller gain opt

    def get_controller_gains(self):
        return self.controller_gains

    def get_controller_gains_along_trajectory(self):
        return self.controller_gains_along_traj

    def get_cost_along_trajectory(self):
        return self.traj_cost

    def update_from_flat_params(self, params):
        stack = np.reshape(params, (self.N_waypoints, -1))
        waypoints = stack[:, 1:]
        waypoints_t = stack[:, 0]
        indices = np.flatnonzero(np.diff(self._waypoints_t_mask))
        order = np.arange(self.N_waypoints, dtype=int)
        corrected_times = [np.zeros(0)]
        for start, end in zip(indices[::2], indices[1::2]+1):
            ti = self._waypoints_t[start]
            tf = self._waypoints_t[end]
            dt = (tf - ti)*1e-2
            ids = start + 1 + np.argsort(waypoints_t[start+1:end])
            order[start+1:end] = ids
            t_vec = waypoints_t[ids]
            t_vec[1:] = np.maximum(
                t_vec[:-1]+dt,
                t_vec[1:]
            )
            if t_vec[0] < ti + dt:
                t_vec += (ti + dt) - t_vec[0]
            if t_vec[-1] > tf - dt:
                t_vec = (t_vec - t_vec[0])*(tf - dt - t_vec[0])/(t_vec[-1] - t_vec[0]) + t_vec[0]
            corrected_times.append(t_vec)
        if len(indices) % 2 == 1:
            start = indices[-1]
            ti = self._waypoints_t[start]
            tf = self._waypoints_t[-1]
            dt = (tf - ti) * 1e-3
            ids = start + 1 + np.argsort(waypoints_t[start+1:])
            order[start+1:] = ids
            t_vec = waypoints_t[ids]
            t_vec[1:] = np.maximum(
                t_vec[:-1]+dt,
                t_vec[1:]
            )
            if t_vec[0] < ti + dt:
                t_vec += (ti + dt) - t_vec[0]
            corrected_times.append(t_vec)
        self._waypoints[self._waypoints_mask] = waypoints[order].reshape((-1, self._N_jc, self._N_dim))[self._waypoints_mask]
        self._waypoints_t[self._waypoints_t_mask] = np.concatenate(corrected_times)

    def part_traj(self, i):
        if not (0<=i<=self._N_waypoints-1):
            raise RuntimeError(
                f"i is not a valid index, must be between 0 and N_waypoints-1 ({i} not in [0; {self._N_waypoints-1}])"
            )
        start_conditions = self._waypoints[i]
        end_conditions = self._waypoints[i+1]
        dt = self._waypoints_t[i+1] - self._waypoints_t[i]
        M = get_constraint_matrix_function(self.degree, self.N_deriv, self.N_dim)
        b = []
        for dim in range(self.N_dim):
            b.append(start_conditions[:, dim])
            b.append(end_conditions[:, dim])
        b = np.hstack(b)
        #debug (also you can display dt, which sometimes goes to 0..)
        # print(M(dt))
        #print(b)
        return spl.solve(M(dt), b), self._waypoints_t[i], self._waypoints_t[i+1]

    def traj_iter(self):
        for i in range(self.N_waypoints-1):
            yield self.part_traj(i)

    def __iter__(self):
        for i in range(self.N_waypoints):
            yield self.waypoints_t[i], self.waypoints[i]

    def insert_waypoint(self, t, waypoint: np.ndarray, free_mask=True, t_free=True):
        if isinstance(free_mask, bool):
            free_mask = np.full((self.N_jc, self.N_dim), free_mask)
        if not(isinstance(free_mask, np.ndarray)) or np.shape(free_mask) != (self.N_jc, self.N_dim) or free_mask.dtype != np.bool_:
            raise ValueError("free_mask must either a boolean or a (N_jc, N_dim) numpy array of dtype bool")
        index = np.searchsorted(self._waypoints_t, t)
        if self._waypoints_t[index] == t:
            raise RuntimeError(f"waypoint already defined at t={t}")

        self._waypoints_t = self._waypoints_t.astype(float)
        self._waypoints_t = np.insert(self._waypoints_t, index, t)
        self._waypoints = np.insert(self._waypoints, index, waypoint, axis=0)
        self._waypoints_mask = np.insert(self._waypoints_mask, index, free_mask, axis=0)
        self._waypoints_t_mask = np.insert(self._waypoints_t_mask, index, t_free, axis=0)
        self._N_waypoints += 1

    def interpolate_waypoint_at(self, t, free=True):
        if t < self._waypoints_t[0] or t > self._waypoints_t[-1]:
            raise RuntimeError(f"Cannot interpolate waypoint outside of already defined trajectory")
        if t in self._waypoints_t:
            raise RuntimeError(f"waypoint already defined at t={t}")
        index = np.searchsorted(self._waypoints_t, t) - 1
        ctrl_points, ti, tf = self.part_traj(index)
        waypoint = np.zeros((self.N_jc, self.N_dim))
        for deriv in range(self.N_jc):
            waypoint[deriv, :] = bspline_deriv(ctrl_points, t, ti, tf, deriv, self.degree, self.N_dim)
        self.insert_waypoint(t, waypoint, free_mask=free)



def _pre_opti_cost_from_states(model: Model, states, time_vec, target, target_reach_flag):
    curve = states[:, model.output_indices()]
    dcurve = np.diff(curve.T).T
    # length
    length = np.sum(norm(dcurve, axis=1))
    # curvature
    max_curve = 1 / min_curvature_radius(curve, time_vec, dcurve)
    # min_curve = min_curvature_radius(curve, time_vec, dcurve)
    # print(length, max_curve, dispersion)
    kl = 10
    kc = 0.02
    # kc = 0.02
    # k_target = 38
    k_target = 10
    distance_to_target = norm(states[-1, :3] - target[0, :3])
    # return kl*length + kc * max_curve + 100*(time_vec[-1]-time_vec[0]) + k_target*distance_to_target
    cost1 = kl*length + kc * max_curve
    # minimum snap trajectory
    pos = states[:, model.output_indices()]  # get 3D position
    vel = np.gradient(pos, time_vec, axis=0)  # velocity
    acc = np.gradient(vel, time_vec, axis=0)  # acceleration
    jerk = np.gradient(acc, time_vec, axis=0)  # jerk
    snap = np.gradient(jerk, time_vec, axis=0)  # snap
    cost2 = 0
    for i in range(np.shape(snap)[0]):
        cost2 += np.linalg.norm(snap[i, :])
    cost = cost1 + cost2

    return distance_to_target



def min_curvature_radius(curve, time_vec, dcurve=None):
    """
    Input must be of form (N_samples, N_outputs)
    :param curve:
    :param time_vec:
    :param dcurve:
    :return:
    """
    dt_vec = np.diff(time_vec)
    if dcurve is None:
        dcurve = (np.diff(curve.T) / dt_vec).T
    ddcurve = (np.diff(dcurve.T) / dt_vec[:-1]).T
    dcurve = dcurve[:ddcurve.shape[0], :]
    #r_curve = (dcurve[:, 0] ** 2 + dcurve[:, 1] ** 2) ** (3 / 2) / (
    #            dcurve[:, 0] * ddcurve[:, 1] - dcurve[:, 1] * ddcurve[:, 0])
    if curve.shape[1] == 2:
        r_curve = norm(dcurve, axis=1)**3/np.cross(ddcurve, dcurve, axis=1)
    elif curve.shape[1] == 3:
        if np.any(norm(np.cross(ddcurve, dcurve, axis=1), axis=1) == 0.0):
            print(end="")
        r_curve = norm(dcurve, axis=1)**3/norm(np.cross(ddcurve, dcurve, axis=1), axis=1)
    else:
        raise ValueError("Must be of dimension (N_samples, 2) or (N_samples, 3)")
    #r_curve = norm(dcurve, axis=1) ** 3 / np.sqrt(
    #    np.sum(ddcurve**2, axis=1)*np.sum(dcurve**2, axis=1) - np.sum(ddcurve*dcurve, axis=1)**2)
    return norm(r_curve, ord=-np.inf)
