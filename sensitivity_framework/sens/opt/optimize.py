from numpy.linalg import norm

from gen.lib.base_model import Model
from gen.lib.sym_gen import JitParam
from utils.trajectory import PiecewiseSplineTrajectory
import numpy as np
import nlopt
from time import time
from jitcode import UnsuccessfulIntegration
import matplotlib.pyplot as plt
from cnst.constant import constants
cost_eval_count = 0


def cost(trajectory: PiecewiseSplineTrajectory, ODE: JitParam, num=100, PI_mask=None):

    c = constants()
    global cost_eval_count
    cost_eval_count += 1
    ODE.set_initial_value()
    dt_total = trajectory.waypoints_t[-1] - trajectory.waypoints_t[0]

    for ctrl_points, ti, tf in trajectory.traj_iter():
        time_vector = np.linspace(ti, tf, round(num*(tf-ti)/dt_total))
        ODE["a"] = ctrl_points
        try:
            ODE.integrate_on_time_vector(time_vector)
        except UnsuccessfulIntegration:  # trying to salvage the situation by adding noise
            print("Unsuccessful Integration. Attempting salvage by adding noise to the control points")
            successful_integration = False
            for i in range(100):
                print(f"salvage attempt {i}")
                ODE["a"] += np.random.normal(scale=0.001, size=ctrl_points.shape)
                ODE.set_initial_value()
                try:
                    ODE.integrate_on_time_vector(time_vector)
                except UnsuccessfulIntegration:
                    pass
                else:
                    successful_integration = True
                    break
            if not successful_integration:
                raise

    states = ODE.last_result
    N_p = len(ODE.states_indices["PI"]) // len(ODE.states_indices["q"])
    PI = np.reshape(states[-1, ODE.states_indices["PI"]], (-1, N_p))
    PI = PI[:3, :]

    A = PI @ c.W_range @ PI.T
    eigvals, eigvecs = np.linalg.eig(A)  # get eigenvalues of this kernel matrix
    eigvals, eigvecs = eigvals.real, eigvecs.real

    return np.sqrt(np.max(eigvals))



# This function is just for the opt of the control points
def sensitivity_optimisation(model: Model, trajectory: PiecewiseSplineTrajectory, nonlcon, target_point=None,
                             lower_bounds=None, upper_bounds=None, PI_mask=None, optimization_time=None, delta=None):
    """

    :param PI_mask:
    :param target_point:
    :param lower_bounds:
    :param upper_bounds:
    :param output_times:
    :param ODE:
    :param trajectory:
    :param nonlcon:
        must be of form cost = nonlcon(grad, x, time_vec, states_vec)
        with states_vec of shape (N_samples, N_states)
        and time_vec contains the time of each sample
    :return:
    """

    cost_list = []
    param_list = []
    global cost_eval_count
    trajectory = trajectory.deepcopy()  # local copy
    N = 500
    ODE: JitParam = model.ODE
    if trajectory.N_dim*(trajectory.degree+1) != np.size(ODE["a"]):
        raise RuntimeError()

    x_ini = trajectory.get_flat_free_params()


    if "grad" in ODE._modulename.split("_"):
        opt = nlopt.opt(nlopt.LD_MMA, len(x_ini))
    elif "nograd" in ODE._modulename.split("_"):
        opt = nlopt.opt(nlopt.LN_COBYLA, len(x_ini))

        # opt = nlopt.opt(nlopt.GN_ISRES, len(x_ini))
    else:
        raise ValueError("ODE is not valid")

    if lower_bounds is None:
        lower_bounds = np.full(np.size(x_ini), -np.inf)
    if upper_bounds is None:
        upper_bounds = np.full(np.size(x_ini), np.inf)

    lb, ub = trajectory.generate_bounds()
    if lower_bounds.shape == trajectory.waypoints[0].shape:
        for i in range(lb.shape[0]):
            lb[i, 1:] = np.maximum(lower_bounds.flatten(), lb[i, 1:])
    elif lower_bounds.size == lb.size:
        lb = np.maximum(lower_bounds.flatten(), lb.flatten())
    if upper_bounds.shape == trajectory.waypoints[0].shape:
        for i in range(ub.shape[0]):
            ub[i, 1:] = np.minimum(upper_bounds.flatten(), ub[i, 1:])
    elif upper_bounds.size == ub.size:
        ub = np.minimum(upper_bounds.flatten(), ub.flatten())
    opt.set_lower_bounds(lb.flatten())
    opt.set_upper_bounds(ub.flatten())
    opt.set_population(int((len(x_ini)-np.sum(ub == lb)+1)*20))


    def nlc(result, x, grad):
        if np.any(x != last_x):
            trajectory.update_from_flat_params(x)
            model.integrate_along_trajectory(trajectory, N=N)
        states = ODE.last_result
        time_vector = ODE.time_points
        cond1 = nonlcon(grad, x, time_vector, states)
        result[:] = cond1
        return result[:]

    def _cost(x, grad):
        if grad is not None and grad.size > 0:
            raise NotImplementedError("Will see later for gradient based optimisation")
        global last_x
        last_x = x
        trajectory.update_from_flat_params(x)
        PI = cost(trajectory, ODE, num=N, PI_mask=PI_mask)
        cost_list.append(PI)
        param_list.append(x.copy())

        return PI

    opt.add_inequality_mconstraint(nlc, model.nonlcon_tol())
    if model.eq_constraints_tol(target_point) is not None:
        def eq_constraints(result, x, grad):
            states = ODE.last_result
            times_vec = ODE.time_points
            result[:] = model.eq_constraints(grad, states, times_vec, target_point)
            return result
        opt.add_equality_mconstraint(eq_constraints, model.eq_constraints_tol(target_point))
    opt.set_ftol_rel(1e-6)
    opt.set_xtol_abs(1e-6)
    opt.set_maxeval(10000)
    opt.set_maxtime(60*optimization_time)
    opt.set_initial_step(np.maximum(-5e-2*(np.minimum(ub.flatten(), 10)-np.maximum(lb.flatten(), -10)), delta))



    # optimize PI
    cost_eval_count = 0
    t0 = time()
    opt.set_min_objective(_cost)
    PI_init_cost = _cost(x_ini, None)
    PI_a_opt = opt.optimize(x_ini)
    PI_opt_cost = opt.last_optimum_value()
    T_PI = time() - t0

    cost_list = np.array(cost_list)
    param_list = np.array(param_list)

    # take the minimum index of the opt
    index = np.argmin(cost_list)


    print(
        f"PI improvement: {PI_init_cost:.2e} --> {cost_list[index]:.2e} in {T_PI} with exit code "
        f"{opt.last_optimize_result()} in {opt.get_numevals()} ({cost_eval_count}) evals"
    )

    PI_a_opt = param_list[index, :]
    PI_out = trajectory.deepcopy()
    PI_out.save_cost_along_trajectory(cost_list)
    PI_out.set_opt_time(optimization_time)

    PI_out.update_from_flat_params(PI_a_opt)

    return PI_out, cost_list[index], cost_list



def sensitivity_joint_optimisation(model: Model, trajectory: PiecewiseSplineTrajectory, nonlcon, target_point=None,
                             lower_bounds=None, upper_bounds=None, PI_mask=None, gains=None,optimization_time=None, delta=None):
    """

    :param PI_mask:
    :param target_point:
    :param lower_bounds:
    :param upper_bounds:
    :param output_times:
    :param ODE:
    :param trajectory:
    :param nonlcon:
        must be of form cost = nonlcon(grad, x, time_vec, states_vec)
        with states_vec of shape (N_samples, N_states)
        and time_vec contains the time of each sample
    :return:
    """

    param_list = []
    k = []
    cost_list = []

    global cost_eval_count
    trajectory = trajectory.deepcopy()  # local copy
    N = 500
    ODE: JitParam = model.ODE
    if trajectory.N_dim*(trajectory.degree+1) != np.size(ODE["a"]):
        raise RuntimeError()

    # we want to add the length of the controller gains to the length of x_ini which would be now (control points+ controller gains)
    x_ini_gains = gains
    x_ini = trajectory.get_flat_free_params()
    L = len(x_ini)+len(x_ini_gains)

    #in this we have to add the length of x_ini_gains because we will concatenate the two later on in x_ini
    if "grad" in ODE._modulename.split("_"):
        opt = nlopt.opt(nlopt.LD_MMA, L)
    elif "nograd" in ODE._modulename.split("_"):
        # this is the object instructor for optimisation opt = nlopt.opt(algorithm, n) # n is the length of opt parameters
        opt = nlopt.opt(nlopt.LN_COBYLA, L)
    else:
        raise ValueError("ODE is not valid")


    '''The algorithm and dimension parameters of the object are immutable (cannot be changed without constructing a new object), but you can query them for a given object by the methods:
        opt.get_algorithm()
        opt.get_dimension()   '''

    if lower_bounds is None:
        lower_bounds = np.full(np.size(x_ini), -np.inf)
    if upper_bounds is None:
        upper_bounds = np.full(np.size(x_ini), np.inf)

    lb, ub = trajectory.generate_bounds()
    if lower_bounds.shape == trajectory.waypoints[0].shape:
        for i in range(lb.shape[0]):
            lb[i, 1:] = np.maximum(lower_bounds.flatten(), lb[i, 1:])
    elif lower_bounds.size == lb.size:
        lb = np.maximum(lower_bounds.flatten(), lb.flatten())
    if upper_bounds.shape == trajectory.waypoints[0].shape:
        for i in range(ub.shape[0]):
            ub[i, 1:] = np.minimum(upper_bounds.flatten(), ub[i, 1:])
    elif upper_bounds.size == ub.size:
        ub = np.minimum(upper_bounds.flatten(), ub.flatten())

    # for setting lower bounds , we need to add the bounds of the controller gains
    # For now , we consider the bounds for the controller gains as follows:
    '''  For setting lower bounds , we need to add the bounds of the controller gains
    For now , we consider the bounds for the controller gains as follows:
    1- lower bounds for the gains are considerd 2 percent of the manually tunned gains
    2- the upper bound is considered 400 percent of the manually tunned gains
    so, we can write the following: '''
    lb_gains = 0.5    * x_ini_gains
    ub_gains = 2       * x_ini_gains

    lb_ctrl_p = lb.flatten()
    ub_ctrl_p = ub.flatten()

    lb = np.concatenate((lb.flatten(), lb_gains), axis=None)
    ub = np.concatenate((ub.flatten(), ub_gains), axis=None)

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)



    def nlc(result, x_t, grad):
        if np.any(x_t != last_x):
            x = x_t[:len(x_ini)]
            trajectory.update_from_flat_params(x)
            #here we update the controller gains
            ODE["Array_gains"] = x_t[len(x_ini):]
            ODE.apply_parameters()
            model.integrate_along_trajectory(trajectory, N=N)
        x = x_t[:len(x_ini)]
        states = ODE.last_result
        time_vector = ODE.time_points
        cond1 = nonlcon(grad, x, time_vector, states)
        result[:] = cond1
        return result[:]

    def _cost(x_t, grad):
        if grad is not None and grad.size > 0:
            raise NotImplementedError("Will see later for gradient based optimisation")
        global last_x
        last_x = x_t
        x = x_t[:len(x_ini)]
        trajectory.update_from_flat_params(x)
        tmp = x_t[len(x_ini):]
        ODE["Array_gains"] = tmp
        ODE.apply_parameters()
        # it is not efficient, due to time constrain and existent of a bug, i wrote them explicity
        #TODO: we should find the bug and why that we can't use array notation instead of explicit def
        param_list.append(x.copy())
        k.append(tmp.copy())
        PI = cost(trajectory, ODE, num=N, PI_mask=PI_mask)
        cost_list.append(PI)

        return PI

    opt.add_inequality_mconstraint(nlc, model.nonlcon_tol())
    if model.eq_constraints_tol(target_point) is not None:
        def eq_constraints(result, x, grad):
            states = ODE.last_result
            times_vec = ODE.time_points
            result[:] = model.eq_constraints(grad, states, times_vec, target_point)
            return result
        opt.add_equality_mconstraint(eq_constraints, model.eq_constraints_tol(target_point))

    #Set relative tolerance on function value.
    opt.set_ftol_rel(1e-6)
    #Set absolute tolerance on function value.
    opt.set_xtol_abs(1e-6)
    x_ini_t = np.concatenate((x_ini, x_ini_gains), axis=None)
    opt.set_maxeval(10000)
    opt.set_maxtime(60*optimization_time)

    dx1 = np.maximum(-5e-2 * (np.minimum(ub_ctrl_p, 10) - np.maximum(lb_ctrl_p, -10)), delta)
    dx2 = delta * x_ini_gains
    dx = np.concatenate((dx1, dx2), axis=0)
    opt.set_initial_step(dx)


    # optimize PI
    cost_eval_count = 0
    t0 = time()
    opt.set_min_objective(_cost)
    PI_init_cost = _cost(x_ini_t, None)
    PI_x_opt = opt.optimize(x_ini_t)
    PI_opt_cost = opt.last_optimum_value()
    T_PI = time() - t0

    cost_list = np.array(cost_list)
    param_list = np.array(param_list)
    k = np.array(k)

    # take the minimum index of the opt
    index = np.argmin(cost_list)

    print(
        f"PI improvement: {PI_init_cost:.2e} --> {cost_list[index]:.2e} in {T_PI} with exit code "
        f"{opt.last_optimize_result()} in {opt.get_numevals()} ({cost_eval_count}) evals"
    )

    PI_a_opt = param_list[index, :]
    #parameters for the gains
    PI_gains_opt = k[index, :]
    print(f'here we are printing the new gains from the optimisation: {PI_gains_opt} ')


    PI_out = trajectory.deepcopy()
    PI_out.save_controller_gains_along_trajectory(k)
    PI_out.save_cost_along_trajectory(cost_list)
    PI_out.set_opt_time(optimization_time)
    # We save the gains here after optimisation
    PI_out.update_controller_gains(PI_gains_opt)

    # update the waypoints of the trajectory
    PI_out.update_from_flat_params(PI_a_opt)

    return PI_out, cost_list[index], cost_list




def sensitivity_gain_optimisation(model: Model, trajectory: PiecewiseSplineTrajectory, nonlcon, target_point=None,
                                                 PI_mask=None, gains=None,optimization_time=None, delta= None):
    """

    :param PI_mask:
    :param target_point:
    :param lower_bounds:
    :param upper_bounds:
    :param output_times:
    :param ODE:
    :param trajectory:
    :param nonlcon:
        must be of form cost = nonlcon(grad, x, time_vec, states_vec)
        with states_vec of shape (N_samples, N_states)
        and time_vec contains the time of each sample
    :return:
    """
    k = []
    cost_list = []
    global cost_eval_count
    trajectory = trajectory.deepcopy()  # local copy
    N = 500
    ODE: JitParam = model.ODE
    if trajectory.N_dim*(trajectory.degree+1) != np.size(ODE["a"]):
        raise RuntimeError()


    x_ini_gains = gains
    L = len(x_ini_gains)

    #in this we have to add the length of x_ini_gains because we will concatenate the two later on in x_ini
    if "grad" in ODE._modulename.split("_"):
        opt = nlopt.opt(nlopt.LD_MMA, L)
    elif "nograd" in ODE._modulename.split("_"):
        # this is the object instructor for optimisation opt = nlopt.opt(algorithm, n) # n is the length of opt parameters
        opt = nlopt.opt(nlopt.LN_COBYLA, L)
         # opt = nlopt.opt(nlopt.GN_ISRES, L)



    else:
        raise ValueError("ODE is not valid")

    # for setting lower bounds , we need to add the bounds of the controller gains
    # For now , we consider the bounds for the controller gains as follows:
    '''  For setting lower bounds , we need to add the bounds of the controller gains
    For now , we consider the bounds for the controller gains as follows:
    1- lower bounds for the gains are considerd 2 percent of the manually tunned gains
    2- the upper bound is considered 400 percent of the manually tunned gains
    so, we can write the following: '''
    lb_gains = 0.5    * x_ini_gains
    ub_gains = 2       * x_ini_gains
    opt.set_lower_bounds(lb_gains)
    opt.set_upper_bounds(ub_gains)

    def nlc(result, x_t, grad):
        if np.any(x_t != last_x):
            #here we update the controller gains
            ODE["Array_gains"] = x_t
            ODE.apply_parameters()
            model.integrate_along_trajectory(trajectory, N=N)
        x = x_t
        states = ODE.last_result
        time_vector = ODE.time_points
        cond1 = nonlcon(grad, x, time_vector, states)
        result[:] = cond1
        return result[:]

    def _cost(x_t, grad):
        if grad is not None and grad.size > 0:
            raise NotImplementedError("Will see later for gradient based optimisation")
        global last_x
        last_x = x_t
        tmp = x_t
        ODE["Array_gains"] = tmp
        ODE.apply_parameters()
        k.append(tmp.copy())
        PI = cost(trajectory, ODE, num=N, PI_mask=PI_mask)
        cost_list.append(PI)

        return PI

    opt.add_inequality_mconstraint(nlc, model.nonlcon_tol())
    if model.eq_constraints_tol(target_point) is not None:
        def eq_constraints(result, x, grad):
            states = ODE.last_result
            times_vec = ODE.time_points
            result[:] = model.eq_constraints(grad, states, times_vec, target_point)
            return result
        opt.add_equality_mconstraint(eq_constraints, model.eq_constraints_tol(target_point))

    #Set relative tolerance on function value.
    opt.set_ftol_rel(1e-6)
    #Set absolute tolerance on function value.
    opt.set_xtol_abs(1e-6)
    x_ini_t = x_ini_gains
    opt.set_maxeval(10000)
    opt.set_maxtime(60*optimization_time)
    dx2 = delta * x_ini_gains
    opt.set_initial_step(dx2)


    # optimize PI
    cost_eval_count = 0
    t0 = time()
    opt.set_min_objective(_cost)

    PI_init_cost = _cost(x_ini_t, None)
    PI_x_opt = opt.optimize(x_ini_t)
    PI_opt_cost = opt.last_optimum_value()
    T_PI = time() - t0


    cost_list = np.array(cost_list)
    param_list = np.array(k)
    # take the minimum index of the opt
    index = np.argmin(cost_list)

    print(
        f"PI improvement: {PI_init_cost:.2e} --> {cost_list[index]:.2e} in {T_PI} with exit code "
        f"{opt.last_optimize_result()} in {opt.get_numevals()} ({cost_eval_count}) evals"
    )


    PI_gains_opt = param_list[index, :]
    print(f'here we are printing the new gains from the optimisation: {PI_gains_opt} ')

    PI_out = trajectory.deepcopy()
    PI_out.save_controller_gains_along_trajectory(param_list)
    PI_out.save_cost_along_trajectory(cost_list)
    PI_out.set_opt_time(optimization_time)
    # We save the gains here after optimisation
    PI_out.update_controller_gains(PI_gains_opt)

    # trajectory + cost optimized + all cost list + all parameters
    return PI_out, cost_list[index], cost_list













