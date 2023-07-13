import time
import datetime

from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
from scipy.linalg import norm
from opt.optimize import sensitivity_gain_optimisation, sensitivity_optimisation, sensitivity_joint_optimisation
# from opt.optimize import sensitivity_gain_optimisation, sensitivity_optimisation, sensitivity_joint_optimisation
import matplotlib.pyplot as plt
import pickle
from base64 import b64encode
import numpy as np
import random
import math
from cnst.constant import constants
import utils.Functions as Fct

c = constants() # class where we define all the constants to be used in all the framework
# latex for figures
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \usepackage{lmodern}'
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)
plt.rcParams.update({'figure.max_open_warning': 0})



now = datetime.datetime.now()



#######################################################################################3
# saving trajectories
traj_filename = "../Paper_Journal_ICRA/save/trajectories/pd_" + str(c.N_traj) +"_"+ now.strftime('%Y-%m-%d') +'.straj'
t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True)
model.set_default_state(c.init_waypoint)
model.ODE.set_integrator("dopri5")
t1 = time.time()
print(f" Loading the model took {t1 - t0} s")
########################################
PI_mask = np.full((jetson_pd.N_states, jetson_pd.N_par), False)
PI_mask[:3, :] = True  # only improve the xyz position for all parameters

nlc = lambda grad, x, time_vec, states: model.nonlcon(grad, states, time_vec, c.umin, c.umax, 0.01)

trajectories = np.empty(c.N_traj, np.object_)
INIT_JustWaypoints = np.empty(c.N_traj, np.object_)
INIT_InputSat_out = np.empty(c.N_traj, np.object_)
INIT_Target_out = np.empty(c.N_traj, np.object_)

INIT = np.empty(c.N_traj, np.object_)
PI = np.empty(c.N_traj, np.object_)
PI_g = np.empty(c.N_traj, np.object_)
PI_c_g = np.empty(c.N_traj, np.object_)


with open(traj_filename, "wb") as dump:
    pass

###########################################
# Always initial point is defined in constant class
point1 = c.init_waypoint

# list of target points
target_points = []

for i in range(c.N_traj):
    position = Fct.choose_position(c.init_waypoint, c.ub[0, :3], c.lb[0, :3])
    pos2 = np.hstack((position, np.pi * (np.random.rand() - 0.5)))
    point2 = np.vstack([
        pos2,
        [0, 0, 0, 0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    target_points.append(point2)

# here we will have set of target point desired such that they are selected according to crriteria of Fct.choose_position
target_points = np.array(target_points)


#opt step for PI and time
##########################################################################################
t_start = time.time()
i = 0

gains = ODE["Array_gains"]

while i < c.N_traj: # loop for the trajectories

    # ODE["Array_gains"] = gains
    # ODE.apply_parameters()
    t1 = time.time()

    check = 1
    while check != 0:
        trajectories[i] = PiecewiseSplineTrajectory(np.array([0, c.Tf]), [point1, target_points[i]])
        check = Fct.interpolate_and_check_constraints(trajectories[i], c)

    # # additional waypoints interpolation so that the optimizer gets free variables to play with
    # for k in range(c.N_waypoints - 2):  # -2 because 2 waypoints were already defined just above
    #     trajectories[i].interpolate_waypoint_at((k+1)*c.Tf/(c.N_pieces) , free=True)

    # masking stuff, change accordingly the desired behaviour on the waypoints behaviour during opt
    trajectories[i].free_waypoint(-1)
    trajectories[i]._waypoints_t_mask[:] = False  # check later if they can be modified (for now pre-opt doesn't let it work)
    trajectories[i]._waypoints_mask[:, 0, 3] = False  # fixed yaw (but not speed and acc)

    INIT_JustWaypoints[i] = trajectories[i].deepcopy()  # save initial trajectory with just waypoints before conditionning


    # Preconditioning for opt
    t0 = time.time()
    cost_init = trajectories[i].optimize(model, nlc, target_points[i], use_target=True, lower_bounds=c.lb,
                                         upper_bounds=c.ub, optim_time=c.init_opt_time, N=c.N, dx_init=c.dx_init)
    INIT[i] = trajectories[i].deepcopy()  # save


    PI[i], PI_opt_cost, cost_time = sensitivity_optimisation(model, INIT[i], nonlcon=nlc,
                                                      target_point=target_points[i], lower_bounds=c.lb, upper_bounds=c.ub,
                                                      PI_mask=PI_mask, optimization_time=c.optimization_time_PI, delta=c.dx_PI)


    # PI_g[i], PI_opt_cost_g, cost_time_g= sensitivity_gain_optimisation(model, INIT[i], nonlcon=nlc,
    #                                                   target_point=target_points[i], PI_mask=PI_mask, gains=gains,
    #                                                   optimization_time=c.optimization_time_PI, delta =c.dx_PI)



    # PI_c_g[i], PI_opt_cost_c_g, cost_time_c_g = sensitivity_joint_optimisation(model, INIT[i], nonlcon=nlc,
    #                                                   target_point=target_points[i], lower_bounds=c.lb, upper_bounds=c.ub,
    #                                                   PI_mask=PI_mask, gains=gains, optimization_time=c.optimization_time_PI, delta =c.dx_PI)



    with open(traj_filename, "ab") as dump:
        # dump.write(b"\n".join(map(lambda x: b64encode(pickle.dumps(x)), [INIT[i], PI[i], PI_c_g[i], PI_g[i]])))
        dump.write(b"\n".join(map(lambda x: b64encode(pickle.dumps(x)), [INIT[i], PI[i]])))
        dump.write(b"\n" + b64encode(pickle.dumps(target_points[i])))
        dump.write(b"\n")
    plt.pause(1e-6)
    i = i + 1


t_end = time.time()

print(f'########################################')
print(f"GENERATED {c.N_traj} TRAJECTORIES IN {t_end - t_start} s")
print(f'########################################')


