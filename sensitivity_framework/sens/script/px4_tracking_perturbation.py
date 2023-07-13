import time
from typing import Sequence
import os
import datetime
from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
import pickle
from base64 import b64decode
import numpy as np
import matplotlib.pyplot as plt
from cnst.constant import constants
import math
import utils.Functions as Fct

c = constants()
# latex for figures
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \usepackage{lmodern}'
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)



plt.rcParams.update({'figure.max_open_warning': 0})

now = datetime.datetime.now()
save_path = '../Paper_Journal_ICRA/save/controller_perturbation/controllers_Check_tracking' + now.strftime('%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# Model initialization

t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True) # true if you changed the model and need to compile, otherwise false
ODE.set_integrator("dopri5")
model.set_default_state(c.init_waypoint)
t1 = time.time()
print(f"generating of PD model took {t1-t0} s")



filename = "../Paper_Journal_ICRA/save/trajectories/pd_1_2023-05-31.straj"

INIT: Sequence[PiecewiseSplineTrajectory] = []
PI: Sequence[PiecewiseSplineTrajectory] = []
PI_k: Sequence[PiecewiseSplineTrajectory] = []
PI_ak: Sequence[PiecewiseSplineTrajectory] = []
TARGETS: Sequence[np.ndarray] = []
# cases = [INIT, PI, PI_k, PI_ak, TARGETS]
cases = [INIT, PI, TARGETS]

with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))





figure_1 = plt.figure(figsize=(16, 9))

# Choose a trajectory
traj = INIT[0]
model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
# trajectory construction of the reference with all pos, vel and acceleration
traj.Construct_trajectory()
axis_1 = figure_1.add_subplot(1, 2, 1, projection="3d")
axis_1.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k-', linewidth=3)
axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2.5)
axis_1.plot(*TARGETS[0][0, :3], "*", color="b", linewidth=3.5, markersize=10)
axis_1.set_title(r'INIT', fontsize=c.fontsizetitle)
axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)



# Choose a trajectory
traj = PI[0]
model.integrate_along_trajectory(traj, c.N)  # inorder to get the last results
states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
# trajectory construction of the reference with all pos, vel and acceleration
traj.Construct_trajectory()
axis_2 = figure_1.add_subplot(1, 2, 2, projection="3d")
axis_2.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k-', linewidth=3)
axis_2.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', linewidth=2.5)
axis_2.plot(*TARGETS[0][0, :3], "*", color="b", linewidth=3.5, markersize=10)
axis_2.set_title(r'PI', fontsize=c.fontsizetitle)
axis_2.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_2.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_2.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)


targets_INIT = []
targets_PI = []
N_sim = 20

gains = ODE["Array_gains"]
for j in range(N_sim):

    traj = INIT[0]

    delta_p = np.array([np.random.uniform(-c.dev, c.dev, *np.shape(c.dev))*c.true_p[0],
                        np.random.uniform(-c.dev, c.dev, *np.shape(c.dev))*c.true_p[1],
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(0, c.dev2, *np.shape(c.dev2))*c.true_p[4]])

    r = delta_p.T @ np.linalg.inv(c.W_range) @ delta_p
    delta_p = delta_p/math.sqrt(r)

    ODE["kf"]   = c.true_p[0] + delta_p[0]
    ODE["ktau"] = c.true_p[1] + delta_p[1]
    ODE["gx"]   = c.true_p[2] + delta_p[2]
    ODE["gy"]   = c.true_p[3] + delta_p[3]
    ODE["m"]    = c.true_p[4] + delta_p[4]
    # ODE["Array_gains"] = gains


    # gathering the states when perturbating the system for plotting later on
    ODE.apply_parameters()
    model.integrate_along_trajectory(traj, c.N)
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    targets_INIT.append(states[-1, :3])
    axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'g--', linewidth=1.5, alpha=0.4)


    traj = PI[0]
    ODE["kf"] = c.true_p[0] + delta_p[0]
    ODE["ktau"] = c.true_p[1] + delta_p[1]
    ODE["gx"] = c.true_p[2] + delta_p[2]
    ODE["gy"] = c.true_p[3] + delta_p[3]
    ODE["m"] = c.true_p[4] + delta_p[4]

    # ODE["Array_gains"] = np.random.uniform(0.5, 1, *np.shape(c.off)) * ODE["Array_gains"]

    # gathering the states when perturbating the system for plotting later on
    ODE.apply_parameters()
    model.integrate_along_trajectory(traj, c.N)
    states = model.ODE.last_result[:, model.ODE.states_indices["q"]]
    targets_PI.append(states[-1, :3])
    axis_2.plot3D(states[:, 0], states[:, 1], states[:, 2], 'g--', linewidth=1.5, alpha=0.4)


targets_INIT = np.array(targets_INIT)
targets_PI = np.array(targets_PI)

# Fct.draw_ellipse(TARGETS[0, :3], targets_INIT, axis_1, 0.04)
# Fct.draw_ellipse(TARGETS[0, :3], targets_PI, axis_2, 0.04)

plt.savefig(save_path + '/tracking_all__'+ '_Nsim_' + str(N_sim) + '.pdf')

plt.show()