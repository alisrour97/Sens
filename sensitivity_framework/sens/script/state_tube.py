import math
import time
from typing import Sequence
import datetime
import os
from numpy.linalg import norm
from gen.models import jetson_pd as jetson_pd
from utils.trajectory import PiecewiseSplineTrajectory
import pickle
from base64 import b64decode
import numpy as np
import matplotlib.pyplot as plt
from cnst.constant import constants
import utils.Functions as Fct

# latex for figures
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \usepackage{lmodern}'
params = {'text.usetex': True,
          'font.size': 16,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)

"""class of constants"""
c = constants()

""" Load The Model """

t0 = time.time()
model = jetson_pd.Jetson(c.N_coeffs)
ODE = model.generate(3, verbose=True, mode=jetson_pd.Mode.NOGRAD, overwrite=True)
ODE.set_integrator("dopri5", nsteps=10**6)
model.set_default_state(c.init_waypoint)
t1 = time.time()
print(f"generating took {t1-t0} s")


"""Path to save figures"""

now = datetime.datetime.now()
save_path ='../Paper_Journal_ICRA/save/tube_states/tube_check_'+ now.strftime(
    '%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)


"""Data Reading trajectory"""
filename = "../Paper_Journal_ICRA/save/trajectories/pd_1_2023-05-31.straj"


INIT: Sequence[PiecewiseSplineTrajectory] = []
PI: Sequence[PiecewiseSplineTrajectory] = []
PI_k: Sequence[PiecewiseSplineTrajectory] = []
PI_ak: Sequence[PiecewiseSplineTrajectory] = []
TARGETS: Sequence[np.ndarray] = []
# cases = [INIT, PI_a, PI_k, PI_ak, TARGETS]
cases = [INIT, PI, TARGETS]


with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))



"""Choose a trajectory"""

traj = PI[0]

"""Integrate along the Trajectory"""
model.integrate_along_trajectory(traj, c.N)
states = ODE.last_result
states_nominal = model.ODE.last_result[:, model.ODE.states_indices["q"]]
t = ODE.time_points

"""Tube Calculations"""
states_max, states_min = Fct.tube_states(states, model)


"""Perturbing the System"""
Nsim = 20 #In pertubation
list_states = []

for i in range (Nsim):
    delta_p = np.array([np.random.uniform(-c.dev, c.dev, *np.shape(c.dev))*c.true_p[0],
                        np.random.uniform(-c.dev, c.dev, *np.shape(c.dev))*c.true_p[1],
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(-c.off, c.off, *np.shape(c.off)),
                        np.random.uniform(-c.dev2, c.dev2, *np.shape(c.dev2))*c.true_p[4]])

    r = delta_p.T @ np.linalg.inv(c.W_range) @ delta_p
    delta_p = delta_p/math.sqrt(r)

    ODE["kf"]   = c.true_p[0] + delta_p[0]
    ODE["ktau"] = c.true_p[1] + delta_p[1]
    ODE["gx"]   = c.true_p[2] + delta_p[2]
    ODE["gy"]   = c.true_p[3] + delta_p[3]
    ODE["m"]    = c.true_p[4] + delta_p[4]

    # gathering the states when perturbating the system for plotting later on
    ODE.apply_parameters()

    model.integrate_along_trajectory(traj, c.N)
    tmp = model.ODE.last_result[:, ODE.states_indices["q"]]
    list_states.append(tmp)


"""Plotting the tubes"""
q = ["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]

for j, param in enumerate(q):

    figure_0 = plt.figure(figsize=(16, 9))
    ax = figure_0.add_subplot(1, 1, 1)
    plt.plot(t, states_nominal[:, j])
    plt.plot(t, states_max[:, j], 'r-')
    plt.plot(t, states_min[:, j], 'r-')
    for i in range(len(list_states)):
        plt.plot(t, list_states[i][:, j], 'k--')
    ax.set_xlabel(r'$time (sec)\ [\text{m}]$', fontsize=c.fontsizelabel)
    ax.set_ylabel("The state " + param, fontsize=c.fontsizetitle)

    ax.set_title("Tube of the state " + param + " with perturbating the parameters kf, ktau, gx, gy and m ")

    plt.savefig(save_path + '/state_tube_check' + "-" + " All Params " + " _ " + " of the state " + " _ " + param + '.pdf')


print("Finished")

