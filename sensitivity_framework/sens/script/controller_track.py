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
save_path = '../Paper_Journal_ICRA/save/controller_tracking/controllers_Check_tracking' + now.strftime('%Y-%m-%d')
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



filename = "../Paper_Journal_ICRA/save/trajectories/pd_1_2023-05-30.straj"

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



# Choose a trajectory
traj = INIT[0]
# obtain reference trajectory according to "numtimestep" in constants class
traj.Construct_trajectory()

# simulate the trajectory over decided timesteps in the class constant
model.integrate_along_trajectory(traj, c.N)  # Inorder to get the last results
states = model.ODE.last_result
# Attain the states "xyz" or you can have all the states as you like
states = model.ODE.last_result[:, model.output_indices()]

figure_1 = plt.figure(figsize=(16, 9))
axis_1 = figure_1.add_subplot(1, 2, 1, projection="3d")
axis_1.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k--', label='Reference Signal')
axis_1.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', label='PD Controller Behavior')
axis_1.set_title(r'INIT' , fontsize=c.fontsizetitle)
axis_1.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_1.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_1.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_1.legend(loc='upper center', fancybox=True, ncol=2, bbox_to_anchor=(0.5, 0))
axis_1.set_zticks(np.arange(0, 2.501, 0.5))


# Choose a trajectory
traj = PI[0]
# obtain reference trajectory according to "numtimestep" in constants class
traj.Construct_trajectory()

# simulate the trajectory over decided timesteps in the class constant
model.integrate_along_trajectory(traj, c.N)  # Inorder to get the last results
states = model.ODE.last_result
# Attain the states "xyz" or you can have all the states as you like
states = model.ODE.last_result[:, model.output_indices()]


axis_2 = figure_1.add_subplot(1, 2, 2, projection="3d")
axis_2.plot3D(traj.pos_all[:, 0], traj.pos_all[:, 1], traj.pos_all[:, 2], 'k--', label='Reference Signal')
axis_2.plot3D(states[:, 0], states[:, 1], states[:, 2], 'r-', label='PD Controller Behavior')
axis_2.set_title(r'PI', fontsize=c.fontsizetitle)
axis_2.set_xlabel(r'$x\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_2.set_ylabel(r'$y\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_2.set_zlabel(r'$z\ [\text{m}]$', fontsize=c.fontsizelabel)
axis_2.legend(loc='upper center', fancybox=True, ncol=2, bbox_to_anchor=(0.5, 0))
axis_2.set_zticks(np.arange(0, 2.501, 0.5))


plt.savefig(save_path + '/track_check_Pd' + '.pdf')
plt.show()