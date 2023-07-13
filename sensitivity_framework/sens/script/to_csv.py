from typing import Sequence
import os
import datetime
from utils.trajectory import PiecewiseSplineTrajectory
import pickle
from base64 import b64decode
import numpy as np
from cnst.constant import constants
import csv

c = constants()
now = datetime.datetime.now()
"""Save path and if it doesn't exist , create it"""

save_path = 'sens/Paper_Journal_ICRA/save/traj_to_csv/csv_file_' + now.strftime('%Y-%m-%d')
if not os.path.isdir(save_path):
    os.mkdir(save_path)


"""Traj file location to convert to CSV"""

filename = "../Paper_Journal_ICRA/save/trajectories/pd_1_2023-05-30.straj"

'''All possible traj types found inside the trajectory with the directory filename'''

INIT: Sequence[PiecewiseSplineTrajectory] = [] # initial
PI: Sequence[PiecewiseSplineTrajectory] = [] # optimized for a
PI_k: Sequence[PiecewiseSplineTrajectory] = [] # optimized for k
PI_ak: Sequence[PiecewiseSplineTrajectory] = [] # optimized fo {ak}
TARGETS: Sequence[np.ndarray] = [] # Targets
# cases = [INIT, PI, PI_k, PI_ak, TARGETS] # all cases in a list
cases = [INIT, PI,TARGETS] # all cases in a list

# Read all of this from the filename
with open(filename, "rb") as dump:
    for i, line in enumerate(dump):
        cases[i % len(cases)].append(pickle.loads(b64decode(line)))


# Choose a trajectory
traj = PI[0]

# Obtain reference trajectory according to "numtimestep" in constants class
traj.Construct_trajectory()
t = np.linspace(0, c.Tf, len(traj.pos_all))


# Combine the time, position, velocity, and acceleration arrays into a single NumPy array
data = np.column_stack((t, traj.pos_all, traj.vel_all, traj.acc_all))

# Open a new file for writing
with open(save_path + '/drone_data.csv', 'w', newline='') as file:

    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['time', 'pos_x', 'pos_y', 'pos_z', 'yaw', 'vel_x', 'vel_y', 'vel_z', 'vel_yaw', 'acc_x', 'acc_y', 'acc_z', 'acc_yaw'])

    # Write the data to the CSV file
    writer.writerows(data)


print("Finished")



