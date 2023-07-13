import numpy as np
import math
import random
import symengine as se
from cnst.constant import constants

c = constants()
def choose_position(waypoint, ub, lb):
    x = random.uniform(lb[0], ub[0])
    y = random.uniform(lb[1], ub[1])
    z = random.uniform(2*lb[2], ub[2])

    distance = math.sqrt((x-waypoint[0, 0])**2 + (y-waypoint[0, 1])**2)

    if distance > 5:
        return (x, y, z)
    else:
        return choose_position(waypoint, ub, lb)

def choose_velocity(ub, lb):
    vx = random.uniform(lb[0], ub[0])
    vy = random.uniform(lb[1], ub[1])
    vz = random.uniform(lb[2], ub[2])

    velocity = math.sqrt(vx**2 + vy**2 + vz**2)

    if velocity > 2 and velocity < 2.5:
        return (vx, vy, vz)
    else:
        return choose_velocity(ub, lb)


def interpolate_and_check_constraints(traj, c):
    for k in range(c.N_waypoints - 2):
        traj.interpolate_waypoint_at((k + 1) * c.Tf / (c.N_pieces), free=True)

    for i in range(1, c.N_waypoints-1):
        # Check if waypoint is within limits
        if (traj.waypoints[i] >= c.lb).all() and (traj.waypoints[i] <= c.ub).all():
            print("Waypoint is within limits")
            return 0

        else:
            # Find where the violation occurs
            violation_indices = np.where((traj.waypoints[i] < c.lb) | (traj.waypoints[i] > c.ub))
            print("Waypoint violates limits at indices:", violation_indices)
            return 1


def Rx(th):
    # RX Summary of this function goes here
    # Detailed explanation goes here
    R = np.array([[1, 0, 0], [0, se.cos(th), -se.sin(th)], [0, se.sin(th), se.cos(th)]])
    return R

def Ry(th):
    # RY Summary of this function goes here
    # Detailed explanation goes here
    R = np.array([[se.cos(th), 0, se.sin(th)], [0, 1, 0], [-se.sin(th), 0, se.cos(th)]])
    return R


def Rz(th):
    # RZ Summary of this function goes here
    # Detailed explanation goes here
    R = np.array([[se.cos(th), -se.sin(th), 0], [se.sin(th), se.cos(th), 0], [0, 0, 1]])
    return R

def quat2rotm(q):
    eta = q[0]
    vec = q[1:4]

    R = se.DenseMatrix([[2*(eta**2 + vec[0]**2)-1, 2*(vec[0]*vec[1] - eta*vec[2]), 2*(vec[0]*vec[2] + eta*vec[1])],
                  [2*(vec[0]*vec[1] + eta*vec[2]), 2*(eta**2 + vec[1]**2)-1, 2*(vec[1]*vec[2] - eta*vec[0])],
                  [2*(vec[0]*vec[2] - eta*vec[1]), 2*(vec[1]*vec[2] + eta*vec[0]), 2*(eta**2 + vec[2]**2)-1]])
    return R

def rotm2eul(R):
    yaw = se.atan2(R[1, 0], R[0, 0])
    pitch = se.atan2(-R[2, 0], se.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = se.atan2(R[2, 1], R[2, 2])

    RPY = np.array([roll, pitch, yaw])
    return RPY



def skewMat(v):
    v_skwmat = se.DenseMatrix([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    return v_skwmat



def quat_norm2(q):
    return (q.T*q)[0]

def quat_to_mat(q):
    # a, b, c, d = q[:]
    # return se.DenseMatrix([
    #     [a**2+b**2-c**2-d**2, 2*b*c - 2*a*d, 2*a*c + 2*b*d],
    #     [2*a*d + 2*b*c, a**2-b**2+c**2-d**2, 2*c*d - 2*a*b],
    #     [2*b*d - 2*a*c, 2*a*b + 2*c*d, a**2-b**2-c**2+d**2],
    # ]) / quat_norm2(q)

    a, b, c, d = q[:]
    return se.DenseMatrix([
        [a**2+b**2-c**2-d**2, 2*b*c - 2*a*d, 2*a*c + 2*b*d],
        [2*a*d + 2*b*c, a**2-b**2+c**2-d**2, 2*c*d - 2*a*b],
        [2*b*d - 2*a*c, 2*a*b + 2*c*d, a**2-b**2-c**2+d**2],
    ])


def rotm2quat(R):
    trace_R = R[0, 0] + R[1, 1] + R[2, 2]
    eta = 0.5 * se.sqrt(trace_R + 1)

    q = se.DenseMatrix([
        eta,
        (R[2, 1] - R[1, 2]) / (4 * eta),
        (R[0, 2] - R[2, 0]) / (4 * eta),
        (R[1, 0] - R[0, 1]) / (4 * eta)
    ])

    return q




# def calculate_error_quaternion(qd, q):
#     qd_conj = [qd[0], -qd[1], -qd[2], -qd[3]]
#     q_error = [qd_conj[i] * q[i] for i in range(4)]
#     q_norm = se.sqrt(sum(q_error[i] ** 2 for i in range(4)))
#     q_error = [q_error[i] / q_norm for i in range(4)]
#     return q_error


# def calculate_error_quaternion(qd, q):
#     qd_vec = se.DenseMatrix([qd[1:]])
#     q_vec = se.DenseMatrix([q[1:]])
#
#     q_err = [
#         q[0] * qd[0] + qd_vec.dot(q_vec.transpose()),
#         q[0] * qd_vec.transpose() - qd[0] * q_vec.transpose() - skewMat(qd_vec).dot(q_vec.transpose())
#     ]
#     return q_err


# def rotm2quat(R):
#
#     trace_R = R[0, 0] + R[1, 1] + R[2, 2]
#     eta = 0.5 * se.sqrt(trace_R + 1)
#     q = 0.5* se.DenseMatrix([
#         2* eta,
#         se.tanh(10000*(R[2, 1] - R[1, 2])) * se.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1),
#         se.tanh(10000*(R[0, 2] - R[2, 0])) * se.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1),
#         se.tanh(10000*(R[1, 0] - R[0, 1])) * se.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)
#     ])
#
#     return q


def mat_to_quat(mat):
    assert mat.shape == (3, 3)
    q = se.zeros(4, 1)
    q[0] = se.sqrt(1 + mat[0, 0] + mat[1, 1] + mat[2, 2])/2
    q[1] = (mat[2, 1] - mat[1, 2])/(4*q[0])
    q[2] = (mat[0, 2] - mat[2, 0])/(4*q[0])
    q[3] = (mat[1, 0] - mat[0, 1])/(4*q[0])
    return q

def hat_map(vec):
    x, y, z = vec
    return se.DenseMatrix([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def draw_ellipse(target, targets, axis, ep):
    # your ellispsoid and center in matrix form
    A = np.array([[target[0], 0, 0],
                  [0, target[1], 0],
                  [0, 0, target[2]]])
    center = [target[0], target[1], target[2]]

    # find the rotation matrix and radii of the axes
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0 / np.sqrt(s) * ep # reduce radii by factor 0.3

    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 60)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    # axis.plot_surface(x, y, z, color='r', linewidth=0.2, alpha=0.25, shade=True)
    axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], color='red', alpha=0.35, label='Final position $r(t_f)$')


def tube_states(states_vec, model):
    N_states = c.N_states
    N_par = np.shape(c.W_range)[0]
    PI = np.reshape(states_vec[:, model.ODE.states_indices["PI"]], (-1, N_states, N_par))
    N_time, N_dim, N_par = np.shape(PI)
    ei = np.eye(N_dim)  # base vector in data space
    data_plus_r = np.zeros(np.shape(states_vec))
    data_min_r = np.zeros(np.shape(states_vec))
    data_plus_r[0, :], data_min_r[0, :] = states_vec[0, :], states_vec[0, :]

    for i in range(N_time - 1):  # each time
        sens = PI[i + 1, :, :]  # get the sensitivity matrix for this specific time
        mat = sens @ c.W_range @ sens.T  # kernel matrix of the sensitivity

        for k in range(N_dim):
            ri = np.sqrt(ei[:, k].T @ mat @ ei[:, k])  # eq (9) of Thommaso notes
            data_plus_r[i + 1, k] = states_vec[i + 1, k] + ri  # upper tube
            data_min_r [i + 1, k] = states_vec[i + 1, k] - ri  # lower tube

    return data_plus_r, data_min_r


def tube_inputs(states_vec, model):
    time_vec = model.ODE.time_points
    TH_i = np.reshape(states_vec[:, model.ODE.states_indices["TH_i"]], (-1, c.N_inputs, c.N_par))  # get integral of TH
    u_int = states_vec[:, model.ODE.states_indices["u_int"]]  # get u_int
    u = (np.diff(u_int, axis=0).T / np.diff(time_vec, axis=0).T).T
    N_time, N_dim, _ = np.shape(TH_i)  # get sizes of interest from integral of Theta (input sensitivity)
    ei = np.eye(N_dim)  # base vectors in data space
    u_plus_r = np.zeros(np.shape(u))  # empty list for upper tube
    u_min_r = np.zeros(np.shape(u))  # empty list for lower tube
    u_plus_r[0, :], u_min_r[0, :] = u[0, :], u[0, :]  # init first values of those arrays
    for i in range(N_time - 2):  # each time but first and last (because of differentiation)
        TH = np.subtract(TH_i[i + 1, :, :], TH_i[i, :, :]) / (
                time_vec[i + 1] - time_vec[i])  # compute Theta for one step
        mat = TH @ c.W_range @ TH.T  # kernel matrix of the input sensitivity
        for k in range(N_dim):  # each state
            ri = np.sqrt(ei[:, k].T @ mat @ ei[:, k])  # eq (9) of Thommaso notes
            u_plus_r[i + 1, k] = u[i + 1, k] + ri  # upper tube
            u_min_r[i + 1, k] = u[i + 1, k] - ri  # lower tube

    return u_plus_r,  u_min_r

