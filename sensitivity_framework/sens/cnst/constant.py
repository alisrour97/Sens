import numpy as np

class constants():

    def __init__(self):

        """Problem constants and dimensions"""
        self.const = {
            'K_NORM_QUAT': 0.1,  # constante de normalisation des quaternions
            'N_states': 13,  # q = [x y z vx vy vz Q(4*1) Omega(3*1)]
            'N_inputs': 4,  # u = [f(4-1)]
            'N_outputs': 4,  # y = [x y z yaw]
            'N_ctrl_states': 12,  # xi
            'N_par': 5,  # p = [kf, ktau, gx, gy, m]
            'N_par_aux': 5,  # p_aux = [Jx, Jy, Jz, l, g]
        }
        self.N_par = self.const['N_par']
        self.N_states = self.const['N_states']
        self.N_inputs = self.const['N_inputs']

        """About nominal gains and actuator"""
        # max-min motor speeds
        self.umax = 1100 ** 2
        self.umin = 250 ** 2

        """Related to integration & Trajectory"""
        self.Ts = 0.02 # sampling time
        self.Ti = 0 # we start at t0
        self.Tf = 4 # Final time
        # number of time steps to integrate the system along a trajectory
        self.N = int(self.Tf/self.Ts) # length of time vector
        self.numTimestep = int(self.N+1)

        """Number of trajectories to be optimized"""
        self.N_traj = 1
        self.N_waypoints = 4
        self.N_pieces = self.N_waypoints - 1

        """Ellipsoid parameters to be decided on """
        self.true_p = np.array([5.84*1e-6, 0.06, 0, 0, 1.535])  # [kf, ktau, gx, gy], make sure they correspond to the ones in the model
        self.dev = 0.15  # 15 percentage
        self.dev2 = 0.3 # 30 percentage for mass deviation
        self.off = 0.05  # 5 cm deviation
        self.W_range = np.diag([(self.dev * self.true_p[0]) ** 2, (self.dev * self.true_p[1]) ** 2, self.off ** 2, self.off ** 2, (self.dev2 * self.true_p[4]) ** 2])

        """OPT_{a, k, ak} trajectories cnst related to opt"""
        self.optimization_time_PI = 10
        self.dx_PI = 0.12
        self.init_opt_time = 0.5
        self.dx_init = 0.15

        """upper bounds and lower bounds on each waypoint"""
        self.ub = np.array([
            [3, 3, 2, np.pi/4],
            [2.5, 2.5, 2.5, 0.5],
            [2, 2, 2, 0.5],
        ])

        self.lb = -np.array([
            [3, 3, -1, np.pi/4],
            [2.5, 2.5, 2.5, 0.5],
            [2, 2, 2, 0.5],
        ])

        """Setting initial params for waypoint initial"""""
        self.init_waypoint = np.array([
            [-2, -2, 1, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        self.N_jc, self.N_dim = self.init_waypoint.shape
        self.N_coeffs = self.N_jc * 2

        """For Latex"""
        # font sizes for figures
        self.fontsizetitle = 36
        self.fontsizelabel = 25
        self.fontsizetick = 12
        self.fontsizelegend = 18
