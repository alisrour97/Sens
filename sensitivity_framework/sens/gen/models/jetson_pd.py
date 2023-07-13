import math

from gen.lib.sym_gen import JitParam, ODEproblem, t
from gen.lib.base_model import Model, Mode
from math import comb
import symengine as se
import numpy as np
from sympy import symbols, Matrix
from sympy.physics.mechanics import *
from utils.trajectory import min_curvature_radius
import time
import utils.Functions as Fct
from cnst.constant import constants



c = constants()

K_NORM_QUAT = c.const['K_NORM_QUAT']      # constante de normalisation des quaternions
N_states = c.const['N_states']           # q = [x y z vx vy vz Q(4*1) Omega(3*1)]
N_inputs = c.const['N_inputs']           # u = [f(4-1)]
N_outputs = c.const['N_outputs']          # y = [x y z yaw]
N_ctrl_states = c.const['N_ctrl_states']       # xi = [e_xi(3)]
N_par = c.const['N_par']               # p = [kf, ktau, gx, gy]
N_par_aux = c.const['N_par_aux']          # p_aux = [m, Jx, Jy, Jz, l, g]

class Jetson(Model):

    def nonlcon(self, grad, states_vec, time_vec, umin, umax, r_min):
        N_par = np.shape(c.W_range)[0]
        TH_i = np.reshape(states_vec[:, self.ODE.states_indices["TH_i"]], (-1, N_inputs, N_par))  # get integral of TH
        u_int = states_vec[:, self.ODE.states_indices["u_int"]]  # get u_int
        u = (np.diff(u_int, axis=0).T / np.diff(time_vec, axis=0).T).T
        N_time, N_dim, _ = np.shape(TH_i)  # get sizes of interest from integral of Theta (input sensitivity)
        ei = np.eye(N_dim)  # base vectors in data space
        u_plus_r = np.zeros(np.shape(u))  # empty list for upper tube
        u_min_r = np.zeros(np.shape(u))  # empty list for lower tube
        u_plus_r[0, :], u_min_r[0, :] = u[0, :], u[0, :]  # init first values of those arrays
        # NEW TUBES:
        for i in range(N_time - 2):  # each time but first and last (because of differentiation)
            TH = np.subtract(TH_i[i + 1, :, :], TH_i[i, :, :]) / (
                        time_vec[i + 1] - time_vec[i])  # compute Theta for one step
            mat = TH @ c.W_range @ TH.T  # kernel matrix of the input sensitivity
            for k in range(N_dim):  # each state
                ri = np.sqrt(ei[:, k].T @ mat @ ei[:, k])  # eq (9) of Thommaso notes
                u_plus_r[i + 1, k] = u[i + 1, k] + ri  # upper tube
                u_min_r[i + 1, k] = u[i + 1, k] - ri  # lower tube

        n_max = np.max(u_plus_r, axis=0)
        n_min = np.min(u_min_r, axis=0)
        c_rmin = r_min - min_curvature_radius(states_vec[:, self.output_indices()], time_vec)
        c0 = np.block([n_max - umax, umin - n_min, c_rmin])
        return c0

    def nonlcon_tol(self):
        return [1e-6]*N_inputs + [1e-6]*N_inputs + [1e-6]

    def eq_constraints(self, grad, states_vec, time_vec, target_point):
        N_jc, _ = target_point.shape
        traj = states_vec[-(N_jc):, self.output_indices()]
        dt_vec = np.diff(time_vec[-N_jc:])
        tmp = np.zeros((N_jc, 3))
        tmp[0, :] = traj[-1, :]
        for i in range(1, N_jc):
            traj = np.gradient(traj, np.mean(dt_vec), axis=0)
            tmp[i, :] = traj[-1, :]
        return tmp.flatten() - target_point[:, :3].flatten()

    def eq_constraints_tol(self, target_point=None):
        return [1e-6]*(target_point.shape[0]*3)

    def output_indices(self):
        return self.ODE.states_indices["q"][:3]

    def generate(self, N_lc, verbose=False, mode: Mode=Mode.NOGRAD, token="", overwrite=False) -> (JitParam):
        if not isinstance(mode, Mode):
            raise ValueError(f"mode must either be a member of the Mode enum")

        self.control_mask = np.concatenate([np.zeros(N_lc), np.ones(self.N_ctrl_points - 2 * N_lc), np.zeros(N_lc)] * N_outputs).astype(bool)

        module_name = "".join([f"jitced_jetson_pd_n{self.N_ctrl_points}_{mode.value}", f"_{token}" if token else "", ".so"])

        helpers = dict()
        system = dict()
        subs = dict()
        self._problem = pb = ODEproblem()

        # ========== Defining the system with Symengine ===========================================

        # State variables
        # ---------------
        x, y, z, vx, vy, vz = dynamicsymbols('x y z vx vy vz')
        qw, qx, qy, qz, wx, wy, wz = dynamicsymbols('qw qx qy qz wx wy wz')
        q = pb.add_states("q", N_states)
        subs["q"] = [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz], q

        # Parameters, forces and torques
        # ------------------------------
        m, Jx, Jy, Jz, gx, gy, l, kf, ktau, g = symbols('m Jx Jy Jz gx gy l kf ktau g')

        p = pb.new_parameter("p", N_par)  # system parameters used for sensitivity
        subs["p"] = [kf, ktau, gx, gy, m], p
        p_c = pb.new_parameter("p_c", N_par)  # system parameters implanted in the controller
        subs["p_c"] = [kf, ktau, gx, gy, m], p_c
        p_aux = pb.new_parameter("p_aux", N_par_aux)  # auxiliary parameters
        subs["p_aux"] = [Jx, Jy, Jz, l, g], p_aux

        # Inputs
        # ------
        u = pb.new_sym_matrix("u", N_inputs)  # u1 = w1**2

        # Allocation matrix
        # -----------------
        S = kf * se.DenseMatrix([
            [     1,      1,      1,         1],
            [   -gy, l - gy,    -gy, -(l + gy)],
            [gx - l,     gx, l + gx,        gx],
            [  ktau,  -ktau,   ktau,     -ktau]])


        S_C = S.subs(*subs["p_aux"]).subs(*subs["p_c"])

        S = S.subs(*subs["p_aux"]).subs(*subs["p"])


        '''Reference Trajectory'''
        N_ctrl_points = self.N_ctrl_points
        bp = pb.new_sym_matrix("bp", N_ctrl_points)
        a = pb.new_parameter("a", N_outputs, N_ctrl_points)
        ti = pb.new_parameter("ti", 1)
        tf = pb.new_parameter("tf", 1)

        for i in range(N_ctrl_points):
            bp[i] = comb(N_ctrl_points - 1, i) * ((t - ti) ** i) * ((tf - t) ** (N_ctrl_points - 1 - i)) / (
                        tf - ti) ** (N_ctrl_points - 1)

        # introducing helpers
        xd = pb.new_sym_matrix("xd", 3)
        vd = pb.new_sym_matrix("vd", 3)
        ad = pb.new_sym_matrix("ad", 3)
        yawd = pb.new_sym_matrix("yaw", 1)

        helpers.update({
            xd: a[:3, :] * bp,
            vd: se.diff(a[:3, :] * bp, t),
            ad: se.diff(se.diff(a[:3, :] * bp, t), t),
            yawd: (a[3, :] * bp)[0],
        })

        Array_gains = pb.new_parameter("Array_gains", 24, real=True)
        Pv = se.diag(*Array_gains[:3])
        Pr = se.diag(*Array_gains[3:6])
        Dv = se.diag(*Array_gains[6:9])
        Iv = se.diag(*Array_gains[9:12])
        Kq = se.diag(*Array_gains[12:15])
        P = se.diag(*Array_gains[15:18])
        D = se.diag(*Array_gains[18:21])
        I = se.diag(*Array_gains[21:24])


        xi = pb.add_states("xi", N_ctrl_states)
        xi_v_pos = se.DenseMatrix(xi[:3])
        xi_a_pos = se.DenseMatrix(xi[3:6])
        xi_om_att = se.DenseMatrix(xi[6:9])
        xi_a_att = se.DenseMatrix(xi[9:12])


        ################################## PX4 Controller ####################################################################
        norm = lambda vec: se.sqrt(se.Add(*[vi ** 2 for vi in vec]))

        # == Frame Transformation
        NED2FLU = se.DenseMatrix([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]])

        ''' Aliases'''
        # States
        x = NED2FLU * q[:3, :]
        v = NED2FLU * q[3:6, :]
        Q = se.DenseMatrix([q[6, :], NED2FLU * q[7:10, :]])
        Omega = NED2FLU * q[10:, :]

        # Reference
        xd = NED2FLU * xd
        vd = NED2FLU * vd
        ad = NED2FLU * ad
        #TODO: Check yaw singularity problem
        yawd = -yawd

        Jx, Jy, Jz, l, g = p_aux

        '''Constants'''
        delta = 0.02
        delta_PX4_pos = 0.02
        delta_PX4_att = 0.001
        ht = 0.5  # hovering thrust (normalized)
        omega_c = 2 * math.pi * 5
        omega_c_att = 2 * math.pi * 30 # cut-off frequency
        n_max = (1100)**2
        l_arm = 0.28

        # torque_max_x = abs(5.84*1e-6 * l_arm * (np.sin(np.pi / 4) + np.sin(3 * np.pi / 4)) * n_max)
        # torque_max_y = abs(-5.84*1e-6 * l_arm * (np.cos(3 * np.pi / 4) + np.cos(5 * np.pi / 4)) * n_max)
        # torque_max_z = 2.0 * 0.06 * 5.84*1e-6 * n_max

        # torque_max_x = kf * l * 1.41421356 * n_max
        # torque_max_y = kf * l * 1.41421356 * n_max
        # torque_max_z = 2.0 * kf * ktau * n_max
        # thrust_max = 4 * kf * n_max

        thrust_max = 4 * 5.84*1e-6 * n_max
        torque_max_x = 2.7981516
        torque_max_y = 2.7981516
        torque_max_z = 0.84796799999

        '''Thrust Calculation'''
        v_sp = vd + Pr*(xd - x)
        a_sp = ad + Pv * (v_sp - v) - Dv * (xi_a_pos + omega_c * v) + Iv * xi_v_pos
        bz = se.DenseMatrix([-a_sp[0], -a_sp[1], 9.81])
        bz = bz/norm(bz)
        C_T = (a_sp[2]*ht/9.81 - ht)/bz[2]
        t_sp = C_T * bz
        Thrust = -norm(t_sp)

        '''Rotation desired'''
        b3d = pb.new_sym_matrix("b3d", 3)
        b3d_eval = -t_sp /norm(t_sp)
        xc = se.DenseMatrix(3, 1, [-se.sin(yawd), se.cos(yawd), 0.0])
        b1 = Fct.hat_map(xc) * b3d
        b1 = b1/norm(b1)
        Rd = se.DenseMatrix([b1.T, (Fct.hat_map(b3d) * b1).T, b3d.T]).transpose()
        qd = Fct.rotm2quat(Rd)


        '''Quaternion error'''
        q_err = pb.new_sym_matrix("q_err", 4)
        qd_sub = se.DenseMatrix(3, 1, [qd[1], qd[2], qd[3]])
        Q_sub = se.DenseMatrix(3, 1, [Q[1], Q[2], Q[3]])
        a = qd[0] * Q[0] + (qd_sub.T * Q_sub)[0, 0]
        b = Q[0] * qd_sub - qd[0] * Q_sub - Fct.hat_map(qd_sub) * Q_sub
        q_err_eval = se.DenseMatrix([a, b])


        helpers[b3d] = b3d_eval
        helpers[q_err] = q_err_eval
        '''Torque Calculation'''
        K = 1
        om_sp = 2 * Kq * se.tanh(10000 * q_err[0]) * se.DenseMatrix([q_err[1], q_err[2], q_err[3]])
        Torque = K*P*(om_sp - Omega) - K*D*(xi_a_att + omega_c_att*Omega) + K*I*xi_om_att

        g_xi = se.DenseMatrix([v_sp - v,
                               -omega_c*xi_a_pos - (omega_c**2)*v,
                                om_sp - Omega,
                               -omega_c_att * xi_a_att - (omega_c_att ** 2) * Omega,
                                ])

        '''Control inputs with T^-1 for FLU2NED'''
        tmpp = NED2FLU * Torque
        U1 = -Thrust * thrust_max
        U2 = tmpp[0] * torque_max_x
        U3 = tmpp[1] * torque_max_y
        U4 = tmpp[2] * torque_max_z

        # U, Troll, Tpitch, Tyaw = U1, U2, U3, U4
        ######################################## End of px4 controller #############################################################
        helpers[u] = h = S_C.inv() * se.DenseMatrix([U1, U2, U3, U4])
        tmp = S*u
        U, Troll, Tpitch, Tyaw = tmp[0], tmp[1], tmp[2], tmp[3]

        # # a = S_C.inv()*S
        # v = se.DenseMatrix([U, Troll, Tpitch, Tyaw])
        # v = v.subs(*subs["q"])
        # v = v.subs(*subs["p_aux"]).subs(*subs["p"])

        Q = se.DenseMatrix([q[6, :], q[7:10, :]])
        Q_sub = NED2FLU * Q_sub

        I_mat = se.diag(Jx, Jy, Jz)
        w = se.DenseMatrix([wx, wy, wz])
        R_bw = Fct.quat_to_mat(Q)
        Tor = se.DenseMatrix([Troll, Tpitch, Tyaw])
        dot_p = se.DenseMatrix([vx, vy, vz])
        dot_v = se.DenseMatrix([0, 0, -g]) + (1/m)*R_bw*se.DenseMatrix([0, 0, U])
        dot_q = se.DenseMatrix([-0.5 * Q_sub.T * w, 0.5 * (Q[0] * se.eye(3) - Fct.hat_map(Q_sub)) * w])
        dot_w = I_mat.inv()*(-Fct.hat_map(w) * I_mat * w + Tor)

        f = se.DenseMatrix([dot_p, dot_v, dot_q, dot_w])
        f = f.subs(*subs["q"])
        f = f.subs(*subs["p_aux"]).subs(*subs["p"])

        # f = se.DenseMatrix([
        #     [vx],
        #     [vy],
        #     [vz],
        #     [(qw * qy + qx * qz) * 2 * U / m],
        #     [(-qw * qx + qy * qz) * 2 * U / m],
        #     [(qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2) * U / m - g],
        #     [-0.5 * wx * qx - 0.5 * wy * qy - 0.5 * qz * wz],
        #     [0.5 * wx * qw - 0.5 * wy * qz + 0.5 * qy * wz],
        #     [0.5 * wx * qz + 0.5 * wy * qw - 0.5 * qx * wz],
        #     [-0.5 * wx * qy + 0.5 * wy * qx + 0.5 * qw * wz],
        #     [((Jy-Jz)/Jx)*wy*wz+Troll/Jx],
        #     [((Jz-Jx)/Jy)*wx*wz+Tpitch/Jy],
        #     [((Jx-Jy)/Jz)*wx*wy+Tyaw/Jz]])
        #
        #
        # f = f.subs(*subs["q"])
        # f = f.subs(*subs["p_aux"]).subs(*subs["p"])

        u_int = pb.add_states("u_int", N_inputs)
        # v_int = pb.add_states("v_int", 4)
        system.update({
            q: f,
            xi: g_xi,
            u_int: u,
            # v_int: v,
        })

        PI = self._problem.add_states("PI", N_states, N_par)
        PI_xi = self._problem.add_states("PI_xi", N_ctrl_states, N_par)
        TH_i = self._problem.add_states("TH_i", N_inputs, N_par)  # integral of TH
        TH = self._problem.new_sym_matrix("TH", (N_inputs, N_par))

        df_dq = self._problem.new_sym_matrix("df_dq", (N_states, N_states))
        df_du = self._problem.new_sym_matrix("df_du", (N_states, N_inputs))
        df_dp = self._problem.new_sym_matrix("df_dp", (N_states, N_par))
        dh_dq = self._problem.new_sym_matrix("dh_dq", N_inputs, N_states)
        dh_dxi = self._problem.new_sym_matrix("dh_dxi", N_inputs, N_ctrl_states)
        dg_dq = self._problem.new_sym_matrix("dg_dq", N_ctrl_states, N_states)
        dg_dxi = self._problem.new_sym_matrix("dg_dxi", N_ctrl_states, N_ctrl_states)



        helpers.update({
            df_dq: f.jacobian(q),
            df_du: f.jacobian(u),
            df_dp: f.jacobian(p),
            dh_dq: h.jacobian(q) + h.jacobian(q_err) * (q_err_eval.jacobian(q) + q_err_eval.jacobian(b3d) * b3d_eval.jacobian(q)),
            dh_dxi: h.jacobian(xi) + h.jacobian(q_err) * (q_err_eval.jacobian(xi) + q_err_eval.jacobian(b3d) * b3d_eval.jacobian(xi)),
            dg_dq: g_xi.jacobian(q),
            dg_dxi: g_xi.jacobian(xi),
            TH: dh_dq * PI + dh_dxi * PI_xi,
        })

        system.update({
            PI: df_dq * PI + df_du * TH + df_dp,
            PI_xi: dg_dq * PI + dg_dxi * PI_xi,
            TH_i: TH,
        })

        self._problem.register_helpers(helpers)
        self._problem.register_system(system)
        ODE = self._problem.init_ODE(verbose=verbose, module_location=module_name, overwrite=overwrite)

        def set_check_time_params(ODE, time_vector, **_):
            ODE["ti"] = ODE.t
            ODE["tf"] = time_vector[-1]

        ODE.register_pre_integration_callback(set_check_time_params)

        # default parameters

        ODE["p"] = np.array([5.84*1e-6, 0.06, 0, 0, 1.535])
        ODE["p_c"] = np.array([5.84*1e-6, 0.06, 0, 0, 1.535])  # [kf, ktau, gx, gy, m]
        ODE.param_alias.update({
            "kf": ("p", 0),
            "ktau": ("p", 1),
            "gx": ("p", 2),
            "gy": ("p", 3),
            "m": ("p", 4),
            "kf_c": ("p_c", 0),
            "ktau_c": ("p_c", 1),
            "gx_c": ("p_c", 2),
            "gy_c": ("p_c", 3),
            "m_c": ("p_c", 4),
        })

        ODE["p_aux"] = np.array([0.029125, 0.029125, 0.055225, 0.28, 9.81]) # [Jx, Jy, Jz, l, g]
        ODE.param_alias.update({
            "Jx": ("p_aux", 1),
            "Jy": ("p_aux", 2),
            "Jz": ("p_aux", 3),
            "J": ("p_aux", np.array([1, 2, 3])),
            "l": ("p_aux", 4),
            "g": ("p_aux", 5),
        })

        tPv =  2 * np.array([0.95, 0.95, 1])
        tPr =  2 * np.array([1.8, 1.8, 4])
        # tDv = delta_PX4_pos / delta * np.array([0.2, 0.2, 0.0001])
        tDv = np.array([0.2, 0.2, 0.0001])
        # tIv = delta / delta_PX4_pos * np.array([0.4, 0.4, 2])
        tIv = np.array([0.4, 0.4, 2])
        tKq = np.array([6.5, 6.5, 2.8])
        tP = np.array([0.15, 0.15, 0.2])
        # tD = delta_PX4_att / delta * np.array([0.003, 0.003, 0.0001])
        tD = np.array([0.003, 0.003, 0.0001])
        # tI = delta / delta_PX4_att * np.array([0.2, 0.2, 0.1])
        tI = np.array([0.2, 0.2, 0.1])

        ODE["Array_gains"] = np.concatenate([tPv, tPr, tDv, tIv, tKq, tP, tD, tI])

        ODE["ti"] = 0
        ODE["tf"] = 5
        self.ODE = ODE
        return ODE

    def set_default_state(self, init_waypoint):
        ODE = self.ODE

        p_0 = init_waypoint[0, :-1]
        v_0 = init_waypoint[1, :-1]
        q_0 = np.array([1, 0, 0, 0])
        Omega = np.zeros(3)

        ODE.set_default_initial_state({
            ODE.states["q"]: np.concatenate((p_0, v_0, q_0, Omega))
        })


if __name__ == '__main__':
    model = Jetson(6)
    t = time.time()
    model.generate(3, verbose=True, overwrite=True)
    t2 = time.time()
    print(t2-t)