from abc import ABCMeta, abstractmethod
from typing import Optional
from enum import Enum
import numpy as np

from gen.lib.sym_gen import JitParam

class Mode(str, Enum):
    SIMU = "simu"
    GRAD = "grad"
    NOGRAD = "nograd"

class Model(metaclass=ABCMeta):
    N_ctrl_points: int
    ODE: Optional[JitParam]

    def __init__(self, N_ctrl_points):
        super().__init__()
        self.N_ctrl_points = N_ctrl_points
        self._problem = None
        self.ODE = None

    @abstractmethod
    def generate(self, N_lc, verbose=False, mode="nograd", token="", overwrite=False) -> (JitParam) :
        pass

    @abstractmethod
    def set_default_state(self, init_waypoint):
        pass

    @abstractmethod
    def output_indices(self):
        pass

    @abstractmethod
    def nonlcon(self, grad, states_vec, time_vec):
        pass

    @abstractmethod
    def nonlcon_tol(self):
        pass

    def eq_constraints(self, grad, states_vec, time_vec):
        return []

    def eq_constraints_tol(self, target_point):
        return []

    def integrate_along_trajectory(self, trajectory, N: int=100):
        if self.ODE is None:
            raise RuntimeError("model not generated")
        self.ODE.set_initial_value()
        dt_total = trajectory.waypoints_t[-1] - trajectory.waypoints_t[0]
        for ctrl_points, ti, tf in trajectory.traj_iter():
            time_vector = np.linspace(ti, tf, round(N * (tf - ti) / dt_total))
            self.ODE["a"] = ctrl_points
            self.ODE.integrate_on_time_vector(time_vector)


def tdiff(function, first_deriv, second_deriv):
    tmp_deriv = function.jacobian(first_deriv)
    df_dq_dq = np.moveaxis(np.array([tmp_deriv.diff(var) for var in second_deriv]), 0, -1)
    return df_dq_dq

def mat_diff(mat, deriv):
    return np.moveaxis(np.array([mat.diff(var) for var in deriv]), 0, -1)