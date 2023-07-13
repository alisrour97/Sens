import os
import sys
import warnings
from abc import ABC
from os import path

from jitcode import y, t, jitcode
import numpy as np
import symengine as se
from collections import OrderedDict
from functools import wraps

from jitcxde_common.modules_33 import add_suffix

import sympy as sympy

def flatten_dict(system):
    out = dict()
    for key, val in system.items():
        if isinstance(key, type(y(0))) or isinstance(key, se.Symbol):
            out[key] = val
        else:
            if isinstance(val, se.DenseMatrix):
                for deriv, eq in zip(key[:], val[:]):
                    out[deriv] = eq
            elif isinstance(val, np.ndarray):
                for deriv, eq in zip(key[:], val.flatten()):
                    out[deriv] = eq
    return out


def flatten_list(src_list):
    out = []
    for element in src_list:
        try:
            iterable = list(element)
        except TypeError:  # not an iterable
            out.append(element)
        else:  # iterable
            out.extend(flatten_list(iterable))
    return out


class JitParam(jitcode, ABC):

    @wraps(jitcode.__init__)
    def __init__(self,
                 ODEproblem, *,
                 wants_jacobian=False,
                 n=None,
                 callback_functions=(),
                 verbose=True,
                 module_location=None,
                 overwrite=False):
        self._pre_integration_callbacks = []
        self.is_copy = False
        self.verbose = verbose
        self.id = None
        self._set_integrator_call = None
        self.last_result = None
        self.time_points = None
        self.ODEproblem = ODEproblem
        self.param_alias = ODEproblem.param_alias
        f_sym = ODEproblem.ODE_system
        helpers = ODEproblem.helpers
        self.states = ODEproblem.states.copy()
        self.states_indices = ODEproblem.states_indices.copy()
        self._param_dict = ODEproblem.parameters.copy()
        for key, param in self._param_dict.items():
            self._param_dict[key] = np.zeros(np.shape(param))
        control_pars = flatten_list(ODEproblem.parameters.values())
        self.default_init_state = OrderedDict()
        for key in f_sym.keys():
            if isinstance(key, se.DenseMatrix):
                self.default_init_state[key] = np.zeros(key.shape)
            elif isinstance(key, type(y(0))):
                self.default_init_state[key] = 0


        # check if jitced.so module is already compiled
        destination = add_suffix(module_location)
        if path.isfile(destination) and overwrite:
            module_location = None
        if not path.isfile(destination):
            module_location = None

        super().__init__(flatten_dict(f_sym),
                         helpers=helpers,
                         wants_jacobian=wants_jacobian,
                         n=n,
                         control_pars=control_pars,
                         callback_functions=callback_functions,
                         verbose=verbose,
                         module_location=module_location)

        if self.compile_attempt is not True:
            self.generate_f_C(simplify=False)
            #self.compile_C(extra_compile_args=["-v", "-O4", "-time", "-ftime-report", "-fopenmp", "-lpthread"], verbose=True)
            self.save_compiled(destination, overwrite=overwrite)

        self._modulename = self.f.__module__

    def _compile_and_load(self, verbose, extra_compile_args, extra_link_args=None, omp=False):

        path = f"{self._tmpfile()}/{self._modulename}"
        ARGS = [
            "-c",
            f"{path}.c",
            "-o",
            f"{path}.o"
        ]

        from sysconfig import get_paths
        DEFAULT_COMPILE_ARGS = [
            "-std=c11",
            "-Os",
            "-ffast-math",
            "-funsafe-math-optimizations",
            "-g0",
            "-fopenmp",
            "-march=native",
            "-mtune=native",
            "-Wno-unknown-pragmas",
            "-fPIC",
            "-fwrapv",
            "-fstack-protector-strong",
            "-D_FORTIFY_SOURCE=2",
            f"-I{np.get_include()}",
            f"-I{get_paths()['include']}"
        ]


        VERBOSE_ARGS = [
            "-v",
            "-ftime-report",
        ] if verbose or self.verbose else []

        import subprocess
        subprocess.run(["clang"] + DEFAULT_COMPILE_ARGS + VERBOSE_ARGS + ARGS,
                       stdout=sys.stdout, stderr=sys.stderr)
        subprocess.run(["clang", "-shared", "-fopenmp", "-o", f"{path}.so", f"{path}.o"] + VERBOSE_ARGS,
                       stdout=sys.stdout, stderr=sys.stderr)

        from jitcxde_common.modules import find_and_load_module
        self.jitced = find_and_load_module(self._modulename, self._tmpfile())
        self.compile_attempt = True

    def __setitem__(self, key, value):
        if key in self._param_dict:
            self._param_dict[key] = value
        elif key in self.param_alias:
            key, index = self.param_alias[key]
            self._param_dict[key][index] = value
        else:
            KeyError(f"Key '{key}' is not a valid parameter for the model")

    def __getitem__(self, item):
        if item in self._param_dict:
            return self._param_dict[item]
        elif item in self.param_alias:
            key, index = self.param_alias[item]
            return self._param_dict[key][index]
        else:
            KeyError(f"Key '{item}' is not a valid parameter for the model")

    def apply_parameters(self):
        # print(flatten_list(self._param_dict.values()))
        self.set_parameters(flatten_list(self._param_dict.values()))

    def set_default_initial_state(self, initial_sate_dict={}, ti=0.0):
        for key, val in initial_sate_dict.items():
            try:
                if len(key) == len(val):
                    self.default_init_state[key] = np.array(val).reshape(self.default_init_state[key].shape)
                else:
                    raise RuntimeWarning(f"Length do not agree")
            except KeyError:
                raise RuntimeWarning(f"Invalid state. State {key} not present in the system")
            except TypeError:  # no length
                self.default_init_state[key] = val

    def set_initial_value(self, initial_value={}, time=0.0):
        init_val_dict = self.default_init_state.copy()
        for key, val in initial_value.items():
            try:
                if len(key) == len(val):
                    init_val_dict[key] = np.array(val).reshape(self.default_init_state[key].shape)
                else:
                    raise RuntimeError("Length do not agree")
            except KeyError:
                raise RuntimeError(f"Invalid state. State {key} not present in the system")
            except TypeError:  # no length
                init_val_dict[key] = val

        # pprint(flatten_dict(init_val_dict))
        super().set_initial_value(flatten_dict(init_val_dict), time)
        self.time_points = np.ones(1) * self.t
        self.last_result = np.zeros((1, self.n))
        self.last_result[0, :] = self.y

    def set_integrator(self, name, nsteps=10**6, interpolate=True, **integrator_params):
        self._set_integrator_call = (
            name,
            {
                "nsteps": nsteps,
                "interpolate": interpolate,
            }
        )
        self._set_integrator_call[1].update(integrator_params)

        return super().set_integrator(name, nsteps=nsteps, interpolate=interpolate, **integrator_params)

    def integrate(self, *args, **kwargs):
        warnings.warn("You might want to use integrate_on_time_vector instead", DeprecationWarning)
        self.apply_parameters()
        return super().integrate(*args, **kwargs)

    def register_pre_integration_callback(self, func):
        """
        Register a function to be executed before integrate_on_time_vector (and auto-application of parameters)
        Each function will be passed the JitParam object as first argument and time_vector as second.
        Every other argument must be keyword only
        Functions are executed in order
        :param func: func(ODE:JitParam, time_vector:1darray, **kwargs) -> None
        :return:
        """
        if func not in self._pre_integration_callbacks:
            self._pre_integration_callbacks.append(func)

    def integrate_on_time_vector(self, time_vector, kwargs=None):
        if kwargs is None:
            kwargs = dict()
        for func in self._pre_integration_callbacks:
            func(self, time_vector, **kwargs)
        self.apply_parameters()

        if time_vector[0] == self.t:
            states = np.zeros((len(time_vector), self.n))
            states[0, :] = self.y
            self.time_points = np.concatenate((self.time_points, time_vector[1:]))
            for i, t in enumerate(time_vector[1:], start=1):
                self.apply_parameters()
                # print(states[i-1, :])
                states[i, :] = super().integrate(t)
        else:
            states = np.zeros((len(time_vector)+1, self.n))
            states[0, :] = self.y
            self.time_points = np.concatenate((self.time_points, time_vector))
            for i, t in enumerate(time_vector, start=1):
                states[i, :] = super().integrate(t)
        self.last_result = np.concatenate((self.last_result, states[1:, :]), axis=0)
        return states

    def __getstate__(self):
        orig_path = os.getcwd() + self._modulename
        if not os.path.isfile(orig_path):
            self.save_compiled(orig_path)
        #id = secrets.token_urlsafe(10)

        # copy of states for pickling
        states = dict()
        for symbol, state in self.states.items():
            shape = np.shape(state)
            if len(shape) == 0:
                states[symbol] = int(state.args[0])
            else:
                states[symbol] = np.zeros(shape, dtype=int)
                for index in np.ndindex(shape):
                    states[symbol][index] = state[index].args[0]

        # copy of default_init_state for pickling
        default_init_state = []
        for state, values in self.default_init_state.items():
            shape = np.shape(state)
            if len(shape) == 0:
                default_init_state.append((int(state.args[0]), values))
            else:
                tmp = np.zeros(shape, dtype=int)
                for index in np.ndindex(shape):
                    tmp[index] = state[index].args[0]
                default_init_state.append((tmp, values))

        save_dict = {
            "states": states,
            "states_indices": self.states_indices,
            "_param_dict": self._param_dict,
            "default_init_state": default_init_state,
            "n": self.n,
            "save_path": orig_path,
            "is_copy": True,
            "last_result": self.last_result,
        }
        return save_dict, self._set_integrator_call

    def __setstate__(self, import_state):
        import_state, set_integrator_call = import_state
        super().__init__(n=import_state["n"], module_location=import_state["save_path"])
        self.__dict__.update(import_state)
        # unpack states
        for symbol, state_numbers in import_state["states"].items():
            shape = np.shape(state_numbers)
            if len(shape) == 0:
                self.states[symbol] = y(state_numbers)
            else:
                self.states[symbol] = se.zeros(*shape)
                for index in np.ndindex(shape):
                    self.states[symbol][index] = y(state_numbers[index])

        # unpack default_params
        self.default_init_state = dict()
        for state_numbers, values in import_state["default_init_state"]:
            shape = np.shape(state_numbers)
            if len(shape) == 0:
                self.default_init_state[y(state_numbers)] = values
            else:
                tmp = se.zeros(*shape)
                for index in np.ndindex(shape):
                    tmp[index] = y(state_numbers[index])
                self.default_init_state[tmp] = values
        if set_integrator_call is not None:
            self.set_integrator(set_integrator_call[0], **set_integrator_call[1])
        self.set_initial_value()


class ODEproblem(ABC):

    def __init__(self):
        self.parameters = OrderedDict()
        self.ODE_system = None
        self.helpers = None
        self.states = dict()
        self.states_indices = dict()
        self.output = None
        self.used_states = 0
        self.param_alias = dict()



    def add_states(self, symbol, *dimensions):
        tmp_us = self.used_states
        if dimensions == (1,):
            out = y(self.used_states)
            self.used_states += 1
        elif len(dimensions) == 1:
            n_states = np.prod(dimensions)
            out = se.DenseMatrix([
                y(i) for i in range(self.used_states, self.used_states + n_states)
            ])
            self.used_states += n_states
        elif len(dimensions) == 2:
            n_states = np.prod(dimensions)
            out = se.DenseMatrix([
                y(i) for i in range(self.used_states, self.used_states + n_states)
            ]).reshape(*dimensions)
            self.used_states += n_states
        else:
            raise RuntimeError("Takes only 2 dimensions")
        self.states[symbol] = out
        self.states_indices[symbol] = np.array(range(tmp_us, self.used_states))
        return out

    def new_sym_matrix(self, symbol, *dims, **kwargs):
        if type(dims[0]) == tuple:
            dims = dims[0]
        if dims == (1, ):
            return se.Symbol(symbol, **kwargs)
        elif len(dims) == 1:
            tmp = np.array([sympy.Symbol(symbol+'_{}'.format(i)) for i in range(dims[0])])
            return se.DenseMatrix([se.sympify(str(pi)) for pi in tmp])
        elif len(dims) == 2:
            return se.DenseMatrix(*dims, se.symarray(symbol, dims, **kwargs).flatten())

    def new_parameter(self, symbol, *dims, **kwargs):
        out = self.new_sym_matrix(symbol, *dims, **kwargs)
        self.parameters[symbol] = out
        return out

    def register_system(self, system_dict):
        if np.any([np.shape(key) != np.shape(val) for key, val in system_dict.items()]):
            raise ValueError("Shapes must agree for each key, value pair")
        self.ODE_system = system_dict.copy()

    def register_helpers(self, helpers_dict):
        if np.any([np.shape(key) != np.shape(val) for key, val in helpers_dict.items()]):
            raise ValueError("Shapes must agree for each key, value pair")
        self.helpers = list(flatten_dict(helpers_dict).items())

    def register_output(self, output_dict):
        raise NotImplementedError
        self.output = output_dict
        se.lambdify()

    def init_ODE(self, *,
                 wants_jacobian=False,
                 n=None,
                 callback_functions=(),
                 verbose=True,
                 module_location=None,
                 overwrite=False):
        return JitParam(self,
                        wants_jacobian=wants_jacobian,
                        n=n,
                        callback_functions=callback_functions,
                        verbose=verbose,
                        module_location=module_location,
                        overwrite=overwrite)

    def __copy__(self):
        new_obj = ODEproblem.__new__(ODEproblem)
        for key, val in self.__dict__.items():
            try:
                new_obj.__dict__[key] = val.copy()
            except AttributeError:
                new_obj.__dict__[key] = val
        return new_obj

    def copy(self):
        return self.__copy__()
