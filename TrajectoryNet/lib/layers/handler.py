import torch
import torch.nn as nn

from torchdiffeq import odeint as odeint

from lib.layers.wrappers.regularization import RegularizedVanillaODEfunc

import inspect

__all__ = ["ODEHandler"]

class ODEHandler(nn.Module):
    def __init__(self, odefunc, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(ODEHandler, self).__init__()

        nreg = 0
        if len(regularization_fns) > 0:
            odefunc = RegularizedVanillaODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, integration_times, reverse=False):

        if reverse:
            integration_times = _flip(integration_times, 0)
        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        if self.regularization_states is not None:
            reg_states = tuple(torch.full((z.size(0), ), torch.mean(reg).item()).to(z) for reg in self.get_regularization_states())
        else:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for _ in range(self.nreg))
        if self.training:
            state_t = odeint(
                self.odefunc,
                (z,) + reg_states,
                integration_times.to(z),
                atol=[self.atol] + [1e20] * len(reg_states) if self.solver == 'dopri5' else self.atol,
                rtol=[self.rtol] + [1e20] * len(reg_states) if self.solver == 'dopri5' else self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z,),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t = state_t[:1]
        
        if self.training:
            self.regularization_states = state_t[1:]

        return z_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]