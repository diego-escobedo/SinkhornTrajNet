import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ODEnet", "AutoencoderDiffEqNet", "VanillaODEfunc", "ODEfunc", "AutoencoderODEfunc", "DiffeqNet"]


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
#     print(dx.shape, dx.requires_grad, y.shape, y.requires_grad)
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


# def divergence_bf(f, y, **unused_kwargs):
#     jac = _get_minibatch_jacobian(f, y)
#     diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]
#     return torch.sum(diagonal, 1)


def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
        y: (N, ...) with a total of D_y elements in ...
        x: (N, ...) with a total of D_x elements in ...
    Returns:
        The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                        create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}

class VanillaODEfunc(nn.Module):

    def __init__(self, diffeq, rademacher=False):
        super(VanillaODEfunc, self).__init__()

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.rademacher = rademacher

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        assert type(states) == tuple, "pass in tupled states to ODEfunc"

        z = states[0]
        # increment num evals
        self._num_evals += 1

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(z)
            else:
                self._e = sample_gaussian_like(z)

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[1:]:
                s_.requires_grad_(True)
            dz = self.diffeq(t, z)

        return tuple([dz,] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[1:]])


class DiffeqNet(nn.Module):
    """
    Actual NN used to approximate the derivative of the hidden state
    """
    def __init__(self, vector_field_dims,
                        ff_hidden_dims,
                        *,
                        #tim2vec params
                        time2vec=True,
                        time2vec_dims=64,
                        time2vec_activation=torch.sin, #default, not in args
                        #sape params
                        sape=True,
                        sape_incremental_mask=True,
                        sape_n_freq=128,
                        sape_sigma=2, #default, not in args
                        sape_use_time=False,
                        #feedforward network params
                        nonlinearity="tanh", 
                        nonl_final=False):
        super(DiffeqNet, self).__init__()

        self.vector_field_dims = vector_field_dims

        self.ff_network_d_in = 0

        #handle time2vec
        if time2vec:
            self.t2v = Time2Vec(time2vec_dims, t2v_activation=time2vec_activation)
            self.ff_network_d_in += self.t2v.d_out
        
        #handle sape
        if sape:
            self.sape_use_time = sape_use_time
            imap_d_in = vector_field_dims
            if sape_use_time: imap_d_in += 1 #add time as a dimension
            self.feature_mapping = InputMapping(d_in=imap_d_in, 
                                                n_freq=sape_n_freq, 
                                                incrementalMask=sape_incremental_mask, 
                                                sigma=sape_sigma)
            self.ff_network_d_in += self.feature_mapping.d_out

        #handle ffn
        #if no sape or time2vec just feed everything in directly
        input_size =  self.ff_network_d_in if self.ff_network_d_in > 0 else vector_field_dims+1 
        stack = [nn.Linear(input_size, ff_hidden_dims[0])]
        for output_size in ff_hidden_dims[1:]:
            input_size = stack[-1].out_features
            stack.append(nn.Linear(input_size, output_size))
            stack.append(NONLINEARITIES[nonlinearity])
        stack.append(nn.Linear(stack[-1].out_features, vector_field_dims))
        if nonl_final:
            stack.append(NONLINEARITIES[nonlinearity])
        self.ffn = nn.Sequential(*stack)

    def forward(self, t, state):
        #vectorize time
        state, t = self._handle_dims(t, state)

        #check inptus to ffn
        ffn_input = []
        if getattr(self, 't2v', None):
            ffn_input.append(self.t2v(t))
        if getattr(self, 'feature_mapping', None):
            feat_map_input = torch.cat((state, t), dim=-1).to(torch.float32) if self.sape_use_time else state
            ffn_input.append(self.feature_mapping(feat_map_input))
        
        if len(ffn_input) == 0:
            ffn_input = torch.cat((state, t), dim=-1).to(torch.float32)
        elif len(ffn_input) == 1:
            ffn_input = ffn_input[0]
        else:
            ffn_input = torch.cat(ffn_input, dim=-1).to(torch.float32)

        ffn_output = self.ffn(ffn_input)

        return ffn_output

    def _handle_dims(self, t, state):
        #we want state to be num_samples x num_dims
        state = state.reshape(-1, self.vector_field_dims)    
        
        #project time onto same dimension as vector
        t = t.reshape(-1, 1)
        if torch.numel(t) == 1: #case in which we get passed a single time for all data points
            t = t.repeat(state.shape[0], 1)
        else:  #case in which we have as many times as we have data points -- ACTUALLY CHECK 
            assert torch.numel(t) == state.shape[0]
        
        return state, t.to(torch.float32)

class Time2Vec(nn.Module):
    "Make a time2vec component"
    def __init__(self, projection_dims, t2v_activation=torch.sin):
        self.time_proj_unactivated = nn.Linear(1, 1) # will take from 1x1 -> 1x1
        if projection_dims > 0:
            self.time_proj_activated = nn.Linear(1, projection_dims)
        else:
            self.time_proj_activated = None
        
        self.t2v_activation = t2v_activation

        self.d_out = 1 + projection_dims
    
    def forward(self, t):
        #run the multiplicatiosn needed
        time_projection = self.time_proj_unactivated(t) # num_samples x 1... this is the "unactivated" projection
        if self.time_proj_activated:
            tp_activated = self.t2v_activation(self.time_proj_activated(t)) # num_samples x time_projection_dims
            time_projection = torch.cat((time_projection, tp_activated), dim=-1) # num_samples x time_projection_dims+1
        
        return time_projection


class InputMapping(nn.Module):
    """Fourier features mapping"""

    def __init__(self, d_in, n_freq, sigma=2, tdiv=2, incrementalMask=True, Tperiod=None):
        super().__init__()
        Bmat = torch.randn(d_in, n_freq) * np.pi* sigma/np.sqrt(d_in)  # gaussian
        
        # time frequencies are a quarter of spacial frequencies.
        # Bmat[:, d_in-1] /= tdiv
        #Bmat[:, 0] /= tdiv

        self.Tperiod = Tperiod
        if Tperiod is not None:
            # Tcycles = (Bmat[:, d_in-1]*Tperiod/(2*np.pi)).round()
            # K = Tcycles*(2*np.pi)/Tperiod
            # Bmat[:, d_in-1] = K
            Tcycles = (Bmat[:, 0]*Tperiod/(2*np.pi)).round()
            K = Tcycles*(2*np.pi)/Tperiod
            Bmat[:, 0] = K
        
        Bnorms = torch.norm(Bmat, p=2, dim=1)
        sortedBnorms, sortIndices = torch.sort(Bnorms)
        Bmat = Bmat[sortIndices, :]

        self.d_in = d_in
        self.n_freq = n_freq
        self.d_out = n_freq * 2 + d_in if Tperiod is None else n_freq * 2 + d_in - 1
        self.register_buffer('B', Bmat)
        self.register_buffer('mask', torch.zeros(1, n_freq))
        #self.B = nn.Linear(d_in, self.d_out, bias=False)
        # with torch.no_grad():
        #     self.B.weight = nn.Parameter(Bmat, requires_grad=False)
        #     self.mask = nn.Parameter(torch.zeros(
        #         1, n_freq), requires_grad=False)

        self.incrementalMask = incrementalMask
        if not incrementalMask:
            self.mask = nn.Parameter(torch.ones(
                1, n_freq), requires_grad=False)

    def step(self, progressPercent):
        if self.incrementalMask:
            float_filled = (progressPercent*self.n_freq)/.7
            int_filled = int(float_filled // 1)
            remainder = float_filled % 1

            if int_filled >= self.n_freq:
                self.mask[0, :] = 1
            else:
                self.mask[0, 0:int_filled] = 1
                # self.mask[0, int_filled] = remainder

    def forward(self, xi):
        # pdb.set_trace()
        dim = xi.shape[1]-1
        y =  torch.matmul(xi, self.B) 
        if self.Tperiod is None:
            return torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi], dim=-1)
        else:
            return torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi[:,1:dim+1]], dim=-1)